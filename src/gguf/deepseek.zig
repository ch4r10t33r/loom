//! deepseek2-family inference over GGUF (issue #1) — the architecture Kimi K2,
//! DeepSeek V2/V3, and GLM-class MoE models ship as.
//!
//! MLA attention: q through a LoRA bottleneck (q_a -> norm -> q_b), k/v through
//! a compressed latent (kv_a_mqa -> norm = c_kv, cached) plus one shared
//! rope-carrying k head; per-head k_nope/v are re-expanded from the cached
//! latent via kv_b every step. Rope on the decoupled slice is **NORM-style**
//! (adjacent pairs): DeepSeek's reference code views the rope dims as
//! (d/2, 2) and transposes before rotate_half, which makes its net effect an
//! adjacent-pair rotation on the stored layout — validated empirically on
//! real V2-Lite weights (NEOX produces degenerate output, NORM is coherent).
//!
//! MoE FFN: router (ffn_gate_inp) scored by sigmoid or softmax
//! (expert_gating_func), optional selection-only bias (exp_probs_b, the
//! noaux_tc trick), top-k experts with optionally renormalized gates scaled by
//! expert_weights_scale, plus always-on shared expert(s). Leading
//! `leading_dense_block_count` layers use a plain dense SwiGLU FFN.
//!
//! Weights stay in GGML formats in the read-only memory map (fused kernels,
//! ggml.zig). Validated against real DeepSeek-V2-Lite Q4_K_M weights.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const tensor = @import("../core/tensor.zig");
const llama = @import("llama.zig");
const bpe = @import("bpe.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");

/// Real DeepSeek/Kimi GGUFs ship a gpt2-style BPE tokenizer; fixtures (and
/// some conversions) use SPM. Selected by `tokenizer.ggml.model`.
pub const Tok = union(enum) {
    spm: llama.Tokenizer,
    bpe: bpe.Bpe,

    pub fn encode(self: *const Tok, gpa: std.mem.Allocator, text: []const u8, add_bos: bool, parse_special: bool) ![]u32 {
        return switch (self.*) {
            .spm => |*t| t.encode(gpa, text, add_bos, parse_special),
            .bpe => |*t| t.encode(gpa, text, add_bos, parse_special),
        };
    }
    pub fn decode(self: *const Tok, w: *Io.Writer, id: u32) !void {
        return switch (self.*) {
            .spm => |*t| t.decode(w, id),
            .bpe => |*t| t.decode(w, id),
        };
    }
    pub fn eosId(self: *const Tok) u32 {
        return switch (self.*) {
            .spm => |*t| t.eos,
            .bpe => |*t| t.eos,
        };
    }
    pub fn deinit(self: *Tok, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .spm => |*t| t.deinit(gpa),
            .bpe => |*t| t.deinit(gpa),
        }
    }
};

pub const GatingFunc = enum(u32) { softmax = 1, sigmoid = 2 };

/// Caps for config values that index fixed-size stack buffers in `step`
/// (security issue #29). `n_used` slices `sel_buf`/`ids_buf` and `dim` slices
/// `acc_buf`; without these an oversized metadata value smashes the stack.
pub const MAX_SELECTED: usize = 64;
pub const MAX_DIM: usize = 8192;

pub const Config = struct {
    dim: usize,
    n_layers: usize,
    n_dense_layers: usize,
    n_heads: usize,
    q_lora_rank: usize, // 0 = direct q projection (V2-Lite)
    kv_lora_rank: usize,
    rope_dim: usize,
    nope_dim: usize, // key_length - rope_dim
    v_head_dim: usize,
    ffn: usize,
    moe_ffn: usize,
    n_expert: usize,
    n_used: usize,
    n_shared: usize,
    gating: GatingFunc,
    weights_norm: bool,
    weights_scale: f32,
    vocab: usize,
    ctx_len: usize,
    rope_base: f32,
    eps: f32,
    // YaRN context extension (deepseek2.rope.scaling.*). factor == 0 -> off.
    yarn_factor: f32,
    yarn_orig_ctx: f32,
    yarn_log_mul: f32,
    // attention softmax scale, mscale^2 / sqrt(key_dim) under YaRN
    attn_scale: f32,

    pub fn keyDim(self: Config) usize {
        return self.nope_dim + self.rope_dim;
    }
};

pub const Tensor = struct {
    ty: ggml.Type,
    data: []const u8,
    ne0: usize, // row length (input dim)
    ne1: usize, // rows (output dim)
    ne2: usize, // experts (3D tensors), else 1

    /// Byte slice of expert `e` in a 3D tensor. `e` comes from the router,
    /// bounded by the config's expert count, while `ne2` comes from the file:
    /// a file declaring fewer experts than the config would slice past the
    /// mapping (security issue #29), so bound it here.
    pub fn expert(self: Tensor, e: usize) !Tensor {
        if (e >= self.ne2) return error.ExpertOutOfRange;
        const per = self.ne1 * ggml.rowBytes(self.ty, self.ne0);
        const start = e * per;
        if (start + per > self.data.len) return error.ExpertOutOfRange;
        return .{ .ty = self.ty, .data = self.data[start..][0..per], .ne0 = self.ne0, .ne1 = self.ne1, .ne2 = 1 };
    }
};

const LayerT = struct {
    attn_norm: Tensor,
    // q path: either direct or LoRA
    attn_q: ?Tensor,
    attn_q_a: ?Tensor,
    attn_q_a_norm: ?Tensor,
    attn_q_b: ?Tensor,
    // kv path
    attn_kv_a_mqa: Tensor,
    attn_kv_a_norm: Tensor,
    attn_kv_b: Tensor,
    attn_output: Tensor,
    ffn_norm: Tensor,
    // dense layers
    ffn_gate: ?Tensor,
    ffn_up: ?Tensor,
    ffn_down: ?Tensor,
    // moe layers
    ffn_gate_inp: ?Tensor,
    exp_probs_b: ?Tensor,
    ffn_gate_exps: ?Tensor,
    ffn_up_exps: ?Tensor,
    ffn_down_exps: ?Tensor,
    ffn_gate_shexp: ?Tensor,
    ffn_up_shexp: ?Tensor,
    ffn_down_shexp: ?Tensor,
    is_moe: bool,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    io: Io,
    parsed: gguf.Parsed,
    file: Io.File,
    mm: Io.File.MemoryMap,
    cfg: Config,
    token_embd: Tensor,
    output_norm: Tensor,
    output: Tensor,
    layers: []LayerT,
    tok: Tok,
    /// Distributed expert source (issue #3): when set, routed-expert weights
    /// come through Source.get() — local tier or peer fetch — instead of the
    /// (possibly sparse) memory map.
    dist: ?*expert_fetch.Source = null,
    /// (layer * n_expert + e) -> manifest shard id; built by attachDist.
    expert_shard: []usize = &.{},

    pub fn encodePrompt(self: *const Model, gpa: std.mem.Allocator, text: []const u8, parse_special: bool) ![]u32 {
        return self.tok.encode(gpa, text, true, parse_special);
    }
    pub fn decodeToken(self: *const Model, w: *Io.Writer, id: u32) !void {
        return self.tok.decode(w, id);
    }
    pub fn eosToken(self: *const Model) u32 {
        return self.tok.eosId();
    }
    /// The GGUF `tokenizer.chat_template` string, if the model carries one.
    pub fn chatTemplate(self: *const Model) ?[]const u8 {
        return self.parsed.getString("tokenizer.chat_template");
    }

    pub fn deinit(self: *Model) void {
        if (self.expert_shard.len > 0) self.gpa.free(self.expert_shard);
        self.tok.deinit(self.gpa);
        self.gpa.free(self.layers);
        self.mm.destroy(self.io);
        self.file.close(self.io);
        self.parsed.deinit();
    }

    /// Resolve a tensor by name, validating every size against the file with
    /// overflow-checked arithmetic (security issue #29). Unchecked, a malicious
    /// GGUF could wrap `ne0*4` or `start+bytes` to a small value, pass the
    /// bounds test, and have kernels walk far past the mapping.
    fn resolve(self: *const Model, name: []const u8) !Tensor {
        const t = self.parsed.findTensor(name) orelse return error.MissingTensor;
        if (!ggml.Type.supported(t.ggml_type)) return error.UnsupportedTensorType;
        const ty: ggml.Type = @enumFromInt(t.ggml_type);
        if (t.dims.len == 0) return error.BadTensorShape;
        const ne0: usize = std.math.cast(usize, t.dims[0]) orelse return error.BadTensorShape;
        const ne1: usize = if (t.dims.len > 1) std.math.cast(usize, t.dims[1]) orelse return error.BadTensorShape else 1;
        const ne2: usize = if (t.dims.len > 2) std.math.cast(usize, t.dims[2]) orelse return error.BadTensorShape else 1;
        const bytes = ggml.tensorBytesChecked(ty, ne0, ne1, ne2) catch return error.BadTensorShape;
        const off = std.math.add(u64, self.parsed.data_offset, t.offset) catch return error.TruncatedGguf;
        const start = std.math.cast(usize, off) orelse return error.TruncatedGguf;
        const end = std.math.add(usize, start, bytes) catch return error.TruncatedGguf;
        if (end > self.mm.memory.len) return error.TruncatedGguf;
        return .{ .ty = ty, .data = self.mm.memory[start..end], .ne0 = ne0, .ne1 = ne1, .ne2 = ne2 };
    }

    /// Assert a resolved tensor has exactly the shape the config implies.
    /// Without this the kernels trust the file's dims and the config
    /// independently: e.g. a `[dim, 1<<20]` q_a weight against an 8-float
    /// destination is a heap overflow write (security issue #29).
    /// Assert a tensor really is F32. `asF32` reinterprets the mapped bytes as
    /// floats behind a debug assert, which is a no-op in ReleaseFast: a norm
    /// weight declared as q8_0 would otherwise be reinterpreted (security
    /// issue #29). Alignment is checked too, since the data offset is
    /// file-controlled and `general.alignment` may legally be 1 or 2.
    fn expectF32(t: Tensor, name: []const u8) !void {
        if (t.ty != .f32) {
            std.debug.print("gguf: tensor {s} must be f32, got {s}\n", .{ name, @tagName(t.ty) });
            return error.BadTensorType;
        }
        if (@intFromPtr(t.data.ptr) % @alignOf(f32) != 0) return error.BadTensorAlignment;
        if (t.data.len % @sizeOf(f32) != 0) return error.BadTensorType;
    }

    fn expectShape(t: Tensor, ne0: usize, ne1: usize, name: []const u8) !void {
        if (t.ne0 != ne0 or t.ne1 != ne1) {
            std.debug.print("gguf: tensor {s} has shape [{d},{d}], config implies [{d},{d}]\n", .{ name, t.ne0, t.ne1, ne0, ne1 });
            return error.BadTensorShape;
        }
    }

    fn resolveOpt(self: *const Model, name: []const u8) ?Tensor {
        return self.resolve(name) catch null;
    }
};

/// Reject metadata that would divide by zero, overflow a size computation, or
/// index past a fixed-size buffer (security issue #29). Everything here comes
/// from an untrusted file: a peer-supplied GGUF or a Hugging Face download.
fn validateConfig(cfg: Config) !void {
    if (cfg.dim == 0 or cfg.n_heads == 0 or cfg.n_layers == 0) return error.BadConfig;
    if (cfg.dim > MAX_DIM) return error.BadConfig; // acc_buf in step()
    if (cfg.n_used > MAX_SELECTED) return error.BadConfig; // sel_buf/ids_buf
    if (cfg.n_expert != 0 and cfg.n_used > cfg.n_expert) return error.BadConfig;
    if (cfg.n_dense_layers > cfg.n_layers) return error.BadConfig;
    if (cfg.kv_lora_rank == 0 or cfg.v_head_dim == 0) return error.BadConfig;
    if (cfg.rope_dim == 0 or cfg.nope_dim == 0) return error.BadConfig;
    if (cfg.ctx_len == 0) return error.BadConfig;
    // KV cache is n_layers * ctx_len * kv_lora_rank floats; keep the product
    // representable so State.init cannot wrap to a small allocation.
    const kv = std.math.mul(usize, cfg.n_layers, cfg.ctx_len) catch return error.BadConfig;
    _ = std.math.mul(usize, kv, cfg.kv_lora_rank) catch return error.BadConfig;
    const kr = std.math.mul(usize, cfg.n_layers, cfg.ctx_len) catch return error.BadConfig;
    _ = std.math.mul(usize, kr, cfg.rope_dim) catch return error.BadConfig;
}

pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Model {
    var parsed = try gguf.parse(gpa, io, path);
    errdefer parsed.deinit();

    const arch = parsed.getString("general.architecture") orelse return error.NoArchitecture;
    if (!std.mem.eql(u8, arch, "deepseek2")) return error.UnsupportedArchitecture;

    const file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    var mm = try file.createMemoryMap(io, .{
        .len = @intCast(parsed.file_size),
        .protection = .{ .read = true, .write = false },
    });
    errdefer mm.destroy(io);

    const dim: usize = @intCast(parsed.getUint("deepseek2.embedding_length") orelse return error.BadConfig);
    const n_heads: usize = @intCast(parsed.getUint("deepseek2.attention.head_count") orelse return error.BadConfig);
    const rope_dim: usize = @intCast(parsed.getUint("deepseek2.rope.dimension_count") orelse return error.BadConfig);
    const key_len: usize = @intCast(parsed.getUint("deepseek2.attention.key_length") orelse return error.BadConfig);
    if (key_len <= rope_dim) return error.BadConfig;

    const cfg = Config{
        .dim = dim,
        .n_layers = @intCast(parsed.getUint("deepseek2.block_count") orelse return error.BadConfig),
        .n_dense_layers = @intCast(parsed.getUint("deepseek2.leading_dense_block_count") orelse 0),
        .n_heads = n_heads,
        .q_lora_rank = @intCast(parsed.getUint("deepseek2.attention.q_lora_rank") orelse 0),
        .kv_lora_rank = @intCast(parsed.getUint("deepseek2.attention.kv_lora_rank") orelse return error.BadConfig),
        .rope_dim = rope_dim,
        .nope_dim = key_len - rope_dim,
        .v_head_dim = @intCast(parsed.getUint("deepseek2.attention.value_length") orelse return error.BadConfig),
        .ffn = @intCast(parsed.getUint("deepseek2.feed_forward_length") orelse return error.BadConfig),
        .moe_ffn = @intCast(parsed.getUint("deepseek2.expert_feed_forward_length") orelse 0),
        .n_expert = @intCast(parsed.getUint("deepseek2.expert_count") orelse 0),
        .n_used = @intCast(parsed.getUint("deepseek2.expert_used_count") orelse 0),
        .n_shared = @intCast(parsed.getUint("deepseek2.expert_shared_count") orelse 0),
        .gating = switch (parsed.getUint("deepseek2.expert_gating_func") orelse 1) {
            2 => .sigmoid,
            else => .softmax,
        },
        .weights_norm = parsed.getBool("deepseek2.expert_weights_norm") orelse false,
        .weights_scale = @floatCast(parsed.getFloat("deepseek2.expert_weights_scale") orelse 1.0),
        .vocab = 0, // from token_embd below
        .ctx_len = @intCast(parsed.getUint("deepseek2.context_length") orelse 2048),
        .rope_base = @floatCast(parsed.getFloat("deepseek2.rope.freq_base") orelse 10000.0),
        .eps = @floatCast(parsed.getFloat("deepseek2.attention.layer_norm_rms_epsilon") orelse 1e-6),
        .yarn_factor = 0,
        .yarn_orig_ctx = 0,
        .yarn_log_mul = 0,
        .attn_scale = 0, // both fixed up below
    };
    if (cfg.n_layers > cfg.n_dense_layers and (cfg.n_expert == 0 or cfg.n_used == 0 or cfg.moe_ffn == 0))
        return error.BadConfig;
    if (cfg.n_expert > 512) return error.BadConfig;
    try validateConfig(cfg);

    // YaRN: DeepSeek applies the mscale to the attention softmax scale (the
    // in-rope mscale is cancelled by attn_factor = 1/(1 + 0.1 ln(1/freq_scale)),
    // as in llama.cpp's deepseek2 graph): kq_scale = mscale^2 / sqrt(key_dim),
    // mscale = 1 + yarn_log_mul * ln(factor).
    var cfg2 = cfg;
    const scaling_type = parsed.getString("deepseek2.rope.scaling.type") orelse "";
    if (std.mem.eql(u8, scaling_type, "yarn")) {
        cfg2.yarn_factor = @floatCast(parsed.getFloat("deepseek2.rope.scaling.factor") orelse 1.0);
        cfg2.yarn_orig_ctx = @floatCast(parsed.getFloat("deepseek2.rope.scaling.original_context_length") orelse 4096.0);
        cfg2.yarn_log_mul = @floatCast(parsed.getFloat("deepseek2.rope.scaling.yarn_log_multiplier") orelse 0.1);
    }
    const mscale: f32 = if (cfg2.yarn_factor > 1.0)
        1.0 + cfg2.yarn_log_mul * @log(cfg2.yarn_factor)
    else
        1.0;
    cfg2.attn_scale = mscale * mscale / @sqrt(@as(f32, @floatFromInt(cfg2.keyDim())));

    var model = Model{
        .gpa = gpa,
        .io = io,
        .parsed = parsed,
        .file = file,
        .mm = mm,
        .cfg = cfg2,
        .token_embd = undefined,
        .output_norm = undefined,
        .output = undefined,
        .layers = &.{},
        .tok = undefined,
    };
    model.token_embd = try model.resolve("token_embd.weight");
    model.output_norm = try model.resolve("output_norm.weight");
    try Model.expectF32(model.output_norm, "output_norm.weight");
    model.output = model.resolveOpt("output.weight") orelse model.token_embd;
    model.cfg.vocab = model.token_embd.ne1;

    const layers = try gpa.alloc(LayerT, cfg.n_layers);
    errdefer gpa.free(layers);
    var nb: [128]u8 = undefined;
    for (layers, 0..) |*l, i| {
        const N = struct {
            fn f(buf: []u8, li: usize, comptime s: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "blk.{d}." ++ s, .{li}) catch unreachable;
            }
        };
        l.is_moe = i >= cfg.n_dense_layers;
        l.attn_norm = try model.resolve(N.f(&nb, i, "attn_norm.weight"));
        try Model.expectF32(l.attn_norm, "attn_norm.weight");
        if (cfg.q_lora_rank > 0) {
            l.attn_q = null;
            l.attn_q_a = try model.resolve(N.f(&nb, i, "attn_q_a.weight"));
            // st.q_a is q_lora_rank floats: a larger ne1 here is a heap
            // overflow write in mv() (security issue #29)
            try Model.expectShape(l.attn_q_a.?, cfg.dim, cfg.q_lora_rank, "attn_q_a");
            l.attn_q_a_norm = try model.resolve(N.f(&nb, i, "attn_q_a_norm.weight"));
            try Model.expectF32(l.attn_q_a_norm.?, "attn_q_a_norm.weight");
            l.attn_q_b = try model.resolve(N.f(&nb, i, "attn_q_b.weight"));
            try Model.expectShape(l.attn_q_b.?, cfg.q_lora_rank, cfg.n_heads * cfg.keyDim(), "attn_q_b");
        } else {
            l.attn_q = try model.resolve(N.f(&nb, i, "attn_q.weight"));
            try Model.expectShape(l.attn_q.?, cfg.dim, cfg.n_heads * cfg.keyDim(), "attn_q");
            l.attn_q_a = null;
            l.attn_q_a_norm = null;
            l.attn_q_b = null;
        }
        l.attn_kv_a_mqa = try model.resolve(N.f(&nb, i, "attn_kv_a_mqa.weight"));
        try Model.expectShape(l.attn_kv_a_mqa, cfg.dim, cfg.kv_lora_rank + cfg.rope_dim, "attn_kv_a_mqa");
        l.attn_kv_a_norm = try model.resolve(N.f(&nb, i, "attn_kv_a_norm.weight"));
        try Model.expectF32(l.attn_kv_a_norm, "attn_kv_a_norm.weight");
        l.attn_kv_b = try model.resolve(N.f(&nb, i, "attn_kv_b.weight"));
        try Model.expectShape(l.attn_kv_b, cfg.kv_lora_rank, cfg.n_heads * (cfg.nope_dim + cfg.v_head_dim), "attn_kv_b");
        l.attn_output = try model.resolve(N.f(&nb, i, "attn_output.weight"));
        try Model.expectShape(l.attn_output, cfg.n_heads * cfg.v_head_dim, cfg.dim, "attn_output");
        l.ffn_norm = try model.resolve(N.f(&nb, i, "ffn_norm.weight"));
        try Model.expectF32(l.ffn_norm, "ffn_norm.weight");
        if (!l.is_moe) {
            l.ffn_gate = try model.resolve(N.f(&nb, i, "ffn_gate.weight"));
            l.ffn_up = try model.resolve(N.f(&nb, i, "ffn_up.weight"));
            l.ffn_down = try model.resolve(N.f(&nb, i, "ffn_down.weight"));
            l.ffn_gate_inp = null;
            l.exp_probs_b = null;
            l.ffn_gate_exps = null;
            l.ffn_up_exps = null;
            l.ffn_down_exps = null;
            l.ffn_gate_shexp = null;
            l.ffn_up_shexp = null;
            l.ffn_down_shexp = null;
        } else {
            l.ffn_gate = null;
            l.ffn_up = null;
            l.ffn_down = null;
            l.ffn_gate_inp = try model.resolve(N.f(&nb, i, "ffn_gate_inp.weight"));
            l.exp_probs_b = model.resolveOpt(N.f(&nb, i, "exp_probs_b.bias"));
            if (l.exp_probs_b) |b| try Model.expectF32(b, "exp_probs_b.bias");
            l.ffn_gate_exps = try model.resolve(N.f(&nb, i, "ffn_gate_exps.weight"));
            l.ffn_up_exps = try model.resolve(N.f(&nb, i, "ffn_up_exps.weight"));
            l.ffn_down_exps = try model.resolve(N.f(&nb, i, "ffn_down_exps.weight"));
            if (cfg.n_shared > 0) {
                l.ffn_gate_shexp = try model.resolve(N.f(&nb, i, "ffn_gate_shexp.weight"));
                l.ffn_up_shexp = try model.resolve(N.f(&nb, i, "ffn_up_shexp.weight"));
                l.ffn_down_shexp = try model.resolve(N.f(&nb, i, "ffn_down_shexp.weight"));
            } else {
                l.ffn_gate_shexp = null;
                l.ffn_up_shexp = null;
                l.ffn_down_shexp = null;
            }
        }
    }
    model.layers = layers;

    const tok_model = model.parsed.getString("tokenizer.ggml.model") orelse "llama";
    model.tok = if (std.mem.eql(u8, tok_model, "gpt2"))
        .{ .bpe = try bpe.Bpe.init(gpa, &model.parsed) }
    else
        .{ .spm = try llama.Tokenizer.init(gpa, &model.parsed) };
    return model;
}

/// Attach a distributed expert source. Maps every (layer, expert) to its
/// manifest shard by matching the gate slice's file offset against the shard's
/// first extent — no reliance on shard ordering conventions.
pub fn attachDist(m: *Model, gpa: std.mem.Allocator, src: *expert_fetch.Source) !void {
    const mani = &src.store.manifest;
    if (mani.mode != .expert) return error.NotExpertManifest;

    var by_off = std.AutoHashMap(u64, usize).init(gpa);
    defer by_off.deinit();
    var i: usize = mani.n_resident;
    while (i < mani.nRanges()) : (i += 1) {
        try by_off.put(mani.shardExtents(i)[0].offset, i);
    }

    const map = try gpa.alloc(usize, m.cfg.n_layers * m.cfg.n_expert);
    errdefer gpa.free(map);
    @memset(map, std.math.maxInt(usize));

    var nb: [128]u8 = undefined;
    for (m.layers, 0..) |l, li| {
        if (!l.is_moe) continue;
        const name = try std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_exps.weight", .{li});
        const t = m.parsed.findTensor(name) orelse return error.MissingTensor;
        const ty: ggml.Type = @enumFromInt(t.ggml_type);
        const per: u64 = @intCast(t.dims[1] * ggml.rowBytes(ty, @intCast(t.dims[0])));
        var e: usize = 0;
        while (e < m.cfg.n_expert) : (e += 1) {
            const off = m.parsed.data_offset + t.offset + per * e;
            map[li * m.cfg.n_expert + e] = by_off.get(off) orelse return error.ShardMapMismatch;
        }
    }
    m.expert_shard = map;
    m.dist = src;
}

// ---- state & forward ---------------------------------------------------------

pub const State = struct {
    cfg: Config,
    x: []f32,
    normed: []f32,
    q_a: []f32,
    q: []f32, // n_heads * keyDim
    kv_a: []f32, // kv_lora + rope
    k_nope: []f32,
    v_t: []f32,
    head_out: []f32, // n_heads * v_head_dim
    proj_out: []f32,
    gate: []f32, // max(ffn, moe_ffn * max(1, n_shared))
    up: []f32,
    act: []f32,
    ffn_out: []f32,
    router: []f32, // n_expert
    scores: []f32, // ctx
    logits: []f32,
    c_kv_cache: []f32, // [layers][ctx][kv_lora]
    k_rope_cache: []f32, // [layers][ctx][rope_dim]

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !State {
        const ffn_max = @max(cfg.ffn, cfg.moe_ffn * @max(@as(usize, 1), cfg.n_shared));
        return .{
            .cfg = cfg,
            .x = try gpa.alloc(f32, cfg.dim),
            .normed = try gpa.alloc(f32, cfg.dim),
            .q_a = try gpa.alloc(f32, @max(cfg.q_lora_rank, 1)),
            .q = try gpa.alloc(f32, cfg.n_heads * cfg.keyDim()),
            .kv_a = try gpa.alloc(f32, cfg.kv_lora_rank + cfg.rope_dim),
            .k_nope = try gpa.alloc(f32, cfg.nope_dim),
            .v_t = try gpa.alloc(f32, cfg.v_head_dim),
            .head_out = try gpa.alloc(f32, cfg.n_heads * cfg.v_head_dim),
            .proj_out = try gpa.alloc(f32, cfg.dim),
            .gate = try gpa.alloc(f32, ffn_max),
            .up = try gpa.alloc(f32, ffn_max),
            .act = try gpa.alloc(f32, ffn_max),
            .ffn_out = try gpa.alloc(f32, cfg.dim),
            .router = try gpa.alloc(f32, @max(cfg.n_expert, 1)),
            .scores = try gpa.alloc(f32, cfg.ctx_len),
            .logits = try gpa.alloc(f32, cfg.vocab),
            .c_kv_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * cfg.kv_lora_rank),
            .k_rope_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * cfg.rope_dim),
        };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        inline for (.{
            self.x,      self.normed,     self.q_a,          self.q,        self.kv_a,
            self.k_nope, self.v_t,        self.head_out,     self.proj_out, self.gate,
            self.up,     self.act,        self.ffn_out,      self.router,   self.scores,
            self.logits, self.c_kv_cache, self.k_rope_cache,
        }) |sl| gpa.free(sl);
    }
};

fn mv(t: Tensor, out: []f32, x: []const f32) void {
    ggml.matvec(t.ty, out, t.data, x, t.ne1, t.ne0);
}

/// NORM adjacent-pair rope, with YaRN frequency correction when enabled: each
/// pair's angle blends the interpolated (theta/factor) and extrapolated
/// frequency by a ramp over the corrected dimension range (beta_fast=32,
/// beta_slow=1), as in ggml's rope_yarn. The in-rope mscale is omitted —
/// deepseek2 cancels it and applies mscale to the attention scale instead.
fn ropeApply(cfg: Config, vec: []f32, pos: usize) void {
    if (cfg.yarn_factor <= 1.0) {
        ropeNormPlain(vec, pos, cfg.rope_base);
        return;
    }
    const d: f32 = @floatFromInt(vec.len);
    const two_pi = 2.0 * std.math.pi;
    const corr = struct {
        fn dim(nd: f32, orig: f32, beta: f32, base: f32) f32 {
            return nd * @log(orig / (beta * two_pi)) / (2.0 * @log(base));
        }
    };
    const low = @max(0.0, @floor(corr.dim(d, cfg.yarn_orig_ctx, 32.0, cfg.rope_base)));
    const high = @min(d - 1.0, @ceil(corr.dim(d, cfg.yarn_orig_ctx, 1.0, cfg.rope_base)));

    var i: usize = 0;
    while (2 * i + 1 < vec.len) : (i += 1) {
        const exponent = @as(f32, @floatFromInt(2 * i)) / d;
        const freq = std.math.pow(f32, cfg.rope_base, -exponent);
        const theta_extrap = @as(f32, @floatFromInt(pos)) * freq;
        const theta_interp = theta_extrap / cfg.yarn_factor;
        const y = (@as(f32, @floatFromInt(i)) - low) / @max(0.001, high - low);
        const ramp_mix = 1.0 - std.math.clamp(y, 0.0, 1.0); // 1 = extrapolate
        const theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        const c = @cos(theta);
        const sn = @sin(theta);
        const a = vec[2 * i];
        const b = vec[2 * i + 1];
        vec[2 * i] = a * c - b * sn;
        vec[2 * i + 1] = a * sn + b * c;
    }
}

/// NORM-style rope: adjacent pairs (2i, 2i+1), freq by pair index.
fn ropeNormPlain(vec: []f32, pos: usize, base: f32) void {
    const d: f32 = @floatFromInt(vec.len);
    var i: usize = 0;
    while (2 * i + 1 < vec.len) : (i += 1) {
        const exponent = @as(f32, @floatFromInt(2 * i)) / d;
        const freq = std.math.pow(f32, base, -exponent);
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const c = @cos(angle);
        const sn = @sin(angle);
        const a = vec[2 * i];
        const b = vec[2 * i + 1];
        vec[2 * i] = a * c - b * sn;
        vec[2 * i + 1] = a * sn + b * c;
    }
}

fn asF32(t: Tensor) []const f32 {
    std.debug.assert(t.ty == .f32);
    return @alignCast(std.mem.bytesAsSlice(f32, t.data));
}

/// Dense SwiGLU FFN into st.ffn_out.
fn denseFFN(st: *State, gate_w: Tensor, up_w: Tensor, down_w: Tensor) void {
    const f = gate_w.ne1;
    mv(gate_w, st.gate[0..f], st.normed);
    mv(up_w, st.up[0..f], st.normed);
    tensor.swiglu(st.act[0..f], st.gate[0..f], st.up[0..f]);
    mv(down_w, st.ffn_out, st.act[0..f]);
}

const Selected = struct { expert: usize, gate: f32 };

/// deepseek2 routing: gating func over router logits; selection may add a bias
/// (noaux_tc) that does NOT affect the returned gate weights; top-k gates are
/// optionally renormalized, then scaled.
fn route(cfg: Config, router_logits: []const f32, bias: ?[]const f32, sel: []Selected) void {
    var scores_buf: [512]f32 = undefined;
    const scores = scores_buf[0..cfg.n_expert];
    switch (cfg.gating) {
        .sigmoid => for (router_logits, 0..) |l, i| {
            scores[i] = tensor.sigmoid(l);
        },
        .softmax => {
            @memcpy(scores, router_logits);
            tensor.softmax(scores);
        },
    }
    var choice_buf: [512]f32 = undefined;
    const choice = choice_buf[0..cfg.n_expert];
    @memcpy(choice, scores);
    if (bias) |b| for (choice, b) |*c, bv| {
        c.* += bv;
    };

    var used = [_]bool{false} ** 512;
    for (sel) |*s| {
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (choice, 0..) |c, e| {
            if (!used[e] and c > best_v) {
                best_v = c;
                best = e;
            }
        }
        used[best] = true;
        s.* = .{ .expert = best, .gate = scores[best] };
    }
    if (cfg.weights_norm) {
        var sum: f32 = 0;
        for (sel) |s| sum += s.gate;
        if (sum > 0) for (sel) |*s| {
            s.gate /= sum;
        };
    }
    for (sel) |*s| s.gate *= cfg.weights_scale;
}

/// One token step; logits land in st.logits. Errors only when a distributed
/// expert shard has no reachable holder (fail loud, not silently degraded).
pub fn step(m: *const Model, st: *State, token: u32, pos: usize) !void {
    const cfg = m.cfg;
    const kd = cfg.keyDim();
    const nope = cfg.nope_dim;
    const rope = cfg.rope_dim;
    const kvr = cfg.kv_lora_rank;
    const vd = cfg.v_head_dim;
    const scale = cfg.attn_scale;

    // A token id indexes token_embd rows directly. Ids come from the
    // tokenizer, whose ceiling is an independent metadata array, so bound it
    // here rather than trusting the two to agree (security issue #29).
    if (token >= cfg.vocab) return error.TokenOutOfRange;
    ggml.dequantRow(m.token_embd.ty, st.x, m.token_embd.data, token, cfg.dim);

    for (m.layers, 0..) |l, li| {
        // ---- MLA attention ----
        tensor.rmsnorm(st.normed, st.x, asF32(l.attn_norm), cfg.eps);

        if (cfg.q_lora_rank > 0) {
            mv(l.attn_q_a.?, st.q_a, st.normed);
            tensor.rmsnorm(st.q_a, st.q_a, asF32(l.attn_q_a_norm.?), cfg.eps);
            mv(l.attn_q_b.?, st.q, st.q_a);
        } else {
            mv(l.attn_q.?, st.q, st.normed);
        }

        mv(l.attn_kv_a_mqa, st.kv_a, st.normed);
        const c_kv = st.c_kv_cache[(li * cfg.ctx_len + pos) * kvr ..][0..kvr];
        tensor.rmsnorm(c_kv, st.kv_a[0..kvr], asF32(l.attn_kv_a_norm), cfg.eps);
        const k_rope = st.k_rope_cache[(li * cfg.ctx_len + pos) * rope ..][0..rope];
        @memcpy(k_rope, st.kv_a[kvr .. kvr + rope]);
        ropeApply(cfg, k_rope, pos);

        var h: usize = 0;
        while (h < cfg.n_heads) : (h += 1) {
            ropeApply(cfg, st.q[h * kd + nope ..][0..rope], pos);
        }

        const seq = pos + 1;
        h = 0;
        while (h < cfg.n_heads) : (h += 1) {
            const q_nope = st.q[h * kd ..][0..nope];
            const q_rope = st.q[h * kd + nope ..][0..rope];
            // kv_b rows for this head: [nope rows of k][vd rows of v], each over kvr
            const kb_base = h * (nope + vd);
            const k_rows = l.attn_kv_b.data[kb_base * ggml.rowBytes(l.attn_kv_b.ty, kvr) ..];
            const v_rows = l.attn_kv_b.data[(kb_base + nope) * ggml.rowBytes(l.attn_kv_b.ty, kvr) ..];

            var t_i: usize = 0;
            while (t_i < seq) : (t_i += 1) {
                const c_t = st.c_kv_cache[(li * cfg.ctx_len + t_i) * kvr ..][0..kvr];
                ggml.matvec(l.attn_kv_b.ty, st.k_nope, k_rows, c_t, nope, kvr);
                const kr_t = st.k_rope_cache[(li * cfg.ctx_len + t_i) * rope ..][0..rope];
                var dot: f32 = 0;
                for (q_nope, st.k_nope) |a, b| dot += a * b;
                for (q_rope, kr_t) |a, b| dot += a * b;
                st.scores[t_i] = dot * scale;
            }
            tensor.softmax(st.scores[0..seq]);

            const oh = st.head_out[h * vd ..][0..vd];
            @memset(oh, 0);
            t_i = 0;
            while (t_i < seq) : (t_i += 1) {
                const c_t = st.c_kv_cache[(li * cfg.ctx_len + t_i) * kvr ..][0..kvr];
                ggml.matvec(l.attn_kv_b.ty, st.v_t, v_rows, c_t, vd, kvr);
                const w = st.scores[t_i];
                for (oh, st.v_t) |*o, vv| o.* += w * vv;
            }
        }
        mv(l.attn_output, st.proj_out, st.head_out);
        tensor.add(st.x, st.proj_out);

        // ---- FFN ----
        tensor.rmsnorm(st.normed, st.x, asF32(l.ffn_norm), cfg.eps);
        if (!l.is_moe) {
            denseFFN(st, l.ffn_gate.?, l.ffn_up.?, l.ffn_down.?);
            tensor.add(st.x, st.ffn_out);
        } else {
            mv(l.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
            var sel_buf: [64]Selected = undefined;
            const sel = sel_buf[0..cfg.n_used];
            const bias: ?[]const f32 = if (l.exp_probs_b) |b| asF32(b) else null;
            route(cfg, st.router[0..cfg.n_expert], bias, sel);

            var acc_buf: [8192]f32 = undefined;
            const acc = acc_buf[0..cfg.dim];
            @memset(acc, 0);
            if (m.dist) |src| {
                // warm the missing shards in parallel: per-layer miss latency
                // becomes max(fetch), not sum(fetch)
                var ids_buf: [64]usize = undefined;
                for (sel, 0..) |s, k| ids_buf[k] = m.expert_shard[li * cfg.n_expert + s.expert];
                src.prefetch(ids_buf[0..sel.len]);
            }
            for (sel) |s| {
                if (m.dist) |src| {
                    const blk = try src.get(m.expert_shard[li * cfg.n_expert + s.expert]);
                    const gt = l.ffn_gate_exps.?;
                    const ut = l.ffn_up_exps.?;
                    const dt = l.ffn_down_exps.?;
                    const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
                    const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
                    const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
                    denseFFN(
                        st,
                        .{ .ty = gt.ty, .data = blk[0..gl], .ne0 = gt.ne0, .ne1 = gt.ne1, .ne2 = 1 },
                        .{ .ty = ut.ty, .data = blk[gl..][0..ul], .ne0 = ut.ne0, .ne1 = ut.ne1, .ne2 = 1 },
                        .{ .ty = dt.ty, .data = blk[gl + ul ..][0..dl], .ne0 = dt.ne0, .ne1 = dt.ne1, .ne2 = 1 },
                    );
                } else {
                    denseFFN(
                        st,
                        try l.ffn_gate_exps.?.expert(s.expert),
                        try l.ffn_up_exps.?.expert(s.expert),
                        try l.ffn_down_exps.?.expert(s.expert),
                    );
                }
                for (acc, st.ffn_out) |*a, v| a.* += s.gate * v;
            }
            if (l.ffn_gate_shexp) |gs| {
                denseFFN(st, gs, l.ffn_up_shexp.?, l.ffn_down_shexp.?);
                tensor.add(acc, st.ffn_out);
            }
            tensor.add(st.x, acc);
        }
    }

    tensor.rmsnorm(st.normed, st.x, asF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
}

test "route: sigmoid gating with selection bias picks by biased score, gates from raw" {
    const cfg = Config{
        .dim = 8,
        .n_layers = 1,
        .n_dense_layers = 0,
        .n_heads = 1,
        .q_lora_rank = 0,
        .kv_lora_rank = 4,
        .rope_dim = 2,
        .nope_dim = 2,
        .v_head_dim = 2,
        .ffn = 8,
        .moe_ffn = 8,
        .n_expert = 4,
        .n_used = 2,
        .n_shared = 0,
        .gating = .sigmoid,
        .weights_norm = true,
        .weights_scale = 1.0,
        .vocab = 8,
        .ctx_len = 8,
        .rope_base = 10000,
        .eps = 1e-6,
        .yarn_factor = 0,
        .yarn_orig_ctx = 0,
        .yarn_log_mul = 0,
        .attn_scale = 1.0,
    };
    // raw logits favor experts 0,1; bias flips selection to 2,3
    const logits = [_]f32{ 2.0, 1.5, 0.0, -0.5 };
    const bias = [_]f32{ 0, 0, 10, 10 };
    var sel: [2]Selected = undefined;
    route(cfg, &logits, &bias, &sel);
    for (sel) |s| try std.testing.expect(s.expert == 2 or s.expert == 3);
    // gates renormalized over raw sigmoid scores of the selected pair
    var sum: f32 = 0;
    for (sel) |s| sum += s.gate;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // without bias, raw scores pick 0,1
    var sel2: [2]Selected = undefined;
    route(cfg, &logits, null, &sel2);
    for (sel2) |s| try std.testing.expect(s.expert == 0 or s.expert == 1);
}
