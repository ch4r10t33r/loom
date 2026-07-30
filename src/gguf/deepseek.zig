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
const backend = @import("../compute/backend.zig");
const tensor = @import("../core/tensor.zig");
const llama = @import("llama.zig");
const tokmod = @import("tok.zig");
const moe = @import("moe.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");

/// The file's tokenizer (SPM or gpt2-BPE); see tok.zig.
pub const Tok = tokmod.Tok;

pub const GatingFunc = moe.GatingFunc;

/// Cap on `dim`, which slices `acc_buf` in step() (security issue #29).
/// The expert-count caps live in moe.zig alongside the router that uses them.
pub const MAX_SELECTED: usize = moe.MAX_SELECTED;
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

    pub fn routeCfg(self: Config) moe.RouteCfg {
        return .{
            .n_expert = self.n_expert,
            .n_used = self.n_used,
            .gating = self.gating,
            .weights_norm = self.weights_norm,
            .weights_scale = self.weights_scale,
        };
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
    /// The GGUF `general.name` string, if present. Used to recognise loom's
    /// own random-weight fixtures, which are stamped "loom <arch> fixture".
    pub fn generalName(self: *const Model) ?[]const u8 {
        return self.parsed.getString("general.name");
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
    if (cfg.n_expert > moe.MAX_EXPERTS) return error.BadConfig; // route() score buffers
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
    // The whole mapping goes to the compute backend as an arena, so the GPU
    // paths read it through a resident buffer instead of faulting file-backed
    // pages per token. Measured on the fused MoE path before this: 0.2 tok/s
    // with every bucket -- host attention included -- 10x slower, which is
    // memory pressure, not arithmetic. Registration only records the region;
    // the buffers are made when a device exists.
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

    model.tok = try Tok.init(gpa, &model.parsed);
    // The whole mapping goes to the compute backend as an arena, so GPU paths
    // read it through a resident buffer instead of faulting file-backed pages
    // per token. Measured on the fused MoE path before this: 0.2 tok/s, every
    // bucket -- host attention included -- 10x slower, which is memory
    // pressure rather than arithmetic. Registration records the region; the
    // buffers are made once a device exists.
    _ = backend.registerArena(model.mm.memory);
    return model;
}

/// Attach a distributed expert source. Maps every (layer, expert) to its
/// manifest shard by matching the gate slice's file offset against the shard's
/// first extent — no reliance on shard ordering conventions.
pub fn attachDist(m: *Model, gpa: std.mem.Allocator, src: *expert_fetch.Source) !void {
    const is_moe = try gpa.alloc(bool, m.cfg.n_layers);
    defer gpa.free(is_moe);
    for (m.layers, 0..) |l, i| is_moe[i] = l.is_moe;
    m.expert_shard = try moe.buildExpertShardMap(
        gpa,
        &m.parsed,
        &src.store.manifest,
        m.cfg.n_layers,
        m.cfg.n_expert,
        is_moe,
    );
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
    q_abs: []f32, // W_k^T q_nope, kv_lora_rank
    c_acc: []f32, // sum_t p_t c_t, kv_lora_rank
    row_tmp: []f32, // one dequantized kv_b row, kv_lora_rank
    // Every head's absorbed query, rope query and compressed output laid out
    // contiguously, which is what the Metal kernel takes: it does all heads in
    // one dispatch, so a per-head buffer would mean a dispatch per head.
    mla_qa: []f32, // n_heads * kv_lora_rank
    mla_qr: []f32, // n_heads * rope_dim
    mla_ol: []f32, // n_heads * kv_lora_rank
    q_nope_all: []f32, // n_heads * nope_dim, gathered for the absorb kernel
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
            .q_abs = try gpa.alloc(f32, cfg.kv_lora_rank),
            .c_acc = try gpa.alloc(f32, cfg.kv_lora_rank),
            .row_tmp = try gpa.alloc(f32, cfg.kv_lora_rank),
            .mla_qa = try gpa.alloc(f32, cfg.n_heads * cfg.kv_lora_rank),
            .mla_qr = try gpa.alloc(f32, cfg.n_heads * cfg.rope_dim),
            .mla_ol = try gpa.alloc(f32, cfg.n_heads * cfg.kv_lora_rank),
            .q_nope_all = try gpa.alloc(f32, cfg.n_heads * cfg.nope_dim),
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
            self.x,      self.normed, self.q_a,        self.q,            self.kv_a,
            self.q_abs,  self.c_acc,  self.row_tmp,    self.head_out,     self.proj_out,
            self.gate,   self.up,     self.act,        self.ffn_out,      self.router,
            self.scores, self.logits, self.c_kv_cache, self.k_rope_cache, self.mla_qa,
            self.mla_qr, self.mla_ol, self.q_nope_all,
        }) |sl| gpa.free(sl);
    }
};

fn mv(t: Tensor, out: []f32, x: []const f32) void {
    backend.matvec(t.ty, out, t.data, x, t.ne1, t.ne0);
}

/// NORM adjacent-pair rope, with YaRN frequency correction when enabled: each
/// pair's angle blends the interpolated (theta/factor) and extrapolated
/// frequency by a ramp over the corrected dimension range (beta_fast=32,
/// beta_slow=1), as in ggml's rope_yarn. The in-rope mscale is omitted —
/// deepseek2 cancels it and applies mscale to the attention scale instead.
/// Test seam: the Metal rope kernel is verified against this exact function.
pub fn ropeApplyForTest(cfg: Config, vec: []f32, pos: usize) void {
    ropeApply(cfg, vec, pos);
}

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
    backend.swiglu(st.act[0..f], st.gate[0..f], st.up[0..f]);
    mv(down_w, st.ffn_out, st.act[0..f]);
}

/// One expert's FFN applied to `n` activation vectors in a single pass over its
/// weights.
///
/// This is the whole point of batching a MoE model. Calling `denseFFN` once per
/// sequence walks the expert's ~5.7 MB again for each one; `matmul` walks it
/// once and produces `n` outputs, so a layer costs its *union* of experts
/// rather than the sum over sequences. With several sequences in flight the
/// unions overlap heavily -- that is the "throughput hides latency" argument
/// the design rests on, and it only holds if the read is shared.
fn expertFfnBatched(
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    xs: []const f32, // n * dim
    n: usize,
    dim: usize,
    scratch_gate: []f32, // n * ffn
    scratch_up: []f32, // n * ffn
    out: []f32, // n * dim
) void {
    const f = gate_w.ne1;
    backend.matmul(gate_w.ty, scratch_gate[0 .. n * f], gate_w.data, xs, n, f, dim);
    backend.matmul(up_w.ty, scratch_up[0 .. n * f], up_w.data, xs, n, f, dim);
    for (0..n) |k| backend.swiglu(scratch_gate[k * f ..][0..f], scratch_gate[k * f ..][0..f], scratch_up[k * f ..][0..f]);
    backend.matmul(down_w.ty, out[0 .. n * dim], down_w.data, scratch_gate[0 .. n * f], n, dim, f);
}

const Selected = moe.Selected;

/// One expert's gate/up/down bytes, from the distributed store when there is
/// one and straight out of the model mapping otherwise.
fn expertParts(m: *const Model, l: anytype, li: usize, expert: usize) !expert_fetch.Source.Parts {
    const gt = l.ffn_gate_exps.?;
    const ut = l.ffn_up_exps.?;
    const dt = l.ffn_down_exps.?;
    if (m.dist) |src| {
        const sid = m.expert_shard[li * m.cfg.n_expert + expert];
        if (src.getMapped(sid)) |p| return p;
        const blk = try src.get(sid);
        const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
        const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
        const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
        return .{ .gate = blk[0..gl], .up = blk[gl..][0..ul], .down = blk[gl + ul ..][0..dl] };
    }
    return .{
        .gate = (try gt.expert(expert)).data,
        .up = (try ut.expert(expert)).data,
        .down = (try dt.expert(expert)).data,
    };
}

/// One token step; logits land in st.logits. Errors only when a distributed
/// expert shard has no reachable holder (fail loud, not silently degraded).
/// Per-phase timing for the decode loop, enabled with LOOM_PROFILE=1.
///
/// Added because two plausible bottleneck theories -- hashing every expert on
/// every access, and O(seq) KV decompression in attention -- each predicted a
/// large win, and fixing both moved 1.07 -> 1.3 tok/s. When a hypothesis
/// survives a measurement that should have refuted it, the measurement is the
/// thing to fix.
pub const Profile = struct {
    var on: ?bool = null;
    var attn_ns: i128 = 0;
    var route_ns: i128 = 0;
    var acc_ns: i128 = 0;
    var head_ns: i128 = 0;
    var expert_get_ns: i128 = 0;
    var expert_ffn_ns: i128 = 0;
    var other_ns: i128 = 0;
    var tokens: u64 = 0;
    /// Filled by the engine so the report can say whether the RAM tier is
    /// actually being hit, rather than leaving that to be inferred from a
    /// timing that did not move.
    pub var src_stats: ?expert_fetch.Stats = null;
    pub var cache_slots: usize = 0;
    /// Command buffers the backend submitted, summed over the profiled tokens.
    pub var cmdbufs: u64 = 0;
    /// Split out of `expert ffn`: the routed experts go to the backend as one
    /// block per layer, the shared expert does not, and the bucket was 5x what
    /// the block measures in isolation.
    var shexp_ns: i128 = 0;
    /// MoE layers that took the backend block, and that fell back to the host
    /// loop. A path that is declining silently has cost this project two long
    /// detours already.
    var moe_block: u64 = 0;
    var moe_host: u64 = 0;

    fn reset() void {
        attn_ns = 0;
        route_ns = 0;
        acc_ns = 0;
        head_ns = 0;
        expert_get_ns = 0;
        expert_ffn_ns = 0;
        other_ns = 0;
        tokens = 0;
        cmdbufs = 0;
        shexp_ns = 0;
        moe_block = 0;
        moe_host = 0;
    }

    pub fn enabled() bool {
        if (on == null) on = std.c.getenv("LOOM_PROFILE") != null;
        return on.?;
    }
    /// 0.16 has no std.time.Timer or nanoTimestamp; the monotonic clock is
    /// reached through Io, and a decode step has no Io. CLOCK_MONOTONIC via
    /// libc is the cheapest thing that works from here, and this is a
    /// diagnostic path anyway.
    fn now() i128 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    }
    /// Straight to stderr: this is a diagnostic, and routing it through the
    /// node's status thread made it depend on plumbing that was itself under
    /// investigation.
    pub fn dump() void {
        const tot = attn_ns + expert_get_ns + expert_ffn_ns + other_ns;
        if (tot == 0 or tokens == 0) return;
        const t: f64 = @floatFromInt(tot);
        const n: f64 = @floatFromInt(tokens);
        std.debug.print(
            "profile {d} tok: attn {d:.1}/{d:.0}%  get {d:.1}/{d:.0}%  ffn {d:.1}/{d:.0}%  other {d:.1}/{d:.0}%  total {d:.1}ms  cb/tok {d:.1}\n",
            .{
                tokens,
                @as(f64, @floatFromInt(attn_ns)) / 1e6 / n,
                100 * @as(f64, @floatFromInt(attn_ns)) / t,
                @as(f64, @floatFromInt(expert_get_ns)) / 1e6 / n,
                100 * @as(f64, @floatFromInt(expert_get_ns)) / t,
                @as(f64, @floatFromInt(expert_ffn_ns)) / 1e6 / n,
                100 * @as(f64, @floatFromInt(expert_ffn_ns)) / t,
                @as(f64, @floatFromInt(other_ns)) / 1e6 / n,
                100 * @as(f64, @floatFromInt(other_ns)) / t,
                t / 1e6 / n,
                @as(f64, @floatFromInt(cmdbufs)) / n,
            },
        );
    }

    /// Report the window since the last call, then reset.
    ///
    /// Cumulative totals hide exactly what a running node needs to show. The
    /// first touch of an expert hashes its whole 6 MB shard to verify it, so a
    /// short profile is mostly that transient and a long one averages it away
    /// -- neither says what the node is doing now. A window does.
    pub fn report(w: anytype) !void {
        const tot = attn_ns + expert_get_ns + expert_ffn_ns + other_ns;
        if (tot == 0 or tokens == 0) return;
        defer reset();
        const pct = struct {
            fn f(a: i128, b: i128) f64 {
                return 100.0 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
            }
        }.f;
        const ms = struct {
            fn f(a: i128, n: u64) f64 {
                return @as(f64, @floatFromInt(a)) / 1e6 / @as(f64, @floatFromInt(n));
            }
        }.f;
        try w.print("profile: last {d} tokens (ms/token)\n", .{tokens});
        try w.print("  attention       {d:8.1}  {d:5.1}%\n", .{ ms(attn_ns, tokens), pct(attn_ns, tot) });
        try w.print("  expert get      {d:8.1}  {d:5.1}%\n", .{ ms(expert_get_ns, tokens), pct(expert_get_ns, tot) });
        try w.print("  expert ffn      {d:8.1}  {d:5.1}%\n", .{ ms(expert_ffn_ns, tokens), pct(expert_ffn_ns, tot) });
        try w.print("  everything else {d:8.1}  {d:5.1}%\n", .{ ms(other_ns, tokens), pct(other_ns, tot) });
        try w.print("  total           {d:8.1}\n", .{ms(tot, tokens)});
        if (cmdbufs > 0) {
            // A command buffer costs ~262 us fixed on an M5 whatever it holds,
            // so this product is a floor on the token that no kernel work can
            // move -- only handing the backend larger units can. Printed next
            // to the total because the comparison is the whole point.
            const per_tok = @as(f64, @floatFromInt(cmdbufs)) / @as(f64, @floatFromInt(tokens));
            try w.print("  gpu submissions {d:8.1} /token  (~{d:.1} ms of the above at 262 us each)\n", .{
                per_tok, per_tok * 0.262,
            });
        }
        // The "everything else" bucket was 17.5% and unattributed, which is
        // exactly the size at which it stops being a rounding error and starts
        // being the next thing to fix. Split into the three candidates.
        try w.print("    of which: router {d:8.1}  expert-accum {d:8.1}  norm+lmhead {d:8.1}  shared-expert {d:8.1}\n", .{
            ms(route_ns, tokens), ms(acc_ns, tokens), ms(head_ns, tokens), ms(shexp_ns, tokens),
        });
        try w.print("  moe layers/token: {d:.1} on the backend block, {d:.1} on the host loop\n", .{
            @as(f64, @floatFromInt(moe_block)) / @as(f64, @floatFromInt(tokens)),
            @as(f64, @floatFromInt(moe_host)) / @as(f64, @floatFromInt(tokens)),
        });
        if (src_stats) |g| {
            const acc = g.mapped + g.ram + g.local + g.fetched;
            if (acc > 0) try w.print(
                "  expert tier: mapped {d} ({d:.0}%)  ram {d}  disk {d}  peer {d}   cache slots {d}\n",
                .{ g.mapped, 100.0 * @as(f64, @floatFromInt(g.mapped)) / @as(f64, @floatFromInt(acc)), g.ram, g.local, g.fetched, cache_slots },
            );
        }
    }
};

/// One layer's MLA attention for one sequence, including the residual add.
///
/// Extracted so the batched decode path cannot drift from the single-sequence
/// one. Attention is the part a batch cannot share -- every sequence has its
/// own KV cache and its own position -- so it stays per sequence either way,
/// and only the FFN below it gets to amortize a weight read across the batch.
/// Returns true when the fused layer tail ran, in which case the caller must
/// not run its own FFN: the tail already did attention *and* the whole routed
/// MoE, one submission. `allow_tail` exists because decodeBatch shares this
/// function and batches the FFN itself -- a tail firing there would run the
/// FFN twice with different routing, and the output would be plausible.
fn attnLayer(m: *const Model, l: anytype, li: usize, st: *State, pos: usize, t_attn: *i128, allow_tail: bool) bool {
    const prof = Profile.enabled();
    const cfg = m.cfg;
    const kd = cfg.keyDim();
    const nope = cfg.nope_dim;
    const rope = cfg.rope_dim;
    const kvr = cfg.kv_lora_rank;
    const vd = cfg.v_head_dim;
    const scale = cfg.attn_scale;
    backend.rmsnorm(st.normed, st.x, asF32(l.attn_norm), cfg.eps);

    if (cfg.q_lora_rank > 0) {
        mv(l.attn_q_a.?, st.q_a, st.normed);
        backend.rmsnorm(st.q_a, st.q_a, asF32(l.attn_q_a_norm.?), cfg.eps);
        mv(l.attn_q_b.?, st.q, st.q_a);
    } else {
        mv(l.attn_q.?, st.q, st.normed);
    }

    mv(l.attn_kv_a_mqa, st.kv_a, st.normed);
    const c_kv = st.c_kv_cache[(li * cfg.ctx_len + pos) * kvr ..][0..kvr];
    backend.rmsnorm(c_kv, st.kv_a[0..kvr], asF32(l.attn_kv_a_norm), cfg.eps);
    const k_rope = st.k_rope_cache[(li * cfg.ctx_len + pos) * rope ..][0..rope];
    @memcpy(k_rope, st.kv_a[kvr .. kvr + rope]);
    ropeApply(cfg, k_rope, pos);
    // Mirror onto the device if there is one. The host copy above stays
    // authoritative: this can decline at any point and the fallback below must
    // still be correct.
    _ = backend.mlaAppend(li, pos, c_kv, k_rope);

    var h: usize = 0;
    while (h < cfg.n_heads) : (h += 1) {
        ropeApply(cfg, st.q[h * kd + nope ..][0..rope], pos);
    }

    // MLA attention, in compressed space.
    //
    // The obvious way to write this is to decompress the cache: for every
    // head, at every cached position, run W_k over c_t to rebuild that
    // position's key, and W_v to rebuild its value. That is what this did,
    // and it is O(seq) matvecs of (nope x kvr) and (vd x kvr) per head per
    // layer -- at seq=100, 16 heads and 27 layers, about 86,000 matvecs
    // per token. It also throws away the reason MLA exists: the whole
    // point of a compressed KV cache is that K and V are never
    // materialized.
    //
    // Both projections commute with the sum over positions, so both move
    // out of the loop:
    //
    //   score_t = q_nope . (W_k c_t)  =  (W_k^T q_nope) . c_t  =  q_abs . c_t
    //   o       = sum_t p_t (W_v c_t) =  W_v (sum_t p_t c_t)   =  W_v c_acc
    //
    // So W_k is absorbed into q once per head, and the value side
    // accumulates in compressed space and decompresses once. Per head the
    // per-position work drops from two matvecs (~131k MACs) to two dot
    // products (kvr + rope = 576 MACs), against a fixed cost of one
    // absorb and one matvec.
    const seq = pos + 1;
    const t_a0 = if (prof) Profile.now() else 0;

    // Absorb W_k into q for every head. Both paths need it, and both need
    // every head's laid out together.
    //
    // The host form dequantizes one kv_b row at a time, which is a
    // synchronization point in the middle of the layer -- the last one, once
    // attention itself moved to the device.
    h = 0;
    while (h < cfg.n_heads) : (h += 1) {
        @memcpy(st.q_nope_all[h * nope ..][0..nope], st.q[h * kd ..][0..nope]);
        @memcpy(st.mla_qr[h * rope ..][0..rope], st.q[h * kd + nope ..][0..rope]);
    }
    // One call, two dispatches, one command buffer: absorption then attention,
    // with q_abs staying on the device between them.
    // The whole rest of the layer -- absorb, attention, W_v, projection,
    // residual, norm, router, routed MoE, second residual -- as one
    // submission, when the backend will take it. Local MoE layers only.
    const t_tail0 = if (prof) Profile.now() else 0;
    // LOOM_NO_TAIL bisects the fused tail against the two-buffer path in one
    // binary; cross-run comparison on this machine has misled before.
    if (allow_tail and l.is_moe and m.dist == null and std.c.getenv("LOOM_NO_TAIL") == null) {
        if (l.ffn_gate_exps) |gt| {
            const rbias: ?[]const f32 = if (l.exp_probs_b) |b| asF32(b) else null;
            const shexp: ?[3]backend.WeightRef = if (l.ffn_gate_shexp) |gs| .{
                .{ .ty = gs.ty, .data = gs.data },
                .{ .ty = l.ffn_up_shexp.?.ty, .data = l.ffn_up_shexp.?.data },
                .{ .ty = l.ffn_down_shexp.?.ty, .data = l.ffn_down_shexp.?.data },
            } else null;
            if (backend.mlaLayerTail(
                li,
                pos,
                st.x,
                st.q_nope_all,
                st.mla_qr,
                .{ .ty = l.attn_kv_b.ty, .data = l.attn_kv_b.data },
                .{ .ty = l.attn_output.ty, .data = l.attn_output.data },
                asF32(l.ffn_norm),
                cfg.eps,
                .{ .ty = l.ffn_gate_inp.?.ty, .data = l.ffn_gate_inp.?.data },
                rbias,
                .{ .ty = gt.ty, .data = gt.data },
                .{ .ty = l.ffn_up_exps.?.ty, .data = l.ffn_up_exps.?.data },
                .{ .ty = l.ffn_down_exps.?.ty, .data = l.ffn_down_exps.?.data },
                shexp,
                gt.ne1,
                if (l.ffn_gate_shexp) |gs| gs.ne1 else 0,
                .{
                    .n_expert = cfg.n_expert,
                    .n_used = cfg.n_used,
                    .gating_sigmoid = cfg.gating == .sigmoid,
                    .weights_norm = cfg.weights_norm,
                    .weights_scale = cfg.weights_scale,
                },
                cfg.n_heads,
                nope,
                vd,
                scale,
            )) {
                if (prof) t_attn.* += Profile.now() - t_tail0;
                return true;
            }
        }
    }

    // One call, three dispatches, one command buffer: absorption, attention,
    // and W_v -- the result is head_out, ready for the output projection. The
    // fallback below is the full host loop including its own W_v.
    const on_device = backend.mlaAttnHeads(
        li,
        pos,
        st.q_nope_all,
        st.mla_qr,
        .{ .ty = l.attn_kv_b.ty, .data = l.attn_kv_b.data },
        st.head_out[0 .. cfg.n_heads * vd],
        cfg.n_heads,
        nope,
        vd,
        scale,
    );
    if (!on_device) {
        h = 0;
        while (h < cfg.n_heads) : (h += 1) {
            const q_nope = st.q[h * kd ..][0..nope];
            const kb_base = h * (nope + vd);
            const k_rows = l.attn_kv_b.data[kb_base * ggml.rowBytes(l.attn_kv_b.ty, kvr) ..];
            const qa = st.mla_qa[h * kvr ..][0..kvr];
            @memset(qa, 0);
            for (q_nope, 0..) |qr, r| {
                backend.dequantRow(l.attn_kv_b.ty, st.row_tmp, k_rows, r, kvr);
                backend.axpy(qa, st.row_tmp, qr);
            }
        }
    }

    h = 0;
    while (h < cfg.n_heads) : (h += 1) {
        if (on_device) break; // head_out is already complete
        const kb_base = h * (nope + vd);
        const v_rows = l.attn_kv_b.data[(kb_base + nope) * ggml.rowBytes(l.attn_kv_b.ty, kvr) ..];
        const o_lat = blk: {
            // Host fallback: scores, softmax and the weighted sum, all in
            // compressed space.
            const qa = st.mla_qa[h * kvr ..][0..kvr];
            const q_rope = st.mla_qr[h * rope ..][0..rope];
            var t_i: usize = 0;
            while (t_i < seq) : (t_i += 1) {
                const c_t = st.c_kv_cache[(li * cfg.ctx_len + t_i) * kvr ..][0..kvr];
                const kr_t = st.k_rope_cache[(li * cfg.ctx_len + t_i) * rope ..][0..rope];
                st.scores[t_i] = (backend.dotF32(qa, c_t) + backend.dotF32(q_rope, kr_t)) * scale;
            }
            backend.softmax(st.scores[0..seq]);
            @memset(st.c_acc, 0);
            t_i = 0;
            while (t_i < seq) : (t_i += 1) {
                const c_t = st.c_kv_cache[(li * cfg.ctx_len + t_i) * kvr ..][0..kvr];
                backend.axpy(st.c_acc, c_t, st.scores[t_i]);
            }
            break :blk st.c_acc;
        };
        // W_v applied to the compressed result, once per head either way.
        backend.matvec(l.attn_kv_b.ty, st.head_out[h * vd ..][0..vd], v_rows, o_lat, vd, kvr);
    }
    if (prof) t_attn.* += Profile.now() - t_a0;
    mv(l.attn_output, st.proj_out, st.head_out);
    backend.add(st.x, st.proj_out);
    return false;
}

/// Largest batch a decode step will take. Bounded by `backend.MAX_BATCH`,
/// which is what the batched kernels are written for.
pub const MAX_DECODE_BATCH = backend.MAX_BATCH;

/// Decode one token for each of several sequences at once.
///
/// Attention stays per sequence -- each has its own KV cache and its own
/// position, so there is nothing to share -- and the FFN is where the batch
/// pays: a MoE layer costs the *union* of the sequences' experts rather than
/// the sum, because `expertFfnBatched` walks each expert's weights once for
/// every sequence that routed to it.
///
/// Falls back to `step` for a single sequence, which keeps the recorded
/// whole-layer GPU path in use where it is the faster one.
///
/// Deliberately *not* named `stepBatch`: that name already means batched
/// prefill of one sequence over consecutive positions, and the engine
/// dispatches on `@hasDecl(E, "stepBatch")`. Reusing it made the prefill path
/// call this with a single state where a slice was wanted -- caught only by
/// the Linux cross-build, since the two differ in which engines they
/// instantiate.
pub fn decodeBatch(
    m: *const Model,
    states: []const *State,
    tokens: []const u32,
    positions: []const usize,
) !void {
    const n = states.len;
    std.debug.assert(n == tokens.len and n == positions.len);
    if (n == 0) return;
    if (n == 1) return step(m, states[0], tokens[0], positions[0]);
    if (n > MAX_DECODE_BATCH) return error.BatchTooLarge;

    const cfg = m.cfg;
    var t_attn: i128 = 0;

    for (states, tokens) |st, token| {
        if (token >= cfg.vocab) return error.TokenOutOfRange;
        backend.dequantRow(m.token_embd.ty, st.x, m.token_embd.data, token, cfg.dim);
    }

    // Per-batch scratch. Sized for the widest FFN any expert in this model has,
    // which is the shared expert where there is one.
    var xs: [MAX_DECODE_BATCH * 8192]f32 = undefined;
    var og: [MAX_DECODE_BATCH * 8192]f32 = undefined;
    var ou: [MAX_DECODE_BATCH * 8192]f32 = undefined;
    var od: [MAX_DECODE_BATCH * 8192]f32 = undefined;
    var acc: [MAX_DECODE_BATCH * 8192]f32 = undefined;
    const widest = @max(cfg.ffn, cfg.moe_ffn * @max(@as(usize, 1), cfg.n_shared));
    if (cfg.dim > 8192 or widest > 8192) return error.ModelTooWideForBatch;

    for (m.layers, 0..) |l, li| {
        for (states, positions) |st, pos| _ = attnLayer(m, l, li, st, pos, &t_attn, false);
        for (states) |st| backend.rmsnorm(st.normed, st.x, asF32(l.ffn_norm), cfg.eps);

        if (l.ffn_gate_exps == null) {
            // Dense prefix layer: one FFN, every sequence through it at once.
            for (states, 0..) |st, k| @memcpy(xs[k * cfg.dim ..][0..cfg.dim], st.normed);
            expertFfnBatched(l.ffn_gate.?, l.ffn_up.?, l.ffn_down.?, xs[0 .. n * cfg.dim], n, cfg.dim, &og, &ou, &od);
            for (states, 0..) |st, k| backend.add(st.x, od[k * cfg.dim ..][0..cfg.dim]);
            continue;
        }

        @memset(acc[0 .. n * cfg.dim], 0);

        // Route every sequence, then invert: expert -> the sequences that
        // chose it. The inversion is the batching; without it each expert is
        // read once per sequence.
        var picks: [MAX_DECODE_BATCH][moe.MAX_SELECTED]Selected = undefined;
        for (states, 0..) |st, k| {
            mv(l.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
            const bias: ?[]const f32 = if (l.exp_probs_b) |b| asF32(b) else null;
            moe.route(cfg.routeCfg(), st.router[0..cfg.n_expert], bias, picks[k][0..cfg.n_used]);
        }

        var seen: [MAX_DECODE_BATCH * moe.MAX_SELECTED]usize = undefined;
        var n_seen: usize = 0;
        for (0..n) |k| {
            for (picks[k][0..cfg.n_used]) |sel| {
                var dup = false;
                for (seen[0..n_seen]) |e| {
                    if (e == sel.expert) dup = true;
                }
                if (!dup) {
                    seen[n_seen] = sel.expert;
                    n_seen += 1;
                }
            }
        }

        for (seen[0..n_seen]) |expert| {
            // The sequences that picked this expert, and with what weight.
            var rows: [MAX_DECODE_BATCH]usize = undefined;
            var gates: [MAX_DECODE_BATCH]f32 = undefined;
            var nr: usize = 0;
            for (0..n) |k| {
                for (picks[k][0..cfg.n_used]) |sel| {
                    if (sel.expert == expert) {
                        rows[nr] = k;
                        gates[nr] = sel.gate;
                        nr += 1;
                        break;
                    }
                }
            }
            for (rows[0..nr], 0..) |k, r| @memcpy(xs[r * cfg.dim ..][0..cfg.dim], states[k].normed);

            const parts = try expertParts(m, l, li, expert);
            const gt = l.ffn_gate_exps.?;
            const ut = l.ffn_up_exps.?;
            const dt = l.ffn_down_exps.?;
            expertFfnBatched(
                .{ .ty = gt.ty, .data = parts.gate, .ne0 = gt.ne0, .ne1 = gt.ne1, .ne2 = 1 },
                .{ .ty = ut.ty, .data = parts.up, .ne0 = ut.ne0, .ne1 = ut.ne1, .ne2 = 1 },
                .{ .ty = dt.ty, .data = parts.down, .ne0 = dt.ne0, .ne1 = dt.ne1, .ne2 = 1 },
                xs[0 .. nr * cfg.dim],
                nr,
                cfg.dim,
                &og,
                &ou,
                &od,
            );
            for (rows[0..nr], gates[0..nr], 0..) |k, g, r| {
                const dst = acc[k * cfg.dim ..][0..cfg.dim];
                for (dst, od[r * cfg.dim ..][0..cfg.dim]) |*a, v| a.* += g * v;
            }
        }

        if (l.ffn_gate_shexp) |gs| {
            // Always on for every sequence, so the whole batch goes through it.
            for (states, 0..) |st, k| @memcpy(xs[k * cfg.dim ..][0..cfg.dim], st.normed);
            expertFfnBatched(gs, l.ffn_up_shexp.?, l.ffn_down_shexp.?, xs[0 .. n * cfg.dim], n, cfg.dim, &og, &ou, &od);
            for (0..n) |k| {
                const dst = acc[k * cfg.dim ..][0..cfg.dim];
                for (dst, od[k * cfg.dim ..][0..cfg.dim]) |*a, v| a.* += v;
            }
        }

        for (states, 0..) |st, k| backend.add(st.x, acc[k * cfg.dim ..][0..cfg.dim]);
    }

    for (states) |st| {
        backend.rmsnorm(st.normed, st.x, asF32(m.output_norm), cfg.eps);
        mv(m.output, st.logits, st.normed);
    }
}

pub fn step(m: *const Model, st: *State, token: u32, pos: usize) !void {
    const prof = Profile.enabled();
    const t_step = if (prof) Profile.now() else 0;
    var t_attn: i128 = 0;
    var t_get: i128 = 0;
    var t_ffn: i128 = 0;
    var t_shexp: i128 = 0;
    var n_block: u64 = 0;
    var n_host: u64 = 0;
    var t_route: i128 = 0;
    var t_acc: i128 = 0;
    const cfg = m.cfg;

    // A token id indexes token_embd rows directly. Ids come from the
    // tokenizer, whose ceiling is an independent metadata array, so bound it
    // here rather than trusting the two to agree (security issue #29).
    if (token >= cfg.vocab) return error.TokenOutOfRange;
    backend.dequantRow(m.token_embd.ty, st.x, m.token_embd.data, token, cfg.dim);

    for (m.layers, 0..) |l, li| {
        // ---- MLA attention ----
        if (attnLayer(m, l, li, st, pos, &t_attn, true)) {
            if (prof) n_block += 1;
            continue;
        }

        // ---- FFN ----
        backend.rmsnorm(st.normed, st.x, asF32(l.ffn_norm), cfg.eps);
        if (!l.is_moe) {
            denseFFN(st, l.ffn_gate.?, l.ffn_up.?, l.ffn_down.?);
            backend.add(st.x, st.ffn_out);
        } else {
            const t_r0 = if (prof) Profile.now() else 0;
            mv(l.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
            var sel_buf: [moe.MAX_SELECTED]Selected = undefined;
            const sel = sel_buf[0..cfg.n_used];
            const bias: ?[]const f32 = if (l.exp_probs_b) |b| asF32(b) else null;
            moe.route(cfg.routeCfg(), st.router[0..cfg.n_expert], bias, sel);
            if (prof) t_route += Profile.now() - t_r0;

            var acc_buf: [8192]f32 = undefined;
            const acc = acc_buf[0..cfg.dim];
            @memset(acc, 0);
            // Local path first: hand the backend the router *logits* and let it
            // route, select and reduce in one command buffer. Routing on the
            // host is a read-back between attention and the FFN, and that
            // read-back is the one thing keeping them in separate submissions.
            // The distributed path keeps host routing: its experts are offsets
            // into a store, not planes of a tensor.
            routed: {
                if (m.dist != null) break :routed;
                const gt = l.ffn_gate_exps orelse break :routed;
                const rbias: ?[]const f32 = if (l.exp_probs_b) |b| asF32(b) else null;
                const shexp: ?[3]backend.WeightRef = if (l.ffn_gate_shexp) |gs| .{
                    .{ .ty = gs.ty, .data = gs.data },
                    .{ .ty = l.ffn_up_shexp.?.ty, .data = l.ffn_up_shexp.?.data },
                    .{ .ty = l.ffn_down_shexp.?.ty, .data = l.ffn_down_shexp.?.data },
                } else null;
                const t_f2 = if (prof) Profile.now() else 0;
                const ok = backend.moeFfnBlockRouted(
                    st.normed,
                    st.router[0..cfg.n_expert],
                    rbias,
                    .{ .ty = gt.ty, .data = gt.data },
                    .{ .ty = l.ffn_up_exps.?.ty, .data = l.ffn_up_exps.?.data },
                    .{ .ty = l.ffn_down_exps.?.ty, .data = l.ffn_down_exps.?.data },
                    shexp,
                    gt.ne1,
                    if (l.ffn_gate_shexp) |gs| gs.ne1 else 0,
                    .{
                        .n_expert = cfg.n_expert,
                        .n_used = cfg.n_used,
                        .gating_sigmoid = cfg.gating == .sigmoid,
                        .weights_norm = cfg.weights_norm,
                        .weights_scale = cfg.weights_scale,
                    },
                    acc,
                );
                if (!ok) break :routed;
                if (prof) {
                    t_ffn += Profile.now() - t_f2;
                    n_block += 1;
                }
                backend.add(st.x, acc);
                continue;
            }
            if (m.dist) |src| {
                // warm the missing shards in parallel: per-layer miss latency
                // becomes max(fetch), not sum(fetch)
                var ids_buf: [moe.MAX_SELECTED]usize = undefined;
                for (sel, 0..) |s, k| ids_buf[k] = m.expert_shard[li * cfg.n_expert + s.expert];
                src.prefetch(ids_buf[0..sel.len]);
            }
            // Gather every selected expert first and hand the layer to the
            // backend as one unit, when it will take it. Dispatching each
            // expert separately costs a command buffer apiece; a MoE layer is
            // six of them plus a shared one.
            //
            // Only when the source hands back pointers that survive the next
            // `get`: with no RAM cache and no mapping every expert lands in
            // the same scratch buffer, so collecting six of them would leave
            // five dangling — and the resulting output is well-formed, which
            // is the failure mode this codebase keeps meeting.
            const batched = blk_b: {
                if (!@hasDecl(backend, "moeFfnBlock")) break :blk_b false;
                if (m.dist) |src| {
                    if (!src.stablePointers(sel.len)) break :blk_b false;
                }
                const gt = l.ffn_gate_exps orelse break :blk_b false;
                var refs: [moe.MAX_SELECTED + 1]backend.ExpertRef = undefined;
                const t_g1 = if (prof) Profile.now() else 0;
                for (sel, 0..) |s2, k| {
                    const parts = expertParts(m, l, li, s2.expert) catch break :blk_b false;
                    refs[k] = .{
                        .gate = .{ .ty = l.ffn_gate_exps.?.ty, .data = parts.gate },
                        .up = .{ .ty = l.ffn_up_exps.?.ty, .data = parts.up },
                        .down = .{ .ty = l.ffn_down_exps.?.ty, .data = parts.down },
                        .weight = s2.gate,
                        .ffn = gt.ne1,
                    };
                }
                var n_refs = sel.len;
                // The shared expert joins as one more expert: wider, always on,
                // unweighted. It lost against the per-expert block, where every
                // step sat behind a barrier; against the phase-parallel one it
                // is a seventh column of the same four phases. On the host it
                // was 20.2 ms/token, a third of the whole expert-FFN bucket.
                if (l.ffn_gate_shexp) |gs| {
                    refs[n_refs] = .{
                        .gate = .{ .ty = gs.ty, .data = gs.data },
                        .up = .{ .ty = l.ffn_up_shexp.?.ty, .data = l.ffn_up_shexp.?.data },
                        .down = .{ .ty = l.ffn_down_shexp.?.ty, .data = l.ffn_down_shexp.?.data },
                        .weight = 1.0,
                        .ffn = gs.ne1,
                    };
                    n_refs += 1;
                }
                if (prof) t_get += Profile.now() - t_g1;
                const t_f1 = if (prof) Profile.now() else 0;
                const ok = backend.moeFfnBlock(st.normed, refs[0..n_refs], acc);
                if (prof) t_ffn += Profile.now() - t_f1;
                break :blk_b ok;
            };

            if (prof) {
                if (batched) n_block += 1 else n_host += 1;
            }
            if (!batched) for (sel) |s| {
                if (m.dist) |src| {
                    const sid = m.expert_shard[li * cfg.n_expert + s.expert];
                    const gt = l.ffn_gate_exps.?;
                    const ut = l.ffn_up_exps.?;
                    const dt = l.ffn_down_exps.?;
                    const t_g0 = if (prof) Profile.now() else 0;
                    // Zero-copy where the store is mapped and the shard checks
                    // out; the copying path is the fallback for peer fetches
                    // and unmappable stores.
                    const parts: expert_fetch.Source.Parts = src.getMapped(sid) orelse blk_p: {
                        const blk = try src.get(sid);
                        const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
                        const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
                        const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
                        break :blk_p .{ .gate = blk[0..gl], .up = blk[gl..][0..ul], .down = blk[gl + ul ..][0..dl] };
                    };
                    if (prof) t_get += Profile.now() - t_g0;
                    const t_f0 = if (prof) Profile.now() else 0;
                    denseFFN(
                        st,
                        .{ .ty = gt.ty, .data = parts.gate, .ne0 = gt.ne0, .ne1 = gt.ne1, .ne2 = 1 },
                        .{ .ty = ut.ty, .data = parts.up, .ne0 = ut.ne0, .ne1 = ut.ne1, .ne2 = 1 },
                        .{ .ty = dt.ty, .data = parts.down, .ne0 = dt.ne0, .ne1 = dt.ne1, .ne2 = 1 },
                    );
                    if (prof) t_ffn += Profile.now() - t_f0;
                } else {
                    denseFFN(
                        st,
                        try l.ffn_gate_exps.?.expert(s.expert),
                        try l.ffn_up_exps.?.expert(s.expert),
                        try l.ffn_down_exps.?.expert(s.expert),
                    );
                }
                const t_ac0 = if (prof) Profile.now() else 0;
                for (acc, st.ffn_out) |*a, v| a.* += s.gate * v;
                if (prof) t_acc += Profile.now() - t_ac0;
            };
            // Timed, not merely run: this is a 2816-wide dense FFN on every
            // layer -- ~225 MB/token, a third of what the routed experts move
            // -- and leaving it outside `t_ffn` put it in "everything else",
            // which was 30% of a token and the largest unexplained bucket.
            //
            // It deliberately stays on the host path. Folding it into
            // `moeFfnBlock` was tried and is 6% *slower* (95.0 against 89.5
            // ms/token, A/B'd in one binary): it moves the work onto a Metal
            // matvec that loses to the CPU at these row counts, which is the
            // same verdict calibration reaches unprompted. Worth revisiting
            // once the kernel wins at 1408-2816 rows, not before.
            // Only when the block did not already take it.
            const folded = batched and l.ffn_gate_shexp != null;
            if (!folded) if (l.ffn_gate_shexp) |gs| {
                const t_s0 = if (prof) Profile.now() else 0;
                denseFFN(st, gs, l.ffn_up_shexp.?, l.ffn_down_shexp.?);
                backend.add(acc, st.ffn_out);
                if (prof) {
                    const d = Profile.now() - t_s0;
                    t_ffn += d;
                    t_shexp += d;
                }
            };
            backend.add(st.x, acc);
        }
    }

    const t_h0 = if (prof) Profile.now() else 0;
    backend.rmsnorm(st.normed, st.x, asF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
    const t_head = if (prof) Profile.now() - t_h0 else 0;

    if (prof) {
        if (m.dist) |src| {
            Profile.src_stats = src.stats;
            Profile.cache_slots = src.cacheSlots();
        }
        Profile.cmdbufs += backend.takeCmdBufCount();
        const total = Profile.now() - t_step;
        Profile.attn_ns += t_attn;
        Profile.expert_get_ns += t_get;
        Profile.expert_ffn_ns += t_ffn;
        Profile.shexp_ns += t_shexp;
        Profile.moe_block += n_block;
        Profile.moe_host += n_host;
        Profile.route_ns += t_route;
        Profile.acc_ns += t_acc;
        Profile.head_ns += t_head;
        Profile.other_ns += total - t_attn - t_get - t_ffn;
        Profile.tokens += 1;
        if (Profile.tokens % 16 == 0) Profile.dump();
    }
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
    moe.route(cfg.routeCfg(), &logits, &bias, &sel);
    for (sel) |s| try std.testing.expect(s.expert == 2 or s.expert == 3);
    // gates renormalized over raw sigmoid scores of the selected pair
    var sum: f32 = 0;
    for (sel) |s| sum += s.gate;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // without bias, raw scores pick 0,1
    var sel2: [2]Selected = undefined;
    moe.route(cfg.routeCfg(), &logits, null, &sel2);
    for (sel2) |s| try std.testing.expect(s.expert == 0 or s.expert == 1);
}

/// Dequantize each layer's W_k rows and hand them to the compute backend.
///
/// Best-effort and all-or-nothing per layer: `mlaAbsorb` declines any layer it
/// was not given, and the host loop covers those, so a partial upload is
/// correct if slower. Called after the context is capped, with the rest of the
/// device-side attention setup.
pub fn uploadAbsorbWeights(m: *const Model, gpa: std.mem.Allocator) void {
    const cfg = m.cfg;
    const kvr = cfg.kv_lora_rank;
    const stride = cfg.nope_dim + cfg.v_head_dim;
    const per_layer = cfg.n_heads * stride * kvr;
    const buf = gpa.alloc(f32, per_layer) catch return;
    defer gpa.free(buf);
    for (m.layers, 0..) |l, li| {
        // Every row of the head's slice, W_k and W_v alike: the kernel strides
        // past W_v rather than expecting them removed, so the layout on the
        // device matches the tensor exactly.
        var r: usize = 0;
        while (r < cfg.n_heads * stride) : (r += 1) {
            backend.dequantRow(l.attn_kv_b.ty, buf[r * kvr ..][0..kvr], l.attn_kv_b.data, r, kvr);
        }
        if (!backend.mlaSetWk(li, buf)) return;
    }
}

/// A small MLA model with f32 weights, built in memory.
///
/// `parsed`, `file` and `mm` are left undefined: nothing in the forward path
/// reads them, and the caller must not call `deinit`. That is the price of
/// testing the engine without a checkpoint on disk, and it is worth paying --
/// the alternative is that the batched decode path has no test at all, which
/// is how three of this session's bugs survived long enough to matter.
const Fixture = struct {
    m: Model,
    bufs: std.ArrayListUnmanaged([]f32),
    gpa: std.mem.Allocator,

    fn t2(self: *Fixture, rnd: std.Random, ne0: usize, ne1: usize) !Tensor {
        const b = try self.gpa.alloc(f32, ne0 * ne1);
        for (b) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
        try self.bufs.append(self.gpa, b);
        return .{ .ty = .f32, .data = std.mem.sliceAsBytes(b), .ne0 = ne0, .ne1 = ne1, .ne2 = 1 };
    }

    fn t3(self: *Fixture, rnd: std.Random, ne0: usize, ne1: usize, ne2: usize) !Tensor {
        const b = try self.gpa.alloc(f32, ne0 * ne1 * ne2);
        for (b) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
        try self.bufs.append(self.gpa, b);
        return .{ .ty = .f32, .data = std.mem.sliceAsBytes(b), .ne0 = ne0, .ne1 = ne1, .ne2 = ne2 };
    }

    fn norm(self: *Fixture, n: usize) !Tensor {
        const b = try self.gpa.alloc(f32, n);
        for (b) |*v| v.* = 1.0;
        try self.bufs.append(self.gpa, b);
        return .{ .ty = .f32, .data = std.mem.sliceAsBytes(b), .ne0 = n, .ne1 = 1, .ne2 = 1 };
    }

    fn deinit(self: *Fixture) void {
        for (self.bufs.items) |b| self.gpa.free(b);
        self.bufs.deinit(self.gpa);
        self.gpa.free(self.m.layers);
    }
};

fn buildFixture(gpa: std.mem.Allocator, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const cfg = Config{
        .dim = 64,
        .n_layers = 3,
        .n_dense_layers = 1, // exercises both the dense prefix and MoE layers
        .n_heads = 2,
        .q_lora_rank = 0, // V2-Lite's direct q projection
        .kv_lora_rank = 16,
        .rope_dim = 8,
        .nope_dim = 8,
        .v_head_dim = 16,
        .ffn = 128,
        .moe_ffn = 32,
        .n_expert = 8,
        .n_used = 3,
        .n_shared = 1,
        .gating = .sigmoid,
        .weights_norm = true,
        .weights_scale = 1.0,
        .vocab = 32,
        .ctx_len = 16,
        .rope_base = 10000.0,
        .eps = 1e-6,
        .yarn_factor = 0,
        .yarn_orig_ctx = 0,
        .yarn_log_mul = 0,
        .attn_scale = 0.25,
    };
    var f = Fixture{ .m = undefined, .bufs = .empty, .gpa = gpa };
    errdefer f.deinit();

    const kd = cfg.keyDim();
    const layers = try gpa.alloc(LayerT, cfg.n_layers);
    for (layers, 0..) |*l, i| {
        const moe_layer = i >= cfg.n_dense_layers;
        l.* = .{
            .attn_norm = try f.norm(cfg.dim),
            .attn_q = try f.t2(rnd, cfg.dim, cfg.n_heads * kd),
            .attn_q_a = null,
            .attn_q_a_norm = null,
            .attn_q_b = null,
            .attn_kv_a_mqa = try f.t2(rnd, cfg.dim, cfg.kv_lora_rank + cfg.rope_dim),
            .attn_kv_a_norm = try f.norm(cfg.kv_lora_rank),
            .attn_kv_b = try f.t2(rnd, cfg.kv_lora_rank, cfg.n_heads * (cfg.nope_dim + cfg.v_head_dim)),
            .attn_output = try f.t2(rnd, cfg.n_heads * cfg.v_head_dim, cfg.dim),
            .ffn_norm = try f.norm(cfg.dim),
            .ffn_gate = if (moe_layer) null else try f.t2(rnd, cfg.dim, cfg.ffn),
            .ffn_up = if (moe_layer) null else try f.t2(rnd, cfg.dim, cfg.ffn),
            .ffn_down = if (moe_layer) null else try f.t2(rnd, cfg.ffn, cfg.dim),
            .ffn_gate_inp = if (moe_layer) try f.t2(rnd, cfg.dim, cfg.n_expert) else null,
            .exp_probs_b = null,
            .ffn_gate_exps = if (moe_layer) try f.t3(rnd, cfg.dim, cfg.moe_ffn, cfg.n_expert) else null,
            .ffn_up_exps = if (moe_layer) try f.t3(rnd, cfg.dim, cfg.moe_ffn, cfg.n_expert) else null,
            .ffn_down_exps = if (moe_layer) try f.t3(rnd, cfg.moe_ffn, cfg.dim, cfg.n_expert) else null,
            .ffn_gate_shexp = if (moe_layer) try f.t2(rnd, cfg.dim, cfg.moe_ffn * cfg.n_shared) else null,
            .ffn_up_shexp = if (moe_layer) try f.t2(rnd, cfg.dim, cfg.moe_ffn * cfg.n_shared) else null,
            .ffn_down_shexp = if (moe_layer) try f.t2(rnd, cfg.moe_ffn * cfg.n_shared, cfg.dim) else null,
            .is_moe = moe_layer,
        };
    }
    f.m = .{
        .gpa = gpa,
        .io = undefined,
        .parsed = undefined,
        .file = undefined,
        .mm = undefined,
        .cfg = cfg,
        .token_embd = try f.t2(rnd, cfg.dim, cfg.vocab),
        .output_norm = try f.norm(cfg.dim),
        .output = try f.t2(rnd, cfg.dim, cfg.vocab),
        .layers = layers,
        .tok = undefined,
    };
    return f;
}

test "decodeBatch agrees with sequential step, token for token" {
    // The property continuous batching rests on: batching changes only which
    // weight reads are shared, never the arithmetic a sequence sees. Two
    // sequences fed different tokens must land on exactly the logits they
    // would have had alone.
    //
    // Different tokens on purpose. Identical ones would pass even if the
    // routing inversion collapsed the batch to one sequence, which is the
    // mistake this is here to catch.
    const gpa = std.testing.allocator;
    var f = try buildFixture(gpa, 0xBA7C4);
    defer f.deinit();
    const cfg = f.m.cfg;

    const seqs = [_][4]u32{ .{ 1, 5, 9, 3 }, .{ 7, 2, 4, 8 } };

    // Sequential: each sequence alone, through `step`.
    var want: [seqs.len][]f32 = undefined;
    for (seqs, 0..) |toks, s| {
        var st = try State.init(gpa, cfg);
        defer st.deinit(gpa);
        for (toks, 0..) |tk, pos| try step(&f.m, &st, tk, pos);
        want[s] = try gpa.dupe(f32, st.logits[0..cfg.vocab]);
    }
    defer for (want) |w| gpa.free(w);

    // Batched: both together, one position at a time.
    var sts: [seqs.len]State = undefined;
    for (&sts) |*st| st.* = try State.init(gpa, cfg);
    defer for (&sts) |*st| st.deinit(gpa);
    var ptrs: [seqs.len]*State = undefined;
    for (&sts, 0..) |*st, i| ptrs[i] = st;

    for (0..seqs[0].len) |pos| {
        var toks: [seqs.len]u32 = undefined;
        var poss: [seqs.len]usize = undefined;
        for (seqs, 0..) |sq, i| {
            toks[i] = sq[pos];
            poss[i] = pos;
        }
        try decodeBatch(&f.m, &ptrs, &toks, &poss);
    }

    for (0..seqs.len) |s| {
        var mass: f32 = 0;
        for (want[s]) |v| mass += @abs(v);
        const tol = (mass / @as(f32, @floatFromInt(cfg.vocab))) * 1e-4;
        for (want[s], sts[s].logits[0..cfg.vocab], 0..) |a, b, k| {
            std.testing.expectApproxEqAbs(a, b, tol) catch |e| {
                std.debug.print("seq {d} logit {d}: sequential {d} vs batched {d}\n", .{ s, k, a, b });
                return e;
            };
        }
        // argmax is what actually decides the token
        var ia: usize = 0;
        var ib: usize = 0;
        for (want[s], 0..) |v, k| if (v > want[s][ia]) {
            ia = k;
        };
        for (sts[s].logits[0..cfg.vocab], 0..) |v, k| if (v > sts[s].logits[ib]) {
            ib = k;
        };
        try std.testing.expectEqual(ia, ib);
    }
}
