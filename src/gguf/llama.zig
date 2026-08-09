//! GQA-attention inference over a GGUF file — the engine behind every
//! architecture loom runs except deepseek2 (whose MLA lives in deepseek.zig).
//!
//! One forward pass covers llama (including Mixtral), qwen2moe, qwen3moe and
//! glm4moe, because in llama.cpp these differ only in optional pieces bolted
//! onto the same skeleton: RMSNorm, GQA attention, RoPE, SwiGLU FFN,
//! tied-or-separate output head. The variable parts are
//!
//!   - RoPE style (adjacent-pair NORM vs split-half NEOX), and how much of
//!     each head is rotated,
//!   - QKV biases, and per-head Q/K RMSNorm before RoPE,
//!   - whether the FFN is dense or a mixture of experts, with or without a
//!     shared expert, and whether that shared expert is sigmoid-gated,
//!   - a post-attention norm standing in for ffn_norm.
//!
//! Everything except the RoPE style is detected from the tensors the file
//! actually contains, so the engine follows the checkpoint rather than a
//! table of beliefs about each architecture.
//!
//! Weights stay in their GGML format in a read-only memory map; every matmul
//! is a fused kernel over the raw bytes (ggml.zig). Routed experts can instead
//! come from the distribution plane — see attachDist.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const backend = @import("../compute/backend.zig");
const tensor = @import("../core/tensor.zig");
const spm = @import("spm.zig");
const moe = @import("moe.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");

/// Re-exported: the SPM tokenizer moved to spm.zig so every engine can share
/// it without an import cycle.
pub const Tokenizer = spm.Tokenizer;

/// The file's tokenizer (SPM or gpt2-BPE); see tok.zig.
pub const Tok = @import("tok.zig").Tok;

/// The GQA-attention architectures this engine covers. Each is a set of
/// optional features over the same skeleton, so they share one forward pass
/// rather than one file each. The only hard per-arch fact is the RoPE style;
/// everything else (QKV bias, Q/K norm, MoE, shared experts, leading dense
/// layers, a post-attention norm standing in for ffn_norm) is detected from
/// the tensors the file actually contains.
///
/// Verified against llama.cpp's `llama_model_rope_type` and the per-model
/// graph builders in `src/models/`.
pub const Arch = struct {
    name: []const u8, // also the metadata key prefix
    rope: RopeStyle,
    /// gpt-oss: swiglu_oai expert activation, per-expert FFN biases,
    /// attention sinks, alternating sliding-window layers, YaRN rope.
    oai: bool = false,
};

pub const RopeStyle = enum { norm, neox };

pub const arches = [_]Arch{
    .{ .name = "llama", .rope = .norm }, // includes Mixtral (MoE llama)
    .{ .name = "qwen2moe", .rope = .neox },
    .{ .name = "qwen3moe", .rope = .neox },
    .{ .name = "glm4moe", .rope = .neox },
    .{ .name = "gpt-oss", .rope = .neox, .oai = true },
};

pub fn archFor(name: []const u8) ?Arch {
    for (arches) |a| {
        if (std.mem.eql(u8, a.name, name)) return a;
    }
    return null;
}

pub const Config = struct {
    arch: Arch,
    dim: usize,
    head_dim: usize, // may differ from dim/n_heads (qwen3 sets key_length)
    n_layers: usize, // excludes trailing MTP/NextN blocks
    /// A NextN/MTP block was loaded; State grows one KV lane and the spec
    /// scratch, and greedy decode may draft ahead (stepSpec).
    has_mtp: bool = false,
    /// Sliding-window size for the layers that use one (gpt-oss: even layers,
    /// window 128); 0 = no windowed layers.
    swa_window: usize = 0,
    /// YaRN rope scaling (gpt-oss: factor 32 over a 4096 original context).
    yarn_factor: f32 = 0,
    yarn_orig_ctx: f32 = 0,
    /// Multiplier on the attention logits: mscale^2 under YaRN (the rope
    /// magnitude scale applied to q and k, folded into the score instead).
    attn_logit_mul: f32 = 1.0,
    n_dense_layers: usize, // leading layers with a plain FFN
    n_heads: usize,
    n_kv_heads: usize,
    ffn: usize,
    moe_ffn: usize, // per-routed-expert hidden size
    shexp_ffn: usize, // shared-expert hidden size (total across n_shared)
    n_expert: usize,
    n_used: usize,
    n_shared: usize,
    route: moe.RouteCfg,
    vocab: usize,
    ctx_len: usize,
    rope_dim: usize, // may be < head_dim (glm4moe rotates a prefix)
    rope_base: f32,
    eps: f32,

    pub fn headDim(self: Config) usize {
        return self.head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
    /// Widest FFN intermediate any layer needs, which is what State sizes its
    /// gate/up/act buffers to.
    pub fn maxFfn(self: Config) usize {
        return @max(self.ffn, @max(self.moe_ffn, self.shexp_ffn));
    }
};

pub const Tensor = struct {
    ty: ggml.Type,
    data: []const u8,
    ne0: usize, // row length (input dim)
    ne1: usize, // rows (output dim)
    ne2: usize = 1, // experts, for the stacked 3D *_exps tensors

    /// Byte slice of expert `e` in a 3D tensor. `e` comes from the router and
    /// is bounded by the config's expert count, while `ne2` comes from the
    /// file: a file declaring fewer experts than the config would slice past
    /// the mapping (security issue #29), so bound it here too.
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
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_output: Tensor,
    // optional attention extras, present or absent per architecture
    attn_q_bias: ?Tensor,
    attn_k_bias: ?Tensor,
    attn_v_bias: ?Tensor,
    attn_q_norm: ?Tensor, // per-head RMSNorm applied before RoPE
    attn_k_norm: ?Tensor,
    /// The norm before the FFN. glm4moe names it `post_attention_norm` and
    /// ships no `ffn_norm`; functionally they are the same slot.
    ffn_norm: Tensor,
    // dense FFN (leading layers, or a non-MoE model)
    ffn_gate: ?Tensor,
    ffn_up: ?Tensor,
    ffn_down: ?Tensor,
    // MoE
    ffn_gate_inp: ?Tensor,
    exp_probs_b: ?Tensor, // selection bias (noaux_tc)
    ffn_gate_exps: ?Tensor,
    ffn_up_exps: ?Tensor,
    ffn_down_exps: ?Tensor,
    ffn_gate_shexp: ?Tensor,
    ffn_up_shexp: ?Tensor,
    ffn_down_shexp: ?Tensor,
    /// qwen2moe gates its shared expert by sigmoid(w . x); the others add the
    /// shared expert output unweighted.
    ffn_gate_inp_shexp: ?Tensor,
    is_moe: bool,
    /// gpt-oss additions: per-head sink logits, sliding-window flag,
    /// router bias, per-expert FFN biases.
    attn_sinks: ?Tensor = null,
    attn_output_b: ?Tensor = null,
    is_swa: bool = false,
    ffn_gate_inp_b: ?Tensor = null,
    ffn_gate_exps_b: ?Tensor = null,
    ffn_up_exps_b: ?Tensor = null,
    ffn_down_exps_b: ?Tensor = null,
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
    output: Tensor, // == token_embd when tied
    layers: []LayerT,

    tok: Tok,

    /// Distributed expert source (issue #3): when set, routed-expert weights
    /// come through Source.get() — local tier or peer fetch — instead of the
    /// (possibly sparse) memory map.
    dist: ?*expert_fetch.Source = null,
    /// PILOT router-lookahead prefetch (see moeLayer). On by default for the
    /// distributed path; LOOM_NO_PILOT=1 disables for A/B measurement.
    pilot_enabled: bool = false,
    /// GLM/DeepSeek-style NextN block: a full extra transformer layer plus
    /// the glue that turns (hidden state, next-token embedding) into a draft
    /// of the token after next. Loaded from blk.{n_layers}; null when the
    /// file has none or any tensor is missing.
    mtp: ?Mtp = null,
    /// Set at load when every layer can be recorded into one command buffer.
    /// See gpuLayersSupported.
    gpu_layers: bool = false,
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

    fn resolveOpt(self: *const Model, name: []const u8) ?Tensor {
        return self.resolve(name) catch null;
    }

    /// Assert a tensor really is F32. `tensorAsF32` reinterprets the mapped
    /// bytes as floats behind a debug assert, which is a no-op in ReleaseFast:
    /// a norm weight declared as q8_0 would otherwise be reinterpreted
    /// (security issue #29). Alignment is checked too, since the data offset is
    /// file-controlled and `general.alignment` may legally be 1 or 2.
    fn expectF32(t: Tensor, name: []const u8) !void {
        if (t.ty != .f32) {
            std.debug.print("gguf: tensor {s} must be f32, got {s}\n", .{ name, @tagName(t.ty) });
            return error.BadTensorType;
        }
        if (@intFromPtr(t.data.ptr) % @alignOf(f32) != 0) return error.BadTensorAlignment;
        if (t.data.len % @sizeOf(f32) != 0) return error.BadTensorType;
    }

    /// Assert a resolved tensor has exactly the shape the config implies.
    /// Without this the kernels trust the file's dims and the config
    /// independently: e.g. a `[dim, 1<<20]` q weight against a smaller
    /// destination is a heap overflow write (security issue #29).
    fn expectShape(t: Tensor, ne0: usize, ne1: usize, name: []const u8) !void {
        if (t.ne0 != ne0 or t.ne1 != ne1) {
            std.debug.print("gguf: tensor {s} has shape [{d},{d}], config implies [{d},{d}]\n", .{ name, t.ne0, t.ne1, ne0, ne1 });
            return error.BadTensorShape;
        }
    }

    /// Same, for the stacked 3D expert tensors.
    fn expectExpertShape(t: Tensor, ne0: usize, ne1: usize, ne2: usize, name: []const u8) !void {
        if (t.ne0 != ne0 or t.ne1 != ne1 or t.ne2 != ne2) {
            std.debug.print("gguf: tensor {s} has shape [{d},{d},{d}], config implies [{d},{d},{d}]\n", .{ name, t.ne0, t.ne1, t.ne2, ne0, ne1, ne2 });
            return error.BadTensorShape;
        }
    }
};

/// Reject metadata that would divide by zero, overflow a size computation, or
/// index past a fixed-size buffer (security issue #29). Everything here comes
/// from an untrusted file: a peer-supplied GGUF or a Hugging Face download.
fn validateConfig(cfg: Config) !void {
    if (cfg.dim == 0 or cfg.n_heads == 0 or cfg.n_layers == 0) return error.BadConfig;
    if (cfg.head_dim == 0 or cfg.ctx_len == 0) return error.BadConfig;
    if (cfg.n_kv_heads == 0 or cfg.n_heads % cfg.n_kv_heads != 0) return error.BadConfig;
    if (cfg.rope_dim > cfg.head_dim) return error.BadConfig;
    if (cfg.n_dense_layers > cfg.n_layers) return error.BadConfig;
    if (cfg.n_used > moe.MAX_SELECTED) return error.BadConfig; // sel_buf/ids_buf
    if (cfg.n_expert > moe.MAX_EXPERTS) return error.BadConfig; // route() score buffers
    if (cfg.n_expert != 0 and cfg.n_used > cfg.n_expert) return error.BadConfig;
    // a model with MoE layers must declare a usable expert configuration
    if (cfg.n_layers > cfg.n_dense_layers and cfg.n_expert != 0) {
        if (cfg.n_used == 0 or cfg.moe_ffn == 0) return error.BadConfig;
    }
    if (cfg.n_expert == 0 and cfg.ffn == 0) return error.BadConfig;
    // scores/kv allocations multiply these; keep the products representable
    const kv = std.math.mul(usize, cfg.n_layers, cfg.ctx_len) catch return error.BadConfig;
    _ = std.math.mul(usize, kv, cfg.kvDim()) catch return error.BadConfig;
    _ = std.math.mul(usize, cfg.n_layers, cfg.n_expert) catch return error.BadConfig;
}

pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Model {
    var parsed = try gguf.parse(gpa, io, path);
    errdefer parsed.deinit();

    const arch_name = parsed.getString("general.architecture") orelse return error.NoArchitecture;
    const arch = archFor(arch_name) orelse return error.UnsupportedArchitecture;

    const file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    var mm = try file.createMemoryMap(io, .{
        .len = @intCast(parsed.file_size),
        .protection = .{ .read = true, .write = false },
    });
    errdefer mm.destroy(io);

    var kb: [128]u8 = undefined;
    const K = struct {
        fn f(buf: []u8, a: []const u8, comptime s: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}." ++ s, .{a}) catch unreachable;
        }
    };

    const dim: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "embedding_length")) orelse return error.BadConfig);
    const n_heads: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "attention.head_count")) orelse return error.BadConfig);
    if (dim == 0 or n_heads == 0) return error.BadConfig;
    // qwen3 sets an explicit head_dim that is *not* dim/n_heads; fall back to
    // the classic division only when the file stays silent.
    const head_dim: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "attention.key_length")) orelse blk: {
        if (dim % n_heads != 0) return error.BadConfig;
        break :blk dim / n_heads;
    });

    // glm4moe ships its MTP/NextN blocks as ordinary trailing layers that the
    // forward pass must not run. Their tensors stay in the file (and so stay
    // shardable), we simply stop before them.
    const block_count: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "block_count")) orelse return error.BadConfig);
    const nextn: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "nextn_predict_layers")) orelse 0);
    if (nextn >= block_count) return error.BadConfig;

    const n_expert: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "expert_count")) orelse 0);
    const n_used: usize = @intCast(parsed.getUint(K.f(&kb, arch_name, "expert_used_count")) orelse 0);

    var cfg = Config{
        .arch = arch,
        .dim = dim,
        .head_dim = head_dim,
        .n_layers = block_count - nextn,
        .n_dense_layers = @intCast(parsed.getUint(K.f(&kb, arch_name, "leading_dense_block_count")) orelse 0),
        .n_heads = n_heads,
        .n_kv_heads = @intCast(parsed.getUint(K.f(&kb, arch_name, "attention.head_count_kv")) orelse n_heads),
        .ffn = @intCast(parsed.getUint(K.f(&kb, arch_name, "feed_forward_length")) orelse 0),
        .moe_ffn = @intCast(parsed.getUint(K.f(&kb, arch_name, "expert_feed_forward_length")) orelse 0),
        .shexp_ffn = @intCast(parsed.getUint(K.f(&kb, arch_name, "expert_shared_feed_forward_length")) orelse 0),
        .n_expert = n_expert,
        .n_used = n_used,
        .n_shared = @intCast(parsed.getUint(K.f(&kb, arch_name, "expert_shared_count")) orelse 0),
        .route = moe.routeCfgFromMeta(&parsed, arch_name, n_expert, n_used),
        .vocab = 0, // fixed up below from token_embd
        .ctx_len = @intCast(parsed.getUint(K.f(&kb, arch_name, "context_length")) orelse 2048),
        .rope_dim = @intCast(parsed.getUint(K.f(&kb, arch_name, "rope.dimension_count")) orelse head_dim),
        .rope_base = @floatCast(parsed.getFloat(K.f(&kb, arch_name, "rope.freq_base")) orelse 10000.0),
        .swa_window = @intCast(parsed.getUint(K.f(&kb, arch_name, "attention.sliding_window")) orelse 0),
        .eps = @floatCast(parsed.getFloat(K.f(&kb, arch_name, "attention.layer_norm_rms_epsilon")) orelse 1e-5),
    };
    // llama.cpp's fallback when a MoE file omits the per-expert FFN size
    if (cfg.n_expert > 0 and cfg.moe_ffn == 0 and cfg.n_used > 0) cfg.moe_ffn = cfg.ffn / cfg.n_used;
    // qwen2moe's shared expert defaults to the dense FFN width; the others
    // size theirs as n_shared whole routed experts.
    if (cfg.shexp_ffn == 0) {
        cfg.shexp_ffn = if (std.mem.eql(u8, arch_name, "qwen2moe")) cfg.ffn else cfg.n_shared * cfg.moe_ffn;
    }
    try validateConfig(cfg);

    // YaRN: statically rescaled rope frequencies plus a magnitude scale on
    // q and k, folded into the attention logits as mscale^2 (the deepseek
    // engine's trick; ggml applies mscale to the rotated components).
    if (parsed.getString(K.f(&kb, arch_name, "rope.scaling.type"))) |sc| {
        if (std.mem.eql(u8, sc, "yarn")) {
            cfg.yarn_factor = @floatCast(parsed.getFloat(K.f(&kb, arch_name, "rope.scaling.factor")) orelse 1.0);
            cfg.yarn_orig_ctx = @floatCast(parsed.getFloat(K.f(&kb, arch_name, "rope.scaling.original_context_length")) orelse 4096.0);
            if (cfg.yarn_factor > 1.0) {
                const mscale = 1.0 + 0.1 * @log(cfg.yarn_factor);
                cfg.attn_logit_mul = mscale * mscale;
            }
        }
    }

    var model = Model{
        .gpa = gpa,
        .io = io,
        .parsed = parsed,
        .file = file,
        .mm = mm,
        .cfg = cfg,
        .token_embd = undefined,
        .output_norm = undefined,
        .output = undefined,
        .layers = &.{},
        .tok = undefined,
    };

    model.token_embd = try model.resolve("token_embd.weight");
    model.output_norm = try model.resolve("output_norm.weight");
    try Model.expectF32(model.output_norm, "output_norm");
    model.output = model.resolveOpt("output.weight") orelse model.token_embd; // tied embeddings
    model.cfg.vocab = model.token_embd.ne1;
    if (model.cfg.vocab == 0) return error.BadConfig;
    cfg = model.cfg;

    const layers = try gpa.alloc(LayerT, cfg.n_layers);
    errdefer gpa.free(layers);
    var nb: [128]u8 = undefined;
    for (layers, 0..) |*l, i| {
        try loadLayerInto(&model, cfg, l, i, &nb);
    }
    model.layers = layers;

    // NextN/MTP: the trailing block is a full glm4moe-style layer plus the
    // glue that turns (hidden state, next-token embedding) into a one-ahead
    // draft. Loaded best-effort: any missing tensor leaves mtp null and the
    // engine decodes normally.
    if (nextn == 1) blk: {
        var ml: LayerT = undefined;
        loadLayerInto(&model, cfg, &ml, cfg.n_layers, &nb) catch break :blk;
        const M = struct {
            fn f(buf: []u8, li: usize, comptime t: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "blk.{d}.nextn." ++ t, .{li}) catch unreachable;
            }
        };
        const enorm = model.resolveOpt(M.f(&nb, cfg.n_layers, "enorm.weight")) orelse break :blk;
        const hnorm = model.resolveOpt(M.f(&nb, cfg.n_layers, "hnorm.weight")) orelse break :blk;
        const eh_proj = model.resolveOpt(M.f(&nb, cfg.n_layers, "eh_proj.weight")) orelse break :blk;
        const embed = model.resolveOpt(M.f(&nb, cfg.n_layers, "embed_tokens.weight")) orelse break :blk;
        const head_norm = model.resolveOpt(M.f(&nb, cfg.n_layers, "shared_head_norm.weight")) orelse break :blk;
        const head = model.resolveOpt(M.f(&nb, cfg.n_layers, "shared_head_head.weight")) orelse break :blk;
        Model.expectShape(eh_proj, 2 * cfg.dim, cfg.dim, "nextn.eh_proj") catch break :blk;
        Model.expectShape(head, cfg.dim, cfg.vocab, "nextn.shared_head_head") catch break :blk;
        model.mtp = .{
            .layer = ml,
            .enorm = enorm,
            .hnorm = hnorm,
            .eh_proj = eh_proj,
            .embed = embed,
            .head_norm = head_norm,
            .head = head,
        };
        model.cfg.has_mtp = true;
    }
    model.layers = layers;

    model.tok = try Tok.init(gpa, &model.parsed);
    return model;
}

fn loadLayerInto(model: *Model, cfg: Config, l: *LayerT, i: usize, nb: *[128]u8) !void {
    const N2 = struct {
        fn f(buf: []u8, li: usize, comptime t: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "blk.{d}." ++ t, .{li}) catch unreachable;
        }
    };
    defer {
        // gpt-oss extras; resolveOpt leaves them null everywhere else. The
        // sliding-window pattern is llama.cpp's default period 2 with dense
        // layers second: even layers windowed, odd layers full.
        l.attn_sinks = model.resolveOpt(N2.f(nb, i, "attn_sinks.weight"));
        l.attn_output_b = model.resolveOpt(N2.f(nb, i, "attn_output.bias"));
        l.is_swa = cfg.swa_window > 0 and (i % 2 == 0);
        l.ffn_gate_inp_b = model.resolveOpt(N2.f(nb, i, "ffn_gate_inp.bias"));
        l.ffn_gate_exps_b = model.resolveOpt(N2.f(nb, i, "ffn_gate_exps.bias"));
        l.ffn_up_exps_b = model.resolveOpt(N2.f(nb, i, "ffn_up_exps.bias"));
        l.ffn_down_exps_b = model.resolveOpt(N2.f(nb, i, "ffn_down_exps.bias"));
    }
    const N = struct {
        fn f(buf: []u8, li: usize, comptime t: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "blk.{d}." ++ t, .{li}) catch unreachable;
        }
    };
    const hd = cfg.head_dim;

    l.attn_norm = try model.resolve(N.f(nb, i, "attn_norm.weight"));
    try Model.expectF32(l.attn_norm, "attn_norm");
    try Model.expectShape(l.attn_norm, cfg.dim, 1, "attn_norm");
    l.attn_q = try model.resolve(N.f(nb, i, "attn_q.weight"));
    l.attn_k = try model.resolve(N.f(nb, i, "attn_k.weight"));
    l.attn_v = try model.resolve(N.f(nb, i, "attn_v.weight"));
    l.attn_output = try model.resolve(N.f(nb, i, "attn_output.weight"));
    try Model.expectShape(l.attn_q, cfg.dim, cfg.qDim(), "attn_q");
    try Model.expectShape(l.attn_k, cfg.dim, cfg.kvDim(), "attn_k");
    try Model.expectShape(l.attn_v, cfg.dim, cfg.kvDim(), "attn_v");
    try Model.expectShape(l.attn_output, cfg.qDim(), cfg.dim, "attn_output");

    // qwen2moe and glm4moe carry QKV biases; llama and qwen3moe do not.
    // Detect rather than tabulate: the file is the authority.
    l.attn_q_bias = model.resolveOpt(N.f(nb, i, "attn_q.bias"));
    l.attn_k_bias = model.resolveOpt(N.f(nb, i, "attn_k.bias"));
    l.attn_v_bias = model.resolveOpt(N.f(nb, i, "attn_v.bias"));
    if (l.attn_q_bias) |b| {
        try Model.expectF32(b, "attn_q.bias");
        try Model.expectShape(b, cfg.qDim(), 1, "attn_q.bias");
    }
    if (l.attn_k_bias) |b| {
        try Model.expectF32(b, "attn_k.bias");
        try Model.expectShape(b, cfg.kvDim(), 1, "attn_k.bias");
    }
    if (l.attn_v_bias) |b| {
        try Model.expectF32(b, "attn_v.bias");
        try Model.expectShape(b, cfg.kvDim(), 1, "attn_v.bias");
    }

    // qwen3moe always has Q/K norms; glm4moe has them only on the 355B
    // variant. Both are per-head RMSNorm over head_dim.
    l.attn_q_norm = model.resolveOpt(N.f(nb, i, "attn_q_norm.weight"));
    l.attn_k_norm = model.resolveOpt(N.f(nb, i, "attn_k_norm.weight"));
    if (l.attn_q_norm) |n| {
        try Model.expectF32(n, "attn_q_norm");
        try Model.expectShape(n, hd, 1, "attn_q_norm");
    }
    if (l.attn_k_norm) |n| {
        try Model.expectF32(n, "attn_k_norm");
        try Model.expectShape(n, hd, 1, "attn_k_norm");
    }

    // glm4moe has no ffn_norm; its post_attention_norm sits in the same
    // place in the graph, so accept either name.
    l.ffn_norm = model.resolveOpt(N.f(nb, i, "ffn_norm.weight")) orelse
        try model.resolve(N.f(nb, i, "post_attention_norm.weight"));
    try Model.expectF32(l.ffn_norm, "ffn_norm");
    try Model.expectShape(l.ffn_norm, cfg.dim, 1, "ffn_norm");

    l.is_moe = cfg.n_expert > 0 and i >= cfg.n_dense_layers;
    l.ffn_gate = null;
    l.ffn_up = null;
    l.ffn_down = null;
    l.ffn_gate_inp = null;
    l.exp_probs_b = null;
    l.ffn_gate_exps = null;
    l.ffn_up_exps = null;
    l.ffn_down_exps = null;
    l.ffn_gate_shexp = null;
    l.ffn_up_shexp = null;
    l.ffn_down_shexp = null;
    l.ffn_gate_inp_shexp = null;

    if (!l.is_moe) {
        l.ffn_gate = try model.resolve(N.f(nb, i, "ffn_gate.weight"));
        l.ffn_up = try model.resolve(N.f(nb, i, "ffn_up.weight"));
        l.ffn_down = try model.resolve(N.f(nb, i, "ffn_down.weight"));
        try Model.expectShape(l.ffn_gate.?, cfg.dim, cfg.ffn, "ffn_gate");
        try Model.expectShape(l.ffn_up.?, cfg.dim, cfg.ffn, "ffn_up");
        try Model.expectShape(l.ffn_down.?, cfg.ffn, cfg.dim, "ffn_down");
    } else {
        l.ffn_gate_inp = try model.resolve(N.f(nb, i, "ffn_gate_inp.weight"));
        try Model.expectShape(l.ffn_gate_inp.?, cfg.dim, cfg.n_expert, "ffn_gate_inp");
        l.exp_probs_b = model.resolveOpt(N.f(nb, i, "exp_probs_b.bias"));
        if (l.exp_probs_b) |b| {
            try Model.expectF32(b, "exp_probs_b.bias");
            try Model.expectShape(b, cfg.n_expert, 1, "exp_probs_b.bias");
        }
        l.ffn_gate_exps = try model.resolve(N.f(nb, i, "ffn_gate_exps.weight"));
        l.ffn_up_exps = try model.resolve(N.f(nb, i, "ffn_up_exps.weight"));
        l.ffn_down_exps = try model.resolve(N.f(nb, i, "ffn_down_exps.weight"));
        try Model.expectExpertShape(l.ffn_gate_exps.?, cfg.dim, cfg.moe_ffn, cfg.n_expert, "ffn_gate_exps");
        try Model.expectExpertShape(l.ffn_up_exps.?, cfg.dim, cfg.moe_ffn, cfg.n_expert, "ffn_up_exps");
        try Model.expectExpertShape(l.ffn_down_exps.?, cfg.moe_ffn, cfg.dim, cfg.n_expert, "ffn_down_exps");

        // Shared expert: qwen2moe always has one, glm4moe has one when
        // expert_shared_count > 0, qwen3moe and Mixtral have none.
        l.ffn_gate_shexp = model.resolveOpt(N.f(nb, i, "ffn_gate_shexp.weight"));
        if (l.ffn_gate_shexp) |g| {
            l.ffn_up_shexp = try model.resolve(N.f(nb, i, "ffn_up_shexp.weight"));
            l.ffn_down_shexp = try model.resolve(N.f(nb, i, "ffn_down_shexp.weight"));
            try Model.expectShape(g, cfg.dim, cfg.shexp_ffn, "ffn_gate_shexp");
            try Model.expectShape(l.ffn_up_shexp.?, cfg.dim, cfg.shexp_ffn, "ffn_up_shexp");
            try Model.expectShape(l.ffn_down_shexp.?, cfg.shexp_ffn, cfg.dim, "ffn_down_shexp");
            l.ffn_gate_inp_shexp = model.resolveOpt(N.f(nb, i, "ffn_gate_inp_shexp.weight"));
            if (l.ffn_gate_inp_shexp) |s| try Model.expectShape(s, cfg.dim, 1, "ffn_gate_inp_shexp");
        }
    }
}

/// Attach a distributed expert source: routed-expert reads become
/// Source.get() calls (local shard or peer fetch) instead of memory-map reads.
pub fn attachDist(m: *Model, gpa: std.mem.Allocator, src: *expert_fetch.Source) !void {
    m.pilot_enabled = std.c.getenv("LOOM_NO_PILOT") == null;
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
// ---- forward pass ------------------------------------------------------------

pub const Mtp = struct {
    layer: LayerT,
    enorm: Tensor,
    hnorm: Tensor,
    eh_proj: Tensor,
    embed: Tensor,
    head_norm: Tensor,
    head: Tensor,
};

/// Widest verify window the per-lane capture buffers accommodate: DSD draft
/// windows are capped at 8 (openai.zig draft_buf), plus the bonus lane.
pub const VERIFY_LANES = 9;

pub const State = struct {
    cfg: Config,
    /// Draft-local mode (DSD, whitepaper roadmap 6): routed experts are read
    /// from local tiers only; a missing expert is skipped and the surviving
    /// gates are rescaled. The output is an approximation whose quality rises
    /// with the hold fraction -- acceptable only because a warm peer verifies
    /// every drafted token against the exact model.
    draft_local: bool = false,
    // activations
    x: []f32,
    normed: []f32,
    q: []f32, // n_heads * head_dim (not necessarily dim)
    k: []f32,
    v: []f32,
    attn_out: []f32,
    proj_out: []f32,
    gate: []f32,
    up: []f32,
    act: []f32,
    ffn_out: []f32,
    moe_acc: []f32,
    router: []f32,
    scores: []f32,
    logits: []f32,
    // kv cache: [n_layers][ctx][kv_dim]
    k_cache: []f32,
    v_cache: []f32,
    // Batched-prefill scratch: the same activations, MAX_BATCH wide. About a
    // megabyte for a 2B-class model, allocated once per request.
    bx: []f32,
    bnormed: []f32,
    bq: []f32,
    bk: []f32,
    bv: []f32,
    battn: []f32,
    bproj: []f32,
    bgate: []f32,
    bup: []f32,
    bact: []f32,
    bffn: []f32,
    // Batched router logits (MAX_BATCH * n_expert) and a gather buffer
    // (MAX_BATCH * dim) for continuous-batching batch-union MoE: the union of
    // experts across the batch is computed once per expert against the lanes
    // that selected it, so each expert's weights are unpacked once for the
    // whole batch instead of once per lane.
    brouter: []f32,
    bgather: []f32,

    // PILOT lookahead (single-stream decode): the experts predicted for the
    // next MoE layer by running its router on the current hidden state, so
    // their fetches overlap this layer's compute. Checked (for the measured
    // hit rate) when that layer actually routes.
    pilot_layer: usize = std.math.maxInt(usize),
    pilot_pred: [moe.MAX_SELECTED]usize = undefined,
    pilot_n: usize = 0,

    // MTP speculative decode (stepSpec). Scratch is allocated only when the
    // model carries a NextN block; counters feed the tokens-per-forward line.
    mtp_x: []f32 = &.{},
    mtp_cat: []f32 = &.{},
    /// Residuals / logits per verify lane. MTP uses two lanes; DSD batch
    /// verification uses up to VERIFY_LANES (the draft-window cap).
    bhidden: []f32 = &.{},
    blogits: []f32 = &.{}, // logits per verify lane (2 * vocab)
    spec_capture: bool = false,
    spec_fwd: u64 = 0,
    spec_acc: u64 = 0,
    /// Opt-in commitment-weighted verifier expert masking (research lever 8,
    /// AcceptMoE): during a spec_capture batched verify, restrict each MoE
    /// layer's routing to a commitment-weighted eligible expert set. NOT
    /// distribution-preserving for lanes past the first -- lane 0 (the
    /// committed token) always keeps its natural top-k, so its verdict is
    /// unchanged. Off by default; the byte-identity test gates the default.
    verify_mask: bool = false,
    /// Distinct-expert counters for the masked verify, summed per MoE layer:
    /// the natural union across lanes vs the eligible set actually allowed.
    /// The difference is the expert traffic the mask avoided.
    mask_natural: u64 = 0,
    mask_eligible: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !State {
        const kvd = cfg.kvDim();
        const ffn_max = cfg.maxFfn();
        const B = backend.MAX_BATCH;
        return .{
            .cfg = cfg,
            .x = try gpa.alloc(f32, cfg.dim),
            .normed = try gpa.alloc(f32, cfg.dim),
            .q = try gpa.alloc(f32, cfg.qDim()),
            .k = try gpa.alloc(f32, kvd),
            .v = try gpa.alloc(f32, kvd),
            .attn_out = try gpa.alloc(f32, cfg.qDim()),
            .proj_out = try gpa.alloc(f32, cfg.dim),
            .gate = try gpa.alloc(f32, ffn_max),
            .up = try gpa.alloc(f32, ffn_max),
            .act = try gpa.alloc(f32, ffn_max),
            .ffn_out = try gpa.alloc(f32, cfg.dim),
            .moe_acc = try gpa.alloc(f32, cfg.dim),
            .router = try gpa.alloc(f32, @max(cfg.n_expert, 1)),
            .scores = try gpa.alloc(f32, cfg.ctx_len),
            .logits = try gpa.alloc(f32, cfg.vocab),
            .k_cache = try gpa.alloc(f32, (cfg.n_layers + @intFromBool(cfg.has_mtp)) * cfg.ctx_len * kvd),
            .v_cache = try gpa.alloc(f32, (cfg.n_layers + @intFromBool(cfg.has_mtp)) * cfg.ctx_len * kvd),
            .mtp_x = if (cfg.has_mtp) try gpa.alloc(f32, cfg.dim) else &.{},
            .mtp_cat = if (cfg.has_mtp) try gpa.alloc(f32, 2 * cfg.dim) else &.{},
            .bhidden = try gpa.alloc(f32, VERIFY_LANES * cfg.dim),
            .blogits = try gpa.alloc(f32, VERIFY_LANES * cfg.vocab),
            .bx = try gpa.alloc(f32, B * cfg.dim),
            .bnormed = try gpa.alloc(f32, B * cfg.dim),
            .bq = try gpa.alloc(f32, B * cfg.qDim()),
            .bk = try gpa.alloc(f32, B * kvd),
            .bv = try gpa.alloc(f32, B * kvd),
            .battn = try gpa.alloc(f32, B * cfg.qDim()),
            .bproj = try gpa.alloc(f32, B * cfg.dim),
            .bgate = try gpa.alloc(f32, B * ffn_max),
            .bup = try gpa.alloc(f32, B * ffn_max),
            .bact = try gpa.alloc(f32, B * ffn_max),
            .bffn = try gpa.alloc(f32, B * cfg.dim),
            .brouter = try gpa.alloc(f32, B * @max(cfg.n_expert, 1)),
            .bgather = try gpa.alloc(f32, B * cfg.dim),
        };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        inline for (.{
            self.x,        self.normed,   self.q,       self.k,       self.v,
            self.attn_out, self.gate,     self.up,      self.act,     self.ffn_out,
            self.moe_acc,  self.router,   self.scores,  self.logits,  self.k_cache,
            self.v_cache,  self.proj_out, self.bx,      self.bnormed, self.bq,
            self.mtp_x,    self.mtp_cat,  self.bhidden, self.blogits, self.bk,
            self.bv,       self.battn,    self.bproj,   self.bgate,   self.bup,
            self.bact,     self.bffn,     self.brouter, self.bgather,
        }) |sl| gpa.free(sl);
    }
};

/// One concurrent sequence for continuous-batching decode: its own KV cache
/// and position, and nothing else. The batched compute scratch (bx/bq/... in
/// `State`) is shared across all sequences in a batch, because the only part
/// of the forward that depends on a sequence's own history is attention over
/// its KV. Sizing matches `State`'s single-sequence KV cache exactly.
pub const Seq = struct {
    k_cache: []f32,
    v_cache: []f32,
    pos: usize = 0,

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !Seq {
        const kvd = cfg.kvDim();
        const lanes = cfg.n_layers + @intFromBool(cfg.has_mtp);
        return .{
            .k_cache = try gpa.alloc(f32, lanes * cfg.ctx_len * kvd),
            .v_cache = try gpa.alloc(f32, lanes * cfg.ctx_len * kvd),
        };
    }

    pub fn deinit(self: *Seq, gpa: std.mem.Allocator) void {
        gpa.free(self.k_cache);
        gpa.free(self.v_cache);
    }
};

fn mv(t: Tensor, out: []f32, x: []const f32) void {
    backend.matvec(t.ty, out, t.data, x, t.ne1, t.ne0);
}

/// Norm weights are always f32 in practice; view the raw bytes as f32.
fn tensorAsF32(t: Tensor) []const f32 {
    std.debug.assert(t.ty == .f32);
    return @alignCast(std.mem.bytesAsSlice(f32, t.data));
}

/// theta for rotation index `i` (0, 2, 4, ...) of a rope_dim-wide rotation.
inline fn ropeTheta(i: usize, rope_dim: usize, pos: usize, base: f32) struct { c: f32, s: f32 } {
    const exponent = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rope_dim));
    const freq = std.math.pow(f32, base, -exponent);
    const angle = @as(f32, @floatFromInt(pos)) * freq;
    return .{ .c = @cos(angle), .s = @sin(angle) };
}

/// NORM-style RoPE: rotate *adjacent* pairs (2i, 2i+1) of the first rope_dim
/// dims. Used by llama (and deepseek2).
fn ropeNorm(vec: []f32, rope_dim: usize, pos: usize, base: f32) void {
    var i: usize = 0;
    while (i + 1 < rope_dim) : (i += 2) {
        const t = ropeTheta(i, rope_dim, pos, base);
        const a = vec[i];
        const b = vec[i + 1];
        vec[i] = a * t.c - b * t.s;
        vec[i + 1] = a * t.s + b * t.c;
    }
}

/// NEOX-style RoPE: rotate *split-half* pairs (i, i + rope_dim/2). Used by
/// qwen2moe, qwen3moe and glm4moe. Same theta progression as NORM, different
/// pairing — swapping the two produces fluent-looking but wrong output, which
/// is why the style is pinned per architecture rather than guessed.
fn ropeNeox(vec: []f32, rope_dim: usize, pos: usize, base: f32) void {
    const half = rope_dim / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const t = ropeTheta(2 * i, rope_dim, pos, base);
        const a = vec[i];
        const b = vec[i + half];
        vec[i] = a * t.c - b * t.s;
        vec[i + half] = a * t.s + b * t.c;
    }
}

/// NEOX-pair rope with YaRN frequency correction (gpt-oss: factor 32 over a
/// 4096 original context; betas 32/1 as ggml hardcodes for these keys). The
/// magnitude scale (mscale) is folded into the attention logits instead of
/// the vectors -- see Config.attn_logit_mul.
fn ropeNeoxYarn(vec: []f32, rope_dim: usize, pos: usize, base: f32, factor: f32, orig_ctx: f32) void {
    const half = rope_dim / 2;
    const d: f32 = @floatFromInt(rope_dim);
    const two_pi = 2.0 * std.math.pi;
    const corr = struct {
        fn dim(nd: f32, orig: f32, beta: f32, b: f32) f32 {
            return nd * @log(orig / (beta * two_pi)) / (2.0 * @log(b));
        }
    };
    const low = @max(0.0, @floor(corr.dim(d, orig_ctx, 32.0, base)));
    const high = @min(d - 1.0, @ceil(corr.dim(d, orig_ctx, 1.0, base)));
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const exponent = @as(f32, @floatFromInt(2 * i)) / d;
        const freq = std.math.pow(f32, base, -exponent);
        const theta_extrap = @as(f32, @floatFromInt(pos)) * freq;
        const theta_interp = theta_extrap / factor;
        const y = (@as(f32, @floatFromInt(i)) - low) / @max(0.001, high - low);
        const ramp_mix = 1.0 - std.math.clamp(y, 0.0, 1.0); // 1 = extrapolate
        const theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        const c = @cos(theta);
        const sn = @sin(theta);
        const a = vec[i];
        const b2 = vec[i + half];
        vec[i] = a * c - b2 * sn;
        vec[i + half] = a * sn + b2 * c;
    }
}

inline fn ropeApplyC(cfg: Config, vec: []f32, pos: usize) void {
    if (cfg.yarn_factor > 1.0) {
        // yarn arches in this engine are neox-paired (gpt-oss)
        return ropeNeoxYarn(vec, cfg.rope_dim, pos, cfg.rope_base, cfg.yarn_factor, cfg.yarn_orig_ctx);
    }
    ropeApply(cfg.arch.rope, vec, cfg.rope_dim, pos, cfg.rope_base);
}

/// Softmax with a per-head sink logit (gpt-oss): the sink joins the
/// denominator and its probability mass is discarded, so heads can attend
/// to nothing.
fn softmaxWithSink(scores: []f32, sink: f32) void {
    var m = sink;
    for (scores) |v| m = @max(m, v);
    var denom: f32 = @exp(sink - m);
    for (scores) |*v| {
        v.* = @exp(v.* - m);
        denom += v.*;
    }
    for (scores) |*v| v.* /= denom;
}

/// gpt-oss's clamped SwiGLU (ggml swiglu_oai, alpha 1.702, limit 7):
/// out = min(g,7) * sigmoid(1.702 * min(g,7)) * (clamp(u,-7,7) + 1).
fn swigluOai(out: []f32, gate: []const f32, up: []const f32) void {
    const alpha: f32 = 1.702;
    const limit: f32 = 7.0;
    for (out, gate, up) |*o, g, u| {
        const x = @min(g, limit);
        const y = std.math.clamp(u, -limit, limit);
        o.* = (x / (1.0 + @exp(-alpha * x))) * (y + 1.0);
    }
}

inline fn ropeApply(style: RopeStyle, vec: []f32, rope_dim: usize, pos: usize, base: f32) void {
    switch (style) {
        .norm => ropeNorm(vec, rope_dim, pos, base),
        .neox => ropeNeox(vec, rope_dim, pos, base),
    }
}

fn addBias(dst: []f32, b: ?Tensor) void {
    const t = b orelse return;
    for (dst, tensorAsF32(t)) |*d, bv| d.* += bv;
}

/// Per-head RMSNorm over head_dim, applied to q/k before RoPE (qwen3moe
/// always; glm4moe on variants that ship the weights).
fn headNorm(vec: []f32, w: ?Tensor, n_heads: usize, hd: usize, eps: f32) void {
    const t = w orelse return;
    const g = tensorAsF32(t);
    var h: usize = 0;
    while (h < n_heads) : (h += 1) {
        const head = vec[h * hd ..][0..hd];
        var ss: f32 = 0;
        for (head) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(hd)) + eps);
        for (head, g) |*v, gv| v.* = v.* * inv * gv;
    }
}

/// SwiGLU FFN into st.ffn_out over an intermediate width taken from `gate_w`.
fn denseFFN(st: *State, gate_w: Tensor, up_w: Tensor, down_w: Tensor) void {
    const n = gate_w.ne1;
    mv(gate_w, st.gate[0..n], st.normed);
    mv(up_w, st.up[0..n], st.normed);
    backend.swiglu(st.act[0..n], st.gate[0..n], st.up[0..n]);
    mv(down_w, st.ffn_out, st.act[0..n]);
}

/// gpt-oss expert FFN: denseFFN plus the per-expert bias columns (gate/up
/// added before the activation, down after) and swiglu_oai in place of plain
/// SwiGLU. Bias tensors are {width, n_expert}; expert e owns column e.
fn oaiExpertFFN(st: *State, gate_w: Tensor, up_w: Tensor, down_w: Tensor, l: LayerT, e: usize) void {
    const n = gate_w.ne1;
    mv(gate_w, st.gate[0..n], st.normed);
    mv(up_w, st.up[0..n], st.normed);
    if (l.ffn_gate_exps_b) |b| backend.add(st.gate[0..n], tensorAsF32(b)[e * n ..][0..n]);
    if (l.ffn_up_exps_b) |b| backend.add(st.up[0..n], tensorAsF32(b)[e * n ..][0..n]);
    swigluOai(st.act[0..n], st.gate[0..n], st.up[0..n]);
    mv(down_w, st.ffn_out, st.act[0..n]);
    const d = st.ffn_out.len;
    if (l.ffn_down_exps_b) |b| backend.add(st.ffn_out, tensorAsF32(b)[e * d ..][0..d]);
}

/// One token step; logits land in `st.logits`. Errors only when a distributed
/// expert shard has no reachable holder (fail loud, not silently degraded).
/// Whether every layer can be recorded into one command buffer.
///
/// Decided once, at load, rather than per layer -- and that is a correctness
/// requirement, not a convenience. Once recording starts the residual lives
/// only in device memory and the KV cache rows for the layers already recorded
/// have been written on the device; a mid-token fall back to the host path
/// would leave the two caches disagreeing and produce fluent, wrong output.
/// So the whole stack qualifies or none of it does.
pub fn gpuLayersSupported(m: *const Model) bool {
    const c = m.cfg;
    if (c.dim % 256 != 0) return false;
    if (c.n_kv_heads == 0 or c.n_heads % c.n_kv_heads != 0) return false;
    for (m.layers) |l| {
        if (l.is_moe) return false;
        // Bias and per-head q/k norm have no kernel; either would need a host
        // step in the middle of the layer.
        if (l.attn_q_bias != null or l.attn_k_bias != null or l.attn_v_bias != null) return false;
        if (l.attn_q_norm != null or l.attn_k_norm != null) return false;
        const g = l.ffn_gate orelse return false;
        const u = l.ffn_up orelse return false;
        const d = l.ffn_down orelse return false;
        if (g.ne1 % 256 != 0) return false;
        for ([_]ggml.Type{ l.attn_q.ty, l.attn_k.ty, l.attn_v.ty, l.attn_output.ty, g.ty, u.ty, d.ty }) |t| {
            // Matches pipelineFor in the Metal backend. A type without a kernel
            // does not merely run slower, it splits the command buffer.
            if (t != .q4_k and t != .q6_k) return false;
        }
    }
    return true;
}

/// Decide whether the recorded path is worth using, by running the real
/// forward pass both ways and timing it.
///
/// Earlier calibration timed one operation in a tight loop, where successive
/// `commitAndWait` calls pipeline and per-submission latency is largely
/// hidden. In the engine each operation is separated by other work and pays it
/// in full, so that measurement reported the fused FFN block at 1.107 ms
/// against a CPU 9.837 ms -- when the same CPU block measured 0.489 ms
/// elsewhere -- and enabling every GPU path on its advice took decode from 56
/// to 9.1 tok/s. The only measurement that cannot lie about submission cost is
/// the whole token, issued exactly as generation issues it.
pub fn calibrateGpuLayers(m: *Model, gpa: std.mem.Allocator) bool {
    if (!gpuLayersSupported(m)) return false;

    var st = State.init(gpa, m.cfg) catch return false;
    defer st.deinit(gpa);

    const now = @import("../core/stats.zig").nowMonoNs;

    const REPS = 6;

    const timeSteps = struct {
        fn f(mm: *Model, ss: *State, recorded: bool, reps: usize, clock: *const fn () i128) ?i128 {
            const saved = mm.gpu_layers;
            mm.gpu_layers = recorded;
            defer mm.gpu_layers = saved;
            // Warm: first token pays shader warm-up and page faults either way.
            step(mm, ss, 1, 0) catch return null;
            var best: i128 = std.math.maxInt(i64);
            for (0..reps) |i| {
                const t0 = clock();
                step(mm, ss, 1, i + 1) catch return null;
                const dt = clock() - t0;
                if (dt < best) best = dt;
            }
            return best;
        }
    }.f;

    // The recorded path first: it is the one that has to justify itself, and
    // running it against a cold cache would flatter the CPU.
    const g = timeSteps(m, &st, true, REPS, &now) orelse return false;
    const c = timeSteps(m, &st, false, REPS, &now) orelse return false;

    // The same 25% margin the other verdicts use: a wrong choice costs every
    // token of the run, a tie resolved to the CPU costs nothing.
    const win = @as(f64, @floatFromInt(g)) * 1.25 < @as(f64, @floatFromInt(c));
    last_layer_gpu_ns = g;
    last_layer_cpu_ns = c;
    return win;
}

/// Last calibration timings, for the startup banner. Nanoseconds per token.
pub var last_layer_gpu_ns: i128 = 0;
pub var last_layer_cpu_ns: i128 = 0;

/// Record every layer of one token into a single command buffer. Returns false
/// before anything is recorded if the backend declines, so the caller can use
/// its own path; once it returns true, `st.x` holds the result.
fn stepRecorded(m: *const Model, st: *State, pos: usize) bool {
    const cfg = m.cfg;
    const hd = cfg.head_dim;
    const scale = cfg.attn_logit_mul / @sqrt(@as(f32, @floatFromInt(hd)));
    if (!backend.beginFrame()) return false;
    if (!backend.frameLoadX(st.x)) {
        backend.endFrame();
        return false;
    }
    for (m.layers, 0..) |l, li| {
        const ok = backend.layerBlock(.{
            .li = li,
            .pos = pos,
            .attn_norm = tensorAsF32(l.attn_norm),
            .ffn_norm = tensorAsF32(l.ffn_norm),
            .eps = cfg.eps,
            .wq = .{ .ty = l.attn_q.ty, .data = l.attn_q.data },
            .wk = .{ .ty = l.attn_k.ty, .data = l.attn_k.data },
            .wv = .{ .ty = l.attn_v.ty, .data = l.attn_v.data },
            .wo = .{ .ty = l.attn_output.ty, .data = l.attn_output.data },
            .gate = .{ .ty = l.ffn_gate.?.ty, .data = l.ffn_gate.?.data },
            .up = .{ .ty = l.ffn_up.?.ty, .data = l.ffn_up.?.data },
            .down = .{ .ty = l.ffn_down.?.ty, .data = l.ffn_down.?.data },
            .dim = cfg.dim,
            .ffn = l.ffn_gate.?.ne1,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .hd = hd,
            .rope_dim = cfg.rope_dim,
            .rope_base = cfg.rope_base,
            .rope_neox = cfg.arch.rope == .neox,
            .attn_scale = scale,
        });
        if (!ok) {
            // Nothing has been committed, so the device KV cache has not been
            // written and st.x is untouched: dropping the frame is clean.
            backend.endFrame();
            return false;
        }
    }
    backend.endFrame();
    return backend.frameStoreX(st.x);
}

pub fn step(m: *const Model, st: *State, token: u32, pos: usize) !void {
    const cfg = m.cfg;
    const hd = cfg.head_dim;
    const kvd = cfg.kvDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = cfg.attn_logit_mul / @sqrt(@as(f32, @floatFromInt(hd)));

    // A token id indexes token_embd rows directly. Ids come from the
    // tokenizer, whose ceiling is an independent metadata array, so bound it
    // here rather than trusting the two to agree (security issue #29).
    if (token >= cfg.vocab) return error.TokenOutOfRange;
    backend.dequantRow(m.token_embd.ty, st.x, m.token_embd.data, token, cfg.dim);

    // The whole layer stack in one command buffer, when the backend can take
    // it. Everything after this point is the per-operation path.
    if (m.gpu_layers and stepRecorded(m, st, pos)) {
        backend.rmsnorm(st.normed, st.x, tensorAsF32(m.output_norm), cfg.eps);
        mv(m.output, st.logits, st.normed);
        return;
    }

    for (m.layers, 0..) |l, li| {
        // ---- attention ----
        backend.rmsnorm(st.normed, st.x, tensorAsF32(l.attn_norm), cfg.eps);
        mv(l.attn_q, st.q, st.normed);
        mv(l.attn_k, st.k, st.normed);
        mv(l.attn_v, st.v, st.normed);
        addBias(st.q, l.attn_q_bias);
        addBias(st.k, l.attn_k_bias);
        addBias(st.v, l.attn_v_bias);

        // Q/K norm comes *before* RoPE (llama.cpp build_norm then rope_ext).
        headNorm(st.q, l.attn_q_norm, cfg.n_heads, hd, cfg.eps);
        headNorm(st.k, l.attn_k_norm, cfg.n_kv_heads, hd, cfg.eps);

        var h: usize = 0;
        while (h < cfg.n_heads) : (h += 1) ropeApplyC(cfg, st.q[h * hd ..][0..hd], pos);
        h = 0;
        while (h < cfg.n_kv_heads) : (h += 1) ropeApplyC(cfg, st.k[h * hd ..][0..hd], pos);

        // append to cache
        const cache_base = (li * cfg.ctx_len + pos) * kvd;
        @memcpy(st.k_cache[cache_base..][0..kvd], st.k);
        @memcpy(st.v_cache[cache_base..][0..kvd], st.v);
        _ = backend.kvAppend(li, pos, st.k, st.v);

        // per-head attention over positions t0..=pos (t0 > 0 only on
        // sliding-window layers, gpt-oss even layers)
        const seq = pos + 1;
        const t0: usize = if (l.is_swa and cfg.swa_window > 0) seq -| cfg.swa_window else 0;
        // Offer the whole head loop to the backend first -- but only for
        // layers the device kernels can express: no window, no sinks.
        const plain = t0 == 0 and l.attn_sinks == null;
        if (!(plain and backend.attnHeads(li, pos, st.q, st.attn_out, cfg.n_heads, cfg.n_kv_heads, hd, scale))) {
            h = 0;
            while (h < cfg.n_heads) : (h += 1) {
                const kvh = h / q_per_kv;
                const qh = st.q[h * hd ..][0..hd];
                var t_i: usize = t0;
                while (t_i < seq) : (t_i += 1) {
                    const kt = st.k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    st.scores[t_i] = backend.dotF32(qh, kt) * scale;
                }
                if (l.attn_sinks) |sk| {
                    softmaxWithSink(st.scores[t0..seq], tensorAsF32(sk)[h]);
                } else {
                    backend.softmax(st.scores[t0..seq]);
                }
                const oh = st.attn_out[h * hd ..][0..hd];
                @memset(oh, 0);
                t_i = t0;
                while (t_i < seq) : (t_i += 1) {
                    const vt = st.v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    backend.axpy(oh, vt, st.scores[t_i]);
                }
            }
        }
        mv(l.attn_output, st.proj_out, st.attn_out);
        addBias(st.proj_out, l.attn_output_b);
        backend.add(st.x, st.proj_out);

        // ---- FFN ----
        if (!l.is_moe) {
            // Offer the backend the whole block before falling back to the
            // pieces. A GPU pays one command buffer for the block instead of
            // one per matvec, and on an M5 that buffer costs ~262 us against
            // ~18 us of kernel -- so which of the two shapes the engine hands
            // over matters more than any kernel in it.
            const g = l.ffn_gate.?;
            if (backend.ffnBlock(
                st.x,
                tensorAsF32(l.ffn_norm),
                cfg.eps,
                .{ .ty = g.ty, .data = g.data },
                .{ .ty = l.ffn_up.?.ty, .data = l.ffn_up.?.data },
                .{ .ty = l.ffn_down.?.ty, .data = l.ffn_down.?.data },
                g.ne1,
            )) continue;
            backend.rmsnorm(st.normed, st.x, tensorAsF32(l.ffn_norm), cfg.eps);
            denseFFN(st, g, l.ffn_up.?, l.ffn_down.?);
            backend.add(st.x, st.ffn_out);
            continue;
        }
        backend.rmsnorm(st.normed, st.x, tensorAsF32(l.ffn_norm), cfg.eps);

        try moeLayer(m, st, l, li, true);
        backend.add(st.x, st.moe_acc);
    }

    backend.rmsnorm(st.normed, st.x, tensorAsF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
}

/// One mixture-of-experts FFN layer: route `st.normed`, run the selected
/// experts (fetching any this node does not hold), add the shared expert, and
/// leave the result in `st.moe_acc`.
///
/// Split out of `step` so the batched prefill path can reuse it per token --
/// tokens in a batch generally select different experts, so unlike the dense
/// projections there is nothing to share across the batch.
fn moeLayer(m: *const Model, st: *State, l: LayerT, li: usize, dist_ok: bool) !void {
    const cfg = m.cfg;
    mv(l.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
    addBias(st.router[0..cfg.n_expert], l.ffn_gate_inp_b);
    var sel_buf: [moe.MAX_SELECTED]moe.Selected = undefined;
    const sel = sel_buf[0..cfg.n_used];
    const bias: ?[]const f32 = if (l.exp_probs_b) |b| tensorAsF32(b) else null;
    moe.route(cfg.route, st.router[0..cfg.n_expert], bias, sel);
    try moeRun(m, st, l, li, dist_ok, sel);
}

/// Everything moeLayer does after routing: PILOT bookkeeping/prefetch, the
/// expert fetch+FFN loop, the draft-local rescale, and the shared expert.
/// Split out so the masked verify path can supply a *restricted* selection
/// while sharing the fetch/compute body verbatim.
fn moeRun(m: *const Model, st: *State, l: LayerT, li: usize, dist_ok: bool, sel: []const moe.Selected) !void {
    const cfg = m.cfg;
    const acc = st.moe_acc;
    @memset(acc, 0);
    if (if (dist_ok and !st.draft_local) m.dist else null) |src| {
        // PILOT scorecard: how many of the experts predicted for this layer
        // (from the previous layer's hidden state) did the router confirm?
        if (st.pilot_layer == li) {
            src.stats.pilot_pred += st.pilot_n;
            for (sel) |s| {
                for (st.pilot_pred[0..st.pilot_n]) |p| {
                    if (p == s.expert) {
                        src.stats.pilot_hit += 1;
                        break;
                    }
                }
            }
            st.pilot_layer = std.math.maxInt(usize);
        }

        // warm the missing shards in parallel: per-layer miss latency
        // becomes max(fetch), not sum(fetch)
        var ids_buf: [moe.MAX_SELECTED]usize = undefined;
        for (sel, 0..) |s, k| ids_buf[k] = m.expert_shard[li * cfg.n_expert + s.expert];
        src.prefetch(ids_buf[0..sel.len]);

        // PILOT lookahead (colibri's router-lookahead, Apache-2.0): run the
        // NEXT MoE layer's router on the CURRENT hidden state and start
        // fetching its likely experts while this layer computes. Routing is
        // strongly correlated across adjacent layers (colibri measures 71.6%
        // one layer ahead); a miss costs nothing but a wasted fetch that
        // still persists to the store.
        if (li + 1 < m.layers.len and m.pilot_enabled) {
            const nl = m.layers[li + 1];
            if (nl.is_moe and nl.ffn_gate_inp != null) {
                mv(nl.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
                addBias(st.router[0..cfg.n_expert], nl.ffn_gate_inp_b);
                var psel_buf: [moe.MAX_SELECTED]moe.Selected = undefined;
                const psel = psel_buf[0..cfg.n_used];
                const pbias: ?[]const f32 = if (nl.exp_probs_b) |b| tensorAsF32(b) else null;
                moe.route(cfg.route, st.router[0..cfg.n_expert], pbias, psel);
                var pids_buf: [moe.MAX_SELECTED]usize = undefined;
                for (psel, 0..) |s, k| {
                    pids_buf[k] = m.expert_shard[(li + 1) * cfg.n_expert + s.expert];
                    st.pilot_pred[k] = s.expert;
                }
                st.pilot_n = psel.len;
                st.pilot_layer = li + 1;
                src.prefetchAsync(pids_buf[0..psel.len]);
            }
        }
    }
    // Draft-local bookkeeping: how much gate mass was actually applied vs
    // selected, so the accumulator can be rescaled to full-model magnitude.
    var gate_all: f32 = 0;
    var gate_used: f32 = 0;
    for (sel) |s| {
        gate_all += s.gate;
        if (if (dist_ok) m.dist else null) |src| {
            // One shard carries this expert's [gate, up, down] slices
            // back to back, in that order (p2p/weights.zig).
            const blk = if (st.draft_local)
                (src.getLocal(m.expert_shard[li * cfg.n_expert + s.expert]) orelse continue)
            else
                try src.get(m.expert_shard[li * cfg.n_expert + s.expert]);
            const gt = l.ffn_gate_exps.?;
            const ut = l.ffn_up_exps.?;
            const dt = l.ffn_down_exps.?;
            const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
            const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
            const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
            const ge = Tensor{ .ty = gt.ty, .data = blk[0..gl], .ne0 = gt.ne0, .ne1 = gt.ne1 };
            const ue = Tensor{ .ty = ut.ty, .data = blk[gl..][0..ul], .ne0 = ut.ne0, .ne1 = ut.ne1 };
            const de = Tensor{ .ty = dt.ty, .data = blk[gl + ul ..][0..dl], .ne0 = dt.ne0, .ne1 = dt.ne1 };
            if (cfg.arch.oai) {
                oaiExpertFFN(st, ge, ue, de, l, s.expert);
            } else {
                denseFFN(st, ge, ue, de);
            }
        } else {
            const ge = try l.ffn_gate_exps.?.expert(s.expert);
            const ue = try l.ffn_up_exps.?.expert(s.expert);
            const de = try l.ffn_down_exps.?.expert(s.expert);
            if (cfg.arch.oai) {
                oaiExpertFFN(st, ge, ue, de, l, s.expert);
            } else {
                denseFFN(st, ge, ue, de);
            }
        }
        gate_used += s.gate;
        for (acc, st.ffn_out) |*a, v| a.* += s.gate * v;
    }
    // Rescale a partial draft-mode mixture back to full gate mass. With no
    // held expert at all the layer contributes nothing and the residual
    // carries the token -- degraded, but the verifier catches the damage.
    if (st.draft_local and gate_used > 0 and gate_used < gate_all) {
        const boost = gate_all / gate_used;
        for (acc) |*a| a.* *= boost;
    }
    if (l.ffn_gate_shexp) |gs| {
        denseFFN(st, gs, l.ffn_up_shexp.?, l.ffn_down_shexp.?);
        if (l.ffn_gate_inp_shexp) |sg| {
            // qwen2moe scales the shared expert by sigmoid(w . x). The
            // others add it unweighted.
            var logit: [1]f32 = undefined;
            mv(sg, &logit, st.normed);
            const g = backend.sigmoid(logit[0]);
            for (acc, st.ffn_out) |*a, v| a.* += g * v;
        } else {
            backend.add(acc, st.ffn_out);
        }
    }
}

/// Commitment-weighted verifier expert masking for one MoE layer of a DSD
/// verify window (research lever 8, AcceptMoE, arXiv:2608.02989). The window
/// is a CHAIN: lane d is committed only if every earlier lane's draft is
/// accepted, so its routing demand is discounted by BETA^d, with BETA set
/// from the devnet's measured draft acceptance. The eligible set is lane 0's
/// natural top-k (the anchor: that token's verdict must not change) plus the
/// top effective-rank experts by discounted demand; on the distributed path,
/// low-demand NONRESIDENT members are pruned under the rerouting budget the
/// mask has already incurred, so eligibility follows the cache the way the
/// fetch tiers do. Every lane then re-routes top-k over the masked logits.
/// NOT distribution-preserving for lanes past the first -- which is why this
/// runs only behind the opt-in flag.
const VERIFY_MASK_BETA: f32 = 0.6; // measured devnet draft acceptance ~0.62
fn moeVerifyMasked(m: *const Model, st: *State, l: LayerT, li: usize, n: usize) !void {
    const cfg = m.cfg;
    const ne = cfg.n_expert;
    const bias: ?[]const f32 = if (l.exp_probs_b) |b| tensorAsF32(b) else null;

    // Pass 1: natural routing for every lane, logits kept for the re-route.
    var sels: [backend.MAX_BATCH][moe.MAX_SELECTED]moe.Selected = undefined;
    for (0..n) |k| {
        const logits = st.brouter[k * ne ..][0..ne];
        mv(l.ffn_gate_inp.?, logits, st.bnormed[k * cfg.dim ..][0..cfg.dim]);
        addBias(logits, l.ffn_gate_inp_b);
        moe.route(cfg.route, logits, bias, sels[k][0..cfg.n_used]);
    }

    // Discounted demand per expert, natural-selection counts, and the union.
    var util = [_]f32{0} ** moe.MAX_EXPERTS;
    var nat_count = [_]u16{0} ** moe.MAX_EXPERTS;
    var in_union = [_]bool{false} ** moe.MAX_EXPERTS;
    var union_n: usize = 0;
    var w: f32 = 1.0;
    for (0..n) |k| {
        for (sels[k][0..cfg.n_used]) |sl| {
            util[sl.expert] += w * sl.gate;
            nat_count[sl.expert] += 1;
            if (!in_union[sl.expert]) {
                in_union[sl.expert] = true;
                union_n += 1;
            }
        }
        w *= VERIFY_MASK_BETA;
    }

    // Anchor: lane 0's natural top-k, always eligible.
    var eligible = [_]bool{false} ** moe.MAX_EXPERTS;
    for (sels[0][0..cfg.n_used]) |sl| eligible[sl.expert] = true;

    // Self-sizing: keep the top effective-rank non-anchor experts by demand,
    // where the effective rank is exp(entropy) of the normalized demand --
    // concentrated demand keeps few, diffuse demand keeps many.
    var cand_buf: [moe.MAX_EXPERTS]usize = undefined;
    var cand_n: usize = 0;
    var z: f32 = 0;
    for (0..ne) |e| {
        if (in_union[e] and !eligible[e]) {
            cand_buf[cand_n] = e;
            cand_n += 1;
            z += util[e];
        }
    }
    var keep: usize = 0;
    if (cand_n > 0 and z > 0) {
        var h: f32 = 0;
        for (cand_buf[0..cand_n]) |e| {
            const q = util[e] / z;
            if (q > 0) h -= q * @log(q);
        }
        keep = @min(cand_n, @as(usize, @intFromFloat(@ceil(@exp(h)))));
    }
    const cands = cand_buf[0..cand_n];
    std.mem.sort(usize, cands, &util, struct {
        fn desc(u: *const [moe.MAX_EXPERTS]f32, a: usize, b: usize) bool {
            return u[a] > u[b];
        }
    }.desc);
    for (cands[0..keep]) |e| eligible[e] = true;
    var elig_n = cfg.n_used + keep;

    // Residency-aware pruning (distributed path only): drop nonresident
    // low-demand non-anchor members, spending at most the rerouting budget
    // the mask has already incurred (natural selections displaced by S), and
    // never shrinking below a full top-k.
    if (if (!st.draft_local) m.dist else null) |src| {
        var budget: usize = 0;
        for (0..ne) |e| {
            if (in_union[e] and !eligible[e]) budget += nat_count[e];
        }
        var i: usize = keep;
        while (i > 0 and elig_n > cfg.n_used) {
            i -= 1;
            const e = cands[i]; // ascending demand from the sorted tail
            if (src.getLocal(m.expert_shard[li * ne + e]) != null) continue;
            if (nat_count[e] > budget) continue;
            eligible[e] = false;
            budget -= nat_count[e];
            elig_n -= 1;
        }
    }
    st.mask_natural += union_n;
    st.mask_eligible += elig_n;

    // Pass 2: re-route every lane over the masked logits and run the shared
    // fetch/compute body with the restricted selection. Lane 0's natural
    // top-k is fully eligible, so its selection, gates and output are
    // byte-identical to the unmasked path.
    for (0..n) |k| {
        const logits = st.brouter[k * ne ..][0..ne];
        for (0..ne) |e| {
            if (!eligible[e]) logits[e] = -1e30;
        }
        var msel_buf: [moe.MAX_SELECTED]moe.Selected = undefined;
        const msel = msel_buf[0..cfg.n_used];
        moe.route(cfg.route, logits, bias, msel);
        @memcpy(st.normed, st.bnormed[k * cfg.dim ..][0..cfg.dim]);
        @memcpy(st.router[0..ne], logits);
        try moeRun(m, st, l, li, true, msel);
        backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.moe_acc);
    }
}

/// Prefill `tokens` in one pass, writing their KV into the cache at
/// `pos_base..`. Only the last token's logits are produced, which is all a
/// prefill needs.
///
/// Decoding reads and unpacks every weight once per token. A prompt is known
/// up front, so the same unpacked weight can serve the whole batch: unpack
/// once, dot N times. That is worth ~2.4x on the projections, and it is why
/// time-to-first-token falls even though decode speed is unchanged.
///
/// Attention is still per token -- it is bound by the KV cache, not by weight
/// reads, so there is nothing to amortize. Routed experts are also still per
/// token, because tokens in a batch generally select different experts and
/// grouping them is a separate change.

// ---- MTP speculative decode (colibri port; Apache-2.0 attribution in the
// decision log). The NextN block drafts the token after next from the pair
// (current residual stream, embedding of the token just chosen); the main
// model verifies the draft in one 2-wide batched forward. Greedy only: at
// temp 0 acceptance is an exact argmax comparison, so MTP-on output is
// token-identical to MTP-off by construction. Host path only: the extra KV
// lane lives past the backend's device cache, and the block is 1/47th of
// the model.

/// Run the NextN transformer layer at `mtp_pos` on `st.mtp_x`, writing the
/// extra KV lane. Attention is host-side over that lane.
fn mtpLayerForward(m: *const Model, st: *State, mtp_pos: usize) !void {
    const cfg = m.cfg;
    const mtp = m.mtp.?;
    const l = mtp.layer;
    const hd = cfg.head_dim;
    const kvd = cfg.kvDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = cfg.attn_logit_mul / @sqrt(@as(f32, @floatFromInt(hd)));
    const lane = cfg.n_layers; // the extra KV lane

    backend.rmsnorm(st.normed, st.mtp_x, tensorAsF32(l.attn_norm), cfg.eps);
    mv(l.attn_q, st.q, st.normed);
    mv(l.attn_k, st.k, st.normed);
    mv(l.attn_v, st.v, st.normed);
    addBias(st.q, l.attn_q_bias);
    addBias(st.k, l.attn_k_bias);
    addBias(st.v, l.attn_v_bias);
    headNorm(st.q, l.attn_q_norm, cfg.n_heads, hd, cfg.eps);
    headNorm(st.k, l.attn_k_norm, cfg.n_kv_heads, hd, cfg.eps);
    var h: usize = 0;
    while (h < cfg.n_heads) : (h += 1) ropeApplyC(cfg, st.q[h * hd ..][0..hd], mtp_pos);
    h = 0;
    while (h < cfg.n_kv_heads) : (h += 1) ropeApplyC(cfg, st.k[h * hd ..][0..hd], mtp_pos);

    const cache_base = (lane * cfg.ctx_len + mtp_pos) * kvd;
    @memcpy(st.k_cache[cache_base..][0..kvd], st.k);
    @memcpy(st.v_cache[cache_base..][0..kvd], st.v);

    const seq = mtp_pos + 1;
    h = 0;
    while (h < cfg.n_heads) : (h += 1) {
        const kvh = h / q_per_kv;
        const qh = st.q[h * hd ..][0..hd];
        var t_i: usize = 0;
        while (t_i < seq) : (t_i += 1) {
            const kt = st.k_cache[(lane * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
            st.scores[t_i] = backend.dotF32(qh, kt) * scale;
        }
        backend.softmax(st.scores[0..seq]);
        const oh = st.attn_out[h * hd ..][0..hd];
        @memset(oh, 0);
        t_i = 0;
        while (t_i < seq) : (t_i += 1) {
            const vt = st.v_cache[(lane * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
            backend.axpy(oh, vt, st.scores[t_i]);
        }
    }
    mv(l.attn_output, st.proj_out, st.attn_out);
    addBias(st.proj_out, l.attn_output_b);
    backend.add(st.mtp_x, st.proj_out);

    backend.rmsnorm(st.normed, st.mtp_x, tensorAsF32(l.ffn_norm), cfg.eps);
    try moeLayer(m, st, l, cfg.n_layers, false);
    backend.add(st.mtp_x, st.moe_acc);
}

/// The NextN glue: mtp_x = eh_proj([enorm(embed(tok)); hnorm(h)]), then the
/// layer forward at `mtp_pos`. `h` is the main model's residual stream that
/// chose `tok` (read-only here).
fn mtpGlue(m: *const Model, st: *State, h: []const f32, tok: u32, mtp_pos: usize) !void {
    const cfg = m.cfg;
    const mtp = m.mtp.?;
    if (tok >= cfg.vocab) return error.TokenOutOfRange;
    backend.dequantRow(mtp.embed.ty, st.mtp_cat[0..cfg.dim], mtp.embed.data, tok, cfg.dim);
    rmsnormInPlace(st.mtp_cat[0..cfg.dim], tensorAsF32(mtp.enorm), cfg.eps);
    @memcpy(st.mtp_cat[cfg.dim..][0..cfg.dim], h);
    rmsnormInPlace(st.mtp_cat[cfg.dim..][0..cfg.dim], tensorAsF32(mtp.hnorm), cfg.eps);
    mv(mtp.eh_proj, st.mtp_x, st.mtp_cat);
    try mtpLayerForward(m, st, mtp_pos);
}

fn rmsnormInPlace(v: []f32, w: []const f32, eps: f32) void {
    var tmp_buf: [16384]f32 = undefined;
    const tmp = tmp_buf[0..v.len];
    backend.rmsnorm(tmp, v, w, eps);
    @memcpy(v, tmp);
}

fn argmaxSlice(v: []const f32) u32 {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (x > v[best]) best = i;
    }
    return @intCast(best);
}

pub const SpecOut = struct {
    n: u8, // tokens advanced: 1 (draft rejected) or 2 (accepted)
    tok2: u32, // the accepted draft, when n == 2
    next: u32, // greedy choice after the last advanced position
};

/// One greedy speculative iteration. `cur` is the token chosen for `pos`
/// (not yet forwarded); st.x must hold the residual stream that chose it.
/// On return st.x/st.logits describe the last advanced position, the KV is
/// current through it, and `next` is the greedy continuation.
pub fn stepSpec(m: *const Model, st: *State, cur: u32, pos: usize) !SpecOut {
    const cfg = m.cfg;
    if (pos + 1 >= cfg.ctx_len) { // no room to verify a draft: plain step
        try step(m, st, cur, pos);
        return .{ .n = 1, .tok2 = 0, .next = argmaxSlice(st.logits) };
    }

    // Draft the token after `cur` (MTP position pos-1: aligned with the
    // residual stream that produced cur).
    try mtpGlue(m, st, st.x, cur, pos -| 1);
    backend.rmsnorm(st.normed, st.mtp_x, tensorAsF32(m.mtp.?.head_norm), cfg.eps);
    mv(m.mtp.?.head, st.logits, st.normed);
    const draft = argmaxSlice(st.logits);

    st.spec_capture = true;
    defer st.spec_capture = false;
    try stepBatch(m, st, &.{ cur, draft }, pos);
    st.spec_fwd += 1;

    const main1 = argmaxSlice(st.blogits[0..cfg.vocab]);
    if (main1 == draft) {
        st.spec_acc += 1;
        // Backfill the MTP lane for position pos (its inputs are lane 0's
        // residual and the accepted draft), keeping the lane contiguous.
        try mtpGlue(m, st, st.bhidden[0..cfg.dim], draft, pos);
        @memcpy(st.x, st.bhidden[cfg.dim..][0..cfg.dim]);
        @memcpy(st.logits, st.blogits[cfg.vocab..][0..cfg.vocab]);
        return .{ .n = 2, .tok2 = draft, .next = argmaxSlice(st.logits) };
    }
    @memcpy(st.x, st.bhidden[0..cfg.dim]);
    @memcpy(st.logits, st.blogits[0..cfg.vocab]);
    return .{ .n = 1, .tok2 = 0, .next = main1 };
}

pub fn stepBatch(m: *const Model, st: *State, tokens: []const u32, pos_base: usize) !void {
    const n = tokens.len;
    std.debug.assert(n > 0 and n <= backend.MAX_BATCH);
    if (n == 1) return step(m, st, tokens[0], pos_base);

    const cfg = m.cfg;
    const hd = cfg.head_dim;
    const kvd = cfg.kvDim();
    const qd = cfg.qDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = cfg.attn_logit_mul / @sqrt(@as(f32, @floatFromInt(hd)));

    for (tokens, 0..) |tok, k| {
        if (tok >= cfg.vocab) return error.TokenOutOfRange;
        if (pos_base + k >= cfg.ctx_len) return error.ContextExhausted;
        backend.dequantRow(m.token_embd.ty, st.bx[k * cfg.dim ..][0..cfg.dim], m.token_embd.data, tok, cfg.dim);
    }

    for (m.layers, 0..) |l, li| {
        // ---- attention ----
        for (0..n) |k| {
            backend.rmsnorm(st.bnormed[k * cfg.dim ..][0..cfg.dim], st.bx[k * cfg.dim ..][0..cfg.dim], tensorAsF32(l.attn_norm), cfg.eps);
        }
        mmul(l.attn_q, st.bq[0 .. n * qd], st.bnormed[0 .. n * cfg.dim], n);
        mmul(l.attn_k, st.bk[0 .. n * kvd], st.bnormed[0 .. n * cfg.dim], n);
        mmul(l.attn_v, st.bv[0 .. n * kvd], st.bnormed[0 .. n * cfg.dim], n);

        for (0..n) |k| {
            const qk = st.bq[k * qd ..][0..qd];
            const kk = st.bk[k * kvd ..][0..kvd];
            const vk = st.bv[k * kvd ..][0..kvd];
            addBias(qk, l.attn_q_bias);
            addBias(kk, l.attn_k_bias);
            addBias(vk, l.attn_v_bias);
            headNorm(qk, l.attn_q_norm, cfg.n_heads, hd, cfg.eps);
            headNorm(kk, l.attn_k_norm, cfg.n_kv_heads, hd, cfg.eps);
            const pos = pos_base + k;
            for (0..cfg.n_heads) |h| ropeApplyC(cfg, qk[h * hd ..][0..hd], pos);
            for (0..cfg.n_kv_heads) |h| ropeApplyC(cfg, kk[h * hd ..][0..hd], pos);
            const base = (li * cfg.ctx_len + pos) * kvd;
            @memcpy(st.k_cache[base..][0..kvd], kk);
            @memcpy(st.v_cache[base..][0..kvd], vk);
            _ = backend.kvAppend(li, pos_base + k, kk, vk);
        }

        // Causal: token k sees positions 0..pos_base+k only, even though the
        // whole batch is already in the cache.
        for (0..n) |k| {
            const seq = pos_base + k + 1;
            const t0: usize = if (l.is_swa and cfg.swa_window > 0) seq -| cfg.swa_window else 0;
            const qk = st.bq[k * qd ..][0..qd];
            const ok = st.battn[k * qd ..][0..qd];
            for (0..cfg.n_heads) |h| {
                const kvh = h / q_per_kv;
                const qh = qk[h * hd ..][0..hd];
                for (t0..seq) |t_i| {
                    const kt = st.k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    st.scores[t_i] = backend.dotF32(qh, kt) * scale;
                }
                if (l.attn_sinks) |sk| {
                    softmaxWithSink(st.scores[t0..seq], tensorAsF32(sk)[h]);
                } else {
                    backend.softmax(st.scores[t0..seq]);
                }
                const oh = ok[h * hd ..][0..hd];
                @memset(oh, 0);
                for (t0..seq) |t_i| {
                    const vt = st.v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    backend.axpy(oh, vt, st.scores[t_i]);
                }
            }
        }
        mmul(l.attn_output, st.bproj[0 .. n * cfg.dim], st.battn[0 .. n * qd], n);
        for (0..n) |k| {
            const row = st.bproj[k * cfg.dim ..][0..cfg.dim];
            addBias(row, l.attn_output_b);
            backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], row);
        }

        // ---- FFN ----
        for (0..n) |k| {
            backend.rmsnorm(st.bnormed[k * cfg.dim ..][0..cfg.dim], st.bx[k * cfg.dim ..][0..cfg.dim], tensorAsF32(l.ffn_norm), cfg.eps);
        }
        if (!l.is_moe) {
            const f = l.ffn_gate.?.ne1;
            mmul(l.ffn_gate.?, st.bgate[0 .. n * f], st.bnormed[0 .. n * cfg.dim], n);
            mmul(l.ffn_up.?, st.bup[0 .. n * f], st.bnormed[0 .. n * cfg.dim], n);
            for (0..n) |k| backend.swiglu(st.bact[k * f ..][0..f], st.bgate[k * f ..][0..f], st.bup[k * f ..][0..f]);
            mmul(l.ffn_down.?, st.bffn[0 .. n * cfg.dim], st.bact[0 .. n * f], n);
            for (0..n) |k| backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.bffn[k * cfg.dim ..][0..cfg.dim]);
            continue;
        }
        // MoE: routing differs per token, so run the expert FFN one token at a
        // time through the existing single-token path -- unless the opt-in
        // verifier expert mask is on, which restricts routing to a
        // commitment-weighted eligible set shared by the whole window.
        if (st.verify_mask and st.spec_capture) {
            try moeVerifyMasked(m, st, l, li, n);
        } else for (0..n) |k| {
            @memcpy(st.normed, st.bnormed[k * cfg.dim ..][0..cfg.dim]);
            try moeLayer(m, st, l, li, true);
            backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.moe_acc);
        }
    }

    // Speculative verify wants the residual and logits of EVERY position,
    // not just the last: acceptance compares the model's choice after lane k
    // with the draft in lane k+1, and the surviving lane's hidden state
    // seeds the next draft.
    if (st.spec_capture) {
        for (0..n) |k| {
            const lane = st.bx[k * cfg.dim ..][0..cfg.dim];
            @memcpy(st.bhidden[k * cfg.dim ..][0..cfg.dim], lane);
            backend.rmsnorm(st.normed, lane, tensorAsF32(m.output_norm), cfg.eps);
            mv(m.output, st.blogits[k * cfg.vocab ..][0..cfg.vocab], st.normed);
        }
    }

    // Only the final token's logits matter for sampling.
    const last = st.bx[(n - 1) * cfg.dim ..][0..cfg.dim];
    backend.rmsnorm(st.normed, last, tensorAsF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
}

/// Batch-union MoE for continuous-batching decode. Routes all `n` lanes from
/// `bnormed`, then walks the union of experts across the batch: each expert's
/// weights are unpacked ONCE and its FFN runs as a mini-matmul over exactly the
/// lanes that selected it (the unpack, not the dot, is the expensive half, so
/// this is where batching a MoE pays off). Gated results are added into
/// `bx[k]`. The shared expert is a plain dense FFN batched across all lanes.
fn moeBatchUnion(m: *const Model, st: *State, l: LayerT, li: usize, n: usize, dist_ok: bool) !void {
    const cfg = m.cfg;
    const dim = cfg.dim;
    const ne = cfg.n_expert;

    mmul(l.ffn_gate_inp.?, st.brouter[0 .. n * ne], st.bnormed[0 .. n * dim], n);
    var sel: [backend.MAX_BATCH][moe.MAX_SELECTED]moe.Selected = undefined;
    const bias: ?[]const f32 = if (l.exp_probs_b) |b| tensorAsF32(b) else null;
    for (0..n) |k| {
        const logits = st.brouter[k * ne ..][0..ne];
        addBias(logits, l.ffn_gate_inp_b);
        moe.route(cfg.route, logits, bias, sel[k][0..cfg.n_used]);
    }

    const src = if (dist_ok) m.dist else null;
    for (0..ne) |e| {
        // Gather the lanes that selected expert e, with their gate weights.
        var lanes: [backend.MAX_BATCH]usize = undefined;
        var gates: [backend.MAX_BATCH]f32 = undefined;
        var mrows: usize = 0;
        for (0..n) |k| {
            for (sel[k][0..cfg.n_used]) |s| if (s.expert == e) {
                lanes[mrows] = k;
                gates[mrows] = s.gate;
                mrows += 1;
                break;
            };
        }
        if (mrows == 0) continue;

        // Resolve this expert's [gate, up, down] once for the whole batch.
        var ge: Tensor = undefined;
        var ue: Tensor = undefined;
        var de: Tensor = undefined;
        if (src) |sp| {
            const blk = try sp.get(m.expert_shard[li * ne + e]);
            const gt = l.ffn_gate_exps.?;
            const ut = l.ffn_up_exps.?;
            const dt = l.ffn_down_exps.?;
            const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
            const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
            const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
            ge = .{ .ty = gt.ty, .data = blk[0..gl], .ne0 = gt.ne0, .ne1 = gt.ne1 };
            ue = .{ .ty = ut.ty, .data = blk[gl..][0..ul], .ne0 = ut.ne0, .ne1 = ut.ne1 };
            de = .{ .ty = dt.ty, .data = blk[gl + ul ..][0..dl], .ne0 = dt.ne0, .ne1 = dt.ne1 };
        } else {
            ge = try l.ffn_gate_exps.?.expert(e);
            ue = try l.ffn_up_exps.?.expert(e);
            de = try l.ffn_down_exps.?.expert(e);
        }
        const f = ge.ne1;

        for (0..mrows) |j| @memcpy(st.bgather[j * dim ..][0..dim], st.bnormed[lanes[j] * dim ..][0..dim]);
        mmul(ge, st.bgate[0 .. mrows * f], st.bgather[0 .. mrows * dim], mrows);
        mmul(ue, st.bup[0 .. mrows * f], st.bgather[0 .. mrows * dim], mrows);
        for (0..mrows) |j| {
            const gbuf = st.bgate[j * f ..][0..f];
            const ubuf = st.bup[j * f ..][0..f];
            if (cfg.arch.oai) {
                if (l.ffn_gate_exps_b) |b| backend.add(gbuf, tensorAsF32(b)[e * f ..][0..f]);
                if (l.ffn_up_exps_b) |b| backend.add(ubuf, tensorAsF32(b)[e * f ..][0..f]);
                swigluOai(st.bact[j * f ..][0..f], gbuf, ubuf);
            } else {
                backend.swiglu(st.bact[j * f ..][0..f], gbuf, ubuf);
            }
        }
        mmul(de, st.bffn[0 .. mrows * dim], st.bact[0 .. mrows * f], mrows);
        for (0..mrows) |j| {
            const outp = st.bffn[j * dim ..][0..dim];
            if (cfg.arch.oai) {
                if (l.ffn_down_exps_b) |b| backend.add(outp, tensorAsF32(b)[e * dim ..][0..dim]);
            }
            const g = gates[j];
            for (st.bx[lanes[j] * dim ..][0..dim], outp) |*a, v| a.* += g * v;
        }
    }

    // Shared expert: a dense FFN, batched across every lane.
    if (l.ffn_gate_shexp) |gs| {
        const sf = gs.ne1;
        mmul(gs, st.bgate[0 .. n * sf], st.bnormed[0 .. n * dim], n);
        mmul(l.ffn_up_shexp.?, st.bup[0 .. n * sf], st.bnormed[0 .. n * dim], n);
        for (0..n) |k| backend.swiglu(st.bact[k * sf ..][0..sf], st.bgate[k * sf ..][0..sf], st.bup[k * sf ..][0..sf]);
        mmul(l.ffn_down_shexp.?, st.bffn[0 .. n * dim], st.bact[0 .. n * sf], n);
        for (0..n) |k| {
            const outp = st.bffn[k * dim ..][0..dim];
            if (l.ffn_gate_inp_shexp) |sg| {
                var logit: [1]f32 = undefined;
                mv(sg, &logit, st.bnormed[k * dim ..][0..dim]);
                const gg = backend.sigmoid(logit[0]);
                for (st.bx[k * dim ..][0..dim], outp) |*a, v| a.* += gg * v;
            } else {
                backend.add(st.bx[k * dim ..][0..dim], outp);
            }
        }
    }
}

/// Continuous-batching decode: advance `n` independent sequences by one token
/// each in a single forward pass. Every weight matmul (QKV, output projection,
/// dense FFN, and the per-token MoE) is batched across the sequences, so each
/// weight is read from memory once and dotted `n` times -- the throughput lever
/// for a memory-/dispatch-bound decode. Attention, the only history-dependent
/// part, runs per sequence against that sequence's own KV cache and position.
///
/// This is `stepBatch` with its one shared sequence replaced by `n` independent
/// ones: identical matmul batching, but each lane keeps its own KV and position
/// rather than occupying consecutive positions of a single causal sequence.
/// Per-sequence logits for the just-decoded token land in `out` (n * vocab),
/// and each sequence's `pos` is advanced. Verified token-identical to `n`
/// separate `step` streams.
pub fn decodeBatch(m: *const Model, st: *State, seqs: []Seq, tokens: []const u32, out: []f32) !void {
    const n = tokens.len;
    std.debug.assert(n > 0 and n <= backend.MAX_BATCH and seqs.len == n);
    const cfg = m.cfg;
    std.debug.assert(out.len == n * cfg.vocab);
    const hd = cfg.head_dim;
    const kvd = cfg.kvDim();
    const qd = cfg.qDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = cfg.attn_logit_mul / @sqrt(@as(f32, @floatFromInt(hd)));

    for (tokens, 0..) |tok, k| {
        if (tok >= cfg.vocab) return error.TokenOutOfRange;
        if (seqs[k].pos >= cfg.ctx_len) return error.ContextExhausted;
        backend.dequantRow(m.token_embd.ty, st.bx[k * cfg.dim ..][0..cfg.dim], m.token_embd.data, tok, cfg.dim);
    }

    for (m.layers, 0..) |l, li| {
        // ---- attention ----
        for (0..n) |k| {
            backend.rmsnorm(st.bnormed[k * cfg.dim ..][0..cfg.dim], st.bx[k * cfg.dim ..][0..cfg.dim], tensorAsF32(l.attn_norm), cfg.eps);
        }
        mmul(l.attn_q, st.bq[0 .. n * qd], st.bnormed[0 .. n * cfg.dim], n);
        mmul(l.attn_k, st.bk[0 .. n * kvd], st.bnormed[0 .. n * cfg.dim], n);
        mmul(l.attn_v, st.bv[0 .. n * kvd], st.bnormed[0 .. n * cfg.dim], n);

        for (0..n) |k| {
            const qk = st.bq[k * qd ..][0..qd];
            const kk = st.bk[k * kvd ..][0..kvd];
            const vk = st.bv[k * kvd ..][0..kvd];
            addBias(qk, l.attn_q_bias);
            addBias(kk, l.attn_k_bias);
            addBias(vk, l.attn_v_bias);
            headNorm(qk, l.attn_q_norm, cfg.n_heads, hd, cfg.eps);
            headNorm(kk, l.attn_k_norm, cfg.n_kv_heads, hd, cfg.eps);
            const pos = seqs[k].pos;
            for (0..cfg.n_heads) |h| ropeApplyC(cfg, qk[h * hd ..][0..hd], pos);
            for (0..cfg.n_kv_heads) |h| ropeApplyC(cfg, kk[h * hd ..][0..hd], pos);
            // Each sequence appends into its own cache at its own position.
            const base = (li * cfg.ctx_len + pos) * kvd;
            @memcpy(seqs[k].k_cache[base..][0..kvd], kk);
            @memcpy(seqs[k].v_cache[base..][0..kvd], vk);
        }

        // Attention: lane k attends only its own sequence's 0..pos_k history.
        for (0..n) |k| {
            const pos = seqs[k].pos;
            const seq = pos + 1;
            const t0: usize = if (l.is_swa and cfg.swa_window > 0) seq -| cfg.swa_window else 0;
            const qk = st.bq[k * qd ..][0..qd];
            const ok = st.battn[k * qd ..][0..qd];
            for (0..cfg.n_heads) |h| {
                const kvh = h / q_per_kv;
                const qh = qk[h * hd ..][0..hd];
                for (t0..seq) |t_i| {
                    const kt = seqs[k].k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    st.scores[t_i] = backend.dotF32(qh, kt) * scale;
                }
                if (l.attn_sinks) |sk| {
                    softmaxWithSink(st.scores[t0..seq], tensorAsF32(sk)[h]);
                } else {
                    backend.softmax(st.scores[t0..seq]);
                }
                const oh = ok[h * hd ..][0..hd];
                @memset(oh, 0);
                for (t0..seq) |t_i| {
                    const vt = seqs[k].v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    backend.axpy(oh, vt, st.scores[t_i]);
                }
            }
        }
        mmul(l.attn_output, st.bproj[0 .. n * cfg.dim], st.battn[0 .. n * qd], n);
        for (0..n) |k| {
            const row = st.bproj[k * cfg.dim ..][0..cfg.dim];
            addBias(row, l.attn_output_b);
            backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], row);
        }

        // ---- FFN ----
        for (0..n) |k| {
            backend.rmsnorm(st.bnormed[k * cfg.dim ..][0..cfg.dim], st.bx[k * cfg.dim ..][0..cfg.dim], tensorAsF32(l.ffn_norm), cfg.eps);
        }
        if (!l.is_moe) {
            const f = l.ffn_gate.?.ne1;
            mmul(l.ffn_gate.?, st.bgate[0 .. n * f], st.bnormed[0 .. n * cfg.dim], n);
            mmul(l.ffn_up.?, st.bup[0 .. n * f], st.bnormed[0 .. n * cfg.dim], n);
            for (0..n) |k| backend.swiglu(st.bact[k * f ..][0..f], st.bgate[k * f ..][0..f], st.bup[k * f ..][0..f]);
            mmul(l.ffn_down.?, st.bffn[0 .. n * cfg.dim], st.bact[0 .. n * f], n);
            for (0..n) |k| backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.bffn[k * cfg.dim ..][0..cfg.dim]);
            continue;
        }
        // MoE: each expert unpacked once for the whole batch, run over the
        // lanes that selected it (batch-union). This is where a batched MoE
        // decode actually amortizes -- the per-lane path does B x the expert
        // work and does not.
        try moeBatchUnion(m, st, l, li, n, true);
    }

    for (0..n) |k| {
        const lane = st.bx[k * cfg.dim ..][0..cfg.dim];
        backend.rmsnorm(st.normed, lane, tensorAsF32(m.output_norm), cfg.eps);
        mv(m.output, out[k * cfg.vocab ..][0..cfg.vocab], st.normed);
        seqs[k].pos += 1;
    }
}

fn mmul(t: Tensor, out: []f32, xs: []const f32, n: usize) void {
    backend.matmul(t.ty, out, t.data, xs, n, t.ne1, t.ne0);
}

// ---- tests -------------------------------------------------------------------

test "NEOX and NORM rope rotate different pairs" {
    // The two styles share a theta progression and differ only in pairing.
    // Mixing them up yields plausible-looking but wrong attention, so pin the
    // distinction rather than trusting the name.
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = a;
    ropeNorm(&a, 4, 1, 10000.0);
    ropeNeox(&b, 4, 1, 10000.0);
    try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(&a), std.mem.sliceAsBytes(&b)));

    // NORM mixes lanes (0,1) and (2,3); NEOX mixes (0,2) and (1,3). With
    // theta_0 = 0 the first pair is identity under both.
    var c = [_]f32{ 1, 2, 3, 4 };
    ropeNorm(&c, 4, 0, 10000.0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &c);
}

test "partial rope leaves the tail of a head untouched" {
    // glm4moe rotates only the first rope_dim of each head.
    var v = [_]f32{ 1, 1, 1, 1, 7, 7, 7, 7 };
    ropeNeox(v[0..4], 4, 3, 10000.0);
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 7 }, v[4..8]);
    try std.testing.expect(v[0] != 1);
}

test "head norm normalizes each head independently" {
    const gpa = std.testing.allocator;
    const w = try gpa.alloc(f32, 2);
    defer gpa.free(w);
    @memset(w, 1.0);
    const t = Tensor{ .ty = .f32, .data = std.mem.sliceAsBytes(w), .ne0 = 2, .ne1 = 1 };
    // two heads of width 2, wildly different magnitudes
    var v = [_]f32{ 3, 4, 300, 400 };
    headNorm(&v, t, 2, 2, 0.0);
    // after per-head RMS norm both heads have the same shape
    try std.testing.expectApproxEqAbs(v[0], v[2], 1e-4);
    try std.testing.expectApproxEqAbs(v[1], v[3], 1e-4);
    // and RMS of each head is 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt((v[0] * v[0] + v[1] * v[1]) / 2.0), 1e-4);
}

test "absent optional tensors are no-ops" {
    var v = [_]f32{ 1, 2, 3, 4 };
    addBias(&v, null);
    headNorm(&v, null, 2, 2, 1e-5);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &v);
}

test "every supported arch resolves and pins its rope style" {
    try std.testing.expectEqual(RopeStyle.norm, archFor("llama").?.rope);
    try std.testing.expectEqual(RopeStyle.neox, archFor("qwen2moe").?.rope);
    try std.testing.expectEqual(RopeStyle.neox, archFor("qwen3moe").?.rope);
    try std.testing.expectEqual(RopeStyle.neox, archFor("glm4moe").?.rope);
    try std.testing.expect(archFor("deepseek2") == null); // MLA, handled by deepseek.zig
    try std.testing.expect(archFor("mamba") == null);
}

test "each supported architecture loads with the features it actually declares" {
    // Round-trips the synthetic fixtures through the real loader, so a
    // regression in feature detection (a missing bias, a Q/K norm that stops
    // being found, an off-by-one on the NextN skip) fails here rather than
    // silently degrading output on a real model.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const Case = struct {
        arch: []const u8,
        qkv_bias: bool,
        qk_norm: bool,
        shexp: bool,
        shexp_gate: bool,
        n_dense: usize,
        rope: RopeStyle,
        head_dim: usize,
        n_layers: usize, // after skipping trailing NextN blocks
    };
    const cases = [_]Case{
        .{ .arch = "llama", .qkv_bias = false, .qk_norm = false, .shexp = false, .shexp_gate = false, .n_dense = 0, .rope = .norm, .head_dim = 16, .n_layers = 3 },
        .{ .arch = "qwen2moe", .qkv_bias = true, .qk_norm = false, .shexp = true, .shexp_gate = true, .n_dense = 0, .rope = .neox, .head_dim = 16, .n_layers = 3 },
        .{ .arch = "qwen3moe", .qkv_bias = false, .qk_norm = true, .shexp = false, .shexp_gate = false, .n_dense = 0, .rope = .neox, .head_dim = 24, .n_layers = 3 },
        .{ .arch = "glm4moe", .qkv_bias = true, .qk_norm = true, .shexp = true, .shexp_gate = false, .n_dense = 1, .rope = .neox, .head_dim = 16, .n_layers = 3 },
        .{ .arch = "gpt-oss", .qkv_bias = true, .qk_norm = false, .shexp = false, .shexp_gate = false, .n_dense = 0, .rope = .neox, .head_dim = 16, .n_layers = 3 },
    };

    for (cases) |c| {
        var pbuf: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-arch-{s}.gguf", .{c.arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf.writeMoeFixture(gpa, io, path, 7, c.arch);

        var m = try load(gpa, io, path);
        defer m.deinit();

        try std.testing.expectEqual(c.rope, m.cfg.arch.rope);
        try std.testing.expectEqual(c.head_dim, m.cfg.head_dim);
        // qwen3's head_dim deliberately is *not* dim/n_heads; that is the
        // whole point of reading attention.key_length.
        if (std.mem.eql(u8, c.arch, "qwen3moe")) {
            try std.testing.expect(m.cfg.head_dim != m.cfg.dim / m.cfg.n_heads);
        }
        // NextN/MTP blocks are present in the file but must not be run
        try std.testing.expectEqual(c.n_layers, m.layers.len);
        try std.testing.expectEqual(c.n_dense, m.cfg.n_dense_layers);

        const last = m.layers[m.layers.len - 1];
        try std.testing.expectEqual(c.qkv_bias, last.attn_q_bias != null);
        try std.testing.expectEqual(c.qkv_bias, last.attn_v_bias != null);
        try std.testing.expectEqual(c.qk_norm, last.attn_q_norm != null);
        try std.testing.expectEqual(c.qk_norm, last.attn_k_norm != null);
        try std.testing.expectEqual(c.shexp, last.ffn_gate_shexp != null);
        try std.testing.expectEqual(c.shexp_gate, last.ffn_gate_inp_shexp != null);
        try std.testing.expect(last.is_moe);
        // leading dense layers keep a plain FFN and no router
        if (c.n_dense > 0) {
            try std.testing.expect(!m.layers[0].is_moe);
            try std.testing.expect(m.layers[0].ffn_gate != null);
            try std.testing.expect(m.layers[0].ffn_gate_inp == null);
        }
        // glm4moe alone routes with sigmoid and a selection bias; gpt-oss
        // selects on raw biased logits then softmaxes the selected k
        const is_glm = std.mem.eql(u8, c.arch, "glm4moe");
        const is_oai = std.mem.eql(u8, c.arch, "gpt-oss");
        try std.testing.expectEqual(
            if (is_glm) moe.GatingFunc.sigmoid else if (is_oai) moe.GatingFunc.softmax_topk else moe.GatingFunc.softmax,
            m.cfg.route.gating,
        );
        try std.testing.expectEqual(is_glm, last.exp_probs_b != null);
        // qwen2moe does not renormalize its gates; gpt-oss's softmax-topk
        // already normalizes, so weights_norm stays off there too
        try std.testing.expectEqual(!std.mem.eql(u8, c.arch, "qwen2moe") and !is_oai, m.cfg.route.weights_norm);
        if (is_oai) {
            // the gpt-oss extras all resolved, and the swa pattern is
            // even-windowed / odd-full
            try std.testing.expectEqual(@as(usize, 4), m.cfg.swa_window);
            try std.testing.expect(m.cfg.attn_logit_mul > 1.0);
            try std.testing.expect(last.attn_sinks != null);
            try std.testing.expect(last.attn_output_b != null);
            try std.testing.expect(last.ffn_gate_inp_b != null);
            try std.testing.expect(last.ffn_gate_exps_b != null);
            try std.testing.expect(last.ffn_up_exps_b != null);
            try std.testing.expect(last.ffn_down_exps_b != null);
            try std.testing.expect(m.layers[0].is_swa);
            try std.testing.expect(!m.layers[1].is_swa);
        }

        // and it actually runs a token
        var st = try State.init(gpa, m.cfg);
        defer st.deinit(gpa);
        try step(&m, &st, 1, 0);
        for (st.logits) |v| try std.testing.expect(!std.math.isNan(v));
    }
}

test "batched prefill matches token-by-token prefill" {
    // The whole point of the batch path is that it changes nothing except how
    // many tokens share an unpacked weight. Anything else -- a causal mask that
    // lets a token see its successors, a KV write at the wrong position, a
    // per-token RoPE angle taken from the batch index instead of the absolute
    // position -- shows up as a divergence here and nowhere else, because the
    // text it produces stays fluent either way.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    for ([_][]const u8{ "llama", "qwen3moe", "glm4moe", "gpt-oss" }) |arch| {
        var pbuf: [96]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-batch-{s}.gguf", .{arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf.writeMoeFixture(gpa, io, path, 11, arch);

        var m = try load(gpa, io, path);
        defer m.deinit();

        const toks = [_]u32{ 7, 21, 4, 90, 33, 8, 12, 5 };

        var a = try State.init(gpa, m.cfg);
        defer a.deinit(gpa);
        for (toks, 0..) |t, i| try step(&m, &a, t, i);

        var b = try State.init(gpa, m.cfg);
        defer b.deinit(gpa);
        try stepBatch(&m, &b, &toks, 0);

        // Logits after the same prefix must agree. Batching reorders only the
        // weight unpack, so the tolerance is for f32 summation order, not for
        // any approximation.
        for (a.logits, b.logits, 0..) |want, got, i| {
            const tol = @max(@abs(want), @abs(got)) * 1e-3 + 1e-3;
            std.testing.expectApproxEqAbs(want, got, tol) catch |e| {
                std.debug.print("{s}: logit {d}: serial {d} vs batched {d}\n", .{ arch, i, want, got });
                return e;
            };
        }

        // and a split batch must continue correctly from a partial prefix
        var c = try State.init(gpa, m.cfg);
        defer c.deinit(gpa);
        try stepBatch(&m, &c, toks[0..3], 0);
        try stepBatch(&m, &c, toks[3..], 3);
        for (a.logits, c.logits) |want, got| {
            try std.testing.expectApproxEqAbs(want, got, @max(@abs(want), @abs(got)) * 1e-3 + 1e-3);
        }
    }
}

test "masked verify shrinks the expert union and never touches lane 0" {
    // The verifier expert mask (lever 8) restricts a verify window's MoE
    // routing to a commitment-weighted eligible set. Two invariants make it
    // safe to ship at all: lane 0 -- the token already committed -- keeps its
    // natural top-k, so its logits are byte-identical to the unmasked path
    // and the accept/reject verdict for the first draft position can never
    // change; and the eligible set is never larger than the natural union,
    // so the mask can only reduce expert traffic. Both are checked here on
    // every GQA arch, because either failing is silent output corruption.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    for ([_][]const u8{ "llama", "qwen3moe", "glm4moe", "gpt-oss" }) |arch| {
        var pbuf: [96]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-vmask-{s}.gguf", .{arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf.writeMoeFixture(gpa, io, path, 11, arch);

        var m = try load(gpa, io, path);
        defer m.deinit();
        const c = m.cfg;

        const prefix = [_]u32{ 7, 21, 4 };
        const window = [_]u32{ 90, 33, 8, 12, 5 };

        var a = try State.init(gpa, c);
        defer a.deinit(gpa);
        try stepBatch(&m, &a, &prefix, 0);
        a.spec_capture = true;
        try stepBatch(&m, &a, &window, prefix.len);
        a.spec_capture = false;

        var b = try State.init(gpa, c);
        defer b.deinit(gpa);
        try stepBatch(&m, &b, &prefix, 0);
        b.spec_capture = true;
        b.verify_mask = true;
        try stepBatch(&m, &b, &window, prefix.len);
        b.verify_mask = false;
        b.spec_capture = false;

        // Lane 0 byte-identical: the committed token's verdict is untouched.
        for (a.blogits[0..c.vocab], b.blogits[0..c.vocab], 0..) |want, got, i| {
            std.testing.expectEqual(want, got) catch |e| {
                std.debug.print("{s}: lane-0 logit {d}: unmasked {d} vs masked {d}\n", .{ arch, i, want, got });
                return e;
            };
        }

        // The mask ran and can only shrink the union.
        try std.testing.expect(b.mask_natural > 0);
        try std.testing.expect(b.mask_eligible <= b.mask_natural);

        // Every masked lane still produced finite logits.
        for (0..window.len) |k| {
            for (b.blogits[k * c.vocab ..][0..c.vocab]) |v| {
                try std.testing.expect(std.math.isFinite(v));
            }
        }
    }
}

test "continuous-batch decode is token-identical to independent single streams" {
    // decodeBatch batches the weight matmuls across *unrelated* sequences, each
    // with its own KV cache and position. The failure modes are all silent:
    // one lane reading another's cache, a KV write at the batch index instead
    // of the sequence's own position, or a RoPE angle from the wrong pos. Each
    // would diverge here and only here, since every lane still produces fluent
    // text on its own. The reference is the same streams run one at a time.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    for ([_][]const u8{ "llama", "qwen3moe", "glm4moe", "gpt-oss" }) |arch| {
        var pbuf: [96]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-cb-{s}.gguf", .{arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf.writeMoeFixture(gpa, io, path, 11, arch);

        var m = try load(gpa, io, path);
        defer m.deinit();

        // Three independent sequences, deliberately different tokens (same
        // length so the batch stays full every step).
        const streams = [_][4]u32{
            .{ 7, 21, 4, 90 },
            .{ 3, 3, 15, 2 },
            .{ 50, 1, 8, 33 },
        };
        const n = streams.len;
        const vocab = m.cfg.vocab;

        // Reference: each stream through its own single-sequence State.
        const want = try gpa.alloc(f32, n * vocab);
        defer gpa.free(want);
        for (streams, 0..) |s, si| {
            var a = try State.init(gpa, m.cfg);
            defer a.deinit(gpa);
            for (s, 0..) |t, i| try step(&m, &a, t, i);
            @memcpy(want[si * vocab ..][0..vocab], a.logits);
        }

        // Batched: n sequences advanced in lockstep, one token per call.
        var st = try State.init(gpa, m.cfg);
        defer st.deinit(gpa);
        var seqs: [n]Seq = undefined;
        for (0..n) |k| seqs[k] = try Seq.init(gpa, m.cfg);
        defer for (0..n) |k| seqs[k].deinit(gpa);

        const got = try gpa.alloc(f32, n * vocab);
        defer gpa.free(got);
        var toks: [n]u32 = undefined;
        for (0..streams[0].len) |i| {
            for (0..n) |k| toks[k] = streams[k][i];
            try decodeBatch(&m, &st, &seqs, &toks, got);
        }

        for (0..n) |k| {
            for (want[k * vocab ..][0..vocab], got[k * vocab ..][0..vocab], 0..) |w, g, j| {
                const tol = @max(@abs(w), @abs(g)) * 1e-3 + 1e-3;
                std.testing.expectApproxEqAbs(w, g, tol) catch |e| {
                    std.debug.print("{s}: lane {d} logit {d}: serial {d} vs batched {d}\n", .{ arch, k, j, w, g });
                    return e;
                };
            }
        }
    }
}

test "MTP speculative decode is token-identical to plain greedy decode" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const path = "test-mtp-glm4moe.gguf";
    defer Io.Dir.cwd().deleteFile(io, path) catch {};
    try gguf.writeMoeFixture(gpa, io, path, 11, "glm4moe");

    var m = try load(gpa, io, path);
    defer m.deinit();
    try std.testing.expect(m.mtp != null);
    try std.testing.expect(m.cfg.has_mtp);

    const N = 24;
    const first: u32 = 5;

    // Reference: plain greedy decode.
    var ref: [N]u32 = undefined;
    {
        var st = try State.init(gpa, m.cfg);
        defer st.deinit(gpa);
        var tok: u32 = first;
        var pos: usize = 0;
        for (0..N) |k| {
            try step(&m, &st, tok, pos);
            pos += 1;
            tok = argmaxSlice(st.logits);
            ref[k] = tok;
        }
    }

    // Speculative: same greedy stream through stepSpec. The invariant is
    // exact equality -- acceptance is an argmax comparison of the same
    // arithmetic, so a divergence is a bug, not a tuning issue.
    var got: [N]u32 = undefined;
    {
        var st = try State.init(gpa, m.cfg);
        defer st.deinit(gpa);
        // one plain step primes st.x/st.logits for the first draft
        try step(&m, &st, first, 0);
        var pos: usize = 1;
        var cur = argmaxSlice(st.logits);
        got[0] = cur;
        var n_out: usize = 1;
        while (n_out < N) {
            const r = try stepSpec(&m, &st, cur, pos);
            pos += r.n;
            if (r.n == 2 and n_out < N) {
                got[n_out] = r.tok2;
                n_out += 1;
            }
            if (n_out < N) {
                got[n_out] = r.next;
                n_out += 1;
            }
            cur = r.next;
        }
        // the drafter must have been exercised, whatever its acceptance
        try std.testing.expect(st.spec_fwd > 0);
    }
    try std.testing.expectEqualSlices(u32, &ref, &got);
}
