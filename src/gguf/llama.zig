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
};

pub const RopeStyle = enum { norm, neox };

pub const arches = [_]Arch{
    .{ .name = "llama", .rope = .norm }, // includes Mixtral (MoE llama)
    .{ .name = "qwen2moe", .rope = .neox },
    .{ .name = "qwen3moe", .rope = .neox },
    .{ .name = "glm4moe", .rope = .neox },
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
    const N = struct {
        fn f(buf: []u8, li: usize, comptime s: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "blk.{d}." ++ s, .{li}) catch unreachable;
        }
    };
    const hd = cfg.head_dim;
    for (layers, 0..) |*l, i| {
        l.attn_norm = try model.resolve(N.f(&nb, i, "attn_norm.weight"));
        try Model.expectF32(l.attn_norm, "attn_norm");
        try Model.expectShape(l.attn_norm, cfg.dim, 1, "attn_norm");
        l.attn_q = try model.resolve(N.f(&nb, i, "attn_q.weight"));
        l.attn_k = try model.resolve(N.f(&nb, i, "attn_k.weight"));
        l.attn_v = try model.resolve(N.f(&nb, i, "attn_v.weight"));
        l.attn_output = try model.resolve(N.f(&nb, i, "attn_output.weight"));
        try Model.expectShape(l.attn_q, cfg.dim, cfg.qDim(), "attn_q");
        try Model.expectShape(l.attn_k, cfg.dim, cfg.kvDim(), "attn_k");
        try Model.expectShape(l.attn_v, cfg.dim, cfg.kvDim(), "attn_v");
        try Model.expectShape(l.attn_output, cfg.qDim(), cfg.dim, "attn_output");

        // qwen2moe and glm4moe carry QKV biases; llama and qwen3moe do not.
        // Detect rather than tabulate: the file is the authority.
        l.attn_q_bias = model.resolveOpt(N.f(&nb, i, "attn_q.bias"));
        l.attn_k_bias = model.resolveOpt(N.f(&nb, i, "attn_k.bias"));
        l.attn_v_bias = model.resolveOpt(N.f(&nb, i, "attn_v.bias"));
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
        l.attn_q_norm = model.resolveOpt(N.f(&nb, i, "attn_q_norm.weight"));
        l.attn_k_norm = model.resolveOpt(N.f(&nb, i, "attn_k_norm.weight"));
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
        l.ffn_norm = model.resolveOpt(N.f(&nb, i, "ffn_norm.weight")) orelse
            try model.resolve(N.f(&nb, i, "post_attention_norm.weight"));
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
            l.ffn_gate = try model.resolve(N.f(&nb, i, "ffn_gate.weight"));
            l.ffn_up = try model.resolve(N.f(&nb, i, "ffn_up.weight"));
            l.ffn_down = try model.resolve(N.f(&nb, i, "ffn_down.weight"));
            try Model.expectShape(l.ffn_gate.?, cfg.dim, cfg.ffn, "ffn_gate");
            try Model.expectShape(l.ffn_up.?, cfg.dim, cfg.ffn, "ffn_up");
            try Model.expectShape(l.ffn_down.?, cfg.ffn, cfg.dim, "ffn_down");
        } else {
            l.ffn_gate_inp = try model.resolve(N.f(&nb, i, "ffn_gate_inp.weight"));
            try Model.expectShape(l.ffn_gate_inp.?, cfg.dim, cfg.n_expert, "ffn_gate_inp");
            l.exp_probs_b = model.resolveOpt(N.f(&nb, i, "exp_probs_b.bias"));
            if (l.exp_probs_b) |b| {
                try Model.expectF32(b, "exp_probs_b.bias");
                try Model.expectShape(b, cfg.n_expert, 1, "exp_probs_b.bias");
            }
            l.ffn_gate_exps = try model.resolve(N.f(&nb, i, "ffn_gate_exps.weight"));
            l.ffn_up_exps = try model.resolve(N.f(&nb, i, "ffn_up_exps.weight"));
            l.ffn_down_exps = try model.resolve(N.f(&nb, i, "ffn_down_exps.weight"));
            try Model.expectExpertShape(l.ffn_gate_exps.?, cfg.dim, cfg.moe_ffn, cfg.n_expert, "ffn_gate_exps");
            try Model.expectExpertShape(l.ffn_up_exps.?, cfg.dim, cfg.moe_ffn, cfg.n_expert, "ffn_up_exps");
            try Model.expectExpertShape(l.ffn_down_exps.?, cfg.moe_ffn, cfg.dim, cfg.n_expert, "ffn_down_exps");

            // Shared expert: qwen2moe always has one, glm4moe has one when
            // expert_shared_count > 0, qwen3moe and Mixtral have none.
            l.ffn_gate_shexp = model.resolveOpt(N.f(&nb, i, "ffn_gate_shexp.weight"));
            if (l.ffn_gate_shexp) |g| {
                l.ffn_up_shexp = try model.resolve(N.f(&nb, i, "ffn_up_shexp.weight"));
                l.ffn_down_shexp = try model.resolve(N.f(&nb, i, "ffn_down_shexp.weight"));
                try Model.expectShape(g, cfg.dim, cfg.shexp_ffn, "ffn_gate_shexp");
                try Model.expectShape(l.ffn_up_shexp.?, cfg.dim, cfg.shexp_ffn, "ffn_up_shexp");
                try Model.expectShape(l.ffn_down_shexp.?, cfg.shexp_ffn, cfg.dim, "ffn_down_shexp");
                l.ffn_gate_inp_shexp = model.resolveOpt(N.f(&nb, i, "ffn_gate_inp_shexp.weight"));
                if (l.ffn_gate_inp_shexp) |s| try Model.expectShape(s, cfg.dim, 1, "ffn_gate_inp_shexp");
            }
        }
    }
    model.layers = layers;

    model.tok = try Tok.init(gpa, &model.parsed);
    return model;
}

/// Attach a distributed expert source: routed-expert reads become
/// Source.get() calls (local shard or peer fetch) instead of memory-map reads.
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
// ---- forward pass ------------------------------------------------------------

pub const State = struct {
    cfg: Config,
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
            .k_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * kvd),
            .v_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * kvd),
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
        };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        inline for (.{
            self.x,        self.normed,   self.q,      self.k,       self.v,
            self.attn_out, self.gate,     self.up,     self.act,     self.ffn_out,
            self.moe_acc,  self.router,   self.scores, self.logits,  self.k_cache,
            self.v_cache,  self.proj_out, self.bx,     self.bnormed, self.bq,
            self.bk,       self.bv,       self.battn,  self.bproj,   self.bgate,
            self.bup,      self.bact,     self.bffn,
        }) |sl| gpa.free(sl);
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

    const now = struct {
        fn f() i128 {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
        }
    }.f;

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
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
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
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

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
        while (h < cfg.n_heads) : (h += 1) ropeApply(cfg.arch.rope, st.q[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);
        h = 0;
        while (h < cfg.n_kv_heads) : (h += 1) ropeApply(cfg.arch.rope, st.k[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);

        // append to cache
        const cache_base = (li * cfg.ctx_len + pos) * kvd;
        @memcpy(st.k_cache[cache_base..][0..kvd], st.k);
        @memcpy(st.v_cache[cache_base..][0..kvd], st.v);
        _ = backend.kvAppend(li, pos, st.k, st.v);

        // per-head attention over positions 0..=pos
        const seq = pos + 1;
        // Offer the whole head loop to the backend first. The host cache above
        // is still written either way: the backend keeps its own device copy,
        // and if it ever declines mid-generation the engine's cache has to be
        // authoritative or the fallback produces garbage.
        if (!backend.attnHeads(li, pos, st.q, st.attn_out, cfg.n_heads, cfg.n_kv_heads, hd, scale)) {
            h = 0;
            while (h < cfg.n_heads) : (h += 1) {
                const kvh = h / q_per_kv;
                const qh = st.q[h * hd ..][0..hd];
                var t_i: usize = 0;
                while (t_i < seq) : (t_i += 1) {
                    const kt = st.k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    st.scores[t_i] = backend.dotF32(qh, kt) * scale;
                }
                backend.softmax(st.scores[0..seq]);
                const oh = st.attn_out[h * hd ..][0..hd];
                @memset(oh, 0);
                t_i = 0;
                while (t_i < seq) : (t_i += 1) {
                    const vt = st.v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    backend.axpy(oh, vt, st.scores[t_i]);
                }
            }
        }
        mv(l.attn_output, st.proj_out, st.attn_out);
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

        try moeLayer(m, st, l, li);
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
fn moeLayer(m: *const Model, st: *State, l: LayerT, li: usize) !void {
    const cfg = m.cfg;
    mv(l.ffn_gate_inp.?, st.router[0..cfg.n_expert], st.normed);
    var sel_buf: [moe.MAX_SELECTED]moe.Selected = undefined;
    const sel = sel_buf[0..cfg.n_used];
    const bias: ?[]const f32 = if (l.exp_probs_b) |b| tensorAsF32(b) else null;
    moe.route(cfg.route, st.router[0..cfg.n_expert], bias, sel);

    const acc = st.moe_acc;
    @memset(acc, 0);
    if (m.dist) |src| {
        // warm the missing shards in parallel: per-layer miss latency
        // becomes max(fetch), not sum(fetch)
        var ids_buf: [moe.MAX_SELECTED]usize = undefined;
        for (sel, 0..) |s, k| ids_buf[k] = m.expert_shard[li * cfg.n_expert + s.expert];
        src.prefetch(ids_buf[0..sel.len]);
    }
    for (sel) |s| {
        if (m.dist) |src| {
            // One shard carries this expert's [gate, up, down] slices
            // back to back, in that order (p2p/weights.zig).
            const blk = try src.get(m.expert_shard[li * cfg.n_expert + s.expert]);
            const gt = l.ffn_gate_exps.?;
            const ut = l.ffn_up_exps.?;
            const dt = l.ffn_down_exps.?;
            const gl = gt.ne1 * ggml.rowBytes(gt.ty, gt.ne0);
            const ul = ut.ne1 * ggml.rowBytes(ut.ty, ut.ne0);
            const dl = dt.ne1 * ggml.rowBytes(dt.ty, dt.ne0);
            denseFFN(
                st,
                .{ .ty = gt.ty, .data = blk[0..gl], .ne0 = gt.ne0, .ne1 = gt.ne1 },
                .{ .ty = ut.ty, .data = blk[gl..][0..ul], .ne0 = ut.ne0, .ne1 = ut.ne1 },
                .{ .ty = dt.ty, .data = blk[gl + ul ..][0..dl], .ne0 = dt.ne0, .ne1 = dt.ne1 },
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
pub fn stepBatch(m: *const Model, st: *State, tokens: []const u32, pos_base: usize) !void {
    const n = tokens.len;
    std.debug.assert(n > 0 and n <= backend.MAX_BATCH);
    if (n == 1) return step(m, st, tokens[0], pos_base);

    const cfg = m.cfg;
    const hd = cfg.head_dim;
    const kvd = cfg.kvDim();
    const qd = cfg.qDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

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
            for (0..cfg.n_heads) |h| ropeApply(cfg.arch.rope, qk[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);
            for (0..cfg.n_kv_heads) |h| ropeApply(cfg.arch.rope, kk[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);
            const base = (li * cfg.ctx_len + pos) * kvd;
            @memcpy(st.k_cache[base..][0..kvd], kk);
            @memcpy(st.v_cache[base..][0..kvd], vk);
            _ = backend.kvAppend(li, pos_base + k, kk, vk);
        }

        // Causal: token k sees positions 0..pos_base+k only, even though the
        // whole batch is already in the cache.
        for (0..n) |k| {
            const seq = pos_base + k + 1;
            const qk = st.bq[k * qd ..][0..qd];
            const ok = st.battn[k * qd ..][0..qd];
            for (0..cfg.n_heads) |h| {
                const kvh = h / q_per_kv;
                const qh = qk[h * hd ..][0..hd];
                for (0..seq) |t_i| {
                    const kt = st.k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    st.scores[t_i] = backend.dotF32(qh, kt) * scale;
                }
                backend.softmax(st.scores[0..seq]);
                const oh = ok[h * hd ..][0..hd];
                @memset(oh, 0);
                for (0..seq) |t_i| {
                    const vt = st.v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                    backend.axpy(oh, vt, st.scores[t_i]);
                }
            }
        }
        mmul(l.attn_output, st.bproj[0 .. n * cfg.dim], st.battn[0 .. n * qd], n);
        for (0..n) |k| backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.bproj[k * cfg.dim ..][0..cfg.dim]);

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
        // time through the existing single-token path.
        for (0..n) |k| {
            @memcpy(st.normed, st.bnormed[k * cfg.dim ..][0..cfg.dim]);
            try moeLayer(m, st, l, li);
            backend.add(st.bx[k * cfg.dim ..][0..cfg.dim], st.moe_acc);
        }
    }

    // Only the final token's logits matter for sampling.
    const last = st.bx[(n - 1) * cfg.dim ..][0..cfg.dim];
    backend.rmsnorm(st.normed, last, tensorAsF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
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
    const gguf_mod = @import("gguf.zig");

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
    };

    for (cases) |c| {
        var pbuf: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-arch-{s}.gguf", .{c.arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf_mod.writeMoeFixture(gpa, io, path, 7, c.arch);

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
        // glm4moe alone routes with sigmoid and a selection bias
        const is_glm = std.mem.eql(u8, c.arch, "glm4moe");
        try std.testing.expectEqual(
            if (is_glm) moe.GatingFunc.sigmoid else moe.GatingFunc.softmax,
            m.cfg.route.gating,
        );
        try std.testing.expectEqual(is_glm, last.exp_probs_b != null);
        // qwen2moe is the only one that does not renormalize its gates
        try std.testing.expectEqual(!std.mem.eql(u8, c.arch, "qwen2moe"), m.cfg.route.weights_norm);

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
    const gguf_mod = @import("gguf.zig");

    for ([_][]const u8{ "llama", "qwen3moe", "glm4moe" }) |arch| {
        var pbuf: [96]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "test-batch-{s}.gguf", .{arch});
        defer Io.Dir.cwd().deleteFile(io, path) catch {};
        try gguf_mod.writeMoeFixture(gpa, io, path, 11, arch);

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
