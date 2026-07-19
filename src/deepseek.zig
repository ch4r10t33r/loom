//! deepseek2-family inference over GGUF (issue #1) — the architecture Kimi K2,
//! DeepSeek V2/V3, and GLM-class MoE models ship as.
//!
//! MLA attention: q through a LoRA bottleneck (q_a -> norm -> q_b), k/v through
//! a compressed latent (kv_a_mqa -> norm = c_kv, cached) plus one shared
//! rope-carrying k head; per-head k_nope/v are re-expanded from the cached
//! latent via kv_b every step. NEOX-style rope on the decoupled rope slice.
//!
//! MoE FFN: router (ffn_gate_inp) scored by sigmoid or softmax
//! (expert_gating_func), optional selection-only bias (exp_probs_b, the
//! noaux_tc trick), top-k experts with optionally renormalized gates scaled by
//! expert_weights_scale, plus always-on shared expert(s). Leading
//! `leading_dense_block_count` layers use a plain dense SwiGLU FFN.
//!
//! Weights stay in GGML formats in the read-only memory map (fused kernels,
//! ggml.zig). Known gaps vs. real checkpoints, tracked in issue #1: BPE (gpt2)
//! tokenizer, YaRN mscale attention scaling, K-quant tensor types.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const tensor = @import("tensor.zig");
const llama = @import("llama.zig");

pub const GatingFunc = enum(u32) { softmax = 1, sigmoid = 2 };

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

    /// Byte slice of expert `e` in a 3D tensor.
    pub fn expert(self: Tensor, e: usize) Tensor {
        const per = self.ne1 * ggml.rowBytes(self.ty, self.ne0);
        return .{ .ty = self.ty, .data = self.data[e * per ..][0..per], .ne0 = self.ne0, .ne1 = self.ne1, .ne2 = 1 };
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
    tok: llama.Tokenizer,

    pub fn deinit(self: *Model) void {
        self.tok.deinit(self.gpa);
        self.gpa.free(self.layers);
        self.mm.destroy(self.io);
        self.file.close(self.io);
        self.parsed.deinit();
    }

    fn resolve(self: *const Model, name: []const u8) !Tensor {
        const t = self.parsed.findTensor(name) orelse return error.MissingTensor;
        if (!ggml.Type.supported(t.ggml_type)) return error.UnsupportedTensorType;
        const ty: ggml.Type = @enumFromInt(t.ggml_type);
        const ne0: usize = @intCast(t.dims[0]);
        const ne1: usize = if (t.dims.len > 1) @intCast(t.dims[1]) else 1;
        const ne2: usize = if (t.dims.len > 2) @intCast(t.dims[2]) else 1;
        const bytes = ggml.tensorBytes(ty, ne0, ne1) * ne2;
        const start: usize = @intCast(self.parsed.data_offset + t.offset);
        if (start + bytes > self.mm.memory.len) return error.TruncatedGguf;
        return .{ .ty = ty, .data = self.mm.memory[start .. start + bytes], .ne0 = ne0, .ne1 = ne1, .ne2 = ne2 };
    }

    fn resolveOpt(self: *const Model, name: []const u8) ?Tensor {
        return self.resolve(name) catch null;
    }
};

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
    };
    if (cfg.n_layers > cfg.n_dense_layers and (cfg.n_expert == 0 or cfg.n_used == 0 or cfg.moe_ffn == 0))
        return error.BadConfig;
    if (cfg.n_expert > 512) return error.BadConfig;

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
        if (cfg.q_lora_rank > 0) {
            l.attn_q = null;
            l.attn_q_a = try model.resolve(N.f(&nb, i, "attn_q_a.weight"));
            l.attn_q_a_norm = try model.resolve(N.f(&nb, i, "attn_q_a_norm.weight"));
            l.attn_q_b = try model.resolve(N.f(&nb, i, "attn_q_b.weight"));
        } else {
            l.attn_q = try model.resolve(N.f(&nb, i, "attn_q.weight"));
            l.attn_q_a = null;
            l.attn_q_a_norm = null;
            l.attn_q_b = null;
        }
        l.attn_kv_a_mqa = try model.resolve(N.f(&nb, i, "attn_kv_a_mqa.weight"));
        l.attn_kv_a_norm = try model.resolve(N.f(&nb, i, "attn_kv_a_norm.weight"));
        l.attn_kv_b = try model.resolve(N.f(&nb, i, "attn_kv_b.weight"));
        l.attn_output = try model.resolve(N.f(&nb, i, "attn_output.weight"));
        l.ffn_norm = try model.resolve(N.f(&nb, i, "ffn_norm.weight"));
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

    model.tok = try llama.Tokenizer.init(gpa, &model.parsed);
    return model;
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
            self.x,     self.normed,   self.q_a,      self.q,       self.kv_a,
            self.k_nope, self.v_t,     self.head_out, self.proj_out, self.gate,
            self.up,    self.act,      self.ffn_out,  self.router,  self.scores,
            self.logits, self.c_kv_cache, self.k_rope_cache,
        }) |sl| gpa.free(sl);
    }
};

fn mv(t: Tensor, out: []f32, x: []const f32) void {
    ggml.matvec(t.ty, out, t.data, x, t.ne1, t.ne0);
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

/// One token step; logits land in st.logits.
pub fn step(m: *const Model, st: *State, token: u32, pos: usize) void {
    const cfg = m.cfg;
    const kd = cfg.keyDim();
    const nope = cfg.nope_dim;
    const rope = cfg.rope_dim;
    const kvr = cfg.kv_lora_rank;
    const vd = cfg.v_head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));

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
        tensor.rope(k_rope, pos, cfg.rope_base); // NEOX half-split

        var h: usize = 0;
        while (h < cfg.n_heads) : (h += 1) {
            tensor.rope(st.q[h * kd + nope ..][0..rope], pos, cfg.rope_base);
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
            for (sel) |s| {
                denseFFN(
                    st,
                    l.ffn_gate_exps.?.expert(s.expert),
                    l.ffn_up_exps.?.expert(s.expert),
                    l.ffn_down_exps.?.expert(s.expert),
                );
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
        .dim = 8, .n_layers = 1, .n_dense_layers = 0, .n_heads = 1, .q_lora_rank = 0,
        .kv_lora_rank = 4, .rope_dim = 2, .nope_dim = 2, .v_head_dim = 2, .ffn = 8,
        .moe_ffn = 8, .n_expert = 4, .n_used = 2, .n_shared = 0, .gating = .sigmoid,
        .weights_norm = true, .weights_scale = 1.0, .vocab = 8, .ctx_len = 8,
        .rope_base = 10000, .eps = 1e-6,
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
