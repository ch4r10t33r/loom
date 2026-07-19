//! Llama-architecture inference over a GGUF file — the engine that runs what
//! the distribution plane ships. Standard llama: RMSNorm, GQA attention with
//! NORM-style RoPE (adjacent-pair rotation), SwiGLU FFN, tied-or-separate
//! output head, and the SentencePiece tokenizer embedded in GGUF metadata.
//!
//! Weights stay in their GGML format (F32/F16/Q4_0/Q8_0) in a read-only memory
//! map; every matmul is a fused kernel over the raw bytes (ggml.zig).

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const tensor = @import("tensor.zig");

pub const Config = struct {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ffn: usize,
    vocab: usize,
    ctx_len: usize,
    rope_dim: usize,
    rope_base: f32,
    eps: f32,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.headDim();
    }
};

pub const Tensor = struct {
    ty: ggml.Type,
    data: []const u8,
    ne0: usize, // row length (input dim)
    ne1: usize, // rows (output dim)
};

const LayerT = struct {
    attn_norm: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_output: Tensor,
    ffn_norm: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
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

    tok: Tokenizer,

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
        const bytes = ggml.tensorBytes(ty, ne0, ne1);
        const start: usize = @intCast(self.parsed.data_offset + t.offset);
        if (start + bytes > self.mm.memory.len) return error.TruncatedGguf;
        return .{ .ty = ty, .data = self.mm.memory[start .. start + bytes], .ne0 = ne0, .ne1 = ne1 };
    }
};

pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Model {
    var parsed = try gguf.parse(gpa, io, path);
    errdefer parsed.deinit();

    const arch = parsed.getString("general.architecture") orelse return error.NoArchitecture;
    if (!std.mem.eql(u8, arch, "llama")) return error.UnsupportedArchitecture;

    const file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    var mm = try file.createMemoryMap(io, .{
        .len = @intCast(parsed.file_size),
        .protection = .{ .read = true, .write = false },
    });
    errdefer mm.destroy(io);

    const dim: usize = @intCast(parsed.getUint("llama.embedding_length") orelse return error.BadConfig);
    const n_heads: usize = @intCast(parsed.getUint("llama.attention.head_count") orelse return error.BadConfig);
    const head_dim = dim / n_heads;
    const cfg = Config{
        .dim = dim,
        .n_layers = @intCast(parsed.getUint("llama.block_count") orelse return error.BadConfig),
        .n_heads = n_heads,
        .n_kv_heads = @intCast(parsed.getUint("llama.attention.head_count_kv") orelse n_heads),
        .ffn = @intCast(parsed.getUint("llama.feed_forward_length") orelse return error.BadConfig),
        .vocab = 0, // fixed up below from token_embd
        .ctx_len = @intCast(parsed.getUint("llama.context_length") orelse 2048),
        .rope_dim = @intCast(parsed.getUint("llama.rope.dimension_count") orelse head_dim),
        .rope_base = @floatCast(parsed.getFloat("llama.rope.freq_base") orelse 10000.0),
        .eps = @floatCast(parsed.getFloat("llama.attention.layer_norm_rms_epsilon") orelse 1e-5),
    };

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
    model.output = model.resolve("output.weight") catch model.token_embd; // tied embeddings
    model.cfg.vocab = model.token_embd.ne1;

    const layers = try gpa.alloc(LayerT, cfg.n_layers);
    errdefer gpa.free(layers);
    var namebuf: [128]u8 = undefined;
    for (layers, 0..) |*l, i| {
        inline for (.{
            .{ "attn_norm", &l.attn_norm },   .{ "attn_q", &l.attn_q },
            .{ "attn_k", &l.attn_k },         .{ "attn_v", &l.attn_v },
            .{ "attn_output", &l.attn_output }, .{ "ffn_norm", &l.ffn_norm },
            .{ "ffn_gate", &l.ffn_gate },     .{ "ffn_up", &l.ffn_up },
            .{ "ffn_down", &l.ffn_down },
        }) |pair| {
            const name = try std.fmt.bufPrint(&namebuf, "blk.{d}.{s}.weight", .{ i, pair[0] });
            pair[1].* = try model.resolve(name);
        }
    }
    model.layers = layers;

    model.tok = try Tokenizer.init(gpa, &model.parsed);
    return model;
}

// ---- SentencePiece tokenizer (from GGUF metadata) ---------------------------

pub const Tokenizer = struct {
    tokens: []const []const u8, // arena-owned by Parsed
    scores: []const f32,
    types: []const i32,
    lookup: std.StringHashMap(u32),
    bos: u32,
    eos: u32,

    const BYTE_TYPE: i32 = 6;

    pub fn init(gpa: std.mem.Allocator, parsed: *const gguf.Parsed) !Tokenizer {
        const tokens = switch (parsed.findMeta("tokenizer.ggml.tokens") orelse return error.NoTokenizer) {
            .array_str => |a| a,
            else => return error.NoTokenizer,
        };
        const scores = switch (parsed.findMeta("tokenizer.ggml.scores") orelse return error.NoTokenizer) {
            .array_f32 => |a| a,
            else => return error.NoTokenizer,
        };
        const types = switch (parsed.findMeta("tokenizer.ggml.token_type") orelse return error.NoTokenizer) {
            .array_i32 => |a| a,
            else => return error.NoTokenizer,
        };
        var lookup = std.StringHashMap(u32).init(gpa);
        errdefer lookup.deinit();
        for (tokens, 0..) |t, i| try lookup.put(t, @intCast(i));
        return .{
            .tokens = tokens,
            .scores = scores,
            .types = types,
            .lookup = lookup,
            .bos = @intCast(parsed.getUint("tokenizer.ggml.bos_token_id") orelse 1),
            .eos = @intCast(parsed.getUint("tokenizer.ggml.eos_token_id") orelse 2),
        };
    }

    pub fn deinit(self: *Tokenizer, gpa: std.mem.Allocator) void {
        _ = gpa;
        self.lookup.deinit();
    }

    /// SPM encode: prefix a space, map ' ' -> U+2581, split into UTF-8 chars,
    /// then greedily merge the adjacent pair whose concatenation is the
    /// highest-scoring vocab entry. Unmatched symbols fall back to byte tokens.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        // preprocess: leading space, spaces -> ▁ (e2 96 81)
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "\xe2\x96\x81");
        for (text) |ch| {
            if (ch == ' ') {
                try buf.appendSlice(gpa, "\xe2\x96\x81");
            } else {
                try buf.append(gpa, ch);
            }
        }
        const s = buf.items;

        // symbols as (start,end) ranges over s, initially one UTF-8 char each
        const Range = struct { start: usize, end: usize };
        var syms = std.ArrayList(Range).empty;
        defer syms.deinit(gpa);
        {
            var i: usize = 0;
            while (i < s.len) {
                const l = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
                const end = @min(i + l, s.len);
                try syms.append(gpa, .{ .start = i, .end = end });
                i = end;
            }
        }

        // greedy best-pair merging by vocab score
        while (true) {
            var best_score: f32 = -std.math.inf(f32);
            var best_i: ?usize = null;
            var i: usize = 0;
            while (i + 1 < syms.items.len) : (i += 1) {
                const merged = s[syms.items[i].start..syms.items[i + 1].end];
                if (self.lookup.get(merged)) |id| {
                    if (self.scores[id] > best_score) {
                        best_score = self.scores[id];
                        best_i = i;
                    }
                }
            }
            const bi = best_i orelse break;
            syms.items[bi].end = syms.items[bi + 1].end;
            _ = syms.orderedRemove(bi + 1);
        }

        // map symbols to ids (byte fallback for stragglers)
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(gpa);
        if (add_bos) try out.append(gpa, self.bos);
        for (syms.items) |r| {
            const piece = s[r.start..r.end];
            if (self.lookup.get(piece)) |id| {
                try out.append(gpa, id);
            } else {
                for (piece) |byte| {
                    var namebuf: [8]u8 = undefined;
                    const bname = std.fmt.bufPrint(&namebuf, "<0x{X:0>2}>", .{byte}) catch unreachable;
                    if (self.lookup.get(bname)) |id| try out.append(gpa, id);
                    // no byte token in vocab: drop the byte
                }
            }
        }
        return out.toOwnedSlice(gpa);
    }

    /// Decode one token into `w`. Byte tokens emit their byte; ▁ becomes space.
    pub fn decode(self: *const Tokenizer, w: *Io.Writer, id: u32) !void {
        if (id >= self.tokens.len) return;
        const piece = self.tokens[id];
        if (id < self.types.len and self.types[id] == BYTE_TYPE) {
            // "<0xXX>"
            if (piece.len == 6) {
                const byte = std.fmt.parseInt(u8, piece[3..5], 16) catch return;
                try w.writeAll(&.{byte});
            }
            return;
        }
        var i: usize = 0;
        while (i < piece.len) {
            if (i + 3 <= piece.len and std.mem.eql(u8, piece[i .. i + 3], "\xe2\x96\x81")) {
                try w.writeAll(" ");
                i += 3;
            } else {
                try w.writeAll(piece[i .. i + 1]);
                i += 1;
            }
        }
    }
};

// ---- forward pass ------------------------------------------------------------

pub const State = struct {
    cfg: Config,
    // activations
    x: []f32,
    normed: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    proj_out: []f32,
    gate: []f32,
    up: []f32,
    act: []f32,
    ffn_out: []f32,
    scores: []f32,
    logits: []f32,
    // kv cache: [n_layers][ctx][kv_dim]
    k_cache: []f32,
    v_cache: []f32,

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !State {
        const kvd = cfg.kvDim();
        return .{
            .cfg = cfg,
            .x = try gpa.alloc(f32, cfg.dim),
            .normed = try gpa.alloc(f32, cfg.dim),
            .q = try gpa.alloc(f32, cfg.dim),
            .k = try gpa.alloc(f32, kvd),
            .v = try gpa.alloc(f32, kvd),
            .attn_out = try gpa.alloc(f32, cfg.dim),
            .proj_out = try gpa.alloc(f32, cfg.dim),
            .gate = try gpa.alloc(f32, cfg.ffn),
            .up = try gpa.alloc(f32, cfg.ffn),
            .act = try gpa.alloc(f32, cfg.ffn),
            .ffn_out = try gpa.alloc(f32, cfg.dim),
            .scores = try gpa.alloc(f32, cfg.ctx_len),
            .logits = try gpa.alloc(f32, cfg.vocab),
            .k_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * kvd),
            .v_cache = try gpa.alloc(f32, cfg.n_layers * cfg.ctx_len * kvd),
        };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        inline for (.{
            self.x,        self.normed, self.q,       self.k,      self.v,
            self.attn_out, self.proj_out, self.gate,  self.up,     self.act,
            self.ffn_out,  self.scores, self.logits,  self.k_cache, self.v_cache,
        }) |sl| gpa.free(sl);
    }
};

fn mv(t: Tensor, out: []f32, x: []const f32) void {
    ggml.matvec(t.ty, out, t.data, x, t.ne1, t.ne0);
}

/// NORM-style RoPE: rotate adjacent pairs (2i, 2i+1) of the first rope_dim
/// dims of each head; freq_i = base^(-2i/rope_dim).
fn ropeNorm(vec: []f32, rope_dim: usize, pos: usize, base: f32) void {
    var i: usize = 0;
    while (i + 1 < rope_dim) : (i += 2) {
        const exponent = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rope_dim));
        const freq = std.math.pow(f32, base, -exponent);
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const c = @cos(angle);
        const sn = @sin(angle);
        const a = vec[i];
        const b = vec[i + 1];
        vec[i] = a * c - b * sn;
        vec[i + 1] = a * sn + b * c;
    }
}

/// One token step; logits land in `st.logits`.
pub fn step(m: *const Model, st: *State, token: u32, pos: usize) void {
    const cfg = m.cfg;
    const hd = cfg.headDim();
    const kvd = cfg.kvDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    ggml.dequantRow(m.token_embd.ty, st.x, m.token_embd.data, token, cfg.dim);

    for (m.layers, 0..) |l, li| {
        // attention
        tensor.rmsnorm(st.normed, st.x, tensorAsF32(l.attn_norm), cfg.eps);
        mv(l.attn_q, st.q, st.normed);
        mv(l.attn_k, st.k, st.normed);
        mv(l.attn_v, st.v, st.normed);

        var h: usize = 0;
        while (h < cfg.n_heads) : (h += 1) ropeNorm(st.q[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);
        h = 0;
        while (h < cfg.n_kv_heads) : (h += 1) ropeNorm(st.k[h * hd ..][0..hd], cfg.rope_dim, pos, cfg.rope_base);

        // append to cache
        const cache_base = (li * cfg.ctx_len + pos) * kvd;
        @memcpy(st.k_cache[cache_base ..][0..kvd], st.k);
        @memcpy(st.v_cache[cache_base ..][0..kvd], st.v);

        // per-head attention over positions 0..=pos
        const seq = pos + 1;
        h = 0;
        while (h < cfg.n_heads) : (h += 1) {
            const kvh = h / q_per_kv;
            const qh = st.q[h * hd ..][0..hd];
            var t_i: usize = 0;
            while (t_i < seq) : (t_i += 1) {
                const kt = st.k_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                var dot: f32 = 0;
                for (qh, kt) |a, b| dot += a * b;
                st.scores[t_i] = dot * scale;
            }
            tensor.softmax(st.scores[0..seq]);
            const oh = st.attn_out[h * hd ..][0..hd];
            @memset(oh, 0);
            t_i = 0;
            while (t_i < seq) : (t_i += 1) {
                const vt = st.v_cache[(li * cfg.ctx_len + t_i) * kvd + kvh * hd ..][0..hd];
                const w = st.scores[t_i];
                for (oh, vt) |*o, vv| o.* += w * vv;
            }
        }
        mv(l.attn_output, st.proj_out, st.attn_out);
        tensor.add(st.x, st.proj_out);

        // FFN
        tensor.rmsnorm(st.normed, st.x, tensorAsF32(l.ffn_norm), cfg.eps);
        mv(l.ffn_gate, st.gate, st.normed);
        mv(l.ffn_up, st.up, st.normed);
        tensor.swiglu(st.act, st.gate, st.up);
        mv(l.ffn_down, st.ffn_out, st.act);
        tensor.add(st.x, st.ffn_out);
    }

    tensor.rmsnorm(st.normed, st.x, tensorAsF32(m.output_norm), cfg.eps);
    mv(m.output, st.logits, st.normed);
}

/// Norm weights are always f32 in practice; view the raw bytes as f32.
fn tensorAsF32(t: Tensor) []const f32 {
    std.debug.assert(t.ty == .f32);
    return @alignCast(std.mem.bytesAsSlice(f32, t.data));
}
