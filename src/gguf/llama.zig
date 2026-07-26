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
const tensor = @import("../core/tensor.zig");
const special = @import("special.zig");

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

    pub fn encodePrompt(self: *const Model, gpa: std.mem.Allocator, text: []const u8, parse_special: bool) ![]u32 {
        return self.tok.encode(gpa, text, true, parse_special);
    }
    pub fn decodeToken(self: *const Model, w: *Io.Writer, id: u32) !void {
        return self.tok.decode(w, id);
    }
    pub fn eosToken(self: *const Model) u32 {
        return self.tok.eos;
    }

    pub fn deinit(self: *Model) void {
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
        if (ne2 != 1) return error.BadTensorShape; // llama tensors are 2D
        return .{ .ty = ty, .data = self.mm.memory[start..end], .ne0 = ne0, .ne1 = ne1 };
    }

    /// Assert a resolved tensor has exactly the shape the config implies.
    /// Without this the kernels trust the file's dims and the config
    /// independently: e.g. a `[dim, 1<<20]` q_a weight against an 8-float
    /// destination is a heap overflow write (security issue #29).
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
    // `head_dim` divides by n_heads, and Config.headDim/kvDim do it again on
    // every forward pass: a metadata head_count of 0 is a guaranteed crash
    // (security issue #29).
    if (dim == 0 or n_heads == 0 or dim % n_heads != 0) return error.BadConfig;
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

    if (cfg.n_layers == 0 or cfg.ctx_len == 0 or cfg.ffn == 0) return error.BadConfig;
    if (cfg.n_kv_heads == 0 or cfg.n_heads % cfg.n_kv_heads != 0) return error.BadConfig;
    // scores/kv allocations multiply these; keep the products representable
    const kv = std.math.mul(usize, cfg.n_layers, cfg.ctx_len) catch return error.BadConfig;
    _ = std.math.mul(usize, kv, cfg.kvDim()) catch return error.BadConfig;

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
    if (model.cfg.vocab == 0) return error.BadConfig;

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
        // The kernels already require these shapes (mv asserts out.len==rows,
        // x.len==cols); assert them here so a hostile file fails loudly at load
        // instead of walking past a buffer in ReleaseFast (security issue #29).
        const hd = cfg.headDim();
        try Model.expectShape(l.attn_norm, cfg.dim, 1, "attn_norm");
        try Model.expectF32(l.attn_norm, "attn_norm");
        try Model.expectShape(l.attn_q, cfg.dim, cfg.n_heads * hd, "attn_q");
        try Model.expectShape(l.attn_k, cfg.dim, cfg.kvDim(), "attn_k");
        try Model.expectShape(l.attn_v, cfg.dim, cfg.kvDim(), "attn_v");
        try Model.expectShape(l.attn_output, cfg.n_heads * hd, cfg.dim, "attn_output");
        try Model.expectShape(l.ffn_norm, cfg.dim, 1, "ffn_norm");
        try Model.expectF32(l.ffn_norm, "ffn_norm");
        try Model.expectShape(l.ffn_gate, cfg.dim, cfg.ffn, "ffn_gate");
        try Model.expectShape(l.ffn_up, cfg.dim, cfg.ffn, "ffn_up");
        try Model.expectShape(l.ffn_down, cfg.ffn, cfg.dim, "ffn_down");
    }
    model.layers = layers;
    try Model.expectF32(model.output_norm, "output_norm");

    model.tok = try Tokenizer.init(gpa, &model.parsed);
    return model;
}

// ---- SentencePiece tokenizer (from GGUF metadata) ---------------------------

pub const Tokenizer = struct {
    tokens: []const []const u8, // arena-owned by Parsed
    scores: []const f32,
    types: []const i32,
    lookup: std.StringHashMap(u32),
    specials: special.Set,
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
        // `scores` and `types` are indexed by a token id derived from `tokens`
        // (encode does `self.scores[id]`), so unequal lengths are an OOB read
        // from a malicious GGUF (security issue #29).
        if (scores.len != tokens.len or types.len != tokens.len) return error.BadTokenizer;
        if (tokens.len == 0) return error.BadTokenizer;
        // bos/eos are metadata and get emitted into the token stream, which
        // indexes token_embd rows; keep them inside the vocab.
        const bos_id = parsed.getUint("tokenizer.ggml.bos_token_id") orelse 1;
        const eos_id = parsed.getUint("tokenizer.ggml.eos_token_id") orelse 2;
        if (bos_id >= tokens.len or eos_id >= tokens.len) return error.BadTokenizer;

        var lookup = std.StringHashMap(u32).init(gpa);
        errdefer lookup.deinit();
        for (tokens, 0..) |t, i| try lookup.put(t, @intCast(i));
        var specials = try special.Set.build(gpa, tokens, types);
        errdefer specials.deinit(gpa);
        return .{
            .tokens = tokens,
            .scores = scores,
            .types = types,
            .lookup = lookup,
            .specials = specials,
            .bos = std.math.cast(u32, bos_id) orelse return error.BadTokenizer,
            .eos = std.math.cast(u32, eos_id) orelse return error.BadTokenizer,
        };
    }

    pub fn deinit(self: *Tokenizer, gpa: std.mem.Allocator) void {
        self.lookup.deinit();
        self.specials.deinit(gpa);
    }

    /// SPM encode: prefix a space, map ' ' -> U+2581, split into UTF-8 chars,
    /// then greedily merge the adjacent pair whose concatenation is the
    /// highest-scoring vocab entry. Unmatched symbols fall back to byte tokens.
    /// `parse_special`: when true, special-token strings are emitted as their
    /// atomic ids; when false they are SPM-encoded as ordinary text (so
    /// untrusted input cannot inject a control token).
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_bos: bool, parse_special: bool) ![]u32 {
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(gpa);
        if (add_bos) try out.append(gpa, self.bos);

        if (!parse_special) {
            try self.encodeSegment(gpa, text, true, &out);
            return out.toOwnedSlice(gpa);
        }

        // Split on special tokens, emitting their ids atomically and
        // SPM-encoding the text between.
        var seg_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (self.specials.matchAt(text[i..])) |sp| {
                if (i > seg_start) try self.encodeSegment(gpa, text[seg_start..i], seg_start == 0, &out);
                try out.append(gpa, sp.id);
                i += sp.text.len;
                seg_start = i;
            } else i += 1;
        }
        if (seg_start < text.len) try self.encodeSegment(gpa, text[seg_start..], seg_start == 0, &out);
        return out.toOwnedSlice(gpa);
    }

    /// SPM-encode one normal segment into `out`. `add_space_prefix` adds the
    /// leading dummy space (SPM add_dummy_prefix); applied only to a segment at
    /// the very start of the input, not to content following a special token.
    fn encodeSegment(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_space_prefix: bool, out: *std.ArrayList(u32)) !void {
        // preprocess: leading space, spaces -> ▁ (e2 96 81)
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(gpa);
        if (add_space_prefix) try buf.appendSlice(gpa, "\xe2\x96\x81");
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
pub fn step(m: *const Model, st: *State, token: u32, pos: usize) !void {
    const cfg = m.cfg;
    const hd = cfg.headDim();
    const kvd = cfg.kvDim();
    const q_per_kv = cfg.n_heads / cfg.n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // A token id indexes token_embd rows directly. Ids come from the
    // tokenizer, whose ceiling is an independent metadata array, so bound it
    // here rather than trusting the two to agree (security issue #29).
    if (token >= cfg.vocab) return error.TokenOutOfRange;
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
