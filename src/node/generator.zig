//! Generation abstraction (issue #17): one `generate` call over two engines,
//! so the serving surfaces (rpc.zig, openai.zig) work with either.
//!
//!   - `loom`  the loom-format byte-tokenizer engine (engine.zig)
//!   - `gguf`  the distributed GGUF deepseek2 engine, whose routed-expert reads
//!             are fetched from peers at token time (expert_fetch + attachDist)
//!
//! Both return the detokenized completion text plus prompt/completion token
//! counts; the loom path also returns the token-id list (the native RPC schema),
//! left empty for gguf.

const std = @import("std");
const Io = std.Io;
const engine_mod = @import("../engine/engine.zig");
const Engine = engine_mod.Engine;
const tokenizer = @import("../engine/tokenizer.zig");
const sampler = @import("../engine/sampler.zig");
const deepseek = @import("../gguf/deepseek.zig");
const llama = @import("../gguf/llama.zig");
const chat_template = @import("../gguf/chat_template.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");

pub const Result = struct {
    text: []u8, // owned detokenized completion bytes
    token_ids: []usize, // owned; loom fills, gguf leaves empty
    prompt_tokens: usize,
    completion_tokens: usize,
    stop: bool, // true = natural stop/eos, false = length cap

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        gpa.free(self.text);
        if (self.token_ids.len > 0) gpa.free(self.token_ids);
    }
};

/// Per-token byte sink for streaming: `emit` receives each token's detokenized
/// bytes as they are produced (used by the OpenAI SSE path). Errors propagate,
/// aborting generation (e.g. the client disconnected).
pub const TokenSink = struct {
    ctx: *anyopaque,
    emit: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!void,
};

/// The loaded GGUF model, by engine. deepseek2 is MLA (deepseek.zig); every
/// other supported architecture is GQA (llama.zig). Both expose the same
/// Model/State/step shape, so the generation loop below is written once and
/// instantiated for each.
pub const GgufModel = union(enum) {
    deepseek: deepseek.Model,
    gqa: llama.Model,

    pub fn deinit(self: *GgufModel) void {
        switch (self.*) {
            inline else => |*m| m.deinit(),
        }
    }
    pub fn ctxLen(self: *const GgufModel) usize {
        return switch (self.*) {
            inline else => |*m| m.cfg.ctx_len,
        };
    }
    pub fn setCtxLen(self: *GgufModel, n: usize) void {
        switch (self.*) {
            inline else => |*m| m.cfg.ctx_len = n,
        }
    }
    pub fn archName(self: *const GgufModel) []const u8 {
        return switch (self.*) {
            .deepseek => "deepseek2",
            .gqa => |*m| m.cfg.arch.name,
        };
    }
    pub fn chatTemplate(self: *const GgufModel) ?[]const u8 {
        return switch (self.*) {
            inline else => |*m| m.chatTemplate(),
        };
    }
    pub fn generalName(self: *const GgufModel) ?[]const u8 {
        return switch (self.*) {
            inline else => |*m| m.generalName(),
        };
    }
};

/// Distributed GGUF generator: the model plus its token-loop expert fetch
/// source. A fresh `State` (KV cache) is built per request.
pub const GgufGen = struct {
    m: GgufModel,
    /// The distributed expert source, or null when this GGUF is served
    /// locally (a dense model has no routed experts to fetch, so there is no
    /// source to attach). Optional rather than undefined: hitRate() reads it
    /// on every request, and an undefined pointer there is a wild read.
    src: ?*expert_fetch.Source = null,
    ctx_cap: usize,
    chat_format: chat_template.Format = .generic,
};

pub const Generator = union(enum) {
    loom: *Engine,
    gguf: *GgufGen,

    /// Generate a completion. `budget`, when set (the client's remaining metering
    /// allowance), clamps completion length to `budget - prompt_tokens`.
    pub fn generate(
        self: Generator,
        gpa: std.mem.Allocator,
        io: Io,
        prompt_text: []const u8,
        max_tokens: usize,
        temp: f32,
        seed: u64,
        budget: ?u64,
        sink: ?TokenSink,
        /// Parse special tokens in `prompt_text` (true only for trusted
        /// chat-template scaffold; false for raw user prompts, so untrusted
        /// input cannot inject control tokens). Ignored by the loom byte engine.
        parse_special: bool,
    ) !Result {
        return switch (self) {
            .loom => |e| genLoom(e, gpa, prompt_text, max_tokens, temp, seed, budget, sink),
            .gguf => |g| genGguf(g, gpa, io, prompt_text, max_tokens, temp, seed, budget, sink, parse_special),
        };
    }

    /// The chat template to render `messages[]` with. The loom byte engine has
    /// no template (generic); the GGUF engine's is detected at load.
    pub fn chatFormat(self: Generator) chat_template.Format {
        return switch (self) {
            .loom => .generic,
            .gguf => |g| g.chat_format,
        };
    }

    /// A serving-side cache/locality hit rate for logging (loom: expert-cache
    /// hit rate; gguf: fraction of expert reads served locally vs peer-fetched).
    pub fn hitRate(self: Generator) f64 {
        return switch (self) {
            .loom => |e| e.cache.stats.hitRate(),
            .gguf => |g| blk: {
                // A locally served GGUF has no fetch tier, so "fraction served
                // locally" is 1 by construction rather than unknown.
                const src = g.src orelse break :blk 1.0;
                const s = src.stats;
                const tot = s.local + s.fetched;
                break :blk if (tot == 0) 0 else @as(f64, @floatFromInt(s.local)) / @as(f64, @floatFromInt(tot));
            },
        };
    }
};

fn clampMax(max_tokens: usize, budget: ?u64, prompt_tokens: usize) usize {
    if (budget) |b| {
        const room = if (b > prompt_tokens) b - prompt_tokens else 0;
        return @intCast(@min(@as(u64, max_tokens), room));
    }
    return max_tokens;
}

/// Adapts a token-id callback (engine.TokenCb) to a byte-level TokenSink by
/// detokenizing each loom token to its single byte.
const LoomSinkAdapter = struct {
    sink: TokenSink,
    fn cb(ctx: *anyopaque, token: usize) anyerror!void {
        const self: *LoomSinkAdapter = @ptrCast(@alignCast(ctx));
        const b = [_]u8{tokenizer.decodeByte(token)};
        try self.sink.emit(self.sink.ctx, &b);
    }
};

fn genLoom(
    e: *Engine,
    gpa: std.mem.Allocator,
    prompt_text: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    budget: ?u64,
    sink: ?TokenSink,
) !Result {
    const toks = try tokenizer.encode(gpa, prompt_text);
    defer gpa.free(toks);
    const maxn = clampMax(max_tokens, budget, toks.len);

    var produced = std.ArrayList(usize).empty;
    errdefer produced.deinit(gpa);
    var adapter: LoomSinkAdapter = undefined;
    const on_token: ?Engine.TokenCb = if (sink) |s| blk: {
        adapter = .{ .sink = s };
        break :blk .{ .ctx = &adapter, .cb = LoomSinkAdapter.cb };
    } else null;
    const n = try e.generate(toks, maxn, temp, seed, &produced, on_token);

    const text = try gpa.alloc(u8, produced.items.len);
    errdefer gpa.free(text);
    for (produced.items, 0..) |t, i| text[i] = tokenizer.decodeByte(t);

    return .{
        .text = text,
        .token_ids = try produced.toOwnedSlice(gpa),
        .prompt_tokens = toks.len,
        .completion_tokens = n,
        .stop = n < maxn,
    };
}

fn genGguf(
    g: *GgufGen,
    gpa: std.mem.Allocator,
    io: Io,
    prompt_text: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    budget: ?u64,
    sink: ?TokenSink,
    parse_special: bool,
) !Result {
    _ = io;
    return switch (g.m) {
        .deepseek => |*m| genGgufInner(deepseek, m, gpa, prompt_text, max_tokens, temp, seed, budget, sink, parse_special),
        .gqa => |*m| genGgufInner(llama, m, gpa, prompt_text, max_tokens, temp, seed, budget, sink, parse_special),
    };
}

/// The GGUF generation loop, instantiated per engine. `E` is an engine module
/// exposing Model, State and step — deepseek.zig or llama.zig.
fn genGgufInner(
    comptime E: type,
    m: *E.Model,
    gpa: std.mem.Allocator,
    prompt_text: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    budget: ?u64,
    sink: ?TokenSink,
    parse_special: bool,
) !Result {
    const c = m.cfg;

    const toks = try m.encodePrompt(gpa, prompt_text, parse_special);
    defer gpa.free(toks);
    const maxn = clampMax(max_tokens, budget, toks.len);

    var st = try E.State.init(gpa, c);
    defer st.deinit(gpa);
    const scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(scratch);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    // prefill (routed-expert reads fault in from peers via attachDist)
    var pos: usize = 0;
    for (toks) |t| {
        if (pos >= c.ctx_len) break;
        try E.step(m, &st, t, pos);
        pos += 1;
    }

    var aw = std.Io.Writer.Allocating.init(gpa);
    defer aw.deinit();
    var produced: usize = 0;
    var eos = false;
    if (toks.len > 0) {
        var last: u32 = @intCast(sampler.sample(scratch, st.logits, temp, rnd));
        while (produced < maxn and pos < c.ctx_len) : (produced += 1) {
            if (last == m.eosToken()) {
                eos = true;
                break;
            }
            const before = aw.writer.buffered().len;
            try m.decodeToken(&aw.writer, last);
            if (sink) |s| try s.emit(s.ctx, aw.writer.buffered()[before..]);
            try E.step(m, &st, last, pos);
            pos += 1;
            last = @intCast(sampler.sample(scratch, st.logits, temp, rnd));
        }
    }

    return .{
        .text = try aw.toOwnedSlice(),
        .token_ids = &.{},
        .prompt_tokens = toks.len,
        .completion_tokens = produced,
        .stop = eos or produced < maxn,
    };
}
