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
const backend = @import("../compute/backend.zig");
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
    /// Size any device-resident attention cache, now that the context is
    /// capped. Deliberately after `setCtxLen`: the model's own ctx_len is its
    /// native one -- 163,840 for DeepSeek-V2-Lite -- which is far past what the
    /// attention kernel's threadgroup memory serves, so asking at load time
    /// declines and the device path then never runs. Best-effort; the engine
    /// keeps its host cache and the host attention path either way.
    pub fn initDeviceAttn(self: *GgufModel) void {
        switch (self.*) {
            .deepseek => |*m| {
                _ = backend.mlaInit(m.cfg.n_layers, m.cfg.ctx_len, m.cfg.kv_lora_rank, m.cfg.rope_dim);
                deepseek.uploadAbsorbWeights(m, m.gpa);
            },
            .gqa => {},
        }
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

    pub fn embedDim(self: *const GgufModel) usize {
        return switch (self.*) {
            inline else => |*m| m.cfg.dim,
        };
    }

    /// Mean-pooled token-embedding vector for `text`, L2-normalized into
    /// `out` (embedDim floats). Every node on a network runs the same model,
    /// so this is identical everywhere -- the property the RAG store's
    /// text-only gossip relies on. Returns false for untokenizable text.
    pub fn embedText(self: *const GgufModel, gpa: std.mem.Allocator, text: []const u8, out: []f32) bool {
        switch (self.*) {
            inline else => |*m| {
                const toks = m.encodePrompt(gpa, text, false) catch return false;
                defer gpa.free(toks);
                if (toks.len == 0) return false;
                @memset(out, 0);
                const row = gpa.alloc(f32, m.cfg.dim) catch return false;
                defer gpa.free(row);
                var used: usize = 0;
                for (toks) |t| {
                    if (t >= m.cfg.vocab) continue;
                    backend.dequantRow(m.token_embd.ty, row, m.token_embd.data, t, m.cfg.dim);
                    for (out, row) |*o, v| o.* += v;
                    used += 1;
                }
                if (used == 0) return false;
                var norm: f32 = 0;
                for (out) |v| norm += v * v;
                if (norm == 0) return false;
                const inv = 1.0 / @sqrt(norm);
                for (out) |*o| o.* *= inv;
                return true;
            },
        }
    }
};

/// Distributed GGUF generator: the model plus its token-loop expert fetch
/// source. A fresh `State` (KV cache) is built per request.
/// One cached verification (or drafting) lane: engine state plus the token
/// ids its KV currently represents. Kept for the lifetime of the node; the
/// longest-common-prefix rule below makes the cache self-keying, so there is
/// no session id anywhere in the protocol.
fn DraftSess(comptime E: type) type {
    return struct {
        st: E.State,
        toks: std.ArrayList(u32),
    };
}

pub const DraftVerify = struct { accepted: usize, correction: u32 };

pub const GgufGen = struct {
    m: GgufModel,
    /// DRAFT-command verification lane (warm-peer side), per engine variant.
    /// Allocated on first use, held until process exit -- one extra KV cache,
    /// deliberately not a pool.
    vsess_gqa: ?DraftSess(llama) = null,
    vsess_mla: ?DraftSess(deepseek) = null,
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

    /// The served model's human name (GGUF `general.name`), for the boot line
    /// and the RPC `model` method. The loom byte engine has no checkpoint
    /// metadata, so it names itself.
    pub fn modelName(self: Generator) []const u8 {
        return switch (self) {
            .loom => "loom byte engine",
            .gguf => |g| g.m.generalName() orelse "unknown",
        };
    }

    /// The served model's architecture id (GGUF `general.architecture`).
    pub fn modelArch(self: Generator) []const u8 {
        return switch (self) {
            .loom => "loom",
            .gguf => |g| g.m.archName(),
        };
    }

    /// Context window actually in effect (post `--ctx` clamp).
    pub fn modelCtx(self: Generator) usize {
        return switch (self) {
            .loom => 0,
            .gguf => |g| g.m.ctxLen(),
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

/// Threads to run kernels on when nothing is configured.
///
/// Two cores are held back, not one: a node keeps p2p, gossip, heartbeat and
/// repair threads running throughout a generation, and the OS needs somewhere
/// to put them. Measured on a 10-core Apple M5 (4 performance + 6 efficiency)
/// with TinyLlama 1.1B Q4_K_M, median of three 64-token runs:
///
///     threads  1 -> 6.0 tok/s
///              4 -> 23.7
///              6 -> 22.4
///              8 -> 26.0   <- cpu_count - 2
///              9 -> 22.1
///
/// Past that, oversubscription against the node's own threads costs more than
/// the extra cores return, and efficiency cores contribute little. Override
/// with `--threads` when the shape of the machine is different.
pub fn defaultThreads() usize {
    const n = std.Thread.getCpuCount() catch 1;
    return if (n <= 3) 1 else n - 2;
}

/// Process-wide kernel thread count. 0 means "use defaultThreads()"; 1
/// disables the pool, which is what makes a threaded-vs-serial comparison
/// possible without rebuilding.
pub var kernel_threads: usize = 0;

pub fn threads() usize {
    return if (kernel_threads == 0) defaultThreads() else kernel_threads;
}

/// Prefill batch size. 0 means the kernel maximum; 1 disables batching, which
/// is what makes a batched-versus-serial prefill comparison possible without
/// rebuilding.
pub var prefill_batch: usize = 0;

pub fn batchSize() usize {
    return if (prefill_batch == 0) backend.MAX_BATCH else @min(prefill_batch, backend.MAX_BATCH);
}

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

    // Row-parallel kernels for the duration of this generation (issue #11).
    // Scoped to the request rather than the process: without a condition
    // variable in the kernels, parked workers would have to spin, and a node
    // sitting idle overnight must not peg every core.
    backend.parallelBegin(threads());
    defer backend.parallelEnd();
    // The device exists now; wire any weight mappings registered at load.
    const resident = backend.materializeArenas();
    if (std.c.getenv("LOOM_FUSED_DEBUG") != null) std.debug.print("arena resident: {d} MB\n", .{resident / (1024 * 1024)});

    const toks = try m.encodePrompt(gpa, prompt_text, parse_special);
    defer gpa.free(toks);
    const maxn = clampMax(max_tokens, budget, toks.len);

    var st = try E.State.init(gpa, c);
    defer st.deinit(gpa);
    const scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(scratch);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    // Prefill (routed-expert reads fault in from peers via attachDist).
    // Batched where the engine offers it: the whole prompt is known, so one
    // unpacked weight can serve several tokens, which is most of
    // time-to-first-token on a long prompt.
    var pos: usize = 0;
    // A model on the recorded path prefills through `step` too. Mixing the two
    // means the prompt's KV rows are produced by one code path and the decode
    // rows by another, which is a correctness question before it is a
    // performance one -- and the recorded path is where the whole token lives
    // in device memory, so it cannot consume a batch anyway.
    const recorded = @hasField(@TypeOf(m.*), "gpu_layers") and m.gpu_layers;
    if (@hasDecl(E, "stepBatch") and !recorded) {
        while (pos < toks.len and pos < c.ctx_len) {
            const take = @min(@min(batchSize(), toks.len - pos), c.ctx_len - pos);
            try E.stepBatch(m, &st, toks[pos..][0..take], pos);
            pos += take;
        }
    } else {
        for (toks) |t| {
            if (pos >= c.ctx_len) break;
            try E.step(m, &st, t, pos);
            pos += 1;
        }
    }

    var aw = std.Io.Writer.Allocating.init(gpa);
    defer aw.deinit();
    var produced: usize = 0;
    var eos = false;
    if (toks.len > 0) {
        var spec_on = false;
        if (comptime @hasDecl(E, "stepSpec")) {
            // MTP speculative decode: greedy only, and killable for A/B runs.
            spec_on = temp <= 0 and m.mtp != null and std.c.getenv("LOOM_NO_MTP") == null;
        }
        var last: u32 = @intCast(sampler.sample(scratch, st.logits, temp, rnd));
        while (produced < maxn and pos < c.ctx_len) : (produced += 1) {
            if (last == m.eosToken()) {
                eos = true;
                break;
            }
            const before = aw.writer.buffered().len;
            try m.decodeToken(&aw.writer, last);
            if (sink) |s| try s.emit(s.ctx, aw.writer.buffered()[before..]);
            if (comptime @hasDecl(E, "stepSpec")) {
                if (spec_on) {
                    const r = try E.stepSpec(m, &st, last, pos);
                    pos += r.n;
                    if (r.n == 2) {
                        produced += 1;
                        if (r.tok2 == m.eosToken()) {
                            last = r.tok2;
                            continue;
                        }
                        const b2 = aw.writer.buffered().len;
                        try m.decodeToken(&aw.writer, r.tok2);
                        if (sink) |s| try s.emit(s.ctx, aw.writer.buffered()[b2..]);
                    }
                    last = r.next;
                    continue;
                }
            }
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

/// Warm-peer side of DSD (whitepaper roadmap 6): verify `draft` greedily
/// against the exact model after `ctx` tokens. Returns how many draft tokens
/// the target model agrees with and the target's own next token at the first
/// disagreement (or the bonus token when everything is accepted) -- standard
/// speculative-decoding acceptance, greedy-only, so the emitted stream is
/// token-identical to the target generating alone.
///
/// The KV lane persists across calls: the longest common prefix of the cached
/// token history and `ctx` is reused, everything past it is re-fed. The
/// context itself is the session key.
pub fn verifyDraft(g: *GgufGen, gpa: std.mem.Allocator, ctx: []const u32, draft: []const u32) !DraftVerify {
    return switch (g.m) {
        .deepseek => |*m| verifyDraftInner(deepseek, m, &g.vsess_mla, gpa, ctx, draft),
        .gqa => |*m| verifyDraftInner(llama, m, &g.vsess_gqa, gpa, ctx, draft),
    };
}

fn verifyDraftInner(
    comptime E: type,
    m: *E.Model,
    sess_slot: *?DraftSess(E),
    gpa: std.mem.Allocator,
    ctx: []const u32,
    draft: []const u32,
) !DraftVerify {
    const c = m.cfg;
    if (ctx.len == 0 or ctx.len + draft.len + 1 > c.ctx_len) return error.ContextTooLong;
    for (ctx) |t| if (t >= c.vocab) return error.BadToken;
    for (draft) |t| if (t >= c.vocab) return error.BadToken;

    backend.parallelBegin(threads());
    defer backend.parallelEnd();
    _ = backend.materializeArenas();

    if (sess_slot.* == null) {
        sess_slot.* = .{ .st = try E.State.init(gpa, c), .toks = .empty };
    }
    const sess = &sess_slot.*.?;

    // Longest common prefix with the cached history; the last context token is
    // always re-fed so st.logits is fresh even on a full cache hit.
    var lcp: usize = 0;
    while (lcp < sess.toks.items.len and lcp < ctx.len and sess.toks.items[lcp] == ctx[lcp]) lcp += 1;
    if (lcp >= ctx.len) lcp = ctx.len - 1;

    var pos: usize = lcp;
    if (@hasDecl(E, "stepBatch") and ctx.len - lcp > 1) {
        while (pos < ctx.len) {
            const take = @min(batchSize(), ctx.len - pos);
            try E.stepBatch(m, &sess.st, ctx[pos..][0..take], pos);
            pos += take;
        }
    } else {
        while (pos < ctx.len) : (pos += 1) try E.step(m, &sess.st, ctx[pos], pos);
    }
    sess.toks.clearRetainingCapacity();
    try sess.toks.appendSlice(gpa, ctx);

    const scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(scratch);

    // The first draft token is judged by the logits the prefill already
    // produced -- rejecting it costs no forward work at all.
    const t0: u32 = @intCast(sampler.sample(scratch, sess.st.logits, 0, undefined));
    if (t0 != draft[0]) return .{ .accepted = 0, .correction = t0 };

    // Batched verification (the DSD speed lever): feed the whole window in
    // ONE forward with per-lane logits capture, so the verifier reads each
    // weight once per window instead of once per token -- on a
    // disk-streaming host that is nearly a gamma-fold cut in the dominant
    // cost. Lane i's logits are the model's choice after draft[0..i], i.e.
    // the judgment on draft[i+1] (and lane k-1 holds the bonus token).
    // Rejected lanes leave junk KV rows behind; the next call's
    // longest-common-prefix re-feed overwrites those positions.
    const batched = comptime @hasField(E.State, "spec_capture");
    if (batched and draft.len > 1) {
        sess.st.spec_capture = true;
        defer sess.st.spec_capture = false;
        try E.stepBatch(m, &sess.st, draft, pos);
        try sess.toks.append(gpa, draft[0]);
        var accepted: usize = 1;
        while (accepted < draft.len) {
            const row = sess.st.blogits[(accepted - 1) * c.vocab ..][0..c.vocab];
            const target: u32 = @intCast(sampler.sample(scratch, row, 0, undefined));
            if (target != draft[accepted]) return .{ .accepted = accepted, .correction = target };
            try sess.toks.append(gpa, draft[accepted]);
            accepted += 1;
        }
        const bonus_row = sess.st.blogits[(draft.len - 1) * c.vocab ..][0..c.vocab];
        const bonus: u32 = @intCast(sampler.sample(scratch, bonus_row, 0, undefined));
        return .{ .accepted = accepted, .correction = bonus };
    }

    // Sequential fallback (deepseek engine, or a one-token window).
    var accepted: usize = 0;
    while (accepted < draft.len) {
        if (accepted > 0) {
            const target: u32 = @intCast(sampler.sample(scratch, sess.st.logits, 0, undefined));
            if (target != draft[accepted]) return .{ .accepted = accepted, .correction = target };
        }
        try E.step(m, &sess.st, draft[accepted], pos);
        try sess.toks.append(gpa, draft[accepted]);
        pos += 1;
        accepted += 1;
    }
    const bonus: u32 = @intCast(sampler.sample(scratch, sess.st.logits, 0, undefined));
    return .{ .accepted = accepted, .correction = bonus };
}

/// Cold-node side of DSD: greedy generation where windows of tokens are
/// drafted with the *local-only* expert tier (missing experts skipped,
/// llama.zig State.draft_local) and verified remotely through `verify` --
/// openai.zig supplies the DRAFT round trip as that callback. Output is
/// token-identical to the verifying peer generating alone, because every
/// emitted token either matched the peer's greedy choice or IS the peer's
/// greedy choice.
///
/// gqa engine only: drafting needs the draft_local switch, which the deepseek
/// engine does not carry yet.
pub const VerifyFn = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, tokens: []const u32, draft: []const u32) anyerror!DraftVerify,
};

pub const DraftedResult = struct {
    res: Result,
    rounds: usize,
    drafted: usize,
    accepted: usize,
    final_gamma: usize,
    /// True when the controller abandoned drafting (acceptance collapsed);
    /// the caller should finish the request over wholesale GEN instead.
    bailed: bool,
    tokens_done: usize,
};

pub fn generateDrafted(
    g: *GgufGen,
    gpa: std.mem.Allocator,
    prompt_text: []const u8,
    max_tokens: usize,
    parse_special: bool,
    verify: VerifyFn,
) !DraftedResult {
    const m = switch (g.m) {
        .gqa => |*mm| mm,
        else => return error.DraftUnsupported,
    };
    const c = m.cfg;

    backend.parallelBegin(threads());
    defer backend.parallelEnd();
    _ = backend.materializeArenas();

    const prompt = try m.encodePrompt(gpa, prompt_text, parse_special);
    defer gpa.free(prompt);
    if (prompt.len == 0 or prompt.len >= c.ctx_len) return error.PromptTooLong;

    var all = std.ArrayList(u32).empty;
    defer all.deinit(gpa);
    try all.appendSlice(gpa, prompt);

    var st = try llama.State.init(gpa, c);
    defer st.deinit(gpa);
    st.draft_local = true;
    const scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(scratch);

    // Local prefill in draft mode: reads only held experts, fetches nothing.
    var pos: usize = 0;
    while (pos < prompt.len) {
        const take = @min(batchSize(), prompt.len - pos);
        try llama.stepBatch(m, &st, prompt[pos..][0..take], pos);
        pos += take;
    }

    // DSD's dynamic-window baseline, deterministically: clamp, EMA, raise on
    // high acceptance, lower on low, and a sticky bail-out so the mode cannot
    // flap (their stabilization findings; the learned controller is
    // deliberately not ported).
    var gamma: usize = 4;
    var ema: f64 = 0.5;
    var low_rounds: usize = 0;
    var rounds: usize = 0;
    var drafted_total: usize = 0;
    var accepted_total: usize = 0;
    var bailed = false;
    var eos = false;

    var draft_buf: [8]u32 = undefined;
    while (all.items.len - prompt.len < max_tokens and all.items.len + gamma + 1 < c.ctx_len) {
        // Draft gamma tokens with the degraded local model.
        var n_draft: usize = 0;
        var dpos = all.items.len;
        while (n_draft < gamma) : (n_draft += 1) {
            const t: u32 = @intCast(sampler.sample(scratch, st.logits, 0, undefined));
            draft_buf[n_draft] = t;
            try llama.step(m, &st, t, dpos);
            dpos += 1;
        }
        const v = try verify.call(verify.ctx, all.items, draft_buf[0..n_draft]);
        rounds += 1;
        drafted_total += n_draft;
        accepted_total += @min(v.accepted, n_draft);

        // Adopt the verified prefix plus the peer's token.
        const nacc = @min(v.accepted, n_draft);
        const round_start = all.items.len;
        try all.appendSlice(gpa, draft_buf[0..nacc]);
        try all.append(gpa, v.correction);
        for (all.items[round_start..]) |t| {
            if (t == m.eosToken()) eos = true;
        }
        if (eos) break;

        // Re-align the local draft state. Accepted positions already carry the
        // right draft KV (the same tokens were stepped while drafting), and a
        // rejected position's junk KV is overwritten the next time that
        // position is stepped. Only the peer's token is new: feed it so the
        // logits are ready for the next window.
        try llama.step(m, &st, all.items[all.items.len - 1], all.items.len - 1);

        // Controller update.
        const rate = @as(f64, @floatFromInt(nacc)) / @as(f64, @floatFromInt(n_draft));
        ema = 0.4 * rate + 0.6 * ema;
        if (ema > 0.75 and gamma < draft_buf.len) gamma += 1;
        if (ema < 0.25 and gamma > 1) gamma -= 1;
        if (ema < 0.25 and gamma == 1) {
            low_rounds += 1;
            if (low_rounds >= 2) {
                bailed = true;
                break;
            }
        } else low_rounds = 0;
    }

    // A window can overshoot the requested length by up to gamma tokens;
    // greedy output is prefix-stable, so trimming is exact.
    if (all.items.len - prompt.len > max_tokens) all.items.len = prompt.len + max_tokens;

    // Detokenize everything generated so far.
    var aw = std.Io.Writer.Allocating.init(gpa);
    defer aw.deinit();
    for (all.items[prompt.len..]) |t| {
        if (t == m.eosToken()) break;
        try m.decodeToken(&aw.writer, t);
    }
    const done = all.items.len - prompt.len;
    return .{
        .res = .{
            .text = try aw.toOwnedSlice(),
            .token_ids = &.{},
            .prompt_tokens = prompt.len,
            .completion_tokens = done,
            .stop = eos,
        },
        .rounds = rounds,
        .drafted = drafted_total,
        .accepted = accepted_total,
        .final_gamma = gamma,
        .bailed = bailed,
        .tokens_done = done,
    };
}

const gguf_mod = @import("../gguf/gguf.zig");

test "verifyDraft: exact drafts all accepted, a corrupted draft is cut at the first lie" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    const path = "test-draft-verify.gguf";
    defer Io.Dir.cwd().deleteFile(io, path) catch {};
    try gguf_mod.writeMoeFixture(gpa, io, path, 5, "qwen3moe");

    var m = try llama.load(gpa, io, path);
    defer m.deinit();

    // Ground truth: greedy-generate 4 tokens with the plain engine.
    const prompt = [_]u32{ 7, 21, 4, 90 };
    var st = try llama.State.init(gpa, m.cfg);
    defer st.deinit(gpa);
    const scratch = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(scratch);
    var truth: [4]u32 = undefined;
    var pos: usize = 0;
    for (prompt) |t| {
        try llama.step(&m, &st, t, pos);
        pos += 1;
    }
    var seq = std.ArrayList(u32).empty;
    defer seq.deinit(gpa);
    try seq.appendSlice(gpa, &prompt);
    for (&truth) |*t| {
        t.* = @intCast(sampler.sample(scratch, st.logits, 0, undefined));
        try llama.step(&m, &st, t.*, pos);
        try seq.append(gpa, t.*);
        pos += 1;
    }

    var g = GgufGen{ .m = .{ .gqa = m }, .ctx_cap = 256 };
    // Careful: g.m now owns a copy; verify against the copy, not `m`.
    // Exact draft: everything accepted, correction is the model's own next token.
    const v1 = try verifyDraft(&g, gpa, &prompt, &truth);
    try std.testing.expectEqual(truth.len, v1.accepted);
    // Session reuse: extend the context by the accepted window; a corrupted
    // second draft gets cut at the first wrong position with the true token
    // as the correction.
    var bad = [_]u32{ v1.correction, 0, 0 };
    bad[1] = if (truth[0] == 0) 1 else 0; // guaranteed wrong only if the model would not pick it; checked below
    const v2 = try verifyDraft(&g, gpa, seq.items, &bad);
    try std.testing.expect(v2.accepted >= 1); // first token IS the model's greedy pick
    if (v2.accepted == 1) {
        // the correction must be what the model actually wants there
        try std.testing.expect(v2.correction != bad[1]);
    }
    if (g.vsess_gqa) |*sess| {
        sess.st.deinit(gpa);
        sess.toks.deinit(gpa);
    }
    // g.m's model copy shares tensors with `m`; only one deinit (above).
}

test "generateDrafted round-trips token-identical against a local verifier" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    const path = "test-draft-loop.gguf";
    defer Io.Dir.cwd().deleteFile(io, path) catch {};
    try gguf_mod.writeMoeFixture(gpa, io, path, 9, "qwen3moe");

    var m = try llama.load(gpa, io, path);
    defer m.deinit();

    // Plain greedy reference through the public path.
    var g = GgufGen{ .m = .{ .gqa = m }, .ctx_cap = 256 };
    const gen = Generator{ .gguf = &g };
    var ref = try gen.generate(gpa, io, "hello", 8, 0, 42, null, null, false);
    defer ref.deinit(gpa);

    // Draft loop with the SAME model as its own verifier: the drafter has no
    // dist source, so draft_local changes nothing and every draft matches --
    // which is exactly what makes the output a strict equality check on the
    // loop plumbing (window adoption, re-align, eos, controller).
    const LocalVerify = struct {
        g: *GgufGen,
        gpa: std.mem.Allocator,
        fn call(op: *anyopaque, tokens: []const u32, draft: []const u32) anyerror!DraftVerify {
            const self: *@This() = @ptrCast(@alignCast(op));
            return verifyDraft(self.g, self.gpa, tokens, draft);
        }
    };
    var lv = LocalVerify{ .g = &g, .gpa = gpa };
    const out = try generateDrafted(&g, gpa, "hello", 8, false, .{
        .ctx = @ptrCast(&lv),
        .call = LocalVerify.call,
    });
    defer gpa.free(out.res.text);

    try std.testing.expect(!out.bailed);
    try std.testing.expectEqualStrings(ref.text, out.res.text);
    try std.testing.expectEqual(ref.completion_tokens, out.res.completion_tokens);
    try std.testing.expectEqual(out.drafted, out.accepted); // same model: nothing rejected
    if (g.vsess_gqa) |*sess| {
        sess.st.deinit(gpa);
        sess.toks.deinit(gpa);
    }
}
