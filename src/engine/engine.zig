//! Single-node inference engine: load a checkpoint, hold the resident dense set
//! + KV cache + expert cache, and drive token generation. This is the v0
//! assembly of every other module behind one `generate` call.

const std = @import("std");
const Io = std.Io;
const model = @import("model.zig");
const ckpt = @import("checkpoint.zig");
const attention = @import("attention.zig");
const forward = @import("forward.zig");
const moe = @import("moe.zig");
const sampler = @import("sampler.zig");
const stats = @import("../core/stats.zig");
const ExpertCacheMod = @import("expert_cache.zig");
const ExpertCache = ExpertCacheMod.ExpertCache;
const ModelConfig = model.ModelConfig;

pub const Options = struct {
    ram_budget_bytes: u64,
    pin_budget_bytes: u64,
    verify: bool = true,
};

pub const Sizes = struct {
    dense_bytes: u64,
    kv_bytes: u64,
    expert_bytes: u64,
    per_token_expert_bytes: u64,
    total_routed_experts: usize,
    unique_expert_bytes: u64,
    lru_capacity: usize,
    pinned_experts: usize,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: Io,
    loaded: ckpt.Loaded,
    cfg: ModelConfig,
    cache: ExpertCache,
    kv: attention.KVCache,
    arena: std.heap.ArenaAllocator,
    x: []f32,
    logits: []f32,
    sizes: Sizes,

    pub fn init(gpa: std.mem.Allocator, io: Io, dir_path: []const u8, opts: Options) !Engine {
        var loaded = try ckpt.load(gpa, io, dir_path);
        errdefer loaded.deinit();
        const cfg = loaded.header.config;

        const block_bytes = cfg.expertBytes();
        const dense_bytes: u64 = loaded.dense.len * @sizeOf(f32);
        const kv_elems: u64 = cfg.nLayers() * cfg.max_seq_len * (cfg.kv_lora_rank + cfg.rope_dim);
        const kv_bytes: u64 = kv_elems * @sizeOf(f32);

        // RAM budget carve-out: what's left after the resident set + KV cache is
        // split between the pinned hot set and the LRU stream cache.
        const overhead = dense_bytes + kv_bytes;
        const cache_budget: u64 = if (opts.ram_budget_bytes > overhead)
            opts.ram_budget_bytes - overhead
        else
            block_bytes; // degenerate: at least one slot
        const pin_budget = @min(opts.pin_budget_bytes, cache_budget);
        const lru_budget = cache_budget - pin_budget;
        const lru_capacity: usize = @max(1, @as(usize, @intCast(lru_budget / block_bytes)));

        var cache = try ExpertCache.init(
            gpa,
            io,
            loaded.experts_file,
            loaded.entries,
            block_bytes,
            lru_capacity,
            opts.verify,
        );
        errdefer cache.deinit();

        var kv = try attention.KVCache.init(gpa, cfg);
        errdefer kv.deinit(gpa);

        const x = try gpa.alloc(f32, cfg.hidden);
        errdefer gpa.free(x);
        const logits = try gpa.alloc(f32, cfg.vocab_size);
        errdefer gpa.free(logits);

        // unique on-disk expert bytes (post-dedup)
        var unique: u64 = 0;
        for (loaded.entries) |e| unique = @max(unique, e.offset + e.len);

        return .{
            .gpa = gpa,
            .io = io,
            .loaded = loaded,
            .cfg = cfg,
            .cache = cache,
            .kv = kv,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .x = x,
            .logits = logits,
            .sizes = .{
                .dense_bytes = dense_bytes,
                .kv_bytes = kv_bytes,
                .expert_bytes = block_bytes,
                .per_token_expert_bytes = cfg.perTokenExpertBytes(),
                .total_routed_experts = cfg.totalRoutedExperts(),
                .unique_expert_bytes = unique,
                .lru_capacity = lru_capacity,
                .pinned_experts = 0,
            },
        };
    }

    pub fn deinit(self: *Engine) void {
        self.arena.deinit();
        self.gpa.free(self.x);
        self.gpa.free(self.logits);
        self.kv.deinit(self.gpa);
        self.cache.deinit();
        self.loaded.deinit();
    }

    /// Pin the hottest experts fitting `pin_budget_bytes`.
    pub fn pinWithinBudget(self: *Engine, access_count: []const u64, pin_budget_bytes: u64) !void {
        const ids = try stats.hottestWithinBudget(self.gpa, access_count, self.sizes.expert_bytes, pin_budget_bytes);
        defer self.gpa.free(ids);
        try self.cache.pin(ids);
        self.sizes.pinned_experts = self.cache.pinnedCount();
    }

    fn state(self: *Engine) forward.State {
        return .{
            .cfg = self.cfg,
            .weights = self.loaded.weights,
            .kv = &self.kv,
            .cache = &self.cache,
            .arena = &self.arena,
            .x = self.x,
        };
    }

    /// Feed `prompt` tokens then generate up to `max_new` tokens, appending each
    /// generated token to `out`. Returns the number generated. Greedy if temp<=0.
    /// Per-token callback for streaming (fires as each token is committed).
    pub const TokenCb = struct {
        ctx: *anyopaque,
        cb: *const fn (ctx: *anyopaque, token: usize) anyerror!void,
    };

    pub fn generate(
        self: *Engine,
        prompt: []const usize,
        max_new: usize,
        temp: f32,
        seed: u64,
        out: *std.ArrayList(usize),
        on_token: ?TokenCb,
    ) !usize {
        self.kv.len = 0;
        var s = self.state();
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const sample_scratch = try self.gpa.alloc(f32, self.cfg.vocab_size);
        defer self.gpa.free(sample_scratch);

        var pos: usize = 0;
        // prefill
        for (prompt) |tok| {
            if (pos >= self.cfg.max_seq_len) return error.SequenceTooLong;
            try forward.step(&s, tok, pos, self.logits);
            pos += 1;
        }
        var last: usize = if (prompt.len == 0) 0 else sampler.sample(sample_scratch, self.logits, temp, rnd);

        var produced: usize = 0;
        while (produced < max_new) : (produced += 1) {
            if (pos >= self.cfg.max_seq_len) break;
            try out.append(self.gpa, last);
            if (on_token) |t| try t.cb(t.ctx, last);
            try forward.step(&s, last, pos, self.logits);
            pos += 1;
            last = sampler.sample(sample_scratch, self.logits, temp, rnd);
        }
        return produced;
    }
};
