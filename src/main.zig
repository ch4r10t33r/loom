//! Loom v0 — single-node expert-streaming MoE inference.
//!
//! Subcommands:
//!   gen <dir> [--glm] [--seed N]         write a synthetic checkpoint
//!   info <dir>                           print manifest + verify Merkle root
//!   iobench <file> [opts]                disk profile (parallel block reads)
//!   run <dir> [opts]                     generate tokens, log tok/s + hit-rate

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

pub const model = @import("engine/model.zig");
pub const hash = @import("core/hash.zig");
const banner = @import("core/banner.zig");
pub const quant = @import("core/quant.zig");
pub const tensor = @import("core/tensor.zig");
pub const checkpoint = @import("engine/checkpoint.zig");
pub const expert_cache = @import("engine/expert_cache.zig");
pub const attention = @import("engine/attention.zig");
pub const moe = @import("engine/moe.zig");
pub const forward = @import("engine/forward.zig");
pub const engine = @import("engine/engine.zig");
pub const gen_checkpoint = @import("engine/gen_checkpoint.zig");
pub const iobench = @import("core/iobench.zig");
pub const sampler = @import("engine/sampler.zig");
pub const tokenizer = @import("engine/tokenizer.zig");
pub const stats = @import("core/stats.zig");
pub const node = @import("node/node.zig");
pub const hf = @import("node/hf.zig");
pub const rpc = @import("node/rpc.zig");
pub const p2p = @import("p2p/p2p.zig");
pub const gguf = @import("gguf/gguf.zig");
pub const weights = @import("p2p/weights.zig");
pub const sync = @import("p2p/sync.zig");
pub const peers = @import("p2p/peers.zig");
pub const gossip = @import("p2p/gossip.zig");
pub const backend = @import("compute/backend.zig");
const generator = @import("node/generator.zig");
pub const llama = @import("gguf/llama.zig");
pub const deepseek = @import("gguf/deepseek.zig");
pub const expert_fetch = @import("p2p/expert_fetch.zig");
pub const bpe = @import("gguf/bpe.zig");
pub const bootnode = @import("p2p/bootnode.zig");
pub const wire = @import("p2p/wire.zig");
pub const light = @import("node/light.zig");
pub const light_openai = @import("node/light_openai.zig");
pub const meter = @import("node/meter.zig");

const GB: f64 = 1024.0 * 1024.0 * 1024.0;
const MB: usize = 1024 * 1024;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var obuf: [4096]u8 = undefined;
    var out_file = Io.File.stdout().writer(io, &obuf);
    const out = &out_file.interface;
    defer out.flush() catch {};

    // The Args iterator is the one portable surface: Windows hands the
    // command line over as WTF-16 and the iterator decodes it; POSIX yields
    // the argv strings as-is.
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args_list = std.ArrayList([]const u8).empty;
    defer args_list.deinit(gpa);
    while (args_it.next()) |a| try args_list.append(gpa, a);
    const args = args_list.items;

    if (args.len < 2) {
        try usage(out);
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "gen")) {
        try cmdGen(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try cmdInfo(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "iobench")) {
        try cmdIobench(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "run")) {
        try cmdRun(gpa, io, out, args, init.environ_map);
    } else if (std.mem.eql(u8, cmd, "node")) {
        try cmdNode(gpa, io, out, args, init.environ_map);
    } else if (std.mem.eql(u8, cmd, "gguf")) {
        try cmdGguf(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "light")) {
        try cmdLight(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try @import("bench.zig").run(gpa, io, out, args);
    } else if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version") or std.mem.eql(u8, cmd, "-V")) {
        try cmdVersion(out);
    } else {
        try out.print("unknown command: {s}\n\n", .{cmd});
        try usage(out);
    }
}

/// Report what this binary is. Release builds are cross-compiled and
/// downloaded, so the target triple matters as much as the version: it is the
/// first thing to check when a binary misbehaves on unexpected hardware.
fn cmdVersion(out: *Io.Writer) !void {
    const info = @import("build_info");
    // Full hash here, unlike the boot banner's short form: this is the command
    // you run to paste into a bug report.
    try out.print("loom {s} ({s})\n", .{ info.version, info.commit });
    try out.print("  target   {s}-{s}-{s}\n", .{
        @tagName(builtin.cpu.arch), @tagName(builtin.os.tag), @tagName(builtin.abi),
    });
    try out.print("  zig      {s}\n", .{builtin.zig_version_string});
    try out.print("  mode     {s}\n", .{@tagName(builtin.mode)});
}

fn usage(out: *Io.Writer) !void {
    const info = @import("build_info");
    try out.print("loom {s} — distributed expert cache for large MoE inference\n\n", .{info.version});
    try out.print(
        \\usage:
        \\
        \\usage:
        \\  loom node [--model SPEC] [--rpc-addr A] [--rpc-port P]
        \\            [--openai-addr A] [--openai-port P] [--ctx N] [--chat-format F]
        \\            [--p2p-addr A] [--p2p-port P] [--ram-gb X] [--pin-gb Y]
        \\            [--seed S] [--stats FILE] [--no-verify]
        \\            [--gguf FILE | --bootstrap HOST:PORT]
        \\            [--peers H:P,H:P,...] [--hold-fraction F] [--range-mb M]
        \\            [--advertise HOST:PORT] [--network devnet|testnet|mainnet] [--network-id N]
        \\            [--rag] [--rag-k N] [--r-target N] [--free-quota TOKENS] [--admin-token TOK]
        \\            [--ui-addr A] [--ui-port P] [--status-secs N] [--threads N] [--batch N]
        \\  loom light [--full-nodes H:P[,...]] [--openai-port P --openai-full-nodes H:P[,...]]
        \\             [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--client-id ID]
        \\  loom gguf gen <file> [--seed N] [--data-mb M] [--arch deepseek2|llama|qwen2moe|qwen3moe|glm4moe]
        \\  loom gguf check <file | https://...>
        \\  loom gguf info <file> [--range-mb M]
        \\  loom gguf shard <file>
        \\  loom gguf run <file.gguf | store-dir> [--prompt STR] [--max-tokens N] [--temp T]
        \\                [--seed S] [--ctx N] [--threads N] [--batch N] [--committee H:P,...] [--peers H:P,...]
        \\  loom gen <dir> [--glm] [--seed N]
        \\  loom info <dir>
        \\  loom iobench <file> [--threads N] [--block-mb M] [--reads R]
        \\  loom run <dir> [--prompt STR] [--max-tokens N] [--ram-gb X]
        \\                 [--pin-gb Y] [--temp T] [--seed S] [--stats FILE] [--no-verify]
        \\  loom bench [--json] [--check]
        \\  loom version
        \\
        \\--model SPEC: <local dir> | tiny | [hf:]org/repo[@rev]   (default: tiny)
        \\env overrides: MODEL, RAM_BUDGET_GB, PIN_GB, MAX_TOKENS, TEMP, SEED, STATS
        \\
    , .{});
}

// ---- gen -------------------------------------------------------------------

fn cmdGen(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    if (args.len < 3) return out.print("gen: need <dir>\n", .{});
    const dir = args[2];
    const cfg = if (hasFlag(args, "--glm")) model.glmShape() else model.tinyShape();
    const seed = try flagU64(args, "--seed", 42);

    try out.print("generating {s} checkpoint -> {s}\n", .{ if (hasFlag(args, "--glm")) "GLM-5.2" else "tiny", dir });
    try out.print("  experts: {d} routed ({d} layers x {d}), {d} bytes each, {d} per-token reads\n", .{
        cfg.totalRoutedExperts(),
        cfg.n_moe_layers,
        cfg.n_experts,
        cfg.expertBytes(),
        cfg.n_moe_layers * cfg.n_routed,
    });
    try out.flush();

    gen_checkpoint.generate(gpa, io, dir, cfg, seed) catch |e| {
        try out.print("gen failed: {s}\n", .{@errorName(e)});
        return;
    };
    try out.print("done. dense {d} f32, per-token working set {d:.3} MB\n", .{
        checkpoint.denseElemCount(cfg),
        @as(f64, @floatFromInt(cfg.perTokenExpertBytes())) / @as(f64, MB),
    });
}

// ---- info ------------------------------------------------------------------

fn cmdInfo(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    if (args.len < 3) return out.print("info: need <dir>\n", .{});
    var loaded = try checkpoint.load(gpa, io, args[2]);
    defer loaded.deinit();
    const c = loaded.header.config;

    try out.print("checkpoint {s}\n", .{args[2]});
    try out.print("  hidden={d} layers={d} (dense {d} + moe {d}) heads={d} head_dim={d}\n", .{
        c.hidden, c.nLayers(), c.n_dense_layers, c.n_moe_layers, c.n_heads, c.headDim(),
    });
    try out.print("  experts={d}/layer, top-{d} routed + {d} shared; kv_lora={d} rope={d}\n", .{
        c.n_experts, c.n_routed, c.n_shared, c.kv_lora_rank, c.rope_dim,
    });
    try out.print("  expert_bytes={d} total_routed={d} per_token={d:.3} MB\n", .{
        c.expertBytes(), c.totalRoutedExperts(), @as(f64, @floatFromInt(c.perTokenExpertBytes())) / @as(f64, MB),
    });
    const root_hex = hash.toHex(loaded.header.merkle_root);
    try out.print("  merkle_root={s}\n", .{root_hex});

    // recompute Merkle root over the index and verify
    var leaves = try gpa.alloc(hash.Digest, loaded.entries.len);
    defer gpa.free(leaves);
    for (loaded.entries, 0..) |e, i| leaves[i] = e.digest;
    const recomputed = try hash.merkleRoot(gpa, leaves);
    try out.print("  merkle_check={s}\n", .{if (hash.eql(recomputed, loaded.header.merkle_root)) "OK" else "MISMATCH"});

    // spot-verify one expert block against its digest
    if (loaded.entries.len > 0) {
        const e = loaded.entries[0];
        const buf = try gpa.alloc(u8, e.len);
        defer gpa.free(buf);
        _ = try loaded.experts_file.readPositionalAll(io, buf, e.offset);
        const got = hash.hashBlock(buf);
        try out.print("  expert0_digest_check={s}\n", .{if (hash.eql(got, e.digest)) "OK" else "MISMATCH"});
    }
}

// ---- iobench ---------------------------------------------------------------

fn cmdIobench(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    if (args.len < 3) return out.print("iobench: need <file>\n", .{});
    const path = args[2];
    const default_threads = std.Thread.getCpuCount() catch 4;
    const threads = try flagUsize(args, "--threads", default_threads);
    const block_mb = try flagUsize(args, "--block-mb", 19);
    const reads = try flagUsize(args, "--reads", 64);

    try out.print("iobench {s}: {d} threads x {d} reads of {d} MB\n", .{ path, threads, reads, block_mb });
    try out.flush();

    const res = iobench.run(gpa, io, path, threads, block_mb * MB, reads, 1234) catch |e| {
        try out.print("iobench failed: {s}\n", .{@errorName(e)});
        return;
    };
    try out.print("  {d} reads, {d:.2} GB in {d:.3} s => {d:.2} GB/s\n", .{
        res.total_reads,
        @as(f64, @floatFromInt(res.bytes)) / 1e9,
        @as(f64, @floatFromInt(res.ns)) / 1e9,
        res.gbps(),
    });
    try out.print("  per-token working set at this bandwidth: see `info` per_token / GB/s\n", .{});
}

// ---- run -------------------------------------------------------------------

fn cmdRun(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8, env: *std.process.Environ.Map) !void {
    const dir = if (args.len >= 3 and !std.mem.startsWith(u8, args[2], "--"))
        args[2]
    else
        (env.get("MODEL") orelse return out.print("run: need <dir> or MODEL env\n", .{}));

    const prompt_str = flagStr(args, "--prompt") orelse "Loom";
    const max_tokens = try flagUsize(args, "--max-tokens", try envUsize(env, "MAX_TOKENS", 32));
    const ram_gb = try flagF64(args, "--ram-gb", try envF64(env, "RAM_BUDGET_GB", defaultRamGb()));
    const pin_gb = try flagF64(args, "--pin-gb", try envF64(env, "PIN_GB", 0.0));
    const temp = try flagF64(args, "--temp", try envF64(env, "TEMP", 0.0));
    const seed = try flagU64(args, "--seed", try envU64(env, "SEED", 42));
    const stats_path = flagStr(args, "--stats") orelse env.get("STATS");
    const verify = !hasFlag(args, "--no-verify");

    const ram_bytes: u64 = @intFromFloat(ram_gb * GB);
    const pin_bytes: u64 = @intFromFloat(pin_gb * GB);

    var eng = engine.Engine.init(gpa, io, dir, .{
        .ram_budget_bytes = ram_bytes,
        .pin_budget_bytes = pin_bytes,
        .verify = verify,
    }) catch |e| {
        try out.print("load failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer eng.deinit();

    // measure-then-pin: if a STATS counts file already exists and PIN>0, pin the
    // hot set before generating.
    if (stats_path) |sp| if (pin_bytes > 0) {
        if (try stats.readCounts(gpa, io, sp, eng.cache.access_count.len)) |counts| {
            defer gpa.free(counts);
            try eng.pinWithinBudget(counts, pin_bytes);
        }
    };

    const s = eng.sizes;
    try out.print("loaded {s}\n", .{dir});
    try out.print("  ram_budget={d:.2} GB dense={d:.3} GB kv={d:.3} GB\n", .{
        ram_gb, @as(f64, @floatFromInt(s.dense_bytes)) / GB, @as(f64, @floatFromInt(s.kv_bytes)) / GB,
    });
    try out.print("  expert_bytes={d} unique_on_disk={d:.3} GB lru_capacity={d} pinned={d}\n", .{
        s.expert_bytes, @as(f64, @floatFromInt(s.unique_expert_bytes)) / GB, s.lru_capacity, s.pinned_experts,
    });
    try out.flush();

    const prompt = try tokenizer.encode(gpa, prompt_str);
    defer gpa.free(prompt);

    var produced = std.ArrayList(usize).empty;
    defer produced.deinit(gpa);

    const t0 = stats.nowNs(io);
    const n = eng.generate(prompt, max_tokens, @floatCast(temp), seed, &produced, null) catch |e| {
        try out.print("generate failed: {s}\n", .{@errorName(e)});
        return;
    };
    const t1 = stats.nowNs(io);

    // decode output bytes
    try out.print("prompt: {s}\n", .{prompt_str});
    try out.print("output: ", .{});
    for (produced.items) |tok| {
        const b = tokenizer.decodeByte(tok);
        // keep terminal sane: show printable bytes, escape others
        if (b >= 32 and b < 127) {
            try out.print("{c}", .{b});
        } else {
            try out.print("\\x{x:0>2}", .{b});
        }
    }
    try out.print("\n", .{});

    const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const cs = eng.cache.stats;
    try out.print("---- stats ----\n", .{});
    try out.print("  tokens={d} time={d:.3}s tok/s={d:.2}\n", .{ n, secs, @as(f64, @floatFromInt(n)) / secs });
    try out.print("  expert accesses={d} pin_hits={d} lru_hits={d} disk_misses={d}\n", .{
        cs.accesses(), cs.pin_hits, cs.lru_hits, cs.disk_misses,
    });
    try out.print("  hit_rate={d:.3} bytes_read={d:.3} MB digest_failures={d}\n", .{
        cs.hitRate(), @as(f64, @floatFromInt(cs.bytes_read)) / @as(f64, MB), cs.digest_failures,
    });
    if (cs.warmup_reads > 0) try out.print("  warmup(pin) reads={d} bytes={d:.3} MB\n", .{
        cs.warmup_reads, @as(f64, @floatFromInt(cs.warmup_bytes)) / @as(f64, MB),
    });
    try out.print("  rss={d:.3} GB\n", .{@as(f64, @floatFromInt(stats.rssBytes())) / GB});

    // persist usage for the next run's PIN
    if (stats_path) |sp| {
        try stats.writeCounts(io, sp, eng.cache.access_count);
        var hbuf: [4096]u8 = undefined;
        const hist = try std.fmt.bufPrint(&hbuf, "{s}.txt", .{sp});
        try stats.writeUsageHistogram(gpa, io, hist, eng.cache.access_count, s.expert_bytes);
        try out.print("  wrote usage -> {s} (+ {s})\n", .{ sp, hist });
    }
}

// ---- node ------------------------------------------------------------------

/// Default RAM budget for the expert cache: a quarter of the machine's memory,
/// clamped to [0.5, 8] GB.
///
/// The old default was a flat 4 GB regardless of the machine, which OOM-killed
/// an origin on an 8 GB box -- 7.4 GB of anonymous RSS on a 7.7 GB machine,
/// dead before it served a request. A cache budget has to be a fraction of what
/// exists, not a constant: on a 64 GB box 4 GB was needlessly small, and on an
/// 8 GB box it was fatal. A quarter leaves room for the model's own mappings,
/// the page cache the store is read through, and the rest of the system.
fn defaultRamGb() f64 {
    const total = std.process.totalSystemMemory() catch return 2.0;
    const gb = @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0 * 1024.0);
    return std.math.clamp(gb / 4.0, 0.5, 8.0);
}

const NODE_FLAGS = [_][]const u8{
    "--model",          "--rpc-addr",     "--rpc-port",       "--openai-addr", "--openai-port",
    "--p2p-addr",       "--p2p-port",     "--ram-gb",         "--pin-gb",      "--seed",
    "--stats",          "--no-verify",    "--gguf",           "--bootstrap",   "--hold-fraction",
    "--range-mb",       "--peers",        "--advertise",      "--network-id",  "--network",
    "--rag",            "--rag-k",        "--r-target",       "--free-quota",  "--admin-token",
    "--ctx",            "--ui-addr",      "--ui-port",        "--status-secs", "--threads",
    "--mmap-weights",   "--gpu-ops",      "--no-gpu-layers",  "--batch",       "--chat-format",
    "--report-metrics", "--alpha-ingest", "--delegate-below",
};

/// A flag-shaped argument this command does not know is an error, not a
/// no-op: `loom node --help` over ssh once started a stray synthetic node
/// that shared the bootnode's port (issues #179/#180).
fn checkNodeFlags(out: *Io.Writer, args: [][]const u8) !bool {
    for (args) |a| {
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            try out.print("usage: loom node [flags]\n\nflags:\n", .{});
            for (NODE_FLAGS) |f| try out.print("  {s}\n", .{f});
            try out.print("\nEvery flag, with defaults and when to change it: docs/CLI.md\n", .{});
            return false;
        }
    }
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (!std.mem.startsWith(u8, a, "-")) continue;
        var known = false;
        for (NODE_FLAGS) |f| {
            if (std.mem.eql(u8, a, f)) known = true;
        }
        if (!known) {
            try out.print("node: unknown flag {s} (try --help, or docs/CLI.md)\n", .{a});
            return false;
        }
    }
    return true;
}

fn cmdNode(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8, env: *std.process.Environ.Map) !void {
    if (!try checkNodeFlags(out, args)) return;
    const model_spec = flagStr(args, "--model") orelse env.get("MODEL") orelse "tiny";
    const rpc_addr = flagStr(args, "--rpc-addr") orelse "127.0.0.1";
    const rpc_port = try flagU16(args, "--rpc-port", 8770);
    const openai_addr = flagStr(args, "--openai-addr") orelse rpc_addr;
    const openai_port = try flagU16(args, "--openai-port", 0); // 0 = disabled
    const p2p_addr = flagStr(args, "--p2p-addr") orelse "0.0.0.0";
    const p2p_port = try flagU16(args, "--p2p-port", 8771);
    const ram_gb = try flagF64(args, "--ram-gb", try envF64(env, "RAM_BUDGET_GB", defaultRamGb()));
    const pin_gb = try flagF64(args, "--pin-gb", try envF64(env, "PIN_GB", 0.0));
    const seed = try flagU64(args, "--seed", try envU64(env, "SEED", 42));
    const stats_path = flagStr(args, "--stats") orelse env.get("STATS");
    const verify = !hasFlag(args, "--no-verify");

    // model cache under $HOME/.cache/loom/models (fallback to ./.loom-cache)
    const cache_root = if (env.get("HOME")) |home|
        try std.fmt.allocPrint(gpa, "{s}/.cache/loom/models", .{home})
    else
        try gpa.dupe(u8, "./.loom-cache");
    defer gpa.free(cache_root);

    const gguf_path = flagStr(args, "--gguf");
    const bootstrap = flagStr(args, "--bootstrap");
    if (gguf_path != null and bootstrap != null) {
        return out.print("node: --gguf and --bootstrap are mutually exclusive\n", .{});
    }
    const hold_fraction = try flagF64(args, "--hold-fraction", 1.0);
    const range_mb = try flagF64(args, "--range-mb", 4.0);

    try node.run(gpa, io, out, .{
        .model = model_spec,
        .rpc_addr = rpc_addr,
        .rpc_port = rpc_port,
        .openai_addr = openai_addr,
        .openai_port = openai_port,
        .p2p_addr = p2p_addr,
        .p2p_port = p2p_port,
        .ram_bytes = @intFromFloat(ram_gb * GB),
        .pin_bytes = @intFromFloat(pin_gb * GB),
        .verify = verify,
        .seed = seed,
        .stats_path = stats_path,
        .cache_root = cache_root,
        .gguf_path = gguf_path,
        .bootstrap = bootstrap,
        .peers = flagStr(args, "--peers"),
        .advertise = flagStr(args, "--advertise"),
        .network_id = if (flagStr(args, "--network-id")) |v| (std.fmt.parseInt(u64, v, 10) catch null) else null,
        .network_name = flagStr(args, "--network"),
        .rag = hasFlag(args, "--rag"),
        .rag_k = try flagUsize(args, "--rag-k", 3),
        .r_target = @intCast(try flagU64(args, "--r-target", 2)),
        .free_quota = try flagU64(args, "--free-quota", 100_000),
        .admin_token = flagStr(args, "--admin-token") orelse "",
        .hold_fraction = @floatCast(std.math.clamp(hold_fraction, 0.0, 1.0)),
        .range_bytes = @intFromFloat(range_mb * @as(f64, MB)),
        .ctx_cap = try flagUsize(args, "--ctx", 4096),
        .ui_addr = flagStr(args, "--ui-addr") orelse "127.0.0.1",
        .ui_port = try flagU16(args, "--ui-port", @intCast(try envU64(env, "UI_PORT", 8555))),
        .status_secs = @intCast(try flagUsize(args, "--status-secs", 30)),
        .kernel_threads = try flagUsize(args, "--threads", 0),
        .mmap_weights = hasFlag(args, "--mmap-weights"),
        .gpu_ops = hasFlag(args, "--gpu-ops"),
        .no_gpu_layers = hasFlag(args, "--no-gpu-layers"),
        .prefill_batch = try flagUsize(args, "--batch", 0),
        .chat_format = flagStr(args, "--chat-format"),
        .report_metrics = hasFlag(args, "--report-metrics"),
        .alpha_ingest = flagStr(args, "--alpha-ingest"),
        .delegate_below = std.math.clamp(try flagF64(args, "--delegate-below", 0.5), 0.0, 1.0),
    });
}

// ---- light -----------------------------------------------------------------

/// Light node (SPEC.md node classes): no weights, no engine — a local RPC that
/// delegates to full nodes and gets metered by them.
fn parseBackends(gpa: std.mem.Allocator, out: *Io.Writer, csv: []const u8, label: []const u8) !?std.ArrayList(sync.PeerAddr) {
    var backends = std.ArrayList(sync.PeerAddr).empty;
    errdefer backends.deinit(gpa);
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |tok| {
        const t = std.mem.trim(u8, tok, " ");
        if (t.len == 0) continue;
        const a = sync.PeerAddr.parse(t) catch {
            try out.print("bad {s} entry: {s}\n", .{ label, t });
            backends.deinit(gpa);
            return null;
        };
        try backends.append(gpa, a);
    }
    return backends;
}

fn lightOpenaiThread(ctx: *light_openai.Ctx) void {
    light_openai.serve(ctx) catch |e| std.debug.print("light-openai: {s}\n", .{@errorName(e)});
}

fn cmdLight(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    try banner.print(out, "light");

    const rpc_addr = flagStr(args, "--rpc-addr") orelse "127.0.0.1";
    const rpc_port = try flagU16(args, "--rpc-port", 8768);
    const openai_addr = flagStr(args, "--openai-addr") orelse rpc_addr;
    const openai_port = try flagU16(args, "--openai-port", 0); // 0 = OpenAI off
    const client_id = flagStr(args, "--client-id") orelse "light-anon";

    // Native RPC backends (--full-nodes) and/or OpenAI backends
    // (--openai-full-nodes). At least one serving surface must be configured.
    var rpc_backends: ?std.ArrayList(sync.PeerAddr) = null;
    defer if (rpc_backends) |*b| b.deinit(gpa);
    if (flagStr(args, "--full-nodes")) |csv| {
        rpc_backends = (try parseBackends(gpa, out, csv, "--full-nodes")) orelse return;
    }
    var oai_backends: ?std.ArrayList(sync.PeerAddr) = null;
    defer if (oai_backends) |*b| b.deinit(gpa);
    if (flagStr(args, "--openai-full-nodes")) |csv| {
        oai_backends = (try parseBackends(gpa, out, csv, "--openai-full-nodes")) orelse return;
    }

    const serve_rpc = rpc_backends != null;
    const serve_openai = openai_port != 0;
    if (!serve_rpc and !serve_openai) {
        return out.print("light: configure --full-nodes (native RPC) and/or --openai-port with --openai-full-nodes\n", .{});
    }
    if (serve_openai and oai_backends == null) {
        return out.print("light: --openai-port needs --openai-full-nodes HOST:OPENAI_PORT[,...]\n", .{});
    }

    try out.print("loom light node up (no weights, no engine)\n", .{});
    if (serve_rpc) try out.print("  local rpc  tcp://{s}:{d}   ({d} backend(s); JSON protocol delegated)\n", .{ rpc_addr, rpc_port, rpc_backends.?.items.len });
    if (serve_openai) try out.print("  openai     http://{s}:{d}/v1  ({d} backend(s); OpenAI requests delegated)\n", .{ openai_addr, openai_port, oai_backends.?.items.len });
    try out.print("  client id  {s}  (forced on requests; metered by full nodes)\n", .{client_id});
    try out.print("  serving... (Ctrl-C to stop)\n", .{});
    try out.flush();

    // OpenAI delegator on a thread; native RPC delegator blocks (or OpenAI does
    // if native is not configured).
    var oai_ctx: light_openai.Ctx = undefined;
    if (serve_openai) {
        oai_ctx = .{ .gpa = gpa, .io = io, .opts = .{
            .addr = openai_addr,
            .port = openai_port,
            .backends = oai_backends.?.items,
            .client_id = client_id,
        } };
        if (serve_rpc) {
            const t = try std.Thread.spawn(.{}, lightOpenaiThread, .{&oai_ctx});
            t.detach();
        }
    }

    if (serve_rpc) {
        var ctx = light.Ctx{ .gpa = gpa, .io = io, .opts = .{
            .rpc_addr = rpc_addr,
            .rpc_port = rpc_port,
            .full_nodes = rpc_backends.?.items,
            .client_id = client_id,
        } };
        light.serve(&ctx) catch |e| try out.print("light rpc error: {s}\n", .{@errorName(e)});
    } else {
        lightOpenaiThread(&oai_ctx); // OpenAI-only: block here
    }
}

// ---- gguf ------------------------------------------------------------------

fn cmdGguf(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    if (args.len < 4) return out.print("gguf: need gen|info <file>\n", .{});
    const sub = args[2];
    const path = args[3];

    if (std.mem.eql(u8, sub, "gen")) {
        const seed = try flagU64(args, "--seed", 42);
        const arch = flagStr(args, "--arch") orelse "demo";
        if (std.mem.eql(u8, arch, "deepseek2")) {
            try out.print("writing synthetic deepseek2 GGUF (MLA + MoE, random weights) -> {s}\n", .{path});
            try out.flush();
            try gguf.writeDeepseekFixture(gpa, io, path, seed);
        } else if (llama.archFor(arch) != null) {
            try out.print("writing synthetic {s} GGUF (GQA + MoE, random weights) -> {s}\n", .{ arch, path });
            try out.flush();
            try gguf.writeMoeFixture(gpa, io, path, seed, arch);
        } else {
            const data_mb = try flagUsize(args, "--data-mb", 8);
            try out.print("writing synthetic GGUF ({d} MB tensor data) -> {s}\n", .{ data_mb, path });
            try out.flush();
            try gguf.writeFixture(gpa, io, path, seed, data_mb);
        }
        try out.print("done\n", .{});
        return;
    }

    if (std.mem.eql(u8, sub, "check")) {
        return @import("gguf/check.zig").run(gpa, io, out, path);
    }

    if (std.mem.eql(u8, sub, "info")) {
        var parsed = gguf.parse(gpa, io, path) catch |e| {
            return out.print("parse failed: {s}\n", .{@errorName(e)});
        };
        defer parsed.deinit();
        try out.print("gguf {s}\n", .{path});
        try out.print("  version={d} alignment={d} data_offset={d} file_size={d}\n", .{
            parsed.version, parsed.alignment, parsed.data_offset, parsed.file_size,
        });
        try out.print("  metadata ({d}):\n", .{parsed.metadata.len});
        for (parsed.metadata) |kv| {
            switch (kv.value) {
                .string => |s| try out.print("    {s} = \"{s}\"\n", .{ kv.key, s }),
                .int => |v| try out.print("    {s} = {d}\n", .{ kv.key, v }),
                .uint => |v| try out.print("    {s} = {d}\n", .{ kv.key, v }),
                .float => |v| try out.print("    {s} = {d}\n", .{ kv.key, v }),
                .boolean => |v| try out.print("    {s} = {}\n", .{ kv.key, v }),
                .array_str => |a| try out.print("    {s} = array(string, n={d})\n", .{ kv.key, a.len }),
                .array_f32 => |a| try out.print("    {s} = array(f32, n={d})\n", .{ kv.key, a.len }),
                .array_i32 => |a| try out.print("    {s} = array(i32, n={d})\n", .{ kv.key, a.len }),
                .array => |a| try out.print("    {s} = array(type={d}, n={d})\n", .{ kv.key, a.elem_type, a.count }),
            }
        }
        try out.print("  tensors ({d}):\n", .{parsed.tensors.len});
        for (parsed.tensors) |t| {
            try out.print("    {s} type={d} dims={any} offset={d}\n", .{ t.name, t.ggml_type, t.dims, t.offset });
        }

        // range manifest preview
        const range_mb = try flagF64(args, "--range-mb", 4.0);
        const range_bytes: u64 = @intFromFloat(range_mb * @as(f64, MB));
        var manifest = try weights.buildManifest(gpa, io, path, range_bytes);
        defer manifest.deinit(gpa);
        try out.print("  distribution: version={s} ranges={d} range_size={d:.1} MB\n", .{
            hash.toHex(manifest.version), manifest.nRanges(), range_mb,
        });
        return;
    }

    if (std.mem.eql(u8, sub, "run")) {
        return cmdGgufRun(gpa, io, out, path, args);
    }

    if (std.mem.eql(u8, sub, "shard")) {
        var m = weights.buildExpertManifest(gpa, io, path) catch |e| switch (e) {
            error.NoExpertTensors => return out.print("no 3D expert tensors found — not a MoE GGUF (fixed-size ranges apply instead)\n", .{}),
            else => return out.print("shard failed: {s}\n", .{@errorName(e)}),
        };
        defer m.deinit(gpa);

        var resident_bytes: u64 = 0;
        var i: usize = 0;
        while (i < m.n_resident) : (i += 1) resident_bytes += m.rangeLen(i);
        const n_expert = m.nRanges() - m.n_resident;
        var emin: u64 = std.math.maxInt(u64);
        var emax: u64 = 0;
        var esum: u64 = 0;
        while (i < m.nRanges()) : (i += 1) {
            const l = m.rangeLen(i);
            emin = @min(emin, l);
            emax = @max(emax, l);
            esum += l;
        }
        try out.print("expert-aligned shard manifest for {s}\n", .{path});
        try out.print("  version        {s}\n", .{hash.toHex(m.version)});
        try out.print("  file           {d:.2} GB ({d} bytes)\n", .{ @as(f64, @floatFromInt(m.file_size)) / GB, m.file_size });
        try out.print("  shards         {d} total = {d} resident + {d} expert\n", .{ m.nRanges(), m.n_resident, n_expert });
        try out.print("  resident       {d:.3} GB in {d} chunks (held by every node)\n", .{ @as(f64, @floatFromInt(resident_bytes)) / GB, m.n_resident });
        if (n_expert > 0) try out.print("  expert shards  {d:.2}..{d:.2} MB (avg {d:.2} MB), {d:.2} GB routed corpus\n", .{
            @as(f64, @floatFromInt(emin)) / @as(f64, MB),
            @as(f64, @floatFromInt(emax)) / @as(f64, MB),
            @as(f64, @floatFromInt(esum)) / @as(f64, @floatFromInt(n_expert)) / @as(f64, MB),
            @as(f64, @floatFromInt(esum)) / GB,
        });
        try out.print("  metadata       manifest {d:.1} KB, holdings bitmap {d} B\n", .{
            blk: {
                const t = try m.serialize(gpa);
                defer gpa.free(t);
                break :blk @as(f64, @floatFromInt(t.len)) / 1024.0;
            },
            (m.nRanges() + 7) / 8,
        });
        return;
    }

    try out.print("gguf: unknown subcommand {s} (want gen|check|info|shard|run)\n", .{sub});
}

fn cmdGgufRun(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, path: []const u8, args: [][]const u8) !void {
    // Same opt-in as the node: the measured-slower GPU paths run only when
    // asked, and this command is where the *local* fused MoE path is
    // exercised, since a model file loaded here has no distributed source.
    if (hasFlag(args, "--gpu-ops")) backend.useGpuOps.* = true;
    // a store directory (partial, expert-sharded) runs distributed
    {
        var pbuf: [4096]u8 = undefined;
        const mp = std.fmt.bufPrint(&pbuf, "{s}/ranges.manifest", .{path}) catch null;
        if (mp != null) {
            if (Io.Dir.cwd().access(io, mp.?, .{})) |_| {
                return runStore(gpa, io, out, path, args);
            } else |_| {}
        }
    }
    // peek the architecture, then dispatch to the matching engine
    var peek = gguf.parse(gpa, io, path) catch |e| {
        return out.print("parse failed: {s}\n", .{@errorName(e)});
    };
    const arch_buf: [64]u8 = undefined;
    _ = arch_buf;
    var arch_store: [64]u8 = undefined;
    const arch_src = peek.getString("general.architecture") orelse "?";
    const arch = arch_store[0..@min(arch_src.len, arch_store.len)];
    @memcpy(@constCast(arch), arch_src[0..arch.len]);
    peek.deinit();

    if (std.mem.eql(u8, arch, "deepseek2")) {
        return runEngine(deepseek, gpa, io, out, path, args);
    }
    if (llama.archFor(arch) != null) {
        return runEngine(llama, gpa, io, out, path, args);
    }
    try out.print("unsupported architecture: {s} (supported: deepseek2", .{arch});
    for (llama.arches) |a| try out.print(", {s}", .{a.name});
    try out.print(")\n", .{});
}

/// Run a MoE model from a *partial* expert-sharded store: held shards come
/// from the local sparse file, missing ones are fetched from --peers in the
/// token loop (digest-verified, persisted — issue #3). The engine is chosen
/// from the store's own model file, so this works for deepseek2 (MLA) and the
/// GQA family alike.
fn runStore(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, dir: []const u8, args: [][]const u8) !void {
    var store = weights.openDir(gpa, io, dir) catch |e| {
        return out.print("store open failed: {s}\n", .{@errorName(e)});
    };
    defer store.deinit();
    if (store.manifest.mode != .expert) {
        return out.print("store is not expert-sharded (mode={s}) — cannot run distributed\n", .{@tagName(store.manifest.mode)});
    }

    var peer_list = std.ArrayList(sync.PeerAddr).empty;
    defer peer_list.deinit(gpa);
    if (flagStr(args, "--peers")) |csv| {
        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " ");
            if (t.len == 0) continue;
            const a = sync.PeerAddr.parse(t) catch return out.print("bad --peers entry: {s}\n", .{t});
            try peer_list.append(gpa, a);
        }
    }
    var committee_list = std.ArrayList(sync.PeerAddr).empty;
    defer committee_list.deinit(gpa);
    if (flagStr(args, "--committee")) |csv| {
        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " ");
            if (t.len == 0) continue;
            const a = sync.PeerAddr.parse(t) catch return out.print("bad --committee entry: {s}\n", .{t});
            try committee_list.append(gpa, a);
        }
    }

    // The RAM tier: without it every routed expert is re-read from disk and
    // re-hashed on every token.
    const cache_gb = try flagF64(args, "--ram-gb", defaultRamGb());
    var src = try expert_fetch.Source.initCached(gpa, io, &store, peer_list.items, @intFromFloat(cache_gb * GB));
    src.committee = committee_list.items;

    // resident completeness gate (audit #5 P0-4): the resident bundle
    // (attention, shared experts, embeddings, router) is mmap'd — a missing
    // extent would read as file-hole zeros and silently corrupt inference.
    // Fail closed unless every resident shard is held+verified, fetching any
    // gaps from peers first.
    {
        var missing_resident: usize = 0;
        var i: usize = 0;
        while (i < store.manifest.n_resident) : (i += 1) {
            _ = src.get(i) catch {
                missing_resident += 1;
            };
        }
        if (missing_resident > 0) {
            return out.print("refusing to run: {d}/{d} resident shards unavailable (mmap'd bundle would read as zeros)\n", .{
                missing_resident, store.manifest.n_resident,
            });
        }
    }
    defer src.deinit();

    const mpath = try std.fmt.allocPrint(gpa, "{s}/model.gguf", .{dir});
    defer gpa.free(mpath);
    var peek = gguf.parse(gpa, io, mpath) catch |e| {
        return out.print("load failed: {s}\n", .{@errorName(e)});
    };
    const is_mla = std.mem.eql(u8, peek.getString("general.architecture") orelse "?", "deepseek2");
    peek.deinit();
    if (is_mla) return runStoreWith(deepseek, gpa, io, out, &store, &src, mpath, args);
    return runStoreWith(llama, gpa, io, out, &store, &src, mpath, args);
}

/// The distributed-store generation loop, instantiated per engine module.
fn runStoreWith(
    comptime E: type,
    gpa: std.mem.Allocator,
    io: Io,
    out: *Io.Writer,
    store: *weights.Store,
    src: *expert_fetch.Source,
    mpath: []const u8,
    args: [][]const u8,
) !void {
    var m = E.load(gpa, io, mpath) catch |e| {
        return out.print("load failed: {s}\n", .{@errorName(e)});
    };
    defer m.deinit();
    E.attachDist(&m, gpa, src) catch |e| {
        return out.print("attach failed: {s}\n", .{@errorName(e)});
    };

    const held_before = store.holdings.count();
    try out.print("distributed store: shards={d} held={d} ({d} resident + experts) peers={d}\n", .{
        store.manifest.nRanges(), held_before, store.manifest.n_resident, src.peers.len,
    });

    const ctx_cap = try flagUsize(args, "--ctx", 4096);
    generator.kernel_threads = try flagUsize(args, "--threads", 0);
    generator.prefill_batch = try flagUsize(args, "--batch", 0);
    m.cfg.ctx_len = @min(m.cfg.ctx_len, ctx_cap);
    const c = m.cfg;
    try out.print("gguf: dim={d} layers={d} heads={d} vocab={d} ctx={d}\n", .{
        c.dim, c.n_layers, c.n_heads, c.vocab, c.ctx_len,
    });
    try out.flush();

    backend.parallelBegin(generator.threads());
    // The device exists now; make any weight mappings registered at load
    // device-resident. Without this the GPU paths read the mmap through bare
    // per-tensor wrappings and fault file-backed pages per access -- measured
    // at ~285 MB/s inside the fused MoE block, 30-100 ms per matvec dispatch
    // that costs under a millisecond resident.
    _ = backend.materializeArenas();
    // Device-side MLA cache too, sized by the capped context -- the model's
    // native ctx_len (163,840 here) is past what the attention kernel's
    // threadgroup memory serves, so asking with it declines and every device
    // attention call then silently falls back to the host. Fifth instance of
    // this gate; hence a comptime check that E either has the fields or not.
    if (@hasField(@TypeOf(m.cfg), "kv_lora_rank")) {
        _ = backend.mlaInit(m.cfg.n_layers, m.cfg.ctx_len, m.cfg.kv_lora_rank, m.cfg.rope_dim);
    }
    // ...and W_k, dequantized to the device, without which every mlaAttnHeads
    // and mlaLayerTail call declines at `li >= mla_wk.items.len`. Sixth
    // instance of a device path silently off because its setup lived on
    // another code path.
    if (@hasDecl(E, "uploadAbsorbWeights")) E.uploadAbsorbWeights(&m, gpa);
    if (@hasDecl(E, "buildFrameDescs")) m.frame_descs = E.buildFrameDescs(&m, gpa);
    defer backend.parallelEnd();

    var st = try E.State.init(gpa, c);
    defer st.deinit(gpa);

    const prompt = flagStr(args, "--prompt") orelse "Once upon a time";
    const max_tokens = try flagUsize(args, "--max-tokens", 128);
    const temp: f32 = @floatCast(try flagF64(args, "--temp", 0.0));
    const seed = try flagU64(args, "--seed", 42);

    const prompt_toks = try m.encodePrompt(gpa, prompt, false);
    defer gpa.free(prompt_toks);
    if (prompt_toks.len == 0) return out.print("empty prompt after tokenization\n", .{});

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const sample_scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(sample_scratch);

    const t0 = stats.nowNs(io);
    var pos: usize = 0;
    try out.print("output: ", .{});
    if (@hasDecl(E, "stepBatch")) {
        // Batched prefill: the prompt is known up front, so weights are
        // unpacked once for several tokens rather than once per token.
        while (pos < prompt_toks.len) {
            if (pos >= c.ctx_len) return out.print("\nprompt exceeds context\n", .{});
            const take = @min(@min(generator.batchSize(), prompt_toks.len - pos), c.ctx_len - pos);
            try E.stepBatch(&m, &st, prompt_toks[pos..][0..take], pos);
            for (prompt_toks[pos..][0..take]) |tok| try m.decodeToken(out, tok);
            pos += take;
        }
    } else for (prompt_toks) |tok| {
        if (pos >= c.ctx_len) return out.print("\nprompt exceeds context\n", .{});
        try E.step(&m, &st, tok, pos);
        try m.decodeToken(out, tok);
        pos += 1;
    }
    try out.flush();

    var produced: usize = 0;
    var last: u32 = @intCast(sampler.sample(sample_scratch, st.logits, temp, rnd));
    while (produced < max_tokens and pos < c.ctx_len) : (produced += 1) {
        if (last == m.eosToken()) break;
        try m.decodeToken(out, last);
        try out.flush();
        try E.step(&m, &st, last, pos);
        pos += 1;
        last = @intCast(sampler.sample(sample_scratch, st.logits, temp, rnd));
    }
    const t1 = stats.nowNs(io);
    const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const fs = src.stats;
    try out.print("\n---- {d} prompt + {d} generated tokens in {d:.2}s ({d:.1} tok/s) ----\n", .{
        prompt_toks.len, produced, secs, @as(f64, @floatFromInt(prompt_toks.len + produced)) / secs,
    });
    try out.print("expert tiers: local={d} peer_fetched={d} ({d:.1} MB, avg {d:.1} ms/fetch) failures={d}\n", .{
        fs.local,
        fs.fetched,
        @as(f64, @floatFromInt(fs.fetch_bytes)) / @as(f64, MB),
        if (fs.fetched > 0) @as(f64, @floatFromInt(fs.fetch_ns)) / @as(f64, @floatFromInt(fs.fetched)) / 1e6 else 0,
        fs.fetch_failures,
    });
    try store.saveSidecars(); // fetched shards persist: this node is now a bigger holder
    try out.print("holdings grew {d} -> {d} shards (fetched experts persisted + advertised)\n", .{
        held_before, store.holdings.count(),
    });
}

/// Generic generation loop over either engine module (same load/State/step shape).
fn runEngine(comptime eng: type, gpa: std.mem.Allocator, io: Io, out: *Io.Writer, path: []const u8, args: [][]const u8) !void {
    const prompt = flagStr(args, "--prompt") orelse "Once upon a time";
    const max_tokens = try flagUsize(args, "--max-tokens", 128);
    const temp: f32 = @floatCast(try flagF64(args, "--temp", 0.0));
    const seed = try flagU64(args, "--seed", 42);

    var m = eng.load(gpa, io, path) catch |e| {
        return out.print("load failed: {s}\n", .{@errorName(e)});
    };
    defer m.deinit();
    // cap the KV allocation: model context lengths can be 100k+
    const ctx_cap = try flagUsize(args, "--ctx", 4096);
    generator.kernel_threads = try flagUsize(args, "--threads", 0);
    generator.prefill_batch = try flagUsize(args, "--batch", 0);
    m.cfg.ctx_len = @min(m.cfg.ctx_len, ctx_cap);
    const c = m.cfg;
    try out.print("gguf: dim={d} layers={d} heads={d} vocab={d} ctx={d}\n", .{
        c.dim, c.n_layers, c.n_heads, c.vocab, c.ctx_len,
    });
    try out.flush();

    backend.parallelBegin(generator.threads());
    // The device exists now; make any weight mappings registered at load
    // device-resident. Without this the GPU paths read the mmap through bare
    // per-tensor wrappings and fault file-backed pages per access -- measured
    // at ~285 MB/s inside the fused MoE block, 30-100 ms per matvec dispatch
    // that costs under a millisecond resident.
    _ = backend.materializeArenas();
    // Device-side MLA cache too, sized by the capped context -- the model's
    // native ctx_len (163,840 here) is past what the attention kernel's
    // threadgroup memory serves, so asking with it declines and every device
    // attention call then silently falls back to the host. Fifth instance of
    // this gate; hence a comptime check that E either has the fields or not.
    if (@hasField(@TypeOf(m.cfg), "kv_lora_rank")) {
        _ = backend.mlaInit(m.cfg.n_layers, m.cfg.ctx_len, m.cfg.kv_lora_rank, m.cfg.rope_dim);
    }
    // ...and W_k, dequantized to the device, without which every mlaAttnHeads
    // and mlaLayerTail call declines at `li >= mla_wk.items.len`. Sixth
    // instance of a device path silently off because its setup lived on
    // another code path.
    if (@hasDecl(eng, "uploadAbsorbWeights")) eng.uploadAbsorbWeights(&m, gpa);
    if (@hasDecl(eng, "buildFrameDescs")) m.frame_descs = eng.buildFrameDescs(&m, gpa);
    defer backend.parallelEnd();

    var st = try eng.State.init(gpa, c);
    defer st.deinit(gpa);

    const prompt_toks = try m.encodePrompt(gpa, prompt, false);
    defer gpa.free(prompt_toks);
    if (prompt_toks.len == 0) return out.print("empty prompt after tokenization\n", .{});

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const sample_scratch = try gpa.alloc(f32, c.vocab);
    defer gpa.free(sample_scratch);

    const t0 = stats.nowNs(io);
    var pos: usize = 0;
    // prefill (echo the prompt as we go)
    try out.print("output: ", .{});
    if (@hasDecl(eng, "stepBatch")) {
        // Batched prefill: unpack each weight once for several tokens.
        while (pos < prompt_toks.len) {
            if (pos >= c.ctx_len) return out.print("\nprompt exceeds context\n", .{});
            const take = @min(@min(generator.batchSize(), prompt_toks.len - pos), c.ctx_len - pos);
            try eng.stepBatch(&m, &st, prompt_toks[pos..][0..take], pos);
            for (prompt_toks[pos..][0..take]) |tok| try m.decodeToken(out, tok);
            pos += take;
        }
    } else for (prompt_toks) |tok| {
        if (pos >= c.ctx_len) return out.print("\nprompt exceeds context\n", .{});
        try eng.step(&m, &st, tok, pos);
        try m.decodeToken(out, tok);
        pos += 1;
    }
    try out.flush();

    var produced: usize = 0;
    var last: u32 = @intCast(sampler.sample(sample_scratch, st.logits, temp, rnd));
    while (produced < max_tokens and pos < c.ctx_len) : (produced += 1) {
        if (last == m.eosToken()) break;
        try m.decodeToken(out, last);
        try out.flush();
        try eng.step(&m, &st, last, pos);
        pos += 1;
        last = @intCast(sampler.sample(sample_scratch, st.logits, temp, rnd));
    }
    const t1 = stats.nowNs(io);
    const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    try out.print("\n---- {d} prompt + {d} generated tokens in {d:.2}s ({d:.1} tok/s) ----\n", .{
        prompt_toks.len, produced, secs, @as(f64, @floatFromInt(prompt_toks.len + produced)) / secs,
    });
}

// ---- arg helpers -----------------------------------------------------------

fn hasFlag(args: [][]const u8, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a, name)) return true;
    return false;
}

fn flagStr(args: [][]const u8, name: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i + 1 < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], name)) return args[i + 1];
    }
    return null;
}

fn flagU64(args: [][]const u8, name: []const u8, default: u64) !u64 {
    if (flagStr(args, name)) |v| return std.fmt.parseInt(u64, v, 10);
    return default;
}
fn flagUsize(args: [][]const u8, name: []const u8, default: usize) !usize {
    if (flagStr(args, name)) |v| return std.fmt.parseInt(usize, v, 10);
    return default;
}
fn flagU16(args: [][]const u8, name: []const u8, default: u16) !u16 {
    if (flagStr(args, name)) |v| return std.fmt.parseInt(u16, v, 10);
    return default;
}
fn flagF64(args: [][]const u8, name: []const u8, default: f64) !f64 {
    if (flagStr(args, name)) |v| return std.fmt.parseFloat(f64, v);
    return default;
}

fn envUsize(env: *std.process.Environ.Map, name: []const u8, default: usize) !usize {
    if (env.get(name)) |v| return std.fmt.parseInt(usize, v, 10);
    return default;
}
fn envU64(env: *std.process.Environ.Map, name: []const u8, default: u64) !u64 {
    if (env.get(name)) |v| return std.fmt.parseInt(u64, v, 10);
    return default;
}
fn envF64(env: *std.process.Environ.Map, name: []const u8, default: f64) !f64 {
    if (env.get(name)) |v| return std.fmt.parseFloat(f64, v);
    return default;
}

test {
    std.testing.refAllDecls(@This());
}

test "default RAM budget scales with the machine and stays in bounds" {
    // The failure this replaces: a flat 4 GB default OOM-killed an origin on a
    // 7.7 GB box (7.4 GB anon RSS, dead before serving a request). A cache
    // budget has to be a fraction of what exists.
    const got = defaultRamGb();
    try std.testing.expect(got >= 0.5);
    try std.testing.expect(got <= 8.0);
    if (std.process.totalSystemMemory()) |total| {
        const gb = @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0 * 1024.0);
        // A quarter of the machine, except where a clamp binds. Stated as an
        // implication so the test says what the rule is rather than restating
        // the arithmetic.
        if (gb / 4.0 > 0.5 and gb / 4.0 < 8.0) {
            try std.testing.expectApproxEqAbs(gb / 4.0, got, 1e-9);
        }
        // Never more than a quarter: the point is leaving room for the model's
        // mappings, the page cache and the rest of the system.
        try std.testing.expect(got <= @max(0.5, gb / 4.0) + 1e-9);
    } else |_| {
        // Unknown memory must not produce an aggressive budget.
        try std.testing.expectEqual(@as(f64, 2.0), got);
    }
}
