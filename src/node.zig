//! `loom node` — a running inference node: load a model (local / synthetic /
//! Hugging Face), serve inference over RPC, and answer expert-directory queries
//! over P2P. This is the single-box daemon that v1 will federate into a cluster.

const std = @import("std");
const Io = std.Io;
const engine_mod = @import("engine.zig");
const Engine = engine_mod.Engine;
const hf = @import("hf.zig");
const rpc = @import("rpc.zig");
const p2p = @import("p2p.zig");
const stats = @import("stats.zig");

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

pub const Options = struct {
    model: []const u8,
    rpc_addr: []const u8,
    rpc_port: u16,
    p2p_addr: []const u8,
    p2p_port: u16,
    ram_bytes: u64,
    pin_bytes: u64,
    verify: bool,
    seed: u64,
    stats_path: ?[]const u8,
    cache_root: []const u8,
};

fn p2pThread(ctx: *p2p.Ctx) void {
    p2p.serve(ctx) catch |e| std.debug.print("p2p: {s}\n", .{@errorName(e)});
}

pub fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, opts: Options) !void {
    const resolved = hf.resolve(gpa, io, opts.model, opts.cache_root) catch |e| {
        try out.print("model resolve failed ({s}): {s}\n", .{ opts.model, @errorName(e) });
        return;
    };
    defer gpa.free(resolved.dir);

    var eng = Engine.init(gpa, io, resolved.dir, .{
        .ram_budget_bytes = opts.ram_bytes,
        .pin_budget_bytes = opts.pin_bytes,
        .verify = opts.verify,
    }) catch |e| {
        try out.print("load failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer eng.deinit();

    if (opts.stats_path) |sp| if (opts.pin_bytes > 0) {
        if (try stats.readCounts(gpa, io, sp, eng.cache.access_count.len)) |counts| {
            defer gpa.free(counts);
            try eng.pinWithinBudget(counts, opts.pin_bytes);
        }
    };

    const c = eng.cfg;
    const s = eng.sizes;
    try out.print("loom node up\n", .{});
    try out.print("  model      {s}  (source: {s})\n", .{ resolved.dir, resolved.source });
    try out.print("  shape      hidden={d} layers={d} experts={d}/layer top-{d}+{d} shared\n", .{
        c.hidden, c.nLayers(), c.n_experts, c.n_routed, c.n_shared,
    });
    try out.print("  resident   dense={d:.3} GB kv={d:.3} GB\n", .{
        @as(f64, @floatFromInt(s.dense_bytes)) / GB, @as(f64, @floatFromInt(s.kv_bytes)) / GB,
    });
    try out.print("  cache      expert_bytes={d} lru_capacity={d} pinned={d} ram_budget={d:.2} GB\n", .{
        s.expert_bytes, s.lru_capacity, s.pinned_experts, @as(f64, @floatFromInt(opts.ram_bytes)) / GB,
    });
    try out.print("  rpc        tcp://{s}:{d}   (json: {{\"prompt\":\"..\",\"max_tokens\":32}})\n", .{ opts.rpc_addr, opts.rpc_port });
    try out.print("  p2p        tcp://{s}:{d}   (HELLO | COUNT | HAS <id> | PING)\n", .{ opts.p2p_addr, opts.p2p_port });
    try out.print("  serving... (Ctrl-C to stop)\n", .{});
    try out.flush();

    var p2p_ctx = p2p.Ctx{
        .gpa = gpa,
        .io = io,
        .entries = eng.loaded.entries,
        .unique_bytes = s.unique_expert_bytes,
        .addr = opts.p2p_addr,
        .port = opts.p2p_port,
    };
    const p2p_handle = try std.Thread.spawn(.{}, p2pThread, .{&p2p_ctx});
    defer p2p_handle.join();

    var rpc_ctx = rpc.Ctx{
        .gpa = gpa,
        .io = io,
        .engine = &eng,
        .addr = opts.rpc_addr,
        .port = opts.rpc_port,
        .seed = opts.seed,
    };
    rpc.serve(&rpc_ctx) catch |e| {
        try out.print("rpc server error: {s}\n", .{@errorName(e)});
        return;
    };
}
