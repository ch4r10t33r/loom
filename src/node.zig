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
const weights = @import("weights.zig");
const sync = @import("sync.zig");
const hashmod = @import("hash.zig");
const peers = @import("peers.zig");
const gossip = @import("gossip.zig");

const GB: f64 = 1024.0 * 1024.0 * 1024.0;
const MBf: f64 = 1024.0 * 1024.0;

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
    // GGUF distribution plane
    gguf_path: ?[]const u8, // serve this complete GGUF (origin/full holder)
    bootstrap: ?[]const u8, // "host:port" — sync weight ranges from this peer
    peers: ?[]const u8, // extra weight peers, comma-separated "host:port"
    hold_fraction: f32, // fraction of ranges to hold when bootstrapping
    range_bytes: u64, // range size when building a fresh manifest
    advertise: ?[]const u8, // our dialable "host:port" (default 127.0.0.1:<p2p_port>)
};

/// Churn-repair policy (ROADMAP #6, decided: maximally eager): whenever the
/// wanted range set is unsatisfied, retry every known peer on a short interval —
/// no waiting for a miss. A peer that was down when we bootstrapped, or that
/// comes back holding what we need, gets drained within one interval.
const REPAIR_INTERVAL_NS: i96 = 2 * std.time.ns_per_s;

const RepairCtx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    /// Candidate holders come from the live gossip table, so repair reaches
    /// peers this node was never explicitly told about.
    table: *peers.Table,
};

fn repairThread(ctx: *RepairCtx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = REPAIR_INTERVAL_NS }, .awake) catch return;
        if (ctx.store.missingCount() == 0) continue;
        const addrs = ctx.table.snapshotAddrs(ctx.gpa) catch continue;
        defer {
            for (addrs) |a| ctx.gpa.free(a);
            ctx.gpa.free(addrs);
        }
        var repaired: usize = 0;
        for (addrs) |addr_str| {
            if (ctx.store.missingCount() == 0) break;
            const addr = sync.PeerAddr.parse(addr_str) catch continue;
            const s = sync.fetchFromPeer(ctx.gpa, ctx.io, ctx.store, addr) catch continue;
            repaired += s.fetched;
        }
        if (repaired > 0) {
            ctx.store.saveSidecars() catch {};
            std.debug.print("repair: recovered {d} ranges, held {d}/{d}\n", .{
                repaired, ctx.store.holdings.count(), ctx.store.wanted.count(),
            });
        }
    }
}

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

    // ---- GGUF weight-distribution store (optional) ----
    var store: ?weights.Store = null;
    defer if (store) |*st| st.deinit();

    // known weight peers = --bootstrap (first) + --peers (comma-separated)
    var peer_list = std.ArrayList(sync.PeerAddr).empty;
    defer peer_list.deinit(gpa);
    var peer_strs = std.ArrayList([]const u8).empty; // same peers, as strings for the gossip table
    defer peer_strs.deinit(gpa);
    if (opts.bootstrap) |bs| {
        const a = sync.PeerAddr.parse(bs) catch {
            try out.print("bad --bootstrap address (want host:port): {s}\n", .{bs});
            return;
        };
        try peer_list.append(gpa, a);
        try peer_strs.append(gpa, bs);
    }
    if (opts.peers) |csv| {
        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " ");
            if (t.len == 0) continue;
            const a = sync.PeerAddr.parse(t) catch {
                try out.print("bad --peers entry (want host:port): {s}\n", .{t});
                return;
            };
            try peer_list.append(gpa, a);
            try peer_strs.append(gpa, t);
        }
    }

    // gossip peer table, seeded with the statically configured peers
    var advertise_buf: [128]u8 = undefined;
    const advertise = opts.advertise orelse
        try std.fmt.bufPrint(&advertise_buf, "127.0.0.1:{d}", .{opts.p2p_port});
    var table = peers.Table.init(gpa, io, advertise);
    defer table.deinit();
    const zero_version = "0" ** 64;
    for (peer_strs.items) |ps| {
        _ = table.merge(ps, zero_version, "", stats.nowNs(io)) catch {};
    }

    if (opts.gguf_path) |gp| {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-origin", .{opts.cache_root});
        defer gpa.free(store_dir);
        store = weights.openFull(gpa, io, gp, store_dir, opts.range_bytes) catch |e| {
            try out.print("gguf store failed ({s}): {s}\n", .{ gp, @errorName(e) });
            return;
        };
    } else if (peer_list.items.len > 0) {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-synced", .{opts.cache_root});
        defer gpa.free(store_dir);
        try out.print("bootstrapping weight ranges from {d} peer(s) (hold-fraction {d:.2})...\n", .{ peer_list.items.len, opts.hold_fraction });
        try out.flush();
        const res = sync.bootstrap(gpa, io, peer_list.items, store_dir, opts.hold_fraction, opts.seed, out) catch |e| {
            try out.print("bootstrap failed: {s}\n", .{@errorName(e)});
            return;
        };
        store = res.store;
        try out.print("  synced {d}/{d} wanted ranges, {d:.1} MB, verified against manifest root\n", .{
            res.fetched, res.wanted, @as(f64, @floatFromInt(res.bytes)) / MBf,
        });
        if (res.fetched < res.wanted) {
            try out.print("  {d} ranges still missing — eager repair will keep chasing peers\n", .{res.wanted - res.fetched});
        }
    }

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
    try out.print("  p2p        tcp://{s}:{d}   (HELLO | HAS | MANIFEST | DIGESTS | HOLDINGS | GETR | GOSSIP | TABLE | PING)\n", .{ opts.p2p_addr, opts.p2p_port });
    try out.print("  gossip     advertising as {s}, {d} seed peer(s), every {d}s\n", .{
        advertise, peer_strs.items.len, @divTrunc(gossip.INTERVAL_NS, std.time.ns_per_s),
    });
    if (store) |*st| {
        try out.print("  weights    version={s} ranges={d} held={d} ({d:.1}%) range_size={d:.1} MB\n", .{
            hashmod.toHex(st.manifest.version),
            st.manifest.nRanges(),
            st.holdings.count(),
            100.0 * @as(f64, @floatFromInt(st.holdings.count())) / @as(f64, @floatFromInt(st.manifest.nRanges())),
            @as(f64, @floatFromInt(st.manifest.range_size)) / MBf,
        });
    }
    try out.print("  serving... (Ctrl-C to stop)\n", .{});
    try out.flush();

    var p2p_ctx = p2p.Ctx{
        .gpa = gpa,
        .io = io,
        .entries = eng.loaded.entries,
        .unique_bytes = s.unique_expert_bytes,
        .addr = opts.p2p_addr,
        .port = opts.p2p_port,
        .store = if (store) |*st| st else null,
        .table = &table,
        .advertise = advertise,
    };
    const p2p_handle = try std.Thread.spawn(.{}, p2pThread, .{&p2p_ctx});
    defer p2p_handle.join();

    // gossip loop: announce ourselves + learn peers-of-peers
    var gossip_ctx = gossip.Ctx{
        .gpa = gpa,
        .io = io,
        .table = &table,
        .store = if (store) |*st| st else null,
        .advertise = advertise,
    };
    {
        const t = try std.Thread.spawn(.{}, gossip.loop, .{&gossip_ctx});
        t.detach();
    }

    // eager churn repair, drawing candidates from the live gossip table
    var repair_ctx: RepairCtx = undefined;
    if (store != null) {
        repair_ctx = .{
            .gpa = gpa,
            .io = io,
            .store = &store.?,
            .table = &table,
        };
        const t = try std.Thread.spawn(.{}, repairThread, .{&repair_ctx});
        t.detach();
    }

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
