//! `loom node` — a running inference node: load a model (local / synthetic /
//! Hugging Face), serve inference over RPC, and answer expert-directory queries
//! over P2P. This is the single-box daemon that v1 will federate into a cluster.

const std = @import("std");
const Io = std.Io;
const engine_mod = @import("../engine/engine.zig");
const Engine = engine_mod.Engine;
const hf = @import("hf.zig");
const rpc = @import("rpc.zig");
const openai = @import("openai.zig");
const generator = @import("generator.zig");
const deepseek = @import("../gguf/deepseek.zig");
const chat_template = @import("../gguf/chat_template.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");
const p2p = @import("../p2p/p2p.zig");
const stats = @import("../core/stats.zig");
const weights = @import("../p2p/weights.zig");
const sync = @import("../p2p/sync.zig");
const hashmod = @import("../core/hash.zig");
const peers = @import("../p2p/peers.zig");
const gossip = @import("../p2p/gossip.zig");
const bootnode = @import("../p2p/bootnode.zig");
const wire = @import("../p2p/wire.zig");
const meter_mod = @import("meter.zig");

const GB: f64 = 1024.0 * 1024.0 * 1024.0;
const MBf: f64 = 1024.0 * 1024.0;

pub const Options = struct {
    model: []const u8,
    rpc_addr: []const u8,
    rpc_port: u16,
    openai_addr: []const u8, // OpenAI-compatible HTTP API bind addr
    openai_port: u16, // 0 = OpenAI surface disabled (SPEC.md client API)
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
    r_target: u16, // committee redundancy target when acting as bootnode
    free_quota: u64, // per-client free token allowance (metering)
    admin_token: []const u8, // gates the credit op (empty = credit disabled)
    ctx_cap: usize, // context-length cap when serving a distributed GGUF engine
    chat_format: ?[]const u8, // --chat-format override (null = auto-detect)
};

/// Committee heartbeat (SPEC.md): PING each committee member on a fixed
/// interval, track liveness, log transitions. A member that stops answering
/// is marked dead locally; its shards become repair candidates.
const HEARTBEAT_INTERVAL_NS: i96 = 5 * std.time.ns_per_s;

const MemberState = struct { alive: bool = true, last_seq: u64 = 0 };

const HeartbeatCtx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    seed: [][]u8, // members known at join time
    table: *peers.Table, // gossip-derived committee view (later joiners)
    committee_id: u32,
    state: std.StringHashMap(MemberState), // key = owned addr string
    p2p_ctx: *p2p.Ctx, // our own state rides the heartbeat (SPEC.md container)

    /// One wire-frame heartbeat exchange. Returns the peer's heartbeat.
    fn exchange(self: *HeartbeatCtx, addr_str: []const u8) ?wire.Heartbeat {
        const gpa = self.gpa;
        const io = self.io;
        const addr = sync.PeerAddr.parse(addr_str) catch return null;
        const ip = std.Io.net.IpAddress.parse(addr.host, addr.port) catch return null;
        const stream = ip.connect(io, .{ .mode = .stream }) catch return null;
        defer stream.close(io);
        var rbuf: [4096]u8 = undefined;
        var wbuf: [1024]u8 = undefined;
        var r = stream.reader(io, &rbuf);
        var w = stream.writer(io, &wbuf);

        const hb = p2p.selfHeartbeat(self.p2p_ctx);
        const body = hb.encodeBody(gpa) catch return null;
        defer gpa.free(body);
        const frame = wire.encodeFrame(gpa, .heartbeat, body) catch return null;
        defer gpa.free(frame);
        wire.writeFrame(&w.interface, frame) catch return null;

        const resp_raw = wire.readFrameAlloc(gpa, &r.interface) catch return null;
        defer gpa.free(resp_raw);
        const dec = wire.decodeFrame(gpa, resp_raw) catch return null;
        defer gpa.free(dec.body);
        if (dec.ty != .heartbeat_resp) return null;
        var out = wire.Heartbeat.parseBody(dec.body) catch return null;
        out.addr = ""; // aliases freed body; not needed by callers
        return out;
    }
};

fn heartbeatThread(ctx: *HeartbeatCtx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = HEARTBEAT_INTERVAL_NS }, .awake) catch return;

        // membership = static seed ∪ gossip-derived view (SPEC.md): earlier
        // members learn later joiners from their committee-tagged announces
        var members = std.ArrayList([]u8).empty;
        defer {
            for (members.items) |m| ctx.gpa.free(m);
            members.deinit(ctx.gpa);
        }
        for (ctx.seed) |m| {
            members.append(ctx.gpa, ctx.gpa.dupe(u8, m) catch continue) catch continue;
        }
        if (ctx.table.committeeMembersAlloc(ctx.gpa, ctx.committee_id)) |derived| {
            defer ctx.gpa.free(derived);
            outer: for (derived) |d| {
                for (members.items) |m| {
                    if (std.mem.eql(u8, m, d)) {
                        ctx.gpa.free(d);
                        continue :outer;
                    }
                }
                members.append(ctx.gpa, d) catch {
                    ctx.gpa.free(d);
                    continue;
                };
            }
        } else |_| {}

        for (members.items) |m| {
            const gop = ctx.state.getOrPut(m) catch continue;
            if (!gop.found_existing) {
                gop.key_ptr.* = ctx.gpa.dupe(u8, m) catch {
                    _ = ctx.state.remove(m);
                    continue;
                };
                gop.value_ptr.* = .{};
                std.debug.print("heartbeat: committee member {s} discovered\n", .{m});
            }
            const st = gop.value_ptr;

            const resp = ctx.exchange(m);
            const ok = resp != null;
            if (ok != st.alive) {
                std.debug.print("heartbeat: committee member {s} {s}\n", .{ m, if (ok) "alive" else "DEAD" });
                st.alive = ok;
                // death re-replication (audit #7 P1 / SPEC): survivors adopt the
                // dead member's advertised shards into their want-set so eager
                // repair re-replicates them from another live holder.
                if (!ok) {
                    if (ctx.p2p_ctx.store) |store| {
                        if (ctx.table.peerBitmapAlloc(ctx.gpa, m) catch null) |hex| {
                            defer ctx.gpa.free(hex);
                            var adopted: usize = 0;
                            var sid: usize = 0;
                            const nsh = store.manifest.nRanges();
                            while (sid < nsh) : (sid += 1) {
                                const bi = sid / 8;
                                if (bi * 2 + 1 >= hex.len) break;
                                const byte = std.fmt.parseInt(u8, hex[bi * 2 ..][0..2], 16) catch 0;
                                if (byte & (@as(u8, 1) << @intCast(sid % 8)) == 0) continue;
                                if (store.wanted.has(sid) or store.holdings.has(sid)) continue;
                                store.wanted.set(sid);
                                adopted += 1;
                            }
                            if (adopted > 0) std.debug.print("heartbeat: adopted {d} shard(s) from dead {s} into wanted\n", .{ adopted, m });
                        }
                    }
                }
            }
            if (resp) |hb| {
                if (hb.holdings_seq != st.last_seq) {
                    std.debug.print("heartbeat: {s} holdings_seq {d} -> {d} (load {d})\n", .{
                        m, st.last_seq, hb.holdings_seq, hb.load,
                    });
                    st.last_seq = hb.holdings_seq;
                }
            }
        }
    }
}

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
    /// Shared with the serving path: repair mutates the store (fetching shards)
    /// and so does the token-loop expert fetch during generation. Serialize both
    /// on the one engine mutex so their store writes never interleave.
    engine_lock: *Io.Mutex,
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
        ctx.engine_lock.lockUncancelable(ctx.io);
        defer ctx.engine_lock.unlock(ctx.io);
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

fn openaiThread(ctx: *openai.Ctx) void {
    openai.serve(ctx) catch |e| std.debug.print("openai: {s}\n", .{@errorName(e)});
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
    var committee_members: [][]u8 = &.{};
    defer {
        for (committee_members) |m| gpa.free(m);
        if (committee_members.len > 0) gpa.free(committee_members);
    }
    var registry: ?bootnode.Registry = null;
    defer if (registry) |*r| r.deinit();
    var joined_committee_id: u32 = wire.NO_COMMITTEE;

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
        _ = table.merge(ps, zero_version, "", peers.NO_COMMITTEE, 0, stats.nowNs(io)) catch {};
    }

    if (opts.gguf_path) |gp| {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-origin", .{opts.cache_root});
        defer gpa.free(store_dir);
        store = weights.openFull(gpa, io, gp, store_dir, opts.range_bytes, null) catch |e| {
            try out.print("gguf store failed ({s}): {s}\n", .{ gp, @errorName(e) });
            return;
        };
        // an expert-sharded origin acts as the bootnode (SPEC.md)
        if (store.?.manifest.mode == .expert) {
            registry = bootnode.Registry.init(
                gpa,
                io,
                store.?.manifest.nRanges(),
                store.?.manifest.n_resident,
                opts.r_target,
            );
        }
    } else if (peer_list.items.len > 0) {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-synced", .{opts.cache_root});
        defer gpa.free(store_dir);

        // SPEC.md join flow: ask the bootnode for a committee + assigned
        // want-set. Falls back to random hold-fraction if the peer is not a
        // bootnode (legacy swarms, fixed-mode manifests).
        var joined: ?sync.JoinInfo = null;
        if (sync.joinSwarm(gpa, io, peer_list.items[0], advertise, opts.hold_fraction)) |ji| {
            joined = ji;
        } else |e| {
            try out.print("join declined ({s}); using random hold-fraction\n", .{@errorName(e)});
        }

        if (joined) |*ji| {
            joined_committee_id = @intCast(ji.committee_id);
            try out.print("joined committee {d} ({d} member(s) already in it)\n", .{ ji.committee_id, ji.members.len });
            // sync preference: committee members first, then the bootnode
            var srcs = std.ArrayList(sync.PeerAddr).empty;
            defer srcs.deinit(gpa);
            for (ji.members) |m| {
                if (sync.PeerAddr.parse(m)) |a| {
                    try srcs.append(gpa, a);
                    // committee members seed the table WITH their committee id
                    _ = table.merge(m, "0" ** 64, "", joined_committee_id, 0, stats.nowNs(io)) catch {};
                } else |_| {}
            }
            try srcs.appendSlice(gpa, peer_list.items);

            // hand manifest+wanted to the store; keep members for heartbeats
            const res = sync.bootstrapWithWanted(gpa, io, srcs.items, store_dir, ji.manifest, ji.wanted, out) catch |e| {
                try out.print("bootstrap failed: {s}\n", .{@errorName(e)});
                for (ji.members) |m| gpa.free(m);
                gpa.free(ji.members);
                return;
            };
            store = res.store;
            committee_members = ji.members; // ownership moves (freed at exit)
            ji.members = &.{};
            try out.print("  synced {d}/{d} assigned shards, {d:.1} MB, verified against manifest root\n", .{
                res.fetched, res.wanted, @as(f64, @floatFromInt(res.bytes)) / MBf,
            });
        } else {
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
    try out.print("  rpc        tcp://{s}:{d}   (metered; free quota {d} tokens/client; credit {s})\n", .{ opts.rpc_addr, opts.rpc_port, opts.free_quota, if (opts.admin_token.len > 0) "admin-gated" else "disabled" });
    if (opts.openai_port != 0) try out.print("  openai     http://{s}:{d}/v1  (OpenAI-compatible; metered; non-streaming)\n", .{ opts.openai_addr, opts.openai_port });
    try out.print("  p2p        tcp://{s}:{d}   (HELLO | HAS | MANIFEST | DIGESTS | HOLDINGS | GETR | GOSSIP | TABLE | PING)\n", .{ opts.p2p_addr, opts.p2p_port });
    try out.print("  gossip     advertising as {s}, {d} seed peer(s), every {d}s\n", .{
        advertise, peer_strs.items.len, @divTrunc(gossip.INTERVAL_NS, std.time.ns_per_s),
    });
    if (registry != null) try out.print("  bootnode   committee registry active (R target {d})\n", .{opts.r_target});
    if (committee_members.len > 0) try out.print("  committee  {d} member(s), heartbeat every {d}s\n", .{
        committee_members.len, @divTrunc(HEARTBEAT_INTERVAL_NS, std.time.ns_per_s),
    });
    if (store) |*st| {
        try out.print("  weights    version={s}\n", .{hashmod.toHex(st.manifest.version)});
        try out.print("  shards     mode={s} total={d} (resident={d}, expert={d}) held={d} ({d:.1}%)\n", .{
            @tagName(st.manifest.mode),
            st.manifest.nRanges(),
            st.manifest.n_resident,
            st.manifest.nRanges() - st.manifest.n_resident,
            st.holdings.count(),
            100.0 * @as(f64, @floatFromInt(st.holdings.count())) / @as(f64, @floatFromInt(st.manifest.nRanges())),
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
        .boot = if (registry) |*r| r else null,
        .committee_id = joined_committee_id,
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
        .committee_id = joined_committee_id,
    };
    {
        const t = try std.Thread.spawn(.{}, gossip.loop, .{&gossip_ctx});
        t.detach();
    }

    // committee heartbeats (SPEC.md)
    var hb_ctx: HeartbeatCtx = undefined;
    if (joined_committee_id != wire.NO_COMMITTEE) {
        hb_ctx = .{
            .gpa = gpa,
            .io = io,
            .seed = committee_members,
            .table = &table,
            .committee_id = joined_committee_id,
            .state = std.StringHashMap(MemberState).init(gpa),
            .p2p_ctx = &p2p_ctx,
        };
        const t = try std.Thread.spawn(.{}, heartbeatThread, .{&hb_ctx});
        t.detach();
    }

    // One engine mutex shared by every serve path (rpc + openai) AND the eager
    // repair loop: generation holds mutable per-request state and mutates the
    // store (token-loop fetch), and repair mutates the store too, so all of it
    // serializes here.
    var engine_lock: Io.Mutex = .init;

    // Choose what serves inference. Default: the loom-format engine. If an
    // expert-sharded GGUF store is attached and its resident bundle is complete,
    // serve the distributed GGUF (deepseek2) engine with token-loop peer fetch.
    var gen = generator.Generator{ .loom = &eng };
    var gguf_gen: generator.GgufGen = undefined;
    var gguf_src: expert_fetch.Source = undefined;
    var committee_peers = std.ArrayList(sync.PeerAddr).empty;
    defer committee_peers.deinit(gpa);
    var serve_gguf = false;
    if (store) |*st| {
        if (st.manifest.mode == .expert) {
            for (committee_members) |m| {
                if (sync.PeerAddr.parse(m)) |a| {
                    committee_peers.append(gpa, a) catch {};
                } else |_| {}
            }
            gguf_src = try expert_fetch.Source.init(gpa, io, st, peer_list.items);
            gguf_src.committee = committee_peers.items;
            // resident gate (audit #5 P0-4): the mmap'd resident bundle must be
            // present+verified, fetching gaps from peers, or inference reads
            // file-hole zeros. Fail closed on the serve path (fall back to loom).
            var missing_resident: usize = 0;
            var ri: usize = 0;
            while (ri < st.manifest.n_resident) : (ri += 1) {
                _ = gguf_src.get(ri) catch {
                    missing_resident += 1;
                };
            }
            if (missing_resident == 0) {
                // origin serves the original --gguf file (openFull leaves it in
                // place); a bootstrapped node serves the sparse copy synced into
                // its store dir.
                const mpath = if (opts.gguf_path) |gp|
                    try gpa.dupe(u8, gp)
                else
                    try std.fmt.allocPrint(gpa, "{s}/model.gguf", .{st.dir});
                defer gpa.free(mpath);
                if (deepseek.load(gpa, io, mpath)) |mdl| {
                    gguf_gen = .{ .m = mdl, .src = &gguf_src, .ctx_cap = opts.ctx_cap };
                    gguf_gen.m.cfg.ctx_len = @min(gguf_gen.m.cfg.ctx_len, opts.ctx_cap);
                    if (deepseek.attachDist(&gguf_gen.m, gpa, &gguf_src)) |_| {
                        gguf_gen.chat_format = if (opts.chat_format) |cf|
                            chat_template.parse(cf) orelse chat_template.detect(gguf_gen.m.chatTemplate(), "deepseek2")
                        else
                            chat_template.detect(gguf_gen.m.chatTemplate(), "deepseek2");
                        gen = .{ .gguf = &gguf_gen };
                        serve_gguf = true;
                        try out.print("  serving    distributed GGUF (deepseek2): dim={d} layers={d} vocab={d} ctx={d} chat={s}\n", .{
                            gguf_gen.m.cfg.dim, gguf_gen.m.cfg.n_layers, gguf_gen.m.cfg.vocab, gguf_gen.m.cfg.ctx_len, @tagName(gguf_gen.chat_format),
                        });
                    } else |e| {
                        try out.print("  gguf serve disabled: attach failed ({s})\n", .{@errorName(e)});
                        gguf_gen.m.deinit();
                        gguf_src.deinit();
                    }
                } else |e| {
                    try out.print("  gguf serve disabled: model load failed ({s}); serving loom engine\n", .{@errorName(e)});
                    gguf_src.deinit();
                }
            } else {
                try out.print("  gguf serve disabled: {d}/{d} resident shards unavailable; serving loom engine\n", .{ missing_resident, st.manifest.n_resident });
                gguf_src.deinit();
            }
        }
    }
    defer if (serve_gguf) {
        gguf_gen.m.deinit();
        gguf_src.deinit();
    };
    try out.flush();

    // eager churn repair, drawing candidates from the live gossip table
    var repair_ctx: RepairCtx = undefined;
    if (store != null) {
        repair_ctx = .{
            .gpa = gpa,
            .io = io,
            .store = &store.?,
            .table = &table,
            .engine_lock = &engine_lock,
        };
        const t = try std.Thread.spawn(.{}, repairThread, .{&repair_ctx});
        t.detach();
    }

    var meter = meter_mod.Meter.init(gpa, io, opts.free_quota);
    defer meter.deinit();

    // OpenAI-compatible HTTP surface (SPEC.md client API), off unless a port is set.
    var openai_ctx: openai.Ctx = undefined;
    if (opts.openai_port != 0) {
        openai_ctx = .{
            .gpa = gpa,
            .io = io,
            .gen = &gen,
            .addr = opts.openai_addr,
            .port = opts.openai_port,
            .seed = opts.seed,
            .model_id = opts.model,
            .engine_lock = &engine_lock,
            .meter = &meter,
        };
        const t = try std.Thread.spawn(.{}, openaiThread, .{&openai_ctx});
        t.detach();
    }

    var rpc_ctx = rpc.Ctx{
        .gpa = gpa,
        .io = io,
        .gen = &gen,
        .addr = opts.rpc_addr,
        .port = opts.rpc_port,
        .seed = opts.seed,
        .meter = &meter,
        .admin_token = opts.admin_token,
        .engine_lock = &engine_lock,
    };
    rpc.serve(&rpc_ctx) catch |e| {
        try out.print("rpc server error: {s}\n", .{@errorName(e)});
        return;
    };
}
