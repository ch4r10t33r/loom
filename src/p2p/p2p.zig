//! Minimal P2P peer server — the seed of the v1 distributed expert directory.
//!
//! A real deployment runs this over Hyperswarm/HyperDHT; here it is a small
//! line protocol over TCP that answers "who has expert N" against this node's
//! immutable, content-addressed index. It exposes exactly the query v1's fetch
//! policy needs (expert_id -> holder + offset/len/digest) without yet doing
//! replication or heat tracking.
//!
//! Protocol (line-delimited; GETR responses carry a binary payload after the
//! header line):
//!   HELLO         -> LOOM/0 experts=<n> unique_bytes=<b>
//!   COUNT         -> <n>
//!   HAS <id>      -> PRESENT <id> off=<o> len=<l> sha256=<hex> | ABSENT <id> | ERR range
//!   PING          -> PONG
//!   -- weight-range distribution (when a GGUF store is attached) --
//!   MANIFEST      -> MANIFEST version=<hex> size=<b> ranges=<n> range_size=<b> mode=<fixed|expert> resident=<n>
//!   MANIFESTFILE  -> MANIFESTFILE len=<n>\n<serialized manifest bytes> (digests + extent lists)
//!   DIGEST <i>    -> DIGEST <i> <hex>
//!   DIGESTS       -> DIGESTS <n>\n<hex>\n x n     (bulk, for bootstrap)
//!   HOLDINGS      -> HOLDINGS <hex bitmap>      (bit i = holds range i; the
//!                    same compact summary destined for ENR metadata + gossip)
//!   GETR <i>      -> DATA <i> len=<l> sha256=<hex>\n<raw bytes> | ERR not_held
//!   -- bootnode (when a committee registry is attached; SPEC.md) --
//!   JOIN addr=<h:p> fraction=<f>
//!                 -> COMMITTEE id=<n> members=<a,b,...> assign=<hex bitmap>
//!   COMMITTEES    -> COMMITTEES <n>\n then n summary lines (debug)
//!   -- gossip / peer exchange --
//!   GOSSIP addr=<h:p> version=<hex> holdings=<hex>
//!                 -> PEERS <n>\n then n lines "addr=.. version=.. holdings=.."
//!                    (the responder's own entry first, then its peer table;
//!                    the announcer is merged into the responder's table)
//!   TABLE         -> same as GOSSIP response but without announcing (debug)
//!   <other>       -> ERR unknown

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const checkpoint = @import("../engine/checkpoint.zig");
const hashmod = @import("../core/hash.zig");
const weights = @import("weights.zig");
const peers = @import("peers.zig");
const stats = @import("../core/stats.zig");
const bootnode = @import("bootnode.zig");
const rag_store = @import("../rag/store.zig");
const wire = @import("wire.zig");
const sockopt = @import("../core/sockopt.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    entries: []const checkpoint.ExpertEntry,
    unique_bytes: u64,
    addr: []const u8,
    port: u16,
    /// Attached GGUF weight store, if this node participates in distribution.
    /// Reads are lock-free (immutable manifest, pread on a shared handle);
    /// holdings bits are atomic.
    store: ?*weights.Store = null,
    /// Dynamic peer table for gossip; when set, GOSSIP/TABLE are served.
    table: ?*peers.Table = null,
    /// Committee registry; when set, this node acts as a bootnode (JOIN).
    boot: ?*bootnode.Registry = null,
    /// This node's committee id (wire.NO_COMMITTEE when not in one).
    committee_id: u32 = 0xFFFF_FFFF,
    network_id: u64 = 0,
    rag: ?*rag_store.Store = null,
    rag_round: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// In-flight expert requests being served (the heartbeat load hint).
    load: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Append-only JSONL sink for opt-in alpha telemetry (--alpha-ingest).
    /// Null means this node declines METRICS lines. Writes serialize on
    /// metrics_mu; the file is capped at 256 MB and then refused.
    metrics_path: ?[]const u8 = null,
    metrics_mu: Io.Mutex = .init,
    /// Per-shard serve counts (allocated alongside the store). GETR bumps
    /// them; HEAT answers with the hottest ids so a syncing peer can fetch
    /// in usefulness order instead of index order.
    heat: ?[]std.atomic.Value(u32) = null,
    /// Our own advertised "host:port" (what we tell peers to dial us on).
    advertise: []const u8 = "",
};

fn selfLine(ctx: *Ctx, gpa: std.mem.Allocator, list: *std.ArrayList(u8)) !void {
    const zero_version = "0" ** 64;
    if (ctx.store) |st| {
        const vhex = hashmod.toHex(st.manifest.version);
        const hhex = try st.holdings.toHex(gpa);
        defer gpa.free(hhex);
        try list.print(gpa, "addr={s} version={s} holdings={s}\n", .{ ctx.advertise, vhex, hhex });
    } else {
        try list.print(gpa, "addr={s} version={s} holdings=\n", .{ ctx.advertise, zero_version });
    }
}

fn sendPeerList(ctx: *Ctx, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;
    const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
    var body = std.ArrayList(u8).empty;
    defer body.deinit(gpa);
    try selfLine(ctx, gpa, &body);
    const n = try table.dump(gpa, &body);
    try wi.print("PEERS {d}\n", .{n + 1});
    try wi.writeAll(body.items);
}

const Conn = struct { ctx: *Ctx, stream: net.Stream };

/// Bound on concurrent connection-handler threads (audit #7 P1 thread-per-
/// accept DoS). Excess connections are refused (closed) rather than spawning
/// unbounded threads.
const MAX_CONNS: u32 = 256;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.addr, ctx.port);
    var server = try address.listen(ctx.io, .{ .reuse_address = true });
    sockopt.ensureReaper(ctx.io);
    defer server.deinit(ctx.io);

    while (true) {
        const stream = server.accept(ctx.io) catch continue;
        if (live_conns.fetchAdd(1, .monotonic) >= MAX_CONNS) {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io); // over capacity — shed load
            continue;
        }
        const conn = ctx.gpa.create(Conn) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            continue;
        };
        conn.* = .{ .ctx = ctx, .stream = stream };
        const t = std.Thread.spawn(.{}, connThread, .{conn}) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            ctx.gpa.destroy(conn);
            continue;
        };
        t.detach();
    }
}

fn connThread(conn: *Conn) void {
    handleConn(conn.ctx, conn.stream) catch {};
    conn.ctx.gpa.destroy(conn);
    _ = live_conns.fetchSub(1, .monotonic);
}

fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    const dl = sockopt.trackServe(ctx.io, stream);
    defer sockopt.untrack(ctx.io, dl);
    // GLM-scale holdings bitmaps make GOSSIP/HOLDINGS lines ~5 KB; size the
    // line buffers well past that
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    while (true) {
        const raw = ri.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        try handleLine(ctx, line, ri, wi);
        try wi.flush();
        // The deadline is there to hang up on a socket that is *idle* -- a
        // slowloris holding a connection open. A peer that just took a shard
        // is the opposite of idle, so push it out, exactly as the HTTP server
        // does between tokens.
        //
        // Without this the whole connection lived inside one 30 s budget,
        // which is invisible on loopback (a full sync finishes in seconds) and
        // fatal over a real link: a 905-shard bootstrap is ~5.4 GB, about 190 s
        // at the 29 MB/s measured Mac-to-Hetzner, so the server shut the socket
        // down mid-transfer and the client reported EndOfStream.
        sockopt.refreshServe(ctx.io, dl);
    }
}

fn handleLine(ctx: *Ctx, line: []const u8, ri: *Io.Reader, wi: *Io.Writer) !void {
    if (line.len == 0) return;
    if (std.mem.eql(u8, line, "PING")) {
        try wi.print("PONG\n", .{});
    } else if (std.mem.eql(u8, line, "HELLO")) {
        try wi.print("LOOM/0 experts={d} unique_bytes={d}\n", .{ ctx.entries.len, ctx.unique_bytes });
    } else if (std.mem.eql(u8, line, "COUNT")) {
        try wi.print("{d}\n", .{ctx.entries.len});
    } else if (std.mem.startsWith(u8, line, "HAS ")) {
        const arg = std.mem.trim(u8, line[4..], " ");
        const id = std.fmt.parseInt(usize, arg, 10) catch {
            try wi.print("ERR bad_id\n", .{});
            return;
        };
        if (id >= ctx.entries.len) {
            try wi.print("ERR range\n", .{});
            return;
        }
        const e = ctx.entries[id];
        const hex = hashmod.toHex(e.digest);
        try wi.print("PRESENT {d} off={d} len={d} sha256={s}\n", .{ id, e.offset, e.len, hex });
    } else if (std.mem.startsWith(u8, line, "METRICS ")) {
        // Opt-in alpha telemetry ingest (SPEC.md). Only persisted when the
        // operator passed --alpha-ingest; every other node declines, so a
        // reporter never accumulates state on peers that did not ask for it.
        const path = ctx.metrics_path orelse return wi.print("ERR no_ingest\n", .{});
        const json = line["METRICS ".len..];
        if (json.len == 0 or json.len > 8192) return wi.print("ERR bad_metrics\n", .{});
        appendMetricsLine(ctx, path, json) catch return wi.print("ERR ingest_failed\n", .{});
        try wi.print("OK\n", .{});
    } else if (std.mem.eql(u8, line, "HEAT")) {
        // The hottest shard ids this node has served, descending. Usage is
        // Zipfian, so a joiner that syncs these first is useful after a
        // fraction of the transfer. Best-effort: no counts yet means an
        // empty list, and an old peer answers ERR unknown -- the client
        // falls back to index order either way.
        const h = ctx.heat orelse return wi.print("HEAT n=0 ids=\n", .{});
        const cap = @min(h.len, 2048);
        const Pair = struct { id: usize, n: u32 };
        const pairs = try ctx.gpa.alloc(Pair, h.len);
        defer ctx.gpa.free(pairs);
        var used: usize = 0;
        for (h, 0..) |*c, i| {
            const n = c.load(.monotonic);
            if (n == 0) continue;
            pairs[used] = .{ .id = i, .n = n };
            used += 1;
        }
        std.mem.sort(Pair, pairs[0..used], {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.n > b.n;
            }
        }.lt);
        const k = @min(used, cap);
        try wi.print("HEAT n={d} ids=", .{k});
        for (pairs[0..k], 0..) |pr, j| {
            if (j != 0) try wi.print(",", .{});
            try wi.print("{d}", .{pr.id});
        }
        try wi.print("\n", .{});
    } else if (std.mem.eql(u8, line, "MANIFEST")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const m = &store.manifest;
        try wi.print("MANIFEST version={s} size={d} ranges={d} range_size={d} mode={s} resident={d}\n", .{
            hashmod.toHex(m.version), m.file_size,  m.nRanges(), m.range_size,
            @tagName(m.mode),         m.n_resident,
        });
    } else if (std.mem.eql(u8, line, "MANIFESTFILE")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const text = try store.manifest.serialize(ctx.gpa);
        defer ctx.gpa.free(text);
        try wi.print("MANIFESTFILE len={d}\n", .{text.len});
        try wi.writeAll(text);
    } else if (std.mem.startsWith(u8, line, "DIGEST ")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const i = std.fmt.parseInt(usize, std.mem.trim(u8, line[7..], " "), 10) catch {
            return wi.print("ERR bad_id\n", .{});
        };
        if (i >= store.manifest.nRanges()) return wi.print("ERR range\n", .{});
        try wi.print("DIGEST {d} {s}\n", .{ i, hashmod.toHex(store.manifest.digests[i]) });
    } else if (std.mem.eql(u8, line, "DIGESTS")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        try wi.print("DIGESTS {d}\n", .{store.manifest.nRanges()});
        for (store.manifest.digests) |d| try wi.print("{s}\n", .{hashmod.toHex(d)});
    } else if (std.mem.eql(u8, line, "HOLDINGS")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const hex = try store.holdings.toHex(ctx.gpa);
        defer ctx.gpa.free(hex);
        try wi.print("HOLDINGS {s}\n", .{hex});
    } else if (std.mem.startsWith(u8, line, "GETR ")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const i = std.fmt.parseInt(usize, std.mem.trim(u8, line[5..], " "), 10) catch {
            return wi.print("ERR bad_id\n", .{});
        };
        if (i >= store.manifest.nRanges()) return wi.print("ERR range\n", .{});
        if (!store.holdings.has(i)) return wi.print("ERR not_held\n", .{});
        const buf = try ctx.gpa.alloc(u8, @intCast(store.manifest.rangeLen(i)));
        defer ctx.gpa.free(buf);
        const data = store.readRangeVerified(i, buf) catch return wi.print("ERR read\n", .{});
        if (ctx.heat) |h| _ = h[i].fetchAdd(1, .monotonic);
        try wi.print("DATA {d} len={d} sha256={s}\n", .{ i, data.len, hashmod.toHex(store.manifest.digests[i]) });
        try wi.writeAll(data);
    } else if (std.mem.startsWith(u8, line, "FRAME ")) {
        const len = std.fmt.parseInt(usize, line[6..], 10) catch return wi.print("ERR bad_frame\n", .{});
        // read the body in chunks so memory tracks bytes actually delivered
        // (security issue #27), not the length a peer merely claims
        const raw = wire.readFrameBodyAlloc(ctx.gpa, ri, len) catch return wi.print("ERR bad_frame\n", .{});
        defer ctx.gpa.free(raw);
        try handleFrame(ctx, raw, wi);
    } else if (std.mem.startsWith(u8, line, "JOIN ")) {
        const reg = ctx.boot orelse return wi.print("ERR no_bootnode\n", .{});
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const addr = fieldOf(line, "addr") orelse return wi.print("ERR bad_join\n", .{});
        const frac_s = fieldOf(line, "fraction") orelse return wi.print("ERR bad_join\n", .{});
        const frac = std.fmt.parseFloat(f32, frac_s) catch return wi.print("ERR bad_join\n", .{});
        const n_experts = store.manifest.nRanges() - store.manifest.n_resident;
        const capacity: usize = @intFromFloat(@max(1.0, @round(std.math.clamp(frac, 0.0, 1.0) * @as(f32, @floatFromInt(n_experts)))));
        var out = reg.join(addr, capacity) catch return wi.print("ERR join_failed\n", .{});
        defer out.deinit(ctx.gpa);
        const hex = try out.assign.toHex(ctx.gpa);
        defer ctx.gpa.free(hex);
        try wi.print("COMMITTEE id={d} members=", .{out.committee_id});
        for (out.members, 0..) |m, k| {
            if (k != 0) try wi.print(",", .{});
            try wi.print("{s}", .{m});
        }
        try wi.print(" assign={s}\n", .{hex});
    } else if (std.mem.eql(u8, line, "COMMITTEES")) {
        const reg = ctx.boot orelse return wi.print("ERR no_bootnode\n", .{});
        var body = std.ArrayList(u8).empty;
        defer body.deinit(ctx.gpa);
        const n = try reg.summary(ctx.gpa, &body);
        try wi.print("COMMITTEES {d}\n", .{n});
        try wi.writeAll(body.items);
    } else if (std.mem.startsWith(u8, line, "GOSSIP ")) {
        const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
        const addr = fieldOf(line, "addr") orelse return wi.print("ERR bad_gossip\n", .{});
        const version = fieldOf(line, "version") orelse return wi.print("ERR bad_gossip\n", .{});
        const holdings = fieldOf(line, "holdings") orelse "";
        // inbound: this peer opened the connection, so it is demonstrably alive
        _ = table.merge(addr, version, holdings, peers.NO_COMMITTEE, 0, stats.nowNs(ctx.io), .first_hand) catch {
            return wi.print("ERR bad_gossip\n", .{});
        };
        try sendPeerList(ctx, wi);
    } else if (std.mem.eql(u8, line, "TABLE")) {
        try sendPeerList(ctx, wi);
    } else {
        try wi.print("ERR unknown\n", .{});
    }
}

/// Dispatch a decoded wire frame (SPEC.md wire messages v1).
fn handleFrame(ctx: *Ctx, raw: []const u8, wi: *Io.Writer) !void {
    const dec = wire.decodeFrame(ctx.gpa, raw) catch return wi.print("ERR bad_frame\n", .{});
    defer ctx.gpa.free(dec.body);
    switch (dec.ty) {
        .heartbeat => {
            // A heartbeat from another LLM network is refused outright --
            // the caller learns it dialed the wrong network instead of
            // silently exchanging state across it.
            if (wire.Heartbeat.parseBody(dec.body)) |hb| {
                if (hb.network_id != ctx.network_id) return wi.print("ERR wrong_network\n", .{});
            } else |_| return wi.print("ERR bad_frame\n", .{});
            // respond with our own state: one exchange refreshes both sides
            const resp = selfHeartbeat(ctx);
            const body = try resp.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .heartbeat_resp, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .announce => {
            const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
            const ann = wire.Announce.parseBody(dec.body) catch return wi.print("ERR bad_frame\n", .{});
            if (ann.network_id != ctx.network_id) return wi.print("ERR wrong_network\n", .{});
            const vhex = hashmod.toHex(ann.manifest_version);
            const hhex = try bytesToHexAlloc(ctx.gpa, ann.holdings_bitmap);
            defer ctx.gpa.free(hhex);
            _ = table.merge(ann.addr, &vhex, hhex, ann.committee_id, ann.holdings_seq, stats.nowNs(ctx.io), .first_hand) catch {};
            try sendAnnounceBatch(ctx, wi);
        },
        .rag_inv => {
            const st = ctx.rag orelse return wi.print("ERR no_rag\n", .{});
            const inv = wire.RagHashes.parseBody(ctx.gpa, dec.body) catch return wi.print("ERR bad_frame\n", .{});
            defer ctx.gpa.free(inv);
            var want = std.ArrayList([32]u8).empty;
            defer want.deinit(ctx.gpa);
            for (inv) |h| {
                if (want.items.len >= wire.RAG_PUSH_MAX) break;
                if (!st.has(h)) try want.append(ctx.gpa, h);
            }
            var resp = wire.RagHashes{ .hashes = want.items };
            const body = try resp.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_want, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .rag_push => {
            const st = ctx.rag orelse return;
            var it = wire.RagPush.iterate(dec.body) catch return;
            while (it.next() catch null) |text| _ = st.add(text);
        },
        .rag_inv_req => {
            const st = ctx.rag orelse return wi.print("ERR no_rag\n", .{});
            var hashes: [wire.RAG_INV_MAX][32]u8 = undefined;
            const round = ctx.rag_round.fetchAdd(1, .monotonic);
            const n = st.invWindow(round, hashes[0..]);
            var inv = wire.RagHashes{ .hashes = hashes[0..n] };
            const body = try inv.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_inv, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .rag_want => {
            // A dialer asking US for chunks it lacks (the pull direction; a
            // NAT'd node can only ever dial out).
            const st = ctx.rag orelse return;
            const wanted = wire.RagHashes.parseBody(ctx.gpa, dec.body) catch return;
            defer ctx.gpa.free(wanted);
            var texts = std.ArrayList([]u8).empty;
            defer {
                for (texts.items) |t| ctx.gpa.free(t);
                texts.deinit(ctx.gpa);
            }
            for (wanted) |h| {
                if (texts.items.len >= wire.RAG_PUSH_MAX) break;
                if (st.textByHashAlloc(ctx.gpa, h)) |t| try texts.append(ctx.gpa, t);
            }
            const body = try wire.RagPush.encodeBody(ctx.gpa, @ptrCast(texts.items));
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_push, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .expert_request => {
            const req = wire.ExpertRequest.parseBody(dec.body) catch {
                return sendExpertResponse(ctx, wi, .{ .request_id = 0, .status = .bad_request, .shard_id = 0 });
            };
            const store = ctx.store orelse {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            };
            if (!std.mem.eql(u8, &req.manifest_version, &store.manifest.version)) {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .version_mismatch, .shard_id = req.shard_id });
            }
            if (req.shard_id >= store.manifest.nRanges() or !store.holdings.has(req.shard_id)) {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            }
            _ = ctx.load.fetchAdd(1, .monotonic);
            defer _ = ctx.load.fetchSub(1, .monotonic);
            const buf = try ctx.gpa.alloc(u8, @intCast(store.manifest.rangeLen(req.shard_id)));
            defer ctx.gpa.free(buf);
            const data = store.readRangeVerified(req.shard_id, buf) catch {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            };
            try sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .ok, .shard_id = req.shard_id, .payload = data });
        },
        else => try wi.print("ERR unexpected_frame\n", .{}),
    }
}

/// Our own heartbeat/announce state snapshot.
pub fn selfHeartbeat(ctx: *Ctx) wire.Heartbeat {
    var hb = wire.Heartbeat{
        .network_id = ctx.network_id,
        .committee_id = ctx.committee_id,
        .load = @intCast(@min(ctx.load.load(.monotonic), std.math.maxInt(u16))),
        .sent_at_ns = @intCast(@mod(stats.nowNs(ctx.io), std.math.maxInt(i64))),
        .addr = ctx.advertise,
    };
    if (ctx.store) |st| {
        hb.manifest_version = st.manifest.version;
        hb.holdings_seq = st.holdingsSeq(); // truly monotonic (audit #7 P1)
        // hash a consistent snapshot, not a bitmap being mutated underneath
        // (security issue #31): otherwise the advertised digest can describe a
        // state that never existed
        var buf: [4096]u8 = undefined;
        if (st.holdings.bits.len <= buf.len) {
            for (st.holdings.bits, 0..) |*bp, i| buf[i] = @atomicLoad(u8, bp, .monotonic);
            hb.holdings_digest = hashmod.hashBlock(buf[0..st.holdings.bits.len]);
        } else if (st.holdings.snapshotAlloc(ctx.gpa)) |snap| {
            defer ctx.gpa.free(snap);
            hb.holdings_digest = hashmod.hashBlock(snap);
        } else |_| {}
    }
    return hb;
}

fn appendMetricsLine(ctx: *Ctx, path: []const u8, json: []const u8) !void {
    ctx.metrics_mu.lockUncancelable(ctx.io);
    defer ctx.metrics_mu.unlock(ctx.io);
    const f = Io.Dir.cwd().openFile(ctx.io, path, .{ .mode = .write_only }) catch
        try Io.Dir.cwd().createFile(ctx.io, path, .{});
    defer f.close(ctx.io);
    const size = (try f.stat(ctx.io)).size;
    if (size > 256 * 1024 * 1024) return error.IngestFull;
    try f.writePositionalAll(ctx.io, json, size);
    try f.writePositionalAll(ctx.io, "\n", size + json.len);
}

fn bytesToHexAlloc(gpa: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const hex = "0123456789abcdef";
    const out = try gpa.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |b, i| {
        out[i * 2] = hex[b >> 4];
        out[i * 2 + 1] = hex[b & 0xf];
    }
    return out;
}

fn hexToBytesAlloc(gpa: std.mem.Allocator, hexs: []const u8) ![]u8 {
    if (hexs.len % 2 != 0) return error.BadHex;
    const out = try gpa.alloc(u8, hexs.len / 2);
    errdefer gpa.free(out);
    for (out, 0..) |*b, i| {
        b.* = std.fmt.parseInt(u8, hexs[i * 2 ..][0..2], 16) catch return error.BadHex;
    }
    return out;
}

/// Respond to a gossip announce: our own entry + the whole table as an
/// AnnounceBatch frame (snappy compresses across the similar bitmaps).
fn sendAnnounceBatch(ctx: *Ctx, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;
    var bodies = std.ArrayList([]u8).empty;
    defer {
        for (bodies.items) |b| gpa.free(b);
        bodies.deinit(gpa);
    }

    // self entry
    {
        var self_snap: ?[]u8 = null;
        defer if (self_snap) |sp| gpa.free(sp);
        var self_ann = wire.Announce{ .committee_id = ctx.committee_id, .addr = ctx.advertise };
        if (ctx.store) |st| {
            self_ann.manifest_version = st.manifest.version;
            self_ann.holdings_seq = @intCast(st.holdings.count());
            self_snap = try st.holdings.snapshotAlloc(gpa);
            self_ann.holdings_bitmap = self_snap.?;
        }
        try bodies.append(gpa, try self_ann.encodeBody(gpa));
    }

    // table entries
    const table = ctx.table.?;
    {
        table.mutex.lockUncancelable(ctx.io);
        defer table.mutex.unlock(ctx.io);
        for (table.entries.items) |e| {
            var ver: [32]u8 = undefined;
            for (&ver, 0..) |*b, i| {
                b.* = std.fmt.parseInt(u8, e.version_hex[i * 2 ..][0..2], 16) catch 0;
            }
            const bitmap = hexToBytesAlloc(gpa, e.holdings_hex) catch continue;
            defer gpa.free(bitmap);
            const ann = wire.Announce{
                .committee_id = e.committee_id,
                .manifest_version = ver,
                .addr = e.addr,
                .holdings_bitmap = bitmap,
            };
            try bodies.append(gpa, try ann.encodeBody(gpa));
        }
    }

    const batch = try wire.AnnounceBatch.encodeBodies(gpa, bodies.items);
    defer gpa.free(batch);
    const frame = try wire.encodeFrame(gpa, .announce_batch, batch);
    defer gpa.free(frame);
    try wire.writeFrame(wi, frame);
}

fn sendExpertResponse(ctx: *Ctx, wi: *Io.Writer, resp: wire.ExpertResponse) !void {
    const body = try resp.encodeBody(ctx.gpa);
    defer ctx.gpa.free(body);
    const frame = try wire.encodeFrame(ctx.gpa, .expert_response, body);
    defer ctx.gpa.free(frame);
    try wire.writeFrame(wi, frame);
}

fn fieldOf(line: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key) and tok.len > key.len and tok[key.len] == '=') {
            return tok[key.len + 1 ..];
        }
    }
    return null;
}
