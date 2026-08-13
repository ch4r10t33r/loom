//! Boot-time weight sync (ROADMAP.md v1 #3): a new node requests weight ranges
//! from a peer via the request-response protocol in p2p.zig instead of
//! re-downloading the model from origin.
//!
//! Flow: connect -> MANIFEST (adopt version/size/range_size) -> DIGESTS (verify
//! their Merkle root matches the advertised version) -> HOLDINGS (what the peer
//! can serve) -> pick our own random subset (--hold-fraction) -> GETR each
//! held-and-available range, verifying every range against its digest before it
//! touches disk. A digest mismatch rejects the range (poisoned-peer defense).

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const stats_mod = @import("../core/stats.zig");
const hashmod = @import("../core/hash.zig");
const weights = @import("weights.zig");
const http_bootstrap = @import("http_bootstrap.zig");
const dns = @import("dns.zig");
const sockopt = @import("../core/sockopt.zig");
const stripe = @import("stripe.zig");

pub const Result = struct {
    store: weights.Store,
    wanted: usize, // ranges we chose to hold
    fetched: usize, // ranges actually obtained from the peer
    bytes: u64,
};

pub const PeerAddr = struct {
    host: []const u8,
    port: u16,

    /// Parse "host:port".
    pub fn parse(s: []const u8) !PeerAddr {
        const colon = std.mem.lastIndexOfScalar(u8, s, ':') orelse return error.BadPeerAddr;
        const port = std.fmt.parseInt(u16, s[colon + 1 ..], 10) catch return error.BadPeerAddr;
        if (colon == 0) return error.BadPeerAddr;
        return .{ .host = s[0..colon], .port = port };
    }
};

pub const Peer = struct {
    stream: net.Stream,
    io: Io,
    r: net.Stream.Reader,
    w: net.Stream.Writer,
    rbuf: []u8,
    wbuf: []u8,
    deadline: ?usize = null,

    pub fn connect(gpa: std.mem.Allocator, io: Io, addr: PeerAddr) !*Peer {
        const address = try dns.resolve(io, addr.host, addr.port);
        const stream = try address.connect(io, .{ .mode = .stream });
        const p = try gpa.create(Peer);
        errdefer gpa.destroy(p);
        p.io = io;
        p.stream = stream;
        p.rbuf = try gpa.alloc(u8, 1 << 16);
        p.wbuf = try gpa.alloc(u8, 4096);
        p.r = stream.reader(io, p.rbuf);
        p.w = stream.writer(io, p.wbuf);
        p.deadline = sockopt.trackPeer(io, stream);
        return p;
    }

    pub fn close(p: *Peer, gpa: std.mem.Allocator) void {
        sockopt.untrack(p.io, p.deadline); // before close: see sockopt fd-reuse note
        p.stream.close(p.io);
        gpa.free(p.rbuf);
        gpa.free(p.wbuf);
        gpa.destroy(p);
    }

    pub fn send(p: *Peer, comptime fmt: []const u8, args: anytype) !void {
        try p.w.interface.print(fmt, args);
        try p.w.interface.flush();
    }

    pub fn recvLine(p: *Peer) ![]u8 {
        const raw = try p.r.interface.takeDelimiterInclusive('\n');
        return @constCast(std.mem.trimEnd(u8, raw, "\r\n"));
    }
};

fn field(line: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key) and tok.len > key.len and tok[key.len] == '=') {
            return tok[key.len + 1 ..];
        }
    }
    return null;
}

fn parseDigestHex(s: []const u8) !hashmod.Digest {
    if (s.len != 64) return error.BadDigestHex;
    var d: hashmod.Digest = undefined;
    for (&d, 0..) |*b, i| {
        b.* = std.fmt.parseInt(u8, s[i * 2 ..][0..2], 16) catch return error.BadDigestHex;
    }
    return d;
}

pub const FetchStats = struct { fetched: usize = 0, bytes: u64 = 0 };

/// The order to fetch missing wanted shards from a peer: mandatory resident
/// chunks first (indices [0, n_resident)), then the peer's hottest shards
/// (usage is Zipfian, so the head of this list covers most token reads),
/// then everything else in index order. Caller frees the slice.
fn fetchOrder(
    gpa: std.mem.Allocator,
    n_ranges: usize,
    n_resident: usize,
    heat_ids: []const usize,
) ![]usize {
    var order = try gpa.alloc(usize, n_ranges);
    var queued = try gpa.alloc(bool, n_ranges);
    defer gpa.free(queued);
    @memset(queued, false);
    var used: usize = 0;
    var i: usize = 0;
    while (i < n_resident and i < n_ranges) : (i += 1) {
        order[used] = i;
        queued[i] = true;
        used += 1;
    }
    for (heat_ids) |id| {
        if (id >= n_ranges or queued[id]) continue;
        order[used] = id;
        queued[id] = true;
        used += 1;
    }
    i = 0;
    while (i < n_ranges) : (i += 1) {
        if (queued[i]) continue;
        order[used] = i;
        used += 1;
    }
    std.debug.assert(used == n_ranges);
    return order;
}

/// Ask a peer which shards run hottest. Best-effort: an old peer answers
/// ERR unknown, a fresh one has no counts; both mean index order.
fn fetchHeat(gpa: std.mem.Allocator, peer: *Peer) ![]usize {
    try peer.send("HEAT\n", .{});
    const line = try peer.recvLine();
    if (!std.mem.startsWith(u8, line, "HEAT ")) return gpa.alloc(usize, 0);
    const csv = field(line, "ids") orelse return gpa.alloc(usize, 0);
    var ids = std.ArrayList(usize).empty;
    errdefer ids.deinit(gpa);
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        const id = std.fmt.parseInt(usize, tok, 10) catch continue;
        try ids.append(gpa, id);
        if (ids.items.len >= 4096) break;
    }
    return ids.toOwnedSlice(gpa);
}

test "capWanted trims excess expert bits, never resident ones (issue #252)" {
    const gpa = std.testing.allocator;
    var wanted = try weights.Holdings.initEmpty(gpa, 10);
    defer wanted.deinit(gpa);
    // 2 resident + 6 of 8 expert bits set
    for (0..8) |i| wanted.set(i);
    capWanted(&wanted, 2, 3);
    try std.testing.expectEqual(@as(usize, 5), wanted.count()); // 2 resident + 3
    try std.testing.expect(wanted.has(0) and wanted.has(1)); // resident untouched
    try std.testing.expect(wanted.has(2) and wanted.has(4) and !wanted.has(5));
    // cap above the set count is a no-op
    capWanted(&wanted, 2, 8);
    try std.testing.expectEqual(@as(usize, 5), wanted.count());
}

test "fetchOrder: resident first, then heat, then the tail in index order" {
    const gpa = std.testing.allocator;
    const heat = [_]usize{ 7, 3, 99, 7 }; // 99 out of range, 7 repeated
    const order = try fetchOrder(gpa, 10, 2, &heat);
    defer gpa.free(order);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 7, 3, 2, 4, 5, 6, 8, 9 }, order);
}

/// Live sync progress. The initial sync is gigabytes over one connection;
/// without a heartbeat the console is silent for minutes and a first-time
/// tester has no idea whether anything is happening (it was the first thing
/// a real alpha user hit). Prints at most every 5 s, always flushed.
pub const Progress = struct {
    out: *Io.Writer,
    io: Io,
    total: usize,
    sources: usize = 1,
    base_fetched: usize = 0,
    base_bytes: u64 = 0,
    t0: i128 = 0,
    last_ns: i128 = 0,

    pub fn init(out: *Io.Writer, io: Io, total: usize, sources: usize) Progress {
        const now = stats_mod.nowNs(io);
        return .{ .out = out, .io = io, .total = total, .sources = sources, .t0 = now, .last_ns = now - 5_000_000_000 };
    }

    fn tick(p: *Progress, peer_fetched: usize, peer_bytes: u64) void {
        const now = stats_mod.nowNs(p.io);
        if (now - p.last_ns < 5_000_000_000) return;
        p.last_ns = now;
        const done = p.base_fetched + peer_fetched;
        const bytes = p.base_bytes + peer_bytes;
        const el_s = @as(f64, @floatFromInt(now - p.t0)) / 1e9;
        const mb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
        const rate = if (el_s > 0.5) mb / el_s else 0;
        var eta_buf: [32]u8 = undefined;
        var eta: []const u8 = "?";
        if (done > 0 and done < p.total) {
            const eta_s: u64 = @intFromFloat(el_s / @as(f64, @floatFromInt(done)) * @as(f64, @floatFromInt(p.total - done)));
            eta = std.fmt.bufPrint(&eta_buf, "{d}m{d:0>2}s", .{ eta_s / 60, eta_s % 60 }) catch "?";
        }
        p.out.print("  sync {d}/{d} shards  {d:.0} MB  {d:.1} MB/s  ~{s} left  ({d} peer(s))\n", .{
            done, p.total, mb, rate, eta, p.sources,
        }) catch {};
        p.out.flush() catch {};
    }
};

/// Fetch the store's wanted-but-missing ranges from one peer. Guards against
/// cross-version mixing: a peer advertising a different manifest version (e.g.
/// across a future hardfork) is rejected wholesale. Every range is verified
/// against its digest before it touches disk. Used by both bootstrap and the
/// eager repair loop.
pub fn fetchFromPeer(gpa: std.mem.Allocator, io: Io, store: *weights.Store, addr: PeerAddr, prog: ?*Progress) !FetchStats {
    var got = FetchStats{};
    return fetchFromPeerInto(gpa, io, store, addr, &got, prog) catch |e| {
        // Shards already written stay written -- `writeRange` persists each one
        // as it lands -- so losing the count on error made a mostly-successful
        // sync report "synced 0/905 ranges (0.0 MB)" while the holdings bitmap
        // plainly disagreed. Hand back what actually arrived.
        if (got.fetched > 0) return got;
        return e;
    };
}

fn fetchFromPeerInto(
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    addr: PeerAddr,
    out: *FetchStats,
    prog: ?*Progress,
) !FetchStats {
    const m = &store.manifest;
    const n_ranges = m.nRanges();

    const peer = try Peer.connect(gpa, io, addr);
    defer peer.close(gpa);

    // same-version guard
    try peer.send("MANIFEST\n", .{});
    const mline = try peer.recvLine();
    if (!std.mem.startsWith(u8, mline, "MANIFEST ")) return error.PeerHasNoStore;
    const version = try parseDigestHex(field(mline, "version") orelse return error.BadManifestLine);
    if (!hashmod.eql(version, m.version)) return error.PeerVersionMismatch;

    // what can this peer serve?
    try peer.send("HOLDINGS\n", .{});
    const hline = try peer.recvLine();
    if (!std.mem.startsWith(u8, hline, "HOLDINGS ")) return error.BadHoldingsLine;
    var peer_holdings = try weights.Holdings.fromHex(gpa, hline[9..], n_ranges);
    defer peer_holdings.deinit(gpa);

    const buf = try gpa.alloc(u8, @intCast(m.maxShardLen()));
    defer gpa.free(buf);

    const heat_ids = fetchHeat(gpa, peer) catch try gpa.alloc(usize, 0);
    defer gpa.free(heat_ids);
    const order = try fetchOrder(gpa, n_ranges, m.n_resident, heat_ids);
    defer gpa.free(order);

    // Everything this peer can give us, in fetch order.
    var needed = std.ArrayList(usize).empty;
    defer needed.deinit(gpa);
    for (order) |i| {
        if (!store.wanted.has(i) or store.holdings.has(i)) continue;
        if (!peer_holdings.has(i)) continue; // this peer can't serve it; another might
        try needed.append(gpa, i);
    }

    const stats = out;

    // Batched path first: one PACKR round trip covers up to 128 shards, so a
    // sync pays RTT per batch instead of per shard. An old peer answers ERR
    // unknown on the first batch and the loop below degrades to GETR.
    var use_pack = true;
    var next: usize = 0;
    while (use_pack and next < needed.items.len) {
        const batch = needed.items[next..@min(next + 128, needed.items.len)];
        var req = std.ArrayList(u8).empty;
        defer req.deinit(gpa);
        try req.print(gpa, "PACKR ", .{});
        for (batch, 0..) |i, j| {
            if (j != 0) try req.print(gpa, ",", .{});
            try req.print(gpa, "{d}", .{i});
        }
        try req.print(gpa, "\n", .{});
        try peer.send("{s}", .{req.items});
        while (true) {
            const rline = try peer.recvLine();
            if (std.mem.startsWith(u8, rline, "END")) break;
            if (std.mem.startsWith(u8, rline, "ABSENT ")) continue;
            if (!std.mem.startsWith(u8, rline, "DATA ")) {
                if (next == 0) {
                    use_pack = false; // old peer: fall back to per-shard GETR
                    break;
                }
                return error.RangeFetchFailed;
            }
            const i = try std.fmt.parseInt(usize, rline[5..std.mem.indexOfScalarPos(u8, rline, 5, ' ').?], 10);
            const len = try std.fmt.parseInt(usize, field(rline, "len") orelse return error.BadDataLine, 10);
            if (len > buf.len) return error.BadDataLine;
            try peer.r.interface.readSliceAll(buf[0..len]);
            // writeRange verifies the digest before anything touches disk
            try store.writeRange(i, buf[0..len]);
            stats.fetched += 1;
            stats.bytes += len;
            if (prog) |p| p.tick(stats.fetched, stats.bytes);
            sockopt.refresh(io, peer.deadline, sockopt.PEER_TIMEOUT_S);
        }
        if (use_pack) next += batch.len;
    }

    for (needed.items[if (use_pack) needed.items.len else 0..]) |i| {
        if (store.holdings.has(i)) continue;

        try peer.send("GETR {d}\n", .{i});
        const rline = try peer.recvLine();
        if (!std.mem.startsWith(u8, rline, "DATA ")) return error.RangeFetchFailed;
        const len = try std.fmt.parseInt(usize, field(rline, "len") orelse return error.BadDataLine, 10);
        if (len > buf.len) return error.BadDataLine;
        try peer.r.interface.readSliceAll(buf[0..len]);
        // writeRange verifies the digest before anything touches disk
        try store.writeRange(i, buf[0..len]);
        stats.fetched += 1;
        stats.bytes += len;
        if (prog) |p| p.tick(stats.fetched, stats.bytes);
        // A shard arrived, so this connection is not idle. `PEER_TIMEOUT_S` is
        // ten seconds and a full sync is thousands of shards over one
        // connection: without this the reaper shut the socket down mid-transfer
        // and the caller reported `synced 0/1571 ranges` as though nothing had
        // gone wrong.
        sockopt.refresh(io, peer.deadline, sockopt.PEER_TIMEOUT_S);
    }
    return stats.*;
}

/// Read a PDATA response for STRIPE g p into buf. Returns false when the
/// peer declines (partial holder, old peer) -- the caller tries another.
fn fetchParity(peer: *Peer, g: usize, p: usize, buf: []u8) !bool {
    try peer.send("STRIPE {d} {d}\n", .{ g, p });
    const line = try peer.recvLine();
    if (!std.mem.startsWith(u8, line, "PDATA ")) return false;
    const len = try std.fmt.parseInt(usize, field(line, "len") orelse return error.BadDataLine, 10);
    // Same manifest on both sides means the same group padding; anything else
    // would desync the stream, so drop this connection's answer entirely.
    if (len != buf.len) return error.BadDataLine;
    try peer.r.interface.readSliceAll(buf);
    return true;
}

/// Striped repair pass (stripe.zig): rebuild a group's missing ranges from
/// parity pieces instead of fetching the originals. A group qualifies when
/// all but at most M_PARITY of its ranges are already held locally and at
/// least one missing range is wanted. Parity rows come from full-group
/// holders (STRIPE), round-robined across peers; the rebuilt ranges pass
/// through writeRange's digest check, so a wrong or malicious parity piece
/// costs one skipped group, never a corrupt store.
///
/// Honesty note (whitepaper decision log): under today's pull model any peer
/// able to serve a group's parity also holds its data and could serve that
/// directly, so this pass buys load spreading across holders, not new
/// availability. Availability arrives when parity-only holders (cold tier)
/// and coded push rollout reuse this wire format.
///
/// Propagation plane only -- never the token loop (principle 7).
pub fn fetchStriped(
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    peer_addrs: []const PeerAddr,
    progress: ?*Io.Writer,
) !FetchStats {
    var out = FetchStats{};
    const m = &store.manifest;
    const n_ranges = m.nRanges();
    if (peer_addrs.len == 0 or n_ranges == 0) return out;

    // Connect and version-guard every responsive peer once up front.
    var conns = std.ArrayList(*Peer).empty;
    defer {
        for (conns.items) |p| p.close(gpa);
        conns.deinit(gpa);
    }
    for (peer_addrs) |addr| {
        const peer = Peer.connect(gpa, io, addr) catch continue;
        const ok = blk: {
            peer.send("MANIFEST\n", .{}) catch break :blk false;
            const line = peer.recvLine() catch break :blk false;
            if (!std.mem.startsWith(u8, line, "MANIFEST ")) break :blk false;
            const v = parseDigestHex(field(line, "version") orelse break :blk false) catch break :blk false;
            break :blk hashmod.eql(v, m.version);
        };
        if (!ok) {
            peer.close(gpa);
            continue;
        }
        try conns.append(gpa, peer);
    }
    if (conns.items.len == 0) return out;

    // One group's pieces live in memory at once: up to K_DATA local data
    // reads plus M_PARITY fetched parity rows, each padded to the group max.
    // Bulk-sync memory, freed before serving starts.
    const max_len: usize = @intCast(m.maxShardLen());
    var bufs: [stripe.K_DATA + stripe.M_PARITY][]u8 = undefined;
    var n_bufs: usize = 0;
    defer for (bufs[0..n_bufs]) |b| gpa.free(b);
    for (&bufs) |*b| {
        b.* = try gpa.alloc(u8, max_len);
        n_bufs += 1;
    }
    var rebuilt: [stripe.K_DATA][]u8 = undefined;
    var n_rebuilt: usize = 0;
    defer for (rebuilt[0..n_rebuilt]) |b| gpa.free(b);
    for (&rebuilt) |*b| {
        b.* = try gpa.alloc(u8, max_len);
        n_rebuilt += 1;
    }

    const n_groups = (n_ranges + stripe.K_DATA - 1) / stripe.K_DATA;
    var rr: usize = 0; // round-robin cursor over conns
    var g: usize = 0;
    groups: while (g < n_groups) : (g += 1) {
        const first = g * stripe.K_DATA;
        const last = @min(first + stripe.K_DATA, n_ranges);
        const k = last - first;
        var missing: [stripe.M_PARITY]usize = undefined;
        var n_missing: usize = 0;
        var want_any = false;
        var plen: usize = 0;
        for (first..last) |i| {
            plen = @max(plen, @as(usize, @intCast(m.rangeLen(i))));
            if (store.holdings.has(i)) continue;
            if (n_missing == stripe.M_PARITY) continue :groups; // hole exceeds parity
            missing[n_missing] = i;
            n_missing += 1;
            if (store.wanted.has(i)) want_any = true;
        }
        if (n_missing == 0 or !want_any) continue;

        // Pieces: held data ranges (zero-padded), then fetched parity rows.
        var pieces: [stripe.K_DATA + stripe.M_PARITY]stripe.Piece = undefined;
        var n_pieces: usize = 0;
        for (first..last) |i| {
            if (!store.holdings.has(i)) continue;
            const rl: usize = @intCast(m.rangeLen(i));
            const b = bufs[n_pieces];
            _ = store.readRangeVerified(i, b[0..rl]) catch continue :groups;
            @memset(b[rl..plen], 0);
            pieces[n_pieces] = .{ .index = i - first, .bytes = b[0..plen] };
            n_pieces += 1;
        }
        var p: usize = 0;
        while (p < n_missing) : (p += 1) {
            const b = bufs[n_pieces];
            var got = false;
            var tries: usize = 0;
            while (tries < conns.items.len) : (tries += 1) {
                const peer = conns.items[rr % conns.items.len];
                rr += 1;
                got = fetchParity(peer, g, p, b[0..plen]) catch false;
                if (got) {
                    sockopt.refresh(io, peer.deadline, sockopt.PEER_TIMEOUT_S);
                    break;
                }
            }
            if (!got) continue :groups; // no reachable full-group holder
            out.bytes += plen;
            pieces[n_pieces] = .{ .index = k + p, .bytes = b[0..plen] };
            n_pieces += 1;
        }

        var outs: [stripe.K_DATA][]u8 = undefined;
        for (0..k) |i| outs[i] = rebuilt[i][0..plen];
        stripe.decode(k, stripe.M_PARITY, pieces[0..n_pieces], outs[0..k]) catch continue :groups;
        for (missing[0..n_missing]) |i| {
            if (!store.wanted.has(i)) continue;
            const rl: usize = @intCast(m.rangeLen(i));
            // writeRange verifies the digest -- the whole integrity story
            store.writeRange(i, outs[i - first][0..rl]) catch continue;
            out.fetched += 1;
            if (progress) |pw| {
                pw.print("  striped repair: range {d} rebuilt from group {d} parity\n", .{ i, g }) catch {};
                pw.flush() catch {};
            }
        }
    }
    return out;
}

/// Fetch the serialized manifest (digests + extent lists) from a peer. The
/// parser verifies the Merkle root pins the digest set. Returns an owned
/// Manifest.
fn adoptManifest(gpa: std.mem.Allocator, peer: *Peer) !weights.Manifest {
    try peer.send("MANIFESTFILE\n", .{});
    const hline = try peer.recvLine();
    if (!std.mem.startsWith(u8, hline, "MANIFESTFILE ")) return error.PeerHasNoStore;
    const len = try std.fmt.parseInt(usize, field(hline, "len") orelse return error.BadManifestLine, 10);
    if (len == 0 or len > 256 * 1024 * 1024) return error.BadManifestLine;
    const bytes = try gpa.alloc(u8, len);
    defer gpa.free(bytes);
    try peer.r.interface.readSliceAll(bytes);
    return weights.parseManifestBytes(gpa, bytes);
}

pub const JoinInfo = struct {
    committee_id: usize,
    members: [][]u8, // owned addr strings
    manifest: weights.Manifest, // owned
    wanted: weights.Holdings, // owned

    pub fn deinit(self: *JoinInfo, gpa: std.mem.Allocator) void {
        for (self.members) |m| gpa.free(m);
        gpa.free(self.members);
        self.manifest.deinit(gpa);
        self.wanted.deinit(gpa);
    }
};

/// Clear expert-range bits past `cap`, keeping the lowest-indexed set bits.
/// Resident bits (below n_resident) are never touched.
fn capWanted(wanted: *weights.Holdings, n_resident: usize, cap: usize) void {
    var kept: usize = 0;
    var i: usize = n_resident;
    while (i < wanted.n) : (i += 1) {
        if (!wanted.has(i)) continue;
        if (kept < cap) {
            kept += 1;
            continue;
        }
        wanted.clear(i);
    }
}

/// SPEC.md join flow: adopt the manifest from the bootnode, then JOIN — the
/// bootnode assigns a committee, member list, and a least-covered-first
/// want-set. Errors with PeerNotBootnode if the peer doesn't run a registry.
pub fn joinSwarm(gpa: std.mem.Allocator, io: Io, addr: PeerAddr, self_addr: []const u8, fraction: f32) !JoinInfo {
    const peer = try Peer.connect(gpa, io, addr);
    defer peer.close(gpa);

    var manifest = try adoptManifest(gpa, peer);
    errdefer manifest.deinit(gpa);

    try peer.send("JOIN addr={s} fraction={d}\n", .{ self_addr, fraction });
    const line = try peer.recvLine();
    if (std.mem.startsWith(u8, line, "ERR no_bootnode")) return error.PeerNotBootnode;
    if (!std.mem.startsWith(u8, line, "COMMITTEE ")) return error.BadJoinResponse;
    const id = try std.fmt.parseInt(usize, field(line, "id") orelse return error.BadJoinResponse, 10);
    const members_csv = field(line, "members") orelse "";
    const assign_hex = field(line, "assign") orelse return error.BadJoinResponse;

    var wanted = try weights.Holdings.fromHex(gpa, assign_hex, manifest.nRanges());
    errdefer wanted.deinit(gpa);

    // Issue #252 defense in depth: a pre-fix bootnode's rejoin path returns
    // the STORED assignment no matter what fraction this JOIN asked for, so
    // a node that ever joined at the default fraction could never come back
    // smaller. Enforce the requested cap locally (same rounding as the
    // bootnode's JOIN handler); against a fixed bootnode this is a no-op.
    const n_experts = manifest.nRanges() - manifest.n_resident;
    const cap: usize = @intFromFloat(@max(1.0, @round(std.math.clamp(fraction, 0.0, 1.0) * @as(f32, @floatFromInt(n_experts)))));
    capWanted(&wanted, manifest.n_resident, cap);

    var members = std.ArrayList([]u8).empty;
    errdefer {
        for (members.items) |m| gpa.free(m);
        members.deinit(gpa);
    }
    var it = std.mem.splitScalar(u8, members_csv, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        try members.append(gpa, try gpa.dupe(u8, tok));
    }

    return .{
        .committee_id = id,
        .members = try members.toOwnedSlice(gpa),
        .manifest = manifest,
        .wanted = wanted,
    };
}

/// Create a store from an adopted manifest + want-set and fill it from
/// `peers` in order. Takes ownership of `manifest` and `wanted`.
pub fn bootstrapWithWanted(
    gpa: std.mem.Allocator,
    io: Io,
    peers: []const PeerAddr,
    store_dir: []const u8,
    manifest: weights.Manifest,
    wanted_bits: weights.Holdings,
    progress: ?*Io.Writer,
    http_url: ?[]const u8,
    http_part_bytes: u64,
) !Result {
    const wanted = wanted_bits.count();
    // Resume if this directory already holds a store for the same manifest.
    //
    // `createFromManifest` opens the model file with `.truncate = true` and
    // starts holdings from empty, which is right for a first sync and
    // catastrophic for a restart: a node that had already fetched 9.4 GB threw
    // all of it away and began again. The manifest version is the identity
    // check -- a store for a different model must not be adopted.
    var store = reopen: {
        // Only a genuinely absent store may fall through to
        // createFromManifest, because that path TRUNCATES model.gguf
        // (security issue #157): collapsing every openDir error into
        // "create fresh" turned any transient or torn-metadata failure into
        // the loss of every range already downloaded. Anything other than
        // file-not-found fails loudly instead; the operator can remove the
        // directory to reset explicitly.
        var existing = weights.openDir(gpa, io, store_dir) catch |e| switch (e) {
            error.FileNotFound => break :reopen null,
            else => {
                if (progress) |pw| {
                    pw.print("  refusing to recreate {s}: metadata unreadable ({t}); remove the directory to reset\n", .{ store_dir, e }) catch {};
                    pw.flush() catch {};
                }
                return e;
            },
        };
        if (!hashmod.eql(existing.manifest.version, manifest.version)) {
            existing.deinit();
            break :reopen null;
        }
        // Keep the caller's want-set; the on-disk one may be from a run with a
        // different --hold-fraction.
        existing.wanted.deinit(gpa);
        existing.wanted = wanted_bits;
        break :reopen existing;
    } orelse try weights.createFromManifest(gpa, io, store_dir, manifest, wanted_bits);
    errdefer store.deinit();

    // Persist manifest/wanted/holdings BEFORE the first fetch (issue #157):
    // a process kill mid-sync previously left gigabytes of verified ranges
    // with no resumable metadata, and the restart discarded them.
    try store.saveSidecars();

    var stats = FetchStats{};
    // HTTP-range mirror tier first (http_bootstrap.zig): bulk bytes come off
    // a static mirror's CDN instead of the origin's uplink; whatever fails
    // or is missing falls through to the peer loop below unchanged.
    if (http_url) |u| {
        if (store.missingCount() > 0) {
            const r = http_bootstrap.fetchMissing(gpa, io, &store, u, http_part_bytes, progress) catch |e| blk: {
                if (progress) |pw| {
                    pw.print("  http sync failed ({t}); falling back to peers\n", .{e}) catch {};
                    pw.flush() catch {};
                }
                break :blk http_bootstrap.Result{};
            };
            stats.fetched += r.fetched;
            stats.bytes += r.bytes;
            store.saveSidecars() catch {}; // checkpoint: the tier's work is resumable
            if (progress) |pw| {
                pw.print("  http sync done: {d} shard(s), {d:.1} MB ({d} left for peers)\n", .{
                    r.fetched, @as(f64, @floatFromInt(r.bytes)) / (1024.0 * 1024.0), store.missingCount(),
                }) catch {};
                pw.flush() catch {};
            }
        }
    }
    var prog: ?Progress = if (progress) |pw| Progress.init(pw, io, wanted, peers.len) else null;
    for (peers) |addr| {
        if (store.missingCount() == 0) break;
        if (prog) |*p| {
            p.base_fetched = stats.fetched;
            p.base_bytes = stats.bytes;
        }
        const s = fetchFromPeer(gpa, io, &store, addr, if (prog) |*p| p else null) catch |e| {
            // Silence here is what made the sync failure invisible: the node
            // printed "synced 0/N" and served anyway, and the only clue was
            // that inference was inexplicably slow.
            if (progress) |pw| {
                pw.print("  sync from {s}:{d} failed: {t}\n", .{ addr.host, addr.port, e }) catch {};
                pw.flush() catch {};
            }
            continue;
        };
        stats.fetched += s.fetched;
        stats.bytes += s.bytes;
        store.saveSidecars() catch {}; // checkpoint after each peer's contribution
        if (progress) |pw| {
            try pw.print("  synced {d}/{d} ranges ({d:.1} MB) after {s}:{d}\n", .{
                stats.fetched, wanted, @as(f64, @floatFromInt(stats.bytes)) / (1024.0 * 1024.0), addr.host, addr.port,
            });
            try pw.flush();
        }
    }
    // Striped fallback: whatever direct fetching left behind may still be
    // recoverable as small holes (<= M_PARITY per group) from parity pieces.
    if (store.missingCount() > 0 and peers.len > 0) {
        if (fetchStriped(gpa, io, &store, peers, progress) catch null) |s| {
            if (s.fetched > 0) {
                stats.fetched += s.fetched;
                stats.bytes += s.bytes;
            }
        }
    }
    try store.saveSidecars();
    return .{ .store = store, .wanted = wanted, .fetched = stats.fetched, .bytes = stats.bytes };
}

/// Bootstrap a local weight store from `peers`: adopt the manifest from the
/// first responsive peer, pick a random `fraction` of ranges to hold (seeded so
/// a restarted node re-picks the same set), then fetch from every peer in turn
/// until the wanted set is satisfied or peers are exhausted. Any remaining
/// shortfall is chased by the eager repair loop after boot.
pub fn bootstrap(
    gpa: std.mem.Allocator,
    io: Io,
    peers: []const PeerAddr,
    store_dir: []const u8,
    fraction: f32,
    seed: u64,
    progress: ?*Io.Writer,
    http_url: ?[]const u8,
    http_part_bytes: u64,
) !Result {
    // adopt a manifest from the first peer that answers
    var manifest: ?weights.Manifest = null;
    for (peers) |addr| {
        const peer = Peer.connect(gpa, io, addr) catch continue;
        defer peer.close(gpa);
        manifest = adoptManifest(gpa, peer) catch continue;
        break;
    }
    var m = manifest orelse return error.NoResponsivePeer;
    var manifest_owned = true; // ownership moves below
    errdefer if (manifest_owned) m.deinit(gpa);

    var wanted_bits = try weights.Holdings.initWanted(gpa, m.nRanges(), m.n_resident, fraction, seed);
    var wanted_owned = true;
    errdefer if (wanted_owned) wanted_bits.deinit(gpa);

    manifest_owned = false;
    wanted_owned = false;
    return bootstrapWithWanted(gpa, io, peers, store_dir, m, wanted_bits, progress, http_url, http_part_bytes);
}

const gguf_fixture_mod = @import("../gguf/gguf.zig");

test "corrupt store metadata refuses recreation instead of truncating (issue #157)" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const model_path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    try gguf_fixture_mod.writeMoeFixture(gpa, io, model_path, 3, "llama");

    var manifest = try weights.buildExpertManifest(gpa, io, model_path);
    var manifest_owned = true;
    defer if (manifest_owned) manifest.deinit(gpa);
    const wanted = try weights.Holdings.initWanted(gpa, manifest.nRanges(), manifest.n_resident, 1.0, 1);
    var sdir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const store_dir = try std.fmt.bufPrint(&sdir_buf, ".zig-cache/tmp/{s}/store", .{tmp.sub_path});

    // Build a store with real sidecars, then note the model file's size.
    {
        var store = try weights.createFromManifest(gpa, io, store_dir, manifest, wanted);
        manifest_owned = false; // store took ownership
        try store.saveSidecars();
        store.deinit();
    }
    var mp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const model_file = try std.fmt.bufPrint(&mp_buf, "{s}/model.gguf", .{store_dir});
    const size_before = blk: {
        const f = try Io.Dir.cwd().openFile(io, model_file, .{});
        defer f.close(io);
        break :blk try f.length(io);
    };

    // Corrupt the manifest sidecar: recovery must fail loudly, NOT fall
    // through to createFromManifest (which truncates model.gguf).
    var cp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sidecar = try std.fmt.bufPrint(&cp_buf, "{s}/ranges.manifest", .{store_dir});
    {
        const f = try Io.Dir.cwd().createFile(io, sidecar, .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, "garbage");
    }

    var manifest2 = try weights.buildExpertManifest(gpa, io, model_path);
    // ownership never transfers: the call below must fail before any store
    // is created from this manifest
    defer manifest2.deinit(gpa);
    var wanted2 = try weights.Holdings.initWanted(gpa, manifest2.nRanges(), manifest2.n_resident, 1.0, 1);
    defer wanted2.deinit(gpa);
    const res = bootstrapWithWanted(gpa, io, &.{}, store_dir, manifest2, wanted2, null, null, 0);
    try std.testing.expect(if (res) |_| false else |_| true);

    const size_after = blk: {
        const f = try Io.Dir.cwd().openFile(io, model_file, .{});
        defer f.close(io);
        break :blk try f.length(io);
    };
    try std.testing.expectEqual(size_before, size_after);
}
