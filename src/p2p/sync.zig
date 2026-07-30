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
const hashmod = @import("../core/hash.zig");
const weights = @import("weights.zig");
const dns = @import("dns.zig");
const sockopt = @import("../core/sockopt.zig");

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

const Peer = struct {
    stream: net.Stream,
    io: Io,
    r: net.Stream.Reader,
    w: net.Stream.Writer,
    rbuf: []u8,
    wbuf: []u8,
    deadline: ?usize = null,

    fn connect(gpa: std.mem.Allocator, io: Io, addr: PeerAddr) !*Peer {
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

    fn close(p: *Peer, gpa: std.mem.Allocator) void {
        sockopt.untrack(p.io, p.deadline); // before close: see sockopt fd-reuse note
        p.stream.close(p.io);
        gpa.free(p.rbuf);
        gpa.free(p.wbuf);
        gpa.destroy(p);
    }

    fn send(p: *Peer, comptime fmt: []const u8, args: anytype) !void {
        try p.w.interface.print(fmt, args);
        try p.w.interface.flush();
    }

    fn recvLine(p: *Peer) ![]u8 {
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

/// Fetch the store's wanted-but-missing ranges from one peer. Guards against
/// cross-version mixing: a peer advertising a different manifest version (e.g.
/// across a future hardfork) is rejected wholesale. Every range is verified
/// against its digest before it touches disk. Used by both bootstrap and the
/// eager repair loop.
pub fn fetchFromPeer(gpa: std.mem.Allocator, io: Io, store: *weights.Store, addr: PeerAddr) !FetchStats {
    var got = FetchStats{};
    return fetchFromPeerInto(gpa, io, store, addr, &got) catch |e| {
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

    const stats = out;
    var i: usize = 0;
    while (i < n_ranges) : (i += 1) {
        if (!store.wanted.has(i) or store.holdings.has(i)) continue;
        if (!peer_holdings.has(i)) continue; // this peer can't serve it; another might

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
        // A shard arrived, so this connection is not idle. `PEER_TIMEOUT_S` is
        // ten seconds and a full sync is thousands of shards over one
        // connection: without this the reaper shut the socket down mid-transfer
        // and the caller reported `synced 0/1571 ranges` as though nothing had
        // gone wrong.
        sockopt.refresh(io, peer.deadline, sockopt.PEER_TIMEOUT_S);
    }
    return stats.*;
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
        var existing = weights.openDir(gpa, io, store_dir) catch break :reopen null;
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

    var stats = FetchStats{};
    for (peers) |addr| {
        if (store.missingCount() == 0) break;
        const s = fetchFromPeer(gpa, io, &store, addr) catch |e| {
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
        if (progress) |pw| {
            try pw.print("  synced {d}/{d} ranges ({d:.1} MB) after {s}:{d}\n", .{
                stats.fetched, wanted, @as(f64, @floatFromInt(stats.bytes)) / (1024.0 * 1024.0), addr.host, addr.port,
            });
            try pw.flush();
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
    return bootstrapWithWanted(gpa, io, peers, store_dir, m, wanted_bits, progress);
}
