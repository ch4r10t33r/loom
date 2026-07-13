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
const hashmod = @import("hash.zig");
const weights = @import("weights.zig");

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

    fn connect(gpa: std.mem.Allocator, io: Io, addr: PeerAddr) !*Peer {
        const address = try net.IpAddress.parse(addr.host, addr.port);
        const stream = try address.connect(io, .{ .mode = .stream });
        const p = try gpa.create(Peer);
        errdefer gpa.destroy(p);
        p.io = io;
        p.stream = stream;
        p.rbuf = try gpa.alloc(u8, 1 << 16);
        p.wbuf = try gpa.alloc(u8, 4096);
        p.r = stream.reader(io, p.rbuf);
        p.w = stream.writer(io, p.wbuf);
        return p;
    }

    fn close(p: *Peer, gpa: std.mem.Allocator) void {
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

/// Bootstrap a local weight store from `peer_addr`, holding a random
/// `fraction` of ranges (seeded so a restarted node re-picks the same set).
pub fn bootstrap(
    gpa: std.mem.Allocator,
    io: Io,
    peer_addr: PeerAddr,
    store_dir: []const u8,
    fraction: f32,
    seed: u64,
    progress: ?*Io.Writer,
) !Result {
    const peer = try Peer.connect(gpa, io, peer_addr);
    defer peer.close(gpa);

    // manifest
    try peer.send("MANIFEST\n", .{});
    const mline = try peer.recvLine();
    if (!std.mem.startsWith(u8, mline, "MANIFEST ")) return error.PeerHasNoStore;
    const version = try parseDigestHex(field(mline, "version") orelse return error.BadManifestLine);
    const file_size = try std.fmt.parseInt(u64, field(mline, "size") orelse return error.BadManifestLine, 10);
    const n_ranges = try std.fmt.parseInt(usize, field(mline, "ranges") orelse return error.BadManifestLine, 10);
    const range_size = try std.fmt.parseInt(u64, field(mline, "range_size") orelse return error.BadManifestLine, 10);
    if (n_ranges == 0 or n_ranges != (file_size + range_size - 1) / range_size) return error.BadManifestLine;

    // digests (bulk), then verify the Merkle root pins the advertised version
    try peer.send("DIGESTS\n", .{});
    const dline = try peer.recvLine();
    if (!std.mem.startsWith(u8, dline, "DIGESTS ")) return error.BadDigestsLine;
    const dn = try std.fmt.parseInt(usize, dline[8..], 10);
    if (dn != n_ranges) return error.BadDigestsLine;
    const digests = try gpa.alloc(hashmod.Digest, n_ranges);
    var digests_owned = true; // ownership moves into the store below
    errdefer if (digests_owned) gpa.free(digests);
    for (digests) |*d| d.* = try parseDigestHex(try peer.recvLine());
    const root = try hashmod.merkleRoot(gpa, digests);
    if (!hashmod.eql(root, version)) return error.PeerManifestRootMismatch;

    // peer's holdings
    try peer.send("HOLDINGS\n", .{});
    const hline = try peer.recvLine();
    if (!std.mem.startsWith(u8, hline, "HOLDINGS ")) return error.BadHoldingsLine;
    var peer_holdings = try weights.Holdings.fromHex(gpa, hline[9..], n_ranges);
    defer peer_holdings.deinit(gpa);

    // our target subset
    var want = try weights.Holdings.initRandom(gpa, n_ranges, fraction, seed);
    defer want.deinit(gpa);
    const wanted = want.count();

    var store = try weights.createFromManifest(gpa, io, store_dir, .{
        .version = version,
        .file_size = file_size,
        .range_size = range_size,
        .digests = digests,
    });
    digests_owned = false;
    errdefer store.deinit();

    const buf = try gpa.alloc(u8, @intCast(range_size));
    defer gpa.free(buf);

    var fetched: usize = 0;
    var bytes: u64 = 0;
    var i: usize = 0;
    while (i < n_ranges) : (i += 1) {
        if (!want.has(i)) continue;
        if (!peer_holdings.has(i)) continue; // peer can't serve it; a later peer might

        try peer.send("GETR {d}\n", .{i});
        const rline = try peer.recvLine();
        if (!std.mem.startsWith(u8, rline, "DATA ")) return error.RangeFetchFailed;
        const len = try std.fmt.parseInt(usize, field(rline, "len") orelse return error.BadDataLine, 10);
        if (len > buf.len) return error.BadDataLine;
        try peer.r.interface.readSliceAll(buf[0..len]);
        // writeRange verifies the digest before anything touches disk
        try store.writeRange(i, buf[0..len]);
        fetched += 1;
        bytes += len;
        if (progress) |pw| {
            if (fetched % 16 == 0) {
                try pw.print("  synced {d}/{d} ranges ({d:.1} MB)\r", .{ fetched, wanted, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0) });
                try pw.flush();
            }
        }
    }
    try store.saveSidecars();

    return .{ .store = store, .wanted = wanted, .fetched = fetched, .bytes = bytes };
}
