//! Wire messages v1 (SPEC.md): binary frames for heartbeat, gossip announce,
//! and the expert request/response, with adaptive snappy compression — the
//! encoder keeps the compressed body only when it is actually smaller
//! (holdings bitmaps compress well; quantized expert payloads usually don't).

const std = @import("std");
const snappy = @import("snappyz");

pub const MAGIC = [2]u8{ 'L', 'M' };
pub const PROTO: u8 = 2; // v2: network_id joins Announce/Heartbeat (chainId semantics)

/// The network identity a node will only peer within -- Ethereum's chainId,
/// for LLM networks: one loom network serves one model. Derived from the
/// manifest by default (`--network-id auto`), or set explicitly so several
/// quantizations of one model can share a network across hardforks.
pub fn networkIdFromManifest(version: [32]u8) u64 {
    return std.mem.readInt(u64, version[0..8], .little);
}
pub const HEADER_LEN: usize = 8;
pub const FLAG_SNAPPY: u8 = 0x01;

/// Hard cap on a decompressed frame body (audit #7 P0 snappy bomb). The
/// largest legitimate body is a holdings bitmap or a small shard payload;
/// 64 MiB is far above any real message and bounds a malicious peer's
/// declared-uncompressed size.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

pub const MsgType = enum(u8) {
    heartbeat = 0x01,
    heartbeat_resp = 0x02,
    announce = 0x03,
    announce_batch = 0x04,
    expert_request = 0x10,
    expert_response = 0x11,
    _,
};

pub const Status = enum(u8) {
    ok = 0,
    not_held = 1,
    version_mismatch = 2,
    busy = 3,
    bad_request = 4,
    _,
};

pub const NO_COMMITTEE: u32 = 0xFFFF_FFFF;

// ---- envelope ----------------------------------------------------------------

/// Wrap `body` in a frame, compressing when it pays. Caller owns the result.
pub fn encodeFrame(gpa: std.mem.Allocator, ty: MsgType, body: []const u8) ![]u8 {
    var flags: u8 = 0;
    var payload: []const u8 = body;
    const compressed = snappy.encode(gpa, body) catch null;
    // the frame copies whichever buffer wins, so c is always freed on exit
    defer if (compressed) |c| gpa.free(c);
    if (compressed) |c| {
        if (c.len < body.len) {
            payload = c;
            flags |= FLAG_SNAPPY;
        }
    }
    // The length field is u32; an oversized payload must be an error, not an
    // @intCast panic (security issue #28: a poisoned peer table could grow an
    // AnnounceBatch past 4 GiB and abort the process here).
    if (payload.len > std.math.maxInt(u32)) return error.FrameTooLarge;
    const out = try gpa.alloc(u8, HEADER_LEN + payload.len);
    out[0] = MAGIC[0];
    out[1] = MAGIC[1];
    out[2] = @intFromEnum(ty);
    out[3] = flags;
    std.mem.writeInt(u32, out[4..8], @intCast(payload.len), .little);
    @memcpy(out[HEADER_LEN..], payload);
    return out;
}

pub const Decoded = struct {
    ty: MsgType,
    body: []u8, // owned by caller (decompressed or duplicated)
};

/// Parse a full frame (header + body). Returns an owned, decompressed body.
pub fn decodeFrame(gpa: std.mem.Allocator, frame: []const u8) !Decoded {
    if (frame.len < HEADER_LEN) return error.ShortFrame;
    if (frame[0] != MAGIC[0] or frame[1] != MAGIC[1]) return error.BadMagic;
    const ty: MsgType = @enumFromInt(frame[2]);
    const flags = frame[3];
    const body_len = std.mem.readInt(u32, frame[4..8], .little);
    if (frame.len != HEADER_LEN + body_len) return error.LengthMismatch;
    const raw = frame[HEADER_LEN..];
    if (body_len > MAX_BODY_BYTES) return error.FrameTooLarge;
    const body = if (flags & FLAG_SNAPPY != 0)
        try snappy.decodeWithMax(gpa, raw, MAX_BODY_BYTES) // caps declared size
    else
        try gpa.dupe(u8, raw);
    return .{ .ty = ty, .body = body };
}

// ---- body helpers ------------------------------------------------------------

const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    fn need(self: *Cursor, n: usize) ![]const u8 {
        if (self.pos + n > self.buf.len) return error.Truncated;
        const s = self.buf[self.pos..][0..n];
        self.pos += n;
        return s;
    }
    fn u8v(self: *Cursor) !u8 {
        return (try self.need(1))[0];
    }
    fn u16v(self: *Cursor) !u16 {
        return std.mem.readInt(u16, (try self.need(2))[0..2], .little);
    }
    fn u32v(self: *Cursor) !u32 {
        return std.mem.readInt(u32, (try self.need(4))[0..4], .little);
    }
    fn u64v(self: *Cursor) !u64 {
        return std.mem.readInt(u64, (try self.need(8))[0..8], .little);
    }
    fn i64v(self: *Cursor) !i64 {
        return @bitCast(try self.u64v());
    }
};

fn appendU16(gpa: std.mem.Allocator, b: *std.ArrayList(u8), v: u16) !void {
    var t: [2]u8 = undefined;
    std.mem.writeInt(u16, &t, v, .little);
    try b.appendSlice(gpa, &t);
}
fn appendU32(gpa: std.mem.Allocator, b: *std.ArrayList(u8), v: u32) !void {
    var t: [4]u8 = undefined;
    std.mem.writeInt(u32, &t, v, .little);
    try b.appendSlice(gpa, &t);
}
fn appendU64(gpa: std.mem.Allocator, b: *std.ArrayList(u8), v: u64) !void {
    var t: [8]u8 = undefined;
    std.mem.writeInt(u64, &t, v, .little);
    try b.appendSlice(gpa, &t);
}

// ---- Heartbeat (0x01 / 0x02) ---------------------------------------------------

pub const Heartbeat = struct {
    proto: u8 = PROTO,
    network_id: u64 = 0,
    committee_id: u32 = NO_COMMITTEE,
    manifest_version: [32]u8 = [_]u8{0} ** 32,
    holdings_seq: u64 = 0,
    holdings_digest: [32]u8 = [_]u8{0} ** 32,
    load: u16 = 0,
    sent_at_ns: i64 = 0,
    addr: []const u8 = "", // sender's advertised host:port (<=255 bytes)

    pub fn encodeBody(self: *const Heartbeat, gpa: std.mem.Allocator) ![]u8 {
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(gpa);
        try b.append(gpa, self.proto);
        try appendU64(gpa, &b, self.network_id);
        try appendU32(gpa, &b, self.committee_id);
        try b.appendSlice(gpa, &self.manifest_version);
        try appendU64(gpa, &b, self.holdings_seq);
        try b.appendSlice(gpa, &self.holdings_digest);
        try appendU16(gpa, &b, self.load);
        try appendU64(gpa, &b, @bitCast(self.sent_at_ns));
        try b.append(gpa, @intCast(@min(self.addr.len, 255)));
        try b.appendSlice(gpa, self.addr[0..@min(self.addr.len, 255)]);
        return b.toOwnedSlice(gpa);
    }

    /// `body` must outlive the result (addr aliases it).
    pub fn parseBody(body: []const u8) !Heartbeat {
        var c = Cursor{ .buf = body };
        var hb = Heartbeat{};
        hb.proto = try c.u8v();
        hb.network_id = try c.u64v();
        hb.committee_id = try c.u32v();
        @memcpy(&hb.manifest_version, try c.need(32));
        hb.holdings_seq = try c.u64v();
        @memcpy(&hb.holdings_digest, try c.need(32));
        hb.load = try c.u16v();
        hb.sent_at_ns = try c.i64v();
        const alen = try c.u8v();
        hb.addr = try c.need(alen);
        return hb;
    }
};

// ---- Announce (0x03) -----------------------------------------------------------

pub const Announce = struct {
    proto: u8 = PROTO,
    network_id: u64 = 0,
    committee_id: u32 = NO_COMMITTEE,
    manifest_version: [32]u8 = [_]u8{0} ** 32,
    holdings_seq: u64 = 0,
    addr: []const u8 = "",
    holdings_bitmap: []const u8 = &.{},

    pub fn encodeBody(self: *const Announce, gpa: std.mem.Allocator) ![]u8 {
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(gpa);
        try b.append(gpa, self.proto);
        try appendU64(gpa, &b, self.network_id);
        try appendU32(gpa, &b, self.committee_id);
        try b.appendSlice(gpa, &self.manifest_version);
        try appendU64(gpa, &b, self.holdings_seq);
        try b.append(gpa, @intCast(@min(self.addr.len, 255)));
        try b.appendSlice(gpa, self.addr[0..@min(self.addr.len, 255)]);
        try appendU32(gpa, &b, @intCast(self.holdings_bitmap.len));
        try b.appendSlice(gpa, self.holdings_bitmap);
        return b.toOwnedSlice(gpa);
    }

    /// `body` must outlive the result (addr/bitmap alias it).
    pub fn parseBody(body: []const u8) !Announce {
        var c = Cursor{ .buf = body };
        var a = Announce{};
        a.proto = try c.u8v();
        a.network_id = try c.u64v();
        a.committee_id = try c.u32v();
        @memcpy(&a.manifest_version, try c.need(32));
        a.holdings_seq = try c.u64v();
        const alen = try c.u8v();
        a.addr = try c.need(alen);
        const blen = try c.u32v();
        a.holdings_bitmap = try c.need(blen);
        return a;
    }
};

// ---- AnnounceBatch (0x04) ------------------------------------------------------
// The gossip exchange response: a batch of Announce bodies (u32 count, then
// count x [u32 len + body]). Snappy at the frame level compresses across the
// similar bitmaps of many peers.

pub const AnnounceBatch = struct {
    pub fn encodeBodies(gpa: std.mem.Allocator, bodies: []const []const u8) ![]u8 {
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(gpa);
        try appendU32(gpa, &b, @intCast(bodies.len));
        for (bodies) |body| {
            try appendU32(gpa, &b, @intCast(body.len));
            try b.appendSlice(gpa, body);
        }
        return b.toOwnedSlice(gpa);
    }

    pub const Iter = struct {
        c: Cursor,
        remaining: u32,

        pub fn next(self: *Iter) !?[]const u8 {
            if (self.remaining == 0) return null;
            self.remaining -= 1;
            const len = try self.c.u32v();
            return try self.c.need(len);
        }
    };

    pub fn iterate(body: []const u8) !Iter {
        var c = Cursor{ .buf = body };
        const n = try c.u32v();
        if (n > 65536) return error.Truncated;
        return .{ .c = c, .remaining = n };
    }
};

// ---- ExpertRequest (0x10) / ExpertResponse (0x11) ------------------------------

pub const ExpertRequest = struct {
    proto: u8 = PROTO,
    request_id: u64,
    manifest_version: [32]u8,
    shard_id: u32,

    pub fn encodeBody(self: *const ExpertRequest, gpa: std.mem.Allocator) ![]u8 {
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(gpa);
        try b.append(gpa, self.proto);
        try appendU64(gpa, &b, self.request_id);
        try b.appendSlice(gpa, &self.manifest_version);
        try appendU32(gpa, &b, self.shard_id);
        return b.toOwnedSlice(gpa);
    }

    pub fn parseBody(body: []const u8) !ExpertRequest {
        var c = Cursor{ .buf = body };
        var r = ExpertRequest{ .request_id = 0, .manifest_version = undefined, .shard_id = 0 };
        r.proto = try c.u8v();
        r.request_id = try c.u64v();
        @memcpy(&r.manifest_version, try c.need(32));
        r.shard_id = try c.u32v();
        return r;
    }
};

pub const ExpertResponse = struct {
    request_id: u64,
    status: Status,
    shard_id: u32,
    payload: []const u8 = &.{},

    pub fn encodeBody(self: *const ExpertResponse, gpa: std.mem.Allocator) ![]u8 {
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(gpa);
        try appendU64(gpa, &b, self.request_id);
        try b.append(gpa, @intFromEnum(self.status));
        try appendU32(gpa, &b, self.shard_id);
        try appendU32(gpa, &b, @intCast(self.payload.len));
        try b.appendSlice(gpa, self.payload);
        return b.toOwnedSlice(gpa);
    }

    /// `body` must outlive the result (payload aliases it).
    pub fn parseBody(body: []const u8) !ExpertResponse {
        var c = Cursor{ .buf = body };
        var r = ExpertResponse{ .request_id = 0, .status = .bad_request, .shard_id = 0 };
        r.request_id = try c.u64v();
        r.status = @enumFromInt(try c.u8v());
        r.shard_id = try c.u32v();
        const plen = try c.u32v();
        r.payload = try c.need(plen);
        return r;
    }
};

// ---- stream transport ("FRAME <len>\n" + bytes over the line protocol) --------

pub const Io = std.Io;

pub fn writeFrame(wi: *Io.Writer, frame: []const u8) !void {
    try wi.print("FRAME {d}\n", .{frame.len});
    try wi.writeAll(frame);
    try wi.flush();
}

/// Reads a `FRAME <len>` header line + body. Caller owns the result.
/// Read `len` bytes into a fresh buffer, growing it as the bytes actually
/// arrive (security issue #27). Allocating `len` up front let a peer reserve
/// the full declared size by sending only the `FRAME <len>` header and then
/// nothing: 256 connections x 64 MiB = 16 GiB for a few hundred bytes of
/// traffic. Chunked reads make resident memory track delivered bytes, and the
/// connection deadline (core/sockopt.zig) bounds how long a stalled transfer
/// can hold even that.
fn readBodyChunked(gpa: std.mem.Allocator, ri: *Io.Reader, len: usize) ![]u8 {
    const CHUNK = 64 * 1024;
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(gpa);
    try buf.ensureTotalCapacity(gpa, @min(len, CHUNK));
    var remaining = len;
    while (remaining > 0) {
        const n = @min(remaining, CHUNK);
        const dst = try buf.addManyAsSlice(gpa, n);
        try ri.readSliceAll(dst);
        remaining -= n;
    }
    return buf.toOwnedSlice(gpa);
}

pub fn readFrameAlloc(gpa: std.mem.Allocator, ri: *Io.Reader) ![]u8 {
    const line = std.mem.trimEnd(u8, try ri.takeDelimiterInclusive('\n'), "\r\n");
    if (!std.mem.startsWith(u8, line, "FRAME ")) return error.NotAFrame;
    const len = try std.fmt.parseInt(usize, line[6..], 10);
    // Same ceiling the server enforces (security issue #27): this client-side
    // reader used to allow 512 MiB, 8x the server cap, and it runs in up to 64
    // concurrent prefetch threads.
    if (len < HEADER_LEN or len > MAX_BODY_BYTES + HEADER_LEN) return error.BadFrameLength;
    return readBodyChunked(gpa, ri, len);
}

/// Server-side counterpart: validate the declared length and read it in
/// bounded chunks. Callers own the returned slice.
pub fn readFrameBodyAlloc(gpa: std.mem.Allocator, ri: *Io.Reader, len: usize) ![]u8 {
    if (len < HEADER_LEN or len > MAX_BODY_BYTES + HEADER_LEN) return error.BadFrameLength;
    return readBodyChunked(gpa, ri, len);
}

// ---- tests --------------------------------------------------------------------

test "heartbeat frame roundtrip" {
    const gpa = std.testing.allocator;
    const hb = Heartbeat{
        .committee_id = 3,
        .manifest_version = [_]u8{0xAB} ** 32,
        .holdings_seq = 42,
        .holdings_digest = [_]u8{0xCD} ** 32,
        .load = 7,
        .sent_at_ns = 123456789,
        .addr = "127.0.0.1:8771",
    };
    const body = try hb.encodeBody(gpa);
    defer gpa.free(body);
    const frame = try encodeFrame(gpa, .heartbeat, body);
    defer gpa.free(frame);

    const dec = try decodeFrame(gpa, frame);
    defer gpa.free(dec.body);
    try std.testing.expect(dec.ty == .heartbeat);
    const back = try Heartbeat.parseBody(dec.body);
    try std.testing.expect(back.committee_id == 3);
    try std.testing.expect(back.holdings_seq == 42);
    try std.testing.expect(back.load == 7);
    try std.testing.expectEqualStrings("127.0.0.1:8771", back.addr);
}

test "announce with compressible bitmap gets snappy flag" {
    const gpa = std.testing.allocator;
    var bitmap = [_]u8{0} ** 2400; // GLM-scale sparse bitmap: compresses hard
    bitmap[3] = 0xFF;
    const ann = Announce{
        .committee_id = 0,
        .holdings_seq = 9,
        .addr = "10.0.0.2:8771",
        .holdings_bitmap = &bitmap,
    };
    const body = try ann.encodeBody(gpa);
    defer gpa.free(body);
    const frame = try encodeFrame(gpa, .announce, body);
    defer gpa.free(frame);
    try std.testing.expect(frame[3] & FLAG_SNAPPY != 0); // compressed
    try std.testing.expect(frame.len < body.len / 4); // and much smaller

    const dec = try decodeFrame(gpa, frame);
    defer gpa.free(dec.body);
    const back = try Announce.parseBody(dec.body);
    try std.testing.expect(back.holdings_bitmap.len == 2400);
    try std.testing.expect(back.holdings_bitmap[3] == 0xFF);
    try std.testing.expect(back.holdings_seq == 9);
}

test "expert request/response roundtrip; incompressible payload stays raw" {
    const gpa = std.testing.allocator;
    const req = ExpertRequest{ .request_id = 77, .manifest_version = [_]u8{1} ** 32, .shard_id = 1729 };
    const rbody = try req.encodeBody(gpa);
    defer gpa.free(rbody);
    const rframe = try encodeFrame(gpa, .expert_request, rbody);
    defer gpa.free(rframe);
    const rdec = try decodeFrame(gpa, rframe);
    defer gpa.free(rdec.body);
    const rback = try ExpertRequest.parseBody(rdec.body);
    try std.testing.expect(rback.request_id == 77 and rback.shard_id == 1729);

    // pseudo-random (quantized-weight-like) payload: snappy should NOT win
    var prng = std.Random.DefaultPrng.init(5);
    var payload: [4096]u8 = undefined;
    prng.random().bytes(&payload);
    const resp = ExpertResponse{ .request_id = 77, .status = .ok, .shard_id = 1729, .payload = &payload };
    const body = try resp.encodeBody(gpa);
    defer gpa.free(body);
    const frame = try encodeFrame(gpa, .expert_response, body);
    defer gpa.free(frame);
    try std.testing.expect(frame[3] & FLAG_SNAPPY == 0); // raw: compression didn't pay

    const dec = try decodeFrame(gpa, frame);
    defer gpa.free(dec.body);
    const back = try ExpertResponse.parseBody(dec.body);
    try std.testing.expect(back.status == .ok);
    try std.testing.expectEqualSlices(u8, &payload, back.payload);
}

test "announce batch roundtrip" {
    const gpa = std.testing.allocator;
    const a1 = Announce{ .committee_id = 0, .holdings_seq = 1, .addr = "a:1", .holdings_bitmap = &.{ 0xFF, 0x01 } };
    const a2 = Announce{ .committee_id = 1, .holdings_seq = 2, .addr = "b:2", .holdings_bitmap = &.{0x0F} };
    const b1 = try a1.encodeBody(gpa);
    defer gpa.free(b1);
    const b2 = try a2.encodeBody(gpa);
    defer gpa.free(b2);
    const batch = try AnnounceBatch.encodeBodies(gpa, &.{ b1, b2 });
    defer gpa.free(batch);

    var it = try AnnounceBatch.iterate(batch);
    const e1 = try Announce.parseBody((try it.next()).?);
    try std.testing.expectEqualStrings("a:1", e1.addr);
    try std.testing.expect(e1.committee_id == 0);
    const e2 = try Announce.parseBody((try it.next()).?);
    try std.testing.expect(e2.holdings_seq == 2);
    try std.testing.expect((try it.next()) == null);
}

test "corrupt frames are rejected" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.ShortFrame, decodeFrame(gpa, "LM"));
    var bad = [_]u8{ 'X', 'Y', 1, 0, 0, 0, 0, 0 };
    try std.testing.expectError(error.BadMagic, decodeFrame(gpa, &bad));
    var mism = [_]u8{ 'L', 'M', 1, 0, 9, 0, 0, 0 }; // claims 9 body bytes, has 0
    try std.testing.expectError(error.LengthMismatch, decodeFrame(gpa, &mism));
}

test "readFrameAlloc rejects a length above the server cap (issue #27)" {
    const gpa = std.testing.allocator;
    // 512 MiB used to be accepted here while the server capped at 64 MiB
    var buf: [64]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "FRAME {d}\n", .{MAX_BODY_BYTES + HEADER_LEN + 1});
    var r: Io.Reader = .fixed(line);
    try std.testing.expectError(error.BadFrameLength, readFrameAlloc(gpa, &r));
}

test "readFrameBodyAlloc reads only the bytes delivered (issue #27)" {
    const gpa = std.testing.allocator;
    const body = "hello frame body";
    var r: Io.Reader = .fixed(body);
    const got = try readFrameBodyAlloc(gpa, &r, body.len);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(body, got);
}

test "network_id survives the announce and heartbeat roundtrip" {
    const gpa = std.testing.allocator;
    var ann = Announce{ .network_id = 0xDEADBEEF12345678, .addr = "1.2.3.4:9" };
    const ab = try ann.encodeBody(gpa);
    defer gpa.free(ab);
    const back = try Announce.parseBody(ab);
    try std.testing.expectEqual(ann.network_id, back.network_id);

    var hb = Heartbeat{ .network_id = 42, .addr = "5.6.7.8:1" };
    const hbb = try hb.encodeBody(gpa);
    defer gpa.free(hbb);
    const hback = try Heartbeat.parseBody(hbb);
    try std.testing.expectEqual(@as(u64, 42), hback.network_id);
}

test "networkIdFromManifest is the manifest's leading eight bytes" {
    var v: [32]u8 = [_]u8{0} ** 32;
    v[0] = 0x11;
    v[7] = 0x22;
    try std.testing.expectEqual(std.mem.readInt(u64, v[0..8], .little), networkIdFromManifest(v));
}
