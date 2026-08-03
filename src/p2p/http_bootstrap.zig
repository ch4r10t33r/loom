//! HTTP-range bootstrap tier: initial sync of wanted shards from a *static
//! mirror* of the model file, instead of (before) the p2p mesh.
//!
//! The manifest maps every shard to byte extents of the original GGUF, so any
//! plain file host that supports Range requests — Hugging Face, R2/S3, an
//! nginx serving one file — can serve every shard with zero loom code on the
//! mirror. Integrity is transport-independent: each shard is verified against
//! the Merkle-rooted manifest digest by `Store.writeRange` exactly as a peer
//! fetch is, so the mirror is untrusted by construction. This is also why
//! plain http URLs are accepted (with a printed notice): a tampered byte
//! fails its digest; the weights themselves are public content.
//!
//! Why it exists: joins are bottlenecked by the single warm origin's uplink
//! (measured 11 MB/s to one joiner on the devnet). A CDN-backed mirror
//! removes the origin from the bulk path entirely, and parallel range
//! streams fill fat downlinks that one TCP connection cannot. The p2p plane
//! remains authoritative for the manifest, the long tail, repair, and heat —
//! a stale mirror (wrong model version) just fails digests loudly and the
//! p2p pass mops up.

const std = @import("std");
const Io = std.Io;
const weights = @import("weights.zig");
const stats_mod = @import("../core/stats.zig");

const MAX_REDIRECTS = 8;
/// Concurrent range workers. Enough to fill a fat downlink past single-TCP
/// WAN limits; few enough to be polite to public mirrors.
pub const WORKERS = 6;

pub const Result = struct { fetched: u64 = 0, bytes: u64 = 0, failed: u64 = 0 };

const Shared = struct {
    store: *weights.Store,
    url: []const u8,
    /// 0 = the mirror hosts one file at `url`. Non-zero = the mirror hosts
    /// fixed-size byte parts named `<url>.part-00000`, `.part-00001`, ... of
    /// exactly this many bytes each (except the last); parts concatenate to
    /// the byte-identical model file. Exists because some hosts cap single
    /// files below model size (Hugging Face: 50 GB).
    part_bytes: u64 = 0,
    ids: []const usize,
    next: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    fetched: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    failed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    io: Io,
};

/// Fetch every missing wanted shard from the mirror. Failures are counted,
/// not fatal — the caller's p2p pass fetches whatever is left. Returns once
/// every shard has been attempted.
pub fn fetchMissing(
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    url: []const u8,
    part_bytes: u64,
    progress: ?*Io.Writer,
) !Result {
    // Work list: wanted but not held, in manifest order (resident chunks
    // first — the shards a node must have before it can serve at all).
    var ids = std.ArrayList(usize).empty;
    defer ids.deinit(gpa);
    const n = store.manifest.nRanges();
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (store.wanted.has(i) and !store.holdings.has(i)) try ids.append(gpa, i);
    }
    if (ids.items.len == 0) return .{};

    if (progress) |pw| {
        if (std.mem.startsWith(u8, url, "http://")) {
            pw.print("  http mirror is plain http; integrity still digest-verified per shard\n", .{}) catch {};
        }
        pw.print("  http sync: {d} shard(s) from {s} ({d} stream(s))\n", .{ ids.items.len, url, WORKERS }) catch {};
        pw.flush() catch {};
    }

    var shared = Shared{ .store = store, .url = url, .part_bytes = part_bytes, .ids = ids.items, .io = io };
    var threads: [WORKERS]?std.Thread = [_]?std.Thread{null} ** WORKERS;
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, worker, .{ gpa, &shared }) catch null;

    // Main thread reports progress while the workers drain the list.
    const t0 = stats_mod.nowNs(io);
    var last_fetched: u64 = 0;
    while (true) {
        Io.sleep(io, .{ .nanoseconds = 2_000_000_000 }, .awake) catch {};
        const done = shared.fetched.load(.monotonic) + shared.failed.load(.monotonic);
        if (progress) |pw| {
            const f = shared.fetched.load(.monotonic);
            if (f != last_fetched) {
                last_fetched = f;
                const secs = @as(f64, @floatFromInt(stats_mod.nowNs(io) - t0)) / 1e9;
                const mb = @as(f64, @floatFromInt(shared.bytes.load(.monotonic))) / (1024.0 * 1024.0);
                pw.print("  http sync {d}/{d} shards  {d:.0} MB  {d:.1} MB/s\n", .{
                    f, ids.items.len, mb, if (secs > 0) mb / secs else 0,
                }) catch {};
                pw.flush() catch {};
            }
        }
        if (done >= ids.items.len) break;
    }
    for (threads) |t| if (t) |th| th.join();

    return .{
        .fetched = shared.fetched.load(.monotonic),
        .bytes = shared.bytes.load(.monotonic),
        .failed = shared.failed.load(.monotonic),
    };
}

fn worker(gpa: std.mem.Allocator, shared: *Shared) void {
    var client = std.http.Client{ .allocator = gpa, .io = shared.io };
    defer client.deinit();
    // One reusable shard buffer per worker, sized for the largest shard.
    const buf = gpa.alloc(u8, @intCast(shared.store.manifest.maxShardLen())) catch return;
    defer gpa.free(buf);

    while (true) {
        const k = shared.next.fetchAdd(1, .monotonic);
        if (k >= shared.ids.len) return;
        const id = shared.ids[k];
        const len: usize = @intCast(shared.store.manifest.rangeLen(id));
        if (fetchShard(gpa, &client, shared, id, buf[0..len])) {
            // Digest-verify + persist + mark held, same as a peer fetch.
            if (shared.store.writeRange(id, buf[0..len])) {
                _ = shared.fetched.fetchAdd(1, .monotonic);
                _ = shared.bytes.fetchAdd(len, .monotonic);
                continue;
            } else |_| {}
        }
        _ = shared.failed.fetchAdd(1, .monotonic);
    }
}

/// Fill `out` with the shard's extents, one Range request each (more when an
/// extent crosses a part boundary on a split mirror). A shard is at most a
/// few extents (gate/up/down), so the per-request overhead is small against
/// multi-MB bodies, and the client's connection pool keeps the TCP and TLS
/// session warm across requests.
fn fetchShard(gpa: std.mem.Allocator, client: *std.http.Client, shared: *Shared, id: usize, out: []u8) bool {
    var off: usize = 0;
    for (shared.store.manifest.shardExtents(id)) |e| {
        const elen: usize = @intCast(e.len);
        fetchExtent(gpa, client, shared, e.offset, out[off..][0..elen]) catch return false;
        off += elen;
    }
    return off == out.len;
}

/// One extent, split across part files when the mirror is chunked. The part
/// name convention is `<url>.part-00000` with fixed-size parts, so the
/// file-offset -> (part, local-offset) map is pure arithmetic.
fn fetchExtent(gpa: std.mem.Allocator, client: *std.http.Client, shared: *Shared, offset: u64, out: []u8) !void {
    if (shared.part_bytes == 0) {
        return fetchRange(gpa, client, shared.url, offset, out.len, out);
    }
    var pos: u64 = offset;
    var done: usize = 0;
    var namebuf: [4096]u8 = undefined;
    while (done < out.len) {
        const part = pos / shared.part_bytes;
        const local = pos % shared.part_bytes;
        const take: usize = @intCast(@min(@as(u64, out.len - done), shared.part_bytes - local));
        const purl = std.fmt.bufPrint(&namebuf, "{s}.part-{d:0>5}", .{ shared.url, part }) catch return error.UrlTooLong;
        try fetchRange(gpa, client, purl, local, take, out[done..][0..take]);
        pos += take;
        done += take;
    }
}

fn fetchRange(gpa: std.mem.Allocator, client: *std.http.Client, url: []const u8, offset: u64, len: usize, out: []u8) !void {
    _ = gpa;
    var aux_storage: [16 * 1024]u8 = undefined;
    var aux: []u8 = &aux_storage;
    var uri = std.Uri.parse(url) catch return error.InvalidUrl;

    var hops: usize = 0;
    while (true) : (hops += 1) {
        if (hops > MAX_REDIRECTS) return error.TooManyRedirects;

        var range_buf: [64]u8 = undefined;
        const range_hdr = [_]std.http.Header{.{
            .name = "Range",
            .value = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, offset + len - 1 }) catch unreachable,
        }};
        var req = try client.request(.GET, uri, .{ .redirect_behavior = .unhandled, .extra_headers = &range_hdr });
        defer req.deinit();
        try req.sendBodiless();
        var redirect_buf: [8192]u8 = undefined;
        var resp = try req.receiveHead(&redirect_buf);

        const status = resp.head.status;
        if (status.class() == .redirect) {
            const loc = resp.head.location orelse return error.BadRedirect;
            if (loc.len > aux.len) return error.RedirectTooLong;
            const copied = aux[0..loc.len];
            @memcpy(copied, loc);
            {
                const body_reader = req.reader.bodyReader(&.{}, resp.head.transfer_encoding, resp.head.content_length);
                _ = body_reader.discardRemaining() catch {};
            }
            uri = uri.resolveInPlace(loc.len, &aux) catch return error.InvalidUrl;
            continue;
        }
        // A mirror that ignores Range answers 200 with the whole file; that
        // would mean re-downloading the model once per shard, so treat it as
        // unusable rather than limping through.
        if (status != .partial_content) return error.RangeNotSupported;

        var tbuf: [1 << 16]u8 = undefined;
        const body = resp.reader(&tbuf);
        var got: usize = 0;
        while (got < len) {
            const r = body.readSliceShort(out[got..]) catch return error.ShortRead;
            if (r == 0) return error.ShortRead;
            got += r;
        }
        return;
    }
}

const gguf_fixture = @import("../gguf/gguf.zig");

/// Minimal single-purpose HTTP/1.1 range server for the test below: parses
/// the Range header, answers 206 with the slice, closes the connection. Runs
/// for exactly `expected` requests, then exits — no shutdown plumbing needed.
const TestMirror = struct {
    io: Io,
    blob: []const u8,
    server: *std.Io.net.Server,
    expected: usize,
    /// Non-zero: pretend to host fixed-size part files; a request path
    /// containing ".part-NNNNN" resolves to that slice of the blob.
    part_bytes: u64 = 0,

    fn run(self: *TestMirror) void {
        var served: usize = 0;
        while (served < self.expected) : (served += 1) {
            const stream = self.server.accept(self.io) catch return;
            defer stream.close(self.io);
            var rbuf: [4096]u8 = undefined;
            var wbuf: [4096]u8 = undefined;
            var r = stream.reader(self.io, &rbuf);
            var w = stream.writer(self.io, &wbuf);
            var start: u64 = 0;
            var end: u64 = 0;
            var base: u64 = 0;
            var limit: u64 = @intCast(self.blob.len);
            while (true) {
                const line = r.interface.takeDelimiterInclusive('\n') catch return;
                const t = std.mem.trimEnd(u8, line, "\r\n");
                if (t.len == 0) break;
                if (std.mem.startsWith(u8, t, "GET ") and self.part_bytes != 0) {
                    const marker = std.mem.indexOf(u8, t, ".part-") orelse return;
                    const num = t[marker + ".part-".len ..][0..5];
                    const part = std.fmt.parseInt(u64, num, 10) catch return;
                    base = part * self.part_bytes;
                    limit = @min(base + self.part_bytes, @as(u64, @intCast(self.blob.len)));
                }
                if (std.mem.startsWith(u8, t, "Range: bytes=")) {
                    const spec = t["Range: bytes=".len..];
                    const dash = std.mem.indexOfScalar(u8, spec, '-') orelse return;
                    start = std.fmt.parseInt(u64, spec[0..dash], 10) catch return;
                    end = std.fmt.parseInt(u64, spec[dash + 1 ..], 10) catch return;
                }
            }
            if (base + end + 1 > limit) return; // range past the part's end
            const body = self.blob[@intCast(base + start)..@intCast(base + end + 1)];
            w.interface.print(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
                .{body.len},
            ) catch return;
            w.interface.writeAll(body) catch return;
            w.interface.flush() catch return;
        }
    }
};

test "fetchMissing fills the whole store from a range mirror, digest-verified" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const model_path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/mirror.gguf", .{tmp.sub_path});
    try gguf_fixture.writeMoeFixture(gpa, io, model_path, 3, "llama");

    const f = try Io.Dir.cwd().openFile(io, model_path, .{});
    const blob = blk: {
        var fr = f.reader(io, &.{});
        const size = try f.length(io);
        const b = try gpa.alloc(u8, @intCast(size));
        errdefer gpa.free(b);
        try fr.interface.readSliceAll(b);
        break :blk b;
    };
    f.close(io);
    defer gpa.free(blob);

    var manifest = try weights.buildExpertManifest(gpa, io, model_path);
    var n_extents: usize = 0;
    for (0..manifest.nRanges()) |i| n_extents += manifest.shardExtents(i).len;

    var address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    const port = server.socket.address.ip4.port;

    var mirror = TestMirror{ .io = io, .blob = blob, .server = &server, .expected = n_extents };
    const th = try std.Thread.spawn(.{}, TestMirror.run, .{&mirror});

    const wanted = try weights.Holdings.initWanted(gpa, manifest.nRanges(), manifest.n_resident, 1.0, 7);
    var sdir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const store_dir = try std.fmt.bufPrint(&sdir_buf, ".zig-cache/tmp/{s}/store", .{tmp.sub_path});
    var store = try weights.createFromManifest(gpa, io, store_dir, manifest, wanted);
    defer store.deinit();

    var ubuf: [64]u8 = undefined;
    const url = try std.fmt.bufPrint(&ubuf, "http://127.0.0.1:{d}/mirror.gguf", .{port});
    const res = try fetchMissing(gpa, io, &store, url, 0, null);
    th.join();

    try std.testing.expectEqual(@as(u64, 0), res.failed);
    try std.testing.expectEqual(@as(usize, 0), store.missingCount());
    try std.testing.expectEqual(manifest.nRanges(), store.holdings.count());
}

test "fetchMissing spans part boundaries on a split mirror" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const model_path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/split.gguf", .{tmp.sub_path});
    try gguf_fixture.writeMoeFixture(gpa, io, model_path, 11, "llama");

    const f = try Io.Dir.cwd().openFile(io, model_path, .{});
    const blob = blk: {
        var fr = f.reader(io, &.{});
        const size = try f.length(io);
        const b = try gpa.alloc(u8, @intCast(size));
        errdefer gpa.free(b);
        try fr.interface.readSliceAll(b);
        break :blk b;
    };
    f.close(io);
    defer gpa.free(blob);

    var manifest = try weights.buildExpertManifest(gpa, io, model_path);
    // A part size smaller than most extents, so nearly every extent crosses
    // a boundary and the offset arithmetic is exercised hard. Request count
    // is per (extent x parts touched); compute it exactly so the mirror can
    // exit deterministically.
    const part_bytes: u64 = 4096;
    var n_reqs: usize = 0;
    for (0..manifest.nRanges()) |i| {
        for (manifest.shardExtents(i)) |e| {
            const first = e.offset / part_bytes;
            const last = (e.offset + e.len - 1) / part_bytes;
            n_reqs += @intCast(last - first + 1);
        }
    }

    var address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    const port = server.socket.address.ip4.port;

    var mirror = TestMirror{ .io = io, .blob = blob, .server = &server, .expected = n_reqs, .part_bytes = part_bytes };
    const th = try std.Thread.spawn(.{}, TestMirror.run, .{&mirror});

    const wanted = try weights.Holdings.initWanted(gpa, manifest.nRanges(), manifest.n_resident, 1.0, 3);
    var sdir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const store_dir = try std.fmt.bufPrint(&sdir_buf, ".zig-cache/tmp/{s}/splitstore", .{tmp.sub_path});
    var store = try weights.createFromManifest(gpa, io, store_dir, manifest, wanted);
    defer store.deinit();

    var ubuf: [64]u8 = undefined;
    const url = try std.fmt.bufPrint(&ubuf, "http://127.0.0.1:{d}/split.gguf", .{port});
    const res = try fetchMissing(gpa, io, &store, url, part_bytes, null);
    th.join();

    try std.testing.expectEqual(@as(u64, 0), res.failed);
    try std.testing.expectEqual(@as(usize, 0), store.missingCount());
}
