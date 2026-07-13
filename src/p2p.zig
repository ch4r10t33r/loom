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
//!   MANIFEST      -> MANIFEST version=<hex> size=<b> ranges=<n> range_size=<b>
//!   DIGEST <i>    -> DIGEST <i> <hex>
//!   DIGESTS       -> DIGESTS <n>\n<hex>\n x n     (bulk, for bootstrap)
//!   HOLDINGS      -> HOLDINGS <hex bitmap>      (bit i = holds range i; the
//!                    same compact summary destined for ENR metadata + gossip)
//!   GETR <i>      -> DATA <i> len=<l> sha256=<hex>\n<raw bytes> | ERR not_held
//!   <other>       -> ERR unknown

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const checkpoint = @import("checkpoint.zig");
const hashmod = @import("hash.zig");
const weights = @import("weights.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    entries: []const checkpoint.ExpertEntry,
    unique_bytes: u64,
    addr: []const u8,
    port: u16,
    /// Attached GGUF weight store, if this node participates in distribution.
    /// Reads are lock-free (immutable manifest, pread on a shared handle).
    store: ?*weights.Store = null,
};

const Conn = struct { ctx: *Ctx, stream: net.Stream };

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.addr, ctx.port);
    var server = try address.listen(ctx.io, .{ .reuse_address = true });
    defer server.deinit(ctx.io);

    while (true) {
        const stream = server.accept(ctx.io) catch continue;
        const conn = ctx.gpa.create(Conn) catch {
            stream.close(ctx.io);
            continue;
        };
        conn.* = .{ .ctx = ctx, .stream = stream };
        const t = std.Thread.spawn(.{}, connThread, .{conn}) catch {
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
}

fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    var rbuf: [4096]u8 = undefined;
    var wbuf: [4096]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    while (true) {
        const raw = ri.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        try handleLine(ctx, line, wi);
        try wi.flush();
    }
}

fn handleLine(ctx: *Ctx, line: []const u8, wi: *Io.Writer) !void {
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
    } else if (std.mem.eql(u8, line, "MANIFEST")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const m = &store.manifest;
        try wi.print("MANIFEST version={s} size={d} ranges={d} range_size={d}\n", .{
            hashmod.toHex(m.version), m.file_size, m.nRanges(), m.range_size,
        });
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
        const data = store.readRange(i, buf) catch return wi.print("ERR read\n", .{});
        try wi.print("DATA {d} len={d} sha256={s}\n", .{ i, data.len, hashmod.toHex(store.manifest.digests[i]) });
        try wi.writeAll(data);
    } else {
        try wi.print("ERR unknown\n", .{});
    }
}
