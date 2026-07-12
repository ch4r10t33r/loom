//! Minimal P2P peer server — the seed of the v1 distributed expert directory.
//!
//! A real deployment runs this over Hyperswarm/HyperDHT; here it is a small
//! line protocol over TCP that answers "who has expert N" against this node's
//! immutable, content-addressed index. It exposes exactly the query v1's fetch
//! policy needs (expert_id -> holder + offset/len/digest) without yet doing
//! replication or heat tracking.
//!
//! Protocol (line-delimited):
//!   HELLO         -> LOOM/0 experts=<n> unique_bytes=<b>
//!   COUNT         -> <n>
//!   HAS <id>      -> PRESENT <id> off=<o> len=<l> sha256=<hex> | ABSENT <id> | ERR range
//!   PING          -> PONG
//!   <other>       -> ERR unknown

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const checkpoint = @import("checkpoint.zig");
const hashmod = @import("hash.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    entries: []const checkpoint.ExpertEntry,
    unique_bytes: u64,
    addr: []const u8,
    port: u16,
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
    } else {
        try wi.print("ERR unknown\n", .{});
    }
}
