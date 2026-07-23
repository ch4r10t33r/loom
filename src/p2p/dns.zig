//! Minimal DNS resolver for peer hostnames (issue #21).
//!
//! Zig 0.16 std has no DNS resolver: `std.Io.net.IpAddress.resolve` only parses
//! IP literals, and there is no classic `std.net.getAddressList`. Peers addressed
//! by name (Docker Compose service names, Kubernetes Service DNS) therefore need
//! this. `resolve()` tries, in order: an IP literal, an `/etc/hosts` entry, then
//! a UDP A-query to the first nameserver in `/etc/resolv.conf` (2 s timeout so a
//! broken resolver cannot hang the p2p connect path). IPv4 only; anything else
//! returns a clear error and the caller falls through to the next peer.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;

const QUERY_TIMEOUT = Io.Timeout{ .duration = .{ .raw = .{ .nanoseconds = 2 * std.time.ns_per_s }, .clock = .awake } };

pub fn resolve(io: Io, host: []const u8, port: u16) !net.IpAddress {
    // 1. IP literal (no lookup).
    if (net.IpAddress.parse(host, port)) |ip| return ip else |_| {}

    // 2. /etc/hosts.
    var hbuf: [64]u8 = undefined;
    if (lookupHosts(io, host, &hbuf)) |ipstr| {
        if (net.IpAddress.parse(ipstr, port)) |ip| return ip else |_| {}
    }

    // 3. DNS A-query.
    const a = try queryDns(io, host);
    var sbuf: [16]u8 = undefined;
    const ipstr = std.fmt.bufPrint(&sbuf, "{d}.{d}.{d}.{d}", .{ a[0], a[1], a[2], a[3] }) catch return error.ResolveFailed;
    return net.IpAddress.parse(ipstr, port) catch error.ResolveFailed;
}

fn readFile(io: Io, path: []const u8, buf: []u8) ![]u8 {
    var f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const n = try f.readPositional(io, &[_][]u8{buf}, 0);
    return buf[0..n];
}

fn stripComment(line: []const u8) []const u8 {
    const h = std.mem.indexOfScalar(u8, line, '#') orelse return line;
    return line[0..h];
}

/// First IP for `host` in /etc/hosts, copied into `out`, or null.
fn lookupHosts(io: Io, host: []const u8, out: []u8) ?[]const u8 {
    var buf: [8192]u8 = undefined;
    const content = readFile(io, "/etc/hosts", &buf) catch return null;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line0| {
        var it = std.mem.tokenizeAny(u8, stripComment(line0), " \t");
        const ip = it.next() orelse continue;
        while (it.next()) |name| {
            if (std.ascii.eqlIgnoreCase(name, host)) {
                if (ip.len > out.len) return null;
                @memcpy(out[0..ip.len], ip);
                return out[0..ip.len];
            }
        }
    }
    return null;
}

/// First `nameserver` in /etc/resolv.conf, copied into `out`, or null.
fn firstNameserver(io: Io, out: []u8) ?[]const u8 {
    var buf: [4096]u8 = undefined;
    const content = readFile(io, "/etc/resolv.conf", &buf) catch return null;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line0| {
        var it = std.mem.tokenizeAny(u8, stripComment(line0), " \t");
        const kw = it.next() orelse continue;
        if (!std.mem.eql(u8, kw, "nameserver")) continue;
        const ns = it.next() orelse continue;
        if (ns.len > out.len) return null;
        @memcpy(out[0..ns.len], ns);
        return out[0..ns.len];
    }
    return null;
}

fn queryDns(io: Io, host: []const u8) ![4]u8 {
    var nsbuf: [64]u8 = undefined;
    const ns = firstNameserver(io, &nsbuf) orelse return error.NoNameserver;
    const ns_addr = net.IpAddress.parse(ns, 53) catch return error.NoNameserver;

    var q: [512]u8 = undefined;
    const qlen = try buildQuery(&q, host);

    var any = net.IpAddress.parse("0.0.0.0", 0) catch unreachable;
    var sock = any.bind(io, .{ .mode = .dgram, .protocol = .udp }) catch return error.ResolveFailed;
    defer sock.close(io);

    sock.send(io, &ns_addr, q[0..qlen]) catch return error.ResolveFailed;
    var rbuf: [512]u8 = undefined;
    const msg = sock.receiveTimeout(io, rbuf[0..], QUERY_TIMEOUT) catch return error.ResolveTimeout;
    return parseAnswer(msg.data);
}

fn be16(hi: u8, lo: u8) u16 {
    return (@as(u16, hi) << 8) | lo;
}

fn buildQuery(buf: []u8, host: []const u8) !usize {
    if (buf.len < 12 + host.len + 6) return error.NameTooLong;
    // header: id, flags(RD), qdcount=1, an/ns/ar=0
    @memcpy(buf[0..12], &[_]u8{ 0x13, 0x37, 0x01, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0 });
    var pos: usize = 12;
    var labels = std.mem.splitScalar(u8, host, '.');
    while (labels.next()) |label| {
        if (label.len == 0 or label.len > 63) continue;
        buf[pos] = @intCast(label.len);
        pos += 1;
        @memcpy(buf[pos..][0..label.len], label);
        pos += label.len;
    }
    buf[pos] = 0; // root
    pos += 1;
    // qtype A(1), qclass IN(1)
    @memcpy(buf[pos..][0..4], &[_]u8{ 0, 1, 0, 1 });
    return pos + 4;
}

/// Advance `pos` past a DNS name (label sequence or compression pointer).
fn skipName(resp: []const u8, pos: *usize) !void {
    while (pos.* < resp.len) {
        const b = resp[pos.*];
        if (b == 0) {
            pos.* += 1;
            return;
        }
        if ((b & 0xC0) == 0xC0) { // compression pointer: 2 bytes, ends the name
            pos.* += 2;
            return;
        }
        pos.* += 1 + b;
    }
    return error.Malformed;
}

/// Return the first A record's 4 bytes from a DNS response.
fn parseAnswer(resp: []const u8) ![4]u8 {
    if (resp.len < 12) return error.Malformed;
    if ((resp[3] & 0x0F) != 0) return error.DnsError; // RCODE != 0
    const ancount = be16(resp[6], resp[7]);
    if (ancount == 0) return error.NoARecord;

    var pos: usize = 12;
    try skipName(resp, &pos); // question name
    pos += 4; // qtype + qclass

    var i: usize = 0;
    while (i < ancount) : (i += 1) {
        try skipName(resp, &pos);
        if (pos + 10 > resp.len) return error.Malformed;
        const rtype = be16(resp[pos], resp[pos + 1]);
        const rdlen = be16(resp[pos + 8], resp[pos + 9]); // after type(2) class(2) ttl(4)
        pos += 10;
        if (pos + rdlen > resp.len) return error.Malformed;
        if (rtype == 1 and rdlen == 4) {
            return .{ resp[pos], resp[pos + 1], resp[pos + 2], resp[pos + 3] };
        }
        pos += rdlen;
    }
    return error.NoARecord;
}

test "buildQuery encodes labels + A/IN question" {
    var buf: [64]u8 = undefined;
    const n = try buildQuery(&buf, "origin");
    // header(12) + [6]"origin"(7) + root(1) + qtype/qclass(4) = 24
    try std.testing.expectEqual(@as(usize, 24), n);
    try std.testing.expectEqual(@as(u8, 6), buf[12]); // label length
    try std.testing.expectEqualStrings("origin", buf[13..19]);
    try std.testing.expectEqual(@as(u8, 0), buf[19]); // root
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 0, 1 }, buf[20..24]);
}

test "parseAnswer extracts A record past a compressed name" {
    // header: id, flags(RCODE 0), qd=1, an=1
    // question: "a" A IN ; answer: ptr(0xC00C) A IN ttl rdlen=4 1.2.3.4
    const resp = [_]u8{
        0x13, 0x37, 0x81, 0x80, 0, 1, 0, 1, 0, 0, 0, 0, // header
        1,    'a',  0,    0,    1, 0, 1, // question: label "a", A, IN
        0xC0, 0x0C, 0,    1,    0, 1, 0, 0, 0, 60, 0, 4, 1, 2, 3, 4, // answer
    };
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4 }, &(try parseAnswer(&resp)));
}
