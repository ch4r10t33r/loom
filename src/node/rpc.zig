//! RPC server: a line-delimited JSON inference endpoint over TCP.
//!
//! Request  (one JSON object per line):
//!   {"prompt": "hello", "max_tokens": 32, "temp": 0.0, "seed": 42}
//! Response (one JSON object per line):
//!   {"ok": true, "generated": N, "text": "...", "tokens": [...],
//!    "tok_per_s": X, "hit_rate": Y}
//!
//! The engine holds mutable per-request state (KV cache, scratch), so requests
//! are served strictly one at a time on this thread — simple and correct for a
//! v0 node. Continuous batching is a later concern.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const generator = @import("generator.zig");
const stats = @import("../core/stats.zig");
const meter_mod = @import("meter.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    gen: *generator.Generator,
    addr: []const u8,
    port: u16,
    seed: u64,
    /// Admin token gating the `credit` op (audit #6 P0-1). Empty = credit
    /// disabled entirely; non-empty = request must present a matching
    /// "admin_token". A payment rail replaces this with proof verification.
    admin_token: []const u8 = "",
    /// Per-client metering (SPEC.md node classes). When set, requests carry a
    /// "client" id, responses carry cost/balance, and exhausted clients get
    /// {"error":"payment_required"} until credited.
    meter: ?*meter_mod.Meter = null,
    /// The engine holds mutable per-request state, so generation is serialized
    /// by this mutex; connections otherwise run concurrently so one idle/slow
    /// client never blocks the accept loop. Shared (by pointer) with the
    /// OpenAI surface (openai.zig) so both serve paths serialize on one engine.
    engine_lock: *Io.Mutex,
};

const Conn = struct { ctx: *Ctx, stream: net.Stream };

/// Bound concurrent RPC handlers (audit #7 P1 thread-per-accept DoS).
const MAX_CONNS: u32 = 128;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.addr, ctx.port);
    var server = try address.listen(ctx.io, .{ .reuse_address = true });
    defer server.deinit(ctx.io);

    while (true) {
        const stream = server.accept(ctx.io) catch continue;
        if (live_conns.fetchAdd(1, .monotonic) >= MAX_CONNS) {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
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
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    while (true) {
        const raw = ri.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        if (line.len == 0) continue;
        handleRequest(ctx, line, wi) catch |e| {
            try wi.print("{{\"ok\":false,\"error\":\"{s}\"}}\n", .{@errorName(e)});
            try wi.flush();
        };
    }
}

fn handleRequest(ctx: *Ctx, line: []const u8, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, line, .{}) catch {
        try wi.print("{{\"ok\":false,\"error\":\"invalid_json\"}}\n", .{});
        try wi.flush();
        return;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        try wi.print("{{\"ok\":false,\"error\":\"expected_object\"}}\n", .{});
        try wi.flush();
        return;
    }
    const obj = parsed.value.object;

    const client: []const u8 = switch (obj.get("client") orelse .null) {
        .string => |s| s,
        else => "anon",
    };

    // settlement seam: {"method":"credit","client":X,"amount":N,"proof":...}
    // v1 accepts the proof unverified (trusted swarm); a payment rail replaces
    // exactly this check.
    if (obj.get("method")) |mv| {
        if (mv == .string and std.mem.eql(u8, mv.string, "credit")) {
            const m = ctx.meter orelse return wi.print("{{\"ok\":false,\"error\":\"no_meter\"}}\n", .{});
            // audit #6 P0-1: credit is admin-gated. No token configured -> the
            // op is disabled (v1 has no payment proof to verify).
            const supplied: []const u8 = switch (obj.get("admin_token") orelse .null) {
                .string => |s2| s2,
                else => "",
            };
            if (ctx.admin_token.len == 0 or !constEql(ctx.admin_token, supplied)) {
                try wi.print("{{\"ok\":false,\"error\":\"credit_forbidden\"}}\n", .{});
                try wi.flush();
                return;
            }
            const amount = @min(jsonUint64(obj.get("amount"), 0), @as(u64, 1) << 40); // cap
            const bal = try m.credit(client, amount);
            std.debug.print("meter: credited client={s} amount={d} balance={d}\n", .{ client, amount, bal });
            try wi.print("{{\"ok\":true,\"credited\":{d},\"balance\":{d}}}\n", .{ amount, bal });
            try wi.flush();
            return;
        }
        if (mv == .string and std.mem.eql(u8, mv.string, "tab")) {
            const m = ctx.meter orelse return wi.print("{{\"ok\":false,\"error\":\"no_meter\"}}\n", .{});
            const acc = m.account(client);
            try wi.print("{{\"ok\":true,\"used\":{d},\"balance\":{d}}}\n", .{ acc.used, acc.balance });
            try wi.flush();
            return;
        }
    }

    const prompt_str = switch (obj.get("prompt") orelse .null) {
        .string => |s| s,
        else => {
            try wi.print("{{\"ok\":false,\"error\":\"missing_prompt\"}}\n", .{});
            try wi.flush();
            return;
        },
    };

    const max_tokens: usize = jsonUint(obj.get("max_tokens"), 32);
    const temp: f32 = @floatCast(jsonFloat(obj.get("temp"), 0.0));
    const seed: u64 = jsonUint64(obj.get("seed"), ctx.seed);

    // metering gate: exhausted clients are refused before any compute; the
    // clamp to remaining allowance happens inside generate via the budget arg.
    var budget: ?u64 = null;
    if (ctx.meter) |m| {
        if (m.remaining(client) == 0) {
            try wi.print("{{\"ok\":false,\"error\":\"payment_required\",\"balance\":0}}\n", .{});
            try wi.flush();
            return;
        }
        budget = m.remaining(client);
    }

    ctx.engine_lock.lockUncancelable(ctx.io);
    const t0 = stats.nowNs(ctx.io);
    // raw user prompt: do not parse special tokens (no injection of control ids)
    var res = ctx.gen.generate(gpa, ctx.io, prompt_str, max_tokens, temp, seed, budget, null, false) catch |e| {
        ctx.engine_lock.unlock(ctx.io);
        return e;
    };
    const t1 = stats.nowNs(ctx.io);
    const hit_rate = ctx.gen.hitRate();
    ctx.engine_lock.unlock(ctx.io);
    defer res.deinit(gpa);
    const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const tok_s = if (secs > 0) @as(f64, @floatFromInt(res.completion_tokens)) / secs else 0;

    // build response (token-id array populated for loom, empty for gguf)
    try wi.print("{{\"ok\":true,\"generated\":{d},\"text\":", .{res.completion_tokens});
    try writeJsonBytes(wi, res.text);
    try wi.print(",\"tokens\":[", .{});
    for (res.token_ids, 0..) |tok, i| {
        if (i != 0) try wi.print(",", .{});
        try wi.print("{d}", .{tok});
    }
    try wi.print("],\"tok_per_s\":{d:.2},\"hit_rate\":{d:.4}", .{ tok_s, hit_rate });
    if (ctx.meter) |m| {
        // cost = prompt tokens processed + tokens generated
        const ch = try m.charge(client, @intCast(res.prompt_tokens + res.completion_tokens));
        try wi.print(",\"cost\":{d},\"balance\":{d}", .{ ch.cost, ch.balance });
    }
    try wi.print("}}\n", .{});
    try wi.flush();
}

/// Emit raw bytes as a JSON string with proper escaping.
fn writeJsonBytes(wi: *Io.Writer, bytes: []const u8) !void {
    try wi.print("\"", .{});
    for (bytes) |b| {
        switch (b) {
            '"' => try wi.print("\\\"", .{}),
            '\\' => try wi.print("\\\\", .{}),
            '\n' => try wi.print("\\n", .{}),
            '\r' => try wi.print("\\r", .{}),
            '\t' => try wi.print("\\t", .{}),
            0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try wi.print("{c}", .{b}),
            else => try wi.print("\\u{x:0>4}", .{b}),
        }
    }
    try wi.print("\"", .{});
}

/// Constant-time-ish equality for the admin token (avoids trivial early-exit).
fn constEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| diff |= x ^ y;
    return diff == 0;
}

fn jsonUint(v: ?std.json.Value, default: usize) usize {
    return switch (v orelse .null) {
        .integer => |i| if (i < 0) default else @intCast(i),
        else => default,
    };
}
fn jsonUint64(v: ?std.json.Value, default: u64) u64 {
    return switch (v orelse .null) {
        .integer => |i| if (i < 0) default else @intCast(i),
        else => default,
    };
}
fn jsonFloat(v: ?std.json.Value, default: f64) f64 {
    return switch (v orelse .null) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => default,
    };
}
