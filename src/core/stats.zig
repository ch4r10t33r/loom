//! Runtime measurement: RSS, throughput, and the per-expert usage histogram
//! that sizes the pinned hot set (STATS -> PIN, CLAUDE latency-hiding lever 1).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// Resident set size in bytes (peak). macOS reports ru_maxrss in bytes; most
/// other Unixes report kilobytes.
pub fn rssBytes() u64 {
    const ru = std.posix.getrusage(std.posix.rusage.SELF);
    const maxrss: i64 = @intCast(ru.maxrss);
    if (maxrss <= 0) return 0;
    return switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos => @intCast(maxrss),
        else => @as(u64, @intCast(maxrss)) * 1024,
    };
}

pub fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.toNanoseconds();
}

/// Write a STATS file: expert_id, access_count sorted by heat (descending), plus
/// the cumulative coverage — exactly what `PIN_GB` reads to choose the hot set.
pub fn writeUsageHistogram(
    gpa: std.mem.Allocator,
    io: Io,
    path: []const u8,
    access_count: []const u64,
    expert_bytes: usize,
) !void {
    const Pair = struct { id: usize, count: u64 };
    var pairs = try gpa.alloc(Pair, access_count.len);
    defer gpa.free(pairs);
    for (access_count, 0..) |c, i| pairs[i] = .{ .id = i, .count = c };
    std.mem.sort(Pair, pairs, {}, struct {
        fn lt(_: void, a: Pair, b: Pair) bool {
            return a.count > b.count;
        }
    }.lt);

    var total: u64 = 0;
    for (access_count) |c| total += c;

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(gpa);
    try buf.print(gpa, "# loom expert usage histogram\n", .{});
    try buf.print(gpa, "# expert_bytes={d} total_accesses={d}\n", .{ expert_bytes, total });
    try buf.print(gpa, "# rank expert_id access_count cumulative_fraction cumulative_pinned_bytes\n", .{});
    var cum: u64 = 0;
    var pinned_bytes: u64 = 0;
    for (pairs, 0..) |p, rank| {
        if (p.count == 0) break;
        cum += p.count;
        pinned_bytes += expert_bytes;
        const frac = if (total == 0) 0.0 else @as(f64, @floatFromInt(cum)) / @as(f64, @floatFromInt(total));
        try buf.print(gpa, "{d} {d} {d} {d:.4} {d}\n", .{ rank, p.id, p.count, frac, pinned_bytes });
    }

    const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
}

/// Persist raw per-expert access counts (binary u64 array) so a later run can
/// pin the hot set — the measure-then-pin loop (run with STATS, rerun with PIN).
pub fn writeCounts(io: Io, path: []const u8, counts: []const u64) !void {
    const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, std.mem.sliceAsBytes(counts));
}

/// Read raw counts written by `writeCounts`. Returns null if the file is absent.
/// The returned length must match the current expert count or it's rejected.
pub fn readCounts(gpa: std.mem.Allocator, io: Io, path: []const u8, expected_len: usize) !?[]u64 {
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer f.close(io);
    const size = (try f.stat(io)).size;
    if (size != expected_len * @sizeOf(u64)) return null;
    const out = try gpa.alloc(u64, expected_len);
    errdefer gpa.free(out);
    _ = try f.readPositionalAll(io, std.mem.sliceAsBytes(out), 0);
    return out;
}

/// Choose the hottest experts fitting in `pin_budget_bytes`, returning their ids.
/// Caller owns the returned slice. This is the STATS->PIN sizing decision.
pub fn hottestWithinBudget(
    gpa: std.mem.Allocator,
    access_count: []const u64,
    expert_bytes: usize,
    pin_budget_bytes: u64,
) ![]usize {
    const Pair = struct { id: usize, count: u64 };
    var pairs = try gpa.alloc(Pair, access_count.len);
    defer gpa.free(pairs);
    for (access_count, 0..) |c, i| pairs[i] = .{ .id = i, .count = c };
    std.mem.sort(Pair, pairs, {}, struct {
        fn lt(_: void, a: Pair, b: Pair) bool {
            return a.count > b.count;
        }
    }.lt);

    const capacity = if (expert_bytes == 0) 0 else pin_budget_bytes / expert_bytes;
    var out = std.ArrayList(usize).empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < pairs.len and i < capacity) : (i += 1) {
        if (pairs[i].count == 0) break; // don't pin never-used experts
        try out.append(gpa, pairs[i].id);
    }
    return out.toOwnedSlice(gpa);
}

test "hottestWithinBudget picks the most-used within capacity" {
    const gpa = std.testing.allocator;
    const counts = [_]u64{ 1, 100, 5, 100, 0, 50 };
    // budget for 2 experts of 10 bytes each
    const ids = try hottestWithinBudget(gpa, &counts, 10, 20);
    defer gpa.free(ids);
    try std.testing.expect(ids.len == 2);
    // the two hottest are ids 1 and 3 (count 100)
    for (ids) |id| try std.testing.expect(id == 1 or id == 3);
}
