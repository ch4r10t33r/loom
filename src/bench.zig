//! `loom bench` — kernel and end-to-end benchmarks, with regression detection.
//!
//! Two kinds of number come out of this, and they are not interchangeable.
//!
//! **Absolute timings** (ms, tokens/sec) are what you quote in a release note.
//! They are a property of the machine as much as the code, so comparing them
//! across hosts is meaningless and comparing them across commits is only
//! meaningful on the same host, idle, on mains power.
//!
//! **Invariants** are ratios that hold regardless of how fast the machine is:
//! batching must beat the same work unbatched, threads must beat one thread,
//! a GPU kernel must beat the CPU it was written to replace. Those are
//! properties of the implementation, so they survive a noisy shared CI runner
//! where an absolute threshold would flap.
//!
//! CI therefore gates on the invariants and merely reports the timings. A
//! regression that matters — the kind where a kernel quietly loses its
//! vectorization, or batching stops being wired into a code path, which has
//! already happened twice here — breaks an invariant, not a threshold.

const std = @import("std");
const Io = std.Io;
const backend = @import("compute/backend.zig");
const ggml = @import("gguf/ggml.zig");
const gguf = @import("gguf/gguf.zig");
const llama = @import("gguf/llama.zig");
const generator = @import("node/generator.zig");
const stats = @import("core/stats.zig");

/// A realistic FFN shape: 2048 -> 5632 is TinyLlama's ffn_up.
const COLS = 2048;
const ROWS = 5632;

fn nowNs(io: Io) i128 {
    return stats.nowNs(io);
}

/// Median of `n` timed runs. Median rather than mean: one scheduler hiccup
/// should not move the number, and rather than best-of, because a best-of
/// quietly reports the machine's luckiest moment as its speed.
fn medianMs(io: Io, comptime n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) void) f64 {
    f(ctx); // warm up: first touch faults in buffers and populates caches
    var samples: [n]f64 = undefined;
    for (&samples) |*s| {
        const t0 = nowNs(io);
        f(ctx);
        s.* = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e6;
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    return samples[n / 2];
}

/// Median of the *ratio* of two alternatives, measured adjacently in each
/// round.
///
/// Timing them separately and dividing is what this did first, and it is
/// wrong on a busy machine: a thermal dip or a noisy neighbour lands on one
/// side of the comparison and not the other, and the ratio moves by more than
/// the effect being measured. Interleaving puts both under the same
/// conditions in every round, so machine-wide noise cancels and what is left
/// is the difference between the two implementations.
fn medianRatio(
    io: Io,
    comptime n: usize,
    ctx: anytype,
    comptime a: fn (@TypeOf(ctx)) void,
    comptime b: fn (@TypeOf(ctx)) void,
) struct { ratio: f64, a_ms: f64, b_ms: f64 } {
    // Warm up first: the batched path faults in a large thread-local
    // activation buffer on its first call, which showed up as a cold ratio of
    // 0.94 against a 0.95 threshold -- a pass, but only just, and it would
    // have flapped on a slower runner.
    a(ctx);
    b(ctx);

    var ratios: [n]f64 = undefined;
    var a_ms: [n]f64 = undefined;
    var b_ms: [n]f64 = undefined;
    for (0..n) |i| {
        const t0 = nowNs(io);
        a(ctx);
        const t1 = nowNs(io);
        b(ctx);
        const t2 = nowNs(io);
        a_ms[i] = @as(f64, @floatFromInt(t1 - t0)) / 1e6;
        b_ms[i] = @as(f64, @floatFromInt(t2 - t1)) / 1e6;
        ratios[i] = a_ms[i] / @max(b_ms[i], 1e-9);
    }
    std.mem.sort(f64, &ratios, {}, std.sort.asc(f64));
    std.mem.sort(f64, &a_ms, {}, std.sort.asc(f64));
    std.mem.sort(f64, &b_ms, {}, std.sort.asc(f64));
    return .{ .ratio = ratios[n / 2], .a_ms = a_ms[n / 2], .b_ms = b_ms[n / 2] };
}

const Result = struct {
    name: []const u8,
    ms: f64,
    note: []const u8 = "",
};

pub fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, args: [][]const u8) !void {
    const json = hasFlag(args, "--json");
    const strict = hasFlag(args, "--check");

    var results = std.ArrayList(Result).empty;
    defer results.deinit(gpa);

    if (!json) {
        try out.print("loom bench — backend {s}, {d} cpu threads\n\n", .{ backend.name, generator.threads() });
        try out.flush();
    }

    // ---- kernels ----
    var prng = std.Random.DefaultPrng.init(0xBE7C);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, COLS);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) - 0.5;
    const dst = try gpa.alloc(f32, ROWS);
    defer gpa.free(dst);

    const types = [_]ggml.Type{ .f32, .f16, .q4_0, .q8_0, .q4_k, .q5_k, .q6_k, .iq4_xs };
    var q4k_1thread: f64 = 0;
    var q4k_nthread: f64 = 0;

    for (types) |t| {
        const data = try gpa.alloc(u8, ggml.tensorBytes(t, COLS, ROWS));
        defer gpa.free(data);
        rnd.bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB; // keep f16 scales finite

        const Ctx = struct { t: ggml.Type, dst: []f32, data: []const u8, x: []const f32 };
        const c = Ctx{ .t = t, .dst = dst, .data = data, .x = x };
        const body = struct {
            fn f(cc: Ctx) void {
                backend.matvec(cc.t, cc.dst, cc.data, cc.x, ROWS, COLS);
            }
        }.f;

        backend.parallelBegin(generator.threads());
        const ms = medianMs(io, 9, c, body);
        backend.parallelEnd();

        if (t == .q4_k) {
            q4k_nthread = ms;
            backend.parallelBegin(1);
            q4k_1thread = medianMs(io, 9, c, body);
            backend.parallelEnd();
        }

        var nb: [48]u8 = undefined;
        try results.append(gpa, .{
            .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "matvec/{s}", .{@tagName(t)})),
            .ms = ms,
        });
    }

    // ---- batched matmul, the prefill path ----
    {
        const t: ggml.Type = .q4_k;
        const data = try gpa.alloc(u8, ggml.tensorBytes(t, COLS, ROWS));
        defer gpa.free(data);
        rnd.bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;

        const n = backend.MAX_BATCH;
        const xs = try gpa.alloc(f32, n * COLS);
        defer gpa.free(xs);
        for (xs) |*v| v.* = rnd.float(f32) - 0.5;
        const outs = try gpa.alloc(f32, n * ROWS);
        defer gpa.free(outs);

        const Ctx = struct { data: []const u8, xs: []const f32, outs: []f32, n: usize };
        const c = Ctx{ .data = data, .xs = xs, .outs = outs, .n = n };
        backend.parallelBegin(generator.threads());
        const pair = medianRatio(io, 11, c, struct {
            fn f(cc: Ctx) void {
                backend.matmul(.q4_k, cc.outs, cc.data, cc.xs, cc.n, ROWS, COLS);
            }
        }.f, struct {
            fn f(cc: Ctx) void {
                for (0..cc.n) |k| {
                    backend.matvec(.q4_k, cc.outs[k * ROWS ..][0..ROWS], cc.data, cc.xs[k * COLS ..][0..COLS], ROWS, COLS);
                }
            }
        }.f);
        backend.parallelEnd();
        const batched = pair.a_ms;
        const serial = pair.b_ms;

        try results.append(gpa, .{ .name = "matmul/q4_k x8", .ms = batched });
        try results.append(gpa, .{ .name = "matvec/q4_k x8 (unbatched)", .ms = serial });

        if (!json) try printTable(out, results.items);

        // ---- invariants ----
        var failures: usize = 0;
        try check(out, json, &failures, "batching beats unbatched", pair.ratio < 0.95, "ratio {d:.2} ({d:.3} ms vs {d:.3} ms)", .{ pair.ratio, batched, serial });

        // Only meaningful for the CPU backend: a GPU backend does not use the
        // CPU worker pool for the kernels it owns, so the comparison would be
        // measuring nothing.
        const cores = generator.threads();
        if (backend.kind == .cpu and cores >= 4) {
            try check(out, json, &failures, "threads beat one thread", q4k_nthread < q4k_1thread * 0.6, "{d:.3} ms vs {d:.3} ms at 1 thread", .{ q4k_nthread, q4k_1thread });
        }

        if (backend.kind != .cpu) {
            // A GPU backend exists to be faster than the CPU it replaces. If it
            // is not, it should be reported, not shipped quietly.
            //
            // The shape here is 5632 rows, which is below the Metal backend's
            // current row threshold, so this measures the *selected* path:
            // it passes because the backend correctly declines to use a GPU
            // that would lose. That is the property worth gating on — not
            // "the GPU is used", but "whatever gets picked is the faster one".
            const cpu_ms = blk: {
                const cpu = @import("compute/cpu.zig");
                const C2 = struct { dst: []f32, data: []const u8, x: []const f32 };
                const c2 = C2{ .dst = dst, .data = data, .x = x };
                cpu.parallelBegin(generator.threads());
                defer cpu.parallelEnd();
                break :blk medianMs(io, 9, c2, struct {
                    fn f(cc: C2) void {
                        cpu.matvec(.q4_k, cc.dst, cc.data, cc.x, ROWS, COLS);
                    }
                }.f);
            };
            // Equality is a pass, not a failure: when the backend declines the
            // GPU for a shape it would lose on, the "gpu" path *is* the CPU
            // path and the two numbers are the same measurement. What must
            // never happen is the selected path being slower.
            try check(out, json, &failures, "selected path is not slower", q4k_nthread <= cpu_ms * 1.10, "{d:.3} ms selected vs {d:.3} ms cpu", .{ q4k_nthread, cpu_ms });
        }

        if (json) try emitJson(out, results.items, failures);

        if (strict and failures > 0) {
            try out.print("\n{d} invariant(s) failed\n", .{failures});
            try out.flush();
            return error.BenchmarkRegression;
        }
    }
    try out.flush();
}

fn check(out: *Io.Writer, json: bool, failures: *usize, name: []const u8, ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (!ok) failures.* += 1;
    if (json) return;
    try out.print("  {s} {s:<28} ", .{ if (ok) "ok  " else "FAIL", name });
    try out.print(fmt, args);
    try out.print("\n", .{});
}

fn printTable(out: *Io.Writer, rows: []const Result) !void {
    try out.print("  {s:<30} {s:>10}\n", .{ "kernel", "ms" });
    try out.print("  {s:-<30} {s:->10}\n", .{ "", "" });
    for (rows) |r| try out.print("  {s:<30} {d:>10.3}\n", .{ r.name, r.ms });
    try out.print("\n  invariants (machine-independent; these are what CI gates on)\n", .{});
    try out.flush();
}

fn emitJson(out: *Io.Writer, rows: []const Result, failures: usize) !void {
    try out.print("{{\"backend\":\"{s}\",\"threads\":{d},\"failures\":{d},\"kernels\":{{", .{ backend.name, generator.threads(), failures });
    for (rows, 0..) |r, i| {
        if (i > 0) try out.print(",", .{});
        try out.print("\"{s}\":{d:.4}", .{ r.name, r.ms });
    }
    try out.print("}}}}\n", .{});
    try out.flush();
}

fn hasFlag(args: [][]const u8, name: []const u8) bool {
    for (args) |a| {
        if (std.mem.eql(u8, a, name)) return true;
    }
    return false;
}
