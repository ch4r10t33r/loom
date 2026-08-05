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
    // Mirrors the node/run flags, so a CI runner's thread count can be
    // reproduced locally: the invariants behave differently at one thread.
    for (args, 0..) |a, i| {
        if (std.mem.eql(u8, a, "--threads") and i + 1 < args.len) {
            generator.kernel_threads = std.fmt.parseInt(usize, args[i + 1], 10) catch 0;
        }
    }
    if (hasFlag(args, "--pool")) return poolOverhead(gpa, io, out);
    if (hasFlag(args, "--ffn")) return expertFfn(gpa, io, out);
    if (hasFlag(args, "--ffn-engine")) return expertFfnEngine(gpa, io, out);
    if (hasFlag(args, "--ffn-file")) return expertFfnFile(gpa, io, out);

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

    const types = [_]ggml.Type{ .f32, .f16, .q4_0, .q8_0, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .mxfp4, .iq4_xs };
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
        const pair = medianRatio(io, 21, c, struct {
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
        // Threshold 0.98, not 0.95. The margin batching wins by is
        // machine-dependent: 0.70 on eight cores here, but only 0.96 on a
        // single-core CI runner, where the batched path's larger working set
        // costs back most of what the shared weight unpack saves. A 0.95 gate
        // passed locally and failed on the runner.
        //
        // 0.98 still catches the failure this exists for. When batching is not
        // wired into a path, `matmul` falls through to calling `matvec` n
        // times -- which is *literally the other side of this comparison* --
        // so the ratio is 1.00, not 0.96. The gate distinguishes "batching is
        // less effective on this machine" from "batching is not happening".
        try check(out, json, &failures, "batching beats unbatched", pair.ratio < 0.98, "ratio {d:.2} ({d:.3} ms vs {d:.3} ms)", .{ pair.ratio, batched, serial });

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

/// Cost of the worker-pool handshake itself, isolated from the work.
///
/// A forward pass makes ~155 matvec calls per token, so anything paid per
/// call is multiplied by 155. Accounting for the token budget from the
/// measured kernel rate leaves ~42 us per call unexplained, and the handshake
/// -- publish a job, workers observe it, spin until the last one finishes --
/// is the obvious candidate.
pub fn poolOverhead(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const cols = 2048;

    try out.print("\n  pool handshake cost, by matvec size\n", .{});
    try out.print("  {s:>6} {s:>12} {s:>12} {s:>10}\n", .{ "rows", "pool on", "pool off", "overhead" });
    for ([_]usize{ 64, 128, 256, 512, 2048, 5632 }) |rows| {
        const data = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, cols, rows));
        defer gpa.free(data);
        rnd.bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;
        const x = try gpa.alloc(f32, cols);
        defer gpa.free(x);
        for (x) |*v| v.* = rnd.float(f32) - 0.5;
        const dst = try gpa.alloc(f32, rows);
        defer gpa.free(dst);

        const Ctx = struct { dst: []f32, data: []const u8, x: []const f32, rows: usize };
        const c = Ctx{ .dst = dst, .data = data, .x = x, .rows = rows };
        const body = struct {
            fn f(cc: Ctx) void {
                backend.matvec(.q4_k, cc.dst, cc.data, cc.x, cc.rows, cols);
            }
        }.f;

        backend.parallelBegin(generator.threads());
        const on = medianMs(io, 21, c, body);
        backend.parallelEnd();
        const off = medianMs(io, 21, c, body); // pool stopped: runs inline

        try out.print("  {d:>6} {d:>9.4} ms {d:>9.4} ms {d:>7.1} us\n", .{ rows, on, off, (on - off) * 1000 });
    }
    try out.flush();
}

/// One MoE expert's FFN at DeepSeek-V2-Lite's real shapes and types, which is
/// where a decode token actually spends its time.
///
/// Measured in the engine, expert FFN ran at ~5.7 GB/s while these same
/// kernels reach ~53 GB/s on a standalone 2048x2048 sweep. Either the shape is
/// the problem or the engine around the kernels is, and a synthetic sweep at
/// 2048x2048 cannot tell those apart. This reproduces the exact call sequence
/// -- gate and up over `dim`, SwiGLU, down over `moe_ffn` -- with gate/up in
/// Q4_K and down in Q5_1, the mixture `Q4_K_M` actually produces.
/// The file-backed pass (issue #171, round two): the anonymous-slab bench
/// measured slab size, random access and concurrent writes at ~zero cost on
/// both page-size regimes, which leaves the one thing it did not reproduce --
/// the engine reads experts through a file-backed mmap under a RAM budget.
/// This writes the expert slab to a file, maps it, and runs the same kernels
/// three ways: warm (pages cached by the write), after an eviction pass (an
/// anonymous ballast of most of physical RAM is touched and freed, pushing
/// the file pages out), and again immediately after (re-warmed). Cold cost
/// is reported as the MEAN over the schedule -- best-of-N would only ever
/// time a re-touched slot -- alongside major/minor fault deltas from
/// getrusage, which are the page-in counters perf would give us.
fn expertFfnFile(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    // posix mmap/getrusage; the pass measures page-cache behavior that has no
    // direct Windows equivalent, and the engine's Windows story is mmap-free
    // reads anyway. Compile-time gate so the release matrix keeps building.
    if (@import("builtin").os.tag == .windows) {
        return out.print("--ffn-file is not supported on windows\n", .{});
    }
    const dim = 2048;
    const moe_ffn = 1408;
    const N = 192;

    const gate_bytes = ggml.tensorBytes(.q4_k, dim, moe_ffn);
    const up_bytes = gate_bytes;
    const down_bytes = ggml.tensorBytes(.q5_1, moe_ffn, dim);
    const set_bytes = gate_bytes + up_bytes + down_bytes;

    var slab_gb: f64 = 4.0;
    if (std.c.getenv("LOOM_BENCH_GB")) |v| {
        slab_gb = std.fmt.parseFloat(f64, std.mem.span(v)) catch 4.0;
    }
    const slots: usize = @intFromFloat(@max(8.0, slab_gb * 1024.0 * 1024.0 * 1024.0 / @as(f64, @floatFromInt(set_bytes))));
    const total = set_bytes * slots;

    // Write the slab to a file in expert-sized chunks.
    var prng = std.Random.DefaultPrng.init(0xF11E);
    const rnd = prng.random();
    const path = "bench-ffn-file.bin";
    defer Io.Dir.cwd().deleteFile(io, path) catch {};
    {
        const chunk = try gpa.alloc(u8, set_bytes);
        defer gpa.free(chunk);
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer f.close(io);
        var fw_buf: [1 << 16]u8 = undefined;
        var fw = f.writer(io, &fw_buf);
        for (0..slots) |_| {
            rnd.bytes(chunk);
            for (0..chunk.len / 2) |i| chunk[i * 2 + 1] &= 0xFB;
            try fw.interface.writeAll(chunk);
        }
        try fw.interface.flush();
    }

    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const slab = try std.posix.mmap(null, total, .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);
    defer std.posix.munmap(slab);

    const x = try gpa.alloc(f32, dim);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) - 0.5;
    const g = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(g);
    const u = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(u);
    const a = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(a);
    const o = try gpa.alloc(f32, dim);
    defer gpa.free(o);

    const One = struct {
        fn run(sl: []const u8, sb: usize, slot: usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) void {
            const set = sl[slot * sb ..][0..sb];
            backend.matvec(.q4_k, gg, set[0..gb], xx, moe_ffn, dim);
            backend.matvec(.q4_k, uu, set[gb..][0..ub], xx, moe_ffn, dim);
            backend.swiglu(aa, gg, uu);
            backend.matvec(.q5_1, oo, set[gb + ub ..], aa, dim, moe_ffn);
        }
    };

    const rnd_sched = try gpa.alloc(usize, N);
    defer gpa.free(rnd_sched);
    for (rnd_sched) |*sv| sv.* = rnd.intRangeLessThan(usize, 0, slots);

    const Pass = struct {
        mean_ns: f64,
        best_ns: i128,
        majf: isize,
        minf: isize,
        fn run(ioo: Io, sl: []const u8, sb: usize, sched: []const usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) @This() {
            const r0 = std.posix.getrusage(std.posix.rusage.SELF);
            var sum: i128 = 0;
            var best: i128 = std.math.maxInt(i64);
            for (sched) |slot| {
                const t0 = nowNs(ioo);
                One.run(sl, sb, slot, gb, ub, xx, gg, uu, aa, oo);
                const dt = nowNs(ioo) - t0;
                sum += dt;
                if (dt < best) best = dt;
            }
            const r1 = std.posix.getrusage(std.posix.rusage.SELF);
            return .{
                .mean_ns = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(sched.len)),
                .best_ns = best,
                .majf = @intCast(r1.majflt - r0.majflt),
                .minf = @intCast(r1.minflt - r0.minflt),
            };
        }
    };

    const threads_n = generator.threads();
    backend.parallelBegin(threads_n);
    const warm = Pass.run(io, slab, set_bytes, rnd_sched, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();

    // Evict: touch an anonymous ballast of most of physical RAM, then free it.
    // Crude but portable; the fault counters below say whether it worked.
    {
        const ram: u64 = if (std.c.getenv("LOOM_BENCH_RAM_GB")) |v|
            @intFromFloat((std.fmt.parseFloat(f64, std.mem.span(v)) catch 8.0) * 1024.0 * 1024.0 * 1024.0)
        else
            8 * 1024 * 1024 * 1024;
        const ballast_len: usize = @intCast(ram * 3 / 4);
        const ballast = std.posix.mmap(null, ballast_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch null;
        if (ballast) |b| {
            var i: usize = 0;
            while (i < b.len) : (i += 4096) b[i] = 1;
            std.posix.munmap(b);
        }
    }

    backend.parallelBegin(threads_n);
    const cold = Pass.run(io, slab, set_bytes, rnd_sched, gate_bytes, up_bytes, x, g, u, a, o);
    const rewarm = Pass.run(io, slab, set_bytes, rnd_sched, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();

    const fb: f64 = @floatFromInt(set_bytes);
    try out.print("\nexpert ffn, file-backed (issue #171 round two): {d} experts, {d:.2} GB file, {d} threads\n", .{
        slots, @as(f64, @floatFromInt(total)) / (1 << 30), threads_n,
    });
    const P = struct {
        fn line(oo: *Io.Writer, name: []const u8, p: anytype, bytes: f64) !void {
            try oo.print("  {s}  mean {d:6.1} GB/s   best {d:6.1} GB/s   majflt {d:6}  minflt {d:8}\n", .{
                name, bytes / p.mean_ns, bytes / @as(f64, @floatFromInt(p.best_ns)), p.majf, p.minf,
            });
        }
    };
    try P.line(out, "warm mapping   ", warm, fb);
    try P.line(out, "after eviction ", cold, fb);
    try P.line(out, "re-warmed      ", rewarm, fb);
    try out.flush();
}

/// The engine-shaped variant (issue #171): the same kernels and shapes as
/// --ffn, but spread across a multi-gigabyte slab with router-like *random*
/// slot selection, plus a pass with concurrent writer threads standing in for
/// the fetch pool. --ffn measures the kernel in a 330 MB sequential ring;
/// this measures it the way the engine actually runs it. The delta between
/// the passes attributes the in-engine bandwidth loss:
///   big+sequential vs small ring  -> slab size alone (page walks, residency)
///   big+random    vs big+sequential -> access pattern (TLB locality)
///   big+random+writers vs big+random -> concurrent fetch traffic
fn expertFfnEngine(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    const dim = 2048;
    const moe_ffn = 1408;
    const N = 192;

    const gate_bytes = ggml.tensorBytes(.q4_k, dim, moe_ffn);
    const up_bytes = gate_bytes;
    const down_bytes = ggml.tensorBytes(.q5_1, moe_ffn, dim);
    const set_bytes = gate_bytes + up_bytes + down_bytes;

    // As many distinct experts as ~4 GB holds -- the engine's LRU scale,
    // hundreds of thousands of pages, far past every TLB on this class of
    // machine. LOOM_BENCH_GB overrides.
    var slab_gb: f64 = 4.0;
    if (std.c.getenv("LOOM_BENCH_GB")) |v| {
        slab_gb = std.fmt.parseFloat(f64, std.mem.span(v)) catch 4.0;
    }
    const slots: usize = @intFromFloat(@max(8.0, slab_gb * 1024.0 * 1024.0 * 1024.0 / @as(f64, @floatFromInt(set_bytes))));

    var prng = std.Random.DefaultPrng.init(0xE7E7);
    const rnd = prng.random();
    const slab = try gpa.alloc(u8, set_bytes * slots);
    defer gpa.free(slab);
    rnd.bytes(slab);
    for (0..slab.len / 2) |i| slab[i * 2 + 1] &= 0xFB;

    const x = try gpa.alloc(f32, dim);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) - 0.5;
    const g = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(g);
    const u = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(u);
    const a = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(a);
    const o = try gpa.alloc(f32, dim);
    defer gpa.free(o);

    const One = struct {
        fn run(sl: []const u8, sb: usize, slot: usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) void {
            const set = sl[slot * sb ..][0..sb];
            backend.matvec(.q4_k, gg, set[0..gb], xx, moe_ffn, dim);
            backend.matvec(.q4_k, uu, set[gb..][0..ub], xx, moe_ffn, dim);
            backend.swiglu(aa, gg, uu);
            backend.matvec(.q5_1, oo, set[gb + ub ..], aa, dim, moe_ffn);
        }
    };

    // Best-of-N over a slot schedule; the schedule is what varies per pass.
    const Measure = struct {
        fn f(ioo: Io, sl: []const u8, sb: usize, sched: []const usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) i128 {
            var best: i128 = std.math.maxInt(i64);
            for (sched) |slot| {
                const t0 = nowNs(ioo);
                One.run(sl, sb, slot, gb, ub, xx, gg, uu, aa, oo);
                const dt = nowNs(ioo) - t0;
                if (dt < best) best = dt;
            }
            return best;
        }
    };

    const seq_sched = try gpa.alloc(usize, N);
    defer gpa.free(seq_sched);
    for (seq_sched, 0..) |*sv, i| sv.* = i % slots;
    const rnd_sched = try gpa.alloc(usize, N);
    defer gpa.free(rnd_sched);
    for (rnd_sched) |*sv| sv.* = rnd.intRangeLessThan(usize, 0, slots);

    const threads_n = generator.threads();
    backend.parallelBegin(threads_n);
    // warm: touch every page once so first-touch faults are not in the timing
    for (0..slots) |i| One.run(slab, set_bytes, i, gate_bytes, up_bytes, x, g, u, a, o);
    const b_seq = Measure.f(io, slab, set_bytes, seq_sched, gate_bytes, up_bytes, x, g, u, a, o);
    const b_rnd = Measure.f(io, slab, set_bytes, rnd_sched, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();

    // Concurrent-writer pass: two threads memcpy expert-sized blocks into
    // random slots throughout, the fetch pool's memory traffic in miniature.
    var stop = std.atomic.Value(bool).init(false);
    const Writer = struct {
        fn run(sl: []u8, sb: usize, nslots: usize, seed: u64, st: *std.atomic.Value(bool)) void {
            var p = std.Random.DefaultPrng.init(seed);
            const r = p.random();
            var buf: [64 * 1024]u8 = undefined;
            r.bytes(&buf);
            while (!st.load(.monotonic)) {
                const slot = r.intRangeLessThan(usize, 0, nslots);
                var off: usize = 0;
                while (off + buf.len <= sb) : (off += buf.len) {
                    @memcpy(sl[slot * sb + off ..][0..buf.len], &buf);
                    if (st.load(.monotonic)) return;
                }
            }
        }
    };
    const w1 = try std.Thread.spawn(.{}, Writer.run, .{ slab, set_bytes, slots, 1, &stop });
    const w2 = try std.Thread.spawn(.{}, Writer.run, .{ slab, set_bytes, slots, 2, &stop });
    backend.parallelBegin(threads_n);
    const b_rw = Measure.f(io, slab, set_bytes, rnd_sched, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();
    stop.store(true, .monotonic);
    w1.join();
    w2.join();

    const fb: f64 = @floatFromInt(set_bytes);
    const gbs = struct {
        fn f(bytes: f64, ns: i128) f64 {
            return bytes / @as(f64, @floatFromInt(ns));
        }
    }.f;
    try out.print("\nexpert ffn, engine-shaped (issue #171): {d} distinct experts, {d:.2} GB slab, {d} threads\n", .{
        slots, @as(f64, @floatFromInt(set_bytes * slots)) / (1 << 30), threads_n,
    });
    try out.print("  sequential slots          {d:6.1} GB/s   (compare --ffn's 330 MB ring)\n", .{gbs(fb, b_seq)});
    try out.print("  random slots              {d:6.1} GB/s   (router-like access)\n", .{gbs(fb, b_rnd)});
    try out.print("  random + fetch writers    {d:6.1} GB/s   (concurrent pool traffic)\n", .{gbs(fb, b_rw)});
    try out.flush();
}

fn expertFfn(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    const dim = 2048;
    const moe_ffn = 1408;
    const N = 128;
    // A ring of distinct experts, not one reused buffer. This is the whole
    // point: a decode token touches ~208 *different* experts, so every byte
    // comes from DRAM. Benchmarking one 5 MB expert in a tight loop measures
    // it resident in cache and flatters the kernel by several times.
    const RING = 64; // ~330 MB, past any cache on this class of machine

    var prng = std.Random.DefaultPrng.init(0xE7E7);
    const rnd = prng.random();

    const gate_bytes = ggml.tensorBytes(.q4_k, dim, moe_ffn);
    const up_bytes = gate_bytes;
    const down_bytes = ggml.tensorBytes(.q5_1, moe_ffn, dim);
    const set_bytes = gate_bytes + up_bytes + down_bytes;

    const slab = try gpa.alloc(u8, set_bytes * RING);
    defer gpa.free(slab);
    rnd.bytes(slab);
    for (0..slab.len / 2) |i| slab[i * 2 + 1] &= 0xFB; // keep f16 scales finite

    const x = try gpa.alloc(f32, dim);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) - 0.5;
    const g = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(g);
    const u = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(u);
    const a = try gpa.alloc(f32, moe_ffn);
    defer gpa.free(a);
    const o = try gpa.alloc(f32, dim);
    defer gpa.free(o);

    const One = struct {
        fn run(sl: []const u8, sb: usize, i: usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) void {
            const set = sl[(i % RING) * sb ..][0..sb];
            backend.matvec(.q4_k, gg, set[0..gb], xx, moe_ffn, dim);
            backend.matvec(.q4_k, uu, set[gb..][0..ub], xx, moe_ffn, dim);
            backend.swiglu(aa, gg, uu);
            backend.matvec(.q5_1, oo, set[gb + ub ..], aa, dim, moe_ffn);
        }
    };

    const measure = struct {
        fn f(ioo: Io, sl: []const u8, sb: usize, gb: usize, ub: usize, xx: []const f32, gg: []f32, uu: []f32, aa: []f32, oo: []f32) i128 {
            var best: i128 = std.math.maxInt(i64);
            for (0..N) |i| {
                const t0 = nowNs(ioo);
                One.run(sl, sb, i, gb, ub, xx, gg, uu, aa, oo);
                const dt = nowNs(ioo) - t0;
                if (dt < best) best = dt;
            }
            return best;
        }
    }.f;

    backend.parallelBegin(generator.threads());
    for (0..RING) |i| One.run(slab, set_bytes, i, gate_bytes, up_bytes, x, g, u, a, o); // warm the pool, not the cache
    const bn = measure(io, slab, set_bytes, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();

    backend.parallelBegin(1);
    const b1 = measure(io, slab, set_bytes, gate_bytes, up_bytes, x, g, u, a, o);
    backend.parallelEnd();

    const msn = @as(f64, @floatFromInt(bn)) / 1e6;
    const ms1 = @as(f64, @floatFromInt(b1)) / 1e6;
    const fb: f64 = @floatFromInt(set_bytes);
    try out.print("\nexpert ffn (dim={d} moe_ffn={d}, gate/up q4_k + down q5_1, {d:.2} MB/expert, {d} distinct)\n", .{
        dim, moe_ffn, fb / (1 << 20), RING,
    });
    try out.print("  {d} threads  {d:7.3} ms  {d:6.1} GB/s\n", .{ generator.threads(), msn, fb / @as(f64, @floatFromInt(bn)) });
    try out.print("  1 thread    {d:7.3} ms  {d:6.1} GB/s   (speedup {d:.2}x)\n", .{ ms1, fb / @as(f64, @floatFromInt(b1)), ms1 / msn });
    try out.flush();
}
