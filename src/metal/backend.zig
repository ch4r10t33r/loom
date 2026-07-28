//! Metal compute backend (issue #12).
//!
//! Status: the substrate is complete and one kernel is real. Q4_K matvec runs
//! on the GPU; everything else still falls through to the CPU kernels, which
//! also serve as the oracle every Metal kernel is validated against. That
//! ordering is deliberate — Q4_K plus Q6_K is the whole of a Q4_K_M model, so
//! the first kernel is already the majority of the work in a real model.
//!
//! Weights are wrapped, never uploaded. `Device.wrapHost` puts the mmap'd GGUF
//! region straight behind an MTLBuffer, so a routed expert that arrived from a
//! peer mid-token is visible to the GPU without a copy. On a discrete GPU that
//! same fetch would need a host-to-VRAM transfer inside the token loop, which
//! is why Apple comes first.

const std = @import("std");
const cpu = @import("../compute/cpu.zig");
const ggml = @import("../gguf/ggml.zig");
const mtl = @import("device.zig");

// Ops not yet on the GPU fall through unchanged.
pub const matmul = cpu.matmul;
pub const dequantRow = cpu.dequantRow;
pub const dotF32 = cpu.dotF32;
pub const axpy = cpu.axpy;
pub const rmsnorm = cpu.rmsnorm;
pub const softmax = cpu.softmax;
pub const swiglu = cpu.swiglu;
pub const add = cpu.add;
pub const sigmoid = cpu.sigmoid;
pub const MAX_BATCH = cpu.MAX_BATCH;

const dmmv_q4k_src = @embedFile("../shaders/metal/dmmv_q4k.metal");

/// Below this many rows the dispatch and synchronization cost more than the
/// work: norm and router projections stay on the CPU.
const MIN_GPU_ROWS = 512;

/// Apple GPUs execute in 32-lane SIMD groups; the kernel assigns one per row.
const SIMD_W = 32;
/// SIMD groups per threadgroup. Four gives 128 threads, enough to keep the
/// scheduler fed without pushing register pressure.
const SIMDGROUPS_PER_GROUP = 4;

const Ctx = struct {
    dev: mtl.Device,
    q4k: mtl.Pipeline,
    /// Weight buffers are keyed by the page-aligned base of the mapping they
    /// came from, so one MTLBuffer serves every tensor in a GGUF rather than
    /// one per tensor.
    wrapped: std.AutoHashMapUnmanaged(usize, mtl.Buffer) = .empty,
    scratch_x: mtl.Buffer,
    scratch_out: mtl.Buffer,
    gpa: std.mem.Allocator,
};

var ctx: ?Ctx = null;

/// Bring up the device and compile the kernels. Falls back to CPU silently
/// only for reasons that are genuinely not errors (no Metal device on a
/// headless box); a shader that will not compile is loud.
pub fn parallelBegin(n: usize) void {
    cpu.parallelBegin(n); // the CPU kernels still run the ops Metal lacks
    if (ctx != null) return;
    const gpa = std.heap.page_allocator;
    var dev = mtl.Device.init() catch {
        std.debug.print("metal: no device available; running on CPU\n", .{});
        return;
    };
    if (!dev.hasUnifiedMemory()) {
        // wrapHost would hand the GPU host pages across PCIe.
        std.debug.print("metal: {s} has no unified memory; running on CPU\n", .{dev.name()});
        dev.deinit();
        return;
    }
    const p = dev.pipeline(dmmv_q4k_src, "dmmv_q4k") catch {
        dev.deinit();
        return;
    };
    const sx = dev.alloc(1 << 20) catch {
        dev.deinit();
        return;
    };
    const so = dev.alloc(1 << 22) catch {
        dev.deinit();
        return;
    };
    ctx = .{ .dev = dev, .q4k = p, .scratch_x = sx, .scratch_out = so, .gpa = gpa };
}

pub fn parallelEnd() void {
    cpu.parallelEnd();
    // The device is kept alive across requests: compiling shaders and creating
    // buffers per generation would cost more than it saves.
}

/// Page-aligned MTLBuffer covering `data`, plus the offset of `data` within
/// it. Cached per mapping so a model produces one buffer, not one per tensor.
fn wrapFor(cx: *Ctx, data: []const u8) ?struct { buf: mtl.Buffer, off: usize } {
    const page = std.heap.pageSize();
    const base = std.mem.alignBackward(usize, @intFromPtr(data.ptr), page);
    const end = std.mem.alignForward(usize, @intFromPtr(data.ptr) + data.len, page);
    const gop = cx.wrapped.getOrPut(cx.gpa, base) catch return null;
    if (!gop.found_existing) {
        const region = @as([*]const u8, @ptrFromInt(base))[0 .. end - base];
        gop.value_ptr.* = cx.dev.wrapHost(region) catch {
            _ = cx.wrapped.remove(base);
            return null;
        };
    }
    // A later tensor may extend past the cached buffer; fall back if so.
    const off = @intFromPtr(data.ptr) - base;
    if (off + data.len > gop.value_ptr.len) return null;
    return .{ .buf = gop.value_ptr.*, .off = off };
}

const Dims = extern struct { rows: u32, cols: u32 };

pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const cx = &(ctx orelse return cpu.matvec(t, out, data, x, rows, cols));
    if (t != .q4_k or rows < MIN_GPU_ROWS or cols % 256 != 0) {
        return cpu.matvec(t, out, data, x, rows, cols);
    }
    if (x.len * 4 > cx.scratch_x.len or out.len * 4 > cx.scratch_out.len) {
        return cpu.matvec(t, out, data, x, rows, cols);
    }
    const w = wrapFor(cx, data) orelse return cpu.matvec(t, out, data, x, rows, cols);

    @memcpy(cx.scratch_x.slice(f32)[0..x.len], x);
    const dims = Dims{ .rows = @intCast(rows), .cols = @intCast(cols) };
    // One SIMD group (32 lanes) per row. The threadgroup holds several of
    // them, so the grid is rows*32 threads.
    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const cb = cx.dev.commandBuffer();
    cb.dispatch(
        cx.q4k,
        &.{ w.buf, cx.scratch_x, cx.scratch_out },
        &.{ w.off, 0, 0 },
        std.mem.asBytes(&dims),
        rows * SIMD_W,
        group,
    );
    cb.commitAndWait();
    @memcpy(out, cx.scratch_out.slice(f32)[0..out.len]);
}

test "metal q4_k matvec agrees with the exact cpu reference" {
    // The oracle has to be the *exact* path, not `cpu.matvec`. The CPU Q4_K
    // kernel quantizes activations to int8 and is approximate by design (~0.4%
    // per element); the Metal kernel keeps activations in f32. Comparing the
    // two directly reports the CPU's approximation as a GPU bug -- which is
    // what the first version of this test did, for an hour.
    //
    // So: dequantize-then-dot in f32 is the reference, and the tolerance is
    // bounded against the mass of the terms summed rather than the result. A
    // dot of random weights against random activations cancels almost
    // completely, so the result can be orders of magnitude smaller than the
    // terms and a result-relative bound would be meaningless.
    const gpa = std.testing.allocator;
    const cols = 2048;
    const rows = MIN_GPU_ROWS;

    var prng = std.Random.DefaultPrng.init(0xA11CE);
    const rnd = prng.random();

    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB; // finite f16 scales

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest; // no Metal device on this host

    const got = try gpa.alloc(f32, rows);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    matvec(.q4_k, got, data, x, rows, cols);

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    for (0..rows) |r| {
        cpu.dequantRow(.q4_k, row, data, r, cols);
        var want: f32 = 0;
        var mass: f32 = 0;
        for (row, x) |w, xv| {
            want += w * xv;
            mass += @abs(w * xv);
        }
        // f32 summation order differs (the GPU accumulates per sub-block);
        // 1e-5 of the summed mass is well inside that and far outside a real
        // indexing error, which shifts a result by whole percent.
        const tol = mass * 1e-5;
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("metal q4_k row {d}: reference {d} vs gpu {d} (tol {d})\n", .{ r, want, got[r], tol });
            return e;
        };
    }
}

test "metal dispatch cost breakdown" {
    // Where does the time actually go? Claiming "submission latency dominates"
    // is worth nothing without the number, so measure three things:
    //   1. a commit-and-wait round trip per dispatch (what the code does now)
    //   2. many dispatches in one command buffer, committed once
    //   3. the kernel's own execution time, inferred from 2
    if (@import("builtin").mode != .ReleaseFast) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const cols = 2048;
    const rows = 5632; // a real ffn_up shape

    // 8 threads, not 1: the CPU comparison has to be the CPU path as it
    // actually ships, or the GPU looks good for the wrong reason.
    parallelBegin(8);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);

    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) - 0.5;
    @memcpy(cx.scratch_x.slice(f32)[0..cols], x);

    const w = wrapFor(cx, data).?;
    const dims = Dims{ .rows = @intCast(rows), .cols = @intCast(cols) };
    const group = @min(cx.q4k.maxThreads(), 64);
    const N = 50;

    var t: std.Io.Threaded = .init(gpa, .{});
    defer t.deinit();
    const io = t.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;

    // 1. one commit+wait per dispatch
    const t0 = now(io);
    for (0..N) |_| {
        const cb = cx.dev.commandBuffer();
        cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), rows, group);
        cb.commitAndWait();
    }
    const per_sync = @divTrunc(now(io) - t0, N);

    // 2. N dispatches, one commit
    const t1 = now(io);
    {
        const cb = cx.dev.commandBuffer();
        for (0..N) |_| {
            cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), rows, group);
        }
        cb.commitAndWait();
    }
    const per_batched = @divTrunc(now(io) - t1, N);

    // 3. the CPU kernel, for scale
    const out = try gpa.alloc(f32, rows);
    defer gpa.free(out);
    const t2 = now(io);
    for (0..N) |_| cpu.matvec(.q4_k, out, data, x, rows, cols);
    const per_cpu = @divTrunc(now(io) - t2, N);

    const ms = struct {
        fn f(ns: i128) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    std.debug.print(
        \\
        \\  one matvec, 2048x5632 Q4_K:
        \\    GPU, commit+wait each   {d:6.3} ms   <- what the code does now
        \\    GPU, batched in one cb  {d:6.3} ms   <- kernel + amortized submit
        \\    submission overhead     {d:6.3} ms   <- the difference
        \\    CPU, 8 threads          {d:6.3} ms   <- what it is beating
        \\
    , .{ ms(per_sync), ms(per_batched), ms(per_sync - per_batched), ms(per_cpu) });
}
