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
const elementwise_src = @embedFile("../shaders/metal/elementwise.metal");

/// Below this many rows the CPU path wins outright, so the GPU is not used.
///
/// The threshold is measured, and the measurement corrected an earlier and
/// much more optimistic one. Sweeping the shape on an M5, one dispatch per
/// command buffer (which is what a dependency chain forces):
///
///     rows      gpu       cpu(8t)   ratio
///     2048      0.383 ms  0.066 ms   0.17x
///     5632      0.522     0.132      0.25x
///     32000     1.374     0.723      0.53x
///     65536     1.685     1.441      0.86x
///     131072    1.808     2.765      1.53x
///
/// So the GPU is roughly 1.9x faster *per row* and carries ~0.36 ms of fixed
/// cost, and it does not overtake eight CPU threads until about 100k rows.
///
/// An earlier version of this constant was 18,000, derived from a kernel
/// measured at 8x the CPU. That figure came from fifty identical dispatches
/// issued back to back with no barriers, where the GPU overlaps them and the
/// number is throughput. A forward pass is a dependency chain and gets
/// latency, not throughput. The lesson is in the constant: benchmark the
/// shape the code actually runs.
///
/// For a 1.1B model nothing clears this bar -- the largest tensor is the
/// 32000-row output head -- so Metal correctly declines every matvec and the
/// build matches the CPU rather than losing to it. That changes with larger
/// models, and with batched prefill, where each dispatch carries far more
/// work. It also changes if the kernel improves: one SIMD group per row with
/// a scalar inner loop is a long way from what the hardware can do.
const MIN_GPU_ROWS = 100_000;

/// Apple GPUs execute in 32-lane SIMD groups; the kernel assigns one per row.
const SIMD_W = 32;
/// SIMD groups per threadgroup. Four gives 128 threads, enough to keep the
/// scheduler fed without pushing register pressure.
const SIMDGROUPS_PER_GROUP = 4;

const Ctx = struct {
    dev: mtl.Device,
    q4k: mtl.Pipeline,
    rmsnorm_p: mtl.Pipeline,
    swiglu_p: mtl.Pipeline,
    add_p: mtl.Pipeline,
    /// Device-resident activation scratch. The whole point of keeping these
    /// on the GPU is that a block of work can be encoded into one command
    /// buffer without the host reading anything in between.
    act: [4]mtl.Buffer,
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
    const pn = dev.pipeline(elementwise_src, "rmsnorm") catch {
        dev.deinit();
        return;
    };
    const ps = dev.pipeline(elementwise_src, "swiglu") catch {
        dev.deinit();
        return;
    };
    const pa = dev.pipeline(elementwise_src, "add_inplace") catch {
        dev.deinit();
        return;
    };
    var act: [4]mtl.Buffer = undefined;
    for (&act) |*b| {
        b.* = dev.alloc(1 << 22) catch {
            dev.deinit();
            return;
        };
    }
    const sx = dev.alloc(1 << 20) catch {
        dev.deinit();
        return;
    };
    const so = dev.alloc(1 << 22) catch {
        dev.deinit();
        return;
    };
    ctx = .{ .dev = dev, .q4k = p, .rmsnorm_p = pn, .swiglu_p = ps, .add_p = pa, .act = act, .scratch_x = sx, .scratch_out = so, .gpa = gpa };
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
    const off = @intFromPtr(data.ptr) - base;

    // A cached entry can be stale in two ways, and both are real rather than
    // theoretical: a later tensor in the same mapping may extend past the
    // region that was wrapped first, and an allocation freed and replaced by
    // a larger one can land on the same page base. Keying by address alone
    // therefore is not enough — re-wrap whenever the cached buffer does not
    // cover what is being asked for, rather than silently handing back a
    // buffer that is too short.
    if (gop.found_existing and off + data.len > gop.value_ptr.len) {
        gop.value_ptr.deinit();
        gop.key_ptr.* = base;
        gop.value_ptr.* = cx.dev.wrapHost(@as([*]const u8, @ptrFromInt(base))[0 .. end - base]) catch {
            _ = cx.wrapped.remove(base);
            return null;
        };
        return .{ .buf = gop.value_ptr.*, .off = off };
    }
    if (!gop.found_existing) {
        const region = @as([*]const u8, @ptrFromInt(base))[0 .. end - base];
        gop.value_ptr.* = cx.dev.wrapHost(region) catch {
            _ = cx.wrapped.remove(base);
            return null;
        };
    }
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

test "metal submission floor" {
    // Where exactly does the 0.358 ms go? An empty command buffer isolates
    // the commit+wait floor from anything the kernel or encoder costs. If the
    // floor is small, the overhead is encoder churn and is fixable cheaply; if
    // it is the whole 0.358 ms, only batching whole layers will help.
    if (@import("builtin").mode != .ReleaseFast) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    parallelBegin(1);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);

    var t: std.Io.Threaded = .init(gpa, .{});
    defer t.deinit();
    const io = t.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;
    const N = 200;

    // 1. empty command buffer: create, commit, wait
    const t0 = now(io);
    for (0..N) |_| cx.dev.commandBuffer().commitAndWait();
    const empty = @divTrunc(now(io) - t0, N);

    // 2. one trivial dispatch (1 row => the kernel does almost nothing)
    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), 16384);
    defer gpa.free(data);
    @memset(data, 0);
    const w = wrapFor(cx, data).?;
    const dims = Dims{ .rows = 1, .cols = 256 };
    const t1 = now(io);
    for (0..N) |_| {
        const cb = cx.dev.commandBuffer();
        cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), SIMD_W, SIMD_W);
        cb.commitAndWait();
    }
    const one = @divTrunc(now(io) - t1, N);

    // 3. ten dispatches, one command buffer (ten encoders, as the shim does)
    const t2 = now(io);
    for (0..N / 10) |_| {
        const cb = cx.dev.commandBuffer();
        for (0..10) |_| cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), SIMD_W, SIMD_W);
        cb.commitAndWait();
    }
    const ten = @divTrunc(now(io) - t2, N / 10);

    const us = struct {
        fn f(ns: i128) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e3;
        }
    }.f;
    std.debug.print(
        \\
        \\  empty commit+wait          {d:8.1} us   <- the irreducible floor
        \\  + 1 trivial dispatch       {d:8.1} us
        \\  10 dispatches, 1 buffer    {d:8.1} us  ({d:6.1} us each)
        \\
    , .{ us(empty), us(one), us(ten), us(ten) / 10 });
}

/// A whole FFN block in one command buffer: rmsnorm, gate and up projections,
/// SwiGLU, the down projection, and the residual add — with every
/// intermediate staying in device memory.
///
/// This is the shape the rest of the layer has to take. Measured on an M5, a
/// command buffer costs ~262 us fixed no matter how many dispatches it holds
/// (ten cost the same as one), while a matvec kernel is ~18 us. So the unit of
/// work that matters is not the kernel, it is the buffer: anything that forces
/// the host to read an intermediate splits one buffer into two and adds 262 us.
///
/// Returns false if the shapes do not fit the scratch buffers, in which case
/// the caller runs the CPU path.
pub fn ffnBlock(
    x: []f32, // in/out: residual stream, updated in place
    norm_w: []const f32,
    eps: f32,
    gate_w: WeightRef,
    up_w: WeightRef,
    down_w: WeightRef,
    ffn: usize,
) bool {
    const cx = &(ctx orelse return false);
    const dim = x.len;
    if (gate_w.ty != .q4_k or up_w.ty != .q4_k or down_w.ty != .q4_k) return false;
    if (dim % 256 != 0 or ffn % 256 != 0) return false;
    if (ffn * 4 > cx.act[0].len or dim * 4 > cx.act[0].len) return false;

    const gw = wrapFor(cx, gate_w.data) orelse return false;
    const uw = wrapFor(cx, up_w.data) orelse return false;
    const dw = wrapFor(cx, down_w.data) orelse return false;

    // a0 = x (residual), a1 = normed, a2 = gate then act, a3 = up then ffn_out
    @memcpy(cx.act[0].slice(f32)[0..dim], x);
    @memcpy(cx.scratch_x.slice(f32)[0..dim], norm_w);

    const nd = extern struct { n: u32, eps: f32 }{ .n = @intCast(dim), .eps = eps };
    const dims_ffn = Dims{ .rows = @intCast(ffn), .cols = @intCast(dim) };
    const dims_down = Dims{ .rows = @intCast(dim), .cols = @intCast(ffn) };
    const len_ffn = extern struct { n: u32 }{ .n = @intCast(ffn) };
    const len_dim = extern struct { n: u32 }{ .n = @intCast(dim) };
    const group = SIMD_W * SIMDGROUPS_PER_GROUP;

    // One encoder for the whole block. A barrier goes only where a dispatch
    // reads what the previous one wrote; the gate and up projections are
    // independent and overlap, which is the point of opening the encoder in
    // concurrent dispatch mode.
    const cb = cx.dev.commandBuffer();
    const e = cb.encoder();
    // normed = rmsnorm(x) * norm_w
    e.dispatch(cx.rmsnorm_p, &.{ cx.act[0], cx.scratch_x, cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
    e.barrier();
    // gate = Wg . normed ; up = Wu . normed  — independent, no barrier between
    e.dispatch(cx.q4k, &.{ gw.buf, cx.act[1], cx.act[2] }, &.{ gw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), ffn * SIMD_W, group);
    e.dispatch(cx.q4k, &.{ uw.buf, cx.act[1], cx.act[3] }, &.{ uw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), ffn * SIMD_W, group);
    e.barrier();
    // act = silu(gate) * up, in place over act[2]
    e.dispatch(cx.swiglu_p, &.{ cx.act[2], cx.act[3], cx.act[2] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_ffn), ffn, 64);
    e.barrier();
    // ffn_out = Wd . act
    e.dispatch(cx.q4k, &.{ dw.buf, cx.act[2], cx.act[3] }, &.{ dw.off, 0, 0 }, std.mem.asBytes(&dims_down), dim * SIMD_W, group);
    e.barrier();
    // x += ffn_out
    e.dispatch(cx.add_p, &.{ cx.act[0], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), dim, 64);
    e.end();
    cb.commitAndWait();

    @memcpy(x, cx.act[0].slice(f32)[0..dim]);
    return true;
}

/// A weight tensor as the engines hold it: a type plus a slice of the mapping.
pub const WeightRef = struct { ty: ggml.Type, data: []const u8 };

test "resident ffn block matches the cpu path" {
    const gpa = std.testing.allocator;
    const dim = 2048;
    const ffn = 5632;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0xFF17);
    const rnd = prng.random();
    const mk = struct {
        fn f(a: std.mem.Allocator, r: std.Random, rows: usize, cols: usize) ![]align(16384) u8 {
            const d = try a.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
            r.bytes(d);
            for (0..d.len / 2) |i| d[i * 2 + 1] &= 0xFB;
            return d;
        }
    }.f;
    const gate = try mk(gpa, rnd, ffn, dim);
    defer gpa.free(gate);
    const up = try mk(gpa, rnd, ffn, dim);
    defer gpa.free(up);
    const down = try mk(gpa, rnd, dim, ffn);
    defer gpa.free(down);

    const nw = try gpa.alloc(f32, dim);
    defer gpa.free(nw);
    for (nw) |*v| v.* = 0.5 + rnd.float(f32);
    const x0 = try gpa.alloc(f32, dim);
    defer gpa.free(x0);
    for (x0) |*v| v.* = (rnd.float(f32) - 0.5) * 0.1;

    // reference: the CPU ops, in the same order
    const want = try gpa.alloc(f32, dim);
    defer gpa.free(want);
    @memcpy(want, x0);
    {
        const normed = try gpa.alloc(f32, dim);
        defer gpa.free(normed);
        const g = try gpa.alloc(f32, ffn);
        defer gpa.free(g);
        const u = try gpa.alloc(f32, ffn);
        defer gpa.free(u);
        const a = try gpa.alloc(f32, ffn);
        defer gpa.free(a);
        const o = try gpa.alloc(f32, dim);
        defer gpa.free(o);
        // Exact f32 matvecs, not cpu.matvec: the CPU Q4_K kernel quantizes
        // activations to int8, and here that error passes through SwiGLU --
        // a nonlinearity, which amplifies it wherever a gate value sits near
        // zero. Comparing against it reports the CPU's own approximation as a
        // GPU bug, compounded three matmuls deep. Same trap as the first
        // dmmv test, one layer up.
        const exactMatvec = struct {
            fn f(a2: std.mem.Allocator, dstv: []f32, data: []const u8, xv: []const f32, rows: usize, cols: usize) !void {
                const rowbuf = try a2.alloc(f32, cols);
                defer a2.free(rowbuf);
                for (0..rows) |r| {
                    cpu.dequantRow(.q4_k, rowbuf, data, r, cols);
                    var acc: f32 = 0;
                    for (rowbuf, xv) |wv, xx| acc += wv * xx;
                    dstv[r] = acc;
                }
            }
        }.f;
        cpu.rmsnorm(normed, want, nw, eps_test);
        try exactMatvec(gpa, g, gate, normed, ffn, dim);
        try exactMatvec(gpa, u, up, normed, ffn, dim);
        cpu.swiglu(a, g, u);
        try exactMatvec(gpa, o, down, a, dim, ffn);
        cpu.add(want, o);
    }

    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);
    @memcpy(got, x0);
    try std.testing.expect(ffnBlock(got, nw, eps_test, .{ .ty = .q4_k, .data = gate }, .{ .ty = .q4_k, .data = up }, .{ .ty = .q4_k, .data = down }, ffn));

    // The CPU matvec quantizes activations to int8 and the Metal one does not,
    // so this is a comparison of two different approximations of the same
    // thing; bound it against the magnitude of the values, not exactly.
    var worst: f32 = 0;
    for (want, got) |a, b| worst = @max(worst, @abs(a - b) / @max(@max(@abs(a), @abs(b)), 1e-2));
    if (worst > 0.05) {
        std.debug.print("resident ffn: worst relative difference {d}\n", .{worst});
        return error.FfnBlockMismatch;
    }
}

const eps_test: f32 = 1e-5;

test "metal elementwise kernels individually" {
    const gpa = std.testing.allocator;
    parallelBegin(1);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);
    const n = 2048;

    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, n);
    defer gpa.free(w);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2;
    for (w) |*v| v.* = 0.5 + rnd.float(f32);

    // rmsnorm
    @memcpy(cx.act[0].slice(f32)[0..n], x);
    @memcpy(cx.scratch_x.slice(f32)[0..n], w);
    const nd = extern struct { n: u32, eps: f32 }{ .n = n, .eps = eps_test };
    {
        const cb = cx.dev.commandBuffer();
        cb.dispatch(cx.rmsnorm_p, &.{ cx.act[0], cx.scratch_x, cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
        cb.commitAndWait();
    }
    const ref = try gpa.alloc(f32, n);
    defer gpa.free(ref);
    cpu.rmsnorm(ref, x, w, eps_test);
    var worst_norm: f32 = 0;
    for (ref, cx.act[1].slice(f32)[0..n]) |a, b| worst_norm = @max(worst_norm, @abs(a - b));

    // swiglu
    const up = try gpa.alloc(f32, n);
    defer gpa.free(up);
    for (up) |*v| v.* = rnd.float(f32) - 0.5;
    @memcpy(cx.act[2].slice(f32)[0..n], x);
    @memcpy(cx.act[3].slice(f32)[0..n], up);
    const ln = extern struct { n: u32 }{ .n = n };
    {
        const cb = cx.dev.commandBuffer();
        cb.dispatch(cx.swiglu_p, &.{ cx.act[2], cx.act[3], cx.act[2] }, &.{ 0, 0, 0 }, std.mem.asBytes(&ln), n, 64);
        cb.commitAndWait();
    }
    cpu.swiglu(ref, x, up);
    var worst_sw: f32 = 0;
    for (ref, cx.act[2].slice(f32)[0..n]) |a, b| worst_sw = @max(worst_sw, @abs(a - b));

    std.debug.print("  rmsnorm max abs diff {d}   swiglu {d}\n", .{ worst_norm, worst_sw });
    try std.testing.expect(worst_norm < 1e-4);
    try std.testing.expect(worst_sw < 1e-5);
}

test "resident ffn block vs the cpu ffn" {
    // The question this whole exercise turns on: does putting a block of work
    // in one command buffer beat the CPU, given a ~262 us fixed cost per
    // buffer against ~18 us kernels?
    if (@import("builtin").mode != .ReleaseFast) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const dim = 2048;
    const ffn = 5632;
    parallelBegin(8);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const mk = struct {
        fn f(a: std.mem.Allocator, r: std.Random, rows: usize, cols: usize) ![]align(16384) u8 {
            const d = try a.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
            r.bytes(d);
            for (0..d.len / 2) |i| d[i * 2 + 1] &= 0xFB;
            return d;
        }
    }.f;
    const gate = try mk(gpa, rnd, ffn, dim);
    defer gpa.free(gate);
    const up = try mk(gpa, rnd, ffn, dim);
    defer gpa.free(up);
    const down = try mk(gpa, rnd, dim, ffn);
    defer gpa.free(down);
    const nw = try gpa.alloc(f32, dim);
    defer gpa.free(nw);
    for (nw) |*v| v.* = 1.0;
    const x = try gpa.alloc(f32, dim);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 0.1;

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    const g = try gpa.alloc(f32, ffn);
    defer gpa.free(g);
    const u = try gpa.alloc(f32, ffn);
    defer gpa.free(u);
    const o = try gpa.alloc(f32, dim);
    defer gpa.free(o);

    var t: std.Io.Threaded = .init(gpa, .{});
    defer t.deinit();
    const io = t.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;
    const N = 40;

    _ = ffnBlock(x, nw, eps_test, .{ .ty = .q4_k, .data = gate }, .{ .ty = .q4_k, .data = up }, .{ .ty = .q4_k, .data = down }, ffn);
    const t0 = now(io);
    for (0..N) |_| _ = ffnBlock(x, nw, eps_test, .{ .ty = .q4_k, .data = gate }, .{ .ty = .q4_k, .data = up }, .{ .ty = .q4_k, .data = down }, ffn);
    const gpu = @divTrunc(now(io) - t0, N);

    const t1 = now(io);
    for (0..N) |_| {
        cpu.rmsnorm(normed, x, nw, eps_test);
        cpu.matvec(.q4_k, g, gate, normed, ffn, dim);
        cpu.matvec(.q4_k, u, up, normed, ffn, dim);
        cpu.swiglu(g, g, u);
        cpu.matvec(.q4_k, o, down, g, dim, ffn);
        cpu.add(x, o);
    }
    const cpu_ns = @divTrunc(now(io) - t1, N);

    const ms = struct {
        fn f(ns: i128) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    std.debug.print("  ffn block: gpu (1 command buffer) {d:.3} ms   cpu (8 threads) {d:.3} ms\n", .{ ms(gpu), ms(cpu_ns) });
}

test "metal scaling: where does the gpu actually win" {
    // A dependency chain on the GPU pays launch latency per step, and a
    // single-token decode of a small model gives each dispatch very little
    // work to amortize it with. Sweep the shape to find where the GPU starts
    // winning, rather than assuming it does.
    if (@import("builtin").mode != .ReleaseFast) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    parallelBegin(8);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);

    var t: std.Io.Threaded = .init(gpa, .{});
    defer t.deinit();
    const io = t.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;

    const cols = 2048;
    std.debug.print("\n  rows      gpu(1 cb)      cpu(8t)   ratio\n", .{});
    for ([_]usize{ 2048, 5632, 16384, 32000, 65536, 131072 }) |rows| {
        const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
        defer gpa.free(data);
        var prng = std.Random.DefaultPrng.init(7);
        prng.random().bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;
        const x = try gpa.alloc(f32, cols);
        defer gpa.free(x);
        for (x) |*v| v.* = prng.random().float(f32) - 0.5;
        const dst = try gpa.alloc(f32, rows);
        defer gpa.free(dst);
        if (rows * 4 > cx.scratch_out.len) continue;
        @memcpy(cx.scratch_x.slice(f32)[0..cols], x);
        const w = wrapFor(cx, data) orelse continue;
        const dims = Dims{ .rows = @intCast(rows), .cols = @intCast(cols) };
        const group = SIMD_W * SIMDGROUPS_PER_GROUP;

        const N = 20;
        // warm
        {
            const cb = cx.dev.commandBuffer();
            cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), rows * SIMD_W, group);
            cb.commitAndWait();
        }
        const t0 = now(io);
        for (0..N) |_| {
            const cb = cx.dev.commandBuffer();
            cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), rows * SIMD_W, group);
            cb.commitAndWait();
        }
        const gpu = @divTrunc(now(io) - t0, N);

        cpu.matvec(.q4_k, dst, data, x, rows, cols);
        const t1 = now(io);
        for (0..N) |_| cpu.matvec(.q4_k, dst, data, x, rows, cols);
        const cpu_ns = @divTrunc(now(io) - t1, N);

        const g: f64 = @as(f64, @floatFromInt(gpu)) / 1e6;
        const c2: f64 = @as(f64, @floatFromInt(cpu_ns)) / 1e6;
        std.debug.print("  {d:>7}   {d:>8.3} ms   {d:>8.3} ms   {d:>5.2}x\n", .{ rows, g, c2, c2 / g });
    }
}
