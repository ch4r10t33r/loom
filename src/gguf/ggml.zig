//! GGML tensor formats — the quantizations GGUF files actually ship.
//!
//! Two families, decoded differently:
//!
//!   *affine*    F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K. A value is
//!               reconstructed arithmetically (`scale * q + min`), so the
//!               kernel is a mask, a shift and a multiply-add. Implemented
//!               here.
//!   *codebook*  the IQ quants and MXFP4. A block stores an index into a
//!               static grid table plus packed signs; values come out of a
//!               lookup. Implemented in iq.zig against tables transcribed
//!               from llama.cpp (iq_tables.zig).
//!
//! Each type gets a fused matvec over the raw tensor bytes so weights are
//! never dequantized wholesale. Note these differ from loom's own q4 expert
//! format (quant.zig, f32 scales): here we implement GGML's layouts exactly.

const std = @import("std");
const builtin = @import("builtin");
const iq = @import("iq.zig");
const iq_tables = @import("iq_tables.zig");

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    iq1_m = 29,
    mxfp4 = 39,
    _,

    pub fn supported(t: u32) bool {
        return switch (@as(Type, @enumFromInt(t))) {
            .f32,
            .f16,
            .q4_0,
            .q4_1,
            .q5_0,
            .q5_1,
            .q8_0,
            .q2_k,
            .q3_k,
            .q4_k,
            .q5_k,
            .q6_k,
            .iq2_xxs,
            .iq2_xs,
            .iq3_xxs,
            .iq1_s,
            .iq4_nl,
            .iq3_s,
            .iq2_s,
            .iq4_xs,
            .iq1_m,
            .mxfp4,
            => true,
            _ => false,
        };
    }

    /// True for the codebook (IQ / MXFP4) family, which decodes through
    /// iq.zig rather than an affine formula.
    pub fn isCodebook(self: Type) bool {
        return switch (self) {
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m, .mxfp4 => true,
            else => false,
        };
    }
};

pub const QK_0: usize = 32; // block width for q4_0 / q8_0
const Q4_0_BLOCK: usize = 2 + QK_0 / 2; // f16 scale + 16 nibble bytes = 18
const Q4_1_BLOCK: usize = 2 + 2 + QK_0 / 2; // f16 scale + f16 min + nibbles = 20
const Q5_0_BLOCK: usize = 2 + 4 + QK_0 / 2; // f16 scale + 32 high bits + nibbles = 22
const Q5_1_BLOCK: usize = 2 + 2 + 4 + QK_0 / 2; // + f16 min = 24
const Q8_0_BLOCK: usize = 2 + QK_0; // f16 scale + 32 int8 = 34

pub const QK_K: usize = 256; // super-block width for K-quants
const Q2_K_BLOCK: usize = QK_K / 16 + QK_K / 4 + 2 + 2; // scales, qs, d, dmin = 84
const Q3_K_BLOCK: usize = QK_K / 8 + QK_K / 4 + 12 + 2; // hmask, qs, scales, d = 110
const Q4_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 2; // d, dmin, 6-bit scales, nibbles = 144
const Q5_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // + high bits = 176
const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql, qh, scales, d = 210

/// Bytes of one row of `n` elements in format `t`. `n` must be a multiple of
/// the block width for quantized types.
pub fn rowBytes(t: Type, n: usize) usize {
    return switch (t) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q4_0 => (n / QK_0) * Q4_0_BLOCK,
        .q4_1 => (n / QK_0) * Q4_1_BLOCK,
        .q5_0 => (n / QK_0) * Q5_0_BLOCK,
        .q5_1 => (n / QK_0) * Q5_1_BLOCK,
        .q8_0 => (n / QK_0) * Q8_0_BLOCK,
        .q2_k => (n / QK_K) * Q2_K_BLOCK,
        .q3_k => (n / QK_K) * Q3_K_BLOCK,
        .q4_k => (n / QK_K) * Q4_K_BLOCK,
        .q5_k => (n / QK_K) * Q5_K_BLOCK,
        .q6_k => (n / QK_K) * Q6_K_BLOCK,
        else => (n / blockElems(t)) * blockBytes(t), // codebook types
    };
}

pub fn tensorBytes(t: Type, ne0: usize, rows: usize) usize {
    return rows * rowBytes(t, ne0);
}

/// Block size in elements for `t` (1 for unquantized types). A quantized row
/// length must be a whole number of blocks: `rowBytes` divides, so a ragged
/// ne0 silently under-counts the bytes a kernel will actually walk.
pub fn blockElems(t: Type) usize {
    return switch (t) {
        .f32, .f16 => 1,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0 => QK_0,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => QK_K,
        // codebook types: iq4_nl and mxfp4 are 32-wide, the rest are 256-wide
        .iq4_nl, .mxfp4 => iq.QK_NL,
        .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => iq.QK_K,
        _ => 1,
    };
}

/// Overflow-checked `tensorBytes` for sizes read from an untrusted file
/// (security issue #29). Also rejects a row length that is not a whole number
/// of quantization blocks.
pub fn tensorBytesChecked(t: Type, ne0: usize, rows: usize, ne2: usize) !usize {
    const blk = blockElems(t);
    if (blk != 1 and ne0 % blk != 0) return error.BadTensorShape;
    const per_row = switch (t) {
        .f32 => try std.math.mul(usize, ne0, 4),
        .f16 => try std.math.mul(usize, ne0, 2),
        else => try std.math.mul(usize, ne0 / blk, blockBytes(t)),
    };
    const per_slice = try std.math.mul(usize, rows, per_row);
    return std.math.mul(usize, per_slice, ne2);
}

/// Bytes per quantization block for `t`.
pub fn blockBytes(t: Type) usize {
    return switch (t) {
        .q4_0 => Q4_0_BLOCK,
        .q4_1 => Q4_1_BLOCK,
        .q5_0 => Q5_0_BLOCK,
        .q5_1 => Q5_1_BLOCK,
        .q8_0 => Q8_0_BLOCK,
        .q2_k => Q2_K_BLOCK,
        .q3_k => Q3_K_BLOCK,
        .q4_k => Q4_K_BLOCK,
        .q5_k => Q5_K_BLOCK,
        .q6_k => Q6_K_BLOCK,
        .iq2_xxs => iq.IQ2_XXS_BLOCK,
        .iq2_xs => iq.IQ2_XS_BLOCK,
        .iq2_s => iq.IQ2_S_BLOCK,
        .iq3_xxs => iq.IQ3_XXS_BLOCK,
        .iq3_s => iq.IQ3_S_BLOCK,
        .iq1_s => iq.IQ1_S_BLOCK,
        .iq1_m => iq.IQ1_M_BLOCK,
        .iq4_xs => iq.IQ4_XS_BLOCK,
        .iq4_nl => iq.IQ4_NL_BLOCK,
        .mxfp4 => iq.MXFP4_BLOCK,
        else => 0,
    };
}

// ---- SIMD helpers (issue #11) ------------------------------------------------

/// Lanes per vector op. 8 f32 is 32 bytes: two NEON registers on aarch64, one
/// AVX register on x86-64, and it lets the backend keep several accumulators
/// in flight. Zig lowers @Vector to whatever the target actually has, so this
/// stays portable rather than becoming per-arch intrinsics.
const LANES = 8;
const Vf = @Vector(LANES, f32);

/// Dot product of two equal-length f32 slices.
///
/// The scalar version was a serial dependency chain: every `acc += a*b` waits
/// on the previous add, so the FPU stalls on latency rather than running at
/// throughput. Four independent accumulators plus vector lanes break that
/// chain, which is most of the win here -- more than the lane count alone
/// suggests.
pub inline fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc0: Vf = @splat(0);
    var acc1: Vf = @splat(0);
    var acc2: Vf = @splat(0);
    var acc3: Vf = @splat(0);
    var i: usize = 0;
    while (i + 4 * LANES <= a.len) : (i += 4 * LANES) {
        acc0 += @as(Vf, a[i..][0..LANES].*) * @as(Vf, b[i..][0..LANES].*);
        acc1 += @as(Vf, a[i + LANES ..][0..LANES].*) * @as(Vf, b[i + LANES ..][0..LANES].*);
        acc2 += @as(Vf, a[i + 2 * LANES ..][0..LANES].*) * @as(Vf, b[i + 2 * LANES ..][0..LANES].*);
        acc3 += @as(Vf, a[i + 3 * LANES ..][0..LANES].*) * @as(Vf, b[i + 3 * LANES ..][0..LANES].*);
    }
    while (i + LANES <= a.len) : (i += LANES) {
        acc0 += @as(Vf, a[i..][0..LANES].*) * @as(Vf, b[i..][0..LANES].*);
    }
    var acc = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < a.len) : (i += 1) acc += a[i] * b[i];
    return acc;
}

/// out += w * v, vectorized. The attention value-accumulation step, which runs
/// once per cached position per head per layer -- scalar, it dominated prefill
/// once the projections were batched.
pub inline fn axpy(out: []f32, v: []const f32, w: f32) void {
    std.debug.assert(out.len == v.len);
    const vw: Vf = @splat(w);
    var i: usize = 0;
    while (i + LANES <= out.len) : (i += LANES) {
        const o: Vf = out[i..][0..LANES].*;
        out[i..][0..LANES].* = o + vw * @as(Vf, v[i..][0..LANES].*);
    }
    while (i < out.len) : (i += 1) out[i] += w * v[i];
}

/// Sum of an f32 slice (the `m * sum(x)` term the affine "_1" kernels hoist).
inline fn sumF32(a: []const f32) f32 {
    var acc0: Vf = @splat(0);
    var i: usize = 0;
    while (i + LANES <= a.len) : (i += LANES) acc0 += @as(Vf, a[i..][0..LANES].*);
    var acc = @reduce(.Add, acc0);
    while (i < a.len) : (i += 1) acc += a[i];
    return acc;
}

/// Low / high nibbles of 8 packed bytes, widened to f32 vectors. The K-quant
/// dequant loops are almost entirely this operation.
inline fn loNib(b: *const [LANES]u8) Vf {
    const v: @Vector(LANES, u8) = b.*;
    return @floatFromInt(v & @as(@Vector(LANES, u8), @splat(0x0F)));
}
inline fn hiNib(b: *const [LANES]u8) Vf {
    const v: @Vector(LANES, u8) = b.*;
    return @floatFromInt(v >> @as(@Vector(LANES, u3), @splat(4)));
}

/// Widen 8 f16 bit patterns to an f32 vector in one step.
inline fn f16x8(bits: *const [LANES]u16) Vf {
    const h: @Vector(LANES, f16) = @bitCast(@as(@Vector(LANES, u16), bits.*));
    return @floatCast(h);
}

inline fn f16FromBytes(b: []const u8) f32 {
    const bits = std.mem.readInt(u16, b[0..2], .little);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ---- row-parallel matvec (issue #11) -----------------------------------------

/// A matvec splits cleanly across rows: each row writes one `out` element and
/// reads shared, immutable weight bytes and input vector, so workers never
/// touch the same memory. Splitting by row therefore needs no locking and no
/// changes to the kernels -- each worker calls the same kernel on a sub-slice.
///
/// The pool is explicitly started and stopped around a generation rather than
/// living for the process lifetime, because without a futex or condition
/// variable (Zig 0.16 moved those under `Io`, which the kernels deliberately
/// do not depend on) idle workers would have to spin. Spawning ~10 threads
/// costs tens of microseconds against a generation lasting seconds, so tying
/// their lifetime to a request is both simpler and cheaper than keeping them
/// parked.
const MAX_WORKERS = 32;

/// Rows below this are not worth distributing: the hand-off costs more than
/// the work. Norm and router projections are tiny and frequent.
const MIN_ROWS_PER_THREAD = 64;

/// Spin iterations before falling back to yielding. A matvec is tens of
/// microseconds, so the interesting waits are far shorter than a context
/// switch; yielding immediately turned an 8x parallel win into 1.1x.
const SPIN_BEFORE_YIELD: u32 = 40_000;

const Pool = struct {
    n: usize = 0, // live workers, 0 = run inline
    threads: [MAX_WORKERS]std.Thread = undefined,
    quit: std.atomic.Value(bool) = .init(false),
    /// Bumped once per job; a worker runs when it sees a value it has not run.
    seq: std.atomic.Value(u64) = .init(0),
    /// Workers still inside the current job.
    active: std.atomic.Value(usize) = .init(0),
    /// Next unclaimed row block.
    cursor: std.atomic.Value(usize) = .init(0),

    // job description, written before `seq` is published and read after
    job_ty: Type = .f32,
    job_out: []f32 = &.{},
    job_data: []const u8 = &.{},
    job_x: []const f32 = &.{},
    job_rows: usize = 0,
    job_cols: usize = 0,
    job_rb: usize = 0,
    job_chunk: usize = 0,
    /// >1 selects the batched kernels, which read activations from qxn_buf.
    job_n: usize = 1,
};

var pool: Pool = .{};

/// Start `n` worker threads (0 or 1 disables parallelism). Safe to call when
/// already started: it is a no-op. Not thread-safe against itself -- call it
/// from the thread that owns the generation.
pub fn parallelBegin(n: usize) void {
    if (pool.n != 0) return;
    const want = @min(n, MAX_WORKERS);
    if (want <= 1) return;
    pool.quit.store(false, .monotonic);
    pool.seq.store(0, .monotonic);
    pool.active.store(0, .monotonic);
    var started: usize = 0;
    // `want` includes the calling thread, so spawn one fewer.
    while (started + 1 < want) : (started += 1) {
        pool.threads[started] = std.Thread.spawn(.{}, worker, .{started}) catch break;
    }
    pool.n = started;
}

pub fn parallelEnd() void {
    if (pool.n == 0) return;
    pool.quit.store(true, .release);
    // wake workers out of their spin so they observe `quit`
    _ = pool.seq.fetchAdd(1, .release);
    for (pool.threads[0..pool.n]) |t| t.join();
    pool.n = 0;
}

fn worker(_: usize) void {
    var seen: u64 = 0;
    var idle: u32 = 0;
    while (true) {
        const s = pool.seq.load(.acquire);
        if (s == seen) {
            if (pool.quit.load(.acquire)) return;
            // Gaps between matvecs during a generation are microseconds, and a
            // yield costs a context switch -- far more than the wait. Spin
            // first, and only fall back to yielding once the pause looks long
            // (between requests), so an idle pool does not burn cores.
            idle += 1;
            if (idle < SPIN_BEFORE_YIELD) {
                std.atomic.spinLoopHint();
            } else {
                std.Thread.yield() catch {};
            }
            continue;
        }
        idle = 0;
        seen = s;
        if (pool.quit.load(.acquire)) return;
        runChunks();
        _ = pool.active.fetchSub(1, .release);
    }
}

/// Claim and run row blocks until the job is exhausted. Shared by the workers
/// and the submitting thread, so the submitter contributes work instead of
/// idling.
fn runChunks() void {
    // Batched jobs read activations from this thread's own qxn_buf, so fill it
    // before claiming any work -- once per worker per job, not per chunk.
    if (pool.job_n > 1) {
        if (usesQ8KActivations(pool.job_ty))
            quantizeXN256(pool.job_x, pool.job_n, pool.job_cols)
        else
            quantizeXN(pool.job_x, pool.job_n, pool.job_cols);
    }
    while (true) {
        const start = pool.cursor.fetchAdd(pool.job_chunk, .acq_rel);
        if (start >= pool.job_rows) return;
        const end = @min(start + pool.job_chunk, pool.job_rows);
        if (pool.job_n > 1) {
            matmulRows(pool.job_ty, pool.job_out, pool.job_data, pool.job_n, pool.job_rows, pool.job_cols, start, end);
            continue;
        }
        dispatch(
            pool.job_ty,
            pool.job_out[start..end],
            pool.job_data[start * pool.job_rb .. end * pool.job_rb],
            pool.job_x,
            end - start,
            pool.job_cols,
        );
    }
}

// ---- batched matmul (prefill) ------------------------------------------------
//
// Decoding one token at a time re-reads and re-unpacks every weight for every
// token. During prefill the whole prompt is known up front, so the same
// unpacked weight can serve several tokens: unpack once, dot N times.
//
// That matters because the unpack, not the dot, is the expensive half of a
// quantized kernel. Measured on a 2048x5632 Q4_K tensor, Apple M5:
//
//     N=1   0.90 ms/token   (the single-vector kernel)
//     N=4   0.57 ms/token   1.6x
//     N=8   0.37 ms/token   2.4x
//     N=16  0.39 ms/token   2.3x   <- register pressure, past the sweet spot
//
// so the batch is capped at 8.

pub const MAX_BATCH = 8;

/// Activations for a batch, grouped by block so all N tokens' lanes for one
/// weight sub-block sit adjacent: the inner loop unpacks a sub-block once and
/// then walks N contiguous activation blocks.
///
/// Thread-local, and each worker fills its own copy at the start of a job. The
/// alternative -- quantize once on the submitting thread and share -- is what
/// this was first written as, and it silently produced wrong results for every
/// row a worker handled, because a worker's thread-local buffer was never
/// filled. Re-quantizing costs O(n*cols) against a chunk's O(rows*cols), under
/// 2% here, which is a cheap price for not sharing mutable state across
/// threads at all.
threadlocal var qxn_buf: [MAX_QX_BLOCKS][MAX_BATCH]XBlock = undefined;

/// Batched Q8_K activations: `qxkn_buf[block][lane]`, numerically identical
/// per lane to `quantizeX256` so the batched kernels stay bit-identical to
/// their matvecs.
threadlocal var qxkn_buf: [MAX_QXK_BLOCKS][MAX_BATCH]XBlockK = undefined;

fn quantizeXN256(xs: []const f32, n: usize, cols: usize) void {
    const nb = cols / QK_K;
    std.debug.assert(nb <= MAX_QXK_BLOCKS and n <= MAX_BATCH);
    for (0..n) |k| {
        const x = xs[k * cols ..][0..cols];
        for (0..nb) |b| {
            const src = x[b * QK_K ..][0..QK_K];
            var amax: f32 = 0;
            for (src) |v| if (std.math.isFinite(v)) {
                amax = @max(amax, @abs(v));
            };
            const d = amax / 127.0;
            const inv: f32 = if (d != 0 and std.math.isFinite(d)) 1.0 / d else 0;
            const dst = &qxkn_buf[b][k];
            for (0..QK_K / 16) |s| {
                var sum: i32 = 0;
                for (0..16) |i| {
                    const q = toI8Saturating(src[s * 16 + i] * inv);
                    dst.q[s * 16 + i] = q;
                    sum += q;
                }
                dst.bsum[s] = sum;
            }
            dst.d = if (std.math.isFinite(d)) d else 0;
        }
    }
}

/// Which types' batched kernels read the 256-wide `qxkn_buf` instead of the
/// 32-wide `qxn_buf`; drives quantizer selection at the two matmul entries.
fn usesQ8KActivations(t: Type) bool {
    return switch (t) {
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

fn quantizeXN(xs: []const f32, n: usize, cols: usize) void {
    const nb = cols / QK_0;
    std.debug.assert(nb <= MAX_QX_BLOCKS and n <= MAX_BATCH);
    for (0..n) |k| {
        const x = xs[k * cols ..][0..cols];
        for (0..nb) |b| {
            const src = x[b * QK_0 ..][0..QK_0];
            var amax: f32 = 0;
            for (src) |v| if (std.math.isFinite(v)) {
                amax = @max(amax, @abs(v));
            };
            const d = amax / 127.0;
            const inv: f32 = if (d != 0 and std.math.isFinite(d)) 1.0 / d else 0;
            var sum: i32 = 0;
            for (src, 0..) |v, i| {
                const q = toI8Saturating(v * inv);
                qxn_buf[b][k].q[i] = q;
                sum += q;
            }
            qxn_buf[b][k].d = if (std.math.isFinite(d)) d else 0;
            qxn_buf[b][k].sum = sum;
        }
    }
}

/// out[k*rows + r] = dot(W[r], xs[k*cols ..]) for k in 0..n.
///
/// Bit-identical to calling `matvec` n times, including on the int8 paths:
/// the activation quantization is per-vector and deterministic, so batching
/// changes only the order weights are unpacked in, never a value.
pub fn matmul(t: Type, out: []f32, data: []const u8, xs: []const f32, n: usize, rows: usize, cols: usize) void {
    std.debug.assert(out.len == n * rows and xs.len == n * cols);
    if (n == 0) return;
    if (n == 1) return matvec(t, out, data, xs, rows, cols);

    // Which types have a *batched* kernel, which is a narrower set than the
    // ones with an int8-activation matvec: adding a type to the single-vector
    // path does not give it a matmul kernel. Claiming otherwise reaches an
    // `unreachable` in matmulRows -- caught by a Debug test here, but silent
    // memory corruption in ReleaseFast, where unreachable is undefined
    // behaviour rather than a panic.
    const batched = switch (t) {
        // q5_0 is deliberately absent. A batched kernel for it is written
        // below but cannot join this set: `matvecQ50` dequantizes exactly in
        // f32 while every batched kernel here quantizes activations to int8,
        // so the two disagree in the last bits and `matmul` must equal `n`
        // matvecs exactly. Closing that needs either an f32-exact batched path
        // -- which wants the raw activations that `matmulRows` is not given --
        // or moving single-vector q5_0 onto int8 activations, which changes
        // decode numerics for every model that uses it. Until then q5_0 falls
        // back to one matvec per row, so a MoE layer whose `ffn_down_exps` is
        // q5_0 gets no amortization from batching.
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_0 => true,
        else => false,
    };
    if (!batched or n > MAX_BATCH) {
        for (0..n) |k| matvec(t, out[k * rows ..][0..rows], data, xs[k * cols ..][0..cols], rows, cols);
        return;
    }

    const workers = pool.n;
    if (workers == 0 or rows < 2 * MIN_ROWS_PER_THREAD) {
        if (usesQ8KActivations(t)) quantizeXN256(xs, n, cols) else quantizeXN(xs, n, cols);
        return matmulRows(t, out, data, n, rows, cols, 0, rows);
    }
    // Reuse the matvec pool by describing the job in the same fields; the
    // batch variant is distinguished by job_n > 1.
    const rb = rowBytes(t, cols);
    const per = @max(MIN_ROWS_PER_THREAD, (rows + (workers + 1) * 2 - 1) / ((workers + 1) * 2));
    pool.job_ty = t;
    pool.job_out = out;
    pool.job_data = data;
    pool.job_x = xs;
    pool.job_rows = rows;
    pool.job_cols = cols;
    pool.job_rb = rb;
    pool.job_chunk = per;
    pool.job_n = n;
    pool.cursor.store(0, .monotonic);
    pool.active.store(workers, .monotonic);
    _ = pool.seq.fetchAdd(1, .release);
    runChunks();
    var spins: u32 = 0;
    while (pool.active.load(.acquire) != 0) {
        spins += 1;
        if (spins < SPIN_BEFORE_YIELD) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
    }
    pool.job_n = 1;
}

/// Batched kernels for rows [r0, r1). Activations must already be in
/// `qxn_buf`; `out` is the whole [n*rows] destination.
fn matmulRows(t: Type, out: []f32, data: []const u8, n: usize, rows: usize, cols: usize, r0: usize, r1: usize) void {
    switch (t) {
        .q2_k => matmulK256(.q2_k, out, data, n, rows, cols, r0, r1),
        .q3_k => matmulK256(.q3_k, out, data, n, rows, cols, r0, r1),
        .q4_k => matmulK256(.q4_k, out, data, n, rows, cols, r0, r1),
        .q5_k => matmulK256(.q5_k, out, data, n, rows, cols, r0, r1),
        .q6_k => matmulK256(.q6_k, out, data, n, rows, cols, r0, r1),
        .q8_0 => matmulQ80(out, data, n, rows, cols, r0, r1),
        else => unreachable,
    }
}

fn matmulQ80(out: []f32, data: []const u8, n: usize, rows: usize, cols: usize, r0: usize, r1: usize) void {
    const bpr = cols / QK_0;
    const rb = bpr * Q8_0_BLOCK;
    for (r0..r1) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: [MAX_BATCH]f32 = @splat(0);
        for (0..bpr) |b| {
            const block = row[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
            const d = f16FromBytes(block[0..2]);
            const W: @Vector(QK_0, i16) = @as(@Vector(QK_0, i8), @bitCast(block[2..][0..QK_0].*));
            const blk = &qxn_buf[b];
            for (0..n) |k| {
                const xb = &blk[k];
                const X: @Vector(QK_0, i16) = @as(@Vector(QK_0, i8), xb.q);
                const dot = @reduce(.Add, @as(@Vector(QK_0, i32), W * X));
                acc[k] += d * xb.d * @as(f32, @floatFromInt(dot));
            }
        }
        for (0..n) |k| out[k * rows + r] = acc[k];
    }
}

/// out[r] = sum_c W[r][c] * x[c] over a row-major tensor of `rows` rows and
/// `cols` columns stored in format `t` at `data`.
pub fn matvec(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len == rows and x.len == cols);
    const workers = pool.n;
    if (workers == 0 or rows < 2 * MIN_ROWS_PER_THREAD) {
        return dispatch(t, out, data, x, rows, cols);
    }

    // Split into more blocks than threads so a slow core does not hold up the
    // whole matvec: workers claim the next block when they finish one.
    const rb = rowBytes(t, cols);
    const per = @max(MIN_ROWS_PER_THREAD, (rows + (workers + 1) * 2 - 1) / ((workers + 1) * 2));
    pool.job_ty = t;
    pool.job_out = out;
    pool.job_data = data;
    pool.job_x = x;
    pool.job_rows = rows;
    pool.job_cols = cols;
    pool.job_rb = rb;
    pool.job_chunk = per;
    pool.cursor.store(0, .monotonic);
    pool.active.store(workers, .monotonic);
    // release: everything above must be visible before a worker sees the seq
    _ = pool.seq.fetchAdd(1, .release);

    runChunks(); // the submitting thread works too

    // Workers are running right now, so this wait is short by construction:
    // spin rather than yield.
    var spins: u32 = 0;
    while (pool.active.load(.acquire) != 0) {
        spins += 1;
        if (spins < SPIN_BEFORE_YIELD) {
            std.atomic.spinLoopHint();
        } else {
            std.Thread.yield() catch {};
        }
    }
}

fn dispatch(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    switch (t) {
        .f32 => matvecF32(out, data, x, rows, cols),
        .f16 => matvecF16(out, data, x, rows, cols),
        .q4_0 => matvecQ40Int(out, data, x, rows, cols),
        .q4_1 => matvecQ41Int(.q4_1, out, data, x, rows, cols),
        .q5_0 => matvecQ50(out, data, x, rows, cols),
        .q5_1 => matvecQ41Int(.q5_1, out, data, x, rows, cols),
        .q8_0 => matvecQ80Int(out, data, x, rows, cols),
        .q2_k => matvecK256(.q2_k, out, data, x, rows, cols),
        .q3_k => matvecK256(.q3_k, out, data, x, rows, cols),
        .q4_k => matvecK256(.q4_k, out, data, x, rows, cols),
        .q6_k => matvecK256(.q6_k, out, data, x, rows, cols),
        .q5_k => matvecK256(.q5_k, out, data, x, rows, cols),
        .mxfp4 => matvecMxfp4Int(out, data, x, rows, cols),
        else => matvecCodebook(t, out, data, x, rows, cols),
    }
}

/// Decode one codebook block into the front of `vals`, returning how many
/// lanes it wrote (32 for iq4_nl/mxfp4, 256 for the rest).
fn dequantBlockCodebook(t: Type, block: []const u8, vals: *[QK_K]f32) usize {
    switch (t) {
        .iq2_xxs => iq.dequantBlockIq2XXS(block, vals),
        .iq2_xs => iq.dequantBlockIq2XS(block, vals),
        .iq2_s => iq.dequantBlockIq2S(block, vals),
        .iq3_xxs => iq.dequantBlockIq3XXS(block, vals),
        .iq3_s => iq.dequantBlockIq3S(block, vals),
        .iq1_s => iq.dequantBlockIq1S(block, vals),
        .iq1_m => iq.dequantBlockIq1M(block, vals),
        .iq4_xs => iq.dequantBlockIq4XS(block, vals),
        .iq4_nl => {
            iq.dequantBlockIq4NL(block, vals[0..iq.QK_NL]);
            return iq.QK_NL;
        },
        .mxfp4 => {
            iq.dequantBlockMxfp4(block, vals[0..iq.QK_NL]);
            return iq.QK_NL;
        },
        // `supported()` gates every type that reaches a kernel, and resolve()
        // rejects the rest at load; an unknown tag here is a loom bug.
        else => unreachable,
    }
    return QK_K;
}

/// Codebook matvec: decode one block at a time through the grid tables and dot
/// it, same shape as matvecK. Slower per byte than the affine kernels (a table
/// lookup per lane rather than a shift), which is the price of the format.
fn matvecCodebook(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const blk = blockElems(t);
    const bs = blockBytes(t);
    std.debug.assert(cols % blk == 0);
    const blocks_per_row = cols / blk;
    const rb = blocks_per_row * bs;
    var vals: [QK_K]f32 = undefined;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const n = dequantBlockCodebook(t, row[b * bs ..][0..bs], &vals);
            acc += dotF32(vals[0..n], x[b * blk ..][0..blk]);
        }
        out[r] = acc;
    }
}

// ---- int8 activation path (issue #11 follow-up) ------------------------------
//
// The dequantize-then-dot shape spent most of its time in the dequantize.
// Measured single-threaded on a 2048x5632 tensor, Apple M5, before and after:
//
//     type   before              after     of which dequant, before
//     q4_k   1.90 ms  3.4 GB/s   1.12 ms   76%
//     q6_k   1.48 ms  6.4 GB/s   1.18 ms   92%
//     q8_0   3.50 ms  3.5 GB/s   0.36 ms   79%
//     f32    0.75 ms 61.7 GB/s   (unchanged: already bandwidth-bound)
//
// f32 nearly saturates memory bandwidth; the quantized kernels sat an order of
// magnitude below it, building a 256-float scratch buffer per block and
// pushing it through L1 only to read it straight back.
//
// The fix is the one llama.cpp uses: quantize the *activation* vector to int8
// once per matvec, then dot it against the packed weights as integers. The
// weights are never widened to f32, the scratch buffer disappears, and integer
// lanes are four times as wide as f32 lanes on the same register.
//
// This is an approximation the f32 path did not make: the activation carries
// ~0.4% per-element error. Over a 2048-long dot those errors are independent
// and largely cancel, and it is what every production CPU inference engine
// does, but it does mean results are no longer bit-identical to the dequant
// path -- so `matvec` is checked against `dequantRow` within a tolerance, not
// exactly.

/// One 32-value block of int8-quantized activations: `x[i] ~= d * q[i]`.
const XBlock = struct {
    d: f32,
    /// Sum of the quantized lanes, needed by the affine formats whose value is
    /// `scale*q + min`: the min term becomes `min * d * sum(q)`.
    sum: i32,
    q: [QK_0]i8,
};

/// Scratch for the quantized activation. One matvec's worth, reused across
/// rows: quantizing is O(cols) while the matvec is O(rows*cols), so it happens
/// once per call rather than once per row.
const MAX_QX_BLOCKS = 1024; // 32k activation elements
threadlocal var qx_buf: [MAX_QX_BLOCKS]XBlock = undefined;

/// Q8_K-style activation block for the K-quant kernels: ONE scale over a whole
/// 256-wide super-block, plus the eight per-32 sub-block sums that the min
/// term of q4_k/q5_k/q2_k needs. The single scale is the entire point: with
/// one activation scale per super-block, a K-quant kernel can accumulate every
/// sub-block's `scale * dot` as integers in a narrow partial vector and touch
/// floats ONCE per super-block, instead of converting and multiplying floats
/// per 32-wide sub-block. That per-sub-block float tail -- two int->float
/// converts and four multiplies, eight times per super-block -- is what kept
/// the old kernels dot-bound and made batched matmul no cheaper per lane than
/// a matvec (bench: matmul/q4_k x8 at 1.75 ms vs 8 matvecs at 1.13 ms ideal).
const XBlockK = struct {
    d: f32,
    q: [QK_K]i8,
    /// Per-16 sums of q, for the `- dmin*min*sum(x)` term of the min-carrying
    /// formats: Q2_K's mins are per-16, so that is the granularity stored;
    /// Q4_K/Q5_K (per-32 mins) pair-sum adjacent entries. Computed during
    /// quantization where the values are in hand.
    bsum: [QK_K / 16]i32,
};
const MAX_QXK_BLOCKS = MAX_QX_BLOCKS / (QK_K / QK_0); // same element budget, 256-wide
threadlocal var qxk_buf: [MAX_QXK_BLOCKS]XBlockK = undefined;

/// Quantize `x` into 256-wide Q8_K-style blocks. Same symmetric
/// round-to-nearest as `quantizeX`, over a 256-value window.
fn quantizeX256(x: []const f32) []const XBlockK {
    const nb = x.len / QK_K;
    std.debug.assert(nb <= MAX_QXK_BLOCKS and nb * QK_K == x.len);
    for (0..nb) |b| {
        const src = x[b * QK_K ..][0..QK_K];
        var amax: f32 = 0;
        for (src) |v| if (std.math.isFinite(v)) {
            amax = @max(amax, @abs(v));
        };
        const d = amax / 127.0;
        const inv: f32 = if (d != 0 and std.math.isFinite(d)) 1.0 / d else 0;
        const dst = &qxk_buf[b];
        for (0..QK_K / 16) |s| {
            var sum: i32 = 0;
            for (0..16) |i| {
                const q = toI8Saturating(src[s * 16 + i] * inv);
                dst.q[s * 16 + i] = q;
                sum += q;
            }
            dst.bsum[s] = sum;
        }
        dst.d = if (std.math.isFinite(d)) d else 0;
    }
    return qxk_buf[0..nb];
}

/// Whether this build can use the aarch64 `sdot` instruction (4-way i8 dot
/// into 4 i32 lanes). Comptime-gated on the *target's* feature set, so a
/// baseline aarch64 release build (no dotprod) takes the portable path
/// instead of emitting an instruction the CPU may not have.
const HAS_SDOT = builtin.cpu.arch == .aarch64 and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod);

/// acc.4s += sdot(a.16b, b.16b): sixteen i8*i8 products summed 4-at-a-time
/// into the four i32 lanes of `acc`. One instruction for 16 MACs.
inline fn sdotAcc(acc: @Vector(4, i32), a: @Vector(16, i8), b: @Vector(16, i8)) @Vector(4, i32) {
    return asm ("sdot %[r].4s, %[a].16b, %[b].16b"
        : [r] "=w" (-> @Vector(4, i32)),
        : [a] "w" (a),
          [b] "w" (b),
          [acc] "0" (acc),
    );
}

/// Unpack 32 packed nibbles (one 32-byte half of a K-quant qs array) into i8
/// lanes: the low nibbles for even sub-blocks, high for odd. Values are
/// 0..15, so the u8->i8 bitcast is value-preserving.
inline fn nib32i8(src: *const [QK_0]u8, high: bool) @Vector(32, i8) {
    const v: @Vector(QK_0, u8) = src.*;
    const sh4: @Vector(QK_0, u3) = @splat(4);
    const m0f: @Vector(QK_0, u8) = @splat(0x0F);
    return @bitCast(if (high) v >> sh4 else v & m0f);
}

/// 16-wide partial dot (one sdot on aarch64+dotprod). The per-16 granularity
/// matches the K-quant scale layout: Q2_K/Q3_K/Q6_K scale every 16 values,
/// Q4_K/Q5_K every 32 (their scale is just duplicated across the pair).
inline fn dot16Partial(w: @Vector(16, i8), x: @Vector(16, i8)) @Vector(4, i32) {
    if (comptime HAS_SDOT) {
        const zero: @Vector(4, i32) = @splat(0);
        return sdotAcc(zero, w, x);
    }
    const prod = @as(@Vector(16, i16), w) * @as(@Vector(16, i16), x);
    const p: @Vector(16, i32) = prod;
    var part: @Vector(4, i32) = @splat(0);
    inline for (0..4) |c| part += std.simd.extract(p, c * 4, 4);
    return part;
}

/// A K-quant super-block decoded to a uniform shape: eight 32-wide i8 weight
/// groups, sixteen per-16 scales, and (for the min-carrying formats) sixteen
/// per-16 mins. Every K-quant type decodes into this, which is what lets one
/// accumulation core -- and one batched matmul -- serve all five: the batched
/// kernel decodes a super-block ONCE and runs every lane against it, which is
/// where the batch amortization actually lives.
const KDec = struct {
    ws: [8]@Vector(QK_0, i8),
    scs: [16]i32,
    mns: [16]i32,
    d: f32,
    dmin: f32,
};

inline fn kHasMin(comptime t: Type) bool {
    return switch (t) {
        .q2_k, .q4_k, .q5_k => true,
        .q3_k, .q6_k => false,
        else => unreachable,
    };
}

inline fn kBlockSize(comptime t: Type) usize {
    return switch (t) {
        .q2_k => Q2_K_BLOCK,
        .q3_k => Q3_K_BLOCK,
        .q4_k => Q4_K_BLOCK,
        .q5_k => Q5_K_BLOCK,
        .q6_k => Q6_K_BLOCK,
        else => unreachable,
    };
}

/// Decode one super-block of type `t` into the uniform KDec shape. Layouts
/// mirror the validated reference dequants exactly; the group->activation
/// mapping is linear (group g covers values g*32..g*32+32, per-16 sub-blocks
/// 2g and 2g+1), which each unpack below preserves.
inline fn kDecode(comptime t: Type, block: *const [kBlockSize(t)]u8) KDec {
    var dec: KDec = undefined;
    switch (t) {
        .q4_k, .q5_k => {
            dec.d = f16FromBytes(block[0..2]);
            dec.dmin = f16FromBytes(block[2..4]);
            const scales: *const [12]u8 = block[4..16];
            const qs = if (t == .q5_k) block[16 + QK_K / 8 ..][0 .. QK_K / 2] else block[16..][0 .. QK_K / 2];
            inline for (0..8) |j| {
                var sc: u8 = undefined;
                var mn: u8 = undefined;
                scaleMinK4(j, scales, &sc, &mn);
                dec.scs[j * 2] = sc;
                dec.scs[j * 2 + 1] = sc;
                dec.mns[j * 2] = mn;
                dec.mns[j * 2 + 1] = mn;
                const nib: @Vector(QK_0, u8) = @bitCast(nib32i8(qs[(j / 2) * QK_0 ..][0..QK_0], j % 2 == 1));
                if (t == .q5_k) {
                    // Fifth bit: sub-block j uses bit j%2 + 2*(j/2) of every
                    // qh byte (the dequant walks 64 values per iteration with
                    // its two masks advancing two bits each time).
                    const qh = block[16..][0 .. QK_K / 8];
                    const mask: u8 = @as(u8, 1) << @intCast(j % 2 + 2 * (j / 2));
                    const hv: @Vector(QK_0, u8) = qh[0..QK_0].*;
                    const zero: @Vector(QK_0, u8) = @splat(0);
                    const hbit: @Vector(QK_0, u8) = @select(
                        u8,
                        (hv & @as(@Vector(QK_0, u8), @splat(mask))) != zero,
                        @as(@Vector(QK_0, u8), @splat(16)),
                        zero,
                    );
                    dec.ws[j] = @bitCast(nib | hbit);
                } else {
                    dec.ws[j] = @bitCast(nib);
                }
            }
        },
        .q6_k => {
            const ql_all = block[0 .. QK_K / 2];
            const qh_all = block[QK_K / 2 ..][0 .. QK_K / 4];
            const sc_all = block[QK_K / 2 + QK_K / 4 ..][0 .. QK_K / 16];
            dec.d = f16FromBytes(block[QK_K / 2 + QK_K / 4 + QK_K / 16 ..][0..2]);
            dec.dmin = 0;
            inline for (0..16) |j| dec.scs[j] = @as(i8, @bitCast(sc_all[j]));
            dec.mns = @splat(0);
            const m0f: @Vector(QK_0, u8) = @splat(0x0F);
            const m3: @Vector(QK_0, u8) = @splat(3);
            const sh4: @Vector(QK_0, u3) = @splat(4);
            const sh2: @Vector(QK_0, u3) = @splat(2);
            const sh6: @Vector(QK_0, u3) = @splat(6);
            const off32: @Vector(QK_0, u8) = @splat(32);
            inline for (0..2) |half| {
                const ql = ql_all[half * 64 ..][0..64];
                const qh = qh_all[half * 32 ..][0..32];
                const a: @Vector(QK_0, u8) = ql[0..QK_0].*;
                const bb: @Vector(QK_0, u8) = ql[32..64].*;
                const h: @Vector(QK_0, u8) = qh[0..QK_0].*;
                const g: [4]@Vector(QK_0, u8) = .{
                    (a & m0f) | ((h & m3) << sh4),
                    (bb & m0f) | (((h >> sh2) & m3) << sh4),
                    (a >> sh4) | (((h >> sh4) & m3) << sh4),
                    (bb >> sh4) | (((h >> sh6) & m3) << sh4),
                };
                // q is 0..63; q-32 in wrapping u8 then bitcast is the correct
                // two's-complement -32..31.
                inline for (0..4) |gg| dec.ws[half * 4 + gg] = @bitCast(g[gg] -% off32);
            }
        },
        .q2_k => {
            const scales = block[0 .. QK_K / 16];
            const qs = block[QK_K / 16 ..][0 .. QK_K / 4];
            dec.d = f16FromBytes(block[QK_K / 16 + QK_K / 4 ..][0..2]);
            dec.dmin = f16FromBytes(block[QK_K / 16 + QK_K / 4 + 2 ..][0..2]);
            inline for (0..16) |j| {
                dec.scs[j] = scales[j] & 0xF;
                dec.mns[j] = scales[j] >> 4;
            }
            const m3: @Vector(QK_0, u8) = @splat(3);
            inline for (0..2) |half| {
                const qb: @Vector(QK_0, u8) = qs[half * QK_0 ..][0..QK_0].*;
                inline for (0..4) |shift| {
                    const shv: @Vector(QK_0, u3) = @splat(@as(u3, 2 * shift));
                    dec.ws[half * 4 + shift] = @bitCast((qb >> shv) & m3);
                }
            }
        },
        .q3_k => {
            const hmask = block[0 .. QK_K / 8];
            const qs = block[QK_K / 8 ..][0 .. QK_K / 4];
            const sc16 = q3kScales(block[QK_K / 8 + QK_K / 4 ..][0..12]);
            dec.d = f16FromBytes(block[QK_K / 8 + QK_K / 4 + 12 ..][0..2]);
            dec.dmin = 0;
            inline for (0..16) |j| dec.scs[j] = @as(i32, sc16[j]) - 32;
            dec.mns = @splat(0);
            const m3: @Vector(QK_0, u8) = @splat(3);
            const zero: @Vector(QK_0, u8) = @splat(0);
            const four: @Vector(QK_0, u8) = @splat(4);
            const hm: @Vector(QK_0, u8) = hmask[0..QK_0].*;
            inline for (0..2) |half| {
                const qb: @Vector(QK_0, u8) = qs[half * QK_0 ..][0..QK_0].*;
                inline for (0..4) |shift| {
                    const idx = half * 4 + shift;
                    const shv: @Vector(QK_0, u3) = @splat(@as(u3, 2 * shift));
                    const mbit: @Vector(QK_0, u8) = @splat(@as(u8, 1) << @intCast(idx));
                    // subtract 4 where the high bit is CLEAR; wrapping u8 then
                    // bitcast yields the correct -4..3.
                    const sub: @Vector(QK_0, u8) = @select(u8, (hm & mbit) != zero, zero, four);
                    dec.ws[idx] = @bitCast(((qb >> shv) & m3) -% sub);
                }
            }
        },
        else => unreachable,
    }
    return dec;
}

/// The shared accumulation core: one decoded super-block against one lane's
/// 256 int8 activations. Pure integer until a single float tail. Used by both
/// the matvec and the batched matmul, so they are bit-identical by
/// construction.
inline fn kBlockDot(comptime t: Type, dec: *const KDec, xb: *const XBlockK) f32 {
    var ivec: @Vector(4, i32) = @splat(0);
    inline for (0..8) |g| {
        const xv: @Vector(QK_0, i8) = xb.q[g * QK_0 ..][0..QK_0].*;
        const wlo: @Vector(16, i8) = std.simd.extract(dec.ws[g], 0, 16);
        const whi: @Vector(16, i8) = std.simd.extract(dec.ws[g], 16, 16);
        const xlo: @Vector(16, i8) = std.simd.extract(xv, 0, 16);
        const xhi: @Vector(16, i8) = std.simd.extract(xv, 16, 16);
        ivec += dot16Partial(wlo, xlo) * @as(@Vector(4, i32), @splat(dec.scs[g * 2]));
        ivec += dot16Partial(whi, xhi) * @as(@Vector(4, i32), @splat(dec.scs[g * 2 + 1]));
    }
    if (comptime kHasMin(t)) {
        var min_isum: i32 = 0;
        inline for (0..16) |j| min_isum += dec.mns[j] * xb.bsum[j];
        return xb.d * (dec.d * @as(f32, @floatFromInt(@reduce(.Add, ivec))) -
            dec.dmin * @as(f32, @floatFromInt(min_isum)));
    }
    return xb.d * dec.d * @as(f32, @floatFromInt(@reduce(.Add, ivec)));
}

/// Generic K-quant matvec over the uniform decode: quantize the activation to
/// Q8_K once, then decode-and-dot each super-block.
fn matvecK256(comptime t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    @setEvalBranchQuota(1_000_000);
    const qx = quantizeX256(x);
    const bs = comptime kBlockSize(t);
    const rb = (cols / QK_K) * bs;
    for (0..rows) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        for (qx, 0..) |*xb, b| {
            const dec = kDecode(t, row[b * bs ..][0..bs]);
            acc += kBlockDot(t, &dec, xb);
        }
        out[r] = acc;
    }
}

/// Generic batched K-quant matmul: decode each super-block ONCE, then run
/// every lane's integer accumulation against it. The decode -- scale/min
/// extraction and nibble/bit unpacking, the scalar-heavy half of the kernel --
/// is amortized across the batch; per lane only the sdot/mla core remains.
fn matmulK256(comptime t: Type, out: []f32, data: []const u8, n: usize, rows: usize, cols: usize, r0: usize, r1: usize) void {
    @setEvalBranchQuota(1_000_000);
    const bs = comptime kBlockSize(t);
    const bpr = cols / QK_K;
    const rb = bpr * bs;
    for (r0..r1) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: [MAX_BATCH]f32 = @splat(0);
        for (0..bpr) |b| {
            const dec = kDecode(t, row[b * bs ..][0..bs]);
            const blk = &qxkn_buf[b];
            for (0..n) |k| acc[k] += kBlockDot(t, &dec, &blk[k]);
        }
        for (0..n) |k| out[k * rows + r] = acc[k];
    }
}

/// Round-to-int that cannot trap on a poisoned model (security issue #165).
/// A NaN or Inf scale decoded from an attacker-supplied tensor propagates into
/// activations, and `@intFromFloat` on a non-finite value is a checked trap in
/// ReleaseSafe and undefined behavior in ReleaseFast. Non-finite maps to 0
/// (the block's own scale is separately zeroed) and finite values saturate
/// into i8 instead of relying on the conversion's range check.
inline fn toI8Saturating(v: f32) i8 {
    if (!std.math.isFinite(v)) return 0;
    const r = @round(v);
    if (r >= 127.0) return 127;
    if (r <= -127.0) return -127;
    return @intFromFloat(r);
}

/// Quantize `x` into `qx_buf`, returning the populated slice. Symmetric
/// round-to-nearest against the block's absolute maximum, matching Q8_0.
fn quantizeX(x: []const f32) []const XBlock {
    const nb = x.len / QK_0;
    std.debug.assert(nb <= MAX_QX_BLOCKS);
    for (0..nb) |b| {
        const src = x[b * QK_0 ..][0..QK_0];
        var amax: f32 = 0;
        for (src) |v| if (std.math.isFinite(v)) {
            amax = @max(amax, @abs(v));
        };
        const d = amax / 127.0;
        const inv: f32 = if (d != 0 and std.math.isFinite(d)) 1.0 / d else 0;
        var sum: i32 = 0;
        for (src, 0..) |v, i| {
            const q = toI8Saturating(v * inv);
            qx_buf[b].q[i] = q;
            sum += q;
        }
        qx_buf[b].d = if (std.math.isFinite(d)) d else 0;
        qx_buf[b].sum = sum;
    }
    return qx_buf[0..nb];
}

/// 16-lane 4-bit codebook gather. On aarch64 this is a single `tbl`
/// instruction (what makes llama.cpp's MXFP4 fast); elsewhere a scalar
/// fallback. The MXFP4 value table is 16 i8 entries, so one tbl covers a
/// full nibble half.
inline fn fp4Gather16(nibbles: @Vector(16, u8)) @Vector(16, i8) {
    if (builtin.cpu.arch == .aarch64) {
        const table: @Vector(16, i8) = iq_tables.kvalues_fp4;
        return asm ("tbl %[o].16b, {%[t].16b}, %[i].16b"
            : [o] "=w" (-> @Vector(16, i8)),
            : [t] "w" (table),
              [i] "w" (nibbles),
        );
    }
    const arr: [16]i8 = iq_tables.kvalues_fp4;
    var out: @Vector(16, i8) = undefined;
    inline for (0..16) |k| out[k] = arr[nibbles[k]];
    return out;
}

/// MXFP4 (gpt-oss, and the MXFP4 quants of DeepSeek-V4 / Ling) against int8
/// activations.
///
/// The E2M1 value table is stored doubled and *integer* -- {0,1,2,3,4,6,8,12}
/// and negatives (`iq_tables.kvalues_fp4`, i8) -- with the compensating 1/2
/// folded into the per-block E8M0 scale. So a weight nibble is a lookup into a
/// 16-entry i8 codebook, exactly the shape the int8 path wants: gather the
/// weights with `tbl`, quantize the activation once, do one int8 dot per
/// 32-value block, scale by `d * dx`. No zero-point, so no min/sum term.
///
/// This replaces the generic dequant-to-f32 `matvecCodebook` for MXFP4, the
/// single kernel gpt-oss decode spends all its time in. The 32 nibbles map
/// low-nibbles-first then high nibbles, matching how the activation is
/// quantized in natural order.
fn matvecMxfp4Int(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const qx = quantizeX(x);
    const blocks_per_row = cols / QK_0; // QK_0 == 32 == MXFP4 block width
    const rb = blocks_per_row * iq.MXFP4_BLOCK;
    const lo_mask: @Vector(QK_0 / 2, u8) = @splat(0x0F);
    const sh4: @Vector(QK_0 / 2, u3) = @splat(4);
    for (0..rows) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        for (0..blocks_per_row) |b| {
            const block = row[b * iq.MXFP4_BLOCK ..][0..iq.MXFP4_BLOCK];
            const d = iq.e8m0ToF32Half(block[0]);
            const qs: @Vector(QK_0 / 2, u8) = block[1..][0 .. QK_0 / 2].*;
            const wlo = fp4Gather16(qs & lo_mask); // 16 i8, vals[0..15]
            const whi = fp4Gather16(qs >> sh4); //    16 i8, vals[16..31]
            const xb = &qx[b];
            const xlo: @Vector(QK_0 / 2, i8) = xb.q[0 .. QK_0 / 2].*;
            const xhi: @Vector(QK_0 / 2, i8) = xb.q[QK_0 / 2 ..][0 .. QK_0 / 2].*;
            // widen i8 -> i32 before multiply-reduce: the SDOT idiom.
            const plo: @Vector(QK_0 / 2, i32) = @as(@Vector(QK_0 / 2, i32), wlo) * @as(@Vector(QK_0 / 2, i32), xlo);
            const phi: @Vector(QK_0 / 2, i32) = @as(@Vector(QK_0 / 2, i32), whi) * @as(@Vector(QK_0 / 2, i32), xhi);
            const dot = @reduce(.Add, plo + phi);
            acc += d * xb.d * @as(f32, @floatFromInt(dot));
        }
        out[r] = acc;
    }
}

/// True for the kernels that quantize the activation vector to int8, and are
/// therefore approximate rather than exact against the dequantize-then-dot
/// reference.
pub fn usesInt8Activations(t: Type) bool {
    return switch (t) {
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q8_0, .q4_1, .q5_1, .mxfp4 => true,
        else => false,
    };
}

/// Unpack the 12 packed bytes of a Q3_K super-block into 16 unsigned 6-bit
/// scales (0..63). Mirrors llama.cpp's aux shuffle exactly.
inline fn q3kScales(packed_scales: []const u8) [16]u8 {
    var aux: [4]u32 = undefined;
    aux[0] = std.mem.readInt(u32, packed_scales[0..4], .little);
    aux[1] = std.mem.readInt(u32, packed_scales[4..8], .little);
    aux[2] = std.mem.readInt(u32, packed_scales[8..12], .little);
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    var out: [16]u8 = undefined;
    inline for (0..4) |i| {
        const bytes = std.mem.toBytes(aux[i]);
        inline for (0..4) |j| out[i * 4 + j] = bytes[j];
    }
    return out;
}

/// Q4_0 against int8 activations: one f16 scale per 32 values and a fixed -8
/// offset, so the offset factors out as `-8 * sum(xq)` and never touches a
/// lane.
fn matvecQ40Int(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const qx = quantizeX(x);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q4_0_BLOCK;
    for (0..rows) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        for (0..blocks_per_row) |b| {
            const block = row[b * Q4_0_BLOCK ..][0..Q4_0_BLOCK];
            const d = f16FromBytes(block[0..2]);
            const qs = block[2..][0 .. QK_0 / 2];
            // Q4_0 packs lane i in the low nibble and lane i+16 in the high
            // nibble of byte i, so the two halves are gathered separately.
            const xb = &qx[b];
            var dot: i32 = 0;
            var sum: i32 = 0;
            for (0..QK_0 / 2) |i| {
                const lo: i32 = qs[i] & 0x0F;
                const hi: i32 = qs[i] >> 4;
                dot += lo * @as(i32, xb.q[i]) + hi * @as(i32, xb.q[i + QK_0 / 2]);
                sum += @as(i32, xb.q[i]) + @as(i32, xb.q[i + QK_0 / 2]);
            }
            acc += d * xb.d * @as(f32, @floatFromInt(dot - 8 * sum));
        }
        out[r] = acc;
    }
}

/// Q8_0 against int8 activations: both sides are already int8, so this is a
/// straight integer dot with two scales.
/// The affine `_1` family (Q4_1, Q5_1) against int8 activations.
///
/// These carry a per-block min as well as a scale, so a weight is `d*q + m`
/// with q unsigned. That separates cleanly over a block:
///
///     sum_i (d*q_i + m) * x_i  =  d*dx*dot(q, xq) + m*dx*sum(xq)
///
/// so the min never enters the inner loop -- one integer dot and one integer
/// sum, exactly as for Q4_0, with `m` folded in at the end.
///
/// This matters more than the type's rarity suggests. A `Q4_K_M` checkpoint is
/// not uniformly Q4_K: llama.cpp's mixtures put `ffn_down` in a different type,
/// and in DeepSeek-Coder-V2-Lite every `ffn_down_exps` -- a third of every
/// expert's weights, and its largest tensor -- is Q5_1. Without this kernel
/// that third fell to the exact dequantize-then-dot path, and expert FFN
/// measured 5.5 GB/s against the ~53 GB/s the int8 kernels reach.
fn matvecQ41Int(comptime t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const bs: usize = if (t == .q4_1) Q4_1_BLOCK else Q5_1_BLOCK;
    const qoff: usize = if (t == .q5_1) 8 else 4;
    const qx = quantizeX(x);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * bs;
    for (0..rows) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        for (0..blocks_per_row) |b| {
            const block = row[b * bs ..][0..bs];
            const d = f16FromBytes(block[0..2]);
            const m = f16FromBytes(block[2..4]);
            const qh: u32 = if (t == .q5_1) std.mem.readInt(u32, block[4..8], .little) else 0;
            const qs = block[qoff..][0 .. QK_0 / 2];
            const xb = &qx[b];
            var dot: i32 = 0;
            var sum: i32 = 0;
            // Lane i is in the low nibble of byte i, lane i+16 in the high
            // nibble, and for Q5_1 the fifth bit comes from `qh`.
            for (0..QK_0 / 2) |i| {
                const sh: u5 = @intCast(i);
                const lo: i32 = (qs[i] & 0x0F) | (if (t == .q5_1) @as(i32, @intCast(((qh >> sh) << 4) & 0x10)) else 0);
                const hi: i32 = (qs[i] >> 4) | (if (t == .q5_1) @as(i32, @intCast((qh >> (sh + 12)) & 0x10)) else 0);
                const x0: i32 = xb.q[i];
                const x1: i32 = xb.q[i + QK_0 / 2];
                dot += lo * x0 + hi * x1;
                sum += x0 + x1;
            }
            acc += xb.d * (d * @as(f32, @floatFromInt(dot)) + m * @as(f32, @floatFromInt(sum)));
        }
        out[r] = acc;
    }
}

fn matvecQ80Int(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const qx = quantizeX(x);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q8_0_BLOCK;
    for (0..rows) |r| {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        for (0..blocks_per_row) |b| {
            const block = row[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
            const d = f16FromBytes(block[0..2]);
            const W: @Vector(QK_0, i16) = @as(@Vector(QK_0, i8), @bitCast(block[2..][0..QK_0].*));
            const X: @Vector(QK_0, i16) = @as(@Vector(QK_0, i8), qx[b].q);
            const dot = @reduce(.Add, @as(@Vector(QK_0, i32), W * X));
            acc += d * qx[b].d * @as(f32, @floatFromInt(dot));
        }
        out[r] = acc;
    }
}

/// K-quant matvec: dequantize one 256-wide super-block at a time and dot it.
fn matvecK(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_K == 0);
    const bs: usize = switch (t) {
        .q2_k => Q2_K_BLOCK,
        .q3_k => Q3_K_BLOCK,
        .q4_k => Q4_K_BLOCK,
        .q5_k => Q5_K_BLOCK,
        .q6_k => Q6_K_BLOCK,
        else => unreachable,
    };
    const blocks_per_row = cols / QK_K;
    const rb = blocks_per_row * bs;
    var vals: [QK_K]f32 = undefined;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * bs ..][0..bs];
            switch (t) {
                .q2_k => dequantBlockQ2K(block, &vals),
                .q3_k => dequantBlockQ3K(block, &vals),
                .q4_k => dequantBlockQ4K(block, &vals),
                .q5_k => dequantBlockQ5K(block, &vals),
                .q6_k => dequantBlockQ6K(block, &vals),
                else => unreachable,
            }
            acc += dotF32(&vals, x[b * QK_K ..][0..QK_K]);
        }
        out[r] = acc;
    }
}

/// 6-bit scale/min unpacking shared by q4_k/q5_k (llama.cpp get_scale_min_k4).
inline fn scaleMinK4(j: usize, scales: *const [12]u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = scales[j] & 63;
        m.* = scales[j + 4] & 63;
    } else {
        sc.* = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m.* = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

fn dequantBlockQ4K(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const dmin = f16FromBytes(block[2..4]);
    const scales: *const [12]u8 = block[4..16];
    const qs = block[16..][0 .. QK_K / 2];
    var is: usize = 0;
    var y: usize = 0;
    var q: usize = 0;
    var j: usize = 0;
    while (j < QK_K) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, scales, &sc, &mn);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(mn));
        scaleMinK4(is + 1, scales, &sc, &mn);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(mn));
        const vd1: Vf = @splat(d1);
        const vm1: Vf = @splat(m1);
        const vd2: Vf = @splat(d2);
        const vm2: Vf = @splat(m2);
        var l: usize = 0;
        while (l < 32) : (l += LANES) {
            const b = qs[q + l ..][0..LANES];
            vals[y + l ..][0..LANES].* = vd1 * loNib(b) - vm1;
            vals[y + 32 + l ..][0..LANES].* = vd2 * hiNib(b) - vm2;
        }
        y += 64;
        q += 32;
        is += 2;
    }
}

fn dequantBlockQ5K(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const dmin = f16FromBytes(block[2..4]);
    const scales: *const [12]u8 = block[4..16];
    const qh = block[16..][0 .. QK_K / 8];
    const qs = block[16 + QK_K / 8 ..][0 .. QK_K / 2];
    var is: usize = 0;
    var y: usize = 0;
    var q: usize = 0;
    var hb1: u8 = 1;
    var hb2: u8 = 2;
    var j: usize = 0;
    while (j < QK_K) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, scales, &sc, &mn);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(mn));
        scaleMinK4(is + 1, scales, &sc, &mn);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(mn));
        const vd1: Vf = @splat(d1);
        const vm1: Vf = @splat(m1);
        const vd2: Vf = @splat(d2);
        const vm2: Vf = @splat(m2);
        const v16: Vf = @splat(16);
        const vz: Vf = @splat(0);
        var l: usize = 0;
        while (l < 32) : (l += LANES) {
            const b = qs[q + l ..][0..LANES];
            const h: @Vector(LANES, u8) = qh[l..][0..LANES].*;
            const set1 = (h & @as(@Vector(LANES, u8), @splat(hb1))) != @as(@Vector(LANES, u8), @splat(0));
            const set2 = (h & @as(@Vector(LANES, u8), @splat(hb2))) != @as(@Vector(LANES, u8), @splat(0));
            vals[y + l ..][0..LANES].* = vd1 * (loNib(b) + @select(f32, set1, v16, vz)) - vm1;
            vals[y + 32 + l ..][0..LANES].* = vd2 * (hiNib(b) + @select(f32, set2, v16, vz)) - vm2;
        }
        y += 64;
        q += 32;
        is += 2;
        hb1 <<= 2;
        hb2 <<= 2;
    }
}

fn dequantBlockQ6K(block: []const u8, vals: *[QK_K]f32) void {
    const ql_all = block[0 .. QK_K / 2];
    const qh_all = block[QK_K / 2 ..][0 .. QK_K / 4];
    const sc_all = block[QK_K / 2 + QK_K / 4 ..][0 .. QK_K / 16];
    const d = f16FromBytes(block[QK_K / 2 + QK_K / 4 + QK_K / 16 ..][0..2]);

    var y: usize = 0;
    var qlo: usize = 0;
    var qho: usize = 0;
    var sco: usize = 0;
    var n: usize = 0;
    const v32: Vf = @splat(32);
    const m0f: @Vector(LANES, u8) = @splat(0x0F);
    const m3: @Vector(LANES, u8) = @splat(3);
    while (n < QK_K) : (n += 128) {
        const ql = ql_all[qlo..];
        const qh = qh_all[qho..];
        const sc = sc_all[sco..];
        // `is` is l/16, so it is constant across each half of the 32-wide run.
        // Splitting the loop hoists the four scale lookups out of it and makes
        // the body pure vector work.
        var half: usize = 0;
        while (half < 2) : (half += 1) {
            const is = half;
            const s1: Vf = @splat(d * i8f(sc[is]));
            const s2: Vf = @splat(d * i8f(sc[is + 2]));
            const s3: Vf = @splat(d * i8f(sc[is + 4]));
            const s4: Vf = @splat(d * i8f(sc[is + 6]));
            var l: usize = half * 16;
            while (l < half * 16 + 16) : (l += LANES) {
                const a: @Vector(LANES, u8) = ql[l..][0..LANES].*;
                const b: @Vector(LANES, u8) = ql[l + 32 ..][0..LANES].*;
                const h: @Vector(LANES, u8) = qh[l..][0..LANES].*;
                const sh4: @Vector(LANES, u3) = @splat(4);
                const sh2: @Vector(LANES, u3) = @splat(2);
                const sh6: @Vector(LANES, u3) = @splat(6);
                const q1: Vf = @floatFromInt((a & m0f) | ((h & m3) << sh4));
                const q2: Vf = @floatFromInt((b & m0f) | (((h >> sh2) & m3) << sh4));
                const q3: Vf = @floatFromInt((a >> sh4) | (((h >> sh4) & m3) << sh4));
                const q4: Vf = @floatFromInt((b >> sh4) | (((h >> sh6) & m3) << sh4));
                vals[y + l ..][0..LANES].* = s1 * (q1 - v32);
                vals[y + l + 32 ..][0..LANES].* = s2 * (q2 - v32);
                vals[y + l + 64 ..][0..LANES].* = s3 * (q3 - v32);
                vals[y + l + 96 ..][0..LANES].* = s4 * (q4 - v32);
            }
        }
        y += 128;
        qlo += 64;
        qho += 32;
        sco += 8;
    }
}

/// Q2_K reference dequant (exact mirror of llama.cpp dequantize_row_q2_K).
/// Used for embeddings lookup and as the golden reference in tests; the hot
/// path is matvecQ2K.
fn dequantBlockQ2K(block: []const u8, vals: *[QK_K]f32) void {
    const scales = block[0 .. QK_K / 16];
    const qs = block[QK_K / 16 ..][0 .. QK_K / 4];
    const d = f16FromBytes(block[QK_K / 16 + QK_K / 4 ..][0..2]);
    const dmin = f16FromBytes(block[QK_K / 16 + QK_K / 4 + 2 ..][0..2]);
    var y: usize = 0;
    var qbase: usize = 0;
    var is: usize = 0;
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        var j: usize = 0;
        while (j < 4) : (j += 1) {
            const shift: u3 = @intCast(j * 2);
            var half: usize = 0;
            while (half < 2) : (half += 1) {
                const sc = scales[is];
                is += 1;
                const dl = d * @as(f32, @floatFromInt(sc & 0xF));
                const ml = dmin * @as(f32, @floatFromInt(sc >> 4));
                var l: usize = 0;
                while (l < 16) : (l += 1) {
                    const q2: i32 = @intCast((qs[qbase + half * 16 + l] >> shift) & 3);
                    vals[y] = dl * @as(f32, @floatFromInt(q2)) - ml;
                    y += 1;
                }
            }
        }
        qbase += 32;
    }
}

/// Q3_K reference dequant (exact mirror of llama.cpp dequantize_row_q3_K).
fn dequantBlockQ3K(block: []const u8, vals: *[QK_K]f32) void {
    const hmask = block[0 .. QK_K / 8];
    const qs = block[QK_K / 8 ..][0 .. QK_K / 4];
    const scales = q3kScales(block[QK_K / 8 + QK_K / 4 ..][0..12]);
    const d_all = f16FromBytes(block[QK_K / 8 + QK_K / 4 + 12 ..][0..2]);
    var y: usize = 0;
    var qbase: usize = 0;
    var is: usize = 0;
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        var j: usize = 0;
        while (j < 4) : (j += 1) {
            const shift: u3 = @intCast(j * 2);
            // hmask bit for this sub-block pair: exponent (n/128)*4 + j, 0..7.
            const m: u8 = @as(u8, 1) << @intCast((n / 128) * 4 + j);
            var half: usize = 0;
            while (half < 2) : (half += 1) {
                const dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                var l: usize = 0;
                while (l < 16) : (l += 1) {
                    const idx = half * 16 + l;
                    const q2: i32 = @intCast((qs[qbase + idx] >> shift) & 3);
                    const off: i32 = if ((hmask[idx] & m) != 0) 0 else 4;
                    vals[y] = dl * @as(f32, @floatFromInt(q2 - off));
                    y += 1;
                }
            }
        }
        qbase += 32;
    }
}

fn i8f(b: u8) f32 {
    return @floatFromInt(@as(i8, @bitCast(b)));
}

fn matvecF32(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data[0 .. rows * cols * 4]));
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        out[r] = dotF32(w[r * cols ..][0..cols], x);
    }
}

fn matvecF16(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data[0 .. rows * cols * 2]));
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = w[r * cols ..][0..cols];
        var vacc: Vf = @splat(0);
        var i: usize = 0;
        while (i + LANES <= cols) : (i += LANES) {
            vacc += f16x8(row[i..][0..LANES]) * @as(Vf, x[i..][0..LANES].*);
        }
        var acc = @reduce(.Add, vacc);
        while (i < cols) : (i += 1) {
            acc += @as(f32, @floatCast(@as(f16, @bitCast(row[i])))) * x[i];
        }
        out[r] = acc;
    }
}

fn matvecQ40(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q4_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q4_0_BLOCK ..][0..Q4_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            // GGML q4_0 nibble layout: byte j holds elements j (low nibble)
            // and j+16 (high nibble); value = (nibble - 8) * scale.
            var j: usize = 0;
            while (j < QK_0 / 2) : (j += 1) {
                const byte = block[2 + j];
                const lo = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0x0f)) - 8));
                const hi = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8));
                partial += lo * xb[j] + hi * xb[j + QK_0 / 2];
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

/// Q4_1 / Q5_1: the affine "_1" variants. Unlike Q4_0/Q5_0 there is no fixed
/// -8/-16 offset; each block carries its own minimum, so a value is
/// `d*q + m` with q unsigned. Rare in modern files, but llama.cpp's quant
/// *mixes* still fold a handful of them into otherwise K-quant checkpoints,
/// and one unreadable tensor blocks a whole model (issue #51).
fn matvecQ41(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q4_1_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q4_1_BLOCK ..][0..Q4_1_BLOCK];
            const d = f16FromBytes(block[0..2]);
            const m = f16FromBytes(block[2..4]);
            const qs = block[4..][0 .. QK_0 / 2];
            const xb = x[b * QK_0 ..][0..QK_0];
            // sum(d*q + m)*x = d*sum(q*x) + m*sum(x): hoist the min out of the
            // inner loop rather than adding it per lane.
            var dot_q: f32 = 0;
            var j: usize = 0;
            while (j < QK_0 / 2) : (j += 1) {
                dot_q += @as(f32, @floatFromInt(qs[j] & 0xF)) * xb[j] +
                    @as(f32, @floatFromInt(qs[j] >> 4)) * xb[j + QK_0 / 2];
            }
            acc += d * dot_q + m * sumF32(xb);
        }
        out[r] = acc;
    }
}

fn matvecQ51(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q5_1_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q5_1_BLOCK ..][0..Q5_1_BLOCK];
            const d = f16FromBytes(block[0..2]);
            const m = f16FromBytes(block[2..4]);
            const qh = std.mem.readInt(u32, block[4..8], .little);
            const qs = block[8..][0 .. QK_0 / 2];
            const xb = x[b * QK_0 ..][0..QK_0];
            var dot_q: f32 = 0;
            var sum_x: f32 = 0;
            var j: u5 = 0;
            while (true) {
                // 5th bit: lane j from bit j, lane j+16 from bit j+16 (the
                // +12 shift lands it at 0x10 directly, as in llama.cpp).
                const xh0: u8 = @intCast(((qh >> j) << 4) & 0x10);
                const xh1: u8 = @intCast((qh >> (j + 12)) & 0x10);
                const x0 = xb[j];
                const x1 = xb[@as(usize, j) + QK_0 / 2];
                dot_q += @as(f32, @floatFromInt((qs[j] & 0xF) | xh0)) * x0 +
                    @as(f32, @floatFromInt((qs[j] >> 4) | xh1)) * x1;
                sum_x += x0 + x1;
                if (j == 15) break;
                j += 1;
            }
            acc += d * dot_q + m * sum_x;
        }
        out[r] = acc;
    }
}

fn matvecQ50(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q5_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q5_0_BLOCK ..][0..Q5_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const qh = std.mem.readInt(u32, block[2..6], .little);
            const qs = block[6..][0 .. QK_0 / 2];
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            var j: u5 = 0;
            while (true) {
                const xh0: u8 = @intCast(((qh >> j) << 4) & 0x10);
                const xh1: u8 = @intCast((qh >> (j + 12)) & 0x10);
                const w0 = @as(f32, @floatFromInt(@as(i32, (qs[j] & 0xF) | xh0) - 16));
                const w1 = @as(f32, @floatFromInt(@as(i32, (qs[j] >> 4) | xh1) - 16));
                partial += w0 * xb[j] + w1 * xb[@as(usize, j) + QK_0 / 2];
                if (j == 15) break;
                j += 1;
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

fn matvecQ80(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q8_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            var j: usize = 0;
            while (j < QK_0) : (j += 1) {
                partial += @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j])))) * xb[j];
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

/// Read row `r` of an F32 tensor directly (embeddings lookup).
pub fn f32Row(data: []const u8, r: usize, cols: usize) []const f32 {
    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
    return w[r * cols ..][0..cols];
}

/// Dequantize row `r` of a tensor in format `t` into `out` (embeddings lookup
/// for non-f32 token_embd).
pub fn dequantRow(t: Type, out: []f32, data: []const u8, r: usize, cols: usize) void {
    switch (t) {
        .f32 => @memcpy(out, f32Row(data, r, cols)),
        .f16 => {
            const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
            for (out, w[r * cols ..][0..cols]) |*o, bits| {
                o.* = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => {
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            const bs: usize = switch (t) {
                .q2_k => Q2_K_BLOCK,
                .q3_k => Q3_K_BLOCK,
                .q4_k => Q4_K_BLOCK,
                .q5_k => Q5_K_BLOCK,
                else => Q6_K_BLOCK,
            };
            var vals: [QK_K]f32 = undefined;
            var b: usize = 0;
            while (b * QK_K < cols) : (b += 1) {
                const block = row[b * bs ..][0..bs];
                switch (t) {
                    .q2_k => dequantBlockQ2K(block, &vals),
                    .q3_k => dequantBlockQ3K(block, &vals),
                    .q4_k => dequantBlockQ4K(block, &vals),
                    .q5_k => dequantBlockQ5K(block, &vals),
                    else => dequantBlockQ6K(block, &vals),
                }
                @memcpy(out[b * QK_K ..][0..QK_K], &vals);
            }
        },
        .q5_0 => {
            // embeddings are never q5_0 in practice; go through a 1-row matvec
            // against basis vectors would be wasteful, so walk blocks directly
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            var b: usize = 0;
            while (b * QK_0 < cols) : (b += 1) {
                const block = row[b * Q5_0_BLOCK ..][0..Q5_0_BLOCK];
                const scale = f16FromBytes(block[0..2]);
                const qh = std.mem.readInt(u32, block[2..6], .little);
                const qs = block[6..][0 .. QK_0 / 2];
                const ob = out[b * QK_0 ..][0..QK_0];
                var j: u5 = 0;
                while (true) {
                    const xh0: u8 = @intCast(((qh >> j) << 4) & 0x10);
                    const xh1: u8 = @intCast((qh >> (j + 12)) & 0x10);
                    ob[j] = @as(f32, @floatFromInt(@as(i32, (qs[j] & 0xF) | xh0) - 16)) * scale;
                    ob[@as(usize, j) + QK_0 / 2] = @as(f32, @floatFromInt(@as(i32, (qs[j] >> 4) | xh1) - 16)) * scale;
                    if (j == 15) break;
                    j += 1;
                }
            }
        },
        .q4_1, .q5_1 => {
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            const bs: usize = if (t == .q4_1) Q4_1_BLOCK else Q5_1_BLOCK;
            var b: usize = 0;
            while (b * QK_0 < cols) : (b += 1) {
                const block = row[b * bs ..][0..bs];
                const d = f16FromBytes(block[0..2]);
                const m = f16FromBytes(block[2..4]);
                const qh: u32 = if (t == .q5_1) std.mem.readInt(u32, block[4..8], .little) else 0;
                const qs = block[if (t == .q5_1) 8 else 4..][0 .. QK_0 / 2];
                const ob = out[b * QK_0 ..][0..QK_0];
                var j: u5 = 0;
                while (true) {
                    const xh0: u8 = if (t == .q5_1) @intCast(((qh >> j) << 4) & 0x10) else 0;
                    const xh1: u8 = if (t == .q5_1) @intCast((qh >> (j + 12)) & 0x10) else 0;
                    ob[j] = @as(f32, @floatFromInt((qs[j] & 0xF) | xh0)) * d + m;
                    ob[@as(usize, j) + QK_0 / 2] = @as(f32, @floatFromInt((qs[j] >> 4) | xh1)) * d + m;
                    if (j == 15) break;
                    j += 1;
                }
            }
        },
        .q4_0, .q8_0 => {
            // one-row matvec against basis vectors would be wasteful; walk blocks
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            const bs: usize = if (t == .q4_0) Q4_0_BLOCK else Q8_0_BLOCK;
            var b: usize = 0;
            while (b * QK_0 < cols) : (b += 1) {
                const block = row[b * bs ..][0..bs];
                const scale = f16FromBytes(block[0..2]);
                const ob = out[b * QK_0 ..][0..QK_0];
                if (t == .q4_0) {
                    var j: usize = 0;
                    while (j < QK_0 / 2) : (j += 1) {
                        const byte = block[2 + j];
                        ob[j] = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0x0f)) - 8)) * scale;
                        ob[j + QK_0 / 2] = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8)) * scale;
                    }
                } else {
                    var j: usize = 0;
                    while (j < QK_0) : (j += 1) {
                        ob[j] = @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j])))) * scale;
                    }
                }
            }
        },
        else => {
            // codebook types: decode block by block into `out`
            const blk = blockElems(t);
            const bs = blockBytes(t);
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            var vals: [QK_K]f32 = undefined;
            var b: usize = 0;
            while (b * blk < cols) : (b += 1) {
                const n = dequantBlockCodebook(t, row[b * bs ..][0..bs], &vals);
                @memcpy(out[b * blk ..][0..n], vals[0..n]);
            }
        },
    }
}

// ---- test-only quantizers (reference encoders for kernel validation) --------

fn quantizeQ40(dst: []u8, src: []const f32) void {
    std.debug.assert(src.len % QK_0 == 0);
    var b: usize = 0;
    while (b * QK_0 < src.len) : (b += 1) {
        const in = src[b * QK_0 ..][0..QK_0];
        const block = dst[b * Q4_0_BLOCK ..][0..Q4_0_BLOCK];
        var amax: f32 = 0;
        var vmax: f32 = 0;
        for (in) |v| {
            if (@abs(v) > amax) {
                amax = @abs(v);
                vmax = v;
            }
        }
        const scale: f32 = vmax / -8.0;
        const half: f16 = @floatCast(scale);
        std.mem.writeInt(u16, block[0..2], @bitCast(half), .little);
        const inv: f32 = if (scale != 0) 1.0 / scale else 0;
        var j: usize = 0;
        while (j < QK_0 / 2) : (j += 1) {
            const lo: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(in[j] * inv + 8.5)), 0, 15));
            const hi: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(in[j + QK_0 / 2] * inv + 8.5)), 0, 15));
            block[2 + j] = lo | (hi << 4);
        }
    }
}

fn quantizeQ80(dst: []u8, src: []const f32) void {
    std.debug.assert(src.len % QK_0 == 0);
    var b: usize = 0;
    while (b * QK_0 < src.len) : (b += 1) {
        const in = src[b * QK_0 ..][0..QK_0];
        const block = dst[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
        var amax: f32 = 0;
        for (in) |v| amax = @max(amax, @abs(v));
        const scale: f32 = amax / 127.0;
        const half: f16 = @floatCast(scale);
        std.mem.writeInt(u16, block[0..2], @bitCast(half), .little);
        const inv: f32 = if (scale != 0) 1.0 / scale else 0;
        for (in, 0..) |v, j| {
            block[2 + j] = @bitCast(@as(i8, @intCast(std.math.clamp(@as(i32, @intFromFloat(@round(v * inv))), -127, 127))));
        }
    }
}

test "q4_0 and q8_0 matvec match f32 reference within quant error" {
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const rows = 4;
    const cols = QK_0 * 2;

    var wf: [rows * cols]f32 = undefined;
    for (&wf) |*v| v.* = rnd.float(f32) - 0.5;
    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;

    var ref: [rows]f32 = undefined;
    matvecF32(&ref, std.mem.sliceAsBytes(&wf), &x, rows, cols);

    var q4: [rows * (cols / QK_0) * Q4_0_BLOCK]u8 = undefined;
    var q8: [rows * (cols / QK_0) * Q8_0_BLOCK]u8 = undefined;
    for (0..rows) |r| {
        quantizeQ40(q4[r * rowBytes(.q4_0, cols) ..][0..rowBytes(.q4_0, cols)], wf[r * cols ..][0..cols]);
        quantizeQ80(q8[r * rowBytes(.q8_0, cols) ..][0..rowBytes(.q8_0, cols)], wf[r * cols ..][0..cols]);
    }

    var out4: [rows]f32 = undefined;
    var out8: [rows]f32 = undefined;
    matvec(.q4_0, &out4, &q4, &x, rows, cols);
    matvec(.q8_0, &out8, &q8, &x, rows, cols);

    for (0..rows) |r| {
        try std.testing.expect(@abs(out4[r] - ref[r]) < 0.6); // int4 is lossy
        try std.testing.expect(@abs(out8[r] - ref[r]) < 0.05); // int8 is close
    }
}

test "q4_k dequant with handcrafted block" {
    var block = [_]u8{0} ** Q4_K_BLOCK;
    // d = 1.0, dmin = 0.0
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    // sub-blocks 0..3: sc = 1 (scales[0..4]=1), min = 0 (scales[4..8]=0)
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    // qs = 0x31: low nibble 1, high nibble 3
    for (block[16..]) |*b| b.* = 0x31;

    var vals: [QK_K]f32 = undefined;
    dequantBlockQ4K(&block, &vals);
    // first 128 elems: groups of (32 x low=1, 32 x high=3); last 128: sc=0 -> 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vals[32], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[64], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vals[96], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[128], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[255], 1e-4);

    // matvec against ones = sum = 32*(1+3)*2 = 256
    const x = [_]f32{1.0} ** QK_K;
    var out: [1]f32 = undefined;
    matvec(.q4_k, &out, &block, &x, 1, QK_K);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), out[0], 1e-2);
}

test "q6_k dequant with handcrafted block" {
    var block = [_]u8{0} ** Q6_K_BLOCK;
    // ql=0, qh=0 -> q = -32 everywhere; scales all 1; d = 0.5
    for (block[QK_K / 2 + QK_K / 4 ..][0 .. QK_K / 16]) |*b| b.* = 1;
    std.mem.writeInt(u16, block[QK_K / 2 + QK_K / 4 + QK_K / 16 ..][0..2], @bitCast(@as(f16, 0.5)), .little);

    var vals: [QK_K]f32 = undefined;
    dequantBlockQ6K(&block, &vals);
    for (vals) |v| try std.testing.expectApproxEqAbs(@as(f32, -16.0), v, 1e-4);
}

test "q5_k high bit adds 16" {
    var block = [_]u8{0} ** Q5_K_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    block[4] = 1; // sc(sub-block 0) = 1
    // qh bit0 set for l=0 -> element 0 gets +16; qs all zero
    block[16] = 1;
    var vals: [QK_K]f32 = undefined;
    dequantBlockQ5K(&block, &vals);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), vals[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[1], 1e-4);
}

test "f16 matvec and dequantRow" {
    const cols = 4;
    var bits: [2 * cols]u16 = undefined;
    const vals = [_]f32{ 1.0, -2.0, 0.5, 4.0, 0.25, 3.0, -1.0, 2.0 };
    for (&bits, vals) |*b, v| b.* = @bitCast(@as(f16, @floatCast(v)));
    const x = [_]f32{ 1, 1, 1, 1 };
    var out: [2]f32 = undefined;
    matvec(.f16, &out, std.mem.sliceAsBytes(&bits), &x, 2, cols);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.25), out[1], 1e-3);

    var row: [cols]f32 = undefined;
    dequantRow(.f16, &row, std.mem.sliceAsBytes(&bits), 1, cols);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), row[0], 1e-3);
}

test "tensorBytesChecked rejects overflow and ragged quant rows (issue #29)" {
    // ne0 * 4 must not wrap: 1<<62 f32 elements overflows usize
    try std.testing.expectError(error.Overflow, tensorBytesChecked(.f32, 1 << 62, 4, 1));
    // rows * per_row must not wrap
    try std.testing.expectError(error.Overflow, tensorBytesChecked(.f32, 1 << 40, 1 << 40, 1));
    // a quantized row must be a whole number of blocks, else rowBytes
    // under-counts the bytes the kernel actually walks
    try std.testing.expectError(error.BadTensorShape, tensorBytesChecked(.q4_0, 33, 1, 1));
    // sane shapes still compute
    try std.testing.expectEqual(@as(usize, 4 * 16), try tensorBytesChecked(.f32, 16, 1, 1));
    try std.testing.expectEqual(@as(usize, Q4_0_BLOCK * 2), try tensorBytesChecked(.q4_0, 64, 1, 1));
}

test "matvec agrees with dequantRow for every supported type" {
    // The two paths are independent implementations of the same thing, and
    // some matvec kernels take algebraic shortcuts the dequant path does not:
    // Q4_1/Q5_1 hoist the per-block minimum out of the inner loop as
    // `d*sum(q*x) + m*sum(x)`. That identity is easy to get subtly wrong and
    // impossible to notice by eye, so pin it.
    const gpa = std.testing.allocator;
    const cols = QK_K; // a whole number of blocks for every type here
    const rows = 3;

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    const types = [_]Type{ .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .f32, .f16 };
    for (types) |t| {
        const bytes = tensorBytes(t, cols, rows);
        const data = try gpa.alloc(u8, bytes);
        defer gpa.free(data);
        // Arbitrary bytes are a valid encoding for every one of these formats:
        // each field is a scale, a min or a quantized lane, none of which have
        // invalid bit patterns. Clear one exponent bit of each f16 so no block
        // scale lands on NaN/Inf, which would make the comparison meaningless.
        rnd.bytes(data);
        for (0..bytes / 2) |i| data[i * 2 + 1] &= 0xFB;

        const got = try gpa.alloc(f32, rows);
        defer gpa.free(got);
        matvec(t, got, data, x, rows, cols);

        const row = try gpa.alloc(f32, cols);
        defer gpa.free(row);
        for (0..rows) |r| {
            dequantRow(t, row, data, r, cols);
            var want: f32 = 0;
            var mass: f32 = 0; // sum |w*x|, the size of the terms being summed
            for (row, x) |w, xv| {
                want += w * xv;
                mass += @abs(w * xv);
            }
            // Two different tolerances, for two different reasons.
            //
            // Exact kernels differ from this reference only by summation
            // order, so they get a tight relative bound.
            //
            // The int8-activation kernels quantize x to 8 bits, which is a
            // real approximation of about 0.4% per element. Bounding that
            // against the *result* would be wrong: a dot of random terms
            // cancels almost completely, so the result can be orders of
            // magnitude smaller than the terms and any per-term error looks
            // enormous next to it. The honest bound is against the mass of
            // the terms actually summed. Real activations (post-RMSNorm,
            // same sign structure) cancel far less than this random case.
            const tol = if (usesInt8Activations(t))
                mass * 0.01
            else
                @max(@abs(want), @abs(got[r])) * 1e-4 + 1e-3;
            std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
                std.debug.print("{s} row {d}: matvec {d} vs dequant-dot {d}\n", .{ @tagName(t), r, got[r], want });
                return e;
            };
        }
    }
}

test "parallel matvec produces bit-identical results to inline" {
    // Row-splitting must not perturb a single element: each output row is one
    // independent dot product, so unlike a reduction split there is no
    // reassociation and the result should match exactly, not approximately.
    // Anything else means rows are being mapped to the wrong weight bytes.
    const gpa = std.testing.allocator;
    const cols = QK_K;
    const rows = 257; // deliberately prime-ish: exercises a ragged last block

    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    // Exact kernels only: row-splitting must be bit-identical, but the
    // int8-activation kernels quantize `x` once per matvec call, so their
    // output legitimately depends on how rows are grouped.
    for ([_]Type{ .q5_0, .q6_k, .iq4_xs, .f32 }) |t| {
        const data = try gpa.alloc(u8, tensorBytes(t, cols, rows));
        defer gpa.free(data);
        rnd.bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB; // keep f16 scales finite

        const serial = try gpa.alloc(f32, rows);
        defer gpa.free(serial);
        matvec(t, serial, data, x, rows, cols); // pool not started: inline

        parallelBegin(8);
        defer parallelEnd();
        try std.testing.expect(pool.n > 0); // otherwise this test proves nothing

        const par = try gpa.alloc(f32, rows);
        defer gpa.free(par);
        // repeat: a race shows up intermittently, so a single pass is weak
        for (0..8) |_| {
            @memset(par, std.math.nan(f32));
            matvec(t, par, data, x, rows, cols);
            try std.testing.expectEqualSlices(f32, serial, par);
        }
    }
}

test "matmul is bit-identical to repeated matvec" {
    // Batching changes only the order weights are unpacked in. The activation
    // quantization is per-vector and deterministic, so a batched result must
    // equal the single-vector result exactly -- not approximately. A tolerance
    // here would hide exactly the bugs this is meant to catch: an off-by-one
    // in the per-token activation stride, or a scale applied to the wrong
    // token's lanes.
    const gpa = std.testing.allocator;
    const cols = QK_K * 2;
    const rows = 133;

    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    for ([_]Type{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_0, .f32 }) |t| {
        const data = try gpa.alloc(u8, tensorBytes(t, cols, rows));
        defer gpa.free(data);
        rnd.bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;

        for ([_]usize{ 2, 3, 5, MAX_BATCH }) |n| {
            const xs = try gpa.alloc(f32, n * cols);
            defer gpa.free(xs);
            // Deliberately different magnitudes per token: a shared or stale
            // activation scale would go unnoticed with uniform inputs.
            for (0..n) |k| {
                for (xs[k * cols ..][0..cols]) |*v| {
                    v.* = (rnd.float(f32) - 0.5) * @as(f32, @floatFromInt(k + 1));
                }
            }

            const want = try gpa.alloc(f32, n * rows);
            defer gpa.free(want);
            for (0..n) |k| {
                matvec(t, want[k * rows ..][0..rows], data, xs[k * cols ..][0..cols], rows, cols);
            }

            const got = try gpa.alloc(f32, n * rows);
            defer gpa.free(got);
            matmul(t, got, data, xs, n, rows, cols);
            std.testing.expectEqualSlices(f32, want, got) catch |e| {
                std.debug.print("{s} n={d}\n", .{ @tagName(t), n });
                return e;
            };

            // and again with the worker pool live, which routes through a
            // different code path in runChunks
            parallelBegin(4);
            defer parallelEnd();
            @memset(got, std.math.nan(f32));
            matmul(t, got, data, xs, n, rows, cols);
            try std.testing.expectEqualSlices(f32, want, got);
        }
    }
}

test "non-finite activations cannot trap the quantizers (security issue #165)" {
    // A poisoned tensor decodes to NaN/Inf scales, which flow into
    // activations; @intFromFloat on those is a checked trap in ReleaseSafe
    // and UB otherwise. The quantizers must survive and stay in range.
    var x: [QK_0 * 2]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    x[0] = std.math.nan(f32);
    x[1] = std.math.inf(f32);
    x[2] = -std.math.inf(f32);

    const blocks = quantizeX(x[0..]);
    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    for (blocks) |b| {
        try std.testing.expect(std.math.isFinite(b.d));
        for (b.q) |q| try std.testing.expect(q >= -127 and q <= 127);
    }

    // an all-non-finite block degrades to zeros rather than trapping
    var y: [QK_0]f32 = @splat(std.math.nan(f32));
    const zb = quantizeX(y[0..]);
    try std.testing.expectEqual(@as(f32, 0), zb[0].d);
    for (zb[0].q) |q| try std.testing.expectEqual(@as(i8, 0), q);
}
