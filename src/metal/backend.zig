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
const GemmDims = extern struct { rows: u32, cols: u32, n: u32 };

/// Batched matvec for prefill. Falls back to the CPU for anything the kernel
/// does not cover, so correctness never depends on the shape.
///
/// Unlike decode, this is *not* gated on the calibrated row threshold. That
/// threshold answers a bandwidth question -- at one token per pass the GPU and
/// the CPU are reading the same bus and the crossover is high. A batch reuses
/// each weight n times, so the arithmetic intensity is n times higher and the
/// answer is different; gating prefill on a decode measurement would be
/// applying the wrong number.
pub fn matmul(t: ggml.Type, out: []f32, data: []const u8, xs: []const f32, n: usize, rows: usize, cols: usize) void {
    if (n <= 1) return matvec(t, out, data, xs, rows, cols);
    const cx = &(ctx orelse return cpu.matmul(t, out, data, xs, n, rows, cols));
    if (!calibrating and (!use_gpu_ops or !gemm_worthwhile) and calibrated) return cpu.matmul(t, out, data, xs, n, rows, cols);
    if (t != .q4_k or n > MAX_BATCH or cols % 256 != 0) {
        return cpu.matmul(t, out, data, xs, n, rows, cols);
    }
    if (xs.len * 4 > cx.scratch_x.len or out.len * 4 > cx.scratch_out.len) {
        return cpu.matmul(t, out, data, xs, n, rows, cols);
    }
    const w = wrapFor(cx, data) orelse return cpu.matmul(t, out, data, xs, n, rows, cols);

    @memcpy(cx.scratch_x.slice(f32)[0..xs.len], xs);
    const dims = GemmDims{ .rows = @intCast(rows), .cols = @intCast(cols), .n = @intCast(n) };
    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const cb = cx.dev.commandBuffer();
    cb.dispatch(
        cx.gemm_q4k,
        &.{ w.buf, cx.scratch_x, cx.scratch_out },
        &.{ w.off, 0, 0 },
        std.mem.asBytes(&dims),
        rows * SIMD_W,
        group,
    );
    cb.commitAndWait();
    @memcpy(out, cx.scratch_out.slice(f32)[0..out.len]);
}
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
const dmmv_q6k_src = @embedFile("../shaders/metal/dmmv_q6k.metal");
const gemm_q4k_src = @embedFile("../shaders/metal/gemm_q4k.metal");
const attn_src = @embedFile("../shaders/metal/attn.metal");
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
/// work.
///
/// Lowered from 100,000 after the kernel was widened from one byte per lane
/// to four (see dmmv_q4k.metal). Best-of-20 at cols=2048 on an M5, GPU ms
/// against 8 CPU threads:
///
///     rows     before   after   cpu
///     32,000    0.639   0.612   0.62    -- a tie, so still declined
///     65,536    1.212   0.919   1.30    -- consistent 1.25-1.48x win
///     131,072   2.381   1.609   2.57    -- 1.5-1.6x
const MIN_GPU_ROWS = 65_536;

/// The decision actually consulted at run time. `MIN_GPU_ROWS` above is only
/// the starting guess; `calibrate` replaces it with a measurement taken on the
/// loaded model's own shapes.
///
/// A static threshold cannot be right for two reasons. It was measured on one
/// machine, and the crossover is a property of the ratio between that
/// machine's GPU and its CPU cores -- an 8-core laptop and a 40-core Mac Pro
/// do not share one. And it is per-shape: the same GPU that loses on a 2,048
/// row matvec wins on a 131,072 row one. Measuring beats believing, which is
/// the same argument the fetch path already makes for its tier order.
var min_rows: usize = MIN_GPU_ROWS;
/// Set false when calibration finds the GPU loses at every shape this model
/// uses, so nothing pays the dispatch check.
var gpu_worthwhile: bool = true;
/// Whether the fused FFN block beat the CPU sequence on this model's shape.
var ffn_worthwhile: bool = false;
/// Whether the batched (prefill) kernel beat the CPU on this model's shape.
/// Measured separately from decode: a batch reuses each weight n times, so the
/// arithmetic intensity -- and therefore the answer -- is different.
var gemm_worthwhile: bool = false;
/// Whether fused attention beat the engine's own head loop. Off until
/// measured: one command buffer per layer is a real cost, and at short
/// sequences the CPU loop is trivial.
var attn_worthwhile: bool = false;
var calibrated: bool = false;
/// True only while `calibrate` is running. Without it the gate below refuses
/// the very call that is supposed to measure the block: `calibrate` marks
/// itself done on entry, and `ffn_worthwhile` starts false, so the probe was
/// declined and the block reported "cpu" without ever having been timed.
var calibrating: bool = false;

/// Master switch for acting on the calibration verdict.
///
/// Off by default, and the reason is a measurement failure worth recording.
/// Calibration times each operation in a tight loop, where consecutive
/// `commitAndWait` calls pipeline and the GPU's per-submission latency is
/// largely hidden. In the engine each operation is separated by other work and
/// pays that latency in full. The gap is not subtle: isolated calibration
/// reported the fused FFN block at 1.107 ms against a CPU 9.837 ms and enabled
/// every GPU path, and end-to-end decode then fell from 56 to 9.1 tok/s on the
/// same model -- a 6x regression chosen by a measurement that looked rigorous.
///
/// The verdict is still computed and printed, because it is useful to see. It
/// is simply not acted on until calibration measures a representative
/// sequence -- a whole layer as the engine issues it -- rather than one
/// operation at a time. `--gpu-ops` opts in for experiments.
pub var use_gpu_ops: bool = false;

/// How much faster the GPU must be before it is chosen. 1.25 = 25%.
///
/// Set from observed flapping rather than taste: at 15% the verdict for
/// TinyLlama still changed between identical runs, and an A/B of the two
/// verdicts showed no throughput difference at all (45.1 vs 44.8 tok/s) --
/// i.e. the paths really are tied there and the measurement was resolving
/// noise. A wrong verdict costs every token of the run; a tie resolved to CPU
/// costs nothing.
const MARGIN: f64 = 1.25;
/// Timed repetitions per path. Higher than a microbenchmark would need,
/// because this runs once at load and a wrong verdict costs every token.
const REPS: usize = 24;

/// What calibration concluded, for the startup banner.
pub const Verdict = struct {
    ran: bool,
    matvec_min_rows: usize,
    matvec_used: bool,
    ffn_used: bool,
    prefill_used: bool = false,
    attn_used: bool = false,
    attn_gpu_ms: f64 = 0,
    attn_cpu_ms: f64 = 0,
    ffn_gpu_ms: f64 = 0,
    ffn_cpu_ms: f64 = 0,
};
var verdict: Verdict = .{ .ran = false, .matvec_min_rows = MIN_GPU_ROWS, .matvec_used = false, .ffn_used = false };

pub fn lastVerdict() Verdict {
    return verdict;
}

/// Time the GPU against the CPU on this model's own shapes and record which
/// wins. Called once at model load, before any token is generated.
///
/// `Shape` describes work the engine will actually issue. Calibrating on
/// synthetic sizes would reproduce the problem it is meant to solve: the
/// static threshold was itself a measurement, just one taken on a different
/// machine and a different tensor.
pub const Shape = struct {
    /// Sample weight bytes laid out as the real tensor is. Borrowed.
    data: []const u8,
    ty: ggml.Type,
    rows: usize,
    cols: usize,
};

pub const AttnShape = struct { n_heads: usize, n_kv_heads: usize, hd: usize, seq: usize };

pub fn calibrate(gpa: std.mem.Allocator, dim: usize, ffn: usize, shapes: []const Shape, triple: ?[3]Shape) void {
    if (calibrated) return;
    calibrated = true;
    calibrating = true;
    defer calibrating = false;
    const cx = &(ctx orelse {
        gpu_worthwhile = false;
        return;
    });
    _ = cx;

    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const tio = thr.io();

    const x = gpa.alloc(f32, @max(dim, ffn)) catch return;
    defer gpa.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.01 - 0.08;

    // matvec: the smallest row count at which the GPU actually won. Shapes the
    // model never issues are not consulted, so a model whose largest tensor
    // loses simply turns the GPU off for matvec.
    var best_win: usize = std.math.maxInt(usize);
    for (shapes) |sh| {
        if (sh.ty != .q4_k or sh.cols % 256 != 0) continue;
        const out = gpa.alloc(f32, sh.rows) catch continue;
        defer gpa.free(out);
        if (sh.rows * 4 > cx_scratchOutLen()) continue;

        // `min_rows` is what matvec consults, so moving it is how the same
        // call is steered down each path in turn.
        min_rows = 0;
        const g = timeMatvec(tio, sh, out, x);
        min_rows = std.math.maxInt(usize);
        const c = timeMatvec(tio, sh, out, x);

        // Require a real margin, not a win. The two paths are within noise of
        // each other around the crossover, and best-of-8 on a machine sharing
        // its GPU with the window server flipped this verdict between
        // otherwise identical runs. A tie should resolve to the CPU: it is the
        // path with no dispatch risk, and choosing it costs nothing when the
        // two are equal.
        if (@as(f64, @floatFromInt(g)) * MARGIN < @as(f64, @floatFromInt(c)) and sh.rows < best_win) best_win = sh.rows;
    }
    if (best_win == std.math.maxInt(usize)) {
        gpu_worthwhile = false;
        min_rows = std.math.maxInt(usize);
    } else {
        min_rows = best_win;
    }

    // The fused block is the decision that matters most: it is the only thing
    // that changes how many command buffers a token costs, and on this
    // hardware that dominates every kernel in them.
    if (triple) |tr| blk: {
        const nw = gpa.alloc(f32, dim) catch break :blk;
        defer gpa.free(nw);
        @memset(nw, 1.0);
        const xs = gpa.alloc(f32, dim) catch break :blk;
        defer gpa.free(xs);
        const g = gpa.alloc(f32, ffn) catch break :blk;
        defer gpa.free(g);
        const u = gpa.alloc(f32, ffn) catch break :blk;
        defer gpa.free(u);
        const a = gpa.alloc(f32, ffn) catch break :blk;
        defer gpa.free(a);
        const o = gpa.alloc(f32, dim) catch break :blk;
        defer gpa.free(o);
        const nm = gpa.alloc(f32, dim) catch break :blk;
        defer gpa.free(nm);

        const gw = WeightRef{ .ty = tr[0].ty, .data = tr[0].data };
        const uw = WeightRef{ .ty = tr[1].ty, .data = tr[1].data };
        const dw = WeightRef{ .ty = tr[2].ty, .data = tr[2].data };

        const now = struct {
            fn f(i: std.Io) i128 {
                return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
            }
        }.f;

        for (xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.01;
        if (!ffnBlock(xs, nw, 1e-5, gw, uw, dw, ffn)) break :blk; // declined outright

        var gms: i128 = std.math.maxInt(i64);
        for (0..8) |_| {
            const t0 = now(tio);
            _ = ffnBlock(xs, nw, 1e-5, gw, uw, dw, ffn);
            const dt = now(tio) - t0;
            if (dt < gms) gms = dt;
        }
        var cms: i128 = std.math.maxInt(i64);
        for (0..8) |_| {
            const t0 = now(tio);
            cpu.rmsnorm(nm, xs, nw, 1e-5);
            cpu.matvec(gw.ty, g, gw.data, nm, ffn, dim);
            cpu.matvec(uw.ty, u, uw.data, nm, ffn, dim);
            cpu.swiglu(a, g, u);
            cpu.matvec(dw.ty, o, dw.data, a, dim, ffn);
            cpu.add(xs, o);
            const dt = now(tio) - t0;
            if (dt < cms) cms = dt;
        }
        ffn_worthwhile = @as(f64, @floatFromInt(gms)) * MARGIN < @as(f64, @floatFromInt(cms));
        verdict.ffn_gpu_ms = @as(f64, @floatFromInt(gms)) / 1e6;
        verdict.ffn_cpu_ms = @as(f64, @floatFromInt(cms)) / 1e6;
    }

    // Prefill, measured on the same tensor at a full batch.
    for (shapes) |sh| {
        if (sh.ty != .q4_k or sh.cols % 256 != 0) continue;
        const nb = MAX_BATCH;
        if (sh.rows * nb * 4 > cx_scratchOutLen()) continue;
        const xb = gpa.alloc(f32, nb * sh.cols) catch continue;
        defer gpa.free(xb);
        for (xb, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.01 - 0.05;
        const ob = gpa.alloc(f32, nb * sh.rows) catch continue;
        defer gpa.free(ob);

        const now2 = struct {
            fn f(i: std.Io) i128 {
                return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
            }
        }.f;
        gemm_worthwhile = true;
        for (0..2) |_| matmul(sh.ty, ob, sh.data, xb, nb, sh.rows, sh.cols);
        var gg: i128 = std.math.maxInt(i64);
        for (0..8) |_| {
            const t0 = now2(tio);
            matmul(sh.ty, ob, sh.data, xb, nb, sh.rows, sh.cols);
            const dt = now2(tio) - t0;
            if (dt < gg) gg = dt;
        }
        gemm_worthwhile = false;
        var cc: i128 = std.math.maxInt(i64);
        for (0..8) |_| {
            const t0 = now2(tio);
            cpu.matmul(sh.ty, ob, sh.data, xb, nb, sh.rows, sh.cols);
            const dt = now2(tio) - t0;
            if (dt < cc) cc = dt;
        }
        gemm_worthwhile = @as(f64, @floatFromInt(gg)) * MARGIN < @as(f64, @floatFromInt(cc));
        break;
    }

    const gms_keep = verdict.ffn_gpu_ms;
    const cms_keep = verdict.ffn_cpu_ms;
    verdict = .{
        .ran = true,
        .matvec_min_rows = min_rows,
        .matvec_used = gpu_worthwhile,
        .ffn_used = ffn_worthwhile,
        .prefill_used = gemm_worthwhile,
        .ffn_gpu_ms = gms_keep,
        .ffn_cpu_ms = cms_keep,
    };
}

fn cx_scratchOutLen() usize {
    const cx = &(ctx orelse return 0);
    return cx.scratch_out.len;
}

fn timeMatvec(io: std.Io, sh: Shape, out: []f32, x: []const f32) i128 {
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;
    // Warm properly: the first dispatch on a pipeline pays costs no later one
    // does, and one warm call left enough of that in the first measurement to
    // change the verdict.
    for (0..4) |_| matvec(sh.ty, out, sh.data, x[0..sh.cols], sh.rows, sh.cols);
    var best: i128 = std.math.maxInt(i64);
    for (0..REPS) |_| {
        const t0 = now(io);
        matvec(sh.ty, out, sh.data, x[0..sh.cols], sh.rows, sh.cols);
        const dt = now(io) - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/// Apple GPUs execute in 32-lane SIMD groups; the kernel assigns one per row.
const SIMD_W = 32;
/// SIMD groups per threadgroup. Four gives 128 threads, enough to keep the
/// scheduler fed without pushing register pressure.
const SIMDGROUPS_PER_GROUP = 4;

const Ctx = struct {
    dev: mtl.Device,
    q4k: mtl.Pipeline,
    q6k: mtl.Pipeline,
    gemm_q4k: mtl.Pipeline,
    attn_p: mtl.Pipeline,
    /// Device-resident KV cache, [layers][ctx][kvd]. Allocated by `attnInit`.
    /// It lives here rather than in the engine's State because the whole point
    /// is that it never crosses the bus: only the one new row per layer per
    /// token is written, O(kvd), instead of the whole cache being staged every
    /// step, O(seq*kvd).
    kv_k: ?mtl.Buffer = null,
    kv_v: ?mtl.Buffer = null,
    kv_ctx: usize = 0,
    kv_kvd: usize = 0,
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
    const p6 = dev.pipeline(dmmv_q6k_src, "dmmv_q6k") catch {
        dev.deinit();
        return;
    };
    const pg = dev.pipeline(gemm_q4k_src, "gemm_q4k") catch {
        dev.deinit();
        return;
    };
    const pattn = dev.pipeline(attn_src, "attn_head") catch {
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
    ctx = .{ .dev = dev, .q4k = p, .q6k = p6, .gemm_q4k = pg, .attn_p = pattn, .rmsnorm_p = pn, .swiglu_p = ps, .add_p = pa, .act = act, .scratch_x = sx, .scratch_out = so, .gpa = gpa };
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

const AttnDims = extern struct {
    n_heads: u32,
    n_kv_heads: u32,
    hd: u32,
    seq: u32,
    kvd: u32,
    scale: f32,
};

/// Threadgroup `scores[]` bounds the context this kernel can serve; see
/// attn.metal. Declining beyond it is the only safe option — silently
/// truncating attention produces text that still reads fluently.
const ATTN_MAX_SEQ: usize = 2048;

/// Allocate the device KV cache. Returns false if it will not fit or there is
/// no device, in which case the engine keeps its own cache and the CPU path.
pub fn attnInit(layers: usize, ctx_len: usize, kvd: usize) bool {
    const cx = &(ctx orelse return false);
    if (cx.kv_k != null) return true;
    if (ctx_len > ATTN_MAX_SEQ) return false;
    const bytes = layers * ctx_len * kvd * @sizeOf(f32);
    if (bytes == 0) return false;
    const k = cx.dev.alloc(bytes) catch return false;
    const v = cx.dev.alloc(bytes) catch {
        var kk = k;
        kk.deinit();
        return false;
    };
    @memset(k.slice(f32), 0);
    @memset(v.slice(f32), 0);
    cx.kv_k = k;
    cx.kv_v = v;
    cx.kv_ctx = ctx_len;
    cx.kv_kvd = kvd;
    return true;
}

/// Time fused attention against the engine's own head loop and record which
/// wins. Separate from `calibrate` because it needs the device cache to exist,
/// which is allocated after the model is loaded.
pub fn calibrateAttn(gpa: std.mem.Allocator, n_heads: usize, n_kv_heads: usize, hd: usize, seq: usize) void {
    const cx = &(ctx orelse return);
    if (cx.kv_k == null or seq == 0 or seq > cx.kv_ctx) return;
    const kvd = cx.kv_kvd;

    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const tio = thr.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;

    const q = gpa.alloc(f32, n_heads * hd) catch return;
    defer gpa.free(q);
    for (q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) * 0.01 - 0.09;
    const kc = gpa.alloc(f32, seq * kvd) catch return;
    defer gpa.free(kc);
    const vc = gpa.alloc(f32, seq * kvd) catch return;
    defer gpa.free(vc);
    for (kc, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 23)) * 0.01 - 0.11;
    for (vc, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 29)) * 0.01 - 0.14;
    const out = gpa.alloc(f32, n_heads * hd) catch return;
    defer gpa.free(out);
    const scores = gpa.alloc(f32, seq) catch return;
    defer gpa.free(scores);

    for (0..seq) |t| _ = kvAppend(0, t, kc[t * kvd ..][0..kvd], vc[t * kvd ..][0..kvd]);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const q_per_kv = n_heads / n_kv_heads;

    attn_worthwhile = true;
    for (0..2) |_| _ = attnHeads(0, seq - 1, q, out, n_heads, n_kv_heads, hd, scale);
    var g: i128 = std.math.maxInt(i64);
    for (0..8) |_| {
        const t0 = now(tio);
        _ = attnHeads(0, seq - 1, q, out, n_heads, n_kv_heads, hd, scale);
        const dt = now(tio) - t0;
        if (dt < g) g = dt;
    }
    attn_worthwhile = false;

    var c: i128 = std.math.maxInt(i64);
    for (0..8) |_| {
        const t0 = now(tio);
        for (0..n_heads) |h| {
            const kvh = h / q_per_kv;
            const qh = q[h * hd ..][0..hd];
            for (0..seq) |t| scores[t] = cpu.dotF32(qh, kc[t * kvd + kvh * hd ..][0..hd]) * scale;
            cpu.softmax(scores);
            const oh = out[h * hd ..][0..hd];
            @memset(oh, 0);
            for (0..seq) |t| cpu.axpy(oh, vc[t * kvd + kvh * hd ..][0..hd], scores[t]);
        }
        const dt = now(tio) - t0;
        if (dt < c) c = dt;
    }
    attn_worthwhile = @as(f64, @floatFromInt(g)) * MARGIN < @as(f64, @floatFromInt(c));
    verdict.attn_used = attn_worthwhile;
    verdict.attn_gpu_ms = @as(f64, @floatFromInt(g)) / 1e6;
    verdict.attn_cpu_ms = @as(f64, @floatFromInt(c)) / 1e6;
}

/// Force fused attention off, whatever calibration concluded.
pub fn disableAttn() void {
    attn_worthwhile = false;
}

/// Mirror one cache row into device memory.
///
/// Separate from `attnHeads` on purpose. Prefill and decode are different code
/// paths in the engine (`stepBatch` and `step`), and only one of them used to
/// do the append -- so the device cache held zeros for every prefilled
/// position while the host cache was complete, and decode attended over them.
/// The output stayed fluent and was wrong, which is the worst failure mode
/// available. Appending wherever the host cache is written, independently of
/// who later reads it, is what makes the two impossible to desynchronise.
pub fn kvAppend(li: usize, pos: usize, k_new: []const f32, v_new: []const f32) bool {
    // No reader, no mirror. When fused attention is off the device cache is
    // dead weight, and writing it is not free: keeping it updated through a
    // 576-token prefill is 12,672 writes into device memory and took decode
    // from 56 to 3 tok/s with the GPU otherwise unused.
    if (!attn_worthwhile) return false;
    const cx = &(ctx orelse return false);
    const kb = cx.kv_k orelse return false;
    const vb = cx.kv_v orelse return false;
    if (pos >= cx.kv_ctx or k_new.len != cx.kv_kvd or v_new.len != cx.kv_kvd) return false;
    const row = (li * cx.kv_ctx + pos) * cx.kv_kvd;
    @memcpy(kb.slice(f32)[row..][0..cx.kv_kvd], k_new);
    @memcpy(vb.slice(f32)[row..][0..cx.kv_kvd], v_new);
    return true;
}

/// One layer's grouped-query attention over the device cache: score, softmax
/// and weight-sum every head in one dispatch. The cache must already contain
/// positions 0..=pos, via `kvAppend`.
///
/// Returns false when the backend cannot take it, so the engine falls back to
/// its own path and its own cache stays authoritative.
pub fn attnHeads(
    li: usize,
    pos: usize,
    q: []const f32,
    out: []f32,
    n_heads: usize,
    n_kv_heads: usize,
    hd: usize,
    scale: f32,
) bool {
    const cx = &(ctx orelse return false);
    const kb = cx.kv_k orelse return false;
    const vb = cx.kv_v orelse return false;
    const seq = pos + 1;
    if (!attn_worthwhile) return false;
    if (seq > cx.kv_ctx or n_kv_heads == 0 or n_heads % n_kv_heads != 0) return false;
    if (q.len * 4 > cx.scratch_x.len or out.len * 4 > cx.scratch_out.len) return false;
    @memcpy(cx.scratch_x.slice(f32)[0..q.len], q);

    const dims = AttnDims{
        .n_heads = @intCast(n_heads),
        .n_kv_heads = @intCast(n_kv_heads),
        .hd = @intCast(hd),
        .seq = @intCast(seq),
        .kvd = @intCast(cx.kv_kvd),
        .scale = scale,
    };
    const layer_off = li * cx.kv_ctx * cx.kv_kvd * @sizeOf(f32);
    const group: usize = 64; // two SIMD groups; hd is 64..128 in practice
    const cb = cx.dev.commandBuffer();
    cb.dispatch(
        cx.attn_p,
        &.{ cx.scratch_x, kb, vb, cx.scratch_out },
        &.{ 0, layer_off, layer_off, 0 },
        std.mem.asBytes(&dims),
        n_heads * group,
        group,
    );
    cb.commitAndWait();
    @memcpy(out, cx.scratch_out.slice(f32)[0..out.len]);
    return true;
}

/// The dmmv pipeline for a quantization, or null when there is no kernel for
/// it and the caller must fall back to the CPU.
fn pipelineFor(cx: *Ctx, t: ggml.Type) ?mtl.Pipeline {
    return switch (t) {
        .q4_k => cx.q4k,
        .q6_k => cx.q6k,
        else => null,
    };
}

pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const cx = &(ctx orelse return cpu.matvec(t, out, data, x, rows, cols));
    const pipe = pipelineFor(cx, t);
    if (!use_gpu_ops or !gpu_worthwhile or pipe == null or rows < min_rows or cols % 256 != 0) {
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
        pipe.?,
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

    const saved_use4 = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use4;

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

test "metal q6_k matvec agrees with the exact cpu reference" {
    // Same oracle discipline as the Q4_K test above: dequantize-then-dot in
    // f32, tolerance against the summed mass rather than the result.
    //
    // What this specifically has to catch is the scale split. Q6_K scales are
    // per 16 values while a run is 32 wide, so lanes below and above position
    // 16 in a run use different scales. Folding a run under one scale produces
    // output that still looks like output -- the CPU kernel carries the same
    // warning because it is not a mistake you notice by reading the result.
    const gpa = std.testing.allocator;
    const cols = 2048;
    const rows = 512;

    var prng = std.Random.DefaultPrng.init(0x6C0FFEE);
    const rnd = prng.random();

    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q6_k, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB; // finite f16 scales

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;

    // Force the GPU path regardless of what calibration would have decided.
    const saved_min = min_rows;
    const saved_ok = gpu_worthwhile;
    const saved_use = use_gpu_ops;
    min_rows = 0;
    gpu_worthwhile = true;
    use_gpu_ops = true;
    defer {
        min_rows = saved_min;
        gpu_worthwhile = saved_ok;
        use_gpu_ops = saved_use;
    }

    const got = try gpa.alloc(f32, rows);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    matvec(.q6_k, got, data, x, rows, cols);

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    for (0..rows) |r| {
        cpu.dequantRow(.q6_k, row, data, r, cols);
        var want: f32 = 0;
        var mass: f32 = 0;
        for (row, x) |wv, xv| {
            want += wv * xv;
            mass += @abs(wv * xv);
        }
        const tol = mass * 1e-5;
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("metal q6_k row {d}: reference {d} vs gpu {d} (tol {d})\n", .{ r, want, got[r], tol });
            return e;
        };
    }
}

test "metal batched matmul agrees with the exact cpu reference" {
    // The seam's contract is that matmul equals n calls to matvec. Here the
    // oracle is again dequantize-then-dot in f32 rather than cpu.matmul, whose
    // batched kernels quantize activations to int8 and are approximate by
    // design -- comparing against those reports the CPU's approximation as a
    // GPU bug.
    //
    // Every batch element is checked, not just the first: the failure this is
    // most likely to catch is an indexing slip between the n activation
    // vectors or the n output vectors, which leaves element 0 correct.
    const gpa = std.testing.allocator;
    const cols = 512;
    const rows = 64;
    const n = MAX_BATCH;

    var prng = std.Random.DefaultPrng.init(0x9E11A);
    const rnd = prng.random();

    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;

    const xs = try gpa.alloc(f32, n * cols);
    defer gpa.free(xs);
    for (xs) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;

    const saved_useg = use_gpu_ops;
    const saved_gemm = gemm_worthwhile;
    use_gpu_ops = true;
    gemm_worthwhile = true;
    defer {
        use_gpu_ops = saved_useg;
        gemm_worthwhile = saved_gemm;
    }

    const got = try gpa.alloc(f32, n * rows);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    matmul(.q4_k, got, data, xs, n, rows, cols);

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    for (0..rows) |r| {
        cpu.dequantRow(.q4_k, row, data, r, cols);
        for (0..n) |k| {
            const x = xs[k * cols ..][0..cols];
            var want: f32 = 0;
            var mass: f32 = 0;
            for (row, x) |wv, xv| {
                want += wv * xv;
                mass += @abs(wv * xv);
            }
            const tol = mass * 1e-5;
            std.testing.expectApproxEqAbs(want, got[k * rows + r], tol) catch |e| {
                std.debug.print("metal gemm row {d} batch {d}: reference {d} vs gpu {d}\n", .{ r, k, want, got[k * rows + r] });
                return e;
            };
        }
    }
}

test "metal fused attention matches the cpu head loop" {
    // Checks three things that are each easy to get wrong and hard to see:
    // the GQA head mapping (query head h reads KV head h/(nh/nkv)), the
    // softmax normalisation, and the layer offset into the shared cache. A
    // wrong head mapping still produces well-formed output.
    const gpa = std.testing.allocator;
    const n_heads = 8;
    const n_kv_heads = 2;
    const hd = 64;
    const kvd = n_kv_heads * hd;
    const layers = 3;
    const ctx_len = 128;
    const li = 2; // not layer 0, so a missing layer offset shows up
    const seq = 37; // not a multiple of the threadgroup size
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    if (!attnInit(layers, ctx_len, kvd)) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0xA77E);
    const rnd = prng.random();

    const q = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(q);
    for (q) |*v| v.* = rnd.float(f32) - 0.5;

    // Host mirror of the cache, filled the same way the engine would.
    const kc = try gpa.alloc(f32, seq * kvd);
    defer gpa.free(kc);
    const vc = try gpa.alloc(f32, seq * kvd);
    defer gpa.free(vc);
    for (kc) |*v| v.* = rnd.float(f32) - 0.5;
    for (vc) |*v| v.* = rnd.float(f32) - 0.5;

    const got = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(got);
    // Append every position, as decode does, so the incremental path is what
    // is under test rather than a bulk upload.
    const saved_attn = attn_worthwhile;
    attn_worthwhile = true;
    defer attn_worthwhile = saved_attn;
    for (0..seq) |t| {
        if (!kvAppend(li, t, kc[t * kvd ..][0..kvd], vc[t * kvd ..][0..kvd])) return error.SkipZigTest;
        if (!attnHeads(li, t, q, got, n_heads, n_kv_heads, hd, scale)) return error.SkipZigTest;
    }

    const scores = try gpa.alloc(f32, seq);
    defer gpa.free(scores);
    const q_per_kv = n_heads / n_kv_heads;
    for (0..n_heads) |h| {
        const kvh = h / q_per_kv;
        const qh = q[h * hd ..][0..hd];
        for (0..seq) |t| {
            scores[t] = cpu.dotF32(qh, kc[t * kvd + kvh * hd ..][0..hd]) * scale;
        }
        cpu.softmax(scores);
        var want = [_]f32{0} ** hd;
        for (0..seq) |t| {
            cpu.axpy(&want, vc[t * kvd + kvh * hd ..][0..hd], scores[t]);
        }
        for (0..hd) |i| {
            std.testing.expectApproxEqAbs(want[i], got[h * hd + i], 1e-4) catch |e| {
                std.debug.print("attn head {d} dim {d}: cpu {d} vs gpu {d}\n", .{ h, i, want[i], got[h * hd + i] });
                return e;
            };
        }
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
    if (!calibrating and (!use_gpu_ops or !ffn_worthwhile) and calibrated) return false;
    // Per tensor, not one type for all three. A `Q4_K_M` checkpoint is a
    // mixture: llama.cpp puts gate and up in Q4_K and `ffn_down` in Q6_K (or
    // Q5_1, on DeepSeek). Requiring a single type meant this block declined
    // every real model while accepting synthetic ones, so it looked
    // implemented and never ran.
    const gp = pipelineFor(cx, gate_w.ty) orelse return false;
    const up = pipelineFor(cx, up_w.ty) orelse return false;
    const dp = pipelineFor(cx, down_w.ty) orelse return false;
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
    e.dispatch(gp, &.{ gw.buf, cx.act[1], cx.act[2] }, &.{ gw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), ffn * SIMD_W, group);
    e.dispatch(up, &.{ uw.buf, cx.act[1], cx.act[3] }, &.{ uw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), ffn * SIMD_W, group);
    e.barrier();
    // act = silu(gate) * up, in place over act[2]
    e.dispatch(cx.swiglu_p, &.{ cx.act[2], cx.act[3], cx.act[2] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_ffn), ffn, 64);
    e.barrier();
    // ffn_out = Wd . act
    e.dispatch(dp, &.{ dw.buf, cx.act[2], cx.act[3] }, &.{ dw.off, 0, 0 }, std.mem.asBytes(&dims_down), dim * SIMD_W, group);
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
    std.debug.print("\n  rows      gpu(1 cb)   gpu(amort)      cpu(8t)   GB/s(a)   ratio\n", .{});
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
        // Best-of rather than mean. This machine shares its GPU with the
        // window server, so a mean over 20 runs measures whatever else was
        // drawing at the time -- repeated runs of an unchanged kernel varied
        // by 50%, which is wider than most of the effects being measured.
        var gpu: i128 = std.math.maxInt(i64);
        for (0..N) |_| {
            const t0 = now(io);
            const cb = cx.dev.commandBuffer();
            cb.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), ((rows + 1) / 2) * SIMD_W, group);
            cb.commitAndWait();
            const dt = now(io) - t0;
            if (dt < gpu) gpu = dt;
        }

        // The same kernel with submission amortized: D dispatches in one
        // command buffer. This is the number to compare against another
        // engine's kernel microbenchmark, which will also have amortized it --
        // ZINC reports 259.81 us for a 27 MB shape, less than the ~262 us a
        // single command buffer costs here, so its figures cannot include one.
        // Comparing a per-dispatch number against an amortized one made this
        // kernel look ~3x slower than it is.
        const D = 20;
        var gpu_amort: i128 = std.math.maxInt(i64);
        for (0..N / 2) |_| {
            const t0 = now(io);
            const cb = cx.dev.commandBuffer();
            const e = cb.encoder();
            for (0..D) |_| e.dispatch(cx.q4k, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), ((rows + 1) / 2) * SIMD_W, group);
            e.end();
            cb.commitAndWait();
            const dt = @divTrunc(now(io) - t0, D);
            if (dt < gpu_amort) gpu_amort = dt;
        }

        cpu.matvec(.q4_k, dst, data, x, rows, cols);
        var cpu_ns: i128 = std.math.maxInt(i64);
        for (0..N) |_| {
            const t1 = now(io);
            cpu.matvec(.q4_k, dst, data, x, rows, cols);
            const dt = now(io) - t1;
            if (dt < cpu_ns) cpu_ns = dt;
        }

        const g: f64 = @as(f64, @floatFromInt(gpu)) / 1e6;
        const ga: f64 = @as(f64, @floatFromInt(gpu_amort)) / 1e6;
        const c2: f64 = @as(f64, @floatFromInt(cpu_ns)) / 1e6;
        const bytes: f64 = @floatFromInt(ggml.tensorBytes(.q4_k, cols, rows));
        std.debug.print("  {d:>7}   {d:>8.3} ms   {d:>8.3} ms   {d:>8.3} ms   {d:>6.1}   {d:>5.2}x\n", .{ rows, g, ga, c2, bytes / @as(f64, @floatFromInt(gpu_amort)), c2 / g });
    }
}

test "probe: peak streaming-read bandwidth" {
    // Our dmmv_q4k tops out near 61 GB/s and that was read as the machine's
    // ceiling. That reading is only sound if a kernel doing nothing but
    // streaming reads cannot do better. This does the least work per byte a
    // kernel can: sum a large f32 buffer, one float4 per lane.
    // Skipped by default: it prints, and the build runner's --listen protocol
    // does not tolerate a large stdout write from a test. Run it with
    // `LOOM_BW_PROBE=1 zig build test -Dgpu=metal`.
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    if (std.c.getenv("LOOM_BW_PROBE") == null) return error.SkipZigTest;
    const src =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void stream_sum(device const float4* x [[buffer(0)]],
        \\                       device float* out      [[buffer(1)]],
        \\                       device const uint* n4  [[buffer(2)]],
        \\                       uint gid [[thread_position_in_grid]],
        \\                       uint gsz [[threads_per_grid]]) {
        \\    float4 acc = 0;
        \\    for (uint i = gid; i < n4[0]; i += gsz) acc += x[i];
        \\    out[gid] = acc.x + acc.y + acc.z + acc.w;
        \\}
    ;
    var thr: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer thr.deinit();
    const tio = thr.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;

    var d = mtl.Device.init() catch return error.SkipZigTest;
    defer d.deinit();
    var pipe = d.pipeline(src, "stream_sum") catch return error.SkipZigTest;
    defer pipe.deinit();

    std.debug.print("\n  device {s}\n", .{d.name()});
    for ([_]usize{ 64, 256, 512 }) |mb| {
        const bytes = mb << 20;
        var xb = try d.alloc(bytes);
        defer xb.deinit();
        @memset(xb.slice(f32), 1.0);
        const grid: usize = 1 << 16;
        var ob = try d.alloc(grid * @sizeOf(f32));
        defer ob.deinit();
        var nb = try d.alloc(@sizeOf(u32));
        defer nb.deinit();
        nb.slice(u32)[0] = @intCast(bytes / 16);

        var best: u64 = std.math.maxInt(u64);
        for (0..10) |it| {
            const t0 = now(tio);
            const cb = d.commandBuffer();
            cb.dispatch(pipe, &.{ xb, ob, nb }, &.{ 0, 0, 0 }, null, grid, 256);
            cb.commitAndWait();
            const dt: u64 = @intCast(now(tio) - t0);
            if (it >= 2 and dt < best) best = dt;
        }
        std.debug.print("  {d:>4} MB  {d:>7.3} ms  {d:>6.1} GB/s\n", .{
            mb,
            @as(f64, @floatFromInt(best)) / 1e6,
            @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best)),
        });
    }
}
