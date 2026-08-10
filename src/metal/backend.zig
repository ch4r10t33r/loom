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
    // No `calibrated` term: an uncalibrated process resolves to the CPU, the
    // same way a calibration tie does. Requiring `calibrated` here meant any
    // process that never loads a model -- `loom bench`, tests -- paid a
    // command buffer per call and measured ~1.4 ms of dispatch overhead on a
    // shape the CPU wins outright.
    if (!calibrating and (!use_gpu_ops or !gemm_worthwhile)) return cpu.matmul(t, out, data, xs, n, rows, cols);
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
        // gemm_q4k is one row per SIMD group -- it spends its register budget
        // on MAX_BATCH accumulators rather than a second row, so `groupsFor`,
        // which describes the dmmv kernels, does not apply here.
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
const dmmv_q4k_id_src = @embedFile("../shaders/metal/dmmv_q4k_id.metal");
const dmmv_q5_0_id_src = @embedFile("../shaders/metal/dmmv_q5_0_id.metal");
const dmmv_q8_0_id_src = @embedFile("../shaders/metal/dmmv_q8_0_id.metal");
const dmmv_q6k_src = @embedFile("../shaders/metal/dmmv_q6k.metal");
const gemm_q4k_src = @embedFile("../shaders/metal/gemm_q4k.metal");
const attn_src = @embedFile("../shaders/metal/attn.metal");
const mla_attn_src = @embedFile("../shaders/metal/mla_attn.metal");
const mla_absorb_src = @embedFile("../shaders/metal/mla_absorb.metal");
const mla_rope_src = @embedFile("../shaders/metal/mla_rope.metal");
const rope_src = @embedFile("../shaders/metal/rope.metal");
const moe_acc_src = @embedFile("../shaders/metal/moe_acc.metal");
const moe_route_src = @embedFile("../shaders/metal/moe_route.metal");
const dmmv_q5_1_src = @embedFile("../shaders/metal/dmmv_q5_1.metal");
const dmmv_q5_0_src = @embedFile("../shaders/metal/dmmv_q5_0.metal");
const dmmv_q8_0_src = @embedFile("../shaders/metal/dmmv_q8_0.metal");
const dmmv_f32_src = @embedFile("../shaders/metal/dmmv_f32.metal");
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
/// Whether the device KV cache has a reader, and so must be kept in step with
/// the engine's own. Fused attention is one reader; the recorded layer path is
/// another, and it does not imply the first -- calibration can decline fused
/// attention as a standalone operation while the layer path still uses it
/// inside a frame. Gating the mirror on the wrong one leaves the device cache
/// holding zeros for every prefilled position, which produces fluent, wrong
/// output rather than an error.
var kv_mirror: bool = false;
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
    q5_1: mtl.Pipeline,
    /// Q4_K with two rows per SIMD group, for matrices large enough that group
    /// count matters more than activation reuse. See `dmmvFor`.
    q4k_wide: mtl.Pipeline,
    /// Q4_K reading its expert plane from a device-side id buffer.
    q4k_id: mtl.Pipeline,
    q5_0_id: mtl.Pipeline,
    q8_0_id: mtl.Pipeline,
    route_p: mtl.Pipeline,
    /// ids and gates the routing kernel writes; the expert kernels read them
    /// without either ever reaching the host.
    route_ids: ?mtl.Buffer = null,
    route_gates: ?mtl.Buffer = null,
    q5_0: mtl.Pipeline,
    q8_0: mtl.Pipeline,
    f32p: mtl.Pipeline,
    gemm_q4k: mtl.Pipeline,
    attn_p: mtl.Pipeline,
    mla_attn_p: mtl.Pipeline,
    mla_wsum_p: mtl.Pipeline,
    mla_absorb_p: mtl.Pipeline,
    mla_rope_p: mtl.Pipeline,
    /// Per layer: `kv_b` dequantized to f32, device-resident.
    mla_wk: std.ArrayListUnmanaged(mtl.Buffer) = .empty,
    /// Identity ids 0..n_heads for the W_v dispatch: every head "selects" its
    /// own plane, which turns 16 per-head host matvecs into one id-kernel
    /// dispatch inside the same command buffer.
    mla_vids: ?mtl.Buffer = null,
    rope_p: mtl.Pipeline,
    kvw_p: mtl.Pipeline,
    sadd_p: mtl.Pipeline,
    reduce_p: mtl.Pipeline,
    reduce_dev_p: mtl.Pipeline,
    zero_p: mtl.Pipeline,
    /// Device-resident KV cache, [layers][ctx][kvd]. Allocated by `attnInit`.
    /// It lives here rather than in the engine's State because the whole point
    /// is that it never crosses the bus: only the one new row per layer per
    /// token is written, O(kvd), instead of the whole cache being staged every
    /// step, O(seq*kvd).
    /// Compressed MLA cache: one c_kv row and one rope key per position, shared
    /// by every head. Separate from `kv_k`/`kv_v` because MLA never
    /// materializes keys or values -- these are ~576 floats per position where
    /// a dense cache would be n_heads * (nope + v_head_dim).
    mla_c: ?mtl.Buffer = null,
    mla_krope: ?mtl.Buffer = null,
    mla_ctx: usize = 0,
    mla_kvr: usize = 0,
    mla_rope: usize = 0,
    kv_k: ?mtl.Buffer = null,
    kv_v: ?mtl.Buffer = null,
    /// Per-layer norm weights, all layers resident at once.
    ///
    /// Recording makes host writes to a *shared* scratch buffer wrong in a way
    /// immediate submission never was: `layerBlock` used to copy each layer's
    /// norms into `scratch_x` and record dispatches reading it, but the
    /// dispatches do not run until `endFrame`, by which point every layer's
    /// dispatch reads whatever the *last* layer wrote. Twenty-two layers all
    /// used layer 21's norms, and the model produced confident nonsense. Each
    /// layer owning a slot removes the aliasing entirely.
    norms: mtl.Buffer,
    kv_ctx: usize = 0,
    kv_kvd: usize = 0,
    rmsnorm_p: mtl.Pipeline,
    swiglu_p: mtl.Pipeline,
    add_p: mtl.Pipeline,
    copy_p: mtl.Pipeline,
    swiglu_slots_p: mtl.Pipeline,
    /// Device-resident activation scratch. The whole point of keeping these
    /// on the GPU is that a block of work can be encoded into one command
    /// buffer without the host reading anything in between.
    /// Device-resident activation slots. Eight rather than four because a
    /// whole layer has to be recordable without touching the host: x, normed,
    /// q, k, v, attn_out, gate, up. Anything that has to come back to the CPU
    /// mid-layer ends the command buffer, which is the cost the whole design
    /// exists to avoid.
    act: [8]mtl.Buffer,
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
    const prope = dev.pipeline(rope_src, "rope_apply") catch {
        dev.deinit();
        return;
    };
    const pkvw = dev.pipeline(rope_src, "kv_write") catch {
        dev.deinit();
        return;
    };
    const p4id = dev.pipeline(dmmv_q4k_id_src, "dmmv_q4k_id") catch {
        dev.deinit();
        return;
    };
    const p50id = dev.pipeline(dmmv_q5_0_id_src, "dmmv_q5_0_id") catch {
        dev.deinit();
        return;
    };
    const p80id = dev.pipeline(dmmv_q8_0_id_src, "dmmv_q8_0_id") catch {
        dev.deinit();
        return;
    };
    const p4w = dev.pipeline("#define NR0 2\n" ++ dmmv_q4k_src, "dmmv_q4k") catch {
        dev.deinit();
        return;
    };
    const p51 = dev.pipeline(dmmv_q5_1_src, "dmmv_q5_1") catch {
        dev.deinit();
        return;
    };
    const p50 = dev.pipeline(dmmv_q5_0_src, "dmmv_q5_0") catch {
        dev.deinit();
        return;
    };
    const pf32 = dev.pipeline(dmmv_f32_src, "dmmv_f32") catch {
        dev.deinit();
        return;
    };
    const p80 = dev.pipeline(dmmv_q8_0_src, "dmmv_q8_0") catch {
        dev.deinit();
        return;
    };
    const pmla = dev.pipeline(mla_attn_src, "mla_attn_head") catch {
        dev.deinit();
        return;
    };
    const pwsum = dev.pipeline(mla_attn_src, "mla_attn_wsum") catch {
        dev.deinit();
        return;
    };
    const pmrope = dev.pipeline(mla_rope_src, "mla_rope") catch {
        dev.deinit();
        return;
    };
    const pabs = dev.pipeline(mla_absorb_src, "mla_absorb") catch {
        dev.deinit();
        return;
    };
    const psadd = dev.pipeline(moe_acc_src, "scaled_add") catch {
        dev.deinit();
        return;
    };
    const proute = dev.pipeline(moe_route_src, "moe_route") catch {
        dev.deinit();
        return;
    };
    const predd = dev.pipeline(moe_acc_src, "moe_reduce_dev") catch {
        dev.deinit();
        return;
    };
    const pred = dev.pipeline(moe_acc_src, "moe_reduce") catch {
        dev.deinit();
        return;
    };
    const pzero = dev.pipeline(moe_acc_src, "zero_fill") catch {
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
    const pss = dev.pipeline(elementwise_src, "swiglu_slots") catch {
        dev.deinit();
        return;
    };
    const pcp = dev.pipeline(elementwise_src, "copy_f32") catch {
        dev.deinit();
        return;
    };
    const pa = dev.pipeline(elementwise_src, "add_inplace") catch {
        dev.deinit();
        return;
    };
    var act: [8]mtl.Buffer = undefined;
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
    const nb = dev.alloc(1 << 22) catch {
        dev.deinit();
        return;
    };
    const so = dev.alloc(1 << 22) catch {
        dev.deinit();
        return;
    };
    ctx = .{ .dev = dev, .q4k = p, .q6k = p6, .q5_1 = p51, .q4k_wide = p4w, .q4k_id = p4id, .q5_0_id = p50id, .q8_0_id = p80id, .q5_0 = p50, .q8_0 = p80, .f32p = pf32, .gemm_q4k = pg, .attn_p = pattn, .mla_attn_p = pmla, .mla_wsum_p = pwsum, .mla_absorb_p = pabs, .mla_rope_p = pmrope, .rope_p = prope, .kvw_p = pkvw, .sadd_p = psadd, .reduce_p = pred, .reduce_dev_p = predd, .route_p = proute, .zero_p = pzero, .rmsnorm_p = pn, .swiglu_p = ps, .add_p = pa, .copy_p = pcp, .swiglu_slots_p = pss, .act = act, .scratch_x = sx, .scratch_out = so, .norms = nb, .gpa = gpa };
}

pub fn parallelEnd() void {
    cpu.parallelEnd();
    // The device is kept alive across requests: compiling shaders and creating
    // buffers per generation would cost more than it saves.
}

/// Page-aligned MTLBuffer covering `data`, plus the offset of `data` within
/// it. Cached per mapping so a model produces one buffer, not one per tensor.
/// A weight tensor resolved to its MTLBuffer and byte offset within it.
pub const Wrapped = struct { buf: mtl.Buffer, off: usize };

/// A registered region, before the device exists to wrap it.
const Region = struct { base: usize, len: usize, done: bool = false };

/// One device allocation covering part of a region.
const Chunk = struct { base: usize, len: usize, buf: mtl.Buffer };

/// Module-level rather than part of `Ctx`, because a store is mapped while the
/// node is still assembling itself and the device is not up until the model
/// loads. Registering into the context would have silently done nothing -- and
/// did, until the page-in count refused to move.
var regions: [MAX_ARENAS]?Region = @splat(null);
var chunks: [MAX_CHUNKS]?Chunk = @splat(null);

/// One per weight store. Two so a node serving its own GGUF *and* a sharded
/// store is not a special case.
const MAX_ARENAS = 2;

/// A region is split across allocations because `maxBufferLength` is around
/// half of physical RAM; eight covers any store this machine could map.
const MAX_CHUNKS = 8;

/// Why the last `materializeArenas` failed, for the caller to report. A
/// mapping that silently declines to become resident is the whole bug this
/// code exists to fix, so the reason has to be reachable.
pub var arena_error: ?[]const u8 = null;

/// Take a whole mapping, so every tensor inside it is reachable by offset.
///
/// The per-slice cache below keys on the page-aligned base of whatever it is
/// handed. For a GGUF read as one mapping that already collapses to a single
/// buffer, but a sharded expert store is read one expert extent at a time, and
/// those bases are all different -- so the GPU ends up holding thousands of
/// separate allocations over the same file, none of which keeps the model
/// resident. Measured against llama.cpp on the same 10.4 GB checkpoint: the
/// mapped-and-resident arrangement runs at 46.8-58.6 tok/s and faults in 249 MB
/// over a generation, where per-extent wrapping faulted 6,728 MB and managed 2.5.
///
/// Registration only records the region; the device may not exist yet.
pub fn registerArena(mem: []const u8) bool {
    const page = std.heap.pageSize();
    const base = std.mem.alignBackward(usize, @intFromPtr(mem.ptr), page);
    const end = std.mem.alignForward(usize, @intFromPtr(mem.ptr) + mem.len, page);
    for (&regions) |*slot| {
        if (slot.*) |r| {
            // Already covered: registering twice is not an error, and the
            // second buffer would be the one wasting address space.
            if (base >= r.base and end <= r.base + r.len) return true;
            continue;
        }
        slot.* = .{ .base = base, .len = end - base };
        return true;
    }
    return false;
}

/// Wrap every registered region now that the device exists, and report how many
/// bytes are device-resident.
///
/// Worth doing eagerly and worth reporting: the first version deferred
/// everything to first use and registered into a context that did not exist
/// yet, so it silently did nothing and only the page-in counter disagreed.
/// Command buffers submitted since the last call, and reset. The engine's
/// profiler reports it per token.
pub fn takeCmdBufCount() usize {
    const n = mtl.Device.cmdbufs;
    mtl.Device.cmdbufs = 0;
    return n;
}

pub fn materializeArenas() usize {
    const cx = &(ctx orelse {
        arena_error = "no metal device";
        return 0;
    });
    const page = std.heap.pageSize();
    // Split at the device's own limit, which is around half of physical RAM:
    // a 9.7 GB store is two allocations on a 16 GB machine and one on a 64 GB
    // one, and asking for it whole simply fails.
    const max_one = std.mem.alignBackward(usize, cx.dev.maxBufferLen(), page);
    if (max_one == 0) {
        arena_error = "device reports no maximum buffer length";
        return 0;
    }

    var total: usize = 0;
    for (&regions) |*slot| {
        const r = if (slot.*) |*x| x else continue;
        if (r.done) {
            total += r.len;
            continue;
        }
        var off: usize = 0;
        while (off < r.len) {
            const take = @min(max_one, r.len - off);
            const ci = freeChunk() orelse {
                arena_error = "too many weight regions to make resident";
                return total;
            };
            const buf = cx.dev.wrapHost(@as([*]const u8, @ptrFromInt(r.base + off))[0..take]) catch |e| {
                arena_error = @errorName(e);
                return total;
            };
            // Wrapping alone only makes the pages addressable by the GPU; they
            // stay as evictable as any other file mapping, which for a model
            // larger than the free page cache means faulting the working set
            // back in every token. Residency is the half that matters.
            _ = cx.dev.makeResident(buf);
            chunks[ci] = .{ .base = r.base + off, .len = take, .buf = buf };
            off += take;
        }
        r.done = true;
        total += r.len;
    }
    if (total > 0) arena_error = null;
    return total;
}

fn freeChunk() ?usize {
    for (chunks, 0..) |c, i| {
        if (c == null) return i;
    }
    return null;
}

fn wrapFor(cx: *Ctx, data: []const u8) ?Wrapped {
    // A registered mapping covers the slice outright, and hits here before any
    // allocation happens.
    const addr = @intFromPtr(data.ptr);
    for (chunks) |slot| {
        const ch = slot orelse continue;
        // Whole slice inside one chunk, or not at all: a tensor straddling a
        // split has no single buffer to name, and the per-slice path below
        // handles the handful that do.
        if (addr >= ch.base and addr + data.len <= ch.base + ch.len) {
            return .{ .buf = ch.buf, .off = addr - ch.base };
        }
    }

    const page = std.heap.pageSize();
    const base = std.mem.alignBackward(usize, @intFromPtr(data.ptr), page);
    const end = std.mem.alignForward(usize, @intFromPtr(data.ptr) + data.len, page);
    const gop = cx.wrapped.getOrPut(cx.gpa, base) catch return null;
    const off = @intFromPtr(data.ptr) - base;

    // A cached entry can be stale in three ways. Two are handled below: a
    // later tensor in the same mapping extending past the region wrapped
    // first, and a freed allocation replaced by a *larger* one on the same
    // page base. The third is the contract, because it cannot be detected
    // from here: a freed allocation replaced by a smaller-or-equal one lands
    // on the same base, the size check passes, and the cached MTLBuffer's GPU
    // mapping may still reference the old physical pages while the CPU's
    // virtual address maps new ones. No CPU-side read can see that -- the
    // buffer *is* the host memory through this address, so any comparison
    // reads the same bytes on both sides. (A content check was tried; it is
    // vacuous for exactly this reason.)
    //
    // So the contract: memory handed to `wrapFor` stays alive for the life of
    // the context. Model weights and store mappings do. A caller that frees
    // and reallocates -- a test looping over fixtures was the one that hit it
    // -- gets well-formed stale results with no error, which presented as an
    // expert kernel computing slot 1 badly while slot 0 was right, and cost a
    // debugging session before the kernel was exonerated.
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

const RopeDims = extern struct {
    n_heads: u32,
    hd: u32,
    rope_dim: u32,
    pos: u32,
    base: f32,
    neox: u32,
};

const KvWriteDims = extern struct { kvd: u32, row: u32 };

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
const ATTN_MAX_SEQ: usize = 4096;

/// Allocate the device KV cache. Returns false if it will not fit or there is
/// no device, in which case the engine keeps its own cache and the CPU path.
pub fn attnInit(layers: usize, ctx_len: usize, kvd: usize) bool {
    const cx = &(ctx orelse return false);
    // Reallocate when the shape differs rather than keeping whatever was
    // allocated first. Returning the existing cache unconditionally is right
    // for a second call with the same model and silently wrong for a different
    // one: the consumer's `kvd != cx.kv_kvd` check then declines forever, or
    // worse, matches by coincidence and indexes a cache laid out for another
    // geometry.
    if (cx.kv_k != null) {
        if (cx.kv_ctx == ctx_len and cx.kv_kvd == kvd) return true;
        if (cx.kv_k) |*b| b.deinit();
        if (cx.kv_v) |*b| b.deinit();
        cx.kv_k = null;
        cx.kv_v = null;
    }
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

// ---- frame: one command buffer for many operations ---------------------------

/// An open recording frame. Operations encode into its encoder and nothing is
/// committed until `endFrame`, so a whole token can cost one submission
/// instead of one per operation.
///
/// This is the entire point of the exercise. A command buffer costs ~262 us on
/// this machine against ~18 us for a matvec kernel, and ZINC -- whose kernels
/// are at parity with loom's -- drops from 53 to 11.8 tok/s when its timing
/// probe forces a commit between dispatches. Loom issued ~150 command buffers
/// per token before this existed.
const Frame = struct {
    cb: mtl.CommandBuffer,
    enc: mtl.Encoder,
    dispatches: u32 = 0,
};
var frame: ?Frame = null;
/// Command buffers and dispatches submitted since the last reset, so the
/// per-token count is observable rather than inferred. ZINC tracks the same
/// numbers for the same reason: without them a 2.89 -> 1.89 change cannot be
/// told from noise.
pub var frames_submitted: u64 = 0;
pub var dispatches_submitted: u64 = 0;

pub fn beginFrame() bool {
    const cx = &(ctx orelse return false);
    if (frame != null) return true;
    const cb = cx.dev.commandBuffer();
    frame = .{ .cb = cb, .enc = cb.encoder() };
    return true;
}

/// Close and submit the frame. Blocking: the caller needs the results.
pub fn endFrame() void {
    if (frame) |*f| {
        f.enc.end();
        f.cb.commitAndWait();
        frames_submitted += 1;
        dispatches_submitted += f.dispatches;
        frame = null;
    }
}

pub fn frameOpen() bool {
    return frame != null;
}

/// Free the device KV cache. Called when calibration decides against every
/// path that would read it.
///
/// Not tidiness: the cache is sized layers x ctx x kvd and reaches hundreds of
/// megabytes on a 7B model. Allocating it, measuring, declining and then
/// holding it adds exactly that much pressure to a machine whose verdict was
/// most likely "no win" *because* it is short of memory.
pub fn releaseKvCache() void {
    const cx = &(ctx orelse return);
    if (cx.kv_k) |*b| b.deinit();
    if (cx.kv_v) |*b| b.deinit();
    cx.kv_k = null;
    cx.kv_v = null;
    cx.kv_ctx = 0;
    cx.kv_kvd = 0;
    kv_mirror = false;
    attn_worthwhile = false;
}

/// Declare that something will read the device KV cache, so `kvAppend` starts
/// mirroring. Called when the recorded layer path is enabled.
pub fn enableKvMirror() void {
    kv_mirror = true;
}

/// Whether a device KV cache exists. Without one the recorded path cannot run
/// at all, so there is nothing to measure.
pub fn hasKvCache() bool {
    const cx = &(ctx orelse return false);
    return cx.kv_k != null;
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
    // No reader, no mirror. An unread device cache is dead weight and writing
    // it is not free: keeping it updated through a 576-token prefill is 12,672
    // device writes, which took decode from 56 to 3 tok/s with the GPU
    // otherwise unused.
    if (!kv_mirror and !attn_worthwhile) return false;
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

/// Size the device-resident compressed MLA cache. False when it will not fit
/// the kernel's bounds, in which case the engine keeps its host cache and this
/// path is simply not used.
pub fn mlaInit(layers: usize, ctx_len: usize, kvr: usize, rope: usize) bool {
    // Records the shape; the buffers are made when there is a device to make
    // them on. The model loads before the Metal context is brought up, so
    // allocating here returned false and every attention call then declined --
    // silently, because declining is also what an unsupported shape does. That
    // is exactly what `registerArena` got wrong first, and this is its shape.
    if (ctx_len > ATTN_MAX_SEQ) {
        mla_want = null;
        return false;
    }
    mla_want = .{ .layers = layers, .ctx_len = ctx_len, .kvr = kvr, .rope = rope };
    return true;
}

/// Allocate the recorded cache, once. False when nothing was recorded or the
/// device will not give the memory.
fn ensureMla(cx: *Ctx) bool {
    if (cx.mla_c != null) return true;
    const w = mla_want orelse return false;
    const c_len = w.layers * w.ctx_len * w.kvr * @sizeOf(f32);
    const r_len = w.layers * w.ctx_len * w.rope * @sizeOf(f32);
    var cb = cx.dev.alloc(c_len) catch return false;
    const rb = cx.dev.alloc(r_len) catch {
        cb.deinit();
        return false;
    };
    cx.mla_c = cb;
    cx.mla_krope = rb;
    cx.mla_ctx = w.ctx_len;
    cx.mla_kvr = w.kvr;
    cx.mla_rope = w.rope;
    return true;
}

/// Append one position's compressed cache row. The host keeps its own copy:
/// this path can decline at any point and the engine's cache has to be the
/// authoritative one when it does -- a device mirror written by one path and
/// not another is how the GQA cache once served zeros at prefilled positions,
/// which reads as fluent, wrong text.
pub fn mlaAppend(li: usize, pos: usize, c_kv: []const f32, k_rope: []const f32) bool {
    const cx = &(ctx orelse return false);
    if (!ensureMla(cx)) return false;
    const cb = cx.mla_c orelse return false;
    const rb = cx.mla_krope orelse return false;
    if (pos >= cx.mla_ctx or c_kv.len != cx.mla_kvr or k_rope.len != cx.mla_rope) return false;
    const ci = (li * cx.mla_ctx + pos) * cx.mla_kvr;
    const ri = (li * cx.mla_ctx + pos) * cx.mla_rope;
    @memcpy(cb.slice(f32)[ci..][0..c_kv.len], c_kv);
    @memcpy(rb.slice(f32)[ri..][0..k_rope.len], k_rope);
    return true;
}

pub fn hasMlaCache() bool {
    const cx = &(ctx orelse return false);
    return ensureMla(cx);
}

/// Hand one layer's `kv_b`, already dequantized to f32, to the device. Layers
/// must arrive in order; returns false if the device will not take it, in
/// which case the host absorption stays in use for every layer.
pub fn mlaSetWk(li: usize, wk_f32: []const f32) bool {
    const cx = &(ctx orelse return false);
    if (li != cx.mla_wk.items.len) return false;
    // Stored f16: half the per-token absorb traffic, and rounding a weight
    // that was quantized to ~4.5 bits anyway is noise against that.
    const b = cx.dev.alloc(wk_f32.len * @sizeOf(f16)) catch return false;
    const dst = b.slice(f16)[0..wk_f32.len];
    for (dst, wk_f32) |*o, v| o.* = @floatCast(v);
    cx.mla_wk.append(cx.gpa, b) catch return false;
    return true;
}

/// q_abs = W_k^T q_nope for every head of one layer, in one dispatch. False
/// when this layer's W_k was never uploaded, so the caller runs its host loop.
/// Every head's MLA attention for one layer, over the compressed cache.
///
/// `q_absorbed` is W_k^T q_nope per head -- the absorption identity, so the
/// kernel dots against the cache directly instead of rebuilding keys. `out` is
/// o_latent per head, still compressed; W_v is applied to it by the caller.
pub fn mlaAttnHeads(
    li: usize,
    pos: usize,
    q_nope: []const f32,
    q_rope: []const f32,
    kv_b: WeightRef,
    out: []f32, // n_heads * v_head_dim: W_v applied, not o_latent
    n_heads: usize,
    nope: usize,
    v_head_dim: usize,
    scale: f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!ensureMla(cx)) return false;
    const cbuf = cx.mla_c orelse return false;
    const rbuf = cx.mla_krope orelse return false;
    // `attn_worthwhile` is set by the GQA attention calibration, which never
    // runs for an MLA model -- deepseek passes zeros for the attention shapes.
    // Until this path has a calibration of its own it is opt-in via --gpu-ops,
    // rather than silently off (which is what it was) or silently on (which
    // would be a speed claim nobody measured).
    if (!use_gpu_ops and !attn_worthwhile) return false;
    const seq = pos + 1;
    if (seq > cx.mla_ctx) return false;
    // The absorption runs here too, in the same command buffer, and its result
    // never comes back to the host. Two dispatches sharing one buffer rather
    // than two buffers each waited on: 26 ops a layer is not the problem,
    // 26 *submissions* is -- ggml encodes an entire graph into about two
    // command buffers and waits once, where each of these was one and a wait.
    if (li >= cx.mla_wk.items.len) return false;
    const kvr = cx.mla_kvr;
    if ((q_nope.len + q_rope.len) * 4 > cx.scratch_x.len) return false;
    if (out.len != n_heads * v_head_dim) return false;
    if (out.len * 4 > cx.scratch_out.len) return false;
    if (n_heads * kvr * 4 > cx.act[1].len or n_heads * kvr * 4 > cx.act[2].len) return false;

    // W_v as one id-kernel dispatch with identity ids: head h's value rows are
    // a plane at stride (nope + v_head_dim) rows, offset nope rows in, and its
    // input is its own o_latent -- exactly the per-slot x-stride shape built
    // for the MoE down projection. This replaces sixteen per-head host
    // matvecs, which were the last host step inside attention.
    const vsel = dmmvIdFor(cx, kv_b.ty) orelse return false;
    if (kvr % colsMultiple(kv_b.ty) != 0) return false;
    if (v_head_dim % vsel.per != 0) return false;
    const vw = wrapFor(cx, kv_b.data) orelse return false;
    const row_bytes = kv_b.data.len / (n_heads * (nope + v_head_dim));
    if (cx.mla_vids == null) {
        const b = cx.dev.alloc(64 * @sizeOf(u32)) catch return false;
        for (0..64) |k| b.slice(u32)[k] = @intCast(k);
        cx.mla_vids = b;
    }

    @memcpy(cx.scratch_x.slice(f32)[0..q_nope.len], q_nope);
    @memcpy(cx.scratch_x.slice(f32)[q_nope.len..][0..q_rope.len], q_rope);

    const ad = AbsorbDims{
        .n_heads = @intCast(n_heads),
        .nope = @intCast(nope),
        .kvr = @intCast(kvr),
        .stride = @intCast(nope + v_head_dim),
        .q_stride = @intCast(nope),
        .q_off = 0,
    };

    const dims = MlaDims{
        .n_heads = @intCast(n_heads),
        .kvr = @intCast(cx.mla_kvr),
        .rope = @intCast(cx.mla_rope),
        .seq = @intCast(seq),
        .qr_stride = @intCast(cx.mla_rope),
        .qr_off = 0,
        .scale = scale,
    };
    const c_off = li * cx.mla_ctx * cx.mla_kvr * @sizeOf(f32);
    const r_off = li * cx.mla_ctx * cx.mla_rope * @sizeOf(f32);
    const group: usize = 64;
    const cb = cx.dev.commandBuffer();
    const e = cb.encoder();
    // q_abs = W_k^T q_nope, into act[1], staying on the device.
    e.dispatch(
        cx.mla_absorb_p,
        &.{ cx.mla_wk.items[li], cx.scratch_x, cx.act[1] },
        &.{ 0, 0, 0 },
        std.mem.asBytes(&ad),
        absorbGrid(n_heads, kvr),
        SIMD_W * SIMDGROUPS_PER_GROUP,
    );
    e.barrier();
    e.dispatch(
        cx.mla_attn_p,
        &.{ cx.act[1], cx.scratch_x, cbuf, rbuf, cx.act[3] },
        &.{ 0, q_nope.len * 4, c_off, r_off, 0 },
        std.mem.asBytes(&dims),
        n_heads * group,
        group,
    );
    e.barrier();
    e.dispatch(cx.mla_wsum_p, &.{ cx.act[3], cbuf, cx.act[2] }, &.{ 0, c_off, 0 }, std.mem.asBytes(&dims), absorbGrid(n_heads, kvr), SIMD_W * SIMDGROUPS_PER_GROUP);
    e.barrier();
    const vd_dims = IdDims{
        .rows = @intCast(v_head_dim),
        .cols = @intCast(kvr),
        .n_used = @intCast(n_heads),
        .plane_stride = @intCast((nope + v_head_dim) * row_bytes),
        .x_stride = @intCast(kvr),
    };
    const vthreads = ((n_heads * v_head_dim + vsel.per - 1) / vsel.per) * SIMD_W;
    const vgrid = ((vthreads + group - 1) / group) * group;
    e.dispatch(
        vsel.pipe,
        &.{ vw.buf, cx.act[2], cx.scratch_out, cx.mla_vids.? },
        &.{ vw.off + nope * row_bytes, 0, 0, 0 },
        std.mem.asBytes(&vd_dims),
        vgrid,
        group,
    );
    e.end();
    cb.commitAndWait();
    @memcpy(out, cx.scratch_out.slice(f32)[0..out.len]);
    return true;
}

/// The id-indexed dmmv pipeline for a quantization, and the rows one SIMD
/// group covers. Null when the type has no id variant, in which case that
/// tensor keeps host-supplied plane pointers.
fn dmmvIdFor(cx: *Ctx, t: ggml.Type) ?struct { pipe: mtl.Pipeline, per: usize } {
    return switch (t) {
        .q4_k => .{ .pipe = cx.q4k_id, .per = 4 },
        .q5_0 => .{ .pipe = cx.q5_0_id, .per = 1 },
        .q8_0 => .{ .pipe = cx.q8_0_id, .per = 1 },
        else => null,
    };
}

/// Score the router logits and pick the top `n_used` experts, on the device.
///
/// `ids_out`/`gates_out` are filled for the caller's benefit -- correctness
/// checks and the host expert path -- but the point is that the device buffers
/// keep their own copies, so the expert kernels can read them without a
/// read-back. That read-back is the only thing preventing a layer's attention
/// and its FFN sharing a command buffer.
pub fn moeRoute(
    logits: []const f32,
    bias: ?[]const f32,
    ids_out: []u32,
    gates_out: []f32,
    gating_sigmoid: bool,
    weights_norm: bool,
    weights_scale: f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!use_gpu_ops) return false;
    const n_expert = logits.len;
    const n_used = ids_out.len;
    if (n_used != gates_out.len or n_used == 0) return false;
    if (logits.len * 4 > cx.scratch_x.len) return false;
    if (cx.route_ids == null) {
        cx.route_ids = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(u32)) catch return false;
        cx.route_gates = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(f32)) catch return false;
    }
    if (n_used > MAX_MOE_EXPERTS) return false;
    const idb = cx.route_ids.?;
    const gb = cx.route_gates.?;

    @memcpy(cx.scratch_x.slice(f32)[0..n_expert], logits);
    // The bias, when there is one, follows the logits in the same staging
    // buffer rather than needing a second.
    if (bias) |b| {
        if ((n_expert * 2) * 4 > cx.scratch_x.len) return false;
        @memcpy(cx.scratch_x.slice(f32)[n_expert..][0..n_expert], b);
    }

    const d = RouteDims{
        .n_expert = @intCast(n_expert),
        .n_used = @intCast(n_used),
        .gating = if (gating_sigmoid) 1 else 0,
        .weights_norm = if (weights_norm) 1 else 0,
        .has_bias = if (bias != null) 1 else 0,
        .weights_scale = weights_scale,
    };
    const cb = cx.dev.commandBuffer();
    cb.dispatch(
        cx.route_p,
        &.{ cx.scratch_x, cx.scratch_x, idb, gb },
        &.{ 0, n_expert * 4, 0, 0 },
        std.mem.asBytes(&d),
        32,
        32,
    );
    cb.commitAndWait();
    @memcpy(ids_out, idb.slice(u32)[0..n_used]);
    @memcpy(gates_out, gb.slice(f32)[0..n_used]);
    return true;
}

/// A whole routed MoE layer in one command buffer: route, every selected
/// expert's gate/up/SwiGLU/down, the gated reduce, and the shared expert --
/// with nothing returning to the host in between.
///
/// This is the assembly `build_moe_ffn` encodes as graph ops: `moe_route` is
/// `ggml_argsort_top_k`, the id kernels are `ggml_mul_mat_id`, and the ids and
/// gates live in device buffers throughout. The caller hands over router
/// *logits*, not routed experts -- routing on the host is exactly the
/// synchronization this exists to remove.
///
/// Local-path only: the experts must be planes of the given 3D tensors. The
/// distributed path's experts are offsets into a store and keep the
/// host-routed block.
pub const RoutedCfg = struct {
    n_expert: usize,
    n_used: usize,
    gating_sigmoid: bool,
    weights_norm: bool,
    weights_scale: f32,
};

pub fn moeFfnBlockRouted(
    normed: []const f32,
    logits: []const f32,
    bias: ?[]const f32,
    gate_w: WeightRef, // full 3D tensor, n_expert planes
    up_w: WeightRef,
    down_w: WeightRef,
    shexp: ?[3]WeightRef, // gate/up/down of the shared expert, or null
    ffn: usize,
    shexp_ffn: usize,
    cfg: RoutedCfg,
    out: []f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!use_gpu_ops) return false;
    const dim = normed.len;
    if (out.len != dim) return false;
    if (cfg.n_used == 0 or cfg.n_used > MAX_MOE_EXPERTS) return false;
    if (cfg.n_expert != logits.len) return false;

    // Staging: normed, then logits, then bias, all in scratch_x.
    const need_x = dim + cfg.n_expert * 2;
    if (need_x * 4 > cx.scratch_x.len) return false;
    // act[1]/act[2] hold every slot's gate/up at an ffn stride; act[3] every
    // slot's down output at a dim stride; act[0] the accumulator.
    const widest = @max(ffn, shexp_ffn);
    if (cfg.n_used * ffn * 4 > cx.act[1].len or widest * 4 > cx.act[1].len) return false;
    if (cfg.n_used * dim * 4 > cx.act[3].len) return false;

    const gsel = dmmvIdFor(cx, gate_w.ty) orelse return false;
    const usel = dmmvIdFor(cx, up_w.ty) orelse return false;
    const dsel = dmmvIdFor(cx, down_w.ty) orelse return false;
    if (dim % colsMultiple(gate_w.ty) != 0 or dim % colsMultiple(up_w.ty) != 0) return false;
    if (ffn % colsMultiple(down_w.ty) != 0) return false;
    const gw = wrapFor(cx, gate_w.data) orelse return false;
    const uw = wrapFor(cx, up_w.data) orelse return false;
    const dw = wrapFor(cx, down_w.data) orelse return false;

    // The shared expert resolves before recording, like everything else: a
    // decline halfway through would leave a partially built buffer.
    var sh_pipes: [3]@TypeOf(dmmvFor(undefined, .q4_k, 0).?) = undefined;
    var sh_wraps: [3]Wrapped = undefined;
    if (shexp) |sw| {
        sh_pipes[0] = dmmvFor(cx, sw[0].ty, shexp_ffn) orelse return false;
        sh_pipes[1] = dmmvFor(cx, sw[1].ty, shexp_ffn) orelse return false;
        sh_pipes[2] = dmmvFor(cx, sw[2].ty, dim) orelse return false;
        if (dim % colsMultiple(sw[0].ty) != 0 or shexp_ffn % colsMultiple(sw[2].ty) != 0) return false;
        for (sw, 0..) |w, i| sh_wraps[i] = wrapFor(cx, w.data) orelse return false;
    }

    if (cx.route_ids == null) {
        cx.route_ids = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(u32)) catch return false;
        cx.route_gates = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(f32)) catch return false;
    }

    @memcpy(cx.scratch_x.slice(f32)[0..dim], normed);
    @memcpy(cx.scratch_x.slice(f32)[dim..][0..cfg.n_expert], logits);
    if (bias) |b| @memcpy(cx.scratch_x.slice(f32)[dim + cfg.n_expert ..][0..cfg.n_expert], b);

    const rd = RouteDims{
        .n_expert = @intCast(cfg.n_expert),
        .n_used = @intCast(cfg.n_used),
        .gating = if (cfg.gating_sigmoid) 1 else 0,
        .weights_norm = if (cfg.weights_norm) 1 else 0,
        .has_bias = if (bias != null) 1 else 0,
        .weights_scale = cfg.weights_scale,
    };
    const gate_plane: u32 = @intCast(gate_w.data.len / cfg.n_expert);
    const up_plane: u32 = @intCast(up_w.data.len / cfg.n_expert);
    const down_plane: u32 = @intCast(down_w.data.len / cfg.n_expert);
    const d_gate = IdDims{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = gate_plane, .x_stride = 0 };
    const d_up = IdDims{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = up_plane, .x_stride = 0 };
    const d_down = IdDims{ .rows = @intCast(dim), .cols = @intCast(ffn), .n_used = @intCast(cfg.n_used), .plane_stride = down_plane, .x_stride = @intCast(ffn) };
    const len_ffn = extern struct { n: u32 }{ .n = @intCast(ffn) };
    const rdim = ReduceDims{ .n = @intCast(cfg.n_used), .dim = @intCast(dim), .w = @splat(0) };

    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const grid = struct {
        fn f(rows: usize, per: usize, n: usize) usize {
            const threads = ((n * rows + per - 1) / per) * SIMD_W;
            return ((threads + group - 1) / group) * group;
        }
    }.f;

    // LOOM_FUSED_DEBUG splits the one buffer into per-phase submissions and
    // prints each phase's wall time -- the only way to see inside a command
    // buffer without a Metal capture. Diagnostic only: the whole point of the
    // fused path is that these are one submission.
    const debug_split = std.c.getenv("LOOM_FUSED_DEBUG") != null;
    if (debug_split) {
        var t = [_]i128{0} ** 6;
        const now = struct {
            fn f() i128 {
                var ts: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
            }
        }.f;
        const phase = struct {
            fn go(cx2: *Ctx, comptime f: anytype, args: anytype) void {
                const cb2 = cx2.dev.commandBuffer();
                const e2 = cb2.encoder();
                @call(.auto, f, .{e2} ++ args);
                e2.end();
                cb2.commitAndWait();
            }
        };
        _ = phase;
        var t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            e2.dispatch(cx.route_p, &.{ cx.scratch_x, cx.scratch_x, cx.route_ids.?, cx.route_gates.? }, &.{ dim * 4, (dim + cfg.n_expert) * 4, 0, 0 }, std.mem.asBytes(&rd), 32, 32);
            e2.end();
            cb2.commitAndWait();
        }
        t[0] = now() - t0;
        t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            e2.dispatch(gsel.pipe, &.{ gw.buf, cx.scratch_x, cx.act[1], cx.route_ids.? }, &.{ gw.off, 0, 0, 0 }, std.mem.asBytes(&d_gate), grid(ffn, gsel.per, cfg.n_used), group);
            e2.end();
            cb2.commitAndWait();
        }
        t[1] = now() - t0;
        t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            e2.dispatch(usel.pipe, &.{ uw.buf, cx.scratch_x, cx.act[2], cx.route_ids.? }, &.{ uw.off, 0, 0, 0 }, std.mem.asBytes(&d_up), grid(ffn, usel.per, cfg.n_used), group);
            e2.end();
            cb2.commitAndWait();
        }
        t[2] = now() - t0;
        t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            for (0..cfg.n_used) |k| {
                e2.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ k * ffn * 4, k * ffn * 4, k * ffn * 4 }, std.mem.asBytes(&len_ffn), ffn, 64);
            }
            e2.end();
            cb2.commitAndWait();
        }
        t[3] = now() - t0;
        t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            e2.dispatch(dsel.pipe, &.{ dw.buf, cx.act[1], cx.act[3], cx.route_ids.? }, &.{ dw.off, 0, 0, 0 }, std.mem.asBytes(&d_down), grid(dim, dsel.per, cfg.n_used), group);
            e2.end();
            cb2.commitAndWait();
        }
        t[4] = now() - t0;
        t0 = now();
        {
            const cb2 = cx.dev.commandBuffer();
            const e2 = cb2.encoder();
            e2.dispatch(cx.reduce_dev_p, &.{ cx.act[0], cx.act[3], cx.route_gates.? }, &.{ 0, 0, 0 }, std.mem.asBytes(&rdim), dim, 64);
            e2.end();
            cb2.commitAndWait();
        }
        t[5] = now() - t0;
        std.debug.print("fused phases us: route {d} gate {d} up {d} swiglu {d} down {d} reduce {d}\n", .{
            @divTrunc(t[0], 1000), @divTrunc(t[1], 1000), @divTrunc(t[2], 1000),
            @divTrunc(t[3], 1000), @divTrunc(t[4], 1000), @divTrunc(t[5], 1000),
        });
        @memcpy(out, cx.act[0].slice(f32)[0..dim]);
        return true;
    }

    const cb = cx.dev.commandBuffer();
    const e = cb.encoder();
    e.dispatch(cx.route_p, &.{ cx.scratch_x, cx.scratch_x, cx.route_ids.?, cx.route_gates.? }, &.{ dim * 4, (dim + cfg.n_expert) * 4, 0, 0 }, std.mem.asBytes(&rd), 32, 32);
    e.barrier();
    e.dispatch(gsel.pipe, &.{ gw.buf, cx.scratch_x, cx.act[1], cx.route_ids.? }, &.{ gw.off, 0, 0, 0 }, std.mem.asBytes(&d_gate), grid(ffn, gsel.per, cfg.n_used), group);
    e.dispatch(usel.pipe, &.{ uw.buf, cx.scratch_x, cx.act[2], cx.route_ids.? }, &.{ uw.off, 0, 0, 0 }, std.mem.asBytes(&d_up), grid(ffn, usel.per, cfg.n_used), group);
    e.barrier();
    for (0..cfg.n_used) |k| {
        e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ k * ffn * 4, k * ffn * 4, k * ffn * 4 }, std.mem.asBytes(&len_ffn), ffn, 64);
    }
    e.barrier();
    e.dispatch(dsel.pipe, &.{ dw.buf, cx.act[1], cx.act[3], cx.route_ids.? }, &.{ dw.off, 0, 0, 0 }, std.mem.asBytes(&d_down), grid(dim, dsel.per, cfg.n_used), group);
    e.barrier();
    e.dispatch(cx.reduce_dev_p, &.{ cx.act[0], cx.act[3], cx.route_gates.? }, &.{ 0, 0, 0 }, std.mem.asBytes(&rdim), dim, 64);
    e.barrier();
    if (shexp != null) {
        const d_sh = Dims{ .rows = @intCast(shexp_ffn), .cols = @intCast(dim) };
        const d_shd = Dims{ .rows = @intCast(dim), .cols = @intCast(shexp_ffn) };
        const len_sh = extern struct { n: u32 }{ .n = @intCast(shexp_ffn) };
        e.dispatch(sh_pipes[0].pipe, &.{ sh_wraps[0].buf, cx.scratch_x, cx.act[1] }, &.{ sh_wraps[0].off, 0, 0 }, std.mem.asBytes(&d_sh), sh_pipes[0].groups, group);
        e.dispatch(sh_pipes[1].pipe, &.{ sh_wraps[1].buf, cx.scratch_x, cx.act[2] }, &.{ sh_wraps[1].off, 0, 0 }, std.mem.asBytes(&d_sh), sh_pipes[1].groups, group);
        e.barrier();
        e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_sh), shexp_ffn, 64);
        e.barrier();
        e.dispatch(sh_pipes[2].pipe, &.{ sh_wraps[2].buf, cx.act[1], cx.act[3] }, &.{ sh_wraps[2].off, 0, 0 }, std.mem.asBytes(&d_shd), sh_pipes[2].groups, group);
        e.barrier();
        const acc_d = AccDims{ .n = @intCast(dim), .alpha = 1.0 };
        e.dispatch(cx.sadd_p, &.{ cx.act[0], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&acc_d), dim, 64);
    }
    e.end();
    cb.commitAndWait();
    @memcpy(out, cx.act[0].slice(f32)[0..dim]);
    return true;
}

/// One MLA layer's tail as a single submission: absorb, attention, W_v, the
/// output projection, the residual add, the FFN rmsnorm, the router matvec,
/// then the whole routed MoE layer -- route, experts, reduce, shared expert.
///
/// This merges what were two command buffers per layer plus four host ops
/// between them. The host ops were cheap; the second submission was not: each
/// buffer ends in a wait, and ~27 of them per token is the difference the
/// submission counter has tracked all along. In: the residual stream and this
/// position's queries. Out: the residual stream after the layer.
///
/// The attention *head* of the layer -- attn norm, q/kv projections, kv_a
/// norm, rope, cache append -- stays on the host for now; it feeds the
/// compressed cache this reads.
pub fn mlaLayerTail(
    li: usize,
    pos: usize,
    x: []f32, // residual stream, updated in place
    q_nope: []const f32,
    q_rope: []const f32,
    kv_b: WeightRef,
    attn_out_w: WeightRef,
    ffn_norm: []const f32,
    eps: f32,
    router_w: WeightRef, // n_expert rows over dim
    router_bias: ?[]const f32,
    gate_w: WeightRef,
    up_w: WeightRef,
    down_w: WeightRef,
    shexp: ?[3]WeightRef,
    ffn: usize,
    shexp_ffn: usize,
    cfg: RoutedCfg,
    n_heads: usize,
    nope: usize,
    v_head_dim: usize,
    scale: f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!use_gpu_ops) return false;
    if (li >= cx.mla_wk.items.len) return false;
    const cbuf = cx.mla_c orelse return false;
    const rbuf = cx.mla_krope orelse return false;
    const seq = pos + 1;
    if (seq > cx.mla_ctx) return false;
    const kvr = cx.mla_kvr;
    const dim = x.len;

    // Staging: q_nope, q_rope, x, bias -- one buffer, fixed offsets.
    const off_qr = q_nope.len;
    const off_x = off_qr + q_rope.len;
    const off_bias = off_x + dim;
    if ((off_bias + cfg.n_expert) * 4 > cx.scratch_x.len) return false;

    // Slot plan. 1: q_abs / per-slot gate. 2: o_latent / per-slot up.
    // 3: head_out then proj then down outputs. 4: x then x' (residual).
    // 5: normed. 6: router logits. 0: MoE accumulator.
    const widest = @max(ffn, shexp_ffn);
    if (cfg.n_used * ffn * 4 > cx.act[1].len or widest * 4 > cx.act[1].len) return false;
    if (@max(cfg.n_used * dim, n_heads * kvr) * 4 > cx.act[3].len) return false;
    if (n_heads * kvr * 4 > cx.act[2].len) return false;

    const vsel = dmmvIdFor(cx, kv_b.ty) orelse return false;
    if (kvr % colsMultiple(kv_b.ty) != 0 or v_head_dim % vsel.per != 0) return false;
    const psel = dmmvFor(cx, attn_out_w.ty, dim) orelse return false;
    if ((n_heads * v_head_dim) % colsMultiple(attn_out_w.ty) != 0) return false;
    const rsel = dmmvFor(cx, router_w.ty, cfg.n_expert) orelse return false;
    if (dim % colsMultiple(router_w.ty) != 0) return false;
    const gsel = dmmvIdFor(cx, gate_w.ty) orelse return false;
    const usel = dmmvIdFor(cx, up_w.ty) orelse return false;
    const dsel = dmmvIdFor(cx, down_w.ty) orelse return false;
    if (dim % colsMultiple(gate_w.ty) != 0 or dim % colsMultiple(up_w.ty) != 0) return false;
    if (ffn % colsMultiple(down_w.ty) != 0) return false;

    const vw = wrapFor(cx, kv_b.data) orelse return false;
    const pw = wrapFor(cx, attn_out_w.data) orelse return false;
    const nw = wrapFor(cx, std.mem.sliceAsBytes(ffn_norm)) orelse return false;
    const rw = wrapFor(cx, router_w.data) orelse return false;
    const gw = wrapFor(cx, gate_w.data) orelse return false;
    const uw = wrapFor(cx, up_w.data) orelse return false;
    const dw = wrapFor(cx, down_w.data) orelse return false;
    var sh_pipes: [3]@TypeOf(dmmvFor(undefined, .q4_k, 0).?) = undefined;
    var sh_wraps: [3]Wrapped = undefined;
    if (shexp) |sw| {
        sh_pipes[0] = dmmvFor(cx, sw[0].ty, shexp_ffn) orelse return false;
        sh_pipes[1] = dmmvFor(cx, sw[1].ty, shexp_ffn) orelse return false;
        sh_pipes[2] = dmmvFor(cx, sw[2].ty, dim) orelse return false;
        if (dim % colsMultiple(sw[0].ty) != 0 or shexp_ffn % colsMultiple(sw[2].ty) != 0) return false;
        for (sw, 0..) |w, i| sh_wraps[i] = wrapFor(cx, w.data) orelse return false;
    }
    const row_bytes = kv_b.data.len / (n_heads * (nope + v_head_dim));
    if (cx.mla_vids == null) {
        const b = cx.dev.alloc(64 * @sizeOf(u32)) catch return false;
        for (0..64) |k| b.slice(u32)[k] = @intCast(k);
        cx.mla_vids = b;
    }
    if (cx.route_ids == null) {
        cx.route_ids = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(u32)) catch return false;
        cx.route_gates = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(f32)) catch return false;
    }

    @memcpy(cx.scratch_x.slice(f32)[0..q_nope.len], q_nope);
    @memcpy(cx.scratch_x.slice(f32)[off_qr..][0..q_rope.len], q_rope);
    @memcpy(cx.scratch_x.slice(f32)[off_x..][0..dim], x);
    if (router_bias) |b| @memcpy(cx.scratch_x.slice(f32)[off_bias..][0..cfg.n_expert], b);

    const ad = AbsorbDims{ .n_heads = @intCast(n_heads), .nope = @intCast(nope), .kvr = @intCast(kvr), .stride = @intCast(nope + v_head_dim), .q_stride = @intCast(nope), .q_off = 0 };
    const dims = MlaDims{ .n_heads = @intCast(n_heads), .kvr = @intCast(kvr), .rope = @intCast(cx.mla_rope), .seq = @intCast(seq), .qr_stride = @intCast(cx.mla_rope), .qr_off = 0, .scale = scale };
    const c_off = li * cx.mla_ctx * kvr * @sizeOf(f32);
    const r_off = li * cx.mla_ctx * cx.mla_rope * @sizeOf(f32);
    const vd_dims = IdDims{ .rows = @intCast(v_head_dim), .cols = @intCast(kvr), .n_used = @intCast(n_heads), .plane_stride = @intCast((nope + v_head_dim) * row_bytes), .x_stride = @intCast(kvr) };
    const pd = Dims{ .rows = @intCast(dim), .cols = @intCast(n_heads * v_head_dim) };
    const len_dim = extern struct { n: u32 }{ .n = @intCast(dim) };
    const nd = extern struct { n: u32, eps: f32 }{ .n = @intCast(dim), .eps = eps };
    const rdd = Dims{ .rows = @intCast(cfg.n_expert), .cols = @intCast(dim) };
    const route_d = RouteDims{ .n_expert = @intCast(cfg.n_expert), .n_used = @intCast(cfg.n_used), .gating = if (cfg.gating_sigmoid) 1 else 0, .weights_norm = if (cfg.weights_norm) 1 else 0, .has_bias = if (router_bias != null) 1 else 0, .weights_scale = cfg.weights_scale };
    const gate_plane: u32 = @intCast(gate_w.data.len / cfg.n_expert);
    const up_plane: u32 = @intCast(up_w.data.len / cfg.n_expert);
    const down_plane: u32 = @intCast(down_w.data.len / cfg.n_expert);
    const d_gate = IdDims{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = gate_plane, .x_stride = 0 };
    const d_up = IdDims{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = up_plane, .x_stride = 0 };
    const d_down = IdDims{ .rows = @intCast(dim), .cols = @intCast(ffn), .n_used = @intCast(cfg.n_used), .plane_stride = down_plane, .x_stride = @intCast(ffn) };
    const len_ffn = extern struct { n: u32 }{ .n = @intCast(ffn) };
    const rdim = ReduceDims{ .n = @intCast(cfg.n_used), .dim = @intCast(dim), .w = @splat(0) };

    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const grid = struct {
        fn f(rows: usize, per: usize, n: usize) usize {
            const threads = ((n * rows + per - 1) / per) * SIMD_W;
            return ((threads + group - 1) / group) * group;
        }
    }.f;

    const cb = cx.dev.commandBuffer();
    const e = cb.encoder();
    // ---- attention ----
    e.dispatch(cx.mla_absorb_p, &.{ cx.mla_wk.items[li], cx.scratch_x, cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&ad), absorbGrid(n_heads, kvr), group);
    e.barrier();
    e.dispatch(cx.mla_attn_p, &.{ cx.act[1], cx.scratch_x, cbuf, rbuf, cx.act[3] }, &.{ 0, off_qr * 4, c_off, r_off, 0 }, std.mem.asBytes(&dims), n_heads * group, group);
    e.barrier();
    e.dispatch(cx.mla_wsum_p, &.{ cx.act[3], cbuf, cx.act[2] }, &.{ 0, c_off, 0 }, std.mem.asBytes(&dims), absorbGrid(n_heads, kvr), group);
    e.barrier();
    e.dispatch(vsel.pipe, &.{ vw.buf, cx.act[2], cx.act[3], cx.mla_vids.? }, &.{ vw.off + nope * row_bytes, 0, 0, 0 }, std.mem.asBytes(&vd_dims), grid(v_head_dim, vsel.per, n_heads), group);
    e.barrier();
    // ---- projection, residual, norm, router ----
    e.dispatch(psel.pipe, &.{ pw.buf, cx.act[3], cx.act[4] }, &.{ pw.off, 0, 0 }, std.mem.asBytes(&pd), psel.groups, group);
    e.barrier();
    // x' = x + proj: add is a += b, so stage x into act[4]? act[4] holds proj;
    // add x from scratch: dispatch add(proj += x_staged) -- order is symmetric.
    e.dispatch(cx.add_p, &.{ cx.act[4], cx.scratch_x }, &.{ 0, off_x * 4 }, std.mem.asBytes(&len_dim), dim, 64);
    e.barrier();
    e.dispatch(cx.rmsnorm_p, &.{ cx.act[4], nw.buf, cx.act[5] }, &.{ 0, nw.off, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
    e.barrier();
    e.dispatch(rsel.pipe, &.{ rw.buf, cx.act[5], cx.act[6] }, &.{ rw.off, 0, 0 }, std.mem.asBytes(&rdd), rsel.groups, group);
    e.barrier();
    // ---- routed MoE, reading normed from act[5] and logits from act[6] ----
    e.dispatch(cx.route_p, &.{ cx.act[6], cx.scratch_x, cx.route_ids.?, cx.route_gates.? }, &.{ 0, off_bias * 4, 0, 0 }, std.mem.asBytes(&route_d), 32, 32);
    e.barrier();
    e.dispatch(gsel.pipe, &.{ gw.buf, cx.act[5], cx.act[1], cx.route_ids.? }, &.{ gw.off, 0, 0, 0 }, std.mem.asBytes(&d_gate), grid(ffn, gsel.per, cfg.n_used), group);
    e.dispatch(usel.pipe, &.{ uw.buf, cx.act[5], cx.act[2], cx.route_ids.? }, &.{ uw.off, 0, 0, 0 }, std.mem.asBytes(&d_up), grid(ffn, usel.per, cfg.n_used), group);
    e.barrier();
    for (0..cfg.n_used) |k| {
        e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ k * ffn * 4, k * ffn * 4, k * ffn * 4 }, std.mem.asBytes(&len_ffn), ffn, 64);
    }
    e.barrier();
    e.dispatch(dsel.pipe, &.{ dw.buf, cx.act[1], cx.act[3], cx.route_ids.? }, &.{ dw.off, 0, 0, 0 }, std.mem.asBytes(&d_down), grid(dim, dsel.per, cfg.n_used), group);
    e.barrier();
    e.dispatch(cx.reduce_dev_p, &.{ cx.act[0], cx.act[3], cx.route_gates.? }, &.{ 0, 0, 0 }, std.mem.asBytes(&rdim), dim, 64);
    e.barrier();
    if (shexp != null) {
        const d_sh = Dims{ .rows = @intCast(shexp_ffn), .cols = @intCast(dim) };
        const d_shd = Dims{ .rows = @intCast(dim), .cols = @intCast(shexp_ffn) };
        const len_sh = extern struct { n: u32 }{ .n = @intCast(shexp_ffn) };
        e.dispatch(sh_pipes[0].pipe, &.{ sh_wraps[0].buf, cx.act[5], cx.act[1] }, &.{ sh_wraps[0].off, 0, 0 }, std.mem.asBytes(&d_sh), sh_pipes[0].groups, group);
        e.dispatch(sh_pipes[1].pipe, &.{ sh_wraps[1].buf, cx.act[5], cx.act[2] }, &.{ sh_wraps[1].off, 0, 0 }, std.mem.asBytes(&d_sh), sh_pipes[1].groups, group);
        e.barrier();
        e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_sh), shexp_ffn, 64);
        e.barrier();
        e.dispatch(sh_pipes[2].pipe, &.{ sh_wraps[2].buf, cx.act[1], cx.act[3] }, &.{ sh_wraps[2].off, 0, 0 }, std.mem.asBytes(&d_shd), sh_pipes[2].groups, group);
        e.barrier();
        const acc_d = AccDims{ .n = @intCast(dim), .alpha = 1.0 };
        e.dispatch(cx.sadd_p, &.{ cx.act[0], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&acc_d), dim, 64);
        e.barrier();
    }
    // x'' = x' + ffn: both on device.
    e.dispatch(cx.add_p, &.{ cx.act[4], cx.act[0] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), dim, 64);
    e.end();
    cb.commitAndWait();
    @memcpy(x, cx.act[4].slice(f32)[0..dim]);
    return true;
}

/// Everything one MLA layer needs, resolved once at load so the per-token
/// frame is emission only.
pub const MlaLayerDesc = struct {
    attn_norm: []const f32,
    wq: WeightRef, // n_heads*kd rows over dim
    kv_a: WeightRef, // kvr+rope rows over dim
    kv_a_norm: []const f32,
    kv_b: WeightRef,
    attn_out: WeightRef,
    ffn_norm: []const f32,
    is_moe: bool,
    // MoE layers:
    router: WeightRef = .{ .ty = .f32, .data = &.{} },
    gate: WeightRef = .{ .ty = .f32, .data = &.{} },
    up: WeightRef = .{ .ty = .f32, .data = &.{} },
    down: WeightRef = .{ .ty = .f32, .data = &.{} },
    shexp: ?[3]WeightRef = null,
    ffn: usize = 0,
    shexp_ffn: usize = 0,
    // Dense layers:
    dgate: WeightRef = .{ .ty = .f32, .data = &.{} },
    dup: WeightRef = .{ .ty = .f32, .data = &.{} },
    ddown: WeightRef = .{ .ty = .f32, .data = &.{} },
    dffn: usize = 0,
};

pub const MlaFrameCfg = struct {
    dim: usize,
    n_heads: usize,
    nope: usize,
    rope: usize,
    kvr: usize,
    v_head_dim: usize,
    eps: f32,
    scale: f32,
    rope_base: f32,
    yarn_factor: f32,
    yarn_orig_ctx: f32,
    routed: RoutedCfg,
};

/// A whole token -- every layer's attention head, attention, and FFN, then the
/// final norm and the lm_head -- as **one** command buffer.
///
/// This is `ggml_metal_graph_compute`'s shape: the residual stream lives in a
/// device slot for the entire token and only the logits come back. Slot plan
/// per layer: 4 = x (persistent), 5 = normed, 6 = q then router logits (q is
/// consumed by attention before the router writes), 7 = kv_a then the
/// projected attention output, 1/2/3/0 = the FFN's working set, exactly as the
/// layer tail uses them.
///
/// Declines -- engine falls back to the per-layer path -- on anything it
/// cannot express: a router bias (not yet staged per layer), a type without a
/// kernel, widths off the kernels' grids.
pub fn mlaTokenFrame(
    descs: []const MlaLayerDesc,
    fc: MlaFrameCfg,
    x: []f32,
    pos: usize,
    out_norm: []const f32,
    lm_head: WeightRef,
    logits: []f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!use_gpu_ops) return false;
    // LOOM_NO_FRAME bisects against the per-layer tail. The first frame
    // measurement said slower and was wrong -- it was the cold run, paying
    // pipeline compiles and the wrap of every tensor. Warm and alternating:
    // 27.9-30.4 ms/token against the tail's 38.7-47.6, at 1.8 command buffers
    // per token against 28.
    if (std.c.getenv("LOOM_NO_FRAME") != null) return false;
    const cbuf = cx.mla_c orelse return false;
    const rbuf = cx.mla_krope orelse return false;
    if (pos + 1 > cx.mla_ctx or descs.len > cx.mla_wk.items.len) return false;
    const dim = fc.dim;
    const kd = fc.nope + fc.rope;
    const seq = pos + 1;
    if (dim * 4 > cx.scratch_x.len or logits.len * 4 > cx.scratch_out.len) return false;
    const vocab = logits.len;

    // Resolve everything for every layer before a single dispatch is encoded.
    const P = struct {
        anw: Wrapped,
        qsel: @TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        qw: Wrapped,
        kasel: @TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        kaw: Wrapped,
        kanw: Wrapped,
        vsel: @TypeOf(dmmvIdFor(undefined, .q4_k).?),
        vw: Wrapped,
        row_bytes: usize,
        psel: @TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        pw: Wrapped,
        fnw: Wrapped,
        rsel: @TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        rw: Wrapped,
        gsel: @TypeOf(dmmvIdFor(undefined, .q4_k).?),
        gw: Wrapped,
        usel: @TypeOf(dmmvIdFor(undefined, .q4_k).?),
        uw: Wrapped,
        dsel: @TypeOf(dmmvIdFor(undefined, .q4_k).?),
        dw: Wrapped,
        sh_pipes: [3]@TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        sh_wraps: [3]Wrapped,
        dq: [3]@TypeOf(dmmvFor(undefined, .q4_k, 0).?),
        dqw: [3]Wrapped,
    };
    var plans_buf: [64]P = undefined;
    if (descs.len > plans_buf.len) return false;
    for (descs, 0..) |d, i| {
        var p: P = undefined;
        p.anw = wrapFor(cx, std.mem.sliceAsBytes(d.attn_norm)) orelse return false;
        p.qsel = dmmvFor(cx, d.wq.ty, fc.n_heads * kd) orelse return false;
        if (dim % colsMultiple(d.wq.ty) != 0) return false;
        p.qw = wrapFor(cx, d.wq.data) orelse return false;
        p.kasel = dmmvFor(cx, d.kv_a.ty, fc.kvr + fc.rope) orelse return false;
        if (dim % colsMultiple(d.kv_a.ty) != 0) return false;
        p.kaw = wrapFor(cx, d.kv_a.data) orelse return false;
        p.kanw = wrapFor(cx, std.mem.sliceAsBytes(d.kv_a_norm)) orelse return false;
        p.vsel = dmmvIdFor(cx, d.kv_b.ty) orelse return false;
        if (fc.kvr % colsMultiple(d.kv_b.ty) != 0 or fc.v_head_dim % p.vsel.per != 0) return false;
        p.vw = wrapFor(cx, d.kv_b.data) orelse return false;
        p.row_bytes = d.kv_b.data.len / (fc.n_heads * (fc.nope + fc.v_head_dim));
        p.psel = dmmvFor(cx, d.attn_out.ty, dim) orelse return false;
        if ((fc.n_heads * fc.v_head_dim) % colsMultiple(d.attn_out.ty) != 0) return false;
        p.pw = wrapFor(cx, d.attn_out.data) orelse return false;
        p.fnw = wrapFor(cx, std.mem.sliceAsBytes(d.ffn_norm)) orelse return false;
        if (d.is_moe) {
            p.rsel = dmmvFor(cx, d.router.ty, fc.routed.n_expert) orelse return false;
            if (dim % colsMultiple(d.router.ty) != 0) return false;
            p.rw = wrapFor(cx, d.router.data) orelse return false;
            p.gsel = dmmvIdFor(cx, d.gate.ty) orelse return false;
            p.usel = dmmvIdFor(cx, d.up.ty) orelse return false;
            p.dsel = dmmvIdFor(cx, d.down.ty) orelse return false;
            if (dim % colsMultiple(d.gate.ty) != 0 or dim % colsMultiple(d.up.ty) != 0) return false;
            if (d.ffn % colsMultiple(d.down.ty) != 0) return false;
            p.gw = wrapFor(cx, d.gate.data) orelse return false;
            p.uw = wrapFor(cx, d.up.data) orelse return false;
            p.dw = wrapFor(cx, d.down.data) orelse return false;
            if (d.shexp) |sw| {
                p.sh_pipes[0] = dmmvFor(cx, sw[0].ty, d.shexp_ffn) orelse return false;
                p.sh_pipes[1] = dmmvFor(cx, sw[1].ty, d.shexp_ffn) orelse return false;
                p.sh_pipes[2] = dmmvFor(cx, sw[2].ty, dim) orelse return false;
                if (d.shexp_ffn % colsMultiple(sw[2].ty) != 0) return false;
                for (sw, 0..) |w, k| p.sh_wraps[k] = wrapFor(cx, w.data) orelse return false;
            }
            if (cfgSlotsShort(cx, d.ffn, d.shexp_ffn, dim, fc.routed.n_used)) return false;
        } else {
            p.dq[0] = dmmvFor(cx, d.dgate.ty, d.dffn) orelse return false;
            p.dq[1] = dmmvFor(cx, d.dup.ty, d.dffn) orelse return false;
            p.dq[2] = dmmvFor(cx, d.ddown.ty, dim) orelse return false;
            if (dim % colsMultiple(d.dgate.ty) != 0 or d.dffn % colsMultiple(d.ddown.ty) != 0) return false;
            p.dqw[0] = wrapFor(cx, d.dgate.data) orelse return false;
            p.dqw[1] = wrapFor(cx, d.dup.data) orelse return false;
            p.dqw[2] = wrapFor(cx, d.ddown.data) orelse return false;
            if (d.dffn * 4 > cx.act[1].len) return false;
        }
        plans_buf[i] = p;
    }
    const onw = wrapFor(cx, std.mem.sliceAsBytes(out_norm)) orelse return false;
    const lmsel = dmmvFor(cx, lm_head.ty, vocab) orelse return false;
    if (dim % colsMultiple(lm_head.ty) != 0) return false;
    const lmw = wrapFor(cx, lm_head.data) orelse return false;
    if (cx.mla_vids == null) {
        const b = cx.dev.alloc(64 * @sizeOf(u32)) catch return false;
        for (0..64) |k| b.slice(u32)[k] = @intCast(k);
        cx.mla_vids = b;
    }
    if (cx.route_ids == null) {
        cx.route_ids = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(u32)) catch return false;
        cx.route_gates = cx.dev.alloc(MAX_MOE_EXPERTS * @sizeOf(f32)) catch return false;
    }

    @memcpy(cx.act[4].slice(f32)[0..dim], x);

    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const grid = struct {
        fn f(rows: usize, per: usize, n: usize) usize {
            const threads = ((n * rows + per - 1) / per) * SIMD_W;
            return ((threads + group - 1) / group) * group;
        }
    }.f;
    const nd = extern struct { n: u32, eps: f32 }{ .n = @intCast(dim), .eps = fc.eps };
    const nd_kv = extern struct { n: u32, eps: f32 }{ .n = @intCast(fc.kvr), .eps = fc.eps };
    const len_dim = extern struct { n: u32 }{ .n = @intCast(dim) };
    const rope_n: u32 = @intCast(fc.rope);
    const ad = AbsorbDims{ .n_heads = @intCast(fc.n_heads), .nope = @intCast(fc.nope), .kvr = @intCast(fc.kvr), .stride = @intCast(fc.nope + fc.v_head_dim), .q_stride = @intCast(kd), .q_off = 0 };
    const at = MlaDims{ .n_heads = @intCast(fc.n_heads), .kvr = @intCast(fc.kvr), .rope = @intCast(fc.rope), .seq = @intCast(seq), .qr_stride = @intCast(kd), .qr_off = @intCast(fc.nope), .scale = fc.scale };
    const qd = Dims{ .rows = @intCast(fc.n_heads * kd), .cols = @intCast(dim) };
    const kad = Dims{ .rows = @intCast(fc.kvr + fc.rope), .cols = @intCast(dim) };
    const pd = Dims{ .rows = @intCast(dim), .cols = @intCast(fc.n_heads * fc.v_head_dim) };
    const q_rope_d = MlaRopeDims{ .n_vec = @intCast(fc.n_heads), .rope = rope_n, .stride = @intCast(kd), .offset = @intCast(fc.nope), .pos = @intCast(pos), .base = fc.rope_base, .yarn_factor = fc.yarn_factor, .yarn_orig_ctx = fc.yarn_orig_ctx };
    const k_rope_d = MlaRopeDims{ .n_vec = 1, .rope = rope_n, .stride = 0, .offset = 0, .pos = @intCast(pos), .base = fc.rope_base, .yarn_factor = fc.yarn_factor, .yarn_orig_ctx = fc.yarn_orig_ctx };
    const rdim = ReduceDims{ .n = @intCast(fc.routed.n_used), .dim = @intCast(dim), .w = @splat(0) };
    const route_d = RouteDims{ .n_expert = @intCast(fc.routed.n_expert), .n_used = @intCast(fc.routed.n_used), .gating = if (fc.routed.gating_sigmoid) 1 else 0, .weights_norm = if (fc.routed.weights_norm) 1 else 0, .has_bias = 0, .weights_scale = fc.routed.weights_scale };
    const rdd = Dims{ .rows = @intCast(fc.routed.n_expert), .cols = @intCast(dim) };

    // LOOM_FRAME_DEBUG: emit each layer as four waited sub-buffers -- head,
    // attention, projection block, FFN -- and print the per-category sums.
    // The only way to see where the fused token's time goes without a Metal
    // capture; the split's own submission overhead (~108 waits) makes the
    // absolute total meaningless, so only the ratios and the comparison with
    // the fused total carry information.
    const debug_split = std.c.getenv("LOOM_FRAME_DEBUG") != null;
    var ph = [_]i128{0} ** 5;
    const nowf = struct {
        fn f() i128 {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
        }
    }.f;

    var cb = cx.dev.commandBuffer();
    var e = cb.encoder();
    for (descs, 0..) |d, li| {
        const p = &plans_buf[li];
        const c_pos = (li * cx.mla_ctx + pos) * fc.kvr * 4;
        const r_pos = (li * cx.mla_ctx + pos) * fc.rope * 4;
        // ---- head ----
        e.dispatch(cx.rmsnorm_p, &.{ cx.act[4], p.anw.buf, cx.act[5] }, &.{ 0, p.anw.off, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
        e.barrier();
        e.dispatch(p.qsel.pipe, &.{ p.qw.buf, cx.act[5], cx.act[6] }, &.{ p.qw.off, 0, 0 }, std.mem.asBytes(&qd), p.qsel.groups, group);
        e.dispatch(p.kasel.pipe, &.{ p.kaw.buf, cx.act[5], cx.act[7] }, &.{ p.kaw.off, 0, 0 }, std.mem.asBytes(&kad), p.kasel.groups, group);
        e.barrier();
        // c_kv: norm straight into the cache row; k_rope: copy then rotate there.
        e.dispatch(cx.rmsnorm_p, &.{ cx.act[7], p.kanw.buf, cbuf }, &.{ 0, p.kanw.off, c_pos }, std.mem.asBytes(&nd_kv), SIMD_W, SIMD_W);
        e.dispatch(cx.copy_p, &.{ cx.act[7], rbuf }, &.{ fc.kvr * 4, r_pos }, std.mem.asBytes(&rope_n), fc.rope, 32);
        e.dispatch(cx.mla_rope_p, &.{cx.act[6]}, &.{0}, std.mem.asBytes(&q_rope_d), fc.n_heads * fc.rope / 2, 32);
        e.barrier();
        e.dispatch(cx.mla_rope_p, &.{rbuf}, &.{r_pos}, std.mem.asBytes(&k_rope_d), fc.rope / 2, 32);
        e.barrier();
        if (debug_split) {
            const t0 = nowf();
            e.end();
            cb.commitAndWait();
            ph[0] += nowf() - t0;
            cb = cx.dev.commandBuffer();
            e = cb.encoder();
        }
        // ---- attention ----
        e.dispatch(cx.mla_absorb_p, &.{ cx.mla_wk.items[li], cx.act[6], cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&ad), absorbGrid(fc.n_heads, fc.kvr), group);
        e.barrier();
        e.dispatch(cx.mla_attn_p, &.{ cx.act[1], cx.act[6], cbuf, rbuf, cx.act[3] }, &.{ 0, 0, li * cx.mla_ctx * fc.kvr * 4, li * cx.mla_ctx * fc.rope * 4, 0 }, std.mem.asBytes(&at), fc.n_heads * group, group);
        e.barrier();
        e.dispatch(cx.mla_wsum_p, &.{ cx.act[3], cbuf, cx.act[2] }, &.{ 0, li * cx.mla_ctx * fc.kvr * 4, 0 }, std.mem.asBytes(&at), absorbGrid(fc.n_heads, fc.kvr), group);
        e.barrier();
        const vd_dims = IdDims{ .rows = @intCast(fc.v_head_dim), .cols = @intCast(fc.kvr), .n_used = @intCast(fc.n_heads), .plane_stride = @intCast((fc.nope + fc.v_head_dim) * p.row_bytes), .x_stride = @intCast(fc.kvr) };
        e.dispatch(p.vsel.pipe, &.{ p.vw.buf, cx.act[2], cx.act[3], cx.mla_vids.? }, &.{ p.vw.off + fc.nope * p.row_bytes, 0, 0, 0 }, std.mem.asBytes(&vd_dims), grid(fc.v_head_dim, p.vsel.per, fc.n_heads), group);
        e.barrier();
        if (debug_split) {
            const t0 = nowf();
            e.end();
            cb.commitAndWait();
            ph[1] += nowf() - t0;
            cb = cx.dev.commandBuffer();
            e = cb.encoder();
        }
        e.dispatch(p.psel.pipe, &.{ p.pw.buf, cx.act[3], cx.act[7] }, &.{ p.pw.off, 0, 0 }, std.mem.asBytes(&pd), p.psel.groups, group);
        e.barrier();
        e.dispatch(cx.add_p, &.{ cx.act[4], cx.act[7] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), dim, 64);
        e.barrier();
        e.dispatch(cx.rmsnorm_p, &.{ cx.act[4], p.fnw.buf, cx.act[5] }, &.{ 0, p.fnw.off, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
        e.barrier();
        if (debug_split) {
            const t0 = nowf();
            e.end();
            cb.commitAndWait();
            ph[2] += nowf() - t0;
            cb = cx.dev.commandBuffer();
            e = cb.encoder();
        }
        if (d.is_moe) {
            e.dispatch(p.rsel.pipe, &.{ p.rw.buf, cx.act[5], cx.act[6] }, &.{ p.rw.off, 0, 0 }, std.mem.asBytes(&rdd), p.rsel.groups, group);
            e.barrier();
            e.dispatch(cx.route_p, &.{ cx.act[6], cx.act[6], cx.route_ids.?, cx.route_gates.? }, &.{ 0, 0, 0, 0 }, std.mem.asBytes(&route_d), 32, 32);
            e.barrier();
            const gate_plane: u32 = @intCast(d.gate.data.len / fc.routed.n_expert);
            const up_plane: u32 = @intCast(d.up.data.len / fc.routed.n_expert);
            const down_plane: u32 = @intCast(d.down.data.len / fc.routed.n_expert);
            const d_gate = IdDims{ .rows = @intCast(d.ffn), .cols = @intCast(dim), .n_used = @intCast(fc.routed.n_used), .plane_stride = gate_plane, .x_stride = 0 };
            const d_up = IdDims{ .rows = @intCast(d.ffn), .cols = @intCast(dim), .n_used = @intCast(fc.routed.n_used), .plane_stride = up_plane, .x_stride = 0 };
            const d_down = IdDims{ .rows = @intCast(dim), .cols = @intCast(d.ffn), .n_used = @intCast(fc.routed.n_used), .plane_stride = down_plane, .x_stride = @intCast(d.ffn) };
            e.dispatch(p.gsel.pipe, &.{ p.gw.buf, cx.act[5], cx.act[1], cx.route_ids.? }, &.{ p.gw.off, 0, 0, 0 }, std.mem.asBytes(&d_gate), grid(d.ffn, p.gsel.per, fc.routed.n_used), group);
            e.dispatch(p.usel.pipe, &.{ p.uw.buf, cx.act[5], cx.act[2], cx.route_ids.? }, &.{ p.uw.off, 0, 0, 0 }, std.mem.asBytes(&d_up), grid(d.ffn, p.usel.per, fc.routed.n_used), group);
            e.barrier();
            const sw2 = extern struct { n: u32, slots: u32 }{ .n = @intCast(d.ffn), .slots = @intCast(fc.routed.n_used) };
            e.dispatch(cx.swiglu_slots_p, &.{ cx.act[1], cx.act[2] }, &.{ 0, 0 }, std.mem.asBytes(&sw2), d.ffn * fc.routed.n_used, 64);
            e.barrier();
            e.dispatch(p.dsel.pipe, &.{ p.dw.buf, cx.act[1], cx.act[3], cx.route_ids.? }, &.{ p.dw.off, 0, 0, 0 }, std.mem.asBytes(&d_down), grid(dim, p.dsel.per, fc.routed.n_used), group);
            e.barrier();
            e.dispatch(cx.reduce_dev_p, &.{ cx.act[0], cx.act[3], cx.route_gates.? }, &.{ 0, 0, 0 }, std.mem.asBytes(&rdim), dim, 64);
            // No barrier: the shared expert's gate/up read act[5] and write
            // act[1]/act[2], all disjoint from the reduce's act[3] -> act[0].
            // The barrier after them orders everything before the shared
            // expert's own swiglu.
            if (d.shexp != null) {
                const d_sh = Dims{ .rows = @intCast(d.shexp_ffn), .cols = @intCast(dim) };
                const d_shd = Dims{ .rows = @intCast(dim), .cols = @intCast(d.shexp_ffn) };
                const len_sh = extern struct { n: u32 }{ .n = @intCast(d.shexp_ffn) };
                e.dispatch(p.sh_pipes[0].pipe, &.{ p.sh_wraps[0].buf, cx.act[5], cx.act[1] }, &.{ p.sh_wraps[0].off, 0, 0 }, std.mem.asBytes(&d_sh), p.sh_pipes[0].groups, group);
                e.dispatch(p.sh_pipes[1].pipe, &.{ p.sh_wraps[1].buf, cx.act[5], cx.act[2] }, &.{ p.sh_wraps[1].off, 0, 0 }, std.mem.asBytes(&d_sh), p.sh_pipes[1].groups, group);
                e.barrier();
                e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_sh), d.shexp_ffn, 64);
                e.barrier();
                e.dispatch(p.sh_pipes[2].pipe, &.{ p.sh_wraps[2].buf, cx.act[1], cx.act[3] }, &.{ p.sh_wraps[2].off, 0, 0 }, std.mem.asBytes(&d_shd), p.sh_pipes[2].groups, group);
                e.barrier();
                const acc_d = AccDims{ .n = @intCast(dim), .alpha = 1.0 };
                e.dispatch(cx.sadd_p, &.{ cx.act[0], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&acc_d), dim, 64);
                e.barrier();
            }
            e.dispatch(cx.add_p, &.{ cx.act[4], cx.act[0] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), dim, 64);
        } else {
            const d_f = Dims{ .rows = @intCast(d.dffn), .cols = @intCast(dim) };
            const d_fd = Dims{ .rows = @intCast(dim), .cols = @intCast(d.dffn) };
            const len_f = extern struct { n: u32 }{ .n = @intCast(d.dffn) };
            e.dispatch(p.dq[0].pipe, &.{ p.dqw[0].buf, cx.act[5], cx.act[1] }, &.{ p.dqw[0].off, 0, 0 }, std.mem.asBytes(&d_f), p.dq[0].groups, group);
            e.dispatch(p.dq[1].pipe, &.{ p.dqw[1].buf, cx.act[5], cx.act[2] }, &.{ p.dqw[1].off, 0, 0 }, std.mem.asBytes(&d_f), p.dq[1].groups, group);
            e.barrier();
            e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_f), d.dffn, 64);
            e.barrier();
            e.dispatch(p.dq[2].pipe, &.{ p.dqw[2].buf, cx.act[1], cx.act[3] }, &.{ p.dqw[2].off, 0, 0 }, std.mem.asBytes(&d_fd), p.dq[2].groups, group);
            e.barrier();
            e.dispatch(cx.add_p, &.{ cx.act[4], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), dim, 64);
        }
        e.barrier();
        if (debug_split) {
            const t0 = nowf();
            e.end();
            cb.commitAndWait();
            ph[3] += nowf() - t0;
            cb = cx.dev.commandBuffer();
            e = cb.encoder();
        }
    }
    // ---- final norm and lm_head ----
    e.dispatch(cx.rmsnorm_p, &.{ cx.act[4], onw.buf, cx.act[5] }, &.{ 0, onw.off, 0 }, std.mem.asBytes(&nd), SIMD_W, SIMD_W);
    e.barrier();
    const lmd = Dims{ .rows = @intCast(vocab), .cols = @intCast(dim) };
    e.dispatch(lmsel.pipe, &.{ lmw.buf, cx.act[5], cx.scratch_out }, &.{ lmw.off, 0, 0 }, std.mem.asBytes(&lmd), lmsel.groups, group);
    {
        const t0 = nowf();
        e.end();
        cb.commitAndWait();
        ph[4] += nowf() - t0;
    }
    if (debug_split) {
        std.debug.print("frame phases us: head {d} attn {d} proj {d} ffn {d} lmhead {d}\n", .{
            @divTrunc(ph[0], 1000), @divTrunc(ph[1], 1000), @divTrunc(ph[2], 1000),
            @divTrunc(ph[3], 1000), @divTrunc(ph[4], 1000),
        });
    }
    @memcpy(x, cx.act[4].slice(f32)[0..dim]);
    @memcpy(logits, cx.scratch_out.slice(f32)[0..vocab]);
    return true;
}

/// Read one position's cache rows back to the host. The frame writes the
/// compressed cache on the device only; the host copy stays authoritative for
/// any later fallback, so the engine mirrors these rows after each frame
/// token -- ~62 KB, against the alternative of a fallback attending over
/// stale rows and producing plausible text.
pub fn mlaReadCache(li: usize, pos: usize, c_kv: []f32, k_rope: []f32) bool {
    const cx = &(ctx orelse return false);
    const cbuf = cx.mla_c orelse return false;
    const rbuf = cx.mla_krope orelse return false;
    if (pos >= cx.mla_ctx or c_kv.len != cx.mla_kvr or k_rope.len != cx.mla_rope) return false;
    @memcpy(c_kv, cbuf.slice(f32)[(li * cx.mla_ctx + pos) * cx.mla_kvr ..][0..c_kv.len]);
    @memcpy(k_rope, rbuf.slice(f32)[(li * cx.mla_ctx + pos) * cx.mla_rope ..][0..k_rope.len]);
    return true;
}

/// One SIMD group per absorbed output element, rounded to whole threadgroups.
fn absorbGrid(n_heads: usize, kvr: usize) usize {
    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const threads = n_heads * kvr * SIMD_W;
    return ((threads + group - 1) / group) * group;
}

fn cfgSlotsShort(cx: *Ctx, ffn: usize, shexp_ffn: usize, dim: usize, n_used: usize) bool {
    const widest = @max(ffn, shexp_ffn);
    if (n_used * ffn * 4 > cx.act[1].len or widest * 4 > cx.act[1].len) return true;
    if (n_used * dim * 4 > cx.act[3].len) return true;
    return false;
}

/// One selected expert: its three weight tensors and its routing gate.
pub const ExpertRef = struct {
    gate: WeightRef,
    up: WeightRef,
    down: WeightRef,
    /// Router weight this expert's output is scaled by before accumulation.
    weight: f32,
    /// This expert's hidden width, per expert because the shared expert is
    /// wider than a routed one.
    ffn: usize,
};

/// A whole MoE layer's experts in one command buffer.
///
/// This is the piece that matters for a MoE model: `expert ffn` is ~40% of a
/// DeepSeek token, and dispatching each expert's four kernels on its own
/// submission costs seven command buffers per layer where one will do. The
/// weighted accumulation runs on device between experts for the same reason —
/// a host-side sum would end the buffer.
///
/// `normed` is the already-normalised input (attention still runs on the host
/// for MLA models, so this is a host array); `out` receives the weighted sum.
/// Returns false if any expert's types or shapes are unsupported, in which case
/// nothing has been recorded and the caller runs its own path.
pub fn moeFfnBlock(
    normed: []const f32,
    experts: []const ExpertRef,
    out: []f32,
) bool {
    const cx = &(ctx orelse return false);
    if (!use_gpu_ops and calibrated and !calibrating) return false;
    const dim = normed.len;
    if (experts.len == 0) return false;
    if (out.len != dim) return false;
    if (dim * 4 > cx.scratch_x.len) return false;
    var widest: usize = dim;
    for (experts) |e| widest = @max(widest, e.ffn);
    if (widest * 4 > cx.act[0].len) return false;

    // Resolve everything before recording: a decline halfway through would
    // leave a partially-built command buffer.
    var pipes: [MAX_MOE_EXPERTS][3]@TypeOf(dmmvFor(undefined, .q4_k, 0).?) = undefined;
    var wraps: [MAX_MOE_EXPERTS][3]Wrapped = undefined;
    if (experts.len > MAX_MOE_EXPERTS) return false;
    for (experts, 0..) |e, i| {
        pipes[i][0] = dmmvFor(cx, e.gate.ty, e.ffn) orelse return false;
        pipes[i][1] = dmmvFor(cx, e.up.ty, e.ffn) orelse return false;
        pipes[i][2] = dmmvFor(cx, e.down.ty, dim) orelse return false;
        // Per tensor, not one rule for the layer: gate and up read `dim`
        // columns, down reads `ffn`, and each type has its own block width.
        if (dim % colsMultiple(e.gate.ty) != 0 or dim % colsMultiple(e.up.ty) != 0) return false;
        if (e.ffn % colsMultiple(e.down.ty) != 0) return false;
        wraps[i][0] = wrapFor(cx, e.gate.data) orelse return false;
        wraps[i][1] = wrapFor(cx, e.up.data) orelse return false;
        wraps[i][2] = wrapFor(cx, e.down.data) orelse return false;
    }

    @memcpy(cx.scratch_x.slice(f32)[0..dim], normed);

    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    var dims_ffn: [MAX_MOE_EXPERTS]Dims = undefined;
    var dims_down: [MAX_MOE_EXPERTS]Dims = undefined;
    var len_ffn: [MAX_MOE_EXPERTS]extern struct { n: u32 } = undefined;
    for (experts, 0..) |ex, k| {
        dims_ffn[k] = .{ .rows = @intCast(ex.ffn), .cols = @intCast(dim) };
        dims_down[k] = .{ .rows = @intCast(dim), .cols = @intCast(ex.ffn) };
        len_ffn[k] = .{ .n = @intCast(ex.ffn) };
    }

    // Phase-parallel across experts, not expert-by-expert.
    //
    // The per-expert form ran gate, up, swiglu, down and accumulate as a chain
    // with a barrier at every step -- about thirty barriers for a layer, each a
    // full pipeline drain, and never more than two dispatches able to run at
    // once. Experts are independent until the final sum, so the whole layer is
    // four phases instead: every gate and up together (twelve-way parallel for
    // six experts), every swiglu, every down, then one fused reduce.
    //
    // Slot 0 is the accumulator; 1 and 2 hold every expert's gate and up,
    // 3 every expert's output, each strided by expert. `ffn * 4` and `dim * 4`
    // are multiples of 256 for every real width, so the offsets are aligned.
    const gate_stride = widest * 4;
    const out_stride = dim * 4;
    if (experts.len * gate_stride > cx.act[1].len) return false;
    if (experts.len * gate_stride > cx.act[2].len) return false;
    if (experts.len * out_stride > cx.act[3].len) return false;

    const cb = cx.dev.commandBuffer();
    const e = cb.encoder();
    for (experts, 0..) |_, i| {
        e.dispatch(pipes[i][0].pipe, &.{ wraps[i][0].buf, cx.scratch_x, cx.act[1] }, &.{ wraps[i][0].off, 0, i * gate_stride }, std.mem.asBytes(&dims_ffn[i]), pipes[i][0].groups, group);
        e.dispatch(pipes[i][1].pipe, &.{ wraps[i][1].buf, cx.scratch_x, cx.act[2] }, &.{ wraps[i][1].off, 0, i * gate_stride }, std.mem.asBytes(&dims_ffn[i]), pipes[i][1].groups, group);
    }
    e.barrier();
    for (experts, 0..) |_, i| {
        e.dispatch(cx.swiglu_p, &.{ cx.act[1], cx.act[2], cx.act[1] }, &.{ i * gate_stride, i * gate_stride, i * gate_stride }, std.mem.asBytes(&len_ffn[i]), experts[i].ffn, 64);
    }
    e.barrier();
    for (experts, 0..) |_, i| {
        e.dispatch(pipes[i][2].pipe, &.{ wraps[i][2].buf, cx.act[1], cx.act[3] }, &.{ wraps[i][2].off, i * gate_stride, i * out_stride }, std.mem.asBytes(&dims_down[i]), pipes[i][2].groups, group);
    }
    e.barrier();
    // One dispatch for the weighted sum: a `scaled_add` per expert all write
    // the same accumulator and so needed a barrier between each.
    var rd = ReduceDims{ .n = @intCast(experts.len), .dim = @intCast(dim), .w = @splat(0) };
    for (experts, 0..) |ex, i| rd.w[i] = ex.weight;
    e.dispatch(cx.reduce_p, &.{ cx.act[0], cx.act[3] }, &.{ 0, 0 }, std.mem.asBytes(&rd), dim, 64);
    e.end();
    cb.commitAndWait();
    @memcpy(out, cx.act[0].slice(f32)[0..dim]);
    return true;
}

/// Experts one call can record. DeepSeek activates 6 routed plus a shared one.
const MAX_MOE_EXPERTS = 16;

const AccDims = extern struct { n: u32, alpha: f32 };
/// The MLA cache shape asked for at model load, before there is a device.
var mla_want: ?struct { layers: usize, ctx_len: usize, kvr: usize, rope: usize } = null;

const RouteDims = extern struct {
    n_expert: u32,
    n_used: u32,
    gating: u32,
    weights_norm: u32,
    has_bias: u32,
    weights_scale: f32,
};

const IdDims = extern struct { rows: u32, cols: u32, n_used: u32, plane_stride: u32, x_stride: u32 };

const MlaRopeDims = extern struct { n_vec: u32, rope: u32, stride: u32, offset: u32, pos: u32, base: f32, yarn_factor: f32, yarn_orig_ctx: f32 };

const AbsorbDims = extern struct { n_heads: u32, nope: u32, kvr: u32, stride: u32, q_stride: u32, q_off: u32 };

const MlaDims = extern struct { n_heads: u32, kvr: u32, rope: u32, seq: u32, qr_stride: u32, qr_off: u32, scale: f32 };
const MAX_MOE_REDUCE = 16;
const ReduceDims = extern struct { n: u32, dim: u32, w: [MAX_MOE_REDUCE]f32 };

/// Everything one GQA layer needs, so the whole layer is one call.
pub const LayerSpec = struct {
    li: usize,
    pos: usize,
    attn_norm: []const f32,
    ffn_norm: []const f32,
    eps: f32,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    gate: WeightRef,
    up: WeightRef,
    down: WeightRef,
    dim: usize,
    ffn: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hd: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_neox: bool,
    attn_scale: f32,
};

/// Record one whole GQA layer into the open frame: norm, q/k/v projections,
/// RoPE, KV append, attention, output projection, residual, then the FFN
/// block. The residual stream stays in act[0] for the entire token and is
/// never read by the host in between.
///
/// Returns false if anything about the layer is unsupported, in which case the
/// caller runs its own path -- but note that a mid-token decline is not free:
/// the engine has to fall back for the *whole* token, because act[0] is the
/// only copy of the residual once recording has started.
pub fn layerBlock(s: LayerSpec) bool {
    const cx = &(ctx orelse return false);
    const f = &(frame orelse return false);
    const kb = cx.kv_k orelse return false;
    const vb = cx.kv_v orelse return false;

    const qd = s.n_heads * s.hd;
    const kvd = s.n_kv_heads * s.hd;
    const gp = dmmvFor(cx, s.gate.ty, s.ffn) orelse return false;
    const upp = dmmvFor(cx, s.up.ty, s.ffn) orelse return false;
    const dp = dmmvFor(cx, s.down.ty, s.dim) orelse return false;
    const qp = dmmvFor(cx, s.wq.ty, qd) orelse return false;
    const kp = dmmvFor(cx, s.wk.ty, kvd) orelse return false;
    const vp = dmmvFor(cx, s.wv.ty, kvd) orelse return false;
    const op = dmmvFor(cx, s.wo.ty, s.dim) orelse return false;

    if (kvd != cx.kv_kvd or s.pos >= cx.kv_ctx) return false;
    if (s.dim % 256 != 0 or s.ffn % 256 != 0) return false;
    if (@max(s.ffn, qd) * 4 > cx.act[0].len) return false;

    const qw = wrapFor(cx, s.wq.data) orelse return false;
    const kw = wrapFor(cx, s.wk.data) orelse return false;
    const vw = wrapFor(cx, s.wv.data) orelse return false;
    const ow = wrapFor(cx, s.wo.data) orelse return false;
    const gw = wrapFor(cx, s.gate.data) orelse return false;
    const uw = wrapFor(cx, s.up.data) orelse return false;
    const dw = wrapFor(cx, s.down.data) orelse return false;

    // Each layer writes its own slot. See Ctx.norms: a shared scratch buffer
    // is correct only when the dispatch runs before the next host write, which
    // recording breaks.
    const norm_off = s.li * 2 * s.dim;
    if ((norm_off + 2 * s.dim) * 4 > cx.norms.len) return false;
    const nrm = cx.norms.slice(f32);
    @memcpy(nrm[norm_off..][0..s.dim], s.attn_norm);
    @memcpy(nrm[norm_off + s.dim ..][0..s.dim], s.ffn_norm);
    const attn_norm_off = norm_off * @sizeOf(f32);
    const ffn_norm_off = (norm_off + s.dim) * @sizeOf(f32);

    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const nd_attn = extern struct { n: u32, eps: f32 }{ .n = @intCast(s.dim), .eps = s.eps };
    const dims_q = Dims{ .rows = @intCast(qd), .cols = @intCast(s.dim) };
    const dims_kv = Dims{ .rows = @intCast(kvd), .cols = @intCast(s.dim) };
    const dims_o = Dims{ .rows = @intCast(s.dim), .cols = @intCast(qd) };
    const dims_ffn = Dims{ .rows = @intCast(s.ffn), .cols = @intCast(s.dim) };
    const dims_down = Dims{ .rows = @intCast(s.dim), .cols = @intCast(s.ffn) };
    const len_ffn = extern struct { n: u32 }{ .n = @intCast(s.ffn) };
    const len_dim = extern struct { n: u32 }{ .n = @intCast(s.dim) };
    const rope_q = RopeDims{
        .n_heads = @intCast(s.n_heads),
        .hd = @intCast(s.hd),
        .rope_dim = @intCast(s.rope_dim),
        .pos = @intCast(s.pos),
        .base = s.rope_base,
        .neox = if (s.rope_neox) 1 else 0,
    };
    const rope_k = RopeDims{
        .n_heads = @intCast(s.n_kv_heads),
        .hd = @intCast(s.hd),
        .rope_dim = @intCast(s.rope_dim),
        .pos = @intCast(s.pos),
        .base = s.rope_base,
        .neox = if (s.rope_neox) 1 else 0,
    };
    const kvw = KvWriteDims{
        .kvd = @intCast(kvd),
        .row = @intCast((s.li * cx.kv_ctx + s.pos) * kvd),
    };
    const attn = AttnDims{
        .n_heads = @intCast(s.n_heads),
        .n_kv_heads = @intCast(s.n_kv_heads),
        .hd = @intCast(s.hd),
        .seq = @intCast(s.pos + 1),
        .kvd = @intCast(kvd),
        .scale = s.attn_scale,
    };
    const layer_off = s.li * cx.kv_ctx * cx.kv_kvd * @sizeOf(f32);

    // Slots: 0 = x (residual, lives across the whole token), 1 = normed,
    // 2 = q, 3 = k, 4 = v, 5 = attn_out, 6 = gate, 7 = up.
    const e = f.enc;
    // normed = rmsnorm(x) * attn_norm
    e.dispatch(cx.rmsnorm_p, &.{ cx.act[0], cx.norms, cx.act[1] }, &.{ 0, attn_norm_off, 0 }, std.mem.asBytes(&nd_attn), SIMD_W, SIMD_W);
    e.barrier();
    // q, k, v — mutually independent, so no barrier between them
    e.dispatch(qp.pipe, &.{ qw.buf, cx.act[1], cx.act[2] }, &.{ qw.off, 0, 0 }, std.mem.asBytes(&dims_q), qp.groups, group);
    e.dispatch(kp.pipe, &.{ kw.buf, cx.act[1], cx.act[3] }, &.{ kw.off, 0, 0 }, std.mem.asBytes(&dims_kv), kp.groups, group);
    e.dispatch(vp.pipe, &.{ vw.buf, cx.act[1], cx.act[4] }, &.{ vw.off, 0, 0 }, std.mem.asBytes(&dims_kv), vp.groups, group);
    e.barrier();
    // RoPE q and k in place; v is not rotated
    e.dispatch(cx.rope_p, &.{cx.act[2]}, &.{0}, std.mem.asBytes(&rope_q), s.n_heads * (s.rope_dim / 2), 64);
    e.dispatch(cx.rope_p, &.{cx.act[3]}, &.{0}, std.mem.asBytes(&rope_k), s.n_kv_heads * (s.rope_dim / 2), 64);
    e.barrier();
    // append k,v to the device cache at [li][pos]
    e.dispatch(cx.kvw_p, &.{ cx.act[3], cx.act[4], kb, vb }, &.{ 0, 0, 0, 0 }, std.mem.asBytes(&kvw), kvd, 64);
    e.barrier();
    // attention over positions 0..=pos
    e.dispatch(cx.attn_p, &.{ cx.act[2], kb, vb, cx.act[5] }, &.{ 0, layer_off, layer_off, 0 }, std.mem.asBytes(&attn), s.n_heads * 64, 64);
    e.barrier();
    // x += Wo . attn_out   (into act[1], then added)
    e.dispatch(op.pipe, &.{ ow.buf, cx.act[5], cx.act[1] }, &.{ ow.off, 0, 0 }, std.mem.asBytes(&dims_o), op.groups, group);
    e.barrier();
    e.dispatch(cx.add_p, &.{ cx.act[0], cx.act[1] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), s.dim, 64);
    e.barrier();
    // ---- FFN ----
    const nd_ffn = extern struct { n: u32, eps: f32 }{ .n = @intCast(s.dim), .eps = s.eps };
    e.dispatch(cx.rmsnorm_p, &.{ cx.act[0], cx.norms, cx.act[1] }, &.{ 0, ffn_norm_off, 0 }, std.mem.asBytes(&nd_ffn), SIMD_W, SIMD_W);
    e.barrier();
    e.dispatch(gp.pipe, &.{ gw.buf, cx.act[1], cx.act[6] }, &.{ gw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), gp.groups, group);
    e.dispatch(upp.pipe, &.{ uw.buf, cx.act[1], cx.act[7] }, &.{ uw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), upp.groups, group);
    e.barrier();
    e.dispatch(cx.swiglu_p, &.{ cx.act[6], cx.act[7], cx.act[6] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_ffn), s.ffn, 64);
    e.barrier();
    e.dispatch(dp.pipe, &.{ dw.buf, cx.act[6], cx.act[1] }, &.{ dw.off, 0, 0 }, std.mem.asBytes(&dims_down), dp.groups, group);
    e.barrier();
    e.dispatch(cx.add_p, &.{ cx.act[0], cx.act[1] }, &.{ 0, 0 }, std.mem.asBytes(&len_dim), s.dim, 64);
    e.barrier();
    f.dispatches += 15;
    return true;
}

/// Load the residual stream into the device slot the layer chain runs on.
pub fn frameLoadX(x: []const f32) bool {
    const cx = &(ctx orelse return false);
    if (x.len * 4 > cx.act[0].len) return false;
    @memcpy(cx.act[0].slice(f32)[0..x.len], x);
    return true;
}

/// Read it back. Only valid after `endFrame`.
pub fn frameStoreX(x: []f32) bool {
    const cx = &(ctx orelse return false);
    if (x.len * 4 > cx.act[0].len) return false;
    @memcpy(x, cx.act[0].slice(f32)[0..x.len]);
    return true;
}

/// How many output rows one SIMD group of a type's dmmv kernel produces.
///
/// Not a detail the caller can guess: `dmmv_q4k` takes two rows per group
/// (NR0=2, so one activation load feeds both) while `dmmv_q6k` takes one. A
/// dispatch that assumes the wrong number launches too few groups and simply
/// leaves the tail of the output vector holding whatever was there before --
/// no error, no NaN, just half a projection. That is what shipped: TinyLlama's
/// `ffn_down` is Q6_K, dispatched with the Q4_K group count, so every layer's
/// FFN output was half stale and the model produced confident nonsense.
/// Column multiple a type's kernel requires — its block width. Q4_K and Q6_K
/// are 256-value super-blocks; Q5_1 is a 32-value block, which is why
/// DeepSeek's 1408-wide expert `down` is legal for it and not for them.
fn colsMultiple(t: ggml.Type) usize {
    return switch (t) {
        .q4_k, .q6_k => 256,
        .q5_1, .q5_0, .q8_0 => 32,
        .f32 => 1,
        else => 256,
    };
}

fn rowsPerGroup(t: ggml.Type, rows: usize) usize {
    return switch (t) {
        // NR0 in dmmv_q4k.metal, and the wide variant compiled from the same
        // source with NR0 forced to 2.
        .q4_k => if (rows >= Q4K_WIDE_ROWS) 2 else 4,
        else => 1,
    };
}

fn groupsFor(t: ggml.Type, rows: usize) usize {
    const per = rowsPerGroup(t, rows);
    const threads = ((rows + per - 1) / per) * SIMD_W;
    // Round up to a whole threadgroup. `dispatchThreads` permits a smaller
    // final threadgroup, and in one the kernel's `simdgroups_per_threadgroup`
    // is less than the full count -- so `tgid * nsg + sgid` addresses the wrong
    // rows and the real tail is never written. `matvec` then copies whatever
    // the scratch output buffer held from a previous dispatch, so the symptom
    // is a plausible-looking number rather than a NaN.
    //
    // Every shape this had been run on divided evenly (65,536 rows at two per
    // group, 1408 at four), so it never fired; 1409 does. The extra threads
    // return immediately on the row bound.
    const per_group = SIMD_W * SIMDGROUPS_PER_GROUP;
    return ((threads + per_group - 1) / per_group) * per_group;
}

/// Above this many rows, Q4_K uses the two-row kernel. Below it, the weights
/// are small enough to sit in cache and re-reading the activation vector once
/// per group is what costs, so four rows per group wins; above it, more groups
/// in flight matters more. 4096 is between the measured crossover points
/// (2816 still wins at four rows, 5632 loses).
const Q4K_WIDE_ROWS = 4096;

/// The dmmv pipeline for a shape *and* the grid it must be dispatched with.
///
/// One function because these two must agree and nothing else enforces it:
/// Q4_K now has two kernels covering different numbers of rows per SIMD group,
/// and dispatching one with the other's grid silently computes part of the
/// output vector -- no error, no NaN, just a tail holding whatever it held
/// before. That is the worst bug in this work and it has happened twice.
fn dmmvFor(cx: *Ctx, t: ggml.Type, rows: usize) ?struct { pipe: mtl.Pipeline, groups: usize } {
    const pipe = switch (t) {
        .q4_k => if (rows >= Q4K_WIDE_ROWS) cx.q4k_wide else cx.q4k,
        .q6_k => cx.q6k,
        .q5_1 => cx.q5_1,
        .q5_0 => cx.q5_0,
        .q8_0 => cx.q8_0,
        .f32 => cx.f32p,
        else => return null,
    };
    return .{ .pipe = pipe, .groups = groupsFor(t, rows) };
}

pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const cx = &(ctx orelse return cpu.matvec(t, out, data, x, rows, cols));
    const sel = dmmvFor(cx, t, rows);
    // `colsMultiple`, not a hardcoded 256: Q4_K and Q6_K are 256-value
    // super-blocks but Q5_1 is a 32-value block, and a stale 256 here sent
    // every Q5_1 matvec to the CPU. That fallback is silent and the CPU Q5_1
    // kernel is approximate (int8 activations), so it presents as a slightly
    // wrong GPU kernel rather than as an unused one -- which cost a long
    // detour through bit-extraction and summation order before the test was
    // taught to detect a fallback directly.
    if (!use_gpu_ops or !gpu_worthwhile or sel == null or rows < min_rows or cols % colsMultiple(t) != 0) {
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
        sel.?.pipe,
        &.{ w.buf, cx.scratch_x, cx.scratch_out },
        &.{ w.off, 0, 0 },
        std.mem.asBytes(&dims),
        sel.?.groups,
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

test "dispatch grids match each kernel's rows per SIMD group" {
    // Pins the arithmetic that produced the worst bug in this work. The dmmv
    // kernels do not agree on how many rows a SIMD group covers -- Q4_K takes
    // four, Q6_K one -- and a dispatch that assumes the wrong number silently
    // computes only part of the output vector. There is no error and no NaN;
    // the tail simply keeps its previous contents, which for a residual stream
    // is a plausible-looking number.
    //
    // This must track `NR0` in dmmv_q4k.metal, and it caught the change that
    // took NR0 from two to four -- which is the whole reason it exists. Q4_K
    // now depends on the row count as well: the two variants are compiled from
    // one source and picked by shape, so the grid has to follow the pick.
    try std.testing.expectEqual(@as(usize, 4), rowsPerGroup(.q4_k, 1408));
    try std.testing.expectEqual(@as(usize, 4), rowsPerGroup(.q4_k, Q4K_WIDE_ROWS - 1));
    try std.testing.expectEqual(@as(usize, 2), rowsPerGroup(.q4_k, Q4K_WIDE_ROWS));
    try std.testing.expectEqual(@as(usize, 2), rowsPerGroup(.q4_k, 102400));
    try std.testing.expectEqual(@as(usize, 1), rowsPerGroup(.q6_k, 4096));
    // Row counts that do not divide must round up, not down -- and then round
    // again to a whole threadgroup, or the final partial threadgroup reports a
    // smaller `simdgroups_per_threadgroup` and addresses the wrong rows.
    const per_group = SIMD_W * SIMDGROUPS_PER_GROUP;
    for ([_]usize{ 65, 128, 1408, 1409, 2048, 8193, 65536 }) |rows| {
        const g = groupsFor(.q4_k, rows);
        try std.testing.expectEqual(@as(usize, 0), g % per_group);
        // and still covers every row
        try std.testing.expect(g / SIMD_W * rowsPerGroup(.q4_k, rows) >= rows);
    }
    for ([_]usize{ 65, 2048, 8193 }) |rows| {
        const g = groupsFor(.q6_k, rows);
        try std.testing.expectEqual(@as(usize, 0), g % per_group);
        try std.testing.expect(g / SIMD_W * rowsPerGroup(.q6_k, rows) >= rows);
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

test "metal layer block matches the cpu layer, end to end" {
    // A whole layer recorded into one command buffer, against the same
    // sequence run on the host, **over several positions**.
    //
    // The position sweep is the point. An earlier version of this test ran
    // only pos=0 and passed while the engine produced garbage, because at
    // pos=0 the rotation angle is zero -- RoPE is the identity -- and softmax
    // over a single score is 1, so attention is a passthrough. The two things
    // most likely to be wrong were both switched off by the fixture.
    //
    // RoPE is reimplemented here rather than called, so the test is an
    // independent statement of the rotation, not the same code twice.
    const gpa = std.testing.allocator;
    // TinyLlama-1.1B's actual geometry. Small synthetic dims passed while the
    // engine produced nonsense, so the fixture has to match a real model:
    // GQA with 8 query heads per KV head, and a 2048-wide residual that the
    // rmsnorm kernel covers with a single 32-thread group.
    const dim = 2048;
    const ffn = 5632;
    const n_heads = 32;
    const n_kv_heads = 4;
    const hd = 64;
    const rope_dim = 64;
    const rope_base: f32 = 10000.0;
    const eps: f32 = 1e-5;
    // Two layers recorded into ONE frame, with different weights. Recording
    // several layers before submitting is its own failure surface: every
    // buffer they share is written by the host at record time but read by the
    // GPU at submit time, so the last writer wins. That is how all 22 layers
    // ended up using layer 21's norm weights.
    const nlayers_test = 2;
    const layers = 3;
    const ctx_len = 64;
    const qd = n_heads * hd;
    const kvd = n_kv_heads * hd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    if (!attnInit(layers, ctx_len, kvd)) return error.SkipZigTest;

    const saved_use = use_gpu_ops;
    const saved_attn = attn_worthwhile;
    use_gpu_ops = true;
    attn_worthwhile = true;
    defer {
        use_gpu_ops = saved_use;
        attn_worthwhile = saved_attn;
    }

    var prng = std.Random.DefaultPrng.init(0x1A4E5);
    const rnd = prng.random();

    // Random bytes give random f16 super-block scales, which reach ~65504;
    // chaining seven matvecs then produces values around 1e22 where a relative
    // comparison means nothing. Pin d and dmin to modest constants (0.0625 and
    // 0.03125) so the layer stays in a range where a tolerance is meaningful.
    // The quantized nibbles stay random, which is what the kernel indexes.
    const mk = struct {
        // The alignment has to survive into the return type or the caller frees
        // with alignment 1 and the allocator rejects it.
        fn f(g: std.mem.Allocator, r: std.Random, rows: usize, cols: usize) ![]align(16384) u8 {
            const d = try g.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
            r.bytes(d);
            var b: usize = 0;
            while (b + 144 <= d.len) : (b += 144) {
                std.mem.writeInt(u16, d[b..][0..2], 0x2C00, .little); // d = 0.0625
                std.mem.writeInt(u16, d[b + 2 ..][0..2], 0x2800, .little); // dmin = 0.03125
            }
            return d;
        }
    }.f;

    var wq: [nlayers_test][]align(16384) u8 = undefined;
    var wk: [nlayers_test][]align(16384) u8 = undefined;
    var wv: [nlayers_test][]align(16384) u8 = undefined;
    var wo: [nlayers_test][]align(16384) u8 = undefined;
    var wg: [nlayers_test][]align(16384) u8 = undefined;
    var wu: [nlayers_test][]align(16384) u8 = undefined;
    var wd: [nlayers_test][]align(16384) u8 = undefined;
    var an: [nlayers_test][]f32 = undefined;
    var fn_: [nlayers_test][]f32 = undefined;
    for (0..nlayers_test) |L| {
        wq[L] = try mk(gpa, rnd, qd, dim);
        wk[L] = try mk(gpa, rnd, kvd, dim);
        wv[L] = try mk(gpa, rnd, kvd, dim);
        wo[L] = try mk(gpa, rnd, dim, qd);
        wg[L] = try mk(gpa, rnd, ffn, dim);
        wu[L] = try mk(gpa, rnd, ffn, dim);
        wd[L] = try mk(gpa, rnd, dim, ffn);
        an[L] = try gpa.alloc(f32, dim);
        fn_[L] = try gpa.alloc(f32, dim);
        for (an[L]) |*v| v.* = 0.5 + rnd.float(f32);
        for (fn_[L]) |*v| v.* = 0.5 + rnd.float(f32);
    }
    defer for (0..nlayers_test) |L| {
        gpa.free(wq[L]);
        gpa.free(wk[L]);
        gpa.free(wv[L]);
        gpa.free(wo[L]);
        gpa.free(wg[L]);
        gpa.free(wu[L]);
        gpa.free(wd[L]);
        gpa.free(an[L]);
        gpa.free(fn_[L]);
    };

    const x0 = try gpa.alloc(f32, dim);
    defer gpa.free(x0);
    for (x0) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const nsteps = 5;
    const specFor = struct {
        fn f(L: usize, pos: usize, a: []const f32, fnm: []const f32, q_: []const u8, k_: []const u8, v_: []const u8, o_: []const u8, g_: []const u8, u_: []const u8, d_: []const u8) LayerSpec {
            return .{
                .li = L,
                .pos = pos,
                .attn_norm = a,
                .ffn_norm = fnm,
                .eps = eps,
                .wq = .{ .ty = .q4_k, .data = q_ },
                .wk = .{ .ty = .q4_k, .data = k_ },
                .wv = .{ .ty = .q4_k, .data = v_ },
                .wo = .{ .ty = .q4_k, .data = o_ },
                .gate = .{ .ty = .q4_k, .data = g_ },
                .up = .{ .ty = .q4_k, .data = u_ },
                .down = .{ .ty = .q4_k, .data = d_ },
                .dim = dim,
                .ffn = ffn,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .hd = hd,
                .rope_dim = rope_dim,
                .rope_base = rope_base,
                .rope_neox = false,
                .attn_scale = scale,
            };
        }
    }.f;

    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);
    const want = try gpa.alloc(f32, dim);
    defer gpa.free(want);
    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    const q = try gpa.alloc(f32, qd);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, kvd);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, kvd);
    defer gpa.free(v);
    const ao = try gpa.alloc(f32, qd);
    defer gpa.free(ao);
    const tmp = try gpa.alloc(f32, @max(dim, ffn));
    defer gpa.free(tmp);
    const gbuf = try gpa.alloc(f32, ffn);
    defer gpa.free(gbuf);
    const ubuf = try gpa.alloc(f32, ffn);
    defer gpa.free(ubuf);
    // The host keeps its own cache, so the comparison covers the KV row
    // offsets as well as the arithmetic.
    const kc = try gpa.alloc(f32, nlayers_test * nsteps * kvd);
    defer gpa.free(kc);
    const vc = try gpa.alloc(f32, nlayers_test * nsteps * kvd);
    defer gpa.free(vc);
    const scores = try gpa.alloc(f32, nsteps);
    defer gpa.free(scores);

    const exact = struct {
        fn f(g: std.mem.Allocator, o: []f32, data: []const u8, xv: []const f32, rows: usize, cols: usize) !void {
            const rowbuf = try g.alloc(f32, cols);
            defer g.free(rowbuf);
            for (0..rows) |r| {
                cpu.dequantRow(.q4_k, rowbuf, data, r, cols);
                var a: f32 = 0;
                for (rowbuf, xv) |wv2, x2| a += wv2 * x2;
                o[r] = a;
            }
        }
    }.f;
    // NORM RoPE, stated independently of the engine's implementation.
    const rope = struct {
        fn f(vec: []f32, nh: usize, head_dim: usize, rd: usize, pos: usize, base: f32) void {
            for (0..nh) |hh| {
                const head = vec[hh * head_dim ..][0..head_dim];
                var i: usize = 0;
                while (i + 1 < rd) : (i += 2) {
                    const fr = std.math.pow(f32, base, -@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rd)));
                    const ang = @as(f32, @floatFromInt(pos)) * fr;
                    const c = @cos(ang);
                    const sn = @sin(ang);
                    const a = head[i];
                    const b = head[i + 1];
                    head[i] = a * c - b * sn;
                    head[i + 1] = a * sn + b * c;
                }
            }
        }
    }.f;
    const q_per_kv = n_heads / n_kv_heads;

    for (0..nsteps) |pos| {
        // Same input at every position, so any difference between the two is
        // the position-dependent machinery and nothing else.
        if (!beginFrame()) return error.SkipZigTest;
        if (!frameLoadX(x0)) return error.SkipZigTest;
        for (0..nlayers_test) |L| {
            if (!layerBlock(specFor(L, pos, an[L], fn_[L], wq[L], wk[L], wv[L], wo[L], wg[L], wu[L], wd[L]))) {
                endFrame();
                return error.SkipZigTest;
            }
        }
        endFrame();
        if (!frameStoreX(got)) return error.SkipZigTest;

        // ---- host reference: the same two layers, in the same order ----
        @memcpy(want, x0);
        for (0..nlayers_test) |L| {
            cpu.rmsnorm(normed, want, an[L], eps);
            try exact(gpa, q, wq[L], normed, qd, dim);
            try exact(gpa, k, wk[L], normed, kvd, dim);
            try exact(gpa, v, wv[L], normed, kvd, dim);
            rope(q, n_heads, hd, rope_dim, pos, rope_base);
            rope(k, n_kv_heads, hd, rope_dim, pos, rope_base);
            const cbase = (L * nsteps + pos) * kvd;
            @memcpy(kc[cbase..][0..kvd], k);
            @memcpy(vc[cbase..][0..kvd], v);

            const seq = pos + 1;
            for (0..n_heads) |hh| {
                const kvh = hh / q_per_kv;
                const qh = q[hh * hd ..][0..hd];
                for (0..seq) |t| {
                    scores[t] = cpu.dotF32(qh, kc[(L * nsteps + t) * kvd + kvh * hd ..][0..hd]) * scale;
                }
                cpu.softmax(scores[0..seq]);
                const oh = ao[hh * hd ..][0..hd];
                @memset(oh, 0);
                for (0..seq) |t| cpu.axpy(oh, vc[(L * nsteps + t) * kvd + kvh * hd ..][0..hd], scores[t]);
            }
            try exact(gpa, tmp[0..dim], wo[L], ao, dim, qd);
            cpu.add(want, tmp[0..dim]);

            cpu.rmsnorm(normed, want, fn_[L], eps);
            try exact(gpa, gbuf, wg[L], normed, ffn, dim);
            try exact(gpa, ubuf, wu[L], normed, ffn, dim);
            cpu.swiglu(gbuf, gbuf, ubuf);
            try exact(gpa, tmp[0..dim], wd[L], gbuf, dim, ffn);
            cpu.add(want, tmp[0..dim]);
        }

        var mass: f32 = 0;
        for (want) |wv2| mass += @abs(wv2);
        // Against the mean magnitude of the result, not each element: a
        // residual stream has elements near zero where a relative bound is
        // meaningless.
        const tol = (mass / @as(f32, @floatFromInt(dim))) * 2e-3;
        for (0..dim) |i| {
            std.testing.expectApproxEqAbs(want[i], got[i], tol) catch |e| {
                std.debug.print("layer block pos {d} dim {d}: cpu {d} vs gpu {d} (tol {d})\n", .{ pos, i, want[i], got[i], tol });
                return e;
            };
        }
    }
}

test "metal q5_1 matvec agrees with the exact cpu reference" {
    // Q5_1 is affine (d*q + m) with a 32-value block, so unlike Q4_K there is
    // no per-sub-block scale and no bias to separate — the thing to get wrong
    // is the fifth bit, which lives in a u32 at byte 4 with value i at bit i
    // and value i+16 at bit i+16. Dropping it silently halves every weight's
    // range and still produces plausible output.
    const gpa = std.testing.allocator;
    const cols = 1408; // DeepSeek's expert `down` width: 44 blocks of 32
    const rows = 256;

    var prng = std.Random.DefaultPrng.init(0x51A1);
    const rnd = prng.random();
    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q5_1, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    // Pin d and m rather than leaving them random.
    //
    // A 1408-column Q5_1 row is 44 blocks against Q4_K's 5 super-blocks, and
    // random f16 scales span five orders of magnitude. Summing 44 terms of
    // wildly different size in f32 is ill-conditioned whatever the kernel
    // does, so the comparison measures summation order rather than
    // correctness -- the first version of this test failed at 2.85e-4 of mass
    // for exactly that reason. The quantized nibbles and the fifth bits, which
    // are what the kernel indexes, stay random.
    var b: usize = 0;
    while (b + 24 <= data.len) : (b += 24) {
        std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little); // d = 0.0625
        std.mem.writeInt(u16, data[b + 2 ..][0..2], 0x2800, .little); // m = 0.03125
    }

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
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
    matvec(.q5_1, got, data, x, rows, cols);

    // Did the GPU path actually run? `matvec` falls back to `cpu.matvec` for
    // any reason it declines, and the CPU Q5_1 kernel quantizes activations to
    // int8 -- approximate by ~0.4% per element. A silent fallback therefore
    // looks exactly like a slightly-wrong kernel, which is how the first
    // version of this test was read for far too long.
    const cpu_ref = try gpa.alloc(f32, rows);
    defer gpa.free(cpu_ref);
    cpu.matvec(.q5_1, cpu_ref, data, x, rows, cols);
    var identical: usize = 0;
    for (got, cpu_ref) |g, c| {
        if (g == c) identical += 1;
    }
    if (identical == rows) {
        std.debug.print("q5_1: matvec fell back to the CPU -- the GPU path never ran\n", .{});
        return error.SkipZigTest;
    }

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    // The reference accumulates in f64, which matters at this width.
    //
    // A 1408-column row is 1408 terms, and a sequential f32 sum carries an
    // error bound around n*eps -- ~1.7e-4 of the summed mass. The GPU sums
    // 32-64 terms per lane and then tree-reduces, so it is *more* accurate
    // than an f32 oracle, and the first version of this test duly reported the
    // oracle's own drift as a kernel bug. It was chased through pinned scales
    // and an all-positive input before the arithmetic was checked.
    for (0..rows) |r| {
        cpu.dequantRow(.q5_1, row, data, r, cols);
        var acc64: f64 = 0;
        var mass64: f64 = 0;
        for (row, x) |wv, xv| {
            acc64 += @as(f64, wv) * @as(f64, xv);
            mass64 += @abs(@as(f64, wv) * @as(f64, xv));
        }
        const want: f32 = @floatCast(acc64);
        const mass: f32 = @floatCast(mass64);
        // With the scales pinned this is back to ordinary reassociation, so
        // the same 1e-5 of summed mass the other kernel tests use. A wrong
        // fifth bit or a swapped nibble half moves a result by percent.
        const tol = mass * 1e-5;
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("metal q5_1 row {d}: reference {d} vs gpu {d} (rel-to-mass {e:.2})\n", .{ r, want, got[r], @abs(want - got[r]) / mass });
            return e;
        };
    }
}

test "metal moe block matches the cpu expert loop" {
    // Several experts and a weighted sum in one command buffer. What this has
    // to catch is the accumulation: a dropped or double-applied router weight
    // still produces a well-formed vector, and with random weights the result
    // looks equally plausible either way. Three experts with deliberately
    // different gates, so a mixed-up ordering shows.
    const gpa = std.testing.allocator;
    const dim = 2048;
    const ffn = 1408; // DeepSeek-V2-Lite's expert width
    const n_exp = 3;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;

    var prng = std.Random.DefaultPrng.init(0x30E5);
    const rnd = prng.random();
    // Types as the real checkpoint has them: gate and up are Q4_K over `dim`
    // columns, down is Q5_1 over `ffn`. That is not incidental -- 1408 is not
    // a multiple of 256, so a Q4_K `down` is not a legal tensor, and the first
    // version of this test built one and was correctly refused.
    const mk = struct {
        fn f(g: std.mem.Allocator, r: std.Random, ty: ggml.Type, rows: usize, cols: usize) ![]align(16384) u8 {
            const d = try g.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(ty, cols, rows));
            r.bytes(d);
            const bs: usize = if (ty == .q4_k) 144 else 24;
            var b: usize = 0;
            while (b + bs <= d.len) : (b += bs) {
                std.mem.writeInt(u16, d[b..][0..2], 0x2C00, .little);
                std.mem.writeInt(u16, d[b + 2 ..][0..2], 0x2800, .little);
            }
            return d;
        }
    }.f;

    var wg: [n_exp][]align(16384) u8 = undefined;
    var wu: [n_exp][]align(16384) u8 = undefined;
    var wd: [n_exp][]align(16384) u8 = undefined;
    for (0..n_exp) |i| {
        wg[i] = try mk(gpa, rnd, .q4_k, ffn, dim);
        wu[i] = try mk(gpa, rnd, .q4_k, ffn, dim);
        wd[i] = try mk(gpa, rnd, .q5_1, dim, ffn);
    }
    defer for (0..n_exp) |i| {
        gpa.free(wg[i]);
        gpa.free(wu[i]);
        gpa.free(wd[i]);
    };

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    for (normed) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const gates = [n_exp]f32{ 0.6, 0.25, 0.15 };

    var refs: [n_exp]ExpertRef = undefined;
    for (0..n_exp) |i| refs[i] = .{
        .gate = .{ .ty = .q4_k, .data = wg[i] },
        .up = .{ .ty = .q4_k, .data = wu[i] },
        .down = .{ .ty = .q5_1, .data = wd[i] },
        .weight = gates[i],
        .ffn = ffn,
    };

    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);
    if (!moeFfnBlock(normed, &refs, got)) return error.SkipZigTest;

    // Reference: exact dequantize-then-dot, not cpu.matvec — the CPU kernel
    // quantizes activations and chaining four of them per expert compounds
    // into drift this would report as a GPU bug.
    const exact = struct {
        fn f(g: std.mem.Allocator, ty: ggml.Type, o: []f32, data: []const u8, xv: []const f32, rows: usize, cols: usize) !void {
            const rb = try g.alloc(f32, cols);
            defer g.free(rb);
            for (0..rows) |r| {
                cpu.dequantRow(ty, rb, data, r, cols);
                var a: f32 = 0;
                for (rb, xv) |wv, x2| a += wv * x2;
                o[r] = a;
            }
        }
    }.f;

    const want = try gpa.alloc(f32, dim);
    defer gpa.free(want);
    @memset(want, 0);
    const gbuf = try gpa.alloc(f32, ffn);
    defer gpa.free(gbuf);
    const ubuf = try gpa.alloc(f32, ffn);
    defer gpa.free(ubuf);
    const obuf = try gpa.alloc(f32, dim);
    defer gpa.free(obuf);
    for (0..n_exp) |i| {
        try exact(gpa, .q4_k, gbuf, wg[i], normed, ffn, dim);
        try exact(gpa, .q4_k, ubuf, wu[i], normed, ffn, dim);
        cpu.swiglu(gbuf, gbuf, ubuf);
        try exact(gpa, .q5_1, obuf, wd[i], gbuf, dim, ffn);
        for (want, obuf) |*a, v| a.* += gates[i] * v;
    }

    var mass: f32 = 0;
    for (want) |v| mass += @abs(v);
    const tol = (mass / @as(f32, @floatFromInt(dim))) * 2e-3;
    for (0..dim) |i| {
        std.testing.expectApproxEqAbs(want[i], got[i], tol) catch |err| {
            std.debug.print("moe block dim {d}: cpu {d} vs gpu {d} (tol {d})\n", .{ i, want[i], got[i], tol });
            return err;
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
    if (!calibrating and (!use_gpu_ops or !ffn_worthwhile) and calibrated) return false;
    // Per tensor, not one type for all three. A `Q4_K_M` checkpoint is a
    // mixture: llama.cpp puts gate and up in Q4_K and `ffn_down` in Q6_K (or
    // Q5_1, on DeepSeek). Requiring a single type meant this block declined
    // every real model while accepting synthetic ones, so it looked
    // implemented and never ran.
    const gp = dmmvFor(cx, gate_w.ty, ffn) orelse return false;
    const up = dmmvFor(cx, up_w.ty, ffn) orelse return false;
    const dp = dmmvFor(cx, down_w.ty, dim) orelse return false;
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
    // `ffn * SIMD_W` here assumed one row per SIMD group whatever the kernel,
    // which over-dispatched by the Q4_K factor -- harmless only because a
    // group past the last row returns immediately.
    e.dispatch(gp.pipe, &.{ gw.buf, cx.act[1], cx.act[2] }, &.{ gw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), gp.groups, group);
    e.dispatch(up.pipe, &.{ uw.buf, cx.act[1], cx.act[3] }, &.{ uw.off, 0, 0 }, std.mem.asBytes(&dims_ffn), up.groups, group);
    e.barrier();
    // act = silu(gate) * up, in place over act[2]
    e.dispatch(cx.swiglu_p, &.{ cx.act[2], cx.act[3], cx.act[2] }, &.{ 0, 0, 0 }, std.mem.asBytes(&len_ffn), ffn, 64);
    e.barrier();
    // ffn_out = Wd . act
    e.dispatch(dp.pipe, &.{ dw.buf, cx.act[2], cx.act[3] }, &.{ dw.off, 0, 0 }, std.mem.asBytes(&dims_down), dp.groups, group);
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

    // Print the measured diffs only on failure: under `zig build test` any
    // stderr from a passing test makes zig 0.16's build runner display the
    // step through its failure path, tagged with a misleading red
    // "failed command:" line (cosmetic -- the build still succeeds).
    if (worst_norm >= 1e-4 or worst_sw >= 1e-5)
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
    // 1408 and 2816 are DeepSeek-V2-Lite's routed and shared expert widths,
    // and 2048 its model dim -- the shapes an expert FFN actually issues. The
    // sweep used to start at 2048 and so never covered the regime the engine
    // spends half a token in.
    for ([_]usize{ 1408, 2048, 2816, 5632, 16384, 32000, 65536, 131072 }) |rows| {
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
        // Through the same selector the engine uses, not a hardcoded pipeline
        // and grid. This had `cx.q4k` with `((rows + 1) / 2) * SIMD_W` left
        // over from when Q4_K covered two rows per group; against the four-row
        // kernel that dispatches twice the groups needed and measures the
        // waste, which is why it reported 47 GB/s at 1408 rows where an
        // honest grid gives 89.
        const sel = dmmvFor(cx, .q4_k, rows) orelse continue;

        const N = 20;
        // warm
        {
            const cb = cx.dev.commandBuffer();
            cb.dispatch(sel.pipe, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), sel.groups, group);
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
            cb.dispatch(sel.pipe, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), sel.groups, group);
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
            for (0..D) |_| e.dispatch(sel.pipe, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), sel.groups, group);
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

test "metal q5_0 matvec agrees with the exact cpu reference" {
    // Q5_0 is symmetric (d*(q-16)) with a 32-value block. Two things are easy to
    // get wrong and both stay plausible: the fifth bit, which lives in a u32 at
    // byte 2 with value i at bit i and value i+16 at bit i+16, and the -16
    // offset, which has to be applied to every element and not to the block
    // sum. Dropping either shifts results without making them obviously wrong.
    const gpa = std.testing.allocator;
    const cols = 1408; // DeepSeek's expert `down` width: 44 blocks of 32
    const rows = 256;

    var prng = std.Random.DefaultPrng.init(0x50A0);
    const rnd = prng.random();
    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q5_0, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    // Pin the scale, as the Q5_1 test does and for the same reason: random f16
    // scales span five orders of magnitude, so summing 44 blocks in f32 is
    // ill-conditioned whatever the kernel does and the comparison would
    // measure summation order rather than correctness. The quants stay random.
    var b: usize = 0;
    while (b + 22 <= data.len) : (b += 22) {
        std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little); // d = 0.0625
    }

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
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
    matvec(.q5_0, got, data, x, rows, cols);

    // Did the GPU path actually run? `matvec` falls back to `cpu.matvec` for
    // any reason it declines, and the CPU kernel is int8-approximate, so a
    // silent fallback presents as a slightly-wrong kernel rather than an
    // unused one. This is the check that would have caught the real bug here:
    // the type had no pipeline at all, so every expert `down` in the
    // checkpoint took the host path and the batched MoE block never ran.
    const cpu_ref = try gpa.alloc(f32, rows);
    defer gpa.free(cpu_ref);
    cpu.matvec(.q5_0, cpu_ref, data, x, rows, cols);
    var identical: usize = 0;
    for (got, cpu_ref) |g, c| {
        if (g == c) identical += 1;
    }
    if (identical == rows) {
        std.debug.print("q5_0: matvec fell back to the CPU -- the GPU path never ran\n", .{});
        return error.SkipZigTest;
    }

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    // f64 oracle: a sequential f32 sum over 1408 terms drifts by ~1.7e-4 of
    // the summed mass, which is larger than the tolerance below, so an f32
    // reference would report its own error as the kernel's.
    for (0..rows) |r| {
        cpu.dequantRow(.q5_0, row, data, r, cols);
        var acc64: f64 = 0;
        var mass64: f64 = 0;
        for (row, x) |wv, xv| {
            acc64 += @as(f64, wv) * @as(f64, xv);
            mass64 += @abs(@as(f64, wv) * @as(f64, xv));
        }
        const want: f32 = @floatCast(acc64);
        const mass: f32 = @floatCast(mass64);
        const tol = mass * 1e-5;
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("metal q5_0 row {d}: reference {d} vs gpu {d} (rel-to-mass {e:.2})\n", .{ r, want, got[r], @abs(want - got[r]) / mass });
            return e;
        };
    }
}

test "metal q8_0 matvec agrees with the exact cpu reference" {
    // Q8_0 has no unpacking at all -- the quants are already signed bytes -- so
    // the only thing to get wrong is signedness. Reading them as unsigned turns
    // every negative weight into a large positive one, which a model absorbs
    // into plausible-looking text rather than failing.
    const gpa = std.testing.allocator;
    const cols = 1408; // DeepSeek's expert `down` width: 44 blocks of 32
    const rows = 256;

    var prng = std.Random.DefaultPrng.init(0x80A0);
    const rnd = prng.random();
    const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q8_0, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    // Pin the scale, as the Q5_1 test does and for the same reason: random f16
    // scales span five orders of magnitude, so summing 44 blocks in f32 is
    // ill-conditioned whatever the kernel does and the comparison would
    // measure summation order rather than correctness. The quants stay random.
    var b: usize = 0;
    while (b + 34 <= data.len) : (b += 34) {
        std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little); // d = 0.0625
    }

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
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
    matvec(.q8_0, got, data, x, rows, cols);

    // Did the GPU path actually run? `matvec` falls back to `cpu.matvec` for
    // any reason it declines, and the CPU kernel is int8-approximate, so a
    // silent fallback presents as a slightly-wrong kernel rather than an
    // unused one. This is the check that would have caught the real bug here:
    // the type had no pipeline at all, so every expert `down` in the
    // checkpoint took the host path and the batched MoE block never ran.
    const cpu_ref = try gpa.alloc(f32, rows);
    defer gpa.free(cpu_ref);
    cpu.matvec(.q8_0, cpu_ref, data, x, rows, cols);
    var identical: usize = 0;
    for (got, cpu_ref) |g, c| {
        if (g == c) identical += 1;
    }
    if (identical == rows) {
        std.debug.print("q8_0: matvec fell back to the CPU -- the GPU path never ran\n", .{});
        return error.SkipZigTest;
    }

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    // f64 oracle: a sequential f32 sum over 1408 terms drifts by ~1.7e-4 of
    // the summed mass, which is larger than the tolerance below, so an f32
    // reference would report its own error as the kernel's.
    for (0..rows) |r| {
        cpu.dequantRow(.q8_0, row, data, r, cols);
        var acc64: f64 = 0;
        var mass64: f64 = 0;
        for (row, x) |wv, xv| {
            acc64 += @as(f64, wv) * @as(f64, xv);
            mass64 += @abs(@as(f64, wv) * @as(f64, xv));
        }
        const want: f32 = @floatCast(acc64);
        const mass: f32 = @floatCast(mass64);
        const tol = mass * 1e-5;
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("metal q8_0 row {d}: reference {d} vs gpu {d} (rel-to-mass {e:.2})\n", .{ r, want, got[r], @abs(want - got[r]) / mass });
            return e;
        };
    }
}

test "moe block accepts every down-projection type real checkpoints use" {
    // The bug this exists to prevent, stated plainly: DeepSeek-V2-Lite Q4_K_M
    // stores `ffn_down_exps` as Q5_0 in some layers and Q8_0 in others, neither
    // of which had a pipeline -- so `moeFfnBlock` declined at
    // `pipelineFor(e.down.ty)` for every MoE layer in the checkpoint and the
    // batched path was dead on the one model it was written for.
    //
    // It survived because the block's other test treats a decline as
    // `error.SkipZigTest`. A backend that refuses everything passes a suite
    // built that way. So this asserts acceptance rather than agreement -- the
    // numbers are the other test's job -- and it *fails* on a decline.
    const gpa = std.testing.allocator;
    const dim = 2048;
    const ffn = 1408;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;

    var prng = std.Random.DefaultPrng.init(0xD0);
    const rnd = prng.random();

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    for (normed) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);

    // gate and up are Q4_K over `dim` in every mix that ships; `down` is the
    // one that varies, and 1408 is not a multiple of 256 so it can never be a
    // K-quant.
    const wg = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn));
    defer gpa.free(wg);
    const wu = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn));
    defer gpa.free(wu);
    rnd.bytes(wg);
    rnd.bytes(wu);

    for ([_]ggml.Type{ .q5_0, .q8_0, .q5_1 }) |down_ty| {
        const wd = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(down_ty, ffn, dim));
        defer gpa.free(wd);
        rnd.bytes(wd);
        const refs = [_]ExpertRef{.{
            .gate = .{ .ty = .q4_k, .data = wg },
            .up = .{ .ty = .q4_k, .data = wu },
            .down = .{ .ty = down_ty, .data = wd },
            .weight = 1.0,
            .ffn = ffn,
        }};
        if (!moeFfnBlock(normed, &refs, got)) {
            std.debug.print("moe block declined a {t} down-projection: every MoE layer of such a checkpoint falls back to the host\n", .{down_ty});
            return error.MoeBlockDeclinedARealType;
        }
    }
}

test "both q4_k kernels agree with the oracle, on each side of the threshold" {
    // Q4_K is two pipelines compiled from one source with different NR0, chosen
    // by row count. The existing matvec test runs at MIN_GPU_ROWS (65,536) and
    // so only ever exercises the wide one; the narrow one is covered only
    // incidentally, by the MoE block tests at 1408 rows. Cover both on purpose,
    // because a variant that is never run under an oracle is a variant nobody
    // knows is correct.
    const gpa = std.testing.allocator;
    const cols = 2048;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
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

    // 1408 picks NR0=4; 8192 picks NR0=2. 1409 and 8193 additionally leave a
    // partial group at the end, which is where a rows-per-group mistake shows.
    for ([_]usize{ 1408, 1409, 8192, 8193 }) |rows| {
        var prng = std.Random.DefaultPrng.init(0xC0FFEE + rows);
        const rnd = prng.random();
        const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
        defer gpa.free(data);
        rnd.bytes(data);
        var b: usize = 0;
        while (b + 144 <= data.len) : (b += 144) {
            std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little); // d
            std.mem.writeInt(u16, data[b + 2 ..][0..2], 0x2800, .little); // dmin
        }
        const x = try gpa.alloc(f32, cols);
        defer gpa.free(x);
        for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

        const got = try gpa.alloc(f32, rows);
        defer gpa.free(got);
        @memset(got, std.math.nan(f32));
        if (rows * 4 > ctx.?.scratch_out.len) continue;
        matvec(.q4_k, got, data, x, rows, cols);

        const row = try gpa.alloc(f32, cols);
        defer gpa.free(row);
        for (0..rows) |r| {
            cpu.dequantRow(.q4_k, row, data, r, cols);
            var acc64: f64 = 0;
            var mass64: f64 = 0;
            for (row, x) |wv, xv| {
                acc64 += @as(f64, wv) * @as(f64, xv);
                mass64 += @abs(@as(f64, wv) * @as(f64, xv));
            }
            const want: f32 = @floatCast(acc64);
            const tol: f32 = @floatCast(mass64 * 1e-5);
            std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
                std.debug.print("q4_k rows={d} row {d}: reference {d} vs gpu {d}\n", .{ rows, r, want, got[r] });
                return e;
            };
        }
    }
}

test "q4_k variants, interleaved so drift cannot favour one" {
    // Why this exists rather than another sweep: comparing kernel variants
    // across separate runs does not work on this machine. The same binary
    // reported 62.4, 52.9, 42.8 and 46.1 GB/s at 2816 rows on four
    // consecutive runs -- a +-20% band that swallows every difference between
    // the variants being compared, and which produced two confident wrong
    // conclusions before it was noticed.
    //
    // Best-of-N inside one run is not enough either, because the drift is
    // between runs. So the variants are interleaved round-by-round within a
    // single run: whatever the window server is doing, it is doing it to all
    // three at once, and best-of over rounds then means the same thing for
    // each.
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
    const group = SIMD_W * SIMDGROUPS_PER_GROUP;
    const Variant = struct { name: []const u8, pipe: mtl.Pipeline, per: usize };
    const variants = [_]Variant{
        .{ .name = "NR0=4", .pipe = cx.q4k, .per = 4 },
        .{ .name = "NR0=2", .pipe = cx.q4k_wide, .per = 2 },
    };

    std.debug.print("\n  rows   {s:>14} {s:>14}   (GB/s, best of 40 interleaved rounds)\n", .{ variants[0].name, variants[1].name });
    for ([_]usize{ 1408, 2048, 2816 }) |rows| {
        const data = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, cols, rows));
        defer gpa.free(data);
        var prng = std.Random.DefaultPrng.init(7);
        prng.random().bytes(data);
        for (0..data.len / 2) |i| data[i * 2 + 1] &= 0xFB;
        const x = try gpa.alloc(f32, cols);
        defer gpa.free(x);
        for (x) |*v| v.* = prng.random().float(f32) - 0.5;
        if (rows * 4 > cx.scratch_out.len) continue;
        @memcpy(cx.scratch_x.slice(f32)[0..cols], x);
        const w = wrapFor(cx, data) orelse continue;
        const dims = Dims{ .rows = @intCast(rows), .cols = @intCast(cols) };

        var best: [variants.len]i128 = @splat(std.math.maxInt(i64));
        const D = 32; // dispatches per buffer: amortizes the ~262 us submission
        for (0..40) |_| {
            for (variants, 0..) |v, vi| {
                const threads = ((rows + v.per - 1) / v.per) * SIMD_W;
                const grid = ((threads + group - 1) / group) * group;
                const t0 = now(io);
                const cb = cx.dev.commandBuffer();
                const e = cb.encoder();
                for (0..D) |_| e.dispatch(v.pipe, &.{ w.buf, cx.scratch_x, cx.scratch_out }, &.{ w.off, 0, 0 }, std.mem.asBytes(&dims), grid, group);
                e.end();
                cb.commitAndWait();
                const dt = @divTrunc(now(io) - t0, D);
                if (dt < best[vi]) best[vi] = dt;
            }
        }
        const bytes: f64 = @floatFromInt(ggml.tensorBytes(.q4_k, cols, rows));
        var gbs: [variants.len]f64 = undefined;
        for (best, 0..) |b, i| gbs[i] = bytes / @as(f64, @floatFromInt(b));
        std.debug.print("  {d:>5}   {d:>14.1} {d:>14.1}\n", .{ rows, gbs[0], gbs[1] });
    }
}

test "moe block: back to back against a host gap between calls" {
    // The block measures ~11 ms/token over 26 layers in isolation; the engine
    // spends far more. The difference in how it is called is that the engine
    // does attention, norms and routing between layers, leaving the GPU idle
    // for around a millisecond each time. If the device clocks down across
    // those gaps, it shows here and nowhere else.
    //
    // The gap is spun on the CPU rather than slept, because that is what the
    // engine does -- host work, not a wait.
    if (@import("builtin").mode != .ReleaseFast) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const dim = 2048;
    const ffn = 1408;
    const n_exp = 7; // six routed plus the shared one, as a layer now issues

    parallelBegin(8);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;
    var t: std.Io.Threaded = .init(gpa, .{});
    defer t.deinit();
    const io = t.io();
    const now = struct {
        fn f(i: std.Io) i128 {
            return std.Io.Clock.Timestamp.now(i, .awake).raw.toNanoseconds();
        }
    }.f;

    var prng = std.Random.DefaultPrng.init(0xB10C);
    const rnd = prng.random();
    var wg: [n_exp][]align(16384) u8 = undefined;
    var wu: [n_exp][]align(16384) u8 = undefined;
    var wd: [n_exp][]align(16384) u8 = undefined;
    for (0..n_exp) |i| {
        wg[i] = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn));
        wu[i] = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn));
        wd[i] = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q5_0, ffn, dim));
        rnd.bytes(wg[i]);
        rnd.bytes(wu[i]);
        rnd.bytes(wd[i]);
        for (0..wg[i].len / 2) |k| wg[i][k * 2 + 1] &= 0xFB;
        for (0..wu[i].len / 2) |k| wu[i][k * 2 + 1] &= 0xFB;
    }
    defer for (0..n_exp) |i| {
        gpa.free(wg[i]);
        gpa.free(wu[i]);
        gpa.free(wd[i]);
    };

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    for (normed) |*v| v.* = 0.1;
    const out = try gpa.alloc(f32, dim);
    defer gpa.free(out);

    var refs: [n_exp]ExpertRef = undefined;
    for (0..n_exp) |i| refs[i] = .{
        .gate = .{ .ty = .q4_k, .data = wg[i] },
        .up = .{ .ty = .q4_k, .data = wu[i] },
        .down = .{ .ty = .q5_0, .data = wd[i] },
        .weight = 1.0 / @as(f32, n_exp),
        .ffn = ffn,
    };
    if (!moeFfnBlock(normed, &refs, out)) return error.SkipZigTest;

    var tight: i128 = std.math.maxInt(i64);
    for (0..30) |_| {
        const t0 = now(io);
        _ = moeFfnBlock(normed, &refs, out);
        const dt = now(io) - t0;
        if (dt < tight) tight = dt;
    }
    // A rotating working set, which is what the engine actually has: every
    // layer of every token reads a different set of experts, ~1.1 GB per
    // token, where the loop above reads the same 35 MB thirty times over.
    const SETS = 20;
    var many: [SETS][n_exp]ExpertRef = undefined;
    var pool: [SETS][n_exp][3][]align(16384) u8 = undefined;
    var built: usize = 0;
    outer: for (0..SETS) |sx| {
        for (0..n_exp) |i| {
            pool[sx][i][0] = gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn)) catch break :outer;
            pool[sx][i][1] = gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, dim, ffn)) catch break :outer;
            pool[sx][i][2] = gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q5_0, ffn, dim)) catch break :outer;
            for (pool[sx][i]) |b| rnd.bytes(b);
            for (0..2) |q| {
                const b = pool[sx][i][q];
                for (0..b.len / 2) |k| b[k * 2 + 1] &= 0xFB;
            }
            many[sx][i] = .{
                .gate = .{ .ty = .q4_k, .data = pool[sx][i][0] },
                .up = .{ .ty = .q4_k, .data = pool[sx][i][1] },
                .down = .{ .ty = .q5_0, .data = pool[sx][i][2] },
                .weight = 1.0 / @as(f32, n_exp),
                .ffn = ffn,
            };
        }
        built = sx + 1;
    }
    defer for (0..built) |sx| {
        for (0..n_exp) |i| for (pool[sx][i]) |b| gpa.free(b);
    };
    var rotating: i128 = std.math.maxInt(i64);
    if (built > 1) {
        for (many[0..built]) |*r| _ = moeFfnBlock(normed, r, out); // wrap + resident
        for (0..30) |round| {
            const r = &many[round % built];
            const t0 = now(io);
            _ = moeFfnBlock(normed, r, out);
            const dt = now(io) - t0;
            if (dt < rotating) rotating = dt;
        }
    }

    // Does the *size of the buffer* cost anything, independent of how much of
    // it is touched? The engine reads every expert as an offset into one
    // multi-gigabyte allocation over the store mapping; this benchmark gives
    // each expert its own small one. Same 35 MB of weights either way, placed
    // deep inside a 2 GB anonymous mapping registered as an arena.
    //
    // Anonymous rather than heap because the allocator will not hand out a
    // slab this size on this machine, and the pages are only touched where the
    // weights are copied, so the mapping costs address space and not memory.
    var big_ms: f64 = 0;
    big: {
        const span: usize = 2 << 30;
        const slab = std.posix.mmap(null, span, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch break :big;
        defer std.posix.munmap(slab);
        var off: usize = 1 << 30; // deep in, so it is not the first page either
        var big_refs: [n_exp]ExpertRef = undefined;
        for (0..n_exp) |i| {
            var parts: [3][]u8 = undefined;
            inline for (.{ wg[i], wu[i], wd[i] }, 0..) |src, q| {
                off = std.mem.alignForward(usize, off, 16384);
                @memcpy(slab[off..][0..src.len], src);
                parts[q] = slab[off..][0..src.len];
                off += src.len;
            }
            big_refs[i] = .{
                .gate = .{ .ty = .q4_k, .data = parts[0] },
                .up = .{ .ty = .q4_k, .data = parts[1] },
                .down = .{ .ty = .q5_0, .data = parts[2] },
                .weight = 1.0 / @as(f32, n_exp),
                .ffn = ffn,
            };
        }
        if (!registerArena(slab)) break :big;
        if (materializeArenas() == 0) break :big;
        _ = moeFfnBlock(normed, &big_refs, out);
        var best_b: i128 = std.math.maxInt(i64);
        for (0..30) |_| {
            const t0 = now(io);
            _ = moeFfnBlock(normed, &big_refs, out);
            const dt = now(io) - t0;
            if (dt < best_b) best_b = dt;
        }
        big_ms = @as(f64, @floatFromInt(best_b)) / 1e6;
    }

    var gapped: i128 = std.math.maxInt(i64);
    var sink: f64 = 0;
    for (0..30) |_| {
        const until = now(io) + 1_000_000; // ~1 ms of host work
        while (now(io) < until) sink += 1.0;
        const t0 = now(io);
        _ = moeFfnBlock(normed, &refs, out);
        const dt = now(io) - t0;
        if (dt < gapped) gapped = dt;
    }
    std.testing.expect(sink > 0) catch {};

    var bytes: f64 = 0;
    for (0..n_exp) |i| bytes += @floatFromInt(wg[i].len + wu[i].len + wd[i].len);
    const mb = bytes / (1024 * 1024);
    const t_ms = @as(f64, @floatFromInt(tight)) / 1e6;
    const g_ms = @as(f64, @floatFromInt(gapped)) / 1e6;
    const r_ms = @as(f64, @floatFromInt(rotating)) / 1e6;
    std.debug.print("\n  moe block, {d} experts, {d:.1} MB per call\n    same weights   {d:.3} ms  {d:5.1} GB/s   x26 = {d:.1} ms/token\n    1 ms cpu gap   {d:.3} ms  {d:5.1} GB/s   x26 = {d:.1} ms/token\n    {d} sets, {d:.0} MB working set  {d:.3} ms  {d:5.1} GB/s   x26 = {d:.1} ms/token\n", .{
        n_exp,                                   mb,
        t_ms,                                    bytes / @as(f64, @floatFromInt(tight)),
        t_ms * 26,                               g_ms,
        bytes / @as(f64, @floatFromInt(gapped)), g_ms * 26,
        built,                                   mb * @as(f64, @floatFromInt(built)),
        r_ms,                                    bytes / @as(f64, @floatFromInt(rotating)),
        r_ms * 26,
    });
    if (big_ms > 0) {
        std.debug.print("    inside a 2 GB arena  {d:.3} ms  {d:5.1} GB/s   x26 = {d:.1} ms/token\n", .{
            big_ms, bytes / (big_ms * 1e6), big_ms * 26,
        });
    } else {
        std.debug.print("    inside a 2 GB arena  did not run\n", .{});
    }
}

fn backend_ctx_absent() bool {
    return ctx == null;
}

test "mla attention matches an exact cpu reference" {
    // The identity being tested is the absorption one: scores are
    // q_absorbed . c_t + q_rope . k_rope_t, and the weighted sum stays in
    // compressed space. Getting either half wrong still produces a normalized,
    // plausible output vector -- dropping the rope term entirely just makes
    // attention position-blind, which reads as slightly worse text rather than
    // as a failure.
    const gpa = std.testing.allocator;
    const n_heads = 4;
    // A multiple of 256: kv_b is q4_k and the W_v id dispatch declines a width
    // its kernel cannot walk. 512 on the real model.
    const kvr = 256;
    const rope = 16;
    const seq = 37; // not a multiple of the threadgroup size, on purpose
    const layers = 2;
    const ctx_len = 64;
    const scale: f32 = 0.3;

    parallelBegin(1);
    defer parallelEnd();
    if (backend_ctx_absent()) return error.SkipZigTest;
    const saved_attn = attn_worthwhile;
    attn_worthwhile = true;
    defer attn_worthwhile = saved_attn;

    if (!mlaInit(layers, ctx_len, kvr, rope)) return error.SkipZigTest;
    if (!hasMlaCache()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0x11A);
    const rnd = prng.random();
    const qa = try gpa.alloc(f32, n_heads * kvr);
    defer gpa.free(qa);
    const qr = try gpa.alloc(f32, n_heads * rope);
    defer gpa.free(qr);
    for (qa) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (qr) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const c_rows = try gpa.alloc(f32, seq * kvr);
    defer gpa.free(c_rows);
    const r_rows = try gpa.alloc(f32, seq * rope);
    defer gpa.free(r_rows);
    for (c_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (r_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const li = 1; // not layer zero, so a missing layer stride shows
    for (0..seq) |t| {
        if (!mlaAppend(li, t, c_rows[t * kvr ..][0..kvr], r_rows[t * rope ..][0..rope])) return error.SkipZigTest;
    }

    // The absorption is now inside `mlaAttnHeads`, so the test drives it the
    // way the engine does: hand it q_nope and an identity W_k, which makes
    // q_abs == q_nope and leaves the attention arithmetic the thing under test.
    const nope = 8;
    const vd = 8;
    const stride = nope + vd;
    const wk = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(wk);
    @memset(wk, 0);
    const q_nope = try gpa.alloc(f32, n_heads * nope);
    defer gpa.free(q_nope);
    for (q_nope) |*v| v.* = 0;
    // Row r of head h contributes q_nope[h][r] * wk[h][r][:] to q_abs[h][:].
    // One row carrying the whole absorbed query reproduces `qa` exactly.
    for (0..n_heads) |hh| {
        q_nope[hh * nope] = 1.0;
        @memcpy(wk[(hh * stride) * kvr ..][0..kvr], qa[hh * kvr ..][0..kvr]);
    }
    // Layers arrive in order, so layer 0 goes first even though the test uses
    // layer 1 -- which is the point: a missing layer stride would show.
    const zero = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(zero);
    @memset(zero, 0);
    if (!mlaSetWk(0, zero)) return error.SkipZigTest;
    if (!mlaSetWk(li, wk)) return error.SkipZigTest;

    // W_v: q4_k planes, pinned scales, random quants -- the id dispatch inside
    // mlaAttnHeads applies it, so the oracle below must too.
    const kvb = try gpa.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(.q4_k, kvr, n_heads * stride));
    defer gpa.free(kvb);
    rnd.bytes(kvb);
    {
        var b: usize = 0;
        while (b + 144 <= kvb.len) : (b += 144) {
            std.mem.writeInt(u16, kvb[b..][0..2], 0x2C00, .little);
            std.mem.writeInt(u16, kvb[b + 2 ..][0..2], 0x2800, .little);
        }
    }

    const got = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(got);
    if (!mlaAttnHeads(li, seq - 1, q_nope, qr, .{ .ty = .q4_k, .data = kvb }, got, n_heads, nope, vd, scale)) return error.SkipZigTest;

    // Reference in f64: softmax over 37 terms then a weighted sum is exactly
    // the kind of reduction where an f32 oracle reports its own drift.
    const want_lat = try gpa.alloc(f32, n_heads * kvr);
    defer gpa.free(want_lat);
    const sc = try gpa.alloc(f64, seq);
    defer gpa.free(sc);
    for (0..n_heads) |h| {
        var mx: f64 = -std.math.inf(f64);
        for (0..seq) |t| {
            var s: f64 = 0;
            for (0..kvr) |i| s += @as(f64, qa[h * kvr + i]) * @as(f64, c_rows[t * kvr + i]);
            for (0..rope) |i| s += @as(f64, qr[h * rope + i]) * @as(f64, r_rows[t * rope + i]);
            sc[t] = s * scale;
            mx = @max(mx, sc[t]);
        }
        var sum: f64 = 0;
        for (sc) |*v| {
            v.* = @exp(v.* - mx);
            sum += v.*;
        }
        for (0..kvr) |i| {
            var acc: f64 = 0;
            for (0..seq) |t| acc += sc[t] * @as(f64, c_rows[t * kvr + i]);
            want_lat[h * kvr + i] = @floatCast(acc / sum);
        }
    }

    // Then W_v exactly: head h's value rows start nope rows into its plane.
    const want = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(want);
    const vrow = try gpa.alloc(f32, kvr);
    defer gpa.free(vrow);
    for (0..n_heads) |h| {
        for (0..vd) |i| {
            cpu.dequantRow(.q4_k, vrow, kvb, h * stride + nope + i, kvr);
            var acc: f64 = 0;
            for (vrow, want_lat[h * kvr ..][0..kvr]) |wv, ov| acc += @as(f64, wv) * @as(f64, ov);
            want[h * vd + i] = @floatCast(acc);
        }
    }

    var mass: f32 = 0;
    for (want) |v| mass += @abs(v);
    const tol = (mass / @as(f32, @floatFromInt(want.len))) * 2e-3;
    for (want, got, 0..) |a, b, k| {
        std.testing.expectApproxEqAbs(a, b, tol) catch |e| {
            std.debug.print("mla head {d} dim {d}: cpu {d} vs gpu {d}\n", .{ k / vd, k % vd, a, b });
            return e;
        };
    }
}

test "id-indexed matvec equals the plain kernel, for every expert type" {
    // Same assertion for all three: an id-indexed dispatch over several planes
    // must equal the plain kernel run once per plane, bit for bit -- same
    // arithmetic in both, so a tolerance would only hide a stride bug landing
    // on a neighbouring plane. Q4_K covers gate and up; ffn_down_exps is Q5_0
    // in half of DeepSeek-V2-Lite's layers and Q8_0 in the rest, so all three
    // are needed before a layer can route on the device.
    //
    // Every type's data is allocated up front and freed only at the end,
    // deliberately. The first version allocated and freed per type in a loop,
    // and the q5_0 pass reused the q4_k pass's freed address: `wrapFor` keys
    // its MTLBuffer cache by page base and only re-wraps when the region
    // *grew*, so the q5_0 dispatch read a stale wrapping of freed memory. It
    // presented as slot 1 wrong with slot 0 right -- exactly like a kernel
    // stride bug -- and the kernel was fine.
    const gpa = std.testing.allocator;
    const cols = 2048; // multiple of 256 and of 32: legal for all three types
    const rows = 1408;
    const n_planes = 6;
    const ids = [_]u32{ 3, 0, 5, 1, 4, 2 };
    const types = [_]ggml.Type{ .q4_k, .q5_0, .q8_0 };

    parallelBegin(1);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);
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

    var prng = std.Random.DefaultPrng.init(0x1D5);
    const rnd = prng.random();

    var datas: [types.len][]align(16384) u8 = undefined;
    var n_alloc: usize = 0;
    for (types, 0..) |ty, i| {
        const plane_bytes = ggml.tensorBytes(ty, cols, rows);
        datas[i] = try gpa.alignedAlloc(u8, .fromByteUnits(16384), plane_bytes * n_planes);
        n_alloc = i + 1;
        rnd.bytes(datas[i]);
        // Pin every block scale: a random f16 can be inf, and both sides then
        // produce NaN, which compares unequal to itself and reads as a kernel
        // bug.
        const bs: usize = switch (ty) {
            .q4_k => 144,
            .q5_0 => 22,
            .q8_0 => 34,
            else => unreachable,
        };
        var b: usize = 0;
        while (b + bs <= datas[i].len) : (b += bs) {
            std.mem.writeInt(u16, datas[i][b..][0..2], 0x2C00, .little); // d
            if (ty == .q4_k) std.mem.writeInt(u16, datas[i][b + 2 ..][0..2], 0x2800, .little); // dmin
        }
    }
    defer for (datas[0..n_alloc]) |d| gpa.free(d);

    // Enough distinct vectors for the strided case; the shared case uses the
    // first. Strided is the down projection's shape -- each slot's input is
    // its own expert's SwiGLU output -- and a kernel that ignored x_stride
    // would compute every slot against slot 0's vector: plausible output,
    // since the matrices still differ per slot.
    const xs = try gpa.alloc(f32, ids.len * cols);
    defer gpa.free(xs);
    for (xs) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    for (types, 0..) |ty, i| {
        const data = datas[i];
        const plane_bytes = ggml.tensorBytes(ty, cols, rows);

        for ([_]bool{ false, true }) |strided| {
            // Reference: the plain kernel, once per plane, ids out of order --
            // 3,0,5,1,4,2 -- because a stride bug gives a correct first expert
            // and garbage after it, which in a MoE layer is a plausible output
            // vector: the first expert usually carries the largest gate.
            const want = try gpa.alloc(f32, ids.len * rows);
            defer gpa.free(want);
            for (ids, 0..) |id, slot| {
                const xv = if (strided) xs[slot * cols ..][0..cols] else xs[0..cols];
                matvec(ty, want[slot * rows ..][0..rows], data[id * plane_bytes ..][0..plane_bytes], xv, rows, cols);
            }

            const sel = dmmvIdFor(cx, ty) orelse return error.SkipZigTest;
            const idbuf = cx.dev.alloc(ids.len * @sizeOf(u32)) catch return error.SkipZigTest;
            @memcpy(idbuf.slice(u32)[0..ids.len], &ids);
            const outbuf = cx.dev.alloc(ids.len * rows * @sizeOf(f32)) catch return error.SkipZigTest;
            const w = wrapFor(cx, data) orelse return error.SkipZigTest;
            const n_x: usize = if (strided) ids.len * cols else cols;
            @memcpy(cx.scratch_x.slice(f32)[0..n_x], xs[0..n_x]);

            const dims = IdDims{
                .rows = @intCast(rows),
                .cols = @intCast(cols),
                .n_used = @intCast(ids.len),
                .plane_stride = @intCast(plane_bytes),
                .x_stride = if (strided) @intCast(cols) else 0,
            };
            const group = SIMD_W * SIMDGROUPS_PER_GROUP;
            const threads = ((ids.len * rows + sel.per - 1) / sel.per) * SIMD_W;
            const grid = ((threads + group - 1) / group) * group;
            const cb = cx.dev.commandBuffer();
            cb.dispatch(
                sel.pipe,
                &.{ w.buf, cx.scratch_x, outbuf, idbuf },
                &.{ w.off, 0, 0, 0 },
                std.mem.asBytes(&dims),
                grid,
                group,
            );
            cb.commitAndWait();

            const got = outbuf.slice(f32)[0 .. ids.len * rows];
            for (want, got, 0..) |a, c, k| {
                std.testing.expectEqual(a, c) catch |e| {
                    std.debug.print("{t} strided={} id matvec slot {d} (expert {d}) row {d}: plain {d} vs id {d}\n", .{ ty, strided, k / rows, ids[k / rows], k % rows, a, c });
                    return e;
                };
            }
        }
    }
}

test "device routing agrees with moe.route, bias and all" {
    // Must match the host router exactly, because the two coexist: the
    // distributed path stays host-routed for now, so a divergence would make
    // the same model route differently depending on how its weights arrived.
    //
    // Bias is the case worth the test. It shifts the *selection* scores only,
    // and the emitted gate comes from the unbiased probability -- reverse that
    // and the model still produces text, just worse. So the bias here is large
    // enough to change which experts win.
    const gpa = std.testing.allocator;
    const moe = @import("../gguf/moe.zig");
    const n_expert = 64;
    const n_used = 6;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;

    var prng = std.Random.DefaultPrng.init(0x120117);
    const rnd = prng.random();
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    const bias = try gpa.alloc(f32, n_expert);
    defer gpa.free(bias);
    for (logits) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;
    for (bias) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    inline for (.{ true, false }) |sig| {
        inline for (.{ true, false }) |with_bias| {
            inline for (.{ true, false }) |norm| {
                const cfg = moe.RouteCfg{
                    .n_expert = n_expert,
                    .n_used = n_used,
                    .gating = if (sig) .sigmoid else .softmax,
                    .weights_norm = norm,
                    .weights_scale = 2.5,
                };
                var want: [n_used]moe.Selected = undefined;
                moe.route(cfg, logits, if (with_bias) bias else null, &want);

                var ids: [n_used]u32 = undefined;
                var gates: [n_used]f32 = undefined;
                if (!moeRoute(logits, if (with_bias) bias else null, &ids, &gates, sig, norm, 2.5)) return error.SkipZigTest;

                for (want, 0..) |w, k| {
                    std.testing.expectEqual(@as(u32, @intCast(w.expert)), ids[k]) catch |e| {
                        std.debug.print("route sigmoid={} bias={} norm={}: slot {d} host expert {d} vs device {d}\n", .{ sig, with_bias, norm, k, w.expert, ids[k] });
                        return e;
                    };
                    // exp() differs in the last bits between the two, so the
                    // gates are compared against the mass they carry rather
                    // than exactly; the *choice* above is exact.
                    //
                    // Not covered: the renormalization divisor's clamp to the
                    // smallest normal f16. Random logits never drive the gate
                    // sum near zero, so removing the clamp still passes this.
                    // Covering it needs logits contrived to make every
                    // selected gate underflow.
                    try std.testing.expectApproxEqAbs(w.gate, gates[k], @abs(w.gate) * 1e-5 + 1e-7);
                }
            }
        }
    }
}

test "fused routed layer equals host routing plus the verified block" {
    // Differential between two paths whose parts are each verified: the fused
    // buffer (route + id kernels + device-gated reduce + shared expert) against
    // host moe.route feeding the host-pointer moeFfnBlock. Same kernels for
    // the arithmetic, routing verified to pick identical experts, so the two
    // must agree closely -- the tolerance covers only the gates' exp() drift.
    const gpa = std.testing.allocator;
    const moe = @import("../gguf/moe.zig");
    const dim = 2048;
    const ffn = 1408;
    const shexp_ffn = 2816;
    const n_expert = 8;
    const n_used = 3;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;

    var prng = std.Random.DefaultPrng.init(0xF05E);
    const rnd = prng.random();
    const mk = struct {
        fn f(g: std.mem.Allocator, r: std.Random, ty: ggml.Type, rows: usize, cols: usize, planes: usize) ![]align(16384) u8 {
            const d = try g.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(ty, cols, rows) * planes);
            r.bytes(d);
            const bs: usize = if (ty == .q4_k) 144 else 22;
            var b: usize = 0;
            while (b + bs <= d.len) : (b += bs) {
                std.mem.writeInt(u16, d[b..][0..2], 0x2C00, .little);
                if (ty == .q4_k) std.mem.writeInt(u16, d[b + 2 ..][0..2], 0x2800, .little);
            }
            return d;
        }
    }.f;

    // Held for the whole test: wrapFor's contract.
    const gw = try mk(gpa, rnd, .q4_k, ffn, dim, n_expert);
    defer gpa.free(gw);
    const uw = try mk(gpa, rnd, .q4_k, ffn, dim, n_expert);
    defer gpa.free(uw);
    const dw = try mk(gpa, rnd, .q5_0, dim, ffn, n_expert);
    defer gpa.free(dw);
    const sg = try mk(gpa, rnd, .q4_k, shexp_ffn, dim, 1);
    defer gpa.free(sg);
    const su = try mk(gpa, rnd, .q4_k, shexp_ffn, dim, 1);
    defer gpa.free(su);
    const sd = try mk(gpa, rnd, .q5_0, dim, shexp_ffn, 1);
    defer gpa.free(sd);

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    for (normed) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    for (logits) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;

    const rcfg = RoutedCfg{
        .n_expert = n_expert,
        .n_used = n_used,
        .gating_sigmoid = true,
        .weights_norm = true,
        .weights_scale = 1.0,
    };
    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);
    if (!moeFfnBlockRouted(
        normed,
        logits,
        null,
        .{ .ty = .q4_k, .data = gw },
        .{ .ty = .q4_k, .data = uw },
        .{ .ty = .q5_0, .data = dw },
        .{ .{ .ty = .q4_k, .data = sg }, .{ .ty = .q4_k, .data = su }, .{ .ty = .q5_0, .data = sd } },
        ffn,
        shexp_ffn,
        rcfg,
        got,
    )) return error.SkipZigTest;

    // Reference: host routing into the verified host-pointer block, shared
    // expert as its extra column, exactly as the engine builds it today.
    var sel: [n_used]moe.Selected = undefined;
    moe.route(.{
        .n_expert = n_expert,
        .n_used = n_used,
        .gating = .sigmoid,
        .weights_norm = true,
        .weights_scale = 1.0,
    }, logits, null, &sel);
    const plane_g = ggml.tensorBytes(.q4_k, dim, ffn);
    const plane_u = plane_g;
    const plane_d = ggml.tensorBytes(.q5_0, ffn, dim);
    var refs: [n_used + 1]ExpertRef = undefined;
    for (sel, 0..) |sv, k| refs[k] = .{
        .gate = .{ .ty = .q4_k, .data = gw[sv.expert * plane_g ..][0..plane_g] },
        .up = .{ .ty = .q4_k, .data = uw[sv.expert * plane_u ..][0..plane_u] },
        .down = .{ .ty = .q5_0, .data = dw[sv.expert * plane_d ..][0..plane_d] },
        .weight = sv.gate,
        .ffn = ffn,
    };
    refs[n_used] = .{
        .gate = .{ .ty = .q4_k, .data = sg },
        .up = .{ .ty = .q4_k, .data = su },
        .down = .{ .ty = .q5_0, .data = sd },
        .weight = 1.0,
        .ffn = shexp_ffn,
    };
    const want = try gpa.alloc(f32, dim);
    defer gpa.free(want);
    if (!moeFfnBlock(normed, &refs, want)) return error.SkipZigTest;

    var mass: f32 = 0;
    for (want) |v| mass += @abs(v);
    const tol = (mass / @as(f32, @floatFromInt(dim))) * 1e-4;
    for (want, got, 0..) |a, c, k| {
        std.testing.expectApproxEqAbs(a, c, tol) catch |e| {
            std.debug.print("fused routed dim {d}: host-routed {d} vs fused {d}\n", .{ k, a, c });
            return e;
        };
    }
}

test "layer tail equals its constituent buffers plus host glue" {
    // mlaLayerTail is assembly: absorb+attention+W_v (verified against an f64
    // oracle), projection/residual/norm/router (kernels each verified), and
    // the routed MoE block (verified differentially). This test pins the
    // assembly itself -- the offsets and slot plan -- by running the same
    // inputs through the constituent calls with host glue between them. A
    // wrong slot or offset here produces a plausible residual stream, which
    // 32 tokens of identical text can miss but 2048 compared floats cannot.
    const gpa = std.testing.allocator;
    const dim = 2048;
    const n_heads = 4;
    const nope = 64;
    const vd = 64; // n_heads * vd = 256, a legal q4_k width for the projection
    const kvr = 256;
    const rope = 16;
    const seq = 9;
    const ffn = 1408;
    const n_expert = 8;
    const n_used = 3;

    parallelBegin(1);
    defer parallelEnd();
    if (ctx == null) return error.SkipZigTest;
    const saved_use = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved_use;

    var prng = std.Random.DefaultPrng.init(0x7A11);
    const rnd = prng.random();
    const mkq = struct {
        fn f(g: std.mem.Allocator, r: std.Random, ty: ggml.Type, rows: usize, cols: usize, planes: usize) ![]align(16384) u8 {
            const d = try g.alignedAlloc(u8, .fromByteUnits(16384), ggml.tensorBytes(ty, cols, rows) * planes);
            r.bytes(d);
            const bs: usize = if (ty == .q4_k) 144 else 22;
            var b: usize = 0;
            while (b + bs <= d.len) : (b += bs) {
                std.mem.writeInt(u16, d[b..][0..2], 0x2C00, .little);
                if (ty == .q4_k) std.mem.writeInt(u16, d[b + 2 ..][0..2], 0x2800, .little);
            }
            return d;
        }
    }.f;

    // MLA cache and W_k, on layer 1 so a layer stride shows.
    if (!mlaInit(2, 64, kvr, rope)) return error.SkipZigTest;
    const stride = nope + vd;
    const wk = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(wk);
    for (wk) |*v| v.* = (rnd.float(f32) - 0.5) * 0.1;
    // Uploads are once per context and another test may have gone first. That
    // is fine for a differential: both sides below read the *same* device W_k,
    // whichever upload won. All that matters is that layer 1 exists.
    const zero = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(zero);
    @memset(zero, 0);
    _ = mlaSetWk(0, zero);
    _ = mlaSetWk(1, wk);
    if (!hasMlaCache()) return error.SkipZigTest;
    const li = 1;
    const c_rows = try gpa.alloc(f32, seq * kvr);
    defer gpa.free(c_rows);
    const r_rows = try gpa.alloc(f32, seq * rope);
    defer gpa.free(r_rows);
    for (c_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (r_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (0..seq) |t| {
        if (!mlaAppend(li, t, c_rows[t * kvr ..][0..kvr], r_rows[t * rope ..][0..rope])) return error.SkipZigTest;
    }

    const kvb = try mkq(gpa, rnd, .q4_k, n_heads * stride, kvr, 1);
    defer gpa.free(kvb);
    const proj = try mkq(gpa, rnd, .q4_k, dim, n_heads * vd, 1);
    defer gpa.free(proj);
    const gw = try mkq(gpa, rnd, .q4_k, ffn, dim, n_expert);
    defer gpa.free(gw);
    const uw = try mkq(gpa, rnd, .q4_k, ffn, dim, n_expert);
    defer gpa.free(uw);
    const dw = try mkq(gpa, rnd, .q5_0, dim, ffn, n_expert);
    defer gpa.free(dw);
    const router = try gpa.alloc(f32, n_expert * dim);
    defer gpa.free(router);
    for (router) |*v| v.* = (rnd.float(f32) - 0.5) * 0.05;
    const norm_w = try gpa.alloc(f32, dim);
    defer gpa.free(norm_w);
    for (norm_w) |*v| v.* = 1.0;

    const q_nope = try gpa.alloc(f32, n_heads * nope);
    defer gpa.free(q_nope);
    const q_rope = try gpa.alloc(f32, n_heads * rope);
    defer gpa.free(q_rope);
    for (q_nope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (q_rope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const x0 = try gpa.alloc(f32, dim);
    defer gpa.free(x0);
    for (x0) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const rcfg = RoutedCfg{ .n_expert = n_expert, .n_used = n_used, .gating_sigmoid = true, .weights_norm = true, .weights_scale = 1.0 };
    const eps: f32 = 1e-6;

    // ---- tail ----
    const x_tail = try gpa.dupe(f32, x0);
    defer gpa.free(x_tail);
    if (!mlaLayerTail(
        li,
        seq - 1,
        x_tail,
        q_nope,
        q_rope,
        .{ .ty = .q4_k, .data = kvb },
        .{ .ty = .q4_k, .data = proj },
        norm_w,
        eps,
        .{ .ty = .f32, .data = std.mem.sliceAsBytes(router) },
        null,
        .{ .ty = .q4_k, .data = gw },
        .{ .ty = .q4_k, .data = uw },
        .{ .ty = .q5_0, .data = dw },
        null,
        ffn,
        0,
        rcfg,
        n_heads,
        nope,
        vd,
        0.25,
    )) return error.SkipZigTest;

    // ---- constituents with host glue ----
    const head_out = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(head_out);
    if (!mlaAttnHeads(li, seq - 1, q_nope, q_rope, .{ .ty = .q4_k, .data = kvb }, head_out, n_heads, nope, vd, 0.25)) return error.SkipZigTest;
    const x_ref = try gpa.dupe(f32, x0);
    defer gpa.free(x_ref);
    const pvec = try gpa.alloc(f32, dim);
    defer gpa.free(pvec);
    matvec(.q4_k, pvec, proj, head_out, dim, n_heads * vd);
    for (x_ref, pvec) |*a, v| a.* += v;
    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    cpu.rmsnorm(normed, x_ref, norm_w, eps);
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    for (0..n_expert) |e2| {
        var acc: f32 = 0;
        for (router[e2 * dim ..][0..dim], normed) |wv, nv| acc += wv * nv;
        logits[e2] = acc;
    }
    const ffn_out = try gpa.alloc(f32, dim);
    defer gpa.free(ffn_out);
    if (!moeFfnBlockRouted(normed, logits, null, .{ .ty = .q4_k, .data = gw }, .{ .ty = .q4_k, .data = uw }, .{ .ty = .q5_0, .data = dw }, null, ffn, 0, rcfg, ffn_out)) return error.SkipZigTest;
    for (x_ref, ffn_out) |*a, v| a.* += v;

    var mass: f32 = 0;
    for (x_ref) |v| mass += @abs(v);
    const tol = (mass / @as(f32, @floatFromInt(dim))) * 1e-3;
    for (x_ref, x_tail, 0..) |a, b, k| {
        std.testing.expectApproxEqAbs(a, b, tol) catch |e| {
            std.debug.print("layer tail dim {d}: constituents {d} vs tail {d}\n", .{ k, a, b });
            return e;
        };
    }
}

test "mla rope kernel matches the host ropeApply, plain and yarn, strided" {
    // Rotates in place where q lives -- stride kd, offset nope -- so a wrong
    // stride rotates the wrong halves of the wrong heads, which is exactly as
    // silent as every other rope bug.
    const gpa = std.testing.allocator;
    const deepseek = @import("../gguf/deepseek.zig");
    const n_vec = 4;
    const rope = 16;
    const stride = 48; // like kd: rope section sits `offset` in
    const offset = 24;
    const pos = 37;

    parallelBegin(1);
    defer parallelEnd();
    const cx = &(ctx orelse return error.SkipZigTest);
    var prng = std.Random.DefaultPrng.init(0x1207E);
    const rnd = prng.random();
    const host = try gpa.alloc(f32, n_vec * stride);
    defer gpa.free(host);
    for (host) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    inline for (.{ 1.0, 40.0 }) |yf| {
        const dev_buf = cx.dev.alloc(host.len * @sizeOf(f32)) catch return error.SkipZigTest;
        @memcpy(dev_buf.slice(f32)[0..host.len], host);
        const want = try gpa.dupe(f32, host);
        defer gpa.free(want);
        // Host reference through the engine's own function per vector.
        var cfg: deepseek.Config = undefined;
        cfg.rope_base = 10000.0;
        cfg.yarn_factor = yf;
        cfg.yarn_orig_ctx = 4096.0;
        cfg.yarn_log_mul = 0;
        for (0..n_vec) |k| deepseek.ropeApplyForTest(cfg, want[k * stride + offset ..][0..rope], pos);

        const d = MlaRopeDims{ .n_vec = n_vec, .rope = rope, .stride = stride, .offset = offset, .pos = pos, .base = 10000.0, .yarn_factor = yf, .yarn_orig_ctx = 4096.0 };
        const cb = cx.dev.commandBuffer();
        cb.dispatch(cx.mla_rope_p, &.{dev_buf}, &.{0}, std.mem.asBytes(&d), n_vec * rope / 2, 32);
        cb.commitAndWait();

        const got = dev_buf.slice(f32)[0..host.len];
        for (want, got, 0..) |a, b, k| {
            std.testing.expectApproxEqAbs(a, b, 1e-5) catch |e| {
                std.debug.print("rope yf={d} idx {d}: host {d} vs gpu {d}\n", .{ yf, k, a, b });
                return e;
            };
        }
    }
}
