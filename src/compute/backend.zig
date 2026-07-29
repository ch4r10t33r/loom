//! Compute backend selection (issue #10).
//!
//! Every operation the forward pass performs goes through this file, so a
//! backend is a single import away from replacing all of them. The set is
//! small because the SIMD, threading and batching work concentrated it:
//! two matrix entry points, two attention primitives, and the elementwise ops.
//!
//! Selection is at **comptime**, not runtime, following the structure ZINC
//! used for its Apple Silicon bring-up: the inactive backend is not compiled
//! into the binary at all. That keeps the seam free of dispatch cost and stops
//! a GPU toolchain's headers and link requirements leaking into builds that do
//! not want them.
//!
//!     zig build                      # CPU
//!     zig build -Dgpu=metal          # macOS
//!     zig build -Dgpu=vulkan         # Linux, Windows
//!
//! Why route the elementwise ops through here too, when only the matmuls are
//! expensive: on a GPU the activations have to live in device memory, and an
//! RMSNorm left on the host would force a round trip between every pair of
//! matmuls. The op that costs nothing is exactly the one that must not be
//! stranded on the wrong side of the bus.

const std = @import("std");
const builtin = @import("builtin");
const options = @import("build_info");

pub const Kind = enum { cpu, metal, vulkan };

/// Resolved backend. `build_info.gpu` comes from `-Dgpu=`; "none" means CPU.
pub const kind: Kind = blk: {
    const want = options.gpu;
    if (std.mem.eql(u8, want, "none")) break :blk .cpu;
    if (std.mem.eql(u8, want, "metal")) {
        if (builtin.os.tag != .macos) @compileError("-Dgpu=metal is macOS only");
        break :blk .metal;
    }
    if (std.mem.eql(u8, want, "vulkan")) {
        if (builtin.os.tag == .macos) @compileError("-Dgpu=vulkan is not supported on macOS; use -Dgpu=metal");
        break :blk .vulkan;
    }
    @compileError("unknown -Dgpu value: " ++ want ++ " (want none, metal or vulkan)");
};

const impl = switch (kind) {
    .cpu => @import("cpu.zig"),
    .metal => @import("../metal/backend.zig"),
    .vulkan => @compileError("the Vulkan backend is not implemented yet (issue #13); build without -Dgpu"),
};

pub const name = @tagName(kind);

// ---- matrix ops --------------------------------------------------------------

/// out[r] = dot(W[r], x). The single hottest call in the engine.
pub const matvec = impl.matvec;

/// Batched form for prefill: out[k*rows + r] = dot(W[r], xs[k*cols..]).
/// Must be exactly equal to `n` calls to matvec.
pub const matmul = impl.matmul;

/// Dequantize one row (the embedding lookup, which indexes rather than dots).
pub const dequantRow = impl.dequantRow;

// ---- attention primitives ----------------------------------------------------

pub const dotF32 = impl.dotF32;
pub const axpy = impl.axpy;

// ---- fused blocks ------------------------------------------------------------

/// Weight tensor as the fused entry points take it.
pub const WeightRef = impl.WeightRef;

/// A whole FFN block submitted as one unit. Returns false when the backend
/// declines the shape or type, in which case the caller runs the pieces.
///
/// This is on the seam because the unit of work a GPU cares about is not the
/// kernel. Measured on an M5, a command buffer costs ~262 us fixed no matter
/// how many dispatches it holds while a matvec kernel is ~18 us, so an engine
/// that only ever hands the backend individual matvecs forces one command
/// buffer per matvec and can never be fast, whatever the kernels do.
pub const ffnBlock = impl.ffnBlock;

/// One expert of a MoE layer, as `moeFfnBlock` takes them.
pub const ExpertRef = impl.ExpertRef;

/// A whole MoE layer -- every routed expert's FFN and the weighted sum of
/// their outputs -- submitted as one unit. False when the backend declines,
/// in which case the caller runs the experts itself.
///
/// Separate from `ffnBlock` because the accumulation between experts has to
/// stay on the backend's side: a host-side weighted add between two expert
/// FFNs ends the command buffer and puts each expert back on its own
/// submission, which is the whole cost this exists to avoid.
pub const moeFfnBlock = impl.moeFfnBlock;

/// Register a whole weight mapping as one device allocation.
///
/// A backend that wraps host pointers on demand keys them by whatever slice it
/// is handed, which for a sharded store is one allocation per expert -- and the
/// model is then never resident, only faulted in a piece at a time. Handing the
/// mapping over once lets every tensor in it be addressed by offset. False when
/// the backend has no such notion (the CPU) or the region is too large for it.
pub const registerArena = impl.registerArena;

/// Make every registered mapping device-resident now that the device is up,
/// returning the bytes resident. Call once the model is loaded; registration
/// happens earlier, while the store is being opened.
pub const materializeArenas = impl.materializeArenas;

/// Why the last `materializeArenas` came back empty, when it did.
pub const arenaError = &impl.arena_error;

/// Allocate a device-resident KV cache. False means the backend declines and
/// the engine keeps its own.
pub const attnInit = impl.attnInit;

/// Mirror one KV cache row into the backend's device copy. Must be called
/// wherever the engine writes its own cache, prefill and decode alike.
pub const kvAppend = impl.kvAppend;
pub const disableAttn = impl.disableAttn;
/// Declare a reader for the device KV cache so appends start mirroring.
pub const enableKvMirror = impl.enableKvMirror;
/// Free the device KV cache when nothing will read it.
pub const releaseKvCache = impl.releaseKvCache;
/// Whether a device KV cache exists; without one the recorded path cannot run.
pub const hasKvCache = impl.hasKvCache;

// ---- frames ------------------------------------------------------------------
//
// The unit a GPU backend cares about is the submission, not the kernel. On an
// M5 a command buffer costs ~262 us against ~18 us for a matvec, and ZINC --
// whose kernel is at parity with loom's -- drops from 53 to 11.8 tok/s when its
// timing probe forces a commit between dispatches. So the seam has to let a
// caller record many operations and submit once, which one-shot entry points
// cannot express however good the kernels behind them are.

/// Open a recording frame. False means the backend has nothing to batch.
pub const beginFrame = impl.beginFrame;
/// Submit and wait. Results are only valid after this returns.
pub const endFrame = impl.endFrame;
pub const frameOpen = impl.frameOpen;

/// One whole GQA layer recorded into the open frame, residual included. The
/// residual stream stays in device memory for the entire token.
pub const LayerSpec = impl.LayerSpec;
pub const layerBlock = impl.layerBlock;
pub const frameLoadX = impl.frameLoadX;
pub const frameStoreX = impl.frameStoreX;

/// One layer of grouped-query attention against that cache, in one submission.
/// False means the engine runs its own path.
pub const attnHeads = impl.attnHeads;

/// One shape the engine will issue, for calibration.
pub const Shape = impl.Shape;

/// Measure this backend against the CPU on the loaded model's own shapes and
/// decide, per operation, which to use. Called once at model load.
///
/// The alternative -- a compiled-in row threshold -- cannot be right: the
/// crossover depends on the ratio between a machine's GPU and its CPU cores,
/// so a constant measured on one laptop is wrong on the next, and it is
/// per-shape besides. This is the same argument the fetch path already makes
/// for probing disk against network rather than assuming an order.
/// Whether to act on the calibration verdict; see the backend for why this
/// defaults off.
pub const useGpuOps = &impl.use_gpu_ops;
pub const calibrate = impl.calibrate;
pub const calibrateAttn = impl.calibrateAttn;
pub const lastVerdict = impl.lastVerdict;

// ---- elementwise -------------------------------------------------------------

pub const rmsnorm = impl.rmsnorm;
pub const softmax = impl.softmax;
pub const swiglu = impl.swiglu;
pub const add = impl.add;
pub const sigmoid = impl.sigmoid;

// ---- kernel worker pool ------------------------------------------------------
// Scoped to a generation; see cpu.zig. A GPU backend implements these as no-ops
// or as its own queue setup.

pub const parallelBegin = impl.parallelBegin;
pub const parallelEnd = impl.parallelEnd;

/// Largest prefill batch the backend's kernels accept.
pub const MAX_BATCH = impl.MAX_BATCH;

test "the selected backend is the one the build asked for" {
    // A build that silently fell back to CPU when a GPU was requested would be
    // reported as a mysterious performance regression, never as a bug.
    if (std.mem.eql(u8, options.gpu, "none")) {
        try std.testing.expectEqual(Kind.cpu, kind);
        try std.testing.expectEqualStrings("cpu", name);
    }
}
