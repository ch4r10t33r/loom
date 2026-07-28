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
    .metal => @compileError("the Metal backend is not implemented yet (issue #12); build without -Dgpu"),
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
