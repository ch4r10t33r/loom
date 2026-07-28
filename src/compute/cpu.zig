//! The CPU backend: SIMD kernels over mmap'd weights, row-parallel across a
//! worker pool.
//!
//! This is a binding layer, not an implementation — the kernels live in
//! `gguf/ggml.zig` (quantized matrix ops) and `core/tensor.zig` (elementwise).
//! Collecting them here is what lets `backend.zig` swap the whole set for a
//! GPU without every engine knowing.
//!
//! It also stays the correctness oracle once GPU backends exist: every GPU
//! kernel is checked against its CPU counterpart, the same way the codebook
//! decoders are checked against llama.cpp's.

const ggml = @import("../gguf/ggml.zig");
const tensor = @import("../core/tensor.zig");

pub const matvec = ggml.matvec;
pub const matmul = ggml.matmul;
pub const dequantRow = ggml.dequantRow;
pub const dotF32 = ggml.dotF32;
pub const axpy = ggml.axpy;
pub const MAX_BATCH = ggml.MAX_BATCH;
pub const parallelBegin = ggml.parallelBegin;
pub const parallelEnd = ggml.parallelEnd;

pub const rmsnorm = tensor.rmsnorm;
pub const softmax = tensor.softmax;
pub const swiglu = tensor.swiglu;
pub const add = tensor.add;
pub const sigmoid = tensor.sigmoid;
