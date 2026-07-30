// C surface over Metal, so the Zig side never sees Objective-C.
//
// One file, deliberately: ARC, Objective-C object lifetime, and the
// NSError/autorelease conventions all stay behind this boundary. Everything
// above it is plain Zig against opaque pointers.
#ifndef LOOM_METAL_SHIM_H
#define LOOM_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef void *loom_mtl_device;
typedef void *loom_mtl_buffer;
typedef void *loom_mtl_pipeline;
typedef void *loom_mtl_queue;
typedef void *loom_mtl_cmdbuf;
typedef void *loom_mtl_encoder;

// ---- device ----
loom_mtl_device loom_mtl_device_create(void);
void loom_mtl_device_release(loom_mtl_device d);
const char *loom_mtl_device_name(loom_mtl_device d);
int loom_mtl_device_has_unified_memory(loom_mtl_device d);
size_t loom_mtl_device_max_threads_per_group(loom_mtl_device d);

// ---- buffers ----
// Wrap host memory with no copy. `ptr` and `len` must be page aligned; the
// caller keeps ownership of the pages and must outlive the buffer. This is the
// whole point of the Apple path: an mmap'd GGUF region becomes a GPU buffer
// without being staged or uploaded.
loom_mtl_buffer loom_mtl_buffer_wrap(loom_mtl_device d, void *ptr, size_t len);

/* Ask the OS to make `b`'s pages resident and keep them so, and attach the set
 * to `q` so every command buffer it vends inherits the residency. Returns 0
 * when the OS is older than residency sets, in which case the pages are still
 * addressable -- just evictable. */
/* Largest single allocation the device will make. Around half of physical RAM
 * on Apple silicon, so a multi-gigabyte model has to be split across several. */
size_t loom_mtl_max_buffer(loom_mtl_device d);

int loom_mtl_resident(loom_mtl_device d, loom_mtl_queue q, loom_mtl_buffer b);
// Ordinary shared-storage allocation, for activations.
loom_mtl_buffer loom_mtl_buffer_alloc(loom_mtl_device d, size_t len);
void *loom_mtl_buffer_contents(loom_mtl_buffer b);
void loom_mtl_buffer_release(loom_mtl_buffer b);

// ---- pipelines ----
// Compile MSL source and get a compute pipeline for `fn_name`. On failure
// returns NULL and writes a message into `err` (up to `err_len`).
loom_mtl_pipeline loom_mtl_pipeline_create(loom_mtl_device d, const char *src,
                                           const char *fn_name, char *err,
                                           size_t err_len);
void loom_mtl_pipeline_release(loom_mtl_pipeline p);
size_t loom_mtl_pipeline_max_threads(loom_mtl_pipeline p);

// ---- command submission ----
loom_mtl_queue loom_mtl_queue_create(loom_mtl_device d);
void loom_mtl_queue_release(loom_mtl_queue q);
loom_mtl_cmdbuf loom_mtl_cmdbuf_create(loom_mtl_queue q);
// Encode one dispatch. `buffers`/`offsets` are bound to indices 0..n_buffers-1.
void loom_mtl_encode(loom_mtl_cmdbuf cb, loom_mtl_pipeline p,
                     const loom_mtl_buffer *buffers, const size_t *offsets,
                     size_t n_buffers, const void *constants, size_t constants_len,
                     size_t grid_x, size_t group_x);
void loom_mtl_cmdbuf_commit_wait(loom_mtl_cmdbuf cb);

// One encoder, many dispatches.
//
// `loom_mtl_encode` above opens a fresh compute encoder per dispatch, and an
// encoder boundary is a full pipeline drain: a six-dispatch block pays six of
// them. These let a caller open one encoder, issue several dispatches into it,
// and place a barrier only where one dispatch genuinely reads what the
// previous one wrote.
loom_mtl_encoder loom_mtl_encoder_begin(loom_mtl_cmdbuf cb);
void loom_mtl_encoder_dispatch(loom_mtl_encoder e, loom_mtl_pipeline p,
                               const loom_mtl_buffer *buffers, const size_t *offsets,
                               size_t n_buffers, const void *constants,
                               size_t constants_len, size_t grid_x, size_t group_x);
void loom_mtl_encoder_barrier(loom_mtl_encoder e);
void loom_mtl_encoder_end(loom_mtl_encoder e);

#endif
