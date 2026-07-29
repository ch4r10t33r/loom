// The only Objective-C in the project. See shim.h for why.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "shim.h"

loom_mtl_device loom_mtl_device_create(void) {
    return (__bridge_retained void *)MTLCreateSystemDefaultDevice();
}
void loom_mtl_device_release(loom_mtl_device d) {
    if (d) CFRelease(d);
}
const char *loom_mtl_device_name(loom_mtl_device d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    return [dev.name UTF8String];
}
int loom_mtl_device_has_unified_memory(loom_mtl_device d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    return dev.hasUnifiedMemory ? 1 : 0;
}
size_t loom_mtl_device_max_threads_per_group(loom_mtl_device d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    return (size_t)dev.maxThreadsPerThreadgroup.width;
}

loom_mtl_buffer loom_mtl_buffer_wrap(loom_mtl_device d, void *ptr, size_t len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    // No deallocator: the caller owns the mapping. Returning nil here means the
    // pointer or length was not page aligned, which the Zig side checks first.
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                               length:len
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    return (__bridge_retained void *)buf;
}
size_t loom_mtl_max_buffer(loom_mtl_device d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    return (size_t)dev.maxBufferLength;
}
int loom_mtl_resident(loom_mtl_device d, loom_mtl_queue q, loom_mtl_buffer b) {
    if (@available(macOS 15.0, *)) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)q;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b;
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.initialCapacity = 1;
        NSError *err = nil;
        id<MTLResidencySet> set = [dev newResidencySetWithDescriptor:desc error:&err];
        if (set == nil) return 0;
        [set addAllocation:buf];
        [set commit];
        // requestResidency is the part that actually wires the pages; commit
        // alone only publishes the membership.
        [set requestResidency];
        [queue addResidencySet:set];
        return 1;
    }
    return 0;
}
loom_mtl_buffer loom_mtl_buffer_alloc(loom_mtl_device d, size_t len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    id<MTLBuffer> buf = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    return (__bridge_retained void *)buf;
}
void *loom_mtl_buffer_contents(loom_mtl_buffer b) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b;
    return buf.contents;
}
void loom_mtl_buffer_release(loom_mtl_buffer b) {
    if (b) CFRelease(b);
}

loom_mtl_pipeline loom_mtl_pipeline_create(loom_mtl_device d, const char *src,
                                           const char *fn_name, char *err,
                                           size_t err_len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    NSError *nserr = nil;
    NSString *source = [NSString stringWithUTF8String:src];
    id<MTLLibrary> lib = [dev newLibraryWithSource:source options:nil error:&nserr];
    if (!lib) {
        if (err && err_len) snprintf(err, err_len, "%s", [[nserr localizedDescription] UTF8String]);
        return NULL;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!fn) {
        if (err && err_len) snprintf(err, err_len, "no such kernel: %s", fn_name);
        return NULL;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&nserr];
    if (!pso) {
        if (err && err_len) snprintf(err, err_len, "%s", [[nserr localizedDescription] UTF8String]);
        return NULL;
    }
    return (__bridge_retained void *)pso;
}
void loom_mtl_pipeline_release(loom_mtl_pipeline p) {
    if (p) CFRelease(p);
}
size_t loom_mtl_pipeline_max_threads(loom_mtl_pipeline p) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)p;
    return (size_t)pso.maxTotalThreadsPerThreadgroup;
}

loom_mtl_queue loom_mtl_queue_create(loom_mtl_device d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
    return (__bridge_retained void *)[dev newCommandQueue];
}
void loom_mtl_queue_release(loom_mtl_queue q) {
    if (q) CFRelease(q);
}
loom_mtl_cmdbuf loom_mtl_cmdbuf_create(loom_mtl_queue q) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)q;
    // Retained here and released on commit_wait, so the Zig side holds a plain
    // pointer across the encode calls.
    return (__bridge_retained void *)[queue commandBuffer];
}
void loom_mtl_encode(loom_mtl_cmdbuf cb, loom_mtl_pipeline p,
                     const loom_mtl_buffer *buffers, const size_t *offsets,
                     size_t n_buffers, const void *constants, size_t constants_len,
                     size_t grid_x, size_t group_x) {
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)cb;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)p;
    id<MTLComputeCommandEncoder> enc = [buf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (size_t i = 0; i < n_buffers; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)buffers[i] offset:offsets[i] atIndex:i];
    }
    if (constants && constants_len) {
        [enc setBytes:constants length:constants_len atIndex:n_buffers];
    }
    [enc dispatchThreads:MTLSizeMake(grid_x, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(group_x, 1, 1)];
    [enc endEncoding];
}
loom_mtl_encoder loom_mtl_encoder_begin(loom_mtl_cmdbuf cb) {
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)cb;
    // Concurrent dispatch type: without it Metal serializes every dispatch in
    // the encoder even when they are independent, which is most of what this
    // change is trying to avoid. Ordering where it matters comes from the
    // explicit barriers below.
    id<MTLComputeCommandEncoder> enc =
        [buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    return (__bridge_retained void *)enc;
}

void loom_mtl_encoder_dispatch(loom_mtl_encoder e, loom_mtl_pipeline p,
                               const loom_mtl_buffer *buffers, const size_t *offsets,
                               size_t n_buffers, const void *constants,
                               size_t constants_len, size_t grid_x, size_t group_x) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)e;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)p;
    [enc setComputePipelineState:pso];
    for (size_t i = 0; i < n_buffers; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)buffers[i] offset:offsets[i] atIndex:i];
    }
    if (constants && constants_len) {
        [enc setBytes:constants length:constants_len atIndex:n_buffers];
    }
    [enc dispatchThreads:MTLSizeMake(grid_x, 1, 1)
  threadsPerThreadgroup:MTLSizeMake(group_x, 1, 1)];
}

void loom_mtl_encoder_barrier(loom_mtl_encoder e) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)e;
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void loom_mtl_encoder_end(loom_mtl_encoder e) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)e;
    [enc endEncoding];
    CFRelease(e);
}

void loom_mtl_cmdbuf_commit_wait(loom_mtl_cmdbuf cb) {
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)cb;
    [buf commit];
    [buf waitUntilCompleted];
    CFRelease(cb);
}
