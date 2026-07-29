//! Zig side of the Metal substrate: device, buffers, pipelines, dispatch.
//!
//! Everything here is plain Zig against opaque pointers; Objective-C stays in
//! shim.m. The one Apple-specific idea that leaks upward is unified memory,
//! and it leaks on purpose — `wrapHost` is why this backend exists.

const std = @import("std");

const c = @cImport({
    @cInclude("shim.h");
});

pub const Error = error{
    NoMetalDevice,
    NotPageAligned,
    BufferCreateFailed,
    ShaderCompileFailed,
};

pub const Device = struct {
    handle: c.loom_mtl_device,
    queue: c.loom_mtl_queue,

    pub fn init() Error!Device {
        const d = c.loom_mtl_device_create() orelse return error.NoMetalDevice;
        const q = c.loom_mtl_queue_create(d) orelse {
            c.loom_mtl_device_release(d);
            return error.NoMetalDevice;
        };
        return .{ .handle = d, .queue = q };
    }

    pub fn deinit(self: *Device) void {
        c.loom_mtl_queue_release(self.queue);
        c.loom_mtl_device_release(self.handle);
    }

    pub fn name(self: Device) []const u8 {
        return std.mem.span(c.loom_mtl_device_name(self.handle));
    }

    /// Apple Silicon is unified; a discrete AMD eGPU on an Intel Mac is not.
    /// The distinction decides whether `wrapHost` is sound: without unified
    /// memory the GPU would be reading host pages across PCIe.
    pub fn hasUnifiedMemory(self: Device) bool {
        return c.loom_mtl_device_has_unified_memory(self.handle) != 0;
    }

    pub fn maxThreadsPerGroup(self: Device) usize {
        return c.loom_mtl_device_max_threads_per_group(self.handle);
    }

    /// Wrap host memory as a GPU buffer with no copy.
    ///
    /// This is the whole Apple argument: an mmap'd GGUF region becomes
    /// readable by the GPU for the price of a page-table entry, so weights are
    /// never staged or uploaded. It matters twice over here, because loom's
    /// routed experts can arrive from a peer *during* a token — on a discrete
    /// GPU that would be a host-to-VRAM copy inside the critical path.
    ///
    /// Metal requires page-aligned base and length; callers align outward to
    /// the containing pages and pass the offset separately.
    pub fn wrapHost(self: Device, mem: []const u8) Error!Buffer {
        const page = std.heap.pageSize();
        if (@intFromPtr(mem.ptr) % page != 0 or mem.len % page != 0) return error.NotPageAligned;
        const b = c.loom_mtl_buffer_wrap(self.handle, @ptrCast(@constCast(mem.ptr)), mem.len) orelse
            return error.BufferCreateFailed;
        return .{ .handle = b, .len = mem.len };
    }

    /// Largest single allocation this device will make.
    pub fn maxBufferLen(self: Device) usize {
        return c.loom_mtl_max_buffer(self.handle);
    }

    /// Ask the OS to wire `buf`'s pages and keep them wired for this queue.
    /// False when the OS predates residency sets: the buffer still works, its
    /// pages are just evictable, which for a multi-gigabyte weight mapping is
    /// the difference between resident and faulted in per use.
    pub fn makeResident(self: Device, buf: Buffer) bool {
        return c.loom_mtl_resident(self.handle, self.queue, buf.handle) != 0;
    }

    /// Shared-storage allocation for activations, which the CPU also touches.
    pub fn alloc(self: Device, len: usize) Error!Buffer {
        const b = c.loom_mtl_buffer_alloc(self.handle, len) orelse return error.BufferCreateFailed;
        return .{ .handle = b, .len = len };
    }

    /// Compile MSL source and return a pipeline for `fn_name`. Shaders are
    /// compiled at startup from source rather than shipped as a metallib: it
    /// keeps the build free of the Metal toolchain, and leaves room to
    /// specialize a kernel on the model's actual shapes.
    pub fn pipeline(self: Device, src: [:0]const u8, fn_name: [:0]const u8) Error!Pipeline {
        var errbuf: [1024]u8 = undefined;
        @memset(&errbuf, 0);
        const p = c.loom_mtl_pipeline_create(self.handle, src.ptr, fn_name.ptr, &errbuf, errbuf.len) orelse {
            std.debug.print("metal: compiling {s}: {s}\n", .{ fn_name, std.mem.sliceTo(&errbuf, 0) });
            return error.ShaderCompileFailed;
        };
        return .{ .handle = p };
    }

    pub fn commandBuffer(self: Device) CommandBuffer {
        return .{ .handle = c.loom_mtl_cmdbuf_create(self.queue).? };
    }
};

pub const Buffer = struct {
    handle: c.loom_mtl_buffer,
    len: usize,

    pub fn deinit(self: *Buffer) void {
        c.loom_mtl_buffer_release(self.handle);
    }

    /// Host view of a shared buffer. Valid for `alloc`ed buffers and for
    /// wrapped ones (where it is the original pointer).
    pub fn slice(self: Buffer, comptime T: type) []T {
        const p: [*]T = @ptrCast(@alignCast(c.loom_mtl_buffer_contents(self.handle)));
        return p[0 .. self.len / @sizeOf(T)];
    }
};

pub const Pipeline = struct {
    handle: c.loom_mtl_pipeline,

    pub fn deinit(self: *Pipeline) void {
        c.loom_mtl_pipeline_release(self.handle);
    }
    pub fn maxThreads(self: Pipeline) usize {
        return c.loom_mtl_pipeline_max_threads(self.handle);
    }
};

pub const CommandBuffer = struct {
    handle: c.loom_mtl_cmdbuf,

    /// Encode one dispatch. Buffers bind to indices 0..n-1 and `constants` to
    /// index n, matching the shaders' `[[buffer(k)]]` attributes.
    pub fn dispatch(
        self: CommandBuffer,
        p: Pipeline,
        buffers: []const Buffer,
        offsets: []const usize,
        constants: ?[]const u8,
        grid: usize,
        group: usize,
    ) void {
        std.debug.assert(buffers.len == offsets.len and buffers.len <= 8);
        var handles: [8]c.loom_mtl_buffer = undefined;
        for (buffers, 0..) |b, i| handles[i] = b.handle;
        c.loom_mtl_encode(
            self.handle,
            p.handle,
            &handles,
            offsets.ptr,
            buffers.len,
            if (constants) |k| k.ptr else null,
            if (constants) |k| k.len else 0,
            grid,
            group,
        );
    }

    /// Open one encoder for several dispatches. Each `dispatch` call above
    /// opens its own encoder, and an encoder boundary is a full pipeline
    /// drain; a block of six dispatches pays six of them.
    pub fn encoder(self: CommandBuffer) Encoder {
        return .{ .handle = c.loom_mtl_encoder_begin(self.handle).? };
    }

    /// Submit and block. Batching many dispatches into one command buffer
    /// before committing is what keeps submission overhead off the per-op
    /// path; a commit per matvec would dominate.
    pub fn commitAndWait(self: CommandBuffer) void {
        c.loom_mtl_cmdbuf_commit_wait(self.handle);
    }
};

pub const Encoder = struct {
    handle: c.loom_mtl_encoder,

    pub fn dispatch(
        self: Encoder,
        p: Pipeline,
        buffers: []const Buffer,
        offsets: []const usize,
        constants: ?[]const u8,
        grid: usize,
        group: usize,
    ) void {
        std.debug.assert(buffers.len == offsets.len and buffers.len <= 8);
        var handles: [8]c.loom_mtl_buffer = undefined;
        for (buffers, 0..) |b, i| handles[i] = b.handle;
        c.loom_mtl_encoder_dispatch(
            self.handle,
            p.handle,
            &handles,
            offsets.ptr,
            buffers.len,
            if (constants) |k| k.ptr else null,
            if (constants) |k| k.len else 0,
            grid,
            group,
        );
    }

    /// Order the dispatches issued so far before those issued after. Needed
    /// only where a dispatch reads what a previous one wrote — the encoder is
    /// opened in concurrent mode, so independent dispatches overlap.
    pub fn barrier(self: Encoder) void {
        c.loom_mtl_encoder_barrier(self.handle);
    }

    pub fn end(self: Encoder) void {
        c.loom_mtl_encoder_end(self.handle);
    }
};
