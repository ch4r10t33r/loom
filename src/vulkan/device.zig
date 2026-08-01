//! Minimal Vulkan compute: one device, one queue, host-visible buffers, one
//! descriptor set per dispatch, submit-and-wait. The deliberate mirror of what
//! src/metal/device.zig was on day one of that backend -- enough to run one
//! kernel against an oracle, nothing speculative. The occupancy and batching
//! lessons from the Metal effort transfer as *lessons*, not as code, once a
//! physical GPU exists to measure on; llvmpipe validates correctness only.
//!
//! Bindings are hand-declared extern C rather than a generated vulkan.zig:
//! the surface actually used is ~25 functions, and a full binding is another
//! dependency for no reach.
const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{ NoVulkan, NoDevice, NoMemoryType, CreateFailed, MapFailed };

const V = struct {
    vkCreateInstance: *const fn (*const InstanceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult,
    vkEnumeratePhysicalDevices: *const fn (Handle, *u32, ?[*]Handle) callconv(.c) VkResult,
    vkGetPhysicalDeviceQueueFamilyProperties: *const fn (Handle, *u32, ?[*]QueueFamilyProperties) callconv(.c) void,
    vkGetPhysicalDeviceMemoryProperties: *const fn (Handle, *PhysicalDeviceMemoryProperties) callconv(.c) void,
    vkGetPhysicalDeviceFeatures2: *const fn (Handle, *Features2) callconv(.c) void,
    vkCreateDevice: *const fn (Handle, *const DeviceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult,
    vkGetDeviceQueue: *const fn (Handle, u32, u32, *Handle) callconv(.c) void,
    vkCreateBuffer: *const fn (Handle, *const BufferCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkGetBufferMemoryRequirements: *const fn (Handle, NDHandle, *MemoryRequirements) callconv(.c) void,
    vkAllocateMemory: *const fn (Handle, *const MemoryAllocateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkBindBufferMemory: *const fn (Handle, NDHandle, NDHandle, u64) callconv(.c) VkResult,
    vkMapMemory: *const fn (Handle, NDHandle, u64, u64, u32, *?*anyopaque) callconv(.c) VkResult,
    vkCreateShaderModule: *const fn (Handle, *const ShaderModuleCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkCreateDescriptorSetLayout: *const fn (Handle, *const DescriptorSetLayoutCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkCreatePipelineLayout: *const fn (Handle, *const PipelineLayoutCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkCreateComputePipelines: *const fn (Handle, NDHandle, u32, *const ComputePipelineCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkCreateDescriptorPool: *const fn (Handle, *const DescriptorPoolCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkAllocateDescriptorSets: *const fn (Handle, *const DescriptorSetAllocateInfo, [*]NDHandle) callconv(.c) VkResult,
    vkUpdateDescriptorSets: *const fn (Handle, u32, [*]const WriteDescriptorSet, u32, ?*anyopaque) callconv(.c) void,
    vkCreateCommandPool: *const fn (Handle, *const CommandPoolCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkAllocateCommandBuffers: *const fn (Handle, *const CommandBufferAllocateInfo, [*]Handle) callconv(.c) VkResult,
    vkBeginCommandBuffer: *const fn (Handle, *const CommandBufferBeginInfo) callconv(.c) VkResult,
    vkCmdBindPipeline: *const fn (Handle, u32, NDHandle) callconv(.c) void,
    vkCmdBindDescriptorSets: *const fn (Handle, u32, NDHandle, u32, u32, [*]const NDHandle, u32, ?*anyopaque) callconv(.c) void,
    vkCmdPushConstants: *const fn (Handle, NDHandle, u32, u32, u32, *const anyopaque) callconv(.c) void,
    vkCmdDispatch: *const fn (Handle, u32, u32, u32) callconv(.c) void,
    vkCmdCopyBuffer: *const fn (Handle, NDHandle, NDHandle, u32, [*]const BufferCopy) callconv(.c) void,
    vkCmdPipelineBarrier: *const fn (Handle, u32, u32, u32, u32, ?[*]const MemoryBarrier, u32, ?*anyopaque, u32, ?*anyopaque) callconv(.c) void,
    vkEndCommandBuffer: *const fn (Handle) callconv(.c) VkResult,
    vkQueueSubmit: *const fn (Handle, u32, *const SubmitInfo, NDHandle) callconv(.c) VkResult,
    vkQueueWaitIdle: *const fn (Handle) callconv(.c) VkResult,
    vkResetCommandPool: *const fn (Handle, NDHandle, u32) callconv(.c) VkResult,
    vkCreateQueryPool: *const fn (Handle, *const QueryPoolCreateInfo, ?*anyopaque, *NDHandle) callconv(.c) VkResult,
    vkCmdResetQueryPool: *const fn (Handle, NDHandle, u32, u32) callconv(.c) void,
    vkCmdWriteTimestamp: *const fn (Handle, u32, NDHandle, u32) callconv(.c) void,
    vkGetQueryPoolResults: *const fn (Handle, NDHandle, u32, u32, usize, *anyopaque, u64, u32) callconv(.c) VkResult,
    vkResetDescriptorPool: *const fn (Handle, NDHandle, u32) callconv(.c) VkResult,
};
var v: V = undefined;
var v_loaded = false;

fn loadVulkan() Error!void {
    if (v_loaded) return;
    if (builtin.os.tag == .windows) {
        // std.DynLib has no Windows implementation in this std snapshot;
        // LoadLibrary/GetProcAddress is the whole surface we need anyway.
        const w = struct {
            extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
            extern "kernel32" fn GetProcAddress(mod: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*const anyopaque;
        };
        const mod = w.LoadLibraryA("vulkan-1.dll") orelse return error.NoVulkan;
        inline for (@typeInfo(V).@"struct".fields) |f| {
            const p = w.GetProcAddress(mod, f.name) orelse return error.NoVulkan;
            @field(v, f.name) = @ptrCast(@alignCast(p));
        }
    } else {
        var lib = std.DynLib.open("libvulkan.so.1") catch return error.NoVulkan;
        inline for (@typeInfo(V).@"struct".fields) |f| {
            @field(v, f.name) = lib.lookup(f.type, f.name) orelse return error.NoVulkan;
        }
    }
    v_loaded = true;
}

// ---- the C surface, exactly what is used -------------------------------------
//
// Loaded at runtime with dlopen rather than linked: a loom binary must build
// on a machine with no Vulkan SDK and run on one with no Vulkan driver, and
// in both cases the backend has to decline cleanly rather than fail to link
// or fail to start. This is also what lets the macOS host cross-compile the
// Linux test binary it cannot itself run.
const VkResult = i32;
const VK_SUCCESS: VkResult = 0;
const Handle = ?*opaque {};
const NDHandle = u64; // non-dispatchable handles are 64-bit always

const InstanceCreateInfo = extern struct { sType: u32 = 1, pNext: ?*anyopaque = null, flags: u32 = 0, pApplicationInfo: ?*const ApplicationInfo = null, enabledLayerCount: u32 = 0, ppEnabledLayerNames: ?*anyopaque = null, enabledExtensionCount: u32 = 0, ppEnabledExtensionNames: ?*anyopaque = null };
const ApplicationInfo = extern struct { sType: u32 = 0, pNext: ?*anyopaque = null, pApplicationName: ?[*:0]const u8 = null, applicationVersion: u32 = 0, pEngineName: ?[*:0]const u8 = null, engineVersion: u32 = 0, apiVersion: u32 = (1 << 22) | (2 << 12) };
const QueueFamilyProperties = extern struct { queueFlags: u32, queueCount: u32, timestampValidBits: u32, minImageTransferGranularity: [3]u32 };
const PhysicalDeviceMemoryProperties = extern struct { memoryTypeCount: u32, memoryTypes: [32]extern struct { propertyFlags: u32, heapIndex: u32 }, memoryHeapCount: u32, memoryHeaps: [16]extern struct { size: u64, flags: u32 } };
const DeviceQueueCreateInfo = extern struct { sType: u32 = 2, pNext: ?*anyopaque = null, flags: u32 = 0, queueFamilyIndex: u32, queueCount: u32 = 1, pQueuePriorities: *const f32 };
const Features2 = extern struct { sType: u32 = 1000059000, pNext: ?*anyopaque = null, features: [55]u32 = @splat(0) }; // VkPhysicalDeviceFeatures body, unused
const FeaturesIntDot = extern struct { sType: u32 = 1000280000, pNext: ?*anyopaque = null, shaderIntegerDotProduct: u32 = 0 };
const Features8Bit = extern struct { sType: u32 = 1000177000, pNext: ?*anyopaque = null, storageBuffer8BitAccess: u32 = 1, uniformAndStorageBuffer8BitAccess: u32 = 0, storagePushConstant8: u32 = 0 };
const FeaturesInt8 = extern struct { sType: u32 = 1000082000, pNext: ?*anyopaque = null, shaderFloat16: u32 = 1, shaderInt8: u32 = 1 };
const DeviceCreateInfo = extern struct { sType: u32 = 3, pNext: ?*anyopaque = null, flags: u32 = 0, queueCreateInfoCount: u32 = 1, pQueueCreateInfos: *const DeviceQueueCreateInfo, enabledLayerCount: u32 = 0, ppEnabledLayerNames: ?*anyopaque = null, enabledExtensionCount: u32 = 0, ppEnabledExtensionNames: ?*const [*:0]const u8 = null, pEnabledFeatures: ?*anyopaque = null };
// usage: STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST, so any buffer can be
// a staging source or a device-local copy target.
const BufferCreateInfo = extern struct { sType: u32 = 12, pNext: ?*anyopaque = null, flags: u32 = 0, size: u64, usage: u32 = 0x23, sharingMode: u32 = 0, queueFamilyIndexCount: u32 = 0, pQueueFamilyIndices: ?*anyopaque = null };
const BufferCopy = extern struct { srcOffset: u64 = 0, dstOffset: u64 = 0, size: u64 };
const MemoryBarrier = extern struct { sType: u32 = 46, pNext: ?*anyopaque = null, srcAccessMask: u32, dstAccessMask: u32 };
const MemoryRequirements = extern struct { size: u64, alignment: u64, memoryTypeBits: u32 };
const MemoryAllocateInfo = extern struct { sType: u32 = 5, pNext: ?*anyopaque = null, allocationSize: u64, memoryTypeIndex: u32 };
const ShaderModuleCreateInfo = extern struct { sType: u32 = 16, pNext: ?*anyopaque = null, flags: u32 = 0, codeSize: usize, pCode: [*]const u32 };
const DescriptorSetLayoutBinding = extern struct { binding: u32, descriptorType: u32 = 7, descriptorCount: u32 = 1, stageFlags: u32 = 0x20, pImmutableSamplers: ?*anyopaque = null };
const DescriptorSetLayoutCreateInfo = extern struct { sType: u32 = 32, pNext: ?*anyopaque = null, flags: u32 = 0, bindingCount: u32, pBindings: [*]const DescriptorSetLayoutBinding };
const PushConstantRange = extern struct { stageFlags: u32 = 0x20, offset: u32 = 0, size: u32 };
const PipelineLayoutCreateInfo = extern struct { sType: u32 = 30, pNext: ?*anyopaque = null, flags: u32 = 0, setLayoutCount: u32 = 1, pSetLayouts: *const NDHandle, pushConstantRangeCount: u32 = 1, pPushConstantRanges: *const PushConstantRange };
const PipelineShaderStageCreateInfo = extern struct { sType: u32 = 18, pNext: ?*anyopaque = null, flags: u32 = 0, stage: u32 = 0x20, module: NDHandle, pName: [*:0]const u8 = "main", pSpecializationInfo: ?*anyopaque = null };
const ComputePipelineCreateInfo = extern struct { sType: u32 = 29, pNext: ?*anyopaque = null, flags: u32 = 0, stage: PipelineShaderStageCreateInfo, layout: NDHandle, basePipelineHandle: NDHandle = 0, basePipelineIndex: i32 = 0 };
const DescriptorPoolSize = extern struct { type: u32 = 7, descriptorCount: u32 };
const DescriptorPoolCreateInfo = extern struct { sType: u32 = 33, pNext: ?*anyopaque = null, flags: u32 = 0, maxSets: u32, poolSizeCount: u32 = 1, pPoolSizes: *const DescriptorPoolSize };
const DescriptorSetAllocateInfo = extern struct { sType: u32 = 34, pNext: ?*anyopaque = null, descriptorPool: NDHandle, descriptorSetCount: u32 = 1, pSetLayouts: *const NDHandle };
const DescriptorBufferInfo = extern struct { buffer: NDHandle, offset: u64 = 0, range: u64 };
const WriteDescriptorSet = extern struct { sType: u32 = 35, pNext: ?*anyopaque = null, dstSet: NDHandle, dstBinding: u32, dstArrayElement: u32 = 0, descriptorCount: u32 = 1, descriptorType: u32 = 7, pImageInfo: ?*anyopaque = null, pBufferInfo: *const DescriptorBufferInfo, pTexelBufferView: ?*anyopaque = null };
const QueryPoolCreateInfo = extern struct { sType: u32 = 11, pNext: ?*anyopaque = null, flags: u32 = 0, queryType: u32 = 2, queryCount: u32, pipelineStatistics: u32 = 0 };
const CommandPoolCreateInfo = extern struct { sType: u32 = 39, pNext: ?*anyopaque = null, flags: u32 = 2, queueFamilyIndex: u32 };
const CommandBufferAllocateInfo = extern struct { sType: u32 = 40, pNext: ?*anyopaque = null, commandPool: NDHandle, level: u32 = 0, commandBufferCount: u32 = 1 };
const CommandBufferBeginInfo = extern struct { sType: u32 = 42, pNext: ?*anyopaque = null, flags: u32 = 1, pInheritanceInfo: ?*anyopaque = null };
const SubmitInfo = extern struct { sType: u32 = 4, pNext: ?*anyopaque = null, waitSemaphoreCount: u32 = 0, pWaitSemaphores: ?*anyopaque = null, pWaitDstStageMask: ?*anyopaque = null, commandBufferCount: u32 = 1, pCommandBuffers: *const Handle, signalSemaphoreCount: u32 = 0, pSignalSemaphores: ?*anyopaque = null };

// ---- the API loom uses -------------------------------------------------------

pub const Buffer = struct {
    handle: NDHandle,
    mem: NDHandle,
    // Null for device-local memory with no host mapping; such a buffer is
    // written through `upload` and never sliced.
    ptr: ?[*]u8,
    len: usize,

    pub fn slice(self: Buffer, comptime T: type) []T {
        return @as([*]T, @ptrCast(@alignCast(self.ptr.?)))[0 .. self.len / @sizeOf(T)];
    }
};

pub const Device = struct {
    instance: Handle,
    phys: Handle,
    dev: Handle,
    queue: Handle,
    family: u32,
    cmd_pool: NDHandle,
    desc_pool: NDHandle,
    set_layout: NDHandle,
    pipe_layout: NDHandle,
    mem_props: PhysicalDeviceMemoryProperties,
    staging: ?Buffer = null,
    // LOOM_VK_KERNEL_PROF: a timestamp after every recorded dispatch. The
    // stream is fully barriered, so consecutive timestamps are per-dispatch
    // GPU durations -- the profiler Nsight cannot be for Vulkan compute on a
    // headless box. Totals aggregate per pipeline across submissions.
    prof: bool = false,
    int_dot: bool = false, // VK_KHR_shader_integer_dot_product available
    query_pool: NDHandle = 0,
    qcount: u32 = 0,
    qpipes: [2048]NDHandle = undefined,
    prof_pipes: [64]NDHandle = @splat(0),
    prof_ticks: [64]u64 = @splat(0),
    prof_calls: [64]u64 = @splat(0),

    pub fn init() Error!Device {
        try loadVulkan();
        var inst: Handle = null;
        const app = ApplicationInfo{};
        var ici = InstanceCreateInfo{ .pApplicationInfo = &app };
        if (v.vkCreateInstance(&ici, null, &inst) != VK_SUCCESS) return error.NoVulkan;

        var n: u32 = 0;
        _ = v.vkEnumeratePhysicalDevices(inst, &n, null);
        if (n == 0) return error.NoDevice;
        var devs: [8]Handle = @splat(null);
        n = @min(n, 8);
        _ = v.vkEnumeratePhysicalDevices(inst, &n, &devs);
        const phys = devs[0];

        var qn: u32 = 0;
        v.vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, null);
        var qf: [16]QueueFamilyProperties = undefined;
        qn = @min(qn, 16);
        v.vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, &qf);
        var family: u32 = 0;
        var found = false;
        for (qf[0..qn], 0..) |q, i| {
            if (q.queueFlags & 0x2 != 0) { // COMPUTE
                family = @intCast(i);
                found = true;
                break;
            }
        }
        if (!found) return error.NoDevice;

        // The q4_k kernel reads bytes, so 8-bit storage + int8 are required
        // features, chained; a driver without them fails device creation and
        // the backend falls back to CPU cleanly. Integer dot product is
        // PROBED, not assumed: the int8-activation kernels are optional and
        // llvmpipe must keep working without them.
        var idp_probe = FeaturesIntDot{};
        var f2 = Features2{ .pNext = &idp_probe };
        v.vkGetPhysicalDeviceFeatures2(phys, &f2);
        const has_idp = idp_probe.shaderIntegerDotProduct == 1;
        var f8 = Features8Bit{};
        var fi8 = FeaturesInt8{ .pNext = &f8 };
        var fidp = FeaturesIntDot{ .shaderIntegerDotProduct = 1 };
        if (has_idp) f8.pNext = &fidp;
        const prio: f32 = 1.0;
        const qci = DeviceQueueCreateInfo{ .queueFamilyIndex = family, .pQueuePriorities = &prio };
        const ext1: [1][*:0]const u8 = .{"VK_KHR_8bit_storage"};
        const ext2: [2][*:0]const u8 = .{ "VK_KHR_8bit_storage", "VK_KHR_shader_integer_dot_product" };
        var dci = DeviceCreateInfo{ .pNext = &fi8, .pQueueCreateInfos = &qci, .enabledExtensionCount = if (has_idp) 2 else 1, .ppEnabledExtensionNames = if (has_idp) @ptrCast(&ext2) else @ptrCast(&ext1) };
        var dev: Handle = null;
        if (v.vkCreateDevice(phys, &dci, null, &dev) != VK_SUCCESS) return error.NoDevice;
        var queue: Handle = null;
        v.vkGetDeviceQueue(dev, family, 0, &queue);

        var mp: PhysicalDeviceMemoryProperties = undefined;
        v.vkGetPhysicalDeviceMemoryProperties(phys, &mp);

        var pool: NDHandle = 0;
        const cpci = CommandPoolCreateInfo{ .queueFamilyIndex = family };
        if (v.vkCreateCommandPool(dev, &cpci, null, &pool) != VK_SUCCESS) return error.CreateFailed;

        const binds = [_]DescriptorSetLayoutBinding{ .{ .binding = 0 }, .{ .binding = 1 }, .{ .binding = 2 }, .{ .binding = 3 }, .{ .binding = 4 } };
        var slci = DescriptorSetLayoutCreateInfo{ .bindingCount = binds.len, .pBindings = &binds };
        var slayout: NDHandle = 0;
        if (v.vkCreateDescriptorSetLayout(dev, &slci, null, &slayout) != VK_SUCCESS) return error.CreateFailed;
        const pcr = PushConstantRange{ .size = 64 };
        var plci = PipelineLayoutCreateInfo{ .pSetLayouts = &slayout, .pPushConstantRanges = &pcr };
        var playout: NDHandle = 0;
        if (v.vkCreatePipelineLayout(dev, &plci, null, &playout) != VK_SUCCESS) return error.CreateFailed;

        const ps = DescriptorPoolSize{ .descriptorCount = 8192 };
        var dpci = DescriptorPoolCreateInfo{ .maxSets = 2048, .pPoolSizes = &ps };
        var dpool: NDHandle = 0;
        if (v.vkCreateDescriptorPool(dev, &dpci, null, &dpool) != VK_SUCCESS) return error.CreateFailed;

        var self = Device{ .instance = inst, .phys = phys, .dev = dev, .queue = queue, .family = family, .cmd_pool = pool, .desc_pool = dpool, .set_layout = slayout, .pipe_layout = playout, .mem_props = mp };
        self.int_dot = has_idp;
        if (std.c.getenv("LOOM_VK_KERNEL_PROF") != null) {
            var qp: NDHandle = 0;
            var qpci = QueryPoolCreateInfo{ .queryCount = 2048 };
            if (v.vkCreateQueryPool(dev, &qpci, null, &qp) == VK_SUCCESS) {
                self.query_pool = qp;
                self.prof = true;
            }
        }
        return self;
    }

    fn allocIn(self: *Device, len: usize, want_flags: u32) Error!Buffer {
        var buf: NDHandle = 0;
        // Padded by a word: the vectorized kernels' unaligned loadU32 helper
        // may read the word straddling the last byte.
        var bci = BufferCreateInfo{ .size = len + 4 };
        if (v.vkCreateBuffer(self.dev, &bci, null, &buf) != VK_SUCCESS) return error.CreateFailed;
        var req: MemoryRequirements = undefined;
        v.vkGetBufferMemoryRequirements(self.dev, buf, &req);
        var idx: u32 = 0;
        var ok = false;
        for (0..self.mem_props.memoryTypeCount) |i| {
            const flags = self.mem_props.memoryTypes[i].propertyFlags;
            if (req.memoryTypeBits & (@as(u32, 1) << @intCast(i)) != 0 and flags & want_flags == want_flags) {
                idx = @intCast(i);
                ok = true;
                break;
            }
        }
        if (!ok) return error.NoMemoryType;
        var mem: NDHandle = 0;
        var mai = MemoryAllocateInfo{ .allocationSize = req.size, .memoryTypeIndex = idx };
        if (v.vkAllocateMemory(self.dev, &mai, null, &mem) != VK_SUCCESS) return error.CreateFailed;
        if (v.vkBindBufferMemory(self.dev, buf, mem, 0) != VK_SUCCESS) return error.CreateFailed;
        // Map when the chosen type allows it; a device-local-only type stays
        // unmapped and is written through `upload`.
        var ptr: ?*anyopaque = null;
        const flags = self.mem_props.memoryTypes[idx].propertyFlags;
        if (flags & 0x6 == 0x6) {
            if (v.vkMapMemory(self.dev, mem, 0, len, 0, &ptr) != VK_SUCCESS) return error.MapFailed;
        }
        return .{ .handle = buf, .mem = mem, .ptr = if (ptr) |p| @ptrCast(p) else null, .len = len };
    }

    /// Host-visible, host-coherent storage buffer: activations, staging, and
    /// anything the host reads back.
    pub fn alloc(self: *Device, len: usize) Error!Buffer {
        return self.allocIn(len, 0x6); // HOST_VISIBLE|HOST_COHERENT
    }

    /// Device-local storage for weights. On a discrete GPU this is VRAM and
    /// comes back unmapped; on llvmpipe and UMA devices the device-local type
    /// is also host-visible, so it maps and `upload` degenerates to memcpy.
    /// Either way the caller writes it through `upload` only.
    pub fn allocDevice(self: *Device, len: usize) Error!Buffer {
        return self.allocIn(len, 0x1) catch self.alloc(len); // DEVICE_LOCAL, else wherever fits
    }

    /// Write `data` into `dst`: direct memcpy when mapped, otherwise a staged
    /// copy through a grow-only host-visible bounce buffer and a one-shot
    /// transfer submission.
    pub fn upload(self: *Device, dst: Buffer, data: []const u8) Error!void {
        if (dst.ptr != null) {
            @memcpy(dst.slice(u8)[0..data.len], data);
            return;
        }
        if (self.staging == null or self.staging.?.len < data.len) self.staging = try self.alloc(data.len);
        const st = self.staging.?;
        @memcpy(st.slice(u8)[0..data.len], data);
        const c = try self.beginCmd();
        const region = BufferCopy{ .size = data.len };
        v.vkCmdCopyBuffer(c.cmd, st.handle, dst.handle, 1, @ptrCast(&region));
        try self.submitWait(c);
    }

    pub fn pipeline(self: *Device, spv: []const u8) Error!NDHandle {
        var mod: NDHandle = 0;
        var smci = ShaderModuleCreateInfo{ .codeSize = spv.len, .pCode = @ptrCast(@alignCast(spv.ptr)) };
        if (v.vkCreateShaderModule(self.dev, &smci, null, &mod) != VK_SUCCESS) return error.CreateFailed;
        var pipe: NDHandle = 0;
        var cpci = ComputePipelineCreateInfo{ .stage = .{ .module = mod }, .layout = self.pipe_layout };
        if (v.vkCreateComputePipelines(self.dev, 0, 1, &cpci, null, &pipe) != VK_SUCCESS) return error.CreateFailed;
        return pipe;
    }

    /// Offset variants for row-granular cache access.
    pub fn uploadAt(self: *Device, dst: Buffer, dst_off: usize, data: []const u8) Error!void {
        if (dst.ptr != null) {
            @memcpy(dst.slice(u8)[dst_off..][0..data.len], data);
            return;
        }
        if (self.staging == null or self.staging.?.len < data.len) self.staging = try self.alloc(data.len);
        const st = self.staging.?;
        @memcpy(st.slice(u8)[0..data.len], data);
        const c = try self.beginCmd();
        const region = BufferCopy{ .dstOffset = dst_off, .size = data.len };
        v.vkCmdCopyBuffer(c.cmd, st.handle, dst.handle, 1, @ptrCast(&region));
        try self.submitWait(c);
    }

    pub fn downloadAt(self: *Device, src: Buffer, src_off: usize, out: []u8) Error!void {
        if (src.ptr != null) {
            @memcpy(out, src.slice(u8)[src_off..][0..out.len]);
            return;
        }
        if (self.staging == null or self.staging.?.len < out.len) self.staging = try self.alloc(out.len);
        const st = self.staging.?;
        const c = try self.beginCmd();
        const region = BufferCopy{ .srcOffset = src_off, .size = out.len };
        v.vkCmdCopyBuffer(c.cmd, src.handle, st.handle, 1, @ptrCast(&region));
        try self.submitWait(c);
        @memcpy(out, st.slice(u8)[0..out.len]);
    }

    /// Read a device-local buffer back to host memory: direct memcpy when
    /// mapped, otherwise a staged copy through the bounce buffer.
    pub fn download(self: *Device, src: Buffer, out: []u8) Error!void {
        if (src.ptr != null) {
            @memcpy(out, src.slice(u8)[0..out.len]);
            return;
        }
        if (self.staging == null or self.staging.?.len < out.len) self.staging = try self.alloc(out.len);
        const st = self.staging.?;
        const c = try self.beginCmd();
        const region = BufferCopy{ .size = out.len };
        v.vkCmdCopyBuffer(c.cmd, src.handle, st.handle, 1, @ptrCast(&region));
        try self.submitWait(c);
        @memcpy(out, st.slice(u8)[0..out.len]);
    }

    /// An open command buffer recording dispatches. Descriptor sets come from
    /// the shared pool, reset at `beginCmd` -- one recording can hold up to
    /// the pool's 64 sets. This is what turns a chain of ops into one
    /// submission, which on a discrete GPU is the difference between the
    /// device working and the device waiting on the host ~500 times a token.
    pub const Cmd = struct { cmd: Handle };

    pub fn beginCmd(self: *Device) Error!Cmd {
        _ = v.vkResetDescriptorPool(self.dev, self.desc_pool, 0);
        _ = v.vkResetCommandPool(self.dev, self.cmd_pool, 0);
        var cmd: Handle = null;
        var cbai = CommandBufferAllocateInfo{ .commandPool = self.cmd_pool };
        if (v.vkAllocateCommandBuffers(self.dev, &cbai, @ptrCast(&cmd)) != VK_SUCCESS) return error.CreateFailed;
        const begin = CommandBufferBeginInfo{};
        _ = v.vkBeginCommandBuffer(cmd, &begin);
        if (self.prof) {
            v.vkCmdResetQueryPool(cmd, self.query_pool, 0, 2048);
            v.vkCmdWriteTimestamp(cmd, 0x2000, self.query_pool, 0); // BOTTOM_OF_PIPE baseline
            self.qcount = 1;
        }
        return .{ .cmd = cmd };
    }

    pub fn record(self: *Device, c: Cmd, pipe: NDHandle, bufs: []const Buffer, push: []const u8, groups_x: u32) Error!void {
        var set: NDHandle = 0;
        var dsai = DescriptorSetAllocateInfo{ .descriptorPool = self.desc_pool, .pSetLayouts = &self.set_layout };
        if (v.vkAllocateDescriptorSets(self.dev, &dsai, @ptrCast(&set)) != VK_SUCCESS) return error.CreateFailed;
        var infos: [5]DescriptorBufferInfo = undefined;
        var writes: [5]WriteDescriptorSet = undefined;
        for (bufs, 0..) |b, i| {
            infos[i] = .{ .buffer = b.handle, .range = b.len };
            writes[i] = .{ .dstSet = set, .dstBinding = @intCast(i), .pBufferInfo = &infos[i] };
        }
        v.vkUpdateDescriptorSets(self.dev, @intCast(bufs.len), &writes, 0, null);
        v.vkCmdBindPipeline(c.cmd, 1, pipe); // COMPUTE bind point
        v.vkCmdBindDescriptorSets(c.cmd, 1, self.pipe_layout, 0, 1, @ptrCast(&set), 0, null);
        v.vkCmdPushConstants(c.cmd, self.pipe_layout, 0x20, 0, @intCast(push.len), push.ptr);
        v.vkCmdDispatch(c.cmd, groups_x, 1, 1);
        if (self.prof and self.qcount < 2048) {
            self.qpipes[self.qcount] = pipe;
            v.vkCmdWriteTimestamp(c.cmd, 0x2000, self.query_pool, self.qcount);
            self.qcount += 1;
        }
    }

    /// Compute-to-compute execution and memory dependency: everything written
    /// by dispatches recorded before it is visible to dispatches after it.
    /// (An intermittent wrong-output hunt once blamed this barrier's scope
    /// and widened it to ALL_COMMANDS; the real culprit was a stale entry in
    /// the backend's pointer-keyed weight cache, and the soak that cleared
    /// the barrier's name is in the frame test's history.)
    pub fn barrier(self: *Device, c: Cmd) void {
        _ = self;
        // LOOM_NO_BARRIER: diagnostic ONLY -- output is garbage, but the wall
        // time bounds what barrier drains cost. Never in a correctness path.
        if (std.c.getenv("LOOM_NO_BARRIER") != null) return;
        const mb = MemoryBarrier{ .srcAccessMask = 0x40, .dstAccessMask = 0x20 | 0x40 }; // SHADER_WRITE -> SHADER_READ|WRITE
        v.vkCmdPipelineBarrier(c.cmd, 0x800, 0x800, 0, 1, @ptrCast(&mb), 0, null, 0, null); // COMPUTE -> COMPUTE
    }

    pub fn submitWait(self: *Device, c: Cmd) Error!void {
        _ = v.vkEndCommandBuffer(c.cmd);
        var cmd = c.cmd;
        var si = SubmitInfo{ .pCommandBuffers = &cmd };
        if (v.vkQueueSubmit(self.queue, 1, &si, 0) != VK_SUCCESS) return error.CreateFailed;
        _ = v.vkQueueWaitIdle(self.queue);
        if (self.prof and self.qcount > 1) {
            var ts: [2048]u64 = undefined;
            if (v.vkGetQueryPoolResults(self.dev, self.query_pool, 0, self.qcount, self.qcount * 8, &ts, 8, 1 | 2) == VK_SUCCESS) {
                for (1..self.qcount) |i| {
                    const dur = ts[i] -% ts[i - 1];
                    for (&self.prof_pipes, &self.prof_ticks, &self.prof_calls) |*ph, *pt, *pn| {
                        if (ph.* == self.qpipes[i] or ph.* == 0) {
                            ph.* = self.qpipes[i];
                            pt.* += dur;
                            pn.* += 1;
                            break;
                        }
                    }
                }
            }
            self.qcount = 0;
        }
    }

    /// One dispatch as one submission -- `beginCmd`/`record`/`submitWait` for
    /// the single-op callers.
    pub fn dispatchWait(self: *Device, pipe: NDHandle, bufs: []const Buffer, push: []const u8, groups_x: u32) Error!void {
        const c = try self.beginCmd();
        try self.record(c, pipe, bufs, push, groups_x);
        try self.submitWait(c);
    }
};
