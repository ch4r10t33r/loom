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

pub const Error = error{ NoVulkan, NoDevice, NoMemoryType, CreateFailed, MapFailed };

const V = struct {
    vkCreateInstance: *const fn (*const InstanceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult,
    vkEnumeratePhysicalDevices: *const fn (Handle, *u32, ?[*]Handle) callconv(.c) VkResult,
    vkGetPhysicalDeviceQueueFamilyProperties: *const fn (Handle, *u32, ?[*]QueueFamilyProperties) callconv(.c) void,
    vkGetPhysicalDeviceMemoryProperties: *const fn (Handle, *PhysicalDeviceMemoryProperties) callconv(.c) void,
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
    vkEndCommandBuffer: *const fn (Handle) callconv(.c) VkResult,
    vkQueueSubmit: *const fn (Handle, u32, *const SubmitInfo, NDHandle) callconv(.c) VkResult,
    vkQueueWaitIdle: *const fn (Handle) callconv(.c) VkResult,
    vkResetCommandPool: *const fn (Handle, NDHandle, u32) callconv(.c) VkResult,
    vkResetDescriptorPool: *const fn (Handle, NDHandle, u32) callconv(.c) VkResult,
};
var v: V = undefined;
var v_loaded = false;

fn loadVulkan() Error!void {
    if (v_loaded) return;
    var lib = std.DynLib.open("libvulkan.so.1") catch return error.NoVulkan;
    inline for (@typeInfo(V).@"struct".fields) |f| {
        @field(v, f.name) = lib.lookup(f.type, f.name) orelse return error.NoVulkan;
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
const ApplicationInfo = extern struct { sType: u32 = 0, pNext: ?*anyopaque = null, pApplicationName: ?[*:0]const u8 = null, applicationVersion: u32 = 0, pEngineName: ?[*:0]const u8 = null, engineVersion: u32 = 0, apiVersion: u32 = (1 << 22) | (1 << 12) };
const QueueFamilyProperties = extern struct { queueFlags: u32, queueCount: u32, timestampValidBits: u32, minImageTransferGranularity: [3]u32 };
const PhysicalDeviceMemoryProperties = extern struct { memoryTypeCount: u32, memoryTypes: [32]extern struct { propertyFlags: u32, heapIndex: u32 }, memoryHeapCount: u32, memoryHeaps: [16]extern struct { size: u64, flags: u32 } };
const DeviceQueueCreateInfo = extern struct { sType: u32 = 2, pNext: ?*anyopaque = null, flags: u32 = 0, queueFamilyIndex: u32, queueCount: u32 = 1, pQueuePriorities: *const f32 };
const Features8Bit = extern struct { sType: u32 = 1000177000, pNext: ?*anyopaque = null, storageBuffer8BitAccess: u32 = 1, uniformAndStorageBuffer8BitAccess: u32 = 0, storagePushConstant8: u32 = 0 };
const FeaturesInt8 = extern struct { sType: u32 = 1000082000, pNext: ?*anyopaque = null, shaderFloat16: u32 = 0, shaderInt8: u32 = 1 };
const DeviceCreateInfo = extern struct { sType: u32 = 3, pNext: ?*anyopaque = null, flags: u32 = 0, queueCreateInfoCount: u32 = 1, pQueueCreateInfos: *const DeviceQueueCreateInfo, enabledLayerCount: u32 = 0, ppEnabledLayerNames: ?*anyopaque = null, enabledExtensionCount: u32 = 0, ppEnabledExtensionNames: ?*const [*:0]const u8 = null, pEnabledFeatures: ?*anyopaque = null };
const BufferCreateInfo = extern struct { sType: u32 = 12, pNext: ?*anyopaque = null, flags: u32 = 0, size: u64, usage: u32 = 0x20, sharingMode: u32 = 0, queueFamilyIndexCount: u32 = 0, pQueueFamilyIndices: ?*anyopaque = null };
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
const CommandPoolCreateInfo = extern struct { sType: u32 = 39, pNext: ?*anyopaque = null, flags: u32 = 2, queueFamilyIndex: u32 };
const CommandBufferAllocateInfo = extern struct { sType: u32 = 40, pNext: ?*anyopaque = null, commandPool: NDHandle, level: u32 = 0, commandBufferCount: u32 = 1 };
const CommandBufferBeginInfo = extern struct { sType: u32 = 42, pNext: ?*anyopaque = null, flags: u32 = 1, pInheritanceInfo: ?*anyopaque = null };
const SubmitInfo = extern struct { sType: u32 = 4, pNext: ?*anyopaque = null, waitSemaphoreCount: u32 = 0, pWaitSemaphores: ?*anyopaque = null, pWaitDstStageMask: ?*anyopaque = null, commandBufferCount: u32 = 1, pCommandBuffers: *const Handle, signalSemaphoreCount: u32 = 0, pSignalSemaphores: ?*anyopaque = null };

// ---- the API loom uses -------------------------------------------------------

pub const Buffer = struct {
    handle: NDHandle,
    mem: NDHandle,
    ptr: [*]u8,
    len: usize,

    pub fn slice(self: Buffer, comptime T: type) []T {
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0 .. self.len / @sizeOf(T)];
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
        // the backend falls back to CPU cleanly.
        var f8 = Features8Bit{};
        var fi8 = FeaturesInt8{ .pNext = &f8 };
        const prio: f32 = 1.0;
        const qci = DeviceQueueCreateInfo{ .queueFamilyIndex = family, .pQueuePriorities = &prio };
        const ext: [1][*:0]const u8 = .{"VK_KHR_8bit_storage"};
        var dci = DeviceCreateInfo{ .pNext = &fi8, .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = @ptrCast(&ext) };
        var dev: Handle = null;
        if (v.vkCreateDevice(phys, &dci, null, &dev) != VK_SUCCESS) return error.NoDevice;
        var queue: Handle = null;
        v.vkGetDeviceQueue(dev, family, 0, &queue);

        var mp: PhysicalDeviceMemoryProperties = undefined;
        v.vkGetPhysicalDeviceMemoryProperties(phys, &mp);

        var pool: NDHandle = 0;
        const cpci = CommandPoolCreateInfo{ .queueFamilyIndex = family };
        if (v.vkCreateCommandPool(dev, &cpci, null, &pool) != VK_SUCCESS) return error.CreateFailed;

        const binds = [_]DescriptorSetLayoutBinding{ .{ .binding = 0 }, .{ .binding = 1 }, .{ .binding = 2 }, .{ .binding = 3 } };
        var slci = DescriptorSetLayoutCreateInfo{ .bindingCount = binds.len, .pBindings = &binds };
        var slayout: NDHandle = 0;
        if (v.vkCreateDescriptorSetLayout(dev, &slci, null, &slayout) != VK_SUCCESS) return error.CreateFailed;
        const pcr = PushConstantRange{ .size = 32 };
        var plci = PipelineLayoutCreateInfo{ .pSetLayouts = &slayout, .pPushConstantRanges = &pcr };
        var playout: NDHandle = 0;
        if (v.vkCreatePipelineLayout(dev, &plci, null, &playout) != VK_SUCCESS) return error.CreateFailed;

        const ps = DescriptorPoolSize{ .descriptorCount = 256 };
        var dpci = DescriptorPoolCreateInfo{ .maxSets = 64, .pPoolSizes = &ps };
        var dpool: NDHandle = 0;
        if (v.vkCreateDescriptorPool(dev, &dpci, null, &dpool) != VK_SUCCESS) return error.CreateFailed;

        return .{ .instance = inst, .phys = phys, .dev = dev, .queue = queue, .family = family, .cmd_pool = pool, .desc_pool = dpool, .set_layout = slayout, .pipe_layout = playout, .mem_props = mp };
    }

    /// Host-visible, host-coherent storage buffer -- llvmpipe and every UMA
    /// device serves these; a discrete-GPU staging path is future work with
    /// the hardware to justify it.
    pub fn alloc(self: *Device, len: usize) Error!Buffer {
        var buf: NDHandle = 0;
        var bci = BufferCreateInfo{ .size = len };
        if (v.vkCreateBuffer(self.dev, &bci, null, &buf) != VK_SUCCESS) return error.CreateFailed;
        var req: MemoryRequirements = undefined;
        v.vkGetBufferMemoryRequirements(self.dev, buf, &req);
        var idx: u32 = 0;
        var ok = false;
        for (0..self.mem_props.memoryTypeCount) |i| {
            const flags = self.mem_props.memoryTypes[i].propertyFlags;
            if (req.memoryTypeBits & (@as(u32, 1) << @intCast(i)) != 0 and flags & 0x6 == 0x6) { // HOST_VISIBLE|HOST_COHERENT
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
        var ptr: ?*anyopaque = null;
        if (v.vkMapMemory(self.dev, mem, 0, len, 0, &ptr) != VK_SUCCESS) return error.MapFailed;
        return .{ .handle = buf, .mem = mem, .ptr = @ptrCast(ptr.?), .len = len };
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

    /// Bind up to four buffers, push constants, dispatch, wait. One-shot per
    /// call -- the bring-up shape, not the fast one.
    pub fn dispatchWait(self: *Device, pipe: NDHandle, bufs: []const Buffer, push: []const u8, groups_x: u32) Error!void {
        _ = v.vkResetDescriptorPool(self.dev, self.desc_pool, 0);
        var set: NDHandle = 0;
        var dsai = DescriptorSetAllocateInfo{ .descriptorPool = self.desc_pool, .pSetLayouts = &self.set_layout };
        if (v.vkAllocateDescriptorSets(self.dev, &dsai, @ptrCast(&set)) != VK_SUCCESS) return error.CreateFailed;
        var infos: [4]DescriptorBufferInfo = undefined;
        var writes: [4]WriteDescriptorSet = undefined;
        for (bufs, 0..) |b, i| {
            infos[i] = .{ .buffer = b.handle, .range = b.len };
            writes[i] = .{ .dstSet = set, .dstBinding = @intCast(i), .pBufferInfo = &infos[i] };
        }
        v.vkUpdateDescriptorSets(self.dev, @intCast(bufs.len), &writes, 0, null);

        _ = v.vkResetCommandPool(self.dev, self.cmd_pool, 0);
        var cmd: Handle = null;
        var cbai = CommandBufferAllocateInfo{ .commandPool = self.cmd_pool };
        if (v.vkAllocateCommandBuffers(self.dev, &cbai, @ptrCast(&cmd)) != VK_SUCCESS) return error.CreateFailed;
        const begin = CommandBufferBeginInfo{};
        _ = v.vkBeginCommandBuffer(cmd, &begin);
        v.vkCmdBindPipeline(cmd, 1, pipe); // COMPUTE bind point
        v.vkCmdBindDescriptorSets(cmd, 1, self.pipe_layout, 0, 1, @ptrCast(&set), 0, null);
        v.vkCmdPushConstants(cmd, self.pipe_layout, 0x20, 0, @intCast(push.len), push.ptr);
        v.vkCmdDispatch(cmd, groups_x, 1, 1);
        _ = v.vkEndCommandBuffer(cmd);
        var si = SubmitInfo{ .pCommandBuffers = &cmd };
        if (v.vkQueueSubmit(self.queue, 1, &si, 0) != VK_SUCCESS) return error.CreateFailed;
        _ = v.vkQueueWaitIdle(self.queue);
    }
};
