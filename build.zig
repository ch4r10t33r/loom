const std = @import("std");

/// Single source of truth for the release version: build.zig.zon. Baked into
/// the binary so a downloaded executable can report exactly what it is.
const version = @import("build.zig.zon").version;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const snappy = b.dependency("zig_snappy", .{ .target = target, .optimize = optimize });

    // `-Dcommit=<sha>` from CI; "unknown" for a plain local build, which is
    // itself useful information when someone reports a bug.
    const commit = b.option([]const u8, "commit", "Git commit the build came from") orelse "unknown";
    // Compute backend, resolved at comptime in src/compute/backend.zig. The
    // inactive backends are not compiled, so a GPU toolchain never becomes a
    // requirement for a CPU build (issue #10).
    // Default to the platform's GPU backend where one exists. macOS gets
    // Metal; everything else stays on the CPU until the Vulkan backend is
    // written (issue #13), because defaulting to a backend that does not
    // compile would just be a build failure with extra steps.
    //
    // Enabling a backend at build time is not the same as using it: the
    // engine probes CPU against GPU on the loaded model's own shapes at
    // startup and picks per operation. Building it in only makes that choice
    // available.
    const default_gpu: []const u8 = if (target.result.os.tag == .macos) "metal" else "none";
    const gpu = b.option([]const u8, "gpu", "Compute backend: metal (default on macOS), vulkan, none") orelse default_gpu;
    const build_info = b.addOptions();
    build_info.addOption([]const u8, "version", version);
    build_info.addOption([]const u8, "commit", commit);
    build_info.addOption([]const u8, "gpu", gpu);

    const exe = b.addExecutable(.{
        .name = "loom",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "snappyz", .module = snappy.module("snappyz") },
                .{ .name = "build_info", .module = build_info.createModule() },
            },
        }),
    });
    // Metal needs one Objective-C translation unit and two system frameworks.
    // Only compiled when the backend is selected, so a CPU build never needs a
    // macOS SDK path or an ObjC compiler.
    if (std.mem.eql(u8, gpu, "metal")) {
        addMetal(b, exe);
    }
    b.installArtifact(exe);

    // `zig build run -- <args>`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the loom CLI");
    run_step.dependOn(&run_cmd.step);

    // `zig build test`
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "snappyz", .module = snappy.module("snappyz") },
                .{ .name = "build_info", .module = build_info.createModule() },
            },
        }),
    });
    if (std.mem.eql(u8, gpu, "metal")) addMetal(b, unit_tests);
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}

/// Compile the Metal shim and link the frameworks it needs. ARC is on so the
/// shim can use ordinary Objective-C object lifetime; modules so it can
/// `@import` the frameworks without a bridging header.
fn addMetal(b: *std.Build, c: *std.Build.Step.Compile) void {
    const m = c.root_module;
    m.addCSourceFile(.{
        .file = b.path("src/metal/shim.m"),
        .flags = &.{ "-fobjc-arc", "-fmodules" },
    });
    m.addIncludePath(b.path("src/metal"));
    m.linkFramework("Metal", .{});
    m.linkFramework("Foundation", .{});
    m.link_libc = true;
}
