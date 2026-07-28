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
    const gpu = b.option([]const u8, "gpu", "Compute backend: none (default), metal, vulkan") orelse "none";
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
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
