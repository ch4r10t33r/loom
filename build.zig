const std = @import("std");

/// Default version source: build.zig.zon. A release build overrides it with
/// `-Dversion=<v>` derived from the git tag, so cutting a release is merge +
/// tag with no version-bump commit. Baked into
/// the binary so a downloaded executable can report exactly what it is.
const version = @import("build.zig.zon").version;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const early_native_macos = target.result.os.tag == .macos and target.query.isNative();

    // A native macOS build otherwise inherits the BUILD machine's OS as its
    // deployment target: a binary built on a new runner is stamped with that
    // minos and dyld refuses to load it anywhere older ("built for macOS
    // 26.x which is newer than running OS", hit by the first alpha tester),
    // and @available-guarded classes (MTLResidencySet, macOS 15+) get strong
    // classrefs instead of weak ones. Pinning the minimum keeps the runtime
    // guards meaningful and the binary loadable back to macOS 13.
    const eff_target = blk: {
        if (!early_native_macos) break :blk target;
        var q = target.query;
        q.os_version_min = .{ .semver = .{ .major = 13, .minor = 0, .patch = 0 } };
        break :blk b.resolveTargetQuery(q);
    };

    const snappy = b.dependency("zig_snappy", .{ .target = eff_target, .optimize = optimize });
    const vector_index = b.dependency("vector_index", .{ .target = eff_target, .optimize = optimize });

    // `-Dcommit=<sha>` from CI; "unknown" for a plain local build, which is
    // itself useful information when someone reports a bug.
    const commit = b.option([]const u8, "commit", "Git commit the build came from") orelse "unknown";
    const effective_version = b.option([]const u8, "version", "Release version (from the git tag in CI)") orelse version;
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
    // Only for a *native* macOS build. Cross-compiling to macOS cannot link
    // the Metal and Foundation frameworks -- Zig finds the SDK through xcrun
    // for the host target only -- so defaulting to Metal on any macOS target
    // breaks `-Dtarget=aarch64-macos`, which is exactly how the release
    // workflow builds. A release job that wants the GPU path passes
    // `-Dgpu=metal` and must build natively.
    const native_macos = target.result.os.tag == .macos and target.query.isNative();
    const default_gpu: []const u8 = if (native_macos) "metal" else "none";

    const gpu = b.option([]const u8, "gpu", "Compute backend: metal (default on macOS), vulkan, none") orelse default_gpu;
    const build_info = b.addOptions();
    build_info.addOption([]const u8, "version", effective_version);
    build_info.addOption([]const u8, "commit", commit);
    build_info.addOption([]const u8, "gpu", gpu);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = eff_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "snappyz", .module = snappy.module("snappyz") },
            .{ .name = "vector_index", .module = vector_index.module("vector_index") },
            .{ .name = "build_info", .module = build_info.createModule() },
        },
    });
    // Unconditionally, not only for the Metal build. macOS links libc whatever
    // you ask for, so `std.c` calls compile there and fail to cross-compile
    // for Linux with "dependency on libc must be explicitly specified" -- which
    // is how the Linux build broke without anyone noticing. A networked daemon
    // linking libc is unremarkable; silently building for one platform only is
    // not.
    exe_mod.link_libc = true;
    addBrotli(b, exe_mod);
    const exe = b.addExecutable(.{ .name = "loom", .root_module = exe_mod });
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
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = eff_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "snappyz", .module = snappy.module("snappyz") },
            .{ .name = "vector_index", .module = vector_index.module("vector_index") },
            .{ .name = "build_info", .module = build_info.createModule() },
        },
    });
    // Same reason as the executable: macOS links libc implicitly, so a `std.c`
    // call compiles here and fails to build on Linux. The test binary needs it
    // stated as explicitly as the exe does — CI runs on Linux and caught this
    // when only the exe had been fixed.
    test_mod.link_libc = true;
    addBrotli(b, test_mod);
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    if (std.mem.eql(u8, gpu, "metal")) addMetal(b, unit_tests);
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}

/// Compile the Metal shim and link the frameworks it needs. ARC is on so the
/// shim can use ordinary Objective-C object lifetime; modules so it can
/// `@import` the frameworks without a bridging header.
/// Brotli, vendored: pure C, zero dependencies, MIT -- it compiles into the
/// static binaries of every release target, so at-rest chunk compression is
/// always available. (FAISS stays dlopen by explicit decision: C++ + BLAS +
/// OpenMP does not belong in a 4-target musl-static cross build, and loom's
/// exact scan is the same algorithm FAISS-Flat runs at loom's scale.)
fn addBrotli(b: *std.Build, m: *std.Build.Module) void {
    const dep = b.dependency("brotli", .{});
    m.addIncludePath(dep.path("c/include"));
    m.addCSourceFiles(.{ .root = dep.path("c"), .files = &.{
        "common/constants.c",      "common/context.c",                 "common/dictionary.c",
        "common/platform.c",       "common/shared_dictionary.c",       "common/transform.c",
        "dec/bit_reader.c",        "dec/decode.c",                     "dec/huffman.c",
        "dec/state.c",             "enc/backward_references.c",        "enc/backward_references_hq.c",
        "enc/bit_cost.c",          "enc/block_splitter.c",             "enc/brotli_bit_stream.c",
        "enc/cluster.c",           "enc/command.c",                    "enc/compound_dictionary.c",
        "enc/compress_fragment.c", "enc/compress_fragment_two_pass.c", "enc/dictionary_hash.c",
        "enc/encode.c",            "enc/encoder_dict.c",               "enc/entropy_encode.c",
        "enc/fast_log.c",          "enc/histogram.c",                  "enc/literal_cost.c",
        "enc/memory.c",            "enc/metablock.c",                  "enc/static_dict.c",
        "enc/utf8_util.c",
    }, .flags = &.{} });
}

fn addMetal(b: *std.Build, c: *std.Build.Step.Compile) void {
    const m = c.root_module;
    m.addCSourceFile(.{
        .file = b.path("src/metal/shim.m"),
        .flags = &.{ "-fobjc-arc", "-fmodules" },
    });
    m.addIncludePath(b.path("src/metal"));
    // With an explicit os_version_min the target no longer counts as native,
    // so Zig stops resolving the SDK through xcrun on its own. Point the
    // framework and header search at the host SDK explicitly; the pinned
    // minos in the target still governs availability and the load command.
    const sdk = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), " \n\r\t");
    m.addSystemFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    m.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{sdk}) });
    m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/usr/lib", .{sdk}) });
    m.linkFramework("Metal", .{});
    m.linkFramework("Foundation", .{});
    m.link_libc = true;
}
