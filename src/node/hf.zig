//! Model resolution: turn a `--model` spec into a local checkpoint directory,
//! downloading a loom-format checkpoint from the Hugging Face Hub if needed.
//!
//! Accepted specs:
//!   <local dir>            a directory that already contains manifest.loom
//!   tiny                   a synthetic tiny checkpoint (auto-generated, for demos)
//!   [hf:]org/repo[@rev]    download manifest.loom + dense.blob + experts.blob from
//!                          https://huggingface.co/org/repo/resolve/<rev>/<file>
//!
//! Only the loom on-disk format is fetched here. Converting raw GLM-5.2
//! safetensors (FP8 -> int4) is a separate offline step (colibri's converter);
//! it is not a node-startup operation.

const std = @import("std");
const Io = std.Io;
const model = @import("../engine/model.zig");
const gen = @import("../engine/gen_checkpoint.zig");

pub const files = [_][]const u8{ "manifest.loom", "dense.blob", "experts.blob" };

pub const Resolved = struct {
    dir: []const u8, // owned; contains manifest.loom
    source: []const u8, // "local" | "synthetic" | "huggingface"
};

fn hasManifest(io: Io, dir: []const u8) bool {
    var pbuf: [4096]u8 = undefined;
    const p = std.fmt.bufPrint(&pbuf, "{s}/manifest.loom", .{dir}) catch return false;
    Io.Dir.cwd().access(io, p, .{}) catch return false;
    return true;
}

/// Resolve `spec` to a ready-to-load checkpoint directory. `cache_root` is where
/// downloaded/synthetic checkpoints are materialized (e.g. ~/.cache/loom/models).
pub fn resolve(gpa: std.mem.Allocator, io: Io, spec: []const u8, cache_root: []const u8) !Resolved {
    // 1. an existing local checkpoint directory
    if (hasManifest(io, spec)) {
        return .{ .dir = try gpa.dupe(u8, spec), .source = "local" };
    }

    // 2. the built-in synthetic tiny model
    if (std.mem.eql(u8, spec, "tiny")) {
        const dir = try std.fmt.allocPrint(gpa, "{s}/tiny", .{cache_root});
        if (!hasManifest(io, dir)) {
            try makePath(io, dir);
            try gen.generate(gpa, io, dir, model.tinyShape(), 42);
        }
        return .{ .dir = dir, .source = "synthetic" };
    }

    // 3. a Hugging Face repo id
    var repo = spec;
    if (std.mem.startsWith(u8, repo, "hf:")) repo = repo[3..];
    var rev: []const u8 = "main";
    if (std.mem.indexOfScalar(u8, repo, '@')) |at| {
        rev = repo[at + 1 ..];
        repo = repo[0..at];
    }
    if (std.mem.indexOfScalar(u8, repo, '/') == null) return error.InvalidModelSpec;

    // cache dir name: org__repo
    const safe = try gpa.dupe(u8, repo);
    defer gpa.free(safe);
    for (safe) |*c| {
        if (c.* == '/') c.* = '_';
    }
    const dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ cache_root, safe });
    errdefer gpa.free(dir);

    if (!hasManifest(io, dir)) {
        try makePath(io, dir);
        try downloadRepo(gpa, io, repo, rev, dir);
    }
    return .{ .dir = dir, .source = "huggingface" };
}

/// mkdir -p: create every parent component of `path`, ignoring ones that exist.
fn makePath(io: Io, path: []const u8) !void {
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (i != path.len and path[i] != '/') continue;
        const prefix = path[0..i];
        if (prefix.len == 0) continue;
        Io.Dir.cwd().createDir(io, prefix, .default_dir) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };
    }
}

fn downloadRepo(gpa: std.mem.Allocator, io: Io, repo: []const u8, rev: []const u8, dir: []const u8) !void {
    var client = std.http.Client{ .allocator = gpa, .io = io };
    defer client.deinit();

    for (files) |name| {
        const url = try std.fmt.allocPrint(
            gpa,
            "https://huggingface.co/{s}/resolve/{s}/{s}",
            .{ repo, rev, name },
        );
        defer gpa.free(url);
        const dest = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, name });
        defer gpa.free(dest);
        try downloadFile(io, &client, url, dest);
    }
}

fn downloadFile(io: Io, client: *std.http.Client, url: []const u8, dest_path: []const u8) !void {
    var fbuf: [1 << 16]u8 = undefined;
    const f = try Io.Dir.cwd().createFile(io, dest_path, .{ .truncate = true });
    defer f.close(io);
    var fw = f.writer(io, &fbuf);

    const res = client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &fw.interface,
    }) catch |e| return e;
    try fw.interface.flush();
    if (res.status != .ok) return error.HttpDownloadFailed;
}
