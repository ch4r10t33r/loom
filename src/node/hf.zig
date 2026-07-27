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

/// Written last, after every file has landed. `hasManifest` gates on it so a
/// download interrupted midway is never mistaken for a usable checkpoint
/// (security issue #32): manifest.loom is fetched first, so an abort after it
/// but before the blobs used to leave a directory that looked complete forever.
const DONE_MARKER = ".loom-complete";

/// Per-file ceiling. `experts.blob` is legitimately large, so this is a sanity
/// bound against a hostile or broken endpoint filling the disk, not a tight
/// limit (security issue #32).
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024 * 1024; // 512 GiB

/// Redirect hops to follow. Each hop is re-checked for https.
const MAX_REDIRECTS: usize = 5;

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

/// A *downloaded* checkpoint counts as usable only if the completion marker is
/// present (security issue #32). Locally-provided directories are judged by
/// manifest.loom alone, since nothing downloaded them.
fn isComplete(io: Io, dir: []const u8) bool {
    var pbuf: [4096]u8 = undefined;
    const p = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ dir, DONE_MARKER }) catch return false;
    Io.Dir.cwd().access(io, p, .{}) catch return false;
    return hasManifest(io, dir);
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

    if (!isComplete(io, dir)) {
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
        try downloadFile(gpa, io, &client, url, dest);
    }
    // every file landed: mark the directory usable
    const marker = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, DONE_MARKER });
    defer gpa.free(marker);
    const mf = try Io.Dir.cwd().createFile(io, marker, .{ .truncate = true });
    mf.close(io);
}

/// Download `url` to `dest_path`, following redirects ourselves so every hop
/// can be required to stay on https (security issue #32): `Client.fetch`
/// follows redirects internally and accepts a downgrade to plain http, which
/// would hand an on-path attacker the weights. The body is written to a
/// `.part` file and renamed only on success, so an interrupted transfer never
/// leaves a file that looks complete.
fn downloadFile(gpa: std.mem.Allocator, io: Io, client: *std.http.Client, url: []const u8, dest_path: []const u8) !void {
    const part_path = try std.fmt.allocPrint(gpa, "{s}.part", .{dest_path});
    defer gpa.free(part_path);

    var fbuf: [1 << 16]u8 = undefined;
    const f = try Io.Dir.cwd().createFile(io, part_path, .{ .truncate = true });
    defer f.close(io);
    errdefer Io.Dir.cwd().deleteFile(io, part_path) catch {};
    var fw = f.writer(io, &fbuf);

    // Redirect targets are resolved into this buffer; `uri` may point into it
    // after the first hop, so it must outlive the loop.
    var aux_storage: [16 * 1024]u8 = undefined;
    var aux: []u8 = &aux_storage;
    var uri = std.Uri.parse(url) catch return error.InvalidUrl;

    var hops: usize = 0;
    while (true) : (hops += 1) {
        if (hops > MAX_REDIRECTS) return error.TooManyRedirects;
        // Checked on EVERY hop, which is the point: Client.fetch follows
        // redirects internally and will happily continue over plain http,
        // handing an on-path attacker the weights (security issue #32).
        if (!std.mem.eql(u8, uri.scheme, "https")) return error.InsecureRedirect;

        var req = try client.request(.GET, uri, .{ .redirect_behavior = .unhandled });
        defer req.deinit();
        try req.sendBodiless();
        var redirect_buf: [8192]u8 = undefined;
        var resp = try req.receiveHead(&redirect_buf);

        const status = resp.head.status;
        if (status.class() == .redirect) {
            const loc = resp.head.location orelse return error.BadRedirect;
            if (loc.len > aux.len) return error.RedirectTooLong;
            // copy before touching the body: reading it invalidates head strings
            const copied = aux[0..loc.len];
            @memcpy(copied, loc);
            {
                const body_reader = req.reader.bodyReader(&.{}, resp.head.transfer_encoding, resp.head.content_length);
                _ = body_reader.discardRemaining() catch {};
            }
            // resolves relative Locations (HF redirects to a CDN path) against
            // the current URI, exactly as std's own redirect handling does
            uri = uri.resolveInPlace(loc.len, &aux) catch return error.InvalidUrl;
            continue;
        }
        if (status != .ok) return error.HttpDownloadFailed;
        if (resp.head.content_length) |len| {
            if (len > MAX_FILE_BYTES) return error.DownloadTooLarge;
        }

        var tbuf: [1 << 16]u8 = undefined;
        const body = resp.reader(&tbuf);
        // `stream` moves at most one chunk per call, so loop to EOF. The
        // running total caps the transfer even when the server declares no
        // Content-Length (security issue #32).
        var total: u64 = 0;
        var idle: usize = 0;
        while (true) {
            // NOTE: a zero return is normal here (std's own streamRemaining
            // loops until EndOfStream and never treats 0 as the end); only
            // EndOfStream terminates. The idle counter is just a stuck-peer
            // backstop.
            const n = body.stream(&fw.interface, .limited64(1 << 16)) catch |e| switch (e) {
                error.EndOfStream => break,
                else => return e,
            };
            if (n == 0) {
                idle += 1;
                if (idle > 1000) return error.DownloadStalled;
                continue;
            }
            idle = 0;
            total += n;
            if (total > MAX_FILE_BYTES) return error.DownloadTooLarge;
        }
        try fw.interface.flush();
        break;
    }
    // publish under the real name only once every byte has landed
    try Io.Dir.cwd().rename(part_path, Io.Dir.cwd(), dest_path, io);
}

test "download refuses a non-https URL (redirect-downgrade guard, issue #32)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var client = std.http.Client{ .allocator = gpa, .io = io };
    defer client.deinit();
    try std.testing.expectError(error.InsecureRedirect,
        downloadFile(gpa, io, &client, "http://huggingface.co/x/y", "/tmp/dl-nope.bin"));
}
