//! Disk profile: parallel random block reads, the exact access pattern the
//! engine issues on a miss (colibri's iobench). The per-token working set is
//! `n_moe_layers * n_routed` such reads, so this GB/s is what sets the tier
//! order (CLAUDE: probe local disk bandwidth, don't assume).

const std = @import("std");
const Io = std.Io;

pub const Result = struct {
    threads: usize,
    block_bytes: usize,
    total_reads: usize,
    bytes: u64,
    ns: i128,

    pub fn gbps(self: Result) f64 {
        if (self.ns <= 0) return 0;
        const secs = @as(f64, @floatFromInt(self.ns)) / 1e9;
        return (@as(f64, @floatFromInt(self.bytes)) / 1e9) / secs;
    }
};

const Ctx = struct {
    io: Io,
    path: []const u8,
    block_bytes: usize,
    reads: usize,
    max_offset: u64,
    seed: u64,
    bytes_out: u64 = 0,
    err: ?anyerror = null,
};

fn worker(gpa: std.mem.Allocator, ctx: *Ctx) void {
    runWorker(gpa, ctx) catch |e| {
        ctx.err = e;
    };
}

fn runWorker(gpa: std.mem.Allocator, ctx: *Ctx) !void {
    const f = try Io.Dir.cwd().openFile(ctx.io, ctx.path, .{});
    defer f.close(ctx.io);
    const buf = try gpa.alloc(u8, ctx.block_bytes);
    defer gpa.free(buf);

    var prng = std.Random.DefaultPrng.init(ctx.seed);
    const rnd = prng.random();

    var i: usize = 0;
    while (i < ctx.reads) : (i += 1) {
        const span = ctx.max_offset + 1;
        const offset = if (span == 0) 0 else rnd.uintLessThan(u64, span);
        _ = try f.readPositionalAll(ctx.io, buf, offset);
        ctx.bytes_out += ctx.block_bytes;
    }
}

pub fn run(
    gpa: std.mem.Allocator,
    io: Io,
    path: []const u8,
    threads: usize,
    block_bytes: usize,
    reads_per_thread: usize,
    seed: u64,
) !Result {
    const probe = try Io.Dir.cwd().openFile(io, path, .{});
    const size = (try probe.stat(io)).size;
    probe.close(io);
    if (size < block_bytes) return error.FileTooSmall;
    const max_offset = size - block_bytes;

    const ctxs = try gpa.alloc(Ctx, threads);
    defer gpa.free(ctxs);
    const handles = try gpa.alloc(std.Thread, threads);
    defer gpa.free(handles);

    const t0 = std.Io.Clock.Timestamp.now(io, .awake).raw.toNanoseconds();
    for (ctxs, 0..) |*c, i| {
        c.* = .{
            .io = io,
            .path = path,
            .block_bytes = block_bytes,
            .reads = reads_per_thread,
            .max_offset = max_offset,
            .seed = seed ^ (0x9e3779b97f4a7c15 *% (i + 1)),
        };
        handles[i] = try std.Thread.spawn(.{}, worker, .{ gpa, c });
    }
    for (handles) |h| h.join();
    const t1 = std.Io.Clock.Timestamp.now(io, .awake).raw.toNanoseconds();

    var bytes: u64 = 0;
    for (ctxs) |c| {
        if (c.err) |e| return e;
        bytes += c.bytes_out;
    }

    return .{
        .threads = threads,
        .block_bytes = block_bytes,
        .total_reads = threads * reads_per_thread,
        .bytes = bytes,
        .ns = t1 - t0,
    };
}
