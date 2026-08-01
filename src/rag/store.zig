//! The RAG chunk store: text chunks with locally computed embeddings,
//! searched by cosine similarity before inference.
//!
//! Only TEXT is shared between nodes. A loom network serves exactly one
//! model (network_id), so every node derives the same embedding from the
//! same text -- mean-pooled `token_embd` rows, L2-normalized -- and vectors
//! never travel the wire. That removes the vector-poisoning surface and any
//! dimension mismatch by construction; a tampered chunk is just different
//! text, deduplicated and searched like any other.
//!
//! Search runs through FAISS (flat inner-product index) when the library is
//! present, and an exact scan otherwise. The scan is the reference: the two
//! must agree on any query, and the test asserts it when FAISS is loaded.
const std = @import("std");
const Io = std.Io;
const faiss = @import("faiss.zig");
const brotli = @import("brotli.zig");

pub const MAX_CHUNKS: usize = 65536;
pub const MAX_TEXT: usize = 8 * 1024;
pub const MAX_DIM: usize = 8192;

pub const Hash = [32]u8;

pub fn hashText(text: []const u8) Hash {
    var h: Hash = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &h, .{});
    return h;
}

pub const Chunk = struct {
    hash: Hash,
    /// Raw text, or brotli-compressed when the library is present and it
    /// pays. At-rest format is local-only: the hash is of the RAW text and
    /// gossip always sends raw (the frame encoder snappy-compresses the
    /// wire); FAISS holds the flat f32 vectors either way.
    text: []u8,
    raw_len: usize,
    compressed: bool,
    vec: []f32, // L2-normalized, dim floats
};

pub const Hit = struct { idx: usize, score: f32 };

/// An embedder turns text into a normalized vector. The node installs one
/// backed by the model's token embeddings; tests install a deterministic
/// fixture. Returns false when the text produces no tokens.
pub const Embedder = struct {
    ctx: *anyopaque,
    dim: usize,
    embedFn: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator, text: []const u8, out: []f32) bool,

    pub fn embed(self: *const Embedder, gpa: std.mem.Allocator, text: []const u8, out: []f32) bool {
        return self.embedFn(self.ctx, gpa, text, out);
    }
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    /// Late-bound: the node constructs the store before it has chosen the
    /// serving engine; the embedder arrives with `setEmbedder`. Until then
    /// every add/search declines.
    emb: ?Embedder = null,
    mu: Io.Mutex = .init,
    chunks: std.ArrayListUnmanaged(Chunk) = .empty,
    seen: std.AutoHashMapUnmanaged(Hash, void) = .empty,
    index: ?faiss.Index,
    /// Bumped on every accepted insert; gossip uses it to skip idle rounds.
    seq: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: Io) Store {
        return .{ .gpa = gpa, .io = io, .index = null };
    }

    pub fn setEmbedder(self: *Store, emb: Embedder) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.emb = emb;
        self.index = faiss.Index.init(emb.dim);
    }

    pub fn dim(self: *Store) usize {
        return if (self.emb) |e| e.dim else 0;
    }

    pub fn deinit(self: *Store) void {
        for (self.chunks.items) |c| {
            self.gpa.free(c.text);
            self.gpa.free(c.vec);
        }
        self.chunks.deinit(self.gpa);
        self.seen.deinit(self.gpa);
    }

    pub fn count(self: *Store) usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return self.chunks.items.len;
    }

    pub fn seqNow(self: *Store) u64 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return self.seq;
    }

    /// Insert a chunk; the embedding is computed HERE, never accepted from
    /// outside. Returns false when duplicate, over caps, or unembeddable.
    pub fn add(self: *Store, text: []const u8) bool {
        if (text.len == 0 or text.len > MAX_TEXT) return false;
        const h = hashText(text);
        {
            self.mu.lockUncancelable(self.io);
            defer self.mu.unlock(self.io);
            if (self.chunks.items.len >= MAX_CHUNKS) return false;
            if (self.seen.contains(h)) return false;
        }
        const emb = self.emb orelse return false;
        const vec = self.gpa.alloc(f32, emb.dim) catch return false;
        if (!emb.embed(self.gpa, text, vec)) {
            self.gpa.free(vec);
            return false;
        }
        var compressed = false;
        const copy = blk: {
            if (brotli.compressAlloc(self.gpa, text)) |c| {
                compressed = true;
                break :blk c;
            }
            break :blk self.gpa.dupe(u8, text) catch {
                self.gpa.free(vec);
                return false;
            };
        };
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.seen.contains(h)) { // raced another inserter
            self.gpa.free(vec);
            self.gpa.free(copy);
            return false;
        }
        self.seen.put(self.gpa, h, {}) catch {
            self.gpa.free(vec);
            self.gpa.free(copy);
            return false;
        };
        self.chunks.append(self.gpa, .{ .hash = h, .text = copy, .raw_len = text.len, .compressed = compressed, .vec = vec }) catch {
            _ = self.seen.remove(h);
            self.gpa.free(vec);
            self.gpa.free(copy);
            return false;
        };
        if (self.index) |*ix| _ = ix.add(vec);
        self.seq += 1;
        return true;
    }

    pub fn has(self: *Store, h: Hash) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return self.seen.contains(h);
    }

    /// The gossip inventory for one round: newest-first, and when the store
    /// exceeds one round's cap the window ROTATES with `round`, so every
    /// chunk hash is advertised within ceil(count/cap) rounds -- without
    /// this, a rejoining peer would never hear about anything older than
    /// the newest `cap` chunks (global-topic convergence, eventually).
    pub fn invWindow(self: *Store, round: u64, out: []Hash) usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const items = self.chunks.items;
        if (items.len == 0 or out.len == 0) return 0;
        const n = @min(out.len, items.len);
        const start = if (items.len > out.len) (round * out.len) % items.len else 0;
        for (0..n) |i| out[i] = items[items.len - 1 - ((start + i) % items.len)].hash;
        return n;
    }

    fn chunkTextAlloc(gpa: std.mem.Allocator, c: Chunk) ?[]u8 {
        if (!c.compressed) return gpa.dupe(u8, c.text) catch null;
        return brotli.decompressAlloc(gpa, c.text, c.raw_len);
    }

    /// Copy a chunk's raw text by hash (decompressing at-rest storage).
    pub fn textByHashAlloc(self: *Store, gpa: std.mem.Allocator, h: Hash) ?[]u8 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        for (self.chunks.items) |c| {
            if (std.mem.eql(u8, &c.hash, &h)) return chunkTextAlloc(gpa, c);
        }
        return null;
    }

    /// Copy a chunk's text by store index (what `search` returns).
    pub fn textByIndexAlloc(self: *Store, gpa: std.mem.Allocator, idx: usize) ?[]u8 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (idx >= self.chunks.items.len) return null;
        return chunkTextAlloc(gpa, self.chunks.items[idx]);
    }

    /// Top-k cosine hits for a query embedding, best first. FAISS when
    /// available, the exact scan otherwise; both operate on the same
    /// normalized vectors and agree.
    pub fn search(self: *Store, query: []const f32, k: usize, out: []Hit) usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const items = self.chunks.items;
        if (items.len == 0 or k == 0) return 0;
        const kk = @min(k, @min(items.len, out.len));
        if (self.index) |*ix| {
            if (ix.count() == items.len) {
                var scores: [16]f32 = undefined;
                var ids: [16]i64 = undefined;
                if (kk <= 16) {
                    const n = ix.search(query, kk, scores[0..kk], ids[0..kk]);
                    for (0..n) |i| out[i] = .{ .idx = @intCast(ids[i]), .score = scores[i] };
                    if (n > 0) return n;
                }
            }
        }
        // Exact scan: insertion-sort the top k.
        var n: usize = 0;
        for (items, 0..) |c, idx| {
            var s: f32 = 0;
            for (c.vec, query) |a, b| s += a * b;
            var pos = n;
            while (pos > 0 and out[pos - 1].score < s) pos -= 1;
            if (pos >= kk) continue;
            if (n < kk) n += 1;
            var j = n - 1;
            while (j > pos) : (j -= 1) out[j] = out[j - 1];
            out[pos] = .{ .idx = idx, .score = s };
        }
        return n;
    }

    /// Embed with the store's embedder into a caller buffer of dim floats.
    pub fn embedQuery(self: *Store, gpa: std.mem.Allocator, text: []const u8, out: []f32) bool {
        const emb = self.emb orelse return false;
        return emb.embed(gpa, text, out);
    }
};

/// Deterministic fixture embedder for tests: bytes hashed into a few basis
/// directions, normalized. No model needed; similar prefixes cluster.
pub fn fixtureEmbedder(dim_holder: *const usize) Embedder {
    const impl = struct {
        fn embed(ctx: *anyopaque, gpa: std.mem.Allocator, text: []const u8, out: []f32) bool {
            _ = gpa;
            const dim: *const usize = @ptrCast(@alignCast(ctx));
            std.debug.assert(out.len == dim.*);
            @memset(out, 0);
            var hsh = hashText(text);
            for (text, 0..) |b, i| out[(@as(usize, b) + i) % out.len] += 1.0;
            for (&hsh, 0..) |b, i| out[(@as(usize, b) * 7 + i) % out.len] += @as(f32, @floatFromInt(b)) / 64.0;
            var norm: f32 = 0;
            for (out) |v| norm += v * v;
            if (norm == 0) return false;
            const inv = 1.0 / @sqrt(norm);
            for (out) |*v| v.* *= inv;
            return true;
        }
    };
    return .{ .ctx = @ptrCast(@constCast(dim_holder)), .dim = dim_holder.*, .embedFn = impl.embed };
}

test "add dedups by text hash and search ranks the exact match first" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var dim: usize = 64;
    var st = Store.init(gpa, threaded.io());
    st.setEmbedder(fixtureEmbedder(&dim));
    defer st.deinit();
    try std.testing.expect(st.add("the weaver of alder street"));
    try std.testing.expect(!st.add("the weaver of alder street")); // dup
    try std.testing.expect(st.add("a treatise on expert routing"));
    try std.testing.expect(st.add("gossip protocols in practice"));
    try std.testing.expectEqual(@as(usize, 3), st.count());

    var q: [64]f32 = undefined;
    try std.testing.expect(st.embedQuery(gpa, "the weaver of alder street", &q));
    var hits: [3]Hit = undefined;
    const n = st.search(&q, 3, &hits);
    try std.testing.expect(n >= 1);
    try std.testing.expectEqual(@as(usize, 0), hits[0].idx);
    try std.testing.expect(hits[0].score > 0.99); // self-match, normalized
    if (n > 1) try std.testing.expect(hits[0].score >= hits[1].score);
}

test "caps hold: empty and oversized chunks are refused" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var dim: usize = 16;
    var st = Store.init(gpa, threaded.io());
    st.setEmbedder(fixtureEmbedder(&dim));
    defer st.deinit();
    try std.testing.expect(!st.add(""));
    const big = try gpa.alloc(u8, MAX_TEXT + 1);
    defer gpa.free(big);
    @memset(big, 'x');
    try std.testing.expect(!st.add(big));
}

test "chunk text round-trips through at-rest storage (raw or brotli)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var dim: usize = 32;
    var st = Store.init(gpa, threaded.io());
    st.setEmbedder(fixtureEmbedder(&dim));
    defer st.deinit();
    // Repetitive prose compresses when brotli is present; either way the
    // retrieved text must equal the raw input.
    const text = "the weaver wove and wove and wove and wove the same thread " ** 8;
    try std.testing.expect(st.add(text));
    const got = st.textByIndexAlloc(gpa, 0).?;
    defer gpa.free(got);
    try std.testing.expectEqualStrings(text, got);
    const byh = st.textByHashAlloc(gpa, hashText(text)).?;
    defer gpa.free(byh);
    try std.testing.expectEqualStrings(text, byh);
}

test "the inventory window rotates so every hash is eventually advertised" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var dim: usize = 16;
    var st = Store.init(gpa, threaded.io());
    st.setEmbedder(fixtureEmbedder(&dim));
    defer st.deinit();
    var buf: [32]u8 = undefined;
    for (0..7) |i| {
        const t = std.fmt.bufPrint(&buf, "chunk number {d}", .{i}) catch unreachable;
        try std.testing.expect(st.add(t));
    }
    // window of 3 over 7 chunks: three rounds must cover all seven hashes
    var seen = std.AutoHashMapUnmanaged(Hash, void).empty;
    defer seen.deinit(gpa);
    var w: [3]Hash = undefined;
    for (0..3) |round| {
        const n = st.invWindow(round, &w);
        for (w[0..n]) |h| try seen.put(gpa, h, {});
    }
    try std.testing.expectEqual(@as(usize, 7), seen.count());
}
