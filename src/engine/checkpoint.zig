//! On-disk checkpoint: a directory holding three files.
//!
//!   manifest.loom  header + ModelConfig + Merkle root + the expert index
//!                  (expert_id -> {offset, len, sha256}) — content-addressed.
//!   dense.blob     the resident set (embeddings, attention, dense FFN, shared
//!                  experts, norms, lm_head) as f32. Loaded fully into RAM.
//!   experts.blob   concatenated int4 routed-expert blocks, deduplicated by
//!                  hash. Streamed on the per-token fetch path via pread.
//!
//! The expert index and Merkle root give principle 5 (content-addressed +
//! Merkle-rooted): a poisoned expert fails its digest check on read for free.

const std = @import("std");
const Io = std.Io;
const model = @import("model.zig");
const hashmod = @import("../core/hash.zig");
const ModelConfig = model.ModelConfig;

pub const MAGIC: [8]u8 = "LOOMCKP1".*;
pub const FORMAT_VERSION: u32 = 1;

pub const ExpertEntry = extern struct {
    offset: u64,
    len: u64,
    digest: hashmod.Digest,
};

pub const Header = extern struct {
    magic: [8]u8,
    format_version: u32,
    _pad: u32 = 0,
    config: ModelConfig,
    merkle_root: hashmod.Digest,
    dense_len: u64, // bytes in dense.blob
    experts_len: u64, // bytes in experts.blob
    n_experts_index: u64, // number of ExpertEntry records that follow the header
};

// ---- dense weight layout ---------------------------------------------------

pub const LayerWeights = struct {
    input_norm: []f32,
    post_attn_norm: []f32,
    q_a_proj: []f32,
    q_a_norm: []f32,
    q_b_proj: []f32,
    kv_a_proj: []f32,
    kv_a_norm: []f32,
    kv_b_proj: []f32,
    o_proj: []f32,
    // dense-layer FFN (empty on MoE layers)
    ffn_gate: []f32,
    ffn_up: []f32,
    ffn_down: []f32,
    // MoE bits (empty on dense layers)
    router: []f32,
    shared_gate: []f32,
    shared_up: []f32,
    shared_down: []f32,
    is_moe: bool,
};

pub const Weights = struct {
    token_embedding: []f32,
    layers: []LayerWeights,
    final_norm: []f32,
    lm_head: []f32,
    denseEnd: usize, // total f32 elements carved (valid even in count mode)
};

const Carver = struct {
    buf: []f32, // empty in count mode
    pos: usize = 0,

    fn take(self: *Carver, n: usize) []f32 {
        const start = self.pos;
        self.pos += n;
        if (self.buf.len == 0) return &.{};
        return self.buf[start .. start + n];
    }
};

/// Carve dense weights out of `blob` (f32), filling `layers_out` (length ==
/// cfg.nLayers()). Pass an empty `blob` and empty `layers_out` to only count:
/// the returned Weights slices are empty but `denseElemCount` reads the total.
/// The generator and loader both call this, so their layouts are identical.
pub fn carveWeights(cfg: ModelConfig, blob: []f32, layers_out: []LayerWeights) Weights {
    var c = Carver{ .buf = blob };
    const h: usize = cfg.hidden;

    const token_embedding = c.take(cfg.vocab_size * h);

    const n_layers = cfg.nLayers();
    var li: usize = 0;
    while (li < n_layers) : (li += 1) {
        const is_moe = li >= cfg.n_dense_layers;
        const lw = carveLayer(cfg, &c, is_moe);
        if (layers_out.len != 0) layers_out[li] = lw;
    }

    const final_norm = c.take(h);
    const lm_head = c.take(cfg.vocab_size * h);

    return .{
        .token_embedding = token_embedding,
        .layers = if (layers_out.len != 0) layers_out[0..n_layers] else &.{},
        .final_norm = final_norm,
        .lm_head = lm_head,
        .denseEnd = c.pos,
    };
}

fn carveLayer(cfg: ModelConfig, c: *Carver, is_moe: bool) LayerWeights {
    const h: usize = cfg.hidden;
    const head_all = cfg.n_heads * cfg.headDim();
    const kv_a_out = cfg.kv_lora_rank + cfg.rope_dim;
    const kv_b_out = cfg.n_heads * (cfg.nope_dim + cfg.v_head_dim);

    var lw: LayerWeights = undefined;
    lw.is_moe = is_moe;
    lw.input_norm = c.take(h);
    lw.post_attn_norm = c.take(h);
    lw.q_a_proj = c.take(cfg.q_lora_rank * h);
    lw.q_a_norm = c.take(cfg.q_lora_rank);
    lw.q_b_proj = c.take(head_all * cfg.q_lora_rank);
    lw.kv_a_proj = c.take(kv_a_out * h);
    lw.kv_a_norm = c.take(cfg.kv_lora_rank);
    lw.kv_b_proj = c.take(kv_b_out * cfg.kv_lora_rank);
    lw.o_proj = c.take(h * cfg.n_heads * cfg.v_head_dim);

    if (!is_moe) {
        lw.ffn_gate = c.take(cfg.dense_ffn * h);
        lw.ffn_up = c.take(cfg.dense_ffn * h);
        lw.ffn_down = c.take(h * cfg.dense_ffn);
        lw.router = &.{};
        lw.shared_gate = &.{};
        lw.shared_up = &.{};
        lw.shared_down = &.{};
    } else {
        lw.ffn_gate = &.{};
        lw.ffn_up = &.{};
        lw.ffn_down = &.{};
        lw.router = c.take(cfg.n_experts * h);
        lw.shared_gate = c.take(cfg.n_shared * cfg.moe_ffn * h);
        lw.shared_up = c.take(cfg.n_shared * cfg.moe_ffn * h);
        lw.shared_down = c.take(cfg.n_shared * h * cfg.moe_ffn);
    }
    return lw;
}

/// Number of f32 elements in the dense blob for `cfg`.
pub fn denseElemCount(cfg: ModelConfig) usize {
    var empty: [0]LayerWeights = .{};
    const w = carveWeights(cfg, &.{}, &empty);
    return w.denseEnd;
}

// ---- expert addressing -----------------------------------------------------

/// Stable expert id: layer index within the MoE layers (0..n_moe_layers) times
/// n_experts, plus the expert index. Dense layers have no experts.
pub fn expertId(cfg: ModelConfig, moe_layer: usize, expert: usize) usize {
    return moe_layer * cfg.n_experts + expert;
}

// ---- writing ---------------------------------------------------------------

fn joinPath(buf: []u8, dir_path: []const u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir_path, name });
}

fn writeFile(io: Io, dir_path: []const u8, name: []const u8, bytes: []const u8) !void {
    var pbuf: [4096]u8 = undefined;
    const path = try joinPath(&pbuf, dir_path, name);
    const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
}

/// Write a checkpoint directory. `dense` is the resident f32 blob laid out by
/// `carveWeights`; `experts_flat` is `totalRoutedExperts * expertBytes(cfg)`
/// bytes of contiguous int4 expert blocks. Identical experts are deduplicated
/// by SHA-256 (content addressing) and the manifest is Merkle-rooted.
pub fn writeCheckpoint(
    gpa: std.mem.Allocator,
    io: Io,
    dir_path: []const u8,
    cfg: ModelConfig,
    dense: []const f32,
    experts_flat: []const u8,
) !void {
    const eb = cfg.expertBytes();
    const total = cfg.totalRoutedExperts();
    std.debug.assert(experts_flat.len == total * eb);

    Io.Dir.cwd().createDir(io, dir_path, .default_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };

    var entries = try gpa.alloc(ExpertEntry, total);
    defer gpa.free(entries);
    var leaves = try gpa.alloc(hashmod.Digest, total);
    defer gpa.free(leaves);

    var blob = std.ArrayList(u8).empty;
    defer blob.deinit(gpa);
    var dedup = std.AutoHashMap(hashmod.Digest, u64).init(gpa);
    defer dedup.deinit();

    var id: usize = 0;
    while (id < total) : (id += 1) {
        const block = experts_flat[id * eb ..][0..eb];
        const digest = hashmod.hashBlock(block);
        leaves[id] = digest;
        const offset = if (dedup.get(digest)) |off| off else blk: {
            const off: u64 = blob.items.len;
            try blob.appendSlice(gpa, block);
            try dedup.put(digest, off);
            break :blk off;
        };
        entries[id] = .{ .offset = offset, .len = eb, .digest = digest };
    }

    const root = try hashmod.merkleRoot(gpa, leaves);

    try writeFile(io, dir_path, "dense.blob", std.mem.sliceAsBytes(dense));
    try writeFile(io, dir_path, "experts.blob", blob.items);

    const header = Header{
        .magic = MAGIC,
        .format_version = FORMAT_VERSION,
        .config = cfg,
        .merkle_root = root,
        .dense_len = std.mem.sliceAsBytes(dense).len,
        .experts_len = blob.items.len,
        .n_experts_index = total,
    };

    var manifest = std.ArrayList(u8).empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, std.mem.asBytes(&header));
    try manifest.appendSlice(gpa, std.mem.sliceAsBytes(entries));
    try writeFile(io, dir_path, "manifest.loom", manifest.items);
}

// ---- loading ---------------------------------------------------------------

pub const Loaded = struct {
    gpa: std.mem.Allocator,
    io: Io,
    header: Header,
    dense: []f32, // owned; resident set
    layers: []LayerWeights, // owned; slices into dense
    weights: Weights,
    entries: []ExpertEntry, // owned; expert index
    experts_file: Io.File, // open for streaming reads

    pub fn deinit(self: *Loaded) void {
        self.experts_file.close(self.io);
        self.gpa.free(self.dense);
        self.gpa.free(self.layers);
        self.gpa.free(self.entries);
    }
};

pub fn load(gpa: std.mem.Allocator, io: Io, dir_path: []const u8) !Loaded {
    var pbuf: [4096]u8 = undefined;

    // manifest
    const mpath = try joinPath(&pbuf, dir_path, "manifest.loom");
    const mf = try Io.Dir.cwd().openFile(io, mpath, .{});
    const msize = (try mf.stat(io)).size;
    const mbytes = try gpa.alloc(u8, msize);
    defer gpa.free(mbytes);
    _ = try mf.readPositionalAll(io, mbytes, 0);
    mf.close(io);

    if (mbytes.len < @sizeOf(Header)) return error.TruncatedManifest;
    var header: Header = undefined;
    @memcpy(std.mem.asBytes(&header), mbytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &header.magic, &MAGIC)) return error.BadMagic;
    if (header.format_version != FORMAT_VERSION) return error.BadVersion;
    try header.config.validate();

    const n = header.n_experts_index;
    const entries = try gpa.alloc(ExpertEntry, n);
    errdefer gpa.free(entries);
    const entry_bytes = mbytes[@sizeOf(Header)..][0 .. n * @sizeOf(ExpertEntry)];
    @memcpy(std.mem.sliceAsBytes(entries), entry_bytes);

    // dense blob -> resident RAM
    const expect_dense = denseElemCount(header.config);
    if (header.dense_len != expect_dense * @sizeOf(f32)) return error.DenseSizeMismatch;
    const dense = try gpa.alloc(f32, expect_dense);
    errdefer gpa.free(dense);
    const dpath = try joinPath(&pbuf, dir_path, "dense.blob");
    const df = try Io.Dir.cwd().openFile(io, dpath, .{});
    _ = try df.readPositionalAll(io, std.mem.sliceAsBytes(dense), 0);
    df.close(io);

    const layers = try gpa.alloc(LayerWeights, header.config.nLayers());
    errdefer gpa.free(layers);
    const weights = carveWeights(header.config, dense, layers);

    const epath = try joinPath(&pbuf, dir_path, "experts.blob");
    const experts_file = try Io.Dir.cwd().openFile(io, epath, .{});

    return .{
        .gpa = gpa,
        .io = io,
        .header = header,
        .dense = dense,
        .layers = layers,
        .weights = weights,
        .entries = entries,
        .experts_file = experts_file,
    };
}

test "dense layout counting matches a materialized carve" {
    const gpa = std.testing.allocator;
    const cfg = model.tinyShape();
    const n = denseElemCount(cfg);
    try std.testing.expect(n > 0);

    const blob = try gpa.alloc(f32, n);
    defer gpa.free(blob);
    const layers = try gpa.alloc(LayerWeights, cfg.nLayers());
    defer gpa.free(layers);

    const w = carveWeights(cfg, blob, layers);
    try std.testing.expect(w.token_embedding.len == cfg.vocab_size * cfg.hidden);
    try std.testing.expect(w.layers.len == cfg.nLayers());
    try std.testing.expect(!w.layers[0].is_moe); // first layers are dense
    try std.testing.expect(w.layers[cfg.nLayers() - 1].is_moe); // last is MoE
    // the last carved slice must end exactly at n (no over/under-count)
    try std.testing.expect(w.lm_head.ptr + w.lm_head.len == blob.ptr + n);
}
