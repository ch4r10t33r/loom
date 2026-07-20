//! Deterministic synthetic checkpoint generator.
//!
//! v0's acceptance ("token-exact vs a transformers oracle") needs the real GLM
//! weights + tokenizer, which aren't shipped here. This generator produces a
//! valid, self-consistent checkpoint in the same on-disk format so the whole
//! engine — streaming cache, int4 kernels, MLA, routing, STATS, Merkle check —
//! runs end-to-end on commodity hardware. Swap in a real converted checkpoint
//! and nothing in the engine changes.

const std = @import("std");
const Io = std.Io;
const model = @import("model.zig");
const ckpt = @import("checkpoint.zig");
const quant = @import("../core/quant.zig");
const ModelConfig = model.ModelConfig;

fn fillSmall(rnd: std.Random, dst: []f32, scale: f32) void {
    for (dst) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0 * scale;
}

fn quantizeMatrix(gpa: std.mem.Allocator, dst: []u8, rnd: std.Random, rows: usize, cols: usize, scale: f32) !void {
    const rowf = try gpa.alloc(f32, cols);
    defer gpa.free(rowf);
    const rb = quant.rowBytes(cols);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        fillSmall(rnd, rowf, scale);
        quant.quantizeRow(dst[r * rb ..][0..rb], rowf);
    }
}

pub fn generate(gpa: std.mem.Allocator, io: Io, dir_path: []const u8, cfg: ModelConfig, seed: u64) !void {
    try cfg.validate();

    // dense resident set
    const dense = try gpa.alloc(f32, ckpt.denseElemCount(cfg));
    defer gpa.free(dense);
    const layers = try gpa.alloc(ckpt.LayerWeights, cfg.nLayers());
    defer gpa.free(layers);
    const w = ckpt.carveWeights(cfg, dense, layers);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    fillSmall(rnd, w.token_embedding, 0.04);
    fillSmall(rnd, w.lm_head, 0.04);
    @memset(w.final_norm, 1.0);
    for (w.layers) |lw| {
        @memset(lw.input_norm, 1.0);
        @memset(lw.post_attn_norm, 1.0);
        @memset(lw.q_a_norm, 1.0);
        @memset(lw.kv_a_norm, 1.0);
        fillSmall(rnd, lw.q_a_proj, 0.04);
        fillSmall(rnd, lw.q_b_proj, 0.04);
        fillSmall(rnd, lw.kv_a_proj, 0.04);
        fillSmall(rnd, lw.kv_b_proj, 0.04);
        fillSmall(rnd, lw.o_proj, 0.04);
        if (!lw.is_moe) {
            fillSmall(rnd, lw.ffn_gate, 0.04);
            fillSmall(rnd, lw.ffn_up, 0.04);
            fillSmall(rnd, lw.ffn_down, 0.04);
        } else {
            fillSmall(rnd, lw.router, 0.08);
            fillSmall(rnd, lw.shared_gate, 0.04);
            fillSmall(rnd, lw.shared_up, 0.04);
            fillSmall(rnd, lw.shared_down, 0.04);
        }
    }

    // routed experts (int4), one distinct block per (layer, expert)
    const eb = cfg.expertBytes();
    const total = cfg.totalRoutedExperts();
    const experts_flat = try gpa.alloc(u8, total * eb);
    defer gpa.free(experts_flat);

    const h = cfg.hidden;
    const f = cfg.moe_ffn;
    const gate_bytes = ModelConfig.q4MatrixBytes(f, h);
    const up_bytes = ModelConfig.q4MatrixBytes(f, h);

    var id: usize = 0;
    while (id < total) : (id += 1) {
        // seed per expert so blocks are distinct (dedup won't collapse them)
        var ep = std.Random.DefaultPrng.init(seed ^ (0x9e3779b97f4a7c15 *% (id + 1)));
        const er = ep.random();
        const block = experts_flat[id * eb ..][0..eb];
        try quantizeMatrix(gpa, block[0..gate_bytes], er, f, h, 0.05);
        try quantizeMatrix(gpa, block[gate_bytes .. gate_bytes + up_bytes], er, f, h, 0.05);
        try quantizeMatrix(gpa, block[gate_bytes + up_bytes ..], er, h, f, 0.05);
    }

    try ckpt.writeCheckpoint(gpa, io, dir_path, cfg, dense, experts_flat);
}
