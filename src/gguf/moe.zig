//! Mixture-of-experts routing, shared by every MoE architecture loom runs.
//!
//! Extracted from deepseek.zig once it became clear the routing was never
//! DeepSeek-specific: llama.cpp funnels Mixtral, Qwen2/3-MoE, GLM-4.5-MoE and
//! DeepSeek-V2/V3 through one `build_moe_ffn`, and the only things that vary
//! are four knobs — the gating function, whether a selection bias is added,
//! whether the top-k gates are renormalized, and a constant scale. `route`
//! below implements exactly that function, so a new architecture needs an
//! attention variant, not a new router.
//!
//! Per-architecture settings (verified against llama.cpp's model graphs):
//!
//!   arch        gating   norm_w  exp_probs_b  shared expert
//!   llama       softmax  yes     no           no            (Mixtral)
//!   qwen2moe    softmax  **no**  no           yes, sigmoid-gated
//!   qwen3moe    softmax  yes     no           no
//!   glm4moe     metadata metadata yes         yes, added plainly
//!   deepseek2   metadata metadata yes         yes, added plainly
//!
//! "metadata" means the file carries `<arch>.expert_gating_func` /
//! `expert_weights_norm`, so those two follow the checkpoint rather than a
//! hardcoded belief about the architecture.
//!
//! This module also owns the binding from a routed expert to a distribution
//! shard id (`buildExpertShardMap`), which is what makes an expert fetchable
//! from a peer mid-token.

const std = @import("std");
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const backend = @import("../compute/backend.zig");
const tensor = @import("../core/tensor.zig");
const weights = @import("../p2p/weights.zig");

pub const GatingFunc = enum(u32) {
    softmax = 1,
    sigmoid = 2,
    /// gpt-oss: select top-k on the raw (biased) logits, then softmax over
    /// only the selected k. The router bias is folded into the logits by the
    /// caller, so route() itself needs no bias here.
    softmax_topk = 3,
};

/// Caps on config values that index fixed-size buffers here and in the
/// engines' `step` (security issue #29). Everything routing touches comes from
/// an untrusted file, so an oversized `expert_used_count` must be rejected at
/// load rather than smashing a stack buffer mid-token.
pub const MAX_SELECTED: usize = 64;
pub const MAX_EXPERTS: usize = 1024;

pub const Selected = struct { expert: usize, gate: f32 };

pub const RouteCfg = struct {
    n_expert: usize,
    n_used: usize,
    gating: GatingFunc,
    weights_norm: bool,
    weights_scale: f32,
};

/// Read the gating knobs a file declares, falling back to the architecture's
/// llama.cpp defaults for the arches that hardcode them.
pub fn routeCfgFromMeta(
    parsed: *const gguf.Parsed,
    arch: []const u8,
    n_expert: usize,
    n_used: usize,
) RouteCfg {
    var kb: [128]u8 = undefined;
    const key = struct {
        fn f(buf: []u8, a: []const u8, comptime s: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}." ++ s, .{a}) catch unreachable;
        }
    };
    // qwen2moe is the one arch that does *not* renormalize the top-k gates;
    // gpt-oss softmaxes over the selected k instead of the full set.
    const default_norm = !std.mem.eql(u8, arch, "qwen2moe") and !std.mem.eql(u8, arch, "gpt-oss");
    if (std.mem.eql(u8, arch, "gpt-oss")) {
        return .{
            .n_expert = n_expert,
            .n_used = n_used,
            .gating = .softmax_topk,
            .weights_norm = false,
            .weights_scale = 1.0,
        };
    }
    return .{
        .n_expert = n_expert,
        .n_used = n_used,
        .gating = switch (parsed.getUint(key.f(&kb, arch, "expert_gating_func")) orelse 1) {
            2 => .sigmoid,
            else => .softmax,
        },
        .weights_norm = parsed.getBool(key.f(&kb, arch, "expert_weights_norm")) orelse default_norm,
        .weights_scale = @floatCast(parsed.getFloat(key.f(&kb, arch, "expert_weights_scale")) orelse 1.0),
    };
}

/// Score the router logits, pick the top `n_used` experts, and return their
/// gate weights.
///
/// `bias` (DeepSeek-V3's noaux_tc trick, also used by GLM-4.5) shifts the
/// *selection* scores only — the returned gates come from the unbiased
/// probabilities. Getting that backwards is silent: the model still runs and
/// still produces text, just worse.
pub fn route(cfg: RouteCfg, router_logits: []const f32, bias: ?[]const f32, sel: []Selected) void {
    std.debug.assert(cfg.n_expert <= MAX_EXPERTS and sel.len <= MAX_SELECTED);
    var scores_buf: [MAX_EXPERTS]f32 = undefined;
    const scores = scores_buf[0..cfg.n_expert];
    switch (cfg.gating) {
        .sigmoid => for (router_logits, 0..) |l, i| {
            scores[i] = backend.sigmoid(l);
        },
        .softmax => {
            @memcpy(scores, router_logits);
            backend.softmax(scores);
        },
        .softmax_topk => @memcpy(scores, router_logits),
    }
    var choice_buf: [MAX_EXPERTS]f32 = undefined;
    const choice = choice_buf[0..cfg.n_expert];
    @memcpy(choice, scores);
    if (bias) |b| for (choice, b) |*c, bv| {
        c.* += bv;
    };

    var used = [_]bool{false} ** MAX_EXPERTS;
    for (sel) |*s| {
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (choice, 0..) |c, e| {
            if (!used[e] and c > best_v) {
                best_v = c;
                best = e;
            }
        }
        used[best] = true;
        s.* = .{ .expert = best, .gate = scores[best] };
    }
    if (cfg.gating == .softmax_topk) {
        var sel_buf: [MAX_SELECTED]f32 = undefined;
        for (sel, 0..) |sv, i| sel_buf[i] = sv.gate;
        backend.softmax(sel_buf[0..sel.len]);
        for (sel, 0..) |*sv, i| sv.gate = sel_buf[i];
    }
    if (cfg.weights_norm) {
        var sum: f32 = 0;
        for (sel) |s| sum += s.gate;
        // llama.cpp clamps the divisor to the smallest normal f16 rather than
        // testing for zero, so an all-but-zero gate vector scales up the same
        // way it does there instead of being left unnormalized.
        const denom = @max(sum, 6.103515625e-5);
        for (sel) |*s| s.gate /= denom;
    }
    if (cfg.weights_scale != 1.0) {
        for (sel) |*s| s.gate *= cfg.weights_scale;
    }
}

/// Map `(layer, expert)` to the manifest shard that holds it.
///
/// The link is the *file offset*: an expert shard's first extent starts at the
/// same byte as expert `e` of that layer's `ffn_gate_exps` tensor, which is how
/// the expert-aligned manifest was built (p2p/weights.zig). Matching on offset
/// rather than on a name or an index means a manifest built by a different
/// build, or for a re-ordered file, cannot silently map to the wrong weights —
/// it fails with ShardMapMismatch.
///
/// `is_moe[li]` selects the layers that have expert tensors at all; dense
/// leading layers get no entries. Caller owns the returned slice.
pub fn buildExpertShardMap(
    gpa: std.mem.Allocator,
    parsed: *const gguf.Parsed,
    mani: *const weights.Manifest,
    n_layers: usize,
    n_expert: usize,
    is_moe: []const bool,
) ![]usize {
    if (mani.mode != .expert) return error.NotExpertManifest;

    var by_off = std.AutoHashMap(u64, usize).init(gpa);
    defer by_off.deinit();
    var i: usize = mani.n_resident;
    while (i < mani.nRanges()) : (i += 1) {
        try by_off.put(mani.shardExtents(i)[0].offset, i);
    }

    const map = try gpa.alloc(usize, n_layers * n_expert);
    errdefer gpa.free(map);
    @memset(map, std.math.maxInt(usize));

    var nb: [128]u8 = undefined;
    for (0..n_layers) |li| {
        if (!is_moe[li]) continue;
        const name = try std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_exps.weight", .{li});
        const t = parsed.findTensor(name) orelse return error.MissingTensor;
        const ty: ggml.Type = @enumFromInt(t.ggml_type);
        const per: u64 = @intCast(t.dims[1] * ggml.rowBytes(ty, @intCast(t.dims[0])));
        var e: usize = 0;
        while (e < n_expert) : (e += 1) {
            const off = parsed.data_offset + t.offset + per * e;
            map[li * n_expert + e] = by_off.get(off) orelse return error.ShardMapMismatch;
        }
    }
    return map;
}

// ---- tests -------------------------------------------------------------------

test "sigmoid gating with selection bias picks by biased score, gates from raw" {
    const cfg = RouteCfg{ .n_expert = 4, .n_used = 2, .gating = .sigmoid, .weights_norm = false, .weights_scale = 1.0 };
    const logits = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    // Without a bias, experts 3 and 2 win. The bias below lifts expert 0 above
    // both, but must not change the gate value it is given.
    const bias = [_]f32{ 10.0, 0.0, 0.0, 0.0 };
    var sel: [2]Selected = undefined;
    route(cfg, &logits, &bias, &sel);
    try std.testing.expectEqual(@as(usize, 0), sel[0].expert);
    try std.testing.expectEqual(@as(usize, 3), sel[1].expert);
    try std.testing.expectApproxEqAbs(backend.sigmoid(0.0), sel[0].gate, 1e-6);
    try std.testing.expectApproxEqAbs(backend.sigmoid(3.0), sel[1].gate, 1e-6);
}

test "renormalization makes the top-k gates sum to one" {
    // Mixtral and Qwen3 renormalize; Qwen2-MoE does not. The difference is a
    // silent quality change, so pin both directions.
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var sel: [2]Selected = undefined;

    var cfg = RouteCfg{ .n_expert = 4, .n_used = 2, .gating = .softmax, .weights_norm = true, .weights_scale = 1.0 };
    route(cfg, &logits, null, &sel);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sel[0].gate + sel[1].gate, 1e-6);

    cfg.weights_norm = false;
    route(cfg, &logits, null, &sel);
    try std.testing.expect(sel[0].gate + sel[1].gate < 0.95); // raw softmax mass of the top 2
}

test "weights_scale multiplies after normalization" {
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var sel: [2]Selected = undefined;
    const cfg = RouteCfg{ .n_expert = 4, .n_used = 2, .gating = .softmax, .weights_norm = true, .weights_scale = 2.5 };
    route(cfg, &logits, null, &sel);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), sel[0].gate + sel[1].gate, 1e-5);
}

test "qwen2moe is the only arch that defaults to no gate renormalization" {
    // A file carrying no gating metadata, so the per-arch defaults apply.
    var parsed = gguf.Parsed{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .version = 3,
        .alignment = 32,
        .metadata = &.{},
        .tensors = &.{},
        .data_offset = 0,
        .file_size = 0,
    };
    defer parsed.deinit();
    try std.testing.expect(!routeCfgFromMeta(&parsed, "qwen2moe", 60, 4).weights_norm);
    try std.testing.expect(routeCfgFromMeta(&parsed, "qwen3moe", 128, 8).weights_norm);
    try std.testing.expect(routeCfgFromMeta(&parsed, "llama", 8, 2).weights_norm);
    // and softmax is the default gating everywhere the file stays silent
    try std.testing.expectEqual(GatingFunc.softmax, routeCfgFromMeta(&parsed, "llama", 8, 2).gating);
}
