//! Pre-configured loom networks -- Ethereum's chain registry, for LLM
//! networks. A name resolves to a stable network id (never derived, never
//! changing across model hardforks) and the canonical model the network
//! serves. Every checkpoint below was verified against this codebase with
//! `loom gguf check` before earning its row.
//!
//!   devnet   1337  GLM-4.5-Air (106B-A12B, glm4moe)  -- PoC validation;
//!                  runs distributed on commodity boxes today.
//!   testnet  2     GLM-4.6 (357B-A32B, glm4moe)      -- performance tier;
//!                  ~200 GB at Q4_K_M, one 256 GB box or a small cluster.
//!   mainnet  1     GLM 5.2 (744B-A40B, glm-dsa)      -- production; needs
//!                  the glm-dsa engine variant (dense-attention fallback)
//!                  and 256 GB (2-bit) to 512 GB (4-bit) class hardware.
//!                  Kimi K3 is the recorded alternative candidate.
//!
//! Arch policy: on testnet and mainnet a node REFUSES to serve a model whose
//! architecture differs from the network's -- one network, one model is the
//! invariant everything else (RAG embedding equality, expert fetch) rests
//! on. devnet only warns, because a PoC network is where mismatches are
//! discovered on purpose.
const std = @import("std");

pub const Network = struct {
    name: []const u8,
    id: u64,
    arch: []const u8,
    model: []const u8,
    desc: []const u8,
    /// Refuse (true) or merely warn (false) on serving a mismatched arch.
    strict: bool,
    /// Default bootnodes: `--network NAME` with no explicit --bootstrap
    /// joins through these. Empty until a network is stood up.
    bootnodes: []const []const u8 = &.{},
};

pub const registry = [_]Network{
    .{ .name = "devnet", .id = 1337, .arch = "glm4moe", .model = "unsloth/GLM-4.5-Air-GGUF (Q4_K_M)", .desc = "PoC validation -- GLM-4.5-Air 106B-A12B", .strict = false, .bootnodes = &.{"REDACTED-IP:8771"} },
    .{ .name = "testnet", .id = 2, .arch = "glm4moe", .model = "unsloth/GLM-4.6-GGUF (Q4_K_M)", .desc = "performance tier -- GLM-4.6 357B-A32B", .strict = true },
    .{ .name = "mainnet", .id = 1, .arch = "glm-dsa", .model = "unsloth/GLM-5.2-GGUF (UD-Q4_K_XL)", .desc = "production -- GLM 5.2 744B-A40B", .strict = true },
};

pub fn byName(name: []const u8) ?*const Network {
    for (&registry) |*n| {
        if (std.mem.eql(u8, n.name, name)) return n;
    }
    return null;
}

test "registry ids are the stable constants the wire depends on" {
    // These are protocol constants now: changing one partitions a network.
    try std.testing.expectEqual(@as(u64, 1337), byName("devnet").?.id);
    try std.testing.expectEqual(@as(u64, 2), byName("testnet").?.id);
    try std.testing.expectEqual(@as(u64, 1), byName("mainnet").?.id);
    try std.testing.expectEqual(@as(?*const Network, null), byName("nope"));
}

test "strictness: production networks refuse, devnet warns" {
    try std.testing.expect(!byName("devnet").?.strict);
    try std.testing.expect(byName("testnet").?.strict);
    try std.testing.expect(byName("mainnet").?.strict);
}
