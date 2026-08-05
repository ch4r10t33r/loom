//! `loom gguf check` — will this model load, answered from its header
//! (issue #51).
//!
//! A GGUF's whole structure lives in its header: architecture, shapes,
//! tokenizer, and the type of every tensor. That is the first megabyte or two
//! of a file that may be tens of gigabytes, so the question "will loom run
//! this?" is answerable over a range request rather than a download.
//!
//! This exists because the alternative happened. A 8.9 GB checkpoint was
//! fetched, sharded, and only then refused with `UnsupportedTensorType` —
//! naming neither the type nor the tensor. It turned out to be three Q5_1
//! tensors out of 377. Diagnosing that took a hand-written header parser;
//! it should have taken one command before the download.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");
const llama = @import("llama.zig");
const hf = @import("../node/hf.zig");

/// Headers are typically well under a megabyte; a 900-expert MoE's runs to a
/// few. Fetch generously — it is still four orders of magnitude less than the
/// model — and report if even that was not enough.
const HEAD_BYTES: u64 = 16 * 1024 * 1024;

pub fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, target: []const u8) !void {
    var path = target;
    var tmp_buf: [256]u8 = undefined;
    var fetched = false;

    if (std.mem.startsWith(u8, target, "http://") or std.mem.startsWith(u8, target, "https://")) {
        path = std.fmt.bufPrint(&tmp_buf, ".loom-gguf-check-{d}.tmp", .{target.len}) catch return error.PathTooLong;
        try out.print("fetching header only ({d} MB max, not the model)\n", .{HEAD_BYTES / (1 << 20)});
        try out.flush();
        hf.downloadHead(gpa, io, target, path, HEAD_BYTES) catch |e| {
            return out.print("could not fetch: {s}\n", .{@errorName(e)});
        };
        fetched = true;
    }
    defer if (fetched) Io.Dir.cwd().deleteFile(io, path) catch {};

    var parsed = gguf.parse(gpa, io, path) catch |e| {
        if (fetched and e == error.TruncatedGguf) {
            return out.print(
                "header is larger than {d} MB — cannot check remotely; download and re-run on the file\n",
                .{HEAD_BYTES / (1 << 20)},
            );
        }
        return out.print("not a readable GGUF: {s}\n", .{@errorName(e)});
    };
    defer parsed.deinit();

    // ---- architecture ----
    const arch = parsed.getString("general.architecture") orelse "(missing)";
    const name = parsed.getString("general.name") orelse "";
    const is_mla = std.mem.eql(u8, arch, "deepseek2");
    const gqa = llama.archFor(arch);
    try out.print("\n  file       {s}\n", .{target});
    if (name.len > 0) try out.print("  name       {s}\n", .{name});
    try out.print("  arch       {s:<24} ", .{arch});
    if (is_mla) {
        try out.print("engine: MLA\n", .{});
    } else if (gqa != null) {
        try out.print("engine: GQA\n", .{});
    } else {
        try out.print("NO ENGINE\n", .{});
    }

    // ---- shape ----
    var kb: [96]u8 = undefined;
    const key = struct {
        fn f(buf: []u8, a: []const u8, comptime k: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}." ++ k, .{a}) catch unreachable;
        }
    };
    const blocks = parsed.getUint(key.f(&kb, arch, "block_count"));
    const n_expert = parsed.getUint(key.f(&kb, arch, "expert_count")) orelse 0;
    const n_used = parsed.getUint(key.f(&kb, arch, "expert_used_count")) orelse 0;
    if (blocks) |b| {
        try out.print("  shape      {d} blocks", .{b});
        if (n_expert > 0) try out.print(", {d} experts, {d} active", .{ n_expert, n_used });
        try out.print("\n", .{});
    }
    const tok_model = parsed.getString("tokenizer.ggml.model") orelse "(none)";
    try out.print("  tokenizer  {s:<24} chat template: {s}\n", .{
        tok_model,
        if (parsed.getString("tokenizer.chat_template") != null) "yes" else "no",
    });

    // ---- tensor types ----
    // Small and fixed: a GGUF uses a handful of distinct types, so a flat
    // array keyed by the ggml type id beats a hash map and keeps the report
    // in a stable order.
    var counts = [_]usize{0} ** 64;
    var first_bad: []const u8 = "";
    var n_bad: usize = 0;
    var has_experts = false;
    for (parsed.tensors) |t| {
        if (t.ggml_type < counts.len) counts[t.ggml_type] += 1;
        if (!ggml.Type.supported(t.ggml_type)) {
            n_bad += 1;
            if (first_bad.len == 0) first_bad = t.name;
        }
        if (std.mem.indexOf(u8, t.name, "_exps.") != null) has_experts = true;
    }
    try out.print("  tensors    {d} total\n", .{parsed.tensors.len});
    for (counts, 0..) |n, ty| {
        if (n == 0) continue;
        const ok = ggml.Type.supported(@intCast(ty));
        try out.print("             {s:<10} {d:>5}{s}\n", .{
            typeName(@intCast(ty)), n, if (ok) "" else "   <- not implemented",
        });
    }
    try out.print("  sharding   {s}\n", .{
        if (has_experts) "expert-aligned (distributable)" else "fixed ranges (dense model)",
    });

    // ---- verdict ----
    try out.print("\n", .{});
    if (is_mla or gqa != null) {
        if (n_bad == 0) {
            try out.print("  VERDICT: loads and runs.\n", .{});
        } else {
            try out.print("  VERDICT: will NOT load — {d} tensor(s) use a type loom does not\n", .{n_bad});
            try out.print("           implement, e.g. {s}. Pick a different quantization.\n", .{first_bad});
        }
    } else {
        try out.print("  VERDICT: will NOT load — no engine for architecture '{s}'.\n", .{arch});
        try out.print("           supported: deepseek2", .{});
        for (llama.arches) |a| try out.print(", {s}", .{a.name});
        try out.print("\n", .{});
    }
    try out.flush();
}

fn typeName(t: u32) []const u8 {
    return switch (@as(ggml.Type, @enumFromInt(t))) {
        .f32 => "F32",
        .f16 => "F16",
        .q4_0 => "Q4_0",
        .q4_1 => "Q4_1",
        .q5_0 => "Q5_0",
        .q5_1 => "Q5_1",
        .q8_0 => "Q8_0",
        .q2_k => "Q2_K",
        .q3_k => "Q3_K",
        .q4_k => "Q4_K",
        .q5_k => "Q5_K",
        .q6_k => "Q6_K",
        .iq2_xxs => "IQ2_XXS",
        .iq2_xs => "IQ2_XS",
        .iq3_xxs => "IQ3_XXS",
        .iq1_s => "IQ1_S",
        .iq4_nl => "IQ4_NL",
        .iq3_s => "IQ3_S",
        .iq2_s => "IQ2_S",
        .iq4_xs => "IQ4_XS",
        .iq1_m => "IQ1_M",
        .mxfp4 => "MXFP4",
        // The unsupported ones still need naming: "type 10" tells a user
        // nothing, and knowing it is Q2_K tells them to pick another file.
        _ => switch (t) {
            3 => "Q4_1",
            7 => "Q5_1",
            10 => "Q2_K",
            11 => "Q3_K",
            15 => "Q8_K",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            40 => "NVFP4",
            else => "unknown",
        },
    };
}
