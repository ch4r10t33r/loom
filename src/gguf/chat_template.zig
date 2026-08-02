//! Per-model chat templates (issue #19): render OpenAI `messages[]` into the
//! prompt string a given model expects.
//!
//! This is format detection plus built-in renderers, the approach llama.cpp
//! used before it embedded a Jinja engine, not a full Jinja2 interpreter (the
//! GGUF `tokenizer.chat_template` is arbitrary Jinja; interpreting it is out of
//! scope). The format is detected from that template string (or the model
//! arch), with a CLI override.
//!
//! The BOS is added by the tokenizer (not embedded here). Special-token markers
//! this emits (chatml `<|im_start|>`, llama3 `<|start_header_id|>`, gemma
//! `<start_of_turn>`) tokenize to their atomic ids via special.zig when the
//! model's vocab contains them as control / user-defined tokens.

const std = @import("std");

pub const Format = enum {
    generic,
    deepseek,
    chatml,
    llama2,
    llama3,
    gemma,
    mistral,
    /// OpenAI Harmony (gpt-oss): `<|start|>role<|message|>...<|end|>` turns;
    /// assistant replies carry a channel tag and end with `<|return|>`.
    harmony,
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

/// Parse a `--chat-format` override value; null if unrecognized.
pub fn parse(name: []const u8) ?Format {
    inline for (@typeInfo(Format).@"enum".fields) |f| {
        if (std.mem.eql(u8, name, f.name)) return @field(Format, f.name);
    }
    return null;
}

/// Detect the chat format from the GGUF `tokenizer.chat_template` string (and
/// the model arch as a fallback signal). Substring markers mirror llama.cpp's
/// detection.
pub fn detect(template: ?[]const u8, arch: []const u8) Format {
    if (template) |t| {
        if (contains(t, "<|channel|>")) return .harmony;
        if (contains(t, "<|im_start|>")) return .chatml;
        if (contains(t, "<|start_header_id|>")) return .llama3;
        if (contains(t, "<start_of_turn>")) return .gemma;
        if (contains(t, "[INST]")) return if (contains(t, "<<SYS>>")) .llama2 else .mistral;
        if (contains(t, "Assistant:") or contains(t, "\u{ff5c}Assistant\u{ff5c}")) return .deepseek;
    }
    if (std.mem.eql(u8, arch, "deepseek2")) return .deepseek;
    if (std.mem.eql(u8, arch, "gpt-oss")) return .harmony;
    // Qwen MoE and GLM-4.5 MoE both use ChatML markers. Only reached when the
    // file carries no chat_template at all — the template string above is the
    // authority when present.
    if (std.mem.eql(u8, arch, "qwen2moe") or std.mem.eql(u8, arch, "qwen3moe") or
        std.mem.eql(u8, arch, "glm4moe")) return .chatml;
    return .generic;
}

fn contains(h: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, h, needle) != null;
}

/// True if the format's scaffolding uses special-token markers (chatml, llama3,
/// gemma) that must be tokenized as their atomic ids. For these, the rendered
/// prompt is encoded with parse_special=true. Text-marker formats (deepseek,
/// llama2, mistral, generic) have no special scaffolding, so their prompts are
/// encoded with parse_special=false and user content cannot inject control ids.
pub fn usesSpecialMarkers(fmt: Format) bool {
    return switch (fmt) {
        .chatml, .llama3, .gemma, .harmony => true,
        .deepseek, .llama2, .mistral, .generic => false,
    };
}

fn roleIs(m: Message, r: []const u8) bool {
    return std.mem.eql(u8, m.role, r);
}

/// Render messages into a prompt. The tokenizer prepends BOS, so BOS is not
/// embedded here. `add_generation_prompt` appends the assistant cue (true for a
/// completion request whose last turn is the user).
pub fn render(gpa: std.mem.Allocator, fmt: Format, msgs: []const Message, add_generation_prompt: bool) ![]u8 {
    var b = std.ArrayList(u8).empty;
    errdefer b.deinit(gpa);
    switch (fmt) {
        .deepseek => {
            for (msgs) |m| {
                if (roleIs(m, "system")) {
                    try b.print(gpa, "{s}\n\n", .{m.content});
                } else if (roleIs(m, "assistant")) {
                    try b.print(gpa, "Assistant: {s}", .{m.content});
                } else {
                    try b.print(gpa, "User: {s}\n\n", .{m.content});
                }
            }
            if (add_generation_prompt) try b.appendSlice(gpa, "Assistant:");
        },
        .chatml => {
            for (msgs) |m| try b.print(gpa, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ m.role, m.content });
            if (add_generation_prompt) try b.appendSlice(gpa, "<|im_start|>assistant\n");
        },
        .llama3 => {
            for (msgs) |m| try b.print(gpa, "<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ m.role, m.content });
            if (add_generation_prompt) try b.appendSlice(gpa, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        },
        .gemma => {
            // gemma has no system role; fold system into the first user turn
            for (msgs) |m| {
                if (roleIs(m, "system")) {
                    try b.print(gpa, "<start_of_turn>user\n{s}<end_of_turn>\n", .{m.content});
                } else {
                    const who = if (roleIs(m, "assistant")) "model" else "user";
                    try b.print(gpa, "<start_of_turn>{s}\n{s}<end_of_turn>\n", .{ who, m.content });
                }
            }
            if (add_generation_prompt) try b.appendSlice(gpa, "<start_of_turn>model\n");
        },
        .llama2 => {
            // basic single/first-turn [INST] with optional <<SYS>>
            try b.appendSlice(gpa, "[INST] ");
            for (msgs) |m| {
                if (roleIs(m, "system")) {
                    try b.print(gpa, "<<SYS>>\n{s}\n<</SYS>>\n\n", .{m.content});
                } else if (roleIs(m, "user")) {
                    try b.print(gpa, "{s} ", .{m.content});
                }
            }
            try b.appendSlice(gpa, "[/INST]");
        },
        .mistral => {
            try b.appendSlice(gpa, "[INST] ");
            for (msgs) |m| {
                if (roleIs(m, "user") or roleIs(m, "system")) try b.print(gpa, "{s} ", .{m.content});
            }
            try b.appendSlice(gpa, "[/INST]");
        },
        .harmony => {
            // Minimal Harmony: enough scaffold that gpt-oss answers on the
            // "final" channel. Prior assistant turns are replayed as final
            // messages; the cue leaves the channel open for the model to pick
            // (it emits `<|channel|>analysis` or `final` itself).
            for (msgs) |m| {
                if (roleIs(m, "assistant")) {
                    try b.print(gpa, "<|start|>assistant<|channel|>final<|message|>{s}<|end|>", .{m.content});
                } else {
                    try b.print(gpa, "<|start|>{s}<|message|>{s}<|end|>", .{ m.role, m.content });
                }
            }
            if (add_generation_prompt) try b.appendSlice(gpa, "<|start|>assistant");
        },
        .generic => {
            for (msgs) |m| try b.print(gpa, "{s}: {s}\n", .{ m.role, m.content });
            if (add_generation_prompt) try b.appendSlice(gpa, "assistant:");
        },
    }
    return b.toOwnedSlice(gpa);
}

test "detect from markers and arch" {
    try std.testing.expectEqual(Format.chatml, detect("...<|im_start|>...", "llama"));
    try std.testing.expectEqual(Format.llama3, detect("...<|start_header_id|>...", "llama"));
    try std.testing.expectEqual(Format.gemma, detect("...<start_of_turn>...", "gemma"));
    try std.testing.expectEqual(Format.mistral, detect("...[INST]...", "llama"));
    try std.testing.expectEqual(Format.llama2, detect("...[INST]...<<SYS>>...", "llama"));
    try std.testing.expectEqual(Format.deepseek, detect(null, "deepseek2"));
    try std.testing.expectEqual(Format.harmony, detect(null, "gpt-oss"));
    try std.testing.expectEqual(Format.harmony, detect("...<|channel|>...", "gpt-oss"));
    // Qwen/GLM MoE default to ChatML when the file carries no template
    try std.testing.expectEqual(Format.chatml, detect(null, "qwen2moe"));
    try std.testing.expectEqual(Format.chatml, detect(null, "qwen3moe"));
    try std.testing.expectEqual(Format.chatml, detect(null, "glm4moe"));
    // Mixtral is arch "llama" and ships a mistral-style template
    try std.testing.expectEqual(Format.generic, detect(null, "llama"));
    try std.testing.expectEqual(Format.mistral, detect("{{ '[INST] ' }}", "llama"));
    // an explicit template always beats the arch fallback
    try std.testing.expectEqual(Format.llama3, detect("<|start_header_id|>", "qwen3moe"));
}

test "harmony render user turn + cue" {
    const gpa = std.testing.allocator;
    const msgs = [_]Message{.{ .role = "user", .content = "hi" }};
    const out = try render(gpa, .harmony, &msgs, true);
    defer gpa.free(out);
    try std.testing.expectEqualStrings("<|start|>user<|message|>hi<|end|><|start|>assistant", out);
}

test "deepseek render single user turn" {
    const gpa = std.testing.allocator;
    const msgs = [_]Message{.{ .role = "user", .content = "hi" }};
    const out = try render(gpa, .deepseek, &msgs, true);
    defer gpa.free(out);
    try std.testing.expectEqualStrings("User: hi\n\nAssistant:", out);
}
