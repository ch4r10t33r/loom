//! The tokenizer a GGUF actually carries, selected by `tokenizer.ggml.model`.
//!
//! Real DeepSeek/Kimi/Qwen/GLM checkpoints ship a gpt2-style BPE; llama and
//! Mixtral ship SentencePiece; fixtures (and some conversions) use SPM too.
//! Both engines take this union rather than picking one, so an architecture's
//! tokenizer is a property of the file, not of the engine that reads it.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const spm = @import("spm.zig");
const bpe = @import("bpe.zig");

pub const Tok = union(enum) {
    spm: spm.Tokenizer,
    bpe: bpe.Bpe,

    /// Build whichever tokenizer the file declares. `tokenizer.ggml.model` is
    /// "gpt2" for BPE; anything else (llama, or absent) is treated as SPM,
    /// matching what conversions emit.
    pub fn init(gpa: std.mem.Allocator, parsed: *const gguf.Parsed) !Tok {
        const model = parsed.getString("tokenizer.ggml.model") orelse "llama";
        if (std.mem.eql(u8, model, "gpt2")) {
            return .{ .bpe = try bpe.Bpe.init(gpa, parsed) };
        }
        return .{ .spm = try spm.Tokenizer.init(gpa, parsed) };
    }

    pub fn encode(self: *const Tok, gpa: std.mem.Allocator, text: []const u8, add_bos: bool, parse_special: bool) ![]u32 {
        return switch (self.*) {
            .spm => |*t| t.encode(gpa, text, add_bos, parse_special),
            .bpe => |*t| t.encode(gpa, text, add_bos, parse_special),
        };
    }
    pub fn decode(self: *const Tok, w: *Io.Writer, id: u32) !void {
        return switch (self.*) {
            .spm => |*t| t.decode(w, id),
            .bpe => |*t| t.decode(w, id),
        };
    }
    pub fn eosId(self: *const Tok) u32 {
        return switch (self.*) {
            .spm => |*t| t.eos,
            .bpe => |*t| t.eos,
        };
    }
    pub fn deinit(self: *Tok, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .spm => |*t| t.deinit(gpa),
            .bpe => |*t| t.deinit(gpa),
        }
    }
};
