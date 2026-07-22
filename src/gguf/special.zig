//! Special-token matching (issue: special-token-aware tokenization).
//!
//! GGUF marks control tokens (type 3) and user-defined/added tokens (type 4)
//! in `tokenizer.ggml.token_type`. Their vocab strings are stored literally
//! (not byte-level or SPM encoded), so chat-template markers like
//! `<|im_start|>`, `<|start_header_id|>`, `<start_of_turn>`, or a model's
//! `<bos>` string must be matched atomically in the input and emitted as their
//! id rather than split by the normal BPE/SPM tokenizer.
//!
//! This module builds the set of such tokens and does longest-match lookup at a
//! position. Both tokenizers (bpe.zig, llama.zig SPM) use it to split the input
//! into special-token ids and normal-text segments before their own encoding.
//!
//! Note: matching is always on (parse_special = true). A raw prompt containing
//! a marker string is therefore interpreted as that control token; segment-wise
//! parse control (to prevent user content injecting role markers) is a follow-up.

const std = @import("std");

const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;

pub const Tok = struct { text: []const u8, id: u32 };

pub const Set = struct {
    toks: []Tok, // owned slice; sorted by descending text length (longest-match)
    first_byte: [256]bool,

    /// Collect control/user-defined tokens. `tokens` and `types` are borrowed
    /// (arena-owned by the GGUF Parsed), so the stored slices stay valid for the
    /// tokenizer's lifetime.
    pub fn build(gpa: std.mem.Allocator, tokens: []const []const u8, types: []const i32) !Set {
        var list = std.ArrayList(Tok).empty;
        errdefer list.deinit(gpa);
        var fb = [_]bool{false} ** 256;
        for (tokens, 0..) |t, i| {
            if (i >= types.len) break;
            if ((types[i] == CONTROL or types[i] == USER_DEFINED) and t.len > 0) {
                try list.append(gpa, .{ .text = t, .id = @intCast(i) });
                fb[t[0]] = true;
            }
        }
        const arr = try list.toOwnedSlice(gpa);
        std.sort.pdq(Tok, arr, {}, cmpDescLen);
        return .{ .toks = arr, .first_byte = fb };
    }

    fn cmpDescLen(_: void, a: Tok, b: Tok) bool {
        return a.text.len > b.text.len;
    }

    pub fn deinit(self: *Set, gpa: std.mem.Allocator) void {
        gpa.free(self.toks);
    }

    pub fn isEmpty(self: *const Set) bool {
        return self.toks.len == 0;
    }

    /// Longest special token whose string is a prefix of `rest`, or null. The
    /// first-byte filter skips the common no-match case cheaply, and the sorted
    /// order makes the first prefix match the longest one.
    pub fn matchAt(self: *const Set, rest: []const u8) ?Tok {
        if (rest.len == 0 or !self.first_byte[rest[0]]) return null;
        for (self.toks) |t| {
            if (rest.len >= t.text.len and std.mem.startsWith(u8, rest, t.text)) return t;
        }
        return null;
    }
};

test "longest-match and first-byte filter" {
    const gpa = std.testing.allocator;
    const tokens = [_][]const u8{ "a", "<|im_start|>", "<|im_end|>", "<|im_", "n" };
    const types = [_]i32{ 1, 4, 4, 4, 1 }; // only the type-4 ones are special
    var set = try Set.build(gpa, &tokens, &types);
    defer set.deinit(gpa);

    // longest match: "<|im_start|>" wins over the "<|im_" prefix token
    const m = set.matchAt("<|im_start|>hello").?;
    try std.testing.expectEqual(@as(u32, 1), m.id);
    try std.testing.expectEqualStrings("<|im_start|>", m.text);
    // normal text: no special starts here
    try std.testing.expect(set.matchAt("hello") == null);
    // "a"/"n" are normal (type 1), not matched
    try std.testing.expect(set.matchAt("apple") == null);
}
