//! Byte-level BPE tokenizer (gpt2-style) from GGUF metadata — what real
//! DeepSeek/Kimi GGUFs ship (`tokenizer.ggml.model = "gpt2"`).
//!
//! Vocab tokens and merge rules are stored in GPT-2's byte-level encoding:
//! every input byte maps to a printable unicode codepoint (identity for the
//! printable ranges, 0x100+n for the rest), the text becomes a sequence of
//! those codepoints, and merges apply by rank (earlier merge = lower rank =
//! higher priority). Decoding maps codepoints back to raw bytes.
//!
//! Pretokenization note: reference implementations split text with a
//! model-specific regex before merging. We use a simplified splitter
//! (letter/digit/space/punct runs, space attached to the following word).
//! Token boundaries can differ from the reference on unusual input; decode is
//! exact regardless (it is a pure table lookup).

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const special = @import("special.zig");

/// GPT-2 byte -> unicode codepoint table.
fn byteToUnicode(b: u8) u21 {
    // printable ranges map to themselves
    if ((b >= '!' and b <= '~') or (b >= 0xa1 and b <= 0xac) or (b >= 0xae and b <= 0xff)) {
        return b;
    }
    // others get 0x100 + running index; recompute the index deterministically
    var n: u21 = 0;
    var i: usize = 0;
    while (i < b) : (i += 1) {
        const c: u8 = @intCast(i);
        if (!((c >= '!' and c <= '~') or (c >= 0xa1 and c <= 0xac) or (c >= 0xae and c <= 0xff))) n += 1;
    }
    return 0x100 + n;
}

pub const Bpe = struct {
    tokens: []const []const u8, // arena-owned by Parsed
    types: []const i32,
    lookup: std.StringHashMap(u32), // byte-level token string -> id
    merge_rank: std.StringHashMap(u32), // "left right" -> rank
    unicode_to_byte: std.AutoHashMap(u21, u8),
    specials: special.Set,
    bos: u32,
    eos: u32,
    add_bos: bool,

    pub fn init(gpa: std.mem.Allocator, parsed: *const gguf.Parsed) !Bpe {
        const tokens = switch (parsed.findMeta("tokenizer.ggml.tokens") orelse return error.NoTokenizer) {
            .array_str => |a| a,
            else => return error.NoTokenizer,
        };
        const types: []const i32 = switch (parsed.findMeta("tokenizer.ggml.token_type") orelse return error.NoTokenizer) {
            .array_i32 => |a| a,
            else => &.{},
        };
        const merges = switch (parsed.findMeta("tokenizer.ggml.merges") orelse return error.NoMerges) {
            .array_str => |a| a,
            else => return error.NoMerges,
        };

        // token_type is indexed by token id (special.Set.build, decode), so a
        // shorter array than `tokens` is an OOB read (security issue #29).
        if (tokens.len == 0 or (types.len != 0 and types.len != tokens.len)) return error.BadTokenizer;
        const bos_id = parsed.getUint("tokenizer.ggml.bos_token_id") orelse 0;
        const eos_id = parsed.getUint("tokenizer.ggml.eos_token_id") orelse 0;
        if (bos_id >= tokens.len or eos_id >= tokens.len) return error.BadTokenizer;

        var lookup = std.StringHashMap(u32).init(gpa);
        errdefer lookup.deinit();
        for (tokens, 0..) |t, i| try lookup.put(t, @intCast(i));

        var merge_rank = std.StringHashMap(u32).init(gpa);
        errdefer merge_rank.deinit();
        for (merges, 0..) |m, i| try merge_rank.put(m, @intCast(i));

        var u2b = std.AutoHashMap(u21, u8).init(gpa);
        errdefer u2b.deinit();
        var b: usize = 0;
        while (b < 256) : (b += 1) {
            try u2b.put(byteToUnicode(@intCast(b)), @intCast(b));
        }

        var specials = try special.Set.build(gpa, tokens, types);
        errdefer specials.deinit(gpa);

        return .{
            .tokens = tokens,
            .types = types,
            .lookup = lookup,
            .merge_rank = merge_rank,
            .unicode_to_byte = u2b,
            .specials = specials,
            .bos = @intCast(parsed.getUint("tokenizer.ggml.bos_token_id") orelse 0),
            .eos = @intCast(parsed.getUint("tokenizer.ggml.eos_token_id") orelse 0),
            // Default matches llama.cpp: for BPE vocabs add_bos is OFF unless
            // the file says otherwise or the pre-tokenizer family turns it on
            // (llama3 does; deepseek and gpt-4o/o200k do not). The previous
            // blanket `true` happened to match the deepseek GGUFs this engine
            // grew up on, which DO ship an explicit add_bos_token=true.
            .add_bos = parsed.getBool("tokenizer.ggml.add_bos_token") orelse blk: {
                const pre = parsed.getString("tokenizer.ggml.pre") orelse "";
                break :blk std.mem.eql(u8, pre, "llama-bpe") or std.mem.eql(u8, pre, "llama3") or
                    std.mem.eql(u8, pre, "llama-v3");
            },
        };
    }

    pub fn deinit(self: *Bpe, gpa: std.mem.Allocator) void {
        self.lookup.deinit();
        self.merge_rank.deinit();
        self.unicode_to_byte.deinit();
        self.specials.deinit(gpa);
    }

    const CharClass = enum { letter, digit, space, other };

    fn classify(b: u8) CharClass {
        if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return .space;
        if (b >= '0' and b <= '9') return .digit;
        if ((b >= 'a' and b <= 'z') or (b >= 'A' and b <= 'Z') or b >= 0x80) return .letter;
        return .other;
    }

    /// Simplified gpt2-style pretokenizer over raw bytes: runs of letters,
    /// digits, punctuation, or whitespace; a single leading space attaches to
    /// a following letter/digit/punct run. Returns byte ranges.
    fn pretokenize(gpa: std.mem.Allocator, text: []const u8) !std.ArrayList([2]usize) {
        var out = std.ArrayList([2]usize).empty;
        errdefer out.deinit(gpa);
        var i: usize = 0;
        while (i < text.len) {
            const start = i;
            // one space may prefix a word
            if (text[i] == ' ' and i + 1 < text.len and classify(text[i + 1]) != .space) {
                i += 1;
            }
            if (i >= text.len) {
                try out.append(gpa, .{ start, text.len });
                break;
            }
            const cls = classify(text[i]);
            i += 1;
            while (i < text.len and classify(text[i]) == cls and cls != .space) : (i += 1) {}
            if (cls == .space) {
                while (i < text.len and classify(text[i]) == .space) : (i += 1) {}
                // trailing space run: keep the last space for the next word
                if (i < text.len and text[i - 1] == ' ') i -= 1;
                if (i <= start) i = start + 1;
            }
            try out.append(gpa, .{ start, i });
        }
        return out;
    }

    /// Encode one pretoken (raw bytes) by byte-level mapping + rank merging,
    /// appending token ids to `out`.
    fn encodePiece(self: *const Bpe, gpa: std.mem.Allocator, piece: []const u8, out: *std.ArrayList(u32)) !void {
        // byte-level encode into per-symbol utf8 strings
        var syms = std.ArrayList([]const u8).empty;
        defer {
            for (syms.items) |s| gpa.free(s);
            syms.deinit(gpa);
        }
        for (piece) |b| {
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(byteToUnicode(b), &buf) catch unreachable;
            try syms.append(gpa, try gpa.dupe(u8, buf[0..n]));
        }

        // merge loop: repeatedly apply the lowest-rank adjacent pair
        var pairbuf = std.ArrayList(u8).empty;
        defer pairbuf.deinit(gpa);
        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_i: ?usize = null;
            var i: usize = 0;
            while (i + 1 < syms.items.len) : (i += 1) {
                pairbuf.clearRetainingCapacity();
                try pairbuf.appendSlice(gpa, syms.items[i]);
                try pairbuf.append(gpa, ' ');
                try pairbuf.appendSlice(gpa, syms.items[i + 1]);
                if (self.merge_rank.get(pairbuf.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_i = i;
                    }
                }
            }
            const bi = best_i orelse break;
            const merged = try std.mem.concat(gpa, u8, &.{ syms.items[bi], syms.items[bi + 1] });
            gpa.free(syms.items[bi]);
            gpa.free(syms.items[bi + 1]);
            syms.items[bi] = merged;
            _ = syms.orderedRemove(bi + 1);
        }

        for (syms.items) |s| {
            if (self.lookup.get(s)) |id| {
                try out.append(gpa, id);
            }
            // symbol not in vocab: drop (shouldn't happen — all 256 byte
            // symbols exist in a well-formed byte-level vocab)
        }
    }

    /// BPE-encode a normal (non-special) text segment.
    fn encodeNormal(self: *const Bpe, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        var ranges = try pretokenize(gpa, text);
        defer ranges.deinit(gpa);
        for (ranges.items) |r| {
            try self.encodePiece(gpa, text[r[0]..r[1]], out);
        }
    }

    /// `parse_special`: when true, special-token strings (chat-template markers,
    /// control tokens) in `text` are emitted as their atomic ids; when false
    /// they are encoded as ordinary text (so untrusted input cannot inject a
    /// control token).
    pub fn encode(self: *const Bpe, gpa: std.mem.Allocator, text: []const u8, add_bos: bool, parse_special: bool) ![]u32 {
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(gpa);
        if (add_bos and self.add_bos) try out.append(gpa, self.bos);

        if (!parse_special) {
            try self.encodeNormal(gpa, text, &out);
            return out.toOwnedSlice(gpa);
        }

        // Split on special tokens, emitting their ids atomically and
        // BPE-encoding the text between.
        var seg_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (self.specials.matchAt(text[i..])) |sp| {
                if (i > seg_start) try self.encodeNormal(gpa, text[seg_start..i], &out);
                try out.append(gpa, sp.id);
                i += sp.text.len;
                seg_start = i;
            } else i += 1;
        }
        if (seg_start < text.len) try self.encodeNormal(gpa, text[seg_start..], &out);
        return out.toOwnedSlice(gpa);
    }

    /// Decode one token: map each byte-level codepoint back to its raw byte.
    /// Control tokens (type 3) print nothing.
    pub fn decode(self: *const Bpe, w: *Io.Writer, id: u32) !void {
        if (id >= self.tokens.len) return;
        if (id < self.types.len and self.types[id] == 3) return; // control
        const piece = self.tokens[id];
        var i: usize = 0;
        while (i < piece.len) {
            const l = std.unicode.utf8ByteSequenceLength(piece[i]) catch {
                i += 1;
                continue;
            };
            if (i + l > piece.len) break;
            const cp = std.unicode.utf8Decode(piece[i .. i + l]) catch {
                i += l;
                continue;
            };
            if (self.unicode_to_byte.get(cp)) |b| {
                try w.writeAll(&.{b});
            } else {
                try w.writeAll(piece[i .. i + l]); // passthrough (shouldn't happen)
            }
            i += l;
        }
    }
};

test "byte-unicode table roundtrips all 256 bytes" {
    var seen = std.AutoHashMap(u21, void).init(std.testing.allocator);
    defer seen.deinit();
    var b: usize = 0;
    while (b < 256) : (b += 1) {
        const cp = byteToUnicode(@intCast(b));
        try std.testing.expect(!seen.contains(cp)); // injective
        try seen.put(cp, {});
    }
    // spot checks against the reference table
    try std.testing.expect(byteToUnicode('A') == 'A');
    try std.testing.expect(byteToUnicode(' ') == 0x120); // 'Ġ'
    try std.testing.expect(byteToUnicode('\n') == 0x10a); // 'Ċ'
}

test "pretokenizer attaches single space to following word" {
    const gpa = std.testing.allocator;
    var r = try Bpe.pretokenize(gpa, "hello world 42!");
    defer r.deinit(gpa);
    // "hello", " world", " 42", "!"
    try std.testing.expect(r.items.len == 4);
    try std.testing.expectEqualSlices(usize, &.{ 0, 5 }, &r.items[0]);
    try std.testing.expectEqualSlices(usize, &.{ 5, 11 }, &r.items[1]);
    try std.testing.expectEqualSlices(usize, &.{ 11, 14 }, &r.items[2]);
    try std.testing.expectEqualSlices(usize, &.{ 14, 15 }, &r.items[3]);
}
