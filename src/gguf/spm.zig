//! SentencePiece (SPM) tokenizer read out of GGUF metadata.
//!
//! Split out of llama.zig so both engines can share it without an import
//! cycle: tok.zig unions this with the gpt2-style BPE in bpe.zig, and both
//! engines depend on tok.zig.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const special = @import("special.zig");

pub const Tokenizer = struct {
    tokens: []const []const u8, // arena-owned by Parsed
    scores: []const f32,
    types: []const i32,
    lookup: std.StringHashMap(u32),
    specials: special.Set,
    bos: u32,
    eos: u32,

    const BYTE_TYPE: i32 = 6;

    pub fn init(gpa: std.mem.Allocator, parsed: *const gguf.Parsed) !Tokenizer {
        const tokens = switch (parsed.findMeta("tokenizer.ggml.tokens") orelse return error.NoTokenizer) {
            .array_str => |a| a,
            else => return error.NoTokenizer,
        };
        const scores = switch (parsed.findMeta("tokenizer.ggml.scores") orelse return error.NoTokenizer) {
            .array_f32 => |a| a,
            else => return error.NoTokenizer,
        };
        const types = switch (parsed.findMeta("tokenizer.ggml.token_type") orelse return error.NoTokenizer) {
            .array_i32 => |a| a,
            else => return error.NoTokenizer,
        };
        // `scores` and `types` are indexed by a token id derived from `tokens`
        // (encode does `self.scores[id]`), so unequal lengths are an OOB read
        // from a malicious GGUF (security issue #29).
        if (scores.len != tokens.len or types.len != tokens.len) return error.BadTokenizer;
        if (tokens.len == 0) return error.BadTokenizer;
        // bos/eos are metadata and get emitted into the token stream, which
        // indexes token_embd rows; keep them inside the vocab.
        const bos_id = parsed.getUint("tokenizer.ggml.bos_token_id") orelse 1;
        const eos_id = parsed.getUint("tokenizer.ggml.eos_token_id") orelse 2;
        if (bos_id >= tokens.len or eos_id >= tokens.len) return error.BadTokenizer;

        var lookup = std.StringHashMap(u32).init(gpa);
        errdefer lookup.deinit();
        for (tokens, 0..) |t, i| try lookup.put(t, @intCast(i));
        var specials = try special.Set.build(gpa, tokens, types);
        errdefer specials.deinit(gpa);
        return .{
            .tokens = tokens,
            .scores = scores,
            .types = types,
            .lookup = lookup,
            .specials = specials,
            .bos = std.math.cast(u32, bos_id) orelse return error.BadTokenizer,
            .eos = std.math.cast(u32, eos_id) orelse return error.BadTokenizer,
        };
    }

    pub fn deinit(self: *Tokenizer, gpa: std.mem.Allocator) void {
        self.lookup.deinit();
        self.specials.deinit(gpa);
    }

    /// SPM encode: prefix a space, map ' ' -> U+2581, split into UTF-8 chars,
    /// then greedily merge the adjacent pair whose concatenation is the
    /// highest-scoring vocab entry. Unmatched symbols fall back to byte tokens.
    /// `parse_special`: when true, special-token strings are emitted as their
    /// atomic ids; when false they are SPM-encoded as ordinary text (so
    /// untrusted input cannot inject a control token).
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_bos: bool, parse_special: bool) ![]u32 {
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(gpa);
        if (add_bos) try out.append(gpa, self.bos);

        if (!parse_special) {
            try self.encodeSegment(gpa, text, true, &out);
            return out.toOwnedSlice(gpa);
        }

        // Split on special tokens, emitting their ids atomically and
        // SPM-encoding the text between.
        var seg_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (self.specials.matchAt(text[i..])) |sp| {
                if (i > seg_start) try self.encodeSegment(gpa, text[seg_start..i], seg_start == 0, &out);
                try out.append(gpa, sp.id);
                i += sp.text.len;
                seg_start = i;
            } else i += 1;
        }
        if (seg_start < text.len) try self.encodeSegment(gpa, text[seg_start..], seg_start == 0, &out);
        return out.toOwnedSlice(gpa);
    }

    /// SPM-encode one normal segment into `out`. `add_space_prefix` adds the
    /// leading dummy space (SPM add_dummy_prefix); applied only to a segment at
    /// the very start of the input, not to content following a special token.
    fn encodeSegment(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_space_prefix: bool, out: *std.ArrayList(u32)) !void {
        // preprocess: leading space, spaces -> ▁ (e2 96 81)
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(gpa);
        if (add_space_prefix) try buf.appendSlice(gpa, "\xe2\x96\x81");
        for (text) |ch| {
            if (ch == ' ') {
                try buf.appendSlice(gpa, "\xe2\x96\x81");
            } else {
                try buf.append(gpa, ch);
            }
        }
        const s = buf.items;

        // symbols as (start,end) ranges over s, initially one UTF-8 char each
        const Range = struct { start: usize, end: usize };
        var syms = std.ArrayList(Range).empty;
        defer syms.deinit(gpa);
        {
            var i: usize = 0;
            while (i < s.len) {
                const l = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
                const end = @min(i + l, s.len);
                try syms.append(gpa, .{ .start = i, .end = end });
                i = end;
            }
        }

        // greedy best-pair merging by vocab score
        while (true) {
            var best_score: f32 = -std.math.inf(f32);
            var best_i: ?usize = null;
            var i: usize = 0;
            while (i + 1 < syms.items.len) : (i += 1) {
                const merged = s[syms.items[i].start..syms.items[i + 1].end];
                if (self.lookup.get(merged)) |id| {
                    if (self.scores[id] > best_score) {
                        best_score = self.scores[id];
                        best_i = i;
                    }
                }
            }
            const bi = best_i orelse break;
            syms.items[bi].end = syms.items[bi + 1].end;
            _ = syms.orderedRemove(bi + 1);
        }

        // map symbols to ids (byte fallback for stragglers)
        for (syms.items) |r| {
            const piece = s[r.start..r.end];
            if (self.lookup.get(piece)) |id| {
                try out.append(gpa, id);
            } else {
                for (piece) |byte| {
                    var namebuf: [8]u8 = undefined;
                    const bname = std.fmt.bufPrint(&namebuf, "<0x{X:0>2}>", .{byte}) catch unreachable;
                    if (self.lookup.get(bname)) |id| try out.append(gpa, id);
                    // no byte token in vocab: drop the byte
                }
            }
        }
    }

    /// Decode one token into `w`. Byte tokens emit their byte; ▁ becomes space.
    pub fn decode(self: *const Tokenizer, w: *Io.Writer, id: u32) !void {
        if (id >= self.tokens.len) return;
        const piece = self.tokens[id];
        if (id < self.types.len and self.types[id] == BYTE_TYPE) {
            // "<0xXX>"
            if (piece.len == 6) {
                const byte = std.fmt.parseInt(u8, piece[3..5], 16) catch return;
                try w.writeAll(&.{byte});
            }
            return;
        }
        var i: usize = 0;
        while (i < piece.len) {
            if (i + 3 <= piece.len and std.mem.eql(u8, piece[i .. i + 3], "\xe2\x96\x81")) {
                try w.writeAll(" ");
                i += 3;
            } else {
                try w.writeAll(piece[i .. i + 1]);
                i += 1;
            }
        }
    }
};
