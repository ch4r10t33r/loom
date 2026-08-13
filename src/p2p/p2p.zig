//! Minimal P2P peer server — the seed of the v1 distributed expert directory.
//!
//! A real deployment runs this over Hyperswarm/HyperDHT; here it is a small
//! line protocol over TCP that answers "who has expert N" against this node's
//! immutable, content-addressed index. It exposes exactly the query v1's fetch
//! policy needs (expert_id -> holder + offset/len/digest) without yet doing
//! replication or heat tracking.
//!
//! Protocol (line-delimited; GETR responses carry a binary payload after the
//! header line):
//!   HELLO         -> LOOM/0 experts=<n> unique_bytes=<b>
//!   COUNT         -> <n>
//!   HAS <id>      -> PRESENT <id> off=<o> len=<l> sha256=<hex> | ABSENT <id> | ERR range
//!   PING          -> PONG
//!   -- weight-range distribution (when a GGUF store is attached) --
//!   MANIFEST      -> MANIFEST version=<hex> size=<b> ranges=<n> range_size=<b> mode=<fixed|expert> resident=<n>
//!   MANIFESTFILE  -> MANIFESTFILE len=<n>\n<serialized manifest bytes> (digests + extent lists)
//!   DIGEST <i>    -> DIGEST <i> <hex>
//!   DIGESTS       -> DIGESTS <n>\n<hex>\n x n     (bulk, for bootstrap)
//!   HOLDINGS      -> HOLDINGS <hex bitmap>      (bit i = holds range i; the
//!                    same compact summary destined for ENR metadata + gossip)
//!   GETR <i>      -> DATA <i> len=<l> sha256=<hex>\n<raw bytes> | ERR not_held
//!   STRIPE <g> <p> -> PDATA <g> <p> len=<l>\n<raw bytes> | ERR not_held
//!                    (parity row p of range group g; stripe.zig. Served only
//!                    by full-group holders; ranges zero-padded to the group
//!                    max. Propagation plane only -- never the token loop.)
//!   -- bootnode (when a committee registry is attached; SPEC.md) --
//!   JOIN addr=<h:p> fraction=<f>
//!                 -> COMMITTEE id=<n> members=<a,b,...> assign=<hex bitmap>
//!   COMMITTEES    -> COMMITTEES <n>\n then n summary lines (debug)
//!   -- gossip / peer exchange --
//!   GOSSIP addr=<h:p> version=<hex> holdings=<hex>
//!                 -> PEERS <n>\n then n lines "addr=.. version=.. holdings=.."
//!                    (the responder's own entry first, then its peer table;
//!                    the announcer is merged into the responder's table)
//!   TABLE         -> same as GOSSIP response but without announcing (debug)
//!   <other>       -> ERR unknown

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const checkpoint = @import("../engine/checkpoint.zig");
const hashmod = @import("../core/hash.zig");
const weights = @import("weights.zig");
const peers = @import("peers.zig");
const stats = @import("../core/stats.zig");
const bootnode = @import("bootnode.zig");
const rag_store = @import("../rag/store.zig");
const generator = @import("../node/generator.zig");
const wire = @import("wire.zig");
const sockopt = @import("../core/sockopt.zig");
const stripe = @import("stripe.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    entries: []const checkpoint.ExpertEntry,
    unique_bytes: u64,
    addr: []const u8,
    port: u16,
    /// Attached GGUF weight store, if this node participates in distribution.
    /// Reads are lock-free (immutable manifest, pread on a shared handle);
    /// holdings bits are atomic.
    store: ?*weights.Store = null,
    /// Dynamic peer table for gossip; when set, GOSSIP/TABLE are served.
    table: ?*peers.Table = null,
    /// Committee registry; when set, this node acts as a bootnode (JOIN).
    boot: ?*bootnode.Registry = null,
    /// This node's committee id (wire.NO_COMMITTEE when not in one).
    committee_id: u32 = 0xFFFF_FFFF,
    network_id: u64 = 0,
    rag: ?*rag_store.Store = null,
    rag_round: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// In-flight expert requests being served (the heartbeat load hint).
    load: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Append-only JSONL sink for opt-in alpha telemetry (--alpha-ingest).
    /// Null means this node declines METRICS lines. Writes serialize on
    /// metrics_mu; the file is capped at 256 MB and then refused.
    metrics_path: ?[]const u8 = null,
    metrics_mu: Io.Mutex = .init,
    /// Per-shard serve counts (allocated alongside the store). GETR bumps
    /// them; HEAT answers with the hottest ids so a syncing peer can fetch
    /// in usefulness order instead of index order.
    heat: ?[]std.atomic.Value(u32) = null,
    /// Generation surface for delegate-while-cold (the GEN command). All
    /// three are set together by the node once its engine choice is final;
    /// until gen_ready flips true, GEN answers ERR not_ready.
    gen: ?*generator.Generator = null,
    engine_lock: ?*Io.Mutex = null,
    gen_ready: ?*std.atomic.Value(bool) = null,
    /// Our own advertised "host:port" (what we tell peers to dial us on).
    advertise: []const u8 = "",
};

fn selfLine(ctx: *Ctx, gpa: std.mem.Allocator, list: *std.ArrayList(u8)) !void {
    const zero_version = "0" ** 64;
    if (ctx.store) |st| {
        const vhex = hashmod.toHex(st.manifest.version);
        const hhex = try st.holdings.toHex(gpa);
        defer gpa.free(hhex);
        try list.print(gpa, "addr={s} version={s} holdings={s}\n", .{ ctx.advertise, vhex, hhex });
    } else {
        try list.print(gpa, "addr={s} version={s} holdings=\n", .{ ctx.advertise, zero_version });
    }
}

fn sendPeerList(ctx: *Ctx, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;
    const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
    var body = std.ArrayList(u8).empty;
    defer body.deinit(gpa);
    try selfLine(ctx, gpa, &body);
    const n = try table.dump(gpa, &body);
    try wi.print("PEERS {d}\n", .{n + 1});
    try wi.writeAll(body.items);
}

const Conn = struct { ctx: *Ctx, stream: net.Stream };

/// Bound on concurrent connection-handler threads (audit #7 P1 thread-per-
/// accept DoS). Excess connections are refused (closed) rather than spawning
/// unbounded threads.
const MAX_CONNS: u32 = 256;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.addr, ctx.port);
    var server = try address.listen(ctx.io, .{ .reuse_address = true });
    sockopt.ensureReaper(ctx.io);
    defer server.deinit(ctx.io);

    while (true) {
        const stream = server.accept(ctx.io) catch continue;
        if (live_conns.fetchAdd(1, .monotonic) >= MAX_CONNS) {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io); // over capacity — shed load
            continue;
        }
        const conn = ctx.gpa.create(Conn) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            continue;
        };
        conn.* = .{ .ctx = ctx, .stream = stream };
        const t = std.Thread.spawn(.{}, connThread, .{conn}) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            ctx.gpa.destroy(conn);
            continue;
        };
        t.detach();
    }
}

fn connThread(conn: *Conn) void {
    handleConn(conn.ctx, conn.stream) catch {};
    conn.ctx.gpa.destroy(conn);
    _ = live_conns.fetchSub(1, .monotonic);
}

fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    const dl = sockopt.trackServe(ctx.io, stream);
    defer sockopt.untrack(ctx.io, dl);
    // GLM-scale holdings bitmaps make GOSSIP/HOLDINGS lines ~5 KB; size the
    // line buffers well past that
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    while (true) {
        const raw = ri.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        try handleLine(ctx, line, ri, wi, dl);
        try wi.flush();
        // The deadline is there to hang up on a socket that is *idle* -- a
        // slowloris holding a connection open. A peer that just took a shard
        // is the opposite of idle, so push it out, exactly as the HTTP server
        // does between tokens.
        //
        // Without this the whole connection lived inside one 30 s budget,
        // which is invisible on loopback (a full sync finishes in seconds) and
        // fatal over a real link: a 905-shard bootstrap is ~5.4 GB, about 190 s
        // at the 29 MB/s measured Mac-to-Hetzner, so the server shut the socket
        // down mid-transfer and the client reported EndOfStream.
        sockopt.refreshServe(ctx.io, dl);
    }
}

fn handleLine(ctx: *Ctx, line: []const u8, ri: *Io.Reader, wi: *Io.Writer, dl: ?usize) !void {
    if (line.len == 0) return;
    if (std.mem.eql(u8, line, "PING")) {
        try wi.print("PONG\n", .{});
    } else if (std.mem.eql(u8, line, "HELLO")) {
        try wi.print("LOOM/0 experts={d} unique_bytes={d}\n", .{ ctx.entries.len, ctx.unique_bytes });
    } else if (std.mem.eql(u8, line, "COUNT")) {
        try wi.print("{d}\n", .{ctx.entries.len});
    } else if (std.mem.startsWith(u8, line, "HAS ")) {
        const arg = std.mem.trim(u8, line[4..], " ");
        const id = std.fmt.parseInt(usize, arg, 10) catch {
            try wi.print("ERR bad_id\n", .{});
            return;
        };
        if (id >= ctx.entries.len) {
            try wi.print("ERR range\n", .{});
            return;
        }
        const e = ctx.entries[id];
        const hex = hashmod.toHex(e.digest);
        try wi.print("PRESENT {d} off={d} len={d} sha256={s}\n", .{ id, e.offset, e.len, hex });
    } else if (std.mem.startsWith(u8, line, "METRICS ")) {
        // Opt-in alpha telemetry ingest (SPEC.md). Only persisted when the
        // operator passed --alpha-ingest; every other node declines, so a
        // reporter never accumulates state on peers that did not ask for it.
        const path = ctx.metrics_path orelse return wi.print("ERR no_ingest\n", .{});
        const json = line["METRICS ".len..];
        if (json.len == 0 or json.len > 8192) return wi.print("ERR bad_metrics\n", .{});
        appendMetricsLine(ctx, path, json) catch return wi.print("ERR ingest_failed\n", .{});
        try wi.print("OK\n", .{});
    } else if (std.mem.startsWith(u8, line, "DRAFT ")) {
        // DSD verification (SPEC.md): greedy-check a drafted window against
        // the exact model. Same trust plane and gating as GEN.
        const g = ctx.gen orelse return wi.print("ERR no_engine\n", .{});
        const lock = ctx.engine_lock orelse return wi.print("ERR no_engine\n", .{});
        const ready = ctx.gen_ready orelse return wi.print("ERR no_engine\n", .{});
        if (!ready.load(.acquire)) return wi.print("ERR not_ready\n", .{});
        const gg = switch (g.*) {
            .gguf => |p| p,
            else => return wi.print("ERR no_engine\n", .{}),
        };
        const parsed = std.json.parseFromSlice(std.json.Value, ctx.gpa, line[6..], .{}) catch
            return wi.print("ERR bad_json\n", .{});
        defer parsed.deinit();
        if (parsed.value != .object) return wi.print("ERR bad_json\n", .{});
        const obj = parsed.value.object;
        const ctx_toks = jsonTokenArray(ctx.gpa, obj.get("ctx"), 32768) catch
            return wi.print("ERR bad_draft\n", .{});
        defer ctx.gpa.free(ctx_toks);
        const draft_toks = jsonTokenArray(ctx.gpa, obj.get("draft"), 8) catch
            return wi.print("ERR bad_draft\n", .{});
        defer ctx.gpa.free(draft_toks);
        if (ctx_toks.len == 0 or draft_toks.len == 0) return wi.print("ERR bad_draft\n", .{});
        // a verify window is seconds of work, but a cold-cache re-prefill of a
        // long context is not: use the generation deadline
        sockopt.refreshServe(ctx.io, dl);
        lock.lockUncancelable(ctx.io);
        const v = generator.verifyDraft(gg, ctx.gpa, ctx_toks, draft_toks) catch |e| {
            lock.unlock(ctx.io);
            return wi.print("ERR draft_{s}\n", .{@errorName(e)});
        };
        lock.unlock(ctx.io);
        try wi.print("DRAFTR ok=1 accepted={d} correction={d}\n", .{ v.accepted, v.correction });
    } else if (std.mem.startsWith(u8, line, "GEN ")) {
        // Delegate-while-cold (SPEC.md): a cold peer forwards a generation
        // here and relays the answer. Same trust plane as every v1 command.
        const g = ctx.gen orelse return wi.print("ERR no_engine\n", .{});
        const lock = ctx.engine_lock orelse return wi.print("ERR no_engine\n", .{});
        const ready = ctx.gen_ready orelse return wi.print("ERR no_engine\n", .{});
        if (!ready.load(.acquire)) return wi.print("ERR not_ready\n", .{});
        const parsed = std.json.parseFromSlice(std.json.Value, ctx.gpa, line[4..], .{}) catch
            return wi.print("ERR bad_json\n", .{});
        defer parsed.deinit();
        if (parsed.value != .object) return wi.print("ERR bad_json\n", .{});
        const obj = parsed.value.object;
        const prompt = if (obj.get("prompt")) |v| (if (v == .string) v.string else "") else "";
        if (prompt.len == 0 or prompt.len > 65536) return wi.print("ERR bad_prompt\n", .{});
        var max_tokens: usize = 128;
        if (obj.get("max_tokens")) |v| {
            if (v == .integer and v.integer > 0) max_tokens = @min(@as(usize, @intCast(v.integer)), 2048);
        }
        var temp: f32 = 0.7;
        if (obj.get("temperature")) |v| {
            if (v == .float) temp = @floatCast(v.float);
            if (v == .integer) temp = @floatFromInt(v.integer);
        }
        var seed: u64 = 42;
        if (obj.get("seed")) |v| {
            if (v == .integer and v.integer >= 0) seed = @intCast(v.integer);
        }
        const parse_special = if (obj.get("parse_special")) |v| v == .bool and v.bool else false;
        // A generation is minutes of work, not a read: switch this
        // connection to the generation deadline for the duration.
        sockopt.refreshServe(ctx.io, dl);
        const t0 = stats.nowNs(ctx.io);
        lock.lockUncancelable(ctx.io);
        var res = g.generate(ctx.gpa, ctx.io, prompt, max_tokens, temp, seed, null, null, parse_special) catch |e| {
            lock.unlock(ctx.io);
            return wi.print("ERR gen_{s}\n", .{@errorName(e)});
        };
        const hit = g.hitRate();
        lock.unlock(ctx.io);
        defer res.deinit(ctx.gpa);
        const secs = @as(f64, @floatFromInt(stats.nowNs(ctx.io) - t0)) / 1e9;
        const tok_s = if (secs > 0) @as(f64, @floatFromInt(res.completion_tokens)) / secs else 0;
        try wi.print("GENR ok=1 prompt_tokens={d} completion_tokens={d} tok_per_s={d:.2} hit_rate={d:.4} len={d}\n", .{
            res.prompt_tokens, res.completion_tokens, tok_s, hit, res.text.len,
        });
        try wi.writeAll(res.text);
    } else if (std.mem.startsWith(u8, line, "PACKR ")) {
        // Batched shard stream: one request, many DATA payloads, one END.
        // Per-shard GETR pays a round trip each; at WAN RTTs that is minutes
        // of pure latency over a full sync. Same verification story as GETR:
        // the receiver checks every shard against its manifest digests.
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        var ids_buf: [256]usize = undefined;
        var n_ids: usize = 0;
        var it = std.mem.splitScalar(u8, std.mem.trim(u8, line[6..], " "), ',');
        while (it.next()) |tok| {
            if (tok.len == 0) continue;
            if (n_ids >= ids_buf.len) return wi.print("ERR too_many\n", .{});
            ids_buf[n_ids] = std.fmt.parseInt(usize, tok, 10) catch return wi.print("ERR bad_id\n", .{});
            n_ids += 1;
        }
        for (ids_buf[0..n_ids]) |i| {
            if (i >= store.manifest.nRanges() or !store.holdings.has(i)) {
                try wi.print("ABSENT {d}\n", .{i});
                continue;
            }
            const buf = try ctx.gpa.alloc(u8, @intCast(store.manifest.rangeLen(i)));
            defer ctx.gpa.free(buf);
            const data = store.readRangeVerified(i, buf) catch {
                try wi.print("ABSENT {d}\n", .{i});
                continue;
            };
            if (ctx.heat) |h| _ = h[i].fetchAdd(1, .monotonic);
            try wi.print("DATA {d} len={d} sha256={s}\n", .{ i, data.len, hashmod.toHex(store.manifest.digests[i]) });
            try wi.writeAll(data);
            try wi.flush();
            // a multi-gigabyte stream outlives the read deadline many times
            sockopt.refreshServe(ctx.io, dl);
        }
        try wi.print("END\n", .{});
    } else if (std.mem.eql(u8, line, "HEAT")) {
        // The hottest shard ids this node has served, descending. Usage is
        // Zipfian, so a joiner that syncs these first is useful after a
        // fraction of the transfer. Best-effort: no counts yet means an
        // empty list, and an old peer answers ERR unknown -- the client
        // falls back to index order either way.
        const h = ctx.heat orelse return wi.print("HEAT n=0 ids=\n", .{});
        const cap = @min(h.len, 2048);
        const Pair = struct { id: usize, n: u32 };
        const pairs = try ctx.gpa.alloc(Pair, h.len);
        defer ctx.gpa.free(pairs);
        var used: usize = 0;
        for (h, 0..) |*c, i| {
            const n = c.load(.monotonic);
            if (n == 0) continue;
            pairs[used] = .{ .id = i, .n = n };
            used += 1;
        }
        std.mem.sort(Pair, pairs[0..used], {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.n > b.n;
            }
        }.lt);
        const k = @min(used, cap);
        try wi.print("HEAT n={d} ids=", .{k});
        for (pairs[0..k], 0..) |pr, j| {
            if (j != 0) try wi.print(",", .{});
            try wi.print("{d}", .{pr.id});
        }
        try wi.print("\n", .{});
    } else if (std.mem.eql(u8, line, "MANIFEST")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const m = &store.manifest;
        try wi.print("MANIFEST version={s} size={d} ranges={d} range_size={d} mode={s} resident={d}\n", .{
            hashmod.toHex(m.version), m.file_size,  m.nRanges(), m.range_size,
            @tagName(m.mode),         m.n_resident,
        });
    } else if (std.mem.eql(u8, line, "MANIFESTFILE")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const text = try store.manifest.serialize(ctx.gpa);
        defer ctx.gpa.free(text);
        try wi.print("MANIFESTFILE len={d}\n", .{text.len});
        try wi.writeAll(text);
    } else if (std.mem.startsWith(u8, line, "DIGEST ")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const i = std.fmt.parseInt(usize, std.mem.trim(u8, line[7..], " "), 10) catch {
            return wi.print("ERR bad_id\n", .{});
        };
        if (i >= store.manifest.nRanges()) return wi.print("ERR range\n", .{});
        try wi.print("DIGEST {d} {s}\n", .{ i, hashmod.toHex(store.manifest.digests[i]) });
    } else if (std.mem.eql(u8, line, "DIGESTS")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        try wi.print("DIGESTS {d}\n", .{store.manifest.nRanges()});
        for (store.manifest.digests) |d| try wi.print("{s}\n", .{hashmod.toHex(d)});
    } else if (std.mem.eql(u8, line, "HOLDINGS")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const hex = try store.holdings.toHex(ctx.gpa);
        defer ctx.gpa.free(hex);
        try wi.print("HOLDINGS {s}\n", .{hex});
    } else if (std.mem.startsWith(u8, line, "GETR ")) {
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const i = std.fmt.parseInt(usize, std.mem.trim(u8, line[5..], " "), 10) catch {
            return wi.print("ERR bad_id\n", .{});
        };
        if (i >= store.manifest.nRanges()) return wi.print("ERR range\n", .{});
        if (!store.holdings.has(i)) return wi.print("ERR not_held\n", .{});
        const buf = try ctx.gpa.alloc(u8, @intCast(store.manifest.rangeLen(i)));
        defer ctx.gpa.free(buf);
        const data = store.readRangeVerified(i, buf) catch return wi.print("ERR read\n", .{});
        if (ctx.heat) |h| _ = h[i].fetchAdd(1, .monotonic);
        try wi.print("DATA {d} len={d} sha256={s}\n", .{ i, data.len, hashmod.toHex(store.manifest.digests[i]) });
        try wi.writeAll(data);
    } else if (std.mem.startsWith(u8, line, "STRIPE ")) {
        // Parity piece for a range group (propagation plane; stripe.zig).
        // Parity is a linear mix of every data range in the group, so only a
        // full-group holder can serve it -- a partial holder answers not_held
        // and the client falls back to plain GETR. Parity is computed on the
        // fly from verified reads; deterministic Cauchy coefficients mean
        // every honest holder produces byte-identical pieces. The receiver's
        // digest check on the reconstructed ranges is the integrity story;
        // no separate parity commitment exists.
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        var pit = std.mem.tokenizeScalar(u8, line[7..], ' ');
        const g_s = pit.next() orelse return wi.print("ERR bad_id\n", .{});
        const p_s = pit.next() orelse return wi.print("ERR bad_id\n", .{});
        const g = std.fmt.parseInt(usize, g_s, 10) catch return wi.print("ERR bad_id\n", .{});
        const p = std.fmt.parseInt(usize, p_s, 10) catch return wi.print("ERR bad_id\n", .{});
        const n_ranges = store.manifest.nRanges();
        const first = g *| stripe.K_DATA;
        if (first >= n_ranges or p >= stripe.M_PARITY) return wi.print("ERR range\n", .{});
        const last = @min(first + stripe.K_DATA, n_ranges);
        var plen: usize = 0;
        for (first..last) |i| {
            if (!store.holdings.has(i)) return wi.print("ERR not_held\n", .{});
            plen = @max(plen, @as(usize, @intCast(store.manifest.rangeLen(i))));
        }
        const out = try ctx.gpa.alloc(u8, plen);
        defer ctx.gpa.free(out);
        @memset(out, 0);
        const buf = try ctx.gpa.alloc(u8, plen);
        defer ctx.gpa.free(buf);
        for (first..last) |i| {
            const rl: usize = @intCast(store.manifest.rangeLen(i));
            const data = store.readRangeVerified(i, buf[0..rl]) catch return wi.print("ERR read\n", .{});
            stripe.accumulate(stripe.coeff(p, i - first, stripe.M_PARITY), data, out);
        }
        try wi.print("PDATA {d} {d} len={d}\n", .{ g, p, plen });
        try wi.writeAll(out);
    } else if (std.mem.startsWith(u8, line, "FRAME ")) {
        const len = std.fmt.parseInt(usize, line[6..], 10) catch return wi.print("ERR bad_frame\n", .{});
        // read the body in chunks so memory tracks bytes actually delivered
        // (security issue #27), not the length a peer merely claims
        const raw = wire.readFrameBodyAlloc(ctx.gpa, ri, len) catch return wi.print("ERR bad_frame\n", .{});
        defer ctx.gpa.free(raw);
        try handleFrame(ctx, raw, wi);
    } else if (std.mem.startsWith(u8, line, "JOIN ")) {
        const reg = ctx.boot orelse return wi.print("ERR no_bootnode\n", .{});
        // Rate-limit placement changes (security issue #144): JOIN is
        // unauthenticated by design in v1, and each call creates or grows a
        // committee, so an unthrottled dialer can inflate placement state
        // with unique addresses. A token bucket bounds the damage to a slow
        // drip without blocking a real swarm's joins (they are rare).
        if (!joinBucket.allow(stats.nowNs(ctx.io))) return wi.print("ERR join_rate_limited\n", .{});
        const store = ctx.store orelse return wi.print("ERR no_store\n", .{});
        const addr = fieldOf(line, "addr") orelse return wi.print("ERR bad_join\n", .{});
        const frac_s = fieldOf(line, "fraction") orelse return wi.print("ERR bad_join\n", .{});
        const frac = std.fmt.parseFloat(f32, frac_s) catch return wi.print("ERR bad_join\n", .{});
        const n_experts = store.manifest.nRanges() - store.manifest.n_resident;
        const capacity: usize = @intFromFloat(@max(1.0, @round(std.math.clamp(frac, 0.0, 1.0) * @as(f32, @floatFromInt(n_experts)))));
        var out = reg.join(addr, capacity) catch return wi.print("ERR join_failed\n", .{});
        defer out.deinit(ctx.gpa);
        const hex = try out.assign.toHex(ctx.gpa);
        defer ctx.gpa.free(hex);
        try wi.print("COMMITTEE id={d} members=", .{out.committee_id});
        for (out.members, 0..) |m, k| {
            if (k != 0) try wi.print(",", .{});
            try wi.print("{s}", .{m});
        }
        try wi.print(" assign={s}\n", .{hex});
    } else if (std.mem.eql(u8, line, "COMMITTEES")) {
        const reg = ctx.boot orelse return wi.print("ERR no_bootnode\n", .{});
        var body = std.ArrayList(u8).empty;
        defer body.deinit(ctx.gpa);
        const n = try reg.summary(ctx.gpa, &body);
        try wi.print("COMMITTEES {d}\n", .{n});
        try wi.writeAll(body.items);
    } else if (std.mem.startsWith(u8, line, "GOSSIP ")) {
        const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
        const addr = fieldOf(line, "addr") orelse return wi.print("ERR bad_gossip\n", .{});
        const version = fieldOf(line, "version") orelse return wi.print("ERR bad_gossip\n", .{});
        const holdings = fieldOf(line, "holdings") orelse "";
        // inbound: this peer opened the connection, so it is demonstrably alive
        _ = table.merge(addr, version, holdings, peers.NO_COMMITTEE, 0, stats.nowNs(ctx.io), .first_hand) catch {
            return wi.print("ERR bad_gossip\n", .{});
        };
        try sendPeerList(ctx, wi);
    } else if (std.mem.eql(u8, line, "TABLE")) {
        try sendPeerList(ctx, wi);
    } else {
        try wi.print("ERR unknown\n", .{});
    }
}

/// Dispatch a decoded wire frame (SPEC.md wire messages v1).
fn handleFrame(ctx: *Ctx, raw: []const u8, wi: *Io.Writer) !void {
    const dec = wire.decodeFrame(ctx.gpa, raw) catch return wi.print("ERR bad_frame\n", .{});
    defer ctx.gpa.free(dec.body);
    switch (dec.ty) {
        .heartbeat => {
            // A heartbeat from another LLM network is refused outright --
            // the caller learns it dialed the wrong network instead of
            // silently exchanging state across it.
            if (wire.Heartbeat.parseBody(dec.body)) |hb| {
                if (hb.network_id != ctx.network_id) return wi.print("ERR wrong_network\n", .{});
            } else |_| return wi.print("ERR bad_frame\n", .{});
            // respond with our own state: one exchange refreshes both sides
            const resp = selfHeartbeat(ctx);
            const body = try resp.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .heartbeat_resp, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .announce => {
            const table = ctx.table orelse return wi.print("ERR no_gossip\n", .{});
            const ann = wire.Announce.parseBody(dec.body) catch return wi.print("ERR bad_frame\n", .{});
            if (ann.network_id != ctx.network_id) return wi.print("ERR wrong_network\n", .{});
            const vhex = hashmod.toHex(ann.manifest_version);
            const hhex = try bytesToHexAlloc(ctx.gpa, ann.holdings_bitmap);
            defer ctx.gpa.free(hhex);
            _ = table.merge(ann.addr, &vhex, hhex, ann.committee_id, ann.holdings_seq, stats.nowNs(ctx.io), .first_hand) catch {};
            try sendAnnounceBatch(ctx, wi);
        },
        .rag_inv => {
            const st = ctx.rag orelse return wi.print("ERR no_rag\n", .{});
            const inv = wire.RagHashes.parseBody(ctx.gpa, dec.body) catch return wi.print("ERR bad_frame\n", .{});
            defer ctx.gpa.free(inv);
            var want = std.ArrayList([32]u8).empty;
            defer want.deinit(ctx.gpa);
            for (inv) |h| {
                if (want.items.len >= wire.RAG_PUSH_MAX) break;
                if (!st.has(h)) try want.append(ctx.gpa, h);
            }
            var resp = wire.RagHashes{ .hashes = want.items };
            const body = try resp.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_want, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .rag_push => {
            const st = ctx.rag orelse return;
            var it = wire.RagPush.iterate(dec.body) catch return;
            while (it.next() catch null) |text| _ = st.add(text);
        },
        .rag_inv_req => {
            const st = ctx.rag orelse return wi.print("ERR no_rag\n", .{});
            var hashes: [wire.RAG_INV_MAX][32]u8 = undefined;
            const round = ctx.rag_round.fetchAdd(1, .monotonic);
            const n = st.invWindow(round, hashes[0..]);
            var inv = wire.RagHashes{ .hashes = hashes[0..n] };
            const body = try inv.encodeBody(ctx.gpa);
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_inv, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .rag_want => {
            // A dialer asking US for chunks it lacks (the pull direction; a
            // NAT'd node can only ever dial out).
            const st = ctx.rag orelse return;
            const wanted = wire.RagHashes.parseBody(ctx.gpa, dec.body) catch return;
            defer ctx.gpa.free(wanted);
            var texts = std.ArrayList([]u8).empty;
            defer {
                for (texts.items) |t| ctx.gpa.free(t);
                texts.deinit(ctx.gpa);
            }
            for (wanted) |h| {
                if (texts.items.len >= wire.RAG_PUSH_MAX) break;
                if (st.textByHashAlloc(ctx.gpa, h)) |t| try texts.append(ctx.gpa, t);
            }
            const body = try wire.RagPush.encodeBody(ctx.gpa, @ptrCast(texts.items));
            defer ctx.gpa.free(body);
            const frame = try wire.encodeFrame(ctx.gpa, .rag_push, body);
            defer ctx.gpa.free(frame);
            try wire.writeFrame(wi, frame);
        },
        .expert_request => {
            const req = wire.ExpertRequest.parseBody(dec.body) catch {
                return sendExpertResponse(ctx, wi, .{ .request_id = 0, .status = .bad_request, .shard_id = 0 });
            };
            const store = ctx.store orelse {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            };
            if (!std.mem.eql(u8, &req.manifest_version, &store.manifest.version)) {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .version_mismatch, .shard_id = req.shard_id });
            }
            if (req.shard_id >= store.manifest.nRanges() or !store.holdings.has(req.shard_id)) {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            }
            _ = ctx.load.fetchAdd(1, .monotonic);
            defer _ = ctx.load.fetchSub(1, .monotonic);
            const buf = try ctx.gpa.alloc(u8, @intCast(store.manifest.rangeLen(req.shard_id)));
            defer ctx.gpa.free(buf);
            const data = store.readRangeVerified(req.shard_id, buf) catch {
                return sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .not_held, .shard_id = req.shard_id });
            };
            try sendExpertResponse(ctx, wi, .{ .request_id = req.request_id, .status = .ok, .shard_id = req.shard_id, .payload = data });
        },
        else => try wi.print("ERR unexpected_frame\n", .{}),
    }
}

/// Our own heartbeat/announce state snapshot.
pub fn selfHeartbeat(ctx: *Ctx) wire.Heartbeat {
    var hb = wire.Heartbeat{
        .network_id = ctx.network_id,
        .committee_id = ctx.committee_id,
        .load = @intCast(@min(ctx.load.load(.monotonic), std.math.maxInt(u16))),
        .sent_at_ns = @intCast(@mod(stats.nowNs(ctx.io), std.math.maxInt(i64))),
        .addr = ctx.advertise,
    };
    if (ctx.store) |st| {
        hb.manifest_version = st.manifest.version;
        hb.holdings_seq = st.holdingsSeq(); // truly monotonic (audit #7 P1)
        // hash a consistent snapshot, not a bitmap being mutated underneath
        // (security issue #31): otherwise the advertised digest can describe a
        // state that never existed
        var buf: [4096]u8 = undefined;
        if (st.holdings.bits.len <= buf.len) {
            for (st.holdings.bits, 0..) |*bp, i| buf[i] = @atomicLoad(u8, bp, .monotonic);
            hb.holdings_digest = hashmod.hashBlock(buf[0..st.holdings.bits.len]);
        } else if (st.holdings.snapshotAlloc(ctx.gpa)) |snap| {
            defer ctx.gpa.free(snap);
            hb.holdings_digest = hashmod.hashBlock(snap);
        } else |_| {}
    }
    return hb;
}

fn appendMetricsLine(ctx: *Ctx, path: []const u8, json: []const u8) !void {
    ctx.metrics_mu.lockUncancelable(ctx.io);
    defer ctx.metrics_mu.unlock(ctx.io);
    const f = Io.Dir.cwd().openFile(ctx.io, path, .{ .mode = .write_only }) catch
        try Io.Dir.cwd().createFile(ctx.io, path, .{});
    defer f.close(ctx.io);
    const size = (try f.stat(ctx.io)).size;
    if (size > 256 * 1024 * 1024) return error.IngestFull;
    try f.writePositionalAll(ctx.io, json, size);
    try f.writePositionalAll(ctx.io, "\n", size + json.len);
}

fn bytesToHexAlloc(gpa: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const hex = "0123456789abcdef";
    const out = try gpa.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |b, i| {
        out[i * 2] = hex[b >> 4];
        out[i * 2 + 1] = hex[b & 0xf];
    }
    return out;
}

fn hexToBytesAlloc(gpa: std.mem.Allocator, hexs: []const u8) ![]u8 {
    if (hexs.len % 2 != 0) return error.BadHex;
    const out = try gpa.alloc(u8, hexs.len / 2);
    errdefer gpa.free(out);
    for (out, 0..) |*b, i| {
        b.* = std.fmt.parseInt(u8, hexs[i * 2 ..][0..2], 16) catch return error.BadHex;
    }
    return out;
}

/// Respond to a gossip announce: our own entry + the whole table as an
/// AnnounceBatch frame (snappy compresses across the similar bitmaps).
fn sendAnnounceBatch(ctx: *Ctx, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;
    var bodies = std.ArrayList([]u8).empty;
    defer {
        for (bodies.items) |b| gpa.free(b);
        bodies.deinit(gpa);
    }

    // self entry
    {
        var self_snap: ?[]u8 = null;
        defer if (self_snap) |sp| gpa.free(sp);
        var self_ann = wire.Announce{ .network_id = ctx.network_id, .committee_id = ctx.committee_id, .addr = ctx.advertise };
        if (ctx.store) |st| {
            self_ann.manifest_version = st.manifest.version;
            self_ann.holdings_seq = @intCast(st.holdings.count());
            self_snap = try st.holdings.snapshotAlloc(gpa);
            self_ann.holdings_bitmap = self_snap.?;
        }
        try bodies.append(gpa, try self_ann.encodeBody(gpa));
    }

    // table entries
    const table = ctx.table.?;
    {
        table.mutex.lockUncancelable(ctx.io);
        defer table.mutex.unlock(ctx.io);
        for (table.entries.items) |e| {
            var ver: [32]u8 = undefined;
            for (&ver, 0..) |*b, i| {
                b.* = std.fmt.parseInt(u8, e.version_hex[i * 2 ..][0..2], 16) catch 0;
            }
            const bitmap = hexToBytesAlloc(gpa, e.holdings_hex) catch continue;
            defer gpa.free(bitmap);
            // Table entries share our network_id by construction: both merge
            // surfaces refuse a record from another network before it can
            // enter the table.
            const ann = wire.Announce{
                .network_id = ctx.network_id,
                .committee_id = e.committee_id,
                .manifest_version = ver,
                .addr = e.addr,
                .holdings_bitmap = bitmap,
            };
            try bodies.append(gpa, try ann.encodeBody(gpa));
        }
    }

    const batch = try wire.AnnounceBatch.encodeBodies(gpa, bodies.items);
    defer gpa.free(batch);
    const frame = try wire.encodeFrame(gpa, .announce_batch, batch);
    defer gpa.free(frame);
    try wire.writeFrame(wi, frame);
}

fn sendExpertResponse(ctx: *Ctx, wi: *Io.Writer, resp: wire.ExpertResponse) !void {
    const body = try resp.encodeBody(ctx.gpa);
    defer ctx.gpa.free(body);
    const frame = try wire.encodeFrame(ctx.gpa, .expert_response, body);
    defer ctx.gpa.free(frame);
    try wire.writeFrame(wi, frame);
}

fn fieldOf(line: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key) and tok.len > key.len and tok[key.len] == '=') {
            return tok[key.len + 1 ..];
        }
    }
    return null;
}

/// Token bucket for unauthenticated JOINs (security issue #144). Process-wide
/// rather than per-connection, because the attack is many short connections.
const JoinBucket = struct {
    const CAPACITY: i64 = 32; // burst: a fleet restarting at once
    const REFILL_NS: i64 = std.time.ns_per_s; // one token per second

    /// Milli-tokens x 1000 held as an atomic, plus the last refill stamp.
    /// A benign race only mis-refills by a tick, which a rate limiter can
    /// absorb; the point is the bound, not exactness.
    tokens: std.atomic.Value(i64) = std.atomic.Value(i64).init(CAPACITY),
    last_ns: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),

    fn allow(self: *JoinBucket, now_i128: i128) bool {
        const now: i64 = @truncate(now_i128);
        const last = self.last_ns.load(.monotonic);
        if (last == 0) {
            self.last_ns.store(now, .monotonic);
        } else if (now > last) {
            const gained = @divTrunc(now - last, REFILL_NS);
            if (gained > 0) {
                self.last_ns.store(now, .monotonic);
                const cur = self.tokens.load(.monotonic);
                self.tokens.store(@min(CAPACITY, cur + gained), .monotonic);
            }
        }
        return self.tokens.fetchSub(1, .monotonic) > 0;
    }
};
var joinBucket: JoinBucket = .{};

/// Parse a JSON array of token ids with a length cap; any non-integer,
/// negative, or oversized input is an error (the caller answers ERR).
fn jsonTokenArray(gpa: std.mem.Allocator, v: ?std.json.Value, max: usize) ![]u32 {
    const arr = switch (v orelse return error.Bad) {
        .array => |a| a,
        else => return error.Bad,
    };
    if (arr.items.len > max) return error.Bad;
    const out = try gpa.alloc(u32, arr.items.len);
    errdefer gpa.free(out);
    for (arr.items, 0..) |it, i| {
        if (it != .integer or it.integer < 0 or it.integer > std.math.maxInt(u32)) return error.Bad;
        out[i] = @intCast(it.integer);
    }
    return out;
}

test "announce batch entries carry the responder's network_id" {
    // The regression this pins (observed live on devnet): proto v2 gave
    // Announce a network_id and gossip.exchange a merge-side filter, but
    // sendAnnounceBatch left the field at its zero default on every entry it
    // returned. On any network with a nonzero id the dialer discarded the
    // entire response, so a dial-out-only (NAT'd) node could never refresh a
    // peer's liveness: its announces arrived (the bootnode showed "peers 1")
    // while its own table decayed past PEER_TTL_NS and stayed at "peers 0",
    // leaving warmest()/holdersOf() empty-handed for good.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var table = peers.Table.init(gpa, io, "1.2.3.4:8771");
    defer table.deinit();
    _ = try table.merge("5.6.7.8:8771", "0" ** 64, "ff", peers.NO_COMMITTEE, 1, stats.nowNs(io), .first_hand);

    var ctx = Ctx{
        .gpa = gpa,
        .io = io,
        .entries = &.{},
        .unique_bytes = 0,
        .addr = "127.0.0.1",
        .port = 0,
        .table = &table,
        .network_id = 1337,
        .advertise = "1.2.3.4:8771",
    };

    var buf: [1 << 16]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try sendAnnounceBatch(&ctx, &w);

    var r: Io.Reader = .fixed(w.buffered());
    const raw = try wire.readFrameAlloc(gpa, &r);
    defer gpa.free(raw);
    const dec = try wire.decodeFrame(gpa, raw);
    defer gpa.free(dec.body);
    try std.testing.expect(dec.ty == .announce_batch);

    // every entry must survive the dialer's `network_id != ours` filter
    var survivors: usize = 0;
    var it = try wire.AnnounceBatch.iterate(dec.body);
    while (try it.next()) |entry_body| {
        const e = try wire.Announce.parseBody(entry_body);
        try std.testing.expectEqual(@as(u64, 1337), e.network_id);
        survivors += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), survivors); // self + the table entry
}

const gguf_fixture = @import("../gguf/gguf.zig");

test "STRIPE serves decodable parity; partial holders decline" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const model_path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    try gguf_fixture.writeMoeFixture(gpa, io, model_path, 3, "llama");
    var sbuf: [std.fs.max_path_bytes]u8 = undefined;
    const store_dir = try std.fmt.bufPrint(&sbuf, ".zig-cache/tmp/{s}/store", .{tmp.sub_path});

    var store = try weights.openFull(gpa, io, model_path, store_dir, 0, null);
    defer store.deinit();
    const n_ranges = store.manifest.nRanges();
    const k = @min(stripe.K_DATA, n_ranges); // group 0

    var ctx = Ctx{
        .gpa = gpa,
        .io = io,
        .entries = &.{},
        .unique_bytes = 0,
        .addr = "127.0.0.1",
        .port = 0,
        .store = &store,
    };

    // the group's originals, zero-padded to the group max, straight from disk
    var plen: usize = 0;
    for (0..k) |i| plen = @max(plen, @as(usize, @intCast(store.manifest.rangeLen(i))));
    var originals: [stripe.K_DATA][]u8 = undefined;
    var n_orig: usize = 0;
    defer for (originals[0..n_orig]) |b| gpa.free(b);
    for (0..k) |i| {
        const b = try gpa.alloc(u8, plen);
        originals[n_orig] = b;
        n_orig += 1;
        const rl: usize = @intCast(store.manifest.rangeLen(i));
        _ = try store.readRangeVerified(i, b[0..rl]);
        @memset(b[rl..], 0);
    }

    // fetch both parity rows through the wire handler
    var parity: [stripe.M_PARITY][]u8 = undefined;
    var n_par: usize = 0;
    defer for (parity[0..n_par]) |b| gpa.free(b);
    const wbuf = try gpa.alloc(u8, plen + 128);
    defer gpa.free(wbuf);
    for (0..stripe.M_PARITY) |p| {
        var w: Io.Writer = .fixed(wbuf);
        var r: Io.Reader = .fixed("");
        var lbuf: [32]u8 = undefined;
        const line = try std.fmt.bufPrint(&lbuf, "STRIPE 0 {d}", .{p});
        try handleLine(&ctx, line, &r, &w, null);
        var resp: Io.Reader = .fixed(w.buffered());
        const header = try resp.takeDelimiterInclusive('\n');
        var hbuf: [64]u8 = undefined;
        const want = try std.fmt.bufPrint(&hbuf, "PDATA 0 {d} len={d}\n", .{ p, plen });
        try std.testing.expectEqualStrings(want, header);
        const b = try gpa.alloc(u8, plen);
        parity[n_par] = b;
        n_par += 1;
        try resp.readSliceAll(b);
        // parity must be the deterministic Cauchy mix of the originals
        const expect = try gpa.alloc(u8, plen);
        defer gpa.free(expect);
        var data_const: [stripe.K_DATA][]const u8 = undefined;
        for (0..k) |i| data_const[i] = originals[i];
        stripe.encodeRow(p, stripe.M_PARITY, data_const[0..k], expect);
        try std.testing.expectEqualSlices(u8, expect, b);
    }

    // client half: erase up to M_PARITY data pieces, decode from the rest
    const n_drop = @min(stripe.M_PARITY, k);
    var pieces: [stripe.K_DATA + stripe.M_PARITY]stripe.Piece = undefined;
    var n_pieces: usize = 0;
    for (n_drop..k) |i| {
        pieces[n_pieces] = .{ .index = i, .bytes = originals[i] };
        n_pieces += 1;
    }
    for (0..n_drop) |p| {
        pieces[n_pieces] = .{ .index = k + p, .bytes = parity[p] };
        n_pieces += 1;
    }
    var rebuilt: [stripe.K_DATA][]u8 = undefined;
    var n_reb: usize = 0;
    defer for (rebuilt[0..n_reb]) |b| gpa.free(b);
    var outs: [stripe.K_DATA][]u8 = undefined;
    for (0..k) |i| {
        rebuilt[i] = try gpa.alloc(u8, plen);
        n_reb += 1;
        outs[i] = rebuilt[i];
    }
    try stripe.decode(k, stripe.M_PARITY, pieces[0..n_pieces], outs[0..k]);
    for (0..k) |i| try std.testing.expectEqualSlices(u8, originals[i], rebuilt[i]);

    // a partial holder must decline: parity needs the whole group
    store.holdings.clear(0);
    {
        var ebuf: [256]u8 = undefined;
        var w: Io.Writer = .fixed(&ebuf);
        var r: Io.Reader = .fixed("");
        try handleLine(&ctx, "STRIPE 0 0", &r, &w, null);
        try std.testing.expectEqualStrings("ERR not_held\n", w.buffered());
    }
    // out-of-range parity row and group
    store.holdings.set(0);
    {
        var ebuf: [256]u8 = undefined;
        var w: Io.Writer = .fixed(&ebuf);
        var r: Io.Reader = .fixed("");
        try handleLine(&ctx, "STRIPE 0 99", &r, &w, null);
        try std.testing.expectEqualStrings("ERR range\n", w.buffered());
        var w2: Io.Writer = .fixed(&ebuf);
        try handleLine(&ctx, "STRIPE 999999 0", &r, &w2, null);
        try std.testing.expectEqualStrings("ERR range\n", w2.buffered());
    }
}
