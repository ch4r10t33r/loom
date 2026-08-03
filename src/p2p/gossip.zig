//! Gossip loop (ROADMAP #7 / SPEC.md wire messages v1): on a short interval,
//! dial every peer in the table, send our Announce frame (addr, committee id,
//! manifest version, holdings seq + bitmap — snappy'd at the frame level), and
//! merge the AnnounceBatch that comes back (the responder's own entry + its
//! table). Discovery is transitive, and because announces carry committee_id,
//! the table doubles as the gossip-derived committee view: earlier committee
//! members learn later joiners from their announces.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const peers = @import("peers.zig");
const weights = @import("weights.zig");
const rag_store = @import("../rag/store.zig");
const sync = @import("sync.zig");
const dns = @import("dns.zig");
const sockopt = @import("../core/sockopt.zig");
const hashmod = @import("../core/hash.zig");
const stats = @import("../core/stats.zig");
const wire = @import("wire.zig");

pub const INTERVAL_NS: i96 = 3 * std.time.ns_per_s;

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    table: *peers.Table,
    store: ?*weights.Store,
    advertise: []const u8,
    committee_id: u32 = wire.NO_COMMITTEE,
    network_id: u64 = 0,
    rag: ?*rag_store.Store = null,
    rag_round: u64 = 0,
};

/// One gossip exchange with one peer: announce ourselves, merge their batch.
fn exchange(ctx: *Ctx, addr_str: []const u8) !void {
    const gpa = ctx.gpa;
    const io = ctx.io;
    const addr = try sync.PeerAddr.parse(addr_str);
    const ip = try dns.resolve(io, addr.host, addr.port);
    const stream = try ip.connect(io, .{ .mode = .stream });
    const dl = sockopt.trackPeer(io, stream);
    defer sockopt.untrack(io, dl);
    defer stream.close(io);

    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var w = stream.writer(io, &wbuf);

    // self announce (SPEC.md Announce container)
    var ann = wire.Announce{ .network_id = ctx.network_id, .committee_id = ctx.committee_id, .addr = ctx.advertise };
    // atomic snapshot: reading `bits` directly races the repair/fetch threads'
    // RMWs and advertises a torn bitmap (security issue #31)
    var snap: ?[]u8 = null;
    defer if (snap) |sp| gpa.free(sp);
    if (ctx.store) |st| {
        ann.manifest_version = st.manifest.version;
        ann.holdings_seq = st.holdingsSeq();
        snap = try st.holdings.snapshotAlloc(gpa);
        ann.holdings_bitmap = snap.?;
    }
    const body = try ann.encodeBody(gpa);
    defer gpa.free(body);
    const frame = try wire.encodeFrame(gpa, .announce, body);
    defer gpa.free(frame);
    try wire.writeFrame(&w.interface, frame);

    // merge their AnnounceBatch
    const resp_raw = try wire.readFrameAlloc(gpa, &r.interface);
    defer gpa.free(resp_raw);
    const dec = try wire.decodeFrame(gpa, resp_raw);
    defer gpa.free(dec.body);
    if (dec.ty != .announce_batch) return error.BadGossipResponse;

    const now = stats.nowNs(io);
    var it = try wire.AnnounceBatch.iterate(dec.body);
    while (try it.next()) |entry_body| {
        const e = wire.Announce.parseBody(entry_body) catch continue;
        // chainId semantics: a record from another LLM network never enters
        // the peer table, however it arrived.
        if (e.network_id != ctx.network_id) continue;
        const vhex = hashmod.toHex(e.manifest_version);
        const hhex = bytesToHexAlloc(gpa, e.holdings_bitmap) catch continue;
        defer gpa.free(hhex);
        // The dialed peer's own entry is proof it answered; the rest of its
        // batch is what it has heard from others.
        const ev: peers.Evidence = if (std.mem.eql(u8, e.addr, addr_str)) .first_hand else .hearsay;
        _ = ctx.table.merge(e.addr, &vhex, hhex, e.committee_id, e.holdings_seq, now, ev) catch continue;
    }

    // RAG piggyback: inventory -> want -> push, all on this same stream.
    // Text only; the peer recomputes embeddings with its own model.
    if (ctx.rag) |st| {
        if (st.count() > 0) {
            var hashes: [wire.RAG_INV_MAX][32]u8 = undefined;
            ctx.rag_round +%= 1;
            const n = st.invWindow(ctx.rag_round, hashes[0..]);
            var inv = wire.RagHashes{ .hashes = hashes[0..n] };
            const ibody = try inv.encodeBody(gpa);
            defer gpa.free(ibody);
            const iframe = try wire.encodeFrame(gpa, .rag_inv, ibody);
            defer gpa.free(iframe);
            try wire.writeFrame(&w.interface, iframe);

            const want_raw = try wire.readFrameAlloc(gpa, &r.interface);
            defer gpa.free(want_raw);
            const wdec = try wire.decodeFrame(gpa, want_raw);
            defer gpa.free(wdec.body);
            if (wdec.ty == .rag_want) {
                const wanted = wire.RagHashes.parseBody(gpa, wdec.body) catch return;
                defer gpa.free(wanted);
                var texts = std.ArrayList([]u8).empty;
                defer {
                    for (texts.items) |t| gpa.free(t);
                    texts.deinit(gpa);
                }
                for (wanted) |h| {
                    if (texts.items.len >= wire.RAG_PUSH_MAX) break;
                    if (st.textByHashAlloc(gpa, h)) |t| try texts.append(gpa, t);
                }
                if (texts.items.len > 0) {
                    const pbody = try wire.RagPush.encodeBody(gpa, @ptrCast(texts.items));
                    defer gpa.free(pbody);
                    const pframe = try wire.encodeFrame(gpa, .rag_push, pbody);
                    defer gpa.free(pframe);
                    try wire.writeFrame(&w.interface, pframe);
                }
            }

            // Pull direction: ask for the peer's window and fetch what we
            // lack. Without this a dial-out-only node (NAT) sends chunks but
            // never receives any.
            const qframe = try wire.encodeFrame(gpa, .rag_inv_req, "");
            defer gpa.free(qframe);
            try wire.writeFrame(&w.interface, qframe);
            const pinv_raw = try wire.readFrameAlloc(gpa, &r.interface);
            defer gpa.free(pinv_raw);
            const pdec = try wire.decodeFrame(gpa, pinv_raw);
            defer gpa.free(pdec.body);
            if (pdec.ty == .rag_inv) {
                const theirs = wire.RagHashes.parseBody(gpa, pdec.body) catch return;
                defer gpa.free(theirs);
                var lacks = std.ArrayList([32]u8).empty;
                defer lacks.deinit(gpa);
                for (theirs) |h| {
                    if (lacks.items.len >= wire.RAG_PUSH_MAX) break;
                    if (!st.has(h)) try lacks.append(gpa, h);
                }
                if (lacks.items.len > 0) {
                    var wnt = wire.RagHashes{ .hashes = lacks.items };
                    const wbody = try wnt.encodeBody(gpa);
                    defer gpa.free(wbody);
                    const wframe = try wire.encodeFrame(gpa, .rag_want, wbody);
                    defer gpa.free(wframe);
                    try wire.writeFrame(&w.interface, wframe);
                    const push_raw = try wire.readFrameAlloc(gpa, &r.interface);
                    defer gpa.free(push_raw);
                    const push_dec = try wire.decodeFrame(gpa, push_raw);
                    defer gpa.free(push_dec.body);
                    if (push_dec.ty == .rag_push) {
                        var it2 = wire.RagPush.iterate(push_dec.body) catch return;
                        while (it2.next() catch null) |text| _ = st.add(text);
                    }
                }
            }
        }
    }
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

/// The loop: every INTERVAL_NS, exchange with every currently known peer.
/// Unreachable peers are retried next round (eager-churn policy: never
/// forgotten).
pub fn loop(ctx: *Ctx) void {
    std.debug.print("gossip: loop started\n", .{});
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = INTERVAL_NS }, .awake) catch |e| {
            std.debug.print("gossip: sleep failed: {t}; loop exiting\n", .{e});
            return;
        };
        const addrs = ctx.table.snapshotAddrs(ctx.gpa) catch continue;
        if (addrs.len == 0) std.debug.print("gossip: empty snapshot (table has {d} entries)\n", .{ctx.table.count()});
        defer {
            for (addrs) |a| ctx.gpa.free(a);
            ctx.gpa.free(addrs);
        }
        for (addrs) |a| {
            exchange(ctx, a) catch |e| {
                std.debug.print("gossip: exchange with {s} failed: {t}\n", .{ a, e });
                continue;
            };
            std.debug.print("gossip: exchange with {s} ok\n", .{a});
        }
    }
}
