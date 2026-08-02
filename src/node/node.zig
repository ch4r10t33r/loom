//! `loom node` — a running inference node: load a model (local / synthetic /
//! Hugging Face), serve inference over RPC, and answer expert-directory queries
//! over P2P. This is the single-box daemon that v1 will federate into a cluster.

const std = @import("std");
const Io = std.Io;
const engine_mod = @import("../engine/engine.zig");
const Engine = engine_mod.Engine;
const hf = @import("hf.zig");
const rpc = @import("rpc.zig");
const dns = @import("../p2p/dns.zig");
const sockopt = @import("../core/sockopt.zig");
const openai = @import("openai.zig");
const backend = @import("../compute/backend.zig");
const generator = @import("generator.zig");
const rag_store = @import("../rag/store.zig");
const networks = @import("../p2p/networks.zig");
const alpha = @import("alpha.zig");
const net = std.Io.net;
const deepseek = @import("../gguf/deepseek.zig");
const llama = @import("../gguf/llama.zig");
const gguf_mod = @import("../gguf/gguf.zig");
const banner = @import("../core/banner.zig");
const status_mod = @import("status.zig");
const chat_template = @import("../gguf/chat_template.zig");
const expert_fetch = @import("../p2p/expert_fetch.zig");
const p2p = @import("../p2p/p2p.zig");
const stats = @import("../core/stats.zig");
const weights = @import("../p2p/weights.zig");
const sync = @import("../p2p/sync.zig");
const hashmod = @import("../core/hash.zig");
const peers = @import("../p2p/peers.zig");
const gossip = @import("../p2p/gossip.zig");
const bootnode = @import("../p2p/bootnode.zig");
const wire = @import("../p2p/wire.zig");
const meter_mod = @import("meter.zig");

const GB: f64 = 1024.0 * 1024.0 * 1024.0;
const MBf: f64 = 1024.0 * 1024.0;

pub const Options = struct {
    model: []const u8,
    /// Opt-in alpha telemetry: report numeric operational metrics to the
    /// bootstrap peer once a minute (docs/ALPHA.md lists every field).
    report_metrics: bool = false,
    /// Collect METRICS lines from consenting peers into this JSONL file.
    alpha_ingest: ?[]const u8 = null,
    rpc_addr: []const u8,
    rpc_port: u16,
    openai_addr: []const u8, // OpenAI-compatible HTTP API bind addr
    openai_port: u16, // 0 = OpenAI surface disabled (SPEC.md client API)
    ui_addr: []const u8 = "127.0.0.1", // bundled chat UI bind addr
    ui_port: u16 = 8555, // 0 = UI disabled
    status_secs: u32 = 30, // periodic console status; 0 = off
    kernel_threads: usize = 0, // 0 = auto (cpu count - 2)
    /// Serve routed experts zero-copy from a read-only mapping of the store
    /// rather than copying them into the RAM cache. See node.zig for why this
    /// is off by default.
    mmap_weights: bool = false,
    /// Act on the compute backend's calibration verdict. Off by default; see
    /// the backend for the measurement problem that makes it unsafe.
    gpu_ops: bool = false,
    /// Force the recorded whole-layer path off. It is otherwise decided by
    /// measurement: the node times a real token both ways at load and uses
    /// whichever wins. Separate from `gpu_ops` so the per-operation GPU paths
    /// and the recorded path can be bisected against each other -- they are
    /// different mechanisms and can fail independently.
    no_gpu_layers: bool = false,
    prefill_batch: usize = 0, // 0 = kernel max; 1 disables batched prefill
    p2p_addr: []const u8,
    p2p_port: u16,
    /// The LLM-network identity (Ethereum's chainId, for models): peers on a
    /// different network are refused at the gossip and p2p layers. null =
    /// auto: derived from the weight manifest when a store is attached,
    /// 0 (the open default network) otherwise.
    network_id: ?u64 = null,
    /// Named pre-configured network (devnet | testnet | mainnet): resolves
    /// to a stable network id and the canonical model. Conflicts with an
    /// explicit --network-id.
    network_name: ?[]const u8 = null,
    /// Retrieval-augmented generation: keep a gossiped chunk store and
    /// prepend the closest chunks to every prompt. Off unless asked -- it
    /// changes what the model sees.
    rag: bool = false,
    /// Chunks to retrieve per request when RAG is on.
    rag_k: usize = 3,
    ram_bytes: u64,
    pin_bytes: u64,
    verify: bool,
    seed: u64,
    stats_path: ?[]const u8,
    cache_root: []const u8,
    // GGUF distribution plane
    gguf_path: ?[]const u8, // serve this complete GGUF (origin/full holder)
    bootstrap: ?[]const u8, // "host:port" — sync weight ranges from this peer
    peers: ?[]const u8, // extra weight peers, comma-separated "host:port"
    hold_fraction: f32, // fraction of ranges to hold when bootstrapping
    range_bytes: u64, // range size when building a fresh manifest
    advertise: ?[]const u8, // our dialable "host:port" (default 127.0.0.1:<p2p_port>)
    r_target: u16, // committee redundancy target when acting as bootnode
    free_quota: u64, // per-client free token allowance (metering)
    admin_token: []const u8, // gates the credit op (empty = credit disabled)
    ctx_cap: usize, // context-length cap when serving a distributed GGUF engine
    chat_format: ?[]const u8, // --chat-format override (null = auto-detect)
};

/// Committee heartbeat (SPEC.md): PING each committee member on a fixed
/// interval, track liveness, log transitions. A member that stops answering
/// is marked dead locally; its shards become repair candidates.
const HEARTBEAT_INTERVAL_NS: i96 = 5 * std.time.ns_per_s;

const MemberState = struct { alive: bool = true, last_seq: u64 = 0 };

const HeartbeatCtx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    seed: [][]u8, // members known at join time
    table: *peers.Table, // gossip-derived committee view (later joiners)
    committee_id: u32,
    state: std.StringHashMap(MemberState), // key = owned addr string
    p2p_ctx: *p2p.Ctx, // our own state rides the heartbeat (SPEC.md container)

    /// One wire-frame heartbeat exchange. Returns the peer's heartbeat.
    fn exchange(self: *HeartbeatCtx, addr_str: []const u8) ?wire.Heartbeat {
        const gpa = self.gpa;
        const io = self.io;
        const addr = sync.PeerAddr.parse(addr_str) catch return null;
        const ip = dns.resolve(io, addr.host, addr.port) catch return null;
        const stream = ip.connect(io, .{ .mode = .stream }) catch return null;
        const dl = sockopt.trackPeer(io, stream);
        defer sockopt.untrack(io, dl);
        defer stream.close(io);
        var rbuf: [4096]u8 = undefined;
        var wbuf: [1024]u8 = undefined;
        var r = stream.reader(io, &rbuf);
        var w = stream.writer(io, &wbuf);

        const hb = p2p.selfHeartbeat(self.p2p_ctx);
        const body = hb.encodeBody(gpa) catch return null;
        defer gpa.free(body);
        const frame = wire.encodeFrame(gpa, .heartbeat, body) catch return null;
        defer gpa.free(frame);
        wire.writeFrame(&w.interface, frame) catch return null;

        const resp_raw = wire.readFrameAlloc(gpa, &r.interface) catch return null;
        defer gpa.free(resp_raw);
        const dec = wire.decodeFrame(gpa, resp_raw) catch return null;
        defer gpa.free(dec.body);
        if (dec.ty != .heartbeat_resp) return null;
        var out = wire.Heartbeat.parseBody(dec.body) catch return null;
        out.addr = ""; // aliases freed body; not needed by callers
        return out;
    }
};

fn heartbeatThread(ctx: *HeartbeatCtx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = HEARTBEAT_INTERVAL_NS }, .awake) catch return;

        // membership = static seed ∪ gossip-derived view (SPEC.md): earlier
        // members learn later joiners from their committee-tagged announces
        var members = std.ArrayList([]u8).empty;
        defer {
            for (members.items) |m| ctx.gpa.free(m);
            members.deinit(ctx.gpa);
        }
        for (ctx.seed) |m| {
            members.append(ctx.gpa, ctx.gpa.dupe(u8, m) catch continue) catch continue;
        }
        if (ctx.table.committeeMembersAlloc(ctx.gpa, ctx.committee_id)) |derived| {
            defer ctx.gpa.free(derived);
            outer: for (derived) |d| {
                for (members.items) |m| {
                    if (std.mem.eql(u8, m, d)) {
                        ctx.gpa.free(d);
                        continue :outer;
                    }
                }
                members.append(ctx.gpa, d) catch {
                    ctx.gpa.free(d);
                    continue;
                };
            }
        } else |_| {}

        for (members.items) |m| {
            const gop = ctx.state.getOrPut(m) catch continue;
            if (!gop.found_existing) {
                gop.key_ptr.* = ctx.gpa.dupe(u8, m) catch {
                    _ = ctx.state.remove(m);
                    continue;
                };
                gop.value_ptr.* = .{};
                std.debug.print("heartbeat: committee member {s} discovered\n", .{m});
            }
            const st = gop.value_ptr;

            const resp = ctx.exchange(m);
            const ok = resp != null;
            if (ok != st.alive) {
                std.debug.print("heartbeat: committee member {s} {s}\n", .{ m, if (ok) "alive" else "DEAD" });
                st.alive = ok;
                // death re-replication (audit #7 P1 / SPEC): survivors adopt the
                // dead member's advertised shards into their want-set so eager
                // repair re-replicates them from another live holder.
                if (!ok) {
                    if (ctx.p2p_ctx.store) |store| {
                        if (ctx.table.peerBitmapAlloc(ctx.gpa, m) catch null) |hex| {
                            defer ctx.gpa.free(hex);
                            var adopted: usize = 0;
                            var sid: usize = 0;
                            const nsh = store.manifest.nRanges();
                            while (sid < nsh) : (sid += 1) {
                                const bi = sid / 8;
                                if (bi * 2 + 1 >= hex.len) break;
                                const byte = std.fmt.parseInt(u8, hex[bi * 2 ..][0..2], 16) catch 0;
                                if (byte & (@as(u8, 1) << @intCast(sid % 8)) == 0) continue;
                                if (store.wanted.has(sid) or store.holdings.has(sid)) continue;
                                store.wanted.set(sid);
                                adopted += 1;
                            }
                            if (adopted > 0) std.debug.print("heartbeat: adopted {d} shard(s) from dead {s} into wanted\n", .{ adopted, m });
                        }
                    }
                }
            }
            if (resp) |hb| {
                if (hb.holdings_seq != st.last_seq) {
                    std.debug.print("heartbeat: {s} holdings_seq {d} -> {d} (load {d})\n", .{
                        m, st.last_seq, hb.holdings_seq, hb.load,
                    });
                    st.last_seq = hb.holdings_seq;
                }
            }
        }
    }
}

/// Churn-repair policy (ROADMAP #6, decided: maximally eager): whenever the
/// wanted range set is unsatisfied, retry every known peer on a short interval —
/// no waiting for a miss. A peer that was down when we bootstrapped, or that
/// comes back holding what we need, gets drained within one interval.
const REPAIR_INTERVAL_NS: i96 = 2 * std.time.ns_per_s;

const RepairCtx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    /// Candidate holders come from the live gossip table, so repair reaches
    /// peers this node was never explicitly told about.
    table: *peers.Table,
};

fn repairThread(ctx: *RepairCtx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = REPAIR_INTERVAL_NS }, .awake) catch return;
        if (ctx.store.missingCount() == 0) continue;
        const addrs = ctx.table.snapshotAddrs(ctx.gpa) catch continue;
        defer {
            for (addrs) |a| ctx.gpa.free(a);
            ctx.gpa.free(addrs);
        }
        // Security issue #25: never hold the engine mutex across peer I/O. A
        // peer that accepts and then goes silent would otherwise block every
        // inference request for as long as it stayed connected. Store mutation
        // needs no coarse lock: writeRange digest-verifies, writes disjoint
        // extents, and flips an atomic bit; saveSidecars self-serializes.
        var repaired: usize = 0;
        for (addrs) |addr_str| {
            if (ctx.store.missingCount() == 0) break;
            const addr = sync.PeerAddr.parse(addr_str) catch continue;
            const s = sync.fetchFromPeer(ctx.gpa, ctx.io, ctx.store, addr, null) catch continue;
            repaired += s.fetched;
        }
        if (repaired > 0) {
            ctx.store.saveSidecars() catch {};
            std.debug.print("repair: recovered {d} ranges, held {d}/{d}\n", .{
                repaired, ctx.store.holdings.count(), ctx.store.wanted.count(),
            });
        }
    }
}

/// Terminate the process instead of unwinding `run`'s stack frame.
///
/// Security issue #31: the gossip, heartbeat, repair and OpenAI loops are
/// detached and hold pointers into that frame (`&meter`, `&table`, `&store.?`,
/// `&gguf_src`, and their own contexts). Returning from `run` fires its defer
/// chain — `meter.deinit()`, `gguf_src.deinit()`, `committee_peers.deinit()` —
/// while those threads are still running, which is a use-after-free.
///
/// Every exit from `run` after the threads start is a failed listener, i.e.
/// fatal for a daemon. Exiting the process makes that explicit and removes the
/// whole teardown-race class, rather than trying to sequence a shutdown that
/// nothing else needs.
fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt, args);
    std.process.exit(1);
}

fn p2pThread(ctx: *p2p.Ctx) void {
    p2p.serve(ctx) catch |e| fatal("p2p server failed: {s}\n", .{@errorName(e)});
}

/// Reopen a previously synced store when no peer answers at boot. A node
/// that already holds verified shards has no reason to die because the
/// bootnode is mid-restart: it serves what it holds, and the eager-repair
/// loop keeps dialing until the swarm returns.
fn reopenLocalStore(gpa: std.mem.Allocator, io: Io, store_dir: []const u8, out: *Io.Writer) ?weights.Store {
    const st = weights.openDir(gpa, io, store_dir) catch return null;
    out.print("peers unreachable; starting from the existing local store ({d}/{d} shards held)\n", .{
        st.holdings.count(), st.manifest.nRanges(),
    }) catch {};
    out.print("eager repair keeps dialing and reconnects when a peer returns\n", .{}) catch {};
    out.flush() catch {};
    return st;
}

fn openaiThread(ctx: *openai.Ctx) void {
    openai.serve(ctx) catch |e| fatal("openai server failed: {s}\n", .{@errorName(e)});
}

/// Load a GGUF through whichever engine its `general.architecture` selects.
/// deepseek2 is MLA; everything else supported is GQA (llama.zig).
/// Set by `loadGgufEngine` so the banner can report whether the whole layer
/// stack records into one command buffer.
var gguf_layers_gpu: bool = false;

fn loadGgufEngine(gpa: std.mem.Allocator, io: Io, path: []const u8, gpu_ops: bool, no_gpu_layers: bool) !generator.GgufModel {
    var peek = try gguf_mod.parse(gpa, io, path);
    const arch_src = peek.getString("general.architecture") orelse "?";
    var arch_buf: [64]u8 = undefined;
    const n = @min(arch_src.len, arch_buf.len);
    @memcpy(arch_buf[0..n], arch_src[0..n]);
    peek.deinit();
    const arch = arch_buf[0..n];

    if (std.mem.eql(u8, arch, "deepseek2")) {
        var m = try deepseek.load(gpa, io, path);
        calibrateFor(gpa, gpu_ops, m.cfg.dim, m.cfg.ffn, m.cfg.n_layers, m.cfg.ctx_len, 0, 0, 0, 0, &m.layers[0], m.output);
        return .{ .deepseek = m };
    }
    var m = try llama.load(gpa, io, path);
    calibrateFor(gpa, gpu_ops, m.cfg.dim, m.cfg.ffn, m.cfg.n_layers, m.cfg.ctx_len, m.cfg.kvDim(), m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.head_dim, &m.layers[0], m.output);
    // Decided by measurement, after the device KV cache exists. The mirror has
    // to be on *before* the timing run or the recorded path reads a cache the
    // host path never filled and the comparison is between a correct forward
    // pass and a broken one.
    // `hasKvCache` gates this because calibration builds a scratch `State`,
    // and that is sized by the model's context: 8 GB for a 7B advertising
    // 32768. Measuring a path that cannot run -- attnInit declines any context
    // past what the attention kernel's threadgroup memory serves -- would
    // allocate all of it to learn nothing.
    if (!no_gpu_layers and backend.hasKvCache() and llama.gpuLayersSupported(&m)) {
        backend.enableKvMirror();
        backend.parallelBegin(generator.threads());
        m.gpu_layers = llama.calibrateGpuLayers(&m, gpa);
        backend.parallelEnd();
        // Nothing reads the device cache if both the recorded path and fused
        // attention lost, and it is hundreds of megabytes on a 7B model.
        if (!m.gpu_layers and !backend.lastVerdict().attn_used) backend.releaseKvCache();
    }
    gguf_layers_gpu = m.gpu_layers;
    return .{ .gqa = m };
}

/// Report what calibration decided. Printed after the model is loaded, since
/// before that there is nothing measured to report.
fn printCompute(out: *Io.Writer, gpu_layers: bool) !void {
    const v = backend.lastVerdict();
    // Before the verdict, because a weight mapping that failed to become
    // resident is invisible in every other number until the page-in counter is
    // read -- which is how it went unnoticed the first time.
    const resident = backend.materializeArenas();
    if (resident > 0) {
        try out.print("  weights    {d:.1} GB device-resident\n", .{
            @as(f64, @floatFromInt(resident)) / (1024.0 * 1024.0 * 1024.0),
        });
    } else if (backend.arenaError.*) |why| {
        try out.print("  weights    not device-resident: {s}\n", .{why});
    }
    if (!v.ran) return out.print("  compute    backend {s}\n", .{backend.name});
    try out.print("  compute    backend {s}; measured on this model: matvec {s}", .{
        backend.name,
        if (v.matvec_used) "gpu" else "cpu",
    });
    if (v.matvec_used) try out.print(" (>= {d} rows)", .{v.matvec_min_rows});
    try out.print(", ffn block {s}", .{if (v.ffn_used) "gpu" else "cpu"});
    if (v.ffn_gpu_ms > 0) try out.print(" (gpu {d:.3} ms vs cpu {d:.3} ms)", .{ v.ffn_gpu_ms, v.ffn_cpu_ms });
    try out.print(", prefill {s}", .{if (v.prefill_used) "gpu" else "cpu"});
    try out.print(", attn {s}", .{if (v.attn_used) "gpu" else "cpu"});
    if (gpu_layers) {
        try out.print(", layers gpu (1 cmd buffer/token)", .{});
    } else if (llama.last_layer_cpu_ns > 0) {
        try out.print(", layers cpu", .{});
    }
    if (llama.last_layer_cpu_ns > 0) try out.print(" (gpu {d:.2} ms/tok vs cpu {d:.2} ms/tok)", .{
        @as(f64, @floatFromInt(llama.last_layer_gpu_ns)) / 1e6,
        @as(f64, @floatFromInt(llama.last_layer_cpu_ns)) / 1e6,
    });
    if (v.attn_gpu_ms > 0) try out.print(" (gpu {d:.3} ms vs cpu {d:.3} ms)", .{ v.attn_gpu_ms, v.attn_cpu_ms });
    try out.print("\n", .{});
}

/// Hand the compute backend the shapes this model will actually issue and let
/// it decide, per operation, whether it is worth using. A backend built in is
/// not a backend used: on unified memory the GPU loses at small shapes and
/// wins at large ones, and where the line falls is a property of the machine.
fn calibrateFor(gpa: std.mem.Allocator, gpu_ops: bool, dim: usize, ffn: usize, n_layers: usize, ctx_len: usize, kvd: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, l: anytype, out_head: anytype) void {
    var shapes: [3]backend.Shape = undefined;
    var n: usize = 0;
    if (l.ffn_gate) |g| {
        shapes[n] = .{ .data = g.data, .ty = g.ty, .rows = g.ne1, .cols = g.ne0 };
        n += 1;
    }
    if (l.ffn_down) |d| {
        shapes[n] = .{ .data = d.data, .ty = d.ty, .rows = d.ne1, .cols = d.ne0 };
        n += 1;
    }
    shapes[n] = .{ .data = out_head.data, .ty = out_head.ty, .rows = out_head.ne1, .cols = out_head.ne0 };
    n += 1;
    // The pool has to be up for this: it is what brings the GPU device online,
    // and the CPU side of the comparison has to run with the same thread count
    // a real generation gets or the measurement is rigged against it.
    backend.useGpuOps.* = gpu_ops;
    backend.parallelBegin(generator.threads());
    defer backend.parallelEnd();
    const triple: ?[3]backend.Shape = if (l.ffn_gate != null and l.ffn_up != null and l.ffn_down != null) .{
        .{ .data = l.ffn_gate.?.data, .ty = l.ffn_gate.?.ty, .rows = l.ffn_gate.?.ne1, .cols = l.ffn_gate.?.ne0 },
        .{ .data = l.ffn_up.?.data, .ty = l.ffn_up.?.ty, .rows = l.ffn_up.?.ne1, .cols = l.ffn_up.?.ne0 },
        .{ .data = l.ffn_down.?.data, .ty = l.ffn_down.?.ty, .rows = l.ffn_down.?.ne1, .cols = l.ffn_down.?.ne0 },
    } else null;
    backend.calibrate(gpa, dim, ffn, shapes[0..n], triple);
    // kvd == 0 means the engine has no GQA-shaped cache to hand over —
    // deepseek is MLA and keeps a compressed one — so there is nothing to
    // allocate and the backend keeps declining `attnHeads`.
    if (kvd != 0 and backend.attnInit(n_layers, ctx_len, kvd)) {
        // Calibrate at a mid-length sequence: fused attention's cost is one
        // command buffer regardless of seq, while the CPU loop grows with it,
        // so the verdict at seq=1 and seq=ctx differ. 256 is a plausible
        // working context rather than either extreme.
        backend.calibrateAttn(gpa, n_heads, n_kv_heads, head_dim, @min(@as(usize, 256), ctx_len));
        if (!gpu_ops) backend.disableAttn();
    }
}

fn attachGgufDist(m: *generator.GgufModel, gpa: std.mem.Allocator, src: *expert_fetch.Source) !void {
    switch (m.*) {
        .deepseek => |*d| try deepseek.attachDist(d, gpa, src),
        .gqa => |*g| try llama.attachDist(g, gpa, src),
    }
}

pub fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, opts: Options) !void {
    try banner.print(out, "node");
    generator.kernel_threads = opts.kernel_threads;
    generator.prefill_batch = opts.prefill_batch;

    const resolved = hf.resolve(gpa, io, opts.model, opts.cache_root) catch |e| {
        try out.print("model resolve failed ({s}): {s}\n", .{ opts.model, @errorName(e) });
        return;
    };
    defer gpa.free(resolved.dir);

    var eng = Engine.init(gpa, io, resolved.dir, .{
        .ram_budget_bytes = opts.ram_bytes,
        .pin_budget_bytes = opts.pin_bytes,
        .verify = opts.verify,
    }) catch |e| {
        try out.print("load failed: {s}\n", .{@errorName(e)});
        return;
    };
    defer eng.deinit();

    if (opts.stats_path) |sp| if (opts.pin_bytes > 0) {
        if (try stats.readCounts(gpa, io, sp, eng.cache.access_count.len)) |counts| {
            defer gpa.free(counts);
            try eng.pinWithinBudget(counts, opts.pin_bytes);
        }
    };

    // ---- GGUF weight-distribution store (optional) ----
    var store: ?weights.Store = null;
    defer if (store) |*st| st.deinit();
    var committee_members: [][]u8 = &.{};
    defer {
        for (committee_members) |m| gpa.free(m);
        if (committee_members.len > 0) gpa.free(committee_members);
    }
    var registry: ?bootnode.Registry = null;
    defer if (registry) |*r| r.deinit();
    var joined_committee_id: u32 = wire.NO_COMMITTEE;

    // known weight peers = --bootstrap (first) + --peers (comma-separated)
    var peer_list = std.ArrayList(sync.PeerAddr).empty;
    defer peer_list.deinit(gpa);
    var peer_strs = std.ArrayList([]const u8).empty; // same peers, as strings for the gossip table
    defer peer_strs.deinit(gpa);
    // Named-network resolution happens up front: the registry supplies the
    // default bootnode, so `loom node --network devnet` alone is a complete
    // join command.
    var preset: ?*const networks.Network = null;
    if (opts.network_name) |nm| {
        preset = networks.byName(nm) orelse {
            try out.print("unknown --network '{s}' (want devnet | testnet | mainnet)\n", .{nm});
            return;
        };
        if (opts.network_id != null and opts.network_id.? != preset.?.id) {
            try out.print("--network {s} is id {d}; conflicting --network-id {d}\n", .{ nm, preset.?.id, opts.network_id.? });
            return;
        }
    }
    var eff_bootstrap: ?[]const u8 = opts.bootstrap;
    if (preset) |p| {
        if (eff_bootstrap == null and opts.gguf_path == null and p.bootnodes.len > 0) {
            eff_bootstrap = p.bootnodes[0];
            try out.print("  bootnode   {s} (registry default for {s})\n", .{ eff_bootstrap.?, p.name });
            try out.flush();
        }
    }

    if (eff_bootstrap) |bs| {
        const a = sync.PeerAddr.parse(bs) catch {
            try out.print("bad --bootstrap address (want host:port): {s}\n", .{bs});
            return;
        };
        try peer_list.append(gpa, a);
        try peer_strs.append(gpa, bs);
    }
    if (opts.peers) |csv| {
        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " ");
            if (t.len == 0) continue;
            const a = sync.PeerAddr.parse(t) catch {
                try out.print("bad --peers entry (want host:port): {s}\n", .{t});
                return;
            };
            try peer_list.append(gpa, a);
            try peer_strs.append(gpa, t);
        }
    }

    // RAG chunk store (text + locally computed embeddings); created before
    // the p2p contexts that carry its pointer, embedder bound later.
    var rag: ?rag_store.Store = if (opts.rag) rag_store.Store.init(gpa, io) else null;
    defer if (rag) |*r| r.deinit();

    // gossip peer table, seeded with the statically configured peers
    var advertise_buf: [128]u8 = undefined;
    const advertise = opts.advertise orelse
        try std.fmt.bufPrint(&advertise_buf, "127.0.0.1:{d}", .{opts.p2p_port});
    // Loopback is the right default for a single box and silently wrong the
    // moment a peer is not on it: the node still gossips, still fetches, and
    // still looks healthy, because it only ever dials outward. What breaks is
    // everything that needs someone to dial *in* -- the >=2-sources guarantee,
    // and any repair a peer would initiate to recover shards this node holds.
    // Measured across two machines: the remote origin listed this node as a
    // holder of hundreds of shards it could not reach.
    if (opts.advertise == null and peer_strs.items.len > 0) {
        var external = false;
        for (peer_strs.items) |ps| {
            if (!std.mem.startsWith(u8, ps, "127.0.0.1") and !std.mem.startsWith(u8, ps, "localhost")) external = true;
        }
        if (external) {
            try out.print(
                "  WARNING    advertising {s} to non-local peers; they cannot dial back.\n" ++
                    "             Pass --advertise <reachable-host>:{d} or this node can never\n" ++
                    "             serve a shard it holds, and redundancy targets will not be met.\n",
                .{ advertise, opts.p2p_port },
            );
            try out.flush();
        }
    }
    var table = peers.Table.init(gpa, io, advertise);
    defer table.deinit();
    const zero_version = "0" ** 64;
    for (peer_strs.items) |ps| {
        _ = table.merge(ps, zero_version, "", peers.NO_COMMITTEE, 0, stats.nowNs(io), .hearsay) catch {};
    }

    if (opts.gguf_path) |gp| {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-origin", .{opts.cache_root});
        defer gpa.free(store_dir);
        store = weights.openFull(gpa, io, gp, store_dir, opts.range_bytes, null) catch |e| {
            try out.print("gguf store failed ({s}): {s}\n", .{ gp, @errorName(e) });
            return;
        };
        // an expert-sharded origin acts as the bootnode (SPEC.md)
        if (store.?.manifest.mode == .expert) {
            registry = bootnode.Registry.init(
                gpa,
                io,
                store.?.manifest.nRanges(),
                store.?.manifest.n_resident,
                opts.r_target,
            );
        }
    } else if (peer_list.items.len > 0) sync_block: {
        const store_dir = try std.fmt.allocPrint(gpa, "{s}/gguf-synced", .{opts.cache_root});
        defer gpa.free(store_dir);

        // SPEC.md join flow: ask the bootnode for a committee + assigned
        // want-set. Falls back to random hold-fraction if the peer is not a
        // bootnode (legacy swarms, fixed-mode manifests).
        var joined: ?sync.JoinInfo = null;
        var join_try: usize = 0;
        while (join_try < 4) : (join_try += 1) {
            if (sync.joinSwarm(gpa, io, peer_list.items[0], advertise, opts.hold_fraction)) |ji| {
                joined = ji;
                break;
            } else |e| {
                // A refused dial is usually a bootnode mid-restart (a big
                // origin takes minutes to reopen its store), so wait it out
                // briefly before falling back.
                if (e == error.ConnectionRefused and join_try < 3) {
                    try out.print("bootnode not answering (attempt {d}/4); retrying in 30s...\n", .{join_try + 1});
                    try out.flush();
                    Io.sleep(io, .{ .nanoseconds = 30 * std.time.ns_per_s }, .awake) catch {};
                    continue;
                }
                try out.print("join declined ({s}); using random hold-fraction\n", .{@errorName(e)});
                try out.flush();
                break;
            }
        }

        if (joined) |*ji| {
            joined_committee_id = @intCast(ji.committee_id);
            try out.print("joined committee {d} ({d} member(s) already in it)\n", .{ ji.committee_id, ji.members.len });
            try out.flush();
            // sync preference: committee members first, then the bootnode
            var srcs = std.ArrayList(sync.PeerAddr).empty;
            defer srcs.deinit(gpa);
            for (ji.members) |m| {
                if (sync.PeerAddr.parse(m)) |a| {
                    try srcs.append(gpa, a);
                    // committee members seed the table WITH their committee id
                    _ = table.merge(m, "0" ** 64, "", joined_committee_id, 0, stats.nowNs(io), .hearsay) catch {};
                } else |_| {}
            }
            try srcs.appendSlice(gpa, peer_list.items);

            // hand manifest+wanted to the store; keep members for heartbeats
            const res = sync.bootstrapWithWanted(gpa, io, srcs.items, store_dir, ji.manifest, ji.wanted, out) catch |e| {
                for (ji.members) |m| gpa.free(m);
                gpa.free(ji.members);
                if (reopenLocalStore(gpa, io, store_dir, out)) |st| {
                    store = st;
                } else {
                    try out.print("bootstrap failed: {s}\n", .{@errorName(e)});
                    return;
                }
                break :sync_block;
            };
            store = res.store;
            committee_members = ji.members; // ownership moves (freed at exit)
            ji.members = &.{};
            try out.print("  synced {d}/{d} assigned shards, {d:.1} MB, verified against manifest root\n", .{
                res.fetched, res.wanted, @as(f64, @floatFromInt(res.bytes)) / MBf,
            });
        } else {
            try out.print("bootstrapping weight ranges from {d} peer(s) (hold-fraction {d:.2})...\n", .{ peer_list.items.len, opts.hold_fraction });
            try out.flush();
            try out.flush();
            const res = sync.bootstrap(gpa, io, peer_list.items, store_dir, opts.hold_fraction, opts.seed, out) catch |e| {
                if (reopenLocalStore(gpa, io, store_dir, out)) |st| {
                    store = st;
                } else {
                    try out.print("bootstrap failed: {s}\n", .{@errorName(e)});
                    return;
                }
                break :sync_block;
            };
            store = res.store;
            try out.print("  synced {d}/{d} wanted ranges, {d:.1} MB, verified against manifest root\n", .{
                res.fetched, res.wanted, @as(f64, @floatFromInt(res.bytes)) / MBf,
            });
            if (res.fetched < res.wanted) {
                try out.print("  {d} ranges still missing — eager repair will keep chasing peers\n", .{res.wanted - res.fetched});
            }
        }
    }

    const c = eng.cfg;
    const s = eng.sizes;
    try out.print("loom node up\n", .{});
    try out.print("  model      {s}  (source: {s})\n", .{ resolved.dir, resolved.source });
    try out.print("  shape      hidden={d} layers={d} experts={d}/layer top-{d}+{d} shared\n", .{
        c.hidden, c.nLayers(), c.n_experts, c.n_routed, c.n_shared,
    });
    try out.print("  resident   dense={d:.3} GB kv={d:.3} GB\n", .{
        @as(f64, @floatFromInt(s.dense_bytes)) / GB, @as(f64, @floatFromInt(s.kv_bytes)) / GB,
    });
    try out.print("  cache      expert_bytes={d} lru_capacity={d} pinned={d} ram_budget={d:.2} GB\n", .{
        s.expert_bytes, s.lru_capacity, s.pinned_experts, @as(f64, @floatFromInt(opts.ram_bytes)) / GB,
    });
    try out.print("  rpc        tcp://{s}:{d}   (metered; free quota {d} tokens/client; credit {s})\n", .{ opts.rpc_addr, opts.rpc_port, opts.free_quota, if (opts.admin_token.len > 0) "admin-gated" else "disabled" });
    if (opts.openai_port != 0) try out.print("  openai     http://{s}:{d}/v1  (OpenAI-compatible; metered; non-streaming)\n", .{ opts.openai_addr, opts.openai_port });
    try out.print("  p2p        tcp://{s}:{d}   (HELLO | HAS | MANIFEST | DIGESTS | HOLDINGS | GETR | GOSSIP | TABLE | PING)\n", .{ opts.p2p_addr, opts.p2p_port });
    try out.print("  gossip     advertising as {s}, {d} seed peer(s), every {d}s\n", .{
        advertise, peer_strs.items.len, @divTrunc(gossip.INTERVAL_NS, std.time.ns_per_s),
    });
    if (registry != null) try out.print("  bootnode   committee registry active (R target {d})\n", .{opts.r_target});
    if (committee_members.len > 0) try out.print("  committee  {d} member(s), heartbeat every {d}s\n", .{
        committee_members.len, @divTrunc(HEARTBEAT_INTERVAL_NS, std.time.ns_per_s),
    });
    if (store) |*st| {
        // Now that the shard count is known, announces carrying a holdings
        // bitmap of any other size are rejected (security issue #28).
        table.expected_holdings_hex_len = ((st.manifest.nRanges() + 7) / 8) * 2;
        try out.print("  weights    version={s}\n", .{hashmod.toHex(st.manifest.version)});
        try out.print("  shards     mode={s} total={d} (resident={d}, expert={d}) held={d} ({d:.1}%)\n", .{
            @tagName(st.manifest.mode),
            st.manifest.nRanges(),
            st.manifest.n_resident,
            st.manifest.nRanges() - st.manifest.n_resident,
            st.holdings.count(),
            100.0 * @as(f64, @floatFromInt(st.holdings.count())) / @as(f64, @floatFromInt(st.manifest.nRanges())),
        });
    }
    try out.print("  serving... (Ctrl-C to stop)\n", .{});
    try out.flush();

    // The LLM-network identity every peer must share (chainId semantics).
    // Resolution order: a named network (devnet/testnet/mainnet -- stable
    // registry constants), else an explicit --network-id, else derived from
    // the manifest. Computed HERE, after the store is final -- the first
    // version ran before any store existed and every node silently derived
    // id 0, which the three-machine test caught.
    const network_id: u64 = if (preset) |p| p.id else (opts.network_id orelse
        (if (store) |*st| wire.networkIdFromManifest(st.manifest.version) else 0));
    if (preset) |p| {
        try out.print("  network    {s} (id {d}) -- {s}\n", .{ p.name, p.id, p.desc });
    } else {
        try out.print("  network    id {d}\n", .{network_id});
    }

    // A node's p2p port is an identity, not a load-balancing group, but the
    // listener's reuse_address (needed for fast restarts) also sets
    // SO_REUSEPORT on POSIX, which lets a second node bind the same port and
    // silently split incoming connections with this one (issue #180, found
    // when a stray process made a healthy bootnode answer ERR no_store on
    // half its connections). Probe before binding and refuse instead.
    {
        const probe_host = if (std.mem.eql(u8, opts.p2p_addr, "0.0.0.0")) "127.0.0.1" else opts.p2p_addr;
        if (net.IpAddress.parse(probe_host, opts.p2p_port)) |pa| {
            if (pa.connect(io, .{ .mode = .stream })) |ps| {
                ps.close(io);
                try out.print("p2p port {d} is already serving (another loom node?); refusing to share it\n", .{opts.p2p_port});
                try out.flush();
                return;
            } else |_| {}
        } else |_| {}
    }

    var p2p_ctx = p2p.Ctx{
        .gpa = gpa,
        .io = io,
        .entries = eng.loaded.entries,
        .unique_bytes = s.unique_expert_bytes,
        .addr = opts.p2p_addr,
        .port = opts.p2p_port,
        .store = if (store) |*st| st else null,
        .table = &table,
        .boot = if (registry) |*r| r else null,
        .committee_id = joined_committee_id,
        .advertise = advertise,
        .network_id = network_id,
        .rag = if (rag) |*r| r else null,
        .metrics_path = opts.alpha_ingest,
    };
    const p2p_handle = try std.Thread.spawn(.{}, p2pThread, .{&p2p_ctx});
    defer p2p_handle.join();

    // gossip loop: announce ourselves + learn peers-of-peers
    var gossip_ctx = gossip.Ctx{
        .gpa = gpa,
        .io = io,
        .table = &table,
        .store = if (store) |*st| st else null,
        .advertise = advertise,
        .committee_id = joined_committee_id,
        .network_id = network_id,
        .rag = if (rag) |*r| r else null,
    };
    {
        const t = try std.Thread.spawn(.{}, gossip.loop, .{&gossip_ctx});
        t.detach();
    }

    // committee heartbeats (SPEC.md)
    var hb_ctx: HeartbeatCtx = undefined;
    if (joined_committee_id != wire.NO_COMMITTEE) {
        hb_ctx = .{
            .gpa = gpa,
            .io = io,
            .seed = committee_members,
            .table = &table,
            .committee_id = joined_committee_id,
            .state = std.StringHashMap(MemberState).init(gpa),
            .p2p_ctx = &p2p_ctx,
        };
        const t = try std.Thread.spawn(.{}, heartbeatThread, .{&hb_ctx});
        t.detach();
    }

    // One engine mutex shared by every serve path (rpc + openai) AND the eager
    // repair loop: generation holds mutable per-request state and mutates the
    // store (token-loop fetch), and repair mutates the store too, so all of it
    // serializes here.
    var engine_lock: Io.Mutex = .init;

    // The status thread writes to the same `out` as the main thread and the
    // event logs; interleaved prints from two threads would corrupt lines.
    var out_lock: Io.Mutex = .init;

    // Choose what serves inference. Default: the loom-format engine. If an
    // expert-sharded GGUF store is attached and its resident bundle is complete,
    // serve the distributed GGUF (deepseek2) engine with token-loop peer fetch.
    var gen = generator.Generator{ .loom = &eng };
    var gguf_gen: generator.GgufGen = undefined;
    var gguf_src: expert_fetch.Source = undefined;
    var committee_peers = std.ArrayList(sync.PeerAddr).empty;
    defer committee_peers.deinit(gpa);
    var serve_gguf = false;
    // `--model` names the loom-format checkpoint. When an expert-sharded GGUF
    // takes over the serve path, that spec is no longer what answers requests,
    // and echoing it in /v1/models and the UI header is simply wrong.
    var served_model_id: []const u8 = opts.model;
    // `hf.resolve` reports how the loom-format checkpoint was obtained; a
    // generated one has random weights. The GGUF path re-decides this below
    // from the file's own metadata.
    var synthetic = std.mem.eql(u8, resolved.source, "synthetic");
    if (store) |*st| {
        if (st.manifest.mode == .expert) {
            for (committee_members) |m| {
                if (sync.PeerAddr.parse(m)) |a| {
                    committee_peers.append(gpa, a) catch {};
                } else |_| {}
            }
            // Read weights straight out of a read-only mapping instead of
            // copying them into the heap cache. Off by default, because on the
            // one machine this has been measured -- 16 GB serving an 8.9 GB
            // model -- it is slower, and the reason is worth recording.
            //
            //   config              get     ffn    total ms/token
            //   no cache          375.9    64.8    480.0
            //   2 GB heap cache   269.0    73.7    409.7
            //   8 GB heap cache    87.9   179.6    362.4
            //   mmap, zero-copy    34.5   369.7    484.9
            //
            // Zero-copy does exactly what it claims: `get` falls to a pointer
            // return. But the model does not fit in this machine's page cache,
            // so ~1.16 GB of expert weights per token is faulted from disk
            // either way, and reading them through the mapping moves that cost
            // into whoever touches the pages -- the matmul.
            //
            // That table is superseded on a GPU build, and the reason it was
            // ever true is below: the mapping was left to the kernel to keep
            // resident, and the kernel evicted it. Handing the mapping to the
            // compute backend, which wires it, reverses the ordering -- 4.1-5.4
            // tok/s against 3.5-3.6 for the 8 GB heap cache on the same
            // machine, with page-ins over a generation down from ~6.9 GB to
            // ~1.4-2.2 GB. The heap cache still wins where there is no backend
            // to hand it to, which is why this stays a flag rather than the
            // default.
            if (opts.mmap_weights) {
                st.mapReadOnly();
                // Hand the whole mapping to the compute backend as one
                // allocation. Without this the GPU wraps each expert extent
                // separately and the model is never resident -- which is the
                // measured difference between 2.5 tok/s and llama.cpp's 47+ on
                // this same checkpoint.
                const mem = st.mapping();
                const took = if (mem) |m2| backend.registerArena(m2) else false;
                // Printed either way: a mapping that failed, or one the backend
                // would not take, is otherwise invisible until someone reads a
                // page-in counter -- which is exactly how it went unnoticed.
                try out.print("  weights    mmap {s}{s}\n", .{
                    if (mem) |m2| blk_s: {
                        var b: [32]u8 = undefined;
                        break :blk_s try std.fmt.bufPrint(&b, "{d:.1} GB", .{
                            @as(f64, @floatFromInt(m2.len)) / (1024.0 * 1024.0 * 1024.0),
                        });
                    } else "failed",
                    if (took) ", registered with the compute backend" else "",
                });
                try out.flush();
            }
            // --hold-fraction is a cap, not just a bootstrap target. Without
            // this it bounded only the initial sync: a shard fetched from a
            // peer at token time was persisted and marked held with nothing to
            // evict it, so a node measured across two machines went from 3.6%
            // to 93.1% of the corpus while generating 24 tokens -- every
            // serving node converging on a full replica, which is the opposite
            // of storing the corpus once across a swarm.
            if (opts.hold_fraction < 1.0) {
                const experts = st.manifest.nRanges() - st.manifest.n_resident;
                const cap: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(experts)) * opts.hold_fraction));
                st.setCap(cap);
                try out.print("  capacity   holding at most {d} of {d} expert shards ({d:.0}%), coldest evicted\n", .{
                    cap, experts, opts.hold_fraction * 100.0,
                });
            }
            gguf_src = try expert_fetch.Source.initCached(gpa, io, st, peer_list.items, opts.ram_bytes);
            gguf_src.committee = committee_peers.items;
            // resident gate (audit #5 P0-4): the mmap'd resident bundle must be
            // present+verified, fetching gaps from peers, or inference reads
            // file-hole zeros. Fail closed on the serve path (fall back to loom).
            var missing_resident: usize = 0;
            var ri: usize = 0;
            while (ri < st.manifest.n_resident) : (ri += 1) {
                _ = gguf_src.get(ri) catch {
                    missing_resident += 1;
                };
            }
            if (missing_resident == 0) {
                // origin serves the original --gguf file (openFull leaves it in
                // place); a bootstrapped node serves the sparse copy synced into
                // its store dir.
                const mpath = if (opts.gguf_path) |gp|
                    try gpa.dupe(u8, gp)
                else
                    try std.fmt.allocPrint(gpa, "{s}/model.gguf", .{st.dir});
                defer gpa.free(mpath);
                // Pick the engine from the file's own architecture: MLA
                // (deepseek2) or the GQA family (llama/Mixtral, qwen2moe,
                // qwen3moe, glm4moe). Both attach to the same expert source.
                if (loadGgufEngine(gpa, io, mpath, opts.gpu_ops, opts.no_gpu_layers)) |mdl| {
                    gguf_gen = .{ .m = mdl, .src = &gguf_src, .ctx_cap = opts.ctx_cap };
                    gguf_gen.m.setCtxLen(@min(gguf_gen.m.ctxLen(), opts.ctx_cap));
                    gguf_gen.m.initDeviceAttn();
                    const arch_name = gguf_gen.m.archName();
                    if (attachGgufDist(&gguf_gen.m, gpa, &gguf_src)) |_| {
                        gguf_gen.chat_format = if (opts.chat_format) |cf|
                            chat_template.parse(cf) orelse chat_template.detect(gguf_gen.m.chatTemplate(), arch_name)
                        else
                            chat_template.detect(gguf_gen.m.chatTemplate(), arch_name);
                        gen = .{ .gguf = &gguf_gen };
                        serve_gguf = true;
                        served_model_id = if (opts.gguf_path) |gp| std.fs.path.basename(gp) else arch_name;
                        // `loom gguf gen` stamps its fixtures "loom <arch> fixture".
                        const gname = gguf_gen.m.generalName() orelse "";
                        synthetic = std.mem.startsWith(u8, gname, "loom ") and std.mem.endsWith(u8, gname, "fixture");
                        try printCompute(out, gguf_layers_gpu);
                        try out.print("  serving    distributed GGUF ({s}): ctx={d} chat={s}\n", .{
                            arch_name, gguf_gen.m.ctxLen(), @tagName(gguf_gen.chat_format),
                        });
                    } else |e| {
                        try out.print("  gguf serve disabled: attach failed ({s})\n", .{@errorName(e)});
                        gguf_gen.m.deinit();
                        gguf_src.deinit();
                    }
                } else |e| {
                    try out.print("  gguf serve disabled: model load failed ({s}); serving loom engine\n", .{@errorName(e)});
                    gguf_src.deinit();
                }
            } else {
                try out.print("  gguf serve disabled: {d}/{d} resident shards unavailable; serving loom engine\n", .{ missing_resident, st.manifest.n_resident });
                gguf_src.deinit();
            }
        }
    }

    // Local-only fallback: a dense (non-MoE) model has no routed experts, so it
    // shards into fixed ranges and cannot be served *distributed*. It can still
    // be served — it is a complete model file sitting right there. Without this,
    // pointing --gguf at an ordinary model silently answered from the synthetic
    // loom checkpoint instead, which looks like a broken model rather than an
    // unsupported topology.
    var local_gguf = false;
    if (!serve_gguf) {
        if (opts.gguf_path) |gp| {
            if (loadGgufEngine(gpa, io, gp, opts.gpu_ops, opts.no_gpu_layers)) |mdl| {
                gguf_gen = .{ .m = mdl, .src = null, .ctx_cap = opts.ctx_cap };
                gguf_gen.m.setCtxLen(@min(gguf_gen.m.ctxLen(), opts.ctx_cap));
                gguf_gen.m.initDeviceAttn();
                const arch_name = gguf_gen.m.archName();
                gguf_gen.chat_format = if (opts.chat_format) |cf|
                    chat_template.parse(cf) orelse chat_template.detect(gguf_gen.m.chatTemplate(), arch_name)
                else
                    chat_template.detect(gguf_gen.m.chatTemplate(), arch_name);
                gen = .{ .gguf = &gguf_gen };
                local_gguf = true;
                served_model_id = std.fs.path.basename(gp);
                const gname = gguf_gen.m.generalName() orelse "";
                synthetic = std.mem.startsWith(u8, gname, "loom ") and std.mem.endsWith(u8, gname, "fixture");
                try printCompute(out, gguf_layers_gpu);
                try out.print("  serving    local GGUF ({s}): ctx={d} chat={s}  (not expert-sharded: no distributed fetch)\n", .{
                    arch_name, gguf_gen.m.ctxLen(), @tagName(gguf_gen.chat_format),
                });
            } else |e| {
                try out.print("  gguf serve disabled: model load failed ({s}); serving loom engine\n", .{@errorName(e)});
            }
        }
    }

    defer if (serve_gguf) {
        gguf_gen.m.deinit();
        gguf_src.deinit();
    } else if (local_gguf) {
        gguf_gen.m.deinit();
    };
    try out.flush();

    // eager churn repair, drawing candidates from the live gossip table
    var repair_ctx: RepairCtx = undefined;
    if (store != null) {
        repair_ctx = .{
            .gpa = gpa,
            .io = io,
            .store = &store.?,
            .table = &table,
        };
        const t = try std.Thread.spawn(.{}, repairThread, .{&repair_ctx});
        t.detach();
    }

    // Named-network arch policy: one network serves one model. Production
    // networks refuse a mismatched architecture; devnet warns -- a PoC
    // network is where mismatches get discovered on purpose.
    if (preset) |p| {
        const served_arch: []const u8 = if (serve_gguf) gguf_gen.m.archName() else "loom-checkpoint";
        if (!std.mem.eql(u8, served_arch, p.arch)) {
            if (p.strict) {
                try out.print("REFUSING: network '{s}' serves arch '{s}', this node loaded '{s}'\n", .{ p.name, p.arch, served_arch });
                try out.print("          the canonical model is {s}\n", .{p.model});
                return;
            }
            try out.print("  WARNING    network '{s}' expects arch '{s}', serving '{s}' ({s})\n", .{ p.name, p.arch, served_arch, p.model });
        }
    }

    // RAG: the store exists from startup (the p2p/gossip contexts hold its
    // pointer), the embedder arrives once the serve path is chosen below.
    // Embeddings come from THIS node's model and only text is gossiped, so
    // every node on the network derives the same vector for the same chunk.
    if (opts.rag) {
        if (serve_gguf) {
            rag.?.setEmbedder(.{
                .ctx = @ptrCast(&gguf_gen.m),
                .dim = gguf_gen.m.embedDim(),
                .embedFn = struct {
                    fn f(ectx: *anyopaque, g: std.mem.Allocator, text: []const u8, vec: []f32) bool {
                        const m: *generator.GgufModel = @ptrCast(@alignCast(ectx));
                        return m.embedText(g, text, vec);
                    }
                }.f,
            });
            try out.print("  rag        on (dim {d}, k {d})\n", .{ gguf_gen.m.embedDim(), opts.rag_k });
        } else {
            try out.print("  rag        inert: needs the GGUF serve path (no embedder)\n", .{});
        }
    }

    var meter = meter_mod.Meter.init(gpa, io, opts.free_quota);
    defer meter.deinit();

    // OpenAI-compatible HTTP surface (SPEC.md client API), off unless a port is set.
    // Shared generation aggregates: the status line reads them, the serving
    // surfaces write them, and the opt-in alpha reporter ships them.
    var alpha_metrics = alpha.Metrics{ .io = io };

    var openai_ctx: openai.Ctx = undefined;
    if (opts.openai_port != 0) {
        openai_ctx = .{
            .gpa = gpa,
            .io = io,
            .gen = &gen,
            .addr = opts.openai_addr,
            .port = opts.openai_port,
            .seed = opts.seed,
            .model_id = served_model_id,
            .engine_lock = &engine_lock,
            .meter = &meter,
            .synthetic = synthetic,
            .rag = if (rag) |*r| r else null,
            .rag_k = opts.rag_k,
            .console = out,
            .console_lock = &out_lock,
            .alpha_metrics = &alpha_metrics,
        };
        openai_ctx.peers = &table;
        const t = try std.Thread.spawn(.{}, openaiThread, .{&openai_ctx});
        t.detach();
    }

    // Bundled chat UI. A second listener rather than a route on the API port,
    // so the JSON API can stay bound to one interface while the UI stays on
    // loopback (or the reverse). It reuses the same HTTP implementation and
    // the same generator, so the page is same-origin with the API it calls and
    // there is no CORS story and no host to configure in the page.
    var ui_ctx: openai.Ctx = undefined;
    if (opts.ui_port != 0) {
        ui_ctx = .{
            .gpa = gpa,
            .io = io,
            .gen = &gen,
            .addr = opts.ui_addr,
            .port = opts.ui_port,
            .seed = opts.seed,
            .model_id = served_model_id,
            .engine_lock = &engine_lock,
            .meter = &meter,
            .serve_ui = true,
            .peers = &table,
            .synthetic = synthetic,
            .rag = if (rag) |*r| r else null,
            .rag_k = opts.rag_k,
            .console = out,
            .console_lock = &out_lock,
            .alpha_metrics = &alpha_metrics,
        };
        const t = try std.Thread.spawn(.{}, openaiThread, .{&ui_ctx});
        t.detach();
        try out.print("  chat ui    http://{s}:{d}\n", .{ opts.ui_addr, opts.ui_port });
        try out.flush();
    }

    // Periodic console status: membership and holdings move without any
    // request arriving, so an event-only log makes a churning node look idle.
    var status_ctx: status_mod.Reporter = undefined;
    if (opts.status_secs != 0) {
        status_ctx = .{
            .io = io,
            .out = out,
            .out_lock = &out_lock,
            .table = &table,
            .store = if (store) |*st| st else null,
            .gen = &gen,
            .committee = committee_members.len,
            .r_target = opts.r_target,
            .alpha = &alpha_metrics,
            .gpa = gpa,
            .interval_ns = @as(u64, opts.status_secs) * std.time.ns_per_s,
            .start_ns = stats.nowNs(io),
        };
        const t = try std.Thread.spawn(.{}, status_mod.thread, .{&status_ctx});
        t.detach();
    }

    // Opt-in alpha telemetry: report the numeric snapshot to the bootstrap
    // peer once a minute. Reporting without a bootstrap peer is meaningless
    // (there is nobody to tell), so it is silently inert for an origin.
    var alpha_ctx: alpha.ReporterCtx = undefined;
    if (opts.report_metrics and peer_list.items.len > 0) {
        alpha_ctx = .{
            .gpa = gpa,
            .io = io,
            .metrics = &alpha_metrics,
            .store = if (store) |*st| st else null,
            .table = &table,
            .target = peer_list.items[0],
            .network_id = network_id,
            .version = @import("build_info").version,
            .hold_fraction = opts.hold_fraction,
            .boot_id = @truncate(@as(u128, @bitCast(stats.nowNs(io)))),
            .started_ns = stats.nowNs(io),
        };
        const t = try std.Thread.spawn(.{}, alpha.reporterLoop, .{&alpha_ctx});
        t.detach();
        try out.print("  metrics    reporting numeric telemetry to {s} every 60s (docs/ALPHA.md)\n", .{peer_strs.items[0]});
        try out.flush();
    }

    var rpc_ctx = rpc.Ctx{
        .gpa = gpa,
        .io = io,
        .gen = &gen,
        .addr = opts.rpc_addr,
        .port = opts.rpc_port,
        .seed = opts.seed,
        .meter = &meter,
        .admin_token = opts.admin_token,
        .engine_lock = &engine_lock,
        .alpha_metrics = &alpha_metrics,
        .console = out,
        .console_lock = &out_lock,
    };
    rpc.serve(&rpc_ctx) catch |e| {
        // Do not return: the loops above are detached and still hold pointers
        // into this frame (security issue #31).
        out.print("rpc server error: {s}\n", .{@errorName(e)}) catch {};
        out.flush() catch {};
        fatal("node: shutting down\n", .{});
    };
}
