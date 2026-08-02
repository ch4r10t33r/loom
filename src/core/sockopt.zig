//! Connection deadlines (security issue #24).
//!
//! Zig 0.16's `std.Io.net` has no read deadline for TCP streams:
//! `Stream.Reader.readVec` calls `netRead` with no timeout (only datagram
//! `receiveTimeout` and `connect` accept one). Without a deadline every accept
//! loop's connection cap becomes a DoS budget: a peer that connects and sends
//! nothing holds a handler slot forever.
//!
//! `SO_RCVTIMEO` is NOT usable here. The `Threaded` backend issues blocking
//! `recvmsg` and treats `EAGAIN` as a programmer bug: `netReadPosix` calls
//! `errnoBug(AGAIN)`, which panics the process. Setting the socket option turns
//! an idle connection into a remote crash, which is worse than the DoS.
//!
//! So deadlines are enforced out-of-band: connections register here, and a
//! single reaper thread `shutdown()`s any that outlive their deadline. A
//! blocked `recvmsg` on a shut-down socket returns 0, which surfaces as
//! `error.EndOfStream` and unwinds the handler through its existing error path.
//! Shutdown (not close) is deliberate: closing an fd another thread is blocked
//! on yields `EBADF`, which is another `errnoBug` panic.
//!
//! fd-reuse safety: the reaper holds `mutex` while shutting a socket down, and
//! `untrack` takes the same mutex *before* the handler closes its stream. A
//! handler therefore cannot close (and the OS cannot recycle) an fd while the
//! reaper is acting on it.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const stats = @import("stats.zig");

/// Inbound connections: a client that stalls mid-request is dropped.
pub const SERVE_TIMEOUT_S: i64 = 30;

/// Deadline for a connection whose request has been fully parsed and is being
/// answered.
///
/// `SERVE_TIMEOUT_S` guards the *read* phase — a slowloris opens a socket and
/// dribbles headers forever (security issue #24). Once a complete request has
/// been read, that risk is gone and the connection is doing work. Leaving the
/// 30 s bound in place instead kills legitimate answers: measured across two
/// machines on a 27 ms link, a distributed request could not produce its first
/// token inside 30 s, so every one died mid-prefill with no error the client
/// could distinguish from a crash.
///
/// Refreshing per token is not enough on its own, because there is no token to
/// refresh on until prefill completes. So the phase change is explicit: parsed
/// request -> generous deadline, then each token pushes it out again.
///
/// An hour, not 900 s: a cold prefill against the devnet's 106B model at a
/// 20% hold fraction moves gigabytes of experts over a home link before the
/// first token exists, and 900 s killed real requests. The bound is a
/// last-resort reaper for wedged connections, not a performance target;
/// concurrency stays bounded by the connection cap either way.
pub const GENERATE_TIMEOUT_S: i64 = 3600;

/// Outbound peer dials: a peer that accepts and then goes silent must not hang
/// gossip, repair, heartbeat, or the token-loop expert fetch.
pub const PEER_TIMEOUT_S: i64 = 10;

// NOTE: `ConnectOptions.timeout` exists in the std API but is NOT implemented
// by the Threaded backend: `netConnectIpPosix` does
// `if (options.timeout != .none) @panic("TODO implement ...")`. Passing one
// crashes on the first dial, so connects fall back to the OS TCP timeout. The
// reaper below cannot cover the handshake either (there is no socket to track
// until connect returns); a blackholed SYN therefore still blocks the dialling
// thread for the platform default. Post-connect reads ARE covered.

const MAX_TRACKED = 4096;
const REAP_INTERVAL_NS: i96 = 2 * std.time.ns_per_s;

const Slot = struct {
    active: bool = false,
    stream: net.Stream = undefined,
    deadline_ns: i128 = 0,
};

var mutex: Io.Mutex = .init;
var slots: [MAX_TRACKED]Slot = [_]Slot{.{}} ** MAX_TRACKED;
var reaper_started = std.atomic.Value(bool).init(false);

/// Start the reaper once per process. Safe to call from every accept loop.
pub fn ensureReaper(io: Io) void {
    if (reaper_started.swap(true, .monotonic)) return;
    const t = std.Thread.spawn(.{}, reaperLoop, .{io}) catch {
        reaper_started.store(false, .monotonic);
        return;
    };
    t.detach();
}

fn reaperLoop(io: Io) void {
    while (true) {
        Io.sleep(io, .{ .nanoseconds = REAP_INTERVAL_NS }, .awake) catch return;
        const now = stats.nowNs(io);
        mutex.lockUncancelable(io);
        defer mutex.unlock(io); // held across shutdown: see fd-reuse note above
        for (&slots) |*s| {
            if (!s.active or now < s.deadline_ns) continue;
            s.stream.shutdown(io, .both) catch {};
            s.active = false; // shut down once; the handler still owns the close
        }
    }
}

/// Register `stream` to be shut down `seconds` from now. Returns a slot handle
/// for `untrack`, or null if the table is full (deadline simply not enforced).
pub fn track(io: Io, stream: net.Stream, seconds: i64) ?usize {
    const deadline = stats.nowNs(io) + @as(i128, seconds) * std.time.ns_per_s;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    for (&slots, 0..) |*s, i| {
        if (s.active) continue;
        s.* = .{ .active = true, .stream = stream, .deadline_ns = deadline };
        return i;
    }
    return null;
}

/// Push a tracked connection's deadline out by `seconds` from now.
///
/// The deadline exists to kill connections that are *idle* — a slowloris holds
/// a socket open sending nothing (security issue #24). Applied to the whole
/// connection lifetime it kills working ones too: a generation slower than
/// SERVE_TIMEOUT_S has its socket shut down mid-answer, with no error the
/// client can distinguish from a crash. Measured across two machines on a
/// 27 ms link, every distributed request died at exactly 30 s regardless of
/// how much progress it had made — which is precisely the regime this project
/// exists to serve.
///
/// So progress refreshes the deadline. A connection that is producing output
/// is not idle; one that has gone quiet still reaps on schedule.
pub fn refresh(io: Io, slot: ?usize, seconds: i64) void {
    const i = slot orelse return;
    const deadline = stats.nowNs(io) + @as(i128, seconds) * std.time.ns_per_s;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (slots[i].active) slots[i].deadline_ns = deadline;
}

/// Refresh an inbound connection by the generation timeout. Used once when a
/// request has been parsed, and again on every token produced.
pub fn refreshServe(io: Io, slot: ?usize) void {
    refresh(io, slot, GENERATE_TIMEOUT_S);
}

/// Release a slot. MUST be called before the handler closes its stream.
pub fn untrack(io: Io, slot: ?usize) void {
    const i = slot orelse return;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    slots[i].active = false;
}

/// Track an accepted (inbound) connection.
pub fn trackServe(io: Io, stream: net.Stream) ?usize {
    return track(io, stream, SERVE_TIMEOUT_S);
}

/// Track a connection we opened to a peer (outbound).
pub fn trackPeer(io: Io, stream: net.Stream) ?usize {
    return track(io, stream, PEER_TIMEOUT_S);
}
