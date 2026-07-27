//! Startup banner for the long-running commands (`loom node`, `loom light`).
//!
//! It exists to answer one question at a glance, in a terminal or three months
//! later in a log file: *which build is this?* Nodes get left running, binaries
//! get copied between machines, and a swarm can easily end up mixed — so the
//! version, the commit and the target triple are printed before anything else
//! happens.
//!
//! Deliberately plain text with no ANSI colour: this output is as likely to be
//! read from a redirected log or a container's stdout as from a terminal.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const info = @import("build_info");

/// Short form of the build commit for display; the full hash stays in
/// `loom version`.
pub fn shortCommit() []const u8 {
    const c = info.commit;
    return if (c.len > 12) c[0..12] else c;
}

/// Print the banner. `role` names the command it is booting ("node", "light").
pub fn print(out: *Io.Writer, role: []const u8) !void {
    try out.print(
        \\
        \\   ██╗      ██████╗  ██████╗ ███╗   ███╗
        \\   ██║     ██╔═══██╗██╔═══██╗████╗ ████║
        \\   ██║     ██║   ██║██║   ██║██╔████╔██║
        \\   ██║     ██║   ██║██║   ██║██║╚██╔╝██║
        \\   ███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
        \\   ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
        \\   distributed expert cache for large MoE inference
        \\
        \\
    , .{});
    try out.print("   version  {s}\n", .{info.version});
    try out.print("   commit   {s}\n", .{shortCommit()});
    try out.print("   build    {s}-{s} {s}, zig {s}\n", .{
        @tagName(builtin.cpu.arch),
        @tagName(builtin.os.tag),
        @tagName(builtin.mode),
        builtin.zig_version_string,
    });
    try out.print("   role     {s}\n\n", .{role});
    try out.flush();
}

test "short commit truncates a full hash but leaves a placeholder alone" {
    // The build stamps a 40-char sha in CI and the literal "unknown" locally;
    // both have to render sensibly in the banner.
    const s = shortCommit();
    try std.testing.expect(s.len <= 12);
    try std.testing.expect(s.len > 0);
    try std.testing.expect(std.mem.startsWith(u8, info.commit, s));
}
