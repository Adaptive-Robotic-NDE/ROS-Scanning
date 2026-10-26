#!/usr/bin/env python3
#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# Copyright 2023-2024 University of Dayton                              #
#                                                                       #
# Redistribution and use in source and binary forms, with or without    #
# modification, are permitted provided that the following conditions    #
# are met:                                                              #
#                                                                       #
# 1. Redistributions of source code must retain the above copyright     #
#    notice, this list of conditions and the following disclaimer.      #
# 2. Redistributions in binary form must reproduce the above copyright  #
#    notice, this list of conditions and the following disclaimer in    #
#    the documentation and/or other materials provided with the         #
#    distribution.                                                      #
# 3. Neither the name of the copyright holder nor the names of its      #
#    contributors may be used to endorse or promote products derived    #
#    from this software without specific prior written permission.      #
#                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   #
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,#
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT      #
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY #
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE #
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  #
#########################################################################
# This material is based in part upon work supported by the U.S. Air    #
# Force under Contract FA8650-19-D-5230.                                #
#                                                                       #
# DISTRIBUTION A. Approved for public release: distribution unlimited.  #
#########################################################################
# File:  rviz_probe.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Measures RViz rendering cost to find what makes it slow.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-09-29:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""Why is RViz slow? Measure it, instead of guessing on the production system.

The failure this is built around: RViz is usable all session and then, from the
Keyence measurement onward, renders about one frame every few seconds and never
recovers. It is not a leak and not a message rate. The fine-mesh pipeline
publishes each of its stage clouds ONCE, at /stop_and_process, and an RViz
PointCloud2 display with Decay Time 0 keeps the last message FOREVER -- "decay
0" means "never expire", not "clear". So seven displays that cost nothing all
session begin costing every frame at that moment and keep costing it. The
largest, /sanitized_pointcloud, carries the RAW cloud despite its name: millions
of points on a full-size plate. The fix is in view_robot.rviz; the whole
write-up is docs/rviz-performance.md.

WHAT THE THREE SUBCOMMANDS MEASURE, and how to read them:

    rviz_probe.py displays [CONFIG.rviz]     which displays are enabled, and an
                                             estimate of what each would hold at
                                             ~300 bytes per point. No ROS needed.
                                             Any stage topic reading ON is the
                                             fault, still present.

    rviz_probe.py census [--secs N]          point count and estimated RViz cost
                                             of the last message each display
                                             topic carried. The stage topics
                                             publish once, so run it right after
                                             /stop_and_process. A topic with a
                                             publisher and no message received
                                             means "too big to sample here", not
                                             "empty" -- see the closing note it
                                             prints.

    rviz_probe.py watch [--csv FILE]         RViz's CPU and RSS once a second
                                             until Ctrl-C, flagging step growth.
                                             Run it across a whole scan. RSS flat
                                             across the Keyence measurement means
                                             fixed; a step that never comes down
                                             means something still holds a cloud;
                                             a gradual climb is a different fault
                                             (a real leak).

RSS and its permanence are what `watch` measures because they are
GPU-independent. Absolute frame rates and CPU percentages are not comparable
between a software-GL container and a cell PC with a discrete GPU; the SHAPE --
a step at a known moment that never comes back -- is.

Linux only (it reads /proc). `census` needs ROS sourced; the others do not.
"""
import argparse
import os
import re
import sys
import time

#: Where the shipped config lives on a controlling computer. Overridable on the
#: command line.
DEFAULT_CONFIG = (
    "ros_ws/src/tari_ros/tari_ur_description/rviz/view_robot.rviz")

#: Bytes RViz spends per point of a PointCloud2 display: 864 MB of RSS for one
#: 2,850,000-point message. It is ~25x the 12 bytes the message itself carries,
#: because the display builds per-point billboard geometry. Used only to
#: ESTIMATE what a display would cost before you enable it; `watch` measures the
#: real thing.
BYTES_PER_POINT = 303.0


# ==============================================================================
# /proc, so there is no psutil dependency on a controlling computer
# ==============================================================================

def find_pid(name="rviz2"):
    """The PID of the running RViz, or None. Matches the executable, not the
    command line -- matching a command line is how a probe kills its own
    shell."""
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/comm" % d) as f:
                if f.read().strip() == name:
                    return int(d)
        except OSError:
            continue
    return None


def cpu_ticks(pid):
    """utime + stime, in clock ticks."""
    with open("/proc/%d/stat" % pid) as f:
        parts = f.read().rsplit(")", 1)[1].split()
    # After the ") " the fields are state, ppid, ...; utime is field 14 and
    # stime 15 counting from 1 at `pid`, so index 11 and 12 here.
    return int(parts[11]) + int(parts[12])


def rss_mb(pid):
    with open("/proc/%d/status" % pid) as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])/1024.0
    return float("nan")


# ==============================================================================
# displays -- read the config, no ROS
# ==============================================================================

def read_displays(path):
    """Every PointCloud2/Marker/MarkerArray display: (enabled, group, name,
    topic, decay).

    Parsed with yaml when yaml parses it, and by hand when it does not --
    tari_ur_soft_launch's copy of view_robot.rviz is malformed (it fails at
    line 796), and a diagnostic that dies on the file it was pointed at is
    not a diagnostic.
    """
    try:
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f)
    except Exception as e:                                  # noqa: BLE001
        return _read_displays_by_hand(path, why=str(e).splitlines()[0])

    out = []

    def walk(items, group=""):
        for x in items or []:
            if x.get("Class") == "rviz_common/Group":
                walk(x.get("Displays"), x.get("Name") or "")
                continue
            cls = (x.get("Class") or "").split("/")[-1]
            if cls in ("PointCloud2", "Marker", "MarkerArray"):
                topic = x.get("Topic")
                topic = topic.get("Value") if isinstance(topic, dict) else topic
                out.append({"enabled": bool(x.get("Enabled")), "group": group,
                            "name": x.get("Name"), "topic": topic, "class": cls,
                            "decay": x.get("Decay Time")})
    walk(doc["Visualization Manager"]["Displays"])
    return out


def _read_displays_by_hand(path, why=""):
    """A line scan, for a config yaml will not parse."""
    out, cur = [], {}
    for line in open(path):
        s = line.strip()
        m = re.match(r"^Class: rviz_default_plugins/(PointCloud2|Marker|MarkerArray)$", s)
        if m:
            cur = {"class": m.group(1), "group": "", "decay": None}
            out.append(cur)
        elif cur is not None and out:
            if s.startswith("Name: ") and "name" not in out[-1]:
                out[-1]["name"] = s[6:]
            elif s.startswith("Enabled: ") and "enabled" not in out[-1]:
                out[-1]["enabled"] = s.endswith("true")
            elif s.startswith("Decay Time: ") and out[-1].get("decay") is None:
                out[-1]["decay"] = s.split(": ", 1)[1]
            elif s.startswith("Value: /") and "topic" not in out[-1]:
                out[-1]["topic"] = s[7:]
    if why:
        print("  (this config is not valid YAML -- %s -- so it was read by hand "
              "and the grouping is lost)" % (why,))
    return [d for d in out if d.get("name")]


def cmd_displays(args):
    path = args.config or DEFAULT_CONFIG
    if not os.path.exists(path):
        sys.exit("no such config: %s\nPass the path RViz was started with "
                 "(-d ...)." % (path,))
    rows = read_displays(path)
    print("%s\n" % (path,))
    print("  %-4s %-26s %-34s %s" % ("", "GROUP", "TOPIC", "NAME"))
    on = 0
    for d in rows:
        on += bool(d.get("enabled"))
        print("  %-4s %-26s %-34s %s"
              % ("ON" if d.get("enabled") else "off", (d.get("group") or "")[:26],
                 d.get("topic"), d.get("name")))
    print("\n  %d display(s) that hold geometry, %d enabled." % (len(rows), on))
    print("""
  WHAT TO LOOK AT. A PointCloud2 display with Decay Time 0 keeps the LAST
  message forever -- it does not clear. Any display whose topic is published
  once, with a big cloud, costs every frame from that moment until RViz is
  restarted. The fine-mesh stage topics are exactly that shape, and
  /sanitized_pointcloud carries the RAW cloud despite its name.

  At about %.0f bytes per point, a display holding N points costs roughly
  N x %.0f bytes of RSS: 1M points is ~%.0f MB.""" % (
        BYTES_PER_POINT, BYTES_PER_POINT, 1e6*BYTES_PER_POINT/1e6))
    return 0


# ==============================================================================
# census -- what is each topic actually carrying?
# ==============================================================================

def cmd_census(args):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as e:
        sys.exit("census needs ROS sourced (%s).\n"
                 "  source /opt/ros/iron/setup.bash && source ros_ws/install/setup.bash"
                 % (e,))

    path = args.config or DEFAULT_CONFIG
    rows = read_displays(path) if os.path.exists(path) else []
    wanted = [d for d in rows if d.get("topic")]
    if not wanted:
        sys.exit("no displays found in %s" % (path,))

    rclpy.init()
    node = Node("rviz_probe_census")
    seen = {}

    def on(topic, kind):
        def cb(msg):
            if kind == "PointCloud2":
                n = int(msg.width)*int(msg.height)
                nbytes = len(msg.data)
            elif kind == "Marker":
                n = len(msg.points)
                nbytes = 32*n
            else:
                n = sum(len(m.points) for m in msg.markers)
                nbytes = 32*n
            prev = seen.get(topic)
            seen[topic] = {"n": n, "bytes": nbytes,
                           "count": (prev["count"] + 1) if prev else 1}
        return cb

    # RELIABLE AND DEPTH 10, both deliberately. Neither is negotiable, and both
    # fail silently the wrong way round.
    #
    # BEST_EFFORT looks like the polite choice for a census -- do not perturb
    # what you measure -- but a DDS subscription's reliability is its own and
    # takes nothing from RViz's, and BEST_EFFORT never reassembles a very large
    # message because a 34 MB cloud is fragmented and lost fragments are not
    # retransmitted. With BEST_EFFORT this reports 0 messages on
    # /sanitized_pointcloud while 2,500,000-point clouds are published on it,
    # and 29 on the small /accumulated_cloud: it misses precisely the topic it
    # exists to find.
    #
    # Depth 1 fails the same way for the same reason -- a one-deep reliable
    # reader cannot hold a 30 MB fragmented sample while it reassembles -- where
    # a depth-10 subscriber alongside it receives 4 of 6.
    qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE)
    types = {"PointCloud2": PointCloud2, "Marker": Marker, "MarkerArray": MarkerArray}
    for d in wanted:
        node.create_subscription(types[d["class"]], d["topic"],
                                 on(d["topic"], d["class"]), qos)

    print("listening for %d s on %d topic(s)...\n" % (args.secs, len(wanted)))
    t0 = time.time()
    while time.time() - t0 < args.secs:
        rclpy.spin_once(node, timeout_sec=0.2)

    print("  %-4s %-34s %12s %10s %8s  %s"
          % ("", "TOPIC", "POINTS", "EST. RSS", "MSGS", "NAME"))
    total = 0.0
    silent_with_publisher = []
    for d in wanted:
        s = seen.get(d["topic"])
        if not s:
            # A PUBLISHER WITH NOTHING RECEIVED IS ITSELF A RESULT: either
            # the topic did not publish in the window (the stage topics
            # publish once, at /stop_and_process) or the sample is too big
            # for this process to reassemble. See the closing note.
            npub = node.count_publishers(d["topic"])
            if npub:
                silent_with_publisher.append(d["topic"])
            print("  %-4s %-34s %12s %10s %8s  %s"
                  % ("ON" if d["enabled"] else "off", d["topic"],
                     "(%d pub)" % npub if npub else "-", "-", 0, d["name"]))
            continue
        est = s["n"]*BYTES_PER_POINT/1e6
        if d["enabled"]:
            total += est
        print("  %-4s %-34s %12d %9.0fM %8d  %s"
              % ("ON" if d["enabled"] else "off", d["topic"], s["n"], est,
                 s["count"], d["name"]))
    print("\n  Enabled displays that reported a message would hold about "
          "%.0f MB between them." % (total,))
    print("  Nothing seen on a topic only means it did not publish during these "
          "%d s --\n  the stage topics publish ONCE, at /stop_and_process, so run "
          "this right after\n  a Keyence measurement to see them." % (args.secs,))
    if silent_with_publisher:
        note = [
            "",
            "  %d topic(s) HAVE a publisher and delivered nothing here:"
            % (len(silent_with_publisher),),
            "      " + "\n      ".join(silent_with_publisher),
            "",
            "  READ THAT AS \"probably too big to sample\", not as \"empty\". A",
            "  multi-million point cloud is a 30 MB sample fragmented across thousands",
            "  of UDP datagrams, and a subscriber that misses one fragment never",
            "  reassembles it. Against a 2,500,000-point cloud: a node with ONE",
            "  subscription gets 4 of 6 messages; the same node with a second",
            "  subscription on the 30 Hz /profiles gets 2; a census with all %d gets 0."
            % (len(wanted),),
            "  RViz does receive them -- it is a C++ subscriber with its own buffers",
            "  that has been running for minutes.",
            "",
            "  THE AUTHORITATIVE SIZE IS ON DISK. /stop_and_process writes every stage",
            "  as a .ply beside the scan, and the dgpy log prints the stage table with",
            "  exact counts (\"raw ... 340680 points\"). Use those to decide what a",
            "  display would cost; use `watch` to measure what it did cost.",
        ]
        print("\n".join(note))
    node.destroy_node()
    rclpy.shutdown()
    return 0


# ==============================================================================
# watch -- the measurement that actually settles it
# ==============================================================================

def cmd_watch(args):
    pid = args.pid or find_pid()
    if pid is None:
        sys.exit("no rviz2 process found. Start RViz first, or pass --pid.")
    hz = os.sysconf("SC_CLK_TCK")
    print("watching rviz2 (pid %d). Ctrl-C to stop.\n" % (pid,))
    print("  %8s %8s %9s   %s" % ("ELAPSED", "CPU %", "RSS MB", "NOTE"))

    out = open(args.csv, "w") if args.csv else None
    if out:
        out.write("elapsed_s,cpu_pct,rss_mb,step\n")
    t0 = time.time()
    last_t, last_ticks = time.time(), cpu_ticks(pid)
    base_rss = None
    peak_rss = 0.0        # both set on the first sample, below
    try:
        while True:
            time.sleep(args.interval)
            try:
                now, ticks, rss = time.time(), cpu_ticks(pid), rss_mb(pid)
            except (OSError, IOError):
                print("  rviz2 (pid %d) has gone." % (pid,))
                break
            dt = now - last_t
            cpu = 100.0*(ticks - last_ticks)/hz/dt if dt > 0 else 0.0
            last_t, last_ticks = now, ticks
            if base_rss is None:
                # BOTH, on the first sample. peak_rss must not be left at 0
                # here, or the very first line reports the whole of RSS as a
                # step -- exactly the false alarm this column exists to avoid.
                base_rss = peak_rss = rss
            # A STEP, not a drift. The fault this script is for is a permanent
            # jump at one moment, so that is what gets flagged: growth of more
            # than `step_mb` since the last sample.
            step = ""
            if rss - peak_rss > args.step_mb:
                step = "STEP +%.0f MB (now +%.0f MB over the start)" % (
                    rss - peak_rss, rss - base_rss)
            peak_rss = max(peak_rss, rss)
            # flush=True: this is meant to be watched live and piped to a
            # file at the same time, and a buffered stdout loses everything
            # when the run is ended with `timeout` (SIGTERM, no flush) rather
            # than Ctrl-C.
            print("  %8.0f %8.1f %9.0f   %s" % (now - t0, cpu, rss, step),
                  flush=True)
            if out:
                out.write("%.1f,%.1f,%.1f,%s\n"
                          % (now - t0, cpu, rss, step.replace(",", ";")))
                out.flush()
    except KeyboardInterrupt:
        pass
    if out:
        out.close()
        print("\n  written to %s" % (args.csv,))
    if base_rss is not None:
        print("\n  RSS went from %.0f MB to %.0f MB (+%.0f MB) and RViz does not "
              "give it back." % (base_rss, peak_rss, peak_rss - base_rss))
        print("  A step at the Keyence measurement that never comes down is the "
              "fault described in\n  docs/rviz-performance.md. Note WHEN it "
              "stepped and run `census` to see which topic did it.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("displays", help="which displays are on (no ROS needed)")
    d.add_argument("config", nargs="?", default=None)
    d.set_defaults(func=cmd_displays)

    c = sub.add_parser("census", help="what each display's topic is carrying")
    c.add_argument("--config", default=None)
    c.add_argument("--secs", type=int, default=20)
    c.set_defaults(func=cmd_census)

    w = sub.add_parser("watch", help="sample RViz CPU and RSS over a run")
    w.add_argument("--pid", type=int, default=None)
    w.add_argument("--interval", type=float, default=1.0)
    w.add_argument("--step-mb", type=float, default=50.0, dest="step_mb")
    w.add_argument("--csv", default=None)
    w.set_defaults(func=cmd_watch)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
