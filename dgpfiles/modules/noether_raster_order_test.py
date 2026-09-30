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
# File:  noether_raster_order_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the order of travel orient_segments_along() gives the passes of
#   a toolpath: raster (serpentine) or comb.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-09-29:  Initial Commit
#
#########################################################################
"""Which way each pass of a toolpath runs, and what that must not change.

orient_segments_along() in noether_path.dpi is the one place the direction of
travel of a pass is decided. Two orders matter:

  RASTER (serpentine)  successive passes run in opposite directions, so each
                       starts where the last one ended and the move between them
                       is one line spacing sideways.
  COMB                 every pass runs the same way, so the probe is carried the
                       whole length of the part back to the same side between
                       passes. Correct when the reading depends on the direction
                       of travel; otherwise it is pure overhead -- on an 81-pass
                       eddy-current plan, 6.8 minutes of a 19.5 minute scan.

The raster is the default, and the invariant that makes it safe is that
reversing a pass changes only the ORDER of its waypoints. Each waypoint keeps
the orientation it was planned with, so the tool frame at any given point on the
surface is the same in either direction. A frame recomputed from the direction of
travel flips 180 degrees at every turn, which points the tool z into the part.

Runs on the host with numpy only: the pure functions are pulled out of the .dpi
and geometry_msgs is stubbed for ScanPath.

    python3 noether_raster_order_test.py    # every check, exit 1 on failure
    python3 -m pytest -q noether_raster_order_test.py

Both modes run the same checks. Standalone runs every section even after one
fails; under pytest each section is its own test.
"""
import ast
import functools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve on a host with no ROS. Nothing here builds a message, so ensure_msgs()
# only has to make the import work, and it ADDS rather than overwriting whatever
# another suite in this directory has already installed.
ros_msg_stubs.ensure_msgs()

DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "def X" is a SyntaxError, so strip those lines.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "noether_raster_order_test", "os": os, "np": np}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
# noether_path.dpi reports progress through progress.dpi, which the config
# includes ahead of it. Silenced: this suite is about waypoint order.
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
ns["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)

orient = ns["orient_segments_along"]
ScanPath = ns["ScanPath"]
locked_quats = ns["locked_quats_from_axes"]
quat_to_matrix = ns["quat_to_matrix"]
twist = ns["tool_twist_deg"]

#: SCAN_LINE_SPACING in ECAutomatedScan.dgp: the distance BETWEEN passes.
LINE_SPACING = 0.0003
#: NOETHER_ROLL_REFERENCE: the tool x axis the eddy-current roll is locked to.
REFERENCE = (1.0, 0.0, 0.0)
#: EC_TRAVEL_DIRECTION: the direction of the first pass.
TRAVEL = (-1.0, 0.0, 0.0)
#: One pass of the observed 81-pass plan: 7.646 m of travel over 81 passes.
PASS_LENGTH_M = 0.094
NPASS = 8
NPOINT = 12

FAILS = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILS)
        fn(*args, **kwargs)
        failed = FAILS[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


def comb_path():
    """The planner's own output: every pass running +x, one segment per pass.

    The surface tilts along the pass and across it, so every waypoint has a
    DIFFERENT orientation and no two passes share one. That is what makes the
    frame check below able to fail: pairing an orientation with the wrong
    waypoint, or with the wrong end of a pass, is visible.
    """
    pos, axes, starts = [], [], []
    for j in range(NPASS):
        starts.append(len(pos))
        for i in range(NPOINT):
            f = i/float(NPOINT - 1)
            x = 0.30 + f*PASS_LENGTH_M
            y = j*LINE_SPACING
            pos.append((x, y, 0.99))
            t = np.radians(4.0 + 16.0*f + 1.5*j)         # tilt along the pass
            axes.append((np.sin(t), 0.1*np.sin(t), np.cos(t)))
    axes = np.asarray(axes, dtype=np.float64)
    axes = axes/np.linalg.norm(axes, axis=1, keepdims=True)
    return ScanPath(np.array(pos, dtype=np.float64), locked_quats(axes, REFERENCE),
                    starts, source="test plate")


def pass_directions(path):
    """Unit vector from the first to the last waypoint of each pass."""
    out = []
    for a, b in path.segments:
        d = path.positions[b-1] - path.positions[a]
        out.append(d/np.linalg.norm(d))
    return out


def frame_by_position(path):
    """{position -> tool rotation matrix}, keyed to the nearest nanometre.

    The whole point of the invariant: a physical point on the surface has ONE
    tool frame, whichever direction the pass through it runs.
    """
    out = {}
    for p, q in zip(path.positions, path.quats):
        out[tuple(np.round(p, 9))] = quat_to_matrix(q)
    return out


def worst_frame_difference(a, b):
    """Largest rotation, in degrees, between two {position -> frame} maps.

    Positions present in only one of them count as a total mismatch, because
    that means the reordering moved a waypoint rather than just reordering it.
    """
    if set(a) != set(b):
        return 180.0
    worst = 0.0
    for key, Ra in a.items():
        c = (float(np.trace(Ra.T @ b[key])) - 1.0)/2.0
        worst = max(worst, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return worst


@_checked
def test_successive_passes_run_in_opposite_directions():
    print("\n--- a raster alternates the direction of travel --------------------")
    raster = orient(comb_path(), TRAVEL, serpentine=True)
    dirs = pass_directions(raster)
    check(len(dirs) == NPASS, "all %d passes survive the reordering (%d)"
          % (NPASS, len(dirs)))
    check(np.allclose(dirs[0], TRAVEL, atol=1e-9),
          "the first pass travels along EC_TRAVEL_DIRECTION %s (%s)"
          % (TRAVEL, np.round(dirs[0], 6)))
    dots = [float(np.dot(dirs[k], dirs[k+1])) for k in range(len(dirs) - 1)]
    check(all(d < -0.999 for d in dots),
          "every pass runs opposite to the one before it (worst dot %.6f, want -1)"
          % (max(dots),))
    check(raster.serpentine is True and raster.segments_reversed == NPASS//2,
          "the path records that it was rastered, and half the passes were "
          "reversed to do it (%s, %d)"
          % (raster.serpentine, raster.segments_reversed))


@_checked
def test_reversing_a_pass_does_not_flip_the_tool_frame():
    print("\n--- reversing a pass reorders waypoints and nothing else -----------")
    comb = comb_path()
    raster = orient(comb, TRAVEL, serpentine=True)
    check(len(raster) == len(comb),
          "no waypoint is gained or lost (%d in, %d out)" % (len(comb), len(raster)))
    worst = worst_frame_difference(frame_by_position(comb), frame_by_position(raster))
    check(worst < 1e-9,
          "every point on the surface keeps the tool frame it was planned with "
          "(worst difference %.2e deg)" % (worst,))
    # The failure this guards against: a frame taken from the direction of travel
    # flips 180 degrees at a turn, which puts the tool z into the part.
    # A flip about the tool x axis would make this negative on every pose of
    # every reversed pass; the planned surface tilts up to 30 degrees, so the
    # honest value is the cosine of that, not 1.
    up = np.array([quat_to_matrix(q)[2, 2] for q in raster.quats])
    check(float(up.min()) > 0.0,
          "no pose has its tool z turned into the surface (smallest world z "
          "component %.4f, want positive)" % (float(up.min()),))
    check(float(np.abs(twist(raster.quats, reference=REFERENCE)).max()) < 1e-6,
          "the roll is still locked to the reference: nothing turned about the "
          "tool axis")
    # And the orientation step the arm pays at a turn is the step between two
    # neighbouring points, not a half turn.
    turns = []
    for a, b in raster.segments[:-1]:
        Ra, Rb = quat_to_matrix(raster.quats[b-1]), quat_to_matrix(raster.quats[b])
        c = (float(np.trace(Ra.T @ Rb)) - 1.0)/2.0
        turns.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    check(max(turns) < 5.0,
          "the wrist turns %.2f deg at the worst pass boundary, not 180"
          % (max(turns),))


@_checked
def test_each_pass_starts_where_the_last_one_ended():
    print("\n--- and so the move between passes is one line spacing -------------")
    comb = orient(comb_path(), TRAVEL, serpentine=False)
    raster = orient(comb_path(), TRAVEL, serpentine=True)
    gaps = []
    for k in range(len(raster.segments) - 1):
        a, b = raster.segments[k]
        gaps.append(float(np.linalg.norm(raster.positions[raster.segments[k+1][0]]
                                        - raster.positions[b-1])))
    check(max(gaps) <= LINE_SPACING + 1e-9,
          "the end of each pass is within one line spacing (%.2f mm) of the start "
          "of the next (worst %.4f mm)" % (LINE_SPACING*1e3, max(gaps)*1e3))
    comb_gaps = [float(np.linalg.norm(comb.positions[comb.segments[k+1][0]]
                                     - comb.positions[comb.segments[k][1]-1]))
                 for k in range(len(comb.segments) - 1)]
    check(min(comb_gaps) > 0.9*PASS_LENGTH_M,
          "a comb has to be carried the length of a pass instead (%.1f mm)"
          % (min(comb_gaps)*1e3,))
    # Arc length over the WHOLE path, link steps included, is what the arm flies.
    scanning = sum(raster.segment_length_m(s) for s in raster.segments)
    check(abs(raster.total_length_m() - scanning) < NPASS*LINE_SPACING,
          "the rastered path is the scanning travel plus the line spacings "
          "(%.3f m flown for %.3f m scanned)"
          % (raster.total_length_m(), scanning))
    check(raster.total_length_m() < 0.55*comb.total_length_m(),
          "which is %.3f m against the comb's %.3f m for the same coverage"
          % (raster.total_length_m(), comb.total_length_m()))


@_checked
def test_the_comb_order_is_still_reachable():
    print("\n--- and the comb is still one setting away -------------------------")
    comb = orient(comb_path(), TRAVEL, serpentine=False)
    dirs = pass_directions(comb)
    check(all(np.allclose(d, TRAVEL, atol=1e-9) for d in dirs),
          "with serpentine off every pass travels along EC_TRAVEL_DIRECTION %s"
          % (TRAVEL,))
    check(comb.serpentine is False and comb.segments_reversed == NPASS,
          "all %d passes were reversed onto that direction (%d)"
          % (NPASS, comb.segments_reversed))
    worst = worst_frame_difference(frame_by_position(comb_path()),
                                  frame_by_position(comb))
    check(worst < 1e-9,
          "and a comb keeps the planned tool frames too (worst difference %.2e deg)"
          % (worst,))


@_checked
def test_the_eddy_current_config_asks_for_a_raster():
    print("\n--- the config's own wiring ----------------------------------------")
    src = open(DGP).read()
    tree = ast.parse(src, filename="ECAutomatedScan.dgp")
    values = [n.value for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "EC_SERPENTINE"
                      for t in n.targets)]
    check(len(values) == 1, "EC_SERPENTINE is set exactly once (%d)" % (len(values),))
    check(bool(values) and getattr(values[0], "value", None) is True,
          "and it defaults to the raster order")
    calls = [c for c in ast.walk(tree)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "orient_segments_along"]
    check(len(calls) == 1,
          "the passes are ordered in exactly one place (%d calls)" % (len(calls),))
    check(bool(calls) and any(k.arg == "serpentine"
                             and getattr(k.value, "id", None) == "EC_SERPENTINE"
                             for k in calls[0].keywords),
          "and that call is the one EC_SERPENTINE steers")
    # A comb's return is a long move between passes; a raster's is one line
    # spacing, which is inside the short-hop threshold, so the approach must not
    # be forced to the long one.
    check("EC_SHORT_HOP_M" in src and "EC_APPROACH_M" in src,
          "the between-pass approach still chooses by distance (EC_SHORT_HOP_M)")


@_checked
def test_not_blind():
    print("\n--- the suite is not blind -----------------------------------------")
    check(orient.__code__.co_filename == "noether_path.dpi",
          "the subject came out of noether_path.dpi, not a stand-in")
    a = orient(comb_path(), TRAVEL, serpentine=True)
    b = orient(comb_path(), TRAVEL, serpentine=False)
    check(not np.allclose(a.positions, b.positions),
          "the two orders really do produce different paths")
    # KNOWN NEGATIVE for the frame check: reverse the positions of every pass
    # and leave the orientation array where it is. That is exactly the mistake
    # the invariant forbids, and worst_frame_difference() has to catch it.
    comb = comb_path()
    pos = comb.positions.copy()
    for s, e in comb.segments:
        pos[s:e] = pos[s:e][::-1]
    mangled = ScanPath(pos, comb.quats.copy(), list(comb.segment_starts),
                       source="mangled")
    worst = worst_frame_difference(frame_by_position(comb), frame_by_position(mangled))
    check(worst > 1.0,
          "orientations left behind by a reversal are detected (%.2f deg)" % (worst,))
    check(worst_frame_difference(frame_by_position(comb),
                                 frame_by_position(comb)) < 1e-12,
          "...while a path compared with itself reports no difference")


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            getattr(fn, "__wrapped__", fn)()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  FAIL %s" % (f,))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(_main())
