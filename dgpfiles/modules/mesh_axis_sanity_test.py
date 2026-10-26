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
# File:  mesh_axis_sanity_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the tool-axis backstop in mesh_scan.dpi and its wrist-branch
#   diagnostic.
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
"""mesh_scan.dpi's tool-axis backstop, and the wrist-branch diagnostic.

WHAT THIS IS ABOUT. The eddy-current toolpath takes its orientation from the
measured mesh's normals, one per waypoint. A Keyence mesh whose normals point up
99.6% of the time still yields a raw path with up to 90 deg of tilt from
vertical, 178 deg steps between two ADJACENT waypoints of one pass, and most
passes varying by more than a hundred degrees along their own length -- on a
surface that is essentially flat. With the roll locked, none of that is roll.
The arm obeys a toolpath like that: it stops mid-sweep, with the probe on the
part, and rotates.

Two mechanisms are not enough on their own:

  * EC_MAX_TILT_DEG DROPS waypoints -- a few percent of them on a mesh like that
    -- which pays coverage for a measurement error and only acts after the fact.
  * EC_AXIS_SMOOTHING_M averages the tool axis over 6 mm and still lets tens of
    degrees through, because a MEAN of unit vectors has no defence against an
    outlier.

sanitize_tool_axis() makes the axis robust first (a rolling MEDIAN, which an
outlier cannot move), cuts a pass where the surface genuinely turns inside one
waypoint spacing, and bounds the turn rate in what is left. Nothing is dropped
and no position is touched.

THE DIVISION OF LABOUR WITH THE MESH REPAIR. The governing rule is that the
probe's tilt with respect to the SURFACE NORMAL must be zero, so where the mesh
is repairable the repair belongs upstream: repair_mesh_normals() in
noether_path.dpi fits a local surface to the mesh's POSITIONS and never reads a
normal at all, and scanpath_from_noether(orient="normal") makes the tool axis
that surface's normal. mesh_normal_repair_test.py is that path's suite.

So this function keeps two jobs and defers one, selected by the `assert_only`
flag that plan_mesh_path() passes when the axis is already constrained:

  * it is the REPAIR for a path whose mesh was NOT repaired -- a mesh handed in
    by hand, MESH_REPAIR_NORMALS=False, orient="capped" for a non-contact
    sensor. That is the default, and every check below exercises it.
  * it is the place a pass is CUT at a real discontinuity, which is a motion
    decision rather than a repair and is right either way.
  * with assert_only=True it MEASURES the first and third stages instead of
    acting: clamping an axis that is already the surface normal would tilt the
    probe off that normal. test_assert_only_measures_without_acting() pins that;
    the rest of the division is tested in mesh_normal_repair_test.py.

The checks below are in three groups:

  * SYNTHETIC, always run. One defect at a time on a written axis field, so each
    property can be shown to hold for its own reason: an isolated spike is
    repaired, a wide real step is NOT (it is geometry, and flattening it would
    put the probe edge-on into a wall), the turn-rate bound is met EXACTLY
    afterwards rather than approximately, no waypoint is ever lost, and the
    whole thing is a no-op on a path that is already clean.
  * THE REPORTING, always run. A single warning line can mislead in three
    separate ways -- counting pass boundaries as mid-pass rotation, measuring a
    pass against its first pose, and reporting the raw path rather than the one
    that flies. tool_axis_continuity() must do none of the three, and each is
    pinned here.
  * THE WRIST BRANCH, always run. wrist3_branch_in_band() must identify a
    -189.9 deg refusal as an IK branch choice (+170.1 is in band) and must NEVER
    be mistaken for permission to substitute it.

Plus the REAL meshes when they are present -- they are captured scan data of
gigabytes, not repository data, so their absence SKIPS loudly rather than
failing.

Runs on the host with numpy only: no ROS, no planner, no container, no dgpy. A
backstop that needs the cell to test is a backstop nobody runs.

    python3 mesh_axis_sanity_test.py       # prints every check, exit 1 on failure
    python3 -m pytest -q mesh_axis_sanity_test.py
"""
import ast
import functools
import math
import os
import sys
import types

import numpy as np

import test_data

HERE = os.path.dirname(os.path.abspath(__file__))

#: The two captured runs the real-mesh sections grade against. They live in the
#: fixture repository, not here, so when it is absent those sections SKIP.
REAL = {
    # The hard case: the bench and a real 28.4 mm step are inside this mesh, so
    # steep geometry that must NOT be flattened sits next to the noise.
    "run1": ("keyence", "curved_a",
             "specimen_keyence_ball_pivot_smoothed_filtered.ply"),
    # The easy case: the part top and nothing else, 0.5 mm of relief, so every
    # bad normal on it is an artefact.
    "run2": ("keyence", "curved_b",
             "specimen_keyence_ball_pivot_smoothed_filtered.ply"),
}

fails = []
skips = []


def check(name, ok, detail=""):
    print("  %-66s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so a standalone run reports every failure
    in one pass; under pytest that would be a test that passes while printing
    FAIL. main() calls fn.__wrapped__ to keep the standalone behaviour. Same
    device as progress_test.py.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# ------------------------------------------------------------------------------
# Loading the modules under test
# ------------------------------------------------------------------------------

def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve on a host with no ROS. Assigning sys.modules["geometry_msgs.msg"]
# directly would overwrite whatever another suite in this directory installed;
# ensure_msgs() only ADDS. This suite builds no message -- every axis field below
# is numpy.
ros_msg_stubs.ensure_msgs()

NS = {"__name__": "mesh_axis_sanity_test", "os": os, "np": np, "math": math}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), NS)
# noether_path.dpi reports through progress.dpi, which a config includes ahead of
# it. Silenced: these checks are about geometry and progress.dpi has its own test.
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), NS)
NS["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), NS)
# mesh_scan.dpi is a scan config's worth of globals, but a .dpi is a function
# body: exec'ing it only DEFINES things. PATH_MIN_SEGMENT_M is the one config
# constant the functions under test read.
NS["PATH_MIN_SEGMENT_M"] = 0.005
exec(compile(_load("mesh_scan.dpi"), "mesh_scan.dpi", "exec"), NS)

ScanPath = NS["ScanPath"]
sanitize_tool_axis = NS["sanitize_tool_axis"]
tool_axis_continuity = NS["tool_axis_continuity"]
describe_tool_axis_continuity = NS["describe_tool_axis_continuity"]
direction_spread = NS["_direction_spread_deg"]
rolling_median = NS["_rolling_median_dirs"]
locked_quats_from_axes = NS["locked_quats_from_axes"]
tool_tilt_deg = NS["tool_tilt_deg"]
tool_twist_deg = NS["tool_twist_deg"]
smooth_tool_axis = NS["smooth_tool_axis"]
read_ply_mesh = NS["read_ply_mesh"]

CNS = {"__name__": "mesh_axis_sanity_test_cable"}
exec(compile(_load("cable_envelope.dpi"), "cable_envelope.dpi", "exec"), CNS)
wrist3_branch_in_band = CNS["wrist3_branch_in_band"]
check_wrist3 = CNS["check_wrist3"]
CableRefused = CNS["CableRefused"]
cable_envelope_from_rows = CNS["cable_envelope_from_rows"]


# ------------------------------------------------------------------------------
# Building an axis field to break
# ------------------------------------------------------------------------------

#: A measured waypoint spacing along a Noether pass: median step 0.4465 mm.
#: Everything below is built at that spacing so the median window derived from
#: it is the one a real path gets (14 waypoints at 6 mm).
STEP_M = 0.0004465


def axes_of(quats):
    """Tool z for each (x,y,z,w). The same extraction the functions do."""
    q = np.asarray(quats, float).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    q = q/n
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2.0*(x*z + y*w), 2.0*(y*z - x*w), 1.0 - 2.0*(x*x + y*y)],
                    axis=1)


def tilted(deg, about=(1.0, 0.0, 0.0)):
    """A unit axis `deg` from vertical, leaning about a horizontal axis."""
    a = np.asarray(about, float)
    a = a/np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    perp = np.cross(a, z)
    return math.cos(math.radians(deg))*z + math.sin(math.radians(deg))*perp


def make_path(axes, step_m=STEP_M, starts=None):
    """A LOCKED ScanPath running along +x with the given tool axes."""
    axes = np.asarray(axes, float).reshape(-1, 3)
    axes = axes/np.linalg.norm(axes, axis=1, keepdims=True)
    n = len(axes)
    pos = np.column_stack([np.arange(n)*step_m,
                           np.zeros(n), np.full(n, 0.94)])
    p = ScanPath(pos, locked_quats_from_axes(axes, (1.0, 0.0, 0.0)),
                 [0] if starts is None else list(starts), source="synthetic")
    p.roll_mode = "locked"
    p.roll_reference = (1.0, 0.0, 0.0)
    p.noether_flipped = 0
    p.worst_roll_step_deg = NS["_worst_orientation_step"](p.quats)
    return p


def worst_within_pass(path):
    """Largest orientation step between consecutive waypoints of ONE pass."""
    z = axes_of(path.quats)
    worst = 0.0
    for a, b in path.segments:
        if b - a < 2:
            continue
        c = np.clip(np.einsum("ij,ij->i", z[a:b-1], z[a+1:b]), -1.0, 1.0)
        worst = max(worst, float(np.degrees(np.arccos(c)).max()))
    return worst


# ==============================================================================
# SYNTHETIC: one defect at a time
# ==============================================================================

@_checked
def test_isolated_spikes_are_repaired():
    """A normal that disagrees with its neighbours is noise, and goes.

    The shape a real mesh produces: four consecutive waypoints carrying
    near-horizontal axes, ALTERNATING IN SIGN, between neighbours tilted a few
    degrees. A mean cannot survive that; a median does not notice it.
    """
    print("\n--- an isolated run of bad normals is repaired, not dropped -------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    for i, s in zip((28, 29, 30, 31), (-1.0, +1.0, -1.0, -1.0)):
        z[i] = np.array([s*0.9919, s*0.1245, 0.0232])
    p = make_path(z)
    before = worst_within_pass(p)
    check("the input really is that bad (%.1f deg between two waypoints)"
          % (before,), before > 170.0, "%.2f" % before)
    s = sanitize_tool_axis(p, verbose=False)
    after = worst_within_pass(s)
    info = s.axis_sanity
    check("the four bad axes are the four repaired (%d)" % (info["repaired"],),
          info["repaired"] == 4, str(info["repaired"]))
    check("the worst within-pass step falls below 5 deg (%.2f)" % (after,),
          after < 5.0, "%.2f" % after)
    check("tilt comes back to the surface's own (%.2f deg, was %.1f)"
          % (tool_tilt_deg(s.quats).max(), tool_tilt_deg(p.quats).max()),
          tool_tilt_deg(s.quats).max() < 2.0,
          "%.2f" % tool_tilt_deg(s.quats).max())
    check("NOT ONE WAYPOINT DROPPED (%d in, %d out)" % (len(p.positions),
                                                        len(s.positions)),
          len(s.positions) == len(p.positions))
    check("positions are untouched to the bit",
          np.array_equal(np.asarray(s.positions), np.asarray(p.positions)))
    check("the roll is still locked (twist %.3f deg)"
          % (np.abs(tool_twist_deg(s.quats)).max(),),
          np.abs(tool_twist_deg(s.quats)).max() < 1e-6)


@_checked
def test_a_wide_real_step_is_left_alone():
    """A wide steep region is geometry, and flattening it is worse than churn.

    A run WIDER than half the median window is the surface really turning. The
    median must not outvote it -- an axis forced vertical over a vertical wall
    puts the probe edge-on into the wall while the position still says it is on
    the surface, which is the failure the tilt cap exists for.
    """
    print("\n--- a wide, consistent steep region survives the repair ----------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (120, 1)).astype(float)
    wall = tilted(80.0)
    z[50:90] = wall                    # 40 waypoints, ~18 mm of real wall
    p = make_path(z)
    s = sanitize_tool_axis(p, verbose=False)
    zs = axes_of(s.quats)
    middle = float(np.degrees(np.arccos(np.clip(np.dot(zs[70], wall), -1, 1))))
    check("the middle of the wall still leans 80 deg (%.1f deg from it)"
          % (middle,), middle < 1.0, "%.2f" % middle)
    check("the flat either side is still flat (%.2f deg)"
          % (float(np.degrees(np.arccos(np.clip(zs[10][2], -1, 1)))),),
          float(np.degrees(np.arccos(np.clip(zs[10][2], -1, 1)))) < 1.0)
    check("the two lips are CUT rather than rotated through (%d cut)"
          % (s.axis_sanity["split"],), s.axis_sanity["split"] == 2,
          str(s.axis_sanity["split"]))
    check("cutting cost no waypoints (%d -> %d)"
          % (len(p.positions), len(s.positions)),
          len(s.positions) == len(p.positions))
    check("and it is the pass count that grew (%d -> %d)"
          % (s.axis_sanity["passes_before"], s.axis_sanity["passes_after"]),
          s.axis_sanity["passes_after"] == s.axis_sanity["passes_before"] + 2)
    check("no step WITHIN any resulting pass exceeds the split threshold "
          "(%.2f deg)" % (worst_within_pass(s),),
          worst_within_pass(s) <= s.axis_sanity["split_deg"] + 1e-6,
          "%.2f" % worst_within_pass(s))


@_checked
def test_the_turn_rate_bound_is_met_exactly():
    """Not approximately, and not after iterating: exactly, after two sweeps.

    An iterated pairwise relaxation does not terminate across a discontinuity
    the bound cannot express: it leaves over a hundred degrees outstanding after
    a dozen sweeps while distorting the good axes either side. Cutting first and
    then sweeping forward-then-backward settles every pair by construction.
    """
    print("\n--- the turn-rate bound holds exactly, with nothing left over ----")
    rng = np.random.RandomState(7)
    # A noisy ramp: real curvature plus grit, which is what a reconstruction is.
    lean = np.linspace(0.0, 35.0, 200) + rng.normal(0.0, 4.0, 200)
    z = np.array([tilted(float(v)) for v in lean])
    p = make_path(z)
    s = sanitize_tool_axis(p, max_turn_deg_per_mm=10.0, min_step_deg=1.0,
                           verbose=False)
    check("nothing is left over the bound (residual %.3g deg)"
          % (s.axis_sanity["residual_over_bound_deg"],),
          s.axis_sanity["residual_over_bound_deg"] <= 1e-9,
          "%.3g" % s.axis_sanity["residual_over_bound_deg"])
    zs = axes_of(s.quats)
    pos = np.asarray(s.positions)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)*1e3
    ang = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", zs[:-1], zs[1:]), -1.0, 1.0)))
    bound = np.maximum(1.0, 10.0*d)
    check("and every measured step is inside it (worst overshoot %.3g deg)"
          % (float((ang - bound).max()),), float((ang - bound).max()) <= 1e-9)
    tight = sanitize_tool_axis(p, max_turn_deg_per_mm=3.0, min_step_deg=1.0,
                              verbose=False)
    zt = axes_of(tight.quats)
    at = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", zt[:-1], zt[1:]), -1.0, 1.0)))
    check("3 deg/mm gives a smaller worst step than 10 (%.2f vs %.2f)"
          % (at.max(), ang.max()), at.max() <= ang.max() + 1e-9)
    check("the ramp is still a ramp -- the ends are %.1f deg apart (asked 35)"
          % (float(np.degrees(np.arccos(np.clip(np.dot(zs[0], zs[-1]), -1, 1)))),),
          25.0 < float(np.degrees(np.arccos(np.clip(np.dot(zs[0], zs[-1]),
                                                    -1, 1)))) < 45.0)
    # ONE SWEEP IS ENOUGH: a second, backward sweep changes not one axis,
    # because the forward sweep leaves nothing over the bound for it to act on.
    # If that stops being true, the "one sweep, provably" claim is wrong.
    z0 = axes_of(p.quats)
    dev = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", zs, z0), -1.0, 1.0)))
    zb = zs.copy()
    lim_rad = np.radians(bound)
    moved = 0
    for i in range(len(zb) - 2, -1, -1):
        c = float(np.clip(np.dot(zb[i+1], zb[i]), -1.0, 1.0))
        th = math.acos(c)
        if th > lim_rad[i] + 1e-12:
            moved += 1
    check("a backward sweep would find nothing to do (%d pairs over the bound)"
          % (moved,), moved == 0, str(moved))
    # THE TOTAL ORIENTATION TRAVEL MUST NOT GROW. Individual pairs CAN come out
    # further apart -- moving axis i toward i-1 moves it away from i+1, and the
    # sweep then clamps that pair in turn -- so pairwise contraction is NOT the
    # invariant. What must hold is what the arm pays for: the sum of the
    # orientation changes along the pass, which is the wrist travel.
    def travel(a):
        return float(np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", a[:-1], a[1:]), -1.0, 1.0))).sum())
    raw_travel = travel(axes_of(sanitize_tool_axis(
        p, max_turn_deg_per_mm=0.0, verbose=False).quats))
    check("the total orientation travel along the pass falls (%.0f deg -> %.0f; "
          "worst accumulated correction %.2f deg, which a one-directional clamp "
          "is allowed to accumulate past one step's %.2f deg bound)"
          % (raw_travel, travel(zs), dev.max(), bound.max()),
          travel(zs) <= raw_travel + 1e-6,
          "%.2f -> %.2f" % (raw_travel, travel(zs)))


@_checked
def test_a_clean_path_is_untouched():
    """The backstop must cost nothing when there is nothing wrong.

    A gentle real curve, sampled well: no repair, no cut, and an axis field that
    comes back within a whisker of what went in. A filter that quietly reshapes
    good data is worse than no filter, because its output looks fine.
    """
    print("\n--- a clean path comes back unchanged ----------------------------")
    z = np.array([tilted(float(v)) for v in np.linspace(0.0, 6.0, 300)])
    p = make_path(z)
    s = sanitize_tool_axis(p, verbose=False)
    info = s.axis_sanity
    check("nothing repaired (%d)" % (info["repaired"],), info["repaired"] == 0)
    check("nothing cut (%d)" % (info["split"],), info["split"] == 0)
    check("nothing clamped (%d)" % (info["clamped"],), info["clamped"] == 0)
    moved = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", axes_of(s.quats), axes_of(p.quats)), -1.0, 1.0)))
    check("the axis moved at most %.4f deg anywhere" % (moved.max(),),
          moved.max() < 1e-6, "%.4g" % moved.max())
    check("the pass structure is identical (%d -> %d)"
          % (len(p.segments), len(s.segments)),
          len(s.segments) == len(p.segments))


@_checked
def test_it_declines_a_roll_it_would_destroy():
    """Every stage rebuilds the orientation from the tool axis alone, which
    re-locks the roll. On a path whose roll is deliberately NOT locked that is
    silent damage, so it must refuse instead -- the same way plan_ec_path()
    declines to average one."""
    print("\n--- it refuses to touch a path whose roll is not locked ----------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (40, 1)).astype(float)
    z[20] = tilted(70.0)
    for mode in ("fixed", "path", "planner", None):
        p = make_path(z)
        p.roll_mode = mode
        s = sanitize_tool_axis(p, verbose=False)
        check("roll_mode %r: handed straight back" % (mode,), s is p)
    p = make_path(z)
    check("and roll_mode 'locked' IS processed",
          sanitize_tool_axis(p, verbose=False) is not p)


@_checked
def test_two_point_slivers_are_handled():
    """A pass of exactly two waypoints must still be checked.

    Slivers are where the worst single step tends to live, and a stage that
    skips anything under three points misses all of them.
    """
    print("\n--- a two-waypoint pass is not skipped ---------------------------")
    z = np.array([[0.0, 0.0, 1.0], tilted(120.0),
                  [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    p = make_path(z, starts=[0, 2])          # a 2-point pass, then a 3-point one
    check("the sliver really is a pass of two (%s)" % (p.segments[:1],),
          p.segments[0] == (0, 2))
    s = sanitize_tool_axis(p, verbose=False)
    check("it is either cut or bounded, not left at 120 deg (%.1f)"
          % (worst_within_pass(s),),
          worst_within_pass(s) <= s.axis_sanity["split_deg"] + 1e-6,
          "%.2f" % worst_within_pass(s))
    check("no waypoint lost (%d -> %d)" % (len(p.positions), len(s.positions)),
          len(s.positions) == len(p.positions))


@_checked
def test_the_median_is_what_makes_it_robust():
    """Why the window is a median and not a mean.

    Same window, same data: a box MEAN of the axes is dragged tens of degrees by
    a minority of garbage and the MEDIAN is not. If this stops being true, the
    design argument in mesh_scan.dpi is wrong.
    """
    print("\n--- median against mean, on the same window ----------------------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (15, 1)).astype(float)
    z[7:11] = tilted(88.0)                   # 4 of 15, a minority
    med = rolling_median(z, 15)
    mean = z.mean(axis=0)
    mean = mean/np.linalg.norm(mean)
    med_err = float(np.degrees(np.arccos(np.clip(med[7][2], -1.0, 1.0))))
    mean_err = float(np.degrees(np.arccos(np.clip(mean[2], -1.0, 1.0))))
    check("the median stays with the majority (%.2f deg from vertical)"
          % (med_err,), med_err < 5.0, "%.2f" % med_err)
    check("the mean is dragged away (%.1f deg from vertical)" % (mean_err,),
          mean_err > 15.0, "%.2f" % mean_err)
    # And the majority has to be a majority: 8 of 15 wins.
    z2 = np.tile(np.array([0.0, 0.0, 1.0]), (15, 1)).astype(float)
    z2[4:12] = tilted(88.0)
    med2 = rolling_median(z2, 15)
    kept = float(np.degrees(np.arccos(np.clip(med2[7][2], -1.0, 1.0))))
    check("8 of 15 steep and the median goes WITH them (%.1f deg)" % (kept,),
          kept > 45.0, "%.2f" % kept)


# ==============================================================================
# THE REPORTING: three ways a single warning line misleads
# ==============================================================================

@_checked
def test_boundary_steps_are_not_reported_as_mid_pass_rotation():
    """A pass boundary is a lift, a link move and an approach, so the
    orientation change across it is expected and costs the probe nothing.
    Measuring the largest step over the WHOLE array counts it as mid-pass
    rotation; the report must book it to the boundary instead."""
    print("\n--- a pass boundary is not a step between waypoints --------------")
    z = np.vstack([np.tile([0.0, 0.0, 1.0], (30, 1)),
                   np.tile(tilted(60.0), (30, 1))])
    p = make_path(z, starts=[0, 30])
    rep = tool_axis_continuity(p, warn_deg=5.0)
    check("the whole-array metric still sees 60 deg (%.1f)"
          % (p.worst_roll_step_deg,), p.worst_roll_step_deg > 55.0)
    check("but WITHIN a pass it reports %.2f deg"
          % (rep["whole"]["step_max_deg"],),
          rep["whole"]["step_max_deg"] < 1e-6,
          "%.3g" % rep["whole"]["step_max_deg"])
    check("and the 60 deg is booked to the boundary (%.1f over %d boundary)"
          % (rep["boundary_max_deg"], rep["boundaries"]),
          rep["boundaries"] == 1 and rep["boundary_max_deg"] > 55.0)
    check("no pass is reported as varying (%d)"
          % (rep["whole"]["passes_varying"],),
          rep["whole"]["passes_varying"] == 0)


@_checked
def test_one_bad_pose_does_not_condemn_a_pass():
    """Measuring a pass's variation as the angle from its FIRST pose reports the
    whole pass as churning when only pose 0 is bad, and under-reports an outlier
    in the middle. The spread (the diameter of the set of directions) is the
    honest measure and is never smaller than an anchored distance."""
    print("\n--- pass variation is a spread, not a distance from pose 0 -------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    z[0] = tilted(70.0)
    anchored = max(float(np.degrees(np.arccos(np.clip(
        np.dot(z[0], z[i]), -1.0, 1.0)))) for i in range(60))
    spread = direction_spread(z)
    check("anchoring at pose 0 reads %.1f deg" % (anchored,), anchored > 65.0)
    check("the spread reads the same %.1f deg here, because the outlier IS the "
          "spread" % (spread,), spread > 65.0)
    # ... and the case that separates them: the outlier is in the middle, which
    # anchoring UNDER-reports for every pass whose first pose is good.
    z2 = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    z2[30] = tilted(70.0)
    anchored2 = max(float(np.degrees(np.arccos(np.clip(
        np.dot(z2[0], z2[i]), -1.0, 1.0)))) for i in range(60))
    check("a diameter is never smaller than an anchored distance (%.1f >= %.1f)"
          % (direction_spread(z2), anchored2),
          direction_spread(z2) >= anchored2 - 1e-9)
    check("a monotone ramp measures exactly its own extent (%.2f deg of 30)"
          % (direction_spread(np.array([tilted(float(v))
                                        for v in np.linspace(0, 30, 50)])),),
          abs(direction_spread(np.array([tilted(float(v))
                                         for v in np.linspace(0, 30, 50)]))
              - 30.0) < 0.1)


@_checked
def test_the_report_says_what_survives_the_tilt_cap():
    """The number that matters is the one for the path that FLIES. Measured on
    the raw path, before EC_MAX_TILT_DEG drops waypoints, the figure is largely
    orientation that is about to be thrown away. Both views have to be
    reported: the raw one and the one for the survivors."""
    print("\n--- the report covers the path after the cap, not just before ----")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (100, 1)).astype(float)
    z[40:60] = tilted(85.0)              # steep enough that the cap drops it
    p = make_path(z)
    rep = tool_axis_continuity(p, max_tilt_deg=45.0, warn_deg=5.0)
    check("the raw path is reported as varying (%.1f deg)"
          % (rep["whole"]["spread_max_deg"],),
          rep["whole"]["spread_max_deg"] > 80.0)
    check("the capped view exists and counts the drops (%d)"
          % (rep.get("dropped_by_cap"),), rep.get("dropped_by_cap") == 20)
    check("and reports the survivors as steady (%.2f deg)"
          % (rep["capped"]["spread_max_deg"],),
          rep["capped"]["spread_max_deg"] < 1e-6,
          "%.3g" % rep["capped"]["spread_max_deg"])
    lines = describe_tool_axis_continuity(rep)
    check("describe() prints both views and the boundaries (%d lines)"
          % (len(lines),),
          any("EC_MAX_TILT_DEG" in l for l in lines)
          and any("boundar" in l for l in lines))
    check("and a turn RATE, which is what distinguishes 34 deg over 7 mm from "
          "34 deg standing still",
          any("deg/mm" in l for l in lines))


@_checked
def test_the_rate_tells_a_curve_from_a_rotation():
    """The absolute step cannot: Noether's spacing inside one pass spans four
    orders of magnitude, so 30 deg over a 7 mm step and 30 deg over a 9 micron
    step are the same number and opposite situations."""
    print("\n--- the turn rate separates a curve from a standing rotation -----")
    z = np.array([tilted(0.0), tilted(30.0)])
    slow = make_path(z, step_m=0.007)
    fast = make_path(z, step_m=0.000009)
    # min_length_m=0 so the 9 um pass counts as a pass here: the default filter
    # is PATH_MIN_SEGMENT_M, which correctly calls it a link move.
    rs = tool_axis_continuity(slow, warn_deg=1e9, min_length_m=0.0)["whole"]
    rf = tool_axis_continuity(fast, warn_deg=1e9, min_length_m=0.0)["whole"]
    check("the absolute step is the same for both (%.1f, %.1f)"
          % (rs["step_max_deg"], rf["step_max_deg"]),
          abs(rs["step_max_deg"] - rf["step_max_deg"]) < 1e-6)
    check("the rate is not: %.1f deg/mm over 7 mm against %.0f over 9 um"
          % (rs["rate_max_deg_per_mm"], rf["rate_max_deg_per_mm"]),
          rf["rate_max_deg_per_mm"] > 100.0*rs["rate_max_deg_per_mm"])


# ==============================================================================
# THE WRIST BRANCH
# ==============================================================================

#: A cell's measured cable envelope. The untilted band is 16.1..301.1 deg,
#: which is the one a parking refusal quotes. The tilted rows are here because
#: the model interpolates between samples and cannot answer for a tilt outside
#: every row -- a one-row model refuses everything, which is not the case under
#: test.
CELL_ROWS = [
    {"tilt": "no tilt",        "lo_deg": 16.1,  "hi_deg": 301.1},
    {"tilt": "tilt x +20 deg", "lo_deg": 38.1,  "hi_deg": 323.1},
    {"tilt": "tilt x +40 deg", "lo_deg": 72.1,  "hi_deg": 312.1},
    {"tilt": "tilt x -20 deg", "lo_deg": 127.7, "hi_deg": 277.7},
    {"tilt": "tilt x -40 deg", "lo_deg": 179.2, "hi_deg": 269.2},
    {"tilt": "tilt y +20 deg", "lo_deg": 78.4,  "hi_deg": 303.4},
    {"tilt": "tilt y -20 deg", "lo_deg": 60.0,  "hi_deg": 285.0},
    {"tilt": "tilt y -40 deg", "lo_deg": 90.9,  "hi_deg": 300.9},
]


@_checked
def test_the_minus_190_refusal_is_an_ik_branch_choice():
    """A refusal at -189.9 deg is a branch choice, not an unreachable pose.

    -189.9 + 360 = +170.1, which is inside 16.2..301.1, so the orientation is
    reachable and the solver picked the wrong IK branch. The guard is right to
    refuse; the fault is upstream of it.
    """
    print("\n--- -189.9 deg is diagnosed as a branch choice -------------------")
    band = (math.radians(16.2), math.radians(301.1))
    alt = wrist3_branch_in_band(math.radians(-189.9), band)
    check("an in-band branch is found", alt is not None)
    check("and it is +170.1 deg (%.1f)" % (math.degrees(alt),),
          abs(math.degrees(alt) - 170.1) < 0.05, "%.3f" % math.degrees(alt))
    check("which is where a preceding raster leaves the arm (+159..+177 deg)",
          159.0 <= math.degrees(alt) <= 177.0)
    # The other kind of failure: an angle with NO in-band branch at all, which
    # is a pose the cables cannot hold whichever way the joint turns.
    none_case = wrist3_branch_in_band(math.radians(5.0),
                                      (math.radians(179.2), math.radians(269.2)))
    check("a tight band with no branch available answers None (%s)"
          % (none_case,), none_case is None)
    # wrist_3 has +/-360 deg of travel (tari_ur_description/config/
    # joint_limits.yaml), so a branch outside that is not a branch the arm has.
    # A band of 350..400 deg contains 5 + 360 = 365, and 365 is past the limit.
    beyond = wrist3_branch_in_band(math.radians(5.0),
                                   (math.radians(350.0), math.radians(400.0)))
    check("a branch past the joint's own +/-360 deg of travel is NOT offered "
          "(%s)" % (beyond,), beyond is None)


@_checked
def test_the_diagnostic_never_becomes_permission():
    """The binding rule: the ABSOLUTE joint angle governs and there is no
    +/-360 substitution, because the cable physically wraps. check_wrist3() must
    therefore still REFUSE -189.9 deg -- and must say, in the refusal, that the
    fault is a branch choice rather than a cable limit."""
    print("\n--- the diagnostic does not soften the refusal -------------------")
    model = cable_envelope_from_rows(CELL_ROWS, step_deg=15.0)
    try:
        check_wrist3([math.radians(-189.9)], "the park rise", model=model)
        check("-189.9 deg is still REFUSED", False, "it was allowed")
        return
    except CableRefused as e:
        msg = str(e)
    check("-189.9 deg is still REFUSED", True)
    check("the refusal still states the absolute-angle rule",
          "ABSOLUTE joint angle" in msg)
    check("it names the in-band branch (+170.1)", "+170.1" in msg)
    check("it says the bug is upstream", "UPSTREAM" in msg)
    check("it forbids substituting the branch here",
          "do not substitute" in msg.lower())
    check("and it names the measured cause (the IK mode, not the band)",
          "mode: global" in msg and "mtc_hub_node" in msg)
    # A pose with NO in-band branch must get the OTHER diagnosis. 310 deg is one:
    # the band is 16.1..301.1, 310 - 360 = -50 is outside it too, and 310 + 360
    # is past the joint's travel. That is the toolpath asking for a pose the
    # cables cannot hold whichever way the wrist turns.
    check("310 deg really has no in-band branch",
          wrist3_branch_in_band(math.radians(310.0),
                                (math.radians(16.1), math.radians(301.1)))
          is None)
    try:
        check_wrist3([math.radians(310.0)], "something", model=model)
        other = ""
    except CableRefused as e:
        other = str(e)
    check("a pose no branch can reach is blamed on the TOOLPATH",
          "TOOLPATH" in other, other[-90:])
    check("an in-band angle is still allowed",
          check_wrist3([math.radians(170.1)], "fine", model=model) is not None)


# ==============================================================================
# THE REAL MESHES, when they are present
# ==============================================================================

def _emulated(mesh, line_spacing=0.0003, rotation_offset=math.pi/2):
    """The orientations principal_axis_noether would return over `mesh`.

    The planner is a ROS service and these checks must run on the host, so the
    ONE thing that matters here is reproduced instead: a Noether pose carries
    tool z along the mesh's interpolated vertex normal at the waypoint, and every
    orientation number in the log is a function of those normals and of which
    pass each waypoint belongs to. So the mesh is sliced with the planner's own
    cutting planes (noether_cut_axes(), which transcribes its two lines of C++),
    the crossings are chained into polylines, and the vertex normals are sampled
    along them.

    Validated against the real planner's own output over the same mesh, at the
    same 0.3 mm spacing and the same pi/2 rotation offset (planner / this):

        waypoints            67672  /  68000
        passes                 648  /  672
        poses flipped           15  /  18
        worst step          168.16  /  178.13 deg
        tilt max / mean    89.7/3.4 /  89.7/3.5 deg
        passes varying >5      429  /  423
        dropped by the cap    1804  /  2009

    Close enough to argue with, which is all it is for.
    """
    v, vn, t = read_ply_mesh(mesh)
    if vn is None:
        raise RuntimeError("this needs the mesh's own vertex normals")
    cut = NS["noether_cut_axes"](v, rotation_offset=rotation_offset,
                                 line_spacing=line_spacing)
    n = cut["plane_normal"]
    proj = v @ n
    lo, hi = proj.min(), proj.max()
    offsets = lo + (np.arange(cut["planes"]) + 0.5)*(hi - lo)/cut["planes"]
    tri = np.asarray(t, np.int64)
    s = v @ n
    polys = []
    for d in offsets:
        sv = s - d
        a = sv[tri]
        nneg = (~(a > 0.0)).sum(axis=1)
        segs = []
        for f in np.flatnonzero((nneg > 0) & (nneg < 3)):
            pts, nms = [], []
            for i, j in ((tri[f, 0], tri[f, 1]), (tri[f, 1], tri[f, 2]),
                         (tri[f, 2], tri[f, 0])):
                si, sj = sv[i], sv[j]
                if (si > 0.0) == (sj > 0.0):
                    continue
                w = si/(si - sj)
                pts.append(v[i] + w*(v[j] - v[i]))
                nms.append(vn[i] + w*(vn[j] - vn[i]))
            if len(pts) == 2:
                segs.append((pts[0], pts[1], nms[0], nms[1]))
        polys.extend(_chain(segs))
    polys = [p for p in polys if len(p[0]) >= 2]
    pos = np.vstack([p[0] for p in polys])
    nms = np.vstack([p[1] for p in polys])
    nms = nms/np.linalg.norm(nms, axis=1, keepdims=True)
    # orient_tool_away_from_surface() corrects the SIGN; do the same thing to the
    # axes directly, which for a locked roll is the identical operation.
    flip = nms[:, 2] < 0.0
    nms[flip] = -nms[flip]
    starts, k = [], 0
    for p in polys:
        starts.append(k)
        k += len(p[0])
    path = ScanPath(pos, locked_quats_from_axes(nms, (1.0, 0.0, 0.0)), starts,
                    source="emulated noether (%d poses)" % len(pos))
    path.roll_mode = "locked"
    path.roll_reference = (1.0, 0.0, 0.0)
    path.noether_flipped = int(flip.sum())
    path.worst_roll_step_deg = NS["_worst_orientation_step"](path.quats)
    return path


def _chain(segs, tol=1e-7):
    """Chain plane-crossing segments into polylines by shared endpoints. A break
    is a hole or a boundary, which is the fragmentation the planner reports."""
    def key(p):
        return (round(float(p[0])/tol), round(float(p[1])/tol),
                round(float(p[2])/tol))
    nodes = {}
    for idx, (p, q, _a, _b) in enumerate(segs):
        nodes.setdefault(key(p), []).append((idx, 0))
        nodes.setdefault(key(q), []).append((idx, 1))
    used = [False]*len(segs)
    out = []
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        p, q, np_, nq = segs[start]
        cp, cn = [p, q], [np_, nq]
        for end in (1, 0):
            while True:
                tip = cp[-1] if end else cp[0]
                nxt = next(((i, s) for i, s in nodes.get(key(tip), ())
                            if not used[i]), None)
                if nxt is None:
                    break
                i, side = nxt
                used[i] = True
                far = segs[i][1] if side == 0 else segs[i][0]
                farn = segs[i][3] if side == 0 else segs[i][2]
                if end:
                    cp.append(far)
                    cn.append(farn)
                else:
                    cp.insert(0, far)
                    cn.insert(0, farn)
        out.append((np.asarray(cp, float), np.asarray(cn, float)))
    return out


def _as_it_flies(path, cap=45.0, smooth_m=0.006):
    """The path plan_ec_path() would hand the arm: tilt cap, then the 6 mm
    average. Positions and the region crop are irrelevant to orientation."""
    tilt = tool_tilt_deg(path.quats)
    keep = tilt <= cap
    pos = np.asarray(path.positions)
    q = np.asarray(path.quats)
    starts, P, Q, k = [], [], [], 0
    for a, b in path.segments:
        m = keep[a:b]
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                if j - i >= 2:
                    starts.append(k)
                    P.append(pos[a+i:a+j])
                    Q.append(q[a+i:a+j])
                    k += j - i
                i = j
            else:
                i += 1
    out = ScanPath(np.vstack(P), np.vstack(Q), starts, source=path.source)
    out = smooth_tool_axis(out, smooth_m, reference=(1.0, 0.0, 0.0))
    out.roll_mode = "locked"
    out.roll_reference = (1.0, 0.0, 0.0)
    return out


def _real(which):
    return test_data.path(*REAL[which])


@_checked
def test_real_run2_mesh_is_fully_cleaned():
    """The easy mesh is the part top and nothing else, so every bad normal on it
    is an isolated artefact. The repair must remove ALL of them, and must leave
    the tens of thousands that were already right exactly where they were."""
    print("\n--- REAL run 2 mesh (the clean one) ------------------------------")
    mesh = _real("run2")
    if mesh is None:
        skips.append("run 2's mesh: %s" % (test_data.why_missing(*REAL["run2"]),))
        print("  SKIP  %s" % (skips[-1],))
        return
    p = _emulated(mesh)
    b = tool_axis_continuity(p, max_tilt_deg=45.0)
    s = sanitize_tool_axis(p, verbose=False)
    a = tool_axis_continuity(s, max_tilt_deg=45.0)
    print("      before: tilt max %.1f, within-pass step %.2f, %d/%d passes vary"
          % (b["tilt_max_deg"], b["whole"]["step_max_deg"],
             b["whole"]["passes_varying"], b["whole"]["passes"]))
    print("      after : tilt max %.1f, within-pass step %.2f, %d/%d passes vary"
          % (a["tilt_max_deg"], a["whole"]["step_max_deg"],
             a["whole"]["passes_varying"], a["whole"]["passes"]))
    check("before, 5 passes churn and one step is 83.5 deg (%d, %.1f)"
          % (b["whole"]["passes_varying"], b["whole"]["step_max_deg"]),
          b["whole"]["passes_varying"] >= 4
          and b["whole"]["step_max_deg"] > 80.0)
    check("after, NO pass varies by more than 5 deg (%d)"
          % (a["whole"]["passes_varying"],), a["whole"]["passes_varying"] == 0)
    check("after, the worst within-pass step is under 3 deg (%.2f, was %.1f)"
          % (a["whole"]["step_max_deg"], b["whole"]["step_max_deg"]),
          a["whole"]["step_max_deg"] < 3.0)
    check("after, tilt max is under 3 deg (%.1f, was %.1f)"
          % (a["tilt_max_deg"], b["tilt_max_deg"]), a["tilt_max_deg"] < 3.0)
    check("nothing is left for EC_MAX_TILT_DEG to drop (%d, was %d)"
          % (a.get("dropped_by_cap"), b.get("dropped_by_cap")),
          a.get("dropped_by_cap") == 0 and b.get("dropped_by_cap") > 0)
    check("no waypoint dropped (%d)" % (len(s.positions),),
          len(s.positions) == len(p.positions))
    fb, fa = _as_it_flies(p), _as_it_flies(s)
    print("      as it flies: the 6 mm average moved the axis up to %.1f deg, "
          "now %.1f" % (fb.axis_smoothing["moved_max_deg"],
                        fa.axis_smoothing["moved_max_deg"]))
    check("the 6 mm average now has almost nothing to do (%.1f deg, was %.1f)"
          % (fa.axis_smoothing["moved_max_deg"],
             fb.axis_smoothing["moved_max_deg"]),
          fa.axis_smoothing["moved_max_deg"] < 3.0
          and fb.axis_smoothing["moved_max_deg"] > 15.0)
    # THE KEYENCE BODY reaches 178 mm from probe_end (the clearance gate's own
    # figure), and with the tip on the surface a tilt of theta displaces a point
    # at radius R by at most 2*R*sin(theta/2). Reaching the -122.7 mm clearance
    # that fails the gate takes 40.3 deg of tilt.
    t = tool_tilt_deg(fa.quats)
    swing = 2.0*178.0*np.sin(np.radians(t)/2.0)
    print("      tilt max %.1f deg -> the Keyence body can swing at most %.1f mm"
          % (t.max(), swing.max()))
    check("the Keyence body cannot swing the 122.7 mm the gate refuses at "
          "(bound %.1f mm)" % (swing.max(),), swing.max() < 20.0,
          "%.1f" % swing.max())


@_checked
def test_real_run1_mesh_improves_and_says_what_is_left():
    """The hard mesh carries the bench and a real 28.4 mm step. The repair must
    take out the noise -- and must NOT pretend to fix the step, which is real
    measured geometry and is the ROI crop's problem, not this function's."""
    print("\n--- REAL run 1 mesh (bench and a 28.4 mm step in it) -------------")
    mesh = _real("run1")
    if mesh is None:
        skips.append("run 1's mesh: %s" % (test_data.why_missing(*REAL["run1"]),))
        print("  SKIP  %s" % (skips[-1],))
        return
    p = _emulated(mesh)
    b = tool_axis_continuity(p, max_tilt_deg=45.0)
    s = sanitize_tool_axis(p, verbose=False)
    a = tool_axis_continuity(s, max_tilt_deg=45.0)
    print("      before: within-pass step %.1f (p99 %.1f), %d/%d passes vary up "
          "to %.1f deg" % (b["whole"]["step_max_deg"], b["whole"]["step_p99_deg"],
                           b["whole"]["passes_varying"], b["whole"]["passes"],
                           b["whole"]["spread_max_deg"]))
    print("      after : within-pass step %.1f (p99 %.1f), %d/%d passes vary up "
          "to %.1f deg" % (a["whole"]["step_max_deg"], a["whole"]["step_p99_deg"],
                           a["whole"]["passes_varying"], a["whole"]["passes"],
                           a["whole"]["spread_max_deg"]))
    check("the emulated raw path reproduces the planner's 178 deg step (%.1f)"
          % (b["whole"]["step_max_deg"],), b["whole"]["step_max_deg"] > 170.0)
    check("and its 89.7 deg of tilt (%.1f)" % (b["tilt_max_deg"],),
          88.0 < b["tilt_max_deg"] < 91.0)
    check("the worst within-pass step falls below 30 deg (%.1f, was %.1f)"
          % (a["whole"]["step_max_deg"], b["whole"]["step_max_deg"]),
          a["whole"]["step_max_deg"] < 30.0)
    check("the p99 within-pass step falls by more than half (%.2f, was %.2f)"
          % (a["whole"]["step_p99_deg"], b["whole"]["step_p99_deg"]),
          a["whole"]["step_p99_deg"] < 0.5*b["whole"]["step_p99_deg"])
    check("the cap has less to drop (%d, was %d)"
          % (a.get("dropped_by_cap"), b.get("dropped_by_cap")),
          a.get("dropped_by_cap") < b.get("dropped_by_cap"))
    check("no waypoint dropped to achieve it (%d)" % (len(s.positions),),
          len(s.positions) == len(p.positions))
    fb, fa = _as_it_flies(p), _as_it_flies(s)
    print("      as it flies: the 6 mm average moved the axis up to %.1f deg, "
          "now %.1f; %d/%d passes vary, was %d/%d"
          % (fb.axis_smoothing["moved_max_deg"],
             fa.axis_smoothing["moved_max_deg"],
             tool_axis_continuity(fa)["whole"]["passes_varying"],
             tool_axis_continuity(fa)["whole"]["passes"],
             tool_axis_continuity(fb)["whole"]["passes_varying"],
             tool_axis_continuity(fb)["whole"]["passes"]))
    check("the 6 mm average no longer has a discontinuity to average across "
          "(%.1f deg, was %.1f)"
          % (fa.axis_smoothing["moved_max_deg"],
             fb.axis_smoothing["moved_max_deg"]),
          fa.axis_smoothing["moved_max_deg"]
          < 0.6*fb.axis_smoothing["moved_max_deg"])
    check("fewer passes churn than before (%d vs %d)"
          % (tool_axis_continuity(fa)["whole"]["passes_varying"],
             tool_axis_continuity(fb)["whole"]["passes_varying"]),
          tool_axis_continuity(fa)["whole"]["passes_varying"]
          < tool_axis_continuity(fb)["whole"]["passes_varying"])
    # AND THE HONEST PART: the 28.4 mm step is still there, so the tool still
    # tilts far enough to swing the Keyence body into the part. Asserted so that
    # nobody reads the improvement above as "the clearance fault is fixed".
    t = tool_tilt_deg(fa.quats)
    swing = 2.0*178.0*np.sin(np.radians(t)/2.0)
    print("      tilt max %.1f deg -> the Keyence body can still swing %.1f mm; "
          "%d poses past the 40.3 deg that -122.7 mm needs"
          % (t.max(), swing.max(), int((t > 40.3).sum())))
    check("the steep geometry is NOT flattened away (tilt still %.1f deg)"
          % (t.max(),), t.max() > 40.0)
    check("so keyence_metal CAN still go negative on this mesh -- the ROI crop "
          "is the fix, not this (%d poses past 40.3 deg)"
          % (int((t > 40.3).sum()),), int((t > 40.3).sum()) > 0)


@_checked
def test_real_meshes_cost_little_to_sanitise():
    """A planning stage is allowed to be slow; it is not allowed to be slow for
    nothing. The clamp's cost is proportional to the damage, so both real meshes
    must go through in about the time one numpy pass over them takes."""
    print("\n--- what it costs on the real meshes -----------------------------")
    import time
    for which in ("run1", "run2"):
        mesh = _real(which)
        if mesh is None:
            skips.append("%s's mesh is not present" % (which,))
            print("  SKIP  %s" % (skips[-1],))
            continue
        p = _emulated(mesh)
        t0 = time.time()
        s = sanitize_tool_axis(p, verbose=False)
        dt = time.time() - t0
        print("      %s: %d waypoints in %.2f s (%d repaired, %d cut, %d clamped)"
              % (which, len(p.positions), dt, s.axis_sanity["repaired"],
                 s.axis_sanity["split"], s.axis_sanity["clamped"]))
        check("%s sanitises in under 5 s (%.2f)" % (which, dt), dt < 5.0,
              "%.2f" % dt)
        check("%s leaves nothing over the turn-rate bound (%.3g deg)"
              % (which, s.axis_sanity["residual_over_bound_deg"]),
              s.axis_sanity["residual_over_bound_deg"] <= 1e-9)


# ==============================================================================

# ==============================================================================
# THE DIVISION OF LABOUR WITH THE MESH REPAIR
# ==============================================================================

@_checked
def test_assert_only_measures_without_acting():
    """With the mesh repaired upstream, this MEASURES stages 1 and 3.

    Same input, same thresholds, two modes. The default REPAIRS (and every other
    section here exercises that). assert_only=True reports the identical counts
    and changes not one axis, because on a path whose axis IS the repaired
    surface normal a clamp is a tilt AWAY from that normal, which the zero-tilt
    rule forbids. The cut at a real discontinuity still happens in both, because
    that is a motion decision and not a repair.
    """
    print("\n--- assert_only measures stages 1 and 3, and still cuts ----------")
    z = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    for i, sg in zip((28, 29, 30, 31), (-1.0, +1.0, -1.0, -1.0)):
        z[i] = np.array([sg*0.9919, sg*0.1245, 0.0232])
    p = make_path(z)
    acted = sanitize_tool_axis(p, verbose=False)
    told = sanitize_tool_axis(p, verbose=False, assert_only=True)
    a, t = acted.axis_sanity, told.axis_sanity
    check("both modes find the same %d disagreement(s)" % (t["repaired"],),
          t["repaired"] == a["repaired"] == 4,
          "%d vs %d" % (t["repaired"], a["repaired"]))
    check("assert mode CUTS at the %d step(s) it can see and the default cuts %d, "
          "because stage 1 has already taken those steps out before stage 2 looks: "
          "the cut follows the axis field as it really is"
          % (t["split"], a["split"]),
          t["split"] > 0 and t["split"] >= a["split"])
    check("the flag is on the report (%s / %s)"
          % (t["assert_only"], a["assert_only"]),
          t["assert_only"] is True and a["assert_only"] is False)
    moved_told = np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", axes_of(told.quats), axes_of(p.quats)), -1.0, 1.0)))
    moved_acted = np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", axes_of(acted.quats), axes_of(p.quats)), -1.0, 1.0)))
    check("assert_only changes NO axis (worst %.2e deg, the locked-quaternion "
          "round trip) while the default changes one by %.1f deg"
          % (moved_told.max(), moved_acted.max()),
          moved_told.max() < 1e-4 and moved_acted.max() > 80.0)
    check("on this input the cuts absorb every discontinuity, so neither mode "
          "has anything left to clamp (%d/%d)" % (t["clamped"], a["clamped"]),
          t["clamped"] == a["clamped"] == 0)

    # STAGE 3 ON ITS OWN. A smooth ramp that turns faster than the bound but
    # never steps past split_deg: nothing to repair, nothing to cut, and the
    # clamp is the only stage with an opinion.
    ramp = make_path([tilted(15.0*i) for i in range(6)])
    ra = sanitize_tool_axis(ramp, verbose=False).axis_sanity
    rt = sanitize_tool_axis(ramp, verbose=False, assert_only=True).axis_sanity
    check("a 15 deg-per-0.45 mm ramp is over the 10 deg/mm bound: the default "
          "clamps %d step(s) and settles to %.2e deg over"
          % (ra["clamped"], ra["residual_over_bound_deg"]),
          ra["clamped"] > 0 and ra["residual_over_bound_deg"] < 1e-6
          and ra["split"] == 0)
    check("assert_only counts the same %d step(s) and REPORTS %.1f deg of excess "
          "instead of removing it" % (rt["clamped"], rt["over_turn_bound_deg"]),
          rt["clamped"] == ra["clamped"] and rt["over_turn_bound_deg"] > 1.0
          and rt["residual_over_bound_deg"] == 0.0,
          "residual_over_bound_deg is the invariant on a clamp that RAN; in "
          "assert mode nothing ran, so the excess is reported separately")
    ramp_moved = np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", axes_of(sanitize_tool_axis(
            ramp, verbose=False, assert_only=True).quats),
        axes_of(ramp.quats)), -1.0, 1.0)))
    check("and it leaves the ramp's axes exactly where the surface put them "
          "(worst %.2e deg)" % (ramp_moved.max(),), ramp_moved.max() < 1e-4)
    check("neither mode loses a waypoint (%d, %d of %d)"
          % (len(told), len(acted), len(p)),
          len(told) == len(acted) == len(p))
    # And the same two modes on a path that needs nothing: both are no-ops.
    clean = make_path(np.tile(tilted(12.0), (40, 1)))
    c1 = sanitize_tool_axis(clean, verbose=False).axis_sanity
    c2 = sanitize_tool_axis(clean, verbose=False, assert_only=True).axis_sanity
    check("on a clean path both modes report nothing (%d/%d repaired, %d/%d "
          "clamped)" % (c1["repaired"], c2["repaired"], c1["clamped"],
                        c2["clamped"]),
          c1["repaired"] == c2["repaired"] == 0
          and c1["clamped"] == c2["clamped"] == 0)


@_checked
def test_not_blind():
    """The subject ran, on a known-positive and a known-negative.

    Also checks that nothing here reads a geometry_msgs class: ensure_msgs() may
    hand this process a stub module another suite wrote, so "nothing reads it"
    is a claim worth checking rather than asserting in a comment.
    """
    print("\n--- the test is not blind -----------------------------------------")
    check("the subject came out of mesh_scan.dpi, not a stand-in",
          sanitize_tool_axis.__code__.co_filename == "mesh_scan.dpi"
          and tool_axis_continuity.__code__.co_filename == "mesh_scan.dpi")
    check("this suite builds no ROS message, so no other suite's stub can "
          "decide what it measures",
          not any(isinstance(v, type)
                  and getattr(v, "__module__", "").startswith("geometry_msgs")
                  for v in list(globals().values())))
    # KNOWN-NEGATIVE: one run of four flipped axes, which must be repaired.
    bad = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    for i, s_ in zip((28, 29, 30, 31), (-1.0, +1.0, -1.0, -1.0)):
        bad[i] = np.array([s_*0.9919, s_*0.1245, 0.0232])
    n_bad = sanitize_tool_axis(make_path(bad), verbose=False).axis_sanity["repaired"]
    # KNOWN-POSITIVE: the same field with nothing wrong, which must be untouched.
    good = np.tile(np.array([0.0, 0.0, 1.0]), (60, 1)).astype(float)
    n_good = sanitize_tool_axis(make_path(good), verbose=False).axis_sanity["repaired"]
    check("a clean axis field repairs %d and one with four flipped axes repairs "
          "%d -- the repair is responding to its input" % (n_good, n_bad),
          n_good == 0 and n_bad == 4)


def main():
    """Every section, in order, reporting all failures rather than the first."""
    del fails[:]
    del skips[:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            # __wrapped__ is the undecorated section: it records failures instead
            # of raising, so one run reports all of them.
            getattr(fn, "__wrapped__", fn)()
    print("")
    for s in skips:
        print("SKIPPED: %s" % (s,))
    if fails:
        print("%d FAILED: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("all mesh_scan.dpi tool-axis checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
