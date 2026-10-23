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
# File:  ec_axis_lock_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-22
# Description:
#   Tests the axis-locked eddy-current toolpath -- one tool orientation for the
#   whole path instead of the measured surface normal -- and that the single
#   pipeline command can ask for it.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-22:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""The axis-locked eddy-current scan: one orientation, chosen rather than found.

WHY THERE IS SUCH A THING. A surface-following path takes the commanded tool
axis from the mesh's normals, and on a coarse reconstruction of a nearly flat
plate those normals wander: the arm rocks along a pass, the axis turn rate is
what TOTG prices the trajectory on, and the probe's tilt is what sets how much
mesh error turns into flaw-like signal. Locking the axis removes all of that in
one move, at the cost of following a real curve. On a plate it should cost
nothing and it is the comparison that says so -- which means the locked path has
to be the SAME path every time it is planned.

THE FOUR PROPERTIES UNDER TEST

  ONE ORIENTATION    every waypoint of every pass carries the identical
                     quaternion. Not "within a degree": identical.
  CHOSEN, NOT FOUND  it is the probe's own vertical reference with the set
                     rotation in it, so it does not depend on what the arm
                     happened to be holding when the plan was made -- which is
                     what plan_mesh_path() falls back to when nobody says, and
                     that fallback is still there for the sweeps that want it.
  ROTATION HELD      the set probe rotation is what the locked path holds, so a
                     locked scan and a surface-following scan are comparable on
                     the one parameter that decides tilt sensitivity.
  NOTHING CROPPED    a vertical axis tilts zero degrees everywhere, so the
                     EC_MAX_TILT_DEG crop that follows keeps the whole path
                     rather than refusing it.

Runs on the host with numpy only: no ROS, no planner, no container, no dgpy.

    python3 ec_axis_lock_test.py
    python3 -m pytest -q ec_axis_lock_test.py
"""
import ast
import functools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")

sys.path.insert(0, HERE)
import ros_msg_stubs

ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "ec_axis_lock_test"}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
ns["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)

scanpath_from_noether = ns["scanpath_from_noether"]
probe_rotation_reference = ns["probe_rotation_reference"]
probe_rotation_of = ns["probe_rotation_of"]
tool_tilt_deg = ns["tool_tilt_deg"]

#: The roll reference ECAutomatedScan.dgp is configured with.
REFERENCE = (1.0, 0.0, 0.0)

FAILS = []


def check(cond, what, detail=""):
    print("%s  %s%s" % ("ok  " if cond else "FAIL", what,
                        "" if cond or not detail else "   %s" % (detail,)))
    if not cond:
        FAILS.append(what)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        before = len(FAILS)
        fn(*args, **kwargs)
        assert len(FAILS) == before, "; ".join(FAILS[before:])
    return wrapper


class _P(object):
    pass


def fake_response(pos, qs, starts):
    res = _P()
    res.noether_pose_array = _P()
    res.noether_pose_array.poses = []
    for p, q in zip(pos, qs):
        pose = _P()
        pose.position = _P()
        pose.orientation = _P()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p]
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = [float(v) for v in q]
        res.noether_pose_array.poses.append(pose)
    res.segment_start_indices = list(starts)
    return res


def pose_with_normal(n, roll):
    """Some orientation with tool z on n and an arbitrary roll, as Noether gives."""
    m2q = ns["matrix_to_quat"]
    n = np.asarray(n, dtype=np.float64)
    n = n/np.linalg.norm(n)
    a = np.array([math.cos(roll), math.sin(roll), 0.37])
    x = a - np.dot(a, n)*n
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(n, np.array([0.0, 0.0, 1.0]))
    x = x/np.linalg.norm(x)
    y = np.cross(n, x)
    return m2q(np.column_stack([x, y, n]))


def cylinder_path(radius_mm=32.0, passes=4, per_pass=25):
    """A serpentine raster over a cylinder: normals that really do wander."""
    r = radius_mm/1000.0
    pos = []
    qs = []
    starts = []
    for p in range(passes):
        starts.append(len(pos))
        ts = np.linspace(-0.9, 0.9, per_pass)
        if p % 2:
            ts = ts[::-1]
        for t in ts:
            ang = t
            n = np.array([math.sin(ang), 0.0, math.cos(ang)])
            pos.append([r*math.sin(ang), 0.02*p, r*math.cos(ang)])
            qs.append(pose_with_normal(n, 0.3 + 0.1*p))
    return np.array(pos), np.array(qs), starts


# -----------------------------------------------------------------------------
# The locked path itself
# -----------------------------------------------------------------------------
@_checked
def test_axis_locked_path_holds_one_orientation():
    pos, qs, starts = cylinder_path()
    res = fake_response(pos, qs, starts)

    for rotation in (0.0, 45.0, -90.0):
        ref = probe_rotation_reference(rotation, REFERENCE)
        path = scanpath_from_noether(res, orient="fixed", fixed_quat=ref)

        spread = np.abs(path.quats - np.asarray(ref)).max()
        check(spread == 0.0,
              "every waypoint carries the identical orientation (rotation %+.0f)"
              % (rotation,), "max deviation %.3g" % (spread,))

        tilt = tool_tilt_deg(path.quats)
        check(tilt.max() < 1e-9,
              "the locked axis is vertical everywhere (rotation %+.0f)" % (rotation,),
              "max tilt %.4g deg" % (tilt.max(),))

        held = probe_rotation_of(path.quats, reference=REFERENCE)
        check(np.abs(held - rotation).max() < 1e-9,
              "the locked path holds the SET probe rotation (%+.0f)" % (rotation,),
              "read %+.4f..%+.4f" % (held.min(), held.max()))


@_checked
def test_the_surface_following_path_really_does_wander():
    """Otherwise the checks above are about a flat test fixture, not a lock."""
    pos, qs, starts = cylinder_path()
    res = fake_response(pos, qs, starts)
    path = scanpath_from_noether(res, roll="locked",
                                 roll_reference=probe_rotation_reference(0.0, REFERENCE))
    tilt = tool_tilt_deg(path.quats)
    check(tilt.max() > 40.0,
          "the same mesh followed by its normals tilts the tool past 40 deg",
          "max tilt %.1f deg" % (tilt.max(),))


@_checked
def test_nothing_is_cropped_by_the_tilt_cap():
    """EC_MAX_TILT_DEG crops; a locked vertical axis is never near it."""
    pos, qs, starts = cylinder_path()
    res = fake_response(pos, qs, starts)
    ref = probe_rotation_reference(0.0, REFERENCE)

    locked = scanpath_from_noether(res, orient="fixed", fixed_quat=ref)
    kept = int((tool_tilt_deg(locked.quats) <= 45.0).sum())
    check(kept == len(locked.quats),
          "the 45 deg crop keeps every waypoint of the locked path",
          "%d of %d" % (kept, len(locked.quats)))

    followed = scanpath_from_noether(res, roll="locked", roll_reference=ref)
    dropped = int((tool_tilt_deg(followed.quats) > 45.0).sum())
    check(dropped > 0,
          "and it does drop waypoints of the surface-following path on this mesh",
          "%d dropped" % (dropped,))


@_checked
def test_a_locked_path_is_not_normal_constrained():
    """So plan_ec_path() skips the normality postcondition rather than failing it."""
    pos, qs, starts = cylinder_path()
    res = fake_response(pos, qs, starts)
    ref = probe_rotation_reference(0.0, REFERENCE)
    locked = scanpath_from_noether(res, orient="fixed", fixed_quat=ref)
    check(locked.normal_constrained is False,
          "a locked path does not claim to be normal-constrained")
    check(locked.roll_mode == "fixed orientation",
          "and its roll mode says the orientation is fixed",
          repr(locked.roll_mode))
    check(locked.max_tilt_deg is None,
          "and it carries no tilt cap, because there is no tilt to cap")


# -----------------------------------------------------------------------------
# The way in: one command, and a reproducible orientation
# -----------------------------------------------------------------------------
def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


@_checked
def test_plan_ec_path_fixes_the_orientation_from_the_probe_reference():
    """Not from live TF: a locked plan must not depend on where the arm was."""
    fn = _func(ast.parse(open(DGP).read()), "plan_ec_path")
    check(fn is not None, "plan_ec_path() is in ECAutomatedScan.dgp")
    if fn is None:
        return
    src = ast.unparse(fn)
    check("ec_probe_rotation_reference" in src
          and "fixed_quat" in src,
          "plan_ec_path() supplies fixed_quat from ec_probe_rotation_reference()")
    check("'fixed'" in src or '"fixed"' in src,
          "and it only does so for orient='fixed'")


@_checked
def test_plan_mesh_path_takes_a_fixed_quat_and_still_falls_back_to_tf():
    tree = ast.parse(open(os.path.join(HERE, "mesh_scan.dpi")).read())
    fn = _func(tree, "plan_mesh_path")
    check(fn is not None, "plan_mesh_path() is in mesh_scan.dpi")
    if fn is None:
        return
    names = [a.arg for a in fn.args.args]
    check("fixed_quat" in names,
          "plan_mesh_path() takes fixed_quat")
    src = ast.unparse(fn)
    check("ros.t" in src,
          "and the live-TF fallback is still there for callers that pass none")


@_checked
def test_the_single_command_can_ask_for_a_locked_axis():
    fn = _func(ast.parse(open(DGP).read()), "run_automated_scan")
    check(fn is not None, "run_automated_scan() is in ECAutomatedScan.dgp")
    if fn is None:
        return
    names = [a.arg for a in fn.args.args]
    check("orient" in names,
          "run_automated_scan() takes orient=")
    src = ast.unparse(fn)
    check("orient" in src.split("plan_ec_path")[0].split("def ")[-1]
          and "plan_ec_path" in src,
          "and hands it to plan_ec_path()")


SECTIONS = (test_axis_locked_path_holds_one_orientation,
            test_the_surface_following_path_really_does_wander,
            test_nothing_is_cropped_by_the_tilt_cap,
            test_a_locked_path_is_not_normal_constrained,
            test_plan_ec_path_fixes_the_orientation_from_the_probe_reference,
            test_plan_mesh_path_takes_a_fixed_quat_and_still_falls_back_to_tf,
            test_the_single_command_can_ask_for_a_locked_axis)

if __name__ == "__main__":
    for section in SECTIONS:
        print("")
        print(section.__name__)
        try:
            section()
        except AssertionError:
            pass
    print("")
    if FAILS:
        print("%d FAILED" % (len(FAILS),))
        sys.exit(1)
    print("all checks passed")
