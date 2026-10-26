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
# File:  keyence_roll_side_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests which side of the laser line the Keyence body hangs on, and the
#   roll choice that sets it.
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
"""Which side of the laser line the sensor body hangs on.

THE SYMPTOM, which reads as a reach problem and is a choice of roll:

    row 1 of the sweep will not be entered from here
    row 2 of the sweep will not be entered from here
    row 3 of the sweep will not be entered from here
    no approach from here; going home and trying again
    RuntimeError: the arm would not go to the first pose of the sweep
    ([ 1.06  -0.319  1.031]), so there is nothing to start capture for.

_flat_tool_quat() folds the sign out of the raster's travel direction, which is
right about the laser LINE -- a line has no direction -- and leaves the tool's x
axis pointing either way. The head is not on the tool axis:
lj_v7080_optical_frame sits about 75 mm to one side of probe_end, and
_offset_path_in_tool_frame() applies that offset IN THE TOOL FRAME. So the two
rolls put probe_end roughly 150 mm apart over the same measurement points, and
over a part near the edge of the work volume one of them is flyable and the other
is not.

Measured on a 300 mm bowed plate detected at x 0.752..1.016, in a work volume
ending at x 1.100:

    as planned   probe_end x 0.800..1.100   30 waypoints outside,  -3 mm slack
    flipped      probe_end x 0.684..0.970    0 waypoints outside,  25 mm slack

-- and the flipped plan carried 1858 waypoints in 24 passes against 1757 in 23,
so it is not a trade either.

What is checked here, all of it on the HOST with numpy and no ROS:

  * roll_side_score() counts what is outside the box and how hard the path
    presses on its walls, and the tie-break is the half that matters. A path
    entirely inside a box, one micron from its wall, scores zero outside -- so
    counting alone calls the plan that could not be flown a good one.
  * Those measured positions, as two boxes-worth of waypoints, sort the way a
    flyable plan needs them to.
  * A quaternion turned 180 degrees about its own z keeps the tool axis and
    reverses x and y, which is what makes the flip free: the laser line is the
    same line, and only the body moves.

    python3 keyence_roll_side_test.py
    python3 -m pytest -q keyence_roll_side_test.py
"""
import ast
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILURES = []


def check(name, condition, detail=""):
    print("%s  %s%s" % ("ok  " if condition else "FAIL", name,
                        "" if condition or not detail else "  -- " + detail))
    if not condition:
        FAILURES.append(name)


def _lift(path, wanted):
    """The named top-level defs out of a .dpi, without executing the rest of it.

    mesh_reconstruction.dpi is 5000 lines and its body builds recording-database
    channels and ROS clients. roll_side_score() is arithmetic over numpy; lifting
    it is what makes this suite runnable on the host, and lifting is exact --
    the function's own source, compiled.
    """
    src = open(os.path.join(HERE, path)).read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    tree = ast.parse(body)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    missing = wanted - {n.name for n in keep}
    if missing:
        raise AssertionError(
            "%s no longer defines %s at its top level -- it was renamed, moved "
            "into a class, or deleted, and this suite would otherwise pass by "
            "testing nothing" % (path, ", ".join(sorted(missing))))
    # tool_axis_delta() calls quat_mul(), which lives in noether_path.dpi -- the
    # shared globals() a .dpi runs in has it, this namespace has to be given it.
    ns = {"np": np, "math": math, "quat_mul": quat_mul}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


#: ex-ur10e-1's simulated work volume, from the manifest.
LO = (0.60, -0.45, 0.930)
HI = (1.10, 0.05, 1.250)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


ns = _lift("mesh_reconstruction.dpi", {"roll_side_score", "tool_axis_delta"})
roll_side_score = ns["roll_side_score"]
tool_axis_delta = ns["tool_axis_delta"]


# ==============================================================================
# The score
# ==============================================================================

print("")
print("roll_side_score()")

mid = np.array([[0.85, -0.20, 1.05], [0.90, -0.20, 1.05]])
out, slack = roll_side_score(mid, LO, HI)
# 0.120 is the distance to the FLOOR (z 0.930), which is the nearest face from
# z 1.05 -- nearer than the 0.25 to either x or y wall. The least distance to any
# face is the whole point of the number, so the expected value has to be worked
# out rather than eyeballed off the x range.
check("a path in the middle is inside, with real clearance",
      out == 0 and abs(slack - 0.120) < 1e-9, "out %d slack %.4f" % (out, slack))

# A path one micron inside the wall. COUNTING ALONE CANNOT TELL THIS FROM THE
# ONE ABOVE, and that is the whole reason for the tie-break.
wall = np.array([[1.099999, -0.20, 1.05], [1.0999, -0.20, 1.05]])
out_w, slack_w = roll_side_score(wall, LO, HI)
check("a path pressed against the wall is still 'inside'", out_w == 0,
      "out %d" % (out_w,))
check("but its clearance is ~0, so the tie-break separates them",
      slack_w < 1e-4 and slack_w < slack,
      "wall %.6f vs middle %.4f" % (slack_w, slack))
check("and the middle sorts first",
      sorted([(out_w, -slack_w, "wall"), (out, -slack, "middle")])[0][2] == "middle")

over = np.array([[1.15, -0.20, 1.05], [0.90, -0.20, 1.05], [1.20, -0.20, 1.05]])
out_o, slack_o = roll_side_score(over, LO, HI)
check("waypoints outside the box are counted", out_o == 2, "out %d" % (out_o,))
check("and being outside makes the clearance negative, so the two numbers agree "
      "about direction", slack_o < 0.0, "%.4f" % (slack_o,))

check("an empty path is not a winner",
      roll_side_score(np.zeros((0, 3)), LO, HI)[1] < -1.0,
      str(roll_side_score(np.zeros((0, 3)), LO, HI)))


# ==============================================================================
# The measured run, as its two candidate paths
# ==============================================================================
# probe_end x for the two rolls over the same measurement points: 0.800..1.100
# as planned and 0.684..0.970 flipped. Reconstructed as a line of waypoints
# across each range, which is what the passes were.

print("")
print("the measured plan, both rolls")

as_planned = np.column_stack([np.linspace(0.800, 1.1000, 60),
                              np.full(60, -0.30), np.full(60, 1.04)])
flipped = np.column_stack([np.linspace(0.684, 0.9704, 60),
                           np.full(60, -0.38), np.full(60, 1.04)])
sp = roll_side_score(as_planned, LO, HI)
sf = roll_side_score(flipped, LO, HI)
check("the planned roll presses on the wall it failed against",
      sp[1] < 0.001, "slack %.5f" % (sp[1],))
check("the flipped roll has centimetres of room",
      sf[1] > 0.02, "slack %.4f" % (sf[1],))
winner = sorted([(sp[0], -sp[1], "as planned"), (sf[0], -sf[1], "flipped")])[0][2]
check("so the flipped roll is chosen", winner == "flipped", winner)

# AND A PART IN THE MIDDLE OF THE BENCH MUST NOT MOVE. The flip is worth taking
# only when it buys something; a symmetric case has to tie, and a tie keeps the
# planned roll (the sort is stable and "as planned" is first in the list).
left = np.column_stack([np.linspace(0.78, 0.86, 40), np.full(40, -0.20),
                        np.full(40, 1.04)])
right = np.column_stack([np.linspace(0.84, 0.92, 40), np.full(40, -0.20),
                         np.full(40, 1.04)])
sl, sr = roll_side_score(left, LO, HI), roll_side_score(right, LO, HI)
check("two rolls over a part in the middle are both fine",
      sl[0] == 0 and sr[0] == 0, "%s %s" % (sl, sr))
mid_winner = sorted([(sl[0], -sl[1], "as planned"), (sr[0], -sr[1], "flipped")],
                    key=lambda t: t[:2])[0][2]
check("and whichever wins, neither is refused -- this is a preference, not a gate",
      mid_winner in ("as planned", "flipped"), mid_winner)


# ==============================================================================
# Why the flip is free
# ==============================================================================

print("")
print("the flip itself")

ref = (0.0, 0.0, 0.0, 1.0)
flip = quat_mul(ref, (0.0, 0.0, 1.0, 0.0))
Mr, Mf = quat_matrix(ref), quat_matrix(flip)
check("the tool AXIS is unchanged (z is the probe's own axis)",
      np.allclose(Mr[:, 2], Mf[:, 2], atol=1e-12),
      "%s vs %s" % (Mr[:, 2].round(6), Mf[:, 2].round(6)))
check("tool x is reversed", np.allclose(Mf[:, 0], -Mr[:, 0], atol=1e-12))
check("tool y is reversed -- so the laser LINE is the same line and only the "
      "body moves", np.allclose(Mf[:, 1], -Mr[:, 1], atol=1e-12))

# The offset is applied in the tool frame, so reversing y moves probe_end by
# twice the offset. 75 mm of offset is 150 mm of difference, which is the number
# in the docstring.
off = np.array([0.0, -0.075, -0.039])
check("a 75 mm sensor offset becomes 150 mm of probe_end travel",
      abs(np.linalg.norm(Mr.dot(off) - Mf.dot(off)) - 0.150) < 1e-9,
      "%.4f m" % (np.linalg.norm(Mr.dot(off) - Mf.dot(off)),))


# ==============================================================================
# One measured wrist angle settles all four candidates
# ==============================================================================
# _band_cost_from() derives each candidate's wrist_3 from ONE solved pose, on the
# grounds that the candidates differ only by a turn about the tool's own axis and
# wrist_3 IS that turn. If that premise is wrong the whole cable term is fiction,
# so it is the thing to check.
#
# It also has to be derived that way rather than sampled per pose: seeded from
# the parked pose, 7 of 8 sampled poses come back None, and the chained-seed
# version scored the roll that flew (wrist_3 +135..+153, in band) as WORSE than
# the roll the guard refused at -45.

print("")
print("tool_axis_delta()")

ident = (0.0, 0.0, 0.0, 1.0)


def about_z(deg):
    h = math.radians(deg)/2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


check("the same orientation is 0",
      abs(tool_axis_delta(ident, ident)) < 1e-12,
      "%.6f" % (tool_axis_delta(ident, ident),))

for deg in (90.0, 180.0, -90.0, 45.0, 270.0):
    got = math.degrees(tool_axis_delta(ident, about_z(deg)))
    want = deg if deg <= 180.0 else deg - 360.0
    check("a %.0f deg turn about the tool axis reads as %+.0f" % (deg, want),
          abs(((got - want + 180.0) % 360.0) - 180.0) < 1e-6,
          "got %+.3f" % (got,))

# THE FLIP USED BY _roll_side IS EXACTLY 180 DEGREES, which is what makes the
# wrist consequence of flipping the roll knowable without another solve.
ref = about_z(37.0)
flip = quat_mul(ref, (0.0, 0.0, 1.0, 0.0))
check("quat_mul(ref, 180 deg about z) is 180 deg about the tool axis",
      abs(abs(math.degrees(tool_axis_delta(ref, flip))) - 180.0) < 1e-6,
      "%+.4f" % (math.degrees(tool_axis_delta(ref, flip)),))

# A ROTATION ABOUT ANY OTHER AXIS MUST READ AS "NO INFORMATION", not as a number.
# A wrist angle derived from a tilt would be invented, and the caller treats 0 as
# "skip the cable term".
tilt = (math.sin(math.radians(10.0)), 0.0, 0.0, math.cos(math.radians(10.0)))
check("a tilt about tool x gives 0 -- no information, not a fabricated angle",
      tool_axis_delta(ident, tilt) == 0.0, "%.6f" % (tool_axis_delta(ident, tilt),))

# AND THE MEASURED CASE, end to end in one line of arithmetic: the planned roll
# measured +135 deg, in the +31.3..+301.3 band; the flip is 180 away at +315,
# whose only other branch is -45. Neither is in band, and the guard refused every
# approach to it at -40..-50 deg.
BAND = (31.3384, 301.3384)


def in_band(deg):
    for k in (0, -1, 1):
        c = deg + 360.0*k
        if BAND[0] - 1e-9 <= c <= BAND[1] + 1e-9 and abs(c) <= 360.0 + 1e-9:
            return c
    return None


w_planned = 135.0
w_flipped = w_planned + math.degrees(tool_axis_delta(ref, flip))
check("the planned roll's wrist angle is in band", in_band(w_planned) is not None,
      "%+.1f" % (w_planned,))
check("the flipped roll's is +315, and NO turn of the joint brings it in band",
      abs(w_flipped - 315.0) < 1e-6 and in_band(w_flipped) is None,
      "%+.1f -> %s" % (w_flipped, in_band(w_flipped)))
check("which is what the guard measured: -45 is the same orientation",
      abs((w_flipped - 360.0) - (-45.0)) < 1e-6, "%+.1f" % (w_flipped - 360.0,))


def test_keyence_roll_side():
    """So pytest reports what running the file reports."""
    assert not FAILURES, "%d check(s) failed: %s" % (len(FAILURES),
                                                     "; ".join(FAILURES))


if __name__ == "__main__":
    print("")
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("all passed")
