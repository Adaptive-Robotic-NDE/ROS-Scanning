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
# File:  motion_volume_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the work-volume check and the report that says which face is
#   violated.
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
"""Checking the motion volume, and saying which face is wrong.

TWO VOLUMES, which mean different things:

    work_volume    where a SAMPLE may be placed, and where the eddy-current probe
                   TIP must stay. TOUCHED OFF -- the number is where the arm
                   actually was when the tooling reached each corner.
    motion_volume  where the arm and its TOOLING may navigate. It cannot be
                   touched off: its corners are in free space, and the whole point
                   of it is that the tooling goes where the tip does not.

So the motion volume is DERIVED -- the work volume widened by the tooling's own
reach -- and therefore has to be CHECKED rather than measured. check_motion_volume()
is that check: /compute_ik for reach and /check_state_validity for collision, over
a grid of the box, with nothing moving.

What is checked here, on the HOST with numpy and no ROS:

  * the report names WHICH FACE failed, because "the box is too big" is not
    actionable and "18 points on the y = -0.659 face" is;
  * COLLISION and UNKNOWN fail; REACH DOES NOT. The box is a PERMISSION box, and
    a pose the arm cannot reach fails at the planner whether or not the box
    allowed it. An arm's reach is roughly spherical and a box is not, so its
    corners fall outside first: on a serviceable box, 14 of 27 grid points can
    have no IK solution. Failing on that is crying wolf;
  * UNKNOWN IS NOT OK -- a service that did not answer is not permission to widen
    a box the arm will be allowed to fly through;
  * a clean report says what it does NOT cover, which is anything in the cell that
    is not in the URDF.

The sampling and the service calls are not testable without a cell; the reporting
is, and the reporting is what an operator acts on.

    python3 motion_volume_test.py
    python3 -m pytest -q motion_volume_test.py
"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILURES = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILURES.append(what)


def _lift(path, wanted):
    src = open(os.path.join(HERE, path)).read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    tree = ast.parse(body)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    missing = wanted - {n.name for n in keep}
    if missing:
        raise AssertionError(
            "%s no longer defines %s at its top level -- renamed, moved into a "
            "class, or deleted, and this suite would otherwise test nothing"
            % (path, ", ".join(sorted(missing))))
    ns = {"np": np, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


ns = _lift("work_cell.dpi", {"describe_motion_volume_check",
                            "motion_volume_ok"})
describe = ns["describe_motion_volume_check"]
verdict = ns["motion_volume_ok"]

#: ex-ur10e-1's hardware boxes, from the manifest.
WORK = ((0.5596, -0.5093, 0.9248), (1.1450, 0.4539, 1.2448))
MOVE = ((0.4096, -0.6593, 0.9248), (1.2950, 0.6039, 1.5448))
VOLUME = ("world", MOVE[0], MOVE[1])


def report(points, n=27):
    bad = {"no IK solution": 0, "in collision": 0,
           "validity service did not answer": 0}
    for _p, why in points:
        bad[why] += 1
    return {"volume": VOLUME, "n": n, "points": list(points),
            "unreachable": bad["no IK solution"],
            "colliding": bad["in collision"],
            "unknown": bad["validity service did not answer"],
            "ok": verdict({"unreachable": bad["no IK solution"],
                           "colliding": bad["in collision"],
                           "unknown": bad["validity service did not answer"]})}


print("")
print("a clean check")

lines = describe(report([]))
check(any("27 point(s) checked" in l for l in lines),
      "it says how many points were checked")
check(any("PASS: nothing in the box is in collision" in l for l in lines),
      "a clean box passes on the collision question, which is the one that matters")
check(any("not in the URDF" in l for l in lines),
      "...and says what that does NOT cover -- anything the model does not carry")
check(not any("PULL THOSE FACES IN" in l for l in lines),
      "...and does not tell anyone to change a number that is fine")

# UNREACHABLE CORNERS ARE NORMAL and must not fail the box. A serviceable box
# can have 14 of its 27 grid points without an IK solution, spread over all six
# faces, and still have zero collisions.
corners = [((0.45, -0.2, 0.93), "no IK solution")]*14
rep = report(corners)
check(rep["ok"], "a box whose only failures are unreachable corners still PASSES")
lines = describe(rep)
check(any("That is NOT a fault" in l for l in lines),
      "...and the report says so rather than leaving it looking like one")
check(any("collision-aware" in l for l in lines),
      "...while flagging that /compute_ik may report a blocked pose this way")
check(not any("PULL THOSE FACES IN" in l for l in lines),
      "...and does not send anyone to edit the manifest over it")


print("")
print("which face is wrong")

# Every failure on the y = -0.6593 face, so the report has exactly one face to
# name.
face = [((0.5, -0.6593, 1.0), "in collision"),
        ((0.8, -0.6593, 1.0), "in collision"),
        ((1.2, -0.6593, 1.2), "in collision")]
lines = describe(report(face))
check(any("3 point(s) in collision" in l for l in lines),
      "it counts the colliding points")
check(any("-0.659 face" in l for l in lines),
      "...and NAMES the face they are on, which is the number to change")
check(any("PULL THOSE FACES IN" in l for l in lines),
      "...and says to pull it in")
check(any("CONTAIN the work volume" in l for l in lines),
      "...while saying the box must still contain the work volume")
check(any("the WORK volume is the thing that is too big" in l for l in lines),
      "...and what it means if it cannot be pulled in that far")


print("")
print("the three failure kinds are reported separately")

mixed = [((0.41, 0.0, 1.0), "no IK solution"),
         ((1.29, 0.0, 1.0), "in collision"),
         ((0.8, 0.6039, 1.5448), "validity service did not answer")]
lines = describe(report(mixed))
for why in ("no IK solution", "in collision",
            "validity service did not answer"):
    check(any(why in l for l in lines), "it reports %r on its own line" % (why,))
rep = report(mixed)
check(rep["unreachable"] == 1 and rep["colliding"] == 1 and rep["unknown"] == 1,
      "and the three are counted separately, not lumped together")


print("")
print("unknown is not ok")

# A service that did not answer is NOT permission. This is the same rule
# CheckStateValidity() states for itself: None is not True.
# THE REAL RULE, lifted from work_cell.dpi -- not this file's own arithmetic.
check(verdict({"unreachable": 0, "colliding": 0, "unknown": 1}) is False,
      "motion_volume_ok(): one unanswered service is NOT ok")
check(verdict({"unreachable": 0, "colliding": 0, "unknown": 0}) is True,
      "...and a clean report is")
check(verdict({"unreachable": 0, "colliding": 1, "unknown": 0}) is False,
      "...a collision is not")
check(verdict({"unreachable": 99, "colliding": 0, "unknown": 0}) is True,
      "...and REACH is not a failure: the box says where the arm may TRY, and "
      "an unreachable pose fails at the planner anyway")
check(verdict(None) is False and verdict({}) is False,
      "...and no report at all is not a pass")

only_unknown = report([((0.8, 0.0, 1.2), "validity service did not answer")])
check(not only_unknown["ok"],
      "a box whose only fault is an unanswered service is NOT ok")
lines = describe(only_unknown)
check(any("PULL THOSE FACES IN" in l for l in lines),
      "...and it says to act rather than passing quietly")


print("")
print("the two boxes, as the manifest declares them")

check(all(m <= w for m, w in zip(MOVE[0], WORK[0]))
      and all(m >= w for m, w in zip(MOVE[1], WORK[1])),
      "the motion volume contains the work volume")
pad = min([WORK[0][i] - MOVE[0][i] for i in (0, 1)]
          + [MOVE[1][i] - WORK[1][i] for i in (0, 1)])
check(abs(pad - 0.150) < 1e-6,
      "it clears it by 150 mm in x-y -- the tooling's 75 mm reach plus the "
      "Keyence's 73 mm offset, which is what the geometry asks for")
check(abs(MOVE[0][2] - WORK[0][2]) < 1e-9,
      "and they share a floor: the tooling has no business below the surface a "
      "part rests on either")


def test_motion_volume():
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
