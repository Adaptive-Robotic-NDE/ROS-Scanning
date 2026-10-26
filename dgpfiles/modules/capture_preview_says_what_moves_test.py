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
# File:  capture_preview_says_what_moves_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-06
# Description:
#   Tests that the confirm preview describes what the ARM does, not only
#   where the tool tip ends up.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-06:  Initial Commit
#
#########################################################################
"""The preview has to describe the arm, not just the tip.

A capture pass at a different probe rotation moves the tip a few centimetres and
swings wrist 3 through ninety degrees. Described by its endpoint alone it reads

    contact rot +90, +0 deg: over contact_b: straight to the target
      1 waypoint(s), 335 mm straight-line, near the part: 10.0 mm/s
      probe_end -> (0.8061, 0.0156, 0.9582)

and the operator is asked to approve a straight line. On 2026-10-06 a
`rotations_deg=(90.0,)` that was meant to be -90 -- the sign lost while
recalling the line from history -- took wrist 3 from the taught +173 to +263 and
was stopped with the emergency stop. The angle was inside the measured cable
band so nothing refused it, and nothing in front of the operator said a number.

capture_move() already solves IK along the path, to bound the scaling and to
check the cable band, and then discarded it. These tests pin that the preview
now reports it. Runs on the host with no ROS.
"""
import functools
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    src = open(os.path.join(HERE, "cell_capture_motion.dpi")).read()
    m = re.search(r"^def _cap_what_the_arm_does\(.*?(?=^global |^def )",
                  src, re.S | re.M)
    assert m, "_cap_what_the_arm_does() moved in cell_capture_motion.dpi"
    ns = {"np": np, "math": math}
    exec(compile(m.group(0), "cell_capture_motion.dpi", "exec"), ns)
    return ns


def _joints(w3_deg, others=(0.1, -1.2, 1.4, -0.3, 1.0)):
    return list(others) + [math.radians(w3_deg)]


fails = []


def check(name, ok, detail=""):
    print("  %-62s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


@_checked
def test_it_names_the_wrist_3_angle_at_both_ends():
    """The number that would have caught 2026-10-06."""
    print("=== it names the wrist 3 angle at both ends ===")
    ns = _load()
    out = ns["_cap_what_the_arm_does"]([_joints(173), _joints(218), _joints(263)])
    check("the start angle is there", "+173" in out, out)
    check("the end angle is there", "+263" in out, out)
    check("and it is labelled as wrist 3", "wrist 3" in out, out)
    print()


@_checked
def test_a_pure_rotation_is_not_described_as_a_short_move():
    """The regression. A 90 degree wrist swing has to be visible in the text."""
    print("=== a pure rotation is not described as a short move ===")
    ns = _load()
    out = ns["_cap_what_the_arm_does"]([_joints(173), _joints(263)])
    check("the largest joint excursion is reported", "largest joint" in out, out)
    check("and it is the 90 degrees the wrist turned", "90 deg" in out, out)
    print()


@_checked
def test_it_reports_how_far_the_tool_itself_turns():
    """Wrist 3 is the arm's answer; the TOOL turning is the operator's."""
    print("=== it reports how far the tool itself turns ===")
    ns = _load()
    half = math.radians(45.0)
    q0 = (0.0, 0.0, 0.0, 1.0)
    q1 = (0.0, 0.0, math.sin(half), math.cos(half))        # 90 deg about z
    out = ns["_cap_what_the_arm_does"]([_joints(173), _joints(263)],
                                       quats=[q0, q1])
    check("it says the tool turns", "TOOL turns" in out, out)
    check("by 90 degrees", "90 deg" in out, out)

    # A pure translation must NOT claim a turn: noise in the quaternion would
    # otherwise put a spurious number in every prompt.
    out = ns["_cap_what_the_arm_does"]([_joints(173), _joints(173)],
                                       quats=[q0, q0])
    check("a move that does not turn says nothing about turning",
          "TOOL turns" not in out, out)
    print()


@_checked
def test_it_says_how_much_cable_band_is_left():
    print("=== it says how much cable band is left ===")
    ns = _load()
    ns["wrist3_band_deg"] = lambda quat=None: (16.1, 301.3)
    out = ns["_cap_what_the_arm_does"]([_joints(173), _joints(263)],
                                       quats=[(0, 0, 0, 1)]*2)
    check("the remaining room is reported", "from the cable limit" in out, out)
    # 301.3 - 263 = 38
    check("and it is the distance to the NEAR edge", "38 deg" in out, out)
    print()


@_checked
def test_it_never_breaks_the_prompt():
    """This runs inside the confirm path. An exception here would stop a move
    that was otherwise fine, which is a worse failure than a missing line."""
    print("=== it never breaks the prompt ===")
    ns = _load()
    f = ns["_cap_what_the_arm_does"]
    for bad in (None, [], [None], [[1, 2]], "nonsense", [_joints(10), None]):
        try:
            got = f(bad)
            check("%r is handled" % (bad,), isinstance(got, str))
        except Exception as e:                                # noqa: BLE001
            check("%r is handled" % (bad,), False, repr(e))
    # A band function that throws must not take the prompt with it.
    ns["wrist3_band_deg"] = lambda quat=None: (_ for _ in ()).throw(
        RuntimeError("no envelope"))
    got = f([_joints(173), _joints(263)], quats=[(0, 0, 0, 1)]*2)
    check("a failing cable band still leaves the rest of the line",
          "wrist 3" in got and "cable limit" not in got, got)
    print()


_SECTIONS = ["test_it_names_the_wrist_3_angle_at_both_ends",
             "test_a_pure_rotation_is_not_described_as_a_short_move",
             "test_it_reports_how_far_the_tool_itself_turns",
             "test_it_says_how_much_cable_band_is_left",
             "test_it_never_breaks_the_prompt"]


def _main():
    for name in _SECTIONS:
        fn = globals()[name]
        getattr(fn, "__wrapped__", fn)()
    if fails:
        print("FAILURES:", fails)
        return 1
    print("all preview checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
