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
# File:  motion_kind_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the move-kind overlay in scan_plan.dpi against a stubbed motion
#   module.
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
"""Check the move-kind overlay in scan_plan.dpi against a stubbed motion module.

The overlay picks the IK position_threshold from the shape of the move. What
has to hold, and what breaks if it does not:

  * a ONE-pose CartesianMove is a step and takes the tight threshold. Under the
    loose one every scan step lands up to 0.1 mm short and the c-scan's columns
    drift: a commanded 0.2 mm step moves 0.106 mm.
  * a MANY-pose CartesianMove is a row and takes the loose one. Under the tight
    one the row truncates part way and the sweep has a hole in it: a 69-pose row
    completes 100% at a 1e-4 threshold and only 21-62% at 1e-5.
  * whatever the overlay changed is put back, including when the move raises.
    A profile left in the wrong state makes the NEXT stage fail instead.
  * a cell whose motion_profiles.yaml has no move_kinds at all still flies. A
    safety knob that refuses to run is not a safety knob.

Runs on the host: `ros` is a stub that records what it was asked.

    python3 motion_kind_test.py
"""
import functools
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

SPEC = {
    "group": "ur_manipulator",
    "nodes": ["/move_group", "/mtc_hub_node", "/dgpython_action_server"],
    "profiles": {
        "scan": {"parameters": {"mode": "local",
                                "minimal_displacement_weight": 0.0,
                                "position_threshold": 1.0e-5},
                 "nodes": {"/mtc_hub_node":
                           {"minimal_displacement_weight": 0.01}}},
    },
    "move_kinds": {
        "point_step": {"parameters": {"position_threshold": 1.0e-5}},
        "cartesian_path": {"parameters": {"position_threshold": 1.0e-4}},
    },
}


class Ros(object):
    """Just enough of dgpython_rosmotion to answer the overlay."""

    def __init__(self, spec=SPEC, fail=False):
        self.spec = spec
        self.held = {node: {"position_threshold": 1.0e-5,
                            "minimal_displacement_weight": 0.0,
                            "mode": "local"}
                     for node in spec["nodes"]}
        self.applied = []
        self.moves = []
        self.fail = fail

    def MotionProfiles(self):
        return self.spec

    def SetMotionParameters(self, values, timeout=5.0):
        previous = {}
        for node, per_node in values.items():
            previous[node] = {k: self.held[node][k] for k in per_node}
            self.held[node].update(per_node)
        self.applied.append(values)
        return previous

    def CartesianMove(self, poses, **kwargs):
        # The threshold IN FORCE at the moment the move is made is the only
        # thing that matters, so record it here rather than trusting the calls.
        self.moves.append(
            {"n": len(getattr(poses, "poses", poses)),
             "threshold": self.held["/mtc_hub_node"]["position_threshold"],
             "weight": self.held["/mtc_hub_node"]["minimal_displacement_weight"]})
        if self.fail:
            raise RuntimeError("the move failed")
        return "moved"


def load(ros, extra=None):
    src = open(os.path.join(HERE, "scan_plan.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "motion_kind_test", "ros": ros, "PROJECTION_MODE": "off",
          "projector": None, "trace_accum": None}
    ns.update(extra or {})
    exec(compile(src, "scan_plan.dpi", "exec"), ns)
    return ns


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok    %s" % (name,))
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILURES.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


class Poses(object):
    def __init__(self, n):
        self.poses = list(range(n))


@_checked
def test_the_kind_is_taken_from_the_move():
    print("the kind is taken from the move, not from the caller")
    ros = Ros()
    ns = load(ros)
    ns["cartesian_move"](Poses(1))
    check("one pose is a step: IK stops at 0.01 mm",
          ros.moves[-1]["threshold"] == 1.0e-5, ros.moves[-1])
    ns["cartesian_move"](Poses(69))
    check("69 poses is a row: IK stops at 0.1 mm",
          ros.moves[-1]["threshold"] == 1.0e-4, ros.moves[-1])
    ns["cartesian_move"](Poses(2))
    check("two poses is a row as well -- it is a polyline, not a landing",
          ros.moves[-1]["threshold"] == 1.0e-4, ros.moves[-1])
    ns["cartesian_move"](Poses(69), kind="point_step")
    check("a caller who names a kind gets it",
          ros.moves[-1]["threshold"] == 1.0e-5, ros.moves[-1])
    ns["cartesian_move"]([0, 1, 2, 3])
    check("a plain list of poses works too (the reconstruction sweep passes one)",
          ros.moves[-1]["n"] == 4 and ros.moves[-1]["threshold"] == 1.0e-4,
          ros.moves[-1])


@_checked
def test_the_overlay_is_put_back():
    print("the overlay is put back")
    ros = Ros()
    ns = load(ros)
    ns["cartesian_move"](Poses(69))
    check("after a row, the node holds what it held before",
          ros.held["/mtc_hub_node"]["position_threshold"] == 1.0e-5,
          ros.held["/mtc_hub_node"])
    check("and nothing else was touched",
          ros.held["/mtc_hub_node"]["minimal_displacement_weight"] == 0.0,
          ros.held["/mtc_hub_node"])

    ros = Ros(fail=True)
    ns = load(ros)
    try:
        ns["cartesian_move"](Poses(69))
    except RuntimeError:
        pass
    check("and it is put back even when the move raises",
          ros.held["/mtc_hub_node"]["position_threshold"] == 1.0e-5,
          ros.held["/mtc_hub_node"])


@_checked
def test_a_cell_without_move_kinds_still_flies():
    print("a cell without move_kinds still flies")
    spec = {k: v for k, v in SPEC.items() if k != "move_kinds"}
    ros = Ros(spec=spec)
    ns = load(ros)
    ns["cartesian_move"](Poses(69))
    check("the move happens, with whatever the profile holds",
          len(ros.moves) == 1 and ros.moves[-1]["threshold"] == 1.0e-5,
          ros.moves)
    check("and nothing was applied, so nothing is left behind",
          ros.applied == [], ros.applied)

    ros = Ros()
    ns = load(ros)
    ns["cartesian_move"](Poses(69), kind="no_such_kind")
    check("an unknown kind is a warning, not a refusal",
          len(ros.moves) == 1, ros.moves)


@_checked
def test_per_node_overrides_reach_only_that_node():
    print("per-node overrides in a kind reach only that node")
    spec = dict(SPEC)
    spec["move_kinds"] = dict(SPEC["move_kinds"])
    spec["move_kinds"]["cartesian_path"] = {
        "parameters": {"position_threshold": 1.0e-4},
        "nodes": {"/move_group": {"position_threshold": 1.0e-3}}}
    ros = Ros(spec=spec)
    ns = load(ros)
    ns["cartesian_move"](Poses(69))
    applied = ros.applied[0]
    check("the common value goes everywhere",
          all(v.get("position_threshold") is not None for v in applied.values()),
          applied)
    check("and the per-node value wins on its own node",
          applied["/move_group"]["position_threshold"] == 1.0e-3
          and applied["/mtc_hub_node"]["position_threshold"] == 1.0e-4,
          applied)

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all checks passed")


_SECTIONS = ["test_the_kind_is_taken_from_the_move", "test_the_overlay_is_put_back", "test_a_cell_without_move_kinds_still_flies", "test_per_node_overrides_reach_only_that_node"]


if __name__ == "__main__":
    sys.exit(_main())
