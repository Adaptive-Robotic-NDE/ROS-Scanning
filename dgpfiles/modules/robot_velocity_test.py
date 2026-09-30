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
# File:  robot_velocity_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests turning a measured velocity ladder into the robot velocity
#   calibration table.
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
"""Turning a velocity ladder into a ROBOT_VELOCITY_CALIBRATION table.

WHY THIS MATTERS MORE THAN IT LOOKS. The table is not a record of a
measurement -- it is consumed, by velocity_from_scaling() and
velocity_scaling_for() in scan_parameters.dpi, which interpolate between
consecutive rows. Those two functions are how every technician-facing mm/s in
the pipeline becomes a MoveIt scaling.

A BAD TABLE DOES NOT CRASH THEM, and that is checked here rather than assumed.
Both functions open with an early return for the below-the-first-row case and
then test `s <= s1` / `v <= v1` in order, which makes every degenerate pair
unreachable: a duplicate scaling, two equal speeds, a zero in a later row. All
five shapes return a number.

WHICH IS WHY THE CHECKING HAS TO HAPPEN HERE. The number is wrong and nothing
downstream can tell:

  * a rung whose speed FELL as the scaling rose makes the curve claim that more
    scaling produces less speed -- and every speed inside that dip has two
    scalings, of which velocity_scaling_for() silently returns the lower. That
    is "the arm is too slow" baked into a config constant.
  * a zero-speed rung sorts to the FRONT, which is the one row both functions
    use as their below-the-table anchor, and is the only input that can really
    divide by zero.

Neither is hypothetical. A real ladder produces both: capture_move() computes
its own scaling from the path's kinematics and only *caps* it at what was asked
(cell_capture_motion.dpi), so two requested speeds can be clamped to one
scaling; and one overshooting move makes _slow_down() cut every later move, so
a higher rung can come back slower than a lower one.

Everything here runs on the HOST with numpy only -- the arithmetic that decides
what gets written into a config has no business needing a robot to check.

    python3 robot_velocity_test.py
    python3 -m pytest -q robot_velocity_test.py
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


def _strip_globals(name):
    src = open(os.path.join(HERE, name)).read()
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


#: scan_parameters.dpi is loadable whole -- it is pure numpy. The real
#: interpolators, not a second copy of their arithmetic: a test that reimplements
#: the thing it is checking cannot catch a disagreement with it.
ns = {"np": np, "os": os}
exec(compile(_strip_globals("scan_parameters.dpi"), "scan_parameters.dpi", "exec"), ns)

#: cell_capture_scenarios.dpi is not loadable here (it needs the capture node, TF
#: and the motion gate), so lift the two functions under test by AST -- the same
#: trick keyence_standoff_test.py uses for _crop_path_by_mask.
_tree = ast.parse(_strip_globals("cell_capture_scenarios.dpi"))
# describe_velocity_table() now prints the manifest block to paste, so the two
# helpers that build it come along. Lifted by name, so a rename there is a failure
# here rather than a report that quietly loses its most useful half.
_want = {"velocity_table_from_rungs", "describe_velocity_table",
         "velocity_manifest_block", "_velocity_manifest_target"}
_fns = [n for n in _tree.body
        if isinstance(n, ast.FunctionDef) and n.name in _want]
missing = _want - {f.name for f in _fns}
assert not missing, (
    "cell_capture_scenarios.dpi no longer defines %s at its top level -- renamed, "
    "moved, or deleted, and this suite would otherwise test nothing"
    % ", ".join(sorted(missing)))
exec(compile(ast.Module(body=_fns, type_ignores=[]), "cell_capture_scenarios.dpi",
             "exec"), ns)

table_from_rungs = ns["velocity_table_from_rungs"]
describe = ns["describe_velocity_table"]
manifest_block = ns["velocity_manifest_block"]
velocity_from_scaling = ns["velocity_from_scaling"]
velocity_scaling_for = ns["velocity_scaling_for"]
ns["JOINT_MOVE_SCALING"] = 0.015


def rung(scaling, observed, ok=True, commanded=None):
    return {"scaling": scaling, "observed_mps": observed, "ok": ok,
            "commanded_mps": commanded if commanded is not None else observed}


# =============================================================================
print("")
print("a clean ladder becomes a table the interpolators can use")

# Shaped like the existing (simulated) table: rising, saturating at the top.
clean = [rung(0.0079, 0.0050), rung(0.0156, 0.0100), rung(0.0303, 0.0200),
         rung(0.0472, 0.0350), rung(0.0529, 0.0500)]
out = table_from_rungs(clean)
check(out["ok"] and len(out["table"]) == 5,
      "five good rungs become five rows")
check(not out["dropped"], "...and nothing is dropped")
check(out["table"] == sorted(out["table"]),
      "...sorted ascending by scaling, which is what the interpolators assume")

# THE ROUND TRIP, through the real functions. This is the property that makes it
# a calibration rather than a list.
worst = 0.0
for s, v in out["table"]:
    worst = max(worst, abs(velocity_from_scaling(velocity_scaling_for(v, out["table"]),
                                                 out["table"]) - v))
check(worst < 1e-9,
      "every row round-trips speed -> scaling -> speed (worst %.2e)" % (worst,))
mid = velocity_from_scaling(0.0200, out["table"])
check(0.0100 < mid < 0.0200,
      "...and a scaling between two rows interpolates between their speeds "
      "(%.4f m/s at scaling 0.0200)" % (mid,))


# =============================================================================
print("")
print("two rungs at one scaling")

# capture_move() clamps the scaling it will use, so two requested speeds can
# come back at the SAME scaling. Left in, the table asserts two different
# speeds for one command -- self-contradictory as a record, even though the
# interpolators survive it (see the module docstring).
dup = [rung(0.0079, 0.0050), rung(0.0156, 0.0100),
       rung(0.0156, 0.0121), rung(0.0303, 0.0200)]
out = table_from_rungs(dup)
check(out["ok"], "the table is still produced")
check(len({s for s, _v in out["table"]}) == len(out["table"]),
      "every scaling in it is distinct")
check(any("duplicate scaling" in w for _s, _v, w in out["dropped"]),
      "...and the duplicate is reported, not silently merged")
check((0.0156, 0.0121) in out["table"],
      "...keeping the FASTER of the two, which is the one that shows what the "
      "arm did at that scaling")

# The interpolators survive the raw duplicate -- asserted so that if a future
# change to them makes it crash, this suite says so instead of a scan doing it.
raw = [(r["scaling"], r["observed_mps"]) for r in dup]
try:
    velocity_from_scaling(0.0200, raw)
    check(True, "the interpolator survives a raw duplicate, so the dedup is "
                "about the table being TRUE and not about avoiding a crash")
except ZeroDivisionError:
    check(False, "velocity_from_scaling() divides by zero on a duplicate "
                 "scaling, which it did not before: the dedup is now "
                 "load-bearing for a second reason")


# =============================================================================
print("")
print("a speed that falls as the scaling rises")

# What _slow_down() produces: one move overshoots, every later move is cut, and
# a higher rung comes back slower. Interpolating over that inverts wrongly.
fell = [rung(0.0079, 0.0050), rung(0.0156, 0.0100), rung(0.0303, 0.0090),
        rung(0.0472, 0.0350)]
out = table_from_rungs(fell)
check(out["ok"], "the rest of the ladder still yields a table")
check(all(out["table"][i][1] < out["table"][i + 1][1]
          for i in range(len(out["table"]) - 1)),
      "the speeds in it rise monotonically")
check(any("did not rise" in w for _s, _v, w in out["dropped"]),
      "...and the rung that fell is named rather than smoothed away")
check((0.0303, 0.0090) not in out["table"],
      "...and is not in the table")

lines = describe(out)
check(any("dropped scaling 0.0303" in l for l in lines),
      "the operator is told which rung was dropped and why")

# THE HARM, DEMONSTRATED. Left in, the curve claims that MORE scaling gives
# LESS speed -- which is false about any arm and is what _slow_down() leaves
# behind after one overshoot.
bad = [(r["scaling"], r["observed_mps"]) for r in fell]
check(velocity_from_scaling(0.0303, bad) < velocity_from_scaling(0.0156, bad),
      "the UNCHECKED table says scaling 0.0303 is SLOWER than 0.0156 "
      "(%.1f vs %.1f mm/s) -- the wrongness the drop prevents"
      % (1e3*velocity_from_scaling(0.0303, bad),
         1e3*velocity_from_scaling(0.0156, bad)))
good = out["table"]
check(velocity_from_scaling(0.0303, good) > velocity_from_scaling(0.0156, good),
      "...and the checked table does not")
# And the dip is not invertible: a speed inside it has two scalings, and the
# function returns the lower, which is the one that scans slowly forever.
check(velocity_scaling_for(0.0095, bad) < 0.0303,
      "asking the UNCHECKED table for a speed inside the dip returns the LOWER "
      "of the two scalings that produce it (%.4f, not ~0.029)"
      % (velocity_scaling_for(0.0095, bad),))


print("")
print("a rung that never moved")

# A refused or stalled move reports 0 m/s. Zero sorts to the FRONT, which is the
# row both interpolators use as their below-the-table anchor.
zeroed = [rung(0.0079, 0.0000), rung(0.0156, 0.0100), rung(0.0303, 0.0200)]
out0 = table_from_rungs(zeroed)
check(all(v > 0.0 for _s, v in out0["table"]),
      "a zero-speed rung is not in the table")
check(all(s > 0.0 for s, _v in out0["table"]),
      "...nor is a zero-scaling one")
check(out0["ok"] and len(out0["table"]) == 2,
      "...and the rungs that did move still make a curve")
# It is the one input that can genuinely divide by zero, at v <= 0.
try:
    velocity_scaling_for(0.0, [(0.0156, 0.0000), (0.0303, 0.0200)])
    check(False, "a zero first row divides by zero at v = 0")
except ZeroDivisionError:
    check(True, "a zero first row DOES divide by zero at v = 0, which is why "
                "zero rungs are dropped rather than merely distrusted")


# =============================================================================
print("")
print("not enough to make a curve")

check(table_from_rungs([])["ok"] is False,
      "no rungs is not a table")
check(table_from_rungs([rung(0.0156, 0.0100)])["ok"] is False,
      "one rung is not a table -- a curve needs two points")
one = table_from_rungs([rung(0.0156, 0.0100)])
check("at least two" in one["why"], "...and it says why")
check(table_from_rungs([rung(0.0079, 0.0050, ok=False),
                        rung(0.0156, 0.0100, ok=False)])["ok"] is False,
      "rungs whose moves did not complete are not measurements")
check(table_from_rungs([rung(0.0079, None), rung(0.0156, 0.0100)])["ok"] is False,
      "...nor are rungs that reported no speed")

lines = describe({"table": [], "why": "only 1 usable rung(s)"})
check(any("NO VELOCITY TABLE" in l for l in lines),
      "an empty result says so rather than printing an empty block")


# =============================================================================
print("")
print("what the operator is told, and can paste")

out = table_from_rungs(clean)
out["joint"] = [{"scaling": 0.010, "observed_mps": 0.031, "ok": True,
                 "over_limit": False},
                {"scaling": 0.015, "observed_mps": 0.047, "ok": True,
                 "over_limit": False},
                {"scaling": 0.020, "observed_mps": 0.062, "ok": True,
                 "over_limit": True}]
lines = describe(out)
text = "\n".join(lines)

check("robot_velocity" in text and "MANIFEST" in text.upper(),
      "it points at the CELL MANIFEST, which is where the pipeline reads the "
      "curve from -- a config constant has to be transcribed by every machine "
      "and eventually is not")

# THE BLOCK MUST BE VALID YAML THAT ROUND-TRIPS TO THE MEASURED TABLE. A
# formatting slip here is pasted into a manifest and read as gospel.
import yaml as _yaml
block_lines = manifest_block(out)
doc = _yaml.safe_load("\n".join(block_lines))
check(list(doc.keys()) == ["robot_velocity"],
      "the block is a single robot_velocity: mapping, ready to paste")
modes = doc["robot_velocity"]
check(len(modes) == 1, "and declares exactly one mode, not both")
rows = list(modes.values())[0]["table"]
check(len(rows) == len(out["table"]),
      "the block parses as YAML and has one row per measurement")
check(all(abs(a - c) < 1e-9 and abs(b - d) < 1e-9
          for (a, b), (c, d) in zip([tuple(r) for r in rows], out["table"])),
      "...and its numbers are the measured ones, to full precision")
# And it must survive being fed straight back in.
pasted = [tuple(r) for r in rows]
check(abs(velocity_from_scaling(0.0200, pasted)
          - velocity_from_scaling(0.0200, out["table"])) < 1e-12,
      "...and the loaded table behaves identically to the measured one")
# THE SPAN TRAVELS WITH IT. The curve is path-length dependent, and without the
# span nothing downstream can tell whether it was measured over a distance like
# the one it is being applied to.
out_with_span = dict(out)
out_with_span["span_m"] = 0.3
spanned = _yaml.safe_load("\n".join(manifest_block(out_with_span)))
check(list(spanned["robot_velocity"].values())[0].get("span_m") == 0.3,
      "the span the curve was measured over is written into the block")

check("globals()['ROBOT_VELOCITY_CALIBRATION']" in text,
      "the session-only override is still offered, for trying it without an edit")
check("does NOT persist" in text,
      "...and is marked as not persisting, because pasting it at the prompt in "
      "five separate sessions is what this replaces")

check(any("joint move's tip speed depends on the arm's configuration" in l
          for l in lines),
      "the joint half is reported as a SEPARATE question, not as a row of the "
      "curve")
check(any("0.0150" in l and "47.0 mm/s" in l for l in lines),
      "...with the tip peak each joint scaling produced")
check(any("OVER THE LIMIT" in l for l in lines),
      "...and the scaling that broke the limit is marked")
check(any("largest scaling that stayed inside the limit was 0.0150" in l
          for l in lines),
      "...and it names the largest scaling that stayed inside it, which is the "
      "number JOINT_MOVE_SCALING wants")


# =============================================================================
print("")
print("the saturating top of the curve is preserved")

# The existing table saturates: 0.0472 -> 35 mm/s but 0.0529 -> 50. Above the
# last row velocity_from_scaling() is FLAT by design (the cell stops speeding
# up), and velocity_scaling_for() must not extrapolate past it.
out = table_from_rungs(clean)
top_s, top_v = out["table"][-1]
check(abs(velocity_from_scaling(top_s*2.0, out["table"]) - top_v) < 1e-12,
      "a scaling above the top row returns the top row's speed, not more")
check(abs(velocity_scaling_for(top_v*2.0, out["table"]) - top_s) < 1e-12,
      "and asking for a speed the cell cannot reach returns the most it can do")


def test_robot_velocity():
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
