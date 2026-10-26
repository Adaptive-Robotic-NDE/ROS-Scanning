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
# File:  sensor_orientation_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-01
# Description:
#   Host tests for the safe motion volume, the sensor orientation
#   toggle, and the Keyence normal-following fallback.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-01:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""Three things that decide where the arm goes and how the sensors are pointed.

NO ROS, NO CELL, NO ARM. Everything here is arithmetic over numbers something
else measured, lifted out of the sources by AST so the test cannot drift into
testing a copy:

  1. THE SAFE MOTION VOLUME (work_cell.dpi's derive_safe_motion_volume, plus the
     ex-ur10e-1 manifest's own block). The operator's specification was explicit
     about three faces that must NOT move -- y at both ends, the x_max wall side,
     and the z floor -- and the reason this file exists is that "unchanged" is
     the kind of claim that quietly stops being true. The derivation also has to
     contain the work volume, because a safe motion volume that forbids a pose
     the work volume allows makes every legal part position unreachable with no
     message anywhere.

  2. THE ORIENTATION TOGGLE (ECAutomatedScan.dgp's check/set/describe trio). The
     default has to preserve what this cell has always flown, both sensors must
     refuse a mode they cannot hold, and "orbit" must be refused for the Keyence
     specifically -- a line profiler at a fixed working distance has nothing to
     do with the sides of a part.

  3. THE KEYENCE FALLBACK (mesh_reconstruction.dpi's keyence_normal_fallback).
     Following the surface normal is right for the measurement and is also the
     one change in this pipeline that can drive the eddy-current probe into the
     part, 73 mm to the side of the optical frame on the same plate. So every
     refusal has to send that waypoint back to the locked-Z pose rather than
     costing the stage, and NaN has to count as a refusal everywhere.

Run it either way:

    python3 sensor_orientation_test.py
    python3 -m pytest -q sensor_orientation_test.py
"""

import ast
import functools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")
MANIFEST = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "ros_ws", "src", "tari_site", "tari_cell_configs",
    "config", "cells", "ex-ur10e-1.yaml"))

fails = []


def check(name, ok, detail=""):
    print("  %-74s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises so a standalone run reports every failure
    in one pass; under pytest that would be a test that passes while printing
    FAIL, so this asserts on whatever the section added.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, ("%d check(s) failed: %s"
                            % (len(failed), "; ".join(failed)))
    wrapper.__wrapped__ = fn
    return wrapper


# ==============================================================================
# Lifting things out of the sources
# ==============================================================================

def _tree(path, label):
    return ast.parse(open(path).read(), filename=label)


def _constant(tree, name, label):
    """The value of a top-level `NAME = <literal>`."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError("%s has no %s" % (label, name))


def _functions(tree, wanted, label, extra=None, said=None):
    """`wanted` functions compiled on their own, with `extra` as their globals.

    The namespace IS their globals(), which is what lets a constant be steered
    from here, and print() is captured so a setter's output can be asserted on
    instead of going to stdout.
    """
    nodes = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) \
                and node.name in wanted:
            if node.name in nodes:
                raise AssertionError("%s() is defined twice in %s"
                                     % (node.name, label))
            nodes[node.name] = node
    missing = [w for w in wanted if w not in nodes]
    if missing:
        raise AssertionError(
            "%s no longer defines %s at its top level -- renamed, moved into a "
            "class, or deleted, and this suite would otherwise test nothing"
            % (label, ", ".join(missing)))
    ns = {"np": np, "os": os}
    ns.update(extra or {})
    if said is not None:
        ns["print"] = lambda *a: said.append(" ".join(str(x) for x in a))
    mod = ast.Module(body=[nodes[w] for w in wanted], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), label, "exec"), ns)
    return ns


def _dpi_tree(name):
    """A .dpi parsed with its `global` declarations stripped.

    A .dpi is compiled as a FUNCTION BODY, so `global x` at its top level is
    legal there and a syntax error in a module -- which is what ast.parse treats
    the text as.
    """
    src = open(os.path.join(HERE, name)).read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    return ast.parse(body, filename=name)


# ==============================================================================
# 1. THE SAFE MOTION VOLUME
# ==============================================================================
# The operator's words, 2026-10-01, which this section exists to hold the code
# to:
#
#   x     toward the robot past the world origin and on behind it, out to the
#         arm's reach. The far side does NOT move: there is a wall behind the
#         table.
#   y     IDENTICAL TO THE MOTION VOLUME. "It is also not safe to extend it
#         left or right either" means not beyond what is already allowed there,
#         and what was already allowed there is the motion volume -- the box
#         that bounded every non-contact sweep before this one existed. Taking
#         y from the WORK volume instead made this box tighter sideways than
#         the one it replaces, which would start refusing sweeps that flew.
#   z     floor unchanged (the table, touched off); ceiling to the arm's reach.
#
# THE TOUCHED-OFF BOX, from results16's startup banner and
# cell_override/ex-ur10e-1.work_volume.json. This is the one in force at the
# cell, and it is NOT the manifest's own box -- which is why the derivation takes
# the work volume as an argument instead of reading six numbers out of a file.
WORK_TOUCHED = ((0.587, -0.465, 0.923), (1.121, 0.401, 1.243))
#: The manifest's own hardware work volume, for the same derivation on a session
#: that is not using the override file.
WORK_MANIFEST = ((0.5596, -0.5093, 0.9248), (1.1450, 0.4539, 1.2448))
#: Where base_link sits in `world`, from tari_cell_ex_ur10e_1's
#: environment.urdf.xacro (world -> robot_base at the origin, robot_base -> the
#: arm at "0 0 0.766996"). Read, not measured.
BASE_XYZ = (0.0, 0.0, 0.766996)
#: The motion volume: the work volume plus the tooling pad. y, x_max and z_min
#: come from here, not from the work volume -- see the note above.
MOTION_HARDWARE = ((0.4370, -0.6593, 0.9230), (1.2950, 0.6039, 1.3930))
#: The UR10e's reach: a2 + a3 + d5 from ur_description's ur10e
#: default_kinematics.yaml (0.6127 + 0.57155 + 0.11985), which is the 1300 mm
#: Universal Robots publishes.
ARM_REACH = 1.30410

_wc = _functions(_dpi_tree("work_cell.dpi"), ("derive_safe_motion_volume",),
                 "work_cell.dpi")
derive = _wc["derive_safe_motion_volume"]


@_checked
def test_safe_volume_arithmetic():
    print("")
    print("the safe motion volume, derived from the TOUCHED-OFF work volume")
    lo, hi = derive(WORK_TOUCHED, BASE_XYZ, ARM_REACH)

    # --- the three faces the operator said not to move ------------------------
    check("y is UNCHANGED at both ends -- nothing may widen it left or right",
          abs(lo[1] - WORK_TOUCHED[0][1]) < 1e-12
          and abs(hi[1] - WORK_TOUCHED[1][1]) < 1e-12,
          "y came back %.4f..%.4f" % (lo[1], hi[1]))
    check("the WALL side is not extended: x_max is the work volume's own",
          abs(hi[0] - WORK_TOUCHED[1][0]) < 1e-12,
          "x_max came back %.4f" % (hi[0],))
    check("the floor is UNCHANGED -- it is the table, and it was touched off",
          abs(lo[2] - WORK_TOUCHED[0][2]) < 1e-12,
          "z_min came back %.4f" % (lo[2],))

    # --- the two faces that do move ------------------------------------------
    check("x reaches BEHIND the robot, to base_x - reach",
          abs(lo[0] - (BASE_XYZ[0] - ARM_REACH)) < 1e-12,
          "x_min came back %.4f" % (lo[0],))
    check("...which is past the world origin, as the operator asked",
          lo[0] < 0.0, "x_min came back %.4f" % (lo[0],))
    check("the ceiling is base_z + reach",
          abs(hi[2] - (BASE_XYZ[2] + ARM_REACH)) < 1e-12,
          "z_max came back %.4f" % (hi[2],))

    # --- and it is a superset of the box the probe tip is held to -------------
    check("it CONTAINS the touched-off work volume on all six faces",
          all(a <= b + 1e-12 for a, b in zip(lo, WORK_TOUCHED[0]))
          and all(a >= b - 1e-12 for a, b in zip(hi, WORK_TOUCHED[1])),
          "%s..%s" % (lo, hi))
    # Containment is by CONSTRUCTION, not by luck, so it must also hold for the
    # other work volume a session can be running on.
    mlo, mhi = derive(WORK_MANIFEST, BASE_XYZ, ARM_REACH)
    check("...and the manifest's own work volume too, which is a different box",
          all(a <= b + 1e-12 for a, b in zip(mlo, WORK_MANIFEST[0]))
          and all(a >= b - 1e-12 for a, b in zip(mhi, WORK_MANIFEST[1])),
          "%s..%s" % (mlo, mhi))
    # --- y is the MOTION volume's, on the nose ------------------------------
    ylo, yhi = derive(WORK_TOUCHED, BASE_XYZ, ARM_REACH, motion=MOTION_HARDWARE)
    check("y is identical to the motion volume's y",
          abs(ylo[1] - MOTION_HARDWARE[0][1]) < 1e-12
          and abs(yhi[1] - MOTION_HARDWARE[1][1]) < 1e-12,
          "y came back %.4f..%.4f, motion volume is %.4f..%.4f"
          % (ylo[1], yhi[1], MOTION_HARDWARE[0][1], MOTION_HARDWARE[1][1]))
    # THE REGRESSION THIS EXISTS TO PREVENT. A volume introduced to give tilted
    # poses more room must not take room away on another axis, so it must not be
    # tighter than the motion volume on ANY face.
    check("it is no tighter than the motion volume on any of the six faces",
          all(a <= b + 1e-12 for a, b in zip(ylo, MOTION_HARDWARE[0]))
          and all(a >= b - 1e-12 for a, b in zip(yhi, MOTION_HARDWARE[1])),
          "%s..%s against motion %s..%s"
          % (ylo, yhi, MOTION_HARDWARE[0], MOTION_HARDWARE[1]))
    check("the far x face does not move past the motion volume -- "
          "there is a wall behind the table",
          abs(yhi[0] - MOTION_HARDWARE[1][0]) < 1e-12,
          "x_max came back %.4f, motion volume is %.4f"
          % (yhi[0], MOTION_HARDWARE[1][0]))
    # A manifest that declares a motion volume TIGHTER than the measured work
    # volume must not shrink this box: the work volume is touched off and the
    # motion volume is not.
    tight = ((0.70, -0.10, 1.00), (0.90, 0.10, 1.10))
    glo, ghi = derive(WORK_TOUCHED, BASE_XYZ, ARM_REACH, motion=tight)
    check("a motion volume tighter than the work volume cannot shrink it",
          all(a <= b + 1e-12 for a, b in zip(glo, WORK_TOUCHED[0]))
          and all(a >= b - 1e-12 for a, b in zip(ghi, WORK_TOUCHED[1])),
          "%s..%s" % (glo, ghi))
    check("deriving with no motion volume at all falls back to the work volume",
          abs(mlo[1] - WORK_MANIFEST[0][1]) < 1e-12
          and abs(mhi[1] - WORK_MANIFEST[1][1]) < 1e-12,
          "y came back %.4f..%.4f" % (mlo[1], mhi[1]))

    # --- the numbers themselves, so a changed constant is visible ------------
    check("with the motion volume, the derived box is "
          "x -1.3041..1.2950 y -0.6593..0.6039 z 0.9230..2.0711",
          all(abs(a - b) < 1e-4 for a, b in
              zip(ylo + yhi, (-1.3041, -0.6593, 0.9230,
                              1.2950, 0.6039, 2.0711))),
          "%s..%s" % (ylo, yhi))

    # --- a (frame, lo, hi) triple, which is what work_volume() hands back ----
    tlo, thi = derive(("world",) + WORK_TOUCHED, BASE_XYZ, ARM_REACH)
    check("a (frame, lo, hi) triple gives the same answer as a bare pair",
          tlo == lo and thi == hi)


@_checked
def test_safe_volume_refuses_nonsense():
    print("")
    print("what it refuses rather than returns")
    for reach, why in ((0.0, "zero reach"), (-1.0, "a negative reach"),
                       (float("nan"), "a reach that is not a number")):
        try:
            derive(WORK_TOUCHED, BASE_XYZ, reach)
        except ValueError:
            check("%s is refused" % (why,), True)
        else:
            check("%s is refused" % (why,), False, "it returned a box")
    # A reach typed in MILLIMETRES, which is the live failure mode: 1300 instead
    # of 1.3 widens the box absurdly, but 0.3 instead of 1.3041 SHRINKS the
    # ceiling under the work volume's top and every legal part becomes
    # unreachable. That has to raise, not return.
    try:
        derive(WORK_TOUCHED, BASE_XYZ, 0.3)
    except ValueError as e:
        check("a reach too small to contain the work volume is refused",
              "does not contain the work volume" in str(e), str(e)[:60])
    else:
        check("a reach too small to contain the work volume is refused", False,
              "it returned a box smaller than the work volume")
    check("...while a reach of 1300 (millimetres) still contains it, so the "
          "containment check is not the units check",
          derive(WORK_TOUCHED, BASE_XYZ, 1300.0) is not None)
    check("no work volume means no safe volume, rather than an invented one",
          derive(None, BASE_XYZ, ARM_REACH) is None)


@_checked
def test_manifest_block():
    print("")
    print("what the ex-ur10e-1 manifest declares")
    # THE FILE HAS TO BE THERE. A skip when the tree is not where this expects
    # it is the same shape of silence the whole suite is against: the check does
    # not run, everything still says passed, and the number nobody checked
    # bounds where a robot may move.
    check("the ex-ur10e-1 manifest is where this suite expects it",
          os.path.exists(MANIFEST))
    try:
        import yaml
    except ImportError as e:                      # no PyYAML in this interpreter
        print("skip  the manifest check (%s)" % (e,))
        return
    data = yaml.safe_load(open(MANIFEST))
    block = data.get("safe_motion_volume") or {}
    check("it declares a safe_motion_volume block", bool(block))
    hw = block.get("hardware") or {}
    check("hardware declares the derivation rather than six numbers",
          bool(hw.get("arm_reach")) and bool(hw.get("base_xyz"))
          and not hw.get("min") and not hw.get("max"))
    check("the reach it states is the ur10e's a2 + a3 + d5",
          abs(float(hw.get("arm_reach", 0.0)) - ARM_REACH) < 1e-6,
          "it says %r" % (hw.get("arm_reach"),))
    check("the base position it states is the one in environment.urdf.xacro",
          [float(v) for v in hw.get("base_xyz", ())] == list(BASE_XYZ),
          "it says %r" % (hw.get("base_xyz"),))
    # sim is deliberately absent: nobody has walked that room, and absent means
    # "behave exactly as before", which is the motion volume.
    check("sim declares none, so simulation behaves as it did before",
          not (block.get("sim") or {}))
    # AND THE SAFE VOLUME STILL CONTAINS THE MANIFEST'S OWN WORK VOLUME when the
    # manifest's numbers are used, which is the invariant the whole box rests on.
    wv = ((data.get("work_volume") or {}).get("hardware") or {})
    lo, hi = derive((tuple(wv["min"]), tuple(wv["max"])),
                    hw["base_xyz"], hw["arm_reach"])
    check("the manifest's own numbers derive a box containing its work volume",
          all(a <= b + 1e-9 for a, b in zip(lo, wv["min"]))
          and all(a >= b - 1e-9 for a, b in zip(hi, wv["max"])),
          "%s..%s" % (lo, hi))
    check("...and it does not widen y by a millimetre",
          abs(lo[1] - float(wv["min"][1])) < 1e-9
          and abs(hi[1] - float(wv["max"][1])) < 1e-9)
    # The motion volume is a DIFFERENT box and is not touched by any of this.
    mv = ((data.get("motion_volume") or {}).get("hardware") or {})
    check("the motion volume is left exactly as it was (it has its own test)",
          [float(v) for v in mv.get("min", ())] == [0.4096, -0.6593, 0.9248]
          and [float(v) for v in mv.get("max", ())] == [1.2950, 0.6039, 1.5448],
          "%s..%s" % (mv.get("min"), mv.get("max")))

    # ==========================================================================
    # WHERE THE TWO BOXES DISAGREE, PINNED RATHER THAN LEFT TO BE DISCOVERED
    # ==========================================================================
    # The motion volume was derived as the work volume plus 150 mm of pad in x
    # AND y. The operator's specification pads neither y face and does not pad
    # the wall side. So the safe volume is vastly larger toward and behind the
    # robot and overhead, and SMALLER on three faces -- and a sensor sweep over a
    # part at the y edge of the work volume can be refused where it used to fly.
    # That trade is deliberate and it is the operator's call; this check is here
    # so that if somebody later "fixes" it by padding y, the test says what they
    # have undone.
    check("the safe volume reaches far further toward and behind the robot "
          "than the motion volume (-1.304 against +0.410)",
          lo[0] < float(mv["min"][0]) - 0.8)
    check("...and far higher (2.071 against 1.545)",
          hi[2] > float(mv["max"][2]) + 0.4)
    check("it is TIGHTER on the wall side, which is the operator's instruction "
          "not to extend x past the work volume's far edge",
          hi[0] < float(mv["max"][0]) - 1e-9,
          "safe x_max %.4f vs motion %.4f" % (hi[0], float(mv["max"][0])))
    check("and TIGHTER on both y faces, which is the operator's instruction "
          "that the cell is not clear left or right",
          lo[1] > float(mv["min"][1]) + 1e-9
          and hi[1] < float(mv["max"][1]) - 1e-9,
          "safe y %.4f..%.4f vs motion %.4f..%.4f"
          % (lo[1], hi[1], float(mv["min"][1]), float(mv["max"][1])))
    check("the floors are the same, as they are for every box in this cell",
          abs(lo[2] - float(mv["min"][2])) < 1e-9)


# ==============================================================================
# 2. THE ORIENTATION TOGGLE
# ==============================================================================
# The pattern is set_ec_preload()'s: one validator both ways in, a setter that
# prints what it did, and a describe_* the banner and the stage share. What this
# section is really pinning is the DEFAULT: an unchanged run has to fly the
# sweeps this work cell has always flown, so "z" for both and "fixed" once it
# reaches the planner.

ORIENT_WANT = ("check_sensor_orientation", "set_sensor_orientation",
               "sensor_orientation", "describe_sensor_orientation")


def load_orientation(azure=None, keyence=None):
    """The toggle's four functions, with their constants out of the .dgp.

    -> (namespace, printed lines). `azure`/`keyence` None take the .dgp's own
    declared defaults, which is how the "the default is unchanged" check gets to
    read them rather than restate them.
    """
    tree = _tree(DGP, "ECAutomatedScan.dgp")
    said = []
    extra = {
        "SENSOR_ORIENTATION_MODES": _constant(tree, "SENSOR_ORIENTATION_MODES",
                                              "ECAutomatedScan.dgp"),
        "SENSOR_ORIENTATION_ALIASES": _constant(
            tree, "SENSOR_ORIENTATION_ALIASES", "ECAutomatedScan.dgp"),
        "KEYENCE_NORMAL_MAX_TILT_DEG": _constant(
            tree, "KEYENCE_NORMAL_MAX_TILT_DEG", "ECAutomatedScan.dgp"),
        "KEYENCE_ORIENT_FALLBACK": _constant(tree, "KEYENCE_ORIENT_FALLBACK",
                                             "ECAutomatedScan.dgp"),
        "AZURE_ORIENTATION": (_constant(tree, "AZURE_ORIENTATION",
                                        "ECAutomatedScan.dgp")
                              if azure is None else azure),
        "KEYENCE_ORIENTATION": (_constant(tree, "KEYENCE_ORIENTATION",
                                          "ECAutomatedScan.dgp")
                                if keyence is None else keyence),
    }
    return _functions(tree, ORIENT_WANT, "ECAutomatedScan.dgp",
                      extra=extra, said=said), said


@_checked
def test_default_preserves_todays_behaviour():
    print("")
    print("the default, which must be what this cell has always flown")
    ns, _said = load_orientation()
    tree = _tree(DGP, "ECAutomatedScan.dgp")
    check("AZURE_ORIENTATION defaults to 'z' -- the straight-down dome at one "
          "fixed orientation",
          _constant(tree, "AZURE_ORIENTATION", "ECAutomatedScan.dgp") == "z")
    check("KEYENCE_ORIENTATION defaults to 'z' -- the laser line across travel, "
          "whatever the surface does",
          _constant(tree, "KEYENCE_ORIENTATION", "ECAutomatedScan.dgp") == "z")
    got = ns["sensor_orientation"]()
    check("sensor_orientation() reads both back as 'z'",
          got == {"azure": "z", "keyence": "z"}, repr(got))
    # AND 'z' HAS TO REACH THE PLANNER AS 'fixed', or the default is a different
    # sweep wearing the old name. keyence_orient_mode() is the one translation;
    # it is lifted at the bottom of this file (section 3).
    mode = orient_mode
    check("the planner is handed 'fixed' for 'z', which is bit-for-bit the "
          "orientation flown before this toggle existed",
          mode("z") == "fixed")
    check("...and 'fixed' itself still means 'fixed', so an older config or a "
          "script passing orient='fixed' is unaffected",
          mode("fixed") == "fixed")
    check("AZURE_ORBIT_REQUIRED defaults to False, so asking for an orbit that "
          "cannot be flown never costs the stage",
          _constant(tree, "AZURE_ORBIT_REQUIRED",
                    "ECAutomatedScan.dgp") is False)


@_checked
def test_setter_validates():
    print("")
    print("what the setter accepts and what it refuses")
    ns, said = load_orientation()
    check("set_sensor_orientation(keyence='normal') takes",
          ns["set_sensor_orientation"](keyence="normal")["keyence"] == "normal")
    check("...and it says so, naming itself as the way to change it",
          any("set_sensor_orientation" in l for l in said), repr(said[-1:]))
    check("...and says the Keyence falls back to locked-Z per waypoint",
          any("fall" in l.lower() and "locked-Z" in l for l in said),
          repr(said[-1:]))
    check("set_sensor_orientation(azure='orbit') takes",
          ns["set_sensor_orientation"](azure="orbit")["azure"] == "orbit")

    for bad, why in (("sideways", "a mode that does not exist"),
                     ("", "an empty string"),
                     (None, "None")):
        try:
            ns["set_sensor_orientation"](keyence=bad)
        except ValueError:
            check("%s is refused" % (why,), True)
        else:
            check("%s is refused" % (why,), False, "it was accepted")

    # ORBIT IS A CAMERA'S OPTION. A profiler measures a line at a fixed working
    # distance; flying it round the part is not a mode, it is a raster with the
    # rows in the wrong place.
    try:
        ns["check_sensor_orientation"]("keyence", "orbit")
    except ValueError as e:
        check("'orbit' is refused for the KEYENCE specifically",
              "orbit" in str(e).lower() or "camera" in str(e).lower(),
              str(e)[:70])
    else:
        check("'orbit' is refused for the KEYENCE specifically", False,
              "it was accepted")
    try:
        ns["check_sensor_orientation"]("lidar", "z")
    except ValueError:
        check("a sensor this cell does not have is refused", True)
    else:
        check("a sensor this cell does not have is refused", False)

    # ONE BAD MODE MUST NOT HALF-APPLY A CALL. Both are validated before either
    # is stored.
    ns2, _ = load_orientation()
    try:
        ns2["set_sensor_orientation"](azure="normal", keyence="nonsense")
    except ValueError:
        pass
    check("a call naming one good mode and one bad one changes NEITHER",
          ns2["sensor_orientation"]() == {"azure": "z", "keyence": "z"},
          repr(ns2["sensor_orientation"]()))
    try:
        ns2["set_sensor_orientation"]()
    except ValueError:
        check("set_sensor_orientation() with no arguments is refused rather "
              "than silently doing nothing", True)
    else:
        check("set_sensor_orientation() with no arguments is refused rather "
              "than silently doing nothing", False)


@_checked
def test_describe_is_never_blank():
    print("")
    print("the banner line")
    ns, _said = load_orientation()
    text = ns["describe_sensor_orientation"]()
    check("it names every sensor", "azure" in text and "keyence" in text)
    check("it names the setter that changes it",
          "set_sensor_orientation" in text)
    check("the default reads as locked to world Z for both",
          text.count("LOCKED TO WORLD Z") == 2, text)
    # Aliases resolve in the description too, so a config that still says
    # "fixed" is not reported as broken.
    text = ns["describe_sensor_orientation"](azure="fixed", keyence="fixed")
    check("'fixed' is described as the locked-Z mode, not as an error",
          text.count("LOCKED TO WORLD Z") == 2, text)
    text = ns["describe_sensor_orientation"](azure="orbit", keyence="normal")
    check("orbiting says it is pre-checked and what it falls back to",
          "ORBITING" in text and "pre-check" in text.lower()
          and "falls back" in text.lower(), text)
    check("the normal mode states the tilt cap as a number",
          "NORMAL TO THE SPECIMEN" in text and "30" in text, text)
    # An invalid value must be REPORTED by the banner rather than crashing it:
    # the banner runs at startup, and a bad constant in a config is exactly when
    # somebody needs to be told.
    text = ns["describe_sensor_orientation"](keyence="sideways")
    check("an invalid setting is named in the banner rather than raising",
          "NOT A VALID MODE" in text, text)


# ==============================================================================
# 3. THE KEYENCE NORMAL-FOLLOWING FALLBACK
# ==============================================================================
# Four questions per waypoint; the first one that refuses is the reason reported;
# a refused waypoint is flown at the locked-Z pose rather than lost. The point of
# the whole mode is that it CANNOT fail more often than the vertical raster does,
# so these checks are mostly about refusals producing fallbacks and not
# exceptions.

_MR_TREE = _dpi_tree("mesh_reconstruction.dpi")
REASONS = _constant(_MR_TREE, "KEYENCE_FALLBACK_REASONS",
                    "mesh_reconstruction.dpi")
_mr = _functions(_MR_TREE,
                 ("keyence_normal_fallback", "keyence_orient_mode",
                  "keyence_tilt_summary", "smooth_fallback_mask",
                  "tool_axis_step_deg"), "mesh_reconstruction.dpi",
                 extra={"KEYENCE_FALLBACK_REASONS": REASONS,
                        "KEYENCE_ORIENT_ALIASES": _constant(
                            _MR_TREE, "KEYENCE_ORIENT_ALIASES",
                            "mesh_reconstruction.dpi")})
fallback = _mr["keyence_normal_fallback"]
orient_mode = _mr["keyence_orient_mode"]
tilt_summary = _mr["keyence_tilt_summary"]
smooth = _mr["smooth_fallback_mask"]
axis_step = _mr["tool_axis_step_deg"]

#: The LJ-V7080's guarded window with the 5 mm margin the raster already uses:
#: the hard range is 57..103 mm and the plan is held to 62..98.
WINDOW = (0.062, 0.098)


@_checked
def test_fallback_picks_locked_z_when_refused():
    print("")
    print("one waypoint per refusal, and every refusal is a fallback")
    # Five waypoints: one good, then one of each refusal.
    tilt = np.array([10.0, 44.0, 10.0, 10.0, 10.0])
    cable = np.array([True, True, False, True, True])
    clear = np.array([0.030, 0.030, 0.030, 0.001, 0.030])
    stand = np.array([0.080, 0.080, 0.080, 0.080, 0.115])
    use, reason, tally = fallback(tilt, cable_ok=cable, clearance_m=clear,
                                 standoff_m=stand, max_tilt_deg=30.0,
                                 clearance_limit_m=0.005, window_m=WINDOW)
    check("the good waypoint keeps the normal pose", bool(use[0]))
    check("tilt past the cap falls back", not use[1] and reason[1] == "tilt",
          reason[1])
    check("a cable band that does not admit the tilt falls back",
          not use[2] and reason[2] == "cable", reason[2])
    check("tooling clearance inside the limit falls back -- this is the "
          "eddy-current probe going into the part",
          not use[3] and reason[3] == "clearance", reason[3])
    check("a standoff outside the guarded window falls back",
          not use[4] and reason[4] == "standoff", reason[4])
    check("the tally adds up: 1 normal, 4 fallbacks",
          tally["normal"] == 1 and tally["fallback"] == 4, repr(tally))
    check("...and every fallback is attributed to exactly one reason",
          sum(tally[r] for r in REASONS) == tally["fallback"], repr(tally))
    check("the reason is a NAME, not a number, so a log line says what happened",
          set(reason) <= set(("",)) | set(REASONS), repr(reason))


@_checked
def test_fallback_is_conservative():
    print("")
    print("what counts as a refusal")
    # NaN IS A REFUSAL EVERYWHERE. A clearance that could not be measured, a ray
    # that hit nothing, a tilt that is not a number: none of those is permission
    # to lean the tool over.
    nan = float("nan")
    use, reason, _t = fallback(np.array([nan, 10.0, 10.0, 10.0]),
                               clearance_m=np.array([0.03, nan, 0.03, 0.03]),
                               standoff_m=np.array([0.08, 0.08, nan, 0.08]),
                               cable_ok=np.array([True, True, True, True]),
                               max_tilt_deg=30.0, clearance_limit_m=0.005,
                               window_m=WINDOW)
    check("a tilt that is not a number falls back",
          not use[0] and reason[0] == "tilt")
    check("a clearance that could not be measured falls back",
          not use[1] and reason[1] == "clearance")
    check("a standoff ray that hit nothing falls back",
          not use[2] and reason[2] == "standoff")
    check("and the measurable waypoint beside them is unaffected", bool(use[3]))

    # EXACTLY AT A LIMIT IS ALLOWED, at both ends of the window and at the cap.
    # A boundary that refuses would turn the mode off over a flat plate measured
    # exactly at the reference distance.
    use, _r, _t = fallback(np.array([30.0, 10.0, 10.0]),
                           clearance_m=np.array([0.005, 0.005, 0.005]),
                           standoff_m=np.array([0.080, WINDOW[0], WINDOW[1]]),
                           max_tilt_deg=30.0, clearance_limit_m=0.005,
                           window_m=WINDOW)
    check("a waypoint exactly at the tilt cap, the clearance limit and either "
          "end of the window is allowed", bool(use.all()), repr(use))

    # AN UNMEASURED CABLE ENVELOPE IS NOT A REFUSAL. cable_ok=None means nothing
    # has been measured for this cell; refusing every tilted waypoint on that
    # basis would silently turn the normal mode back into the vertical one while
    # reporting that it was on.
    use, _r, tally = fallback(np.array([10.0, 20.0]), cable_ok=None,
                              clearance_m=np.array([0.03, 0.03]),
                              max_tilt_deg=30.0, clearance_limit_m=0.005)
    check("no cable envelope measured is not by itself a refusal",
          bool(use.all()) and tally["cable"] == 0, repr(tally))
    # With nothing measured at all, nothing is refused -- the caller's own
    # vertical raster is still the floor.
    use, _r, tally = fallback(np.array([5.0, 7.0]), max_tilt_deg=30.0)
    check("with no measurements supplied at all, the tilt cap is the only gate",
          bool(use.all()) and tally["fallback"] == 0, repr(tally))


@_checked
def test_fallback_extremes():
    print("")
    print("the two ends: everything tilted, nothing tilted")
    # A surface too steep everywhere: EVERY waypoint falls back, and that is the
    # vertical raster -- the one flown today -- not a failure.
    use, _r, tally = fallback(np.full(50, 70.0), max_tilt_deg=30.0)
    check("a surface too steep everywhere falls back completely rather than "
          "raising, which is the raster flown today",
          not use.any() and tally["fallback"] == 50, repr(tally))
    use, _r, tally = fallback(np.zeros(50), max_tilt_deg=30.0)
    check("a flat plate keeps the normal pose everywhere (it IS vertical there)",
          bool(use.all()) and tally["fallback"] == 0, repr(tally))
    check("an empty path is an empty answer, not an exception",
          fallback(np.zeros(0))[2]["fallback"] == 0)


@_checked
def test_tilt_summary():
    print("")
    print("the tilt distribution line")
    check("an empty distribution has nothing to say", tilt_summary([]) is None)
    line = tilt_summary(np.concatenate([np.zeros(90), np.full(10, 40.0)]))
    # PERCENTILES, NOT A MEAN. What decides whether a raster flies is the worst
    # few percent of its waypoints, and a mean over a mostly-flat plate hides
    # them: this distribution has a mean of 4 deg and a worst case of 40.
    check("it reports the worst case, which a mean would hide",
          "40.0" in line, line)
    check("...and the p95 and p99, which is where a raster actually breaks",
          "p95" in line and "p99" in line, line)


@_checked
def test_no_alternating_along_a_pass():
    print("")
    print("runs shorter than the minimum are absorbed, so the tool is not "
          "tilted on and off")
    # ALTERNATION IS THE WORST CASE. Every flip costs a full orientation step
    # between two waypoints a millimetre or two apart, which is a Cartesian path
    # whose IK branch can change mid-pass.
    alt = np.array([True, False]*20)
    out = smooth(alt, (), min_run=8)
    check("an alternating mask collapses to one choice for the whole pass",
          len(set(out.tolist())) == 1, repr(out.tolist()))
    # A real crossing -- a long following stretch then a long fallback -- is left
    # exactly as it is: that is one step, at one place, and it is the answer.
    real = np.array([True]*30 + [False]*30)
    out = smooth(real, (), min_run=8)
    check("a single genuine crossing is left alone", bool((out == real).all()))
    # A SHORT ISLAND loses to its surroundings, whichever way round it is.
    island = np.array([True]*20 + [False]*3 + [True]*20)
    out = smooth(island, (), min_run=8)
    check("a 3-waypoint island of fallback inside a following stretch is "
          "absorbed", bool(out.all()), repr(out.tolist()))
    island = np.array([False]*20 + [True]*3 + [False]*20)
    out = smooth(island, (), min_run=8)
    check("...and a 3-waypoint island of following inside a fallen-back "
          "stretch too", not out.any(), repr(out.tolist()))
    # RUNS DO NOT CROSS A PASS BOUNDARY: the arm lifts and re-enters there, so
    # a change at a boundary costs nothing, and a 4-long run that is a whole
    # short pass must not be absorbed into the next pass.
    two = np.array([True]*4 + [False]*4)
    out = smooth(two, (0, 4), min_run=8)
    check("a run that IS a whole pass is left alone, however short the pass",
          bool((out == two).all()), repr(out.tolist()))
    check("min_run of 1 is the way to turn the smoothing off",
          bool((smooth(alt, (), min_run=1) == alt).all()))
    check("an empty mask is an empty answer", len(smooth([], (), 8)) == 0)


@_checked
def test_tool_axis_step():
    print("")
    print("the worst orientation step on the path that will fly")
    # Identity is the tool pointing along +z; a 90 deg rotation about x points
    # it along -y. Two poses, one step.
    ident = (0.0, 0.0, 0.0, 1.0)
    rot90x = (np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4))
    step = axis_step([ident, rot90x])
    check("it measures the angle between adjacent tool axes",
          len(step) == 1 and abs(step[0] - 90.0) < 1e-6, repr(step))
    # A ROLL ABOUT THE TOOL AXIS IS NOT A STEP: that is the wrist-3 question,
    # and the cable guard bounds it. Measuring it here would warn about every
    # raster that turns the laser line.
    rot90z = (0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4))
    step = axis_step([ident, rot90z])
    check("a pure roll about the tool axis is NOT counted as a step",
          abs(step[0]) < 1e-6, repr(step))
    # A STEP ACROSS A PASS BOUNDARY IS NOT COUNTED: the arm lifts, approaches
    # and re-enters there.
    step = axis_step([ident, ident, rot90x, rot90x], segment_starts=(0, 2))
    check("a step at a pass boundary is excluded", float(np.max(step)) < 1e-6,
          repr(step))
    step = axis_step([ident, rot90x, rot90x, rot90x], segment_starts=(0, 2))
    check("...while a step INSIDE a pass is counted",
          abs(float(np.max(step)) - 90.0) < 1e-6, repr(step))
    check("one pose has no steps", len(axis_step([ident])) == 0)


@_checked
def test_orient_vocabulary():
    print("")
    print("the orientation vocabulary, resolved in one place")
    for name, want in (("z", "fixed"), ("Z", "fixed"), ("vertical", "fixed"),
                       ("locked", "fixed"), ("fixed", "fixed"),
                       ("normal", "normal"), ("specimen", "normal"),
                       ("surface", "surface"), ("capped", "capped"),
                       ("auto", "auto")):
        check("%r reaches the planner as %r" % (name, want),
              orient_mode(name) == want, repr(orient_mode(name)))
    check("'normal' is its OWN mode, not an alias for 'capped' -- what makes it "
          "usable is the per-waypoint fallback, and a capped path has none",
          orient_mode("normal") != "capped")
    check("an unknown name is passed through for the planner to refuse with its "
          "own message", orient_mode("sideways") == "sideways")
    check("None stays None, which means 'whatever the class default is'",
          orient_mode(None) is None)


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            getattr(_fn, "__wrapped__", _fn)()
    print("")
    if fails:
        print("%d FAILED" % len(fails))
        for _f in fails:
            print("  " + _f)
        sys.exit(1)
    print("all passed")
