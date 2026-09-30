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
# File:  ec_preload_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   The eddy-current contact preload: its sign, its frame, and the
#   limits that keep it from being set by accident.
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
"""How hard the eddy-current probe is pressed into the part, and what bounds it.

The preload is the distance the tip is commanded PAST the measured surface. It
loads the spring fixture so the probe face stays on the part over the
millimetre-scale error in the reconstructed mesh. It drives a real arm with a
probe touching metal, so three things about it have to be true and none of them
is obvious from reading the call:

  THE SIGN. The offset is (0, 0, -preload) in the tool frame, and on this tooling
  tool z points AWAY from the part -- probe_end is built with tool z on the
  OUTWARD surface normal, which is why tool_tilt_deg() measures it against
  straight up and why a 45 degree cap means anything. So -z is into the material
  and +z would be lift-off. Get it backwards and the scan quietly lifts the probe
  off the part by the amount it was told to press.

  THE FRAME. It is applied in each pose's OWN frame, so on a curved part the push
  follows that pose's normal. A world-z push of the same size would press the
  flanks of a curved part at an angle and cut into them.

  THE LIMITS. Negative is refused because it is lift-off and not preload; above
  EC_PRELOAD_CEILING_M is refused because past there the arm presses on the part
  rather than on the spring.

Everything here runs on the HOST with numpy: no ROS, no robot, no instrument, no
mesh. The gate and the setter are lifted out of ECAutomatedScan.dgp by its AST --
a .dgp is a function body and is not importable -- so this cannot pass against a
stale copy, and the constants come out of the same source rather than being
repeated here.

    python3 ec_preload_test.py
    python3 -m pytest -q ec_preload_test.py
"""
import ast
import functools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")

#: The functions the preload is made of, all in the .dgp.
WANT = ("check_ec_preload", "set_ec_preload", "ec_preload", "describe_ec_preload",
        "say_path_preload")

fails = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so a standalone run reports every failure
    in one pass. Under pytest that would be a test that passes while printing
    FAIL, so the wrapper asserts on whatever the section added.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed), "; ".join(failed))
    wrapper.__wrapped__ = fn
    return wrapper


# ==============================================================================
# The .dgp's preload functions and constants, out of the source
# ==============================================================================

def dgp_tree():
    return ast.parse(open(DGP).read(), filename="ECAutomatedScan.dgp")


def dgp_constant(tree, name):
    """The value of a top-level `NAME = <literal>` in the .dgp."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError("ECAutomatedScan.dgp has no %s" % (name,))


def load_preload():
    """The preload functions, compiled on their own. -> (namespace, printed list)

    The namespace IS their globals(), so EC_PRELOAD_M and EC_PRELOAD_CEILING_M
    are steered from here, and print() is captured rather than going to stdout.
    """
    tree = dgp_tree()
    nodes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            if node.name in nodes:
                raise AssertionError("%s() is defined twice in "
                                     "ECAutomatedScan.dgp" % (node.name,))
            nodes[node.name] = node
    missing = [w for w in WANT if w not in nodes]
    if missing:
        raise AssertionError("ECAutomatedScan.dgp has no %s()"
                             % (", ".join(missing),))
    said = []
    ns = {"np": np, "print": lambda *a: said.append(" ".join(str(x) for x in a)),
          "EC_PRELOAD_M": dgp_constant(tree, "EC_PRELOAD_M"),
          "EC_PRELOAD_CEILING_M": dgp_constant(tree, "EC_PRELOAD_CEILING_M")}
    mod = ast.Module(body=[nodes[w] for w in WANT], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "ECAutomatedScan.dgp", "exec"), ns)
    return ns, said


CEILING_M = dgp_constant(dgp_tree(), "EC_PRELOAD_CEILING_M")


# ==============================================================================
# The geometry: ScanPath, the quaternion helpers, and the tool-frame offset
# ==============================================================================

def load_geometry():
    """_offset_path_in_tool_frame() with everything it needs, from the .dpi files.

    mesh_reconstruction.dpi owns the offset; toolpath_segments.dpi owns ScanPath;
    noether_path.dpi owns the quaternion helpers and the two functions that
    define the tool-axis convention this whole file is about. The work-volume
    lookups are stubbed because nothing here plans a path.
    """
    box = ((0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    ns = {"work_volume": lambda *a, **k: box,
          "require_work_volume": lambda *a, **k: box}

    def _exec(name, drop=()):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                        for l in src.splitlines())
        for d in drop:
            src = src.replace(d, "pass  # " + d.replace("\n", " "))
        exec(compile(src, name, "exec"), ns)

    _exec("toolpath_segments.dpi",
          drop=("from geometry_msgs.msg import PoseArray, Pose",))
    # Only the pieces of noether_path.dpi this needs: the whole file wants ROS
    # service types. Taken by name so a rename here is a failure rather than a
    # silently missing check.
    src = open(os.path.join(HERE, "noether_path.dpi")).read()
    lines, keep, taking = src.splitlines(), [], False
    wanted = ("def quat_mul(", "def quat_to_matrix(", "def matrix_to_quat(",
              "def _unit_quats(", "def _swing_tool_axis_onto(",
              "def tool_tilt_deg(")
    for line in lines:
        if any(line.startswith(w) for w in wanted):
            taking = True
        elif taking and line and not line[0].isspace() and not line.startswith(")"):
            taking = False
        if taking:
            keep.append(line)
    exec(compile("import numpy as np\n" + "\n".join(keep), "noether_path.dpi",
                 "exec"), ns)
    for name in wanted:
        fn = name[4:-1]
        if fn not in ns:
            raise AssertionError("noether_path.dpi no longer defines %s()" % (fn,))
    _exec("mesh_reconstruction.dpi")
    return ns


GEOM = load_geometry()


def path_on_surface(normals, spacing=0.001):
    """A pass of poses whose tool z sits exactly on `normals`, as the planner builds it.

    Positions walk along +x; the orientation of each is the reference "probe
    pointing straight down" pose swung onto that waypoint's outward normal by
    _swing_tool_axis_onto() -- the same call orient='normal' uses. So these are
    poses of the shape plan_ec_path() hands to the preload, not hand-written
    quaternions that happen to look right.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n = len(normals)
    pos = np.zeros((n, 3))
    pos[:, 0] = spacing*np.arange(n)
    pos[:, 2] = 1.0
    # Tool z up, tool x along +x: the untilted reference orientation.
    upright = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    quats = GEOM["_swing_tool_axis_onto"](upright, normals)
    return GEOM["ScanPath"](pos, quats, [0], source="test pass")


# ==============================================================================
print("")
print("THE CONVENTION: tool z points away from the part")


@_checked
def test_tool_axis_is_the_outward_normal():
    """An upright pose's tool z is world +z, which is out of a horizontal part.

    This is the fact the sign of the offset rests on. tool_tilt_deg() measures
    the tool axis against straight UP and EC_MAX_TILT_DEG caps that at 45
    degrees, so a pose whose tool z pointed into the part would read 180 degrees
    and be thrown away -- the planner's poses cannot be the other convention.
    """
    path = path_on_surface([[0.0, 0.0, 1.0]]*4)
    z = np.array([GEOM["quat_to_matrix"](q)[:, 2] for q in path.quats])
    check("an upright pose's tool z is world +z",
          np.allclose(z, np.array([0.0, 0.0, 1.0]), atol=1e-12), z[0])
    tilt = GEOM["tool_tilt_deg"](path.quats)
    check("and tool_tilt_deg() reads 0 for it, not 180",
          np.allclose(tilt, 0.0, atol=1e-9), tilt)
    # A pose on a 30 degree flank: still inside the 45 degree cap.
    flank = [math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))]
    tilted = GEOM["tool_tilt_deg"](path_on_surface([flank]*3).quats)
    check("a pose on a 30 deg flank tilts 30 deg, inside EC_MAX_TILT_DEG",
          np.allclose(tilted, 30.0, atol=1e-6), tilted)


# ==============================================================================
print("")
print("THE OFFSET: into the part, along each pose's own normal, by exactly the preload")


def apply_preload(path, preload):
    """What plan_ec_path() does with a nonzero preload, and nothing else."""
    return GEOM["_offset_path_in_tool_frame"](path, (0.0, 0.0, -float(preload)))


@_checked
def test_preload_moves_the_tip_into_a_flat_part():
    preload = 0.0003
    before = path_on_surface([[0.0, 0.0, 1.0]]*5)
    after = apply_preload(before, preload)
    moved = after.positions - before.positions
    check("every pose moves exactly the preload",
          np.allclose(np.linalg.norm(moved, axis=1), preload, atol=1e-15),
          np.linalg.norm(moved, axis=1))
    check("and it moves DOWN, into the material, not up off it",
          np.allclose(moved[:, 2], -preload, atol=1e-15), moved[:, 2])
    check("x and y are untouched on a flat part",
          np.allclose(moved[:, :2], 0.0, atol=1e-15), moved[:, :2])
    check("orientations are untouched -- this moves where the tool is",
          np.allclose(after.quats, before.quats, atol=1e-15))


@_checked
def test_preload_follows_the_surface_normal_not_world_z():
    """On a flank the push leans with the normal, which world z cannot do."""
    preload = 0.0005
    for deg in (15.0, 30.0, 44.0):
        nrm = np.array([math.sin(math.radians(deg)), 0.0, math.cos(math.radians(deg))])
        before = path_on_surface([nrm]*4)
        after = apply_preload(before, preload)
        moved = after.positions - before.positions
        check("%.0f deg flank: the move is exactly the preload long" % (deg,),
              np.allclose(np.linalg.norm(moved, axis=1), preload, atol=1e-15),
              np.linalg.norm(moved, axis=1))
        check("%.0f deg flank: the move is along MINUS the surface normal" % (deg,),
              np.allclose(moved, -preload*nrm, atol=1e-15), moved[0])
        # The tell that it is not a world-z push: the vertical component is
        # short by cos(tilt) and there is sideways travel to make up for it.
        check("%.0f deg flank: the vertical drop is preload*cos(tilt), less than "
              "the preload" % (deg,),
              abs(moved[0, 2] + preload*math.cos(math.radians(deg))) < 1e-15,
              moved[0, 2])
        check("%.0f deg flank: the tip also moves sideways, which a world-z push "
              "would not" % (deg,),
              abs(moved[0, 0]) > 0.1*preload*math.sin(math.radians(deg)), moved[0, 0])


@_checked
def test_one_offset_serves_poses_with_different_normals():
    """Each pose gets its own direction in one call, not the path's average."""
    normals = [[0.0, 0.0, 1.0],
               [0.5, 0.0, math.sqrt(0.75)],
               [0.0, -0.5, math.sqrt(0.75)],
               [-0.3, 0.2, math.sqrt(1.0 - 0.09 - 0.04)]]
    preload = 0.0008
    before = path_on_surface(normals)
    moved = apply_preload(before, preload).positions - before.positions
    want = -preload*np.array([np.asarray(n, dtype=np.float64)
                              / np.linalg.norm(n) for n in normals])
    check("four poses, four different push directions, all of length preload",
          np.allclose(moved, want, atol=1e-15), moved - want)


@_checked
def test_zero_preload_leaves_the_path_alone():
    before = path_on_surface([[0.0, 0.0, 1.0]]*3)
    after = apply_preload(before, 0.0)
    check("a zero preload moves nothing",
          np.allclose(after.positions, before.positions, atol=1e-18))


# ==============================================================================
print("")
print("THE GATE: what it accepts, and what it refuses by name")


@_checked
def test_gate_accepts_sane_values():
    ns, _ = load_preload()
    for metres in (0.0, 1e-6, 0.0003, 0.001, CEILING_M):
        check("%.4f mm accepted" % (metres*1e3,),
              ns["check_ec_preload"](metres) == float(metres))
    check("an int 0 is accepted and comes back a float",
          isinstance(ns["check_ec_preload"](0), float))


@_checked
def test_gate_refuses_negative():
    ns, _ = load_preload()
    for metres in (-1e-9, -0.0003, -0.05):
        try:
            ns["check_ec_preload"](metres)
        except ValueError as e:
            msg = str(e)
            check("%.4f mm refused" % (metres*1e3,), True)
            check("  the refusal says it is lift-off",
                  "lift-off" in msg.lower(), msg)
            check("  and names the allowed range including the ceiling",
                  ("%.2f mm" % (CEILING_M*1e3,)) in msg, msg)
        else:
            check("%.4f mm refused" % (metres*1e3,), False,
                  "a negative preload was accepted")


@_checked
def test_gate_refuses_over_the_ceiling():
    ns, _ = load_preload()
    for metres in (CEILING_M + 1e-9, 0.005, 0.01):
        try:
            ns["check_ec_preload"](metres)
        except ValueError as e:
            msg = str(e)
            check("%.2f mm refused" % (metres*1e3,), True)
            check("  the refusal names the ceiling in millimetres",
                  ("%.2f mm" % (CEILING_M*1e3,)) in msg, msg)
            check("  names the constant to change",
                  "EC_PRELOAD_CEILING_M" in msg, msg)
            check("  and says what happens past it",
                  "bottoms out" in msg and "saturates" in msg, msg)
        else:
            check("%.2f mm refused" % (metres*1e3,), False,
                  "a preload over the ceiling was accepted")


@_checked
def test_the_ceiling_is_a_millimetre_or_two():
    """A ceiling in centimetres would not be a ceiling."""
    check("EC_PRELOAD_CEILING_M is between 0.5 and 3 mm",
          0.0005 <= CEILING_M <= 0.003, CEILING_M)


@_checked
def test_gate_refuses_nonsense():
    ns, _ = load_preload()
    for value in (None, "0.3 mm", float("nan"), float("inf")):
        try:
            ns["check_ec_preload"](value)
        except ValueError as e:
            check("%r refused" % (value,), "metres" in str(e), str(e))
        else:
            check("%r refused" % (value,), False, "it was accepted")


# ==============================================================================
print("")
print("THE SETTER: one way in, metres, and it reports in millimetres")


@_checked
def test_setter_sets_and_reports():
    ns, said = load_preload()
    check("the config starts at 0", ns["EC_PRELOAD_M"] == 0.0)
    out = ns["set_ec_preload"](0.0003)
    check("set_ec_preload returns the metres it set", out == 0.0003, out)
    check("and the shared value is updated", ns["EC_PRELOAD_M"] == 0.0003)
    check("it reports the millimetre equivalent",
          "0.30 mm" in said[-1], said[-1])
    check("and not the raw metres, which would read as millimetres",
          "0.0003" not in said[-1], said[-1])
    ns["ec_preload"]()
    check("ec_preload() reads the same value back",
          "0.30 mm" in said[-1], said[-1])


@_checked
def test_setter_refuses_through_the_same_gate():
    ns, said = load_preload()
    ns["set_ec_preload"](0.0005)
    for bad in (-0.0001, CEILING_M*2):
        try:
            ns["set_ec_preload"](bad)
        except ValueError:
            check("set_ec_preload(%.4f) refused" % (bad,), True)
        else:
            check("set_ec_preload(%.4f) refused" % (bad,), False, "accepted")
    check("a refused setting leaves the previous value in force",
          ns["EC_PRELOAD_M"] == 0.0005, ns["EC_PRELOAD_M"])


@_checked
def test_the_plan_line_states_the_preload_every_time():
    ns, _ = load_preload()
    zero = ns["describe_ec_preload"](0.0)
    check("zero still prints a preload line", "preload" in zero, zero)
    check("and states 0.00 mm rather than saying nothing",
          "0.00 mm" in zero, zero)
    check("and says what zero means for mesh error",
          "1:1" in zero, zero)
    set_ = ns["describe_ec_preload"](0.0012)
    check("a nonzero preload is stated in millimetres to two places",
          "1.20 mm" in set_, set_)
    check("and names the setter", "set_ec_preload" in set_, set_)
    check("and the ceiling it was checked against",
          ("%.2f mm" % (CEILING_M*1e3,)) in set_, set_)


@_checked
def test_a_path_reports_the_preload_it_was_planned_with():
    ns, said = load_preload()

    class Planned(object):
        pass

    path = Planned()
    path.ec_preload_m = 0.0004
    # The current setting is something else entirely: the path's own value is
    # what the scan is about to command.
    ns["EC_PRELOAD_M"] = 0.0018
    out = ns["say_path_preload"](path)
    check("say_path_preload returns the path's value", out == 0.0004, out)
    check("and prints it, not the current setting",
          "0.40 mm" in said[-1] and "1.80 mm" not in said[-1], said[-1])
    out = ns["say_path_preload"](Planned())
    check("a path that records no preload reports UNKNOWN rather than guessing",
          out is None and "UNKNOWN" in said[-1], said[-1])
    check("and says how to fix it", "plan_ec_path" in said[-1], said[-1])


# ==============================================================================
print("")
print("THE CARRY: a preloaded path is still the path it was")


@_checked
def test_the_preload_keeps_everything_the_path_knows_about_itself():
    """A rebuilt path must not forget what it is.

    _offset_path_in_tool_frame() builds a new ScanPath, and a rebuild that carries
    only some attributes is silent: nothing raises, the geometry is right, and the
    stages afterwards are simply misinformed. Losing `normal_constrained` tells the
    flip check and the resampler that the path is not tied to a surface, so they
    stop consulting it; losing `normal_surface` leaves them nothing to consult even
    if they did. `probe_rotation_deg` is what a run prints as the rotation it is
    holding.

    Checked against the ONE list in toolpath_segments.dpi rather than a copy here,
    so a new attribute added there is covered without touching this file.
    """
    ns = GEOM
    path = path_on_surface([(0.0, 0.0, 1.0)]*6)
    marks = {"normal_constrained": True,
             "normal_surface": "specimen_ecplan_roi_normals.ply",
             "normal_offset": 0.0,
             "normal_report": {"waypoints": 6},
             "roll_mode": "locked",
             "roll_reference": (1.0, 0.0, 0.0),
             "orient_mode": "normal",
             "max_tilt_deg": 45.0,
             "probe_rotation_deg": 37.0,
             "serpentine": True,
             "noether_flipped": 3,
             "worst_roll_step_deg": 1.25}
    for key, value in marks.items():
        setattr(path, key, value)

    out = ns["_offset_path_in_tool_frame"](path, (0.0, 0.0, -0.0003))
    for key, value in marks.items():
        check("the preload keeps %s" % (key,),
              getattr(out, key, "<missing>") == value,
              getattr(out, key, "<missing>"))
    check("and the source records that the offset was applied",
          out.source.endswith("+tool offset"), out.source)

    # Every attribute the shared list names, not just the ones spelled out above.
    carried = _carry_list()
    missing = [a for a in carried
               if hasattr(path, a) and not hasattr(out, a)]
    check("nothing in _carry_path_attrs()'s list is dropped (%d checked)"
          % (len(carried),), not missing, missing)


@_checked
def test_a_crop_keeps_everything_too():
    """A crop removes waypoints; it does not change what the path IS.

    The work-volume crop, an ROI and the tilt limit are all one function, so a
    path that survives any of them with fewer waypoints is still the same kind of
    path.
    """
    ns = GEOM
    path = path_on_surface([(0.0, 0.0, 1.0)]*12)
    path.normal_constrained = True
    path.normal_surface = "specimen.ply"
    path.probe_rotation_deg = 90.0
    keep = np.ones(12, dtype=bool)
    keep[:4] = False          # drop a run from the start, leaving 8
    out = ns["_crop_path_by_mask"](path, keep, what="path", min_points=4,
                                   quiet=True)
    check("the crop dropped waypoints", len(out) == 8, len(out))
    for key in ("normal_constrained", "normal_surface", "probe_rotation_deg"):
        check("the crop keeps %s" % (key,),
              getattr(out, key, "<missing>") == getattr(path, key),
              getattr(out, key, "<missing>"))


def _carry_list():
    """The attribute names _carry_path_attrs() actually carries.

    Read out of toolpath_segments.dpi's source so this test cannot drift from it.
    """
    src = open(os.path.join(HERE, "toolpath_segments.dpi")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_carry_path_attrs":
            for sub in ast.walk(node):
                if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Tuple):
                    return [e.value for e in sub.iter.elts
                            if isinstance(e, ast.Constant)]
    raise AssertionError("toolpath_segments.dpi has no _carry_path_attrs() with "
                         "a tuple of attribute names")


# ==============================================================================
print("")
print("THE WIRING: plan_ec_path() uses the gate, the tool frame and the right sign")


def plan_ec_path_node():
    for node in ast.walk(dgp_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "plan_ec_path":
            return node
    raise AssertionError("ECAutomatedScan.dgp has no plan_ec_path()")


@_checked
def test_plan_ec_path_is_wired_to_the_gate():
    src = ast.unparse(plan_ec_path_node())
    check("the preload argument goes through check_ec_preload(), so a scripted "
          "plan_ec_path(preload=) cannot bypass the ceiling",
          "check_ec_preload(EC_PRELOAD_M if preload is None else preload)" in src,
          src[:200])
    check("the offset is applied in the TOOL frame",
          "_offset_path_in_tool_frame(path, (0.0, 0.0, -float(preload)))" in src)
    check("the sign is negative -- along minus tool z, into the part",
          "(0.0, 0.0, -float(preload))" in src)
    check("the plan line is printed unconditionally, zero included",
          "print(describe_ec_preload(preload))" in src)
    check("and the finished path records what it was planned with",
          "path.ec_preload_m = preload" in src)


@_checked
def test_the_preload_is_applied_before_the_crops():
    """A pose the preload pushes out of the work volume must still be cropped."""
    src = ast.unparse(plan_ec_path_node())
    offset = src.index("_offset_path_in_tool_frame")
    crop = src.index("_crop_path_to_work_volume")
    check("the tool-frame offset comes before the work-volume crop",
          offset < crop, "offset at %d, crop at %d" % (offset, crop))


@_checked
def test_run_ec_scan_states_the_preload():
    for node in ast.walk(dgp_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "run_ec_scan":
            src = ast.unparse(node)
            break
    else:
        raise AssertionError("ECAutomatedScan.dgp has no run_ec_scan()")
    check("run_ec_scan() states the path's preload before it moves",
          "say_path_preload(path)" in src)


def _main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            getattr(fn, "__wrapped__", fn)()
    print("")
    if fails:
        print("%d FAILED: %s" % (len(fails), "; ".join(fails)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
