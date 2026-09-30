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
# File:  probe_rotation_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the operator-set rotation of the eddy-current probe about its own
#   axis, and the cable guard that refuses one the arm cannot hold.
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
"""The probe's rotation about its own axis: set once, held everywhere, refused
when the cables cannot take it.

WHY THERE IS A SETTING. The eddy-current probe is a split-D differential coil,
and a split-D is DIRECTIONAL: it answers to a crack perpendicular to its split
and is nearly blind to one parallel to it. Which way it is turned is therefore a
measurement parameter the technician chooses, and it was not one -- Noether takes
the tool x from the raster's direction of travel, so the rotation was whatever the
pass ordering happened to give.

WHAT ZERO MEANS, and this suite pins it: at 0 the split lies along
NOETHER_ROLL_REFERENCE in the PART's frame, with the probe axis vertical.
Positive turns it anticlockwise about the probe's own axis seen from outside the
part.

THE FOUR PROPERTIES UNDER TEST

  HELD CONSTANT      the same rotation at every waypoint of every pass, in both
                     serpentine directions, over a curved surface. Not "close
                     to": the "locked" roll is ONE reference orientation swung
                     onto each surface normal, and a swing adds no rotation about
                     the tool's own axis, so the spread is zero by construction
                     and is measured as zero.
  TRAVEL-BLIND       reversing a pass, or reordering the whole raster, changes it
                     by nothing at all.
  REFUSED IN TIME    a rotation whose wrist_3 angle falls outside the measured
                     cable band raises BEFORE any motion, and the message names
                     the rotation, the angle and the band.
  NEVER +/-360       the refusal is never resolved by turning the wrist a whole
                     revolution to the same probe orientation. The cables
                     physically wrap; -6.3 deg and +353.7 deg are one orientation
                     and two cable states, and the ABSOLUTE angle governs.

Runs on the host with numpy only: no ROS, no planner, no container, no dgpy. The
cell's own measured envelope is synthesised here from the numbers in its
capture file, so the band under test is the band the arm was measured at.

    python3 probe_rotation_test.py
    python3 -m pytest -q probe_rotation_test.py
"""
import ast
import functools
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")

#: The .dgp functions the probe rotation is made of.
WANT = ("ec_probe_rotation_reference", "check_ec_probe_rotation",
        "set_probe_rotation", "probe_rotation", "describe_probe_rotation",
        "say_path_probe_rotation")

fails = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so a standalone run reports every failure
    in one pass; under pytest that would be a test that passes while printing
    FAIL, so the wrapper asserts on whatever the section added.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    wrapper.__wrapped__ = fn
    return wrapper


# ==============================================================================
# The geometry, and the cable envelope, loaded the way the cell loads them
# ==============================================================================

def _load(name, ns):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    exec(compile(src, name, "exec"), ns)
    return ns


_stub = types.ModuleType("geometry_msgs.msg")
_stub.PoseArray = type("PoseArray", (),
                       {"__init__": lambda self: setattr(self, "poses", [])})
_stub.Pose = type("Pose", (), {})
_stub.PoseStamped = type("PoseStamped", (), {})
sys.modules.setdefault("geometry_msgs", types.ModuleType("geometry_msgs"))
sys.modules.setdefault("geometry_msgs.msg", _stub)

NS = {"__name__": "probe_rotation_test", "os": os, "np": np, "math": math,
      "sys": sys}
_load("toolpath_segments.dpi", NS)
_load("progress.dpi", NS)
NS["PROGRESS_ENABLED"] = False
_load("noether_path.dpi", NS)
_load("cable_envelope.dpi", NS)

ScanPath = NS["ScanPath"]
probe_rotation_reference = NS["probe_rotation_reference"]
probe_rotation_of = NS["probe_rotation_of"]
probe_rotation_wrist3_deg = NS["probe_rotation_wrist3_deg"]
check_probe_rotation = NS["check_probe_rotation"]
locked_quats_from_axes = NS["locked_quats_from_axes"]
roll_reference_quat = NS["roll_reference_quat"]
tool_twist_deg = NS["tool_twist_deg"]
tool_roll_report = NS["tool_roll_report"]
orient_segments_along = NS["orient_segments_along"]
resample_segments = NS["resample_segments"]
CableEnvelope = NS["CableEnvelope"]
CableRefused = NS["CableRefused"]

#: THE CELL'S OWN MEASUREMENT, from the untilted row of the cable-envelope file
#: capture_cable_envelope() wrote for ex-ur10e-1 (home_quat = identity, the same
#: datum as NOETHER_ROLL_REFERENCE): wrist_3 sat at 166.09 deg and the cables
#: allowed 16.09..301.09 deg there. So the rotations this cell can hold with the
#: probe vertical are -150.00..+135.00 deg, and the +/-40 deg tilt rows tighten
#: that further.
WRIST3_DATUM_DEG = 166.09270167005653
UNTILTED_LO_DEG = 16.092701670056524
UNTILTED_HI_DEG = 301.09270167005656

#: The four tilted rows, as measured. Only the numbers that matter here: each
#: tilt has its OWN band and the untilted one is not a bound on the others.
TILT_ROWS = [(-40.0, 0.0, 164.09, 254.09), (40.0, 0.0, 74.09, 341.09),
             (0.0, -40.0, 134.09, 284.09), (0.0, 40.0, 104.09, 311.09)]


def envelope():
    """The measured envelope as a CableEnvelope, with wrist3_band_deg wired to it.

    The .dpi looks the model up through cable_envelope_model(), which reads a cell
    manifest that does not exist on this host, so the module-level cache is filled
    directly. That is the same object the cell builds from its own file.
    """
    samples = [(0.0, 0.0, math.radians(UNTILTED_LO_DEG),
                math.radians(UNTILTED_HI_DEG), "no tilt")]
    for tx, ty, lo, hi in TILT_ROWS:
        samples.append((tx, ty, math.radians(lo), math.radians(hi), None))
    m = CableEnvelope(samples, home_quat=(0.0, 0.0, 0.0, 1.0),
                      cell="ex-ur10e-1", mode="hardware")
    NS["_CABLE"][0] = m
    NS["_CABLE_LOOKED"][0] = True
    return m


def no_envelope():
    """Put the module back to "this cell has never been measured"."""
    NS["_CABLE"][0] = None
    NS["_CABLE_LOOKED"][0] = True
    del NS["_CABLE_WARNED"][:]


def surface_axes(n=240, tilt_deg=25.0):
    """Tool axes over a curved surface: every tilt direction, up to `tilt_deg`."""
    out = []
    for a in np.linspace(0.0, math.radians(tilt_deg), 12):
        for b in np.linspace(0.0, 2*math.pi, n//12, endpoint=False):
            out.append([math.sin(a)*math.cos(b), math.sin(a)*math.sin(b),
                        math.cos(a)])
    return np.asarray(out)


def raster_path(rotation_deg, passes=6, per_pass=25, tilt_deg=20.0):
    """A multi-pass serpentine path built the way plan_ec_path() builds one."""
    ref = probe_rotation_reference(rotation_deg, (1.0, 0.0, 0.0))
    pos, axes, starts = [], [], []
    for p in range(passes):
        starts.append(len(pos))
        y = 0.0 + 0.003*p
        for k in range(per_pass):
            x = 0.78 + 0.002*k
            a = math.radians(tilt_deg)*math.sin(2*math.pi*k/per_pass)
            axes.append([math.sin(a), 0.0, math.cos(a)])
            pos.append([x, y, 0.930 + 0.001*math.sin(3*math.pi*k/per_pass)])
    path = ScanPath(np.asarray(pos),
                    locked_quats_from_axes(np.asarray(axes), ref), starts,
                    source="probe rotation test")
    path.roll_reference = ref
    path.roll_mode = "locked"
    return path


# ==============================================================================
# The .dgp's own setter and gate, out of the source
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


def load_dgp():
    """The rotation functions, compiled on their own. -> (namespace, printed list)

    The namespace IS their globals(), so EC_PROBE_ROTATION_DEG and the wrist datum
    are steered from here and print() is captured instead of going to stdout.
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
    ns = dict(NS)
    ns.update({"np": np,
               "print": lambda *a: said.append(" ".join(str(x) for x in a)),
               "EC_PROBE_ROTATION_DEG": dgp_constant(tree, "EC_PROBE_ROTATION_DEG"),
               "EC_PROBE_ROTATION_WRIST3_DEG":
                   dgp_constant(tree, "EC_PROBE_ROTATION_WRIST3_DEG"),
               "NOETHER_ROLL_REFERENCE":
                   dgp_constant(tree, "NOETHER_ROLL_REFERENCE")})
    mod = ast.Module(body=[nodes[w] for w in WANT], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "ECAutomatedScan.dgp", "exec"), ns)
    return ns, said


# ==============================================================================
# HELD CONSTANT
# ==============================================================================

@_checked
def test_zero_is_the_stated_datum():
    print("\nWHAT ZERO MEANS, AND THAT IT IS WHAT IS DOCUMENTED")
    q0 = probe_rotation_reference(0.0, (1.0, 0.0, 0.0))
    R = NS["quat_to_matrix"](np.asarray(q0))
    check("at 0 deg with the probe vertical, tool z is world +z and tool x is "
          "world +x (x = %s)" % (np.round(R[:, 0], 9),),
          np.allclose(R[:, 2], [0, 0, 1], atol=1e-12)
          and np.allclose(R[:, 0], [1, 0, 0], atol=1e-12),
          "this is the sentence in EC_PROBE_ROTATION_DEG's comment, as a number")
    q90 = probe_rotation_reference(90.0, (1.0, 0.0, 0.0))
    R90 = NS["quat_to_matrix"](np.asarray(q90))
    check("at +90 deg the split has turned onto world +y (x = %s), which is "
          "anticlockwise seen from above the part" % (np.round(R90[:, 0], 9),),
          np.allclose(R90[:, 0], [0, 1, 0], atol=1e-12)
          and np.allclose(R90[:, 2], [0, 0, 1], atol=1e-12))
    # And the datum is unchanged by the rotation: the tool AXIS does not move.
    for deg in (-137.0, -45.0, 0.0, 23.5, 168.0):
        Rd = NS["quat_to_matrix"](np.asarray(
            probe_rotation_reference(deg, (1.0, 0.0, 0.0))))
        if not np.allclose(Rd[:, 2], [0, 0, 1], atol=1e-12):
            check("the rotation never moves the probe AXIS (%+.1f deg)" % (deg,),
                  False, "tool z = %s" % (Rd[:, 2],))
            return
    check("and over -137 to +168 deg it never moves the probe AXIS, only the "
          "split", True)


@_checked
def test_held_constant_over_a_curved_surface():
    print("\nHELD CONSTANT: EVERY WAYPOINT, EVERY TILT DIRECTION")
    axes = surface_axes(tilt_deg=40.0)
    for deg in (0.0, 17.0, -63.0, 135.0, -150.0):
        ref = probe_rotation_reference(deg, (1.0, 0.0, 0.0))
        q = locked_quats_from_axes(axes, ref)
        got = probe_rotation_of(q, (1.0, 0.0, 0.0))
        check("set %+7.1f deg -> %+8.4f..%+8.4f deg over %d poses tilted to 40 "
              "deg in every direction (spread %.1e)"
              % (deg, got.min(), got.max(), len(axes), got.max() - got.min()),
              abs(got.min() - deg) < 1e-9 and abs(got.max() - deg) < 1e-9,
              "a swing of one reference orientation adds no rotation about the "
              "tool's own axis, so this is exact rather than close")


@_checked
def test_travel_direction_does_not_leak_in():
    print("\nTRAVEL-BLIND: PASS DIRECTION AND RASTER ORDER CHANGE IT BY NOTHING")
    deg = 42.0
    path = raster_path(deg)
    got = probe_rotation_of(path.quats, (1.0, 0.0, 0.0))
    check("%d waypoints over %d passes all hold %+.4f deg (spread %.1e)"
          % (len(path), len(path.segments), got.mean(), got.max() - got.min()),
          abs(got.mean() - deg) < 1e-9 and got.max() - got.min() < 1e-9)
    comb = orient_segments_along(path, (-1.0, 0.0, 0.0), serpentine=False)
    snake = orient_segments_along(path, (-1.0, 0.0, 0.0), serpentine=True)
    gc = probe_rotation_of(comb.quats, (1.0, 0.0, 0.0))
    gs = probe_rotation_of(snake.quats, (1.0, 0.0, 0.0))
    check("as a COMB (%d passes reversed) it is still %+.4f..%+.4f deg"
          % (comb.segments_reversed, gc.min(), gc.max()),
          abs(gc.min() - deg) < 1e-9 and abs(gc.max() - deg) < 1e-9)
    check("as a SERPENTINE (%d passes reversed) it is still %+.4f..%+.4f deg"
          % (snake.segments_reversed, gs.min(), gs.max()),
          abs(gs.min() - deg) < 1e-9 and abs(gs.max() - deg) < 1e-9,
          "a frame taken from the direction of travel flips 180 deg at every "
          "serpentine turn; this one is not taken from it")
    check("and both directions were exercised (%d and %d passes reversed)"
          % (comb.segments_reversed, snake.segments_reversed),
          comb.segments_reversed + snake.segments_reversed > 0)
    res = resample_segments(snake, 0.004, reference=snake.roll_reference)
    gr = probe_rotation_of(res.quats, (1.0, 0.0, 0.0))
    check("and the resample to 4.0 mm (%d -> %d waypoints) holds it too "
          "(%+.4f..%+.4f deg)"
          % (len(snake), len(res), gr.min(), gr.max()),
          abs(gr.min() - deg) < 1e-9 and abs(gr.max() - deg) < 1e-9)
    # tool_roll_report() is what the scan log prints; measured against the ROTATED
    # reference a locked path reads zero, which is what "locked" is supposed to
    # mean and is not the same statement as the one above.
    rr = tool_roll_report(snake.quats, snake.segments,
                          reference=snake.roll_reference)
    check("the log's roll figures read 0 against the rotated reference "
          "(%.2e deg across the path, %.2e within a pass)"
          % (rr["twist_range_deg"], rr["twist_in_pass_deg"]),
          rr["twist_range_deg"] < 1e-9 and rr["twist_in_pass_deg"] < 1e-9)


# ==============================================================================
# REFUSED BEFORE ANYTHING MOVES
# ==============================================================================

@_checked
def test_the_band_is_what_the_cell_measured():
    print("\nTHE BAND UNDER TEST IS THE ONE THE ARM WAS MEASURED AT")
    m = envelope()
    lo, hi = NS["wrist3_band_deg"](tilt=(0.0, 0.0))
    check("untilted band %+.2f..%+.2f deg" % (lo, hi),
          abs(lo - UNTILTED_LO_DEG) < 1e-6 and abs(hi - UNTILTED_HI_DEG) < 1e-6)
    check("wrist_3 sits at %+.2f deg at the zero datum, so the holdable "
          "rotations are %+.2f..%+.2f deg"
          % (WRIST3_DATUM_DEG, lo - WRIST3_DATUM_DEG, hi - WRIST3_DATUM_DEG),
          abs((lo - WRIST3_DATUM_DEG) + 150.0) < 1e-3
          and abs((hi - WRIST3_DATUM_DEG) - 135.0) < 1e-3)
    check("and the rotation lands on wrist 3 one degree for one degree "
          "(+30 deg -> %+.2f deg)"
          % (probe_rotation_wrist3_deg(30.0, WRIST3_DATUM_DEG),),
          abs(probe_rotation_wrist3_deg(30.0, WRIST3_DATUM_DEG)
              - (WRIST3_DATUM_DEG + 30.0)) < 1e-12)


@_checked
def test_a_rotation_outside_the_band_is_refused():
    print("\nREFUSED BEFORE ANY MOTION, WITH THE BAND IN THE MESSAGE")
    envelope()
    ok = check_probe_rotation(120.0, datum_deg=WRIST3_DATUM_DEG, verbose=False)
    check("+120 deg is accepted: wrist 3 at %+.1f deg, inside %+.1f..%+.1f"
          % (ok["wrist3_deg"], ok["band_deg"][0], ok["band_deg"][1]),
          ok["checked"] and ok["wrist3_deg"] is not None)
    # The refusal the real log recorded: wrist 3 at -6.3 deg against a band of
    # +16.1..+301.1, which is a rotation of -172.4 deg from this datum.
    asked = -6.3 - WRIST3_DATUM_DEG
    raised = None
    try:
        check_probe_rotation(asked, datum_deg=WRIST3_DATUM_DEG, verbose=False)
    except Exception as exc:                                      # noqa: BLE001
        raised = exc
    if raised is None:
        check("a rotation of %+.1f deg is refused" % (asked,), False,
              "it was accepted")
        return
    msg = str(raised)
    check("a rotation of %+.1f deg RAISES %s"
          % (asked, type(raised).__name__), isinstance(raised, CableRefused))
    check("the message names the rotation (%+.1f deg)" % (asked,),
          "%+.1f" % (asked,) in msg, msg[:160])
    check("the message names the wrist_3 angle it implies (-6.3 deg)",
          "-6.3" in msg, msg[:160])
    check("the message names the allowed band (+16.1..+301.1 deg)",
          "+16.1" in msg and "+301.1" in msg, msg[:160])
    check("and it names the rotations that ARE holdable (-150.0..+135.0 deg)",
          "-150.0" in msg and "+135.0" in msg, msg[:200])
    check("the message says an absolute joint angle is what governs",
          "ABSOLUTE" in msg and "360" in msg, msg[:400])


@_checked
def test_nothing_is_resolved_by_a_full_turn():
    print("\nNEVER +/-360: THE SAME ORIENTATION, A DIFFERENT CABLE STATE")
    envelope()
    asked = -6.3 - WRIST3_DATUM_DEG
    # The SAME probe orientation, one turn the other way: wrist 3 at +353.7 deg,
    # which is outside the band at the other end. Both must be refused, and the
    # refusal must not offer the other as a fix.
    for deg, where in ((asked, "-6.3 deg"), (asked + 360.0, "+353.7 deg")):
        try:
            check_probe_rotation(deg, datum_deg=WRIST3_DATUM_DEG, verbose=False)
            check("wrist 3 at %s is refused" % (where,), False, "accepted")
        except CableRefused as e:
            check("wrist 3 at %s is refused (%+.1f deg of rotation)"
                  % (where, deg), True)
            check("  and the refusal does not offer the other turn as a fix",
                  "do not add or subtract 360" in str(e).lower(), str(e)[:200])
    # AND NEITHER TURN IS REACHABLE, which is the stronger statement and the one
    # that says whose fault this is: when no branch of the angle is in band the
    # TOOLPATH is asking for a pose the cables cannot hold, and no solver choice
    # would have saved it.
    band = (math.radians(UNTILTED_LO_DEG), math.radians(UNTILTED_HI_DEG))
    alt = NS["wrist3_branch_in_band"](math.radians(-6.3), band)
    check("no turn of wrist 3 reaches -6.3 deg's orientation inside %+.1f..%+.1f "
          "deg, so this is the rotation being unholdable rather than a solver "
          "branch" % (UNTILTED_LO_DEG, UNTILTED_HI_DEG), alt is None,
          "wrist3_branch_in_band() is the diagnostic that tells the two apart, "
          "and it is never a substitution")
    # THE GUARD ITSELF CONTAINS NO 360 ARITHMETIC. Checked on the parsed source,
    # not on the text, because the refusal MESSAGE says the word 360 -- telling
    # the operator not to do it is the opposite of doing it.
    tree = ast.parse(open(os.path.join(HERE, "noether_path.dpi")).read()
                     .replace("\nglobal ", "\n# global "))
    guard = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) \
                and node.name == "check_probe_rotation":
            guard = node
    check("check_probe_rotation() is in noether_path.dpi", guard is not None)
    if guard is not None:
        nums = [n.value for n in ast.walk(guard)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)]
        check("and no number in it is a full turn (%s)"
              % (sorted({abs(float(x)) for x in nums}),),
              not any(abs(abs(float(x)) - 360.0) < 1e-9
                      or abs(abs(float(x)) - 2*math.pi) < 1e-9 for x in nums),
              "a guard that can reduce an angle into a band is not a guard")


@_checked
def test_a_rotation_can_be_holdable_at_one_end_and_not_the_other():
    print("\nPER-PASS: HOLDABLE OVER PART OF THE SPECIMEN AND NOT THE REST")
    envelope()
    # +130 deg is inside the untilted band (+296.1 against +301.1) and outside
    # the band at a +40 deg tilt about world y, where the measured hi is +311.1
    # ... so the case is built from the row that really tightens: ty = -40 has
    # hi = +284.1, and a pass that leans that way cannot hold it.
    lo0, hi0 = NS["wrist3_band_deg"](tilt=(0.0, 0.0))
    deg = 130.0
    w3 = probe_rotation_wrist3_deg(deg, WRIST3_DATUM_DEG)
    check("+%.0f deg is inside the untilted band (wrist 3 %+.1f of %+.1f..%+.1f)"
          % (deg, w3, lo0, hi0), lo0 <= w3 <= hi0)
    flat = raster_path(deg, passes=3, per_pass=12, tilt_deg=0.0)
    r = check_probe_rotation(deg, path=flat, datum_deg=WRIST3_DATUM_DEG,
                             verbose=False)
    check("and over a FLAT path every pass holds it (%d outside)"
          % (len(r["passes_outside"]),), not r["passes_outside"])
    # Now lean the later passes about world x so their band is the ty row's.
    leaned = raster_path(deg, passes=4, per_pass=12, tilt_deg=0.0)
    q = np.array(leaned.quats, copy=True)
    h = math.radians(-38.0)/2.0
    dq = (math.sin(h), 0.0, 0.0, math.cos(h))       # world x, applied on the left
    for i in range(len(q)//2, len(q)):
        q[i] = NS["quat_mul"](dq, tuple(q[i]))
    leaned = ScanPath(np.asarray(leaned.positions), q,
                      list(leaned.segment_starts), source=leaned.source)
    leaned.roll_reference = probe_rotation_reference(deg, (1.0, 0.0, 0.0))
    leaned.roll_mode = "locked"
    try:
        check_probe_rotation(deg, path=leaned, datum_deg=WRIST3_DATUM_DEG,
                             verbose=False)
        check("the leaned half of the same path refuses it", False, "accepted")
        return
    except CableRefused as e:
        msg = str(e)
    check("the leaned half of the same path REFUSES it", True)
    check("and the message says it is holdable over part of the specimen and "
          "not the rest", "part of the specimen" in msg, msg[:300])
    check("it names a pass and the band there", "pass " in msg and "deg" in msg,
          msg[:300])
    check("it still forbids the 360 fix", "360" in msg, msg[:400])


@_checked
def test_an_unmeasured_cell_says_unchecked_rather_than_passing():
    print("\nWITH NOTHING MEASURED THE CHECK SAYS SO, AND DOES NOT PASS QUIETLY")
    no_envelope()
    r = check_probe_rotation(30.0, datum_deg=WRIST3_DATUM_DEG, verbose=False)
    check("no envelope -> checked is False and a reason is given (%s)"
          % (r.get("why", "")[:60],), r["checked"] is False and r.get("why"))
    envelope()
    r2 = check_probe_rotation(30.0, datum_deg=None, verbose=False)
    check("no measured wrist datum -> checked is False and a reason is given (%s)"
          % (r2.get("why", "")[:60],), r2["checked"] is False and r2.get("why"))


# ==============================================================================
# THE CONFIG'S OWN SETTER
# ==============================================================================

@_checked
def test_the_setter_and_what_it_prints():
    print("\nSET FROM THE PROMPT, ECHOED, AND GATED THE SAME WAY")
    envelope()
    ns, said = load_dgp()
    check("the .dgp ships a rotation of %+.1f deg and a measured wrist datum of "
          "%+.2f deg" % (ns["EC_PROBE_ROTATION_DEG"],
                         ns["EC_PROBE_ROTATION_WRIST3_DEG"]),
          ns["EC_PROBE_ROTATION_DEG"] == 0.0
          and abs(ns["EC_PROBE_ROTATION_WRIST3_DEG"] - WRIST3_DATUM_DEG) < 0.01,
          "the datum is the cell's own capture, not a guess")
    got = ns["set_probe_rotation"](45.0)
    check("set_probe_rotation(45) takes effect (%.1f deg)" % (got,), got == 45.0)
    line = said[-1]
    check("and it echoes one line naming the angle and what zero means: %r"
          % (line[:96],),
          "+45.0 deg" in line and "Zero is the split along" in line
          and "wrist 3" in line)
    check("the echo states the part-frame reference (1.0, 0.0, 0.0)",
          "(1.0, 0.0, 0.0)" in line, line)
    # The same gate both ways in.
    asked = -6.3 - WRIST3_DATUM_DEG
    try:
        ns["set_probe_rotation"](asked)
        check("set_probe_rotation(%+.1f) is refused" % (asked,), False,
              "accepted")
    except CableRefused:
        check("set_probe_rotation(%+.1f) is refused by the same gate "
              "plan_ec_path() uses" % (asked,), True)
    check("and the refused value did not take effect (%.1f deg)"
          % (ns["EC_PROBE_ROTATION_DEG"],), ns["EC_PROBE_ROTATION_DEG"] == 45.0)
    for bad in ("sideways", None, float("nan"), 720.0):
        try:
            ns["check_ec_probe_rotation"](bad, verbose=False)
            check("%r is refused" % (bad,), False, "accepted")
        except (ValueError, TypeError):
            check("%r is refused as not an angle this can hold" % (bad,), True)
        except CableRefused:
            check("%r is refused by the cable band" % (bad,), True)


@_checked
def test_the_plan_reports_what_it_is_holding():
    print("\nTHE SCAN PLAN STATES THE ROTATION, AND PROVES IT FROM THE POSES")
    envelope()
    ns, said = load_dgp()
    path = raster_path(-63.0)
    path.probe_rotation_deg = -63.0
    del said[:]
    got = ns["say_path_probe_rotation"](path)
    check("it reports the rotation the PATH was planned with (%+.1f deg)"
          % (got,), got == -63.0)
    joined = "\n".join(said)
    check("and proves it by measuring the poses: %r" % (said[-1][:96],),
          "-63.000..-63.000" in joined and "spread" in joined
          and "Travel direction does not enter it" in joined)
    # A path with no record must say so rather than reporting the setting.
    bare = raster_path(0.0)
    del said[:]
    check("a path that records no rotation says UNKNOWN rather than the current "
          "setting", ns["say_path_probe_rotation"](bare) is None
          and "UNKNOWN" in said[0])


@_checked
def test_plan_ec_path_wires_it_in():
    print("\nplan_ec_path() BUILDS THE REFERENCE AND CHECKS IT BEFORE PLANNING")
    src = open(DGP).read()
    tree = ast.parse(src, filename="ECAutomatedScan.dgp")
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "plan_ec_path":
            fn = node
    if fn is None:
        raise AssertionError("ECAutomatedScan.dgp has no plan_ec_path()")
    body = "\n".join(src.splitlines()[fn.lineno-1:fn.end_lineno])
    check("it takes probe_rotation_deg=",
          "probe_rotation_deg" in [a.arg for a in fn.args.args])
    check("it runs the gate BEFORE the planner is called",
          body.index("check_ec_probe_rotation") < body.index("plan_mesh_path("))
    check("it hands the rotated reference to the planner as roll_reference",
          'kwargs.setdefault("roll_reference", ec_probe_rotation_reference('
          in body)
    check("it prints the rotation every run",
          "describe_probe_rotation(rotation)" in body)
    check("it records the rotation on the path it returns",
          "path.probe_rotation_deg = rotation" in body)
    check("and it re-checks the cables PER PASS on the finished path",
          "check_ec_probe_rotation(rotation, path=path)" in body
          and body.index("check_ec_probe_rotation(rotation, path=path)")
          > body.index("plan_mesh_path("))


@_checked
def test_the_clearance_gate_sees_the_rotated_tool():
    print("\nTHE TOOLING CLEARANCE GATE SEES THE ROTATION, NOT JUST THE AXIS")
    # Rolling the probe swings the Keyence head and the Azure bracket round the
    # tool axis, so the gate has to transform the tooling by the WHOLE pose and
    # not only by its z. It does: path_clearance() builds a full rotation matrix
    # per pose with _quats_to_matrices() and transforms every envelope point by
    # it, so the rotation reaches the gate through path.quats with no extra
    # plumbing. These checks are that fact, on the gate's own function.
    tc = {}
    src = open(os.path.join(HERE, "tooling_clearance.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_quats_to_matrices":
            exec(compile("\n".join(lines[node.lineno-1:node.end_lineno]),
                         "tooling_clearance.dpi", "exec"),
                 {"np": np, "__builtins__": __builtins__}, tc)
    check("_quats_to_matrices() is in tooling_clearance.dpi",
          "_quats_to_matrices" in tc)
    if "_quats_to_matrices" not in tc:
        return
    axes = surface_axes(n=48, tilt_deg=30.0)
    q0 = locked_quats_from_axes(axes, probe_rotation_reference(0.0, (1.0, 0.0, 0.0)))
    q90 = locked_quats_from_axes(axes, probe_rotation_reference(90.0, (1.0, 0.0, 0.0)))
    R0 = tc["_quats_to_matrices"](q0)
    R90 = tc["_quats_to_matrices"](q90)
    # A link 178 mm off the tool axis -- the Keyence head's reach on this plate.
    arm = np.array([0.178, 0.0, 0.0])
    p0 = np.einsum("nij,j->ni", R0, arm)
    p90 = np.einsum("nij,j->ni", R90, arm)
    moved = np.linalg.norm(p90 - p0, axis=1)
    check("a link 178 mm off the axis moves %.1f..%.1f mm when the probe is "
          "turned 90 deg" % (moved.min()*1e3, moved.max()*1e3),
          moved.min() > 0.2, "the gate would be checking the wrong swept volume "
          "if it read only the tool axis")
    check("while the tool AXIS is identical in both (worst %.1e deg apart)"
          % (np.degrees(np.arccos(np.clip(np.einsum(
              "ni,ni->n", R0[:, :, 2], R90[:, :, 2]), -1, 1))).max(),),
          np.allclose(R0[:, :, 2], R90[:, :, 2], atol=1e-12),
          "so nothing but the gate's full-pose transform can tell the two apart")
    one = NS["quat_to_matrix"](np.asarray(q90[3]))
    check("and it agrees with quat_to_matrix() pose by pose (%.1e)"
          % (np.abs(one - R90[3]).max(),),
          np.allclose(one, R90[3], atol=1e-12))


def main():
    for fn in (test_zero_is_the_stated_datum,
               test_held_constant_over_a_curved_surface,
               test_travel_direction_does_not_leak_in,
               test_the_band_is_what_the_cell_measured,
               test_a_rotation_outside_the_band_is_refused,
               test_nothing_is_resolved_by_a_full_turn,
               test_a_rotation_can_be_holdable_at_one_end_and_not_the_other,
               test_an_unmeasured_cell_says_unchecked_rather_than_passing,
               test_the_setter_and_what_it_prints,
               test_the_plan_reports_what_it_is_holding,
               test_plan_ec_path_wires_it_in,
               test_the_clearance_gate_sees_the_rotated_tool):
        fn.__wrapped__()
    print("\n%d check(s) failed" % (len(fails),))
    for f in fails:
        print("  FAIL %s" % (f,))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
