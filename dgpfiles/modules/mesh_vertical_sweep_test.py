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
# File:  mesh_vertical_sweep_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the vertical rung of the pan sweep against a measured cable
#   envelope and a region at the edge of the work volume.
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
"""The straight-down (vertical) rung of the pan sweep, against a MEASURED cable
envelope and a region sitting hard against the edge of the work volume.

THE SITUATION UNDER TEST. A 198 x 143 mm part near the -y edge of the volume,
with the cable envelope measured only to +/-20 degrees of tilt in y. Against
that, a pan sweep off the top LATERAL rung loses every candidate direction:

  * six of them because the tilt that rung needs -- 22 to 29 degrees -- is
    outside every tilt capture_cable_envelope() has measured. That is an absence
    of measurement, not cable strain, and a refusal must say so rather than read
    as a hard physical limit.
  * az 90 and az 270, the only two whose tilt lands inside the measurement, can
    still fail on REACH when the part sits against the edge of the volume.

One admissible direction against AZURE_PAN_MIN_DIRECTIONS = 2 is a refusal, and
that refusal raises out of plan_sweep() and takes the whole run with it.

Two independent mechanisms keep the sweep flyable, and both are covered here:

  1. THE VERTICAL RUNG (AZURE_PAN_VERTICAL). The tilt a pan direction needs is
     atan2(lateral + the region's own half-extent, height), and `lateral` is
     what the height ladder buys the slant range with. At height == slant it is
     ZERO and the only tilt left is the region's own extent -- 8 to 13 degrees
     on this part, comfortably inside the measured hull, and at the UNTILTED
     band, which is the widest the cables have.
  2. THE PURPOSE (AZURE_SWEEP_PURPOSES). A rough Azure pass in front of Noether
     wants the top of the part, and a height field IS that surface. So the
     minimum number of directions is a property of what the sweep is FOR, and
     reconstruct_rough() asks for "planning" (1) rather than the global 2.

WHAT THIS PROVES AND WHAT IT DOES NOT. Everything here is arithmetic on the
rendered tooling transform and on the measurement in the cell manifest, so it
runs on the host with numpy and nothing else. The cable envelope is REAL -- five
bands measured at the ex-ur10e-1 cell, loaded through cable_envelope.dpi's own
model -- so the tilt hull and the band widths are a cell's, not an invention.

The REACH is not real and cannot be. There is no IK solver here; the stub
refuses x >= 1.061, the boundary at which a straight-down sweep has no solution
over this bench, and accepts everything else. Whether the arm agrees at the cell
is /compute_ik's answer, and check_sweep_plan(ik=True) is how to ask it before
the recorder starts.

    python3 mesh_vertical_sweep_test.py
"""
import functools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# One cell's touched-off work volume (ex-ur10e-1).
VOLUME = ("world", (0.5596, -0.5093, 0.9248), (1.1450, 0.4539, 1.2448))
SUPPORT = VOLUME[1][2]
CROP_TOP = VOLUME[2][2]
#: What bounds the TOOL: the work volume's top plus CAPTURE_ABOVE_VOLUME. Not
#: the crop top, which is where a part may be PLACED and where the TSDF stops.
MOTION_CEILING = CROP_TOP + 0.25

#: A detected region as it comes off hardware: a 198 x 143 mm part hard against
#: the -y edge of the work volume, which is the hard case for reach.
REAL_REGION = (0.7449, -0.2853, 0.9432, -0.1419)
#: The same part in the middle of the work volume, so the edge case and the
#: comfortable case can be compared on one run.
_W, _H = REAL_REGION[2] - REAL_REGION[0], REAL_REGION[3] - REAL_REGION[1]
_CX = 0.5*(VOLUME[1][0] + VOLUME[2][0])
_CY = 0.5*(VOLUME[1][1] + VOLUME[2][1])
CENTRED_REGION = (_CX - _W/2, _CY - _H/2, _CX + _W/2, _CY + _H/2)

#: A 31 mm part on that support: the height that makes the standoff wrong.
PART_HEIGHT = 0.031
PART_TOP = SUPPORT + PART_HEIGHT

#: The reach boundary of a straight-down sweep over this bench. Not an IK
#: solver -- see the module docstring.
REACH_X_LIMIT = 1.061

# probe_end -> the depth camera's OPTICAL frame, off the rendered
# ur_cell.urdf.xacro for tooling `ec` with the `redo` Azure bracket.
CAM_T = (0.0765, -0.0005, 0.0753)
CAM_R = np.array([[0.0, -1.0, 0.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0]])

# The lopsided tooling cloud from mesh_pan_wrist_test.py: the Azure bracket out
# along +x, the body above, the eddy-current probe at the origin.
TOOLING = np.array([[0.00, 0.00, 0.000],
                    [0.12, 0.00, 0.075], [0.12, 0.05, 0.075],
                    [0.12, -0.05, 0.075], [0.00, 0.00, 0.192],
                    [0.06, 0.00, 0.192], [-0.06, 0.00, 0.100],
                    [0.00, 0.08, 0.120], [0.00, -0.08, 0.120]])

HOME_QUAT = (0.0, 0.0, 0.0, 1.0)

# ==============================================================================
# THE MEASUREMENT: one cell's capture_cable_envelope() output, as it appears in
# the cell manifest's cable_envelope.hardware block. The x row was swept to
# +/-40 but the y row stops at +/-20, so any direction needing more tilt in y
# than that is unmeasured rather than out of band.
# ==============================================================================
MEASURED = [
    (0.0, 0.0, 31.338424555636937, 301.33842455563695, "no tilt"),
    (40.0, 0.0, 87.34219315297075, 327.3421931529707, "tilt x +40 deg"),
    (-40.0, 0.0, 179.55253199567213, 254.55253199567213, "tilt x -40 deg"),
    (0.0, 20.0, 63.598971284101005, 303.598971284101, "tilt y +20 deg"),
    (0.0, -20.0, 30.28146644195578, 255.28146644195579, "tilt y -20 deg"),
]


def _load_cable():
    """cable_envelope.dpi's model, built from the measurement above."""
    src = open(os.path.join(HERE, "cable_envelope.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "mesh_vertical_sweep_test.cable"}
    exec(compile(src, "cable_envelope.dpi", "exec"), ns)
    model = ns["CableEnvelope"](
        [(tx, ty, math.radians(lo), math.radians(hi), label)
         for tx, ty, lo, hi, label in MEASURED],
        home_quat=HOME_QUAT, cell="ex-ur10e-1", mode="hardware",
        measured="2026-09-14T10:22:07", step_deg=15.0)
    return ns, model


def _load(cable=None):
    """mesh_reconstruction.dpi's geometry, with ScanPath and the quat helpers.

    The loader the other two pan tests use. `cable` wires cable_envelope.dpi's
    two entry points into the namespace under the names _cable_envelope() looks
    them up by, so the band this plans against is the measured one.
    """
    ns = {"work_volume": lambda *a, **k: VOLUME,
          "require_work_volume": lambda *a, **k: VOLUME}

    def _exec(name, drop=()):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                        for l in src.splitlines())
        for d in drop:
            src = src.replace(d, "pass  # " + d.replace("\n", " "))
        exec(compile(src, name, "exec"), ns)

    _exec("toolpath_segments.dpi",
          drop=("from geometry_msgs.msg import PoseArray, Pose",))
    src = open(os.path.join(HERE, "noether_path.dpi")).read()
    keep, taking = [], False
    for l in src.splitlines():
        if l.startswith("def quat_mul(") or l.startswith("def quat_to_matrix(") \
                or l.startswith("def matrix_to_quat("):
            taking = True
        elif taking and l and not l[0].isspace() and not l.startswith(")"):
            taking = False
        if taking:
            keep.append(l)
    exec(compile("import numpy as np\n" + "\n".join(keep), "quats", "exec"), ns)
    # The pan and dome planners report progress while they wait on IK, so the
    # helper has to be in the namespace exactly as the config includes it ahead
    # of mesh_reconstruction.dpi. Silenced: these tests are about the geometry,
    # and progress.dpi has its own test.
    _exec("progress.dpi")
    ns["PROGRESS_ENABLED"] = False
    _exec("mesh_reconstruction.dpi")
    if cable is not None:
        cns, model = cable
        ns["wrist3_band"] = lambda tilt=None, quat=None, **kw: \
            cns["wrist3_band"](tilt=tilt, quat=quat, model=model, **kw)
        ns["check_wrist3"] = lambda qs, what, quats=None, **kw: \
            cns["check_wrist3"](qs, what, quats=quats, model=model, **kw)
        ns["wrist3_tilt_of"] = cns["wrist3_tilt_of"]
    ns["_cable_envelope_warned"] = True
    return ns


fails = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


CABLE = _load_cable()
ns = _load(CABLE)
cam_tf = (CAM_T, ns["matrix_to_quat"](CAM_R))
SLANT = 0.50 + ns["AZURE_STANDOFF_MARGIN"]
LATERAL_LADDER = ns["AZURE_PAN_HEIGHTS"]


def synth_wrist(position, quat):
    """wrist_3 for a probe_end pose, as this tooling's arm would hold it.

    Exact in the roll -- on this tooling a roll about the optical axis IS a
    rotation about the tool axis, so wrist_3 moves with it degree for degree --
    plus a shoulder-pan term so two azimuths do not share an offset. The same
    stub mesh_pan_wrist_test.py uses, anchored at the middle of the untilted
    band so the arm starts somewhere legal.
    """
    twist = math.radians(ns["_twist_between"](HOME_QUAT, quat))
    pan = math.atan2(float(position[1]), float(position[0]))
    return math.radians(166.0) + twist + 0.15*pan


def reach(position, quat):
    """The reach boundary of a near-vertical sweep, expressed as a solution.

    NOT AN IK SOLVER. x >= 1.061 is where a straight-down sweep has no solution
    over this bench; everything nearer is accepted. It answers with a joint
    solution rather than a bool so the cable check has a wrist_3 to look at,
    which is what makes a direction that could only be flown out of band a
    plan-time refusal instead of a surprise at the cell.
    """
    if float(position[0]) >= REACH_X_LIMIT:
        return None
    return [0.0, 0.0, 0.0, 0.0, 0.0, synth_wrist(position, quat)]


def plan(region, surface_z, heights=None, reachable=reach, **kw):
    """One pan plan, or the ValueError it was refused with."""
    try:
        return ns["pan_sweep_path"](
            region, cam_tf, surface_z=surface_z, slant=SLANT,
            ceiling=MOTION_CEILING, tooling_points=TOOLING,
            reachable=reachable, start_quat=HOME_QUAT,
            start_wrist=math.radians(166.0), heights=heights, quiet=True, **kw)
    except ValueError as e:
        return e


# ==============================================================================


@_checked
def test_the_ladder_puts_the_vertical_rung_on_top():
    print("=== the ladder: the vertical rung is on top, and only when asked ===")
    rungs = ns["_pan_rungs"](SLANT)
    check("the top rung is the slant range itself (lateral zero)",
          abs(rungs[0] - SLANT) < 1e-12, "%s" % (rungs,))
    check("the lateral ladder is still underneath it",
          tuple(rungs[1:]) == tuple(float(h) for h in LATERAL_LADDER),
          "%s" % (rungs,))
    check("a rung that cannot hold the range at all is dropped",
          all(h <= SLANT + 1e-9 for h in ns["_pan_rungs"](0.40)),
          "%s" % (ns["_pan_rungs"](0.40),))
    check("an explicit ladder is taken exactly as given",
          ns["_pan_rungs"](SLANT, heights=(0.5, 0.4)) == [0.5, 0.4],
          "%s" % (ns["_pan_rungs"](SLANT, heights=(0.5, 0.4)),))
    ns["AZURE_PAN_VERTICAL"] = False
    try:
        check("AZURE_PAN_VERTICAL = False leaves the old ladder untouched",
              ns["_pan_rungs"](SLANT) == [float(h) for h in LATERAL_LADDER],
              "%s" % (ns["_pan_rungs"](SLANT),))
    finally:
        ns["AZURE_PAN_VERTICAL"] = True


@_checked
def test_the_tilt_each_rung_asks_for_against_the_measurement():
    print("=== the tilt each rung asks the cables for, against the MEASUREMENT ===")
    # One row per rung: the top LATERAL rung needs tilts the envelope has never
    # been measured at, and the VERTICAL rung above it does not.
    for h in ns["_pan_rungs"](SLANT):
        lat, _d = ns["pan_geometry"](SLANT, h)
        tilts, measured = [], 0
        for az in ns["AZURE_PAN_AZIMUTHS"]:
            aims, _u, _s = ns["_pan_aims"](REAL_REGION, PART_TOP, az,
                                           ns["AZURE_PAN_STOPS"])
            off = ns["_pan_offset"](REAL_REGION, az, h, lat, aims=aims)
            th = float(np.degrees(np.arctan2(np.hypot(off[0], off[1]), off[2])))
            a = np.radians(float(az))
            tilts.append(th)
            try:
                CABLE[1].band(tilt=(-th*np.sin(a), th*np.cos(a)))
                measured += 1
            except Exception:                                        # noqa: BLE001
                pass
        print("    %-9s lateral %3.0f mm, tilt %4.1f-%4.1f deg: %d of 8 directions "
              "are inside the measured hull"
              % ("VERTICAL" if lat < 1e-9 else "%.0f mm" % (1e3*h,),
                 1e3*lat, min(tilts), max(tilts), measured))
        if lat < 1e-9:
            check("every direction of the VERTICAL rung is a measured tilt",
                  measured == 8, "%d of 8" % measured)
        elif abs(h - 0.50) < 1e-9:
            # Six of eight go for want of a measurement. The two that survive
            # are az 90 and az 270, whose tilt is pure world x -- the row that
            # WAS swept to +/-40.
            check("the rung below it leaves only the two pure-x directions",
                  measured == 2, "%d of 8" % measured)


@_checked
def test_the_lateral_ladder_has_no_margin():
    print("=== the lateral ladder has no margin at all ===")
    # WHAT THE HOST CAN AND CANNOT SHOW. The cable half is exact -- the
    # measurement in the cell manifest, run through cable_envelope.dpi's own
    # model. The reach half is not: a part against the edge of the arm's reach
    # loses az 90 and az 270 on /compute_ik, and there is no IK solver here to
    # reproduce that. What IS shown is why a single reach failure is fatal:
    # after the cables, the lateral ladder is down to exactly the closed-surface
    # minimum, so losing ONE direction refuses the run.
    old = plan(REAL_REGION, PART_TOP, heights=LATERAL_LADDER)
    check("the lateral ladder plans, but on the cables alone",
          not isinstance(old, ValueError), "%s" % (old,))
    if not isinstance(old, ValueError):
        kept_az = sorted(int(a) for a, _r, _z, _t in old.pan_directions)
        print("    lateral ladder: rung %.0f mm, directions %s, margin %+d against "
              "the closed-surface minimum of %d"
              % (1e3*old.pan_height, kept_az, old.pan_direction_margin,
                 old.pan_min_directions))
        check("it keeps only the two directions the envelope was measured at",
              kept_az == [90, 270], "%s" % (kept_az,))
        check("which is the minimum exactly -- margin zero, no room for a refusal",
              old.pan_direction_margin == 0, "%+d" % old.pan_direction_margin)
        check("and the six it dropped say NO MEASUREMENT, not 'the tooling'",
              len(old.pan_dropped_why) == 6
              and all("NO MEASUREMENT" in w for _a, w in old.pan_dropped_why),
              "%s" % (old.pan_dropped_why,))

    # Now take one of those two away, as an arm at the edge of its reach does.
    _RCX = 0.5*(REAL_REGION[0] + REAL_REGION[2])
    _RCY = 0.5*(REAL_REGION[1] + REAL_REGION[3])


    def reach_minus_90(position, quat):
        """`reach`, with the far +y stand-off refused.

        NOT A SOLVER: the smallest stub that reproduces losing one of the two
        surviving directions when the part sits at the edge of the arm's reach.
        It bites only a pose more than 200 mm from the middle of the part on the
        +y side, which is what a LATERAL rung's 248 mm stand-off does and what
        the vertical rung's 72 mm never does -- so the two plans below are
        judged by the same rule.
        """
        if (float(position[1]) > _RCY
                and math.hypot(float(position[0]) - _RCX,
                               float(position[1]) - _RCY) > 0.20):
            return None
        return reach(position, quat)


    refused = plan(REAL_REGION, PART_TOP, heights=LATERAL_LADDER,
                   reachable=reach_minus_90)
    check("lose one of the two and the lateral ladder REFUSES the run",
          isinstance(refused, ValueError), "%s" % (refused,))
    if isinstance(refused, ValueError):
        print("    %s" % (str(refused).splitlines()[0][:150],))
        check("the refusal names the unmeasured tilt, not the tooling",
              "NO MEASUREMENT" in str(refused), str(refused)[:300])
        check("it points at the pre-flight rather than at the cell floor",
              "check_sweep_plan()" in str(refused), str(refused)[:400])
        check("and it offers purpose='planning' as one of the ways out",
              "purpose='planning'" in str(refused), str(refused)[:600])

    # THE SAME REACH STUB, now on the vertical rung: the check is not looser,
    # the geometry is closer in.
    recovered = plan(REAL_REGION, PART_TOP, reachable=reach_minus_90)
    check("the VERTICAL rung flies the identical case the lateral ladder refused",
          not isinstance(recovered, ValueError)
          and len(recovered.pan_directions) >= ns["AZURE_PAN_MIN_DIRECTIONS"],
          "%s" % (recovered,))


@_checked
def test_the_vertical_rung_flies_it():
    print("=== the vertical rung flies it ===")
    results = {}
    for label, region in (("AT THE EDGE", REAL_REGION),
                          ("CENTRED", CENTRED_REGION)):
        p = plan(region, PART_TOP)
        results[label] = p
        if isinstance(p, ValueError):
            print("    %-16s REFUSED: %s" % (label, str(p).splitlines()[0][:110]))
            check("%s: the vertical rung plans" % (label,), False, str(p)[:200])
            continue
        tops = np.asarray(p.positions)[:, 2]
        print("    %-16s %d directions %s, %s rung %.0f mm, tilt %.1f-%.1f deg, "
              "probe_end %.3f..%.3f (%.0f mm of headroom), wrist_3 %s"
              % (label, len(p.pan_directions),
                 [int(a) for a, _r, _z, _t in p.pan_directions],
                 "VERTICAL" if p.pan_vertical else "lateral", 1e3*p.pan_height,
                 min(t for _a, _r, _z, t in p.pan_directions),
                 max(t for _a, _r, _z, t in p.pan_directions),
                 tops.min(), tops.max(), 1e3*(MOTION_CEILING - tops.max()),
                 ["%+.0f" % w for w in p.pan_wrist_deg]))
        check("%s: it is the VERTICAL rung" % (label,), p.pan_vertical,
              "%.0f mm lateral" % (1e3*p.pan_lateral,))
        check("%s: at least the closed-surface minimum of %d directions"
              % (label, ns["AZURE_PAN_MIN_DIRECTIONS"]),
              len(p.pan_directions) >= ns["AZURE_PAN_MIN_DIRECTIONS"],
              "%d" % len(p.pan_directions))
        check("%s: probe_end stays under the motion ceiling" % (label,),
              tops.max() <= MOTION_CEILING + 1e-9, "%.4f" % tops.max())
        check("%s: every pose is inside the measured cable band" % (label,),
              all(ns["_in_wrist_band"](synth_wrist(pos, q), p.pan_wrist_band)
                  for pos, q in zip(p.positions, p.quats)),
              "band %s" % (p.pan_wrist_band,))
        check("%s: no camera pose is inside the depth mode's minimum" % (label,),
              min(np.linalg.norm(c - ns["_region_nearest"](region, PART_TOP, c))
                  for c in p.camera_positions) >= 0.50 - 1e-9,
              "%.4f" % min(np.linalg.norm(c - ns["_region_nearest"](region, PART_TOP, c))
                           for c in p.camera_positions))


@_checked
def test_the_purpose_decides_the_minimum():
    print("=== what the sweep is FOR decides the minimum, and it is never silent ===")
    need, why = ns["sweep_min_directions"]("planning")
    check("purpose='planning' asks for one direction", need == 1, "%d (%s)" % (need, why))
    need, why = ns["sweep_min_directions"]("closed")
    check("purpose='closed' holds the global minimum",
          need == ns["AZURE_PAN_MIN_DIRECTIONS"], "%d (%s)" % (need, why))
    check("no purpose at all is 'closed', so nothing changed for an old caller",
          ns["sweep_min_directions"](None)[0] == ns["AZURE_PAN_MIN_DIRECTIONS"],
          "%s" % (ns["sweep_min_directions"](None),))
    check("min_directions= still wins outright",
          ns["sweep_min_directions"]("closed", 1)[0] == 1,
          "%s" % (ns["sweep_min_directions"]("closed", 1),))
    try:
        ns["sweep_min_directions"]("rough")
        check("a purpose nobody defined is refused, listing the ones that exist",
              False, "accepted")
    except ValueError as e:
        check("a purpose nobody defined is refused, listing the ones that exist",
              "planning" in str(e) and "closed" in str(e), str(e))

    # ONE direction, and it must say what that costs. The message is the
    # deliverable: a height field handed on as though it were a closed surface
    # is the failure this prevents, and it must not be possible to get one
    # silently.
    said = []
    ns["print"] = lambda *v: said.append(" ".join(str(x) for x in v))
    try:
        one = ns["pan_sweep_path"](
            REAL_REGION, cam_tf, surface_z=PART_TOP, slant=SLANT,
            ceiling=MOTION_CEILING, tooling_points=TOOLING, reachable=reach,
            start_quat=HOME_QUAT, start_wrist=math.radians(166.0),
            azimuths=(0.0,), purpose="planning")
    finally:
        del ns["print"]
    text = "\n".join(said)
    check("a plan with fewer than the usual minimum still PLANS, not raises",
          not isinstance(one, Exception) and len(one.pan_directions) >= 1,
          "%s" % (one,))
    check("it says up front that it may be flown from fewer directions",
          "below the usual minimum" in text, text)
    check("and it names what that costs, in the words a mesh consumer needs",
          "HEIGHT FIELD" in text, text)
    check("the margin over the minimum is recorded, not merely the count",
          one.pan_direction_margin == len(one.pan_directions) - one.pan_min_directions,
          "%s vs %s" % (one.pan_direction_margin, one.pan_min_directions))
    check("and the minimum says where it came from",
          "planning" in str(one.pan_min_directions_why), one.pan_min_directions_why)


@_checked
def test_the_standoff_is_held_to_the_part():
    print("=== the standoff is held to the PART, not to the bench ===")
    z, why = ns["sweep_surface_z"](REAL_REGION, quiet=True)
    check("with nothing detected it is the work volume's floor, as before",
          abs(z - SUPPORT) < 1e-12 and "floor" in why, "%.4f %s" % (z, why))
    ns["detected_region"] = REAL_REGION
    ns["detected_info"] = {"class_name": "plate", "surface_z": PART_TOP}
    z, why = ns["sweep_surface_z"](REAL_REGION, quiet=True)
    check("the detector's measured top is used for the region it detected",
          abs(z - PART_TOP) < 1e-12, "%.4f %s" % (z, why))
    z, _w = ns["sweep_surface_z"](CENTRED_REGION, quiet=True)
    check("a STALE detection is not used for a different region",
          abs(z - SUPPORT) < 1e-12, "%.4f" % z)
    z, why = ns["sweep_surface_z"](REAL_REGION, surface_z=0.60, quiet=True)
    check("a top outside the work volume falls back to the floor, loudly",
          abs(z - SUPPORT) < 1e-12 and "outside the volume" in why,
          "%.4f %s" % (z, why))
    check("an explicit top inside the volume wins",
          abs(ns["sweep_surface_z"](REAL_REGION, surface_z=PART_TOP,
                                    quiet=True)[0] - PART_TOP) < 1e-12)
    del ns["detected_region"]
    del ns["detected_info"]

    check("sweep_height_for() takes the part's top when it is given one",
          abs(ns["sweep_height_for"]("NFOV_UNBINNED", support_z=PART_TOP)
              - (PART_TOP + 0.53)) < 1e-12)
class _Src(object):
    """Just enough AzureReconstruction for _report_standoff and the pre-flight."""
    depth_mode = "NFOV_UNBINNED"
    depth_range = (0.50, 3.86)
    depth_topic = "/azure/depth_to_rgb/image_raw"
    allow_dead_zone = False
    sweep_aim_z = None
    tf = cam_tf

    def camera_transform(self):
        return self.tf

    def sweep_tooling_points(self):
        return TOOLING

    def _sweep_start_wrist(self):
        return HOME_QUAT, math.radians(166.0)

    def _sweep_reachable(self):
        return reach




@_checked
def test_an_unaimed_camera_is_refused():
    print("=== an un-aimed camera is REFUSED, not planned around ===")
    # If the camera driver is not publishing, camera_transform() falls back to
    # identity. A plan built on identity aims the TOOL at the part instead of
    # the camera -- roughly 180 degrees from the cable envelope's home -- so
    # every rung refuses every direction and every message blames the cables.
    # The fallback must therefore be detected and named as a driver fault.
    check("the real mount is not mistaken for the fallback",
          not ns["_camera_tf_missing"](cam_tf), "%s" % (cam_tf,))
    check("identity IS the fallback: a zero translation is not a camera position",
          ns["_camera_tf_missing"](((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))))
    check("and so is identity with the quaternion written the other way",
          ns["_camera_tf_missing"](((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, -1.0))))
    unaimed = _Src()
    unaimed.tf = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    err = None
    ns["print"] = lambda *v: None
    try:
        ns["AzureReconstruction"]._plan_pan(unaimed, REAL_REGION)
    except Exception as e:                                           # noqa: BLE001
        err = e
    finally:
        del ns["print"]
    check("a pan sweep with no camera transform is refused", isinstance(err, ValueError),
          "%r" % (err,))
    check("and the refusal names the DRIVER, not the cables",
          isinstance(err, ValueError) and "Azure driver" in str(err)
          and "cannot be aimed" in str(err), "%s" % (err,))
    ns["print"] = lambda *v: None
    try:
        ok = ns["AzureReconstruction"]._plan_pan(_Src(), REAL_REGION)
    finally:
        del ns["print"]
    check("with the mount known it plans as usual",
          getattr(ok, "pan_vertical", False), "%s" % (ok,))


    # THE ONE-MILLIMETRE CASE, which is why the standoff is measured from the
    # aim plane. A dome flown at floor + 530 mm is 530 mm from the BENCH but
    # 499 mm from the top of a 31 mm part: one millimetre inside
    # NFOV_UNBINNED's minimum. Measured from the bench it passes; measured from
    # the part's top it must be refused.
    dome = ns["dome_sweep_path"](x=3, y=3, length=_W, width=_H,
                                 origin=(REAL_REGION[0], REAL_REGION[1]),
                                 height=SUPPORT + 0.53, rot=False,
                                 base_quat=HOME_QUAT)
    for aim, expect_raise in ((SUPPORT, False), (PART_TOP, True)):
        src = _Src()
        src.sweep_aim_z = aim
        err = None
        ns["print"] = lambda *v: None
        try:
            ns["AzureReconstruction"]._report_standoff(src, dome, region=REAL_REGION)
        except Exception as e:                                       # noqa: BLE001
            err = e
        finally:
            del ns["print"]
        check("a floor+530 sweep over a %s is %s"
              % ("31 mm part" if expect_raise else "bare bench",
                 "REFUSED" if expect_raise else "accepted"),
              (err is not None) == expect_raise, "%r" % (err,))


@_checked
def test_the_preflight_plans_without_moving_anything():
    print("=== the pre-flight: plan it without moving anything ===")
    # There is no motion module in this namespace, which is what proves nothing
    # here can move: `ros` does not exist, so any call that tried would raise.
    check("there is no motion module to move with", ns.get("ros") is None)
    ns["detected_region"] = REAL_REGION
    ns["detected_info"] = {"class_name": "plate", "surface_z": PART_TOP}
    said = []
    ns["print"] = lambda *v: said.append(" ".join(str(x) for x in v))
    try:
        pre = ns["check_sweep_plan"](purpose="planning", source=_Src(), ik=False,
                                     tooling_points=TOOLING, camera_tf=cam_tf)
    finally:
        del ns["print"]
        del ns["detected_region"]
        del ns["detected_info"]
    text = "\n".join(said)
    print("\n".join("    " + l for l in text.splitlines()))
    check("it plans without a region argument, from the detection", pre["ok"],
          "%s" % (pre["error"],))
    check("it takes the aim plane from the detected top",
          ("%.4f" % PART_TOP) in text, text[:400])
    check("it says where the camera is before anything else can look reassuring",
          "camera mount: probe_end ->" in text, text[:600])
    ladder_rows = [l.strip() for l in said
                   if l.strip().startswith(("VERTICAL ", "500 mm", "450 mm",
                                            "400 mm", "350 mm", "300 mm"))]
    check("it prints the whole height ladder, vertical rung first",
          len(ladder_rows) == 6 and ladder_rows[0].startswith("VERTICAL"),
          "%s" % (ladder_rows,))
    check("it names the motion ceiling as well as the crop top",
          ("%.4f" % MOTION_CEILING) in text, text)
    check("it prints a tilt and a band for every candidate direction",
          sum(1 for l in said if l.strip().startswith("az ")) == 8,
          "%d rows" % sum(1 for l in said if l.strip().startswith("az ")))
    check("and a verdict with the MARGIN, not just a pass",
          "MARGIN" in text and pre["margin"] == pre["directions"] - pre["need"],
          "%s" % (pre,))
    check("the margin over a planning minimum is positive here",
          pre["margin"] > 0, "%s" % (pre,))

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILED: %s" % ", ".join(fails))
        sys.exit(1)
    print("all vertical sweep checks passed")


_SECTIONS = ["test_the_ladder_puts_the_vertical_rung_on_top", "test_the_tilt_each_rung_asks_for_against_the_measurement", "test_the_lateral_ladder_has_no_margin", "test_the_vertical_rung_flies_it", "test_the_purpose_decides_the_minimum", "test_the_standoff_is_held_to_the_part", "test_an_unaimed_camera_is_refused", "test_the_preflight_plans_without_moving_anything"]


if __name__ == "__main__":
    sys.exit(_main())
