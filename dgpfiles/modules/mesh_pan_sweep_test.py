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
# File:  mesh_pan_sweep_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the pan sweep's geometry, with no ROS and no robot.
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
"""The pan sweep's geometry, on the host, with no ROS and no robot.

WHY A PAN SWEEP. A dome sweep buys its standoff with height alone, and the
height NFOV_UNBINNED needs over this bench -- camera 1.4548, probe_end 1.3795,
floor + 455 mm -- has no IK solution at all over the far x column. A pan sweep
holds the same camera-to-surface distance by standing off to the side instead.

Everything below is arithmetic on the rendered tooling transform, so it runs
anywhere. What it cannot check is whether the arm agrees; that is
AzureReconstruction._sweep_reachable(), which is passed in as a callable
precisely so this file can stub it.
"""
import functools
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# One cell's touched-off work volume (ex-ur10e-1).
VOLUME = ("world", (0.5596, -0.5093, 0.9248), (1.1450, 0.4539, 1.2448))
SUPPORT = VOLUME[1][2]
CEILING = VOLUME[2][2]

# probe_end -> the depth camera's OPTICAL frame, read off the rendered
# ur_cell.urdf.xacro for tooling `ec` with the `redo` Azure bracket:
# translation (0.0765, -0.0005, 0.0753) m, and the optical axes
# x = -probe_end y, y = -probe_end x, z = -probe_end z.
CAM_T = (0.0765, -0.0005, 0.0753)
CAM_R = np.array([[0.0, -1.0, 0.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0]])


def _load():
    """mesh_reconstruction.dpi's geometry, with ScanPath and the quat helpers."""
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
    # noether_path.dpi owns quat_mul/quat_to_matrix/matrix_to_quat and pulls in
    # a great deal else; take just the four helpers by name.
    src = open(os.path.join(HERE, "noether_path.dpi")).read()
    lines = src.splitlines()
    keep, taking = [], False
    for i, l in enumerate(lines):
        if l.startswith("def quat_mul(") or l.startswith("def quat_to_matrix(") \
                or l.startswith("def matrix_to_quat("):
            taking = True
        elif taking and l and not l[0].isspace() and not l.startswith(")"):
            taking = False
        if taking:
            keep.append(l)
    exec(compile("import numpy as np\n" + "\n".join(keep), "quats", "exec"), ns)
    _exec("mesh_reconstruction.dpi")
    return ns


fails = []


def check(name, ok, detail=""):
    print("  %-62s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
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


ns = _load()
cam_tf = (CAM_T, ns["matrix_to_quat"](CAM_R))
REGION = (0.80, -0.25, 0.95, -0.10)

# The fixture the sections below share. None of it asserts anything, so it is
# safe at import time -- which is what lets pytest collect the sections as
# ordinary, independent tests.
#
# The MOTION ceiling, not the crop ceiling: what bounds the tool is the work
# volume's top plus CAPTURE_ABOVE_VOLUME (0.25 m). Testing the tooling against
# the crop top instead collapses every pan sweep to 250 mm and 68 degrees.
MOTION_CEILING = CEILING + 0.25

# The aim used by the two orientation sections: one station of a sweep that
# stands 437 mm out and 300 mm up from a point on the bench.
R = ns["_optical_rotation"]((0.437, 0.0, 0.30), 0.0)
aim = np.array([0.85, -0.2, SUPPORT])
cam = aim + np.array([0.437, 0.0, 0.30])

# The plan every geometry section below is measured against.
path = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                            ceiling=MOTION_CEILING, quiet=True)


@_checked
def test_the_camera_is_aimed():
    print("=== the camera is AIMED, not merely placed ===")
    axis = R[:, 2]
    want = (aim - cam)/np.linalg.norm(aim - cam)
    check("the optical axis points from the camera at the aim point",
          np.allclose(axis, want, atol=1e-12), "%s vs %s" % (axis, want))
    check("the frame is right-handed",
          abs(np.linalg.det(R) - 1.0) < 1e-12, "det %.6f" % np.linalg.det(R))
    check("roll 0 puts image-down on the OUTWARD side (where the tooling hangs)",
          R[0, 1] > 0.5, "%s" % (R[:, 1],))


@_checked
def test_probe_end_is_placed_so_the_camera_lands_where_asked():
    print("=== probe_end is placed so that the CAMERA lands where asked ===")
    p, q = ns["_probe_end_from_camera"](R, cam, cam_tf)
    back = p + ns["quat_to_matrix"](q).dot(np.asarray(CAM_T))
    check("the camera comes back to the requested position",
          np.allclose(back, cam, atol=1e-12), "%s vs %s" % (back, cam))
    Rb = ns["quat_to_matrix"](q).dot(CAM_R)
    check("and to the requested orientation",
          np.allclose(Rb, R, atol=1e-9), "%s" % (Rb,))
    check("probe_end is NOT the camera (the 107 mm mount offset is real)",
          0.10 < np.linalg.norm(p - cam) < 0.11,
          "%.4f m" % np.linalg.norm(p - cam))


@_checked
def test_the_height_ladder():
    print("=== the height is chosen, not assumed: highest rung that works ===")
    rungs = ns["pan_height_for"](0.53, SUPPORT, MOTION_CEILING, cam_tf, REGION)
    print("    admissible rungs: %s"
          % ", ".join("%.3f m (%.0f deg)" % (h, d) for h, _l, d in rungs))
    # THE TOP RUNG IS VERTICAL: height == slant, so the lateral stand-off is
    # zero and the only tilt left is the region's own half-extent. That matters
    # because the lateral rungs need 22-29 degrees of tilt, which is outside
    # every tilt the cable envelope is measured at, while 8-13 degrees is not.
    check("the top rung is the VERTICAL one (height == slant)",
          rungs and abs(rungs[0][0] - 0.53) < 1e-9, "%s" % (rungs[:1],))
    check("its lateral stand-off is exactly zero",
          rungs and rungs[0][1] < 1e-12, "%s" % (rungs[:1],))
    check("and it is under 20 deg from vertical -- inside the measured tilts",
          rungs and rungs[0][2] < 20.0, "%s" % (rungs[:1],))
    check("the lateral rungs are still on the ladder below it",
          [abs(h - 0.50) < 1e-9 for h, _l, _d in rungs].count(True) == 1,
          "%s" % ([r[0] for r in rungs],))
    check("the ladder is ordered highest first",
          [r[0] for r in rungs] == sorted([r[0] for r in rungs], reverse=True),
          "%s" % ([r[0] for r in rungs],))
    for h, lat, _d in rungs:
        check("h=%.3f: lat^2 + h^2 = slant^2" % h,
              abs(np.hypot(lat, h) - 0.53) < 1e-9, "%.4f" % np.hypot(lat, h))
    # An explicit ladder is taken exactly as given: a caller that names its own
    # rungs has said what it wants, and the vertical rung is not inserted into it.
    lat_only = ns["pan_height_for"](0.53, SUPPORT, MOTION_CEILING, cam_tf, REGION,
                                    heights=ns["AZURE_PAN_HEIGHTS"])
    check("an explicit heights= is not given a vertical rung",
          lat_only and abs(lat_only[0][0] - 0.50) < 1e-9, "%s" % (lat_only[:1],))
    check("and that rung is the 30-45 deg one the ladder was measured for",
          lat_only and 25.0 < lat_only[0][2] < 45.0, "%s" % (lat_only[:1],))
    # AZURE_PAN_VERTICAL = False puts the ladder back exactly as it was.
    ns["AZURE_PAN_VERTICAL"] = False
    try:
        off_rungs = ns["pan_height_for"](0.53, SUPPORT, MOTION_CEILING, cam_tf,
                                         REGION)
    finally:
        ns["AZURE_PAN_VERTICAL"] = True
    check("AZURE_PAN_VERTICAL = False restores the old ladder",
          off_rungs and abs(off_rungs[0][0] - 0.50) < 1e-9, "%s" % (off_rungs[:1],))
    # A ceiling so low that probe_end cannot fit under it at any height admits none.
    check("a ceiling the tool cannot fit under admits no rung",
          not ns["pan_height_for"](0.53, SUPPORT, SUPPORT + 0.05, cam_tf, REGION),
          "admitted something")
    # And so does a tooling clearance nothing can satisfy.
    check("a tooling clearance nothing can satisfy admits no rung",
          not ns["pan_height_for"](0.53, SUPPORT, MOTION_CEILING, cam_tf, REGION,
                                   clearance=2.0),
          "admitted something")


@_checked
def test_every_pose_holds_the_slant_range():
    print("=== every pose holds the slant range to the NEAREST region point ===")
    cams = path.camera_positions
    d = np.array([np.linalg.norm(c - ns["_region_nearest"](REGION, SUPPORT, c))
                  for c in cams])
    check("no pose is inside NFOV_UNBINNED's 500 mm minimum",
          d.min() >= 0.50 - 1e-9, "nearest %.4f m" % d.min())
    check("and none is far beyond the range it was asked for",
          d.max() < 0.53 + 0.20, "farthest %.4f m" % d.max())
    check("the camera is at the top rung, 0.530 m straight above the part",
          abs(cams[:, 2].max() - SUPPORT - 0.53) < 1e-6,
          "%.3f" % (cams[:, 2].max() - SUPPORT))
    check("and the plan says it is vertical", path.pan_vertical and path.pan_lateral < 1e-12,
          "%s %.4f" % (path.pan_vertical, path.pan_lateral))
    check("probe_end never leaves the MOTION ceiling",
          path.positions[:, 2].max() <= MOTION_CEILING + 1e-9,
          "%.4f" % path.positions[:, 2].max())
    # THE HEADROOM IS THE NUMBER TO WATCH, not merely the pass/fail. A vertical
    # NFOV sweep is the highest thing this arm does, and the margin under the
    # motion envelope says whether a taller part would still fit.
    head = MOTION_CEILING - path.positions[:, 2].max()
    print("    probe_end tops out at %.3f, %.0f mm under the motion ceiling %.3f"
          % (path.positions[:, 2].max(), 1e3*head, MOTION_CEILING))
    check("with at least 50 mm of headroom under it", head > 0.050,
          "%.0f mm" % (1e3*head,))
    check("and it is above the TSDF crop top, so the tooling cannot integrate",
          path.positions[:, 2].min() > CEILING,
          "%.4f vs %.4f" % (path.positions[:, 2].min(), CEILING))
    # THE TILT IS WHAT THE CABLES PAY. Directions needing more tilt than the
    # envelope has been measured at are refused for want of a measurement, not
    # for strain, so keeping every direction under 20 degrees keeps them inside
    # the measured hull.
    check("every direction is under 20 deg from vertical",
          max(t for _a, _r, _z, t in path.pan_directions) < 20.0,
          "%s" % ([round(t, 1) for _a, _r, _z, t in path.pan_directions],))


@_checked
def test_it_covers_the_region_from_several_directions():
    print("=== it covers the region from several directions ===")
    check("four directions by default", len(path.pan_directions) == 4,
          "%d" % len(path.pan_directions))
    azs = sorted(a for a, _r, _z, _t in path.pan_directions)
    check("and they are spread, not four names for one view",
          max(azs) - min(azs) >= 90.0, "%s" % (azs,))
    check("one segment per row per direction",
          len(path.segments) == 4*ns["AZURE_PAN_STOPS"][1],
          "%d segments" % len(path.segments))


@_checked
def test_a_row_is_a_pure_translation():
    print("=== a row is a pure translation: no wrist rotation inside a pass ===")
    worst = 0.0
    for a, b in path.segments:
        for i in range(a + 1, b):
            worst = max(worst, float(np.abs(path.quats[i] - path.quats[a]).max()))
    check("orientation is constant along every row", worst < 1e-12, "%.3g" % worst)


@_checked
def test_the_tooling_hangs_away_from_the_part():
    print("=== the tooling hangs AWAY from the part ===")
    for a, roll, low, _t in path.pan_directions:
        pass
    lows = [low - SUPPORT for _a, _r, low, _t in path.pan_directions]
    check("the lowest tooling point is at least 150 mm above the support",
          min(lows) > 0.15, "%.0f mm" % (1e3*min(lows),))


@_checked
def test_unreachable_directions_are_dropped_and_too_few_is_refused():
    print("=== unreachable directions are dropped, and too few is a REFUSAL ===")
    seen = {"n": 0}


    def _half(position, quat):
        seen["n"] += 1
        return position[1] < -0.05        # refuse everything on the +y side


    # ON THE LATERAL LADDER, deliberately. A vertical sweep stands the camera
    # almost over its own aim point, so a half-plane 50 mm outside the region
    # refuses nothing and there is no dropped direction to check. The behaviour
    # under test -- a direction the arm refuses is dropped and recorded -- belongs
    # to the rungs that reach sideways, so the ladder is named rather than derived.
    path2 = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                                 ceiling=MOTION_CEILING, reachable=_half,
                                 heights=ns["AZURE_PAN_HEIGHTS"], quiet=True)
    check("the IK check was actually consulted", seen["n"] > 0, "%d calls" % seen["n"])
    check("the surviving directions are all on the reachable side",
          all(p[1] < -0.05 for p in path2.positions), "")
    check("and the ones that were dropped are recorded",
          len(path2.pan_dropped) > 0, "%s" % (path2.pan_dropped,))
    try:
        ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                             ceiling=MOTION_CEILING,
                             reachable=lambda p, q: False, quiet=True)
        check("nothing reachable is REFUSED, not flown", False, "accepted")
    except ValueError as e:
        check("nothing reachable is REFUSED, not flown",
              "refused" in str(e) and "height field" in str(e), str(e))


@_checked
def test_the_standoff_report_reads_a_pan_sweep_as_a_range():
    print("=== the standoff report reads a pan sweep as a RANGE ===")


    class _Self(object):
        depth_mode = "NFOV_UNBINNED"
        depth_range = (0.50, 3.86)
        allow_dead_zone = False


    out = []
    ns["print"] = lambda *v: out.append(" ".join(str(x) for x in v))
    err = None
    try:
        ns["AzureReconstruction"]._report_standoff(_Self(), path, region=REGION)
    except Exception as e:                                   # noqa: BLE001
        err = e
    finally:
        del ns["print"]
    text = "\n".join(out)
    check("a 300 mm-high pan sweep is ACCEPTED because it is 530 mm away",
          err is None, "%r\n%s" % (err, text))
    check("and it is reported as a range, not a height",
          "camera-to-nearest-surface range" in text, text)
    check("it says how far to the side and how oblique",
          "deg from vertical" in text and "directions" in text, text)

    short = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.40,
                                 height=0.25, ceiling=MOTION_CEILING, quiet=True)
    err = None
    ns["print"] = lambda *v: None
    try:
        ns["AzureReconstruction"]._report_standoff(_Self(), short, region=REGION)
    except Exception as e:                                   # noqa: BLE001
        err = e
    finally:
        del ns["print"]
    check("a pan sweep that would fly inside the minimum is still REFUSED",
          isinstance(err, ValueError) and "INSIDE" in str(err), "%r" % (err,))
    check("and the refusal names the pan controls, not only the dome's height",
          "SPECIMEN_SWEEP_SLANT" in str(err)
          and "SPECIMEN_SWEEP_CAMERA_HEIGHT" in str(err), str(err))


@_checked
def test_every_top_level_def_is_declared_global():
    print("=== every top-level def is declared global ===")
    # A .dpi body compiles as a FUNCTION body, so a top-level `def` that is not
    # declared global is a local of the include and raises NameError at the one
    # moment it is called -- a failure that no import and no other test reaches.
    # Checking the declarations mechanically is the only thing that catches it.
    src = open(os.path.join(HERE, "mesh_reconstruction.dpi")).read()
    declared = set(re.findall(r"^global\s+(\w+)", src, re.M))
    defined = set(re.findall(r"^def\s+(\w+)", src, re.M)) \
        | set(re.findall(r"^class\s+(\w+)", src, re.M))
    missing = sorted(defined - declared)
    check("mesh_reconstruction.dpi declares all %d of them" % len(defined),
          not missing, "undeclared: %s" % (", ".join(missing),))

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
    print("all pan sweep checks passed")


_SECTIONS = ["test_the_camera_is_aimed", "test_probe_end_is_placed_so_the_camera_lands_where_asked", "test_the_height_ladder", "test_every_pose_holds_the_slant_range", "test_it_covers_the_region_from_several_directions", "test_a_row_is_a_pure_translation", "test_the_tooling_hangs_away_from_the_part", "test_unreachable_directions_are_dropped_and_too_few_is_refused", "test_the_standoff_report_reads_a_pan_sweep_as_a_range", "test_every_top_level_def_is_declared_global"]


if __name__ == "__main__":
    sys.exit(_main())
