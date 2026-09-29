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
# File:  mesh_dome_sweep_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the fixed-orientation dome raster flown to build a rough
#   planning surface.
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
"""The fixed-orientation dome raster that stage 2 flies.

Stage 2's output is a planning surface for a Noether raster over the TOP of a
thin flat plate. Two properties of the raster follow from that, and this file
covers both.

  1. ONE ORIENTATION, SO NO WRIST TRAVEL. An orbiting pan sweep has to ROLL the
     camera to face each azimuth, and that roll is wrist_3. What the orbit buys
     is views of the SIDES of the part -- which cannot be reached from this
     cell and are not part of the output. A straight-down raster at a single
     fixed orientation yields the same top surface: the orientation is the one
     the arm is already standing in, so it is inside the cable band by
     construction, and parallax comes from translating in x and y instead.

  2. THE PLAN AND THE RUN SCREEN THE SAME POSES. The dome lays its grid out for
     the CAMERA, but probe_end -- what the arm holds, and what IK is asked
     about -- hangs 77 mm off it. The reach screen therefore runs on probe_end
     poses, after the camera offset, which is the only frame in which the
     question means anything.

  A consequence of the geometry: at 530 mm NFOV_UNBINNED sees 813 x 675 mm
  against a 135 x 286 mm part, so EVERY stop frames the whole part. The raster
  is redundant views, not coverage, which makes its span negotiable in a way a
  pan sweep's directions are not: a grid whose corners do not reach is pulled
  in toward its centre rather than refusing the stage.

The region, work volume, camera transform and cable envelope here are the
ex-ur10e-1 cell's measured values. REACH is a stub -- there is no IK solver on
the host -- and refuses x >= 1.061, the boundary at which a straight-down sweep
fails over this bench.

    python3 mesh_dome_sweep_test.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mesh_vertical_sweep_test import (VOLUME, CAM_T, HOME_QUAT, PART_TOP,
                                      REACH_X_LIMIT, _load, _load_cable)

#: A detect_part() region from real hardware: a plate near the middle of the
#: cell, used here so the planner is exercised on cell-shaped numbers.
REGION_2026_09_16 = (0.7560744107266263, -0.10576199110458422,
                     0.8906979088875827, 0.18054951127101138)

CHECKS = []
FAILURES = []


def check(what, cond, detail=""):
    CHECKS.append(what)
    if cond:
        print("  ok    %s%s" % (what, (" -- " + detail) if detail else ""))
    else:
        FAILURES.append("%s%s" % (what, (" -- " + detail) if detail else ""))
        print("  FAIL  %s%s" % (what, (" -- " + detail) if detail else ""))


def _stub(ns, reach):
    """An AzureReconstruction just real enough to plan a dome."""
    AR = ns["AzureReconstruction"]

    class Stub(object):
        depth_mode = "NFOV_UNBINNED"
        sweep_aim_z = None

        def camera_offset(self):
            return CAM_T

        def _sweep_reachable(self):
            return reach

    Stub._plan_dome = AR._plan_dome
    Stub._dome_reachable_grid = AR._dome_reachable_grid
    return Stub()


def _reach_x(limit=REACH_X_LIMIT, joints=(0.0,)*5 + (math.radians(166.0),)):
    """Reachable below `limit` in x. Returns the SOLUTION, as the cell's does."""
    def ok(position, quat):
        return list(joints) if float(position[0]) < limit else None
    return ok


def _plan(ns, region, reach, **kw):
    self = _stub(ns, reach)
    xmin, ymin, xmax, ymax = region
    nx, ny = int(kw.get("x", 5)), int(kw.get("y", 5))
    kw.setdefault("length", xmax - xmin)
    kw.setdefault("width", ymax - ymin)
    kw.setdefault("origin",
                  (0.5*(xmin + xmax) - 0.5*kw["length"]*(nx - 1)/nx,
                   0.5*(ymin + ymax) - 0.5*kw["width"]*(ny - 1)/ny))
    kw.setdefault("base_quat", HOME_QUAT)
    kw.setdefault("rot", False)
    return self._plan_dome(region=region, **kw)


def main():
    cable_ns, model = _load_cable()
    ns = _load(cable=(cable_ns, model))

    print("\n1. THE RASTER IS CENTRED ON THE PART, not started from its corner")
    path = _plan(ns, REGION_2026_09_16, _reach_x())
    cam = np.asarray(path.camera_path.positions, dtype=np.float64)
    cx, cy = 0.5*(REGION_2026_09_16[0] + REGION_2026_09_16[2]), \
             0.5*(REGION_2026_09_16[1] + REGION_2026_09_16[3])
    gx, gy = 0.5*(cam[:, 0].min() + cam[:, 0].max()), \
             0.5*(cam[:, 1].min() + cam[:, 1].max())
    check("the camera grid's centre is the region's centre",
          abs(gx - cx) < 1e-9 and abs(gy - cy) < 1e-9,
          "grid (%.4f, %.4f) vs region (%.4f, %.4f)" % (gx, gy, cx, cy))

    print("\n2. ONE ORIENTATION, EVERY STOP -- so wrist_3 never moves")
    q = np.asarray(path.quats, dtype=np.float64)
    check("every stop holds the same quaternion",
          float(np.abs(q - q[0]).max()) < 1e-12,
          "max deviation %.1e" % float(np.abs(q - q[0]).max()))
    check("that quaternion is the one the arm is already in",
          float(np.abs(q[0] - np.asarray(HOME_QUAT)).max()) < 1e-12)

    print("\n3. THE SCREEN RUNS ON PROBE_END, not on the camera grid")
    # The camera sits 76.5 mm off probe_end, so a reach limit that falls
    # between the two grids must refuse probe_end even though the camera grid
    # would still fit inside it.
    tool = np.asarray(path.positions, dtype=np.float64)
    check("probe_end and the camera are NOT at the same place",
          float(np.abs(tool[:, :2] - cam[:, :2]).max()) > 0.01,
          "%.1f mm apart in xy"
          % (1e3*float(np.abs(tool[:, :2] - cam[:, :2]).max())))
    edge = float(tool[:, 0].max())
    strict = _plan(ns, REGION_2026_09_16, _reach_x(limit=edge - 1e-6))
    st = np.asarray(strict.positions, dtype=np.float64)
    check("a limit just inside probe_end's far column changes the plan",
          len(st) < len(tool) or float(st[:, 0].max()) < edge - 1e-9,
          "%d stops -> %d, far column %.4f -> %.4f"
          % (len(tool), len(st), edge, float(st[:, 0].max())))

    print("\n4. A GRID THAT DOES NOT REACH IS PULLED IN, not refused")
    # An ISLAND of reach around the middle of the part: the full-span grid's
    # corners are 150 mm from its centre, so only the centre stop survives --
    # fewer than two poses, which is not a path. Refusing there costs the whole
    # run; shrinking the grid until it fits the island costs only parallax.
    hub = np.asarray([cx - CAM_T[0], cy + CAM_T[1]], dtype=np.float64)

    def island(position, quat, r=0.06):
        d = float(np.linalg.norm(np.asarray(position[:2]) - hub))
        return (list((0.0,)*5 + (math.radians(166.0),)) if d < r else None)

    tight = _plan(ns, REGION_2026_09_16, island)
    tp = np.asarray(tight.positions, dtype=np.float64)
    check("a reach island too small for the full grid still plans",
          len(tp) >= 2, "%d stops" % len(tp))
    span = max(float(tp[:, 0].max() - tp[:, 0].min()),
               float(tp[:, 1].max() - tp[:, 1].min()))
    full = max(REGION_2026_09_16[2] - REGION_2026_09_16[0],
               REGION_2026_09_16[3] - REGION_2026_09_16[1])
    check("and it did it by SHRINKING the raster, not by dropping to nothing",
          span < full, "span %.0f mm against the region's %.0f mm"
                       % (1e3*span, 1e3*full))
    check("every stop of the shrunk raster is inside the island",
          all(island(p, None) is not None for p in tp))

    print("\n5. THE CAMERA STILL SEES THE WHOLE PART FROM EVERY KEPT STOP")
    # NFOV_UNBINNED's footprint at this standoff, as the config computes it.
    fw, fh, _mmpx = ns["depth_mode_footprint"]("NFOV_UNBINNED")
    tcam = np.asarray(tight.camera_path.positions, dtype=np.float64)
    worst = 0.0
    for c in tcam:
        for rx in (REGION_2026_09_16[0], REGION_2026_09_16[2]):
            for ry in (REGION_2026_09_16[1], REGION_2026_09_16[3]):
                worst = max(worst, abs(rx - c[0])/(0.5*fw),
                            abs(ry - c[1])/(0.5*fh))
    check("even the shrunk raster frames the whole part from every stop",
          worst <= 1.0,
          "worst corner at %.2f of the half-footprint (%.0f x %.0f mm)"
          % (worst, 1e3*fw, 1e3*fh))

    print("\n6. NOTHING REACHES -> A CLEAR REFUSAL, naming reach not cable")
    try:
        _plan(ns, REGION_2026_09_16, lambda p, q: None)
        check("an unreachable region is refused", False, "it planned anyway")
    except ValueError as e:
        msg = str(e)
        check("an unreachable region is refused", True)
        check("and the message says REACH, not cable",
              "reach" in msg and "cable" in msg and "out of cable" in msg,
              msg.splitlines()[0][:90])

    print("\n7. THE CABLE BAND IS CHECKED, at the untilted band")
    # wrist_3 parked outside the untilted band (16..301 deg for this cell).
    out = _reach_x(joints=(0.0,)*5 + (math.radians(-100.0),))
    try:
        _plan(ns, REGION_2026_09_16, out)
        check("a raster outside the cable band is refused", False,
              "it planned anyway")
    except ValueError:
        check("a raster outside the cable band is refused", True)

    print("\n8. A CELL-SHAPED REGION PLANS AS A DOME")
    ok = _plan(ns, REGION_2026_09_16, _reach_x())
    n = len(np.asarray(ok.positions))
    check("a region a pan sweep cannot orbit still plans as a dome",
          n >= 4, "%d stops, %d segments" % (n, len(ok.segment_starts)))
    z = np.asarray(ok.positions, dtype=np.float64)[:, 2]
    ceiling = VOLUME[2][2] + 0.25
    check("and probe_end stays under the motion ceiling",
          float(z.max()) < ceiling,
          "probe_end %.3f..%.3f against %.3f" % (z.min(), z.max(), ceiling))
    camz = float(np.asarray(ok.camera_path.positions, dtype=np.float64)[0, 2])
    # Measured against the plane the plan aimed at, i.e. what sweep_surface_z
    # returned. With nothing measuring the part's top, that plane is the work
    # volume floor, so a part standing proud of it sits that much closer to the
    # camera than the plan believes. That is a property of the aim plane, not
    # of this raster, and SPECIMEN_SURFACE_Z is the knob that corrects it.
    aim = ns["sweep_surface_z"](REGION_2026_09_16, quiet=True)[0]
    check("the camera holds NFOV_UNBINNED's 500 mm minimum above the aim plane",
          camz - aim >= 0.500 - 1e-9,
          "%.0f mm above the aim plane %.4f" % (1e3*(camz - aim), aim))
    check("and a part standing proud of that plane eats into the minimum, "
          "which is what SPECIMEN_SURFACE_Z is for",
          camz - PART_TOP < 0.500,
          "%.0f mm above a %.0f mm part" % (1e3*(camz - PART_TOP),
                                            1e3*(PART_TOP - aim)))

    print("\n%d checks, %d failing" % (len(CHECKS), len(FAILURES)))
    for f in FAILURES:
        print("  FAIL  %s" % f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())


def test_dome_sweep():
    assert main() == 0, FAILURES
