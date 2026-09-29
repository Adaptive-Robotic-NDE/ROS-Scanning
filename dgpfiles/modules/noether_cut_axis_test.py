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
# File:  noether_cut_axis_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests which way a Noether raster runs, and the rotation offset that
#   sets it.
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
"""Which way a Noether raster runs, pinned so the rule stops drifting.

principal_axis_noether_srv slices a mesh with parallel planes. Which axis those
planes are normal to is set by `rotation_offset`. Two things are easy to state
backwards, and both have been:

  * THE MECHANISM. With no offset the planes are normal to the FIRST principal
    axis, not the second.
  * THE CONCLUSION. With no offset the rows run along the SECOND-longest
    dimension, not the longest.

The service is not reachable from this host, so the rule is established from
three independent places and this file pins all three:

  1. THE SOURCE. ros_ws/src/noether/noether_tpp --
     PrincipalAxisDirectionGenerator::generate() returns
         AngleAxisd(rotation_offset, pca.col(2)) * pca.col(1)
     and PlaneSlicerRasterPlanner::planImpl() uses that as `cut_direction` --
     the direction the raster lines RUN -- and slices along
         cut_normal = cut_direction x mesh_normal,   mesh_normal = pca.col(2).
     pcl::PCA sorts eigenvectors by descending eigenvalue, so col(0) is the
     longest direction and col(1) the second-longest.

  2. A REAL RUN with NOETHER_ROTATION_OFFSET = pi/2 over a mesh whose principal
     extents are 313.3 / 169.7 / 27.3 mm. It produced 566 cutting planes at a
     0.30 mm line spacing -- 169.7/0.30, the SECOND-longest extent -- and all
     2355 of its acquiring passes ran within 7 degrees of the FIRST principal
     axis.

  3. A NUMPY REIMPLEMENTATION of the slice built from (1), which reproduces that
     run's output: 458806 poses against the cell's 458385, and 143.348 m of
     contour travel against 143.053.

So:

    rotation_offset = 0     lines run along the SECOND principal axis;
                            planes are normal to the FIRST and step along the
                            part's LONGEST dimension.
    rotation_offset = pi/2  lines run along the FIRST (longest) axis;
                            planes are normal to the SECOND and step along the
                            SECOND-longest dimension.

noether_cut_axes() in noether_path.dpi is the one statement of that rule, and
everything downstream -- the mesh gate's pass count, ec_plan_cost()'s
along/across swap -- goes through it rather than restating it.

Runs on the host with numpy only: no ROS, no service.

    python3 noether_cut_axis_test.py     # prints every check, exit 1 on failure
    python3 -m pytest -q noether_cut_axis_test.py

Both modes run the same checks. Standalone runs every section even after one
fails, so one run tells you everything that is wrong; under pytest each section
is its own test and the checks that failed are named in the assertion.
"""
import functools
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve on a host with no ROS. It used to be `sys.modules[...] = mod` here, which
# overwrote whatever another suite in this directory had installed -- see
# ros_msg_stubs.py for the outage that caused. ensure_msgs() only ADDS, and nothing
# in this suite builds a message: it is all numpy meshes and axes.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "def X" is a SyntaxError, so strip those lines.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "noether_cut_axis_test", "os": os, "np": np}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
ns["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)

cut_axes = ns["noether_cut_axes"]
pass_axis = ns["noether_pass_axis"]
principal = ns["noether_principal_axes"]
report = ns["mesh_plan_report"]

#: SCAN_LINE_SPACING in ECAutomatedScan.dgp -- the spacing the numbers above
#: were measured at.
LINE_SPACING = 0.0003

FAILS = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILS)
        fn(*args, **kwargs)
        failed = FAILS[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def parallel(u, w, tol=1e-6):
    """Sign-free: a raster line is a line, not a ray."""
    u = np.asarray(u, float)
    w = np.asarray(w, float)
    u = u/np.linalg.norm(u)
    w = w/np.linalg.norm(w)
    return abs(abs(float(u.dot(w))) - 1.0) <= tol


# ---------------------------------------------------------------------------
# A DELIBERATELY NON-SQUARE PLATE, so "longest" and "second-longest" are never
# the same answer and the two cases cannot be confused. 300 x 120 mm: the ratio
# is 2.5, nowhere near the 1.0 at which PCA stops having an opinion.
#
# It is also ROTATED out of the world axes, by an amount that is nobody's
# special angle, so that "the longest principal axis" cannot accidentally agree
# with "+x" and let an x/y bug pass.
LONG_M, SHORT_M = 0.300, 0.120


def rotation(axis, angle):
    a = np.asarray(axis, float)
    a = a/np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle)*K + (1.0 - math.cos(angle))*(K @ K)


def plate(long_m=LONG_M, short_m=SHORT_M, nu=201, nv=81, R=None,
          at=(0.80, 0.05, 0.94)):
    """A grid plate `long_m` along its own x and `short_m` along its own y.

    Uniform in both directions, so the PCA extents ARE the two side lengths --
    which is what lets the plane counts below be exact numbers rather than
    approximations. Returns (vertices, triangles).
    """
    u = np.linspace(-0.5*long_m, 0.5*long_m, nu)
    w = np.linspace(-0.5*short_m, 0.5*short_m, nv)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    v = np.column_stack([uu.ravel(), ww.ravel(), np.zeros(uu.size)])
    if R is not None:
        v = v @ np.asarray(R, float).T
    v = v + np.asarray(at, float)
    idx = np.arange(nu*nv).reshape(nu, nv)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    t = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return v, t


R = rotation((0.13, -0.24, 0.96), 0.7137)      # not a nice angle on purpose
V, T = plate(R=R)
LONG_DIR = R @ np.array([1.0, 0.0, 0.0])       # the 300 mm direction, in world
SHORT_DIR = R @ np.array([0.0, 1.0, 0.0])      # the 120 mm direction
THIN_DIR = R @ np.array([0.0, 0.0, 1.0])

# The three derived facts every section below shares. Nothing here asserts, so
# it is safe at import time -- which is what lets pytest collect the sections as
# ordinary, independent tests.
axes, centroid, extent = principal(V)
z = cut_axes(V, 0.0, LINE_SPACING)
h = cut_axes(V, math.pi/2, LINE_SPACING)


@_checked
def test_the_axes_themselves():
    print("--- the axes themselves -------------------------------------------")
    check(extent[0] > extent[1] > extent[2],
          "principal extents come back longest first (%.1f, %.1f, %.3f mm)"
          % tuple(extent*1e3))
    check(near(extent[0]*1e3, LONG_M*1e3, 1e-6) and near(extent[1]*1e3, SHORT_M*1e3, 1e-6),
          "...and they are the plate's own 300.0 and 120.0 mm")
    check(parallel(axes[0], LONG_DIR) and parallel(axes[1], SHORT_DIR)
          and parallel(axes[2], THIN_DIR),
          "a0/a1/a2 are the plate's long, short and thin directions -- and the "
          "plate is rotated, so this is not x/y/z falling out by luck")


@_checked
def test_offset_zero_runs_along_the_second_axis():
    print("\n--- rotation_offset = 0: the passes run along the SECOND axis ------")
    check(parallel(z["pass_direction"], SHORT_DIR),
          "the raster lines run along the SECOND-longest (120 mm) direction")
    check(parallel(z["plane_normal"], LONG_DIR),
          "...so the cutting planes are normal to the FIRST principal axis")
    check(z["pass_axis"] == 1, "pass_axis says 1 (the second axis)")
    check(near(z["pass_extent_m"], SHORT_M, 1e-9),
          "a pass is 120 mm long (%.1f)" % (z["pass_extent_m"]*1e3,))
    check(near(z["step_extent_m"], LONG_M, 1e-9),
          "and the planes have 300 mm to cover (%.1f)" % (z["step_extent_m"]*1e3,))
    check(z["planes"] == 1000,
          "which is 1000 planes at 0.30 mm (%d)" % (z["planes"],))


@_checked
def test_offset_half_pi_runs_along_the_longest():
    print("\n--- rotation_offset = pi/2: they run along the LONGEST -------------")
    check(parallel(h["pass_direction"], LONG_DIR),
          "the raster lines run along the LONGEST (300 mm) direction")
    check(parallel(h["plane_normal"], SHORT_DIR),
          "...so the cutting planes are normal to the SECOND principal axis")
    check(h["pass_axis"] == 0, "pass_axis says 0 (the longest axis)")
    check(near(h["pass_extent_m"], LONG_M, 1e-9),
          "a pass is 300 mm long (%.1f)" % (h["pass_extent_m"]*1e3,))
    check(near(h["step_extent_m"], SHORT_M, 1e-9),
          "and the planes have 120 mm to cover (%.1f)" % (h["step_extent_m"]*1e3,))
    check(h["planes"] == 400,
          "which is 400 planes at 0.30 mm (%d)" % (h["planes"],))


@_checked
def test_the_two_cases_are_each_others_opposite():
    print("\n--- the two cases really are each other's opposite -----------------")
    check(parallel(z["pass_direction"], h["plane_normal"])
          and parallel(z["plane_normal"], h["pass_direction"]),
          "pi/2 swaps the pass direction and the plane normal outright")
    check(near(h["planes"]*h["pass_extent_m"], z["planes"]*z["pass_extent_m"], 0.05),
          "the CONTOUR TRAVEL is the same either way -- %.1f m against %.1f m -- "
          "because a rectangle has the same area whichever way it is combed"
          % (h["planes"]*h["pass_extent_m"], z["planes"]*z["pass_extent_m"]))
    check(h["planes"]*2.5 == z["planes"],
          "what pi/2 buys is 2.5x FEWER passes (%d against %d), and every pass "
          "costs an approach, a lead-in and a lift. That is why the config sets it"
          % (h["planes"], z["planes"]))


@_checked
def test_the_construction_as_the_planner_writes_it():
    print("\n--- the construction, exactly as the planner writes it --------------")
    # PlaneSlicerRasterPlanner::planImpl():
    #     cut_normal = cut_direction.cross(mesh_normal),  mesh_normal = pca.col(2)
    for off in (0.0, 0.3, math.pi/4, math.pi/2, 1.9, -0.6):
        r = cut_axes(V, off)
        want = np.cross(r["pass_direction"], axes[2])
        check(parallel(r["plane_normal"], want),
              "offset %+.3f rad: plane_normal == pass_direction x a2" % (off,))
        check(abs(float(np.dot(r["pass_direction"], r["plane_normal"]))) < 1e-9,
              "  ...and the two are perpendicular")
        check(abs(float(np.dot(r["pass_direction"], axes[2]))) < 1e-9,
              "  ...and the passes stay in the surface, off the thin axis")

    # PrincipalAxisDirectionGenerator::generate():
    #     AngleAxisd(rotation_offset, pca.col(2)) * pca.col(1)
    for off in (0.0, 0.3, math.pi/4, math.pi/2, 1.9, -0.6):
        r = cut_axes(V, off)
        a1, a2 = axes[1], axes[2]
        want = math.cos(off)*a1 + math.sin(off)*np.cross(a2, a1)
        check(parallel(r["pass_direction"], want),
              "offset %+.3f rad: the pass direction is a1 turned about a2" % (off,))


@_checked
def test_the_offset_is_modulo_pi():
    print("\n--- the offset is modulo pi, because a line is not a ray ------------")
    check(pass_axis(0.0) == 1 and pass_axis(math.pi) == 1
          and pass_axis(-math.pi) == 1,
          "0, pi and -pi all mean the SECOND axis")
    check(pass_axis(math.pi/2) == 0 and pass_axis(-math.pi/2) == 0
          and pass_axis(3*math.pi/2) == 0,
          "pi/2, -pi/2 and 3pi/2 all mean the LONGEST")
    check(pass_axis(math.pi/4) == 1 and pass_axis(math.pi/3) == 0,
          "a diagonal reads as whichever axis it is nearer; exactly pi/4 ties to "
          "the second")
    check(parallel(cut_axes(V, 0.0)["pass_direction"],
                   cut_axes(V, math.pi)["pass_direction"]),
          "and cut_axes agrees: offset pi gives the same raster as offset 0")


@_checked
def test_the_real_run_reproduced_as_arithmetic():
    print("\n--- a real run, reproduced as arithmetic ----------------------------")
    # The mesh that ran has principal extents 313.3 / 169.7 mm, and the planner
    # returned 566 cutting planes at 0.30 mm with NOETHER_ROTATION_OFFSET = pi/2.
    # A uniform plate of those two side lengths has those PCA extents exactly, so
    # the plane count is checkable without the scan data.
    Vr, Tr = plate(0.3133, 0.1697, nu=314, nv=170)
    real = cut_axes(Vr, math.pi/2, LINE_SPACING)
    check(real["planes"] == 566,
          "pi/2 over a 313.3 x 169.7 mm mesh gives the run's own 566 planes (%d) "
          "-- it stepped along the SECOND-longest extent" % (real["planes"],))
    check(parallel(real["pass_direction"], [1.0, 0.0, 0.0]),
          "...with the passes along the 313.3 mm side, as all 2355 of the run's "
          "acquiring passes were")
    check(cut_axes(Vr, 0.0, LINE_SPACING)["planes"] == 1045,
          "and offset 0 would have given 1045 planes across the LONGEST extent "
          "instead (%d)" % (cut_axes(Vr, 0.0, LINE_SPACING)["planes"],))


@_checked
def test_a_square_plate_has_no_answer():
    print("\n--- a square plate has no answer, and says so -----------------------")
    sq = cut_axes(plate(0.120, 0.120, nu=81, nv=81)[0], 0.0, LINE_SPACING)
    check(sq["squareness"] < 0.01,
          "a 120 x 120 mm plate reads as square (%.4f), which is the gate's cue "
          "that the raster direction is arbitrary" % (sq["squareness"],))
    check(h["squareness"] > 0.5,
          "the 300 x 120 plate does not (%.3f)" % (h["squareness"],))


@_checked
def test_the_mesh_gate_estimate_follows_the_same_rule():
    print("\n--- and the mesh gate's estimate follows the same rule --------------")
    # This is where the convention actually bites: the number of passes and the
    # length of each one are what every cost estimate is built on, and the gate used
    # to hardcode the pi/2 case (and label it the other way round).
    r0 = report((V, None, T), line_spacing=LINE_SPACING, rotation_offset=0.0)
    rh = report((V, None, T), line_spacing=LINE_SPACING, rotation_offset=math.pi/2)
    check(r0["measured"]["estimated_passes"] == 1000
          and rh["measured"]["estimated_passes"] == 400,
          "the gate estimates 1000 passes at offset 0 and 400 at pi/2 (%d, %d)"
          % (r0["measured"]["estimated_passes"], rh["measured"]["estimated_passes"]))
    check(near(r0["measured"]["pass_extent_mm"], 120.0, 1e-3)
          and near(rh["measured"]["pass_extent_mm"], 300.0, 1e-3),
          "...over 120 mm passes and 300 mm passes respectively")
    check(r0["measured"]["pass_axis"] == 1 and rh["measured"]["pass_axis"] == 0,
          "...and it records which axis it assumed, so a report can be read back")
    check(rh["measured"]["estimated_waypoints_per_pass"]
          > r0["measured"]["estimated_waypoints_per_pass"],
          "a longer pass has more waypoints in it, which is what the "
          "waypoints_per_pass check grades")

    # The default, when nobody passes one: the config's NOETHER_ROTATION_OFFSET, and
    # the planner's own 0.0 when no config set one.
    check(report((V, None, T), line_spacing=LINE_SPACING)["rotation_offset"] == 0.0,
          "with no NOETHER_ROTATION_OFFSET in scope the gate assumes 0.0, which is "
          "principal_axis_noether's own default")
    ns["NOETHER_ROTATION_OFFSET"] = math.pi/2
    try:
        cfg = report((V, None, T), line_spacing=LINE_SPACING)
        check(cfg["rotation_offset"] == math.pi/2
              and cfg["measured"]["estimated_passes"] == 400,
              "and it follows the config's value when there is one")
    finally:
        del ns["NOETHER_ROTATION_OFFSET"]


@_checked
def test_ec_plan_cost_goes_through_the_same_helper():
    print("\n--- ec_plan_cost() goes through the same helper ---------------------")
    # It prices an area before any mesh exists, so it works from a rectangle rather
    # than from principal axes -- but WHICH SIDE the passes run along has to be the
    # same decision, and it used to be a bare truthiness test on the offset (which
    # would also have swapped for rotation_offset = pi, where nothing swaps).
    _dgp = open(os.path.join(HERE, "..", "ECAutomatedScan.dgp")).read()
    check("noether_pass_axis(globals().get(\"NOETHER_ROTATION_OFFSET\"" in _dgp,
          "ec_plan_cost() asks noether_pass_axis() which side the passes run "
          "along, rather than restating the rule")
    check(pass_axis(math.pi) == pass_axis(0.0),
          "...which matters: offset pi swaps nothing, and a truthiness test on the "
          "offset would have swapped it")


@_checked
def test_not_blind():
    print("\n--- the test is not blind ------------------------------------------")
    check(cut_axes.__code__.co_filename == "noether_path.dpi"
          and pass_axis.__code__.co_filename == "noether_path.dpi",
          "the subject came out of noether_path.dpi, not a stand-in")
    check(not any(isinstance(v, type)
                  and getattr(v, "__module__", "").startswith("geometry_msgs")
                  for v in list(globals().values())),
          "this suite builds no ROS message, so no other suite's stub can "
          "decide what it measures")
    # A KNOWN-POSITIVE AND A KNOWN-NEGATIVE through the same call: the whole rule
    # is that the two offsets give DIFFERENT answers on a non-square plate. If
    # noether_cut_axes() were not running, or were returning a constant, the two
    # would agree and every check above would still print ok.
    a = cut_axes(V, 0.0, LINE_SPACING)["pass_direction"]
    b = cut_axes(V, math.pi/2, LINE_SPACING)["pass_direction"]
    check(not parallel(a, b),
          "offset 0 and offset pi/2 really do give different pass directions "
          "(%s vs %s)" % (np.round(a, 3), np.round(b, 3)))
    check(parallel(a, SHORT_DIR) and parallel(b, LONG_DIR),
          "and each lands on the side of the plate the rule names: offset 0 "
          "along the 120 mm direction, pi/2 along the 300 mm one")


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            # __wrapped__ is the undecorated section: it records failures
            # instead of raising, so one run reports all of them.
            getattr(fn, "__wrapped__", fn)()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  FAIL %s" % (f,))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(_main())
