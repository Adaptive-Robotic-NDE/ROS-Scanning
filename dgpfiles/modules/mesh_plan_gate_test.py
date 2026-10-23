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
# File:  mesh_plan_gate_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the mesh gate in noether_path.dpi against real reconstructed
#   meshes.
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
"""Check the mesh gate in noether_path.dpi against real reconstructed meshes.

The gate exists because the planner answers success=False and puts the reason in
a service log nobody reads, which makes a failed eddy-current stage look like an
unexplained one. Every threshold the gate applies is measured against a Keyence
reconstruction chain, and this file pins those measurements down so that a later
change to the gate has to argue with the data rather than with an opinion.

Two halves:

  * synthetic cases, which always run. One defect at a time on a written plate,
    so each check can be shown to fire for its own reason and nothing else.
  * the REAL meshes, when they are present. Set NOETHER_REAL_MESH_DIR, or leave
    the default; the expected verdicts recorded in REAL below are what
    principal_axis_noether_srv itself does with those files. They are skipped,
    loudly, when the directory is not there -- these are captured scan data, not
    repository data, and they are not committed.

Runs on the host with numpy only: no ROS, no open3d, no service. A gate that
needs the planner running is a gate that cannot refuse before the planner runs.

    python3 mesh_plan_gate_test.py     # prints every check, exit 1 on failure
    python3 -m pytest -q mesh_plan_gate_test.py

Both modes run the same checks. Standalone runs every section even after one
fails, so one run tells you everything that is wrong; under pytest each section
is its own test and the checks that failed are named in the assertion. The
meshes shared between sections are written at module scope, alongside the
sections that read them; nothing written there asserts anything.
"""
import functools
import os
import sys
import tempfile
import types

import numpy as np

import test_data

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve on a host with no ROS. Assigning sys.modules[...] directly would
# overwrite whatever another suite in this directory installed; ensure_msgs()
# only ADDS. This suite builds no message: every mesh below is a written PLY or a
# numpy array.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "mesh_plan_gate_test", "os": os, "np": np}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
# noether_path.dpi reports progress through progress.dpi, which the config
# includes ahead of it; load it the same way here. Silenced, because these
# checks are about the geometry and progress.dpi has its own test.
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
ns["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)

# mesh_scan.dpi is a scan config's worth of globals; pull out the one helper
# that decides which normals plan_mesh_path() will ask the planner to use.
import ast as _ast
for _node in _ast.parse(_load("mesh_scan.dpi")).body:
    if isinstance(_node, _ast.FunctionDef) and _node.name == "_mesh_has_vertex_normals":
        exec(compile(_ast.Module(body=[_node], type_ignores=[]),
                     "mesh_scan.dpi", "exec"), ns)
has_normals = ns["_mesh_has_vertex_normals"]

report = ns["mesh_plan_report"]
plannable = ns["mesh_plannable"]
require = ns["require_plannable_mesh"]
describe = ns["describe_mesh_plan_report"]
requirements = ns["mesh_plan_requirements"]
write_plate_ply = ns["write_plate_ply"]
read_ply_mesh = ns["read_ply_mesh"]
clean_mesh = ns["clean_mesh_for_planning"]
MeshNotPlannable = ns["MeshNotPlannable"]

TMP = tempfile.mkdtemp(prefix="mesh_gate_")

#: One cell's work volume, as touch-off measured it.
WORK_VOLUME = ([0.5869960242599606, -0.4648912774417954, 0.9228375632711517],
               [1.1211007192022888, 0.4011034430451927, 1.2428375632711517])

#: The line spacing the gate's predictions below were MEASURED at. Everything
#: the gate predicts scales on it, so this stays at 0.3 mm even though
#: SCAN_LINE_SPACING is 0.25 mm now -- changing it here would invalidate the
#: recorded numbers rather than re-measure them.
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


def near_eq(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def write_ply(path, v, n, t):
    """A PLY the gate can read, with or without normals, faces or not."""
    props = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    if n is not None:
        props += [("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8")]
    vd = np.zeros(len(v), dtype=np.dtype(props))
    vd["x"], vd["y"], vd["z"] = v[:, 0], v[:, 1], v[:, 2]
    if n is not None:
        vd["nx"], vd["ny"], vd["nz"] = n[:, 0], n[:, 1], n[:, 2]
    head = ["ply", "format binary_little_endian 1.0", "element vertex %d" % len(v)]
    head += ["property double %s" % c for c, _ in props]
    if t is not None:
        head += ["element face %d" % len(t),
                 "property list uchar uint vertex_indices"]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(vd.tobytes())
        if t is not None:
            fd = np.zeros(len(t), dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]))
            fd["n"] = 3
            fd["v"] = t
            f.write(fd.tobytes())
    return path


def plate(name, nu=121, nv=121, bow=0.0):
    """A plate over ground this cell can reach, at the real specimen's place.

    121 x 121 over 120 x 120 mm is a 1.0 mm median edge -- the resolution a real
    Poisson reconstruction gives (0.98 mm), so a threshold that passes here
    passes there for the same reason.
    """
    p = os.path.join(TMP, name)
    write_plate_ply(p, (0.757, 0.030, 0.9245), (0.120, 0.0, 0.0),
                    (0.0, 0.120, 0.0), nu=nu, nv=nv, bow=bow)
    return p


def levels(rep, name):
    for c in rep["checks"]:
        if c["name"] == name:
            return c
    return None


def gate(path, **kw):
    kw.setdefault("line_spacing", LINE_SPACING)
    kw.setdefault("work_volume", WORK_VOLUME)
    return report(path, **kw)


# ---------------------------------------------------------------------------
# The clean plate every later defect below is cut from.
good = plate("good.ply")
@_checked
def test_a_clean_plate_is_plannable():
    print("--- a clean plate is plannable ------------------------------------")
    rep = gate(good)
    check(rep["verdict"] == "plan", "a clean plate at the real specimen's place plans")
    check(not rep["fatal"] and not rep["blocking"],
          "nothing fatal or blocking on a clean plate (%s)" % (rep["fatal"] + rep["blocking"],))
    check(abs(rep["measured"]["edge_mm"]["median"] - 1.0) < 0.5,
          "median edge reads about 1 mm (%.2f)" % (rep["measured"]["edge_mm"]["median"],))
    check(rep["measured"]["nonmanifold_edges"] == 0, "a written plate is manifold")
    check(rep["measured"]["component_count"] == 1, "a written plate is one component")
    check(rep["measured"]["boundary_loops"] == 1,
          "a written plate has one boundary loop -- its own rim, not a hole")
    ok, why = plannable(good, line_spacing=LINE_SPACING, work_volume=WORK_VOLUME)
    check(ok, "mesh_plannable() agrees: %s" % (why,))
    # The plate HAS analytic normals, and the shipped default throws them away.
    c = levels(rep, "normal_source")
    check(c is not None and not c["ok"] and c["level"] == "warn",
          "a mesh with its own normals warns that estimate_normals=True discards them")
    check(levels(gate(good, estimate_normals=False), "normal_source")["ok"],
          "...and planning from the mesh's own normals is not a warning")

# ---------------------------------------------------------------------------
# The one-defect meshes a later section reads too: a NaN vertex (fatal, and an
# abort inside the planner), and a PLY with points and no faces -- which is what
# most of the files a Keyence reconstruction chain writes actually are.
v, n, t = read_ply_mesh(good)
vn = v.copy()
vn[17] = np.nan
nanmesh = write_ply(os.path.join(TMP, "nan.ply"), vn, n, t)
cloud = write_ply(os.path.join(TMP, "cloud.ply"), v, None, None)
@_checked
def test_one_defect_at_a_time():
    print("\n--- one defect at a time ------------------------------------------")
    # NaN. THE ONE THAT KILLS THE SERVICE, not just the plan: PCL's KdTreeFLANN
    # asserts on a non-finite point inside normal estimation and aborts
    # principal_axis_noether_srv, so every later plan in the session reports "no
    # principal_axis_noether service" and points at the wrong thing.
    rep = gate(nanmesh)
    check("finite" in rep["fatal"], "one NaN vertex is fatal")
    check(rep["measured"]["nonfinite_vertices"] == 1, "and it is counted (1)")
    check(rep["verdict"] == "repair", "a NaN is repairable, not a dead end")
    # Non-manifold: a third triangle on an edge that already has two. Pick the edge
    # by counting, not by index: the first triangle's first edge is on the rim of a
    # grid plate and has only one face, so hanging a triangle off it makes a
    # perfectly manifold fin.
    _e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _k = _e[:, 0]*len(v) + _e[:, 1]
    _uk, _idx, _cnt = np.unique(_k, return_index=True, return_counts=True)
    _interior = _e[_idx[_cnt == 2][0]]
    t3 = np.vstack([t, [[_interior[0], _interior[1], len(v) - 1]]])
    nm = write_ply(os.path.join(TMP, "nonmanifold.ply"), v, n, t3)
    rep = gate(nm)
    check("manifold" in rep["fatal"],
          "an edge shared by three triangles is fatal (the planner's 'Too many "
          "connecting vertices')")
    check(rep["measured"]["nonmanifold_edges"] >= 1, "and it is counted")
    # A crumb: a detached triangle far too small to be part of the specimen.
    vc = np.vstack([v, v[:3] + np.array([0.0, 0.0, 0.004])])
    nc = np.vstack([n, n[:3]])
    tc = np.vstack([t, [[len(v), len(v) + 1, len(v) + 2]]])
    crumb = write_ply(os.path.join(TMP, "crumb.ply"), vc, nc, tc)
    rep = gate(crumb)
    check("components" in rep["fatal"],
          "a detached crumb is fatal ('Connecting vertex does not exist')")
    # Zero area.
    vz = v.copy()
    vz[t[5, 1]] = vz[t[5, 0]]
    zero = write_ply(os.path.join(TMP, "zeroarea.ply"), vz, n, t)
    rep = gate(zero)
    check("degenerate_faces" in rep["fatal"], "a zero-area face is fatal")
    # A point cloud. Most of the .ply files a Keyence reconstruction chain writes
    # are point clouds, and _mesh_vertices() reads every one of them happily.
    rep = gate(cloud)
    check(rep["verdict"] == "refuse" and "readable" in rep["fatal"],
          "a PLY with points and no faces is refused as a point cloud")
    check("POINT CLOUD" in levels(rep, "readable")["why"],
          "...and the reason says so rather than 'could not read'")
    # Too coarse. A WARNING, not a refusal, and the distinction is measured: the
    # planner refuses quadric-decimated meshes at 2.68 and 3.73 mm median edge
    # but accepts 6.00 mm marching-cubes Azure meshes. So coarseness alone is not
    # what it chokes on -- the decimator's thin triangles are; coarseness costs
    # pose quality.
    coarse = plate("coarse.ply", nu=25, nv=25)
    rep = gate(coarse)
    check("density" in rep["warn"],
          "a 5 mm median edge warns (%.2f mm)" % (rep["measured"]["edge_mm"]["median"],))
    check("density" not in rep["blocking"] and rep["verdict"] != "refuse",
          "...and does NOT refuse: a 6 mm Azure mesh plans, so a gate that refused "
          "one here would be refusing what the planner accepts")
    check(gate(plate("fine.ply", nu=81, nv=81))["verdict"] == "plan",
          "a 1.5 mm median edge plans with nothing to say")
    # Out of the cell. A mesh authored about its own origin lands in the floor.
    vo = v - np.array([0.757, 0.030, 0.9245])
    origin_mesh = write_ply(os.path.join(TMP, "atorigin.ply"), vo, n, t)
    rep = gate(origin_mesh)
    check("in_work_volume" in rep["fatal"],
          "a mesh authored about its own origin is refused before planning")

# ---------------------------------------------------------------------------
# The 12 mm skirt: the Poisson reconstruction wrapping round the plate edge.
vs = v.copy()
edge = ((np.abs(vs[:, 0] - vs[:, 0].min()) < 0.006) |
        (np.abs(vs[:, 0] - vs[:, 0].max()) < 0.006))
vs[edge, 2] -= 0.012
skirt = write_ply(os.path.join(TMP, "skirt.ply"), vs, None, t)
@_checked
def test_the_shape_checks_cost_coverage_not_a_refusal():
    print("\n--- the shape checks, which cost coverage rather than refusing -----")
    # A skirt: a Poisson reconstruction wraps round the plate's 12 mm edge and
    # those faces stand near vertical. On a real mesh that is about 6% of the area
    # past the 45 deg cap, costing a similar fraction of the waypoints.
    rep = gate(skirt)
    c = levels(rep, "tilt_loss")
    check(c["value"] > 0.05,
          "a 12 mm skirt puts %.1f%% of the area past the 45 deg cap" % (c["value"]*100.0,))
    check(c["level"] == "warn" and rep["verdict"] != "refuse",
          "...which is lost coverage, not a refusal -- the planner plans Azure "
          "meshes with 8-45% of their area there")
    check(gate(good)["measured"]["tilt_area_fraction_over_cap"] < 1e-6,
          "a flat plate has no area past the cap")
    check(levels(gate(skirt, max_tilt_deg=89.0), "tilt_loss")["ok"],
          "...and raising the cap is a way out, which the gate reports rather than hides")
    # Holes fragment, they do not refuse. Measured on a real mesh: eight 2 mm
    # holes take the path from 354 passes to 444 (+25%) for 1.1% of the travel.
    cent = v[t].mean(axis=1)
    mid = np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()])
    holed = write_ply(os.path.join(TMP, "holed.ply"), v, n,
                      t[np.linalg.norm(cent - mid, axis=1) > 0.012])
    rep = gate(holed)
    check(rep["measured"]["boundary_loops"] == 2,
          "a punched hole shows as a second boundary loop")
    check(rep["measured"]["hole_extra_passes"] > 100,
          "a 24 mm hole splits about %.0f extra passes at a 0.3 mm line spacing"
          % (rep["measured"]["hole_extra_passes"],))
    check(levels(rep, "holes")["level"] == "warn",
          "...it trips the fragmentation warning (+%.0f%% passes)"
          % (rep["measured"]["hole_fragmentation"]*100.0,))
    check(rep["verdict"] != "refuse",
          "...and it is only a warning: the planner DOES plan a holed mesh, it just "
          "cuts every pass that crosses the hole in two")
    # Region coverage: a plan can only be flown where there is surface -- but
    # short coverage is a WARNING, never a block. A scan is not worth refusing
    # for missing a coverage figure, and the shortfall is usually the profiler's
    # own lateral offset at the part's edge (about 73 mm here), which no replan
    # fixes: a 200 x 200 mm plate measures 187.3 x 199.1, which is 93% of the
    # PART and 86% of the region.
    rep = gate(good, region=(0.757, 0.030, 0.957, 0.230))
    check(levels(rep, "region_coverage")["level"] == "warn"
          and "region_coverage" not in rep["blocking"] + rep["fatal"],
          "a region twice the size of the mesh WARNS and does not block "
          "(%.0f%% covered)"
          % (rep["measured"]["region_coverage"]*100.0,))
    check(not levels(rep, "region_coverage")["ok"],
          "...and the warning is tripped, not merely present")
    check(rep["verdict"] != "refuse",
          "...so the plan is not refused for coverage alone")
    rep = gate(good, region=(0.780, 0.050, 0.850, 0.120))
    check(rep["verdict"] == "plan", "a region inside the mesh plans")
    # ---------------------------------------------------------------------------
@_checked
def test_repair_and_the_refusal_that_carries_its_reason():
    print("\n--- repair, and the refusal that carries its reason ----------------")
    fixed = os.path.join(TMP, "nan_fixed.ply")
    out = require(nanmesh, repair_to=fixed, verbose=False,
                  line_spacing=LINE_SPACING, work_volume=WORK_VOLUME)
    check(out == fixed, "require_plannable_mesh() repairs a NaN mesh and returns the copy")
    check(gate(fixed)["verdict"] == "plan", "the repaired copy plans")
    check(read_ply_mesh(fixed)[0].shape[0] < v.shape[0],
          "and the repair actually removed something")
    try:
        require(cloud, verbose=False, line_spacing=LINE_SPACING)
        check(False, "a point cloud raises MeshNotPlannable")
    except MeshNotPlannable as exc:
        check("POINT CLOUD" in str(exc),
              "a point cloud raises MeshNotPlannable, naming the reason")
    except Exception as exc:
        check(False, "a point cloud raises MeshNotPlannable, not %r" % (exc,))
    # ...and it still does with an ROI in hand, which reads the mesh one step
    # earlier. A ValueError out of the crop would replace that diagnosis with a
    # worse one.
    try:
        require(cloud, roi=(0.78, 0.05, 0.85, 0.12), verbose=False,
                line_spacing=LINE_SPACING)
        check(False, "a point cloud with an ROI raises MeshNotPlannable")
    except MeshNotPlannable as exc:
        check("POINT CLOUD" in str(exc),
              "a point cloud with an ROI still names the point cloud, not the ROI")
    except Exception as exc:
        check(False, "a point cloud with an ROI raises MeshNotPlannable, not %r"
              % (exc,))
    try:
        require(os.path.join(TMP, "nothere.ply"), roi=(0.78, 0.05, 0.85, 0.12),
                verbose=False, line_spacing=LINE_SPACING)
        check(False, "a missing file with an ROI raises MeshNotPlannable")
    except MeshNotPlannable as exc:
        check("no such file" in str(exc),
              "a missing file with an ROI does too")
    except Exception as exc:
        check(False, "a missing file with an ROI raises MeshNotPlannable, not %r"
              % (exc,))
    # A surface standing on its edge: nothing is left after the tilt crop, and no
    # amount of cleaning changes that.
    _wall = v.copy()
    _wall[:, 2] = 0.9245 + 2.5*(_wall[:, 1] - _wall[:, 1].min())
    wall = write_ply(os.path.join(TMP, "wall.ply"), _wall, None, t)
    rep = gate(wall)
    check("tilt_loss" in rep["blocking"],
          "a surface leaning %.0f deg over %.0f%% of its area is blocked"
          % (rep["measured"]["tilt_deg"]["p50"],
             rep["measured"]["tilt_area_fraction_over_cap"]*100.0))
    try:
        require(wall, repair_to=os.path.join(TMP, "wall_fixed.ply"), verbose=False,
                line_spacing=LINE_SPACING, work_volume=WORK_VOLUME)
        check(False, "a wall raises rather than being repaired")
    except MeshNotPlannable as exc:
        check("leans more than" in str(exc),
              "a wall raises: cleaning does not make a flank face upward")
    # estimate_normals=False needs normals in the file, and the planner says so only
    # in its own log.
    rep = gate(skirt, estimate_normals=False)
    check("normal_source" in rep["fatal"],
          "estimate_normals=False on a mesh with no vertex normals is fatal")

@_checked
def test_which_normals_plan_mesh_path_will_use():
    print("\n--- which normals plan_mesh_path() will use ------------------------")
    # plan_mesh_path() now leaves estimate_normals=None meaning "the mesh's own if it
    # has any", decided from the header alone. It has to agree with read_ply_mesh()
    # on every real file or the A/B in its comment does not apply to what runs.
    check(has_normals(good), "a written plate carries vertex normals")
    check(not has_normals(cloud), "a point cloud does not")
    check(not has_normals(os.path.join(TMP, "nothere.ply")), "and neither does a missing file")
    _nn = write_ply(os.path.join(TMP, "nonormals.ply"), v, None, t)
    check(not has_normals(_nn), "a mesh written without them reads as having none")
    check(read_ply_mesh(_nn)[1] is None,
          "...and read_ply_mesh() agrees, which is what the planner will see")

@_checked
def test_requirements_stated_before_a_reconstruction():
    print("\n--- requirements, stated before a reconstruction exists ------------")
    req = requirements(LINE_SPACING, 45.0)
    check(req["median_edge_m"] <= 0.002, "requirements name a median edge <= 2 mm")
    check(req["voxel_m"] <= 0.001, "...and a TSDF voxel at half of it")
    check(req["nonfinite_vertices"] == 0 and req["nonmanifold_edges"] == 0,
          "...and zero for the two defects that stop the planner dead")
    txt = ns["describe_mesh_plan_requirements"](LINE_SPACING, 45.0)
    check("%%" not in txt, "the requirements text is not double-escaped")

# ---------------------------------------------------------------------------
# WHICH SURFACE THE QUALITY CHECKS MEASURE
# ---------------------------------------------------------------------------
# Roughness and tilt measured over the WHOLE mesh grade the part AND everything
# reconstructed with it: the optical breadboard it is bolted to, the clamps, a
# stepped mount block. Two scans of one plate that differ only in surface
# preparation (Kapton tape against sublimating spray) measure:
#
#                                    kapton     spray    ratio
#   normal turn p99, whole mesh       55.5 deg  52.6 deg  1.05
#   normal turn p99, EC_SCAN_REGION   53.8 deg  48.7 deg  1.10
#   normal turn p99, the plate alone  10.5 deg   4.5 deg  2.33
#
# A statistic that separates a bad scan from a good one by 6% cannot grade a
# surface preparation. A detected rectangle barely helps, because it can itself
# be far larger than the part (135 x 287 mm for an 83 x 165 mm plate). An
# operator-drawn ROI is what works.
#
# The mesh below is that situation in miniature: one connected 200 mm plate whose
# central 60 mm is flat part and whose surround is 1 mm-pitch rubble standing in
# for the breadboard, its holes and the clamps. RandomState, not default_rng,
# because its stream is frozen across numpy versions and these numbers are
# assertions.
@_checked
def test_the_gate_measures_the_surface_it_is_told_to():
    print("\n--- the gate measures the surface it is told to ---------------------")
    _N, _SIDE = 201, 0.200
    _PART = 0.060                      # the flat square in the middle
    _gx = np.linspace(0.0, _SIDE, _N)
    _X, _Y = np.meshgrid(_gx, _gx, indexing="ij")
    _flat = (np.abs(_X - 0.5*_SIDE) < 0.5*_PART) & (np.abs(_Y - 0.5*_SIDE) < 0.5*_PART)
    _Z = np.where(_flat, 0.0,
                  np.random.RandomState(12345).uniform(-0.001, 0.001, (_N, _N)))
    _bv = np.column_stack([(_X + 0.757).ravel(), (_Y - 0.070).ravel(),
                           (_Z + 0.9245).ravel()])
    _i = np.arange(_N*_N).reshape(_N, _N)
    _bt = np.vstack([
        np.column_stack([_i[:-1, :-1].ravel(), _i[1:, :-1].ravel(), _i[1:, 1:].ravel()]),
        np.column_stack([_i[:-1, :-1].ravel(), _i[1:, 1:].ravel(), _i[:-1, 1:].ravel()])])
    bench = write_ply(os.path.join(TMP, "part_on_bench.ply"), _bv, None, _bt)
    # The ROI: the flat middle, with 4 mm of margin so the step at its rim is not
    # measured as part surface. As roi_select() would hand it over -- a region
    # tuple, a vertex mask and a 3D bounds box.
    _PX0, _PX1 = 0.757 + 0.5*(_SIDE - _PART) + 0.004, 0.757 + 0.5*(_SIDE + _PART) - 0.004
    _PY0, _PY1 = -0.070 + 0.5*(_SIDE - _PART) + 0.004, -0.070 + 0.5*(_SIDE + _PART) - 0.004
    PART_REGION = (_PX0, _PY0, _PX1, _PY1)
    PART_BOX = (_PX0, _PY0, 0.9240, _PX1, _PY1, 0.9250)
    _pmask = ((_bv[:, 0] >= _PX0) & (_bv[:, 0] <= _PX1) &
              (_bv[:, 1] >= _PY0) & (_bv[:, 1] <= _PY1))
    PART_ROI = {"region": PART_REGION, "bounds3d": PART_BOX, "mask": _pmask}
    whole = gate(bench)
    c = levels(whole, "normal_roughness")
    check(not c["ok"] and c["value"] > 45.0,
          "over the WHOLE mesh the roughness p99 is %.1f deg and warns -- the "
          "fixturing, not the part" % (c["value"],))
    check(whole["measured"]["surface_scope"]["restricted"] is False
          and whole["measured"]["normal_turn_deg"]
          == whole["measured"]["normal_turn_deg_whole_mesh"],
          "...and with no roi the judged number IS the whole-mesh number, so "
          "nothing an existing caller relies on has moved")
    check("WHOLE MESH" in c["why"] and "roi=" in c["why"],
          "...but it now says so, and says how to measure the part instead")
    part = gate(bench, roi=PART_ROI)
    c = levels(part, "normal_roughness")
    check(c["ok"] and c["value"] < 1.0,
          "restricted to the operator's ROI it is %.2f deg and does not warn"
          % (c["value"],))
    check(near_eq(part["measured"]["normal_turn_deg_whole_mesh"]["p99"],
                  whole["measured"]["normal_turn_deg"]["p99"]),
          "...while the whole-mesh figure is still reported alongside it (%.1f), "
          "so the restriction is never a way of hiding one"
          % (part["measured"]["normal_turn_deg_whole_mesh"]["p99"],))
    check("normal_roughness" in whole["warn"] and "normal_roughness" not in part["warn"],
          "the verdict really does change -- which is the point, and why it is "
          "opt-in rather than the default")
    # Tilt is the other statistic with the same disease: the rubble leans far past
    # the cap over most of the mesh and the part leans not at all. Past 40% of area
    # the gate stops warning and REFUSES, so on this mesh the restriction is the
    # difference between a refusal and a plan -- which is the sharpest possible
    # statement of why it may not be applied silently.
    check(whole["measured"]["tilt_area_fraction_over_cap"] > 0.5
          and part["measured"]["tilt_area_fraction_over_cap"] < 1e-6,
          "tilt_loss is restricted too: %.0f%% of the whole mesh is past the 45 deg "
          "cap, 0%% of the part"
          % (whole["measured"]["tilt_area_fraction_over_cap"]*100.0,))
    check(whole["verdict"] == "refuse" and part["verdict"] == "plan",
          "the whole mesh REFUSES (%s) and the part plans (%s)"
          % (whole["summary"], part["summary"]))
    check(near_eq(part["measured"]["tilt_area_fraction_over_cap_whole_mesh"],
                  whole["measured"]["tilt_area_fraction_over_cap"]),
          "...and that one carries its whole-mesh figure too")
    # The topology checks are NOT restricted, and deliberately: the planner reads
    # the whole file, so a defect outside the ROI still stops it.
    check(part["measured"]["faces"] == whole["measured"]["faces"]
          and part["measured"]["nonmanifold_edges"] == whole["measured"]["nonmanifold_edges"]
          and part["measured"]["boundary_loops"] == whole["measured"]["boundary_loops"],
          "the topology the planner chokes on is still counted over the whole file")
    print("\n--- the forms an roi takes -----------------------------------------")
    forms = {
        "a select_roi() dict": PART_ROI,
        "its 3D bounds alone": {"bounds3d": PART_BOX},
        "a bare 6-number box": PART_BOX,
        "a bare 4-number rectangle": PART_REGION,
        "a vertex mask": _pmask,
        "roi='region'": "region",
    }
    for name, form in forms.items():
        kw = {"region": PART_REGION} if isinstance(form, str) else {}
        r = gate(bench, roi=form, **kw)
        check(r["measured"]["surface_scope"]["restricted"]
              and levels(r, "normal_roughness")["value"] < 1.0,
              "%s restricts to the part (p99 %.2f deg over %d faces)"
              % (name, levels(r, "normal_roughness")["value"],
                 r["measured"]["surface_scope"]["faces"]))
    # An ROI selected on the AZURE ROUGH MESH -- which is where
    # run_automated_scan() asks for one -- carries a vertex mask indexed over a
    # DIFFERENT mesh from the Keyence one gated later. The box travels between
    # meshes; the indices do not.
    stale_idx = dict(PART_ROI, mask=np.ones(17, bool))
    r = gate(bench, roi=stale_idx)
    s = r["measured"]["surface_scope"]
    check(s["restricted"] and "x-y rectangle" in s["what"]
          and "over a different mesh" in s["what"],
          "a mask from another mesh falls through to the ROI's x-y region -- the "
          "case run_automated_scan() actually hits -- and says why: %s"
          % (s["what"][:90],))
    check("bounds3d" not in s["what"] and s["z_ignored"] == (0.9240, 0.9250)
          and levels(r, "normal_roughness")["value"] < 1.0,
          "...and the ROI's z is REPORTED (%s) and NOT APPLIED: an ROI drawn on "
          "an Azure surface that reads ~12 mm long has a z box that keeps no "
          "Keyence faces at all" % (s["z_ignored"],))
    # And with a footprint, which is what select_roi() now writes, the DRAWN
    # SHAPE travels rather than its bounding rectangle.
    import roi_select_geom as _G
    _fp = _G.footprint(_bv, _pmask, tris=_bt)
    withfp = dict(PART_ROI, mask=np.ones(17, bool),
                  footprint=_G.footprint_encode(_fp))
    r = gate(bench, roi=withfp)
    check(r["measured"]["surface_scope"]["restricted"]
          and "drawn x-y footprint" in r["measured"]["surface_scope"]["what"]
          and levels(r, "normal_roughness")["value"] < 1.0,
          "a footprint beats the rectangle and still finds the part (%d faces)"
          % (r["measured"]["surface_scope"]["faces"],))
    # An L: the rectangle round it contains bench, the footprint does not. This is
    # the case a region tuple cannot express at all.
    _L = _pmask & ~((_bv[:, 0] > 0.5*(_PX0 + _PX1)) & (_bv[:, 1] > 0.5*(_PY0 + _PY1)))
    _lfp = _G.footprint_encode(_G.footprint(_bv, _L, tris=_bt))
    _nl = gate(bench, roi={"footprint": _lfp})["measured"]["surface_scope"]["faces"]
    _nr = gate(bench, roi=withfp)["measured"]["surface_scope"]["faces"]
    check(_nl < 0.85*_nr,
          "an L-shaped selection keeps %d faces where the full rectangle over the "
          "same part keeps %d (%.0f%%) -- the missing corner is what a region "
          "tuple cannot express at all" % (_nl, _nr, 100.0*_nl/_nr))
    _notch = np.array([[0.5*(_PX0 + _PX1) + 0.010, 0.5*(_PY0 + _PY1) + 0.010]])
    check(not bool(_G.footprint_mask(_lfp, _notch)[0])
          and bool(_G.footprint_mask(_G.footprint_from_region(
              _G.footprint_bounds(_G.footprint_decode(_lfp))), _notch)[0]),
          "...and a point 10 mm inside the notch is OUT of the footprint and IN "
          "its bounding rectangle, which is the difference in one point")
    try:
        gate(bench, roi={"mask": np.ones(17, bool)})
        check(False, "a dict with neither a fitting mask nor bounds raises")
    except ValueError as exc:
        check("bounds3d" in str(exc),
              "...and with no bounds either it raises rather than guessing: %s"
              % (str(exc)[:60],))
    print("\n--- when there is no usable region, it says so ----------------------")
    r = gate(bench, roi="region")
    check(not r["measured"]["surface_scope"]["restricted"]
          and "no region" in r["measured"]["surface_scope"]["what"],
          "roi='region' with no region measures the whole mesh and names the reason")
    r = gate(bench, roi=False)
    check(not r["measured"]["surface_scope"]["restricted"]
          and r["measured"]["normal_turn_deg"] == whole["measured"]["normal_turn_deg"],
          "roi=False is the deliberate 'the whole mesh, deliberately'")
    # THE LOUD CASE, AND THE TWO FACTS IT MUST NOT CONFLATE. Printing a note and
    # planning the whole mesh anyway is how a 30 x 84 mm selection becomes an
    # 80 x 157 mm plan with the bench in it.
    r = gate(bench, roi=(0.20, 0.20, 0.21, 0.21))
    s = r["measured"]["surface_scope"]
    check(not s["restricted"] and s["verdict"] == "mismatch"
          and "roi_mesh_agreement" in r["blocking"] and r["verdict"] == "refuse",
          "an ROI that lands nowhere near the mesh now BLOCKS (%s) rather than "
          "falling back quietly" % (r["summary"],))
    check("DO NOT DESCRIBE THE SAME SURFACE" in s["what"]
          and "x %.3f" % (_bv[:, 0].min(),) in s["what"],
          "...and says so in those words, with the mesh's own extent to compare: "
          "%s" % (s["what"][:80],))
    check(levels(r, "normal_roughness")["value"]
          == levels(whole, "normal_roughness")["value"],
          "...while the statistics still fall back to the whole mesh rather than "
          "to nothing, so the report is complete")
    # A SMALL SELECTION IS THE OTHER FACT: it is about this mesh, the crop is
    # believed, and only the percentiles are refused.
    _tiny = (0.757 + 0.5*_SIDE, -0.070 + 0.5*_SIDE,
             0.757 + 0.5*_SIDE + 0.0025, -0.070 + 0.5*_SIDE + 0.0025)
    r = gate(bench, roi=_tiny)
    s = r["measured"]["surface_scope"]
    check(s["verdict"] == "few_faces" and "roi_few_faces" in r["warn"]
          and "roi_mesh_agreement" not in r["blocking"],
          "a 2.5 mm selection is a SMALL REGION, not a mismatch: %d faces, "
          "%.0f%% of its own footprint covered, and it warns rather than blocking"
          % (s["roi_faces"], s["covered"]*100.0))
    check("small selection and not a mismatch" in s["what"]
          and "roi_mesh_agreement" not in r["fatal"] + r["blocking"],
          "...and the two are worded so a log says which happened")
    # And the scope is printed every time, restricted or not: a report that does not
    # say which surface it measured is the thing this whole section is about.
    _out = []
    _print = print
    try:
        import builtins
        builtins.print = lambda *a, **k: _out.append(" ".join(str(x) for x in a))
        describe(part)
        describe(whole)
    finally:
        builtins.print = _print
    check(sum("surface measured" in l for l in _out) == 2,
          "describe_mesh_plan_report() names the surface in both cases")
    check(any("rotation offset" in l and "passes run along" in l for l in _out),
          "...and which way the passes run, which decides the pass count")


def _sheet(x, y, zf, dx=0.0, dy=0.0):
    """A triangulated height field over x cross y. -> (verts, tris)."""
    gx, gy = np.meshgrid(x, y, indexing="ij")
    v = np.column_stack([(gx + dx).ravel(), (gy + dy).ravel(),
                         zf(gx, gy).ravel()])
    nx, ny = len(x), len(y)
    i = np.arange(nx*ny).reshape(nx, ny)
    t = np.vstack([
        np.column_stack([i[:-1, :-1].ravel(), i[1:, :-1].ravel(),
                         i[1:, 1:].ravel()]),
        np.column_stack([i[:-1, :-1].ravel(), i[1:, 1:].ravel(),
                         i[:-1, 1:].ravel()])])
    return v, t


@_checked
def test_the_roi_z_band():
    print("\n--- the ROI's z band: per cell, not a slab --------------------------")
    # WHAT THIS SECTION IS FOR. A GLOBAL z box taken from the ROI's own bounds
    # can keep zero faces, which makes dropping z from the crop look like the
    # fix. It is not: without z, a selection on top of a plate lets the profiler
    # scan part of the table underneath. The crop must stay inside the ROI in z
    # as well, to a sensible tolerance -- so z is applied as a band around a
    # PER-CELL height map. This section shows the band excludes the bench, keeps
    # the part, and on a curved specimen does something no slab can.
    import roi_select_geom as _G
    crop = ns["crop_mesh_to_roi"]
    RoiDoesNotFitMesh = ns["RoiDoesNotFitMesh"]
    faces = ns["_gate_roi_faces"]
    shape_of = ns["_gate_roi_shape"]
    TOL = ns["GATE_ROI_Z_TOLERANCE_M"]
    check(abs(TOL - 0.006) < 1e-12,
          "GATE_ROI_Z_TOLERANCE_M is %.1f mm: 2.4x the 2.54 mm worst per-cell "
          "disagreement measured between a cell's two sensors, and under a third "
          "of the 19.2 mm from a plate top to the bench" % (TOL*1e3,))

    # ---- A FLAT PLATE ON A BENCH, the common case ---------------------------
    # The plate 19 mm above the bench: bench 0.9218, plate 0.9410.
    BENCH, PLATE = 0.9218, 0.9410
    ax = np.arange(0.800, 0.8801, 0.002)
    ay = np.arange(0.000, 0.1201, 0.002)
    pv, pt = _sheet(ax, ay, lambda gx, gy: np.full(gx.shape, PLATE))
    # The bench reaches 20 mm past the plate on every side, so a crop with the
    # 5 mm planning margin reaches it whenever a selection runs to the edge of
    # the plate. That is how the table ends up in a scan.
    bx = np.arange(0.780, 0.9001, 0.002)
    by = np.arange(-0.020, 0.1401, 0.002)
    bvv, bt = _sheet(bx, by, lambda gx, gy: np.full(gx.shape, BENCH))
    v = np.vstack([pv, bvv])
    t = np.vstack([pt, bt + len(pv)])
    ctr = v[t].mean(axis=1)
    is_bench = ctr[:, 2] < BENCH + 0.0005
    mesh = write_ply(os.path.join(TMP, "zband_plate_on_bench.ply"), v, None, t)

    # The ROI as select_roi() writes it: the whole plate, drawn on a COARSE 6 mm
    # surface reading 1.3 mm high, which is the Azure's measured bias on a part
    # like this.
    cx = np.arange(0.800, 0.8801, 0.006)
    cy = np.arange(0.000, 0.1201, 0.006)
    cv, ct = _sheet(cx, cy, lambda gx, gy: np.full(gx.shape, PLATE + 0.0013))
    fp = _G.footprint(cv, tris=ct)
    roi = {"region": list(_G.bounding_region(cv)),
           "bounds3d": list(_G.bounding_box3d(cv)),
           "footprint": _G.footprint_encode(fp)}
    shape = shape_of(roi, None, nvertices=len(v), nfaces=len(t))
    check("z within +/-6.0 mm of its own surface" in shape["what"],
          "the shape says z is in the decision and what the band is: %s"
          % (shape["what"][-96:],))

    m_xy, _cov, _z = faces(v, t, shape, margin_m=0.005, z_tolerance=0.0)
    m_bd, _cov, zb = faces(v, t, shape, margin_m=0.005)
    check(int((m_xy & is_bench).sum()) > 2000,
          "x-y ALONE lets %d bench faces through, because the 5 mm planning "
          "margin runs off the edge of the plate -- this is how the table gets "
          "scanned" % (int((m_xy & is_bench).sum()),))
    check(int((m_bd & is_bench).sum()) == 0,
          "the +/-6 mm per-cell band excludes ALL %d of them (%d left)"
          % (int((m_xy & is_bench).sum()), int((m_bd & is_bench).sum())))
    check(int((m_bd & ~is_bench).sum()) == int((m_xy & ~is_bench).sum())
          and int((m_bd & ~is_bench).sum()) > 3000,
          "...and keeps every one of the %d PART faces x-y kept: it does not "
          "collapse the way the global box did"
          % (int((m_xy & ~is_bench).sum()),))
    # COVERAGE IS STILL MEASURED ON THE ROI AS DRAWN, not on the ROI plus the
    # planning margin, and not on what the band kept. It decides
    # roi_mesh_agreement, and it has to stay able to say "laterally these are the
    # same surface" while the band says the heights disagree -- which is the
    # signature of an ROI drawn on a differently calibrated sensor.
    _c0 = faces(v, t, shape, margin_m=0.0)[1]
    _c5 = faces(v, t, shape, margin_m=0.005)[1]
    _cz = faces(v, t, shape, margin_m=0.005, z_tolerance=0.0)[1]
    check(near_eq(_c0, _c5) and near_eq(_c0, _cz),
          "coverage (%.4f) does not move with the planning margin or with the "
          "band: it is a property of the drawn ROI and this mesh" % (_c0,))
    check(abs(zb["offset_mm"] - 1.3) < 0.2 and zb["no_height"] == 0,
          "and it reports the sensor disagreement it absorbed on the surface it "
          "KEPT, %.2f mm, with nothing left unjudged" % (zb["offset_mm"],))
    check(zb["offset_all_mm"] > 15.0,
          "...measured over every x-y face instead it would read %.1f mm, "
          "because that averages the plate with the bench the margin swept up -- "
          "which is why there are two numbers and the report shows the first"
          % (zb["offset_all_mm"],))

    # The crop writes the smaller file, and the report says all four things.
    out = crop(mesh, os.path.join(TMP, "zband_cropped.ply"), roi=roi,
               region=roi["region"], verbose=False)
    cv2, _cn2, ct2 = read_ply_mesh(out)
    check(len(ct2) == int(m_bd.sum()) and cv2[:, 2].min() > PLATE - TOL,
          "crop_mesh_to_roi() writes %d faces and the lowest is at z %.4f -- no "
          "bench in the file the planner reads" % (len(ct2), cv2[:, 2].min()))
    cr = ns["last_roi_crop_report"]
    check(cr["z_band"] is not None
          and cr["z_band"]["dropped"] == int((m_xy & ~m_bd).sum()),
          "...and the crop report carries the band: %d faces excluded by z"
          % (cr["z_band"]["dropped"],))

    # THE WAY BACK, both of them, and neither is silent.
    out0 = crop(mesh, os.path.join(TMP, "zband_noz.ply"), roi=roi,
                region=roi["region"], verbose=False, z_tolerance=0.0)
    _v0, _n0, t0 = read_ply_mesh(out0)
    check(len(t0) == int(m_xy.sum()) and ns["last_roi_crop_report"]["z_band"] is None,
          "z_tolerance=0 is the way back to the x-y crop (%d faces) and the "
          "report says the band was not applied" % (len(t0),))
    _keep = ns["GATE_CROP_TO_ROI"]
    try:
        ns["GATE_CROP_TO_ROI"] = False
        check(crop(mesh, roi=roi, region=roi["region"], verbose=False) == mesh,
              "GATE_CROP_TO_ROI=False still skips the crop entirely")
    finally:
        ns["GATE_CROP_TO_ROI"] = _keep

    # A RECTANGLE SAYS NOTHING ABOUT z, and the report must not imply it did.
    rect = crop(mesh, os.path.join(TMP, "zband_rect.ply"),
                roi=roi["region"], region=roi["region"], verbose=False)
    rr = ns["last_roi_crop_report"]
    check(rr["z_band"] is None,
          "a plain rectangle ROI carries no height map, so no band is applied")
    _out = []
    _print = print
    try:
        import builtins
        builtins.print = lambda *a, **k: _out.append(" ".join(str(x) for x in a))
        describe(gate(mesh, roi=roi["region"], region=roi["region"]))
    finally:
        builtins.print = _print
    check(any("z: NOT applied" in l and "no height map" in l for l in _out),
          "...and the report SAYS z was not applied rather than staying quiet")
    _out = []
    try:
        import builtins
        builtins.print = lambda *a, **k: _out.append(" ".join(str(x) for x in a))
        describe(gate(mesh, roi=roi, region=roi["region"]))
    finally:
        builtins.print = _print
    check(any("z: within +/-6.0 mm of the selected surface" in l
              and "it excluded" in l for l in _out),
          "while with a height map it names the band, its surface, and how many "
          "faces it cost: %s"
          % (next((l.strip() for l in _out if "z: within" in l), "MISSING")[:110],))
    check(any("GLOBAL z box" in l and "provenance, not a crop" in l
              for l in _out),
          "...and the ROI's global z box is STILL reported as provenance only")
    check(any("the height map came from" in l and "sampled directly" in l
              and "filled from neighbours" in l for l in _out),
          "...and where the height map came from, which no count shows: %s"
          % (next((l.strip() for l in _out if "height map came from" in l),
                  "MISSING")[:110],))

    # A NaN VERTEX must not take the band down. A handful of NaNs in a few
    # thousand vertices is what a smoothing stage writes, and every report line
    # reads the band's numbers.
    vn = v.copy()
    vn[7, 2] = np.nan
    nm, _c, nzb = faces(vn, t, shape, margin_m=0.005)
    check(nzb is not None and np.isfinite(nzb["offset_mm"])
          and (nzb["dropped_z"] is None or np.isfinite(nzb["dropped_z"]).all()),
          "a NaN vertex leaves the band's numbers finite (%d dropped, offset "
          "%.2f mm)" % (nzb["dropped"], nzb["offset_mm"]))
    check(abs(int(nm.sum()) - int(m_bd.sum())) <= 6,
          "...and costs at most the faces that touch it (%d against %d)"
          % (int(nm.sum()), int(m_bd.sum())))

    # ---- A CURVED PART -------------------------------------------------------
    # A cylindrical crown, radius 60 mm over a 100 mm chord: 26.8 mm of relief,
    # its edges coming down onto the bench. THE POINT: a global slab wide enough
    # to contain the crown also contains the bench under it, so a slab plus a
    # tolerance is not an alternative to a per-cell band -- it is the global box
    # again with a wider box.
    R, W, XC, CB = 0.060, 0.050, 0.850, 0.9200

    def crown(px, py):
        d = np.clip(np.abs(px - XC), 0.0, W)
        return CB + np.sqrt(R*R - d*d) - np.sqrt(R*R - W*W)

    kx = np.arange(XC - W, XC + W + 1e-9, 0.001)
    ky = np.arange(0.000, 0.1201, 0.001)
    arc_v, arc_t = _sheet(kx, ky, crown)
    bench_v, bench_t = _sheet(kx, ky, lambda gx, gy: np.full(gx.shape, CB))
    v2 = np.vstack([arc_v, bench_v])
    t2 = np.vstack([arc_t, bench_t + len(arc_v)])
    ctr2 = v2[t2].mean(axis=1)
    bench2 = ctr2[:, 2] < CB + 0.0005
    acx = np.arange(XC - W, XC + W + 1e-9, 0.006)
    acy = np.arange(0.000, 0.1201, 0.006)
    acv, act = _sheet(acx, acy, crown)
    cfp = _G.footprint(acv, tris=act)
    relief = (np.nanmax(cfp["heights"]) - np.nanmin(cfp["heights"]))*1e3
    check(abs(relief - 26.8) < 1.0,
          "the crown's height map has %.1f mm of relief -- it FOLLOWS the curve "
          "instead of being one number" % (relief,))
    cshape = shape_of({"footprint": _G.footprint_encode(cfp)}, None,
                      nvertices=len(v2), nfaces=len(t2))
    cm_xy, _c, _z = faces(v2, t2, cshape, margin_m=0.005, z_tolerance=0.0)
    cm_bd, _c, czb = faces(v2, t2, cshape, margin_m=0.005)
    lo = float(np.nanmin(cfp["heights"])) - TOL
    hi = float(np.nanmax(cfp["heights"])) + TOL
    slab = cm_xy & (ctr2[:, 2] >= lo) & (ctr2[:, 2] <= hi)
    check(int((slab & bench2).sum()) == int((cm_xy & bench2).sum())
          and int((cm_xy & bench2).sum()) > 2000,
          "a GLOBAL slab of the same +/-6 mm is %.1f mm tall and keeps all %d "
          "bench faces under the crown -- no better than no z at all"
          % ((hi - lo)*1e3, int((slab & bench2).sum())))
    check(int((cm_bd & bench2).sum()) < 0.25*int((cm_xy & bench2).sum()),
          "...while the PER-CELL band keeps %d of those %d (%.0f%%), and what is "
          "left is where the arc genuinely comes within 6 mm of the bench at its "
          "edges"
          % (int((cm_bd & bench2).sum()), int((cm_xy & bench2).sum()),
             100.0*(cm_bd & bench2).sum()/max(int((cm_xy & bench2).sum()), 1)))
    check(int((cm_bd & ~bench2).sum()) == int((cm_xy & ~bench2).sum())
          and int((cm_bd & ~bench2).sum()) > 10000,
          "...and it keeps ALL %d faces of the curved surface itself: following "
          "the curve is what lets the band be tight"
          % (int((cm_bd & ~bench2).sum()),))

    # A FACE SPANNING CELLS on a slope is judged against the surface it spans,
    # not against one cell. On this crown the steepest faces climb 0.3 mm across
    # their own 1 mm, and none of them is dropped for it.
    steep = (~bench2) & (np.abs(ctr2[:, 0] - XC) > 0.9*W)
    check(bool(cm_bd[steep].all()),
          "every one of the %d steepest faces on the crown survives the band, "
          "which is what taking the RANGE of a face's cells is for"
          % (int(steep.sum()),))

    # ---- THE THREE NEAR-EMPTY CROPS ARE THREE DIFFERENT FACTS ---------------
    # A height map 40 mm above this mesh: laterally a perfect match, vertically
    # impossible. That is an ROI drawn on a differently calibrated surface, and
    # it must NOT be reported as a lateral mismatch, as a small selection, or as
    # a quiet retreat to x-y.
    high = _G.footprint(cv + np.array([0.0, 0.0, 0.040]), tris=ct)
    hroi = dict(roi, footprint=_G.footprint_encode(high))
    try:
        crop(mesh, os.path.join(TMP, "never_z.ply"), roi=hroi,
             region=roi["region"], verbose=False)
        check(False, "an ROI 40 mm too high raises RoiDoesNotFitMesh")
    except RoiDoesNotFitMesh as exc:
        check("AT DIFFERENT HEIGHTS" in str(exc)
              and "median offset" in str(exc),
              "an ROI 40 mm too high REFUSES and names the height, not the "
              "place: %s" % (str(exc)[:76],))
        check("do not describe the same surface" not in str(exc),
              "...and is worded differently from the lateral mismatch, because it "
              "sends you to a different question")
    except Exception as exc:
        check(False, "it raises RoiDoesNotFitMesh, not %r" % (exc,))
    r = gate(mesh, roi=hroi, region=roi["region"])
    s = r["measured"]["surface_scope"]
    check(s["verdict"] == "z_mismatch" and "roi_z_band" in r["blocking"]
          and r["verdict"] == "refuse",
          "the gate blocks on its OWN check name, roi_z_band, not on "
          "roi_mesh_agreement (%s)" % (r["summary"],))
    check("roi_mesh_agreement" not in r["blocking"]
          and "roi_few_faces" not in r["warn"],
          "...and neither of the other two fires, so a log says which happened")
    check(not s["restricted"] and levels(r, "roi_z_band")["value"] < 0.10,
          "...with the number that decided it: %.1f%% of the x-y faces survived "
          "the band" % (levels(r, "roi_z_band")["value"]*100.0,))
    # And the lateral mismatch still reports itself as a lateral one even though
    # its band is trivially empty too -- lateral is tested FIRST on purpose.
    far = _G.footprint(cv + np.array([0.30, 0.0, 0.0]), tris=ct)
    r = gate(mesh, roi=dict(roi, footprint=_G.footprint_encode(far)),
             region=roi["region"])
    check(r["measured"]["surface_scope"]["verdict"] == "mismatch"
          and "roi_mesh_agreement" in r["blocking"]
          and "roi_z_band" not in r["blocking"],
          "an ROI 300 mm away in x is still a LATERAL mismatch, not a height one "
          "-- otherwise it would send you to the calibration instead of to which "
          "scan the selection came from")


# ---------------------------------------------------------------------------
# The real meshes, with the verdicts principal_axis_noether_srv itself gives
# them. Skipped when the scan data is not present.
#: Every stage of one plate reconstruction, in the fixture repository.
REAL_SUBDIR = ("keyence", "plate_stages")

#: (file, expected verdict, expected fatal checks, what the planner does with it)
REAL = [
    ("specimen_keyence_poisson.ply", "repair", {"manifold", "components"},
     "REFUSED by the planner at 1.5, 0.5 and 0.3 mm; plans at all twelve "
     "spacings once the 8 faces on its 2 non-manifold edges are dropped"),
    ("specimen_keyence_qudric_decimation.ply", "plan", set(),
     "plans at 5, 2, 1.5, 1, 0.6, 0.5, 0.4 and 0.3 mm as it is"),
    ("specimen_keyence_ball_pivot_smoothed.ply", "repair", {"finite", "manifold"},
     "6 NaN vertices; ABORTS principal_axis_noether_srv on a PCL assertion"),
    ("specimen_keyence_raw.ply", "refuse", {"readable"}, "point cloud, 563654 points"),
    ("specimen_keyence_sor_filtered.ply", "refuse", {"readable"}, "point cloud"),
    ("specimen_keyence_cropped.ply", "refuse", {"readable"}, "point cloud"),
    ("specimen_keyence_merged.ply", "refuse", {"readable"}, "point cloud"),
    ("specimen_keyence_final_points.ply", "refuse", {"readable"}, "point cloud"),
]

@_checked
def test_the_cells_own_meshes():
    print("\n--- a real reconstruction chain's own meshes ------------------------")
    if test_data.path(*REAL_SUBDIR) is None:
        print("SKIPPED: %s" % (test_data.why_missing(*REAL_SUBDIR),))
    else:
        for name, verdict, fatal, note in REAL:
            p = test_data.path(*REAL_SUBDIR, name)
            if p is None:
                print("skip %s (absent)" % (name,))
                continue
            rep = gate(p)
            check(rep["verdict"] == verdict,
                  "%s -> %s (%s)" % (name, verdict, note))
            check(set(rep["fatal"]) == fatal,
                  "  its fatal checks are exactly %s (got %s)"
                  % (sorted(fatal), sorted(rep["fatal"])))
        # The numbers the thresholds were set from.
        p = test_data.path(*REAL_SUBDIR, "specimen_keyence_poisson.ply")
        if os.path.exists(p):
            m = gate(p)["measured"]
            check(m["nonmanifold_edges"] == 2,
                  "the Poisson mesh has exactly 2 non-manifold edges (%d)"
                  % (m["nonmanifold_edges"],))
            check(m["faces_on_nonmanifold_edges"] == 8,
                  "on 8 of its 22899 faces -- 0.035%% of the mesh (%d)"
                  % (m["faces_on_nonmanifold_edges"],))
            check(abs(m["edge_mm"]["median"] - 0.98) < 0.05,
                  "median edge 0.98 mm (%.2f)" % (m["edge_mm"]["median"],))
            check(abs(m["tilt_area_fraction_over_cap"] - 0.062) < 0.01,
                  "6.2%% of its area is past the 45 deg cap, and the planner "
                  "then drops 7.0%% of the waypoints (%.3f)"
                  % (m["tilt_area_fraction_over_cap"],))
            check(m["in_work_volume"] > 0.99,
                  "and all of it is inside the measured work volume (%.3f)"
                  % (m["in_work_volume"],))
            # Repairing it is what makes the production spacing plan.
            rep = gate(p)
            check(rep["repairable"], "the Poisson mesh is repairable, not hopeless")
            repaired = os.path.join(TMP, "poisson_repaired.ply")
            cr = clean_mesh(p, repaired)
            check(cr["faces_out"] == 22885,
                  "clean_mesh_for_planning() keeps 22885 of 22899 faces (%d)"
                  % (cr["faces_out"],))
            check(gate(repaired)["verdict"] == "plan",
                  "and the repaired copy gates 'plan' -- the copy the planner "
                  "turns into 74100 poses at 0.3 mm")
        p = test_data.path(*REAL_SUBDIR, "specimen_keyence_ball_pivot_smoothed.ply")
        if os.path.exists(p):
            m = gate(p)["measured"]
            check(m["nonfinite_vertices"] == 6,
                  "the smoothed ball-pivot mesh has 6 NaN vertices in 8466 (%d)"
                  % (m["nonfinite_vertices"],))

# ---------------------------------------------------------------------------
# The pair of scans the ROI reasoning above is measured on: same cell, same
# plate, same everything except the surface preparation -- Kapton tape on one,
# sublimating spray on the other. The spray scan is the good one.
#
# These are the numbers in the note above the ROI section, kept here so a change
# to the gate has to reproduce them. Scan data, so skipped loudly when the tree
# is not present.
#: The planner-ready mesh each ROI run is graded on, in the fixture repository.
ROI_MESH = "specimen_keyence_poisson_filtered_ecplan.ply"

#: (run directory, EC_SCAN_REGION as the detector gave it, whole-mesh p99,
#:  plate p99, whole-mesh tilt fraction, plate tilt fraction)
ROI_RUNS = [
    (("keyence", "roi_kapton"),
     (0.756, -0.105, 0.891, 0.181), 55.5, 10.5, 0.145, 0.0),
    (("keyence", "roi_spray"),
     (0.761, -0.116, 0.884, 0.177), 52.6, 4.5, 0.104, 0.0),
]

#: The plate's top face in world metres, as roi_select() would hand it back:
#: an 83 x 165 mm plate at z 939.7 mm +/- 4.
ROI_PLATE = {"region": (0.785, 0.005, 0.852, 0.140),
             "bounds3d": (0.785, 0.005, 0.9357, 0.852, 0.140, 0.9437)}

@_checked
def test_the_2026_09_16_pair():
    print("\n--- two preparations of one plate: whole mesh against the plate -----")
    if test_data.root() is None:
        print("SKIPPED: %s" % (test_data.why_missing("keyence"),))
    else:
        seen = []
        for sub, reg, wp99, pp99, wtilt, ptilt in ROI_RUNS:
            p = test_data.path(*sub, ROI_MESH)
            if p is None:
                print("skip %s (absent)" % (sub[-1],))
                continue
            kw = dict(line_spacing=LINE_SPACING, max_tilt_deg=45.0, region=reg,
                      work_volume=WORK_VOLUME, rotation_offset=np.pi/2)
            w = report(p, **kw)["measured"]
            r = report(p, roi=ROI_PLATE, **kw)["measured"]
            seen.append((w, r))
            name = sub[-1]
            check(abs(w["normal_turn_deg"]["p99"] - wp99) < 0.5,
                  "%s: the whole mesh reads %.1f deg (%.1f expected)"
                  % (name, w["normal_turn_deg"]["p99"], wp99))
            check(abs(r["normal_turn_deg"]["p99"] - pp99) < 0.5,
                  "  restricted to the plate it reads %.1f deg (%.1f expected)"
                  % (r["normal_turn_deg"]["p99"], pp99))
            check(abs(w["tilt_area_fraction_over_cap"] - wtilt) < 0.01
                  and r["tilt_area_fraction_over_cap"] <= ptilt + 1e-3,
                  "  and %.1f%% of the whole mesh is past the tilt cap against "
                  "%.1f%% of the plate"
                  % (w["tilt_area_fraction_over_cap"]*100.0,
                     r["tilt_area_fraction_over_cap"]*100.0))
            # The convention, on a mesh that flew: pi/2 steps the planes across
            # the SECOND-longest extent.
            check(r["pass_axis"] == 0 and abs(w["plane_step_extent_mm"]
                                              - min(w["principal_extent_mm"][:2])) < 0.1,
                  "  pi/2 ran the passes along the LONGEST principal axis (%.1f mm) "
                  "and stepped the planes across the second (%.1f mm), %d of them"
                  % (w["pass_extent_mm"], w["plane_step_extent_mm"],
                     w["estimated_passes"]))
        if len(seen) == 2:
            (wk, rk), (ws, rs) = seen
            wr = wk["normal_turn_deg"]["p99"]/ws["normal_turn_deg"]["p99"]
            pr = rk["normal_turn_deg"]["p99"]/rs["normal_turn_deg"]["p99"]
            check(wr < 1.15 and pr > 2.0,
                  "THE POINT: over the whole mesh the bad scan and the good one are "
                  "%.2fx apart, and over the plate they are %.2fx. Only the second "
                  "can grade a surface preparation." % (wr, pr))
            check(wk["estimated_passes"] == 566 and ws["estimated_passes"] == 543,
                  "and the gate's pass estimate is the planner's own slice count, "
                  "566 and 543 (%d, %d)"
                  % (wk["estimated_passes"], ws["estimated_passes"]))

# ---------------------------------------------------------------------------
# AN ROI THAT CROPS NOTHING, ON REAL MESHES
# ---------------------------------------------------------------------------
# The case the ROI crop exists for: one operator selection drawn on the AZURE
# rough mesh and applied to the KEYENCE mesh, over two runs.
#
#   the ROI's 3D bounds (Azure)                      z 0.9416 .. 0.9435
#   specimen_keyence_..._filtered.ply, run 1         z 0.9127 .. 0.9411
#   specimen_keyence_..._filtered.ply, run 2         z 0.9406 .. 0.9410
#
# The z ranges DO NOT OVERLAP, so an ROI 3D box keeps 0 faces of 20327 and 0 of
# 11896. Falling back to the whole mesh with a note turns a 30 x 84 mm selection
# into an 80 x 157 mm plan with the optical breadboard in it. In x and y the ROI
# is comfortably inside both meshes, so the fault is entirely in z.
#
# The meshes live in the fixture repository; these sections skip without it.
R12_MEMBERS = (("keyence", "roi_submesh.ply"),
               ("keyence", "curved_a",
                "specimen_keyence_ball_pivot_smoothed_filtered.ply"),
               ("keyence", "curved_b",
                "specimen_keyence_ball_pivot_smoothed_filtered.ply"))


def _r12_files():
    """The three real meshes as paths, or None when the fixtures are absent."""
    got = [test_data.path(*m) for m in R12_MEMBERS]
    return got if all(got) else None


@_checked
def test_the_roi_that_cropped_nothing():
    print("\n--- an ROI that crops nothing, on real meshes -----------------------")
    files = _r12_files()
    if files is None:
        print("SKIPPED: %s" % (test_data.why_missing(*R12_MEMBERS[0]),))
        return
    import roi_select_geom as _G
    crop = ns["crop_mesh_to_roi"]
    RoiDoesNotFitMesh = ns["RoiDoesNotFitMesh"]
    sub, r1, r2 = files

    # The ROI as the selector wrote it: a sub-mesh of the AZURE surface, a region
    # tuple, a 3D bounds box, and a per-vertex mask over the AZURE mesh's 17239
    # vertices -- which is the mask that cannot fit either Keyence mesh.
    sv, sn, st = _G.read_ply(sub)
    roi = {"region": list(_G.bounding_region(sv)),
           "bounds3d": list(_G.bounding_box3d(sv)),
           "mask": np.ones(17239, bool),
           "footprint": _G.footprint_encode(_G.footprint(sv, tris=st))}
    b = roi["bounds3d"]
    check(abs(b[2] - 0.9416) < 1e-3 and abs(b[5] - 0.9435) < 1e-3,
          "the ROI's own z is %.4f..%.4f, off the Azure rough mesh" % (b[2], b[5]))

    EXPECT = {"run 1": (r1, 20327, 5705, 8944, 0.9127, 0.9411),
              "run 2": (r2, 11896, 5525, 8531, 0.9406, 0.9410)}
    for name in ("run 1", "run 2"):
        path, nface, n_fp, n_margin, zlo, zhi = EXPECT[name]
        v, n, t = read_ply_mesh(path)
        check(len(t) == nface, "%s: %d faces" % (name, len(t)))
        check(abs(v[:, 2].min() - zlo) < 1e-3 and abs(v[:, 2].max() - zhi) < 1e-3,
              "  it is at z %.4f..%.4f, entirely BELOW the ROI's z floor %.4f"
              % (v[:, 2].min(), v[:, 2].max(), b[2]))
        # The failure mode, reproduced: the ROI's 3D box applied as a crop.
        ctr = v[t].mean(axis=1)
        lo = np.array([min(b[0], b[3]), min(b[1], b[4]), min(b[2], b[5])])
        hi = np.array([max(b[0], b[3]), max(b[1], b[4]), max(b[2], b[5])])
        old = int(np.all((ctr >= lo) & (ctr <= hi), axis=1).sum())
        check(old == 0,
              "  BEFORE: the ROI's 3D bounds keep %d of %d faces -- an empty crop, "
              "not a lateral mismatch" % (old, len(t)))
        # ...and in x-y alone it is comfortably inside: there is no lateral fault.
        xy = int(((ctr[:, 0] >= lo[0]) & (ctr[:, 0] <= hi[0]) &
                  (ctr[:, 1] >= lo[1]) & (ctr[:, 1] <= hi[1])).sum())
        check(xy > 2000,
              "  the SAME box in x-y alone keeps %d faces, so the fault was "
              "entirely in z" % (xy,))
        # AFTER: the drawn footprint, which is what the ROI now carries.
        got = int(_G.footprint_mask(roi["footprint"], ctr).sum())
        check(got == n_fp,
              "  AFTER: the drawn x-y footprint keeps %d of %d faces (%.0f%%)"
              % (got, len(t), 100.0*got/len(t)))
        check(_G.footprint_mask(roi["footprint"], ctr, margin_m=0.005).sum()
              == n_margin,
              "  and %d with the 5 mm planning margin, so Noether's passes can "
              "reach the region's edge" % (n_margin,))
        # The gate no longer applies the z, and says so.
        rep = report(path, roi=roi, line_spacing=LINE_SPACING,
                     region=roi["region"], work_volume=WORK_VOLUME)
        s = rep["measured"]["surface_scope"]
        check(s["restricted"] and s["roi_faces"] == n_fp
              and "roi_mesh_agreement" not in rep["blocking"],
              "  the gate restricts to those %d faces and does not block"
              % (s["roi_faces"],))
        check(tuple(np.round(s["z_ignored"], 4)) == (0.9416, 0.9435),
              "  ...reporting the ROI's GLOBAL z box %s as provenance and "
              "applying none of it"
              % (tuple(np.round(s["z_ignored"], 4)),))
        # AND APPLYING THE PER-CELL BAND, which is how z stays in the decision.
        # On this flat plate it keeps every face the x-y footprint kept -- the two
        # sensors disagree by 1.3-1.4 mm per cell, well inside the 6 mm band --
        # which is what says the band does not repeat the global box's failure.
        zb = s["z_band"]
        check(zb is not None and zb["faces"] == n_fp and zb["dropped"] == 0,
              "  ...and its PER-CELL z band keeps all %d of them (%d dropped): "
              "the band that replaced the box does not empty the crop"
              % (zb["faces"] if zb else -1, zb["dropped"] if zb else -1))
        check(0.8 < zb["offset_mm"] < 2.6 and zb["no_height"] == 0,
              "  ...while measuring the sensor disagreement it absorbed: %.2f mm, "
              "inside the 0.8-2.6 mm the tolerance was derived from"
              % (zb["offset_mm"],))
        # THE MESH ITSELF IS CROPPED, which is what stopped the bench being
        # planned. The margin is why this is bigger than the statistics' count.
        out = crop(path, os.path.join(TMP, "cropped_%s.ply" % (name[-1],)),
                   roi=roi, region=roi["region"], verbose=False)
        cv, cn, ct = read_ply_mesh(out)
        check(len(ct) == n_margin and len(ct) < len(t),
              "  crop_mesh_to_roi(): %d of %d faces written to %s"
              % (len(ct), len(t), os.path.basename(out)))
        check(cn is not None,
              "  ...with its vertex normals kept -- left to the planner's own "
              "estimate, a sixth of the poses can come back facing into the part")
        got2 = require(out, roi=roi, region=roi["region"], verbose=False,
                       line_spacing=LINE_SPACING, work_volume=WORK_VOLUME,
                       repair_to=os.path.join(TMP, "cropped_%s_gated.ply" % (name[-1],)),
                       crop_to_roi=False)
        check(os.path.exists(got2),
              "  and the cropped mesh gates for planning (%s)"
              % (os.path.basename(got2),))

    # WHAT THE BAND IS WORTH ON THIS DATA, which takes a selection reaching the
    # edge of the plate to show. The recorded ROI is 36 x 78 mm of an 84 x 126 mm
    # plate -- so far inside it that even the 5 mm planning margin never leaves
    # the top face, and an x-y crop alone already excludes everything below.
    # Grown by 25 mm it is what "select the whole plate" produces, and then x-y
    # alone reaches DOWN THE SIDE OF THE PLATE and past the bench top.
    wide_fp = _G.footprint_grown(_G.footprint(sv, tris=st), 0.025)
    wide = {"footprint": _G.footprint_encode(wide_fp)}
    wshape = ns["_gate_roi_shape"](wide, None, nvertices=1, nfaces=2)
    WIDE = {"run 1": (r1, 19107, 1695, 0.9190), "run 2": (r2, 11896, 0, None)}
    for name in ("run 1", "run 2"):
        path, n_xy, n_drop, zmin = WIDE[name]
        v, n, t = read_ply_mesh(path)
        ctr = v[t].mean(axis=1)
        m_xy, _c, _z = ns["_gate_roi_faces"](v, t, wshape, margin_m=0.005,
                                             z_tolerance=0.0)
        m_bd, _c, zb = ns["_gate_roi_faces"](v, t, wshape, margin_m=0.005)
        check(int(m_xy.sum()) == n_xy,
              "  %s: a whole-plate selection keeps %d faces in x-y alone, down to "
              "z %.4f%s" % (name, int(m_xy.sum()), ctr[m_xy, 2].min(),
                            "" if zmin is None else
                            " -- %.1f mm BELOW the bench top at 0.9218"
                            % ((0.9218 - ctr[m_xy, 2].min())*1e3,)))
        check(zb["dropped"] == n_drop,
              "  ...and the +/-6 mm band removes %d of them, which x-y alone kept%s"
              % (zb["dropped"],
                 "" if not zb["dropped"] else
                 " (at z %.4f..%.4f, %.0f-%.0f mm below the selected surface)"
                 % (zb["dropped_z"][0], zb["dropped_z"][1],
                    (0.9410 - zb["dropped_z"][1])*1e3,
                    (0.9410 - zb["dropped_z"][0])*1e3)))
        check(zb["faces"] > 0.85*int(m_xy.sum()),
              "  ...while keeping %.1f%% of what x-y kept: it trims the sides, it "
              "does not collapse" % (zb["kept_fraction"]*100.0,))

    # THE CHAIN, WITH NOTHING PASSED BY HAND. plan_ec_path() -> plan_mesh_path()
    # -> _gate_planning_mesh() -> require_plannable_mesh(roi=...) is how an
    # operator's selection reaches this file, and the MESH that comes back is
    # cropped. Copied into TMP first because a scan writes the cropped copy
    # beside its input and the scan data may be read-only.
    import shutil
    local = os.path.join(TMP, "chain_specimen.ply")
    shutil.copyfile(r2, local)
    out = require(local, roi=roi, region=roi["region"], verbose=False,
                  line_spacing=LINE_SPACING, work_volume=WORK_VOLUME,
                  repair_to=os.path.join(TMP, "chain_specimen_gated.ply"))
    cr = ns["last_roi_crop_report"]
    check(out != local and cr["cropped"] and cr["faces_out"] < cr["faces_in"],
          "require_plannable_mesh(roi=...) crops the planning mesh with no extra "
          "argument: %d of %d faces, into %s"
          % (cr["faces_out"], cr["faces_in"], os.path.basename(str(out))))
    check(read_ply_mesh(out)[0].shape[0] < read_ply_mesh(local)[0].shape[0],
          "...and the file the planner will read really is the smaller one "
          "(%d vertices against %d)"
          % (read_ply_mesh(out)[0].shape[0], read_ply_mesh(local)[0].shape[0]))
    out2 = require(local, roi=roi, region=roi["region"], verbose=False,
                   crop_to_roi=False, line_spacing=LINE_SPACING,
                   work_volume=WORK_VOLUME)
    check(out2 == local and not ns["last_roi_crop_report"]["cropped"],
          "crop_to_roi=False is the way back to what this did before")

    # OVERSHOOT IS NOT A REFUSAL. An 18 x 60 mm ROI yields a 54.7 x 122.2 mm
    # mesh: most of it is outside the region, legitimately, because the Keyence
    # line is ~31 mm wide and every pass overruns the edge.
    v2, _n2, t2 = read_ply_mesh(r2)
    check(abs((v2[:, 0].max() - v2[:, 0].min())*1e3 - 54.7) < 0.5
          and abs((v2[:, 1].max() - v2[:, 1].min())*1e3 - 122.2) < 0.5,
          "run 2's mesh is %.1f x %.1f mm for an 18 x 60 mm ROI -- the raster "
          "overshoot" % ((v2[:, 0].max() - v2[:, 0].min())*1e3,
                         (v2[:, 1].max() - v2[:, 1].min())*1e3))
    tight = (0.819, 0.033, 0.837, 0.093)        # the 18 x 60 mm ROI
    rep = report(r2, region=tight, roi=False, line_spacing=LINE_SPACING,
                 work_volume=WORK_VOLUME)
    m = rep["measured"]
    check(m["mesh_outside_region"] > 0.8,
          "%.0f%% of it lies outside that region" % (m["mesh_outside_region"]*100.0,))
    check(levels(rep, "region_overshoot")["ok"]
          and "region_overshoot" not in rep["blocking"] + rep["fatal"],
          "...which is CROPPED, not refused: region_overshoot is a warning at "
          "worst and never blocks")
    check(m["region_coverage"] >= 0.9 and "region_coverage" not in rep["blocking"],
          "...while coverage of that region is %.0f%%, which is the number to "
          "look at" % (m["region_coverage"]*100.0,))
    wide = (0.780, -0.030, 0.900, 0.150)        # a region twice the mesh
    rep = report(r2, region=wide, roi=False, line_spacing=LINE_SPACING,
                 work_volume=WORK_VOLUME)
    check(not levels(rep, "region_coverage")["ok"]
          and "region_coverage" not in rep["blocking"] + rep["fatal"]
          and rep["measured"]["mesh_outside_region"] < 0.05,
          "and coverage genuinely short of the region WARNS without blocking "
          "(%.0f%% covered, %.0f%% outside) -- the two are still independent, "
          "and neither is a refusal"
          % (rep["measured"]["region_coverage"]*100.0,
             rep["measured"]["mesh_outside_region"]*100.0))

    # A STALE ROI IS LOUD. 300 mm away in x: nothing about the mesh is wrong, so
    # this is not MeshNotPlannable.
    stale = dict(roi, footprint=_G.footprint_encode(
        _G.footprint(sv + np.array([0.30, 0.0, 0.0]), tris=st)))
    try:
        crop(r2, os.path.join(TMP, "never.ply"), roi=stale, verbose=False)
        check(False, "a stale ROI raises RoiDoesNotFitMesh")
    except RoiDoesNotFitMesh as exc:
        check("do not describe the same surface" in str(exc)
              and "80 x 157" in str(exc),
              "a stale ROI raises RoiDoesNotFitMesh and names both extents: "
              "%s" % (str(exc)[:70],))
    except Exception as exc:
        check(False, "a stale ROI raises RoiDoesNotFitMesh, not %r" % (exc,))
    rep = report(r2, roi=stale, line_spacing=LINE_SPACING, work_volume=WORK_VOLUME)
    check("roi_mesh_agreement" in rep["blocking"] and rep["verdict"] == "refuse",
          "...and the gate blocks on roi_mesh_agreement rather than planning the "
          "whole mesh (%s)" % (rep["summary"],))


@_checked
def test_what_the_pipeline_hands_the_gate():
    print("\n--- what the PIPELINE hands the gate --------------------------------")
    # =============================================================================
    # _gate_planning_mesh() in mesh_scan.dpi is what every pipeline caller
    # reaches; everything above tests mesh_plan_report(), which it wraps. The
    # decision under test is which SURFACE the roughness and tilt statistics
    # measure when the caller gave a region and no explicit roi.
    #
    # Grading the whole mesh refuses a cylindrical specimen outright: 40.9% of a
    # 63 mm cylinder leans past 45 degrees, while the crown that is actually
    # going to be scanned is nearly flat. When plan_ec_path() has been told which
    # part it means, the gate must grade that part.
    import ast as _ast
    _ms = _ast.parse("def _b():\n" + "".join(
        "    " + l + "\n" for l in open(os.path.join(HERE, "mesh_scan.dpi")).read().splitlines()))
    _fn = [n for n in _ms.body[0].body
           if isinstance(n, _ast.FunctionDef) and n.name == "_gate_planning_mesh"]
    assert len(_fn) == 1, [f.name for f in _fn]
    _SEEN = {}
    def _fake_report(*a, **kw):
        return None                     # only its presence is checked
    def _fake_require(mesh, repair_to=None, **kw):
        _SEEN.clear()
        _SEEN.update(kw)
        return mesh
    def _gate_ns():
        ns = {"globals": None, "os": os,
              "mesh_plan_report": _fake_report,
              "require_plannable_mesh": _fake_require,
              "work_volume": lambda: (_ for _ in ()).throw(RuntimeError("none")),
              "cable_envelope_model": lambda: (_ for _ in ()).throw(RuntimeError("none")),
              "__builtins__": __builtins__}
        cfg = {"NOETHER_MESH_GATE": True, "NOETHER_MESH_GATE_REPAIR": False,
               "mesh_plan_report": _fake_report,
               "require_plannable_mesh": _fake_require,
               "EC_SCAN_REGION": None, "EC_MAX_TILT_DEG": 45.0}
        ns["globals"] = lambda: dict(cfg, **{k: v for k, v in ns.items()
                                             if not k.startswith("__")})
        exec(compile(_ast.Module(body=_fn, type_ignores=[]), "mesh_scan.dpi",
                     "exec"), ns)
        return ns["_gate_planning_mesh"]
    _gate_mesh = _gate_ns()
    REG = (0.770, -0.270, 0.885, -0.228)
    _gate_mesh("m.ply", region=REG)
    check(_SEEN.get("roi") == "region",
          "a region given and no roi: the gate grades THAT region (%r)"
          % (_SEEN.get("roi"),))
    check(tuple(_SEEN.get("region") or ()) == REG,
          "  and the region itself still reaches the report")
    _gate_mesh("m.ply", region=REG, roi=False)
    check(_SEEN.get("roi") is False,
          "roi=False still means the whole mesh, deliberately (%r)"
          % (_SEEN.get("roi"),))
    _gate_mesh("m.ply")
    check("roi" not in _SEEN,
          "no region and no roi: nothing is invented (%r)" % (_SEEN.get("roi"),))
    _explicit = {"mask": np.ones(4, bool)}
    _gate_mesh("m.ply", region=REG, roi=_explicit)
    check(_SEEN.get("roi") is _explicit,
          "an explicit roi beats the region it was given alongside")

@_checked
def test_the_recorder_is_closed_whatever_goes_wrong():
    print("\n--- the recorder is closed whatever goes wrong -----------------------")
    # =============================================================================
    # _reconstruct() opens a recording with start() and only stop() closes it. If
    # the try/finally begins BELOW the speed derivation, anything raising in
    # between leaves the recorder running -- after which /start_recording answers
    # "Already recording." for the rest of the session and the node needs a manual
    # restart. So start() must be the last statement before the try.
    _mr = _ast.parse("def _b():\n" + "".join(
        "    " + l + "\n" for l in open(os.path.join(HERE, "mesh_reconstruction.dpi")).read().splitlines()))
    def _find(node, name):
        for n in _ast.walk(node):
            if isinstance(n, _ast.FunctionDef) and n.name == name:
                return n
        return None
    _rec = _find(_mr, "_reconstruct")
    check(_rec is not None, "_reconstruct() is there to check")
    if _rec is not None:
        _starts = [n for n in _rec.body
                   if isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Call)
                   and getattr(n.value.func, "attr", None) == "start"]
        _tries = [n for n in _rec.body if isinstance(n, _ast.Try)]
        check(not _starts or not _tries or
              _starts[-1].lineno < _tries[0].lineno,
              "self.start() comes before the try that guarantees stop()")
        _stops = [n for n in _ast.walk(_rec)
                  if isinstance(n, _ast.Call)
                  and getattr(n.func, "attr", None) == "stop"]
        check(bool(_tries) and bool(_tries[0].finalbody),
              "  and that try has a finally")
        if _tries and _tries[0].finalbody:
            _fin = _ast.dump(_ast.Module(body=_tries[0].finalbody, type_ignores=[]))
            check("attr='stop'" in _fin,
                  "  whose finally is the stop() call")
            # Nothing that can raise may sit between start() and the try.
            _between = [n for n in _rec.body
                        if _starts and _tries
                        and _starts[-1].lineno < n.lineno < _tries[0].lineno]
            check(not _between,
                  "  and NOTHING sits between start() and the try (%d statement(s))"
                  % (len(_between),))

@_checked
def test_not_blind():
    """The gate ran, on a known-positive and a known-negative.

    Also checks that nothing here reads a geometry_msgs class: ensure_msgs() may
    hand this process a stub module another suite wrote, so "nothing reads it" is
    a claim worth checking rather than asserting in a comment.
    """
    print("\n--- the test is not blind -----------------------------------------")
    check(report.__code__.co_filename == "noether_path.dpi"
          and plannable.__code__.co_filename == "noether_path.dpi",
          "the subject came out of noether_path.dpi, not a stand-in")
    check(not any(isinstance(vv, type)
                  and getattr(vv, "__module__", "").startswith("geometry_msgs")
                  for vv in list(globals().values())),
          "this suite builds no ROS message, so no other suite's stub can "
          "decide what it measures")
    # KNOWN-POSITIVE AND KNOWN-NEGATIVE through the same call: the clean plate
    # plans and the NaN mesh -- the one defect that aborts the real planner -- is
    # refused. A gate that answered "plan" unconditionally would pass every other
    # check in this file that only looks at a good mesh.
    good_ok, good_why = plannable(good, line_spacing=LINE_SPACING,
                                  work_volume=WORK_VOLUME)
    nan_ok, nan_why = plannable(nanmesh, line_spacing=LINE_SPACING,
                                work_volume=WORK_VOLUME)
    check(good_ok and not nan_ok,
          "a clean plate plans (%s) and a mesh with a NaN vertex does not (%s)"
          % (good_why, nan_why))
    try:
        require(nanmesh, line_spacing=LINE_SPACING, work_volume=WORK_VOLUME)
    except MeshNotPlannable:
        check(True, "and require_plannable_mesh() raises on it rather than "
                    "handing it to the planner")
    except Exception as _e:
        check(False, "require_plannable_mesh() raised %s, not MeshNotPlannable"
                     % (type(_e).__name__,))
    else:
        check(False, "require_plannable_mesh() accepted a NaN mesh: the gate is "
                     "not being consulted")


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
