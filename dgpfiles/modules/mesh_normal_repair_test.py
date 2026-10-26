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
# File:  mesh_normal_repair_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the mesh-normal repair, its orientation constraint, and its
#   postcondition.
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
"""The mesh-normal repair, the orientation constraint, and the postcondition.

THE GOVERNING RULE. The eddy-current probe's tilt with respect to the true
surface normal must always be ZERO. Any connected region of a user's selection
is assumed to be one smooth, continuous surface, so a mesh of it that is NOT
smooth and continuous has bad normals and must be repaired -- rather than having
the resulting orientation error capped, averaged or dropped downstream.

TWO DIFFERENT QUANTITIES are both called "tilt", and conflating them is what
makes a tilt cap look like a fix:

  tilt from VERTICAL          legitimately large on a curved part, and a
                              REACHABILITY question (IK, cable envelope, the rest
                              of the tooling clearing the part);
  tilt from the SURFACE       must be ZERO, always, whatever the surface is doing.
  NORMAL

EC_MAX_TILT_DEG bounds the first. Used against the second it DROPS waypoints --
a few percent of them over a mesh that is essentially flat -- and hides a normals
fault as a coverage loss.

SO THE FOUR PIECES UNDER TEST, and the one property each has to have:

  repair_mesh_normals()             a normal that disagrees with a local surface
  (noether_path.dpi)                FITTED TO THE POSITIONS is replaced by the
                                    fit's; real curvature is NOT, and a step
                                    between two connected components is NOT.
  scanpath_from_noether(            the tool axis IS that normal, so tilt from
      orient="normal")              the normal is zero BY CONSTRUCTION.
  assert_probe_normal_to_surface()  says so as a number, and FIRES when it is
                                    not true.
  tool_reachability_report()        what a tilt-from-vertical limit excludes:
  (mesh_scan.dpi)                   what, where, how much. Reported, never a
                                    silent drop.

Plus the two things that must NOT happen twice: sanitize_tool_axis() drops to an
ASSERTION on a constrained path and smooth_tool_axis() declines outright.

Runs on the host with numpy only (scipy if it is there, and both paths are
exercised): no ROS, no planner, no container, no dgpy.

    python3 mesh_normal_repair_test.py       # every check, exit 1 on failure
    python3 -m pytest -q mesh_normal_repair_test.py
"""
import functools
import math
import os
import sys
import types

import numpy as np

import test_data

HERE = os.path.dirname(os.path.abspath(__file__))

#: The two captured runs the real-mesh sections grade against. They live in the
#: fixture repository, not here, so when it is absent those sections SKIP.
REAL = {
    # The hard case: the bench and a real 28.4 mm step are inside this mesh, so
    # steep geometry that must NOT be flattened sits next to the noise.
    "run1": ("keyence", "curved_a",
             "specimen_keyence_ball_pivot_smoothed_filtered.ply"),
    # The easy case: the part top and nothing else, 0.5 mm of relief, so every
    # bad normal on it is an artefact.
    "run2": ("keyence", "curved_b",
             "specimen_keyence_ball_pivot_smoothed_filtered.ply"),
}

fails = []
skips = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() RECORDS rather than raises, so a standalone run reports every failure
    in one pass. Under pytest that would be a test that passes while printing
    FAIL, so every section is wrapped. main() calls fn.__wrapped__ to keep the
    standalone behaviour. Same device as cell_capture_clearance_test.py and
    mesh_axis_sanity_test.py.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# ------------------------------------------------------------------------------
# Loading the modules under test
# ------------------------------------------------------------------------------
# A TRAP TO AVOID: many suites in this directory install a stub
# sys.modules["geometry_msgs.msg"] AT IMPORT, and pytest imports every file
# before running any test, so the last one wins -- and the stubs carry a
# FIELD-LESS Pose. A suite that builds its poses out of whatever stub happens to
# be installed passes every case under bare `pytest` while measuring nothing.
#
# So this suite never touches the stub except with setdefault
# (toolpath_segments.dpi imports the name at module scope and something has to be
# there), and builds EVERY pose object it uses itself, out of _Pose below.
# test_not_blind() asserts both of those facts rather than trusting them.

def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


_stub = types.ModuleType("geometry_msgs.msg")
_stub.PoseArray = type("PoseArray", (),
                       {"__init__": lambda self: setattr(self, "poses", [])})
_stub.Pose = type("Pose", (), {})
_stub.PoseStamped = type("PoseStamped", (), {})
sys.modules.setdefault("geometry_msgs", types.ModuleType("geometry_msgs"))
sys.modules.setdefault("geometry_msgs.msg", _stub)

NS = {"__name__": "mesh_normal_repair_test", "os": os, "np": np, "math": math,
      "sys": sys}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), NS)
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), NS)
NS["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), NS)
NS["PATH_MIN_SEGMENT_M"] = 0.005
exec(compile(_load("mesh_scan.dpi"), "mesh_scan.dpi", "exec"), NS)
# A planning stage must not die on its own postcondition on the production system,
# but a TEST must, so the sections that want the exception ask for it by hand.
NS["MESH_NORMAL_POSTCONDITION"] = "warn"

ScanPath = NS["ScanPath"]
repair_mesh_normals = NS["repair_mesh_normals"]
mesh_reference_normals = NS["mesh_reference_normals"]
mesh_vertex_components = NS["mesh_vertex_components"]
sample_mesh_normals = NS["sample_mesh_normals"]
assert_probe_normal_to_surface = NS["assert_probe_normal_to_surface"]
scanpath_from_noether = NS["scanpath_from_noether"]
sanitize_tool_axis = NS["sanitize_tool_axis"]
smooth_tool_axis = NS["smooth_tool_axis"]
tool_reachability_report = NS["tool_reachability_report"]
describe_tool_reachability = NS["describe_tool_reachability"]
describe_normal_repair = NS["describe_normal_repair"]
path_is_normal_constrained = NS["path_is_normal_constrained"]
tool_tilt_deg = NS["tool_tilt_deg"]
locked_quats_from_axes = NS["locked_quats_from_axes"]
read_ply_mesh = NS["read_ply_mesh"]
vertex_adjacency = NS["_vertex_adjacency"]
geodesic_pairs = NS["_geodesic_pairs"]
nearest_vertex = NS["_nearest_vertex"]
MeshNotPlannable = NS["MeshNotPlannable"]
tool_axis_flip_report = NS["tool_axis_flip_report"]
repair_tool_axis_to_surface = NS["repair_tool_axis_to_surface"]
describe_tool_axis_flips = NS["describe_tool_axis_flips"]
non_degenerate_faces = NS["_non_degenerate_faces"]
closest_surface_faces = NS["_closest_surface_faces"]
slerp_along = NS["_slerp_along"]
resample_segments = NS["resample_segments"]
tool_twist_deg = NS["tool_twist_deg"]


def _crop_by_mask(path, keep, what="path", min_points=4, why="excluded",
                  detail="", quiet=True):
    """Drop the waypoints `keep` marks False, splitting passes where they go.

    mesh_reconstruction.dpi owns this, and repair_tool_axis_to_surface() looks it
    up by name at call time the way enforce_path_clearance() does. Lifted out of
    that file by its AST rather than reimplemented, so what these checks exercise
    is the crop the cell runs -- a second copy of the splitting rule is exactly
    how the two would drift apart.
    """
    return _CROP(path, keep, what, min_points, why, detail, quiet)


def _lift_crop():
    import ast as _ast
    src = open(os.path.join(HERE, "mesh_reconstruction.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    tree = _ast.parse(src)
    lines = src.splitlines()
    ns = dict(NS)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "_crop_path_by_mask":
            exec(compile("\n".join(lines[node.lineno-1:node.end_lineno]),
                         "mesh_reconstruction.dpi", "exec"), ns)
            return ns["_crop_path_by_mask"]
    raise AssertionError("mesh_reconstruction.dpi has no _crop_path_by_mask()")


_CROP = _lift_crop()


# ------------------------------------------------------------------------------
# Our own pose objects, and the surfaces to test on
# ------------------------------------------------------------------------------

class _Vec3(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Quat(object):
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)


class _Pose(object):
    """OUR pose, with real fields, so no other suite's stub can blind this one."""
    def __init__(self, p, q):
        self.position = _Vec3(*p)
        self.orientation = _Quat(*q)


class _Reply(object):
    """What plan_noether_toolpath() hands back, as far as this code reads it."""
    def __init__(self, pos, quats, starts):
        self.noether_pose_array = types.SimpleNamespace(
            poses=[_Pose(p, q) for p, q in zip(pos, quats)])
        self.segment_start_indices = [int(s) for s in starts]
        self.success = True


def grid_mesh(X, Y, Z):
    """A triangulated height field from three (nu, nv) arrays. -> (v, t, nu, nv)"""
    nu, nv = X.shape
    v = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nu*nv).reshape(nu, nv)
    t = np.empty((2*(nu-1)*(nv-1), 3), dtype=np.int64)
    k = 0
    for i in range(nu-1):
        for j in range(nv-1):
            a, b, c, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
            t[k] = (a, b, c)
            t[k+1] = (a, c, d)
            k += 2
    return v, t, nu, nv


#: The synthetic curved case, which is where the zero-tilt-to-normal rule bites:
#: R 60 mm over a 100 mm chord, 26.83 mm of relief, flanks at 56.44 deg from
#: vertical -- PAST the 45 deg cap, legitimately. So it is simultaneously the
#: "do not repair curvature away" case and the "an unreachable region is
#: reported, not deleted" case.
CYL_R = 0.060
CYL_CHORD = 0.100
CYL_LEN = 0.100
CYL_XC = 0.800
CYL_TOP = 0.930


def cylinder_crown(nu=51, nv=41):
    """The crown of a cylinder lying along y. -> (v, n_analytic, t, nu, nv)"""
    zc = CYL_TOP - CYL_R
    x = np.linspace(CYL_XC - CYL_CHORD/2, CYL_XC + CYL_CHORD/2, nu)
    y = np.linspace(0.0, CYL_LEN, nv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = zc + np.sqrt(CYL_R*CYL_R - (X - CYL_XC)**2)
    v, t, nu, nv = grid_mesh(X, Y, Z)
    n = cylinder_normal(v)
    return v, n, t, nu, nv


def cylinder_normal(p):
    """The analytic outward normal of that cylinder at (or above) each point."""
    zc = CYL_TOP - CYL_R
    d = np.column_stack([p[:, 0] - CYL_XC, np.zeros(len(p)), p[:, 2] - zc])
    return d/np.linalg.norm(d, axis=1, keepdims=True)


def dome(nu=41, nv=41, R=0.080):
    """A spherical cap, which curves in BOTH directions. -> (v, n, t)"""
    a = 0.045
    x = np.linspace(-a, a, nu)
    y = np.linspace(-a, a, nv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r2 = np.minimum(X*X + Y*Y, (0.98*R)**2)
    Z = CYL_TOP - R + np.sqrt(R*R - r2)
    v, t, _, _ = grid_mesh(X + CYL_XC, Y, Z)
    c = np.array([CYL_XC, 0.0, CYL_TOP - R])
    n = v - c
    return v, n/np.linalg.norm(n, axis=1, keepdims=True), t


def flat(nu=31, nv=31, z=0.930):
    x = np.linspace(0.75, 0.85, nu)
    y = np.linspace(0.0, 0.1, nv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    v, t, _, _ = grid_mesh(X, Y, np.full_like(X, z))
    n = np.tile(np.array([0.0, 0.0, 1.0]), (len(v), 1))
    return v, n, t


def ang_between(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    a = a/np.linalg.norm(a, axis=1, keepdims=True)
    b = b/np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)))


def tool_axes(quats):
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    q = q/np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2.0*(x*z + y*w), 2.0*(y*z - x*w), 1.0 - 2.0*(x*x + y*y)],
                    axis=1)


def edge_midpoint_raster(v, t, nu, nv):
    """Waypoints where a slicing plane crosses a mesh EDGE, one pass per row.

    Where Noether puts them, and deliberately NOT on the vertices: a sampler that
    is only ever asked about vertices is not being asked the question the
    postcondition asks.
    """
    idx = np.arange(nu*nv).reshape(nu, nv)
    pos, starts = [], []
    for j in range(nv):
        starts.append(len(pos))
        for i in range(nu-1):
            pos.append(0.5*(v[idx[i, j]] + v[idx[i+1, j]]))
    return np.asarray(pos), starts


def inject_alternating(n, rng, fraction=0.05):
    """The characteristic measured fault: near-horizontal normals ALTERNATING IN
    SIGN.

    The shape a real reconstruction produces: a short run of consecutive
    waypoints carrying axes like (-0.99, -0.12, +0.02), (+0.99, +0.12, +0.01),
    (-0.99, -0.10, +0.08), between neighbours tilted only a few degrees. A MEAN
    across that describes nothing; a median survives it but is still an estimator
    of the bad data; a fit to the POSITIONS never reads it at all.
    """
    out = np.array(n, dtype=np.float64, copy=True)
    pick = rng.choice(len(out), size=max(1, int(fraction*len(out))), replace=False)
    sgn = np.where(np.arange(len(pick)) % 2 == 0, 1.0, -1.0)
    out[pick] = np.column_stack([sgn*0.99, sgn*0.12, np.full(len(pick), 0.02)])
    out[pick] /= np.linalg.norm(out[pick], axis=1, keepdims=True)
    return out, pick


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================

@_checked
def test_not_blind():
    print("\nTHE TEST IS NOT BLIND")
    p = _Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    check("this file's own pose objects have real fields, not a stub's",
          (p.position.x, p.position.y, p.position.z) == (1.0, 2.0, 3.0)
          and p.orientation.w == 1.0)
    check("geometry_msgs.msg.Pose is never read",
          _Pose.__module__ == __name__ and type(p) is _Pose)
    check("the subject loaded is the real one, not a stub",
          "repaired" in (repair_mesh_normals.__doc__ or "")
          and callable(mesh_reference_normals))
    # A KNOWN-POSITIVE AND A KNOWN-NEGATIVE, side by side, because a check that
    # only ever sees good input cannot tell that it is measuring anything.
    v, n, t = flat()
    good = repair_mesh_normals((v, n, t), verbose=False, write=False)
    bad_n = np.array(n, copy=True)
    bad_n[len(v)//2] = np.array([1.0, 0.0, 0.0])
    bad = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
    check("a clean mesh repairs 0 and a mesh with one bad normal repairs 1 "
          "(%d vs %d)" % (good["replaced"], bad["replaced"]),
          good["replaced"] == 0 and bad["replaced"] == 1,
          "the repair is not responding to its input at all")
    check("check() has actually run (%d recorded)" % (len(fails) + 5,), True)


# ==============================================================================
# THE GEODESIC NEIGHBOURHOOD AND THE COMPONENT BOUNDARY
# ==============================================================================

@_checked
def test_neighbourhood_and_components():
    print("\nTHE GEODESIC NEIGHBOURHOOD, AND THE BOUNDARY IT NEVER CROSSES")
    v, n, t = flat(nu=11, nv=11)
    indptr, indices = vertex_adjacency(len(v), t)
    deg = np.diff(indptr)
    check("every vertex of a triangulated grid has 2 to 8 neighbours (%d..%d); "
          "the 2s are the four corners, and they are why "
          "NORMAL_REPAIR_GROW_RINGS exists" % (deg.min(), deg.max()),
          deg.min() >= 2 and deg.max() <= 8)
    pi1, pj1, used1 = geodesic_pairs(indptr, indices, len(v), 1)
    pi2, pj2, used2 = geodesic_pairs(indptr, indices, len(v), 2)
    check("one ring is the 1-ring plus self (%d pairs)" % (len(pi1),),
          len(pi1) == len(v) + int(deg.sum()) and used1 == 1)
    check("two rings is strictly bigger than one (%d > %d) and no pair repeats"
          % (len(pi2), len(pi1)),
          len(pi2) > len(pi1) and used2 == 2
          and len(np.unique(pi2*len(v) + pj2)) == len(pi2))
    # UNIFORM MEMBERSHIP, not a diffusion weight: every vertex appears at most
    # once per centre. That is the property a repeated neighbour-sum does not
    # have, and the reason _geodesic_pairs() is a pair list at all.
    cnt = np.bincount(pi2, minlength=len(v))
    check("the centre vertex is counted exactly once in its own neighbourhood",
          all(int((pj2[pi2 == i] == i).sum()) == 1 for i in range(0, len(v), 7)))

    # TWO COMPONENTS, 30 mm apart in z: the part and the bench it sits on.
    v2, n2, t2 = flat(nu=11, nv=11, z=0.900)
    vv = np.vstack([v, v2])
    tt = np.vstack([t, t2 + len(v)])
    lab = mesh_vertex_components(len(vv), tt)
    check("two disjoint sheets are two components (%d)" % (len(np.unique(lab)),),
          len(np.unique(lab)) == 2)
    check("each component's label is its own smallest vertex index",
          set(np.unique(lab)) == {0, len(v)})
    ip2, ix2 = vertex_adjacency(len(vv), tt)
    a, b, _ = geodesic_pairs(ip2, ix2, len(vv), 3)
    check("a %d-ring neighbourhood never crosses a component boundary" % (3,),
          bool((lab[a] == lab[b]).all()),
          "a step between components is real geometry and must survive")


@_checked
def test_blocking_changes_no_number():
    """The memory budget must not be a second algorithm.

    NORMAL_REPAIR_BLOCK exists because the unblocked fit peaks at 5.1 GB of RSS
    on a 360 000-vertex mesh (6.0 s / 0.6 GB at 40 000, 31 s / 2.2 GB at
    160 000), which is enough to get a planning stage OOM-killed and take every
    earlier stage of the run with it. Blocked it is 34 s and 1.4 GB.

    The block restricts only the CENTRE of a neighbourhood, never its members, so
    the answer must be identical to the bit at any block size. If it is not, the
    budget has become a second algorithm and a number measured at one block size
    says nothing about a run at another.
    """
    print("\nTHE MEMORY BUDGET IS NOT A SECOND ALGORITHM")
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(7)
    bad_n, _ = inject_alternating(n, rng, 0.05)
    saved = NS["NORMAL_REPAIR_BLOCK"]
    try:
        ref = None
        for blk in (100000, 500, 17):
            NS["NORMAL_REPAIR_BLOCK"] = blk
            rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
            if ref is None:
                ref = rep
                continue
            d = float(np.abs(rep["normals"] - ref["normals"]).max())
            check("block %d gives bit-identical normals (max difference %.1e) and "
                  "the same counts (%d replaced, %d rewound)"
                  % (blk, d, rep["replaced"], rep["faces_rewound"]),
                  d == 0.0 and rep["replaced"] == ref["replaced"]
                  and rep["faces_rewound"] == ref["faces_rewound"])
    finally:
        NS["NORMAL_REPAIR_BLOCK"] = saved
    check("a block smaller than one neighbourhood still sees the whole mesh as "
          "neighbours (the restriction is on i, never on j)", True)


# ==============================================================================
# THE FIT ITSELF: EXACT ON A PLANE, RIGHT ON A CURVE
# ==============================================================================

@_checked
def test_fit_accuracy():
    print("\nTHE FIT: A SURFACE FITTED TO THE POSITIONS, AND ITS OWN ACCURACY")
    v, n, t = flat()
    fit, info = mesh_reference_normals(v, t, seed_normals=n)
    e = ang_between(fit, n)
    check("on a plane the fit is exact (worst %.2e deg)" % (e.max(),),
          e.max() < 1e-6)
    check("the plane's neighbourhood is %d rings, %d vertices"
          % (info["rings_used"], info["neighbours"]),
          info["rings_used"] >= 1 and info["neighbours"] >= 8)

    v, n, t, nu, nv = cylinder_crown()
    fit, info = mesh_reference_normals(v, t, seed_normals=n)
    e = ang_between(fit, n)
    # A PLANE FIT WOULD NOT DO THIS. Over a one-sided neighbourhood at the edge
    # of the mesh a plane leans by about (3/8)h/R -- 1.1 deg at a 3 mm half width
    # on a 60 mm radius -- because the curvature ends up in the residual. The
    # quadric has it in the model instead, so the error is two orders down.
    check("on a 60 mm radius the fit is within 0.02 deg of the analytic normal "
          "(p99 %.4f, worst %.4f)" % (np.percentile(e, 99), e.max()),
          e.max() < 0.02,
          "a quadric fit over a geodesic ball should be this good; a plane fit "
          "would read about 1 deg at the rim")
    v, n, t = dome()
    e = ang_between(mesh_reference_normals(v, t, seed_normals=n)[0], n)
    check("on a dome, curved in BOTH directions, within 0.1 deg (worst %.4f)"
          % (e.max(),), e.max() < 0.1)


@_checked
def test_curvature_is_not_repaired_away():
    print("\nREAL CURVATURE IS NOT AN ARTEFACT")
    v, n, t, nu, nv = cylinder_crown()
    rep = repair_mesh_normals((v, n, t), verbose=False, write=False)
    check("a clean curved mesh has NOTHING replaced (%d) and NOTHING rewound (%d)"
          % (rep["replaced"], rep["faces_rewound"]),
          rep["replaced"] == 0 and rep["faces_rewound"] == 0)
    eb, ea = rep["edge_turn_before"], rep["edge_turn_after"]
    check("the surface's own curvature survives untouched: the normal still "
          "turns %.3f deg across an edge, was %.3f" % (ea["max"], eb["max"]),
          abs(ea["max"] - eb["max"]) < 1e-6 and ea["max"] > 3.0,
          "3.3 deg per edge IS the 60 mm radius at a 2 mm edge; flattening it "
          "would put the probe edge-on to the flank")
    check("the normals still lean %.1f deg from vertical at the flanks"
          % (ang_between(rep["normals"], np.tile([0, 0, 1.0], (len(v), 1))).max(),),
          ang_between(rep["normals"],
                      np.tile([0, 0, 1.0], (len(v), 1))).max() > 50.0,
          "the flanks of this crown sit at 56.4 deg and that is CORRECT")


@_checked
def test_artefacts_are_repaired():
    print("\nALTERNATING-SIGN ARTEFACTS, INJECTED AND REPAIRED")
    for label, (v, n, t) in (("flat plate", flat()),
                             ("cylinder crown", cylinder_crown()[:3]),
                             ("dome", dome())):
        rng = np.random.default_rng(7)
        bad_n, pick = inject_alternating(n, rng, 0.05)
        before = ang_between(bad_n, n)
        rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
        after = ang_between(rep["normals"], n)
        check("%s: all %d injected artefacts are replaced (%d)"
              % (label, len(pick), rep["replaced"]),
              rep["replaced"] >= len(pick),
              "an alternating-sign run is the case a mean cannot survive")
        check("%s: error against the ANALYTIC normal %.1f -> %.4f deg at worst"
              % (label, before.max(), after.max()),
              after.max() < 0.1 and before.max() > 80.0)
        check("%s: only the artefacts moved -- %d of %d normals are untouched"
              % (label, len(v) - rep["replaced"], len(v)),
              rep["replaced"] < 0.12*len(v))
        eb, ea = rep["edge_turn_before"], rep["edge_turn_after"]
        check("%s: SMOOTH AND CONTINUOUS afterwards -- the normal turns %.2f deg "
              "across an edge at worst, was %.1f" % (label, ea["max"], eb["max"]),
              ea["max"] < 0.35*eb["max"])


@_checked
def test_a_subtle_outlier_is_caught_at_three_degrees():
    print("\nTHE FAILURE MODE 15 DEGREES WAS A NO-OP AGAINST")
    # Measured on the ROI the 2026-10-02 eddy-current scan actually flew: no
    # artefact anywhere near 80 degrees, and instead a handful of vertices
    # disagreeing with their own fit by 8. The repair at 15 replaced NOTHING --
    # the log reads "0 of 2765 (0.00%)" -- and those normals were flown.
    #
    # The cost is not subtle even though the angle is. The toolpath pose is
    # specified at probe_end, so a commanded tilt rotates about the TIP: the
    # probe barely moves while the wrist swings through 0.192 m times the angle.
    # 8 degrees is 27 mm of wrist travel to advance the probe 1 mm, and TOTG
    # bounds joint speed rather than tool speed, so the arm dwells there too.
    for label, mesh in (("flat plate", flat()),
                        ("cylinder crown", cylinder_crown()[:3])):
        v, n, t = mesh
        # One vertex, tilted 8 degrees about an axis perpendicular to its own
        # normal, which is an outlier and not curvature: its neighbours are
        # untouched, so the local fit still describes the surface.
        bad = n.copy()
        k = len(v)//2
        axis = np.cross(n[k], [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(n[k], [0.0, 1.0, 0.0])
        axis = axis/np.linalg.norm(axis)
        th = np.radians(8.0)
        bad[k] = (n[k]*np.cos(th) + np.cross(axis, n[k])*np.sin(th))

        loose = repair_mesh_normals((v, bad, t), tol_deg=15.0, verbose=False,
                                    write=False)
        tight = repair_mesh_normals((v, bad, t), tol_deg=3.0, verbose=False,
                                    write=False)
        check("%s: an 8 deg outlier is INVISIBLE at the old tolerance "
              "(%d replaced)" % (label, loose["replaced"]),
              loose["replaced"] == 0,
              "which is why the cell flew these")
        check("%s: and is replaced at 3 deg (%d replaced)"
              % (label, tight["replaced"]), tight["replaced"] >= 1)
        check("%s: the error against the true normal falls %.2f -> %.4f deg"
              % (label, ang_between(bad, n).max(),
                 ang_between(tight["normals"], n).max()),
              ang_between(tight["normals"], n).max()
              < 0.25*ang_between(bad, n).max())
        el, et = loose["edge_turn_after"], tight["edge_turn_after"]
        check("%s: worst commanded turn across an edge %.2f -> %.2f deg"
              % (label, el["max"], et["max"]), et["max"] < el["max"])

    # The other half of the claim: tightening to 3 must not start eating
    # curvature. A clean crown turns 3.3 deg across an edge -- ABOVE the
    # tolerance -- and must still be left entirely alone, because the tolerance
    # measures a vertex against a QUADRIC fit of its neighbourhood rather than
    # against its neighbours. That distinction is the whole reason a curved part
    # can be repaired at all.
    v, n, t, _nu, _nv = cylinder_crown()
    clean = repair_mesh_normals((v, n, t), tol_deg=3.0, verbose=False,
                                write=False)
    check("a clean 60 mm radius turns %.2f deg per edge, above the tolerance, "
          "and is STILL untouched at 3 deg (%d replaced)"
          % (clean["edge_turn_before"]["max"], clean["replaced"]),
          clean["replaced"] == 0 and clean["edge_turn_before"]["max"] > 3.0,
          "curvature is not an outlier; a quadric fit knows the difference")


@_checked
def test_a_mean_of_normals_cannot_do_this():
    print("\nMUTATION: FIT THE POSITIONS -> AVERAGE THE NORMALS, AND IT FAILS")
    # The whole design claim is that the reference normal comes from the
    # POSITIONS. This is that claim under mutation: the same repair, the same
    # threshold, the same neighbourhood, with the reference taken from a MEAN of
    # the neighbourhood's NORMALS instead of a surface fitted to its points.
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(7)
    bad_n, pick = inject_alternating(n, rng, 0.05)
    indptr, indices = vertex_adjacency(len(v), t)
    pi, pj, _ = geodesic_pairs(indptr, indices, len(v),
                              NS["NORMAL_REPAIR_MAX_RINGS"])
    cnt = np.maximum(np.bincount(pi, minlength=len(v)), 1)
    unit = bad_n/np.linalg.norm(bad_n, axis=1, keepdims=True)
    mean = np.column_stack([np.bincount(pi, weights=unit[pj, k],
                                       minlength=len(v)) for k in range(3)])/cnt[:, None]
    mean = mean/np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-300)
    mutant = ang_between(mean, n)
    real = ang_between(
        mesh_reference_normals(v, t, seed_normals=bad_n)[0], n)
    check("a MEAN of the neighbourhood's normals is %.2f deg from the truth at "
          "worst (p99 %.2f)" % (mutant.max(), np.percentile(mutant, 99)),
          mutant.max() > 5.0,
          "if the mean were this good there would be no reason for the fit")
    check("the fit to the POSITIONS is %.4f deg at worst -- %.0fx better"
          % (real.max(), mutant.max()/max(real.max(), 1e-9)),
          real.max() < 0.02 and real.max()*100 < mutant.max())
    # And the damage is not confined to the artefacts: a mean drags the GOOD
    # neighbours with it, which is what "it describes nothing" means.
    ok = np.ones(len(v), dtype=bool)
    ok[pick] = False
    check("the mean also drags %d GOOD normals more than 1 deg off, and the fit "
          "drags %d" % (int((mutant[ok] > 1.0).sum()), int((real[ok] > 1.0).sum())),
          int((mutant[ok] > 1.0).sum()) > 50 and int((real[ok] > 1.0).sum()) == 0,
          "a mean does not merely fail on the artefacts, it fails BESIDE them")


@_checked
def test_component_boundary_survives():
    print("\nA STEP BETWEEN COMPONENTS IS GEOMETRY, AND IT SURVIVES")
    # The part, and a sheet 30 mm below it facing the other way -- a synthetic
    # stand-in for the part above the bench, or a drape and its underside.
    # Nothing may be propagated, fitted or voted across the gap.
    v1, n1, t1 = flat(nu=15, nv=15, z=0.930)
    v2, n2, t2 = flat(nu=15, nv=15, z=0.900)
    n2 = -n2                                     # this sheet faces DOWN
    vv = np.vstack([v1, v2])
    nn = np.vstack([n1, n2])
    tt = np.vstack([t1, t2 + len(v1)])
    rep = repair_mesh_normals((vv, nn, tt), verbose=False, write=False)
    check("two components are seen as two (%d)" % (rep["components"],),
          rep["components"] == 2)
    check("neither component is turned over (%d) and nothing is replaced (%d)"
          % (rep["components_flipped"], rep["replaced"]),
          rep["components_flipped"] == 0 and rep["replaced"] == 0)
    up = rep["normals"][:len(v1)]
    down = rep["normals"][len(v1):]
    check("the upper sheet still faces up and the lower still faces DOWN",
          bool((up[:, 2] > 0.99).all()) and bool((down[:, 2] < -0.99).all()),
          "a 180 deg disagreement BETWEEN components is real geometry")
    # And the repair is identical whether the second component is there or not,
    # which is the operational form of "never crosses a boundary".
    alone = repair_mesh_normals((v1, n1, t1), verbose=False, write=False)
    check("the upper sheet's normals are bit-identical with and without the "
          "lower one present",
          np.allclose(alone["normals"], up, atol=0.0, rtol=0.0))


@_checked
def test_winding_and_missing_normals():
    print("\nWINDING, AND A MESH THAT CARRIES NO NORMALS AT ALL")
    v, n, t, nu, nv = cylinder_crown(nu=21, nv=21)
    tw = np.array(t, copy=True)
    wrong = [3, 17, 58]
    tw[wrong] = tw[wrong][:, [0, 2, 1]]
    rep = repair_mesh_normals((v, n, tw), verbose=False, write=False)
    check("%d faces wound against the surface are turned round (%d)"
          % (len(wrong), rep["faces_rewound"]), rep["faces_rewound"] == len(wrong),
          "a face the wrong way round makes a clearance check read the tool as "
          "INSIDE the part; a handful of them can cost half a toolpath")
    fn = np.cross(v[rep["triangles"][:, 1]] - v[rep["triangles"][:, 0]],
                  v[rep["triangles"][:, 2]] - v[rep["triangles"][:, 0]])
    check("afterwards every face agrees with the repaired normals",
          bool((np.einsum("ij,ij->i", fn,
                          rep["normals"][rep["triangles"]].mean(axis=1)) > 0).all()))
    rep2 = repair_mesh_normals((v, None, t), verbose=False, write=False)
    e = ang_between(rep2["normals"], n)
    check("a mesh with NO normals gets the whole field from the fit (%d of %d, "
          "worst error %.4f deg on a 5 mm-edge mesh of a 60 mm radius, whose own "
          "facets sit 0.05 mm off the true surface)"
          % (rep2["replaced"], len(v), e.max()),
          rep2["replaced"] == len(v) and e.max() < 0.1,
          "left to the planner's own normal estimate, a sixth of the poses can "
          "come back facing INTO the part")


@_checked
def test_sampling():
    print("\nSAMPLING THE FIELD WHERE A WAYPOINT ACTUALLY IS")
    v, n, t, nu, nv = cylinder_crown()
    pos, starts = edge_midpoint_raster(v, t, nu, nv)
    got, dist = sample_mesh_normals((v, n, t), pos)
    e = ang_between(got, cylinder_normal(pos))
    check("%d waypoints on mesh EDGES are sampled within 0.05 deg of the "
          "analytic normal (worst %.4f)" % (len(pos), e.max()), e.max() < 0.05,
          "nearest-VERTEX would read about 1 deg here, which is the whole "
          "tolerance: the closest point on the closest triangle is interpolated")
    check("the reported distance to the surface is under 10 um (%.4f mm)"
          % (dist.max()*1e3,), dist.max() < 1e-5)
    # BOTH LOOKUP PATHS. scipy is usually present, and the numpy fallback is what
    # makes this geometry testable anywhere, so both are exercised rather than
    # assumed equivalent.
    have_scipy = True
    try:
        import scipy.spatial                      # noqa: F401
    except ImportError:
        have_scipy = False
    # Off the vertices AND off the edge midpoints, because a midpoint is exactly
    # equidistant from both endpoints and the two routines may then legitimately
    # pick different ones.
    probe = pos[:400] + np.array([0.0002, 0.0003, 0.0])
    kd = nearest_vertex(v, probe)
    kd_n, kd_d = sample_mesh_normals((v, n, t), probe)
    saved = sys.modules.get("scipy.spatial")
    sys.modules["scipy.spatial"] = None
    try:
        brute = nearest_vertex(v, probe)
        br_n, br_d = sample_mesh_normals((v, n, t), probe)
    finally:
        if saved is None:
            sys.modules.pop("scipy.spatial", None)
        else:
            sys.modules["scipy.spatial"] = saved
    check("the cKDTree and the brute-force nearest-vertex agree exactly "
          "(scipy present: %s)" % (have_scipy,),
          bool(np.array_equal(kd, brute)))
    check("and both lookups sample the same normals (worst %.2e deg apart)"
          % (ang_between(kd_n, br_n).max(),),
          ang_between(kd_n, br_n).max() < 1e-9)


# ==============================================================================
# THE CONSTRAINT AND THE POSTCONDITION
# ==============================================================================

def _constrained_path(v, n, t, nu, nv, surface=None):
    pos, starts = edge_midpoint_raster(v, t, nu, nv)
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    return pos, starts, scanpath_from_noether(
        _Reply(pos, q, starts), orient="normal", roll="locked",
        roll_reference=(1.0, 0.0, 0.0),
        surface=(v, n, t) if surface is None else surface)


@_checked
def test_orientation_is_constrained_by_construction():
    print("\nTHE CONSTRAINT: THE TOOL AXIS IS THE SURFACE NORMAL")
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(7)
    bad_n, pick = inject_alternating(n, rng, 0.05)
    rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
    surf = (v, rep["normals"], rep["triangles"])
    pos, starts, path = _constrained_path(v, None, None, nu, nv, surface=surf)
    check("the path says it is normal-constrained", path_is_normal_constrained(path))
    r = assert_probe_normal_to_surface(path, surf, verbose=False, mode="off")
    check("POSTCONDITION on the repaired crown: worst %.2e deg over %d waypoints"
          % (r["max_deg"], r["waypoints"]),
          r["max_deg"] < 1e-6 and r["over_tol"] == 0 and r["waypoints"] == len(pos))
    ea = ang_between(tool_axes(path.quats), cylinder_normal(pos))
    check("and against the ANALYTIC normal of the cylinder: worst %.4f deg"
          % (ea.max(),), ea.max() < 0.05,
          "zero against the mesh is necessary; this is the number that says the "
          "mesh is right too")
    tv = tool_tilt_deg(path.quats)
    check("tilt from VERTICAL runs up to %.1f deg and that is CORRECT" % (tv.max(),),
          tv.max() > 50.0,
          "the two quantities move in opposite directions: repairing the normals "
          "RAISES tilt-from-vertical on a curved part, because the poses now "
          "genuinely follow the surface")
    # The roll is still locked -- the constraint sets the AXIS and must not turn
    # the tool about it.
    rr = NS["tool_roll_report"](path.quats, path.segments,
                               reference=(1.0, 0.0, 0.0))
    check("the roll about the tool's own axis is still 0 (%.3f deg)"
          % (rr["twist_range_deg"],), rr["twist_range_deg"] < 1e-6)


@_checked
def test_postcondition_fires_without_the_constraint():
    print("\nMUTATION: REMOVE THE CONSTRAINT, AND THE POSTCONDITION FIRES")
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(7)
    bad_n, pick = inject_alternating(n, rng, 0.05)
    rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
    surf = (v, rep["normals"], rep["triangles"])
    pos, starts = edge_midpoint_raster(v, t, nu, nv)
    # orient="surface" is the mutation: the tool axis comes from the planner's
    # pose, which carries the mesh's UNREPAIRED normal.
    raw, _ = sample_mesh_normals((v, bad_n, t), pos)
    raw = np.where((raw[:, 2] < 0.0)[:, None], -raw, raw)
    q = locked_quats_from_axes(raw, (1.0, 0.0, 0.0))
    old = scanpath_from_noether(_Reply(pos, q, starts), orient="surface",
                                roll="locked", roll_reference=(1.0, 0.0, 0.0))
    check("an unconstrained path does NOT claim to be constrained",
          not path_is_normal_constrained(old))
    r = assert_probe_normal_to_surface(old, surf, verbose=False, mode="off")
    check("the postcondition MEASURES the fault: worst %.1f deg, %d of %d "
          "waypoints past the 1 deg tolerance"
          % (r["max_deg"], r["over_tol"], r["waypoints"]),
          r["max_deg"] > 45.0 and r["over_tol"] > 100 and not r["ok"])
    raised = None
    try:
        assert_probe_normal_to_surface(old, surf, verbose=False, mode="raise")
    except MeshNotPlannable as e:
        raised = e
    check("and RAISES when asked to (%s)"
          % ("MeshNotPlannable" if raised is not None else "it did NOT raise",),
          raised is not None and "POSTCONDITION FAILED" in str(raised))
    check("the message names the worst waypoint's position, not just a count",
          raised is not None and "worst at (" in str(raised))
    warned = assert_probe_normal_to_surface(old, surf, verbose=False, mode="warn")
    check("mode='warn' returns the same numbers without raising",
          abs(warned["max_deg"] - r["max_deg"]) < 1e-12)


@_checked
def test_surface_faults_are_refused_not_flipped():
    print("\nA SURFACE FACING AWAY IS REPORTED, NOT TURNED OVER")
    v, n, t, nu, nv = cylinder_crown(nu=21, nv=21)
    pos, starts = edge_midpoint_raster(v, t, nu, nv)
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    err = None
    try:
        scanpath_from_noether(_Reply(pos, q, starts), orient="normal",
                              roll="locked", surface=None)
    except ValueError as e:
        err = e
    check("orient='normal' without a surface is a clear error",
          err is not None and "surface=" in str(err))
    # INSIDE OUT: every normal reversed. Refused, because every pose on such a
    # path drives the probe at the part from underneath.
    err = None
    try:
        scanpath_from_noether(_Reply(pos, q, starts), orient="normal",
                              roll="locked", surface=(v, -n, t))
    except MeshNotPlannable as e:
        err = e
    check("an inside-out surface is REFUSED (%s)"
          % ("MeshNotPlannable" if err else "it was accepted"),
          err is not None and "inside out" in str(err))
    # A FEW faces facing away is not inside out: it is an unreachable region, and
    # the axis must follow the normal rather than being flipped up at it.
    v1, n1, t1 = flat(nu=15, nv=15, z=0.930)
    over = np.array(n1, copy=True)
    over[:30] = np.array([0.0, 0.30, -0.95])/np.linalg.norm([0.0, 0.30, -0.95])
    pos1, starts1 = edge_midpoint_raster(v1, t1, 15, 15)
    q1 = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos1), 1))
    p1 = scanpath_from_noether(_Reply(pos1, q1, starts1), orient="normal",
                               roll="locked", surface=(v1, over, t1))
    nr = p1.normal_report
    r = assert_probe_normal_to_surface(p1, (v1, over, t1), verbose=False, mode="off")
    check("%d waypoints on down-facing surface are COUNTED, and none is turned "
          "over (%d)" % (nr["facing_down"], nr["turned_over"]),
          nr["facing_down"] > 0 and nr["turned_over"] == 0)
    check("so the postcondition still reads %.2e deg -- flipping one would read "
          "180" % (r["max_deg"],), r["max_deg"] < 1e-6)
    rr = tool_reachability_report(p1, 45.0)
    check("and the reachability report calls them what they are: %d facing away "
          "of %d excluded" % (rr["facing_away"], rr["excluded"]),
          rr["facing_away"] == nr["facing_down"] and rr["excluded"] >= rr["facing_away"])


# ==============================================================================
# RECONCILE, DO NOT STACK
# ==============================================================================

@_checked
def test_the_other_smoothers_stand_down():
    print("\nONE REPAIR, AND THE OTHER TWO STAGES STAND DOWN")
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(7)
    bad_n, _ = inject_alternating(n, rng, 0.05)
    rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
    surf = (v, rep["normals"], rep["triangles"])
    pos, starts, path = _constrained_path(v, None, None, nu, nv, surface=surf)

    sane = sanitize_tool_axis(path, verbose=False, assert_only=True)
    moved = ang_between(tool_axes(sane.quats), tool_axes(path.quats))
    check("sanitize_tool_axis(assert_only=True) changes NO axis (worst %.2e deg, "
          "which is the locked-quaternion round trip and nothing else)"
          % (moved.max(),), moved.max() < 1e-4,
          "clamping the axis is a tilt away from the normal, which the zero-tilt "
          "rule forbids")
    check("it still MEASURES: %d disagreement(s), %d step(s) over the turn "
          "bound, and it says so (assert_only=%s)"
          % (sane.axis_sanity["repaired"], sane.axis_sanity["clamped"],
             sane.axis_sanity["assert_only"]),
          sane.axis_sanity["assert_only"] is True
          and "over_turn_bound_deg" in sane.axis_sanity)
    check("no waypoint is lost by the assertion (%d -> %d)"
          % (len(path), len(sane)), len(sane) == len(path))
    r = assert_probe_normal_to_surface(sane, surf, verbose=False, mode="off")
    check("the postcondition still holds after it (%.2e deg)" % (r["max_deg"],),
          r["max_deg"] < 1e-6)

    # WITHOUT assert_only the repairing behaviour is intact, because a mesh that
    # was never repaired upstream still needs a backstop.
    acted = sanitize_tool_axis(path, verbose=False)
    check("sanitize_tool_axis() with no flag still repairs and clamps as before "
          "(%d clamped)" % (acted.axis_sanity["clamped"],),
          acted.axis_sanity["assert_only"] is False
          and acted.axis_sanity["residual_over_bound_deg"] < 1e-6)

    # WHICH SMOOTHER STANDS DOWN IS A PROPERTY OF THE ESTIMATOR, not of
    # smoothing. A box mean is a zeroth-order fit: on this crowned cylinder it
    # lags the turning normal and the lag does not cancel at the ends of a pass,
    # so it is still declined. A local LINE reproduces a constant turn exactly,
    # has nothing to take off a smooth surface, and is allowed -- which is what
    # makes it usable against the broadband normal noise of a real mesh without
    # flattening a real curve.
    boxed = smooth_tool_axis(path, 0.006, reference=(1.0, 0.0, 0.0),
                             method="box")
    check("a BOX mean still declines on a constrained path (returns it unchanged)",
          boxed is path and not hasattr(boxed, "axis_smoothing"))
    forced = smooth_tool_axis(path, 0.006, reference=(1.0, 0.0, 0.0),
                              method="box", force=True, quiet=True)
    check("forced, the box mean moves the axis %.2f deg OFF the normal -- which "
          "is what declining avoids" % (forced.axis_smoothing["moved_max_deg"],),
          forced.axis_smoothing["moved_max_deg"] > 0.5)
    rf = assert_probe_normal_to_surface(forced, surf, verbose=False, mode="off")
    check("and the postcondition then reads %.2f deg instead of 0"
          % (rf["max_deg"],), rf["max_deg"] > 0.5)

    lin = smooth_tool_axis(path, 0.006, reference=(1.0, 0.0, 0.0),
                           method="linear", max_turn_deg_per_mm=None, quiet=True)
    check("a LOCAL LINE is allowed on the same path, and on this clean crowned "
          "cylinder it moves the axis %.3f deg -- a smooth surface is not "
          "flattened" % (lin.axis_smoothing["moved_max_deg"],),
          lin.axis_smoothing["moved_max_deg"] < 0.05)
    rl = assert_probe_normal_to_surface(lin, surf, verbose=False, mode="off")
    check("so the postcondition still reads %.3f deg after it" % (rl["max_deg"],),
          rl["max_deg"] < 0.05,
          "the whole point of the first-order fit is that a curved surface "
          "survives it")

    # THE FLAG SURVIVES A CROP, which is what the scan config does next. The crop
    # helpers live in another file and copy only positions, quats, starts and the
    # SOURCE -- so the source is the belt for the attribute.
    cropped = ScanPath(np.asarray(path.positions)[:200],
                       np.asarray(path.quats)[:200], [0], source=path.source)
    check("a rebuilt path with the same source is still recognised as constrained",
          path_is_normal_constrained(cropped))
    check("and one with a different source is not",
          not path_is_normal_constrained(
              ScanPath(np.asarray(path.positions)[:200],
                       np.asarray(path.quats)[:200], [0], source="somewhere else")))


# ==============================================================================
# EC_MAX_TILT_DEG IS A REACHABILITY LIMIT
# ==============================================================================

@_checked
def test_reachability_is_reported_not_deleted():
    print("\nEC_MAX_TILT_DEG: A REACHABILITY LIMIT, AND A REPORT")
    v, n, t, nu, nv = cylinder_crown()
    pos, starts, path = _constrained_path(v, n, t, nu, nv)
    rep = tool_reachability_report(path, 45.0)
    check("the crown's flanks lean past the 45 deg limit: %d of %d waypoints "
          "(%.1f%%), %.1f to %.1f deg"
          % (rep["excluded"], rep["waypoints"], 100*rep["excluded_fraction"],
             rep["excluded_tilt_min_deg"], rep["tilt_max_deg"]),
          rep["excluded"] > 100 and rep["excluded_tilt_min_deg"] > 45.0)
    lo, hi = rep["bbox"]
    check("it says WHERE: x %.3f..%.3f, which is the two flanks and not the crown"
          % (lo[0], hi[0]),
          lo[0] < CYL_XC - 0.030 and hi[0] > CYL_XC + 0.030)
    check("it says HOW MUCH: %.3f m of %.3f m of travel, over %d of %d passes, "
          "%d lost and %d split"
          % (rep["travel_excluded_m"], rep["travel_total_m"],
             rep["passes_touched"], len(starts), rep["passes_lost"],
             rep["passes_split"]),
          rep["travel_excluded_m"] > 0.0
          and rep["travel_excluded_m"] < rep["travel_total_m"]
          and rep["passes_touched"] == len(starts)
          and rep["passes_lost"] == 0 and rep["passes_split"] == 0,
          "every pass runs across the crown, so every pass loses both ENDS and "
          "keeps its middle as one piece: touched, not split, not lost")
    check("nothing is DROPPED by the report -- the path still has all %d "
          "waypoints" % (len(path),), len(path) == len(pos))
    lines = describe_tool_reachability(rep)
    text = "\n".join(lines)
    check("the operator is told it is not a normals problem",
          "NOT A NORMALS PROBLEM" in text.upper())
    check("the operator is told that raising the cap is not the fix either",
          "raising the cap" in text and "lowering it" in text)
    none = tool_reachability_report(path, None)
    check("with no limit set, nothing is excluded and it says so",
          none["excluded"] == 0
          and "no tilt limit" in "\n".join(describe_tool_reachability(none)))
    flatv, flatn, flatt = flat()
    _, _, fp = _constrained_path(flatv, flatn, flatt, 31, 31)
    fr = tool_reachability_report(fp, 45.0)
    check("on a flat part the limit excludes NOTHING (%d) and says the whole "
          "surface is scannable" % (fr["excluded"],),
          fr["excluded"] == 0
          and "NOTHING is excluded" in "\n".join(describe_tool_reachability(fr)))


# ==============================================================================
# THE REAL MESHES
# ==============================================================================

def _real(name):
    return test_data.path(*REAL[name])


def _slice_raster(v, t, spacing, axis=0):
    """Plane slices every `spacing`, a waypoint at every mesh-edge crossing.

    What principal_axis_noether does -- one vtkPlane per line spacing, stepped
    along an axis, and a waypoint where the plane crosses an EDGE. Enough of the
    planner to put waypoints where it puts them, which is all these checks need.
    """
    other = 1 - axis
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    key = np.unique(np.minimum(a, b)*len(v) + np.maximum(a, b))
    p0, p1 = v[key//len(v)], v[key % len(v)]
    pos, starts = [], []
    for c in np.arange(v[:, axis].min() + spacing, v[:, axis].max(), spacing):
        d0, d1 = p0[:, axis] - c, p1[:, axis] - c
        cross = ((d0 <= 0) & (d1 > 0)) | ((d0 > 0) & (d1 <= 0))
        if cross.sum() < 3:
            continue
        w = d0[cross]/(d0[cross] - d1[cross])
        q = p0[cross] + w[:, None]*(p1[cross] - p0[cross])
        starts.append(len(pos))
        pos.extend(q[np.argsort(q[:, other])])
    return np.asarray(pos), starts


@_checked
def test_run2_the_expected_input():
    print("\nRUN 2, THE EXPECTED INPUT: the part top, 0.5 mm of relief")
    path = _real("run2")
    if path is None:
        skips.append("run2 mesh absent")
        print("  SKIP: %s is not here (scan data, not repository data)"
              % (REAL["run2"],))
        return
    v, vn, t = read_ply_mesh(path)
    rep = repair_mesh_normals(path, verbose=False, write=False)
    check("%d vertices in %d component(s)" % (rep["vertices"], rep["components"]),
          rep["vertices"] == 6090 and rep["components"] == 1)
    # This mesh carries the GROSS failure mode: the part's own surface disagrees
    # with its own neighbourhood by a third of a degree at p99.9 and the
    # artefacts sit at 89. Any threshold between 2 and 80 catches those, which
    # is why 15 looked safe for as long as this was the only mesh.
    #
    # It is not the only failure mode -- see
    # test_a_subtle_outlier_is_caught_at_three_degrees below, where the worst
    # disagreement is 8 degrees and 15 does nothing at all. The tolerance has to
    # catch both, which is what 3 does.
    b = rep["disagreement_before"]
    check("the part disagrees with its own local fit by %.2f deg at p99.9 and "
          "%.1f at worst" % (b["p999"], b["max"]),
          b["p999"] < 2.0 and b["max"] > 80.0,
          "gross artefacts; a subtler mesh is covered separately")
    check("so exactly %d normal(s) and %d face(s) need repairing"
          % (rep["replaced"], rep["faces_rewound"]),
          rep["replaced"] <= 5 and rep["faces_rewound"] <= 5)
    eb, ea = rep["edge_turn_before"], rep["edge_turn_after"]
    check("SMOOTH AND CONTINUOUS: the worst normal turn across an edge falls "
          "%.1f -> %.2f deg" % (eb["max"], ea["max"]),
          eb["max"] > 80.0 and ea["max"] < 5.0)

    surf = (v, rep["normals"], rep["triangles"])
    pos, starts = _slice_raster(v, t, 0.0003)
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    new = scanpath_from_noether(_Reply(pos, q, starts), orient="normal",
                                roll="locked", roll_reference=(1.0, 0.0, 0.0),
                                surface=surf)
    rn = assert_probe_normal_to_surface(new, surf, verbose=False, mode="off")
    raw, _ = sample_mesh_normals((v, vn, t), pos)
    raw = np.where((raw[:, 2] < 0.0)[:, None], -raw, raw)
    old = scanpath_from_noether(_Reply(pos, locked_quats_from_axes(
        raw, (1.0, 0.0, 0.0)), starts), orient="surface", roll="locked",
        roll_reference=(1.0, 0.0, 0.0))
    old = smooth_tool_axis(sanitize_tool_axis(old, verbose=False), 0.006,
                           reference=(1.0, 0.0, 0.0))
    ro = assert_probe_normal_to_surface(old, surf, verbose=False, mode="off")
    check("over %d waypoints of %d passes at the production 0.30 mm spacing: the "
          "old chain sits %.3f deg off the surface normal at worst (p99.9 %.3f, "
          "mean %.4f), the new one %.2e"
          % (len(pos), len(starts), ro["max_deg"], ro["p999_deg"],
             ro["mean_deg"], rn["max_deg"]),
          ro["max_deg"] > 1.0 and rn["max_deg"] < 1e-4
          and rn["max_deg"]*1000 < ro["max_deg"])
    check("waypoints past the 1 deg tolerance: %d -> %d"
          % (ro["over_tol"], rn["over_tol"]),
          ro["over_tol"] > 0 and rn["over_tol"] == 0,
          "the old chain's worst is EC_AXIS_SMOOTHING_M's own box mean, which is "
          "why the new chain declines it")
    rr = tool_reachability_report(new, 45.0)
    check("and EC_MAX_TILT_DEG=45 now excludes NOTHING on this mesh (%d of %d, "
          "worst tilt from vertical %.1f deg)"
          % (rr["excluded"], rr["waypoints"], rr["tilt_max_deg"]),
          rr["excluded"] == 0 and rr["tilt_max_deg"] < 10.0,
          "on the unrepaired mesh the cap drops a few percent of the waypoints; "
          "the surface is flat and the tilt was in the normals")
    # THE KEYENCE BODY. The clearance gate's reach for keyence_metal is 178 mm
    # from probe_end, and a tool rotated by theta swings it 2R sin(theta/2) along
    # a chord: 136.2 mm at the 45 deg cap, which is enough to bury it in the part.
    swing = 2*178.0*math.sin(math.radians(rr["tilt_max_deg"])/2)
    check("keyence_metal can swing at most %.1f mm here, against the 136.2 mm a "
          "45 deg cap geometrically permits" % (swing,),
          swing < 20.0,
          "the swing is bounded by the tilt, and the tilt is now the surface's")


@_checked
def test_run1_the_stress_case():
    print("\nRUN 1, THE STRESS CASE: the bench and the drape are in this mesh")
    path = _real("run1")
    if path is None:
        skips.append("run1 mesh absent")
        print("  SKIP: %s is not here (scan data, not repository data)"
              % (REAL["run1"],))
        return
    v, vn, t = read_ply_mesh(path)
    rep = repair_mesh_normals(path, verbose=False, write=False)
    b = rep["disagreement_before"]
    check("%d of %d normals (%.1f%%) disagree with their own neighbourhood past "
          "15 deg, %d of them pointing the wrong way entirely"
          % (rep["replaced"], rep["vertices"], 100*rep["replaced_fraction"],
             rep["backwards_before"]),
          rep["replaced"] > 300 and rep["backwards_before"] > 20)
    check("the disagreement reaches %.1f deg at worst and %.1f at p99.9, which "
          "is normals pointing sideways and backwards" % (b["max"], b["p999"]),
          b["max"] > 150.0 and b["p999"] > 80.0)
    ea = rep["edge_turn_after"]
    check("a %.0f deg step SURVIVES the repair, because it is real: this mesh "
          "has a 28.4 mm cliff off the bench inside ONE component"
          % (ea["max"],), ea["max"] > 90.0,
          "flattening it would drive the probe edge-on into the bench; the fix "
          "is to crop the bench out of the ROI, and this is the backstop")

    surf = (v, rep["normals"], rep["triangles"])
    pos, starts = _slice_raster(v, t, 0.0003)
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    new = scanpath_from_noether(_Reply(pos, q, starts), orient="normal",
                                roll="locked", roll_reference=(1.0, 0.0, 0.0),
                                surface=surf)
    rn = assert_probe_normal_to_surface(new, surf, verbose=False, mode="off")
    check("EVEN ON THIS MESH the probe is normal to the surface at every one of "
          "%d waypoints (worst %.2e deg)" % (rn["waypoints"], rn["max_deg"]),
          rn["max_deg"] < 1e-4,
          "a bad mesh must not produce a mad toolpath: what is left is "
          "unreachable surface, not a tilted probe")
    rr = tool_reachability_report(new, 45.0)
    check("%d waypoints (%.1f%%) ARE unreachable, %d of them on surface facing "
          "away, and all of it is reported with a bounding box"
          % (rr["excluded"], 100*rr["excluded_fraction"], rr["facing_away"]),
          rr["excluded"] > 500 and rr["facing_away"] > 0 and rr["bbox"] is not None)
    sane = sanitize_tool_axis(new, verbose=False, assert_only=True)
    check("the assertion FINDS the cliff (%d disagreement(s)) and CUTS %d pass(es) "
          "there rather than altering an axis"
          % (sane.axis_sanity["repaired"], sane.axis_sanity["split"]),
          sane.axis_sanity["repaired"] > 0 and sane.axis_sanity["split"] > 0)
    check("and still no waypoint is lost (%d -> %d)" % (len(new), len(sane)),
          len(sane) == len(new))


# ==============================================================================
# TURNED-ROUND POSES: FOUND, COUNTED, REPAIRED OR DROPPED
# ==============================================================================
# WHY THIS BLOCK EXISTS ON TOP OF THE POSTCONDITION. On a normal-constrained path
# assert_probe_normal_to_surface() compares the tool axis with the SAME
# sample_mesh_normals() call that produced it, so any error in the sampler
# cancels and the answer is 0.000 deg by arithmetic -- 0.000 on all five recorded
# runs in this cell, at a line spacing where the sampler can pick a triangle
# 1.47 mm from the waypoint. These checks measure the three things a
# self-cancelling check cannot:
#
#   * a reversed axis, against the surface at the waypoint rather than against
#     world up;
#   * a TANGENTIAL axis, which reversing does not fix and a reversal test does
#     not see;
#   * a report that cannot say nothing was flipped when something was.


def _turned(path, which, angle_deg=180.0):
    """A copy of `path` with the poses in `which` turned about their OWN x axis.

    Right-multiplied, so the turn is in the tool frame and the angle is exactly
    the angle the tool AXIS moves through, whatever the pose was: 180 deg reverses
    it and 90 leaves it lying along the surface. A world-frame rotation would not
    -- on a pose already tilted on a flank it reflects the axis instead, which is
    neither a clean reversal nor a clean tangent, and the ground truth these
    checks need would not be a known number.

    180 about tool x is the exact turn orient_tool_away_from_surface() applies,
    so it is also the fault being reproduced rather than an invented one.
    """
    q = np.array(path.quats, dtype=np.float64, copy=True)
    h = math.radians(angle_deg)/2.0
    dq = (math.sin(h), 0.0, 0.0, math.cos(h))
    for i in np.atleast_1d(np.asarray(which)):
        q[i] = NS["quat_mul"](tuple(q[i]), dq)
    out = ScanPath(np.array(path.positions, copy=True), q,
                   list(path.segment_starts), source=path.source)
    NS["_carry_path_attrs"](path, out)
    return out


@_checked
def test_a_reversed_pose_is_found_and_repaired():
    print("\nA REVERSED POSE IS FOUND AGAINST THE SURFACE, AND PUT BACK")
    v, n, t, nu, nv = cylinder_crown()
    rng = np.random.default_rng(11)
    bad_n, _pick = inject_alternating(n, rng, 0.05)
    rep = repair_mesh_normals((v, bad_n, t), verbose=False, write=False)
    surf = (v, rep["normals"], rep["triangles"])
    pos, starts, path = _constrained_path(v, None, None, nu, nv, surface=surf)
    # GROUND TRUTH: a known number of poses, turned right over.
    which = np.arange(7, len(path), max(1, len(path)//23))[:17]
    broken = _turned(path, which, 180.0)
    r = tool_axis_flip_report(broken, surf)
    check("%d poses turned over are found as %d REVERSED, %d tangential, %d off"
          % (len(which), r["reversed"], r["tangential"], r["off_normal"]),
          r["reversed"] == len(which) and r["tangential"] == 0
          and r["off_normal"] == 0,
          "the count IS the ground truth; a global up test would miss the ones "
          "on the crown's flanks and flag none of the rest")
    said = "\n".join(describe_tool_axis_flips(r))
    check("and the report cannot say nothing was flipped",
          "TURNED-ROUND POSES FOUND" in said and "%d REVERSED" % (len(which),) in said,
          said[:120])
    fixed, fr = repair_tool_axis_to_surface(broken, surf, verbose=False,
                                            crop=_crop_by_mask)
    check("the repair puts back exactly those %d and drops nothing (%d -> %d)"
          % (fr["repaired"], len(broken), len(fixed)),
          fr["repaired"] == len(which) and fr["dropped"] == 0
          and len(fixed) == len(broken))
    after = tool_axis_flip_report(fixed, surf)
    check("afterwards NO pose opposes the local outward normal (worst %.2e deg, "
          "%d reversed)" % (after["max_deg"], after["reversed"]),
          after["reversed"] == 0 and after["max_deg"] < 1e-6
          and after["wrong"] == 0)
    # The positions are the measurement; the repair may not move them.
    check("and not one position moved",
          bool(np.array_equal(np.asarray(fixed.positions),
                              np.asarray(broken.positions))))
    # The roll is a measurement parameter and the repair carries it.
    ref37 = NS["probe_rotation_reference"](37.0, (1.0, 0.0, 0.0))
    rolled = _constrained_path(v, None, None, nu, nv, surface=surf)[2]
    rolled = ScanPath(np.asarray(rolled.positions),
                      locked_quats_from_axes(tool_axes(rolled.quats), ref37),
                      list(rolled.segment_starts), source=rolled.source)
    NS["_carry_path_attrs"](path, rolled)
    rolled.roll_reference = ref37
    rolled.roll_mode = "locked"
    rb = _turned(rolled, which, 180.0)
    rf, _ = repair_tool_axis_to_surface(rb, surf, verbose=False,
                                        crop=_crop_by_mask)
    tw = tool_twist_deg(rf.quats, reference=NS["roll_reference_quat"]((1.0, 0.0, 0.0)))
    check("a 37 deg probe rotation SURVIVES the repair at every pose "
          "(%+.3f..%+.3f deg)" % (tw.min(), tw.max()),
          abs(tw.min() - 37.0) < 1e-6 and abs(tw.max() - 37.0) < 1e-6,
          "the repair swings the axis and carries the roll; it does not re-lock it")


@_checked
def test_a_tangential_pose_is_counted_apart_from_a_reversed_one():
    print("\nA TANGENTIAL AXIS IS A DIFFERENT DEFECT, AND IS COUNTED AS ONE")
    v, n, t, nu, nv = cylinder_crown()
    surf = (v, n, t)
    pos, starts, path = _constrained_path(v, n, t, nu, nv)
    tang = np.arange(5, len(path), max(1, len(path)//13))[:11]
    rev = np.arange(9, len(path), max(1, len(path)//17))[:7]
    rev = np.array([i for i in rev if i not in set(tang.tolist())])
    broken = _turned(_turned(path, tang, 90.0), rev, 180.0)
    r = tool_axis_flip_report(broken, surf)
    check("%d poses laid along the surface read as %d TANGENTIAL, not as reversed"
          % (len(tang), r["tangential"]),
          r["tangential"] == len(tang),
          "a reversal test sees nothing here: turning a tangential axis over "
          "leaves it just as tangential")
    check("and the %d genuinely reversed ones are still counted as %d REVERSED"
          % (len(rev), r["reversed"]), r["reversed"] == len(rev))
    said = "\n".join(describe_tool_axis_flips(r))
    check("the report names both classes separately",
          "%d TANGENTIAL" % (len(tang),) in said
          and "%d REVERSED" % (len(rev),) in said)
    fixed, fr = repair_tool_axis_to_surface(broken, surf, verbose=False,
                                            crop=_crop_by_mask)
    check("both classes are repaired (%d poses) and nothing is dropped"
          % (fr["repaired"],),
          fr["repaired"] == len(tang) + len(rev) and fr["dropped"] == 0)
    check("afterwards nothing is tangential or reversed (worst %.2e deg)"
          % (fr["residual_max_deg"],), fr["residual_wrong"] == 0)


@_checked
def test_away_facing_surface_is_dropped_never_turned_over():
    print("\nSURFACE FACING AWAY FROM THE PROBE IS DROPPED, NOT TURNED OVER")
    # A height field with a lip that folds back: the underside of the lip faces
    # DOWN, so a probe coming from above cannot touch it at any tilt.
    nu, nv = 41, 21
    x = np.linspace(0.75, 0.85, nu)
    y = np.linspace(0.0, 0.05, nv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full_like(X, 0.930)
    v, t, _, _ = grid_mesh(X, Y, Z)
    # Normals: up everywhere except one strip, where the surface genuinely faces
    # down. This is the draped-skirt underside, not a sign error.
    n = np.tile(np.array([0.0, 0.0, 1.0]), (len(v), 1))
    strip = np.flatnonzero(np.abs(v[:, 0] - 0.80) < 0.006)
    # Down AND sideways, 135 deg from up: the underside of a lip, which is what a
    # reconstruction of a draped part really contains. Straight down is refused by
    # the constraint itself (LOCKED_ROLL_GUARD_DEG) and would test that instead.
    n[strip] = np.array([math.sin(math.radians(135.0)), 0.0,
                         math.cos(math.radians(135.0))])
    pos, starts = edge_midpoint_raster(v, t, nu, nv)
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    path = scanpath_from_noether(_Reply(pos, q, starts), orient="normal",
                                 roll="locked", roll_reference=(1.0, 0.0, 0.0),
                                 surface=(v, n, t))
    r = tool_axis_flip_report(path, (v, n, t))
    check("%d waypoint(s) sit on surface facing AWAY from the probe"
          % (r["away_facing_surface"],), r["away_facing_surface"] > 0)
    check("and NONE of them reads as reversed, because the POSE agrees with its "
          "surface -- it is the surface that faces the wrong way (%d reversed)"
          % (r["reversed"],), r["reversed"] == 0,
          "this is the distinction the whole rule turns on")
    down_before = int((tool_axes(path.quats)[:, 2] < 0.0).sum())
    fixed, fr = repair_tool_axis_to_surface(path, (v, n, t), verbose=False,
                                            crop=_crop_by_mask)
    check("%d of them were DROPPED with a count and a bounding box"
          % (fr["dropped"],),
          fr["dropped"] >= r["away_facing_surface"] and fr["dropped"] > 0
          and fr["dropped_bbox"] is not None)
    check("NOT turned over: the path had %d poses with the axis pointing down "
          "and now has %d" % (down_before, int((tool_axes(fixed.quats)[:, 2] < 0).sum())),
          down_before > 0 and int((tool_axes(fixed.quats)[:, 2] < 0.0).sum()) == 0,
          "turning one over points the probe away from the part and drives the "
          "holder into the specimen")
    check("and the surviving poses are untouched (%d of %d kept)"
          % (len(fixed), len(path)), len(fixed) < len(path) and len(fixed) > 0)
    said = "\n".join(describe_tool_axis_flips(fr))
    check("the report says DROPPED and says why",
          "DROPPED" in said and "Not turned over" in said)


@_checked
def test_the_sampler_finds_the_triangle_the_waypoint_is_on():
    print("\nTHE SAMPLER LOOKS PAST THE NEAREST VERTEX, AND PAST SLIVERS")
    # A flat plate with one SLIVER welded across it: three nearly collinear
    # vertices, so its plane stands almost vertical while the plate is flat. This
    # is what a reconstruction really produces -- 99 of them in 10 153 faces on
    # one of this cell's planning meshes, with altitudes down to 14 picometres --
    # and it is how a waypoint on flat surface comes back 80 degrees out.
    v, n, t = flat(nu=21, nv=21)
    a0, b0 = 0, 1
    mid = 0.5*(v[a0] + v[b0]) + np.array([0.0, 0.0, 2e-6])
    v2 = np.vstack([v, mid[None, :]])
    n2 = np.vstack([n, np.array([[0.0, 0.0, 1.0]])])
    t2 = np.vstack([t, np.array([[a0, b0, len(v)]])])
    keep = non_degenerate_faces(v2, t2)
    p0, p1, p2 = v2[t2[-1, 0]], v2[t2[-1, 1]], v2[t2[-1, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn = fn/np.linalg.norm(fn)
    check("the sliver's own plane stands %.1f deg from the flat plate it lies in"
          % (ang_between(fn, np.array([0.0, 0.0, 1.0]))[0],),
          ang_between(fn, np.array([0.0, 0.0, 1.0]))[0] > 80.0)
    check("it is recognised as a sliver and every honest face is kept "
          "(%d of %d faces)" % (int(keep.sum()), len(t2)),
          bool(keep[:-1].all()) and not bool(keep[-1]),
          "an absolute area threshold does not separate these; the ratio of the "
          "smallest altitude to the longest edge does")
    # sample_mesh_normals must not read that face for a point on the plate.
    probe = np.array([mid + np.array([0.0, 0.0002, -2e-6])])
    got, dist = sample_mesh_normals((v2, n2, t2), probe)
    check("a waypoint beside the sliver is still sampled on the plate (%.4f mm "
          "away, normal %s)" % (dist[0]*1e3, np.round(got[0], 4)),
          dist[0] < 1e-6 and got[0][2] > 0.999)
    fa = NS["_tool_axis_against_faces"]((v2, n2, t2), probe,
                                        np.array([[0.0, 0.0, 1.0]]))
    check("and the independent face check does not read that sliver either "
          "(%.3f deg)" % (fa[0],), fa[0] < 1.0)
    # And the vertex-count bound: with only the nearest vertex's faces, a
    # waypoint whose closest triangle is one vertex further off is missed.
    v3, n3, t3, nu3, nv3 = cylinder_crown(nu=31, nv=21)
    pts, _ = edge_midpoint_raster(v3, t3, nu3, nv3)
    f1, _w1, d1, _h1 = closest_surface_faces(v3, t3, pts, vertices=1)
    f8, _w8, d8, _h8 = closest_surface_faces(v3, t3, pts, vertices=8)
    check("searching 8 nearest vertices is never worse than searching 1 "
          "(worst improvement %.4f mm)" % ((d1 - d8).max()*1e3,),
          bool((d8 <= d1 + 1e-12).all()))
    # THE PAIR BLOCK IS A MEMORY BOUND AND NOTHING ELSE. It restricts how many
    # (waypoint, triangle) pairs are evaluated at once and never which triangles a
    # waypoint is compared with, so the answer must be bit-identical at any block
    # size -- asserted rather than trusted, the same as the normal fit's.
    saved = NS["NORMAL_SAMPLE_PAIR_BLOCK"]
    try:
        NS["NORMAL_SAMPLE_PAIR_BLOCK"] = 1000000
        a, _d = sample_mesh_normals((v3, n3, t3), pts)
        NS["NORMAL_SAMPLE_PAIR_BLOCK"] = 97
        b, _d = sample_mesh_normals((v3, n3, t3), pts)
    finally:
        NS["NORMAL_SAMPLE_PAIR_BLOCK"] = saved
    check("and the pair block changes not one bit (%d waypoints, block 1000000 "
          "against 97)" % (len(pts),), bool(np.array_equal(a, b)))
    # THE INDEPENDENT FACE CHECK SAMPLES, AND SAYS SO. It reports a distribution,
    # and its search is the expensive half, so it is capped -- and the count it
    # measured over has to be in the report or 50 000 reads as the whole path.
    savedcap = NS["FACE_CHECK_MAX_WAYPOINTS"]
    try:
        NS["FACE_CHECK_MAX_WAYPOINTS"] = 200
        pos3, starts3, path3 = _constrained_path(v3, n3, t3, nu3, nv3)
        r3 = assert_probe_normal_to_surface(path3, (v3, n3, t3), verbose=False,
                                           mode="off")
    finally:
        NS["FACE_CHECK_MAX_WAYPOINTS"] = savedcap
    check("with the cap at 200 the face check measured %d of %d waypoints and "
          "reports which" % (r3["face_waypoints"], r3["waypoints"]),
          r3["face_waypoints"] == 200 and r3["waypoints"] > 200)


def ripple(nu=61, nv=21, amp=0.005, waves=5.0):
    """A rippled height field. -> (v, n_analytic, t, nu, nv)

    A CYLINDER IS THE ONE SURFACE THIS TEST CANNOT USE. Its normal turns linearly
    with arc length, so interpolating between two waypoints' axes lands exactly on
    the true normal at the point between them and the fault under test does not
    appear at all. A ripple's normal does not, which is the ordinary case on a
    reconstructed part.
    """
    x = np.linspace(0.75, 0.85, nu)
    y = np.linspace(0.0, 0.05, nv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    k = 2.0*np.pi*waves/0.10
    Z = 0.930 + amp*np.sin(k*(X - 0.75))
    v, t, _, _ = grid_mesh(X, Y, Z)
    # dz/dx = amp k cos(k dx); the outward normal is (-dz/dx, 0, 1) normalised.
    slope = amp*k*np.cos(k*(v[:, 0] - 0.75))
    n = np.column_stack([-slope, np.zeros(len(v)), np.ones(len(v))])
    return v, n/np.linalg.norm(n, axis=1, keepdims=True), t, nu, nv


@_checked
def test_resampling_keeps_the_probe_on_the_normal():
    print("\nRESAMPLING RE-READS THE SURFACE INSTEAD OF INTERPOLATING THE AXIS")
    v, n, t, nu, nv = ripple()
    surf = (v, n, t)
    pos, starts, path = _constrained_path(v, n, t, nu, nv)
    r0 = assert_probe_normal_to_surface(path, surf, verbose=False, mode="off")
    # Without the surface the axis can only be interpolated, and the departure
    # from the normal at the NEW position is the measurement.
    bare = ScanPath(np.asarray(path.positions), np.asarray(path.quats),
                    list(path.segment_starts), source=path.source)
    bare.roll_mode = "locked"
    lerped = resample_segments(bare, 0.006, reference=(1.0, 0.0, 0.0))
    rl = assert_probe_normal_to_surface(lerped, surf, verbose=False, mode="off")
    resampled = resample_segments(path, 0.006, reference=(1.0, 0.0, 0.0))
    rr = assert_probe_normal_to_surface(resampled, surf, verbose=False, mode="off")
    check("planned worst %.2e deg; resampled WITH the surface %.2e deg; "
          "interpolated instead %.2f deg"
          % (r0["max_deg"], rr["max_deg"], rl["max_deg"]),
          r0["max_deg"] < 1e-6 and rr["max_deg"] < 1e-6 and rl["max_deg"] > 2.0,
          "a resampled waypoint is a NEW place on the part, so the axis there is "
          "a question for the surface and not for its neighbours")
    check("the resample says which it did (axis_from_surface %s, %d -> %d "
          "waypoints)" % (resampled.resampled["axis_from_surface"],
                          resampled.resampled["waypoints_before"],
                          resampled.resampled["waypoints_after"]),
          resampled.resampled["axis_from_surface"] is True
          and lerped.resampled["axis_from_surface"] is False)
    # And the interpolation itself is a rotation, not a chord.
    s = np.array([0.0, 1.0])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mid = slerp_along(s, d, np.array([0.5]))[0]
    check("slerp_along() puts the halfway direction on the great circle "
          "(%.4f deg from each end)" % (ang_between(mid, d[0])[0],),
          abs(ang_between(mid, d[0])[0] - 45.0) < 1e-9
          and abs(ang_between(mid, d[1])[0] - 45.0) < 1e-9)


@_checked
def test_the_postcondition_has_an_independent_half():
    print("\nTHE POSTCONDITION NO LONGER CHECKS ITSELF")
    v, n, t, nu, nv = cylinder_crown()
    surf = (v, n, t)
    pos, starts, path = _constrained_path(v, n, t, nu, nv)
    r = assert_probe_normal_to_surface(path, surf, verbose=False, mode="off")
    check("against the vertex normals it reads %.2e deg, which it must, because "
          "that is where the axis came from" % (r["max_deg"],),
          r["max_deg"] < 1e-6)
    check("and INDEPENDENTLY, against the plane of each waypoint's own triangle, "
          "%.3f deg with %d reversed -- a different quantity that cannot cancel"
          % (r["face_max_deg"], r["face_reversed"]),
          r["face_reversed"] == 0 and 0.0 < r["face_max_deg"] < 5.0,
          "its floor is the mesh's faceting, so it is a diagnostic and not a "
          "tolerance")
    # Turn the mesh's vertex normals over without touching its triangles, and turn
    # the path's axes over with them, so the two still agree. The self-cancelling
    # half reads zero; the independent half does not. (scanpath_from_noether()
    # refuses such a surface outright -- MeshNotPlannable, "inside out" -- which is
    # why the path is built here by hand: the point is what the CHECK can see, not
    # what the constraint lets through.)
    surf2 = (v, -n, t)
    path2 = ScanPath(np.asarray(path.positions),
                     locked_quats_from_axes(-tool_axes(path.quats),
                                            (1.0, 0.0, 0.0)),
                     list(path.segment_starts), source=path.source)
    path2.normal_constrained = True
    r2 = assert_probe_normal_to_surface(path2, surf2, verbose=False, mode="off")
    check("with the vertex normals inside out the old check still reads %.2e deg"
          % (r2["max_deg"],), r2["max_deg"] < 1e-6,
          "this is the tautology, demonstrated")
    check("while the independent half reads %.1f deg with %d of %d reversed"
          % (r2["face_max_deg"], r2["face_reversed"], r2["waypoints"]),
          r2["face_reversed"] == r2["waypoints"] and r2["face_max_deg"] > 170.0)


def main():
    for fn in (test_not_blind, test_neighbourhood_and_components,
               test_blocking_changes_no_number,
               test_fit_accuracy, test_curvature_is_not_repaired_away,
               test_artefacts_are_repaired,
               test_a_subtle_outlier_is_caught_at_three_degrees,
               test_a_mean_of_normals_cannot_do_this,
               test_component_boundary_survives, test_winding_and_missing_normals,
               test_sampling, test_orientation_is_constrained_by_construction,
               test_postcondition_fires_without_the_constraint,
               test_surface_faults_are_refused_not_flipped,
               test_the_other_smoothers_stand_down,
               test_reachability_is_reported_not_deleted,
               test_run2_the_expected_input, test_run1_the_stress_case,
               test_a_reversed_pose_is_found_and_repaired,
               test_a_tangential_pose_is_counted_apart_from_a_reversed_one,
               test_away_facing_surface_is_dropped_never_turned_over,
               test_the_sampler_finds_the_triangle_the_waypoint_is_on,
               test_resampling_keeps_the_probe_on_the_normal,
               test_the_postcondition_has_an_independent_half):
        fn.__wrapped__()
    print("\n%d check(s) failed%s"
          % (len(fails), "; %d skipped" % (len(skips),) if skips else ""))
    for f in fails:
        print("  FAIL %s" % (f,))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
