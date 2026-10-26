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
# File:  projection_surface_identity_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-01
# Description:
#   Which surface a projector is holding, and whether it is under
#   the toolpath at all.
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
"""A projection surface is identified by its CONTENTS, not by its filename.

THE RUN THIS FILE IS ABOUT. In scandata/results16 the second measurement
acquired 3531 eddy-current traces, reported every pass at 100% of its expected
count, and projected ZERO of them. Two defects combined:

  THE FILENAME WAS SHARED. The parameterized .x3d was written into the one
  cell-wide work directory under a name derived from the mesh's BASENAME, while
  the mesh itself lived in a per-run session directory. Every run of the
  pipeline therefore wrote the same file and overwrote the last one.

  THE REUSE TEST COMPARED THE PATH STRING. So the overwritten file was treated
  as unchanged: the process kept run B's loaded geometry -- a 6489-triangle flat
  plate spanning x 0.797..0.828 -- and scanned run C's part, a 6782-triangle
  curved patch spanning x 0.852..0.906. The two are disjoint. Closest approach
  31.3 mm, 0 of 3531 tool-axis lines intersecting.

Neither half raises. raytrace_sensor_evaluate_zdist returns inf for a ray that
misses and project_onto_parameterization drops the point with no log and no
throw, so the scan looked perfect from every angle an operator can see.

So three things are pinned here:

  * two meshes in two session directories get two .x3d paths (mesh_scan.dpi)
  * a file REWRITTEN at the same path is not reused (surface_projection.dpi)
  * a surface disjoint from the toolpath is REFUSED before the arm moves, and
    an overlapping one -- including a legitimately stood-off one -- is not

Everything runs on the HOST with numpy and real files in a temp directory. The
functions are lifted out of the .dpi files by their AST, because a .dpi is a
function body and is not importable, and because surface_projection.dpi imports
spatialnde2 at the top. Lifting them by NAME also means a rename is a failure
here rather than a silent loss of coverage.

    python3 -m pytest -q projection_surface_identity_test.py
"""
import ast
import os
import shutil
import stat
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_DPI = os.path.join(HERE, "surface_projection.dpi")
MESH_DPI = os.path.join(HERE, "mesh_scan.dpi")
EC_DPI = os.path.join(HERE, "ec_acquisition.dpi")
ACC_DPI = os.path.join(HERE, "scan_accumulation.dpi")


# ==============================================================================
# Lifting the code under test out of the .dpi files
# ==============================================================================

#: surface_projection.dpi functions this file exercises. By name, so that a
#: rename breaks the test instead of quietly removing a check.
PROJ_WANT = ("write_plane_x3d", "projection_surface_geometry",
             "projection_surface_fingerprint", "projection_surface_matches",
             "describe_surface_counts", "projection_surface_reusable",
             "describe_bbox", "toolpath_world_bbox", "bbox_gap",
             "projection_file_world_bbox", "check_projection_covers_path")

#: ... and its constants, taken from the source rather than repeated here.
PROJ_CONSTS = ("PROJECTION_FOOTPRINT_SLACK_M", "PROJECTION_RAY_WINDOW_M",
               "PROJECTION_RAY_WINDOW_NEAR_M")


def _tree(path):
    return ast.parse(open(path).read(), filename=os.path.basename(path))


def _named_function(tree, name, where):
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, "%s defines %s() %d times, expected once" % (
        where, name, len(found))
    return found[0]


def _method(tree, class_name, method_name, where):
    cls = [n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == class_name]
    assert len(cls) == 1, "%s does not define class %s once" % (where, class_name)
    found = [n for n in cls[0].body
             if isinstance(n, ast.FunctionDef) and n.name == method_name]
    assert len(found) == 1, "%s.%s is not defined exactly once" % (
        class_name, method_name)
    return found[0]


def load_projection():
    """surface_projection.dpi's surface-identity half, compiled on its own.

    SurfaceProjector itself needs a recording database, so only the one method
    the guard reads -- surface_world_bbox() -- is lifted, onto a stand-in class
    whose attributes the test sets. That keeps the method under test REAL: the
    offset it applies to the file's bounding box is the projector's own code,
    not a reimplementation of it here.
    """
    tree = _tree(PROJ_DPI)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PROJ_WANT:
            body.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in PROJ_CONSTS:
            body.append(node)
    got = set(n.name for n in body if isinstance(n, ast.FunctionDef))
    missing = [w for w in PROJ_WANT if w not in got]
    assert not missing, "surface_projection.dpi no longer defines %s" % (missing,)
    gotc = set(n.targets[0].id for n in body if isinstance(n, ast.Assign))
    assert not [c for c in PROJ_CONSTS if c not in gotc], \
        "surface_projection.dpi no longer defines %s" % (PROJ_CONSTS,)

    holder = ast.parse("class StandInProjector(object):\n    pass\n").body[0]
    holder.body = [_method(tree, "SurfaceProjector", "surface_world_bbox",
                           "surface_projection.dpi")]
    body.append(holder)

    said = []
    ns = {"np": np, "os": os, "re": __import__("re"),
          "hashlib": __import__("hashlib"),
          "print": lambda *a: said.append(" ".join(str(x) for x in a))}
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "surface_projection.dpi",
                 "exec"), ns)
    ns["_printed"] = said
    return ns


def load_mesh_x3d_naming(work_dir):
    """mesh_scan.dpi's mesh_projection_x3d() and the work directory it falls back on."""
    tree = _tree(MESH_DPI)
    body = [_named_function(tree, "mesh_projection_x3d", "mesh_scan.dpi"),
            _named_function(tree, "_scan_work_dir", "mesh_scan.dpi")]
    ns = {"os": os, "hashlib": __import__("hashlib"),
          "SCAN_STORAGE_DIR": work_dir, "scan_storage_manager": object()}
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "mesh_scan.dpi", "exec"), ns)
    return ns


PROJ = load_projection()
SLACK_M = PROJ["PROJECTION_FOOTPRINT_SLACK_M"]
REACH_M = PROJ["PROJECTION_RAY_WINDOW_M"]


# ==============================================================================
# Fixtures: real .x3d files, and a stand-in projector holding one
# ==============================================================================

@pytest.fixture
def tmpdir_path():
    d = tempfile.mkdtemp(prefix="projsurf_")
    try:
        yield d
    finally:
        for root, dirs, _ in os.walk(d):
            for sub in dirs:
                os.chmod(os.path.join(root, sub),
                         stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        shutil.rmtree(d, ignore_errors=True)


def write_patch(path, lo, hi):
    """A rectangular UV patch spanning lo..hi in x and y, flat at lo[2]. -> path

    Written with the .dpi's own write_plane_x3d(), so the files these tests read
    are exactly the shape the plane scan produces and the parser under test has
    to cope with the real markup rather than a hand-rolled approximation.
    """
    origin = np.array([lo[0], lo[1], lo[2]], dtype=np.float64)
    u = np.array([hi[0] - lo[0], 0.0, 0.0])
    v = np.array([0.0, hi[1] - lo[1], hi[2] - lo[2]])
    return PROJ["write_plane_x3d"](path, origin, u, v)


def stand_in(path, offset=(0.0, 0.0, 0.0)):
    """A projector that has loaded `path`, as the reuse test and the guard see it."""
    p = PROJ["StandInProjector"]()
    p.offset = tuple(float(v) for v in offset)
    p.x3d_path = path
    p.x3d_fingerprint = PROJ["projection_surface_fingerprint"](path)
    p.min_dist = -PROJ["PROJECTION_RAY_WINDOW_NEAR_M"]
    p.max_dist = REACH_M
    return p


def positions(lo, hi, n=7):
    """A toolpath whose positions span exactly lo..hi."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return np.asarray(lo) + t*(np.asarray(hi) - np.asarray(lo))


class Path(object):
    """The one attribute of ScanPath the guard reads."""

    def __init__(self, pos):
        self.positions = np.asarray(pos, dtype=np.float64)


# ==============================================================================
# 1. Two meshes, two surfaces. The filename collision.
# ==============================================================================

def test_meshes_in_different_session_dirs_get_different_x3d_paths(tmpdir_path):
    """THE results16 COLLISION, as a test.

    Identical basenames -- which is what the pipeline produces, because the
    mesh name is built from the specimen and the stage rather than from the run
    -- in two per-run session directories. The old derivation threw the
    directory away and kept the basename, so both runs wrote one file in the
    shared work directory and the second silently replaced the first.
    """
    work = os.path.join(tmpdir_path, "scandata")
    os.makedirs(work)
    ns = load_mesh_x3d_naming(work)

    name = "specimen_keyence_ball_pivot_smoothed_filtered_ecplan_roi.ply"
    paths = []
    for run in ("run_b", "run_c"):
        session = os.path.join(work, run)
        os.makedirs(session)
        mesh = os.path.join(session, name)
        open(mesh, "w").close()
        paths.append(ns["mesh_projection_x3d"](mesh))

    assert paths[0] != paths[1], (
        "two runs' meshes still resolve to one .x3d (%s) -- this is the "
        "results16 collision" % (paths[0],))
    # And each one is BESIDE its own mesh, which is what makes that true for
    # any pair of runs rather than only for these two.
    for run, p in zip(("run_b", "run_c"), paths):
        assert os.path.dirname(p) == os.path.join(work, run)
        assert p.endswith("_uv.x3d")
    # Not in the shared work directory, which is where the collision lived.
    assert all(os.path.dirname(p) != work for p in paths)


def test_the_same_mesh_always_gets_the_same_surface(tmpdir_path):
    """Idempotent, because planning twice must not strand a second surface."""
    work = os.path.join(tmpdir_path, "scandata")
    os.makedirs(work)
    ns = load_mesh_x3d_naming(work)
    mesh = os.path.join(work, "plate.ply")
    open(mesh, "w").close()
    assert ns["mesh_projection_x3d"](mesh) == ns["mesh_projection_x3d"](mesh)
    # A relative path and an absolute one are the same mesh.
    cwd = os.getcwd()
    try:
        os.chdir(work)
        assert ns["mesh_projection_x3d"]("plate.ply") == \
            ns["mesh_projection_x3d"](mesh)
    finally:
        os.chdir(cwd)


def test_an_unwritable_mesh_dir_still_gets_a_unique_name(tmpdir_path):
    """The fallback must not reintroduce the collision it exists to avoid.

    A mesh can live somewhere unwritable -- a package's meshes/ directory, a
    read-only dataset -- and the parameterization has to go somewhere. Falling
    back to the work directory under the BASENAME would be the original bug
    again, so the fallback name carries a digest of the mesh's absolute path.
    """
    work = os.path.join(tmpdir_path, "scandata")
    os.makedirs(work)
    ns = load_mesh_x3d_naming(work)

    paths = []
    for run in ("a", "b"):
        ro = os.path.join(tmpdir_path, "readonly_%s" % (run,))
        os.makedirs(ro)
        mesh = os.path.join(ro, "plate.ply")
        open(mesh, "w").close()
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        if os.access(ro, os.W_OK):
            pytest.skip("this user can write to a mode 500 directory (root?)")
        paths.append(ns["mesh_projection_x3d"](mesh))

    for p in paths:
        assert os.path.dirname(p) == work, \
            "the fallback did not land in the work directory: %s" % (p,)
    assert paths[0] != paths[1], (
        "two unwritable meshes with the same basename still collide at %s"
        % (paths[0],))


# ==============================================================================
# 2. A rewritten file is a different surface.
# ==============================================================================

def test_the_same_path_rewritten_is_not_reused(tmpdir_path):
    """DEFECT (2): the reuse test used to compare only the path string.

    The projector is holding run B's flat plate. Run C writes its own curved
    patch to the same filename. Reuse would restart() the loaded geometry --
    which is still run B's -- onto run C's part.
    """
    path = os.path.join(tmpdir_path, "specimen_uv.x3d")
    write_patch(path, (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    proj = stand_in(path)

    ok, why = PROJ["projection_surface_reusable"](proj, path)
    assert ok, "an untouched surface must still be reusable (%s)" % (why,)

    write_patch(path, (0.852, 0.005, 0.959), (0.906, 0.055, 0.980))
    ok, why = PROJ["projection_surface_reusable"](proj, path)
    assert not ok, "a rewritten surface was accepted for reuse"
    assert "REWRITTEN" in why, why
    # And the reason has to be diagnosable, not merely different: it says which
    # surface is there now.
    assert "0.8520" in why and "0.9060" in why, why


def test_identical_bytes_at_a_new_timestamp_are_still_reusable(tmpdir_path):
    """MTIME IS NOT PART OF THE TEST, deliberately, and this pins that.

    A rewrite always changes the digest, so comparing mtime detects nothing
    extra. What it would add is a FALSE mismatch -- a copy, a restore, a
    parameterization that reran and produced the same output -- and every false
    mismatch forces a rebuild that strands a model tree whose channels can never
    be handed back. The digest is strictly stronger.
    """
    path = os.path.join(tmpdir_path, "plate_uv.x3d")
    write_patch(path, (0.0, 0.0, 0.5), (0.1, 0.1, 0.5))
    proj = stand_in(path)
    body = open(path, "rb").read()
    os.utime(path, ns=(0, 0))
    with open(path, "wb") as fh:
        fh.write(body)
    assert proj.x3d_fingerprint["mtime_ns"] != \
        PROJ["projection_surface_fingerprint"](path)["mtime_ns"], \
        "the timestamp did not change, so this test proves nothing"
    ok, why = PROJ["projection_surface_reusable"](proj, path)
    assert ok, why


def test_a_different_path_is_never_reused(tmpdir_path):
    """The old half of the test still has to hold."""
    a = write_patch(os.path.join(tmpdir_path, "a_uv.x3d"),
                    (0.0, 0.0, 0.5), (0.1, 0.1, 0.5))
    b = write_patch(os.path.join(tmpdir_path, "b_uv.x3d"),
                    (0.0, 0.0, 0.5), (0.1, 0.1, 0.5))
    proj = stand_in(a)
    ok, why = PROJ["projection_surface_reusable"](proj, b)
    assert not ok and "different surface file" in why, why


def test_unknown_is_not_a_match(tmpdir_path):
    """No projector, no surface, or an unreadable file: rebuild, never reuse.

    Answering "reusable" on a question that could not be answered is exactly
    how an empty image comes to be reported as a complete scan.
    """
    path = write_patch(os.path.join(tmpdir_path, "c_uv.x3d"),
                       (0.0, 0.0, 0.5), (0.1, 0.1, 0.5))
    assert PROJ["projection_surface_reusable"](None, path)[0] is False
    proj = stand_in(path)
    assert PROJ["projection_surface_reusable"](proj, None)[0] is False

    # A projector from before fingerprints existed, or one whose file has gone.
    stale = stand_in(path)
    stale.x3d_fingerprint = None
    ok, why = PROJ["projection_surface_reusable"](stale, path)
    assert not ok and "no fingerprint" in why, why

    os.unlink(path)
    ok, why = PROJ["projection_surface_reusable"](proj, path)
    assert not ok, "a surface file that no longer exists was reused"


def test_the_fingerprint_counts_what_the_loader_will_count(tmpdir_path):
    """nverts/nfaces/bbox have to be the file's real content to be diagnosable.

    These are the numbers that let an operator line the report up against
    spatialnde2's own "Inplanemat calculation complete; numtris=..." and against
    where the part is. A plane patch is four vertices and two triangles.
    """
    path = write_patch(os.path.join(tmpdir_path, "d_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    fp = PROJ["projection_surface_fingerprint"](path)
    assert fp["error"] is None, fp["error"]
    assert (fp["nverts"], fp["nfaces"]) == (4, 2), fp
    lo, hi = fp["bbox"]
    assert np.allclose(lo, (0.797, -0.007, 0.941), atol=1e-9), lo
    assert np.allclose(hi, (0.828, 0.103, 0.941), atol=1e-9), hi
    assert len(fp["sha1"]) == 40

    # A file with no geometry in it reports the failure rather than a bbox.
    empty = os.path.join(tmpdir_path, "empty_uv.x3d")
    open(empty, "w").write("<X3D/>\n")
    fp = PROJ["projection_surface_fingerprint"](empty)
    assert fp["bbox"] is None and fp["error"], fp
    assert fp["sha1"] is not None, "the digest should still stand"


def test_the_offset_moves_the_surface_box(tmpdir_path):
    """The loaded surface is the file PLUS the registration offset.

    The same offset moves the toolpath, so a guard that compared the raw file
    box against an offset path would refuse a correctly placed mesh.
    """
    path = write_patch(os.path.join(tmpdir_path, "e_uv.x3d"),
                       (0.0, 0.0, 0.0), (0.1, 0.2, 0.0))
    proj = stand_in(path, offset=(1.0, -0.5, 0.25))
    lo, hi = proj.surface_world_bbox()
    assert np.allclose(lo, (1.0, -0.5, 0.25), atol=1e-9), lo
    assert np.allclose(hi, (1.1, -0.3, 0.25), atol=1e-9), hi


# ==============================================================================
# 3. The pre-flight guard: is the surface under the toolpath at all?
# ==============================================================================

def test_results16_is_refused(tmpdir_path):
    """The measured run, with its measured numbers.

    Surface: run B's flat plate, x 0.797..0.828  y -0.007..0.103  z 0.941.
    Toolpath: run C's, x 0.852..0.906  y 0.005..0.055  z 0.959..0.980.
    Disjoint in x by 24 mm, which is well past the lateral slack. The refusal
    has to name both boxes and the gap, because the operator's next question is
    always "which surface did it have, then?".
    """
    path = write_patch(os.path.join(tmpdir_path, "runb_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    proj = stand_in(path)
    tp = Path(positions((0.852, 0.005, 0.959), (0.906, 0.055, 0.980)))

    with pytest.raises(RuntimeError) as exc:
        PROJ["check_projection_covers_path"](proj, tp)
    msg = str(exc.value)
    assert "NOT UNDER THE TOOLPATH" in msg, msg
    assert "0.7970..0.8280" in msg, msg          # the surface it is holding
    assert "0.8520..0.9060" in msg, msg          # the path it was asked to scan
    assert "24.0" in msg, msg                    # the x gap, mm
    assert "2 triangles" in msg or "triangles" in msg, msg


def test_the_same_refusal_off_the_file_with_no_projector(tmpdir_path):
    """Asked at PLANNING time, before any projector exists and before motion.

    This is the one that saves the run: the projector is not built until the
    first reading has been staged, by which time the probe is on the part.
    """
    path = write_patch(os.path.join(tmpdir_path, "runb_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    tp = Path(positions((0.852, 0.005, 0.959), (0.906, 0.055, 0.980)))
    with pytest.raises(RuntimeError):
        PROJ["check_projection_covers_path"](path, tp)
    # ... and it agrees with the projector form about the gap.
    over = Path(positions((0.800, 0.000, 0.941), (0.825, 0.100, 0.941)))
    a = PROJ["check_projection_covers_path"](path, over)
    b = PROJ["check_projection_covers_path"](stand_in(path), over)
    assert a["ok"] and b["ok"]
    assert a["surface_bbox"] == b["surface_bbox"]
    assert a["separation_m"] == b["separation_m"] == 0.0


def test_an_overlapping_surface_is_accepted(tmpdir_path):
    """A contact path on the surface, which is the eddy-current case."""
    path = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    proj = stand_in(path)
    tp = Path(positions((0.800, 0.000, 0.941), (0.825, 0.100, 0.941)))
    out = PROJ["check_projection_covers_path"](proj, tp)
    assert out["ok"] and out["gap"] == (0.0, 0.0, 0.0), out
    assert "overlap" in out["reason"], out["reason"]


def test_a_legitimate_standoff_is_accepted(tmpdir_path):
    """50 mm off the surface is UTPlaneScan's NOETHER_Z_OFFSET, not a fault.

    This is the reason the vertical allowance is the projector's own ray window
    rather than a tolerance written down somewhere: a scan may legitimately
    stand off, and only the projector knows how far it will look.
    """
    path = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    proj = stand_in(path)
    for standoff in (0.0, 0.05, REACH_M - 0.001):
        tp = Path(positions((0.800, 0.000, 0.941 + standoff),
                            (0.825, 0.100, 0.941 + standoff)))
        out = PROJ["check_projection_covers_path"](proj, tp)
        assert out["ok"], "a %.0f mm standoff was refused: %s" % (
            standoff*1e3, out["reason"])


def test_past_the_ray_window_is_refused(tmpdir_path):
    """Beyond min_dist..max_dist no pose can hit the surface at any tilt."""
    path = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    proj = stand_in(path)
    tp = Path(positions((0.800, 0.000, 0.941 + REACH_M + 0.02),
                        (0.825, 0.100, 0.941 + REACH_M + 0.03)))
    with pytest.raises(RuntimeError) as exc:
        PROJ["check_projection_covers_path"](proj, tp)
    assert "ray window" in str(exc.value), str(exc.value)


def test_the_lateral_slack_is_where_it_says_it_is(tmpdir_path):
    """Just inside the slack passes, just outside it is refused.

    The slack is not a coverage allowance: the guard only ever fires when the
    two footprints do not overlap at all. It is there for a surface whose own
    footprint is narrower than the lateral reach of a tilted tool.
    """
    path = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.0, 0.0, 0.5), (0.1, 0.1, 0.5))
    proj = stand_in(path)

    inside = Path(positions((0.1 + SLACK_M*0.5, 0.0, 0.5),
                            (0.2, 0.1, 0.5)))
    assert PROJ["check_projection_covers_path"](proj, inside)["ok"]

    outside = Path(positions((0.1 + SLACK_M*2.0, 0.0, 0.5),
                             (0.2, 0.1, 0.5)))
    with pytest.raises(RuntimeError):
        PROJ["check_projection_covers_path"](proj, outside)


def test_a_plane_grid_is_accepted_as_a_toolpath(tmpdir_path):
    """The plane scan has (uu, vv, ww) grids, not a ScanPath.

    The guard has to work in all three scan modes or it is only installed in
    some of them, and the one it is missing from is the one that fails silently.
    """
    uu, vv = np.meshgrid(np.linspace(0.80, 0.82, 5), np.linspace(0.0, 0.09, 4))
    ww = np.full(uu.shape, 0.941)
    path = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    out = PROJ["check_projection_covers_path"](stand_in(path), (uu, vv, ww),
                                               what="scan grid")
    assert out["ok"], out["reason"]
    assert out["path_bbox"][0][2] == 0.941


def test_what_cannot_be_measured_is_reported_not_refused(tmpdir_path):
    """This guard must never be the thing that stops a scan it cannot grade.

    An unreadable surface extent, or a path with no finite positions, is a
    WARNING saying the check did not happen -- not a refusal, and not a silent
    pass either.
    """
    empty = os.path.join(tmpdir_path, "empty_uv.x3d")
    open(empty, "w").write("<X3D/>\n")
    tp = Path(positions((0.80, 0.0, 0.94), (0.82, 0.1, 0.94)))
    before = len(PROJ["_printed"])
    out = PROJ["check_projection_covers_path"](stand_in(empty), tp)
    assert out["ok"] and "not checked" in out["reason"], out
    said = "\n".join(PROJ["_printed"][before:])
    assert "WARNING" in said and "NOT checked" in said, said

    good = write_patch(os.path.join(tmpdir_path, "plate_uv.x3d"),
                       (0.797, -0.007, 0.941), (0.828, 0.103, 0.941))
    nan = Path(np.full((4, 3), np.nan))
    out = PROJ["check_projection_covers_path"](stand_in(good), nan)
    assert out["ok"] and "not checked" in out["reason"], out


def test_bbox_gap_is_zero_on_an_overlapping_axis():
    """The primitive, on its own, because every message quotes it."""
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert PROJ["bbox_gap"](a, ((0.5, 0.5, 0.5), (2.0, 2.0, 2.0))) == (0.0,)*3
    assert PROJ["bbox_gap"](a, ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))) == (1.0, 0.0, 0.0)
    # Symmetric, and it does not care which box is which.
    assert PROJ["bbox_gap"](((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)), a) == (1.0, 0.0, 0.0)


# ==============================================================================
# 4. restart() must re-seed the probe pose, and every scan mode must ask
# ==============================================================================

def test_restart_reseeds_the_probe_pose_before_reinstantiating():
    """A cleared image must not receive one stale pose from the previous scan.

    The probe pose channel is a math-function INPUT, so a freshly instantiated
    projector splats whatever is on it -- and after a scan that is the last pose
    of that scan. project_onto_parameterization only skips an INVALID
    orientation, so the seed has to be there, and it has to be there BEFORE the
    new math function is added in the same transaction.

    Structural rather than behavioural because instantiating the real thing
    needs a recording database and a loaded surface; what can be checked on the
    host is that the call is present and in the right order, which is the part
    that regressed.
    """
    restart = _method(_tree(PROJ_DPI), "SurfaceProjector", "restart",
                      "surface_projection.dpi")
    order = []
    for node in restart.body:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr == "_probe_pose_in_transaction":
                args = [ast.dump(a) for a in sub.args]
                assert any("snde_invalid_orientation3" in a for a in args), \
                    "restart() seeds the probe pose with something other than " \
                    "snde_invalid_orientation3()"
                order.append("seed")
            elif isinstance(fn, ast.Attribute) and fn.attr == "_instantiate_projection":
                order.append("instantiate")
    assert "seed" in order, \
        "restart() does not re-seed the probe pose; a stale pose from the " \
        "previous scan will be splatted into the cleared image"
    assert order.index("seed") < order.index("instantiate"), \
        "restart() seeds the probe pose AFTER re-instantiating the projection"


#: Every place a projector is reused or built, and must therefore ask both
#: questions. Three identical copies of the string comparison is what made
#: results16 a whole-pipeline failure rather than a one-mode one.
REUSE_SITES = (("ec_acquisition.dpi", EC_DPI),
               ("mesh_scan.dpi", MESH_DPI),
               ("scan_accumulation.dpi", ACC_DPI))


@pytest.mark.parametrize("name,path", REUSE_SITES)
def test_every_scan_mode_asks_both_questions(name, path):
    """No scan mode may decide reuse on the path string, or skip the guard."""
    src = open(path).read()
    assert "projection_surface_reusable(" in src, (
        "%s decides projector reuse without projection_surface_reusable(); a "
        "rewritten surface file will be reused as if unchanged" % (name,))
    assert "check_projection_covers_path(" in src, (
        "%s builds or reuses a projector without checking the surface against "
        "the toolpath" % (name,))
    assert 'x3d_path", None) ==' not in src and 'x3d_path", None) == x3d' not in src, (
        "%s still compares only the projector's x3d_path string" % (name,))
