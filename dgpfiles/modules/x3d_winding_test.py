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
# File:  x3d_winding_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Proves an inconsistently wound projection surface is detected and repaired
#   instead of aborting the process inside the geometry loader.
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
"""Is an unloadable projection surface caught before the loader sees it?

WHAT THIS PROTECTS. x3d_load_geometry() sorts the edges around each vertex using
the triangle winding, and one interior edge wound the same way by both of its
triangles makes that sort impossible. The loader does not raise: it fails an
assertion, so the process takes SIGABRT with no traceback. A scan in progress
loses its recording database with the probe on the part.

The condition arises in normal operation. Mesh cleaning turns individual faces
round so each one's normal agrees with the measured surface, and a face turned
that way disagrees with its neighbours about the edges they share.

The end-to-end proof -- that a repaired surface actually loads -- needs
spatialnde2 and is the last test here. Everything before it is pure geometry and
runs anywhere.
"""

import os
import sys
import tempfile

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import x3d_winding


X3D_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<X3D profile="Interchange" version="3.3">\n  <Scene>\n'
            '    <Transform>\n      <Shape>\n')
X3D_TAIL = ('        </IndexedFaceSet>\n      </Shape>\n'
            '    </Transform>\n  </Scene>\n</X3D>\n')


def _write(path, tris, verts, uv_tris=None):
    """An X3D IndexedFaceSet with the given winding."""
    ci = " ".join(" ".join(str(v) for v in t) + " -1" for t in tris)
    uv_tris = tris if uv_tris is None else uv_tris
    ti = " ".join(" ".join(str(v) for v in t) + " -1" for t in uv_tris)
    pts = " ".join("%.9g %.9g %.9g" % tuple(v) for v in verts)
    uvs = " ".join("%.9g %.9g" % (v[0], v[1]) for v in verts)
    with open(path, "w") as fh:
        fh.write(X3D_HEAD)
        fh.write('        <IndexedFaceSet solid="false" coordIndex="%s" '
                 'texCoordIndex="%s">\n' % (ci, ti))
        fh.write('          <Coordinate point="%s"/>\n' % pts)
        fh.write('          <TextureCoordinate point="%s"/>\n' % uvs)
        fh.write(X3D_TAIL)
    return path


def _grid(n=6):
    """An n x n vertex height-field patch, consistently wound, as (verts, tris)."""
    verts = [(float(i), float(j), 0.0) for j in range(n) for i in range(n)]
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j*n + i
            b = a + 1
            c = a + n
            d = c + 1
            tris.append([a, b, d])
            tris.append([a, d, c])
    return verts, tris


def _tmp(name):
    return os.path.join(tempfile.mkdtemp(prefix="x3dwind_"), name)


def test_a_consistent_surface_is_reported_loadable_and_left_alone():
    verts, tris = _grid()
    p = _write(_tmp("clean.x3d"), tris, verts)
    rep = x3d_winding.winding_report(x3d_winding.read_indexed_face_set(p)[0])
    assert rep["inconsistent_edges"] == 0
    assert rep["overshared_edges"] == 0
    assert rep["boundary_edges"] > 0, "a bounded patch must have a boundary"
    assert rep["loadable"] is True
    # No repair, and no sibling file written.
    assert x3d_winding.loadable_x3d(p, quiet=True) == p


def test_one_reversed_face_is_detected():
    """The exact defect that abort the loader, at its smallest."""
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    tris[7] = [tris[7][0], tris[7][2], tris[7][1]]
    p = _write(_tmp("onebad.x3d"), tris, verts)
    rep = x3d_winding.winding_report(x3d_winding.read_indexed_face_set(p)[0])
    assert rep["inconsistent_edges"] > 0, (
        "a single reversed interior face left the winding looking consistent, so "
        "this check would not have caught the defect that aborts the loader")
    assert rep["loadable"] is False


def test_repair_clears_it_and_changes_no_geometry():
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    for k in (3, 7, 12):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    p = _write(_tmp("threebad.x3d"), tris, verts)

    out = x3d_winding.loadable_x3d(p, quiet=True)
    assert out != p, "the repair returned the original file unchanged"

    after = x3d_winding.winding_report(x3d_winding.read_indexed_face_set(out)[0])
    assert after["inconsistent_edges"] == 0
    assert after["loadable"] is True

    before_f = x3d_winding.read_indexed_face_set(p)[0]
    after_f = x3d_winding.read_indexed_face_set(out)[0]
    assert len(before_f) == len(after_f)
    assert (sorted(tuple(sorted(t)) for t in before_f)
            == sorted(tuple(sorted(t)) for t in after_f)), (
        "the repair changed which vertices are joined, not just the order they "
        "are written in")


def test_repair_moves_as_few_faces_as_possible():
    """Of the two consistent windings, the one nearer the input is kept.

    Flipping the whole surface is also consistent and is the wrong answer: it
    turns the outside of the part into the inside.
    """
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    for k in (3, 7, 12):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    flip, rep = x3d_winding.orient_consistently(tris)
    assert rep["faces_reversed"] == 3, (
        "expected the 3 turned faces to be turned back, got %d of %d"
        % (rep["faces_reversed"], rep["faces"]))
    assert sorted(np.nonzero(flip)[0].tolist()) == [3, 7, 12]


def test_texture_indices_are_reversed_with_their_faces():
    """A face written the other way round takes its texture triangle with it.

    Reverse one and not the other and the projected image shears across that
    triangle, which is far harder to spot than a crash.
    """
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    tris[5] = [tris[5][0], tris[5][2], tris[5][1]]
    p = _write(_tmp("tex.x3d"), tris, verts)
    out = x3d_winding.loadable_x3d(p, quiet=True)
    faces, extra = x3d_winding.read_indexed_face_set(out)
    assert "texCoordIndex" in extra
    for f, t in zip(faces, extra["texCoordIndex"]):
        assert list(f) == list(t), (
            "coordIndex and texCoordIndex disagree after the repair: %s vs %s"
            % (f, t))


def test_an_edge_shared_by_three_faces_is_refused_not_repaired():
    """No single surface there, so there is nothing to orient consistently."""
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    # An edge already used by two faces, so a third makes it over-shared. A
    # boundary edge would only reach two and would prove nothing.
    used = {}
    for t in tris:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(e), max(e))
            used[k] = used.get(k, 0) + 1
    interior = next(k for k, n in sorted(used.items()) if n == 2)
    tris.append([interior[0], interior[1], len(verts)])
    verts = list(verts) + [(0.5, 0.5, 1.0)]
    p = _write(_tmp("overshared.x3d"), tris, verts)
    rep = x3d_winding.winding_report(x3d_winding.read_indexed_face_set(p)[0])
    assert rep["overshared_edges"] >= 1
    assert rep["loadable"] is False
    try:
        x3d_winding.loadable_x3d(p, quiet=True)
    except RuntimeError as e:
        assert "more than two" in str(e)
    else:
        raise AssertionError(
            "an edge shared by three faces was accepted; the loader aborts the "
            "process on this")


def test_a_repaired_surface_actually_loads():
    """End to end, against the real loader. This is the test that matters.

    Everything above is a proxy for "the process will not abort". This asks the
    loader.
    """
    try:
        import spatialnde2 as snde
    except ImportError:
        import pytest
        pytest.skip("spatialnde2 is not importable, so the loader cannot be "
                    "asked whether the repair worked")

    verts, tris = _grid(8)
    tris = [list(t) for t in tris]
    for k in (4, 11, 20, 33):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    p = _write(_tmp("load.x3d"), tris, verts)
    assert not x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(p)[0])["loadable"]

    out = x3d_winding.loadable_x3d(p, quiet=True)

    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()
    graphman = snde.graphics_storage_manager(
        "/graphics/", recdb.lowlevel_alloc, recdb.alignment_requirements,
        recdb.lockmgr, 1e-8, 2000000000)
    trans = recdb.start_transaction()
    trans.define_channel("/graphics/", "main", False, graphman)
    snde.x3d_load_geometry(
        trans, graphman, out, 0, "main", "/graphics/specimen/", None,
        ["reindex_vertices", "reindex_tex_vertices"]
        + snde.project_onto_parameterization_processing_tags)
    trans.end_transaction().globalrev_wait().wait_complete()
    assert "/graphics/specimen/meshed" in set(recdb.latest.list_recordings())


def test_a_projector_builds_on_a_surface_that_would_abort():
    """The guard is actually wired into the thing that loads surfaces.

    The repair is worth nothing if SurfaceProjector does not call it, and that
    cannot be checked by reading the code: the failure is a signal, so a wrong
    answer here takes the test runner down with it rather than failing.
    """
    try:
        import spatialnde2 as snde                          # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("spatialnde2 is not importable, so SurfaceProjector cannot "
                    "be built")

    ns = {}
    exec("import os, sys, threading, time\n"
         "import numpy as np\nimport spatialnde2 as snde\n", ns)
    src = open(os.path.join(MODULE_DIR, "surface_projection.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    exec(compile(src, "surface_projection.dpi", "exec"), ns)

    verts, tris = _grid(8)
    tris = [list(t) for t in tris]
    for k in (4, 11, 20, 33):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    # Metres, so the patch is the size of a real specimen rather than 8 m across.
    verts = [(v[0]*0.01, v[1]*0.01, v[2]) for v in verts]
    p = _write(_tmp("projector.x3d"), tris, verts)
    assert not x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(p)[0])["loadable"], (
        "the surface this test builds is already loadable, so it does not "
        "exercise the guard")

    import spatialnde2 as snde
    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()

    proj = ns["SurfaceProjector"](recdb, p, prefix="/guard/proj",
                                  horizontal_pixels=16, vertical_pixels=16)
    assert proj.x3d_path == p, (
        "the projector recorded the repaired path as its surface; callers compare "
        "against the path they asked for to decide whether to reuse it")
    assert snde.recdb_path_join(proj.model_tree, "meshed") in set(
        recdb.latest.list_recordings())
    proj.teardown()


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (name, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
