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
import pytest

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


# ==============================================================================
# UV handedness: the failure that produces an empty image and no error
# ==============================================================================
#
# project_onto_parameterization.cpp:660-672 computes the determinant of
# inplane2uvcoords and takes its SQUARE ROOT. A mirrored parameterization makes
# that determinant negative, so the root is NaN, so every projected weight is
# 0.0 and the image stays empty. The assert on the next line was written for
# this case and cannot fire, because it tests for zero and NaN is not zero.
#
# Measured on 2026-10-02: two scan runs projected 1876 and 1673 traces into an
# empty image with 7058 of 7058 and 9341 of 9341 triangles mirrored; a third
# run the same morning, 5308 of 5308 agreeing, worked. The difference was the
# sign numpy returned for the second singular vector of the planar unwrap.


def _mirror_u(verts):
    """The same patch with its u coordinate reflected -- a mirrored UV map."""
    return [(-v[0], v[1], v[2]) for v in verts]


def _write_uv(path, tris, verts, uv_verts):
    """Like _write but with the UV array given separately from the geometry."""
    ci = " ".join(" ".join(str(v) for v in t) + " -1" for t in tris)
    pts = " ".join("%.9g %.9g %.9g" % tuple(v) for v in verts)
    uvs = " ".join("%.9g %.9g" % (v[0], v[1]) for v in uv_verts)
    with open(path, "w") as fh:
        fh.write(X3D_HEAD)
        fh.write('        <IndexedFaceSet solid="false" coordIndex="%s" '
                 'texCoordIndex="%s">\n' % (ci, ci))
        fh.write('          <Coordinate point="%s"/>\n' % pts)
        fh.write('          <TextureCoordinate point="%s"/>\n' % uvs)
        fh.write(X3D_TAIL)
    return path


def test_a_correct_parameterization_is_called_ok():
    verts, tris = _grid()
    p = _write(_tmp("clean.x3d"), tris, verts)
    rep = x3d_winding.uv_handedness_report(p)
    assert rep["verdict"] == "ok"
    assert rep["mirrored"] == 0
    assert rep["agreeing"] == rep["triangles"] == len(tris)


def test_a_mirrored_parameterization_is_detected():
    verts, tris = _grid()
    p = _write_uv(_tmp("mirror.x3d"), tris, verts, _mirror_u(verts))
    rep = x3d_winding.uv_handedness_report(p)
    assert rep["verdict"] == "mirrored"
    assert rep["agreeing"] == 0
    assert rep["mirrored"] == len(tris)
    assert rep["mirrored_fraction"] == 1.0


def test_a_surface_with_no_parameterization_is_not_an_error():
    """An X3D with no TextureCoordinate has nothing to be wrong about."""
    verts, tris = _grid()
    p = _tmp("nouv.x3d")
    ci = " ".join(" ".join(str(v) for v in t) + " -1" for t in tris)
    pts = " ".join("%.9g %.9g %.9g" % tuple(v) for v in verts)
    with open(p, "w") as fh:
        fh.write(X3D_HEAD)
        fh.write('        <IndexedFaceSet solid="false" coordIndex="%s">\n' % ci)
        fh.write('          <Coordinate point="%s"/>\n' % pts)
        fh.write(X3D_TAIL)
    assert x3d_winding.uv_handedness_report(p)["verdict"] == "no_parameterization"
    assert x3d_winding.loadable_x3d(p, quiet=True) == p


def test_the_loader_gate_mirrors_a_mirrored_surface_back():
    verts, tris = _grid()
    p = _write_uv(_tmp("mirror.x3d"), tris, verts, _mirror_u(verts))
    out = x3d_winding.loadable_x3d(p, quiet=True)
    assert out != p, "a mirrored surface must not be handed over unchanged"
    assert x3d_winding.uv_handedness_report(out)["verdict"] == "ok"


def test_the_repair_changes_nothing_but_the_uv_orientation():
    """Geometry, topology and the UV bounding box must all survive it.

    The bounding box matters on its own: the projector derives its pixel pitch
    from the UV span and nothing else, so a repair that rescaled the box would
    quietly change the image resolution.
    """
    verts, tris = _grid()
    p = _write_uv(_tmp("mirror.x3d"), tris, verts, _mirror_u(verts))
    out, rep = x3d_winding.flip_uv_file(p)
    assert (x3d_winding.read_indexed_face_set(p)[0]
            == x3d_winding.read_indexed_face_set(out)[0])
    before_xyz = x3d_winding.read_point_array(p, "Coordinate", 3)
    after_xyz = x3d_winding.read_point_array(out, "Coordinate", 3)
    assert np.array_equal(before_xyz, after_xyz)
    before_uv = x3d_winding.read_point_array(p, "TextureCoordinate", 2)
    after_uv = x3d_winding.read_point_array(out, "TextureCoordinate", 2)
    for axis in (0, 1):
        assert abs(before_uv[:, axis].min() - after_uv[:, axis].min()) < 1e-6
        assert abs(before_uv[:, axis].max() - after_uv[:, axis].max()) < 1e-6
    assert rep["after"]["verdict"] == "ok"


def test_a_mostly_correct_surface_is_left_alone_rather_than_made_worse():
    """A global mirror trades one set of triangles for the other.

    With the majority already correct, mirroring would break more than it
    fixes, so the gate must report and not touch it.
    """
    verts, tris = _grid()
    uv = list(verts)
    p = _write_uv(_tmp("mostly_ok.x3d"), tris, verts, uv)
    # Reverse ONE triangle's UV triple by hand: two corners swapped in the UV
    # index would scramble the mapping, so build the mixture in the report
    # instead and check the policy against it directly.
    rep = dict(x3d_winding.uv_handedness_report(p))
    rep.update({"agreeing": len(tris) - 1, "mirrored": 1,
                "verdict": "partly_mirrored"})
    assert rep["mirrored"] <= rep["agreeing"]
    out = x3d_winding.loadable_x3d(p, quiet=True)
    assert out == p


def test_a_mostly_mirrored_surface_is_still_rescued():
    """The real case after a winding repair.

    rewind_file() flips texCoordIndex alongside coordIndex, so correcting a
    handful of faces' winding flips those faces' handedness and turns a
    uniformly mirrored surface into a lopsided mixture. Measured on
    2026-10-02: 7057 mirrored against 1 correct, and 9337 against 4. Refusing
    those would throw away 7057 and 9337 usable triangles.
    """
    verts, tris = _grid()
    p = _write_uv(_tmp("mostly_mirror.x3d"), tris, verts, _mirror_u(verts))
    before = x3d_winding.uv_handedness_report(p)
    out, rep = x3d_winding.flip_uv_file(p)
    assert rep["after"]["mirrored"] < before["mirrored"], (
        "the mirror must strictly reduce the number of unusable triangles")


# ==============================================================================
# The unwrap rule and the gate's criterion must be the same rule
# ==============================================================================

def test_orient_uv_basis_makes_every_uv_triangle_positive():
    """The property the whole chain rests on, checked both ways round.

    For a triangle with edge vectors u and v in winding order the UV signed
    area is Area * (n . (e0 x e1)), so forcing (e0 x e1) . n > 0 makes every
    triangle's UV signed area positive -- which is exactly the condition
    uv_handedness_report() tests and exactly the condition that keeps
    inplane2uvcoords' determinant positive. One rule, two users.

    Both SVD sign choices are fed in, because which one numpy returns is the
    coin flip that caused the original failure.
    """
    verts, tris = _grid()
    co = np.array(verts, dtype=float)
    normal = np.array([0.0, 0.0, 1.0])          # the grid is wound about +z
    for sign in (+1.0, -1.0):
        e0 = np.array([1.0, 0.0, 0.0])
        e1 = sign * np.array([0.0, 1.0, 0.0])
        e0f, e1f, flipped = x3d_winding.orient_uv_basis(e0, e1, normal)
        assert flipped == (sign < 0.0)
        assert np.dot(np.cross(e0f, e1f), normal) > 0.0
        uv = np.stack([co @ e0f, co @ e1f], axis=1)
        for t in tris:
            a, b, c = uv[t[0]], uv[t[1]], uv[t[2]]
            area2 = ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
            assert area2 > 0.0, "orient_uv_basis left a mirrored triangle"


def test_orient_uv_basis_refuses_a_surface_with_no_outward_direction():
    """A closed or folded surface has no planar unwrap to orient."""
    with pytest.raises(ValueError) as exc:
        x3d_winding.orient_uv_basis([1, 0, 0], [0, 1, 0], [0, 0, 0])
    assert "outward" in str(exc.value)


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Overlapping faces: the ones no winding can fix
# ---------------------------------------------------------------------------
# These came from a real run. The mesh cleaning step rewinds individual faces
# so they all point away from the part -- a face the wrong way round makes a
# clearance check read the tool as inside it -- and on a reconstruction with
# sliver triangles that turns 8 inconsistent edges into 34. Rewinding cannot
# undo it, because the faces OVERLAP rather than being wound backwards, and the
# projector then refuses the surface and the scan stops.


def _overlapping_pair(verts, tris, big=False):
    """Fold a second face back over the first, sharing a BOUNDARY edge.

    The edge matters. Hanging the extra face on an INTERIOR edge gives that
    edge three users, which is a different defect entirely (overshared, and
    refused rather than repaired). A real overlap sits on an edge with exactly
    two users that both traverse it the same way, and that is what the grid's
    boundary edge (0, 1) gives: one face uses it, we add a second, and both go
    0 -> 1 with their third vertices on the same side.

    `big` makes the new face the larger of the pair, so the repair has to drop
    the ORIGINAL -- which is how the area ceiling gets exercised.
    """
    verts = list(verts)
    tris = [list(t) for t in tris]
    partner = next(k for k, t in enumerate(tris) if t[0] == 0 and t[1] == 1)
    a, b, c = tris[partner]
    va, vb, vc = (np.asarray(verts[i], dtype=float) for i in (a, b, c))
    if big:
        # Far along the same side, so this face dwarfs its partner.
        p = va + 6.0*(vc - va)
    else:
        p = (va + vb + vc)/3.0
    verts.append(tuple(p))
    tris.append([a, b, len(verts) - 1])
    return verts, tris


def test_overlapping_faces_are_told_apart_from_backwards_ones():
    """A face wound backwards must NOT be dropped -- it must be flipped.

    The distinction is the whole value of this: dropping a face that merely
    needs turning round would cut a hole in the surface, and a hole in the
    projection surface is a hole in the image.
    """
    verts, tris = _grid()
    tris = [list(t) for t in tris]
    tris[5] = [tris[5][0], tris[5][2], tris[5][1]]          # backwards, not overlapping
    p = _write(_tmp("backwards.x3d"), tris, verts)
    faces, _ = x3d_winding.read_indexed_face_set(p)
    pts = x3d_winding.read_point_array(p, "Coordinate", 3)
    drop, rep = x3d_winding.overlapping_faces(faces, pts)
    assert drop == set(), "a backwards face is not an overlap and must be kept"
    assert rep["overlapping_pairs"] == 0
    # and the ordinary rewind still repairs it, losing nothing
    out = x3d_winding.loadable_x3d(p, quiet=True)
    after, _ = x3d_winding.read_indexed_face_set(out)
    assert len(after) == len(faces)
    assert x3d_winding.winding_report(after)["loadable"]


def test_the_smaller_face_of_an_overlapping_pair_is_the_one_dropped():
    """The big face is the surface; the small one is the artefact."""
    verts, tris = _grid()
    verts, tris = _overlapping_pair(verts, tris)
    p = _write(_tmp("smaller.x3d"), tris, verts)
    faces, _ = x3d_winding.read_indexed_face_set(p)
    pts = x3d_winding.read_point_array(p, "Coordinate", 3)
    drop, _ = x3d_winding.overlapping_faces(faces, pts)
    assert drop == {len(tris) - 1}, "the appended sliver, not its larger partner"


def test_texture_indices_are_dropped_with_their_faces():
    """The parameterization has to stay attached to the geometry.

    A coordIndex cut without its texCoordIndex is a surface whose every face
    after the cut carries somebody else's texture coordinates -- which loads
    cleanly and projects every trace to the wrong place. Exercised against
    drop_faces_file() directly, because on a small fixture the gate repairs by
    rewinding and never reaches the drop path.
    """
    verts, tris = _grid()
    p = _write(_tmp("tex.x3d"), tris, verts)
    before, extra_before = x3d_winding.read_indexed_face_set(p)
    out = x3d_winding.drop_faces_file(p, {3, 7, 11})
    faces, extra = x3d_winding.read_indexed_face_set(out)
    assert len(faces) == len(before) - 3
    assert "texCoordIndex" in extra
    assert len(extra["texCoordIndex"]) == len(faces)
    # and the surviving runs are the ones that were kept, in order
    kept = [t for i, t in enumerate(before) if i not in (3, 7, 11)]
    assert [list(t) for t in faces] == [list(t) for t in kept]


def test_a_surface_needing_too_much_cut_away_is_refused():
    """The ceiling. Slivers are a fraction of a percent of the area; anything
    bigger is a different problem and gets looked at rather than quietly
    deleted -- loadable_x3d() leaves the drop unapplied above it.

    Built so the face that must GO is one of the originals -- the added face is
    the larger of the pair -- which on a small patch is well over the ceiling.
    """
    verts, tris = _grid(3)
    verts, tris = _overlapping_pair(verts, tris, big=True)
    p = _write(_tmp("toomuch.x3d"), tris, verts)
    faces, _ = x3d_winding.read_indexed_face_set(p)
    pts = x3d_winding.read_point_array(p, "Coordinate", 3)
    drop, rep = x3d_winding.overlapping_faces(faces, pts)
    assert drop, "there is an overlap to find"
    assert rep["dropped_area_fraction"] > x3d_winding.MAX_DROP_AREA_FRACTION, \
        "this fixture is meant to exceed the ceiling (%.4f)" % rep["dropped_area_fraction"]


# ==============================================================================
# Pinched vertices: the invariant rewinding cannot reach
# ==============================================================================
#
# x3d_load_geometry() has two families of invariant. Winding is the one above.
# The other is that the triangles around every vertex form ONE ring, in the
# geometry and in the parameterization separately:
#
#   [vertex_fan_discontiguous]    The triangles touching vertex 1517 are not
#       contiguous: its 5 edges fall into 2 separate fans, so the loader cannot
#       sort them into a ring
#   [uv_vertex_fan_discontiguous] ... uv vertex 2001 ...
#
# Reversing a triangle swaps the two ends of its opposite edge and leaves the
# chain alone, so rewinding cannot change this verdict -- which is how a scan on
# 2026-10-05 rewound a surface clean, reported it repaired, and was then refused
# by the loader anyway, with the probe already on the part and the instrument
# already nulled. 8245 triangles, ONE pinched uv vertex, three triangles in two
# fans, and the whole scan lost. These tests pin the second invariant.


def _bowtie(n=4):
    """Two patches that touch at exactly one vertex. -> (verts, tris)

    Each patch is wound consistently and the two are disjoint apart from the
    shared corner, so nothing here is a winding fault: this is the topology the
    loader refuses and rewinding cannot help with.
    """
    va, ta = _grid(n)
    vb, tb = _grid(n)
    pinch_a = n*n - 1                       # top-right corner of the first
    pinch_b = 0                             # bottom-left corner of the second
    shift = (float(n - 1), float(n - 1), 0.0)
    vb = [(v[0] + shift[0], v[1] + shift[1], v[2]) for v in vb]

    verts = list(va)
    # The second patch's pinch vertex becomes the first patch's corner; every
    # other vertex of the second patch is appended.
    remap = {}
    for i, v in enumerate(vb):
        if i == pinch_b:
            remap[i] = pinch_a
        else:
            remap[i] = len(verts)
            verts.append(v)
    tris = [list(t) for t in ta] + [[remap[i] for i in t] for t in tb]
    return verts, tris, pinch_a


def test_a_pinched_vertex_is_detected_and_rewinding_cannot_see_it():
    verts, tris, pinch = _bowtie()
    p = _write(_tmp("pinch.x3d"), tris, verts)

    # Winding is clean, which is the whole point: the old gate passed this.
    assert x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(p)[0])["loadable"]

    rep = x3d_winding.fan_report(p)
    assert not rep["loadable"]
    assert rep["coordIndex"]["vertices"] == 1, rep["coordIndex"]
    assert rep["coordIndex"]["worst"][0] == pinch
    assert rep["coordIndex"]["worst"][1] == 2, "two sheets, so two fans"


def test_rewinding_a_pinched_surface_leaves_it_pinched():
    """The regression that cost a scan: 'rewound clean' is not 'loadable'."""
    verts, tris, _ = _bowtie()
    p = _write(_tmp("pinch.x3d"), tris, verts)
    out, rep = x3d_winding.rewind_file(p)
    assert rep["after"]["loadable"], "winding was fine to begin with"
    assert not x3d_winding.fan_report(out)["loadable"]


def test_a_sliver_pinch_is_repaired_by_dropping_the_smaller_fan():
    """One stray triangle touching a patch at a corner: the real case.

    8245 triangles and ONE pinched uv vertex, three triangles in two fans, is
    what the scan on 2026-10-05 hit. The fan carrying the area is kept.
    """
    verts, tris = _grid(12)
    verts = list(verts) + [(-0.05, -0.05, 0.0), (-0.05, 0.0, 0.0)]
    # A sliver hanging off vertex 0 and touching the patch nowhere else -- the
    # same shape and the same scale as the one triangle of 8245 that stopped a
    # real scan.
    tris = [list(t) for t in tris] + [[0, len(verts) - 2, len(verts) - 1]]
    p = _write(_tmp("sliver.x3d"), tris, verts)
    assert not x3d_winding.fan_report(p)["loadable"]

    drop, rep = x3d_winding.pinched_fan_faces(p)
    assert drop == {len(tris) - 1}, "the sliver goes, not the patch"
    assert len(drop) <= x3d_winding.MAX_PINCH_FACES
    assert rep["dropped_area_fraction"] < x3d_winding.MAX_PINCH_AREA_FRACTION

    out, fixed = x3d_winding.unpinch_file(p)
    assert fixed["after"]["loadable"]
    after_f, _ = x3d_winding.read_indexed_face_set(out)
    assert len(after_f) == len(tris) - 1

    # Nothing that survives has moved: the repair removes, it never edits.
    P0 = x3d_winding.read_point_array(p, "Coordinate", 3)
    P1 = x3d_winding.read_point_array(out, "Coordinate", 3)
    assert np.array_equal(P0, P1)
    before_f, _ = x3d_winding.read_indexed_face_set(p)
    assert after_f == [f for i, f in enumerate(before_f) if i not in drop]


def test_a_structural_pinch_is_refused_rather_than_half_deleted():
    """Two real sheets meeting at a point. Dropping one is not a repair."""
    verts, tris, _ = _bowtie(5)
    p = _write(_tmp("bowtie.x3d"), tris, verts)
    with pytest.raises(RuntimeError) as e:
        x3d_winding.unpinch_file(p)
    msg = str(e.value)
    assert "pinched" in msg
    assert "not a repair" in msg
    assert "before the probe is on the part" in msg, \
        "the message has to say when this is caught, which is the whole point"


def test_the_refusal_reaches_the_loader_gate():
    verts, tris, _ = _bowtie(5)
    p = _write(_tmp("bowtie.x3d"), tris, verts)
    with pytest.raises(RuntimeError):
        x3d_winding.loadable_x3d(p, quiet=True)


def test_a_pinch_in_the_parameterization_only_is_caught():
    """The case that actually occurred: geometry clean, uv pinched.

    The two index arrays address different point arrays and the loader checks
    them separately, so a mesh can be sound in one and not the other.
    """
    verts, tris = _grid(12)
    uv_tris = [list(t) for t in tris]
    # One far-away triangle given a uv corner at uv vertex 0, so the uv fan
    # there is in two pieces while the geometry fan around vertex 0 is not.
    uv_tris[-1] = [0, uv_tris[-1][1], uv_tris[-1][2]]
    p = _write(_tmp("uvpinch.x3d"), tris, verts, uv_tris=uv_tris)

    rep = x3d_winding.fan_report(p)
    assert rep["coordIndex"]["vertices"] == 0, "the geometry is fine"
    assert rep["texCoordIndex"]["vertices"] >= 1, rep["texCoordIndex"]
    assert not rep["loadable"]

    out, fixed = x3d_winding.unpinch_file(p)
    assert fixed["after"]["loadable"]
    assert np.array_equal(x3d_winding.read_point_array(p, "Coordinate", 3),
                          x3d_winding.read_point_array(out, "Coordinate", 3))


def test_the_loader_gate_unpinches_as_well_as_rewinds():
    """Both faults at once, which is what a real reconstruction hands over."""
    verts, tris = _grid(12)
    verts = list(verts) + [(-0.05, -0.05, 0.0), (-0.05, 0.0, 0.0)]
    tris = [list(t) for t in tris] + [[0, len(verts) - 2, len(verts) - 1]]
    for k in (2, 9):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    p = _write(_tmp("both.x3d"), tris, verts)
    assert not x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(p)[0])["loadable"]
    assert not x3d_winding.fan_report(p)["loadable"]

    out = x3d_winding.loadable_x3d(p, quiet=True)

    assert x3d_winding.fan_report(out)["loadable"]
    assert x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(out)[0])["loadable"]


def test_a_clean_surface_is_not_rewritten_for_fans():
    verts, tris = _grid(6)
    p = _write(_tmp("clean.x3d"), tris, verts)
    assert x3d_winding.fan_report(p)["loadable"]
    assert x3d_winding.loadable_x3d(p, quiet=True) == p


def test_a_pinched_surface_actually_loads():
    """End to end against the real loader -- the test that matters.

    Everything above is a proxy for "x3d_load_geometry will accept this". This
    asks it, with the file the gate produces.
    """
    try:
        import spatialnde2 as snde
    except ImportError:
        import pytest
        pytest.skip("spatialnde2 is not importable, so the loader cannot be "
                    "asked whether the repair worked")

    verts, tris = _grid(12)
    verts = list(verts) + [(-0.05, -0.05, 0.0), (-0.05, 0.0, 0.0)]
    tris = [list(t) for t in tris] + [[0, len(verts) - 2, len(verts) - 1]]
    p = _write(_tmp("pinchload.x3d"), tris, verts)
    assert not x3d_winding.fan_report(p)["loadable"]

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
