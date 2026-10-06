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
# File:  projection_surface_plan_time_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-06
# Description:
#   Tests that the projection surface is proved loadable where it is
#   written, and that both halves of the check are needed.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-06:  Initial Commit
#
#########################################################################
"""The surface is checked where it is written, with two checks, both needed.

SurfaceProjector is built when the SCAN starts -- after the approach, the
touchdown and the hardware null -- so a surface x3d_load_geometry() will not take
costs the whole setup with the probe pressed into the part. Three runs went that
way on 2026-10-05 and 2026-10-06, and the file had existed since the
parameterization minutes earlier each time.

THE TWO CHECKS ARE COMPLEMENTARY AND THAT IS MEASURED, not assumed. The headline
test here is test_neither_check_alone_is_enough: the 2026-10-06 scan surface had
219 inconsistently wound edges, passed x3d_validate_geometry() with ZERO
problems, and ABORTED THE PROCESS when loaded. Winding is not in the loader's
validated list; it is still a bare assert. Conversely the validator covers
out-of-range indices, degenerate triangles and orphan vertices, which
x3d_winding.py knows nothing about.

The load half needs spatialnde2 and is skipped without it.
"""
import functools
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import x3d_winding                                              # noqa: E402

X3D_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<X3D profile="Interchange" version="3.3">\n  <Scene>\n'
            '    <Transform>\n      <Shape>\n')
X3D_TAIL = ('        </IndexedFaceSet>\n      </Shape>\n'
            '    </Transform>\n  </Scene>\n</X3D>\n')


def _write(path, tris, verts):
    ci = " ".join(" ".join(str(v) for v in t) + " -1" for t in tris)
    pts = " ".join("%.9g %.9g %.9g" % tuple(v) for v in verts)
    uvs = " ".join("%.9g %.9g" % (v[0], v[1]) for v in verts)
    with open(path, "w") as fh:
        fh.write(X3D_HEAD)
        fh.write('        <IndexedFaceSet solid="false" coordIndex="%s" '
                 'texCoordIndex="%s">\n' % (ci, ci))
        fh.write('          <Coordinate point="%s"/>\n' % pts)
        fh.write('          <TextureCoordinate point="%s"/>\n' % uvs)
        fh.write(X3D_TAIL)
    return path


def _grid(n=8):
    verts = [(float(i), float(j), 0.0) for j in range(n) for i in range(n)]
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j*n + i
            tris.append([a, a + 1, a + n + 1])
            tris.append([a, a + n + 1, a + n])
    return verts, tris


def _snde():
    try:
        import spatialnde2
        return spatialnde2
    except ImportError:
        pytest.skip("spatialnde2 is not importable")


def test_the_validator_does_not_see_inconsistent_winding():
    """Half of why both checks are needed. The loader does not validate winding.

    This is the specific hole: x3d_validate_geometry() reports ZERO problems on a
    surface whose winding makes x3d_load_geometry() fail an assert. A plan-time
    gate built on the validator alone would pass it, and the scan would then take
    the whole process down -- which is worse than the refusal it replaced.
    """
    snde = _snde()
    verts, tris = _grid(8)
    tris = [list(t) for t in tris]
    for k in (4, 11, 20, 33):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    p = _write(os.path.join(tempfile.mkdtemp(), "wound.x3d"), tris, verts)

    assert not x3d_winding.winding_report(
        x3d_winding.read_indexed_face_set(p)[0])["loadable"], \
        "the fixture is not inconsistently wound"

    tags = (["reindex_vertices", "reindex_tex_vertices"]
            + list(snde.project_onto_parameterization_processing_tags))
    problems = [q for q in snde.x3d_validate_geometry(p, 0, tags) if q.fatal]
    assert not problems, \
        ("x3d_validate_geometry now catches inconsistent winding (%s). That is "
         "good news, and it means the repair ordering in check_projection_"
         "surface() can be revisited -- but read the docstring first."
         % [q.check for q in problems])


def test_the_repair_does_not_see_what_the_validator_does():
    """The other half: an index fault x3d_winding.py has no opinion about."""
    snde = _snde()
    verts, tris = _grid(8)
    tris = [list(t) for t in tris] + [[0, 1, 99999]]        # out of range
    p = _write(os.path.join(tempfile.mkdtemp(), "oob.x3d"), tris, verts)

    tags = (["reindex_vertices", "reindex_tex_vertices"]
            + list(snde.project_onto_parameterization_processing_tags))
    kinds = {q.check for q in snde.x3d_validate_geometry(p, 0, tags) if q.fatal}
    assert kinds, "the fixture is not invalid"
    assert "coord_index_out_of_range" in kinds, kinds


def test_neither_check_alone_is_enough():
    """Stated as one assertion, because it is the reason the gate has two halves.

    Run in a SUBPROCESS: proving the point means loading a surface that aborts
    the process, and an assert failure in C++ takes the test runner with it.
    """
    _snde()
    verts, tris = _grid(8)
    tris = [list(t) for t in tris]
    for k in (4, 11, 20, 33):
        tris[k] = [tris[k][0], tris[k][2], tris[k][1]]
    d = tempfile.mkdtemp()
    p = _write(os.path.join(d, "wound.x3d"), tris, verts)

    prog = textwrap.dedent("""
        import sys
        import spatialnde2 as snde
        f = sys.argv[1]
        tags = (["reindex_vertices", "reindex_tex_vertices"]
                + list(snde.project_onto_parameterization_processing_tags))
        assert not [q for q in snde.x3d_validate_geometry(f, 0, tags) if q.fatal]
        recdb = snde.recdatabase()
        snde.setup_cpu(recdb, [], 2)
        snde.setup_storage_manager(recdb)
        snde.setup_math_functions(recdb, [])
        recdb.startup()
        gm = snde.graphics_storage_manager("/g/", recdb.lowlevel_alloc,
                                           recdb.alignment_requirements,
                                           recdb.lockmgr, 1e-8, 2000000000)
        tr = recdb.start_transaction()
        tr.define_channel("/g/", "main", False, gm)
        snde.x3d_load_geometry(tr, gm, f, 0, "main", "/g/s/", None, tags)
        tr.end_transaction().globalrev_wait().wait_complete()
        print("LOADED")
    """)
    r = subprocess.run([sys.executable, "-c", prog, p],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "LOADED" not in r.stdout, (
        "a surface the validator calls clean now loads. If that is a real "
        "spatialnde2 improvement, check_projection_surface() can lose a half; "
        "do not simply delete this test.\n" + r.stdout + r.stderr)

    # ...and the repair makes the same surface load, in a fresh process.
    good = x3d_winding.loadable_x3d(p, quiet=True)
    r = subprocess.run([sys.executable, "-c", prog.replace(
        "assert not [q for q in snde.x3d_validate_geometry(f, 0, tags) if q.fatal]",
        "pass"), good], capture_output=True, text=True, timeout=300)
    assert "LOADED" in r.stdout, (
        "the repaired surface does not load:\n" + r.stdout + r.stderr)


def test_the_gate_is_called_where_the_surface_is_written():
    """Worth nothing if plan_mesh_path() does not call it, and that cannot be
    checked without a cell."""
    src = open(os.path.join(HERE, "mesh_scan.dpi")).read()
    i = src.index("mesh_uv_x3d(mesh, x3d, method=uv_method)")
    after = src[i:i + 600]
    assert "check_projection_surface(x3d)" in after, \
        "the surface is written and not checked where it is written"
