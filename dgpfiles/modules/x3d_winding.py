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
# File:  x3d_winding.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Checks and repairs triangle winding consistency in an X3D IndexedFaceSet, so
#   a surface can be loaded for projection without aborting the process.
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
"""Make an X3D surface loadable, or say exactly why it is not.

WHY THIS EXISTS. spatialnde2's x3d_load_geometry() sorts the edges around every
vertex into rotational order, and it walks that order through the per-edge
"next edge around triangle a / triangle b" links. Those links come from the
triangle winding. If one interior edge is traversed the SAME way round by both of
the triangles sharing it, the walk cannot close and the loader fails its own
assertion:

    x3d.hpp: Assertion `edgecheck < vertexnum_edges.second.size()' failed.
    Aborted

That is an assert, not an exception: SIGABRT, the whole process, with no
traceback and nothing to catch. A live scan loses its recording database and the
arm stops with the probe on the part. ONE bad edge in ten thousand is enough.

The condition is purely combinatorial -- it does not depend on vertex positions
or on whether the fan around a vertex is geometrically tidy. A consistently wound
mesh loads; one bad edge does not. So it can be decided here, cheaply, before the
loader is called, and usually repaired.

WHAT "REPAIRED" MEANS. An orientable surface has exactly two consistent windings
per connected component. Propagating a choice across shared edges finds one of
them; which of the two is then picked by majority vote against the winding that
came in, so the repair moves as few faces as possible and keeps the surface
facing the way its author intended. Geometry is untouched: the same triangles
between the same vertices, some of them written the other way round.

A surface that is NOT orientable cannot be repaired this way and is reported
instead. So is one with an edge shared by more than two triangles, which has no
well-defined "the other side" to propagate through.
"""

import os
import re

import numpy as np

#: Attributes of IndexedFaceSet that are per-triangle index lists and therefore
#: have to be reversed in step with coordIndex. A face written the other way
#: round must take its texture coordinates with it or the image shears.
INDEX_ATTRS = ("coordIndex", "texCoordIndex", "normalIndex", "colorIndex")


def _polygons(text):
    """The -1 terminated index runs of an X3D index attribute."""
    out = []
    cur = []
    for tok in text.split():
        v = int(tok)
        if v == -1:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(v)
    if cur:
        out.append(cur)
    return out


def read_indexed_face_set(path):
    """The index arrays of the first IndexedFaceSet in an X3D file.

    Returns (faces, extra) where faces is a list of vertex-index lists and extra
    maps each other per-triangle index attribute to its own list of lists.
    """
    with open(path, "r") as fh:
        text = fh.read()
    m = re.search(r"<IndexedFaceSet\b([^>]*)>", text)
    if m is None:
        raise ValueError("%s has no IndexedFaceSet" % (path,))
    attrs = m.group(1)
    got = {}
    for name in INDEX_ATTRS:
        a = re.search(r'\b%s="([^"]*)"' % (name,), attrs, re.S)
        if a is not None:
            got[name] = _polygons(a.group(1))
    if "coordIndex" not in got:
        raise ValueError("%s has an IndexedFaceSet with no coordIndex" % (path,))
    faces = got.pop("coordIndex")
    return faces, got


def winding_report(faces):
    """How, and how badly, a triangle soup's winding disagrees with itself.

        faces                 triangles examined
        skipped_non_triangles polygons that were not triangles
        inconsistent_edges    interior edges both of whose faces traverse them
                              the same way round -- each one of these is enough
                              to abort the loader
        overshared_edges      edges shared by more than two faces
        boundary_edges        edges used once, which are fine and expected
        loadable              whether x3d_load_geometry can be called safely
    """
    tris = [f for f in faces if len(f) == 3]
    directed = {}
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    undirected = {}
    for (a, b), n in directed.items():
        k = (a, b) if a < b else (b, a)
        undirected[k] = undirected.get(k, 0) + n
    inconsistent = 0
    overshared = 0
    boundary = 0
    for k, n in undirected.items():
        if n == 1:
            boundary += 1
        elif n > 2:
            overshared += 1
        elif directed.get(k, 0) == 2 or directed.get((k[1], k[0]), 0) == 2:
            # Both users went the same way round this edge.
            inconsistent += 1
    return {"faces": len(tris),
            "skipped_non_triangles": len(faces) - len(tris),
            "inconsistent_edges": inconsistent,
            "overshared_edges": overshared,
            "boundary_edges": boundary,
            "loadable": inconsistent == 0 and overshared == 0}


def orient_consistently(faces):
    """One consistent winding per connected component, as close to the input as
    possible.

    Returns (flip, report). `flip[i]` is True where face i has to be written the
    other way round. Faces are only ever reversed, never moved or removed, so a
    caller can apply the same flags to texture and normal indices.

    Components are walked through shared edges. An edge used by more than two
    faces is not walked through -- there is no single "other side" -- so the
    faces beyond it are reached another way or land in their own component,
    which is the honest outcome rather than an arbitrary choice.
    """
    tris = [f for f in faces if len(f) == 3]
    n = len(tris)
    flip = np.zeros(n, dtype=bool)

    # edge -> the faces using it, with which way round they went
    users = {}
    for i, (a, b, c) in enumerate(tris):
        for (u, v) in ((a, b), (b, c), (c, a)):
            k = (u, v) if u < v else (v, u)
            users.setdefault(k, []).append((i, 1 if u < v else -1))

    comp = -np.ones(n, dtype=np.int64)
    ncomp = 0
    conflicts = 0
    for seed in range(n):
        if comp[seed] >= 0:
            continue
        comp[seed] = ncomp
        members = [seed]
        stack = [seed]
        while stack:
            i = stack.pop()
            a, b, c = tris[i]
            for (u, v) in ((a, b), (b, c), (c, a)):
                k = (u, v) if u < v else (v, u)
                use = users[k]
                if len(use) != 2:
                    continue          # boundary, or shared by more than two
                for (j, dj) in use:
                    if j == i:
                        continue
                    di = 1 if u < v else -1
                    # Two faces are consistent across an edge when they traverse
                    # it in OPPOSITE directions once their own flips are applied.
                    want = bool(flip[i]) != (di == dj)
                    if comp[j] < 0:
                        comp[j] = ncomp
                        flip[j] = want
                        members.append(j)
                        stack.append(j)
                    elif bool(flip[j]) != want:
                        # Reachable two ways that disagree: this component is not
                        # orientable, so no assignment of flips can satisfy it.
                        conflicts += 1
        # Of the two consistent windings for this component, keep the one nearer
        # the input.
        idx = np.array(members, dtype=np.int64)
        if flip[idx].sum()*2 > len(idx):
            flip[idx] = ~flip[idx]
        ncomp += 1

    return flip, {"faces": n, "components": int(ncomp),
                  "faces_reversed": int(flip.sum()),
                  "non_orientable_conflicts": int(conflicts)}


def rewind_file(path, out_path=None):
    """Write a consistently wound copy of an X3D surface.

    Returns (out_path, report). The report carries `before` and `after` winding
    reports plus what orient_consistently() did, so a caller can say what it
    changed and refuse if it did not work.

    Only the index attributes are rewritten; coordinates, texture coordinates and
    everything else in the file are copied through byte for byte.
    """
    faces, extra = read_indexed_face_set(path)
    before = winding_report(faces)
    flip, rep = orient_consistently(faces)

    tri_pos = [i for i, f in enumerate(faces) if len(f) == 3]
    new = {}
    for name, polys in [("coordIndex", faces)] + sorted(extra.items()):
        if len(polys) != len(faces):
            # Not one run per face: leave it alone rather than corrupt it.
            continue
        out = [list(p) for p in polys]
        for k, i in enumerate(tri_pos):
            if flip[k]:
                out[i] = [out[i][0], out[i][2], out[i][1]]
        new[name] = out

    if out_path is None:
        base, ext = os.path.splitext(os.path.abspath(path))
        out_path = base + "_rewound" + ext

    with open(path, "r") as fh:
        text = fh.read()
    for name, polys in new.items():
        body = " ".join(" ".join(str(v) for v in p) + " -1" for p in polys)
        text = re.sub(r'(\b%s=")[^"]*(")' % (name,),
                      lambda m: m.group(1) + body + m.group(2), text, count=1)
    with open(out_path, "w") as fh:
        fh.write(text)

    after = winding_report(read_indexed_face_set(out_path)[0])
    rep.update({"before": before, "after": after,
                "path": path, "out_path": out_path})
    return out_path, rep


def loadable_x3d(path, quiet=False):
    """The surface to hand to x3d_load_geometry, repairing it if it needs it.

    Returns the path to load. Raises RuntimeError when the surface cannot be made
    loadable, because the alternative is that the loader aborts the process.
    """
    faces, _ = read_indexed_face_set(path)
    rep = winding_report(faces)
    if rep["loadable"]:
        return path

    if rep["overshared_edges"]:
        raise RuntimeError(
            "%s cannot be loaded for projection: %d edge(s) are shared by more "
            "than two triangles.\n"
            "  There is no single surface there to project onto, and "
            "x3d_load_geometry aborts the process rather than returning an "
            "error, so this is refused here instead.\n"
            "  It comes from the reconstruction, not from this file: check the "
            "mesh cleaning step, which drops faces on non-manifold edges."
            % (os.path.basename(path), rep["overshared_edges"]))

    out_path, fixed = rewind_file(path)
    if not fixed["after"]["loadable"]:
        raise RuntimeError(
            "%s cannot be loaded for projection: %d interior edge(s) are wound "
            "inconsistently and the surface is not orientable, so no consistent "
            "winding exists (%d conflict(s) while propagating one).\n"
            "  x3d_load_geometry aborts the process on this rather than "
            "returning an error, so it is refused here instead."
            % (os.path.basename(path), rep["inconsistent_edges"],
               fixed["non_orientable_conflicts"]))
    if not quiet:
        print("  projection surface rewound: %d of %d face(s) reversed to clear "
              "%d inconsistently wound edge(s) -- %s"
              % (fixed["faces_reversed"], fixed["faces"],
                 rep["inconsistent_edges"], os.path.basename(out_path)))
        print("    the geometry is unchanged; a single inconsistent edge makes "
              "the loader abort the process, so it cannot be left alone")
    return out_path
