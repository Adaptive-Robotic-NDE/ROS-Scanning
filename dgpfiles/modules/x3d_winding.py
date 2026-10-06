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


def read_point_array(path, tag, width):
    """The `point` attribute of the first `tag` node, as an (N, width) array.

    -> None when the node is absent, which is legitimate: an X3D with no
    TextureCoordinate simply has no parameterization to check.
    """
    with open(path, "r") as fh:
        text = fh.read()
    m = re.search(r"<%s\b((?:[^>\"]|\"[^\"]*\")*)>" % (tag,), text, re.S)
    if m is None:
        return None
    a = re.search(r'\bpoint="([^"]*)"', m.group(1), re.S)
    if a is None:
        return None
    vals = np.array([float(t) for t in a.group(1).replace(",", " ").split()])
    if vals.size % width:
        raise ValueError("%s: <%s point=...> has %d values, not a multiple of %d"
                         % (os.path.basename(path), tag, vals.size, width))
    return vals.reshape(-1, width)


def uv_handedness_report(path):
    """Does the UV parameterization preserve orientation, per triangle?

        triangles   triangles that carry a UV triple
        agreeing    triangles whose UV winding matches their face winding
        mirrored    triangles whose UV winding is reversed -- THESE TAKE NO DATA
        degenerate  triangles whose UV triple has zero area
        mirrored_fraction, verdict ("ok" / "partly_mirrored" / "mirrored" /
        "no_parameterization")

    WHY THIS MATTERS, AND WHY NOTHING ELSE CATCHES IT.
    spatialnde2 builds a per-triangle in-plane 2D basis that is forced
    right-handed about the triangle's trinormal, and the trinormal comes from
    the winding (`inplanemat_calculation.cpp`). The map from that frame to UV is
    `inplane2uvcoords`, and `project_onto_parameterization.cpp:660-672` takes

        i2uv_det = ...                 # negative when the UV frame is mirrored
        lambda_magnitude = sqrt(i2uv_det)

    so a mirrored parameterization makes that a **NaN**. The assert on the next
    line was written for exactly this case -- its comment reads "if this fails
    then inplane2uvcoords could be doing some kind of weird mirroring maybe" --
    but it tests for zero, and NaN is not zero, so it never fires. The NaN
    becomes the projection spot radius, every accumulated weight comes out 0.0,
    and the image stays empty with no error raised anywhere.

    Because the in-plane frame is right-handed about the trinormal, the whole
    test reduces to the SIGN OF THE UV SIGNED AREA taken in the face's own
    winding order: det > 0 exactly when that area is positive. No 3D normal and
    no spatialnde2 are needed, which is why this can run on a bare file minutes
    before the arm moves.

    Measured on 2026-10-02: two consecutive scan runs projected 1876 and 1673
    traces into an empty image with every triangle mirrored (7058 of 7058 and
    9341 of 9341); the third run of the same morning, 5308 of 5308 agreeing,
    projected correctly. The difference was the sign numpy happened to return
    for the second singular vector in the planar unwrap.
    """
    faces, extra = read_indexed_face_set(path)
    uv = read_point_array(path, "TextureCoordinate", 2)
    out = {"triangles": 0, "agreeing": 0, "mirrored": 0, "degenerate": 0,
           "mirrored_fraction": 0.0, "verdict": "no_parameterization"}
    if uv is None or len(uv) == 0:
        return out
    # texCoordIndex is optional: without it, UV is indexed by coordIndex.
    uv_faces = extra.get("texCoordIndex") or faces
    if len(uv_faces) != len(faces):
        raise ValueError(
            "%s: texCoordIndex has %d face(s) and coordIndex has %d; they index "
            "the same faces and must agree"
            % (os.path.basename(path), len(uv_faces), len(faces)))
    for face, uv_face in zip(faces, uv_faces):
        if len(face) != 3 or len(uv_face) != 3:
            continue
        out["triangles"] += 1
        a, b, c = uv[uv_face[0]], uv[uv_face[1]], uv[uv_face[2]]
        area2 = ((b[0] - a[0]) * (c[1] - a[1])
                 - (c[0] - a[0]) * (b[1] - a[1]))
        if area2 > 0.0:
            out["agreeing"] += 1
        elif area2 < 0.0:
            out["mirrored"] += 1
        else:
            out["degenerate"] += 1
    if out["triangles"]:
        out["mirrored_fraction"] = out["mirrored"] / float(out["triangles"])
    if out["mirrored"] == 0:
        out["verdict"] = "ok"
    elif out["agreeing"] == 0:
        out["verdict"] = "mirrored"
    else:
        out["verdict"] = "partly_mirrored"
    return out


def orient_uv_basis(e0, e1, outward):
    """Force a 2D UV basis right-handed about the surface's outward normal.

    -> (e0, e1, flipped).

    `numpy.linalg.svd` pins down the PLANE of a planar unwrap but not the
    handedness of the basis in it: it is free to return -e1, and does, about
    half the time. Nothing downstream notices, because the resulting UV map is
    a perfectly good mirror image -- right up to the point where
    project_onto_parameterization takes the square root of a negative
    determinant. See uv_handedness_report() for that chain.

    The condition is exact rather than heuristic. For a triangle with edge
    vectors `u` and `v` in winding order, the UV signed area is
    `0.5 * (u x v) . (e0 x e1)` = `Area * (n . (e0 x e1))`, so it is positive
    for every triangle precisely when `(e0 x e1) . n > 0`.
    """
    e0 = np.asarray(e0, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    outward = np.asarray(outward, dtype=float)
    norm = float(np.linalg.norm(outward))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(
            "the surface has no net outward direction (|mean normal| = %r), so "
            "a planar parameterization of it cannot be oriented -- and would "
            "not be bijective either. This is a closed or folded surface being "
            "unwrapped as a plate." % (norm,))
    handedness = float(np.dot(np.cross(e0, e1), outward / norm))
    if handedness < 0.0:
        return e0, -e1, True
    return e0, e1, False


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


#: How much of a surface's area loadable_x3d() will remove on its own to make
#: it orientable. Overlapping slivers are a fraction of a percent (0.015% on
#: the mesh this was written for); anything needing more than this is not a
#: sliver problem and should be refused and looked at, not quietly cut away.
MAX_DROP_AREA_FRACTION = 0.01


def same_way_edges(faces):
    """Edges that two faces both traverse in the same direction. -> {edge: [i, j]}

    One of these is enough to make x3d_load_geometry abort, and each is either
    a face wound the wrong way round -- which rewinding fixes -- or two faces
    that OVERLAP, which rewinding cannot fix. overlapping_faces() tells them
    apart.
    """
    out = {}
    for i, f in enumerate(faces):
        if len(f) != 3:
            continue
        a, b, c = f
        for e in ((a, b), (b, c), (c, a)):
            out.setdefault(e, []).append(i)
    return {e: u for e, u in out.items() if len(u) >= 2}


def overlapping_faces(faces, points):
    """Faces to DROP so that a consistent winding can exist. -> (drop, report)

    WHAT THIS IS AND IS NOT. Winding consistency is COMBINATORIAL, not
    geometric: an isolated pair of overlapping triangles can always be made
    consistent by flipping one of them, and orient_consistently() will do it
    (measured -- a single fold on a boundary edge rewinds clean). So an overlap
    is not by itself unrepairable, and this function is not a claim that it is.

    What overlaps do is FRUSTRATE the orientation globally. Each one demands a
    flip, those demands conflict around cycles, and a mesh carrying enough of
    them has no consistent assignment at all: on the surface this was written
    for, rewinding took 26 inconsistent edges to 104 with 208 conflicts while
    propagating, and dropping 16 overlapping faces took the same mesh to 0.
    That is the empirical ground for this, and the mechanism beyond "the flips
    conflict" has not been characterised further.

    So this is the SECOND tool, reached only when rewinding has already failed,
    and the geometric overlap test is how it chooses which faces to remove
    rather than a proof that they are the only possible choice. The test needs
    no normals: project each face's third vertex onto the plane perpendicular
    to the shared edge; if the two land on the same side, the faces overlap.

    The SMALLER face of each pair goes. These are slivers thrown off by
    surface reconstruction and smoothing -- on the mesh this was written for,
    37 faces carrying 0.16% of the area, a median of 5x smaller than typical --
    so the larger one is the surface and the smaller one is the artefact.

    Returns the set of face indices to drop and a report carrying the area
    dropped, so a caller can refuse a repair that would eat real surface
    instead of slivers.
    """
    pts = np.asarray(points, dtype=np.float64)
    tri_idx = [i for i, f in enumerate(faces) if len(f) == 3]
    area = {}
    for i in tri_idx:
        a, b, c = (pts[v] for v in faces[i])
        area[i] = 0.5*float(np.linalg.norm(np.cross(b - a, c - a)))
    total_area = sum(area.values())

    drop = set()
    examined = 0
    overlaps = 0
    # Iterated, because dropping one face can resolve several edges at once and
    # can also expose an edge that was previously shared by three.
    for _ in range(8):
        live = [f if i not in drop else None for i, f in enumerate(faces)]
        bad = same_way_edges([f if f is not None else () for f in live])
        if not bad:
            break
        progress = False
        for (p0, p1), users in bad.items():
            users = [u for u in users if u not in drop]
            if len(users) < 2:
                continue
            examined += 1
            i, j = users[0], users[1]
            e = pts[p1] - pts[p0]
            n = np.linalg.norm(e)
            if n == 0.0:
                continue
            e = e/n
            sides = []
            for f in (i, j):
                other = [v for v in faces[f] if v != p0 and v != p1]
                if not other:
                    sides = []
                    break
                w = pts[other[0]] - pts[p0]
                sides.append(w - float(w @ e)*e)
            if len(sides) != 2:
                continue
            if float(sides[0] @ sides[1]) <= 0.0:
                # Opposite sides: a genuine winding error, and rewinding owns it.
                continue
            overlaps += 1
            drop.add(i if area[i] <= area[j] else j)
            progress = True
        if not progress:
            break

    dropped_area = sum(area.get(i, 0.0) for i in drop)
    return drop, {"dropped": len(drop),
                  "overlapping_pairs": overlaps,
                  "edges_examined": examined,
                  "dropped_area_m2": dropped_area,
                  "total_area_m2": total_area,
                  "dropped_area_fraction": (dropped_area/total_area
                                            if total_area > 0 else 0.0)}


def drop_faces_file(path, drop, out_path=None):
    """Write a copy of an X3D surface with `drop` faces removed.

    Every per-face index attribute is cut in step -- coordIndex, texCoordIndex,
    normalIndex -- so the parameterization stays attached to the geometry. An
    attribute that does not have one run per face is left alone rather than
    corrupted, the same rule rewind_file() follows.
    """
    faces, extra = read_indexed_face_set(path)
    keep = [i for i in range(len(faces)) if i not in drop]
    if out_path is None:
        base, ext = os.path.splitext(os.path.abspath(path))
        out_path = base + "_dropped" + ext

    with open(path, "r") as fh:
        text = fh.read()
    for name, polys in [("coordIndex", faces)] + sorted(extra.items()):
        if len(polys) != len(faces):
            continue
        body = " ".join(" ".join(str(v) for v in polys[i]) + " -1" for i in keep)
        text = re.sub(r'(\b%s=")[^"]*(")' % (name,),
                      lambda m: m.group(1) + body + m.group(2), text, count=1)
    with open(out_path, "w") as fh:
        fh.write(text)
    return out_path


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


def flip_uv_file(path, out_path=None):
    """Write a copy whose UV parameterization is mirrored, and so un-mirrored.

    Returns (out_path, report) with `before` and `after` handedness reports.

    Only the TextureCoordinate `point` attribute is rewritten; coordinates,
    indices and everything else are copied through byte for byte, so this
    changes no geometry and no topology -- it is the UV analogue of
    rewind_file().

    The mirror is of the FIRST uv coordinate about the midpoint of its own
    range, so the uv bounding box comes out identical. That matters: the
    projector derives its pixel pitch from the uv span and nothing else, so a
    repair that rescaled the box would silently change the image resolution.

    This is necessarily a GLOBAL mirror: there is no way to reverse one
    triangle's orientation without tearing it away from its neighbours. So on a
    surface that is only partly mirrored it trades one set of triangles for the
    other, and whether that is worth doing is the caller's decision -- this
    function reports what it did and judges nothing.
    """
    before = uv_handedness_report(path)
    uv = read_point_array(path, "TextureCoordinate", 2)
    if uv is None:
        raise ValueError("%s has no TextureCoordinate to mirror"
                         % (os.path.basename(path),))
    mid = 0.5 * (float(uv[:, 0].min()) + float(uv[:, 0].max()))
    uv = uv.copy()
    uv[:, 0] = 2.0 * mid - uv[:, 0]

    if out_path is None:
        base, ext = os.path.splitext(os.path.abspath(path))
        out_path = base + "_uvflip" + ext

    with open(path, "r") as fh:
        text = fh.read()
    body = " ".join("%.9g %.9g" % (row[0], row[1]) for row in uv)
    m = re.search(r"<TextureCoordinate\b((?:[^>\"]|\"[^\"]*\")*)>", text, re.S)
    head, tail = text[:m.start()], text[m.end():]
    node = re.sub(r'(\bpoint=")[^"]*(")',
                  lambda mm: mm.group(1) + body + mm.group(2), m.group(0),
                  count=1)
    with open(out_path, "w") as fh:
        fh.write(head + node + tail)

    after = uv_handedness_report(out_path)
    return out_path, {"before": before, "after": after, "path": path,
                      "out_path": out_path, "mirror_about": mid}


#: The index attribute that addresses each point array, with the X3D node that
#: holds that array and its width. x3d_load_geometry() sorts the triangles
#: around every vertex of BOTH into a ring, and reports them separately
#: ([vertex_fan_discontiguous] and [uv_vertex_fan_discontiguous]), so both have
#: to be checked and both can need the same repair.
FAN_ARRAYS = (("coordIndex", "Coordinate", 3),
              ("texCoordIndex", "TextureCoordinate", 2))


def vertex_fans(faces):
    """Which triangles around each vertex form one fan, as {vertex: [fan, ...]}.

    A fan is a set of triangle indices. The test is the one the loader makes:
    the triangles touching a vertex have to form a single ring, which they do
    exactly when the OPPOSITE edges -- for a triangle (v, b, c) the edge b-c --
    form one connected chain. Two chains mean two sheets of surface meeting at
    that single point and nothing else, and no ring can be sorted through it.

    Deliberately independent of winding: reversing a triangle swaps b and c and
    leaves the chain alone, so this answers the same before and after
    rewind_file() -- which is why rewinding a mesh never clears it.
    """
    touching = {}
    for fi, f in enumerate(faces):
        if len(f) != 3 or len(set(f)) != 3:
            continue                        # degenerate: it bounds no fan
        for k in range(3):
            touching.setdefault(f[k], []).append((fi, f[(k+1) % 3], f[(k+2) % 3]))
    out = {}
    for v, tris in touching.items():
        adj = {}
        for _fi, b, c in tris:
            adj.setdefault(b, set()).add(c)
            adj.setdefault(c, set()).add(b)
        seen = set()
        groups = []
        for start in list(adj):
            if start in seen:
                continue
            comp = set()
            stack = [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(adj[u] - comp)
            seen |= comp
            groups.append(comp)
        if len(groups) < 2:
            continue
        fans = [set() for _ in groups]
        for fi, b, c in tris:
            for gi, comp in enumerate(groups):
                if b in comp:
                    fans[gi].add(fi)
                    break
        out[v] = fans
    return out


def fan_report(path):
    """What x3d_load_geometry()'s two fan invariants say about this file.

    -> {"coordIndex": {...}, "texCoordIndex": {...}} with `vertices` (how many
    are pinched), `extra_fans` (how many duplicate points a repair would add)
    and `worst` (the first offending vertex and its fan count), plus `loadable`
    over both.
    """
    faces, extra = read_indexed_face_set(path)
    arrays = {"coordIndex": faces}
    arrays.update({k: v for k, v in extra.items()})
    out = {}
    for name, _node, _width in FAN_ARRAYS:
        polys = arrays.get(name)
        if polys is None or len(polys) != len(faces):
            out[name] = {"vertices": 0, "extra_fans": 0, "worst": None,
                         "present": polys is not None}
            continue
        bad = vertex_fans(polys)
        worst = None
        if bad:
            v = min(bad)
            worst = (int(v), len(bad[v]), sum(len(f) for f in bad[v]))
        out[name] = {"vertices": len(bad),
                     "extra_fans": sum(len(f) - 1 for f in bad.values()),
                     "worst": worst, "present": True}
    out["loadable"] = not any(out[n]["vertices"]
                              for n, _n2, _w in FAN_ARRAYS)
    return out


#: How much of a surface's area a pinch repair will remove on its own.
#:
#: A pinch that is a meshing artefact is a sliver hanging off a corner and costs
#: a few thousandths of a percent. A pinch joining two real sheets of surface
#: costs half the part, and there the right answer is to say so and let somebody
#: look, not to delete one of the sheets. 1% sits three orders of magnitude above
#: the first and two below the second, so it separates them with room to spare.
#:
#: THERE WAS A FACE-COUNT BOUND HERE TOO, and it was wrong. It was set to 4 from
#: the first case this ever saw -- one triangle of 8245 -- and read as "an
#: artefact is one or two triangles". It is not: dropping 121 overlapping faces
#: to make a surface orientable left 26 pinched vertices behind it on
#: 2026-10-06, costing 0.236% of the area, and a count bound refused the repair
#: while the area bound would have allowed it with a factor of four to spare.
#: The count was a proxy for "is this structural", and area answers that
#: question directly on any mesh density.
MAX_PINCH_AREA_FRACTION = 0.01


def _triangle_areas(faces, points):
    tri = np.array([f for f in faces if len(f) == 3], dtype=np.intp)
    if not len(tri):
        return np.zeros(0), []
    p = np.asarray(points, dtype=float)[:, :3]
    a = p[tri[:, 1]] - p[tri[:, 0]]
    b = p[tri[:, 2]] - p[tri[:, 0]]
    return 0.5*np.linalg.norm(np.cross(a, b), axis=1), [
        i for i, f in enumerate(faces) if len(f) == 3]


def pinched_fan_faces(path):
    """Which faces to remove so that every vertex fan becomes contiguous.

    -> (drop, report). At each pinched vertex the fan carrying the most AREA is
    kept and the others are dropped, in the geometry index and the
    parameterization index alike -- the loader checks both and a mesh can be
    pinched in either.

    DROPPING RATHER THAN SPLITTING, and this is not the first choice. Giving
    each fan its own copy of the point is the repair that loses nothing, and it
    does not survive the loader: x3d_load_geometry() is called with
    `reindex_vertices`, which keys a hash map on the coordinate itself
    (x3d.hpp, vertexnum_bycoord.find), so two points at the same position are
    welded back into one and the pinch comes straight back. Moving the copy
    instead trades this failure for the other one this module exists to catch --
    a sliver with a near-zero determinant projects NaN weights and leaves a hole
    with no error. So the fan goes.
    """
    faces, extra = read_indexed_face_set(path)
    points = read_point_array(path, "Coordinate", 3)
    arrays = {"coordIndex": faces}
    arrays.update(extra)
    areas, tri_pos = _triangle_areas(faces, points) if points is not None \
        else (None, [])
    area_of = {fi: float(areas[k]) for k, fi in enumerate(tri_pos)} \
        if areas is not None else {}
    total = float(sum(area_of.values())) if area_of else 0.0

    drop = set()
    per_array = {}
    for name, _node, _width in FAN_ARRAYS:
        polys = arrays.get(name)
        if polys is None or len(polys) != len(faces):
            per_array[name] = 0
            continue
        n = 0
        for _v, fans in sorted(vertex_fans(polys).items()):
            keep = max(range(len(fans)),
                       key=lambda i: sum(area_of.get(fi, 1.0) for fi in fans[i]))
            for i, fan in enumerate(fans):
                if i != keep:
                    drop |= set(fan)
                    n += 1
        per_array[name] = n
    lost = float(sum(area_of.get(fi, 0.0) for fi in drop))
    return drop, {"dropped": len(drop),
                  "fans_dropped": per_array,
                  "dropped_area": lost,
                  "dropped_area_fraction": (lost/total) if total > 0 else 0.0}


def unpinch_file(path, out_path=None, max_rounds=4):
    """Write a copy of an X3D surface with no pinched vertex, or say why not.

    Returns (out_path, report). Removing a fan can expose another pinch in the
    triangles that bounded it, so this repeats until the surface is clean or
    nothing more is removed; `rounds` says how many it took and the area
    fractions accumulate.

    Raises RuntimeError when the repair would cost more than
    MAX_PINCH_AREA_FRACTION of the surface, which means the pinch is structural
    rather than a sliver.
    """
    before = fan_report(path)
    cur = path
    total_drop = 0
    total_area = 0.0
    rounds = 0
    tmp_paths = []
    while rounds < max_rounds:
        drop, rep = pinched_fan_faces(cur)
        if not drop:
            break
        total_area += rep["dropped_area_fraction"]
        total_drop += len(drop)
        if total_area > MAX_PINCH_AREA_FRACTION:
            raise RuntimeError(
                "%s cannot be loaded for projection: clearing its pinched "
                "vertex(es) would remove %d face(s) carrying %.3f%% of the "
                "area, over the %.1f%% this will remove without being asked.\n"
                "  A vertex whose triangles fall into two fans is two sheets of "
                "surface meeting at a single point. A sliver there is an "
                "artefact and goes; this much area is a join between real "
                "pieces of the part, and deleting one of them is not a "
                "repair.\n"
                "  x3d_load_geometry() sorts the triangles around every vertex "
                "into a ring and REFUSES a file it cannot, so this is refused "
                "here -- before the probe is on the part -- rather than after "
                "a null and a first pass. Re-run the parameterization, or check "
                "the reconstruction for two surfaces touching at a corner."
                % (os.path.basename(path), total_drop, 100.0*total_area,
                   100.0*MAX_PINCH_AREA_FRACTION))
        if out_path is None:
            base, ext = os.path.splitext(os.path.abspath(path))
            target = base + "_unpinched" + ext
        else:
            target = out_path
        cur = drop_faces_file(cur, drop, out_path=target)
        rounds += 1
    after = fan_report(cur)
    if not after["loadable"]:
        raise RuntimeError(
            "%s cannot be loaded for projection: %d round(s) of removing "
            "pinched fans left %d geometry and %d parameterization vertex(es) "
            "still pinched.\n"
            "  x3d_load_geometry() refuses these, so this is refused here "
            "instead -- before anything moves."
            % (os.path.basename(path), rounds,
               after["coordIndex"]["vertices"],
               after["texCoordIndex"]["vertices"]))
    return cur, {"before": before, "after": after, "rounds": rounds,
                 "dropped": total_drop, "dropped_area_fraction": total_area,
                 "path": path, "out_path": cur}


def loadable_x3d(path, quiet=False):
    """The surface to hand to x3d_load_geometry, repairing it if it needs it.

    Returns the path to load. Raises RuntimeError when the surface cannot be made
    loadable, because the alternative is that the loader aborts the process.

    THREE REPAIRS, AND THE ORDER BETWEEN THEM IS LOAD-BEARING:

      1. winding -- make every shared edge consistent, dropping overlapping
         faces if that is the only way (_wound_consistently)
      2. fans -- make the triangles around every vertex form one ring
         (_fans_split)
      3. uv handedness -- make the parameterization able to receive data
         (_uv_oriented)

    FANS AFTER WINDING, because the winding repair can REMOVE FACES and removing
    a face from the middle of a fan splits it in two. That is not hypothetical:
    on 2026-10-06 a surface with 219 inconsistent edges and zero pinched
    vertices had 121 overlapping faces dropped to make it orientable, which left
    16 pinched geometry vertices and 10 pinched uv vertices, and the loader
    refused it -- with the probe on the part and the instrument already nulled.
    The fan repair had run first, found nothing to do, and never looked again.

    The reverse order is safe and is why this one works: dropping faces from a
    consistently wound mesh leaves it consistently wound, and unpinching only
    drops faces.
    """
    return _uv_oriented(_fans_split(_wound_consistently(path, quiet=quiet),
                                    quiet=quiet), quiet=quiet)


def _wound_consistently(path, quiet=False):
    """The same surface with every shared edge wound consistently."""
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
        # REWINDING WAS NOT ENOUGH, so try the other tool before giving up.
        # It comes back WORSE than it went in -- 26 inconsistent edges became
        # 104, with 208 conflicts while propagating -- which says the flips it
        # needs conflict with each other and no consistent assignment exists.
        # Removing the overlapping faces removes that frustration. See
        # overlapping_faces() for what is and is not claimed about why.
        #
        # Ordered this way round deliberately: rewinding removes nothing and is
        # tried first, so a surface that is merely wound backwards is repaired
        # without losing a single triangle. Dropping is the fallback.
        points = read_point_array(path, "Coordinate", 3)
        dropped = None
        # Captured BEFORE the second rewind overwrites `fixed`: what the
        # message is about is why the first attempt failed.
        first_conflicts = fixed["non_orientable_conflicts"]
        if points is not None:
            drop, dropped = overlapping_faces(faces, points)
            if drop and dropped["dropped_area_fraction"] <= MAX_DROP_AREA_FRACTION:
                pruned = drop_faces_file(path, drop)
                out_path, fixed = rewind_file(pruned)
                if fixed["after"]["loadable"]:
                    if not quiet:
                        print("  MESHUV rewinding could not orient this surface "
                              "(%d inconsistent edge(s), %d conflict(s)), so %d "
                              "overlapping face(s) carrying %.4f%% of the area "
                              "were dropped and it rewound clean"
                              % (rep["inconsistent_edges"], first_conflicts,
                                 dropped["dropped"],
                                 100.0*dropped["dropped_area_fraction"]))
                    return out_path

    if not fixed["after"]["loadable"]:
        extra = ""
        if dropped is not None and dropped["dropped"]:
            extra = (" %d overlapping face(s) carrying %.4f%% of the area were "
                     "identified, which is over the %.1f%% this will remove "
                     "without being asked -- a surface that needs more than "
                     "that cut away is not a sliver problem."
                     % (dropped["dropped"],
                        100.0*dropped["dropped_area_fraction"],
                        100.0*MAX_DROP_AREA_FRACTION))
        raise RuntimeError(
            "%s cannot be loaded for projection: %d interior edge(s) are wound "
            "inconsistently and the surface is not orientable, so no consistent "
            "winding exists (%d conflict(s) while propagating one).%s\n"
            "  x3d_load_geometry aborts the process on this rather than "
            "returning an error, so it is refused here instead."
            % (os.path.basename(path), rep["inconsistent_edges"],
               locals().get("first_conflicts",
                            fixed["non_orientable_conflicts"]), extra))
    if not quiet:
        print("  projection surface rewound: %d of %d face(s) reversed to clear "
              "%d inconsistently wound edge(s) -- %s"
              % (fixed["faces_reversed"], fixed["faces"],
                 rep["inconsistent_edges"], os.path.basename(out_path)))
        print("    the geometry is unchanged; a single inconsistent edge makes "
              "the loader abort the process, so it cannot be left alone")
    return out_path


def _fans_split(path, quiet=False):
    """The same surface with every vertex fan contiguous, repairing if needed."""
    rep = fan_report(path)
    if rep["loadable"]:
        return path
    out_path, fixed = unpinch_file(path)
    if not quiet:
        for name, _node, _width in FAN_ARRAYS:
            b = rep[name]
            if not b["vertices"]:
                continue
            where = "geometry" if name == "coordIndex" else "parameterization"
            print("  projection surface unpinched: %d %s vertex(es) had "
                  "triangles in more than one fan (worst vertex %d, %d fans "
                  "from %d triangles)"
                  % (b["vertices"], where, b["worst"][0], b["worst"][1],
                     b["worst"][2]))
        print("    %d face(s) carrying %.4f%% of the area were removed in %d "
              "round(s) -- %s"
              % (fixed["dropped"], 100.0*fixed["dropped_area_fraction"],
                 fixed["rounds"], os.path.basename(out_path)))
        print("    a pinched vertex is two sheets of surface meeting at one "
              "point; the loader REFUSES one, and rewinding cannot clear it, "
              "so it cannot be left alone")
    return out_path


def _uv_oriented(path, quiet=False):
    """The same surface with a parameterization the projector can use.

    A mirrored parameterization is the failure that produces an EMPTY IMAGE AND
    NO ERROR -- see uv_handedness_report() for the NaN chain that does it. It
    is checked here, on the file that is about to be loaded, because at this
    point nothing has moved yet and the alternative is finding out after a
    whole scan's worth of traces has been thrown away.
    """
    uvrep = uv_handedness_report(path)
    if uvrep["verdict"] in ("ok", "no_parameterization"):
        return path

    # A global mirror reverses EVERY triangle, so it is worth doing exactly
    # when it leaves fewer triangles mirrored than it finds. On a uniformly
    # mirrored surface that is all of them. On a mixed one it still rescues the
    # majority, which is the common case after rewind_file() has corrected the
    # winding of a handful of faces -- that correction flips those faces'
    # handedness and is what makes an otherwise uniform surface mixed.
    if uvrep["mirrored"] > uvrep["agreeing"]:
        out_path, flipped = flip_uv_file(path)
        after = flipped["after"]
        if after["mirrored"] >= uvrep["mirrored"]:
            raise RuntimeError(
                "%s has %d of %d triangle(s) with a mirrored UV "
                "parameterization and mirroring it back left %d. Refusing it "
                "rather than projecting into an image that stays empty with no "
                "error raised."
                % (os.path.basename(path), uvrep["mirrored"],
                   uvrep["triangles"], after["mirrored"]))
        if not quiet:
            print("  projection surface UV mirrored back: %d of %d triangle(s) "
                  "were wound against their faces -- %s"
                  % (uvrep["mirrored"], uvrep["triangles"],
                     os.path.basename(out_path)))
            print("    left alone, every ray would have landed and deposited "
                  "ZERO WEIGHT, and the image would have stayed empty with no "
                  "error anywhere. The geometry is unchanged.")
            _warn_residual_mirroring(after, out_path)
        return out_path

    if not quiet:
        _warn_residual_mirroring(uvrep, path)
    return path


def _warn_residual_mirroring(uvrep, path):
    """Say which triangles still cannot take data, and why there are any."""
    if not uvrep["mirrored"]:
        return
    print("  ** %d of %d triangle(s) of %s are still mirrored and CANNOT "
          "RECEIVE DATA"
          % (uvrep["mirrored"], uvrep["triangles"], os.path.basename(path)))
    print("    the projector takes the square root of a negative determinant "
          "there and the resulting NaN zeroes every weight. It raises no "
          "error, so the only symptom is holes in the image.")
    print("    a mixture cannot be fixed by mirroring -- that only trades one "
          "set for the other. Re-run the parameterization.")
