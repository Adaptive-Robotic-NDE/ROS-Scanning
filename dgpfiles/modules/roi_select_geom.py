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
# File:  roi_select_geom.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Region-of-interest selection geometry: masks, bounds, sub-meshes and
#   PLY I/O.
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
"""Region-of-interest selection geometry: masks, bounds, sub-meshes, PLY I/O.

THE POINT OF THE EXERCISE. The YOLO detector's box can be several times the size
of the part -- an 83 x 165 mm plate has come back as a 135 x 287 mm box -- and
everything else inside it (the optical breadboard, the clamps and pins holding
the plate down, a stepped mount block) is then measured, meshed and rastered.
That yields a 198 k-triangle mesh only 30% of whose vertices are the plate, a
412 k-pose toolpath where about 45 000 poses over the plate alone would have
done, and tool poses 89.8 degrees off vertical where the planner runs over the
rim of the mount block.

NO Qt AND NO VTK IN THIS FILE, deliberately. Everything here is numpy over
plain arrays, so the selection logic can be tested on a headless box
(roi_select_test.py) while the window it drives cannot be. The UI
(roi_select_ui.py) projects vertices to screen coordinates with the live camera
and then calls exactly these functions, so what the test covers is what the
operator gets -- not a reimplementation of it.

THE SELECTION SPACE IS THE SCREEN, not the world. The operator draws on what
they can see, and the tools here all take (N, 2) display coordinates. That is
what makes a lasso around a plate mean what it looks like it means from any
viewpoint, and it is why front_surface_mask() exists: a shape drawn on screen
catches the bench BEHIND the part as well, and at a top-down view -- the one an
operator actually uses here -- those are the same pixels.

Python 3.10 compatible (the cell runs Ubuntu 22.04); numpy only.
"""

import math
import os
import sys

import numpy as np

#: How a new shape combines with what is already selected.
MODES = ("replace", "add", "subtract", "toggle")

#: PLY scalar type names -> numpy dtypes, little-endian.
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


# ==============================================================================
# PLY I/O -- MESH *OR* POINT CLOUD
# ==============================================================================
# read_ply_mesh() in noether_path.dpi raises "no vertices or no faces" on a
# point cloud, on purpose: everything downstream of it needs triangles. This
# stage is upstream of that. The Azure rough mesh is a mesh, the Keyence
# intermediates are point clouds, and an operator must be able to draw a region
# on either -- so faces are optional here and `tris is None` is a normal,
# supported state rather than an error.

def read_ply(filename):
    """(verts N x 3, normals N x 3 or None, tris M x 3 or None) from a PLY.

    ASCII or binary little-endian, as Open3D, trimesh and Blender write them.
    Colours and any other vertex property are read past and dropped. A file
    with no face element gives tris=None rather than raising -- that is a point
    cloud, and this stage accepts one.
    """
    with open(filename, "rb") as fh:
        data = fh.read()
    try:
        end = data.index(b"end_header")
    except ValueError:
        raise ValueError("%s is not a PLY file (no end_header)" % (filename,))
    end = data.index(b"\n", end) + 1

    fmt = None
    elements = []
    for line in data[:end].decode("ascii", "replace").splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            elements.append([w[1], int(w[2]), []])
        elif w[0] == "property" and elements:
            if w[1] == "list":
                elements[-1][2].append((w[4], "list", _PLY_TYPES[w[2]],
                                        _PLY_TYPES[w[3]]))
            else:
                elements[-1][2].append((w[2], _PLY_TYPES[w[1]]))
    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError("%s: PLY format %r is not supported" % (filename, fmt))

    body = data[end:]
    verts = norms = tris = None

    if fmt == "ascii":
        rows = body.decode("ascii", "replace").split("\n")
        at = 0
        for name, count, props in elements:
            lines = [l for l in rows[at:at + count]]
            at += count
            if name == "vertex":
                arr = np.array([l.split() for l in lines], dtype=np.float64)
                names = [p[0] for p in props]
                verts = arr[:, [names.index(c) for c in ("x", "y", "z")]]
                if all(c in names for c in ("nx", "ny", "nz")):
                    norms = arr[:, [names.index(c) for c in ("nx", "ny", "nz")]]
            elif name == "face" and count:
                faces = [l.split() for l in lines]
                tris = _fan(faces, filename)
    else:
        off = 0
        for name, count, props in elements:
            if any(p[1] == "list" for p in props):
                if name != "face" or len(props) != 1:
                    raise ValueError("%s: list property in element %r"
                                     % (filename, name))
                ct = np.dtype(props[0][2])
                it = np.dtype(props[0][3])
                dt = np.dtype([("n", ct), ("v", it, 3)])
                if count:
                    f = np.frombuffer(body, dtype=dt, count=count, offset=off)
                    if (f["n"] != 3).any():
                        raise ValueError("%s: only triangle meshes are supported"
                                         % (filename,))
                    tris = f["v"].astype(np.int64)
                off += count*dt.itemsize
                continue
            dt = np.dtype([(p[0], p[1]) for p in props])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=off)
            off += count*dt.itemsize
            if name == "vertex":
                verts = np.column_stack([arr[c].astype(np.float64)
                                         for c in ("x", "y", "z")])
                if all(c in (dt.names or ()) for c in ("nx", "ny", "nz")):
                    norms = np.column_stack([arr[c].astype(np.float64)
                                             for c in ("nx", "ny", "nz")])

    if verts is None or not len(verts):
        raise ValueError("%s has no vertices" % (filename,))
    return verts, norms, tris


def _fan(faces, filename):
    """Triangles from ASCII face rows. Quads are fanned; anything else refused."""
    out = []
    for row in faces:
        if not row:
            continue
        n = int(row[0])
        idx = [int(v) for v in row[1:1 + n]]
        if n == 3:
            out.append(idx)
        elif n == 4:
            out.append([idx[0], idx[1], idx[2]])
            out.append([idx[0], idx[2], idx[3]])
        else:
            raise ValueError("%s: face with %d vertices is not supported"
                             % (filename, n))
    return np.array(out, dtype=np.int64) if out else None


def write_ply(filename, verts, tris=None, normals=None):
    """Write a binary little-endian PLY that read_ply_mesh() can read back.

    A point cloud (tris None or empty) is written with NO face element rather
    than with an empty one. read_ply_mesh() then refuses it by name -- "no
    vertices or no faces" -- which is the honest answer; an `element face 0`
    would sail past that check and hand the planner a mesh with no triangles.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be N x 3, got %s" % (verts.shape,))
    has_n = normals is not None and len(normals) == len(verts)

    head = ["ply", "format binary_little_endian 1.0",
            "comment written by roi_select_geom.py",
            "element vertex %d" % (len(verts),),
            "property double x", "property double y", "property double z"]
    if has_n:
        head += ["property double nx", "property double ny", "property double nz"]
    cols = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    if has_n:
        cols += [("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8")]

    ntri = 0 if tris is None else len(tris)
    if ntri:
        head += ["element face %d" % (ntri,),
                 "property list uchar int vertex_indices"]
    head.append("end_header")

    vrec = np.zeros(len(verts), dtype=np.dtype(cols))
    for i, c in enumerate(("x", "y", "z")):
        vrec[c] = verts[:, i]
    if has_n:
        nrm = np.asarray(normals, dtype=np.float64)
        for i, c in enumerate(("nx", "ny", "nz")):
            vrec[c] = nrm[:, i]

    with open(filename, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(vrec.tobytes())
        if ntri:
            frec = np.zeros(ntri, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]))
            frec["n"] = 3
            frec["v"] = np.asarray(tris, dtype=np.int64)
            fh.write(frec.tobytes())
    return filename


# ==============================================================================
# THE SELECTION TOOLS
# ==============================================================================
# Each returns a fresh boolean mask over the input points for the shape alone.
# Combining it with what was already selected is combine()'s job, so that "the
# same lasso, but taking away" is one argument rather than a second code path.
# That matters: the deselect variants are where a hand-rolled selector usually
# grows its bugs.

def mask_from_box(xy, x0, y0, x1, y1):
    """Points inside the axis-aligned screen rectangle. Corners in any order."""
    xy = _as_xy(xy)
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    return ((xy[:, 0] >= lo_x) & (xy[:, 0] <= hi_x) &
            (xy[:, 1] >= lo_y) & (xy[:, 1] <= hi_y))


def mask_from_polygon(xy, poly):
    """Points inside a closed polygon, even-odd rule. Vectorised over points.

    The polygon is closed implicitly, so the caller does not repeat the first
    vertex, and it may be concave or self-intersecting -- a lasso drawn by hand
    usually is. Fewer than three vertices selects nothing rather than raising:
    a click that never became a shape is not an error, it is an empty shape.
    """
    xy = _as_xy(xy)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(xy), dtype=bool)
    if len(poly) < 3 or not len(xy):
        return inside
    px, py = xy[:, 0], xy[:, 1]
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    for k in range(len(poly)):
        # Does the horizontal ray from the point cross this edge? A horizontal
        # edge (yj == yi) never straddles, so `straddles` is already False and
        # the division's nan cannot reach the answer.
        straddles = (yi[k] > py) != (yj[k] > py)
        if not straddles.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = (xj[k] - xi[k])*(py - yi[k])/(yj[k] - yi[k]) + xi[k]
        inside ^= straddles & (px < cross)
    return inside


def mask_from_brush(xy, cx, cy, radius):
    """Points within `radius` of one screen position: a single brush dab."""
    xy = _as_xy(xy)
    d = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
    return d <= float(radius)**2


def mask_from_stroke(xy, stroke, radius):
    """Points within `radius` of a painted polyline -- a whole brush drag.

    The UI applies one segment per mouse-move so the cost per event is O(N),
    but the whole stroke has to give the same answer as the sum of its parts or
    a replayed selection would not match what the operator drew. Distance is to
    the SEGMENTS, not to the sampled points: a fast drag samples sparsely, and
    dabs alone would leave gaps the operator never saw.
    """
    xy = _as_xy(xy)
    pts = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
    mask = np.zeros(len(xy), dtype=bool)
    if not len(pts) or not len(xy):
        return mask
    if len(pts) == 1:
        return mask_from_brush(xy, pts[0, 0], pts[0, 1], radius)
    r2 = float(radius)**2
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        ab = b - a
        denom = float(ab @ ab)
        ap = xy - a
        if denom < 1e-12:
            d2 = (ap**2).sum(axis=1)
        else:
            t = np.clip((ap @ ab)/denom, 0.0, 1.0)
            d2 = ((ap - t[:, None]*ab[None, :])**2).sum(axis=1)
        mask |= d2 <= r2
    return mask


def combine(current, new, mode="replace"):
    """Fold a tool's shape into the running selection. This is where deselect is.

    "add"      union -- the default modifier-free second stroke
    "subtract" the deselect variant of every tool (Ctrl in the UI)
    "toggle"   flip inside the shape; handy for a stripe pattern
    "replace"  what a bare drag does with no modifier held
    """
    current = np.asarray(current, dtype=bool)
    new = np.asarray(new, dtype=bool)
    if current.shape != new.shape:
        raise ValueError("mask shapes differ: %s vs %s"
                         % (current.shape, new.shape))
    if mode == "replace":
        return new.copy()
    if mode == "add":
        return current | new
    if mode == "subtract":
        return current & ~new
    if mode == "toggle":
        return current ^ new
    raise ValueError("unknown selection mode %r (want one of %s)"
                     % (mode, ", ".join(MODES)))


def front_surface_mask(xy, depth, candidate, cell=6.0, tol=None):
    """Drop what a screen shape caught THROUGH the part: keep the near surface.

    A rectangle drawn over a plate also catches the breadboard visible between
    the clamps and, at any oblique view, the underside of the plate itself.
    Neither is what the operator drew a box around.

    Pure numpy on purpose -- no z-buffer read, no vtkHardwareSelector. The cell
    is driven over VNC and those paths are the first thing to misbehave on a
    software GL stack. Screen space is bucketed into `cell`-pixel squares and a
    candidate survives only if it is within `tol` of the nearest candidate in
    its own bucket, which needs nothing but the depths already computed for the
    projection.

    THE TOLERANCE IS NOT A FIXED FRACTION. A fraction of the overall depth
    spread works from directly above, where a screen cell covers a nearly
    constant depth. Tilt the view and each cell spans real depth of its own, so
    the far half of EVERY cell fails the test and the selection comes out
    speckled -- not merely ugly: those vertices are genuinely deselected, and
    the sub-mesh written from them has holes.

    So the floor is measured from the data: the MEDIAN per-cell depth span is
    what a single continuous surface costs at this viewing angle, whatever the
    angle is, and anything within twice that is the same surface. A second
    surface behind the first is separated by much more than that -- a plate
    over a bench is 15 mm where the plate's own cells span a fraction of a
    millimetre -- so it is still dropped. Where most cells DO straddle two
    surfaces the median rises and the filter quietly weakens, which is the
    right way for it to fail: it stops removing things rather than starting to
    remove the wrong ones.
    """
    xy = _as_xy(xy)
    depth = np.asarray(depth, dtype=np.float64).ravel()
    candidate = np.asarray(candidate, dtype=bool)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return candidate.copy()

    d = depth[idx]
    cell = float(cell)
    gx = np.floor(xy[idx, 0]/cell).astype(np.int64)
    gy = np.floor(xy[idx, 1]/cell).astype(np.int64)
    # One integer key per bucket. Arithmetic rather than np.unique(axis=0):
    # the latter is minutes-slow shaped like milliseconds at 100k vertices,
    # and this runs on every mouse release.
    rows = int(gy.max() - gy.min()) + 1
    key = (gx - gx.min())*rows + (gy - gy.min())
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    ncell = int(inv.max()) + 1

    nearest = np.full(ncell, np.inf)
    np.minimum.at(nearest, inv, d)

    if tol is None:
        spread = float(d.max() - d.min())
        farthest = np.full(ncell, -np.inf)
        np.maximum.at(farthest, inv, d)
        spans = farthest - nearest
        spans = spans[np.isfinite(spans) & (spans > 0.0)]
        local = 2.0*float(np.median(spans)) if spans.size else 0.0
        tol = max(0.02*spread, local) if spread > 0.0 else 0.0

    out = np.zeros_like(candidate)
    out[idx[d <= nearest[inv] + tol]] = True
    return out


# ==============================================================================
# WHAT THE PIPELINE ACTUALLY CONSUMES
# ==============================================================================

def bounding_region(verts, mask=None):
    """(xmin, ymin, xmax, ymax) in WORLD METRES over the selected vertices.

    The tuple the rest of the pipeline takes -- reconstruct_fine(region=),
    plan_ec_path(region=) -- in their order, which is NOT (xmin, xmax, ymin,
    ymax). Transposing the two has sent a dome sweep 696 mm from the part. z is
    deliberately absent: every consumer of a region here treats it as a
    footprint.
    """
    verts = np.asarray(verts, dtype=np.float64)
    sel = verts if mask is None else verts[np.asarray(mask, dtype=bool)]
    if not len(sel):
        raise ValueError("nothing is selected, so there is no region")
    return (float(sel[:, 0].min()), float(sel[:, 1].min()),
            float(sel[:, 0].max()), float(sel[:, 1].max()))


def bounding_box3d(verts, mask=None):
    """(xmin, ymin, zmin, xmax, ymax, zmax) over the selection, world metres."""
    verts = np.asarray(verts, dtype=np.float64)
    sel = verts if mask is None else verts[np.asarray(mask, dtype=bool)]
    if not len(sel):
        raise ValueError("nothing is selected, so there is no bounding box")
    return (float(sel[:, 0].min()), float(sel[:, 1].min()), float(sel[:, 2].min()),
            float(sel[:, 0].max()), float(sel[:, 1].max()), float(sel[:, 2].max()))


def region_mask(verts, region):
    """The mask a (xmin, ymin, xmax, ymax) region implies -- the initial= path.

    The operator starts from the detector's box rather than from nothing, so
    the common case is correcting a selection instead of drawing one.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if region is None:
        return np.zeros(len(verts), dtype=bool)
    xmin, ymin, xmax, ymax = (float(v) for v in region)
    return ((verts[:, 0] >= xmin) & (verts[:, 0] <= xmax) &
            (verts[:, 1] >= ymin) & (verts[:, 1] <= ymax))


# ==============================================================================
# THE FOOTPRINT: THE ONE PART OF A SELECTION THAT TRAVELS BETWEEN MESHES
# ==============================================================================
# A selection is drawn on the AZURE rough mesh and then used to crop the KEYENCE
# mesh, and neither of the exact forms of a selection survives that trip.
#
# A GLOBAL 3D BOX DOES NOT TRANSFER. The Azure reads this cell's surfaces 11.6 to
# 13.6 mm too far, which is more than the relief of a plate on a bench, so an
# ROI's z range measured on the Azure can sit entirely ABOVE the Keyence mesh of
# the same part -- ranges that do not overlap at all, and a 3D box crop that
# keeps zero faces of twenty thousand. In x and y the same ROI sits comfortably
# inside both meshes; the fault is entirely in z. z is not what the operator drew
# anyway: they were looking straight down at a surface.
#
# THE PER-VERTEX MASK DOES NOT TRANSFER EITHER: it has one entry per Azure vertex
# while the mesh being cropped is the Keyence one, with a different vertex count
# and ordering. The mask cannot transfer by index.
#
# A FOOTPRINT TRANSFERS, because it is in world x-y: the drawn shape rasterised
# onto a grid of `cell`-metre squares. It keeps what a rectangle throws away --
# an L-shaped part's corner is not in it -- and no GLOBAL z range is applied to
# it, so no amount of sensor-to-sensor disagreement in surface height can empty
# a crop.
#
# ------------------------------------------------------------------------------
# AND EACH CELL CARRIES THE HEIGHT OF THE SELECTED SURFACE THERE
# ------------------------------------------------------------------------------
# Dropping z from the decision entirely is not acceptable. A region selected on
# top of a flat plate must not let the Keyence scan part of the table under it,
# and the parts here will not stay flat: a curved specimen has to stay inside its
# ROI in z as well, to within a sensible tolerance.
#
# So `heights` is a second grid over the SAME cells: the z of the selected
# surface in each one, NaN where the selection never covered it. That is a
# height map, not a slab, and the difference is the whole point:
#
#   flat plate    every cell holds the same height, and a band around it IS a
#                 slab -- the wanted behaviour.
#   curved part   the band follows the curve. A global slab over a crown with
#                 25 mm of relief has to be 25 mm tall before it accepts the
#                 crown's own edges, and a 25 mm slab accepts the bench.
#
# A GLOBAL box empties a crop because its z floor is the MINIMUM over the whole
# selection, and the other sensor's whole mesh can sit below that. Per cell the
# two sensors disagree by 0.8-2.6 mm on a flat plate -- measured; see
# GATE_ROI_Z_TOLERANCE_M in noether_path.dpi -- which a tolerance absorbs and a
# floor does not.
#
# Cells with no height (the dilation ring, and any hole in the selection) are
# filled from their neighbours before the map is used, so the band is defined
# everywhere the footprint is. A footprint built from a plain RECTANGLE has no
# height information at all; its map is empty and z is then not applied, which
# the report says out loud rather than implying.
#
# The grid is DILATED by one cell by default. A selection is a set of vertices,
# and the mesh being cropped is sampled somewhere else: the Azure's vertices are
# 6 mm apart and the Keyence's 0.7 mm, so a cell that the Azure happened not to
# land in is still part surface. Dilating outward is the safe direction -- it
# keeps a little too much, which the toolpath crop then trims exactly, rather
# than eating into the region the operator asked for.

#: Cell size of the footprint grid, metres. 2 mm is finer than the Azure rough
#: mesh's own 6 mm sampling and coarser than the Keyence's 0.7 mm, so it
#: resolves the drawn shape without resolving the sampling.
FOOTPRINT_CELL_M = 0.002

#: How many cells the grid is grown by after rasterising. See above: one cell of
#: 2 mm is a third of the Azure mesh's own edge length.
FOOTPRINT_DILATE_CELLS = 1

#: Quantisation of the PACKED height map, metres. 0.1 mm is 60 times finer than
#: the +/-6 mm band it feeds (GATE_ROI_Z_TOLERANCE_M) and 26 times finer than
#: the 2.6 mm sensor-to-sensor disagreement that band exists to absorb, so the
#: packing is lossless for every decision made with it -- and it halves what the
#: map costs in result.json against float32.
FOOTPRINT_Z_QUANTUM_M = 0.0001

#: "no height known in this cell", in the packed form. int16's own minimum, so
#: no real height can collide with it.
_FOOTPRINT_Z_NONE = -32768


def _footprint_grow(g):
    """One cell of 8-neighbour dilation. 8 and not 4: a diagonal staircase on
    the boundary of a hand-drawn shape would otherwise stay open."""
    h = g.copy()
    h[1:, :] |= g[:-1, :]
    h[:-1, :] |= g[1:, :]
    h[:, 1:] |= g[:, :-1]
    h[:, :-1] |= g[:, 1:]
    h[1:, 1:] |= g[:-1, :-1]
    h[:-1, :-1] |= g[1:, 1:]
    h[1:, :-1] |= g[:-1, 1:]
    h[:-1, 1:] |= g[1:, :-1]
    return h


def _footprint_fill_heights(h, grid):
    """Give every cell of `grid` a height, from its neighbours outward.

    -> (filled copy, number of passes). Each pass sets every still-unknown cell
    to the mean of its known 8-neighbours, so the map grows outward from where
    the selection actually sampled it one cell at a time and stops the moment
    nothing more can be reached.

    WHY FILL AT ALL. Two kinds of cell have no sample: the dilation ring, which
    is deliberately outside the drawn shape, and a hole in the middle of a
    selection -- a bolt hole, or a patch the reconstruction missed. Leaving them
    NaN would mean "no height here", and a face over one would then be judged on
    x-y alone: a silent hole in exactly the protection this map provides. Filling
    from the neighbours puts the local surface height there, which on a curved
    part is the right answer and on a flat one is the only answer.

    A mean and not a nearest: on a curve, averaging the known neighbours
    continues the surface, while copying one of them steps.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    grid = np.asarray(grid, dtype=bool)
    n = 0
    # Bounded by the grid: one pass reaches one cell further, so max(nx, ny)
    # passes reach every cell that is reachable at all.
    for _ in range(max(h.shape) + 1):
        miss = grid & ~np.isfinite(h)
        if not miss.any():
            return h, n
        known = np.isfinite(h)
        v = np.where(known, h, 0.0)
        k = known.astype(np.float64)
        sv = np.zeros_like(v)
        sk = np.zeros_like(k)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                # Shifted by slicing, not np.roll: a roll wraps the far edge of
                # the grid round to the near one and would fill a cell from a
                # surface 100 mm away.
                si = slice(max(di, 0), h.shape[0] + min(di, 0))
                di_ = slice(max(-di, 0), h.shape[0] + min(-di, 0))
                sj = slice(max(dj, 0), h.shape[1] + min(dj, 0))
                dj_ = slice(max(-dj, 0), h.shape[1] + min(-dj, 0))
                sv[si, sj] += v[di_, dj_]
                sk[si, sj] += k[di_, dj_]
        upd = miss & (sk > 0)
        if not upd.any():
            return h, n                     # unreachable: nothing to fill from
        h[upd] = sv[upd]/sk[upd]
        n += 1
    return h, n


def footprint_has_heights(fp):
    """Does this footprint carry a height map with anything in it?"""
    h = footprint_decode(fp).get("heights")
    return h is not None and bool(np.isfinite(h).any())


_BARY = {}


def _bary(m):
    """Barycentric weights of a triangular lattice with `m` steps per edge."""
    if m not in _BARY:
        w = [(i, j, m - i - j)
             for i in range(m + 1) for j in range(m + 1 - i)]
        _BARY[m] = np.asarray(w, dtype=np.float64)/float(m)
    return _BARY[m]


def _footprint_cells_from_tris(verts, tris, cell):
    """Every (i, j) cell the TRIANGLES cover in plan. -> (i, j, z) arrays.

    `z` is the height of the surface at each sample, which is what the per-cell
    height map is built from. It comes out of the same barycentric samples as the
    cells, because the alternative -- rasterising twice -- is two chances to
    disagree about which cells the selection covers.

    WHY TRIANGLES AND NOT VERTICES. The Azure rough mesh -- the surface an ROI
    is drawn on in run_automated_scan() -- has a 6.00 mm median edge, because
    reconstruct_rough() sets the TSDF voxel to 0.006. Rasterising its VERTICES
    onto a 2 mm grid gives a 6 mm-pitch lattice of single cells: a leopard print,
    not a region, and a crop built from it would drop stripes of the finer mesh
    it is applied to. The selection is a SURFACE, so the surface is what is
    rasterised, and the grid resolution is then independent of how coarsely the
    mesh that carried the selection happened to be sampled.

    Sampled barycentrically at better than half a cell, with the sample density
    bucketed by triangle size so one big triangle does not set the cost for all
    of them.
    """
    verts = np.asarray(verts, dtype=np.float64)
    # Three columns whatever came in, so the z that feeds the height map is a
    # column of NaN rather than an IndexError when the caller has x-y only.
    if verts.shape[1] < 3:
        verts = np.column_stack([verts[:, 0], verts[:, 1],
                                 np.full(len(verts), np.nan)])
    tri = verts[tris][:, :, :3]
    # A NaN corner would floor() to INT_MIN and ask for a grid the size of the
    # integers, and the reconstruction does emit a handful of NaN vertices per
    # mesh. Only x and y decide that: a NaN z costs the cell its height, not its
    # place.
    good = np.isfinite(tri[:, :, :2]).all(axis=(1, 2))
    if not good.all():
        tri = tri[good]
    if not len(tri):
        raise ValueError("every selected triangle has a non-finite corner")
    e = np.stack([tri[:, 1, :2] - tri[:, 0, :2], tri[:, 2, :2] - tri[:, 1, :2],
                  tri[:, 0, :2] - tri[:, 2, :2]], axis=1)
    longest = np.linalg.norm(e, axis=2).max(axis=1)
    need = np.clip(np.ceil(2.0*longest/cell), 1, 512).astype(np.int64)
    # Powers of two, so a mesh of mixed triangle sizes costs a handful of passes
    # rather than one per distinct size.
    steps = (1 << np.ceil(np.log2(np.maximum(need, 1))).astype(np.int64))
    out_i, out_j, out_z = [], [], []
    for m in np.unique(steps):
        sel = np.flatnonzero(steps == m)
        w = _bary(int(m))
        # Chunked so the (faces x samples x 3) intermediate stays bounded no
        # matter how big the selection is.
        step = max(1, int(2000000//max(len(w), 1)))
        for a in range(0, len(sel), step):
            blk = tri[sel[a:a + step]]
            pts = np.einsum("kb,fbc->fkc", w, blk).reshape(-1, 3)
            out_i.append(np.floor(pts[:, 0]/cell).astype(np.int64))
            out_j.append(np.floor(pts[:, 1]/cell).astype(np.int64))
            out_z.append(pts[:, 2])
    return (np.concatenate(out_i), np.concatenate(out_j),
            np.concatenate(out_z))


def footprint(verts, mask=None, tris=None, cell=None, dilate=None, keep="strict",
              fill=True):
    """The selection's x-y SHAPE as an occupancy grid, in world metres.

    Returns {"cell", "i0", "j0", "grid" (nx x ny bool), "heights" (nx x ny
    float, the z of the selected surface in each cell), "dilated",
    "height_source"}. Cell (i, j) of the grid is world x in
    [(i0 + i)*cell, (i0 + i + 1)*cell) and likewise for y, so the grid is
    anchored to the world and two footprints built from different meshes line up
    cell for cell.

    `heights` is what makes the footprint more than a shadow: each cell holds the
    height of the surface the operator selected THERE, so a crop can ask "is this
    face near the surface I chose" instead of "is this face inside one slab". On
    a flat plate it is a constant; on a curved part it follows the curve, which
    is the case a global slab cannot serve. Cells the selection
    did not sample -- the dilation ring, a hole -- are filled from their
    neighbours, so the map is defined everywhere the grid is. Heights are the
    MEAN of the samples in a cell; see below.

    Pass `tris` whenever the selection is a MESH -- the drawn surface is then
    rasterised, which is what makes the grid resolution independent of the
    mesh's own sampling; see _footprint_cells_from_tris(). With vertices alone
    the points are grown by their own mean spacing instead, because a point set
    that is not filled in is a lattice of dots rather than a region.

    fill=False turns off the point-spacing fill described above, for data known
    to be denser than the cell. It has no effect when `tris` is given.

    keep="strict" (the default here, and NOT submesh()'s default) takes only
    triangles all three of whose corners are selected. The two want opposite
    things: a sub-mesh must not have a ragged boundary, so it carries the
    unselected corners of its rim triangles along -- which on the 6 mm Azure
    mesh pushes its outline 6 mm past what the operator drew. A footprint is
    used to decide what is part surface, and it is the rim of the selection --
    where a plate meets the bench -- that must NOT be in it. Margins are added
    back deliberately and outward, by whoever crops.

    WHY THE MEAN AND NOT THE MAXIMUM for a cell's height. The maximum is "the top
    of the selected surface here", which sounds like what a top-down selection
    means, but it is the statistic a single spike sets: one vertex 5 mm high in a
    2 mm cell would lift the band off the real surface and start dropping part
    faces. The mean of a cell's samples moves by at most the surface's own relief
    across 2 mm, and the band's tolerance is sized for a millimetre of that
    anyway. A cell straddling the rim of a plate reads the mean of the plate and
    the drape, which is LOW rather than high -- the safe direction is the other
    one, so whoever crops keeps the rim out of the footprint (keep="strict") and
    treats the band as a band, not as a floor.
    """
    verts = np.asarray(verts, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool) if mask is not None else None
    cell = FOOTPRINT_CELL_M if cell is None else float(cell)
    if not cell > 0.0:
        raise ValueError("the footprint cell must be positive, got %r" % (cell,))
    dilate = FOOTPRINT_DILATE_CELLS if dilate is None else int(dilate)
    dilate = max(0, dilate)
    if keep not in ("all", "strict"):
        raise ValueError("keep must be 'all' or 'strict', not %r" % (keep,))

    grow = dilate
    if tris is not None and len(tris):
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if sel is not None:
            hit = sel[tris]
            want = hit.all(axis=1) if keep == "strict" else hit.any(axis=1)
            if not want.any() and keep == "strict":
                # A selection thinner than one triangle has no strict interior.
                # Falling back is better than refusing: the operator drew
                # something, and "all" still describes where it is.
                want = hit.any(axis=1)
            tris = tris[want]
        if not len(tris):
            raise ValueError("nothing is selected, so there is no footprint")
        i, j, z = _footprint_cells_from_tris(verts, tris, cell)
        source = "the selected triangles, sampled"
    else:
        pts = verts if sel is None else verts[sel]
        pts = pts[np.isfinite(pts[:, :2]).all(axis=1)] if len(pts) else pts
        if not len(pts):
            raise ValueError("nothing is selected, so there is no footprint")
        i = np.floor(pts[:, 0]/cell).astype(np.int64)
        j = np.floor(pts[:, 1]/cell).astype(np.int64)
        z = (pts[:, 2] if pts.shape[1] >= 3
             else np.full(len(pts), np.nan))
        source = "the selected points"
        # Points, not a surface: fill by their own mean spacing. Without this a
        # sparse cloud rasterises to isolated dots and the crop built from it
        # drops most of the part -- the same disease triangle rasterisation
        # cures for a mesh. Two estimates, whichever is larger: sqrt(area/n) for
        # a patch, and span/n for a strip, whose bounding-box area is zero and
        # which would otherwise get no fill at all.
        dx = float(pts[:, 0].max() - pts[:, 0].min())
        dy = float(pts[:, 1].max() - pts[:, 1].min())
        n_pts = max(len(pts), 1)
        spacing = max(math.sqrt(max(dx*dy, 0.0)/n_pts), max(dx, dy)/n_pts)
        if fill:
            grow = max(grow, int(np.ceil(spacing/cell - 1e-9)))

    # The dilation margin is allocated up front, so growing never clips at the
    # edge of the array -- which would quietly square off the drawn shape.
    i0, j0 = int(i.min()) - grow, int(j.min()) - grow
    nx = int(i.max()) - i0 + 1 + grow
    ny = int(j.max()) - j0 + 1 + grow
    g = np.zeros((nx, ny), dtype=bool)
    g[i - i0, j - j0] = True
    # THE HEIGHT MAP, accumulated over the same samples that set the cells: sum
    # and count, then the mean. np.add.at and not fancy indexing, which keeps
    # only the last sample to land in a cell rather than averaging them.
    s = np.zeros((nx, ny), dtype=np.float64)
    c = np.zeros((nx, ny), dtype=np.float64)
    zf = np.isfinite(z)
    np.add.at(s, (i[zf] - i0, j[zf] - j0), z[zf])
    np.add.at(c, (i[zf] - i0, j[zf] - j0), 1.0)
    h = np.full((nx, ny), np.nan)
    h[c > 0] = s[c > 0]/c[c > 0]
    sampled = int((c > 0).sum())
    for _ in range(grow):
        g = _footprint_grow(g)
    h, passes = _footprint_fill_heights(h, g)
    if not sampled:
        source = "%s (no z: %d cells have no height)" % (source, int(g.sum()))
    return {"cell": cell, "i0": i0, "j0": j0, "grid": g, "heights": h,
            "dilated": grow,
            "height_source": "%s; %d of %d cells sampled directly, the rest "
                             "filled from neighbours in %d pass%s"
                             % (source, sampled, int(g.sum()), passes,
                                "" if passes == 1 else "es")}


def footprint_from_region(region, cell=None, dilate=0):
    """The footprint a plain (xmin, ymin, xmax, ymax) rectangle implies.

    What an ROI that carries no drawn shape falls back to, and what makes a
    rectangle and a drawn shape the same kind of object downstream.

    ITS HEIGHT MAP IS EMPTY, and that is not an oversight: four numbers say
    nothing about how high the surface is. A crop given one of these applies x-y
    only, and says so -- which is honest, and distinguishable from a drawn
    selection whose heights happen to be flat.
    """
    xmin, ymin, xmax, ymax = (float(a) for a in region)
    cell = FOOTPRINT_CELL_M if cell is None else float(cell)
    lo_x, hi_x = min(xmin, xmax), max(xmin, xmax)
    lo_y, hi_y = min(ymin, ymax), max(ymin, ymax)
    # Built directly and NOT through footprint(): a rectangle is two corners, and
    # handing footprint() two points makes its point-spacing fill measure the
    # diagonal as the "spacing" and grow the region by half of it, which turns an
    # 18 x 60 mm region into 80 x 122 mm.
    i0 = int(np.floor(lo_x/cell))
    j0 = int(np.floor(lo_y/cell))
    nx = int(np.floor(hi_x/cell)) - i0 + 1
    ny = int(np.floor(hi_y/cell)) - j0 + 1
    fp = {"cell": cell, "i0": i0, "j0": j0,
          "grid": np.ones((nx, ny), dtype=bool),
          "heights": np.full((nx, ny), np.nan), "dilated": 0,
          "height_source": "a rectangle, which carries no height at all"}
    dilate = max(0, int(dilate))
    return fp if not dilate else footprint_grown(fp, dilate*cell)


def footprint_grown(fp, margin_m):
    """A copy grown outward by `margin_m` metres, rounded up to whole cells.

    The HEIGHT MAP grows with it: the new ring has no samples of its own, so it
    is filled from the cells it touches. That is what keeps the band defined over
    the planning margin -- Noether's passes run a few millimetres past the drawn
    edge, and a margin with no height would be a margin with no z protection.
    """
    fp = footprint_decode(fp)
    n = int(np.ceil(float(margin_m)/fp["cell"] - 1e-9)) if margin_m else 0
    if n <= 0:
        return fp
    g = fp["grid"]
    h = fp.get("heights")
    big = np.zeros((g.shape[0] + 2*n, g.shape[1] + 2*n), dtype=bool)
    big[n:n + g.shape[0], n:n + g.shape[1]] = g
    bigh = np.full(big.shape, np.nan)
    if h is not None:
        bigh[n:n + g.shape[0], n:n + g.shape[1]] = h
    for _ in range(n):
        big = _footprint_grow(big)
    bigh, _passes = _footprint_fill_heights(bigh, big)
    return {"cell": fp["cell"], "i0": fp["i0"] - n, "j0": fp["j0"] - n,
            "grid": big, "heights": bigh, "dilated": fp.get("dilated", 0) + n,
            "height_source": "%s; then grown %d cell%s outward"
                             % (fp.get("height_source", "unrecorded"), n,
                                "" if n == 1 else "s")}


def footprint_mask(fp, points, margin_m=0.0):
    """Which of `points` (N x 2 or N x 3) fall in the footprint. Ignores z."""
    fp = footprint_decode(fp)
    if margin_m:
        fp = footprint_grown(fp, margin_m)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("footprint_mask() takes N x 2 or N x 3 points, got %s"
                         % (p.shape,))
    out = np.zeros(len(p), dtype=bool)
    if not len(p):
        return out
    g = fp["grid"]
    # A NaN point is in no footprint, and floor()ing one is a cast warning on
    # every numpy this runs on. The reconstruction's own smoothing does emit a
    # handful of NaN vertices per mesh, so this is not hypothetical.
    fin = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
    i = np.zeros(len(p), dtype=np.int64)
    j = np.zeros(len(p), dtype=np.int64)
    i[fin] = np.floor(p[fin, 0]/fp["cell"]).astype(np.int64) - fp["i0"]
    j[fin] = np.floor(p[fin, 1]/fp["cell"]).astype(np.int64) - fp["j0"]
    ok = fin & (i >= 0) & (j >= 0) & (i < g.shape[0]) & (j < g.shape[1])
    out[ok] = g[i[ok], j[ok]]
    return out


def footprint_bounds(fp):
    """(xmin, ymin, xmax, ymax) of the occupied cells, world metres."""
    fp = footprint_decode(fp)
    i, j = np.nonzero(fp["grid"])
    if not len(i):
        raise ValueError("the footprint is empty")
    c = fp["cell"]
    return (float((fp["i0"] + i.min())*c), float((fp["j0"] + j.min())*c),
            float((fp["i0"] + i.max() + 1)*c), float((fp["j0"] + j.max() + 1)*c))


def footprint_area_m2(fp):
    """How much surface the footprint covers, square metres."""
    fp = footprint_decode(fp)
    return float(fp["grid"].sum())*fp["cell"]**2


def footprint_encode(fp):
    """A JSON-safe form: one hex-packed bit row per x cell, plus the heights.

    Compact enough to sit in the ROI's result.json -- a 30 x 84 mm selection at
    a 2 mm cell is 16 rows of 12 hex characters of shape, and 16 rows of 4 hex
    characters per cell of height -- and both decode in four lines, which is what
    lets noether_path.dpi read one without importing this file.

    Heights are int16 multiples of FOOTPRINT_Z_QUANTUM_M above `z0`, with
    _FOOTPRINT_Z_NONE for "not known". z0 is the map's own minimum, so the
    quantised values start at zero and the int16 range is spent on the part's
    relief rather than on its distance from the robot base.
    """
    fp = footprint_decode(fp)
    g = fp["grid"]
    out = {"cell": float(fp["cell"]), "i0": int(fp["i0"]), "j0": int(fp["j0"]),
           "nx": int(g.shape[0]), "ny": int(g.shape[1]),
           "cells": int(g.sum()), "dilated": int(fp.get("dilated", 0))}
    if fp.get("height_source"):
        out["height_source"] = str(fp["height_source"])
    out["rows"] = [np.packbits(g[i], bitorder="little").tobytes().hex()
                   for i in range(g.shape[0])]
    h = fp.get("heights")
    if h is not None and np.isfinite(h).any():
        q = FOOTPRINT_Z_QUANTUM_M
        z0 = float(np.nanmin(h))
        k = np.full(h.shape, _FOOTPRINT_Z_NONE, dtype=np.int16)
        fin = np.isfinite(h)
        # Clipped one short of the sentinel at each end, so a height 3.3 m off z0
        # -- which no cell workspace has -- saturates rather than decoding as
        # "unknown" or wrapping to the other sign.
        k[fin] = np.clip(np.rint((h[fin] - z0)/q),
                         _FOOTPRINT_Z_NONE + 1, 32767).astype(np.int16)
        out["z0"] = z0
        out["z_quantum"] = float(q)
        out["z_lo"] = float(np.nanmin(h))
        out["z_hi"] = float(np.nanmax(h))
        out["z_cells"] = int(fin.sum())
        out["zrows"] = [k[i].tobytes().hex() for i in range(h.shape[0])]
    return out


def footprint_decode(fp):
    """The in-memory form, from either form. Idempotent on purpose."""
    if fp is None:
        raise ValueError("there is no footprint")
    if "grid" in fp:
        return fp
    nx, ny = int(fp["nx"]), int(fp["ny"])
    g = np.zeros((nx, ny), dtype=bool)
    rows = fp["rows"]
    if len(rows) != nx:
        raise ValueError("footprint has %d rows for nx %d" % (len(rows), nx))
    for i, h in enumerate(rows):
        b = np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8),
                          bitorder="little")
        g[i] = b[:ny]
    heights = np.full((nx, ny), np.nan)
    zrows = fp.get("zrows")
    if zrows is not None:
        if len(zrows) != nx:
            raise ValueError("footprint has %d height rows for nx %d"
                             % (len(zrows), nx))
        q = float(fp.get("z_quantum", FOOTPRINT_Z_QUANTUM_M))
        z0 = float(fp["z0"])
        for i, s in enumerate(zrows):
            k = np.frombuffer(bytes.fromhex(s), dtype="<i2")
            if len(k) != ny:
                raise ValueError("footprint height row %d has %d cells for ny %d"
                                 % (i, len(k), ny))
            row = z0 + k.astype(np.float64)*q
            row[k == _FOOTPRINT_Z_NONE] = np.nan
            heights[i] = row
    return {"cell": float(fp["cell"]), "i0": int(fp["i0"]), "j0": int(fp["j0"]),
            "grid": g, "heights": heights,
            "dilated": int(fp.get("dilated", 0)),
            "height_source": fp.get("height_source", "unrecorded")}


def footprint_heights_at(fp, points, margin_m=0.0):
    """The SELECTED SURFACE's height under each of `points`. -> N floats.

    NaN where the point is outside the footprint, or in a cell the height map
    does not know -- which for a drawn selection is nowhere, because the map is
    filled, and for a rectangle is everywhere.
    """
    fp = footprint_decode(fp)
    if margin_m:
        fp = footprint_grown(fp, margin_m)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("footprint_heights_at() takes N x 2 or N x 3 points, "
                         "got %s" % (p.shape,))
    out = np.full(len(p), np.nan)
    if not len(p):
        return out
    g, h = fp["grid"], fp.get("heights")
    if h is None:
        return out
    fin = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
    i = np.zeros(len(p), dtype=np.int64)
    j = np.zeros(len(p), dtype=np.int64)
    i[fin] = np.floor(p[fin, 0]/fp["cell"]).astype(np.int64) - fp["i0"]
    j[fin] = np.floor(p[fin, 1]/fp["cell"]).astype(np.int64) - fp["j0"]
    ok = fin & (i >= 0) & (j >= 0) & (i < g.shape[0]) & (j < g.shape[1])
    inside = np.zeros(len(p), dtype=bool)
    inside[ok] = g[i[ok], j[ok]]
    out[inside] = h[i[inside], j[inside]]
    return out


def footprint_band_mask(fp, points, z_tolerance, margin_m=0.0):
    """Which of `points` (N x 3) are in the footprint AND near its surface.

    -> (mask, no_height). `mask` is the x-y test AND |z - the cell's height| <=
    z_tolerance. `no_height` flags the points that fell in the footprint but
    where the map holds no height, so a caller can say how much of its answer
    z had no say in rather than pretending z was applied. Those points are IN --
    the x-y test is the older and weaker of the two, not a reason to drop
    surface -- but they are counted.

    z_tolerance is required, and deliberately has no default here: the number is
    a property of the two sensors being compared, and it lives with the crop that
    knows which those are (GATE_ROI_Z_TOLERANCE_M in noether_path.dpi).
    """
    fp = footprint_decode(fp)
    if margin_m:
        fp = footprint_grown(fp, margin_m)      # once, not once per test below
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3:
        raise ValueError("footprint_band_mask() takes N x 3 points, got %s"
                         % (p.shape,))
    inxy = footprint_mask(fp, p)
    h = footprint_heights_at(fp, p)
    known = np.isfinite(h)
    no_height = inxy & ~known
    tol = abs(float(z_tolerance))
    band = inxy.copy()
    band[inxy & known] = (np.abs(p[inxy & known, 2] - h[inxy & known]) <= tol)
    return band, no_height


def describe_footprint(fp):
    """One line: how big the drawn shape is, how much of its box it fills, and
    what height its surface is at."""
    fp = footprint_decode(fp)
    x0, y0, x1, y1 = footprint_bounds(fp)
    area = footprint_area_m2(fp)
    box = max((x1 - x0)*(y1 - y0), 1e-12)
    h = fp.get("heights")
    if h is None or not np.isfinite(h).any():
        z = "; NO height map (%s)" % (fp.get("height_source", "unrecorded"),)
    else:
        z = ("; surface at z %.4f..%.4f (%.1f mm of relief across it)"
             % (np.nanmin(h), np.nanmax(h),
                (np.nanmax(h) - np.nanmin(h))*1e3))
    return ("%.1f x %.1f mm, %.1f cm2 of surface in %d cells of %.1f mm "
            "(%.0f%% of its bounding box)%s"
            % ((x1 - x0)*1e3, (y1 - y0)*1e3, area*1e4,
               int(fp["grid"].sum()), fp["cell"]*1e3, 100.0*area/box, z))


def submesh(verts, tris, mask, normals=None, keep="all"):
    """The selected part as a standalone mesh, re-indexed from zero.

    Returns (verts2, tris2, normals2, vertex_index_map). `keep` decides which
    triangles survive a partly-selected edge:

      "all"  a triangle with ANY selected vertex is kept, and its unselected
             corners come along. The selection then has no ragged edge, which
             is what a planner wants -- Noether's plane slicer refuses a mesh
             whose boundary is a comb of single triangles.
      "strict"  only triangles with all three vertices selected.

    A point cloud (tris None) gives tris2 None and just the selected points.
    """
    verts = np.asarray(verts, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(verts):
        raise ValueError("mask has %d entries for %d vertices"
                         % (len(mask), len(verts)))
    if not mask.any():
        raise ValueError("nothing is selected, so there is no sub-mesh")

    if tris is None or not len(tris):
        idx = np.flatnonzero(mask)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        return (verts[idx], None,
                None if normals is None else np.asarray(normals)[idx], remap)

    tris = np.asarray(tris, dtype=np.int64)
    hit = mask[tris]
    if keep == "strict":
        want = hit.all(axis=1)
    elif keep == "all":
        want = hit.any(axis=1)
    else:
        raise ValueError("keep must be 'all' or 'strict', not %r" % (keep,))
    kept = tris[want]
    if not len(kept):
        raise ValueError("the selection touches no triangle")

    idx = np.unique(kept)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    return (verts[idx], remap[kept],
            None if normals is None else np.asarray(normals)[idx], remap)


def describe(verts, mask):
    """One line an operator and a log can both read."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if not n:
        return "nothing selected (0 of %d)" % (len(mask),)
    x0, y0, x1, y1 = bounding_region(verts, mask)
    return ("%d of %d points (%.1f%%), %.1f x %.1f mm at x %.3f..%.3f "
            "y %.3f..%.3f" % (n, len(mask), 100.0*n/len(mask),
                              (x1 - x0)*1e3, (y1 - y0)*1e3, x0, x1, y0, y1))


# ==============================================================================
# PROJECTION
# ==============================================================================

def project(verts, matrix, width, height):
    """World points -> (display xy, depth), given a 4x4 composite camera matrix.

    VTK hands out exactly this matrix (GetCompositeProjectionTransformMatrix),
    so the UI does not roll its own camera maths -- but having it here means
    the test can drive the whole selection chain through a hand-written
    orthographic matrix with no VTK installed at all.

    Display y is flipped to VTK's convention (origin at the BOTTOM left), which
    is the frame QVTKRenderWindowInteractor reports mouse positions in. Depth
    comes back with SMALLER MEANING NEARER, which is what front_surface_mask()
    wants; VTK's normalised device z already runs that way.
    """
    verts = np.asarray(verts, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    h = np.column_stack([verts, np.ones(len(verts))]) @ m.T
    w = h[:, 3].copy()
    w[np.abs(w) < 1e-12] = 1e-12
    ndc = h[:, :3]/w[:, None]
    xy = np.column_stack([(ndc[:, 0] + 1.0)*0.5*width,
                          (ndc[:, 1] + 1.0)*0.5*height])
    return xy, ndc[:, 2]


def _as_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("screen points must be N x 2, got %s" % (xy.shape,))
    return xy


def height_colors(z, lut=None):
    """Per-vertex RGB uint8 shading a surface by height. Blue low, red high.

    A flat plate on a bench is 20 mm of relief in a 200 mm scene, so the ramp
    is stretched over the 2nd..98th percentile rather than the full range: one
    stray vertex under the table would otherwise flatten the whole part to a
    single colour, which is precisely the case this tool exists to handle.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    lo, hi = np.percentile(z, 2.0), np.percentile(z, 98.0)
    if hi - lo < 1e-9:
        lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-9:
        t = np.zeros(len(z))
    else:
        t = np.clip((z - lo)/(hi - lo), 0.0, 1.0)
    if lut is None:
        lut = np.array([[ 49,  54, 149], [ 69, 117, 180], [116, 173, 209],
                        [171, 217, 233], [224, 243, 248], [254, 224, 144],
                        [253, 174,  97], [244, 109,  67], [215,  48,  39]],
                       dtype=np.float64)
    pos = t*(len(lut) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(lut) - 2)
    f = (pos - i0)[:, None]
    rgb = lut[i0]*(1.0 - f) + lut[i0 + 1]*f
    return np.clip(rgb, 0, 255).astype(np.uint8)
