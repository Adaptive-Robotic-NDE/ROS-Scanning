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
# File:  roi_select_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the region-of-interest selection geometry, with no window
#   involved.
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
"""The ROI selection geometry, with no window anywhere near it.

WHY THIS EXISTS. The selection logic and the thing that draws it are separate
files precisely so this can run: roi_select_geom.py imports numpy and nothing
else, so every tool the operator has -- rectangle, lasso, brush, and the
deselect variant of each -- is exercised on a headless box with no X server, no
VTK and no Qt. What cannot be tested here is the mouse; what can be tested is
every answer the mouse produces.

Two sections run against REAL scan data and skip themselves, loudly, when it is
not on the machine -- these are gigabytes that cannot be committed:

  * a Keyence reconstruction whose YOLO box was 135 x 287 mm for an 83 x 165 mm
    plate: a rectangle over the plate has to recover the plate.
  * a pair of Keyence meshes an Azure-drawn ROI cropped to NOTHING. The ROI's
    3D bounds came off the Azure rough mesh at z 0.9416..0.9435 and both Keyence
    meshes of the same part top out at 0.9411, so a 3D-box crop kept 0 faces of
    20327 and 0 of 11896. The drawn x-y footprint keeps 5705 and 5525. That is
    the measurement the footprint exists for.

z IS IN THE DECISION as a PER-CELL height band and not a global box. The real
data above pins the tolerance (the two sensors disagree by 0.8-2.6 mm per cell,
so +/-6 mm keeps all 5705 and 5525) and test_footprint_heights() pins the part
the real data cannot: a 26.8 mm cylindrical crown, where a global slab wide
enough to hold the crown holds the bench under it and a per-cell band does not.

    python3 roi_select_test.py
"""

import functools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_data
import roi_select_geom as G

#: A real stage-3 Keyence reconstruction, from the fixture repository.
REAL_MESH = test_data.path("keyence", "roi_spray",
                           "specimen_keyence_poisson_filtered_ecplan.ply")
REAL_POINTS = test_data.path("keyence", "roi_spray",
                             "specimen_keyence_final_points.ply")

#: Where the plate actually is, measured off that mesh.
PLATE = (0.775, -0.02, 0.858, 0.145)

fails = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Make the checks one section ran into that section's pytest verdict.

    check() records rather than raises, so a standalone run reports every false
    fact in one pass instead of stopping at the first. Under pytest that is a
    test which passes while printing FAIL, so the wrapper asserts on whatever
    its section added -- and pytest then names the section that broke rather
    than the file-wide backstop at the bottom. main() calls fn.__wrapped__ to
    keep the report-everything behaviour of the standalone run.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        added = fails[mark:]
        assert not added, "%d check(s) failed: %s" % (len(added),
                                                      "; ".join(added))
    return wrapper


def raises(fn, what, must_say=()):
    try:
        fn()
    except Exception as e:
        missing = [s for s in must_say if s not in str(e)]
        check(what, not missing, "message lacks %s: %s" % (missing, e))
        return
    check(what, False, "did not raise")


def grid(n=11, lo=0.0, hi=10.0):
    """An n x n lattice of screen points, and its own (x, y) columns."""
    a = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(a, a, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


# ==============================================================================

@_checked
def test_box():
    print("=== rectangle select ===")
    xy = grid(11)                      # 0..10 step 1, 121 points
    m = G.mask_from_box(xy, 2.0, 3.0, 5.0, 7.0)
    check("a 4 x 5 lattice window selects 20", int(m.sum()) == 20,
          "got %d" % m.sum())
    inside = xy[m]
    check("every selected point is inside the rectangle",
          bool((inside[:, 0] >= 2).all() and (inside[:, 0] <= 5).all() and
               (inside[:, 1] >= 3).all() and (inside[:, 1] <= 7).all()))

    flip = G.mask_from_box(xy, 5.0, 7.0, 2.0, 3.0)
    check("corners in either order give the same rectangle",
          bool((flip == m).all()))

    check("a zero-area drag selects the lattice point under it",
          int(G.mask_from_box(xy, 4.0, 4.0, 4.0, 4.0).sum()) == 1)
    check("a rectangle off the part selects nothing",
          int(G.mask_from_box(xy, 50.0, 50.0, 60.0, 60.0).sum()) == 0)


@_checked
def test_polygon():
    print("=== lasso / polygon select ===")
    xy = grid(11)

    square = [(2.0, 2.0), (2.0, 6.0), (6.0, 6.0), (6.0, 2.0)]
    m = G.mask_from_polygon(xy, square)
    # Even-odd with a `px < cross` ray: the closed left/bottom edges land in,
    # the right/top edges land out. 4 x 4 = 16 either way.
    check("a square lasso selects a square block", int(m.sum()) == 16,
          "got %d" % m.sum())
    sel = xy[m]
    check("the block is where the square was",
          bool(sel[:, 0].min() == 2 and sel[:, 0].max() == 5 and
               sel[:, 1].min() == 2 and sel[:, 1].max() == 5),
          "%s" % (sel.min(axis=0).tolist() + sel.max(axis=0).tolist(),))

    check("winding order does not matter",
          bool((G.mask_from_polygon(xy, square[::-1]) == m).all()))

    # A C-shape: the concave bite must NOT be selected. This is the case a
    # naive convex-hull or bounding-box "lasso" gets wrong, and it is the
    # reason the lasso exists at all -- a plate with a mount block at one end
    # is exactly this shape.
    cee = [(1.0, 1.0), (1.0, 9.0), (9.0, 9.0), (9.0, 7.0), (3.0, 7.0),
           (3.0, 3.0), (9.0, 3.0), (9.0, 1.0)]
    c = G.mask_from_polygon(xy, cee)
    bite = G.mask_from_box(xy, 4.0, 4.0, 8.0, 6.0)
    check("the concave bite of a C is not selected", not bool((c & bite).any()),
          "%d points of the bite were selected" % int((c & bite).sum()))
    check("the arms of the C are selected",
          bool(c[np.flatnonzero(G.mask_from_box(xy, 4.0, 8.0, 8.0, 8.0))].all()))

    check("two points are not a shape",
          int(G.mask_from_polygon(xy, [(0.0, 0.0), (5.0, 5.0)]).sum()) == 0)

    # A horizontal edge is where an even-odd test divides by zero.
    flat = [(2.0, 4.0), (8.0, 4.0), (8.0, 8.0), (2.0, 8.0)]
    f = G.mask_from_polygon(xy, flat)
    check("a horizontal edge does not poison the crossing test",
          int(f.sum()) == 24 and bool(np.isfinite(xy[f]).all()),
          "got %d" % int(f.sum()))


@_checked
def test_brush():
    print("=== paint / brush select ===")
    xy = grid(11)

    m = G.mask_from_brush(xy, 5.0, 5.0, 1.2)
    check("a radius-1.2 dab takes the point and its 4 neighbours",
          int(m.sum()) == 5, "got %d" % m.sum())
    check("radius 1.5 reaches the diagonals too, at 1.414",
          int(G.mask_from_brush(xy, 5.0, 5.0, 1.5).sum()) == 9,
          "got %d" % int(G.mask_from_brush(xy, 5.0, 5.0, 1.5).sum()))
    check("a bigger radius takes more",
          int(G.mask_from_brush(xy, 5.0, 5.0, 3.0).sum()) >
          int(m.sum()))

    # A drag samples sparsely. Distance-to-segment is what stops a fast drag
    # leaving holes the operator never saw.
    stroke = [(1.0, 5.0), (9.0, 5.0)]
    s = G.mask_from_stroke(xy, stroke, 0.5)
    check("a straight drag paints the whole line it crossed",
          int(s.sum()) == 9, "got %d" % s.sum())
    check("the painted line is the row it was drawn along",
          bool((xy[s][:, 1] == 5.0).all()))

    dabs = np.zeros(len(xy), dtype=bool)
    for p in stroke:
        dabs |= G.mask_from_brush(xy, p[0], p[1], 0.5)
    check("two dabs alone would have missed 7 of those points",
          int(dabs.sum()) == 2)

    # The whole stroke must equal the sum of its segments, because the UI
    # applies one segment per mouse-move and a replay must match the drawing.
    path = [(1.0, 1.0), (5.0, 1.0), (5.0, 9.0), (9.0, 9.0)]
    whole = G.mask_from_stroke(xy, path, 0.6)
    piece = np.zeros(len(xy), dtype=bool)
    for k in range(len(path) - 1):
        piece |= G.mask_from_stroke(xy, path[k:k + 2], 0.6)
    check("a stroke equals the union of its segments",
          bool((whole == piece).all()))
    check("an empty stroke selects nothing",
          int(G.mask_from_stroke(xy, [], 1.0).sum()) == 0)
    check("a one-point stroke is a dab",
          bool((G.mask_from_stroke(xy, [(5.0, 5.0)], 1.5) ==
                G.mask_from_brush(xy, 5.0, 5.0, 1.5)).all()))


@_checked
def test_combine():
    print("=== deselect, and the other ways a shape folds in ===")
    xy = grid(11)
    left = G.mask_from_box(xy, 0.0, 0.0, 5.0, 10.0)
    low = G.mask_from_box(xy, 0.0, 0.0, 10.0, 5.0)

    check("replace forgets what was there",
          bool((G.combine(left, low, "replace") == low).all()))
    check("add is a union",
          bool((G.combine(left, low, "add") == (left | low)).all()))
    check("subtract is the deselect variant of every tool",
          bool((G.combine(left, low, "subtract") == (left & ~low)).all()))
    check("toggle flips inside the shape",
          bool((G.combine(left, low, "toggle") == (left ^ low)).all()))

    # Draw the part, then take back the clamp: the actual workflow.
    sel = G.mask_from_box(xy, 2.0, 2.0, 8.0, 8.0)
    n_before = int(sel.sum())
    sel = G.combine(sel, G.mask_from_brush(xy, 8.0, 8.0, 1.2), "subtract")
    check("painting with deselect removes only what it covered",
          int(sel.sum()) == n_before - 3, "got %d from %d" % (sel.sum(), n_before))
    check("deselecting outside the selection changes nothing",
          int(G.combine(sel, G.mask_from_box(xy, 50, 50, 60, 60),
                        "subtract").sum()) == int(sel.sum()))

    all_on = np.ones(len(xy), dtype=bool)
    check("select-all then clear is empty",
          int(G.combine(all_on, np.zeros(len(xy), bool), "replace").sum()) == 0)
    raises(lambda: G.combine(left, low, "sideways"), "an unknown mode is refused",
           ("sideways", "replace"))
    raises(lambda: G.combine(left, low[:5]), "a mismatched mask is refused",
           ("shapes differ",))


@_checked
def test_front_surface():
    print("=== keeping the near surface, not what is behind it ===")
    # A plate at depth 0.1 over a bench at depth 0.5, both under the cursor.
    n = 200
    rng = np.random.default_rng(7)
    xy_plate = rng.uniform(2.0, 8.0, size=(n, 2))
    xy_bench = rng.uniform(0.0, 10.0, size=(n, 2))
    xy = np.vstack([xy_plate, xy_bench])
    depth = np.concatenate([np.full(n, 0.1), np.full(n, 0.5)])

    box = G.mask_from_box(xy, 3.0, 3.0, 7.0, 7.0)
    check("the raw rectangle catches bench as well as plate",
          bool(box[n:].any()) and bool(box[:n].any()))

    near = G.front_surface_mask(xy, depth, box, cell=1.0, tol=0.05)
    check("the near-surface filter keeps plate points",
          int(near[:n].sum()) > 0.8*int(box[:n].sum()),
          "kept %d of %d" % (near[:n].sum(), box[:n].sum()))
    check("and drops the bench seen through the same pixels",
          int(near[n:].sum()) < 0.35*int(box[n:].sum()),
          "kept %d of %d bench points" % (near[n:].sum(), box[n:].sum()))
    check("it never adds anything the shape did not catch",
          not bool((near & ~box).any()))
    check("an empty candidate stays empty",
          int(G.front_surface_mask(xy, depth, np.zeros(len(xy), bool)).sum()) == 0)

    flat = G.front_surface_mask(xy, np.zeros(len(xy)), box, cell=1.0)
    check("a surface with no depth spread is left alone",
          bool((flat == box).all()))

    # THE TILTED-SURFACE REGRESSION. One continuous plane seen obliquely, so
    # every screen cell spans real depth of its own. A tolerance that is a
    # fixed fraction of the overall spread then fails the far half of EVERY
    # cell and speckles the selection -- and those vertices are really
    # deselected, so the sub-mesh written from them has holes in it. On a real
    # reconstruction a fixed fraction keeps 0.7% of the plate from an edge-on
    # view and 92.0% from an isometric one; the adaptive tolerance below keeps
    # 100.0% and 98.7% of the same selections.
    n2 = 4000
    rng2 = np.random.default_rng(11)
    sx = rng2.uniform(0.0, 100.0, n2)
    sy = rng2.uniform(0.0, 100.0, n2)
    tilt = np.column_stack([sx, sy])
    # Depth rises steeply across the view: a 6-pixel cell spans 0.06 of it.
    tilt_depth = sx*0.01
    whole = np.ones(n2, dtype=bool)
    kept = G.front_surface_mask(tilt, tilt_depth, whole, cell=6.0)
    check("one tilted surface is NOT speckled by its own depth gradient",
          int(kept.sum()) > 0.97*n2, "kept %d of %d" % (kept.sum(), n2))

    # np.ptp(a), not a.ptp(): the method was removed in numpy 2.0, and this has to
    # run on the production system's 22.04 numpy as well as on a current one.
    stale = G.front_surface_mask(tilt, tilt_depth, whole, cell=6.0,
                                 tol=0.02*float(np.ptp(tilt_depth)))
    check("a fixed-fraction tolerance speckles it, which is why it is not used",
          int(stale.sum()) < 0.75*n2, "kept %d of %d" % (stale.sum(), n2))

    # ...and the same adaptive tolerance must STILL separate two surfaces:
    # a tilted plate over the middle of a bench, which is the actual scene.
    # Three distinct populations, and all three answers matter -- the bench
    # you can SEE is not hidden and must be kept, or a rectangle drawn round
    # the whole table would come back with holes in it.
    nb = 8000
    bx = rng2.uniform(0.0, 100.0, nb)
    by = rng2.uniform(0.0, 100.0, nb)
    # The plate is the middle 40% of the view; the bench fills all of it and
    # sits 5 units further away.
    mid = ((tilt[:, 0] > 30) & (tilt[:, 0] < 70) &
           (tilt[:, 1] > 30) & (tilt[:, 1] < 70))
    npl = int(mid.sum())
    scene = np.vstack([tilt[mid], np.column_stack([bx, by])])
    scene_depth = np.concatenate([tilt_depth[mid], bx*0.01 + 5.0])
    behind = (bx > 30) & (bx < 70) & (by > 30) & (by < 70)

    sep = G.front_surface_mask(scene, scene_depth,
                               np.ones(len(scene), dtype=bool), cell=6.0)
    check("the near surface is kept whole",
          int(sep[:npl].sum()) > 0.97*npl,
          "kept %d of %d" % (sep[:npl].sum(), npl))
    check("the far surface HIDDEN BEHIND it is dropped",
          int(sep[npl:][behind].sum()) < 0.05*int(behind.sum()),
          "kept %d of %d" % (sep[npl:][behind].sum(), behind.sum()))
    check("but the far surface you can plainly see is kept",
          int(sep[npl:][~behind].sum()) > 0.9*int((~behind).sum()),
          "kept %d of %d" % (sep[npl:][~behind].sum(), (~behind).sum()))


@_checked
def test_region():
    print("=== the (xmin, ymin, xmax, ymax) the pipeline consumes ===")
    v = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0],
                  [0.0, 2.0, 1.0], [9.0, 9.0, 9.0]])
    m = np.array([True, True, True, True, False])
    check("the region is the selection's footprint, in metres",
          G.bounding_region(v, m) == (0.0, 0.0, 1.0, 2.0),
          "%s" % (G.bounding_region(v, m),))
    check("the order is xmin, ymin, xmax, ymax -- not xmin, xmax, ymin, ymax",
          G.bounding_region(v, m)[1] == 0.0 and G.bounding_region(v, m)[2] == 1.0)
    check("the outlier is excluded because it is not selected",
          G.bounding_region(v, m)[2] == 1.0)
    check("z is carried by the 3D box, not the region",
          G.bounding_box3d(v, m) == (0.0, 0.0, 1.0, 1.0, 2.0, 1.0),
          "%s" % (G.bounding_box3d(v, m),))
    raises(lambda: G.bounding_region(v, np.zeros(5, bool)),
           "an empty selection has no region", ("nothing is selected",))

    back = G.region_mask(v, (0.0, 0.0, 1.0, 2.0))
    check("region_mask round-trips the region it came from",
          bool((back == m).all()), "%s" % (back.tolist(),))
    check("no initial region means nothing preselected",
          int(G.region_mask(v, None).sum()) == 0)


def _sampled_surface(pitch=0.006, x0=0.800, y0=0.050, size=0.042, z=0.9420):
    """A flat triangulated surface sampled every `pitch` metres. -> (verts, tris).

    The Azure rough mesh's 6 mm sampling by default, which is the quantisation
    every region measured off vertex coordinates inherits.
    """
    a = np.arange(0.0, size + 1e-9, pitch) + x0
    b = np.arange(0.0, size + 1e-9, pitch) + y0
    gx, gy = np.meshgrid(a, b, indexing="ij")
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    n = len(a)
    idx = np.arange(n*n).reshape(n, n)
    tris = np.vstack([
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                         idx[1:, 1:].ravel()]),
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(),
                         idx[:-1, 1:].ravel()])])
    return verts, tris


@_checked
def test_selection_region():
    print("=== the region the pipeline is told to scan ===")
    # THE FAILURE THIS SECTION EXISTS FOR. A region measured as the minimum and
    # maximum of the SELECTED VERTICES is quantised by the mesh's sampling, not
    # by what the operator drew. On a 6 mm surface a band a few eddy-current
    # lines wide (0.3 mm apiece) catches one column of vertices, and the
    # rectangle comes back with xmin == xmax -- which reached the Keyence
    # reconstruction and stopped the run with "no extent" ninety seconds later.
    cell = G.FOOTPRINT_CELL_M
    v = np.array([[0.8070, 0.0270, 0.9420],
                  [0.8070, 0.0570, 0.9420],
                  [0.8070, 0.0870, 0.9420]])
    m = np.ones(3, dtype=bool)

    raw = G.bounding_region(v, m)
    check("one column of vertices spans zero in x -- the raw box that stopped "
          "a run", raw[2] - raw[0] == 0.0, "%s" % (raw,))

    r = G.selection_region(v, m)
    check("the reported region has a whole cell of extent on x instead",
          abs((r[2] - r[0]) - cell) < 1e-12,
          "%.3f mm" % ((r[2] - r[0])*1e3,))
    check("and it contains the points it was measured from",
          r[0] <= raw[0] and r[1] <= raw[1] and r[2] >= raw[2] and r[3] >= raw[3],
          "%s against %s" % (r, raw))
    check("it lands on the selection lattice, so the rectangle and the drawn "
          "shape share one grid",
          all(abs(a/cell - round(a/cell)) < 1e-6 for a in r), "%s" % (r,))

    # A band the operator can see and the lattice cannot: narrower than one cell,
    # and not zero.
    thin = np.array([[0.5000, 0.1000, 0.0], [0.5009, 0.1400, 0.0]])
    rt = G.selection_region(thin, np.ones(2, dtype=bool))
    check("a band 0.9 mm wide -- under one cell -- still reports a cell",
          rt[2] - rt[0] >= cell - 1e-12, "%.3f mm" % ((rt[2] - rt[0])*1e3,))
    check("...and that cell is the one the band is in",
          rt[0] <= 0.5000 and rt[2] >= 0.5009, "%s" % (rt,))

    # EVERY selection with a point in it, whatever shape, over a range of sizes
    # from one point to a patch. Zero extent on either axis is the failure.
    rng = np.random.default_rng(7)
    worst = 1.0
    for _ in range(400):
        k = int(rng.integers(1, 9))
        p = rng.uniform([0.70, -0.05], [0.90, 0.15], size=(k, 2))
        pv = np.column_stack([p, np.zeros(k)])
        rr = G.selection_region(pv, np.ones(k, dtype=bool))
        worst = min(worst, rr[2] - rr[0], rr[3] - rr[1])
    check("no selection of any shape or size reports a zero extent",
          worst >= cell - 1e-12, "the narrowest was %.4f mm" % (worst*1e3,))

    raises(lambda: G.selection_region(v, np.zeros(3, dtype=bool)),
           "an empty selection still has no region at all",
           ("nothing is selected",))

    # The reconstruction emits a few NaN vertices per mesh. One inside the
    # selection must cost itself and not the region -- a NaN cell index is an
    # error, not a wide region.
    nan_v = np.vstack([v, [[np.nan, np.nan, 0.9420]]])
    rn = G.selection_region(nan_v, np.ones(4, dtype=bool))
    check("a NaN vertex in the selection costs itself and not the region",
          all(np.isfinite(rn)) and abs((rn[2] - rn[0]) - cell) < 1e-12,
          "%s" % (rn,))
    raises(lambda: G.selection_region(np.array([[np.nan, np.nan, 0.0]]),
                                      np.ones(1, dtype=bool)),
           "a selection of nothing but NaN has no region",
           ("nothing is selected",))


@_checked
def test_selection_region_covers_the_drawn_shape():
    print("=== the region covers every cell the operator hit ===")
    verts, tris = _sampled_surface()
    # A strip one mesh row wide: the case where the drawn shape reaches past the
    # vertices that carry it, because a strip has no strict triangle interior.
    strip = (verts[:, 0] >= 0.806) & (verts[:, 0] <= 0.812)
    fp = G.footprint(verts, strip, tris=tris)
    r = G.selection_region(verts, strip, footprint=fp)
    fb = G.footprint_bounds(fp)
    check("the region contains the drawn footprint's whole bounding box",
          r[0] <= fb[0] + 1e-12 and r[1] <= fb[1] + 1e-12
          and r[2] >= fb[2] - 1e-12 and r[3] >= fb[3] - 1e-12,
          "region %s against footprint %s" % (r, fb))

    i, j = np.nonzero(fp["grid"])
    cx = (fp["i0"] + i + 0.5)*fp["cell"]
    cy = (fp["j0"] + j + 0.5)*fp["cell"]
    check("and every one of its %d occupied cells is inside the region" % (len(i),),
          bool(((cx >= r[0]) & (cx <= r[2]) & (cy >= r[1])
                & (cy <= r[3])).all()),
          "%d cells fell outside"
          % (int((~((cx >= r[0]) & (cx <= r[2]) & (cy >= r[1])
                    & (cy <= r[3]))).sum()),))

    without = G.selection_region(verts, strip)
    check("folding the footprint in can only widen the region, never narrow it",
          r[0] <= without[0] and r[1] <= without[1]
          and r[2] >= without[2] and r[3] >= without[3],
          "%s against %s" % (r, without))

    # A footprint's bounds are ALREADY on the lattice. Rounding them a second
    # time would hand back a spare cell on every side, every time.
    box = G.footprint_from_region((0.806, 0.026, 0.812, 0.032))
    bb = G.footprint_bounds(box)
    inside = np.array([[0.8065, 0.0265, 0.0]])
    rb = G.selection_region(inside, np.ones(1, dtype=bool), footprint=box)
    check("a cell-aligned footprint is not rounded a second time",
          all(abs(a - b) < 1e-12 for a, b in zip(rb, bb)),
          "%s against %s" % (rb, bb))

    imin, jmin, imax, jmax = G.footprint_cell_range(fp)
    check("footprint_cell_range describes the same rectangle as "
          "footprint_bounds",
          abs(imin*fp["cell"] - fb[0]) < 1e-12
          and abs((imax + 1)*fp["cell"] - fb[2]) < 1e-12
          and abs(jmin*fp["cell"] - fb[1]) < 1e-12
          and abs((jmax + 1)*fp["cell"] - fb[3]) < 1e-12)


@_checked
def test_selection_region_refusals():
    print("=== what is refused while the operator can still fix it ===")
    v = np.array([[0.8070, 0.0270, 0.9420], [0.8070, 0.0870, 0.9420]])
    m = np.ones(2, dtype=bool)

    # THE FLOOR IS IN LINE SPACINGS, NOT IN CELLS. One 2 mm cell is six passes
    # at the eddy-current 0.3 mm and one pass at 1.5 mm, and only the second is
    # a region not worth scanning.
    check("at the 0.3 mm eddy-current spacing one cell is six passes, so a "
          "drawn selection is never refused for width",
          int(G.SELECTION_LINE_SPACING_M*1e6) == 300
          and G.selection_region(v, m)[2] > 0.0)

    raises(lambda: G.selection_region(v, m, line_spacing=0.0015),
           "a region too narrow for two passes is refused, naming the size, "
           "the spacing and how many lines that is",
           ("2.0 mm across on x", "line spacing is 1.5 mm", "this is 1 line"))
    try:
        G.selection_region(v, m, line_spacing=0.0015)
        check("the refusal is SelectionRefused", False, "did not raise")
    except G.SelectionRefused as exc:
        check("the refusal is SelectionRefused, and still a ValueError",
              isinstance(exc, ValueError), "%s" % (type(exc).__name__,))
    check("line_spacing=0 turns the floor off for a caller that has no scan "
          "to size it against",
          G.selection_region(v, m, line_spacing=0.0)[2] > 0.0)

    # THE TAUGHT WORK AREA IS THE CEILING. Rounding outward grows commanded
    # motion, so the result is held inside what was taught.
    area = (0.8065, 0.0300, 0.9000, 0.0800)
    rc = G.selection_region(v, m, work_area=area)
    check("the region is clipped to the taught work area, never grown past it",
          rc[0] >= area[0] - 1e-12 and rc[1] >= area[1] - 1e-12
          and rc[2] <= area[2] + 1e-12 and rc[3] <= area[3] + 1e-12,
          "%s against %s" % (rc, area))
    check("the clip is the only thing that takes it off the lattice, and it "
          "still has extent",
          rc[2] - rc[0] > 0.0 and rc[3] - rc[1] > 0.0, "%s" % (rc,))
    check("without a work area nothing is clipped",
          G.selection_region(v, m)[3] > area[3])

    far = np.array([[2.0, 2.0, 0.0], [2.01, 2.01, 0.0]])
    raises(lambda: G.selection_region(far, np.ones(2, dtype=bool),
                                      work_area=area),
           "a selection outside the work area is refused, naming both boxes",
           ("outside the taught work area", "the work area is"))

    check("describe_selection_region says what was reported and what was drawn",
          "rounded out to whole" in G.describe_selection_region(
              rc, G.bounding_region(v, m)))


@_checked
def test_submesh():
    print("=== the sub-mesh handed downstream ===")
    # Two squares sharing nothing: the "part" and the "bench".
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                  [5., 5., 0.], [6., 5., 0.], [6., 6., 0.], [5., 6., 0.]])
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    m = np.array([True]*4 + [False]*4)

    v2, t2, n2, remap = G.submesh(v, t, m)
    check("only the selected square survives", len(v2) == 4 and len(t2) == 2,
          "%d verts %d tris" % (len(v2), len(t2)))
    check("triangles are re-indexed from zero",
          int(t2.min()) == 0 and int(t2.max()) == 3, "%s" % (t2.tolist(),))
    check("the re-indexed mesh describes the same geometry",
          bool(np.allclose(np.sort(v2, axis=0), np.sort(v[:4], axis=0))))

    # A half-selected triangle: "all" keeps it whole, "strict" drops it.
    # Vertex 1 alone is in exactly one triangle, [0, 1, 2], so "all" must give
    # back that one triangle and BOTH its unselected corners.
    half = np.array([False, True, False, False] + [False]*4)
    va, ta, _, _ = G.submesh(v, t, half, keep="all")
    check("keep='all' carries unselected corners along so the edge is not ragged",
          len(ta) == 1 and len(va) == 3, "%d tris %d verts" % (len(ta), len(va)))
    raises(lambda: G.submesh(v, t, half, keep="strict"),
           "keep='strict' drops a triangle that is only partly selected",
           ("touches no triangle",))

    pv, pt, _, _ = G.submesh(v, None, m)
    check("a point cloud sub-selection has points and no faces",
          len(pv) == 4 and pt is None)
    raises(lambda: G.submesh(v, t, np.zeros(8, bool)),
           "an empty selection has no sub-mesh", ("nothing is selected",))
    raises(lambda: G.submesh(v, t, np.zeros(3, bool)),
           "a wrong-length mask is refused", ("3 entries for 8",))


@_checked
def test_footprint():
    print("=== the footprint: the part of a selection that travels ===")
    # A 20 x 20 mm patch of points at 1 mm, at a plausible cell height.
    a = np.arange(0.0, 0.0201, 0.001)
    gx, gy = np.meshgrid(a + 0.800, a + 0.050, indexing="ij")
    v = np.column_stack([gx.ravel(), gy.ravel(),
                         np.full(gx.size, 0.9400)])
    fp = G.footprint(v, dilate=0, fill=False)
    check("a 20 mm patch at a 2 mm cell is 11 x 11 cells",
          fp["grid"].shape == (11, 11) and fp["grid"].all(),
          "%s" % (fp["grid"].shape,))
    x0, y0, x1, y1 = G.footprint_bounds(fp)
    check("its bounds are the cells it occupies, not its points",
          abs(x0 - 0.800) < 1e-9 and abs(x1 - 0.822) < 1e-9,
          "%.4f..%.4f" % (x0, x1))
    check("area is cells x cell^2", abs(G.footprint_area_m2(fp) - 121*4e-6) < 1e-12)

    # Dilation grows OUTWARD and does not clip at the edge of the array.
    d = G.footprint(v, dilate=1, fill=False)
    check("one cell of dilation grows the grid by one cell on every side",
          d["grid"].shape == (13, 13) and d["grid"].all(),
          "%s" % (d["grid"].shape,))
    check("...and the anchor moves with it, so the world position is unchanged",
          d["i0"] == fp["i0"] - 1 and d["j0"] == fp["j0"] - 1)

    # THE REGRESSION THIS EXISTS FOR. The Azure rough mesh has a 6 mm median
    # edge; rasterising its VERTICES onto a 2 mm grid gives a lattice of dots,
    # and a crop built from that drops stripes of the finer mesh. Rasterising its
    # TRIANGLES does not.
    c = np.arange(0.0, 0.0361, 0.006)
    cx, cy = np.meshgrid(c + 0.800, c + 0.050, indexing="ij")
    cv = np.column_stack([cx.ravel(), cy.ravel(), np.full(cx.size, 0.94)])
    n = len(c)
    idx = np.arange(n*n).reshape(n, n)
    ct = np.vstack([
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                         idx[1:, 1:].ravel()]),
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(),
                         idx[:-1, 1:].ravel()])])
    pts_only = G.footprint(cv, dilate=0, fill=False)
    with_tris = G.footprint(cv, tris=ct, dilate=0)
    check("a 6 mm mesh's VERTICES fill only a third of their own box at 2 mm",
          pts_only["grid"].mean() < 0.4,
          "%.0f%% of %s" % (100*pts_only["grid"].mean(), pts_only["grid"].shape))
    check("...and its TRIANGLES fill all of it, which is what a crop needs",
          with_tris["grid"].all(),
          "%.0f%%" % (100*with_tris["grid"].mean(),))
    # Sample the gaps: this is the stripe a vertex-only footprint would drop.
    mid = np.array([[0.803, 0.053], [0.809, 0.059], [0.815, 0.065]])
    check("a point between two vertices is IN the triangle footprint",
          bool(G.footprint_mask(with_tris, mid).all()))
    check("...and OUT of the vertex-only one",
          not bool(G.footprint_mask(pts_only, mid).any()))

    # Packed and back, and JSON-safe -- it has to survive result.json.
    import json
    enc = G.footprint_encode(with_tris)
    check("the packed form is JSON-safe", json.loads(json.dumps(enc)) == enc)
    back = G.footprint_decode(json.loads(json.dumps(enc)))
    check("it round-trips exactly",
          bool((back["grid"] == with_tris["grid"]).all())
          and back["i0"] == with_tris["i0"] and back["j0"] == with_tris["j0"])
    check("decode is idempotent", G.footprint_decode(back) is back)
    check("the height map round-trips to within its own quantum",
          np.nanmax(np.abs(back["heights"] - with_tris["heights"]))
          <= G.FOOTPRINT_Z_QUANTUM_M/2 + 1e-12,
          "%.2e m" % (np.nanmax(np.abs(back["heights"]
                                      - with_tris["heights"])),))
    # THE HEIGHT MAP IS THE BULK OF THE PACKED FORM and that is the price of
    # putting z back in the decision: 361 cells of shape pack into 284 characters
    # of bit rows, and 361 int16 heights into another 1782. Still a couple of
    # kilobytes in result.json for a real selection, against 86 MB of scan data
    # beside it.
    shape_only = {k: v for k, v in enc.items()
                  if k not in ("zrows", "z0", "z_quantum", "z_lo", "z_hi",
                               "z_cells", "height_source")}
    check("a 36 mm selection's SHAPE still packs into a few hundred characters",
          len(json.dumps(shape_only)) < 900,
          "%d" % (len(json.dumps(shape_only)),))
    check("...and its shape and heights together into a few thousand",
          len(json.dumps(enc)) < 4000, "%d" % (len(json.dumps(enc)),))

    # A RECTANGLE'S FOOTPRINT IS THAT RECTANGLE, to the work cell. It is built
    # directly rather than by handing footprint() two corners: two points make
    # the point-spacing fill measure the DIAGONAL as the spacing and grow the
    # region by half of it, which turned an 18 x 60 mm region into 80 x 122 mm.
    r = G.footprint_from_region((0.819, 0.033, 0.837, 0.093))
    rx0, ry0, rx1, ry1 = G.footprint_bounds(r)
    check("an 18 x 60 mm region gives a 20 x 62 mm footprint (the cell it "
          "rounds out to), not more",
          r["grid"].shape == (10, 31) and r["grid"].all()
          and abs((rx1 - rx0)*1e3 - 20) < 0.01 and abs((ry1 - ry0)*1e3 - 62) < 0.01,
          "%s, %.1f x %.1f mm" % (r["grid"].shape, (rx1 - rx0)*1e3, (ry1 - ry0)*1e3))
    check("...and it contains the region's own corners",
          bool(G.footprint_mask(r, np.array([[0.819, 0.033], [0.837, 0.093],
                                             [0.828, 0.060]])).all()))
    check("...and not a point 10 mm outside it",
          not bool(G.footprint_mask(r, np.array([[0.850, 0.060]]))[0]))
    check("footprint_grown() adds whole cells, outward",
          G.footprint_grown(r, 0.005)["grid"].shape == (16, 37),
          "%s" % (G.footprint_grown(r, 0.005)["grid"].shape,))

    # AN L-SHAPE: the thing a region tuple cannot express. The bench in the
    # corner of a rectangle round an L-shaped part is the reason ROIs exist.
    keep = ~((cv[:, 0] > 0.818) & (cv[:, 1] > 0.068))
    ell = G.footprint(cv, keep, tris=ct, dilate=0)
    rect = G.footprint_from_region(G.footprint_bounds(ell))
    corner = np.array([[0.830, 0.080]])
    check("the notch of an L is OUT of its footprint",
          not bool(G.footprint_mask(ell, corner)[0]))
    check("...and IN its bounding rectangle, which is the whole difference",
          bool(G.footprint_mask(rect, corner)[0]))
    check("the L covers less area than its own rectangle",
          G.footprint_area_m2(ell) < 0.85*G.footprint_area_m2(rect),
          "%.1f vs %.1f cm2" % (G.footprint_area_m2(ell)*1e4,
                                G.footprint_area_m2(rect)*1e4))

    # keep="strict" is the default HERE and "all" is submesh()'s, deliberately:
    # a footprint must not include the rim where a plate meets the bench.
    s = G.footprint(cv, keep, tris=ct, dilate=0, keep="strict")
    b = G.footprint(cv, keep, tris=ct, dilate=0, keep="all")
    check("keep='strict' gives a smaller footprint than keep='all'",
          G.footprint_area_m2(s) < G.footprint_area_m2(b),
          "%.1f vs %.1f cm2" % (G.footprint_area_m2(s)*1e4,
                                G.footprint_area_m2(b)*1e4))
    one = np.zeros(len(cv), bool)
    one[0] = True
    check("a selection thinner than one triangle falls back rather than raising",
          G.footprint(cv, one, tris=ct, dilate=0)["grid"].any())

    # A POINT CLOUD is filled by its own mean spacing, for the same reason.
    sparse = cv[::7]
    f = G.footprint(sparse, dilate=0)
    check("a sparse cloud is grown by its own spacing, not left as dots",
          f["dilated"] >= 1 and f["grid"].mean() > 0.5,
          "dilated %d, %.0f%% full" % (f["dilated"], 100*f["grid"].mean()))

    # z IS NOT IN THE FOOTPRINT'S x-y TEST AT ALL: the same x-y with 30 mm of z
    # difference between two sensors still matches.
    high = np.column_stack([mid, np.full(len(mid), 0.9435)])
    low = np.column_stack([mid, np.full(len(mid), 0.9127)])
    check("a footprint matches the same x-y at any z -- the Azure reads 12 mm "
          "long and it must not matter",
          bool((G.footprint_mask(with_tris, high) ==
                G.footprint_mask(with_tris, low)).all()))

    # A NaN vertex must not ask for a grid the size of the integers. The
    # reconstruction's own smoothing does emit a handful of NaN vertices per
    # mesh, so this is not hypothetical.
    nanv = cv.copy()
    nanv[0, 0] = np.nan
    check("a NaN vertex is dropped rather than sizing the grid",
          G.footprint(nanv, tris=ct, dilate=0)["grid"].shape
          == with_tris["grid"].shape,
          "%s" % (G.footprint(nanv, tris=ct, dilate=0)["grid"].shape,))
    check("...and it matches nothing",
          not bool(G.footprint_mask(with_tris,
                                    np.array([[np.nan, np.nan]]))[0]))

    raises(lambda: G.footprint(v, np.zeros(len(v), bool)),
           "an empty selection has no footprint", ("nothing is selected",))
    raises(lambda: G.footprint(v, cell=0.0),
           "a zero cell is refused", ("must be positive",))
    raises(lambda: G.footprint(v, keep="most"),
           "an unknown keep is refused", ("'all' or 'strict'",))


def _arc_sheet(x, y, zf):
    """A triangulated height field over the x cross y lattice. -> (verts, tris)."""
    gx, gy = np.meshgrid(x, y, indexing="ij")
    v = np.column_stack([gx.ravel(), gy.ravel(), zf(gx, gy).ravel()])
    nx, ny = len(x), len(y)
    idx = np.arange(nx*ny).reshape(nx, ny)
    t = np.vstack([
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                         idx[1:, 1:].ravel()]),
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(),
                         idx[:-1, 1:].ravel()])])
    return v, t


@_checked
def test_footprint_heights():
    print("=== the height map: z back in the decision, per cell ===")
    # A FLAT PLATE, the common case. The band over a constant height map IS a
    # slab, which is the wanted behaviour -- the point is that it is a slab
    # because the surface is flat, not by construction.
    a = np.arange(0.0, 0.0201, 0.001)
    gx, gy = np.meshgrid(a + 0.800, a + 0.050, indexing="ij")
    flat = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.9400)])
    fp = G.footprint(flat, dilate=0, fill=False)
    h = fp["heights"]
    check("a flat selection's height map is that height everywhere",
          bool(np.isfinite(h[fp["grid"]]).all())
          and float(np.nanmax(np.abs(h[fp["grid"]] - 0.9400))) < 1e-9,
          "%.4f..%.4f" % (np.nanmin(h), np.nanmax(h)))
    check("every cell of the grid has a height -- no holes to fall through",
          not bool((fp["grid"] & ~np.isfinite(h)).any()))
    pts = np.array([[0.810, 0.060, 0.9400],      # on the surface
                    [0.810, 0.060, 0.9445],      # 4.5 mm above: inside 6 mm
                    [0.810, 0.060, 0.9330],      # 7.0 mm below: outside
                    [0.810, 0.060, 0.9218],      # the bench, 18 mm down
                    [0.900, 0.060, 0.9400]])     # right height, wrong place
    band, nh = G.footprint_band_mask(fp, pts, 0.006)
    check("the band takes the surface and 4.5 mm above it",
          bool(band[0]) and bool(band[1]))
    check("...and refuses 7 mm below it, and the bench 18 mm below it",
          not bool(band[2]) and not bool(band[3]))
    check("...and x-y still decides first: the right height 90 mm away is out",
          not bool(band[4]))
    check("nothing was left unjudged", not bool(nh.any()))

    # A CURVED PART, which is what these specimens are becoming. The crown of a
    # cylinder 60 mm in radius over a 100 mm chord: 26.8 mm of relief, its edges
    # sitting on a bench. THE WHOLE POINT: a global slab wide enough to hold the
    # crown holds the bench too, and a per-cell band does not.
    BENCH, R, W, XC = 0.9200, 0.060, 0.050, 0.850

    def crown(px, py):
        d = np.clip(np.abs(px - XC), 0.0, W)
        return BENCH + np.sqrt(R*R - d*d) - np.sqrt(R*R - W*W)

    cx = np.arange(XC - W, XC + W + 1e-9, 0.006)
    cy = np.arange(0.0, 0.1201, 0.006)
    cv, ct = _arc_sheet(cx, cy, crown)
    arc = G.footprint(cv, tris=ct, dilate=0)
    ah = arc["heights"]
    relief = (np.nanmax(ah) - np.nanmin(ah))*1e3
    check("the crown's height map has 26.8 mm of relief, not one height",
          abs(relief - 26.8) < 1.0, "%.1f mm" % (relief,))
    check("...and it FOLLOWS the curve: the middle cell is the tallest",
          float(ah[ah.shape[0]//2, ah.shape[1]//2]) > float(np.nanmin(ah)) + 0.020,
          "middle %.4f against min %.4f"
          % (ah[ah.shape[0]//2, ah.shape[1]//2], np.nanmin(ah)))

    # Sample the bench under the crown and ask both questions.
    bx, by = np.meshgrid(np.arange(XC - W, XC + W + 1e-9, 0.002),
                         np.arange(0.0, 0.1201, 0.002), indexing="ij")
    bench = np.column_stack([bx.ravel(), by.ravel(),
                             np.full(bx.size, BENCH)])
    TOL = 0.006
    inxy = G.footprint_mask(arc, bench)
    slab = (inxy & (bench[:, 2] >= np.nanmin(ah) - TOL)
            & (bench[:, 2] <= np.nanmax(ah) + TOL))
    band, _nh = G.footprint_band_mask(arc, bench, TOL)
    check("a GLOBAL slab of the same tolerance accepts the bench under the "
          "crown -- the case a curved specimen presents",
          int(slab.sum()) == int(inxy.sum()) and int(inxy.sum()) > 1000,
          "%d of %d bench samples" % (int(slab.sum()), int(inxy.sum())))
    check("...and the PER-CELL band refuses almost all of it",
          int(band.sum()) < 0.3*int(inxy.sum()),
          "%d of %d (%.0f%%)" % (int(band.sum()), int(inxy.sum()),
                                 100.0*band.sum()/max(int(inxy.sum()), 1)))
    # What it does keep is where the arc genuinely comes down to the bench.
    if band.any():
        check("...keeping only where the arc really is within the tolerance of "
              "the bench, at its edges",
              float(np.abs(bench[band][:, 0] - XC).min()) > 0.8*W,
              "nearest kept bench sample is %.1f mm from the crown's centre"
              % (np.abs(bench[band][:, 0] - XC).min()*1e3,))
    crownpts = np.column_stack([bx.ravel(), by.ravel(),
                               crown(bx, by).ravel()])
    kept, _ = G.footprint_band_mask(arc, crownpts, TOL)
    inxy_c = G.footprint_mask(arc, crownpts)
    check("and it keeps ALL of the curved surface it was drawn on",
          int(kept.sum()) == int(inxy_c.sum()) and int(kept.sum()) > 1000,
          "%d of %d" % (int(kept.sum()), int(inxy_c.sum())))

    # A HOLE IN THE SELECTION still gets a height, from the cells around it --
    # otherwise a bolt hole is a hole in the z protection, and a face over one is
    # judged on x-y alone. The dilation is what closes a hole this small, which is
    # exactly how a cell ends up in the grid with no sample of its own.
    holed = ~((np.abs(cv[:, 0] - XC) < 0.004) & (np.abs(cv[:, 1] - 0.060) < 0.004))
    hf = G.footprint(cv, holed, tris=ct, dilate=2)
    mid = np.array([[XC, 0.060]])
    inhole = bool(G.footprint_mask(hf, mid)[0])
    check("the dilation closes a small hole in the selection", inhole)
    if inhole:
        hz = float(G.footprint_heights_at(hf, mid)[0])
        want = float(crown(np.array([XC]), np.array([0.060]))[0])
        check("a cell inside that hole still has a height, and it is the "
              "surrounding surface's",
              np.isfinite(hz) and abs(hz - want) < 0.002,
              "%.4f against the true %.4f" % (hz, want))
    check("every cell of a holed selection's grid has a height",
          not bool((hf["grid"] & ~np.isfinite(hf["heights"])).any()),
          "%d without one"
          % (int((hf["grid"] & ~np.isfinite(hf["heights"])).sum()),))

    # THE DILATION RING AND THE PLANNING MARGIN. Growing a footprint has to grow
    # its height map, or the margin Noether's passes run into has no z at all.
    grown = G.footprint_grown(fp, 0.006)
    check("growing a footprint gives the new ring a height too",
          not bool((grown["grid"] & ~np.isfinite(grown["heights"])).any()),
          "%d cells without one"
          % (int((grown["grid"] & ~np.isfinite(grown["heights"])).sum()),))
    check("...and it is the neighbouring surface's height, not zero and not NaN",
          float(np.nanmax(np.abs(grown["heights"][grown["grid"]] - 0.9400)))
          < 1e-9)

    # A RECTANGLE SAYS NOTHING ABOUT z, and must not pretend to.
    r = G.footprint_from_region((0.819, 0.033, 0.837, 0.093))
    check("a rectangle's footprint carries NO height map",
          not G.footprint_has_heights(r))
    rb, rnh = G.footprint_band_mask(
        r, np.array([[0.828, 0.060, 0.9400], [0.828, 0.060, 0.5000]]), 0.006)
    check("...so a band over one applies x-y alone rather than guessing",
          bool(rb.all()) and bool(rnh.all()))
    check("and describe_footprint() says so out loud",
          "NO height map" in G.describe_footprint(r),
          G.describe_footprint(r))
    check("while a drawn selection's description names its z",
          "surface at z 0.9400" in G.describe_footprint(fp),
          G.describe_footprint(fp))

    # A NaN z costs a cell its height, never its place: x-y is what a NaN vertex
    # must not be allowed to move.
    nanz = cv.copy()
    nanz[:, 2] = np.where(np.arange(len(cv)) % 50 == 0, np.nan, nanz[:, 2])
    nf = G.footprint(nanz, tris=ct, dilate=0)
    check("a NaN z does not change the footprint's shape",
          nf["grid"].shape == arc["grid"].shape
          and int(nf["grid"].sum()) == int(arc["grid"].sum()),
          "%s vs %s" % (nf["grid"].shape, arc["grid"].shape))
    check("...and every cell still ends up with a height",
          not bool((nf["grid"] & ~np.isfinite(nf["heights"])).any()))

    raises(lambda: G.footprint_band_mask(fp, np.zeros((3, 2)), 0.006),
           "a band over N x 2 points is refused -- there is no z to test",
           ("N x 3",))


#: The three meshes the ROI crop was measured on, in the fixture repository.
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
    print("=== real data: the ROI whose z excluded the whole Keyence mesh ===")
    files = _r12_files()
    if files is None:
        print("  (skipped: %s)" % (test_data.why_missing(*R12_MEMBERS[0]),))
        return
    sub, r1, r2 = files
    sv, sn, st = G.read_ply(sub)
    box = G.bounding_box3d(sv)
    check("the ROI the operator drew is at z 0.9416..0.9435 on the AZURE mesh",
          abs(box[2] - 0.9416) < 1e-3 and abs(box[5] - 0.9435) < 1e-3,
          "%.4f..%.4f" % (box[2], box[5]))
    fp = G.footprint(sv, tris=st)
    print("  " + G.describe_footprint(fp))
    for name, path, faces, kept in (("run 1", r1, 20327, 5705),
                                    ("run 2", r2, 11896, 5525)):
        v, n, t = G.read_ply(path)
        ctr = v[t].mean(axis=1)
        check("%s tops out at z %.4f, BELOW the ROI's z floor %.4f"
              % (name, v[:, 2].max(), box[2]), v[:, 2].max() < box[2])
        in3d = int(((ctr[:, 0] >= box[0]) & (ctr[:, 0] <= box[3]) &
                    (ctr[:, 1] >= box[1]) & (ctr[:, 1] <= box[4]) &
                    (ctr[:, 2] >= box[2]) & (ctr[:, 2] <= box[5])).sum())
        check("  as a 3D box, the ROI keeps 0 of %d faces" % (faces,),
              in3d == 0 and len(t) == faces, "%d of %d" % (in3d, len(t)))
        got = int(G.footprint_mask(G.footprint_encode(fp), ctr).sum())
        check("  as a drawn x-y footprint it keeps %d (%.0f%% of the mesh)"
              % (kept, 100.0*kept/faces), got == kept, "%d" % (got,))
        sel = G.footprint_mask(G.footprint_encode(fp), v)
        v2, t2, n2, _ = G.submesh(v, t, sel)
        check("  ...and that is a real mesh: %d verts, %d tris"
              % (len(v2), 0 if t2 is None else len(t2)),
              t2 is not None and len(t2) > 1000)

        # AND THE PER-CELL HEIGHT BAND, which is how z reaches the crop. THE
        # NUMBER THAT MATTERS: the two sensors disagree about this height by
        # 0.8-2.6 mm per cell, so a 6 mm band keeps everything the x-y footprint
        # kept. A global z box with the same 6 mm would still keep 0: its floor
        # is the MINIMUM over the whole selection, and this mesh tops out below
        # it by more than that.
        h = G.footprint_heights_at(G.footprint_encode(fp), ctr)
        inxy = np.isfinite(h)
        off = (h[inxy] - ctr[inxy, 2])*1e3
        check("  the ROI's own surface reads %.2f mm above this mesh's per cell "
              "(max %.2f), which is sensor disagreement and not a fault"
              % (np.median(off), np.abs(off).max()),
              0.5 < float(np.median(np.abs(off))) < 3.0
              and float(np.abs(off).max()) < 4.0,
              "median %.2f max %.2f mm" % (np.median(off), np.abs(off).max()))
        band, nh = G.footprint_band_mask(G.footprint_encode(fp), ctr, 0.006)
        check("  ...so the +/-6 mm PER-CELL band keeps all %d of them -- it does "
              "not collapse the way a global box does" % (kept,),
              int(band.sum()) == kept and not bool(nh.any()),
              "%d, %d unjudged" % (int(band.sum()), int(nh.sum())))
        # HONESTY ABOUT WHAT THIS DATA DOES AND DOES NOT PROVE. On a FLAT plate a
        # global slab widened by the same 6 mm recovers the same faces, because
        # the selection's own z range is 1.8 mm tall and a 13.8 mm slab contains
        # the other sensor's surface. That is the right answer for a flat plate
        # and it is NOT an argument for a slab: the moment the part is curved the
        # slab has to be as tall as the relief before it accepts its own edges,
        # and a slab that tall accepts the bench. test_footprint_heights() shows
        # that on a synthetic 26.8 mm crown, because this real data has no curved
        # part in it.
        glo, ghi = box[2] - 0.006, box[5] + 0.006
        wide = int((G.footprint_mask(G.footprint_encode(fp), ctr)
                    & (ctr[:, 2] >= glo) & (ctr[:, 2] <= ghi)).sum())
        check("  ...and on THIS flat plate a global slab widened by 6 mm would "
              "keep the same %d -- the per-cell band earns its keep on curvature, "
              "which this data does not have" % (wide,),
              wide == kept, "%d against %d" % (wide, kept))


@_checked
def test_what_the_window_writes():
    print("=== what the window writes into result.json ===")
    # roi_select_ui.py cannot be imported here -- it pulls in PyQt5 and VTK at
    # module scope, which is the point of the split. Its write_result() is pure
    # numpy and json, so it is lifted out by AST and run against a stub window.
    # What this pins is that the FOOTPRINT actually reaches the file: without it
    # every crop downstream falls back to a rectangle.
    import ast
    import json
    import tempfile
    src = open(os.path.join(HERE, "roi_select_ui.py")).read()
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "write_result"]
    check("write_result() is there to lift out", len(fn) == 1)
    if len(fn) != 1:
        return
    ns = {"G": G, "np": np, "json": json, "sys": sys, "__builtins__": __builtins__}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "roi_select_ui.py",
                 "exec"), ns)

    # A 6 mm-pitch surface, as the Azure rough mesh is, with the middle selected.
    v, t = _sampled_surface()
    mask = ((v[:, 0] >= 0.806) & (v[:, 0] <= 0.836) &
            (v[:, 1] >= 0.056) & (v[:, 1] <= 0.086))

    class _Win(object):
        pass
    win = _Win()
    win.verts, win.tris, win.normals, win.mask = v, t, None, mask
    win.n, win.front_only = len(v), True
    win.work_area, win.line_spacing = (0.60, -0.20, 1.00, 0.30), 0.0003
    d = tempfile.mkdtemp(prefix="roi_ui_test_")
    req = {"input": "stub.ply", "output": os.path.join(d, "result.json"),
           "mask": os.path.join(d, "mask.npy"),
           "submesh": os.path.join(d, "sub.ply")}
    out = ns["write_result"](win, req)

    with open(req["output"]) as fh:
        onfile = json.load(fh)
    check("the result is JSON on disk, not just a dict in memory",
          onfile["n_selected"] == int(mask.sum()))
    check("it carries a footprint", onfile.get("footprint") is not None
          and onfile.get("footprint_error") is None,
          "%s" % (out.get("footprint_error"),))
    fp = G.footprint_decode(onfile["footprint"])
    x0, y0, x1, y1 = G.footprint_bounds(fp)
    sx0, sy0, sx1, sy1 = G.bounding_region(v, mask)
    check("the footprint contains everything that was selected, and overhangs "
          "it by at most the cell and the dilation",
          x0 <= sx0 and y0 <= sy0 and x1 >= sx1 and y1 >= sy1
          and (sx0 - x0) <= 0.005 and (x1 - sx1) <= 0.005,
          "footprint %.3f..%.3f against selection %.3f..%.3f"
          % (x0, x1, sx0, sx1))
    check("it is SOLID at the 2 mm cell even though the mesh is 6 mm",
          G.footprint_area_m2(fp) > 0.9*(x1 - x0)*(y1 - y0),
          "%.0f%%" % (100*G.footprint_area_m2(fp)/((x1 - x0)*(y1 - y0)),))
    check("bounds3d is still written, and flagged as provenance",
          onfile.get("bounds3d_is_provenance") is True
          and abs(onfile["bounds3d"][2] - 0.9420) < 1e-9)
    check("the mask is saved separately, at full length",
          len(np.load(req["mask"])) == len(v))
    check("and the sub-mesh is a mesh", os.path.exists(req["submesh"])
          and G.read_ply(req["submesh"])[2] is not None)

    # THE REGION IN THE FILE IS THE ROUNDED, CLIPPED ONE, and the raw span of the
    # selected vertices is beside it as provenance.
    reg = onfile["region"]
    drawn = onfile["region_drawn"]
    check("the region written out has extent on both axes",
          reg[2] - reg[0] > 0.0 and reg[3] - reg[1] > 0.0, "%s" % (reg,))
    check("it contains the raw span of the selected vertices",
          reg[0] <= drawn[0] and reg[1] <= drawn[1]
          and reg[2] >= drawn[2] and reg[3] >= drawn[3],
          "%s against %s" % (reg, drawn))
    check("it contains the footprint it will be planned on",
          reg[0] <= x0 + 1e-12 and reg[1] <= y0 + 1e-12
          and reg[2] >= x1 - 1e-12 and reg[3] >= y1 - 1e-12,
          "%s against %s" % (reg, [x0, y0, x1, y1]))
    check("and a line of it goes in the file for the scan log",
          "rounded out to whole" in onfile["region_described"])

    # THE SELECTION THAT KILLED A RUN, through the whole of write_result: a band
    # on a 6 mm surface that catches one column of vertices.
    win.mask = np.isclose(v[:, 0], 0.806) | np.isclose(v[:, 0], 0.8061)
    req3 = dict(req, output=os.path.join(d, "r3.json"),
                mask=os.path.join(d, "m3.npy"),
                submesh=os.path.join(d, "s3.ply"))
    out3 = ns["write_result"](win, req3)
    check("a one-column selection that used to report zero extent reports a cell",
          out3["region_drawn"][2] - out3["region_drawn"][0] == 0.0
          and out3["region"][2] - out3["region"][0] >= G.FOOTPRINT_CELL_M - 1e-12,
          "drawn %s, reported %s" % (out3["region_drawn"], out3["region"]))
    win.mask = mask

    # A POINT CLOUD has no faces, and must still produce a footprint.
    win.tris = None
    req2 = dict(req, output=os.path.join(d, "r2.json"),
                mask=os.path.join(d, "m2.npy"),
                submesh=os.path.join(d, "s2.ply"))
    out2 = ns["write_result"](win, req2)
    check("a point-cloud selection also carries a footprint",
          out2.get("footprint") is not None and out2.get("footprint_error") is None,
          "%s" % (out2.get("footprint_error"),))


@_checked
def test_projection():
    print("=== projection: world -> screen ===")
    # A plain orthographic matrix over x in -1..1, y in -1..1, z in -1..1,
    # written out by hand so this test needs no VTK.
    m = np.eye(4)
    xy, d = G.project(np.array([[-1., -1., 0.], [1., 1., 0.], [0., 0., -1.]]),
                      m, 100, 200)
    check("the lower-left corner maps to (0, 0)",
          bool(np.allclose(xy[0], [0.0, 0.0])), "%s" % (xy[0],))
    check("the upper-right corner maps to (width, height)",
          bool(np.allclose(xy[1], [100.0, 200.0])), "%s" % (xy[1],))
    check("the centre maps to the middle", bool(np.allclose(xy[2], [50.0, 100.0])))
    check("smaller depth means nearer", float(d[2]) < float(d[0]))

    # Selection through the real chain: project, then draw a box on screen.
    v = np.column_stack([np.linspace(-1, 1, 21), np.zeros(21), np.zeros(21)])
    sxy, _ = G.project(v, m, 100, 100)
    sel = G.mask_from_box(sxy, 45.0, 0.0, 100.0, 100.0)
    # screen x 45 is world x -0.1, so the 12 samples from -0.1 to 1.0 inclusive.
    check("a screen box selects the world points under it",
          int(sel.sum()) == 12 and abs(float(v[sel][:, 0].min()) + 0.1) < 1e-9,
          "%d selected, from x %.4f" % (int(sel.sum()), v[sel][:, 0].min()))


@_checked
def test_ply_roundtrip():
    print("=== PLY in and out ===")
    import tempfile
    v = np.array([[0., 0., 0.9], [1., 0., 0.9], [1., 1., 0.9], [0., 1., 0.91]])
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    t = np.array([[0, 1, 2], [0, 2, 3]])
    d = tempfile.mkdtemp(prefix="roi_select_test_")

    f = G.write_ply(os.path.join(d, "mesh.ply"), v, t, n)
    rv, rn, rt = G.read_ply(f)
    check("a mesh round-trips its vertices", bool(np.allclose(rv, v)))
    check("a mesh round-trips its triangles", bool((rt == t).all()))
    check("a mesh round-trips its normals",
          rn is not None and bool(np.allclose(rn, n)))

    f = G.write_ply(os.path.join(d, "points.ply"), v)
    rv, rn, rt = G.read_ply(f)
    check("a point cloud round-trips", bool(np.allclose(rv, v)))
    check("a point cloud has no faces and says so with None", rt is None)
    check("a point cloud PLY has no face element at all",
          b"element face" not in open(f, "rb").read(400))

    # noether_path.read_ply_mesh() must accept what we write for a mesh. Its
    # header expectations are narrow, so check the bytes rather than trusting.
    head = open(os.path.join(d, "mesh.ply"), "rb").read(400)
    check("the header is binary_little_endian with a uchar/int face list",
          b"binary_little_endian" in head and
          b"property list uchar int vertex_indices" in head)


@_checked
def test_real_mesh():
    print("=== a real Keyence reconstruction ===")
    if REAL_MESH is None:
        print("  (skipped: %s)"
              % (test_data.why_missing("keyence", "roi_spray"),))
        return

    v, n, t = G.read_ply(REAL_MESH)
    check("the ecplan mesh loads as a mesh",
          t is not None and len(t) > 100000, "%d verts, %s tris"
          % (len(v), None if t is None else len(t)))

    whole = G.bounding_region(v)
    check("the whole reconstruction is the 163 x 319 mm the YOLO box asked for",
          abs((whole[2] - whole[0])*1e3 - 163) < 3 and
          abs((whole[3] - whole[1])*1e3 - 319) < 3,
          "%.0f x %.0f mm" % ((whole[2] - whole[0])*1e3, (whole[3] - whole[1])*1e3))

    plate = G.region_mask(v, PLATE)
    reg = G.bounding_region(v, plate)
    check("a rectangle over the plate gives back an 83 x 165 mm region",
          abs((reg[2] - reg[0])*1e3 - 83) < 4 and
          abs((reg[3] - reg[1])*1e3 - 165) < 4,
          "%.0f x %.0f mm" % ((reg[2] - reg[0])*1e3, (reg[3] - reg[1])*1e3))

    frac = float(plate.mean())
    check("the plate is about 30% of the vertices -- the rest is bench",
          0.25 < frac < 0.35, "%.1f%%" % (100*frac,))

    sv, st, sn, _ = G.submesh(v, t, plate)
    check("the sub-mesh is a real mesh, not a fragment",
          st is not None and len(st) > 10000, "%s tris" % (None if st is None else len(st),))
    check("it is a large fraction smaller than what was rastered",
          len(st) < 0.45*len(t),
          "%d of %d triangles (%.0f%%)" % (len(st), len(t), 100.0*len(st)/len(t)))
    check("every triangle index is in range",
          int(st.min()) >= 0 and int(st.max()) < len(sv))

    # The area ratio is the toolpath ratio, near enough: the raster is planned
    # over the surface, so a third of the footprint is about a third of the
    # 412 374 poses the uncropped box produces.
    area_all = (whole[2] - whole[0])*(whole[3] - whole[1])
    area_sel = (reg[2] - reg[0])*(reg[3] - reg[1])
    print("    footprint %.0f cm2 -> %.0f cm2 (%.0f%%); at 412 374 poses that "
          "is about %.0f thousand" % (area_all*1e4, area_sel*1e4,
                                      100*area_sel/area_all,
                                      412374*area_sel/area_all/1000.0))

    if REAL_POINTS is not None:
        pv, pn, pt = G.read_ply(REAL_POINTS)
        check("the point cloud loads and is recognised as one",
              pt is None and len(pv) > 40000,
              "%d points, %s tris" % (len(pv), pt))
        pm = G.region_mask(pv, PLATE)
        check("the same rectangle selects the same fraction of the points",
              0.25 < float(pm.mean()) < 0.35, "%.1f%%" % (100*pm.mean(),))

    print("  " + G.describe(v, plate))


def main():
    for fn in (test_box, test_polygon, test_brush, test_combine,
               test_front_surface, test_region, test_selection_region,
               test_selection_region_covers_the_drawn_shape,
               test_selection_region_refusals,
               test_submesh, test_footprint,
               test_footprint_heights,
               test_the_roi_that_cropped_nothing, test_what_the_window_writes,
               test_projection, test_ply_roundtrip, test_real_mesh):
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one standalone run reports all of them.
        getattr(fn, "__wrapped__", fn)()
        print("")
    if fails:
        print("%d FAILED: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("all selection geometry checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_roi_select_geometry():
    """So pytest reports the same thing as running it.

    AND ASSERTS. check() records rather than raising, so that a standalone run
    reports every failure in one pass -- but under pytest that alone would be a
    test which passes while printing FAIL, which is worse than a test that does
    not collect at all.

    @_checked asserts per section, so a broken check is reported against the
    section that ran it. This stays as the backstop because it is free and it
    catches what that cannot: a check() run from outside any section.
    """
    mark = len(fails)
    main()
    added = fails[mark:]
    assert not added, "%d check(s) failed: %s" % (len(added), "; ".join(added))
    assert not fails, fails
