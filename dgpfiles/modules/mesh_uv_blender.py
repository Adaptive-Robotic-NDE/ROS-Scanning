"""UV-parameterize a mesh in Blender and write it as X3D for spatialnde2.

Run as a subprocess by mesh_uv_x3d() in noether_path.dpi:

    python3 mesh_uv_blender.py -- <input mesh> <output .x3d> [planar|angle_based]

Not `blender --background --python`: the apt blender binary in this container is
older than the bpy module beside it and has no wm.ply_import. `python3` with the
bpy module (4.0.0) does.

The operator sequence is the one from dgpfiles/x3dConverter.py, which is the
existing eddy-current workflow, with three deliberate differences:

  * the X3D is written here rather than by bpy.ops.export_scene.x3d, because
    that exporter rounds texture coordinates to four decimal places and
    spatialnde2 reindexes texture vertices by exact coordinate. On an 18780
    vertex mesh the rounding welded eight distinct UV vertices, which produced
    UV edges shared by three triangles and x3d_load_geometry refused the file;
  * no remove_doubles by default. The legacy 2 mm merge is right for a raw
    triangulated point cloud and wrong for a mesh that is already a mesh: on
    curved_scan.ply it created twelve non-manifold vertices, and the loader
    asserts when the triangles around a vertex do not form one contiguous fan;
  * normals_make_consistent, because the loader also depends on winding.
"""
import sys

import bpy
import numpy as np

# Pure numpy, no bpy: the same module the projection loader uses to check a
# surface, so the orientation rule is written down in exactly one place.
import x3d_winding


def load_mesh(src):
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    low = src.lower()
    if low.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=src)
    elif low.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=src)
    elif low.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=src)
    else:
        raise SystemExit("unsupported mesh type: %s" % src)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def clean(obj, merge_threshold=0.0):
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    if merge_threshold > 0.0:
        bpy.ops.mesh.remove_doubles(threshold=merge_threshold, use_unselected=True)
    bpy.ops.mesh.delete_loose(use_faces=True)
    bpy.ops.mesh.dissolve_degenerate(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.editmode_toggle()


def unwrap_angle_based(obj):
    """Seams at sharp edges, angle-based unwrap: the x3dConverter.py recipe."""
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.edges_select_sharp()
    bpy.ops.mesh.mark_seam(clear=False)
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.uv.pack_islands(margin=0.01)
    bpy.ops.object.editmode_toggle()


def unwrap_planar(obj):
    """UV is the surface's own footprint on its principal plane.

    One island, no seams, and bijective for anything plate-like -- which a
    scanned specimen surface generally is. It is also the parameterization that
    makes the projected image directly comparable to a rectangular c-scan, the
    same property the synthesised plane patch has in surface_projection.dpi.
    """
    me = obj.data
    co = np.array([v.co[:] for v in me.vertices])
    centre = co.mean(axis=0)
    _, sv, vt = np.linalg.svd(co - centre, full_matrices=False)
    e0, e1 = vt[0], vt[1]

    # THE SVD PINS DOWN THE PLANE BUT NOT THE HANDEDNESS OF THE BASIS IN IT.
    # numpy is free to return -e1, and does, about half the time; the resulting
    # UV map is a perfectly good MIRROR IMAGE and nothing downstream notices
    # until project_onto_parameterization takes the square root of a negative
    # determinant, gets a NaN, and silently deposits zero weight for every ray.
    # Two consecutive scan runs on 2026-10-02 threw away 1876 and 1673 traces
    # into an empty image this way, and a third run of the same morning worked
    # -- the only difference was this sign. Pin it to the surface's own outward
    # direction so the orientation is a property of the part rather than of the
    # decomposition, which also makes the image comparable between runs.
    areas = np.array([poly.area for poly in me.polygons])
    normals = np.array([poly.normal[:] for poly in me.polygons])
    outward = (normals * areas[:, None]).sum(axis=0)
    e0, e1, flipped = x3d_winding.orient_uv_basis(e0, e1, outward)
    print("MESHUV planar handedness %s (|mean normal| %.4f of %.4f m2 of area)"
          % ("FLIPPED to match the surface normal" if flipped
             else "already matched the surface normal",
             float(np.linalg.norm(outward)), float(areas.sum())))

    proj = np.stack([(co-centre) @ e0, (co-centre) @ e1], axis=1)
    lo = proj.min(axis=0)
    span = float((proj.max(axis=0) - lo).max())
    norm = (proj - lo)/span
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uvl = me.uv_layers.active.data
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            uvl[li].uv = (float(norm[vi, 0]), float(norm[vi, 1]))
    print("MESHUV planar basis e0 %s e1 %s, span %.4f m"
          % (np.round(e0, 4), np.round(e1, 4), span))
    # The scale factor matters downstream: UV runs 0..1 over `span` metres, so
    # a projection spot radius in metres and the image pixel pitch relate through
    # this number and nothing else.
    print("MESHUV uv_span_m %.6f" % span)


def _nonmanifold_fan_vertices(tris):
    """Vertices whose incident triangles do not form ONE contiguous fan.

    Two triangles at a vertex are neighbours when they share an edge *through*
    that vertex; the fan is contiguous when that relation connects all of them.
    """
    from collections import defaultdict

    vert_tris = defaultdict(list)
    for ti, t in enumerate(tris):
        for v in t:
            vert_tris[v].append(ti)

    bad = []
    for v, tl in vert_tris.items():
        if len(tl) < 2:
            continue
        by_edge = defaultdict(list)
        for ti in tl:
            t = tris[ti]
            for k in range(3):
                a, b = t[k], t[(k+1) % 3]
                if a == v or b == v:
                    by_edge[(a, b) if a < b else (b, a)].append(ti)
        adj = defaultdict(set)
        for ts in by_edge.values():
            for i in ts:
                for j in ts:
                    if i != j:
                        adj[i].add(j)
        seen = {tl[0]}
        stack = [tl[0]]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(seen) != len(set(tl)):
            bad.append(v)
    return bad


def _undefined_normal_triangles(verts, tris):
    """Triangles with no 3D normal once the vertices are stored as float32.

    snde_normalcalc_triangle() normalizes the two edge vectors leaving a corner
    and crosses them. Three collinear vertices give exactly (0,0,0), and the
    normalizevec3() that follows then divides by zero -- its "edges too
    parallel" fallback recomputes W but never recomputes N from it, so it only
    prints and does not rescue the case. The NaN normal reaches inplanemat and
    then point_in_polygon_2d_c(), where every comparison against a NaN is false
    and geometry_ops.h:381 assert(0) fires. That happens inside
    boxes_calculation_3d on a spatialnde2 WORKER THREAD, so there is no Python
    frame anywhere in the stack: it is a bare SIGABRT, not something a caller
    can catch.

    Checked in float32 on purpose. snde_coord is float unless
    SNDE_DOUBLEPREC_COORDS, and collinearity is a property of the stored
    precision, not of the mesh Blender is holding. On the cal plate's Poisson
    reconstruction SIX triangles are collinear in float32 and only ONE of them
    is collinear in float64 -- so a check done in Blender's own double
    precision, and equally clean()'s dissolve_degenerate(), catches one in six
    and the mesh still aborts.

    Every corner is tried as the apex: collinearity does not depend on which
    corner the loader's winged-edge traversal happens to call vertex 0.
    """
    if len(tris) == 0:
        return []

    v = verts.astype(np.float32)[np.asarray(tris, dtype=np.intp)]  # (ntris, 3, 3)
    bad = np.zeros(len(tris), dtype=bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        for rot in range(3):
            V = v[:, (rot + 1) % 3] - v[:, rot]
            W = v[:, (rot + 2) % 3] - v[:, rot]
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            W = W / np.linalg.norm(W, axis=1, keepdims=True)
            N = np.cross(V, W)
            # ~(mag > 0) rather than (mag == 0), so NaN counts as bad too
            bad |= ~(np.linalg.norm(N, axis=1) > 0.0)

    return np.flatnonzero(bad).tolist()


def _repair_vertex_fans(tris, tri_uv, rounds=6):
    """Drop the triangles around non-contiguous vertex fans until stable.

    Returns (tris, tri_uv, dropped, still_bad).
    """
    dropped = 0
    bad = _nonmanifold_fan_vertices(tris)
    for _ in range(rounds):
        if not bad:
            break
        badset = set(bad)
        keep = [i for i, t in enumerate(tris)
                if not (t[0] in badset or t[1] in badset or t[2] in badset)]
        dropped += len(tris) - len(keep)
        tris = [tris[i] for i in keep]
        tri_uv = [tri_uv[i] for i in keep]
        bad = _nonmanifold_fan_vertices(tris)
    return tris, tri_uv, dropped, len(bad)


def write_x3d(me, path, precision=9):
    me.calc_loop_triangles()
    verts = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    uvl = me.uv_layers.active.data

    uv_index = {}
    uvs = []
    tris = []
    tri_uv = []
    for lt in me.loop_triangles:
        tri = []
        tuv = []
        for vi, li in zip(lt.vertices, lt.loops):
            uv = (float(uvl[li].uv[0]), float(uvl[li].uv[1]))
            key = (round(uv[0], precision), round(uv[1], precision))
            j = uv_index.get(key)
            if j is None:
                j = len(uvs)
                uv_index[key] = j
                uvs.append(uv)
            tri.append(int(vi))
            tuv.append(j)
        tris.append(tri)
        tri_uv.append(tuv)

    # DROP TRIANGLES WHOSE UV IS DEGENERATE, then say plainly whether what is
    # left is manifold. The loader's complaint -- "Edge involving original uv
    # vertices ... is shared by more than two triangles" -- arrives with no clue
    # as to which stage produced it.
    #
    # A triangle whose three loops land on fewer than three distinct UV points
    # has zero area in UV: it can never receive a projected sample, so it
    # carries no data and dropping it loses nothing. It is also the thing that
    # MAKES the layout non-manifold -- a triangle (a, a, b) contributes the edge
    # (a, b) twice on its own, so any real edge it shares with its two proper
    # neighbours is immediately over-subscribed. Measured on the first Keyence
    # reconstruction: 30 degenerate triangles and exactly 30 over-shared edges
    # out of 161148, and removing the 30 left a layout the loader accepts.
    #
    # They come from slivers in the reconstructed surface -- ball pivoting
    # leaves a few -- which survive dissolve_degenerate() in 3D because they are
    # degenerate only once projected.
    from collections import Counter
    keep = [i for i, t in enumerate(tri_uv) if len(set(t)) == 3]
    dropped = len(tri_uv) - len(keep)
    if dropped:
        tris = [tris[i] for i in keep]
        tri_uv = [tri_uv[i] for i in keep]

    # AND DROP TRIANGLES WITH NO 3D NORMAL, which is a THIRD condition again --
    # not uv-degenerate (these have three distinct UV points) and not a fan
    # problem. Their three vertices are distinct but collinear, so the triangle
    # has zero area in 3D and no normal exists. See
    # _undefined_normal_triangles() for the abort path; the short version is
    # that it is the one failure here that lands on a worker thread, where it
    # cannot be caught at all.
    #
    # This is what was still killing the cal plate after the uv-degenerate,
    # fan and orphan repairs below were all in place and all reported clean:
    # 6 collinear triangles out of 150 711. Dropping exactly those 6 and
    # changing nothing else makes the mesh load.
    #
    # A zero-area triangle can never receive a projected sample, so dropping it
    # loses no data -- same argument as the uv-degenerate case above.
    collinear = _undefined_normal_triangles(verts, tris)
    if collinear:
        badset = set(collinear)
        tris = [t for i, t in enumerate(tris) if i not in badset]
        tri_uv = [t for i, t in enumerate(tri_uv) if i not in badset]

    edges = Counter()
    for t in tri_uv:
        for i in range(3):
            a, b = t[i], t[(i+1) % 3]
            edges[(a, b) if a < b else (b, a)] += 1
    over = sum(1 for c in edges.values() if c > 2)

    # AND THE FAN CONDITION, which is a different check and the one that ABORTS.
    # x3d_load_geometry sorts the edges around every vertex into a ring, and
    # asserts if it cannot:
    #     x3d.hpp:2111: assert(edgecheck < vertexnum_edges.second.size())
    #     "Could be that the mesh has multiple holes such that the triangles
    #      touching this vertex are not contiguous."
    # That is assert(), not an exception -- SIGABRT takes the whole dgpy
    # process down, viewer and all, the moment the projection surface loads.
    #
    # The UV checks above do not imply it. Measured on the Poisson mesh of the
    # cal plate: 0 uv-degenerate, 0 over-shared uv edges, 0 over-shared 3D
    # edges -- and TWO vertices out of 76 547, 0.003%, whose triangle fan came
    # in two pieces. That was enough. The test plate's mesh has none, which is
    # why this never appeared until a part with holes was reconstructed.
    #
    # Dropping the triangles around such a vertex can break its neighbours'
    # fans, so repeat until it is stable. It costs a handful of triangles.
    tris, tri_uv, fanfix, fanleft = _repair_vertex_fans(tris, tri_uv)

    # DROPPING TRIANGLES ORPHANS VERTICES, AND AN ORPHAN IS ALSO FATAL. The
    # loader builds its vertex->edges map only from triangles, then looks every
    # vertex up in it with .at(); a vertex no triangle references any more is
    # not in the map and the lookup throws:
    #     terminate called after throwing an instance of 'std::out_of_range'
    #       what():  _Map_base::at
    # which is an uncaught exception in a C++ constructor -- process gone, same
    # as the assert. Repairing the fans above and writing the vertex array
    # unchanged traded one abort for another. So reindex: keep only the
    # vertices some triangle still uses.
    used = sorted({int(i) for t in tris for i in t})
    orphans = len(verts) - len(used)
    if orphans:
        remap = {v: k for k, v in enumerate(used)}
        verts = verts[used]
        tris = [[remap[int(i)] for i in t] for t in tris]

    # AND THE SAME FOR TEXTURE VERTICES, which this pass used to skip. The uv
    # leg of the loader is the same code shape as the 3D leg above and had the
    # same .at() lookup, so an unused TextureCoordinate point aborted just as an
    # unused Coordinate point did -- x3d.hpp:2604 rather than the 3D leg's
    # tolerant operator[]. That is what the cal plate hit AFTER the fan and
    # orphan repairs above were both clean: 76 543 coordinates against 76 545
    # texture coordinates, two of them referenced by nothing.
    #
    # spatialnde2 now tolerates this (the uv leg was changed to operator[] to
    # match the 3D leg, and x3d_validate_geometry() reports it as a non-fatal
    # uv_orphan_vertex), but every triangle dropped above orphans more of them,
    # and an older installed spatialnde2 -- the container's, for one -- still
    # aborts. Reindexing here costs nothing and does not depend on which
    # version is loading the file.
    uv_used = sorted({int(i) for t in tri_uv for i in t})
    uv_orphans = len(uvs) - len(uv_used)
    if uv_orphans:
        uv_remap = {v: k for k, v in enumerate(uv_used)}
        uvs = [uvs[i] for i in uv_used]
        tri_uv = [[uv_remap[int(i)] for i in t] for t in tri_uv]

    print("MESHUV topology: %d tris, %d verts, %d distinct uv, %d uv-degenerate "
          "dropped, %d collinear (no normal) dropped, %d uv edges shared>2, "
          "%d dropped to repair vertex fans, %d orphan verts removed, "
          "%d orphan uv verts removed"
          % (len(tris), len(verts), len(uvs), dropped, len(collinear), over,
             fanfix, orphans, uv_orphans))
    if over or fanleft:
        print("MESHUV WARNING: this layout will be rejected by x3d_load_geometry "
              "(%d over-shared uv edges, %d non-contiguous vertex fans left)"
              % (over, fanleft))

    fmt = "%%.%dg" % precision
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" '
                '"http://www.web3d.org/specifications/x3d-3.3.dtd">\n')
        f.write('<X3D profile="Interchange" version="3.3">\n  <Scene>\n    <Transform>\n')
        f.write('      <Shape>\n')
        f.write('        <IndexedFaceSet solid="false" coordIndex="%s" texCoordIndex="%s">\n'
                % (" ".join("%d %d %d -1" % tuple(t) for t in tris),
                   " ".join("%d %d %d -1" % tuple(t) for t in tri_uv)))
        f.write('          <Coordinate point="%s"/>\n'
                % " ".join(" ".join(fmt % c for c in p) for p in verts))
        f.write('          <TextureCoordinate point="%s"/>\n'
                % " ".join("%s %s" % (fmt % u[0], fmt % u[1]) for u in uvs))
        f.write('        </IndexedFaceSet>\n')
        f.write('        <Appearance><Material diffuseColor="0.8 0.8 0.8"/></Appearance>\n')
        f.write('      </Shape>\n    </Transform>\n  </Scene>\n</X3D>\n')


def main():
    argv = sys.argv[sys.argv.index("--")+1:]
    src, dst = argv[0], argv[1]
    method = argv[2] if len(argv) > 2 else "planar"
    merge = float(argv[3]) if len(argv) > 3 else 0.0

    obj = load_mesh(src)
    print("MESHUV loaded %s: %d verts %d polys"
          % (src, len(obj.data.vertices), len(obj.data.polygons)))
    clean(obj, merge_threshold=merge)

    if method == "angle_based":
        unwrap_angle_based(obj)
    elif method == "planar":
        unwrap_planar(obj)
    else:
        raise SystemExit("method must be 'planar' or 'angle_based', not %r" % method)

    write_x3d(obj.data, dst)
    v = np.array([v.co[:] for v in obj.data.vertices])
    print("MESHUV wrote %s: world extent x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f"
          % (dst, v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
             v[:, 2].min(), v[:, 2].max()))


main()
