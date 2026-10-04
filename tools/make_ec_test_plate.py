#!/usr/bin/env python3
"""Generate a simulated eddy-current test plate: mesh, Gazebo model, flaw list.

A flat aluminium plate carrying notches whose dimensions are EXACTLY on the
response table's nodes, so the simulated probe response needs no interpolation
in flaw geometry and any error we see is the pipeline's rather than the model's.

    python3 make_ec_test_plate.py --out-dir <dir>

THE NOTCHES ARE NOT IN THE MESH, AND THAT IS DELIBERATE.

A 0.15 mm wide notch is below what the Keyence resolves, and the probe is
4.75 mm across -- it bridges the notch rather than dropping into it. Modelling
the notches geometrically would do two wrong things: it would invite the
toolpath planner to follow the tool into a slot the probe never enters, and it
would make the simulated reconstruction carry a feature the real one could not
see. So the plate is smooth and the flaws exist only in flaws.json, which the
eddy-current simulator reads. The geometry and the flaws are deliberately
separate, exactly as they are on a real part.

OUTPUTS

    ec_test_plate.ply     the surface in WORLD coordinates, metres, with
                          vertex normals
    ec_test_plate.stl     the same, for Gazebo
    model.sdf             a Gazebo model that places it in the cell
    flaws.json            every notch in PLATE coordinates, plus where the
                          plate was placed
    README.txt            what this is and how the numbers were chosen

THE MESHES ARE IN WORLD COORDINATES AND THE FLAW LIST IS NOT, and that split is
deliberate. sim_specimen.dpi's place_gazebo_mesh() and write_keyence_sim_scene()
both require world coordinates -- they write a model with an identity pose, so
that nothing has to be kept in step between the Gazebo model and the simulated
profiler's own scene. The flaws, on the other hand, are a property of the PLATE:
a flaw table in world coordinates would be a property of one plate on one bench
and would be wrong the moment it moved. So the manifest describes the plate in
its own frame and records where it was put, and ec_plate_model.py does that one
transform at load.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
from datetime import datetime, timezone

import numpy as np

# Plate, in metres. Sized to sit inside the simulated cell's work volume
# (x 0.600..1.100, y -0.450..0.050, z 0.930..1.250) with room for the probe to
# run off each end of a pass.
PLATE_X = 0.180
PLATE_Y = 0.120
PLATE_Z = 0.012
TOP_CELL = 0.002          # tessellation of the scanned face

# Where the plate sits in the cell. The top face lands at this z. Overridable
# with --place / --yaw-deg, because where the plate sits is a decision about
# this bench and not a property of the plate.
PLACE_XYZ = (0.850, -0.200, 0.950)
PLACE_YAW_DEG = 0.0

#: Notch dimensions ON THE RESPONSE TABLE'S NODES -- see
#: tools/generate_ec_table.py. Lengths are the UniWest 99073 standard's four,
#: so results are directly comparable with the real calibration block.
LENGTHS_M = (0.001016, 0.001524, 0.002032, 0.002540)   # 0.040..0.100 in
DEPTHS_M = (0.00020, 0.00050)
WIDTH_M = 0.00015

#: Row pitch and column pitch. The response field is +/-5 mm, so notches must
#: be far enough apart that one probe position never sees two of them.
COL_X = (-0.0525, -0.0175, 0.0175, 0.0525)
ROW_Y = (0.030, 0.0, -0.030)


def flaws():
    """Every notch, in PLATE coordinates: origin at the plate centre, z=0 at
    the scanned face, x along the plate's long axis."""
    out = []
    # Rows A and B: all four lengths at each of the two depths, square across
    # the scan direction, which is where a differential probe responds most.
    for row, depth in ((0, DEPTHS_M[0]), (1, DEPTHS_M[1])):
        for col, length in enumerate(LENGTHS_M):
            out.append(dict(name="L%d_D%d" % (col, row),
                            x=COL_X[col], y=ROW_Y[row], z=0.0,
                            length_m=length, width_m=WIDTH_M, depth_m=depth,
                            yaw_deg=90.0))
    # Row C: one size at four orientations, to exercise the yaw axis -- a
    # differential probe is nearly blind to a notch aligned with its split.
    for col, yaw in enumerate((0.0, 30.0, 60.0, 90.0)):
        out.append(dict(name="YAW%03d" % int(yaw),
                        x=COL_X[col], y=ROW_Y[2], z=0.0,
                        length_m=LENGTHS_M[2], width_m=WIDTH_M,
                        depth_m=DEPTHS_M[1], yaw_deg=yaw))
    return out


def plate_mesh():
    """A closed box with the TOP face tessellated. -> (verts, normals, tris)"""
    hx, hy = 0.5*PLATE_X, 0.5*PLATE_Y
    nx = max(2, int(round(PLATE_X/TOP_CELL)) + 1)
    ny = max(2, int(round(PLATE_Y/TOP_CELL)) + 1)
    xs = np.linspace(-hx, hx, nx)
    ys = np.linspace(-hy, hy, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    top = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j*nx + i; b = a + 1; c = a + nx; d = c + 1
            # Wound counter-clockwise seen from +z, so the face normal is +z.
            tris.append((a, b, d))
            tris.append((a, d, c))

    # The sides and bottom, so the model is a solid for Gazebo. Coarse on
    # purpose: nothing scans them and every vertex costs the planner.
    #
    # THEY GET THEIR OWN VERTICES rather than sharing the top face's boundary
    # ring. Sharing makes each boundary vertex average a +z normal with a side
    # normal, which leans it 45 degrees -- so the scanned face would present a
    # violent tilt along all four edges and the toolpath would dutifully follow
    # it. That is precisely the commanded-tilt artefact this plate exists to
    # study, and manufacturing it here would be self-defeating. Duplicated, the
    # scanned face is exactly +z everywhere and the seam is a boundary edge,
    # which the loader allows.
    base = len(top)
    ring_xy = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    side_top = np.array([(x, y, 0.0) for x, y in ring_xy])
    side_bot = np.array([(x, y, -PLATE_Z) for x, y in ring_xy])
    verts = np.vstack([top, side_top, side_bot])
    tb = base + 4
    tris.append((tb + 2, tb + 1, tb + 0))
    tris.append((tb + 3, tb + 2, tb + 0))
    for k in range(4):
        t0, t1 = base + k, base + (k + 1) % 4
        b0, b1 = tb + k, tb + (k + 1) % 4
        tris.append((t0, b0, b1))
        tris.append((t0, b1, t1))

    tris = np.array(tris, dtype=np.int64)
    # Normals from the faces, which for the scanned top face is exactly +z.
    normals = np.zeros_like(verts)
    p = verts[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    for k in range(3):
        np.add.at(normals, tris[:, k], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    return verts, normals, tris


def place(verts, normals, place_xyz, yaw_deg):
    """Plate coordinates to WORLD coordinates. -> (verts, normals)

    A yaw about z and a translation. The normals rotate and do not translate,
    which is the one half of this that is easy to get wrong and impossible to
    see afterwards: a mesh with translated normals loads, renders, and gives
    the toolpath planner garbage orientations.
    """
    a = math.radians(float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (verts @ R.T + np.asarray(place_xyz, dtype=np.float64),
            normals @ R.T)


def write_ply(path, v, n, t):
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(b"comment simulated eddy-current test plate; flaws are in flaws.json, NOT in this mesh\n")
        f.write(("element vertex %d\n" % len(v)).encode())
        f.write(b"property double x\nproperty double y\nproperty double z\n")
        f.write(b"property double nx\nproperty double ny\nproperty double nz\n")
        f.write(("element face %d\n" % len(t)).encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        f.write(np.column_stack([v, n]).astype("<f8").tobytes())
        for tri in t:
            f.write(struct.pack("<B3i", 3, *tri))


def write_stl(path, v, t):
    p = v[t]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    with open(path, "wb") as f:
        f.write(b"\0"*80)
        f.write(struct.pack("<I", len(t)))
        for k in range(len(t)):
            f.write(struct.pack("<3f", *fn[k]))
            for c in range(3):
                f.write(struct.pack("<3f", *p[k, c]))
            f.write(b"\0\0")


SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="ec_test_plate">
    <static>true</static>
    <link name="plate">
      <visual name="visual">
        <geometry><mesh><uri>model://ec_test_plate/ec_test_plate.stl</uri></mesh></geometry>
        <material><ambient>0.7 0.7 0.72 1</ambient><diffuse>0.8 0.8 0.82 1</diffuse></material>
      </visual>
      <collision name="collision">
        <geometry><mesh><uri>model://ec_test_plate/ec_test_plate.stl</uri></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="ec_test_plate")
    ap.add_argument("--place", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    default=list(PLACE_XYZ),
                    help="where the plate's top face goes, world metres "
                         "(default %s)" % (list(PLACE_XYZ),))
    ap.add_argument("--yaw-deg", type=float, default=PLACE_YAW_DEG,
                    help="rotation of the plate about world z, degrees")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    v, n, t = plate_mesh()
    v, n = place(v, n, args.place, args.yaw_deg)
    write_ply(os.path.join(args.out_dir, "ec_test_plate.ply"), v, n, t)
    write_stl(os.path.join(args.out_dir, "ec_test_plate.stl"), v, t)
    with open(os.path.join(args.out_dir, "model.sdf"), "w") as f:
        f.write(SDF)

    fl = flaws()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plate": {"size_m": [PLATE_X, PLATE_Y, PLATE_Z],
                  "place_xyz_m": [float(c) for c in args.place],
                  "place_yaw_deg": float(args.yaw_deg),
                  "frame": "plate centre, z=0 at the scanned face, x along the long axis",
                  "meshes_frame": "world"},
        "material": "aluminium 6061-T6",
        "flaws": fl,
        "notes": ("Flaw dimensions are on the response table's nodes so no "
                  "interpolation in flaw geometry is needed. The notches are "
                  "NOT in the mesh: a 0.15 mm slot is below what the profiler "
                  "resolves and the 4.75 mm probe bridges it, so modelling it "
                  "geometrically would both mislead the toolpath planner and "
                  "give the simulated reconstruction a feature the real one "
                  "could not see."),
    }
    with open(os.path.join(args.out_dir, "flaws.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("plate %gx%gx%g mm, %d vertices, %d faces"
          % (PLATE_X*1e3, PLATE_Y*1e3, PLATE_Z*1e3, len(v), len(t)))
    print("meshes are in WORLD coordinates: top face at z = %g m, placed at "
          "%s, yaw %g deg" % (args.place[2], tuple(args.place), args.yaw_deg))
    print("%d notches:" % len(fl))
    for d in fl:
        print("   %-8s (%+7.1f, %+6.1f) mm  L %.3f  D %.2f  W %.2f mm  yaw %3.0f deg"
              % (d["name"], d["x"]*1e3, d["y"]*1e3, d["length_m"]*1e3,
                 d["depth_m"]*1e3, d["width_m"]*1e3, d["yaw_deg"]))
    print("\nwrote %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
