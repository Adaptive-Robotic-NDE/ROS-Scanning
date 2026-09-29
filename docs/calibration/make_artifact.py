"""A 3D-printable calibration artifact for the ex-ur10e-1 cell.

WHAT IT HAS TO DO. Three sensors with nothing in common must all locate the
same rigid body:

  Azure RGB    sees the Creaform targets as high-contrast circles.
  Azure depth  sees them in /azure/ir/image_raw, where the retroreflector is a
               bright blob, pixel-registered to the depth sensor and carrying
               its intrinsics. (The DEPTH at a retroreflector is unreliable --
               they saturate a ToF sensor -- so the amplitude image plus PnP is
               the right route, not the point cloud.)
  Keyence      sees height only. It gets the pad TOPS as flat planes at known
               heights, the pad EDGES as sharp circular steps, and the CONES,
               whose apex is recovered by fitting the flanks and extrapolating
               -- which is robust to the nozzle rounding the tip.

WHY THE HEIGHTS VARY. A flat target plate conditions a hand-eye solve badly.
Pads at 3..20 mm give real out-of-plane structure, and no rotation of the
pattern maps it onto itself, so the pose is unambiguous.

THE ARTIFACT DOES NOT NEED TO BE ACCURATE, IT NEEDS TO BE KNOWN. Print it,
then characterise the ACTUAL pad heights and target centres once with the
Keyence itself. From that moment the measured geometry is the reference and
the printer's tolerance stops mattering. That is what makes 3D printing a
legitimate choice here rather than a compromise.
"""
import struct
import numpy as np

MM = 1.0
BASE_X, BASE_Y, BASE_Z = 160.0, 160.0, 4.0
CHAMFER = 25.0                      # one corner cut: makes the outline asymmetric
PAD_D, PAD_PITCH, PAD_0 = 22.0, 45.0, 35.0
#: mm ABOVE the base top. No rotation maps this onto itself.
PAD_H = [3.0, 10.0, 17.0,
         20.0, 6.0, 13.0,
         10.0, 3.0, 17.0]
CONE_D, CONE_H = 26.0, 9.0
CONE_XY = [(57.5, 57.5), (102.5, 57.5), (57.5, 102.5), (102.5, 102.5)]
NSEG = 96

tris = []


def quad(a, b, c, d):
    tris.append((a, b, c)); tris.append((a, c, d))


def prism(poly, z0, z1):
    """A closed extrusion of a CCW polygon between two z planes."""
    n = len(poly)
    lo = [(x, y, z0) for x, y in poly]
    hi = [(x, y, z1) for x, y in poly]
    for i in range(1, n - 1):                     # bottom, wound downward
        tris.append((lo[0], lo[i + 1], lo[i]))
    for i in range(1, n - 1):                     # top
        tris.append((hi[0], hi[i], hi[i + 1]))
    for i in range(n):                            # walls
        j = (i + 1) % n
        quad(lo[i], lo[j], hi[j], hi[i])


def circle(cx, cy, r, n=NSEG):
    t = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    return [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in t]


def cone(cx, cy, r, z0, h, n=NSEG):
    base = circle(cx, cy, r, n)
    lo = [(x, y, z0) for x, y in base]
    apex = (cx, cy, z0 + h)
    for i in range(1, n - 1):
        tris.append((lo[0], lo[i + 1], lo[i]))
    for i in range(n):
        j = (i + 1) % n
        tris.append((lo[i], lo[j], apex))


# --- the base, with one corner chamfered so the outline cannot be flipped ---
base_poly = [(CHAMFER, 0.0), (BASE_X, 0.0), (BASE_X, BASE_Y),
             (0.0, BASE_Y), (0.0, CHAMFER)]
prism(base_poly, 0.0, BASE_Z)

# --- nine target pads. Each starts INSIDE the base so the solids overlap and
# --- any slicer unions them cleanly.
centres = []
for r in range(3):
    for c in range(3):
        cx = PAD_0 + c*PAD_PITCH
        cy = PAD_0 + r*PAD_PITCH
        h = PAD_H[r*3 + c]
        centres.append((cx, cy, BASE_Z + h))
        prism(circle(cx, cy, PAD_D/2.0), 0.0, BASE_Z + h)

# --- four cones between them, for the Keyence ---
# EXTENDED 1 mm INTO THE BASE rather than sat on top of it. A cone whose base
# disc exactly coincides with the base's top face gives two coincident,
# oppositely-wound faces; that is legal but it is the kind of thing a slicer
# can get wrong. Overlapping closed volumes union cleanly everywhere. The base
# radius is scaled so the FLANK ANGLE and the APEX HEIGHT are unchanged.
SINK = 1.0
cone_r = (CONE_H + SINK)*(CONE_D/2.0)/CONE_H
for cx, cy in CONE_XY:
    cone(cx, cy, cone_r, BASE_Z - SINK, CONE_H + SINK)

# --- write binary STL ---
with open("tari_calibration_artifact.stl", "wb") as f:
    f.write(b"tari calibration artifact ex-ur10e-1".ljust(80, b" "))
    f.write(struct.pack("<I", len(tris)))
    for a, b, c in tris:
        a, b, c = np.array(a), np.array(b), np.array(c)
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        nrm = nrm/ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        f.write(struct.pack("<3f", *nrm))
        for v in (a, b, c):
            f.write(struct.pack("<3f", *v))
        f.write(struct.pack("<H", 0))

print("tari_calibration_artifact.stl: %d triangles" % len(tris))
print("footprint %.0f x %.0f mm, tallest point %.1f mm"
      % (BASE_X, BASE_Y, BASE_Z + max(PAD_H)))
print("\npad centres (x, y, TOP z) in artifact coordinates, mm:")
for i, (cx, cy, z) in enumerate(centres):
    print("  pad %d  (%6.1f, %6.1f)  top z %5.1f" % (i, cx, cy, z))
print("cone apices:")
for cx, cy in CONE_XY:
    print("  cone   (%6.1f, %6.1f)  apex z %5.1f" % (cx, cy, BASE_Z + CONE_H))
np.save("nominal_geometry.npy",
        {"pads": centres, "cones": [(x, y, BASE_Z + CONE_H) for x, y in CONE_XY],
         "base_z": BASE_Z, "pad_d": PAD_D, "cone_d": CONE_D}, allow_pickle=True)
