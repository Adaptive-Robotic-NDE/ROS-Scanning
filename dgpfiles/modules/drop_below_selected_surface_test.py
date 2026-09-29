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
# File:  drop_below_selected_surface_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that the bench beneath a selected part is dropped from the
#   planning mesh.
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
"""The planning mesh must not carry the bench the part is standing on.

WHAT GOES WRONG WITHOUT IT. The detected region is a BOX around the part and it
overhangs the part's edges -- a few millimetres on a flat plate. Cropping the
planning mesh to the region in x-y therefore keeps a thin ring of BENCH, tens of
millimetres below the plate. Noether plans a pass along that ring, and the
pre-flight standoff gate measures the first row against what was commanded and
refuses the whole sweep:

    RuntimeError: this keyence sweep is refused before it flies: 100.0% of these
    points are within 5 mm of the NEAR limit (57 mm) ... the plan commanded
    78.5 mm here and the sensor is measuring 60.2 mm, so the surface this raster
    was planned over is 18.3 mm BELOW the real one

The gate is right; the ring should never have been planned over.

drop_below_selected_surface() removes it, and the bound it uses is the SENSOR'S:
the LJ-V7080 measures over a 46 mm window, so a planning surface deeper than that
cannot be rastered at one standoff whatever the toolpath does.

What is checked here, on the HOST with numpy and no ROS:

  * a plate-on-a-bench mesh loses the bench and keeps every face of the plate;
  * the top is a PERCENTILE, so a few high TSDF outliers do not drag the band up
    and take the part with it;
  * a mesh it disagrees with -- one where most faces are "deep" -- is returned
    UNTOUCHED, so the gate downstream gets the real mesh and says so itself;
  * relief SHALLOWER than the depth survives, which is the case that must not be
    trimmed: a real step inside the operator's region;
  * the file it writes reads back with the same faces, through the same reader
    the planner uses.

    python3 drop_below_selected_surface_test.py
    python3 -m pytest -q drop_below_selected_surface_test.py
"""
import ast
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILURES = []


def check(name, condition, detail=""):
    print("%s  %s%s" % ("ok  " if condition else "FAIL", name,
                        "" if condition or not detail else "  -- " + detail))
    if not condition:
        FAILURES.append(name)


def _lift(path, wanted, extra=None):
    """The named top-level defs and assignments out of a .dpi, unexecuted rest."""
    src = open(os.path.join(HERE, path)).read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    tree = ast.parse(body)
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted:
            keep.append(n)
        elif isinstance(n, ast.Assign):
            names = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if names & wanted:
                keep.append(n)
    got = set()
    for n in keep:
        got.add(getattr(n, "name", None) or
                next(t.id for t in n.targets if isinstance(t, ast.Name)))
    missing = wanted - got
    if missing:
        raise AssertionError(
            "%s no longer defines %s at its top level -- renamed, moved into a "
            "class, or deleted, and this suite would otherwise test nothing"
            % (path, ", ".join(sorted(missing))))
    ns = {"np": np, "os": os}
    ns.update(extra or {})
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


ply = _lift("noether_path.dpi", {"read_ply_mesh", "_gate_write_ply"})
mr = _lift("mesh_reconstruction.dpi",
           {"drop_below_selected_surface", "region_of_measured_surface",
            "KEYENCE_PLAN_SURFACE_DEPTH_M", "KEYENCE_PLAN_DEPTH_MAX_DROP"},
           extra={"read_ply_mesh": ply["read_ply_mesh"],
                  "_gate_write_ply": ply["_gate_write_ply"]})
drop = mr["drop_below_selected_surface"]
tighten = mr["region_of_measured_surface"]
DEPTH = mr["KEYENCE_PLAN_SURFACE_DEPTH_M"]
MAX_DROP = mr["KEYENCE_PLAN_DEPTH_MAX_DROP"]

TMP = tempfile.mkdtemp(prefix="dropdepth-")


def grid(x0, x1, y0, y1, z, n=6):
    """A flat n x n triangulated patch at height z. -> (vertices, triangles)."""
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    v = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, float(z))])
    t = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i*n + j
            t += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return v, np.asarray(t, int)


def write(name, parts):
    """One PLY holding several patches. -> (path, per-patch face counts)."""
    V, T, counts = [], [], []
    for v, t in parts:
        T.append(t + len(np.vstack(V)) if V else t)
        V.append(v)
        counts.append(len(t))
    v = np.vstack(V)
    t = np.vstack(T)
    p = os.path.join(TMP, name)
    ply["_gate_write_ply"](p, v, None, t, "test")
    return p, counts


print("")
print("the constant")
check("the depth is the sensor's 46 mm window, rounded up, not something larger",
      0.040 <= DEPTH <= 0.060, "%.3f m" % (DEPTH,))
check("and the decline threshold leaves a real disagreement alone",
      0.3 <= MAX_DROP <= 0.9, "%.2f" % (MAX_DROP,))


print("")
print("a plate on a bench")

# Plate top at z 1.000 on a bench at 0.9286, so 71 mm apart -- well past the
# 50 mm depth. The bench ring is the minority of the faces.
plate = grid(0.75, 0.95, -0.35, -0.15, 1.000, n=8)
bench = grid(0.75, 0.95, -0.36, -0.355, 0.9286, n=4)
path, (n_plate, n_bench) = write("plate_on_bench.ply", [plate, bench])
out = drop(path, verbose=False)
check("it wrote a new file", isinstance(out, str) and out != path, str(out))
v2, _n2, t2 = ply["read_ply_mesh"](out)
check("the bench is gone and every plate face is kept",
      len(t2) == n_plate, "%d faces out of %d plate + %d bench"
      % (len(t2), n_plate, n_bench))
check("and nothing below the plate survives",
      float(np.min(v2[:, 2])) > 0.99, "min z %.4f" % (float(np.min(v2[:, 2])),))


print("")
print("a few high outliers must not drag the band up")

# One spike 200 mm above the plate. With the MAXIMUM as the top, the band would
# run 1.200..1.150 and take the whole plate with it; with a percentile it does not.
spike = (np.array([[0.85, -0.25, 1.200], [0.851, -0.25, 1.200],
                   [0.85, -0.251, 1.200]]), np.array([[0, 1, 2]]))
path2, _ = write("plate_with_spike.ply", [plate, spike])
out2 = drop(path2, verbose=False)
v3, _n3, t3 = ply["read_ply_mesh"](out2)
check("the plate survives a 200 mm outlier", len(t3) >= n_plate,
      "%d faces, expected at least %d" % (len(t3), n_plate))


print("")
print("relief shallower than the depth is NOT trimmed")

step = grid(0.75, 0.95, -0.25, -0.15, 1.000 - 0.5*DEPTH, n=8)
path3, (a, b) = write("stepped.ply", [plate, step])
out3 = drop(path3, verbose=False)
check("a step half the depth deep keeps every face", out3 == path3,
      "returned %s" % ("a new file" if out3 != path3 else "the original",))


print("")
print("a mesh it disagrees with is left alone")

# Most of the faces are deep: this is not a bench ring, it is two surfaces the
# crop has no business choosing between.
deep = grid(0.75, 0.95, -0.35, -0.15, 1.000 - 3.0*DEPTH, n=12)
path4, (few, many) = write("mostly_deep.ply", [plate, deep])
check("the deep part really is the majority", many > few, "%d vs %d" % (many, few))
out4 = drop(path4, verbose=False)
check("it declines and returns the mesh untouched", out4 == path4,
      "returned %s" % (out4,))


print("")
print("degenerate inputs")

check("depth 0 is a no-op", drop(path, depth_m=0.0, verbose=False) == path)
check("a non-path is returned as it came", drop(None, verbose=False) is None)
check("a missing file is returned as it came",
      drop(os.path.join(TMP, "nope.ply"), verbose=False).endswith("nope.ply"))


print("")
print("region_of_measured_surface()")

# A 200 x 200 mm plate detected as a 200 x 217 mm region: 8% of the box is bench,
# so the coverage ceiling is 200/217 = 92% against a 90% gate. Tightening the box
# to the plate lifts the ceiling to 100%.
DETECTED = (0.750, -0.354, 0.950, -0.137)          # 200 x 217 mm
plate_only, _ = write("plate_only.ply", [plate])
got = tighten(plate_only, DETECTED, verbose=False)
check("the region is tightened in y, where the box overhangs",
      abs((got[3] - got[1]) - 0.200) < 2e-3,
      "%.1f mm tall, was %.1f" % (1e3*(got[3] - got[1]),
                                  1e3*(DETECTED[3] - DETECTED[1])))
check("and x is unchanged, because the box matched the part there",
      abs(got[0] - 0.750) < 2e-3 and abs(got[2] - 0.950) < 2e-3,
      "x %.3f..%.3f" % (got[0], got[2]))
check("the coverage CEILING goes from 92% to 100%",
      abs((0.200*0.200)/((got[2]-got[0])*(got[3]-got[1])) - 1.0) < 0.03,
      "%.0f%%" % (100.0*(0.200*0.200)/((got[2]-got[0])*(got[3]-got[1])),))

# IT ONLY SHRINKS. A mesh that reaches beyond the operator's box must not widen
# it: the box is a bound the operator set, and the crops downstream enforce it.
wide, _ = write("wide.ply", [grid(0.60, 1.05, -0.45, 0.00, 1.000, n=8)])
same = tighten(wide, DETECTED, verbose=False)
check("a mesh wider than the region does not widen it",
      all(abs(a - b) < 1e-9 for a, b in zip(same, DETECTED)), str(same))

# THE BENCH MUST NOT COUNT as measured surface -- that is the whole point, and it
# is the same depth test drop_below_selected_surface() uses.
with_bench, _ = write("tighten_bench.ply", [plate, bench])
got2 = tighten(with_bench, DETECTED, verbose=False)
check("a bench ring 71 mm down does not stretch the region back out",
      abs((got2[3] - got2[1]) - 0.200) < 2e-3,
      "%.1f mm tall" % (1e3*(got2[3] - got2[1]),))

check("no region in, no region out", tighten(plate_only, None) is None)
check("a non-path leaves the region alone",
      tighten(None, DETECTED) == DETECTED)


def test_drop_below_selected_surface():
    """So pytest reports what running the file reports."""
    assert not FAILURES, "%d check(s) failed: %s" % (len(FAILURES),
                                                     "; ".join(FAILURES))


if __name__ == "__main__":
    print("")
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("all passed")
