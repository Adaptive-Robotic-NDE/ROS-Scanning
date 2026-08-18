"""Check segment inference against paths whose structure we know."""
import sys
import types
import numpy as np

# The .dpi imports geometry_msgs; stub it so this runs on the host without ROS.
mod = types.ModuleType("geometry_msgs.msg")
class PoseArray(object):
    def __init__(self): self.poses = []
class Pose(object): pass
mod.PoseArray = PoseArray
mod.Pose = Pose
sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
sys.modules["geometry_msgs.msg"] = mod

src = open("/home/tylerl/dataguzzler-work-dir/ROS-Scanning/dgpfiles/modules/toolpath_segments.dpi").read()
src = src.replace("global ScanPath", "#").replace("global infer_segment_starts", "#")
src = src.replace("global scanpath_from_pose_array", "#").replace("global scanpath_from_plane_grid", "#")
ns = {}
exec(compile(src, "toolpath_segments.dpi", "exec"), ns)
ScanPath = ns["ScanPath"]
infer_segment_starts = ns["infer_segment_starts"]

fails = []
def check(name, got, want):
    ok = (got == want)
    print("  %-52s %s" % (name, "ok" if ok else "FAIL  got=%s want=%s" % (got, want)))
    if not ok: fails.append(name)

# --- serpentine raster, 5 rows x 20 columns, 5 mm pitch, 10 mm row spacing
rows, cols = 5, 20
pos = []
truth = []
for j in range(rows):
    truth.append(len(pos))
    xs = np.arange(cols)*0.005
    if j % 2: xs = xs[::-1]
    for x in xs:
        pos.append((x, j*0.010, 0.0))
pos = np.array(pos)
quat = np.tile([0, 0, 0, 1.0], (len(pos), 1))

print("=== serpentine raster (the case the planner's flattening hides) ===")
inferred = infer_segment_starts(pos)
check("inferred boundaries match the true row starts", inferred, truth)

sp = ScanPath(pos, quat, None)
check("segment count", len(sp.segments), rows)
check("all rows look like scanning passes", len(sp.acquiring_segments()), rows)
# arc length is global and monotonic
print("  total path length %.4f m (rows %.3f + links %.3f)"
      % (sp.total_length_m(), rows*(cols-1)*0.005, (rows-1)*0.010))
check("arc length monotonic", bool(np.all(np.diff(sp.s_m) >= 0)), True)

# trace count at 0.5 mm pitch: each row 95 mm -> 190 intervals + 1
expect = rows*(int(np.floor(0.095/0.0005))+1)
check("trace count at 0.5 mm pitch", sp.trace_count(0.0005), expect)

print()
print("=== planner-supplied boundaries win over inference ===")
sp2 = ScanPath(pos, quat, truth)
check("uses the supplied list", sp2.segments_inferred, False)
check("same segments", sp2.segment_starts, truth)

print()
print("=== disconnected regions joined by a long jump ===")
a = np.array([(x*0.005, 0.0, 0.0) for x in range(10)])
b = np.array([(x*0.005, 0.5, 0.0) for x in range(10)])   # 0.5 m away
pos2 = np.vstack([a, b])
check("jump detected", infer_segment_starts(pos2), [0, 10])

print()
print("=== a single straight line must NOT be chopped up ===")
line = np.array([(x*0.005, 0.0, 0.0) for x in range(50)])
check("one segment", infer_segment_starts(line), [0])

print()
print("=== gentle curve must NOT be chopped up ===")
th = np.linspace(0, np.pi/3, 60)
curve = np.stack([0.2*np.cos(th), 0.2*np.sin(th), np.zeros_like(th)], axis=1)
check("one segment", infer_segment_starts(curve), [0])

print()
print("=== unevenly spaced waypoints, as a real planner emits ===")
# Noether puts a waypoint wherever a raster plane crosses a mesh edge, so step
# length inherits the mesh triangle size. On curved_scan.ply it varied by 4.5x
# WITHIN one raster line. jump_factor was 2.5, so the long-step test fired in
# the middle of good lines: 251 boundaries inferred where the planner reported
# 73, 188 of them spurious. Reproduced here without needing the mesh.
rng = np.random.RandomState(20260812)
# Long-tailed, like edge crossings on an irregular mesh: median 1.4 mm, max
# about 4.5x that -- the measured shape on curved_scan.ply.
row_steps = 0.0014*np.exp(rng.normal(0.0, 0.6, size=140))
uneven = []
uneven_truth = []
y = 0.0
for j in range(4):
    uneven_truth.append(len(uneven))
    x = 0.0
    uneven.append((x, y, 0.0))
    for s in row_steps:
        x += s
        uneven.append((x, y, 0.0))
    y += 0.020                                      # line spacing, the real link
uneven = np.array(uneven)
print("  step lengths: median %.4f m, max %.4f m (%.1fx)"
      % (np.median(row_steps), row_steps.max(), row_steps.max()/np.median(row_steps)))
check("uneven spacing does not manufacture boundaries",
      infer_segment_starts(uneven), uneven_truth)
# and prove this is a real guard: the old default chopped these lines up
check("the old jump_factor=2.5 would have failed this",
      infer_segment_starts(uneven, jump_factor=2.5) == uneven_truth, False)

print()
if fails:
    print("FAILURES:", fails); sys.exit(1)
print("all segment checks passed")
