"""Check segment inference against paths whose structure is known."""
import functools
import os
import sys
import types
import numpy as np

# Resolve the .dpi next to THIS file, never by absolute path: the suite has to
# run from any checkout, including the controlling computer.
HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# The .dpi imports geometry_msgs at module scope, so the name has to resolve on a
# host with no ROS. Never `sys.modules[...] = mod` here: sys.modules is
# process-wide and that would replace whatever the other suites in this directory
# installed. ensure_msgs() only ADDS, and this suite builds no messages of its
# own, so which stub is in force is not its business. See ros_msg_stubs.py.
ros_msg_stubs.ensure_msgs()

src = open(os.path.join(HERE, "toolpath_segments.dpi")).read()
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



def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


@_checked
def test_serpentine_raster():
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


@_checked
def test_supplied_boundaries_win():
    print()
    print("=== planner-supplied boundaries win over inference ===")
    sp2 = ScanPath(pos, quat, truth)
    check("uses the supplied list", sp2.segments_inferred, False)
    check("same segments", sp2.segment_starts, truth)


@_checked
def test_disconnected_regions():
    print()
    print("=== disconnected regions joined by a long jump ===")
    a = np.array([(x*0.005, 0.0, 0.0) for x in range(10)])
    b = np.array([(x*0.005, 0.5, 0.0) for x in range(10)])   # 0.5 m away
    pos2 = np.vstack([a, b])
    check("jump detected", infer_segment_starts(pos2), [0, 10])


@_checked
def test_straight_line_not_chopped():
    print()
    print("=== a single straight line must NOT be chopped up ===")
    line = np.array([(x*0.005, 0.0, 0.0) for x in range(50)])
    check("one segment", infer_segment_starts(line), [0])


@_checked
def test_gentle_curve_not_chopped():
    print()
    print("=== gentle curve must NOT be chopped up ===")
    th = np.linspace(0, np.pi/3, 60)
    curve = np.stack([0.2*np.cos(th), 0.2*np.sin(th), np.zeros_like(th)], axis=1)
    check("one segment", infer_segment_starts(curve), [0])


@_checked
def test_uneven_waypoints():
    print()
    print("=== unevenly spaced waypoints, as a real planner emits ===")
    # Noether puts a waypoint wherever a raster plane crosses a mesh edge, so step
    # length inherits the mesh triangle size and can vary by 4.5x WITHIN one raster
    # line. A jump_factor that low therefore fires in the middle of good lines and
    # infers several times as many boundaries as the planner reported.
    rng = np.random.RandomState(20260812)
    # Long-tailed, like edge crossings on an irregular mesh: median 1.4 mm, max
    # about 4.5x that.
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
    # and prove the guard is doing work: a lower factor does chop these lines up
    check("a jump_factor of 2.5 would split them",
          infer_segment_starts(uneven, jump_factor=2.5) == uneven_truth, False)


@_checked
def test_plane_grid_overtravel():
    print()
    print("=== plane grid rows with overtravel end ON the far edge, never past it ===")
    # 50 columns over a 60 mm taught row, held half a pitch short as the triggered
    # raster does, then each row extended by that same half pitch. The extension
    # must land exactly on the taught edge -- the probe may be in contact with the
    # part, and the taught rectangle is the limit of where it goes.
    m, n = 50, 3
    taught = 0.060
    margin = 0.5
    pitch = taught/(m-1+margin)
    uprime = np.linspace(0.0, taught*(m-1)/(m-1+margin), m)   # what _plane_grid does with end_margin
    uu = np.tile(uprime[:, None], (1, n))
    vv = np.tile((np.arange(n)*0.010)[None, :], (m, 1))
    ww = np.zeros((m, n))
    rot = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    sp = ns["scanpath_from_plane_grid"](uu, vv, ww, rot, serpentine=False, overtravel_pitches=margin)
    check("one extra waypoint per row", len(sp), n*(m+1))
    check("row count unchanged", len(sp.segments), n)
    ends = [sp.positions[e-1] for s_, e in sp.segments]
    check("every row ends on the taught far edge",
          bool(max(abs(p[0] - taught) for p in ends) < 1e-12), True)
    check("last COLUMN is half a pitch inside it",
          bool(abs(uprime[-1] - (taught - 0.5*pitch)) < 1e-12), True)
    check("row length is (m-1+margin) pitches",
          bool(abs(sp.segment_length_m(sp.segments[0]) - (m-1+margin)*pitch) < 1e-12), True)
    check("trace count at the grid pitch is m per row, no coin flip",
          sp.trace_count(pitch), m*n)
    sp2 = ns["scanpath_from_plane_grid"](uu, vv, ww, rot, serpentine=True, overtravel_pitches=margin)
    ends2 = [sp2.positions[e-1] for s_, e in sp2.segments]
    check("serpentine: alternate rows overtravel toward u=0",
          [round(p[0], 6) for p in ends2], [round(taught, 6), round(uprime[0] - 0.5*pitch, 6), round(taught, 6)])
    sp0 = ns["scanpath_from_plane_grid"](uu, vv, ww, rot, serpentine=False)
    check("overtravel_pitches=0 adds no waypoints at all", len(sp0), n*m)



@_checked
def test_not_blind():
    print()
    print("=== the test is not blind ===")
    check("the subject came out of toolpath_segments.dpi, not a stand-in",
          (ScanPath.__init__.__code__.co_filename,
           infer_segment_starts.__code__.co_filename),
          ("toolpath_segments.dpi", "toolpath_segments.dpi"))
    # KNOWN-POSITIVE AND KNOWN-NEGATIVE through the same call: a 0.5 m jump is a
    # boundary and a 5 mm step is not. If infer_segment_starts() were not running,
    # or were returning a constant, one of the two would be wrong.
    far = np.vstack([np.array([(x*0.005, 0.0, 0.0) for x in range(6)]),
                     np.array([(x*0.005, 0.5, 0.0) for x in range(6)])])
    near = np.array([(x*0.005, 0.0, 0.0) for x in range(12)])
    check("a 0.5 m jump is a boundary and a 5 mm step is not",
          (infer_segment_starts(far), infer_segment_starts(near)),
          ([0, 6], [0]))
    # scanpath_from_pose_array is the ONE thing in this .dpi that reads the
    # message classes, and it reads them with isinstance(). Hand it OUR OWN pose
    # objects -- a plain list, which is the documented bare-PoseArray path -- so
    # no other suite's stub can decide what this measures.
    class _Vec(object):
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Q(object):
        def __init__(self):
            self.x = self.y = self.z = 0.0
            self.w = 1.0

    class _MyPose(object):
        def __init__(self, x):
            self.position = _Vec(x, 0.0, 0.0)
            self.orientation = _Q()

    mine = ns["scanpath_from_pose_array"]([_MyPose(x*0.005) for x in range(12)])
    check("scanpath_from_pose_array reads OUR pose objects' fields",
          (len(mine), round(float(mine.positions[-1][0]), 6)), (12, 0.055))
    check("and it never needed geometry_msgs' own Pose",
          _MyPose.__module__, __name__)


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for name in _SECTIONS:
        fn = globals()[name]
        # __wrapped__ is the undecorated section: it records failures instead
        # of raising, so one run reports all of them.
        getattr(fn, "__wrapped__", fn)()
    print()
    if fails:
        print("FAILURES:", fails)
        return 1
    print("all segment checks passed")
    return 0


_SECTIONS = ['test_serpentine_raster', 'test_supplied_boundaries_win', 'test_disconnected_regions', 'test_straight_line_not_chopped', 'test_gentle_curve_not_chopped', 'test_uneven_waypoints', 'test_plane_grid_overtravel', 'test_not_blind']


if __name__ == "__main__":
    sys.exit(_main())
