"""Check the position trigger in sim_ascan_source.dpi against known paths.

The trigger is the piece that makes the simulated digitizer behave like a
position trigger box: one A-scan per trigger_pitch_m of ARC LENGTH along
whatever path the machine follows. What has to hold is that trigger k lands at
arc length k*pitch from trigger 0, exactly, no matter how the path is sampled --
because that is the property that makes the trigger index and the column index
the same thing.

Runs on the host with no ROS and no recording database: the class is pulled out
of the .dpi and _publish is stubbed, so this tests the trigger and nothing else.
"""
import ast
import sys
import time
import numpy as np

SRC = "/home/tylerl/dataguzzler-work-dir/ROS-Scanning/dgpfiles/modules/sim_ascan_source.dpi"

tree = ast.parse(open(SRC).read())
cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimAScanSource"][0]
ns = {"np": np, "time": time}
exec(compile(ast.Module(body=[cls], type_ignores=[]), SRC, "exec"), ns)
SimAScanSource = ns["SimAScanSource"]

fails = []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print("  %-54s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


class Harness(SimAScanSource):
    """Drives _position_trigger_loop over a fixed list of observed positions."""

    def __init__(self, pitch, path):
        self.trigger_mode = "position"
        self.trigger_pitch_m = pitch
        self.poll_period = 0.0
        self._quit = False
        self.triggers = 0
        self.count = 0
        self.errors = 0
        self.travelled_m = 0.0
        self.burst_dropped = 0
        self._sink = None
        self._armed = False
        self._reset_phase = False
        self._emit_on_arm = True
        self.armed_triggers = 0
        self.armed_s_m = 0.0
        self._path = [np.asarray(p, dtype=float) for p in path]
        self._i = 0
        self.fired = []

    def current_position(self):
        if self._i >= len(self._path):
            self._quit = True
            return None
        p = self._path[self._i]
        self._i += 1
        return p

    def _publish(self, pos=None, s_m=None):
        # Everything except the recording-database write, so the real sink
        # dispatch in _deliver() is under test rather than a copy of it.
        self.fired.append(np.asarray(pos, dtype=float))
        self.count += 1
        self._deliver(None, pos, s_m)

    def run(self):
        self._position_trigger_loop()
        return self


def arc_at_triggers(path, fired):
    """Arc length along `path` at each fired position.

    Walks the path once, matching each trigger to the segment it lies on, so a
    trigger that spans a corner is measured along the path rather than across it.
    """
    out = []
    seg = 0
    base = 0.0
    for f in fired:
        while seg < len(path)-1:
            a, b = path[seg], path[seg+1]
            ab = b - a
            L = np.linalg.norm(ab)
            if L > 0:
                t = float(np.dot(f-a, ab)/(L*L))
                if -1e-9 <= t <= 1+1e-9 and np.linalg.norm(a + t*ab - f) < 1e-9:
                    out.append(base + t*L)
                    break
            base += L
            seg += 1
        else:
            out.append(base)
    return np.array(out)


print("=== straight line, unevenly sampled (TF jitter) ===")
rng = np.random.RandomState(7)
xs = np.cumsum(rng.uniform(0.001, 0.02, size=400))
xs = xs/xs[-1]*1.0
path = [np.array([x, 0.0]) for x in xs]
h = Harness(0.05, path).run()
fired = np.array(h.fired)
s = fired[:, 0] - fired[0, 0]           # arc length from trigger 0
check("trigger count", h.triggers, 1 + int(h.travelled_m//0.05))
check("spacing is exactly the pitch", float(np.abs(np.diff(s) - 0.05).max()), 0.0, 1e-12)
check("trigger k at exactly k*pitch", float(np.abs(s - np.arange(len(s))*0.05).max()), 0.0, 1e-12)
check("travelled equals path length", h.travelled_m, xs[-1]-xs[0], 1e-12)
print("    %d triggers over %.4f m at 0.050 m pitch" % (h.triggers, h.travelled_m))

print()
print("=== serpentine raster with link moves ===")
# Arc length is global, so the link between rows is real travel and fires real
# triggers. That is correct: the scan ignores them, and it is what makes the
# same trigger work on a path with no rows at all.
rows = []
y = 0.0
for j in range(4):
    xr = np.linspace(0, 0.30, 50)
    if j % 2:
        xr = xr[::-1]
    rows += [np.array([float(x), y]) for x in xr]
    y += 0.02
    rows += [np.array([float(xr[-1]), y])]
h2 = Harness(0.00508, rows).run()
arc = arc_at_triggers(rows, h2.fired)
check("trigger count", h2.triggers, 1 + int(h2.travelled_m//0.00508))
check("arc length spacing is exactly the pitch",
      float(np.abs(np.diff(arc) - 0.00508).max()), 0.0, 1e-9)
check("no bursts dropped", h2.burst_dropped, 0)
print("    %d triggers over %.4f m (rows 1.200 + links 0.080) at 5.08 mm pitch"
      % (h2.triggers, h2.travelled_m))

print()
print("=== a stationary probe must not trigger ===")
h3 = Harness(0.005, [np.array([0.1, 0.2])]*50).run()
check("only the initial trigger", h3.triggers, 1)

print()
print("=== a step longer than the pitch fires every crossing, not one ===")
# One 0.1 m jump at a 0.005 m pitch is 20 crossings. Missing them would silently
# lose 20 columns; firing one would put the whole rest of the scan out by 19.
h4 = Harness(0.005, [np.array([0.0, 0.0]), np.array([0.1, 0.0])]).run()
check("crossings fired", h4.triggers, 21)
check("evenly spaced across the jump",
      float(np.abs(np.diff([f[0] for f in h4.fired]) - 0.005).max()), 0.0, 1e-12)

print()
print("=== diagonal travel is measured as distance, not as dx ===")
h5 = Harness(0.01, [np.array([0.0, 0.0]), np.array([0.3, 0.4])]).run()   # 0.5 m
check("triggers along a 0.5 m diagonal", h5.triggers, 51)
check("travelled", h5.travelled_m, 0.5, 1e-12)

print()
print("=== arming: one pass, phase restarts at the pass origin ===")
# Two 0.30 m passes joined by a link move, as a raster does. Arming per pass is
# what ScanPath.trace_count() counts, and it is what makes trigger k of a pass
# the k-th column of that row.
PITCH = 0.005084745762711864          # 0.30/59, the 60-column grid pitch
passA = [np.array([x, 0.0]) for x in np.linspace(0.0, 0.30, 400)]
link = [np.array([0.30, y]) for y in np.linspace(0.0, 0.02, 20)]
passB = [np.array([x, 0.02]) for x in np.linspace(0.30, 0.60, 400)]

got = {"A": [], "B": []}
h6 = Harness(PITCH, passA + link + passB)
# Drive the loop manually so we can arm and disarm partway, the way a scan does.
h6._path = passA
h6._i = 0
h6._quit = False
h6.arm(lambda data, pos, s, i: got["A"].append((s, i)))
h6._position_trigger_loop()
h6.disarm()
check("pass A: triggers delivered", len(got["A"]), 60)
check("pass A: trigger k at k*pitch",
      float(np.abs(np.array([s for s, i in got["A"]]) - np.arange(60)*PITCH).max()), 0.0, 1e-12)
check("pass A: sink index is the trigger index",
      [i for s, i in got["A"]], list(range(60)))

# The link move must not feed the sink at all.
h6._path = link
h6._i = 0
h6._quit = False
h6._position_trigger_loop()          # still disarmed
check("link move delivers nothing", len(got["A"]) + len(got["B"]), 60)

# Second pass: phase restarts, so it is not offset by the link travel.
h6._path = passB
h6._i = 0
h6._quit = False
h6.arm(lambda data, pos, s, i: got["B"].append((s, i)))
h6._position_trigger_loop()
h6.disarm()
check("pass B: triggers delivered", len(got["B"]), 60)
check("pass B: trigger k at k*pitch, not offset by the link",
      float(np.abs(np.array([s for s, i in got["B"]]) - np.arange(60)*PITCH).max()), 0.0, 1e-12)

print()
print("=== emit_on_arm=False leaves trace 0 to the caller ===")
# The scan stages the pass-start trace itself so the accumulators can be set up
# on it; the source must then not stage a duplicate, but must still measure from
# that point.
got2 = []
h7 = Harness(PITCH, passA)
h7.arm(lambda data, pos, s, i: got2.append((s, i)), emit_on_arm=False)
h7._position_trigger_loop()
h7.disarm()
check("one fewer delivered, the caller's", len(got2), 59)
check("first delivered trigger is at one pitch, not zero", got2[0][0], PITCH, 1e-12)
check("still exactly on the grid",
      float(np.abs(np.array([s for s, i in got2]) - (np.arange(59)+1)*PITCH).max()), 0.0, 1e-12)

print()
print("=== a pass that CLIMBS is measured in 3D, not in plan view ===")
# The regression for probe_position_relative_to_scan() returning three
# components. A surface-following pass that rises while it advances travels
# further than its plan view says: measuring in xy alone under-counts by
# 1/cos(slope), which loses triggers systematically and shifts every later trace
# along the pass. On the bowed test plate every pass happens to run at constant
# height, so this stayed invisible there -- a property of that plate, not of
# surface following.
SLOPE = np.radians(20.0)
n = 500
dist = np.linspace(0.0, 0.50, n)                       # 0.50 m ALONG the surface
climb3 = [np.array([d*np.cos(SLOPE), 0.0, d*np.sin(SLOPE)]) for d in dist]
h8 = Harness(0.01, climb3).run()
check("triggers over 0.50 m of 3D travel", h8.triggers, 51)
check("travelled measured along the surface", h8.travelled_m, 0.50, 1e-9)

# and the same path seen only in plan view, which is what the bug did
climb2 = [np.array([p3[0], p3[1]]) for p3 in climb3]
h9 = Harness(0.01, climb2).run()
check("plan view under-counts travel", h9.travelled_m, 0.50*np.cos(SLOPE), 1e-9)
check("plan view loses triggers", h9.triggers, 47)
print("   plan view saw %.4f m and fired %d; 3D saw %.4f m and fired %d"
      % (h9.travelled_m, h9.triggers, h8.travelled_m, h8.triggers))

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("all trigger checks passed")
