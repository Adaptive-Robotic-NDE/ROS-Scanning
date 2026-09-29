"""Check the position trigger in sim_ascan_source.dpi against known paths.

The trigger is the piece that makes the simulated digitizer behave like a
position trigger box: one A-scan per trigger_pitch_m of ARC LENGTH along
whatever path the machine follows. What has to hold is that trigger k lands at
arc length k*pitch from trigger 0, exactly, no matter how the path is sampled --
because that is the property that makes the trigger index and the column index
the same thing.

Runs on the host with no ROS and no recording database: the class is pulled out
of the .dpi and _publish is stubbed, so this tests the trigger and nothing else.
The crossing arithmetic itself is PitchCrossingDetector's and has its own test
(pitch_crossings_test.py); what is checked here is that the simulator drives
it the way a scan needs -- arming, phase, the budget, the sink.
"""
import ast
import functools
import os
import sys
import time
import threading
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sim_ascan_source.dpi")
DET = os.path.join(HERE, "pitch_crossings.dpi")

ns = {"np": np, "time": time, "threading": threading}
for path, name in ((DET, "PitchCrossingDetector"), (SRC, "SimAScanSource")):
    tree = ast.parse(open(path).read())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name][0]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, "exec"), ns)
SimAScanSource = ns["SimAScanSource"]

fails = []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print("  %-54s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


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


class Harness(SimAScanSource):
    """Drives _position_trigger_loop over a fixed list of observed positions."""

    def __init__(self, pitch, path):
        self.trigger_mode = "position"
        self.trigger_pitch_m = pitch
        self.poll_period = 0.0
        self._quit = False
        self.count = 0
        self.errors = 0
        self._init_trigger_state(pitch)
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


# Two 0.30 m passes joined by a link move, as a raster does. Arming per pass is
# what ScanPath.trace_count() counts, and it is what makes trigger k of a pass
# the k-th column of that row. Module scope, because three sections below are
# measured against the SAME pass and nothing here asserts.
PITCH = 0.005084745762711864          # 0.30/59, the 60-column grid pitch
passA = [np.array([x, 0.0]) for x in np.linspace(0.0, 0.30, 400)]
link = [np.array([0.30, y]) for y in np.linspace(0.0, 0.02, 20)]
passB = [np.array([x, 0.02]) for x in np.linspace(0.30, 0.60, 400)]


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


@_checked
def test_straight_line_unevenly_sampled():
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


@_checked
def test_tf_noise_must_not_manufacture_a_trigger():
    print("=== TF noise must not manufacture a trigger ===")
    # Summing raw pose deltas inflates the travel by the noise, and 2% of
    # inflation is enough to fire one extra trigger per pass and shear the
    # volume. 50 columns over 60 mm with the half-pitch end margin, 0.06 mm true
    # step per poll, 0.02 mm rms of noise per axis.
    rng = np.random.RandomState(11)
    npoll = 1000
    truth = np.column_stack([np.linspace(0.0, 0.060, npoll), np.zeros(npoll), np.zeros(npoll)])
    noisy = truth + rng.normal(0.0, 0.02e-3, size=truth.shape)
    noisy[0] = truth[0]
    PITCH50 = 0.060/49.5
    h2 = Harness(PITCH50, noisy).run()
    print("    summed travel %.1f mm (%.0f%%) for a 60 mm pass; progress %.1f mm"
          % (1e3*h2.raw_travelled_m, 100*h2.raw_travelled_m/0.060, 1e3*h2.travelled_m))
    check("the sum IS inflated", h2.raw_travelled_m > 1.01*0.060, True)
    check("exactly 50 triggers: origin + 49 columns", h2.triggers, 50)
    check("no bursts dropped", h2.burst_dropped, 0)

    print()


@_checked
def test_a_stationary_probe_must_not_trigger():
    print("=== a stationary probe must not trigger ===")
    h3 = Harness(0.005, [np.array([0.1, 0.2])]*50).run()
    check("only the initial trigger", h3.triggers, 1)

    print()


@_checked
def test_a_long_step_fires_every_crossing():
    print("=== a step longer than the pitch fires every crossing, not one ===")
    # One 0.1 m jump at a 0.005 m pitch is 20 crossings. Missing them would silently
    # lose 20 columns; firing one would put the whole rest of the scan out by 19.
    h4 = Harness(0.005, [np.array([0.0, 0.0]), np.array([0.1, 0.0])]).run()
    check("crossings fired", h4.triggers, 21)
    check("evenly spaced across the jump",
          float(np.abs(np.diff([f[0] for f in h4.fired]) - 0.005).max()), 0.0, 1e-12)

    print()


@_checked
def test_diagonal_travel_is_distance():
    print("=== diagonal travel is measured as distance, not as dx ===")
    h5 = Harness(0.01, [np.array([0.0, 0.0]), np.array([0.3, 0.4])]).run()   # 0.5 m
    check("triggers along a 0.5 m diagonal", h5.triggers, 51)
    check("travelled", h5.travelled_m, 0.5, 1e-12)

    print()


@_checked
def test_arming_restarts_the_phase_at_each_pass():
    print("=== arming: one pass, phase restarts at the pass origin ===")
    got = {"A": [], "B": []}
    h6 = Harness(PITCH, passA + link + passB)
    # Drive the loop manually so it can be armed and disarmed partway, as a scan does.
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


@_checked
def test_the_budget_caps_delivery():
    print("=== the budget: a pass delivers at most what it was armed for ===")
    # passA again, but told it has 40 triggers. The other 20 crossings are real
    # travel and are counted, and none of them reaches the sink.
    got3 = []
    h6b = Harness(PITCH, passA)
    h6b.arm(lambda data, pos, s, i: got3.append((s, i)), max_triggers=40)
    h6b._position_trigger_loop()
    h6b.disarm()
    check("delivered exactly the budget", len(got3), 40)
    check("surplus crossings counted", h6b.armed_over_budget, 20)
    check("crossings still fired (and published, unarmed-style) past the budget", h6b.triggers, 60)

    print()


@_checked
def test_emit_on_arm_false_leaves_trace_zero_to_the_caller():
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


@_checked
def test_a_climbing_pass_is_measured_in_3d():
    print("=== a pass that CLIMBS is measured in 3D, not in plan view ===")
    # probe_position_relative_to_scan() must return all three components. A
    # surface-following pass that rises while it advances travels further than its
    # plan view says: measuring in xy alone under-counts by 1/cos(slope), which
    # loses triggers systematically and shifts every later trace along the pass. A
    # specimen whose passes all run at constant height hides this entirely, so it
    # has to be provoked with a path that climbs.
    SLOPE = np.radians(20.0)
    n = 500
    dist = np.linspace(0.0, 0.50, n)                       # 0.50 m ALONG the surface
    climb3 = [np.array([d*np.cos(SLOPE), 0.0, d*np.sin(SLOPE)]) for d in dist]
    h8 = Harness(0.01, climb3).run()
    check("triggers over 0.50 m of 3D travel", h8.triggers, 51)
    check("travelled measured along the surface", h8.travelled_m, 0.50, 1e-9)

    # the same path seen only in plan view, for contrast
    climb2 = [np.array([p3[0], p3[1]]) for p3 in climb3]
    h9 = Harness(0.01, climb2).run()
    check("plan view under-counts travel", h9.travelled_m, 0.50*np.cos(SLOPE), 1e-9)
    check("plan view loses triggers", h9.triggers, 47)
    print("   plan view saw %.4f m and fired %d; 3D saw %.4f m and fired %d"
          % (h9.travelled_m, h9.triggers, h8.travelled_m, h8.triggers))

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("all trigger checks passed")


_SECTIONS = ["test_straight_line_unevenly_sampled", "test_tf_noise_must_not_manufacture_a_trigger", "test_a_stationary_probe_must_not_trigger", "test_a_long_step_fires_every_crossing", "test_diagonal_travel_is_distance", "test_arming_restarts_the_phase_at_each_pass", "test_the_budget_caps_delivery", "test_emit_on_arm_false_leaves_trace_zero_to_the_caller", "test_a_climbing_pass_is_measured_in_3d"]


if __name__ == "__main__":
    sys.exit(_main())
