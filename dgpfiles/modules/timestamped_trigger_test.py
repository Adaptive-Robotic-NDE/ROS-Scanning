"""Check that a crossing is stamped at the crossing, not at the poll.

The timestamp IS the correlation in modules/timestamped_trigger.dpi: the pose
transaction is stamped at time T, and the recording database hands back the
A-scan that was current at T. So a timestamp taken when the poll noticed a
crossing rather than when the crossing happened is not a cosmetic inaccuracy --
two crossings inside one poll interval share a timestamp, correlate to the same
shot of the digitizer, and record one A-scan twice under two different
positions.

Crossings per poll is velocity/(step * poll_hz). One at 50 mm/s and a 5.08 mm
step; ten at 50 mm/s and a 0.5 mm step, which is exactly what happens when a
technician turns the step size down.

Runs on the host with no ROS, no recording database and no spatialnde2: the
class is pulled out of the .dpi, the clock is a fake with a known rate, and
_publish_pose records instead of writing.
"""
import ast
import sys
import types
import numpy as np

SRC = "/home/tylerl/dataguzzler-work-dir/ROS-Scanning/dgpfiles/modules/timestamped_trigger.dpi"

# _watch_loop imports dataguzzler_python.context for InitFreeThread.
dgp = types.ModuleType("dataguzzler_python")
ctx = types.ModuleType("dataguzzler_python.context")
ctx.InitFreeThread = lambda: None
dgp.context = ctx
sys.modules["dataguzzler_python"] = dgp
sys.modules["dataguzzler_python.context"] = ctx

tree = ast.parse(open(SRC).read())
cls = [n for n in tree.body
       if isinstance(n, ast.ClassDef) and n.name == "TimestampedTriggerSource"][0]
ns = {"np": np, "time": types.SimpleNamespace(sleep=lambda s: None,
                                              monotonic=lambda: 0.0)}
exec(compile(ast.Module(body=[cls], type_ignores=[]), SRC, "exec"), ns)
TimestampedTriggerSource = ns["TimestampedTriggerSource"]

fails = []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print("  %-56s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


class FakeTime(object):
    """measurement_time's interface, in integer nanoseconds."""

    def __init__(self, ns):
        self.ns = int(ns)

    def difference_seconds(self, other):
        return (self.ns - other.ns)*1e-9

    def seconds_nanos(self):
        return (self.ns//10**9, self.ns % 10**9)


class FakeClock(object):
    def __init__(self, supports_construction=True):
        self.ns = 0
        self.supports_construction = supports_construction

    def get_current_time(self):
        return FakeTime(self.ns)

    def time_from_secs_nanos(self, secs, nanos):
        if not self.supports_construction:
            # What a Python director subclass that only implements
            # get_current_time() would do -- the trigger box case.
            raise NotImplementedError("this clock cannot build a time")
        return FakeTime(secs*10**9 + nanos)


class Harness(TimestampedTriggerSource):
    """Drives _watch_loop over a straight pass at a known constant velocity."""

    def __init__(self, pitch, length_m, velocity_mps, poll_hz,
                 clock=None):
        self.clock = FakeClock() if clock is None else clock
        self.trigger_pitch_m = pitch
        self.poll_period = 1.0/poll_hz
        self.velocity = velocity_mps
        self._quit = False
        self._armed = False
        self._sink = None
        self._emit_on_arm = True
        self._reset_phase = False
        self.travelled_m = 0.0
        self.triggers = 0
        self.armed_triggers = 0
        self.armed_s_m = 0.0
        self.errors = 0
        self.last_error = None
        self.interpolated_timestamps = 0
        self.uninterpolated_timestamps = 0
        self.max_crossings_per_poll = 0
        self._interp_warned = False

        # The pass, sampled once per poll at constant velocity.
        self.polls = int(np.floor(length_m/(velocity_mps*self.poll_period))) + 1
        self._i = 0
        self.fired = []          # (timestamp_ns, x)

    def position_fn(self):
        if self._i >= self.polls:
            self._quit = True
            return None
        x = self._i*self.velocity*self.poll_period
        # The clock reads the time this position was observed at, which is what
        # _watch_loop takes immediately after this returns.
        self.clock.ns = int(round(self._i*self.poll_period*1e9))
        self._i += 1
        return (x, 0.0, 0.0)

    def _publish_pose(self, timestamp, pos):
        self.fired.append((timestamp.ns, float(pos[0])))
        return object()

    def run(self):
        self._watch_loop()
        return self


def expected_crossings(h):
    """One at arming, then one per pitch of the travel actually observed."""
    return 1 + int(h.travelled_m/h.trigger_pitch_m + 1e-9)


print("=== one crossing per poll: the case that always worked ===")
# 50 mm/s at a 5.08 mm step and a 200 Hz poll is 0.05 crossings per poll.
h = Harness(pitch=0.00508, length_m=0.30, velocity_mps=0.050, poll_hz=200.0).run()
check("crossings", len(h.fired), expected_crossings(h))
check("at most one crossing per poll", h.max_crossings_per_poll, 1)
ts = np.array([t for t, x in h.fired], dtype=np.float64)*1e-9
check("every timestamp distinct", len(set(ts)), len(ts))

print()
print("=== five crossings per poll: 50 mm/s at a 0.5 mm step ===")
# velocity/(step*poll_hz) = 0.050/(0.0005*20) = 5.
h = Harness(pitch=0.0005, length_m=0.30, velocity_mps=0.050, poll_hz=20.0).run()
xs = np.array([x for t, x in h.fired])
ts = np.array([t for t, x in h.fired], dtype=np.float64)*1e-9
print("    %d crossings, worst %d in one poll interval"
      % (len(h.fired), h.max_crossings_per_poll))
check("crossings share polls, which is the whole problem",
      h.max_crossings_per_poll, 5)
check("no two crossings share a timestamp", len(set(ts)), len(ts))
# The pass is at constant velocity, so the truth is exact: crossing k happened
# at k*pitch/velocity. This is the property the whole mechanism rests on.
truth = np.arange(len(ts))*0.0005/0.050
check("crossing k stamped at k*pitch/velocity, to the nanosecond",
      float(np.abs((ts - ts[0]) - truth).max()), 0.0, 2e-9)
check("timestamps strictly increasing",
      bool(np.all(np.diff(ts) > 0)), True)
check("positions still exactly on the pitch",
      float(np.abs(np.diff(xs) - 0.0005).max()), 0.0, 1e-12)
check("nothing fell back to an uninterpolated stamp",
      h.uninterpolated_timestamps, 0)
# A crossing landing exactly on the end of its poll needs no interpolation and
# is not counted as one, which is why this is not len(fired)-1 exactly.
check("effectively all of them were interpolated",
      h.interpolated_timestamps > 0.95*(len(h.fired) - 1), True)

print()
print("=== what the old behaviour did to the same pass ===")
# Stamping at poll time is what the code did before: reconstruct it by rounding
# every crossing time up to the end of the poll it was noticed in.
poll = 1.0/20.0
old = np.ceil((ts - ts[0])/poll + 1e-12)*poll
dup = len(old) - len(set(old))
print("    %d of %d crossings would have shared a timestamp with another"
      % (dup, len(old)))
check("the old scheme collapses crossings onto the same A-scan", dup > 0, True)
check("worst timestamp error was up to a whole poll interval",
      float(np.abs(old - (ts - ts[0])).max()) <= poll + 1e-9, True)
print("    worst error %.1f ms against a %.1f ms poll"
      % (1e3*float(np.abs(old - (ts - ts[0])).max()), 1e3*poll))

print()
print("=== a clock that cannot build a time falls back, and says so ===")
h = Harness(pitch=0.0005, length_m=0.02, velocity_mps=0.050, poll_hz=20.0,
            clock=FakeClock(supports_construction=False)).run()
check("nothing was interpolated", h.interpolated_timestamps, 0)
check("and the fallbacks were counted",
      h.uninterpolated_timestamps, len(h.fired) - 1)
check("crossings still fired, none lost", len(h.fired), expected_crossings(h))

print()
print("=== a stationary probe still does not trigger ===")
h = Harness(pitch=0.0005, length_m=0.0, velocity_mps=0.050, poll_hz=20.0)
h.polls = 50
h.velocity = 0.0
h.run()
check("only the arm-time trigger", len(h.fired), 1)

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("all timestamp checks passed")
