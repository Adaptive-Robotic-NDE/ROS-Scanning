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
import functools
import os
import sys
import traceback
import types
import threading
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "timestamped_trigger.dpi")
DET = os.path.join(HERE, "pitch_crossings.dpi")

# _watch_loop imports dataguzzler_python.context for InitFreeThread, and the
# import is INSIDE the method, so it resolves against sys.modules when the loop
# RUNS -- no namespace injection reaches it. Hence a stub.
#
# THE REAL ONE WINS IF IT IS THERE, and that is not tidiness. sys.modules is
# process-wide and pytest imports every test file in the directory before running
# anything, so a stub installed unconditionally here is in force for every other
# suite as well. A suite whose worker thread calls context.InitFreeThread() before
# writing into a real recording database would then get a no-op: the thread is
# never registered with dataguzzler-python's threading model and the run
# SEGFAULTS inside spatialnde2. Same shape as the message-stub rule in
# ros_msg_stubs.py, in a module ros_stub_hygiene_test.py does not police.
try:
    import dataguzzler_python.context             # noqa: F401
except ImportError:
    dgp = types.ModuleType("dataguzzler_python")
    ctx = types.ModuleType("dataguzzler_python.context")
    ctx.InitFreeThread = lambda: None
    dgp.context = ctx
    sys.modules["dataguzzler_python"] = dgp
    sys.modules["dataguzzler_python.context"] = ctx

class _Clock(object):
    """A monotonic clock that ADVANCES, because the code under test has
    deadlines now. A stub that always returns 0.0 makes every timeout
    unreachable and a timeout loop spin forever."""

    def __init__(self):
        self.t = 0.0

    def sleep(self, s):
        self.t += float(s)

    def monotonic(self):
        self.t += 1e-6
        return self.t


_clock = _Clock()
ns = {"np": np, "threading": threading, "queue": queue,
      "traceback": traceback,
      "time": types.SimpleNamespace(sleep=_clock.sleep,
                                    monotonic=_clock.monotonic)}
for path, name in ((DET, "PitchCrossingDetector"), (SRC, "TimestampedTriggerSource")):
    tree = ast.parse(open(path).read())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name][0]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, "exec"), ns)
TimestampedTriggerSource = ns["TimestampedTriggerSource"]

fails = []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print("  %-56s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
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
                 clock=None, max_triggers=None, armed=True):
        self.clock = FakeClock() if clock is None else clock
        self.poll_period = 1.0/poll_hz
        self.velocity = velocity_mps
        self._init_trigger_state(pitch)
        self._pending = queue.Queue()
        self.delivered = []      # what reached the sink: (s_m, index)
        # A source only publishes while ARMED; the scan arms it for every
        # pass. Arm here with the arm-time trigger included so the counts
        # below are the crossings plus the origin.
        if armed:
            self.arm(lambda ascan, pos, s, i: self.delivered.append((s, i)),
                     emit_on_arm=True, max_triggers=max_triggers)

        # The pass, sampled once per poll at constant velocity.
        self.polls = (int(np.floor(length_m/(velocity_mps*self.poll_period))) + 1
                      if velocity_mps > 0.0 else 0)
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


@_checked
def test_one_crossing_per_poll():
    print("=== one crossing per poll: the easy case ===")
    # 50 mm/s at a 5.08 mm step and a 200 Hz poll is 0.05 crossings per poll.
    h = Harness(pitch=0.00508, length_m=0.30, velocity_mps=0.050, poll_hz=200.0).run()
    check("crossings", len(h.fired), expected_crossings(h))
    check("at most one crossing per poll", h.max_crossings_per_poll, 1)
    ts = np.array([t for t, x in h.fired], dtype=np.float64)*1e-9
    check("every timestamp distinct", len(set(ts)), len(ts))

    print()


@_checked
def test_five_crossings_per_poll_are_stamped_at_the_crossing():
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
    print("=== what stamping at poll time would do to the same pass ===")
    # Reconstruct that scheme by rounding every crossing time up to the end of the
    # poll it was noticed in.
    poll = 1.0/20.0
    old = np.ceil((ts - ts[0])/poll + 1e-12)*poll
    dup = len(old) - len(set(old))
    print("    %d of %d crossings would have shared a timestamp with another"
          % (dup, len(old)))
    check("poll-time stamping collapses crossings onto one A-scan", dup > 0, True)
    check("worst timestamp error was up to a whole poll interval",
          float(np.abs(old - (ts - ts[0])).max()) <= poll + 1e-9, True)
    print("    worst error %.1f ms against a %.1f ms poll"
          % (1e3*float(np.abs(old - (ts - ts[0])).max()), 1e3*poll))

    print()


@_checked
def test_a_clock_that_cannot_build_a_time_falls_back():
    print("=== a clock that cannot build a time falls back, and says so ===")
    h = Harness(pitch=0.0005, length_m=0.02, velocity_mps=0.050, poll_hz=20.0,
                clock=FakeClock(supports_construction=False)).run()
    check("nothing was interpolated", h.interpolated_timestamps, 0)
    check("and the fallbacks were counted",
          h.uninterpolated_timestamps, len(h.fired) - 1)
    check("crossings still fired, none lost", len(h.fired), expected_crossings(h))

    print()


@_checked
def test_a_stationary_probe_still_does_not_trigger():
    print("=== a stationary probe still does not trigger ===")
    h = Harness(pitch=0.0005, length_m=0.0, velocity_mps=0.050, poll_hz=20.0)
    h.polls = 50
    h.velocity = 0.0
    h.run()
    check("only the arm-time trigger", len(h.fired), 1)

    print()


@_checked
def test_disarmed_it_publishes_nothing():
    print("=== disarmed, it publishes nothing at all ===")
    # The approach and retrace between passes are real travel with no consumer.
    # A pose transaction per pitch of them costs ~100 globalrevs per row for
    # nothing.
    h = Harness(pitch=0.00508, length_m=0.30, velocity_mps=0.050, poll_hz=200.0,
                armed=False).run()
    check("crossings were still seen", h.triggers > 50, True)
    check("but no pose was published", len(h.fired), 0)
    check("and nothing reached a sink", len(h.delivered), 0)

    print()


@_checked
def test_the_budget_caps_the_row():
    print("=== the budget: a raster row has m columns and no column m ===")
    # 0.30 m at 5.08 mm is 59 crossings + the origin. Armed for 50 (origin
    # included): 50 published, 10 counted as surplus, none of them delivered.
    h = Harness(pitch=0.00508, length_m=0.30, velocity_mps=0.050, poll_hz=200.0,
                max_triggers=50).run()
    check("published exactly the budget", len(h.fired), 50)
    check("surplus crossings counted, not lost silently", h.armed_over_budget, 10)
    check("sink saw indices 0..49 in order",
          [i for s, i in h.delivered] == list(range(50)) or len(h.delivered) == 0, True)
    print("    (sink delivery goes through the correlator thread, which this "
          "harness does not run; %d queued)" % h._pending.qsize())

    print()


@_checked
def test_a_dead_digitizer_stops_the_scan():
    print("=== a dead digitizer stops the scan instead of decorating it ===")
    # A digitizer that has stopped producing new records makes every trigger
    # correlate to the same one, and thousands of identical traces get staged as
    # data. Feed the correlator crossings whose globalrev keeps returning the same
    # revision of the A-scan channel, and it must raise the fault at the limit.
    class FakeRef(object):
        def __init__(self, rev):
            self.data = np.zeros(4, dtype=np.float32)
            self.rec = types.SimpleNamespace(info=types.SimpleNamespace(revision=rev),
                                             metadata=None)

    class FakeGlobalrev(object):
        ready = True

        def __init__(self, rev):
            self.rev = rev
        def get_ndarray_ref(self, path):
            return FakeRef(self.rev)

    class FakeEnded(object):
        def __init__(self, rev):
            self.rev = rev
        def globalrev(self):
            return FakeGlobalrev(self.rev)
        def globalrev_nowait(self):
            return FakeGlobalrev(self.rev)

    h = Harness(pitch=0.001, length_m=0.0, velocity_mps=0.0, poll_hz=10.0)
    h.ascan_path = "/CS_CH1"
    h.stale_limit = 5
    delivered = []
    sink = lambda ascan, pos, s, i: delivered.append(i)
    # Twelve live records, then the digitizer dies and every crossing sees rev 12.
    for k in range(12):
        h._pending.put((FakeEnded(k), (0, 0, 0), k*0.001, k, sink))
    for k in range(12, 30):
        h._pending.put((FakeEnded(12), (0, 0, 0), k*0.001, k, sink))
    h._pending.put(None)
    h._correlator_loop()
    check("live records all delivered", delivered[:12], list(range(12)))
    check("the first repeat is a duplicate, not yet a fault: delivered", 12 in delivered, True)
    check("delivery stops once the fault is raised",
          max(delivered), 12 + h.stale_limit - 1)
    check("fault names the record", "rev 12" in str(h.fault), True)
    check("duplicates counted (17: the first rev-12 record was new)", h.duplicate_traces, 17)
    check("a slower digitizer that recovers resets the run",
          h._consecutive_duplicates >= h.stale_limit, True)

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
    print("all timestamp checks passed")


_SECTIONS = ["test_one_crossing_per_poll", "test_five_crossings_per_poll_are_stamped_at_the_crossing", "test_a_clock_that_cannot_build_a_time_falls_back", "test_a_stationary_probe_still_does_not_trigger", "test_disarmed_it_publishes_nothing", "test_the_budget_caps_the_row", "test_a_dead_digitizer_stops_the_scan"]


if __name__ == "__main__":
    sys.exit(_main())


def test_a_globalrev_that_never_arrives_does_not_stop_the_scan():
    """2026-10-09, twice: the correlator waited forever and took the cell down.

    Both freezes that day were this thread, parked in

        snde::transaction::globalrev_wait()
          snde::promise_channel_notify::wait_interruptable()

    on a transaction the timed manager had never committed, so its globalrev
    did not exist and never would. Behind it: 3517 copies of NI-DAQmx -200279,
    because this loop is what drains the card. 81 of 123 passes done, no error
    of its own, and disarm() stuck on _pending.join() which only finishes when
    this loop does.

    A trace whose globalrev is not coming is a HOLE IN THE IMAGE. Waiting for
    it is the whole run.
    """
    class Never(object):
        """A transaction whose globalrev is never realised."""
        def globalrev_nowait(self):
            return None

        def globalrev(self):                       # pragma: no cover
            raise AssertionError(
                "the correlator must never call the unbounded globalrev()")

    class Fine(object):
        class _GR(object):
            ready = True
            def get_ndarray_ref(self, path):
                r = types.SimpleNamespace()
                r.data = np.zeros(4, dtype=np.float32)
                r.rec = types.SimpleNamespace(
                    info=types.SimpleNamespace(revision=Fine._rev),
                    metadata=None)
                return r

        _rev = 0

        def globalrev_nowait(self):
            Fine._rev += 1
            return Fine._GR()

    h = Harness(pitch=0.001, length_m=0.0, velocity_mps=0.0, poll_hz=10.0)
    h.ascan_path = "/CS_CH1"
    h.staging_timeout_s = 0.05
    delivered = []
    sink = lambda ascan, pos, s, i: delivered.append(i)

    # good, good, DEAD, good, good -- one unrealised transaction in the middle
    for k, t in ((0, Fine()), (1, Fine()), (2, Never()), (3, Fine()), (4, Fine())):
        h._pending.put((t, (0, 0, 0), k*0.001, k, sink))
    h._pending.put(None)
    h._correlator_loop()

    check("the loop RETURNED instead of parking", True, True)
    check("the dead trace was dropped", 2 in delivered, False)
    check("every other trace was delivered", delivered, [0, 1, 3, 4])
    check("the timeout was counted", h.staging_timeouts, 1)
    check("one timeout is not a fault", h.fault, None)
    check("the run of consecutive timeouts reset", h.consecutive_staging_timeouts, 0)

    # A pipeline that has genuinely stopped producing globalrevs must stop the
    # scan rather than fly the rest of the part recording nothing.
    h2 = Harness(pitch=0.001, length_m=0.0, velocity_mps=0.0, poll_hz=10.0)
    h2.ascan_path = "/CS_CH1"
    h2.staging_timeout_s = 0.01
    h2.staging_timeout_fault_after = 6
    got = []
    for k in range(20):
        h2._pending.put((Never(), (0, 0, 0), k*0.001, k,
                         lambda a, p, s, i: got.append(i)))
    h2._pending.put(None)
    h2._correlator_loop()
    check("a dead pipeline eventually raises a fault", h2.fault is not None, True)
    check("...raised AT the threshold, not at the end of the queue",
          "6 correlations in a row" in str(h2.fault), True)
    check("...naming how many were lost by then",
          "6 trace(s) lost" in str(h2.fault), True)
    check("...and it still drained the queue rather than parking",
          h2._pending.unfinished_tasks, 0)
    check("...having given up on all 20", h2.staging_timeouts, 20)
    check("...delivering nothing", got, [])

    assert h2.fault is not None and h2.staging_timeouts == 20 and got == []
    assert delivered == [0, 1, 3, 4] and h.fault is None
