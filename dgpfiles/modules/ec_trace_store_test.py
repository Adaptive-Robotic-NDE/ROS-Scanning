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
# File:  ec_trace_store_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   A scan that dies mid-run leaves its completed passes on disk, and
#   those files alone rebuild the projected image.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-02:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""A scan that dies leaves its completed passes on disk, and they replay.

THE RUN THIS FILE IS ABOUT. On 2026-10-02 an eddy-current scan ran 59 of 64
passes over 28 minutes -- live projection working, real defect indications
visible -- and then wedged. The process was killed and all 59 passes went with
it, because an EC scan wrote nothing to disk at any point: the traces and their
poses lived only in the in-process recording database. The operator asked for
the run to be reprojected and there was nothing to reproject.

So the properties worth pinning are not "the writer works". They are:

  * a pass that COMPLETED is on disk before the next one starts, so a scan that
    dies at pass 60 leaves 59 passes;
  * failing to write must never cost a pass. The scan is mid-measurement with
    the probe on the part and the data is in the database either way;
  * ... but a Ctrl-C during the write must still stop the scan;
  * the files alone rebuild the image: for every trace, the same pose and the
    same value the live projection was handed;
  * and the conditioning rotation can be chosen AFTERWARDS, over the same
    traces. That is the whole reason the raw complex impedance is kept rather
    than only the conditioned scalar -- the operator's note from the lost run
    was that they had rotated the scan-line noise horizontal and the notches
    were then showing up as a vertical response, which is a decision made by
    eye while watching, and the only way to revisit it is from the raw value.

ON THE HOST, no snde, no ROS, no robot. The scan loop is the REAL
run_triggered_path_scan() driven over a three-pass path with a fake trigger; the
recorder is the REAL ECPassRecorder, lifted out of ec_trace_store.dpi by its
AST and handed a dict of arrays where the recording database would be -- which
is what it was given a `read_array` callable for.

WHAT IS NOT COVERED HERE. The projector itself: splatting onto a parameterized
surface needs x3d_load_geometry and a recording database, so "the same image"
is pinned as "the same (pose, value) offered for every trace", which is all the
projector is ever given. And the conditioning arithmetic is checked here against
the published formula; that it matches the COMPILED function trace for trace is
ec_replay_conditioning_test.py, which runs the real C++.

    python3 -m pytest -q ec_trace_store_test.py
"""
import ast
import json
import os
import shutil
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ec_trace_store as store

STORE_DPI = os.path.join(HERE, "ec_trace_store.dpi")
PLAN_DPI = os.path.join(HERE, "scan_plan.dpi")


# ==============================================================================
# Lifting the real code out of the .dpi files
# ==============================================================================

def _lift(path, names, ns=None):
    """Named top-level defs or classes out of one .dpi, into a namespace.

    A .dpi is compiled as a function body and is not importable, and this one
    opens by importing numpy and reaching for a recording database that does
    not exist on the host. Taking the pieces under test by AST keeps the test
    to them.
    """
    tree = ast.parse(open(path).read(), filename=os.path.basename(path))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            assert node.name not in found, "%s is defined twice in %s" % (
                node.name, os.path.basename(path))
            found[node.name] = node
    missing = [n for n in names if n not in found]
    assert not missing, "%s has no %s" % (os.path.basename(path),
                                          ", ".join(missing))
    ns = {} if ns is None else ns
    ns.setdefault("np", np)
    ns.setdefault("os", os)
    ns.setdefault("json", json)
    import time as _time
    ns.setdefault("time", _time)
    mod = ast.Module(body=[found[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), os.path.basename(path), "exec"), ns)
    return ns


def recorder_class():
    """The real ECPassRecorder, with the real ec_trace_store under it.

    `_store` is a free name in the class -- in the .dpi it is a local of the
    file's function body, reached through the method's closure -- so here it
    resolves out of the namespace, and the namespace hands it the genuine
    module. Nothing about the file format is stubbed.
    """
    ns = _lift(STORE_DPI, ["ECPassRecorder"], {"_store": lambda: store})
    return ns["ECPassRecorder"]


def scan_loop(say=None):
    """A namespace with the REAL triggered scan loop in it."""
    printed = [] if say is None else say
    ns = {"np": np,
          "print": lambda *a: printed.append(" ".join(str(x) for x in a))}
    exec("import threading, time, traceback\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n"
         "import numpy as np\n", ns)
    for name in ("toolpath_segments.dpi", "triggered_acquisition.dpi"):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                        for l in src.splitlines())
        src = src.replace("from geometry_msgs.msg import PoseArray, Pose",
                          "pass  # ROS message types, unused here")
        exec(compile(src, name, "exec"), ns)
    return ns, printed


# ==============================================================================
# A scan with no robot, no instrument and a dict where the database goes
# ==============================================================================

#: The conditioning the fake scan runs with. Not the identity: a test that
#: conditions with null (0,0), rotation 0 and gain (1,1) cannot tell a replay
#: that applies the parameters from one that ignores them.
SCAN_NULL = (0.37, -0.12)
SCAN_ROTATION = 31.0
SCAN_GAIN = (1.0, 2.5)
SCAN_X3D = "/data/scandata/specimen_keyence_clean_uv.x3d"
PITCH = 0.002


class FakeAccumulator(object):
    """The three things ECPassRecorder asks a TraceAccumulator for.

    ALLOCATED FOR THE WHOLE SCAN AND FILLED AS IT GOES, which is what the real
    one does: nd_accumulate_once is instantiated with K slots and `staged`
    counts the ones that mean anything. A recorder that trusted the array length
    instead would save the unwritten tail as zeros at an invalid pose.
    """

    traces_path = "/ec/traces"
    poses_path = "/ec/trace_poses"

    def __init__(self, K):
        self.K = int(K)
        self.staged = 0
        self.traces = np.zeros((self.K, 1), dtype=np.complex128)
        self.poses = np.zeros((self.K, 7), dtype=np.float32)
        self.lock = threading.Lock()

    def stage(self, z, pose7):
        with self.lock:
            k = self.staged
            if k >= self.K:
                return None
            self.traces[k, 0] = z
            self.poses[k, :] = pose7
            self.staged = k + 1
            return k


class FakeDatabase(object):
    """recdb.latest.ref[path].data, as a dict and one callable.

    The conditioned channel is FLOAT32, because the compiled conditioning emits
    float32 scalars and that is the precision the live projection was handed.
    Making it float64 here would hide the only real difference between the live
    value and a replay that recomputes in double.
    """

    def __init__(self, accum, cond_path="/ec/cond/vertical"):
        self.accum = accum
        self.cond_path = cond_path

    def conditioned(self):
        vals = store.component_values(self.accum.traces.reshape(-1), "vertical",
                                      null=SCAN_NULL,
                                      rotation_deg=SCAN_ROTATION,
                                      gain=SCAN_GAIN)
        return vals.astype(np.float32)

    def read(self, path):
        if path == self.accum.traces_path:
            return self.accum.traces
        if path == self.accum.poses_path:
            return self.accum.poses
        if path == self.cond_path:
            return self.conditioned()
        raise KeyError("no channel %r" % (path,))


class FakeSource(object):
    """A position trigger that fires its whole armed budget on demand."""

    trigger_mode = "timestamped-ec"
    trigger_pitch_m = PITCH

    def __init__(self):
        self.travelled_m = 0.0
        self.fault = None
        self.armed_over_budget = 0
        self._sink = None
        self._budget = 0
        self.index = 0

    def arm(self, sink, emit_on_arm=True, max_triggers=None):
        self._sink = sink
        self._budget = 0 if max_triggers is None else int(max_triggers)

    def disarm(self):
        self._sink = None

    def synthesize(self, *args):
        # A reading that is different for every trace, so a replay that mixed
        # two traces up would be visible rather than lucky.
        self.index += 1
        return np.array([0.4 + 0.01*self.index
                         + 1j*(-0.3 + 0.013*self.index)],
                        dtype=np.complex128)

    def metadata(self):
        return None

    def trigger_report(self):
        return {"mode": "timestamped-ec", "pitch_m": PITCH,
                "triggers": self.index}

    def crossings(self, pitch_m):
        for k in range(self._budget):
            self.travelled_m += pitch_m
            self._sink(self.synthesize(), None, (k + 1)*pitch_m, k)


def straight_path(ns, passes=3, per_pass=5, pitch_m=PITCH):
    pos, starts = [], []
    for p in range(passes):
        starts.append(len(pos))
        for k in range(per_pass):
            pos.append([0.80 + k*pitch_m, -0.30 + 0.01*p, 0.95])
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    return ns["ScanPath"](np.array(pos), quats, starts, source="test path")


def run_fake_scan(run_dir, passes=3, per_pass=5, die_at=None, saver=None,
                  cond_path="/ec/cond/vertical", wire_after_first=True):
    """Drive the real scan loop with the real recorder. -> dict

    `die_at` makes the sweep of that pass (1-based) raise, which is what a
    pendant dropping External Control or a killed node looks like from in here:
    the scan stops mid-run and whatever is on disk is what there is.

    `wire_after_first` mirrors the real ordering -- the conditioning does not
    exist until the first reading has been staged, so the recorder must not
    have a conditioned channel for the first pass's save if that save happened
    first. It does not: the save happens at the END of the pass, by which time
    setup_after_first() has run. Pass False to check the recorder survives
    having no conditioning at all.
    """
    said = []
    ns, printed = scan_loop(said)
    path = straight_path(ns, passes=passes, per_pass=per_pass)
    segs = path.acquiring_segments(min_length_m=0.001)
    source = FakeSource()
    accum = FakeAccumulator(K=passes*per_pass + 8)
    db = FakeDatabase(accum, cond_path=cond_path)
    wired = {"on": False}
    live = []              # what the live projection was offered, in order
    lock = threading.Lock()

    def approach(idx, end_idx):
        return True

    def sweep(start, end):
        if die_at is not None and (start // per_pass) + 1 == die_at:
            raise RuntimeError("External Control dropped")
        source.crossings(PITCH)
        return True

    def stage(ascan, meta, pose):
        with lock:
            k = accum.stage(complex(np.asarray(ascan).reshape(-1)[0]), pose)
            if k is None:
                return
            # THE LIVE PROJECTION, exactly as stage_ec_reading() does it: the
            # CONDITIONED value for this trace, at the pose it was taken at,
            # read back off the math channel after the transaction.
            if wired["on"]:
                live.append((np.array(pose, dtype=np.float32),
                             float(db.conditioned()[k])))

    def pose_at(seg, s_m, xy):
        start, end = seg
        i = min(start + int(round(s_m/PITCH)), end - 1)
        p = path.positions[i]
        return np.array([1.0, 0.0, 0.0, 0.0, p[0], p[1], p[2]],
                        dtype=np.float32)

    def setup_after_first():
        wired["on"] = bool(wire_after_first)

    def state():
        return {"accum": accum,
                "cond_paths": ({"vertical": cond_path} if wired["on"] else None),
                "component": "vertical",
                "null": SCAN_NULL, "rotation_deg": SCAN_ROTATION,
                "gain": SCAN_GAIN,
                "x3d": SCAN_X3D, "mesh": "/data/scandata/specimen_keyence.ply",
                "pitch_m": PITCH,
                "projection_radius_m": 0.0015, "projection_pixels": 1024,
                "trigger_report": source.trigger_report,
                "label": "eddy-current path scan"}

    if saver is None:
        saver = recorder_class()(run_dir, db.read, state, say=said.append)

    error = None
    report = None
    try:
        report = ns["run_triggered_path_scan"](
            path, source, PITCH, approach, sweep, stage, pose_at,
            setup_after_first=setup_after_first, segments=segs,
            on_segment_failure="stop", on_pass=saver)
    except BaseException as e:                                # noqa: BLE001
        error = e
    return {"report": report, "error": error, "said": said, "printed": printed,
            "saver": saver, "accum": accum, "db": db, "live": live,
            "passes": passes, "per_pass": per_pass}


# ==============================================================================
# 1. A completed pass is on disk, exactly as it was acquired
# ==============================================================================

def test_a_written_pass_round_trips_to_the_same_arrays(tmp_path):
    """Bit for bit, not approximately.

    The pass file IS the measurement once the process is gone, so a lossy write
    is a corrupted measurement that nothing would ever report.
    """
    out = run_fake_scan(str(tmp_path), passes=1, per_pass=6)
    assert out["error"] is None
    assert out["saver"].saved == 1, out["said"]

    run = store.read_run(str(tmp_path))
    accum = out["accum"]
    n = accum.staged
    assert run["traces"] == n
    assert np.array_equal(run["impedance"],
                          accum.traces.reshape(-1)[:n].astype(np.complex128))
    assert np.array_equal(run["pose"], accum.poses[:n].astype(np.float64))
    assert np.array_equal(run["position"], accum.poses[:n, 4:7].astype(np.float64))
    assert np.array_equal(run["index"], np.arange(n))


def test_the_pass_file_carries_what_a_reprojection_needs(tmp_path):
    """A replay must need these files and the .x3d and nothing else.

    Any of these missing sends somebody back to a process that no longer
    exists: the surface, which decides where a trace lands; the conditioning
    parameters, without which the saved scalar cannot be reproduced or
    reconsidered; the pitch, which is what makes s_m mean anything; and the
    projector's own settings, without which the replayed image is a different
    resolution from the one the operator was watching.
    """
    run_fake_scan(str(tmp_path), passes=2, per_pass=4)
    run = store.read_run(str(tmp_path))
    assert run["x3d"] == SCAN_X3D
    assert run["component"] == "vertical"
    assert run["null"] == SCAN_NULL
    assert run["rotation_deg"] == SCAN_ROTATION
    assert run["gain"] == SCAN_GAIN
    assert run["pitch_m"] == PITCH
    assert run["projection_radius_m"] == 0.0015
    assert run["projection_pixels"] == 1024
    assert run["conditioned"] is not None
    first = run["passes"][0]
    assert first["mesh"].endswith("specimen_keyence.ply")
    assert first["trigger_report"]["mode"] == "timestamped-ec"
    assert first["poses_channel"] == "/ec/trace_poses"


def test_each_pass_is_its_own_slice_with_no_gap_and_no_overlap(tmp_path):
    """The trace list is one flat array; a pass is a slice of it.

    Getting the bookkeeping wrong is not a visible error -- it is a file that
    re-saves the previous pass, or skips a trace, and the replayed image is
    subtly wrong in a way no count would show.
    """
    out = run_fake_scan(str(tmp_path), passes=4, per_pass=5)
    files = store.pass_files(str(tmp_path))
    assert len(files) == 4
    seen = []
    for path in files:
        p = store.read_pass(path)
        lo = p["meta"]["trace_index_first"]
        hi = p["meta"]["trace_index_last"]
        assert hi - lo + 1 == p["meta"]["traces"] == len(p["pose"])
        seen.append((lo, hi))
    assert seen[0][0] == 0
    for (lo0, hi0), (lo1, _hi1) in zip(seen, seen[1:]):
        assert lo1 == hi0 + 1, seen
    assert seen[-1][1] + 1 == out["accum"].staged


def test_a_scan_that_dies_at_pass_three_leaves_two_passes_on_disk(tmp_path):
    """THE WHOLE POINT. 59 of 64 passes must not be worth nothing.

    The pass that was interrupted is genuinely lost -- it was never finished and
    its traces are not a pass -- but every pass that COMPLETED is on disk, which
    is the difference between a run to look at and a run to repeat.
    """
    out = run_fake_scan(str(tmp_path), passes=5, per_pass=4, die_at=3)
    assert isinstance(out["error"], RuntimeError), out["error"]

    run = store.read_run(str(tmp_path))
    assert len(run["files"]) == 2
    # Two complete passes: the pass-start trace plus one per crossing.
    assert run["traces"] == 2*4, run["traces"]
    assert out["saver"].saved == 2
    assert out["saver"].errors == 0


def test_flush_rescues_the_pass_that_was_in_flight(tmp_path):
    """An abort stops mid-sweep, and that pass's traces are nobody's otherwise.

    The per-pass hook fires only on a pass that FINISHED, so a Ctrl-C or a
    pendant dropping External Control part way along pass 3 leaves pass 3's
    traces unsaved. They are real measurements at real poses. flush() is what
    the eddy-current scan's abort path calls; it cannot help a process that is
    killed outright -- nothing inside the process can -- but both EC scans of
    2026-10-01 ended as aborts, which it does cover.

    Marked partial, because they are not a line: the coverage is whatever the
    arm managed.
    """
    out = run_fake_scan(str(tmp_path), passes=5, per_pass=4, die_at=3)
    assert isinstance(out["error"], RuntimeError)
    saver = out["saver"]
    before = saver.traces_saved
    assert saver.saved == 2

    saved, why = saver.flush()
    assert saved, why
    assert saver.saved == 3
    assert saver.traces_saved > before

    run = store.read_run(str(tmp_path))
    assert run["traces"] == out["accum"].staged
    partial = [p for p in run["passes"] if p.get("partial")]
    assert len(partial) == 1
    assert partial[0]["pass_index"] == 2
    # ... and a second flush has nothing left to do rather than duplicating it.
    again, why = saver.flush()
    assert not again and "no new traces" in why


def test_the_unwritten_tail_of_the_trace_list_is_never_saved(tmp_path):
    """The accumulators are sized for the whole scan; only `staged` is real.

    Saving the allocated tail would put zero-valued traces at a zero
    quaternion into the file, and a replay would offer them to the projector.
    """
    out = run_fake_scan(str(tmp_path), passes=2, per_pass=3)
    run = store.read_run(str(tmp_path))
    assert run["traces"] == out["accum"].staged < out["accum"].K
    # A zero quaternion is "never staged"; none may be in the file.
    assert np.all(np.linalg.norm(run["pose"][:, :4], axis=1) > 1e-6)


# ==============================================================================
# 2. Failing to write must never cost a pass -- and Ctrl-C must still stop
# ==============================================================================

class ExplodingRecorder(object):
    """A per-pass hook that always raises, like a full or read-only disk."""

    def __init__(self):
        self.calls = 0

    def __call__(self, index, report=None):
        self.calls += 1
        raise IOError("[Errno 28] No space left on device")


def test_a_save_failure_does_not_propagate_out_of_the_scan_loop(tmp_path):
    """A failure to save is a warning. The measurement is already acquired.

    This is the inverse of the bug being fixed: making a scan survivable must
    not itself become a new way to lose one. The probe is on the part, the
    traces are in the recording database, and the scan has more passes to fly.
    """
    boom = ExplodingRecorder()
    out = run_fake_scan(str(tmp_path), passes=4, per_pass=3, saver=boom)
    assert out["error"] is None, out["error"]
    report = out["report"]
    assert boom.calls == 4
    # Every pass still flew and every trace was still staged.
    assert len(report["per_segment"]) == 4
    assert report["staged"] == 4*3
    assert len(report["pass_hook_errors"]) == 4
    assert "No space left" in report["pass_hook_errors"][0]["error"]
    # ... and it said so, rather than failing silently.
    assert any("per-pass hook failed" in line for line in out["printed"]), \
        out["printed"]


def test_the_recorders_own_errors_are_caught_before_the_loop_sees_them(tmp_path):
    """ECPassRecorder never raises either: it counts and reports.

    Belt and braces on purpose. The loop's try/except is the backstop; the
    recorder's own is what gives the operator a count and a last message
    readable from the prompt mid-scan, which the loop's report cannot do
    because the report is written after the last pass.
    """
    # A run_dir that is a FILE: makedirs raises NotADirectoryError on every
    # pass, which is as close to a read-only mount as a test can get portably.
    blocked = os.path.join(str(tmp_path), "a-file-not-a-directory")
    with open(blocked, "w") as fh:
        fh.write("not a directory\n")
    out = run_fake_scan(blocked, passes=3, per_pass=3)
    assert out["error"] is None, out["error"]
    assert out["report"]["staged"] == 9
    assert out["report"].get("pass_hook_errors") is None, \
        "the recorder let an error reach the scan loop"
    assert out["saver"].errors == 3
    assert out["saver"].saved == 0
    assert any("was NOT saved to disk" in s for s in out["said"]), out["said"]
    assert "NOTHING WAS WRITTEN" in out["saver"].describe()


class Interrupter(object):
    """A per-pass hook that raises KeyboardInterrupt, like a Ctrl-C."""

    def __init__(self):
        self.calls = 0

    def __call__(self, index, report=None):
        self.calls += 1
        raise KeyboardInterrupt()


def test_a_ctrl_c_during_the_save_still_stops_the_scan(tmp_path):
    """Exception, not BaseException.

    The operator stopping a scan must not be swallowed by the thing that was
    meant to make the scan survivable. Catching BaseException here would make a
    contact scan with the probe on the part unstoppable from the keyboard
    between passes.
    """
    stop = Interrupter()
    out = run_fake_scan(str(tmp_path), passes=4, per_pass=3, saver=stop)
    assert isinstance(out["error"], KeyboardInterrupt)
    assert stop.calls == 1


def test_a_pass_whose_save_failed_is_absorbed_by_the_next_one(tmp_path):
    """The index only advances on a successful write.

    A transient failure -- a full disk that is then cleared, a network mount
    that blinked -- must not punch a permanent hole in the record. The next
    pass's save covers the range the failed one did not.
    """
    said = []
    calls = {"n": 0}
    real_write = store.write_pass

    def flaky_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise IOError("the mount blinked")
        return real_write(*args, **kwargs)

    class FlakyStore(object):
        """The real store with one write made to fail."""

        def __getattr__(self, name):
            return getattr(store, name)

        write_pass = staticmethod(flaky_write)

    ns = _lift(STORE_DPI, ["ECPassRecorder"], {"_store": lambda: FlakyStore()})

    # Built by hand, so the flaky store is the one in play.
    accum = FakeAccumulator(K=64)
    db = FakeDatabase(accum)
    rec = ns["ECPassRecorder"](str(tmp_path), db.read,
                              lambda: {"accum": accum,
                                       "cond_paths": {"vertical": "/ec/cond/vertical"},
                                       "component": "vertical",
                                       "null": SCAN_NULL,
                                       "rotation_deg": SCAN_ROTATION,
                                       "gain": SCAN_GAIN, "x3d": SCAN_X3D,
                                       "pitch_m": PITCH},
                              say=said.append)
    for p in range(3):
        for k in range(4):
            accum.stage(complex(0.1*(4*p + k), 0.2), [1, 0, 0, 0, 0.8, 0.0, 0.95])
        rec(p)

    assert rec.saved == 2 and rec.errors == 1, (rec.saved, rec.errors)
    run = store.read_run(str(tmp_path))
    # Nothing is missing: 12 traces over two files, the second covering eight.
    assert run["traces"] == 12
    assert np.array_equal(run["index"], np.arange(12))


# ==============================================================================
# 3. The files alone rebuild the image
# ==============================================================================

def test_replay_offers_the_same_pose_and_value_for_every_trace(tmp_path):
    """"The same image" means the same (pose, value) stream.

    The projector is handed nothing else -- project(pose7, value) -- so if the
    stream matches, the image matches. Reconstructed from the files with no
    recording database in sight, which is the situation the replay exists for.

    The value tolerance is float32 and not zero on purpose: the compiled
    conditioning emits float32 scalars and the live projection was handed those,
    while the replay recomputes in double from the raw impedance. That
    difference is real, it is one part in ten million, and claiming exactness
    would be claiming something false.
    """
    out = run_fake_scan(str(tmp_path), passes=3, per_pass=6)
    assert out["error"] is None
    live = out["live"]
    assert len(live) > 0

    run = store.read_run(str(tmp_path))
    values = store.component_values(run["impedance"], run["component"],
                                    null=run["null"],
                                    rotation_deg=run["rotation_deg"],
                                    gain=run["gain"])
    # The first trace of pass one is staged BEFORE setup_after_first() wires the
    # conditioning, so the live projection never saw it -- exactly as on the
    # real scan. Line the two up on the saved trace index.
    assert len(values) == run["traces"]
    offset = run["traces"] - len(live)
    assert 0 <= offset <= 1, (run["traces"], len(live))

    for j, (pose, value) in enumerate(live):
        k = j + offset
        assert np.array_equal(run["pose"][k].astype(np.float32), pose), k
        assert abs(values[k] - value) <= 1e-6*max(1.0, abs(value)), (k, values[k], value)


def test_the_replays_conditioning_reproduces_the_scans(tmp_path):
    """The self-check the replay runs on itself, over a whole saved run."""
    run_fake_scan(str(tmp_path), passes=2, per_pass=8)
    run = store.read_run(str(tmp_path))
    chk = store.condition_matches_stored(run)
    assert chk["checked"] == run["traces"]
    assert chk["ok"], chk["why"]


def test_a_replay_with_no_conditioned_column_still_works(tmp_path):
    """A scan whose conditioning never got wired still leaves usable traces.

    The raw impedance and the pose are the measurement; the conditioned column
    is a convenience and a cross-check. A run that lost it -- the math never
    wired, the channel unreadable -- must still replay, with the parameters
    chosen at replay time.
    """
    out = run_fake_scan(str(tmp_path), passes=2, per_pass=4,
                        wire_after_first=False)
    assert out["error"] is None
    assert out["saver"].saved == 2
    run = store.read_run(str(tmp_path))
    assert run["conditioned"] is None
    assert run["traces"] == 8
    chk = store.condition_matches_stored(run)
    assert chk["checked"] == 0 and chk["ok"]
    vals = store.component_values(run["impedance"], "vertical",
                                  rotation_deg=40.0)
    assert np.all(np.isfinite(vals))


# ==============================================================================
# 4. The conditioning can be reconsidered afterwards
# ==============================================================================

def test_the_rotation_override_changes_the_result(tmp_path):
    """The operator's actual question: what does it look like rotated?

    If an override did not change the numbers, the whole reason for keeping the
    raw complex impedance would be gone and the offline path would only be able
    to redraw the decision somebody already made while the arm was moving.
    """
    run_fake_scan(str(tmp_path), passes=2, per_pass=8)
    run = store.read_run(str(tmp_path))

    as_scanned = store.component_values(
        run["impedance"], "vertical", null=run["null"],
        rotation_deg=run["rotation_deg"], gain=run["gain"])
    rotated = store.component_values(
        run["impedance"], "vertical", null=run["null"],
        rotation_deg=run["rotation_deg"] + 37.0, gain=run["gain"])

    assert not np.allclose(as_scanned, rotated)
    # And it is a ROTATION, not a rescaling: the magnitude is invariant.
    m0 = store.component_values(run["impedance"], "magnitude", null=run["null"],
                                rotation_deg=run["rotation_deg"],
                                gain=(1.0, 1.0))
    m1 = store.component_values(run["impedance"], "magnitude", null=run["null"],
                                rotation_deg=run["rotation_deg"] + 37.0,
                                gain=(1.0, 1.0))
    assert np.allclose(m0, m1)


def test_the_quoted_angle_is_the_angle_that_gets_flattened(tmp_path):
    """The sign convention, pinned on a case with an unambiguous answer.

    An instrument's rotation is quoted as WHERE THE FEATURE IS: "lift-off at 40
    degrees" means set 40, and the data is then turned by -40 to lay that
    feature on the horizontal axis. So a feature sitting on the VERTICAL axis is
    at +90, and rotation_deg=+90 is what flattens it -- rotation_deg=-90 turns
    it the other way and lands it on negative horizontal instead.

    A sign error here is invisible in every other test -- the image is still an
    image -- and it would mean an operator turning the dial the wrong way
    offline, with the data agreeing with them.
    """
    run_fake_scan(str(tmp_path), passes=1, per_pass=10)
    run = store.read_run(str(tmp_path))
    v = store.component_values(run["impedance"], "vertical", null=run["null"],
                               rotation_deg=0.0, gain=(1.0, 1.0))
    flattened = store.component_values(
        run["impedance"], "horizontal", null=run["null"], rotation_deg=90.0,
        gain=(1.0, 1.0))
    assert np.allclose(v, flattened, atol=1e-12)
    # ... and what WAS vertical is now zero, which is the point of doing it.
    gone = store.component_values(run["impedance"], "vertical", null=run["null"],
                                  rotation_deg=90.0, gain=(1.0, 1.0))
    h0 = store.component_values(run["impedance"], "horizontal", null=run["null"],
                                rotation_deg=0.0, gain=(1.0, 1.0))
    assert np.allclose(gone, -h0, atol=1e-12)
    # The other way round is the opposite sign, so the two are distinguishable.
    wrong_way = store.component_values(
        run["impedance"], "horizontal", null=run["null"], rotation_deg=-90.0,
        gain=(1.0, 1.0))
    assert np.allclose(wrong_way, -v, atol=1e-12)


def test_the_gain_override_scales_one_axis_only():
    """Per-axis gain, because after rotation the axes carry different things."""
    z = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex128)
    base = store.condition_traces(z, rotation_deg=20.0, gain=(1.0, 1.0))
    scaled = store.condition_traces(z, rotation_deg=20.0, gain=(1.0, 4.0))
    assert np.allclose(scaled["horizontal"], base["horizontal"])
    assert np.allclose(scaled["vertical"], 4.0*base["vertical"])


def test_the_conditioning_matches_the_published_formula():
    """null, THEN rotate by minus the quoted angle, THEN gain.

    Hand-derived from ec_condition_cpp.hpp rather than from this module, so the
    two are not checking each other. The ORDER is the part that matters: nulling
    after rotating turns the null point into a moving offset, which is the
    classic way to get a display that drifts as the rotation is adjusted, and no
    amount of agreement about magnitudes would catch it.
    """
    z = np.array([3.0 + 1.0j], dtype=np.complex128)
    null = (1.0, -0.5)
    rot = 30.0
    gain = (2.0, 3.0)

    re_ = 3.0 - 1.0
    im_ = 1.0 - (-0.5)
    th = -np.radians(rot)
    want_h = (re_*np.cos(th) - im_*np.sin(th))*2.0
    want_v = (re_*np.sin(th) + im_*np.cos(th))*3.0

    got = store.condition_traces(z, null=null, rotation_deg=rot, gain=gain)
    assert np.allclose(got["horizontal"], [want_h])
    assert np.allclose(got["vertical"], [want_v])
    assert np.allclose(got["magnitude"], [np.hypot(want_h, want_v)])
    assert np.allclose(got["conditioned"], [want_h + 1j*want_v])

    # Nulling first is NOT the same as nulling after -- the thing the order is
    # there for.
    wrong = store.condition_traces(z - complex(*null), null=(0.0, 0.0),
                                   rotation_deg=rot, gain=gain)
    assert np.allclose(wrong["vertical"], got["vertical"])
    swapped = store.condition_traces(z, null=(0.0, 0.0), rotation_deg=rot,
                                     gain=gain)
    assert not np.allclose(swapped["vertical"], got["vertical"])


def test_an_unknown_component_is_refused_rather_than_guessed():
    """A typo must not silently become "vertical"."""
    try:
        store.component_values(np.array([1 + 1j]), "verticle")
    except ValueError as e:
        assert "horizontal, vertical, magnitude" in str(e)
    else:
        raise AssertionError("a misspelled component was accepted")


# ==============================================================================
# 5. The files themselves: partial writes, corruption, and what is missing
# ==============================================================================

def test_a_half_written_file_is_never_offered_to_a_reader(tmp_path):
    """The write goes through a temporary and is renamed into place.

    A scan killed DURING a write would otherwise leave a truncated zip, np.load
    would raise on it, and read_run() would lose the whole run over the one
    pass that was in flight. The temporary is also named so that PASS_GLOB
    cannot match it.
    """
    d = str(tmp_path)
    run_fake_scan(d, passes=1, per_pass=3)
    # A completed write leaves the pass file and nothing else behind.
    assert sorted(os.listdir(d)) == [store.pass_filename(0)], os.listdir(d)
    # The name the writer uses while writing must not look like a pass.
    tmpname = "_writing_%s" % (store.pass_filename(7),)
    assert store.pass_index_of(tmpname) is None
    open(os.path.join(d, tmpname), "w").close()
    assert len(store.pass_files(d)) == 1


def test_one_corrupt_pass_does_not_cost_the_others(tmp_path):
    """Per-pass files exist precisely so that one bad file is one bad pass."""
    d = str(tmp_path)
    run_fake_scan(d, passes=4, per_pass=4)
    with open(os.path.join(d, store.pass_filename(2)), "wb") as fh:
        fh.write(b"PK\x03\x04 truncated rubbish")

    run = store.read_run(d)
    assert len(run["files"]) == 3
    assert run["traces"] == 3*4
    assert any("pass002.npz" in w for w in run["warnings"]), run["warnings"]


def test_a_file_with_no_poses_is_refused_where_it_can_be_explained(tmp_path):
    """A trace is an impedance AND the pose it was taken at.

    Neither is recoverable from the other, so a run missing either is not a
    partial measurement -- it is not one. Refused in read_run(), which can name
    the directory and say why, rather than coming apart inside the surface
    check with a TypeError about None.
    """
    d = str(tmp_path)
    z = np.array([1 + 1j, 2 + 1j])
    np.savez(os.path.join(d, store.pass_filename(0)), impedance=z,
             meta=json.dumps({"pass_index": 0}))
    try:
        store.read_run(d)
    except IOError as e:
        assert "a pose column" in str(e)
    else:
        raise AssertionError("a run with no poses was accepted")


def test_position_is_derived_when_a_file_does_not_carry_it(tmp_path):
    """It is a view of the pose, so it is never a reason to refuse a run."""
    d = str(tmp_path)
    poses = np.tile(np.array([1.0, 0, 0, 0, 0.81, -0.29, 0.95]), (2, 1))
    z = np.array([1 + 1j, 2 + 1j])
    np.savez(os.path.join(d, store.pass_filename(0)), pose=poses, impedance=z,
             meta=json.dumps({"pass_index": 0}))
    run = store.read_run(d)
    assert np.array_equal(run["position"], poses[:, 4:7])


def test_a_directory_with_no_passes_says_what_that_means(tmp_path):
    """Including that a scan can legitimately have written nothing yet."""
    try:
        store.read_run(str(tmp_path))
    except IOError as e:
        assert "one per completed pass" in str(e)
    else:
        raise AssertionError("an empty directory was read as a run")


def test_parameters_that_changed_mid_scan_are_reported_not_hidden(tmp_path):
    """The operator may re-null or turn the rotation between passes.

    That is legitimate -- it re-conditions everything already acquired, so the
    last setting is what the live image was showing -- but a replay has to say
    so rather than quietly picking one. The raw impedance is unaffected either
    way, which is why the run is still replayable.
    """
    d = str(tmp_path)
    poses = np.tile(np.array([1.0, 0, 0, 0, 0.8, 0.0, 0.95]), (3, 1))
    z = np.array([1 + 1j, 2 + 1j, 3 + 1j])
    store.write_pass(d, 0, poses, z, meta={"rotation_deg": 10.0, "x3d": "a.x3d"})
    store.write_pass(d, 1, poses, z, meta={"rotation_deg": 55.0, "x3d": "a.x3d"})
    run = store.read_run(d)
    assert run["rotation_deg"] == 55.0
    assert any("rotation_deg" in w for w in run["warnings"]), run["warnings"]


def test_runs_are_found_by_name_and_only_when_they_have_passes(tmp_path):
    """latest_run() is by directory NAME, which carries the scan's start time.

    By mtime it would be wrong in the one case that matters: copying files out
    of an older run touches its directory and would make that run the newest.
    """
    root = str(tmp_path)
    poses = np.tile(np.array([1.0, 0, 0, 0, 0.8, 0.0, 0.95]), (2, 1))
    z = np.array([1 + 1j, 2 + 1j])
    for name in ("ec_scan_2026-10-02_090000", "ec_scan_2026-10-02_143000"):
        store.write_pass(os.path.join(root, name), 0, poses, z, meta={})
    # Started and never written to: not a run.
    os.makedirs(os.path.join(root, "ec_scan_2026-10-02_150000"))
    # Not a scan directory at all.
    os.makedirs(os.path.join(root, "specimen_meshes"))

    runs = store.run_directories(root)
    assert [os.path.basename(r) for r in runs] == [
        "ec_scan_2026-10-02_090000", "ec_scan_2026-10-02_143000"]
    assert os.path.basename(store.latest_run(root)) == "ec_scan_2026-10-02_143000"

    # Touching the older one must not promote it.
    older = os.path.join(root, "ec_scan_2026-10-02_090000")
    shutil.copy(os.path.join(older, store.pass_filename(0)),
                os.path.join(str(tmp_path), "copied.npz"))
    os.utime(older, None)
    assert os.path.basename(store.latest_run(root)) == "ec_scan_2026-10-02_143000"


def test_the_run_directory_is_named_by_when_the_scan_started(tmp_path):
    """One directory per run, because the mesh names are NOT unique.

    Every run of the pipeline writes specimen_azure.ply and
    specimen_keyence.ply to the same place and each overwrites the last. Trace
    data must not work that way: a scan is the thing that cannot be repeated.
    """
    ns = _lift(STORE_DPI, ["ec_scan_run_dir"],
               {"_scan_work_dir": lambda: str(tmp_path)})
    a = ns["ec_scan_run_dir"](when=1790964000.0)
    b = ns["ec_scan_run_dir"](when=1790964900.0)
    assert a != b
    assert os.path.isdir(a) and os.path.isdir(b)
    assert os.path.basename(a).startswith("ec_scan_")


def test_nowhere_to_write_is_refused_loudly():
    """With no storage directory the scan exists only in memory. Say so."""
    ns = _lift(STORE_DPI, ["ec_scan_run_dir"], {"_scan_work_dir": lambda: None})
    try:
        ns["ec_scan_run_dir"]()
    except RuntimeError as e:
        assert "SCAN_STORAGE_DIR" in str(e)
        assert "2026-10-02" in str(e)
    else:
        raise AssertionError("a scan with nowhere to write its traces was allowed")


# ==============================================================================
# 6. The silent no-op: finish_projection() for an eddy-current scan
# ==============================================================================

def _projection_ns(**over):
    """scan_plan.dpi's two projection entry points, with their globals faked."""
    ns = {"PROJECTION_MODE": "deferred", "projector": None, "trace_accum": None,
          "cscan_gates": None, "recdb": None, "np": np}
    said = []
    ns["print"] = lambda *a: said.append(" ".join(str(x) for x in a))
    ns.update(over)
    _lift(PLAN_DPI, ["project_traces", "finish_projection"], ns)
    return ns, said


def test_finish_projection_was_a_silent_no_op_for_ec_and_is_not_now():
    """"deferred" used to mean "record the traces and never replay them".

    project_traces() and finish_projection() read `projector`, `trace_accum`
    and `cscan_gates` -- scan_accumulation.dpi's ULTRASOUND names, which an
    eddy-current scan never sets. So finish_projection() returned None on the
    first line and said nothing, and the operator got an empty image from a
    scan that had acquired perfectly good data.
    """
    ns, said = _projection_ns()
    assert ns["finish_projection"]() is None
    assert any("NOTHING WAS REPLAYED" in line for line in said), said
    assert any("no projector" in line for line in said), said


def test_an_instrument_with_its_own_replay_gets_used():
    """scan_replay_projection is how eddy current says "that one is mine"."""
    calls = []

    def replay(**kw):
        calls.append(kw)
        return {"projected": 1234}

    ns, said = _projection_ns(scan_replay_projection=replay)
    out = ns["finish_projection"]()
    assert out == {"projected": 1234}
    assert calls == [{}]
    assert not any("NOTHING WAS REPLAYED" in line for line in said), said

    # ... and project_traces() routes to it too, so the operator's habitual
    # call and the settings panel's re-project button both work.
    ns["project_traces"](first=5, count=10, quiet=True)
    assert calls[-1] == {"first": 5, "count": 10, "quiet": True}


def test_a_replay_that_raises_does_not_lose_the_scan():
    """A presentation failure must not read as a failed measurement."""
    def replay(**kw):
        raise RuntimeError("the conditioning is not wired")

    ns, said = _projection_ns(scan_replay_projection=replay)
    assert ns["finish_projection"]() is None
    assert any("DEFERRED PROJECTION DID NOT RUN" in line for line in said), said
    assert any("carries the pose it was taken at" in line for line in said), said


def test_live_mode_still_does_nothing_at_the_end():
    """Only "deferred" replays. Live mode has already drawn the image."""
    ns, said = _projection_ns(PROJECTION_MODE="live",
                              scan_replay_projection=lambda **kw: 1/0)
    assert ns["finish_projection"]() is None
    assert said == []


# ==============================================================================
# 7. The .dpi scoping rules, for the names this change adds
# ==============================================================================

def _globals_of(path):
    """Every name a .dpi declares `global`, read as the loader compiles it."""
    src = open(path).read()
    tree = ast.parse("def _dpi_body():\n" + "".join(
        "    " + l + "\n" for l in src.splitlines()))
    out = set()
    for node in ast.walk(tree.body[0]):
        if isinstance(node, ast.Global):
            out.update(node.names)
    return out


def test_the_new_names_are_exported_from_their_own_files():
    """A .dpi is a FUNCTION BODY: without `global` a name is invisible.

    dpi_exports_test.py catches a name USED across files and not exported, but
    these are reached through globals().get() and from the dgpy prompt, where
    the symptom is not a NameError -- it is None, and a caller quietly taking
    its fallback path. replay_ec_scan() missing its `global` would simply not
    exist at the prompt after a crash, which is the one moment it is wanted.
    """
    exported = _globals_of(STORE_DPI)
    for name in ("ECPassRecorder", "ec_scan_run_dir", "ec_pass_recorder",
                 "ec_save_status", "replay_ec_scan", "list_ec_runs",
                 "ec_run_dir", "ec_replay_projector", "ec_replay_report"):
        assert name in exported, "%s is not declared global in %s" % (
            name, os.path.basename(STORE_DPI))

    ec = _globals_of(os.path.join(HERE, "ec_acquisition.dpi"))
    for name in ("project_ec_traces", "scan_replay_projection", "ec_pass_saver",
                 "ec_conditioning_in_force"):
        assert name in ec, "%s is not declared global in ec_acquisition.dpi" % name


def test_the_config_loads_the_store_before_the_acquisition_that_drives_it():
    """Include order, because a .dpi that reads a name at load time needs it."""
    dgp = open(os.path.join(HERE, "..", "ECAutomatedScan.dgp")).read()
    store_at = dgp.index('"ec_trace_store.dpi"')
    acq_at = dgp.index('"ec_acquisition.dpi"')
    assert store_at < acq_at


def test_the_replay_loads_its_surface_through_the_same_gates_a_scan_does():
    """A mirrored parameterization is an empty image and NO error anywhere.

    x3d_winding.loadable_x3d() is the gate: it repairs or refuses a surface
    whose winding would make x3d_load_geometry ABORT THE PROCESS, and one whose
    UV parameterization is mirrored -- where the projector takes the square root
    of a negative determinant, gets a NaN, and deposits zero weight for every
    ray that lands. That gate is inside SurfaceProjector, so the replay has to
    go through SurfaceProjector rather than loading the x3d itself. Two runs
    were ruined by exactly this on the morning of 2026-10-02, which is why it
    is checked as a property of the file and not left to a reviewer.

    The same reasoning covers check_projection_covers_path(): a surface that is
    not under the poses paints nothing and reports nothing.
    """
    src = open(STORE_DPI).read()
    assert "SurfaceProjector(" in src, \
        "the replay must build its projector through SurfaceProjector"
    calls = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            fn = node.func
            calls.add(fn.id if isinstance(fn, ast.Name)
                      else getattr(fn, "attr", None))
    assert "x3d_load_geometry" not in calls, \
        "the replay must not load an x3d itself -- that skips loadable_x3d()"
    assert "check_projection_covers_path(" in src, \
        "the replay must check the surface is under the recorded poses"
    assert "projection_surface_reusable(" in src, \
        "a second replay must not strand another model tree"


def test_the_scan_loop_offers_a_per_pass_hook_and_documents_it():
    """The hook is the mechanism; a future reader has to find out why.

    Pinned because a parameter with no explanation is the first thing an
    unrelated cleanup removes, and what it costs is not visible until the next
    scan dies.
    """
    src = open(os.path.join(HERE, "triggered_acquisition.dpi")).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "run_triggered_path_scan"]
    assert len(fn) == 1
    assert "on_pass" in [a.arg for a in fn[0].args.args]
    doc = ast.get_docstring(fn[0]) or ""
    assert "on_pass" in doc
    assert "KeyboardInterrupt" in doc

    surf = open(os.path.join(HERE, "surface_path_scan.dpi")).read()
    assert "on_pass=on_pass" in surf, \
        "run_surface_path_scan does not pass the hook through"
    acq = open(os.path.join(HERE, "ec_acquisition.dpi")).read()
    assert "on_pass=ec_pass_saver" in acq, \
        "the eddy-current scan does not install its recorder"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
