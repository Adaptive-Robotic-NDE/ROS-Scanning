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
# File:  calibration_resilience_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that a calibration run survives failed poses and leaves a
#   dataset that can be solved offline.
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
"""A calibration run finishes, or leaves a dataset that can finish without it.

A Keyence calibration run is 26-34 poses and 20-40 minutes of arm time, and a
single failed move used to end it in the middle with nothing solved. Two
distinct failures do that:

  * a pose refused before any motion -- the cable guard rejecting it by a
    couple of degrees;
  * a pose that plans partway and does not run -- typically a long straight
    line that also turns the wrist through most of a revolution.

Catching CaptureRefused alone covers only the first: CaptureMotionFailed is a
RuntimeError and NOT a CaptureRefused, so a move refused BEFORE MOVING costs
its pose while a move that RAN AND DID NOT COMPLETE costs the whole run.

This file pins the required behaviour against a cell that does not exist. Five
separate claims, and they fail independently:

  * a pose whose move fails costs THAT POSE and the run goes on;
  * a retry takes a DIFFERENT ROUTE (joint space), not the same Cartesian
    request again -- a replan of an identical request fails identically;
  * CaptureStopped and CaptureAbort END THE FLYING but the poses already in
    hand are still solved and saved;
  * every pose is ON DISK before the next move starts, so a process that dies
    outright still leaves a solvable dataset;
  * a surviving set that CANNOT determine the transform is still refused --
    resilience is "finish and tell the truth", not "always emit a number".

Runs on the host: the .dpi functions are lifted out by AST and executed against
stubs, so there is no ROS, no recorder, no robot and no cell. The synthetic
profiler and its artifact come from calibration_test.py, which is where the
arithmetic they exercise is already pinned.

    python3 calibration_resilience_test.py          (or pytest)
"""
import ast
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_solve as cs                                  # noqa: E402
import calibration_test as ct                                   # noqa: E402


# ------------------------------------------------------------------------------
# The exception classes, with the real hierarchy
# ------------------------------------------------------------------------------
# NOT SIMPLIFIED. CaptureMotionFailed really is a plain RuntimeError and
# CaptureStopped really does subclass CaptureRefused; test_the_exception_
# hierarchy_is_what_this_file_assumes checks that against the source. If these
# were flattened here, every other test in the file would pass for the wrong
# reason.

class Refused(Exception):
    """calibration.dpi's CalibrationRefused."""


class CaptureAbort(Exception):
    """The operator said stop."""


class CaptureSkip(Exception):
    """The operator skipped one step."""


class CaptureRefused(RuntimeError):
    """A motion broke a rule and was NOT run."""


class CaptureMotionFailed(RuntimeError):
    """A motion WAS run and did not complete."""


class CaptureStopped(CaptureRefused):
    """The monitor stopped the arm mid-move."""


class CableRefused(RuntimeError):
    """Outside the measured cable envelope."""


# ------------------------------------------------------------------------------
# A cell that does not exist
# ------------------------------------------------------------------------------

#: A BARE GROUND FLAT, which is what CAL_ARTIFACT = "flat" and an empty
#: CAL_BALL_CENTRES give -- the artifact the work cell actually calibrates
#: against. It matters here and not only for realism: calibrate_keyence()
#: derives the artifact's starting pose from a PLANE FIT, whose in-plane
#: rotation is arbitrary, and a plate with balls on it cannot be started that
#: way.
ARTIFACT = cs.PlateArtifact(np.zeros((0, 3)), 0.0127, (0.13, 0.10),
                            "bare flat")

#: Eight tilts at three standoffs is twenty-four poses. The eight are
#: calibration_test.py's, which are inside this cell's measured cable envelope,
#: and three standoffs are what makes a bare flat determine all six degrees of
#: freedom (CAL_KEYENCE_STANDOFFS, and the test that pins it).
TILTS = list(ct.KEY_TILTS)
STANDOFFS = (-0.015, 0.0, 0.015)
AIMS = [[0.0, 0.0, 0.0], [0.03, -0.02, 0.0]]


class Cell(object):
    """The poses, what the profiler returns at each, and what a move does.

    `script` maps a pose label to a list of outcomes, one per ATTEMPT: an
    exception instance to raise, or "ok". Attempts run out into "ok", so a
    label with ["boom"] fails its straight line and succeeds on the retry.
    """

    def __init__(self, tmpdir, script=None, grab_fails=(), n_per_line=140,
                 standoffs=STANDOFFS):
        rng = np.random.default_rng(11)
        # STANDOFF INNERMOST, exactly as _cal_candidate_poses() now generates
        # them, so any prefix of the flight order spans all the rungs.
        self.poses, G = [], []
        for tx, ty, roll in TILTS:
            for d in standoffs:
                g = ct._sensor_poses(AIMS, [(tx, ty, roll)], rng,
                                     standoff=d)[len(self.poses) % len(AIMS)]
                G.append(g)
                self.poses.append({
                    "label": "keyence t%+d,%+d r%+.0f d%.0f"
                             % (tx, ty, roll, d*1e3),
                    "xyz": [float(v) for v in g[:3, 3]],
                    "quat": tuple(float(v) for v in cs.R_to_quat(g[:3, :3])),
                    "tilt": (float(tx), float(ty)), "roll": float(roll),
                    "distance": float(d), "aim": [0.0, 0.0, 0.0],
                    "joints": [0.1*len(self.poses)]*6})
        self.G = G
        self.P = ct._profile_points(G, ct.KEY_TRUE, ct.W_TRUE, ARTIFACT, rng,
                                    n_per_line=n_per_line)
        self.index = {p["label"]: i for i, p in enumerate(self.poses)}
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.grab_fails = set(grab_fails)
        self.at = 0
        self.calls = []
        self.saved = []
        self.tmpdir = tmpdir

    # -- motion ------------------------------------------------------------
    def attempt(self, kind, label):
        self.calls.append((kind, label))
        plan = self.script.get(label)
        what = plan.pop(0) if plan else "ok"
        if what == "ok":
            self.at = self.index[label]
            return None
        raise what

    def flown(self):
        """The pose labels that a grab was actually taken at."""
        return [lb for k, lb in self.calls if k == "grab"]


def _lift(names, ns):
    """Execute the named top-level defs of calibration.dpi into `ns`.

    The .dpi is a function body when dgpy loads it, but its top-level defs are
    top-level statements either way, so one AST pass picks them out. Executing
    them into a namespace we control is what makes the WIRING testable without
    a cell -- which is the half a unit test of calibration_solve.py cannot see.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    body = [n for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in body}
    assert not missing, "calibration.dpi has no %s" % (sorted(missing),)
    exec(compile(ast.Module(body=body, type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    return ns


LIFTED = ("_cal_fly_to", "_cal_fly_or_refuse", "_cal_stop_why",
          "_cal_offline_hint", "_cal_safe_name", "_cal_label_standoff",
          "_cal_atomic_write", "_cal_dataset_start", "_cal_dataset_index",
          "_cal_dataset_add", "_cal_dataset_finish", "_cal_dataset_load",
          "_cal_keyence_solve", "calibrate_keyence", "solve_keyence_from_take",
          "_cal_thin_for_solve", "_cal_solve_progress")


def _ns(cell, x0_error=None):
    """calibration.dpi's keyence path, bound to `cell` instead of to a robot."""
    said = []

    class Take(object):
        def __init__(self):
            self.dir = os.path.join(cell.tmpdir, "take")
            os.makedirs(self.dir, exist_ok=True)
            self.results = {}
            self.events = []

        def event(self, **kw):
            self.events.append(kw)
            return kw

        def keep_file(self, path, name=None):
            return None

    take = Take()

    @contextlib.contextmanager
    def capture_scenario(scenario, **kw):
        """The real one swallows CaptureAbort and re-raises everything else."""
        try:
            yield take
        except CaptureAbort:
            take.results["aborted"] = True

    X0 = ct.KEY_TRUE.dot(cs.se3_exp(
        x0_error if x0_error is not None
        else [0.004, -0.003, 0.004, 0.020, -0.015, 0.012]))

    def _cal_tf(target, source, timeout=5.0):
        if (target, source) == ("world", "probe_end"):
            return np.asarray(cell.G[cell.at], float)
        if (target, source) == ("probe_end", "lj_v7080_optical_frame"):
            return X0.copy()
        raise Refused("no transform %s <- %s" % (target, source))

    def _cal_profile_points(topic, count=None, timeout=15.0, check=True):
        label = cell.poses[cell.at]["label"]
        cell.calls.append(("grab", label))
        if label in cell.grab_fails:
            raise Refused("NO SCATTER: the whole line came back one value.")
        return cell.P[cell.at], "lj_v7080_optical_frame", 60

    def write_calibration(sensor, res, frame, take=None):
        cell.saved.append((sensor, frame, res))
        return {"record": "/dev/null", "xacro": "/dev/null"}

    ns = {
        "np": np, "os": os, "sys": sys, "json": json, "math": math,
        "re": __import__("re"), "time": __import__("time"),
        "datetime": __import__("datetime"),
        "contextlib": contextlib,
        # -- the exception vocabulary, with the real hierarchy ---------------
        "CalibrationRefused": Refused, "CaptureAbort": CaptureAbort,
        "CaptureSkip": CaptureSkip, "CaptureRefused": CaptureRefused,
        "CaptureMotionFailed": CaptureMotionFailed,
        "CaptureStopped": CaptureStopped, "CableRefused": CableRefused,
        # -- settings --------------------------------------------------------
        "CAL_SETTLE_S": 0.0, "CAL_PROFILES_PER_POSE": 60,
        "CAL_SOLVE_POINTS_PER_POSE": 1500, "time": time,
        "CAL_KEYENCE_REF_DIST": 0.080, "CAL_KEYENCE_RANGE_M": (0.057, 0.103),
        "CAL_RECOVERY_STRIDE": 3, "CAL_ARTIFACT": "flat",
        "CAL_MOVE_ROUTES": 2, "CAPTURE_ROOT": cell.tmpdir,
        "CELL_NAME": "nowhere", "CELL_MODE": "sim",
        "_CAL": {},
        # -- the work cell
        # --------------------------------------------------------
        "_cal_geom": lambda: cs,
        "_cal_say": lambda m: said.append(str(m)),
        "cal_artifact": lambda kind=None: ARTIFACT,
        "sensor_topic": lambda *a, **k: "/profiles",
        "keyence_calibration_poses": lambda *a, **k: list(cell.poses),
        "_cal_dropped_line": lambda what: "",
        "capture_scenario": capture_scenario,
        "capture_move": lambda poses, label, **k: cell.attempt("cartesian",
                                                               label),
        "capture_move_joints": lambda q, label, **k: cell.attempt(
            "joint", label.split(" (")[0]),
        "_ik_target": lambda x, q: [0.0]*6,
        "_local_ik": contextlib.nullcontext,
        "_cal_tf": _cal_tf,
        "_cal_profile_points": _cal_profile_points,
        "write_calibration": write_calibration,
    }
    ns["__builtins__"] = __builtins__
    _lift(LIFTED, ns)
    ns["_said"] = said
    ns["_take"] = take
    ns["_X0"] = X0
    return ns


def _run(cell, **kw):
    ns = _ns(cell)
    try:
        res = ns["calibrate_keyence"](**kw)
        err = None
    except BaseException as e:                                # noqa: BLE001
        res, err = None, e
    return ns, res, err


def _error(X, truth=None):
    truth = ct.KEY_TRUE if truth is None else truth
    dt, dr = cs.pose_delta(np.asarray(X, float), truth)
    return dt*1e3, dr*1e3


@contextlib.contextmanager
def _tmp():
    d = tempfile.mkdtemp(prefix="cal_resilience_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------
# 1. The hole itself
# ------------------------------------------------------------------------------

def test_the_exception_hierarchy_is_what_this_file_assumes():
    """The bug WAS the hierarchy, so the hierarchy is pinned, not assumed.

    `except (CableRefused, CaptureRefused)` looked like it covered every way a
    move can fail. It did not: CaptureMotionFailed is a RuntimeError, so the
    one clause caught "refused before moving" and missed "ran and did not
    complete". If somebody later makes CaptureMotionFailed a CaptureRefused,
    every other test here would still pass while the distinction this file
    exists to draw quietly disappeared.
    """
    src = open(os.path.join(HERE, "cell_capture_node.dpi")).read()
    bases = {n.name: [b.id for b in n.bases if isinstance(b, ast.Name)]
             for n in ast.parse(src).body if isinstance(n, ast.ClassDef)}
    assert bases["CaptureMotionFailed"] == ["RuntimeError"], bases
    assert bases["CaptureRefused"] == ["RuntimeError"], bases
    assert bases["CaptureStopped"] == ["CaptureRefused"], bases
    assert bases["CaptureAbort"] == ["Exception"], bases


def test_no_calibration_pose_loop_calls_capture_move_directly():
    """Every pose in both calibrations goes through _cal_fly_to().

    A direct capture_move() in one of these loops rebuilds the failure: one
    route, no retry, and an exception that ends the run. Checked in the source
    because it is a property of the code and not of any one run.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    tree = ast.parse(src)
    for name in ("calibrate_keyence", "calibrate_azure", "verify_calibration"):
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        direct = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id in ("capture_move", "capture_move_joints")]
        assert not direct, "%s still moves the arm without a fallback" % name




# ------------------------------------------------------------------------------
# 2. One pose failing costs one pose
# ------------------------------------------------------------------------------

def _fail(label, n=3):
    """A label whose next `n` attempts all fail as a partial plan does."""
    return {label: [CaptureMotionFailed(
        "%s did not complete (planner reached 51%%)" % label)]*n}


def test_a_failed_move_costs_one_pose_and_not_the_run():
    """A pose that will not fly by any route.

    THE CLAIM IS NOT "it did not crash". It is that every OTHER pose is kept,
    solved, and produces the right transform, and that the lost one is named in
    the take's record where a human will find it.
    """
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[5]["label"]              # pose 6, as it happens
        cell.script = _fail(label)
        ns, res, err = _run(cell)
        assert err is None, err
        assert res is not None, "the run did not produce a calibration"
        assert len(cell.flown()) == len(cell.poses) - 1, cell.flown()
        assert ns["_take"].results["used"] == len(cell.poses) - 1
        missed = dict(ns["_take"].results["missed"])
        assert label in missed, missed
        assert "planner reached 51%" in missed[label], missed[label]
        mm, mrad = _error(res.X)
        assert mm < 0.5, "%.3f mm\n%s" % (mm, res.report())
        assert mrad < 3.0, "%.3f mrad\n%s" % (mrad, res.report())


def test_refused_before_moving_and_failed_while_moving_cost_the_same():
    """The asymmetry, asserted away as an equality.

    A CaptureRefused costing one pose while a CaptureMotionFailed costs the
    run is the asymmetry. There is no defensible reason for those to differ:
    both mean the arm is not at that pose.
    """
    outcomes = {}
    for name, exc in (("refused", CaptureRefused("outside the cable band")),
                      ("failed", CaptureMotionFailed("planner reached 51%")),
                      ("cable", CableRefused("+305.0 deg against +302.8")),
                      ("skipped", CaptureSkip("pose"))):
        with _tmp() as d:
            cell = Cell(d)
            label = cell.poses[5]["label"]
            cell.script = {label: [exc]*3}
            ns, res, err = _run(cell)
            outcomes[name] = (err, None if res is None else len(cell.flown()))
    assert all(e is None for e, _n in outcomes.values()), outcomes
    # one pose lost in every case, and the run finished in every case
    assert {n for _e, n in outcomes.values()} == {23}, outcomes


def test_a_bad_grab_still_only_costs_its_pose():
    """A pose that flew but came back a constant. Unchanged, and pinned here
    because the same loop now owns both kinds of loss."""
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[7]["label"]
        cell.grab_fails = {label}
        ns, res, err = _run(cell)
        assert err is None, err
        assert ns["_take"].results["used"] == len(cell.poses) - 1
        assert "NO SCATTER" in dict(ns["_take"].results["missed"])[label]


# ------------------------------------------------------------------------------
# 3. The retry is a DIFFERENT route
# ------------------------------------------------------------------------------

def test_a_retry_takes_a_different_route_and_recovers_the_pose():
    """The straight line fails; the joint move to the same endpoint works.

    Replanning the identical Cartesian request would be pointless --
    computeCartesianPath is deterministic given the same start state, so a leg
    that reached 51% reaches 51% again. The retry has to change something, and
    what it changes is the space the plan is made in.
    """
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[5]["label"]
        cell.script = {label: [CaptureMotionFailed("planner reached 51%")]}
        ns, res, err = _run(cell)
        assert err is None, err
        kinds = [k for k, lb in cell.calls if lb == label]
        assert kinds == ["cartesian", "joint", "grab"], kinds
        # nothing was lost: every pose contributed
        assert len(cell.flown()) == len(cell.poses), cell.flown()
        assert ns["_take"].results["missed"] == []


def test_the_retry_is_never_the_same_cartesian_request_again():
    """No route after the first is a Cartesian move.

    Stated over the CALLS rather than over the source, so it holds however
    _cal_fly_to() is rearranged: for any one pose there is at most one
    cartesian attempt.
    """
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[3]["label"]
        cell.script = {label: [CaptureMotionFailed("a"),
                               CaptureMotionFailed("b"),
                               CaptureMotionFailed("c")]}
        _run(cell)
        per_pose = {}
        for kind, lb in cell.calls:
            if kind != "grab":
                per_pose.setdefault(lb, []).append(kind)
        for lb, kinds in per_pose.items():
            assert kinds.count("cartesian") == 1, (lb, kinds)


def test_a_pose_with_no_ik_solution_says_so_and_is_not_retried_forever():
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[2]["label"]
        cell.script = {label: [CaptureRefused("no IK")]*3}
        ns = _ns(cell)
        ns["_ik_target"] = lambda x, q: None
        p = dict(cell.poses[2])
        p.pop("joints")
        out = ns["_cal_fly_to"](p)
        assert out["ok"] is False
        assert "no joint solution" in out["why"], out
        assert [k for k, lb in cell.calls if lb == label] == ["cartesian"]


# ------------------------------------------------------------------------------
# 4. A stop ends the flying, not the run
# ------------------------------------------------------------------------------

def test_capture_stopped_ends_the_flying_but_still_solves_and_saves():
    """The monitor stopped the arm. That is not a reason to lose the data.

    CaptureStopped subclasses CaptureRefused, so the old except clause
    SWALLOWED it and carried on to the next pose -- the one thing it must never
    do, because the arm was physically stopped for overspeed or overforce and a
    human has to look at the cell. It now ends the flying, the solve runs on
    what was collected, and the exception leaves afterwards.
    """
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[15]["label"]
        cell.script = {label: [CaptureStopped(
            "%s: over the force limit -- 41.2 N" % label)]*3}
        ns, res, err = _run(cell)
        assert isinstance(err, CaptureStopped), err
        # it stopped where it was told to
        assert len(cell.flown()) == 15, cell.flown()
        after = [lb for k, lb in cell.calls if k == "cartesian"]
        assert after[-1] == label, after
        # ... and the nine poses in hand were solved and saved
        assert ns["_CAL"]["keyence"] is not None
        assert cell.saved and cell.saved[0][0] == "keyence", cell.saved
        mm, _mrad = _error(ns["_CAL"]["keyence"].X)
        assert mm < 1.0, mm
        text = "\n".join(ns["_said"])
        assert "THE MONITOR STOPPED THE ARM" in text, text
        assert "NO MORE POSES WILL BE FLOWN" in text, text
        assert "SOLVED FROM 15 OF 24 POSES" in text, text


def test_an_operator_abort_keeps_what_was_collected():
    """Abandoning the REMAINING poses must not abandon the COLLECTED ones.

    capture_scenario() swallows CaptureAbort (the take is kept), so an abort
    comes back as a returned result rather than as an exception -- which is
    right: the operator stopped the run and still gets their calibration.
    """
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[12]["label"]
        cell.script = {label: [CaptureAbort(label)]*3}
        ns, res, err = _run(cell)
        assert err is None, err
        assert res is not None
        assert len(cell.flown()) == 12, cell.flown()
        assert ns["_take"].results["used"] == 12
        assert "ABORTED by the operator" in "\n".join(ns["_said"])
        # HALF THE POSE SET IS A WORSE CALIBRATION, and the bound says so
        # rather than pretending otherwise: twelve poses over four tilts get
        # to a few millimetres where twenty-four over eight get to a few
        # tenths (test_a_failed_move_costs_one_pose_and_not_the_run).
        # The claim here is that the run KEPT them, not that they are as good.
        mm, _mrad = _error(res.X)
        assert mm < 5.0, mm


def test_a_keyboard_interrupt_solves_before_it_leaves():
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[15]["label"]
        cell.script = {label: [KeyboardInterrupt()]*3}
        ns, res, err = _run(cell)
        assert isinstance(err, KeyboardInterrupt), err
        assert ns["_CAL"].get("keyence") is not None, "nothing was solved"
        assert cell.saved, "nothing was saved"
        assert "INTERRUPTED at the keyboard" in "\n".join(ns["_said"])


# ------------------------------------------------------------------------------
# 5. On disk as it is collected
# ------------------------------------------------------------------------------

def _poses_dir(take_dir):
    return os.path.join(take_dir, "keyence_poses")


def test_every_pose_is_on_disk_before_the_next_move_starts():
    """Checked BETWEEN moves, not at the end.

    The file has to exist while the arm is still flying, because the run this
    is for is the one that never reaches its end.
    """
    with _tmp() as d:
        cell = Cell(d)
        ns = _ns(cell)
        seen = []
        real_move = ns["capture_move"]

        def watched(poses, label, **k):
            pd = _poses_dir(ns["_take"].dir)
            seen.append(len([f for f in os.listdir(pd)
                             if f.endswith(".npz")]) if os.path.isdir(pd) else 0)
            return real_move(poses, label, **k)

        ns["capture_move"] = watched
        ns["calibrate_keyence"]()
        # before move k there are exactly k-1 pose files
        assert seen == list(range(len(cell.poses))), seen


def test_a_killed_run_leaves_a_dataset_that_solves_without_the_arm():
    """kill -9 in the middle: no combined file, no index close, no solve.

    Simulated by driving the collection by hand and simply never finishing it,
    which is what a killed process leaves behind. The per-pose files and the
    index that were written as it went are the whole of what survives, and they
    have to be enough.
    """
    with _tmp() as d:
        cell = Cell(d)
        ns = _ns(cell)
        data = ns["_cal_dataset_start"]("keyence", ns["_take"],
                                        len(cell.poses), header={})
        for i in range(15):
            cell.at = i
            ns["_cal_dataset_add"](data, {
                "label": cell.poses[i]["label"], "G": cell.G[i],
                "X_urdf": ns["_X0"], "points": cell.P[i],
                "distance": cell.poses[i]["distance"], "n_msgs": 60})
        # ... and here the process dies. No _cal_dataset_finish().
        pd = _poses_dir(ns["_take"].dir)
        assert not os.path.exists(
            os.path.join(pd, "keyence_calibration.npz")), "finish() ran"
        assert json.load(open(os.path.join(pd, "index.json")))["closed"] is False

        fresh = _ns(cell)                        # a new process, no state
        res = fresh["solve_keyence_from_take"](pd, save=False)
        mm, mrad = _error(res.X)
        assert mm < 1.0, "%.3f mm\n%s" % (mm, res.report())
        assert mrad < 5.0, "%.3f mrad\n%s" % (mrad, res.report())


def test_the_offline_solve_finds_the_dataset_from_the_take_directory():
    """The operator has the take path, not the pose path. Both work, and so
    does the combined npz on its own."""
    with _tmp() as d:
        cell = Cell(d)
        ns, res, err = _run(cell)
        assert err is None, err
        take_dir = ns["_take"].dir
        pd = _poses_dir(take_dir)
        combined = os.path.join(pd, "keyence_calibration.npz")
        assert os.path.exists(combined)
        for path in (take_dir, pd, combined, os.path.dirname(take_dir)):
            again = _ns(cell)["solve_keyence_from_take"](path, save=False)
            mm, _mrad = _error(again.X)
            assert mm < 0.5, (path, mm)


def test_the_combined_npz_is_the_shape_the_offline_tooling_reads():
    """dgpfiles/utils/cal_extract.py emits labels/G/X_urdf/counts/points out
    of a bag. A live run emits the same thing, so one solver reads both."""
    with _tmp() as d:
        cell = Cell(d)
        ns, res, err = _run(cell)
        z = np.load(os.path.join(_poses_dir(ns["_take"].dir),
                                 "keyence_calibration.npz"))
        for key in ("labels", "G", "X_urdf", "counts", "points"):
            assert key in z, sorted(z.keys())
        assert z["G"].shape == (len(cell.poses), 4, 4), z["G"].shape
        assert int(z["counts"].sum()) == z["points"].shape[0]


def test_a_dataset_with_no_recorded_standoff_recovers_it_from_the_label():
    """An npz lifted out of an old bag carries no standoff column. The spread
    is a GUARD, not a decoration, so it is recovered rather than skipped."""
    with _tmp() as d:
        cell = Cell(d)
        ns = _ns(cell)
        assert abs(ns["_cal_label_standoff"]("keyence t+0,-15 r+90 d-15")
                   + 0.015) < 1e-9
        assert ns["_cal_label_standoff"]("keyence t+25,+0 r+90 d0") == 0.0
        assert ns["_cal_label_standoff"]("no standoff here") is None


def test_the_failure_message_names_the_offline_entry_point():
    """The operator is at the work cell reading a traceback, not a docstring."""
    with _tmp() as d:
        cell = Cell(d)
        ns = _ns(cell)
        ns["calibrate_keyence"]()
        text = "\n".join(ns["_said"])
        assert "solve_keyence_from_take(" in text, text
        assert "if this run is interrupted" in text, text


# ------------------------------------------------------------------------------
# 6. Honest about what survived
# ------------------------------------------------------------------------------

def test_a_pose_set_at_one_standoff_never_leaves_the_ground():
    """Resilience is not "always emit a number".

    At one standoff, rotation of the head about its own laser line moves no
    measured point at first order: the fit converges, reports a residual at the
    noise floor, and hands back whatever the URDF said. A set that was PLANNED
    that way is refused before the arm moves, which is the cheapest place to
    refuse anything.
    """
    with _tmp() as d:
        cell = Cell(d, standoffs=(0.0,))
        ns, res, err = _run(cell)
        assert isinstance(err, Refused), err
        assert "all at one standoff" in str(err), str(err)
        assert cell.calls == [], "the arm moved for a degenerate set"
        assert cell.saved == []


def test_a_run_that_loses_a_whole_standoff_rung_refuses_and_says_which():
    """A run that loses everything after an early pose.

    With the standoffs interleaved an early prefix spans all three rungs, so
    this only reproduces by scripting a failure at every pose of two rungs. The
    point is what the refusal SAYS: which poses were lost, not just that the
    set is inadequate.
    """
    with _tmp() as d:
        cell = Cell(d)
        cell.script = {p["label"]: [CaptureMotionFailed("planner 51%")]*3
                       for p in cell.poses if p["distance"] != 0.0}
        ns, res, err = _run(cell)
        assert isinstance(err, Refused), err
        msg = str(err)
        assert "all at one standoff" in msg, msg
        assert "pose(s) were lost" in msg, msg
        assert "planner 51%" in msg, msg
        assert cell.saved == []


def test_fewer_than_three_poses_is_still_refused():
    with _tmp() as d:
        cell = Cell(d)
        keep = {p["label"] for p in cell.poses[:2]}
        cell.script = {p["label"]: [CaptureRefused("nope")]*3
                       for p in cell.poses if p["label"] not in keep}
        ns, res, err = _run(cell)
        assert isinstance(err, Refused), err
        assert "nothing to solve" in str(err), str(err)
        assert cell.saved == []


def test_the_run_says_how_many_poses_the_answer_came_from():
    """A calibration from 23 of 24 poses is a different object from one from
    24, and the operator must be able to tell without reading a log."""
    with _tmp() as d:
        cell = Cell(d)
        label = cell.poses[4]["label"]
        cell.script = {label: [CaptureRefused("outside the cable band")]*3}
        ns, res, err = _run(cell)
        assert err is None, err
        text = "\n".join(ns["_said"])
        assert "SOLVED FROM 23 OF 24 POSES" in text, text
        assert label in text and "cable band" in text, text
        assert ns["_take"].results["standoffs_flown_m"] == [-0.015, 0.0, 0.015]


# ------------------------------------------------------------------------------
# 7. The flight order itself
# ------------------------------------------------------------------------------

def _candidate_poses(distances, tilts):
    """_cal_candidate_poses() with the work cell replaced by nothing at all."""
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_cal_candidate_poses")
    ns = {"np": np, "math": math, "_CAL": {}, "CalibrationRefused": Refused,
          "CableRefused": CableRefused,
          "tooling_envelope": lambda: None,
          "_cal_say": lambda m: None,
          "_cal_roll_budget": lambda tx, ty, m=None: [0.0, 90.0],
          "_cal_tool_quat": lambda tx, ty, r: (0.0, 0.0, 0.0, 1.0),
          "_cal_pose_for_sensor": lambda aim, q, d, s: np.array([0.0, 0.0, d]),
          "_cap_inside": lambda pts: True,
          "_cal_ik": lambda xyz, q, seed=None: [0.0]*6,
          "check_wrist3": lambda *a, **k: None,
          "_cal_hits_artifact": lambda *a, **k: (None, None)}
    ns["__builtins__"] = __builtins__
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    out, _dropped = ns["_cal_candidate_poses"](
        np.zeros(3), "lj_v7080_optical_frame", distances, tilts, "keyence")
    return out


def test_the_standoffs_are_interleaved_in_flight_order():
    """Surviving the failure is not enough -- the survivors must be SOLVABLE.

    Poses are flown in the order _cal_candidate_poses() returns them. With
    distance outermost the first third of the run is one standoff and nothing
    else, so a run cut short early holds no standoff spread and is refused as
    indeterminate however many poses it kept. Innermost, every three
    consecutive poses span all three rungs.
    """
    so = (-0.015, 0.0, 0.015)
    poses = _candidate_poses(so, [(0, 0), (25, 0), (-25, 0), (0, 15)])
    assert len(poses) == 4*2*3, len(poses)
    first3 = {round(p["distance"], 6) for p in poses[:3]}
    assert first3 == set(so), [p["label"] for p in poses[:6]]
    # and it holds for EVERY prefix of three, not just the first
    for k in range(0, len(poses) - 2, 3):
        assert {round(p["distance"], 6) for p in poses[k:k + 3]} == set(so), k


def test_interleaving_makes_fewer_large_reorientations():
    """It is less motion, not more: one tour of the tilts instead of three.

    Large reorientations are what fails -- a 150 mm leg carrying 180 degrees
    of wrist rotation -- so the ordering reduces the exposure as well as the
    cost of it.
    """
    tilts = [(0, 0), (25, 0), (-25, 0), (0, 15), (0, -15)]
    poses = _candidate_poses((-0.015, 0.0, 0.015), tilts)
    changes = sum(1 for a, b in zip(poses, poses[1:])
                  if (a["tilt"], a["roll"]) != (b["tilt"], b["roll"]))
    assert changes == len(tilts)*2 - 1, changes        # 9, against 29 before

# ------------------------------------------------------------------------------
# 8. The Azure pose loop has the same hole
# ------------------------------------------------------------------------------
# calibrate_azure()'s pose loop must not call capture_move() bare: with no
# handling, any refusal or incomplete leg throws away every view already taken,
# and its set is 39 poses. Same requirement, so the same tests.

AZ_LIFTED = ("_cal_fly_to", "_cal_stop_why", "_cal_offline_hint",
             "_cal_safe_name", "_cal_label_standoff", "_cal_atomic_write",
             "_cal_dataset_start", "_cal_dataset_index", "_cal_dataset_add",
             "_cal_dataset_finish", "_cal_dataset_load", "calibrate_azure",
             "solve_azure_from_take")


class AzureCell(object):
    """Board views at 24 tool attitudes. C = X^-1 G^-1 W, exactly."""

    def __init__(self, tmpdir, script=None):
        rng = np.random.default_rng(5)
        self.G = ct._robot_poses(ct.GOOD_TILTS[:6], ct.GOOD_ROLLS, rng=rng)
        self.C = ct._observations(self.G, rng=rng)
        self.poses = [{"label": "azure pose %d" % i,
                       "xyz": [float(v) for v in g[:3, 3]],
                       "quat": tuple(float(v) for v in cs.R_to_quat(g[:3, :3])),
                       "tilt": (0.0, 0.0), "roll": 0.0, "distance": 0.55,
                       "joints": [0.1*i]*6}
                      for i, g in enumerate(self.G)]
        self.index = {p["label"]: i for i, p in enumerate(self.poses)}
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.at = 0
        self.calls = []
        self.saved = []
        self.tmpdir = tmpdir

    attempt = Cell.attempt

    def grabbed(self):
        return [lb for k, lb in self.calls if k == "grab"]


def _azure_ns(cell):
    said = []

    class Take(object):
        def __init__(self):
            self.dir = os.path.join(cell.tmpdir, "take_az")
            os.makedirs(self.dir, exist_ok=True)
            self.results = {}
            self.events = []

        def event(self, **kw):
            self.events.append(kw)
            return kw

        def keep_file(self, path, name=None):
            return None

    take = Take()

    @contextlib.contextmanager
    def capture_scenario(scenario, **kw):
        try:
            yield take
        except CaptureAbort:
            take.results["aborted"] = True

    # calibrate_azure() imports sensor_msgs.msg inside the function.
    import types
    for m in ("sensor_msgs", "sensor_msgs.msg"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["sensor_msgs.msg"].Image = object
    sys.modules["sensor_msgs"].msg = sys.modules["sensor_msgs.msg"]

    def _cal_tf(target, source, timeout=5.0):
        if (target, source) == ("world", "probe_end"):
            return np.asarray(cell.G[cell.at], float)
        return ct.X_TRUE.dot(cs.se3_exp([0.004, -0.003, 0.004,
                                         0.02, -0.015, 0.012]))

    def detect_board(gray, K, D, board=None, refine=True):
        label = cell.poses[cell.at]["label"]
        cell.calls.append(("grab", label))
        return (np.asarray(cell.C[cell.at], float), np.zeros((54, 2)),
                np.zeros((54, 3)))

    ns = {
        "np": np, "os": os, "sys": sys, "json": json, "math": math,
        "re": __import__("re"), "time": __import__("time"),
        "datetime": __import__("datetime"), "contextlib": contextlib,
        "CalibrationRefused": Refused, "CaptureAbort": CaptureAbort,
        "CaptureSkip": CaptureSkip, "CaptureRefused": CaptureRefused,
        "CaptureMotionFailed": CaptureMotionFailed,
        "CaptureStopped": CaptureStopped, "CableRefused": CableRefused,
        "CAL_SETTLE_S": 0.0, "CAL_AZURE_IR_TOPIC": "/azure/ir/image_raw",
        "CAL_SOLVE_POINTS_PER_POSE": 1500, "CAL_PROFILES_PER_POSE": 60,
        "time": time,
        "CAL_MOVE_ROUTES": 2, "CAPTURE_ROOT": cell.tmpdir,
        "CELL_NAME": "nowhere", "CELL_MODE": "sim", "_CAL": {},
        "_cal_geom": lambda: cs,
        "_cal_say": lambda m: said.append(str(m)),
        "sensor_topics": lambda s: {"color": "/azure/rgb/image_raw",
                                    "camera_info": "/azure/rgb/camera_info",
                                    "frame": "rgb_camera_link"},
        "sensor_topic": lambda *a, **k: "/azure/rgb/image_raw",
        "azure_calibration_poses": lambda *a, **k: list(cell.poses),
        "check_calibration_poses": lambda poses, what="", quiet=False: {
            "verdict": "good", "problems": []},
        "_cal_intrinsics": lambda info, timeout=15.0: (
            np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]]),
            np.zeros(5), 1280, 720, "rgb_camera_link"),
        "describe_calibration_artifact": lambda: "a board",
        "cal_board": lambda: {"cols": 9, "rows": 6, "square": 0.02,
                              "object_points": None},
        "_cal_one_message": lambda *a, **k: [object()],
        "_cal_gray": lambda m: np.zeros((720, 1280), np.uint8),
        "detect_board": detect_board,
        "capture_scenario": capture_scenario,
        "capture_move": lambda poses, label, **k: cell.attempt("cartesian",
                                                               label),
        "capture_move_joints": lambda q, label, **k: cell.attempt(
            "joint", label.split(" (")[0]),
        "_ik_target": lambda x, q: [0.0]*6,
        "_local_ik": contextlib.nullcontext,
        "_cal_tf": _cal_tf,
        "write_calibration": lambda sensor, res, frame, take=None: (
            cell.saved.append((sensor, frame, res)) or {"record": "/dev/null"}),
    }
    ns["__builtins__"] = __builtins__
    _lift(AZ_LIFTED, ns)
    ns["_said"] = said
    ns["_take"] = take
    return ns


def test_the_azure_loop_no_longer_loses_every_view_to_one_pose():
    with _tmp() as d:
        cell = AzureCell(d)
        label = cell.poses[4]["label"]
        cell.script = {label: [CaptureMotionFailed("planner reached 51%")]*3}
        ns = _azure_ns(cell)
        res = ns["calibrate_azure"]()
        assert len(cell.grabbed()) == len(cell.poses) - 1, cell.grabbed()
        assert ns["_take"].results["used"] == len(cell.poses) - 1
        assert label in dict(ns["_take"].results["missed"])
        mm, mrad = ct._error(res.X)
        assert mm < 0.5 and mrad < 2.0, (mm, mrad, res.report())


def test_an_azure_stop_still_solves_what_was_seen():
    with _tmp() as d:
        cell = AzureCell(d)
        label = cell.poses[12]["label"]
        cell.script = {label: [CaptureStopped("over the force limit")]*3}
        ns = _azure_ns(cell)
        try:
            ns["calibrate_azure"]()
        except CaptureStopped:
            pass
        else:
            raise AssertionError("a protective stop did not end the run")
        assert len(cell.grabbed()) == 12, cell.grabbed()
        assert cell.saved, "twelve good views were thrown away"
        assert "THE MONITOR STOPPED THE ARM" in "\n".join(ns["_said"])


def test_an_azure_take_can_be_re_solved_offline():
    with _tmp() as d:
        cell = AzureCell(d)
        ns = _azure_ns(cell)
        ns["calibrate_azure"]()
        pd = os.path.join(ns["_take"].dir, "azure_poses")
        again = _azure_ns(cell)["solve_azure_from_take"](pd, save=False)
        mm, mrad = ct._error(again.X)
        assert mm < 0.5 and mrad < 2.0, (mm, mrad)



# ------------------------------------------------------------------------------
# 9. And it survives being loaded the way dgpy loads it
# ------------------------------------------------------------------------------

def test_calibration_dpi_still_works_as_a_function_body():
    """The one thing AST-lifting cannot see.

    dgpy compiles a .dpi as a FUNCTION BODY, so `import re` at its top is a
    LOCAL of that body -- reachable from the functions defined in it only as a
    CLOSURE, because their __globals__ is the shared dict and the import never
    went there. Every test above execs the lifted functions at module level
    with a namespace of our own, which resolves those names from globals and so
    would pass even if the closure did not exist.

    This one executes the whole file the way the loader does and then CALLS the
    new functions, which is where a missing closure or a missing `global`
    actually bites -- at run time, hours into a run.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    body = "def _dpi_body():\n" + "".join("    " + ln + "\n"
                                          for ln in src.splitlines())
    d = tempfile.mkdtemp(prefix="cal_body_")
    try:
        G = {"__builtins__": __builtins__,
             "SCAN_MODULE_DIR": os.path.join(HERE, "x.dpi"),
             "CAPTURE_ROOT": d, "CELL_NAME": "nowhere", "CELL_MODE": "sim",
             "CaptureAbort": CaptureAbort, "CaptureSkip": CaptureSkip,
             "CaptureRefused": CaptureRefused, "CableRefused": CableRefused,
             "CaptureMotionFailed": CaptureMotionFailed,
             "CaptureStopped": CaptureStopped,
             "_CAP": {}, "_local_ik": contextlib.nullcontext,
             "_ik_target": lambda x, q: [0.0]*6,
             "capture_move_joints": lambda q, label, **k: None,
             "capture_move": lambda poses, label, **k: (_ for _ in ()).throw(
                 CaptureMotionFailed("%s did not complete (51%%)" % label))}
        exec(compile(body, "calibration.dpi", "exec"), G)
        G["_dpi_body"]()
        for name in ("_cal_fly_to", "_cal_dataset_start", "_cal_dataset_add",
                     "_cal_dataset_finish", "_cal_dataset_load",
                     "_cal_keyence_solve", "solve_keyence_from_take",
                     "solve_azure_from_take", "_cal_stop_why",
                     "_cal_offline_hint", "_cal_label_standoff",
                     "CAL_MOVE_ROUTES"):
            assert name in G, "not exported from the body: %s" % name
        said = []
        G["_cal_say"] = lambda m: said.append(str(m))

        # `re`, reachable only as a closure over the body's import
        assert abs(G["_cal_label_standoff"]("keyence t+0,-15 r+90 d-15")
                   + 0.015) < 1e-9

        # os / json / datetime / numpy, likewise, through a full round trip
        class Take(object):
            dir = os.path.join(d, "take")
        os.makedirs(Take.dir, exist_ok=True)
        data = G["_cal_dataset_start"]("keyence", Take(), 3, header={"a": 1})
        G["_cal_dataset_add"](data, {
            "label": "keyence t+0,+0 r+0 d-15", "G": np.eye(4),
            "X_urdf": np.eye(4), "points": np.zeros((5, 3)),
            "distance": -0.015, "tilt": [0.0, 0.0], "roll": 0.0})
        G["_cal_dataset_finish"](data, note="stopped")
        back = G["_cal_dataset_load"](data["dir"])
        assert len(back["rows"]) == 1
        assert back["rows"][0]["points"].shape == (5, 3)

        # and the retry really does reach capture_move_joints from in there
        out = G["_cal_fly_to"]({"label": "keyence t+0,-15 r+90 d-15",
                                "xyz": [0.0, 0.0, 0.0],
                                "quat": (0.0, 0.0, 0.0, 1.0),
                                "joints": [0.2]*6})
        assert out["ok"] and out["route"].startswith("joint move"), out
    finally:
        shutil.rmtree(d, ignore_errors=True)



def test_a_session_holding_both_sensors_hands_back_the_right_one():
    """A session directory has an azure_poses AND a keyence_poses in it.

    Whichever sorts last is not the answer: the Azure's views would LOAD
    (a 4x4 is a 4x4) and solve to nonsense, and nothing downstream would say
    so. Each entry point looks for its own sensor's directory by name.
    """
    with _tmp() as d:
        cell = Cell(d)
        ns = _ns(cell)
        ns["calibrate_keyence"]()
        az = AzureCell(d)
        ans = _azure_ns(az)
        ans["calibrate_azure"]()
        session = d                       # holds take/ and take_az/
        assert os.path.isdir(os.path.join(ns["_take"].dir, "keyence_poses"))
        assert os.path.isdir(os.path.join(ans["_take"].dir, "azure_poses"))

        k = _ns(cell)["solve_keyence_from_take"](session, save=False)
        assert _error(k.X)[0] < 0.5, _error(k.X)
        a = _azure_ns(az)["solve_azure_from_take"](session, save=False)
        assert ct._error(a.X)[0] < 0.5, ct._error(a.X)

        # and a sensor with nothing under the path says which one it wanted
        with _tmp() as empty:
            try:
                _ns(cell)["solve_keyence_from_take"](empty)
            except Refused as e:
                assert "keyence_poses/index.json" in str(e), str(e)
            else:
                raise AssertionError("an empty directory was accepted")



def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            failed.append((name, e))
            print("  FAIL  %s\n        %s" % (name, e))
        else:
            print("  ok    %s" % name)
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    if failed:
        raise SystemExit(str(failed[0][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
