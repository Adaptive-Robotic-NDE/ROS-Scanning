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
# File:  azure_gauge_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the Keyence gauge that measures rough-mesh depth error, and the
#   standoff limits it must respect.
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
"""The automated Keyence gauge: does it measure the mesh error, and can it hurt?

WHAT IT IS FOR. The Azure's depth reads every surface about 12.45 mm too far.
A constant correction fixes that to under 1 mm -- but a constant is a
calibration, and a drifted one is silent. A Keyence raster planned over a mesh
13.6 mm low across a plate and 22 mm low at one end flies the head at 58 mm
against a 57 mm hard limit, brings 26-39% of the plate back as points scattered
over 16 mm, and leaves the head nearly in contact with the part, with nothing
in the pipeline noticing. gauge_mesh_against_keyence() is what measures that
residual, on the part, before a raster is planned, with no human touch-off.

THE ARITHMETIC UNDER TEST, re-derived here rather than copied:

    the LJ-V7080 measures 57..103 mm from the head (a 46 mm window) about a
    reference distance of 80 mm
    the head is placed so the MESH-PREDICTED surface is 100 mm away
    delta = z_keyence - z_mesh          + means the MESH READS LOW
    actual standoff = 100 - delta
    57 <= 100 - delta <= 103    <=>     -3 <= delta <= +43 mm

so one position tolerates 43 mm of error on the side that hurts, this cell
sits at +12.45, and THE FIRST GRAB NORMALLY JUST WORKS. The step-down ladder
is only for a mesh that reads HIGH, which puts the surface beyond the far
limit and returns nothing at all.

EVERY TEST THAT MATTERS HERE IS A SAFETY TEST. The gauge exists because the
head got close to a part, so the cases worth the most are the ones where it is
asked to fly closer: a caller who starts inside the near limit, a ladder asked
for more rungs than the arithmetic supports, and a sensor whose own heights
say the surface is nearer than the mesh claimed.

Runs on the HOST with plain pytest -- no ROS, no container, no robot, no arm.
The .dpi files are exec'd the way the loader execs them, `_cal_pose_for_sensor`
is AST-lifted OUT OF calibration.dpi so the real pose arithmetic is what is
being tested, check_profile() is the real one out of calibration_solve.py, and
only the four ROS-facing helpers (fly, grab, TF, topic) are stubbed.

Long-stable numpy and math only: the host is on numpy 2.5, the container on
1.26 and THE WORK CELL ON 1.23 with scipy 1.15, so nothing here may depend on
behaviour that differs between them.

    python3 azure_gauge_test.py
    python3 -m pytest -q azure_gauge_test.py
"""
import ast
import functools
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calibration_solve as cs                            # noqa: E402

FAILURES = []

#: The sensor's own numbers, from the datasheet and from CAL_KEYENCE_*.
REF = 0.080
RANGE = (0.057, 0.103)
MARGIN = 0.005

#: The Azure's standing bias at this cell: eight independent measurements,
#: -11.6 to -13.6 mm, mean -12.4. A mesh that reads LOW by 12.45 mm.
BIAS_M = 0.01245


# ------------------------------------------------------------------------------
# Loading the .dpi files the way the loader loads them
# ------------------------------------------------------------------------------
# A .dpi is compiled as a FUNCTION BODY, so `global X` at its top level is a
# SyntaxError at module scope. Strip those lines and exec into ONE shared dict,
# which is exactly the globals() every .dpi shares at runtime -- and which is
# why azure_gauge.dpi can call keyence_standoff.dpi's gate without importing
# anything.

def _load(name):
    src = open(os.path.join(HERE, name)).read()
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


def _lift(name, want):
    """The named top-level defs out of a .dpi, by AST. Nothing else runs."""
    src = open(os.path.join(HERE, name)).read()
    body = ast.parse("def _b():\n" + "".join(
        "    " + l + "\n" for l in src.splitlines())).body[0].body
    fns = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(f.name for f in fns) == sorted(want), \
        (name, sorted(f.name for f in fns))
    return fns


def _quat_to_matrix(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


class _Refused(Exception):
    """Stands in for CalibrationRefused, which lives in calibration.dpi."""


class _CaptureRefused(Exception):
    """CaptureRefused: a capture guard refused BEFORE anything moved.

    What _check_envelope() and _check_wrist3() raise in cell_capture_motion.dpi,
    and what the unattended route has to turn into a CalibrationRefused that
    costs one sample point.
    """


class _Stopped(Exception):
    """CaptureStopped: the arm was physically stopped."""


class _Abort(Exception):
    """CaptureAbort: the operator pressed q."""


class _Prompted(AssertionError):
    """THE TRIPWIRE. Raised if anything on the unattended route reaches the
    operator -- confirm(), input(), or _decide().

    An AssertionError rather than a plain Exception on purpose: the gauge's own
    handlers catch CalibrationRefused and the stop-the-run trio, so a prompt
    that arrived as one of those would be quietly turned into "this sample point
    failed" and the suite would go green over the exact fault it is here to
    catch.
    """


class _Result(object):
    """A motion result in the shape the pipeline reads off every sweep row:
    `.result.success` and `.result.fraction` (see _fly_scanpath())."""

    def __init__(self, success=True, fraction=1.0):
        self.result = type("result", (object,), {
            "success": bool(success), "fraction": float(fraction)})()


class _NoFlag(object):
    """What ros.MoveJoints hands back when there is NO execute_joint_move
    server: it falls back to a direct trajectory command and returns that
    goal's result, which carries no success flag at all. The gauge must then
    ask live TF whether the head arrived rather than assume it did."""


#: The cell's motion envelope for the harness: generous, so a test that narrows
#: `envelope` to steer VERTEX SELECTION does not also forbid probe_end from
#: going where the gauge puts it.
#:
#: AT THE CELL THESE ARE ONE BOX -- capture_envelope() feeds both
#: gauge_sample_points() and _check_envelope(). The harness splits them because
#: probe_end sits 75 mm to the side of the point the sensor measures (T_PS), so
#: a box drawn tightly round the sampled vertices is not a box probe_end fits
#: in, and the two questions want different boxes in a unit test.
VOLUME = {"lo": np.array([0.30, -0.60, 0.50]),
          "hi": np.array([1.30, 0.60, 1.40]),
          "top": 1.40, "floor": 0.50, "ceiling": 1.60}


def fresh():
    """A namespace with the gauge, the standoff guard and the real geometry."""
    ns = {"__name__": "azure_gauge_ns", "os": os, "np": np,
          "quat_to_matrix": _quat_to_matrix,
          "CalibrationRefused": _Refused, "CaptureRefused": _CaptureRefused,
          "CaptureStopped": _Stopped, "CaptureAbort": _Abort,
          "CAL_KEYENCE_REF_DIST": REF, "CAL_KEYENCE_RANGE_M": RANGE,
          "CAL_KEYENCE_RANGE_MARGIN_M": MARGIN,
          # Zero, so the suite does not spend a second per pose waiting for an
          # arm that is not there.
          "CAL_SETTLE_S": 0.0, "CAL_PROFILES_PER_POSE": 3}
    exec(compile(_load("keyence_standoff.dpi"), "keyence_standoff.dpi",
                 "exec"), ns)
    exec(compile(_load("azure_gauge.dpi"), "azure_gauge.dpi", "exec"), ns)
    # THE REAL POSE ARITHMETIC, lifted rather than re-implemented. Where the
    # optical frame goes for a commanded standoff is the whole safety argument;
    # a stand-in here would test the stand-in.
    exec(compile(ast.Module(body=_lift("calibration.dpi",
                                       {"_cal_pose_for_sensor"}),
                            type_ignores=[]), "calibration.dpi", "exec"), ns)
    ns["_cal_geom"] = lambda: cs
    return ns


NS = fresh()


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _checked(fn):
    """Make the checks one test function ran into that test's pytest verdict.

    check() records rather than raises, so a standalone run reports every false
    fact in one pass instead of stopping at the first. Under pytest that is a
    test which passes while printing FAIL, so the wrapper asserts on whatever
    its test added -- and pytest then names the test that broke rather than the
    file-wide backstop at the bottom. main() calls fn.__wrapped__ to keep the
    report-everything behaviour of the standalone run.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# ------------------------------------------------------------------------------
# A synthetic cell: one real surface, and a mesh that disagrees with it
# ------------------------------------------------------------------------------

#: probe_end -> lj_v7080_optical_frame on the ec tooling: a pure Rz(+90 deg)
#: with translation (-0.07515, -0.0022, -0.03428), read off lj_v7080_macro.xacro
#: and confirmed by chaining the rendered URDF.
T_PS = np.array([[0.0, -1.0, 0.0, -0.07515],
                 [1.0, 0.0, 0.0, -0.0022],
                 [0.0, 0.0, 1.0, -0.03428],
                 [0.0, 0.0, 0.0, 1.0]])
#: The head itself sits REF above the optical frame, which is the plane a raw
#: reading of zero means.
T_PH = T_PS.copy()
T_PH[2, 3] += REF


def flat_mesh(zf, x0=0.70, x1=0.90, y0=-0.10, y1=0.10, nx=21, ny=21):
    """A triangulated grid whose vertex heights come from `zf(x, y)`."""
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = zf(X, Y)
    V = np.column_stack([X.ravel(), Y.ravel(), np.asarray(Z, float).ravel()])
    t = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i*ny + j
            t.append((a, a + 1, a + ny))
            t.append((a + 1, a + ny + 1, a + ny))
    return V, np.asarray(t, np.int64)


class Cell(object):
    """A cell where the truth and the mesh differ by a known amount.

    `truth` and `delta` are floats or callables of (x, y); the MESH is built at
    `truth - delta`, so delta is exactly what the gauge must report:
    delta = z_keyence - z_mesh, positive meaning the mesh reads LOW.
    """

    def __init__(self, ns, truth=0.9218, delta=BIAS_M, noise=2.0e-5,
                 degenerate=False, refuse=(), stop_at=None, envelope=None,
                 seed=1, volume=VOLUME, refuse_cartesian=(), refuse_joints=(),
                 no_ik=(), wrist_band=None, wrist3=0.0, wrist3_global=None,
                 joint_bulge=0.0,
                 no_fk=False, joints_no_flag=False, joints_no_arrive=False,
                 no_pipeline=False, no_wrapper=False, park=(0.80, 0.0, 1.10),
                 answer_prompts=False, **mesh_kw):
        self.ns = ns
        self._truth = truth
        self._delta = delta
        self.noise = float(noise)
        self.degenerate = bool(degenerate)
        self.refuse = tuple(refuse)
        self.stop_at = stop_at
        self.envelope = envelope
        self.rng = np.random.default_rng(seed)
        self.verts, self.tris = flat_mesh(
            lambda X, Y: self.truth(X, Y) - self.delta(X, Y), **mesh_kw)
        self.flown = []
        self.grabs = []
        self.laser = []
        self.said = []
        #: Where probe_end is parked before the first gauge move -- inside the
        #: work volume, because the arm has already flown an Azure sweep by the
        #: time stage 2a runs. An identity T_wo would put it under the bench and
        #: every straight line would then be an ENTRY into the volume rather
        #: than a move inside it, which is a different code path.
        self.T_wo = np.eye(4)
        self.T_wo[:3, 3] = np.asarray(park, float).reshape(3) + T_PS[:3, 3]
        # -- the unattended (pipeline) route ---------------------------------
        #: The work volume _check_envelope() holds probe_end to. None means the
        #: cell declares none, which refuses every move.
        self.volume = volume
        #: Label substrings whose Cartesian move fails, so the joint fallback is
        #: exercised; and whose joint move fails, so the refusal is.
        self.refuse_cartesian = tuple(refuse_cartesian)
        self.refuse_joints = tuple(refuse_joints)
        #: Label substrings for which _ik_target() has no answer.
        self.no_ik = tuple(no_ik)
        #: (lo, hi) radians for wrist 3, or None for a cell whose cables have
        #: never been measured -- which does NOT refuse, and says so once.
        self.wrist_band = wrist_band
        #: What IK puts wrist 3 at: a number, or a callable of the target xyz.
        self._wrist3 = wrist3
        #: What a GLOBAL-mode solve offers instead, when one is asked for. None
        #: means this cell's two modes agree.
        self._wrist3_global = wrist3_global
        #: Every _ik_target() call: (xyz, local_only). The gauge's PREDICTION
        #: must be local-only -- a prediction made in a mode the executing node
        #: is not in refuses moves that node would have flown.
        self.ik_asks = []
        #: Metres the FAKE FK bulges the joint path's tip in +z at its midpoint,
        #: so a joint route whose ENDPOINTS are both fine can still be refused.
        self.joint_bulge = float(joint_bulge)
        self.no_fk = bool(no_fk)
        self.joints_no_flag = bool(joints_no_flag)
        self.joints_no_arrive = bool(joints_no_arrive)
        self.cartesian = []
        self.joint_moves = []
        self.confirmed = []
        self.wrist_checks = []
        self.envelope_checks = []
        self.checkpoints = []
        self.prompts = []
        #: Is somebody at the keyboard? False everywhere but a deliberate
        #: confirm_moves=True test, so any prompt on any other path fails.
        self.answer_prompts = bool(answer_prompts)
        self._pending = ""
        self._pending_xyz = self._probe_end_now()
        self._pending_kind = None
        self._install()
        if no_wrapper:
            # A config without scan_plan.dpi: the gauge must fall back to
            # ros.CartesianMove, which is what _fly_scanpath() does too.
            ns.pop("cartesian_move", None)
        if no_pipeline:
            # A session with no cell_capture_motion.dpi at all: every guard is
            # missing, so the unattended route must refuse rather than fly.
            for k in ("_check_envelope", "_check_wrist3", "_speed_class",
                      "_cap_densify", "_cap_pose", "_ik_target",
                      "capture_node", "live_pose", "ros"):
                ns.pop(k, None)

    # -- the two surfaces ---------------------------------------------------
    def truth(self, x, y):
        v = self._truth
        return v(x, y) if callable(v) else np.full_like(np.asarray(x, float), v)

    def delta(self, x, y):
        v = self._delta
        return v(x, y) if callable(v) else np.full_like(np.asarray(x, float), v)

    def say(self, msg):
        self.said.append(str(msg))

    @property
    def mesh(self):
        return (self.verts, self.tris)

    # -- the ROS-facing helpers ---------------------------------------------
    def _install(self):
        ns = self.ns
        ns["_CAL"] = {}
        ns["sensor_topic"] = lambda *a, **k: "/profiles"
        ns["_cal_tool_quat"] = lambda tx, ty, roll: (0.0, 0.0, 0.0, 1.0)
        ns["_cal_tf"] = self._tf
        ns["_cal_fly_or_refuse"] = self._fly
        ns["_cal_profile_points"] = self._profile_points
        ns["_cal_stop_why"] = lambda e: "stopped: %s" % (e,)
        ns["_cal_atomic_write"] = self._atomic_write
        ns["capture_envelope"] = lambda: self.envelope
        ns["capture_laser"] = self.laser.append
        ns.pop("rough_mesh", None)
        ns.pop("detected_region", None)
        # -- the PIPELINE's motion path, which is the gauge's default --------
        # These are cell_capture_motion.dpi's exported guards and the motion
        # module the sweep flies. Nothing among them can ask a human anything.
        ns["_check_envelope"] = self._check_envelope
        ns["_check_wrist3"] = self._check_wrist3
        ns["_speed_class"] = self._speed_class
        ns["_cap_densify"] = self._densify
        ns["_cap_pose"] = lambda xyz, q: (np.asarray(xyz, float).reshape(3),
                                          tuple(float(v) for v in q))
        ns["_ik_target"] = self._ik_target
        ns["capture_node"] = self
        ns["live_pose"] = self
        ns["ros"] = self
        ns["scan_checkpoint"] = self._checkpoint
        ns["joint_move_kwargs"] = lambda: {"vel": 0.02, "acc": 0.02}
        ns["motion_scaling"] = lambda v: 0.64*float(v)
        ns["path_motion_scaling"] = lambda poses, v, quiet=True: 0.5*float(v)
        ns["cartesian_move"] = self.cartesian_move
        # THE TRIPWIRE. On the unattended route nothing may reach the operator,
        # and the three doors to one are all shut here rather than left to hang.
        ns["confirm"] = self._prompt
        ns["_decide"] = self._prompt
        ns["input"] = self._prompt

    def _prompt(self, *a, **k):
        """confirm() / _decide() / input(): a TRIPWIRE unless a test has said
        an operator is standing there (answer_prompts=True, which is only right
        for a confirm_moves=True run)."""
        self.prompts.append(a[0] if a else "")
        if self.answer_prompts:
            return "go"
        raise _Prompted(
            "THE OPERATOR WAS PROMPTED. The autonomous pipeline must never ask "
            "anybody to approve a move: %r" % ((a[0] if a else ""),))

    def _checkpoint(self, what="", timeout=None):
        self.checkpoints.append(str(what))
        self._maybe_stop(what)
        return True

    def _maybe_stop(self, where):
        if self.stop_at is not None and len(self.flown) == self.stop_at:
            raise _Stopped("the arm was stopped at %s" % (where,))

    # -- the guards, as cell_capture_motion.dpi asks them -------------------
    def _check_envelope(self, points, what, tol=0.001, allow_entry=True):
        """_check_envelope(), faithfully: no volume refuses everything, leaving
        the box refuses, and a path that STARTS outside may enter it.

        Records every call, and the move it is about to be asked about --
        `ros.CartesianMove` carries no label, so this is where the harness
        learns which gauge pose is in flight. That ordering is itself the point:
        the guard is asked before anything is commanded, so if it were ever
        asked afterwards the stubs below would have nothing to record.
        """
        self.envelope_checks.append(str(what))
        self._pending = str(what)
        p = np.asarray(points, float).reshape(-1, 3)
        self._pending_xyz = p[-1].copy()
        env = self.volume
        if env is None:
            raise _CaptureRefused(
                "%s: this cell has no work volume, and nothing moves without "
                "one. Measure it with capture_touch_off()." % (what,))
        lo, hi = env["lo"], env["hi"]
        inside = ((p[:, 0] >= lo[0] - tol) & (p[:, 0] <= hi[0] + tol) &
                  (p[:, 1] >= lo[1] - tol) & (p[:, 1] <= hi[1] + tol) &
                  (p[:, 2] >= env["floor"] - tol) &
                  (p[:, 2] <= env["ceiling"] + tol))
        if inside.all():
            return env
        if allow_entry and not inside[0] and inside.any():
            k = int(np.argmax(inside))
            if (inside[k:].all()
                    and (p[:k, 2] >= min(env["floor"], p[0, 2]) - tol).all()):
                return env
        raise _CaptureRefused(
            "%s would leave the work volume. Nothing moved." % (what,))

    def _check_wrist3(self, qs, what, quats=None):
        """_check_wrist3(): the band for the tilt commanded, or no check at all
        when this cell's cables have never been measured."""
        self.wrist_checks.append((str(what), [list(q) for q in qs],
                                  None if quats is None else list(quats)))
        if self.wrist_band is None:
            return None
        lo, hi = self.wrist_band
        for q in qs:
            w = float(q[5])
            if not (lo - 1e-9 <= w <= hi + 1e-9):
                raise _CaptureRefused(
                    "%s would turn wrist 3 to %.1f deg, outside the measured "
                    "cable band %.1f..%.1f deg for the tilt it commands"
                    % (what, math.degrees(w), math.degrees(lo),
                       math.degrees(hi)))
        return (lo, hi)

    def _speed_class(self, samples):
        """Near the part, because a gauge is: 50 mm/s, the approved limit."""
        return "near", 0.050

    @staticmethod
    def _densify(points, step=0.005):
        pts = [np.asarray(p, float).reshape(3) for p in points]
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(math.ceil(float(np.linalg.norm(b - a))/step)))
            out.extend(a + (b - a)*k/float(n) for k in range(1, n + 1))
        return np.array(out)

    def _wrist3_for(self, xyz):
        v = self._wrist3
        return float(v(np.asarray(xyz, float)) if callable(v) else v)

    def _ik_target(self, x, q, local_only=False):
        """_ik_target(): joints for a pose. Joint 0 carries the target's x, so
        the fake FK below can recover where along the path a configuration is --
        real joints do not mean anything here and a recoverable parameter is
        what makes the joint-path check testable at all.

        `local_only` IS THE INTERESTING ARGUMENT and this stub answers it
        differently on purpose. A real _ik_target() solves in local mode first
        and falls back to global; local stays in the arm's branch, global is free
        to leave it. `wrist3_global` is what this cell's global solver offers, so
        a test can build the case that matters: a destination reachable in band
        in the branch the arm is in, whose global branch is a whole turn away.
        """
        self.ik_asks.append((tuple(float(v) for v in np.asarray(x, float).reshape(3)),
                             bool(local_only)))
        if any(p in self._pending for p in self.no_ik):
            return None
        x = np.asarray(x, float).reshape(3)
        w = self._wrist3_for(x)
        if not local_only and self._wrist3_global is not None:
            w = float(self._wrist3_global)
        return [float(x[0]), -1.2, 1.0, -0.5, -1.5708, w]

    # -- capture_node and live_pose ----------------------------------------
    def current_joints(self):
        here = self._probe_end_now()
        return [float(here[0]), -1.2, 1.0, -0.5, -1.5708, self._wrist3_for(here)]

    def fk(self, q, links=("probe_end",)):
        """A stand-in for /compute_fk: the tip interpolated between where the
        tool is and where the joint move is going, plus `joint_bulge` in +z at
        the midpoint -- so a joint path whose ENDS are both legal can still be
        refused for where it goes in between, which is the only thing this check
        exists to catch."""
        if self.no_fk:
            return None
        a = self._probe_end_now()
        b = np.asarray(self._pending_xyz, float).reshape(3)
        span = float(b[0] - a[0])
        t = (0.0 if abs(span) < 1e-12
             else (float(np.asarray(q, float)[0]) - float(a[0]))/span)
        t = min(1.0, max(0.0, t))
        p = a + t*(b - a) + np.array([0.0, 0.0,
                                      4.0*t*(1.0 - t)*self.joint_bulge])
        return [(p, (0.0, 0.0, 0.0, 1.0)) for _k in links]

    def _probe_end_now(self):
        """probe_end in world, from wherever the optical frame was left."""
        return self.T_wo[:3, 3] - T_PS[:3, 3]

    def snapshot(self):
        """live_pose.snapshot(): (xyz, quat, stamp) for probe_end."""
        return (self._probe_end_now(), (0.0, 0.0, 0.0, 1.0), len(self.flown))

    # -- `ros`: the two primitives the sweep flies -------------------------
    def cartesian_move(self, poses, kind=None, **kwargs):
        """scan_plan.dpi's wrapper: ros.CartesianMove with the IK settings this
        KIND of move needs. A single waypoint is a step and wants the tight
        threshold, which is what a gauge pose is."""
        count = len(getattr(poses, "poses", poses))
        self._pending_kind = kind or ("point_step" if count <= 1
                                      else "cartesian_path")
        return self.CartesianMove(poses, **kwargs)

    def CartesianMove(self, poses, vel=0.1, acc=0.1, min_fraction=1.0, **k):
        self.cartesian.append({"n": len(poses), "vel": float(vel),
                               "acc": float(acc), "label": self._pending,
                               "kind": self._pending_kind})
        self._maybe_stop(self._pending)
        if any(p in self._pending for p in self.refuse_cartesian):
            return _Result(False, 0.0)
        xyz, q = poses[-1]
        self._arrive(xyz, q, self._pending, "cartesian")
        return _Result(True, 1.0)

    def MoveJoints(self, positions, secs=5.0, plan=True, vel=None, acc=None,
                   **k):
        self.joint_moves.append({"q": list(positions), "secs": float(secs),
                                 "vel": vel, "label": self._pending})
        self._maybe_stop(self._pending)
        if any(p in self._pending for p in self.refuse_joints):
            return _Result(False, 0.0)
        if not self.joints_no_arrive:
            self._arrive(self._pending_xyz, (0.0, 0.0, 0.0, 1.0),
                         self._pending, "joint move")
        return _NoFlag() if self.joints_no_flag else _Result(True, 1.0)

    def _arrive(self, xyz, quat, label, route):
        T_wp = np.eye(4)
        T_wp[:3, :3] = _quat_to_matrix(quat)
        T_wp[:3, 3] = np.asarray(xyz, float).reshape(3)
        self.T_wo = T_wp.dot(T_PS)
        self.flown.append((label, np.asarray(xyz, float), self.T_wo.copy()))

    @staticmethod
    def _atomic_write(path, write):
        with open(path, "wb") as f:
            write(f)

    def _tf(self, target, source, timeout=5.0):
        if (target, source) == ("probe_end", "lj_v7080_optical_frame"):
            return T_PS.copy()
        if (target, source) == ("probe_end", "lj_v7080_frame"):
            return T_PH.copy()
        if target == "world":
            return self.T_wo.copy()
        raise AssertionError("unexpected transform %s <- %s" % (target, source))

    def _fly(self, xyz, quat, label):
        """_cal_fly_or_refuse(): the CONFIRMED route, which is what
        confirm_moves=True selects and what the pipeline must not use.

        IT REALLY ASKS. The chain at the cell is _cal_fly_or_refuse() ->
        capture_move() -> _decide() -> confirm(), and a stub that skipped the
        confirm() would make the tripwire prove nothing: with the gauge's default
        switched to the confirmed route the suite would stay green over the exact
        fault it exists to catch. So the confirmed route goes through
        ns["confirm"] like the real one, and a Cell only answers it when a test
        said the operator is there.
        """
        self.confirmed.append(str(label))
        self.ns["confirm"]("%s\n  [Enter=go, s=skip, q=abort]" % (label,))
        self._maybe_stop(label)
        for pat in self.refuse:
            if pat in label:
                raise _Refused("%s: the arm did not reach this pose by any "
                               "route" % (label,))
        self._arrive(xyz, quat, label, "cartesian")
        return {"ok": True, "route": "cartesian", "why": None, "tried": []}

    # -- what the profiler returns from wherever the arm left it ------------
    def profile_rows(self, count, m=200):
        """`count` profiles of `m` columns, +Inf outside the window.

        The line runs along the optical frame's x. Each column's world xy comes
        from the pose, its world z from the TRUE surface, and the device's own
        reading is `z = z_true - z_optical` -- positive TOWARDS THE HEAD, which
        is the convention keyence_driver_node packs and the one the standoff
        expression `ref - z` is built on. Out-of-range columns are +Inf, which
        is REP-117 and is what makes "the surface is not in the window" arrive
        as a coverage fact rather than as silence.
        """
        R, t = self.T_wo[:3, :3], self.T_wo[:3, 3]
        xs = np.linspace(-0.0155, 0.0155, m)
        base = t + np.outer(xs, R[:, 0])
        Z = self.truth(base[:, 0], base[:, 1])
        rows = []
        for _ in range(int(count)):
            zc = (Z - t[2])/R[2, 2]
            if self.degenerate:
                # One repeated number, which is what a triangulation profiler
                # returns at the bottom of its range and what check_profile's
                # SCATTER and REPEATED VALUE tests exist to refuse.
                zc = np.full_like(zc, float(np.median(zc)))
            elif self.noise:
                zc = zc + self.rng.normal(0.0, self.noise, zc.shape)
            stand = REF - zc
            zc = np.where((stand >= RANGE[0]) & (stand <= RANGE[1]), zc, np.inf)
            rows.append(np.column_stack([xs, np.zeros_like(xs), zc]))
        return np.vstack(rows)

    def _profile_points(self, topic, count=None, timeout=15.0, check=True):
        """_cal_profile_points(), faithfully: the REAL check_profile decides.

        Including where it leaves its verdict -- _CAL["last_profile_check"] --
        because that is what RULE B reads to tell a surface inside the near
        limit from one beyond the far limit.
        """
        count = int(self.ns["CAL_PROFILES_PER_POSE"] if count is None else count)
        raw = self.profile_rows(count)
        self.grabs.append(raw)
        if raw.shape[0] == 0:
            raise _Refused("no profiles on %s in %.0f s." % (topic, timeout))
        ok = cs.check_profile(raw, ref_dist=REF, range_m=RANGE,
                              source="the Keyence on %s" % topic,
                              n_profiles=count)
        self.ns["_CAL"]["last_profile_check"] = ok
        if check and not ok["ok"]:
            raise _Refused(ok["message"])
        return raw[np.isfinite(raw).all(axis=1)], "lj_v7080_optical_frame", count

    # -- what the arm was asked for, in the units that matter ---------------
    def commanded_standoffs(self):
        """(mm) head-to-MESH for every pose flown, recovered from the pose.

        Computed from where the optical frame landed and the mesh's own height
        under it, not from what the gauge said it intended -- so an arithmetic
        slip between the two shows up here.
        """
        out = []
        for _label, _xyz, T in self.flown:
            t = T[:3, 3]
            k = int(np.argmin((self.verts[:, 0] - t[0])**2
                              + (self.verts[:, 1] - t[1])**2))
            out.append(1e3*(t[2] + REF - self.verts[k, 2]))
        return out

    def actual_standoffs(self):
        """(mm) head-to-TRUTH for every pose flown. The number that hurts."""
        out = []
        for _label, _xyz, T in self.flown:
            t = T[:3, 3]
            out.append(1e3*(t[2] + REF - float(self.truth(t[0], t[1]))))
        return out


def run(cell, **kw):
    kw.setdefault("mesh", cell.mesh)
    kw.setdefault("say", cell.say)
    return cell.ns["gauge_mesh_against_keyence"](**kw)


# ==============================================================================
# The arithmetic, re-derived
# ==============================================================================

@_checked
def test_the_window_arithmetic_is_what_the_header_claims():
    """100 mm tolerates -3..+43 mm, and the three rungs cover -43..+43."""
    ns = fresh()
    rungs, why = ns["gauge_standoff_rungs"]()
    lo, hi = RANGE
    s0 = rungs[0]
    check("start standoff is the reference distance + 20 mm",
          abs(s0 - (REF + 0.020)) < 1e-12, "%.4f" % s0)
    # delta for which the grab returns, at the first rung
    d_lo, d_hi = s0 - hi, s0 - lo
    check("one position at 100 mm returns for -3 .. +43 mm of mesh error",
          abs(1e3*d_lo + 3.0) < 1e-6 and abs(1e3*d_hi - 43.0) < 1e-6,
          "%.1f .. %.1f mm" % (1e3*d_lo, 1e3*d_hi))
    check("the cell's own +12.45 mm is inside that, with room",
          d_lo < BIAS_M < d_hi)
    check("the ladder is 100, 80, 60 mm",
          [round(1e3*v) for v in rungs] == [100, 80, 60],
          str([1e3*v for v in rungs]))
    # Contiguous coverage: each rung's window must overlap the next.
    wins = [(s - hi, s - lo) for s in rungs]
    gaps = [wins[i][0] - wins[i + 1][1] for i in range(len(wins) - 1)]
    check("consecutive rungs overlap, so a 20 mm step cannot skip the surface",
          all(g < 0 for g in gaps), str([1e3*g for g in gaps]))
    check("the three of them cover -43 .. +43 mm of mesh error",
          abs(1e3*wins[-1][0] + 43.0) < 1e-6 and abs(1e3*wins[0][1] - 43.0) < 1e-6,
          "%.1f .. %.1f" % (1e3*wins[-1][0], 1e3*wins[0][1]))
    check("and it says how much mesh error the ladder can find",
          "return for" in why or "refused" in why, why)


@_checked
def test_the_ladder_will_not_descend_past_the_contact_floor():
    """tries=10 still gives three rungs, and names the reading that stops it."""
    ns = fresh()
    rungs, why = ns["gauge_standoff_rungs"](tries=10)
    check("ten rungs asked for, three flown", len(rungs) == 3,
          str([1e3*v for v in rungs]))
    check("the refusal is the near-side reading of a no-return",
          "NEAR side" in why and "%.0f mm" % (1e3*(rungs[0] - RANGE[0])) in why,
          why)
    # And the fourth rung it refused would really have been dangerous: under
    # the near-side reading the head would be past the surface.
    s4 = rungs[-1] - 0.020
    check("the fourth rung would imply contact under that reading",
          s4 - (rungs[0] - RANGE[0]) < 0.0,
          "%.1f mm" % (1e3*(s4 - (rungs[0] - RANGE[0]))))


@_checked
def test_a_step_wider_than_the_window_is_refused():
    """A 50 mm step over a 46 mm window can step right over the surface."""
    ns = fresh()
    try:
        ns["gauge_standoff_rungs"](step=0.050)
        check("a step wider than the window is refused", False, "it was not")
    except _Refused as e:
        check("a step wider than the window is refused",
              "SKIP THE SURFACE" in str(e), str(e)[:90])


# ==============================================================================
# The happy path
# ==============================================================================

@_checked
def test_the_first_grab_finds_the_twelve_millimetre_bias():
    """+12.45 mm, three points, one try each, and no second pose flown."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = run(cell)
    check("ok", r["ok"] is True, r["why"])
    check("three of three measured",
          (r["n_measured"], r["n_requested"]) == (3, 3), str(r))
    check("dz is the bias, to a hundredth of a millimetre",
          abs(r["dz_mm"] - 1e3*BIAS_M) < 0.01, "%.3f mm" % r["dz_mm"])
    check("positive means the mesh reads LOW", r["dz_mm"] > 0)
    check("one try at every point",
          [s["tries_used"] for s in r["samples"]] == [1, 1, 1],
          str([s["tries_used"] for s in r["samples"]]))
    check("three poses flown, no more", len(cell.flown) == 3,
          str(len(cell.flown)))
    check("why is empty when it worked", r["why"] == "", r["why"])
    check("and it says so out loud",
          any("MESH READS" in s for s in cell.said))
    # the two independent readings of delta agree
    gaps = [abs(s["delta_mm"] - s["delta_from_standoff_mm"])
            for s in r["samples"]]
    check("the world-frame and sensor-frame readings agree",
          max(gaps) < 0.01, "%.4f mm" % max(gaps))


@_checked
def test_the_commanded_standoff_is_a_hundred_millimetres_off_the_mesh():
    """The pose arithmetic, checked against the mesh rather than the intent."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    run(cell)
    cmd = cell.commanded_standoffs()
    act = cell.actual_standoffs()
    check("every pose stands the head 100 mm off the MESH",
          all(abs(v - 100.0) < 0.5 for v in cmd), str(cmd))
    check("which is 87.55 mm off the real surface",
          all(abs(v - (100.0 - 1e3*BIAS_M)) < 0.5 for v in act), str(act))
    check("and that is 30 mm clear of the 57 mm limit",
          min(act) - 1e3*RANGE[0] > 25.0, "%.1f mm" % (min(act) - 57.0))


@_checked
def test_the_samples_are_spread_over_the_mesh_and_lie_on_it():
    """Three non-collinear points on the mesh's own vertices, not a cluster."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    P = ns["gauge_sample_points"](("mesh", cell.verts, cell.tris), n=3)
    check("three points", P.shape == (3, 3), str(P.shape))
    on = [bool(np.any(np.all(np.abs(cell.verts - p) < 1e-12, axis=1)))
          for p in P]
    check("each one is a vertex of the mesh, so its z IS the prediction",
          all(on), str(on))
    d = [float(np.hypot(*(P[i, :2] - P[j, :2])))
         for i in range(3) for j in range(i + 1, 3)]
    check("spread over most of a 200 mm mesh", min(d) > 0.08,
          "%.3f m apart at the closest" % min(d))
    # The 2-D cross product written out: np.cross on 2-vectors was removed in
    # numpy 2 and the host is on it while the cell is on 1.23.
    u, v = P[1, :2] - P[0, :2], P[2, :2] - P[0, :2]
    area = abs(float(u[0]*v[1] - u[1]*v[0]))
    check("and not collinear, so a tilt is measurable in both directions",
          area > 1e-3, "%.5f" % area)


# ==============================================================================
# The ladder
# ==============================================================================

@_checked
def test_a_mesh_that_reads_high_needs_two_steps_down():
    """-30 mm: nothing at 100, nothing at 80, the answer at 60."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=-0.030)
    r = run(cell, n=1)
    check("measured in the end", r["ok"] is True, r["why"])
    check("it took three rungs -- two steps down",
          r["samples"][0]["tries_used"] == 3,
          str(r["samples"][0]["tries_used"]))
    check("and the head ended at 60 mm off the mesh",
          abs(1e3*r["samples"][0]["standoff_used"] - 60.0) < 1e-6,
          str(r["samples"][0]["standoff_used"]))
    check("dz is -30 mm", abs(r["dz_mm"] + 30.0) < 0.05,
          "%.3f mm" % r["dz_mm"])
    cmd = cell.commanded_standoffs()
    check("the three poses were 100, 80 and 60 mm off the mesh",
          [round(v) for v in cmd] == [100, 80, 60], str(cmd))
    act = cell.actual_standoffs()
    check("and NONE of them was ever inside the near limit",
          min(act) > 1e3*RANGE[0], "%.1f mm at the closest" % min(act))
    check("the real standoffs walked DOWN from 130 to 90 mm",
          [round(v) for v in act] == [130, 110, 90], str(act))


@_checked
def test_a_point_that_never_returns_is_a_failure_with_a_reason():
    """-60 mm is past the whole ladder: ok False, why populated, nothing hurt."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=-0.060)
    r = run(cell, n=1)
    check("not ok", r["ok"] is False)
    check("nothing measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("one point was asked for", r["n_requested"] == 1)
    check("why says no rung returned",
          "no rung of the ladder" in r["why"], r["why"])
    check("all three rungs were tried", r["samples"][0]["tries_used"] == 3,
          str(r["samples"][0]["tries_used"]))
    check("three poses flown and then it stopped", len(cell.flown) == 3,
          str(len(cell.flown)))
    act = cell.actual_standoffs()
    check("and the head never came within 100 mm of the part",
          min(act) > 100.0, "%.1f mm" % min(act))


@_checked
def test_the_sensor_saying_NEAR_stops_the_run_instead_of_stepping_down():
    """RULE B. A no-return is not always the far side, and the ladder must not
    assume it is when the sensor itself has said otherwise.

    The mesh reads 41 mm low, so the head at 100 mm is really at 59 mm -- inside
    the guarded window, at the bottom of the range, where a triangulation
    profiler does not fail cleanly but returns one repeated number. That is the
    failure mode this gauge exists to catch. check_profile() refuses the grab AND
    reports the standoff it implies, so the gauge can tell this from a surface
    beyond the far limit and must NOT step the head down another 20 mm."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=0.041, degenerate=True)
    try:
        r = run(cell, n=1)
        check("RULE B stops the run", False, "it returned %r" % (r["why"],))
    except _Refused as e:
        check("RULE B stops the run", "inside the" in str(e)
              and "near limit" in str(e), str(e)[:120])
        check("and says the mesh reads LOW", "READS LOW" in str(e), str(e)[:120])
    check("exactly one pose was flown -- it did not descend",
          len(cell.flown) == 1, str([f[0] for f in cell.flown]))
    act = cell.actual_standoffs()
    check("the one pose was at 59 mm, and a 20 mm step would have been 39",
          abs(act[0] - 59.0) < 0.5, "%.1f mm" % act[0])
    check("the last result is kept for an operator who lost the traceback",
          ns["last_gauge_result"] is not None)


@_checked
def test_a_grab_inside_the_guard_band_is_reported_even_when_it_works():
    """35 mm low still measures -- and still has to be said out loud."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=0.039)
    r = run(cell, n=1)
    check("it measured", r["ok"] is True, r["why"])
    check("dz is 39 mm", abs(r["dz_mm"] - 39.0) < 0.1, "%.2f" % r["dz_mm"])
    check("and the guard band is named in the notes",
          any("near limit" in n for n in r["notes"]), str(r["notes"]))
    check("a raster over this mesh would fly inside the hard limit",
          r["implied_standoff_mm"] < 1e3*RANGE[0],
          "%.1f mm" % r["implied_standoff_mm"])
    check("so it is over tolerance", r["within_tolerance"] is False)


# ==============================================================================
# Refusals -- the tests that are worth the most
# ==============================================================================

@_checked
def test_a_start_standoff_inside_the_near_limit_is_refused():
    """RULE A, at the first rung, where the mesh is all there is to go on."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    try:
        run(cell, n=1, start_standoff=0.060)
        check("a 60 mm start is refused", False, "it was not")
    except _Refused as e:
        check("a 60 mm start is refused", "near limit" in str(e), str(e)[:120])
    check("and nothing flew", cell.flown == [], str(cell.flown))
    # 62 mm is the boundary: lo + margin. One millimetre either side of it.
    for s, want in ((0.0619, False), (0.0621, True)):
        ns2 = fresh()
        c2 = Cell(ns2, truth=0.9218, delta=0.0)
        try:
            run(c2, n=1, start_standoff=s)
            got = True
        except _Refused:
            got = False
        check("start at %.1f mm is %s" % (1e3*s, "allowed" if want else "refused"),
              got is want)


@_checked
def test_there_is_no_mesh_and_it_refuses_rather_than_guessing():
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    try:
        cell.ns["gauge_mesh_against_keyence"](say=cell.say)
        check("no mesh is refused", False, "it was not")
    except _Refused as e:
        check("no mesh is refused", "no mesh" in str(e), str(e)[:120])
        check("and it says the arm has not moved", "not moved" in str(e),
              str(e)[:200])
    check("nothing flew", cell.flown == [])


@_checked
def test_a_plane_surface_needs_the_points_spelled_out():
    """A plane is infinite; there is no extent to spread three points over."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    try:
        cell.ns["gauge_mesh_against_keyence"](mesh=0.9218 - BIAS_M, say=cell.say)
        check("a bare plane with no points is refused", False, "it was not")
    except _Refused as e:
        check("a bare plane with no points is refused", "points=" in str(e),
              str(e)[:120])
    r = cell.ns["gauge_mesh_against_keyence"](
        mesh=0.9218 - BIAS_M, say=cell.say,
        points=[(0.75, -0.05), (0.85, -0.05), (0.80, 0.06)])
    check("and works when they are given", r["ok"] is True, r["why"])
    check("with the right answer", abs(r["dz_mm"] - 1e3*BIAS_M) < 0.05,
          "%.3f" % r["dz_mm"])


@_checked
def test_a_pose_that_cannot_be_flown_costs_that_point_and_not_the_run():
    """The other two points are still a measurement.

    ON BOTH ROUTES, which is why both are driven here: the unattended one
    exhausts the straight line and then the joint move, the confirmed one
    refuses out of _cal_fly_or_refuse(), and the sample bookkeeping has to be
    the same either way.
    """
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                refuse_cartesian=("gauge 2/3",), refuse_joints=("gauge 2/3",))
    r = run(cell)
    check("not ok, because a point is missing", r["ok"] is False)
    check("two of three measured",
          (r["n_measured"], r["n_requested"]) == (2, 3), str(r["n_measured"]))
    check("and the two that flew still give the bias",
          abs(r["dz_mm"] - 1e3*BIAS_M) < 0.02, "%.3f" % r["dz_mm"])
    check("why names the point that did not fly",
          "gauge 2/3" in r["why"], r["why"])
    check("it did not step the refused point down the ladder",
          r["samples"][1]["tries_used"] == 1,
          str(r["samples"][1]["tries_used"]))
    check("and the refusal names every route it tried",
          "cartesian" in r["samples"][1]["why"]
          and "joint move" in r["samples"][1]["why"],
          r["samples"][1]["why"])
    # THE CONFIRMED ROUTE, same bookkeeping.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M, refuse=("gauge 2/3",),
              answer_prompts=True)
    r2 = run(c2, confirm_moves=True)
    check("the confirmed route loses the same one point",
          (r2["n_measured"], r2["n_requested"]) == (2, 3),
          str(r2["n_measured"]))
    check("and gives the same bias from the other two",
          abs(r2["dz_mm"] - 1e3*BIAS_M) < 0.02, "%.3f" % r2["dz_mm"])


@_checked
def test_a_monitor_stop_is_not_swallowed():
    """CaptureStopped means a human has to look at the cell. It goes straight
    up, and the partial result is still printed and kept."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, stop_at=2)
    try:
        run(cell)
        check("a monitor stop propagates", False, "it did not")
    except _Stopped:
        check("a monitor stop propagates", True)
    r = ns["last_gauge_result"]
    check("and the two points already measured are kept",
          r is not None and r["n_measured"] == 2,
          str(None if r is None else r["n_measured"]))
    check("the laser was switched off on the way out",
          cell.laser == [True, False], str(cell.laser))


@_checked
def test_the_work_volume_decides_which_part_of_the_mesh_is_sampled():
    """A vertex the arm may not be over is not a sample point."""
    ns = fresh()
    env = {"lo": np.array([0.78, -0.10, 0.0]),
           "hi": np.array([0.90, 0.10, 1.5]),
           "top": 1.5, "floor": 0.0, "ceiling": 1.6}
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, envelope=env)
    r = run(cell)
    xs = [s["xy"][0] for s in r["samples"]]
    check("every sample is inside the work volume in x",
          min(xs) >= 0.78 - 1e-9, str(xs))
    check("and it still measured all three", r["n_measured"] == 3, r["why"])
    ns2 = fresh()
    tiny = {"lo": np.array([0.70, -0.10, 0.0]),
            "hi": np.array([0.701, -0.099, 1.5]),
            "top": 1.5, "floor": 0.0, "ceiling": 1.6}
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M, envelope=tiny)
    try:
        run(c2)
        check("a volume that excludes the part is refused", False, "it was not")
    except _Refused as e:
        check("a volume that excludes the part is refused",
              "work volume" in str(e), str(e)[:120])


@_checked
def test_the_standoff_gate_is_asked_before_every_flight():
    """The plan-time guard runs on the gauge's own poses, not around them.

    It casts the sensor's own optical axis from the pose and measures how far
    the first surface really is -- which is not the same question as "how far
    was the pose intended to be". Here the first aim point claims the surface is
    50 mm lower than the mesh says, so the head is placed 50 mm too low and the
    ray finds the real mesh 50 mm before the waypoint: a commanded standoff of
    50 mm against a near floor of 62. Refused, before the arm moves, and it
    costs that point and not the other two."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    V = cell.verts
    # Three aim points as (x, y, z) triples, so the mesh's own heights are
    # NOT what places the head -- one of them lies about where the surface is.
    picks = [V[np.argmin((V[:, 0] - x)**2 + (V[:, 1] - y)**2)]
             for x, y in ((0.75, -0.05), (0.85, -0.05), (0.80, 0.06))]
    pts = [list(picks[0][:2]) + [float(picks[0][2]) - 0.050]] + \
          [[float(p[0]), float(p[1]), float(p[2])] for p in picks[1:]]
    r = cell.ns["gauge_mesh_against_keyence"](mesh=cell.mesh, points=pts,
                                              say=cell.say)
    refused = [s for s in r["samples"] if "standoff gate" in (s["why"] or "")]
    check("the pose the gate can see through the part is refused",
          len(refused) == 1, str([(s["why"] or "")[:70] for s in r["samples"]]))
    check("nothing flew for it", len(cell.flown) == 2, str(len(cell.flown)))
    check("the other two still measured", r["n_measured"] == 2, r["why"])
    check("and the run is not ok, because a point is missing",
          r["ok"] is False)


@_checked
def test_rule_a_refuses_the_band_the_sensors_window_would_allow():
    """57 mm is inside the LJ-V7080's range and is still too close to fly.

    The gate is held to the sensor's own hard window, so a pose whose ray finds
    the surface at 58 mm passes it -- 58 mm is a legal standoff. RULE A is the
    separate check that says a GAUGE will not command the last five
    millimetres of that range: it is the band where the profiler stops returning
    a surface and starts returning noise. Here the aim
    point claims the surface is 42 mm below where the mesh puts it."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    V = cell.verts
    picks = [V[np.argmin((V[:, 0] - x)**2 + (V[:, 1] - y)**2)]
             for x, y in ((0.75, -0.05), (0.85, -0.05), (0.80, 0.06))]
    pts = [[float(picks[0][0]), float(picks[0][1]), float(picks[0][2]) - 0.042]] \
        + [[float(p[0]), float(p[1]), float(p[2])] for p in picks[1:]]
    r = cell.ns["gauge_mesh_against_keyence"](mesh=cell.mesh, points=pts,
                                              say=cell.say)
    why = r["samples"][0]["why"] or ""
    check("RULE A refuses it", "standoff gate" in why, why[:140])
    check("and names the floor it is under", "floor this rung may command" in why,
          why[:200])
    check("the ray really did find the surface at 58 mm, inside the sensor's "
          "own range", "58.0 mm from the head" in why, why[:140])
    check("nothing flew for it", len(cell.flown) == 2, str(len(cell.flown)))
    check("the other two still measured", r["n_measured"] == 2, r["why"])


# ==============================================================================
# The plane fit
# ==============================================================================

@_checked
def test_the_plane_fit_recovers_a_known_tilt():
    """13.6 mm across the plate and 22 mm at one end is a tilt, not an offset.

    A 40 mrad gradient in the mesh error over the sampled ring, recovered from
    four heights. The tilt is what says "one number will not fix this"."""
    ns = fresh()
    grad = 0.040                       # metres of error per metre of x
    x0 = 0.80
    cell = Cell(ns, truth=0.9218,
                delta=lambda X, Y: BIAS_M + grad*(np.asarray(X, float) - x0))
    r = run(cell, n=4)
    check("four points measured", r["n_measured"] == 4, r["why"])
    check("the mean offset is the bias at the centre",
          abs(r["dz_mm"] - 1e3*BIAS_M) < 0.5, "%.3f mm" % r["dz_mm"])
    check("the tilt is 40 mrad",
          r["tilt_mrad"] is not None and abs(r["tilt_mrad"] - 1e3*grad) < 1.0,
          str(r["tilt_mrad"]))
    check("and the residual about that plane is small",
          r["residual_mm"] < 0.05, "%.4f mm" % r["residual_mm"])
    check("the resolution floor is reported beside it",
          r["tilt_resolution_mrad"] is not None
          and r["tilt_resolution_mrad"] < r["tilt_mrad"],
          str(r["tilt_resolution_mrad"]))
    check("a tilted mesh error is not one number",
          r["dz_sd_mm"] > 1.0, "%.3f mm sd" % r["dz_sd_mm"])


@_checked
def test_three_points_fit_a_plane_exactly_and_the_residual_says_nothing():
    ns = fresh()
    cell = Cell(ns, truth=0.9218,
                delta=lambda X, Y: BIAS_M + 0.02*(np.asarray(X, float) - 0.80))
    r = run(cell, n=3)
    check("three points still give a tilt", r["tilt_mrad"] is not None,
          str(r["tilt_mrad"]))
    check("and the residual is exactly zero, which is arithmetic not evidence",
          r["residual_mm"] < 1e-9, "%.3e" % r["residual_mm"])


@_checked
def test_fewer_than_three_points_give_a_dz_and_no_tilt():
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = run(cell, n=2)
    check("two points measured",
          (r["n_measured"], r["n_requested"]) == (2, 2), str(r))
    check("dz is still the bias", abs(r["dz_mm"] - 1e3*BIAS_M) < 0.02,
          "%.3f" % r["dz_mm"])
    check("and the tilt is None, not a number", r["tilt_mrad"] is None,
          str(r["tilt_mrad"]))
    check("it says so rather than going quiet",
          any("tilt: not measured" in s for s in cell.said))
    ns1 = fresh()
    r1 = run(Cell(ns1, truth=0.9218, delta=BIAS_M), n=1)
    check("one point gives a dz and no tilt",
          r1["tilt_mrad"] is None and abs(r1["dz_mm"] - 1e3*BIAS_M) < 0.02,
          str(r1["tilt_mrad"]))


@_checked
def test_a_short_baseline_gets_no_tilt_at_all():
    """verify_calibration reported 15.0 mrad on a patch whose true tilt was
    1.1 mrad. A tilt over 20 mm of baseline is the sensor's noise."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = cell.ns["gauge_mesh_against_keyence"](
        mesh=cell.mesh, say=cell.say,
        points=[(0.800, 0.000), (0.810, 0.000), (0.805, 0.008)])
    check("it measured all three", r["n_measured"] == 3, r["why"])
    check("the offset is still right", abs(r["dz_mm"] - 1e3*BIAS_M) < 0.05,
          "%.3f" % r["dz_mm"])
    check("but no tilt is reported over a 13 mm baseline",
          r["tilt_mrad"] is None, str(r["tilt_mrad"]))
    check("and the reason is in the notes",
          any("baseline" in n or "span" in n for n in r["notes"]),
          str(r["notes"]))


@_checked
def test_collinear_points_get_no_tilt():
    """Three points on a line measure a gradient in one direction and let the
    fit choose the other -- the 90.06 degree answer, one dimension down."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = cell.ns["gauge_mesh_against_keyence"](
        mesh=cell.mesh, say=cell.say,
        points=[(0.72, 0.0), (0.80, 0.0), (0.88, 0.0)])
    check("three collinear points measured", r["n_measured"] == 3, r["why"])
    check("no tilt from a line", r["tilt_mrad"] is None, str(r["tilt_mrad"]))
    check("the offset is unaffected", abs(r["dz_mm"] - 1e3*BIAS_M) < 0.05,
          "%.3f" % r["dz_mm"])


# ==============================================================================
# The contract, and the .dpi as the loader will really run it
# ==============================================================================

@_checked
def test_the_contract_keys_are_all_there():
    """Another module is coded against these names. They do not move."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = run(cell)
    for k in ("ok", "n_measured", "n_requested", "dz_mm", "tilt_mrad",
              "residual_mm", "samples", "why"):
        check("result carries %r" % k, k in r, str(sorted(r)))
    for k in ("xy", "z_mesh", "z_keyence", "standoff_used", "tries_used",
              "n_profiles", "n_points"):
        check("each sample carries %r" % k, all(k in s for s in r["samples"]),
              str(sorted(r["samples"][0])))
    check("ok is a bool", isinstance(r["ok"], bool))
    check("the counts are ints",
          isinstance(r["n_measured"], int) and isinstance(r["n_requested"], int))
    check("dz_mm is a float", isinstance(r["dz_mm"], float))
    check("why is a string", isinstance(r["why"], str))
    for name, want in (("AZURE_GAUGE_START_STANDOFF", 0.100),
                       ("AZURE_GAUGE_POINTS", 3),
                       ("AZURE_GAUGE_MAX_MM", 2.0)):
        check("%s is %r" % (name, want), ns[name] == want, str(ns.get(name)))


@_checked
def test_the_tolerance_is_the_thing_a_caller_gates_on():
    ns = fresh()
    good = Cell(ns, truth=0.9218, delta=0.0015)
    r = run(good)
    check("1.5 mm passes the 2.0 mm tolerance", r["within_tolerance"] is True,
          "%.3f" % r["dz_mm"])
    ns2 = fresh()
    bad = Cell(ns2, truth=0.9218, delta=BIAS_M)
    r2 = run(bad)
    check("12.45 mm does not", r2["within_tolerance"] is False,
          "%.3f" % r2["dz_mm"])
    check("and the correction to apply is the disagreement",
          abs(r2["mesh_correction_mm"] - r2["dz_mm"]) < 1e-12)
    check("the consequence is stated in standoff",
          abs(r2["implied_standoff_mm"] - (1e3*REF - r2["dz_mm"])) < 1e-9,
          "%.2f" % r2["implied_standoff_mm"])
    check("which is 67.55 mm, 10.55 mm off the hard limit",
          abs(r2["margin_to_limit_mm"] - 10.55) < 0.05,
          "%.2f" % r2["margin_to_limit_mm"])


@_checked
def test_the_result_is_written_into_the_take():
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    with tempfile.TemporaryDirectory() as d:
        take = type("T", (), {"dir": d, "results": {}})()
        r = run(cell, take=take)
        p = os.path.join(d, "azure_gauge", "gauge.json")
        check("the samples and the verdict are on disk", os.path.exists(p), p)
        import json
        got = json.load(open(p))
        check("and they are the same numbers",
              abs(got["dz_mm"] - r["dz_mm"]) < 1e-12, str(got.get("dz_mm")))
        check("the take carries it too", "azure_gauge" in take.results)
    check("with a take, the laser is the caller's", cell.laser == [],
          str(cell.laser))


@_checked
def test_the_dpi_still_works_as_a_function_body():
    """The only test here that would catch a missing import or `global`.

    A .dpi is compiled and executed as a FUNCTION BODY. Its top-level
    `import numpy as np` is a LOCAL of that body, reachable from the functions
    only as a CLOSURE, and its `def`s are invisible to every other module
    unless declared global first. Exec'ing the file at module scope (which is
    what the rest of this suite does, and what every AST-lifting test does)
    passes whether or not those hold."""
    src = open(os.path.join(HERE, "azure_gauge.dpi")).read()
    body = "def _dpi_body():\n" + "".join(
        "    " + l + "\n" for l in src.splitlines())
    g = {"__name__": "azure_gauge_body"}
    exec(compile(body, "azure_gauge.dpi", "exec"), g)
    g["_dpi_body"]()
    for name in ("gauge_mesh_against_keyence", "gauge_standoff_rungs",
                 "gauge_sample_points", "describe_gauge",
                 "AZURE_GAUGE_START_STANDOFF", "AZURE_GAUGE_POINTS",
                 "AZURE_GAUGE_MAX_MM"):
        check("running it as a function body exports %r" % name, name in g,
              str(sorted(k for k in g if k.startswith(("gauge", "AZURE")))))
    # And the closure really is there: call something that needs np and math.
    g["CAL_KEYENCE_RANGE_M"] = RANGE
    g["CAL_KEYENCE_RANGE_MARGIN_M"] = MARGIN
    g["CalibrationRefused"] = _Refused
    rungs, _why = g["gauge_standoff_rungs"]()
    check("and its functions can still reach math and numpy",
          [round(1e3*v) for v in rungs] == [100, 80, 60], str(rungs))


@_checked
def test_describe_gauge_says_something_useful():
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = run(cell)
    lines = ns["describe_gauge"](r)
    text = "\n".join(lines)
    check("it names the direction", "LOW" in text, text[:200])
    check("it lists every sample", len(lines) >= 1 + len(r["samples"]),
          str(len(lines)))
    check("and it works with no argument at all",
          "LOW" in "\n".join(ns["describe_gauge"]()))


@_checked
def test_the_docstring_says_it_never_touches_the_part():
    """A standing instruction, and the kind that is only kept if it is written
    where the next person will read it."""
    doc = NS["gauge_mesh_against_keyence"].__doc__ or ""
    check("the docstring says it is an optical gauge",
          "OPTICAL GAUGE" in doc.upper(), doc[:100])
    check("and that nothing touches the part",
          "NOTHING HERE TOUCHES THE PART" in doc.upper(), doc[:200])
    src = open(os.path.join(HERE, "azure_gauge.dpi")).read()
    check("and there is no contact step anywhere in the file",
          "teach_point(" not in src and "capture_touch_off()" not in
          src.replace("capture_touch_off() measures it", ""))


# ==============================================================================
# THE PIPELINE IS NEVER PROMPTED BEFORE A MOVE
# ==============================================================================
# An autonomous run that flies _cal_fly_or_refuse() -> capture_move() ->
# _decide() -> confirm() stops at "[Enter=go, s=skip, q=abort] >" and waits for
# a human, once per pose. That chain is right for a calibration and wrong for
# the pipeline.
#
# EVERY Cell INSTALLS A TRIPWIRE: confirm, _decide and input all raise
# _Prompted, which is an AssertionError and so is not caught by any of the
# gauge's own handlers. So every test above ALREADY proves no prompt is issued
# on the default route -- any one of them would fail if one were. The tests
# here prove it deliberately, and prove the guards that route keeps.

@_checked
def test_the_pipeline_route_never_prompts():
    """The default route reaches no operator, and does not go near the
    confirmed one."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    r = run(cell)
    check("it measured all three points", r["n_measured"] == 3, r["why"])
    check("NOTHING PROMPTED THE OPERATOR", cell.prompts == [],
          str(cell.prompts))
    check("...and confirm/_decide/input are really wired to fail if reached",
          isinstance(ns["confirm"], type(cell._prompt)))
    check("it did not go through _cal_fly_or_refuse at all",
          cell.confirmed == [], str(cell.confirmed))
    check("it flew ros.CartesianMove instead, once per point",
          len(cell.cartesian) == 3, str(len(cell.cartesian)))
    check("the result says which route it took",
          r["confirm_moves"] is False, str(r.get("confirm_moves")))
    check("and the running commentary says so too, in words",
          any("no prompt" in s for s in cell.said),
          str([s for s in cell.said if "moves:" in s]))


@_checked
def test_the_tripwire_really_would_fire():
    """A test that proves the test. If the tripwire could not detect a prompt,
    every claim above would be vacuous."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    try:
        ns["confirm"]("go?")
        check("the tripwire fires when confirm() is called", False, "it did not")
    except _Prompted as e:
        check("the tripwire fires when confirm() is called",
              "PROMPTED" in str(e), str(e)[:80])
    check("...and it is an AssertionError, so no handler in the gauge eats it",
          issubclass(_Prompted, AssertionError))
    # And it is NOT one of the types the gauge catches.
    for t in (_Refused, _CaptureRefused, _Stopped, _Abort):
        check("a prompt is not a %s" % t.__name__,
              not issubclass(_Prompted, t))
    # AND THE CONFIRMED ROUTE REALLY GOES THROUGH IT. Without this the tripwire
    # would be watching a door nothing uses: the confirmed route is stubbed, and
    # a stub that skipped confirm() would let the default be switched back to it
    # with the suite still green.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M)
    try:
        run(c2, confirm_moves=True)
        check("the confirmed route trips it when nobody is answering", False,
              "it flew four moves with no operator")
    except _Prompted:
        check("the confirmed route trips it when nobody is answering", True)
    check("...on the very first move, before anything flew", c2.flown == [],
          str(c2.flown))


@_checked
def test_confirm_moves_true_is_the_hand_driven_route():
    """The confirmed route is still there for a call at the dgpy prompt, which
    IS calibration-ish. It must be opt-in, never the default."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, answer_prompts=True)
    r = run(cell, confirm_moves=True)
    check("it measured all three points", r["n_measured"] == 3, r["why"])
    check("it went through _cal_fly_or_refuse", len(cell.confirmed) == 3,
          str(cell.confirmed))
    check("AND IT DID PROMPT -- that is what this route is for",
          len(cell.prompts) == 3, str(cell.prompts))
    check("and issued no ros motion of its own", cell.cartesian == [],
          str(cell.cartesian))
    check("the result records that this run was confirmed",
          r["confirm_moves"] is True)
    check("and says so in the commentary",
          any("CONFIRMED" in s for s in cell.said),
          str([s for s in cell.said if "moves:" in s]))
    import inspect
    sig = inspect.signature(ns["gauge_mesh_against_keyence"])
    check("THE DEFAULT IS THE UNATTENDED ONE",
          sig.parameters["confirm_moves"].default is False,
          str(sig.parameters["confirm_moves"].default))


@_checked
def test_nothing_in_the_file_bypasses_the_operators_confirm():
    """THE UNATTENDED ROUTE MUST NOT BE A HARDWARE BYPASS.

    confirm() in cell_capture_node.dpi refuses to be automated anywhere but
    simulation, on purpose, and that contract is right: a capture scenario and a
    calibration run have somebody at the keyboard and every motion at the cell
    is theirs to approve. Dropping the pipeline's prompt by defeating that gate
    -- setting CAPTURE_AUTO_CONFIRM, rebinding confirm in the shared globals(),
    calling it and ignoring the answer -- would silently drop it from the
    scenarios too.

    READ BY AST, NOT BY grep. The header and three docstrings discuss all of
    these by name, and a substring test over the source cannot tell a warning
    from a violation. This looks at the code.
    """
    src = open(os.path.join(HERE, "azure_gauge.dpi")).read()
    tree = ast.parse("def _b():\n" + "".join(
        "    " + l + "\n" for l in src.splitlines()))
    names, calls, subs = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Call):
            f = node.func
            calls.add(f.id if isinstance(f, ast.Name)
                      else getattr(f, "attr", ""))
        elif isinstance(node, (ast.Subscript,)):
            k = node.slice
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                subs.add(k.value)
    check("the name CAPTURE_AUTO_CONFIRM appears nowhere in the CODE",
          "CAPTURE_AUTO_CONFIRM" not in names and
          "CAPTURE_AUTO_CONFIRM" not in subs, "it does")
    check("confirm() is never called", "confirm" not in calls, sorted(calls)[:5])
    check("_decide() is never called", "_decide" not in calls)
    check("and nothing is stored into globals()['confirm'] or ['_decide']",
          "confirm" not in subs and "_decide" not in subs, sorted(subs))
    check("input() is never called", "input" not in calls)
    check("CELL_MODE is never consulted, so there is no sim-only shortcut",
          "CELL_MODE" not in names, "it is")
    # And the one name it IS allowed to use is the parameter.
    check("what it does have is a confirm_moves parameter",
          "confirm_moves" in names)


# ==============================================================================
# The guards the unattended route keeps
# ==============================================================================
# Removing a prompt must not remove a check. Each of these drives one guard to
# a refusal and then asserts THAT NOTHING WAS COMMANDED -- a guard that fires
# after the arm has moved is not a guard.

@_checked
def test_the_work_volume_still_refuses_an_out_of_envelope_pose():
    """A volume the gauge's poses do not fit in: refused, per point, with the
    arm still parked."""
    ns = fresh()
    # A narrow strip in y that the park pose sits in (probe_end y = -0.0022) and
    # that every gauge pose leaves (y = +0.032, +0.032, -0.058). Leaving is
    # refused; entering would not be, which is why the strip and not a ceiling.
    box = {"lo": np.array([0.30, -0.02, 0.50]),
           "hi": np.array([1.30, 0.02, 1.40]),
           "top": 1.40, "floor": 0.50, "ceiling": 1.60}
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, volume=box)
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("the work volume is the reason", "work volume" in r["why"], r["why"])
    check("AND NOTHING WAS EVER COMMANDED", cell.cartesian == []
          and cell.joint_moves == [], "%r %r" % (cell.cartesian,
                                                 cell.joint_moves))
    check("the guard was asked once per rung of every point",
          len(cell.envelope_checks) >= 3, str(len(cell.envelope_checks)))
    check("and the arm never left the park pose", cell.flown == [],
          str(cell.flown))


@_checked
def test_a_cell_with_no_work_volume_flies_nothing():
    """capture_envelope() is None only when no box is declared, and then
    nothing moves -- the same rule capture_move() enforces."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, volume=None)
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("and it says the cell has no work volume",
          "no work volume" in r["why"], r["why"])
    check("nothing was commanded", cell.cartesian == [] and cell.flown == [])


@_checked
def test_incomplete_is_not_reported_as_out_of_tolerance():
    """A gauge that could not finish must not blame the mesh.

    `within_tolerance` is about the MESH; `ok` is about whether the gauge got to
    finish. Down one branch, a run that measures 2 of 3 points and finds the mesh
    0.70 mm out against a 2.0 mm tolerance -- the third point lost to a wrist
    angle -- would print "OVER 2.0 mm ... Correct the Azure depth by -0.70 mm".
    That is a correction nobody needs for a fault that is not the mesh, and
    acting on it means reconstructing again and being refused again.
    """
    ns = fresh()
    # The mesh is good; ONE point cannot be flown by either route, so `ok` is
    # False while `within_tolerance` stays True.
    cell = Cell(ns, truth=0.9218, delta=0.0007,
                refuse_cartesian=("gauge 2",), refuse_joints=("gauge 2",))
    r = run(cell)
    check("the mesh itself is inside tolerance", bool(r["within_tolerance"]),
          "dz %.2f vs %.1f" % (r["dz_mm"], r["tolerance_mm"]))
    check("but the gauge did not finish", not r["ok"], str(r["ok"]))
    check("it says INCOMPLETE", any("INCOMPLETE" in t for t in cell.said),
          str([t for t in cell.said if "mm" in t][-3:]))
    check("and does NOT say the mesh is over tolerance",
          not any("OVER %.1f mm" % (r["tolerance_mm"],) in t
                  for t in cell.said),
          str([t for t in cell.said if "OVER" in t]))
    check("and does NOT tell anyone to correct the Azure depth",
          not any("Correct the Azure depth" in t for t in cell.said),
          str([t for t in cell.said if "Correct" in t]))

    # AND THE REAL OUT-OF-TOLERANCE CASE STILL SAYS SO. Splitting the branch must
    # not lose the sentence that stops a raster over a bad mesh.
    ns2 = fresh()
    bad = Cell(ns2, truth=0.9218, delta=BIAS_M)
    r2 = run(bad)
    check("a mesh genuinely out of tolerance is still refused in words",
          not r2["within_tolerance"]
          and any("DO NOT PLAN A KEYENCE RASTER" in t for t in bad.said),
          "dz %.2f; %s" % (r2["dz_mm"],
                           [t for t in bad.said if "OVER" in t]))


@_checked
def test_the_prediction_is_the_executors_branch_not_a_global_one():
    """A wrist angle only a GLOBAL solve would choose must not refuse the move.

    THE CASE. The arm is parked at over_table (wrist_3 = +154 deg) and the next
    gauge point is a 162 mm lateral step. A global solve offers
    wrist_3 = -178.5 deg, which is the same TOOL ORIENTATION as +181.5, one turn
    away: +181.5 is 27.5 deg from where the arm stands and inside the cable band,
    while -178.5 is 332 deg away and outside it. No solver that follows its seed
    offers it -- but under the `transit` profile move_group is `mode: global`,
    which does not follow its seed, so _ik_target() must be asked local-only or
    the gauge refuses a point before trying either route.

    THE BAND IS NOT WIDENED BY ANY OF THIS. The destination is still checked --
    against the branch the node that will fly the move would choose, which is
    what a prediction has to be about. The second half of this test is the same
    cell with the LOCAL branch out of band, and it still refuses with nothing
    commanded.
    """
    ns = fresh()
    # In band where the arm is (+154) and in band at the destination (+181.5);
    # the global solver offers -178.5, which is not.
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                wrist_band=(math.radians(31.3), math.radians(301.3)),
                wrist3=math.radians(181.5),
                wrist3_global=math.radians(-178.5))
    r = run(cell)
    check("the gauge measured every point despite the global branch",
          r["n_measured"] == 3, "%d measured: %s" % (r["n_measured"], r["why"]))
    check("and it asked for the prediction LOCAL-ONLY",
          any(local for _x, local in cell.ik_asks), str(cell.ik_asks[:3]))
    check("no wrist check ever saw the global branch",
          not any(abs(q[5] - math.radians(-178.5)) < 1e-9
                  for _w, qs, _t in cell.wrist_checks for q in qs),
          str([[round(math.degrees(q[5]), 1) for q in qs]
               for _w, qs, _t in cell.wrist_checks][:3]))

    # THE SAME CELL WITH THE LOCAL BRANCH OUT OF BAND still refuses, and refuses
    # before anything is commanded. Removing a false refusal must not remove the
    # true one.
    ns2 = fresh()
    bad = Cell(ns2, truth=0.9218, delta=BIAS_M,
               wrist_band=(math.radians(31.3), math.radians(301.3)),
               wrist3=math.radians(-178.5),
               wrist3_global=math.radians(181.5))
    r2 = run(bad)
    check("an out-of-band LOCAL branch is still refused", r2["n_measured"] == 0,
          "%d measured" % (r2["n_measured"],))
    check("and still with nothing commanded",
          bad.cartesian == [] and bad.joint_moves == [],
          "%d cartesian, %d joint" % (len(bad.cartesian), len(bad.joint_moves)))


@_checked
def test_the_cable_band_still_refuses_before_anything_moves():
    """install_cable_guard()'s question, asked by the gauge itself on the
    configurations at both ends, at the tilt each one commands.

    An unchecked wrist 3 walked from +29 deg to -113 pulls the cables far enough
    that the run ends at the emergency stop. That check must survive the removal
    of the prompt, and it is not enough for `ros` to make it on the way out -- a
    refusal there is a failed move, while a refusal HERE is a named reason with
    the arm still parked.
    """
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                wrist_band=(math.radians(31.3), math.radians(301.3)),
                wrist3=math.radians(-113.0))
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("wrist 3 and the band are both named", "wrist 3" in r["why"]
          and "cable band" in r["why"], r["why"])
    check("AND NOTHING WAS EVER COMMANDED", cell.cartesian == []
          and cell.joint_moves == [])
    check("the band was asked for at the tilt commanded, not a global one",
          all(w[2] is not None for w in cell.wrist_checks),
          str([w[0] for w in cell.wrist_checks][:2]))
    # And a pose INSIDE the band flies.
    ns2 = fresh()
    ok = Cell(ns2, truth=0.9218, delta=BIAS_M,
              wrist_band=(math.radians(31.3), math.radians(301.3)),
              wrist3=math.radians(154.0))
    r2 = run(ok)
    check("a wrist 3 inside the band is not refused", r2["n_measured"] == 3,
          r2["why"])


@_checked
def test_an_unmeasured_cable_envelope_does_not_lock_the_cell_out():
    """No measurement means no check and it says so once -- a new cell has to
    be able to move before its cables have been measured, because the cable
    test is itself a motion."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, wrist_band=None,
                wrist3=math.radians(-113.0))
    r = run(cell)
    check("it still measured", r["n_measured"] == 3, r["why"])
    check("but the band was still ASKED, so an armed cell would refuse",
          len(cell.wrist_checks) >= 3, str(len(cell.wrist_checks)))


@_checked
def test_the_speed_limit_is_converted_and_not_handed_through():
    """CartesianMove's vel is a scaling, not a speed. path_motion_scaling()
    first, the cell-wide curve when IK along the path cannot be had."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    run(cell)
    vels = sorted(set(round(c["vel"], 6) for c in cell.cartesian))
    check("the scaling came from the path, not from the 50 mm/s limit",
          vels == [0.025], str(vels))     # path_motion_scaling = 0.5*v
    # Acceleration is NOT tied to velocity. It used to be, and that is a
    # different mistake from handing a speed through as a scaling: the units
    # are right and the limit is wrong. acc == vel makes the ramp TIME the
    # constant v_max/a_max -- 0.42-0.63 s on this arm whatever the speed -- so
    # a 56-60 mm descent at the 10 mm/s class spends over a second ramping and
    # never settles. See _gauge_accel() for what bounds the replacement.
    check("acceleration is NOT the velocity scaling",
          all(c["acc"] > c["vel"] + 1e-9 for c in cell.cartesian),
          str([(c["vel"], c["acc"]) for c in cell.cartesian]))
    check("it is the pipeline's near-the-part acceleration",
          all(abs(c["acc"] - 0.05) < 1e-12 for c in cell.cartesian),
          str(sorted(set(c["acc"] for c in cell.cartesian))))
    # With no path-aware bound, the cell-wide curve decides.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M)
    ns2["path_motion_scaling"] = lambda poses, v, quiet=True: None
    run(c2)
    vels2 = sorted(set(round(c["vel"], 6) for c in c2.cartesian))
    check("it falls back to the cell-wide velocity curve",
          vels2 == [0.032], str(vels2))   # motion_scaling = 0.64*v
    # And with neither, the limit itself -- which is what every caller did
    # before motion_scaling() existed.
    ns3 = fresh()
    c3 = Cell(ns3, truth=0.9218, delta=BIAS_M)
    ns3["path_motion_scaling"] = lambda poses, v, quiet=True: None
    ns3.pop("motion_scaling")
    run(c3)
    check("and with no calibration at all, the limit itself",
          sorted(set(round(c["vel"], 6) for c in c3.cartesian)) == [0.05],
          str([c["vel"] for c in c3.cartesian]))


@_checked
def test_the_acceleration_scaling_is_not_the_velocity_scaling():
    """_gauge_accel(): what the approach asks for, and what bounds it.

    THE POINT OF THE NUMBER. Velocity scaling and acceleration scaling are
    fractions of two different joint limits, so tying them together fixes the
    RAMP TIME at v_max/a_max regardless of speed -- 0.42-0.63 s on this arm --
    and at the gauge's own 0.0032 the descent is nearly all ramp. Raising the
    acceleration cannot raise the peak speed (the velocity scaling bounds joint
    speed and tari_cpp_mtc's checkTrajectory() enforces max_tip_speed in m/s),
    and it SHORTENS the travel after a stop is commanded, v^2/(2a): 3.1 mm at
    scaling 0.0032 against 0.2 mm at 0.05, at the 10 mm/s near-the-part class.
    """
    ns = fresh()
    accel = ns["_gauge_accel"]
    floor = ns["AZURE_GAUGE_ACC_SCALING"]
    check("the near-the-part acceleration is the one the EC raster already "
          "flies in contact", abs(floor - 0.05) < 1e-12, str(floor))
    check("a tiny velocity scaling does not drag the acceleration down with it",
          abs(accel(0.0032) - floor) < 1e-12, str(accel(0.0032)))
    check("nor does the eddy-current scaling",
          abs(accel(0.0047) - floor) < 1e-12, str(accel(0.0047)))
    # A fast away-from-the-part leg keeps at least its own velocity scaling:
    # handing such a move LESS acceleration than velocity is what puts the long
    # ramp back.
    check("a faster move is never given less acceleration than velocity",
          accel(0.2) >= 0.2 - 1e-12, str(accel(0.2)))
    # The ramp arithmetic the number exists for, at the arm's own limits:
    # 5.0 rad/s^2 and 180 deg/s (tari_ur_description/config/joint_limits.yaml).
    v_max, a_max = math.radians(180.0), 5.0
    tied = (0.0032 * v_max) / (0.0032 * a_max)
    fixed = (0.0032 * v_max) / (accel(0.0032) * a_max)
    check("tying them leaves a ramp of v_max/a_max however slow the move is",
          abs(tied - v_max/a_max) < 1e-9, "%.3f s" % tied)
    check("decoupling them shortens that ramp by more than an order of "
          "magnitude", fixed < tied / 10.0, "%.4f s vs %.4f s" % (fixed, tied))


@_checked
def test_a_gauge_step_is_flown_as_a_point_step():
    """cartesian_move() picks the IK settings a KIND of move needs, and a
    single waypoint is a step: 0.01 mm rather than 0.1, or it lands short."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    run(cell)
    kinds = sorted(set(c["kind"] for c in cell.cartesian))
    check("every gauge move went out as a point_step", kinds == ["point_step"],
          str(kinds))
    check("one waypoint per move", all(c["n"] == 1 for c in cell.cartesian))
    # A config without scan_plan.dpi still flies, through ros.CartesianMove.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M, no_wrapper=True)
    r2 = run(c2)
    check("with no cartesian_move() it falls back to ros.CartesianMove",
          r2["n_measured"] == 3 and len(c2.cartesian) == 3, r2["why"])
    check("and then there is no move kind to set",
          all(c["kind"] is None for c in c2.cartesian))


@_checked
def test_the_operator_can_still_stop_it_from_the_pendant():
    """scan_checkpoint() between points, which is how a sweep notices that
    External Control was stopped -- the unattended route's only operator
    control, since with no prompt there is nothing to answer and _Monitor is not
    reachable from here."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    run(cell)
    check("the supervisor was asked once per point",
          len(cell.checkpoints) == 3, str(cell.checkpoints))
    check("and it names the point", "1/3" in cell.checkpoints[0],
          str(cell.checkpoints[0]))
    # A config with no robot_supervisor.dpi has no checkpoint, and must still
    # fly: the supervisor is optional and this is not the place to require it.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M)
    ns2.pop("scan_checkpoint")
    r2 = run(c2)
    check("with no supervisor it still measures", r2["n_measured"] == 3,
          r2["why"])


@_checked
def test_a_pendant_abort_is_not_swallowed_and_keeps_what_it_measured():
    """ScanAborted is what scan_checkpoint() raises, and it is NOT one of the
    three exception types the confirmed route could produce -- so it had to be
    added to the stop-the-run tuple or a partial gauge would be thrown away and
    `last_gauge_result` would still hold the previous run's numbers."""

    class ScanAborted(Exception):
        """robot_supervisor.dpi's own type, declared where the gauge looks it
        up: in the shared globals(), at call time."""

    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    ns["ScanAborted"] = ScanAborted
    ns["last_gauge_result"] = "the previous run"
    calls = []

    def _abort(what="", timeout=None):
        calls.append(what)
        if len(calls) == 3:
            raise ScanAborted("%s: the operator aborted the scan" % (what,))
        return True

    ns["scan_checkpoint"] = _abort
    try:
        run(cell)
        check("a pendant abort propagates", False, "it did not")
    except ScanAborted:
        check("a pendant abort propagates", True)
    r = ns["last_gauge_result"]
    check("and the two points already measured were kept, not discarded",
          isinstance(r, dict) and r["n_measured"] == 2,
          str(r if not isinstance(r, dict) else r["n_measured"]))
    check("with a dz from those two", isinstance(r, dict)
          and abs(r["dz_mm"] - 1e3*BIAS_M) < 0.02,
          "%r" % (r.get("dz_mm") if isinstance(r, dict) else r,))
    check("the laser was switched off on the way out",
          cell.laser == [True, False], str(cell.laser))
    # And with no ScanAborted in the session, the other three still apply.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M)
    ns2.pop("ScanAborted", None)
    stops = ns2["_gauge_stops"]()
    check("the tuple falls back to the three that always exist",
          stops == (_Stopped, _Abort, KeyboardInterrupt), str(stops))
    ns2["ScanAborted"] = ScanAborted
    check("and picks ScanAborted up when it is there",
          ScanAborted in ns2["_gauge_stops"](), str(ns2["_gauge_stops"]()))
    ns2["ScanAborted"] = "not a class"
    check("a non-class ScanAborted is ignored rather than crashing the except",
          len(ns2["_gauge_stops"]()) == 3, str(ns2["_gauge_stops"]()))


@_checked
def test_a_session_with_no_capture_module_refuses_rather_than_flying():
    """No _check_envelope means nothing would check the work volume. That is a
    refusal, and it names the confirmed route as the alternative."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, no_pipeline=True)
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("it names the guard that is missing",
          "_check_envelope" in r["why"], r["why"])
    check("and says the arm has not moved", "has not moved" in r["why"],
          r["why"])
    check("and offers confirm_moves=True", "confirm_moves=True" in r["why"],
          r["why"])


# ==============================================================================
# The joint-space fallback, and why its check is FK and not its endpoints
# ==============================================================================

@_checked
def test_a_straight_line_that_will_not_plan_retries_in_joint_space():
    """verify_calibration's second route, kept: only the ENDPOINT of a gauge
    move is measured, so any route that arrives is as good as any other."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, refuse_cartesian=("gauge 2/3",))
    r = run(cell)
    check("all three points were still measured", r["n_measured"] == 3,
          r["why"])
    check("exactly one of them needed the joint route",
          len(cell.joint_moves) == 1, str(len(cell.joint_moves)))
    check("and it was the one whose straight line was refused",
          "gauge 2/3" in cell.joint_moves[0]["label"],
          cell.joint_moves[0]["label"])
    check("the joint move went at JOINT_MOVE_SCALING, not MoveIt's 0.1",
          abs(cell.joint_moves[0]["vel"] - 0.02) < 1e-12,
          str(cell.joint_moves[0]["vel"]))
    check("and it was given at least as long as the straight line would take",
          cell.joint_moves[0]["secs"] >= 4.0,
          str(cell.joint_moves[0]["secs"]))


@_checked
def test_a_joint_path_that_bulges_out_of_the_volume_is_refused():
    """THE WHOLE POINT OF CHECKING FK. Both ENDS of this move are inside the
    box and the middle is well above the ceiling: a joint move's tip does not
    travel in a straight line, so checking its endpoints checks nothing.

    The park pose is at z 1.10 and the target at 0.964, against a 1.60 m motion
    ceiling -- so an 800 mm bulge at the midpoint leaves the box and a 20 mm one
    does not, and the only difference between the two runs below is that.
    """
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                refuse_cartesian=("gauge",), joint_bulge=0.80)
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("the joint route was refused, not flown", cell.joint_moves == [],
          str(cell.joint_moves))
    check("and the refusal names the work volume", "work volume" in r["why"],
          r["why"])
    # The same bulge, small enough to stay inside: it flies.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M,
              refuse_cartesian=("gauge",), joint_bulge=0.02)
    r2 = run(c2)
    check("a bulge that stays inside the box is not refused",
          r2["n_measured"] == 3 and len(c2.joint_moves) == 3, r2["why"])


@_checked
def test_no_fk_means_no_joint_move():
    """capture_move_joints() refuses a move it cannot check. So does this: an
    unchecked joint path near a part is what the guards are for."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                refuse_cartesian=("gauge",), no_fk=True)
    r = run(cell)
    check("nothing was measured", r["n_measured"] == 0, str(r["n_measured"]))
    check("no joint move was made", cell.joint_moves == [],
          str(cell.joint_moves))
    check("and it says the path could not be checked",
          "compute_fk" in r["why"], r["why"])


@_checked
def test_no_ik_for_the_pose_leaves_only_the_straight_line():
    """No joint solution is not a reason to refuse the point -- the straight
    line may still plan -- but it is a reason there is no second route."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                refuse_cartesian=("gauge 1/3",), no_ik=("gauge 1/3",))
    r = run(cell)
    check("the point with no route at all failed", r["n_measured"] == 2,
          str(r["n_measured"]))
    check("and the refusal names both routes tried",
          "no IK solution" in r["samples"][0]["why"], r["samples"][0]["why"])
    check("the other two flew as straight lines", len(cell.joint_moves) == 0,
          str(cell.joint_moves))


@_checked
def test_a_move_with_no_success_flag_is_decided_by_arrival():
    """ros.MoveJoints falls back to a direct trajectory command when there is
    no execute_joint_move server, and returns a goal result with no success
    flag. A gauge must not read that as yes without checking."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                refuse_cartesian=("gauge",), joints_no_flag=True)
    r = run(cell)
    check("a move with no flag that DID arrive counts as flown",
          r["n_measured"] == 3, r["why"])
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M, refuse_cartesian=("gauge",),
              joints_no_flag=True, joints_no_arrive=True)
    r2 = run(c2)
    check("one that did NOT arrive does not", r2["n_measured"] == 0,
          str(r2["n_measured"]))
    check("and it says how far off the tool is",
          "from where it was sent" in r2["samples"][0]["why"],
          r2["samples"][0]["why"])


# ==============================================================================
# SPREAD OVER THE MESH, THE GAUGE MEASURES THE BENCH
# ==============================================================================
# With a detected region of x 0.783..0.909, y -0.147..0.137, a ring spread over
# a TSDF mesh of that scene puts its three points at (1.005, 0.057),
# (0.699, 0.057) and (0.855, -0.207) -- 96, 84 and 60 mm outside the region --
# where they return mesh z 0.9196/0.9209/0.9242 against a part top of 0.943.
# "The rough mesh agrees with the Keyence to -0.00 mm" is then a statement about
# the TABLE.

#: Measured geometry, so these are regression tests and not examples.
CELL_REGION = (0.783, -0.147, 0.909, 0.137)
CELL_BENCH_POINTS = ((1.005, 0.057), (0.699, 0.057), (0.855, -0.207))


@_checked
def test_the_2026_09_25_points_are_outside_the_region_they_claimed():
    """The premise, checked first: if these were inside the region there would
    be nothing to guard against."""
    x0, y0, x1, y1 = CELL_REGION
    outside = [(x, y) for x, y in CELL_BENCH_POINTS
               if not (x0 <= x <= x1 and y0 <= y <= y1)]
    check("all three of the flown points were outside the detected region",
          len(outside) == 3, str(outside))


@_checked
def test_sample_points_land_inside_a_given_region():
    """A mesh that extends 100 mm past the part on every side -- which is what
    a TSDF does -- and a region that says where the part is."""
    ns = fresh()
    x0, y0, x1, y1 = CELL_REGION
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                x0=x0 - 0.10, x1=x1 + 0.10, y0=y0 - 0.10, y1=y1 + 0.10,
                nx=61, ny=61)
    geom = ("mesh", cell.verts, cell.tris)
    P = ns["gauge_sample_points"](geom, n=3, region=CELL_REGION)
    inside = [(x0 <= float(p[0]) <= x1 and y0 <= float(p[1]) <= y1) for p in P]
    check("every sample point is inside the region", all(inside), str(P[:, :2]))
    # And still spread, not huddled in the middle: over AZURE_GAUGE_MIN_SPREAD_M,
    # which is the baseline below which the file will not report a tilt at all.
    d = [float(np.hypot(*(P[i, :2] - P[j, :2])))
         for i in range(3) for j in range(i + 1, 3)]
    check("and the baseline is still long enough to measure a tilt over",
          min(d) > ns["AZURE_GAUGE_MIN_SPREAD_M"], "%.3f m apart" % min(d))
    # WITHOUT a region: spread over the whole mesh, which is how points land on
    # the bench.
    Q = ns["gauge_sample_points"](geom, n=3, region=None)
    out = [(float(p[0]) < x0 or float(p[0]) > x1 or
            float(p[1]) < y0 or float(p[1]) > y1) for p in Q]
    check("with no region they spread over the whole mesh",
          any(out), str(Q[:, :2]))
    check("and all of them are still vertices of the mesh",
          all(bool(np.any(np.all(np.abs(cell.verts - p) < 1e-12, axis=1)))
              for p in np.vstack([P, Q])))


@_checked
def test_the_region_is_taken_from_the_detected_one_with_no_caller_change():
    """Stage 1 of run_automated_scan() leaves the region in `detected_region` on
    both of its branches, so stage 2a is already correct without anybody
    passing it -- the same fallback keyence_standoff.dpi uses."""
    ns = fresh()
    x0, y0, x1, y1 = CELL_REGION
    cell = Cell(ns, truth=0.9218, delta=BIAS_M,
                x0=x0 - 0.10, x1=x1 + 0.10, y0=y0 - 0.10, y1=y1 + 0.10,
                nx=61, ny=61)
    ns["detected_region"] = CELL_REGION
    r = run(cell)
    check("it measured all three", r["n_measured"] == 3, r["why"])
    check("the result records the region it used",
          r["region"] == list(CELL_REGION), str(r["region"]))
    check("and where it came from", r["region_from"] == "the detected region",
          str(r["region_from"]))
    inside = [(x0 <= s["xy"][0] <= x1 and y0 <= s["xy"][1] <= y1)
              for s in r["samples"]]
    check("EVERY POINT IT FLEW WAS OVER THE PART", all(inside),
          str([s["xy"] for s in r["samples"]]))
    check("and it says where it sampled", any("sampled inside" in s
                                              for s in cell.said),
          str([s for s in cell.said if "sampled" in s]))


@_checked
def test_the_caller_can_override_the_detected_region_either_way():
    """An explicit region wins; region=False says "the whole mesh" out loud."""
    ns = fresh()
    x0, y0, x1, y1 = CELL_REGION
    kw = dict(x0=x0 - 0.10, x1=x1 + 0.10, y0=y0 - 0.10, y1=y1 + 0.10,
              nx=61, ny=61)
    cell = Cell(ns, truth=0.9218, delta=BIAS_M, **kw)
    ns["detected_region"] = CELL_REGION
    mine = (0.80, -0.05, 0.86, 0.05)
    r = run(cell, region=mine)
    check("an explicit region wins over the detected one",
          r["region"] == list(mine) and r["region_from"] == "given to the gauge",
          "%r %r" % (r["region"], r["region_from"]))
    check("and the points are inside it",
          all(mine[0] <= s["xy"][0] <= mine[2]
              and mine[1] <= s["xy"][1] <= mine[3] for s in r["samples"]),
          str([s["xy"] for s in r["samples"]]))
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M, **kw)
    ns2["detected_region"] = CELL_REGION
    r2 = run(c2, region=False)
    check("region=False spreads over the whole mesh", r2["region"] is None,
          str(r2["region"]))
    check("and says so, rather than looking like it had no choice",
          "region=False" in (r2["region_from"] or ""), str(r2["region_from"]))
    check("...and warns that this can measure the bench instead of the part",
          any("MEASURE THE BENCH" in s for s in c2.said),
          str([s for s in c2.said if "NOT over a region" in s]))


@_checked
def test_no_region_anywhere_keeps_the_old_behaviour_exactly():
    """A hand call with nothing but a PLY has no region to be given, and must
    still work."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    check("the session has no detected region",
          ns.get("detected_region") is None)
    r = run(cell)
    check("it still measured all three", r["n_measured"] == 3, r["why"])
    check("the result says there was no region", r["region"] is None,
          str(r["region"]))
    check("and why", "nothing has detected a region" in (r["region_from"] or ""),
          str(r["region_from"]))


@_checked
def test_a_region_that_does_not_match_the_mesh_is_a_refusal():
    """A region left over from another run, or a mesh of something else. Loud,
    with the way out in the message, and nothing flown."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)     # mesh x 0.70..0.90
    try:
        run(cell, region=(1.50, -0.05, 1.60, 0.05))
        check("a region off the mesh is refused", False, "it was not")
    except _Refused as e:
        check("a region off the mesh is refused",
              "inside the region" in str(e), str(e)[:140])
        check("and it offers region=False as the way out",
              "region=False" in str(e), str(e)[:200])
    check("nothing was flown", cell.flown == [] and cell.cartesian == [])


@_checked
def test_the_region_goes_through_check_region_when_it_is_loaded():
    """(xmin, xmax, ymin, ymax) is what a person types, and the wrong order is
    silent. The gauge does not invent a second opinion about that -- it asks
    check_region()."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    seen = []

    def _check_region(region, what="this stage", quiet=False):
        seen.append((tuple(float(v) for v in region), what))
        raise ValueError("region for %s is (xmin, xmax, ymin, ymax); pass "
                         "(xmin, ymin, xmax, ymax)" % (what,))

    ns["check_region"] = _check_region
    try:
        run(cell, region=(0.78, 0.90, -0.10, 0.10))
        check("check_region's refusal is not swallowed", False, "it was")
    except ValueError as e:
        check("check_region's refusal is not swallowed",
              "xmin, ymin, xmax, ymax" in str(e), str(e)[:100])
    check("and it was asked with the gauge named", seen and "gauge" in seen[0][1],
          str(seen))
    check("nothing flew", cell.flown == [])


@_checked
def test_explicit_points_ignore_the_region():
    """points= says where to measure. That is the caller being explicit and the
    region has nothing to add to it."""
    ns = fresh()
    cell = Cell(ns, truth=0.9218, delta=BIAS_M)
    ns["detected_region"] = (0.86, -0.02, 0.88, 0.02)
    r = run(cell, points=[(0.75, -0.05), (0.85, 0.00), (0.80, 0.05)])
    check("all three named points were measured", r["n_measured"] == 3,
          r["why"])
    xs = sorted(round(s["xy"][0], 3) for s in r["samples"])
    check("and they are the points that were named, not the region's",
          xs == [0.75, 0.80, 0.85], str(xs))
    check("the commentary says the caller named them",
          any("the caller named" in s for s in cell.said),
          str([s for s in cell.said if "caller" in s]))
    check("and the saved result does not claim a region it did not use",
          r["region"] is None and "caller named" in (r["region_from"] or ""),
          "%r %r" % (r["region"], r["region_from"]))
    # An explicit region alongside explicit points says so rather than being
    # silently dropped.
    ns2 = fresh()
    c2 = Cell(ns2, truth=0.9218, delta=BIAS_M)
    r2 = run(c2, points=[(0.75, -0.05), (0.85, 0.00)],
             region=(0.86, -0.02, 0.88, 0.02))
    check("passing both says the region is ignored",
          any("region= is ignored" in s for s in c2.said),
          str([s for s in c2.said if "ignored" in s]))
    check("and the points still won", r2["n_measured"] == 2, r2["why"])


@_checked
def test_the_gauge_used_to_pass_over_a_mesh_that_was_wrong_on_the_part():
    """A MESH THAT IS RIGHT ON THE BENCH AND WRONG ON THE PART.

    A part 21 mm above the bench, and a mesh that is RIGHT on the bench and
    12.45 mm low on the part. That combination is the whole failure: the raster
    is planned over the part, so the part is the only place the disagreement
    matters.

    The mesh is 250 x 350 mm against a 126 x 284 mm region, which is what a TSDF
    of that part looks like, and with those extents the ring lands entirely
    outside the region -- all three points on the bench. Spread over the mesh
    the gauge then reports about half a millimetre and PASSES a mesh that would
    fly the head 12 mm closer to the part than intended; spread inside the
    region it measures the 12.45 mm and refuses.
    """
    x0, y0, x1, y1 = CELL_REGION

    def truth(X, Y):
        on = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
        return np.where(on, 0.943, 0.922)

    def delta(X, Y):
        """BIAS_M on the part; small and not quite flat on the bench, which is
        what turns the bench-to-part step into an apparent sd of 2.23 mm and a
        'tilt' of 18 mrad."""
        on = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
        return np.where(on, BIAS_M, 0.0005 + 0.010*(X - 0.85))

    kw = dict(x0=0.60, x1=1.10, y0=-0.35, y1=0.35, nx=81, ny=81)

    # -- SPREAD OVER THE MESH (region=False): it passes ---------------------
    ns2 = fresh()
    c2 = Cell(ns2, truth=truth, delta=delta, **kw)
    r2 = run(c2, region=False)
    on_part = [(x0 <= s["xy"][0] <= x1 and y0 <= s["xy"][1] <= y1)
               for s in r2["samples"]]
    check("with no region, none of the three points is on the part",
          not any(on_part), str([s["xy"] for s in r2["samples"]]))
    check("...so it reports about half a millimetre",
          abs(r2["dz_mm"]) < 1.0, "%.3f mm" % r2["dz_mm"])
    check("...AND PASSES A MESH THAT IS 12 mm WRONG WHERE THE RASTER WILL FLY",
          r2["within_tolerance"] is True, str(r2["within_tolerance"]))
    check("...with a spread and a tilt that are the bench, not the part",
          r2["dz_sd_mm"] > 0.3, "%.3f mm" % r2["dz_sd_mm"])

    # -- SPREAD INSIDE THE REGION: it refuses --------------------------------
    ns = fresh()
    cell = Cell(ns, truth=truth, delta=delta, **kw)
    r = run(cell, region=CELL_REGION)
    check("inside the region every point is on the part",
          all(x0 <= s["xy"][0] <= x1 and y0 <= s["xy"][1] <= y1
              for s in r["samples"]), str([s["xy"] for s in r["samples"]]))
    check("it measures the real 12.45 mm",
          abs(r["dz_mm"] - 1e3*BIAS_M) < 0.1, "%.3f mm" % r["dz_mm"])
    check("AND REFUSES THE MESH", r["within_tolerance"] is False,
          str(r["within_tolerance"]))
    check("the spread is now the sensor's, not a bench-to-part step",
          r["dz_sd_mm"] < 0.2, "%.3f mm" % r["dz_sd_mm"])
    check("and no tilt is invented",
          r["tilt_mrad"] is None or abs(r["tilt_mrad"]) < 1.0,
          str(r["tilt_mrad"]))
    check("it also says what a raster over this mesh would fly at",
          r["margin_to_limit_mm"] < 12.0, "%.1f mm" % r["margin_to_limit_mm"])


def test_zz_no_check_anywhere_in_this_file_failed():
    """THE BACKSTOP: a check that fired outside any test function.

    check() records a failure and prints it; it does not raise, which is what
    lets one test function report a dozen independent facts instead of stopping
    at the first false one. Undecorated, that means pytest reports every test
    above as PASSED however many of those facts are false, and the whole file is
    decorative under `pytest -q`, which is how the suite is run.

    @_checked asserts per test, so a broken check is reported against the test
    that ran it. This stays because it is free and it catches what that cannot:
    a check() called at import time, from a module-level fixture, or from a
    helper nobody decorated.

    Last in the file on purpose: pytest runs tests in definition order.
    """
    assert not FAILURES, "%d check(s) failed:\n  %s" % (
        len(FAILURES), "\n  ".join(FAILURES))


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and k != "test_zz_no_check_anywhere_in_this_file_failed"]
    for name, t in tests:
        print("\n%s" % name)
        # __wrapped__ is the undecorated test: it records failures instead of
        # raising, so one standalone run reports all of them.
        getattr(t, "__wrapped__", t)()
    print("\n%d test function(s), %d failure(s)" % (len(tests), len(FAILURES)))
    for f in FAILURES:
        print("  FAILED: %s" % f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
