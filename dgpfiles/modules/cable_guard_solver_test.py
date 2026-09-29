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
# File:  cable_guard_solver_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that the cable guard judges a move with the same IK solver that
#   will execute it.
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
"""The cable guard must judge a move with the solver that will FLY it.

THE FAULT this pins down, seen when parking after an EC plan was refused:

    CABLE GUARD REFUSED CartesianMove: a Cartesian move of 2 poses would turn
    wrist 3 to -189.9 deg (the move spans -189.9..-189.9). The cables allow
    +16.2..+301.1 deg at tilt (x -0.0, y +0.0) deg. Nothing moved.
    could not rise straight up to z 1.243; parking from z 1.026

-189.9 + 360 = +170.1 is in band and the arm was standing at +159..+177 deg, so
this reads like a guard that needs a +/-360 substitution. IT IS NOT, AND IT MUST
NOT GET ONE: the cable physically wraps, THE ABSOLUTE ANGLE GOVERNS, and
check_wrist3() was right about every number it was given.

It was given the wrong number. The guard predicts a pose move by calling
ros.SolveIK, which /move_group serves; a CartesianMove is executed by
/mtc_hub_node. Under the `transit` profile -- what a cell holds outside a
motion_profile() block -- /mtc_hub_node is `mode: local` and /move_group is
`mode: global`, deliberately, because move_group also serves RViz, teaching and
one-off branch-changing moves. A global solver does not follow its seed, so the
guard is handed a branch a full turn from where the arm stands and refuses a move
the executing node would have solved in band.

What is checked here, all of it against stubs (no ROS, no arm, no container):

  * THE VERDICT FLIPS WITH THE EXECUTING NODE'S MODE, and with nothing else. The
    same poses, the same stub solver, the same band: legal when the node that
    flies the move is local, refused when that node is itself global.
  * A GENUINELY OUT-OF-ENVELOPE MOVE IS STILL REFUSED. Matching the solver is
    not a way to get out of the band; an angle no turn of wrist 3 brings into
    band is refused in local mode too. That is the whole point of the guard,
    and the measured envelope is why it exists.
  * THE MODE IS PUT BACK -- after a pass, after a refusal, after the solve
    raises, after the check raises. A cell left in the guard's mode would refuse
    the operator's next interactive move, which is a worse failure than the one
    being fixed.
  * THE REFUSAL SAYS WHICH OF THE TWO FAULTS IT IS. "An IK solver picked the
    wrong branch" and "the toolpath wants a pose the cables cannot hold" have
    different fixes in different files, so they must not print identically:
    _cable_guard_refuse() must not print only the FIRST line of the message,
    because check_wrist3()'s branch note is the last.
  * A MOTION MODULE WITHOUT THE MODE API STILL GETS A GUARD. Degrading to
    "unverified prediction" is acceptable; degrading to "no check" is not.

    python3 cable_guard_solver_test.py        # every section, all failures
    python3 -m pytest -q cable_guard_solver_test.py
"""
import contextlib
import functools
import io
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


# NO geometry_msgs STUB HERE, DELIBERATELY, and it is not an oversight.
#
# Twelve suites in this directory install their own geometry_msgs stub into
# sys.modules at import time, and the last import wins -- pytest imports every
# file before it RUNS any of them. cell_capture_clearance_test.py's stub has a
# Pose with no fields and a Quaternion that takes no keywords, so _cap_pose()
# raised, the guard's _ik() swallowed it (an IK service that will not answer is
# a normal thing), _check() reported that it could judge no pose of the move,
# and EVERY MOVE PASSED. Running this file alone looked perfect; running bare
# `pytest` did not, and a guard test that passes everything is worse than none.
#
# (cable_envelope_test.py is exposed to the same collision and survives it only
# because its checks run at import time, alphabetically before that stub lands.)
#
# So nothing here goes near sys.modules: the poses below are objects with
# .position/.orientation, which is what _cap_xyzq() reads, and _ik() only builds
# a geometry_msgs Pose for a pose passed as a TUPLE.
class Pose(object):
    """A pose the guard can read without importing anything."""

    def __init__(self, xyz=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
        self.position = types.SimpleNamespace(x=float(xyz[0]), y=float(xyz[1]),
                                             z=float(xyz[2]))
        self.orientation = types.SimpleNamespace(x=float(q[0]), y=float(q[1]),
                                                 z=float(q[2]), w=float(q[3]))


_ENV_NS = {"__name__": "cable_guard_solver_test.envelope"}
exec(compile(_load("cable_envelope.dpi"), "cable_envelope.dpi", "exec"), _ENV_NS)

CableRefused = _ENV_NS["CableRefused"]
cable_envelope_from_rows = _ENV_NS["cable_envelope_from_rows"]
wrist3_branch_in_band = _ENV_NS["wrist3_branch_in_band"]

# Cell ex-ur10e-1's envelope as capture_cable_envelope() measured it. Verbatim
# from cable_limits_ex-ur10e-1_hardware.json, as cable_envelope_test.py carries
# it. The untilted band is 31.34..301.34 deg and every angle below is chosen
# against that, so the numbers here mean what the measurement means.
MEASURED = [
    {"tilt": "no tilt",         "lo_deg": 31.338424555636937, "hi_deg": 301.33842455563695},
    {"tilt": "tilt x +40 deg",  "lo_deg": 87.34219315297075,  "hi_deg": 327.3421931529707},
    {"tilt": "tilt x -40 deg",  "lo_deg": 179.55253199567213, "hi_deg": 254.55253199567213},
    {"tilt": "tilt y +20 deg",  "lo_deg": 63.598971284101005, "hi_deg": 303.598971284101},
    {"tilt": "tilt y -20 deg",  "lo_deg": 30.28146644195578,  "hi_deg": 255.28146644195579},
]
ENV = cable_envelope_from_rows(MEASURED, step_deg=15.0, cell="ex-ur10e-1",
                              mode="hardware", measured="2026-09-14T10:22:07")

#: The two branches of one refused pose. GLOBAL_DEG is what move_group hands
#: the guard; LOCAL_DEG is the same tool orientation one turn away, which is
#: where mtc_hub_node's local solver goes and is in band.
GLOBAL_DEG = -189.9
LOCAL_DEG = 170.1

#: Where the arm stands for that refusal: a Keyence raster leaves wrist 3 at
#: +159..+177 deg.
STANDING_DEG = 166.34

#: An angle NO turn of wrist 3 brings into the untilted band: +330 is past the
#: 301.34 top and -390 is outside the joint's +/-360 travel. This is the other
#: failure -- the toolpath asking for a pose the cables cannot hold.
UNREACHABLE_DEG = -30.0

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)
POSE = Pose(q=IDENTITY_Q)


def joints(wrist3_deg):
    """A plausible UR configuration with this wrist_3, in radians."""
    return [0.1, -1.2, 1.4, -1.7, -1.55, math.radians(wrist3_deg)]


# ------------------------------------------------------------------------------
# The stubs
# ------------------------------------------------------------------------------
class Node(object):
    """capture_node: the arm's own joints, and no FK (so the untilted band)."""

    def __init__(self, standing_deg=STANDING_DEG):
        self.standing = joints(standing_deg)
        self.events = []

    def current_joints(self):
        return list(self.standing)

    def fk(self, q, links=("probe_end",)):
        return None

    def event(self, payload):
        self.events.append(payload)


class Ros(object):
    """dgpython_rosmotion, with a SEPARATE IK mode per node -- which is the
    whole subject. SolveIK answers from move_group's mode, because that is the
    node that serves it: global hands back the far branch and ignores the seed,
    local stays on the seed's.

    Records every mode SolveIK was actually called in, so a test can pin the
    mechanism rather than only the verdict.
    """

    def __init__(self, modes=None, answers=None, no_mode_api=False,
                 get_raises=False, set_raises=False, set_raises_after=False,
                 solve_raises=False, move_raises=False, unknown_node=False):
        # /move_group global, everything else local: the `transit` profile, which
        # is what a cell holds outside a motion_profile() block.
        self.modes = dict(modes or {"/move_group": "global",
                                    "/mtc_hub_node": "local",
                                    "/dgpython_action_server": "local"})
        self.answers = dict(answers or {"global": joints(GLOBAL_DEG),
                                        "local": joints(LOCAL_DEG)})
        self.no_mode_api = no_mode_api
        self.get_raises = get_raises
        self.set_raises = set_raises
        # set_raises_after applies the change and THEN raises, which is what a
        # set_parameters whose reply times out looks like: the mode really is
        # different and SetIKMode's return value never arrives.
        self.set_raises_after = set_raises_after
        self.solve_raises = solve_raises
        self.move_raises = move_raises
        self.unknown_node = unknown_node
        self.commanded = []
        self.solve_modes = []
        self.seeds = []
        self.set_calls = []

    # -- what the guard asks about the executing node ----------------------
    def GetMotionParameters(self, names, nodes=None, timeout=5.0):
        if self.no_mode_api:
            raise AttributeError("GetMotionParameters")
        if self.get_raises:
            raise RuntimeError("/mtc_hub_node did not answer get_parameters")
        if self.unknown_node:
            return {}
        return {n: {k: self.modes[n] for k in names if k == "mode"}
                for n in (nodes or list(self.modes)) if n in self.modes}

    # -- move_group's own mode --------------------------------------------
    def IKMode(self, timeout=5.0):
        if self.no_mode_api:
            raise AttributeError("IKMode")
        return self.modes.get("/move_group")

    def SetIKMode(self, mode, timeout=5.0):
        if self.no_mode_api:
            raise AttributeError("SetIKMode")
        if mode not in ("global", "local"):
            raise ValueError("IK mode must be 'global' or 'local', not %r" % (mode,))
        if self.set_raises:
            raise RuntimeError("/move_group/set_parameters is not available")
        previous = self.modes.get("/move_group")
        self.set_calls.append(mode)
        if previous == mode:
            return previous
        self.modes["/move_group"] = mode
        if self.set_raises_after:
            raise RuntimeError("timed out setting the IK mode")
        return previous

    def SolveIK(self, pose, seed=None, timeout=2.0, **kw):
        mode = self.modes.get("/move_group")
        self.solve_modes.append(mode)
        self.seeds.append(None if seed is None else round(math.degrees(seed[5]), 1))
        if self.solve_raises:
            raise RuntimeError("/compute_ik did not answer")
        answer = self.answers.get(mode)
        return None if answer is None else list(answer)

    # -- the motions -------------------------------------------------------
    def _did(self, name):
        self.commanded.append(name)
        if self.move_raises:
            raise RuntimeError("the move failed")
        return "moved"

    def CartesianMove(self, *a, **k):
        return self._did("CartesianMove")

    def FreeMove(self, *a, **k):
        return self._did("FreeMove")

    def JointMove(self, *a, **k):
        return self._did("JointMove")

    def MoveJoints(self, *a, **k):
        return self._did("MoveJoints")

    def Move(self, *a, **k):
        return self._did("Move")

    def ApproachInConfiguration(self, *a, **k):
        self._did("ApproachInConfiguration")
        return [0.0]*6


def guarded(ros, node=None, envelope=ENV):
    """The REAL _CableGuardedRos from cell_capture_motion.dpi over `ros`."""
    ns = dict(_ENV_NS)
    ns.update(CELL_NAME="ex-ur10e-1", CELL_MODE="hardware",
              CAPTURE_ROOT="/nonexistent", _capture_say=lambda m: None,
              capture_node=node if node is not None else Node(),
              _cable_limits_file=lambda: "/nonexistent/none.json")
    exec(compile(_load("cell_capture_motion.dpi"), "cell_capture_motion.dpi", "exec"), ns)
    ns["_CAP"] = {"real_ros": ros}
    ns["_CABLE"][0] = envelope            # the measurement, without touching disk
    ns["_CABLE_LOOKED"][0] = True
    ns["ros"] = ros
    return ns["install_cable_guard"](quiet=True), ns


def run(fn):
    """Call fn(), returning (result, printed text). The guard reports a refusal
    by printing, so the text is part of what is under test."""
    buf = io.StringIO()
    out = None
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def not_blind(ns, ros, where):
    """The guard must have had something to judge.

    _check() returns without refusing when IK answered for NO pose of a move --
    correct behaviour, and indistinguishable from a working guard unless it is
    asked about. That is exactly how this file passed alone and passed nothing
    under bare pytest (see the note on the Pose class above), so every section
    that asserts a refusal asserts this too.
    """
    check("%s: the guard had IK answers to judge" % (where,),
          bool(ros.solve_modes) and not ns["_CAP"].get("cable_guard_blind"),
          "solves %s, blind %r" % (ros.solve_modes, ns["_CAP"].get("cable_guard_blind")))


def refused(res):
    """Did the guard hand back its 'failed move' stub rather than flying?"""
    inner = getattr(res, "result", None)
    return inner is not None and inner.success is False and inner.fraction == 0.0


# ------------------------------------------------------------------------------
# check() records; _checked turns a section's records into a pytest verdict.
# ------------------------------------------------------------------------------
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok    %s%s" % (name, ("  -- " + detail) if detail else ""))
    else:
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))
        FAILURES.append(name)


def _checked(fn):
    """check() records rather than raises, so a standalone run reports every
    failure in one pass. Under pytest that would be a test that PASSES while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# ==============================================================================
# 0. The fixture reproduces the refused pose
# ==============================================================================
@_checked
def test_the_fixture_reproduces_the_two_branches():
    print("\n0. the fixture is the refused pose's arithmetic")
    lo, hi = ENV.band_deg(tilt=(0.0, 0.0))
    check("the untilted band is the measured one",
          abs(lo - 31.3384) < 1e-3 and abs(hi - 301.3384) < 1e-3,
          "%.4f..%.4f deg" % (lo, hi))
    check("the global branch (-189.9 deg) is OUT of band",
          not ENV.contains(math.radians(GLOBAL_DEG), tilt=(0.0, 0.0)))
    check("the local branch (+170.1 deg) is IN band",
          ENV.contains(math.radians(LOCAL_DEG), tilt=(0.0, 0.0)))
    alt = wrist3_branch_in_band(math.radians(GLOBAL_DEG), (math.radians(lo), math.radians(hi)))
    check("so an in-band branch of the refused angle exists -- a SOLVER fault",
          alt is not None and abs(math.degrees(alt) - LOCAL_DEG) < 0.2,
          "%+.1f deg" % (math.degrees(alt),) if alt is not None else "none")
    check("the arm was standing in band, which is why it looked like a false alarm",
          ENV.contains(math.radians(STANDING_DEG), tilt=(0.0, 0.0)))
    bad = wrist3_branch_in_band(math.radians(UNREACHABLE_DEG),
                                (math.radians(lo), math.radians(hi)))
    check("and %+.0f deg has NO in-band branch -- a TOOLPATH fault" % (UNREACHABLE_DEG,),
          bad is None, "" if bad is None else "%+.1f" % (math.degrees(bad),))


# ==============================================================================
# 1. THE FIX: the verdict follows the EXECUTING node's mode
# ==============================================================================
@_checked
def test_the_verdict_follows_the_executing_nodes_mode():
    print("\n1. the same move, the same solver, the same band: only the "
          "executor's mode differs")

    # transit: move_group global, mtc_hub_node local -- the cell's wiring.
    ros = Ros()
    guard, ns = guarded(ros)
    res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
    not_blind(ns, ros, "transit cell")
    check("a CartesianMove is solved in the mode /mtc_hub_node holds",
          ros.solve_modes and set(ros.solve_modes) == {"local"}, str(ros.solve_modes))
    check("so the park rise FLIES instead of being refused",
          ros.commanded == ["CartesianMove"] and not refused(res),
          str(ros.commanded))
    check("and nothing was printed, because nothing was refused", text == "",
          repr(text))
    check("the solve was seeded from the arm's own joints",
          ros.seeds and ros.seeds[0] == round(STANDING_DEG, 1), str(ros.seeds))

    # A cell whose executing node really IS global: the refusal is then true.
    ros = Ros(modes={"/move_group": "global", "/mtc_hub_node": "global",
                     "/dgpython_action_server": "local"})
    guard, ns = guarded(ros)
    res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
    not_blind(ns, ros, "global-executor cell")
    check("with /mtc_hub_node global, the SAME move is refused",
          refused(res) and ros.commanded == [], str(ros.commanded))
    check("solved in global, because that is what the executor holds",
          set(ros.solve_modes) == {"global"}, str(ros.solve_modes))
    check("and the refusal says the prediction matched the executor",
          "/mtc_hub_node" in text and "BRANCH THE MOVE WOULD HAVE TAKEN" in text,
          text)

    # FreeMove is the same node; Move is dgpython_action_server.
    ros = Ros()
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.FreeMove(POSE))
    check("FreeMove is judged as /mtc_hub_node too",
          ros.commanded == ["FreeMove"] and set(ros.solve_modes) == {"local"},
          "%s %s" % (ros.commanded, ros.solve_modes))
    ros = Ros()
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.Move([POSE]))
    check("ros.Move is judged as /dgpython_action_server",
          ros.commanded == ["Move"] and set(ros.solve_modes) == {"local"},
          "%s %s" % (ros.commanded, ros.solve_modes))

    # ApproachInConfiguration chooses its configuration with its OWN SolveIK
    # calls, so move_group IS the deciding solver and matching it means matching
    # global. The refusal there is genuine.
    ros = Ros()
    guard, _ns = guarded(ros)
    out, text = run(lambda: guard.ApproachInConfiguration(POSE, POSE))
    check("ApproachInConfiguration is judged in MOVE_GROUP's mode, not local",
          set(ros.solve_modes) == {"global"}, str(ros.solve_modes))
    check("and is refused, because that is the branch it would really command",
          out is None and ros.commanded == [], str(ros.commanded))


# ==============================================================================
# 2. Matching the solver is NOT a way out of the band
# ==============================================================================
@_checked
def test_a_genuinely_out_of_envelope_move_is_still_refused():
    print("\n2. an angle no turn of wrist 3 brings into band is still refused")

    # The local solver -- the one that flies the move -- answers out of band, and
    # no branch of that orientation is in band either.
    ros = Ros(answers={"global": joints(UNREACHABLE_DEG),
                       "local": joints(UNREACHABLE_DEG)})
    guard, ns = guarded(ros)
    res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
    not_blind(ns, ros, "unreachable angle")
    check("refused, in the executing node's own mode",
          refused(res) and ros.commanded == [] and set(ros.solve_modes) == {"local"},
          "%s %s" % (ros.commanded, ros.solve_modes))
    check("and named as a TOOLPATH fault, not a solver one",
          "No turn of this joint reaches" in text, text)
    check("no 'one turn away' claim, because there is no such turn",
          "one turn away" not in text, text)

    # And the far branch out of band while the near one is in it is still
    # refused when the EXECUTOR is the one that picked the far branch.
    ros = Ros(answers={"global": joints(GLOBAL_DEG), "local": joints(GLOBAL_DEG)})
    guard, _ns = guarded(ros)
    res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("a local solver that still lands out of band is refused as well",
          refused(res) and ros.commanded == [], str(ros.commanded))
    check("and named as a SOLVER fault, because an in-band branch exists",
          "one turn away" in text and "IS inside the band" in text, text)

    # Joint moves carry their own values: the six out-of-band angles must
    # still never reach the arm, and nothing about IK mode may change that.
    for deg in (29.3, 21.9, -32.3, -55.1, -113.0):
        ros = Ros()
        guard, _ns = guarded(ros)
        _res, text = run(lambda d=deg: guard.MoveJoints(joints(d)))
        check("MoveJoints(wrist 3 %+7.1f deg) never reaches the arm" % (deg,),
              ros.commanded == [], str(ros.commanded))
        check("  and says the angle was COMMANDED, not solved (%+.1f)" % (deg,),
              "not a solve" in text, text)
    ros = Ros()
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.MoveJoints(joints(200.0)))
    check("a legal +200 deg joint move still goes straight through",
          ros.commanded == ["MoveJoints"], str(ros.commanded))
    check("and a joint move asks no solver at all",
          ros.solve_modes == [] and ros.set_calls == [],
          "%s %s" % (ros.solve_modes, ros.set_calls))


# ==============================================================================
# 3. The mode is put back -- however the block ends
# ==============================================================================
@_checked
def test_the_ik_mode_is_restored():
    print("\n3. move_group is left in the mode it was found in")

    ros = Ros()
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("after a move that FLEW, move_group is global again",
          ros.modes["/move_group"] == "global", ros.modes["/move_group"])
    check("and it really was switched, not left alone",
          ros.set_calls == ["local", "global"], str(ros.set_calls))

    # Already in the executor's mode -- the `scan` profile, where every node is
    # local -- and the guard must not touch it.
    ros = Ros(modes={"/move_group": "local", "/mtc_hub_node": "local",
                     "/dgpython_action_server": "local"})
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("when the mode is already right, the guard sets nothing",
          ros.set_calls == [] and set(ros.solve_modes) == {"local"},
          "%s %s" % (ros.set_calls, ros.solve_modes))

    # The mode change TOOK EFFECT and then the reply failed. Relying on
    # SetIKMode's return value here would leave the cell in the guard's mode,
    # which is why the mode is read before the switch rather than after it.
    ros = Ros(set_raises_after=True)
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("a set that applied and THEN failed is still put back",
          ros.modes["/move_group"] == "global", ros.modes["/move_group"])

    # The refusal path: CableRefused is raised from inside the hold.
    ros = Ros(modes={"/move_group": "global", "/mtc_hub_node": "global",
                     "/dgpython_action_server": "local"})
    guard, _ns = guarded(ros)
    res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("after a REFUSAL, move_group is global again",
          refused(res) and ros.modes["/move_group"] == "global",
          ros.modes["/move_group"])

    # The solve itself raising.
    ros = Ros(solve_raises=True)
    guard, _ns = guarded(ros)
    _res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("after every SolveIK raised, move_group is global again",
          ros.modes["/move_group"] == "global", ros.modes["/move_group"])

    # An arbitrary exception out of the check, which is the case a `finally`
    # exists for: nothing in the guard catches this one.
    ros = Ros()
    guard, ns = guarded(ros)

    def boom(*a, **k):
        raise KeyError("the model blew up")

    ns["check_wrist3"] = boom
    raised = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            guard.CartesianMove([POSE, POSE])
    except KeyError as e:
        raised = e
    check("an unexpected exception is NOT swallowed", raised is not None,
          repr(raised))
    check("and move_group is still put back", ros.modes["/move_group"] == "global",
          ros.modes["/move_group"])

    # The real move raising, after the guard has already let go.
    ros = Ros(move_raises=True)
    guard, _ns = guarded(ros)
    raised = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            guard.CartesianMove([POSE, POSE])
    except RuntimeError as e:
        raised = e
    check("a move that raises leaves the mode restored too",
          raised is not None and ros.modes["/move_group"] == "global",
          ros.modes["/move_group"])

    # Nested: a guarded move inside another hold must not restore to the inner
    # value. ApproachInConfiguration wants global, CartesianMove local; running
    # the second inside the first has to come back to global for the outer.
    ros = Ros()
    guard, _ns = guarded(ros)
    with contextlib.redirect_stdout(io.StringIO()):
        with guard._solving_as("ApproachInConfiguration"):
            inner_before = ros.modes["/move_group"]
            guard.CartesianMove([POSE, POSE])
            inner_after = ros.modes["/move_group"]
    check("a guarded move inside an outer hold restores the OUTER mode",
          inner_before == "global" and inner_after == "global",
          "%s -> %s" % (inner_before, inner_after))
    check("and the outermost hold leaves the cell as it found it",
          ros.modes["/move_group"] == "global", ros.modes["/move_group"])


# ==============================================================================
# 4. When the mode cannot be matched, say so -- and still refuse
# ==============================================================================
@_checked
def test_an_unverified_prediction_is_labelled_and_still_refuses():
    print("\n4. the guard cannot match the executor: fail safe, and say why")

    # This is the OLD behaviour exactly -- solve in whatever move_group holds --
    # plus the warning that was missing. The refusal still happens.
    for name, kwargs in (("the executing node does not answer", {"get_raises": True}),
                         ("the node is not in the profile file", {"unknown_node": True}),
                         ("move_group refuses the mode change", {"set_raises": True}),
                         ("the motion module has no mode API at all", {"no_mode_api": True})):
        ros = Ros(**kwargs)
        guard, _ns = guarded(ros)
        res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
        check("%s: the move is still REFUSED, not waved through" % (name,),
              refused(res) and ros.commanded == [], str(ros.commanded))
        check("  and the refusal says the prediction is unverified",
              "PREDICTION IS UNVERIFIED" in text, text)
        check("  and it does not claim to match the executor",
              "BRANCH THE MOVE WOULD HAVE TAKEN" not in text, text)
        check("  and move_group is not left in the guard's mode",
              ros.modes["/move_group"] == "global", ros.modes["/move_group"])

    # A cell with no envelope measured still moves -- that rule is unchanged.
    ros = Ros()
    guard, _ns = guarded(ros, envelope=None)
    _res, _t = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("no measurement: wrist 3 is unchecked and the move flies (unchanged)",
          ros.commanded == ["CartesianMove"], str(ros.commanded))


# ==============================================================================
# 5. The whole refusal reaches the console
# ==============================================================================
@_checked
def test_the_operator_sees_the_whole_refusal():
    print("\n5. every line of the refusal is printed, not just the first")

    ros = Ros(modes={"/move_group": "global", "/mtc_hub_node": "global",
                     "/dgpython_action_server": "local"})
    node = Node()
    guard, _ns = guarded(ros, node=node)
    res, text = run(lambda: guard.CartesianMove([POSE, POSE]))
    check("the headline is still one line the operator can grep",
          text.splitlines()[0].startswith("  CABLE GUARD REFUSED CartesianMove:"),
          text.splitlines()[0] if text else "(nothing printed)")
    check("the angle and the band are there",
          "wrist 3 to -189.9 deg" in text and "+31.3..+301.3 deg" in text, text)
    check("so is the ABSOLUTE-angle rule, which is why there is no +/-360 fix",
          "ABSOLUTE joint angle" in text, text)
    check("so is the branch note that tells the two faults apart",
          "one turn away" in text, text)
    check("so is which solver answered and in which mode",
          "'global' mode" in text and "/mtc_hub_node" in text, text)
    check("more than one line survived -- the first-line-only print was the bug",
          len(text.strip().splitlines()) >= 5, "%d lines" % len(text.strip().splitlines()))

    # The event payload keeps both: a one-line summary to index on and the whole
    # text, so a replay can tell a solver fault from a toolpath one.
    ev = [e for e in node.events if e.get("event") == "cable_refused"]
    check("the event carries a one-line summary", len(ev) == 1 and "\n" not in ev[0]["why"],
          str(ev[:1]))
    check("and the full detail alongside it",
          len(ev) == 1 and "one turn away" in ev[0].get("detail", ""), str(ev[:1]))


# ==============================================================================
# 6. cable_guard_off() and the pass-through are unaffected
# ==============================================================================
@_checked
def test_the_escape_hatch_and_the_pass_through_still_work():
    print("\n6. what the fix must not have changed")

    ros = Ros(answers={"global": joints(UNREACHABLE_DEG),
                       "local": joints(UNREACHABLE_DEG)})
    guard, ns = guarded(ros)
    with contextlib.redirect_stdout(io.StringIO()):
        with ns["cable_guard_off"]("measuring the band"):
            guard.CartesianMove([POSE, POSE])
    check("cable_guard_off(): capture_cable_envelope() can still leave the band",
          ros.commanded == ["CartesianMove"], str(ros.commanded))
    check("and asks no solver while it is off -- there is nothing to judge",
          ros.solve_modes == [] and ros.set_calls == [],
          "%s %s" % (ros.solve_modes, ros.set_calls))
    with contextlib.redirect_stdout(io.StringIO()):
        res = guard.CartesianMove([POSE, POSE])
    check("and the guard is back on afterwards", refused(res), str(ros.commanded))

    # A named SRDF state has no pose to solve, so there is nothing to match.
    ros = Ros()
    guard, _ns = guarded(ros)
    with contextlib.redirect_stdout(io.StringIO()):
        guard.Move("over_table")
    check("ros.Move('over_table') is passed through, as before",
          ros.commanded == ["Move"] and ros.set_calls == [],
          "%s %s" % (ros.commanded, ros.set_calls))


@_checked
def test_ik_target_solves_in_local_mode():
    """_ik_target() is the other half of the same fault, and it is the half the
    GAUGE hits.

    The guard predicts correctly by asking the executing node what mode it is
    in. _ik_target() -- cell_capture_motion.dpi's "joints for this pose", which
    every calibration leg and every gauge point goes through -- needs the same
    thing: passing `seed=` is not enough, because the IK mode is a NODE
    parameter, a seed does not change it, `transit` leaves move_group global,
    and a global solver is free to leave the seed's branch.

    With the arm parked at over_table (+154 deg), a 162 mm lateral gauge step
    solved globally comes back at wrist_3 = -178.5 deg -- 332 deg from where the
    arm stands, when the same orientation in band is +181.5. A joint route built
    from such a branch can also swing the tip clean out of the work volume. Both
    surface to the operator as "the arm cannot reach this pose", so neither is
    self-diagnosing.
    """
    # _cap_pose() builds a real geometry_msgs Pose, and the note above this
    # file's Pose class says why nothing here may leave a stub in sys.modules.
    # ros_msg_stubs.override_msgs() is the answer to exactly that: a PRIVATE
    # geometry_msgs.msg for the duration, handed back entry for entry, so the
    # eleven other suites in this directory see nothing.
    import ros_msg_stubs
    token = ros_msg_stubs.override_msgs()
    try:
        _run_ik_target_checks()
    finally:
        ros_msg_stubs.restore_msgs(token)
    check("the geometry_msgs stub was handed back",
          ros_msg_stubs.msg_module() is None
          or not getattr(ros_msg_stubs.msg_module(), "_solver_test_private", False),
          "a private stub was left installed")


def _run_ik_target_checks():
    ros = Ros()
    _guard, ns = guarded(ros)
    sol = ns["_ik_target"]([0.9, -0.15, 1.05], (0.0, 0.0, 0.0, 1.0))

    check("the solve happened in LOCAL mode", ros.solve_modes == ["local"],
          str(ros.solve_modes))
    check("so it returned the branch near the seed, not the far one",
          sol is not None
          and abs(math.degrees(sol[5]) - LOCAL_DEG) < 1e-6,
          "%s" % (None if sol is None else round(math.degrees(sol[5]), 1),))
    check("and it was seeded from where the arm stands",
          ros.seeds == [round(STANDING_DEG, 1)], str(ros.seeds))
    check("move_group is put back in the mode the cell had",
          ros.modes["/move_group"] == "global", ros.modes["/move_group"])

    # THE GLOBAL FALLBACK IS STILL THERE, because a branch the arm is not in is a
    # legitimate thing to TRY when local does not converge -- the joint move made
    # to it is FK-checked over its whole path before anything is commanded.
    ros2 = Ros(answers={"local": None, "global": joints(GLOBAL_DEG)})
    _g2, ns2 = guarded(ros2)
    sol2 = ns2["_ik_target"]([0.9, -0.15, 1.05], (0.0, 0.0, 0.0, 1.0))
    check("local first, then global, in that order",
          ros2.solve_modes == ["local", "global"], str(ros2.solve_modes))
    check("and the global branch is what comes back",
          sol2 is not None and abs(math.degrees(sol2[5]) - GLOBAL_DEG) < 1e-6,
          "%s" % (None if sol2 is None else round(math.degrees(sol2[5]), 1),))
    check("with the mode put back after that too",
          ros2.modes["/move_group"] == "global", ros2.modes["/move_group"])

    # local_only=True IS FOR A PREDICTION, and a prediction must not be made in a
    # mode the executing node is not in. It never asks global, even when local has
    # no answer: "I do not know" refuses nothing, while a global branch refuses a
    # move the executor would have flown.
    ros3 = Ros(answers={"local": None, "global": joints(GLOBAL_DEG)})
    _g3, ns3 = guarded(ros3)
    sol3 = ns3["_ik_target"]([0.9, -0.15, 1.05], (0.0, 0.0, 0.0, 1.0),
                             local_only=True)
    check("local_only=True never asks the global solver",
          ros3.solve_modes == ["local"], str(ros3.solve_modes))
    check("and answers None rather than a branch from the wrong mode",
          sol3 is None, str(sol3))


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    print(__doc__.splitlines()[0])
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            getattr(fn, "__wrapped__", fn)()
    print("\n%d checks failed" % len(FAILURES))
    for f in FAILURES:
        print("  - %s" % f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(_main())
