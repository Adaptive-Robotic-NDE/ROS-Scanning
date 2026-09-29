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
# File:  robot_supervisor_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the pause, abort and resume state machine in
#   robot_supervisor.dpi.
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
"""Check the pause/abort/resume state machine in robot_supervisor.dpi.

What has to hold: stopping the robot program PAUSES a measurement instead of
failing every remaining row against a deactivated controller, a pause holds
until somebody resumes it, an abort never blocks, and silence -- a simulation
with no UR driver -- is "nothing to watch" rather than "not ready".

Runs on the host with no ROS: the supervisor's node startup degrades to a
printed warning when rclpy is not importable, which is exactly the path a host
test takes.
"""
import functools
import os
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before defining them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "robot_supervisor_test"}
exec(compile(_load("robot_supervisor.dpi"), "robot_supervisor.dpi", "exec"), ns)

Supervisor = ns["_RobotSupervisor"]
ScanAborted = ns["ScanAborted"]

FAILS = []


def check(cond, what):
    if cond:
        print("  ok    %s" % what)
    else:
        FAILS.append(what)
        print("  FAIL  %s" % what)


def _checked(fn):
    """Make the checks a test recorded into a pytest pass or failure.

    check() RECORDS rather than raises, so one standalone run reports every
    false fact instead of stopping at the first. Under pytest -- which is how
    the suite is run -- that alone would be a test that passes while printing
    FAIL, so the wrapper asserts on whatever the test added to FAILS. The
    standalone driver calls fn.__wrapped__ and reports everything as before.
    """
    @functools.wraps(fn)
    def wrapper(*a, **k):
        mark = len(FAILS)
        fn(*a, **k)
        failed = FAILS[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(map(str, failed)))
    return wrapper

def fresh():
    """A supervisor with no ROS behind it, as on the host."""
    s = Supervisor()
    s.clear()
    return s


class _Msg(object):
    def __init__(self, data):
        self.data = data


@_checked
def test_silence_is_ready():
    s = fresh()
    ok, why = s.ready()
    check(ok, "an unwatched arm (nothing published) reports ready, not blocked")
    check(why == "", "and gives no reason to hold")
    # A checkpoint must return promptly rather than waiting out the timeout.
    began = time.monotonic()
    s.checkpoint("a sim row", timeout=2.0)
    check(time.monotonic() - began < 0.5, "checkpoint on a sim cell returns at once")


@_checked
def test_program_false_blocks():
    s = fresh()
    s._on_program(_Msg(True))
    ok, _why = s.ready()
    check(ok, "program running -> ready")
    s._on_program(_Msg(False))
    ok, why = s.ready()
    check(not ok, "program stopped -> not ready")
    check("External Control" in why, "and the reason names External Control")


@_checked
def test_program_drop_auto_pauses():
    """The pendant stops the program mid-sweep."""
    s = fresh()
    s._on_program(_Msg(True))
    check(s.state()[0] == "running", "a running scan starts running")
    s._on_program(_Msg(False))
    st, reason = s.state()
    check(st == "paused", "the program stopping pauses the scan by itself")
    check("robot program stopped" in reason, "and records why: %r" % reason)


@_checked
def test_pause_holds_until_resume():
    s = fresh()
    s._on_program(_Msg(True))
    s._command("pause", "a test")
    done = []

    def run():
        try:
            s.checkpoint("row 5", timeout=10.0)
            done.append("returned")
        except BaseException as e:            # noqa: BLE001 - the test wants it
            done.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.3)
    check(not done, "a paused scan does not walk on to the next row")
    s._command("resume", "a test")
    t.join(5.0)
    check(done == ["returned"], "and continues once resumed (got %r)" % (done,))


@_checked
def test_resume_needs_the_program_back_too():
    """Resuming with the program still stopped must not release the scan."""
    s = fresh()
    s._on_program(_Msg(True))
    s._on_program(_Msg(False))          # auto-paused
    s._command("resume", "a test")
    done = []

    def run():
        try:
            s.checkpoint("row 5", timeout=10.0)
            done.append("returned")
        except BaseException as e:            # noqa: BLE001
            done.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.3)
    check(not done, "resume alone does not release a scan whose program is down")
    s._on_program(_Msg(True))
    t.join(5.0)
    check(done == ["returned"], "the program coming back does (got %r)" % (done,))


@_checked
def test_abort_raises_and_does_not_block():
    s = fresh()
    s._command("abort", "a test")
    began = time.monotonic()
    try:
        s.checkpoint("row 5", timeout=30.0)
        check(False, "abort raises ScanAborted")
    except ScanAborted as e:
        check(True, "abort raises ScanAborted")
        check("aborted from" in str(e), "and says who: %r" % str(e))
    check(time.monotonic() - began < 0.5, "an abort never waits out the timeout")


@_checked
def test_abort_wins_over_pause():
    s = fresh()
    s._command("pause", "a test")
    s._command("abort", "a test")
    try:
        s.checkpoint("row 5", timeout=5.0)
        check(False, "abort overrides an existing pause")
    except ScanAborted:
        check(True, "abort overrides an existing pause")
    s._command("resume", "a test")
    check(s.state()[0] == "aborting", "and resume cannot un-abort it")


@_checked
def test_pause_times_out_into_an_abort():
    s = fresh()
    s._command("pause", "a test")
    began = time.monotonic()
    try:
        s.checkpoint("row 5", timeout=1.0)
        check(False, "a pause nobody lifts becomes an abort")
    except ScanAborted as e:
        check(True, "a pause nobody lifts becomes an abort")
        check("no resume" in str(e), "and says so: %r" % str(e))
    check(1.0 <= time.monotonic() - began < 3.0, "at about the timeout")


@_checked
def test_controller_inactive_blocks():
    s = fresh()
    s._on_program(_Msg(True))
    s._controllers = {"scaled_joint_trajectory_controller": "inactive",
                      "io_and_status_controller": "active"}
    ok, why = s.ready()
    check(not ok, "a deactivated trajectory controller holds the scan")
    check("scaled_joint_trajectory_controller" in why, "and names it: %r" % why)
    s._controllers["scaled_joint_trajectory_controller"] = "active"
    ok, _why = s.ready()
    check(ok, "and releases it when the controller comes back")


@_checked
def test_unlisted_controllers_are_not_a_fault():
    """A cell that lists no trajectory controller at all is a sim, not a fault."""
    s = fresh()
    s._on_program(_Msg(True))
    s._controllers = {"io_and_status_controller": "active"}
    ok, _why = s.ready()
    check(ok, "controllers we do not manage do not hold a scan")


@_checked
def test_bad_command_is_ignored():
    s = fresh()
    s._on_program(_Msg(True))
    s._on_command(_Msg("resune"))
    check(s.state()[0] == "running", "a typo on the command topic changes nothing")
    s._on_command(_Msg(" ABORT "))
    check(s.state()[0] == "aborting", "case and whitespace are tolerated")


@_checked
def test_supervised_clears_stale_state():
    s = fresh()
    s._command("abort", "a previous run")
    ns["scan_supervisor"] = s
    with ns["supervised"]("a new measurement"):
        check(s.state()[0] == "running", "entering a new measurement clears a stale abort")


@_checked
def test_ctrl_c_is_an_abort():
    s = fresh()
    ns["scan_supervisor"] = s
    try:
        with ns["supervised"]("a measurement"):
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    check(s.state()[0] == "aborting", "Ctrl-C out of a measurement records an abort")


@_checked
def test_disabled_supervisor_never_blocks():
    s = fresh()
    s._command("pause", "a test")
    ns["SCAN_SUPERVISOR_ENABLED"] = False
    try:
        began = time.monotonic()
        s.checkpoint("row 5", timeout=30.0)
        check(time.monotonic() - began < 0.5,
              "SCAN_SUPERVISOR_ENABLED=False turns checkpointing off entirely")
    finally:
        ns["SCAN_SUPERVISOR_ENABLED"] = True


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print("%s:" % name)
            getattr(fn, "__wrapped__", fn)()
    print()
    if FAILS:
        print("%d FAILED:" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        sys.exit(1)
    print("all supervisor checks passed")
