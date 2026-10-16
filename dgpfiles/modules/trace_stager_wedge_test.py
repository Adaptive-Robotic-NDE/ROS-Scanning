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
# File:  trace_stager_wedge_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   Host tests for the two faults that let a scan freeze silently
#   between passes: an error record that pins the transaction it was
#   raised inside, and a drain() whose timeout did not bound the wait
#   that actually blocks.
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
"""The two ways a scan froze between pass 59 and pass 60 of 64.

Both faults are in the staging path, both are silent by construction, and both
cost a 28-minute scan and all 59 of its completed passes on 2026-10-02.

ONE: `self.last_error = e` retained the exception object, which retains
e.__traceback__, which retains the frame it was raised in, which retains that
frame's locals -- including the spatialnde2 active_transaction the staging code
was holding. An active_transaction ends itself in its destructor, so an
exception mid-transaction normally self-heals when the frame is collected.
Pinned, it is never ended; and under the timed transaction manager the
transaction map commits strictly from its head, so one unended transaction
stops every globalrev wait in the process, permanently, printing nothing.

TWO: `drain(timeout=120.0)` bounded a polling loop and then called
`self._q.join()` unconditionally, with no timeout. The wait that actually
blocks had no deadline, so the 120 s did nothing.

These tests use a stand-in for active_transaction rather than spatialnde2
itself: the property under test is Python reference retention, not anything
about the database, and it must hold on the host with no recording database
open.
"""
import ast
import gc
import os
import sys
import threading
import time
import weakref

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load():
    """`triggered_acquisition.dpi`'s namespace, with what it prints captured.

    A .dpi is compiled as a function body, so it is not importable: top-level
    `global` statements are a syntax error to ast.parse. They are commented out
    here, which is the same approach ec_instrument_null_test.py takes.
    """
    printed = []
    ns = {"print": lambda *a: printed.append(" ".join(str(x) for x in a))}
    exec("import threading, time, queue, traceback\nimport numpy as np\n", ns)
    src = open(os.path.join(HERE, "triggered_acquisition.dpi")).read()
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                     for l in src.splitlines())
    src = src.replace("from geometry_msgs.msg import PoseArray, Pose",
                      "pass  # ROS message types, unused here")
    exec(compile(src, "triggered_acquisition.dpi", "exec"), ns)
    return ns, printed


class StandInTransaction(object):
    """What spatialnde2's active_transaction does that matters here.

    It ends itself when it is destroyed. If it is never destroyed it is never
    ended, and an unended transaction at the head of the timed manager's map
    blocks the whole database.
    """

    def __init__(self, ended):
        self._ended = ended

    def __del__(self):
        self._ended.append(True)


# ==============================================================================
# 1. An error must not pin the transaction it was raised inside
# ==============================================================================

def test_a_staging_error_does_not_pin_the_transaction_it_was_raised_inside():
    """The fault that wedged the 2026-10-02 scan, asserted directly."""
    ns, _printed = load()
    ended = []
    alive = []

    def stage(_trace):
        transact = StandInTransaction(ended)
        alive.append(weakref.ref(transact))
        raise RuntimeError("the globalrev never completed")

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        stager.drain(timeout=5.0)
        assert stager.errors == 1, "the error must still be counted"
        gc.collect()
        assert ended == [True], (
            "the transaction was never ended -- an error record is pinning the "
            "frame it was raised in, which is what stops the whole database")
        assert alive[0]() is None
    finally:
        stager.stop()


def test_the_error_is_kept_as_text_so_it_holds_no_references():
    ns, _printed = load()

    def stage(_trace):
        raise RuntimeError("a distinctive failure")

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        stager.drain(timeout=5.0)
        assert isinstance(stager.last_error, str), (
            "last_error must be a formatted traceback, not an exception object")
        assert "a distinctive failure" in stager.last_error
        assert "Traceback" in stager.last_error, (
            "the traceback is the useful half; keep the text, drop the object")
        assert not hasattr(stager.last_error, "__traceback__")
    finally:
        stager.stop()


def test_a_staging_error_is_printed_when_it_happens():
    """The end-of-scan report is exactly what a frozen scan never reaches."""
    ns, printed = load()

    def stage(_trace):
        raise RuntimeError("a distinctive failure")

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        stager.drain(timeout=5.0)
        text = "\n".join(printed)
        assert "staging error 1" in text
        assert "a distinctive failure" in text
    finally:
        stager.stop()


def test_only_the_first_few_errors_are_printed():
    """One recurring fault per trace must not bury the log."""
    ns, printed = load()

    def stage(_trace):
        raise RuntimeError("every single time")

    stager = ns["TraceStager"](stage, print_errors=2)
    try:
        for i in range(7):
            stager.submit("trace-%d" % i)
        stager.drain(timeout=5.0)
        assert stager.errors == 7
        text = "\n".join(printed)
        # Count the headers, not the message: the message appears twice in each
        # printed traceback, once in the source line and once in the summary.
        assert text.count("staging error 1 of this scan") == 1
        assert text.count("staging error 2 of this scan") == 1
        assert "staging error 3 of this scan" not in text
    finally:
        stager.stop()


# ==============================================================================
# 2. drain() must bound what it actually waits on
# ==============================================================================

def test_drain_returns_promptly_when_nothing_is_stuck():
    ns, printed = load()
    staged = []
    stager = ns["TraceStager"](staged.append)
    try:
        for i in range(5):
            stager.submit("trace-%d" % i)
        t0 = time.monotonic()
        assert stager.drain(timeout=10.0) == 5
        assert time.monotonic() - t0 < 5.0
        assert len(staged) == 5
        assert "STAGING STALLED" not in "\n".join(printed)
    finally:
        stager.stop()


def test_drain_says_so_when_it_stalls_rather_than_waiting_in_silence():
    """27 minutes of total silence is what made the freeze expensive."""
    ns, printed = load()
    release = threading.Event()

    def stage(_trace):
        release.wait(30.0)

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        done = threading.Event()

        def drainer():
            stager.drain(timeout=0.2)
            done.set()

        t = threading.Thread(target=drainer)
        t.daemon = True
        t.start()
        time.sleep(1.0)
        text = "\n".join(printed)
        assert "STAGING STALLED" in text, (
            "a wait this code cannot bound must at least announce itself")
        assert "1 trace(s) queued" in text
        assert "has not returned rather than one that failed" in text, (
            "the note must distinguish a stuck stage from a failed one")
        assert not done.is_set(), (
            "it must keep waiting -- abandoning a half-staged trace is worse")
        release.set()
        assert done.wait(10.0), "drain must return once the stage completes"
    finally:
        release.set()
        stager.stop()


def test_the_stall_note_says_where_the_worker_is():
    """The counters say a stage has not returned; only the stack says WHICH.

    On 2026-10-15 the note read "37 trace(s) queued, 2845 staged, 0 error(s)"
    and that was the whole evidence base. stage_ec_reading() has three places
    it can park -- start_transaction(), the globalrev wait after
    end_transaction(), and the projection after that -- and they are three
    different faults. sys._current_frames() tells them apart from another
    thread with no debugger and no signal.
    """
    ns, printed = load()
    release = threading.Event()

    def _inner_wait():
        release.wait(30.0)

    def stage(_trace):
        _inner_wait()

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        t = threading.Thread(target=lambda: stager.drain(timeout=0.2))
        t.daemon = True
        t.start()
        time.sleep(1.0)
        text = "\n".join(printed)
        assert "the staging worker is here" in text, text
        assert "_inner_wait" in text, (
            "the stack must reach the call that is actually blocked:\n%s" % text)
        release.set()
        t.join(10.0)
    finally:
        release.set()
        stager.stop()


def test_the_stall_note_repeats_and_says_whether_anything_moved():
    """One note cannot distinguish a crawl from a death.

    Consecutive notes can, and that is the first question anyone asks. The
    2026-10-15 stall printed once and then nothing for seven minutes.
    """
    ns, printed = load()
    release = threading.Event()

    def stage(_trace):
        release.wait(30.0)

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        t = threading.Thread(target=lambda: stager.drain(timeout=0.2))
        t.daemon = True
        t.start()
        time.sleep(1.5)
        text = "\n".join(printed)
        assert text.count("STAGING STALLED") >= 2, (
            "%d note(s) in 1.5 s at a 0.2 s timeout" % text.count("STAGING STALLED"))
        assert "has not staged a single trace since" in text, text
        release.set()
        t.join(10.0)
    finally:
        release.set()
        stager.stop()


def test_the_stall_note_says_ctrl_c_will_not_work():
    """Because it will not, and the operator spent three of them finding out.

    dataguzzler-python runs the GUI loop on the real main thread; SIGINT is
    delivered there, inside Qt, with the GIL released, so no handler runs. The
    note has to name the way out instead of implying the usual one.
    """
    ns, printed = load()
    release = threading.Event()
    stager = ns["TraceStager"](lambda _t: release.wait(30.0))
    try:
        stager.submit("trace-0")
        t = threading.Thread(target=lambda: stager.drain(timeout=0.2))
        t.daemon = True
        t.start()
        time.sleep(1.0)
        text = "\n".join(printed)
        assert "CTRL-C WILL NOT WORK" in text, text
        assert "capture_stall.sh" in text, text
        assert "kill %1" in text, text
        release.set()
        t.join(10.0)
    finally:
        release.set()
        stager.stop()


def test_the_stall_note_reports_a_recorded_error_when_there_is_one():
    """The discriminator: a stuck stage behind an EARLIER swallowed error."""
    ns, printed = load()
    release = threading.Event()
    calls = []

    def stage(_trace):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the first trace failed mid-transaction")
        release.wait(30.0)

    stager = ns["TraceStager"](stage)
    try:
        stager.submit("trace-0")
        stager.drain(timeout=5.0)
        del printed[:]
        stager.submit("trace-1")
        t = threading.Thread(target=lambda: stager.drain(timeout=0.2))
        t.daemon = True
        t.start()
        time.sleep(1.0)
        text = "\n".join(printed)
        assert "STAGING STALLED" in text
        assert "the first trace failed mid-transaction" in text, (
            "the earlier error is the thing that explains the stall")
        assert "known way to wedge" in text
        release.set()
        t.join(10.0)
    finally:
        release.set()
        stager.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
