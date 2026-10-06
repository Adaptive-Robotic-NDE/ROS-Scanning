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
# File:  capture_run_on_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-05
# Description:
#   Tests the operator's "r = run the rest of this take without asking"
#   answer in confirm(), and the scoping that keeps it from leaking.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-05:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""`r` stops the prompting for one take, and cannot leak past it.

A tilt-and-rotation matrix is ten passes of about five prompts each, and by the
second pass the operator is confirming the identical descent for the ninth time.
That is a reflex rather than supervision, so `r` lets them say "I have watched
this, run the rest" -- a decision made in the middle of a run by the person
watching it.

It is NOT CAPTURE_AUTO_CONFIRM, which is set before anything runs by someone who
has not seen the arm move, is refused outside simulation, and stays refused.
The distinction is the whole safety argument, so these tests pin it: that the
hardware refusal still fires, that arming is scoped to the take OBJECT so a
later scenario asks again, that `r` outside a take does nothing, and that a
scenario's own keys are never shadowed.

confirm() is lifted out of the .dpi by text rather than exec'ing the module,
which imports rclpy. Runs on the host with no ROS.
"""
import functools
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


class _Take(object):
    """Only what confirm() touches."""

    def __init__(self):
        self.confirmations = []


class _Node(object):
    def now_ns(self):
        return 0


def _load(mode="hardware", auto_confirm=False):
    """confirm() and the namespace it ran in."""
    src = open(os.path.join(HERE, "cell_capture_node.dpi")).read()
    m = re.search(r"^def confirm\(prompt, extra=None\):.*?(?=^def |\Z)",
                  src, re.S | re.M)
    assert m, "confirm() is not where this test expects it in cell_capture_node.dpi"
    said = []
    ns = {"_CAP": {}, "CELL_MODE": mode, "CAPTURE_AUTO_CONFIRM": auto_confirm,
          "capture_node": _Node(), "_capture_say": said.append,
          "CaptureRefused": type("CaptureRefused", (RuntimeError,), {})}
    exec(compile(m.group(0), "cell_capture_node.dpi", "exec"), ns)
    ns["_said"] = said
    return ns


def _answers(ns, *keys):
    """Feed confirm() a scripted operator, and record what it was shown."""
    seen = []
    it = iter(keys)

    def _input(text):
        seen.append(text)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("confirm() asked more times than expected:\n"
                                 + text)
    ns["input"] = _input
    ns["_seen"] = seen
    return seen


fails = []


def check(name, ok, detail=""):
    print("  %-66s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


@_checked
def test_r_is_offered_only_inside_a_take():
    print("=== r is offered only inside a take ===")
    ns = _load()
    take = _Take()

    ns["_CAP"]["take"] = take
    seen = _answers(ns, "")
    ns["confirm"]("move the arm")
    check("the prompt offers it during a take",
          "r=run the rest of this take without asking" in seen[0], seen[0])

    ns["_CAP"]["take"] = None
    seen = _answers(ns, "")
    ns["confirm"]("move the arm")
    check("and does not outside one", "r=run the rest" not in seen[0], seen[0])

    # ...and answering r there is not silently accepted as something else.
    ns["_CAP"]["take"] = None
    seen = _answers(ns, "r", "")
    check("r outside a take is rejected and asked again",
          ns["confirm"]("move the arm") == "go" and len(seen) == 2, seen)
    print()


@_checked
def test_r_arms_the_take_and_stops_the_prompting():
    print("=== r arms the take and stops the prompting ===")
    ns = _load()
    take = _Take()
    ns["_CAP"]["take"] = take

    seen = _answers(ns, "r")          # exactly one answer available
    check("r returns go for the move it was typed on",
          ns["confirm"]("descend onto the part") == "go")
    # Any further input() would raise AssertionError from _answers.
    for i in range(6):
        check("prompt %d after arming does not ask" % (i + 1),
              ns["confirm"]("pass %d" % i) == "go")
    check("only the first move was ever shown to the operator", len(seen) == 1,
          "%d prompts" % len(seen))
    print()


@_checked
def test_arming_does_not_stop_the_record_or_the_telling():
    print("=== arming does not stop the record or the telling ===")
    ns = _load()
    take = _Take()
    ns["_CAP"]["take"] = take
    _answers(ns, "r")
    ns["confirm"]("descend onto the part")
    ns["confirm"]("pass a -> b")
    ns["confirm"]("lift off")

    check("every move is still recorded", len(take.confirmations) == 3,
          str(len(take.confirmations)))
    check("the armed ones are marked auto",
          [c["auto"] for c in take.confirmations] == [False, True, True],
          str([c["auto"] for c in take.confirmations]))
    check("and attributed to the operator, not to sim",
          [c["auto_by"] for c in take.confirmations] == [None, "operator", "operator"],
          str([c["auto_by"] for c in take.confirmations]))
    check("every armed move is still printed",
          sum("running on" in s for s in ns["_said"]) == 2,
          str(ns["_said"]))
    check("the arming itself says how to get out",
          any("Ctrl-C" in s for s in ns["_said"]), str(ns["_said"]))
    print()


@_checked
def test_arming_cannot_leak_into_the_next_take():
    """The scoping is by take IDENTITY, so nothing has to remember to clear it.

    This is the test that matters: a capture_contact() an hour later over
    different taught points must ask again from its first move.
    """
    print("=== arming cannot leak into the next take ===")
    ns = _load()
    first = _Take()
    ns["_CAP"]["take"] = first
    _answers(ns, "r")
    ns["confirm"]("first take, first move")
    ns["confirm"]("first take, armed through")

    second = _Take()
    ns["_CAP"]["take"] = second
    seen = _answers(ns, "")
    check("the next take asks again", ns["confirm"]("second take") == "go"
          and len(seen) == 1, seen)

    ns["_CAP"]["take"] = None
    seen = _answers(ns, "")
    check("and so does a prompt outside any take",
          ns["confirm"]("no take") == "go" and len(seen) == 1, seen)
    print()


@_checked
def test_a_scenarios_own_keys_are_never_shadowed():
    """The cable test owns letters of its own. If one of them were ever `r`,
    taking it for run-on would answer a different question than was asked."""
    print("=== a scenario's own keys are never shadowed ===")
    ns = _load()
    take = _Take()
    ns["_CAP"]["take"] = take

    seen = _answers(ns, "r")
    got = ns["confirm"]("step the wrist", {"r": "reached the limit"})
    check("the scenario's r wins", got == "reached the limit", got)
    check("and run-on was not offered at all",
          "r=run the rest" not in seen[0], seen[0])
    check("so the take was not armed", ns["_CAP"].get("auto_rest") is None)

    # An unrelated extra key still leaves run-on available.
    ns2 = _load()
    ns2["_CAP"]["take"] = _Take()
    seen = _answers(ns2, "")
    ns2["confirm"]("step the wrist", {"t": "tight"})
    check("an unrelated extra key leaves run-on offered",
          "r=run the rest" in seen[0] and "t=tight" in seen[0], seen[0])
    print()


@_checked
def test_auto_confirm_is_still_refused_on_hardware():
    """The contract r does NOT weaken. CAPTURE_AUTO_CONFIRM is set before
    anything runs, by someone who has not watched the arm; it stays refused."""
    print("=== CAPTURE_AUTO_CONFIRM is still refused on hardware ===")
    ns = _load(mode="hardware", auto_confirm=True)
    ns["_CAP"]["take"] = _Take()
    try:
        ns["confirm"]("move the arm")
        check("it raises on hardware", False, "it did not raise")
    except Exception as e:                                    # noqa: BLE001
        check("it raises on hardware", type(e).__name__ == "CaptureRefused",
              type(e).__name__)
        check("and says which mode", "hardware" in str(e), str(e))

    ns = _load(mode="sim", auto_confirm=True)
    take = _Take()
    ns["_CAP"]["take"] = take
    _answers(ns)                       # no input available: it must not ask
    check("in sim it answers without asking", ns["confirm"]("move") == "go")
    check("and is attributed to sim rather than to the operator",
          take.confirmations[0]["auto_by"] == "sim",
          str(take.confirmations[0]))
    print()


_SECTIONS = ["test_r_is_offered_only_inside_a_take",
             "test_r_arms_the_take_and_stops_the_prompting",
             "test_arming_does_not_stop_the_record_or_the_telling",
             "test_arming_cannot_leak_into_the_next_take",
             "test_a_scenarios_own_keys_are_never_shadowed",
             "test_auto_confirm_is_still_refused_on_hardware"]


def _main():
    for name in _SECTIONS:
        fn = globals()[name]
        getattr(fn, "__wrapped__", fn)()
    if fails:
        print("FAILURES:", fails)
        return 1
    print("all run-on checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
