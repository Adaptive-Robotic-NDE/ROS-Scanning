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
# File:  capture_event_audience_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that a late subscriber on the capture event topic is not
#   reported as a fault.
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
"""A late subscriber on /capture/events is not a fault, and must not be reported as one.

start_take() publishes take_start and then asks how many subscribers the event
publisher has. Asked ONCE, immediately, that question prints

    ** THE RECORDER IS NOT RECEIVING /capture/events. The bag will hold the
    sensors and the robot but not what this tool did, so an offline replay
    cannot be lined up with the take.

on every take, and it is false: the bag's own report comes back with
/capture/events received, written, nothing lost.

The recorder subscribes to the scenario's own topics AFTER the readiness gate
has returned -- about 0.8 s behind the core topics -- and the publisher hears
about a new subscriber through discovery some time after the subscriber logs
it. None of that loses an event, because the publisher is RELIABLE +
TRANSIENT_LOCAL depth 100 (cell_capture_node.dpi), so a subscriber that matches
late still receives everything already published.

So the question has to be asked over a window: a late match costs nothing, and
only a match that never happens is worth telling the operator about.

Runs on the host: the .dpi is exec'd with its `global` lines commented out, the
same way cell_capture_estimate_test.py does it. No ROS, no recorder, no cell.

    python3 capture_event_audience_test.py
"""
import functools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def _load():
    src = open(os.path.join(HERE, "cell_capture_take.dpi")).read()
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                    for l in src.splitlines())
    ns = {"np": np}
    exec(compile(src, "cell_capture_take.dpi", "exec"), ns)
    return ns


NS = _load()


class FakePub(object):
    """A publisher whose audience appears after `after` polls."""

    def __init__(self, after):
        self.after = after
        self.polls = 0

    def get_subscription_count(self):
        self.polls += 1
        return 1 if self.polls > self.after else 0


class FakeClock(object):
    """Time only moves when something sleeps, so the test never waits."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


wait = NS["_wait_for_audience"]


@_checked
def test_an_audience_already_there_costs_one_poll():
    print("a subscriber that is already there costs one poll and no wait")
    clk = FakeClock()
    pub = FakePub(after=0)
    n = wait(pub, seconds=5.0, sleep=clk.sleep, clock=clk)
    check("returns the count", n == 1, "got %r" % (n,))
    check("polled once", pub.polls == 1, "polled %d times" % pub.polls)
    check("did not wait", clk.t == 0.0, "waited %.1f s" % clk.t)


@_checked
def test_a_subscriber_that_matches_late_is_found():
    print("\na subscriber that matches 0.8 s late -- the measured case -- is found")
    clk = FakeClock()
    pub = FakePub(after=4)                     # 4 polls x 0.2 s = 0.8 s
    n = wait(pub, seconds=5.0, sleep=clk.sleep, clock=clk)
    check("found it", n == 1, "got %r" % (n,))
    check("waited about 0.8 s, not the whole window",
          0.7 <= clk.t <= 0.9, "waited %.1f s" % clk.t)


@_checked
def test_a_subscriber_that_never_matches_is_reported():
    print("\na subscriber that NEVER matches is still reported, after the window")
    clk = FakeClock()
    pub = FakePub(after=10**9)
    n = wait(pub, seconds=5.0, sleep=clk.sleep, clock=clk)
    check("reports nobody", n == 0, "got %r" % (n,))
    check("waited the whole window", clk.t >= 5.0, "waited %.1f s" % clk.t)
    check("did not spin", pub.polls <= 40, "polled %d times" % pub.polls)


@_checked
def test_the_window_is_a_named_constant():
    print("\nthe window is a named constant, not a literal buried in start_take()")
    s = NS.get("CAPTURE_EVENT_AUDIENCE_S")
    check("CAPTURE_EVENT_AUDIENCE_S exists and is sane",
          isinstance(s, float) and 1.0 <= s <= 30.0, "got %r" % (s,))


@_checked
def test_start_take_asks_through_the_helper_and_says_how_long_it_waited():
    print("\nstart_take() asks through the helper, so it cannot go back to one read")
    src = open(os.path.join(HERE, "cell_capture_take.dpi")).read()
    body = src[src.index("def start_take"):]
    check("start_take calls _wait_for_audience",
          "_wait_for_audience(capture_node.events_pub)" in body)
    check("start_take does not read the count directly",
          "events_pub.get_subscription_count()" not in body,
          "a bare one-shot read is back in start_take()")

    print("\nthe warning text no longer claims a replay is impossible without qualification")
    check("the warning says how long it waited",
          "after %.0f s" in src or "after %.1f s" in src)


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    print("\n%s" % ("ALL OK" if not FAILURES else "%d FAILED: %s"
                    % (len(FAILURES), ", ".join(FAILURES))))
    return 1 if FAILURES else 0


_SECTIONS = ["test_an_audience_already_there_costs_one_poll", "test_a_subscriber_that_matches_late_is_found", "test_a_subscriber_that_never_matches_is_reported", "test_the_window_is_a_named_constant", "test_start_take_asks_through_the_helper_and_says_how_long_it_waited"]


if __name__ == "__main__":
    sys.exit(_main())
