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
# File:  progress_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the progress reporting that a long unattended scan depends on.
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
"""progress.dpi: the five properties a long unattended scan needs from it.

Without progress output an operator waiting on a silent stage will kill a
clearance gate that would have finished. The helper that fixes that has to be
worth trusting on five counts, and each of them is a way it could quietly go
wrong:

  * RATE LIMITING actually limits. A progress helper that prints per item turns
    a 284514-waypoint gate into 284514 lines and the log becomes useless in the
    other direction.
  * THE FAST PATH IS CHEAP. step() is called once per waypoint; if it reads the
    clock every time, the reporting costs more than some of the work it reports
    on. A million steps must be well under a second, and must read the clock a
    handful of times rather than a million.
  * NO FAKE ETA. An estimate from the first few items is worse than none: an
    operator told "4 s left" who then waits half an hour stops believing every
    later estimate too.
  * THE FINAL LINE PRINTS ON EXCEPTION. This is the one that matters most: a
    stage that is killed must still say how long it had been running, or nobody
    can tell how close it was.
  * UNKNOWN-TOTAL MODE WORKS. Most of the slow stages are blocking calls into a
    ROS service or a subprocess, where there is nothing to count. A heartbeat is
    the only thing that distinguishes working from hung.

Plus: no ANSI, no carriage returns. The output goes into a `script` typescript
that is read later as a file.

    python3 progress_test.py     # prints every check, exit 1 on failure
    python3 -m pytest -q progress_test.py

Host-side, no ROS, no dgpy: the .dpi is exec'd with its `global` lines commented
out, which is how every other test in this directory loads one.

The sections below take no arguments on purpose: a section taking (ns, lines)
reads to pytest as a request for two fixtures of those names, and errors with a
message about fixtures rather than about progress.dpi. They share one namespace
through _env() instead, in the order main() runs them in, which is the order
pytest collects them in too.
"""
import functools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

fails = []


def check(name, ok, detail=""):
    print("  %-62s %s" % (name, "ok" if ok else "FAIL  " + detail))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


class _CountingClock(object):
    """time, with monotonic() counted. Proves the fast path is a compare."""

    def __init__(self, real):
        self.real = real
        self.n = 0

    def monotonic(self):
        self.n += 1
        return self.real.monotonic()

    def __getattr__(self, name):
        return getattr(self.real, name)


def _load():
    """progress.dpi in a namespace of its own, with a line sink installed."""
    src = open(os.path.join(HERE, "progress.dpi")).read()
    # A .dpi is compiled as a function body, so its `global` lines are not valid
    # at module scope here. Everything else at top level is imports, constants
    # and defs.
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                    for l in src.splitlines())
    ns = {}
    exec(compile(src, "progress.dpi", "exec"), ns)
    lines = []
    # progress_say() prefers _capture_say() when the session defines one -- the
    # pipeline's own print-plus-log path. Using it here both captures the output
    # and exercises that branch, instead of letting the lines go to stdout.
    ns["_capture_say"] = lines.append
    return ns, lines


_ENV = None


def _env():
    """The one (namespace, captured lines) pair every section below shares."""
    global _ENV
    if _ENV is None:
        _ENV = _load()
    return _ENV


@_checked
def test_rate_limiting():
    ns, lines = _env()
    print("=== rate limiting ===")
    progress = ns["progress"]

    # A loop that runs for ~0.9 s with a 0.2 s reporting interval: somewhere
    # between three and six lines, definitely not one per item.
    del lines[:]
    with progress("slow loop", total=90, every=0.2, eta_after=1e9) as p:
        for _ in range(90):
            time.sleep(0.01)
            p.step()
    updates = [l for l in lines if "/90" in l]
    check("a ~0.9 s loop at every=0.2 gives 3..6 update lines",
          3 <= len(updates) <= 6, "got %d: %s" % (len(updates), updates))
    check("not one line per item", len(updates) < 90, "%d lines" % len(updates))
    check("the closing line reports the count and the rate",
          lines and "90 in " in lines[-1] and "/s)" in lines[-1],
          lines[-1] if lines else "no lines")

    # The same loop with a reporting interval longer than the loop: no updates
    # at all, just the opening and closing lines.
    del lines[:]
    with progress("quick loop", total=1000, every=30.0) as p:
        for _ in range(1000):
            p.step()
    check("a loop shorter than the interval emits no update lines",
          len(lines) == 2, "got %d: %s" % (len(lines), lines))

    print("=== the lines themselves ===")
    bad = [l for l in lines if "\r" in l or "\x1b" in l or "\n" in l]
    check("no carriage returns, newlines or ANSI escapes", not bad, str(bad))
    check("every line carries a wall clock",
          all(l.startswith("  [") and l[3:11].count(":") == 2 for l in lines),
          str(lines))


@_checked
def test_fast_path():
    ns, _lines = _env()
    print("=== the fast path ===")
    progress = ns["progress"]
    clock = _CountingClock(time)
    ns["time"] = clock
    try:
        began = time.perf_counter()
        with progress("a million", total=1000000, every=3600.0,
                      announce=False) as p:
            for _ in range(1000000):
                p.step()
        cost = time.perf_counter() - began
    finally:
        ns["time"] = time
    check("a million step()s cost well under a second (%.3f s)" % (cost,),
          cost < 0.6, "%.3f s" % (cost,))
    # Two for the block's own start/finish, the rest are threshold re-aims. A
    # per-step clock read would be 1000000 of them.
    check("...and read the clock a handful of times (%d)" % (clock.n,),
          clock.n < 100, "%d clock reads" % (clock.n,))
    check("the count is exact", p.n == 1000000, str(p.n))


@_checked
def test_eta_honesty():
    ns, lines = _env()
    print("=== the ETA ===")
    progress = ns["progress"]

    # eta_after is the whole of this: before it, elapsed and fraction only.
    del lines[:]
    with progress("estimating", total=200, every=0.05, eta_after=0.4) as p:
        for _ in range(200):
            time.sleep(0.004)
            p.step()
    updates = [l for l in lines if "/200" in l]
    early = [l for l in updates if float(l.split("(")[1].split("%")[0]) < 40.0]
    check("there are early and late update lines to compare",
          len(updates) >= 4 and early, "%d updates, %d early"
          % (len(updates), len(early)))
    check("no ETA in the first eta_after seconds",
          all("left" not in l for l in early), str(early))
    check("an ETA appears once the block has earned one",
          any("left" in l for l in updates), str(updates))
    check("every update line reports elapsed and fraction",
          all("elapsed" in l and "%)" in l for l in updates), str(updates))

    # Under 2% done is not enough however long it has taken -- a loop whose
    # first item is slow must not extrapolate from it.
    del lines[:]
    with progress("one slow item", total=1000, every=0.05, eta_after=0.05) as p:
        time.sleep(0.25)
        p.step()
        time.sleep(0.25)
        p.step()
    updates = [l for l in lines if "/1000" in l]
    check("no ETA from 2 items of 1000, however long they took",
          updates and all("left" not in l for l in updates), str(updates))


@_checked
def test_final_line_on_exception():
    ns, lines = _env()
    print("=== the closing line, when it goes wrong ===")
    progress = ns["progress"]

    del lines[:]
    try:
        with progress("a stage that breaks", total=10, every=30.0) as p:
            p.step()
            raise ValueError("boom")
    except ValueError:
        pass
    check("a failed block still prints a closing line", len(lines) == 2,
          str(lines))
    check("...saying how long it ran and what went wrong",
          lines and "FAILED after" in lines[-1] and "ValueError" in lines[-1],
          lines[-1] if lines else "")
    check("...and how far it got", "1 of 10 done" in lines[-1], lines[-1])

    # The operator kills the stage.
    del lines[:]
    try:
        with progress("a stage that is killed", total=100, every=30.0) as p:
            for _ in range(7):
                p.step()
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    check("a killed block says INTERRUPTED and how long it ran",
          lines and "INTERRUPTED after" in lines[-1], str(lines))
    check("...and how far it got", "7 of 100 done" in lines[-1], lines[-1])

    # A nested block that raises speaks even though a nested block that merely
    # finished quickly would not.
    del lines[:]
    try:
        with progress("outer", total=2, every=30.0):
            with progress("inner", total=2, every=30.0):
                raise RuntimeError("inner boom")
    except RuntimeError:
        pass
    inner = [l for l in lines if "inner" in l]
    check("a nested block that raises still prints", len(inner) == 1, str(lines))
    check("...indented under its parent",
          inner and "]   inner: FAILED after" in inner[0], str(inner))


@_checked
def test_unknown_total():
    ns, lines = _env()
    print("=== work with no total ===")
    progress = ns["progress"]
    progress_step = ns["progress_step"]

    # Heartbeat: nothing to count, a blocking call to wait on.
    del lines[:]
    with progress("a blocking service call", every=0.1, heartbeat=True):
        time.sleep(0.45)
    beats = [l for l in lines if "working," in l]
    check("a heartbeat block reports while it is blocked", len(beats) >= 2,
          "%d beats: %s" % (len(beats), lines))
    check("...with elapsed time on every beat",
          all("elapsed" in l for l in beats), str(beats))
    check("...and never an ETA", all("left" not in l for l in beats), str(beats))
    check("...and a closing line with the total time",
          lines and "done in" in lines[-1], str(lines))

    # Counting with no total: a count and elapsed, no fraction and no ETA.
    del lines[:]
    with progress("counting", every=0.1) as p:
        for _ in range(60):
            time.sleep(0.005)
            p.step()
    updates = [l for l in lines if "so far" in l]
    check("an unknown total still reports a count", len(updates) >= 2,
          str(lines))
    check("...with no percentage and no ETA",
          all("%)" not in l and "left" not in l for l in updates), str(updates))

    # progress_step: silent when fast, a heartbeat when slow, always one line.
    del lines[:]
    with progress_step("something quick") as s:
        s.result("3 -> 2 things")
    check("progress_step prints exactly one line", len(lines) == 1, str(lines))
    check("...carrying the result and the elapsed time",
          lines and "3 -> 2 things" in lines[0] and " s)" in lines[0],
          str(lines))

    del lines[:]
    with progress_step("something slow", every=0.1):
        time.sleep(0.35)
    check("a slow progress_step speaks before it finishes",
          len([l for l in lines if "working," in l]) >= 2, str(lines))


@_checked
def test_nesting():
    ns, lines = _env()
    print("=== nesting ===")
    progress = ns["progress"]
    progress_step = ns["progress_step"]
    progress_note = ns["progress_note"]

    del lines[:]
    with progress("outer", total=3, every=30.0) as outer:
        for i in range(3):
            with progress_step("inner %d" % i):
                pass
            outer.step()
    check("a fast nested block with nothing to say prints nothing",
          len(lines) == 2, str(lines))

    del lines[:]
    with progress("outer", total=3, every=30.0) as outer:
        for i in range(3):
            with progress_step("inner %d" % i) as s:
                s.result("%d points" % (100*i,))
            outer.step()
    inner = [l for l in lines if l.lstrip().startswith("[") and "inner" in l]
    check("a nested block WITH a result prints one line each",
          len(inner) == 3, str(lines))
    check("...indented one level under the parent",
          all("]   inner" in l for l in inner), str(inner))
    check("the depth is back to zero afterwards", ns["progress_depth"]() == 0,
          str(ns["progress_depth"]()))

    del lines[:]
    with progress("outer", total=1, every=30.0):
        progress_note("2968678 raw points")
    check("progress_note lands at the current depth",
          len(lines) == 3 and "]   2968678 raw points" in lines[1],
          str(lines))


@_checked
def test_iterator():
    ns, lines = _env()
    print("=== progress_over ===")
    progress_over = ns["progress_over"]

    del lines[:]
    got = [x for x in progress_over(list(range(50)), "iterating", every=30.0,
                                    unit="poses")]
    check("it yields every item", got == list(range(50)), str(got[:5]))
    check("...and reports the unit",
          lines and "50 poses in" in lines[-1], str(lines))

    # Breaking out of the loop abandons the generator; that is not a failure.
    del lines[:]
    for x in progress_over(list(range(50)), "abandoned", every=30.0):
        if x == 3:
            break
    import gc
    gc.collect()
    check("breaking out of the loop is reported as stopping early, not failing",
          any("stopped early" in l for l in lines)
          and not any("FAILED" in l for l in lines), str(lines))


@_checked
def test_sinks_and_switch():
    ns, lines = _env()
    print("=== delivery ===")
    progress = ns["progress"]
    copies = []
    ns["progress_sink"](copies.append)
    try:
        del lines[:]
        with progress("watched", total=1, every=30.0):
            pass
        check("a sink gets a copy of every line", copies == lines, str(copies))
    finally:
        ns["progress_sink"](copies.append, remove=True)

    del lines[:]
    del copies[:]
    with progress("watched again", total=1, every=30.0):
        pass
    check("a removed sink gets nothing", not copies, str(copies))

    ns["PROGRESS_ENABLED"] = False
    try:
        del lines[:]
        with progress("silenced", total=1, every=30.0) as p:
            p.step()
        check("PROGRESS_ENABLED=False silences everything", not lines,
              str(lines))
        check("...but the block still times itself", p.seconds >= 0.0,
              str(p.seconds))
    finally:
        ns["PROGRESS_ENABLED"] = True


@_checked
def test_logfile():
    ns, lines = _env()
    print("=== the log file ===")
    import tempfile
    progress = ns["progress"]
    d = tempfile.mkdtemp(prefix="progress_test_")
    path = os.path.join(d, "deeper", "progress.log")
    ns["progress_logfile"](path)
    try:
        del lines[:]
        with progress("logged", total=1, every=30.0):
            pass
        written = open(path).read().splitlines()
        check("lines are appended to the configured log file",
              written == lines, "%s vs %s" % (written, lines))
    finally:
        ns["progress_logfile"](None)
    check("progress_logfile(None) turns it off", ns["PROGRESS_LOG"] is None,
          str(ns["PROGRESS_LOG"]))


def main():
    """Every section, in order, reporting all failures rather than the first."""
    global _ENV
    _ENV = None                     # a fresh progress.dpi for this run
    del fails[:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            # __wrapped__ is the undecorated section: it records failures
            # instead of raising, so one run reports all of them.
            getattr(fn, "__wrapped__", fn)()
    print("")
    if fails:
        print("%d FAILED: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("all progress.dpi checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
