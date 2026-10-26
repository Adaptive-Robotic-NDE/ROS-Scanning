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
# File:  ec_instrument_null_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   The eddy-current instrument is balanced once, on the part, before
#   the first trace -- and says so when it cannot be.
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
"""WHEN the eddy-current instrument is nulled, and what a failure costs.

An eddy-current bridge has to be balanced with the probe ON unflawed material.
Balanced in air, the operating point is in the wrong place, the gain is applied
about that point, and every trace of the scan rides at an offset -- and no
software correction afterwards moves it. A scan that ran un-nulled is not a scan
to repeat; it is data that cannot be recovered.

So the null has exactly one correct moment, and the order around it is tight:

    prepare(n)          accumulators sized. The probe is IN THE AIR
    approach(pass 1)    lift / transit / descend -- the probe touches down
    >>> the null <<<
    stage(trace 0)      the first reading, which must already be balanced
    setup_after_first   the math that consumes trace 0 is wired on it

prepare() is too early and setup_after_first() is too late. And the probe
descends onto the part at the start of EVERY pass, so "on contact" has to mean
the FIRST contact -- re-balancing per pass would step the baseline between
passes and stripe the image, which is the same fault EC_SERPENTINE avoids.

WHAT IS CHECKED, all of it on the host with no ROS, no robot and no instrument:

  * the real run_triggered_path_scan() loop, driven over a three-pass path with a
    fake trigger source, with every approach / null / stage / setup event logged
    in order. One null, before the first trace, after the first touchdown.
  * a first pass that cannot be entered is not a touchdown: the null lands on the
    first pass actually reached.
  * nulling turned off calls nothing at all.
  * an instrument that raises: refused loudly when the null is required (and
    nothing is acquired), reported and survived when it is not. Never swallowed.
  * and that the config and the two .dpi files are wired to each other the way
    the above assumes.

The two functions under test are lifted out of their .dpi files by AST rather
than copied here -- ec_acquisition.dpi cannot simply be executed, because it
imports the compiled conditioning extension at its top and none of that is
involved in nulling an instrument.

THE OTHER NULL. /ec/cond/null is a complex offset subtracted from recorded traces
in software (null_here()), live and re-runnable. This file is about the hardware
balance underneath it. ec_conditioning_test.py covers the software one.

    python3 ec_instrument_null_test.py
    python3 -m pytest -q ec_instrument_null_test.py
"""
import ast
import functools
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")

fails = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """check() records rather than raises, so a standalone run reports every
    failure in one pass; under pytest the wrapper asserts on what was added."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed), "; ".join(failed))
    wrapper.__wrapped__ = fn
    return wrapper


# ==============================================================================
# Loading: the real scan loop, plus the two functions by name
# ==============================================================================

def lift(path, names):
    """Named top-level functions out of one .dpi or .dgp, as AST nodes.

    A .dpi is compiled as a function body and a .dgp opens a recording database
    when it runs, so neither is importable. Parsing and compiling just the
    functions wanted gives a namespace this file controls -- which is also how
    the module globals they write (ec_instrument_null) are observed.
    """
    tree = ast.parse(open(path).read(), filename=os.path.basename(path))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            if node.name in found:
                raise AssertionError("%s() is defined twice in %s"
                                     % (node.name, os.path.basename(path)))
            found[node.name] = node
    missing = [n for n in names if n not in found]
    if missing:
        raise AssertionError("%s has no %s()"
                            % (os.path.basename(path), ", ".join(missing)))
    return [found[n] for n in names]


def load(say=None):
    """One namespace with the scan loop and the nulling functions in it.

    `say` collects everything the nulling code prints, so the messages can be
    asserted on instead of scrolling past.
    """
    printed = [] if say is None else say
    ns = {"np": np, "print": lambda *a: printed.append(" ".join(str(x) for x in a))}
    exec("import threading, time\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n"
         "import numpy as np\n", ns)
    for name in ("toolpath_segments.dpi", "triggered_acquisition.dpi"):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                        for l in src.splitlines())
        src = src.replace("from geometry_msgs.msg import PoseArray, Pose",
                          "pass  # ROS message types, unused here")
        exec(compile(src, name, "exec"), ns)
    nodes = lift(os.path.join(HERE, "surface_path_scan.dpi"), ["first_touchdown_hook"])
    nodes += lift(os.path.join(HERE, "ec_acquisition.dpi"), ["ec_first_contact_null"])
    mod = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "lifted", "exec"), ns)
    ns["ec_instrument_null"] = None
    return ns, printed


# ==============================================================================
# A scan with no robot and no instrument
# ==============================================================================

class FakeSource(object):
    """A position trigger that fires on demand instead of on travel.

    trigger_mode is what run_surface_path_scan() screens on; the rest is the
    contract run_triggered_path_scan() needs: arm with a budget, deliver that
    many crossings, disarm.
    """

    trigger_mode = "timestamped-ec"

    def __init__(self):
        self.travelled_m = 0.0
        self.fault = None
        self.armed_over_budget = 0
        self._sink = None
        self._budget = 0

    def arm(self, sink, emit_on_arm=True, max_triggers=None):
        self._sink = sink
        self._budget = 0 if max_triggers is None else int(max_triggers)

    def disarm(self):
        self._sink = None

    def synthesize(self, *args):
        return np.zeros(1, dtype=np.complex128)

    def metadata(self):
        return None

    def crossings(self, pitch_m):
        """Fire the whole armed budget, as a sweep of that length would."""
        for k in range(self._budget):
            self.travelled_m += pitch_m
            self._sink(self.synthesize(), None, (k + 1)*pitch_m, k)


class Instrument(object):
    """A stand-in Uniwest: counts its nulls, or raises instead."""

    def __init__(self, raises=None):
        self.nulls = 0
        self.raises = raises

    def null(self):
        self.nulls += 1
        if self.raises is not None:
            raise self.raises


def three_pass_path(ns, passes=3, per_pass=6, pitch_m=0.010):
    """A path of straight passes, each `per_pass` waypoints at `pitch_m`."""
    pos, starts = [], []
    for p in range(passes):
        starts.append(len(pos))
        for k in range(per_pass):
            pos.append([k*pitch_m, 0.1*p, 1.0])
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
    return ns["ScanPath"](np.array(pos), quats, starts, source="test path")


def drive(nulling="on", raises=None, required=True, refuse_first_pass=False,
          passes=3, pitch_m=0.010):
    """Run the real scan loop over a multi-pass path. -> (events, report, said, inst)

    `events` is every approach / null / stage / setup, in the order they
    happened -- which is the whole question here. An exception from the loop is
    returned in place of the report.
    """
    said = []
    ns, printed = load(said)
    path = three_pass_path(ns, passes=passes, pitch_m=pitch_m)
    segs = path.acquiring_segments(min_length_m=0.005)
    source = FakeSource()
    inst = Instrument(raises=raises)
    events, lock = [], threading.Lock()

    def log(*item):
        with lock:
            events.append(item)

    def approach(idx, end_idx):
        log("approach", idx)
        if refuse_first_pass and idx == segs[0][0]:
            return False
        return True

    def sweep(start, end):
        source.crossings(pitch_m)
        return True

    def stage(ascan, meta, pose):
        log("stage",)

    def pose_at(seg, s_m, xy):
        return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def setup_after_first():
        log("setup",)

    on_contact = None
    if nulling == "on":
        def null_fn():
            log("null",)
            inst.null()
        on_contact = ns["ec_first_contact_null"](null_fn, required=required)
        approach = ns["first_touchdown_hook"](approach, on_contact)

    try:
        report = ns["run_triggered_path_scan"](
            path, source, pitch_m, approach, sweep, stage, pose_at,
            setup_after_first=setup_after_first, segments=segs,
            on_segment_failure="skip")
    except Exception as e:                                        # noqa: BLE001
        report = e
    return events, report, printed, inst, ns


# ==============================================================================
print("")
print("ONE NULL, AT THE FIRST TOUCHDOWN, BEFORE THE FIRST TRACE")


@_checked
def test_null_fires_once_before_any_trace():
    events, report, said, inst, ns = drive()
    kinds = [e[0] for e in events]
    check("three passes were entered", kinds.count("approach") == 3, kinds)
    check("the instrument was nulled exactly once",
          kinds.count("null") == 1 and inst.nulls == 1,
          "%d events, %d calls" % (kinds.count("null"), inst.nulls))
    check("and it happened AFTER the probe reached the first pass",
          kinds.index("approach") < kinds.index("null"), kinds[:4])
    check("and BEFORE the first trace was staged",
          kinds.index("null") < kinds.index("stage"), kinds[:4])
    check("and before the accumulator math was wired on that trace",
          kinds.index("null") < kinds.index("setup"), kinds[:4])
    check("with nothing between the touchdown and the null",
          kinds[:2] == ["approach", "null"], kinds[:3])
    check("traces were staged for every pass",
          isinstance(report, dict) and report["staged"] == 3*6, report)
    check("the outcome is recorded as nulled",
          ns["ec_instrument_null"] == {"nulled": True, "why": "",
                                       "required": True},
          ns["ec_instrument_null"])
    check("and the printed line names it the INSTRUMENT null",
          any("INSTRUMENT" in l for l in said), said)
    check("and says the conditioning null is a separate thing",
          any("/ec/cond/null" in l for l in said), said)


@_checked
def test_null_is_not_repeated_per_pass():
    """The probe descends once per pass; the balance must not follow it."""
    for passes in (2, 5, 9):
        events, report, said, inst, ns = drive(passes=passes)
        kinds = [e[0] for e in events]
        check("%d passes, %d approaches, one null"
              % (passes, kinds.count("approach")),
              kinds.count("approach") == passes and inst.nulls == 1,
              "%d nulls" % (inst.nulls,))


@_checked
def test_a_refused_pass_is_not_a_touchdown():
    """The null belongs to the first pass actually entered, not the first tried."""
    events, report, said, inst, ns = drive(refuse_first_pass=True)
    kinds = [e[0] for e in events]
    check("the first pass was refused and skipped",
          isinstance(report, dict) and len(report["skipped"]) == 1, report)
    check("the instrument was still nulled exactly once", inst.nulls == 1,
          inst.nulls)
    check("and not until the second approach, which is the first touchdown",
          kinds[:3] == ["approach", "approach", "null"], kinds[:4])
    check("still before any trace",
          kinds.index("null") < kinds.index("stage"), kinds[:5])


@_checked
def test_nulling_off_calls_nothing():
    events, report, said, inst, ns = drive(nulling="off")
    kinds = [e[0] for e in events]
    check("no null event", "null" not in kinds, kinds)
    check("the instrument was never touched", inst.nulls == 0)
    check("nothing was printed about nulling", said == [], said)
    check("and the scan ran in full",
          isinstance(report, dict) and report["staged"] == 3*6, report)
    check("with no outcome recorded", ns["ec_instrument_null"] is None)


# ==============================================================================
print("")
print("AN INSTRUMENT THAT WILL NOT NULL: reported, never swallowed")


@_checked
def test_a_failing_null_stops_a_required_scan():
    boom = IOError("serial port closed")
    events, report, said, inst, ns = drive(raises=boom, required=True)
    kinds = [e[0] for e in events]
    check("the scan raised rather than acquiring", isinstance(report, Exception),
          report)
    check("and it is a RuntimeError naming what went wrong",
          isinstance(report, RuntimeError) and "serial port closed" in str(report),
          report)
    check("the message says it is STOPPING rather than scanning un-nulled",
          "STOPPING" in str(report), report)
    check("it says why an un-nulled scan cannot be repaired afterwards",
          "no software null repairs" in str(report), report)
    check("and it names the override",
          "set_ec_nulling(required=False)" in str(report), report)
    check("it stopped at the first pass, with NO trace staged",
          "stage" not in kinds, kinds)
    check("the outcome records the failure and its reason",
          ns["ec_instrument_null"]["nulled"] is False
          and "serial port closed" in ns["ec_instrument_null"]["why"],
          ns["ec_instrument_null"])


@_checked
def test_a_failing_null_is_survivable_when_it_is_not_required():
    boom = IOError("no instrument on port 33")
    events, report, said, inst, ns = drive(raises=boom, required=False)
    kinds = [e[0] for e in events]
    check("the scan completed", isinstance(report, dict)
          and report["staged"] == 3*6, report)
    check("the failure was printed as a WARNING, not swallowed",
          any("WARNING" in l and "no instrument on port 33" in l for l in said),
          said)
    check("and the warning says the scan is going ahead UN-NULLED",
          any("UN-NULLED" in l for l in said), said)
    check("and that it cannot be corrected afterwards",
          any("cannot be corrected afterwards" in l for l in said), said)
    check("the null was attempted once and not retried per pass",
          inst.nulls == 1 and kinds.count("null") == 1, inst.nulls)
    check("the outcome records that it did not null",
          ns["ec_instrument_null"]["nulled"] is False
          and ns["ec_instrument_null"]["required"] is False,
          ns["ec_instrument_null"])


# ==============================================================================
print("")
print("THE WIRING: the config, the EC layer and the acquisition agree")


def source_of(path, name):
    """One function's source, normalised by ast.unparse.

    Unparsing rather than slicing the file means the checks below are about the
    code's structure and not about how it happens to be laid out -- but it also
    reflows nothing inside a docstring, so prose is matched through flat().
    """
    return ast.unparse(lift(path, [name])[0])


def flat(text):
    """`text` with every run of whitespace collapsed, for matching wrapped prose."""
    return " ".join(text.split())


@_checked
def test_the_acquisition_calls_the_hook_before_the_first_pass():
    src = source_of(os.path.join(HERE, "surface_path_scan.dpi"),
                    "run_surface_path_scan")
    check("run_surface_path_scan() takes on_first_contact",
          "on_first_contact=None" in src)
    check("and wraps the approach with first_touchdown_hook()",
          "first_touchdown_hook(approach, on_first_contact)" in src)
    prepare = src.index("prepare(path.trace_count")
    wrap = src.index("first_touchdown_hook(approach")
    check("the wrap happens after the accumulators are sized, where the callables "
          "are built", prepare < wrap, "%d, %d" % (prepare, wrap))


@_checked
def test_the_ec_layer_builds_the_hook_and_reports_it():
    src = source_of(os.path.join(HERE, "ec_acquisition.dpi"), "run_ec_path_scan")
    check("run_ec_path_scan() takes null_instrument and null_required",
          "null_instrument=None" in src and "null_required=True" in src)
    check("it builds the hook with ec_first_contact_null()",
          "ec_first_contact_null(null_instrument, required=null_required)" in src)
    check("None means no hook at all",
          "None if null_instrument is None" in src)
    check("the hook is handed to the acquisition as on_first_contact",
          "on_first_contact=on_contact" in src)
    check("and the outcome comes back in the scan report",
          "report['instrument_null'] = ec_instrument_null" in src)


@_checked
def test_run_ec_scan_decides_before_it_moves():
    src = source_of(DGP, "run_ec_scan")
    check("run_ec_scan() states what it will do about the null",
          "describe_ec_nulling()" in src)
    check("it refuses when a null is required and there is no instrument",
          "EC_NULL_ON_CONTACT and EC_NULL_REQUIRED" in src
          and "ec_instrument is None" in src and "raise RuntimeError" in src)
    refuse = src.index("EC_NULL_ON_CONTACT and EC_NULL_REQUIRED")
    light = src.index("warning_light('eddy-current scan')")
    check("and it refuses BEFORE the arm is asked to move",
          refuse < light, "%d, %d" % (refuse, light))
    check("it hands the instrument null to the scan",
          "null_instrument=null_ec_instrument if EC_NULL_ON_CONTACT else None" in src)
    check("with the config's own answer about whether it is required",
          "null_required=EC_NULL_REQUIRED" in src)


@_checked
def test_the_two_nulls_are_not_conflated():
    """The hardware balance and the software offset are named apart everywhere."""
    hook = flat(source_of(os.path.join(HERE, "ec_acquisition.dpi"),
                          "ec_first_contact_null"))
    check("the instrument hook's docstring says it is not the conditioning null",
          "NOT THE CONDITIONING NULL" in hook, hook[:400])
    check("and says how the two relate",
          "no software offset repairs an instrument balanced in air" in hook, hook)
    here = flat(source_of(DGP, "null_here"))
    check("null_here() says it sets the CONDITIONING null",
          "CONDITIONING null" in here, here[:300])
    check("and that it is not the instrument null",
          "NOT the instrument null" in here, here[:600])
    check("its printed line says which null it set",
          "conditioning null set to" in here, here)


def _main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            getattr(fn, "__wrapped__", fn)()
    print("")
    if fails:
        print("%d FAILED: %s" % (len(fails), "; ".join(fails)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
