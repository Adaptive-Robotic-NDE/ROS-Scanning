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
# File:  warning_light_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the warning-light hold and the work-volume refusal in
#   work_cell.dpi.
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
"""Check the warning-light hold and the work-volume refusal in work_cell.dpi.

The light (UR digital output 0 on ex-ur10e-1) has to be ON whenever an
automated stage moves the arm and OFF afterwards -- including when the stage
raises, and including a Ctrl-C. Holds nest, so a whole run can hold it while
each stage holds it for itself, and the light must not blink off between them.
On hardware a light that cannot be switched on stops the stage before it
moves; in simulation, where there is no UR I/O controller, it must not.

The refusal is the first thing a new cell hits on hardware, so its text is
checked for the parts an operator needs: how to measure, where to write it,
how to reload.

Runs on the host: set_digital_out and tari_bringup are stubs.
"""
import functools
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "\n".join(l for l in open(os.path.join(HERE, "work_cell.dpi")).read().splitlines()
                if not l.startswith("global "))

FAILS = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILS)
        fn(*args, **kwargs)
        failed = FAILS[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


def load(mode="sim", pin=0, fail_on=False, fail_off=False, required=None,
         fail_on_times=None, attempts=None):
    """The module with set_digital_out stubbed, and the calls it made.

    `fail_on_times` fails the first N switch-ON attempts and then succeeds, which
    is what a transient set_io looks like. `fail_on` fails all of them.
    """
    calls = []
    state = {"on_failures": 0}

    def set_digital_out(p, state_):
        calls.append((p, bool(state_)))
        if state_ and fail_on_times is not None:
            state["on_failures"] += 1
            if state["on_failures"] <= fail_on_times:
                raise RuntimeError("/io_and_status_controller/set_io did not "
                                   "answer within 15 s")
            return True
        if (state_ and fail_on) or (not state_ and fail_off):
            raise RuntimeError("no /io_and_status_controller/set_io service")
        return True

    ns = {"__name__": "work_cell_test", "CELL_NAME": "ex-ur10e-1",
          "CELL_MODE": mode, "WARNING_LIGHT_PIN": pin,
          "set_digital_out": set_digital_out}
    if required is not None:
        ns["WARNING_LIGHT_REQUIRED"] = required
    exec(compile(SRC, "work_cell.dpi", "exec"), ns)
    # No real waiting in a test; the retry COUNT is what is under test, not the
    # gap between attempts.
    ns["WARNING_LIGHT_RETRY_S"] = 0.0
    if attempts is not None:
        ns["WARNING_LIGHT_ATTEMPTS"] = attempts
    return ns, calls


@_checked
def test_on_for_the_block_off_after_it():
    # --- on, off ------------------------------------------------------------------
    ns, calls = load()
    with ns["warning_light"]("a stage"):
        inside = list(calls)
    check(inside == [(0, True)] and calls == [(0, True), (0, False)],
          "on for the block, off after it: %s" % (calls,))


@_checked
def test_off_on_error_and_on_ctrl_c():
    # --- off on error, and the error still propagates -----------------------------
    ns, calls = load()
    try:
        with ns["warning_light"]("a stage that fails"):
            raise ValueError("boom")
    except ValueError:
        raised = True
    else:
        raised = False
    check(raised and calls == [(0, True), (0, False)],
          "off after an exception, which still propagates: %s" % (calls,))

    ns, calls = load()
    try:
        with ns["warning_light"]("interrupted"):
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    check(calls == [(0, True), (0, False)], "off after Ctrl-C: %s" % (calls,))


@_checked
def test_nested_holds():
    # --- nesting: one on, one off, and it stays on between stages -----------------
    ns, calls = load()
    with ns["warning_light"]("the whole run"):
        with ns["warning_light"]("stage 1"):
            pass
        between = list(calls)
        with ns["warning_light"]("stage 2"):
            pass
    check(between == [(0, True)] and calls == [(0, True), (0, False)],
          "nested holds: on once, off once, on between stages: %s" % (calls,))


@_checked
def test_no_light_in_this_cell_is_a_no_op():
    # --- no light in this cell ----------------------------------------------------
    ns, calls = load(pin=None)
    with ns["warning_light"]("no light"):
        pass
    check(calls == [], "WARNING_LIGHT_PIN = None is a no-op")


@_checked
def test_hardware_refuses_to_move_without_the_light():
    # --- hardware: cannot switch on -> the block does not run ---------------------
    ns, calls = load(mode="hardware", fail_on=True)
    ran = []
    try:
        with ns["warning_light"]("hardware stage"):
            ran.append(True)
        refused = False
    except RuntimeError as e:
        refused = "Refusing to move the arm" in str(e)
    check(refused and not ran and calls[-1] == (0, False),
          "hardware: a light that will not switch on stops the stage before it moves "
          "(and it still tries to switch off): %s" % (calls,))

    ns, calls = load(mode="hardware", fail_on=True, required=False)
    ran = []
    with ns["warning_light"]("hardware, override"):
        ran.append(True)
    check(ran == [True], "WARNING_LIGHT_REQUIRED = False lets a hardware stage run dark")


@_checked
def test_simulation_runs_without_a_light():
    # --- simulation: no I/O controller -> carry on, and stop asking ---------------
    ns, calls = load(mode="sim", fail_on=True)
    ran = []
    with ns["warning_light"]("sim stage"):
        ran.append(True)
    with ns["warning_light"]("sim stage 2"):
        ran.append(True)
    check(ran == [True, True] and calls == [(0, True)],
          "simulation: runs without a light, and asks set_io only once: %s" % (calls,))


@_checked
def test_a_failed_switch_off_warns_rather_than_raising():
    # --- off fails: no exception from the finally --------------------------------
    ns, calls = load(mode="hardware", fail_off=True)
    ok = True
    try:
        with ns["warning_light"]("off fails"):
            pass
    except Exception:
        ok = False
    check(ok and calls == [(0, True), (0, False)],
          "a failed switch-off warns rather than raising over the stage's own result")


@_checked
def test_a_transient_set_io_does_not_cost_the_run():
    """Two of set_io's failures mean "nobody could ask", not "the lamp is off".

    The service missing its discovery window, or being found and not answering,
    both raise from set_digital_out and both used to refuse the whole run before
    the arm moved. Neither is evidence the light is dark.
    """
    ns, calls = load(mode="hardware", fail_on_times=2)
    ran = []
    with ns["warning_light"]("transient set_io"):
        ran.append(True)
    on_calls = [c for c in calls if c[1]]
    check(ran == [True],
          "a set_io that fails twice and then works does not refuse the run")
    check(len(on_calls) == 3,
          "the switch-on was attempted 3 times, not %d: %s" % (len(on_calls), calls))
    check(calls[-1] == (0, False), "the light was still switched off afterwards")


@_checked
def test_a_light_that_never_answers_still_refuses():
    """The retry must not become a way to move with the light dark.

    Retrying is only allowed to change how long it takes to decide; the decision
    itself is unchanged.
    """
    ns, calls = load(mode="hardware", fail_on=True, attempts=3)
    ran = []
    try:
        with ns["warning_light"]("never answers"):
            ran.append(True)
        refused = False
    except RuntimeError as e:
        refused = "Refusing to move the arm" in str(e)
    on_calls = [c for c in calls if c[1]]
    check(refused and not ran,
          "a light that never switches on still stops the stage: %s" % (calls,))
    check(len(on_calls) == 3,
          "it gave up after %d attempts rather than 3: %s" % (len(on_calls), calls))


@_checked
def test_the_switch_off_path_is_not_retried():
    """Switching off cannot save a run, and the operator is told to use the pendant.

    Retrying it would add a delay to the end of every stage for nothing.
    """
    ns, calls = load(mode="hardware", fail_off=True, attempts=3)
    with ns["warning_light"]("off fails"):
        pass
    off_calls = [c for c in calls if not c[1]]
    check(len(off_calls) == 1,
          "the switch-off was retried %d times; it should be attempted once: %s"
          % (len(off_calls), calls))


@_checked
def test_simulation_is_not_retried():
    """Simulation has no I/O controller at all, and already stops asking.

    Three attempts per hold would add the full service wait to every stage of
    every simulated run.
    """
    ns, calls = load(mode="sim", fail_on=True, attempts=3)
    with ns["warning_light"]("sim stage"):
        pass
    check([c for c in calls if c[1]] == [(0, True)],
          "simulation asked set_io more than once to switch on: %s" % (calls,))


@_checked
def test_the_version_report_names_versions_and_paths():
    """A log that does not say what was loaded cannot answer "why did it do that".

    Two of this stack's expensive failures are version skew rather than logic: an
    install behind the workspace, and a compiled extension older than the library
    it links against. Both are invisible in a traceback and obvious in one startup
    line, so the line has to carry the version AND the path -- on a machine with a
    system install and a workspace checkout, the path is the part that decides it.
    """
    ns, _calls = load()
    text = ns["describe_software_versions"]()
    facts = ns["software_versions"]()

    check("python" in text and sys.version.split()[0] in text,
          "the report names the Python actually running")
    for module_name, label in ns["SOFTWARE_OF_INTEREST"]:
        check(label in text,
              "%r is missing from the report, so its version would be unknown"
              % (label,))
        check(module_name in facts,
              "software_versions() does not carry %r" % (module_name,))

    # An absent component has to SAY it is absent. Left out silently, a
    # reconstruction that cannot run looks like a reconstruction that was not
    # asked to.
    missing = [m for m, f in facts.items() if f["version"] is None]
    for m in missing:
        label = dict(ns["SOFTWARE_OF_INTEREST"])[m]
        check("NOT IMPORTABLE" in text and label in text,
              "%r is not importable and the report does not say so" % (label,))

    present = [f for f in facts.values() if f["version"] is not None]
    check(present, "nothing at all was importable, so this test proved nothing")
    check(all(f["path"] for f in present),
          "a component was reported with no path: %s"
          % ([f for f in present if not f["path"]],))


@_checked
def test_the_work_volume_refusal_tells_the_operator_what_to_do():
    # --- the refusal ---------------------------------------------------------------
    # A real FILE under an install/ tree, so the refusal takes its INSTALLED COPY
    # branch wherever this runs. An installed manifest in a --symlink-install
    # workspace is a symlink and correctly takes the other branch, so this has to
    # be a fixture rather than the real installed path.
    import tempfile
    _inst = os.path.join(tempfile.mkdtemp(), "install", "tari_cell_configs", "share",
                         "tari_cell_configs", "config", "cells")
    os.makedirs(_inst)
    manifest = os.path.realpath(os.path.join(_inst, "ex-ur10e-1.yaml"))
    open(manifest, "w").close()


    class _Cell(object):
        def work_volume(self):
            return None


    cells = types.ModuleType("tari_bringup.cells")
    cells.load_cell = lambda name, mode: _Cell()
    cells.find_manifest = lambda name: manifest
    sys.modules["tari_bringup"] = types.ModuleType("tari_bringup")
    sys.modules["tari_bringup.cells"] = cells

    ns, calls = load(mode="hardware")
    ns["SPECIMEN_SWEEP_HEIGHT"] = 1.35
    try:
        ns["require_work_volume"]("the Azure reconstruction")
        msg = ""
    except RuntimeError as e:
        msg = str(e)
    for piece in ("the Azure reconstruction", "work_volume.hardware is null",
                  "capture_touch_off()", "local override manifest", "cell_capture.dpi",
                  "tf2_echo world probe_end",
                  "WITHOUT compressing the spring", "min: [xmin, ymin, zmin]",
                  "max: [xmax, ymax, zmax]", "hardware:", "at most 1.25",
                  "work_volume_reload()", "tari_cell_info --cell ex-ur10e-1 --mode hardware",
                  "INSTALLED COPY"):
        check(piece in msg, "refusal says %r" % (piece,))
    check(manifest in msg, "refusal names the manifest file tari_bringup actually reads")

    print("")


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        sys.exit(1)
    print("all passed")


_SECTIONS = ["test_on_for_the_block_off_after_it", "test_off_on_error_and_on_ctrl_c", "test_nested_holds", "test_no_light_in_this_cell_is_a_no_op", "test_hardware_refuses_to_move_without_the_light", "test_simulation_runs_without_a_light", "test_a_failed_switch_off_warns_rather_than_raising", "test_the_work_volume_refusal_tells_the_operator_what_to_do"]


if __name__ == "__main__":
    sys.exit(_main())
