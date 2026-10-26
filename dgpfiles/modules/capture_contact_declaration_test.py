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
# File:  capture_contact_declaration_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-05
# Description:
#   Tests that a contact capture scenario declares the probe's contact with
#   the specimen for its duration, and undoes only what it did.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-05:  Initial Commit
#
#########################################################################
"""A contact scenario declares contact, and withdraws only what it declared.

capture_contact() presses the probe into a specimen that is a collision object
and then makes moves FROM that state. With nothing declared, the start state is
illegal and the planner correctly refuses to interpolate out of it. On
2026-10-05 that produced `onto contact_a along the tool axis` at 86%, `off
contact_a` at 14%, and `over contact_a: lift off` reporting a 100% plan that
then moved 1.0 mm of the 10 it was commanded -- all three read as planner
faults, and the operator had to know to call allow_specimen_contact() by hand
first.

The interesting half is the UNDO. Taking away a protection setting somebody else
chose is worse than leaving one on, so this withdraws only when it was the thing
that turned it on. These tests pin that, and that a scene with no specimen says
so rather than raising.

The context manager is lifted out of the .dpi by text: the module is a function
body that imports numpy and expects the config's namespace. Runs on the host
with no ROS.
"""
import functools
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

LINKS = ("ec_probe", "probe_holder")


class _Scene(object):
    """A stand-in for the planning scene and its allowed-collision matrix."""

    def __init__(self, objects=("specimen",), allowed=(), fail_on=()):
        self.objects = list(objects)
        self.allowed = tuple(allowed)
        self.fail_on = set(fail_on)        # "allow", "clear"
        self.calls = []

    def planning_scene_objects(self):
        return list(self.objects)

    def allow_specimen_contact(self, links, allow=True):
        self.calls.append((tuple(links), bool(allow)))
        if ("allow" if allow else "clear") in self.fail_on:
            raise RuntimeError("move_group did not answer within 30 s")
        self.allowed = tuple(links) if allow else ()
        return tuple(links)


def _load(scene, links=LINKS):
    src = open(os.path.join(HERE, "cell_capture_pipeline.dpi")).read()
    m = re.search(r"^@contextlib\.contextmanager\ndef _cap_specimen_contact\(.*?"
                  r"(?=^@|^def )", src, re.S | re.M)
    assert m, ("_cap_specimen_contact() is not where this test expects it in "
               "cell_capture_pipeline.dpi")
    said = []
    import contextlib
    ns = {"contextlib": contextlib, "_capture_say": said.append,
          "TOOLING_CONTACT_LINKS": links, "SPECIMEN_SCENE_ID": "specimen",
          "specimen_contact_allowed": scene.allowed,
          "allow_specimen_contact": scene.allow_specimen_contact,
          "planning_scene_objects": scene.planning_scene_objects}
    exec(compile(m.group(0), "cell_capture_pipeline.dpi", "exec"), ns)
    ns["_said"] = said
    return ns


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
def test_contact_is_declared_for_the_take_and_withdrawn_after():
    print("=== contact is declared for the take and withdrawn after ===")
    scene = _Scene()
    ns = _load(scene)
    with ns["_cap_specimen_contact"]() as declared:
        check("it reports that it declared", declared is True)
        check("the matrix says so during the take", scene.allowed == LINKS,
              str(scene.allowed))
    check("and it is withdrawn afterwards", scene.allowed == (),
          str(scene.allowed))
    check("exactly one declaration and one withdrawal",
          scene.calls == [(LINKS, True), (LINKS, False)], str(scene.calls))
    print()


@_checked
def test_a_declaration_somebody_else_made_is_left_alone():
    """The undo is of what THIS did, not of whatever it found."""
    print("=== a declaration somebody else made is left alone ===")
    scene = _Scene(allowed=LINKS)
    ns = _load(scene)
    with ns["_cap_specimen_contact"]() as declared:
        check("it declares nothing", declared is False)
        check("and calls nothing", scene.calls == [], str(scene.calls))
    check("the operator's declaration survives the take",
          scene.allowed == LINKS, str(scene.allowed))
    check("and it says it left it as it was",
          any("already declared" in s for s in ns["_said"]), str(ns["_said"]))
    print()


@_checked
def test_no_specimen_in_the_scene_says_so_rather_than_raising():
    """With nothing in the scene there is no contact to declare -- AND nothing
    is collision checked against the part, which is the larger fact."""
    print("=== no specimen in the scene says so rather than raising ===")
    scene = _Scene(objects=())
    ns = _load(scene)
    with ns["_cap_specimen_contact"]() as declared:
        check("it declares nothing", declared is False)
    check("nothing was asked of the scene", scene.calls == [], str(scene.calls))
    check("and it says nothing is checked against the part either",
          any("nothing is checked against the" in s for s in ns["_said"]),
          str(ns["_said"]))
    print()


@_checked
def test_declare_contact_false_keeps_every_protection():
    print("=== declare_contact=False keeps every protection ===")
    scene = _Scene()
    ns = _load(scene)
    with ns["_cap_specimen_contact"](declare=False) as declared:
        check("it declares nothing", declared is False)
    check("and touches nothing", scene.calls == [] and scene.allowed == (),
          str(scene.calls))
    print()


@_checked
def test_a_scene_that_will_not_answer_warns_and_runs_anyway():
    """A take that cannot reach move_group's matrix has bigger problems than
    the matrix; refusing to start is worse than warning."""
    print("=== a scene that will not answer warns and runs anyway ===")
    scene = _Scene(fail_on=("allow",))
    ns = _load(scene)
    ran = []
    with ns["_cap_specimen_contact"]() as declared:
        ran.append(True)
        check("it reports that it did not declare", declared is False)
    check("the body still ran", ran == [True])
    check("and it warned, naming the recovery",
          any("WARNING" in s and "allow_specimen_contact" in s
              for s in ns["_said"]), str(ns["_said"]))
    check("nothing is withdrawn that was never declared",
          [c for c in scene.calls if c[1] is False] == [], str(scene.calls))
    print()


@_checked
def test_the_withdrawal_happens_even_when_the_take_fails():
    """CaptureAbort, a cable refusal, Ctrl-C -- the matrix still goes back."""
    print("=== the withdrawal happens even when the take fails ===")
    scene = _Scene()
    ns = _load(scene)
    with pytest.raises(RuntimeError, match="the pass was refused"):
        with ns["_cap_specimen_contact"]():
            raise RuntimeError("the pass was refused")
    check("it was withdrawn anyway", scene.allowed == (), str(scene.allowed))
    check("one declaration, one withdrawal",
          scene.calls == [(LINKS, True), (LINKS, False)], str(scene.calls))
    print()


@_checked
def test_a_failed_withdrawal_warns_without_masking_the_real_error():
    """An exception raised while unwinding would replace the one that matters."""
    print("=== a failed withdrawal warns without masking the real error ===")
    scene = _Scene(fail_on=("clear",))
    ns = _load(scene)
    with pytest.raises(RuntimeError, match="the pass was refused"):
        with ns["_cap_specimen_contact"]():
            raise RuntimeError("the pass was refused")
    check("it warned about the withdrawal",
          any("could not withdraw" in s for s in ns["_said"]), str(ns["_said"]))

    # ...and on a clean exit it is still only a warning.
    scene = _Scene(fail_on=("clear",))
    ns = _load(scene)
    with ns["_cap_specimen_contact"]():
        pass
    check("a clean take with a failed withdrawal does not raise", True)
    check("and says how to clear it by hand",
          any("allow=False" in s for s in ns["_said"]), str(ns["_said"]))
    print()


@_checked
def test_capture_contact_uses_it_and_can_opt_out():
    """The context manager is worth nothing if capture_contact does not enter
    it, and that cannot be checked by running the scenario without a cell."""
    print("=== capture_contact uses it and can opt out ===")
    src = open(os.path.join(HERE, "cell_capture_pipeline.dpi")).read()
    body = re.search(r"^def capture_contact\(.*?(?=^@|^def |\Z)",
                     src, re.S | re.M)
    assert body, "capture_contact() moved"
    body = body.group(0)
    check("it takes declare_contact", "declare_contact=True" in body)
    check("it enters the context manager",
          "_cap_specimen_contact(declare=declare_contact)" in body)
    check("around the whole scenario, not one move",
          body.index("_cap_specimen_contact") < body.index('capture_scenario("contact"'))
    check("and the take records what it did",
          '"declare_contact": bool(declare_contact)' in body)
    print()


_SECTIONS = ["test_contact_is_declared_for_the_take_and_withdrawn_after",
             "test_a_declaration_somebody_else_made_is_left_alone",
             "test_no_specimen_in_the_scene_says_so_rather_than_raising",
             "test_declare_contact_false_keeps_every_protection",
             "test_a_scene_that_will_not_answer_warns_and_runs_anyway",
             "test_the_withdrawal_happens_even_when_the_take_fails",
             "test_a_failed_withdrawal_warns_without_masking_the_real_error",
             "test_capture_contact_uses_it_and_can_opt_out"]


def _main():
    for name in _SECTIONS:
        fn = globals()[name]
        getattr(fn, "__wrapped__", fn)()
    if fails:
        print("FAILURES:", fails)
        return 1
    print("all contact-declaration checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
