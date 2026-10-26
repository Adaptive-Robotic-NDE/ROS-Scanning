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
# File:  start_state_recovery_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Proves an arm left touching the specimen is backed off along the tool axis,
#   and that any other collision is reported instead of moved.
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
"""Can a run recover from an arm left resting on the part?

WHAT THIS PROTECTS. A contact scan that stops between "probe down" and "probe
lifted" leaves the tip on the specimen. The planner validates the start state
before it plans anything, so the NEXT run is refused at its first move -- a move
to a named pose that has nothing to do with the part -- and the only way out was
to jog the arm off by hand.

The recovery is one motion: straight off along the tool axis. The tests that
matter here are the ones that prove it is NOT attempted in the other cases,
because the wrong instinct next to real hardware is to move the arm on a guess
about why the planner is unhappy.

Runs on the host with stubs. No ROS, no hardware.
"""

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

LINKS = ("ec_probe", "probe_holder")


class FakeRos(object):
    """CheckStateValidity with the answer set per call, and a call log.

    `answers` maps frozenset(allow_contact_links) -> True / False / None, which is
    exactly the distinction the recovery turns on: invalid with nothing excused
    but valid with the contact links excused means the probe is on the part.
    """

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def CheckStateValidity(self, positions, allow_contact_links=None, timeout=5.0):
        key = frozenset(allow_contact_links or [])
        self.calls.append(key)
        return self.answers[key]


class FakeNode(object):
    def __init__(self, joints=(0.0,)*6):
        self.joints = list(joints) if joints is not None else None

    def current_joints(self, max_age=1.0):
        return None if self.joints is None else list(self.joints)


def _load(**injected):
    """The .dpi in its own namespace, with the work cell's names stubbed in.

    The module reads the cell through globals().get(), and a .dpi exec'd with one
    dict for globals and locals puts its own definitions in that same dict -- so
    injecting here is what including it into a live config does.
    """
    ns = {}
    exec("import numpy as np\n", ns)
    src = open(os.path.join(MODULE_DIR, "start_state_recovery.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    exec(compile(src, "start_state_recovery.dpi", "exec"), ns)
    ns.update(injected)
    return ns


def _lifts():
    """A stand-in contact_lift that records what it was asked to do."""
    done = []

    def contact_lift(settings, height, phase=None):
        done.append(float(height))
        return True
    return contact_lift, done


def test_a_valid_start_state_moves_nothing():
    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): True}),
               capture_node=FakeNode(),
               contact_lift=lift,
               TOOLING_CONTACT_LINKS=LINKS)
    rep = ns["recover_contact_start_state"](quiet=True)
    assert rep["needed"] is False
    assert rep["recovered"] is True
    assert done == [], "the arm was moved from a state the planner already accepts"


def test_a_probe_on_the_part_is_backed_off_along_the_tool_axis():
    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): False,
                            frozenset(LINKS): True}),
               capture_node=FakeNode(),
               contact_lift=lift,
               TOOLING_CONTACT_LINKS=LINKS)
    rep = ns["recover_contact_start_state"](retreat_m=0.05, quiet=True)
    assert rep["needed"] is True
    assert len(done) == 1, "expected exactly one retreat, got %r" % (done,)
    assert done[0] == 0.05
    # The verdict is re-read after the move, so a still-invalid state is not
    # reported as recovered.
    assert rep["recovered"] is False, (
        "the stub keeps answering 'invalid', so the recovery must not claim "
        "success -- it has to re-check rather than assume the lift worked")


def test_recovery_reports_success_only_after_re_checking():
    lift, done = _lifts()
    answers = {frozenset(): False, frozenset(LINKS): True}
    ros = FakeRos(answers)

    def lift_and_clear(settings, height, phase=None):
        done.append(float(height))
        answers[frozenset()] = True          # off the part now
        return True

    ns = _load(ros=ros, capture_node=FakeNode(), contact_lift=lift_and_clear,
               TOOLING_CONTACT_LINKS=LINKS)
    rep = ns["recover_contact_start_state"](retreat_m=0.04, quiet=True)
    assert rep["needed"] is True
    assert rep["recovered"] is True
    assert rep["retreat_m"] == 0.04
    assert frozenset() in ros.calls, "the state was never re-checked after moving"


def test_a_collision_elsewhere_is_reported_and_nothing_moves():
    """The case where backing off is the WRONG answer.

    Invalid even with the contact links excused means something other than the
    probe is touching. Retreating along the tool axis would not fix it and would
    move the arm for no reason.
    """
    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): False,
                            frozenset(LINKS): False}),
               capture_node=FakeNode(),
               contact_lift=lift,
               TOOLING_CONTACT_LINKS=LINKS)
    rep = ns["recover_contact_start_state"](quiet=True)
    assert rep["needed"] is True
    assert rep["recovered"] is False
    assert done == [], "the arm was moved despite the collision not being the probe"


def test_an_unanswered_service_moves_nothing():
    """None is not False. An unverifiable state is not permission to move."""
    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): None}),
               capture_node=FakeNode(),
               contact_lift=lift,
               TOOLING_CONTACT_LINKS=LINKS)
    rep = ns["recover_contact_start_state"](quiet=True)
    assert rep["needed"] is None
    assert rep["recovered"] is False
    assert done == []


def test_no_joint_states_moves_nothing():
    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): True}),
               capture_node=FakeNode(joints=None),
               contact_lift=lift,
               TOOLING_CONTACT_LINKS=LINKS)
    verdict, why = ns["start_state_validity"]()
    assert verdict is None
    assert "joint_states" in why
    rep = ns["recover_contact_start_state"](quiet=True)
    assert rep["needed"] is None
    assert done == []


def test_a_known_toolpath_retracts_by_the_remaining_distance():
    """With the path in hand, only the shortfall is flown.

    Lifting a fixed height every time would walk the arm up and out of reach
    across repeated attempts.
    """
    calls = []

    def contact_retract(settings, positions=None, phase=None):
        calls.append(len(positions))
        return True

    lift, done = _lifts()
    ns = _load(ros=FakeRos({frozenset(): False, frozenset(LINKS): True}),
               capture_node=FakeNode(), contact_lift=lift,
               contact_retract=contact_retract,
               TOOLING_CONTACT_LINKS=LINKS)
    ns["recover_contact_start_state"](positions=[(0.0, 0.0, 0.0)]*7, quiet=True)
    assert calls == [7], "contact_retract was not preferred when the path is known"
    assert done == [], "it lifted a fixed height as well as retracting"


def test_describe_start_state_names_the_probe_case():
    """The operator-facing line has to distinguish the two collisions.

    They have different fixes -- one is automatic, the other needs a person -- so
    a single 'in collision' message would be useless.
    """
    lift, _ = _lifts()
    probe = _load(ros=FakeRos({frozenset(): False, frozenset(LINKS): True}),
                  capture_node=FakeNode(), contact_lift=lift,
                  TOOLING_CONTACT_LINKS=LINKS)["describe_start_state"]()
    other = _load(ros=FakeRos({frozenset(): False, frozenset(LINKS): False}),
                  capture_node=FakeNode(), contact_lift=lift,
                  TOOLING_CONTACT_LINKS=LINKS)["describe_start_state"]()
    assert "PROBE IS ON THE PART" in probe
    assert "PROBE IS ON THE PART" not in other
    assert "not at the contact links" in other


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (name, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
