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
# File:  tool_speed_toggle_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-01
# Description:
#   Host tests for the tool-speed solver toggle: that it is inert
#   until it is switched on, and that both call sites honour it.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-01:  Initial Commit
#
#########################################################################
"""The toggle that decides whether tool speed is commanded in m/s.

NO ROS, NO CELL, NO ARM. scan_plan.dpi's helpers are lifted out by AST, the same
way sensor_orientation_test.py does it, so the test cannot drift into testing a
copy.

WHAT MATTERS HERE, in order:

  1. IT IS INERT BY DEFAULT. tool_speed_kwargs() must return an EMPTY dict until
     set_tool_speed_solver(True) is called, because the whole safety argument for
     adding it to live call sites is that the calls are unchanged until someone
     deliberately changes them. A default that leaked a tool_speed_mps into every
     CartesianMove would silently alter the speed of every move in the pipeline.

  2. IT CANNOT INVENT A SPEED. A stage that does not know its own speed in m/s
     must get an empty dict rather than a guess -- the planner treats
     tool_speed_mps as the speed to deliver, and a wrong value there is a wrong
     speed near the part.

  3. BOTH CALL SITES GO THROUGH IT. surface_path_scan.dpi's raster row and
     cell_capture_motion.dpi's capture move are the two places converted; a
     third call site added later without the kwargs would silently keep the
     curve, which is the bug this check exists for.

Run it either way:

    python3 tool_speed_toggle_test.py
    python3 -m pytest -q tool_speed_toggle_test.py
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _dpi_tree(name):
    """A .dpi parsed with its `global` declarations stripped.

    A .dpi is compiled as a FUNCTION BODY, so `global x` at its top level is
    legal there and a syntax error in a module, which is what ast.parse treats
    the text as.
    """
    src = open(os.path.join(HERE, name)).read()
    body = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    return ast.parse(body, filename=name)


def _helpers(plan=None, solver_on=None):
    """scan_plan.dpi's tool-speed helpers, sharing one namespace as globals.

    The namespace IS their globals(), which is what lets TOOL_SPEED_SOLVER and
    scan_plan be steered from here. TOOL_SPEED_SOLVER's own default is taken
    from the source unless `solver_on` overrides it -- so the default is under
    test rather than restated.
    """
    tree = _dpi_tree("scan_plan.dpi")
    wanted = ("set_tool_speed_solver", "tool_speed_solver",
              "describe_tool_speed_solver", "raster_tool_speed",
              "tool_speed_kwargs")
    nodes = {}
    default = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes[node.name] = node
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "TOOL_SPEED_SOLVER"):
            default = ast.literal_eval(node.value)
    missing = [w for w in wanted if w not in nodes]
    assert not missing, (
        "scan_plan.dpi no longer defines %s at its top level -- renamed, moved, "
        "or deleted, and this suite would otherwise test nothing"
        % ", ".join(missing))
    assert default is not None, (
        "scan_plan.dpi no longer assigns TOOL_SPEED_SOLVER at its top level")
    said = []
    ns = {"scan_plan": plan,
          "TOOL_SPEED_SOLVER": default if solver_on is None else solver_on,
          "print": lambda *a: said.append(" ".join(str(x) for x in a))}
    mod = ast.Module(body=[nodes[w] for w in wanted], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "scan_plan.dpi", "exec"), ns)
    ns["_said"] = said
    ns["_source_default"] = default
    return ns


PLAN = {"effective_velocity_mps": 0.0085, "requested_velocity_mps": 0.010,
        "velocity_scaling": 0.00473}


# ==============================================================================
# 1. Inert by default
# ==============================================================================

def test_the_source_default_is_off():
    """If this flips, every CartesianMove in the pipeline changes at once."""
    ns = _helpers()
    assert ns["_source_default"] is False, (
        "TOOL_SPEED_SOLVER defaults to %r in scan_plan.dpi. The call sites were "
        "converted on the promise that they are unchanged until someone asks, "
        "so a True default silently alters the speed of every move."
        % (ns["_source_default"],))


def test_kwargs_are_empty_with_the_source_default():
    ns = _helpers(plan=PLAN)
    assert ns["tool_speed_kwargs"](0.010) == {}
    assert ns["tool_speed_kwargs"]() == {}, (
        "even with a plan in hand, the default must send nothing")


def test_turning_it_on_sends_the_speed():
    ns = _helpers(plan=PLAN)
    ns["set_tool_speed_solver"](True)
    assert ns["TOOL_SPEED_SOLVER"] is True
    assert ns["tool_speed_kwargs"](0.010) == {"tool_speed_mps": 0.010}


def test_turning_it_back_off_restores_exactly_nothing():
    ns = _helpers(plan=PLAN)
    ns["set_tool_speed_solver"](True)
    ns["set_tool_speed_solver"](False)
    assert ns["TOOL_SPEED_SOLVER"] is False
    assert ns["tool_speed_kwargs"](0.010) == {}, (
        "reverting has to be exact, or the toggle is not a toggle")


# ==============================================================================
# 2. It cannot invent a speed
# ==============================================================================

def test_no_plan_and_no_speed_sends_nothing():
    ns = _helpers(plan=None, solver_on=True)
    assert ns["tool_speed_kwargs"]() == {}, (
        "with no plan there is no speed to deliver; a stage that does not know "
        "its own speed must not have one guessed for it")


def test_the_plan_supplies_the_speed_when_the_caller_does_not():
    ns = _helpers(plan=PLAN, solver_on=True)
    assert ns["tool_speed_kwargs"]() == {"tool_speed_mps": 0.0085}


def test_it_is_the_effective_speed_not_the_requested_one():
    """The effective one is what the scaling was derived from."""
    ns = _helpers(plan=PLAN, solver_on=True)
    got = ns["tool_speed_kwargs"]()["tool_speed_mps"]
    assert got == PLAN["effective_velocity_mps"]
    assert got != PLAN["requested_velocity_mps"], (
        "this test's premise is that the two differ in the fixture")


def test_zero_and_negative_send_nothing():
    """None legitimately falls back to the plan; 0.0 must not.

    A zero speed is a missing value, not a command to stand still, and the
    planner would read it as "no tool speed given" anyway -- but only because
    tool_speed_kwargs refuses it first. Asserted so that stays true.
    """
    ns = _helpers(plan=PLAN, solver_on=True)
    assert ns["tool_speed_kwargs"](0.0) == {}
    assert ns["tool_speed_kwargs"](-0.010) == {}


def test_a_plan_with_no_speed_in_it_sends_nothing():
    ns = _helpers(plan={"velocity_scaling": 0.005}, solver_on=True)
    assert ns["tool_speed_kwargs"]() == {}
    ns = _helpers(plan={"effective_velocity_mps": 0.0}, solver_on=True)
    assert ns["tool_speed_kwargs"]() == {}


# ==============================================================================
# 3. The description says which mode is in force
# ==============================================================================

def test_the_description_distinguishes_the_two_modes():
    off = _helpers(solver_on=False)["describe_tool_speed_solver"]()
    on = _helpers(solver_on=True)["describe_tool_speed_solver"]()
    assert off and on and off != on
    assert "CURVE" in off.upper(), (
        "the off state must name the curve, so a reader knows what is "
        "converting their m/s")
    assert "M/S" in on.upper()
    # The setter prints it, so an operator who flips it sees what they got.
    ns = _helpers(solver_on=False)
    ns["set_tool_speed_solver"](True)
    assert ns["_said"], "set_tool_speed_solver() must report what it did"


# ==============================================================================
# 4. Both converted call sites actually pass it through
# ==============================================================================

#: file -> the function whose CartesianMove call must carry the kwargs.
CALL_SITES = {
    "surface_path_scan.dpi": "sweep",
    "cell_capture_motion.dpi": "capture_move",
}


def _calls_tool_speed_kwargs(tree, func_name):
    """True when `func_name` expands tool_speed_kwargs into a call."""
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        return None
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            # `**(globals().get("tool_speed_kwargs") or ...)(speed)` is a
            # keyword with arg=None whose value mentions the name as a string.
            if kw.arg is not None:
                continue
            if "tool_speed_kwargs" in ast.dump(kw.value):
                return True
    return False


def test_both_call_sites_pass_the_kwargs():
    for filename, func in CALL_SITES.items():
        tree = _dpi_tree(filename)
        found = _calls_tool_speed_kwargs(tree, func)
        assert found is not None, (
            "%s no longer defines %s(), so the call site this checks is gone"
            % (filename, func))
        assert found, (
            "%s's %s() commands a Cartesian move without expanding "
            "tool_speed_kwargs(), so it will always use the calibration curve "
            "no matter what set_tool_speed_solver() says" % (filename, func))


def test_the_startup_banner_states_which_mode_is_in_force():
    """Two modes that produce identical logs everywhere else must be announced."""
    dgp = open(os.path.join(HERE, "..", "ECAutomatedScan.dgp")).read()
    assert "describe_tool_speed_solver()" in dgp, (
        "ECAutomatedScan.dgp's startup banner no longer prints the tool-speed "
        "mode. A run flown on the solver and a run flown on the curve are "
        "otherwise indistinguishable in the log.")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
