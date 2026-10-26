"""Where the work cell's speed curve lives, and what happens when it does not.

WHY THIS MATTERS. A MoveIt velocity scaling is a fraction of the arm's JOINT
limits, not a speed. The technician's control is mm/s, so every commanded speed
is converted through a measured curve -- and the curve is a property of the cell,
not of the code. Get it from the wrong cell and the arm sweeps at a speed nobody
asked for.

That is not hypothetical. With a SIMULATION-derived curve in force on hardware, an
eddy-current scan asked for 10 mm/s, swept at 4.1, and stopped after one pass of
sixty-two. The measurement that fixed it existed in a capture file at the time;
there was nowhere to put it except a `globals()` assignment at the prompt, which
does not persist -- so it was pasted again in five separate sessions.

The fix is that the curve belongs in the CELL MANIFEST, per mode, the way the work
volume and the cable envelope already do: the manifest travels with the cell and
the capture file does not. These tests hold that in place, and hold the refusals
that stop a bad table being loaded silently.

Runs on the host. tari_bringup is stubbed, so no ROS is needed.
"""

import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: The work cell whose manifest this checks, and the mode the hardware curve is
#: for.
CELL = "ex-ur10e-1"
#: The workspace root, found from this file rather than hardcoded: these
#: suites run from a checkout that may live anywhere.
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MANIFEST = os.path.join(WORKSPACE, "ros_ws", "src", "tari_site",
                        "tari_cell_configs", "config", "cells",
                        "%s.yaml" % (CELL,))

#: A transcribed curve standing in for whatever a config happens to hold.
CONFIG_TABLE = [(0.0050, 0.0079), (0.0100, 0.0156), (0.0200, 0.0303),
                (0.0350, 0.0472), (0.0500, 0.0529)]


#: sys.modules entries to put back after each test, innermost last.
_SAVED = []


def _stub_bringup(manifest):
    """find_manifest, without ROS.

    The stub has to stay installed while the loader RUNS, not only while the .dpi
    is compiled: robot_velocity_calibration() imports find_manifest at call time.
    Removing it too early is why this harness first reported the manifest as
    absent.
    """
    _SAVED.append((sys.modules.get("tari_bringup"),
                   sys.modules.get("tari_bringup.cells")))
    pkg = types.ModuleType("tari_bringup")
    pkg.__path__ = []
    cells = types.ModuleType("tari_bringup.cells")
    cells.find_manifest = lambda name: manifest
    pkg.cells = cells
    sys.modules["tari_bringup"] = pkg
    sys.modules["tari_bringup.cells"] = cells


def _restore_all():
    while _SAVED:
        for name, mod in zip(("tari_bringup", "tari_bringup.cells"), _SAVED.pop()):
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _clean_bringup():
    yield
    _restore_all()


def _load(mode="hardware", manifest=MANIFEST, config=None):
    """work_cell.dpi in its own namespace, for one cell and mode.

    Leaves the tari_bringup stub installed; the fixture above takes it out.
    """
    _stub_bringup(manifest)
    src = open(os.path.join(HERE, "work_cell.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "robot_velocity_manifest_test",
          "CELL_NAME": CELL, "CELL_MODE": mode,
          "WARNING_LIGHT_PIN": 0,
          "set_digital_out": lambda pin, state: True}
    exec(compile(src, "work_cell.dpi", "exec"), ns)
    ns["ROBOT_VELOCITY_CALIBRATION"] = list(
        CONFIG_TABLE if config is None else config)
    return ns


def _write(tmp_path, block):
    """A minimal manifest carrying just a robot_velocity block."""
    path = tmp_path / "cell.yaml"
    path.write_text("cell: %s\nmodes:\n  hardware: {}\n  sim: {}\n%s"
                    % (CELL, block))
    return str(path)


# ---------------------------------------------------------------------------
# The real manifest
# ---------------------------------------------------------------------------

def test_the_cell_manifest_declares_a_curve_for_both_modes():
    """The location. If this fails the curve is back in a config constant."""
    if not os.path.exists(MANIFEST):
        pytest.skip("this cell's manifest is not on this machine")
    import yaml
    doc = yaml.safe_load(open(MANIFEST))
    block = doc.get("robot_velocity")
    assert block, (
        "%s declares no robot_velocity: block, so the speed curve has to be "
        "transcribed into every config that drives this cell" % (MANIFEST,))
    for mode in ("sim", "hardware"):
        assert block.get(mode), "robot_velocity has no %r block" % (mode,)
        assert block[mode].get("table"), "robot_velocity.%s has no table" % (mode,)


def test_the_hardware_curve_is_loaded_and_beats_the_config():
    if not os.path.exists(MANIFEST):
        pytest.skip("this cell's manifest is not on this machine")
    ns = _load("hardware")
    table, from_manifest = ns["adopt_measured_velocity_curve"](quiet=True)
    assert from_manifest is True
    assert table != CONFIG_TABLE, "the transcribed table is still in force"
    assert ns["ROBOT_VELOCITY_CALIBRATION"] == table
    # The config's own value stays reachable, so a run can say what it would
    # have used.
    assert ns["ROBOT_VELOCITY_CALIBRATION_CONFIG"] == CONFIG_TABLE


def test_each_mode_gets_its_own_curve_and_they_differ():
    """Never copied between modes: that is the failure this exists to stop."""
    if not os.path.exists(MANIFEST):
        pytest.skip("this cell's manifest is not on this machine")
    hw = _load("hardware")["robot_velocity_calibration"](quiet=True)
    sim = _load("sim")["robot_velocity_calibration"](quiet=True)
    assert hw and sim
    assert hw["table"] != sim["table"], (
        "the two modes declare the same curve; a simulated curve on hardware is "
        "what made a scan sweep at 4.1 mm/s when asked for 10")


def test_the_loaded_curve_matches_the_capture_file():
    """The manifest is a transcription of the measurement -- check it is faithful."""
    import json
    cap = os.path.join(WORKSPACE, "scandata", "results14", "scandata", "capture",
                       "robot_velocity_%s_hardware.json" % (CELL,))
    if not (os.path.exists(cap) and os.path.exists(MANIFEST)):
        pytest.skip("the capture file or the manifest is not on this machine")
    measured = json.load(open(cap))
    cal = _load("hardware")["robot_velocity_calibration"](quiet=True)
    assert cal is not None
    assert len(cal["table"]) == len(measured["table"])
    for (s, v), row in zip(cal["table"], sorted(measured["table"])):
        assert abs(s - row[0]) < 1e-5 and abs(v - row[1]) < 1e-5, (
            "the manifest's row (%.5f, %.5f) does not match the measurement %r"
            % (s, v, row))
    assert cal["measured"] == measured["measured"], (
        "the manifest is dated %r and the measurement %r -- one of them is from "
        "a different run" % (cal["measured"], measured["measured"]))
    assert abs(cal["span_m"] - measured["span_m"]) < 1e-9


# ---------------------------------------------------------------------------
# What happens without one, and with a bad one
# ---------------------------------------------------------------------------

def test_no_block_falls_back_and_says_so(tmp_path):
    """A missing curve must be loud, not silent: every speed depends on it."""
    path = _write(tmp_path, "")
    ns = _load("hardware", manifest=path)
    table, from_manifest = ns["adopt_measured_velocity_curve"](quiet=True)
    assert from_manifest is False
    assert table == CONFIG_TABLE
    line = ns["describe_robot_velocity"](fallback=CONFIG_TABLE)
    assert "NOT MEASURED" in line


def test_the_other_modes_curve_is_never_borrowed(tmp_path):
    """A hardware run must not silently use the simulated curve."""
    path = _write(tmp_path, "robot_velocity:\n  sim:\n    table:\n"
                            "      - [0.005, 0.0079]\n      - [0.010, 0.0156]\n")
    assert _load("hardware", manifest=path)["robot_velocity_calibration"](
        quiet=True) is None
    assert _load("sim", manifest=path)["robot_velocity_calibration"](
        quiet=True) is not None


def test_a_speed_that_falls_as_the_scaling_rises_is_refused(tmp_path):
    """The shape that makes the inverse ambiguous.

    velocity_scaling_for() returns the LOWER of the two scalings that produce a
    speed inside the dip, so the cell is commanded slower than asked for from then
    on, permanently, and nothing says so. It has to be refused at load.
    """
    path = _write(tmp_path, "robot_velocity:\n  hardware:\n    table:\n"
                            "      - [0.005, 0.0156]\n      - [0.010, 0.0090]\n")
    ns = _load("hardware", manifest=path)
    assert ns["robot_velocity_calibration"](quiet=True) is None
    # And the refusal is specific, not a bare None: check the validator directly.
    with pytest.raises(ValueError) as e:
        ns["_robot_velocity_from_block"](
            {"table": [[0.005, 0.0156], [0.010, 0.0090]]}, CELL, "hardware")
    assert "falls as the scaling rises" in str(e.value)


def test_a_repeated_scaling_is_refused(tmp_path):
    ns = _load("hardware")
    with pytest.raises(ValueError) as e:
        ns["_robot_velocity_from_block"](
            {"table": [[0.005, 0.0079], [0.005, 0.0156]]}, CELL, "hardware")
    assert "two speeds for one command" in str(e.value)


def test_a_one_row_table_is_refused():
    ns = _load("hardware")
    with pytest.raises(ValueError) as e:
        ns["_robot_velocity_from_block"]({"table": [[0.005, 0.0079]]},
                                        CELL, "hardware")
    assert "nothing to interpolate" in str(e.value)


# ---------------------------------------------------------------------------
# Which source wins
# ---------------------------------------------------------------------------

def test_a_fresh_capture_file_beats_the_manifest(tmp_path):
    """The machine that just measured should use what it measured.

    Same order as the cable envelope, and for the same reason: without it an
    operator has to hand-edit a manifest before the measurement they just took has
    any effect, and that is the step that gets skipped.
    """
    import json
    manifest = _write(tmp_path, "robot_velocity:\n  hardware:\n"
                                "    measured: \"from the manifest\"\n"
                                "    table:\n      - [0.005, 0.0079]\n"
                                "      - [0.010, 0.0156]\n")
    capture = tmp_path / ("robot_velocity_%s_hardware.json" % (CELL,))
    capture.write_text(json.dumps({
        "measured": "from the capture file", "span_m": 0.3, "limit_mps": 0.04,
        "table": [[0.00206, 0.00514], [0.01649, 0.03266]]}))

    ns = _load("hardware", manifest=manifest)
    ns["_robot_velocity_file"] = lambda: str(capture)
    cal = ns["robot_velocity_calibration"](quiet=True)
    assert cal["measured"] == "from the capture file", (
        "the manifest won over a fresher capture file on this machine")
    assert cal["source"] == str(capture)
    assert ns["describe_robot_velocity"]().count("from ") >= 1, (
        "the report does not say which file the curve came from")


def test_the_manifest_is_used_when_there_is_no_capture_file(tmp_path):
    manifest = _write(tmp_path, "robot_velocity:\n  hardware:\n"
                                "    measured: \"from the manifest\"\n"
                                "    table:\n      - [0.005, 0.0079]\n"
                                "      - [0.010, 0.0156]\n")
    ns = _load("hardware", manifest=manifest)
    ns["_robot_velocity_file"] = lambda: str(tmp_path / "does-not-exist.json")
    cal = ns["robot_velocity_calibration"](quiet=True)
    assert cal["measured"] == "from the manifest"


def test_an_unreadable_capture_file_falls_through_to_the_manifest(tmp_path):
    """A corrupt fresh file must not mean no curve at all."""
    manifest = _write(tmp_path, "robot_velocity:\n  hardware:\n"
                                "    measured: \"from the manifest\"\n"
                                "    table:\n      - [0.005, 0.0079]\n"
                                "      - [0.010, 0.0156]\n")
    bad = tmp_path / ("robot_velocity_%s_hardware.json" % (CELL,))
    bad.write_text("{ this is not json")
    ns = _load("hardware", manifest=manifest)
    ns["_robot_velocity_file"] = lambda: str(bad)
    cal = ns["robot_velocity_calibration"](quiet=True)
    assert cal is not None and cal["measured"] == "from the manifest"


# ---------------------------------------------------------------------------
# The path-length caveat
# ---------------------------------------------------------------------------

def test_a_curve_measured_far_longer_than_the_pass_is_flagged():
    """The curve is path-length dependent and the passes are much shorter.

    Measured over 300 mm; the eddy-current passes are 10-52 mm. A move that much
    shorter is mostly acceleration ramp and never reaches the speed the table
    promises, which is why the scan report's observed speed is the number to
    trust. Saying so is the point -- nothing here can fix it.
    """
    if not os.path.exists(MANIFEST):
        pytest.skip("this cell's manifest is not on this machine")
    ns = _load("hardware")
    short = ns["describe_robot_velocity"](fallback=CONFIG_TABLE,
                                         pass_length_m=0.052)
    assert "MEASURED OVER 300 mm" in short
    assert "52 mm" in short
    # A pass comparable to the span says nothing about it.
    same = ns["describe_robot_velocity"](fallback=CONFIG_TABLE,
                                         pass_length_m=0.30)
    assert "MEASURED OVER" not in same


def test_the_span_is_recorded_so_the_comparison_is_possible():
    if not os.path.exists(MANIFEST):
        pytest.skip("this cell's manifest is not on this machine")
    cal = _load("hardware")["robot_velocity_calibration"](quiet=True)
    assert cal["span_m"], (
        "the manifest records no span_m, so nothing can tell whether the curve "
        "was measured over a distance like the one it is being applied to")


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            print("SKIP  %s (needs pytest's tmp_path)" % name)
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (name, e))
        finally:
            _restore_all()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
