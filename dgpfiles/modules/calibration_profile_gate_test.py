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
# File:  calibration_profile_gate_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that a degenerate Keyence grab is refused where profile data
#   enters a calibration.
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
"""A bad Keyence grab is refused where it enters, not where it is noticed.

calibration_test.py proves the arithmetic in calibration_solve.check_profile().
This proves the WIRING: that _cal_profile_points() in calibration.dpi -- the one
place profile data enters a calibration, used by both calibrate_keyence() and
verify_calibration() -- actually runs that check, raises CalibrationRefused
before returning anything, and drops the +Inf columns that pc2's skip_nans does
not.

Wiring is exactly what a unit test of the solver cannot see. With no check at
the entry point, a degenerate grab fits perfectly and the pipeline reports

    keyence  z = 0.6632 m   0.000 mm rms flat, 48000 points from 60 profiles
    AZURE IS +244.33 mm AND 1571.91 mrad OFF THE KEYENCE

as a calibration verdict about the Azure -- a Keyence fault blamed on the other
sensor.

Runs on the host: the .dpi function is lifted out by AST and executed against
stub messages, so there is no ROS, no recorder and no cell.

    python3 calibration_profile_gate_test.py
"""
import ast
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_solve as cs                                  # noqa: E402


class Refused(Exception):
    """Stands in for calibration.dpi's CalibrationRefused."""


class _Msg(object):
    def __init__(self, pts):
        self.pts = pts
        self.header = types.SimpleNamespace(frame_id="lj_v7080_optical_frame")


def _stub_ros():
    for m in ("sensor_msgs", "sensor_msgs.msg", "sensor_msgs_py",
              "sensor_msgs_py.point_cloud2"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["sensor_msgs.msg"].PointCloud2 = object
    sys.modules["sensor_msgs"].msg = sys.modules["sensor_msgs.msg"]
    sys.modules["sensor_msgs_py"].point_cloud2 = \
        sys.modules["sensor_msgs_py.point_cloud2"]
    sys.modules["sensor_msgs_py.point_cloud2"].read_points_numpy = \
        lambda m, field_names=None, skip_nans=False: m.pts


def _profile_points(messages, state=None):
    """calibration.dpi's _cal_profile_points, bound to stubs and to `messages`."""
    _stub_ros()
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_cal_profile_points")
    ns = {"np": np, "CalibrationRefused": Refused,
          "_CAL": {} if state is None else state,
          "CAL_PROFILES_PER_POSE": 60,
          "CAL_KEYENCE_REF_DIST": 0.080,
          "CAL_KEYENCE_RANGE_M": (0.057, 0.103),
          "_cal_geom": lambda: cs,
          "_cal_one_message": lambda *a, **k: list(messages)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "calibration.dpi", "exec"), ns)
    return ns["_cal_profile_points"]


def _grab(z0=0.0, noise=7.0e-6, n_cols=800, seed=1):
    """One profile in the head's own frame: x across the line, y = 0, z the raw
    reading (zero at the reference distance, positive towards the head)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.0155, 0.0155, n_cols)
    z = np.full(n_cols, float(z0))
    if noise:
        z = z + rng.normal(0.0, noise, n_cols)
    return np.column_stack([x, np.zeros(n_cols), z])


def _sixty(pts):
    return [_Msg(pts) for _ in range(60)]


def test_a_good_grab_is_returned_and_recorded():
    state = {}
    pts, frame, n = _profile_points(_sixty(_grab()), state)("/profiles")
    assert pts.shape == (48000, 3), pts.shape
    assert frame == "lj_v7080_optical_frame" and n == 60
    # and the measurements are kept for the take's record even on a good day
    chk = state["last_profile_check"]
    assert chk["ok"] and abs(chk["standoff"] - 0.080) < 1e-4, chk


def test_a_constant_never_reaches_the_caller():
    try:
        _profile_points(_sixty(_grab(noise=0.0)))("/profiles")
    except Refused as e:
        msg = str(e)
    else:
        raise AssertionError("48 000 identical heights were accepted")
    assert "SCATTER" in msg and "REPEATED VALUE" in msg, msg
    assert "standing off the surface" in msg, msg
    assert "no solve and no verdict has been reached" in msg, msg


def test_the_2026_09_17_grab_is_refused_with_the_range_named():
    """Constant AND 337 mm from a head that measures over 57..103 mm."""
    try:
        _profile_points(_sixty(_grab(z0=-0.2575, noise=0.0)))("/profiles")
    except Refused as e:
        msg = str(e)
    else:
        raise AssertionError("a surface 337 mm from the head was accepted")
    assert "337.5 mm from the head" in msg, msg
    assert "57..103 mm" in msg, msg


def test_out_of_range_columns_are_dropped_before_any_fit():
    """keyence_driver_node emits them as +Inf and pc2's skip_nans tests isnan,
    so they arrive here. solve_surface_fit() does not filter them and one
    infinity in a least-squares residual poisons every parameter in it."""
    half = _grab()
    half[::2, 2] = np.inf
    pts, _frame, _n = _profile_points(_sixty(half))("/profiles")
    assert pts.shape[0] == 24000, pts.shape
    assert np.isfinite(pts).all()


def test_an_all_infinite_grab_is_refused_for_coverage():
    try:
        inf = _grab()
        inf[:, 2] = np.inf
        _profile_points(_sixty(inf))("/profiles")
    except Refused as e:
        assert "POINTS:" in str(e) and "+Inf" in str(e), str(e)
    else:
        raise AssertionError("a grab with no finite point was accepted")


def test_the_check_can_be_turned_off_but_is_on_by_default():
    """check=False exists for looking at a bad grab deliberately. Nothing in
    the calibration path passes it, and the default must never change."""
    fn = _profile_points(_sixty(_grab(noise=0.0)))
    pts, _f, _n = fn("/profiles", check=False)
    assert pts.shape[0] == 48000
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    assert "def _cal_profile_points(topic, count=None, timeout=15.0, check=True)" in src
    assert "check=False" not in src, "something in the calibration path opted out"


# ------------------------------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            failed.append((name, e))
            print("  FAIL  %s\n        %s" % (name, e))
        else:
            print("  ok    %s" % name)
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    if failed:
        raise SystemExit(str(failed[0][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
