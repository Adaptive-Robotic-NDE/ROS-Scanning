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
# File:  ec_liftoff_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   Host tests for the offline lift-off characterisation, including a
#   check that its conditioning math still matches the C++ it was
#   copied from.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-02:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""The lift-off characterisation, on synthesised passes with known answers.

NO ROS, NO CELL, NO ARM, NO spatialnde2. Every pass here is built from a known
lift-off axis, a known tilt sensitivity and a known amount of wander, so the
tests check that the analysis RECOVERS numbers rather than that it runs.

The test that matters most is the first one: ec_liftoff.condition() is a
reproduction of C++ that lives in another repository, and a reproduction that
drifts is worse than no reproduction, because every number downstream would
still look plausible. So the formula is re-derived from the C++ source text
itself and compared.
"""
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ec_liftoff                                            # noqa: E402

#: The C++ this file's condition() reproduces.
EC_CPP = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "spatialnde2-ec-functions",
    "spatialnde2_ec_functions", "ec_condition_cpp.hpp"))


# ==============================================================================
# 1. The reproduced math must still match its original
# ==============================================================================

@pytest.mark.skipif(not os.path.exists(EC_CPP),
                    reason="spatialnde2-ec-functions is not checked out here")
def test_the_conditioning_still_matches_the_cpp():
    """Guard against the reproduction drifting from the C++ it copies.

    Not a text comparison -- the C++ could be reformatted freely. Instead the
    three things that are easy to get wrong and impossible to notice are
    asserted from the source text:

      the SIGN of the rotation   theta = -rot_deg * pi/180. Get this backwards
                                 and the lift-off axis comes out as the defect
                                 axis, with every number still looking sane.
      the ORDER                  null, then rotate, then gain. Rotate first and
                                 the null becomes a moving offset.
      which output is which      h -> horizontal (lift-off), v -> vertical.
    """
    src = open(EC_CPP).read()
    assert "const snde_float64 theta = -rot_deg * M_PI / 180.0;" in src, (
        "the C++ no longer negates the rotation the way condition() assumes. "
        "If the sign changed there, condition() is now computing the DEFECT "
        "axis and calling it lift-off.")
    # null before rotate: the subtractions must precede the h/v combination.
    null_at = src.find("re -= null_re;")
    rot_at = src.find("snde_float64 h = re*ct - im*st;")
    assert 0 < null_at < rot_at, (
        "the C++ no longer nulls before rotating; condition() does")
    assert "horizontal->element(pos) = (snde_float32)h;" in src, (
        "`horizontal` is no longer the h component in the C++")
    assert "vertical->element(pos) = (snde_float32)v;" in src


def test_condition_matches_a_hand_computed_case():
    """One case worked by hand, so the formula is pinned independently."""
    # z = 3 + 4j, null = 1 + 1j  ->  2 + 3j. Rotate by rotation_deg = 90, i.e.
    # multiply by exp(-i*pi/2) = -i:  (2 + 3j)(-i) = 3 - 2j.
    h, v = ec_liftoff.condition([3 + 4j], null=(1.0, 1.0), rotation_deg=90.0)
    assert h[0] == pytest.approx(3.0, abs=1e-12)
    assert v[0] == pytest.approx(-2.0, abs=1e-12)


def test_rotation_moves_a_feature_onto_the_horizontal_axis():
    """The instrument convention: 'lift-off at 40 degrees' means rotate by 40.

    This is the whole reason the sign is what it is, and it is the property an
    operator relies on when they dial the rotation by eye.
    """
    angle = 40.0
    z = np.exp(1j * np.radians(angle)) * np.array([0.5, 1.0, 1.5])
    h, v = ec_liftoff.condition(z, rotation_deg=angle)
    assert np.allclose(v, 0.0, atol=1e-12), (
        "a feature lying at +40 deg must land flat on the horizontal axis when "
        "the rotation is set to 40")
    assert np.allclose(h, [0.5, 1.0, 1.5], atol=1e-12)


def test_gain_scales_the_axes_independently():
    h, v = ec_liftoff.condition([1 + 2j], gain=(10.0, 100.0))
    assert h[0] == pytest.approx(10.0)
    assert v[0] == pytest.approx(200.0)


# ==============================================================================
# 2. Synthesising passes with known answers
# ==============================================================================

TRUE_AXIS_DEG = 37.0        #: the lift-off direction these fixtures are built on
PER_DEGREE = 0.004          #: signal units of lift-off per degree of tilt


def _pass(tilt_deg, n=400, travel_m=0.12, wander=0.0, trend=0.0,
          defect=0.0, seed=0, name=None):
    """A synthesised pass with a known lift-off axis and known content.

    lift-off axis carries: the tilt offset, a linear trend, and short-correlated
    noise. The defect axis carries an independent sinusoid, so the two can be
    told apart.
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, travel_m, n)
    liftoff = (PER_DEGREE * tilt_deg
               + trend * (s / travel_m)
               + wander * rng.standard_normal(n))
    defect_axis = defect * np.sin(2.0 * np.pi * s / (travel_m / 3.0))
    z = (liftoff + 1j * defect_axis) * np.exp(1j * np.radians(TRUE_AXIS_DEG))
    return {"name": name or ("ec_tilt%g.npz" % tilt_deg),
            "tilt_deg": float(tilt_deg), "impedance": z, "s_m": s,
            "position": np.zeros((n, 3)), "index": np.arange(n),
            "wall_time": s * 10.0, "path": name or "synthetic"}


DEFAULT_TILTS = (0.0, 3.0, -3.0, 6.0, -6.0)


def _take(**kw):
    return [_pass(t, seed=i, **kw) for i, t in enumerate(DEFAULT_TILTS)]


# ==============================================================================
# 3. The axis must be recovered from the data
# ==============================================================================

def test_the_axis_is_recovered_from_the_tilt_sweep():
    """The point of the whole exercise: do not trust the dialled-in rotation.

    The rotation is set at setup by eye, against a display, by an operator
    turning it until lift-off looks flat. That is right for an operator and
    wrong to build a measurement on. The tilt sweep is a commanded perturbation
    along the lift-off axis, so the axis is recoverable from the data.
    """
    axis, detail = ec_liftoff.liftoff_axis_from_tilts(_take(wander=1e-4))
    assert axis == pytest.approx(TRUE_AXIS_DEG, abs=1.0), (
        "recovered %.2f deg against a true %.2f" % (axis, TRUE_AXIS_DEG))
    assert detail["linearity"] > 0.95
    assert detail["passes_used"] == len(DEFAULT_TILTS)


def test_the_axis_survives_a_strong_defect_signal():
    """The part must not pull the axis. This is why per-pass MEANS are used.

    A defect signal ten times the tilt steps is present in every pass but
    averages out, because it is the same along every pass while the tilt offset
    is not.
    """
    axis, _ = ec_liftoff.liftoff_axis_from_tilts(
        _take(wander=1e-4, defect=0.3))
    assert axis == pytest.approx(TRUE_AXIS_DEG, abs=2.0)


def test_one_tilt_is_refused_rather_than_guessed():
    """A take with no sweep cannot locate the axis, and must say so."""
    axis, detail = ec_liftoff.liftoff_axis_from_tilts([_pass(0.0)])
    assert axis is None
    assert "two distinct tilts" in detail["reason"]


def test_the_axis_is_reported_in_a_single_half_turn():
    """+theta and theta+180 are the same axis; two runs must not disagree."""
    flipped = [_pass(-t, seed=i) for i, t in enumerate(DEFAULT_TILTS)]
    axis, _ = ec_liftoff.liftoff_axis_from_tilts(flipped)
    assert -90.0 < axis <= 90.0


# ==============================================================================
# 4. The ruler
# ==============================================================================

def test_the_tilt_ruler_recovers_the_sensitivity():
    """Spreads are meaningless in volts; the commanded tilt steps calibrate them."""
    passes = _take(wander=1e-4)
    axis, _ = ec_liftoff.liftoff_axis_from_tilts(passes)
    per_deg, detail = ec_liftoff.tilt_step_size(passes, axis)
    assert per_deg == pytest.approx(PER_DEGREE, rel=0.05), (
        "recovered %.6g per degree against a true %.6g" % (per_deg, PER_DEGREE))
    assert detail["residual_rms"] < 0.1 * PER_DEGREE


# ==============================================================================
# 5. Trend and residual must be separated -- this chooses the fix
# ==============================================================================

def test_a_registration_trend_is_identified_as_a_trend():
    """A tilted or offset mesh shows as a trend, and compliance cannot fix it.

    The whole value of separating these is that they lead to different work:
    a trend is calibration, a residual is compliance.
    """
    p = _pass(0.0, trend=10 * PER_DEGREE, wander=1e-5)
    r = ec_liftoff.analyse_pass(p, TRUE_AXIS_DEG, PER_DEGREE)
    assert r["trend_share"] > 0.9, (
        "a pure trend must be attributed to the trend, got %.2f"
        % r["trend_share"])
    assert r["trend_total_tiltdeg"] == pytest.approx(10.0, rel=0.05)


def test_short_wavelength_noise_is_identified_as_residual():
    """Per-vertex mesh noise is what a spring is for, and what a 1-2 Hz loop misses."""
    p = _pass(0.0, wander=5 * PER_DEGREE)
    r = ec_liftoff.analyse_pass(p, TRUE_AXIS_DEG, PER_DEGREE)
    assert r["trend_share"] < 0.2, (
        "pure noise must not be attributed to a trend, got %.2f"
        % r["trend_share"])
    assert r["residual_std_tiltdeg"] == pytest.approx(5.0, rel=0.15)
    cl = r["residual_corr_length_m"]
    assert cl is not None and cl < 0.005, (
        "uncorrelated noise must come back with a short correlation length, "
        "got %r m" % (cl,))


def test_a_smooth_excursion_comes_back_with_a_long_correlation_length():
    """Long means calibration's problem; short means compliance's.

    A sinusoid on the lift-off axis over the pass is smooth, so its residual
    stays correlated far longer than per-vertex noise does.
    """
    n, travel = 400, 0.12
    s = np.linspace(0.0, travel, n)
    liftoff = 5 * PER_DEGREE * np.sin(2.0 * np.pi * s / (travel / 2.0))
    z = liftoff * np.exp(1j * np.radians(TRUE_AXIS_DEG))
    p = {"name": "smooth", "tilt_deg": 0.0, "impedance": z, "s_m": s}
    r = ec_liftoff.analyse_pass(p, TRUE_AXIS_DEG, PER_DEGREE)
    assert r["residual_corr_length_m"] > 0.005


def test_the_verdict_has_an_honest_middle_band():
    """A hard 50% boundary gave contradictory advice on identical passes.

    Two passes flown back to back differ by a few percent of trend share by
    chance. With one threshold at 50% the report printed "fix the calibration"
    on one and "soften the spring" on the other from the same data, and both
    read as authoritative. The middle band exists so that case says so.
    """
    assert "COMPARABLE" in ec_liftoff._verdict(0.50)
    assert "COMPARABLE" in ec_liftoff._verdict(0.49)
    assert "COMPARABLE" in ec_liftoff._verdict(0.51)
    assert "REGISTRATION" in ec_liftoff._verdict(0.90)
    assert "RESIDUAL" in ec_liftoff._verdict(0.10)
    assert ec_liftoff.RESIDUAL_DOMINATES < ec_liftoff.TREND_DOMINATES, (
        "the bands must not overlap or invert")


def test_liftoff_against_the_defect_axis_is_reported():
    """If lift-off wanders more than the part varies, the image is mesh error."""
    noisy = ec_liftoff.analyse_pass(
        _pass(0.0, wander=10 * PER_DEGREE, defect=PER_DEGREE),
        TRUE_AXIS_DEG, PER_DEGREE)
    assert noisy["liftoff_over_defect"] > 1.0
    clean = ec_liftoff.analyse_pass(
        _pass(0.0, wander=0.01 * PER_DEGREE, defect=10 * PER_DEGREE),
        TRUE_AXIS_DEG, PER_DEGREE)
    assert clean["liftoff_over_defect"] < 1.0


def test_a_one_trace_pass_is_called_what_it_is():
    """One trace is the position trigger not firing, not a lift-off problem.

    This cell has recorded whole rows one trace wide, and reporting that as a
    lift-off statistic of zero spread would be actively misleading.
    """
    r = ec_liftoff.analyse_pass(
        {"name": "ec_tilt0.npz", "tilt_deg": 0.0,
         "impedance": np.array([1 + 1j]), "s_m": np.array([0.0])},
        TRUE_AXIS_DEG, PER_DEGREE)
    assert "position trigger" in r["reason"]


# ==============================================================================
# 6. Housekeeping
# ==============================================================================

@pytest.mark.parametrize("name,expect", [
    ("ec_tilt0.npz", 0.0), ("ec_tilt3.npz", 3.0), ("ec_tilt-3.npz", -3.0),
    ("ec_tilt6.npz", 6.0), ("ec_tilt-6.npz", -6.0),
    ("ec_contact_a.npz", None), ("something_else.npz", None),
])
def test_the_tilt_comes_off_the_filename(name, expect):
    """The filename is the ONLY place the tilt is recorded."""
    assert ec_liftoff._tilt_from_name(name) == expect


def test_an_empty_directory_says_what_is_missing():
    with pytest.raises(ValueError) as exc:
        ec_liftoff.load_take(HERE)
    assert "ec=True" in str(exc.value), (
        "the message must say how to make the files appear")


def test_the_report_renders_a_whole_take():
    """It has to produce readable text, not just a dict."""
    passes = _take(wander=2 * PER_DEGREE, defect=3 * PER_DEGREE)
    axis, axis_detail = ec_liftoff.liftoff_axis_from_tilts(passes)
    per_deg, step_detail = ec_liftoff.tilt_step_size(passes, axis)
    text = ec_liftoff.report({
        "passes": [ec_liftoff.analyse_pass(p, axis, per_deg) for p in passes],
        "axis_deg": axis, "axis_detail": axis_detail,
        "rotation_used_deg": axis, "dialled_rotation_deg": 10.0,
        "per_degree": per_deg, "step_detail": step_detail})
    assert "LIFT-OFF CHARACTERISATION" in text
    assert "tilt-deg" in text, "the ruler must appear in the output"
    assert "disagreement" in text, (
        "a 27 deg gap from the dialled-in rotation must be called out")
    for p in passes:
        assert p["name"] in text


def test_the_report_survives_a_take_with_no_sweep():
    """One tilt is a legitimate thing to have; it must not crash the report."""
    passes = [_pass(0.0, wander=PER_DEGREE)]
    text = ec_liftoff.report({
        "passes": [ec_liftoff.analyse_pass(passes[0], 0.0, None)],
        "axis_deg": None,
        "axis_detail": {"reason": "fewer than two distinct tilts"},
        "rotation_used_deg": 0.0, "dialled_rotation_deg": None,
        "per_degree": None, "step_detail": {}})
    assert "COULD NOT BE DERIVED" in text
    assert "UNAVAILABLE" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
