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
# File:  ec_bench_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-17
# Description:
#   Every estimator in ec_bench.py is checked against a signal whose
#   answer is known by construction.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-17:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""An estimator that has never been given a known answer is not a measurement.

bench_ec() will be run once, at a cell, on an instrument nobody has
characterised, and whatever it says will be believed -- there is nothing to
check it against there. So every number it produces is checked HERE against a
signal built to have that number: a first-order filter at a known corner, a
block average of samples that are independent by construction, a mains line of
known power, a block stream with known gaps.

The one that matters most is the corner frequency, because the speed limit is
proportional to it and the whole point of the exercise is to decide whether
2.5 mm/s is right or ten times too slow.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_bench as B                                          # noqa: E402

FS = 2000.0
N = 200000


def _lowpass(x, fc, fs):
    """One-pole IIR at `fc`, which is the shape an instrument's output filter
    has and the shape corner_frequency() is written to find."""
    a = math.exp(-2.0*math.pi*fc/fs)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = a*acc + (1.0 - a)*v
        y[i] = acc
    return y


def _rng(seed=7):
    return np.random.default_rng(seed)


def test_the_corner_frequency_is_the_filter_that_was_applied():
    """Within 20% from 25 Hz to 400 Hz, on two seeds. The speed limit is
    proportional to this number, so an error here is an error in mm/s of the
    same size -- and the decision it feeds has a factor-of-1.5 threshold."""
    for seed in (3, 11):
        for fc in (25.0, 50.0, 100.0, 200.0, 400.0):
            x = _lowpass(_rng(seed).normal(size=N), fc, FS)
            f, p = B.noise_spectrum(x, FS)
            got = B.corner_frequency(f, p)["corner_hz"]
            assert got is not None, (seed, fc)
            assert abs(got - fc)/fc < 0.20, \
                "corner %.1f Hz for a %.1f Hz filter (seed %d)" % (got, fc, seed)


def test_a_low_corner_is_found_at_all_even_though_it_is_blurred():
    """5 and 10 Hz are where the bin width and the smoothing are a sizeable
    fraction of the corner. The estimate is biased HIGH there, which is the
    optimistic direction, so what is pinned is that it is found and that the
    bias is bounded -- not that it is accurate."""
    for fc in (5.0, 10.0):
        x = _lowpass(_rng().normal(size=N), fc, FS)
        f, p = B.noise_spectrum(x, FS)
        c = B.corner_frequency(f, p)
        assert c["corner_hz"] is not None, fc
        assert c["corner_hz"] < 2.0*fc, c
        assert c["corner_low_hz"] <= c["corner_hz"], c


def test_a_single_quiet_bin_is_not_read_as_a_roll_off():
    """A periodogram bin scatters by its own size. Requiring a RUN of bins
    under the threshold is what separates a corner from a quiet bin -- without
    it a 400 Hz filter read as 240 Hz and a 100 Hz filter as 74."""
    x = _lowpass(_rng().normal(size=N), 400.0, FS)
    f, p = B.noise_spectrum(x, FS)
    p = p.copy()
    # Drop one bin on the flat part to the floor; it must change nothing.
    i = int(np.searchsorted(f, 50.0))
    before = B.corner_frequency(f, p)["corner_hz"]
    p[i] = p[i]*1e-6
    after = B.corner_frequency(f, p)["corner_hz"]
    assert before is not None and after is not None
    assert abs(after - before) < 1.0, (before, after)


def test_a_wideband_signal_reports_no_corner_rather_than_a_wrong_one():
    """Refusing is the right answer when the filter is above what the record
    can see: a number invented here becomes a speed limit at the cell."""
    f, p = B.noise_spectrum(_rng().normal(size=N), FS, nperseg=4096)
    c = B.corner_frequency(f, p)
    assert c["corner_hz"] is None, c
    assert "never falls" in c["why"], c["why"]
    lim = B.speed_limit_from_bandwidth(c["corner_hz"], 0.001)
    assert lim["v_max_mps"] is None
    assert "no corner frequency measured" in lim["why"]


def test_averaging_independent_samples_buys_sqrt_n():
    """The control. If this does not read 100%, the efficiency number means
    nothing when it reads 30%."""
    g = B.averaging_gain(_rng().normal(size=N), 100)
    assert abs(g["efficiency"] - 1.0) < 0.15, g


def test_averaging_correlated_samples_buys_almost_nothing():
    """THE MEASUREMENT THE SCAN NEEDS. A 2 Hz filter sampled at 2 kHz barely
    changes across a 100-sample (50 ms) block, so those 100 samples are one
    measurement read 100 times and the average is no quieter."""
    g = B.averaging_gain(_lowpass(_rng().normal(size=N), 2.0, FS), 100)
    assert g["efficiency"] < 0.3, g


def test_slow_drift_is_not_counted_as_sample_noise():
    """Detrending per block is what separates the two. A ramp is not noise the
    averaging can remove, and counting it as such would flatter the
    efficiency into saying the block is worth keeping when it is not."""
    x = np.linspace(0.0, 10.0, N) + 1e-6*_rng().normal(size=N)
    g = B.averaging_gain(x, 100)
    # The per-block detrended sd is dominated by the ramp within a block, which
    # is tiny; what must NOT happen is the whole-record ramp being counted.
    assert g["sigma_single"] < 0.01, g


def test_mains_is_found_and_quantified():
    t = np.arange(N)/FS
    r = _rng()
    hum = 0.02*np.sin(2.0*math.pi*60.0*t) + 0.002*r.normal(size=N)
    f, p = B.noise_spectrum(hum, FS, nperseg=4096)
    m = B.mains_content(f, p)
    assert m["fraction"] > 0.9, m
    clean = 0.002*r.normal(size=N)
    f2, p2 = B.noise_spectrum(clean, FS, nperseg=4096)
    assert B.mains_content(f2, p2)["fraction"] < 0.05


def test_mains_does_not_push_the_corner_out():
    """A 60 Hz line sitting on a 10 Hz filter's stopband is louder than the
    plateau, and a corner search that did not exclude it would report 60 Hz --
    a 6x error in the speed limit, in the dangerous direction."""
    t = np.arange(N)/FS
    x = _lowpass(_rng().normal(size=N), 10.0, FS)
    x = x/np.std(x) + 3.0*np.sin(2.0*math.pi*60.0*t)
    f, p = B.noise_spectrum(x, FS, nperseg=4096)
    got = B.corner_frequency(f, p, exclude_hz=(60.0, 120.0, 180.0))["corner_hz"]
    assert got is not None and got < 20.0, got


def test_block_timing_finds_dropped_blocks_and_not_jitter():
    clean = np.arange(400)/50.0
    t = B.block_timing(clean, nominal_hz=50.0)
    assert t["gaps"] == 0 and t["ok"], t
    assert abs(t["rate_hz"] - 50.0) < 0.5

    jittery = clean + 0.002*_rng().normal(size=clean.size)
    tj = B.block_timing(jittery, nominal_hz=50.0)
    assert tj["gaps"] == 0, "jitter is not a dropped block"
    assert tj["jitter_s"] > 0.0005

    dropped = np.delete(clean, [100, 250, 300])
    td = B.block_timing(dropped, nominal_hz=50.0)
    assert td["gaps"] == 3, td
    assert td["ok"] is False

    slow = np.arange(400)/40.0
    ts = B.block_timing(slow, nominal_hz=50.0)
    assert ts["ok"] is False and ts["gaps"] == 0, \
        "a uniformly slow card is a different fault from a dropping one"


def test_the_speed_limit_is_proportional_to_what_it_says_it_is():
    a = B.speed_limit_from_bandwidth(100.0, 0.001)["v_max_mps"]
    b = B.speed_limit_from_bandwidth(200.0, 0.001)["v_max_mps"]
    c = B.speed_limit_from_bandwidth(100.0, 0.002)["v_max_mps"]
    assert abs(b/a - 2.0) < 1e-9 and abs(c/a - 2.0) < 1e-9
    # and the number itself, against the arithmetic in the docstring
    assert abs(a - 100.0*0.001/B.SPEED_LIMIT_SAMPLES_PER_FEATURE) < 1e-12
    # A 10 Hz instrument allows about what the 2026-10-16 scan commanded, which
    # is the whole reason the measurement is worth making.
    assert abs(B.speed_limit_from_bandwidth(10.0, 0.001)["v_max_mps"]
               - 0.002) < 1e-9


def test_short_or_degenerate_records_do_not_raise():
    """bench_ec() runs at a cell with the probe on a part. Every estimator here
    returns a dict saying why, because a traceback there costs a setup."""
    for fn, args in ((B.block_timing, (np.zeros(0),)),
                     (B.block_timing, (np.array([1.0, 1.0, 1.0]),)),
                     (B.averaging_gain, (np.zeros(5), 100)),
                     (B.mains_content, (np.zeros(2), np.zeros(2))),
                     (B.corner_frequency, (np.zeros(2), np.zeros(2)))):
        out = fn(*args)
        assert isinstance(out, dict) and out.get("why"), (fn.__name__, out)
    f, p = B.noise_spectrum(np.zeros(3), FS)
    assert f.size == 0 and p.size == 0
    assert B.noise_spectrum(np.zeros(100), 0.0)[0].size == 0


def test_describe_bench_survives_a_report_with_nothing_in_it():
    text = B.describe_bench({})
    assert "DIGITIZER" in text and "WHAT THAT ALLOWS" in text


def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:                                 # noqa: BLE001
            failed.append(name)
            print("  FAIL  %s\n        %s" % (name, e))
        else:
            print("  ok    %s" % name)
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
