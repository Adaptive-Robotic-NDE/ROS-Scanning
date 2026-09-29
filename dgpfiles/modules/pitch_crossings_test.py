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
# File:  pitch_crossings_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that PitchCrossingDetector is exact on the pitch, immune to
#   noise, and honest about direction.
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
"""Check PitchCrossingDetector: exact on the pitch, immune to noise, honest
about direction.

The properties a trigger index has to have for "trigger k is column k" to be
true, plus the one that makes chord progress necessary instead of a sum of
arc lengths: TF noise must not manufacture crossings. Runs on the host; no ROS,
no database.
"""
import ast
import functools
import os
import sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch_crossings.dpi")
tree = ast.parse(open(SRC).read())
cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PitchCrossingDetector"][0]
ns = {"np": np}
exec(compile(ast.Module(body=[cls], type_ignores=[]), SRC, "exec"), ns)
PitchCrossingDetector = ns["PitchCrossingDetector"]

fails = []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print("  %-62s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


def run(det, path):
    out = []
    for p in path:
        out += det.feed(p)
    return out


@_checked
def test_straight_line_unevenly_sampled():
    print("=== straight line, unevenly sampled ===")
    rng = np.random.RandomState(7)
    xs = np.cumsum(rng.uniform(0.001, 0.02, size=400))
    xs = xs/xs[-1]
    det = PitchCrossingDetector(0.05)
    fired = run(det, [(x, 0.0) for x in xs])
    pos = np.array([f[0][0] for f in fired])
    check("origin then one crossing per pitch", len(fired), 1 + int((xs[-1]-xs[0])/0.05 + 1e-9))
    check("crossing k at exactly k*pitch", float(np.abs(pos - pos[0] - np.arange(len(pos))*0.05).max()), 0.0, 1e-12)
    check("s_m is k*pitch", float(np.abs(np.array([f[1] for f in fired]) - np.arange(len(fired))*0.05).max()), 0.0, 1e-12)
    check("travelled equals the line", det.travelled_m, xs[-1]-xs[0], 1e-12)
    check("raw travel equals the line too (no wobble)", det.raw_travelled_m, xs[-1]-xs[0], 1e-12)

    print()


@_checked
def test_an_exact_multiple_of_the_pitch_fires_the_last_crossing():
    print("=== an exact multiple of the pitch fires the last crossing ===")
    det = PitchCrossingDetector(0.01)
    fired = run(det, [(x, 0.0) for x in np.linspace(0.0, 0.5, 501)])   # 0.5 = 50 pitches
    check("51 = origin + 50", len(fired), 51)

    print()


@_checked
def test_tf_noise_does_not_manufacture_crossings():
    print("=== TF noise must not manufacture crossings and shift the c-scan ===")
    # 60 mm at 12 mm/s polled at 200 Hz is 0.06 mm per poll. Add 0.03 mm rms of
    # lateral noise per poll -- half the true step -- which is what a joint-space
    # controller and a TF lookup between them can do.
    rng = np.random.RandomState(3)
    n = 1000
    x = np.linspace(0.0, 0.060, n)
    noise = rng.normal(0.0, 0.03e-3, size=(n, 3))
    noise[0] = 0.0
    path = np.column_stack([x, np.zeros(n), np.zeros(n)]) + noise
    PITCH = 0.060/49.5                  # 50 columns with the half-pitch end margin
    det = PitchCrossingDetector(PITCH)
    fired = run(det, path)
    print("    raw arc-sum saw %.1f mm for a 60.0 mm pass (%.0f%%); chord progress %.1f mm"
          % (1e3*det.raw_travelled_m, 100.0*det.raw_travelled_m/0.060, 1e3*det.travelled_m))
    check("the arc-sum IS inflated by this noise", det.raw_travelled_m > 1.02*0.060, True)
    check("chord progress is not (within 0.5%)", det.travelled_m, 0.060, 0.005*0.060)
    check("exactly 49 crossings + origin, not 50 or 51", len(fired), 50)
    xk = np.array([f[0][0] for f in fired])
    check("crossing k within a tenth of a pitch of k*pitch along x",
          float(np.abs(xk - np.arange(len(xk))*PITCH).max()) < 0.1*PITCH, True)

    print()
    print("=== a stationary probe with noise fires nothing ===")
    det = PitchCrossingDetector(0.0012)
    fired = run(det, np.array([0.1, 0.2, 0.3]) + rng.normal(0.0, 0.05e-3, size=(500, 3)))
    check("only the origin", len(fired), 1)
    check("raw arc-sum would have fired anyway", det.raw_travelled_m/0.0012 > 3, True)

    print()


@_checked
def test_a_sub_pitch_backstep_is_absorbed():
    print("=== a backstep shorter than the pitch is absorbed, not counted ===")
    # Progress is displacement from the last crossing. Wobble back and forth
    # under a pitch fires nothing (the arc-sum counted every reversal as forward
    # travel); a full pitch of reversal IS a pitch of displacement and fires,
    # which is what an encoder does too.
    det = PitchCrossingDetector(0.01)
    run(det, [(x, 0.0) for x in np.linspace(0.0, 0.025, 26)])          # origin, 0.01, 0.02
    fired = run(det, [(x, 0.0) for x in np.linspace(0.025, 0.012, 14)])   # back to 2 mm past the last crossing
    check("sub-pitch backstep fires nothing", len(fired), 0)
    fired = run(det, [(x, 0.0) for x in np.linspace(0.012, 0.025, 14)])   # forward again to the same point
    check("and coming back fires nothing either", len(fired), 0)
    check("raw arc-sum counted the excursion", det.raw_travelled_m, 0.025 + 0.026, 1e-12)
    check("progress did not", det.travelled_m, 0.025, 1e-12)
    fired = run(det, [(x, 0.0) for x in np.linspace(0.025, 0.0, 26)])
    check("a full pitch of reversal is a crossing", len(fired), 2)

    print()


@_checked
def test_a_long_step_fires_every_crossing():
    print("=== a step longer than the pitch fires every crossing, evenly ===")
    det = PitchCrossingDetector(0.005)
    fired = run(det, [(0.0, 0.0), (0.1, 0.0)])
    check("21 = origin + 20", len(fired), 21)
    check("evenly spaced across the jump",
          float(np.abs(np.diff([f[0][0] for f in fired]) - 0.005).max()), 0.0, 1e-12)
    check("frac interpolates within the step", [round(f[2], 3) for f in fired[1:4]], [0.05, 0.1, 0.15])

    print()


@_checked
def test_bursts_are_capped_and_counted():
    print("=== bursts are capped and counted, not lost silently ===")
    det = PitchCrossingDetector(0.001, max_burst=16)
    fired = run(det, [(0.0, 0.0), (0.1, 0.0)])
    check("capped at max_burst", len(fired), 17)
    check("dropped count", det.burst_dropped, 100-16)

    print()


@_checked
def test_diagonal_and_3d_travel_are_distance():
    print("=== diagonal and 3D travel are distance, not dx ===")
    det = PitchCrossingDetector(0.01)
    fired = run(det, [(0.0, 0.0, 0.0), (0.3, 0.4, 0.0)])
    check("0.5 m diagonal", len(fired), 51)
    det = PitchCrossingDetector(0.01)
    s = np.linspace(0.0, 0.5, 500)
    fired = run(det, np.column_stack([s*np.cos(0.35), 0*s, s*np.sin(0.35)]))
    check("0.5 m climbing pass", len(fired), 51)

    print()


@_checked
def test_reset_starts_a_new_phase_without_losing_progress():
    print("=== reset(): a new phase, but travelled_m never goes backwards ===")
    det = PitchCrossingDetector(0.01)
    run(det, [(x, 0.0) for x in np.linspace(0.0, 0.037, 38)])   # 3 crossings + 7 mm of chord
    before = det.travelled_m
    det.reset()
    check("progress folded in, not discarded", det.travelled_m, before, 1e-12)
    fired = run(det, [(0.037, 0.0)] + [(0.037 + x, 0.0) for x in np.linspace(0.0, 0.02, 21)])
    check("new origin, then crossings from it", [round(f[1], 3) for f in fired], [0.0, 0.01, 0.02])
    check("total progress continuous", det.travelled_m, before + 0.02, 1e-12)

    print()


@_checked
def test_a_gentle_curve_keeps_every_crossing():
    print("=== a gentle curve: chord is arc to (pitch/R)^2/24 ===")
    # Each chord consumes arc pitch*(1 + (pitch/R)^2/24): 2.6e-5 of a pitch here,
    # so 40 crossings fall 1e-3 of a pitch short of 40 pitches of arc. A pass
    # sized to an exact multiple sits on that boundary, which is why the raster
    # leaves an end margin; half a pitch beyond, every crossing is there.
    R = 0.20
    th = np.linspace(0.0, 1.0125, 2000)             # 40.5 pitches of arc
    det = PitchCrossingDetector(0.005)
    fired = run(det, np.column_stack([R*np.sin(th), R*(1-np.cos(th))]))
    check("all 40 crossings on a 0.2 m arc of radius 0.2 m", len(fired), 41)

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("all crossing checks passed")


_SECTIONS = ["test_straight_line_unevenly_sampled", "test_an_exact_multiple_of_the_pitch_fires_the_last_crossing", "test_tf_noise_does_not_manufacture_crossings", "test_a_sub_pitch_backstep_is_absorbed", "test_a_long_step_fires_every_crossing", "test_bursts_are_capped_and_counted", "test_diagonal_and_3d_travel_are_distance", "test_reset_starts_a_new_phase_without_losing_progress", "test_a_gentle_curve_keeps_every_crossing"]


if __name__ == "__main__":
    sys.exit(_main())
