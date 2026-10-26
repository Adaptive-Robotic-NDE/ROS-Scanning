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
# File:  calibration_solve_cost_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that the calibration solve stays within its time budget and
#   reports progress while it runs.
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
"""The solve has to finish, and it has to say so while it is working.

A full keyence take is ~35 poses and ~1.5 M points, and the solve over all of
them runs for over ELEVEN MINUTES printing nothing, while the recorder keeps
writing /profiles at 2.8 MB/s. A silent eleven-minute solve is indistinguishable
from a hang, and an interrupted one throws away a complete, good pose set.

Two things are pinned here.

ONE: THE FIT DOES NOT NEED SIXTY COPIES OF ONE LINE. CAL_PROFILES_PER_POSE
profiles at a STATIC pose are the same 31 mm line measured over and over. The
grab keeps them -- they are what averages the per-profile noise down, and they
are in the bag regardless -- but handing them all to the solver costs time and
buys nothing. Measured over a full take:

    1/1    1 492 668 pts   rms 0.0658 mm   standoff -4.413 mm   37.9 s
    1/10     149 277 pts   rms 0.0658 mm            -4.415 mm    3.2 s
    1/30      49 765 pts   rms 0.0658 mm            -4.408 mm    2.8 s
    1/60      24 887 pts   rms 0.0657 mm            -4.403 mm    1.6 s
    1/120     12 451 pts   rms 0.0656 mm            -4.407 mm    1.0 s

0.012 mm of spread over a 120x range of thinning, for 38x the time.

TWO: A SOLVE THAT TAKES MINUTES MUST REPORT. And it must report the number the
result will report -- the rms over the KEPT points. The rms that sets the trim
threshold is over every point that landed, outliers included; showing that one
counts down to 1.5 mm and then announces 0.066 mm, which reads as a bug.

Runs on the host: the .dpi is exec'd with its `global` lines commented out, the
same way cell_capture_estimate_test.py does it. No ROS, no cell.

    python3 calibration_solve_cost_test.py
"""
import functools
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibration_solve as cs                                  # noqa: E402

FAILURES = []


def _helpers():
    """_cal_thin_for_solve and _cal_solve_progress out of calibration.dpi.

    Only the slice that defines them: the module as a whole wants ROS, a cell
    manifest and a capture session, none of which this is about.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                    for l in src.splitlines())
    start = src.index("CAL_PROFILES_PER_POSE = 60")
    consts = src[start:src.index("_CAL = {\"geom\"")]
    body = src[src.index("def _cal_thin_for_solve"):src.index("def _cal_geom()")]
    ns = {"np": np, "time": time}
    exec(compile(consts + body, "calibration.dpi", "exec"), ns)
    return ns


NS = _helpers()


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _checked(fn):
    """Make the checks a test recorded into a pytest pass or failure.

    check() RECORDS rather than raises, so one standalone run reports every
    false fact instead of stopping at the first. Under pytest that is a test
    which passes while printing FAIL -- this whole file was decorative under
    `pytest -q`, which is how the suite is run. The wrapper asserts on whatever
    the test added to FAILURES; the standalone driver calls fn.__wrapped__, so its
    behaviour is unchanged.
    """
    @functools.wraps(fn)
    def wrapper(*a, **k):
        mark = len(FAILURES)
        fn(*a, **k)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(map(str, failed)))
    return wrapper

@_checked
def test_thinning_samples_the_whole_grab_not_a_corner_of_it():
    """The two ways to get this wrong, both of which this helper did.

    A pose is 60 profiles of 800 columns laid end to end. Taking the FIRST 1500
    points takes two profiles. Taking every 30th ALIASES against the 800-column
    line and samples the same 25 columns over and over -- the count looks right
    and five sixths of the line is missing.
    """
    thin = NS["_cal_thin_for_solve"]
    cols, profiles = 800, 60
    x = np.tile(np.linspace(-0.0155, 0.0155, cols), profiles)
    # each profile is tagged in z, so we can see WHICH profiles were sampled
    tag = np.repeat(np.arange(profiles, dtype=float), cols)
    pose = np.column_stack([x, np.zeros(x.size), tag])
    out = thin([pose], budget=1500)[0]
    print("\none pose of %d profiles x %d columns, budget 1500" % (profiles, cols))
    check("thinned to the budget", out.shape[0] == 1500, "got %d" % out.shape[0])

    n_cols = len(np.unique(np.round(out[:, 0], 9)))
    n_prof = len(np.unique(out[:, 2]))
    print("    sampled %d distinct columns and %d of %d profiles"
          % (n_cols, n_prof, profiles))
    check("samples most of the 800 columns", n_cols > 600, "%d" % n_cols)
    check("samples every profile", n_prof == profiles, "%d" % n_prof)
    span = out[:, 0].max() - out[:, 0].min()
    check("and spans the line", span > 0.98*(x.max() - x.min()),
          "%.4f of %.4f m" % (span, x.max() - x.min()))

    # what a stride would have done, which is what this used to do
    strided = pose[::pose.shape[0]//1500]
    n_cols_s = len(np.unique(np.round(strided[:, 0], 9)))
    print("    a stride would have sampled %d distinct columns" % n_cols_s)
    check("...where a stride aliases badly", n_cols_s < n_cols/4,
          "stride got %d, sample got %d" % (n_cols_s, n_cols))

    # and what taking the first N would have done
    head = pose[:1500]
    check("...and the first N sees almost no profiles",
          len(np.unique(head[:, 2])) <= 2, "%d" % len(np.unique(head[:, 2])))


@_checked
def test_thinning_is_deterministic():
    thin = NS["_cal_thin_for_solve"]
    pose = np.random.default_rng(1).normal(size=(50000, 3))
    print("\nthe same input gives the same points every time")
    a, b = thin([pose], budget=1500)[0], thin([pose], budget=1500)[0]
    check("two calls agree exactly", np.array_equal(a, b))


@_checked
def test_thinning_is_optional_and_never_grows_a_pose():
    thin = NS["_cal_thin_for_solve"]
    P = [np.zeros((10, 3)), np.zeros((100000, 3))]
    print("\nthe budget is a ceiling, not a target")
    same = thin(P, budget=0)
    check("a falsy budget disables thinning",
          [p.shape[0] for p in same] == [10, 100000],
          str([p.shape[0] for p in same]))
    dflt = thin(P)
    check("budget None means the configured default",
          dflt[1].shape[0] == NS["CAL_SOLVE_POINTS_PER_POSE"],
          "got %d" % dflt[1].shape[0])
    out = thin(P, budget=1500)
    check("a small pose is untouched", out[0].shape[0] == 10,
          "got %d" % out[0].shape[0])
    check("a big pose is cut", out[1].shape[0] < 3000, "got %d" % out[1].shape[0])
    check("the default budget is sane",
          500 <= NS["CAL_SOLVE_POINTS_PER_POSE"] <= 20000,
          "got %r" % NS["CAL_SOLVE_POINTS_PER_POSE"])


def _synthetic():
    """A stepped-artifact run whose points are deliberately repeated, the way
    sixty profiles at a static pose repeat."""
    import calibration_test as ct
    G, P, _X0, W0 = ct._aimed_run(ct.SPREAD_AIMS, (-0.015, 0.0, 0.015),
                                  solve=False)
    return G, [np.repeat(p, 8, axis=0) for p in P], W0, ct


@_checked
def test_thinning_does_not_move_the_answer():
    G, P, W0, ct = _synthetic()
    thin = NS["_cal_thin_for_solve"]
    print("\nthe same fit, with and without the repeats")
    t0 = time.time()
    full = cs.solve_surface_fit(G, P, ct.STEPPED, ct.KEY_TRUE.copy(), W0)
    t_full = time.time() - t0
    Pt = thin(P, budget=200)
    t0 = time.time()
    cut = cs.solve_surface_fit(G, Pt, ct.STEPPED, ct.KEY_TRUE.copy(), W0)
    t_cut = time.time() - t0
    dz = abs(full.X[2, 3] - cut.X[2, 3])*1e3
    n_full = sum(p.shape[0] for p in P)
    n_cut = sum(p.shape[0] for p in Pt)
    print("    %d pts %.1f s  ->  %d pts %.1f s" % (n_full, t_full, n_cut, t_cut))
    check("the standoff is unchanged", dz < 0.05, "moved %.4f mm" % dz)
    check("the residual is unchanged", abs(full.rms_mm - cut.rms_mm) < 0.02,
          "%.4f vs %.4f mm" % (full.rms_mm, cut.rms_mm))
    check("and it used far fewer points", n_cut < n_full/4,
          "%d of %d" % (n_cut, n_full))


@_checked
def test_the_solve_reports_progress_and_reports_the_right_number():
    G, P, W0, ct = _synthetic()
    seen = []
    print("\nprogress, and whether it agrees with the result")
    res = cs.solve_surface_fit(G, NS["_cal_thin_for_solve"](P, budget=200),
                               ct.STEPPED, ct.KEY_TRUE.copy(), W0,
                               progress=seen.append)
    check("it reported at all", len(seen) >= 3, "%d callbacks" % len(seen))
    check("iterations count up from 1",
          [s["iteration"] for s in seen] == list(range(1, len(seen)+1)),
          str([s["iteration"] for s in seen])[:70])
    check("every report carries max_iter",
          all(s["max_iter"] >= s["iteration"] for s in seen))
    last = seen[-1]["rms_mm"]
    check("the LAST report is the rms the result reports",
          abs(last - res.rms_mm) < 1e-6,
          "progress said %.6f, result says %.6f" % (last, res.rms_mm))
    # ... which is the whole point: an operator watching it converge to 1.5 mm
    # and then being told 0.07 mm would reasonably think something was wrong.
    check("it converges downwards", seen[-1]["rms_mm"] <= seen[0]["rms_mm"],
          "%.4f -> %.4f" % (seen[0]["rms_mm"], seen[-1]["rms_mm"]))
    check("point counts are reported", all(s["points"] > 0 for s in seen))


@_checked
def test_the_progress_callback_is_optional():
    G, P, W0, ct = _synthetic()
    print("\nnothing changes when nobody is watching")
    a = cs.solve_surface_fit(G, P, ct.STEPPED, ct.KEY_TRUE.copy(), W0)
    b = cs.solve_surface_fit(G, P, ct.STEPPED, ct.KEY_TRUE.copy(), W0,
                             progress=lambda info: None)
    check("same answer with and without a callback",
          float(np.max(np.abs(a.X - b.X))) < 1e-12,
          "%.3g" % float(np.max(np.abs(a.X - b.X))))


def main():
    for fn in (test_thinning_samples_the_whole_grab_not_a_corner_of_it,
               test_thinning_is_deterministic,
               test_thinning_is_optional_and_never_grows_a_pose,
               test_thinning_does_not_move_the_answer,
               test_the_solve_reports_progress_and_reports_the_right_number,
               test_the_progress_callback_is_optional):
        getattr(fn, "__wrapped__", fn)()
    print("\n%s" % ("ALL OK" if not FAILURES else "%d FAILED: %s"
                    % (len(FAILURES), ", ".join(FAILURES))))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
