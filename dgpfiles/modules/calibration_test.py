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
# File:  calibration_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests calibration_solve.py against synthetic transforms it is not
#   given.
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
"""calibration_solve.py against transforms it is not allowed to know.

    python3 calibration_test.py          (or pytest)

Runs on the HOST: numpy only, no cv2, no rclpy, no robot. That is the whole
reason the arithmetic lives in calibration_solve.py instead of in
calibration.dpi.

THE TEST THAT MATTERS IS test_under_constrained_is_detected. Everything else
here checks that a solver recovers a transform it was given noiseless data
about, which is table stakes. That one checks the failure this whole procedure
exists to prevent: a pose set that CANNOT determine the answer, fitted anyway,
producing a small residual and a wrong transform. It asserts both halves --
that the residual really is small (so nobody can claim the residual would have
caught it) and that the error really is large -- and then that the solver says
so out loud rather than handing back the number.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibration_solve as cs                                  # noqa: E402


# ------------------------------------------------------------------------------
# Building a cell that does not exist
# ------------------------------------------------------------------------------

def _T(xyz, rotvec):
    return cs.rt_to_T(cs.so3_exp(np.asarray(rotvec, float)), np.asarray(xyz, float))


#: The transform under test: a camera about 120 mm off the tool tip and turned
#: well away from it. Deliberately not small, not axis-aligned and not round,
#: so a solver that quietly returns its initial guess cannot pass.
X_TRUE = _T([0.0731, -0.0412, 0.1563], [0.31, -0.22, 1.17])

#: Where the artifact is bolted to the bench.
W_TRUE = _T([0.85, -0.20, 0.94], [0.02, -0.01, 0.4])


def _robot_poses(tilts, rolls, centre=(0.85, -0.20, 1.25), rng=None):
    """Flange poses that look roughly down at the bench from varied attitudes.

    `tilts` is [(tx_deg, ty_deg)] and `rolls` [deg] about the tool axis; the
    product of the two is flown, which is how calibration.dpi builds its set.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for tx, ty in tilts:
        for roll in rolls:
            R = (cs.so3_exp([math.radians(tx), 0.0, 0.0])
                 .dot(cs.so3_exp([0.0, math.radians(ty), 0.0]))
                 .dot(cs.so3_exp([0.0, 0.0, math.radians(roll)])))
            xyz = np.asarray(centre, float) + rng.uniform(-0.05, 0.05, 3)
            out.append(cs.rt_to_T(R, xyz))
    return out


def _observations(G, X=X_TRUE, W=W_TRUE, t_noise=0.0, r_noise=0.0, rng=None):
    """What the sensor would report at those poses: C = X^-1 G^-1 W."""
    rng = np.random.default_rng(1) if rng is None else rng
    out = []
    for g in G:
        C = cs.T_inv(X).dot(cs.T_inv(g)).dot(W)
        if t_noise or r_noise:
            C = C.dot(cs.se3_exp(np.concatenate([
                rng.normal(0.0, t_noise, 3), rng.normal(0.0, r_noise, 3)])))
        out.append(C)
    return out


def _error(X_est, X_true=X_TRUE):
    """(mm, mrad) between an estimate and the truth."""
    dt, dr = cs.pose_delta(X_true, X_est)
    return dt*1e3, dr*1e3


#: A pose set with real rotation diversity, inside this cell's measured cable
#: envelope: tool tilts no further than +/-40 deg about x and +/-20 about y,
#: which is all ex-ur10e-1's cables allow (cable_envelope.dpi).
GOOD_TILTS = [(0, 0), (40, 0), (-40, 0), (20, 0), (-20, 0),
              (0, 20), (0, -20), (0, 10), (0, -10),
              (30, 15), (-30, -15), (25, -12), (-25, 12)]
GOOD_ROLLS = [0, 60, 120, 180]


# ------------------------------------------------------------------------------
# SE(3)
# ------------------------------------------------------------------------------

def test_se3_round_trips():
    rng = np.random.default_rng(7)
    for _ in range(50):
        xi = rng.normal(0.0, 1.0, 6)
        # Keep the rotation under half a turn: beyond pi the log returns the
        # SHORTER way round, which is the same rotation and a different vector.
        if np.linalg.norm(xi[3:]) > 3.0:
            xi[3:] *= 3.0/np.linalg.norm(xi[3:])
        assert np.allclose(cs.se3_log(cs.se3_exp(xi)), xi, atol=1e-10)
    for _ in range(50):
        w = rng.normal(0.0, 1.5, 3)
        if np.linalg.norm(w) > 3.0:
            w *= 3.0/np.linalg.norm(w)
        R = cs.so3_exp(w)
        assert np.allclose(R.dot(R.T), np.eye(3), atol=1e-12)
        assert np.allclose(cs.so3_exp(cs.so3_log(R)), R, atol=1e-10)
        assert np.allclose(cs.quat_to_R(cs.R_to_quat(R)), R, atol=1e-10)
        assert np.allclose(cs.rpy_to_R(*cs.R_to_rpy(R)), R, atol=1e-10)


def test_so3_log_near_pi():
    """A half turn is where the antisymmetric formula dies, and a calibration
    that turns the wrist 180 degrees reaches it."""
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0, 0, 1.0]), np.array([1.0, 1.0, 0.3])):
        axis = axis/np.linalg.norm(axis)
        for angle in (math.pi - 1e-8, math.pi, math.pi - 1e-4):
            R = cs.so3_exp(axis*angle)
            w = cs.so3_log(R)
            assert np.allclose(cs.so3_exp(w), R, atol=1e-6), (axis, angle)


# ------------------------------------------------------------------------------
# Hand-eye, the happy path
# ------------------------------------------------------------------------------

def test_recovers_exactly_from_noiseless_observations():
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    C = _observations(G)
    res = cs.solve_handeye(G, C, name="azure")
    mm, mrad = _error(res.X)
    assert mm < 1e-6, res.report()
    assert mrad < 1e-6, res.report()
    assert res.residual_mm < 1e-6
    assert res.adequate
    assert res.conditioning["verdict"] == "good", res.conditioning


def test_recovers_the_target_pose_too():
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    res = cs.solve_handeye(G, _observations(G))
    dt, dr = cs.pose_delta(W_TRUE, res.W)
    assert dt*1e3 < 1e-6 and dr*1e3 < 1e-6


def test_closed_form_alone_is_already_close():
    """refine=False exercises Park-Martin and the translation least squares on
    their own, so a bug in one cannot be hidden by the Gauss-Newton mopping up
    after it."""
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    res = cs.solve_handeye(G, _observations(G), refine=False)
    mm, mrad = _error(res.X)
    assert mm < 1e-6 and mrad < 1e-6


def test_survives_realistic_noise():
    """0.4 mm and 3 mrad per observation is about what a checkerboard at 0.5 m
    gives. The answer should land inside a millimetre -- two orders below the
    13.6 mm this procedure exists to catch."""
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    C = _observations(G, t_noise=0.0004, r_noise=0.003)
    res = cs.solve_handeye(G, C, name="azure")
    mm, mrad = _error(res.X)
    assert mm < 1.0, "%.3f mm\n%s" % (mm, res.report())
    assert mrad < 6.0, "%.3f mrad\n%s" % (mrad, res.report())
    assert res.adequate


def test_reported_sigma_is_not_a_fantasy():
    """The 1 sigma the fit reports has to bracket the error the fit actually
    made. A covariance that is optimistic is worse than none: it is what turns
    'the solve looked fine' into a scan flown 13.6 mm low."""
    inside = 0
    trials = 12
    for seed in range(trials):
        rng = np.random.default_rng(100 + seed)
        G = _robot_poses(GOOD_TILTS, GOOD_ROLLS, rng=rng)
        C = _observations(G, t_noise=0.0004, r_noise=0.003, rng=rng)
        res = cs.solve_handeye(G, C)
        err = np.abs(cs.T_inv(X_TRUE).dot(res.X)[:3, 3])*1e3
        if np.all(err < 3.0*np.asarray(res.sigma[:3])):
            inside += 1
    assert inside >= trials - 2, "%d/%d errors inside 3 sigma" % (inside, trials)


def test_outliers_are_named_not_averaged():
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    C = _observations(G)
    bad = 5
    C[bad] = C[bad].dot(cs.se3_exp([0.02, -0.015, 0.008, 0.0, 0.0, 0.0]))
    res = cs.solve_handeye(G, C)
    names = [p["pose"] for p in res.outliers()]
    assert bad in names, res.report()
    clean = cs.solve_handeye(G, C, drop=(bad,))
    mm, _mrad = _error(clean.X)
    assert mm < 1e-6, clean.report()


# ------------------------------------------------------------------------------
# THE ONE THAT MATTERS
# ------------------------------------------------------------------------------

def test_under_constrained_is_detected():
    """All poses at nearly the same orientation: translated, barely turned.

    This is the pose set an operator produces by jogging the camera around
    above a target without ever tilting it, and it is the one that has to fail
    loudly. The assertions below are in the order the argument runs:

      1. the fit's residual is SMALL -- so nobody may say a residual check
         would have caught it,
      2. the recovered transform is nevertheless WRONG by centimetres,
      3. the solver says the pose set is inadequate, and refuses when asked to.
    """
    rng = np.random.default_rng(3)
    G = []
    for _ in range(20):
        R = cs.so3_exp(rng.normal(0.0, math.radians(0.5), 3))    # ~half a degree
        G.append(cs.rt_to_T(R, [0.85, -0.20, 1.25] + rng.uniform(-0.08, 0.08, 3)))
    C = _observations(G, t_noise=0.0004, r_noise=0.003, rng=rng)

    res = cs.solve_handeye(G, C, name="azure")

    # 1. It fits. Beautifully.
    assert res.residual_mm < 1.0, (
        "expected a small residual from an under-constrained fit, got %.3f mm"
        % res.residual_mm)

    # 2. And it is wrong.
    mm, _mrad = _error(res.X)
    assert mm > 10.0, ("expected this pose set to produce a badly wrong answer, "
                       "got %.3f mm -- if this ever fails the test has stopped "
                       "testing anything" % mm)

    # 3. And it says so, in all three of the ways a caller might ask.
    assert not res.adequate, res.report()
    assert res.conditioning["verdict"] == "inadequate", res.conditioning
    assert res.conditioning["problems"], "inadequate with nothing to say about it"
    text = res.report()
    assert "DOES NOT DETERMINE" in text, text
    try:
        cs.solve_handeye(G, C, require_adequate=True)
    except cs.PoseSetInadequate:
        pass
    else:
        raise AssertionError("require_adequate returned an answer anyway")


def test_single_rotation_axis_is_detected():
    """Rotations that all share one axis. The total rotation is large -- 180
    degrees of wrist roll -- so a check that only looks at how far the arm
    turned passes it, and the offset along that axis is still unmeasured."""
    G = []
    rng = np.random.default_rng(5)
    for roll in np.linspace(0.0, 180.0, 16):
        R = cs.so3_exp([0.0, 0.0, math.radians(roll)])
        G.append(cs.rt_to_T(R, [0.85, -0.20, 1.25] + rng.uniform(-0.03, 0.03, 3)))
    C = _observations(G, t_noise=0.0004, r_noise=0.003, rng=rng)
    res = cs.solve_handeye(G, C)
    cond = res.conditioning
    assert cond["max_rotation_deg"] > 100.0, cond     # it really did turn a lot
    assert min(cond["rot_axis_rms_deg"]) < 1.0, cond  # about one axis only
    assert cond["verdict"] == "inadequate", res.report()
    assert not res.adequate


def test_a_proposed_pose_set_can_be_judged_before_flying_it():
    """pose_set_conditioning needs only the robot poses, so a pose set can be
    rejected at the prompt instead of after an hour at the cell."""
    good = cs.pose_set_conditioning(_robot_poses(GOOD_TILTS, GOOD_ROLLS))
    assert good["verdict"] == "good", good
    assert good["amplification"] < cs.AMPLIFICATION_GOOD

    flat = cs.pose_set_conditioning(_robot_poses([(0, 0)], [0, 3, 6, 9]))
    assert flat["verdict"] == "inadequate", flat
    assert flat["problems"]


def test_too_few_poses_refuses():
    G = _robot_poses([(0, 0), (30, 0)], [0])
    try:
        cs.solve_handeye(G, _observations(G))
    except cs.PoseSetInadequate:
        return
    raise AssertionError("two poses should not produce a hand-eye answer")


# ------------------------------------------------------------------------------
# The checkerboard's half-turn ambiguity
# ------------------------------------------------------------------------------

def test_inconsistent_board_flips_are_found_and_undone():
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    C = _observations(G)
    F = cs.rt_to_T(cs.so3_exp([0.0, 0.0, math.pi]), np.zeros(3))
    flipped_at = [2, 7, 11, 19]
    C_bad = [(c.dot(F) if i in flipped_at else c) for i, c in enumerate(C)]

    wrong = cs.solve_handeye(G, C_bad)
    assert _error(wrong.X)[0] > 1.0, "a mixed flip should wreck the answer"

    fixed = cs.solve_handeye(G, C_bad, fix_flips=True)
    mm, mrad = _error(fixed.X)
    assert mm < 1e-6 and mrad < 1e-6, fixed.report()
    assert sorted(fixed.flips) == flipped_at, fixed.flips


def test_a_consistent_flip_is_harmless():
    """Every detection turned the same way round is absorbed into the target
    pose and changes nothing about the sensor transform, which is why
    canonicalise_flips only has to make them AGREE, not be right."""
    G = _robot_poses(GOOD_TILTS, GOOD_ROLLS)
    F = cs.rt_to_T(cs.so3_exp([0.0, 0.0, math.pi]), np.zeros(3))
    res = cs.solve_handeye(G, [c.dot(F) for c in _observations(G)])
    assert _error(res.X)[0] < 1e-6, res.report()


# ------------------------------------------------------------------------------
# Geometry the sensors are fitted to
# ------------------------------------------------------------------------------

def test_fit_plane():
    rng = np.random.default_rng(11)
    n_true = np.array([0.02, -0.03, 1.0])
    n_true = n_true/np.linalg.norm(n_true)
    P = rng.uniform(-0.1, 0.1, (400, 3))
    P[:, 2] = (0.94 - (P[:, 0]*n_true[0] + P[:, 1]*n_true[1])/n_true[2]
               + rng.normal(0.0, 1e-4, 400))
    n, c, rms, k = cs.fit_plane(P)
    assert k == 400
    assert np.degrees(math.acos(abs(float(n.dot(n_true))))) < 0.2
    assert rms < 3e-4
    assert n[2] > 0.0                 # always up, so two fits can be compared


def test_fit_sphere_free_radius_reveals_a_depth_scale_error():
    """A ball of known diameter measured by a sensor with a 3% range error fits
    a sphere 3% too big. That is the diagnostic that separates 'the camera is
    in the wrong place' from 'the camera measures distance wrongly', and only
    the free-radius fit shows it."""
    rng = np.random.default_rng(13)
    R_true, c_true = 0.0127, np.array([0.9, -0.2, 0.96])
    eye = np.array([0.9, -0.2, 1.4])                  # looking straight down
    d = rng.normal(0.0, 1.0, (600, 3))
    d = d/np.linalg.norm(d, axis=1)[:, None]
    d = d[d[:, 2] > 0.3]                              # the visible cap only
    P = c_true + R_true*d
    c, r, rms, _n = cs.fit_sphere(P)
    assert abs(r - R_true) < 1e-6 and np.allclose(c, c_true, atol=1e-6)
    assert rms < 1e-9

    scale = 1.03
    P_scaled = eye + (P - eye)*scale
    _c2, r2, _rms2, _n2 = cs.fit_sphere(P_scaled)
    assert abs(r2/R_true - scale) < 0.02, r2/R_true


def test_plane_agreement_reproduces_the_failure_it_exists_to_catch():
    """The numbers from an uncalibrated cell: the Keyence puts the plate top at
    0.9409 and the Azure at 0.9273, so a raster planned over the Azure's mesh
    flies 13.6 mm low and a third of a 10 inch plate comes back as garbage."""
    rng = np.random.default_rng(17)
    xy = rng.uniform(-0.12, 0.12, (500, 2))
    key = np.column_stack([xy, np.full(500, 0.9409) + rng.normal(0, 1.0e-4, 500)])
    azu = np.column_stack([xy, np.full(500, 0.9273) + rng.normal(0, 1.5e-3, 500)])
    a = cs.plane_agreement(key, azu, name_a="keyence", name_b="azure")
    assert abs(a["dz_mm"] + 13.6) < 0.3, a
    assert a["tilt_mrad"] < 10.0, a
    assert a["rms_a_mm"] < a["rms_b_mm"]


# ------------------------------------------------------------------------------
# Is it a measurement at all?
# ------------------------------------------------------------------------------
# A degenerate Keyence grab makes verify_calibration() print a confident FAIL
# with a specific number:
#
#     keyence  z = 0.6632 m   0.000 mm rms flat, 48000 points from 60 profiles
#     AZURE IS +244.33 mm AND 1571.91 mrad OFF THE KEYENCE
#
# which reads as "the Azure is badly out" and says nothing about the Azure.
# These tests pin BOTH directions, because a check that refuses a constant by
# refusing everything is not a check: an excellent real profile (7 um RMS, the
# best this cell has measured) has to pass.

#: The head's measured within-profile noise on this cell: 7 um on a sprayed
#: flat, 24 um on bare metal. 800 columns over the 31 mm line, 0.5 um quantum.
_KEY_NOISE_SPRAYED = 7.0e-6
_KEY_NOISE_BARE = 24.0e-6


def _profile_grab(z0=0.0, noise=_KEY_NOISE_SPRAYED, tilt_deg=0.0, n_profiles=60,
                  n_cols=800, quantum=0.5e-6, seed=3):
    """What one static grab off an LJ-V7080 looks like, in its OWN frame.

    x runs across the 31 mm laser line, y is identically 0 (which is the whole
    reason a plane must not be fitted to one grab), and z is the device's raw
    reading -- zero at the reference distance, POSITIVE TOWARDS THE HEAD.
    """
    rng = np.random.default_rng(seed)
    x = np.tile(np.linspace(-0.0155, 0.0155, n_cols), n_profiles)
    z = z0 + math.tan(math.radians(tilt_deg))*x
    if noise:
        z = z + rng.normal(0.0, noise, x.size)
    if quantum:
        z = np.round(z/quantum)*quantum
    return np.column_stack([x, np.zeros_like(x), z])


def test_profile_a_real_noisy_flat_passes():
    """THE OVER-REJECTION HALF. 7 um RMS is the best this cell has ever measured
    within a profile; if the check cannot tell that from a constant it is not a
    check, it is a way of refusing to calibrate."""
    for noise in (_KEY_NOISE_SPRAYED, _KEY_NOISE_BARE):
        for tilt in (0.0, 25.0):
            r = cs.check_profile(_profile_grab(noise=noise, tilt_deg=tilt),
                                 n_profiles=60)
            assert r["ok"], (noise, tilt, r["problems"])
            # and the scatter it reports is the noise it was given, not the
            # tilt: a flat at 25 degrees is a straight line across the profile
            # and its slope is not noise.
            assert abs(r["rms"] - noise) < 0.2*noise, (tilt, r["rms"])
    # WITH HOW MUCH ROOM. The threshold is 1 um; the best real measurement this
    # cell has is 7 um, so a good profile clears it by a factor of seven. That
    # margin is the whole argument for the threshold, so it is asserted.
    r = cs.check_profile(_profile_grab(noise=_KEY_NOISE_SPRAYED), n_profiles=60)
    assert r["rms"] > 6.0*cs.KEYENCE_MIN_PROFILE_RMS, r["rms"]
    # THE DELIBERATE COST, stated rather than discovered: a profile scattering
    # at the head's bare 0.5 um repeatability spec would be refused. No real
    # surface does that -- 0.5 um is a static single-point figure, and across a
    # 31 mm line on a sprayed flat this cell measures fourteen times it.
    assert not cs.check_profile(_profile_grab(noise=0.5e-6))["ok"]


def test_profile_a_constant_is_refused():
    """48 000 points at exactly one height -- the shape a grab takes when there
    is nothing inside the head's measuring window."""
    r = cs.check_profile(_profile_grab(noise=0.0, quantum=0.0), n_profiles=60)
    assert not r["ok"]
    assert any(p.startswith("SCATTER") for p in r["problems"]), r["problems"]
    assert any(p.startswith("REPEATED VALUE") for p in r["problems"]), r["problems"]
    assert r["rms"] == 0.0
    # The message has to be about the standoff, not about a calibration.
    assert "standing off the surface" in r["message"]
    assert "no solve and no verdict" in r["message"]


def test_profile_an_out_of_range_constant_names_the_range():
    """The head cannot see anything 337 mm away, whatever it reports. The
    refusal has to say so in millimetres and name the window, because that is
    the sentence that sends the operator to the standoff and not to the Azure."""
    far = _profile_grab(noise=0.0, quantum=0.0)
    far[:, 2] = -0.2575                      # standoff = 0.080 + 0.2575
    r = cs.check_profile(far, n_profiles=60)
    assert not r["ok"]
    rng_msg = [p for p in r["problems"] if p.startswith("RANGE")]
    assert rng_msg, r["problems"]
    assert "337.5 mm from the head" in rng_msg[0], rng_msg
    assert "57..103 mm" in rng_msg[0], rng_msg
    assert abs(r["standoff"] - 0.3375) < 1e-9


def test_profile_range_is_refused_even_with_honest_noise():
    """A GOOD surface at a BAD standoff. Nothing is wrong with the scatter, and
    it still cannot be where it says: out of the window is out of the window."""
    near = _profile_grab(noise=_KEY_NOISE_SPRAYED, z0=+0.030)   # 50 mm standoff
    r = cs.check_profile(near, n_profiles=60)
    assert not r["ok"]
    assert [p for p in r["problems"] if p.startswith("RANGE")], r["problems"]
    assert not [p for p in r["problems"] if p.startswith("SCATTER")], r["problems"]
    # and the edges of the window itself are NOT refused.
    for z0 in (+0.0225, -0.0225):
        assert cs.check_profile(_profile_grab(noise=_KEY_NOISE_SPRAYED,
                                              z0=z0))["ok"], z0


def test_profile_a_stuck_value_that_dithers_is_still_refused():
    """One repeated height with a handful of points scattered around it beats
    the RMS test by a hair. It does not beat the repeated-value test."""
    P = _profile_grab(noise=0.0, quantum=0.0, seed=5)
    rng = np.random.default_rng(5)
    k = rng.choice(P.shape[0], size=int(0.05*P.shape[0]), replace=False)
    P[k, 2] = rng.normal(0.0, 2.0e-3, k.size)        # 2 mm of junk, 5% of it
    r = cs.check_profile(P, n_profiles=60)
    assert not r["ok"]
    assert [p for p in r["problems"] if p.startswith("REPEATED VALUE")], r["problems"]
    assert r["rms"] > cs.KEYENCE_MIN_PROFILE_RMS      # the RMS test alone missed it


def test_profile_out_of_range_columns_are_counted_not_fitted():
    """keyence_driver_node emits an out-of-range column as +Inf (REP-117) and
    pc2's skip_nans does not remove an infinity. They must not reach a fit."""
    P = _profile_grab(noise=_KEY_NOISE_SPRAYED)
    P[::2, 2] = np.inf                                # half the line off the part
    r = cs.check_profile(P, n_profiles=60)
    assert r["ok"], r["problems"]
    assert r["n"] == P.shape[0] and r["n_finite"] == P.shape[0]//2
    assert abs(r["finite_fraction"] - 0.5) < 1e-6
    # all of it infinite is a refusal that talks about coverage
    P[:, 2] = np.inf
    r = cs.check_profile(P, n_profiles=60)
    assert not r["ok"]
    assert [p for p in r["problems"] if p.startswith("POINTS")], r["problems"]


def test_plane_support_refuses_one_profiler_pose_and_reproduces_the_artefact():
    """WHERE 0.000 mm AND 90.06 DEGREES CAME FROM, in four lines.

    Every point of one grab has y = 0 in the sensor's frame, so the set is a
    LINE. fit_plane() is total least squares and always succeeds: it returns
    the direction of least scatter, which is that y axis -- roughly HORIZONTAL
    when the head looks down -- at a residual of exactly zero. The rms and the
    tilt a single-line grab prints are that, and they are the same for
    perfectly good data, which is why no scatter test can catch this one."""
    grab = _profile_grab(noise=_KEY_NOISE_SPRAYED)
    assert cs.check_profile(grab)["ok"]                  # the DATA is fine

    R = cs.so3_exp([0.0, 0.0, 0.5])                      # line along some bearing
    world = grab.dot(R.T) + np.array([0.788, -0.262, 0.9207])
    n, _c, rms, _k = cs.fit_plane(world)
    assert rms*1e3 < 1e-6, rms                           # "0.000 mm rms flat"
    assert abs(n[2]) < 1e-6, n                           # the normal is horizontal
    vertical = np.array([0.0, 0.0, 1.0])
    tilt_mrad = math.acos(abs(float(np.clip(n.dot(vertical), -1, 1))))*1e3
    assert abs(tilt_mrad - 1570.8) < 1.0, tilt_mrad      # 90 degrees, as printed

    sup = cs.check_plane_support(world, name="the keyence patch")
    assert not sup["ok"]
    assert sup["length"]*1e3 > 30.0 and sup["width"]*1e3 < 1.0, sup
    assert "lie on a LINE" in sup["message"]
    assert "across its own laser line" in sup["message"]


def test_plane_support_accepts_a_head_panned_across_its_line():
    """And the fix: grabs at a few positions ACROSS the line are a patch, the
    normal comes back vertical, and the residual is the surface's own flatness
    instead of zero. This is what CAL_VERIFY_KEYENCE_OFFSETS buys."""
    R = cs.so3_exp([0.0, 0.0, 0.5])
    across = R[:, 1]
    world = np.vstack([
        _profile_grab(noise=_KEY_NOISE_SPRAYED, seed=10 + i).dot(R.T)
        + np.array([0.788, -0.262, 0.9207]) + across*o
        for i, o in enumerate((-0.020, 0.0, 0.020))])
    sup = cs.check_plane_support(world, name="the keyence patch")
    assert sup["ok"], sup["message"]
    # 40 mm of pan by 31 mm of laser line: both extents are real, which is the
    # difference between a patch and a single degenerate line.
    assert sup["width"]*1e3 > 25.0 and sup["length"]*1e3 > 35.0, sup
    assert sup["thickness"] < 1e-3, sup
    n, _c, rms, _k = cs.fit_plane(world)
    assert abs(abs(n[2]) - 1.0) < 1e-3, n
    assert abs(rms - _KEY_NOISE_SPRAYED) < 3.0e-6, rms

    # and the comparison it feeds now measures the Azure, which is the point:
    # an Azure 3 mm low over the same patch reads as 3 mm, not as 244.
    rng = np.random.default_rng(11)
    xy = rng.uniform(-0.045, 0.045, (1000, 2)) + np.array([0.788, -0.262])
    az = np.column_stack([xy, np.full(1000, 0.9207 - 0.003)
                          + rng.normal(0, 1.1e-3, 1000)])
    a = cs.plane_agreement(world, az, at=np.array([0.788, -0.262, 0.0]),
                           name_a="keyence", name_b="azure")
    assert abs(a["dz_mm"] + 3.0) < 0.3, a
    assert a["tilt_mrad"] < 10.0, a


# ------------------------------------------------------------------------------
# The profiler, fitted to geometry rather than to a pattern
# ------------------------------------------------------------------------------

KEY_TRUE = _T([0.0, -0.0752, 0.0231], [0.01, -0.02, 0.005])

ARTIFACT = cs.PlateArtifact(
    sphere_centres=[[-0.06, 0.0, 0.0127], [0.06, 0.0, 0.0127],
                    [0.0, 0.055, 0.0127]],
    sphere_radius=0.0127, plate_extent=(0.13, 0.10), name="cal plate")


#: The LJ-V7080 measures over 0.057..0.103 m and its optical frame sits at the
#: 0.080 m reference distance, so a surface is visible only within +/-23 mm of
#: z = 0 in the sensor frame. Outside that the head returns nothing, which is
#: exactly what turns a third of a plate into garbage when the raster is
#: planned over a mesh that is millimetres low.
KEY_HALF_RANGE = 0.023
#: 800 points over the ~40 mm line at the reference distance.
KEY_HALF_LINE = 0.020


def _profile_points(G, X, W, artifact, rng, noise=1.0e-4, n_per_line=200):
    """What a LINE PROFILER standing at each pose returns, in its own frame.

    Modelled as a line and not as a patch, because the difference is the whole
    argument for putting balls on the artifact. The head measures along its own
    x at y = 0, and reports the range z to the first surface it meets; so for
    each x this intersects the ray (x, 0, z) with the plate and with each ball,
    in the ARTIFACT's frame, and keeps the nearest hit inside the head's range.
    """
    out = []
    M_art = np.asarray(W, float)
    for g in G:
        M = cs.T_inv(M_art).dot(np.asarray(g, float)).dot(X)   # sensor -> artifact
        R, t = cs.T_to_rt(M)
        ex, ez = R[:, 0], R[:, 2]
        P = []
        for x in np.linspace(-KEY_HALF_LINE, KEY_HALF_LINE, n_per_line):
            q = t + ex*x                                # the ray's origin
            zs = []
            if abs(ez[2]) > 1e-9:                       # the plate, z_art = 0
                zp = -q[2]/ez[2]
                hit = q + ez*zp
                if (artifact.plate_extent is None
                        or (abs(hit[0]) <= artifact.plate_extent[0]
                            and abs(hit[1]) <= artifact.plate_extent[1])):
                    zs.append(zp)
            for c in artifact.centres:                  # each ball
                d = q - c
                b = 2.0*ez.dot(d)
                cc = d.dot(d) - artifact.radius**2
                disc = b*b - 4.0*cc
                if disc > 0.0:
                    zs += [(-b - math.sqrt(disc))/2.0, (-b + math.sqrt(disc))/2.0]
            zs = [z for z in zs if abs(z) <= KEY_HALF_RANGE]
            if zs:
                P.append([x, 0.0, min(zs)])             # the nearest surface
        P = np.asarray(P, float).reshape(-1, 3)
        if P.size:
            P = P + rng.normal(0.0, noise, P.shape)
        out.append(P)
    return out


def _sensor_poses(aims, tilts, rng, standoff=0.0):
    """probe_end poses that put the sensor's optical frame over each aim point.

    The real procedure does exactly this in reverse (_cap_probe_for_optical):
    decide where the SENSOR should be, then work back to where the probe tip
    has to go. `aims` are points in the artifact frame, `tilts` are
    (tx_deg, ty_deg, roll_deg) triples.
    """
    G = []
    for aim in aims:
        for tx, ty, roll in tilts:
            # Tool z DOWN onto the artifact, then tilted and rolled.
            R = (cs.so3_exp([math.radians(tx), 0.0, 0.0])
                 .dot(cs.so3_exp([0.0, math.radians(ty), 0.0]))
                 .dot(cs.so3_exp([math.pi, 0.0, 0.0]))
                 .dot(cs.so3_exp([0.0, 0.0, math.radians(roll)])))
            p = (np.asarray(aim, float) + rng.uniform(-0.002, 0.002, 3)
                 + np.array([0.0, 0.0, standoff]))
            O = cs.rt_to_T(R, W_TRUE[:3, :3].dot(p) + W_TRUE[:3, 3])
            G.append(O.dot(cs.T_inv(KEY_TRUE)))       # probe_end, not the head
    return G


#: Tilts and rolls inside this cell's measured cable envelope.
KEY_TILTS = [(0, 0, 0), (0, 0, 90), (25, 0, 30), (-25, 0, 150),
             (0, 15, 60), (0, -15, 120), (15, 10, 0), (-15, -10, 90)]


def test_profiler_recovers_its_transform_from_plate_and_balls():
    rng = np.random.default_rng(29)
    aims = list(ARTIFACT.centres[:, :2].tolist())
    aims = [[a[0], a[1], 0.020] for a in aims] + [[0.0, -0.04, 0.0]]
    G = _sensor_poses(aims, KEY_TILTS, rng)
    P = _profile_points(G, KEY_TRUE, W_TRUE, ARTIFACT, rng)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.003, -0.002, 0.003, 0.015, -0.010, 0.008]))
    W0 = W_TRUE.dot(cs.se3_exp([0.002, 0.002, -0.001, 0.008, 0.008, -0.008]))
    res = cs.solve_surface_fit(G, P, ARTIFACT, X0, W0)
    mm, mrad = _error(res.X, KEY_TRUE)
    assert res.adequate, res.report()
    assert mm < 0.5, "%.3f mm\n%s" % (mm, res.report())
    assert mrad < 8.0, "%.3f mrad\n%s" % (mrad, res.report())
    assert res.rms_mm < 0.3, res.report()
    assert any(k.startswith("ball") for k in res.feature_counts), res.feature_counts


def test_profiler_at_one_standoff_is_under_constrained_and_says_so():
    """The operator mistake that produces a confidently wrong profiler.

    The obvious way to take profiler calibration data is to hold the head at
    its reference distance over the artifact at every pose, because that is
    where it is happiest. Do that and rotation of the head about its own laser
    line is UNMEASURED: at zero range that rotation moves no measured point at
    first order, so every profile is identical whatever the angle.

    The fit converges, the residual is at the sensor's noise floor, and the
    roll of the laser plane is whatever the URDF said. On a 40 mm line a 5 mrad
    roll error is only 0.1 mm at the ends -- but it is a constant tilt applied
    to every profile in a raster, and it accumulates across a 250 mm scan.
    """
    rng = np.random.default_rng(31)
    bare = cs.PlateArtifact(sphere_centres=np.zeros((0, 3)), sphere_radius=0.0127,
                            plate_extent=(0.13, 0.10), name="bare flat")
    G = _sensor_poses([[0.0, 0.0, 0.0], [0.03, -0.02, 0.0]], KEY_TILTS, rng)
    P = _profile_points(G, KEY_TRUE, W_TRUE, bare, rng)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.003, -0.002, 0.003, 0.015, -0.010, 0.008]))
    res = cs.solve_surface_fit(G, P, bare, X0, W_TRUE, require_adequate=False)

    assert res.rms_mm < 0.15, res.report()         # at the head's noise floor
    assert not res.adequate, res.report()          # and still not determined
    assert "DOES NOT DETERMINE" in res.report()

    # The free direction is the one the geometry says it must be: rotation
    # about the laser line, index 3 of (vx vy vz | wx wy wz).
    worst = min(res.null_space, key=lambda sv_v: sv_v[0])[1]
    assert abs(worst[3]) > 0.9, (
        "expected the free direction to be rotation about the laser line, got "
        "%s\n%s" % (["%+.2f" % v for v in worst], res.report()))

    try:
        cs.solve_surface_fit(G, P, bare, X0, W_TRUE, require_adequate=True)
    except cs.PoseSetInadequate:
        pass
    else:
        raise AssertionError("one standoff should not yield a 6 DOF answer")


def test_the_artifact_is_not_a_plane_and_the_datum_is_not_its_lowest_point():
    """The datum is the CENTRE CYLINDER, 10 mm up a 24 mm object.

    Its top is 10 mm above the bottom of the base, so the rest of the artifact
    spans -10 mm (the bench around it) to +14 mm (the tallest pads) relative to
    the taught point -- and every one of those millimetres comes out of the
    head's 46 mm window. A standoff that is right for the datum can put the
    tallest pads inside the near limit.
    """
    ref, lo, hi, margin = 0.080, 0.057, 0.103, 0.005
    heights = {"bench": -0.010, "base top": -0.006, "lowest pads": -0.003,
               "datum pad": 0.0, "cones": 0.003, "tallest pads": 0.014}
    assert abs(min(heights.values())) == 0.010, "the datum is 10 mm up"
    assert max(heights.values()) == 0.014, "14 mm of artifact above the datum"

    def in_range(d):
        stand = ref + d
        return {k for k, h in heights.items()
                if (lo + margin) <= stand - h <= (hi - margin)}

    # The wide default: never blind, but never complete either.
    assert "tallest pads" not in in_range(-0.015), "-15 loses the tallest pads"
    assert in_range(0.0) == set(heights), "the reference distance sees it all"
    assert "bench" not in in_range(+0.015), "+15 loses the surrounding bench"
    # What matters is that every standoff still sees REAL FEATURES, not that it
    # sees all of them -- the pads and cones are what the solve reads.
    for d in (-0.015, 0.0, 0.015):
        assert in_range(d), "standoff %+.0f mm is blind" % (1e3*d)
        assert in_range(d) & {"datum pad", "cones", "tallest pads"}, \
            "standoff %+.0f mm sees no artifact feature" % (1e3*d)
    # And a standoff that IS blind must be refusable.
    assert not in_range(-0.080), "a zero standoff sees nothing -- refuse it"


def test_the_optical_frame_is_the_reference_plane_not_the_window():
    """Placing lj_v7080_optical_frame ON the surface IS the 80 mm standoff.

    lj_v7080_macro.xacro joins the optical frame to lj_v7080_frame at
    z = -(keyence_ref_dist + keyence_ref_dist_cal), with the comment "the
    optical frame sits this far in front of the head, on the plane a raw
    reading of zero means". So the optical frame is the reference PLANE, not
    the sensor's window, and `distance` is an offset within the measuring
    range rather than the whole standoff.

    Adding the reference distance a second time stands the head off 160 mm --
    outside the 57..103 mm it can measure -- and every grab comes back empty.
    The name is the trap: "optical frame" reads like the window.
    """
    ref, lo, hi = 0.080, 0.057, 0.103
    for d in (-0.015, 0.0, 0.015):
        assert lo <= ref + d <= hi, d               # head standoff, correct
        assert not (lo <= ref + ref + d <= hi), d   # the double count, refused


def _hits_artifact():
    """The REAL _cal_hits_artifact out of calibration.dpi, pulled by AST.

    It is exercised here rather than re-derived, because a test that recomputes
    the box arithmetic inline passes while the shipped function raises
    TypeError on the first candidate of every real run -- reading
    tooling_envelope()'s return value as an array when it is an object -- and
    the caller's except-clause turns that into one line of output and no
    clearance checking at all.
    """
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration.dpi")).read()
    tree = ast.parse("def _b():\n" + "".join("    " + l + "\n"
                                             for l in src.splitlines()))
    want = {"_cal_hits_artifact", "_cal_clearance_profile"}
    fns = [n for n in tree.body[0].body
           if isinstance(n, ast.FunctionDef) and n.name in want]
    assert len(fns) == len(want), [f.name for f in fns]
    ns = {"np": np, "_cal_geom": lambda: cs, "_CAL": {},
          "CAL_ARTIFACT_FOOTPRINT_M": (0.160, 0.160),
          "CAL_ARTIFACT_CLEARANCE_M": 0.005,
          "CAL_ARTIFACT_HEIGHTS": {"the bench around it": -0.010,
                                   "the base top": -0.006,
                                   "the lowest pads": -0.003,
                                   "the datum pad": 0.000,
                                   "the cone apices": 0.003,
                                   "the tallest pads": 0.014}}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    return ns["_cal_hits_artifact"]


class _Envelope(object):
    """The shape tooling_envelope() really returns: an OBJECT with .links.

    Not a dict and not an array. The links carry the measured offsets of the
    real tooling in the probe_end frame, read off the cell's own description:
    probe_end is the eddy-current probe TIP (z 0 at the tip, the barrel running
    back up to +68 mm) and the Keyence sits about 75 mm to one side of it. That
    lever arm is the whole mechanism -- tilt the tool and the probe swings
    through it.
    """

    def __init__(self, links):
        self.links = [{"name": n, "points": np.asarray(p, float)}
                      for n, p in links]
        self.points = np.vstack([l["points"] for l in self.links])


def _real_tooling():
    def barrel(x0, y0, z0, z1, r, n=9):
        th = np.linspace(0, 2*np.pi, n, endpoint=False)
        z = np.linspace(z0, z1, n)
        return np.array([[x0 + r*math.cos(t), y0 + r*math.sin(t), zz]
                         for zz in z for t in th])
    return _Envelope([
        ("ec_probe",       barrel(0.0, 0.0, 0.000, 0.068, 0.007)),
        ("probe_holder",   barrel(0.0, 0.0, 0.009, 0.065, 0.012)),
        ("lj_v7080_frame", barrel(-0.115, 0.003, 0.045, 0.117, 0.055)),
        ("outer_shell",    barrel(0.0, 0.0, 0.083, 0.186, 0.071)),
    ])


def _printed_artifact():
    """The printed artifact as built, with MEASURED pad tops."""
    pitch, first, base_z, pad_d = 0.045, 0.035, 0.004, 0.022
    pads = [(first + c*pitch, first + r*pitch)
            for r in range(3) for c in range(3)]
    tops = [0.007134, 0.014138, 0.021092,
            0.024068, 0.010150, 0.017094,
            0.014109, 0.007098, 0.021034]
    cones = [(0.0575, 0.0575), (0.1025, 0.0575),
             (0.0575, 0.1025), (0.1025, 0.1025)]
    base = [(0.010, 0.0), (0.160, 0.0), (0.160, 0.160),
            (0.0, 0.160), (0.0, 0.010)]
    return cs.SteppedArtifact(pads, tops, pad_d, cones, base_z + 0.009,
                              0.026, base, base_z)


def _tool_quat(tx_deg, ty_deg, roll_deg):
    """_cal_tool_quat()'s convention: world tilts on the left, roll on the right."""
    R = (cs.so3_exp([math.radians(tx_deg), 0.0, 0.0])
         .dot(cs.so3_exp([0.0, math.radians(ty_deg), 0.0]))
         .dot(cs.so3_exp([0.0, 0.0, math.radians(roll_deg)])))
    return cs.R_to_quat(R)


def test_a_tilted_pose_can_put_the_tool_inside_the_artifact():
    """A tilted calibration pose driving the tool into the artifact.

    Screening a candidate for the work volume, for IK and for the cable
    envelope does not ask whether the TOOL would be inside the thing it is
    measuring -- and that does not need asking until the pose is tilted.
    probe_end sits about 75 mm to the side of the optical frame, so a 25 degree
    tilt swings it through that offset by ~35 mm.

    THESE ARE REAL COMMANDED POSES from a calibrate_keyence() run, with what
    the arm actually did to each one. The check has to reproduce that column:
    refuse the one that went into the surface and pass the ten that flew.
    Refusing more than that is not "safe" -- it costs the d-15 rung, and the
    standoff spread is what measures rotation of the head about its own laser
    line.
    """
    hits = _hits_artifact()
    env, art = _real_tooling(), _printed_artifact()
    aim = np.array([0.8304, 0.0472, 0.9318])            # the taught cal_flat

    # (label, tilt, roll, commanded probe_end, did it fly)
    real = [("t+0,+0 r-90 d-15",  (0, 0),   -90, (0.8322, -0.0281, 0.9511), True),
            ("t+0,+0 r+90 d-15",  (0, 0),    90, (0.8286,  0.1224, 0.9511), True),
            ("t+25,+0 r+90 d-15", (25, 0),   90, (0.8286,  0.1072, 0.9811), True),
            ("t-25,+0 r-57 d-15", (-25, 0), -57, (0.8724, -0.0013, 0.9757), True),
            ("t+0,+15 r-90 d-15", (0, 15),  -90, (0.8372, -0.0281, 0.9500), True),
            ("t+0,+0 r-90 d0",    (0, 0),   -90, (0.8326, -0.0280, 0.9661), True),
            ("t+0,+0 r+90 d0",    (0, 0),    90, (0.8282,  0.1223, 0.9661), True),
            ("t+25,+0 r-90 d0",   (25, 0),  -90, (0.8326, -0.0354, 0.9311), False),
            ("t+25,+0 r+90 d0",   (25, 0),   90, (0.8282,  0.1008, 0.9946), True),
            ("t-25,+0 r-57 d0",   (-25, 0), -57, (0.8727,  0.0053, 0.9891), True),
            ("t+0,+15 r-90 d0",   (0, 15),  -90, (0.8414, -0.0280, 0.9643), True)]
    for label, (tx, ty), roll, xyz, flew in real:
        depth, who = hits(np.asarray(xyz), _tool_quat(tx, ty, roll), aim, env,
                          artifact=art)
        if flew:
            assert depth is None, (
                "%s flew on the real cell and the check refuses it (%s %.1f mm "
                "in). Over-refusing costs the standoff spread."
                % (label, who, (depth or 0)*1e3))
        else:
            assert depth is not None, (
                "%s commanded probe_end to z %.4f against a datum at %.4f -- "
                "0.7 mm INTO the surface -- and the check passes it"
                % (label, xyz[2], aim[2]))
            assert who == "ec_probe", who

    # And the failure mode that hides it: the real return value is an OBJECT.
    # Reading it as an array raises TypeError on every candidate, which the
    # caller swallows into "NO TOOLING ENVELOPE" and carries on unguarded.
    depth, _who = hits(np.array([0.8326, -0.0354, 0.9311]),
                       _tool_quat(25, 0, -90), aim, env, artifact=art)
    assert depth is not None


def test_the_clearance_envelope_never_sits_below_the_real_surface():
    """The radial envelope is an UPPER BOUND on the height field, at any yaw.

    The operator teaches one point -- the top of the centre pad -- so the
    artifact's yaw is unknown when the poses are generated and the height field
    cannot be evaluated. The maximum height on each circle about the centre can
    be, for every yaw at once. It is only worth using instead of "the whole
    footprint is as tall as its tallest pad" if it is genuinely sound, so this
    checks it against the artifact's own height() at two dozen yaws.
    """
    art = _printed_artifact()
    r, h = cs.radial_clearance_profile(art)
    centre = art.pads.mean(axis=0)
    datum = art.pad_h[int(np.argmin(np.linalg.norm(art.pads - centre, axis=1)))]
    rng = np.random.default_rng(1)
    worst = -np.inf
    for yaw in np.linspace(0.0, 2*np.pi, 24, endpoint=False):
        P = rng.uniform(-0.13, 0.13, size=(4000, 2))
        c, s_ = math.cos(yaw), math.sin(yaw)
        Q = np.column_stack([P[:, 0]*c - P[:, 1]*s_,
                             P[:, 0]*s_ + P[:, 1]*c]) + centre
        true = art.height(np.column_stack([Q, np.zeros(len(Q))]))[0] - datum
        worst = max(worst, float((true - np.interp(np.linalg.norm(P, axis=1),
                                                   r, h)).max()))
    assert worst <= 0.0, ("the envelope is %.3f mm BELOW the real surface "
                          "somewhere, which is the unsafe direction" % (worst*1e3))

    # And it is worth having: at the plate edge it is 20 mm lower than the
    # flat-box model, which is what keeps the near-standoff poses.
    assert np.interp(0.078, r, h) < max(art.pad_h) - datum - 0.015


def test_varying_the_standoff_is_what_fixes_it():
    """And the fix is free: the same flat, the same tilts, three standoffs.

    This is the measurement behind the design document's claim that the Keyence
    does NOT need a sphere artifact. A bare ground flat, tilted and taken at
    -15, 0 and +15 mm of standoff, determines all six degrees of freedom at
    this head's real 0.1 mm noise.
    """
    rng = np.random.default_rng(41)
    bare = cs.PlateArtifact(np.zeros((0, 3)), 0.0127, (0.13, 0.10), "bare flat")
    aims = [[0.0, 0.0, 0.0], [0.03, -0.02, 0.0]]
    G = []
    for standoff in (-0.015, 0.0, 0.015):
        G += _sensor_poses(aims, KEY_TILTS, rng, standoff=standoff)
    P = _profile_points(G, KEY_TRUE, W_TRUE, bare, rng)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.004, -0.003, 0.004, 0.020, -0.015, 0.012]))
    res = cs.solve_surface_fit(G, P, bare, X0, W_TRUE)

    assert res.adequate, res.report()
    mm, mrad = _error(res.X, KEY_TRUE)
    assert mm < 0.30, "%.3f mm\n%s" % (mm, res.report())
    assert mrad < 2.0, "%.3f mrad\n%s" % (mrad, res.report())


def test_balls_help_but_are_not_what_makes_it_work():
    """Honest bookkeeping on the artifact recommendation.

    Three balls on the plate improve the profiler solve, and the improvement is
    real but small next to the difference varying the standoff makes. The
    design document says so rather than implying the balls are load-bearing;
    this pins the claim so nobody has to re-derive it.
    """
    aims = [[0.0, 0.0, 0.0], [0.03, -0.02, 0.0]]
    ball_aims = [[c[0], c[1], 0.020] for c in ARTIFACT.centres]
    X0 = KEY_TRUE.dot(cs.se3_exp([0.004, -0.003, 0.004, 0.020, -0.015, 0.012]))

    def solve(artifact, extra_aims, seed):
        rng = np.random.default_rng(seed)
        G = []
        for standoff in (-0.015, 0.0, 0.015):
            G += _sensor_poses(aims + extra_aims, KEY_TILTS, rng, standoff=standoff)
        P = _profile_points(G, KEY_TRUE, W_TRUE, artifact, rng)
        return cs.solve_surface_fit(G, P, artifact, X0, W_TRUE)

    bare = cs.PlateArtifact(np.zeros((0, 3)), 0.0127, (0.13, 0.10), "bare flat")
    flat = solve(bare, [], 51)
    balls = solve(ARTIFACT, ball_aims, 51)

    assert flat.adequate and balls.adequate, (flat.report(), balls.report())
    assert _error(flat.X, KEY_TRUE)[0] < 0.5, flat.report()      # good enough
    assert _error(balls.X, KEY_TRUE)[0] <= _error(flat.X, KEY_TRUE)[0] + 0.05
    assert any(k.startswith("ball") for k in balls.feature_counts)


def test_board_object_points():
    g = cs.board_object_points(9, 6, 0.020)
    assert g.shape == (54, 3)
    assert np.allclose(g[0], [0.0, 0.0, 0.0])
    assert np.allclose(g[8], [0.160, 0.0, 0.0])      # along a row first
    assert np.allclose(g[9], [0.0, 0.020, 0.0])
    assert np.allclose(g[:, 2], 0.0)


# ------------------------------------------------------------------------------
# The 3D-printed stepped artifact
# ------------------------------------------------------------------------------
# docs/calibration/make_artifact.py generates the STL; these are its design
# numbers, in metres, in the artifact's own frame with the TABLE at z = 0.

PAD_TOPS = [0.007, 0.014, 0.021, 0.024, 0.010, 0.017, 0.014, 0.007, 0.021]
PAD_XY = [(0.035 + 0.045*c, 0.035 + 0.045*r) for r in range(3) for c in range(3)]
CONE_XY = [(0.0575, 0.0575), (0.1025, 0.0575), (0.0575, 0.1025), (0.1025, 0.1025)]
BASE_POLY = [(0.025, 0.0), (0.160, 0.0), (0.160, 0.160), (0.0, 0.160), (0.0, 0.025)]

STEPPED = cs.SteppedArtifact(PAD_XY, PAD_TOPS, 0.022, CONE_XY, 0.013, 0.026,
                             BASE_POLY, 0.004, table_extent=0.30)

#: The artifact frame's centre, so a pose set can be aimed at the middle of it
#: rather than at the corner the coordinates start from.
STEP_CENTRE = (0.080, 0.080, 0.012)


#: mm of extra apparent range per unit of (1 - cos tilt), measured at this cell.
#: A tilt ladder over the same bench flat reads 0.23 mm LOWER at 20 degrees of
#: tilt than at normal incidence, symmetrically in both tilt axes and both
#: signs, and 1 - cos(20 deg) = 0.0603, so the coefficient is 3.81 mm. Sensor
#: angle response on a tilted surface and the arm's own pose error at a
#: reconfigured wrist both have this shape; neither is a standoff error, and a
#: solve with the surface free absorbs it as one. See
#: test_a_tilt_correlated_range_bias_is_absorbed_as_a_standoff_error.
KEY_ANGLE_BIAS_M = 3.81e-3


def _stepped_profile_points(G, X, W, artifact, rng, noise=1.0e-4,
                            n_per_line=200, dz=2.0e-4, angle_bias=0.0):
    """What a line profiler returns over a HEIGHT FIELD, by first hit.

    Marched along the ray rather than solved, because a stepped artifact has
    vertical walls and a cone flank, and the fixed-point iteration that works
    for a plane oscillates at every edge. Marching is slower and cannot get
    the answer wrong. Points whose first hit is outside the head's +/-23 mm
    window are dropped, exactly as the head drops them.

    `angle_bias` adds a TILT-CORRELATED error to the reported range, in metres
    per unit of (1 - cos tilt) -- the real head's behaviour, and enough on its
    own to move a solve by several mm. It is not noise: it is identical for
    every point of a pose and grows with how far that pose is tilted off the
    surface normal, so no number of poses averages it away.
    """
    out = []
    M_art = np.asarray(W, float)
    ts = np.arange(-KEY_HALF_RANGE, KEY_HALF_RANGE, dz)
    for g in G:
        M = cs.T_inv(M_art).dot(np.asarray(g, float)).dot(X)
        R, t0 = cs.T_to_rt(M)
        ex, ez = R[:, 0], R[:, 2]
        xs = np.linspace(-KEY_HALF_LINE, KEY_HALF_LINE, n_per_line)
        # (line sample, march step, 3) in the artifact frame
        Q = (t0[None, None, :] + ex[None, None, :]*xs[:, None, None]
             + ez[None, None, :]*ts[None, :, None])
        h, _w, _m = artifact.height(Q.reshape(-1, 3))
        above = (Q[:, :, 2] - h.reshape(Q.shape[:2])) > 0.0
        # the first march step that is NOT above the surface
        hit = np.argmax(~above, axis=1)
        ok = (~above).any(axis=1) & above[:, 0]      # started clear, then hit
        P = []
        for k in np.nonzero(ok)[0]:
            P.append([xs[k], 0.0, ts[hit[k]]])
        P = np.asarray(P, float).reshape(-1, 3)
        if P.size:
            P = P + rng.normal(0.0, noise, P.shape)
            if angle_bias:
                # ez is the sensor's own z axis in the artifact frame, and the
                # surface it looks at is the artifact's xy plane, so |ez[2]| is
                # cos(tilt). The bias is on the RANGE -- the sensor frame's z --
                # and it is ONE number for the whole pose.
                P[:, 2] += angle_bias*(1.0 - abs(float(ez[2])))
        out.append(P)
    return out


def test_stepped_artifact_height_field():
    """Every feature at the height it was designed at, the chamfer outside the
    base, and a millimetre either side of each edge thrown away."""
    assert [round(v*1e3, 1) for v in STEPPED.heights()] == [
        0.0, 4.0, 7.0, 10.0, 13.0, 14.0, 17.0, 21.0, 24.0]

    P = np.array([
        [0.200, 0.200, 0.000],          # the table, well clear
        [0.010, 0.140, 0.004],          # the base top
        [0.035, 0.080, 0.024],          # the tallest pad
        [0.0575, 0.0575, 0.013],        # a cone apex
        [0.0575 + 0.0065, 0.0575, 0.0085],   # halfway down that cone's flank
        [0.010, 0.010, 0.000],          # inside the chamfer, so table
    ])
    d, w = STEPPED.signed_distance(P)
    assert np.allclose(d, 0.0, atol=1e-9), d*1e3
    assert [STEPPED.feature_name(x) for x in w] == [
        "table", "base top", "pad 3", "cone 0", "cone 0", "table"]

    # A point right on a pad rim is thrown away rather than fitted, because a
    # printed edge is rounded and a profiler crossing one returns a smear.
    edge = np.array([[0.035 + 0.011, 0.035, 0.007]])
    assert not np.isfinite(STEPPED.signed_distance(edge)[0][0])


def _stepped_run(standoffs, rng_seed=31):
    rng = np.random.default_rng(rng_seed)
    aims = [[STEP_CENTRE[0], STEP_CENTRE[1], STEP_CENTRE[2]],
            [STEP_CENTRE[0] - 0.03, STEP_CENTRE[1] + 0.02, STEP_CENTRE[2]]]
    G = []
    for s in standoffs:
        G += _sensor_poses(aims, KEY_TILTS, rng, standoff=s)
    P = _stepped_profile_points(G, KEY_TRUE, W_TRUE, STEPPED, rng)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.003, -0.002, 0.003, 0.015, -0.010, 0.008]))
    W0 = W_TRUE.dot(cs.se3_exp([0.002, 0.002, -0.001, 0.004, 0.004, -0.004]))
    return P, cs.solve_surface_fit(G, P, STEPPED, X0, W0)


def _flat_run(standoffs, rng_seed=31):
    rng = np.random.default_rng(rng_seed)
    bare = cs.PlateArtifact(np.zeros((0, 3)), 0.0127, plate_extent=(0.13, 0.10),
                            name="bare flat")
    G = []
    for s in standoffs:
        G += _sensor_poses([[0.0, 0.0, 0.0], [0.03, -0.02, 0.0]], KEY_TILTS,
                           rng, standoff=s)
    P = _profile_points(G, KEY_TRUE, W_TRUE, bare, rng)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.003, -0.002, 0.003, 0.015, -0.010, 0.008]))
    return P, cs.solve_surface_fit(G, P, bare, X0, W_TRUE)


def _weakest(res):
    return min([s for s, _v in res.null_space] or [float("inf")])


def test_the_printed_artifact_helps_the_weak_direction_but_does_not_replace_the_standoffs():
    """WHAT THE STEPPED ARTIFACT ACTUALLY BUYS, measured rather than argued.

    It is tempting to say that because the artifact's features span 24 mm, a
    single pass already sees nine ranges and the single-standoff degeneracy is
    gone. It is not, and the reason is worth knowing.

    The free direction is rotation about the laser line. To first order it
    displaces each measured point along the sensor's OWN y by -z*delta -- and
    on a horizontal top face that displacement is TANGENTIAL, so it changes no
    measured height whatever z is. The artifact is a stack of horizontal top
    faces. Only its four cone flanks have a y gradient, and they are a small
    part of any one profile line.

    Measured here, same poses, same noise, 0.1 mm per point:

        bare flat,  one standoff    0.079 mm   free direction, sv 0.032
        printed,    one standoff    0.085 mm   free direction, sv 0.067
        bare flat,  three standoffs 0.021 mm   determined
        printed,    three standoffs 0.110 mm   determined

    So the artifact roughly DOUBLES the leverage on the weak direction and
    still does not determine it. Varying the standoff across the pose set is
    what does, which is why keyence_calibration_poses() refuses a single one
    whatever the artifact. What the artifact is for is everything else -- it is
    an asymmetric rigid body all three sensors can locate, where a flat is a
    body none of them can locate laterally at all.
    """
    P, one = _stepped_run([0.0])
    assert min(p.shape[0] for p in P) > 50, [p.shape[0] for p in P]
    mm, mrad = _error(one.X, KEY_TRUE)
    assert mm < 0.5, "%.3f mm\n%s" % (mm, one.report())
    assert not one.adequate, one.report()
    assert "DOES NOT DETERMINE" in one.report()
    worst = min(one.null_space, key=lambda sv_v: sv_v[0])[1]
    assert abs(worst[3]) > 0.9, ["%+.2f" % v for v in worst]

    _Pf, flat = _flat_run([0.0])
    assert not flat.adequate, flat.report()
    assert _weakest(one) > 1.5*_weakest(flat), (_weakest(one), _weakest(flat))

    # And with the standoffs the procedure actually flies, it is determined.
    _P3, three = _stepped_run([-0.010, 0.0, 0.010])
    assert three.adequate, three.report()
    mm3, mrad3 = _error(three.X, KEY_TRUE)
    assert mm3 < 0.5, "%.3f mm\n%s" % (mm3, three.report())
    assert mrad3 < 8.0, "%.3f mrad\n%s" % (mrad3, three.report())
    # The pads and cones really were seen, not just the table around them.
    seen = set(three.feature_counts)
    assert any(k.startswith("pad") for k in seen), seen
    assert any(k.startswith("cone") for k in seen), seen
    del mrad


#: A realistic partial pose set: the tilts and rolls a calibrate_keyence() run
#: gets through before the cable guard refuses a pose part-way in.
FLOWN_TILTS = [(0, 0, -90), (0, 0, 90), (25, 0, 90), (-25, 0, -57),
               (0, 15, -90), (0, 15, 90), (0, -15, -90), (0, -15, 90)]

#: CAL_KEYENCE_AIM_OFFSETS: the centre of the artifact and half a pad-pitch out
#: in each direction.
SPREAD_AIMS = [(0.0, 0.0), (+0.0225, 0.0), (-0.0225, 0.0),
               (0.0, +0.0225), (0.0, -0.0225)]


def _aimed_run(aim_offsets, standoffs, tilts=None, dz_start=0.0, seed=17,
               fix_W=False, W0=None, solve=True, angle_bias=0.0):
    """A stepped-artifact run aimed at STEP_CENTRE plus each (dx, dy) offset."""
    rng = np.random.default_rng(seed)
    tilts = FLOWN_TILTS if tilts is None else tilts
    aims = [[STEP_CENTRE[0] + dx, STEP_CENTRE[1] + dy, STEP_CENTRE[2]]
            for dx, dy in aim_offsets]
    G = []
    for st in standoffs:
        G += _sensor_poses(aims, tilts, rng, standoff=st)
    P = _stepped_profile_points(G, KEY_TRUE, W_TRUE, STEPPED, rng,
                                angle_bias=angle_bias)
    X0 = KEY_TRUE.dot(cs.se3_exp([0.0, 0.0, dz_start, 0.0, 0.0, 0.0]))
    if W0 is None:
        W0 = W_TRUE.dot(cs.se3_exp([0.001, 0.001, -0.001, 0.003, 0.003, -0.003]))
    if not solve:
        return G, P, X0, W0
    return P, cs.solve_surface_fit(G, P, STEPPED, X0, W0, fix_W=fix_W)


def test_a_tilt_correlated_range_bias_is_absorbed_as_a_standoff_error():
    """HOW A KEYENCE SOLVE RETURNS A STANDOFF IT HAS NOT MEASURED -- the
    mechanism, and which of the available fixes are enough.

    A run designed with three standoff rungs and aimed at the taught datum, cut
    short by a refused pose, reaches the solver as ONE aim point at TWO
    standoffs. The free fit then moves the sensor 3.80 mm down and the artifact
    3.46 mm down, with a 0.179 mm residual over 494 017 points.

    THE CAUSE IS NOT GEOMETRY, and that is why this test injects a sensor error
    rather than only thinning the pose set. On clean data that pose set does
    determine the standoff. What breaks it is the head's own TILT-CORRELATED
    RANGE ERROR: on a tilt ladder the same bench flat reads 0.23 mm lower at
    20 degrees of tilt than at normal incidence, symmetrically in both axes and
    both signs. A solve with the surface free has exactly one
    way to explain "the tilted poses read low" -- move the sensor along its own
    standoff -- because the residual's sensitivity to the standoff and to the
    surface's height are proportional for every point unless the head is
    tilted, and their ratio is 1 - cos(tilt) whatever the artifact's shape. At
    the 20 degrees the cable envelope allows that is 0.060, and
    0.23 / 0.060 = 3.8 mm. Sensor angle response and the arm's own pose error at
    a reconfigured wrist both look like this; neither is a standoff.

    So no artifact shape fixes it and no number of poses averages it away -- the
    error is one offset per pose and grows with tilt. What fixes it is geometry
    that pins the surface independently. Measured here:

        the truncated set                3.04 mm out, 0.194 mm rms
        FIX 1  the third standoff rung   1.63 mm out  -- halves it, not enough
        FIX 2  aims spread over the pads 0.40 mm out  -- enough
        FIX 3  artifact held at the
               touch-off                 0.20 mm out  -- enough

    FIX 3 is why the calibration is allowed a taught point even though the
    routine pipeline must have none.
    """
    B = KEY_ANGLE_BIAS_M

    def out_by(aims, standoffs, **kw):
        """How far the fitted standoff lands from the truth, in mm."""
        W0 = W_TRUE if kw.get("fix_W") else None
        G, P, _X0, W_start = _aimed_run(aims, standoffs, angle_bias=B,
                                        solve=False, W0=W0)
        res = cs.solve_surface_fit(G, P, STEPPED, KEY_TRUE.copy(), W_start,
                                   fix_W=bool(kw.get("fix_W")))
        return abs(float(res.X[2, 3] - KEY_TRUE[2, 3]))*1e3, res

    # The truncated set: a small residual and a standoff millimetres out.
    flown, ref = out_by([(0.0, 0.0)], (-0.015, 0.0))
    assert flown > 2.0, (
        "the tilt bias should cost this pose set millimetres of standoff, and "
        "it only moved %.2f mm" % flown)
    assert ref.rms_mm < 0.25, (
        "and it should do it QUIETLY -- a %.3f mm residual would have been "
        "noticed" % ref.rms_mm)
    # One aim point reaches one pad, which is the whole mechanism.
    assert len([k for k in ref.feature_counts if k.startswith("pad")]) == 1, \
        ref.feature_counts

    # FIX 1: the third standoff rung -- what an abort denies the set. It helps,
    # and on its own it is NOT enough. Interleaving the standoffs in the flight
    # order (see _cal_candidate_poses) is what keeps all three rungs in a run
    # that loses poses; it is not a substitute for FIX 2 or FIX 3.
    three, _r = out_by([(0.0, 0.0)], (-0.015, 0.0, 0.015))
    assert three < flown, "the third rung should help; %.2f mm" % three
    assert three > 1.0, (
        "the third rung alone should NOT be treated as a fix -- it left "
        "%.2f mm, and this test is what says so" % three)

    # FIX 2: aim across the artifact, so one profile crosses several pad
    # heights. This is what makes two rungs enough, and two rungs is what a run
    # that loses poses in flight actually ends up with.
    spread, ref_s = out_by(SPREAD_AIMS, (-0.015, 0.0))
    assert spread < 0.5, "spread aims should pin it; %.2f mm left" % spread
    assert len([k for k in ref_s.feature_counts if k.startswith("pad")]) >= 3, \
        ref_s.feature_counts

    # FIX 3: hold the artifact where the touch-off put it. Allowed at
    # calibration time even though the routine pipeline must have none.
    held, _r3 = out_by([(0.0, 0.0)], (-0.015, 0.0), fix_W=True)
    assert held < 0.5, "holding the artifact should pin it; %.2f mm" % held

    # AND THE GUARD CATCHES THE TWO THAT ARE NOT ENOUGH. A check that only
    # agrees with the fixes is worth nothing; it has to refuse the failures.
    for aims, standoffs, what in (([(0.0, 0.0)], (-0.015, 0.0), "the truncated set"),
                                  ([(0.0, 0.0)], (-0.015, 0.0, 0.015),
                                   "the third rung alone")):
        G, P, _X0, W0 = _aimed_run(aims, standoffs, angle_bias=B, solve=False)
        res = cs.solve_surface_fit(G, P, STEPPED, KEY_TRUE.copy(), W0)
        rec = cs.surface_fit_recovery(G, P, STEPPED, res.X, res.W)
        assert rec is not None and not rec["ok"], \
            "%s must not pass the recovery check: %r" % (what, rec)


def test_a_small_residual_does_not_mean_the_answer_was_measured():
    """The recovery check, against the case every other diagnostic passes.

    solve_surface_fit() reports this pose set as adequate, with an empty null
    space and a formal 1 sigma of microns, because it divides by the point
    count and those points are 60 copies of one line at each of 14 static
    poses. surface_fit_recovery() asks the question directly instead: push the
    answer and see what comes back.

    THE BIAS IS PART OF THE CASE, not decoration. On clean data this pose set
    IS determined; building it out of perfect data and relying on the solve
    coming back wrong tests the solver's convergence, not its observability.
    What defeats the real run is the head's tilt-correlated range error, so
    that is what the case has to contain.
    """
    G, P, _X0, W0 = _aimed_run([(0.0, 0.0)], (-0.015, 0.0), solve=False,
                               angle_bias=KEY_ANGLE_BIAS_M)
    res = cs.solve_surface_fit(G, P, STEPPED, KEY_TRUE.copy(), W0)
    # Everything the old report looks at says this is fine.
    assert res.adequate, res.report()
    assert not res.null_space
    assert res.rms_mm < 0.25, res.report()
    # ... and it is 3 mm out in the standoff. That is the whole point.
    assert abs(res.X[2, 3] - KEY_TRUE[2, 3])*1e3 > 2.0, res.report()

    bad = cs.surface_fit_recovery(G, P, STEPPED, res.X, res.W)
    assert not bad["ok"], bad
    assert min(bad["recovered"]) < cs.SURFACE_FIT_MIN_RECOVERY, bad
    text = "\n".join(cs.describe_surface_fit_recovery(bad))
    assert "NOT MEASURED" in text, text

    # ... and the set the procedure now flies passes it.
    G2, P2, _X2, W2 = _aimed_run(SPREAD_AIMS, (-0.015, 0.0, 0.015), solve=False)
    good = cs.solve_surface_fit(G2, P2, STEPPED, KEY_TRUE.copy(), W2)
    ok = cs.surface_fit_recovery(G2, P2, STEPPED, good.X, good.W)
    assert ok["ok"], ok


def test_the_artifacts_range_spread_fits_inside_the_heads_window():
    """A stepped artifact taller than the head's window is a stepped artifact
    most of which returns nothing. 24 mm inside a 46 mm window leaves 11 mm of
    placement margin at each end."""
    z = STEPPED.heights()
    spread = z[-1] - z[0]
    window = 0.103 - 0.057
    assert spread < window, (spread, window)
    assert (window - spread)/2.0 > 0.010, "less than 10 mm of placement margin"
    # Centred on the 80 mm reference distance, this is what one pass sees.
    mid = 0.5*(z[0] + z[-1])
    d = sorted(0.080 + (mid - v) for v in z)
    assert [round(v*1e3) for v in d] == [68, 71, 75, 78, 79, 82, 85, 88, 92]


# ------------------------------------------------------------------------------
# The drop-in file: solve -> write -> read back -> the same transform
# ------------------------------------------------------------------------------
# THE ONLY QUESTION THAT MATTERS is whether the numbers written into the cell's
# xacro make the deployed chain publish the transform that was solved. These
# tests stand in for the chain with two arbitrary transforms -- A, the
# correction's parent link to probe_end, and B, the correction's child link to
# the sensor -- both of which calibration.dpi reads from live TF. Neither is
# small, neither is axis-aligned, and B carries a large rotation, because that
# is what makes the two composition orders differ.

A_CHAIN = _T([0.012, -0.031, 0.204], [0.05, 3.10, -0.02])       # mounting_plate
B_CHAIN = _T([0.041, 0.0, 0.0301], [-1.5708, -1.5708, 0.0])     # the redo bracket


def test_mount_cal_makes_the_chain_publish_what_was_solved():
    """Cal = A X B^-1, and A^-1 Cal B is X again. Exactly, not nearly."""
    Cal = cs.mount_cal(X_TRUE, A_CHAIN, B_CHAIN)
    back = cs.sensor_transform(Cal, A_CHAIN, B_CHAIN)
    assert np.allclose(back, X_TRUE, atol=1e-12), _error(back)


def test_the_wrong_composition_order_is_plausible_and_wrong():
    """The bug this is all arranged to prevent.

    Writing the correction in the sensor's frame instead of the parent's gives
    a transform that is just as rigid, just as small, and renders just as
    cleanly. Here it is 76 mm and 3 radians out, and nothing downstream would
    say a word about it.
    """
    right = cs.mount_cal(X_TRUE, A_CHAIN, B_CHAIN)
    wrong = cs.T_inv(A_CHAIN).dot(X_TRUE).dot(B_CHAIN)   # child-frame version
    dt, dr = cs.pose_delta(right, wrong)
    assert dt*1e3 > 10.0 and dr > 0.1, (dt*1e3, dr)
    # and the wrong one does NOT reproduce the solve
    dt2, _dr2 = cs.pose_delta(cs.sensor_transform(wrong, A_CHAIN, B_CHAIN), X_TRUE)
    assert dt2*1e3 > 10.0, dt2*1e3


def test_emitted_file_round_trips_through_text():
    """solve -> the xacro text an operator copies -> parsed back -> the same
    transform. The file is the deliverable, so the file is what is tested."""
    for sensor, X in (("azure", X_TRUE), ("keyence", KEY_TRUE)):
        Cal = cs.mount_cal(X, A_CHAIN, B_CHAIN)
        text = cs.render_calibration_xacro({sensor: Cal})
        props = cs.parse_calibration_xacro(text)
        p_xyz, p_rpy = cs.CAL_XACRO_PROPERTIES[sensor]
        assert p_xyz in props and p_rpy in props, props
        back = cs.properties_to_cal(props[p_xyz], props[p_rpy])
        dt, dr = cs.pose_delta(Cal, back)
        assert dt*1e3 < 1e-4, "%s lost %.6f mm in the file" % (sensor, dt*1e3)
        assert dr*1e3 < 1e-4, "%s lost %.6f mrad in the file" % (sensor, dr*1e3)
        # and through the chain, which is the statement that matters
        X_back = cs.sensor_transform(back, A_CHAIN, B_CHAIN)
        assert _error(X_back, X)[0] < 1e-4, _error(X_back, X)

        # The sensor that was NOT calibrated comes back as identity rather
        # than as a missing property, so the file is always complete.
        other = "keyence" if sensor == "azure" else "azure"
        o_xyz, o_rpy = cs.CAL_XACRO_PROPERTIES[other]
        assert np.allclose(cs.properties_to_cal(props[o_xyz], props[o_rpy]),
                           np.eye(4), atol=1e-12)


def test_the_file_always_zeroes_the_keyence_standoff_correction():
    """ONE OWNER FOR THE STANDOFF. keyence_ref_dist_cal and the z of the 6 DOF
    correction are the same physical quantity, and their sum is a clean
    constant offset that looks exactly like a real surface."""
    text = cs.render_calibration_xacro({"keyence": cs.mount_cal(
        KEY_TRUE, A_CHAIN, B_CHAIN)})
    props = cs.parse_calibration_xacro(text)
    assert cs.CAL_REF_DIST_PROPERTY in props
    assert float(props[cs.CAL_REF_DIST_PROPERTY]) == 0.0


def test_re_solving_does_not_compound_onto_the_correction_already_deployed():
    """A calibration run on a calibrated cell must produce the new TOTAL.

    That is why B is read from the correction's CHILD link and not from its
    parent: everything below the child is nominal bracket geometry whatever
    correction is deployed, so the arithmetic never sees the old one. Read it
    from the parent instead and every calibration is relative to the last,
    which is the classic way a correction gets applied twice.
    """
    first = cs.mount_cal(X_TRUE, A_CHAIN, B_CHAIN)
    moved = X_TRUE.dot(cs.se3_exp([0.004, -0.002, 0.006, 0.01, -0.02, 0.005]))
    second = cs.mount_cal(moved, A_CHAIN, B_CHAIN)
    assert np.allclose(cs.sensor_transform(second, A_CHAIN, B_CHAIN), moved,
                       atol=1e-12)
    # It does not matter what was deployed in between.
    assert not np.allclose(first, second, atol=1e-9)


def test_the_generated_file_cannot_carry_yaml_breaking_text():
    """gazebo_ros2_control parses robot_description as YAML, comments included,
    so ': ' or ' #' anywhere in it means no controller manager and a cell that
    comes up with no way to move the arm. A generated file is exactly where
    such a sequence arrives unnoticed, so the generator refuses instead."""
    ok = cs.render_calibration_xacro({}, notes=["measured 2026-09-16T18"])
    assert ": " not in ok and " #" not in ok
    for bad in ("azure: 13.6 mm off", "flat # the bench"):
        try:
            cs.render_calibration_xacro({}, notes=[bad])
        except ValueError:
            pass
        else:
            raise AssertionError("accepted %r" % bad)


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
            print("  FAIL  %s\n        %s" % (name, str(e).splitlines()[0]
                                              if str(e) else type(e).__name__))
        else:
            print("  ok    %s" % name)
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    if failed:
        print("\nfirst failure in full:\n")
        print(failed[0][0])
        raise SystemExit(str(failed[0][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
