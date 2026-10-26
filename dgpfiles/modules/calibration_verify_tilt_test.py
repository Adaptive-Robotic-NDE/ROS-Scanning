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
# File:  calibration_verify_tilt_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests verify_calibration()'s tilt criterion, its brightness gate, and
#   the advice it prints.
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
"""verify_calibration()'s tilt criterion, its brightness gate, and its advice.

Four failure modes are pinned here so they cannot come back. All four are pure
arithmetic or pure string, so all four run on the host with no ROS, no arm and
no cell.

  1. THE BRIGHTNESS GATE MUST ACTUALLY RUN. _cal_patch_brightness() looks up
     the Azure colour topic in the manifest; the key is "color", and asking for
     "rgb" raises and silently returns None. That gate is the one diagnostic
     separating "the Azure cannot measure this target" from "the mount is
     wrong", so a silent None sends the investigation at the mount.

  2. THE TILT CRITERION MUST NOT MEASURE NOISE. A +/-45 mm patch of Azure depth
     cannot resolve 10 mrad: the residual is 0.64 mm rms, correlated over about
     25 mm, and 0.5 mm of correlated structure across 90 mm is already 6 mrad.
     On a level optics table the reported tilt scatters from 4.7 to 48.3 mrad
     across essentially identical setups, so a 15.04 mrad reading against a
     10.0 mrad tolerance is scatter, not a real disagreement.

  3. RETIRED NUMBERS MUST NOT PRINT IN THE ADVICE. A single -3.6 mm dz reading
     over a sprayed specimen is the lone outlier against eight measurements of
     -11.6 to -13.6 mm, and it came from a patch with a 36% return rate.
     Quoting it tells the operator the Azure is fine on real parts when it is
     not.

  4. WIDENING THE AZURE WINDOW LEAVES THE KEYENCE SIDE AS THE WHOLE ERROR. It
     takes the disagreement to 8.33 +/- 1.26 mrad -- a pass with 1.67 mrad of
     headroom against a 10 mrad gate, which is a coin toss. The Keyence's own
     normal off a 40 x 35 mm patch accounts for 8.93 mrad of that, and 8.52 of
     THAT is the slope along its own laser line, which panning across the line
     does not touch. Across clean takes the along-line slope scatters by
     9.4 mrad and the across-line one by 3.4 mrad. So the offsets are a
     five-grab GRID, both slopes come from between-grab heights, and the
     Keyence's sigma is a delete-one-GRAB jackknife -- the unit of independence
     is an arm pose, not a spatial tile.

THE AZURE GRAB IS THE WITNESS for (2), and a recorded one is used directly when
it is present in the fixtures. When it is not -- no fixture checkout, the cell
PC -- a synthetic field built to a measured variogram (0.40 mm nugget, 0.65 mm
sill, 25 mm range) stands in, so the file still has content everywhere.

THE KEYENCE PATCH IS SYNTHETIC HERE AND THAT IS WHY verify_calibration() SAVES
ONE. A capture take records only a normal, an rms, an extent and a count,
so every Keyence number in this file is rebuilt from those four values.
verify_calibration() now writes <take>/keyence_patches/verify.npz with the
points AND the grab each one came from, so pose-to-pose scatter can be measured
rather than assumed.

    python3 -m pytest -q calibration_verify_tilt_test.py
    python3 calibration_verify_tilt_test.py
"""
import ast
import glob
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

import test_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibration_solve as cs                                  # noqa: E402

#: A Keyence plane measured by verify_calibration(), verbatim from a take's
#: scenario.json: 127860 points over 40.0 x 35.5 mm at 0.0743 mm rms, and this
#: normal. Its tilt from horizontal is 8.93 mrad, over a patch built from three
#: arm positions 20 mm apart -- most of the 15.04 mrad the old criterion
#: reported, and why widening the AZURE window does not take the disagreement
#: to zero.
KEYENCE_NORMAL_2026_09_24 = (-0.002682904526389311, -0.008520542900340813,
                             0.9999601003900034)
KEYENCE_N = 127860
KEYENCE_RMS_M = 0.07434870557728587e-3          # the take reports it in mm
KEYENCE_LINE_HALF_M = 0.017750
KEYENCE_OFFSETS_M = (-0.020, 0.0, 0.020)

#: What the take reported, so the test says what it is reproducing.
RECORDED_TILT_MRAD = 15.043509038985656
RECORDED_DZ_MM = -11.62071633922701
RECORDED_AIM = (0.8164168694869862, 0.08199304146367607, 0.9218207651829601)
RECORDED_PATCH_HALF_M = 0.045

#: The colour intrinsics from the same grab. A patch subtends a real number of
#: pixels through these; an identity K makes every patch 0 x 0 and the
#: brightness gate declines for the wrong reason.
AZURE_K = np.array([[973.347168, 0.0, 1026.54639],
                    [0.0, 973.202637, 773.58551],
                    [0.0, 0.0, 1.0]])

LIFTED = ("_cal_fmt_tilt", "_cal_plane_slopes", "_cal_slope_sigma",
          "_cal_tilt_plane", "_cal_tilt_agreement", "_cal_patch_brightness",
          "_cal_verify_why", "_cal_keyence_offsets", "_cal_keyence_reach",
          "_cal_keyence_scatter", "_cal_save_keyence_patch", "_cal_safe_name",
          "_cal_atomic_write")
#: CAL_VERIFY_KEYENCE_OFFSETS is DELIBERATELY NOT HERE. _lift() execs these
#: assignments AFTER the stubs go in, so anything in this tuple overrides what
#: a test asked for; the offsets are the one constant a test needs to vary.
#: dpi_const() reads the shipped value without executing it.
CONSTS = ("CAL_VERIFY_TILT_PATCH", "CAL_VERIFY_TILT_BAND",
          "CAL_VERIFY_TILT_SIGMA_K", "CAL_VERIFY_TILT_BLOCKS",
          "CAL_VERIFY_MIN_GREY", "CAL_VERIFY_TILT_MRAD", "CAL_VERIFY_DZ_MM",
          "CAL_VERIFY_PATCH", "CAL_VERIFY_POINTS", "CAL_VERIFY_KEYENCE_BAND")

FAILURES = []


class Refused(Exception):
    """calibration.dpi's CalibrationRefused."""


def _lift(extra=None):
    """The named defs and constants of calibration.dpi, executed without ROS.

    The .dpi is a function body when dgpy loads it, but its top-level defs are
    top-level statements either way, so one AST pass picks them out.

    A FRESH NAMESPACE EVERY CALL, and `extra` goes in BEFORE the exec. A lifted
    function's globals are the dict it was exec'd into, so stubbing by copying
    the dict afterwards changes nothing -- which is a quiet way to test the
    stubs instead of the code.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    tree = ast.parse(src)
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in LIFTED:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in CONSTS for t in n.targets):
            body.append(n)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    missing = set(LIFTED) - got
    assert not missing, "calibration.dpi has no %s" % (sorted(missing),)
    ns = {"np": np, "math": math, "os": os, "json": json,
          "_cal_geom": lambda: cs, "CalibrationRefused": Refused}
    ns.update(extra or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    ns["_SRC"] = src
    return ns


def dpi_const(name, src=None):
    """The value calibration.dpi SHIPS for a top-level constant.

    Not the one a test stubbed and not the one _lift() happened to exec: the
    literal in the file. ast.literal_eval, so it can only read data.
    """
    tree = ast.parse(src if src is not None
                     else open(os.path.join(HERE, "calibration.dpi")).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError("calibration.dpi defines no %s" % (name,))


NS = _lift()
SRC = NS["_SRC"]

#: calibration.dpi with its whole-line comments removed. The tests that say a
#: string is gone mean gone from what the operator SEES; a retired string may
#: still be named in a comment, and that is not a failure.
CODE = "\n".join(l for l in SRC.splitlines()
                 if not l.lstrip().startswith("#"))


# ------------------------------------------------------------------------------
# The witnesses: one real grab, one synthetic field built to its variogram
# ------------------------------------------------------------------------------

def real_grab_path():
    """The saved verify grab THE RECORDED_* CONSTANTS CAME FROM, or None.

    Not a fixture and not checked in -- 6 MB of depth image per take. Absent on
    a fresh checkout and on the cell PC, where the synthetic stands in.

    IDENTIFIED BY ITS OWN `aim`, NOT BY BEING THE NEWEST. Taking sorted(...)[-1]
    is a live trap: the RECORDED_* constants describe one take (aim x 0.8164),
    and as soon as any later take is added to the fixtures the glob hands these
    tests a DIFFERENT depth image while they go on asserting 15.0435 mrad. A
    test that finds its own input by globbing a mutable data directory is not
    reproducible; matching the aim ties the witness to the numbers that describe
    it.
    """
    root = test_data.root()
    if root is None:
        return None
    want = np.asarray(RECORDED_AIM, float)
    for path in sorted(glob.glob(os.path.join(
            root, "**", "registered.npz"), recursive=True)):
        try:
            with np.load(path, allow_pickle=True) as d:
                aim = np.asarray(d["aim"], float).reshape(-1)[:3]
        except Exception:
            continue
        if aim.size == 3 and float(np.max(np.abs(aim - want))) < 1e-6:
            return path
    return None


def grab_world_points(path, stride=4, max_range=2.0, tilt_mrad=0.0, axis=1):
    """The saved grab as world-frame points, exactly as _cal_azure_surface does.

    `tilt_mrad` rotates the whole cloud about `aim`, which is how a GENUINE
    mount rotation error would present: the entire field tips, at every scale.
    """
    d = np.load(path, allow_pickle=True)
    a = d["depth"].astype(np.float64)
    K, T = np.asarray(d["K"], float), np.asarray(d["T_world_cam"], float)
    v, u = np.mgrid[0:a.shape[0]:stride, 0:a.shape[1]:stride]
    z = a[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < max_range)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    x = (u - K[0, 2])*z/K[0, 0]
    y = (v - K[1, 2])*z/K[1, 1]
    P = np.column_stack([x, y, z]).dot(T[:3, :3].T) + T[:3, 3]
    aim = np.asarray(d["aim"], float)
    if tilt_mrad:
        th = float(tilt_mrad)*1e-3
        c, s = math.cos(th), math.sin(th)
        R = (np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == 0
             else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
        P = (P - aim).dot(R.T) + aim
    return P, aim, float(d["half"])


def _correlated_field(nx, ny, corr_cells, seed):
    """A band-limited random field: white noise on a coarse grid, bilinear up.

    Correlation length is one coarse cell, which is what a variogram range IS.
    Separable np.interp, so no scipy and nothing to tune.
    """
    rng = np.random.RandomState(seed)
    cx = int(max(2, round(nx/float(corr_cells)))) + 1
    cy = int(max(2, round(ny/float(corr_cells)))) + 1
    coarse = rng.normal(size=(cy, cx))
    xs, ys = np.arange(nx), np.arange(ny)
    xc = np.linspace(0, nx - 1, cx)
    yc = np.linspace(0, ny - 1, cy)
    mid = np.empty((cy, nx))
    for j in range(cy):
        mid[j] = np.interp(xs, xc, coarse[j])
    out = np.empty((ny, nx))
    for i in range(nx):
        out[:, i] = np.interp(ys, yc, mid[:, i])
    return out/out.std()


def synthetic_grab(aim=RECORDED_AIM, half_extent=0.22, spacing=0.00226,
                   nugget_m=0.00040, corr_m=0.025, corr_sd_m=0.00051,
                   tilt_mrad=0.0, seed=7):
    """A LEVEL surface measured by something with the recorded grab's variogram.

    nugget 0.40 mm, correlated part 0.51 mm sd over a 25 mm range, total sill
    0.65 mm -- the three numbers measured off the real grab's residual. The
    surface itself is exactly flat, so any tilt this produces is the estimator's
    and not the world's, which is the whole point of the fixture.
    """
    aim = np.asarray(aim, float)
    n = int(2*half_extent/spacing) + 1
    g = np.linspace(-half_extent, half_extent, n)
    X, Y = np.meshgrid(g, g)
    Z = (nugget_m*np.random.RandomState(seed + 1).normal(size=X.shape)
         + corr_sd_m*_correlated_field(n, n, corr_m/spacing, seed))
    if tilt_mrad:
        Z = Z + float(tilt_mrad)*1e-3*X
    return np.column_stack([(X + aim[0]).ravel(), (Y + aim[1]).ravel(),
                            (Z + aim[2]).ravel()])


def keyence_patch(normal=KEYENCE_NORMAL_2026_09_24, at=RECORDED_AIM,
                  n=KEYENCE_N, rms_m=KEYENCE_RMS_M, seed=11):
    """The Keyence patch rebuilt from what the take recorded.

    THREE DISCRETE LINES, not a uniform cloud, because that is what
    CAL_VERIFY_KEYENCE_OFFSETS produces and it is what the jackknife's blocks
    have to cope with: every point in one grab shares one y.
    """
    rng = np.random.RandomState(seed)
    at = np.asarray(at, float)
    sx, sy = -normal[0]/normal[2], -normal[1]/normal[2]
    per = int(n//len(KEYENCE_OFFSETS_M))
    out = []
    for o in KEYENCE_OFFSETS_M:
        x = rng.uniform(-KEYENCE_LINE_HALF_M, KEYENCE_LINE_HALF_M, per)
        y = np.full(per, float(o))
        z = sx*x + sy*y + rng.normal(0.0, rms_m, per)
        out.append(np.column_stack([x + at[0], y + at[1], z + at[2]]))
    return np.vstack(out)


#: What the work cell actually ships, read out of the file rather than assumed.
GRID = [tuple(float(v) for v in o) for o in
        dpi_const("CAL_VERIFY_KEYENCE_OFFSETS")]

#: The pose-to-pose height scatter the grid was sized from, metres. Two
#: independent routes to it in the constant's own comment: the across-line
#: slope repeats to 1.77 mrad over a 40 mm baseline, and the rms about the
#: fitted plane carries s/sqrt(3) of it and is at worst 0.077 mm.
KEYENCE_SCATTER_M = 0.10e-3

#: How much the profile's own 35 mm shape moves the along-line slope, mrad.
#: Measured as the spread of that slope over six clean takes.
KEYENCE_LINE_SHAPE_MRAD = 9.4


def keyence_grid(offsets=None, at=RECORDED_AIM, per=4000,
                 line_half=KEYENCE_LINE_HALF_M, within_rms_m=0.05e-3,
                 scatter_m=KEYENCE_SCATTER_M,
                 line_shape_mrad=KEYENCE_LINE_SHAPE_MRAD,
                 slopes_mrad=(0.0, 0.0), seed=11, lift_m=None):
    """(points, grab labels) as the Keyence patch ACTUALLY ARRIVES.

    ONE LINE PER ARM POSE, and that is the whole modelling claim: every point
    in a grab shares that grab's height error, so the errors are not
    independent per point and a spatial jackknife of them is meaningless.

    The convention matches _verify_ns(), where _cal_tf() is the identity and
    the tool quaternion is unity, so the head's x (the laser line) is world x
    and its y (across the line) is world y. `offsets` are (across, along).

      scatter_m         a per-grab height error. The robot, plus whatever
                        waviness the surface has at the grid's scale.
      line_shape_mrad   a ramp across the 35 mm line, IDENTICAL IN EVERY GRAB.
                        Whatever mixture of head and surface it is, panning
                        across the line cannot average it away -- which is the
                        thing a grid fixes and a wider pan does not.
      lift_m            {grab index: metres} to push one grab off the flat,
                        for the edge-of-the-target check.
    """
    rng = np.random.RandomState(seed)
    at = np.asarray(at, float)
    sx, sy = (v*1e-3 for v in slopes_mrad)
    P, G = [], []
    for i, (ac, al) in enumerate(GRID if offsets is None else offsets):
        x = np.linspace(-line_half, line_half, per) + al
        y = np.full(per, float(ac))
        z = (sx*x + sy*y + line_shape_mrad*1e-3*(x - al)
             + rng.normal(0.0, within_rms_m, per)
             + rng.normal(0.0, scatter_m) + float((lift_m or {}).get(i, 0.0)))
        P.append(np.column_stack([x + at[0], y + at[1], z + at[2]]))
        G.append(np.full(per, i, np.int16))
    return np.vstack(P), np.concatenate(G)


def iid_slope_sigma_mrad(P):
    """The TEXTBOOK least-squares slope uncertainty, for comparison only.

    This is the number the blocked jackknife replaces. It is right for
    independent residuals and wrong here, and the test that says so is the
    justification for not using it.
    """
    P = np.asarray(P, float)
    A = np.column_stack([P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean(),
                         np.ones(P.shape[0])])
    sol, _r, _rk, _s = np.linalg.lstsq(A, P[:, 2], rcond=None)
    resid = P[:, 2] - A.dot(sol)
    s2 = float(resid.dot(resid))/max(1, P.shape[0] - 3)
    C = s2*np.linalg.inv(A.T.dot(A))
    return 1e3*math.hypot(math.sqrt(C[0, 0]), math.sqrt(C[1, 1]))


def patch_of(P, aim, half):
    m = ((np.abs(P[:, 0] - aim[0]) < half) & (np.abs(P[:, 1] - aim[1]) < half))
    return P[m]


def old_tilt_mrad(P, aim, half, key):
    """What the criterion used to compute: the +/-45 mm patch against the Keyence."""
    return cs.plane_agreement(key, patch_of(P, aim, half), at=aim)["tilt_mrad"]


# ------------------------------------------------------------------------------
# 1. THE BRIGHTNESS GATE
# ------------------------------------------------------------------------------

def _brightness_ns(topics):
    """_cal_patch_brightness() with every ROS-facing helper stubbed out."""
    seen = {}

    class Msg(object):
        pass

    def one_message(topic, _cls, **kw):
        seen["topic"] = topic
        return [Msg()]

    def sensor_topics(_sensor="azure", quiet=False):
        return dict(topics) if topics else None

    def sensor_topic(sensor, key, fallback=None):
        t = (topics or {}).get(key)
        if t:
            return t
        raise RuntimeError("no %s.%s topic for cell ex-ur10e-1 mode hardware: "
                           "the manifest declares no %s.%s"
                           % (sensor, key, sensor, key))

    stubs = dict(
        sensor_topics=sensor_topics, sensor_topic=sensor_topic,
        _cal_one_message=one_message,
        _cal_decode_image=lambda m: np.full((1536, 2048, 3), 200, np.uint8),
        _cal_gray=lambda a: np.full(a.shape[:2], 214.0),
        _cal_tf=lambda a, b: np.array([[1., 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 1, 0.55], [0, 0, 0, 1]]))
    sys.modules.setdefault("sensor_msgs", _fake_sensor_msgs())
    return _lift(stubs), seen


def _fake_sensor_msgs():
    import types
    m = types.ModuleType("sensor_msgs")
    m.msg = types.ModuleType("sensor_msgs.msg")
    m.msg.Image = type("Image", (), {})
    sys.modules.setdefault("sensor_msgs.msg", m.msg)
    return m


def test_brightness_reads_the_manifests_color_key():
    """The work cell declares "color", so the lookup must find it under that key."""
    ns, seen = _brightness_ns({"color": "/rgb8/rgb/image_raw",
                               "depth": "/azure/depth_to_rgb/image_raw"})
    why = []
    got = ns["_cal_patch_brightness"](np.array([0.0, 0.0, 0.0]), 0.045,
                                      AZURE_K, "rgb_camera_link",
                                      why=why.append)
    assert why == [], why
    assert got is not None and abs(got - 214.0) < 1e-6, got
    assert seen["topic"] == "/rgb8/rgb/image_raw", seen


def test_brightness_still_prefers_an_explicit_rgb_key():
    """A cell that does declare rgb keeps working, and wins."""
    ns, seen = _brightness_ns({"rgb": "/azure/rgb/image_raw",
                               "color": "/rgb8/rgb/image_raw"})
    why = []
    ns["_cal_patch_brightness"](np.zeros(3), 0.045, AZURE_K,
                                "rgb_camera_link", why=why.append)
    assert seen["topic"] == "/azure/rgb/image_raw", seen
    assert why == [], why


def test_brightness_says_why_when_neither_key_is_declared():
    """No colour at all is still a stated reason, not a silent None."""
    ns, _seen = _brightness_ns({"depth": "/azure/depth_to_rgb/image_raw"})
    why = []
    got = ns["_cal_patch_brightness"](np.zeros(3), 0.045, AZURE_K,
                                      "rgb_camera_link", why=why.append)
    assert got is None
    assert len(why) == 1 and "azure.color" in why[0], why


def test_nothing_asks_for_an_rgb_topic_without_a_color_fallback():
    """The same fault refused at the source level.

    sensor_topic("azure", "rgb") on its own can only ever raise on this cell.
    """
    assert 'sensor_topic("azure", "rgb")' not in CODE
    assert 'sensor_topic("azure", "color")' in CODE


# ------------------------------------------------------------------------------
# 2. THE TILT CRITERION
# ------------------------------------------------------------------------------

def test_the_09_24_grab_reproduces_the_recorded_numbers():
    """Before anything is judged: is this the same data the work cell measured?"""
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    P, aim, half = grab_world_points(path)
    key = keyence_patch(at=aim)
    assert abs(half - RECORDED_PATCH_HALF_M) < 1e-9, half
    a = cs.plane_agreement(key, patch_of(P, aim, half), at=aim)
    assert abs(a["tilt_mrad"] - RECORDED_TILT_MRAD) < 0.05, a["tilt_mrad"]
    assert abs(a["rms_b_mm"] - 0.6417) < 0.01, a["rms_b_mm"]
    assert patch_of(P, aim, half).shape[0] == 1522


def test_the_15_mrad_failure_no_longer_fails():
    """THE HEADLINE, in the three stages it took to get there.

    15.04 mrad on a level table was never a calibration fault. Widening the
    AZURE window took it to 8.33 +/- 1.26 -- a pass, with 1.67 mrad of headroom
    against a 10 mrad gate, which is a coin toss, because 8.9 of it was the
    KEYENCE's own normal off a 40 x 35 mm patch. Both sides are on a real
    baseline now.
    """
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    P, aim, half = grab_world_points(path)
    key = keyence_patch(at=aim)
    old = old_tilt_mrad(P, aim, half, key)
    assert old > 10.0, old                          # the old criterion fails
    tl = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0)
    print("    OLD   patch azure / 3-line keyence   %5.2f mrad, gate 10.00  "
          "FAIL" % (old,))
    print("    then  wide azure  / 3-line keyence   %5.2f +/- %.2f, gate %5.2f  "
          "%s (keyence alone is %.2f mrad from horizontal)"
          % (tl["tilt_mrad"], tl["tilt_sigma_mrad"] or -1,
             tl["tilt_gate_mrad"], "PASS" if tl["tilt_pass"] else "FAIL",
             tl["tilt_keyence"]["tilt_from_horizontal_mrad"]))
    assert tl["tilt_pass"], tl
    assert tl["tilt_mrad"] < 10.0, tl["tilt_mrad"]
    assert tl["tilt_keyence"]["tilt_from_horizontal_mrad"] > 8.0, tl
    # and the old number is kept, so a take can be compared with its ancestors
    assert abs(tl["tilt_patch_mrad"] - old) < 0.05, (tl["tilt_patch_mrad"], old)
    # the azure side is fitted over the wide window, and it is nearly level
    m = tl["tilt_azure"]
    assert m["basis"] == "wide", m
    assert m["extent_mm"][0] > 200.0 and m["extent_mm"][1] > 200.0, m
    assert m["tilt_from_horizontal_mrad"] < 3.0, m

    # NOW: the same Azure grab against the five-grab grid. The Keyence is
    # modelled from THIS take's own recorded numbers -- its +8.52 mrad
    # along-line slope as the profile's own 35 mm shape, and its +2.68 mrad
    # across-line slope as a 0.076 mm per-grab height scatter over the 40 mm
    # baseline it was measured on. There is no saved Keyence patch to use
    # instead, which is exactly what _cal_save_keyence_patch() fixes.
    got = []
    for seed in range(8):
        gkey, g = keyence_grid(at=aim, scatter_m=0.076e-3,
                              line_shape_mrad=8.52, seed=300 + seed)
        got.append(NS["_cal_tilt_agreement"](gkey, P, aim, half, 10.0,
                                             key_groups=g))
    t = float(np.mean([v["tilt_mrad"] for v in got]))
    s = float(np.mean([v["tilt_sigma_mrad"] for v in got]))
    gate = float(np.mean([v["tilt_gate_mrad"] for v in got]))
    kh = float(np.mean([v["tilt_keyence"]["tilt_from_horizontal_mrad"]
                        for v in got]))
    print("    NEW   wide azure  / 5-grab grid      %5.2f +/- %.2f, gate %5.2f  "
          "%s (keyence alone is %.2f mrad from horizontal)"
          % (t, s, gate, "PASS" if all(v["tilt_pass"] for v in got) else "FAIL",
             kh))
    assert all(v["tilt_pass"] for v in got)
    assert t < 3.0, t                  # the disagreement is what is left: 1.7
    assert kh < 2.5, kh                # the 8.9 mrad was the 35 mm baseline
    assert gate < 12.0, gate           # and 3 sigma is near the tolerance
    assert all(v["tilt_sigma_missing"] == [] for v in got), got[0]
    assert all(v["tilt_keyence"]["unit"] == "grab" for v in got), got[0]


def test_a_real_tilt_is_still_caught_on_the_real_grab():
    """Tip the whole cloud and the criterion has to notice, and scale with it."""
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    key = keyence_patch(at=np.asarray(RECORDED_AIM))
    prev = None
    for inject in (0.0, 10.0, 30.0, 60.0):
        P, aim, half = grab_world_points(path, tilt_mrad=inject)
        tl = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0)
        print("    inject %5.1f mrad -> %6.2f +/- %.2f mrad, gate %.2f, %s"
              % (inject, tl["tilt_mrad"], tl["tilt_sigma_mrad"] or -1,
                 tl["tilt_gate_mrad"], "PASS" if tl["tilt_pass"] else "FAIL"))
        if inject == 0.0:
            assert tl["tilt_pass"], tl
        else:
            assert not tl["tilt_pass"], (inject, tl)
            # The injection adds to a disagreement that is already 8.3 mrad in
            # another direction, so the angles add as vectors and not as
            # scalars; what must hold is that the number GROWS with the fault
            # and that a large fault is reported large.
            assert tl["tilt_mrad"] > prev, (inject, tl["tilt_mrad"], prev)
        prev = tl["tilt_mrad"]
    assert prev > 50.0, prev                      # 60 mrad injected, 60 seen


def test_the_patch_cannot_resolve_the_tolerance_and_says_so():
    """Forced back onto the +/-45 mm patch, the gate must widen, not fire.

    This is the fallback path: a flat too small to widen over. The blocked sigma
    is what stops it becoming the old false alarm again.
    """
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    P, aim, half = grab_world_points(path)
    key = keyence_patch(at=aim)
    tl = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0, wide=half)
    print("    patch only: %.2f +/- %.2f mrad, gate %.2f -> %s"
          % (tl["tilt_mrad"], tl["tilt_sigma_mrad"], tl["tilt_gate_mrad"],
             "PASS" if tl["tilt_pass"] else "FAIL"))
    assert tl["tilt_azure"]["basis"] == "patch", tl["tilt_azure"]
    assert tl["tilt_sigma_mrad"] > 4.0, tl              # measured 5.56 mrad
    assert tl["tilt_gate_mrad"] > 10.0, tl              # 3 sigma widens it
    assert tl["tilt_pass"], tl


def test_the_blocked_sigma_is_the_only_honest_one_on_azure_data():
    """Why the textbook covariance is not used: it is six times too small."""
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    P, aim, half = grab_world_points(path)
    Q = patch_of(P, aim, half)
    sx, sy, rms, _n = NS["_cal_plane_slopes"](Q)
    tilt = 1e3*math.hypot(sx, sy)
    sig, B = NS["_cal_slope_sigma"](Q)
    blocked = 1e3*math.hypot(*sig)
    iid = iid_slope_sigma_mrad(Q)
    print("    patch tilt %.2f mrad, rms %.3f mm, blocked sigma %.2f mrad "
          "over %d blocks, iid sigma %.2f mrad" % (tilt, 1e3*rms, blocked, B,
                                                  iid))
    assert B >= 12, B
    assert blocked > 4.0*iid, (blocked, iid)
    # and the blocked one actually covers the measured tilt at 2 sigma, which
    # is the claim "this patch's tilt is consistent with a level surface"
    assert tilt < 2.0*blocked, (tilt, blocked)


def test_the_blocked_sigma_does_not_just_inflate_everything():
    """On genuinely independent residuals it must agree with the textbook one.

    Otherwise the gate would be a licence to pass anything.
    """
    rng = np.random.RandomState(3)
    n = 1600
    x = rng.uniform(-0.045, 0.045, n)
    y = rng.uniform(-0.045, 0.045, n)
    z = 0.9218 + rng.normal(0.0, 0.00064, n)
    Q = np.column_stack([x, y, z])
    sig, B = NS["_cal_slope_sigma"](Q)
    blocked = 1e3*math.hypot(*sig)
    iid = iid_slope_sigma_mrad(Q)
    print("    white noise: blocked %.3f mrad, iid %.3f mrad" % (blocked, iid))
    assert 0.4 < blocked/iid < 2.5, (blocked, iid, B)


def test_a_synthetic_level_surface_cannot_be_resolved_on_the_patch():
    """The same story without the grab, so the file has content everywhere.

    The surface is EXACTLY flat and the measurement has the grab's variogram,
    so every milliradian below is the estimator's. The claim is not about one
    draw of the noise -- it is that the PATCH's uncertainty swallows the 10 mrad
    tolerance and the wide window's does not, which is true of every draw.
    """
    aim = np.asarray(RECORDED_AIM, float)
    P = synthetic_grab(aim=aim)
    key = keyence_patch(at=aim, normal=(0.0, 0.0, 1.0))
    half = RECORDED_PATCH_HALF_M
    tl = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0)
    narrow = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0, wide=half)
    print("    synthetic level surface: patch %.2f +/- %.2f (gate %.2f), "
          "wide %.2f +/- %.2f (gate %.2f)"
          % (narrow["tilt_mrad"], narrow["tilt_sigma_mrad"],
             narrow["tilt_gate_mrad"], tl["tilt_mrad"],
             tl["tilt_sigma_mrad"], tl["tilt_gate_mrad"]))
    assert tl["tilt_azure"]["basis"] == "wide", tl["tilt_azure"]
    assert narrow["tilt_sigma_mrad"] > 2.5, narrow     # cannot see 10 mrad
    assert tl["tilt_sigma_mrad"] < 0.5*narrow["tilt_sigma_mrad"], (tl, narrow)
    assert tl["tilt_pass"] and narrow["tilt_pass"], (tl, narrow)


def test_synthetic_injected_tilt_fails_at_both_window_sizes():
    """A genuine 30 mrad tip is not rescued by either the window or the sigma."""
    aim = np.asarray(RECORDED_AIM, float)
    P = synthetic_grab(aim=aim, tilt_mrad=30.0)
    key = keyence_patch(at=aim, normal=(0.0, 0.0, 1.0))
    half = RECORDED_PATCH_HALF_M
    wide = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0)
    narrow = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0, wide=half)
    print("    injected 30 mrad: wide %.2f (gate %.2f), patch %.2f (gate %.2f)"
          % (wide["tilt_mrad"], wide["tilt_gate_mrad"],
             narrow["tilt_mrad"], narrow["tilt_gate_mrad"]))
    assert not wide["tilt_pass"], wide
    assert not narrow["tilt_pass"], narrow
    assert abs(wide["tilt_mrad"] - 30.0) < 2.0, wide["tilt_mrad"]


def test_a_wider_window_that_is_not_one_flat_is_refused():
    """The widening must not reach past the flat it was aimed at.

    A 100 mm square of flat with the bench 30 mm lower all around it: the wide
    window is mostly bench, so the fit falls back to the patch and says why.
    """
    aim = np.asarray(RECORDED_AIM, float)
    P = synthetic_grab(aim=aim)
    off = ((np.abs(P[:, 0] - aim[0]) > 0.050)
           | (np.abs(P[:, 1] - aim[1]) > 0.050))
    P[off, 2] -= 0.030
    tl = NS["_cal_tilt_agreement"](keyence_patch(at=aim, normal=(0, 0, 1.0)),
                                   P, aim, RECORDED_PATCH_HALF_M, 10.0)
    m = tl["tilt_azure"]
    print("    small flat: basis %s -- %s" % (m["basis"], m.get("why_patch")))
    assert m["basis"] == "patch", m
    assert m.get("why_patch"), m
    # and it is still not a false alarm, because the sigma widened the gate
    assert tl["tilt_gate_mrad"] > 10.0 and tl["tilt_pass"], tl


def test_a_step_inside_the_window_is_rejected_not_fitted():
    """A specimen sitting on the flat must not drag the plane round.

    Same surface, but a 120 x 120 mm block 20 mm proud off to one side. It is
    outside the band, so it never enters the fit.
    """
    aim = np.asarray(RECORDED_AIM, float)
    P = synthetic_grab(aim=aim)
    blk = ((np.abs(P[:, 0] - aim[0] - 0.10) < 0.060)
           & (np.abs(P[:, 1] - aim[1] - 0.10) < 0.060))
    clean = NS["_cal_tilt_plane"](P, aim, RECORDED_PATCH_HALF_M)
    P[blk, 2] += 0.020
    dirty = NS["_cal_tilt_plane"](P, aim, RECORDED_PATCH_HALF_M)
    print("    block of %d points: tilt %.3f -> %.3f mrad, n %d -> %d"
          % (int(blk.sum()), clean[2]["tilt_from_horizontal_mrad"],
             dirty[2]["tilt_from_horizontal_mrad"], clean[2]["n"],
             dirty[2]["n"]))
    assert blk.sum() > 500, int(blk.sum())
    assert dirty[2]["basis"] == "wide", dirty[2]
    assert dirty[2]["n"] < clean[2]["n"], (clean[2]["n"], dirty[2]["n"])
    assert abs(dirty[2]["tilt_from_horizontal_mrad"]
               - clean[2]["tilt_from_horizontal_mrad"]) < 1.0, (clean, dirty)


def test_a_wider_keyence_span_is_not_silently_cropped():
    """CAL_VERIFY_KEYENCE_OFFSETS is a constant somebody will widen.

    The Keyence patch goes into the tilt fit WHOLE. If it were cropped to the
    Azure's +/-45 mm dz window, widening the offsets to +/-60 mm would quietly
    drop the outer two lines -- the exact lines that were added to make the fit
    mean anything -- and the printed count would not say so.
    """
    aim = np.asarray(RECORDED_AIM, float)
    wide_key = keyence_patch(at=aim, normal=(0.0, 0.0, 1.0))
    wide_key = np.vstack([wide_key,
                          wide_key + np.array([0.0, 0.060, 0.0]),
                          wide_key - np.array([0.0, 0.060, 0.0])])
    _s, _sig, m = NS["_cal_tilt_plane"](wide_key, aim, None)
    print("    keyence span %.0f mm, %d of %d points kept"
          % (m["extent_mm"][1], m["n"], wide_key.shape[0]))
    assert m["n"] == wide_key.shape[0], m
    assert m["extent_mm"][1] > 100.0, m


# ------------------------------------------------------------------------------
# 2b. THE KEYENCE SIDE: A GRID, AND A JACKKNIFE WHOSE UNIT IS AN ARM POSE
# ------------------------------------------------------------------------------

def test_the_shipped_offsets_are_a_grid_that_fits_a_150_mm_flat():
    """Four properties of the constant, each of which something depends on."""
    ac = [o[0] for o in GRID]
    al = [o[1] for o in GRID]
    span_ac = 1e3*(max(ac) - min(ac))
    span_al = 1e3*(max(al) - min(al)) + 2e3*KEYENCE_LINE_HALF_M
    print("    %d grabs, %.0f mm across x %.0f mm along (line included)"
          % (len(GRID), span_ac, span_al))
    # four estimates or the jackknife has nothing to say
    assert len(GRID) >= 4, GRID
    # both axes, or the along-line slope is still the profile's own shape
    assert max(ac) > min(ac) and max(al) > min(al), GRID
    # and it has to fit what an operator can put under the arm: A4 is
    # 210 x 297 mm, US Letter 216 x 279, the calibration artifact 160 square
    assert span_ac <= 150.0 and span_al <= 150.0, (span_ac, span_al)
    assert (0.0, 0.0) in GRID, GRID        # continuity with every older take


def test_a_spatial_jackknife_cannot_see_pose_to_pose_scatter():
    """WHY THE KEYENCE NEEDED ITS OWN UNIT OF DELETION.

    Its points arrive as one line per arm pose and every point in a line shares
    that pose's height error. A 4x4 tile cuts a quarter off one line, which
    that error moves not at all -- so the tiled sigma understates the truth,
    the mirror image of the iid sigma's six-fold understatement on the Azure.
    On a three-offset single-axis geometry the grouped jackknife cannot run at
    all, and says so with (None, 0) rather than a number.
    """
    aim = np.asarray(RECORDED_AIM, float)
    legacy = [(-0.020, 0.0), (0.0, 0.0), (0.020, 0.0)]
    fitted, tiled = [], []
    for seed in range(60):
        key, g = keyence_grid(offsets=legacy, at=aim, scatter_m=0.36e-3,
                              line_shape_mrad=0.0, seed=seed)
        sx, sy, _r, _n = NS["_cal_plane_slopes"](key)
        fitted.append((sx, sy))
        sig, _B = NS["_cal_slope_sigma"](key)
        tiled.append(1e3*math.hypot(*sig))
        grouped, B = NS["_cal_slope_sigma"](key, groups=g)
        assert grouped is None and B == 0, (grouped, B)
    A = np.asarray(fitted)
    truth = 1e3*math.hypot(A[:, 0].std(), A[:, 1].std())
    print("    3 offsets, 0.36 mm scatter: the tilt really scatters %.2f mrad, "
          "the tiled jackknife says %.2f, the grouped one refuses"
          % (truth, np.mean(tiled)))
    assert np.mean(tiled) < 0.6*truth, (np.mean(tiled), truth)


def test_the_grouped_sigma_covers_the_spread_it_is_supposed_to():
    """On the shipped grid it must be right, and conservative if it is wrong.

    A gate of 3 sigma is only honest if sigma is not an underestimate. Measured
    over 120 draws: the grouped jackknife reads 1.2 to 1.5 times the actual sd
    of the fitted tilt, which widens the gate rather than narrowing it.
    """
    aim = np.asarray(RECORDED_AIM, float)
    fitted, jk = [], []
    for seed in range(120):
        key, g = keyence_grid(at=aim, scatter_m=0.10e-3, seed=seed)
        sx, sy, _r, _n = NS["_cal_plane_slopes"](key)
        fitted.append((sx, sy))
        sig, B = NS["_cal_slope_sigma"](key, groups=g)
        assert B == len(GRID), (B, len(GRID))
        jk.append(1e3*math.hypot(*sig))
    A = np.asarray(fitted)
    truth = 1e3*math.hypot(A[:, 0].std(), A[:, 1].std())
    print("    grid, 0.10 mm scatter: true sd %.2f mrad, grouped jackknife "
          "%.2f mrad (x%.2f)" % (truth, np.mean(jk), np.mean(jk)/truth))
    assert np.mean(jk) >= truth, (np.mean(jk), truth)
    assert np.mean(jk) < 2.0*truth, (np.mean(jk), truth)


def test_panning_across_the_line_leaves_the_along_line_slope_alone():
    """WHY THE OFFSETS ARE A GRID, and why a longer pan is not the fix.

    A 9.4 mrad ramp across the profile -- whatever mixture of the head's own
    field and the surface's waviness it is -- passes through ANY number of
    offsets taken across the line, because every one of them measures the same
    35 mm of surface. Stepping along the line as well leaves it 3.5% of the
    leverage.
    """
    aim = np.asarray(RECORDED_AIM, float)
    line5 = [(-0.060, 0.0), (-0.030, 0.0), (0.0, 0.0), (0.030, 0.0),
             (0.060, 0.0)]
    legacy = [(-0.020, 0.0), (0.0, 0.0), (0.020, 0.0)]
    got = {}
    for name, offs in (("3 across, +/-20 mm", legacy),
                       ("5 across, +/-60 mm", line5),
                       ("5 on a grid, +/-55 mm", None)):
        h = []
        for seed in range(30):
            key, g = keyence_grid(offsets=offs, at=aim, scatter_m=0.10e-3,
                                  line_shape_mrad=9.4, seed=seed)
            _s, _sig, m = NS["_cal_tilt_plane"](key, aim, None, groups=g)
            h.append(m["tilt_from_horizontal_mrad"])
        got[name] = float(np.mean(h))
        print("    %-22s keyence tilt from horizontal %.2f mrad"
              % (name, got[name]))
    assert got["3 across, +/-20 mm"] > 8.0
    assert got["5 across, +/-60 mm"] > 8.0        # a longer pan changes nothing
    assert got["5 on a grid, +/-55 mm"] < 2.5     # the grid does


def test_the_grid_measures_an_injected_mount_tilt_on_the_real_grab():
    """The whole Azure cloud tipped: the criterion has to report that number.

    NOT "10 mrad still fails", which is what the three-offset geometry did
    because its own bias was 8.9 mrad and anything added to that cleared the
    gate. This measures: what comes out has to be the injected rotation
    combined with the surface's own 1.7 mrad, to about 1.5 mrad, and the sign
    of the injection then decides whether a fault AT the tolerance falls inside
    or outside a gate that IS the tolerance. 15 mrad and up always fail.
    """
    path = real_grab_path()
    if path is None:
        return _skip("no saved grab matching the recorded aim in the fixtures")
    for axis, label in ((0, "about x"), (1, "about y")):
        base = None
        for inject in (0.0, 10.0, 15.0, 30.0, 60.0):
            P, aim, half = grab_world_points(path, tilt_mrad=inject, axis=axis)
            t, gates = [], []
            for seed in range(8):
                key, g = keyence_grid(at=aim, scatter_m=0.10e-3,
                                      line_shape_mrad=8.52, seed=400 + seed)
                tl = NS["_cal_tilt_agreement"](key, P, aim, half, 10.0,
                                               key_groups=g)
                t.append(tl["tilt_mrad"])
                gates.append(tl["tilt_gate_mrad"])
            m, gt = float(np.mean(t)), float(np.mean(gates))
            print("    %s inject %5.1f mrad -> %6.2f mrad, gate %5.2f, %s"
                  % (label, inject, m, gt, "FAIL" if m > gt else "PASS"))
            if inject == 0.0:
                base = m
                assert m < gt, (m, gt)            # a level table must pass
                assert m < 3.0, m                 # and read nearly level
            else:
                # the measurement tracks the fault, combined with the surface's
                # own tilt, so it is bracketed rather than equal to it
                assert abs(inject) - base - 1.5 < m < inject + base + 1.5, \
                    (inject, m, base)
            if inject >= 15.0:
                assert m > gt, (inject, m, gt)    # always caught


def test_the_pose_to_pose_scatter_is_measured_every_run():
    """The number the grid was sized from, re-measured from the data itself.

    0.10 mm is a claim in a comment until a run checks it. _cal_keyence_scatter
    takes the fit's three parameters out of the grab residuals and reports what
    is left, so a cell whose arm or target is worse says so instead of
    inheriting this cell's number.
    """
    aim = np.asarray(RECORDED_AIM, float)
    for s in (0.05e-3, 0.10e-3, 0.30e-3):
        got = []
        for seed in range(40):
            key, g = keyence_grid(at=aim, scatter_m=s, line_shape_mrad=8.52,
                                  seed=seed)
            got.append(NS["_cal_keyence_scatter"](key, g)["scatter_mm"])
        print("    injected %.3f mm -> measured %.3f mm"
              % (1e3*s, float(np.mean(got))))
        assert 0.7*1e3*s < np.mean(got) < 1.4*1e3*s, (s, np.mean(got))
    # and below four grabs it declines rather than guessing
    key, g = keyence_grid(offsets=[(-0.02, 0.0), (0.0, 0.0), (0.02, 0.0)],
                          at=aim, seed=1)
    assert NS["_cal_keyence_scatter"](key, g)["scatter_mm"] is None


def test_a_grab_off_the_edge_of_the_target_is_caught():
    """What a 73 mm reach buys that a 20 mm one did not: a way to miss.

    The plane through "the others" still contains the outlier, so every grab
    then measures off the flat -- which is why the refusal names the worst one.
    """
    aim = np.asarray(RECORDED_AIM, float)
    key, g = keyence_grid(at=aim, scatter_m=0.10e-3, seed=3,
                          lift_m={1: 0.010})
    sc = NS["_cal_keyence_scatter"](key, g)
    print("    grab 1 lifted 10 mm -> off %s, worst grab %d at %+.2f mm"
          % (["%+.2f" % v for v in sc["off_mm"]], sc["worst_grab"],
             sc["worst_off_mm"]))
    assert sc["worst_grab"] == 1, sc
    assert abs(sc["worst_off_mm"] - 10.0) < 2.0, sc
    assert 1 in sc["off_flat"], sc
    # a clean grid is nowhere near the band
    clean = NS["_cal_keyence_scatter"](*keyence_grid(at=aim, seed=3))
    assert clean["off_flat"] == [], clean
    assert max(abs(v) for v in clean["off_mm"]) < 0.5, clean


def test_the_legacy_scalar_offsets_are_still_read():
    """A cell config or a saved parameter block may still carry bare numbers.

    They must produce a shorter baseline, not a TypeError with the arm parked
    over the flat.
    """
    f = NS["_cal_keyence_offsets"]
    assert f((-0.020, 0.0, 0.020)) == [(-0.020, 0.0), (0.0, 0.0), (0.020, 0.0)]
    assert f([[-0.05, -0.05], (0.05, 0.05)]) == [(-0.05, -0.05), (0.05, 0.05)]
    assert f(GRID) == [tuple(o) for o in GRID]
    for bad in ((0.02,), [], [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1)]):
        try:
            f(bad)
        except Refused as e:
            print("    %r refused: %s" % (bad, str(e).splitlines()[0][:70]))
        else:
            raise AssertionError("%r was accepted" % (bad,))


def test_the_tilt_is_never_printed_without_its_sigma():
    f = NS["_cal_fmt_tilt"]
    assert f(8.7, 5.56) == "8.70 +/- 5.56 mrad"
    assert f(8.7, None) == "8.70 mrad (sigma unknown)"
    assert f(None) == "(not measured)"
    assert "_cal_fmt_tilt(" in CODE


def test_the_verdict_uses_the_gate_and_not_the_raw_tolerance():
    """Source level, because the wiring is the part a unit test cannot see."""
    assert 'ok = (abs(a["dz_mm"]) <= tol and tl["tilt_pass"])' in CODE
    # dz still comes from plane_agreement over the +/-45 mm patch
    assert ('a = g.plane_agreement(key, az, at=aim, name_a="keyence", '
            'name_b="azure")') in CODE


# ------------------------------------------------------------------------------
# THE WIRING: verify_calibration() itself, flown against a cell that is not there
# ------------------------------------------------------------------------------
# A unit test of _cal_tilt_agreement() cannot see whether verify_calibration()
# CALLS it, and the call sits inside a broad `except Exception` -- exactly the
# shape that leaves a diagnostic silently dead. So it is flown here, end to
# end, with every ROS-facing helper stubbed.

def _verify_ns(surface_z=0.9218, azure_bias_m=-0.0005, azure_tilt_mrad=0.0,
               offsets=None, keyence_scatter_m=KEYENCE_SCATTER_M,
               keyence_line_shape_mrad=KEYENCE_LINE_SHAPE_MRAD,
               keyence_slopes_mrad=(0.0, 0.0), lift_m=None,
               inside=None, ik=None, cable_refuses=(), take_dir=None):
    """verify_calibration() over a synthetic flat, with no cell behind it.

    THE KEYENCE IS SYNTHESISED PER FLOWN POSE, not handed over as one array.
    The old stub returned the same patch at every offset, which triples a count
    and changes no fit -- fine for a criterion that only looked at the whole
    cloud, and useless now that the answer depends on the grab a point came
    from. Here the line is built around the pose actually commanded, so the
    across/along basis, the grid, and the labelling are all exercised rather
    than assumed.
    """
    said = []
    aim = np.array([0.0, 0.0, surface_z])
    azure = synthetic_grab(aim=aim + np.array([0.0, 0.0, azure_bias_m]),
                           tilt_mrad=azure_tilt_mrad)
    offs = GRID if offsets is None else list(offsets)
    # One height error per GRAB, drawn once: every point in a grab shares it.
    errs = list(np.random.RandomState(29).normal(0.0, keyence_scatter_m,
                                                 max(1, len(offs))))
    flown = {"n": 0, "xyz": None, "labels": []}

    class Take(object):
        dir = take_dir
        results = {}

    import contextlib

    @contextlib.contextmanager
    def capture_scenario(*a, **k):
        yield Take()

    def azure_surface(_d, _i, at, half, tag, take=None, say=None,
                      keep_world=None):
        if keep_world is not None:
            keep_world["pts"] = azure
        return patch_of(azure, at, half), {"tag": tag}

    def fly(xyz, quat, label):
        flown["xyz"] = np.asarray(xyz, float)
        flown["labels"].append(label)
        return None

    def profile_points(_topic, **_kw):
        i, c = flown["n"], flown["xyz"]
        flown["n"] += 1
        per = 4000
        x = np.linspace(-KEYENCE_LINE_HALF_M, KEYENCE_LINE_HALF_M, per)
        sx, sy = (v*1e-3 for v in keyence_slopes_mrad)
        z = (sx*(c[0] + x) + sy*c[1] + keyence_line_shape_mrad*1e-3*x
             + np.random.RandomState(100 + i).normal(0.0, 0.05e-3, per)
             + errs[i % len(errs)] + float((lift_m or {}).get(i, 0.0)))
        return (np.column_stack([c[0] + x, np.full(per, c[1]), c[2] + z]),
                "lj_v7080_optical_frame", 60)

    class Cable(RuntimeError):
        pass

    def wrist3(qs, what, quats=None, tilt=None, **kw):
        for bad in cable_refuses:
            if bad in str(what):
                raise Cable("wrist 3 outside the band for %s" % what)
        return (0.0, 1.0)

    stubs = dict(
        np=np, math=math, os=os, json=json,
        time=type("t", (), {"sleep": staticmethod(lambda s: None)}),
        _cal_geom=lambda: cs, CalibrationRefused=Refused, CableRefused=Cable,
        CAL_SETTLE_S=0.0, CAL_AZURE_DISTANCE=0.55, CAL_VERIFY_DEPTH="registered",
        CAL_AZURE_RAW_TOPICS={}, CAL_KEYENCE_REF_DIST=0.080,
        CAL_KEYENCE_RANGE=(0.057, 0.103),
        CAL_VERIFY_KEYENCE_OFFSETS=tuple(offs),
        _CAL={}, taught_points={"cal_flat": aim},
        _cap_point=lambda n: aim,
        _cap_inside=(lambda pts: True) if inside is None else inside,
        _cal_ik=(lambda xyz, quat, seed=None: [0.0]*6) if ik is None else ik,
        check_wrist3=wrist3,
        sensor_topics=lambda s="azure", quiet=False: {
            "depth": "/d", "camera_info": "/i", "color": "/c"},
        sensor_topic=lambda s, k, fallback=None: "/%s" % k,
        _cal_intrinsics=lambda t: (AZURE_K, None, 2048, 1536, "rgb_camera_link"),
        _cal_tool_quat=lambda *a: np.array([0.0, 0.0, 0.0, 1.0]),
        _cal_pose_for_sensor=lambda at, q, d, f: np.asarray(at, float),
        _cal_fly_or_refuse=fly,
        _cal_tf=lambda a, b: np.eye(4),
        _cal_profile_points=profile_points,
        _cal_azure_surface=azure_surface,
        # the REAL _cal_patch_brightness runs (it is lifted), so its ROS
        # helpers are stubbed rather than the function
        _cal_one_message=lambda *a, **k: [object()],
        _cal_decode_image=lambda m: np.full((1536, 2048, 3), 200, np.uint8),
        _cal_gray=lambda a: np.full(a.shape[:2], 214.0),
        _cal_say=said.append,
        capture_scenario=capture_scenario)
    ns = _lift_with(("verify_calibration",), stubs)
    ns["_SAID"] = said
    ns["_FLOWN"] = flown
    ns["_TAKE"] = Take
    return ns


def _lift_with(extra_names, stubs):
    """_lift(), but pulling in extra top-level defs by name."""
    global LIFTED
    keep = LIFTED
    try:
        LIFTED = tuple(set(LIFTED) | set(extra_names))
        return _lift(stubs)
    finally:
        LIFTED = keep


def test_verify_calibration_actually_uses_the_new_tilt():
    ns = _verify_ns()
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"] if l.strip()))
    assert "tilt_error" not in out, out.get("tilt_error")
    assert out["tilt_sigma_mrad"] is not None, out
    assert out["tilt_azure"]["basis"] == "wide", out["tilt_azure"]
    assert "+/-" in said and "mrad" in said, said
    assert "from horizontal keyence" in said, said
    assert out["pass"], out
    assert "PASS" in said, said
    # dz is still the +/-45 mm patch answer, to the millimetre
    assert abs(out["dz_mm"] + 0.5) < 0.2, out["dz_mm"]


def test_verify_calibration_fails_on_height_with_the_new_advice():
    """A real height disagreement, flown end to end: FAIL on height, with the
    advice that replaces the retired -3.6 mm figure."""
    ns = _verify_ns(azure_bias_m=-0.0116)
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"]
                                 if "FAIL" in l or "AZURE IS" in l
                                 or "standing" in l or "adapter" in l))
    assert not out["pass"] and out["tilt_pass"], out
    assert "FAIL on height" in said, said
    assert "-12.45 mm" in said and "adapter" in said, said
    assert "-3.6 mm" not in said, said


def test_verify_calibration_fails_on_a_real_tilt_and_names_it():
    ns = _verify_ns(azure_bias_m=0.0, azure_tilt_mrad=40.0)
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"] if "FAIL" in l
                                 or "TILT" in l or "HEIGHTS" in l))
    assert not out["pass"], out
    assert "FAIL on tilt" in said, said
    assert abs(out["tilt_mrad"] - 40.0) < 3.0, out["tilt_mrad"]
    # and the advice is the tilt one, not the depth one
    assert "adapter" not in said.split("FAIL on tilt")[-1], said


def test_the_keyence_patch_is_saved_with_its_grouping():
    """F2: the Keyence side has to be re-derivable offline, like the Azure's.

    THE AZURE GRAB IS WHY THIS FILE EXISTS -- every number in the 15.04 mrad
    investigation came out of <take>/azure_grabs/registered.npz on a host with
    no cell. The Keyence patch was not saved, so the same question about IT had
    to be answered from a synthetic patch rebuilt out of a recorded normal, an
    rms and a count. This test loads the file back and re-measures the
    pose-to-pose scatter from it, which is the thing that could not be done.
    """
    d = tempfile.mkdtemp(prefix="verify_take_")
    try:
        ns = _verify_ns(take_dir=d, keyence_scatter_m=0.22e-3)
        out = ns["verify_calibration"](point="cal_flat")
        path = os.path.join(d, "keyence_patches", "verify.npz")
        assert os.path.exists(path), sorted(os.listdir(d))
        assert not glob.glob(os.path.join(d, "**", "*.part"), recursive=True)
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        print("    %s: %d points, %d grabs, %.0f kB"
              % (os.path.relpath(path, d), z["points"].shape[0],
                 meta["grabs"], os.path.getsize(path)/1024.0))
        for k in ("points", "group", "offsets", "T_world_sensor", "probe_end",
                  "aim", "across", "along", "meta"):
            assert k in z, (k, sorted(z))
        assert z["points"].shape == (z["group"].shape[0], 3)
        assert sorted(set(int(v) for v in z["group"])) == list(range(len(GRID)))
        assert z["offsets"].shape == (len(GRID), 2)
        assert z["T_world_sensor"].shape == (len(GRID), 4, 4)
        assert meta["labels"][0] == "+0,+0 mm", meta["labels"]
        # AND THE POINT OF IT: the scatter comes back out of the file, without
        # the take, the work cell, or an assumption about the rms
        again = NS["_cal_keyence_scatter"](np.asarray(z["points"], float),
                                           z["group"])
        print("      re-measured offline: scatter %.3f mm, off %s"
              % (again["scatter_mm"],
                 ["%+.2f" % v for v in again["off_mm"]]))
        assert abs(again["scatter_mm"]
                   - out["keyence_scatter"]["scatter_mm"]) < 0.01, again
        assert again["scatter_mm"] > 0.05, again
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_offsets_are_screened_before_the_arm_moves():
    """A grab the cables refuse is dropped, not discovered on the fourth move.

    The old geometry reached 20 mm and every offset was load-bearing, so
    _cal_fly_or_refuse() retried rather than skipped. The grid reaches 73 mm,
    four grabs are enough, and a leg that cannot be flown should cost precision
    the jackknife then reports -- not a two-minute run that ends with nothing.
    """
    ns = _verify_ns(cable_refuses=("+55,+55",))
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"]
                                 if "DROPPED" in l or "grabs:" in l))
    assert out["keyence_dropped"] == [["+55,+55 mm",
                                       "refused by the cable envelope"]], out
    assert "DROPPED: refused by the cable envelope" in said, said
    # screened BEFORE flying: the refused pose was never commanded
    legs = [l for l in ns["_FLOWN"]["labels"] if "keyence" in l]
    assert len(legs) == len(GRID) - 1, legs
    assert not any("+55,+55" in l for l in legs), legs
    # four grabs on both axes still produce a sigma
    assert out["keyence_grabs"] == len(GRID) - 1, out["keyence_grabs"]
    assert out["tilt_sigma_missing"] == [], out
    assert out["tilt_sigma_mrad"] is not None and out["pass"], out


def test_a_grid_reduced_to_one_axis_keeps_dz_and_says_the_tilt_is_blind():
    """The trade this makes deliberately: never lose the height to save the tilt.

    dz is the expensive number: a height error of this size costs 26-39% of a
    plate. A grid cut down to one axis still measures it; the tilt then has no
    uncertainty, and that
    is said out loud rather than shown as a gate that quietly reverted to the
    bare tolerance.
    """
    # everything at across >= 0 is outside this cell's work volume
    ns = _verify_ns(inside=lambda pts: float(np.asarray(pts)[0][1]) < 0.030)
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"]
                                 if "DROPPED" in l or "NO UNCERTAINTY" in l
                                 or "only" in l or "AZURE IS" in l))
    assert len(out["keyence_dropped"]) == 2, out["keyence_dropped"]
    assert all(w == "outside the work volume"
               for _l, w in out["keyence_dropped"]), out["keyence_dropped"]
    assert "outside the work volume" in said, said
    # the height is still measured, and still right
    assert abs(out["dz_mm"] + 0.5) < 0.3, out["dz_mm"]
    # the tilt is not: three grabs cannot produce four jackknife estimates
    assert out["tilt_sigma_missing"] == ["keyence"], out
    assert out["tilt_gate_mrad"] == out["tol_mrad"], out
    assert "NO UNCERTAINTY FOR THE KEYENCE TILT" in said, said
    assert "only 3 grab(s)" in said, said


def test_the_work_volume_is_hard_and_the_ik_screen_is_not():
    """A screen that can answer nothing must drop nothing, and say so.

    _cal_ik() returns None with no live ROS behind it -- a rehearsal, a unit
    test, a cell whose driver is not up. Treating that as "unreachable" would
    refuse every grab and blame the taught point.
    """
    ns = _verify_ns(ik=lambda xyz, quat, seed=None: None)
    out = ns["verify_calibration"](point="cal_flat")
    said = "\n".join(ns["_SAID"])
    print("    " + "\n    ".join(l for l in ns["_SAID"] if "UNSCREENED" in l))
    assert out["keyence_dropped"] == [], out
    assert out["keyence_grabs"] == len(GRID), out
    assert "UNSCREENED" in said, said
    assert out["pass"], out


def test_verify_refuses_when_a_grab_walks_off_the_target():
    """Nothing is compared, and the message says how big a flat this needs."""
    ns = _verify_ns(lift_m={1: 0.010})
    try:
        ns["verify_calibration"](point="cal_flat")
    except Refused as e:
        text = str(e)
        print("    " + text[:300])
    else:
        raise AssertionError("a grab 10 mm off the flat was fitted anyway")
    assert "-55,-55 mm" in text, text
    assert "same surface" in text, text
    assert "150 mm across" in text, text
    assert "CAL_VERIFY_KEYENCE_OFFSETS" in text, text


# ------------------------------------------------------------------------------
# 3. THE ADVICE TEXT
# ------------------------------------------------------------------------------

def test_the_retired_minus_3_6_mm_is_gone():
    """It told the operator the Azure was fine on real parts. It is an outlier."""
    assert "-3.6 mm" not in CODE
    assert "3.6 mm low" not in CODE
    assert "sprayed specimen" not in CODE


def test_the_advice_states_the_measured_bias_and_where_the_fix_lives():
    why = NS["_cal_verify_why"]({"dz_mm": -11.62, "tol_mm": 2.0,
                                 "tol_mrad": 10.0, "tilt_mrad": 8.35},
                                bright=214.0)
    text = " ".join(why)
    print("    " + "\n    ".join(why))
    assert "-12.45 mm" in text, text
    assert "8 measurements" in text, text
    assert "touch-off" in text and "Keyence" in text, text
    assert "adapter" in text, text
    assert len(why) <= 4, len(why)               # the operator reads these


def test_a_tilt_only_failure_does_not_get_depth_advice():
    """Every other line is about a height error, so none of them applies."""
    why = NS["_cal_verify_why"]({"dz_mm": -0.4, "tol_mm": 2.0, "tol_mrad": 10.0,
                                 "tilt_mrad": 18.0, "tilt_sigma_mrad": 1.3,
                                 "tilt_gate_mrad": 10.0}, bright=214.0)
    text = " ".join(why)
    print("    " + "\n    ".join(why))
    assert len(why) == 1, why
    assert "HEIGHTS agree" in text, text
    assert "adapter" not in text and "-12.45" not in text, text


def test_a_dark_patch_is_still_the_first_answer():
    why = NS["_cal_verify_why"]({"dz_mm": -18.1, "tol_mm": 2.0,
                                 "tol_mrad": 10.0}, bright=103.0)
    text = " ".join(why)
    assert "grey 103" in text, text
    assert "18.1 mm low" in text, text
    assert "adapter" not in text, text           # it returns before that


# ------------------------------------------------------------------------------

def _skip(why):
    print("  skip (%s)" % why)
    return None


def main():
    ns = dict(globals())
    for name in sorted(n for n in ns if n.startswith("test_")):
        try:
            print("%s" % name)
            ns[name]()
            print("  ok")
        except AssertionError as e:                           # noqa: PERF203
            FAILURES.append("%s: %s" % (name, e))
            print("  FAIL %s" % (e,))
    print("\n%d check(s) failed" % len(FAILURES))
    for f in FAILURES:
        print("  - %s" % f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
