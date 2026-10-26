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
# File:  ec_plate_model_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Host tests for the pose-to-volts model: the tilt decomposition is
#   checked against ecsim's own pose convention, and the instrument
#   behaviour against what the real one does.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-04:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""Probe pose in, instrument volts out, with the geometry checked not assumed.

NO ROS, NO CELL, NO ARM, NO spatialnde2, NO Gazebo. Poses are quaternions built
here, so these tests check that the model RECOVERS the pose it was given.

THE TEST THAT MATTERS MOST is test_tilt_decomposition_matches_ecsim. The whole
model rests on reading a standoff and two tilt angles out of a quaternion in
the same convention ecsim used when it filled the table, and a sign error there
would invert every flaw signature while leaving every amplitude plausible. So
the decomposition is checked against ecsim's own probe_pose() over random
poses, not against this module's idea of itself.

The rest divide into the geometry (which flaw is where, in whose frame) and the
instrument (what nulling does, what gain does, what the rails do).

    python3 -m pytest -q ec_plate_model_test.py
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
_ECSIM = os.environ.get("ECSIM_DIR") or os.path.expanduser("~/ecsim")
if os.path.isdir(os.path.join(_ECSIM, "ecsim")) and _ECSIM not in sys.path:
    sys.path.append(_ECSIM)

pytest.importorskip("h5py")

import ec_plate_model as P                                  # noqa: E402
import ec_response_table as R                               # noqa: E402
import make_ec_stub_table as STUB                           # noqa: E402
import make_ec_test_plate as PLATE                          # noqa: E402

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def quat_about(axis, angle):
    """(qx, qy, qz, qw) for a rotation about a unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a/np.linalg.norm(a)
    s = math.sin(0.5*angle)
    return (float(a[0]*s), float(a[1]*s), float(a[2]*s), math.cos(0.5*angle))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("ecsim_consumer")
    STUB.write(str(d))
    PLATE.main(["--out-dir", str(d)])
    return str(d)


@pytest.fixture(scope="module")
def table(built):
    t = R.ECResponseTable(os.path.join(built, "al6061_2mhz_response.h5"))
    yield t
    t.close()


@pytest.fixture
def model(table, built):
    return P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")),
                          rng=np.random.default_rng(7))


def notch(model, name):
    return [f for f in model.flaws if f["name"] == name][0]


# ---------------------------------------------------------------------------
# The geometry, against the convention that filled the table
# ---------------------------------------------------------------------------

def test_tilt_decomposition_matches_ecsim():
    """Standoff and the two tilts, recovered exactly from ecsim's own poses.

    THE TEST THE WHOLE MODEL RESTS ON. ecsim built the table with
    R = Rz(yaw) Rx(tilt_x) Ry(tilt_y) and tabulated the field against
    tilt_parallel = tilt_y (about the split line, the flaw-like one) and
    tilt_perpendicular = tilt_x. Reading those back out of a world-frame
    quaternion is where a sign error would live, and a wrong sign on the
    parallel tilt inverts every flaw signature without changing its size --
    so this is checked against probe_pose() itself rather than against any
    algebra written here.
    """
    geometry = pytest.importorskip("ecsim.geometry")
    rng = np.random.default_rng(11)
    worst_par = worst_perp = 0.0
    for _ in range(500):
        yaw = float(rng.uniform(-math.pi, math.pi))
        tx = float(rng.uniform(-0.25, 0.25))        # ecsim tilt_x  = perpendicular
        ty = float(rng.uniform(-0.25, 0.25))        # ecsim tilt_y  = parallel
        # Exactly the pose ecsim's scan_positions() builds. Note it passes -yaw.
        rot = geometry.probe_pose(0.0, 0.0, 0.0, -yaw, tx, ty).rotation
        n = np.array([0.0, 0.0, 1.0])               # surface outward normal
        ex, ey, ez = rot[:, 0], rot[:, 1], rot[:, 2]
        n_probe = np.array([n @ ex, n @ ey, n @ ez])
        got_par = math.atan2(-n_probe[0], n_probe[2])
        got_perp = math.atan2(n_probe[1], math.hypot(n_probe[0], n_probe[2]))
        worst_par = max(worst_par, abs(got_par - ty))
        worst_perp = max(worst_perp, abs(got_perp - tx))
    assert worst_par < 1e-12, "parallel tilt off by %.3e rad" % worst_par
    assert worst_perp < 1e-12, "perpendicular tilt off by %.3e rad" % worst_perp


def test_model_reads_back_the_tilt_it_was_given(model, built):
    """The same decomposition, this time through the model's own entry point."""
    f = notch(model, "L3_D0")
    for deg in (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0):
        q = quat_about((0.0, 1.0, 0.0), math.radians(deg))    # about probe y
        _, par, perp = model.pose_parameters(f["centre"], q)
        assert math.degrees(par) == pytest.approx(deg, abs=1e-9)
        assert math.degrees(perp) == pytest.approx(0.0, abs=1e-9)
        q = quat_about((1.0, 0.0, 0.0), math.radians(deg))    # about probe x
        _, par, perp = model.pose_parameters(f["centre"], q)
        assert math.degrees(perp) == pytest.approx(deg, abs=1e-9)
        assert math.degrees(par) == pytest.approx(0.0, abs=1e-9)


def test_the_spring_floors_the_standoff_at_zero(model):
    """Driving the tip below the surface does not give a negative standoff.

    The probe is spring-loaded: the arm commands the tip past the measured
    surface by the contact preload, the spring takes up the difference, and
    the wear face rides on the part. A measured 0.45 mm of preload against a
    perfect mesh is therefore zero standoff and not minus 0.45 mm -- which the
    table would refuse, its lift-off axis starting at zero.
    """
    f = notch(model, "L3_D0")
    for below in (0.0, 0.0001, 0.00045, 0.005):
        p = f["centre"] - np.array([0.0, 0.0, below])
        s, _, _ = model.pose_parameters(p, IDENTITY)
        assert s == 0.0
    for above in (0.0001, 0.0005, 0.001):
        p = f["centre"] + np.array([0.0, 0.0, above])
        s, _, _ = model.pose_parameters(p, IDENTITY)
        assert s == pytest.approx(above)


def test_flaw_world_placement_follows_the_plate(table, built, tmp_path):
    """A yawed, moved plate moves its notches and turns their long axes.

    The manifest describes the plate in the PLATE's frame and records where it
    was put, because a flaw table in world coordinates is a property of one
    plate on one bench. This is the one transform that makes that work.
    """
    out = str(tmp_path/"yawed")
    PLATE.main(["--out-dir", out, "--place", "0.9", "-0.15", "0.96",
                "--yaw-deg", "30"])
    m = P.ECPlateModel(table, P.load_plate(os.path.join(out, "flaws.json")))
    f = notch(m, "L3_D0")
    a = math.radians(30.0)
    c, s = math.cos(a), math.sin(a)
    want = np.array([0.9 + c*0.0525 - s*0.030, -0.15 + s*0.0525 + c*0.030, 0.96])
    assert np.allclose(f["centre"], want)
    # the notch was at 90 degrees in the plate frame, so 120 in the world
    assert math.degrees(math.atan2(f["axis"][1], f["axis"][0])) == pytest.approx(120.0)
    assert m.surface_z == pytest.approx(0.96)


def test_the_meshes_are_in_world_coordinates(built):
    """sim_specimen.dpi spawns a model with an IDENTITY pose, so the mesh must
    already be where the plate is -- otherwise the simulated camera and the
    simulated profiler see the plate in one place and this model puts the
    flaws in another, with nothing anywhere saying so."""
    d = json.load(open(os.path.join(built, "flaws.json")))
    assert d["plate"]["meshes_frame"] == "world"
    with open(os.path.join(built, "ec_test_plate.ply"), "rb") as f:
        raw = f.read()
    head = raw[:raw.index(b"end_header\n")].decode()
    nv = int([l for l in head.splitlines()
              if l.startswith("element vertex")][0].split()[-1])
    v = np.frombuffer(raw[raw.index(b"end_header\n") + 11:][:nv*48],
                      dtype="<f8").reshape(nv, 6)
    assert v[:, 2].max() == pytest.approx(d["plate"]["place_xyz_m"][2])
    cx = 0.5*(v[:, 0].min() + v[:, 0].max())
    assert cx == pytest.approx(d["plate"]["place_xyz_m"][0], abs=1e-9)


def test_yaw_seen_by_the_table_is_relative_to_the_coil_axis(model):
    """Turning the probe turns every notch's apparent orientation.

    A split-D answers to cracks across its split and is nearly blind to one
    along it, so this angle decides whether a notch is seen at all -- and it
    is measured from the PROBE, not from the world.
    """
    f = notch(model, "L3_D0")               # long axis along world +y
    p = f["centre"] + np.array([0.002, 0.0, 0.0])
    _, info = model.impedance(p, IDENTITY, detail=True)
    assert info["flaws"][0]["yaw_deg"] == pytest.approx(90.0, abs=1e-6)
    # rotate the probe 30 degrees about its own axis: the notch is now at 60
    q = quat_about((0.0, 0.0, 1.0), math.radians(30.0))
    _, info = model.impedance(f["centre"] + np.array([0.0017, 0.001, 0.0]),
                              q, detail=True)
    assert info["flaws"][0]["yaw_deg"] == pytest.approx(60.0, abs=1e-6)


def test_the_coil_axis_offset_rotates_every_notch(table, built):
    """COIL_AXIS_DEG is unknown for the real probe, so it must be a knob that
    visibly does something -- and what it does is turn every notch's apparent
    yaw by the same amount."""
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    a = P.ECPlateModel(table, plate, coil_axis_deg=0.0)
    b = P.ECPlateModel(table, plate, coil_axis_deg=25.0)
    f = notch(a, "L3_D0")
    p = f["centre"] + np.array([0.002, 0.0, 0.0])
    _, ia = a.impedance(p, IDENTITY, detail=True)
    _, ib = b.impedance(p, IDENTITY, detail=True)
    assert ia["flaws"][0]["yaw_deg"] - ib["flaws"][0]["yaw_deg"] \
        == pytest.approx(25.0, abs=1e-6)


def test_off_the_footprint_reads_nothing(model):
    """There is no part under the probe off the end of the plate.

    Between passes the arm lifts clear and runs off the plate, and a model that
    answered those poses with a full-amplitude tilt reading would fill the
    trace store with numbers that look like data.
    """
    assert model.impedance(np.array([0.3, 0.3, 0.95]), IDENTITY) == 0.0
    assert model.off_part == 1
    # just inside the edge, it reads
    inside = model.place_xyz + np.array([model.half_x - 0.001, 0.0, 0.0])
    assert abs(model.impedance(inside, IDENTITY)) > 0.0


def test_the_probe_axis_sign_is_checked_not_trusted(model):
    """A probe whose axis points INTO the part is counted and reported.

    Read off the URDF, so it can be wrong for new tooling -- and if it is,
    every parallel tilt has the wrong sign, which inverts every flaw signature
    while leaving the amplitudes plausible. Checkable at runtime, so it is
    checked on every query.
    """
    f = notch(model, "L3_D0")
    model.impedance(f["centre"], IDENTITY)
    assert model.axis_sign_disagreements == 0
    model.impedance(f["centre"], quat_about((1.0, 0.0, 0.0), math.pi))
    assert model.axis_sign_disagreements == 1
    assert model.report()["axis_sign_disagreements"] == 1


def test_flipping_the_axis_sign_keeps_the_frame_right_handed(table, built):
    """Both settings give a right-handed frame, so a wrong sign points the
    wrong way without ALSO turning the frame inside out -- two bugs being
    harder to reason about than one."""
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    for sign in (+1, -1):
        m = P.ECPlateModel(table, plate, probe_axis_sign=sign)
        ex, ey, ez = m.probe_frame(quat_about((0.0, 1.0, 0.0), 0.3))
        assert np.allclose(np.cross(ex, ey), ez, atol=1e-12)


# ---------------------------------------------------------------------------
# The impedance
# ---------------------------------------------------------------------------

def test_the_baseline_is_included_not_left_out(model):
    """Over good material, square on, the reading is the flaw-free trajectory.

    The baseline is not a correction to the flaw response: with no flaw under
    the probe it is the ENTIRE signal. Leaving it out would make a tilted scan
    look clean, which is the opposite of the finding.
    """
    f = notch(model, "L3_D0")
    good = f["centre"] + np.array([0.0, 0.015, 0.0])    # well clear of any notch
    z, info = model.impedance(good, IDENTITY, detail=True)
    assert info["flaws"] == []
    assert z == pytest.approx(model.table.baseline(0.0, 0.0, 0.0))
    assert abs(z) > 0.0


def test_a_pass_over_a_notch_traces_a_loop(model):
    """The flaw contribution is antisymmetric about the notch centre.

    The double lobe, through the whole geometry chain this time rather than
    straight out of the table: positive, through a null on the notch, to
    negative. Reversing the sense of dx would invert it, and nothing
    downstream would notice.
    """
    f = notch(model, "L3_D0")
    base = model.table.baseline(0.0, 0.0, 0.0)
    for dx in (0.001, 0.002, 0.003):
        a = model.impedance(f["centre"] + np.array([+dx, 0.0, 0.0]), IDENTITY) - base
        b = model.impedance(f["centre"] + np.array([-dx, 0.0, 0.0]), IDENTITY) - base
        assert abs(a) > 1e-4
        assert abs(a + b) <= 1e-6*abs(a)
    on_it = model.impedance(f["centre"], IDENTITY) - base
    assert abs(on_it) < 1e-9


def test_tilt_swamps_the_notch(model):
    """A few degrees of tilt reads far larger than the biggest notch does.

    The finding this whole simulation exists to let us study, asserted end to
    end: if a change ever made the notch the larger of the two, every
    conclusion drawn in simulation about separating them would be worthless.
    """
    f = notch(model, "L3_D0")
    base = model.table.baseline(0.0, 0.0, 0.0)
    notch_peak = max(
        abs(model.impedance(f["centre"] + np.array([dx, 0.0, 0.0]), IDENTITY) - base)
        for dx in np.linspace(-0.004, 0.004, 33))
    good = f["centre"] + np.array([0.0, 0.015, 0.0])
    tilted = abs(model.impedance(good, quat_about((0.0, 1.0, 0.0),
                                                 math.radians(2.0))) - base)
    assert tilted > 10.0*notch_peak


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

def test_volts_are_raw_and_carry_the_standing_offset(model):
    """Nothing is conditioned here. The offset null_here() removes is present.

    A simulator that handed over already-nulled, already-rotated numbers could
    not be used to exercise either step, and exercising them without an
    instrument is most of the reason this exists.
    """
    f = notch(model, "L3_D0")
    good = f["centre"] + np.array([0.0, 0.015, 0.0])
    xs, ys = model.volts(good, IDENTITY, n=8, noise_v=0.0)
    assert len(xs) == 8 and len(ys) == 8
    z = model.table.baseline(0.0, 0.0, 0.0)
    phase = math.radians(model.instrument_phase_deg)
    want = z*complex(math.cos(phase), math.sin(phase))*model.v_per_ohm
    assert xs[0] == pytest.approx(model.base_volts[0] + want.real)
    assert ys[0] == pytest.approx(model.base_volts[1] + want.imag)


def test_nulling_removes_the_operating_point_but_not_the_offset(table, built):
    """Two nulls, two jobs, and neither replaces the other.

    The hardware null moves the instrument's operating point in OHMS, ahead of
    the gain stage -- which is what makes a usable gain possible. It does NOT
    remove the analog chain's voltage offset; that is the software null,
    /ec/cond/null and null_here(). This is the real instrument's arrangement
    and the test exists because collapsing the two is the obvious shortcut.
    """
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    pose = {"v": None}
    m = P.ECPlateModel(table, plate, pose_fn=lambda: pose["v"])
    f = notch(m, "L3_D0")
    good = f["centre"] + np.array([0.0, 0.015, 0.0])
    pose["v"] = (good, IDENTITY)
    out = m.null()
    assert out["off_part"] is False
    assert out["ohms"] == pytest.approx(table.baseline(0.0, 0.0, 0.0))
    # on the spot it was nulled, the reading is now the bare offset
    xs, ys = m.volts(good, IDENTITY, n=1, noise_v=0.0)
    assert xs[0] == pytest.approx(m.base_volts[0])
    assert ys[0] == pytest.approx(m.base_volts[1])
    # and the notch still shows
    xs2, _ = m.volts(f["centre"] + np.array([0.002, 0.0, 0.0]), IDENTITY,
                     n=1, noise_v=0.0)
    assert abs(xs2[0] - m.base_volts[0]) > 0.0


def test_nulling_in_the_air_says_so(table, built):
    """The bench mistake, reproduced rather than prevented.

    An instrument balanced in air has its operating point in the wrong place
    and no software null repairs it. The model lets it happen -- it has to, or
    the scan's own refusal to run un-nulled could never be exercised -- and
    reports that the probe was off the part when it did.
    """
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    pose = {"v": (np.array([0.2, 0.2, 1.2]), IDENTITY)}
    m = P.ECPlateModel(table, plate, pose_fn=lambda: pose["v"])
    out = m.null()
    assert out["off_part"] is True
    assert out["ohms"] == 0.0


def test_nulling_with_no_pose_refuses(table, built):
    """Before TF has produced anything there is nothing to null against."""
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    m = P.ECPlateModel(table, plate, pose_fn=lambda: None)
    with pytest.raises(RuntimeError, match="pose"):
        m.null()
    m2 = P.ECPlateModel(table, plate)
    with pytest.raises(RuntimeError, match="pose source"):
        m2.null()


def test_the_rails_clip_and_are_counted(table, built):
    """The digitiser rails are real and a real pass has hit them.

    In the measured tilt ladder the +6 degree pass sat at exactly -5.0000 for
    all 64 samples, standard deviation exactly zero, which made it look like
    the quietest pass in the set. ec_liftoff.saturation() exists to catch
    that, and it can only be exercised against a simulator that can clip.
    """
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    m = P.ECPlateModel(table, plate, v_per_ohm=5.0)
    f = notch(m, "L3_D0")
    good = f["centre"] + np.array([0.0, 0.015, 0.0])
    xs, ys = m.volts(good, quat_about((0.0, 1.0, 0.0), math.radians(10.0)),
                     n=16, noise_v=0.0)
    assert np.all(np.abs(xs) <= m.rail_volts + 1e-12)
    assert np.all(np.abs(ys) <= m.rail_volts + 1e-12)
    assert np.any(np.abs(xs) == m.rail_volts) or np.any(np.abs(ys) == m.rail_volts)
    assert m.clipped >= 1


def test_the_suggested_gain_needs_a_nulled_instrument(table, built):
    """The suggestion is computed from the flaw alone, so un-nulled it rails.

    And it rails over GOOD MATERIAL AT ZERO TILT, not merely under tilt: even
    square on, the flaw-free reading is an order of magnitude up on the
    biggest notch, so a gain chosen to put that notch at 2 V puts the standing
    reading past the digitiser. That is the real instrument's behaviour, it is
    why an eddy-current bridge is balanced before the gain is set, and it is
    why this is a suggestion that gets reported rather than the default the
    model starts with.
    """
    plate = P.load_plate(os.path.join(built, "flaws.json"))
    pose = {"v": None}
    m = P.ECPlateModel(table, plate, pose_fn=lambda: pose["v"])
    assert m.suggested_v_per_ohm > m.v_per_ohm
    f = notch(m, "L3_D0")
    good = f["centre"] + np.array([0.0, 0.015, 0.0])
    on_notch = f["centre"] + np.array([0.002, 0.0, 0.0])

    m.v_per_ohm = m.suggested_v_per_ohm
    m.volts(good, IDENTITY, n=1, noise_v=0.0)
    assert m.clipped == 1, "un-nulled, the standing reading should rail"

    pose["v"] = (good, IDENTITY)
    m.null()
    m.clipped = 0
    xs, ys = m.volts(good, IDENTITY, n=1, noise_v=0.0)
    assert m.clipped == 0
    xs, ys = m.volts(on_notch, IDENTITY, n=1, noise_v=0.0)
    assert m.clipped == 0
    # and the notch is now a signal of order a volt rather than a millivolt
    assert abs(xs[0] - m.base_volts[0]) > 0.1

    # Tilt still rails it, nulled or not: that is the problem, not a bug.
    m.volts(good, quat_about((0.0, 1.0, 0.0), math.radians(5.0)),
            n=1, noise_v=0.0)
    assert m.clipped == 1


def test_noise_is_per_sample_not_per_block(model):
    """A block is many samples of one pose, and the noise is independent.

    The block IS what a buffered card returns per read, so EC_BLOCK_REDUCE's
    "mean" has to have something to average: identical samples would make the
    averaging choice untestable and the measured smear meaningless.
    """
    f = notch(model, "L3_D0")
    xs, ys = model.volts(f["centre"], IDENTITY, n=512)
    assert xs.std() == pytest.approx(P.NOISE_V, rel=0.2)
    assert ys.std() == pytest.approx(P.NOISE_V, rel=0.2)
    assert len(set(xs.tolist())) > 500


def test_report_carries_the_provenance_through(model):
    """A run's report must say which table and which plate produced it, and
    whether the table was a stub -- a renamed stub is otherwise
    indistinguishable from the real thing."""
    r = model.report()
    assert r["plate"].endswith("flaws.json")
    assert r["flaws"] == 12
    assert r["table"]["stub"] is True
    assert r["table"]["path"].endswith("al6061_2mhz_response.h5")
