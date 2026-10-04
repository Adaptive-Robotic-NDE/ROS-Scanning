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
# File:  sim_ec_model_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Host tests for the seam between the simulated instrument and the
#   response-table model: that the model drives the published channels,
#   that it cannot wedge the publish loop, and that a block says which
#   simulation produced it.
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
"""Where the model meets the instrument, which is where a scan silently lies.

NO ROS, NO CELL, NO ARM. spatialnde2 IS needed, because the thing under test is
what reaches the recording database. Needs h5py too, for the stub table.

THE FAILURE THIS FILE IS FOR. A simulated scan that was supposed to run from
the response table but quietly ran on the synthetic bumps instead produces a
complete, plausible dataset that means nothing -- and there is nothing in the
numbers to tell the two apart afterwards. So:

  * a model that is wired must actually be the thing filling the channels, not
    merely present;
  * a model that raises must not stop the instrument, because a stalled
    instrument is a silent flatline on the acquisition thread -- but it must be
    counted, so the run can be disbelieved;
  * every published block must record WHICH simulation filled it, and whether
    the table was a stub.

    python3 -m pytest -q sim_ec_model_test.py
"""
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

snde = pytest.importorskip("spatialnde2")
pytest.importorskip("h5py")

import ec_plate_model as P                                  # noqa: E402
import ec_response_table as R                               # noqa: E402
import make_ec_stub_table as STUB                           # noqa: E402
import make_ec_test_plate as PLATE                          # noqa: E402

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def load_sim_ec_source():
    """Run sim_ec_source.dpi's body into a namespace, as the loader does.

    A .dpi is a function body with include() injected, so it cannot be
    imported; the `global` lines are stripped because at module scope they are
    a syntax error on a name already assigned. Same approach as
    ec_scan_wiring_test.py, which is where it came from.
    """
    ns = {}
    exec("import math, os, threading, time, traceback\n"
         "import numpy as np\nimport spatialnde2 as snde\n", ns)
    src = open(os.path.join(HERE, "sim_ec_source.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    exec(compile(src, "sim_ec_source.dpi", "exec"), ns)
    return ns


def recdb():
    db = snde.recdatabase()
    snde.setup_cpu(db, [], 4)
    snde.setup_storage_manager(db)
    snde.setup_math_functions(db, [])
    db.startup()
    return db


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim_ec_model")
    STUB.write(str(d))
    PLATE.main(["--out-dir", str(d)])
    return str(d)


@pytest.fixture(scope="module")
def table(built):
    t = R.ECResponseTable(os.path.join(built, "al6061_2mhz_response.h5"))
    yield t
    t.close()


@pytest.fixture(scope="module")
def SimECSource():
    return load_sim_ec_source()["SimECSource"]


def a_notch(model, name="L3_D0"):
    return [f for f in model.flaws if f["name"] == name][0]


def wait_for_blocks(sim, n, timeout=20.0):
    deadline = time.monotonic() + timeout
    while sim.blocks < n and time.monotonic() < deadline:
        time.sleep(0.02)
    return sim.blocks


# ---------------------------------------------------------------------------

def test_a_model_without_a_pose_source_is_refused(table, built, SimECSource):
    """position_fn is not enough for a model, and the refusal says why.

    Lift-off and both tilts come out of the ORIENTATION, and tilt is the
    larger signal -- so a model driven from position alone would hold the
    probe permanently square-on and hide exactly the effect the table was
    generated to study. Silently accepting that would be the worst outcome
    available here.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    db = recdb()
    with pytest.raises(ValueError, match="pose_fn"):
        SimECSource(db, None, "/sim/x", "/sim/y", position_fn=lambda: None,
                    model=model, pose_fn=None)


def test_the_model_is_what_fills_the_channels(table, built, SimECSource):
    """The published volts are the model's, not the synthetic signal's.

    Checked by VALUE against the model called directly, because a model that
    is merely attached and never consulted is the failure this file exists
    for.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    f = a_notch(model)
    here = f["centre"] + np.array([0.002, 0.0, 0.0])
    pose = (tuple(float(v) for v in here), IDENTITY, 0.0)

    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: pose[0],
                      pose_fn=lambda: pose, model=model,
                      sample_rate=1000.0, block_hz=100.0, seed=3)
    try:
        assert wait_for_blocks(sim, 3) >= 3
        xs = np.asarray(db.latest.ref["/sim/x"].data).reshape(-1)
        ys = np.asarray(db.latest.ref["/sim/y"].data).reshape(-1)
        # The noise is per sample, so compare the means against the model's
        # own noiseless answer at the same pose.
        wx, wy = model.volts(np.asarray(here), IDENTITY, n=1, noise_v=0.0)
        assert xs.mean() == pytest.approx(float(wx[0]), abs=5.0*P.NOISE_V)
        assert ys.mean() == pytest.approx(float(wy[0]), abs=5.0*P.NOISE_V)
        # and it is NOT the synthetic signal, whose null point is elsewhere
        assert abs(xs.mean() - sim.BASE[0]) > 1e-6
    finally:
        sim.stop()


def test_the_flaw_shows_up_in_the_published_channels(table, built, SimECSource):
    """Moving the probe across a notch changes what is published.

    With identical traces everywhere, a projection that lands every trace in
    the wrong place and a projection that works look exactly the same -- which
    is why the synthetic signal was position-dependent too, and why the
    model's being so has to be checked at the channel rather than at the model.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    f = a_notch(model)
    pose = {"v": None}
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: (pose["v"] or ((0, 0, 0),))[0],
                      pose_fn=lambda: pose["v"], model=model,
                      sample_rate=1000.0, block_hz=100.0, seed=5)
    try:
        got = []
        for dx in (-0.002, 0.0, +0.002):
            pose["v"] = (tuple(f["centre"] + np.array([dx, 0.0, 0.0])),
                         IDENTITY, 0.0)
            start = sim.blocks
            wait_for_blocks(sim, start + 4)
            got.append(float(np.asarray(db.latest.ref["/sim/y"].data)
                             .reshape(-1).mean()))
        # The lobes either side of the notch differ from the null on it, and
        # differ from each other in the opposite sense.
        assert abs(got[0] - got[1]) > 3.0*P.NOISE_V
        assert abs(got[2] - got[1]) > 3.0*P.NOISE_V
        assert (got[0] - got[1])*(got[2] - got[1]) < 0.0
    finally:
        sim.stop()


def test_a_model_that_raises_does_not_stop_the_instrument(table, built,
                                                          SimECSource):
    """It falls back, keeps publishing, and counts -- loudly in report().

    On the acquisition thread an exception is a lost block, and a run of lost
    blocks is a silent flatline that reads as an instrument that stopped. But
    falling back QUIETLY would be worse than stopping, so the count and the
    formatted traceback are the point of this test as much as the survival is.

    FORMATTED, not the exception object: retaining the object pins its
    traceback, and the frame in it, and the active transaction in that frame --
    which never ends and blocks every globalrev wait in the process. That bug
    cost 59 passes once; see the note in triggered_acquisition.dpi.
    """
    class Exploding(object):
        table = None

        def __init__(self):
            self.calls = 0

        def volts(self, xyz, quat, n=1):
            self.calls += 1
            raise RuntimeError("no")

        def report(self):
            return {"calls": self.calls}

    bad = Exploding()
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: (0.85, -0.2, 0.95),
                      pose_fn=lambda: ((0.85, -0.2, 0.95), IDENTITY, 0.0),
                      model=bad, sample_rate=1000.0, block_hz=100.0, seed=1)
    try:
        assert wait_for_blocks(sim, 5) >= 5
        assert bad.calls >= 5
        r = sim.report()
        assert r["model_errors"] >= 5
        assert r["errors"] == 0           # the PUBLISH loop is healthy
        assert isinstance(r["last_error"], str)
        assert "RuntimeError" in r["last_error"]
        # and the channels carry the synthetic fallback rather than nothing
        xs = np.asarray(db.latest.ref["/sim/x"].data).reshape(-1)
        assert xs.size > 0
        assert np.isfinite(xs).all()
    finally:
        sim.stop()


def test_no_pose_yet_is_counted_apart_from_a_failure(table, built, SimECSource):
    """TF not up yet is a different thing from a model that broke.

    They want opposite responses: this one resolves itself as soon as the cell
    comes up, and a nonzero count after a finished scan means the model was
    never consulted at all.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: None, pose_fn=lambda: None,
                      model=model, sample_rate=1000.0, block_hz=100.0, seed=2)
    try:
        assert wait_for_blocks(sim, 4) >= 4
        r = sim.report()
        assert r["model_no_pose"] >= 4
        assert r["model_errors"] == 0
    finally:
        sim.stop()


def test_every_block_says_which_simulation_filled_it(table, built, SimECSource):
    """The metadata distinguishes the model from the synthetic bumps.

    A stored dataset whose traces came from the synthetic signal and one whose
    traces came from the response table are not comparable, and nothing else
    in the file tells them apart. The stub flag rides along for the same
    reason: a renamed stub is otherwise indistinguishable from the real table.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: (0.85, -0.2, 0.95),
                      pose_fn=lambda: ((0.85, -0.2, 0.95), IDENTITY, 0.0),
                      model=model, sample_rate=1000.0, block_hz=100.0, seed=4)
    try:
        assert wait_for_blocks(sim, 2) >= 2
        md = db.latest.ref["/sim/x"].rec.metadata
        assert md.GetMetaDatumStr("EC-sim-source", "") == "table-model"
        assert md.GetMetaDatumInt("EC-sim-table-stub", -1) == 1
        assert md.GetMetaDatumStr("EC-sim-table", "").endswith(".h5")
        assert md.GetMetaDatumInt("EC-simulated", 0) == 1
    finally:
        sim.stop()

    db2 = recdb()
    plain = SimECSource(db2, None, "/sim/x", "/sim/y", position_fn=None,
                        sample_rate=1000.0, block_hz=100.0, seed=4)
    try:
        assert wait_for_blocks(plain, 2) >= 2
        md = db2.latest.ref["/sim/x"].rec.metadata
        assert md.GetMetaDatumStr("EC-sim-source", "") == "synthetic"
        assert plain.report()["source"] == "synthetic"
    finally:
        plain.stop()


def test_the_synthetic_path_still_works_with_no_model(SimECSource):
    """No table, no h5py, no plate: the architectural stand-in still runs.

    It has to. Most machines that work on this pipeline have no table, and the
    wiring failures the synthetic signal was written to catch -- a math
    function attached to a channel that does not exist yet -- are still the
    ones that take the whole process down.
    """
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y", position_fn=None,
                      sample_rate=1000.0, block_hz=100.0, seed=9)
    try:
        assert wait_for_blocks(sim, 3) >= 3
        r = sim.report()
        assert r["source"] == "synthetic"
        assert r["errors"] == 0
        assert r["model_errors"] == 0
        xs = np.asarray(db.latest.ref["/sim/x"].data).reshape(-1)
        assert xs.mean() == pytest.approx(sim.BASE[0], abs=5.0*sim.NOISE_V)
    finally:
        sim.stop()


def test_the_block_is_many_samples_of_one_pose(table, built, SimECSource):
    """A buffered card returns a BLOCK per read, and so does this.

    EC_BLOCK_REDUCE ("mean"/"last"/"first") chooses what to do with it and
    report_smear() measures how far the probe moved during it; a
    one-sample-per-transaction simulator would make both untestable.
    """
    model = P.ECPlateModel(table, P.load_plate(os.path.join(built, "flaws.json")))
    db = recdb()
    sim = SimECSource(db, None, "/sim/x", "/sim/y",
                      position_fn=lambda: (0.85, -0.2, 0.95),
                      pose_fn=lambda: ((0.85, -0.2, 0.95), IDENTITY, 0.0),
                      model=model, sample_rate=1000.0, block_hz=50.0, seed=6)
    try:
        assert wait_for_blocks(sim, 2) >= 2
        xs = np.asarray(db.latest.ref["/sim/x"].data).reshape(-1)
        assert xs.size == sim.block_samples == 20
        assert xs.std() > 0.0            # per-sample noise, not a flat block
        md = db.latest.ref["/sim/x"].rec.metadata
        assert md.GetMetaDatumDbl("ande_array-axis0_scale", 0.0) \
            == pytest.approx(1.0/1000.0)
        assert md.GetMetaDatumStr("ande_array-ampl_units", "") == "V"
    finally:
        sim.stop()
