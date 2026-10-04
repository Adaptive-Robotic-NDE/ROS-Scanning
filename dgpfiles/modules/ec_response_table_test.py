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
# File:  ec_response_table_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Host tests for the response-table reader, including a check that it
#   agrees with ecsim's own reader wherever ecsim is importable.
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
"""The response-table reader, against a table whose every value is known.

NO ROS, NO CELL, NO ARM, NO spatialnde2. The table is written by
tools/make_ec_stub_table.py from closed-form expressions, so these tests check
that the reader RECOVERS known numbers rather than that it runs.

Two tests matter more than the rest.

THE FIRST is the agreement with ecsim.table.ResponseTable. This reader exists
because ecsim's is too slow for an acquisition loop, but ecsim's is the
AUTHORITY on the format -- so a disagreement means this one is wrong. It is
skipped where ecsim is not installed, which includes the container, and that is
the point of having the stub validate against ecsim's own validator too.

THE SECOND is that the schema constants here match ecsim's. They are duplicated
rather than imported, because the container has h5py and will never have ecsim,
and a duplicated constant that drifts is worse than an import that fails.

    python3 -m pytest -q ec_response_table_test.py
"""
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

# ecsim is developed in its own repository and is deliberately NOT a dependency
# of anything here -- the container will never have it. But where a checkout
# exists it is the authority on the table format, and the two tests that
# compare against it are the most valuable in this file, so they are worth
# finding rather than skipping by default. ECSIM_DIR overrides; ~/ecsim is
# where it lives on the machine this was written on; absent, those two skip.
_ECSIM = os.environ.get("ECSIM_DIR") or os.path.expanduser("~/ecsim")
if os.path.isdir(os.path.join(_ECSIM, "ecsim")) and _ECSIM not in sys.path:
    sys.path.append(_ECSIM)

pytest.importorskip("h5py")

import ec_response_table as R                               # noqa: E402
import make_ec_stub_table as STUB                           # noqa: E402


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ecstub")
    STUB.write(str(d))
    return str(d)


@pytest.fixture(scope="module")
def table(stub_dir):
    t = R.ECResponseTable(os.path.join(stub_dir, "al6061_2mhz_response.h5"))
    yield t
    t.close()


# ---------------------------------------------------------------------------
# The schema, and the authority on it
# ---------------------------------------------------------------------------

def test_schema_constants_match_ecsim():
    """The duplicated schema names must equal ecsim's own.

    This is the test that makes the duplication safe. ec_response_table.py
    cannot import ecsim -- the container has h5py and will never have the
    simulator -- so the names are copied, and a copy nobody checks is how two
    readers end up disagreeing about which axis is which.
    """
    schema = pytest.importorskip("ecsim.table.schema")
    assert R.RESPONSE_PATH == schema.RESPONSE_PATH
    assert R.WRITTEN_PATH == schema.WRITTEN_MASK_PATH
    assert R.AXES_GROUP == schema.AXES_GROUP
    assert R.BASELINE_PATH == schema.BASELINE_PATH
    assert R.PARAM_AXIS_ORDER_ATTR == schema.PARAM_AXIS_ORDER_ATTR
    assert R.YAW_PERIOD_ATTR == schema.YAW_PERIOD_ATTR
    assert R.UNITS_ATTR == schema.UNITS_ATTR
    assert R.FLAW_AXES == schema.FLAW_AXES
    assert R.POSE_AXES == schema.LIFTOFF_TILT_AXES
    assert R.SPATIAL_AXES == schema.SPATIAL_AXES
    assert R.AXIS_UNITS == schema.AXIS_UNITS


def test_agrees_with_ecsim_reader(table):
    """Same file, same query, same answer -- to the stored precision.

    ecsim's reader is the authority. It reads the whole field and samples it;
    this one collapses the flaw axes first and then samples. Different
    arithmetic, and it has to come out the same.
    """
    ecsim_table = pytest.importorskip("ecsim.table")
    other = ecsim_table.ResponseTable(table.path)
    try:
        rng = np.random.default_rng(4)
        ax = table.axes
        for _ in range(40):
            d = float(rng.choice(ax["depth"]))
            l = float(rng.choice(ax["length"]))
            w = float(ax["width"][0])
            y = float(rng.choice(ax["yaw"]))
            lo = float(rng.choice(ax["liftoff"]))
            tp = float(rng.choice(ax["tilt_parallel"]))
            tq = float(rng.choice(ax["tilt_perpendicular"]))
            dx = float(rng.uniform(ax["x"][0], ax["x"][-1]))
            dy = float(rng.uniform(ax["y"][0], ax["y"][-1]))
            mine = table.response(d, l, w, y, lo, tp, tq, dx=dx, dy=dy)
            theirs = other.response(d, l, w, y, lo, tp, tq, dx=dx, dy=dy)
            assert abs(mine - theirs) <= 1e-6*max(1.0, abs(theirs))
    finally:
        other.close()


def test_refuses_a_table_with_no_axis_order(tmp_path):
    """A file with no _param_axis_order cannot be indexed, so it is refused.

    Not defaulted to the canonical order: a file that did not come from the
    generator may have any order at all, and reading one as another silently
    swaps depth for length.
    """
    import h5py
    p = str(tmp_path/"bad.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("/response", data=np.zeros((1, 1), dtype=np.complex64))
    with pytest.raises(ValueError, match="_param_axis_order"):
        R.ECResponseTable(p)


def test_refuses_wrong_units(stub_dir, tmp_path):
    """Mils, declared as mils, are refused rather than converted.

    The spec called a silent 25.4x the kind of mistake that survives review.
    The existing VIC-3D table really is in mils, so this is not hypothetical.
    """
    import shutil
    import h5py
    p = str(tmp_path/"mils.h5")
    shutil.copy(os.path.join(stub_dir, "al6061_2mhz_response.h5"), p)
    with h5py.File(p, "r+") as f:
        f["/axes/length"].attrs["units"] = "mil"
    with pytest.raises(ValueError, match="mil"):
        R.ECResponseTable(p)


def test_refuses_a_factorised_table(stub_dir, tmp_path):
    """A /liftoff_tilt/G table is refused, not read as if it were dense.

    Reading it as dense would drop the ENTIRE pose dependence -- lift-off and
    both tilts -- and the scan would come out looking beautifully clean.
    """
    import shutil
    import h5py
    p = str(tmp_path/"fact.h5")
    shutil.copy(os.path.join(stub_dir, "al6061_2mhz_response.h5"), p)
    with h5py.File(p, "r+") as f:
        f.create_dataset("/liftoff_tilt/G", data=np.ones((2, 2, 2), dtype=np.complex64))
    with pytest.raises(ValueError, match="FACTORISED"):
        R.ECResponseTable(p)


def test_knows_it_is_a_stub(table):
    """Told from the FILE, not from its name: a renamed stub is otherwise
    indistinguishable from the real table."""
    assert table.is_stub is True
    assert "STUB" in table.describe()
    assert table.report()["stub"] is True


# ---------------------------------------------------------------------------
# Does it recover the values that were written?
# ---------------------------------------------------------------------------

def test_on_node_query_is_exact(table):
    """An on-node query returns the stored value with no interpolation.

    Every notch on the test plate is on a node by construction, so this is the
    case the simulator spends its whole life in.
    """
    ax = table.axes
    d, l, w, y = (float(ax["depth"][1]), float(ax["length"][3]),
                  float(ax["width"][0]), float(ax["yaw"][3]))
    lo, tp, tq = 0.0, 0.0, 0.0
    want = STUB.flaw_field(d, l, w, y, lo, tp, tq, ax["x"], ax["y"])
    for xi in (0, 5, 7, 12):
        for yi in (0, 6, 12):
            got = table.response(d, l, w, y, lo, tp, tq,
                                 dx=float(ax["x"][xi]), dy=float(ax["y"][yi]))
            assert abs(got - complex(want[yi, xi])) <= 1e-6


def test_the_flaw_signature_is_double_lobed(table):
    """Antisymmetric in x about the flaw centre, with a null on it.

    This is what makes it a DIFFERENTIAL probe: the leading coil sees the flaw
    first, then both, then the trailing coil, so the response goes positive,
    through a null, to negative. A reader that mirrored its x axis would
    invert every signature and nothing downstream would notice, which is why
    this is checked and not assumed.
    """
    ax = table.axes
    d, l, w = float(ax["depth"][1]), float(ax["length"][3]), float(ax["width"][0])
    y = math.pi/2.0
    at_centre = table.response(d, l, w, y, dx=0.0, dy=0.0)
    assert abs(at_centre) < 1e-9
    for dx in (0.001, 0.002, 0.003):
        a = table.response(d, l, w, y, dx=+dx, dy=0.0)
        b = table.response(d, l, w, y, dx=-dx, dy=0.0)
        assert abs(a) > 1e-4
        assert abs(a + b) <= 1e-6*abs(a)


def test_yaw_is_reduced_modulo_pi(table):
    """A notch is a line SEGMENT, so yaw and yaw+180 degrees are the same flaw.

    The real table does not declare yaw_period_rad, so without this a query at
    3.3 rad -- an ordinary orientation on the plate -- would be refused.
    """
    ax = table.axes
    d, l, w = float(ax["depth"][0]), float(ax["length"][2]), float(ax["width"][0])
    base = table.response(d, l, w, math.radians(30.0), dx=0.002, dy=0.0)
    for extra in (math.pi, 2.0*math.pi, -math.pi):
        got = table.response(d, l, w, math.radians(30.0) + extra, dx=0.002, dy=0.0)
        assert abs(got - base) <= 1e-9
    assert table.reduce_yaw(math.radians(200.0)) == pytest.approx(math.radians(20.0))


def test_baseline_comes_from_the_sidecar(table, stub_dir):
    """The flaw-free trajectory is found beside the table, not inside it.

    A dense lift-off/tilt table has nowhere in the schema to put it, so the
    generator writes a .npz. Without it the simulator reads nothing at all
    over good material, which is exactly what a TILTED differential probe does
    not do -- and tilt is the dominant signal.
    """
    assert table.has_baseline
    assert table.baseline_source.endswith("al6061_2mhz_baseline.npz")
    for lo, tp, tq in ((0.0, 0.0, 0.0), (0.00025, math.radians(5.0), 0.0),
                       (0.001, math.radians(-10.0), math.radians(10.0))):
        assert table.baseline(lo, tp, tq) == pytest.approx(
            STUB.baseline_value(lo, tp, tq), rel=1e-5, abs=1e-6)


def test_baseline_is_odd_in_parallel_tilt_and_even_in_perpendicular(table):
    """The sign of the asymmetry is the sign of the parallel tilt.

    Parallel tilt raises one D and lowers the other, so reversing it reverses
    the reading; perpendicular tilt moves both the same way and a differential
    probe cannot tell which way. Getting these the wrong way round would make
    the common-mode tilt look like the dangerous one.
    """
    a = table.baseline(0.0, math.radians(10.0), 0.0)
    b = table.baseline(0.0, math.radians(-10.0), 0.0)
    sq = table.baseline(0.0, 0.0, math.radians(10.0))
    sq2 = table.baseline(0.0, 0.0, math.radians(-10.0))
    z = table.baseline(0.0, 0.0, 0.0)
    assert abs((a - z) + (b - z)) <= 1e-4*abs(a - z)
    assert sq == pytest.approx(sq2, rel=1e-6)


def test_tilt_dominates_the_flaw(table):
    """The ordering that makes the whole problem hard, asserted.

    A few degrees of tilt about the split line gives a far bigger reading than
    the largest notch in the table does. If a change ever made the flaw the
    larger of the two, every conclusion drawn in simulation about separating
    them would be worthless -- so it is a test and not a remark.
    """
    ax = table.axes
    d, l, w = float(ax["depth"][-1]), float(ax["length"][-1]), float(ax["width"][0])
    block = table.pose_block(d, l, w, math.pi/2.0)
    flaw_peak = float(np.abs(block).max())
    tilt = abs(table.baseline(0.0, math.radians(5.0), 0.0)
               - table.baseline(0.0, 0.0, 0.0))
    assert tilt > 10.0*flaw_peak


# ---------------------------------------------------------------------------
# Out of domain: counted, never raised
# ---------------------------------------------------------------------------

def test_outside_the_field_is_zero_and_counted(table):
    """A flaw further away than the stored field contributes nothing.

    NORMAL, not an error: most flaws are out of reach most of the time, and
    the field's edge is deliberately clean baseline. Counted so that a run
    which never came within reach of anything can be told from one that did.
    """
    before = table.report()["domain"]["outside_field"]
    got = table.response(0.0005, 0.00254, 0.00015, math.pi/2.0, dx=0.05, dy=0.0)
    assert got == 0.0
    assert table.report()["domain"]["outside_field"] == before + 1


def test_a_pose_off_the_table_is_clamped_not_raised(table):
    """Out-of-domain tilt clamps and counts. It must NEVER raise.

    On the acquisition thread an exception is a lost block, and a run of lost
    blocks is a silent flatline that looks like an instrument that stopped.
    """
    big = math.radians(45.0)
    got = table.baseline(0.0, big, 0.0)
    edge = table.baseline(0.0, float(table.axes["tilt_parallel"][-1]), 0.0)
    assert got == pytest.approx(edge, rel=1e-9)
    assert table.report()["domain"]["clamped"].get("tilt_parallel", 0) >= 1


def test_standoff_past_the_table_decays_to_nothing(table):
    """Off the part the reading goes to zero, on the table's OWN decay length.

    Not clamped: clamping would hold a full-amplitude tilt signal while the
    arm flies between passes, and that looks like data. The decay length is
    recovered from the written baseline -- the stub writes 0.4 mm -- rather
    than chosen here.
    """
    assert table.standoff_decay_m == pytest.approx(STUB.LIFTOFF_DECAY_M, rel=1e-6)
    top = float(table.axes["liftoff"][-1])
    at_top = abs(table.baseline(top, math.radians(5.0), 0.0))
    seq = [abs(table.baseline(top + k*table.standoff_decay_m,
                              math.radians(5.0), 0.0)) for k in (1, 2, 4)]
    assert at_top > seq[0] > seq[1] > seq[2]
    assert seq[0] == pytest.approx(at_top*math.exp(-1.0), rel=1e-6)
    # and far enough out it is simply zero
    assert table.baseline(top + 50.0*table.standoff_decay_m,
                          math.radians(5.0), 0.0) == 0.0


def test_missing_baseline_is_refused_not_zeroed(stub_dir, tmp_path):
    """No trajectory means no answer, loudly.

    Returning zero would be a simulator that reads nothing over good material
    however the probe is tilted -- which hides the one effect the table was
    generated to show.
    """
    import shutil
    p = str(tmp_path/"nobase_response.h5")
    shutil.copy(os.path.join(stub_dir, "al6061_2mhz_response.h5"), p)
    t = R.ECResponseTable(p)
    try:
        assert not t.has_baseline
        with pytest.raises(RuntimeError, match="flaw-free"):
            t.baseline(0.0, 0.0, 0.0)
    finally:
        t.close()


# ---------------------------------------------------------------------------
# The cache, which is the only reason this is fast enough
# ---------------------------------------------------------------------------

def test_the_pose_block_is_cached_per_flaw(table):
    """One block per flaw geometry, reused for every position.

    The point of the whole design: a scan queries the same handful of flaw
    geometries tens of thousands of times at different positions, and
    re-collapsing the flaw axes every time is what made ecsim's reader too
    slow to use here.
    """
    ax = table.axes
    args = (float(ax["depth"][0]), float(ax["length"][1]),
            float(ax["width"][0]), float(ax["yaw"][2]))
    first = table.pose_block(*args)
    assert table.pose_block(*args) is first
    # and a different yaw is a different block
    assert table.pose_block(args[0], args[1], args[2],
                            float(ax["yaw"][3])) is not first


def test_the_cache_is_bounded(stub_dir):
    """A bounded cache evicts rather than growing without limit.

    The real table's block is 2.9 MB; a scan over a plate of a dozen notches
    is fine, but nothing here should be able to consume a machine if the flaw
    list grows or a geometry comes in off-node every time.
    """
    t = R.ECResponseTable(os.path.join(stub_dir, "al6061_2mhz_response.h5"),
                          max_cache_mb=0.5)
    try:
        ax = t.axes
        for k, yaw in enumerate(np.linspace(0.0, math.pi, 25)):
            t.pose_block(float(ax["depth"][0]), float(ax["length"][0]),
                         float(ax["width"][0]), float(yaw))
        r = t.report()
        assert r["cache_mb"] <= 0.5
        assert r["cached_flaws"] < 25
    finally:
        t.close()


def test_a_query_is_fast_enough_for_the_publish_loop(table):
    """The reader has to keep up with 50 blocks a second over a dozen flaws.

    A loose bound -- it is a timing test and the machine is shared -- but it
    would catch a change that went back to reading the whole field per query,
    which is three orders of magnitude slower and the reason this module
    exists.
    """
    import time
    ax = table.axes
    args = (float(ax["depth"][0]), float(ax["length"][2]), float(ax["width"][0]),
            math.pi/2.0)
    table.pose_block(*args)                       # warm, as a scan would be
    n = 2000
    t0 = time.perf_counter()
    for k in range(n):
        table.response(*args, standoff=0.0001, tilt_parallel=0.01,
                       tilt_perpendicular=0.0,
                       dx=0.004*(k % 3 - 1), dy=0.0)
    per = (time.perf_counter() - t0)/n
    # 50 Hz x 12 flaws is 600 queries a second; 200 us each leaves 8x headroom.
    assert per < 200e-6, "%.1f us per query" % (per*1e6,)
