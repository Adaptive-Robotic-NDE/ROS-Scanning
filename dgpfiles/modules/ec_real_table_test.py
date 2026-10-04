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
# File:  ec_real_table_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Checks the GENERATED eddy-current response table, as opposed to the
#   stub the other suites build: the acceptance tests the model
#   specification asked for, and the properties the consumer relies on
#   that a stub cannot vouch for.
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
"""The real table, not the stub.

NO ROS, NO CELL, NO ARM, NO spatialnde2, NO ecsim. Skips when the fixture is
not on this machine, and a skip is not a pass.

WHY THIS EXISTS SEPARATELY. ec_response_table_test.py and ec_plate_model_test.py
work against tools/make_ec_stub_table.py, which is the only way they could have
been written before the table existed. A stub proves the reader handles the
SCHEMA. It cannot prove anything about the physics, and worse, it can quietly
teach the reader a property the real table does not have -- the stub's
flaw-free reading is nonzero square-on, and the real one is identically zero,
so a fit written against the stub could divide by it forever and never know.

Two kinds of test here.

FIRST, the acceptance tests docs/ec-probe-model-spec.md asked for, which are
about the generated physics and not about our code: response amplitudes ordered
by notch length, and a differential probe's antisymmetric double lobe.

SECOND, the properties the consumer RELIES ON and the stub cannot vouch for:
the file declares no yaw period (so the reader supplies pi), the stand-off decay
has to be recoverable from a real baseline, and tilt about the split line has to
be the signed, flaw-like one while the perpendicular tilt is common-mode.

    python3 -m pytest -q ec_real_table_test.py
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

pytest.importorskip("h5py")

import ec_response_table as R                               # noqa: E402
import test_data                                            # noqa: E402

TABLE = ("ec_response", "al6061_2mhz_response.h5")

#: What counts as zero here. The table was generated with
#: solver_precision = "single", so float32's eps of 1.2e-7 is the floor on any
#: cancellation: the null on a flaw centre comes out at 5.9e-7 ohm against a
#: 1.8 ohm peak, and the flaw-free reading square-on at 1e-15 ohm against a
#: 6 ohm signal at tilt. Those are exact to the solver's precision and the
#: tests below say so with a tolerance rather than demanding bitwise zero --
#: which the first draft of this file did, and which fails on real output while
#: passing on any stub that writes a literal 0.
ZERO_REL = 1e-5


@pytest.fixture(scope="module")
def table():
    p = test_data.path(*TABLE)
    if p is None:
        pytest.skip("the generated response table is not here: %s"
                    % (test_data.why_missing(*TABLE),))
    t = R.ECResponseTable(p)
    yield t
    t.close()


def flaw_peak(table, depth, length, yaw, standoff=0.0):
    """Peak |dZ| over the stored field at one on-node pose."""
    ax = table.axes
    j0 = int(np.argmin(np.abs(ax["tilt_parallel"])))
    k0 = int(np.argmin(np.abs(ax["tilt_perpendicular"])))
    i = int(np.argmin(np.abs(ax["liftoff"] - standoff)))
    block = table.pose_block(depth, length, float(ax["width"][0]), yaw)
    return float(np.abs(block[i, j0, k0]).max())


# ---------------------------------------------------------------------------
# It is the real thing
# ---------------------------------------------------------------------------

def test_it_is_not_a_stub_and_says_what_it_is(table):
    """Provenance read from the FILE. A renamed stub is otherwise
    indistinguishable from this, which is what the spec's provenance section
    exists to prevent."""
    assert table.is_stub is False
    a = table.attrs
    assert a["frequency_hz"] == pytest.approx(2.0e6)
    assert "6061" in str(a["material"])
    assert a["conductivity_s_per_m"] == pytest.approx(2.5e7, rel=0.2)
    assert "split-D" in str(a["probe_type"])
    assert a["shielded"]
    # The caveat has to travel WITH the data, because absolute amplitudes are
    # not to be trusted until the coil geometry is fitted, and a number quoted
    # from this table without that caveat is a wrong number.
    assert "PLACEHOLDER" in str(a["notes"]).upper()


def test_every_pose_was_computed(table):
    """A partially-written table loads and validates and silently reads zero
    wherever the run did not reach, so completeness is checked, not assumed."""
    import h5py
    with h5py.File(table.path, "r") as f:
        written = np.asarray(f["/written"])
    assert written.size == 4200
    assert int(written.sum()) == written.size


# ---------------------------------------------------------------------------
# The acceptance tests the specification asked for
# ---------------------------------------------------------------------------

def test_amplitude_is_ordered_by_notch_length(table):
    """"A pass over each of the four calibration notches at nominal lift-off
    should give response amplitudes ordered by notch length." -- spec section 7.

    The cheap acceptance check, and the one an operator would make first on a
    real calibration block. Asserted at both depths.
    """
    ax = table.axes
    for depth in ax["depth"]:
        peaks = [flaw_peak(table, float(depth), float(l), math.pi/2.0)
                 for l in ax["length"]]
        assert peaks == sorted(peaks), (
            "depth %.2f mm: peaks not ordered by length: %s"
            % (depth*1e3, ["%.4f" % p for p in peaks]))
        assert peaks[-1] > 1.5*peaks[0]


def test_amplitude_is_ordered_by_depth(table):
    """Deeper reads larger at the same length. Not in the spec's acceptance
    list, but it is the other axis of the same claim and it costs nothing."""
    ax = table.axes
    for length in ax["length"]:
        shallow = flaw_peak(table, float(ax["depth"][0]), float(length), math.pi/2.0)
        deep = flaw_peak(table, float(ax["depth"][-1]), float(length), math.pi/2.0)
        assert deep > shallow


def test_the_signature_is_an_antisymmetric_double_lobe(table):
    """"A single-lobed magnitude model is not a differential probe and will not
    let us test anything downstream." -- spec section 2.1.

    Positive, through a null ON the flaw, to negative. Checked on the real
    field: exactly odd in x, and the null exact rather than merely small.
    """
    ax = table.axes
    block = table.pose_block(float(ax["depth"][-1]), float(ax["length"][-1]),
                             float(ax["width"][0]), math.pi/2.0)
    j0 = int(np.argmin(np.abs(ax["tilt_parallel"])))
    k0 = int(np.argmin(np.abs(ax["tilt_perpendicular"])))
    field = block[0, j0, k0]
    xi0 = int(np.argmin(np.abs(ax["x"])))
    yi0 = int(np.argmin(np.abs(ax["y"])))
    peak = float(np.abs(field).max())
    assert abs(complex(field[yi0, xi0])) <= ZERO_REL*peak
    for k in (1, 2, 4, 8):
        a = complex(field[yi0, xi0 + k])
        b = complex(field[yi0, xi0 - k])
        assert abs(a + b) <= 1e-5*peak, "not odd in x at %d steps" % k
    # and it is LOCAL: gone well inside the stored field, which is what makes
    # the field's edge usable as baseline.
    assert abs(complex(field[yi0, -1])) < 0.01*peak


def test_the_response_is_narrower_than_the_trace_pitch_is_coarse(table):
    """How wide the lobes actually are, against our 1 mm trace pitch.

    Not a pass/fail about the table -- it is a fact about it that decides
    whether the scan samples the response at all. The lobes of the largest
    notch peak about 1 mm either side of centre, so a 1 mm pitch puts roughly
    one trace on each lobe and the null between them, with no margin. This
    test exists to make that number visible and to notice if it changes.
    """
    ax = table.axes
    block = table.pose_block(float(ax["depth"][-1]), float(ax["length"][-1]),
                             float(ax["width"][0]), math.pi/2.0)
    j0 = int(np.argmin(np.abs(ax["tilt_parallel"])))
    k0 = int(np.argmin(np.abs(ax["tilt_perpendicular"])))
    yi0 = int(np.argmin(np.abs(ax["y"])))
    line = np.abs(block[0, j0, k0][yi0])
    x_at_peak = abs(float(ax["x"][int(np.argmax(line))]))
    assert 0.0005 <= x_at_peak <= 0.0025, (
        "the lobe peaks %.2f mm from centre; the 1 mm trace pitch and "
        "SCAN_STEP_SIZE want rechecking" % (x_at_peak*1e3,))


def test_yaw_has_period_pi_in_the_data(table):
    """A notch is a line SEGMENT, so yaw and yaw+180 degrees are one flaw.

    The reader reduces modulo pi because this file declares no period. That
    reduction is only correct if the DATA has that period, so it is checked
    against the stored nodes rather than assumed from the geometry.
    """
    ax = table.axes
    yaw = ax["yaw"]
    assert float(yaw[0]) == 0.0
    assert float(yaw[-1]) == pytest.approx(math.pi)
    for a, b in ((0, -1), (1, -2), (2, -3)):
        fa = flaw_peak(table, float(ax["depth"][-1]), float(ax["length"][-1]),
                       float(yaw[a]))
        fb = flaw_peak(table, float(ax["depth"][-1]), float(ax["length"][-1]),
                       float(yaw[b]))
        assert fa == pytest.approx(fb, rel=1e-5), (
            "yaw %.0f and %.0f deg differ, so the period is not pi"
            % (math.degrees(yaw[a]), math.degrees(yaw[b])))


# ---------------------------------------------------------------------------
# What the consumer relies on, and a stub cannot vouch for
# ---------------------------------------------------------------------------

def test_the_file_declares_no_yaw_period(table):
    """So the reader has to supply it, and the fallback path is the live one.

    If a later generator starts writing yaw_period_rad this test fails, which
    is the right outcome: it means the fallback is no longer what runs and the
    note in ec_response_table.py explaining why it exists is stale.
    """
    assert table.yaw_period_declared is False
    assert table.yaw_period == pytest.approx(math.pi)


def test_the_baseline_is_beside_the_table_not_in_it(table):
    """A dense lift-off/tilt table has nowhere in the schema for /baseline, so
    the generator writes a .npz and the reader finds it by name. Without it the
    probe reads nothing over good material at any tilt, which is the one effect
    the table was generated to show."""
    assert table.has_baseline
    assert table.baseline_source.endswith("al6061_2mhz_baseline.npz")
    assert os.path.dirname(table.baseline_source) == os.path.dirname(table.path)


def test_the_standoff_decay_is_recoverable_from_the_real_baseline(table):
    """The fit must SUCCEED on real data, not fall back.

    THE TEST THAT CAUGHT A REAL BUG. The fit samples the top two lift-off
    nodes, and the first version sampled them at ZERO tilt -- where this probe's
    flaw-free reading turns out to be identically zero, so it was the ratio of
    two zeros and fell back to a constant every time. It samples the largest
    parallel tilt instead, which is where the flaw-free signal lives. A stub
    could not have shown this, because a stub chooses its own baseline.
    """
    assert table.standoff_decay_m != pytest.approx(R.FALLBACK_STANDOFF_DECAY_M)
    assert 1e-4 < table.standoff_decay_m < 5e-3
    # and it has to actually decay, monotonically, over the whole axis
    mags = [abs(table.baseline(float(s), math.radians(10.0), 0.0))
            for s in table.axes["liftoff"]]
    assert all(a > b for a, b in zip(mags, mags[1:])), mags


def test_parallel_tilt_is_signed_and_perpendicular_is_blind(table):
    """The claim the whole specification turns on, checked on real data.

    Parallel tilt raises one D and lowers the other, so reversing it reverses
    the reading -- exactly odd here, and exactly zero at zero tilt. The
    perpendicular tilt moves both coils together and this probe cannot see it
    at all. Getting these the wrong way round would make the common-mode tilt
    look like the dangerous one.
    """
    for s in (0.0, 0.00025, 0.0005):
        scale = abs(table.baseline(s, math.radians(10.0), 0.0))
        assert scale > 0.0
        assert abs(table.baseline(s, 0.0, 0.0)) <= ZERO_REL*scale
        for deg in (5.0, 10.0):
            a = table.baseline(s, math.radians(+deg), 0.0)
            b = table.baseline(s, math.radians(-deg), 0.0)
            assert abs(a) > 0.1*scale
            assert abs(a + b) <= ZERO_REL*abs(a)
        for deg in (-10.0, 10.0):
            assert abs(table.baseline(s, 0.0, math.radians(deg))) <= ZERO_REL*scale


def test_tilt_outweighs_the_largest_notch_and_by_how_much(table):
    """The ordering that makes the problem hard -- with the REAL numbers.

    The stub exaggerates this: there, five degrees of tilt is more than ten
    times the biggest notch. Here it is under four, and the figure that
    matters is the crossing point, which is under a degree and a half. That is
    inside what the mesh alone contributes: measured normal error on the ROI
    ran to a p99.9 of 1.88 degrees before repair, and NORMAL_REPAIR_TOL_DEG is
    3.0. So mesh-driven tilt is at flaw level on its own, which is the finding,
    and the assertions below are deliberately loose so that they check the
    ORDERING rather than pinning numbers that a fitted coil geometry will move.
    """
    ax = table.axes
    biggest = max(flaw_peak(table, float(ax["depth"][-1]), float(ax["length"][-1]),
                            float(y)) for y in ax["yaw"])
    assert biggest > 1.0

    at_five = abs(table.baseline(0.0, math.radians(5.0), 0.0))
    assert at_five > biggest, (
        "five degrees of parallel tilt (%.3f ohm) no longer outweighs the "
        "largest notch (%.3f ohm); every simulated conclusion about separating "
        "them rests on this ordering" % (at_five, biggest))

    # Where they cross. Loose bounds: the claim is "a small number of degrees",
    # not a particular one.
    degs = np.linspace(0.0, 10.0, 401)
    mags = np.array([abs(table.baseline(0.0, math.radians(float(d)), 0.0))
                     for d in degs])
    crossing = float(degs[int(np.argmin(np.abs(mags - biggest)))])
    assert 0.3 < crossing < 4.0, (
        "the largest notch is matched by %.2f deg of tilt" % crossing)


def test_lift_off_alone_produces_nothing(table):
    """THE SURPRISE, and it changes how the instrument is set up.

    This probe is a perfectly balanced differential: square on, the flaw-free
    reading is identically zero at every lift-off, so lift-off ALONE moves the
    operating point nowhere. A real probe has a residual imbalance and this
    model has none.

    Consequences, both recorded in docs/ec-simulated-probe-response.md: the
    hardware null has nothing to remove unless the probe is tilted, and
    set_rotation() has no LIFT-OFF direction to find -- the trajectory it
    would rotate onto an axis is the TILT one. A simulated conditioning
    procedure that expects to find a lift-off direction will not find one here,
    and that is the model's property rather than a fault in the procedure.
    """
    # Tilted, lift-off does move it -- which is the only reason the decay fit
    # above has anything to measure, and it is the scale "nothing" is measured
    # against here.
    tilted = [abs(table.baseline(float(s), math.radians(5.0), 0.0))
              for s in table.axes["liftoff"]]
    assert tilted[0] > tilted[-1] > 0.0
    for s, scale in zip(table.axes["liftoff"], tilted):
        assert abs(table.baseline(float(s), 0.0, 0.0)) <= ZERO_REL*scale


def test_a_whole_pass_runs_at_a_usable_rate(table):
    """Timing on the real field, which is 14x the stub's in points.

    49x49 rather than 13x13, so the cached pose block is 2.9 MB and a query
    reads more. Still has to keep ahead of 50 blocks a second over a dozen
    notches, which is the reason this reader exists rather than ecsim's.
    """
    import time
    ax = table.axes
    args = (float(ax["depth"][-1]), float(ax["length"][-1]),
            float(ax["width"][0]), math.pi/2.0)
    table.pose_block(*args)
    n = 2000
    t0 = time.perf_counter()
    for k in range(n):
        table.response(*args, standoff=0.0001, tilt_parallel=0.02,
                       tilt_perpendicular=0.0,
                       dx=0.004*(k % 3 - 1), dy=0.0)
    per = (time.perf_counter() - t0)/n
    assert per < 200e-6, "%.1f us per query" % (per*1e6,)
