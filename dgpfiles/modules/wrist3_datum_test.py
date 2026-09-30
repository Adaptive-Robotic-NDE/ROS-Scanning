"""The wrist-3 datum: where zero probe rotation sits inside the cable band.

WHY IT HAS TO TRAVEL WITH THE BAND. The band is in ABSOLUTE wrist-3 degrees, and
it says how far the wrist may turn -- not where a rotation of zero starts from.
The envelope capture measures both in the same joint state, on every tilt:

    no tilt      datum 166.09   band  16.09 .. 301.09   so -150 .. +135 of rotation
    tilt x +40   datum 147.12   band  72.12 .. 312.12
    tilt x -40   datum 179.19   band 179.19 .. 269.19   (the tightest)

The loader used to keep the band and drop the datum, which left the conversion
from "turn the probe 90 degrees" to "wrist 3 goes to X" as a constant typed into a
config -- a transcription that goes stale silently when the tooling is unbolted,
and which cannot express the tilt dependence at all.

Two things are tested: that the number survives every hop between the measurement
and the check, and that a rotation is graded against the datum and the band taken
at the SAME tilt.

Runs on the host: pure model, numpy only, no ROS and no robot.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: The cell's own measurement, as capture_cable_envelope() wrote it.
#: The workspace root, found from this file rather than hardcoded: these
#: suites run from a checkout that may live anywhere.
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CELL_FILE = os.path.join(WORKSPACE, "scandata", "results14", "scandata",
                         "capture", "cable_limits_ex-ur10e-1_hardware.json")


def _load():
    src = open(os.path.join(HERE, "cable_envelope.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "wrist3_datum_test"}
    exec(compile(src, "cable_envelope.dpi", "exec"), ns)
    return ns


NS = _load()

#: A five-point cross with a datum on every row, shaped as schema 2 writes it.
ROWS = [
    {"tilt": "no tilt", "tx_deg": 0.0, "ty_deg": 0.0,
     "lo_deg": 16.0, "hi_deg": 301.0, "wrist3_start_deg": 166.0},
    {"tilt": "tilt x +40 deg", "tx_deg": 40.0, "ty_deg": 0.0,
     "lo_deg": 72.0, "hi_deg": 312.0, "wrist3_start_deg": 147.0},
    {"tilt": "tilt x -40 deg", "tx_deg": -40.0, "ty_deg": 0.0,
     "lo_deg": 179.0, "hi_deg": 269.0, "wrist3_start_deg": 179.0},
    {"tilt": "tilt y +20 deg", "tx_deg": 0.0, "ty_deg": 20.0,
     "lo_deg": 78.0, "hi_deg": 303.0, "wrist3_start_deg": 168.0},
    {"tilt": "tilt y -20 deg", "tx_deg": 0.0, "ty_deg": -20.0,
     "lo_deg": 60.0, "hi_deg": 285.0, "wrist3_start_deg": 165.0},
]


def _env(rows=None):
    return NS["cable_envelope_from_rows"](rows or ROWS, step_deg=20.0,
                                          cell="test", mode="hardware")


def test_every_measured_datum_comes_back_exactly():
    env = _env()
    for row in ROWS:
        got = env.wrist3_datum_deg(tilt=(row["tx_deg"], row["ty_deg"]))
        assert got is not None, row["tilt"]
        assert abs(got - row["wrist3_start_deg"]) < 1e-9, (
            "%s: datum came back %.4f, measured %.4f"
            % (row["tilt"], got, row["wrist3_start_deg"]))


def test_a_tilt_between_samples_is_interpolated_not_invented():
    env = _env()
    mid = env.wrist3_datum_deg(tilt=(20.0, 0.0))
    assert abs(mid - 0.5*(166.0 + 147.0)) < 1e-9, mid
    # Both axes at once pays for both, the same rule the band uses.
    both = env.wrist3_datum_deg(tilt=(40.0, 20.0))
    assert abs(both - (166.0 + (147.0 - 166.0) + (168.0 - 166.0))) < 1e-9, both


def test_a_measurement_without_the_datum_says_None():
    """None, not a guess. Where zero sits cannot be derived from the band."""
    rows = [dict(r) for r in ROWS]
    for r in rows:
        r.pop("wrist3_start_deg")
    env = _env(rows)
    assert env.wrist3_datum_deg(tilt=(0.0, 0.0)) is None
    # The band is unaffected: the two are independent facts.
    lo, hi = env.band_deg(tilt=(0.0, 0.0))
    assert abs(lo - 16.0) < 1e-9 and abs(hi - 301.0) < 1e-9


def test_the_datum_survives_a_save_and_reload():
    """The hop that used to lose it."""
    env = _env()
    doc = env.document()
    for row in doc["per_tilt"]:
        assert "wrist3_start_deg" in row, (
            "the document dropped the datum for %r, so a saved measurement "
            "cannot answer what wrist angle a rotation asks for" % (row["tilt"],))
    back = NS["_cable_envelope_from_doc"](doc, source="round trip")
    for row in ROWS:
        got = back.wrist3_datum_deg(tilt=(row["tx_deg"], row["ty_deg"]))
        assert abs(got - row["wrist3_start_deg"]) < 1e-9, row["tilt"]


def test_the_datum_reaches_the_cell_manifest():
    """The manifest is what travels to the cell; the capture file does not."""
    env = _env()
    block = NS["cable_envelope_manifest_block"](model=env, mode="hardware")
    assert "wrist3_start_deg" in block, (
        "the manifest block omits the datum, so a cell configured from the "
        "manifest is back to a constant in a config:\n%s" % (block,))
    for row in ROWS:
        assert ("wrist3_start_deg: %.4f" % (row["wrist3_start_deg"],)) in block, (
            "%s's datum is missing from the manifest block" % (row["tilt"],))


def test_the_real_measurement_from_this_cell_carries_it():
    """Against the file the cell actually wrote, not a fixture.

    This is what makes the change worth anything: the datum was already being
    measured and written out, on every row, and was thrown away on load.
    """
    if not os.path.exists(CELL_FILE):
        import pytest
        pytest.skip("this cell's cable_limits file is not on this machine")
    doc = json.load(open(CELL_FILE))
    rows = [r for r in doc["per_tilt"] if r.get("lo_deg") is not None]
    assert rows, "the measurement has no usable rows"
    have = [r for r in rows if r.get("wrist3_start_deg") is not None]
    assert len(have) == len(rows), (
        "%d of %d measured rows carry no datum" % (len(rows) - len(have), len(rows)))
    env = NS["_cable_envelope_from_doc"](doc, source=CELL_FILE)
    for row in rows:
        got = env.wrist3_datum_deg(tilt=(row["tx_deg"], row["ty_deg"]))
        assert abs(got - row["wrist3_start_deg"]) < 1e-6, row["tilt"]
    # The number the config hardcoded, recovered from the measurement instead.
    untilted = env.wrist3_datum_deg(tilt=(0.0, 0.0))
    assert abs(untilted - 166.09) < 0.01, untilted
    # And it genuinely varies with tilt, which a single constant cannot express.
    spread = [env.wrist3_datum_deg(tilt=(r["tx_deg"], r["ty_deg"])) for r in rows]
    assert max(spread) - min(spread) > 10.0, (
        "the datum barely varies across tilts (%.2f deg), so the per-tilt lookup "
        "is not buying anything -- check the measurement" % (max(spread)-min(spread),))


def test_the_band_edges_are_whole_steps_from_the_datum():
    """The proof that the datum IS the origin the band was swept from.

    capture_cable_envelope() starts at the reference pose and steps the wrist by
    step_deg until the cables bind, so every band edge must be an exact multiple
    of step_deg away from the datum of its own row. On this cell, with 15 deg
    steps, all eight rows come out on multiples of 15:

        no tilt      -150 .. +135      tilt x -40     +0 .. +90
        tilt x +20   -120 .. +165      tilt y +20    -90 .. +135
        tilt x +40    -75 .. +165      tilt y -20   -105 .. +120
        tilt x -20    -45 .. +105      tilt y -40    -75 .. +135

    If this fails, the datum and the band are not from the same measurement and
    adding a rotation to one to test against the other is meaningless.
    """
    if not os.path.exists(CELL_FILE):
        import pytest
        pytest.skip("this cell's cable_limits file is not on this machine")
    doc = json.load(open(CELL_FILE))
    step = float(doc.get("step_deg") or 0.0)
    if step <= 0:
        import pytest
        pytest.skip("the measurement records no step_deg")
    env = NS["_cable_envelope_from_doc"](doc, source=CELL_FILE)
    for row in doc["per_tilt"]:
        if row.get("lo_deg") is None:
            continue
        tilt = (row["tx_deg"], row["ty_deg"])
        datum = env.wrist3_datum_deg(tilt=tilt)
        lo, hi = env.band_deg(tilt=tilt)
        for edge, name in ((lo, "lo"), (hi, "hi")):
            steps = (edge - datum)/step
            assert abs(steps - round(steps)) < 1e-6, (
                "%s: %s edge is %.4f deg from the datum, which is %.4f steps of "
                "%.1f deg -- not a whole number, so the two are not from one "
                "measurement" % (row["tilt"], name, edge - datum, steps, step))


def test_the_untilted_datum_misstates_the_tilted_limits():
    """Why one constant is not enough, in the refusing AND permitting directions.

    Using the untilted datum at every tilt is what a config constant can express.
    On this cell at tilt x -40 the true holdable rotations are +0..+90; the
    untilted arithmetic reports +13..+103 -- so it would refuse a rotation of zero
    and PERMIT one of +100, which the measurement says the cables cannot hold.
    Under-permitting is an inconvenience; over-permitting is the cables.
    """
    if not os.path.exists(CELL_FILE):
        import pytest
        pytest.skip("this cell's cable_limits file is not on this machine")
    doc = json.load(open(CELL_FILE))
    env = NS["_cable_envelope_from_doc"](doc, source=CELL_FILE)
    zero = env.wrist3_datum_deg(tilt=(0.0, 0.0))
    worst_permissive = 0.0
    for row in doc["per_tilt"]:
        if row.get("lo_deg") is None:
            continue
        tilt = (row["tx_deg"], row["ty_deg"])
        datum = env.wrist3_datum_deg(tilt=tilt)
        lo, hi = env.band_deg(tilt=tilt)
        # The rotation range each datum implies at this tilt.
        true_hi = hi - datum
        assumed_hi = hi - zero
        worst_permissive = max(worst_permissive, assumed_hi - true_hi)
    assert worst_permissive > 5.0, (
        "the untilted datum is within %.1f deg of the per-tilt one everywhere, so "
        "this test is not demonstrating the hazard it claims" % (worst_permissive,))


def test_rotation_is_graded_against_the_datum_and_band_at_one_tilt():
    """Both ends of the comparison taken at the same tilt.

    At tilt x -40 this cell measures a datum of 179.19 in a band of
    179.19..269.19 -- so it holds 0..+90 of rotation there and NOTHING negative.
    Using the untilted datum of 166.09 against that band would report a rotation
    of 0 as already 13 degrees outside it, which is wrong in the refusing
    direction.
    """
    env = _env()
    for tx, ty in ((0.0, 0.0), (-40.0, 0.0), (40.0, 0.0)):
        datum = env.wrist3_datum_deg(tilt=(tx, ty))
        lo, hi = env.band_deg(tilt=(tx, ty))
        # Zero rotation must sit inside the band at the tilt it was measured at:
        # the arm was holding that pose when the band was swept from it.
        assert lo - 1e-6 <= datum <= hi + 1e-6, (
            "tilt (%.0f, %.0f): the datum %.2f is outside its own band %.2f..%.2f"
            % (tx, ty, datum, lo, hi))


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (name, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
