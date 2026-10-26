"""Merging several sweeps of one cell's cable envelope, per tilt.

WHY THERE IS A MERGE AT ALL. Nothing measures where the end-of-arm loom binds.
An operator steps wrist 3 round and says when the cables look taut, so two sweeps
of a cell whose cabling has not been touched come out about a step of the sweep
apart. That spread is in the judgement, not in the cables: neither sweep is the
correction of the other, and keeping only the newer one discards travel that was
watched and found slack.

So `cable_envelope_merge()` takes, at each tilt, the lowest low end and the
highest high end any sweep reported there. What has to hold:

  * the merged band at a tilt is never narrower than any sweep's band at that
    tilt -- otherwise the merge lost measured travel;
  * and never wider than the widest of them -- otherwise it is claiming slack
    nobody watched;
  * a tilt only one sweep visited keeps that sweep's band unchanged;
  * nothing is reconciled by adding or subtracting a turn. The cable physically
    wraps, so +200 and -160 degrees are the same tool orientation and opposite
    cable states, and a sweep recorded on the other branch is a different
    measurement rather than the same one written differently;
  * the datum is not merged. `wrist3_start_deg` says where zero rotation sat in
    the joint state a sweep started from -- a measurement of how the tooling is
    bolted on, not a bound -- so "most permissive" has no meaning for it.

Runs on the host: pure model, numpy only, no ROS and no robot.

    python3 cable_envelope_merge_test.py
    python3 -m pytest cable_envelope_merge_test.py
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _load():
    src = open(os.path.join(HERE, "cable_envelope.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "cable_envelope_merge_test"}
    exec(compile(src, "cable_envelope.dpi", "exec"), ns)
    return ns


NS = _load()

#: An earlier sweep: a five-point cross, no datum recorded.
EARLIER = [
    {"tilt": "no tilt", "tx_deg": 0.0, "ty_deg": 0.0,
     "lo_deg": 31.0, "hi_deg": 301.0},
    {"tilt": "tilt x +40 deg", "tx_deg": 40.0, "ty_deg": 0.0,
     "lo_deg": 87.0, "hi_deg": 327.0},
    {"tilt": "tilt x -40 deg", "tx_deg": -40.0, "ty_deg": 0.0,
     "lo_deg": 179.0, "hi_deg": 254.0},
    {"tilt": "tilt y +20 deg", "tx_deg": 0.0, "ty_deg": 20.0,
     "lo_deg": 63.0, "hi_deg": 303.0},
    {"tilt": "tilt y -20 deg", "tx_deg": 0.0, "ty_deg": -20.0,
     "lo_deg": 30.0, "hi_deg": 255.0},
]

#: A later sweep of the same cell: the same five tilts plus two the earlier one
#: never visited, and a datum on every row.
LATER = [
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
    {"tilt": "tilt x +20 deg", "tx_deg": 20.0, "ty_deg": 0.0,
     "lo_deg": 38.0, "hi_deg": 323.0, "wrist3_start_deg": 158.0},
    {"tilt": "tilt y -40 deg", "tx_deg": 0.0, "ty_deg": -40.0,
     "lo_deg": 90.0, "hi_deg": 300.0, "wrist3_start_deg": 165.0},
]


def _env(rows, measured, step_deg=15.0):
    return NS["cable_envelope_from_rows"](rows, step_deg=step_deg,
                                         cell="test", mode="hardware",
                                         measured=measured)


def _merge(*sources, **kw):
    """The merge, sources NEWEST FIRST, which is the order it documents."""
    return NS["cable_envelope_merge"](list(sources), **kw)


def _rows_of(env):
    """{(tx, ty): (lo_deg, hi_deg)} off the model's own samples."""
    return {(s[0], s[1]): (math.degrees(s[2]), math.degrees(s[3]))
            for s in env.samples}


def _sources():
    return (_env(LATER, "later"), _env(EARLIER, "earlier"))


def test_each_tilt_takes_the_most_permissive_end_recorded_there():
    later, earlier = _sources()
    got = _rows_of(_merge(later, earlier))
    want = {}
    for rows in (LATER, EARLIER):
        for r in rows:
            key = (r["tx_deg"], r["ty_deg"])
            lo, hi = want.get(key, (r["lo_deg"], r["hi_deg"]))
            want[key] = (min(lo, r["lo_deg"]), max(hi, r["hi_deg"]))
    assert set(got) == set(want), (sorted(got), sorted(want))
    for key, (lo, hi) in want.items():
        assert abs(got[key][0] - lo) < 1e-9 and abs(got[key][1] - hi) < 1e-9, (
            "tilt %s merged to %.4f..%.4f, most permissive is %.4f..%.4f"
            % (key, got[key][0], got[key][1], lo, hi))
    # The one this cell actually turns on: the two sweeps put the untilted low
    # end 15 degrees apart and the high ends level, so the merge widens the low
    # end only.
    assert abs(got[(0.0, 0.0)][0] - 16.0) < 1e-9
    assert abs(got[(0.0, 0.0)][1] - 301.0) < 1e-9
    # And one where each sweep won a different end.
    assert abs(got[(40.0, 0.0)][0] - 72.0) < 1e-9
    assert abs(got[(40.0, 0.0)][1] - 327.0) < 1e-9


def test_a_tilt_only_one_sweep_visited_survives_unchanged():
    """There is nothing to widen it with, and dropping it would lose the only
    band known at that tilt."""
    later, earlier = _sources()
    got = _rows_of(_merge(later, earlier))
    only_later = [r for r in LATER
                  if not any(r["tx_deg"] == e["tx_deg"] and r["ty_deg"] == e["ty_deg"]
                             for e in EARLIER)]
    assert len(only_later) == 2, only_later
    for r in only_later:
        lo, hi = got[(r["tx_deg"], r["ty_deg"])]
        assert abs(lo - r["lo_deg"]) < 1e-9 and abs(hi - r["hi_deg"]) < 1e-9, (
            "%s came out %.4f..%.4f, measured %.4f..%.4f"
            % (r["tilt"], lo, hi, r["lo_deg"], r["hi_deg"]))


def test_the_merge_is_never_narrower_and_never_wider_than_the_sources():
    """The two invariants that say a merge softened nothing and invented nothing."""
    later, earlier = _sources()
    merged = _rows_of(_merge(later, earlier))
    per_source = [_rows_of(later), _rows_of(earlier)]
    for key, (lo, hi) in merged.items():
        seen = [s[key] for s in per_source if key in s]
        assert seen, key
        widest_lo = min(a for a, _b in seen)
        widest_hi = max(b for _a, b in seen)
        # Never narrower: it must reach every end any sweep reported.
        assert lo <= widest_lo + 1e-9 and hi >= widest_hi - 1e-9, (
            "tilt %s merged to %.4f..%.4f, which is inside one of %s"
            % (key, lo, hi, seen))
        # Never wider: every end must BE one a sweep reported.
        assert abs(lo - widest_lo) < 1e-9, (
            "tilt %s low end %.4f is below every recorded one %s"
            % (key, lo, [a for a, _b in seen]))
        assert abs(hi - widest_hi) < 1e-9, (
            "tilt %s high end %.4f is above every recorded one %s"
            % (key, hi, [b for _a, b in seen]))


def test_the_order_of_the_sources_does_not_change_the_bands():
    """The bands are a min and a max, so they cannot depend on the order. Only
    the datum does, and that is stated."""
    later, earlier = _sources()
    assert _rows_of(_merge(later, earlier)) == _rows_of(_merge(earlier, later))


def test_one_source_merges_to_itself():
    later, _earlier = _sources()
    merged = _merge(later)
    assert _rows_of(merged) == _rows_of(later)
    for s in later.samples:
        assert (merged.wrist3_datum_deg(tilt=(s[0], s[1]))
                == later.wrist3_datum_deg(tilt=(s[0], s[1])))


def test_nothing_is_reconciled_by_a_turn():
    """A sweep recorded on the other branch is a DIFFERENT measurement.

    The cable wraps, so a band of -344..-59 and a band of 16..301 describe the
    same tool orientations and opposite cable states. Folding one onto the other
    would make two sweeps look like agreement; the merge keeps both absolute
    angles, which widens the band past a full turn and is visible as the fault it
    is rather than hidden.
    """
    a = _env([{"tilt": "no tilt", "tx_deg": 0.0, "ty_deg": 0.0,
               "lo_deg": 16.0, "hi_deg": 301.0}], "on this branch")
    b = _env([{"tilt": "no tilt", "tx_deg": 0.0, "ty_deg": 0.0,
               "lo_deg": 16.0 - 360.0, "hi_deg": 301.0 - 360.0}],
             "a turn away")
    lo, hi = _rows_of(_merge(a, b))[(0.0, 0.0)]
    assert abs(lo - (16.0 - 360.0)) < 1e-9 and abs(hi - 301.0) < 1e-9, (lo, hi)
    assert hi - lo > 360.0, (
        "the two sweeps were folded onto one another (%.1f..%.1f), so a "
        "measurement on the other branch was silently taken for agreement"
        % (lo, hi))


def test_an_angle_and_its_turn_are_still_not_interchangeable():
    """The merged model grades ABSOLUTE wrist_3 angles, like every other one."""
    merged = _merge(*_sources())
    assert merged.contains(math.radians(29.0), tilt=(0.0, 0.0))
    assert not merged.contains(math.radians(29.0 - 360.0), tilt=(0.0, 0.0))
    assert not merged.contains(math.radians(29.0 + 360.0), tilt=(0.0, 0.0))


def test_the_datum_is_taken_from_one_sweep_and_not_combined():
    """It says where the tooling put the wrist, not how far the wrist may go.

    An extreme of two datums would move where zero rotation is believed to be,
    which is a different and falsifiable claim from "the cables held this far".
    So each merged tilt carries ONE datum, from the first source listed that
    recorded one -- which is why the sources are given newest first.
    """
    later, earlier = _sources()
    merged = _merge(later, earlier)
    for r in LATER:
        got = merged.wrist3_datum_deg(tilt=(r["tx_deg"], r["ty_deg"]))
        assert abs(got - r["wrist3_start_deg"]) < 1e-9, (r["tilt"], got)
    # The datum still sits inside the band at its own tilt, which is what makes
    # grading a rotation against the pair mean anything.
    for s in merged.samples:
        tilt = (s[0], s[1])
        lo, hi = merged.band_deg(tilt=tilt)
        datum = merged.wrist3_datum_deg(tilt=tilt)
        assert lo - 1e-6 <= datum <= hi + 1e-6, (tilt, datum, lo, hi)


def test_a_tilt_whose_only_datum_is_in_the_older_sweep_keeps_it():
    """Precedence is "the first source that has one", not "the newest source"."""
    later = _env([r for r in LATER if (r["tx_deg"], r["ty_deg"]) != (0.0, 20.0)]
                 + [{"tilt": "tilt y +20 deg", "tx_deg": 0.0, "ty_deg": 20.0,
                     "lo_deg": 78.0, "hi_deg": 303.0}], "later")
    earlier = _env([dict(r, wrist3_start_deg=170.0) for r in EARLIER], "earlier")
    merged = _merge(later, earlier)
    assert abs(merged.wrist3_datum_deg(tilt=(0.0, 20.0)) - 170.0) < 1e-9
    # Where both have one, the newer wins.
    assert abs(merged.wrist3_datum_deg(tilt=(0.0, 0.0)) - 166.0) < 1e-9


def test_sweeps_from_different_home_orientations_are_refused():
    """A tilt is measured FROM home_quat, so two sweeps taken from different
    reference orientations do not describe the same pose at the same (tx, ty)."""
    later = _env(LATER, "later")
    turned = NS["cable_envelope_from_rows"](
        EARLIER, step_deg=15.0, cell="test", mode="hardware",
        measured="earlier", home_quat=(0.0, 0.0, 0.3826834, 0.9238795))
    try:
        _merge(later, turned)
    except ValueError:
        return
    raise AssertionError("bands measured from two different home orientations "
                         "were merged as if their tilts were the same poses")


def test_the_merge_records_which_sweeps_went_into_it():
    """Without it nobody reading the result can tell a merged band from a
    mis-transcribed one, and the invariants above cannot be checked."""
    merged = _merge(*_sources())
    assert merged.merged_from == ["later", "earlier"], merged.merged_from
    assert "later" in merged.measured and "earlier" in merged.measured
    assert merged.step_deg == 15.0
    # And it survives the round trip through the document and back.
    back = NS["_cable_envelope_from_doc"](merged.document(), "round trip")
    assert back.merged_from == merged.merged_from
    assert _rows_of(back) == _rows_of(merged)


def test_a_step_the_sweeps_disagree_on_is_dropped_rather_than_guessed():
    """step_deg is a property of one sweep. A merged row has no single step when
    the sweeps were stepped differently, and saying nothing beats picking one."""
    merged = _merge(_env(LATER, "later", step_deg=15.0),
                    _env(EARLIER, "earlier", step_deg=20.0))
    assert merged.step_deg is None


def test_the_manifest_block_carries_the_merge():
    """The manifest is what travels to the work cell; the sweep files do not."""
    merged = _merge(*_sources())
    block = NS["cable_envelope_manifest_block"](model=merged, mode="hardware")
    assert "merged_from: [\"later\", \"earlier\"]" in block, block
    # The span the rows cover, so "outside this nothing is known" is stated.
    assert "swept_deg: [16.0000, 327.0000]" in block, block
    for key, (lo, hi) in _rows_of(merged).items():
        assert ("lo_deg: %.4f, hi_deg: %.4f" % (lo, hi)) in block, (key, lo, hi)
    # And the block reads back as the same model.
    import yaml
    doc = yaml.safe_load(block)["cable_envelope"]["hardware"]
    doc["home_quat"] = yaml.safe_load(block)["cable_envelope"]["home_quat"]
    back = NS["_cable_envelope_from_doc"](doc, "the manifest block")
    assert _rows_of(back) == _rows_of(merged)
    assert back.merged_from == merged.merged_from


def test_merging_nothing_is_an_error_not_an_empty_band():
    try:
        _merge()
    except ValueError:
        return
    raise AssertionError("merging no measurements produced a cable envelope")


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
