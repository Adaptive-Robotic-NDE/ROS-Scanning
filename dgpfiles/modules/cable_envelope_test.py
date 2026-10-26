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
# File:  cable_envelope_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the per-tilt cable envelope model and the wrist-3 guard built
#   on it.
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
"""Check the cable envelope model in cable_envelope.dpi.

Two faults these guard against, both observed on real hardware:

  * Intersecting five separately-measured tilt bands into one 75 degree window
    (179.6..254.6 deg) and enforcing that at every pose whatever the tilt.
    That refuses moves at 257, 255 and 168..250 deg which are inside the band
    for the tilt actually commanded; an untilted pose has 270 degrees of legal
    travel.

  * Nothing enforcing anything at all outside a capture scenario. An unguarded
    run_automated_scan() commands the MTC node wrist_3 = 29.3, 29.3, 21.9,
    -32.3, -55.1, -113.0 deg, in that order -- below the measured tight point
    of 31.3, through zero, and on to -113 -- with the emergency stop as the only
    thing left to stop it. Those six numbers are in this file, and they are all
    refused.

Runs on the HOST with numpy only: no ROS, no container, no robot, no dgpy.
The model is pure arithmetic over a measurement, which is exactly the kind of
thing that must be testable without the cell -- a safety check that needs the
cell to test is a check nobody runs.

    python3 cable_envelope_test.py          # prints every case
    python3 -m pytest cable_envelope_test.py
"""
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "cable_envelope_test"}
exec(compile(_load("cable_envelope.dpi"), "cable_envelope.dpi", "exec"), ns)

CableEnvelope = ns["CableEnvelope"]
CableRefused = ns["CableRefused"]
load_cable_envelope = ns["load_cable_envelope"]
cable_envelope_from_rows = ns["cable_envelope_from_rows"]
wrist3_tilt_of = ns["wrist3_tilt_of"]
check_wrist3 = ns["check_wrist3"]
quat_matrix = ns["_cable_quat_matrix"]

FAILURES = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILURES.append(what)


def raises(fn, what, exc=CableRefused):
    try:
        fn()
    except exc:
        check(True, what)
        return
    except Exception as e:
        check(False, "%s (raised %s: %s)" % (what, type(e).__name__, e))
        return
    check(False, "%s (did not raise)" % (what,))


def q_axis(axis, deg):
    """A world-frame rotation quaternion, (x, y, z, w) -- _cap_q_axis's twin."""
    a = np.asarray(axis, float)
    a = a/np.linalg.norm(a)
    s = math.sin(math.radians(deg)/2.0)
    return (a[0]*s, a[1]*s, a[2]*s, math.cos(math.radians(deg)/2.0))


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def joints(wrist3_deg):
    """A plausible UR configuration with this wrist_3, in radians."""
    return [0.1, -1.2, 1.4, -1.7, -1.55, math.radians(wrist3_deg)]


# ==============================================================================
# A MEASUREMENT: ex-ur10e-1 hardware, from capture_cable_envelope().
# ==============================================================================
# The 2026-09-14 sweep of this cell, in SCHEMA 1 (labels, no tx_deg/ty_deg),
# because reading a schema 1 file is a requirement -- a measurement written
# before the schema carried tilt coordinates is still on the controlling computer.
#
# It is the earlier of the two sweeps the manifest merges, and it is used here
# for two things: as the fixture the band arithmetic is checked on, and as one of
# the recorded measurements the merged manifest is graded against below.
# ==============================================================================
MEASURED = [
    {"tilt": "no tilt",         "lo_deg": 31.338424555636937, "hi_deg": 301.33842455563695},
    {"tilt": "tilt x +40 deg",  "lo_deg": 87.34219315297075,  "hi_deg": 327.3421931529707},
    {"tilt": "tilt x -40 deg",  "lo_deg": 179.55253199567213, "hi_deg": 254.55253199567213},
    {"tilt": "tilt y +20 deg",  "lo_deg": 63.598971284101005, "hi_deg": 303.598971284101},
    {"tilt": "tilt y -20 deg",  "lo_deg": 30.28146644195578,  "hi_deg": 255.28146644195579},
]

#: The wrist_3 values an unguarded automated scan commanded the MTC node, in the
#: order they were commanded. Every one of them must be refused.
FAILURE_ANGLES_DEG = (29.3, 29.3, 21.9, -32.3, -55.1, -113.0)

env = cable_envelope_from_rows(MEASURED, step_deg=15.0, cell="ex-ur10e-1",
                               mode="hardware", measured="2026-09-14T10:22:07")
print(env.describe())
print("")


def band(tx=0.0, ty=0.0, **kw):
    return env.band_deg(tilt=(tx, ty), **kw)


# --- the old schema is readable, and its five rows came back ------------------
check(len(env.samples) == 5, "the five measured tilts are all loaded")
check([s[:2] for s in env.samples] == [(0.0, 0.0), (40.0, 0.0), (-40.0, 0.0),
                                       (0.0, 20.0), (0.0, -20.0)],
      "the old schema's labels parse to the right tilt coordinates")

# --- every measured tilt returns ITS OWN band, exactly ------------------------
for row in MEASURED:
    tx, ty = ns["_cable_parse_label"](row["tilt"])
    lo, hi = band(tx, ty)
    check(abs(lo - row["lo_deg"]) < 1e-9 and abs(hi - row["hi_deg"]) < 1e-9,
          "%-16s comes back as measured: %.1f..%.1f deg"
          % (row["tilt"], row["lo_deg"], row["hi_deg"]))

# --- an untilted pose gets 270 degrees, not the 75 of the intersection --------
lo0, hi0 = band(0.0, 0.0)
check(abs(lo0 - 31.338424555636937) < 1e-9 and abs(hi0 - 301.33842455563695) < 1e-9,
      "no tilt: the full 31.3..301.3 deg (%.1f..%.1f)" % (lo0, hi0))
check(not (abs(lo0 - 179.55) < 0.5 or abs(hi0 - 254.56) < 0.5),
      "no tilt is NOT the 179.6..254.6 intersection of all five tilts")
check(hi0 - lo0 > 260.0,
      "no tilt has %.0f deg of travel, not the intersection's 75" % (hi0 - lo0))

# --- interpolation lies between the two neighbours ---------------------------
lo_m20, hi_m20 = band(-20.0, 0.0)
lo_m40, hi_m40 = band(-40.0, 0.0)
check(lo_m40 > lo_m20 > lo0 and hi_m40 < hi_m20 < hi0,
      "tx -20 (%.1f..%.1f) lies strictly between no tilt and x -40"
      % (lo_m20, hi_m20))
check(abs(lo_m20 - 0.5*(lo0 + lo_m40)) < 1e-9
      and abs(hi_m20 - 0.5*(hi0 + hi_m40)) < 1e-9,
      "tx -20 is the exact midpoint of its neighbours (lo and hi separately)")
lo_p10, hi_p10 = band(0.0, 10.0)
check(abs(lo_p10 - 0.5*(31.338424555636937 + 63.598971284101005)) < 1e-9,
      "ty +10 interpolates along y the same way (%.1f..%.1f)" % (lo_p10, hi_p10))

# --- a diagonal pays for both tilts -------------------------------------------
# THE RULE, and the one place it departs from a plain intersection.
# Taking the literal max-of-the-los / min-of-the-his of the two ABSOLUTE
# interpolated bands is wrong at three of the five measured samples, in the
# over-restrictive direction: at tx +40 the y row is evaluated at ty = 0, where
# it returns the UNTILTED band -- a statement about the origin, not about
# (+40, 0) -- and intersecting with it cuts the measured 327.3 back to 301.3.
# Twenty-six degrees of confirmed-slack travel lost to arithmetic is the same
# mistake as the global intersection, one axis smaller. So the two rows are
# combined as DEVIATIONS from the untilted band they share, which is exact on
# both rows and still takes both tilts' restrictions on a diagonal.
lo_d, hi_d = band(-40.0, -20.0)
want_lo = 179.55253199567213 + 30.28146644195578 - 31.338424555636937
want_hi = 254.55253199567213 + 255.28146644195579 - 301.33842455563695
check(abs(lo_d - want_lo) < 1e-9 and abs(hi_d - want_hi) < 1e-9,
      "a diagonal composes both axes' deviations (%.1f..%.1f)" % (lo_d, hi_d))
check(lo_d > lo0 and hi_d < min(254.55253199567213, 255.28146644195579) + 1e-9,
      "the (-40, -20) diagonal is tighter than either row alone")
# ...but never past anything measured.
lo_pp, hi_pp = band(40.0, 20.0)
check(abs(hi_pp - 327.3421931529707) < 1e-9,
      "a diagonal whose deviations both widen is clamped to the widest measured "
      "hi (%.1f, not %.1f)" % (hi_pp, 301.33842455563695 + 26.0 + 2.26))
check(lo_pp > 87.34219315297075,
      "and is still tighter at the bottom than either row (%.1f)" % (lo_pp,))
check(band(0.0, 0.0)[0] >= min(r["lo_deg"] for r in MEASURED) - 1e-9
      and band(0.0, 0.0)[1] <= max(r["hi_deg"] for r in MEASURED) + 1e-9,
      "nothing the model returns lies outside the measured extremes")

# --- outside the measured hull: refuse, never extrapolate ---------------------
raises(lambda: band(41.0, 0.0), "tx +41 deg is outside the hull and raises")
raises(lambda: band(-60.0, 0.0), "tx -60 deg is outside the hull and raises")
raises(lambda: band(0.0, 25.0), "ty +25 deg is outside the hull and raises")
raises(lambda: band(0.0, -21.0), "ty -21 deg is outside the hull and raises")
check(abs(band(40.0 + 0.4, 0.0)[0] - 87.34219315297075) < 1e-9,
      "a rounding-sized 0.4 deg past the +40 sample is still the +40 sample")
lo_c, hi_c = band(70.0, 0.0, allow_outside=True)
check(abs(lo_c - 87.34219315297075) < 1e-9,
      "allow_outside=True clamps to the nearest measured tilt instead of raising")

# --- margin_deg ---------------------------------------------------------------
lo_m, hi_m = band(0.0, 0.0, margin_deg=5.0)
check(abs(lo_m - (lo0 + 5.0)) < 1e-9 and abs(hi_m - (hi0 - 5.0)) < 1e-9,
      "margin_deg narrows both ends")
check(ns["CABLE_ENVELOPE_MARGIN_DEG"] == 0.0,
      "the default margin is 0 -- reached_deg is already one step inside 'tight'")

# ==============================================================================
# THE SIX ANGLES AN UNGUARDED SCAN COMMANDED
# ==============================================================================
print("")
for deg in FAILURE_ANGLES_DEG:
    raises(lambda d=deg: check_wrist3([joints(d)], "an unguarded scan",
                                      model=env),
           "unguarded scan: wrist 3 %+7.1f deg is refused at zero tilt" % (deg,))
inside = [d for d in FAILURE_ANGLES_DEG if lo0 <= d <= hi0]
check(not inside,
      "not one of the six is inside the untilted band (%.1f..%.1f)" % (lo0, hi0))

# The refusal has to say what happened, not just that something did.
try:
    check_wrist3([joints(-113.0)], "an unguarded scan", model=env)
    msg = ""
except CableRefused as e:
    msg = str(e)
check("-113.0" in msg and "31.3" in msg and "301.3" in msg and "tilt" in msg,
      "the refusal names the angle, the band and the tilt:\n      %s"
      % (msg.splitlines()[0] if msg else "(nothing raised)",))

# A whole move is checked, not just its end.
raises(lambda: check_wrist3([joints(200.0), joints(-113.0)], "a move", model=env),
       "a move whose END is legal but which passes -113 is refused")

# ==============================================================================
# ABSOLUTE ANGLE: no 360 degree wrapping, anywhere
# ==============================================================================
print("")
check(env.contains(math.radians(200.0)),
      "+200 deg is INSIDE the untilted band")
check(not env.contains(math.radians(-160.0)),
      "-160 deg is OUTSIDE it, though it is the same tool orientation")
check_wrist3([joints(200.0)], "a +200 deg move", model=env)
check(True, "check_wrist3 passes +200 deg and refuses -160 deg (above)")
raises(lambda: check_wrist3([joints(-160.0)], "a -160 deg move", model=env),
       "-160 deg is refused rather than folded onto +200")
for deg in (310.0, 391.3, -328.7, 661.3):
    # 391.3 = 31.3 + 360 and -328.7 = 31.3 - 360: both are the same tool
    # orientation as the legal 31.3 and neither is offered as a substitute.
    raises(lambda d=deg: check_wrist3([joints(d)], "a wrapped angle", model=env),
           "%+7.1f deg is refused, not wrapped into the band" % (deg,))

# ==============================================================================
# TILT FROM A COMMANDED ORIENTATION
# ==============================================================================
print("")
home = (0.0, 0.0, 0.0, 1.0)
check(np.allclose(wrist3_tilt_of(home, home), (0.0, 0.0)),
      "home itself is (0, 0) tilt")
for axis, deg, want in ((( 1, 0, 0),  40.0, ( 40.0,   0.0)),
                        ((1, 0, 0), -40.0, (-40.0,   0.0)),
                        ((0, 1, 0),  20.0, (  0.0,  20.0)),
                        ((0, 1, 0), -20.0, (  0.0, -20.0)),
                        ((1, 0, 0),  12.5, ( 12.5,   0.0))):
    got = wrist3_tilt_of(q_mul(q_axis(axis, deg), home), home)
    check(np.allclose(got, want, atol=1e-9),
          "a world %s tilt of %+.1f deg reads back as (%.1f, %.1f)"
          % ("x" if axis[0] else "y", deg, got[0], got[1]))

# The band must not drift as wrist 3 turns: the envelope is MEASURED by holding
# a tilt and sweeping the wrist through it.
tilted = q_mul(q_axis((1, 0, 0), -40.0), home)
for roll in (-170.0, -90.0, 0.0, 45.0, 166.0, 200.0):
    # wrist 3 turns about the TOOL axis, so the roll multiplies on the right.
    rolled = q_mul(tilted, q_axis((0, 0, 1), roll))
    got = wrist3_tilt_of(rolled, home)
    check(np.allclose(got, (-40.0, 0.0), atol=1e-6),
          "x -40 still reads (%.1f, %.1f) with %+.0f deg of wrist 3 on top"
          % (got[0], got[1], roll))

lo_q, hi_q = env.band_deg(quat=tilted)
check(abs(lo_q - 179.55253199567213) < 1e-9 and abs(hi_q - 254.55253199567213) < 1e-9,
      "a commanded x -40 orientation selects the x -40 band (%.1f..%.1f)"
      % (lo_q, hi_q))
check_wrist3([joints(200.0)], "a tilted move", quats=[tilted], model=env)
check(True, "200 deg is legal at x -40 (inside 179.6..254.6)")
raises(lambda: check_wrist3([joints(280.0)], "a tilted move", quats=[tilted],
                            model=env),
       "280 deg is legal untilted and REFUSED at x -40 -- the tilt is used")
check_wrist3([joints(280.0)], "an untilted move", quats=[home], model=env)
check(True, "the same 280 deg passes with no tilt: one band per tilt, not a global one")

# Each pose of a move gets its own tilt when the counts line up.
raises(lambda: check_wrist3([joints(280.0), joints(280.0)], "a move that tilts",
                            quats=[home, tilted], model=env),
       "a move that starts untilted and ends at x -40 is refused on the tilted end")

# A tilt nobody measured refuses rather than guessing.
steep = q_mul(q_axis((1, 0, 0), 75.0), home)
raises(lambda: check_wrist3([joints(200.0)], "a steep move", quats=[steep],
                            model=env),
       "a 75 deg tilt (never measured) refuses instead of extrapolating")

# ==============================================================================
# SCHEMA: what gets written, and reading it back
# ==============================================================================
print("")
doc = env.document()
check(doc["schema"] == "cable_envelope/2", "the document declares schema 2")
check("lo_rad" not in doc and "hi_rad" not in doc,
      "there is NO top-level enforced band any more")
check(doc["intersection_not_enforced"]["lo_deg"] is not None
      and "NOT the limit" in doc["intersection_not_enforced"]["why"],
      "the old intersection is kept as information, named so nothing enforces it")
check(all("tx_deg" in r and "ty_deg" in r for r in doc["per_tilt"]),
      "every row carries its tilt coordinates outright")
again = ns["_cable_envelope_from_doc"](doc, "round trip")
check(again.band_deg(tilt=(0.0, 0.0)) == env.band_deg(tilt=(0.0, 0.0))
      and again.band_deg(tilt=(-40.0, 0.0)) == env.band_deg(tilt=(-40.0, 0.0)),
      "schema 2 round-trips through the document")

# An unmeasured cell does not get an invented band.
check(load_cable_envelope is not None, "load_cable_envelope is exported")
raises(lambda: ns["wrist3_band"](model=None), "no measurement at all is a KeyError, "
       "not a default band", exc=KeyError)
check(check_wrist3([joints(-113.0)], "unmeasured", model=None) is None,
      "with no measurement check_wrist3 says so once and does not refuse "
      "(the cable test is itself a motion)")
raises(lambda: check_wrist3([joints(-113.0)], "unmeasured", model=None, require=True),
       "require=True refuses when there is no measurement")

# A degenerate measurement is rejected outright rather than silently inverted.
raises(lambda: CableEnvelope([(0.0, 0.0, 1.0, 0.5, "backwards")]),
       "an inside-out sample is rejected", exc=ValueError)

# ==============================================================================
# THE WORK-CELL MANIFEST
# ==============================================================================
# The measurement file lives under CAPTURE_ROOT on whichever machine ran the
# test; the manifest is what every other session can see, and is therefore the
# default source. A typo in it is a wrong cable limit, so it is checked here
# rather than on the production system.
print("")
MANIFEST = os.path.join(
    HERE, "..", "..", "..", "ros_ws", "src", "tari_site", "tari_cell_configs",
    "config", "cells", "ex-ur10e-1.yaml")
# THE FILE HAS TO BE THERE. A skip when the tree is not where this expects it is
# the same shape of silence as the bug below: the check does not run, the suite
# still says all passed, and the number nobody checked is a cable limit.
check(os.path.exists(os.path.normpath(MANIFEST)),
      "the ex-ur10e-1 manifest is where this suite expects it")
try:
    import yaml
    data = yaml.safe_load(open(os.path.normpath(MANIFEST)))
except ImportError as _e:                    # no PyYAML in this interpreter
    data = None
    print("skip  the manifest check (%s)" % (_e,))
if data is not None:
    block = data.get("cable_envelope") or {}
    check(block.get("home_quat") == [0.0, 0.0, 0.0, 1.0],
          "the manifest declares the home orientation the tilts are measured from")

    # SIM CARRIES THE SAME BAND AS HARDWARE, and this check is the reason to
    # trust a simulated run's wrist angles at all.
    #
    # A null `sim:` block -- on the argument that a simulated cell has no cables
    # so it has no cable limit -- makes load_cable_envelope() return None in sim,
    # and None makes install_cable_guard() check NOTHING. The one fault class
    # that has twice stopped a run at the real cell (a solver taking a wrist_3
    # branch outside the loom's travel) would then be the one fault class
    # simulation is structurally incapable of catching, with every sim pass
    # silently saying nothing about it.
    #
    # The numbers do not describe the simulated cell's physics. They describe the
    # constraint the pipeline must plan within, and the pipeline is what is under
    # test. They must not DRIFT from hardware either: a sim enforcing a different
    # band than the work cell proves nothing, and would fail silently the same way.
    sim_b, hw_b = block.get("sim"), block.get("hardware")
    check(isinstance(sim_b, dict) and bool(sim_b.get("per_tilt")),
          "the manifest declares a band for sim, so the guard is exercised there")

    def _rows(b):
        return [(r.get("tx_deg"), r.get("ty_deg"), r.get("lo_deg"), r.get("hi_deg"))
                for r in ((b or {}).get("per_tilt") or [])]

    check(_rows(sim_b) == _rows(hw_b) and _rows(hw_b) != [],
          "sim and hardware carry identical per_tilt rows (%d of them)"
          % (len(_rows(hw_b)),))
    check((sim_b or {}).get("step_deg") == (hw_b or {}).get("step_deg"),
          "sim and hardware agree on step_deg (%r)" % ((hw_b or {}).get("step_deg"),))
    # home_quat lives one level up and is shared, so a per-mode override of it
    # would change every tilt lookup while leaving the rows identical.
    check((sim_b or {}).get("home_quat") is None
          and (hw_b or {}).get("home_quat") is None,
          "neither mode overrides home_quat")

    # THE MANIFEST IS A PER-TILT MERGE OF EVERY SWEEP RECORDED ON THIS CELL, so
    # it is graded against all of them and pinned to none. The limit is judged by
    # eye while the wrist is stepped round, two sweeps of cabling nobody touched
    # come out about a step apart, and the merge keeps whichever end either sweep
    # confirmed slack. So the manifest is DELIBERATELY WIDER than any single
    # sweep, and what has to hold instead is that it softened nothing and
    # invented nothing:
    #
    #   * at every tilt it declares, it is at least as permissive as every sweep
    #     that visited that tilt -- otherwise the merge lost measured travel;
    #   * and no more permissive than the widest of them -- otherwise it is
    #     claiming slack nobody watched;
    #   * every tilt any sweep visited is there, and no tilt nobody visited is.
    CAPTURE = os.path.join(
        HERE, "..", "..", "..", "scandata", "results14", "scandata", "capture",
        "cable_limits_ex-ur10e-1_hardware.json")
    hw = dict(block.get("hardware") or {})
    hw.setdefault("home_quat", block.get("home_quat"))
    hw.setdefault("cell", "ex-ur10e-1")
    hw.setdefault("mode", "hardware")
    from_manifest = ns["_cable_envelope_from_doc"](hw, "the manifest")

    def _by_tilt(samples):
        """{(tx, ty): (lo_deg, hi_deg)} straight off the recorded rows.

        The rows themselves, not band() -- the question here is what the merge
        did with the numbers, and interpolation is checked above.
        """
        return {(round(s[0], 6), round(s[1], 6)):
                (math.degrees(s[2]), math.degrees(s[3])) for s in samples}

    #: Every sweep of this cell's cables that is on record, newest first. The
    #: earlier one is MEASURED above; the later one is the file the capture tool
    #: wrote, read here rather than transcribed.
    RECORDED = []
    taken = None
    if os.path.exists(os.path.normpath(CAPTURE)):
        import json as _json
        taken = ns["_cable_envelope_from_doc"](
            _json.load(open(os.path.normpath(CAPTURE))), "the capture file")
        RECORDED.append((taken.measured, _by_tilt(taken.samples)))
    else:
        print("skip  one recorded sweep (%s is not here)"
              % (os.path.normpath(CAPTURE),))
    RECORDED.append(("2026-09-14T10:22:07", _by_tilt(env.samples)))

    man = _by_tilt(from_manifest.samples)
    check(len(RECORDED) > 1,
          "more than one sweep is on record, so the merge has something to do "
          "(%d)" % (len(RECORDED),))
    # The block has to SAY it is a merge and which sweeps went in, or nobody
    # reading it can tell a merged band from a mis-transcribed one.
    named = hw.get("merged_from") or []
    check(set(n for n, _r in RECORDED) <= set(named),
          "the manifest names every sweep it merges (%s)" % (named,))

    every_tilt = set()
    for _name, rows in RECORDED:
        every_tilt |= set(rows)
    check(every_tilt <= set(man),
          "every tilt any sweep visited reached the manifest (%d tilts)"
          % (len(every_tilt),))
    check(set(man) <= every_tilt,
          "the manifest declares no tilt no sweep ever visited")
    for tilt in sorted(man):
        lo, hi = man[tilt]
        los = [rows[tilt][0] for _n, rows in RECORDED if tilt in rows]
        his = [rows[tilt][1] for _n, rows in RECORDED if tilt in rows]
        if not los:
            continue
        check(lo <= min(los) + 1e-3 and hi >= max(his) - 1e-3,
              "tilt (x %+.0f, y %+.0f): %.4f..%.4f is at least as permissive as "
              "every sweep there (%s)"
              % (tilt[0], tilt[1], lo, hi,
                 ", ".join("%.4f..%.4f" % (a, b) for a, b in zip(los, his))))
        check(abs(lo - min(los)) < 1e-3 and abs(hi - max(his)) < 1e-3,
              "tilt (x %+.0f, y %+.0f): and no wider than the widest of them "
              "(%.4f..%.4f)" % (tilt[0], tilt[1], min(los), max(his)))
        # NOT RECONCILED BY A TURN. Sweeps of one tilt disagree by a step or so.
        # A merged band wider than a full turn, or two recorded ends a turn
        # apart, would mean two branches of the same orientation were treated as
        # one continuous band -- which is the substitution this module exists to
        # refuse, arriving through the merge instead of through a solver.
        check(hi - lo <= 360.0 + 1e-6,
              "tilt (x %+.0f, y %+.0f): the merged band is %.1f deg wide, within "
              "one turn" % (tilt[0], tilt[1], hi - lo))
        check(max(los) - min(los) < 180.0 and max(his) - min(his) < 180.0,
              "tilt (x %+.0f, y %+.0f): the sweeps disagree by %.1f/%.1f deg, "
              "not by a turn" % (tilt[0], tilt[1],
                                 max(los) - min(los), max(his) - min(his)))
    single = [t for t in man if sum(1 for _n, r in RECORDED if t in r) == 1]
    check(bool(single),
          "at least one tilt was visited by a single sweep and survived the "
          "merge with that sweep's band (%d of %d)" % (len(single), len(man)))

    if taken is not None:
        # THE DATUM. The band says how far the wrist may turn; the datum says
        # where zero probe rotation sits inside it, and a manifest carrying one
        # without the other cannot convert a requested rotation into a wrist
        # angle. It is NOT merged -- it is a measurement of the joint state a
        # sweep started from, not a bound -- so every row must carry one of the
        # recorded datums for its own tilt, unchanged.
        datums = [(tx, ty) for tx, ty, _lo, _hi, _l in taken.samples]
        carried = all(from_manifest.wrist3_datum_deg(tilt=t) is not None
                      for t in datums)
        check(carried, "every row's wrist3_start_deg reached the manifest")
        if carried:
            close = all(abs(from_manifest.wrist3_datum_deg(tilt=t)
                            - taken.wrist3_datum_deg(tilt=t)) < 1e-3
                        for t in datums)
            check(close, "...unchanged from the sweep that recorded it, to "
                         "0.001 deg")
            inside = all(
                from_manifest.band_deg(tilt=t)[0] - 1e-6
                <= from_manifest.wrist3_datum_deg(tilt=t)
                <= from_manifest.band_deg(tilt=t)[1] + 1e-6 for t in datums)
            check(inside, "...and still sits inside the merged band at its own "
                          "tilt, so a rotation is graded against a pair that "
                          "belong together")

    lo0, hi0 = from_manifest.band_deg(tilt=(0.0, 0.0))
    check(hi0 > lo0 and (hi0 - lo0) > 180.0,
          "an untilted pose read from the manifest gets %.4f..%.4f" % (lo0, hi0))

    # THE SIX ANGLES AN UNGUARDED SCAN COMMANDED, graded against the band the
    # manifest NOW carries rather than against a remembered verdict.
    #
    # Two of them moved. The two sweeps put the untilted low end at 31.34 and at
    # 16.09 -- one 15 degree step apart, on a limit a person judges by watching
    # the loom go taut -- and the merge keeps the lower. +21.9 and +29.3 sit
    # between the two, so the merged band admits them and the earlier sweep alone
    # did not. That is a real consequence of taking the most permissive recorded
    # end, not a weakened test: what must not change is that an angle outside
    # everything recorded is still refused, and that nothing is admitted by a
    # +/-360 substitution.
    refused_now = [d for d in FAILURE_ANGLES_DEG if not (lo0 <= d <= hi0)]
    inside_now = [d for d in FAILURE_ANGLES_DEG if lo0 <= d <= hi0]
    check(len(refused_now) >= 3,
          "at least the three angles below the band are still refused (%s)"
          % (sorted(set(refused_now)),))
    for deg in refused_now:
        raises(lambda d=deg: check_wrist3([joints(d)], "run_automated_scan",
                                          model=from_manifest),
               "manifest: wrist 3 %+7.1f deg is refused" % (deg,))
    for deg in inside_now:
        # Inside the measured band, so it must NOT raise -- and it must be inside
        # because the band says so, not because a wrap put it there.
        ok = True
        try:
            check_wrist3([joints(deg)], "run_automated_scan", model=from_manifest)
        except Exception:                                         # noqa: BLE001
            ok = False
        check(ok, "manifest: wrist 3 %+7.1f deg is inside the merged band "
                  "%.2f..%.2f and is allowed" % (deg, lo0, hi0))
        check(not (lo0 <= deg + 360.0 <= hi0) or (lo0 <= deg <= hi0),
              "...on its own absolute angle, not a full turn away")

# ==============================================================================
# THE GUARD ITSELF
# ==============================================================================
# A correct model stops nothing unless something ASKS it. So this drives the real
# _CableGuardedRos from cell_capture_motion.dpi over a fake motion module and
# checks that the six commanded angles never reach it.
print("")
# THIS SUITE IS THE ONE THAT ACTUALLY NEEDS A POSE WITH FIELDS. _cap_pose() in
# cell_capture_motion.dpi does `from geometry_msgs.msg import Pose, Quaternion`
# INSIDE the function, so it resolves against sys.modules when the guard is
# exercised, not when the .dpi is loaded -- no namespace injection can reach it,
# the entry itself has to be right at that moment. A field-less Pose there makes
# _cap_pose() raise, _ik_along() swallows the exception, the guard is handed no
# joints, and every case below passes while measuring nothing. ros_msg_stubs.py
# is what keeps that from happening.
#
# ASSIGNING sys.modules here at import time is not enough: any other suite in
# this directory that installs its own stubs later overwrites it, so whether it
# worked would depend on collection order. Instead the fielded classes go in for
# the duration of the guard exercise and are handed back at the end of it, which
# leaves sys.modules exactly as it was for every other suite.
_MSG_TOKEN = ros_msg_stubs.override_msgs()
mod = _MSG_TOKEN[1]

gns = dict(ns)
gns.update(CELL_NAME="ex-ur10e-1", CELL_MODE="hardware", CAPTURE_ROOT="/nonexistent",
           _capture_say=lambda m: None, capture_node=None,
           _cable_limits_file=lambda: "/nonexistent/none.json")
exec(compile(_load("cell_capture_motion.dpi"), "cell_capture_motion.dpi", "exec"), gns)
gns["_CAP"] = {}
gns["_CABLE"][0] = env                 # the measurement, without touching the disk
gns["_CABLE_LOOKED"][0] = True


class FakeRos(object):
    """Stands in for dgpython_rosmotion's motion module. Records what it was
    ASKED to do -- the whole question is whether the arm gets commanded."""

    def __init__(self):
        self.commanded = []
        self.ik_answer = joints(166.0)

    def SolveIK(self, pose, seed=None, timeout=2.0, **kw):
        return list(self.ik_answer)

    def CartesianMove(self, *a, **k):
        self.commanded.append("CartesianMove")
        return "moved"

    def JointMove(self, *a, **k):
        self.commanded.append("JointMove")
        return "moved"

    def MoveJoints(self, *a, **k):
        self.commanded.append("MoveJoints")
        return "moved"

    def FreeMove(self, *a, **k):
        self.commanded.append("FreeMove")
        return "moved"

    def ApproachInConfiguration(self, *a, **k):
        self.commanded.append("ApproachInConfiguration")
        return [0.0]*6


fake = FakeRos()
gns["ros"] = fake
gns["_CAP"]["real_ros"] = fake
guard = gns["install_cable_guard"](quiet=True)
check(gns["ros"] is guard, "install_cable_guard() puts itself in front of `ros`")
check(gns["install_cable_guard"](quiet=True) is guard, "installing twice is a no-op")


def commanded(fn):
    fake.commanded = []
    fn()
    return list(fake.commanded)


# THE PROOF. Every one of the six, through the same call the pipeline makes.
for deg in FAILURE_ANGLES_DEG:
    got = commanded(lambda d=deg: gns["ros"].MoveJoints(joints(d)))
    check(not got, "guard: MoveJoints(wrist 3 %+7.1f deg) never reaches the arm"
          % (deg,))
check(commanded(lambda: gns["ros"].MoveJoints(joints(200.0))) == ["MoveJoints"],
      "guard: a legal +200 deg move goes straight through")
check(not commanded(lambda: gns["ros"].JointMove(joints(-113.0))),
      "guard: JointMove is gated too")
check(commanded(lambda: gns["ros"].JointMove(joints(166.0))) == ["JointMove"],
      "guard: and passes a legal one")

pose = (np.zeros(3), (0.0, 0.0, 0.0, 1.0))
fake.ik_answer = joints(-113.0)
res = None


def _cart():
    global res
    res = gns["ros"].CartesianMove([pose, pose])


check(not commanded(_cart), "guard: a Cartesian row whose IK lands at -113 deg is refused")
check(res is not None and res.result.success is False and res.result.fraction == 0.0,
      "guard: the refusal comes back as a FAILED MOVE, not an exception "
      "(a raise would abandon a running TSDF mid-sweep)")
check(not commanded(lambda: gns["ros"].ApproachInConfiguration(pose, pose)),
      "guard: entering a pass is checked before the arm is placed")
check(gns["ros"].ApproachInConfiguration(pose, pose) is None,
      "guard: and answers None, which is its documented 'IK failed'")
fake.ik_answer = joints(166.0)
check(commanded(_cart) == ["CartesianMove"], "guard: a legal row still flies")

with gns["cable_guard_off"]("measuring the band"):
    out = commanded(lambda: gns["ros"].MoveJoints(joints(-113.0)))
check(out == ["MoveJoints"],
      "cable_guard_off(): capture_cable_envelope() can leave the band to measure it")
check(not commanded(lambda: gns["ros"].MoveJoints(joints(-113.0))),
      "and the guard is back on afterwards")

with gns["gated_pipeline"]():
    inner = type(gns["ros"]).__name__
check(inner == "_GatedRos" and gns["ros"] is guard,
      "a capture scenario swaps the gate in and puts the CABLE GUARD back, not "
      "the raw motion module")

check(gns["remove_cable_guard"]() is fake, "remove_cable_guard() unwraps cleanly")


# ------------------------------------------------------------------------------
# THE GUARD SECTION WAS NOT BLIND
# ------------------------------------------------------------------------------
# Everything above between override_msgs() and here went through _cap_pose(), and
# a Pose with no fields would have made all of it pass while measuring nothing.
# So say it as a check, with a KNOWN-POSITIVE and a KNOWN-NEGATIVE, before the
# stub goes back: the pose really is built, and the guard really does distinguish
# a legal wrist_3 from the one that stopped the work cell.
_probe_pose = gns["_cap_pose"]((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
check(abs(_probe_pose.position.x - 0.1) < 1e-12
      and abs(_probe_pose.position.z - 0.3) < 1e-12
      and abs(_probe_pose.orientation.w - 1.0) < 1e-12,
      "_cap_pose() really built a Pose with fields -- a field-less one makes it "
      "raise, _ik_along() swallow it, and every check above pass blind")
_reinstalled = gns["install_cable_guard"](quiet=True)
fake.ik_answer = joints(-113.0)
_blind_bad = commanded(lambda: gns["ros"].CartesianMove([pose, pose]))
fake.ik_answer = joints(166.0)
_blind_good = commanded(lambda: gns["ros"].CartesianMove([pose, pose]))
check(not _blind_bad and _blind_good == ["CartesianMove"],
      "and the guard discriminates through that pose: IK at -113 deg is stopped "
      "(%r) and at +166 deg flies (%r)" % (_blind_bad, _blind_good))
gns["remove_cable_guard"]()

# The stub goes back exactly as it was found, so no other suite in this directory
# inherits ours. ros_stub_hygiene_test.py is the lint that keeps it that way.
ros_msg_stubs.restore_msgs(_MSG_TOKEN)

# ------------------------------------------------------------------------------
# An explicitly loaded file must SURVIVE the next query
# ------------------------------------------------------------------------------
# If load_cable_envelope(path) sets the model but leaves _CABLE_LOOKED False, the
# next cable_envelope_model() -- which every wrist3_band() goes through --
# reloads from the DEFAULT locations and overwrites it. On the production system
# that is the one command an operator reaches for after re-measuring, so it silently
# doing nothing is the worst available failure.
import json as _json
import tempfile as _tempfile

_doc = ns["cable_envelope_document"](MEASURED, step_deg=15.0)
with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as _h:
    _json.dump(_doc, _h)
    _explicit = _h.name

ns["_CABLE"][0] = None
ns["_CABLE_LOOKED"][0] = False
_loaded = ns["load_cable_envelope"](_explicit)
check(_loaded is not None, "an explicit file loads")
check(ns["cable_envelope_model"]() is _loaded,
      "and is still the model on the next query (not reloaded from the defaults)")
_lo, _hi = ns["wrist3_band_deg"](tilt=(0.0, 0.0))
check(abs(_lo - 31.338) < 0.01 and abs(_hi - 301.338) < 0.01,
      "wrist3_band() then answers from the explicitly loaded file: %.3f..%.3f"
      % (_lo, _hi))
os.unlink(_explicit)



# ==============================================================================
# THE MOTION VOLUME MUST CONTAIN THE WORK VOLUME
# ==============================================================================
# Two volumes, and they are not the same box. `work_volume` is where a SAMPLE may
# be placed and where the eddy-current probe TIP must stay; `motion_volume` is
# where the arm and its tooling may navigate. The tooling stands ~192 mm above
# the tip and reaches ~75 mm to the side, so requiring all of it inside the part
# box puts probe_end on the work volume's wall while the laser is correctly over
# the part, and the arm then refuses the first pose of a sweep.
#
# A motion volume that does not CONTAIN the work volume is the failure that would
# not announce itself: every guarded move would be judged against a box smaller
# than the one a part may legally occupy, so a legal part would be unreachable.

if data is not None:
    _mv = data.get("motion_volume") or {}
    _wv = data.get("work_volume") or {}
    for _mode in ("sim", "hardware"):
        _w, _m = _wv.get(_mode), _mv.get(_mode)
        check(isinstance(_m, dict) and _m.get("min") and _m.get("max"),
              "the manifest declares a motion volume for %s" % (_mode,))
        if not (isinstance(_w, dict) and isinstance(_m, dict)):
            continue
        _contains = (all(float(a) <= float(b) + 1e-9
                         for a, b in zip(_m["min"], _w["min"]))
                     and all(float(a) >= float(b) - 1e-9
                             for a, b in zip(_m["max"], _w["max"])))
        check(_contains,
              "%s: the motion volume CONTAINS the work volume (move %s..%s vs "
              "work %s..%s)" % (_mode, _m["min"], _m["max"], _w["min"], _w["max"]))
        # And it clears it in x-y by at least the tooling's own reach, which is
        # the quantity that made two boxes necessary in the first place.
        _pad = min([float(_w["min"][i]) - float(_m["min"][i]) for i in (0, 1)]
                   + [float(_m["max"][i]) - float(_w["max"][i]) for i in (0, 1)])
        check(_pad >= 0.075,
              "%s: it clears the work volume by %.0f mm in x-y, at least the "
              "tooling's own 75 mm reach" % (_mode, _pad*1e3))
        # The FLOOR is deliberately shared: nothing below the surface parts rest
        # on is ever wanted, for the tooling any more than for the tip.
        check(abs(float(_m["min"][2]) - float(_w["min"][2])) < 1e-9,
              "%s: both volumes share a floor -- the tooling has no business "
              "below the surface a part rests on" % (_mode,))


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================

def test_not_blind():
    """The model and the guard both ran, and both tell two cases apart."""
    assert check_wrist3.__name__ == "check_wrist3" and callable(wrist3_tilt_of)
    # KNOWN-POSITIVE / KNOWN-NEGATIVE on the model: the measured untilted band is
    # 31.3..301.3, so 166 is inside it and -113 is not.
    check_wrist3([joints(166.0)], "not_blind", model=env)
    try:
        check_wrist3([joints(-113.0)], "not_blind", model=env)
    except CableRefused:
        pass
    else:
        raise AssertionError("-113 deg was accepted: the cable model is not "
                             "being consulted")
    # And the guard section above really went through a Pose with fields rather
    # than passing on a swallowed AttributeError. The _cap_pose call is recorded
    # in FAILURES if it did not; assert the object itself here too.
    assert hasattr(_probe_pose, "position") and hasattr(_probe_pose, "orientation")
    assert abs(_probe_pose.position.y - 0.2) < 1e-12, \
        "_cap_pose() did not fill in the pose it built"
    assert not _blind_bad and _blind_good == ["CartesianMove"], \
        "the guard did not discriminate: -113 %r, +166 %r" \
        % (_blind_bad, _blind_good)
    # sys.modules was handed back: this suite leaves nothing behind for the ten
    # others in this directory to inherit.
    assert ros_msg_stubs.msg_module() is not mod, \
        "this suite left its own geometry_msgs.msg installed"


def test_cable_envelope():
    """So `pytest cable_envelope_test.py` reports the same thing as running it."""
    assert not FAILURES, "%d check(s) failed: %s" % (len(FAILURES),
                                                     "; ".join(FAILURES))


try:
    test_not_blind()
    check(True, "the suite is not blind")
except AssertionError as _e:
    check(False, "the suite is not blind (%s)" % (_e,))

if __name__ == "__main__":
    print("")
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("all passed")
