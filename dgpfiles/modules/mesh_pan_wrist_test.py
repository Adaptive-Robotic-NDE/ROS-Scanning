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
# File:  mesh_pan_wrist_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the pan sweep's roll search, judged on what it costs wrist 3.
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
"""The pan sweep's roll search, judged on what it costs wrist_3.

THE FAILURE THIS GUARDS AGAINST. A roll search that scores candidates on tooling
height alone, sweeps the roll over the FULL circle independently per direction,
keeps no memory of the direction flown before it and has no notion of a cable
band will happily pick rolls whose wrist_3 sits in the dead zone around 0 --
outside a measured band of 31..301 degrees -- and will walk the wrist straight
through that zone between directions. Rolls 180 degrees apart are equally good
by the height score, so nothing in that search prefers the one the cables can
take. The search here is cable-aware, continuous and anchored to the wrist
position it starts from, and this file measures the difference.

Everything here is arithmetic and runs on the host with numpy. The arm is
stubbed by a synthetic wrist model that is EXACT in the one respect that
matters: on this tooling the camera's optical axis is antiparallel to probe_end
z, so a roll about the optical axis is exactly a rotation about the tool axis,
and wrist_3 moves with it degree for degree. Between two different azimuths the
rest of the chain moves as well, so the stub adds a shoulder-pan term the search
cannot predict -- which is the point: the prediction ranks the candidates and
the IK check settles them.

The tooling cloud is synthetic too. It is asymmetric about the tool axis, which
is all the roll search needs to have an opinion; the real one comes from
tooling_clearance.dpi's envelope.
"""
import functools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# One cell's touched-off work volume (ex-ur10e-1).
VOLUME = ("world", (0.5596, -0.5093, 0.9248), (1.1450, 0.4539, 1.2448))
SUPPORT = VOLUME[1][2]
CEILING = VOLUME[2][2]
MOTION_CEILING = CEILING + 0.25
REGION = (0.80, -0.25, 0.95, -0.10)

# probe_end -> the depth camera's OPTICAL frame, off the rendered
# ur_cell.urdf.xacro for tooling `ec` with the `redo` Azure bracket.
CAM_T = (0.0765, -0.0005, 0.0753)
CAM_R = np.array([[0.0, -1.0, 0.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0]])

# A deliberately lopsided tooling cloud in probe_end coordinates: the Azure
# bracket out along +x, the body of the tooling above, the eddy-current probe at
# the origin. Points, not a mesh -- the search only ever asks for extremes.
TOOLING = np.array([[0.00, 0.00, 0.000],
                    [0.12, 0.00, 0.075], [0.12, 0.05, 0.075],
                    [0.12, -0.05, 0.075], [0.00, 0.00, 0.192],
                    [0.06, 0.00, 0.192], [-0.06, 0.00, 0.100],
                    [0.00, 0.08, 0.120], [0.00, -0.08, 0.120]])

BAND = (math.radians(31.0), math.radians(301.0))
BAND_CENTRE = math.radians(166.0)
HOME_QUAT = (0.0, 0.0, 0.0, 1.0)


def _load():
    """mesh_reconstruction.dpi's geometry, with ScanPath and the quat helpers.

    The same loader mesh_pan_sweep_test.py uses; kept here rather than imported
    from it because importing that file runs its whole suite.
    """
    ns = {"work_volume": lambda *a, **k: VOLUME,
          "require_work_volume": lambda *a, **k: VOLUME}

    def _exec(name, drop=()):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                        for l in src.splitlines())
        for d in drop:
            src = src.replace(d, "pass  # " + d.replace("\n", " "))
        exec(compile(src, name, "exec"), ns)

    _exec("toolpath_segments.dpi",
          drop=("from geometry_msgs.msg import PoseArray, Pose",))
    src = open(os.path.join(HERE, "noether_path.dpi")).read()
    lines = src.splitlines()
    keep, taking = [], False
    for l in lines:
        if l.startswith("def quat_mul(") or l.startswith("def quat_to_matrix(") \
                or l.startswith("def matrix_to_quat("):
            taking = True
        elif taking and l and not l[0].isspace() and not l.startswith(")"):
            taking = False
        if taking:
            keep.append(l)
    exec(compile("import numpy as np\n" + "\n".join(keep), "quats", "exec"), ns)
    _exec("mesh_reconstruction.dpi")
    return ns


fails = []


def check(name, ok, detail=""):
    print("  %-66s %s" % (name, "ok" if ok else "FAIL  %s" % (detail,)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


ns = _load()
cam_tf = (CAM_T, ns["matrix_to_quat"](CAM_R))
twist_between = ns["_twist_between"]
in_band = ns["_in_wrist_band"]
CLEARANCE = ns["AZURE_PAN_TOOLING_CLEARANCE"]


# --- the synthetic arm --------------------------------------------------------
def synth_wrist(position, quat):
    """wrist_3 in radians for a probe_end pose, as this tooling's arm would.

    Exact in the roll -- a rotation about the tool axis IS wrist_3 here -- plus
    a shoulder-pan term so that two different azimuths do not share an offset.
    """
    twist = math.radians(twist_between(HOME_QUAT, quat))
    pan = math.atan2(float(position[1]), float(position[0]))
    return BAND_CENTRE + twist + 0.15*pan


def reach(position, quat):
    """A stub IK: every pose is reachable, and it answers with the solution."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, synth_wrist(position, quat)]


def wrists_of(path):
    return np.array([synth_wrist(p, q)
                     for p, q in zip(path.positions, path.quats)])


def travel_deg(values):
    """Total wrist_3 the arm turns through over a sequence, degrees."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.degrees(np.abs(np.diff(v)).sum())) if len(v) > 1 else 0.0


@_checked
def test_a_roll_about_the_optical_axis_is_wrist_3():
    print("=== a roll about the optical axis IS wrist_3, degree for degree ===")
    off = np.array([0.35, 0.0, 0.40])
    q0 = ns["_probe_end_from_camera"](ns["_optical_rotation"](off, 0.0),
                                      np.array([0.85, -0.2, SUPPORT + 0.40]),
                                      cam_tf)[1]
    worst = 0.0
    for r in np.linspace(0.0, 2*np.pi, 24, endpoint=False):
        qr = ns["_probe_end_from_camera"](ns["_optical_rotation"](off, float(r)),
                                          np.array([0.85, -0.2, SUPPORT + 0.40]),
                                          cam_tf)[1]
        want = math.degrees((-r + math.pi) % (2*math.pi) - math.pi)
        worst = max(worst, abs((twist_between(q0, qr) - want + 180.0) % 360.0
                               - 180.0))
    check("_twist_between reads a roll of r as exactly -r of tool twist",
          worst < 1e-9, "worst %.3g deg" % worst)


@_checked
def test_the_band_arithmetic():
    print("=== the band arithmetic: a dead-zone sweep reads as out of band ===")
    # Five wrist_3 values a height-only roll search produces, all in the 90 deg
    # dead zone around 0 that a 31..301 deg band excludes.
    stopped = (29.3, 21.9, -32.3, -55.1, -113.0)
    outside = [d for d in stopped if not in_band(math.radians(d), BAND)]
    check("every one of the five commanded wrist_3 values is outside 31..301 deg",
          outside == list(stopped), "%s" % (outside,))
    # The band is an ABSOLUTE joint interval, not an arc: -113 and +247 are the
    # same tool orientation but are NOT the same place for a cable, because one
    # has a turn wound into it that the other has not. cable_envelope.dpi
    # applies the same rule, and this file must agree with it or a plan would be
    # checked against one meaning and flown against the other.
    check("+247 deg is inside the band", in_band(math.radians(247.0), BAND))
    check("and -113 deg, the SAME orientation one turn away, is not",
          not in_band(math.radians(-113.0), BAND))
    check("no band is no constraint", in_band(math.radians(0.0), None))
    check("the band centre is 166 deg",
          abs(math.degrees(ns["_wrist_band_centre"](BAND)) - 166.0) < 1e-9)


@_checked
def test_a_reachability_callable_may_answer_either_way():
    print("=== a reachability callable may answer with a bool or a solution ===")
    check("a bool still means reachable", ns["_reach_result"](True) == (True, None))
    check("and False still means refused", ns["_reach_result"](False) == (False, None))
    check("and None means refused", ns["_reach_result"](None) == (False, None))
    ok, w3 = ns["_reach_result"]([0.1, 0.2, 0.3, 0.4, 0.5, 2.9])
    check("a joint solution yields wrist_3", ok and abs(w3 - 2.9) < 1e-12,
          "%s %s" % (ok, w3))
    ok, w3 = ns["_reach_result"](np.bool_(True))
    check("a numpy bool is not mistaken for a solution", ok and w3 is None,
          "%s %s" % (ok, w3))


@_checked
def test_with_no_cable_envelope_loaded_it_says_so():
    print("=== with no cable_envelope.dpi loaded, it says so and carries on ===")
    ns["_cable_envelope_warned"] = False
    said = []
    ns["print"] = lambda *v: said.append(" ".join(str(x) for x in v))
    env = ns["_cable_envelope"]()
    del ns["print"]
    check("no envelope means no band", env is None)
    check("and it says which check is missing, once",
          any("cable_envelope.dpi is NOT loaded" in s for s in said), "%s" % (said,))
    check("_pan_band with no envelope has neither a band nor a refusal",
          ns["_pan_band"](None) == (None, None), "%s" % (ns["_pan_band"](None),))
# --- what the search did before, reproduced -----------------------------------
def old_plan(height, lateral, want=4, nroll=24, tries=3):
    """A height-only selection: azimuths in LIST ORDER, rolls scored on tooling
    height alone, no memory of the previous direction and no cable band.

    The baseline the cable-aware search is measured against. It is built from
    the live helpers rather than copied: _pan_roll_order() with no anchor sorts
    exactly this way, so the baseline cannot drift away from the real code.
    """
    kept = []
    for az in ns["AZURE_PAN_AZIMUTHS"]:
        aims, _u, starts = ns["_pan_aims"](REGION, SUPPORT, az,
                                           ns["AZURE_PAN_STOPS"])
        offs = ns["_pan_offset"](REGION, az, height, lateral, aims=aims)
        cands = ns["_pan_roll_order"](offs, SUPPORT, cam_tf, TOOLING, nroll,
                                      MOTION_CEILING - 0.005, CLEARANCE)
        cams = ns["_pan_stations"](aims, offs)
        for cand in cands[:tries]:
            R = ns["_optical_rotation"](offs, cand["roll"])
            poses = [ns["_probe_end_from_camera"](R, c, cam_tf) for c in cams]
            if all(reach(p, q) is not None for p, q in poses):
                kept.append((float(az), cand["roll"],
                             [synth_wrist(p, q) for p, q in poses]))
                break
        if len(kept) >= want:
            break
    return kept




@_checked
def test_the_cable_aware_roll_search():
    print("=== BEFORE: list order, height-only rolls, no band ===")
    # THE LATERAL LADDER, named rather than derived. The roll search only has an
    # interesting job at a rung that stands well off to the side -- 500 mm up,
    # 176 mm out, 29 degrees of tilt. The ladder's top rung is now the vertical
    # one (see AZURE_PAN_VERTICAL), where the tilt is 8-13 degrees and the band
    # is the wide untilted one, so deriving the rung would compare the two
    # searches on a geometry where neither has to work for it.
    rungs = ns["pan_height_for"](0.53, SUPPORT, MOTION_CEILING, cam_tf, REGION,
                                 tooling_points=TOOLING,
                                 heights=ns["AZURE_PAN_HEIGHTS"])
    height, lateral, _d = rungs[0]
    print("    at the top lateral rung: %.0f mm up, %.0f mm out"
          % (1e3*height, 1e3*lateral))
    old = old_plan(height, lateral)
    old_w = [w for _a, _r, ws in old for w in ws]
    old_bad = [w for w in old_w if not in_band(w, BAND)]
    old_travel = travel_deg(old_w)
    print("    directions %s, rolls %s"
          % ([int(a) for a, _r, _w in old],
             ["%+.0f" % math.degrees(r) for _a, r, _w in old]))
    print("    wrist_3 %s"
          % ", ".join("%+.0f" % math.degrees(ws[0]) for _a, _r, ws in old))
    print("    %d of %d poses outside the band, wrist_3 %.0f..%.0f deg, %.0f deg "
          "of travel between directions"
          % (len(old_bad), len(old_w), math.degrees(min(old_w)),
             math.degrees(max(old_w)), travel_deg([w[0] for _a, _r, w in old])))
    check("the old search really does leave poses outside the cable band",
          len(old_bad) > 0, "all %d poses were in band" % len(old_w))
    check("and it takes the azimuths in list order, a %.0f deg arc"
          % (max(a for a, _r, _w in old) - min(a for a, _r, _w in old)),
          [a for a, _r, _w in old] == sorted(a for a, _r, _w in old))

    print("=== AFTER: cable-aware, continuous, ordered by what the wrist pays ===")
    calls = []


    def fake_band(tilt=None, quat=None):
        return BAND


    def fake_check(qs, what, quats=None):
        calls.append(what)
        bad = [v for v in np.asarray(qs).reshape(-1) if not in_band(v, BAND)]
        if bad:
            raise ValueError("%s: wrist_3 %s leaves the cable band"
                             % (what, ["%.0f" % math.degrees(v) for v in bad]))


    ns["wrist3_band"] = fake_band
    ns["check_wrist3"] = fake_check
    new = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                               ceiling=MOTION_CEILING, tooling_points=TOOLING,
                               reachable=reach, start_quat=HOME_QUAT,
                               start_wrist=BAND_CENTRE,
                               heights=ns["AZURE_PAN_HEIGHTS"], quiet=True)
    new_w = wrists_of(new)
    check("the cable check was consulted, once per direction",
          len(calls) >= len(new.pan_directions), "%d calls" % len(calls))
    print("    %d of %d poses outside the band, wrist_3 %.0f..%.0f deg"
          % (sum(1 for w in new_w if not in_band(w, BAND)), len(new_w),
             math.degrees(new_w.min()), math.degrees(new_w.max())))
    check("EVERY pose of the plan is inside the cable band (%d poses)" % len(new_w),
          all(in_band(w, BAND) for w in new_w),
          "%s" % (["%.0f" % math.degrees(w) for w in new_w
                   if not in_band(w, BAND)],))
    check("the plan records the wrist_3 it will use, measured not predicted",
          new.pan_wrist_measured and len(new.pan_wrist_deg) == len(new.pan_directions),
          "%s" % (new.pan_wrist_deg,))
    check("and it still keeps four directions", len(new.pan_directions) == 4,
          "%d" % len(new.pan_directions))
    azs = [a for a, _r, _z, _t in new.pan_directions]
    gaps = [min(abs((a - b + 180.0) % 360.0 - 180.0) for b in azs if b != a)
            for a in azs]
    check("which are SPREAD -- every one at least 90 deg from its nearest "
          "neighbour", min(gaps) >= 90.0 - 1e-9, "azimuths %s, gaps %s" % (azs, gaps))

    # The number the cables actually pay: wrist_3 travel from one direction to the
    # next, since a direction holds one orientation for all of its poses.
    old_between = travel_deg([ws[0] for _a, _r, ws in old])
    new_between = travel_deg([math.radians(w) for w in new.pan_wrist_deg])
    print("    wrist_3 between directions: %.0f deg before, %.0f deg after"
          % (old_between, new_between))
    check("wrist_3 travel between directions is cut (%.0f -> %.0f deg)"
          % (old_between, new_between), new_between < old_between,
          "%.0f vs %.0f" % (new_between, old_between))
    check("and no single direction change is more than half a turn",
          max([abs(d) for d in np.diff([w for w in new.pan_wrist_deg])] or [0.0])
          <= 180.0 + 1e-9, "%s" % (new.pan_wrist_deg,))

    print("=== a direction that cannot be flown in band is DROPPED, with a reason ===")
    # A roll is free over the whole circle, so for any band of non-zero width
    # there is almost always SOME roll that satisfies it, and the band check
    # usually costs nothing. What genuinely cannot be flown is a direction where
    # the arm has only one configuration available and that one lies in the dead
    # zone. The stub below makes the +y half of the cell behave that way.
    ns["wrist3_band"] = fake_band


    def reach_pinned(position, quat):
        q = list(reach(position, quat))
        if float(position[1]) > -0.02:
            q[5] = math.radians(10.0)          # the dead zone, and no alternative
        return q


    pinned = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                                  ceiling=MOTION_CEILING, tooling_points=TOOLING,
                                  reachable=reach_pinned, start_quat=HOME_QUAT,
                                  start_wrist=BAND_CENTRE, min_directions=1,
                                  heights=ns["AZURE_PAN_HEIGHTS"], quiet=True)
    print("    kept %s, dropped %s"
          % ([int(a) for a, _r, _z, _t in pinned.pan_directions],
             [(int(a), w) for a, w in pinned.pan_dropped_why]))
    check("the directions that can only be flown in the dead zone are dropped",
          len(pinned.pan_dropped_why) > 0, "nothing was dropped")
    check("and each one says WHY, naming the cable band",
          all("cable band" in w for _a, w in pinned.pan_dropped_why),
          "%s" % (pinned.pan_dropped_why,))
    check("pan_dropped is still a plain list of azimuths",
          all(isinstance(a, float) for a in pinned.pan_dropped),
          "%s" % (pinned.pan_dropped,))
    check("every pose that survived is inside the band",
          all(in_band(reach_pinned(p, q)[5], BAND)
              for p, q in zip(pinned.positions, pinned.quats)), "")

    # A band that excludes everything: the sweep is refused, not flown.
    ns["wrist3_band"] = lambda tilt=None, quat=None: (0.0, 1e-9)
    err = None
    try:
        ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                             ceiling=MOTION_CEILING, tooling_points=TOOLING,
                             reachable=reach, start_quat=HOME_QUAT,
                             start_wrist=BAND_CENTRE, quiet=True)
    except ValueError as e:
        err = e
    check("a band no roll can satisfy REFUSES the sweep rather than flying it",
          err is not None and "refused" in str(err), "%r" % (err,))

    print("=== without the envelope the plan is unchanged from before ===")
    del ns["wrist3_band"]
    del ns["check_wrist3"]
    ns["_cable_envelope_warned"] = True
    plain = ns["pan_sweep_path"](REGION, cam_tf, surface_z=SUPPORT, slant=0.53,
                                 ceiling=MOTION_CEILING, tooling_points=TOOLING,
                                 reachable=reach, quiet=True)
    check("it still plans, with no band and no anchor",
          len(plain.pan_directions) >= ns["AZURE_PAN_MIN_DIRECTIONS"],
          "%d" % len(plain.pan_directions))
    check("and it reports no band", plain.pan_wrist_band is None)
    # The rolls are still chosen on tooling height, which is what the height ladder
    # and every existing check depend on.
    check("the tooling still hangs at least 150 mm above the support",
          min(z - SUPPORT for _a, _r, z, _t in plain.pan_directions) > 0.15,
          "%.0f mm" % (1e3*min(z - SUPPORT
                               for _a, _r, z, _t in plain.pan_directions),))


@_checked
def test_a_refused_direction_says_which_constraint_refused_it():
    print("=== a refused direction says WHICH constraint refused it ===")
    # THE MESSAGE IS THE DELIVERABLE HERE. "No roll clears the tooling, the
    # ceiling and the cable band" reads as a hard physical limit and invites
    # rebuilding the tooling. A direction refused because the envelope has no
    # measurement at that tilt -- the tilt row swept to +/-20 while the sweep
    # wants 22-29 -- is a missing measurement and must say so, first.


    tally = {}
    ns["_pan_tally"](tally, "unmeasured", "tilt y +36.8 deg is outside")
    ns["_pan_tally"](tally, "unmeasured", "tilt y +28.2 deg is outside")
    ns["_pan_tally"](tally, "ceiling", 0.031)
    why = ns["_pan_why"](tally)
    check("an unmeasured tilt is named as such", "NO MEASUREMENT" in why, why)
    check("and it comes before the other causes",
          why.index("NO MEASUREMENT") < why.index("ceiling"), why)
    check("the count of rolls is carried", "2 of the rolls tried" in why, why)
    check("and the first detail survives", "+36.8" in why, why)
    check("a ceiling rejection is reported in mm",
          "31 mm" in why, why)
    check("an empty tally does not pretend to know",
          ns["_pan_why"]({}) == "no roll was admissible")

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILED: %s" % ", ".join(fails))
        sys.exit(1)
    print("all pan wrist checks passed")


_SECTIONS = ["test_a_roll_about_the_optical_axis_is_wrist_3", "test_the_band_arithmetic", "test_a_reachability_callable_may_answer_either_way", "test_with_no_cable_envelope_loaded_it_says_so", "test_the_cable_aware_roll_search", "test_a_refused_direction_says_which_constraint_refused_it"]


if __name__ == "__main__":
    sys.exit(_main())
