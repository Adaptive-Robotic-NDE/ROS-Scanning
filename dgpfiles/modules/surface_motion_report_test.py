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
# File:  surface_motion_report_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the dry-run motion report in surface_path_scan.dpi against
#   flights whose answers are known.
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
"""Check the dry-run arithmetic in surface_path_scan.dpi against flights whose
answers are known.

path_motion_report() is what says, after a dry run, whether the tip followed the
planned line and normal, how fast each kind of motion went, how close the tip
came to the part between passes and how far the wrist turned. It is the only
thing standing between "the sim run finished" and "the sim run did what the scan
will do", so its numbers are checked here against synthetic recordings: a sweep
offset by a known distance, a transit that dips by a known amount, a known speed,
a known roll. Runs on the host; no ROS, no recorder threads.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve -- but nothing here ever builds a message, so WHICH stub is in force is
# no business of this suite's. ensure_msgs() only adds; it does not overwrite what
# another suite in this directory put there, which is why nothing here assigns
# sys.modules directly. See ros_msg_stubs.py.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "surface_motion_report_test"}
for name in ("toolpath_segments.dpi", "noether_path.dpi", "surface_path_scan.dpi"):
    exec(compile(_load(name), name, "exec"), ns)

ScanPath = ns["ScanPath"]
report_fn = ns["path_motion_report"]
fmt = ns["format_motion_report"]
m2q = ns["matrix_to_quat"]
window_speed = ns["_window_speed"]
polyline_nearest = ns["_polyline_nearest"]

FAILS = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def frame(n, ref=(1.0, 0.0, 0.0)):
    n = np.asarray(n, dtype=np.float64)
    n = n/np.linalg.norm(n)
    x = np.asarray(ref, dtype=np.float64) - np.dot(ref, n)*n
    x = x/np.linalg.norm(x)
    return m2q(np.column_stack([x, np.cross(n, x), n]))


# Two passes along x over a plate tilted 30 degrees about x (normal in y-z).
tilt = np.radians(30.0)
n = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
q = frame(n)
xs = np.linspace(0.80, 0.90, 51)
across = np.cross(n, (1.0, 0.0, 0.0))        # in the tilted plane, across the passes
pos, quats, starts = [], [], []
for row in range(2):
    starts.append(len(pos))
    for x in xs:
        pos.append(np.array((x, -0.25, 0.96)) + 0.010*row*across)
        quats.append(q)
path = ScanPath(np.array(pos), np.array(quats), starts, source="test")
segs = path.segments

# --- recorded flight ----------------------------------------------------------
# pass 0: swept at exactly 10 mm/s, 0.4 mm OFF the surface along the normal,
#         the tool rolled 2 degrees about its own axis half way along.
# approach to pass 1: lift 10 mm along n at 10 mm/s, transit 0.1 m at 50 mm/s
#         whose middle dips to 3 mm above the surface, descend at 10 mm/s.
t, ph, sg, xyz, qq = [], [], [], [], []
T = 0.0
dt = 0.02


def add(phase, seg, p, qv):
    global T
    t.append(T)
    ph.append(phase)
    sg.append(seg)
    xyz.append(np.asarray(p, dtype=np.float64))
    qq.append(np.asarray(qv, dtype=np.float64))
    T += dt


def rolled(qv, deg):
    a = np.radians(deg)/2.0
    rz = (0.0, 0.0, np.sin(a), np.cos(a))
    return ns["quat_mul"](tuple(qv), rz)     # about the tool's own z


p0 = np.array(pos[0])
for i in range(int(0.10/0.010/dt) + 1):             # 10 s at 10 mm/s
    s = min(i*0.010*dt, 0.10)
    add("sweep", 0, p0 + np.array([s, 0.0, 0.0]) + 0.0004*n,
        rolled(q, 2.0 if s > 0.05 else 0.0))
end0 = np.array(pos[50]) + 0.0004*n
for i in range(51):                                   # lift 10 mm in 1 s
    add("lift", 1, end0 + n*0.010*i/50.0, q)
top = end0 + n*0.010
start1 = np.array(pos[51])
for i in range(101):                                  # transit, 2 s, 0.1 m
    u = i/100.0
    p = top + (start1 + n*0.010 - top)*u
    p = p - n*(0.0074*np.sin(np.pi*u))               # dips to 0.4+10-7.4 = 3 mm clear
    add("transit", 1, p, q)
for i in range(51):                                   # descend 10 mm in 1 s
    add("descend", 1, start1 + n*0.010*(1.0 - i/50.0), q)
for i in range(int(0.10/0.010/dt) + 1):
    s = min(i*0.010*dt, 0.10)
    add("sweep", 1, start1 + np.array([s, 0.0, 0.0]), q)

poses = {"t": np.array(t), "phase": ph, "seg": np.array(sg),
         "xyz": np.array(xyz), "quat": np.array(qq)}
# wrist_3 sweeps 0 -> 3 deg on pass 0 and swings 40 deg during the approach.
jt, jph, jsg, jq = [], [], [], []
n_sweep0 = sum(1 for p, s_ in zip(ph, sg) if p == "sweep" and s_ == 0)
for i, (tt, p, s_) in enumerate(zip(t, ph, sg)):
    w3 = (3.0*i/(n_sweep0 - 1) if p == "sweep" and s_ == 0
          else (40.0 if p == "transit" else 0.0))
    jt.append(tt)
    jph.append(p)
    jsg.append(s_)
    jq.append([0, 0, 0, 0, 0, np.radians(w3)])
joints = {"t": np.array(jt), "phase": jph, "seg": np.array(jsg), "q": np.array(jq)}
passes = [{"pass": 0, "approach_ok": True, "sweep_ok": True, "fraction": 1.0,
           "approach_s": 1.0, "sweep_s": 10.0},
          {"pass": 1, "approach_ok": True, "sweep_ok": True, "fraction": 1.0,
           "approach_s": 4.0, "sweep_s": 10.0}]

r = report_fn(path, segs, passes, poses, joints)
p0r, p1r = r["passes"]
s = r["summary"]

check(abs(p0r["normal_offset_mean_mm"] - 0.4) < 1e-6 and abs(p0r["path_error_max_mm"] - 0.4) < 1e-6,
      "a sweep 0.4 mm off the surface reads +0.40 mm along the normal (%.4f) and "
      "0.40 mm from the line" % (p0r["normal_offset_mean_mm"],))
check(abs(p1r["normal_offset_max_mm"]) < 1e-6,
      "a sweep on the line reads 0 (%.2e)" % (p1r["normal_offset_max_mm"],))
check(abs(p0r["observed_speed_mps"] - 0.010) < 1e-4,
      "observed sweep speed 10.0 mm/s (%.3f)" % (p0r["observed_speed_mps"]*1e3,))
check(abs(p0r["progress_m"] - 0.10) < 1e-6, "the sweep covered the whole 0.10 m pass")
check(abs(p0r["twist_range_deg"] - 2.0) < 1e-6 and p1r["twist_range_deg"] < 1e-6,
      "a 2 deg roll mid-pass reads 2.00 deg, an unrolled pass 0 (%.4f, %.1e)"
      % (p0r["twist_range_deg"], p1r["twist_range_deg"]))
check(abs(p0r["tilt_max_deg"] - 30.0) < 1e-6 and p1r["tilt_error_max_deg"] < 1e-6,
      "tilt 30 deg, following the planned normal exactly")
check(abs(p1r["lift_peak_speed_mps"] - 0.010) < 5e-4
      and abs(p1r["descend_peak_speed_mps"] - 0.010) < 5e-4,
      "lift and descent at 10 mm/s (%.2f, %.2f)"
      % (p1r["lift_peak_speed_mps"]*1e3, p1r["descend_peak_speed_mps"]*1e3))
check(0.050 < p1r["transit_peak_speed_mps"] < 0.070,
      "transit peak about 50 mm/s plus the dip (%.1f)" % (p1r["transit_peak_speed_mps"]*1e3,))
# Straight-line distance to the nearest planned point. The transit runs
# diagonally between the two passes while it dips, so its closest approach is
# not at the bottom of the dip; brute force says where it is.
T_ = np.array([x for x, p_ in zip(xyz, ph) if p_ == "transit"])
want = np.sqrt(((T_[:, None, :] - np.array(pos)[None, :, :])**2).sum(axis=2)).min()*1e3
check(abs(p1r["transit_min_distance_mm"] - want) < 1e-6 and 4.5 < want < 5.5,
      "a transit dipping between two passes reads %.2f mm from the nearest planned "
      "point (brute force %.2f)" % (p1r["transit_min_distance_mm"], want))
check(abs(p0r["wrist3_sweep_range_deg"] - 3.0) < 0.1
      and abs(s["wrist3_approach_range_max_deg"] - 40.0) < 1e-6,
      "wrist_3: 3 deg within the sweep, 40 across the approach (%.2f, %.1f)"
      % (p0r["wrist3_sweep_range_deg"], s["wrist3_approach_range_max_deg"]))
check(s["passes"] == 2 and s["swept_ok"] == 2 and abs(s["normal_offset_max_mm"] - 0.4) < 1e-6,
      "the summary carries the worst of each")
text = fmt(r, label="test")
check("closest the tip came" in text and "wrist_3" in text, "the report formats")

# --- the building blocks ------------------------------------------------------
d, e, u = polyline_nearest(np.array([[0.5, 0.2, 0.0], [2.0, 0.0, 0.0]]),
                           np.array([[0, 0, 0], [1, 0, 0]], dtype=float))
check(abs(d[0] - 0.2) < 1e-12 and abs(u[0] - 0.5) < 1e-12 and abs(d[1] - 1.0) < 1e-12,
      "polyline distance: beside the middle and past the end")
tt = np.arange(0, 2, 0.01)
xx = np.stack([0.02*tt, 0*tt, 0*tt], axis=1)
xx[50] = xx[49]                           # a repeated TF reading
v = window_speed(tt, xx)
check(abs(v[:150].max() - 0.02) < 1e-3, "windowed speed ignores a repeated reading (%.4f)"
      % (v[:150].max(),))


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================
# Everything above runs at IMPORT time. That is what lets one pytest test stand
# for the lot -- and it is also how a suite can print "ok" down the page while
# the code under test never runs. So: the subject is the real one, and it tells
# a known-positive from a known-negative.

def test_not_blind():
    """path_motion_report() ran, and it responds to what it is given."""
    assert callable(report_fn) and report_fn.__name__ == "path_motion_report"
    assert "normal_offset_mean_mm" in p0r and "passes" in r, \
        "the report came back without the fields under test"
    # KNOWN-POSITIVE / KNOWN-NEGATIVE, same call, two flights: the recorded
    # sweep of pass 0 is 0.4 mm off the surface and pass 1 is on it. If the
    # report were not measuring, both would read the same.
    assert abs(p0r["normal_offset_mean_mm"] - p1r["normal_offset_mean_mm"]) > 0.3, \
        "an offset sweep and an on-surface sweep read the same: nothing is " \
        "being measured (%r vs %r)" % (p0r["normal_offset_mean_mm"],
                                       p1r["normal_offset_mean_mm"])
    # Perturb the input and the answer must move with it.
    moved = dict(poses)
    moved["xyz"] = poses["xyz"] + n*0.001
    r2 = report_fn(path, segs, passes, moved, joints)
    assert abs(r2["passes"][1]["normal_offset_mean_mm"] - 1.0) < 1e-6, \
        "lifting the whole flight 1 mm along the normal did not move the " \
        "reported offset (%r)" % (r2["passes"][1]["normal_offset_mean_mm"],)
    # This suite builds no messages, so it must not depend on anyone's stub.
    assert "geometry_msgs" not in globals(), "a message stub leaked into this suite"


def test_surface_motion_report():
    """pytest reports what the standalone run reports.

    Every check above runs at import. Without a test function that asserts on
    what they recorded, pytest collects nothing here and a failing check prints
    FAIL while the run still passes.
    """
    assert not FAILS, "%d check(s) failed: %s" % (len(FAILS), "; ".join(FAILS))


try:
    test_not_blind()
    check(True, "the suite is not blind")
except AssertionError as _e:
    check(False, "the suite is not blind (%s)" % (_e,))

if __name__ == "__main__":
    print("")
    if FAILS:
        print("%d FAILED" % len(FAILS))
        sys.exit(1)
    print("all passed")
