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
# File:  noether_roll_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the tool-roll modes in noether_path.dpi and the report that
#   measures them.
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
"""Check the tool-roll modes in noether_path.dpi, and the report that measures them.

"locked" is the eddy-current mode: the contact probe must not turn about its own
axis during a scan. That is a cell rule -- it keeps the Azure and Keyence cables
from wrapping, and it simplifies planning. What has to hold
is zero rotation about the tool axis for ANY surface normal, a tool axis exactly
on the normal, agreement with "fixed" wherever "fixed" was already right, and a
report that tells the two apart.

Runs on the host with numpy only: the pure functions are pulled out of the .dpi
and geometry_msgs is stubbed for ScanPath.
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
# resolve. Nothing here builds a message -- fake_response() below makes its own
# pose objects, with real fields -- so ensure_msgs() only has to make the import
# work, and it does that by ADDING rather than by overwriting whatever another
# suite in this directory already installed. ros_msg_stubs.py has the outage the
# old `sys.modules[...] = mod` caused.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "noether_roll_test"}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
# noether_path.dpi reports progress through progress.dpi, which the config
# includes ahead of it; load it the same way here. Silenced, because these
# checks are about the geometry and progress.dpi has its own test.
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
ns["PROGRESS_ENABLED"] = False
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)

reg = ns["regularize_tool_roll"]
twist = ns["tool_twist_deg"]
tilt = ns["tool_tilt_deg"]
report = ns["tool_roll_report"]
m2q = ns["matrix_to_quat"]
q2m = ns["quat_to_matrix"]
ref_quat = ns["roll_reference_quat"]
scanpath_from_noether = ns["scanpath_from_noether"]

FAILS = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def pose_with_normal(n, roll=0.3):
    """Some orientation with tool z on n and an arbitrary roll, as Noether gives."""
    n = np.asarray(n, dtype=np.float64)
    n = n/np.linalg.norm(n)
    a = np.array([np.cos(roll), np.sin(roll), 0.37])
    x = a - np.dot(a, n)*n
    if np.linalg.norm(x) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
        x = a - np.dot(a, n)*n
    x = x/np.linalg.norm(x)
    return m2q(np.column_stack([x, np.cross(n, x), n]))


def normal(tilt_deg, toward_deg):
    t, p = np.radians(tilt_deg), np.radians(toward_deg)
    return (np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t))


def one(q, mode, reference=(1.0, 0.0, 0.0)):
    out, _ = reg(np.zeros((1, 3)), np.array([q]), mode=mode, reference=reference)
    return out[0]


# --- locked: zero twist for any normal in the upper hemisphere ---------------
rng = np.random.default_rng(7)
N = 3000
t_deg = np.degrees(np.arccos(rng.uniform(0.02, 1.0, N)))       # up to ~89 deg
p_deg = rng.uniform(-180.0, 180.0, N)
normals = np.array([normal(t, p) for t, p in zip(t_deg, p_deg)])
quats = np.array([pose_with_normal(n, roll=r)
                  for n, r in zip(normals, rng.uniform(-np.pi, np.pi, N))])
locked, worst = reg(np.zeros((N, 3)), quats, mode="locked")
tz = np.array([q2m(q)[:, 2] for q in locked])
check(np.abs(twist(locked)).max() < 1e-6,
      "locked: no rotation about the tool axis for %d random normals (max %.2e deg)"
      % (N, np.abs(twist(locked)).max()))
check(np.abs(tz - normals).max() < 1e-9,
      "locked: tool z lands exactly on the surface normal")
check(np.abs(tilt(locked) - t_deg).max() < 1e-6,
      "locked: tilt from vertical is the normal's own tilt")

# --- where "fixed" is already right, the two agree ----------------------------
for toward in (0.0, 90.0, 180.0, -90.0):
    for t in (0.0, 15.0, 45.0, 75.0):
        q = pose_with_normal(normal(t, toward))
        a, b = one(q, "fixed"), one(q, "locked")
        ang = np.degrees(2*np.arccos(min(1.0, abs(float(np.dot(a, b))))))
        if ang > 1e-6:
            break
    check(ang < 1e-6, "fixed == locked for tilts toward %+.0f deg (along/across "
          "the reference)" % (toward,))

# --- and where it is not, "fixed" twists and "locked" does not ----------------
expected = {10: 0.439, 30: 4.107, 45: 9.736, 60: 18.435, 80: 35.149}
for t, want in sorted(expected.items()):
    q = pose_with_normal(normal(t, 45.0))
    tf = twist(np.array([one(q, "fixed")]))[0]
    tl = twist(np.array([one(q, "locked")]))[0]
    check(abs(abs(tf) - want) < 0.01 and abs(tl) < 1e-6,
          "tilt %d deg toward 45: fixed twists %.3f deg (docstring says %.3f), "
          "locked %.1e" % (t, abs(tf), want, abs(tl)))

# --- a normal near +reference: fixed's fallback axis ---------------------------
# Within ~8 deg of the reference the projection is undefined and "fixed" falls
# back to another axis, taking the SIGN from the previous waypoint. So its roll
# there depends on waypoint order: a path that BEGINS there (no previous
# waypoint) starts a half-turn out, and one that arrives from vertical is
# rescued by the sign check. "locked" has no fallback and no order dependence.
for n in ((1.0, 0.0, 0.02), (0.999, 0.03, 0.03)):
    q = pose_with_normal(n)
    tf = twist(np.array([one(q, "fixed")]))[0]
    tl = twist(np.array([one(q, "locked")]))[0]
    check(abs(tl) < 1e-6 and abs(abs(tf) - 180.0) < 5.0,
          "a path beginning at the +x end cap %s: fixed starts %.1f deg out, "
          "locked %.1e" % (n, tf, tl))
q = pose_with_normal((-1.0, 0.01, 0.05))
tf = twist(np.array([one(q, "fixed")]))[0]
tl = twist(np.array([one(q, "locked")]))[0]
check(abs(tl) < 1e-6 and abs(tf) < 1.0,
      "near -x (the -x end cap) the fallback happens to agree: fixed %.2f, locked %.1e"
      % (tf, tl))
ramp = [pose_with_normal(normal(t, 0.0)) for t in np.linspace(0.0, 89.5, 180)]
fx = twist(reg(np.zeros((180, 3)), np.array(ramp), mode="fixed")[0])
lk = twist(reg(np.zeros((180, 3)), np.array(ramp), mode="locked")[0])
check(np.abs(fx).max() < 1e-6 and np.abs(lk).max() < 1e-6,
      "arriving at the +x end from vertical, fixed's sign check keeps it at %.1e; "
      "locked %.1e" % (np.abs(fx).max(), np.abs(lk).max()))
down = ramp[::-1]
fx = twist(reg(np.zeros((180, 3)), np.array(down), mode="fixed")[0])
lk = twist(reg(np.zeros((180, 3)), np.array(down), mode="locked")[0])
check(np.abs(fx).max() > 170.0 and np.abs(lk).max() < 1e-6,
      "the SAME poses in reverse order: fixed is %.0f deg out the whole way, "
      "locked 0 -- the roll depends on which end the planner starts from"
      % (np.abs(fx).max(),))

# --- mode="path": the serpentine flip and the degenerate-tangent sawtooth -----
# Two faults with one cause: mode="path" took its direction of travel from the
# step to the NEXT waypoint and did not keep the sign continuous. Both of those
# are what Noether's own createTransform() does, and the arm pays for them in
# wrist_3.
#
# _old_path_roll() below is that naive formulation, kept so the improvement is
# a measured number rather than an assertion.
def _old_path_roll(pos, qs, segs):
    out = []
    for a, b in segs:
        d = np.zeros((b - a, 3))
        d[:-1] = np.diff(pos[a:b], axis=0)
        d[-1] = d[-2]
        for k in range(b - a):
            z = q2m(qs[a + k])[:, 2]
            z = z/np.linalg.norm(z)
            c = d[k]
            if np.linalg.norm(c) < 1e-12:
                c = np.array([1.0, 0.0, 0.0])
            x = c - float(np.dot(c, z))*z
            nx = np.linalg.norm(x)
            if nx < 1e-9:
                out.append(qs[a + k])
                continue
            x = x/nx
            out.append(m2q(np.column_stack([x, np.cross(z, x), z])))
    return np.array(out)


def noether_like_quats(pos, segs, normal=(0.0, 0.0, 1.0)):
    """Poses the way plane_slicer_raster_planner.cpp's createTransform() builds
    them: tool z on the surface normal, tool x towards the NEXT point."""
    n = np.asarray(normal, dtype=np.float64)
    n = n/np.linalg.norm(n)
    out = np.zeros((len(pos), 4))
    for a, b in segs:
        prev = None
        for k in range(a, b):
            d = pos[min(k + 1, b - 1)] - pos[k]
            x = d - float(np.dot(d, n))*n
            if np.linalg.norm(x) < 1e-15:
                x = prev if prev is not None else np.array([1.0, 0.0, 0.0])
            x = x/np.linalg.norm(x)
            prev = x
            out[k] = m2q(np.column_stack([x, np.cross(n, x), n]))
    return out


def travel_deg(quats, segs):
    """Total roll about the tool axis, summed over every step -- the number
    wrist_3 actually turns through, not the net change."""
    tot = 0.0
    for a, b in segs:
        if b - a > 1:
            tot += float(np.abs(np.diff(np.unwrap(np.radians(
                twist(quats[a:b]))))).sum())
    return float(np.degrees(tot))


# A serpentine over a flat plate: four 0.30 m passes, alternate ones reversed.
spos, sstarts = [], []
for k in range(4):
    sstarts.append(len(spos))
    xs = np.linspace(0.60, 0.90, 31)
    for x in (xs if k % 2 == 0 else xs[::-1]):
        spos.append([x, -0.25 + 0.012*k, 0.95])
spos = np.array(spos)
sb = sstarts + [len(spos)]
ssegs = [(sb[i], sb[i+1]) for i in range(4)]
sqs = noether_like_quats(spos, ssegs)
check(float(np.dot(spos[ssegs[0][1]-1] - spos[ssegs[0][0]],
                   spos[ssegs[1][1]-1] - spos[ssegs[1][0]])) < 0.0,
      "the serpentine premise: consecutive passes really do travel opposite ways")
old = _old_path_roll(spos, sqs, ssegs)
new, _w = reg(spos, sqs, segments=ssegs, mode="path")
check(travel_deg(old, ssegs) == 0.0 and report(old, ssegs)["twist_step_deg"] < 1e-9,
      "inside a pass neither version turns the tool (%.1f deg)"
      % travel_deg(old, ssegs))
o_all, n_all = report(old)["twist_step_deg"], report(new)["twist_step_deg"]
check(abs(o_all - 180.0) < 1e-6 and n_all < 1e-6,
      "serpentine turn: the old path roll flipped %.1f deg at every row "
      "boundary, the new one %.2e (3 turns = %.0f deg of wrist_3 saved)"
      % (o_all, n_all, 3*o_all))

# A single straight 0.30 m pass with DEGENERATE waypoints: every eighth point is
# followed by one at the same x, a micron to the side -- which is what a slicing
# plane crossing a mesh edge produces, and what makes createTransform() hand back
# a tool x at 90 degrees to travel.
dpos = []
for k, x in enumerate(np.linspace(0.60, 0.90, 301)):
    dpos.append([x, -0.25, 0.95])
    if k % 8 == 7:
        dpos.append([x, -0.25 + 1e-7, 0.95])
dpos = np.array(dpos)
dsegs = [(0, len(dpos))]
dqs = noether_like_quats(dpos, dsegs)
old = _old_path_roll(dpos, dqs, dsegs)
new, _w = reg(dpos, dqs, segments=dsegs, mode="path")
ro, rn = report(old, dsegs), report(new, dsegs)
check(89.0 < ro["twist_step_deg"] < 91.0 and ro["twist_in_pass_deg"] > 89.0,
      "the degenerate premise: the old path roll steps a full %.1f deg mid-pass "
      "on a plate whose normal never changes" % (ro["twist_step_deg"],))
check(rn["twist_in_pass_deg"] < 0.1 and rn["twist_step_deg"] < 0.1,
      "a %.0f mm chord fit removes it: %.1f -> %.3f deg of roll inside the "
      "pass, and wrist_3 travel over the pass %.0f -> %.2f deg"
      % (1e3*ns["PATH_ROLL_MIN_ARC_M"], ro["twist_in_pass_deg"],
         rn["twist_in_pass_deg"], travel_deg(old, dsegs), travel_deg(new, dsegs)))
check(ns["last_roll_report"]["short_steps"] == 37
      and ns["last_roll_report"]["mode"] == "path",
      "and the report says how many sub-micron steps it saw (%d of %d)"
      % (ns["last_roll_report"]["short_steps"], len(dpos) - 1))

# The direction still tracks a pass that genuinely turns: a quarter circle in
# the table plane, tool z up. The roll has to follow it exactly.
th = np.linspace(0.0, 0.5*np.pi, 200)
cpos = np.column_stack([0.75 + 0.1*np.cos(th), -0.25 + 0.1*np.sin(th),
                        np.full(200, 0.95)])
csegs = [(0, 200)]
cnew, _w = reg(cpos, noether_like_quats(cpos, csegs), segments=csegs, mode="path")
want = np.degrees(th)
got = np.degrees(np.unwrap(np.radians(twist(cnew))))
err = np.abs((got - got[0]) - (want - want[0])).max()
# A chord over min_arc on a 0.1 m radius subtends min_arc/r; the fit cannot be
# sharper than its own window, and at the ends the window is one-sided.
window_deg = np.degrees(ns["PATH_ROLL_MIN_ARC_M"]/0.1)
# At each END the window can only be one-sided, so the fit there reads the
# tangent half a window along and the quarter circle comes out one whole window
# short. That is the price of a chord, and it is 3 degrees, not 90.
check(abs((got[-1] - got[0]) - 90.0) < 1.5*window_deg and err < 1.5*window_deg,
      "a pass that really does turn is still followed: 90 deg of arc gives "
      "%.1f deg of roll, worst error %.2f deg against a %.2f deg window"
      % (got[-1] - got[0], err, window_deg))

# A pass shorter than the fitting window gets ONE direction, not none.
tiny = np.column_stack([np.linspace(0.80, 0.802, 5), np.full(5, -0.25),
                        np.full(5, 0.95)])
tnew, _w = reg(tiny, noether_like_quats(tiny, [(0, 5)]), segments=[(0, 5)],
               mode="path")
check(report(tnew, [(0, 5)])["twist_in_pass_deg"] < 1e-9,
      "a pass shorter than the fitting window gets one direction for all of it")

# --- the profiler case: "fixed" only works about the RIGHT reference ----------
# The Keyence stage plans with roll="fixed" rather than roll="path". That is
# only correct because the reference it uses is the
# raster's own travel direction: the laser line lies along tool y, so tool x has
# to run ALONG travel or every pass sweeps a strip of zero width -- a sweep that
# reports 100% of every row, points on every profile, and 12% of the plate.
#
# Here the rows run along world +y, which is the geometry that failed.
rpos, rstarts = [], []
for k in range(4):
    rstarts.append(len(rpos))
    ys = np.linspace(-0.30, -0.10, 21)
    for y in (ys if k % 2 == 0 else ys[::-1]):
        rpos.append([0.80 + 0.012*k, y, 0.95])
rpos = np.array(rpos)
rb = rstarts + [len(rpos)]
rsegs = [(rb[i], rb[i+1]) for i in range(4)]
rqs = noether_like_quats(rpos, rsegs)
along = ref_quat((0.0, 1.0, 0.0))                 # tool x along the rows
good, _w = reg(rpos, rqs, segments=rsegs, mode="fixed", reference=along)
bad, _w = reg(rpos, rqs, segments=rsegs, mode="fixed")     # the default +x
gx = np.array([q2m(q)[:, 0] for q in good])
bx = np.array([q2m(q)[:, 0] for q in bad])
check(np.abs(np.abs(gx.dot([0.0, 1.0, 0.0])) - 1.0).max() < 1e-9
      and report(good, rsegs)["twist_range_deg"] < 1e-9,
      "rows along world y, reference along the rows: tool x is on travel for "
      "every pose and the roll never changes")
check(np.abs(np.abs(bx.dot([1.0, 0.0, 0.0])) - 1.0).max() < 1e-9,
      "the SAME rows with the default +x reference put tool x across travel -- "
      "the laser line would lie along the pass, which is the 12%-coverage "
      "failure, so the stage has to pass its own reference")
pth, _w = reg(rpos, rqs, segments=rsegs, mode="path")
px = np.array([q2m(q)[:, 0] for q in pth])
check(np.abs(np.abs(px.dot([0.0, 1.0, 0.0])) - 1.0).max() < 1e-9
      and report(pth, rsegs)["twist_range_deg"] < 1e-6,
      "and mode='path' now agrees with it instead of flipping at the turns "
      "(roll range %.2e deg over 4 rows)"
      % (report(pth, rsegs)["twist_range_deg"],))

# --- the reference picks the roll ---------------------------------------------
q = one(pose_with_normal((0.0, 0.0, 1.0)), "locked", reference=(0.0, 1.0, 0.0))
check(np.allclose(q2m(q)[:, 0], (0.0, 1.0, 0.0), atol=1e-9),
      "reference (0,1,0): a level tool has x along world y")
rq = ref_quat((0.0, 1.0, 0.0))
q = one(pose_with_normal(normal(20.0, 30.0)), "locked", reference=rq)
check(abs(twist(np.array([q]), reference=rq)[0]) < 1e-6,
      "a quaternion reference is used as given")

# --- a tool axis pointing into the floor is refused, not guessed --------------
try:
    one(pose_with_normal((0.0, 0.0, -1.0)), "locked")
    check(False, "locked refuses a tool axis opposite the reference")
except ValueError:
    check(True, "locked refuses a tool axis opposite the reference")


# --- the report on a cylinder: the fiducial cylinder's geometry ---------------
def cylinder_path(axis_deg, r=0.03169, length=0.129, max_theta=60.0, spacing_deg=5.0):
    """Passes along a cylinder lying horizontally at `axis_deg` in the table
    plane, one pass per angle around it, normals analytic."""
    a = np.radians(axis_deg)
    ax = np.array([np.cos(a), np.sin(a), 0.0])
    side = np.cross((0.0, 0.0, 1.0), ax)
    pos, qs, starts = [], [], []
    for th in np.arange(-max_theta, max_theta + 1e-9, spacing_deg):
        starts.append(len(pos))
        n = np.cos(np.radians(th))*np.array([0.0, 0.0, 1.0]) + np.sin(np.radians(th))*side
        for s in np.linspace(0.0, length, 40):
            pos.append(np.array([0.8, -0.25, 0.96]) + s*ax + r*n)
            qs.append(pose_with_normal(n, roll=1.1))
    return np.array(pos), np.array(qs), starts


for axis_deg in (0.0, 90.0, 30.0):
    pos, qs, starts = cylinder_path(axis_deg)
    bounds = list(starts) + [len(pos)]
    segs = [(bounds[i], bounds[i+1]) for i in range(len(starts))]
    rf = report(reg(pos, qs, segments=segs, mode="fixed")[0], segs)
    rl = report(reg(pos, qs, segments=segs, mode="locked")[0], segs)
    if axis_deg in (0.0, 90.0):
        check(rf["twist_range_deg"] < 1e-6 and rl["twist_range_deg"] < 1e-6,
              "cylinder along %.0f deg: fixed and locked both 0 roll (%.2e, %.2e)"
              % (axis_deg, rf["twist_range_deg"], rl["twist_range_deg"]))
    else:
        check(rf["twist_range_deg"] > 5.0 and rl["twist_range_deg"] < 1e-6,
              "cylinder at 30 deg: fixed rolls %.1f deg across the part, locked %.1e"
              % (rf["twist_range_deg"], rl["twist_range_deg"]))
    check(abs(rl["tilt_max_deg"] - 60.0) < 1e-6 and rl["twist_in_pass_deg"] < 1e-6,
          "cylinder along %.0f deg: tilt to 60 deg on the flanks, no roll in a pass"
          % (axis_deg,))

# --- wrap: a path sitting at +-180 is a small spread, not 360 -----------------
q180 = np.array([one(pose_with_normal((0.0, 0.0, 1.0)), "locked",
                     reference=(np.cos(np.radians(d)), np.sin(np.radians(d)), 0.0))
                 for d in (178.0, 179.0, -179.0, -178.0)])
r = report(q180)
check(abs(r["twist_range_deg"] - 4.0) < 1e-6,
      "a roll straddling 180 reads a 4 deg spread (%.3f)" % (r["twist_range_deg"],))


# --- scanpath_from_noether carries the reference and applies it ---------------
class _P(object):
    pass


def fake_response(pos, qs, starts):
    res = _P()
    res.noether_pose_array = _P()
    res.noether_pose_array.poses = []
    for p, q in zip(pos, qs):
        pose = _P()
        pose.position = _P()
        pose.orientation = _P()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p]
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = [float(v) for v in q]
        res.noether_pose_array.poses.append(pose)
    res.segment_start_indices = list(starts)
    return res


pos, qs, starts = cylinder_path(30.0)
res = fake_response(pos, qs, starts)
path = scanpath_from_noether(res, roll="locked", roll_reference=(0.0, -1.0, 0.0))
check(path.roll_mode == "locked" and path.roll_reference == (0.0, -1.0, 0.0),
      "the path records which roll mode and reference built it")
check(np.abs(twist(path.quats, reference=(0.0, -1.0, 0.0))).max() < 1e-6,
      "scanpath_from_noether(roll='locked') is locked to the reference it was given")
old = scanpath_from_noether(res)
check(old.roll_mode == "fixed" and old.roll_reference == (1.0, 0.0, 0.0),
      "the default is still 'fixed' about +x, so UTPlaneScan is unchanged")

# --- orient_segments_along: the direction of travel is a choice --------------
osa = ns["orient_segments_along"]
pos = np.array([[0.9, 0, 0], [0.8, 0, 0], [0.7, 0, 0],      # travels -x
                [0.7, 0.01, 0], [0.8, 0.01, 0], [0.9, 0.01, 0],   # travels +x
                [0.8, 0.02, 0], [0.8, 0.03, 0]], dtype=float)     # square to x
qs = np.tile([0.0, 0.0, 0.0, 1.0], (8, 1))
qs[3] = (0.0, 0.0, 0.0871557, 0.9961947)                        # marks a waypoint
p = ScanPath = ns["ScanPath"](pos, qs, [0, 3, 6], source="t")
p.roll_mode = "locked"
o = osa(p, (-1.0, 0.0, 0.0))
check(o.segments_reversed == 1 and np.allclose(o.positions[3:6, 0], [0.9, 0.8, 0.7])
      and np.allclose(o.quats[5], qs[3]) and np.allclose(o.positions[6:8], pos[6:8]),
      "orient_segments_along: the +x pass is reversed with its orientations, the "
      "-x pass and the square one are left alone")
check(o.roll_mode == "locked" and o.segments == [(0, 3), (3, 6), (6, 8)],
      "and the segments and the roll bookkeeping survive")

# --- locked_quats_from_axes / smooth_tool_axis --------------------------------
lqa = ns["locked_quats_from_axes"]
sta = ns["smooth_tool_axis"]
rng = np.random.default_rng(7)
axes = rng.normal(size=(200, 3))
axes[:, 2] = np.abs(axes[:, 2]) + 0.05                 # tool z up, as Noether's are
for ref in ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.6, 0.8, 0.0)):
    qv = lqa(axes, ref)
    ql, _ = reg(np.zeros((200, 3)), np.array([pose_with_normal(a) for a in axes]),
                mode="locked", reference=ref)
    check(max(np.abs(q2m(a) - q2m(b)).max() for a, b in zip(qv, ql)) < 1e-9,
          "locked_quats_from_axes == regularize_tool_roll('locked'), reference %s" % (ref,))

# A cylinder pass runs along the axis: the normal is constant, nothing moves.
pos = np.column_stack([np.linspace(0.8, 0.9, 101), np.full(101, -0.25), np.full(101, 0.99)])
n30 = normal(30.0, 90.0)
cyl = ns["ScanPath"](pos, lqa(np.tile(n30, (101, 1))), [0], source="t")
sm = sta(cyl, 0.006)
check(np.array_equal(sm.positions, cyl.positions) and sm.axis_smoothing["moved_max_deg"] < 1e-6
      and np.abs(tilt(sm.quats) - 30.0).max() < 1e-6,
      "smooth_tool_axis: a constant-normal pass is unchanged, positions untouched")

# A flat pass with +-10 degree waypoint noise: the probe stops rocking.
noisy = np.array([normal(abs(a), b) for a, b in zip(rng.normal(0.0, 6.0, 101),
                                                    rng.uniform(0.0, 360.0, 101))])
flat = ns["ScanPath"](pos, lqa(noisy), [0, 50], source="t")
sm = sta(flat, 0.020)
check(tilt(sm.quats).mean() < 0.4*tilt(flat.quats).mean()
      and np.abs(twist(sm.quats)).max() < 1e-6 and sm.segments == flat.segments,
      "smooth_tool_axis: waypoint noise is averaged out (tilt mean %.1f -> %.1f deg), "
      "roll stays locked, segments kept" % (tilt(flat.quats).mean(), tilt(sm.quats).mean()))

# The average never crosses a segment boundary: two passes with different
# normals keep their own.
two = np.vstack([np.tile(normal(0.0, 0.0), (50, 1)), np.tile(normal(40.0, 0.0), (51, 1))])
tp = ns["ScanPath"](pos, lqa(two), [0, 50], source="t")
sm = sta(tp, 0.050)
check(np.abs(tilt(sm.quats)[:50]).max() < 1e-6 and np.abs(tilt(sm.quats)[50:] - 40.0).max() < 1e-6,
      "smooth_tool_axis: no averaging across a segment boundary")
check(sta(tp, None) is tp and sta(tp, 0.0) is tp, "smooth_tool_axis: None / 0 is off")

# --- the turn-rate repair -----------------------------------------------------
#
# EC_MAX_TILT_DEG bounds how far the probe leans. Nothing bounded how fast the
# lean CHANGES, and that is the one the arm feels: on 2026-10-08 ten passes of
# eighty-six crawled at 1.74 mm/s against a 4.68 median while the arm visibly
# rocked, pass speed against worst turn rate came out at r = -0.855, and the
# median tilt on those passes was 0.429 deg against 0.423 elsewhere. The repair
# has to SMOOTH, not refuse: a gate that only says no leaves an operator with a
# scan they cannot run.

def _cyl_pass(radius_m, n=121, span=0.060, noise_deg=0.0, seed=3):
    """A pass running AROUND a cylinder, so the true normal genuinely turns."""
    r = np.random.default_rng(seed)
    s = np.linspace(-span/2, span/2, n)
    th = s/radius_m
    pos = np.column_stack([radius_m*np.sin(th), np.full(n, -0.25),
                           0.99 + radius_m*(np.cos(th) - 1.0)])
    true = np.column_stack([np.sin(th), np.zeros(n), np.cos(th)])
    ax = true.copy()
    if noise_deg:
        ax[:, 0] += np.radians(noise_deg)*r.standard_normal(n)
        ax[:, 1] += np.radians(noise_deg)*r.standard_normal(n)
        ax = ax/np.linalg.norm(ax, axis=1, keepdims=True)
    return pos, true, ax

def _axis_of(q):
    m = np.array([q2m(a) for a in q])
    return m[:, :, 2]

def _err(a, t):
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, t), -1.0, 1.0)))

# GENERALITY: a curved surface keeps its curvature. The whole point of fitting a
# local LINE rather than taking a mean is that a constant turn survives, so a
# part that is smooth but not flat is not flattened.
pos, true, ax = _cyl_pass(0.060, noise_deg=0.0)
cu = ns["ScanPath"](pos, lqa(ax), [0], source="t")
sm = sta(cu, 0.010, max_turn_deg_per_mm=None, quiet=True)
check(_err(_axis_of(sm.quats), true).max() < 0.02,
      "turn-rate repair: a 60 mm-radius curve is reproduced to %.3f deg -- the real "
      "turn is not smoothed away" % (_err(_axis_of(sm.quats), true).max(),))

# ...and it is reproduced AT THE ENDS, where the window is one-sided. This is
# the bias that made a box mean unsafe on a normal-constrained path.
pos, true, ax = _cyl_pass(0.060, noise_deg=1.0)
cu = ns["ScanPath"](pos, lqa(ax), [0], source="t")
lin = _err(_axis_of(sta(cu, 0.010, max_turn_deg_per_mm=None, quiet=True).quats), true)
box = _err(_axis_of(sta(cu, 0.010, method="box", force=True, quiet=True).quats), true)
ends = np.r_[0:10, len(true)-10:len(true)]
check(lin[ends].mean() < box[ends].mean(),
      "turn-rate repair: linear beats a box mean at the pass ENDS on a curve "
      "(%.3f vs %.3f deg)" % (lin[ends].mean(), box[ends].mean()))
check(lin.mean() < _err(ax, true).mean(),
      "turn-rate repair: and it is closer to the true normal than the raw axis "
      "(%.3f vs %.3f deg)" % (lin.mean(), _err(ax, true).mean()))

# A box mean is still refused on a normal-constrained path; linear is not.
pos, true, ax = _cyl_pass(0.060, noise_deg=1.0)
nc = ns["ScanPath"](pos, lqa(ax), [0], source="t")
nc.normal_constrained = True
check(sta(nc, 0.010, method="box") is nc,
      "turn-rate repair: a box mean is still declined on a normal-constrained path")
out = sta(nc, 0.010, method="linear", max_turn_deg_per_mm=None, quiet=True)
check(out is not nc and out.axis_smoothing["method"] == "linear",
      "turn-rate repair: the unbiased estimator is allowed on one")

# IT REPAIRS RATHER THAN REFUSING: a noisy flat pass over the bound comes back
# under it, with a wider window, and says what it did.
r = np.random.default_rng(11)
n = 121
pos = np.column_stack([np.linspace(0.80, 0.86, n), np.full(n, -0.25), np.full(n, 0.99)])
ax = np.column_stack([np.radians(1.5)*r.standard_normal(n),
                      np.radians(1.5)*r.standard_normal(n), np.ones(n)])
ax = ax/np.linalg.norm(ax, axis=1, keepdims=True)
noisy = ns["ScanPath"](pos, lqa(ax), [0], source="t")
noisy.normal_constrained = True
sm = sta(noisy, 0.004, max_turn_deg_per_mm=0.4, max_window_m=0.020, quiet=True)
a = sm.axis_smoothing
check(a["turn_before_max"] > 0.4 and a["turn_after_max"] <= 0.4 and a["turn_rate_met"],
      "turn-rate repair: %.2f -> %.2f deg/mm, under the 0.40 bound"
      % (a["turn_before_max"], a["turn_after_max"]))
check(a["widened"] and a["window_m"] > a["requested_window_m"],
      "turn-rate repair: the window widened from %.0f to %.0f mm to get there"
      % (a["requested_window_m"]*1e3, a["window_m"]*1e3))
check(np.array_equal(sm.positions, noisy.positions),
      "turn-rate repair: and the tip still goes exactly where the mesh says")

# A surface that genuinely turns faster than the bound is NOT refused -- it is
# reported. A 10 mm radius turns at 5.7 deg/mm and is a real part, not noise.
pos, true, ax = _cyl_pass(0.010, span=0.020, noise_deg=0.0)
sharp = ns["ScanPath"](pos, lqa(ax), [0], source="t")
sharp.normal_constrained = True
sm = sta(sharp, 0.004, max_turn_deg_per_mm=0.4, max_window_m=0.020, quiet=True)
check(sm.axis_smoothing["turn_rate_met"] is False,
      "turn-rate repair: a genuinely sharp curve is reported as over the bound")
check(_err(_axis_of(sm.quats), true).max() < 0.5,
      "turn-rate repair: ...and NOT flattened to meet it (%.3f deg from the true "
      "normal)" % (_err(_axis_of(sm.quats), true).max(),))
check(sm.axis_smoothing["stopped_helping"] and not sm.axis_smoothing["widened"],
      "turn-rate repair: it stopped widening at once, because a wider window "
      "bought no reduction -- the turn is the part, not the mesh")

# The mixed case, which is the one a real curved specimen gives: a sharp curve
# WITH mesh noise on top. The noise must go and the curve must stay.
pos, true, ax = _cyl_pass(0.010, span=0.020, noise_deg=1.0)
mix = ns["ScanPath"](pos, lqa(ax), [0], source="t")
mix.normal_constrained = True
sm = sta(mix, 0.004, max_turn_deg_per_mm=0.4, max_window_m=0.020, quiet=True)
check(_err(_axis_of(sm.quats), true).mean() < _err(ax, true).mean(),
      "turn-rate repair: on a sharp curve WITH noise it still improves the axis "
      "(%.3f -> %.3f deg from truth)"
      % (_err(ax, true).mean(), _err(_axis_of(sm.quats), true).mean()))
check(not sm.axis_smoothing["turn_rate_met"],
      "turn-rate repair: ...and still reports the real 5.7 deg/mm turn rather "
      "than smoothing it away to satisfy the bound")


# --- resample_segments ---------------------------------------------------------
rs = ns["resample_segments"]
x = np.concatenate([[0.8], 0.8 + np.cumsum(rng.uniform(0.0, 0.0005, 400))])
x = np.concatenate([x[:100], np.repeat(x[100], 5), x[100:]])            # near-duplicates
pos = np.column_stack([x, np.full(len(x), -0.25), 0.99 + 0.01*np.sin(40.0*(x - 0.8))])
ax = np.column_stack([-0.4*np.cos(40.0*(x - 0.8)), np.zeros(len(x)), np.ones(len(x))])
p3 = ns["ScanPath"](np.vstack([pos, [[0.8, -0.24, 0.99], [0.8004, -0.24, 0.99]]]),
                    lqa(np.vstack([ax, [[0, 0, 1], [0, 0, 1]]])), [0, len(x)], source="t")
r3 = rs(p3, 0.001)
seg0 = r3.segments[0]
d = np.linalg.norm(np.diff(r3.positions[seg0[0]:seg0[1]], axis=0), axis=1)
check(np.allclose(r3.positions[seg0[0]], pos[0]) and np.allclose(r3.positions[seg0[1]-1], pos[-1])
      and d.max()/d.mean() < 1.01 and abs(d.mean() - 0.001) < 0.0001,
      "resample_segments: ends kept, uniform 1 mm steps (max/mean %.3f; before %.1f)"
      % (d.max()/d.mean(), (lambda e: e.max()/e.mean())(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
check(np.abs(twist(r3.quats)).max() < 1e-6, "resample_segments: the roll stays locked")
check(r3.segments[1] == (seg0[1], seg0[1] + 2) and np.allclose(r3.positions[-2:], p3.positions[-2:]),
      "resample_segments: a segment shorter than the spacing is left as it is")
check(rs(p3, None) is p3, "resample_segments: None is off")

# --- clean_mesh_for_planning ----------------------------------------------------
import tempfile
cm = ns["clean_mesh_for_planning"]
rd = ns["read_ply_mesh"]
g = np.array([[i*0.01, j*0.01, 0.0] for i in range(21) for j in range(21)])   # 20x20 quads
tri = []
for i in range(20):
    for j in range(20):
        k = i*21 + j
        tri += [(k, k + 21, k + 1), (k + 1, k + 21, k + 22)]
nv = len(g)
crumb = np.array([[0.5, 0.5, 0.0], [0.51, 0.5, 0.0], [0.5, 0.51, 0.0]])        # a 1-face crumb
fin = np.array([[0.0, 0.0, 0.01]])                                               # a third face on edge (1, 21)
V = np.vstack([g, crumb, fin])
T = np.array(tri + [(nv, nv + 1, nv + 2), (1, 21, nv + 3)])
d0 = tempfile.mkdtemp()
fn = os.path.join(d0, "m.ply")
with open(fn, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nelement face %d\nproperty list uchar int vertex_indices\n"
            "end_header\n" % (len(V), len(T)))
    f.writelines("%f %f %f\n" % tuple(p) for p in V)
    f.writelines("3 %d %d %d\n" % tuple(t) for t in T)
r = cm(fn, os.path.join(d0, "o.ply"))
v2, n2, t2 = rd(os.path.join(d0, "o.ply"))
check(r["nonmanifold_edges"] == 1 and r["faces_on_nonmanifold_edges"] == 3
      and r["faces_in_small_components"] == 1 and len(t2) == 798 and n2 is None,
      "clean_mesh_for_planning: the three faces on a non-manifold edge and a crumb go, "
      "the rest stays (%d of %d faces)" % (len(t2), len(T)))

# --- faces wound the wrong way ------------------------------------------------
# TWO FACES IN THIRTEEN THOUSAND COST 47% OF AN EDDY-CURRENT TOOLPATH on a
# cylinder. A clearance check's signed distance takes its SIGN from
# the nearest face's winding, and the gate takes the MINIMUM over ~120 000
# tooling sample points -- so one face the wrong way round reports a link 200 mm
# clear of the part as 200 mm inside it, and every pose near that face is
# dropped. The mesh looked fine by every other measure: no NaNs, no degenerate
# faces, no non-manifold edges, one component, 99.98% of faces correct.
up = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(g), 1))
T2 = np.array(tri, dtype=np.int64)
T2[7] = T2[7][[0, 2, 1]]          # one face wound against the normals
T2[113] = T2[113][[0, 2, 1]]      # and a second, the count a real mesh had
fn2 = os.path.join(d0, "wound.ply")
with open(fn2, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\n"
            "property float y\nproperty float z\nproperty float nx\n"
            "property float ny\nproperty float nz\nelement face %d\n"
            "property list uchar int vertex_indices\nend_header\n"
            % (len(g), len(T2)))
    f.writelines("%f %f %f %f %f %f\n" % (tuple(p) + tuple(q))
                 for p, q in zip(g, up))
    f.writelines("3 %d %d %d\n" % tuple(t) for t in T2)

def _down_faces(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return int((fn[:, 2] < 0).sum())

check(_down_faces(g, T2) == 2, "the fixture really has 2 faces wound backwards")
r2 = cm(fn2, os.path.join(d0, "wound_out.ply"))
v3, n3, t3 = rd(os.path.join(d0, "wound_out.ply"))
check(r2.get("faces_rewound") == 2,
      "clean_mesh_for_planning: both backwards faces are REWOUND (%s)"
      % (r2.get("faces_rewound"),))
check(_down_faces(v3, t3) == 0,
      "  and the cleaned mesh has none left (%d)" % (_down_faces(v3, t3),))
check(len(t3) == len(T2) and n3 is not None,
      "  without dropping a face or the normals (%d of %d faces, normals %s)"
      % (len(t3), len(T2), "kept" if n3 is not None else "LOST"))

# A mesh with no normals is left alone: there is nothing to decide against, and
# "make them all point up" would invert the underside of any closed surface.
r3 = cm(fn, os.path.join(d0, "o2.ply"))
check(r3.get("faces_rewound") == 0,
      "a mesh with no vertex normals is not rewound on a guess")

# --- and the faces whose NORMALS are wrong too --------------------------------
# The pass above trusts the vertex normals, so a face whose winding AND normals
# both point into the part survives it. Four such faces in 27 771 still cost
# 17.5% of an eddy-current path once the other 17 are turned.
# On a surface that says it is a height field -- nearly every normal up, which
# is what the reconstruction guarantees -- a downward face is an artefact.
T3 = np.array(tri, dtype=np.int64)
up3 = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(g), 1))
for _i in (11, 250):
    T3[_i] = T3[_i][[0, 2, 1]]
    up3[T3[_i]] = np.array([0.0, 0.0, -1.0])      # its normals are wrong too
fn3 = os.path.join(d0, "hf.ply")
with open(fn3, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\n"
            "property float y\nproperty float z\nproperty float nx\n"
            "property float ny\nproperty float nz\nelement face %d\n"
            "property list uchar int vertex_indices\nend_header\n"
            % (len(g), len(T3)))
    f.writelines("%f %f %f %f %f %f\n" % (tuple(p) + tuple(q))
                 for p, q in zip(g, up3))
    f.writelines("3 %d %d %d\n" % tuple(t) for t in T3)
# NORMALS ARE PER-VERTEX AND VERTICES ARE SHARED, so a fixture cannot mark two
# faces' normals down without touching their neighbours': the first pass sees
# those neighbours disagreeing and turns them DOWN, and the second turns them
# back up. That churn is the point -- a mesh whose normals lie cannot be
# repaired by the normals alone, and the height-field pass is what closes it.
# So assert the OUTCOME rather than counts that depend on connectivity.
r4 = cm(fn3, os.path.join(d0, "hf_out.ply"))
v4, n4, t4 = rd(os.path.join(d0, "hf_out.ply"))
check(r4.get("faces_turned_up", 0) > 0,
      "the height-field pass fires on faces the normals pass could not fix "
      "(%s turned up after %s rewound)"
      % (r4.get("faces_turned_up"), r4.get("faces_rewound")))
check(_down_faces(v4, t4) == 0,
      "  and none point down afterwards (%d)" % (_down_faces(v4, t4),))
check(n4 is not None and int((np.asarray(n4)[:, 2] < 0).sum()) == 0,
      "  with their normals corrected to match")

# A CLOSED mesh is left alone: half of it legitimately faces down, so the
# height-field premise does not hold and flipping would invert its underside.
box_v, box_t = [], []
for z in (0.0, 0.05):
    for x in (0.0, 0.1):
        for y in (0.0, 0.1):
            box_v.append((x, y, z))
box_v = np.array(box_v, float)
box_n = box_v - box_v.mean(axis=0)
box_n /= np.linalg.norm(box_n, axis=1)[:, None]
for a, b, c in ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5),
                (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
                (1, 5, 7), (1, 7, 3)):
    box_t.append((a, b, c))
fn4 = os.path.join(d0, "closed.ply")
with open(fn4, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\n"
            "property float y\nproperty float z\nproperty float nx\n"
            "property float ny\nproperty float nz\nelement face %d\n"
            "property list uchar int vertex_indices\nend_header\n"
            % (len(box_v), len(box_t)))
    f.writelines("%f %f %f %f %f %f\n" % (tuple(p) + tuple(q))
                 for p, q in zip(box_v, box_n))
    f.writelines("3 %d %d %d\n" % t for t in box_t)
r5 = cm(fn4, os.path.join(d0, "closed_out.ply"))
check(r5.get("faces_turned_up") == 0,
      "a CLOSED mesh is not turned up: half of it faces down and should "
      "(%s turned, %.0f%% of normals up)"
      % (r5.get("faces_turned_up"),
         100.0*r5.get("height_field_up_fraction", 0.0)))


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================
# Every check above runs at IMPORT, which is what lets one pytest test stand for
# the lot -- and is also how a suite prints "ok" down the page while the code
# under test never runs. So: the subject is the real one, the poses it is given
# have real fields, and it tells a known-positive from a known-negative.

def test_not_blind():
    """regularize_tool_roll() and scanpath_from_noether() ran, and discriminate."""
    assert reg.__name__ == "regularize_tool_roll" and callable(report)
    # OUR pose objects, not a stub's: fake_response() builds _P instances and
    # scanpath_from_noether() reads their fields. A field-less Pose from another
    # suite cannot reach this.
    probe = fake_response(np.zeros((1, 3)), [[0.0, 0.0, 0.0, 1.0]], [0])
    first = probe.noether_pose_array.poses[0]
    assert type(first) is _P and hasattr(first.position, "x") \
        and hasattr(first.orientation, "w"), "the poses under test have no fields"
    # KNOWN-POSITIVE / KNOWN-NEGATIVE through the same call: a pose tilted 30 deg
    # toward 45 deg twists 4.107 deg under "fixed" and not at all under "locked"
    # (the measured table above). If regularize_tool_roll() were not running, or
    # not reading its mode, the two would be identical.
    q30 = pose_with_normal(normal(30.0, 45.0))
    fx = abs(float(twist(np.array([one(q30, "fixed")]))[0]))
    lk = abs(float(twist(np.array([one(q30, "locked")]))[0]))
    assert abs(fx - 4.107) < 0.01 and lk < 1e-6, \
        "fixed and locked gave %g and %g on a 30 deg tilt toward 45 (want " \
        "4.107 and 0): the roll regularizer is not being consulted" % (fx, lk)
    # And the path constructor really reads the reference it is handed.
    a = scanpath_from_noether(res, roll="locked", roll_reference=(0.0, -1.0, 0.0))
    b = scanpath_from_noether(res, roll="locked", roll_reference=(1.0, 0.0, 0.0))
    assert not np.allclose(a.quats, b.quats), \
        "two different roll references produced the same quaternions"


def test_noether_roll():
    """pytest reports what the standalone run reports.

    Without this the file collected NO tests: `pytest noether_roll_test.py` said
    "no tests ran", exit 5, and a failing check printed FAIL and failed nothing.
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
