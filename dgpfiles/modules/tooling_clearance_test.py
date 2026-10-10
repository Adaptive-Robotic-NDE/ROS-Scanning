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
# File:  tooling_clearance_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the tooling-clearance module in tooling_clearance.dpi.
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
"""Check the tooling-clearance module in tooling_clearance.dpi.

The fault these guard against: a scan stage places the ONE sensor it cares about
and the rest of the tooling goes wherever the geometry sends it. A Keyence
raster can put `lj_v7080_optical_frame` correctly in its +/-23 mm window and the
eddy-current probe barrel into the specimen, because nothing else in the
pipeline asks where the other three sensors are.

Everything here runs on the host with numpy (scipy if present, and the same
answers without it): the geometry comes from URDF text written in the test, the
surfaces are generated, and no ROS, robot, planning scene or container is
involved. That is the point -- a check that needs a cell to test is a check
nobody runs.

    python3 tooling_clearance_test.py
"""
import io
import math
import os
import re
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve. Nothing in this suite builds a message -- the poses are numpy arrays
# and quaternion tuples throughout -- so ensure_msgs() only has to make the
# import work, and it does that by ADDING names rather than by overwriting what
# another suite in this directory already installed. ros_msg_stubs.py has the
# outage the old `sys.modules[...] = mod` caused.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    # A .dpi declares its exports global before assigning them; at module scope
    # "global X" before "class X" is a SyntaxError, so strip them.
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "tooling_clearance_test"}
for _f in ("toolpath_segments.dpi", "noether_path.dpi", "tooling_clearance.dpi"):
    exec(compile(_load(_f), _f, "exec"), ns)

ScanPath = ns["ScanPath"]
matrix_to_quat = ns["matrix_to_quat"]
quat_to_matrix = ns["quat_to_matrix"]
tool_tilt_deg = ns["tool_tilt_deg"]
tool_twist_deg = ns["tool_twist_deg"]
cap_tool_tilt = ns["cap_tool_tilt"]
tooling_envelope = ns["tooling_envelope"]
SurfaceDistance = ns["SurfaceDistance"]
path_clearance = ns["path_clearance"]
enforce_path_clearance = ns["enforce_path_clearance"]
sampled_path_clearance = ns["sampled_path_clearance"]
path_step_travel = ns["path_step_travel"]

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok    %s" % (name,))
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILURES.append(name)


# ------------------------------------------------------------------------------
# Geometry the tests are built from
# ------------------------------------------------------------------------------

#: The real tooling's topology, in miniature: an IK frame at the tip of a barrel
#: that extends up from it, a sensor frame offset sideways and down, and the arm
#: on the other side of a REVOLUTE joint, which the envelope must not cross.
#:
#: The numbers are the measured ones -- probe_end -> lj_v7080_optical_frame is
#: (-75.15, -2.20, -34.28) mm in the cell's current description, and ec_probe is
#: a 7.1 mm-radius barrel 68 mm long whose tip is probe_end.
TOOLING_URDF = """<?xml version="1.0"?>
<robot name="test_tooling">
  <link name="upper_arm_link">
    <collision><geometry><box size="0.1 0.1 0.4"/></geometry></collision>
  </link>
  <link name="wrist_link">
    <collision><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <link name="tool0"/>
  <link name="ec_probe">
    <collision>
      <origin xyz="0 0 0.034" rpy="0 0 0"/>
      <geometry><cylinder radius="0.0071" length="0.068"/></geometry>
    </collision>
  </link>
  <link name="probe_end"/>
  <!-- The sensor BODY is above its own optical frame by its reference
       distance: a profiler measures at a working distance, it does not touch.
       0.04..0.08 m above the frame, as the real head is. -->
  <link name="sensor_body">
    <collision>
      <origin xyz="0 0 0.06" rpy="0 0 0"/>
      <geometry><box size="0.06 0.03 0.04"/></geometry>
    </collision>
  </link>
  <link name="sensor_optical_frame"/>
  <joint name="elbow" type="revolute">
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <parent link="upper_arm_link"/><child link="wrist_link"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <!-- The arm is ABOVE and the probe tip is at the bottom, with probe_end's
       +z pointing back up toward the arm: the projector's convention and the
       cell's. So a link's collision points come out at POSITIVE z in the
       envelope's frame, and anything at negative z is on the part's side. -->
  <joint name="wrist-tool0" type="fixed">
    <origin xyz="0 0 -0.04" rpy="0 0 0"/>
    <parent link="wrist_link"/><child link="tool0"/>
  </joint>
  <joint name="tool0-ec_probe" type="fixed">
    <origin xyz="0 0 -0.068" rpy="0 0 0"/>
    <parent link="tool0"/><child link="ec_probe"/>
  </joint>
  <joint name="ec_probe-probe_end" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="ec_probe"/><child link="probe_end"/>
  </joint>
  <joint name="probe_end-sensor_body" type="fixed">
    <origin xyz="-0.07515 -0.0022 -0.03428" rpy="0 0 0"/>
    <parent link="probe_end"/><child link="sensor_body"/>
  </joint>
  <joint name="sensor_body-optical" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="sensor_body"/><child link="sensor_optical_frame"/>
  </joint>
</robot>
"""

SENSOR_OFFSET = np.array([-0.07515, -0.0022, -0.03428])


def plane_mesh(z=0.0, half=0.25, n=30):
    """A flat plate, triangulated, normals up."""
    xs = np.linspace(-half, half, n + 1)
    ys = np.linspace(-half, half, n + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.column_stack([xx.reshape(-1), yy.reshape(-1),
                             np.full(xx.size, z)])
    tris = []
    for i in range(n):
        for j in range(n):
            a = i*(n + 1) + j
            b = a + 1
            c = a + n + 1
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    return verts, np.asarray(tris, dtype=np.int64)


def cylinder_mesh(radius, length, axis_z=0.0, nphi=120, nt=40, phi_span=100.0):
    """The top of a cylinder lying along x, centred on the origin in x and y.

    phi_span degrees either side of vertical, so the default is a little past
    the tangent lines -- which is where a rough mesh of a resting cylinder
    stops, and where the interesting poses are.
    """
    phi = np.radians(np.linspace(-phi_span, phi_span, nphi + 1))
    t = np.linspace(-length/2.0, length/2.0, nt + 1)
    tt, pp = np.meshgrid(t, phi, indexing="ij")
    verts = np.column_stack([
        tt.reshape(-1),
        (radius*np.sin(pp)).reshape(-1),
        (axis_z + radius*np.cos(pp)).reshape(-1)])
    tris = []
    for i in range(nt):
        for j in range(nphi):
            a = i*(nphi + 1) + j
            b = a + 1
            c = a + nphi + 1
            d = c + 1
            tris.append([a, b, c])
            tris.append([b, d, c])
    return verts, np.asarray(tris, dtype=np.int64)


def sensor_on_surface_path(points, normals, travel, tilt_cap_deg=None):
    """Poses that put the SENSOR frame on `points`, the way a raster does.

    Exactly what _KeyenceReconstruction.plan_sweep() builds: plan on the
    surface, orient the tool (vertical, or following the normal up to a cap),
    then move every pose back along the probe_end -> sensor vector IN THE TOOL
    FRAME. The whole fault is in that last step, so a test that skips it tests
    nothing.
    """
    up = np.array([0.0, 0.0, 1.0])
    quats = []
    for i in range(points.shape[0]):
        z = up if tilt_cap_deg is None else normals[i]/np.linalg.norm(normals[i])
        x = travel - z*float(np.dot(travel, z))
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0.0, 0.0]) - z*float(z[0])
        x = x/np.linalg.norm(x)
        quats.append(matrix_to_quat(np.column_stack([x, np.cross(z, x), z])))
    quats = np.asarray(quats)
    if tilt_cap_deg is not None:
        quats = cap_tool_tilt(quats, tilt_cap_deg)
    pos = np.array(points, dtype=np.float64, copy=True)
    for i in range(pos.shape[0]):
        pos[i] -= quat_to_matrix(quats[i]).dot(SENSOR_OFFSET)
    return ScanPath(pos, quats, [0], source="sensor-on-surface")


# ------------------------------------------------------------------------------
# 1. The envelope comes off the URDF, and stops at the arm
# ------------------------------------------------------------------------------
print("the tooling envelope")
env = tooling_envelope(source=TOOLING_URDF, root="probe_end",
                       rigid_root="tool0", sample_m=0.004)
names = set(env.link_names())
check("every link bolted to the wrist is in it",
      {"ec_probe", "sensor_body", "wrist_link"} <= names, sorted(names))
check("a revolute joint stops the walk -- the arm is not in the envelope",
      "upper_arm_link" not in names, sorted(names))
check("a link with no collision geometry is not in it",
      "tool0" not in names and "probe_end" not in names, sorted(names))

# probe_end is the tip of the barrel, so the barrel starts at the origin of the
# envelope's frame and extends away from the part.
probe = [l for l in env.links if l["name"] == "ec_probe"][0]
d = np.linalg.norm(probe["points"], axis=1)
check("the probe tip is AT the root frame", d.min() < 0.002, d.min())
check("the probe extends 68 mm from it",
      abs(d.max() - 0.0687) < 0.002, d.max())
check("the probe body is on the +z side of probe_end (away from the part)",
      probe["points"][:, 2].min() > -0.001, probe["points"][:, 2].min())
sensor = [l for l in env.links if l["name"] == "sensor_body"][0]
check("the sensor sits where the URDF says, 75 mm off the probe",
      abs(sensor["centre"][0] + 0.07515) < 0.001, sensor["centre"])
wrist = [l for l in env.links if l["name"] == "wrist_link"][0]
check("the wrist is above the tool, not below it",
      wrist["points"][:, 2].min() > 0.05, wrist["points"][:, 2].min())

try:
    broken = TOOLING_URDF.replace(
        '<geometry><cylinder radius="0.0071" length="0.068"/></geometry>',
        '<geometry><mesh filename="file:///nonexistent/probe.stl"/></geometry>')
    tooling_envelope(source=broken, root="probe_end", rigid_root="tool0")
    check("a missing collision mesh is an error, not a silent omission", False)
except IOError as e:
    check("a missing collision mesh is an error, not a silent omission",
          "does not exist" in str(e))

# ------------------------------------------------------------------------------
# 2. The distance is signed, and exact
# ------------------------------------------------------------------------------
print("signed distance to a surface")
pv, pt = plane_mesh(z=0.0)
plane = SurfaceDistance(pv, pt, name="plate")
d = plane.signed_distance(np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]]))
check("above the surface is positive, below is negative",
      abs(d[0] - 0.01) < 1e-9 and abs(d[1] + 0.01) < 1e-9, d)

# The case a vertex or centroid nearest-neighbour search gets wrong: a point
# over the middle of a large triangle. One triangle 200 mm on a side, a point
# 5 mm above its centre -- the nearest VERTEX is 58 mm away.
big = SurfaceDistance(np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                                [0.0, 0.2, 0.0]]),
                      np.array([[0, 1, 2]]), name="one big triangle")
d = big.signed_distance(np.array([[0.05, 0.05, 0.005]]))
check("a point over a triangle's interior measures to the FACE",
      abs(d[0] - 0.005) < 1e-9, d)
d = big.signed_distance(np.array([[-0.03, -0.04, 0.0]]))
check("a point off the edge measures to the nearest VERTEX",
      abs(d[0] - 0.05) < 1e-9, d)

cv, ct = cylinder_mesh(0.100, 0.30)
cyl = SurfaceDistance(cv, ct, name="r100 cylinder")
check("a fine mesh of a curved surface is accurate to its own faceting",
      abs(cyl.signed_distance(np.array([[0.0, 0.0, 0.150]]))[0] - 0.050) < 5e-4,
      cyl.signed_distance(np.array([[0.0, 0.0, 0.150]])))
dome_v, dome_t = cylinder_mesh(0.100, 0.30, phi_span=60.0)
check("normals are oriented outward without being told",
      SurfaceDistance(dome_v, dome_t).normal_disagreement < 0.02)
flipped = SurfaceDistance(dome_v, dome_t[:, ::-1], name="wound inside out")
check("a mesh wound inside out is corrected, not believed",
      flipped.signed_distance(np.array([[0.0, 0.0, 0.150]]))[0] > 0.0,
      flipped.signed_distance(np.array([[0.0, 0.0, 0.150]])))
check("and a surface that wraps past vertical is not mistaken for one",
      0.05 < cyl.normal_disagreement < 0.2, cyl.normal_disagreement)

# ------------------------------------------------------------------------------
# 3. The fault: a laterally offset sensor drives the tool into a rising surface
# ------------------------------------------------------------------------------
# The condition, stated: with the tool held vertical the sensor sits at S and
# probe_end at S + (75.15, 2.2, 34.28) mm in the tool frame. probe_end is safe
# only where the surface 75 mm laterally is less than 34 mm HIGHER. So a
# vertically held raster tolerates a mean surface slope of
# atan(34.28/75.15) = 24.5 degrees over that span and no more -- and the barrel
# takes 7.1 mm of that back.
print("a rising surface, tool held vertical")
RAD = 0.100
cv, ct = cylinder_mesh(RAD, 0.30, axis_z=0.0)
cyl = SurfaceDistance(cv, ct, name="r100 cylinder")
# A row across the flank: travel along x (the axis), stepping up the flank in y,
# so the sensor's 75 mm lateral offset runs ACROSS the curvature.
phi = np.radians(np.linspace(-80.0, -20.0, 40))
pts = np.column_stack([np.zeros(phi.size), RAD*np.sin(phi), RAD*np.cos(phi)])
nrm = pts/np.linalg.norm(pts, axis=1)[:, None]
flat = sensor_on_surface_path(pts, nrm, np.array([0.0, 1.0, 0.0]))
rep = path_clearance(flat, cyl, envelope=env)
worst = min(rep["per_link"], key=lambda k: rep["per_link"][k])
print("    vertical: worst %+.1f mm on %s; sensor placed on the surface to "
      "within %.2f mm"
      % (rep["min_m"]*1e3, worst,
         abs(cyl.signed_distance(
             np.array([p - quat_to_matrix(q).dot(-SENSOR_OFFSET)
                       for p, q in zip(flat.positions, flat.quats)]))).max()*1e3))
check("holding the tool vertical over a rising surface is a STRIKE",
      rep["min_m"] < 0.0, rep["min_m"])
check("the eddy-current probe is one of the things inside the part",
      rep["per_link"]["ec_probe"] < 0.0, rep["per_link"]["ec_probe"])
optical = np.array([p - quat_to_matrix(q).dot(-SENSOR_OFFSET)
                    for p, q in zip(flat.positions, flat.quats)])
check("while the sensor the stage cares about is placed exactly right -- which "
      "is why nothing noticed",
      np.abs(cyl.signed_distance(optical)).max() < 1e-4,
      np.abs(cyl.signed_distance(optical)).max())

# 4. Following the surface, capped, clears it.
print("the same row, tilt following the surface")
best = None
for cap in (0.0, 15.0, 30.0, 45.0, 60.0):
    path = sensor_on_surface_path(pts, nrm, np.array([0.0, 1.0, 0.0]),
                                  tilt_cap_deg=cap)
    rep_c = path_clearance(path, cyl, envelope=env)
    print("    cap %4.1f deg: worst %+7.1f mm (%s), tool tilt <= %.1f deg"
          % (cap, rep_c["min_m"]*1e3,
             min(rep_c["per_link"], key=lambda k: rep_c["per_link"][k]),
             tool_tilt_deg(path.quats).max()))
    if best is None or rep_c["min_m"] > best[1]:
        best = (cap, rep_c["min_m"])
check("a tilt cap large enough to follow the surface clears it",
      best[1] > 0.005, best)
check("and cap 0 is the same as holding it vertical",
      abs(path_clearance(
          sensor_on_surface_path(pts, nrm, np.array([0.0, 1.0, 0.0]),
                                 tilt_cap_deg=0.0),
          cyl, envelope=env)["min_m"] - rep["min_m"]) < 1e-6)

# ------------------------------------------------------------------------------
# 5. A flat plate is untouched -- the check must not cost the acceptance scan
# ------------------------------------------------------------------------------
print("a flat plate")
pv, pt = plane_mesh(z=0.0, half=0.30, n=40)
plate = SurfaceDistance(pv, pt, name="plate")
xs = np.linspace(-0.10, 0.10, 40)
pts = np.column_stack([xs, np.zeros(40), np.zeros(40)])
nrm = np.tile([0.0, 0.0, 1.0], (40, 1))
flat = sensor_on_surface_path(pts, nrm, np.array([1.0, 0.0, 0.0]))
rep = path_clearance(flat, plate, envelope=env)
print("    worst %+.1f mm on %s"
      % (rep["min_m"]*1e3, min(rep["per_link"],
                               key=lambda k: rep["per_link"][k])))
check("over a plate the same geometry leaves the probe its full standoff -- "
      "which is why the cal plate scanned fine",
      abs(rep["min_m"] - (0.03428 - 0.002)) < 0.002, rep["min_m"])
kept, erep = enforce_path_clearance(flat, plate, limit=0.005, envelope=env,
                                    report=False)
check("nothing is dropped, so a flat-plate scan is unchanged",
      erep["dropped"] == 0 and len(kept) == len(flat), erep["dropped"])

# ------------------------------------------------------------------------------
# 6. Dropping, refusing, and declared contact
# ------------------------------------------------------------------------------
print("what the check does about it")
phi = np.radians(np.linspace(-80.0, 80.0, 80))
pts = np.column_stack([np.zeros(phi.size), RAD*np.sin(phi), RAD*np.cos(phi)])
nrm = pts/np.linalg.norm(pts, axis=1)[:, None]
mixed = sensor_on_surface_path(pts, nrm, np.array([0.0, 1.0, 0.0]))
ns["_crop_path_by_mask"] = ns["_crop_path_by_mask"] if "_crop_path_by_mask" in ns else None


def _crop(path, keep, what="path", min_points=4, why="", detail=""):
    keep = np.asarray(keep, dtype=bool)
    positions, quats, starts = [], [], []
    for a, b in path.segments:
        run = []
        for i in list(range(a, b)) + [None]:
            if i is not None and keep[i]:
                run.append(i)
                continue
            if len(run) >= min_points:
                starts.append(len(positions))
                positions.extend(path.positions[j] for j in run)
                quats.extend(path.quats[j] for j in run)
            run = []
    if not positions:
        raise RuntimeError("everything dropped")
    return ScanPath(positions, quats, starts, source=path.source)


ns["_crop_path_by_mask"] = _crop
kept, erep = enforce_path_clearance(mixed, cyl, limit=0.005, envelope=env,
                                    report=False)
print("    %d of %d poses dropped, %.1f%% of the path given up"
      % (erep["dropped"], len(mixed), erep["coverage_lost"]*100.0))
check("the poses that strike are dropped and the rest are kept",
      0 < erep["dropped"] < len(mixed), erep["dropped"])
check("what it cost is reported, not hidden",
      0.0 < erep["coverage_lost"] < 1.0, erep["coverage_lost"])
kept_rep = path_clearance(kept, cyl, envelope=env)
check("and what is left really does clear the limit",
      kept_rep["min_m"] >= 0.005, kept_rep["min_m"])

plain = path_clearance(mixed, cyl, envelope=env)
allowed = path_clearance(mixed, cyl, envelope=env, allow_links=("ec_probe",))
check("a declared contact link is excluded from the verdict",
      allowed["min_m"] > plain["min_m"],
      (plain["min_m"], allowed["min_m"]))
check("but it is still measured and reported",
      allowed["per_link"]["ec_probe"] == plain["per_link"]["ec_probe"],
      allowed["per_link"]["ec_probe"])
check("and the worst thing overall is still available",
      allowed["min_any_m"] == plain["min_any_m"])

try:
    inside = sensor_on_surface_path(
        pts[:20], nrm[:20], np.array([0.0, 1.0, 0.0]))
    enforce_path_clearance(inside, cyl, limit=0.200, envelope=env, report=False)
    check("a surface nothing can clear is refused outright", False)
except RuntimeError as e:
    check("a surface nothing can clear is refused outright",
          "cannot be scanned" in str(e), str(e)[:80])

# ------------------------------------------------------------------------------
# 7. The tilt cap itself
# ------------------------------------------------------------------------------
print("cap_tool_tilt")
rng = np.random.default_rng(7)
axes = rng.normal(size=(200, 3))
axes[:, 2] = np.abs(axes[:, 2]) + 0.05
axes /= np.linalg.norm(axes, axis=1)[:, None]
quats = ns["locked_quats_from_axes"](axes)
for cap in (0.0, 10.0, 25.0, 40.0):
    capped = cap_tool_tilt(quats, cap)
    tilt = tool_tilt_deg(capped)
    check("cap %.0f deg is respected" % cap, tilt.max() <= cap + 1e-6,
          tilt.max())
before = tool_tilt_deg(quats)
capped = cap_tool_tilt(quats, 25.0)
after = tool_tilt_deg(capped)
check("a pose already inside the cap is untouched",
      np.allclose(after[before <= 25.0], before[before <= 25.0], atol=1e-9))
check("capping does not turn the tool about its own axis",
      np.abs(tool_twist_deg(capped) - tool_twist_deg(quats)).max() < 1.0,
      np.abs(tool_twist_deg(capped) - tool_twist_deg(quats)).max())
check("a cap above everything is a no-op",
      np.allclose(cap_tool_tilt(quats, 179.0), ns["_unit_quats"](quats)))

# ------------------------------------------------------------------------------
# 8. The lead-in nobody plans
# ------------------------------------------------------------------------------
print("lead_in_clearance")
lead = ns["lead_in_clearance"]
# Straight across the top of the cylinder at the height of its crest: the
# tooling has to go through it. This is the move computeCartesianPath prepends
# when a row is entered from the wrong side, reporting "100% of N waypoints".
crest = RAD
ok, lrep = lead(((0.0, -0.150, crest - 0.030), (0.0, 0.0, 0.0, 1.0)),
                ((0.0, 0.150, crest - 0.030), (0.0, 0.0, 0.0, 1.0)),
                cyl, envelope=env)
check("a lead-in that crosses the part is refused", not ok, lrep["min_m"])
ok, lrep = lead(((0.0, -0.150, crest + 0.250), (0.0, 0.0, 0.0, 1.0)),
                ((0.0, 0.150, crest + 0.250), (0.0, 0.0, 0.0, 1.0)),
                cyl, envelope=env)
check("one that goes over the top is allowed", ok, lrep["min_m"])
# And it must interpolate the ORIENTATION too: two poses 90 degrees apart sweep
# the tooling through everything in between.
ok, lrep = lead(((0.0, 0.0, crest + 0.120), (0.0, 0.0, 0.0, 1.0)),
                ((0.0, 0.0, crest + 0.120), tuple(cap_tool_tilt(
                    [(0.0, 0.0, 0.0, 1.0)], 85.0)[0])),
                cyl, envelope=env)
print("    turning the tool 85 deg in place at 120 mm: worst %+.1f mm"
      % (lrep["min_m"]*1e3,))
check("a rotation in place is checked through its whole arc",
      len(lrep["clearance"]) >= 24, len(lrep["clearance"]))

# ------------------------------------------------------------------------------
# 8b. Sampling a long path, and what it is allowed to skip
# ------------------------------------------------------------------------------
# The eddy-current toolpath at this cell's 0.30 mm line spacing is ~180 000
# poses; measuring every one takes hours. What makes sampling admissible is not
# the stride, it is the certificate: clearance is 1-Lipschitz in tooling motion,
# so two measured poses bound every pose between them. These check the bound is
# real, that the bound is conservative, and that a violation cannot slip
# through a window.
print("\nsampling a long path")

pv, pt = plane_mesh(z=0.0, half=0.60, n=60)
plate = SurfaceDistance(pv, pt, name="plate")


def long_path(n=1200, dip_at=None, dip_m=0.0, dip_width=40):
    """A straight row 1 mm apart over a plate, optionally with a dip in it.

    The dip drives the WHOLE TOOL down for a short stretch -- the fault this
    gate exists to catch. It is made narrow on purpose, 40 poses of 1200,
    because the question is whether sampling steps over it.
    """
    xs = np.linspace(-0.25, 0.25, n)
    pts = np.column_stack([xs, np.zeros(n), np.zeros(n)])
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    path = sensor_on_surface_path(pts, nrm, np.array([1.0, 0.0, 0.0]))
    if dip_at is None:
        return path
    pos = np.array(path.positions, copy=True)
    lo = max(0, dip_at - dip_width//2)
    hi = min(n, dip_at + dip_width//2)
    pos[lo:hi, 2] -= dip_m
    return ScanPath(pos, path.quats, [0])


clear = long_path()
env_reach = float(np.linalg.norm(env.points, axis=1).max())
check("the envelope knows its reach from the path frame, not just its radius",
      abs(env.reach - env_reach) < 1e-12, (env.reach, env_reach))

travel = path_step_travel(clear, env)
# The bound has to hold for the actual points, not on average. Move the whole
# envelope by each pair of poses and measure the largest displacement.
worst_ratio = 0.0
for i in range(0, len(clear) - 1, 97):
    R1 = quat_to_matrix(clear.quats[i])
    R2 = quat_to_matrix(clear.quats[i+1])
    moved = np.linalg.norm(
        (env.points.dot(R2.T) + clear.positions[i+1])
        - (env.points.dot(R1.T) + clear.positions[i]), axis=1).max()
    worst_ratio = max(worst_ratio, moved/max(travel[i], 1e-12))
check("path_step_travel is an UPPER bound on how far the tooling moves",
      worst_ratio <= 1.0 + 1e-9, worst_ratio)

full = path_clearance(clear, plate, envelope=env)
samp = sampled_path_clearance(clear, plate, limit=0.005, envelope=env,
                              report=False)
check("on a clear path the sampler measures a small fraction of the poses",
      samp["checked"].size < 0.25*len(clear),
      "%d of %d" % (samp["checked"].size, len(clear)))
check("and it agrees with the full check on the verdict",
      (samp["clearance"] >= 0.005).all() == (full["clearance"] >= 0.005).all())
check("and never reads a pose as clearer than measuring it does",
      (samp["clearance"] <= full["clearance"] + 1e-9).all(),
      float((samp["clearance"] - full["clearance"]).max()))
check("nothing is left uncertified",
      samp["exhausted"] == 0, samp["exhausted"])

# The dip: 40 poses of 1200, deep enough to break the limit. A stride of ten
# over a 1 mm pitch would land inside it by luck; the certificate must not.
dipped = long_path(dip_at=700, dip_m=0.030, dip_width=40)
full_d = path_clearance(dipped, plate, envelope=env)
samp_d = sampled_path_clearance(dipped, plate, limit=0.005, envelope=env,
                                report=False)
bad_full = np.nonzero(full_d["clearance"] < 0.005)[0]
bad_samp = np.nonzero(samp_d["clearance"] < 0.005)[0]
print("    dip of 40 poses in 1200: full check refuses %d poses, the sampler "
      "refuses %d and measured %d" % (bad_full.size, bad_samp.size,
                                      samp_d["checked"].size))
check("a narrow violation is found by the sampler",
      bad_samp.size > 0 and bad_full.size > 0)
check("and every pose the full check refuses, the sampler refuses too",
      set(bad_full.tolist()).issubset(set(bad_samp.tolist())),
      "%d missed" % (len(set(bad_full.tolist()) - set(bad_samp.tolist())),))
check("the sampler does not invent a violation far from the dip",
      bad_samp.size and bad_samp.max() - bad_samp.min() < 200,
      (int(bad_samp.min()), int(bad_samp.max())) if bad_samp.size else "none")

# ------------------------------------------------------------------------------
# 8c. Blocking, candidate count and the opening cap
# ------------------------------------------------------------------------------
print("\nblocking and the opening cap")
a_one = path_clearance(clear, plate, envelope=env, block=0)
a_many = path_clearance(clear, plate, envelope=env, block=37)
check("measuring in blocks gives the same answer as measuring in one go",
      np.allclose(a_one["clearance"], a_many["clearance"]),
      float(np.abs(a_one["clearance"] - a_many["clearance"]).max()))

brute = SurfaceDistance(pv, pt, name="brute")
brute._tree = None                       # forces the chunked brute-force path
probe = clear.positions[:40] + np.array([0.0, 0.0, 0.003])
check("the candidate count chosen from the mesh does not change the distance",
      np.allclose(plate.signed_distance(probe),
                  brute.signed_distance(probe, k=12), atol=1e-9),
      float(np.abs(plate.signed_distance(probe)
                   - brute.signed_distance(probe, k=12)).max()))

# What the split cap is for: a link ALLOWED to touch is deep in the part, and
# without the split it becomes the running minimum every other link is compared
# against, so the links the gate is actually about are never measured -- they
# are gated on a bounding-sphere lower bound tens of millimetres loose.
# probe_end 2 mm BELOW the plate: the eddy-current tip (which IS probe_end) is
# in the part by 2 mm, exactly as a contact scan intends, while the sensor body
# 34 mm up the tool and 75 mm to the side is a comfortable 32 mm clear.
deep = long_path(n=200)
_dp = np.array(deep.positions, copy=True)
_dp[:, 2] = -0.002
deep = ScanPath(_dp, deep.quats, [0])
with_contact = path_clearance(deep, plate, envelope=env,
                              allow_links=("ec_probe", "probe_holder"))
check("a deep allowed contact does not drag the gated links down with it",
      with_contact["min_m"] > 0.0, with_contact["min_m"])
check("while the contact itself is still measured and reported",
      with_contact["per_link"]["ec_probe"] < 0.0,
      with_contact["per_link"]["ec_probe"])

# ------------------------------------------------------------------------------
# 8d. The distance kernel: every shortcut is a SOUND bound, and none of the
#     performance knobs is allowed to change an answer
# ------------------------------------------------------------------------------
# What made this gate affordable is that it stops early -- it skips points the
# caller has said it does not care about, it runs the arithmetic in cache-sized
# slabs over several threads, and it batches thousands of poses into one query.
# Every one of those is a correctness claim: the shortcuts may report LESS
# clearance than the truth and must never report more, and the knobs (threads,
# slab size, block size, batch size) must not be visible in the answer at all.
print("\nthe distance kernel")

closest_point_on_triangles = ns["closest_point_on_triangles"]
cv, ct = cylinder_mesh(0.050, 0.30)
kern = SurfaceDistance(cv, ct, name="kernel")
_rng = np.random.default_rng(11)
_lo, _hi = cv.min(axis=0) - 0.05, cv.max(axis=0) + 0.05
probe_pts = _rng.uniform(_lo, _hi, size=(4000, 3))
exact_all = kern.signed_distance(probe_pts)

# Brute force, no tree and no candidate set at all: the definition.
_brute = np.empty(probe_pts.shape[0])
for _i in range(probe_pts.shape[0]):
    _q = closest_point_on_triangles(probe_pts[_i][None, :], kern.a, kern.b, kern.c)
    _d = np.linalg.norm(probe_pts[_i][None, :] - _q, axis=1)
    _j = int(np.argmin(_d))
    _side = float(np.dot(probe_pts[_i] - _q[_j], kern.normals[_j]))
    _brute[_i] = -_d[_j] if _side < 0.0 else _d[_j]
check("the kernel agrees with brute force over every triangle",
      np.allclose(exact_all, _brute, atol=1e-9),
      float(np.abs(exact_all - _brute).max()))

for _w in (1, 2, 8):
    got = kern.signed_distance(probe_pts, workers=_w)
    if not np.array_equal(got, exact_all):
        break
check("threads do not change the answer",
      np.array_equal(kern.signed_distance(probe_pts, workers=1),
                     kern.signed_distance(probe_pts, workers=8)))

_keep = ns["SURFACE_KERNEL_BYTES"]
try:
    ns["SURFACE_KERNEL_BYTES"] = 4096          # one tiny slab per handful
    tiny = kern.signed_distance(probe_pts)
finally:
    ns["SURFACE_KERNEL_BYTES"] = _keep
check("the cache-slab size does not change the answer",
      np.array_equal(tiny, exact_all),
      float(np.abs(tiny - exact_all).max()))

check("the outer block size does not change the answer",
      np.array_equal(kern.signed_distance(probe_pts, block=137), exact_all))

# exact_within is a RELAXATION and has to be a sound one: exact below the line
# the caller drew, and never an over-report above it.
for _far in (0.005, 0.020, 0.080):
    rel = kern.signed_distance(probe_pts, exact_within=_far)
    below = (exact_all < _far) & (exact_all > 0.0)
    ok_exact = np.allclose(rel[below], exact_all[below], atol=1e-12)
    outside = exact_all > 0.0
    ok_sound = bool((rel[outside] <= exact_all[outside] + 1e-9).all())
    check("exact_within=%.3f is exact below the line" % _far, ok_exact,
          float(np.abs(rel[below] - exact_all[below]).max()) if below.any() else "none")
    check("exact_within=%.3f never reports more clearance than the truth" % _far,
          ok_sound, float((rel[outside] - exact_all[outside]).max()))

# exact_each is the same thing per point -- that is what makes batching
# thousands of poses into one query lossless rather than a compromise. It is
# held to exactly the contract exact_within is held to, point by point: exact
# below that point's own line, never an over-report above it. (It is allowed to
# be MORE exact than asking one at a time, and is: points are grouped by their
# threshold and a group is asked the loosest question in it.)
_each = _rng.choice([0.004, 0.015, 0.060], size=probe_pts.shape[0])
batched = kern.signed_distance(probe_pts, exact_each=_each)
_out = exact_all > 0.0
check("exact_each never reports more clearance than the truth",
      bool((batched[_out] <= exact_all[_out] + 1e-9).all()),
      float((batched[_out] - exact_all[_out]).max()))
_below = _out & (exact_all < _each)
check("exact_each is exact below each point's own line",
      np.allclose(batched[_below], exact_all[_below], atol=1e-12),
      float(np.abs(batched[_below] - exact_all[_below]).max()))

# The sign a settled bound asserts is NOT established by the query -- "no
# triangle within X" is as true of a point deep inside the part as of one far
# above it -- so a caller may only relax to a distance at which it can vouch for
# the sign itself. path_clearance() vouches for it by sampling a CLOSED hull:
# a link cannot be inside the part without its hull crossing the surface, and
# the crossing carries sample points that are near the surface, are therefore
# never settled, and are what the link's reported MINIMUM comes from. 8e drives
# whole links into a plate and a cylinder and holds the check to that.
_inside = exact_all < 0.0
check("a settled bound is about distance, not side -- which is why nothing may "
      "relax below zero", bool(_inside.any()))

# A threshold at or below zero is a demand for exactness, not a relaxation:
# a contact scan hands this the running minimum, which is negative.
for _neg in (-0.010, 0.0):
    check("exact_within=%+.3f measures rather than settling" % _neg,
          np.array_equal(kern.signed_distance(probe_pts, exact_within=_neg),
                         exact_all))
check("and the same per point",
      np.array_equal(kern.signed_distance(
          probe_pts, exact_each=np.full(probe_pts.shape[0], -0.01)), exact_all))

# ------------------------------------------------------------------------------
# 8e. The whole check never reports more clearance than a full evaluation
# ------------------------------------------------------------------------------
# The three levels of bounding sphere, the `interest_m` cap and the running
# minimum all EXIST to avoid measuring things, and each one is a place where an
# over-report could hide. This is the end-to-end statement: whatever the check
# says a pose has, evaluating every sample point of every link says at least
# that much.
#
# The case that used to break it: a link whose bounding-sphere bound sits just
# ABOVE interest_m is never opened, so it used to contribute nothing at all,
# while a link measured at 200 mm set the pose's number. That is only visible
# when everything is clear by more than interest_m -- which is exactly a lead-in
# or a transit -- and sampled_path_clearance() turns the number into a certified
# window of that many millimetres, so it is not a cosmetic error.
print("\nthe check never over-reports")


def brute_clearance(path, surface, envelope, pad, allow=()):
    """Every sample point of every link at every pose. The definition."""
    out = np.full(len(path), np.inf)
    for i in range(len(path)):
        R = quat_to_matrix(path.quats[i])
        for link in envelope.links:
            if link["name"] in allow:
                continue
            w = link["points"].dot(R.T) + path.positions[i]
            out[i] = min(out[i], float(surface.signed_distance(w).min()) - pad)
    return out


PAD = ns["TOOLING_CLEARANCE_PAD_M"]
for _name, _path, _mesh, _allow in (
        ("on a plate", long_path(n=24), plate, ()),
        ("over a cylinder", None, cyl, ()),
        ("well clear of everything", None, plate, ()),
        ("in contact", None, plate, ("ec_probe", "probe_holder")),
        ("driven into the part", None, plate, ()),
        ("far under the sheet", None, plate, ())):
    if _name == "over a cylinder":
        _xs = np.linspace(-0.06, 0.06, 24)
        _pts = np.column_stack([np.zeros(24), 0.050*np.sin(_xs/0.050),
                                0.050*np.cos(_xs/0.050)])
        _nrm = _pts/np.linalg.norm(_pts, axis=1)[:, None]
        _path = sensor_on_surface_path(_pts, _nrm, np.array([1.0, 0.0, 0.0]),
                                       tilt_cap_deg=45.0)
    elif _name == "well clear of everything":
        # probe_end 60 mm up: EVERY link is beyond interest_m, which is the
        # configuration the over-report used to live in.
        _base = long_path(n=24)
        _pos = np.array(_base.positions, copy=True)
        _pos[:, 2] += 0.060
        _path = ScanPath(_pos, _base.quats, [0])
    elif _name == "in contact":
        _base = long_path(n=24)
        _pos = np.array(_base.positions, copy=True)
        _pos[:, 2] -= 0.036          # the probe tip pressed into the plate
        _path = ScanPath(_pos, _base.quats, [0])
    elif _name == "far under the sheet":
        # 150 mm UNDER the plate and inside its footprint. The early-out's
        # bound is about DISTANCE -- "no triangle within X" is as true of a
        # point 150 mm below a sheet as of one 150 mm above it -- so without
        # the outward-side condition in _settle_far this comes back as a
        # comfortable +50 mm on every link, which is the exact shape of the
        # fault this whole module exists to catch.
        _base = long_path(n=24)
        _pos = np.array(_base.positions, copy=True)
        _pos[:, 2] -= 0.150
        _path = ScanPath(_pos, _base.quats, [0])
    elif _name == "driven into the part":
        # The whole sensor body 25 mm below the plate, not just grazing it.
        # An interior sample point is exactly what a settled bound mis-signs,
        # so this is the case that says the link's MINIMUM is still caught.
        _base = long_path(n=24)
        _pos = np.array(_base.positions, copy=True)
        _pos[:, 2] -= 0.065
        _path = ScanPath(_pos, _base.quats, [0])
    truth = brute_clearance(_path, _mesh, env, PAD, allow=_allow)
    got = path_clearance(_path, _mesh, envelope=env, allow_links=_allow)
    over = got["clearance"] - truth
    check("%s: the gate never reads clearer than the truth" % _name,
          bool((over <= 1e-9).all()),
          "worst over-report %+.4f mm at pose %d"
          % (1e3*over.max(), int(np.argmax(over))))
    if _name in ("driven into the part", "far under the sheet"):
        check("   and %s is reported negative" % (_name,),
              bool((got["clearance"] < 0.0).all()),
              float(got["clearance"].max()))

# ------------------------------------------------------------------------------
# 8f. A collision the FIRST PASS steps clean over
# ------------------------------------------------------------------------------
# The sampled gate now starts deliberately coarse -- TOOLING_GATE_COARSE_M of
# tooling travel between the first measurements, 25 times the old spacing --
# because what a measurement buys is not a pose but the certified window around
# it. That is only admissible because of the Lipschitz certificate, and this is
# the test of it: a collision placed BETWEEN two first-pass samples, which no
# amount of "sample every nth pose and hope" would find, and which the
# certificate has to drive the sampler down into.
print("\na collision between two first-pass samples")


def straight_path(n, span, dip_at=None, dip_m=0.0, dip_width=30):
    xs = np.linspace(-span/2.0, span/2.0, n)
    pts = np.column_stack([xs, np.zeros(n), np.zeros(n)])
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    out = sensor_on_surface_path(pts, nrm, np.array([1.0, 0.0, 0.0]))
    if dip_at is None:
        return out
    pos = np.array(out.positions, copy=True)
    lo = max(0, dip_at - dip_width//2)
    hi = min(n, dip_at + dip_width//2)
    pos[lo:hi, 2] -= dip_m
    return ScanPath(pos, out.quats, [0])


DIP_AT, DIP_W = 3700, 30
long_clear = straight_path(6000, 1.0)
long_dip = straight_path(6000, 1.0, dip_at=DIP_AT, dip_m=0.040, dip_width=DIP_W)
coarse_m = ns["TOOLING_GATE_COARSE_M"]

# What the first pass actually measures, worked out the way the sampler does.
_cum = np.concatenate(([0.0], np.cumsum(path_step_travel(long_dip, env))))
_marks = np.arange(_cum[0], _cum[-1], coarse_m)[1:]
first_pass = set([0, len(long_dip) - 1]
                 + [int(i) for i in np.searchsorted(_cum, _marks)])
dip_poses = set(range(DIP_AT - DIP_W//2, DIP_AT + DIP_W//2))
check("the first pass genuinely steps over the collision",
      not (first_pass & dip_poses),
      "first pass: %s, dip: %d..%d"
      % (sorted(first_pass), min(dip_poses), max(dip_poses)))

full_long = path_clearance(long_dip, plate, envelope=env)
samp_long = sampled_path_clearance(long_dip, plate, limit=0.005, envelope=env,
                                   report=False)
bad_full = set(np.nonzero(full_long["clearance"] < 0.005)[0].tolist())
bad_samp = set(np.nonzero(samp_long["clearance"] < 0.005)[0].tolist())
print("    %d poses, first pass measures %d of them, the sampler ends up "
      "measuring %d; the full check refuses %d poses and the sampler refuses %d"
      % (len(long_dip), len(first_pass), samp_long["checked"].size,
         len(bad_full), len(bad_samp)))
check("the sampler finds a collision no first-pass sample touches",
      bool(bad_samp) and bad_full <= bad_samp,
      "%d refused by the full check and missed by the sampler"
      % (len(bad_full - bad_samp),))
check("and it is not just refusing everything",
      len(bad_samp) < 0.02*len(long_dip), len(bad_samp))
check("nothing is left uncertified on the long path",
      samp_long["exhausted"] == 0, samp_long["exhausted"])
check("the certified bound is never above what measuring gives",
      bool((samp_long["clearance"] <= full_long["clearance"] + 1e-9).all()),
      float((samp_long["clearance"] - full_long["clearance"]).max()))
samp_ok = sampled_path_clearance(long_clear, plate, limit=0.005, envelope=env,
                                 report=False)
check("a clear path of 6000 poses is certified from a few hundred measurements",
      samp_ok["checked"].size < 0.10*len(long_clear),
      "%d of %d" % (samp_ok["checked"].size, len(long_clear)))

# ------------------------------------------------------------------------------
# 9. Nothing here borrows a name another .dpi keeps private
# ------------------------------------------------------------------------------
# EVERY check above runs with all three .dpi files exec'd into ONE namespace, so
# a private helper in noether_path.dpi is visible from tooling_clearance.dpi --
# and at runtime it is NOT, because a .dpi is compiled as a function body and
# only what it declares `global` escapes it. A borrowed private helper therefore
# passes every check above and raises "NameError: name '_unit_quats' is not
# defined" the first time a real sweep row calls it, taking the whole raster
# down with it. So this reads the file the way dgpy will: what may it
# legitimately reach for?
print("\ncross-module names")


def _declared_globals(name):
    out = set()
    for line in open(os.path.join(HERE, name)):
        if line.startswith("global "):
            out.add(line.split(None, 1)[1].strip())
    return out


def _module_names(code, out=None):
    out = set() if out is None else out
    out.update(code.co_names)
    for const in code.co_consts:
        if hasattr(const, "co_names"):
            _module_names(const, out)
    return out


THIS = "tooling_clearance.dpi"
_code = compile(_load(THIS), THIS, "exec")
_own = set(_code.co_names)          # everything it defines or assigns at top level
_allowed = (_own | set(dir(__builtins__ if isinstance(__builtins__, dict)
                          else __builtins__))
            | set(__builtins__.keys() if isinstance(__builtins__, dict) else dir(__builtins__))
            | _declared_globals("toolpath_segments.dpi")
            | _declared_globals("noether_path.dpi")
            | _declared_globals("mesh_reconstruction.dpi"))
# Attribute names appear in co_names too (np.asarray -> "asarray"), so only
# names that look like a module-level helper and are not resolvable anywhere
# matter. A leaked private helper is exactly "_foo" defined in another .dpi.
_private_elsewhere = set()
for other in ("toolpath_segments.dpi", "noether_path.dpi", "mesh_reconstruction.dpi"):
    src = _load(other)
    for line in src.splitlines():
        if line.startswith("def _") or line.startswith("class _"):
            _private_elsewhere.add(line.split()[1].split("(")[0].split(":")[0])
_private_elsewhere -= _own
_borrowed = sorted(_module_names(_code) & _private_elsewhere)
check("no private helper is borrowed from another .dpi", not _borrowed,
      "borrowed: %s" % (", ".join(_borrowed),) if _borrowed
      else "%d private names exist in the neighbours and none is used" % len(_private_elsewhere))

# ==============================================================================
# THE PROGRESS LINE REACHES SOMEWHERE
# ==============================================================================
# The gate can work for tens of minutes, and a silent process is
# indistinguishable from a hang -- so it prints. But a bare print() from a dgpy
# acquisition thread can be SWALLOWED, which restores the silence while looking
# fixed. So when progress.dpi is loaded the line must go through progress_say(),
# which
# also reaches the capture session's own log; when it is not, the plain line
# must still appear; and a progress sink that throws must never take down the
# gate it is reporting on.
print("\n9. the progress line reaches somewhere, thread or not")

import contextlib as _ctx

def _tick_output(with_progress, sink=None):
    ns = {"__name__": "prog_check", "np": np}

    def _ex(name, drop=()):
        src = _load(name)
        src = "\n".join(("#" + l) if l.startswith("global ") else l
                         for l in src.splitlines())
        for d in drop:
            src = src.replace(d, "pass  # " + d.replace("\n", " "))
        exec(compile(src, name, "exec"), ns)

    if with_progress:
        _ex("progress.dpi")
    _ex("noether_path.dpi",
        drop=("from geometry_msgs.msg import PoseArray, Pose",))
    _ex("tooling_clearance.dpi", drop=("import rclpy",))
    if with_progress and sink is not None:
        ns["PROGRESS_SINKS"].append(sink)
    buf = io.StringIO()
    with _ctx.redirect_stdout(buf):
        tick = ns["_gate_progress"](100, every=0.0001)
        time.sleep(0.01)
        tick(50, force=True)
    return buf.getvalue()

_plain = _tick_output(False)
check("with no progress.dpi the plain line is still printed",
      "gate: 50/100 poses" in _plain, _plain.strip()[:70])
check("and it carries no timestamp, so nothing pretends to be what it is not",
      not re.search(r"\[\d\d:\d\d:\d\d\]", _plain), _plain.strip()[:70])

_sunk = []
_with = _tick_output(True, _sunk.append)
check("with progress.dpi the line is timestamped",
      bool(re.search(r"\[\d\d:\d\d:\d\d\]", _with)), _with.strip()[:70])
check("and it reaches the sink, so it survives into the session log",
      any("gate: 50/100 poses" in t for t in _sunk),
      (_sunk[0][:70] if _sunk else "nothing reached the sink"))
check("it is not printed twice", _with.count("gate: 50/100 poses") == 1,
      "%d copies" % _with.count("gate: 50/100 poses"))


def _explode(_text):
    raise RuntimeError("this sink is broken")


_boom = _tick_output(True, _explode)
check("a sink that throws does not take the gate down with it",
      "gate: 50/100 poses" in _boom, _boom.strip()[:70])


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================
# Every check above runs at IMPORT time, which is what lets one pytest test stand
# for the lot -- and it is also exactly how a suite prints "ok" down the page
# while the code under test never runs. This suite used to install its own
# geometry_msgs stub over everyone else's (see ros_msg_stubs.py); the risk on the
# way back out is that it now RECEIVES a stub it did not write, so the fact that
# the subject still runs and still discriminates has to be a test of its own.

# ==============================================================================
# THE SIGN OF A HEIGHT FIELD COMES FROM THE FIELD
# ==============================================================================
# The 2026-10-09 planning mesh was a reconstructed plate, 99.5% of its normals
# pointing up, and the half-percent that did not inverted the clearance for
# every point nearest to them. The Keyence frame, 141 mm ABOVE the part, was
# reported 141.8 mm inside it.

print("")
print("height-field sign")

# A 100 x 100 mm patch on a 11 x 11 grid: 200 triangles, the real thing's
# shape. One of them is wound the wrong way -- 0.5%, the same share the
# 2026-10-09 mesh had left after repair.
_gx, _gy = np.meshgrid(np.linspace(0.0, 0.1, 11), np.linspace(0.0, 0.1, 11))
_hv = np.c_[_gx.ravel(), _gy.ravel(), np.zeros(_gx.size)]
_tri = []
for _r in range(10):
    for _c in range(10):
        _i = _r*11 + _c
        _tri.append([_i, _i+1, _i+12])
        _tri.append([_i, _i+12, _i+11])
_good = np.array(_tri)
_bad = _good.copy()
_BADFACE = 2*(5*10 + 5)                      # one triangle near the middle
_bad[_BADFACE] = _bad[_BADFACE][::-1]
_clean = SurfaceDistance(_hv, _good, name="patch")
_broken = SurfaceDistance(_hv, _bad, name="patch with one face reversed")

check("a clean patch is a height field with nothing to correct",
      _clean.height_field and _clean.sign_faces_corrected == 0,
      (_clean.height_field, _clean.sign_faces_corrected))
check("1 reversed face of 200 is still a height field, and is corrected",
      _broken.height_field and _broken.sign_faces_corrected == 1,
      (_broken.height_field, _broken.sign_faces_corrected))
check("...and the measured disagreement is still reported as it was (%.3f)"
      % (_broken.normal_disagreement,),
      abs(_broken.normal_disagreement - 1.0/200.0) < 1e-9,
      _broken.normal_disagreement)

# A POINT 141 mm ABOVE THE PATCH, over the broken half. This is the one that
# mattered: before the repair it read -0.141.
_above = np.array([[0.0525, 0.0525, 0.141]])
_d_clean = float(_clean.signed_distance(_above)[0])
_d_broken = float(_broken.signed_distance(_above)[0])
check("a point 141 mm above a reversed face no longer reads as inside",
      _d_broken > 0, _d_broken)
check("...and agrees with the clean patch (%.4f vs %.4f m)"
      % (_d_broken, _d_clean), abs(_d_broken - _d_clean) < 1e-9,
      (_d_broken, _d_clean))

# A POINT BELOW the surface is still inside. The repair must not make
# everything positive.
_below = np.array([[0.0525, 0.0525, -0.004]])
check("a point below the surface is still negative",
      float(_broken.signed_distance(_below)[0]) < 0,
      float(_broken.signed_distance(_below)[0]))

# A CLOSED BODY IS NOT A HEIGHT FIELD and must not be touched: half its faces
# legitimately point away from `outward`.
_cv = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.1],
                [0.1, 0.1, 0.1], [0.0, 0.1, 0.1]])
_ct = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
_box = SurfaceDistance(_cv, _ct, name="closed box")
check("a closed box is NOT a height field", not _box.height_field,
      _box.outward_disagreement_area)
check("...so none of its faces are turned", _box.sign_faces_corrected == 0,
      _box.sign_faces_corrected)


# ==============================================================================
# A "COLLISION" DEEPER THAN THE PART IS THICK
# ==============================================================================
# 2026-10-09. The clearance check reported the Keyence frame 141.8 mm inside a
# planning mesh whose entire z extent was 2 mm, and the outer shell 123.1 mm
# inside it. The sign comes from the nearest triangle's normal, this mesh still
# had faces wound the wrong way after repair, and so points far ABOVE the
# surface read as far inside it.
#
# It was not caught because a large negative number looks exactly like a bad
# collision. It cost 376 of 2925 waypoints, whole passes SKIPPED, 101 passes
# split into 136 and 17.1% of the path -- which is where that run's fragmented,
# stop-start toolpath came from.

print("")
print("impossible penetration")

_depth = ns["_mesh_max_interior_depth"]
_say_bad = ns["_say_impossible_penetration"]

# a 76 x 66 x 2 mm plate, the shape of the real planning mesh
_pv = np.array([[0.0, 0.0, 0.0], [0.076, 0.0, 0.0], [0.076, 0.066, 0.0],
                [0.0, 0.066, 0.0], [0.0, 0.0, 0.002], [0.076, 0.0, 0.002],
                [0.076, 0.066, 0.002], [0.0, 0.066, 0.002]])
_pt = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
_plate = SurfaceDistance(_pv, _pt, name="2 mm plate")
check("a 2 mm plate admits at most 1 mm of penetration",
      abs(_depth(_plate) - 0.001) < 1e-9, _depth(_plate))

_lines = []
_print = __builtins__["print"] if isinstance(__builtins__, dict) else print
import builtins as _b
_orig = _b.print
_b.print = lambda *a, **k: _lines.append(" ".join(str(x) for x in a))
try:
    _say_bad({"clearance": np.array([0.01, 0.02, -0.1418, -0.1231]),
              "per_link": {"lj_v7080_frame": -0.1418, "outer_shell": -0.1231,
                           "ec_probe": -0.0005}},
             _plate, "eddy-current toolpath")
finally:
    _b.print = _orig
_text = " ".join(_lines)
check("a 141.8 mm penetration of a 2 mm plate is called out",
      "IMPOSSIBLE PENETRATION" in _text, _text[:120])
check("...counting the poses it affects", "2 of 4 pose(s)" in _text, _text[:200])
check("...naming the links", "lj_v7080_frame" in _text and "outer_shell" in _text,
      _text[:200])
check("...and not a link only 0.5 mm in, which is plausible",
      "ec_probe" not in _text, _text[:200])
check("...and saying it is the SIGN that is wrong",
      "SIGN of the distance being wrong" in _text, _text[:200])

# A REAL penetration of a thick part must stay silent: this check must not
# explain away a genuine collision.
_bv = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.1],
                [0.1, 0.1, 0.1], [0.0, 0.1, 0.1]])
_block = SurfaceDistance(_bv, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6],
                                        [4, 6, 7]]), name="100 mm block")
_lines = []
_b.print = lambda *a, **k: _lines.append(" ".join(str(x) for x in a))
try:
    _say_bad({"clearance": np.array([0.01, -0.030]),
              "per_link": {"ec_probe": -0.030}}, _block, "path")
finally:
    _b.print = _orig
check("a 30 mm penetration of a 100 mm block is PLAUSIBLE and stays silent",
      _lines == [], _lines)

# AN OPEN SHEET HAS NO THICKNESS, so the bound is vacuous and the check must
# keep quiet rather than call every genuine collision impossible.
_fv = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0]])
_sheet = SurfaceDistance(_fv, np.array([[0, 1, 2], [0, 2, 3]]), name="flat sheet")
check("a flat sheet admits no bound at all", _depth(_sheet) < 1e-9, _depth(_sheet))
_lines = []
_b.print = lambda *a, **k: _lines.append(" ".join(str(x) for x in a))
try:
    _say_bad({"clearance": np.array([-0.018]), "per_link": {"ec_probe": -0.018}},
             _sheet, "path")
finally:
    _b.print = _orig
check("...so a real collision against it is NOT explained away", _lines == [], _lines)


def test_not_blind():
    """The envelope, the surface and the gate all ran, on a positive and a negative."""
    assert callable(tooling_envelope) and callable(path_clearance)
    assert getattr(env, "links", None), "the tooling envelope has no links"
    _names = sorted(l["name"] for l in env.links)
    assert "ec_probe" in _names and "sensor_body" in _names, \
        "the envelope did not come from TOOLING_URDF (%r)" % (_names,)
    # This suite builds no messages at all, so nothing it measures can depend on
    # whose Pose is installed. Prove it rather than assert it in a comment.
    assert not any(isinstance(v, type) and getattr(v, "__module__", "")
                   .startswith("geometry_msgs") for v in globals().values()), \
        "a geometry_msgs class reached this suite's namespace"
    # KNOWN-POSITIVE AND KNOWN-NEGATIVE through the same call. The sensor on a
    # flat plate leaves the probe its whole standoff; the same poses pushed
    # 50 mm down put the barrel through the plate.
    pv_, pt_ = plane_mesh(z=0.0, half=0.30, n=10)
    plate_ = SurfaceDistance(pv_, pt_, name="not-blind plate")
    xs_ = np.linspace(-0.05, 0.05, 8)
    pts_ = np.column_stack([xs_, np.zeros(8), np.zeros(8)])
    nrm_ = np.tile([0.0, 0.0, 1.0], (8, 1))
    ok_ = sensor_on_surface_path(pts_, nrm_, np.array([1.0, 0.0, 0.0]))
    good = path_clearance(ok_, plate_, envelope=env)["min_m"]
    sunk = ScanPath(ok_.positions - np.array([0.0, 0.0, 0.050]), ok_.quats,
                    [0], source="sunk")
    bad = path_clearance(sunk, plate_, envelope=env)["min_m"]
    assert good > 0.005 > bad, \
        "clear reads %+.1f mm and 50 mm into the plate reads %+.1f mm: the " \
        "clearance gate is not responding to its input" % (good*1e3, bad*1e3)
    # And the enforcement has to ACT on that difference, not merely report it:
    # nothing dropped from the clear path, and a refusal for the sunk one.
    _, erep = enforce_path_clearance(ok_, plate_, limit=0.005, envelope=env,
                                     report=False)
    assert erep["dropped"] == 0, \
        "a clear path lost %d poses" % (erep["dropped"],)
    try:
        enforce_path_clearance(sunk, plate_, limit=0.005, envelope=env,
                               report=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a path 50 mm inside the plate was accepted: the "
                             "enforcement is not being consulted")


def test_tooling_clearance():
    """pytest reports what the standalone run reports.

    Without this the file collected NO tests: `pytest tooling_clearance_test.py`
    spent 48 s importing it, said "no tests ran", exit 5, and a failing check
    printed FAIL and failed nothing.
    """
    assert not FAILURES, "%d check(s) failed: %s" % (len(FAILURES),
                                                     "; ".join(FAILURES))


try:
    test_not_blind()
    check("the suite is not blind", True)
except AssertionError as _e:
    check("the suite is not blind (%s)" % (_e,), False)

if __name__ == "__main__":
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all checks passed")
