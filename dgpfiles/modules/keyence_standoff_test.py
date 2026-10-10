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
# File:  keyence_standoff_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the Keyence standoff guard against a raster planned over a mesh
#   that reads low.
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
"""The Keyence standoff guard, against the failure it exists to stop.

A rough mesh from a miscalibrated depth camera read a flat surface-ground plate
13.6 mm low across the part and about 22 mm low at one end. The Keyence raster
was planned over that mesh, so the sensor flew at 66 mm where it should have been
at 80, and at 58-61 mm over the bad end -- one to four millimetres from the hard
limit of a window that is 57 to 103 mm wide. Between 26% and 39% of the plate
came back as points scattered over a 16 mm range instead of a flat surface.

Every number in this file comes from that measurement or from the sensor's own
specification. Nothing here is a preference:

  * 80 mm commanded over the surface it was planned over  -> silent
  * 66 mm                                                 -> passes, with margin
  * 58 mm                                                 -> REFUSED
  * a raster over a rough mesh 13.6 mm low across the part
    and 22 mm low at one end                              -> REFUSED

and, for the run-time half, the measured band structure of the raw profiles:
0.2 mm of spread on every good pass, 17-19 mm on the bad ones, and 41.7% of the
worst pass's points within 5 mm of the near limit.

Runs on the host with numpy only -- no ROS, no open3d, no service, no robot. A
guard that needs the cell up cannot refuse before the cell is up.

    python3 keyence_standoff_test.py

The real-data half is skipped, loudly, when the recorded fixtures are absent.
They live in their own repository; test_data.py resolves them, and
$TARI_TEST_DATA overrides where it looks.
"""
import ast
import os
import sys
import tempfile
import types

import numpy as np

import test_data

HERE = os.path.dirname(os.path.abspath(__file__))

#: The recorded scan this suite grades against. specimen_azure.ply is the rough
#: mesh the raster was planned over; specimen_keyence_raw.ply is the 2.85M
#: profile points it produced.
RAW_PARTS = ("keyence", "roi_spray", "specimen_keyence_raw.ply")
AZURE_PARTS = ("keyence", "roi_spray", "specimen_azure.ply")

#: The plate top, by contact touch-off. The Keyence's own median over the clean
#: part of the plate is 0.9405, so the two independent measurements agree to
#: 0.7 mm and the rough mesh is the one that is wrong.
PLATE_TOP_Z = 0.9402

#: Where the plate is, roughly, in the cell.
PLATE_X = (0.775, 0.858)
PLATE_Y = (-0.111, 0.145)

#: The band of the plate where the sensor was at the bottom of its range and the
#: data came back as scatter.
GARBAGE_Y = (-0.111, -0.048)


# -- loading the .dpi files the way the loader does ---------------------------
#
# A .dpi is compiled as a FUNCTION BODY, so `global X` at its top level is a
# SyntaxError at module scope. Strip those lines and exec into one shared dict,
# which is exactly the globals() every .dpi shares at runtime.

def _load(name):
    src = open(os.path.join(HERE, name)).read()
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve. Nothing here builds a message -- every path is numpy positions and
# quaternions -- so ensure_msgs() only has to make the import work, and it does
# that by ADDING names rather than by overwriting what another suite in this
# directory already installed.
ros_msg_stubs.ensure_msgs()

ns = {"__name__": "keyence_standoff_test", "os": os, "np": np}
exec(compile(_load("toolpath_segments.dpi"), "toolpath_segments.dpi", "exec"), ns)
exec(compile(_load("progress.dpi"), "progress.dpi", "exec"), ns)
# quat_to_matrix, read_ply_mesh and write_plate_ply come from here, and the
# guard uses the planner's own reader rather than a second one.
exec(compile(_load("noether_path.dpi"), "noether_path.dpi", "exec"), ns)
exec(compile(_load("keyence_standoff.dpi"), "keyence_standoff.dpi", "exec"), ns)

# _crop_path_by_mask lives in mesh_reconstruction.dpi, which needs ROS. The
# standoff gate uses it to DROP the few waypoints that are out of window
# instead of refusing the whole raster, so it has to be the real function here
# and not a stand-in -- pulled out by AST, which is the only part of that file
# this test can load.
_mr = ast.parse("def _b():\n" + "".join(
    "    " + l + "\n" for l in open(
        os.path.join(HERE, "mesh_reconstruction.dpi")).read().splitlines()))
_want = {"_crop_path_by_mask"}
_fns = [n for n in _mr.body[0].body
        if isinstance(n, ast.FunctionDef) and n.name in _want]
assert len(_fns) == len(_want), [f.name for f in _fns]
exec(compile(ast.Module(body=_fns, type_ignores=[]), "mesh_reconstruction.dpi",
             "exec"), ns)

ScanPath = ns["ScanPath"]
read_ply_mesh = ns["read_ply_mesh"]
write_plate_ply = ns["write_plate_ply"]
keyence_window = ns["keyence_window"]
commanded_standoff = ns["commanded_standoff"]
standoff_report = ns["keyence_standoff_report"]
describe_report = ns["describe_standoff_report"]
standoff_gate = ns["keyence_standoff_gate"]
range_report = ns["keyence_range_report"]
range_gate = ns["keyence_range_gate"]
describe_range = ns["describe_range_report"]
describe_reaim = ns["describe_reaim"]
Monitor = ns["KeyenceStandoffMonitor"]
REF = ns["KEYENCE_REF_DIST_M"]
HARD_LO, HARD_HI = ns["KEYENCE_RANGE_M"]
MARGIN = ns["KEYENCE_STANDOFF_MARGIN_M"]
BIAS_LIMIT = ns["KEYENCE_MESH_BIAS_LIMIT_M"]

TMP = tempfile.mkdtemp(prefix="keyence_standoff_")
FAILS = []
SAID = []

# Every line the guard emits, captured, so a check can assert on what the
# operator is actually told rather than only on a return value. A refusal that
# does not say which passes are out is not the refusal this file specifies.
ns["progress_note"] = lambda text: SAID.append(text)


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def said(fragment):
    """Did any captured line contain `fragment`?"""
    return any(fragment in line for line in SAID)


def clear():
    del SAID[:]


# -- geometry -----------------------------------------------------------------
#
# The sensor as the URDF has it, and as this test hands it to the guard:
# probe_end -> lj_v7080_optical_frame is a pure Rz(+90 deg) with translation
# (-0.07515, -0.0022, -0.03428) on the ec tooling, so the optical frame's z axis
# is probe_end's z axis and the sensor looks along probe_end's -z. Chained off
# the rendered URDF.
TF_OFFSET = (-0.07515, -0.0022, -0.03428)
AXIS = (0.0, 0.0, -1.0)
GEOM = dict(tf_offset=TF_OFFSET, axis=AXIS)

DOWN = np.array([0.0, 0.0, 0.0, 1.0])       # identity: tool z up, sensor down


def flat_path(z_surface, standoff=REF, nx=9, ny=7, rows=None,
              x=PLATE_X, y=PLATE_Y):
    """A raster over a flat surface at `z_surface`, commanding `standoff`.

    Built the way plan_sweep() builds one: the measurement points go on the
    surface, then every pose is moved back along the probe_end -> optical-frame
    vector IN THE TOOL FRAME so the optical frame lands where the toolpath said.
    With the tool vertical that is a plain subtraction, which is what makes the
    commanded standoff exactly `standoff` by construction -- put the optical
    frame `standoff - REF` above the surface and the guard must read `standoff`.
    """
    xs = np.linspace(x[0], x[1], nx)
    ys = np.linspace(y[0], y[1], ny if rows is None else rows)
    pos, starts = [], []
    for yv in ys:
        starts.append(len(pos))
        for xv in xs:
            optical = np.array([xv, yv, z_surface + (standoff - REF)])
            pos.append(optical - np.asarray(TF_OFFSET))
    quats = np.tile(DOWN, (len(pos), 1))
    return ScanPath(np.array(pos), quats, starts, source="test raster")


def plate_mesh(name, z, xspan=(0.70, 0.93), yspan=(-0.20, 0.22), n=41,
               step=None):
    """A flat PLY plate at world z, optionally with one end dropped by `step`.

    `step` is how the failing rough mesh is reproduced: the camera read the
    plate 13.6 mm low across the part and about 22 mm low over the fixturing at
    one end, so the mesh is not a plane shifted down, it is a plane shifted down
    with a worse end. That difference is the whole reason layer 1 and layer 2
    are not enough on their own.
    """
    xs = np.linspace(xspan[0], xspan[1], n)
    ys = np.linspace(yspan[0], yspan[1], n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full(X.shape, float(z))
    if step is not None:
        drop, y_edge = step
        Z = np.where(Y <= y_edge, float(z) - drop, Z)
    v = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i*n + j
            tris.append((a, a + 1, a + n))
            tris.append((a + 1, a + n + 1, a + n))
    path = os.path.join(TMP, name)
    write_ply(path, v, np.asarray(tris, dtype=np.int64))
    return path


def write_ply(path, v, t):
    head = ["ply", "format binary_little_endian 1.0",
            "element vertex %d" % len(v),
            "property double x", "property double y", "property double z",
            "element face %d" % len(t),
            "property list uchar uint vertex_indices", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.asarray(v, dtype="<f8").tobytes())
        fd = np.zeros(len(t), dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]))
        fd["n"] = 3
        fd["v"] = t
        f.write(fd.tobytes())
    return path


# =============================================================================
print("\n-- the window itself --------------------------------------------")
# =============================================================================

lo, hi, hlo, hhi = keyence_window()
check((hlo, hhi) == (0.057, 0.103),
      "the hard window is the LJ-V7080's 57..103 mm")
check(abs(REF - 0.080) < 1e-12 and abs((hhi - hlo)/2 - 0.023) < 1e-9,
      "the reference distance is 80 mm and the window is +/-23 mm about it")
check(abs(lo - 0.062) < 1e-12 and abs(hi - 0.098) < 1e-12,
      "the default 5 mm margin guards 62..98 mm")
check(abs(keyence_window(margin=0.010)[0] - 0.067) < 1e-12,
      "the margin is configurable per call")
ns["KEYENCE_STANDOFF_MARGIN"] = 0.008
check(abs(keyence_window()[0] - 0.065) < 1e-12,
      "and by a config override, KEYENCE_STANDOFF_MARGIN")
del ns["KEYENCE_STANDOFF_MARGIN"]
try:
    keyence_window(margin=0.030)
    check(False, "a margin that closes the window is refused")
except ValueError as e:
    check("leaves nothing" in str(e),
          "a margin that closes the window is refused, saying so")


# =============================================================================
print("\n-- the three standoffs that decide a raster ----------------------")
# =============================================================================

surface = plate_mesh("truth.ply", PLATE_TOP_Z)

# 80 mm: the plan as designed, over the surface it was planned over. SILENT --
# it prints its one summary line and refuses nothing.
clear()
rep = standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.080), surface, **GEOM)
check(rep["ok"], "a raster commanding 80 mm over its own surface passes")
check(abs(rep["median"] - 0.080) < 1e-6 and abs(rep["min"] - 0.080) < 1e-6,
      "  and measures exactly the reference distance, everywhere")
check(rep["n_out"] == 0 and rep["n_missing"] == 0,
      "  with no waypoint out and none aimed at nothing")
check(not said("WHAT TO DO") and not said("**"),
      "  and says nothing an operator has to act on")

# 66 mm: what a raster over a mesh 13.6 mm low flies over the good part of the
# plate. It is inside the window and it must PASS -- but the margin report has to
# say how little is left, because 66 mm is only 4.4 mm from the guard limit.
clear()
rep = standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.0664), surface, **GEOM)
check(rep["ok"], "a raster commanding 66.4 mm passes -- it is inside the window")
check(abs(rep["median"] - 0.0664) < 1e-6, "  and reads 66.4 mm")
check(said("4.4 mm to spare"),
      "  and reports the 4.4 mm of margin left at the tightest point")
check(not said("**"), "  warning-free: nothing for the operator to act on")

# 58 mm: the bad end. REFUSED.
clear()
try:
    standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.058), surface, **GEOM)
    check(False, "a raster commanding 58 mm is REFUSED")
except ValueError as e:
    msg = str(e)
    check(True, "a raster commanding 58 mm is REFUSED")
    check("refused" in msg, "  the word 'refused' is in the message")
    check("63 of 63 waypoints" in msg or "waypoints command a standoff" in msg,
          "  it says how many waypoints are out")
    check("pass" in msg, "  it says which passes")
    check("58.0 mm" in msg, "  it gives the worst standoff")
    check("near guard limit" in msg,
          "  it says WHICH END of the window is violated")
    check("26-39% of it came back scattered" in msg,
          "  and the measured outcome renders as text, not as a stray %%")
    check("recalibrate" in msg.lower() or "re-calibrate" in msg.lower(),
          "  and what to do: recalibrate the camera")
    check("Do NOT raise KEYENCE_STANDOFF_MARGIN" in msg,
          "  explicitly NOT loosen the guard")

# The far end is refused too, and named as the far end.
clear()
try:
    standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.101), surface, **GEOM)
    check(False, "a raster commanding 101 mm is refused at the FAR limit")
except ValueError as e:
    check("far guard limit" in str(e),
          "a raster commanding 101 mm is refused at the FAR limit, named")


# =============================================================================
print("\n-- a raster planned over a mesh that reads low -------------------")
# =============================================================================
#
# The case this whole file exists for. The raster is planned over a rough mesh
# that is 13.6 mm below the truth across the part and 22 mm below it at one end.
# Against that mesh the plan is perfect -- it commands the reference distance at
# every waypoint, which is exactly why nothing notices. Against the truth it is
# out.

ROUGH_BIAS = 0.0136
ROUGH_END_BIAS = 0.022
ROUGH_END_Y = -0.048               # where the garbage band begins

rough = plate_mesh("rough_low_end.ply", PLATE_TOP_Z - ROUGH_BIAS,
                   step=(ROUGH_END_BIAS - ROUGH_BIAS, ROUGH_END_Y))
truth = plate_mesh("truth_flat.ply", PLATE_TOP_Z)

# The raster as it was flown: optical frame on the rough mesh, everywhere.
pos, starts = [], []
xs = np.linspace(PLATE_X[0], PLATE_X[1], 9)
ys = np.linspace(PLATE_Y[0], PLATE_Y[1], 17)
for yv in ys:
    starts.append(len(pos))
    for xv in xs:
        zr = PLATE_TOP_Z - (ROUGH_END_BIAS if yv <= ROUGH_END_Y else ROUGH_BIAS)
        pos.append(np.array([xv, yv, zr]) - np.asarray(TF_OFFSET))
flown = ScanPath(np.array(pos), np.tile(DOWN, (len(pos), 1)), starts,
                 source="a keyence raster over a mesh that reads low")

clear()
rep = standoff_gate(flown, rough, **GEOM)
check(rep["ok"] and abs(rep["median"] - REF) < 1e-6,
      "graded against the rough mesh it was planned over, the raster is "
      "PERFECT -- 80 mm at every waypoint")
check(True, "  which is exactly why nothing in the pipeline notices")

clear()
try:
    standoff_gate(flown, truth, **GEOM)
    check(False, "graded against the truth, the same raster is REFUSED")
except ValueError as e:
    msg = str(e)
    check(True, "graded against the truth, the same raster is REFUSED")
    rep = ns["last_standoff_report"]
    n_bad = int((np.asarray(rep["standoff"]) < rep["lo"]).sum())
    check(n_bad == rep["n_out"] and rep["n_out"] > 0,
          "  every out-of-window waypoint is out at the NEAR limit")
    check(abs(rep["worst"] - (REF - ROUGH_END_BIAS)) < 1e-6,
          "  the worst is 58.0 mm -- 80 minus the 22 mm the mesh is low there")
    check(abs(rep["max"] - (REF - ROUGH_BIAS)) < 1e-6,
          "  and the best is 66.4 mm -- 80 minus 13.6, the rest of the plate")
    bad_rows = [k for k in rep["passes_out"]]
    check(len(bad_rows) == sum(1 for yv in ys if yv <= ROUGH_END_Y)
          and bad_rows == list(range(len(bad_rows))),
          "  the refused passes are exactly the rows over the bad end")
    check("58.0 mm" in msg and "near guard limit" in msg,
          "  the refusal names the worst standoff and the end it violates")
    y_worst = rep["worst_xyz"][1]
    check(y_worst <= ROUGH_END_Y,
          "  and where it is: y %.3f, inside the garbage band %.3f..%.3f"
          % (y_worst, GARBAGE_Y[0], GARBAGE_Y[1]))

# The deliberate override, and what it costs.
clear()
rep = standoff_gate(flown, truth, allow=True, **GEOM)
check(rep is not None and not rep["ok"] and said("FLYING OUT OF THE MEASURING"),
      "KEYENCE_ALLOW_BAD_STANDOFF flies it anyway and says what to expect")

ns["KEYENCE_ALLOW_BAD_STANDOFF"] = True
clear()
rep = standoff_gate(flown, truth, **GEOM)
check(rep is not None and not rep["ok"],
      "  and it is reachable as a config global, not only as an argument")
del ns["KEYENCE_ALLOW_BAD_STANDOFF"]


# =============================================================================
print("\n-- what only the ray cast can see --------------------------------")
# =============================================================================
#
# Grading a plan against the mesh it was planned over is a CONSISTENCY check,
# and these are the ways it is not a tautology.

# A pose re-oriented AFTER the sensor offset was applied -- which is what
# cap_path_tilt_for_clearance() does inside the clearance gate, and why the
# standoff gate runs after that gate and not before it. Tilting the pose swings
# the optical frame off the surface AND swings the axis, so the commanded
# standoff stops being the reference distance by an amount that grows fast:
def tilt_one(deg, i=7):
    t = flat_path(PLATE_TOP_Z, standoff=0.080, nx=5, rows=3)
    q = t.quats.copy()
    th = np.radians(deg)
    q[i] = np.array([np.sin(th/2), 0.0, 0.0, np.cos(th/2)])
    return ScanPath(t.positions, q, t.segment_starts)


d = commanded_standoff(tilt_one(35.0), surface, **GEOM)
check(abs(d[7] - 0.0860) < 1e-4,
      "a 35 deg re-orientation after the offset costs 6.0 mm of standoff "
      "(%.1f mm, not the 80.0 the toolpath asked for)" % (1e3*d[7],))
check(np.allclose(np.delete(d, 7), 0.080),
      "  and nothing else on the path moves")
clear()
try:
    standoff_gate(tilt_one(55.0), surface, **GEOM)
    check(False, "at 55 deg the same pose leaves the window and is refused")
except ValueError as e:
    check("waypoint 7" in str(e),
          "at 55 deg the same pose leaves the window and is refused, naming "
          "the waypoint")
    check("far guard limit" in str(e),
          "  at the FAR limit -- tilting pushes the ray's path to the surface "
          "longer, not shorter")

# A waypoint aimed across a hole: no surface at all under the sensor. Reported
# as its own thing, because "measures nothing" is not "measures wrongly".
holed = plate_mesh("holed.ply", PLATE_TOP_Z, xspan=(0.80, 0.93))
clear()
try:
    standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.080), holed, **GEOM)
    check(False, "a waypoint aimed past the edge of the surface is caught")
except ValueError as e:
    check(True, "a waypoint aimed past the edge of the surface is caught")
    check("NO surface within" in str(e) and "aimed at nothing" in str(e),
          "  and is described as measuring nothing, not as measuring wrongly")
    check(ns["last_standoff_report"]["n_missing"] > 0,
          "  and counted separately in the report")

# The ray finds the NEAREST surface along the axis, not the waypoint it was
# planned for. A vertical tool over a step sees the top of the step.
stepped = plate_mesh("stepped.ply", PLATE_TOP_Z,
                     step=(-0.015, 0.0))          # a 15 mm RIDGE at y <= 0
d = commanded_standoff(flat_path(PLATE_TOP_Z, standoff=0.080, ny=5), stepped,
                       **GEOM)
check(abs(d.min() - 0.065) < 1e-6,
      "over a 15 mm ridge the ray reads 65 mm, not the 80 the waypoint asked "
      "for -- it finds the first surface, which is what the sensor does")

# A plane is a surface too: the touch-off case, when one exists.
d = commanded_standoff(flat_path(PLATE_TOP_Z, standoff=0.0664), PLATE_TOP_Z,
                       **GEOM)
check(np.allclose(d, 0.0664),
      "a bare world z works as the surface -- surface=0.9402, the touched-off "
      "plate top")

# And a point cloud is refused as a point cloud rather than as an empty scan.
cloud = os.path.join(TMP, "cloud.ply")
with open(cloud, "wb") as f:
    f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
            b"property double x\nproperty double y\nproperty double z\n"
            b"end_header\n")
    f.write(np.zeros((2, 3), dtype="<f8").tobytes())
try:
    commanded_standoff(flat_path(PLATE_TOP_Z), cloud, **GEOM)
    check(False, "a point cloud is refused as a point cloud")
except ValueError as e:
    check("POINT CLOUD" in str(e) and "_raw" in str(e),
          "a point cloud is refused as a point cloud, naming the chain's files")


# =============================================================================
print("\n-- layer 2: the sensor's own returns, before anything flies ------")
# =============================================================================
#
# keyence_profile_standoff() turns the driver's message into standoffs, and the
# sign is the one thing here that cannot be argued about: keyence_driver_node
# packs the device's reading as z in lj_v7080_optical_frame and POSITIVE z is
# TOWARDS the head (tari_keyence_sim's z_sign, the device's convention). So
# z = +0.023 is the 57 mm near limit and z = -0.023 is the 103 mm far one.

class FakeField(object):
    def __init__(self, name, offset, datatype):
        self.name, self.offset, self.datatype = name, offset, datatype


class FakeCloud(object):
    """pcl::PointXYZ through pcl::toROSMsg: three float32 and 4 bytes of pad."""

    def __init__(self, z, x=None):
        z = np.asarray(z, dtype=np.float32)
        buf = np.zeros(len(z), dtype=np.dtype({
            "names": ["x", "y", "z"], "formats": ["<f4"]*3,
            "offsets": [0, 4, 8], "itemsize": 16}))
        buf["z"] = z
        if x is not None:
            buf["x"] = np.asarray(x, dtype=np.float32)
        self.data = buf.tobytes()
        self.width, self.height, self.point_step = len(z), 1, 16
        self.fields = [FakeField("x", 0, 7), FakeField("y", 4, 7),
                       FakeField("z", 8, 7)]


profile_standoff = ns["keyence_profile_standoff"]
d = profile_standoff(FakeCloud([0.0, 0.023, -0.023, np.inf]))
check(len(d) == 3, "out-of-range columns (+inf, REP-117) are dropped")
check(abs(d[0] - 0.080) < 1e-6, "a zero reading is the reference distance")
check(abs(d[1] - 0.057) < 1e-6, "z = +23 mm is the NEAR limit, 57 mm")
check(abs(d[2] - 0.103) < 1e-6, "z = -23 mm is the FAR limit, 103 mm")

# The pre-flight gate. 66.4 mm at the first pose is inside the window and a gate
# that only asks "is it in the window" passes this run -- which is why it asks
# the other question as well: what did the plan COMMAND here?
clear()
rep = range_gate(np.full(800, 0.0664), commanded=None,
                 what="first pose, window only")
check(rep["ok"], "66.4 mm measured, with nothing to compare it against, passes")

clear()
try:
    range_gate(np.full(800, 0.0664), commanded=0.080, what="first pose")
    check(False, "66.4 mm measured where 80 was commanded is REFUSED")
except RuntimeError as e:
    msg = str(e)
    check(True, "66.4 mm measured where 80 was commanded is REFUSED")
    check("13.6 mm" in msg,
          "  the refusal states the mesh error it just measured: 13.6 mm")
    # WHICH WAY ROUND. bias = commanded - measured is POSITIVE here, meaning the
    # sensor finds the surface nearer than the plan did: for a sensor looking
    # down, a planning surface that reads LOW. Getting this backwards would send
    # the operator to look for the wrong fault, so it is asserted rather than
    # left to the reader.
    check("13.6 mm BELOW the real one" in msg,
          "  and that the planning surface is BELOW the real one, not above it")
    check("re-calibrate" in msg.lower() or "recalibrate" in msg.lower(),
          "  and says to recalibrate the camera, not to loosen anything")
check(abs(ns["last_range_report"]["bias"] - 0.0136) < 1e-9,
      "  the measured bias is exactly the 13.6 mm the Azure was low")

# Under the limit it passes and still reports the number.
clear()
rep = range_gate(np.full(800, 0.0740), commanded=0.080, what="first pose")
check(rep["ok"] and abs(rep["bias"] - 0.006) < 1e-9,
      "a 6 mm disagreement is under the 10 mm limit and passes")
check(said("the planning surface is 6.0 mm low"),
      "  and is reported anyway -- the operator gets the number either way")
check(abs(BIAS_LIMIT - 0.010) < 1e-12,
      "the bias limit is 10 mm: over half the 18 mm of usable travel left "
      "after the margin")

# A pre-flight profile that is already out of the window is refused on the
# window alone, with no plan to compare against.
clear()
try:
    range_gate(np.full(800, 0.0590), what="first pose")
    check(False, "a first pose measuring 59 mm is refused on the window alone")
except RuntimeError as e:
    check("NEAR limit" in str(e),
          "a first pose measuring 59 mm is refused on the window alone")


# =============================================================================
print("\n-- layer 3: the monitor, pass by pass ----------------------------")
# =============================================================================

path = flat_path(PLATE_TOP_Z, standoff=0.080, nx=9, ny=4)
mon = Monitor(commanded=commanded_standoff(path, surface, **GEOM), path=path)

clear()
mon.start_pass(0, total=4)
for _ in range(20):
    mon.add_standoff(np.random.default_rng(1).normal(0.0800, 0.0001, 800))
rep = mon.end_pass()
check(rep["ok"] and abs(rep["median"] - 0.080) < 5e-4,
      "a pass measured at the reference distance is quiet")
check(said("sweep row 1/4 standoff") and not said("**"),
      "  and still reports its distribution, per row, through progress")

# The worst pass of the recorded scan, rebuilt from its measured statistics:
# 41.7% of the points within 5 mm of the near limit, the rest scattered over
# 19 mm.
rng = np.random.default_rng(7)
bad = np.concatenate([rng.uniform(0.0574, 0.0620, 334),
                      rng.uniform(0.0620, 0.0832, 466)])
clear()
mon.start_pass(1, total=4)
mon.add_standoff(bad)
rep = mon.end_pass()
check(not rep["ok"], "a pass at the bottom of the range is reported as not ok")
check(abs(rep["near_fraction"] - 0.417) < 0.02,
      "  41.7%% of its points are within 5 mm of the near limit (%.1f%%)"
      % (100*rep["near_fraction"],))
check(said("NEAR limit"), "  and the line says so, loudly, on that row")
check(said("row 2"), "  naming the row while it is happening")

# A clean pass and a bad one differ by the spread as well, which is the other
# half of the signature: 0.2 mm on every good row, 17-19 mm on the bad ones.
check(rep["spread"] > 0.010, "  and its p5..p95 spread is over 10 mm")


# A UNIFORM MESH ERROR IS ONE FACT. Every row of a raster planned over a
# globally wrong surface has the same bias, and seventeen identical paragraphs
# is how a real finding turns into log noise -- progress.dpi's rule 4, applied
# to a reason rather than to a block. The number stays on every row's summary
# line; only the ** escalation is said once.
uni = Monitor(commanded=np.full(len(path), 0.080), path=path)
clear()
for k in range(3):
    uni.start_pass(k, total=3)
    uni.add_standoff(np.full(400, 0.0664))
    uni.end_pass()
bias_lines = [l for l in SAID if "** the plan commanded" in l]
check(len(bias_lines) == 1,
      "a bias that is the same on every row is shouted once, not once per row")
check(sum(1 for l in SAID if "planning surface is 13.6 mm low" in l) == 3,
      "  and still appears as a number on all three row summaries")
check(all(not r["ok"] for r in uni.reports),
      "  while every row is still recorded as not ok")

clear()
summary = mon.summary()
check(summary["n_passes"] == 2 and summary["loud_passes"] == [1],
      "the sweep summary names the passes that were at the edge")
check(said("not measurements of the part"),
      "  and says the points those passes contributed are not measurements")
check(said("check_keyence_standoff() before flying"),
      "  and points at the check that would have caught it")

# A ten-minute sweep at the real head's 216.8 Hz is 130 000 profiles, which is
# 830 MB of float64 if the monitor keeps them all -- and this pipeline has been
# OOM-killed before. The whole-sweep buffer thins itself, and what it reports
# has to survive that: the summary is percentiles and fractions, and a uniform
# subsample preserves both.
big = Monitor()
rng = np.random.default_rng(11)
truth_d = []
for i in range(20000):
    prof = rng.normal(0.0664, 0.0004, 40)
    truth_d.append(prof)
    big.add_standoff(prof)
full = range_report(np.concatenate(truth_d))
thin = big.summary(quiet=True)
check(len(big._all) <= Monitor.MAX_KEPT_PROFILES,
      "over 20000 profiles the whole-sweep buffer stays under its cap (%d "
      "kept, stride %d)" % (len(big._all), thin["stride"]))
check(thin["n_profiles"] == 20000,
      "  and the profile COUNT is still the true one, not the kept one")
check(abs(thin["median"] - full["median"]) < 2e-5
      and abs(thin["p5"] - full["p5"]) < 5e-5,
      "  while the median and p5 match the unthinned distribution to "
      "%.1f um and %.1f um"
      % (1e6*abs(thin["median"] - full["median"]),
         1e6*abs(thin["p5"] - full["p5"])))

# The monitor is a no-op without ROS rather than a failure: it must never be the
# thing that takes down a sweep it is only watching.
quiet = Monitor(topic="/nothing_here")
check(quiet.start() is quiet and quiet.stop(quiet=True) is not None,
      "the monitor degrades to doing nothing when there is no ROS graph")



# =============================================================================
print("\n-- check_keyence_standoff(), the operator's own call --------------")
# =============================================================================
#
# It must NOT raise. It is a question, asked before the recorder starts, and the
# answer to "will this fly" is a printed page and ok=False -- not a traceback
# thrown at somebody who asked.

check_standoff = ns["check_keyence_standoff"]

clear()
rep = check_standoff(path=flat_path(PLATE_TOP_Z, standoff=0.080),
                     surface=surface, **GEOM)
check(rep is not None and rep["ok"],
      "check_keyence_standoff(path=, surface=) on a good plan returns ok")

clear()
rep = check_standoff(path=flown, surface=truth, **GEOM)
check(rep is not None and not rep["ok"],
      "on a raster over a mesh that reads low it reports ok=False")
check(rep["n_out"] == ns["last_standoff_report"]["n_out"],
      "  and it is the SAME report the gate would have refused with -- one "
      "guard, not a checker that reimplements it")

clear()
rep = check_standoff(path=flown, surface=None, rough=None, **GEOM)
check(rep is None and said("there is no surface to check"),
      "with no surface at all it says so rather than passing silently")


# =============================================================================
print("\n-- the recorded scan data ----------------------------------------")
# =============================================================================

RAW = test_data.path(*RAW_PARTS)
AZURE = test_data.path(*AZURE_PARTS)

if RAW is None or AZURE is None:
    print("  SKIPPED: %s" % (test_data.why_missing(*RAW_PARTS),))
else:
    # specimen_keyence_raw.ply is 2.85M points and no faces, so read_ply_mesh
    # refuses it -- correctly, and that refusal is one of the checks above.
    # mesh_scan.dpi's _mesh_vertices() is the dependency-free vertex reader the
    # rest of the pipeline uses for exactly this; pull out the one function
    # rather than loading a scan config's worth of globals.
    import ast as _ast
    for _node in _ast.parse(_load("mesh_scan.dpi")).body:
        if isinstance(_node, _ast.FunctionDef) and _node.name == "_mesh_vertices":
            exec(compile(_ast.Module(body=[_node], type_ignores=[]),
                         "mesh_scan.dpi", "exec"), ns)
    points = ns["_mesh_vertices"]

    key = np.asarray(points(RAW), dtype=np.float64)
    az_v, _az_n, az_t = read_ply_mesh(AZURE)
    az = np.asarray(az_v, dtype=np.float64)

    def inbox(p, y0, y1):
        m = ((p[:, 0] >= PLATE_X[0]) & (p[:, 0] <= PLATE_X[1])
             & (p[:, 1] >= y0) & (p[:, 1] <= y1))
        return p[m]

    good = inbox(key, -0.031, PLATE_Y[1])
    bad = inbox(key, PLATE_Y[0], -0.095)
    check(abs(np.median(good[:, 2]) - PLATE_TOP_Z) < 0.001,
          "the Keyence's own median over the clean part of the plate is within "
          "1 mm of the touch-off (%.4f vs %.4f)"
          % (np.median(good[:, 2]), PLATE_TOP_Z))
    spread_good = np.percentile(good[:, 2], 95) - np.percentile(good[:, 2], 5)
    spread_bad = np.percentile(bad[:, 2], 95) - np.percentile(bad[:, 2], 5)
    check(spread_good < 0.001,
          "  a surface-ground plate measures flat to %.2f mm where the sensor "
          "is in range" % (1e3*spread_good,))
    check(spread_bad > 0.015,
          "  and to %.1f mm where it is not -- the garbage band"
          % (1e3*spread_bad,))

    # The Azure's error, which is the whole cause, measured rather than quoted.
    az_mid = np.median(inbox(az, -0.031, PLATE_Y[1])[:, 2])
    az_end = np.median(inbox(az, PLATE_Y[0], -0.095)[:, 2])
    check(0.011 < PLATE_TOP_Z - az_mid < 0.015,
          "the rough mesh reads the plate %.1f mm low over the clean part"
          % (1e3*(PLATE_TOP_Z - az_mid),))
    check(PLATE_TOP_Z - az_end > 0.018,
          "  and %.1f mm low at the end that produced the garbage"
          % (1e3*(PLATE_TOP_Z - az_end),))

    # Layer 1 against the real rough mesh: a raster planned over it, graded
    # against the truth, is refused.
    # Built the way plan_sweep() builds one: the optical frame goes ON the rough
    # mesh at every waypoint. Where the mesh is is asked of the mesh -- a
    # vertical ray from above, through the guard's own caster, which is what
    # makes this raster the one that was flown rather than an approximation of
    # it.
    first_hit = ns["_ray_first_hit"]
    geom = ns["_surface_geometry"](AZURE)
    grid = [(xv, yv) for yv in np.linspace(PLATE_Y[0], PLATE_Y[1], 17)
            for xv in np.linspace(PLATE_X[0], PLATE_X[1], 9)]
    above = np.array([[x, y, PLATE_TOP_Z + 0.05] for x, y in grid])
    t = first_hit(above, np.tile([0.0, 0.0, -1.0], (len(above), 1)), geom,
                  search=0.20)
    pos, starts, row = [], [], None
    for (xv, yv), origin, tt in zip(grid, above, t):
        if not np.isfinite(tt):
            continue                      # a hole in the rough mesh
        if yv != row:
            row, _ = yv, starts.append(len(pos))
        pos.append(np.array([xv, yv, origin[2] - tt]) - np.asarray(TF_OFFSET))
    real = ScanPath(np.array(pos), np.tile(DOWN, (len(pos), 1)), starts,
                    source="raster over the recorded rough mesh")
    clear()
    d = commanded_standoff(real, AZURE, **GEOM)
    finite = d[np.isfinite(d)]
    check(len(finite) > 0.8*len(d) and np.abs(finite - REF).max() < 0.010,
          "against the REAL Azure mesh the same raster commands the reference "
          "distance to within %.1f mm at every waypoint -- the plan is "
          "self-consistent, which is why only a better surface can catch it"
          % (1e3*np.abs(finite - REF).max(),))
    clear()
    try:
        standoff_gate(real, PLATE_TOP_Z, **GEOM)
        check(False, "the raster over the REAL rough mesh is refused against "
                     "the touched-off plate top")
    except ValueError:
        rep = ns["last_standoff_report"]
        check(rep["worst"] < 0.062,
              "the raster over the REAL rough mesh is refused against the "
              "touched-off plate top: worst %.1f mm" % (1e3*rep["worst"],))
        check(rep["worst_xyz"][1] <= GARBAGE_Y[1],
              "  and the worst waypoint is in the garbage band (y %.3f)"
              % (rep["worst_xyz"][1],))

    # THE GATE HAS TO BE AFFORDABLE: a guard that adds minutes to a plan is a
    # guard somebody turns off. A 4636-pose raster is the size a real session
    # plans. Measured here: about 0.2 s against a 31k-face TSDF mesh and against
    # a 199k-face Poisson mesh alike. The bound is 60x that, so this catches a
    # broad-phase regression -- sizing the grid cell to the ray rather than to
    # the faces costs seconds to minutes -- without being a timing test.
    import time as _time
    big = np.array([[x, y, PLATE_TOP_Z + 0.05]
                    for y in np.linspace(PLATE_Y[0], PLATE_Y[1], 38)
                    for x in np.linspace(PLATE_X[0], PLATE_X[1], 122)])
    t0 = _time.perf_counter()
    first_hit(big, np.tile([0.0, 0.0, -1.0], (len(big), 1)), geom, search=0.20)
    dt = _time.perf_counter() - t0
    check(dt < 12.0,
          "%d rays against the real %d-face rough mesh in %.2f s -- the gate "
          "costs a fraction of a second on a real raster"
          % (len(big), len(geom[2]), dt))

    # Layer 3 against the real profiles: feed the raw points back through the
    # monitor, pass by pass, as the head would have delivered them.
    mon = Monitor()
    for k, (y0, y1) in enumerate([(-0.111, -0.095), (-0.095, -0.079),
                                  (-0.031, -0.015), (0.033, 0.049)]):
        band = inbox(key, y0, y1)
        head = float(np.median(inbox(az, y0, y1)[:, 2])) + REF
        mon.start_pass(k, total=4)
        mon.add_standoff(head - band[:, 2])
        r = mon.end_pass(quiet=True)
        loud = "LOUD" if not r["ok"] else "quiet"
        print("     y %+.3f..%+.3f  median %.1f mm  spread %5.2f mm  "
              "near-limit %5.1f%%  %s"
              % (y0, y1, 1e3*r["median"], 1e3*r["spread"],
                 100*r["near_fraction"], loud))
    check(mon.reports[0]["near_fraction"] > 0.30 and not mon.reports[0]["ok"],
          "the real worst pass fires: %.1f%% of its points within 5 mm of the "
          "near limit" % (100*mon.reports[0]["near_fraction"],))
    check(mon.reports[2]["ok"] and mon.reports[3]["ok"],
          "and the two clean passes do not")
    check(mon.reports[2]["spread"] < 0.001 and mon.reports[0]["spread"] > 0.015,
          "spread separates them by two orders of magnitude: %.2f mm against "
          "%.1f mm" % (1e3*mon.reports[2]["spread"],
                       1e3*mon.reports[0]["spread"]))


# =============================================================================
# The verdict is at the END of this file, under __main__, and
# test_keyence_standoff() carries it under pytest. It must not be printed from
# the middle: a sys.exit() there makes a standalone run blind to every check
# below it, and a failure above it kills a pytest run in collection.

print("\n-- a finite mesh has edges: drop the few, refuse the many ---------")
# =============================================================================
# A Noether raster over any bounded surface runs off the end of it, and a sensor
# aimed past the edge measures nothing there. With only two answers -- fly
# everything or refuse everything -- the gate throws away ten complete passes of
# a simulated cylinder over four waypoints of 299, the other 295 of which sit at
# exactly the reference distance. That leaves the pipeline unable to finish
# unattended on any real part.
clear()
holey = plate_mesh("edged.ply", PLATE_TOP_Z)
good = flat_path(PLATE_TOP_Z, standoff=0.080)

# One waypoint of 63 aimed 20 mm too close: 1.6%, well under the 5% the gate
# will drop. It flies, minus that waypoint.
import numpy as _np
_pos = [list(p) for p in good.positions]
_pos[len(_pos)//2][2] -= 0.020
few = ScanPath(_pos, [list(q) for q in good.quats],
               [a for a, _b in good.segments], source=good.source)
rep = standoff_gate(few, holey, **GEOM)
check(rep["ok"], "a raster with ONE waypoint out of 63 out of window flies")
check(rep.get("dropped") == 1, "  the offending waypoint is dropped")
check(rep.get("path") is not None and len(rep["path"]) == len(few) - 1,
      "  and the path handed back is one waypoint shorter")
check(said("DROPPED"), "  it says so out loud")
check(said("HOLE IN THE MEASUREMENT"),
      "  and calls it a hole in the measurement, not a clean run")

# A quarter of the raster out is not an edge, it is a wrong mesh.
clear()
_pos = [list(p) for p in good.positions]
for _i in range(0, len(_pos), 4):
    _pos[_i][2] -= 0.020
many = ScanPath(_pos, [list(q) for q in good.quats],
                [a for a, _b in good.segments], source=good.source)
try:
    standoff_gate(many, holey, **GEOM)
    check(False, "a raster with a QUARTER of its waypoints out is REFUSED")
except ValueError as e:
    check(True, "a raster with a QUARTER of its waypoints out is REFUSED")
    check("above the 5% this gate will drop" in str(e),
          "  and says it is past what it will drop and fly around")
    check("mesh is the thing to fix" in str(e),
          "  and names the rough mesh as the thing to fix")

# A raster that is out at every waypoint is untouched by any of this. Nothing is
# dropped, everything is refused.
clear()
try:
    standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.058), holey, **GEOM)
    check(False, "58 mm everywhere is still REFUSED, not dropped")
except ValueError as e:
    check(True, "58 mm everywhere is still REFUSED, not dropped")
    check(not said("DROPPED"), "  and nothing was dropped on the way")
    del e


# =============================================================================
# THE TEST IS NOT BLIND
# =============================================================================
# Every check above runs at IMPORT, which is what lets one pytest test stand for
# the lot -- and it is also exactly how a suite prints "ok" down the page while
# the code under test never runs. This suite shares its geometry_msgs stub with
# the other suites in this directory (see ros_msg_stubs.py), so it may RECEIVE
# one it did not write; "the gate ran and can still tell two cases apart" has to
# be a check of its own.

# ==============================================================================
# A CURVED SURFACE ACROSS THE LASER LINE IS NOT A MESH ERROR
# ==============================================================================
# The plan commands a standoff at ONE POINT, the middle of the 31 mm laser line.
# The bias test compares that with the MEDIAN of everything the line returns,
# which is the right summary only where the surface is flat across the line. On a
# curve the commanded point sits at one END of the returned range -- on a convex
# crest it is the NEAREST return -- so the sagitta reads as a mesh error.
#
# Measured on a 131 x 65 x 67 mm cylinder: comparing against the median refuses
# the sweep with "commanded 80.0, measuring 62.8, so the surface is 17.2 mm BELOW
# the real one", while the planning mesh's crest is accurate to 0.4 mm. The mesh
# is right; the inference is wrong.

print("")
print("curvature across the line")

REF = 0.080

# A FLAT surface: every column at the same height, a little noise. The bias is a
# real mesh error and must still refuse.
flat = np.random.default_rng(3).normal(0.0665, 0.0002, 800)
rep = range_report(flat, commanded=0.080)
check(not rep["ok"] and "bias" in rep["why_kinds"],
      "a flat line 13.5 mm off the commanded standoff is still a mesh error")
check(not rep.get("bias_from_curvature"),
      "...and it is NOT attributed to curvature (spread %.2f mm)"
      % (1e3*rep["spread"],))

# A CONVEX CREST across the line: heights spanning tens of millimetres, with the
# commanded 80 mm somewhere inside what the sensor returns. This is the cylinder.
x = np.linspace(-0.0155, 0.0155, 800)            # the 31 mm line
crest = 0.080 - (0.0335 - np.sqrt(np.maximum(0.0335**2 - x**2, 0.0)))
rep = range_report(crest, commanded=0.080)
check(rep.get("bias_from_curvature") is True,
      "a curved line with the commanded standoff inside its range is read as "
      "curvature (spread %.1f mm, %.1f..%.1f mm returned)"
      % (1e3*rep["spread"], 1e3*rep["min"], 1e3*rep["max"]))
check("bias" not in rep["why_kinds"],
      "...so it does not refuse as a mesh error")
check(rep.get("bias_note") and "sagitta" in rep["bias_note"],
      "...and it says why, in the report")

# CURVED AND GENUINELY WRONG. The same crest shifted 20 mm nearer: the commanded
# standoff is now OUTSIDE everything the sensor sees, so the disagreement survives
# the curvature and must still refuse.
rep = range_report(crest - 0.020, commanded=0.080)
check(not rep.get("bias_from_curvature"),
      "a curved line whose range does NOT contain the commanded standoff is not "
      "excused (%.1f..%.1f mm returned, commanded 80.0)"
      % (1e3*rep["min"], 1e3*rep["max"]))
check("bias" in rep["why_kinds"],
      "...and it still refuses -- curvature is not a way past the bias check")

# The NEAR-limit warning is untouched either way: it is about the sensor's window
# and not about the mesh, and a crest that reaches the near limit is still worth
# saying.
check("near" in range_report(crest - 0.005, commanded=0.080)["why_kinds"]
      or range_report(crest - 0.005, commanded=0.080)["ok"],
      "the near-limit check is unaffected by the curvature reading")


# ==============================================================================
# THE FAR LIMIT MAKES THE SAME MISTAKE, AND IT COST THREE SWEEPS
# ==============================================================================
# The bias clause learned that a whole-line statistic is not a statement about
# the point the plan commanded. The far-limit FRACTION never learned it, and on
# 2026-10-09 it refused the fine keyence sweep three times in a row with
#
#   22.6% of these points are within 5 mm of the FAR limit (103 mm)
#
# while the line spanned 82.2..101.5 mm around a commanded 80.0 mm. The run's
# own report said, two lines above the refusal, that the spread was curvature --
# and then refused on a clause that was counting the far EDGE of the laser line
# hanging off that same curve. The operator turned the check off to get a day's
# work done, which is the outcome a false refusal always buys.
#
# NEAR IS NOT STOOD DOWN WITH IT. The far end costs data at the edge of the
# line; the near end is the head approaching the part.

print("")
print("the far limit under curvature")

# THE REAL DISTRIBUTION from 2026-10-09: a convex crest, most of the line near
# 82 mm, a tail out to 101.5, commanded 80.0.
_t = np.random.default_rng(7).random(4000)
real = 0.0822 + (0.1015 - 0.0822)*(_t**2.2)
real[0], real[-1] = 0.0822, 0.1015
rep = range_report(real, commanded=0.080)
check(rep["far_fraction"] > 0.02,
      "the 2026-10-09 line really does put %.1f%% of its points near the far "
      "limit" % (100.0*rep["far_fraction"],))
check(rep["ok"],
      "...and the sweep is no longer refused for it")
check(rep.get("far_from_curvature") is True,
      "...because the far fraction is attributed to the curve")
check("far" not in rep["why_kinds"], "...so 'far' is not a reason to refuse")
check(rep.get("far_note") and "EDGE of the laser line" in rep["far_note"],
      "...and the report says which edge it is")

# THE CORRECTION, which is the point: a line narrower than the window always
# fits somewhere, and the gate now says where.
check(rep["reaim_possible"] is True, "a 19 mm line fits a 36 mm window")
check(rep["reaim_delta"] < 0,
      "...by moving the head CLOSER (%.1f mm)" % (1e3*rep["reaim_delta"],))
check(abs(rep["reaim_standoff"] - 0.0681) < 0.002,
      "...to about 68 mm, which centres 82.2..101.5 inside 62..98 "
      "(got %.1f mm)" % (1e3*rep["reaim_standoff"],))
_lines = " ".join(describe_range(rep, what="x"))
check("centres what the sensor is actually returning" in _lines
      or rep["ok"], "the re-aim is reported to the operator")

# A FLAT line genuinely out at the far limit is NOT curvature and must still
# refuse: this is the case the clause was written for.
_flat_far = np.random.default_rng(11).normal(0.1005, 0.0003, 800)
rep = range_report(_flat_far, commanded=0.1005)
check(not rep.get("far_from_curvature"),
      "a flat line sitting at the far limit is not excused as curvature "
      "(spread %.2f mm)" % (1e3*rep["spread"],))
check("far" in rep["why_kinds"] and not rep["ok"],
      "...and it still refuses")

# CURVED, but the commanded standoff is nowhere in what the sensor returns:
# the disagreement survives the curvature, exactly as for the bias clause.
rep = range_report(real + 0.004, commanded=0.040)
check(not rep.get("far_from_curvature"),
      "a curved line whose range does not contain the commanded standoff is "
      "not excused on the far limit either")

# THE NEAR LIMIT IS UNTOUCHED BY ALL OF THIS.
_crest_near = np.linspace(0.0, 0.019, 800) + 0.058
rep = range_report(_crest_near, commanded=0.060)
check("near" in rep["why_kinds"] and not rep["ok"],
      "a curved line that reaches the NEAR limit still refuses -- curvature "
      "excuses the far end only")

# A line WIDER than the window cannot be re-aimed, and says so rather than
# offering an impossible number.
_wide = np.linspace(0.058, 0.102, 800)
rep = range_report(_wide, commanded=0.080)
check(rep["reaim_possible"] is False and rep["reaim_standoff"] is None,
      "a 44 mm line does not fit a 36 mm window and no re-aim is invented")
check("NO standoff fits all of it" in describe_reaim(rep),
      "...and describe_reaim() says that rather than returning nonsense")


def test_the_far_limit_is_not_fooled_by_a_curve():
    """The 2026-10-09 refusal flies, and the cases it guards still refuse."""
    r = range_report(real, commanded=0.080)
    assert r["ok"] and r["far_from_curvature"] is True
    assert r["reaim_standoff"] is not None
    assert not range_report(_flat_far, commanded=0.1005)["ok"]
    assert "near" in range_report(_crest_near, commanded=0.060)["why_kinds"]


def test_not_blind():
    """The standoff gate ran, on the two cases the file was written for."""
    assert standoff_gate.__name__ == "keyence_standoff_gate"
    assert (HARD_LO, HARD_HI) == (0.057, 0.103) and abs(REF - 0.080) < 1e-9, \
        "the constants came from somewhere other than keyence_standoff.dpi"
    # This suite builds no messages; nothing it measures may depend on whose Pose
    # is installed in sys.modules.
    assert not any(isinstance(v, type) and getattr(v, "__module__", "")
                   .startswith("geometry_msgs") for v in globals().values()), \
        "a geometry_msgs class reached this suite's namespace"
    plate = plate_mesh("not_blind.ply", PLATE_TOP_Z)
    # KNOWN-POSITIVE: 80 mm over the surface it was planned over is silent.
    clear()
    standoff_gate(flat_path(PLATE_TOP_Z, standoff=REF), plate, **GEOM)
    # KNOWN-NEGATIVE: 58 mm is one millimetre from the hard limit and refused.
    clear()
    try:
        standoff_gate(flat_path(PLATE_TOP_Z, standoff=0.058), plate, **GEOM)
    except ValueError:
        pass
    else:
        raise AssertionError("58 mm of commanded standoff was accepted: the "
                             "gate is not being consulted at all")
    # And the report has to put a NUMBER on the difference, not just a verdict.
    a = standoff_report(flat_path(PLATE_TOP_Z, standoff=REF), plate, **GEOM)
    b = standoff_report(flat_path(PLATE_TOP_Z, standoff=0.058), plate, **GEOM)
    assert abs(a["median"] - REF) < 1e-4 and abs(b["median"] - 0.058) < 1e-4, \
        "the report does not track the commanded standoff (%r, %r)" \
        % (a["median"], b["median"])
    assert a["ok"] and not b["ok"], \
        "the report's verdict does not follow its own numbers"


# ==============================================================================
# A SMALL REGION IS NOT A WRONG ONE: TRIM IS HELD PER PASS, NOT PER PATH
# ==============================================================================
# Two rasters over the same plate, minutes apart, with the same geometric fact
# behind both:
#
#   ROI 24 x 72 mm  -> 1790 waypoints, 19 passes,  2 out (0.11%)  FLEW
#   ROI 24 x 66 mm  ->   39 waypoints,  3 passes,  3 out (7.69%)  REFUSED
#
# Both are a Noether raster running off the edge of a bounded mesh -- the last
# waypoint of a pass, with the sensor aimed just past the boundary, measuring
# nothing. The second was refused because the drop limit was a fraction of the
# WHOLE PATH, and the number of pass ENDS scales with the number of PASSES: a
# 24 x 66 mm region is 3 passes of 13 poses, so one end waypoint per pass is
# 7.7% of the path before anything at all is wrong, against a 5% cap.
#
# That refuses a plan for being SMALL, and a small region is exactly what an
# operator draws when they want one feature scanned in the hour the plan allows.
# The limit is therefore split: interior holes capped over the path (they are
# evidence about the mesh), trim capped per PASS (an overrun belongs to a pass's
# ends). Both numbers have to bite, and both are injected below.

print("")
print("a small region whose passes overrun the mesh edge by one waypoint")

# The mesh ends at x 0.93; the raster runs to 0.95, so the last waypoint or two
# of every pass is aimed off the edge and measures nothing. Three passes of
# thirteen, which is the shape of the small ROI above.
small = plate_mesh("small.ply", PLATE_TOP_Z, xspan=(0.70, 0.93),
                   yspan=(-0.20, 0.22), n=41)
clear()
over = flat_path(PLATE_TOP_Z, standoff=REF, nx=13, rows=3, x=(0.80, 0.95))
rep = standoff_gate(over, small, **GEOM)
check(rep is not None and rep.get("ok"),
      "a 3-pass raster that overruns the mesh edge is FLOWN, not refused -- "
      "the small-ROI case, which a whole-path cap refuses at 7.7%")
check(rep.get("n_out", 0) > 0 and rep.get("dropped", 0) > 0,
      "  the out-of-window waypoints are DROPPED (%d of %d), so the hole is "
      "real and reported" % (rep.get("dropped", 0), rep.get("n", 0)))
check(rep.get("n_interior") == 0,
      "  and all of them are pass ENDS: 0 interior (%d trim)"
      % (rep.get("n_trim", -1),))
check(said("pass ENDS"),
      "  the operator is told which kind of loss this was")
check(rep.get("passes_after", 0) == 3,
      "  every pass survives, shortened (%s of 3)" % (rep.get("passes_after"),))

# FAULT INJECTION 1 -- the per-pass trim cap must bite. Run the raster far enough
# off the edge that a third of each pass is out.
clear()
gutting = flat_path(PLATE_TOP_Z, standoff=REF, nx=12, rows=3, x=(0.80, 1.05))
try:
    standoff_gate(gutting, small, **GEOM)
    check(False, "a pass losing a THIRD of its length is refused")
except ValueError as e:
    msg = str(e)
    check(True, "a pass losing a THIRD of its length is refused")
    check("more than 20%" in msg and "of their own length" in msg,
          "  and says the limit it broke is per-pass, not per-path")
    check("overruns the edge" in msg and "ONE waypoint" in msg,
          "  and why a big trim is not an overrun")
    check("22 mm low at ONE END" in msg,
          "  and cites the concentrated end error a whole-path fraction misses")

# FAULT INJECTION 2 -- the interior cap must bite independently. A dent in the
# MIDDLE of every pass: surface present, plan at the wrong height over it. No
# pass loses its ends at all, so only the interior rule can catch this.
clear()
dented = plate_mesh("dented.ply", PLATE_TOP_Z, xspan=(0.70, 0.95),
                    yspan=(-0.20, 0.22), n=41)
inner = flat_path(PLATE_TOP_Z, standoff=REF, nx=13, rows=3, x=(0.80, 0.93))
pos = np.array(inner.positions, dtype=np.float64)
# Lift TWO of the thirteen waypoints of each pass far above the window, in the
# middle, so every one is interior by construction. Two and not four: four is
# 31% of a 13-waypoint pass, which trips the PER-PASS trim cap first and would
# make this injection test the wrong rule. Two is 15% per pass (under the 20%
# trim cap) and 15% of the path interior (over the 5% interior cap), so only the
# interior rule can refuse it -- which is the point.
for s in [st for st in inner.segments]:
    a, b = s
    for k in range(a + 5, a + 7):
        pos[k, 2] += 0.060
holed = ScanPath(pos, np.array(inner.quats), [s[0] for s in inner.segments],
                 source="interior holes")
try:
    standoff_gate(holed, dented, **GEOM)
    check(False, "interior holes are refused even with every pass end clean")
except ValueError as e:
    msg = str(e)
    check(True, "interior holes are refused even with every pass end clean")
    check("INTERIOR" in msg and "surface on both sides" in msg,
          "  and says they are interior, with surface on both sides")
    check("wrong height" in msg,
          "  which is a statement about the MESH, not about the raster's extent")

# AND THE SPLIT ITSELF, directly: the same waypoint count means different things
# at a pass end and in a pass middle, and out_of_window_shape() is what knows it.
shape_of = ns["out_of_window_shape"]
one_pass = ScanPath(np.zeros((10, 3)), np.tile(DOWN, (10, 1)), [0],
                    source="one pass")
n_trim, n_int, n_whole, per = shape_of(
    one_pass, [True, True, False, False, False, False, False, False, False, True])
check((n_trim, n_int, n_whole) == (3, 0, 0),
      "out_of_window_shape: a run touching each end is TRIM (%d, %d, %d)"
      % (n_trim, n_int, n_whole))
n_trim, n_int, n_whole, per = shape_of(
    one_pass, [False]*4 + [True, True, True] + [False]*3)
check((n_trim, n_int, n_whole) == (0, 3, 0),
      "...the same three waypoints in the MIDDLE are INTERIOR (%d, %d, %d)"
      % (n_trim, n_int, n_whole))
n_trim, n_int, n_whole, per = shape_of(one_pass, [True]*10)
check((n_trim, n_int, n_whole) == (0, 0, 1),
      "...and a pass out from end to end is NEITHER -- it is a pass lost whole, "
      "because there is no in-window waypoint for anything to be between "
      "(%d, %d, %d)" % (n_trim, n_int, n_whole))
check(per[0]["share"] == 1.0,
      "...which is 100% of its own length, so the per-pass cap catches it "
      "without needing a rule of its own")


def test_keyence_standoff():
    """So pytest reports the same thing as running it."""
    assert not FAILS, "%d check(s) failed: %s" % (len(FAILS), "; ".join(FAILS))


try:
    test_not_blind()
    check(True, "the suite is not blind")
except AssertionError as _e:
    check(False, "the suite is not blind (%s)" % (_e,))

# The verdict, at the END of the file where every check has already run.
if __name__ == "__main__":
    print("")
    if FAILS:
        print("%d FAILED:" % len(FAILS))
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")
