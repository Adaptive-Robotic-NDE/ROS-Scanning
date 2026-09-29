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
# File:  cell_capture_clearance_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the capture tool's cylinder-pass clearance against the whole
#   tooling rather than the probe tip alone.
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
"""Check the capture tool's cylinder-pass clearance against the WHOLE tooling.

Refusing a cylinder pass on one number -- the distance from `probe_end` to the
cylinder's axis, minus its radius -- is the probe TIP against an analytic
cylinder, and it cannot see the other three things bolted to the same plate: the
barrel the tip belongs to, the Keyence head, the Azure and its bracket. That
blind spot puts the eddy-current probe 5-6 mm inside the fiducial cylinder on
the pipeline's own raster, which is why the pipeline stage goes through
`tooling_clearance.dpi`. The capture tool's cylinder passes are its OWN motions,
planned here, so they need the same routing.

Runs on the host: numpy only, the tooling geometry is URDF text written here, the
cylinder is generated, and no ROS, robot or container is involved.

    python3 cell_capture_clearance_test.py     # prints every check, exit 1 on failure
    python3 -m pytest -q cell_capture_clearance_test.py

Both modes run the same checks. Standalone runs every section even after one
fails, so one run tells you everything that is wrong; under pytest each section
is its own test and the checks that failed are named in the assertion.
"""
import functools
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# toolpath_segments.dpi imports geometry_msgs at module scope, so the name has to
# resolve on a host with no ROS. Planting stub modules directly with
# `sys.modules[...] = ...` is what ros_msg_stubs.py exists to prevent: a
# field-less Point or Quaternion installed here is shared with every other suite
# of a full `pytest` run, and a suite that builds messages then passes every case
# while measuring nothing. ensure_msgs() only ADDS -- and nothing here builds a
# message anyway: the poses below are numpy arrays and quaternion tuples
# throughout.
ros_msg_stubs.ensure_msgs()


def _load(name):
    src = open(os.path.join(HERE, name)).read()
    return "\n".join(l for l in src.splitlines() if not l.startswith("global "))


ns = {"__name__": "cell_capture_clearance_test"}
for _f in ("toolpath_segments.dpi", "noether_path.dpi", "tooling_clearance.dpi"):
    exec(compile(_load(_f), _f, "exec"), ns)

# The capture part-5 file is defs and imports only, so it loads with nothing
# else present; the three functions under test take everything else as
# arguments. _capture_say is the one module-level name they reach for.
_SAID = []
ns["_capture_say"] = _SAID.append
exec(compile(_load("cell_capture_pipeline.dpi"), "cell_capture_pipeline.dpi", "exec"), ns)

matrix_to_quat = ns["matrix_to_quat"]
quat_to_matrix = ns["quat_to_matrix"]
tooling_envelope = ns["tooling_envelope"]
SurfaceDistance = ns["SurfaceDistance"]
_cap_cylinder_mesh = ns["_cap_cylinder_mesh"]
_cap_pass_clearance = ns["_cap_pass_clearance"]
_cap_tooling_envelope = ns["_cap_tooling_envelope"]

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok    %s%s" % (name, ("  -- " + detail) if detail else ""))
    else:
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))
        FAILURES.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# The same fixture as tooling_clearance_test.py: probe_end is the tip of a
# 7.1 mm barrel, the Keyence body sits 75 mm across and 34 mm up, and the arm is
# above (probe_end +z points back toward it).
TOOLING_URDF = """<?xml version="1.0"?>
<robot name="test_tooling">
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
  <link name="sensor_body">
    <collision>
      <origin xyz="0 0 0.06" rpy="0 0 0"/>
      <geometry><box size="0.06 0.03 0.04"/></geometry>
    </collision>
  </link>
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
</robot>
"""

OFF = np.array([-0.07515, -0.0022, -0.03428])
R_CYL = 0.03169          # CAPTURE_CYLINDER_RADIUS, the fiducial cylinder
L_CYL = 0.200
AXIS_Z = 0.950


def taught_cylinder():
    """What _cap_cylinder() builds: axis along world +x, crest up, on a bench."""
    u = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(u, up)
    a = np.array([0.70, 0.0, AXIS_Z + R_CYL])
    return {"a": a, "b": a + L_CYL*u, "R": R_CYL, "u": u, "up": up, "side": side,
            "axis0": a - R_CYL*up, "L": L_CYL}


def tool_quat(z, x_hint):
    z = np.asarray(z, float)/np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z)*z
    x = x/np.linalg.norm(x)
    return tuple(float(v) for v in matrix_to_quat(np.column_stack([x, np.cross(z, x), z])))


def probe_for_optical(optical, q, off=OFF):
    return np.asarray(optical, float) - quat_to_matrix(q).dot(off)


def pass_at(c, deg, margin_mm=10.0, sink_mm=0.0):
    """The pass the capture tool plans at this angle round the axis.

    `sink_mm` pushes the sensor that far INTO the part, which is the fault
    condition the check exists for.
    """
    th = math.radians(deg)
    m = math.cos(th)*c["up"] + math.sin(th)*c["side"]
    t0, t1 = margin_mm/1000.0, c["L"] - margin_mm/1000.0
    r = c["R"] - sink_mm/1000.0
    s0 = c["axis0"] + t0*c["u"] + r*m
    s1 = c["axis0"] + t1*c["u"] + r*m
    q = tool_quat(m, c["u"])
    return probe_for_optical(s0, q), probe_for_optical(s1, q), q


def tip_only(p0, p1, c):
    """The check as it was: probe_end against the analytic cylinder."""
    return min(float(np.linalg.norm(np.cross(p - c["axis0"], c["u"])))
               for p in (p0, p1)) - c["R"]


# The fixture every section shares. None of it asserts anything, so it is safe
# at import time -- which is what lets pytest collect the sections below as
# ordinary, independent tests.
c = taught_cylinder()
verts, tris = _cap_cylinder_mesh(c)
env = tooling_envelope(source=TOOLING_URDF, root="probe_end", rigid_root="tool0")


# ------------------------------------------------------------------------------
# 1. The generated surface is the taught cylinder
# ------------------------------------------------------------------------------
@_checked
def test_the_cylinder_becomes_a_surface():
    print("\n1. the cylinder becomes a surface")
    radial = np.linalg.norm(np.cross(verts - c["axis0"], c["u"]), axis=1)
    check("every vertex is on the cylinder",
          abs(radial - c["R"]).max() < 1e-12,
          "worst %.3g m off a %.2f mm radius" % (abs(radial - c["R"]).max(), c["R"]*1e3))
    along = (verts - c["axis0"]).dot(c["u"])
    check("it spans the taught length and no more",
          along.min() > -1e-12 and abs(along.max() - c["L"]) < 1e-12,
          "%.1f..%.1f mm" % (along.min()*1e3, along.max()*1e3))
    check("the triangles close the grid",
          tris.shape == (2*96*40, 3) and tris.max() == verts.shape[0] - 1,
          "%d triangles over %d vertices" % (tris.shape[0], verts.shape[0]))
    surf = SurfaceDistance(verts, tris, outward=c["up"], name="cylinder")
    check("its normals point out of the part",
          surf.normal_disagreement < 0.30,
          surf.describe())
    probe = c["axis0"] + 0.5*c["L"]*c["u"] + (c["R"] + 0.010)*c["up"]
    inside = c["axis0"] + 0.5*c["L"]*c["u"] + (c["R"] - 0.010)*c["up"]
    d_out = float(surf.signed_distance([probe])[0])
    d_in = float(surf.signed_distance([inside])[0])
    # A facetted cylinder sits INSIDE the true one by the chord sag, R(1-cos(half
    # the facet angle)) = 7.5 um here, so it reads a hair clear. That is 270x
    # smaller than the 2 mm sampling pad the check already subtracts.
    check("outside reads positive and inside reads negative",
          abs(d_out - 0.010) < 2e-5 and abs(d_in + 0.010) < 2e-5,
          "+10 mm -> %+.4f mm, -10 mm -> %+.4f mm (facet sag %.1f um)"
          % (d_out*1e3, d_in*1e3,
             c["R"]*(1.0 - math.cos(math.radians(240.0/96/2)))*1e6))

# ------------------------------------------------------------------------------
# 2. With no envelope it is the old check, and it says so
# ------------------------------------------------------------------------------
@_checked
def test_the_fallback_is_the_old_tip_only_number():
    print("\n2. the fallback is the old tip-only number, not a silent pass")
    p0, p1, q = pass_at(c, 40.0)
    gap, link, how = _cap_pass_clearance(p0, p1, q, c, mesh=None, envelope=None)
    check("fallback equals the old formula exactly",
          abs(gap - tip_only(p0, p1, c)) < 1e-15,
          "%+.2f mm" % (gap*1e3,))
    check("and it is labelled as tip only", (link, how) == ("probe tip", "tip only"))
    _SAID[:] = []
    check("a missing description warns instead of pretending",
          _cap_tooling_envelope() is None and _SAID
          and "TIP only" in _SAID[0] and "WARNING" in _SAID[0],
          (_SAID[0][:70] + "...") if _SAID else "nothing said")


# ------------------------------------------------------------------------------
# 3. The whole tooling is measured, and it can only be tighter than the tip
# ------------------------------------------------------------------------------
@_checked
def test_the_whole_tooling_is_never_looser_than_the_tip():
    print("\n3. the whole tooling, over the angles the capture tool flies")
    worst_delta = 0.0
    how = None
    for deg in (0, 20, -20, 40, -40):
        p0, p1, q = pass_at(c, deg)
        gap, link, how = _cap_pass_clearance(p0, p1, q, c, mesh=(verts, tris), envelope=env)
        tip = tip_only(p0, p1, c)
        worst_delta = max(worst_delta, tip - gap)
        check("%+d deg: whole tooling %+.1f mm <= tip %+.1f mm (%s)"
              % (deg, gap*1e3, tip*1e3, link), gap <= tip + 1e-9)
    check("the whole-tooling check is genuinely stricter somewhere",
          worst_delta > 0.001, "up to %.1f mm tighter than the tip alone" % (worst_delta*1e3,))
    check("it reports which link, not just a number", how == "whole tooling")


# ------------------------------------------------------------------------------
# 4. It fires when the tooling is in the part
# ------------------------------------------------------------------------------
@_checked
def test_a_pass_inside_the_part_is_refused():
    print("\n4. a pass that is in the part comes back negative")
    p0, p1, q = pass_at(c, 40.0, sink_mm=45.0)
    gap, link, how = _cap_pass_clearance(p0, p1, q, c, mesh=(verts, tris), envelope=env)
    check("sunk 45 mm: refused", gap < 0.005, "%+.1f mm at %s" % (gap*1e3, link))
    check("and the offending link is named", link in ("ec_probe", "sensor_body", "wrist_link"), link)
    tip = tip_only(p0, p1, c)
    check("the tip-only number is less alarmed than the truth",
          tip > gap, "tip %+.1f mm vs tooling %+.1f mm" % (tip*1e3, gap*1e3))


# ------------------------------------------------------------------------------
# 5. The test is not blind
# ------------------------------------------------------------------------------
# Shared stub modules cut both ways: ensure_msgs() may hand THIS suite a module
# another suite wrote (ros_msg_stubs.py). So prove the subject still runs and
# still discriminates rather than trusting that it does.
@_checked
def test_not_blind():
    print("\n5. the test is not blind")
    check("the subject came out of cell_capture_pipeline.dpi, not a stand-in",
          _cap_pass_clearance.__code__.co_filename == "cell_capture_pipeline.dpi"
          and _cap_cylinder_mesh.__code__.co_filename == "cell_capture_pipeline.dpi")
    check("this suite builds no ROS message, so no other suite's stub can "
          "decide what it measures",
          not any(isinstance(v, type)
                  and getattr(v, "__module__", "").startswith("geometry_msgs")
                  for v in list(globals().values())))
    # A KNOWN-POSITIVE AND A KNOWN-NEGATIVE side by side: the same call, one pass
    # clear of the cylinder and one driven 45 mm into it. A check that only ever
    # sees good input cannot tell that it is measuring anything.
    p0, p1, q = pass_at(c, 40.0)
    clear_gap = _cap_pass_clearance(p0, p1, q, c, mesh=(verts, tris),
                                    envelope=env)[0]
    p0s, p1s, qs = pass_at(c, 40.0, sink_mm=45.0)
    sunk_gap = _cap_pass_clearance(p0s, p1s, qs, c, mesh=(verts, tris),
                                   envelope=env)[0]
    check("a clear pass reads %+.1f mm and one sunk 45 mm reads %+.1f mm, so "
          "the clearance is responding to its input"
          % (clear_gap*1e3, sunk_gap*1e3),
          clear_gap > 0.005 > sunk_gap)
    # And the envelope really is the tooling, not an empty cloud that would make
    # every pass look clear.
    names = sorted(l["name"] for l in env.links)
    check("the envelope carries the whole tooling (%s)" % (", ".join(names),),
          {"ec_probe", "sensor_body", "wrist_link"} <= set(names))


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    print(__doc__.splitlines()[0])
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            # __wrapped__ is the undecorated section: it records failures
            # instead of raising, so one run reports all of them.
            getattr(fn, "__wrapped__", fn)()
    print("\n%d checks failed" % len(FAILURES))
    for f in FAILURES:
        print("  - %s" % f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(_main())
