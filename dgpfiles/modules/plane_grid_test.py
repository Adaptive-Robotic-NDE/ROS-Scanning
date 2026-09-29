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
# File:  plane_grid_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the triggered raster's geometry and the guard that refuses
#   overtravel past the taught edge.
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
"""Check the triggered raster's geometry: columns inside the taught edge, the
commanded row ending exactly on it, and the guard that refuses anything else.

_plane_grid and _check_overtravel_within_taught are pulled out of
plane_scan.dpi; scanpath_from_plane_grid out of toolpath_segments.dpi. Fake
TransformStamped corners, no ROS, no database.
"""
import ast
import os
import sys
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# NO geometry_msgs STUB, deliberately: pull() below execs only the
# FunctionDef/ClassDef nodes it names, so a .dpi's module-scope
# `from geometry_msgs.msg import ...` never runs, and none of the five things
# pulled out mentions a message type. Installing a stub here would put a fake
# module into sys.modules for every other suite in the directory -- see
# ros_msg_stubs.py.

ns = {"np": np}
def pull(path, names):
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
pull(os.path.join(HERE, "toolpath_segments.dpi"), {"ScanPath", "infer_segment_starts", "scanpath_from_plane_grid"})
pull(os.path.join(HERE, "plane_scan.dpi"), {"_plane_grid", "_check_overtravel_within_taught"})
_plane_grid = ns["_plane_grid"]
_check = ns["_check_overtravel_within_taught"]
scanpath_from_plane_grid = ns["scanpath_from_plane_grid"]

fails = []
def check(name, got, want):
    ok = (got == want)
    print("  %-60s %s" % (name, "ok" if ok else "FAIL  got=%s want=%s" % (got, want)))
    if not ok: fails.append(name)

def T(x, y, z):
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=types.SimpleNamespace(x=x, y=y, z=z),
        rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

# A tilted, slightly skewed taught rectangle: 60 mm along u, 40 mm along v.
t1 = T(0.50, 0.10, 0.30)
t2 = T(0.56, 0.10, 0.31)          # far edge, 60.8 mm away and climbing
t3 = T(0.50, 0.14, 0.30)
m, n = 50, 5
MARGIN = 0.5

print("=== end_margin=0 puts the last column on the taught far edge ===")
uu0, vv0, ww0, up0, vp0 = _plane_grid(t1, t2, t3, m, n)
row = float(np.linalg.norm([0.06, 0.0, 0.01]))
check("last column on the taught far edge", bool(abs(up0[-1] - row) < 1e-12), True)
check("grid pitch row/(m-1)", bool(abs((up0[1]-up0[0]) - row/(m-1)) < 1e-12), True)

print()
print("=== end_margin=0.5 pulls the columns in, not the rows ===")
uu, vv, ww, up, vp = _plane_grid(t1, t2, t3, m, n, end_margin=MARGIN)
pitch = up[1]-up[0]
check("pitch is row/(m-1+0.5)", bool(abs(pitch - row/(m-1+MARGIN)) < 1e-12), True)
check("last column half a pitch inside the edge", bool(abs(row - up[-1] - 0.5*pitch) < 1e-12), True)
check("first column still at the taught corner",
      bool(np.linalg.norm([uu[0,0]-0.50, vv[0,0]-0.10, ww[0,0]-0.30]) < 1e-12), True)
check("row spacing unchanged", bool(np.allclose(vp, vp0)), True)
print("    pitch %.4f mm (was %.4f); scan area gives up %.3f mm at the far edge"
      % (pitch*1e3, (up0[1]-up0[0])*1e3, 0.5*pitch*1e3))

print()
print("=== the commanded row ends ON the taught edge ===")
rot = t1.transform.rotation
sp = scanpath_from_plane_grid(uu, vv, ww, rot, serpentine=False, overtravel_pitches=MARGIN)
edge = np.array([uu0[m-1, 0], vv0[m-1, 0], ww0[m-1, 0]])
check("row 0 ends at the zero-margin grid's last column (the taught edge)",
      bool(np.linalg.norm(sp.positions[sp.segments[0][1]-1] - edge) < 1e-9), True)
check("m traces per row at the grid pitch", sp.trace_count(pitch, sp.segments), m*n)
try:
    _check(sp, t1, t2, t3, m, n)
    check("guard accepts matching margin and overtravel", True, True)
except RuntimeError as e:
    check("guard accepts matching margin and overtravel", str(e), "")

print()
print("=== and refuses anything that would go past it ===")
bad = scanpath_from_plane_grid(uu, vv, ww, rot, serpentine=False, overtravel_pitches=1.0)
try:
    _check(bad, t1, t2, t3, m, n)
    check("overtravel past the edge is refused", "accepted", "refused")
except RuntimeError as e:
    check("overtravel past the edge is refused", "taught far edge" in str(e), True)
short = scanpath_from_plane_grid(uu, vv, ww, rot, serpentine=False, overtravel_pitches=0.0)
try:
    _check(short, t1, t2, t3, m, n)
    check("a row that stops short of the edge is refused too (no overtravel at all)", "accepted", "refused")
except RuntimeError as e:
    check("a row that stops short of the edge is refused too (no overtravel at all)", True, True)


# ==============================================================================
# THE TEST IS NOT BLIND
# ==============================================================================
# The checks above run at IMPORT, which is what lets one pytest test stand for all
# of them -- and is also how a suite can print "ok" forever while measuring
# nothing. So: a known-positive and a known-negative through the same call, and
# proof that the subject loaded is the real one.

def test_not_blind():
    """The subject ran, it is the real one, and it tells two cases apart."""
    assert "_plane_grid" in ns and "end_margin" in _plane_grid.__code__.co_varnames, \
        "_plane_grid was not pulled out of plane_scan.dpi"
    assert "taught" in (_check.__doc__ or "") or _check.__name__ == \
        "_check_overtravel_within_taught"
    # KNOWN-NEGATIVE / KNOWN-POSITIVE: the margin has to move the pitch.
    a = _plane_grid(t1, t2, t3, m, n, end_margin=0.0)[3]
    b = _plane_grid(t1, t2, t3, m, n, end_margin=0.5)[3]
    assert a[-1] > b[-1] + 1e-6, \
        "end_margin changed nothing: the grid is not responding to its input"
    # And the guard has to refuse something, or "accepts" above proves nothing.
    good = scanpath_from_plane_grid(uu, vv, ww, t1.transform.rotation,
                                   serpentine=False, overtravel_pitches=MARGIN)
    bad = scanpath_from_plane_grid(uu, vv, ww, t1.transform.rotation,
                                  serpentine=False, overtravel_pitches=1.0)
    _check(good, t1, t2, t3, m, n)          # must not raise
    try:
        _check(bad, t1, t2, t3, m, n)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the overtravel guard accepted a pass past the "
                             "taught edge: it is not being consulted")
    # No stub of ours, and none of anybody else's reachable from here.
    assert "geometry_msgs" not in repr(sorted(ns)), "a message type leaked into ns"


def test_plane_grid():
    """pytest reports what the standalone run reports.

    Every check above runs at import. Without a test function that asserts on
    what they recorded, pytest collects nothing here and a broken check prints
    FAIL into the log while the run still passes.
    """
    assert not fails, "%d check(s) failed: %s" % (len(fails), "; ".join(fails))


print()
try:
    test_not_blind()
    check("the suite is not blind", True, True)
except AssertionError as _e:
    check("the suite is not blind (%s)" % (_e,), False, True)

if __name__ == "__main__":
    print()
    if fails:
        print("FAILURES:", fails); sys.exit(1)
    print("all plane grid checks passed")
