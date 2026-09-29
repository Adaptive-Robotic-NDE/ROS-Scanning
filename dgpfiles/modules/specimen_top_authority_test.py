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
# File:  specimen_top_authority_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that the pipeline's idea of the part's top comes from the mesh
#   and not from a touch-off.
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
"""The automated pipeline's idea of the part's top owes NOTHING to a touch-off.

WHAT THIS PINS. `_check_reached_the_top()` is the guard that catches a rough
mesh which did not reach the top of the part -- the surface everything
downstream plans over. Its authority chain is exactly:

    the caller's expect_top_z  ->  the detector's surface_z  ->  nothing

and the line it prints says which one it used. No taught point is read, at all.
Put a taught point anywhere -- inside the region, above the part, below it --
and not one byte of this function's behaviour changes. That is the property
every check below exists to hold down.

WHY IT IS A PROPERTY AND NOT A PREFERENCE. Preferring the highest CONTACT
touch-off inside the region is not a safe way to find the part's top, because a
touch-off is taught for whatever purpose the person teaching it had. A
calibration point taught on the BENCH -- z 0.9219, against a part top of 0.943
-- can fall inside the detected region. Using it makes the guard

  * announce the BENCH as "the part's top",
  * print a +21.3 mm "the camera and the probe disagree" alarm that is nothing
    but the part's own height, and
  * hand reconstruction_covers() a top_z of 0.9219, which any mesh of a part
    standing on that bench clears trivially.

That neuters the check and raises a false alarm at the same time. Operator
touch-offs belong to calibration.dpi and to the cell_capture_* teach scenarios,
where a person is standing there to say what they touched. They have no place in
an unattended pipeline.

NO CAMERA-VERSUS-PROBE COMPARISON HERE. There is no second measurement in this
function to compare against, and inventing one would be the same mistake in a
new costume. Camera-against-reality is stage 2a's job --
gauge_mesh_against_keyence() in azure_gauge.dpi, which puts an independent
sensor on the surface. What this guard catches is COVERAGE: a sweep that stopped
short of the crest, a miss that runs to tens of millimetres and is the common
failure.

    python3 specimen_top_authority_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from specimen_scene_test import load

# A detected region as yolo_region() reports one: (x_min, y_min, x_max, y_max)
# in world metres.
REGION = (0.7560744107266263, -0.10576199110458422,
          0.8906979088875827, 0.18054951127101138)
DETECTOR_TOP = 0.9324          # yolo_region()'s surface_z -- an AZURE number

#: A deliberately awkward taught-point set: it spans 19 mm and straddles the
#: region edge. `cal_flat` is a bench-height touch-off (for verify_calibration)
#: that falls INSIDE the detected region and is the one most easily mistaken for
#: the part; flat/edge_a/edge_b are on the plate; the corners are the bench
#: survey and sit outside the region. None of them may reach this function.
BENCH_Z = 0.9219
PLATE_Z = 0.94016
TAUGHT = {
    "corner_1": {"xyz": [0.58699, 0.40110, 0.92134]},
    "corner_2": {"xyz": [1.12110, 0.40107, 0.92133]},
    "corner_3": {"xyz": [1.12108, -0.46380, 0.92132]},
    "corner_4": {"xyz": [0.60801, -0.46489, 0.91942]},
    "cal_flat": {"xyz": [0.80000, 0.00000, BENCH_Z]},
    "flat":     {"xyz": [0.80842, 0.02275, PLATE_Z]},
    "edge_a":   {"xyz": [0.76742, -0.06089, 0.93991]},
    "edge_b":   {"xyz": [0.76575, 0.10588, 0.93988]},
}

CHECKS, FAILURES = [], []


def check(what, cond, detail=""):
    CHECKS.append(what)
    if cond:
        print("  ok    %s%s" % (what, (" -- " + detail) if detail else ""))
    else:
        FAILURES.append(what)
        print("  FAIL  %s%s" % (what, (" -- " + detail) if detail else ""))


def _ns(taught=TAUGHT, detected=True, covers=True, seen=None):
    """specimen.dpi in a namespace, with the coverage measurement stubbed.

    `seen` collects the (top_z, footprint) reconstruction_covers() is actually
    handed -- the number under test. The measurement itself lives in
    mesh_reconstruction.dpi and has its own tests; this file is about WHICH
    NUMBER it is given.
    """
    def _covers(mesh, top, footprint=None, what=""):
        if seen is not None:
            seen.append((float(top), footprint))
        return (True, float(top), 1000) if covers else (False, BENCH_Z, 1000)

    ns = load(lambda *a, **k: None, extra={
        "taught_points": dict(taught),
        "detected_region": REGION if detected else None,
        "detected_info": {"surface_z": DETECTOR_TOP},
        "reconstruction_covers": _covers,
    })
    ns["PROGRESS_ENABLED"] = False
    return ns


def _say(ns, fn, *a, **kw):
    out = []
    ns["print"] = lambda *x: out.append(" ".join(str(v) for v in x))
    try:
        r = fn(*a, **kw)
    finally:
        ns.pop("print", None)
    return r, "\n".join(out)


def _top(ns, expect=None, require=False, seen=None):
    """(printed output, the top_z the coverage check was handed)."""
    _r, out = _say(ns, ns["_check_reached_the_top"], None, REGION, expect,
                   require)
    return out, (seen[-1][0] if seen else None)


def main():
    print("\n1. THE MODULE HAS NO TOUCH-OFF READER AT ALL")
    ns = _ns()
    check("_touched_off_top() does not exist", "_touched_off_top" not in ns)
    src = open(os.path.join(HERE, "specimen.dpi")).read()
    check("and the source names neither it nor taught_points",
          "_touched_off_top" not in src and "taught_points" not in src)
    check("nor is there a stray `global` left for it",
          "global _touched_off_top" not in src)
    # The floor of the work volume is the one exception: it may be read.
    check("the work volume is still read (the one allowed touch-off product)",
          "work_volume()" in src)

    print("\n2. A TAUGHT POINT INSIDE THE REGION CHANGES NOTHING")
    # The bench point is inside the region and 12 mm BELOW the detector's
    # number, so any "highest touch-off in the region" rule would hand the
    # coverage check 0.9219 instead of the part's top.
    inside = [p for p in TAUGHT.values()
              if REGION[0] <= p["xyz"][0] <= REGION[2]
              and REGION[1] <= p["xyz"][1] <= REGION[3]]
    check("the fixture really does put taught points inside the region",
          len(inside) == 4, "%d of %d" % (len(inside), len(TAUGHT)))
    check("including a bench-height one below the detector's top",
          BENCH_Z < DETECTOR_TOP, "%.4f < %.4f" % (BENCH_Z, DETECTOR_TOP))

    seen_with, seen_without = [], []
    out_with, top_with = _top(_ns(seen=seen_with), seen=seen_with)
    out_without, top_without = _top(_ns(taught={}, seen=seen_without),
                                    seen=seen_without)
    check("the top handed to the coverage check is the detector's",
          top_with is not None and abs(top_with - DETECTOR_TOP) < 1e-9,
          "%s" % (top_with,))
    check("NOT the bench touch-off, which is the tempting wrong answer",
          top_with is not None and abs(top_with - BENCH_Z) > 1e-6)
    check("and it is identical with no taught points at all",
          top_with == top_without, "%s vs %s" % (top_with, top_without))
    check("the printed line is identical too", out_with == out_without,
          out_with)

    # The worst arrangement: the ONLY taught point inside the region is the
    # bench, nothing having been touched on the part at all. "The highest
    # touch-off inside the region" would then be the bench, and the guard would
    # verify that the mesh reaches the bench -- which it does by standing on it.
    bench_only = {k: v for k, v in TAUGHT.items()
                  if k.startswith("corner") or k == "cal_flat"}
    seen_bench = []
    out_bench, top_bench = _top(_ns(taught=bench_only, seen=seen_bench),
                                seen=seen_bench)
    check("with only the BENCH taught inside the region, the top is still the "
          "detector's", top_bench == top_without,
          "%s vs %s" % (top_bench, top_without))
    check("the bench height never reaches the coverage check",
          all(abs(t - BENCH_Z) > 1e-6 for t, _f in seen_bench),
          "%s" % (seen_bench,))
    check("and the printed line is unchanged by it", out_bench == out_without,
          out_bench)

    print("\n3. A TAUGHT POINT ABOVE THE PART CHANGES NOTHING EITHER")
    # The other direction: a high taught point must not silently RAISE the bar
    # and fail a mesh that reached the part.
    high = dict(TAUGHT)
    high["oops"] = {"xyz": [0.80, 0.02, DETECTOR_TOP + 0.050]}
    seen_high = []
    out_high, top_high = _top(_ns(taught=high, seen=seen_high), seen=seen_high)
    check("a taught point 50 mm above the part is ignored",
          top_high == top_with, "%s vs %s" % (top_high, top_with))
    check("and says nothing about it", out_high == out_with, out_high)

    print("\n4. THE AUTHORITY CHAIN, AND IT SAYS WHICH IT USED")
    check("with no caller number it is the detector's surface_z",
          "the detector's surface_z (an AZURE measurement)" in out_with,
          out_with)
    check("and the number it names is the detector's",
          "0.9324" in out_with, out_with)
    seen_caller = []
    out_caller, top_caller = _top(_ns(seen=seen_caller), expect=0.9500,
                                  seen=seen_caller)
    check("an explicit expect_top_z wins -- the caller knows best",
          abs(top_caller - 0.9500) < 1e-9, "%s" % (top_caller,))
    check("and the line says so", "caller" in out_caller
          and "0.9500" in out_caller, out_caller)
    # Third rung: nothing. Not the taught points, even though they are there.
    seen_none = []
    ns = _ns(detected=False, seen=seen_none)
    r, out_none = _say(ns, ns["_check_reached_the_top"], None, REGION, None,
                       False)
    check("with no caller number and no detection it returns None",
          r is None, "%r" % (r,))
    check("and does not fall back to a taught point", not seen_none,
          "%s" % (seen_none,))
    check("saying nothing it cannot support", out_none == "", out_none)

    print("\n5. THERE IS NO 'CAMERA AND PROBE DISAGREE' ALARM HERE")
    # With no touch-off there is only one measurement in this function, so such
    # an alarm could only compare the detector with itself. The real
    # camera-against-reality check is stage 2a's.
    for label, text in (("with taught points", out_with),
                        ("with a high taught point", out_high),
                        ("with a caller number", out_caller)):
        check("no DISAGREE alarm %s" % label, "DISAGREE" not in text, text)
    check("and the source carries no such block at all",
          "DISAGREE" not in src)

    print("\n6. THE COVERAGE CHECK STILL BITES, against the detector's number")
    seen_short = []
    out_short, _t = _top(_ns(covers=False, seen=seen_short), seen=seen_short)
    check("a short mesh is reported", "SHORT of the part" in out_short,
          out_short)
    check("against the detector's top, not the bench's",
          "0.9324" in out_short and "%.4f" % BENCH_Z in out_short, out_short)
    ns = _ns(covers=False)
    raised = ""
    try:
        _say(ns, ns["_check_reached_the_top"], None, REGION, None, True)
    except RuntimeError as e:
        raised = str(e)
    check("and require_top=True raises instead of printing",
          "SHORT of the part" in raised, raised[:90])

    print("\n%d checks, %d failing" % (len(CHECKS), len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())


def test_top_authority():
    assert main() == 0, FAILURES
