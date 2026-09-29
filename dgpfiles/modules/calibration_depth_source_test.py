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
# File:  calibration_depth_source_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that calibration verification can tell the raw Azure depth
#   image from the registered one.
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
"""verify_calibration() has to be able to tell the two Azure depth pipelines apart.

Over a sheet of white paper on an optics table the check reads:

    keyence  z = 0.9218 m   0.052 mm rms flat, 127200 points
    azure    z = 0.9100 m   0.690 mm rms flat,   1526 points
    AZURE IS -11.77 mm AND 6.00 mrad OFF THE KEYENCE          FAIL

The Azure's coverage there is FINE -- about 86% of the pixels the patch
subtends, once the image's real size (registered depth is at COLOUR resolution,
not the depth mode's 640x576) and _cal_depth_points's stride=4 are both
accounted for. Get either of those wrong and the same grab reads as "only 30%
of its pixels came back"; this file pins both.

What is NOT settled by that number is WHICH depth. The cell feeds verify

    /azure/depth_to_rgb/image_raw  with  /azure/rgb/camera_info

which is depth resampled into the colour camera's frame through the K4A's
factory depth-to-colour extrinsics. The depth camera's own image is a different
pipeline and a different answer. Grab both AT THE SAME POSE and they separate:

    raw agrees with the Keyence, registered does not  -> the registration
    both read the same -11.8 mm                       -> the depth itself

And the grab is SAVED, because the verify take records no Azure topic at all
(deliberately - registered depth is about 188 MB/s), so without it there is
nothing to re-examine afterwards. One image per source is about 6 MB.

Runs on the host, no ROS: the .dpi's top-level defs are lifted into a namespace
with the ROS-facing helpers stubbed.

    python3 calibration_depth_source_test.py
"""
import functools
import ast
import json
import os
import shutil
import sys
import tempfile

import types

import numpy as np

# sensor_msgs is imported INSIDE _cal_depth_points; stub it so the real function
# body runs on the host. Nothing here touches ROS beyond the name.
_sm = types.ModuleType("sensor_msgs")
_sm.msg = types.ModuleType("sensor_msgs.msg")
_sm.msg.Image = type("Image", (), {})
sys.modules.setdefault("sensor_msgs", _sm)
sys.modules.setdefault("sensor_msgs.msg", _sm.msg)

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []

LIFTED = ("_cal_azure_surface", "_cal_depth_points", "_cal_safe_name",
          "_cal_atomic_write", "verify_calibration")
CONSTS = ("CAL_VERIFY_DEPTH", "CAL_AZURE_RAW_TOPICS", "CAL_AZURE_DISTANCE")


def _lift():
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    tree = ast.parse(src)
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in LIFTED:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in CONSTS for t in n.targets):
            body.append(n)
    ns = {"np": np, "os": os, "json": json}
    exec(compile(ast.Module(body=body, type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    ns["_SRC"] = src
    return ns


NS = _lift()


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _checked(fn):
    """Make the checks a test recorded into a pytest pass or failure.

    check() RECORDS rather than raises, so one standalone run reports every
    false fact instead of stopping at the first. Under pytest that would be a
    test which passes while printing FAIL, i.e. a decorative file. The wrapper
    asserts on whatever the test added to FAILURES; the standalone driver calls
    fn.__wrapped__, so its behaviour is unchanged.
    """
    @functools.wraps(fn)
    def wrapper(*a, **k):
        mark = len(FAILURES)
        fn(*a, **k)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(map(str, failed)))
    return wrapper

class FakeMsg(object):
    class _H(object):
        class _S(object):
            sec, nanosec = 12, 34
        stamp = _S()
        frame_id = "rgb_camera_link"
    header = _H()
    encoding = "16UC1"


class FakeTake(object):
    def __init__(self, d):
        self.dir = d


def _install(ns, image, K, frame="rgb_camera_link", cam_z=1.4674):
    """Stub the ROS-facing helpers so the real code path can run on the host."""
    msg = FakeMsg()
    msg.header.frame_id = frame
    ns["_cal_one_message"] = lambda *a, **k: [msg]
    ns["_cal_decode_image"] = lambda m: image
    ns["_cal_intrinsics"] = lambda topic: (K, None, image.shape[1],
                                           image.shape[0], frame)
    T = np.eye(4)
    T[2, 3] = cam_z
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])      # optical: z down onto the bench
    ns["_cal_tf"] = lambda a, b: T
    return T


def _flat_depth(h, w, K, cam_z, surface_z):
    """A depth image of a level plane at `surface_z`, seen from `cam_z`."""
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2])/K[0, 0]
    y = (v - K[1, 2])/K[1, 1]
    # z along the optical axis to a plane (cam_z - surface_z) below
    return ((cam_z - surface_z)*np.ones_like(x)*1000.0).astype(np.float64)


@_checked
def test_the_stride_and_the_image_size_are_both_in_the_yield():
    """The two things that were got wrong. The yield must come from the IMAGE,
    not from what the depth mode implies, and must divide by stride**2."""
    ns = _lift()
    # registered depth: colour resolution, fx about 1024
    K = np.array([[1024.0, 0, 1024.0], [0, 1024.0, 768.0], [0, 0, 1.0]])
    cam_z, surf = 1.4674, 0.9215
    img = _flat_depth(1536, 2048, K, cam_z, surf)
    _install(ns, img, K)
    aim = np.array([0.0, 0.0, surf])
    pts, meta = ns["_cal_azure_surface"]("/d", "/i", aim, 0.045, "registered")
    print("\nregistered depth, 2048x1536, fx 1024, %.0f mm range"
          % (1e3*meta["range_m"],))
    print("    %d points over the patch, expected %s, yield %s"
          % (meta["n_patch"], meta["expected_sampled"],
             None if meta["yield"] is None else "%.2f" % meta["yield"]))
    check("the image size is read from the image", meta["image"] == [1536, 2048],
          str(meta["image"]))
    check("the stride is recorded", meta["stride"] == 4, str(meta["stride"]))
    check("a fully-valid image yields about 1.0",
          0.85 <= meta["yield"] <= 1.15, "%.3f" % meta["yield"])
    # ... and a real 1526-point patch is ~86% of what this patch subtends,
    # not the 30% a mis-scaled expectation gives.
    check("1526 of that expectation is about 86%, not 30%",
          0.7 <= 1526.0/meta["expected_sampled"] <= 1.05,
          "1526/%d = %.2f" % (meta["expected_sampled"],
                              1526.0/meta["expected_sampled"]))


@_checked
def test_the_grab_is_saved_next_to_the_take():
    ns = _lift()
    K = np.array([[1024.0, 0, 1024.0], [0, 1024.0, 768.0], [0, 0, 1.0]])
    cam_z, surf = 1.4674, 0.9215
    _install(ns, _flat_depth(1536, 2048, K, cam_z, surf), K)
    d = tempfile.mkdtemp()
    try:
        aim = np.array([0.0, 0.0, surf])
        _pts, meta = ns["_cal_azure_surface"]("/d", "/i", aim, 0.045,
                                              "registered", take=FakeTake(d))
        print("\nthe grab on disk")
        check("a file was written", meta.get("file") and os.path.exists(meta["file"]),
              str(meta.get("file")))
        z = np.load(meta["file"], allow_pickle=True)
        check("it carries the depth image", z["depth"].shape == (1536, 2048),
              str(z["depth"].shape))
        check("and the intrinsics", np.allclose(z["K"], K))
        check("and the camera pose", z["T_world_cam"].shape == (4, 4))
        m2 = json.loads(str(z["meta"]))
        check("and the metadata round-trips", m2["stride"] == 4 and
              m2["image"] == [1536, 2048], str(m2)[:80])
        check("no .part file is left behind",
              not any(f.endswith(".part") for f in
                      os.listdir(os.path.join(d, "azure_grabs"))))
    finally:
        shutil.rmtree(d, ignore_errors=True)


@_checked
def test_both_sources_are_grabbed_at_one_pose_and_compared():
    src = NS["_SRC"]
    body = src[src.index("def verify_calibration"):]
    print("\nwhat verify does with depth=")
    check("it takes a depth= argument", "depth=None" in body.split(")")[0] + ")")
    check("it validates the choice", "'registered', 'raw' or 'both'" in body)
    check("the registered grab is always taken",
          '_cal_azure_surface(depth_topic, info_topic' in body)
    check("the raw grab uses CAL_AZURE_RAW_TOPICS",
          'CAL_AZURE_RAW_TOPICS.get("depth")' in body)
    i_fly = body.index("verify: azure over the flat")
    i_raw = body.index("CAL_AZURE_RAW_TOPICS")
    i_key = body.index("verify: keyence over the flat")
    check("both grabs happen BEFORE the arm moves to the Keyence poses",
          i_fly < i_raw < i_key,
          "fly %d, raw %d, keyence %d" % (i_fly, i_raw, i_key))
    check("a missing raw topic is survivable, not fatal",
          "azure[raw]: NOT AVAILABLE" in body)
    check("the verdict still comes from the registered source by default",
          'if which_depth == "raw"' in body)


@_checked
def test_the_comparison_says_which_pipeline_is_at_fault():
    src = NS["_SRC"]
    print("\nthe conclusion it draws")
    check("it reports raw against the Keyence",
          "RAW DEPTH IS %+.2f mm OFF THE KEYENCE" in src)
    check("raw good + registered bad names the registration",
          "THE REGISTRATION IS THE FAULT" in src)
    check("both bad names the depth sensor",
          "BOTH PIPELINES AGREE" in src)
    check("and it admits when neither is settled",
          "still in play" in src)
    check("it tells you what to change",
          "manifest's azure depth topic at" in src)


@_checked
def test_the_azure_is_deliberately_not_in_the_recorded_sets():
    """188 MB/s is the reason, and it has to be written down or someone will
    'fix' it by adding the topic."""
    src = NS["_SRC"]
    i = src.index("def verify_calibration")
    body = src[i:src.index("with capture_scenario", i) + 400]
    print("\nwhy the topic is not recorded")
    check("the sets are still core+keyence",
          'sets=("core", "keyence")' in body, body[-200:])
    check("and the cost is stated", "188 MB/s" in src)
    check("with the grab named as the alternative", "azure_grabs" in src)


def main():
    for fn in (test_the_stride_and_the_image_size_are_both_in_the_yield,
               test_the_grab_is_saved_next_to_the_take,
               test_both_sources_are_grabbed_at_one_pose_and_compared,
               test_the_comparison_says_which_pipeline_is_at_fault,
               test_the_azure_is_deliberately_not_in_the_recorded_sets):
        getattr(fn, "__wrapped__", fn)()
    print("\n%s" % ("ALL OK" if not FAILURES else "%d FAILED: %s"
                    % (len(FAILURES), ", ".join(FAILURES))))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
