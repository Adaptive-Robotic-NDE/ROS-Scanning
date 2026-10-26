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
# File:  cell_capture_preview_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the gate that notices when nothing is receiving the capture
#   preview.
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
"""The preview gate notices when nobody is receiving the preview.

Without it a whole take can run with `absent: /capture/events,
/capture/preview` and publishers=0, and nothing says so until the take's
closing report -- every motion in it confirmed by an operator with no way to
know whether RViz was showing them.

This checks the guard, not the middleware: given publishers whose matched
subscription counts are known, does the tool say the right thing? Runs on the
host with no ROS.
"""
import functools
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class _Pub(object):
    def __init__(self, n):
        self.n = n

    def get_subscription_count(self):
        if self.n is None:
            raise RuntimeError("no middleware")
        return self.n


class _Node(object):
    def __init__(self, markers, display, preview):
        self.marker_pub = _Pub(markers)
        self.display_pub = _Pub(display)
        self.preview_pub = _Pub(preview)


def _load():
    """preview_audience() and _preview_audience_line() out of the .dpi."""
    src = open(os.path.join(HERE, "cell_capture_motion.dpi")).read()
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                    for l in src.splitlines())
    # The .dpi imports ROS message types inside the functions that use them, so
    # module-level exec needs no ROS. It does touch the config's namespace.
    ns = {"np": np, "_CAP": {}, "CELL_MODE": "sim", "CELL_NAME": "test",
          "ros": None, "live_pose": None}
    for name in ("geometry_msgs", "visualization_msgs", "moveit_msgs",
                 "trajectory_msgs", "builtin_interfaces", "sensor_msgs"):
        sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules.setdefault(name + ".msg", types.ModuleType(name + ".msg"))
    exec(compile(src, "cell_capture_motion.dpi", "exec"), ns)
    return ns


fails = []


def check(name, ok, detail=""):
    print("  %-62s %s" % (name, "ok" if ok else "FAIL  " + detail))
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
audience = ns["preview_audience"]
line = ns["_preview_audience_line"]


@_checked
def test_nothing_listening_is_reported():
    print("=== nothing listening at all ===")
    ns["capture_node"] = _Node(0, 0, 0)
    got = audience()
    check("all three topics report 0 subscribers",
          sorted(n for n, _w in got.values()) == [0, 0, 0], str(got))
    text = line()
    check("it says nobody is receiving", "NOBODY IS RECEIVING" in text, text)
    for topic in ("/capture/preview_markers", "/display_planned_path",
                  "/capture/preview"):
        check("it names %s" % topic, topic in text, text)
    check("it says what to check", "RViz" in text, text)
    # No shipped .rviz has a display for /capture/preview_markers, so "is RViz
    # up" is not the answer.
    check("it says how to make the markers appear",
          "MarkerArray display on /capture/preview_markers" in text, text)


@_checked
def test_rviz_up_recorder_missing():
    print("=== RViz up, recorder missing (the recorder is the only gap) ===")
    ns["capture_node"] = _Node(1, 1, 0)
    text = line()
    check("still warns", "NOBODY IS RECEIVING" in text, text)
    check("names only the recorder's topic",
          "/capture/preview " in text + " " and
          "/capture/preview_markers" not in text, text)


@_checked
def test_everything_listening_is_silent():
    print("=== everything listening ===")
    ns["capture_node"] = _Node(1, 1, 1)
    check("silent when the preview is being received", line() == "", line())


@_checked
def test_a_publisher_that_cannot_be_asked_counts_as_dark():
    print("=== the middleware itself is unhappy ===")
    ns["capture_node"] = _Node(None, None, None)
    text = line()
    check("a publisher that cannot be asked counts as dark",
          "NOBODY IS RECEIVING" in text, text)


@_checked
def test_a_missing_publisher_counts_as_dark():
    print("=== a node with no publishers at all (never constructed) ===")
    ns["capture_node"] = object()
    text = line()
    check("a missing publisher counts as dark", "NOBODY IS RECEIVING" in text, text)

    print("")


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILED: %s" % ", ".join(fails))
        return 1
    print("all preview-audience checks passed")
    return 0


_SECTIONS = ["test_nothing_listening_is_reported", "test_rviz_up_recorder_missing", "test_everything_listening_is_silent", "test_a_publisher_that_cannot_be_asked_counts_as_dark", "test_a_missing_publisher_counts_as_dark"]


if __name__ == "__main__":
    sys.exit(_main())
