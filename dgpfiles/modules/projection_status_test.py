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
# File:  projection_status_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-30
# Description:
#   What the projection-status report says in each of the three
#   states an empty projected image can be in.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-09-30:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""What the projection report says, in each state an empty image can be in.

AN EMPTY PROJECTION HAS THREE CAUSES AND THEY LOOK IDENTICAL FROM THE VIEWER.
Nothing was offered; everything was offered and every ray missed the surface; or
the image has data and the display is what is broken. They want completely
different fixes -- wire the projection, move the poses, restart the viewer -- and
the projector cannot tell them apart on its own, because a ray that misses is
dropped inside project_point_onto_parameterization without raising and the
errors that do raise are printed on the acquisition thread, where a block
buffered pipe may never be flushed.

So the report is the only thing standing between an empty image and an
undiagnosable run, and these tests pin what it says. They are about the WORDS:
each state has to name its own cause, and must not claim one of the others.

Runs on the HOST. describe_projection_status is lifted out of
surface_projection.dpi by its AST, so nothing here needs snde, a recording
database, a mesh or a robot.
"""

import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = os.path.join(HERE, "surface_projection.dpi")


def _describe():
    """Lift describe_projection_status() out of the .dpi and return it.

    A .dpi is a function body, not an importable module, and this one opens with
    snde imports that do not resolve on the host. Taking the one function by AST
    keeps the test to the thing under test.
    """
    tree = ast.parse(open(DPI).read(), filename="surface_projection.dpi")
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "describe_projection_status"]
    assert len(fn) == 1, "describe_projection_status is not defined exactly once"
    mod = ast.Module(body=fn, type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(mod), "surface_projection.dpi", "exec"), ns)
    return ns["describe_projection_status"]


def _status(**kw):
    base = {"offered": 0, "landed": None, "cells": None, "pose_nonfinite": 0,
            "value_nonfinite": 0, "pose_bbox": None, "channel": "/ec/proj/projection",
            "surface": "/data/spec_uv.x3d", "error": None}
    base.update(kw)
    return base


def test_nothing_offered_does_not_blame_the_projection():
    """project() never called is NOT a projection fault, and must not read as one.

    This is the state a config that is not in live mode sits in, and an empty
    image is then correct. Calling it a failure sends somebody to debug a
    working projector.
    """
    out = _describe()(_status(offered=0, landed=0, cells=4096))
    assert "NOTHING WAS OFFERED" in out
    assert "not a projection fault" in out
    assert "NONE LANDED" not in out


def test_offered_but_none_landed_says_the_rays_missed():
    """The silent case: every trace offered, nothing accumulated, no error.

    The report has to say the rays missed AND print the pose bbox, because
    comparing that against the surface extent is the whole diagnosis.
    """
    out = _describe()(_status(offered=6232, landed=0, cells=4096,
                              pose_bbox=((0.80, 0.02, 0.94), (0.84, 0.11, 0.95))))
    assert "NONE LANDED" in out
    assert "6232" in out
    assert "0.8000..0.8400" in out and "0.0200..0.1100" in out
    assert "no error was raised" in out


def test_landed_points_blame_the_display_instead():
    """Data in the image means the viewer is the broken half, and it says so.

    Without this the operator re-runs the scan to fix a display bug.
    """
    out = _describe()(_status(offered=6232, landed=512, cells=4096))
    assert "512 of 4096" in out
    assert "12.5%" in out
    assert "DISPLAY fault" in out
    assert "NONE LANDED" not in out


def test_unreadable_image_is_reported_as_unknown_not_as_empty():
    """If the channel cannot be read, the answer is 'unknown', never 'nothing'.

    Reporting an unreadable image as an empty one invents a diagnosis.
    """
    out = _describe()(_status(offered=100, error="KeyError: /ec/proj/projection"))
    assert "COULD NOT READ" in out
    assert "unknown" in out
    assert "NONE LANDED" not in out


def test_nonfinite_poses_are_called_out_wherever_they_occur():
    """A NaN pose cannot project, and it is the cause rather than a footnote."""
    out = _describe()(_status(offered=10, landed=0, cells=64, pose_nonfinite=3,
                              value_nonfinite=1))
    assert "3 pose(s) carried a NaN" in out
    assert "1 value(s) were not finite" in out
