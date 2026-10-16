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
# File:  calibration_carry_forward_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-15
# Description:
#   Regenerating the cell's calibration file must not un-calibrate a
#   sensor whose correction never passed through the JSON record.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-15:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""A sensor with no record keeps what the description in force carries.

WHAT WENT WRONG. calibration_xacro() regenerates the whole drop-in file from
the JSON record every time either sensor is calibrated, which is right: the
file and the record cannot be allowed to disagree. But the record is not the
only way a correction reaches the description. The Keyence's lateral pair was
solved offline and written into the cell's file by hand, with its reasoning in
the comment above it, and there is no JSON entry for it. So every Azure
calibration regenerated the file as "keyence not calibrated" and wrote

    <xacro:property name="keyence_mount_cal_xyz" value="0 0 0"/>

over +0.57 and +1.72 mm of measured offset. That shipped on 2026-10-15: the
Azure re-calibration was deployed, the bring-up restarted, and the fine mesh
of that visit was reconstructed with the profiler 1.8 mm from where it is.

Nothing downstream could see it. check_calibration_applied() iterates the
RECORD, so a sensor with no record is not checked; verify_calibration()
compares the Azure against the Keyence and would read the error as the Azure's.

THE RULE NOW. A sensor with no record keeps whatever is in force, read from
TF, and the file says where the number came from. With no record and no TF the
answer is still identity -- an uncalibrated cell gets an uncalibrated file --
but the note distinguishes "not calibrated" from "nothing to carry over".

Runs on the host, no ROS: the .dpi's top-level defs are lifted into a
namespace with the ROS-facing helpers stubbed.

    python3 calibration_carry_forward_test.py
"""
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibration_solve as cs                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []

LIFTED = ("calibration_xacro", "_cal_correction_in_force",
          "_cal_deployed_file", "_cal_fmt_xyz")

KEY_IN_FORCE = cs.properties_to_cal("0.00056576 0.00172072 0.00000000",
                                    "0 0 0")
AZURE_SOLVED = {
    "measured": "2026-10-15T18:10:11",
    "mount_cal": {"parent": "mounting_plate", "cal_link": "azure_mount_cal",
                  "xyz": [-0.00041667, 0.00096976, 0.00010657],
                  "rpy": [-0.00492105, 0.00965598, -0.00853483],
                  "old_xyz": [0.00066854, 0.00051648, 8.197e-05],
                  "move_mm": 1.1763},
}


def _lift(tf=None, deployed=None, record=None):
    """calibration_xacro() and its helpers, bound to stubs.

    `tf` is {(parent, child): 4x4} as the description in force would answer;
    `deployed` is the path of a cell package file, or None.
    """
    src = open(os.path.join(HERE, "calibration.dpi")).read()
    tree = ast.parse(src)
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in LIFTED]
    assert len(body) == len(LIFTED), [n.name for n in body]

    def _cal_tf(parent, child):
        if tf is None or (parent, child) not in tf:
            raise RuntimeError("no TF from %s to %s" % (parent, child))
        return tf[(parent, child)]

    ns = {"np": np, "os": os, "_cal_geom": lambda: cs, "_cal_tf": _cal_tf,
          "load_calibration": lambda: record,
          "_cal_site_urdf_dir": lambda: (deployed, "stub"),
          "CAL_MOUNTS": {
              "azure": {"parent": "mounting_plate",
                        "cal_link": "azure_mount_cal"},
              "keyence": {"parent": "keyence_metal",
                          "cal_link": "lj_v7080_mount_cal"}},
          "CELL_NAME": "ex-ur10e-1", "CELL_MODE": "hardware"}
    exec(compile(ast.Module(body=body, type_ignores=[]), "calibration.dpi",
                 "exec"), ns)
    return ns


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _keyence_xyz(text):
    props = cs.parse_calibration_xacro(text)
    p_xyz, _p_rpy = cs.CAL_XACRO_PROPERTIES["keyence"]
    return [float(v) for v in props[p_xyz].split()]


def test_an_azure_solve_does_not_zero_the_keyence():
    """The 2026-10-15 regression, as it happened: one sensor in the record,
    the other only in the description."""
    print("\nan azure solve with a hand-written keyence in force")
    ns = _lift(tf={("keyence_metal", "lj_v7080_mount_cal"): KEY_IN_FORCE},
               record={"azure": AZURE_SOLVED})
    text = ns["calibration_xacro"]()
    kept = _keyence_xyz(text)
    check("the keyence lateral pair survives",
          abs(kept[0] - 0.00056576) < 1e-9 and abs(kept[1] - 0.00172072) < 1e-9,
          "%r" % (kept,))
    check("and the azure is the new solve",
          "-0.00041667" in text, text)
    check("and the file says the keyence was carried over, not measured",
          "carried over from the description in force" in text, text)


def test_identity_when_there_is_nothing_to_carry_over():
    """An uncalibrated cell still gets an uncalibrated file. Carrying forward
    must not become inventing a number."""
    print("\nno record and no TF")
    ns = _lift(record={"azure": AZURE_SOLVED})
    text = ns["calibration_xacro"]()
    check("the keyence is identity", _keyence_xyz(text) == [0.0, 0.0, 0.0],
          text)
    check("and it says why", "nothing in force to carry over" in text, text)


def test_the_deployed_file_answers_when_tf_cannot():
    """calibration_xacro() is called from a dgpy prompt, and the arm is not
    always up. The cell package's own file is then the best answer there is --
    and it is labelled as the second-best one."""
    import tempfile
    print("\nTF unreachable, the cell package file readable")
    d = tempfile.mkdtemp(prefix="cal-carry-")
    path = os.path.join(d, "sensor_calibration.urdf.xacro")
    with open(path, "w") as f:
        f.write(cs.render_calibration_xacro({"keyence": KEY_IN_FORCE}))
    ns = _lift(deployed=d, record={"azure": AZURE_SOLVED})
    text = ns["calibration_xacro"]()
    kept = _keyence_xyz(text)
    check("the lateral pair is read out of the deployed file",
          abs(kept[0] - 0.00056576) < 1e-9 and abs(kept[1] - 0.00172072) < 1e-9,
          "%r" % (kept,))
    check("and the note says it came from the file, not from TF",
          "the deployed file (TF unreachable)" in text, text)
    os.remove(path)
    os.rmdir(d)


def test_tf_wins_over_the_deployed_file():
    """The deployed file is what TF would say after a rebuild and a restart,
    which is true often enough to be dangerous. TF is what the arm is using."""
    print("\nboth available")
    import tempfile
    other = cs.properties_to_cal("0.009 0.009 0", "0 0 0")
    d = tempfile.mkdtemp(prefix="cal-carry-")
    path = os.path.join(d, "sensor_calibration.urdf.xacro")
    with open(path, "w") as f:
        f.write(cs.render_calibration_xacro({"keyence": other}))
    ns = _lift(tf={("keyence_metal", "lj_v7080_mount_cal"): KEY_IN_FORCE},
               deployed=d, record={"azure": AZURE_SOLVED})
    kept = _keyence_xyz(ns["calibration_xacro"]())
    check("TF is the answer", abs(kept[0] - 0.00056576) < 1e-9, "%r" % (kept,))
    os.remove(path)
    os.rmdir(d)


def test_the_result_is_still_well_formed_xml():
    """Whatever the notes end up saying. A malformed file takes the whole
    robot description down."""
    from xml.etree import ElementTree
    print("\nthe file still parses")
    for kwargs in ({"record": {"azure": AZURE_SOLVED}},
                   {"record": None},
                   {"tf": {("keyence_metal", "lj_v7080_mount_cal"):
                           KEY_IN_FORCE},
                    "record": {"azure": AZURE_SOLVED}}):
        text = _lift(**kwargs)["calibration_xacro"]()
        try:
            ElementTree.fromstring(text)
        except ElementTree.ParseError as e:
            check("parses (%r)" % (sorted(kwargs),), False, str(e))
            break
    else:
        check("every combination parses", True)


def main():
    for fn in (test_an_azure_solve_does_not_zero_the_keyence,
               test_identity_when_there_is_nothing_to_carry_over,
               test_the_deployed_file_answers_when_tf_cannot,
               test_tf_wins_over_the_deployed_file,
               test_the_result_is_still_well_formed_xml):
        fn()
    print("\n%s" % ("ALL OK" if not FAILURES else "%d FAILED: %s"
                    % (len(FAILURES), ", ".join(FAILURES))))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
