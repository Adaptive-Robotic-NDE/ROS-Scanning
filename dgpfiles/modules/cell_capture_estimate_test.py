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
# File:  cell_capture_estimate_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the take size estimate, including topics the probe cannot see,
#   and the free-space refusal.
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
"""The size estimate has to include the topic the probe cannot see.

start_take() prints "~26.2 MB for 90 s (15 MB/min); 177.2 GB free" and, on
hardware, REFUSES the take if that does not fit above the reserve. The number
comes from tari_bringup's `record probe`, which listens for three seconds and
counts bytes.

The Keyence laser is switched on INSIDE capture_scenario(), after the recorder
has started -- which is after the probe has run. So on a take that records
/profiles the probe hears nothing on the one topic that dominates the bag: over
a 1094 s laser take at the cell, /profiles was 237 989 messages and
3 077 673 748 bytes, 2.82 MB/s, which was 91% of the 3.09 MB/s that take
actually wrote. An estimate that silently omits it is not conservative; it is
wrong by a factor of ten, and it is the number a refusal is based on.

Runs on the host: the .dpi is exec'd with its `global` lines commented out, the
same way cell_capture_preview_test.py does it. No ROS, no recorder, no cell.

    python3 cell_capture_estimate_test.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    """The estimate helpers out of cell_capture_take.dpi."""
    src = open(os.path.join(HERE, "cell_capture_take.dpi")).read()
    src = "\n".join(("#" + l) if l.startswith("global ") else l
                    for l in src.splitlines())
    ns = {"np": np}
    exec(compile(src, "cell_capture_take.dpi", "exec"), ns)
    return ns


NS = _load()

#: What a `record probe` result looks like, trimmed to the fields the estimate
#: reads. The rates are this cell's: joint_states, io_states, speed_scaling and
#: the wrench all run at 500 Hz, and together they are the 265 kB/s a
#: calibration bag is made of.
def _probe(profiles_bps=0.0):
    return {"topics": {
        "/joint_states": {"set": "core", "projected_bps": 158000.0},
        "/force_torque_sensor_broadcaster/wrench": {"set": "core",
                                                    "projected_bps": 38000.0},
        "/io_and_status_controller/io_states": {"set": "core",
                                                "projected_bps": 64000.0},
        "/tf": {"set": "core", "projected_bps": 12600.0},
        "/profiles": {"set": "keyence", "projected_bps": profiles_bps},
    }, "projected_bps": 272600.0 + profiles_bps,
        "projected_mb_per_min": (272600.0 + profiles_bps)*60/1e6,
        "take_bytes": (272600.0 + profiles_bps)*1336.0,
        "free_bytes": 190.0e9, "fits": True}


def test_a_silent_profile_topic_is_counted_at_its_measured_rate():
    gated, notes = NS["_probe_gated_bps"](_probe(0.0), ("core", "keyence"))
    assert gated == NS["CAPTURE_GATED_SET_BPS"]["keyence"], gated
    assert len(notes) == 1, notes
    assert "/profiles" in notes[0]
    assert "laser is switched on after the recorder starts" in notes[0]

    # What that is worth on a real take: a 1336 s calibrate_keyence writes
    # 354.8 MB without /profiles. With it, the honest number is a shade under
    # 4 GB.
    need = _probe(0.0)["take_bytes"] + gated*1336.0
    assert 3.5e9 < need < 4.2e9, need


def test_a_topic_the_probe_did_hear_is_not_counted_twice():
    """The floor is a floor. If the laser happened to be on, the measured rate
    is already in take_bytes and is the better number -- and it is bigger."""
    gated, notes = NS["_probe_gated_bps"](_probe(2.9e6), ("core", "keyence"))
    assert gated == 0.0 and notes == [], (gated, notes)


def test_a_core_only_take_is_unchanged():
    """Nothing is added to a take that never planned the topic. The estimate on
    every other scenario must read exactly as it did."""
    est = _probe(0.0)
    del est["topics"]["/profiles"]
    gated, notes = NS["_probe_gated_bps"](est, ("core",))
    assert gated == 0.0 and notes == [], (gated, notes)


def test_the_declared_rate_is_the_one_that_was_measured():
    """800 columns x 16 bytes/point at 217.6 Hz, plus a header: 12 932 bytes a
    message, 237 989 messages in 1090.2 s. If somebody retunes the head's rate
    this constant has to move with it, so the arithmetic is written down."""
    measured = 3077673748.0/1090.205961
    assert abs(NS["CAPTURE_GATED_SET_BPS"]["keyence"] - measured) < 0.05*measured, (
        NS["CAPTURE_GATED_SET_BPS"]["keyence"], measured)
    per_msg = 3077673748.0/237989
    assert abs(per_msg - 800*16) < 200, per_msg


# ------------------------------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            failed.append((name, e))
            print("  FAIL  %s\n        %s" % (name, e))
        else:
            print("  ok    %s" % name)
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    if failed:
        raise SystemExit(str(failed[0][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
