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
# File:  ec_replay_conditioning_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   The replay's numpy conditioning against the compiled C++ one, on a
#   real recording database, trace for trace.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-02:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""The replay's conditioning against the COMPILED one, trace for trace.

WHY THERE ARE TWO IMPLEMENTATIONS AT ALL. The impedance-plane conditioning is a
compiled spatialnde2 math function over recording channels -- that is what lets
an operator turn the rotation and watch thousands of traces re-render -- and it
cannot be asked to condition a numpy array read back off a file. So the offline
replay has its own, in ec_trace_store.py, and the two must agree: a replay that
conditioned differently from the scan would show the operator a picture of their
data that the scan never produced, and nothing would say so.

This is the test that keeps that duplication honest. It wires the REAL compiled
function onto a real trace list, reads the real float32 output back, and
compares it against the numpy one over every trace, at three different
rotations. It also drives the REAL ECPassRecorder against real spatialnde2
arrays, which is where the shapes actually bite: a trace list is (K, 1) complex
and the poses are float32, and a recorder that mishandled either would still
write a plausible-looking file.

THE SKIP IS NOT A PASS. With the compiled extension unavailable there is nothing
honest to substitute -- spatialnde2 has no Python math-function support, which
is the whole reason the conditioning is C++ -- so this reports a skip rather
than agreeing with itself. ec_test_support.py says how to build it; it takes
about a minute on the host.

    python3 -m pytest -q ec_replay_conditioning_test.py
"""
import os
import sys

import numpy as np
import spatialnde2 as snde

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ec_test_support
import ec_trace_store as store

#: Deliberately not the identity, and deliberately not symmetric: a comparison
#: run at null (0,0), rotation 0, gain (1,1) cannot tell an implementation that
#: applies the parameters from one that ignores them.
NULL = (0.41, -0.17)
GAIN = (1.3, 2.7)
ROTATIONS = (0.0, 31.0, -104.5)


def _settled(recdb):
    """The newest globalrevision, waited to completion.

    recdb.latest is the newest COMPLETE revision, which straight after a
    parameter write can still be the one from before it -- the reader then sees
    the previous rotation and the comparison fails for the wrong reason.
    """
    g = recdb.latest_defined_globalrev()
    g.wait_complete()
    return g


def _load_modules():
    """The .dpi bodies in one namespace, the way a config executes them."""
    ns = {}
    exec("import numpy as np\nimport spatialnde2 as snde\n"
         "import os, threading, time\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n", ns)
    for name in ("timestamped_trigger.dpi", "ec_acquisition.dpi",
                 "ec_trace_store.dpi"):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    # _store() reaches for SCAN_MODULE_DIR when the module is not importable;
    # it is importable here, but say where anyway so the lookup is not a
    # surprise if that ever changes.
    ns["SCAN_MODULE_DIR"] = HERE
    return ns


class StandInAccumulator(object):
    """Channel names and a staged count, which is all the recorder asks for.

    Not a TraceAccumulator: instantiating one needs nd_accumulate_once and the
    staging dance that only makes sense inside a scan. What this test is about
    is the recorder reading REAL spatialnde2 arrays -- (K, 1) complex traces and
    float32 poses -- and slicing the staged part of them.
    """

    traces_path = "/ec/traces"
    poses_path = "/ec/trace_poses"

    def __init__(self, staged):
        self.staged = int(staged)


def main(tmpdir=None):
    ec_status = ec_test_support.ensure_ec_functions()
    ec_test_support.print_status(ec_status, covers_conditioning=True)

    ns = _load_modules()
    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()
    ns["recdb"] = recdb

    # A trace list shaped the way a scan produces one: (K, 1) complex, values
    # spread around the impedance plane so no rotation is a special case.
    K = 48
    rng = np.random.default_rng(20261002)
    z = (rng.normal(scale=0.8, size=K) + 1j*rng.normal(scale=0.8, size=K)
         + (0.4 - 0.25j))
    traces = z.reshape(K, 1).astype(np.complex128)
    poses = np.zeros((K, 7), dtype=np.float32)
    poses[:, 0] = 1.0
    poses[:, 4] = 0.80 + 0.001*np.arange(K)
    poses[:, 5] = -0.30
    poses[:, 6] = 0.952

    trans = recdb.start_transaction()
    tchan = trans.define_channel("/ec/traces", "main")
    tref = snde.create_ndarray_ref(trans, tchan, snde.SNDE_RTN_COMPLEXFLOAT64)
    tref.allocate_storage([K, 1], False)
    tref.data[:] = traces
    tref.rec.mark_data_and_metadata_ready()
    pchan = trans.define_channel("/ec/trace_poses", "main")
    pref = snde.create_ndarray_ref(trans, pchan, snde.SNDE_RTN_FLOAT32)
    pref.allocate_storage([K, 7], False)
    pref.data[:] = poses
    pref.rec.mark_data_and_metadata_ready()
    trans.end_transaction().globalrev_wait().wait_complete()

    ok = True

    if not ec_status["real"]:
        print("")
        print("  SKIPPED, not run: every check in this file goes through the")
        print("                    compiled conditioning. There is no honest")
        print("                    substitute -- see the module docstring.")
        print("EC replay conditioning: SKIPPED (compiled extension unavailable)")
        return 0

    cond = ns["setup_ec_conditioning"](recdb, "/ec/traces", prefix="/ec/cond",
                                       null=NULL, rotation_deg=ROTATIONS[0],
                                       gain=GAIN)

    # ---- the comparison, at three rotations -------------------------------
    for rot in ROTATIONS:
        ns["set_ec_parameter"](cond, "rotation", rot)
        latest = _settled(recdb)
        for name in ("horizontal", "vertical", "magnitude"):
            compiled = np.asarray(
                latest.ref[cond["outputs"][name]].data).reshape(-1)
            mine = store.component_values(z, name, null=NULL,
                                          rotation_deg=rot, gain=GAIN)
            # float32 on the compiled side, double on ours, so the bound is the
            # float32 representation of the values themselves and not an
            # absolute epsilon: an EC reading is volts of whatever size the
            # instrument's gain made it.
            scale = max(float(np.max(np.abs(compiled))), 1e-12)
            err = float(np.max(np.abs(mine - compiled)))
            tol = 1e-5*scale
            print("  rotation %+7.1f deg  %-10s max |numpy - C++| = %.3e "
                  "(tolerance %.3e)" % (rot, name, err, tol))
            if not err <= tol:
                print("  FAIL: the replay's %s does not reproduce the compiled "
                      "one at %.1f deg" % (name, rot)); ok = False

        # And the conditioned COMPLEX channel, which is what the phase plane
        # draws -- the same two numbers, so a disagreement here would mean the
        # replay had the real and imaginary parts the wrong way round.
        compiled_c = np.asarray(
            latest.ref[cond["outputs"]["conditioned"]].data).reshape(-1)
        mine_c = store.condition_traces(z, null=NULL, rotation_deg=rot,
                                        gain=GAIN)["conditioned"]
        scale = max(float(np.max(np.abs(compiled_c))), 1e-12)
        if not float(np.max(np.abs(mine_c - compiled_c))) <= 1e-5*scale:
            print("  FAIL: the conditioned complex value disagrees at %.1f deg"
                  % (rot,)); ok = False

    # ---- the recorder, against real spatialnde2 arrays ---------------------
    ns["set_ec_parameter"](cond, "rotation", ROTATIONS[1])
    _settled(recdb)
    staged = 30                       # fewer than K: the tail is not real
    accum = StandInAccumulator(staged)

    def read_array(path):
        return np.asarray(recdb.latest.ref[path].data)

    directory = tmpdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "_ec_replay_test_tmp")
    import shutil
    if tmpdir is None and os.path.isdir(directory):
        shutil.rmtree(directory)

    recorder = ns["ECPassRecorder"](
        directory, read_array,
        lambda: {"accum": accum, "cond_paths": cond["outputs"],
                 "component": "vertical", "null": NULL,
                 "rotation_deg": ROTATIONS[1], "gain": GAIN,
                 "x3d": "/data/scandata/specimen_uv.x3d", "pitch_m": 0.001})
    saved, why = recorder(0)
    if not saved:
        print("  FAIL: the recorder could not write a pass: %s" % (why,)); ok = False
        print("EC replay conditioning: FAIL")
        return 1
    print("  recorder wrote %s" % (os.path.basename(why),))

    run = store.read_run(directory)
    if run["traces"] != staged:
        print("  FAIL: saved %d trace(s), expected the %d staged"
              % (run["traces"], staged)); ok = False
    if not np.allclose(run["impedance"], z[:staged]):
        print("  FAIL: the raw impedance did not survive the round trip"); ok = False
    if not np.allclose(run["pose"], poses[:staged].astype(np.float64)):
        print("  FAIL: the poses did not survive the round trip"); ok = False

    # The file's conditioned column came from the C++; the replay recomputes it
    # from the raw value. This is the whole claim, measured end to end.
    chk = store.condition_matches_stored(run)
    print("  %s" % (chk["why"],))
    if not (chk["ok"] and chk["checked"] == staged):
        print("  FAIL: the replay does not reproduce the saved conditioned "
              "column (%d checked)" % (chk["checked"],)); ok = False

    ns["teardown_ec_conditioning"](recdb, cond)
    if tmpdir is None:
        shutil.rmtree(directory, ignore_errors=True)

    print("EC replay conditioning: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def test_the_replay_reproduces_the_compiled_conditioning(tmp_path):
    """ONE test, because it is one live recording database.

    The trace channel, the math function and the recorder are built once and
    every check below reads the same state at three successive rotations; a
    rotation only means anything measured against the output of the one before
    it. main() accumulates failures rather than returning at the first, so a
    standalone run prints every false fact in one pass.

    THE SKIP IS NOT A PASS: with the extension unavailable nothing here runs,
    and that has to read as a skip or a missing extension would look green.
    """
    status = ec_test_support.ensure_ec_functions()
    rc = main(tmpdir=str(tmp_path))
    assert rc == 0, ("the replay's conditioning does not match the compiled "
                     "one; the FAIL lines in the captured output say where")
    if not status["real"]:
        import pytest
        pytest.skip("the compiled spatialnde2_ec_functions extension is not "
                    "available, so NOTHING in this file ran -- the agreement "
                    "between the replay's numpy conditioning and the C++ one "
                    "is unchecked. See ec_test_support.py.")


if __name__ == "__main__":
    sys.exit(main())
