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
# File:  ec_conditioning_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the wiring of eddy-current conditioning onto a trace list, as a
#   scan performs it.
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
"""Wire the conditioning onto an EC trace list, exactly as a scan would.

The math itself is tested in spatialnde2-ec-functions/test. What this covers is
the WIRING in ec_acquisition.dpi: that setup produces the four channels a scan
reads, that retuning the rotation re-runs over traces already acquired, that a
null can be measured from traces rather than typed in, and -- the one that bites
on the second scan of a session rather than the first -- that teardown hands
every channel back so setup can run again.

ALL OF THAT NEEDS THE COMPILED FUNCTION. Every check here except the null runs
through a real spatialnde2 math function: instantiate it, wait for a globalrev,
read the outputs back. spatialnde2 has no Python math-function support -- that
is why the conditioning is C++ in the first place -- so there is nothing
honest to substitute. If the extension is unavailable this test runs the part
that does not need it and SKIPS the rest, loudly, rather than passing on a
pretence. ec_test_support says how to make it available; it takes a minute on
the host.
"""
import os
import sys

import numpy as np
import spatialnde2 as snde

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import ec_test_support


def _settled(recdb):
    """The newest globalrevision, waited to completion.

    recdb.latest is the newest COMPLETE revision, which straight after a
    parameter write can still be the one from before that write -- the reader
    then sees the old conditioning and the check fails for the wrong reason.
    Pinning the newest DEFINED revision and waiting for it makes the write
    visible before anything is read from it.
    """
    g = recdb.latest_defined_globalrev()
    g.wait_complete()
    return g


def _load_modules():
    ns = {}
    exec("import numpy as np\nimport spatialnde2 as snde\n"
         "import threading, time\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n", ns)
    for name in ("timestamped_trigger.dpi", "ec_acquisition.dpi"):
        src = open(os.path.join(MODULE_DIR, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    return ns


def main():
    # Before _load_modules(): the .dpi imports the conditioning extension at
    # its top, so the stand-in has to be in sys.modules by then.
    ec_status = ec_test_support.ensure_ec_functions()
    ec_test_support.print_status(ec_status, covers_conditioning=True)

    ns = _load_modules()
    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()

    # A trace list shaped as a scan produces one: (K, 1) complex. The first ten
    # sit on one value -- "over good material" -- and the rest depart from it.
    K = 40
    traces = np.empty((K, 1), dtype=np.complex128)
    traces[:10, 0] = 2 + 3j
    traces[10:, 0] = (2 + 3j) + np.linspace(0.1, 1.0, K - 10) * (1 + 1j)

    trans = recdb.start_transaction()
    tchan = trans.define_channel("/ec/traces", "main")
    tref = snde.create_ndarray_ref(trans, tchan, snde.SNDE_RTN_COMPLEXFLOAT64)
    tref.allocate_storage([K, 1], False)
    tref.data[:] = traces
    tref.rec.mark_data_and_metadata_ready()
    trans.end_transaction().globalrev_wait().wait_complete()

    ok = True

    # null measured from the traces the operator knows were over good material
    null = ns["ec_null_from_traces"](recdb, "/ec/traces", first=0, count=10)
    print("  null from traces      (%.4f, %.4f)" % null)
    if not (abs(null[0] - 2.0) < 1e-9 and abs(null[1] - 3.0) < 1e-9):
        print("  FAIL: null should be exactly the good-material value"); ok = False

    # ec_null_from_traces is the one piece that is pure numpy over the recording
    # database, so it is real coverage even with the extension stubbed.
    # Everything past here instantiates the compiled math function.
    if not ec_status["real"]:
        print("")
        print("  SKIPPED, not run: setup/outputs, live rotation, per-axis gain,")
        print("                    teardown-and-setup-again. All four need the")
        print("                    compiled math function.")
        print("EC conditioning wiring: SKIPPED (%s) -- NOT full coverage%s"
              % ("compiled spatialnde2_ec_functions unavailable",
                 "" if ok else "; and the null check above FAILED"))
        return 0 if ok else 1

    cond = ns["setup_ec_conditioning"](recdb, "/ec/traces", prefix="/ec/cond",
                                       null=null, rotation_deg=0.0, gain=(1.0, 1.0))
    latest = _settled(recdb)
    for name, path in sorted(cond["outputs"].items()):
        try:
            data = np.asarray(latest.ref[path].data)
            print("  %-12s %-22s shape %s" % (name, path, data.shape))
        except Exception as e:
            print("  FAIL: output %s unreadable: %s" % (path, e)); ok = False

    lat0 = _settled(recdb)
    h0 = np.asarray(lat0.ref[cond["outputs"]["horizontal"]].data).reshape(-1)
    v0 = np.asarray(lat0.ref[cond["outputs"]["vertical"]].data).reshape(-1)
    if not (abs(h0[0]) < 1e-6 and abs(v0[0]) < 1e-6):
        print("  FAIL: nulled trace did not condition to zero"); ok = False

    # the departure lies at 45 degrees; rotating by 45 should flatten it onto
    # horizontal and leave vertical at zero
    ns["set_ec_parameter"](cond, "rotation", 45.0)
    lat1 = _settled(recdb)
    h1 = np.asarray(lat1.ref[cond["outputs"]["horizontal"]].data).reshape(-1)
    v1 = np.asarray(lat1.ref[cond["outputs"]["vertical"]].data).reshape(-1)
    print("  after 45 deg rotation: max |vertical| = %.2e (was %.4f)"
          % (np.max(np.abs(v1)), np.max(np.abs(v0))))
    if np.max(np.abs(v1)) > 1e-5:
        print("  FAIL: a 45 deg feature should flatten onto horizontal"); ok = False
    if np.max(np.abs(h1)) < 1e-3:
        print("  FAIL: the signal vanished entirely rather than rotating"); ok = False

    # gain is per axis
    ns["set_ec_parameter"](cond, "gain", (3.0, 1.0))
    h2 = np.asarray(_settled(recdb).ref[cond["outputs"]["horizontal"]].data).reshape(-1)
    if not np.allclose(h2, h1 * 3.0, atol=1e-4):
        print("  FAIL: horizontal gain did not apply"); ok = False
    else:
        print("  horizontal gain x3    applied")

    # teardown must hand every channel back, or a second scan in the same
    # session dies inside setup with "reserved_channel is not current"
    ns["teardown_ec_conditioning"](recdb, cond)
    try:
        cond2 = ns["setup_ec_conditioning"](recdb, "/ec/traces", prefix="/ec/cond",
                                            null=null, rotation_deg=10.0)
        ns["teardown_ec_conditioning"](recdb, cond2)
        print("  setup/teardown/setup  OK (second scan in one session)")
    except Exception as e:
        print("  FAIL: second setup after teardown: %s" % e); ok = False

    print("EC conditioning wiring: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def test_ec_conditioning_wiring():
    """ONE test, because every step past the null runs on the SAME live state.

    The recording database, the trace channel and the conditioning object are
    built once; the rotation is measured against the un-rotated output, the gain
    against the rotated one, and teardown-and-setup-again only means anything
    after a setup has happened. There is no section here that could be run on
    its own without rebuilding all of it.

    main() evaluates every condition and accumulates `ok` rather than stopping at
    the first failure, so a standalone run still prints every false fact in one
    pass; this asserts on its verdict.

    THE SKIP IS NOT A PASS. With the compiled extension unavailable main() runs
    the null check -- real coverage, pure numpy over the recording database --
    and skips the four that need the math function. Under pytest that has to
    read as a skip, or an unavailable extension would look like a green suite.
    """
    status = ec_test_support.ensure_ec_functions()
    rc = main()
    assert rc == 0, ("the EC conditioning wiring failed; the FAIL lines in the "
                     "captured output above say which checks broke")
    if not status["real"]:
        import pytest
        pytest.skip("the compiled spatialnde2_ec_functions extension is not "
                    "available, so setup/outputs, live rotation, per-axis gain "
                    "and teardown-and-setup-again were NOT run. Only the null "
                    "was checked. See ec_test_support.py for how to build it.")


if __name__ == "__main__":
    sys.exit(main())
