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
# File:  ec_acquisition_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the eddy-current trace source against a free-running two-
#   channel DAQ.
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
"""Exercise the EC trace source against a free-running two-channel DAQ.

No robot, no instrument, no ROS. A fake DAQ publishes X and Y blocks on their
own timestamped transactions while a fake probe moves along a line; the source
must produce one complex sample per pitch of travel, each carrying the value the
DAQ held when the probe was there.

The thing being tested is the CORRELATION. A source that returns "whatever was
latest when I asked" also produces the right number of traces -- it just
attaches them to the wrong positions, which no count will reveal. So the fake
DAQ encodes its own block index in the signal, and the test checks that trace k
carries the block that was current at crossing k rather than a later one.

ec_acquisition.dpi opens by importing the COMPILED conditioning extension
(spatialnde2_ec_functions), so loading it needs that extension present even
though nothing here touches conditioning -- the trigger, the correlation and the
DAQ pairing are all this test is about. ec_test_support finds the real extension
when it can and otherwise installs a stub that refuses to do arithmetic, which
costs this test no coverage at all. See its module docstring.
"""
import os
import sys
import threading
import time

import numpy as np
import spatialnde2 as snde

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import ec_test_support


def _load_modules():
    """Execute the .dpi bodies into one namespace, the way a config does."""
    ns = {"snde": snde, "np": np, "numpy": np}
    exec("import numpy as np\nimport spatialnde2 as snde\n"
         "import threading, time\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n", ns)
    for name in ("pitch_crossings.dpi", "timestamped_trigger.dpi", "ec_acquisition.dpi"):
        src = open(os.path.join(MODULE_DIR, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    return ns


class FakeDAQ(object):
    """Two free-running real channels, block n carrying the value n."""

    def __init__(self, recdb, clock, real_path, imag_path, rate_hz=200.0,
                 block=4):
        self.recdb = recdb
        self.clock = clock
        self.real_path = real_path
        self.imag_path = imag_path
        self.period = 1.0/rate_hz
        self.block = block
        self.blocks = 0
        self._quit = False

        trans = recdb.start_transaction(clock.get_current_time())
        self.rchan = trans.define_channel(real_path, "main")
        self.ichan = trans.define_channel(imag_path, "main")
        for chan in (self.rchan, self.ichan):
            ref = snde.create_ndarray_ref(trans, chan, snde.SNDE_RTN_FLOAT64)
            ref.allocate_storage([block], False)
            ref.data[:] = 0.0
            ref.rec.mark_data_and_metadata_ready()
        trans.end_transaction().globalrev_wait().wait_complete()

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        from dataguzzler_python import context
        context.InitFreeThread()
        while not self._quit:
            n = self.blocks
            trans = self.recdb.start_transaction(self.clock.get_current_time())
            for chan, offset in ((self.rchan, 0.0), (self.ichan, 1000.0)):
                ref = snde.create_ndarray_ref(trans, chan, snde.SNDE_RTN_FLOAT64)
                ref.allocate_storage([self.block], False)
                ref.data[:] = float(n) + offset
                ref.rec.mark_data_and_metadata_ready()
            trans.end_transaction()
            self.blocks = n + 1
            time.sleep(self.period)

    def stop(self):
        self._quit = True
        self.thread.join(timeout=2.0)


def main():
    # Before _load_modules(): the .dpi imports the conditioning extension at
    # its top, so the stand-in has to be in sys.modules by then.
    ec_status = ec_test_support.ensure_ec_functions()
    ec_test_support.print_status(ec_status, covers_conditioning=False)

    ns = _load_modules()
    ECTimestampedSource = ns["ECTimestampedSource"]

    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 8)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    # Same construction UTPlaneScan uses. The latency is the ordering window:
    # a transaction stamped at T is held this long so anything stamped earlier
    # can still arrive and commit first. 20 ms there; 2 ms is enough here
    # because the fake DAQ and the trigger are in one process.
    clock = snde.measurement_clock_cpp_steady("")
    transmgr = snde.timed_transaction_manager(clock, 0.002)
    # .upcast() is not optional: assigning the derived type straight in trips a
    # SWIG ownership assertion inside the wrapper. recdb.dpi does the same.
    recdb.transmgr = transmgr.upcast()
    recdb.startup()

    daq = FakeDAQ(recdb, clock, "/daq/ai0", "/daq/ai1", rate_hz=200.0, block=4)

    # a probe walking a straight line at a known speed
    state = {"s": 0.0}
    lock = threading.Lock()

    def position():
        with lock:
            return (state["s"], 0.0, 0.0)

    pitch = 0.005
    src = ECTimestampedSource(recdb, clock, "/daq/ai0", "/daq/ai1",
                              position, pitch, reduce="mean",
                              pose_path="/ec/trigger_pose")

    got = []

    def sink(trace, pos, s_m, index):
        got.append((index, complex(trace[0]), s_m))

    # Wait for the DAQ to actually be producing before arming. Without this the
    # first crossing can correlate to the channel's INITIAL DEFINITION rather
    # than to a block -- X and Y both zero, no 1000 offset -- and the pairing
    # check fails perhaps one run in five. That is not test flakiness to paper
    # over: it is a real exposure, because a scan started before the instrument
    # is producing gets exactly this trace on real hardware too.
    deadline = time.monotonic() + 5.0
    while daq.blocks < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert daq.blocks >= 3, "fake DAQ never started producing"

    src.arm(sink, emit_on_arm=False)
    time.sleep(0.3)
    velocity = 0.05
    dt = 0.01
    travelled = 0.20
    steps = int(travelled/(velocity*dt))
    for _ in range(steps):
        with lock:
            state["s"] += velocity*dt
        time.sleep(dt)
    time.sleep(0.5)
    n = src.disarm()
    src.stop()
    daq.stop()

    expected = int(travelled/pitch)
    print("  travelled        %.3f m at %.3f m/s, pitch %.3f m" % (travelled, velocity, pitch))
    print("  triggers         %d (expected about %d)" % (n, expected))
    print("  traces delivered %d" % len(got))
    rep = src.trigger_report()
    print("  errors           %d   mismatched blocks %d" % (rep["errors"], rep["mismatched_blocks"]))
    print("  block samples    %d   reduce=%s" % (rep["block_samples"], rep["reduce"]))

    ok = True
    if not (expected*0.7 <= len(got) <= expected*1.3):
        print("  FAIL: trace count %d is not near %d" % (len(got), expected)); ok = False
    if rep["errors"]:
        print("  FAIL: %d correlator errors: %s" % (rep["errors"], rep["last_error"])); ok = False
    if rep["mismatched_blocks"]:
        print("  FAIL: %d mismatched block pairs" % rep["mismatched_blocks"]); ok = False

    # indices must be dense and ordered: trigger k is trace k
    idx = [g[0] for g in got]
    if idx != sorted(idx):
        print("  FAIL: trace indices out of order"); ok = False

    # the correlation itself: block numbers must be non-decreasing over the
    # scan, and the imaginary channel must stay exactly 1000 above the real one
    # (which is how the fake DAQ marks the pair as coming from ONE block)
    reals = np.array([g[1].real for g in got])
    imags = np.array([g[1].imag for g in got])
    if got:
        if not np.all(np.diff(reals) >= -1e-9):
            print("  FAIL: correlated block numbers went backwards -- traces are "
                  "not carrying the block current at their crossing"); ok = False
        if not np.allclose(imags - reals, 1000.0, atol=1e-6):
            print("  FAIL: X and Y came from different blocks"); ok = False
        print("  block index over scan: %.0f -> %.0f (monotonic=%s)"
              % (reals[0], reals[-1], bool(np.all(np.diff(reals) >= -1e-9))))

    smear = src.report_smear(velocity_mps=velocity, sample_rate_hz=800.0)
    if smear:
        print("  spatial smear    %.4f m = %.1f%% of the trace pitch"
              % (smear["smear_m"], 100.0*smear["fraction_of_pitch"]))

    print("EC trace source: %s%s"
          % ("PASS" if ok else "FAIL",
             "" if ec_status["real"]
             else "   (conditioning extension stubbed; not used by this test)"))
    return 0 if ok else 1


def test_ec_trace_source():
    """ONE test, because this is one live acquisition run.

    main() stands up a recording database, a free-running fake DAQ on its own
    thread and a moving probe, then drives 0.20 m of travel in real time; every
    check below the run is a different property OF THAT ONE RUN and none of them
    can be re-derived without flying it again. Splitting the sections would mean
    four acquisitions instead of one, and they would not be the same data.

    main() evaluates every condition and accumulates `ok` rather than returning
    at the first failure, so one run prints every false fact. This asserts on
    its verdict.
    """
    assert main() == 0, ("the EC trace source failed; the FAIL lines in the "
                         "captured output above say which properties broke")


if __name__ == "__main__":
    sys.exit(main())
