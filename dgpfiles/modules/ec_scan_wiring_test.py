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
# File:  ec_scan_wiring_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests when eddy-current conditioning may be wired, and what happens
#   when it is wired too early.
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
"""WHEN the eddy-current conditioning may be wired, and what happens if not.

THE FAULT THIS IS ABOUT. Wiring the conditioning before the trace list exists
takes the whole dgpy process out, mid-scan, with no traceback:

    terminate called after throwing an instance of 'snde::snde_error'
      what():  SNDE runtime error: Prerequisite channel /ec/traces of math
               function /ec/cond/conditioned does not exist
    Aborted

Two separate facts make that happen, and this suite covers both.

ONE: /ec/traces IS AN OUTPUT, NOT AN INPUT. The trace list is produced by
nd_accumulate_once inside TraceAccumulator.setup(), which the scan calls from
its setup_after_first hook -- after the first reading has been staged. Naming it
as an input to the conditioning during setup_ec(), minutes earlier, names a
channel that cannot exist, and no amount of retrying will make it.

TWO: UNDER THE TIMED TRANSACTION MANAGER THAT IS SIGABRT, NOT AN EXCEPTION.
spatialnde2 resolves a math function's prerequisites when the transaction is
realized, and which thread that happens on is the transaction manager's choice:

    ordered_transaction_manager   realizes on the caller's thread ->
                                  end_transaction() raises a catchable snde_error
    timed_transaction_manager     realizes on transaction_end_thread_code(),
                                  its own thread -> nothing catches it ->
                                  terminate(), SIGABRT, the whole process

ECAutomatedScan.dgp REQUIRES the timed manager (the eddy-current correlation IS
the timestamp ordering), so this config always gets the second column. That is
why ec_channel_exists() is a hard gate rather than a warning, and it is why one
of the checks below runs in a SUBPROCESS: the only way to show that the guard is
what stands between a readable error and a core dump is to remove it and watch
the child die on a signal. A test that merely asserted "it raises" would pass
against the ordered manager and prove nothing about the config that actually
ships.

    python3 ec_scan_wiring_test.py
    python3 -m pytest -q ec_scan_wiring_test.py
"""
import math
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import spatialnde2 as snde

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import ec_test_support

FAILURES = []


def check(cond, what):
    print("%s  %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILURES.append(what)


def _load(names):
    """Run .dpi bodies into one namespace, the way the loader shares globals()."""
    ns = {}
    exec("import math, os, threading, time\n"
         "import numpy as np\nimport spatialnde2 as snde\n"
         "try:\n    import queue\nexcept ImportError:\n    import Queue as queue\n", ns)
    for name in names:
        src = open(os.path.join(MODULE_DIR, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    return ns


#: Every recording database this file makes, held for the life of the process.
#: snde.recdatabase has no shutdown or close, so one that goes out of scope is
#: destroyed underneath its own background threads, and one of them then locks
#: a weak_ptr to a compute-resource database that is gone -- std::bad_weak_ptr,
#: an uncaught C++ exception, terminate(), and the whole run aborts with "Fatal
#: Python error: Aborted". Measured at roughly one run in three here. Holding
#: them removes the provocation; the teardown race itself is spatialnde2's.
_RECDBS = []


def _recdb(timed=False):
    recdb = snde.recdatabase()
    if timed:
        clock = snde.measurement_clock_cpp_steady("")
        recdb.transmgr = snde.timed_transaction_manager(clock, 0.02).upcast()
        recdb._test_clock = clock
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()
    _RECDBS.append(recdb)
    return recdb


def _trace_list(recdb, traces, path="/ec/traces"):
    trans = recdb.start_transaction()
    chan = trans.define_channel(path, "main")
    ref = snde.create_ndarray_ref(trans, chan, snde.SNDE_RTN_COMPLEXFLOAT64)
    ref.allocate_storage(list(traces.shape), False)
    ref.data[:] = traces
    ref.rec.mark_data_and_metadata_ready()
    trans.end_transaction().globalrev_wait().wait_complete()
    return chan


status = ec_test_support.ensure_ec_functions()
ec_test_support.print_status(status, covers_conditioning=True)
REAL_MATH = bool(status.get("real"))

ns = _load(["timestamped_trigger.dpi", "ec_acquisition.dpi"])


# ==============================================================================
print("")
print("the guard: a missing prerequisite is reported, not realized")

recdb = _recdb()
check(ns["ec_channel_exists"](recdb, "/ec/traces") is False,
      "ec_channel_exists() says no for a channel that does not exist")
_trace_list(recdb, np.zeros((4, 1), dtype=np.complex128))
check(ns["ec_channel_exists"](recdb, "/ec/traces") is True,
      "...and yes once it does")
check(ns["ec_channel_exists"](recdb, "/ec/trace") is False,
      "...and is not fooled by a prefix of a real channel name")

# THE ORIGINAL FAULT, as a call. The channel name is one nobody created.
recdb2 = _recdb()
try:
    ns["setup_ec_conditioning"](recdb2, "/ec/traces", prefix="/ec/cond")
    check(False, "wiring onto an absent trace list must not be allowed")
    why = ""
except RuntimeError as e:
    why = str(e)
    check(True, "wiring onto an absent trace list raises RuntimeError")
except Exception as e:                                     # noqa: BLE001
    why = ""
    check(False, "expected RuntimeError, got %s: %s" % (type(e).__name__, e))

check("/ec/traces" in why,
      "...and the message names the channel that is missing")
check("setup_after_first" in why or "ensure_ec_conditioning" in why,
      "...and says where the wiring belongs instead")
check("SIGABRT" in why or "terminate" in why,
      "...and says why this is a hard refusal and not a warning")

# The refusal must happen BEFORE a transaction is opened, or the guard has
# already lost: a half-built transaction is what realizes on the manager thread.
recdb3 = _recdb()
before = set(recdb3.latest.list_recordings())
try:
    ns["setup_ec_conditioning"](recdb3, "/nope/traces", prefix="/ec/cond")
except RuntimeError:
    pass
after = set(recdb3.latest.list_recordings())
check(before == after,
      "a refused wiring defines NO channels -- it declines before opening a "
      "transaction, so there is nothing half-built to realize")


# ==============================================================================
print("")
print("the guard is what stands between an error and a core dump")

# This is the check that needs a subprocess. Under the TIMED manager -- the only
# one ECAutomatedScan.dgp permits -- the same missing prerequisite is thrown on
# the manager's own thread. With the guard: RuntimeError, exit 0. Without it:
# terminate() and a signal. Nothing in-process can assert that, because the
# failure kills the process doing the asserting.
_CHILD = textwrap.dedent("""
    import os, sys
    import numpy as np, spatialnde2 as snde
    sys.path.insert(0, %(moduledir)r)
    import ec_test_support
    ec_test_support.ensure_ec_functions()

    ns = {}
    exec("import math, os, threading, time\\nimport numpy as np\\n"
         "import spatialnde2 as snde\\nimport queue\\n", ns)
    for name in ("timestamped_trigger.dpi", "ec_acquisition.dpi"):
        src = open(os.path.join(%(moduledir)r, name)).read()
        src = "\\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)

    # EXACTLY what ECAutomatedScan.dgp installs (its lines 91-92 and recdb.dpi:20).
    clock = snde.measurement_clock_cpp_steady("")
    recdb = snde.recdatabase()
    recdb.transmgr = snde.timed_transaction_manager(clock, 0.02).upcast()
    snde.setup_cpu(recdb, [], 2); snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, []); recdb.startup()

    GUARDED = %(guarded)s
    if not GUARDED:
        # Remove the guard, to show what it is guarding.
        ns["ec_channel_exists"] = lambda recdb, path: True
    try:
        ns["setup_ec_conditioning"](recdb, "/ec/traces", prefix="/ec/cond")
        print("WIRED")
    except RuntimeError:
        print("REFUSED")
    sys.stdout.flush()
    os._exit(0)
""")


def _child(guarded):
    src = _CHILD % {"moduledir": MODULE_DIR, "guarded": "True" if guarded else "False"}
    p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=300)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


if not REAL_MATH:
    print("  SKIPPED: needs the compiled ec_condition_function -- the stub "
          "refuses to instantiate, so there is no transaction to realize and "
          "nothing here would be testing spatialnde2's behaviour.")
else:
    rc, out = _child(guarded=True)
    check(rc == 0 and "REFUSED" in out,
          "WITH the guard, the timed manager run refuses cleanly and the "
          "process survives (exit %s)" % (rc,))

    rc, out = _child(guarded=False)
    # A signal death is a negative returncode from subprocess, or 134 when a
    # shell reports SIGABRT. Either way it is not a clean exit.
    died = (rc < 0) or (rc in (134, 139))
    check(died,
          "WITHOUT the guard, the same call KILLS the process on the "
          "transaction manager's thread (exit %s) -- which is what the cell "
          "saw, and is why the guard is not optional" % (rc,))
    check("does not exist" in out or "snde_error" in out,
          "...and the only trace of it is spatialnde2's own message")


# ==============================================================================
print("")
print("parameters can be set before there is anything to wire them onto")

check(ns["ec_cond_pending"]["null"] == (0.0, 0.0)
      and ns["ec_cond_pending"]["rotation"] == 0.0,
      "the pending parameters start at identity, which is 'unconditioned'")

held = ns["ec_conditioning_parameters"](null=(2.0, 3.0), rotation_deg=-35.0,
                                        gain=(1.0, 4.0), cond=None)
check(held["live"] is False,
      "setting a parameter with nothing wired reports that it is NOT live")
check(held["null"] == (2.0, 3.0) and held["rotation"] == -35.0
      and held["gain"] == (1.0, 4.0),
      "...and remembers every one of the three")
check(ns["ec_cond_pending"]["rotation"] == -35.0,
      "...in ec_cond_pending, which is where the scan will read them from")


# ==============================================================================
print("")
print("ensure_ec_conditioning wires it, once, with the pending parameters")

if not REAL_MATH:
    print("  SKIPPED: needs the compiled ec_condition_function.")
else:
    recdb4 = _recdb()
    # Ten traces on the null point, then a departure from it -- "good material,
    # then something".
    K = 30
    traces = np.empty((K, 1), dtype=np.complex128)
    traces[:10, 0] = 2 + 3j
    traces[10:, 0] = (2 + 3j) + np.linspace(0.1, 1.0, K - 10)*(1 + 1j)
    _trace_list(recdb4, traces)

    cond = ns["ensure_ec_conditioning"](recdb4, "/ec/traces", prefix="/ec/cond",
                                        cond=None)
    check(cond is not None, "it wires when there is nothing wired")
    same = ns["ensure_ec_conditioning"](recdb4, "/ec/traces", prefix="/ec/cond",
                                        cond=cond)
    check(same is cond,
          "...and is idempotent: a second call returns the same handle rather "
          "than redefining the channels, which is its own failure two scans "
          "later (see teardown_ec_conditioning)")

    outs = cond["outputs"]
    have = set(recdb4.latest.list_recordings())
    for name in ("conditioned", "horizontal", "vertical", "magnitude"):
        check(outs[name] in have, "it produced %s" % (outs[name],))

    # THE PENDING NULL WAS APPLIED. The first ten traces sat exactly on it, so
    # conditioned[:10] must be zero -- that is the whole point of nulling.
    z = np.asarray(recdb4.latest.ref[outs["conditioned"]].data).reshape(-1)
    check(np.allclose(z[:10], 0.0, atol=1e-9),
          "the PENDING null (2, 3) reached the math: the ten traces that sat on "
          "it condition to exactly zero")
    check(not np.allclose(z[10:], 0.0, atol=1e-9),
          "...and the traces that departed from it did not")

    # Nested output paths, three deep, with no group channel declared for /ec or
    # /ec/cond. Checked because it was raised as a doubt, and because a wrong
    # answer here would look exactly like the fault above.
    check(all(p.count("/") == 3 for p in outs.values()),
          "the outputs are three-deep nested paths (/ec/cond/<name>) and "
          "resolve with no group channel declared for /ec or /ec/cond")


# ==============================================================================
print("")
print("the simulated instrument looks like the DAQ it replaces")

sim_ns = _load(["sim_ec_source.dpi"])
SimECSource = sim_ns["SimECSource"]

recdb5 = _recdb()
sim = SimECSource(recdb5, None, "/PXI1Slot2/ai0", "/PXI1Slot2/ai1",
                  position_fn=None, sample_rate=1000.0, block_hz=100.0, seed=7)
try:
    deadline = time.monotonic() + 20.0
    while sim.blocks < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    check(sim.blocks >= 3,
          "it free-runs: %d block(s) published without anything asking" % (sim.blocks,))
    check(sim.errors == 0,
          "...with no errors (%s)" % (sim.last_error,))

    # ONE GLOBALREV CARRIES BOTH. This is the property the correlation depends
    # on: ECTimestampedSource._sample() reads the two channels out of a single
    # globalrev and pairs them. Two transactions would make that a race.
    g = recdb5.latest
    xr = g.check_for_recording_ref("/PXI1Slot2/ai0")
    yr = g.check_for_recording_ref("/PXI1Slot2/ai1")
    check(xr is not None and yr is not None,
          "both channels are present in ONE globalrev, which is what lets "
          "_sample() pair them")
    x = np.asarray(xr.data)
    y = np.asarray(yr.data)
    check(x.shape == y.shape == (10,),
          "each block is SampsPerChan samples on both channels, equal length "
          "(%s, %s) -- _sample() counts a length mismatch as a fault"
          % (x.shape, y.shape))
    check(x.dtype == np.float64 and y.dtype == np.float64,
          "float64, as dgpython_nidaqmx.ai_voltage writes")
    check(int(xr.rec.info.revision) >= 1 and int(yr.rec.info.revision) >= 1,
          "each block is a NEW revision -- the trigger's stale-digitizer check "
          "counts repeats of the same one")
finally:
    sim.stop()

check(sim.blocks > 0 and sim._thread is None, "stop() ends the thread")


# ==============================================================================
print("")
print("the synthetic signal is position-dependent, which is what makes an "
      "image worth looking at")

recdb6 = _recdb()
sim2 = SimECSource(recdb6, None, "/sim/x", "/sim/y", position_fn=None, seed=1)
try:
    sim2.stop()                      # drive it by hand from here
    # No scan_origin: nothing knows where the part is, so there is no flaw map.
    sim_ns["scan_origin"] = None
    check(sim2.flaw_amplitude((0.008, 0.020)) == 0.0,
          "with no scan_origin there is no flaw map, and it says 0 rather than "
          "inventing a position")

    sim_ns["scan_origin"] = (0.0, 0.0)
    on = sim2.flaw_amplitude((0.008, 0.020))
    off = sim2.flaw_amplitude((0.008 + 0.030, 0.020))
    check(on > 0.5, "on top of a flaw the response is large (%.3f V)" % (on,))
    check(off < 0.01,
          "30 mm away it is essentially nothing (%.4f V), so a scan produces a "
          "picture rather than a flat field" % (off,))
    peak = max(sim2.flaw_amplitude((fx, fy)) for fx, fy, _r, _a in sim2.FLAWS)
    check(peak > 0.8, "each flaw peaks near its own amplitude (%.2f V)" % (peak,))

    # THE SETUP PROCEDURE IS EXERCISABLE. Lift-off and the flaw response must lie
    # along DIFFERENT directions in the raw plane, or there is no rotation to
    # find and set_rotation() cannot be tested against anything.
    sep = abs(sim2.FLAW_DIR_DEG - sim2.LIFTOFF_DIR_DEG) % 180.0
    check(20.0 < sep < 160.0,
          "lift-off and flaw response are %.0f deg apart in the raw impedance "
          "plane, so there is a real rotation for set_rotation() to find" % (sep,))
    check(sim2.BASE != (0.0, 0.0),
          "and the instrument is NOT nulled when you find it, so null_here() "
          "has something to remove")
finally:
    sim2.stop()


# ==============================================================================
print("")
print("end to end: null, rotate, and lift-off leaves the defect axis alone")

if not REAL_MATH:
    print("  SKIPPED: needs the compiled ec_condition_function.")
else:
    # THE EDDY-CURRENT SETUP PROCEDURE, against the simulated instrument: null on
    # good material, then turn the rotation until lift-off lies horizontal. If
    # that works, `vertical` carries the part and nothing else -- which is the
    # claim run_ec_scan(component="vertical") makes when it paints a surface.
    recdb7 = _recdb()
    sim3 = SimECSource(recdb7, None, "/sim/x", "/sim/y", position_fn=None, seed=3)
    sim3.stop()
    sim_ns["scan_origin"] = (0.0, 0.0)
    sim3.NOISE_V = 0.0               # the geometry, not the noise floor

    def impedance(pos):
        xs, ys = sim3.synthesize(pos=pos, n=1)
        return complex(xs[0], ys[0])

    # Two points of clean material with DIFFERENT lift-off, and one over a flaw.
    clean_a = impedance((0.0600, 0.0600))
    clean_b = impedance((0.0600 + 0.25*sim3.LIFTOFF_PERIOD_M, 0.0600))
    flawed = impedance((0.008, 0.020))
    traces = np.array([clean_a, clean_b, flawed], dtype=np.complex128).reshape(3, 1)
    _trace_list(recdb7, traces)

    null = ns["ec_null_from_traces"](recdb7, "/ec/traces", first=0, count=1)
    # The rotation parameter is the angle the feature SITS AT, not its
    # negative: ec_condition computes theta = -rot_deg. Asking for -35 here put
    # lift-off at 70 deg instead of 0, which is how the convention got checked.
    ns["ec_conditioning_parameters"](null=null,
                                     rotation_deg=sim3.LIFTOFF_DIR_DEG,
                                     gain=(1.0, 1.0), cond=None)
    cond = ns["ensure_ec_conditioning"](recdb7, "/ec/traces", prefix="/ec/cond")
    h = np.asarray(recdb7.latest.ref[cond["outputs"]["horizontal"]].data).reshape(-1)
    v = np.asarray(recdb7.latest.ref[cond["outputs"]["vertical"]].data).reshape(-1)

    check(abs(h[0]) < 1e-6 and abs(v[0]) < 1e-6,
          "the trace nulled on conditions to the origin")
    check(abs(v[1]) < 1e-6 and abs(h[1]) > 1e-3,
          "LIFT-OFF MOVES ONLY THE HORIZONTAL AXIS once the rotation is set "
          "(h %.4f, v %.4f) -- which is the whole point of setting it"
          % (h[1], v[1]))
    check(abs(v[2]) > 1e-2,
          "and a flaw shows on the VERTICAL axis (%.4f), which is what "
          "component='vertical' paints onto the part" % (v[2],))
    check(abs(v[2]) > 3.0*abs(v[1]),
          "...with the flaw at %.4f on that axis against %.2e of lift-off "
          "leakage, so the defect axis carries the part and not the surface "
          "finish" % (abs(v[2]), abs(v[1])))


# ==============================================================================
print("")
print("the self-test that turns a stale extension into a message")

# THE FAULT IT IS FOR: when spatialnde2 is rebuilt under an already-built
# ec_condition, the extension still imports cleanly and SEGFAULTS the first time
# the math actually executes. A Cython extension linked against a C++ ABI cannot
# detect that by importing -- only by running -- and running it in-process means
# the failure is a signal with nothing to catch. So check_ec_conditioning() runs
# it in a CHILD.

if not REAL_MATH:
    print("  SKIPPED: needs the compiled ec_condition_function.")
else:
    ok, why = ns["check_ec_conditioning"]()
    check(ok, "check_ec_conditioning() passes against a matched build (%s)"
              % (why if not ok else "arithmetic verified in the child"))

# FAULT INJECTION: a child that dies on a signal must be reported as a crash,
# with the rebuild instructions, and must not be mistaken for "not installed".
# The self-test SOURCE is a separate function precisely so this is injectable --
# there is no honest way to stale an ABI on demand.
_real_source = ns["ec_conditioning_selftest_source"]
try:
    ns["ec_conditioning_selftest_source"] = (
        lambda: "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    ok, why = ns["check_ec_conditioning"](timeout=120.0)
    check(ok is False, "a child that dies on a signal is NOT reported as a pass")
    check("CRASHED THE PROCESS" in why,
          "...it is reported as a crash")
    check("build_ext" in why and "Cython" in why,
          "...with the rebuild command and why the ABI is the suspect")
    check("libspatialnde2" in why,
          "...and tells the operator which two build dates to compare")

    # And a child that merely FAILS is a different message: a wrong answer from
    # working code is not a stale build, and saying so would send somebody to
    # rebuild a library that is fine.
    ns["ec_conditioning_selftest_source"] = (
        lambda: "raise SystemExit('deliberate: the arithmetic was wrong')\n")
    ok, why = ns["check_ec_conditioning"](timeout=120.0)
    check(ok is False, "a child that exits non-zero is not a pass either")
    check("CRASHED THE PROCESS" not in why,
          "...and is NOT blamed on the ABI")
    check("deliberate" in why,
          "...the child's own output is quoted instead")
finally:
    ns["ec_conditioning_selftest_source"] = _real_source


# ==============================================================================
print("")
print("the lift-off rotation is measured from the traces, not dialled in")

# The projection paints /ec/cond/vertical, and that channel only means "the
# part" once lift-off lies along the horizontal axis. The angle belongs to the
# probe, the frequency and the instrument's own phase -- not to a config file --
# so it is measured from the scan's own data. Planting an angle and recovering
# it is the only way to know the measurement works.
#
# ONE recording database for every case here, with a channel per case. Each
# snde.recdatabase carries its own compute threads, and this file already holds
# every one it makes for the life of the process (see _RECDBS); a database per
# case would put eighty-odd threads on an eight-core machine and the earlier
# blocks start timing out. _trace_list() takes the path, so one database does.
_rot_db = _recdb()
_rot_case = [0]


def _rot_angle(z, **kw):
    """Measure the lift-off angle of one synthetic trace set. -> (deg, detail)"""
    _rot_case[0] += 1
    path = "/rot/case%d" % _rot_case[0]
    _trace_list(_rot_db, np.asarray(z).reshape(-1, 1), path=path)
    return ns["ec_liftoff_rotation_from_traces"](_rot_db, path, **kw)


def _traces_along(angle_deg, n=800, along=0.4, across=0.01, centre=(1.2, -0.35),
                  flaws=(), seed=0):
    """Traces scattered along one impedance-plane direction, plus optional flaw
    excursions off it."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal(n)*along
    perp = rng.standard_normal(n)*across
    a = math.radians(angle_deg)
    re = centre[0] + t*math.cos(a) - perp*math.sin(a)
    im = centre[1] + t*math.sin(a) + perp*math.cos(a)
    z = re + 1j*im
    for where, mag, ang in flaws:
        b = math.radians(ang)
        z[where] += mag*complex(math.cos(b), math.sin(b))
    return z


def _axis_err(got, planted):
    """Angle error treating the axis as a LINE: theta and theta+180 agree."""
    d = abs(math.degrees(math.atan2(math.sin(math.radians(got - planted)),
                                    math.cos(math.radians(got - planted)))))
    return min(d, abs(180.0 - d))


for _planted in (-66.0, -30.0, 0.0, 17.5, 72.0):
    _got, _detail = _rot_angle(_traces_along(_planted))
    check(_axis_err(_got, _planted) < 2.0,
          "planted %+.1f deg, recovered %+.2f deg (err %.2f)"
          % (_planted, _got, _axis_err(_got, _planted)))
check(_detail["anisotropy"] > 10.0,
      "...and it reports how clearly there was a direction (%.1fx)"
      % _detail["anisotropy"])

# A scan does not know where its flaws are, so the angle has to survive them
# being left in. They are sparse and lie off the lift-off axis, so they perturb
# it rather than defining it.
_got, _ = _rot_angle(_traces_along(-66.0, n=800,
                                   flaws=[(np.arange(0, 800, 37), 0.25, 20.0)],
                                   seed=3))
check(_axis_err(_got, -66.0) < 5.0,
      "flaws left in move the angle but do not set it: %+.2f deg (err %.2f)"
      % (_got, _axis_err(_got, -66.0)))

# No direction means no angle. Returning one anyway would rotate the image by
# something arbitrary and make it look conditioned.
try:
    _rot_angle(_traces_along(0.0, along=0.05, across=0.05))
    check(False, "a round scatter should have been refused")
except ValueError as _e:
    check("anisotropy" in str(_e),
          "a round scatter is refused rather than given an arbitrary angle")

# An angle from a handful of traces is noise with a decimal point.
try:
    _rot_angle(_traces_along(-66.0, n=40))
    check(False, "40 traces should have been refused")
except ValueError as _e:
    check("min_traces" in str(_e) or "are wanted" in str(_e),
          "too few traces is refused")

# UNWRITTEN SLOTS MUST NOT SET THE DIRECTION. The trace list is allocated for
# the PLANNED trace count, so a scan that drops a pass leaves the tail at 0+0j.
# Those sit about 1.25 V from a cluster that spans 0.01 V, so each carries over
# a hundred times the leverage of a real reading. Left in, they turned a -65 deg
# axis into -18 and the image was conditioned with a rotation wrong by 47 deg.
_z = _traces_along(-66.0, n=800)
_padded = np.concatenate([_z, np.zeros(16, dtype=complex)])
_got, _detail = _rot_angle(_padded)
check(_axis_err(_got, -66.0) < 2.0,
      "16 unwritten slots among 800 traces do not move the axis: %+.2f deg" % _got)
check(_detail["unwritten_dropped"] == 16 and _detail["traces"] == 800,
      "...and it says it ignored them (%d dropped, %d kept)"
      % (_detail["unwritten_dropped"], _detail["traces"]))

# theta and theta-180 are the same rotation, so two scans of one setup must not
# report numbers that differ by a half turn.
for _planted in (120.0, -120.0, 200.0):
    _got, _ = _rot_angle(_traces_along(_planted))
    check(-90.0 < _got <= 90.0,
          "planted %+.1f reports %+.2f, inside (-90, 90]" % (_planted, _got))


def test_ec_scan_wiring():
    """So pytest reports what running the file reports."""
    assert not FAILURES, "%d check(s) failed: %s" % (len(FAILURES),
                                                     "; ".join(FAILURES))


if __name__ == "__main__":
    print("")
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("all passed")
