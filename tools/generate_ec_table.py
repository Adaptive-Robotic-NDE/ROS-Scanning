#!/usr/bin/env python3
"""Generate the eddy-current response table this cell's simulator consumes.

Runs against the `ecsim` package (https://github.com/.../ecsim, Apache-2.0,
developed separately). It is a standalone script: copy it to whatever machine
has ecsim installed, run it, copy the two output files back.

    python3 generate_ec_table.py --out-dir /somewhere/with/space

It is RESUMABLE. Interrupt it and run it again with the same --out-dir and it
picks up from the written mask rather than starting over. It prints a measured
ETA after the first few poses, before committing to the long run.

WHAT IT PRODUCES

    al6061_2mhz_response.h5    the flaw response, schema per
                               docs/ec-probe-model-spec.md
    al6061_2mhz_baseline.npz   the NO-FLAW lift-off/tilt trajectory

Two files rather than one because the schema ties its /baseline dataset to the
factorised lift-off/tilt model, and ecsim measured that lift-off and tilt do
NOT factor out of the flaw response (15-60% residual). So this table is dense in
lift-off and tilt, which leaves nowhere in the schema for the flaw-free
trajectory -- and we need that trajectory, because with no flaw present it is
the entire signal, and it is what the cell's conditioning procedure rotates onto
the horizontal axis. It is written beside the table instead.

SCOPE, AND WHY IT IS NOT THE FULL SPEC TABLE

The spec asks for a table dense enough to interpolate arbitrary flaws. That is
a multi-day job. This one is deliberately much smaller, because its job is to
drive a SIMULATED SPECIMEN whose flaws we choose ourselves: putting those flaws
exactly on table nodes removes the need to interpolate flaw geometry at all.
What stays dense is PROBE POSE -- lift-off, both tilts, yaw and position --
because that is what varies continuously as the arm flies a pass, and it is
where the noise we are trying to study lives.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# THE PROBE.
#
# UniWest PDS-.187-2MHz: 2 MHz differential shielded split-D pencil probe,
# 0.187 in shaft. The shaft is the only dimension we have; the coil geometry
# below is a DOCUMENTED PLACEHOLDER chosen to be plausible inside that shaft,
# NOT a fit and NOT a measurement.
#
# That is deliberate and it is fine for the present purpose: this table exists
# to exercise the scanning pipeline -- conditioning, projection, the cleanup
# we are trying to tune -- and for that it has to be REPRESENTATIVE, not
# predictive. Absolute amplitudes will be wrong until the geometry is fitted.
#
# To fit it: ecsim/scripts/fit_d20.py does exactly this against the VIC-3D
# table, and the same approach applies to our measured tilt ladder once there
# is an unclipped one with the probe's rotation about its own axis recorded.
# Then change these numbers, rerun, and the provenance records which was used.
# ---------------------------------------------------------------------------
PROBE = dict(
    r_inner=0.0004,          # m
    r_outer=0.0016,          # m
    height=0.0012,           # m
    turns=150.0,
    gap=0.0001,              # m, the split
    # THE WINDING IS NOT AT THE SURFACE. A real probe has a wear face, and 0.1
    # mm is reasonable for this one. Zero is not merely optimistic, it is what
    # stalled the first attempted run: at zero stand-off the lowest filament
    # sits 40 um above the conductor, the incident-field map sizes its spectral
    # grid to resolve that, and one coil map then takes tens of minutes. Worse,
    # the flaw-major task order puts every lift-off-zero pose first, so every
    # worker started on the worst case at once -- high CPU, almost no memory,
    # nothing written. ecsim now bounds that grid and warns below 50 um, but
    # zero is physically wrong regardless. At 0.1 mm a coil map is ~48 s at
    # contact and ~10 s at 0.5 mm lift-off.
    z_offset=0.0001,         # m, wear face between winding and surface
    shaft_diameter=0.0047498,   # 0.187 in, the one number we actually know
)
FREQUENCY_HZ = 2.0e6
SHIELDED = True

# ---------------------------------------------------------------------------
# THE GRID. Every axis is in SI (metres, radians) on disk; the readable values
# here are in mm and degrees and converted below.
#
# Flaw lengths are the UniWest 99073 standard's four notch lengths exactly, so
# a simulated specimen can carry the same set as the real calibration block and
# the two are directly comparable.
# ---------------------------------------------------------------------------
DEPTH_MM = [0.20, 0.50]
LENGTH_MM = [1.016, 1.524, 2.032, 2.540]      # 0.040, 0.060, 0.080, 0.100 in
WIDTH_MM = [0.15]
YAW_DEG = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# LIFT-OFF HERE MEANS STANDOFF ABOVE EDGE CONTACT, not absolute height.
#
# A rigid probe face cannot be tilted while its centre stays down: tilting by
# theta about the contact edge lifts the centre by footprint_radius*sin(theta),
# which for this probe's 3.25 mm footprint is 0.28 mm at 5 deg and 0.56 mm at
# 10 deg. Asking ecsim for 10 deg of tilt at 0.1 mm of absolute lift-off puts
# the coil BELOW the surface and it refuses -- correctly, because that pose
# does not exist.
#
# So the axis is standoff ABOVE that geometric minimum, and the absolute
# lift-off handed to the solver is edge_contact_liftoff(tilt) + this. That
# makes the (lift-off, tilt) grid rectangular and every cell physically
# reachable, and it matches our probe: it is spring-loaded and in contact, so
# it rides on its edge at any tilt and mesh error adds standoff on top.
#
# Sampled fine near zero because that is where the response moves fastest and
# where we operate: measured preload is 0.45 mm of spring compression against
# +/-0.5 mm of mesh error.
LIFTOFF_MM = [0.0, 0.10, 0.25, 0.50, 1.00]

# tilt_parallel is the FLAW-LIKE one -- about the split line, raising one D and
# lowering the other. tilt_perpendicular is common-mode and gets fewer nodes
# because it should behave much like uniform lift-off. See the spec's tilt
# section; ecsim calls these tilt_y and tilt_x respectively.
TILT_PARALLEL_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0]
TILT_PERPENDICULAR_DEG = [-10.0, 0.0, 10.0]

# Probe-frame field. +/-6 mm because the shaft alone is 4.75 mm and we need
# clean baseline either side of the response.
#
# 0.25 mm, NOT 0.1. This is the single biggest cost lever and it was measured
# rather than guessed: one 101x101 field at 0.1 mm took 635 s, which over the
# 4200 poses below is 741 CORE-HOURS. At 0.25 mm the grid is 49x49, about 4.3x
# fewer points, and the cost falls with it. That is still 4x oversampled
# against our 1 mm trace pitch, and the response is several mm wide, so nothing
# physical is lost -- what is lost is headroom to study much finer pitches,
# which is not what this table is for.
FIELD_HALF_WIDTH_M = 0.006
FIELD_STEP_M = 0.00025


def build(args):
    from ecsim.material import ALUMINIUM_6061_T6
    from ecsim.probe import split_d_differential
    from ecsim.response import Simulation

    probe = split_d_differential(**PROBE)
    sim = Simulation(probe=probe, material=ALUMINIUM_6061_T6,
                     frequency=FREQUENCY_HZ, solver=args.solver,
                     backend=getattr(args, "backend", "numpy"),
                     precision=getattr(args, "precision", "double"))
    return probe, ALUMINIUM_6061_T6, sim


def axes():
    mm = 1e-3
    return {
        "depth": np.array(DEPTH_MM) * mm,
        "length": np.array(LENGTH_MM) * mm,
        "width": np.array(WIDTH_MM) * mm,
        "yaw": np.radians(YAW_DEG),
        "liftoff": np.array(LIFTOFF_MM) * mm,
        "tilt_parallel": np.radians(TILT_PARALLEL_DEG),
        "tilt_perpendicular": np.radians(TILT_PERPENDICULAR_DEG),
        "y": np.arange(-FIELD_HALF_WIDTH_M, FIELD_HALF_WIDTH_M + 0.5*FIELD_STEP_M,
                       FIELD_STEP_M),
        "x": np.arange(-FIELD_HALF_WIDTH_M, FIELD_HALF_WIDTH_M + 0.5*FIELD_STEP_M,
                       FIELD_STEP_M),
    }


#: Filled in by main() so the table records how it was computed.
PRECISION = ["double"]


def provenance(material):
    return {
        "frequency_hz": FREQUENCY_HZ,
        "material": material.name,
        "conductivity_s_per_m": material.conductivity,
        "permeability_relative": material.mu_r,
        "probe_type": "differential shielded split-D pencil, UniWest PDS-.187-2MHz",
        "probe_shaft_diameter_m": PROBE["shaft_diameter"],
        "coil_diameter_m": 2.0 * PROBE["r_outer"],
        # Centre-to-centre, which is what the spec means by separation; the
        # gap is the slot between the two chords and is a different number.
        # The centroid of a half-annulus sits 4(ro^3-ri^3)/(3 pi (ro^2-ri^2))
        # from its flat.
        "coil_separation_m": 2.0*(
            4.0*(PROBE["r_outer"]**3 - PROBE["r_inner"]**3)
            / (3.0*math.pi*(PROBE["r_outer"]**2 - PROBE["r_inner"]**2))
            + 0.5*PROBE["gap"]),
        "coil_gap_m": PROBE["gap"],
        "coil_z_offset_m": PROBE["z_offset"],
        "shielded": SHIELDED,
        "solver_precision": PRECISION[0],
        "notes": (
            "response = flaw-induced CHANGE in impedance (ohms) relative to the "
            "flaw-free baseline at the same lift-off and tilt. No gain and no "
            "rotation applied. Probe-centred coordinates, x along the scan "
            "direction. THE LIFTOFF AXIS IS STANDOFF ABOVE EDGE CONTACT, not "
            "absolute height: the solver was given "
            "edge_contact_liftoff(tilt) + axis value, because a rigid face "
            "cannot be tilted with its centre still down. "
            "tilt_parallel is rotation about the SPLIT LINE and is "
            "the differential, flaw-like tilt (ecsim tilt_y); "
            "tilt_perpendicular is about the line joining the coil centres and "
            "is common-mode (ecsim tilt_x). "
            "COIL GEOMETRY IS A PLACEHOLDER, not a fit: only the shaft "
            "diameter is known for this probe, so absolute amplitudes are not "
            "to be trusted. The flaw-free lift-off/tilt trajectory is NOT in "
            "this file -- lift-off and tilt do not factor out of the flaw "
            "response, so this table is dense in them and the baseline is "
            "written beside it as al6061_2mhz_baseline.npz."
        ),
    }


# ---------------------------------------------------------------------------
# Parallel evaluation.
#
# One pose is one flaw_field call and they are completely independent, so the
# run is embarrassingly parallel -- which is the whole reason the measured
# 741 core-hours at 0.1 mm was a planning number rather than a wall-clock one.
# HDF5 stays single-writer: workers compute and hand fields back, the parent
# writes. Each worker keeps its own flaw systems, so sorting the work
# flaw-major means a worker usually reuses the one it already built.
# ---------------------------------------------------------------------------

_W = {}


def _setup(solver, backend, precision, ax):
    """Initialise this worker once, lazily on its first task.

    NOT a Pool initializer. An initializer that raises leaves
    multiprocessing waiting for a worker that will never answer, and the run
    hangs with no message -- the worst possible failure for a job left running
    overnight on someone else's machine. Done this way, a failure surfaces as
    an exception on the future and the parent says which pose it was.

    The axes are PASSED IN, not re-derived from the module globals: under the
    "spawn" start method, which Windows has no alternative to, each child
    re-imports this module fresh, so anything the parent changed at runtime
    would be invisible and the workers would compute a different grid from the
    one being indexed.
    """
    if _W.get("ready"):
        return
    class A:
        pass
    a = A()
    a.solver = solver
    a.backend = backend
    a.precision = precision
    probe, material, sim = build(a)
    _W.update(probe=probe, sim=sim, ax=ax, systems={}, ready=True,
              backend=backend, precision=precision)


def _one_pose(task):
    """Compute one field. -> (idx, field).

    Set ECTABLE_FAKE=1 to return a cheap synthetic field instead of solving.
    That is not a simulation of anything -- it exists so the parts of this
    script that have actually had bugs (sharding, the two-lane dispatcher,
    resume, merge) can be exercised end to end in seconds rather than hidden
    behind a solver that takes minutes per pose.
    """
    idx, solver, backend, precision, ax_in = task
    if os.environ.get("ECTABLE_FAKE"):
        import time as _t
        _t.sleep(0.02 if backend == "cupy" else 0.05)   # GPU lane "faster"
        ny, nx = ax_in["y"].size, ax_in["x"].size
        val = complex(sum(idx) + 1, backend == "cupy")
        return idx, np.full((ny, nx), val, dtype=np.complex64)
    from ecsim.flaw import RectangularNotch
    _setup(solver, backend, precision, ax_in)
    di, li, wi, yi, oi, tpi, tqi = idx
    ax, sim, probe = _W["ax"], _W["sim"], _W["probe"]
    key = (di, li, wi)
    if key not in _W["systems"]:
        flaw = RectangularNotch(length=float(ax["length"][li]),
                                width=float(ax["width"][wi]),
                                depth=float(ax["depth"][di]))
        _W["systems"] = {key: (flaw, sim.flaw_system(flaw))}   # keep one
    flaw, system = _W["systems"][key]
    tilt_par = float(ax["tilt_parallel"][tpi])
    tilt_perp = float(ax["tilt_perpendicular"][tqi])
    # BOTH angles together. A face inclined about two axes drops further than
    # either alone, so max() understates it -- 0.56 mm against the true 0.79 mm
    # at 10 deg / 10 deg -- and the pose would be driven into the surface.
    edge = probe.edge_contact_liftoff(tilt_perp, tilt_par)
    field = sim.flaw_field(flaw, ax["x"], ax["y"],
                           liftoff=float(ax["liftoff"][oi]) + edge,
                           yaw=float(ax["yaw"][yi]),
                           tilt_y=tilt_par, tilt_x=tilt_perp, system=system)
    return idx, np.asarray(field, dtype=np.complex64)


def merge(args):
    """Combine shard files into one table. Fields are disjoint by construction,
    so this is a copy guided by each shard's written mask."""
    import h5py
    out = os.path.join(args.out_dir, "al6061_2mhz_response.h5")
    os.makedirs(args.out_dir, exist_ok=True)
    srcs = list(args.merge)
    print("merging %d shard(s) into %s" % (len(srcs), out))
    import shutil
    shutil.copyfile(srcs[0], out)
    total = 0
    with h5py.File(out, "r+") as dst:
        for src_path in srcs[1:]:
            with h5py.File(src_path, "r") as src:
                if src["/response"].shape != dst["/response"].shape:
                    raise SystemExit("shape mismatch: %s" % src_path)
                have = np.asarray(src["/written"])
                add = have & ~np.asarray(dst["/written"])
                for idx in np.argwhere(add):
                    t = tuple(int(i) for i in idx)
                    dst["/response"][t] = src["/response"][t]
                    dst["/written"][t] = True
                    total += 1
            print("   %s contributed %d pose(s)" % (os.path.basename(src_path), total))
        done = int(np.asarray(dst["/written"]).sum())
        size = int(np.asarray(dst["/written"]).size)
        print("merged table: %d of %d poses written" % (done, size))
        if done != size:
            print("   WARNING: %d poses are still missing" % (size - done))

    # AND THE FLAW-FREE TRAJECTORY, which is not in the table and has to end up
    # beside it. Every shard writes an identical copy -- it is seconds of work
    # and needs no flaw system -- so any one of them will do; the consumer finds
    # it by name, replacing _response.h5 with _baseline.npz.
    #
    # Copied here rather than left to the operator because a merged table
    # WITHOUT it is the worst outcome available: it loads, it validates, every
    # flaw reads correctly, and the probe reads nothing at all over good
    # material however far it is tilted -- which is the one effect this table
    # was generated to show.
    base_out = os.path.join(args.out_dir, "al6061_2mhz_baseline.npz")
    for src_path in srcs:
        cand = os.path.join(os.path.dirname(os.path.abspath(src_path)),
                            "al6061_2mhz_baseline.npz")
        if os.path.isfile(cand):
            if os.path.abspath(cand) != os.path.abspath(base_out):
                shutil.copyfile(cand, base_out)
            print("   baseline from %s" % cand)
            break
    else:
        print("   WARNING: NO al6061_2mhz_baseline.npz found beside any shard. "
              "The merged table will load and validate, and the simulated "
              "probe will read nothing over good material at any tilt -- "
              "which is the effect the table exists to show. Copy one in "
              "beside %s before using it." % (out,))
    from ecsim.table import validate_file
    validate_file(out)
    print("ok")
    return 0

def main(argv=None):
    # Declared here because both are read in the argument help below and
    # rebound further down; Python wants the declaration before either.
    global FIELD_STEP_M, FIELD_HALF_WIDTH_M
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".", help="where the two output files go")
    ap.add_argument("--solver", default="fft", choices=("fft", "dense"),
                    help="fft is far faster for a field; dense is the reference")
    ap.add_argument("--probe-poses", type=int, default=3,
                    help="minimum poses to time before estimating the whole "
                         "run; raised to the worker count, because an estimate "
                         "taken before one full wave has landed is wrong by "
                         "the concurrency factor")
    ap.add_argument("--yes", action="store_true",
                    help="do not stop for confirmation after the estimate. USE "
                         "THIS FOR ANY UNATTENDED RUN: otherwise the parent "
                         "blocks on the prompt while every worker keeps "
                         "burning CPU on the task it already has")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard this machine computes, 0-based")
    ap.add_argument("--of", type=int, default=1,
                    help="how many machines the run is split across")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes; 0 means HALF the cores, "
                         "deliberately leaving the machine usable")
    ap.add_argument("--field-step-mm", type=float, default=None,
                    help="spacing of the probe-frame field. Default %.2f mm. "
                         "Cost is dominated by per-pose setup rather than by "
                         "field points, so a finer field is cheaper than it "
                         "looks -- but it is also memory per worker, which is "
                         "what actually runs a machine out" % (FIELD_STEP_M*1e3,))
    ap.add_argument("--field-halfwidth-mm", type=float, default=None,
                    help="half-extent of the field. Default %.1f mm"
                         % (FIELD_HALF_WIDTH_M*1e3,))
    ap.add_argument("--precision", default="double", choices=("double", "single"),
                    help="solver precision. The table is STORED as complex64 "
                         "either way, so single carries no less than the "
                         "deliverable keeps -- and it is what makes a GPU "
                         "worth using, since consumer and RTX workstation "
                         "silicon runs FP64 at 1/32 to 1/64 of its FP32 rate. "
                         "Verify with scripts/gpu_check.py on the machine "
                         "before trusting it for a whole run")
    ap.add_argument("--gpu-workers", type=int, default=0,
                    help="processes using the CuPy GPU backend, ALONGSIDE the "
                         "CPU workers. 1 per NVIDIA GPU is the usual answer; "
                         "0 disables the GPU entirely")
    ap.add_argument("--merge", nargs="+", metavar="SHARD",
                    help="merge finished shard files into --out-dir and exit")
    args = ap.parse_args(argv)
    if args.merge:
        return merge(args)
    if not (0 <= args.shard < args.of):
        ap.error("--shard must be in [0, --of)")

    from ecsim.flaw import RectangularNotch
    from ecsim.table import ResponseTableWriter, validate_file

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.of == 1 else ".shard%dof%d" % (args.shard, args.of)
    table_path = os.path.join(args.out_dir,
                              "al6061_2mhz_response%s.h5" % suffix)
    base_path = os.path.join(args.out_dir, "al6061_2mhz_baseline.npz")

    PRECISION[0] = args.precision
    if args.field_step_mm:
        FIELD_STEP_M = args.field_step_mm*1e-3
    if args.field_halfwidth_mm:
        FIELD_HALF_WIDTH_M = args.field_halfwidth_mm*1e-3
    probe, material, sim = build(args)
    ax = axes()
    shape = tuple(ax[k].size for k in
                  ("depth", "length", "width", "yaw", "liftoff",
                   "tilt_parallel", "tilt_perpendicular"))
    n_poses = int(np.prod(shape))
    n_cells = ax["y"].size * ax["x"].size

    print("eddy-current response table")
    print("  material     %s, sigma %.3g S/m, mu_r %.1f"
          % (material.name, material.conductivity, material.mu_r))
    print("  frequency    %.3f MHz   skin depth %.4f mm"
          % (FREQUENCY_HZ/1e6, sim.skin_depth()*1e3))
    print("  probe        split-D differential, shaft %.3f mm, coil OD %.2f mm"
          % (PROBE["shaft_diameter"]*1e3, 2e3*PROBE["r_outer"]))
    print("  grid         %s = %d poses, field %d x %d"
          % (" x ".join(str(s) for s in shape), n_poses, ax["y"].size, ax["x"].size))
    print("  table        %.0f MB complex64" % (n_poses*n_cells*8/1e6,))
    print("  field        +/-%.1f mm at %.2f mm = %d x %d point(s)"
          % (FIELD_HALF_WIDTH_M*1e3, FIELD_STEP_M*1e3,
             ax["y"].size, ax["x"].size))
    print("  solver       %s, %s precision" % (args.solver, args.precision))
    # MEMORY IS WHAT ENDS A RUN, NOT CPU: a worker holds its flaw system, the
    # incident-field images for both coils, and the reduced basis. Since the
    # image spacing was tied to the field's own scale it is well under 1 GB per
    # worker on this grid -- it was the earlier spacing that could reach 100 GB
    # across 61 workers. No formula here is worth trusting over the first few
    # minutes of Task Manager.
    print("  memory       expect well under 1 GB per worker; WATCH THE FIRST "
          "MINUTES and set --workers from what you see")
    print()

    # Resume if the file is already here, so an interrupted run on a borrowed
    # machine is not a lost day.
    if os.path.exists(table_path):
        writer = ResponseTableWriter.resume(table_path)
        print("resuming %s" % table_path)
    else:
        writer = ResponseTableWriter(table_path)
        writer.create(axes=ax, attrs=provenance(material))
        print("created %s" % table_path)

    try:
        done = np.asarray(writer.written_mask)
        todo = [tuple(int(i) for i in idx) for idx in np.argwhere(~done)]
        # Flaw-major order, then split into contiguous blocks: each machine
        # gets whole flaws where it can, so it builds few flaw systems.
        todo.sort()
        if args.of > 1:
            bounds = np.linspace(0, len(todo), args.of + 1).astype(int)
            mine = todo[bounds[args.shard]:bounds[args.shard + 1]]
            print("shard %d of %d: %d of %d outstanding poses"
                  % (args.shard, args.of, len(mine), len(todo)))
        else:
            mine = todo
            print("%d of %d poses still to do" % (len(mine), n_poses))
        if not mine:
            print("nothing to do")
            return 0

        # HALF the cores by default, not all of them. Each worker holds its
        # own flaw system and its own FFT basis, so the limit that bites first
        # is memory rather than CPU, and a box that starts swapping takes far
        # longer than one that is merely half-loaded. Raise it with --workers
        # once you have watched one run's memory.
        n_cpu = os.cpu_count() or 1
        n_workers = args.workers if args.workers is not None else max(1, n_cpu//2)
        n_gpu = args.gpu_workers
        print("%d CPU worker(s) of %d cores%s, %d GPU worker(s)\n"
              % (n_workers, n_cpu,
                 "" if args.workers else " (half, by default)", n_gpu))

        import concurrent.futures as cf
        import multiprocessing as mp
        from collections import deque

        # TWO LANES, ONE QUEUE.
        #
        # The GPU and the CPU cores are both useful and they are not
        # interchangeable: a GPU worker is fast but there are one or two of
        # them, a CPU worker is slow but there are many. Rather than guess the
        # ratio and split the work up front -- which strands one lane waiting
        # for the other -- both lanes pull from a single pending queue and are
        # topped up as they finish. Whichever is faster simply takes more. No
        # tuning, and no idle hardware at the end of the run.
        #
        # Each lane is its own process pool, so a worker only ever sees one
        # backend and CuPy never has to share a process with numpy work.
        ctx = mp.get_context("spawn")
        lanes = []
        if n_gpu:
            lanes.append({"name": "gpu", "backend": "cupy", "n": n_gpu})
        if n_workers:
            lanes.append({"name": "cpu", "backend": "numpy", "n": n_workers})
        if not lanes:
            print("no workers requested")
            return 2
        for lane in lanes:
            lane["pool"] = cf.ProcessPoolExecutor(max_workers=lane["n"],
                                                  mp_context=ctx)
            lane["done"] = 0
            lane["seconds"] = 0.0

        pending = deque(mine)
        futures = {}
        t_start = time.time()

        def submit(lane):
            if not pending:
                return
            idx = pending.popleft()
            fut = lane["pool"].submit(
                _one_pose, (idx, args.solver, lane["backend"],
                            args.precision, ax))
            futures[fut] = (lane, idx, time.time())

        for lane in lanes:
            for _ in range(lane["n"]):
                submit(lane)

        # DO NOT ESTIMATE BEFORE ONE FULL WAVE HAS LANDED.
        #
        # elapsed/completed only measures throughput once completions are
        # steady. With W workers in flight the first W finish together after
        # roughly one pose time, so estimating at 3 completions gives
        # (one pose)/3 and then multiplies by everything remaining -- which
        # with 61 workers overstates the run by about twenty times. Waiting
        # for W completions makes the same arithmetic correct, because by then
        # the rate really is W poses per pose-time.
        n_total_workers = sum(l["n"] for l in lanes)
        probe_after = max(args.probe_poses, n_total_workers)
        probe_after = min(probe_after, len(mine))

        n = 0
        failures = 0
        try:
            while futures:
                ready, _ = cf.wait(list(futures),
                                   return_when=cf.FIRST_COMPLETED)
                for fut in ready:
                    lane, idx, t_sub = futures.pop(fut)
                    try:
                        got_idx, field = fut.result()
                        writer.write_field(got_idx, field)
                    except Exception as exc:
                        failures += 1
                        print("   POSE %s FAILED on the %s lane: %s"
                              % (idx, lane["name"], exc))
                    lane["done"] += 1
                    lane["seconds"] += time.time() - t_sub
                    n += 1
                    submit(lane)

                    if n == probe_after:
                        per = (time.time() - t_start)/n
                        print("\n   %.1f s per completed pose across %s "
                              "(measured over %d, one full wave)"
                              % (per, " + ".join("%d %s" % (l["n"], l["name"])
                                                 for l in lanes), n))
                        for l in lanes:
                            if l["done"]:
                                print("      %s lane: %d pose(s), %.0f s each"
                                      % (l["name"], l["done"],
                                         l["seconds"]/l["done"]))
                        print("   ESTIMATED REMAINING: %.1f h for %d poses"
                              % (per*(len(mine) - n)/3600.0, len(mine) - n))
                        if not args.yes and sys.stdin.isatty():
                            if input("   continue? [y/N] ").strip().lower() != "y":
                                print("   stopped; rerun to resume")
                                return 0
                        print()
                    if n % 50 == 0:
                        per = (time.time() - t_start)/n
                        print("   %d/%d  %.1f s/pose  %.1f h left"
                              % (n, len(mine), per,
                                 per*(len(mine) - n)/3600.0))
        finally:
            for lane in lanes:
                lane["pool"].shutdown(wait=False, cancel_futures=True)

        print("\n   lane throughput:")
        for l in lanes:
            if l["done"]:
                print("      %-3s %5d pose(s)  %6.0f s each"
                      % (l["name"], l["done"], l["seconds"]/l["done"]))
        if failures:
            print("   %d pose(s) FAILED; rerun to retry just those" % failures)
    finally:
        writer.close()

    # The flaw-free trajectory. Cheap -- no flaw system -- so it is sampled on
    # the same lift-off/tilt grid and costs minutes, not hours.
    print("\nbaseline (no flaw) over the same lift-off/tilt grid")
    base = np.zeros((ax["liftoff"].size, ax["tilt_parallel"].size,
                     ax["tilt_perpendicular"].size), dtype=np.complex128)
    t0 = time.time()
    for a, lo in enumerate(ax["liftoff"]):
        for b, tp in enumerate(ax["tilt_parallel"]):
            for c, tq in enumerate(ax["tilt_perpendicular"]):
                edge = probe.edge_contact_liftoff(float(tq), float(tp))
                base[a, b, c] = sim.liftoff_tilt_response(
                    liftoff=float(lo) + edge,
                    tilt_y=float(tp), tilt_x=float(tq))
    print("   %d poses in %.1f s" % (base.size, time.time()-t0))

    np.savez(base_path, baseline=base.astype(np.complex64),
             liftoff=ax["liftoff"], tilt_parallel=ax["tilt_parallel"],
             tilt_perpendicular=ax["tilt_perpendicular"],
             provenance=json.dumps(provenance(material)))
    print("   wrote %s" % base_path)

    print("\nvalidating %s" % table_path)
    validate_file(table_path)
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
