#!/usr/bin/env python3
"""A STUB eddy-current response table: the real schema, invented physics.

    python3 make_ec_stub_table.py --out-dir /tmp/ecstub

WHY THIS EXISTS. The real table (tools/generate_ec_table.py, run against ecsim
on a machine with the compute for it) takes a night. Everything that consumes
the table -- the reader, the plate model, the simulated instrument, the
conditioning, the projection -- can be written and tested before that night
finishes, but only against a file. This writes that file.

IT IS NOT A MODEL OF AN EDDY-CURRENT PROBE. The fields here are closed-form
expressions chosen so a test can predict the answer exactly, which is the
opposite of what a physics model is for. Nothing it produces is a measurement
and no number from it should ever appear in a result. It is the same bargain
sim_ec_source.dpi's synthetic signal already makes, one layer lower down.

What it DOES reproduce faithfully, because these are what the consumer has to
get right:

  * the on-disk schema, byte for byte -- dataset paths, axis names, axis
    units, the required provenance attributes, and the private
    _param_axis_order attribute the reader uses to know the axis order. A
    consumer that works here works on the real file.
  * DENSE lift-off and tilt, matching generate_ec_table.py. The spec invited a
    factorised G(liftoff, tilt) and ecsim measured that lift-off and tilt do
    NOT factor out (15-60% residual), so the real table is dense and so is
    this. A consumer written against the factorised form would not survive the
    switch.
  * the separate baseline .npz, for the same reason the real generator writes
    one: the schema ties /baseline to the factorised model, so a dense table
    has nowhere to put the flaw-free trajectory, and the flaw-free trajectory
    is the entire signal when there is no flaw under the probe.
  * the LIFT-OFF AXIS MEANING: standoff above edge contact, not absolute
    height.
  * a DOUBLE-LOBED flaw signature, antisymmetric in x. A differential probe's
    response goes positive, through a null, to negative as the flaw passes
    under it, and a consumer tested only against a single bump can be
    sign-blind in a way that never shows up.
  * tilt_parallel mattering far more than tilt_perpendicular, and both
    mattering more than the flaw. That ordering is the whole problem the real
    table exists to let us study, and a stub that got it backwards would make
    the pipeline look better behaved than it is.

The flaw field is

    dZ(x, y) = A * (x/sx) * exp(-(x/sx)^2/2 - (y/sy)^2/2) * exp(i*phi)

with sx, sy and A simple functions of the flaw axes and of yaw, so a test can
evaluate the same expression and compare. The baseline is linear in standoff
and quadratic in each tilt, again so a test can predict it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np

# The real table's grid, from tools/generate_ec_table.py. Kept identical so a
# consumer exercised here meets the same axis lengths, the same single-node
# width axis, and the same yaw range it will meet on the real file.
DEPTH_MM = [0.20, 0.50]
LENGTH_MM = [1.016, 1.524, 2.032, 2.540]
WIDTH_MM = [0.15]
YAW_DEG = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
LIFTOFF_MM = [0.0, 0.10, 0.25, 0.50, 1.00]
TILT_PARALLEL_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0]
TILT_PERPENDICULAR_DEG = [-10.0, 0.0, 10.0]

# The field, COARSER than the real table's 0.25 mm. 1 mm gives 13x13 and a
# 22 MB file instead of 49x49 and 310 MB, which matters when this is written
# into a temporary directory by every test run. The consumer never assumes a
# step, so a coarse one exercises it just as well -- and a stub whose grid
# differed in SHAPE from the real one would hide an axis-order mistake, which
# is why the parameter axes above are kept exact and only the step moves.
FIELD_HALF_WIDTH_M = 0.006
FIELD_STEP_M = 0.001

#: Peak |dZ| of the largest flaw, ohms. Order of magnitude from the real
#: solver's first full run (peak |dZ| 0.788 ohms at 101x101).
FLAW_PEAK_OHMS = 0.8

#: Ohms per degree of tilt_parallel, and the much smaller perpendicular
#: sensitivity. The parallel figure is the order of magnitude ecsim reported
#: for this probe (7.66 ohms at 2 degrees of tilt_y), which is what makes tilt
#: the dominant signal and the flaw the small one.
TILT_PAR_OHMS_PER_DEG = 3.8
TILT_PERP_OHMS_PER_DEG2 = 0.02

#: Residual imbalance, ohms: what the probe reads over good material square
#: on. A real differential probe does not cancel to nothing, and if this were
#: zero then lift-off alone would move the spot nowhere and the work cell's
#: conditioning procedure would have no lift-off direction to find.
COMMON_OHMS = 2.2

#: How fast the flaw-free reading decays as the probe leaves the part. The
#: conductor-induced part of the impedance tends to zero far from the surface,
#: and ec_response_table.py recovers THIS NUMBER from the written baseline to
#: decide how to treat a standoff past the table's last node -- so the stub
#: has to actually decay, or that path is never exercised.
LIFTOFF_DECAY_M = 0.0004

#: Phase of the flaw response and of the lift-off/tilt trajectory, radians.
#: DIFFERENT, and that difference is the point: it is the angle the work cell's
#: conditioning procedure rotates onto an axis, so a stub where they coincided
#: would make set_rotation() untestable.
FLAW_PHASE_RAD = math.radians(109.0)
LIFTOFF_PHASE_RAD = math.radians(35.0)


def axes():
    """The axis node values, SI, exactly as the real generator computes them."""
    mm = 1e-3
    half, step = FIELD_HALF_WIDTH_M, FIELD_STEP_M
    return {
        "depth": np.array(DEPTH_MM)*mm,
        "length": np.array(LENGTH_MM)*mm,
        "width": np.array(WIDTH_MM)*mm,
        "yaw": np.radians(YAW_DEG),
        "liftoff": np.array(LIFTOFF_MM)*mm,
        "tilt_parallel": np.radians(TILT_PARALLEL_DEG),
        "tilt_perpendicular": np.radians(TILT_PERPENDICULAR_DEG),
        "y": np.arange(-half, half + 0.5*step, step),
        "x": np.arange(-half, half + 0.5*step, step),
    }


PARAM_AXES = ("depth", "length", "width", "yaw",
              "liftoff", "tilt_parallel", "tilt_perpendicular")


def flaw_field(depth, length, width, yaw, liftoff, tilt_par, tilt_perp, xs, ys):
    """One pose's flaw field, shape (len(ys), len(xs)), complex ohms.

    Closed form so a test can evaluate it independently. The x factor makes it
    antisymmetric -- the double lobe a differential probe gives -- and the yaw
    factor makes a flaw aligned with the probe's own axis nearly invisible,
    which is the behaviour that makes yaw worth having as an axis at all.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    # Footprint: a longer flaw spreads the response along its own axis, and the
    # probe's own 3.25 mm footprint sets the floor.
    sx = 0.0012 + 0.35*length
    sy = 0.0012 + 0.20*length
    # A flaw across the probe axis (yaw 90 deg) responds; one along it does
    # not. sin, so yaw 0 and yaw 180 both give nothing, which is right: a
    # notch is a line segment and those are the same flaw.
    amp = (FLAW_PEAK_OHMS
           * abs(math.sin(yaw))
           * (depth/max(DEPTH_MM)*1e3)
           * (length/(max(LENGTH_MM)*1e-3))
           * (width/(max(WIDTH_MM)*1e-3)))
    # Lift-off attenuates; tilt does not attenuate the flaw much, but it does
    # not leave it alone either.
    amp *= math.exp(-liftoff/0.0008)*(1.0 - 0.02*abs(math.degrees(tilt_par)))
    gx = xs[None, :]/sx
    gy = ys[:, None]/sy
    env = np.exp(-0.5*gx**2 - 0.5*gy**2)
    return (amp*gx*env*np.exp(1j*FLAW_PHASE_RAD)).astype(np.complex64)


def baseline_value(liftoff, tilt_par, tilt_perp):
    """The no-flaw reading at one pose, complex ohms.

    Odd in tilt_parallel because that tilt raises one D and lowers the other,
    so its sign is the sign of the asymmetry; even in tilt_perpendicular
    because that one moves both coils the same way and a differential probe
    cannot tell which way. That asymmetry is the single most important thing
    about a differential probe and a stub that got it wrong would let a sign
    error through the whole consumer.

    THE TWO TILTS POINT IN DIFFERENT DIRECTIONS IN THE IMPEDANCE PLANE, and
    that is the whole reason this file bothers with a baseline at all. The
    spec's central claim is that tilt about the split line does not merely add
    noise to the flaw channel but produces a signal OF THE SAME KIND -- so
    here it lies along the flaw's own direction, while the common-mode tilt
    and lift-off lie along the lift-off direction. Separating those two is the
    problem the real table exists to let us study, and a stub that put them
    all on one line would make the problem disappear.
    """
    tp = math.degrees(tilt_par)
    tq = math.degrees(tilt_perp)
    atten = math.exp(-liftoff/LIFTOFF_DECAY_M)
    common = (COMMON_OHMS + TILT_PERP_OHMS_PER_DEG2*tq*tq)*atten
    differential = TILT_PAR_OHMS_PER_DEG*tp*atten
    return (common*complex(math.cos(LIFTOFF_PHASE_RAD), math.sin(LIFTOFF_PHASE_RAD))
            + differential*complex(math.cos(FLAW_PHASE_RAD), math.sin(FLAW_PHASE_RAD)))


def provenance():
    return {
        "frequency_hz": 2.0e6,
        "material": "stub",
        "conductivity_s_per_m": 2.5e7,
        "permeability_relative": 1.0,
        "probe_type": "STUB -- not a probe model",
        "probe_shaft_diameter_m": 0.0047498,
        "coil_diameter_m": 0.0032,
        "coil_separation_m": 0.001526,
        "shielded": True,
        "generator": "make_ec_stub_table.py",
        "generator_version": "1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # A reader must be able to tell a stub from the real thing from the
        # FILE, not from its name -- a renamed copy is otherwise
        # indistinguishable, and that is exactly the failure the spec's
        # provenance section exists to prevent. ec_response_table.py refuses
        # to call a table with this flag set anything but a stub, and says so
        # on every load.
        "stub": True,
        "notes": ("STUB TABLE -- INVENTED PHYSICS, NOT A MEASUREMENT OF "
                  "ANYTHING. Written by ROS-Scanning tools/make_ec_stub_table.py "
                  "so the consumer side can be built and tested before the real "
                  "table exists. Schema, axis units, axis order and the meaning "
                  "of every axis match the real table: response = flaw-induced "
                  "CHANGE in impedance (ohms), no gain, no rotation, "
                  "probe-centred coordinates with x along the scan direction, "
                  "and THE LIFTOFF AXIS IS STANDOFF ABOVE EDGE CONTACT rather "
                  "than absolute height. The VALUES are closed-form "
                  "expressions chosen so a test can predict them."),
        # A notch is a line segment, so a flaw at yaw and at yaw+180 deg are
        # the same flaw and the response has period pi. Recorded here so the
        # reader can reduce a query itself instead of hard-coding the
        # periodicity of a file it did not write.
        "yaw_period_rad": math.pi,
    }


def write(out_dir):
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "al6061_2mhz_response.h5")
    base_path = os.path.join(out_dir, "al6061_2mhz_baseline.npz")
    ax = axes()
    shape = tuple(ax[n].size for n in PARAM_AXES) + (ax["y"].size, ax["x"].size)

    with h5py.File(table_path, "w") as f:
        for name in PARAM_AXES + ("y", "x"):
            ds = f.create_dataset("/axes/%s" % name, data=ax[name])
            ds.attrs["units"] = "rad" if name.startswith(("yaw", "tilt")) else "m"
        resp = f.create_dataset("/response", shape=shape, dtype=np.complex64,
                                chunks=(1,)*len(PARAM_AXES) + (ax["y"].size, ax["x"].size))
        f.create_dataset("/written", data=np.ones(shape[:len(PARAM_AXES)], dtype=bool))
        for k, v in provenance().items():
            f.attrs[k] = v
        f.attrs["_param_axis_order"] = np.array(PARAM_AXES, dtype=h5py.special_dtype(vlen=str))

        for idx in np.ndindex(*shape[:len(PARAM_AXES)]):
            vals = {n: float(ax[n][i]) for n, i in zip(PARAM_AXES, idx)}
            resp[idx] = flaw_field(vals["depth"], vals["length"], vals["width"],
                                   vals["yaw"], vals["liftoff"],
                                   vals["tilt_parallel"], vals["tilt_perpendicular"],
                                   ax["x"], ax["y"])

    base = np.zeros((ax["liftoff"].size, ax["tilt_parallel"].size,
                     ax["tilt_perpendicular"].size), dtype=np.complex64)
    for a, lo in enumerate(ax["liftoff"]):
        for b, tp in enumerate(ax["tilt_parallel"]):
            for c, tq in enumerate(ax["tilt_perpendicular"]):
                base[a, b, c] = baseline_value(float(lo), float(tp), float(tq))
    np.savez(base_path, baseline=base, liftoff=ax["liftoff"],
             tilt_parallel=ax["tilt_parallel"],
             tilt_perpendicular=ax["tilt_perpendicular"],
             provenance=json.dumps(provenance()))
    return table_path, base_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="ec_stub_table")
    args = ap.parse_args(argv)
    table, base = write(args.out_dir)
    print("STUB TABLE -- invented physics. Do not report any number from it.")
    print("  %s  (%.1f MB)" % (table, os.path.getsize(table)/1e6))
    print("  %s  (%.1f MB)" % (base, os.path.getsize(base)/1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
