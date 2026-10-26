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
# File:  ec_plate_model.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Turns a probe pose in the work cell into the eddy-current instrument's
#   two output voltages, using the response table and the simulated
#   test plate's flaw list.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-04:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""From a probe pose in the work cell to two volts on a DAQ card.

NO ROS, NO spatialnde2, NO ecsim. Pose in, volts out, so the whole chain is
testable on the host against a stub table -- which is the only way it could
have been written at all, the real table taking a night to generate.

THE FOUR STEPS, and each one is a place a sign error hides:

  1. THE PROBE FRAME, from the pose quaternion. The response table is indexed
     in probe coordinates with x along the coil-separation axis, so we need
     that axis in the cell, not just the probe's position.
  2. STANDOFF AND THE TWO TILTS, from the probe frame against the surface
     normal. Derived in closed form below and checked against ecsim's own
     probe_pose() to 6e-17 rad.
  3. THE IMPEDANCE: the flaw-free reading at that pose, plus the
     flaw-induced change for every notch close enough to couple.
  4. VOLTS: ohms through a declared gain and the instrument's own phase, plus
     the instrument's standing offset, clipped at the digitiser rails.

WHAT IS KNOWN, AND WHAT IS DECLARED. Two things about the real probe are not
measured, and both are constants here rather than being quietly assumed:

  * WHICH WAY THE COILS FACE. The probe's rotation about its own axis was
    never recorded -- it is a pencil probe in a holder -- so the angle from
    the probe_end frame's x axis to the coil-separation axis is unknown.
    COIL_AXIS_DEG. Getting it wrong rotates every flaw's apparent yaw by the
    same amount and swaps which tilt is the dangerous one, so it matters a
    great deal and it is the first thing to measure at the cell.
  * WHICH WAY probe_end's z POINTS. Read off the URDF
    (tari_ur_description/urdf/ec/tooling_macro.xacro: the ec_probe-probe_end
    joint carries rpy="pi 0 0", flipping z relative to the shaft, so
    probe_end +z runs back UP the shaft, away from the surface, which is also
    ecsim's convention). PROBE_AXIS_SIGN. This one is checkable at runtime --
    the probe axis must point away from the part when the probe is on the part
    -- so it is checked on every query and counted, rather than trusted.
"""
import json
import math
import os

import numpy as np

#: Angle from the probe_end frame's +x axis to the probe's coil-separation
#: axis, about the probe axis, degrees. UNKNOWN for the real probe; see the
#: module docstring. Zero means "the coils lie along probe_end x".
COIL_AXIS_DEG = 0.0

#: +1 if probe_end's +z points AWAY from the surface. See the module
#: docstring for where this is read from.
PROBE_AXIS_SIGN = +1

#: Instrument output scale, volts per ohm. The real instrument's gain is set
#: on its front panel and choosing it is one of the things this simulation
#: exists to study, so this is an argument and not a constant of nature. The
#: default puts a 10 degree tilt near the rail and the largest table flaw
#: around 80 mV -- about 20x a 4 mV noise floor -- which is roughly how an
#: operator sets gain on a calibration notch. suggest_v_per_ohm() below
#: computes it from a given table instead of from this guess.
V_PER_OHM = 0.1

#: The instrument's own phase reference, degrees. Rotates the complex
#: impedance before it is split into the two outputs. NOT the conditioning
#: rotation -- that is applied downstream, in ec_condition, and is what
#: set_rotation() chooses. This is the hardware's own arbitrary zero, and its
#: whole purpose here is to make sure the downstream rotation has a real
#: angle to find rather than starting out already aligned.
INSTRUMENT_PHASE_DEG = 18.0

#: The null point: what the instrument reads over good material before anyone
#: balances it, volts. Not (0, 0) on purpose -- an instrument is never nulled
#: when you find it, and null_here() exists to remove exactly this. Same value
#: the synthetic source used, so a config switching between them sees the same
#: offset to remove.
BASE_VOLTS = (1.20, -0.35)

#: Digitiser rails, volts. The Uniwest's analog outputs run +/-5 V into the
#: PXI card and a real pass HAS clipped against them: in the measured tilt
#: ladder the +6 degree pass sat at exactly -5.0000 for all 64 samples, with a
#: standard deviation of exactly zero, which made it look like the quietest
#: pass in the set. Reproducing the rail here is deliberate -- it is a real
#: failure mode, ec_liftoff.saturation() exists to catch it, and a simulator
#: with infinite headroom could not exercise that at all.
RAIL_VOLTS = 5.0

#: Gaussian noise on each channel, volts rms. Measured: 0.17-0.21 N on the
#: force channel is a different quantity, but the EC noise floor in the same
#: take was about 4 mV equivalent, so this is that.
NOISE_V = 0.004

#: How far from a flaw the probe can be before that flaw is skipped entirely,
#: metres. Set from the TABLE's own field extent at load time; this is only
#: the fallback if a table has no spatial axes to ask.
DEFAULT_REACH_M = 0.006


def load_plate(path):
    """The plate manifest written by tools/make_ec_test_plate.py."""
    path = os.path.abspath(os.path.expanduser(path))
    with open(path) as f:
        d = json.load(f)
    if "flaws" not in d or "plate" not in d:
        raise ValueError("%s is not a test-plate manifest (no 'flaws'/'plate')"
                         % (path,))
    d["_path"] = path
    return d


def quat_axes(quat):
    """(ex, ey, ez) of a frame in world coordinates, from (qx, qy, qz, qw).

    Written out rather than taken from scipy or tf_transformations: this
    module must import on the host with nothing but numpy, and the three
    columns of a rotation matrix are nine lines.
    """
    x, y, z, w = (float(v) for v in quat)
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x/n, y/n, z/n, w/n
    ex = np.array([1.0 - 2.0*(y*y + z*z), 2.0*(x*y + z*w), 2.0*(x*z - y*w)])
    ey = np.array([2.0*(x*y - z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + x*w)])
    ez = np.array([2.0*(x*z + y*w), 2.0*(y*z - x*w), 1.0 - 2.0*(x*x + y*y)])
    return ex, ey, ez


def suggest_v_per_ohm(table, target_volts=2.0, depth=None, length=None,
                      width=None):
    """Volts per ohm that puts the largest stored flaw at `target_volts`.

    How an operator actually sets gain: run over the calibration notch and
    turn the knob until the screen deflection is right. Reported at load so
    the number in use can be compared against the table in hand rather than
    against the default guessed for a different one.

    IT ASSUMES THE INSTRUMENT IS NULLED. The flaw response is the only thing
    it measures, and on an unbalanced instrument the standing reading is far
    larger -- at five degrees of parallel tilt the flaw-free reading is two
    orders of magnitude up on the flaw. Set this gain without nulling and
    every block rails. That is the real instrument's behaviour too, which is
    why the default V_PER_OHM is the conservative one and this is a
    suggestion rather than what the model starts with.
    """
    ax = table.axes
    depth = float(ax["depth"][-1]) if depth is None else depth
    length = float(ax["length"][-1]) if length is None else length
    width = float(ax["width"][-1]) if width is None else width
    block = table.pose_block(depth, length, width, math.pi/2.0)
    peak = float(np.abs(block).max())
    if peak <= 0.0:
        return float("nan")
    return target_volts/peak


class ECPlateModel(object):
    """The simulated instrument's reading, as a function of the probe's pose.

    `table` is an ec_response_table.ECResponseTable; `plate` is the manifest
    load_plate() returns. Everything else is the instrument's settings, which
    belong to the instrument and not to the physics.
    """

    def __init__(self, table, plate, coil_axis_deg=COIL_AXIS_DEG,
                 probe_axis_sign=PROBE_AXIS_SIGN, v_per_ohm=None,
                 instrument_phase_deg=INSTRUMENT_PHASE_DEG,
                 base_volts=BASE_VOLTS, rail_volts=RAIL_VOLTS,
                 reach_m=None, rng=None, pose_fn=None):
        self.table = table
        self.plate = plate
        self.pose_fn = pose_fn
        self.coil_axis_deg = float(coil_axis_deg)
        self.probe_axis_sign = 1 if probe_axis_sign >= 0 else -1
        self.instrument_phase_deg = float(instrument_phase_deg)
        self.base_volts = (float(base_volts[0]), float(base_volts[1]))
        self.rail_volts = float(rail_volts)
        self._rng = rng if rng is not None else np.random.default_rng()

        # The reach comes from the TABLE, not from a constant: the stored
        # field is the only thing that knows how far the response extends, and
        # a reach larger than the field would ask for points the table refuses
        # while a smaller one would clip a response the table does have.
        if reach_m is None:
            xs, ys = table.axes.get("x"), table.axes.get("y")
            reach_m = (DEFAULT_REACH_M if xs is None or ys is None
                       else float(min(abs(xs[0]), abs(xs[-1]),
                                      abs(ys[0]), abs(ys[-1]))))
        self.reach_m = float(reach_m)

        self.v_per_ohm = (float(v_per_ohm) if v_per_ohm is not None
                          else V_PER_OHM)
        self.suggested_v_per_ohm = suggest_v_per_ohm(table)

        self._build_flaws()

        self.queries = 0
        self.off_part = 0
        self.axis_sign_disagreements = 0
        self.clipped = 0

        #: The instrument's operating point, ohms, set by null(). Zero means
        #: an instrument nobody has balanced -- which is how one is always
        #: found, and the state the scan's first-touchdown null exists to
        #: leave behind.
        self.instrument_null_ohms = 0.0 + 0.0j
        self.nulls = 0
        self.last_null = None

    # -- the plate ---------------------------------------------------------

    def _build_flaws(self):
        """Flaw centres and long axes in WORLD coordinates.

        The manifest describes the plate in the PLATE's own frame plus where
        the plate was placed, and that is the right way round: a flaw table in
        world coordinates is a property of one part on one bench and would be
        wrong the moment the plate moved. The transform happens here, once, at
        load.
        """
        p = self.plate["plate"]
        place = [float(v) for v in p["place_xyz_m"]]
        yaw = math.radians(float(p.get("place_yaw_deg", 0.0)))
        c, s = math.cos(yaw), math.sin(yaw)
        self.place_xyz = np.array(place)
        self.place_yaw_rad = yaw
        # The scanned face is the plate's z=0 plane, so the top face lands at
        # the placement z. Flat and horizontal: the outward normal is +z.
        self.surface_z = place[2]
        self.surface_normal = np.array([0.0, 0.0, 1.0])
        size = [float(v) for v in p["size_m"]]
        self.half_x, self.half_y = 0.5*size[0], 0.5*size[1]

        self.flaws = []
        for d in self.plate["flaws"]:
            px, py = float(d["x"]), float(d["y"])
            wx = place[0] + c*px - s*py
            wy = place[1] + s*px + c*py
            # The notch's long axis. yaw_deg is measured in the plate frame
            # from the plate's own x axis, so the plate's placement yaw adds.
            a = math.radians(float(d["yaw_deg"])) + yaw
            self.flaws.append({
                "name": d.get("name", "?"),
                "centre": np.array([wx, wy, place[2]]),
                "axis": np.array([math.cos(a), math.sin(a), 0.0]),
                "depth_m": float(d["depth_m"]),
                "length_m": float(d["length_m"]),
                "width_m": float(d["width_m"]),
            })

    def over_plate(self, xyz):
        """Is the probe over the plate's footprint at all?

        Off the footprint there is no part under the probe, which is a
        different thing from being high above it: there is nothing to couple
        to and nothing to be tilted relative to. Between passes the arm lifts
        clear and runs off the end of the plate, and a model that answered
        those poses with a full-amplitude tilt signal would fill the trace
        store with readings that look like data.
        """
        d = np.asarray(xyz, dtype=np.float64)[:3] - self.place_xyz
        c, s = math.cos(-self.place_yaw_rad), math.sin(-self.place_yaw_rad)
        px = c*d[0] - s*d[1]
        py = s*d[0] + c*d[1]
        return abs(px) <= self.half_x and abs(py) <= self.half_y

    # -- the pose ----------------------------------------------------------

    def probe_frame(self, quat):
        """(ex, ey, ez) of the PROBE's own frame in world coordinates.

        ez is the probe axis pointing away from the surface and ex is the
        coil-separation axis, which is what the response table is indexed in.
        Right-handed in both settings of probe_axis_sign: flipping z flips y
        with it, so a wrong sign cannot silently turn the frame inside out on
        top of pointing the wrong way.
        """
        ex, ey, ez = quat_axes(quat)
        if self.probe_axis_sign < 0:
            ey, ez = -ey, -ez
        a = math.radians(self.coil_axis_deg)
        if a:
            ca, sa = math.cos(a), math.sin(a)
            ex, ey = ca*ex + sa*ey, -sa*ex + ca*ey
        return ex, ey, ez

    def pose_parameters(self, xyz, quat):
        """(standoff, tilt_parallel, tilt_perpendicular) at this pose.

        STANDOFF IS ABOVE EDGE CONTACT, which is what the table's lift-off
        axis means, and for this probe it is simply the gap between the
        commanded tip and the real surface. The probe is spring-loaded: when
        the arm drives the tip BELOW the surface the spring takes up the
        difference and the wear face rides on the surface, so standoff floors
        at zero -- which is why a 0.45 mm preload against a perfect mesh gives
        zero standoff rather than a negative one.

        THE TILT DECOMPOSITION. With the surface's outward normal written in
        probe coordinates as n = (nx, ny, nz), and ecsim's pose convention
        R = Rz(yaw) Rx(tilt_x) Ry(tilt_y) giving n = (-ca*sb, sa, ca*cb) for
        a = tilt_x and b = tilt_y:

            tilt_parallel      = atan2(-nx, nz)          (ecsim tilt_y)
            tilt_perpendicular = atan2(ny, hypot(nx, nz)) (ecsim tilt_x)

        Exact for any yaw and tilt -- checked against probe_pose() over 2000
        random poses, worst error 6e-17 rad. tilt_parallel is the flaw-like
        one, about the split line, raising one D and lowering the other.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        ex, ey, ez = self.probe_frame(quat)
        n = self.surface_normal

        # The axis check the module docstring promises. On the part, the probe
        # axis must point away from it; if it does not, PROBE_AXIS_SIGN is
        # wrong for this tooling and every tilt_parallel has the wrong sign --
        # which inverts every flaw signature without changing its size.
        if float(n @ ez) <= 0.0:
            self.axis_sign_disagreements += 1

        np_probe = np.array([float(n @ ex), float(n @ ey), float(n @ ez)])
        tilt_par = math.atan2(-np_probe[0], np_probe[2])
        tilt_perp = math.atan2(np_probe[1], math.hypot(np_probe[0], np_probe[2]))
        standoff = max(0.0, float(xyz[2]) - self.surface_z)
        return standoff, tilt_par, tilt_perp

    # -- the reading -------------------------------------------------------

    def impedance(self, xyz, quat, detail=False):
        """The instrument's complex reading at this pose, ohms.

        baseline(standoff, tilts) + sum over flaws of the flaw-induced change.
        The baseline is NOT a correction to the flaw response: with no flaw
        under the probe it is the entire signal, and once the probe is tilted
        it is much the larger of the two. That ordering -- tilt dominating the
        flaw -- is the problem this whole simulation exists to let us study,
        so it must not be quietly reversed by leaving the baseline out.

        Off the plate's footprint the answer is zero: see over_plate().
        """
        self.queries += 1
        xyz = np.asarray(xyz, dtype=np.float64)
        info = {"standoff_m": None, "tilt_parallel_deg": None,
                "tilt_perpendicular_deg": None, "flaws": [], "off_part": False}

        if not self.over_plate(xyz):
            self.off_part += 1
            info["off_part"] = True
            return (0.0 + 0.0j, info) if detail else 0.0 + 0.0j

        standoff, tilt_par, tilt_perp = self.pose_parameters(xyz, quat)
        info["standoff_m"] = standoff
        info["tilt_parallel_deg"] = math.degrees(tilt_par)
        info["tilt_perpendicular_deg"] = math.degrees(tilt_perp)

        total = self.table.baseline(standoff, tilt_par, tilt_perp)
        info["baseline_ohms"] = total

        ex, ey, _ = self.probe_frame(quat)
        for fl in self.flaws:
            d = xyz[:3] - fl["centre"]
            # Offset of the probe-face centre FROM THE FLAW CENTRE, in probe
            # coordinates. This sense, and not its negative: it is what
            # ecsim's scan_positions() tabulates, and reversing it mirrors
            # every flaw signature -- which, the signature being
            # double-lobed and antisymmetric, means inverting it.
            dx = float(d @ ex)
            dy = float(d @ ey)
            if abs(dx) > self.reach_m or abs(dy) > self.reach_m:
                continue
            yaw = math.atan2(float(fl["axis"] @ ey), float(fl["axis"] @ ex))
            z = self.table.response(
                fl["depth_m"], fl["length_m"], fl["width_m"], yaw,
                standoff=standoff, tilt_parallel=tilt_par,
                tilt_perpendicular=tilt_perp, dx=dx, dy=dy)
            total += z
            if detail:
                info["flaws"].append({"name": fl["name"], "dx_m": dx,
                                      "dy_m": dy, "yaw_deg": math.degrees(yaw),
                                      "ohms": z})
        return (total, info) if detail else total

    def null(self):
        """Balance the bridge: take the reading here as the new zero.

        NO ARGUMENTS, because this stands in for UniWest.null() and the scan
        reaches the hardware null through ec_first_contact_null(null_fn), which
        calls it with none. The pose comes from pose_fn -- the same holder the
        acquisition loop reads -- so the simulated instrument is nulled at
        whatever pose the arm is actually in when the scan nulls it, which is
        the first touchdown. Nulling in the air is therefore possible here,
        exactly as it is on the bench, and gives exactly the wrong answer.

        WHAT IT DOES AND DOES NOT REMOVE. It moves the instrument's operating
        point in OHMS, ahead of the gain stage: that is why a nulled
        instrument can be run at a gain that would otherwise rail. It does NOT
        remove the analog chain's own voltage offset, which is still there
        afterwards -- and removing that is the separate job of the software
        null, /ec/cond/null and null_here(). Both exist on the real
        instrument; neither replaces the other.
        """
        if self.pose_fn is None:
            raise RuntimeError(
                "the simulated instrument has no pose source, so it cannot "
                "tell where the probe is and nulling would be meaningless. "
                "Construct ECPlateModel with pose_fn=.")
        pose = self.pose_fn()
        if pose is None:
            raise RuntimeError(
                "the probe's pose is not known yet (TF has produced nothing), "
                "so there is nothing to null against. On the real instrument "
                "this is the case where the null lands in mid-air.")
        xyz, quat = pose[0], pose[1]
        z, info = self.impedance(xyz, quat, detail=True)
        self.instrument_null_ohms = z
        self.nulls += 1
        self.last_null = {"ohms": z, "xyz": tuple(float(v) for v in xyz[:3]),
                          "standoff_m": info.get("standoff_m"),
                          "off_part": info.get("off_part"),
                          "tilt_parallel_deg": info.get("tilt_parallel_deg")}
        return self.last_null

    def volts(self, xyz, quat, n=1, noise_v=NOISE_V, detail=False):
        """(X, Y) arrays of raw instrument volts, length n.

        RAW, as the DAQ card sees them: the instrument's standing offset is
        present and no conditioning has been applied. Removing the offset is
        what null_here() does and rotating what is left is what set_rotation()
        does, and leaving both to be done downstream is the point -- a
        simulator that handed over already-nulled, already-rotated numbers
        could not be used to exercise either.
        """
        z, info = self.impedance(xyz, quat, detail=True)
        phase = math.radians(self.instrument_phase_deg)
        # The operating point comes off BEFORE the gain, which is what makes
        # the bridge balance worth having: it is the difference between a
        # usable gain and one that rails on the standing reading.
        zz = ((z - self.instrument_null_ohms)
              * complex(math.cos(phase), math.sin(phase))*self.v_per_ohm)
        x = self.base_volts[0] + zz.real
        y = self.base_volts[1] + zz.imag
        xs = np.full(int(n), x, dtype=np.float64)
        ys = np.full(int(n), y, dtype=np.float64)
        if noise_v:
            xs += self._rng.standard_normal(int(n))*noise_v
            ys += self._rng.standard_normal(int(n))*noise_v
        rail = self.rail_volts
        if np.any(np.abs(xs) > rail) or np.any(np.abs(ys) > rail):
            self.clipped += 1
        np.clip(xs, -rail, rail, out=xs)
        np.clip(ys, -rail, rail, out=ys)
        info["volts"] = (x, y)
        info["impedance_ohms"] = z
        info["instrument_null_ohms"] = self.instrument_null_ohms
        return (xs, ys, info) if detail else (xs, ys)

    # -- reporting ---------------------------------------------------------

    def report(self):
        return {
            "plate": self.plate.get("_path"),
            "flaws": len(self.flaws),
            "surface_z_m": self.surface_z,
            "footprint_m": (2.0*self.half_x, 2.0*self.half_y),
            "reach_m": self.reach_m,
            "coil_axis_deg": self.coil_axis_deg,
            "probe_axis_sign": self.probe_axis_sign,
            "v_per_ohm": self.v_per_ohm,
            "suggested_v_per_ohm": self.suggested_v_per_ohm,
            "instrument_phase_deg": self.instrument_phase_deg,
            "base_volts": self.base_volts,
            "rail_volts": self.rail_volts,
            "instrument_null_ohms": self.instrument_null_ohms,
            "nulls": self.nulls,
            "last_null": self.last_null,
            "queries": self.queries,
            "off_part": self.off_part,
            "clipped": self.clipped,
            "axis_sign_disagreements": self.axis_sign_disagreements,
            "table": self.table.report(),
        }

    def describe(self):
        lines = [self.table.describe(),
                 "plate: %d notches, %.0fx%.0f mm, scanned face at z = %.4f m"
                 % (len(self.flaws), 2.0*self.half_x*1e3, 2.0*self.half_y*1e3,
                    self.surface_z),
                 "instrument: %.4f V/ohm (table suggests %.4f for 2 V on the "
                 "largest notch), phase %.1f deg, null at (%.2f, %.2f) V, "
                 "rails +/-%.1f V"
                 % (self.v_per_ohm, self.suggested_v_per_ohm,
                    self.instrument_phase_deg, self.base_volts[0],
                    self.base_volts[1], self.rail_volts),
                 "probe: coil axis %+.1f deg from probe_end x, axis sign %+d"
                 % (self.coil_axis_deg, self.probe_axis_sign)]
        return "\n".join(lines)
