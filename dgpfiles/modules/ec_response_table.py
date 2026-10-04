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
# File:  ec_response_table.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-04
# Description:
#   Reads the eddy-current probe response table and answers probe-pose
#   queries fast enough to run inside the simulated instrument's
#   acquisition loop.
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
"""The response table, read for a simulator rather than for a study.

NO ROS, NO spatialnde2, NO ecsim -- just h5py and numpy, so this is testable on
the host and importable from the container. The file it reads is written by
tools/generate_ec_table.py running against ecsim on some other machine; the
schema is documented in docs/ec-probe-model-spec.md and in ecsim's
ecsim/table/schema.py, and tools/make_ec_stub_table.py writes a conformant
stub so everything here can be exercised before the real table exists.

WHY NOT JUST USE ecsim.table.ResponseTable. It is a perfectly good reader and
it is the authority on the format -- this module is checked against it. But it
is built for a study, and we need it inside an acquisition loop:

  * It re-reads and re-interpolates the WHOLE 2-D field on every call and then
    samples one point from it. The real table's field is 49x49 and the
    pose interpolation touches 2^7 = 128 of them, so one point query reads
    307 kB and does 128 field multiplies. The simulated instrument publishes
    50 blocks a second and sums over every flaw within reach, so that is
    hundreds of megabytes a second of HDF5 reads to produce two numbers.
    Here, the flaw-geometry axes are collapsed ONCE per flaw into a cached
    pose block, and a query is then 8 pose corners x 4 spatial corners of
    scalar arithmetic.
  * It raises DomainError out of domain. On an acquisition thread that is a
    silent flatline: the exception is caught by the publish loop, the block is
    lost, and the scan looks like an instrument that stopped. What a simulator
    needs instead is to CLAMP, keep publishing, and COUNT -- so the run still
    produces data and report() says how far outside the model it went.
  * It does not reduce yaw unless the file says to, and the real table does
    not say to (no yaw_period_rad attribute). A notch is a line SEGMENT: a
    flaw at yaw and at yaw+180 degrees are the same flaw, so the response has
    period pi, and a reader that refuses yaw=3.3 rad would refuse half the
    orientations on the plate. Reduced here, with the period taken from the
    file when it declares one.

THE LIFT-OFF AXIS IS NOT ABSOLUTE HEIGHT. It is STANDOFF ABOVE EDGE CONTACT.
A rigid probe face cannot be tilted while its centre stays down, so the
generator handed the solver edge_contact_liftoff(tilt) + the axis value, and
the axis is what the surface adds on top of that geometric minimum. Our probe
is spring-loaded and in contact, so it rides on its edge at any tilt and mesh
error adds standoff -- which means the axis value is exactly the quantity
ec_plate_model.py computes from the gap between the commanded tip and the real
surface. Call the parameter `standoff` here, never `liftoff`, so the two
cannot be confused at a call site.

WHAT THE VALUES MEAN, because every consumer has to agree: /response is the
flaw-induced CHANGE in impedance in ohms, relative to the flaw-free reading at
the same standoff and tilt; the baseline is the flaw-free reading itself (the
conductor-induced part of the impedance, so it tends to zero far from the
part). Neither has gain or rotation applied -- the instrument applies those in
hardware and choosing them is what we want to study. Probe-centred
coordinates, x along the scan direction.
"""
import json
import math
import os
import threading

import numpy as np

#: Schema names. Duplicated from ecsim's schema module rather than imported,
#: because ecsim is not installed in the container and will not be: this
#: reader must work with h5py alone. ec_response_table_test.py checks these
#: against ecsim's own constants whenever ecsim IS importable, so the
#: duplication cannot drift silently.
RESPONSE_PATH = "/response"
WRITTEN_PATH = "/written"
AXES_GROUP = "/axes"
BASELINE_PATH = "/baseline"
PARAM_AXIS_ORDER_ATTR = "_param_axis_order"
YAW_PERIOD_ATTR = "yaw_period_rad"
UNITS_ATTR = "units"

FLAW_AXES = ("depth", "length", "width", "yaw")
POSE_AXES = ("liftoff", "tilt_parallel", "tilt_perpendicular")
SPATIAL_AXES = ("y", "x")

AXIS_UNITS = {"depth": "m", "length": "m", "width": "m", "yaw": "rad",
              "liftoff": "m", "tilt_parallel": "rad",
              "tilt_perpendicular": "rad", "y": "m", "x": "m"}

#: A notch is a line segment, so yaw and yaw+pi are the same flaw. Used when
#: the file declares no period of its own.
DEFAULT_YAW_PERIOD_RAD = math.pi

#: How far past the largest standoff node the response is extrapolated before
#: it is simply called zero. Beyond a few decay lengths the probe is off the
#: part and the reading is air; the point of extrapolating at all is that a
#: hard clamp would hold a full-amplitude tilt signal while the probe flies
#: between passes, which looks like data and is not.
STANDOFF_EXTRAPOLATION_LIMIT = 8.0

#: Fallback decay length for that extrapolation if the table's own top two
#: lift-off nodes cannot supply one (a single-node axis, or equal magnitudes).
#: 0.2 mm is the scale the real generator's own notes quote for how fast this
#: probe's coil map changes with lift-off; it is only ever used off the part.
FALLBACK_STANDOFF_DECAY_M = 0.0002


class TableDomain(object):
    """How far outside the table a run has strayed, counted rather than raised.

    One counter per axis plus the two spatial ones. `clamped` is a pose the
    table cannot represent and that was answered with its nearest edge;
    `outside_field` is a probe too far from a flaw for the stored field to
    cover, which is answered with zero and is NORMAL -- most flaws are far
    from the probe most of the time, and the field's edge is deliberately
    clean baseline.
    """

    def __init__(self):
        self.clamped = {}
        self.outside_field = 0
        self.extrapolated = 0
        self.beyond_extrapolation = 0

    def clamp(self, axis, value, lo, hi):
        if value < lo:
            self.clamped[axis] = self.clamped.get(axis, 0) + 1
            return lo
        if value > hi:
            self.clamped[axis] = self.clamped.get(axis, 0) + 1
            return hi
        return value

    def report(self):
        return {"clamped": dict(self.clamped),
                "outside_field": self.outside_field,
                "extrapolated": self.extrapolated,
                "beyond_extrapolation": self.beyond_extrapolation}


def _corners(nodes, value):
    """((i0, w0), (i1, w1)) bracketing `value` in ascending `nodes`.

    `value` must already be inside [nodes[0], nodes[-1]]; the caller clamps.
    A single-node axis returns one corner, which is how the width axis of the
    real table (one node, 0.15 mm) is handled without a special case.
    """
    n = len(nodes)
    if n == 1:
        return ((0, 1.0),)
    i = int(np.searchsorted(nodes, value))
    if i <= 0:
        return ((0, 1.0),)
    if i >= n:
        return ((n - 1, 1.0),)
    lo, hi = nodes[i - 1], nodes[i]
    if hi == lo:
        return ((i, 1.0),)
    f = float((value - lo)/(hi - lo))
    if f <= 0.0:
        return ((i - 1, 1.0),)
    if f >= 1.0:
        return ((i, 1.0),)
    return ((i - 1, 1.0 - f), (i, f))


class ECResponseTable(object):
    """The response table, with a per-flaw pose block cache.

    Open it once and keep it: the HDF5 handle is held for the life of the
    object and reads are serialised under a lock, because the simulated
    instrument's publish thread and whatever is at the prompt both reach it.
    h5py is not thread-safe for concurrent reads on one file.
    """

    def __init__(self, path, baseline_path=None, max_cache_mb=256.0):
        import h5py

        self.path = os.path.abspath(os.path.expanduser(path))
        self._lock = threading.Lock()
        self._file = h5py.File(self.path, "r")
        self.domain = TableDomain()
        self.max_cache_bytes = float(max_cache_mb)*1e6
        self._cache = {}
        self._cache_bytes = 0
        self.queries = 0

        order = self._file.attrs.get(PARAM_AXIS_ORDER_ATTR)
        if order is None:
            raise ValueError(
                "%s has no %s attribute, so the order of its parameter axes is "
                "unknown. Every table written by ecsim's ResponseTableWriter "
                "carries it; a file without it was not written by the "
                "generator and cannot be indexed safely."
                % (self.path, PARAM_AXIS_ORDER_ATTR))
        self.param_axes = tuple(
            v.decode() if isinstance(v, bytes) else str(v) for v in order)

        missing = [a for a in FLAW_AXES if a not in self.param_axes]
        if missing:
            raise ValueError("%s is missing the flaw axes %s"
                             % (self.path, missing))

        self.axes = {}
        self.units = {}
        for name in self.param_axes + SPATIAL_AXES:
            ds = self._file["%s/%s" % (AXES_GROUP, name)]
            self.axes[name] = np.asarray(ds[...], dtype=np.float64)
            u = ds.attrs.get(UNITS_ATTR)
            self.units[name] = u.decode() if isinstance(u, bytes) else str(u)
            want = AXIS_UNITS.get(name)
            if want is not None and self.units[name] != want:
                # Refuse rather than convert. The spec called out a silent
                # 25.4x as the kind of mistake that survives review, and a
                # reader that quietly accepted mils would be exactly that.
                raise ValueError(
                    "%s: axis %r is in %r but this pipeline is %r throughout. "
                    "Refusing rather than guessing a conversion."
                    % (self.path, name, self.units[name], want))

        # Dense lift-off/tilt is what the real generator writes, because ecsim
        # measured that they do not factor out of the flaw response. The
        # factorised form the spec invited would need /liftoff_tilt/G folded
        # in here; it is refused rather than ignored, because ignoring it
        # would silently drop the entire pose dependence.
        if "/liftoff_tilt/G" in self._file:
            raise ValueError(
                "%s stores a FACTORISED lift-off/tilt correction "
                "(/liftoff_tilt/G). This reader only handles the dense form "
                "that tools/generate_ec_table.py writes. Reading it as dense "
                "would drop the pose dependence entirely and the scan would "
                "look clean." % (self.path,))

        self.pose_axes = tuple(a for a in POSE_AXES if a in self.param_axes)
        self._resp = self._file[RESPONSE_PATH]
        self.yaw_period = self._file.attrs.get(YAW_PERIOD_ATTR)
        self.yaw_period = (DEFAULT_YAW_PERIOD_RAD if self.yaw_period is None
                           else float(self.yaw_period))
        self.yaw_period_declared = YAW_PERIOD_ATTR in self._file.attrs

        self._load_baseline(baseline_path)
        self._standoff_decay = self._fit_standoff_decay()

    # -- provenance --------------------------------------------------------

    @property
    def attrs(self):
        """Root attributes, without the private ones."""
        out = {}
        for k, v in self._file.attrs.items():
            if k.startswith("_"):
                continue
            out[k] = v.decode() if isinstance(v, bytes) else v
        return out

    @property
    def is_stub(self):
        """True for a table from make_ec_stub_table.py.

        Read from the FILE, not from its name: a renamed stub is otherwise
        indistinguishable from the real thing, which is precisely what the
        spec's provenance section exists to prevent. Everything that prints a
        result from this table has to say so when this is True.
        """
        return bool(self._file.attrs.get("stub", False))

    @property
    def shape(self):
        return tuple(self._resp.shape)

    def describe(self):
        a = self.attrs
        return ("%s%s\n  %s at %.3f MHz, probe: %s\n  generated %s by %s %s\n"
                "  axes: %s\n  field %dx%d at %.3f mm over +/-%.1f mm"
                % (os.path.basename(self.path),
                   "   ** STUB TABLE: INVENTED PHYSICS **" if self.is_stub else "",
                   a.get("material", "?"),
                   float(a.get("frequency_hz", 0.0))/1e6, a.get("probe_type", "?"),
                   a.get("created_utc", "?"), a.get("generator", "?"),
                   a.get("generator_version", "?"),
                   ", ".join("%s=%d" % (n, self.axes[n].size)
                             for n in self.param_axes),
                   self.axes["y"].size, self.axes["x"].size,
                   (self.axes["x"][1] - self.axes["x"][0])*1e3
                   if self.axes["x"].size > 1 else 0.0,
                   self.axes["x"][-1]*1e3))

    # -- the flaw-free trajectory ------------------------------------------

    def _load_baseline(self, baseline_path):
        """The no-flaw lift-off/tilt reading, from /baseline or the sidecar.

        TWO PLACES because the schema only allows /baseline alongside the
        factorised model, and the real table is dense -- so the generator
        writes the trajectory to al6061_2mhz_baseline.npz beside the table
        instead. Both are read, the in-file one winning, so this works with
        either. WITHOUT IT THE SIMULATOR IS WRONG IN THE MOST IMPORTANT WAY:
        with no flaw under the probe the baseline is the ENTIRE signal, and it
        is the trajectory the cell's conditioning procedure rotates onto an
        axis. A missing baseline is reported, not defaulted to zero.
        """
        self.baseline_source = None
        self.baseline_provenance = None
        self._baseline = None
        self._baseline_axes = None

        if BASELINE_PATH in self._file:
            self._baseline = np.asarray(self._file[BASELINE_PATH][...])
            self._baseline_axes = tuple(self.axes[a] for a in POSE_AXES)
            self.baseline_source = self.path + BASELINE_PATH
            return

        if baseline_path is None:
            guess = os.path.join(os.path.dirname(self.path),
                                 os.path.basename(self.path)
                                 .replace("_response.h5", "_baseline.npz"))
            baseline_path = guess if os.path.isfile(guess) else None
        if baseline_path is None:
            return

        baseline_path = os.path.abspath(os.path.expanduser(baseline_path))
        with np.load(baseline_path, allow_pickle=False) as z:
            self._baseline = np.asarray(z["baseline"])
            self._baseline_axes = (np.asarray(z["liftoff"], dtype=np.float64),
                                   np.asarray(z["tilt_parallel"], dtype=np.float64),
                                   np.asarray(z["tilt_perpendicular"], dtype=np.float64))
            try:
                self.baseline_provenance = json.loads(str(z["provenance"]))
            except Exception:                              # noqa: BLE001
                self.baseline_provenance = None
        want = tuple(a.size for a in self._baseline_axes)
        if self._baseline.shape != want:
            raise ValueError("%s: baseline is %s but its axes are %s"
                             % (baseline_path, self._baseline.shape, want))
        self.baseline_source = baseline_path

    @property
    def has_baseline(self):
        return self._baseline is not None

    def _fit_standoff_decay(self):
        """A decay length for standoff beyond the table, from the table itself.

        Taken from the baseline's top two lift-off nodes at zero tilt, so it
        is the file's own measured rate rather than a number chosen here. Used
        only off the part -- see STANDOFF_EXTRAPOLATION_LIMIT.
        """
        if self._baseline is None or self._baseline_axes is None:
            return FALLBACK_STANDOFF_DECAY_M
        lo = self._baseline_axes[0]
        if lo.size < 2:
            return FALLBACK_STANDOFF_DECAY_M
        # LARGEST |tilt_parallel|, not zero tilt. A differential probe reads
        # almost nothing over good material with the probe square on -- the
        # two coils cancel -- so at zero tilt this would be the ratio of two
        # near-zeros and the fit would be noise. The parallel tilt is where
        # the flaw-free signal actually lives.
        tp = int(np.argmax(np.abs(self._baseline_axes[1])))
        tq = int(np.argmin(np.abs(self._baseline_axes[2])))
        a = abs(complex(self._baseline[-2, tp, tq]))
        b = abs(complex(self._baseline[-1, tp, tq]))
        d = float(lo[-1] - lo[-2])
        if not (a > 0.0 and b > 0.0 and b < a and d > 0.0):
            return FALLBACK_STANDOFF_DECAY_M
        return d/math.log(a/b)

    @property
    def standoff_decay_m(self):
        return self._standoff_decay

    # -- pose handling -----------------------------------------------------

    def _standoff_factor(self, standoff):
        """(clamped standoff, extra attenuation) for a standoff off the table.

        Inside the table: the value and 1.0. Above the top node: the top node
        and an exponential taper on the table's own decay length, which goes
        to the right limit (the conductor-induced part of the impedance tends
        to zero as the probe leaves the part) instead of holding a
        full-amplitude tilt signal in mid-air. Far enough out, zero.
        """
        lo = self.axes.get("liftoff")
        if lo is None or lo.size == 0:
            return standoff, 1.0
        top = float(lo[-1])
        if standoff <= top:
            return self.domain.clamp("liftoff", standoff, float(lo[0]), top), 1.0
        n = (standoff - top)/max(self._standoff_decay, 1e-9)
        if n > STANDOFF_EXTRAPOLATION_LIMIT:
            self.domain.beyond_extrapolation += 1
            return top, 0.0
        self.domain.extrapolated += 1
        return top, math.exp(-n)

    def reduce_yaw(self, yaw):
        """Yaw reduced into the stored range. A notch is a line segment."""
        y = float(np.mod(yaw, self.yaw_period))
        nodes = self.axes["yaw"]
        return self.domain.clamp("yaw", y, float(nodes[0]), float(nodes[-1]))

    # -- the cached pose block ---------------------------------------------

    def pose_block(self, depth, length, width, yaw):
        """The (standoff, tilt_par, tilt_perp, y, x) block for one flaw.

        The flaw-geometry axes are interpolated away ONCE and the result kept,
        because a flaw's geometry does not change during a scan while its
        pose-relative position changes every block. For the real table this is
        2.9 MB per flaw and turns a point query from 128 field reads into 32
        scalar lookups.

        yaw IS a flaw-geometry axis here and not a pose axis, even though it
        depends on the probe's heading, because the probe's heading is fixed
        within a pass: the roll is locked during an eddy-current scan and the
        passes are straight. A scan that rotated the probe continuously would
        want yaw moved into the block instead, and would then want the whole
        (yaw, standoff, tilt, tilt) block cached -- 7x bigger, still small.
        """
        yaw = self.reduce_yaw(yaw)
        key = (round(float(depth), 9), round(float(length), 9),
               round(float(width), 9), round(float(yaw), 7))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        want = {"depth": depth, "length": length, "width": width, "yaw": yaw}
        per_axis = []
        for name in self.param_axes:
            if name not in FLAW_AXES:
                per_axis.append(None)
                continue
            nodes = self.axes[name]
            v = self.domain.clamp(name, float(want[name]),
                                  float(nodes[0]), float(nodes[-1]))
            per_axis.append(_corners(nodes, v))

        with self._lock:
            block = None
            for combo in _product(per_axis):
                idx = tuple(c[0] if c is not None else slice(None) for c in combo)
                w = 1.0
                for c in combo:
                    if c is not None:
                        w *= c[1]
                if w == 0.0:
                    continue
                part = np.asarray(self._resp[idx], dtype=np.complex128)*w
                block = part if block is None else block + part

        nbytes = block.nbytes
        if self._cache_bytes + nbytes > self.max_cache_bytes:
            # Oldest first. A scan visits every flaw repeatedly, so any
            # eviction policy is a compromise; insertion order at least keeps
            # the ones the current pass is using.
            for k in list(self._cache):
                if self._cache_bytes + nbytes <= self.max_cache_bytes:
                    break
                self._cache_bytes -= self._cache.pop(k).nbytes
        self._cache[key] = block
        self._cache_bytes += nbytes
        return block

    # -- queries -----------------------------------------------------------

    def response(self, depth, length, width, yaw, standoff=0.0,
                 tilt_parallel=0.0, tilt_perpendicular=0.0, dx=0.0, dy=0.0):
        """Flaw-induced change in impedance, ohms, at one probe position.

        `dx`, `dy` are the offset of the PROBE FACE CENTRE from the FLAW
        CENTRE, expressed in the probe frame -- x along the scan direction.
        That is the sense ecsim's scan_positions() tabulates and the sense the
        stored field is indexed by; getting it backwards mirrors every flaw
        signature and nothing downstream would notice.

        Outside the stored field this returns 0 and counts it: the field is
        +/-6 mm with deliberately clean baseline at its edge, so a flaw
        further away than that genuinely contributes nothing, and most flaws
        are further away than that most of the time.
        """
        self.queries += 1
        xs, ys = self.axes["x"], self.axes["y"]
        if not (xs[0] <= dx <= xs[-1] and ys[0] <= dy <= ys[-1]):
            self.domain.outside_field += 1
            return 0.0 + 0.0j

        standoff, taper = self._standoff_factor(float(standoff))
        if taper == 0.0:
            return 0.0 + 0.0j

        block = self.pose_block(depth, length, width, yaw)
        pose = self._pose_corners(standoff, tilt_parallel, tilt_perpendicular)
        sx = _corners(xs, float(dx))
        sy = _corners(ys, float(dy))

        total = 0.0 + 0.0j
        for combo in _product(pose):
            idx = tuple(c[0] for c in combo)
            w = 1.0
            for c in combo:
                w *= c[1]
            if w == 0.0:
                continue
            plane = block[idx]
            for yi, yw in sy:
                for xi, xw in sx:
                    total += complex(plane[yi, xi])*w*yw*xw
        return total*taper

    def _pose_corners(self, standoff, tilt_parallel, tilt_perpendicular):
        want = {"liftoff": standoff, "tilt_parallel": tilt_parallel,
                "tilt_perpendicular": tilt_perpendicular}
        out = []
        for name in self.pose_axes:
            nodes = self.axes[name]
            v = float(want[name])
            if name != "liftoff":
                v = self.domain.clamp(name, v, float(nodes[0]), float(nodes[-1]))
            out.append(_corners(nodes, v))
        return out

    def baseline(self, standoff=0.0, tilt_parallel=0.0, tilt_perpendicular=0.0):
        """The flaw-free reading at one pose, ohms.

        Raises if there is no baseline. Returning zero instead would be a
        simulator that reads nothing at all over good material -- which is
        exactly what a differential probe does NOT do once it is tilted, and
        the tilt signal is the thing we are trying to study.
        """
        if self._baseline is None:
            raise RuntimeError(
                "no flaw-free trajectory for %s. The generator writes it to "
                "al6061_2mhz_baseline.npz beside the table; without it the "
                "simulator has no signal at all over good material, which is "
                "the opposite of what a tilted differential probe does."
                % (self.path,))
        standoff, taper = self._standoff_factor(float(standoff))
        if taper == 0.0:
            return 0.0 + 0.0j
        lo, tp, tq = self._baseline_axes
        vals = (standoff,
                self.domain.clamp("tilt_parallel", float(tilt_parallel),
                                  float(tp[0]), float(tp[-1])),
                self.domain.clamp("tilt_perpendicular", float(tilt_perpendicular),
                                  float(tq[0]), float(tq[-1])))
        corners = [_corners(ax, v) for ax, v in zip((lo, tp, tq), vals)]
        total = 0.0 + 0.0j
        for combo in _product(corners):
            w = 1.0
            for c in combo:
                w *= c[1]
            if w == 0.0:
                continue
            total += complex(self._baseline[tuple(c[0] for c in combo)])*w
        return total*taper

    # -- housekeeping ------------------------------------------------------

    def report(self):
        return {"path": self.path, "stub": self.is_stub,
                "shape": self.shape,
                "queries": self.queries,
                "cached_flaws": len(self._cache),
                "cache_mb": self._cache_bytes/1e6,
                "baseline": self.baseline_source,
                "yaw_period_rad": self.yaw_period,
                "yaw_period_declared": self.yaw_period_declared,
                "standoff_decay_m": self._standoff_decay,
                "domain": self.domain.report()}

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._cache = {}
        self._cache_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _product(per_axis):
    """Cartesian product over per-axis corner lists, passing None through.

    itertools.product would do it, but None has to survive as a whole-axis
    slice rather than being iterated, and spelling that out is shorter than
    the wrapper it would otherwise need.
    """
    out = [()]
    for corners in per_axis:
        if corners is None:
            out = [row + (None,) for row in out]
            continue
        out = [row + (c,) for row in out for c in corners]
    return out
