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
# File:  ec_replay_image_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   replay_ec_scan() off the saved files against the live projection,
#   pixel for pixel, on a real projector.
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
"""The replayed image against the live one, pixel for pixel.

THE CLAIM BEING TESTED IS THE WHOLE POINT OF THE OFFLINE PATH: that a scan whose
process is gone can be turned back into the picture the operator was watching,
out of the per-pass files and the surface .x3d alone. ec_trace_store_test.py
pins the (pose, value) stream the projector would be handed, which is the
mechanism; this runs the REAL projector on both sides and compares the images it
produces.

Both projections are real spatialnde2 project_point_onto_parameterization over a
real loaded surface. The surface is a synthesised UV-parameterized rectangle --
write_plane_x3d(), the same two triangles a plane scan projects onto -- because
that is the one surface a host test can produce without Blender, a mesh or a
part. What it exercises is identical: the loader, the projection geometry, the
ray trace and the accumulation.

The image is compared in both halves of the fusion recording:

  totals        which parameterization cells received weight. Geometry only, so
                it must agree EXACTLY -- a replay that moved a pose by a
                micron, or dropped one, shows up here and nowhere else.
  accumulator   the weighted values. Compared to float32, because the live scan
                projects the compiled conditioning's float32 scalars while the
                replay recomputes in double from the raw impedance. That
                difference is real and it is one part in ten million; claiming
                exactness would be claiming something false.

    python3 -m pytest -q ec_replay_image_test.py
"""
import os
import sys

import numpy as np
import spatialnde2 as snde

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ec_trace_store as store

#: Not the identity, so a replay that ignored the saved parameters would show.
NULL = (0.22, -0.09)
ROTATION = 26.0
GAIN = (1.1, 1.9)
PIXELS = 64
RADIUS = 0.004
PITCH = 0.002

#: The patch, and the poses over it. In world coordinates, like every surface
#: this pipeline writes.
PATCH_ORIGIN = np.array([0.780, -0.320, 0.950])
PATCH_U = np.array([0.080, 0.0, 0.0])
PATCH_V = np.array([0.0, 0.060, 0.0])


def _namespace(printed):
    """surface_projection.dpi and ec_trace_store.dpi in one namespace.

    Executed the way a config executes them -- as function bodies sharing one
    globals() -- with the top-level `global` declarations dropped, which is what
    makes those names ordinary entries of this dict. `print` is collected so the
    replay's own output can be asserted on instead of scrolling past.
    """
    ns = {"print": lambda *a: printed.append(" ".join(str(x) for x in a))}
    exec("import hashlib, json, os, re, sys, time\n"
         "import numpy as np\nimport spatialnde2 as snde\n", ns)
    for name in ("surface_projection.dpi", "ec_trace_store.dpi"):
        src = open(os.path.join(HERE, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    ns["SCAN_MODULE_DIR"] = HERE
    return ns


def _recdb():
    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()
    return recdb


def _image(recdb, channel):
    """(accumulator, totals) of a projection channel, copied.

    Copied because the projector accumulates in MUTABLE storage -- that is what
    lets it work one point at a time -- so a view of the live image would be
    rewritten by the replay and the comparison would be against itself.
    """
    rec = recdb.latest.ref[channel].rec
    return (np.array(rec.reference_ndarray("accumulator").data, copy=True),
            np.array(rec.reference_ndarray("totals").data, copy=True))


class StandInAccumulator(object):
    """The three things ECPassRecorder asks a TraceAccumulator for."""

    traces_path = "/ec/traces"
    poses_path = "/ec/trace_poses"

    def __init__(self):
        self.staged = 0


def _synthetic_scan(k=36, passes=3):
    """Traces and poses for a serpentine over the patch. -> (z, poses)

    The probe is ON the surface and its tool axis is world +z, which is the
    eddy-current convention: the tool z points AWAY from the part, so the
    projector's own ray window (min_dist below the pose, max_dist above) is what
    finds the surface under it.

    Every trace carries a different impedance, so a replay that paired two
    traces up would be visible rather than lucky.
    """
    per = k//passes
    xs, ys = [], []
    for p in range(passes):
        y = PATCH_ORIGIN[1] + 0.015 + 0.012*p
        for j in range(per):
            x = PATCH_ORIGIN[0] + 0.012 + j*PITCH
            xs.append(x if p % 2 == 0 else
                      PATCH_ORIGIN[0] + 0.012 + (per - 1 - j)*PITCH)
            ys.append(y)
    n = len(xs)
    poses = np.zeros((n, 7), dtype=np.float32)
    poses[:, 0] = 1.0                                  # identity quaternion
    poses[:, 4] = xs
    poses[:, 5] = ys
    poses[:, 6] = PATCH_ORIGIN[2]
    rng = np.random.default_rng(20261002)
    z = (0.35 + 0.004*np.arange(n) + rng.normal(scale=0.05, size=n)
         + 1j*(-0.20 + 0.006*np.arange(n) + rng.normal(scale=0.05, size=n)))
    return z.astype(np.complex128), poses


def main(workdir):
    printed = []
    ns = _namespace(printed)
    recdb = _recdb()
    ns["recdb"] = recdb
    ns["_scan_work_dir"] = lambda: workdir

    x3d = os.path.join(workdir, "patch.x3d")
    ns["write_plane_x3d"](x3d, PATCH_ORIGIN, PATCH_U, PATCH_V)

    z, poses = _synthetic_scan()
    n = len(z)
    # float32, because that is what the compiled conditioning emits and what the
    # live projection is therefore handed. The agreement between this numpy
    # conditioning and the compiled one is ec_replay_conditioning_test.py.
    values = store.component_values(z, "vertical", null=NULL,
                                    rotation_deg=ROTATION,
                                    gain=GAIN).astype(np.float32)

    ok = True

    # ---- the channels a scan would have left behind -----------------------
    trans = recdb.start_transaction()
    tchan = trans.define_channel("/ec/traces", "main")
    tref = snde.create_ndarray_ref(trans, tchan, snde.SNDE_RTN_COMPLEXFLOAT64)
    tref.allocate_storage([n, 1], False)
    tref.data[:] = z.reshape(n, 1)
    tref.rec.mark_data_and_metadata_ready()
    pchan = trans.define_channel("/ec/trace_poses", "main")
    pref = snde.create_ndarray_ref(trans, pchan, snde.SNDE_RTN_FLOAT32)
    pref.allocate_storage([n, 7], False)
    pref.data[:] = poses
    pref.rec.mark_data_and_metadata_ready()
    cchan = trans.define_channel("/ec/cond/vertical", "main")
    cref = snde.create_ndarray_ref(trans, cchan, snde.SNDE_RTN_FLOAT32)
    cref.allocate_storage([n], False)
    cref.data[:] = values
    cref.rec.mark_data_and_metadata_ready()
    trans.end_transaction().globalrev_wait().wait_complete()

    # ---- the LIVE projection, trace by trace as the scan does it ----------
    live = ns["SurfaceProjector"](recdb, x3d, prefix="/live/proj",
                                  radius=RADIUS, horizontal_pixels=PIXELS,
                                  vertical_pixels=PIXELS)
    for k in range(n):
        live.project(poses[k], float(values[k]))
    live_status = live.projection_status(quiet=True)
    print("  live:   %d offered, %d of %d cell(s) have data"
          % (live_status["offered"], live_status["landed"], live_status["cells"]))
    if not live_status["landed"]:
        print("  FAIL: the live projection landed nothing, so there is nothing "
              "to compare a replay against. The patch and the poses disagree.")
        return 1
    live_accum, live_totals = _image(recdb, live.projection_channel)

    # ---- the pass files, written by the REAL recorder ---------------------
    run_dir = os.path.join(workdir, "ec_scan_2026-10-02_140511")
    accum = StandInAccumulator()

    def read_array(path):
        return np.asarray(recdb.latest.ref[path].data)

    recorder = ns["ECPassRecorder"](
        run_dir, read_array,
        lambda: {"accum": accum,
                 "cond_paths": {"vertical": "/ec/cond/vertical"},
                 "component": "vertical", "null": NULL,
                 "rotation_deg": ROTATION, "gain": GAIN, "x3d": x3d,
                 "pitch_m": PITCH, "projection_radius_m": RADIUS,
                 "projection_pixels": PIXELS},
        say=printed.append)
    per = n//3
    for p in range(3):
        accum.staged = min(n, (p + 1)*per)
        saved, why = recorder(p)
        if not saved:
            print("  FAIL: pass %d was not saved: %s" % (p, why)); ok = False
    print("  saved:  %d pass(es), %d trace(s) in %s"
          % (recorder.saved, recorder.traces_saved, os.path.basename(run_dir)))

    # ---- the REPLAY, from those files and the .x3d and nothing else -------
    out = ns["replay_ec_scan"](prefix="/ec/replay")
    print("  replay: %d projected, %d of %d cell(s) have data"
          % (out["projected"], out["status"]["landed"], out["status"]["cells"]))
    if out["projected"] != recorder.traces_saved:
        print("  FAIL: replayed %d of %d saved trace(s)"
              % (out["projected"], recorder.traces_saved)); ok = False
    if out["conditioning"]["overridden"]:
        print("  FAIL: nothing was overridden but the replay says %s"
              % (out["conditioning"]["overridden"],)); ok = False
    if not (out["check"] and out["check"]["ok"]):
        print("  FAIL: the replay's own conditioning check failed: %s"
              % (out["check"],)); ok = False

    replay_accum, replay_totals = _image(recdb, out["channel"])

    # WHICH CELLS GOT WEIGHT is pure geometry, so it must agree exactly. A
    # replay that moved a pose, or dropped or duplicated a trace, shows up here
    # and in nothing else -- the accumulated values would still look plausible.
    if not np.array_equal(replay_totals, live_totals):
        bad = int(np.count_nonzero(replay_totals != live_totals))
        print("  FAIL: %d of %d cell(s) received different weight -- the replay "
              "is not offering the same poses" % (bad, live_totals.size)); ok = False
    else:
        print("  totals identical over %d cell(s) (%d with data)"
              % (live_totals.size, int(np.count_nonzero(live_totals > 0))))

    finite = np.isfinite(live_accum) & np.isfinite(replay_accum)
    scale = max(float(np.max(np.abs(live_accum[finite]))), 1e-12)
    err = float(np.max(np.abs(replay_accum[finite] - live_accum[finite])))
    print("  accumulator max |replay - live| = %.3e (%.1e relative)"
          % (err, err/scale))
    if err > 1e-5*scale:
        print("  FAIL: the replayed image is not the live image"); ok = False
    if np.count_nonzero(finite) != finite.size:
        # Both sides must agree about WHERE there is no data, too.
        if not np.array_equal(np.isfinite(live_accum), np.isfinite(replay_accum)):
            print("  FAIL: the two images disagree about which cells are empty")
            ok = False

    # ---- and the override: a different rotation is a different image ------
    rotated = ns["replay_ec_scan"](rotation_deg=ROTATION + 40.0,
                                   prefix="/ec/replay")
    if rotated["conditioning"]["overridden"] != ["rotation_deg"]:
        print("  FAIL: the override was not recorded as one (%s)"
              % (rotated["conditioning"]["overridden"],)); ok = False
    if rotated["check"] is not None:
        print("  FAIL: an overridden replay must not claim to reproduce the "
              "scan's conditioning"); ok = False
    rot_accum, rot_totals = _image(recdb, rotated["channel"])
    # Same poses, so the same cells: only the VALUES may move.
    if not np.array_equal(rot_totals, live_totals):
        print("  FAIL: changing the rotation changed which cells got data"); ok = False
    same = np.allclose(rot_accum[finite], live_accum[finite],
                       atol=1e-5*scale, rtol=0.0)
    print("  rotated +40 deg: image changed = %s" % (not same,))
    if same:
        print("  FAIL: the rotation override did not change the image, so "
              "choosing the rotation offline would do nothing"); ok = False

    # The surface was reused rather than reloaded, or every replay strands
    # another model tree that can never be handed back.
    if not any("replay surface reused" in line for line in printed):
        print("  FAIL: the second replay rebuilt the surface instead of "
              "reusing it"); ok = False

    print("EC replay image: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def test_the_replayed_image_is_the_live_image(tmp_path):
    """ONE test, because it is one recording database and two real projectors.

    Loading a surface, building the projection geometry and projecting every
    trace twice is the expensive part, and every check here is a different
    property of that one pair of images: the cells that got weight, the values
    in them, that an override moves the values and only the values, and that a
    second replay reuses the loaded surface. None of them can be re-derived
    without doing all of it again.

    main() accumulates failures rather than stopping at the first, so a
    standalone run prints every false fact in one pass.
    """
    assert main(str(tmp_path)) == 0, (
        "the replayed image does not match the live one; the FAIL lines in the "
        "captured output above say which property broke")


if __name__ == "__main__":
    import tempfile
    sys.exit(main(tempfile.mkdtemp()))
