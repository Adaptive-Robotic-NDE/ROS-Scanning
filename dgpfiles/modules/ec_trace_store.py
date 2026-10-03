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
# File:  ec_trace_store.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   On-disk format for eddy-current traces, written one file per
#   completed pass, plus the impedance-plane conditioning in numpy so a
#   saved run can be re-conditioned and re-projected offline.
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
"""An eddy-current scan's traces on disk, one file per completed pass.

WHY THIS EXISTS. An EC scan wrote nothing to disk at any point. The traces and
the pose of each one lived only in the in-process recording database --
TraceAccumulator on /ec/traces and /ec/trace_poses -- so the process WAS the
dataset. On 2026-10-02 a scan ran 59 of 64 passes over 28 minutes, with live
projection working and real indications visible, then wedged; the process was
killed and all 59 passes were gone. Only the meshes survived, and the honest
answer to "reproject it offline" was that there was nothing to reproject.

PER PASS, NOT PER SCAN. End of scan is precisely the case that does not happen:
the scan that needs this is the one that does not reach its own last line. A
pass is the natural unit -- it is the point at which the arm is off the part,
the stager has drained, and the trace list is quiescent for a moment.

WHAT A PASS FILE HAS TO CARRY. Enough that a reprojection needs these files and
the surface .x3d and nothing else:

    pose          per trace, (qw,qx,qy,qz,x,y,z) -- the projector's input, and
                  the only record of WHERE a trace was taken
    impedance     the RAW complex reading, before conditioning. This is the
                  half that makes offline work worth doing: the conditioning
                  rotation is chosen by eye during setup and the operator's note
                  from the lost run was that they had rotated the scan-line
                  noise horizontal and the notches were showing up vertically.
                  Choosing that angle again, offline, over data already taken,
                  is the whole point -- and it is impossible from a conditioned
                  scalar alone.
    conditioned   what the LIVE scan projected, so a replay can be checked
                  against it rather than merely believed.
    the parameters in force, the surface path, the pass index, the trace index
    range in the trace list, and the trigger report.

THE CONDITIONING ARITHMETIC IS HERE AS WELL, in numpy, mirroring
spatialnde2-ec-functions' ec_condition_cpp.hpp operation for operation -- null,
then rotate by MINUS the quoted angle, then per-axis gain. It is duplicated
deliberately: the compiled function is a math function over recording channels
and cannot be asked to condition a numpy array, and a replay that could only use
the parameters the scan happened to run with would not be a replay worth having.
condition_matches_stored() is what keeps the duplicate honest -- it re-derives
the stored `conditioned` from the stored `impedance` and reports the deviation,
which is a direct comparison against the C++ on real data.

Pure numpy. No spatialnde2, no ROS, no recording database: the wiring that reads
the database and drives the projector is ec_trace_store.dpi.
"""
import glob
import json
import os
import re

import numpy as np

#: One file per pass, numbered by the pass's index in the scan's segment list.
#:
#: Zero-padded so a directory listing sorts the way the scan ran, and the index
#: is in the NAME as well as in the metadata so a file that has been copied
#: somewhere else still says which pass it is.
PASS_GLOB = "pass*.npz"
_PASS_RE = re.compile(r"^pass(\d+)\.npz$")

#: Which conditioned scalar a value is. The same three the compiled function
#: emits; "vertical" is the usual scan display once the rotation is set.
COMPONENTS = ("horizontal", "vertical", "magnitude")

#: Prefix of a run directory, so latest_run() can recognise one.
RUN_PREFIX = "ec_scan_"


def pass_filename(index):
    """The file one pass is written to. -> "pass007.npz" """
    return "pass%03d.npz" % (int(index),)


def pass_index_of(path):
    """The pass index a file name carries, or None if it is not a pass file."""
    m = _PASS_RE.match(os.path.basename(path))
    return None if m is None else int(m.group(1))


# ==============================================================================
# THE CONDITIONING, IN NUMPY
# ==============================================================================

def condition_traces(impedance, null=(0.0, 0.0), rotation_deg=0.0,
                     gain=(1.0, 1.0)):
    """Null, rotate and scale raw impedance. -> {horizontal, vertical, magnitude,
    conditioned}

    The same three operations, in the same order, as
    spatialnde2-ec-functions/ec_condition_cpp.hpp:

        null      subtract the reading taken on good material, so what is left
                  is the departure from good material.
        rotate    by MINUS the quoted angle. "lift-off at 40 degrees" means turn
                  the data by -40 to lay lift-off along the horizontal axis,
                  which is how the angle is quoted on an instrument.
        gain      the two axes separately: after the rotation they carry
                  different quantities.

    The order is not interchangeable. Rotating before nulling turns the null
    point itself into a moving offset, which is the classic way to get a display
    that drifts as the rotation is adjusted.

    NOT APPLIED HERE: the stored amplitude scale and offset the compiled
    function reads off the trace list (get_ampl_scale_offset). An EC trace is
    staged as complex128 volts with no scale metadata, so those are 1.0 and 0.0
    for every scan this pipeline runs -- but a digitizer that reported counts
    would need them, and condition_matches_stored() is what would notice.
    """
    z = np.asarray(impedance, dtype=np.complex128).reshape(-1)
    null = tuple(float(v) for v in (tuple(null) + (0.0, 0.0))[:2])
    gain = tuple(float(v) for v in (tuple(gain) + (1.0, 1.0))[:2])

    re_ = z.real - null[0]
    im_ = z.imag - null[1]

    theta = -float(rotation_deg)*np.pi/180.0
    ct, st = np.cos(theta), np.sin(theta)

    h = (re_*ct - im_*st)*gain[0]
    v = (re_*st + im_*ct)*gain[1]
    return {"horizontal": h, "vertical": v,
            "magnitude": np.sqrt(h*h + v*v),
            "conditioned": h + 1j*v}


def component_values(impedance, component="vertical", null=(0.0, 0.0),
                     rotation_deg=0.0, gain=(1.0, 1.0)):
    """One conditioned scalar per trace -- the thing that gets projected."""
    component = str(component)
    if component not in COMPONENTS:
        raise ValueError("component must be one of %s, not %r"
                         % (", ".join(COMPONENTS), component))
    out = condition_traces(impedance, null=null, rotation_deg=rotation_deg,
                           gain=gain)
    return np.asarray(out[component], dtype=np.float64)


def condition_matches_stored(run, atol=None):
    """Does the numpy conditioning reproduce what the compiled one stored? -> dict

    The replay's own self-check, and the only thing standing between this file's
    arithmetic and the C++ it claims to mirror. Re-derives the stored
    `conditioned` column from the stored `impedance` with the stored parameters
    and reports the worst disagreement.

    The tolerance defaults to the float32 the compiled function stores its
    scalars in, scaled by the size of the values -- not an absolute epsilon,
    because an EC reading is volts of whatever size the instrument's gain made
    it. Returns {"checked", "max_abs_error", "tolerance", "ok", "why"}; a run
    with no stored conditioned column answers checked=0 and ok=True, because
    there is nothing to disagree with.
    """
    stored = run.get("conditioned")
    z = run.get("impedance")
    out = {"checked": 0, "max_abs_error": None, "tolerance": None,
           "ok": True, "why": "no stored conditioned column to compare against"}
    if stored is None or z is None:
        return out
    stored = np.asarray(stored, dtype=np.float64).reshape(-1)
    finite = np.isfinite(stored)
    if not finite.any():
        out["why"] = "every stored conditioned value is NaN or inf"
        return out
    mine = component_values(z, run["component"], null=run["null"],
                            rotation_deg=run["rotation_deg"], gain=run["gain"])
    n = min(len(stored), len(mine))
    finite = finite[:n] & np.isfinite(mine[:n])
    err = np.abs(mine[:n][finite] - stored[:n][finite])
    scale = float(np.max(np.abs(stored[:n][finite]))) if finite.any() else 0.0
    tol = (1e-5*max(scale, 1e-9)) if atol is None else float(atol)
    out["checked"] = int(finite.sum())
    out["max_abs_error"] = float(err.max()) if err.size else 0.0
    out["tolerance"] = tol
    out["ok"] = bool(out["max_abs_error"] <= tol)
    out["why"] = ("the replay's conditioning reproduces the scan's to %.3g "
                  "(tolerance %.3g over %d trace(s))"
                  % (out["max_abs_error"], tol, out["checked"])) if out["ok"] else (
        "THE REPLAY'S CONDITIONING DISAGREES WITH THE SCAN'S by up to %.3g "
        "(tolerance %.3g over %d trace(s)). The saved parameters and the saved "
        "conditioned column do not describe the same arithmetic, so a replay "
        "with the saved parameters is not the image the scan showed."
        % (out["max_abs_error"], tol, out["checked"]))
    return out


# ==============================================================================
# WRITING
# ==============================================================================

def write_pass(directory, index, poses, impedance, conditioned=None,
               s_m=None, trace_index=None, meta=None):
    """Write one pass. -> the path written.

    `poses` is (n, 7) as TraceAccumulator stores them -- qw qx qy qz x y z --
    and everything else is per-trace and the same length. `meta` is anything
    else worth keeping; it is JSON-encoded into the file, because a plain npz
    holds arrays and the things a reprojection needs to know (the surface path,
    the conditioning parameters, the component, the trigger report) are not
    arrays.

    WRITTEN THROUGH A TEMPORARY AND RENAMED. A scan that dies DURING the write
    would otherwise leave a truncated npz, and np.load on a truncated zip
    raises -- which would take out the replay of all the other passes as well.
    os.replace is atomic within a directory, so a reader sees either the
    previous state or the whole file.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    z = np.asarray(impedance, dtype=np.complex128).reshape(-1)
    n = min(len(poses), len(z))
    poses, z = poses[:n], z[:n]

    cols = {"pose": poses,
            "impedance": z,
            # xyz on its own as well, under the key capture_contact() uses, so
            # anything that already reads those files reads these too. It is a
            # view of `pose`, not a second measurement.
            "position": poses[:, 4:7].copy()}
    if conditioned is not None:
        cols["conditioned"] = np.asarray(conditioned,
                                         dtype=np.float64).reshape(-1)[:n]
    if s_m is not None:
        cols["s_m"] = np.asarray(s_m, dtype=np.float64).reshape(-1)[:n]
    if trace_index is not None:
        cols["index"] = np.asarray(trace_index, dtype=np.int64).reshape(-1)[:n]
    cols["meta"] = json.dumps(_jsonable(dict(meta or {}, pass_index=int(index),
                                             traces=int(n))))

    if not os.path.isdir(directory):
        os.makedirs(directory)
    path = os.path.join(directory, pass_filename(index))
    # The temporary ends in .npz because np.savez APPENDS .npz to any name that
    # does not -- a temporary called "...part" is written as "...part.npz" and
    # the rename then looks for a file that is not there. And it is PREFIXED
    # rather than suffixed so that PASS_GLOB cannot match it: a half-written
    # file must not be offered to a reader even for the instant before the
    # rename.
    tmp = os.path.join(directory, "_writing_%s" % (pass_filename(index),))
    np.savez(tmp, **cols)
    os.replace(tmp, path)
    return path


def _jsonable(obj):
    """numpy scalars and arrays into something json.dumps will take."""
    if isinstance(obj, dict):
        return dict((str(k), _jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


# ==============================================================================
# READING
# ==============================================================================

def read_pass(path):
    """One pass file back. -> dict of arrays plus its decoded `meta`."""
    with np.load(path, allow_pickle=False) as z:
        out = {}
        for key in z.files:
            if key == "meta":
                continue
            out[key] = np.array(z[key])
        meta = {}
        if "meta" in z.files:
            # .item(), not str(): the metadata is a 0-d unicode array, and
            # numpy's str() of an array is a DISPLAY and is free to abbreviate.
            # json.loads on an abbreviated object raises, which would read as a
            # corrupt file.
            try:
                meta = json.loads(z["meta"].item())
            except (ValueError, AttributeError):
                meta = {}
    out["meta"] = meta
    out["path"] = path
    out["pass_index"] = meta.get("pass_index", pass_index_of(path))
    return out


def pass_files(directory):
    """Every pass file in a run directory, in the order the scan wrote them."""
    found = []
    for path in glob.glob(os.path.join(directory, PASS_GLOB)):
        idx = pass_index_of(path)
        if idx is not None:
            found.append((idx, path))
    return [p for _, p in sorted(found)]


def read_run(directory, passes=None):
    """Every saved pass of a run, concatenated. -> dict

    Keys: `pose` (K, 7), `impedance` (K,), `position` (K, 3), `conditioned`
    (K,) or None, `s_m`, `index`, `pass_of` (which pass each trace came from),
    plus what every pass agreed about -- `x3d`, `component`, `null`,
    `rotation_deg`, `gain`, `pitch_m` -- and `passes`, the per-pass metadata.

    `passes` selects pass indices; None takes them all.

    THE PARAMETERS COME FROM THE LAST PASS THAT STATED THEM, and a run whose
    passes disagree says so in `warnings` rather than picking silently. They can
    legitimately differ: the operator may re-null or turn the rotation between
    passes, which re-conditions everything already acquired, so the final
    setting is the one the live image was showing. The raw impedance is
    unaffected either way, which is why the replay works regardless.
    """
    files = pass_files(directory)
    if passes is not None:
        want = set(int(v) for v in passes)
        files = [f for f in files if pass_index_of(f) in want]
    if not files:
        raise IOError(
            "no pass files (%s) in %s. An eddy-current scan writes one per "
            "completed pass; a directory with none either is not a scan "
            "directory or the scan died before its first pass finished."
            % (PASS_GLOB, directory))

    chunks = []
    bad = []
    for path in files:
        try:
            chunks.append(read_pass(path))
        except Exception as e:                                # noqa: BLE001
            # ONE unreadable pass must not cost the other fifty-eight. That is
            # the whole reason the files are per-pass.
            bad.append("%s: %s: %s" % (os.path.basename(path),
                                       type(e).__name__, e))
    if not chunks:
        raise IOError("every pass file in %s is unreadable:\n  %s"
                      % (directory, "\n  ".join(bad)))

    run = {"dir": directory, "files": [c["path"] for c in chunks],
           "passes": [c["meta"] for c in chunks], "warnings": list(bad)}

    def stack(key, width=None, dtype=np.float64):
        have = [c for c in chunks if key in c]
        if len(have) != len(chunks):
            return None
        arrs = [np.asarray(c[key], dtype=dtype) for c in have]
        if width is None:
            return np.concatenate([a.reshape(-1) for a in arrs])
        return np.concatenate([a.reshape(-1, width) for a in arrs], axis=0)

    run["pose"] = stack("pose", 7)
    run["impedance"] = stack("impedance", dtype=np.complex128)
    # THE TWO COLUMNS WITHOUT WHICH THERE IS NO MEASUREMENT. Said here rather
    # than left to fail further on: a replay with no poses would otherwise come
    # apart inside the surface check with a TypeError about None, which says
    # nothing about the files.
    if run["pose"] is None or run["impedance"] is None:
        raise IOError(
            "the pass files in %s do not all carry %s, so there is nothing to "
            "replay: a trace is a complex impedance AND the pose it was taken "
            "at, and neither is recoverable from the other."
            % (directory,
               "a pose column" if run["pose"] is None else "an impedance column"))
    # Derived rather than required: it is a view of `pose` either way, and a
    # file written by an older writer may not have it.
    run["position"] = stack("position", 3)
    if run["position"] is None:
        run["position"] = run["pose"][:, 4:7].copy()
    run["conditioned"] = stack("conditioned")
    run["s_m"] = stack("s_m")
    run["index"] = stack("index", dtype=np.int64)
    run["pass_of"] = np.concatenate(
        [np.full(len(np.asarray(c["pose"]).reshape(-1, 7)),
                 int(c["pass_index"] if c["pass_index"] is not None else -1),
                 dtype=np.int64) for c in chunks])
    run["traces"] = int(len(run["pose"]))

    # What every pass agreed about. A key missing from every pass comes back as
    # the stated default; one that differs between passes takes the last pass's
    # value and leaves a warning.
    for key, default in (("x3d", None), ("mesh", None), ("component", "vertical"),
                         ("null", (0.0, 0.0)), ("rotation_deg", 0.0),
                         ("gain", (1.0, 1.0)), ("pitch_m", None),
                         ("projection_radius_m", None),
                         ("projection_pixels", None), ("label", None)):
        seen = [c["meta"][key] for c in chunks if key in c["meta"]]
        if not seen:
            run[key] = default
            continue
        run[key] = seen[-1]
        uniq = set(json.dumps(_jsonable(v), sort_keys=True) for v in seen)
        if len(uniq) > 1:
            run["warnings"].append(
                "%r is not the same in every pass (%d different values); the "
                "last pass's %r is used" % (key, len(uniq), run[key]))
    run["null"] = tuple(float(v) for v in (tuple(run["null"]) + (0.0, 0.0))[:2])
    run["gain"] = tuple(float(v) for v in (tuple(run["gain"]) + (1.0, 1.0))[:2])
    run["rotation_deg"] = float(run["rotation_deg"])
    return run


def run_directories(root, prefix=RUN_PREFIX):
    """Scan run directories under `root`, oldest first.

    Recognised by the prefix AND by containing at least one pass file, so a
    directory that was created and never written to is not offered as a run to
    replay.
    """
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name)
        if not name.startswith(prefix) or not os.path.isdir(path):
            continue
        if pass_files(path):
            out.append(path)
    return out


def latest_run(root, prefix=RUN_PREFIX):
    """The most recent run directory under `root` that has passes in it.

    By NAME, not by mtime: the directory name carries the scan's start time, and
    an mtime is changed by anything that touches the directory afterwards --
    copying files out of an older run would make it the newest.
    """
    runs = run_directories(root, prefix=prefix)
    return runs[-1] if runs else None


def describe_run(run):
    """A saved run as a short readable block. -> str"""
    lines = ["  %d trace(s) in %d saved pass(es) from %s"
             % (run["traces"], len(run["files"]), run["dir"])]
    if run["pose"] is not None and len(run["pose"]):
        p = run["position"]
        lines.append("    poses span x %.4f..%.4f  y %.4f..%.4f  z %.4f..%.4f m"
                     % (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(),
                        p[:, 1].max(), p[:, 2].min(), p[:, 2].max()))
    lines.append("    surface %s" % (run["x3d"],))
    lines.append("    conditioning as saved: component %s, null (%.5f, %.5f), "
                 "rotation %.1f deg, gain h=%.2f v=%.2f"
                 % ((run["component"],) + run["null"]
                    + (run["rotation_deg"],) + run["gain"]))
    for w in run["warnings"]:
        lines.append("    WARNING: %s" % (w,))
    return "\n".join(lines)
