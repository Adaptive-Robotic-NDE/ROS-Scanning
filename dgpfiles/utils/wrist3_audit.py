#!/usr/bin/env python3
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
# File:  wrist3_audit.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Reads wrist-3 travel out of a run's logs and checks it against the
#   cable envelope.
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
"""What did wrist 3 actually do? Read it out of a run's logs.

The cables to the Azure Kinect and the Keyence limit how far wrist 3 may turn,
and a sweep that commands an angle outside that band risks tearing them. The
guards in the pipeline refuse such a move before it flies; this answers the
after-the-fact question -- "did every commanded angle stay in band?" -- from a
run's logs in one command, so it can be asked of every run, simulated or from
the cell, instead of being reconstructed by hand.

It is deliberately a REPORTER, not a second copy of the limit model. The
authoritative band lives in modules/cable_envelope.dpi and is per-tilt; a log
line does not say what tilt a pose was at, so what this flags is anything
outside the WIDEST measured band. Everything it flags is wrong under every
tilt. Things it does not flag may still be wrong under a tilt, and the run's own
refusals are where that shows up.

    wrist3_audit.py ros.log
    wrist3_audit.py --limits cable_limits_ex-ur10e-1_hardware.json roslog/mtc_node_*.log
    wrist3_audit.py --band 31.3,301.3 /tmp/results2/ros.log

Runs on the host with the standard library only.
"""
import argparse
import json
import math
import os
import re
import sys

#: "Creating Joint Task to configuration: ... wrist_3_joint=-1.972325" -- every
#: joint-space goal the MTC hub is asked for.
JOINT_TASK = re.compile(r"Joint Task to configuration:(?P<joints>[^\n]*)")
WRIST3 = re.compile(r"wrist_3_joint\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

#: The capture tool's own refusals, so a run that was stopped by the guard reads
#: as such rather than as a gap in the trace.
REFUSED = re.compile(r"would turn wrist 3 to ([-\d.]+)\.\.([-\d.]+) deg")

#: What the capture tool recorded when it measured the envelope.
CABLE_STEP = re.compile(r"wrist 3 to [-+]?[\d.]+ deg from the start "
                        r"\((?P<abs>[-\d.]+) deg absolute\)")


def band_from_limits(path):
    """The WIDEST per-tilt band in a cable_limits JSON, plus how it was found.

    Widest rather than the intersection on purpose: this tool cannot know which
    tilt a logged pose was at, so the only sound universal statement is "outside
    every measured band". Narrowing here would flag poses that were legitimately
    in band for the tilt they were flown at.
    """
    doc = json.load(open(path))
    per = doc.get("per_tilt") or []
    if not per:
        lo, hi = doc.get("lo_deg"), doc.get("hi_deg")
        if lo is None or hi is None:
            raise SystemExit("%s has neither per_tilt nor lo_deg/hi_deg" % path)
        return float(lo), float(hi), "the file's single band"
    widest = max(per, key=lambda r: float(r["hi_deg"]) - float(r["lo_deg"]))
    return (float(widest["lo_deg"]), float(widest["hi_deg"]),
            "the widest of %d measured tilts (%s)" % (len(per), widest.get("tilt", "?")))


def commanded(paths):
    """[(source, line_no, degrees)] for every wrist-3 goal in these logs."""
    out = []
    for path in paths:
        try:
            text = open(path, errors="replace").read()
        except OSError as e:
            print("  skipping %s: %s" % (path, e), file=sys.stderr)
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = JOINT_TASK.search(line)
            if not m:
                continue
            w = WRIST3.search(m.group("joints"))
            if w:
                out.append((os.path.basename(path), n, math.degrees(float(w.group(1)))))
    return out


def measurement_steps(paths):
    """Absolute angles the cable-envelope test itself drove to.

    These are not faults -- measuring the limit means going to it -- so they are
    reported separately rather than counted as violations.
    """
    seen = []
    for path in paths:
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        for line in text.splitlines():
            m = CABLE_STEP.search(line)
            if m:
                seen.append(float(m.group("abs")))
    return seen


def refusals(paths):
    out = []
    for path in paths:
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        for line in text.splitlines():
            m = REFUSED.search(line)
            if m:
                out.append((float(m.group(1)), float(m.group(2))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+",
                    help="ros.log, a mtc_node_*.log, a capture session.log, ...")
    ap.add_argument("--limits", metavar="JSON",
                    help="a cable_limits_<cell>_<mode>.json to take the band from")
    ap.add_argument("--band", metavar="LO,HI",
                    help="an explicit band in degrees, overriding --limits")
    args = ap.parse_args(argv)

    band, how = None, ""
    if args.band:
        lo, hi = (float(v) for v in args.band.split(","))
        band, how = (lo, hi), "given on the command line"
    elif args.limits:
        lo, hi, how = band_from_limits(args.limits)
        band = (lo, hi)

    goals = commanded(args.logs)
    if not goals:
        print("No wrist-3 joint goals found in %d log(s). This tool reads the "
              "MTC hub's \"Creating Joint Task to configuration\" lines; a run "
              "that only moved in Cartesian space leaves none."
              % len(args.logs))
        return 0

    degs = [d for _s, _n, d in goals]
    lo_seen, hi_seen = min(degs), max(degs)
    travel = sum(abs(b - a) for a, b in zip(degs, degs[1:]))
    print("wrist 3 over %d commanded configuration(s)" % len(goals))
    print("  range   %8.1f .. %8.1f deg  (span %.1f)"
          % (lo_seen, hi_seen, hi_seen - lo_seen))
    print("  travel  %8.1f deg total between consecutive goals" % travel)

    steps = measurement_steps(args.logs)
    if steps:
        print("  note    %d of these look like cable-envelope MEASUREMENT steps "
              "(%.1f..%.1f deg); measuring the limit means going to it."
              % (len(steps), min(steps), max(steps)))

    for a, b in refusals(args.logs):
        print("  refused a move that would have reached %.1f..%.1f deg "
              "(the guard worked)" % (a, b))

    if band is None:
        print("\nNo band given, so nothing is judged. Pass --limits or --band "
              "to check these against the measured envelope.")
        return 0

    lo, hi = band
    print("\nchecked against %.1f..%.1f deg -- %s" % (lo, hi, how))
    bad = [(s, n, d) for s, n, d in goals if d < lo - 1e-6 or d > hi + 1e-6]
    if not bad:
        print("  OK: every commanded wrist-3 angle is inside the widest "
              "measured band.")
        return 0
    # The cable test drives PAST the band on purpose -- that is how the band
    # gets found -- so those steps are called out rather than counted as faults.
    # Matching by angle is enough: the measurement records the absolute angle it
    # asked for, and the MTC goal that carried it out holds the same number.
    def is_measurement(d):
        return any(abs(d - s) < 0.5 for s in steps)

    faults = [(s, n, d) for s, n, d in bad if not is_measurement(d)]
    probes = [(s, n, d) for s, n, d in bad if is_measurement(d)]
    if probes:
        print("  %d outside it belong to the cable-envelope test itself "
              "(%s deg) -- expected, that is the measurement."
              % (len(probes), ", ".join("%.1f" % d for _s, _n, d in probes)))
    if not faults:
        print("  OK: every other commanded wrist-3 angle is inside the widest "
              "measured band.")
        return 0
    print("  %d commanded angle(s) OUTSIDE it and NOT part of the measurement "
          "-- these are wrong under EVERY measured tilt:" % len(faults))
    for src, n, d in faults:
        print("    %8.1f deg   %s:%d" % (d, src, n))
    return 1


if __name__ == "__main__":
    sys.exit(main())
