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
# File:  ec_bench.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-17
# Description:
#   Characterise the eddy-current instrument and digitizer from a
#   stationary record: block timing, noise, the instrument's own output
#   bandwidth, and the scan speed that bandwidth allows.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-17:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""Are the instrument and the digitizer set up to support the scan we are asking for?

NOBODY HAS EVER MEASURED THIS. The Uniwest US-454A's frequency, drive, gain,
filter and phase are all set on its front panel, and `its settings were not
touched; the null is the only command this config sends`. The DAQ's sample rate
is not recorded anywhere -- `block_samples` is, but without a rate that number
cannot be turned into a time or a distance. So on 2026-10-16 the record said
"100 samples per trace, reduced by mean" and nothing in it could say whether
that block spanned 4% of the trace pitch or 40% of it.

Three questions, all answerable from ONE stationary record with the probe on
the part and the arm not moving:

1. IS THE DIGITIZER DELIVERING WHAT IT WAS ASKED FOR?
   Block rate against nominal, the jitter on it, and the gaps. A card that is
   dropping blocks correlates traces against stale data and nothing downstream
   notices.

2. IS BLOCK AVERAGING BUYING ANYTHING?
   Averaging n independent samples cuts noise by sqrt(n). If averaging 100
   samples cuts it by 2 instead of by 10, the samples are NOT independent --
   the instrument's output filter is slower than the block, so 98 of those
   samples are re-measuring what the first two already said. That is the single
   clearest statement of "the digitizer is sampling faster than the instrument
   can talk", and it is free to measure.

3. WHAT IS THE INSTRUMENT'S OUTPUT BANDWIDTH, AND WHAT SCAN SPEED DOES IT ALLOW?
   The analog output is filtered inside the instrument. That filter, not the
   robot and not the staging pipeline, sets how fast the probe may cross a flaw
   before the flaw is smeared out. Its corner frequency falls straight out of
   the noise spectrum of a stationary record -- the noise is shaped by the same
   filter the signal is -- and a corner frequency plus a flaw width is a speed
   limit in mm/s.

   It cuts both ways, which is why it is worth measuring rather than assuming.
   A 10 Hz corner says 2.5 mm/s is already at the limit and the slow scan is
   CORRECT. A 500 Hz corner says the instrument would support 100 mm/s and
   every minute of a 44-minute scan is being spent somewhere else.

Everything here is pure numpy over arrays. ec_bench.dpi is what reads the
recording database and calls it; this file is testable on a laptop.
"""
import math

import numpy as np

__all__ = [
    "block_timing", "averaging_gain", "noise_spectrum", "corner_frequency",
    "mains_content", "speed_limit_from_bandwidth", "describe_bench",
    "MAINS_HZ", "SPEED_LIMIT_SAMPLES_PER_FEATURE",
]

#: Mains frequency, for the hum check. 60 Hz in North America.
MAINS_HZ = 60.0

#: How many samples across the narrowest feature the scan must resolve before
#: the instrument's filter is called the limit. Two is Nyquist and is not
#: enough to say anything about a flaw's shape; five is the usual NDE rule of
#: thumb for calling an indication characterised rather than merely detected.
#: See [[ec-step-size-must-resolve-the-flaw]] -- the same argument, applied to
#: the instrument instead of to the step size.
SPEED_LIMIT_SAMPLES_PER_FEATURE = 5.0


def block_timing(times_s, nominal_hz=None):
    """Did the blocks arrive when they were supposed to? -> dict

    `times_s` is one timestamp per block, in arrival order. The measured rate
    is the median interval rather than the mean, because one long gap must not
    be allowed to look like a uniformly slow card -- the two have different
    causes and different fixes.
    """
    t = np.asarray(times_s, dtype=np.float64).reshape(-1)
    out = {"blocks": int(t.size), "span_s": 0.0, "rate_hz": None,
           "nominal_hz": (None if nominal_hz is None else float(nominal_hz)),
           "median_dt_s": None, "jitter_s": None, "worst_gap_s": None,
           "gaps": 0, "ok": None, "why": ""}
    if t.size < 3:
        out["why"] = "fewer than 3 blocks: nothing to measure"
        return out
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 2:
        out["why"] = "the timestamps do not increase"
        return out
    med = float(np.median(dt))
    out["span_s"] = float(t[-1] - t[0])
    out["median_dt_s"] = med
    out["rate_hz"] = 1.0/med if med > 0 else None
    # Median absolute deviation, scaled to a standard deviation, so a handful
    # of long gaps do not become "the card jitters".
    out["jitter_s"] = float(np.median(np.abs(dt - med))*1.4826)
    out["worst_gap_s"] = float(dt.max())
    # A gap is a block that took more than 1.5 intervals: late enough that a
    # block went missing rather than merely arrived late.
    out["gaps"] = int((dt > 1.5*med).sum())
    if nominal_hz:
        err = abs(out["rate_hz"] - float(nominal_hz))/float(nominal_hz)
        out["rate_error"] = float(err)
        out["ok"] = bool(err < 0.05 and out["gaps"] == 0)
        out["why"] = ("%.1f Hz against a nominal %.1f (%.1f%% off), %d gap(s)"
                      % (out["rate_hz"], float(nominal_hz), err*100.0,
                         out["gaps"]))
    else:
        out["ok"] = bool(out["gaps"] == 0)
        out["why"] = ("%.1f Hz measured, no nominal rate to check it against, "
                      "%d gap(s)" % (out["rate_hz"], out["gaps"]))
    return out


def averaging_gain(x, n):
    """Does averaging n samples actually cut the noise by sqrt(n)? -> dict

    THE QUESTION THE SCAN ACTUALLY ASKS. Every trace is the mean of a block of
    `n` samples, and that is only worth doing if the samples are independent.
    They are independent only if the instrument's output changes faster than
    the digitizer samples it; if its filter is slower, consecutive samples are
    the same measurement read twice and the average is no quieter than one
    sample.

    `efficiency` is the measured reduction divided by sqrt(n):

        ~1.0   the samples are independent and the averaging is worth its
               smear -- see report_smear(), which says how far the probe moved
               during the block
        ~0.3   the block is mostly redundant; a slower sample rate or a
               shorter block would cost almost no noise and smear less
        >1.2   the single-sample noise is dominated by something the averaging
               removes entirely, which usually means a periodic interferer
               (see mains_content) rather than white noise
    """
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(n)
    out = {"n": n, "samples": int(v.size), "sigma_single": None,
           "sigma_block": None, "ideal_ratio": None, "measured_ratio": None,
           "efficiency": None, "why": ""}
    if n < 2 or v.size < 4*n:
        out["why"] = ("need at least 4 blocks of %d samples, have %d"
                      % (n, v.size))
        return out
    usable = (v.size//n)*n
    blocks = v[:usable].reshape(-1, n)
    # DETREND PER BLOCK before measuring the single-sample noise, so that slow
    # drift -- which the block mean keeps and which is NOT what averaging is
    # meant to remove -- is not counted as sample noise and does not flatter
    # the efficiency.
    single = float(np.std(blocks - blocks.mean(axis=1, keepdims=True), ddof=1))
    means = blocks.mean(axis=1)
    block = float(np.std(means - means.mean(), ddof=1))
    out["sigma_single"] = single
    out["sigma_block"] = block
    out["ideal_ratio"] = math.sqrt(n)
    if block > 0 and single > 0:
        out["measured_ratio"] = single/block
        out["efficiency"] = (single/block)/math.sqrt(n)
        out["why"] = ("averaging %d samples cuts the noise %.1fx, against "
                      "%.1fx if they were independent (%.0f%%)"
                      % (n, out["measured_ratio"], out["ideal_ratio"],
                         100.0*out["efficiency"]))
    else:
        out["why"] = "one of the two standard deviations is zero"
    return out


def noise_spectrum(x, fs, nperseg=None):
    """One-sided power spectral density of a stationary record. -> (f, pxx)

    Welch's method with a Hann window and 50% overlap, written out rather than
    imported: scipy is not a dependency of this tree and this is thirty lines.
    The absolute scale is V^2/Hz; only the SHAPE is used below, so a constant
    factor would not change any conclusion drawn from it.
    """
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = float(fs)
    if v.size < 16 or fs <= 0:
        return np.zeros(0), np.zeros(0)
    if nperseg is None:
        # RESOLUTION AND SMOOTHNESS TRADE AGAINST EACH OTHER and the corner
        # search needs the second one more. Going to the finest bins the record
        # allows -- about 8 segments to average -- makes every bin so noisy
        # that a single low one crosses the threshold early: measured, a 400 Hz
        # filter read as 240 and a 100 Hz filter as 74. Sixteen or more
        # segments at a coarser bin reads all of them to within a few percent.
        nperseg = min(4096,
                      1 << int(math.floor(math.log2(max(16, v.size//16)))))
    nperseg = int(min(nperseg, v.size))
    step = max(1, nperseg//2)
    win = np.hanning(nperseg)
    scale = 1.0/(fs*np.sum(win**2))
    acc = np.zeros(nperseg//2 + 1)
    segs = 0
    for start in range(0, v.size - nperseg + 1, step):
        seg = v[start:start + nperseg]
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg*win))**2
        acc += spec
        segs += 1
    if not segs:
        return np.zeros(0), np.zeros(0)
    pxx = acc*scale/segs
    pxx[1:-1] *= 2.0                       # one-sided
    f = np.fft.rfftfreq(nperseg, 1.0/fs)
    return f, pxx


def corner_frequency(f, pxx, drop_db=3.0, exclude_hz=()):
    """Where the noise rolls off -- the instrument's output filter. -> dict

    The noise leaving the instrument is shaped by the same output filter the
    SIGNAL is, so the frequency at which the noise power falls `drop_db` below
    its low-frequency plateau is the filter's corner, and the signal cannot be
    faster than that either.

    `exclude_hz` removes narrow interferers (mains and its harmonics) from the
    plateau estimate, because a 60 Hz line sitting on top of the plateau would
    raise it and push the apparent corner out.

    Reported as a RANGE, not a point: the search returns the first frequency at
    or below which the smoothed spectrum stays under the threshold, and the
    smoothing's half-width is the honest uncertainty on it.

    ACCURACY, measured against one-pole filters of known corner in 20 s of
    2 kHz data (ec_bench_test.py): within about 15% from 25 Hz to 400 Hz, and
    10-45% high at 5-10 Hz where the 0.5 Hz bin and the 5-bin smoothing are a
    sizeable fraction of the corner itself. Good enough to decide whether the
    instrument is the thing limiting the scan -- a question with a factor-of-
    1.5 threshold on it -- and not good enough to quote as an instrument
    specification.
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    p = np.asarray(pxx, dtype=np.float64).reshape(-1)
    out = {"corner_hz": None, "plateau": None, "bin_hz": None, "why": ""}
    if f.size < 8 or p.size != f.size:
        out["why"] = "spectrum too short"
        return out
    bin_hz = float(f[1] - f[0]) if f.size > 1 else None
    out["bin_hz"] = bin_hz
    keep = f > 0
    for hz in exclude_hz:
        keep &= np.abs(f - hz) > max(2.0*(bin_hz or 1.0), 2.0)
    if keep.sum() < 8:
        out["why"] = "everything was excluded"
        return out
    fk, pk = f[keep], p[keep]
    # The plateau is the median of the lowest decade of frequencies present,
    # which is robust to a single bin and does not assume the spectrum starts
    # flat at DC.
    lo = fk <= max(fk[0]*10.0, fk[min(8, fk.size - 1)])
    plateau = float(np.median(pk[lo])) if lo.sum() else float(np.median(pk))
    out["plateau"] = plateau
    if plateau <= 0:
        out["why"] = "the low-frequency plateau is zero"
        return out
    thresh = plateau*(10.0**(-abs(drop_db)/10.0))
    # Smooth over 5 bins so one quiet bin cannot be read as the corner.
    k = min(5, pk.size)
    sm = np.convolve(pk, np.ones(k)/k, mode="same")
    # ONLY WHERE THE SMOOTHING WINDOW IS FULLY INSIDE THE BAND, and only below
    # the top of it. np.convolve(mode="same") zero-pads, so the first and last
    # k//2 bins are averages of a partly-empty window and sit ~40% low -- which
    # is within a 3 dB threshold on its own. Both edges were wrong and in
    # opposite ways: a 25 Hz filter reported its corner at the first bin
    # (0.5 Hz, a 50x under-estimate) and flat white noise reported one at
    # Nyquist (a limitless over-estimate). The top of the band is excluded as
    # well because the last bin of a one-sided spectrum is halved by
    # convention, and because a corner found there is not a corner -- it is the
    # edge of what this record can see.
    edge = k//2
    usable = max(edge + 1, int(sm.size*0.95))
    under = sm[edge:usable] < thresh
    # A RUN, NOT A BIN. One bin of a periodogram scatters by its own size even
    # after averaging segments, so the first bin under the threshold is often
    # just a quiet one on the flat part of the spectrum. A real corner stays
    # under it: requiring the next `run` bins to be under as well is what turns
    # "the noise dipped here" into "the spectrum has rolled off here".
    run = min(10, max(1, under.size//8))
    sustained = under.copy()
    for shift in range(1, run):
        sustained[:under.size - shift] &= under[shift:]
    below = np.where(sustained)[0] + edge
    if below.size == 0:
        out["why"] = ("the spectrum never falls %.0f dB below its plateau "
                      "below %.0f Hz, so the filter is at or above what a "
                      "%.0f Hz record can see -- measure again with a faster "
                      "card or accept that this is not the limit"
                      % (abs(drop_db), fk[min(usable, fk.size - 1)], 2.0*f[-1]))
        out["searched_above_hz"] = float(fk[edge])
        return out
    out["corner_hz"] = float(fk[below[0]])
    # THE SMOOTHING BIASES IT HIGH. A k-bin box average at the knee pulls in
    # power from the flat side, so the first bin that falls below the threshold
    # is up to k/2 bins above the true corner. That is reported rather than
    # corrected, because the correction depends on the filter order and the
    # bracket is what a decision should be made on: `corner_low_hz` is the
    # conservative end and is what speed limits should be taken from.
    half = 0.5*k*(bin_hz or 0.0)
    out["uncertainty_hz"] = float(max(bin_hz or 0.0, half))
    out["corner_low_hz"] = float(max(bin_hz or 0.0, out["corner_hz"] - half))
    out["why"] = ("noise power falls %.0f dB by %.1f Hz (%.1f..%.1f Hz once "
                  "the %d-bin smoothing is allowed for)"
                  % (abs(drop_db), out["corner_hz"], out["corner_low_hz"],
                     out["corner_hz"], k))
    return out


def mains_content(f, pxx, mains_hz=None, harmonics=3):
    """How much of the noise is mains hum? -> dict

    A fraction worth naming on its own: hum is fixable (grounding, cable
    routing, the instrument's own filter) and broadband noise usually is not,
    so a scan limited by one needs a different answer from a scan limited by
    the other.
    """
    mains_hz = MAINS_HZ if mains_hz is None else float(mains_hz)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    p = np.asarray(pxx, dtype=np.float64).reshape(-1)
    out = {"mains_hz": mains_hz, "lines": [], "fraction": None, "why": ""}
    if f.size < 4 or p.size != f.size:
        out["why"] = "spectrum too short"
        return out
    bin_hz = float(f[1] - f[0])
    total = float(np.sum(p[f > 0]))
    if total <= 0:
        out["why"] = "no power in the spectrum"
        return out
    inline = 0.0
    for h in range(1, int(harmonics) + 1):
        hz = mains_hz*h
        if hz > f[-1]:
            break
        sel = np.abs(f - hz) <= max(2.0*bin_hz, 1.0)
        if not sel.any():
            continue
        here = float(np.sum(p[sel]))
        inline += here
        out["lines"].append({"hz": hz, "fraction": here/total})
    out["fraction"] = inline/total
    out["why"] = ("%.1f%% of the noise power sits within a couple of bins of "
                  "%.0f Hz and its first %d harmonic(s)"
                  % (100.0*out["fraction"], mains_hz, int(harmonics)))
    return out


def speed_limit_from_bandwidth(corner_hz, feature_m,
                               samples_per_feature=None):
    """The scan speed above which the instrument smears the flaw. -> dict

    A first-order filter at `corner_hz` resolves a feature of width
    `feature_m` only if the probe spends long enough crossing it for the
    filter to respond `samples_per_feature` times:

        v_max = corner_hz * feature_m / samples_per_feature

    This is the instrument's limit and it is independent of the trace pitch,
    the robot and the staging pipeline. Compare it against all three: whichever
    is smallest is what the scan is really waiting for, and the other two are
    not worth optimising until it is.
    """
    spf = (SPEED_LIMIT_SAMPLES_PER_FEATURE if samples_per_feature is None
           else float(samples_per_feature))
    out = {"corner_hz": corner_hz, "feature_m": float(feature_m),
           "samples_per_feature": spf, "v_max_mps": None, "why": ""}
    if not corner_hz or corner_hz <= 0 or feature_m <= 0 or spf <= 0:
        out["why"] = "no corner frequency measured, so no limit can be stated"
        return out
    v = float(corner_hz)*float(feature_m)/spf
    out["v_max_mps"] = v
    out["why"] = ("a %.1f Hz output filter crosses a %.2f mm feature with %.0f "
                  "samples across it at up to %.1f mm/s"
                  % (float(corner_hz), feature_m*1e3, spf, v*1e3))
    return out


def describe_bench(report):
    """The whole measurement as text, with what each number means."""
    L = []
    t = report.get("timing") or {}
    L.append("DIGITIZER")
    if t.get("rate_hz"):
        L.append("  block rate        %.2f Hz over %.1f s (%d blocks)"
                 % (t["rate_hz"], t.get("span_s") or 0.0, t.get("blocks") or 0))
        L.append("  jitter            %.2f ms (robust sd of the interval)"
                 % ((t.get("jitter_s") or 0.0)*1e3,))
        L.append("  worst gap         %.1f ms; %d block(s) arrived more than "
                 "1.5 intervals late"
                 % ((t.get("worst_gap_s") or 0.0)*1e3, t.get("gaps") or 0))
        if t.get("nominal_hz"):
            L.append("  against nominal   %.2f Hz -- %s"
                     % (t["nominal_hz"], "OK" if t.get("ok") else "OFF"))
    else:
        L.append("  not measured: %s" % (t.get("why") or "no blocks",))
    if report.get("sample_rate_hz"):
        L.append("  sample rate       %.0f Hz (%d samples per block, %.1f ms "
                 "of signal per trace)"
                 % (report["sample_rate_hz"], report.get("block_samples") or 0,
                    1e3*(report.get("block_samples") or 0)
                    / report["sample_rate_hz"]))

    L.append("")
    L.append("NOISE, with the probe stationary on the part")
    for name in ("horizontal", "vertical"):
        ch = (report.get("channels") or {}).get(name) or {}
        if not ch:
            continue
        L.append("  %-10s      sd %.4g V single sample, %.4g V after the "
                 "block average" % (name, ch.get("sigma_single") or float("nan"),
                                    ch.get("sigma_block") or float("nan")))
        eff = ch.get("efficiency")
        if eff is not None:
            verdict = ("the block is buying its full sqrt(n)" if eff > 0.8 else
                       "MOST OF THE BLOCK IS REDUNDANT" if eff < 0.5 else
                       "the block is buying some of its sqrt(n)")
            L.append("                    %s -- %s" % (ch.get("why", ""), verdict))
        m = ch.get("mains") or {}
        if m.get("fraction") is not None:
            L.append("                    %s" % (m["why"],))
        c = ch.get("corner") or {}
        L.append("                    %s" % (c.get("why") or "no corner found",))

    L.append("")
    L.append("WHAT THAT ALLOWS")
    lim = report.get("limit") or {}
    if lim.get("v_max_mps"):
        L.append("  %s" % (lim["why"],))
    else:
        L.append("  %s" % (lim.get("why") or "not computed",))
    for line in report.get("verdict") or []:
        L.append("  %s" % (line,))
    return "\n".join(L)
