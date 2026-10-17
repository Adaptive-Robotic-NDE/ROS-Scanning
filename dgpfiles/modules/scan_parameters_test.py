"""Check the scan-plan arithmetic against a reference scan with measured values.

Runs on the host with nothing else present: scan_parameters.dpi is deliberately
free of ROS, spatialnde2 and the recording database, so the derivation that
decides whether a scan is runnable can be checked without a cell.
"""
import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "scan_parameters.dpi")).read()
# A .dpi declares its exports global before assigning them; at module scope that
# is a no-op, but "global X" before "class X" is a SyntaxError, so strip them.
src = "\n".join(line for line in src.splitlines()
                if not line.startswith("global "))
ns = {}
exec(compile(src, "scan_parameters.dpi", "exec"), ns)

plan_scan = ns["plan_scan"]
format_scan_plan = ns["format_scan_plan"]
check_scan_outcome = ns["check_scan_outcome"]
auto_spot_radius = ns["auto_spot_radius"]
traces_per_spot = ns["traces_per_spot"]
auto_projection_pixels = ns["auto_projection_pixels"]
velocity_scaling_for = ns["velocity_scaling_for"]
velocity_from_scaling = ns["velocity_from_scaling"]
ScanLimitError = ns["ScanLimitError"]

# A measured cell curve: linear at ~1.55 m/s full scale up to about 30 mm/s,
# then saturating near 53 mm/s. See ROBOT_VELOCITY_CALIBRATION.
CAL = [(0.0050, 0.0079), (0.0100, 0.0156), (0.0200, 0.0303),
       (0.0350, 0.0472), (0.0500, 0.0529)]

fails = []


def check(name, got, want):
    ok = (got == want)
    print("  %-56s %s" % (name, "ok" if ok else "FAIL  got=%r want=%r" % (got, want)))
    if not ok:
        fails.append(name)


def close(name, got, want, tol):
    ok = abs(got - want) <= tol
    print("  %-56s %s" % (name, "ok" if ok else
                          "FAIL  got=%.6g want=%.6g +-%.3g" % (got, want, tol)))
    if not ok:
        fails.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(fails)
        fn(*args, **kwargs)
        failed = fails[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


# A bowed test plate as Noether plans it: 30 passes over a 0.30 m span, each
# 0.308 m long because the 30 mm bow makes the surface longer than its footprint.
# Every measured number below is against this geometry; 0.308/0.00508 is its
# 61 traces per pass.
PLATE = dict(path_length_m=9.24, passes=30, num_samples=16000, sample_bytes=4,
             file_backed=True, velocity_calibration=CAL)


@_checked
def test_the_step_size_velocity_table():
    print("=== the step-size / velocity table, live projection ===")
    # step size -> max velocity, at the measured 10 traces/s live ceiling.
    for step_mm, want_vmax_mm in ((5.08, 50.8), (1.0, 10.0), (0.5, 5.0), (0.1, 1.0)):
        p = plan_scan(step_m=step_mm*1e-3, velocity_mps=1.0, force=True, **PLATE)
        close("%.2f mm step -> %.1f mm/s ceiling" % (step_mm, want_vmax_mm),
              p["max_velocity_mps"]*1e3, want_vmax_mm, 1e-6)

    print()


@_checked
def test_velocity_is_clamped_and_says_so():
    print("=== velocity is scaled down, and says so ===")
    p = plan_scan(step_m=0.0005, velocity_mps=0.050, force=True, **PLATE)
    check("50 mm/s at a 0.5 mm step is clamped", p["velocity_clamped"], True)
    close("clamped to 5 mm/s", p["effective_velocity_mps"]*1e3, 5.0, 1e-9)
    close("which holds the trace rate at the ceiling", p["trace_rate_hz"], 10.0, 1e-9)
    check("and that is not an error, it is a clamp", p["errors"], [])

    p = plan_scan(step_m=0.00508, velocity_mps=0.030, force=True, **PLATE)
    check("30 mm/s at a 5.08 mm step is under the ceiling", p["velocity_clamped"], False)
    close("so it is used as asked", p["effective_velocity_mps"], 0.030, 1e-12)

    print()


@_checked
def test_far_over_the_ceiling_is_an_error():
    print("=== 10x over the ceiling is an error, not a clamp ===")
    try:
        plan_scan(step_m=0.0001, velocity_mps=0.050, **PLATE)
        check("0.1 mm step at 50 mm/s raises", False, True)
    except ScanLimitError as e:
        check("0.1 mm step at 50 mm/s raises", True, True)
        check("the message names the pipeline ceiling",
              "1.0 mm/s this pipeline can sustain" in str(e), True)

    print()


@_checked
def test_the_measured_reference_scan_comes_back():
    print("=== the measured 5.5 min reference scan comes back out ===")
    # 30 passes of 0.31 m at 50.8 mm/s, plus 5 s/pass of overhead, which measures
    # 329.7 s on the cell.
    p = plan_scan(step_m=0.00508, velocity_mps=0.0508, force=True, **PLATE)
    close("estimated scan time within 10%% of the measured 329.7 s",
          p["scan_time_s"], 329.7, 33.0)
    check("1830 traces, as the 30-pass run staged", p["traces"], 1830)
    # 117,120,000 bytes, which is 117.1 MB decimal and 111.7 MiB. plan_scan()
    # reports MiB, as volume_size_note() does.
    close("trace memory 117.1 MB / 111.7 MiB", p["memory_bytes"]/1e6, 117.1, 0.2)

    print()


@_checked
def test_the_memory_column():
    print("=== the memory column of the same table ===")
    for step_mm, want_mb in ((5.08, 117.0), (1.0, 590.0), (0.5, 1180.0)):
        p = plan_scan(step_m=step_mm*1e-3, velocity_mps=0.001, force=True, **PLATE)
        close("%.2f mm step -> %.0f MB" % (step_mm, want_mb),
              p["memory_bytes"]/1e6, want_mb, 0.03*want_mb)

    print()


@_checked
def test_an_unrunnable_scan_is_refused_with_the_reason():
    print("=== an unrunnable scan is refused, with the reason ===")
    try:
        plan_scan(step_m=0.0001, velocity_mps=0.001, **PLATE)
        check("2.6 h scan raises", False, True)
    except ScanLimitError as e:
        check("2.6 h scan raises", True, True)
        check("named as a time limit", "exceeds the" in str(e) and "limit" in str(e), True)

    try:
        # 1.2 GB of traces with storage in memory.
        plan_scan(step_m=0.0005, velocity_mps=0.005,
                  path_length_m=9.30, passes=30, num_samples=16000, sample_bytes=4,
                  file_backed=False)
        check("1.2 GB in memory raises", False, True)
    except ScanLimitError as e:
        check("1.2 GB in memory raises", True, True)
        check("and points at SCAN_STORAGE_DIR", "SCAN_STORAGE_DIR" in str(e), True)

    p = plan_scan(step_m=0.0005, velocity_mps=0.005, force=True, **PLATE)
    check("the same scan file-backed is a warning, not an error", p["errors"], [])
    check("and the warning says so", any("file-backed" in w for w in p["warnings"]), True)

    print()


@_checked
def test_projection_defaults_follow_the_step():
    print("=== projection defaults follow the step ===")
    close("8 mm spot for the 5.08 mm step that measured 96.6% coverage",
          auto_spot_radius(0.00508)*1e3, 8.13, 0.2)
    check("512 px is enough for that spot over 0.30 m",
          auto_projection_pixels(0.008, 0.30), 512)
    # A 0.5 mm step wants a 0.8 mm spot, which at 512 px over 0.30 m is only
    # 2.7 px, so the default has to go up to 2048.
    check("a 0.5 mm step needs 2048", auto_projection_pixels(auto_spot_radius(0.0005), 0.30),
          2048)
    p = plan_scan(step_m=0.0005, velocity_mps=0.005, projection_pixels=512, force=True,
                  **PLATE)
    check("asking for 512 there warns",
          any("cannot draw it" in w for w in p["warnings"]), True)
    p = plan_scan(step_m=0.00508, velocity_mps=0.0508, spot_radius_m=0.004, force=True,
                  **PLATE)
    check("a spot under 1.2x the step warns",
          any("lattice of dots" in w for w in p["warnings"]), True)

    print()


@_checked
def test_the_spot_radius_follows_both_sample_spacings():
    """The rule is a trace COUNT, so a square grid is unchanged and a raster
    with closely spaced lines stops being blurred by its coarse axis.

    "radius = 1.6 x step" ignored the line spacing entirely. An eddy-current
    scan at a 1.00 mm pitch and a 0.30 mm line spacing therefore got the same
    1.6 mm radius as a 1.00 mm square grid -- 26 traces averaged into every
    pixel over a 3.2 mm disc, against the 8 the rule was calibrated for.
    Measured by reprojecting a real 59-pass scan, that cost a factor of 2.2 in
    notch SNR.
    """
    print("=== the spot radius follows both sample spacings ===")
    # SQUARE GRIDS ARE UNCHANGED, which is what makes this safe to land: the one
    # measured calibration point keeps the radius it was measured at.
    for step_mm, want_mm in ((5.08, 8.11), (1.0, 1.60), (0.5, 0.80), (0.3, 0.48)):
        close("a %.2f mm square grid still gets %.2f mm" % (step_mm, want_mm),
              auto_spot_radius(step_mm*1e-3, step_mm*1e-3)*1e3, want_mm, 0.02)
    close("and so does a caller that passes no line spacing",
          auto_spot_radius(0.00508)*1e3, 8.11, 0.02)

    # THE CASE THAT WAS WRONG.
    close("1.00 mm step x 0.30 mm lines -> 0.87 mm, not 1.60",
          auto_spot_radius(0.001, 0.0003)*1e3, 0.874, 0.01)
    close("the old 1.6 mm radius was averaging 26 traces there",
          traces_per_spot(0.0016, 0.001, 0.0003), 26.8, 0.2)
    close("the new one averages the 8 it is meant to",
          traces_per_spot(auto_spot_radius(0.001, 0.0003), 0.001, 0.0003), 8.0, 0.01)

    # Monotone in each spacing, and never larger than the square-grid answer for
    # the coarser axis -- closer lines may only shrink it.
    check("closer lines shrink the spot",
          auto_spot_radius(0.001, 0.0002) < auto_spot_radius(0.001, 0.001), True)
    check("a line spacing of zero falls back to the step rather than to zero",
          abs(auto_spot_radius(0.001, 0.0)*1e3 - 1.60) < 0.02, True)
    print()


@_checked
def test_a_manual_spot_radius_is_judged_by_trace_count():
    """A radius under the step is fine when the lines are close together.

    The old warning fired on `radius < 1.2 x step`, which would have refused to
    shut up about the very radius this rule now picks: 0.87 mm against a 1.00 mm
    step. The question is how many traces land in the spot, not how the radius
    compares with one of the two spacings.
    """
    print("=== a manual spot radius is judged by trace count ===")
    p = plan_scan(step_m=0.001, velocity_mps=0.005, spot_radius_m=0.0007,
                  line_spacing_m=0.0003, force=True, **PLATE)
    check("0.70 mm over a 1.00 x 0.30 mm cell does not warn",
          any("lattice of dots" in w for w in p["warnings"]), False)
    close("because it still reaches 5 traces", p["traces_per_spot"], 5.13, 0.05)

    p = plan_scan(step_m=0.001, velocity_mps=0.005, spot_radius_m=0.0007,
                  line_spacing_m=0.001, force=True, **PLATE)
    check("the same radius on a square grid DOES warn",
          any("lattice of dots" in w for w in p["warnings"]), True)
    print()


@_checked
def test_deferred_projection_raises_the_ceiling():
    print("=== deferred projection raises the ceiling ===")
    live = plan_scan(step_m=0.001, velocity_mps=0.050, projection_mode="live",
                     force=True, **PLATE)
    defer = plan_scan(step_m=0.001, velocity_mps=0.050, projection_mode="deferred",
                      force=True, **PLATE)
    check("live is bound by the measured 10 traces/s", live["max_trace_rate_hz"], 10.0)
    check("deferred by the measured 22", defer["max_trace_rate_hz"], 22.0)
    check("so deferred runs faster at the same step",
          defer["effective_velocity_mps"] > live["effective_velocity_mps"], True)
    # An unmeasured mode falls back to the serialised-globalrev floor -- 1/(1*0.02)
    # -- and that is optimistic by more than 2x against the 22 actually measured,
    # which is the reason to measure rather than to trust the floor.
    unmeasured = dict(DEFAULT_LIMITS := ns["DEFAULT_LIMITS"])
    unmeasured["max_trace_rate_hz"] = {"deferred": None}
    p = plan_scan(step_m=0.001, velocity_mps=0.050, projection_mode="deferred",
                  limits=unmeasured, force=True, **PLATE)
    close("the floor alone would claim 50 traces/s", p["max_trace_rate_hz"], 50.0, 1e-9)

    print()


@_checked
def test_the_timer_raster_is_bound_by_the_digitizer():
    print("=== the sample-and-bin raster is bound by the digitizer, not the stager ===")
    p = plan_scan(step_m=0.00508, velocity_mps=0.050, trigger_mode="timer",
                  digitizer_rate_hz=50.0, force=True, **PLATE)
    close("50 Hz PRF / 3 samples per column", p["max_trace_rate_hz"], 50.0/3.0, 1e-9)
    close("-> an 84.7 mm/s ceiling at the 5.08 mm column pitch",
          p["max_velocity_mps"]*1e3, 50.0/3.0*5.08, 1e-6)
    check("so the raster's usual 50 mm/s is fine",
          p["velocity_clamped"], False)
    # The failure the clamp prevents: sweep a row faster than the digitizer can
    # fill its columns and _row_entries fabricates the empty ones from a
    # neighbour.
    p = plan_scan(step_m=0.00508, velocity_mps=0.150, trigger_mode="timer",
                  digitizer_rate_hz=50.0, force=True, **PLATE)
    check("150 mm/s is clamped", p["velocity_clamped"], True)
    check("and it is the digitizer that said so",
          "digitizer" in p["rate_limited_by"], True)

    print()


@_checked
def test_velocity_scaling_is_a_measured_curve():
    print("=== velocity scaling is a measured curve, not one ratio ===")
    # Every measured point must come back exactly, in both directions.
    for s, v in CAL:
        close("scaling %.4f -> %.1f mm/s" % (s, v*1e3),
              velocity_from_scaling(s, CAL)*1e3, v*1e3, 1e-9)
        close("%.1f mm/s -> scaling %.4f" % (v*1e3, s),
              velocity_scaling_for(v, CAL), s, 1e-9)
    # Why a table and not one ratio: a single ratio taken at the top of the range
    # over-commands everything slower. A constant scaling of 0.0100 is meant to be
    # 10 mm/s, and this cell runs it at 15.6 mm/s.
    close("a single-constant scaling of 0.0100 for 10 mm/s", 0.010, 0.010, 0.0)
    close("...which this cell really runs at", velocity_from_scaling(0.010, CAL)*1e3,
          15.6, 1e-9)
    close("the table asks for 0.0064 instead", velocity_scaling_for(0.010, CAL),
          0.0064, 0.0002)
    close("and that does produce 10 mm/s",
          velocity_from_scaling(velocity_scaling_for(0.010, CAL), CAL)*1e3, 10.0, 1e-6)

    print()


@_checked
def test_the_cells_own_ceiling_is_reported():
    print("=== the cell's own ceiling is reported, not silently exceeded ===")
    p = plan_scan(step_m=0.010, velocity_mps=0.100, force=True, **PLATE)
    close("100 mm/s comes back as the 52.9 mm/s the cell tops out at",
          p["effective_velocity_mps"]*1e3, 52.9, 1e-6)
    check("and says so", any("tops out" in w for w in p["warnings"]), True)
    p = plan_scan(step_m=0.0005, velocity_mps=0.005, force=True, **PLATE)
    check("below the calibrated range is flagged as extrapolation",
          any("extrapolated through the origin" in w for w in p["warnings"]), True)

    print()


@_checked
def test_the_outcome_check():
    print("=== the outcome check against a run that matched ===")
    p = plan_scan(step_m=0.00508, velocity_mps=0.0508, force=True, **PLATE)
    report = {"elapsed": 329.7, "staged": 1830, "stager_max_queue_depth": 5,
              "per_segment": [{"observed_m": 0.31,
                               "sweep_s": 0.31/p["effective_velocity_mps"]}]*30}
    lines = check_scan_outcome(p, report)
    check("a matching run reports no calibration complaint",
          any("CALIBRATION is wrong" in l for l in lines), False)
    check("and does report the achieved rate",
          any("achieved" in l for l in lines), True)

    # Same plan, but the arm actually moved at half the commanded speed.
    slow = {"elapsed": 660.0, "staged": 1830, "stager_max_queue_depth": 40,
            "per_segment": [{"observed_m": 0.31,
                             "sweep_s": 2*0.31/p["effective_velocity_mps"]}]*30}
    lines = check_scan_outcome(p, slow)
    check("half speed is called out",
          any("TWO THINGS DO THAT" in l for l in lines), True)
    # IT NAMES BOTH CAUSES, not one. The calibration maps a scaling to a PEAK
    # tool speed and this measures the MEAN over a pass; on a surface-following
    # toolpath those differ by 2.5x with the table perfectly correct. The old
    # line asserted the calibration was wrong and sent the next reader to
    # re-measure a table that was fine -- 2026-10-16, where mtc's own
    # trajectory check reported a peak of 2.80 mm/s against 2.5 commanded.
    check("it names the calibration as one cause, with the entry to add",
          any("(%.4f," % p["velocity_scaling"] in l for l in lines), True)
    check("and the trajectory as the other",
          any("TOTG decelerating" in l for l in lines), True)
    check("and says which line of mtc's output tells them apart",
          any("Trajectory check" in l for l in lines), True)
    check("a queue of 40 is called out as backpressure",
          any("ran BEHIND" in l for l in lines), True)

    print()


@_checked
def test_the_summary_a_technician_reads():
    print("=== the summary a technician reads ===")
    p = plan_scan(step_m=0.0005, velocity_mps=0.050, force=True, **PLATE)
    text = format_scan_plan(p)
    check("shows both velocities when clamped",
          "50.0 mm/s REQUESTED -> 5.0 mm/s USED" in text, True)
    print()
    print(text)

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("all scan-plan checks passed")


_SECTIONS = ["test_the_step_size_velocity_table", "test_velocity_is_clamped_and_says_so", "test_far_over_the_ceiling_is_an_error", "test_the_measured_reference_scan_comes_back", "test_the_memory_column", "test_an_unrunnable_scan_is_refused_with_the_reason", "test_projection_defaults_follow_the_step", "test_the_spot_radius_follows_both_sample_spacings", "test_a_manual_spot_radius_is_judged_by_trace_count", "test_deferred_projection_raises_the_ceiling", "test_the_timer_raster_is_bound_by_the_digitizer", "test_velocity_scaling_is_a_measured_curve", "test_the_cells_own_ceiling_is_reported", "test_the_outcome_check", "test_the_summary_a_technician_reads"]


if __name__ == "__main__":
    sys.exit(_main())
