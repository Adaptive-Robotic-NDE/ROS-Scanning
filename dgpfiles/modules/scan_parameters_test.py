"""Check the scan-plan arithmetic against the cases the handoff table gives.

Runs on the host with nothing else present: scan_parameters.dpi is deliberately
free of ROS, spatialnde2 and the recording database, so the derivation that
decides whether a scan is runnable can be checked without a cell.
"""
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
auto_projection_pixels = ns["auto_projection_pixels"]
velocity_scaling_for = ns["velocity_scaling_for"]
velocity_from_scaling = ns["velocity_from_scaling"]
ScanLimitError = ns["ScanLimitError"]

# The cell as measured 2026-08-15 -- linear at ~1.55 m/s full scale to about
# 30 mm/s, then saturating near 53 mm/s. See ROBOT_VELOCITY_CALIBRATION.
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


# The bowed test plate as Noether actually planned it: 30 passes over a 0.30 m
# span, each 0.308 m long because the 30 mm bow makes the surface longer than
# its footprint. That is the geometry every measured number in the handoff was
# taken against, and 0.308/0.00508 is what produced its 61 traces per pass.
PLATE = dict(path_length_m=9.24, passes=30, num_samples=16000, sample_bytes=4,
             file_backed=True, velocity_calibration=CAL)


print("=== the handoff's step-size / velocity table, live projection ===")
# step size -> max velocity, at the measured 10 traces/s live ceiling.
for step_mm, want_vmax_mm in ((5.08, 50.8), (1.0, 10.0), (0.5, 5.0), (0.1, 1.0)):
    p = plan_scan(step_m=step_mm*1e-3, velocity_mps=1.0, force=True, **PLATE)
    close("%.2f mm step -> %.1f mm/s ceiling" % (step_mm, want_vmax_mm),
          p["max_velocity_mps"]*1e3, want_vmax_mm, 1e-6)

print()
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
print("=== 10x over the ceiling is an error, not a clamp ===")
try:
    plan_scan(step_m=0.0001, velocity_mps=0.050, **PLATE)
    check("0.1 mm step at 50 mm/s raises", False, True)
except ScanLimitError as e:
    check("0.1 mm step at 50 mm/s raises", True, True)
    check("the message names the pipeline ceiling",
          "1.0 mm/s this pipeline can sustain" in str(e), True)

print()
print("=== the measured 5.5 min reference scan comes back out ===")
# 30 passes of 0.31 m at 50.8 mm/s, plus 5 s/pass of overhead: the run that was
# actually measured at 329.7 s.
p = plan_scan(step_m=0.00508, velocity_mps=0.0508, force=True, **PLATE)
close("estimated scan time within 10%% of the measured 329.7 s",
      p["scan_time_s"], 329.7, 33.0)
check("1830 traces, as the 30-pass run staged", p["traces"], 1830)
# The handoff quotes 117 MB decimal; this reports MiB, as volume_size_note()
# already does. Same 117,120,000 bytes.
close("trace memory 117.1 MB / 111.7 MiB", p["memory_bytes"]/1e6, 117.1, 0.2)

print()
print("=== the memory column of the same table ===")
for step_mm, want_mb in ((5.08, 117.0), (1.0, 590.0), (0.5, 1180.0)):
    p = plan_scan(step_m=step_mm*1e-3, velocity_mps=0.001, force=True, **PLATE)
    close("%.2f mm step -> %.0f MB" % (step_mm, want_mb),
          p["memory_bytes"]/1e6, want_mb, 0.03*want_mb)

print()
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
print("=== projection defaults follow the step ===")
close("8 mm spot for the 5.08 mm step that measured 96.6% coverage",
      auto_spot_radius(0.00508)*1e3, 8.13, 0.2)
check("512 px is enough for that spot over 0.30 m",
      auto_projection_pixels(0.008, 0.30), 512)
# A 0.5 mm step wants a 0.8 mm spot, which at 512 px over 0.30 m is 2.7 px --
# the handoff's "a 0.5 mm step needs 2048+".
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
print("=== the sample-and-bin raster is bound by the digitizer, not the stager ===")
p = plan_scan(step_m=0.00508, velocity_mps=0.050, trigger_mode="timer",
              digitizer_rate_hz=50.0, force=True, **PLATE)
close("50 Hz PRF / 3 samples per column", p["max_trace_rate_hz"], 50.0/3.0, 1e-9)
close("-> an 84.7 mm/s ceiling at the 5.08 mm column pitch",
      p["max_velocity_mps"]*1e3, 50.0/3.0*5.08, 1e-6)
check("so the 50 mm/s the raster has always run at is fine",
      p["velocity_clamped"], False)
# ... and this is the failure the config's own note warns about: sweep a row
# faster than the digitizer can fill its columns and _row_entries fabricates the
# empty ones from a neighbour.
p = plan_scan(step_m=0.00508, velocity_mps=0.150, trigger_mode="timer",
              digitizer_rate_hz=50.0, force=True, **PLATE)
check("150 mm/s is clamped", p["velocity_clamped"], True)
check("and it is the digitizer that said so",
      "digitizer" in p["rate_limited_by"], True)

print()
print("=== velocity scaling is a measured curve, not one ratio ===")
# Every measured point must come back exactly, in both directions.
for s, v in CAL:
    close("scaling %.4f -> %.1f mm/s" % (s, v*1e3),
          velocity_from_scaling(s, CAL)*1e3, v*1e3, 1e-9)
    close("%.1f mm/s -> scaling %.4f" % (v*1e3, s),
          velocity_scaling_for(v, CAL), s, 1e-9)
# The failure that motivated the table: one ratio taken at the top of the range
# over-commands everything slower. 10 mm/s at the old constant was scaling
# 0.0100, which this cell actually runs at 15.6 mm/s.
close("the old single-constant scaling for 10 mm/s", 0.010, 0.010, 0.0)
close("...which this cell really runs at", velocity_from_scaling(0.010, CAL)*1e3,
      15.6, 1e-9)
close("the table asks for 0.0064 instead", velocity_scaling_for(0.010, CAL),
      0.0064, 0.0002)
close("and that does produce 10 mm/s",
      velocity_from_scaling(velocity_scaling_for(0.010, CAL), CAL)*1e3, 10.0, 1e-6)

print()
print("=== the cell's own ceiling is reported, not silently exceeded ===")
p = plan_scan(step_m=0.010, velocity_mps=0.100, force=True, **PLATE)
close("100 mm/s comes back as the 52.9 mm/s the cell tops out at",
      p["effective_velocity_mps"]*1e3, 52.9, 1e-6)
check("and says so", any("tops out" in w for w in p["warnings"]), True)
p = plan_scan(step_m=0.0005, velocity_mps=0.005, force=True, **PLATE)
check("below the calibrated range is flagged as extrapolation",
      any("extrapolated through the origin" in w for w in p["warnings"]), True)

print()
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
      any("CALIBRATION is wrong" in l for l in lines), True)
check("and it hands back the table entry to add",
      any("Replace or add the entry (%.4f," % p["velocity_scaling"] in l
          for l in lines), True)
check("a queue of 40 is called out as backpressure",
      any("ran BEHIND" in l for l in lines), True)

print()
print("=== the summary a technician reads ===")
p = plan_scan(step_m=0.0005, velocity_mps=0.050, force=True, **PLATE)
text = format_scan_plan(p)
check("shows both velocities when clamped",
      "50.0 mm/s REQUESTED -> 5.0 mm/s USED" in text, True)
print()
print(text)

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("all scan-plan checks passed")
