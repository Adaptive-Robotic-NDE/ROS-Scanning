# Specification: an eddy-current probe response model for the TARI scanning cell

Written 2026-10-03. This is a specification for a tool to be built **outside**
this repository, whose output this pipeline will consume. It is self-contained:
nothing here assumes familiarity with the scanning stack.

The short version: we need **complex probe response as a function of flaw
geometry, probe pose relative to the flaw, and lift-off/tilt**, as a
pre-computed table we can interpolate at scan time. An existing model of this
shape is described in the appendix; it is close to right and wrong in ways that
matter, and this document is mostly about those.

---

## 1. The hardware this must represent

| | |
|---|---|
| probe | **2 MHz differential shielded split-D pencil probe**, 0.187″ (4.75 mm) shaft |
| material | **aluminium** (assume 6061-T6 unless told otherwise, ~2.5e7 S/m, 43% IACS) |
| frequency | **2.000 MHz** |
| skin depth | **0.071 mm** at the above |
| calibration standard | UniWest P/N 99073 aluminium radius standard |
| scan trace pitch | **1 mm**, triggered on measured chord progress |

The calibration standard carries surface notches of **0.040″, 0.060″, 0.080″ and
0.100″ length** (1.016, 1.524, 2.032, 2.540 mm) on the flat, on the corners, and
along four radii (0.500″, 0.375″, 0.250″, 0.125″). Depth and width are not
published. Notches exist in **two orientations**, which is why probe yaw matters
as much as it does.

---

## 2. The physics that must be in the model

### 2.1 It is a DIFFERENTIAL probe, and that changes everything

Two coils side by side; the output is their **difference**. Three consequences,
and the model is not useful unless it has all three:

**A flaw produces a double-lobed signature, not a bump.** As the probe passes
over a flaw, the leading coil sees it first, then both, then the trailing coil:
the response goes positive, through a null, to negative. In the impedance plane
the trace is a **loop**, and the loop's shape is the information that separates a
crack from a pit and deep from shallow. A single-lobed magnitude model is not a
differential probe and will not let us test anything downstream.

**Uniform lift-off is largely common-mode.** Both coils move together and the
difference cancels much of it. This is the probe's selling point and it must be
reproduced, because it is *why* the next point is dangerous.

**Tilt is the blind spot, and it mimics a flaw.** Tilting the probe about the
axis separating the two coils brings one coil closer than the other — precisely
the asymmetry the probe exists to detect. **Tilt does not merely add noise to the
flaw channel; it produces a signal of the same kind.** This is the single most
important thing for the model to get right, because separating the two is the
entire problem we are trying to solve.

### 2.2 Therefore: TWO tilt axes, not one

Tilt must be parameterised as two independent angles:

**Define them by what they DO, because the names alone are ambiguous and an
earlier draft of this section got the geometry backwards.** Put the two coils on
the probe-frame x axis, so the split line between them runs along y:

- **`tilt_parallel`** — rotation about the **split line** (y). One D rises and
  the other falls, so the two coils see different lift-offs. **Differential,
  flaw-like. This is the one that hurts.**
- **`tilt_perpendicular`** — rotation about the **line joining the coil
  centres** (x). Both coils lie on that axis, so neither changes height
  relative to the other. **Common-mode**, and should behave much more like
  uniform lift-off.

The unambiguous test, which overrides any wording above: **if a tilt raises one
coil and lowers the other it is `tilt_parallel`; if it moves both the same way
it is `tilt_perpendicular`.** An implementation that disagrees with a sentence
here should follow the physics and record the mapping in the table provenance.

A single lumped "tilt" parameter conflates two physically different effects and
the model will not be able to reproduce what we measure. Our cell commands tilt
about the **travel** direction, which aligns with the coil axis or not depending
on probe yaw — and rotating the end-of-arm tooling by 90° to reach the
perpendicular notches swaps which case we are in.

### 2.3 Probe yaw runs 0–180°, not 0–90°

A differential probe's response is **antisymmetric** about the coil axis, so the
two halves of the circle are not redundant. A flaw at +30° and one at −30° give
responses of opposite sign, and a flaw aligned with the coil axis gives nearly
nothing while one across it gives maximum.

---

## 3. Output format

Keep the overall shape of the existing model — **an N-dimensional parameter grid
whose entry is a 2-D complex field in probe coordinates** — and change the
following.

### 3.1 File

**HDF5** (`.h5`), chunked, one file. Not MATLAB `.mat`: it needs `scipy.io`, it
cannot be read lazily, and it carries no schema. `.npz` is acceptable for a
small table but does not chunk.

### 3.2 Arrays

```
/response        complex64   (n_depth, n_length, n_width, n_yaw, n_y, n_x)   [see §5 on extra axes]
/axes/depth      float64     (n_depth,)
/axes/length     float64     (n_length,)
/axes/width      float64     (n_width,)
/axes/yaw        float64     (n_yaw,)
/axes/y          float64     (n_y,)
/axes/x          float64     (n_x,)
```

- **One complex array**, not separate real and imaginary. Halves the bookkeeping
  and removes any possibility of pairing them wrongly.
- **`complex64` is enough.** The response does not carry seven significant
  figures and the size matters (§5).
- **Axis order: parameters first, spatial last**, so a single flaw's field is
  contiguous.

### 3.3 Units — SI, and declared in the file

**Every axis in metres or radians**, with the unit as an HDF5 attribute on the
dataset:

```
/axes/length.attrs["units"] = "m"
/axes/yaw.attrs["units"]    = "rad"
```

The existing model is in **mils**, with the units recorded only in a docstring.
That is a standing source of error for us: this pipeline is metres throughout,
and a silent 25.4× is the kind of mistake that survives review.

A reader must be able to validate a query against the table **by name and unit**
and refuse an out-of-domain one, rather than silently extrapolating.

### 3.4 Provenance — in the file, not in the filename

Root attributes, all required:

```
frequency_hz, material, conductivity_s_per_m, permeability_relative,
probe_type, probe_shaft_diameter_m, coil_diameter_m, coil_separation_m,
shielded (bool), generator, generator_version, created_utc, notes
```

The existing table's entire identity is its filename
(`VIClookup_D20_2MHz_Ni_Cr_V439AFIX.mat`), which means a renamed copy is
unattributable and a mismatched one is undetectable.

### 3.5 Spatial field

- **Extent ±5 mm** in both axes about the flaw centre. The existing ±2.03 mm is
  too small for this probe: the shaft alone is 4.75 mm, the response extends
  several millimetres, and we need clean baseline on both sides of it to measure
  amplitude.
- **Spacing 0.1 mm**, giving 101×101. Our trace pitch is 1 mm, so this is 10×
  oversampled — ample, and it leaves room to study pitch choice in simulation.
- **Probe-centred coordinates**, x along the scan direction, y across it, with
  the convention stated in the file's notes.

---

## 4. Parameter ranges

Every range below is justified by a measurement or by the standard we calibrate
against. Node spacing may be **non-uniform** and should be: put resolution where
the response moves fastest.

| axis | range | nodes | why |
|---|---|---|---|
| depth | 0 – 1.0 mm | ~8, fine below 0.2 mm | skin depth is 0.071 mm, so past ~0.5 mm the response saturates and the axis can go coarse. Resolution is needed in the first few skin depths |
| length | 0 – 3.0 mm | ~10 | the standard's notches are 1.016–2.540 mm; the existing table stops at 2.032 mm and cannot represent the largest |
| width | 0 – 0.30 mm | ~5 | EDM notches run 0.004–0.010″ (0.10–0.25 mm); the existing table stops at 0.127 mm |
| yaw | 0 – 180° | ~13 (15°) | antisymmetric response, see §2.3 |
| **lift-off** | 0 – 1.0 mm, fine near 0 | ~6 | the probe is spring-loaded; measured contact force 4.5 N at a measured series stiffness of 10 N/mm is 0.45 mm of compression, and mesh error is ±0.5 mm |
| **tilt_parallel** | −10° – +10° | ~7 | our tilt ladder runs ±6°; mesh-normal error contributes ≤1.9° after repair |
| **tilt_perpendicular** | −10° – +10° | ~5 | as above, expected to matter less |

---

## 5. THE SIZE PROBLEM — read this before generating anything

A dense grid over all of §4 is not feasible:

| | size, `complex64` |
|---|---|
| flaw axes only (8×10×5×13) × 101×101 | **424 MB** |
| × lift-off (6) × tilt∥ (7) × tilt⊥ (5) = 210× | **89 GB** |

### The test to run first

Classic eddy-current lift-off attenuates the response roughly exponentially and
rotates its phase, **largely independently of the flaw's geometry**. If that
holds here:

```
response(flaw, liftoff, tilt, x, y)  ≈  F(flaw, x, y) · G(liftoff, tilt∥, tilt⊥)
```

with `G` complex scalar, then the table is **424 MB plus 210 numbers** instead of
89 GB.

**Please test this before building a product grid.** Generate the full response
at two or three flaw geometries across the lift-off/tilt range, fit `G`, and
report the residual. A few percent is a complete win. If it fails, it will
probably fail in a structured way — e.g. `G` depending on flaw *length* but not
depth or width — and a partial factorisation is still worth most of the saving.

If it genuinely does not factorise, say so with the numbers and we will reduce
the spatial grid or the flaw axes instead. **Do not silently produce an 89 GB
file, and do not silently drop an axis to avoid one.**

---

## 6. What the values must mean

- **The flaw-induced CHANGE in impedance**, not an absolute reading. We add the
  instrument's own offset ourselves; removing it is what nulling does.
- **No gain and no rotation applied.** Our instrument applies both in hardware,
  and choosing them is precisely what we want to study in simulation. A model
  with gain baked in cannot be used for that.
- **Physical units**, stated in the provenance attributes — ohms, or normalised
  with the normalisation declared. Not instrument volts.
- **Lift-off response is wanted as part of the model**, not only as a modifier of
  the flaw response: with no flaw present, varying lift-off and tilt must still
  produce the correct complex trajectory. That trajectory is what the work cell's
  conditioning procedure rotates onto the horizontal axis, and it is half of what
  we need to simulate.

---

## 7. Validation — there is real data to check against

A measured aluminium dataset exists, taken with this exact probe at 2 MHz:
five passes over a taught 63 mm line at tilts **0, +3, −3, +6, −6 degrees**, with
a measured probe pose per trace, 64 traces per pass.

Two caveats on that specific take, both being fixed: the `+6°` pass is **clipped
at the ±5 V digitiser rail for all 64 samples** and `+3°` for 4 of 64, and there
is instrument drift of about **0.016 signal units/s**. A re-flown version at
corrected gain is expected shortly and will be the better reference.

**The acceptance test we care about is "reproduces the measured tilt ladder"** —
the direction and magnitude by which the per-pass mean impedance moves as tilt is
stepped, and the within-pass variation. That is a far stronger check than
anything internal to the generator.

Also worth reproducing, and cheap to check: a pass over each of the four
calibration notches at nominal lift-off should give **response amplitudes
ordered by notch length**, with the 0.040″ notch near the detection floor.

---

## 8. Appendix: the existing model, and why it is not enough

`Skmadigan/eddy-current-scan-analysis`, `andi_mesh_analysis`, Apache 2.0. A
tricubic-spline surrogate of VIC-3D output:
`sim_VIC3Dmodel(PInv, H2, LookUpTable)` interpolating
`x = [depth, length, width, 90 − angle]` and returning a complex field.

What it gets right, and is worth copying: the overall shape (parameter grid →
2-D complex field), interpolating the *field* rather than a scalar, and keeping
probe yaw as a first-class parameter.

Where it does not fit:

| | existing | needed |
|---|---|---|
| material | Ni-Cr | **aluminium** |
| probe | "D20" | **differential shielded split-D, 0.187″** |
| lift-off | **absent** | a parameter |
| tilt | **absent** | **two** parameters |
| length | 0–2.032 mm | to 3.0 mm |
| width | 0–0.127 mm | to 0.30 mm |
| yaw | 0–90° | 0–180° |
| field extent | ±2.03 mm | ±5 mm |
| units | mils, undeclared | SI, declared |
| format | `.mat`, split re/im | HDF5, complex |
| provenance | in the filename | in the file |

Frequency (2 MHz) matches exactly, which is the one thing that does.

---

## 9. How this will be consumed, for context

A module will expose roughly

```python
response(depth, length, width, yaw, liftoff, tilt_par, tilt_perp, dx, dy) -> complex
```

loading the table lazily, refusing out-of-domain queries by axis name, and
feeding the existing simulated eddy-current source in this repository. That
source already separates the flaw contribution from the lift-off contribution
and composes them in the impedance plane, so a model of this shape drops in.

What it replaces: a Gaussian bump along a **fixed, invented** direction in the
impedance plane. The current code says so itself — *"74 degrees between them is
in the range a real probe shows; it is not a measurement of anything."* Replacing
that with a real loop, whose shape varies with flaw geometry and lift-off, is the
whole point of this exercise.
