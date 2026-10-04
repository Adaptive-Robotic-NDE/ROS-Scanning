# Simulating the eddy-current probe from a response table

How the generated response table reaches the simulated instrument, what each
piece is responsible for, and what the result does and does not mean.

The short version:

```bash
# the table already exists -- it lives in the test_data repository
ls ../test_data/ec_response/        # al6061_2mhz_response.h5 + _baseline.npz

# once per plate, anywhere
python3 tools/make_ec_test_plate.py --out-dir ~/ecplate

# every run
EC_SOURCE=sim EC_TABLE=../test_data/ec_response/al6061_2mhz_response.h5 \
    EC_PLATE=~/ecplate/flaws.json \
    dataguzzler-python dgpfiles/ECAutomatedScan.dgp
# then, at the prompt:
place_ec_test_plate()
```

To regenerate it (a night across two machines, wherever ecsim is installed):

```bash
python3 tools/generate_ec_table.py --out-dir ~/ectable --shard 0 --of 2 --yes
python3 tools/generate_ec_table.py --out-dir ~/merged --merge a/*.h5 b/*.h5
```

`--merge` carries the baseline `.npz` across with the table. **Keep the merged
pair and throw the shards away**: each shard file is full-shape, so four shards
are four times the size of the result for nothing.

With no table the simulated instrument runs its old synthetic signal and says
so on startup. Nothing else about the pipeline changes either way: the
instrument is still two free-running channels of volts published the way
`ai_voltage` publishes real ones, and the trigger, the trace accumulator, the
conditioning and the projection cannot tell the difference.

---

## Why bother — what the synthetic signal could not do

The old simulated source put lift-off and the flaw response along two fixed,
invented directions in the impedance plane, 74° apart, and said so in its own
comments. That is enough to exercise the architecture, which is what it was
for, and it is useless for the question we actually have:

**tilt about the probe's split line does not add noise to the flaw channel; it
produces a signal of the same kind.** One D rises, the other falls, and that
asymmetry is precisely what a differential probe exists to detect. Separating
tilt from a flaw is the whole problem, and a model that puts them on different
axes by construction makes the problem disappear.

With the table, **1.37° of tilt about the split line reads the same as the
largest notch in the set** — and the mesh contributes more than that on its own
(measured normal error on the ROI ran to a p99.9 of 1.88° before repair, and
`NORMAL_REPAIR_TOL_DEG` is 3.0°). That ordering is asserted by tests, because if
a change ever reversed it then every conclusion drawn in simulation about
separating the two would be worthless.

**An earlier draft of this page said "more than ten times", which was a stub
number.** The real ratio at 5° is 3.6×; see the measured table below. The
ordering is the robust claim and the tests assert that rather than a figure
that a fitted coil geometry will move.

---

## The pieces

| | |
|---|---|
| `tools/generate_ec_table.py` | Runs against **ecsim**, elsewhere. Writes `al6061_2mhz_response.h5` and `al6061_2mhz_baseline.npz`. Not a dependency of anything here. |
| `tools/make_ec_stub_table.py` | A schema-conformant **stub** in seconds, with invented physics. For working on the pipeline before the real table exists. |
| `tools/make_ec_test_plate.py` | The plate: meshes in world coordinates, flaws in plate coordinates. |
| `dgpfiles/modules/ec_response_table.py` | Reads the table. h5py and numpy only. |
| `dgpfiles/modules/ec_plate_model.py` | Pose in, volts out. numpy only. |
| `dgpfiles/modules/sim_ec_source.dpi` | Publishes the two channels, from the model or from the synthetic signal. |

`ecsim` is **not** installed in the container and is not meant to be — it is a
night's compute on a workstation, and all the container does is read the file it
produced. `python3-h5py` is in the image for that; without it the synthetic
signal still runs.

---

## The conventions, and where each one came from

These are the places a sign error hides, so each is recorded with how it was
settled rather than asserted.

### The lift-off axis is standoff above edge contact

Not absolute height. A rigid probe face cannot be tilted while its centre stays
down — tilting by θ about the contact edge lifts the centre by
`footprint_radius·sin θ`, which is 0.56 mm at 10° for this probe — so the
generator handed the solver `edge_contact_liftoff(tilt) + axis value` and the
axis is what the surface adds on top of that geometric minimum.

That matches our probe exactly: it is spring-loaded and in contact, so it rides
on its edge at any tilt and mesh error adds standoff. The model therefore
computes

```
standoff = max(0, z_tip − z_surface)
```

The `max` is the spring. The arm commands the tip past the measured surface by
the contact preload; the spring takes up the difference and the wear face rides
on the part. A measured 0.45 mm of preload against a perfect mesh is **zero**
standoff, not −0.45 mm — which the table would refuse, its lift-off axis
starting at zero.

The parameter is called `standoff` everywhere in this code and never
`liftoff`, so the two cannot be confused at a call site.

### The two tilts

ecsim built the table with `R = Rz(yaw)·Rx(tilt_x)·Ry(tilt_y)`, and

- **`tilt_parallel`** = ecsim's `tilt_y`, about the **split line**. Raises one D
  and lowers the other. **Differential, flaw-like. This is the one that hurts.**
- **`tilt_perpendicular`** = ecsim's `tilt_x`, about the line joining the coil
  centres. Common-mode.

With the surface's outward normal written in probe coordinates as
`n = (nx, ny, nz)`, that convention gives `n = (−cos a·sin b, sin a, cos a·cos b)`
for `a = tilt_x`, `b = tilt_y`, hence

```
tilt_parallel      = atan2(−nx, nz)
tilt_perpendicular = atan2(ny, hypot(nx, nz))
```

Exact for any yaw and tilt. `test_tilt_decomposition_matches_ecsim` checks it
against `ecsim.geometry.probe_pose()` over 500 random poses — worst error
**6e-17 rad** — rather than against any algebra written here. A wrong sign on
the parallel tilt would invert every flaw signature while leaving every
amplitude plausible, so this is the test the whole model rests on.

### Where the flaw is, in whose frame

`dx`, `dy` are the offset of the **probe-face centre from the flaw centre**,
expressed in the probe frame, x along the coil-separation axis. That is the
sense `ecsim.scan.scan_positions()` tabulates. Reversing it mirrors every
signature — and the signature being double-lobed and antisymmetric, mirroring
it means **inverting** it.

`yaw` is the angle from the probe's x axis to the flaw's long axis, measured by
projecting both onto the probe frame. Exact at zero tilt; it couples with tilt
up to about 3° at 15° of tilt, which is well inside the table's 30° yaw
spacing, and that residual is a genuine geometric ambiguity rather than an
error.

Yaw is reduced **modulo π**, because a notch is a line *segment*: a flaw at yaw
and at yaw+180° are the same flaw. The real table does not declare
`yaw_period_rad`, so without this a query at 3.3 rad — an ordinary orientation
on the plate — would be refused outright. **Worth adding to the generator's
provenance** so the file is self-describing; the reader does not need it.

### The probe frame itself: one measured, one unknown

- **`PROBE_AXIS_SIGN`**, which way `probe_end`'s +z points. Read off
  `tari_ur_description/urdf/ec/tooling_macro.xacro`: the `ec_probe-probe_end`
  joint carries `rpy="pi 0 0"`, flipping z relative to the shaft, so
  `probe_end` +z runs back **up** the shaft, away from the surface — which is
  also ecsim's convention. This is checkable at runtime (on the part, the probe
  axis must point away from it), so it is **checked on every query and
  counted**, not trusted. Both settings give a right-handed frame, so a wrong
  sign points the wrong way without also turning the frame inside out.
- **`COIL_AXIS_DEG`**, the angle from `probe_end`'s x axis to the
  coil-separation axis. **NOT KNOWN.** It is a pencil probe in a holder and its
  rotation about its own axis was never recorded. Getting it wrong rotates
  every notch's apparent yaw by the same amount and swaps which tilt is the
  dangerous one. **Measuring it is the first thing to do at the cell**, and
  until then it is a declared constant set to zero rather than a quiet
  assumption.

---

## The two nulls, which are not the same null

Both exist on the real instrument and neither replaces the other. The model
reproduces both because collapsing them is the obvious shortcut and it would
make either one impossible to exercise.

| | removes | where | how |
|---|---|---|---|
| **hardware** | the operating point, in **ohms**, ahead of the gain stage | the instrument's bridge | `null_ec_instrument()` → `ec_model.null()` |
| **software** | the analog chain's **voltage** offset | `/ec/cond/null` | `null_here()` |

**On the real table the hardware null has nothing to remove until the probe is
tilted**, because this model is a *perfectly balanced* differential probe: square
on, its flaw-free reading is zero at every lift-off (see the measurements below).
What it removes is the TILT trajectory — and since the arm holds whatever tilt
the mesh gives it for a whole pass, that is a standing offset in practice even
though it is not one in principle.

That also means the suggested gain is usable without nulling here, where on the
real instrument it would not be: a real probe has a residual imbalance that this
one has none of. The code is unchanged either way — the suggestion is reported,
never adopted — but do not carry the simulated gain to the bench.

`ec_model.null()` takes **no arguments**, matching `UniWest.null()`, so the
scan reaches it through the same `ec_first_contact_null(null_fn)` path at the
first touchdown. It can therefore be nulled in mid-air exactly as on the bench,
and it **reports that the probe was off the part** when it was.

---

## What is reproduced on purpose

- **The digitiser rails.** ±5 V, and output is clipped at them and counted. A
  real pass has hit them: in the measured tilt ladder the +6° pass sat at
  exactly −5.0000 for all 64 samples with a standard deviation of exactly zero,
  which made it look like the *quietest* pass in the set.
  `ec_liftoff.saturation()` exists to catch that and can only be exercised
  against a simulator that can clip.
- **Lift-off from reconstruction error.** The toolpath is planned from the
  *reconstructed* mesh and the plate's true surface is flat at a known z, so
  mesh error becomes standoff with nothing added by hand. This is the
  mechanism, not a model of it.
- **Running off the part.** Off the plate's footprint the reading is zero.
  Between passes the arm lifts clear and runs off the end, and a model that
  answered those poses with a full-amplitude tilt signal would fill the trace
  store with numbers that look like data.
- **Leaving the part vertically.** Past the table's last lift-off node the
  response tapers to zero on a decay length **recovered from the table's own
  baseline**, not chosen here. Clamping would hold a full-amplitude tilt signal
  in mid-air.
- **Per-sample noise.** A block is many samples of one pose with independent
  noise, so `EC_BLOCK_REDUCE`'s `"mean"` has something to average and
  `report_smear()` means something.

## What out-of-domain does, and why it never raises

On the acquisition thread an exception is a lost block, and a run of lost blocks
is a silent flatline that reads as an instrument that stopped. So the reader
**clamps and counts** instead, and `report()` says how far outside the model the
run went:

- `outside_field` — a flaw further away than the stored ±6 mm field. **Normal**,
  and answered with zero: most flaws are out of reach most of the time, and the
  field's edge is deliberately clean baseline.
- `clamped` — per axis, a pose the table cannot represent, answered with its
  nearest edge.
- `extrapolated` / `beyond_extrapolation` — standoff past the last node.

A model that *raises* does not stop the instrument either: the block falls back
to the synthetic signal and `model_errors` counts it, with the traceback
**formatted** rather than retained — holding the exception object would pin its
traceback, the frame in it, and the active transaction in that frame, which then
never ends and blocks every globalrev wait in the process. That bug cost 59
passes once.

`model_no_pose` is counted separately from `model_errors`, because they want
opposite responses: no pose resolves itself as soon as the cell comes up, and a
nonzero count *after* a finished scan means the model was never consulted at
all.

---

## What the numbers mean

**Shapes are meaningful. Absolute amplitudes are not.** The probe's coil
geometry behind the table is a documented placeholder chosen to be plausible
inside a 0.187″ shaft — not a fit and not a measurement. `ecsim/scripts/fit_d20.py`
fits it against the VIC-3D table and the same approach applies to our own
measured tilt ladder once there is an unclipped one with the probe's rotation
recorded. Change those numbers, rerun, and the table's provenance records which
was used.

A **stub** table is worse than that: its physics is invented outright. It is
flagged in the file as `stub = True`, so it is told from the file and not from
its name — a renamed stub is otherwise indistinguishable from the real thing —
and the startup banner, `describe()`, `report()` and the per-block metadata all
repeat it.

Every published block carries `EC-sim-source` (`"table-model"` or
`"synthetic"`), `EC-sim-table` and `EC-sim-table-stub`. A dataset whose traces
came from the synthetic bumps and one whose traces came from the table are not
comparable, and nothing else in the file tells them apart.

---

## The plate

180 × 120 × 12 mm aluminium, 12 notches: four lengths (the UniWest 99073
standard's 0.040–0.100″) at each of two depths, plus one size at four
orientations to exercise the yaw axis. Every dimension sits **exactly on a
table node**, so no interpolation in flaw geometry is needed and any error seen
is the pipeline's.

**The notches are not in the mesh**, deliberately. A 0.15 mm slot is below what
the Keyence resolves and the 4.75 mm probe bridges it rather than dropping in,
so modelling them geometrically would both mislead the toolpath planner and
give the simulated reconstruction a feature the real one could not see. The
plate is smooth; the flaws live in `flaws.json`. Geometry and flaws are
separate, exactly as on a real part.

The **meshes are in world coordinates** and the **flaw list is not**.
`place_gazebo_mesh()` writes a model with an identity pose so that nothing has
to be kept in step between the Gazebo model and the simulated profiler's own
scene; the flaws, being a property of the plate rather than of one bench, stay
in the plate frame with the placement recorded beside them.
`place_ec_test_plate()` refuses to spawn a mesh from one directory while the
instrument reads flaws from another — that mismatch puts every notch in the
wrong place and comes out looking like a projection fault.

`--place x y z` and `--yaw-deg` move it.

## The plate has to reach three places

They do not talk to each other, and a plate in two of them is worse than a
plate in none:

| | sees | written by |
|---|---|---|
| the Gazebo model | what the simulated camera renders | `place_gazebo_mesh()` |
| the Keyence scene | what the simulated profiler raycasts | `write_keyence_sim_scene()` |
| `ec_model` | where the flaws are | built from `EC_PLATE` at startup |

`place_ec_test_plate()` does the first two and checks the third.

---

## Testing it

Everything runs on the host with no ROS, no cell, no arm and no spatialnde2,
against a stub table built in the test's own temporary directory:

```bash
cd dgpfiles/modules
python3 -m pytest -q ec_response_table_test.py ec_plate_model_test.py
python3 -m pytest -q sim_ec_model_test.py      # this one needs spatialnde2
```

Two tests are worth more than the rest, and both compare against **ecsim
itself** where a checkout is present (`ECSIM_DIR`, default `~/ecsim`):

- `test_agrees_with_ecsim_reader` — same file, same query, same answer.
  `ec_response_table.py` exists because ecsim's reader is too slow for an
  acquisition loop, but ecsim's is the authority on the format, so a
  disagreement means ours is wrong.
- `test_tilt_decomposition_matches_ecsim` — described above.

`test_schema_constants_match_ecsim` makes the duplicated schema names safe: the
container has h5py and will never have ecsim, so the names are copied, and a
copy nobody checks is how two readers end up disagreeing about which axis is
which.

## Why there is a second reader at all

`ecsim.table.ResponseTable` re-reads and re-interpolates the whole 2-D field on
every call and then samples one point from it. The real table's field is 49×49
and the pose interpolation touches 2⁷ = 128 of them, so one point query reads
307 kB and does 128 field multiplies — and the instrument publishes 50 blocks a
second, summing over every flaw within reach. Here the flaw-geometry axes are
collapsed **once per flaw** into a cached pose block, and a query is then 8 pose
corners × 4 spatial corners of scalar arithmetic. The cache is bounded and
evicts.

---

## Measured from the real table, 2026-10-04

`test_data/ec_response/`, 4200 of 4200 poses written, 81 MB, aluminium 6061-T6
at 2 MHz, single-precision solve. All of the following is read off that file and
is asserted by `ec_real_table_test.py`, which skips when the fixture is absent.

**Flaw amplitude is ordered by notch length and by depth** — the acceptance
check the spec asked for, at yaw 90°, zero standoff, no tilt:

| peak \|dZ\| (Ω) | 1.016 mm | 1.524 mm | 2.032 mm | 2.540 mm |
|---|---|---|---|---|
| 0.20 mm deep | 0.619 | 0.766 | 1.048 | 1.305 |
| 0.50 mm deep | 0.756 | 0.912 | 1.175 | 1.587 |

Monotone in both directions, across all eight. The 0.040″ notch is **not** near
a detection floor, which the spec expected it to be — at 0.62 Ω it is two
thirds of the 0.100″ one.

**The flaw-free reading, |Z| in ohms** (tilt⊥ = 0):

| standoff | −10° | −5° | 0° | +5° | +10° |
|---|---|---|---|---|---|
| 0.00 mm | 6.030 | 6.685 | **0.000** | 6.685 | 6.030 |
| 0.10 mm | 4.597 | 4.847 | **0.000** | 4.847 | 4.597 |
| 0.25 mm | 3.157 | 3.124 | **0.000** | 3.124 | 3.157 |
| 0.50 mm | 1.804 | 1.643 | **0.000** | 1.643 | 1.804 |
| 1.00 mm | 0.709 | 0.574 | **0.000** | 0.574 | 0.709 |

Four things in that table, all of which the consumer depends on:

- **Exactly odd in tilt∥.** `Z(−10°) + Z(+10°) = 0` to the solver's precision.
  The sign of the tilt is the sign of the reading, which is what makes it
  flaw-like.
- **Exactly zero in tilt⊥.** Not "much less" — *nothing*. Perfect common-mode
  rejection.
- **Exactly zero square-on, at every lift-off.** So **lift-off alone produces no
  signal.** This is the one real surprise and it has consequences: `set_rotation()`
  has no *lift-off* direction to find in simulation — the trajectory it would
  rotate onto an axis is the tilt one.
- **Not monotone in tilt**: 6.685 Ω at 5° but 6.030 Ω at 10°, because
  `edge_contact_liftoff` lifts the probe centre 0.56 mm at 10° and both coils
  retreat. That is the geometry working as documented, not an artefact.

**Tilt against flaw:** 1.337 Ω/deg near zero, so the 1.834 Ω largest notch
(yaw 0°) is matched at **1.37°**, and 5° of tilt is **3.6×** the notch.

**Yaw matters less than the spec expected.** Peak |dZ| for the largest notch
runs 1.834 Ω at yaw 0°, 1.486 Ω at 60°, 1.587 Ω at 90° — a 1.23× spread, with
the *maximum* at 0° rather than the minimum. The spec anticipated a flaw aligned
with the coil axis giving "nearly nothing". It does not, at least not in peak
amplitude; whatever yaw information there is lives in the shape, not the size.
Worth knowing before leaning on the yaw axis. (The 2.54 mm notch is also long
against the 1.53 mm coil separation, which would blunt orientation selectivity.)

**Yaw period π confirmed in the data**: yaw 0° ≡ 180°, 30° ≡ 150°, 60° ≡ 120°,
to 1e-5 relative. The reader's modulo-π reduction is therefore sound and not an
assumption.

**The stand-off decay fit gives 0.535 mm** from the real baseline's top two
lift-off nodes. This is the number that caught a real bug: the fit originally
sampled zero tilt, where this probe's flaw-free reading is *identically zero*,
so it was the ratio of two zeros and silently fell back to a constant. It
samples the largest parallel tilt instead. **A stub could not have found that**,
because a stub chooses its own baseline.

**A pass across the 0.100″ × 0.020″ notch**, zero standoff, square on:

| dx | dZ (Ω) |
|---|---|
| −3.0 mm | +0.0007 +0.0033j |
| −2.0 mm | −0.0059 +0.1649j |
| −1.0 mm | −0.6793 +1.0041j |
| 0.0 mm | **0 (the null)** |
| +1.0 mm | +0.6793 −1.0041j |
| +2.0 mm | +0.0059 −0.1649j |
| +3.0 mm | −0.0007 −0.0033j |

Textbook: exactly antisymmetric, hard null on the notch, gone by ±3 mm.

**And a warning in that table.** The lobes peak at **±1 mm** and the response is
essentially over by ±2 mm. Our trace pitch is **1 mm**, so a pass puts roughly
one trace on each lobe and one in the null, with no margin — and that is for the
*largest* notch. `test_the_response_is_narrower_than_the_trace_pitch_is_coarse`
exists to keep that number visible. The spec's claim that a 0.1 mm field step
was "10× oversampled" against our pitch was about the FIELD, not about the
response: against the response the pitch is marginal, and `SCAN_STEP_SIZE` is
worth revisiting.

## Verified in the container, 2026-10-04

The config loaded with the stub table, and with a pose placed on the `L3_D0`
notch the two published channels read **(1.3312, −0.1756) V** against a
hand-computed **(1.332, −0.174)**. Sweeping across the notch, the published
`ai1` channel traced the double lobe — −0.1913 V at −2 mm, −0.1750 V on the
notch, −0.1592 V at +2 mm, returning toward baseline by ±4 mm: antisymmetric
about the centre, through the whole chain from pose to recording database.
`null_ec_instrument()` captured 1.802+1.262j Ω on the part and the channels then
read the bare instrument offset, (1.2006, −0.3512) V.

Before a pose existed, `model_no_pose` counted every block and the instrument
kept publishing the synthetic fallback — which is the counter doing exactly its
job: it is how you tell a run that never consulted the model from one that did.

## Open

- **`COIL_AXIS_DEG` is unmeasured.** First thing to settle at the cell.
- **The coil geometry is a placeholder, not a fit.** Needs an unclipped tilt
  ladder with the probe's own rotation recorded.
- **`yaw_period_rad` is not in the real table's provenance.** Handled in the
  reader; the file would be better self-describing.
- **The trace pitch is marginal against the real response width** (lobes at
  ±1 mm, pitch 1 mm). Worth a simulated pitch study now that one is possible.
- **`test_data/` is not mounted into the ros2-workstation container**, so a
  simulated scan there cannot read the table from where it lives. Either mount
  it read-only or copy the pair into `scandata/`, which is mounted.
