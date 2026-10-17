#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# DISTRIBUTION A. Approved for public release: distribution unlimited.  #
#########################################################################
# File:  norms_ec_plugin_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-12
#########################################################################
"""The NORMS metadata plug-in, exercised the way NORMS runs it.

AS A SUBPROCESS, always. Every way this plug-in can fail silently is a property
of its STDOUT -- a stray print, a warning from a library, a number where NORMS
wants a string, a '|' anywhere at all -- and none of those survive being tested
by importing the module and calling a function. NORMS starts the file, reads
stdout to EOF and parses it; so does this.

The failure modes being guarded, in NORMS' own vocabulary:

    Plug-in Not Defined   the executable bit is missing
    JSON Missing          no shebang, or CRLF in the shebang line
    JSON Invalid          anything on stdout that is not the one JSON object
    Plug-in Error         stdout empty and stderr written

and, quieter than any of those: a value NORMS reads as empty because it was a
number rather than a string, a property that appears on a real run but not on
the no-argument run (so it never reaches the documentation form), and a
generated file that is not declared (so it is never hashed or backed up).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.normpath(os.path.join(HERE, "..", "..", "norms", "ec_scan_plugin.py"))

# The sections EASEL owns. A plug-in property that lands in one of these
# REPLACES what the operator typed on the form, so the plug-in must not use
# them (SKILL.md section 3 rule 5).
EASEL_SECTIONS = {"Software", "Overview", "Data File Format", "Notes", "Sample",
                  "Probe", "Scan Classification", "Remote Backup", "JSON Parameters"}


def run(*args, **kw):
    """(stdout, stderr) from the plug-in, exactly as NORMS would collect them."""
    env = dict(os.environ)
    env.update(kw.pop("env", {}))
    p = subprocess.run([PLUGIN] + list(args), capture_output=True, text=True,
                       timeout=300, env=env, stdin=subprocess.DEVNULL)
    return p.stdout, p.stderr


def one_object(stdout):
    """Parse stdout the way NORMS does, and fail the way NORMS would."""
    assert stdout.strip(), "stdout was empty: NORMS records this as JSON Missing"
    assert stdout.lstrip()[0] == "{", \
        "stdout does not start with the JSON object: NORMS records JSON Invalid\n" + stdout[:400]
    return json.loads(stdout)          # a second object or trailing text raises here


def properties(doc):
    for section, body in doc.items():
        for name, field in body.get("Properties", {}).items():
            yield section, name, field


def make_scan(tmpdir, passes=12, per_pass=40, pitch=0.00025, spacing=0.00025,
              flaw_at=(6, 20), ragged=True, liftoff_v=0.30, sample_rate_hz=None,
              serpentine=False):
    """A synthetic ec_scan_* directory: passes along +y, stepping in -x.

    `ragged` shortens alternate passes, which is what a clearance crop or a hole
    does to a real scan and what the pass-spacing measurement has to survive.

    `liftoff_v` is how much lift-off the traces carry. Zero makes the cloud
    round, which is the 2026-10-16 case: there is then no principal direction
    and the angle between the axes is whatever the noise says.
    """
    rng = np.random.default_rng(4)
    scan = os.path.join(tmpdir, "ec_scan_2026-10-12_000000")
    os.makedirs(scan, exist_ok=True)
    for p in range(passes):
        n = per_pass - (7 if (ragged and p % 2) else 0)
        x = np.full(n, 0.85 - p*spacing)
        y = 0.10 + np.arange(n)*pitch
        # A REAL PASS WANDERS. This is what defeated the world x-y grid: a
        # sub-cell wander along the pass is enough to snap neighbouring traces
        # into one cell and leave the next empty.
        x = x + 0.4*pitch*np.sin(np.linspace(0.0, 3.0, n))
        if serpentine and p % 2:
            y = y[::-1]
        pos = np.c_[x, y, np.full(n, 0.94)]
        s_m = np.arange(n)*pitch                  # arc length from the start
        # lift-off along ~86 deg in the impedance plane, plus a local flaw
        t = rng.normal(0.0, float(liftoff_v), n)
        z = t*np.exp(1j*np.radians(86.0))
        if liftoff_v <= 0.0:
            # Round, not empty: a dead instrument and a flat part look the same
            # from here, and both must refuse rather than invent an angle.
            z = z + rng.normal(0.0, 0.02, n) + 1j*rng.normal(0.0, 0.02, n)
        if p == flaw_at[0] and flaw_at[1] < n:
            z[flaw_at[1]] += 0.9*np.exp(1j*np.radians(86.0 + 90.0))
        meta = {
            "pass_index": p, "traces": int(n), "wall_time": 1.7e9 + p*20.0,
            "pitch_m": pitch, "component": "vertical", "rotation_deg": 0.0,
            "null": [0.0, 0.0], "gain": [1.0, 1.0], "partial": False,
            "projection_radius_m": 0.00044, "projection_pixels": 512,
            "mesh": "/data/specimen.ply", "x3d": "/data/specimen_uv.x3d",
            "path_source": "noether (999 poses)",
            "segment_report": {"observed_m": float(n*pitch), "sweep_s": 10.0},
            "trigger_report": {"mode": "timestamped-ec", "real_path": "/PXI1Slot2/ai0",
                               "imag_path": "/PXI1Slot2/ai1", "block_samples": 100,
                               "sample_rate_hz": sample_rate_hz,
                               "block_span_s": (100.0/sample_rate_hz
                                                if sample_rate_hz else None),
                               "reduce": "mean", "duplicate_traces": p, "errors": 0,
                               "dropped_crossings": 0, "over_budget": 2*p,
                               "staging_timeouts": 0, "fault": None,
                               "max_correlation_wait_s": 0.08},
        }
        np.savez(os.path.join(scan, "pass%03d.npz" % p), position=pos, impedance=z,
                 conditioned=z.imag, s_m=s_m, index=np.arange(n),
                 pose=np.zeros((n, 7)), meta=json.dumps(meta))
    return scan


# ==============================================================================
# the file itself
# ==============================================================================

def test_the_plugin_is_startable_at_all():
    """Not defined, and no shebang, are the two ways it never runs."""
    assert os.path.isfile(PLUGIN), PLUGIN
    assert os.access(PLUGIN, os.X_OK), \
        "no executable bit: NORMS reports 'Plug-in Not Defined' and runs nothing"
    with open(PLUGIN, "rb") as f:
        first = f.readline()
    assert first.startswith(b"#!"), "no shebang: NORMS reports 'JSON Missing'"
    assert not first.rstrip(b"\n").endswith(b"\r"), \
        "CRLF in the shebang line: the kernel looks for an interpreter ending in \\r"
    assert b"env " not in first, \
        ("the interpreter must be pinned, not resolved through PATH: NORMS inherits "
         "whatever environment it has, and a sourced ROS overlay puts a python3 first "
         "that does not carry this machine's numpy")
    with open(PLUGIN, "rb") as f:
        assert b"\r\n" not in f.read(), "CRLF line endings"


def test_it_needs_no_ros_and_no_environment():
    """NORMS passes its own environment, unchanged, and that is all there is."""
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    p = subprocess.run([PLUGIN], capture_output=True, text=True, timeout=300,
                       env=env, stdin=subprocess.DEVNULL)
    doc = one_object(p.stdout)
    assert doc, "the empty record should still come out with no environment at all"
    src = open(PLUGIN).read()
    for forbidden in ("import rclpy", "import dataguzzler", "import spatialnde2"):
        assert forbidden not in src, forbidden


# ==============================================================================
# the no-argument contract
# ==============================================================================

def test_the_no_argument_run_lists_every_property_empty():
    """EASEL runs it once with no arguments and CACHES the answer per path for
    the life of the process; that cached list is what the documentation forms
    offer. A property that appears only on a real run never reaches a form."""
    out, _ = run()
    doc = one_object(out)
    seen = list(properties(doc))
    assert seen, "no properties at all"
    for section, name, field in seen:
        assert field["Value"] == "", "%s/%s should be empty on the no-argument run" % (section, name)
        assert isinstance(field["Value"], str)
        assert "Unit" in field
    assert any(f["Unit"] for _, _, f in seen), "no units anywhere: they belong on this run"


def test_the_sections_are_not_easel_s_own():
    out, _ = run()
    doc = one_object(out)
    for section in doc:
        base = section.split("[")[0]
        assert base not in EASEL_SECTIONS, \
            ("section %r is EASEL's: a plug-in property there REPLACES what the "
             "operator typed on the form" % (section,))


def test_a_real_run_fills_in_exactly_the_advertised_properties(tmp_path):
    """The two runs have to agree, or the forms and the record drift apart."""
    scan = make_scan(str(tmp_path))
    advertised = {(s, n) for s, n, _ in properties(one_object(run()[0]))}
    real = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    produced = {(s, n) for s, n, _ in properties(real)
                if not s.startswith("Plugin File")}
    assert produced == advertised, \
        "difference: %s" % (produced.symmetric_difference(advertised),)


# ==============================================================================
# what NORMS does with the values
# ==============================================================================

def test_every_value_is_a_string_and_nothing_contains_a_pipe(tmp_path):
    """A number, bool or null Value is read as "" -- silently. And '|' is
    EASEL's internal field separator: it shifts or crashes the merge."""
    scan = make_scan(str(tmp_path))
    out, _ = run("--files", os.path.join(scan, "pass000.npz"))
    assert "|" not in out, "a '|' reached the output"
    doc = one_object(out)
    for section, name, field in properties(doc):
        assert isinstance(field["Value"], (str, list)), \
            "%s/%s is %r, which NORMS reads as empty" % (section, name, type(field["Value"]))
        assert "|" not in section and "|" not in name


def test_the_measured_values_are_the_measured_values(tmp_path):
    scan = make_scan(str(tmp_path), passes=12, spacing=0.00025)
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    s = doc["Eddy Current Scan"]["Properties"]
    c = doc["Eddy Current Conditioning"]["Properties"]
    assert s["Passes"]["Value"] == "12"
    assert s["Scan Name"]["Value"] == os.path.basename(scan)
    assert abs(float(s["Trace Pitch"]["Value"]) - 0.25) < 1e-6
    # THE ONE THAT WAS WRONG FIRST TIME. Passes do not all cover the same stretch
    # of the part, so their CENTROIDS differ along the pass by far more than the
    # line spacing -- on the 2026-10-09 scan, 3.85 mm against a true 0.25 mm.
    # Only the component ACROSS the fitted pass direction is the spacing.
    assert abs(float(s["Pass Spacing"]["Value"]) - 0.25) < 0.02, \
        "pass spacing should be measured across the pass, not between centroids"
    angle = float(c["Measured Lift-off Angle"]["Value"])
    assert abs(angle - 86.0) < 4.0, "lift-off axis should come back near 86 deg, got %s" % angle
    assert float(c["Peak Flaw SNR"]["Value"]) > 5.0, "the planted flaw should stand out"


def test_both_figures_exist_before_the_plugin_exits_and_are_declared(tmp_path):
    """NORMS looks for declared files immediately after writing the .norms and
    never comes back for one that was not there; an undeclared file in the right
    folder is not hashed or backed up either.

    TWO figures: the raw impedance as acquired, and the same traces rotated onto
    the measured lift-off axis. The conditioned one is where a flaw is visible
    at all -- until lift-off is separated out it is a small feature on a large
    varying background -- and the raw one is the record of what was measured.

    The raw pair is the instrument's HORIZONTAL and VERTICAL analog outputs,
    carrying whatever rotation its front panel was set to. They are not
    in-phase and quadrature against any reference, and the panes used to claim
    they were.
    """
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    declared = [(k, v) for k, v in doc.items() if k.startswith("Plugin File")]
    assert len(declared) == 2, \
        "expected the raw and the conditioned figure, got %r" % ([k for k, _ in declared],)

    # Numbered from 1 with no gaps: a key with no [n] is index 1, so a gap
    # silently overwrites.
    assert sorted(k for k, _ in declared) == ["Plugin File[1]", "Plugin File[2]"]

    names = []
    for key, entry in declared:
        name = entry["Properties"]["File Name"]["Value"]
        rel = entry["Properties"]["Relative Path"]["Value"]
        assert not os.path.isabs(name) and ".." not in rel, (key, name, rel)
        produced = os.path.join(scan, rel, name) if rel else os.path.join(scan, name)
        assert os.path.isfile(produced), \
            "%s declared %r but it is not in the scan folder" % (key, name)
        assert os.path.getsize(produced) > 1000, (key, name)
        # and it must be INSIDE the folder the .norms will be written to
        assert os.path.realpath(produced).startswith(os.path.realpath(scan))
        names.append(name)

    assert any(n.endswith("_ec_raw.png") for n in names), names
    assert any(n.endswith("_ec_conditioned.png") for n in names), names
    # Named after the measurement, so two scans in one folder cannot overwrite
    # each other's figures.
    assert all(n.startswith(os.path.basename(scan)) for n in names), names


def test_the_image_has_one_row_per_pass_and_no_holes(tmp_path):
    """The reason the plug-in left holes, and the reason it no longer does.

    A world x-y grid at the trace pitch cannot reconstruct the scan, because
    the traces are not on that grid: a pass wanders in x along its own length
    and the across spacing scatters, so neighbouring traces snap into one cell
    and the next cell gets none. Measured on the 2026-10-16 scan: 838 cells
    held two or more traces (10.2% of all of them) while 1012 cells (11.7%)
    were empty -- the same thousand events counted from both ends.

    In the scan's own coordinates the samples ARE on the lattice: `s_m` is
    exact arc length so the along axis is uniform by construction, and one
    image row per pass makes an across-collision impossible. The same traces
    then grid with no holes and no collisions at all.
    """
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    s_ = doc["Eddy Current Scan"]["Properties"]
    assert s_["Image Rows"]["Value"] == "12", "one row per pass"
    assert float(s_["Image Gaps"]["Value"]) == 0.0, \
        "a hole inside the measured coverage means the binning is aliasing again"
    assert s_["Scan Coordinates"]["Value"] == "arc length along the pass"


def test_a_pass_that_was_cut_short_still_leaves_a_hole(tmp_path):
    """Filling every cell is not the goal -- not INVENTING cells is.

    A clearance crop or a dropped region is a real gap in the coverage and the
    figure has to show it. What the change removed was holes that were an
    artifact of the grid; a hole that is a missing measurement must survive.
    """
    scan = make_scan(str(tmp_path), ragged=True)
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    s_ = doc["Eddy Current Scan"]["Properties"]
    # the ragged fixture shortens alternate passes, so the image is wider than
    # the short passes reach -- and those cells must be blank, not filled
    assert int(s_["Image Columns"]["Value"]) > 0
    assert float(s_["Image Gaps"]["Value"]) == 0.0, \
        "gaps are counted INSIDE each pass's own span; a short pass is not a gap"
    # TWO DIFFERENT BLANKS. The short passes leave the rectangle part empty,
    # and THAT is the number that says how much of the part was scanned. On
    # the 2026-10-09 run, where the clearance gate cut most of the passes
    # short, it reads 60%.
    cov = float(s_["Image Coverage"]["Value"])
    assert 0.0 < cov < 100.0, cov
    full = make_scan(str(tmp_path / "full"), ragged=False)
    cov2 = float(one_object(run("--files", os.path.join(full, "pass000.npz"))[0])[
        "Eddy Current Scan"]["Properties"]["Image Coverage"]["Value"])
    assert cov2 > cov, "a scan with no cropped passes must cover more (%g vs %g)" % (cov2, cov)


def test_the_record_says_which_coordinates_the_figure_is_in(tmp_path):
    """A figure in scan coordinates and a figure in world coordinates are
    different pictures, and a record that does not say which it holds cannot
    be read later. `Pass Straightness` is how far the arm's own positions
    departed from the model -- the pass's wander, not an error."""
    scan = make_scan(str(tmp_path))
    s_ = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])[
        "Eddy Current Scan"]["Properties"]
    assert s_["Scan Coordinates"]["Value"]
    assert s_["Pass Straightness"]["Value"], s_
    assert float(s_["Pass Straightness"]["Value"]) >= 0.0


def test_a_serpentine_is_folded_out_rather_than_mirrored(tmp_path):
    """Alternate passes run the other way, so their `s_m` increases in the
    opposite world direction. Taken at face value the rows would be mirrored
    against each other and a flaw would appear twice, a few millimetres apart
    -- which is exactly the shape of a real fault
    ([[ec-position-lags-by-a-tenth-of-a-second]]) and must not be manufactured
    by the plotting."""
    # ON AN ODD PASS, which is the one the serpentine reverses. With the flaw
    # on an even pass this test passes whether the folding works or not --
    # that was the first version of it.
    flaw = (7, 20)
    scan = make_scan(str(tmp_path), serpentine=True, ragged=False, flaw_at=flaw)
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    c = doc["Eddy Current Conditioning"]["Properties"]
    along = float(c["Peak Position Along"]["Value"])
    # The fixture plants one flaw at a known fraction along one pass; after
    # folding, it must land there and not at the mirror image of it.
    scan2 = make_scan(str(tmp_path / "comb"), serpentine=False, ragged=False,
                      flaw_at=flaw)
    c2 = one_object(run("--files", os.path.join(scan2, "pass000.npz"))[0])[
        "Eddy Current Conditioning"]["Properties"]
    assert abs(along - float(c2["Peak Position Along"]["Value"])) < 0.6, \
        "the serpentine put the flaw %.2f mm from where the comb put it" % (
            abs(along - float(c2["Peak Position Along"]["Value"])),)


def test_the_figures_are_report_ready_and_carry_no_interpretation(tmp_path):
    """Titled panes and complete axes -- and nothing the plug-in merely thinks.

    What belongs on the figure is what it IS: which component, which axis,
    what unit. What does not is the plug-in's reading of it -- a circled peak,
    a rotation angle written across the top. Those are already in the .norms
    properties, named and attributed, and a reading drawn onto the picture is
    indistinguishable from the measurement once the picture is in a report.
    These go into one as they are; anything else is added later, deliberately,
    by whoever is making the argument.

    Checked on the SOURCE, because the alternative is OCR on a PNG.
    """
    src = open(PLUGIN).read()
    i = src.index("def _pane(")
    plotting = src[i:src.index("def first(", i)]

    for banned in ("suptitle", "annotate", "ax.text(", "ax.plot(",
                   "axvline", "axhline", "legend(", "arrow("):
        assert banned not in plotting, \
            "the figures draw %r on top of the data" % (banned,)

    # and what has to be there
    assert "ax.set_title(title)" in plotting, "the panes need naming"
    assert 'ax.set_xlabel("along the pass (mm)")' in plotting, \
        "the axes name the SCAN's coordinates, not world x and y"
    assert 'ax.set_ylabel("across the passes (mm)")' in plotting
    assert 'label="V"' in plotting, "the colorbar needs its unit"
    # pcolormesh over the real row edges, never imshow onto an even grid: the
    # across spacing scatters a few percent and stretching it onto an even
    # grid misplaces every indication by up to half a line.
    assert "ax.pcolormesh(" in plotting, plotting[:200]
    assert ".imshow(" not in plotting, "imshow would even out the row spacing"


def test_the_panes_are_titled_for_what_they_hold(tmp_path):
    """Horizontal and vertical for the raw pair, lift-off and flaw for the
    conditioned one -- and in the same order, because the flaw channel is the
    component perpendicular to lift-off, so it sits where vertical sits."""
    src = open(PLUGIN).read()
    raw = src[src.index("def render_raw("):src.index("def render_conditioned(")]
    assert '"Horizontal"' in raw and '"Vertical"' in raw, raw
    for wrong in ('"Real"', '"Imaginary"', "in-phase", "quadrature"):
        assert wrong not in raw.split('"""')[2], \
            "render_raw still titles a pane %r" % (wrong,)

    cond = src[src.index("def render_conditioned("):src.index("def first(")]
    assert '"Lift-off"' in cond and '"Flaw"' in cond, cond


def test_the_pane_order_matches_between_the_two_figures(tmp_path):
    """Horizontal then vertical, lift-off then flaw, so the same position in
    both figures holds the same component of the same traces."""
    src = open(PLUGIN).read()
    raw = src[src.index("def render_raw("):src.index("def render_conditioned(")]
    assert raw.index(".real") < raw.index(".imag")
    assert raw.index("axes[0], H") < raw.index("axes[1], V")

    cond = src[src.index("def render_conditioned("):src.index("def first(")]
    assert cond.index("axes[0], L") < cond.index("axes[1], F"), \
        "lift-off must be the left pane, as horizontal is in the raw figure"


def test_the_two_figures_are_different_pictures(tmp_path):
    """A raw figure and a conditioned figure that come out byte-identical would
    mean the rotation never happened -- which is the failure the single old
    figure actually had: it was labelled "Real" and "Imaginary" and plotted the
    CONDITIONED components under those names, so the raw data had no picture at
    all and the pane an operator read as in-phase was the lift-off channel."""
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    paths = [os.path.join(scan, v["Properties"]["File Name"]["Value"])
             for k, v in doc.items() if k.startswith("Plugin File")]
    blobs = [open(p, "rb").read() for p in paths]
    assert blobs[0] != blobs[1], "the two figures are the same image"


def test_a_figure_that_cannot_be_written_leaves_nothing_behind(tmp_path):
    """Half a figure set is worse than none.

    A file written into the scan folder but NOT declared is not hashed and not
    backed up -- it stays in EASEL's modified-files list and is offered to the
    NEXT measurement on its wait-for-files form, so one scan's debris ends up
    attached to another scan's record. Simulated by making the folder
    unwritable, which is also a real failure mode: the data root is a shared,
    group-writable tree.
    """
    scan = make_scan(str(tmp_path))
    before = set(os.listdir(scan))
    mode = os.stat(scan).st_mode
    os.chmod(scan, 0o500)                     # readable, not writable
    try:
        out, _err = run("--files", os.path.join(scan, "pass000.npz"))
    finally:
        os.chmod(scan, mode)
    doc = one_object(out)                     # still exactly one JSON object
    assert not [k for k in doc if k.startswith("Plugin File")], \
        "declared a file it could not write: EASEL records that as an error"
    assert set(os.listdir(scan)) == before, "left debris in the scan folder"
    # and the metadata it could measure is still there
    assert doc["Eddy Current Scan"]["Properties"]["Passes"]["Value"] == "12"


def test_the_flaw_the_record_reports_is_in_the_record_not_on_the_figure(tmp_path):
    """The peak position, response and SNR are properties, not drawings.

    They are the plug-in's reading of the data and they are worth having --
    the flaw on the 2026-10-15 scan is a millimetre across in a 23 x 8 mm field
    -- but a reading drawn onto the picture is indistinguishable from the
    measurement once the picture is in a report. In the record it is named,
    attributed as auto-generated, and machine-readable.
    """
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    c = doc["Eddy Current Conditioning"]["Properties"]
    for name in ("Peak Position Across", "Peak Position Along",
                 "Peak Flaw Response", "Peak Flaw SNR", "Flaw Channel Noise"):
        assert c[name]["Value"], "%s is empty: the reading has to survive" % name
    assert float(c["Peak Flaw SNR"]["Value"]) > 5.0


def test_an_arbitrary_lift_off_angle_is_withheld_rather_than_recorded(tmp_path):
    """A cloud that is not elongated has no principal direction.

    On 2026-10-16 the scan's own ec_liftoff_rotation_from_traces() refused --
    "anisotropy 2.86, under 3.00, so there is no lift-off direction to find and
    any angle would be arbitrary" -- while this plug-in wrote 86.7729 deg into
    the permanent record of the same measurement with no qualification. The two
    now use the same threshold, and the record says which it was.
    """
    scan = make_scan(str(tmp_path), liftoff_v=0.0)     # no lift-off to find
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    c = doc["Eddy Current Conditioning"]["Properties"]
    ratio = c["Lift-off To Flaw Ratio"]["Value"]
    assert ratio, "the ratio itself is always recorded; it is the evidence"
    assert float(ratio) < 3.0, ratio
    assert c["Lift-off Angle Determinate"]["Value"] == "no", c
    assert c["Measured Lift-off Angle"]["Value"] == "", \
        "an arbitrary angle must not reach the record"

    # and the ordinary case still records it
    ok = make_scan(str(tmp_path / "elongated"))
    doc2 = one_object(run("--files", os.path.join(ok, "pass000.npz"))[0])
    c2 = doc2["Eddy Current Conditioning"]["Properties"]
    assert float(c2["Lift-off To Flaw Ratio"]["Value"]) >= 3.0, c2
    assert c2["Lift-off Angle Determinate"]["Value"] == "yes", c2
    assert c2["Measured Lift-off Angle"]["Value"], c2


def test_the_block_is_recorded_in_time_and_distance_not_just_in_samples(tmp_path):
    """100 samples is not a measurement until you know the rate.

    At 0.25 mm pitch and the 1.1 mm/s the 2026-10-16 scan achieved, a
    100-sample block is 4% of the trace pitch at 10 kHz and 43% of it at 1 kHz
    -- the difference between a trace that is a point and a trace that is a
    quarter of the step. The record held the 100 and not the rate, so it could
    not tell those apart.
    """
    scan = make_scan(str(tmp_path), sample_rate_hz=1000.0)
    s_ = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])[
        "Eddy Current Scan"]["Properties"]
    assert s_["Sample Rate"]["Value"] == "1000"
    assert abs(float(s_["Block Span"]["Value"]) - 100.0) < 1e-6, s_   # ms
    # smear = block span x the speed ACHIEVED, not the one commanded
    speed = float(s_["Mean Pass Speed"]["Value"])                     # mm/s
    assert abs(float(s_["Block Smear"]["Value"]) - 0.1*speed) < 1e-6, s_

    # A scan recorded before the rate was saved reports none of the three,
    # rather than guessing one.
    old = make_scan(str(tmp_path / "old"))
    o = one_object(run("--files", os.path.join(old, "pass000.npz"))[0])[
        "Eddy Current Scan"]["Properties"]
    assert o["Sample Rate"]["Value"] == ""
    assert o["Block Span"]["Value"] == ""
    assert o["Block Smear"]["Value"] == ""
    assert o["Samples Per Trace"]["Value"] == "100", "that part was always known"


def test_the_daq_channels_are_named_horizontal_and_vertical(tmp_path):
    """/PXI1Slot2/ai0 and ai1 are the instrument's X and Y analog outputs, and
    it has already applied its own front-panel rotation to produce them. They
    are not in-phase and quadrature against any reference, and the record used
    to call them Real Channel and Imaginary Channel.

    The scan's OWN metadata still says real_path and imag_path -- that is not
    changed, because old runs have to keep reading -- so what is pinned here is
    that the plug-in translates on the way out.
    """
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--files", os.path.join(scan, "pass000.npz"))[0])
    s_ = doc["Eddy Current Scan"]["Properties"]
    assert "Horizontal Channel" in s_ and "Vertical Channel" in s_, sorted(s_)
    assert "Real Channel" not in s_ and "Imaginary Channel" not in s_, sorted(s_)
    # and the no-argument run advertises the same pair, or EASEL's cached
    # property list and the real run's would disagree
    empty = one_object(run()[0])["Eddy Current Scan"]["Properties"]
    assert "Horizontal Channel" in empty and "Vertical Channel" in empty


# ==============================================================================
# every way it can be called, and every way the data can be bad
# ==============================================================================

@pytest.mark.parametrize("how", ["one", "all", "folder", "positional"])
def test_it_resolves_the_scan_however_easel_points_at_it(tmp_path, how):
    """Only Oldest, Restriction None, a sidecar file or the folder: the plug-in
    takes the FOLDER from whatever it is given, so the EASEL configuration can
    change without touching it."""
    scan = make_scan(str(tmp_path))
    files = sorted(f for f in os.listdir(scan) if f.endswith(".npz"))
    if how == "one":
        args = ["--files", os.path.join(scan, files[0])]
    elif how == "all":
        args = ["--files"] + [os.path.join(scan, f) for f in files]
    elif how == "folder":
        args = ["--files", scan]
    else:
        args = [os.path.join(scan, f) for f in files]
    doc = one_object(run(*args)[0])
    assert doc["Eddy Current Scan"]["Properties"]["Passes"]["Value"] == "12", \
        "every route should see the whole scan"


def test_placeholders_easel_cannot_fill_are_passed_through_literally(tmp_path):
    """In the scan flow the .norms, .hfiles and folder do not exist yet, so
    EASEL passes the text '%n', '%h' and '%f' unchanged."""
    scan = make_scan(str(tmp_path))
    doc = one_object(run("--norms", "%n", "--folder", "%f", "--hfiles", "%h",
                         "--files", os.path.join(scan, "pass000.npz"))[0])
    assert doc["Eddy Current Scan"]["Properties"]["Passes"]["Value"] == "12"
    for key in ("Plugin File[1]", "Plugin File[2]"):
        assert os.path.isfile(os.path.join(
            scan, doc[key]["Properties"]["File Name"]["Value"])), \
            "with %f literal the folder has to come from the data files"


@pytest.mark.parametrize("what", ["missing", "corrupt", "empty_dir", "too_few"])
def test_bad_data_still_produces_exactly_one_json_object(tmp_path, what):
    """Never a partial object, never a bare error. An unreadable scan produces
    the EMPTY record and the reason on stderr, which NORMS records as JSON Valid
    with empty values and a measurement that still completes."""
    if what == "missing":
        args = ["--files", str(tmp_path / "nope.npz")]
    elif what == "corrupt":
        bad = tmp_path / "pass000.npz"
        bad.write_bytes(b"PK\x03\x04 not really an npz")
        args = ["--files", str(bad)]
    elif what == "empty_dir":
        d = tmp_path / "ec_scan_empty"
        d.mkdir()
        args = ["--files", str(d)]
    else:
        args = ["--files", os.path.join(make_scan(str(tmp_path), passes=1, per_pass=3,
                                                  ragged=False), "pass000.npz")]
    out, err = run(*args)
    doc = one_object(out)
    assert doc, "still has to be a JSON object"
    for _, _, field in properties(doc):
        assert isinstance(field["Value"], str)
