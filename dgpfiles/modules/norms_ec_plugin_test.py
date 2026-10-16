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
              flaw_at=(6, 20), ragged=True):
    """A synthetic ec_scan_* directory: passes along +y, stepping in -x.

    `ragged` shortens alternate passes, which is what a clearance crop or a hole
    does to a real scan and what the pass-spacing measurement has to survive.
    """
    rng = np.random.default_rng(4)
    scan = os.path.join(tmpdir, "ec_scan_2026-10-12_000000")
    os.makedirs(scan, exist_ok=True)
    for p in range(passes):
        n = per_pass - (7 if (ragged and p % 2) else 0)
        x = np.full(n, 0.85 - p*spacing)
        y = 0.10 + np.arange(n)*pitch
        pos = np.c_[x, y, np.full(n, 0.94)]
        # lift-off along ~86 deg in the impedance plane, plus a local flaw
        t = rng.normal(0.0, 0.30, n)
        z = t*np.exp(1j*np.radians(86.0))
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
                               "reduce": "mean", "duplicate_traces": p, "errors": 0,
                               "dropped_crossings": 0, "over_budget": 2*p,
                               "staging_timeouts": 0, "fault": None,
                               "max_correlation_wait_s": 0.08},
        }
        np.savez(os.path.join(scan, "pass%03d.npz" % p), position=pos, impedance=z,
                 conditioned=z.imag, s_m=np.arange(n)*pitch, index=np.arange(n),
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
    plotting = src[i:src.index("def extract(", i)]

    for banned in ("suptitle", "annotate", "ax.text(", "ax.plot(",
                   "axvline", "axhline", "legend(", "arrow("):
        assert banned not in plotting, \
            "the figures draw %r on top of the data" % (banned,)

    # and what has to be there
    assert "ax.set_title(title)" in plotting, "the panes need naming"
    assert 'ax.set_xlabel("x (mm)")' in plotting, "axes need a quantity and a unit"
    assert 'ax.set_ylabel("y (mm)")' in plotting
    assert 'label="V"' in plotting, "the colorbar needs its unit"


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
    assert raw.index("GR = ") < raw.index("GI = ")
    assert raw.index(".real") < raw.index(".imag")
    assert raw.index("axes[0], GR") < raw.index("axes[1], GI")

    cond = src[src.index("def render_conditioned("):src.index("def first(")]
    assert cond.index("axes[0], GA") < cond.index("axes[1], GP"), \
        "lift-off (along) must be the left pane, as horizontal is in the raw figure"


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
