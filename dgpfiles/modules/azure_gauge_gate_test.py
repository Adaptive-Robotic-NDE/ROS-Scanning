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
# File:  azure_gauge_gate_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that the automated scan refuses to raster over an Azure mesh
#   that was never gauged.
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
"""run_automated_scan() must REFUSE to raster over an Azure mesh nobody gauged.

WHY THIS EXISTS. Stage 3 of run_automated_scan() plans the Keyence raster over
the stage 2 Azure rough mesh, so every millimetre that mesh reads LOW is a
millimetre the LJ-V7080 really flies CLOSER to the part. The head measures only
over 0.057..0.103 m.

An ungated rough mesh that reads 13.6 mm low across a surface-ground flat plate
and 22 mm low at one end flies the head at about 66 mm, then at about 58 mm
against its 57 mm hard limit, brings 26-39% of the plate back as garbage, and
leaves the probe nearly in contact with the part.

verify_calibration() has measured that same disagreement EIGHT times -- -11.6 to
-13.6 mm, every one of them a FAIL -- without anything stopping
run_automated_scan() from flying. Stage 2a is what stops it, and this file is
the proof, branch by branch, of the decision it makes.

WHAT IS TESTED. Both halves of stage 2a in ECAutomatedScan.dgp, on the host,
with no ROS, no Keyence, no container and no mesh:

  * _ec_gauge_verdict() -- the whole decision, factored out as a pure function
    so every branch can be driven from a fake result dict.
  * _ec_azure_gauge() -- the half that prints, raises and no-ops. Its
    globals() is the namespace this file execs it into, so "azure_gauge.dpi is
    not loaded" is tested by simply not putting it there.

Both are lifted OUT OF THE .dgp by its AST rather than copied here, so this
cannot pass against a stale duplicate, and the wiring checks below read the
real stage order out of the same source.

    python3 azure_gauge_gate_test.py
"""
import ast
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")

#: The LJ-V7080's window, as keyence_standoff.dpi declares it. The gate is
#: handed these at the call site (KEYENCE_REF_DIST_M, KEYENCE_RANGE_M[0]); they
#: are repeated here so the arithmetic below is checked against the datasheet
#: rather than against whatever the config happens to hold.
REF_M = 0.080
NEAR_M = 0.057

#: AZURE_GAUGE_MAX_MM. azure_gauge.dpi owns it; this is the value the gate was
#: specified against.
TOL_MM = 2.0

#: verify_calibration()'s eight independent measurements, sign
#: flipped into the gauge's convention (POSITIVE means the MESH READS LOW).
#: Every one of these must refuse.
MEASURED_DZ_MM = (13.6, 12.5, 12.6, 13.0, 12.9, 11.6, 11.8, 11.6)

CHECKS, FAILURES = [], []


def check(what, cond, detail=""):
    CHECKS.append(what)
    if cond:
        print("  ok    %s" % (what,))
    else:
        FAILURES.append(what)
        print("  FAIL  %s%s" % (what, ("  --  " + str(detail)) if detail else ""))


WANT = ("_ec_gauge_verdict", "_ec_azure_gauge")


def load_gate():
    """The two gate functions out of the .dgp, compiled on their own.

    A .dgp is a function body and is not importable, and this one opens a Qt
    viewer and a recording database when it runs. So: parse it, find the two
    FunctionDefs, and compile THOSE into a namespace this file controls. The
    namespace IS their globals(), which is how _ec_azure_gauge()'s
    globals().get() lookups are steered below without any of azure_gauge.dpi,
    calibration.dpi or ROS being present.

    Returns (namespace, {name: ast node}).
    """
    tree = ast.parse(open(DGP).read(), filename="ECAutomatedScan.dgp")
    nodes = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in WANT:
            if n.name in nodes:
                raise AssertionError("%s() is defined twice in "
                                     "ECAutomatedScan.dgp" % (n.name,))
            nodes[n.name] = n
    missing = [w for w in WANT if w not in nodes]
    if missing:
        raise AssertionError("ECAutomatedScan.dgp has no %s()"
                             % (", ".join(missing),))
    ns = {}
    mod = ast.Module(body=[nodes[w] for w in WANT], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "ECAutomatedScan.dgp", "exec"), ns)
    return ns, nodes


def result(dz=None, ok=True, n=18, requested=18, residual=0.30,
           tilt=1.1, why=""):
    """gauge_mesh_against_keyence()'s contract, as a dict."""
    return {"ok": bool(ok), "n_measured": int(n), "n_requested": int(requested),
            "dz_mm": dz, "tilt_mrad": tilt, "residual_mm": residual,
            "samples": [], "why": why}


def verdict(fn, res, allow=False, skip_why=None, tol=TOL_MM):
    v, lines = fn(res, max_mm=tol, ref_dist_m=REF_M, near_limit_m=NEAR_M,
                  allow=allow, skip_why=skip_why)
    return v, "\n".join(lines)


def run_gate(nodes, mesh="/tmp/rough.ply", allow=False, gauge="present",
             res=None, raises=None, points=18, tol=TOL_MM):
    """_ec_azure_gauge() with its globals() steered. Returns (out, err, text).

    A FRESH namespace per call, because that namespace is the function's
    globals(): what is absent from it is absent from the session.
    """
    ns, _ = load_gate()
    said = []
    ns["_cal_say"] = said.append
    ns["KEYENCE_REF_DIST_M"] = REF_M
    ns["KEYENCE_RANGE_M"] = (NEAR_M, 0.103)
    if gauge == "present":
        def _gauge(mesh=None, n=None, say=None):
            ns["_called"] = {"mesh": mesh, "n": n, "say": say}
            if raises is not None:
                raise raises
            return res
        ns["gauge_mesh_against_keyence"] = _gauge
        ns["AZURE_GAUGE_POINTS"] = points
        if tol is not None:
            ns["AZURE_GAUGE_MAX_MM"] = tol
    out, err = None, None
    try:
        out = ns["_ec_azure_gauge"](mesh, allow=allow)
    except Exception as e:                                        # noqa: BLE001
        err = e
    return out, err, "\n".join(said) if said else str(err or "")


def main():
    ns, nodes = load_gate()
    fn, node = ns["_ec_gauge_verdict"], nodes["_ec_gauge_verdict"]

    print("\n0. A NaN dz IS NOT A MEASUREMENT")
    # gauge_mesh_against_keyence() reports dz_mm as NaN when nothing was
    # measured (the mean of no samples). `dz is not None` passes for NaN, which
    # turns the refusal into "would fly at nan mm, which is nan mm INSIDE the
    # head's 57 mm hard limit" -- fabricated arithmetic in the one sentence
    # that tells an operator how close the head flies.
    _nan = float("nan")
    v, text = verdict(fn, result(dz=_nan, ok=False, n=0,
                                 why="nothing was measured"))
    check("a NaN dz still refuses", v == "refuse", "%s: %s" % (v, text))
    check("and says NOTHING about the standoff it would fly at",
          "nan" not in text.lower(), text)
    check("and does not claim a distance from the hard limit",
          "hard limit" not in text or "nan" not in text.lower(), text)
    for _bad in (float("inf"), float("-inf")):
        v2, t2 = verdict(fn, result(dz=_bad, ok=False, n=0,
                                    why="nothing was measured"))
        check("an infinite dz refuses without arithmetic (%r)" % (_bad,),
              v2 == "refuse" and "inf" not in t2.lower(), "%s: %s" % (v2, t2))

    print("\n1. THE GAUGE PASSES -- the mesh may be rastered over")
    v, text = verdict(fn, result(dz=0.31))
    check("a 0.31 mm disagreement passes", v == "pass", "%s: %s" % (v, text))
    check("it says the measured number", "+0.31 mm" in text, text)
    check("and the tolerance it was judged against", "2.0 mm" in text, text)
    check("and which way positive means", "reads low" in text, text)
    # 80 - 0.31 = 79.7 mm flown, 79.7 - 57 = 22.7 mm of margin.
    check("and the standoff the raster will actually fly at",
          "79.7 mm" in text and "22.7 mm clear of" in text, text)
    check("a disagreement exactly at the tolerance still passes",
          verdict(fn, result(dz=TOL_MM))[0] == "pass")
    check("and so does the same error the other way",
          verdict(fn, result(dz=-TOL_MM))[0] == "pass")

    print("\n2. A MESH AT THIS CELL'S STANDING BIAS IS REFUSED")
    # 80 - 12.45 = 67.55 flown, 10.55 mm of margin.
    v, text = verdict(fn, result(dz=12.45))
    check("a 12.45 mm disagreement REFUSES the raster", v == "refuse",
          "%s: %s" % (v, text))
    check("the refusal says stage 3 is refused",
          text.startswith("stage 3 is refused:"), text)
    check("it gives the measured dz and which way it reads",
          "12.45 mm LOW" in text, text)
    check("it gives the tolerance", "tolerance is 2.0 mm" in text, text)
    check("it gives the standoff the raster would have flown at",
          "would fly at 67.5 mm" in text, text)
    check("and the margin to the head's hard limit",
          "10.5 mm clear of the head's 57 mm hard limit" in text, text)
    check("it names the override", "allow_azure_bias=True" in text, text)
    check("it says what to do instead", "verify_calibration()" in text, text)
    check("and it is short enough to read at the cell",
          len(text.splitlines()) <= 6, "%d lines" % len(text.splitlines()))

    print("\n   ...and so is every measurement verify_calibration() has made")
    for dz in MEASURED_DZ_MM:
        v, text = verdict(fn, result(dz=dz))
        check("%.1f mm low is refused" % (dz,), v == "refuse", text)
    # 22 mm is the worst end of a drifted plate: 58.0 mm flown, 1.0 mm from the
    # limit. That is the band that comes back as garbage.
    v, text = verdict(fn, result(dz=22.0))
    check("a 22 mm disagreement is refused at 58.0 mm, 1.0 mm clear",
          v == "refuse" and "58.0 mm" in text and "1.0 mm clear" in text, text)
    # A mesh reading HIGH is the other failure: the head flies past the far end
    # of its window and simply returns nothing. Refused, and said correctly.
    v, text = verdict(fn, result(dz=-12.0))
    check("a mesh reading 12 mm HIGH is refused too, and called HIGH",
          v == "refuse" and "12.00 mm HIGH" in text, text)
    check("...at 92.0 mm, which is reported as clear of the NEAR limit",
          "92.0 mm" in text, text)

    print("\n3. A GAUGE THAT COULD NOT MEASURE IS ALSO A REFUSAL")
    v, text = verdict(fn, result(ok=False, dz=None, n=3,
                                 why="only 3 of 18 points returned a profile"))
    check("ok=False refuses", v == "refuse", "%s: %s" % (v, text))
    check("it quotes the gauge's own reason",
          "only 3 of 18 points returned a profile" in text, text)
    check("it says how many points it got", "3 of 18 points" in text, text)
    check("and it does NOT invent a standoff it cannot know",
          "would fly at" not in text, text)
    # ok=False with a number still present is the same answer: the gauge is the
    # authority on whether its own number means anything.
    check("ok=False refuses even when dz looks fine",
          verdict(fn, result(ok=False, dz=0.1, why="the fit was rejected"))[0]
          == "refuse")
    # A gauge that raised is wrapped into ok=False by the caller; a gauge that
    # returned something that is not a result dict at all must not crash here.
    v, text = verdict(fn, None)
    check("a None result refuses rather than passing", v == "refuse", text)
    check("...and says the mesh is ungauged", "ungauged" in text, text)
    v, text = verdict(fn, result(dz=12.45), tol=None)
    check("a missing AZURE_GAUGE_MAX_MM refuses rather than guessing one",
          v == "refuse" and "AZURE_GAUGE_MAX_MM" in text, text)

    print("\n4. THE OVERRIDE FLIES IT AND SAYS SO")
    v, text = verdict(fn, result(dz=12.45), allow=True)
    check("allow=True does not refuse", v == "override", "%s: %s" % (v, text))
    check("it names the override that was used",
          "allow_azure_bias" in text, text)
    check("it still states the disagreement", "12.45 mm LOW" in text, text)
    check("it still states the standoff that will be flown",
          "67.5 mm" in text, text)
    check("and it still states how much of a plate that costs",
          "26-39%" in text, text)
    check("the override does not touch a gauge that could not measure either",
          verdict(fn, result(ok=False, dz=None, why="no profiles"),
                  allow=True)[0] == "override")
    check("and it changes nothing about a passing gauge",
          verdict(fn, result(dz=0.31), allow=True)[0] == "pass")

    print("\n5. NO MESH: A CLEAN NO-OP THAT SAYS WHY")
    v, text = verdict(fn, None, skip_why="this run has no rough mesh")
    check("no mesh skips rather than refusing", v == "skip", "%s: %s" % (v, text))
    check("it is not silent", bool(text.strip()), repr(text))
    check("it says the mesh was NOT gauged", "NOT gauged" in text, text)
    check("it repeats the caller's reason",
          "this run has no rough mesh" in text, text)
    check("it points at the standalone check",
          "verify_calibration()" in text, text)
    check("it does not claim a pass", "agrees with the Keyence" not in text, text)
    # skip_why wins over anything in the result, so a stale dict cannot make a
    # skip look like a measurement.
    check("a skip reason wins over a leftover result",
          verdict(fn, result(dz=12.45), skip_why="no gauge loaded")[0] == "skip")

    print("\n6. THE FUNCTION STAYS PURE, WHICH IS WHAT MAKES 1-5 POSSIBLE")
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            names.add(n.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            names.add(n.func.id)
    allowed = {"res", "max_mm", "ref_dist_m", "near_limit_m", "allow",
               "skip_why", "dz", "n_meas", "n_req", "resid", "over",
               "consequence", "flown", "margin", "lines", "l",
               "isinstance", "dict", "float", "int", "bool", "abs", "str"}
    check("it reads no global and calls nothing but builtins",
          not (names - allowed), sorted(names - allowed))
    check("it neither prints nor raises -- the caller does both",
          not any(isinstance(n, ast.Raise) for n in ast.walk(node))
          and "print" not in names)

    print("\n7. _ec_azure_gauge(): THE HALF THAT PRINTS AND RAISES")
    # A PASS carries on and says the number. Nothing is raised, and the mesh is
    # NOT shifted -- the gate returns a verdict, never a new surface.
    out, err, text = run_gate(nodes, res=result(dz=0.31))
    check("a passing gauge returns instead of raising",
          err is None and out["verdict"] == "pass", "%r %r" % (err, out))
    check("it prints the number where the operator will see it",
          "+0.31 mm" in text, text)
    check("it hands the result back for the capture record",
          out["result"]["dz_mm"] == 0.31, out)
    check("and it returns no mesh of its own to plan over",
          set(out) == {"verdict", "result", "why"}, sorted(out))

    # A FAILING GAUGE RAISES. run_automated_scan()'s `finally` parks the arm.
    out, err, text = run_gate(nodes, res=result(dz=12.45))
    check("a failing gauge RAISES rather than printing and carrying on",
          isinstance(err, ValueError), "%r" % (err,))
    check("the exception carries the whole refusal",
          "would fly at 67.5 mm" in str(err)
          and "allow_azure_bias=True" in str(err), str(err))
    check("and it does not print the refusal twice", text == str(err), text)

    # THE GAUGE ITSELF DYING IS A REFUSAL WITH A REASON, not a traceback out of
    # the middle of stage 2.
    out, err, text = run_gate(nodes, raises=RuntimeError("no /profiles"))
    check("a gauge that raises becomes a refusal", isinstance(err, ValueError),
          "%r" % (err,))
    check("...naming what went wrong inside it",
          "RuntimeError" in str(err) and "no /profiles" in str(err), str(err))
    check("...and it is not the gauge's own exception type",
          not isinstance(err, RuntimeError), "%r" % (err,))

    # THE CONTRACT IT CALLS: mesh=, n=AZURE_GAUGE_POINTS, say=.
    ns2, _ = load_gate()
    ns2["_cal_say"] = lambda m: None
    ns2["KEYENCE_REF_DIST_M"], ns2["KEYENCE_RANGE_M"] = REF_M, (NEAR_M, 0.103)
    ns2["AZURE_GAUGE_MAX_MM"], ns2["AZURE_GAUGE_POINTS"] = TOL_MM, 24
    seen = {}

    def _spy(mesh=None, n=None, say=None):
        seen.update({"mesh": mesh, "n": n, "say": say})
        return result(dz=0.1)

    ns2["gauge_mesh_against_keyence"] = _spy
    ns2["_ec_azure_gauge"]("/tmp/rough.ply")
    check("it calls the gauge with mesh=, n=AZURE_GAUGE_POINTS and say=",
          seen.get("mesh") == "/tmp/rough.ply" and seen.get("n") == 24
          and callable(seen.get("say")), seen)

    # THE TWO NO-OPS. Neither may raise, and neither may be silent.
    out, err, text = run_gate(nodes, gauge="absent")
    check("no gauge in the session is a no-op, not a refusal",
          err is None and out["verdict"] == "skip", "%r %r" % (err, out))
    check("...and it says azure_gauge.dpi is the reason",
          "azure_gauge.dpi" in text, text)
    out, err, text = run_gate(nodes, mesh=None, res=result(dz=0.1))
    check("no rough mesh is a no-op, not a refusal",
          err is None and out["verdict"] == "skip", "%r %r" % (err, out))
    check("...and it says there is no mesh", "no rough mesh" in text, text)
    check("...and it did not call the gauge", out["result"] is None, out)
    out, err, text = run_gate(nodes, mesh="", res=result(dz=0.1))
    check("an empty mesh path is the same no-op", out["verdict"] == "skip", out)

    # THE OVERRIDE, THROUGH THE WRAPPER.
    out, err, text = run_gate(nodes, res=result(dz=12.45), allow=True)
    check("allow=True carries on and says so",
          err is None and out["verdict"] == "override", "%r %r" % (err, out))
    check("...loudly, with the number it is ignoring",
          "allow_azure_bias" in text and "12.45 mm LOW" in text, text)

    print("\n8. THE GATE IS WIRED IN WHERE IT PROTECTS SOMETHING")
    src = open(DGP).read()
    i_rough = src.index('out["rough"] = reconstruct_rough(')
    i_gauge = src.index('out["gauge"] = _ec_azure_gauge(')
    i_roi = src.index("2b/5 region of interest")
    i_fine = src.index('out["fine"] = reconstruct_fine(')
    check("it runs AFTER the rough mesh exists", i_rough < i_gauge)
    check("and BEFORE the Keyence raster is planned over it", i_gauge < i_fine)
    check("and before the operator is asked to draw a region on that mesh",
          i_gauge < i_roi)
    check("run_automated_scan() takes the override, defaulting False",
          "allow_azure_bias=False" in src)
    check("...and an unattended run can set it in the config",
          "EC_ALLOW_AZURE_BIAS = False" in src
          and 'globals().get("EC_ALLOW_AZURE_BIAS", False)' in src)
    check("azure_gauge.dpi is included after calibration.dpi, whose Keyence "
          "helpers it reuses",
          src.index('"calibration.dpi"') < src.index('"azure_gauge.dpi"'))
    check("...and after keyence_standoff.dpi, which owns the window",
          src.index('"keyence_standoff.dpi"') < src.index('"azure_gauge.dpi"'))
    check("the gate does not shift the mesh behind the operator's back",
          "REFUSE-ONLY" in src)
    # scan=False is what capture_full_pipeline() calls, and it still plans the
    # raster (stages 3-5), so it has to be gated too: the gate is upstream of
    # the early return, not next to it.
    check("scan=False is gated as well -- it plans the raster too",
          i_gauge < src.index("stopping before the scan (scan=False)"))

    print("\n9. EVERY EXISTING CALLER STILL FITS THE SIGNATURE")
    # A new parameter with no default, or a renamed one, would break these at
    # the cell rather than here. cell_capture_pipeline.dpi's scenario 8 and the
    # unattended roi_select=False path are the two that matter.
    dgp_tree = ast.parse(src, filename="ECAutomatedScan.dgp")
    sig = [f for f in ast.walk(dgp_tree)
           if isinstance(f, ast.FunctionDef) and f.name == "run_automated_scan"][0]
    params = [a.arg for a in sig.args.args]
    n_default = len(sig.args.defaults)
    check("every parameter still has a default, so a bare call works",
          n_default == len(params), "%d of %d" % (n_default, len(params)))
    check("allow_azure_bias is one of them, and it defaults False",
          "allow_azure_bias" in params
          and getattr(sig.args.defaults[params.index("allow_azure_bias")
                                        - (len(params) - n_default)],
                      "value", None) is False, params)
    callers = 0
    for path in sorted(glob.glob(os.path.join(HERE, "*.dpi"))
                       + glob.glob(os.path.join(HERE, "..", "*.dgp"))):
        try:
            t = ast.parse(open(path).read(), filename=path)
        except SyntaxError:
            continue
        for c in ast.walk(t):
            if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "run_automated_scan"):
                continue
            callers += 1
            bad = [k.arg for k in c.keywords if k.arg and k.arg not in params]
            check("%s calls it with only parameters it has"
                  % (os.path.basename(path),),
                  not bad and len(c.args) <= len(params), bad)
    check("...and at least one real caller was actually found", callers >= 1,
          "%d" % (callers,))

    print("\n%d checks, %d failing" % (len(CHECKS), len(FAILURES)))
    for f in FAILURES:
        print("  FAILED: %s" % (f,))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())


def test_azure_gauge_gate():
    assert main() == 0, FAILURES
