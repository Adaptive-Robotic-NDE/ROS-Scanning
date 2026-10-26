"""Which surface the eddy-current clearance gate is given, and why that one.

TWO SURFACES EXIST BY THE TIME THE GATE RUNS and they answer different questions:

    the cleaned FULL mesh        where may the TOOLING go
    normal_surface              which way must the PROBE AXIS point

normal_surface is cropped to the region being scanned. The tooling is not: the
Keyence head is 75 mm to the side of the probe and the camera 230 mm, so they
sweep over part that a region-cropped mesh does not contain. Grading them against
it reports clear against specimen that is absent from the model, which is the one
failure mode a clearance gate must not have.

Measured on a real run's planning meshes (2026-09-29 15:02:40): the region crop
takes the part from 56 mm wide to 43 mm and drops 3297 of 13450 faces.

The tempting reason to swap them -- that the repaired mesh is "more correct" --
does not survive measurement either. The signed distance takes its sign from the
nearest face's OWN normal, and clean_mesh_for_planning() turns every face
outward, so the full mesh signs correctly without being consistently wound as a
manifold. On that same run: 0 faces pointing down out of 13450, and points 0.2 mm
above the 165 faces touching an inconsistently wound edge all read positive.

So this file pins the choice, because it looks like an oversight and reverting it
would be a safety regression with no benefit.
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DGP = os.path.join(HERE, "..", "ECAutomatedScan.dgp")


def _plan_ec_path():
    tree = ast.parse(open(DGP).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "plan_ec_path":
            return node
    raise AssertionError("ECAutomatedScan.dgp has no plan_ec_path()")


def _clearance_call():
    for node in ast.walk(_plan_ec_path()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "enforce_path_clearance"):
            return node
    raise AssertionError(
        "plan_ec_path() no longer calls enforce_path_clearance(); the tooling is "
        "then unchecked against the part")


def test_the_gate_gets_the_full_mesh_not_the_constrained_surface():
    call = _clearance_call()
    assert len(call.args) >= 2, ast.unparse(call)
    surface = ast.unparse(call.args[1])
    assert surface == "mesh", (
        "the clearance gate is being given %r. It must be `mesh` -- the cleaned "
        "FULL part. normal_surface is cropped to the scanned region, and the "
        "Keyence head and camera sweep outside it, so they would be graded "
        "against part the model does not contain." % (surface,))


def test_the_constrained_surface_is_not_passed_to_the_gate():
    """Named separately: any spelling of it is the same mistake."""
    src = ast.unparse(_clearance_call())
    for name in ("normal_surface", "roi", "_roi_normals"):
        assert name not in src, (
            "the clearance call mentions %r (%s); the gate needs the full part"
            % (name, src))


def test_the_choice_is_explained_where_it_is_made():
    """A reader has to be able to see why, or they will 'fix' it.

    The whole hazard here is that the correct call looks like an oversight.
    """
    # By line number from the AST, not by searching the text: the file mentions
    # enforce_path_clearance in prose above the call, and the first match is that.
    lines = open(DGP).read().splitlines()
    call_line = _clearance_call().lineno
    before = "\n".join(lines[max(0, call_line - 40):call_line])
    for phrase in ("normal_surface", "cropped"):
        assert phrase in before, (
            "the comment above the clearance call does not mention %r, so the "
            "next reader cannot tell the choice was deliberate" % (phrase,))


def test_the_gate_is_told_which_mesh_it_is_grading():
    """Printed, so a log says what was checked rather than only the verdict."""
    src = ast.unparse(_plan_ec_path())
    assert "_say_clearance_mesh" in src, (
        "plan_ec_path() does not state which surface the clearance gate is "
        "grading; a clearance verdict with no named surface cannot be audited")


def test_the_cleaning_step_turns_every_face_outward():
    """The property the full mesh's correctness rests on.

    clean_mesh_for_planning() has two passes that turn faces round -- one against
    the vertex normals, one against world up on a height field. Together they are
    what make the signed distance right. Checked here at the source, because
    without them the full mesh would NOT be the right input and this file's advice
    would be wrong.
    """
    src = open(os.path.join(HERE, "noether_path.dpi")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "clean_mesh_for_planning"):
            body = ast.unparse(node)
            break
    else:
        raise AssertionError("noether_path.dpi has no clean_mesh_for_planning()")
    assert "faces_rewound" in body, (
        "clean_mesh_for_planning() no longer reports faces_rewound; the clearance "
        "gate's sign depends on that pass")
    assert "faces_turned_up" in body, (
        "clean_mesh_for_planning() no longer turns downward faces up on a height "
        "field; a face pointing into the part makes a clear link read as inside")


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (name, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
