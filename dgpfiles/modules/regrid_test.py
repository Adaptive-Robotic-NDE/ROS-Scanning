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
# File:  regrid_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that a volume cell is a function of position rather than of
#   arrival order.
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
"""Check that a volume cell is a function of POSITION, not of arrival order.

Filling the rectangular volume in arrival order -- what nd_accumulate_once does,
assigning the k-th recording it sees to the k-th cell of a map fixed when it was
instantiated -- is right only while nothing goes missing. Lose one trace (a
transaction the database refused as unorderable, a crossing the trigger never
published, a pass that came up short) and every later trace lands one cell
early: the image shears progressively from the gap onwards and the volume gives
no sign of it, because it is full and the right shape.

volume_cell_indices() takes positions and nothing else. The tests below are
mostly about that "nothing else": the same trace gets the same cell whether or
not its neighbours exist.

Runs on the host; no ROS, no recording database.
"""
import ast
import functools
import os
import sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_accumulation.dpi")
tree = ast.parse(open(SRC).read())
fn = [n for n in tree.body
      if isinstance(n, ast.FunctionDef) and n.name == "volume_cell_indices"][0]
ns = {"np": np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)
volume_cell_indices = ns["volume_cell_indices"]

fails = []


def check(name, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print("  %-62s %s" % (name, "ok" if ok else "FAIL got=%r want=%r" % (got, want)))
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


# A 21 x 11 grid, 4.878 mm along u and 10 mm along v, matching the plane scan's
# geometry: origin at a taught corner, u along the row, v between rows.
M, N = 21, 11
DU, DV = 0.1/20.5, 0.010
GRID = {"origin": np.array([0.75, -0.35, 1.0]),
        "u_step": np.array([DU, 0.0, 0.0]),
        "v_step": np.array([0.0, DV, 0.0]),
        "m": M, "n": N}


def cell_positions(cells):
    return np.array([GRID["origin"] + i*GRID["u_step"] + j*GRID["v_step"]
                     for i, j in cells])


# The raster every section below is measured against. Module scope because
# nothing here asserts, so import stays safe and each section is independent.
full = [(i, j) for j in range(4) for i in range(M)]


@_checked
def test_a_full_raster_maps_onto_the_grid():
    print("=== a full raster maps onto the grid exactly ===")
    ii, jj, resid = volume_cell_indices(GRID, cell_positions(full))
    check("every trace lands in its own cell",
          list(zip(ii.tolist(), jj.tolist())), full)
    check("dead centre of the cell", float(resid.max()), 0.0, 1e-12)

    print()


@_checked
def test_a_missing_trace_does_not_move_the_ones_after_it():
    print("=== a missing trace does not move the ones after it ===")
    # Drop trace 7 of row 0. Under arrival order every later trace shifts one cell
    # and the image shears; placed by position the survivors are untouched and the
    # gap is simply a cell nobody claims.
    gapped = [c for c in full if c != (7, 0)]
    gi, gj, _ = volume_cell_indices(GRID, cell_positions(gapped))
    check("survivors keep the cells they had",
          list(zip(gi.tolist(), gj.tolist())), gapped)
    claimed = set(zip(gi.tolist(), gj.tolist()))
    check("the dropped trace's cell is claimed by nobody", (7, 0) in claimed, False)
    check("no cell is claimed twice", len(claimed), len(gapped))
    # What arrival order would have done, for contrast.
    shifted = full[:len(gapped)]
    moved = sum(1 for a, b in zip(gapped, shifted) if a != b)
    print("    arrival order would have moved %d of %d surviving traces" % (moved, len(gapped)))
    check("and that is what placing by position avoids", moved > 0, True)

    print()


@_checked
def test_a_whole_missing_pass_leaves_a_row_of_holes():
    print("=== a whole missing pass leaves a row of holes, not a shifted image ===")
    norow2 = [c for c in full if c[1] != 2]
    ni, nj, _ = volume_cell_indices(GRID, cell_positions(norow2))
    check("rows 0, 1 and 3 unaffected", sorted(set(nj.tolist())), [0, 1, 3])
    check("every survivor still in its own cell",
          list(zip(ni.tolist(), nj.tolist())), norow2)

    print()


@_checked
def test_positions_off_the_grid_are_identifiable():
    print("=== positions off the grid are identifiable, not silently wrapped ===")
    off = np.array([GRID["origin"] - 2*GRID["u_step"],            # before the start
                    GRID["origin"] + (M+3)*GRID["u_step"],        # past the end
                    GRID["origin"] + (N+1)*GRID["v_step"]])       # past the last row
    oi, oj, _ = volume_cell_indices(GRID, off)
    inside = (oi >= 0) & (oi < M) & (oj >= 0) & (oj < N)
    check("all three are outside the grid", int(inside.sum()), 0)
    check("and the indices say where they went", [int(oi[0]), int(oi[1]), int(oj[2])],
          [-2, M+3, N+1])

    print()


@_checked
def test_half_cell_offsets_are_reported():
    print("=== half-cell offsets are reported, not hidden ===")
    # A probe running 40% of a cell ahead of the grid still lands in the right cell,
    # but the residual says the grid is a poor description of the path.
    drift = cell_positions(full) + 0.4*GRID["u_step"]
    di, dj, dres = volume_cell_indices(GRID, drift)
    check("still the intended cells", list(zip(di.tolist(), dj.tolist())), full)
    check("residual reports the drift", float(dres.max()), 0.4, 1e-9)

    print()


@_checked
def test_a_tilted_skewed_grid_follows_the_basis():
    print("=== a tilted, skewed grid: cells follow the basis, not world axes ===")
    # The taught plane is rarely axis-aligned. u climbs, v is perpendicular to it.
    TILT = {"origin": np.array([0.5, 0.1, 0.30]),
            "u_step": np.array([DU*np.cos(0.3), 0.0, DU*np.sin(0.3)]),
            "v_step": np.array([0.0, DV, 0.0]),
            "m": M, "n": N}
    tp = np.array([TILT["origin"] + i*TILT["u_step"] + j*TILT["v_step"]
                   for i, j in full])
    ti, tj, tres = volume_cell_indices(TILT, tp)
    check("tilted grid maps exactly", list(zip(ti.tolist(), tj.tolist())), full)
    check("no residual from the tilt", float(tres.max()), 0.0, 1e-12)

    print()


@_checked
def test_degenerate_grids_do_not_divide_by_zero():
    print("=== degenerate grids do not divide by zero ===")
    single = dict(GRID, m=1, u_step=None)
    si, sj, _ = volume_cell_indices(single, cell_positions([(0, 0), (0, 3)]))
    check("a one-column grid puts everything in column 0", si.tolist(), [0, 0])
    check("and still resolves the rows", sj.tolist(), [0, 3])

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
    print("all regrid placement checks passed")


_SECTIONS = ["test_a_full_raster_maps_onto_the_grid", "test_a_missing_trace_does_not_move_the_ones_after_it", "test_a_whole_missing_pass_leaves_a_row_of_holes", "test_positions_off_the_grid_are_identifiable", "test_half_cell_offsets_are_reported", "test_a_tilted_skewed_grid_follows_the_basis", "test_degenerate_grids_do_not_divide_by_zero"]


if __name__ == "__main__":
    sys.exit(_main())
