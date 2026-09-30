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
# File:  scan_channel_lifetime_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Proves a scan's recording-database channels can be taken, handed back and
#   taken again, so a second scan in one session does not abort the process.
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
"""Can the scan run twice in one process?

WHAT THIS PROTECTS. A channel belongs to the reserved_channel that defined it.
Defining it again while the old reservation is still current does NOT raise at
the call -- define_channel() hands back a reserved_channel that is no longer the
owner, and the transaction fails later, at realize time:

    realize_transaction: provided reserved_channel for /ec/current by main is
    not current (current owner is main)

Under the timed transaction manager -- which the eddy-current config requires,
because correlating readings IS the timestamp ordering -- realize runs on the
manager's own thread. Nothing there catches it, so it is terminate() and SIGABRT:
the whole process, mid-scan, probe on the part, no traceback.

So every channel a scan takes has to be released before the next scan takes it,
and that cannot be checked by reading the code -- the failure is a signal on
another thread, a long way from the line that caused it. These tests take each
channel twice against a real recording database and let it fail honestly.

THE MODEL TREE IS THE EXCEPTION. x3d_load_geometry() reserves the channels of
the tree it loads into and keeps those reservations private, so they can never be
handed back; a projector rebuilt on the same tree fails the same way. The fix
there is the other direction -- a fresh tree per load -- and the third test is
what holds that in place.

Runs on the host. No ROS, no hardware, no container.
"""

import os
import sys
import tempfile

import numpy as np
import spatialnde2 as snde

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

POSE_FIELDS = 7


def _load(*dpi_names):
    """Execute .dpi sources into one namespace, as the dgpy loader would.

    A .dpi is a function body, not a module: `global` is meaningless at exec
    scope, so those lines are dropped and the shared namespace does the same job.
    """
    ns = {}
    exec("import os, sys, threading, time\n"
         "import numpy as np\nimport spatialnde2 as snde\n", ns)
    for name in dpi_names:
        src = open(os.path.join(MODULE_DIR, name)).read()
        src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
        exec(compile(src, name, "exec"), ns)
    return ns


def _recdb():
    recdb = snde.recdatabase()
    snde.setup_cpu(recdb, [], 4)
    snde.setup_storage_manager(recdb)
    snde.setup_math_functions(recdb, [])
    recdb.startup()
    return recdb


def _seed_complex(recdb, chan, value=0.0):
    """Put one complex sample on a staging channel, as a reading would."""
    trans = recdb.start_transaction()
    ref = snde.create_ndarray_ref(trans, chan, snde.SNDE_RTN_COMPLEXFLOAT64)
    ref.allocate_storage([1], False)
    ref.data[:] = value
    ref.rec.mark_data_and_metadata_ready()
    trans.end_transaction().globalrev_wait().wait_complete()


def test_externally_owned_staging_channel_survives_two_scans():
    """The staging channel is taken, handed back and taken again.

    The accumulator is told to USE this channel rather than make its own, so
    TraceAccumulator.teardown() deliberately leaves it alone -- it releases only
    what it owns. That makes releasing it the caller's job, which is what
    run_ec_path_scan()'s _prepare() does and what this asserts.
    """
    ns = _load("trace_acquisition.dpi")
    TraceAccumulator = ns["TraceAccumulator"]
    recdb = _recdb()
    staging = "/ec/current"

    accum = None
    chan = None
    for scan in (1, 2):
        trans = recdb.start_transaction()
        if chan is not None:
            trans.release_channel(chan)
        chan = trans.define_channel(staging, "main")
        ref = snde.create_ndarray_ref(trans, chan, snde.SNDE_RTN_COMPLEXFLOAT64)
        ref.allocate_storage([1], False)
        ref.data[:] = 0
        ref.rec.mark_data_and_metadata_ready()
        # end_transaction() is where a stale reservation is caught, so this line
        # is the assertion: under the timed manager the same fault is a SIGABRT.
        trans.end_transaction().globalrev_wait().wait_complete()

        accum = TraceAccumulator(
            recdb, 4, 1, prefix="/ec", sample_period=None, time_offset=0.0,
            ascan_typenum=snde.SNDE_RTN_COMPLEXFLOAT64,
            ascan_path=staging, ascan_chan=chan)
        assert accum.own_ascan_channel is False, (
            "the accumulator thinks it owns the staging channel, so this test is "
            "not exercising the externally-owned case it exists for")
        _seed_complex(recdb, chan, 1 + 1j)
        accum.setup()
        accum.teardown()

    assert str(staging) in set(recdb.latest.list_recordings())


def test_accumulator_owned_channels_survive_two_scans():
    """The accumulator's own channels come back on teardown.

    Covers the pose channel, which is created unconditionally, and the staging
    channel in the case where the accumulator does own it.
    """
    ns = _load("trace_acquisition.dpi")
    TraceAccumulator = ns["TraceAccumulator"]
    recdb = _recdb()
    for scan in (1, 2):
        accum = TraceAccumulator(recdb, 4, 8, prefix="/ut",
                                 sample_period=1e-8, time_offset=0.0)
        assert accum.own_ascan_channel is True
        accum.setup()
        accum.teardown()
        assert accum.pose_chan is None, (
            "teardown left the pose channel reservation in place, so the next "
            "scan would redefine a channel it still holds")


def test_projector_can_be_rebuilt_on_a_new_surface():
    """A second SurfaceProjector on the same prefix, on a different surface.

    This is what a second automated scan does: each run writes its mesh to a new
    directory, so the surface path always differs and the projector is always
    rebuilt rather than restarted.

    It works only because each load gets its own model tree. Point two loads at
    one tree and end_transaction() fails on the tree's own channel -- which is
    reserved inside x3d_load_geometry() and never handed out, so no amount of
    releasing at this level can fix it.
    """
    ns = _load("surface_projection.dpi")
    SurfaceProjector = ns["SurfaceProjector"]
    write_plane_x3d = ns["write_plane_x3d"]
    recdb = _recdb()

    tmp = tempfile.mkdtemp(prefix="surfproj_")
    trees = []
    proj = None
    for scan in range(3):
        # A different file each time, as a real run's timestamped output is.
        path = write_plane_x3d(os.path.join(tmp, "plane%d.x3d" % scan),
                               (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))
        if proj is not None:
            proj.teardown()
        proj = SurfaceProjector(recdb, path, prefix="/ec/proj",
                                horizontal_pixels=16, vertical_pixels=16)
        trees.append(proj.model_tree)

    assert len(set(trees)) == 3, (
        "two loads shared a model tree (%s); x3d_load_geometry keeps that "
        "tree's reservations, so the second load cannot succeed" % (trees,))

    chans = set(recdb.latest.list_recordings())
    for tree in trees:
        assert snde.recdb_path_join(tree, "meshed") in chans

    # And the cheap path still works: same surface, fresh picture.
    proj.restart()


def test_projector_hands_back_every_fixed_name_channel():
    """Teardown releases everything whose channel name does not move.

    The model tree is generational and cannot be released; every other channel
    the projector defines has a fixed name, so any one of them left reserved
    blocks the next projector. The textured part is the easy one to miss -- it is
    created after the constructor's transaction, from its own method.
    """
    ns = _load("surface_projection.dpi")
    SurfaceProjector = ns["SurfaceProjector"]
    write_plane_x3d = ns["write_plane_x3d"]
    recdb = _recdb()
    tmp = tempfile.mkdtemp(prefix="surfproj_")
    path = write_plane_x3d(os.path.join(tmp, "plane.x3d"),
                           (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))

    proj = SurfaceProjector(recdb, path, prefix="/ut/proj",
                            horizontal_pixels=16, vertical_pixels=16)
    assert proj.proj_specimen_chan is not None, (
        "the textured part's channel was not kept, so teardown cannot release "
        "it and the next projector on this prefix will fail")
    proj.teardown()
    for attr in ("specimen_pose_chan", "probe_pose_chan", "data_chan",
                 "proj_specimen_chan", "graphics_chan"):
        assert getattr(proj, attr) is None, (
            "teardown left %s reserved" % (attr,))


def main():
    ok = True
    for fn in (test_externally_owned_staging_channel_survives_two_scans,
               test_accumulator_owned_channels_survive_two_scans,
               test_projector_can_be_rebuilt_on_a_new_surface,
               test_projector_hands_back_every_fixed_name_channel):
        try:
            fn()
            print("PASS  %s" % fn.__name__)
        except Exception as e:                                # noqa: BLE001
            ok = False
            print("FAIL  %s: %s" % (fn.__name__, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
