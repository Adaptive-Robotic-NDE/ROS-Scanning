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
# File:  specimen_scene_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests the planning-scene half of specimen.dpi against a stubbed
#   move_group.
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
"""Check the planning-scene half of specimen.dpi against a stubbed move_group.

What has to hold:

  * the specimen reaches the scene AS A MESH, in world, with every triangle.
    With no specimen in the planning scene, every collision check the arm makes
    -- including the one tari_cpp_mtc's raster stage does -- is a check against
    an empty world.
  * the allowed-collision matrix is READ, modified and written back. MoveIt's
    diff handling REPLACES the matrix when a diff carries one, so writing a
    two-entry matrix would delete every self-collision pair the SRDF declares
    and the arm would be free to fold into itself.
  * the contact declaration is narrow: the eddy-current probe may touch the
    specimen, the Keyence head and the Azure may not.
  * the READ-BACK is the verdict, not the acknowledgement. A diff carrying a
    6000-triangle mesh is applied even when the reply does not arrive within the
    30 s timeout, which this middleware does not guarantee.
  * a REFUSAL is still an error. "No answer" and "no" are different.

    python3 specimen_scene_test.py
"""
import functools
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ros_msg_stubs

# ------------------------------------------------------------------------------
# Just enough of the message packages to build what specimen.dpi builds
# ------------------------------------------------------------------------------


def _msg(name, **defaults):
    def init(self, **kwargs):
        for key, value in defaults.items():
            setattr(self, key, value() if callable(value) else value)
        for key, value in kwargs.items():
            setattr(self, key, value)
    return type(name, (), {"__init__": init})


Point = _msg("Point", x=0.0, y=0.0, z=0.0)
Quaternion = _msg("Quaternion", x=0.0, y=0.0, z=0.0, w=0.0)
Pose = _msg("Pose", position=Point, orientation=Quaternion)
Header = _msg("Header", frame_id="")
MeshTriangle = _msg("MeshTriangle", vertex_indices=list)
Mesh = _msg("Mesh", vertices=list, triangles=list)
SolidPrimitive = _msg("SolidPrimitive", type=0, dimensions=list)
SolidPrimitive.BOX = 1
CollisionObject = _msg("CollisionObject", header=Header, id="", meshes=list,
                       mesh_poses=list, primitives=list, primitive_poses=list,
                       operation=0)
CollisionObject.ADD = 0
CollisionObject.REMOVE = 1
PlanningSceneWorld = _msg("PlanningSceneWorld", collision_objects=list)
AllowedCollisionEntry = _msg("AllowedCollisionEntry", enabled=list)
AllowedCollisionMatrix = _msg("AllowedCollisionMatrix", entry_names=list,
                              entry_values=list)
RobotState = _msg("RobotState", is_diff=False)
PlanningScene = _msg("PlanningScene", is_diff=False, robot_state=RobotState,
                     world=PlanningSceneWorld,
                     allowed_collision_matrix=AllowedCollisionMatrix)
PlanningSceneComponents = _msg("PlanningSceneComponents", components=0)
PlanningSceneComponents.WORLD_OBJECT_NAMES = 8
PlanningSceneComponents.ALLOWED_COLLISION_MATRIX = 128

# specimen.dpi imports these INSIDE the functions under test, so the names have
# to resolve in sys.modules when those functions are CALLED. Register them
# through ros_msg_stubs.ensure_msgs() and never by assigning sys.modules
# directly: sys.modules is process-wide, so an assignment here replaces whatever
# another suite in the directory installed, for the rest of the pytest run.
#
# ensure_msgs() only ADDS, so nobody loses a class they installed. Nothing here
# depends on WHICH geometry_msgs.msg is in force: specimen.dpi only reads
# .position/.orientation off whatever Pose it gets, and every check below reads
# the CollisionObject and matrix classes defined above, which are this file's.
ros_msg_stubs.ensure_msgs(
    pkg="moveit_msgs", CollisionObject=CollisionObject,
    PlanningScene=PlanningScene, PlanningSceneWorld=PlanningSceneWorld,
    PlanningSceneComponents=PlanningSceneComponents,
    AllowedCollisionMatrix=AllowedCollisionMatrix,
    AllowedCollisionEntry=AllowedCollisionEntry)
ros_msg_stubs.ensure_msgs(pkg="shape_msgs", Mesh=Mesh,
                          MeshTriangle=MeshTriangle,
                          SolidPrimitive=SolidPrimitive)
ros_msg_stubs.ensure_msgs(pkg="geometry_msgs", Pose=Pose, Point=Point,
                          Quaternion=Quaternion,
                          PoseArray=_msg("PoseArray", poses=list))


class MoveGroup(object):
    """A planning scene that behaves the way MoveIt's does.

    In particular: a diff carrying a non-empty allowed_collision_matrix
    REPLACES the matrix, which is the whole reason specimen.dpi reads it first.
    """

    def __init__(self, links=("ec_probe", "probe_holder", "wrist_3_link"),
                 answer=True):
        self.world = {}
        self.names = list(links)
        self.values = [[False]*len(self.names) for _ in self.names]
        # One pre-existing pair, as an SRDF would declare: the probe holder and
        # the wrist are allowed to touch each other.
        self.values[1][2] = self.values[2][1] = True
        # Which names the SRDF knows. Everything else is a world object, and
        # move_group leaves a world object out of the matrix MESSAGE entirely
        # once nothing is allowed to touch it -- so "absent" is the answer to
        # "may anything touch it?", not evidence that a diff was lost. Reading
        # that absence as a failure is the trap allow_specimen_contact() has to
        # avoid, so the stub reproduces it.
        self.robot_links = set(self.names)
        self.answer = answer
        self.applies = 0

    def __call__(self, key, fields=None, timeout=None, wait=None):
        fields = fields or {}
        if key == "get_scene":
            scene = PlanningScene()
            scene.world = PlanningSceneWorld()
            scene.world.collision_objects = list(self.world.values())
            acm = AllowedCollisionMatrix()
            shown = [i for i, n in enumerate(self.names)
                     if n in self.robot_links or any(self.values[i])]
            acm.entry_names = [self.names[i] for i in shown]
            acm.entry_values = [
                AllowedCollisionEntry(enabled=[self.values[i][j] for j in shown])
                for i in shown]
            scene.allowed_collision_matrix = acm
            return types.SimpleNamespace(scene=scene)
        if key != "apply_scene":
            raise KeyError(key)
        self.applies += 1
        scene = fields["scene"]
        for obj in scene.world.collision_objects:
            if obj.operation == CollisionObject.REMOVE:
                self.world.pop(obj.id, None)
            else:
                self.world[obj.id] = obj
        acm = scene.allowed_collision_matrix
        if acm.entry_names:
            # REPLACES, as MoveIt does.
            self.names = list(acm.entry_names)
            self.values = [list(row.enabled) for row in acm.entry_values]
        if self.answer is None:
            raise RuntimeError("/apply_planning_scene did not answer within 30 s")
        return types.SimpleNamespace(success=bool(self.answer))


def load(move_group, extra=None):
    src = open(os.path.join(HERE, "specimen.dpi")).read()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("global "))
    ns = {"__name__": "specimen_scene_test", "np": np,
          "call_ros_service": move_group,
          "read_ply_mesh": read_ply_mesh,
          "TOOLING_CONTACT_LINKS": ("ec_probe", "probe_holder"),
          "SPECIMEN_SOURCE": "file", "SPECIMEN_CLEAN_MESH": False,
          "NOETHER_MESH": None, "SPECIMEN_REGION": (0.7, -0.3, 0.9, -0.2),
          "check_region": lambda r, what="", quiet=False: r,
          "work_volume": lambda: ("world", (0.6, -0.45, 0.93), (1.1, 0.05, 1.25))}
    # The stage wrappers report through progress.dpi, which the config includes
    # ahead of specimen.dpi; load it the same way here. Silenced, because this
    # test is about what reaches move_group and progress.dpi has its own test.
    _progress = open(os.path.join(HERE, "progress.dpi")).read()
    exec(compile("\n".join(l for l in _progress.splitlines()
                           if not l.startswith("global ")),
                 "progress.dpi", "exec"), ns)
    ns["PROGRESS_ENABLED"] = False
    ns.update(extra or {})
    exec(compile(src, "specimen.dpi", "exec"), ns)
    return ns


# The PLY reader is the real one, out of noether_path.dpi.
_np_src = open(os.path.join(HERE, "noether_path.dpi")).read()
_np_ns = {"__name__": "ply", "np": np}
exec(compile("\n".join(l for l in _np_src.splitlines()
                       if not l.startswith("global ")),
             "noether_path.dpi", "exec"), _np_ns)
read_ply_mesh = _np_ns["read_ply_mesh"]


def write_ply(path, n=12):
    """A little dome, so there is a real mesh with real triangles."""
    xs = np.linspace(0.75, 0.90, n + 1)
    ys = np.linspace(-0.30, -0.20, n + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = 0.96 + 0.02*np.cos(8*(xx - 0.825))*np.cos(20*(yy + 0.25))
    verts = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])
    tris = []
    for i in range(n):
        for j in range(n):
            a = i*(n + 1) + j
            tris.append((a, a + 1, a + n + 1))
            tris.append((a + 1, a + n + 2, a + n + 1))
    with open(path, "w") as handle:
        handle.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(verts))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("element face %d\n"
                     "property list uchar int vertex_indices\nend_header\n"
                     % len(tris))
        for v in verts:
            handle.write("%.6f %.6f %.6f\n" % tuple(v))
        for t in tris:
            handle.write("3 %d %d %d\n" % t)
    return len(verts), len(tris)


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok    %s" % (name,))
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILURES.append(name)


def _checked(fn):
    """Turn the checks a section ran into a pytest pass or failure.

    check() records rather than raises, so that a standalone run reports every
    failure in one pass. Under pytest that would be a test that passes while
    printing FAIL, so the wrapper asserts on whatever the section added.
    _main() calls fn.__wrapped__ to keep the standalone behaviour.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mark = len(FAILURES)
        fn(*args, **kwargs)
        failed = FAILURES[mark:]
        assert not failed, "%d check(s) failed: %s" % (len(failed),
                                                       "; ".join(failed))
    return wrapper


import tempfile
MESH = os.path.join(tempfile.mkdtemp(), "dome.ply")
NV, NT = write_ply(MESH)


@_checked
def test_the_specimen_reaches_the_scene():
    print("the specimen reaches the scene")
    mg = MoveGroup()
    ns = load(mg)
    state = ns["specimen_to_planning_scene"](mesh=MESH)
    obj = mg.world["specimen"]
    check("it is there, under one name", list(mg.world) == ["specimen"], list(mg.world))
    check("as a MESH, not a bounding box",
          len(obj.meshes) == 1 and not obj.primitives,
          (len(obj.meshes), len(obj.primitives)))
    check("with every triangle", len(obj.meshes[0].triangles) == NT,
          (len(obj.meshes[0].triangles), NT))
    check("in world, where the reconstructions already are",
          obj.header.frame_id == "world", obj.header.frame_id)
    check("added, so a second reconstruction replaces it",
          obj.operation == CollisionObject.ADD, obj.operation)
    check("and the bounding box is reported for a human to sanity-check",
          abs(state["bbox"][0][0] - 0.75) < 1e-6
          and abs(state["bbox"][1][2] - 0.98) < 1e-3, state["bbox"])

    ns["specimen_to_planning_scene"](mesh=MESH)
    check("applying twice leaves ONE object",
          list(mg.world) == ["specimen"], list(mg.world))


@_checked
def test_a_mesh_too_big_is_reduced():
    print("a mesh too big for the planner is reduced, and says so")
    mg = MoveGroup()
    ns = load(mg)
    state = ns["specimen_to_planning_scene"](mesh=MESH, max_triangles=40)
    check("it is reduced to fit", state["triangles"] <= 40, state["triangles"])
    check("the grid it was reduced on is reported",
          state["cluster_grid_m"] > 0.0, state["cluster_grid_m"])
    check("and it is still a surface, not a point cloud",
          state["triangles"] > 0 and state["vertices"] > 2, state)


@_checked
def test_the_contact_declaration():
    print("the contact declaration")
    mg = MoveGroup()
    ns = load(mg)
    ns["specimen_to_planning_scene"](mesh=MESH)
    ns["allow_specimen_contact"](["ec_probe"])
    i = mg.names.index("specimen")
    j = mg.names.index("ec_probe")
    check("the probe may touch the specimen",
          mg.values[i][j] and mg.values[j][i], (mg.values[i][j], mg.values[j][i]))
    k = mg.names.index("wrist_3_link")
    check("and nothing else may",
          not mg.values[i][k], mg.values[i][k])
    check("the matrix grew rather than being replaced wholesale",
          len(mg.names) == 4 and len(mg.values) == 4,
          (len(mg.names), len(mg.values)))
    p = mg.names.index("probe_holder")
    check("the pairs the SRDF declared are still there",
          mg.values[p][k] and mg.values[k][p], (mg.values[p][k], mg.values[k][p]))
    ns["allow_specimen_contact"](["ec_probe"], allow=False)
    check("and the declaration can be taken back",
          not mg.values[i][j], mg.values[i][j])
    check("clearing leaves the object out of the matrix message, and that is the "
          "answer rather than a failure",
          "specimen" not in ns["call_ros_service"](
              "get_scene", {"components": None}).scene.allowed_collision_matrix.entry_names)
    check("and specimen_contact_allowed says nothing may touch it",
          ns["specimen_contact_allowed"] == (), ns["specimen_contact_allowed"])

    try:
        ns["allow_specimen_contact"](["ec_prob"])
        check("a misspelled link name is an error, not a silent no-op", False)
    except ValueError as e:
        check("a misspelled link name is an error, not a silent no-op",
              "no link" in str(e), str(e)[:70])


@_checked
def test_removal():
    print("removal")
    mg = MoveGroup()
    ns = load(mg)
    ns["specimen_to_planning_scene"](mesh=MESH)
    ns["remove_specimen_from_scene"]()
    check("the object goes", "specimen" not in mg.world, list(mg.world))


@_checked
def test_the_read_back_is_the_verdict():
    print("the read-back is the verdict, not the acknowledgement")
    mg = MoveGroup(answer=None)          # applies the diff, never replies
    ns = load(mg)
    state = ns["specimen_to_planning_scene"](mesh=MESH)
    check("a lost reply is tolerated when the scene really holds it",
          state["triangles"] == NT and "specimen" in mg.world)

    mg = MoveGroup(answer=False)         # replies "no"
    ns = load(mg)
    try:
        ns["specimen_to_planning_scene"](mesh=MESH)
        check("a REFUSAL is still an error", False)
    except RuntimeError as e:
        check("a REFUSAL is still an error", "REFUSED" in str(e), str(e)[:70])


    class Silent(MoveGroup):
        """Applies nothing and replies yes: an acknowledgement that is a lie."""

        def __call__(self, key, fields=None, timeout=None, wait=None):
            if key == "apply_scene":
                return types.SimpleNamespace(success=True)
            return MoveGroup.__call__(self, key, fields, timeout, wait)


    mg = Silent()
    ns = load(mg)
    try:
        ns["specimen_to_planning_scene"](mesh=MESH)
        check("a scene that says yes and holds nothing is caught", False)
    except RuntimeError as e:
        check("a scene that says yes and holds nothing is caught",
              "not in the scene" in str(e), str(e)[:70])


@_checked
def test_the_primitive_for_the_stage_before_a_mesh():
    print("the primitive, for the stage before a mesh exists")
    mg = MoveGroup()
    ns = load(mg)
    state = ns["specimen_box_to_planning_scene"](region=(0.75, -0.30, 0.90, -0.20),
                                                 height=0.02, support_z=0.9286)
    box = mg.world["specimen"]
    check("it is a box", box.primitives and box.primitives[0].type == SolidPrimitive.BOX)
    check("the size comes from the region and the measured height",
          abs(box.primitives[0].dimensions[0] - 0.15) < 1e-9
          and abs(box.primitives[0].dimensions[2] - 0.02) < 1e-9,
          box.primitives[0].dimensions)
    check("and it sits ON the support, not through it",
          abs(box.primitive_poses[0].position.z - (0.9286 + 0.01)) < 1e-9,
          box.primitive_poses[0].position.z)
    try:
        ns["specimen_box_to_planning_scene"](region=(0.75, -0.30, 0.90, -0.20),
                                             support_z=0.9286)
        check("a box with no measured height is refused", False)
    except ValueError as e:
        check("a box with no measured height is refused", "needs a height" in str(e))


@_checked
def test_obtaining_a_mesh_puts_it_in_the_scene():
    print("obtaining a mesh is what puts it in the scene; no other caller does")
    mg = MoveGroup()
    ns = load(mg, {"NOETHER_MESH": MESH})
    mesh, x3d = ns["obtain_specimen_mesh"](source="file", x3d="unused.x3d")
    check("the mesh the scan will be planned over is now a world object",
          "specimen" in mg.world and mesh == MESH, list(mg.world))

    mg = MoveGroup()
    ns = load(mg, {"NOETHER_MESH": MESH})
    ns["obtain_specimen_mesh"](source="file", x3d="unused.x3d", to_scene=False)
    check("to_scene=False is the deliberate opt-out", not mg.world, list(mg.world))


    def _no_scene(key, fields=None, timeout=None, wait=None):
        raise RuntimeError("no /get_planning_scene service (moveit_msgs.srv/"
                           "GetPlanningScene). It is started by tari_bringup")


    ns = load(_no_scene, {"NOETHER_MESH": MESH})
    ns["obtain_specimen_mesh"](source="file", x3d="unused.x3d")
    check("a tree with no cell says so and carries on", True)
    try:
        ns["obtain_specimen_mesh"](source="file", x3d="unused.x3d", to_scene=True)
        check("but to_scene=True with no planning scene is an error", False)
    except RuntimeError as e:
        check("but to_scene=True with no planning scene is an error",
              "get_planning_scene" in str(e), str(e)[:60])

    print()


def _main():
    """Every section, in order, reporting all failures rather than the first."""
    for _name in _SECTIONS:
        # __wrapped__ is the undecorated section: it records failures
        # instead of raising, so one run reports all of them.
        _fn = globals()[_name]
        getattr(_fn, "__wrapped__", _fn)()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all checks passed")


_SECTIONS = ["test_the_specimen_reaches_the_scene", "test_a_mesh_too_big_is_reduced", "test_the_contact_declaration", "test_removal", "test_the_read_back_is_the_verdict", "test_the_primitive_for_the_stage_before_a_mesh", "test_obtaining_a_mesh_puts_it_in_the_scene"]


if __name__ == "__main__":
    sys.exit(_main())
