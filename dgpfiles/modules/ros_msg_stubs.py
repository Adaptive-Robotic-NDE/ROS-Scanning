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
# File:  ros_msg_stubs.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Stand-in ROS message classes shared by the host-side test suites.
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
"""Stand-in ROS message classes, so the host-side suites stop fighting each other.

WHY THIS FILE EXISTS. The .dpi files in this directory import geometry_msgs at
module scope (toolpath_segments.dpi) and inside functions (cell_capture_motion.dpi,
specimen.dpi, mesh_reconstruction.dpi). The suites that exercise them run on the
host with no ROS, so something has to be in sys.modules under that name. The trap
is each suite writing its OWN module there at IMPORT time, with a bare

    sys.modules["geometry_msgs.msg"] = <a stub>

pytest imports every test file in a directory before it runs any test, so the
last file collected wins and every other file then runs against a stub it did not
write.

That is not a tidiness argument, because those stubs typically carry a Pose with
NO FIELDS. A suite can then pass EVERY case under bare `pytest` while measuring
nothing -- a field-less Pose makes _cap_pose() raise, _ik() swallows the
exception, and the guard under test is never consulted -- and pass cleanly
standalone as well. Only mutation testing shows the difference.

So this module owns the stubs, and two rules keep them from racing:

  * ensure_msgs() only ever ADDS. The package and its .msg submodule go in with
    setdefault, and a name is written onto an existing module only when it is
    MISSING. A suite that ran first keeps the classes it put there.
  * the classes have REAL FIELDS. Whoever lands first, everybody downstream gets
    a Pose with .position and .orientation, so a suite cannot be blinded by
    another suite's placeholder.

A suite that must have ITS OWN class reach the code under test -- because the
import it depends on is inside a function and therefore resolves at CALL time,
which no namespace injection can reach -- uses override_msgs()/restore_msgs()
around the calls and leaves sys.modules exactly as it found it.

ros_stub_hygiene_test.py is the lint that fails if a *_test.py goes back to
assigning sys.modules at module scope.

Not a test file and not a dataguzzler-python module: plain host Python, imported
by the suites next to it.
"""
import sys
import types

__all__ = ["Header", "Point", "Pose", "PoseArray", "PoseStamped", "PointStamped",
           "Quaternion", "Vector3", "WrenchStamped", "DEFAULT_NAMES",
           "ensure_msgs", "override_msgs", "restore_msgs", "msg_module"]


class Header(object):
    def __init__(self, frame_id="", stamp=None, seq=0):
        self.frame_id = frame_id
        self.stamp = stamp
        self.seq = seq


class Vector3(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return "%s(x=%r, y=%r, z=%r)" % (type(self).__name__, self.x, self.y, self.z)


class Point(Vector3):
    pass


class Quaternion(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __repr__(self):
        return "Quaternion(x=%r, y=%r, z=%r, w=%r)" % (self.x, self.y, self.z, self.w)


class Pose(object):
    """A Pose WITH FIELDS. The whole point of this module.

    geometry_msgs/Pose default-constructs its members, and so does this: code
    under test writes pose.position.x, never a constructor argument.
    """

    def __init__(self, position=None, orientation=None):
        self.position = Point() if position is None else position
        self.orientation = Quaternion() if orientation is None else orientation

    def __repr__(self):
        return "Pose(position=%r, orientation=%r)" % (self.position, self.orientation)


class PoseStamped(object):
    def __init__(self, header=None, pose=None):
        self.header = Header() if header is None else header
        self.pose = Pose() if pose is None else pose


class PoseArray(object):
    def __init__(self, header=None, poses=None):
        self.header = Header() if header is None else header
        self.poses = [] if poses is None else list(poses)


class PointStamped(object):
    def __init__(self, header=None, point=None):
        self.header = Header() if header is None else header
        self.point = Point() if point is None else point


class WrenchStamped(object):
    def __init__(self, header=None):
        self.header = Header() if header is None else header
        self.wrench = types.SimpleNamespace(force=Vector3(), torque=Vector3())


#: Everything the .dpi files in this directory import from geometry_msgs.msg.
#: All of them, every time: a name that is present costs nothing, and a name that
#: is missing is an ImportError from inside whichever function reaches for it.
DEFAULT_NAMES = {
    "Header": Header,
    "Point": Point,
    "PointStamped": PointStamped,
    "Pose": Pose,
    "PoseArray": PoseArray,
    "PoseStamped": PoseStamped,
    "Quaternion": Quaternion,
    "Vector3": Vector3,
    "WrenchStamped": WrenchStamped,
}


def msg_module(pkg="geometry_msgs"):
    """The module currently installed as <pkg>.msg, or None."""
    return sys.modules.get(pkg + ".msg")


def ensure_msgs(pkg="geometry_msgs", **classes):
    """Make <pkg>.msg importable without taking it away from another suite.

    setdefault for the two sys.modules entries, and setattr only for names that
    are MISSING, so whichever suite was imported first keeps its own classes and
    every suite still finds every name it imports.

    Returns the <pkg>.msg module that is actually in force -- which may be
    another suite's. Nothing in this directory should depend on WHICH; a suite
    that genuinely needs its own class to reach the code under test calls
    override_msgs() instead.
    """
    names = dict(DEFAULT_NAMES)
    names.update(classes)
    parent = sys.modules.setdefault(pkg, types.ModuleType(pkg))
    msg = sys.modules.setdefault(pkg + ".msg", types.ModuleType(pkg + ".msg"))
    for name, cls in names.items():
        if not hasattr(msg, name):
            setattr(msg, name, cls)
    if getattr(parent, "msg", None) is not msg:
        parent.msg = msg
    return msg


def override_msgs(pkg="geometry_msgs", **classes):
    """Install a PRIVATE <pkg>.msg carrying `classes`, and return a restore token.

    For the one thing ensure_msgs() cannot do: a `from geometry_msgs.msg import
    Pose, Quaternion` that lives INSIDE the function under test resolves against
    sys.modules when the function is CALLED, so no amount of namespace injection
    reaches it -- the entry itself has to be right at that moment. Wrap the calls,
    then hand the token to restore_msgs(); sys.modules ends up exactly as it was,
    so this is not the unconditional overwrite the lint forbids.
    """
    names = dict(DEFAULT_NAMES)
    names.update(classes)
    mine = types.ModuleType(pkg + ".msg")
    for name, cls in names.items():
        setattr(mine, name, cls)
    parent = sys.modules.setdefault(pkg, types.ModuleType(pkg))
    token = (pkg, pkg in sys.modules, sys.modules.get(pkg),
             pkg + ".msg" in sys.modules, sys.modules.get(pkg + ".msg"),
             hasattr(parent, "msg"), getattr(parent, "msg", None))
    sys.modules[pkg + ".msg"] = mine
    parent.msg = mine
    return token, mine


def restore_msgs(token):
    """Undo one override_msgs(), entry for entry."""
    if isinstance(token, tuple) and len(token) == 2:
        token = token[0]
    (pkg, had_pkg, old_pkg, had_msg, old_msg, had_attr, old_attr) = token
    if had_msg:
        sys.modules[pkg + ".msg"] = old_msg
    else:
        sys.modules.pop(pkg + ".msg", None)
    if had_pkg:
        sys.modules[pkg] = old_pkg
        if had_attr:
            old_pkg.msg = old_attr
        else:
            try:
                del old_pkg.msg
            except AttributeError:
                pass
    else:
        sys.modules.pop(pkg, None)
