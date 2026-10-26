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
# File:  test_data.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Locates the recorded scan fixtures the test suites grade against.
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
"""Locate the recorded scan fixtures the test suites grade against.

The datasets live in their own repository (test_data_staging) because they are
large and are shared by several suites. Nothing here hardcodes a path to them:
`root()` resolves `$TARI_TEST_DATA` first, then looks for a `test_data`
directory by walking up from this file, so the suites work from any checkout
layout.

When no directory resolves, `path()` returns None and the caller SKIPS. A skip
is not a pass -- report the dataset that was missing so a green run with no real
data cannot be mistaken for a verified one.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

#: Directory name looked for when walking up from this file.
DIRNAME = "test_data"

#: How far up to walk. The suites sit three levels inside their repository, and
#: the fixture checkout is normally a sibling of that repository.
MAX_DEPTH = 8


def root():
    """The fixture directory, or None when it is not on this machine."""
    env = os.environ.get("TARI_TEST_DATA")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    d = HERE
    for _ in range(MAX_DEPTH):
        candidate = os.path.join(d, DIRNAME)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


#: First bytes of a git-lfs pointer file. A checkout made without the lfs
#: filters configured contains these instead of the data, and the paths still
#: exist -- so a fixture has to be checked for content, not just for presence,
#: or a suite reads a pointer where it expects a mesh.
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/"


def is_pointer(p):
    """True when p is a git-lfs pointer rather than the file itself."""
    try:
        with open(p, "rb") as fh:
            return fh.read(len(_LFS_MAGIC)) == _LFS_MAGIC
    except OSError:
        return False


def path(*parts):
    """A fixture path, or None when the fixture is absent or unfetched.

    Returning None rather than raising lets a suite skip one section and still
    run everything that needs no recorded data.
    """
    base = root()
    if base is None:
        return None
    p = os.path.join(base, *parts)
    if not os.path.exists(p) or is_pointer(p):
        return None
    return p


def why_missing(*parts):
    """A one-line explanation for a skip message."""
    base = root()
    if base is None:
        return ("no fixture directory: set TARI_TEST_DATA, or check out "
                "test_data_staging as %s beside this repository" % (DIRNAME,))
    p = os.path.join(base, *parts)
    if os.path.exists(p) and is_pointer(p):
        return ("%s is a git-lfs pointer, not content: run "
                "`git -C %s lfs pull`" % (os.path.join(*parts), base))
    return "%s is not in %s" % (os.path.join(*parts), base)
