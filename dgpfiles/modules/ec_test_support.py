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
# File:  ec_test_support.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Locates the compiled eddy-current conditioning function for host-side
#   tests, or reports why it is absent.
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
"""Find the compiled EC conditioning function for a host-side test, or say so.

WHY THIS EXISTS. ec_acquisition.dpi opens with

    from spatialnde2_ec_functions import ec_condition_function

and that package is a COMPILED Cython extension linked against spatialnde2 --
the impedance-plane conditioning (null, rotation, per-axis gain), which is in
C++ because every parameter is a recording and retuning re-runs the math over
thousands of traces. Importing it from its source directory does nothing: the
__init__ needs the built .so beside it.

So any host-side test that loads ec_acquisition.dpi needs the extension, even
when the thing it is testing has nothing to do with conditioning. Without this
helper such a test dies at the .dpi's import line with

    ModuleNotFoundError: No module named 'spatialnde2_ec_functions'

which reads as "a dependency is missing from the outside world". It is not
missing. It is in this workspace at spatialnde2-ec-functions/, it builds on the
host (Arch, python3.14) in about a minute, and it is installed in the container.
What it is not is *installed on this host by workspace setup* -- dgpy_install.sh
does not list it.

THREE OUTCOMES, in order of preference:

  installed   plain `import spatialnde2_ec_functions` works. Nothing to do; the
              tests run against the real compiled math. This is the container,
              and the host once the package is installed or built.

  in-tree     not installed, but the workspace has a copy that has been built
              in place (`python3 setup.py build_ext --inplace`, which is the
              build its own README documents). Its directory goes on sys.path
              and the tests still run against the REAL compiled math.

  stub        neither. A narrow stand-in module is installed under the real
              name so the .dpi can be loaded, and the caller is told, loudly.
              The stub carries no arithmetic at all and REFUSES to be
              instantiated as a math function -- there is no Python math-
              function support in spatialnde2 (see the package README), so
              anything that pretended to condition data here would be testing
              the pretence. A test that can still do its job with the stub says
              so; a test that cannot must skip that part out loud.

The stub pattern is the one noether_roll_test.py uses for geometry_msgs: replace
a dependency the test does not exercise, not one it does.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

#: Directory name of the package inside a workspace checkout.
SOURCE_DIRNAME = "spatialnde2-ec-functions"

#: How far up from here to look for it. modules/ -> dgpfiles/ -> ROS-Scanning/
#: -> the workspace root, so four is already one more than needed.
_SEARCH_DEPTH = 6

BUILD_HINT = (
    "build it in place:  cd %s && python3 setup.py build_ext --inplace\n"
    "    or install it:  ./dgpy_install.sh --with-extras\n"
    "    (spatialnde2 must be installed first -- its setup.py reads the\n"
    "     compile flags out of the installed package. On the ROS container,\n"
    "     prefix the build with SETUPTOOLS_USE_DISTUTILS=stdlib; on Python\n"
    "     3.12+ that variable breaks setuptools instead. See its README.)"
    % (SOURCE_DIRNAME,))


def workspace_source_dir():
    """The spatialnde2-ec-functions checkout above us, or None."""
    directory = HERE
    for _ in range(_SEARCH_DEPTH):
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
        candidate = os.path.join(directory, SOURCE_DIRNAME)
        if os.path.isfile(os.path.join(candidate, "spatialnde2_ec_functions",
                                       "__init__.py")):
            return candidate
    return None


class StubECConditionFunction(object):
    """Stands where the compiled math function would be, and refuses to pretend.

    It is deliberately not a duck-typed math_function. The conditioning is
    tested by the package's own suite against the real C++; a Python imitation
    wired into the recording database would produce numbers that prove nothing
    about the function a scan actually runs.
    """

    is_stub = True

    def instantiate(self, *args, **kwargs):
        raise RuntimeError(
            "ec_condition_function is a STUB in this test run -- the compiled "
            "spatialnde2_ec_functions extension is not available, so the "
            "impedance-plane conditioning cannot be instantiated as a math "
            "function. Whatever asked for it needs the real thing:\n    "
            + BUILD_HINT)

    def __repr__(self):
        return "<stub ec_condition_function (not the compiled math function)>"


def ensure_ec_functions():
    """Make `spatialnde2_ec_functions` importable one way or another.

    Returns a dict with:
        real    True if the compiled extension is what got imported
        how     "installed", "in-tree" or "stub"
        origin  the file it came from, or why there is none
    """
    try:
        import spatialnde2_ec_functions as mod
    except ImportError:
        pass
    else:
        return {"real": True, "how": "installed", "origin": mod.__file__}

    source = workspace_source_dir()
    if source is not None:
        sys.path.insert(0, source)
        try:
            import spatialnde2_ec_functions as mod
        except ImportError as e:
            sys.path.remove(source)
            # Leave nothing half-imported: a failed `from .ec_condition import`
            # can still leave the parent package in sys.modules, and the next
            # import would then succeed and hand back a module with no
            # ec_condition_function on it.
            sys.modules.pop("spatialnde2_ec_functions", None)
            why = "%s is present but not built for this interpreter (%s)" % (
                source, e)
        else:
            return {"real": True, "how": "in-tree", "origin": mod.__file__}
    else:
        why = "no %s directory above %s" % (SOURCE_DIRNAME, HERE)

    stub = types.ModuleType("spatialnde2_ec_functions")
    stub.__file__ = "<stub installed by %s>" % (os.path.basename(__file__),)
    stub.ec_condition_function = StubECConditionFunction()
    stub.is_stub = True
    sys.modules["spatialnde2_ec_functions"] = stub
    return {"real": False, "how": "stub", "origin": why}


def print_status(status, covers_conditioning):
    """Say which of the three outcomes happened, before any test output.

    `covers_conditioning` is whether the calling test needs the arithmetic. It
    decides whether a stub is a footnote or a hole, and the caller has to state
    it rather than leaving the reader to guess.
    """
    if status["real"]:
        print("  ec_condition_function: REAL compiled extension (%s)"
              % (status["how"],))
        print("    %s" % (status["origin"],))
        return
    print("  " + "!" * 72)
    print("  !! ec_condition_function is STUBBED -- the compiled "
          "spatialnde2_ec_functions")
    print("  !! extension is not available here.")
    print("  !! %s" % (status["origin"],))
    if covers_conditioning:
        print("  !! THIS TEST CANNOT CHECK THE CONDITIONING MATH IN THIS STATE.")
    else:
        print("  !! This test does not use the conditioning math, so what it "
              "checks is")
        print("  !! unaffected -- but ec_conditioning_test.py needs the real "
              "extension.")
    print("  !! To get it: " + BUILD_HINT.replace("\n", "\n  !! "))
    print("  " + "!" * 72)
