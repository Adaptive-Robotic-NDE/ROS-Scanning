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
# File:  ros_stub_hygiene_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that no suite takes sys.modules entries away from another.
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
"""No suite in this directory may take sys.modules away from another one.

WHY THE RULE EXISTS. The .dpi files here import ROS message packages, and the
suites that exercise them run on a host with no ROS, so something has to stand in.
A suite that installs its own stub by ASSIGNING sys.modules at module scope:

    sys.modules["geometry_msgs.msg"] = <my stub>          # <-- this

changes that entry for every other suite in the directory: pytest imports every
test file before it runs any test, so the file collected LAST wins and the rest
then run against a stub they did not write.

The failure mode is SILENT and looks like a green suite. A stub message class
with no fields makes a helper raise, an outer layer swallows the exception, the
code under test is handed nothing, and every case passes while measuring nothing
-- and the same file passes honestly when it is run on its own.

THE RULE. In a *_test.py, at module scope, do not assign sys.modules[<a ROS message
package>]. Use ros_msg_stubs.ensure_msgs(), which setdefaults the entry and fills
in only the names that are MISSING, so whoever got there first keeps their classes.
A suite that genuinely needs its own class to reach the code under test -- because
the import it depends on is inside the function and therefore resolves at CALL
time -- uses ros_msg_stubs.override_msgs()/restore_msgs() and leaves sys.modules
exactly as it found it. Both of those live in a helper module, not in a *_test.py,
so neither trips this lint.

setdefault is not flagged: it cannot take an entry away from anyone. An assignment
INSIDE a function is not flagged either -- it is scoped to a call, and the suites
that do it restore what they replaced.

Runs on the host with the standard library only: it parses, it does not import.

    python3 ros_stub_hygiene_test.py     # prints every file, exit 1 on a violation
    python3 -m pytest -q ros_stub_hygiene_test.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: A sys.modules key naming a ROS interface package. ROS message packages are
#: named <thing>_msgs (geometry_msgs, sensor_msgs, std_msgs, moveit_msgs, ...) with
#: a .msg/.srv/.action submodule, plus the _msgs_py helper packages. Anything else
#: -- "dataguzzler_python", "tari_bringup" -- is a suite standing in for its own
#: project's module and is nobody else's business.
def _is_ros_msg_key(key):
    head = key.split(".")[0]
    return head.endswith("_msgs") or head.endswith("_msgs_py")


def _module_level_nodes(tree):
    """Every node at MODULE scope: the body, and inside if/for/while/try/with.

    A function or class body is skipped -- an assignment there happens when it is
    called, which is a different thing entirely and is how the legitimate
    temporary override works.
    """
    out = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Lambda)):
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(node, field, []) or [])
    return out


def _subscript_key(target):
    """The constant string key of a `sys.modules[...]` subscript, or None."""
    if not isinstance(target, ast.Subscript):
        return None
    value = target.value
    if not (isinstance(value, ast.Attribute) and value.attr == "modules"
            and isinstance(value.value, ast.Name) and value.value.id == "sys"):
        return None
    sl = target.slice
    if isinstance(sl, ast.Index):                      # Python < 3.9
        sl = sl.value
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    return None


def violations(path):
    """[(lineno, key)] for every module-scope `sys.modules["*_msgs*"] = ...`."""
    tree = ast.parse(open(path).read(), filename=path)
    found = []
    for node in _module_level_nodes(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            key = _subscript_key(t)
            if key is not None and _is_ros_msg_key(key):
                found.append((node.lineno, key))
    return sorted(found)


def suite_files():
    return sorted(f for f in os.listdir(HERE)
                  if f.endswith("_test.py") and f != os.path.basename(__file__))


def run(verbose=True):
    bad = {}
    for name in suite_files():
        try:
            v = violations(os.path.join(HERE, name))
        except SyntaxError as e:
            # Not this lint's business, and dpi_exports_test.py would not parse a
            # broken file either. Say so rather than passing quietly.
            if verbose:
                print("  ??    %-44s does not parse: %s" % (name, e))
            continue
        if v:
            bad[name] = v
            if verbose:
                for lineno, key in v:
                    print("  FAIL  %-44s line %d assigns sys.modules[%r]"
                          % (name, lineno, key))
        elif verbose:
            print("  ok    %s" % (name,))
    return bad


def test_no_suite_overwrites_a_ros_message_stub():
    """A *_test.py must not assign sys.modules[<ros msgs>] at module scope."""
    bad = run(verbose=False)
    assert not bad, (
        "%d file(s) overwrite a ROS message stub for every other suite in this "
        "directory: %s. Use ros_msg_stubs.ensure_msgs() instead (or "
        "override_msgs()/restore_msgs() around the calls that need your own "
        "class). See ros_msg_stubs.py."
        % (len(bad), "; ".join("%s:%s" % (n, ",".join(str(l) for l, _ in v))
                               for n, v in sorted(bad.items()))))


def test_the_lint_is_not_blind():
    """It flags the pattern it exists for and does not flag the fixes.

    A lint that never fires is indistinguishable from a lint that cannot fire, so
    the detector is run over source written here: the two spellings it exists to
    catch, the two supported alternatives, a function-scope assignment
    (legitimate: it is scoped to a call), and a non-ROS module (somebody else's
    business).
    """
    import tempfile

    def v(src):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as h:
            h.write(src)
            p = h.name
        try:
            return violations(p)
        finally:
            os.unlink(p)

    # KNOWN-POSITIVE: both spellings that take the entry away from everyone else.
    assert v('import sys\nsys.modules["geometry_msgs.msg"] = 1\n')
    assert v('import sys\nsys.modules["sensor_msgs"] = 1\n')
    # ...including one hidden inside a module-level try, which is still a module
    # -level overwrite however it is guarded.
    assert v('import sys\ntry:\n    sys.modules["moveit_msgs.msg"] = 1\n'
             'except Exception:\n    pass\n')
    # KNOWN-NEGATIVE: the supported alternatives, and what is not this lint's
    # business.
    assert not v('import sys\nsys.modules.setdefault("geometry_msgs.msg", 1)\n')
    assert not v('import ros_msg_stubs\nros_msg_stubs.ensure_msgs()\n')
    assert not v('import sys\ndef f():\n    sys.modules["geometry_msgs.msg"] = 1\n')
    assert not v('import sys\nsys.modules["dataguzzler_python"] = 1\n')
    assert not v('import sys\nsys.modules["tari_bringup.cells"] = 1\n')
    # And it is looking at real files, not an empty list.
    names = suite_files()
    assert len(names) > 10, "found only %d suites to lint: %r" % (len(names), names)


def main():
    print(__doc__.splitlines()[0])
    print("")
    bad = run(verbose=True)
    print("")
    if bad:
        print("%d file(s) overwrite a ROS message stub at module scope. Use "
              "ros_msg_stubs.ensure_msgs()." % (len(bad),))
        return 1
    print("%d suites checked, none overwrites a ROS message stub"
          % (len(suite_files()),))
    return 0


if __name__ == "__main__":
    sys.exit(main())
