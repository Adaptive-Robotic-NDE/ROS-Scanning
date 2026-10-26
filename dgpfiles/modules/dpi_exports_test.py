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
# File:  dpi_exports_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Tests that every name a .dpi shares across modules is declared
#   global.
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
"""Every name one .dpi uses from another must be declared `global` where it is defined.

A .dpi is compiled and executed as a FUNCTION BODY, and all of them share one
globals() dict. So a `def` or an assignment at the top of a .dpi is a LOCAL of
that body unless the file says `global` first -- invisible to every other
module, however obvious the name looks. Nothing catches it at load time: the
include succeeds, the banner prints, and the failure arrives later as a
NameError from inside whichever function first reaches for it.

Worse, a missing export does not always crash. A name reached through
globals().get() comes back None instead of raising, so the caller quietly takes
its fallback path -- a default in place of a measurement, a stale file in place
of a fresh one -- and the run carries on with the wrong answer. That is why this
is a test rather than a note in a docstring.

Four checks:

  * check_duplicate_exports() -- a name EXPORTED by more than one .dpi. Same
    shared globals(), read the other way: two files writing one name. For a def
    or a class the later include replaces the earlier and it is gone. For a value
    it is shared state, sometimes deliberately, which is what
    DUPLICATE_EXPORT_ALLOWLIST is for.
  * main() -- a name USED across files but never declared `global` where it is
    defined.
  * check_undefined_constants() -- an UPPERCASE name used and defined nowhere.
  * check_borrowed_imports() -- a .dpi using a MODULE it does not import,
    borrowing it from whichever config happens to include it.

    python3 dpi_exports_test.py
"""
import ast
import builtins
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: Files that are deliberately self-contained, or are tests rather than modules.
SKIP = set()

#: Names more than one .dpi deliberately writes into the shared globals().
#:
#: An entry is name -> (exact set of files, why). EXACT: a fourth file joining an
#: allowlisted three still fails, because "these three agree about this variable"
#: is a fact about those three and nothing else.
#:
#: Only SHARED STATE belongs here -- one variable several scan modes take turns
#: owning. A def or a class exported twice is never allowlistable: there the
#: later include does not share anything, it replaces the earlier definition and
#: the earlier one is gone.
DUPLICATE_EXPORT_ALLOWLIST = {
    "ndaccumulator": (
        ("mesh_scan.dpi", "scan_accumulation.dpi"),
        "handle on the /ut_scan_volume accumulator; scan_accumulation.dpi builds "
        "it, mesh_scan.dpi must delete the plane scan's before its own passes."),
    "cscanfunction": (
        ("mesh_scan.dpi", "scan_accumulation.dpi"),
        "handle on the /ut_cscan math function, deleted by whichever scan mode "
        "takes those channels over next."),
    "cscan_gates": (
        ("mesh_scan.dpi", "scan_accumulation.dpi"),
        "the gate chain; each gate bound is a channel, so the previous scan's "
        "must be RELEASED before setup_gated_cscan can define them again."),
    "projector": (
        ("mesh_scan.dpi", "scan_accumulation.dpi"),
        "the SurfaceProjector on prefix /ut/proj; a second one on that prefix "
        "cannot be built, so both modes must see the same object to reuse it."),
    "scan_origin": (
        ("mesh_scan.dpi", "plane_scan.dpi", "surface_path_scan.dpi"),
        "the running scan's footprint corner; readers want the CURRENT scan's, "
        "and only one scan mode runs at a time."),
    "scan_plan": (
        ("plane_scan.dpi", "scan_plan.dpi"),
        "scan_plan.dpi owns it; plane_scan.dpi's Scan2DPlane() only ever clears "
        "it, because a point-to-point scan has no sweep velocity to plan."),
}

FAILURES = []


def _body(path):
    """The .dpi parsed the way the loader compiles it: as a function body."""
    src = open(path).read()
    tree = ast.parse("def _dpi_body():\n" + "".join(
        "    " + l + "\n" for l in src.splitlines()))
    return tree.body[0]


def _declared_global(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _defined_at_top(fn):
    """Names this body binds at ITS top level -- what another file might want."""
    names = set()
    for node in fn.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _scope_nodes(node):
    """Every node inside ONE scope, not descending into nested scopes.

    A nested def or class is yielded (its NAME is bound in this scope) but its
    body is not walked, because `global` binds per scope: an inner function that
    assigns a name without its own `global` makes a local, and counting that as
    an export invents clashes.
    """
    for ch in ast.iter_child_nodes(node):
        yield ch
        if not isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in _scope_nodes(ch):
                yield sub


def _exported_kinds(fn):
    """name -> kinds, for names this file actually WRITES into shared globals().

    A bare `global N` is NOT an export. Without an assignment in the same scope
    it is a no-op -- reads fall through to globals() anyway -- so flagging one
    invents a clash that cannot happen. A dead `global` declaration left behind
    after the definition moved to another file would otherwise look like a third
    writer of somebody else's variable. So a name counts here only where one
    scope both declares it global AND binds it.

    The KIND matters, because a collision means different things:

      * a value assigned in several files is ordinary shared state -- one
        variable that whichever scan mode is driving takes over. Legitimate,
        and allowlistable.
      * a DEF or a CLASS in several files is not shared anything. The file
        included last wins and the earlier definition is gone, silently, with no
        error at load. Every caller of the earlier one then reaches an unrelated
        function of the same name, and typically dies on an unexpected keyword
        argument somewhere far from the clash.
    """
    scopes = [fn] + [n for n in ast.walk(fn)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))]
    out = {}
    for scope in scopes:
        decl, bound = set(), {}
        for n in _scope_nodes(scope):
            if isinstance(n, ast.Global):
                decl.update(n.names)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.setdefault(n.name, set()).add("def")
            elif isinstance(n, ast.ClassDef):
                bound.setdefault(n.name, set()).add("class")
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.setdefault(n.id, set()).add("value")
            elif isinstance(n, ast.alias):
                bound.setdefault((n.asname or n.name).split(".")[0],
                                 set()).add("value")
        for name in decl & set(bound):
            out.setdefault(name, set()).update(bound[name])
    return out


def _loaded(fn):
    """Every name this body READS, anywhere inside it."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # globals().get("name") hides the reference from the AST as a
            # string, so string constants have to count as reads as well.
            names.add(node.value)
    return names


# ------------------------------------------------------------------------------
# ... and a name EXPORTED by more than one file.
#
# This is the other half of the .dpi scoping problem, and the expensive half. All
# of them are compiled as function bodies sharing ONE globals(), so `global N`
# in two files is two files writing one variable. Nothing reports it -- not at
# load, not at first use.
#
# Which of the two outcomes you get depends only on what N is bound to:
#
#   a def or a class  the later include REPLACES the earlier definition and it
#                     is gone. Callers of the earlier one silently reach the
#                     later file's function of the same name, and fail wherever
#                     the two signatures differ -- far from the clash itself.
#                     Never allowlist this shape; rename one of them.
#   a value           the two files share one variable. Often that is exactly
#                     what is wanted: scan_accumulation.dpi and mesh_scan.dpi
#                     take turns owning the accumulators because the second scan
#                     of a session has to take the first one's channels down.
#                     Those belong in DUPLICATE_EXPORT_ALLOWLIST, with a reason
#                     and with the EXACT set of files that agree about them.
#
# Exported is the operative word both times. A bare `def foo` at the top of a
# .dpi binds a LOCAL of that body and is invisible to every other file -- _hms
# exists in both scan_parameters.dpi and tooling_clearance.dpi and they never
# meet, so an unexported name repeated across files is not a clash.

def check_duplicate_exports(exported_kinds):
    """exported_kinds: file -> {name: kinds}, from _exported_kinds()."""
    by_name = {}
    for f in sorted(exported_kinds):
        for n, kinds in exported_kinds[f].items():
            by_name.setdefault(n, {})[f] = kinds

    bad = 0
    for n in sorted(by_name):
        where = by_name[n]
        if len(where) < 2:
            continue
        fs = sorted(where)
        code_files = sorted(f for f in fs if where[f] & {"def", "class"})
        allowed, reason = DUPLICATE_EXPORT_ALLOWLIST.get(n, (None, None))

        if len(code_files) > 1:
            # Not allowlistable, whatever the allowlist says: one of these
            # definitions does not survive the load.
            msg = ("%r is defined AND exported by %s. Every .dpi shares one "
                   "globals(), so the file included LAST wins and the earlier "
                   "definition is silently gone, and its callers reach the "
                   "other one. Rename one."
                   % (n, " and ".join(code_files)))
        elif allowed is not None and tuple(sorted(allowed)) == tuple(fs):
            continue                             # deliberate shared state
        elif allowed is not None:
            msg = ("%r is exported by %s, but DUPLICATE_EXPORT_ALLOWLIST says "
                   "only %s share it (%s). A new writer of a variable several "
                   "scan modes take turns owning is exactly the thing this "
                   "check exists to notice: either it is wrong, or the "
                   "allowlist entry needs to say why it is not."
                   % (n, " and ".join(fs), " and ".join(sorted(allowed)), reason))
        else:
            msg = ("%r is exported by %s. Every .dpi shares one globals(), so "
                   "those files are writing ONE variable and the last write "
                   "wins -- silently, with no error at load. If that is "
                   "deliberate, add it to DUPLICATE_EXPORT_ALLOWLIST with a "
                   "reason and document it at each site; otherwise give them "
                   "separate names." % (n, " and ".join(fs)))
        FAILURES.append(msg)
        print("  FAIL  " + msg)
        bad += 1

    # An allowlist entry that no longer describes a real collision is a pardon
    # waiting for the next name that happens to match it. Say so while the
    # reason for removing it is still obvious.
    live = set(n for n in by_name if len(by_name[n]) > 1)
    for n in sorted(set(DUPLICATE_EXPORT_ALLOWLIST) - live):
        msg = ("DUPLICATE_EXPORT_ALLOWLIST still pardons %r, but only one .dpi "
               "exports it now. Delete the entry, or it will silently pardon "
               "the next real clash on that name." % (n,))
        FAILURES.append(msg)
        print("  FAIL  " + msg)
        bad += 1

    if not bad:
        print("every exported name is exported by exactly one .dpi "
              "(%d allowlisted)" % len(DUPLICATE_EXPORT_ALLOWLIST))
    return 1 if bad else 0


def main():
    files = sorted(f for f in os.listdir(HERE)
                   if f.endswith(".dpi") and f not in SKIP)
    defines, exports, uses, callables = {}, {}, {}, {}
    for f in files:
        try:
            fn = _body(os.path.join(HERE, f))
        except SyntaxError as e:
            FAILURES.append("%s does not parse: %s line %s" % (f, e.msg, e.lineno))
            continue
        defines[f] = _defined_at_top(fn)
        callables[f] = _exported_kinds(fn)
        exports[f] = _declared_global(fn)
        uses[f] = _loaded(fn)

    # name -> the files that define it, and the files that export it
    owner, exported = {}, {}
    for f in defines:
        for n in defines[f]:
            owner.setdefault(n, set()).add(f)
        for n in exports[f]:
            exported.setdefault(n, set()).add(f)

    problems = []
    for f in uses:
        for n in sorted(uses[f]):
            if n in defines.get(f, ()):          # its own
                continue
            homes = owner.get(n, set()) - {f}
            if not homes:                        # builtin, import, or unknown
                continue
            if any(n in exports.get(h, ()) for h in homes):
                continue                         # somebody exports it
            problems.append((f, n, sorted(homes)))

    check_duplicate_exports(callables)

    print("checked %d .dpi files\n" % len(files))
    if FAILURES and not problems:
        return 1
    if not problems:
        print("every cross-module name is declared global where it is defined")
        return 0
    for f, n, homes in problems:
        msg = ("%s uses %r, which is defined in %s but never declared global "
               "there -- it will be invisible at runtime"
               % (f, n, ", ".join(homes)))
        FAILURES.append(msg)
        print("  FAIL  " + msg)
    return 1




# ------------------------------------------------------------------------------
# ... and a name that is used but defined NOWHERE.
#
# The check above compares `global X` against `X = ...`, so it is blind to the
# case where BOTH are deleted. An over-wide edit that removes a constant along
# with its `global` line leaves a file that still parses and still passes every
# other suite, and raises NameError the first time the line that reads it runs.
#
# Restricted to UPPERCASE names: they are module constants by convention and are
# almost never locals, so this is precise rather than a scope analysis. Names a
# .dgp config sets, and the ones cell_capture_node.dpi injects into globals()
# at import, are resolved too -- they are legitimately defined elsewhere.

def _constants_defined_anywhere(files):
    """Every UPPERCASE name any .dpi, .dgp or runtime injection provides."""
    known = set(dir(builtins))
    dgp = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "*.dgp")))
    for f in list(files) + dgp:
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for nn in ast.walk(t):
                        if isinstance(nn, ast.Name):
                            known.add(nn.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    known.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                known.add(n.name)
            # ("CAPTURE_ROOT", default) pairs: names pushed into globals() at
            # import by _capture_defaults(). Defined at runtime, not by an
            # assignment anything can see statically.
            elif isinstance(n, ast.Tuple) and n.elts and \
                    isinstance(n.elts[0], ast.Constant) and \
                    isinstance(n.elts[0].value, str) and \
                    n.elts[0].value.isupper():
                known.add(n.elts[0].value)
    return known


def _undefined_constants(files):
    known = _constants_defined_anywhere(files)
    out = []
    for f in files:
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                    and n.id.isupper() and len(n.id) > 3:
                used.add(n.id)
        for name in sorted(used - known):
            out.append((os.path.basename(f), name))
    return out


def check_undefined_constants(files):
    bad = _undefined_constants(files)
    for f, name in bad:
        msg = ("%s uses %r, which NOTHING defines -- not this file, not another "
               ".dpi, not a .dgp config. It will raise NameError the first time "
               "that line runs." % (f, name))
        FAILURES.append(msg)
        print("  FAIL  " + msg)
    if not bad:
        print("every UPPERCASE name used is defined somewhere")
    return 1 if bad else 0


# ------------------------------------------------------------------------------
# ... and a MODULE a .dpi uses without importing it.
#
# The same function-body scoping, third face. `import math` at the top of a .dpi
# binds a LOCAL of that body: it is visible to defs in that file through their
# closure and INVISIBLE to every other .dpi. Only the including .dgp's own
# top-level imports land in the shared globals(), so which modules a .dpi can
# borrow depends entirely on the config that loaded it -- under one config `os`,
# `np`, `time`, `copy` and `itertools` may be present while `math`, `json`,
# `re`, `struct`, `hashlib`, `contextlib` and `functools` are not.
#
# So a .dpi that uses `math.` without importing it works or raises NameError
# depending on which config loaded it, and the difference does not show up until
# the line runs -- which may be a rarely taken retry path, minutes into a run,
# after the arm has already moved.
#
# NOT caught by check_undefined_constants(): that one is restricted to UPPERCASE
# names and takes the union over every file, which is the wrong question here --
# another .dpi importing math does not help this one.
#
# A .dpi is allowed a module if IT imports it, or if EVERY .dgp that includes it
# imports it at top level. The intersection, not the union: a module that works
# under one config and NameErrors under another is the bug, not a pass.

#: Modules a .dpi might reasonably reach for. Deliberately a list rather than
#: "anything that looks like a module": `warnings = []` is a local list in
#: scan_parameters.dpi and `re.match` on a local named `re` would be fine too, so
#: membership here only decides what is WORTH asking about -- the scope walk below
#: still has to find the name unbound.
KNOWN_MODULES = frozenset((
    "math", "os", "sys", "time", "json", "copy", "itertools", "hashlib",
    "struct", "glob", "shutil", "threading", "subprocess", "collections",
    "contextlib", "functools", "re", "csv", "datetime", "tempfile", "traceback",
    "textwrap", "random", "warnings", "uuid", "io", "ast", "pickle", "platform",
    "socket", "errno", "stat", "base64", "gzip", "zipfile", "statistics",
    "bisect", "heapq", "weakref", "types", "inspect", "yaml",
    "np", "numpy", "snde", "rclpy", "rosm", "dgpy", "plt",
))


def _imports_of(tree_body):
    """Module names bound by the import statements in this list of statements."""
    out = set()
    for n in tree_body:
        for a in ast.walk(n):
            if isinstance(a, ast.Import):
                for x in a.names:
                    out.add((x.asname or x.name).split(".")[0])
            elif isinstance(a, ast.ImportFrom):
                for x in a.names:
                    out.add(x.asname or x.name)
    return out


def _dgp_files():
    """The real configs. _agentctl_*.dgp are the control-port wrappers that
    tooling writes next to a config; they are copies and must not be counted."""
    return [f for f in sorted(glob.glob(os.path.join(os.path.dirname(HERE), "*.dgp")))
            if not os.path.basename(f).startswith("_agentctl")]


def _includes(path):
    """The .dpi basenames this file includes, from its include() calls.

    Read out of the SOURCE rather than by importing anything: include()'s
    argument is built with os.path.join(SCAN_MODULE_DIR, "name.dpi"), so the
    basename is a string constant in the call and that is all this needs.
    """
    out = set()
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "include":
            for a in ast.walk(n):
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.endswith(".dpi"):
                    out.add(os.path.basename(a.value))
    return out


def _providers(files):
    """dpi basename -> the set of module names EVERY including .dgp provides.

    Transitive: a .dpi included by another .dpi inherits that one's includers.
    A .dpi nothing includes gets an empty set, which is the strict reading and
    the right one -- it has no config to borrow from.
    """
    by_name = {os.path.basename(f): f for f in files}
    direct = {}                      # dgp path -> its top-level imports
    for d in _dgp_files():
        try:
            direct[d] = _imports_of(ast.parse(open(d).read()).body)
        except SyntaxError:
            continue
    reach = {}                       # dpi basename -> set of dgp paths
    for d in direct:
        seen, frontier = set(), list(_includes(d))
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            reach.setdefault(name, set()).add(d)
            if name in by_name:
                frontier.extend(_includes(by_name[name]))
    out = {}
    for name in by_name:
        dgps = reach.get(name)
        if not dgps:
            out[name] = set()
            continue
        out[name] = set.intersection(*[direct[d] for d in dgps])
    return out


def _borrowed_imports(files):
    """[(dpi, module, line, function)] -- a module used and never imported."""
    provided = _providers(files)
    bad = {}
    for f in files:
        name = os.path.basename(f)
        src = open(f).read()
        body = "\n".join(l for l in src.splitlines()
                          if not l.startswith("global "))
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        top = _imports_of(tree.body)
        allowed = top | provided.get(name, set())

        def binds(fn):
            b = set()
            for a in ast.walk(fn):
                if isinstance(a, ast.arg):
                    b.add(a.arg)
                elif isinstance(a, ast.Name) and isinstance(a.ctx, ast.Store):
                    b.add(a.id)
                elif isinstance(a, ast.Import):
                    for x in a.names:
                        b.add((x.asname or x.name).split(".")[0])
                elif isinstance(a, ast.ImportFrom):
                    for x in a.names:
                        b.add(x.asname or x.name)
                elif isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    b.add(a.name)
                elif isinstance(a, ast.ExceptHandler) and a.name:
                    b.add(a.name)
            return b

        # Every function, with the bindings of every scope enclosing it: a
        # nested def legitimately closes over the outer one's `import`.
        def visit(node, stack, where):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner = stack + [binds(ch)]
                    bound = set().union(*inner)
                    for a in ast.walk(ch):
                        if isinstance(a, ast.Attribute) \
                                and isinstance(a.value, ast.Name) \
                                and isinstance(a.value.ctx, ast.Load):
                            m = a.value.id
                            if m in KNOWN_MODULES and m not in bound \
                                    and m not in allowed:
                                bad.setdefault((name, m), (a.lineno, ch.name))
                    visit(ch, inner, ch.name)
                else:
                    visit(ch, stack, where)

        visit(tree, [binds_of_module(tree)], "<module scope>")
    return [(f, m, ln, fn) for (f, m), (ln, fn) in sorted(bad.items())]


def binds_of_module(tree):
    """Names bound at the .dpi's own top level (its body's locals)."""
    out = set()
    for n in tree.body:
        for a in ast.walk(n):
            if isinstance(a, ast.Import):
                for x in a.names:
                    out.add((x.asname or x.name).split(".")[0])
            elif isinstance(a, ast.ImportFrom):
                for x in a.names:
                    out.add(x.asname or x.name)
            elif isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                out.add(a.name)
            elif isinstance(a, ast.Name) and isinstance(a.ctx, ast.Store):
                out.add(a.id)
    return out


def check_borrowed_imports(files):
    bad = _borrowed_imports(files)
    for f, module, line, fn in bad:
        msg = ("%s:%d %s() uses %r, and this file never imports it. A .dpi is a "
               "function body, so another .dpi's import is invisible here and "
               "only the including .dgp's own top-level imports are shared. "
               "Import it in this file." % (f, line, fn, module))
        FAILURES.append(msg)
        print("  FAIL  " + msg)
    if not bad:
        print("every module a .dpi uses is one it imports")
    return 1 if bad else 0


if __name__ == "__main__":
    rc = main()
    rc = check_undefined_constants(
        sorted(glob.glob(os.path.join(HERE, "*.dpi")))) or rc
    rc = check_borrowed_imports(
        sorted(glob.glob(os.path.join(HERE, "*.dpi")))) or rc
    if FAILURES:
        print("\n%d export problem(s)" % len(FAILURES))
    sys.exit(rc)


def test_dpi_exports():
    """So pytest reports the same thing as running it."""
    main()
    check_undefined_constants(sorted(glob.glob(os.path.join(HERE, "*.dpi"))))
    check_borrowed_imports(sorted(glob.glob(os.path.join(HERE, "*.dpi"))))
    assert not FAILURES, FAILURES
