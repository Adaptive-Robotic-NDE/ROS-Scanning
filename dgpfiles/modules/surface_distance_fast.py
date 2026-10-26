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
# File:  surface_distance_fast.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   BVH backend for SurfaceDistance.signed_distance().
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
"""A BVH backend for SurfaceDistance.signed_distance().

WHAT THIS IS FOR.  The clearance gate asks for the signed distance from a
tooling sample point to the scanned surface tens of millions of times per
toolpath.  The numpy implementation in tooling_clearance.dpi costs about 90 us
per point against a 198 422-triangle reconstruction on this hardware, which is
a nineteen-hour gate.  This module answers the same question from a bounding
volume hierarchy in C++ and hands back the identical numbers.

HOW IT IS INSTALLED.  Include surface_distance_fast.dpi after
tooling_clearance.dpi in the config -- that is the whole wiring, and it edits
no line of tooling_clearance.dpi.  From plain Python (a test, a script) it is
one call:

    import surface_distance_fast
    surface_distance_fast.install(SurfaceDistance)

install() replaces SurfaceDistance.signed_distance() IN PLACE and returns the
same class.  In place rather than by subclassing because surface_distance() in
tooling_clearance.dpi dispatches on isinstance(mesh, SurfaceDistance): a
subclass would leave any instance built before the swap failing that test and
being re-read as a (vertices, triangles) pair.  Nothing else changes -- the
constructor, the normal-orientation rule and its reporting, describe(),
distance(), triangle_radius, k0 and every caller are untouched, and the numpy
implementation stays reachable as SurfaceDistance.signed_distance_numpy() so
any surface can be A/B'd against it.  install() is idempotent, and uninstall()
puts the original back.

If the native kernel cannot be built or loaded for any reason, install()
returns the class unchanged and the pipeline runs exactly as it does today;
pass required=True to make that an error instead, which is what an acceptance
test should do.  To see which one you got: SurfaceDistance.accelerated, or
surface_distance_fast.status().

ONE DELIBERATE DIFFERENCE IN BEHAVIOUR: exact_within and exact_each are
accepted and not used for pruning, because an unsigned cap can report a
positive number for a point that is inside the surface.  The full argument, and
the measurements behind it, are in signed_distance's own docstring below.  It
is the only place this backend does not simply reproduce what it replaced, and
it makes the gate more conservative, never less.

WHY ctypes AND NOT pybind11/SWIG/Cython.  The two machines that have to run this
are the Ubuntu 22.04 cell (Python 3.10, gcc 11) and this Arch dev host (Python
3.14, gcc 16).  A CPython extension is built against one ABI and will not load
on the other, so anything Python-aware means two builds and two ways to get it
wrong.  A plain C ABI shared object has no Python in it at all: the same source,
compiled by whatever g++ is on the machine, is loaded by both interpreters
through ctypes.  It also keeps spatialnde2 out of the loop -- see the note at
the end of this docstring.

THE .so IS BUILT ON FIRST USE and cached in ~/.cache/tari-surface-distance/,
keyed by the source hash, the compiler version and the machine architecture, so
the host build and the container build cannot be mistaken for each other the way
a shared cmake cache can.  It takes a couple of seconds, once.  Set
TARI_SURFACE_DISTANCE_LIB to a prebuilt .so to skip compilation entirely (that
is what a cell with no compiler needs), or TARI_SURFACE_DISTANCE_CACHE to move
the cache.

IF YOU DO SHIP A PREBUILT .so, BUILD IT WITHOUT -march=native.  The first flag
set below uses it, which is right when the machine that compiles is the machine
that runs -- and wrong for an object carried to a different CPU, where the
result is SIGILL, not an exception this module can catch and fall back from.
For a portable object:

    g++ -std=c++17 -shared -fPIC -fvisibility=hidden -pthread \
        -ffp-contract=off -O3 -mavx2 -mfma -fno-math-errno \
        surface_distance_kernel.cpp -o surface_distance_kernel.so

(drop -mavx2 -mfma for anything older than about 2013).  -ffp-contract=off is
not optional in any build -- see the comment on _COMMON.

WHAT IT DOES NOT CHANGE.  The sign still comes from the closest triangle's own
normal, and the normals are still oriented by the class's own constructor, so an
open sheet still reads correctly and a mesh whose normals disagree still says so.
The answer is still exact in the same sense -- the nearest triangle, not an
approximation to it -- but proven by box pruning rather than by widening a
candidate set until it stops mattering, so the `k`/`max_k` knobs have nothing
left to tune (see signed_distance below).  Measured against brute force over
every triangle, it is exact where the numpy path is not: on 300 000 points
against the cell's reconstruction the BVH deviates from ground truth by 0.000e+00
and the numpy path by up to 1.9e-04 m, in the optimistic direction.

NOT PUT IN spatialnde2, deliberately.  That is where geometry kernels belong
architecturally, and if this outlives the gate it should move there.  It is not
there today because spatialnde2 is a compiled, SWIG-wrapped, non-editable
install whose host build is reportedly blocked by SWIG 4.5.0 outright, and every
downstream package has to be rebuilt after an ABI change; that cycle is longer
than this kernel is, and it would have to be crossed on the cell too.  A
self-contained 600-line .cpp with a C ABI can be dropped onto the cell and
verified there in a minute.
"""

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import threading
import weakref

import numpy as np

__all__ = ["install", "accelerate", "uninstall", "status", "available",
           "load_kernel", "SurfaceDistanceKernel"]


# The .cpp sits next to this file.  Keep them together.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "surface_distance_kernel.cpp")

_ABI = 7

# Tried in order; the first that compiles wins.  -march=native is worth roughly
# a third of the runtime here and is safe because the object is built on the
# machine that will run it and cached under that machine's architecture -- but
# it is not portable and not universally supported, hence the ladder.
# -ffast-math is deliberately ABSENT: this has to agree with a numpy reference
# to a nanometre, and reassociation would put that at the compiler's discretion.
_FLAG_SETS = (
    ["-O3", "-march=native", "-fno-math-errno"],
    ["-O3", "-mavx2", "-mfma", "-fno-math-errno"],
    ["-O3", "-fno-math-errno"],
    ["-O2"],
)

# -ffp-contract=off is not optional.  With contraction on, gcc fuses `a + ab*v`
# into an FMA whose rounding differs from numpy's separate multiply and add, and
# the closest-point arithmetic stops being bit-for-bit reproducible.  That does
# not matter for a distance -- it is one ulp -- but it decides which of two
# equidistant triangles is "exactly" equidistant, and that decides the SIGN.
_COMMON = ["-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-pthread",
           "-ffp-contract=off"]


_state = {"lib": None, "why": None, "path": None, "tried": False}


# ----------------------------------------------------------------------------
# building and loading
# ----------------------------------------------------------------------------

def _compilers():
    out = []
    env = os.environ.get("CXX")
    if env:
        out.append(env)
    out += ["g++", "c++", "clang++"]
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _compiler_id(cxx):
    try:
        v = subprocess.run([cxx, "--version"], capture_output=True, text=True,
                           timeout=30)
        return (v.stdout or v.stderr).splitlines()[0].strip()
    except Exception:
        return None


def _cache_dir():
    d = os.environ.get("TARI_SURFACE_DISTANCE_CACHE")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        d = os.path.join(base, "tari-surface-distance")
    return d


def _build(verbose=False):
    """Compile the kernel, cached.  Returns a path or raises."""
    if not os.path.exists(_SRC):
        raise RuntimeError("kernel source missing: %s" % (_SRC,))
    with open(_SRC, "rb") as f:
        src = f.read()

    errors = []
    for cxx in _compilers():
        ident = _compiler_id(cxx)
        if ident is None:
            errors.append("%s: not found" % (cxx,))
            continue
        for flags in _FLAG_SETS:
            key = hashlib.sha256()
            key.update(src)
            key.update(("\0".join([ident, platform.machine(),
                                   platform.system()] + flags + _COMMON)
                        ).encode("utf-8"))
            tag = key.hexdigest()[:16]
            cache = _cache_dir()
            so = os.path.join(cache, "sdk-%d-%s.so" % (_ABI, tag))
            if os.path.exists(so):
                return so
            try:
                os.makedirs(cache, exist_ok=True)
            except Exception as e:
                raise RuntimeError("cannot create %s: %s" % (cache, e))
            # Compile to a unique name and rename into place, so two processes
            # racing to build the same kernel cannot hand each other a half
            # written object.  Two dgpy instances starting together is normal.
            fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
            os.close(fd)
            cmd = [cxx] + _COMMON + list(flags) + [_SRC, "-o", tmp]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=600)
            except Exception as e:
                _unlink(tmp)
                errors.append("%s %s: %s" % (cxx, " ".join(flags), e))
                continue
            if r.returncode != 0:
                _unlink(tmp)
                errors.append("%s %s: %s"
                              % (cxx, " ".join(flags),
                                 (r.stderr or "").strip().splitlines()[-1:]))
                continue
            os.chmod(tmp, 0o644)
            os.replace(tmp, so)
            if verbose:
                sys.stderr.write("surface_distance_fast: built %s with %s %s\n"
                                 % (so, cxx, " ".join(flags)))
            return so
    raise RuntimeError("no compiler could build the kernel; tried: "
                       + "; ".join(str(e) for e in errors))


def _unlink(p):
    try:
        os.unlink(p)
    except Exception:
        pass


def _bind(lib):
    P = ctypes.POINTER(ctypes.c_double)
    lib.sdk_abi_version.restype = ctypes.c_int
    lib.sdk_abi_version.argtypes = []
    lib.sdk_hardware_threads.restype = ctypes.c_int
    lib.sdk_hardware_threads.argtypes = []
    lib.sdk_build.restype = ctypes.c_void_p
    lib.sdk_build.argtypes = [P, P, P, P, ctypes.c_long]
    lib.sdk_free.restype = None
    lib.sdk_free.argtypes = [ctypes.c_void_p]
    lib.sdk_nnodes.restype = ctypes.c_long
    lib.sdk_nnodes.argtypes = [ctypes.c_void_p]
    lib.sdk_triangle_radius.restype = ctypes.c_double
    lib.sdk_triangle_radius.argtypes = [ctypes.c_void_p]
    lib.sdk_query.restype = None
    lib.sdk_query.argtypes = [ctypes.c_void_p, P, ctypes.c_long, P,
                              ctypes.c_double, P, ctypes.c_int]
    return lib


def load_kernel(verbose=False):
    """The loaded ctypes library, or None with the reason in status()."""
    if _state["tried"]:
        return _state["lib"]
    _state["tried"] = True
    try:
        so = os.environ.get("TARI_SURFACE_DISTANCE_LIB")
        if so:
            if not os.path.exists(so):
                raise RuntimeError("TARI_SURFACE_DISTANCE_LIB=%s does not exist"
                                   % (so,))
        else:
            so = _build(verbose=verbose)
        lib = _bind(ctypes.CDLL(so))
        got = int(lib.sdk_abi_version())
        if got != _ABI:
            raise RuntimeError("kernel ABI %d, this module wants %d (stale "
                               "cache or stale TARI_SURFACE_DISTANCE_LIB)"
                               % (got, _ABI))
        _state["lib"] = lib
        _state["path"] = so
    except Exception as e:
        _state["why"] = "%s: %s" % (type(e).__name__, e)
        if verbose:
            sys.stderr.write("surface_distance_fast: %s\n" % (_state["why"],))
    return _state["lib"]


def available():
    return load_kernel() is not None


def status():
    """One line saying whether the fast path is in use, and why not if not."""
    lib = load_kernel()
    if lib is None:
        return "surface_distance_fast: UNAVAILABLE (%s)" % (_state["why"],)
    return ("surface_distance_fast: ready, %s, %d hardware threads"
            % (_state["path"], int(lib.sdk_hardware_threads())))


def _ptr(a):
    """A double* into an array the CALLER is keeping alive.

    Deliberately does not convert: a np.ascontiguousarray() here would return a
    temporary on any non-contiguous input, and the only reference to it would
    be inside this expression -- freed before the C call that uses the pointer
    ever runs.  Every caller below converts first and holds the result.
    """
    if not (a.flags["C_CONTIGUOUS"] and a.dtype == np.float64):
        raise TypeError("internal: array must be C-contiguous float64")
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ----------------------------------------------------------------------------
# the kernel object
# ----------------------------------------------------------------------------

class SurfaceDistanceKernel(object):
    """A built BVH over one set of triangles.  Owns C memory; free() or drop.

    a, b, c and normals are (nt, 3) float64: the triangle vertices and the
    ALREADY ORIENTED unit normals.  This object does not decide orientation and
    does not cull degenerate triangles -- SurfaceDistance.__init__ does both,
    and it stays the one place that does, so the two implementations cannot
    drift apart on which triangles exist or which way they face.
    """

    def __init__(self, a, b, c, normals, nthreads=0):
        lib = load_kernel()
        if lib is None:
            raise RuntimeError(status())
        a = np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 3)
        b = np.ascontiguousarray(b, dtype=np.float64).reshape(-1, 3)
        c = np.ascontiguousarray(c, dtype=np.float64).reshape(-1, 3)
        n = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        if not (a.shape == b.shape == c.shape == n.shape):
            raise ValueError("a, b, c and normals must have the same shape")
        if a.shape[0] == 0:
            raise ValueError("no triangles")
        # Hold references: the C side keeps its own copy, but these must stay
        # alive for the duration of sdk_build().
        self._lib = lib
        self._nthreads = int(nthreads)
        self._handle = lib.sdk_build(_ptr(a), _ptr(b), _ptr(c), _ptr(n),
                                     ctypes.c_long(a.shape[0]))
        if not self._handle:
            raise RuntimeError("sdk_build failed")
        self.ntris = int(a.shape[0])
        self.nnodes = int(lib.sdk_nnodes(ctypes.c_void_p(self._handle)))
        self.triangle_radius = float(
            lib.sdk_triangle_radius(ctypes.c_void_p(self._handle)))

    def query(self, points, exact_within=None, exact_each=None,
              nthreads=None):
        """Signed distance for each point.

        exact_each is a per-point cap, broadcast against `points`, and it wins
        over exact_within when both are given -- the clearance check batches
        thousands of poses into one query and each pose has its own running
        minimum to beat, so one scalar would have to be the loosest of them.
        """
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        n = pts.shape[0]
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        far = 0.0
        if exact_within is not None:
            far = float(exact_within)
            if not np.isfinite(far):
                far = 0.0
        each = None
        eachp = ctypes.POINTER(ctypes.c_double)()
        if exact_each is not None:
            # np.broadcast_to gives a read-only, possibly zero-strided view;
            # ascontiguousarray materialises it, which is what the C side needs
            # and what keeps a scalar exact_each from being read out of bounds.
            each = np.ascontiguousarray(
                np.broadcast_to(
                    np.asarray(exact_each, dtype=np.float64).reshape(-1), (n,)),
                dtype=np.float64)
            eachp = _ptr(each)
        nt = self._nthreads if nthreads is None else int(nthreads)
        self._lib.sdk_query(ctypes.c_void_p(self._handle), _ptr(pts),
                            ctypes.c_long(n), _ptr(out),
                            ctypes.c_double(far), eachp, ctypes.c_int(nt))
        # `each` must outlive the call; naming it here is what guarantees that.
        del each
        return out

    def free(self):
        h = getattr(self, "_handle", None)
        if h:
            self._lib.sdk_free(ctypes.c_void_p(h))
            self._handle = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# the hook
# ----------------------------------------------------------------------------

_warned = set()


def _warn_once(key, message):
    if key in _warned:
        return
    _warned.add(key)
    sys.stderr.write("surface_distance_fast: " + message + "\n")


def _resolve_workers(owner, workers):
    """Thread count, decided by the base module's policy where there is one.

    tooling_clearance.dpi has _kernel_workers(), which folds in
    SURFACE_QUERY_WORKERS and clamps to the machine.  Reaching it through the
    original method's own globals keeps that policy in ONE place: change
    SURFACE_QUERY_WORKERS and this follows, with no second knob to forget.
    """
    try:
        fn = owner.__globals__.get("_kernel_workers")
        if fn is not None:
            return int(fn(workers))
    except Exception:
        pass
    if workers is None:
        return 0                      # 0 means "every hardware thread"
    try:
        w = int(workers)
    except (TypeError, ValueError):
        return 0
    return w if w > 0 else 0


# Built BVHs, keyed weakly by the SurfaceDistance they belong to.
#
# WEAKLY, and not as an attribute on the instance, for two reasons.  A ctypes
# handle in __dict__ breaks pickle and deepcopy, and these objects get cached
# and passed around; keeping it outside means a copy simply builds its own on
# first use instead of exploding.  And a weak key means the kernel is freed when
# the surface is, with no __del__ of our own to get wrong.
_KERNELS = weakref.WeakKeyDictionary()
_KERNELS_LOCK = threading.Lock()
_FAILED = object()


def _kernel_for(surface, nthreads):
    """The BVH for this surface, built on first use, or None if it cannot be.

    Lazy rather than built in __init__ so that installing the backend cannot
    make constructing a SurfaceDistance any slower or any more likely to fail,
    and so that a surface nobody ever queries costs nothing.
    """
    got = _KERNELS.get(surface)
    if got is not None:
        return None if got is _FAILED else got
    with _KERNELS_LOCK:
        got = _KERNELS.get(surface)
        if got is None:
            try:
                got = SurfaceDistanceKernel(surface.a, surface.b, surface.c,
                                            surface.normals, nthreads=nthreads)
            except Exception as e:
                got = _FAILED
                _warn_once("build", "could not index %r (%s: %s); that surface "
                                    "falls back to the numpy path"
                                    % (getattr(surface, "name", "?"),
                                       type(e).__name__, e))
            try:
                _KERNELS[surface] = got
            except TypeError:
                pass                  # not weak-referenceable; just rebuild
    return None if got is _FAILED else got


_FAST_DOC = """Signed distance for each point; negative means inside the part.

        SERVED BY THE BVH KERNEL in surface_distance_fast.  Identical contract
        to the numpy implementation this replaced, which is still available as
        SurfaceDistance.signed_distance_numpy() if you want to A/B a surface.

        `k`, `max_k` and `block` are accepted and ignored, and that is not a
        shortcut:

          k / max_k sized the candidate set for the widening loop, which existed
          to prove that no unconsidered triangle could win.  The BVH proves that
          from the bounding boxes instead, so there is no candidate set to size
          -- the answer is the nearest triangle outright, for any k the caller
          might have asked for.  Ignoring them makes the result MORE exact than
          a small max_k would have allowed, never less.

          block bounded the (points x k x 3) intermediates the barycentric test
          held in numpy, which reached gigabytes.  This path holds one float64
          per point and nothing else, so there is nothing for it to bound.

        exact_within and exact_each ARE ACCEPTED AND NOT USED FOR PRUNING, and
        this is the one place where the fast path deliberately does something
        different from the numpy one.  Both parameters mean "beyond this I will
        accept a lower bound instead of a measurement", and tooling_clearance's
        own docstring says of exact_each that it "is a pure RELAXATION: an
        implementation that ignores it and measures everything exactly is still
        correct, only slower".  This is that implementation, for a reason worth
        stating in full:

          A cap on the search is a cap on the UNSIGNED distance.  When it fires,
          the honest answer is "farther away than the cap" -- and that is not a
          lower bound on a SIGNED distance, because a point beyond the cap can
          be beyond it on the inside.  The numpy path returns a positive number
          in exactly that case.  Measured against a flat 6 962-triangle plate: a
          point 100 mm BELOW the surface comes back as +0.100 with
          exact_within=0.05, and a point 5 mm below comes back as +0.005 with
          exact_within=0.002.  A tool 100 mm inside the part reported as 100 mm
          clear is the precise failure this module was written to catch, and no
          amount of bound-tightening fixes it -- a tighter bound makes the
          early-out fire SOONER, so it fires more often.  ROS-Scanning's own
          tooling_clearance_test.py catches one instance of this ("while the
          contact itself is still measured and reported"): the numpy path passes
          it only because its bound, nearest-centroid minus the largest
          triangle's radius, is too loose to trigger on a 2 mm contact.

          So the cap is dropped rather than reproduced.  Every point gets its
          true signed distance, which is a valid lower bound on itself, so every
          caller comparing a bound against a running minimum still works -- and
          can now only become MORE conservative, never less.  It also makes the
          answer independent of the caps, and therefore of the order in which
          path_clearance happens to measure its links.

          It costs: measured on the 198 422-triangle reconstruction, the cap
          buys nothing at all within a few millimetres of the surface (1.06x,
          and 0.94x on one set -- i.e. inside the noise) and 13-25x on points
          hundreds of millimetres away, where the uncapped query already costs
          only 0.18-0.25 us.  On a realistic mix that is roughly 4x, taking a
          ten-million-point gate from about 0.6 s to about 2.7 s.  Two seconds
          is the right price for not inverting the sign on a penetration.

          SurfaceDistanceKernel.query() still takes both, for anyone who wants
          an unsigned distance quickly and knows that is what they are asking
          for.

        `workers` is resolved by the module's own _kernel_workers(), so
        SURFACE_QUERY_WORKERS keeps meaning what it means everywhere else.

        ANY OTHER KEYWORD hands the whole call back to the numpy
        implementation.  tooling_clearance.dpi is under active development and
        signed_distance() has already grown two parameters; a backend that
        silently ignored a third would be quietly wrong instead of merely slow.
        A warning is printed once so it cannot pass unnoticed.
        """


def install(cls, required=False, verbose=False, nthreads=0):
    """Give `cls` the BVH signed_distance(), in place.  Returns `cls`.

    IN PLACE, rather than by subclassing, on purpose.  surface_distance() in
    tooling_clearance.dpi dispatches with isinstance(mesh, SurfaceDistance); a
    subclass would leave any instance built before the swap failing that test
    and being re-read as a (vertices, triangles) pair.  Patching the method
    changes no class identity, so every instance -- already built, cached in a
    module-level dictionary, or made later -- answers the same way, and there is
    no window in which half the pipeline has the old class.

    Idempotent: installing twice is a no-op, not a double wrap.

    `required=True` raises instead of falling back, which is what an acceptance
    test should pass -- "silently ran the slow path" is otherwise
    indistinguishable from "was fast".  `nthreads` is the default thread count
    for queries; 0 means every hardware thread, and a caller's `workers`
    argument overrides it per call.
    """
    lib = load_kernel(verbose=verbose)
    if lib is None:
        if required:
            raise RuntimeError(status())
        return cls
    if getattr(cls, "_sd_fast_installed", False):
        return cls

    original = cls._signed_distance_impl
    try:
        import inspect
        original_params = set(
            inspect.signature(original).parameters)
    except Exception:
        original_params = None
    default_threads = int(nthreads)

    def _numpy_path(self, points, **kw):
        """The original implementation, given only the arguments it declares."""
        if original_params is not None:
            kw = {n: v for n, v in kw.items() if n in original_params}
        return original(self, points, **kw)

    def signed_distance(self, points, k=None, max_k=None, block=None,
                        exact_within=None, exact_each=None, workers=None,
                        **unknown):
        if unknown:
            _warn_once("kw:" + ",".join(sorted(unknown)),
                       "signed_distance() was called with %s, which this "
                       "backend does not know about, so those calls run on the "
                       "numpy path. Correct but slow -- teach the backend the "
                       "new argument." % (", ".join(sorted(unknown)),))
            return _numpy_path(self, points, k=k, max_k=max_k, block=block,
                               exact_within=exact_within,
                               exact_each=exact_each, workers=workers,
                               **unknown)
        bvh = _kernel_for(self, default_threads)
        if bvh is None:
            return _numpy_path(self, points, k=k, max_k=max_k, block=block,
                               exact_within=exact_within,
                               exact_each=exact_each, workers=workers)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.zeros(0)
        # exact_within / exact_each are deliberately NOT passed through --
        # see the docstring.  They are accepted so the signature matches and
        # callers need no change; they buy about 4x and can invert the sign of
        # a penetrating point, so they are not taken.
        return bvh.query(pts, nthreads=_resolve_workers(original, workers))

    signed_distance.__doc__ = _FAST_DOC
    signed_distance.__name__ = "signed_distance"
    # THE IMPLEMENTATION, NOT THE ENTRY POINT. SurfaceDistance.signed_distance()
    # is a thin wrapper that corrects the sign of points off the end of an open
    # sheet (see its docstring), and that correction has to apply to whichever
    # backend computed the number. Replacing the wrapper would skip it.
    cls._signed_distance_impl = signed_distance
    cls.signed_distance_numpy = original
    cls._sd_fast_installed = True
    cls.accelerated = True
    return cls


def accelerate(base, required=False, verbose=False, nthreads=0):
    """install(), spelled as an assignment.

        SurfaceDistance = surface_distance_fast.accelerate(SurfaceDistance)

    Returns the same class object it was given, so the assignment is cosmetic
    and nothing depends on it happening.
    """
    return install(base, required=required, verbose=verbose, nthreads=nthreads)


def uninstall(cls):
    """Put the numpy signed_distance() back.  For A/B measurement."""
    if not getattr(cls, "_sd_fast_installed", False):
        return cls
    cls._signed_distance_impl = cls.signed_distance_numpy
    del cls.signed_distance_numpy
    cls._sd_fast_installed = False
    cls.accelerated = False
    return cls


if __name__ == "__main__":
    print(status())
