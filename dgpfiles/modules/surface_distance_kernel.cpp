/*#######################################################################
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
# File:  surface_distance_kernel.cpp
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Exact signed point-to-mesh distance over a bounding-volume hierarchy.
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
#######################################################################*/
/* surface_distance_kernel.cpp -- exact signed point-to-mesh distance over a BVH.
 *
 * The kernel behind SurfaceDistance.signed_distance() in ROS-Scanning's
 * tooling_clearance.dpi.  Same question, same answer, same sign convention;
 * about 400x faster per point on eight cores.
 *
 * WHAT "EXACT" MEANS HERE, and why a BVH is the right shape for it.
 *
 *   The numpy version asks a kD-tree for the k nearest triangle CENTROIDS, runs
 *   the barycentric closest-point test over those, and widens k until a triangle
 *   outside the candidate set provably cannot win (its centroid is farther than
 *   best + the largest triangle's radius).  That is exact in the limit, but the
 *   limit is not always reached: the widening stops at max_k, and for a point
 *   that is simply far from the sheet the test never fires at all because every
 *   centroid is about equally far away.  Measured on the cell's 198 422-triangle
 *   reconstruction, roughly one random point in ten thousand comes back with a
 *   distance up to 0.14 mm LARGER than the true one -- the optimistic direction,
 *   on a gate whose job is to notice a probe inside the part.
 *
 *   A BVH answers the question directly.  Descend nearest-child-first, prune any
 *   node whose box is already farther than the best triangle found so far, and
 *   what comes out is the true nearest triangle: no radius slack, no candidate
 *   count to tune, no widening, and no k at which it gives up.  Pruning is sound
 *   because each box is an OUTWARD-rounded superset of the triangles beneath it,
 *   so its distance is a genuine lower bound.
 *
 * SIGN.  From the winning triangle's own normal: negative when (p - q) points
 *   against it.  Not a winding test -- these surfaces are open sheets (a
 *   top-half scan, a reconstruction patch) and a winding test needs a closed
 *   solid.  Normals arrive already oriented from the caller, which is where the
 *   majority-agrees-with-`outward` rule and its reporting live.  This file never
 *   decides orientation; it only reads it.
 *
 * TIES, which are not an edge case here.  When the closest point is a vertex,
 *   every triangle in the fan around that vertex returns that same vertex and
 *   therefore the identical distance, bit for bit -- on random points over this
 *   mesh that happens to more than half of them.  The distance is not in doubt;
 *   the SIGN can be, because the fan's normals differ and for a point lying
 *   nearly in the local tangent plane (a grazing point off the rim of the sheet)
 *   some of them read "above" and some "below".  Which triangle wins is then the
 *   whole answer, so it must not be an accident of traversal order.
 *
 *   The rule, matching what the numpy path does by construction: among triangles
 *   at exactly the same distance, the one whose CENTROID is nearest to the query
 *   point wins, and if that still ties, the lower original triangle index.
 *   (numpy gets this for free -- its candidates come back from cKDTree sorted by
 *   centroid distance and np.argmin takes the first minimum.)  Reproducing it
 *   deliberately is what lets this kernel be dropped under a running gate
 *   without moving any number that gate has already accepted.
 *
 *   For the same reason the closest-point arithmetic below mirrors the numpy
 *   implementation operation for operation -- p-b rather than (p-a)-(b-a), a
 *   division rather than a multiply by a reciprocal, the same guards on the same
 *   zero denominators.  Those choices are worth about one unit in the last place
 *   each, which is nothing in a distance and everything in deciding which of two
 *   equidistant triangles is "exactly" equidistant.  The build turns off FP
 *   contraction for the same reason.
 *
 * exact_within / exact_each.  A cap on the search.  Points whose true distance
 *   exceeds it come back as a positive lower bound rather than an exact value;
 *   points inside it are exact as always.  The bound returned is the tightest
 *   one actually proved -- the smallest box distance among the subtrees the cap
 *   let us skip, or the distance to a triangle tested and rejected.
 *
 *   IT IS A BOUND ON THE UNSIGNED DISTANCE, AND THAT IS A TRAP.  When the cap
 *   fires, all that is known is "farther away than the cap" -- and a point can
 *   be farther away on the INSIDE, where the signed answer is negative and the
 *   bound returned is positive.  Measured against a flat 6 962-triangle plate,
 *   the numpy implementation this replaces returns +0.100 for a point 100 mm
 *   BELOW the surface when exact_within=0.05.  A tool 100 mm inside the part
 *   reported as 100 mm clear is the exact failure a clearance gate exists to
 *   catch, and tightening the bound makes it worse, not better: a tighter bound
 *   fires sooner, so it fires more often.
 *
 *   For that reason surface_distance_fast.py does NOT pass the caps through
 *   from signed_distance(); it measures everything exactly, which the caller's
 *   own documentation explicitly permits ("a pure RELAXATION").  The cap stays
 *   here for a caller who wants an unsigned distance quickly and knows that is
 *   what they are asking for.
 *
 * Geometry is double throughout.  Node bounding boxes are float, rounded
 * OUTWARD, which is safe for pruning and keeps a node at 32 bytes.
 *
 * Build: g++ -O3 -march=native -ffp-contract=off -std=c++17 -shared -fPIC
 * Nothing beyond libstdc++ and pthreads -- deliberately no pybind11 and no
 * Python headers, so ONE .so serves Python 3.10 on the cell and 3.14 on the dev
 * host through ctypes.
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <limits>

#define SDK_API extern "C" __attribute__((visibility("default")))

#define SDK_ABI_VERSION 7

namespace {

static const uint32_t NOTRI = 0xffffffffu;

/* ---------------------------------------------------------------- geometry */

/* The hot record: 72 bytes, exactly the three vertices.  Normals, centroids and
 * original indices live in cold arrays beside it because the inner loop never
 * needs them -- only the winner does. */
struct TriGeom {
    double a[3], b[3], c[3];
};

static inline double dot3(const double *u, const double *v)
{
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

static inline double clip01(double t)
{
    return (t < 0.0) ? 0.0 : (t > 1.0) ? 1.0 : t;
}

/* Closest point to p on one triangle: the barycentric region test from Ericson,
 * Real-Time Collision Detection.
 *
 * Operation for operation this is closest_point_on_triangles() in
 * tooling_clearance.dpi, including its zero-denominator guards and its clips.
 * That one evaluates every region and lets the last np.where win, which gives
 * vertex regions priority over edge regions over the interior; this one tests
 * vertices first, then edges, then falls through to the interior, which is the
 * same priority with an early return.  Where two regions share a boundary the
 * shared condition drives the edge parameter to 0 or 1 and both land on the same
 * vertex, so the orderings cannot diverge. */
static inline void closest_point_on_triangle(const double *p, const TriGeom &t,
                                             double *q)
{
    const double *a = t.a, *b = t.b, *c = t.c;

    const double ab[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
    const double ac[3] = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
    const double ap[3] = { p[0]-a[0], p[1]-a[1], p[2]-a[2] };
    const double d1 = dot3(ab, ap);
    const double d2 = dot3(ac, ap);

    /* Vertex regions come first -- they are the highest priority AND the
     * common case for a query point out beyond the rim of an open sheet, so
     * the later dot products are computed only when the earlier tests miss.
     * bp and cp are formed from b and c directly rather than as ap-ab: that is
     * what the numpy version does, and one ulp here decides a tie. */
    if (d1 <= 0.0 && d2 <= 0.0) {                       /* A */
        q[0] = a[0]; q[1] = a[1]; q[2] = a[2];
        return;
    }

    const double bp[3] = { p[0]-b[0], p[1]-b[1], p[2]-b[2] };
    const double d3 = dot3(ab, bp);
    const double d4 = dot3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {                        /* B */
        q[0] = b[0]; q[1] = b[1]; q[2] = b[2];
        return;
    }

    const double cp[3] = { p[0]-c[0], p[1]-c[1], p[2]-c[2] };
    const double d5 = dot3(ab, cp);
    const double d6 = dot3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {                        /* C */
        q[0] = c[0]; q[1] = c[1]; q[2] = c[2];
        return;
    }

    const double vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {          /* edge AB */
        const double den = d1 - d3;
        const double v = clip01(d1/((den == 0.0) ? 1.0 : den));
        q[0] = a[0]+ab[0]*v; q[1] = a[1]+ab[1]*v; q[2] = a[2]+ab[2]*v;
        return;
    }

    const double vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {          /* edge AC */
        const double den = d2 - d6;
        const double w = clip01(d2/((den == 0.0) ? 1.0 : den));
        q[0] = a[0]+ac[0]*w; q[1] = a[1]+ac[1]*w; q[2] = a[2]+ac[2]*w;
        return;
    }

    const double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {   /* edge BC */
        const double den = (d4 - d3) + (d5 - d6);
        const double w = clip01((d4 - d3)/((den == 0.0) ? 1.0 : den));
        /* from b along (c - b), exactly as the numpy edge() helper is called */
        q[0] = b[0] + (c[0]-b[0])*w;
        q[1] = b[1] + (c[1]-b[1])*w;
        q[2] = b[2] + (c[2]-b[2])*w;
        return;
    }

    /* interior: the foot of the perpendicular */
    const double denom = va + vb + vc;
    const double safe = (std::fabs(denom) < 1e-30) ? 1.0 : denom;
    const double v = vb/safe;
    const double w = vc/safe;
    q[0] = a[0] + ab[0]*v + ac[0]*w;
    q[1] = a[1] + ab[1]*v + ac[1]*w;
    q[2] = a[2] + ab[2]*v + ac[2]*w;
}

/* ---------------------------------------------------------------- the BVH  */

struct Node {
    float    bmin[3];
    float    bmax[3];
    uint32_t start;   /* leaf: first triangle; inner: index of the left child */
    uint32_t count;   /* 0 => inner node                                      */
};
static_assert(sizeof(Node) == 32, "node must stay half a cache line");

/* Round a bound outward on the way to float, so the float box always contains
 * the double box.  Too big costs a wasted visit; too small loses the answer. */
static inline float down_f(double x)
{
    float f = (float)x;
    return ((double)f > x) ? std::nextafterf(f, -INFINITY) : f;
}
static inline float up_f(double x)
{
    float f = (float)x;
    return ((double)f < x) ? std::nextafterf(f, INFINITY) : f;
}

static inline double box_dist2(const Node &nd, const double *p)
{
    double dx = 0.0, dy = 0.0, dz = 0.0;
    if (p[0] < (double)nd.bmin[0])      dx = (double)nd.bmin[0] - p[0];
    else if (p[0] > (double)nd.bmax[0]) dx = p[0] - (double)nd.bmax[0];
    if (p[1] < (double)nd.bmin[1])      dy = (double)nd.bmin[1] - p[1];
    else if (p[1] > (double)nd.bmax[1]) dy = p[1] - (double)nd.bmax[1];
    if (p[2] < (double)nd.bmin[2])      dz = (double)nd.bmin[2] - p[2];
    else if (p[2] > (double)nd.bmax[2]) dz = p[2] - (double)nd.bmax[2];
    return dx*dx + dy*dy + dz*dz;
}

struct AABB {
    double lo[3], hi[3];
    void reset()
    {
        lo[0] = lo[1] = lo[2] =  std::numeric_limits<double>::infinity();
        hi[0] = hi[1] = hi[2] = -std::numeric_limits<double>::infinity();
    }
    void add(const double *p)
    {
        for (int i = 0; i < 3; i++) {
            if (p[i] < lo[i]) lo[i] = p[i];
            if (p[i] > hi[i]) hi[i] = p[i];
        }
    }
    void add(const AABB &o)
    {
        for (int i = 0; i < 3; i++) {
            if (o.lo[i] < lo[i]) lo[i] = o.lo[i];
            if (o.hi[i] > hi[i]) hi[i] = o.hi[i];
        }
    }
    double area() const
    {
        double dx = hi[0]-lo[0], dy = hi[1]-lo[1], dz = hi[2]-lo[2];
        if (dx < 0.0 || dy < 0.0 || dz < 0.0) return 0.0;
        return 2.0*(dx*dy + dy*dz + dz*dx);
    }
};

struct Bvh {
    std::vector<TriGeom>  tris;    /* reordered: every leaf is contiguous  */
    std::vector<double>   nrm;     /* 3 per triangle, same order, cold     */
    std::vector<double>   cen;     /* 3 per triangle, same order, cold     */
    std::vector<uint32_t> orig;    /* original triangle index, cold        */
    std::vector<Node>     nodes;
    std::vector<uint32_t> right;   /* right child per inner node           */
    double  radius;
    long    ntris;
};

static const int    NBINS     = 16;
static const int    LEAFSIZE  = 6;
static const double TRAV_COST = 1.0;

struct PrimRef {
    AABB     box;
    double   cen[3];
    uint32_t idx;
};

/* Binned surface-area-heuristic split.  Returns how many prims land on the
 * left, or 0 when no split beats making a leaf here. */
static size_t sah_split(PrimRef *prims, size_t n, const AABB &cbox)
{
    int axis = 0;
    double ext = cbox.hi[0] - cbox.lo[0];
    for (int i = 1; i < 3; i++) {
        double e = cbox.hi[i] - cbox.lo[i];
        if (e > ext) { ext = e; axis = i; }
    }
    if (!(ext > 0.0)) return 0;

    AABB   binbox[NBINS];
    size_t bincnt[NBINS];
    for (int i = 0; i < NBINS; i++) { binbox[i].reset(); bincnt[i] = 0; }

    const double scale = NBINS/ext;
    for (size_t i = 0; i < n; i++) {
        int b = (int)((prims[i].cen[axis] - cbox.lo[axis])*scale);
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS-1;
        bincnt[b]++;
        binbox[b].add(prims[i].box);
    }

    double rarea[NBINS];
    size_t rcnt[NBINS];
    AABB acc; acc.reset();
    size_t c = 0;
    for (int i = NBINS-1; i >= 1; i--) {
        acc.add(binbox[i]); c += bincnt[i];
        rarea[i] = acc.area(); rcnt[i] = c;
    }

    AABB parent; parent.reset();
    for (int i = 0; i < NBINS; i++) parent.add(binbox[i]);
    const double parea = parent.area();

    double best = (double)n;        /* cost of just making a leaf */
    int    bestcut = -1;
    acc.reset(); c = 0;
    for (int i = 0; i < NBINS-1; i++) {
        acc.add(binbox[i]); c += bincnt[i];
        if (c == 0 || rcnt[i+1] == 0) continue;
        double cost = TRAV_COST
                    + (acc.area()*(double)c + rarea[i+1]*(double)rcnt[i+1])
                      /(parea > 0.0 ? parea : 1.0);
        if (cost < best) { best = cost; bestcut = i; }
    }
    if (bestcut < 0) return 0;

    const double cut = cbox.lo[axis] + (double)(bestcut+1)/scale;
    PrimRef *mid = std::partition(prims, prims+n,
        [axis, cut](const PrimRef &pr) { return pr.cen[axis] < cut; });
    size_t left = (size_t)(mid - prims);
    if (left == 0 || left == n) return 0;
    return left;
}

/* A Node has room for one child index, not two.  The usual fix is to emit the
 * left child immediately after its parent so it is implicit, which needs a
 * two-pass build; instead we keep an explicit right-child table alongside the
 * node array.  Four bytes per node, touched only on inner nodes we actually
 * descend into. */
static uint32_t build_node(Bvh &bv, std::vector<PrimRef> &prims,
                           size_t first, size_t n,
                           const std::vector<TriGeom> &srcg,
                           const double *srcn, const double *srcc,
                           int depth)
{
    AABB box;  box.reset();
    AABB cbox; cbox.reset();
    for (size_t i = 0; i < n; i++) {
        box.add(prims[first+i].box);
        cbox.add(prims[first+i].cen);
    }

    const uint32_t me = (uint32_t)bv.nodes.size();
    bv.nodes.push_back(Node());
    bv.right.push_back(0);
    {
        Node &nd = bv.nodes[me];
        nd.bmin[0] = down_f(box.lo[0]); nd.bmin[1] = down_f(box.lo[1]);
        nd.bmin[2] = down_f(box.lo[2]);
        nd.bmax[0] = up_f(box.hi[0]);   nd.bmax[1] = up_f(box.hi[1]);
        nd.bmax[2] = up_f(box.hi[2]);
        nd.start = 0; nd.count = 0;
    }

    size_t left = 0;
    if (n > (size_t)LEAFSIZE && depth < 62)
        left = sah_split(&prims[first], n, cbox);

    if (left == 0) {
        const uint32_t start = (uint32_t)bv.tris.size();
        for (size_t i = 0; i < n; i++) {
            const uint32_t oi = prims[first+i].idx;
            bv.tris.push_back(srcg[oi]);
            for (int k = 0; k < 3; k++) bv.nrm.push_back(srcn[3*(size_t)oi+k]);
            for (int k = 0; k < 3; k++) bv.cen.push_back(srcc[3*(size_t)oi+k]);
            bv.orig.push_back(oi);
        }
        bv.nodes[me].start = start;
        bv.nodes[me].count = (uint32_t)n;
        return me;
    }

    uint32_t l = build_node(bv, prims, first, left, srcg, srcn, srcc, depth+1);
    uint32_t r = build_node(bv, prims, first+left, n-left, srcg, srcn, srcc,
                            depth+1);
    bv.nodes[me].start = l;
    bv.nodes[me].count = 0;
    bv.right[me] = r;
    return me;
}

/* ------------------------------------------------------------- traversal   */

static inline double cen_dist2(const Bvh &bv, uint32_t i, const double *p)
{
    const double *c = &bv.cen[3*(size_t)i];
    const double dx = p[0]-c[0], dy = p[1]-c[1], dz = p[2]-c[2];
    return dx*dx + dy*dy + dz*dz;
}

/* The cap for point i: its own entry when the caller gave a per-point array,
 * otherwise the scalar.  A non-positive or non-finite entry means no cap. */
static inline double point_cap2(double cap2, const double *each, long i)
{
    if (each == nullptr) return cap2;
    const double e = each[i];
    if (!(e > 0.0) || !std::isfinite(e))
        return std::numeric_limits<double>::infinity();
    return e*e;
}

/* One query.  Returns the signed distance, or -- when a cap was given and
 * nothing beat it -- a positive lower bound on the UNSIGNED distance (see the
 * trap described at the top of this file).
 *
 * TWO SUBTLETIES, both of which decide sign rather than distance, and both of
 * which cost a sign flip on roughly one random point in seven thousand before
 * they were handled.
 *
 *   The comparison is on d, not on d*d.  numpy takes argmin over
 *   np.linalg.norm(), so two triangles whose squared distances differ by one
 *   ulp but whose square roots round to the same double are a TIE there and
 *   would not be one here.  Squared distance still does the fast rejection; the
 *   square root is taken only for a candidate that is at least as good as the
 *   current best, which after the first few triangles is almost none of them.
 *
 *   Pruning is by a hair more than the current best, not by the current best.
 *   When the closest point is a vertex, that vertex sits ON the boundary of the
 *   leaf boxes of every triangle in the fan around it, so those boxes are at
 *   EXACTLY the best distance.  Pruning them with >= throws away most of the
 *   fan before the tie-break can see it, and the tie-break is the whole answer
 *   for a grazing point.  The slack is a few ulps: it cannot open a node that
 *   matters less, only ones that tie. */
static double query_one(const Bvh &bv, const double *p, double cap2)
{
    const Node     *nodes = bv.nodes.data();
    const uint32_t *right = bv.right.data();
    const TriGeom  *tris  = bv.tris.data();

    const double INF = std::numeric_limits<double>::infinity();

    double   best2    = cap2;
    double   best2_hi = (cap2 == INF) ? INF : cap2*(1.0 + 1e-15);
    double   bestd    = std::sqrt(cap2);
    double   minskip2 = INF;
    uint32_t besti    = NOTRI;
    double   bestcen2 = 0.0;      /* lazily filled, only once a tie shows up */
    bool     havecen  = false;
    double   bestq[3] = {0.0, 0.0, 0.0};

    /* Short stack.  build_node refuses to split past depth 62 and we push at
     * most one sibling per level, so 128 is slack rather than a limit. */
    struct Entry { uint32_t node; double d2; };
    Entry stack[128];
    int sp = 0;

    const double d2root = box_dist2(nodes[0], p);
    if (d2root > best2_hi) return std::sqrt(d2root);
    stack[sp].node = 0; stack[sp].d2 = d2root; sp++;

    while (sp > 0) {
        --sp;
        const uint32_t ni  = stack[sp].node;
        const double   nd2 = stack[sp].d2;
        if (nd2 > best2_hi) {
            /* a nearer triangle turned up after this was pushed */
            if (nd2 < minskip2) minskip2 = nd2;
            continue;
        }
        const Node &nd = nodes[ni];
        if (nd.count != 0) {
            for (uint32_t i = nd.start; i < nd.start + nd.count; i++) {
                double q[3];
                closest_point_on_triangle(p, tris[i], q);
                const double dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
                const double d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > best2_hi) {
                    /* The common case -- but under a cap it is also the only
                     * evidence we will ever have about this triangle, and the
                     * bound we return has to be a bound over ALL of them.  A
                     * skipped subtree contributes its box distance; a tested
                     * and rejected triangle contributes its own, which is
                     * tighter.  Leaving these out let the "lower bound" come
                     * back LARGER than the true distance on 17 points in
                     * 200 000 at exact_within=0.05. */
                    if (d2 < minskip2) minskip2 = d2;
                    continue;
                }
                const double d = std::sqrt(d2);
                if (d < bestd) {
                    bestd    = d;
                    best2    = d2;
                    best2_hi = d2*(1.0 + 1e-15);
                    besti    = i;
                    havecen  = false;
                    bestq[0] = q[0]; bestq[1] = q[1]; bestq[2] = q[2];
                } else if (d == bestd && besti != NOTRI) {
                    /* Exactly equidistant -- a vertex fan, almost always.  The
                     * distance is settled; which normal signs it is not, so
                     * break the tie the way the numpy path does rather than by
                     * whichever leaf this traversal happened to reach first. */
                    if (!havecen) {
                        bestcen2 = cen_dist2(bv, besti, p);
                        havecen  = true;
                    }
                    const double c2 = cen_dist2(bv, i, p);
                    if (c2 < bestcen2 ||
                        (c2 == bestcen2 && bv.orig[i] < bv.orig[besti])) {
                        besti    = i;
                        bestcen2 = c2;
                        bestq[0] = q[0]; bestq[1] = q[1]; bestq[2] = q[2];
                    }
                }
            }
            continue;
        }
        const uint32_t li = nd.start;
        const uint32_t ri = right[ni];
        const double ld2 = box_dist2(nodes[li], p);
        const double rd2 = box_dist2(nodes[ri], p);
        /* push the farther child first, so the nearer one is popped first and
         * best2_hi tightens before the far side is opened */
        if (ld2 <= rd2) {
            if (rd2 <= best2_hi) { stack[sp].node = ri; stack[sp].d2 = rd2; sp++; }
            else if (rd2 < minskip2) minskip2 = rd2;
            if (ld2 <= best2_hi) { stack[sp].node = li; stack[sp].d2 = ld2; sp++; }
            else if (ld2 < minskip2) minskip2 = ld2;
        } else {
            if (ld2 <= best2_hi) { stack[sp].node = li; stack[sp].d2 = ld2; sp++; }
            else if (ld2 < minskip2) minskip2 = ld2;
            if (rd2 <= best2_hi) { stack[sp].node = ri; stack[sp].d2 = rd2; sp++; }
            else if (rd2 < minskip2) minskip2 = rd2;
        }
    }

    if (besti == NOTRI) {
        /* nothing inside the cap: the tightest bound we actually proved */
        if (!(minskip2 < INF)) minskip2 = cap2;
        return std::sqrt(minskip2);
    }

    const double delta[3] = { p[0]-bestq[0], p[1]-bestq[1], p[2]-bestq[2] };
    const double side = dot3(delta, &bv.nrm[3*(size_t)besti]);
    return (side < 0.0) ? -bestd : bestd;
}

}  /* anonymous namespace */

/* ------------------------------------------------------------- the C ABI   */

SDK_API int sdk_abi_version(void) { return SDK_ABI_VERSION; }

SDK_API int sdk_hardware_threads(void)
{
    unsigned n = std::thread::hardware_concurrency();
    return (int)(n ? n : 1);
}

/* Build.  a, b, c, n are each (nt, 3) C-contiguous float64, already culled of
 * degenerate triangles and already normal-oriented.  The caller owns both of
 * those decisions -- and the reporting that goes with them -- so the numpy and
 * BVH paths cannot drift apart on which triangles exist or which way they face. */
SDK_API void *sdk_build(const double *a, const double *b, const double *c,
                        const double *n, long nt)
{
    if (nt <= 0) return nullptr;

    std::vector<TriGeom> srcg((size_t)nt);
    std::vector<double>  srcc(3*(size_t)nt);
    std::vector<PrimRef> prims((size_t)nt);
    double maxr2 = 0.0;

    for (long i = 0; i < nt; i++) {
        TriGeom &t = srcg[(size_t)i];
        for (int k = 0; k < 3; k++) {
            t.a[k] = a[3*i+k];
            t.b[k] = b[3*i+k];
            t.c[k] = c[3*i+k];
        }
        PrimRef &pr = prims[(size_t)i];
        pr.box.reset();
        pr.box.add(t.a); pr.box.add(t.b); pr.box.add(t.c);
        for (int k = 0; k < 3; k++) {
            /* the same expression numpy uses: (a + b + c)/3 */
            pr.cen[k] = (a[3*i+k] + b[3*i+k] + c[3*i+k])/3.0;
            srcc[3*(size_t)i+k] = pr.cen[k];
        }
        pr.idx = (uint32_t)i;

        for (int v = 0; v < 3; v++) {
            const double *pv = (v == 0) ? t.a : (v == 1) ? t.b : t.c;
            const double dx = pv[0]-pr.cen[0], dy = pv[1]-pr.cen[1],
                         dz = pv[2]-pr.cen[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > maxr2) maxr2 = r2;
        }
    }

    Bvh *bv = new Bvh();
    bv->ntris = nt;
    bv->radius = std::sqrt(maxr2);
    bv->tris.reserve((size_t)nt);
    bv->nrm.reserve(3*(size_t)nt);
    bv->cen.reserve(3*(size_t)nt);
    bv->orig.reserve((size_t)nt);
    bv->nodes.reserve((size_t)nt/2 + 8);
    bv->right.reserve((size_t)nt/2 + 8);
    build_node(*bv, prims, 0, (size_t)nt, srcg, n, srcc.data(), 0);
    return (void *)bv;
}

SDK_API void sdk_free(void *h)
{
    if (h) delete (Bvh *)h;
}

SDK_API long sdk_nnodes(void *h)
{
    return h ? (long)((Bvh *)h)->nodes.size() : 0;
}

SDK_API double sdk_triangle_radius(void *h)
{
    return h ? ((Bvh *)h)->radius : 0.0;
}

/* Query.  pts is (np, 3) C-contiguous float64, out is (np,) float64.
 *
 * exact_within <= 0 or non-finite means no cap -- exact everywhere.
 * exact_each, when non-null, is a (np,) float64 array of PER-POINT caps that
 * replaces exact_within: the clearance check batches thousands of poses into
 * one query and each pose has its own running minimum to beat, so one scalar
 * cap would have to be the loosest of them.  A cap is a pure relaxation -- any
 * entry that is non-positive or non-finite simply means "be exact for this
 * one", which is always a correct answer, only slower.
 *
 * nthreads <= 0 means every hardware thread. */
SDK_API void sdk_query(void *h, const double *pts, long np, double *out,
                       double exact_within, const double *exact_each,
                       int nthreads)
{
    if (!h || np <= 0) return;
    const Bvh &bv = *(const Bvh *)h;

    const double INF = std::numeric_limits<double>::infinity();
    double cap2 = INF;
    if (exact_within > 0.0 && std::isfinite(exact_within))
        cap2 = exact_within*exact_within;

    int nt = nthreads;
    if (nt <= 0) nt = sdk_hardware_threads();
    /* Below about a thousand points a thread costs more to start than it saves.
     * Spawning per call rather than keeping a pool alive is deliberate: at the
     * sizes this is called with, the ~25 us a thread takes to start is a couple
     * of per cent of the work it then does, and a long-lived thread pool inside
     * a library that a forking Python process might load is a much worse
     * problem than the couple of per cent. */
    if ((long)nt > np/1024 + 1) nt = (int)(np/1024 + 1);
    if (nt < 1) nt = 1;

    if (nt == 1) {
        for (long i = 0; i < np; i++)
            out[i] = query_one(bv, &pts[3*i], point_cap2(cap2, exact_each, i));
        return;
    }

    /* Dynamic chunks: cost per point spans two orders of magnitude between a
     * point sitting on the surface and one the cap rejects at the root, so a
     * static split would leave most threads idle behind the one that drew the
     * hard block. */
    std::atomic<long> next(0);
    const long CHUNK = 2048;
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt);
    for (int t = 0; t < nt; t++) {
        pool.emplace_back([&bv, pts, np, out, cap2, exact_each, &next]() {
            for (;;) {
                long start = next.fetch_add(CHUNK, std::memory_order_relaxed);
                if (start >= np) return;
                long end = start + CHUNK;
                if (end > np) end = np;
                for (long i = start; i < end; i++)
                    out[i] = query_one(bv, &pts[3*i],
                                       point_cap2(cap2, exact_each, i));
            }
        });
    }
    for (auto &th : pool) th.join();
}
