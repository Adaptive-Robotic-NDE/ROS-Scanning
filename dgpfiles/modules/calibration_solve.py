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
# File:  calibration_solve.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Numpy-only arithmetic that turns calibration observations into
#   sensor-to-robot transforms.
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
"""The arithmetic behind sensor-to-robot calibration. Numpy only.

DELIBERATELY FREE OF cv2 AND OF ROS. calibration.dpi does the moving, the
grabbing and the checkerboard finding; everything that turns observations into
a transform, and everything that decides whether the observations deserved to
be turned into one, is here. Three reasons that split is worth the extra file:

  * it can be tested on the host, where there is no cv2 and no rclpy
    (calibration_test.py),
  * the cell runs whatever OpenCV Ubuntu 22.04 ships, and a hand-eye solver
    that lives in numpy cannot be broken by an OpenCV API change,
  * the part that is easy to get silently wrong is the arithmetic, and code
    that needs a robot to run is code nobody tests.

THE PROBLEM. A sensor is bolted to the flange. The unknown is X, the transform
from the flange to the sensor's own frame. At pose i the robot reports G_i
(flange in the base frame) and the sensor reports C_i (the calibration target
in the sensor's frame). The target does not move, so

    G_i . X . C_i  =  W          for every i, with W (target in base) unknown.

Eliminating W between two poses gives the classical AX = XB:

    A_ij = G_j^-1 G_i            the relative motion of the flange
    B_ij = C_j C_i^-1            the relative motion the sensor saw
    A_ij X = X B_ij

WHAT MAKES THIS DANGEROUS, and the whole reason for the conditioning section
below. The translation of X only enters through (R_A - I) t_X, and (R_A - I)
is SINGULAR: it annihilates the axis of R_A, and its other two singular values
are 2*sin(theta/2). So

  * a pose set whose rotations are all small determines t_X barely at all --
    the amplification is 1/(2 sin(theta/2)), which is 5.7x at 10 degrees and
    57x at 1 degree,
  * a pose set whose rotations all share ONE axis never determines t_X along
    that axis, at any angle,

and in BOTH cases the residual of the fit is small, because a wrong t_X in the
null direction fits the data exactly. A small residual is not evidence. The
amplification and the singular values are the evidence, and solve_handeye()
returns them whether or not anyone asks.

Angles are radians and lengths metres throughout, except where a name says
_deg or _mm. Quaternions are (x, y, z, w) -- ROS order, which is what the rest
of this tree passes around.
"""
import math

import numpy as np

__all__ = [
    "PoseSetInadequate",
    "rt_to_T", "T_to_rt", "T_inv", "quat_to_R", "R_to_quat", "R_to_rpy",
    "rpy_to_R", "so3_log", "so3_exp", "se3_log", "se3_exp", "pose_delta",
    "handeye_pairs", "pose_set_conditioning", "solve_handeye", "refine_handeye",
    "canonicalise_flips", "HandEyeResult",
    "fit_plane", "fit_sphere", "plane_agreement",
    "check_profile", "check_plane_support",
    "KEYENCE_MIN_PROFILE_RMS", "KEYENCE_MAX_SAME_FRACTION",
    "KEYENCE_MIN_PROFILE_POINTS", "PLANE_MIN_WIDTH",
    "PlateArtifact", "SteppedArtifact", "solve_surface_fit", "SurfaceFitResult",
    "board_object_points",
    "CAL_XACRO_PROPERTIES", "CAL_REF_DIST_PROPERTY",
    "mount_cal", "sensor_transform", "cal_to_properties", "properties_to_cal",
    "render_calibration_xacro", "parse_calibration_xacro",
]


class PoseSetInadequate(Exception):
    """The observations cannot determine the transform, whatever they fit."""


# ------------------------------------------------------------------------------
# SE(3), the boring half
# ------------------------------------------------------------------------------

def rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def T_to_rt(T):
    T = np.asarray(T, float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def T_inv(T):
    R, t = T_to_rt(T)
    return rt_to_T(R.T, -R.T.dot(t))


def quat_to_R(q):
    """(x, y, z, w) -> 3x3. ROS order; normalised on the way in."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def R_to_quat(R):
    """3x3 -> (x, y, z, w), w >= 0 so two calls on the same rotation agree."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0)*2.0
        q = np.array([(R[2, 1] - R[1, 2])/s, (R[0, 2] - R[2, 0])/s,
                      (R[1, 0] - R[0, 1])/s, 0.25*s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])*2.0
        q = np.array([0.25*s, (R[0, 1] + R[1, 0])/s, (R[0, 2] + R[2, 0])/s,
                      (R[2, 1] - R[1, 2])/s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])*2.0
        q = np.array([(R[0, 1] + R[1, 0])/s, 0.25*s, (R[1, 2] + R[2, 1])/s,
                      (R[0, 2] - R[2, 0])/s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])*2.0
        q = np.array([(R[0, 2] + R[2, 0])/s, (R[1, 2] + R[2, 1])/s, 0.25*s,
                      (R[1, 0] - R[0, 1])/s])
    q = q/np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[3] >= 0.0 else -q))


def R_to_rpy(R):
    """(roll, pitch, yaw) in the fixed-axis XYZ convention urdf `rpy` means."""
    R = np.asarray(R, float)
    sy = -R[2, 0]
    sy = min(max(sy, -1.0), 1.0)
    pitch = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock: fold yaw into roll
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def rpy_to_R(roll, pitch, yaw):
    """The inverse of R_to_rpy -- Rz(yaw) Ry(pitch) Rx(roll), as urdf reads it."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def so3_log(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1.0)/2.0
    c = min(max(c, -1.0), 1.0)
    theta = math.acos(c)
    if theta < 1e-9:
        # First order; the antisymmetric part IS the rotation vector here.
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])/2.0
    if abs(theta - math.pi) < 1e-6:
        # Near pi the antisymmetric part vanishes; take the axis from R + I,
        # whose columns are all parallel to it, and pick the biggest.
        A = (R + np.eye(3))/2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k]
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.zeros(3)
        axis = axis/n
        # Sign from whichever off-diagonal is well away from zero.
        s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(s, axis) < 0.0:
            axis = -axis
        return axis*theta
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])*(theta/(2.0*math.sin(theta)))


def so3_exp(w):
    w = np.asarray(w, float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if theta < 1e-9:
        return np.eye(3) + K + 0.5*K.dot(K)
    return (np.eye(3) + (math.sin(theta)/theta)*K
            + ((1.0 - math.cos(theta))/(theta*theta))*K.dot(K))


def se3_exp(xi):
    """xi = (vx, vy, vz, wx, wy, wz) -> 4x4. Translation first, as the residual
    vectors below are ordered, so a reported sigma reads mm then mrad."""
    xi = np.asarray(xi, float).reshape(6)
    v, w = xi[:3], xi[3:]
    R = so3_exp(w)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        V = np.eye(3)
    else:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])/theta
        V = (np.eye(3) + ((1.0 - math.cos(theta))/theta)*K
             + ((theta - math.sin(theta))/theta)*K.dot(K))
    return rt_to_T(R, V.dot(v))


def se3_log(T):
    """4x4 -> (vx, vy, vz, wx, wy, wz), the inverse of se3_exp."""
    R, t = T_to_rt(T)
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        v = t.copy()
    else:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])/theta
        Vinv = (np.eye(3) - (theta/2.0)*K
                + (1.0 - (theta/2.0)/math.tan(theta/2.0))*K.dot(K))
        v = Vinv.dot(t)
    return np.concatenate([v, w])


def pose_delta(Ta, Tb):
    """(metres, radians) between two poses -- how far apart and how far turned."""
    D = T_inv(Ta).dot(Tb)
    R, t = T_to_rt(D)
    return float(np.linalg.norm(t)), float(np.linalg.norm(so3_log(R)))


# ------------------------------------------------------------------------------
# The pose set, judged before it is trusted
# ------------------------------------------------------------------------------

def handeye_pairs(G, C, consecutive_only=False):
    """[(A, B)] for AX = XB, over every pair of poses.

    ALL pairs, not consecutive ones. Consecutive pairing is the textbook
    presentation and it throws away most of the rotation diversity a set has:
    a 20 pose set has 190 pairs and 19 consecutive ones, and the 19 are the
    SMALLEST rotations in the set, which is exactly the direction that makes
    the translation ill-conditioned. Pairs with almost no relative rotation
    contribute nothing but noise and are dropped below.
    """
    G = [np.asarray(g, float) for g in G]
    C = [np.asarray(c, float) for c in C]
    if len(G) != len(C):
        raise ValueError("%d robot poses against %d observations" % (len(G), len(C)))
    idx = ([(i, i + 1) for i in range(len(G) - 1)] if consecutive_only else
           [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))])
    out = []
    for i, j in idx:
        A = T_inv(G[j]).dot(G[i])
        B = C[j].dot(T_inv(C[i]))
        out.append((A, B))
    return out


#: Below this relative rotation a pair says nothing useful about translation
#: and a great deal of noise: 1 degree of rotation amplifies the observation
#: error 57-fold. Pairs under it are dropped, and the count is reported.
MIN_PAIR_ROTATION_RAD = math.radians(5.0)

#: The verdict thresholds. `amplification` is how much an error in the observed
#: target position is multiplied on its way into the sensor offset; it is
#: 1/(2 sin(theta/2)) for a single pair and the honest generalisation of it for
#: a set. 3.0 is roughly what a set spanning +/-40 degrees achieves.
AMPLIFICATION_GOOD = 3.0
AMPLIFICATION_MARGINAL = 10.0

#: rms rotation about the WEAKEST principal axis of the pose set. Under this
#: the pose set is effectively planar in rotation and one component of the
#: sensor offset is determined by nothing.
WEAK_AXIS_GOOD_DEG = 15.0
WEAK_AXIS_MARGINAL_DEG = 5.0


def pose_set_conditioning(G, C=None, pairs=None):
    """What this pose set can and cannot determine, before anything is fitted.

    Returns a dict. The three numbers that matter:

      rot_axis_rms_deg  three values, the rms relative rotation about each
                        principal axis of the set. The SMALLEST is the one to
                        read: it is the axis about which the arm barely turned,
                        and the sensor offset along it is the component the fit
                        will invent. Three big values is a good pose set.
      amplification     multiply an observation error by this to get the error
                        it causes in the sensor's translation. Infinite when the
                        rotations share an axis.
      verdict           "good" / "marginal" / "inadequate", and `problems` says
                        why in words meant for the operator.

    Depends only on the ROBOT poses (the A side), so it can be run on a
    PROPOSED pose set, before the arm has moved -- which is the point. A set
    found inadequate at the prompt costs nothing; one found inadequate after an
    hour at the cell costs the hour.
    """
    if pairs is None:
        if C is None:
            C = [np.eye(4)]*len(G)
        pairs = handeye_pairs(G, C)
    kept, dropped = [], 0
    for A, B in pairs:
        if float(np.linalg.norm(so3_log(A[:3, :3]))) < MIN_PAIR_ROTATION_RAD:
            dropped += 1
            continue
        kept.append((A, B))

    out = {"n_poses": len(G), "n_pairs": len(pairs), "n_pairs_used": len(kept),
           "n_pairs_dropped": dropped, "problems": []}
    if len(kept) < 3:
        out.update(rot_axis_rms_deg=[0.0, 0.0, 0.0], max_rotation_deg=0.0,
                   amplification=float("inf"), verdict="inadequate")
        out["problems"].append(
            "only %d pose pair%s turn the arm more than %.0f degrees. Hand-eye "
            "cannot locate a sensor from poses that do not rotate: the "
            "translation enters only through (R-I), which is zero when R is. "
            "Add poses at genuinely different tool orientations."
            % (len(kept), "" if len(kept) == 1 else "s",
               math.degrees(MIN_PAIR_ROTATION_RAD)))
        return out

    alpha = np.array([so3_log(A[:3, :3]) for A, _ in kept])
    S = alpha.T.dot(alpha)/len(kept)
    rot_sv = np.sqrt(np.maximum(np.linalg.svd(S, compute_uv=False), 0.0))
    out["rot_axis_rms_deg"] = [float(math.degrees(v)) for v in rot_sv]
    out["max_rotation_deg"] = float(math.degrees(
        max(np.linalg.norm(a) for a in alpha)))

    # The matrix the translation is actually solved through. Its smallest
    # singular value IS the conditioning of t_X, and nothing else is.
    K = np.vstack([A[:3, :3] - np.eye(3) for A, _ in kept])
    sv = np.linalg.svd(K, compute_uv=False)/math.sqrt(len(kept))
    out["translation_sv"] = [float(v) for v in sv]
    out["amplification"] = (float("inf") if sv[-1] < 1e-12 else float(1.0/sv[-1]))

    amp, weak = out["amplification"], min(out["rot_axis_rms_deg"])
    if amp > AMPLIFICATION_MARGINAL or weak < WEAK_AXIS_MARGINAL_DEG:
        out["verdict"] = "inadequate"
    elif amp > AMPLIFICATION_GOOD or weak < WEAK_AXIS_GOOD_DEG:
        out["verdict"] = "marginal"
    else:
        out["verdict"] = "good"

    if weak < WEAK_AXIS_MARGINAL_DEG:
        out["problems"].append(
            "the arm turns only %.1f degrees rms about the weakest axis of this "
            "pose set (the other two: %.1f, %.1f). The sensor offset along that "
            "axis is not measured by these poses -- whatever comes out of the "
            "fit for it came from the initial guess and the noise, and the "
            "residual will not show it. Add poses that turn the tool about a "
            "different axis."
            % tuple([weak] + sorted(out["rot_axis_rms_deg"])[1:]))
    elif weak < WEAK_AXIS_GOOD_DEG:
        out["problems"].append(
            "only %.1f degrees rms of rotation about the weakest axis of the "
            "pose set; %.0f is comfortable. The offset along that axis is the "
            "least trustworthy part of this answer." % (weak, WEAK_AXIS_GOOD_DEG))
    if amp > AMPLIFICATION_MARGINAL:
        out["problems"].append(
            "observation error is amplified %.0fx into the sensor offset. A "
            "0.5 mm error in locating the target becomes %.0f mm of error in "
            "where the sensor is thought to be. Wider rotations, not more "
            "poses, are the fix." % (amp, amp*0.5))
    elif amp > AMPLIFICATION_GOOD:
        out["problems"].append(
            "observation error is amplified %.1fx into the sensor offset "
            "(%.1f mm of offset error per 0.5 mm of observation error)."
            % (amp, amp*0.5))
    return out


# ------------------------------------------------------------------------------
# The closed form
# ------------------------------------------------------------------------------

def _rotation_park_martin(kept):
    """R_X from the rotation parts alone (Park & Martin 1994).

    alpha_i = log(R_A), beta_i = log(R_B), and R_X beta = alpha. With
    M = sum beta alpha^T the closed form R_X = (M^T M)^-1/2 M^T is exact when
    the data are, and is the least-squares answer when they are not.
    """
    alpha = np.array([so3_log(A[:3, :3]) for A, _ in kept])
    beta = np.array([so3_log(B[:3, :3]) for _, B in kept])
    M = beta.T.dot(alpha)
    w, V = np.linalg.eigh(M.T.dot(M))
    w = np.maximum(w, 1e-18)
    R = V.dot(np.diag(1.0/np.sqrt(w))).dot(V.T).dot(M.T)
    # Re-orthonormalise: noise leaves the product slightly off SO(3), and a
    # determinant of -1 (which happens when the pose set is rank deficient)
    # must be turned back into a rotation rather than propagated.
    U, _s, Vt = np.linalg.svd(R)
    R = U.dot(np.diag([1.0, 1.0, float(np.linalg.det(U.dot(Vt)))])).dot(Vt)
    return R


def _translation_ls(kept, R_X):
    """t_X from (R_A - I) t_X = R_X t_B - t_A, stacked and least-squared."""
    rows, rhs = [], []
    for A, B in kept:
        rows.append(A[:3, :3] - np.eye(3))
        rhs.append(R_X.dot(B[:3, 3]) - A[:3, 3])
    t, _res, _rank, _sv = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs),
                                          rcond=None)
    return t


class HandEyeResult(object):
    """A solved transform and everything needed to disbelieve it."""

    def __init__(self, X, W, conditioning, per_pose, sigma, name="sensor",
                 refined=False, n_iter=0, flips=()):
        #: flange -> sensor, 4x4. The answer.
        self.X = np.asarray(X, float)
        #: base -> target, 4x4. Where the solve places the artifact.
        self.W = np.asarray(W, float)
        self.conditioning = conditioning
        #: [{pose, t_mm, rot_mrad}] -- how far each observation sits from the
        #: consensus. Per observation, because one bad detection is a different
        #: illness from a bad pose set and an rms hides both.
        self.per_pose = list(per_pose)
        #: 1 sigma on X's six degrees of freedom, IN THE FLANGE FRAME:
        #: (x_mm, y_mm, z_mm, rx_mrad, ry_mrad, rz_mrad). None when not refined.
        self.sigma = sigma
        self.name = name
        self.refined = refined
        self.n_iter = n_iter
        #: indices whose detection was 180-degree flipped and corrected.
        self.flips = list(flips)

    # -- the numbers an operator reads --------------------------------------

    @property
    def xyz(self):
        return tuple(float(v) for v in self.X[:3, 3])

    @property
    def quat(self):
        return R_to_quat(self.X[:3, :3])

    @property
    def rpy(self):
        return R_to_rpy(self.X[:3, :3])

    @property
    def adequate(self):
        return self.conditioning.get("verdict") != "inadequate"

    @property
    def residual_mm(self):
        return (float(np.sqrt(np.mean([p["t_mm"]**2 for p in self.per_pose])))
                if self.per_pose else 0.0)

    @property
    def residual_mrad(self):
        return (float(np.sqrt(np.mean([p["rot_mrad"]**2 for p in self.per_pose])))
                if self.per_pose else 0.0)

    @property
    def worst(self):
        return max(self.per_pose, key=lambda p: p["t_mm"]) if self.per_pose else None

    def outliers(self, k=3.0):
        """Observations more than `k` times the rms from the consensus. One of
        these is a misdetection, not a calibration error; drop it and re-solve
        rather than averaging it in."""
        r = self.residual_mm
        if r <= 0.0:
            return []
        return [p for p in self.per_pose if p["t_mm"] > k*r]

    def report(self):
        """The whole truth, as text. Printed by the .dpi and written to the
        result file, because a residual nobody reads is a residual nobody
        checked."""
        c = self.conditioning
        L = []
        L.append("%s calibration -- flange -> %s" % (self.name, self.name))
        L.append("  xyz  %+.5f %+.5f %+.5f  m" % self.xyz)
        L.append("  rpy  %+.5f %+.5f %+.5f  rad  (%+.2f %+.2f %+.2f deg)"
                 % (self.rpy + tuple(math.degrees(v) for v in self.rpy)))
        L.append("  quat %+.6f %+.6f %+.6f %+.6f  (x y z w)" % self.quat)
        L.append("")
        L.append("  POSE SET: %d poses, %d of %d pairs used (%d too small to help)"
                 % (c.get("n_poses", 0), c.get("n_pairs_used", 0),
                    c.get("n_pairs", 0), c.get("n_pairs_dropped", 0)))
        if "rot_axis_rms_deg" in c:
            L.append("    rotation about the set's three axes: %.1f, %.1f, %.1f deg rms"
                     % tuple(sorted(c["rot_axis_rms_deg"], reverse=True)))
            L.append("    largest single rotation: %.1f deg" % c.get("max_rotation_deg", 0.0))
        amp = c.get("amplification", float("inf"))
        L.append("    observation error amplified %s into the offset"
                 % ("infinitely" if not np.isfinite(amp) else "%.1fx" % amp))
        L.append("    VERDICT: %s" % c.get("verdict", "?").upper())
        for p in c.get("problems", ()):
            L.append("      - " + p)
        L.append("")
        L.append("  RESIDUALS (each observation against the consensus target pose)")
        L.append("    rms %.3f mm, %.2f mrad over %d observations"
                 % (self.residual_mm, self.residual_mrad, len(self.per_pose)))
        for p in self.per_pose:
            mark = "  <-- worst" if p is self.worst and len(self.per_pose) > 1 else ""
            L.append("    pose %-3s %7.3f mm %7.2f mrad%s"
                     % (p.get("pose"), p["t_mm"], p["rot_mrad"], mark))
        out = self.outliers()
        if out:
            L.append("    %d observation(s) more than 3x the rms out: %s. That is a "
                     "MISDETECTION, not a calibration error -- drop them and solve "
                     "again." % (len(out), ", ".join(str(p.get("pose")) for p in out)))
        if self.flips:
            L.append("    %d detection(s) were 180 degrees flipped and were "
                     "corrected: %s" % (len(self.flips), self.flips))
        if self.sigma is not None:
            L.append("")
            L.append("  UNCERTAINTY (1 sigma, in the flange frame, from the fit)")
            L.append("    translation  %.3f  %.3f  %.3f  mm" % tuple(self.sigma[:3]))
            L.append("    rotation     %.3f  %.3f  %.3f  mrad" % tuple(self.sigma[3:]))
            worst_t = max(self.sigma[:3])
            if worst_t > 2.0:
                L.append("    THIS IS NOT GOOD ENOUGH TO SCAN FROM: %.1f mm of "
                         "uncertainty in the sensor position. The error this "
                         "procedure exists to prevent is about 13 mm." % worst_t)
        if not self.adequate:
            L.append("")
            L.append("  *** THE POSE SET DOES NOT DETERMINE THIS TRANSFORM. The "
                     "numbers above fit the observations and are still wrong. "
                     "Do not write them anywhere. ***")
        return "\n".join(L)

    def __repr__(self):
        return ("<HandEyeResult %s xyz=(%.4f, %.4f, %.4f) rms=%.3f mm %s>"
                % ((self.name,) + self.xyz + (self.residual_mm,
                                              self.conditioning.get("verdict"))))


def canonicalise_flips(G, C, board_symmetry=math.pi):
    """Undo the 180-degree ambiguity a checkerboard detection has.

    A checkerboard's inner-corner grid maps onto itself when turned half a turn
    in its own plane, so a detector may return the corners either way round and
    nothing in the image says which. A CONSISTENT choice is harmless -- it is
    absorbed into W, the target's pose in the base frame, which nobody
    independently knows, and the sensor transform comes out identical. An
    INCONSISTENT one is fatal: two poses then tell incompatible stories about
    where the target is, and the solver fits a transform to their average.

    HOW THE FLIP IS FOUND WITHOUT KNOWING X. AX = XB makes A and B conjugate,
    so however the sensor is mounted, the observed relative rotation must have
    the SAME ANGLE as the arm's:

        angle( C_j C_i^-1 )  ==  angle( G_j^-1 G_i )

    which holds for every pair and involves no unknown. A flip of one
    observation breaks it by a large margin, so each observation is taken as it
    is or turned half a turn about its own z, whichever keeps that equality
    against every observation already accepted. (Comparing an observation to
    its NEIGHBOUR instead is the obvious thing and does not work: a calibration
    pose set turns the tool a long way on purpose, so a real rotation is not
    small and cannot be told from a flip by size alone.)

    Returns (C_fixed, flipped_indices).
    """
    G = [np.asarray(g, float) for g in G]
    C = [np.asarray(c, float) for c in C]
    if len(C) < 2:
        return C, []
    F = rt_to_T(so3_exp(np.array([0.0, 0.0, board_symmetry])), np.zeros(3))
    out = [C[0]]
    kept = [0]
    flipped = []
    for i in range(1, len(C)):
        cand = [C[i], C[i].dot(F)]
        cost = []
        for c in cand:
            e = 0.0
            for j, Cj in zip(kept, out):
                a = np.linalg.norm(so3_log(T_inv(G[i]).dot(G[j])[:3, :3]))
                b = np.linalg.norm(so3_log(c.dot(T_inv(Cj))[:3, :3]))
                e += (a - b)**2
            cost.append(e)
        k = int(np.argmin(cost))
        if k:
            flipped.append(i)
        out.append(cand[k])
        kept.append(i)
    return out, flipped


def solve_handeye(G, C, name="sensor", refine=True, require_adequate=False,
                  fix_flips=False, drop=()):
    """Solve for flange -> sensor from robot poses G and observations C.

    G[i] is the flange in the base frame (4x4); C[i] is the calibration target
    in the sensor's frame (4x4) at that same pose. Both as homogeneous
    matrices, metres.

    `require_adequate=True` RAISES PoseSetInadequate instead of returning an
    answer the pose set cannot support -- which is what an automated caller
    should pass, and what a human at the prompt should not, because the report
    on a bad set is how you find out what to add.

    `fix_flips=True` applies canonicalise_flips() first; it is off by default
    because it is only right for a target with a half-turn symmetry.

    Returns a HandEyeResult. It carries the answer, the conditioning, the
    per-observation residuals and a 1 sigma on all six degrees of freedom, and
    `.adequate` is False when the answer should not be believed whatever the
    residual says.
    """
    G = [np.asarray(g, float) for g in G]
    C = [np.asarray(c, float) for c in C]
    labels = list(range(len(G)))
    if drop:
        keep = [i for i in range(len(G)) if i not in set(drop)]
        G, C, labels = [G[i] for i in keep], [C[i] for i in keep], keep
    if len(G) < 3:
        raise PoseSetInadequate(
            "%d pose(s). Hand-eye needs at least 3, and wants 15 or more at "
            "genuinely different tool orientations." % len(G))

    flips = []
    if fix_flips:
        C, flips = canonicalise_flips(G, C)

    pairs = handeye_pairs(G, C)
    cond = pose_set_conditioning(G, pairs=pairs)
    kept = [(A, B) for A, B in pairs
            if float(np.linalg.norm(so3_log(A[:3, :3]))) >= MIN_PAIR_ROTATION_RAD]
    if len(kept) < 3:
        res = HandEyeResult(np.eye(4), np.eye(4), cond, [], None, name=name)
        if require_adequate:
            raise PoseSetInadequate(res.report())
        return res

    R_X = _rotation_park_martin(kept)
    t_X = _translation_ls(kept, R_X)
    X = rt_to_T(R_X, t_X)
    W = _consensus_target(G, C, X)

    n_iter = 0
    sigma = None
    if refine:
        X, W, n_iter, sigma = refine_handeye(G, C, X, W)

    per_pose = []
    for lab, Gi, Ci in zip(labels, G, C):
        dt, dr = pose_delta(Gi.dot(X).dot(Ci), W)
        per_pose.append({"pose": lab, "t_mm": dt*1e3, "rot_mrad": dr*1e3})

    res = HandEyeResult(X, W, cond, per_pose, sigma, name=name,
                        refined=bool(refine), n_iter=n_iter, flips=flips)
    if require_adequate and not res.adequate:
        raise PoseSetInadequate(res.report())
    return res


def _consensus_target(G, C, X):
    """The target pose in the base frame the observations agree on -- the
    average of G_i X C_i, with the rotations averaged on the manifold rather
    than component-wise, which is not a rotation."""
    Ts = [np.asarray(g, float).dot(X).dot(np.asarray(c, float))
          for g, c in zip(G, C)]
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    R = Ts[0][:3, :3]
    for _ in range(20):
        d = np.mean([so3_log(R.T.dot(T[:3, :3])) for T in Ts], axis=0)
        R = R.dot(so3_exp(d))
        if np.linalg.norm(d) < 1e-12:
            break
    return rt_to_T(R, t)


def refine_handeye(G, C, X0, W0, max_iter=40, sigma_t=None, sigma_r=None):
    """Gauss-Newton over X and W together, and the covariance that falls out.

    The closed form above uses pose PAIRS, which double-counts noise and
    weights nothing. This minimises, over all twelve parameters at once,

        r_i = log( (G_i X C_i)^-1 W )      in se(3)

    which is the direct statement that the target did not move. The Jacobian is
    numeric: 12 columns over a handful of poses is nothing, and a hand-derived
    SE(3) Jacobian is a sign error waiting to happen in code that is only
    exercised at a work cell.

    WHY THE COVARIANCE IS THE POINT. sigma comes from (J^T J)^-1 scaled by the
    residual variance actually observed, restricted to X's six parameters. It
    is the number that distinguishes "fits well" from "is determined": a
    direction the pose set does not constrain has a huge sigma AND a small
    residual, and only one of those two is visible in an rms.

    Returns (X, W, n_iter, sigma) with sigma = (x_mm, y_mm, z_mm, rx_mrad,
    ry_mrad, rz_mrad), 1 sigma, in the flange frame.
    """
    G = [np.asarray(g, float) for g in G]
    C = [np.asarray(c, float) for c in C]
    X, W = np.asarray(X0, float).copy(), np.asarray(W0, float).copy()

    # Rotation residuals are radians and translation metres; unweighted, a
    # metre and a radian would count the same and the fit would be all
    # rotation. These scale them to comparable significance and are re-derived
    # from the fit afterwards, so the answer does not depend much on them.
    st = 0.0005 if sigma_t is None else float(sigma_t)
    sr = 0.002 if sigma_r is None else float(sigma_r)
    w = np.array([1.0/st]*3 + [1.0/sr]*3)

    def residuals(X_, W_):
        return np.concatenate([w*se3_log(T_inv(g.dot(X_).dot(c)).dot(W_))
                               for g, c in zip(G, C)])

    r = residuals(X, W)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        J = np.zeros((r.size, 12))
        eps = 1e-7
        for k in range(12):
            d = np.zeros(12)
            d[k] = eps
            Xp = X.dot(se3_exp(d[:6])) if k < 6 else X
            Wp = W if k < 6 else W.dot(se3_exp(d[6:]))
            J[:, k] = (residuals(Xp, Wp) - r)/eps
        step, _res, _rank, _sv = np.linalg.lstsq(J, -r, rcond=None)
        X = X.dot(se3_exp(step[:6]))
        W = W.dot(se3_exp(step[6:]))
        r_new = residuals(X, W)
        if np.linalg.norm(r_new - r) < 1e-12 or np.linalg.norm(step) < 1e-12:
            r = r_new
            break
        r = r_new

    # Covariance. dof = observations - parameters; W's six are free too, and a
    # three pose set has none to spare, which is itself worth saying.
    n_obs = r.size
    dof = max(n_obs - 12, 1)
    # Unweight, then re-weight by what the residuals actually were: the fit's
    # own estimate of the observation noise beats a guess at it.
    rr = r.reshape(-1, 6)/w
    st_hat = max(float(np.sqrt(np.sum(rr[:, :3]**2)/dof)), 1e-9)
    sr_hat = max(float(np.sqrt(np.sum(rr[:, 3:]**2)/dof)), 1e-12)
    Wt = np.array([1.0/st_hat]*3 + [1.0/sr_hat]*3)
    Jn = J/w[None, :].repeat(n_obs//6, axis=0).reshape(-1, 1)*Wt[None, :].repeat(
        n_obs//6, axis=0).reshape(-1, 1)
    try:
        cov = np.linalg.pinv(Jn.T.dot(Jn), rcond=1e-14)[:6, :6]
        s = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sigma = np.array([s[0]*1e3, s[1]*1e3, s[2]*1e3,
                          s[3]*1e3, s[4]*1e3, s[5]*1e3])
    except np.linalg.LinAlgError:
        sigma = np.array([float("inf")]*6)
    return X, W, n_iter, sigma


# ------------------------------------------------------------------------------
# What the sensors actually see on the artifact
# ------------------------------------------------------------------------------

def fit_plane(points):
    """(normal, point_on_plane, rms, n) by total least squares.

    The normal is unit and its sign is chosen so it points towards +z, because
    every surface this is used on is looked at from above and a normal that
    flips between two measurements makes their comparison nonsense.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    if P.shape[0] < 3:
        raise ValueError("a plane needs 3 finite points, got %d" % P.shape[0])
    c = P.mean(axis=0)
    _u, _s, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    if n[2] < 0.0:
        n = -n
    rms = float(np.sqrt(np.mean(((P - c).dot(n))**2)))
    return n, c, rms, int(P.shape[0])


def fit_sphere(points, radius=None):
    """(centre, radius, rms, n). Algebraic fit, then one Gauss-Newton polish.

    `radius=None` fits it. GIVE THE KNOWN RADIUS INSTEAD when there is one and
    you want the centre: a partial cap -- which is all a camera or a profiler
    ever sees of a ball -- determines the radius terribly and the centre well
    only once the radius is fixed. Fitting the radius FREE is worth doing once,
    on purpose, as a diagnostic: a ball of known diameter whose free-fit radius
    comes out 3% large is a depth SCALE error, and one whose radius is right but
    whose centre sits along the line of sight from where it should be is a depth
    OFFSET. The two need different fixes and a fixed-radius fit hides both.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    if P.shape[0] < 4:
        raise ValueError("a sphere needs 4 finite points, got %d" % P.shape[0])
    # |p|^2 = 2 c.p + (r^2 - |c|^2): linear in (c, k).
    A = np.hstack([2.0*P, np.ones((P.shape[0], 1))])
    b = np.sum(P*P, axis=1)
    sol, _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(max(sol[3] + c.dot(c), 1e-12))
    if radius is not None:
        r = float(radius)
    for _ in range(20):                     # geometric distance, not algebraic
        d = P - c
        n = np.linalg.norm(d, axis=1)
        n = np.where(n < 1e-12, 1e-12, n)
        u = d/n[:, None]
        resid = n - r
        if radius is None:
            J = np.hstack([-u, -np.ones((P.shape[0], 1))])
            step, _r2, _rk, _s2 = np.linalg.lstsq(J, -resid, rcond=None)
            c = c + step[:3]
            r = r + step[3]
        else:
            step, _r2, _rk, _s2 = np.linalg.lstsq(-u, -resid, rcond=None)
            c = c + step
        if np.linalg.norm(step) < 1e-12:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r)**2)))
    return c, float(r), rms, int(P.shape[0])


def plane_agreement(points_a, points_b, at=None, name_a="A", name_b="B"):
    """Do two sensors agree about one flat surface? THE routine check.

    Both point sets are in the WORLD frame, each from one sensor, each looking
    at the same flat patch. Returns a dict with

      dz_mm       how far apart the two planes are at `at` (default: the
                  centroid of the two patches). The sign is b - a.
      tilt_mrad   the angle between the two fitted normals.
      rms_a/rms_b each sensor's own flatness about its own fit, which says
                  whether either fit deserves to be compared at all.

    This is the whole verification step, and it needs no touch-off, no
    artifact, and no known surface -- only one flat thing both sensors can see.
    An uncalibrated Azure shows up here directly, as a dz of order -13 mm.
    """
    na, ca, rms_a, n_a = fit_plane(points_a)
    nb, cb, rms_b, n_b = fit_plane(points_b)
    at = (ca + cb)/2.0 if at is None else np.asarray(at, float).reshape(3)
    # Height of each plane above `at`, measured along world z, which is the
    # direction a standoff error actually acts in.
    za = ca[2] - (at[:2] - ca[:2]).dot(na[:2])/na[2] if abs(na[2]) > 1e-6 else float("nan")
    zb = cb[2] - (at[:2] - cb[:2]).dot(nb[:2])/nb[2] if abs(nb[2]) > 1e-6 else float("nan")
    cosang = float(np.clip(na.dot(nb), -1.0, 1.0))
    return {"name_a": name_a, "name_b": name_b,
            "z_a": float(za), "z_b": float(zb),
            "dz_mm": float((zb - za)*1e3),
            "tilt_mrad": float(math.acos(cosang)*1e3),
            "rms_a_mm": rms_a*1e3, "rms_b_mm": rms_b*1e3,
            "n_a": n_a, "n_b": n_b,
            "at": [float(v) for v in at],
            "normal_a": [float(v) for v in na], "normal_b": [float(v) for v in nb]}


# ------------------------------------------------------------------------------
# IS IT A MEASUREMENT AT ALL?
# ------------------------------------------------------------------------------
# A degenerate Keyence grab makes verify_calibration() print something like
#
#     keyence  z = 0.6632 m   0.000 mm rms flat, 48000 points from 60 profiles
#     azure    z = 0.9076 m   1.109 mm rms flat, 1034 points
#     AZURE IS +244.33 mm AND 1571.91 mrad OFF THE KEYENCE
#
# which reads, reasonably, as "the Azure is badly out" and says nothing
# whatever about the Azure. Three of those numbers are impossible before anyone
# gets as far as disagreeing about a plate:
#
#   0.000 mm rms of 48 000 points   a real LJ-V7080 scatters 7 um within a
#                                   profile on a sprayed flat and 24 um on bare
#                                   metal, and its own repeatability is 0.5 um.
#                                   Exactly zero is a constant, not a surface.
#   z = 0.6632 m                    257 mm from the optical frame, in a head
#                                   that can only measure over 57..103 mm.
#   1571.91 mrad                    90.06 degrees. Not an error: an artefact of
#                                   fitting a plane to points that lie on a line.
#
# Two checks, neither of which needs to know the right answer:
#
#   check_profile()        one grab off the head, in the head's own frame. Is
#                          there scatter, and is the surface in the window?
#   check_plane_support()  a point set about to be handed to fit_plane(). Does
#                          it have extent in TWO directions, or is the plane
#                          about to be invented?
#
# Both are pure geometry over numpy arrays, so both are tested on the host
# against synthetic data that is deliberately good and deliberately bad --
# calibration_test.py, test_profile_* and test_plane_support_*.

#: Scatter below which an LJ-V7080 profile is a CONSTANT and not a surface:
#: the RMS residual of the profile's own heights about its own best-fit line.
#:
#: PICKED FROM THE HEAD, NOT FROM ZERO. The LJ-V7080's z repeatability is
#: 0.5 um, so 0.5 um is the smallest scatter physics allows even on a perfect
#: surface; the best this cell has actually measured within a profile is 7 um
#: on a sprayed flat and 24 um on bare metal. 1 um is twice the head's own
#: repeatability floor and seven times below the best real measurement, so an
#: excellent profile passes with a factor of seven in hand and a repeated
#: value -- which scores exactly 0 -- cannot.
KEYENCE_MIN_PROFILE_RMS = 1.0e-6

#: Refuse when this fraction of the finite points carry one identical height.
#: A stuck reading that dithers in the last bit defeats the RMS test above by a
#: hair; it does not defeat this. 0.90 is far above anything real: at the head's
#: 0.5 um resolution, 7 um of noise spreads a flat over ~50 quantisation steps
#: and no single value holds more than a few percent of the points.
KEYENCE_MAX_SAME_FRACTION = 0.90

#: Fewest finite points a grab must have before the scatter test means anything
#: (a line fit through 3 points has no residual by construction). A real grab is
#: 800 columns per profile.
KEYENCE_MIN_PROFILE_POINTS = 10

#: Narrowest a point set may be, across its own long axis, and still determine
#: a plane. One static Keyence pose is 31 mm long and MICRONS wide -- a line.
#: 5 mm is far below any honest patch (the Azure sees 90 mm, a Keyence swept
#: across its line 40 mm) and far above any single profile.
PLANE_MIN_WIDTH = 0.005


def check_profile(points, ref_dist=0.080, range_m=(0.057, 0.103),
                  min_rms=None, max_same=None, min_points=None,
                  source="the Keyence", n_profiles=None):
    """Is this grab off a profiler a measurement? Returns a dict, never raises.

    `points` are N x 3 IN THE HEAD'S OWN OPTICAL FRAME, as keyence_driver_node
    publishes them: x across the laser line, y identically 0, z the device's own
    reading with POSITIVE z TOWARDS THE HEAD. So the distance from the head is

        standoff = ref_dist - z

    -- the same expression, and the same sign, as keyence_profile_standoff() in
    keyence_standoff.dpi, which is where that convention is written down and
    tested.

    `["ok"]` is the answer and `["message"]` is what to tell the operator. The
    measurements are returned whether it passes or fails, because "7 um of
    scatter, standoff 80.1 mm" is worth printing on a good day too.

    THE SCATTER TEST IS ON z ABOUT A LINE, NOT ON A PLANE FIT. Every point in
    one grab has y exactly 0, so a plane fitted to them sits in the y = 0 plane
    and its residual is exactly zero for good data and bad alike. The profile's
    own shape -- height against position across the line -- is the thing that
    has noise in it. A flat seen at a tilt IS a line, so the linear part is
    removed before the residual is taken; anything curved only makes the
    residual bigger, and this test only ever refuses it for being too small.
    """
    min_rms = KEYENCE_MIN_PROFILE_RMS if min_rms is None else float(min_rms)
    max_same = (KEYENCE_MAX_SAME_FRACTION if max_same is None else float(max_same))
    min_points = (KEYENCE_MIN_PROFILE_POINTS if min_points is None
                  else int(min_points))
    lo, hi = (float(range_m[0]), float(range_m[1]))
    ref = float(ref_dist)

    P = np.asarray(points, float).reshape(-1, 3)
    finite = np.isfinite(P).all(axis=1)
    F = P[finite]
    out = {"n": int(P.shape[0]), "n_finite": int(F.shape[0]),
           "finite_fraction": (float(F.shape[0])/P.shape[0]) if P.shape[0] else 0.0,
           "n_profiles": (None if n_profiles is None else int(n_profiles)),
           "ref_dist": ref, "range_m": [lo, hi],
           "min_rms": float(min_rms), "max_same": float(max_same),
           "rms": None, "span": None, "same_fraction": None,
           "standoff": None, "standoff_min": None, "standoff_max": None,
           "problems": [], "ok": False, "message": ""}

    if F.shape[0] < max(3, min_points):
        out["problems"].append(
            "POINTS: %d of %d points are finite, and %d are needed before any "
            "of this means anything. Out-of-range columns are +Inf (REP-117), "
            "so this IS the coverage: almost nothing is inside the head's "
            "%.0f..%.0f mm window."
            % (F.shape[0], P.shape[0], max(3, min_points), lo*1e3, hi*1e3))
        out["message"] = _profile_message(source, out)
        return out

    x, z = F[:, 0], F[:, 2]
    # Remove the linear part -- a flat at a tilt is a straight line across the
    # profile and its slope is not noise -- and measure what is left.
    A = np.column_stack([np.ones_like(x), x])
    coef, _res, _rank, _sv = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A.dot(coef)
    dof = max(F.shape[0] - 2, 1)
    out["rms"] = float(math.sqrt(float(resid.dot(resid))/dof))
    out["span"] = float(x.max() - x.min())
    out["slope"] = float(coef[1])

    vals, counts = np.unique(z, return_counts=True)
    out["same_fraction"] = float(counts.max())/F.shape[0]
    out["same_value"] = float(vals[int(np.argmax(counts))])

    stand = ref - z
    out["standoff"] = float(np.median(stand))
    out["standoff_min"] = float(stand.min())
    out["standoff_max"] = float(stand.max())

    if out["rms"] < min_rms:
        out["problems"].append(
            "SCATTER: the profile varies by %.4f mm RMS about its own best-fit "
            "line, over %d points. The head's repeatability is 0.5 um and this "
            "cell measures 7 um within a profile on a sprayed flat and 24 um on "
            "bare metal, so anything under %.4f mm is one repeated number and "
            "not a surface."
            % (out["rms"]*1e3, F.shape[0], min_rms*1e3))
    if not (lo <= out["standoff"] <= hi):
        out["problems"].append(
            "RANGE: the heights imply the surface is %.1f mm from the head "
            "(%.1f..%.1f mm across the line), outside the %.0f..%.0f mm the "
            "LJ-V7080 can measure over. Nothing physical is there."
            % (out["standoff"]*1e3, out["standoff_min"]*1e3,
               out["standoff_max"]*1e3, lo*1e3, hi*1e3))
    if out["same_fraction"] > max_same:
        out["problems"].append(
            "REPEATED VALUE: %.1f%% of the %d finite points carry the identical "
            "height %.6f m. At the head's 0.5 um resolution a real surface "
            "never does."
            % (out["same_fraction"]*100.0, F.shape[0], out["same_value"]))

    out["ok"] = not out["problems"]
    out["message"] = _profile_message(source, out)
    return out


def _profile_message(source, out):
    """What to tell the operator about a check_profile() result."""
    if out["ok"]:
        return ("%s returned %d points%s: %.4f mm RMS about the profile, "
                "%.1f mm of standoff, %.0f%% of columns in range."
                % (source, out["n_finite"],
                   "" if out["n_profiles"] is None
                   else " from %d profiles" % out["n_profiles"],
                   (out["rms"] or 0.0)*1e3, (out["standoff"] or 0.0)*1e3,
                   out["finite_fraction"]*100.0))
    return ("%s returned %d points%s, and they are not a measurement:\n  "
            % (source, out["n"],
               "" if out["n_profiles"] is None
               else " from %d profiles" % out["n_profiles"])
            + "\n  ".join("* " + p for p in out["problems"])
            + "\n  LIKELY CAUSE: the head is not standing off the surface "
              "correctly. Every one of these is what a profiler does when there "
              "is nothing inside its measuring window -- a pose that put the "
              "window on the surface instead of %.0f mm above it, a taught "
              "point that is not where the surface is, or the laser off. It is "
              "NOT a calibration error, and nothing downstream of this should "
              "be believed: no solve and no verdict has been reached."
              % (out["ref_dist"]*1e3,))


def check_plane_support(points, min_width=None, name="the point set"):
    """Do these points determine a plane, or only a line? A dict, never raises.

    THE OTHER HALF OF THE CHECK ABOVE. fit_plane() is a total-least-squares fit
    and it always succeeds: handed points that lie on a line it returns the
    direction of least scatter, which for a Keyence grab is the sensor's own y
    axis -- exactly 0 mm of scatter, because every point in a profile has y = 0.
    That is what produces "0.000 mm rms flat", a tilt of 1571.91 mrad (90.06
    deg, the angle between a horizontal artefact normal and the Azure's
    vertical one), and a height that is a near-vertical plane extrapolated
    sideways to the centre of the patch.

    ONE STATIC KEYENCE POSE MEASURES ONE LINE. It cannot determine a plane and
    no amount of averaging changes that; the head has to be moved ACROSS its own
    laser line and grabbed again. This says so instead of returning a number.

    Returns the extents of the point set along its own three principal axes,
    largest first: length, width, and thickness. Thickness is the flatness the
    surface actually has; width is what has to be non-zero for the fit to mean
    anything.
    """
    min_width = PLANE_MIN_WIDTH if min_width is None else float(min_width)
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    out = {"n": int(P.shape[0]), "min_width": float(min_width),
           "length": None, "width": None, "thickness": None,
           "problems": [], "ok": False, "message": ""}
    if P.shape[0] < 3:
        out["problems"].append(
            "%s has %d finite points; a plane needs 3." % (name, P.shape[0]))
        out["message"] = out["problems"][0]
        return out
    c = P.mean(axis=0)
    _u, _s, Vt = np.linalg.svd(P - c, full_matrices=False)
    proj = (P - c).dot(Vt.T)
    ext = proj.max(axis=0) - proj.min(axis=0)
    out["length"], out["width"], out["thickness"] = (float(ext[0]), float(ext[1]),
                                                     float(ext[2]))
    if out["width"] < min_width:
        out["problems"].append(
            "%s is %.1f mm long and %.3f mm wide: the points lie on a LINE, not "
            "a surface. A plane fitted to them has a normal that is chosen by "
            "nothing -- its residual comes out near zero however wrong it is, "
            "its tilt against another sensor is an artefact, and its height away "
            "from the line is extrapolation. One static profiler pose measures "
            "ONE line across the surface; move the head across its own laser "
            "line and grab again."
            % (name, out["length"]*1e3, out["width"]*1e3))
    out["ok"] = not out["problems"]
    out["message"] = ("; ".join(out["problems"]) if out["problems"] else
                      "%s is %.0f x %.0f mm and %.3f mm thick"
                      % (name, out["length"]*1e3, out["width"]*1e3,
                         out["thickness"]*1e3))
    return out


def board_object_points(cols, rows, square_m):
    """The checkerboard's inner corners in the board's own frame, z = 0.

    Ordered the way every OpenCV detector returns them: along a row first, rows
    increasing. `cols` and `rows` count INNER CORNERS, not squares -- a board of
    10 x 7 squares has 9 x 6 inner corners, and giving the square count is the
    classic way to get a detection that never succeeds.
    """
    g = np.zeros((rows*cols, 3), float)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)*float(square_m)
    return g


# ------------------------------------------------------------------------------
# The profiler: a sensor that cannot see a printed pattern
# ------------------------------------------------------------------------------

class PlateArtifact(object):
    """The calibration plate as the Keyence sees it: a plane and some balls.

    Geometry is in the ARTIFACT frame: the plate's ground top face is z = 0 with
    +z out of it, and each ball centre is given in that frame. For a ball of
    radius R seated in a shallow cone or bonded to the top face, its centre sits
    at z = R + seat_offset; whoever makes the plate says what that is, and it is
    measured once with the contact probe (four touches round the equator) rather
    than believed from a drawing.

    WHAT THE BALLS ARE AND ARE NOT FOR. The textbook artifact for a line
    profiler is a set of precision spheres, on the reasoning that a flat gives
    a profiler nothing but a straight line. That reasoning is wrong here, and
    calibration_test.py measures it rather than arguing: a bare ground flat,
    viewed at varied tilts AND AT VARIED STANDOFFS, determines all six degrees
    of freedom to 0.05 mm and 0.6 mrad at this head's 0.1 mm noise. Adding
    three balls improves that to 0.02 mm and 0.6 mrad -- real, and not the
    difference between working and not.

    What a flat genuinely cannot do is separate a RANGE error from a POSE
    error, and it cannot do it for the Azure either. A ball of known diameter
    can: fit_sphere() with a free radius returns the wrong radius for a range
    scale error and the right radius displaced along the line of sight for a
    range offset. That, and being the one feature all three sensors including
    the contact probe can agree on, is what the balls earn their place with.

    THE DEGENERACY THAT IS REAL, and that a ball does not fix either: every
    profile taken at the SAME STANDOFF leaves rotation of the head about its
    own laser line unmeasured, because at zero range the rotation moves no
    point at first order. Vary the standoff across the pose set. This is not a
    theoretical worry -- holding the head at its reference distance everywhere
    is the obvious way to take the data, and solve_surface_fit() reports the
    resulting null direction rather than quietly returning the initial guess.
    """

    def __init__(self, sphere_centres, sphere_radius, plate_extent=None,
                 name="plate"):
        self.centres = np.asarray(sphere_centres, float).reshape(-1, 3)
        self.radius = float(sphere_radius)
        #: (x_half, y_half) of the flat top face, for deciding what is plate and
        #: what is stray. None: accept any point that is not near a ball.
        self.plate_extent = plate_extent
        self.name = name

    def signed_distance(self, P, ball_margin=None):
        """(distance to the nearest feature, feature index) for each point.

        Feature -1 is the plate; 0.. are the balls. A point is assigned to a
        ball when it is within `ball_margin` of that ball's surface, which is
        how a partial cap gets claimed by the ball rather than pulled onto the
        plate. Points that belong to neither come back as NaN and are dropped.
        """
        P = np.asarray(P, float).reshape(-1, 3)
        margin = 0.35*self.radius if ball_margin is None else float(ball_margin)
        d = P[:, 2].copy()                            # plate: z = 0
        which = np.full(P.shape[0], -1, int)
        for k, c in enumerate(self.centres):
            dk = np.linalg.norm(P - c, axis=1) - self.radius
            take = np.abs(dk) < margin
            d = np.where(take, dk, d)
            which = np.where(take, k, which)
        if self.plate_extent is not None:
            hx, hy = self.plate_extent
            off = (np.abs(P[:, 0]) > hx) | (np.abs(P[:, 1]) > hy)
            d = np.where(off & (which < 0), np.nan, d)
        return d, which

    def describe(self):
        return ("%s: %d ball%s of radius %.4f m, centres %s"
                % (self.name, self.centres.shape[0],
                   "" if self.centres.shape[0] == 1 else "s", self.radius,
                   ", ".join("(%.3f, %.3f, %.3f)" % tuple(c) for c in self.centres)))


class SteppedArtifact(object):
    """The 3D-printed calibration artifact, as a height field.

    WHY A STEPPED ARTIFACT AND NOT A FLAT. solve_surface_fit() over a bare flat
    leaves rotation of the head about its own laser line unmeasured at any one
    standoff, because at zero range that rotation moves no measured point at
    first order (PlateArtifact's docstring has the measurement). The cure is to
    vary the standoff -- and a flat can only do that by FLYING the head at
    different heights, three poses where one would do.

    A stepped artifact varies it inside a single pose. This one spans 24 mm of
    height across a 160 mm square, so one profile line laid across it measures
    surfaces at seven different distances from the head at once, and the pose
    set stops having to carry that job on its own.

    THE MODEL IS A HEIGHT FIELD, NOT A MESH. Every feature is a horizontal top
    face or a cone flank, so the surface is a function z = h(x, y) and the
    residual is simply `z - h(x, y)`. On a cone flank that is the vertical
    distance rather than the perpendicular one -- larger than the true distance
    by 1/cos(slope), 1.22x for these cones -- which reweights the flanks
    slightly and moves the zero of the fit not at all.

    WHAT IS DELIBERATELY THROWN AWAY. Points within `edge_margin` horizontally
    of any feature boundary come back NaN. A pad's wall is vertical, a printed
    edge is rounded by the nozzle, and a profiler crossing one returns a smear
    of points belonging to neither face. There are ~1600 points per profile
    line and dozens of profiles, so discarding a millimetre either side of each
    edge costs nothing and removes the only systematically wrong data.

    Geometry is in the ARTIFACT frame, metres: the artifact's BOTTOM face is
    z = 0, which is also the table it stands on, and x, y run from the
    chamfered corner. That makes the surrounding table part of the model --
    a real, large, flat feature the head sees at every pose -- rather than
    stray data to be trimmed.

    THE PRINTED GEOMETRY IS NOT THE MODEL. A printer is good to a few tenths of
    a millimetre and that is the same size as the answer. `pad_heights` and the
    rest are the DESIGN; characterise the printed part once with the Keyence
    (or a height gauge) and pass the measured numbers, after which the
    printer's tolerance stops mattering. docs/calibration/make_artifact.py
    carries the design and says the same thing.
    """

    def __init__(self, pad_centres, pad_tops, pad_diameter,
                 cone_centres, cone_apex_z, cone_diameter,
                 base_polygon, base_top, table_extent=0.30,
                 edge_margin=0.001, name="printed artifact"):
        #: EVERY HEIGHT IS ABSOLUTE, measured from the table the artifact
        #: stands on, because that is how they are measured in practice -- a
        #: height gauge or a Keyence pass reads the table and the feature, not
        #: the feature and some intermediate datum.
        self.pads = np.asarray(pad_centres, float).reshape(-1, 2)
        self.pad_h = np.asarray(pad_tops, float).reshape(-1)
        self.pad_r = float(pad_diameter)/2.0
        self.cones = np.asarray(cone_centres, float).reshape(-1, 2)
        self.cone_z = float(cone_apex_z)
        self.cone_r = float(cone_diameter)/2.0
        self.base = np.asarray(base_polygon, float).reshape(-1, 2)
        self.base_h = float(base_top)
        #: Half-width of the surrounding table that is still believed, metres.
        self.table_extent = float(table_extent)
        self.edge_margin = float(edge_margin)
        self.name = name
        if self.pads.shape[0] != self.pad_h.shape[0]:
            raise ValueError("%d pad centres against %d heights"
                             % (self.pads.shape[0], self.pad_h.shape[0]))

    # -- geometry ---------------------------------------------------------
    def _inside_polygon_margin(self, P):
        """Signed distance to the base outline, positive inside. Convex only."""
        n = self.base.shape[0]
        d = np.full(P.shape[0], np.inf)
        for i in range(n):
            a, b = self.base[i], self.base[(i + 1) % n]
            e = b - a
            L = math.hypot(e[0], e[1])
            if L < 1e-12:
                continue
            # Outward normal for a counter-clockwise polygon.
            nx, ny = e[1]/L, -e[0]/L
            d = np.minimum(d, -((P[:, 0] - a[0])*nx + (P[:, 1] - a[1])*ny))
        return d

    def height(self, P):
        """(height at each (x, y), feature index, margin to the nearest edge)."""
        P = np.asarray(P, float).reshape(-1, 3)
        h = np.zeros(P.shape[0])
        which = np.full(P.shape[0], -1, int)
        # margin: how far this point is from the boundary of whatever feature
        # it is on. Negative nowhere -- it is a distance to the nearest edge.
        base_m = self._inside_polygon_margin(P)
        on_base = base_m > 0.0
        h = np.where(on_base, self.base_h, 0.0)
        which = np.where(on_base, 0, -1)
        margin = np.abs(base_m)
        for k in range(self.pads.shape[0]):
            r = np.hypot(P[:, 0] - self.pads[k, 0], P[:, 1] - self.pads[k, 1])
            take = r < self.pad_r
            h = np.where(take, self.pad_h[k], h)
            which = np.where(take, 1 + k, which)
            margin = np.minimum(margin, np.abs(r - self.pad_r))
        n_pad = self.pads.shape[0]
        rise = self.cone_z - self.base_h
        for k in range(self.cones.shape[0]):
            r = np.hypot(P[:, 0] - self.cones[k, 0], P[:, 1] - self.cones[k, 1])
            take = r < self.cone_r
            h = np.where(take, self.base_h + rise*(1.0 - r/self.cone_r), h)
            which = np.where(take, 1 + n_pad + k, which)
            # The cone's APEX is a feature, not an edge, so only its base rim
            # is a boundary. Its flank is measured and kept.
            margin = np.minimum(margin, np.abs(r - self.cone_r))
        return h, which, margin

    def signed_distance(self, P, ball_margin=None):
        """(distance above the artifact surface, feature index) per point.

        `ball_margin` is accepted and ignored -- solve_surface_fit() passes it
        for PlateArtifact and this artifact has no balls. Feature -1 is the
        table, 0 the base top, 1..9 the pads, 10.. the cones.
        """
        P = np.asarray(P, float).reshape(-1, 3)
        h, which, margin = self.height(P)
        d = P[:, 2] - h
        off = (np.abs(P[:, 0]) > self.table_extent) | \
              (np.abs(P[:, 1]) > self.table_extent)
        bad = off | (margin < self.edge_margin)
        return np.where(bad, np.nan, d), which

    def feature_name(self, w):
        if w < 0:
            return "table"
        if w == 0:
            return "base top"
        if w <= self.pads.shape[0]:
            return "pad %d" % (w - 1)
        return "cone %d" % (w - 1 - self.pads.shape[0])

    def heights(self):
        """Every distinct surface height the artifact offers, metres, sorted.

        This is the whole reason it exists -- calibrate_keyence() prints it so
        the operator can see the range spread a single pass will produce, and
        compare it against the +/-23 mm the head can measure over.
        """
        z = [0.0, self.base_h, self.cone_z] + list(self.pad_h)
        return sorted(set(round(float(v), 6) for v in z))

    def describe(self):
        return ("%s: %d pads %.1f..%.1f mm, %d cones to %.1f mm, base %.1f mm, "
                "on a table believed to +/-%.0f mm"
                % (self.name, self.pads.shape[0],
                   self.pad_h.min()*1e3, self.pad_h.max()*1e3,
                   self.cones.shape[0], self.cone_z*1e3,
                   self.base_h*1e3, self.table_extent*1e3))


def radial_clearance_profile(artifact, datum_height=None, n=257):
    """(radii, max height above the datum) -- a yaw-free envelope of a shape.

    WHY RADIAL. Clearance-checking a calibration pose needs to know how tall
    the artifact is under a given bit of tooling, and at pose-generation time
    the artifact's YAW is not known: all the operator taught is one point on
    top of it. A height field cannot be evaluated without the yaw; the envelope
    over all yaws can, and it is exactly the maximum height on each circle
    about the artifact's centre.

    THAT IS WORTH DOING RATHER THAN CALLING THE WHOLE FOOTPRINT AS TALL AS ITS
    TALLEST FEATURE. On this artifact the tall pads live on a 45 mm grid and
    reach r = 75 mm; outside that the shape is a 4 mm base plate. The flat-box
    model therefore refuses every near-standoff pose over the plate edge --
    which is most of the d-15 set, and the standoff spread is the only thing
    that measures rotation of the head about its own laser line. The radial
    envelope keeps those poses and still refuses a tool that is genuinely over
    a pad.

    Heights come back RELATIVE TO THE DATUM -- the top of the centre pad, which
    is the point the operator touches off -- because that is the one height the
    caller knows in world coordinates. `datum_height` overrides which feature
    that is; by default it is the pad nearest the centre of the pad grid.

    Returns (r, h) with r ascending from 0. h is a step function sampled at r;
    read it with np.interp and it is conservative between samples only to the
    extent the sampling is fine, so the sampling is fine.
    """
    pads = np.asarray(getattr(artifact, "pads", np.zeros((0, 2))), float).reshape(-1, 2)
    pad_h = np.asarray(getattr(artifact, "pad_h", np.zeros(0)), float).reshape(-1)
    cones = np.asarray(getattr(artifact, "cones", np.zeros((0, 2))), float).reshape(-1, 2)
    base = np.asarray(getattr(artifact, "base", np.zeros((0, 2))), float).reshape(-1, 2)
    pad_r = float(getattr(artifact, "pad_r", 0.0))
    cone_r = float(getattr(artifact, "cone_r", 0.0))
    cone_z = float(getattr(artifact, "cone_z", 0.0))
    base_h = float(getattr(artifact, "base_h", 0.0))

    if pads.shape[0]:
        centre = pads.mean(axis=0)
    elif base.shape[0]:
        centre = base.mean(axis=0)
    else:
        raise ValueError("this artifact has neither pads nor a base outline, "
                         "so it has no centre to take radii about")

    if datum_height is None:
        if not pads.shape[0]:
            raise ValueError("no pads, so there is no centre pad to be the datum")
        k = int(np.argmin(np.linalg.norm(pads - centre, axis=1)))
        datum_height = float(pad_h[k])

    # How far out anything reaches: the base outline's farthest corner, or the
    # outermost feature, whichever is greater.
    reach = 0.0
    if base.shape[0]:
        reach = float(np.linalg.norm(base - centre, axis=1).max())
    for C, rad in ((pads, pad_r), (cones, cone_r)):
        if C.shape[0]:
            reach = max(reach, float(np.linalg.norm(C - centre, axis=1).max()) + rad)
    r = np.linspace(0.0, reach*1.05 + 1e-6, int(n))
    h = np.full(r.shape, 0.0)           # the table the artifact stands on

    # The base plate covers every radius inside its inscribed circle for
    # certain, and MAY cover radii out to its circumradius depending on yaw --
    # so the envelope carries base_h all the way to the circumradius.
    if base.shape[0]:
        h = np.where(r <= float(np.linalg.norm(base - centre, axis=1).max()),
                     base_h, h)
    # A disc feature at radius r_c with radius rad covers [r_c - rad, r_c + rad].
    for k in range(pads.shape[0]):
        rc = float(np.linalg.norm(pads[k] - centre))
        h = np.where((r >= rc - pad_r) & (r <= rc + pad_r),
                     np.maximum(h, float(pad_h[k])), h)
    for k in range(cones.shape[0]):
        rc = float(np.linalg.norm(cones[k] - centre))
        # The cone's apex is its tallest point and sits at its own centre, so
        # the envelope over a circle gets the cone's height at the radial
        # offset |r - rc| in from its rim -- taken at the apex value for the
        # whole disc, which is the safe direction and costs 9 mm of nothing.
        h = np.where((r >= rc - cone_r) & (r <= rc + cone_r),
                     np.maximum(h, cone_z), h)
    # DILATE BY ONE SAMPLE EACH WAY so that reading this back with np.interp
    # can never under-report. Between two samples interp draws a straight line,
    # and across a step UP that line sits below the true envelope -- which is
    # the unsafe direction. Widening every step by one sample interval (0.5 mm
    # at the default n) makes the profile an upper bound everywhere at a cost
    # of half a millimetre of lateral conservatism.
    if h.size > 2:
        h = np.maximum(h, np.maximum(np.r_[h[1:], h[-1]], np.r_[h[0], h[:-1]]))
    return r, h - float(datum_height)


class SurfaceFitResult(object):
    """A profiler transform solved by making its points land on known geometry."""

    def __init__(self, X, W, per_pose, sigma, null_space, name="keyence",
                 n_points=0, n_iter=0, rms_mm=0.0, feature_counts=None):
        self.X = np.asarray(X, float)
        self.W = np.asarray(W, float)
        self.per_pose = list(per_pose)
        self.sigma = sigma
        #: [(singular value, the 6-vector of X it belongs to)] for directions
        #: the data barely constrain. Empty is what a good artifact gives.
        self.null_space = list(null_space)
        self.name = name
        self.n_points = n_points
        self.n_iter = n_iter
        self.rms_mm = rms_mm
        self.feature_counts = dict(feature_counts or {})

    xyz = HandEyeResult.xyz
    quat = HandEyeResult.quat
    rpy = HandEyeResult.rpy

    @property
    def adequate(self):
        return not self.null_space

    @property
    def worst(self):
        return max(self.per_pose, key=lambda p: p["rms_mm"]) if self.per_pose else None

    def report(self):
        L = []
        L.append("%s calibration -- probe_end -> %s, by fitting %d points to "
                 "known geometry" % (self.name, self.name, self.n_points))
        L.append("  xyz  %+.5f %+.5f %+.5f  m" % self.xyz)
        L.append("  rpy  %+.5f %+.5f %+.5f  rad  (%+.2f %+.2f %+.2f deg)"
                 % (self.rpy + tuple(math.degrees(v) for v in self.rpy)))
        L.append("")
        if self.feature_counts:
            L.append("  POINTS BY FEATURE: " + ", ".join(
                "%s %d" % (k, v) for k, v in sorted(self.feature_counts.items())))
        L.append("  RESIDUALS (each point's distance to the surface it landed on)")
        L.append("    rms %.3f mm over %d points, %d iterations"
                 % (self.rms_mm, self.n_points, self.n_iter))
        for p in self.per_pose:
            mark = "  <-- worst" if p is self.worst and len(self.per_pose) > 1 else ""
            L.append("    pose %-3s %7.3f mm rms, %5d points%s"
                     % (p.get("pose"), p["rms_mm"], p["n"], mark))
        if self.sigma is not None:
            L.append("")
            L.append("  UNCERTAINTY (1 sigma, in probe_end's frame)")
            L.append("    translation  %.3f  %.3f  %.3f  mm" % tuple(self.sigma[:3]))
            L.append("    rotation     %.3f  %.3f  %.3f  mrad" % tuple(self.sigma[3:]))
        if self.null_space:
            L.append("")
            L.append("  *** THIS ARTIFACT DOES NOT DETERMINE THE TRANSFORM. %d "
                     "direction(s) of it are free:" % len(self.null_space))
            for s, v in self.null_space:
                L.append("      sv %.2e : translation (%+.2f %+.2f %+.2f) "
                         "rotation (%+.2f %+.2f %+.2f)"
                         % ((s,) + tuple(v[:3]) + tuple(v[3:])))
            L.append("    A component near (0 0 0 | 1 0 0) is rotation about "
                     "the laser line, and means every profile was taken at the "
                     "same standoff: repeat some poses 15 mm nearer and 15 mm "
                     "further. Otherwise put more of the artifact's shape -- "
                     "the balls, an edge -- under the line.")
        return "\n".join(L)

    def __repr__(self):
        return ("<SurfaceFitResult %s xyz=(%.4f, %.4f, %.4f) rms=%.3f mm %s>"
                % ((self.name,) + self.xyz + (self.rms_mm,
                   "ok" if self.adequate else "UNDER-CONSTRAINED")))


#: A direction of the 6-DOF transform whose singular value falls this far below
#: the largest is not measured by the data, it is inherited from the guess.
NULL_SPACE_RATIO = 1e-3

#: How many times solve_surface_fit() may halve a Gauss-Newton step looking for
#: one that improves the fit. 12 halvings is a factor of 4096, far more than a
#: well-scaled problem needs; it costs residual evaluations only when the full
#: step was going to overshoot, which is exactly when it is worth paying for.
LINE_SEARCH_HALVINGS = 12

#: The fraction of the currently-landing points a candidate step must keep on
#: the artifact to be considered at all. Without this a step can "improve" the
#: rms by throwing the points that disagree off the surface entirely, which is
#: how an unconstrained fit walks away while its residual gets better.
LINE_SEARCH_MIN_KEEP = 0.5


def solve_surface_fit(G, P, artifact, X0, W0, max_iter=30, ball_margin=None,
                      trim=3.0, require_adequate=False, name="keyence",
                      progress=None,
                      free=None, fix_W=False):
    """Solve probe_end -> sensor by landing the sensor's points on known geometry.

    G[i] is probe_end in the base frame at pose i; P[i] is the points that pose
    measured, in the SENSOR's own frame, metres. `artifact` is a PlateArtifact,
    X0 the starting guess for probe_end -> sensor (the URDF's value), W0 the
    starting guess for the artifact's pose in the base frame (from a touch-off,
    or from the Azure, or from where it was put).

    Minimises, over X and W together, the distance from every point to the
    artifact surface it lands on:

        r_ik = signed_distance( W^-1 G_i X p_ik )

    which says only "these points are on this object" -- no correspondences, no
    features, no detection. That is why it works for a sensor that cannot see a
    pattern.

    `trim` re-assigns and discards points more than `trim` times the rms out
    after each iteration: a profile crossing the edge of a plate or the rim of
    a ball seat is real data about something that is not the artifact.

    Returns a SurfaceFitResult. `.adequate` is False, and `.null_space` says
    which directions, when the artifact in view cannot pin all six degrees of
    freedom -- which is the normal outcome over a bare flat and is the reason
    the balls exist.
    """
    G = [np.asarray(g, float) for g in G]
    P = [np.asarray(p, float).reshape(-1, 3) for p in P]
    if len(G) != len(P):
        raise ValueError("%d poses against %d point sets" % (len(G), len(P)))
    X, W = np.asarray(X0, float).copy(), np.asarray(W0, float).copy()

    # WHICH OF THE TWELVE PARAMETERS MAY MOVE. All of them by default, which is
    # the honest thing when nothing else is known -- but the artifact's height
    # and the sensor's standoff are very nearly the same parameter, and over a
    # pose set that does not separate them the solver trades one against the
    # other and reports a small residual for both. On real cell data that looks
    # like this: the fit moves the sensor DOWN 3.80 mm and the artifact DOWN
    # 3.46 mm, which cancel, every measured point lands within 0.2 mm of where
    # the untouched calibration already put it, and the residual is 0.179 mm
    # either way -- so nothing in the fit can tell them apart.
    #
    # `fix_W` holds the artifact where it was put -- which is what a touch-off
    # on its datum gives -- and makes the sensor's standoff answerable at all.
    # `free` is a 12-long mask for finer control.
    mask = np.ones(12, bool) if free is None else np.asarray(free, bool).copy()
    if fix_W:
        mask[6:] = False
    if not mask.any():
        raise ValueError("every parameter is fixed; there is nothing to solve")

    def world_pts(X_, i):
        p = P[i]
        R, t = T_to_rt(G[i].dot(X_))
        return p.dot(R.T) + t

    # What to call each feature in the report. PlateArtifact numbers balls;
    # SteppedArtifact numbers pads and cones and names them itself.
    namer = getattr(artifact, "feature_name", None)
    if namer is None:
        namer = lambda w: "plate" if w < 0 else "ball %d" % w   # noqa: E731

    def residuals(X_, W_, keep=None):
        Wi = T_inv(W_)
        out, counts = [], {}
        for i in range(len(P)):
            q = world_pts(X_, i)
            q = q.dot(Wi[:3, :3].T) + Wi[:3, 3]
            d, which = artifact.signed_distance(q, ball_margin=ball_margin)
            if keep is not None:
                d = np.where(keep[i], d, np.nan)
            out.append(d)
            for w in np.unique(which[np.isfinite(d)]):
                k = namer(w)
                counts[k] = counts.get(k, 0) + int(np.sum(
                    (which == w) & np.isfinite(d)))
        return out, counts

    keep = [np.ones(p.shape[0], bool) for p in P]
    J = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        r_list, counts = residuals(X, W, keep)
        r = np.concatenate(r_list)
        good = np.isfinite(r)
        if good.sum() < 20:
            raise ValueError("only %d of %d points land on the artifact. Either "
                             "the starting guess is far out or the sensor was "
                             "not looking at it." % (int(good.sum()), r.size))
        r = r[good]
        J = np.zeros((r.size, 12))
        eps = 1e-7
        for k in range(12):
            d = np.zeros(12)
            d[k] = eps
            Xp = X.dot(se3_exp(d[:6])) if k < 6 else X
            Wp = W if k < 6 else W.dot(se3_exp(d[6:]))
            rp = np.concatenate(residuals(Xp, Wp, keep)[0])[good]
            # A POINT MAY LEAVE THE ARTIFACT UNDER THE PERTURBATION, which
            # makes rp NaN where r was finite -- a point sitting exactly on the
            # rim of a pad, the edge of the plate, or the margin round a ball.
            # Its derivative is genuinely undefined, so it contributes none.
            # Left as NaN it poisons the whole column, and lstsq then fails
            # outright with "SVD did not converge", which reads as a numerical
            # accident rather than as a point on an edge.
            J[:, k] = np.where(np.isfinite(rp), (rp - r)/eps, 0.0)
        step_m, _res, _rank, _sv = np.linalg.lstsq(J[:, mask], -r, rcond=None)
        step = np.zeros(12)
        step[mask] = step_m
        # ACCEPT THE STEP ONLY IF IT IMPROVES THE FIT, and shorten it until it
        # does. An unconditional Gauss-Newton step is fine near the solution
        # and not fine a few mm away from it: the surface model is PIECEWISE --
        # a point that leaves a pad, a plate edge or a ball margin has NO
        # residual rather than a large one -- so a full step can overshoot
        # until every point leaves the artifact, and the next residuals() call
        # raises
        #
        #     no point lands on the artifact any more. The fit has walked away
        #     from it, which means the starting guess was too far out ...
        #
        # Without the line search, whether that happens depends on the LAPACK
        # build to the last bit: the same inputs converge on one numpy and
        # diverge on another, and the host, the container and the cell all run
        # different versions. surface_fit_recovery() pushes the answer off by
        # design, so it drives the fit into this deliberately and then scores a
        # determined direction 0.0.
        #
        # A COLLAPSING POINT COUNT IS NOT AN IMPROVEMENT either, even when the
        # rms of what is left looks better: shedding the points that disagree is
        # how a fit walks away while its residual improves. A candidate keeping
        # fewer than LINE_SEARCH_MIN_KEEP of the points the current iterate had
        # is rejected however good its rms.
        def _cost(X_, W_, n_ref):
            d = np.concatenate(residuals(X_, W_, keep)[0])
            d = d[np.isfinite(d)]
            if d.size < 20 or d.size < LINE_SEARCH_MIN_KEEP*n_ref:
                return float("inf")
            return float(np.sqrt(np.mean(d**2)))

        n_ref = int(good.sum())
        c0 = _cost(X, W, n_ref)
        t, taken = 1.0, False
        for _ in range(LINE_SEARCH_HALVINGS):
            Xn = X.dot(se3_exp(t*step[:6]))
            Wn = W.dot(se3_exp(t*step[6:]))
            if _cost(Xn, Wn, n_ref) <= c0:
                X, W, taken = Xn, Wn, True
                break
            t *= 0.5
        if not taken:
            # No fraction of this step improves anything: this IS the minimum
            # for the current trim set. Stop on the last good iterate rather
            # than stepping somewhere worse.
            break
        step = t*step
        if trim:
            r2_list, _c = residuals(X, W)
            flat_r = np.concatenate(r2_list)
            flat_r = flat_r[np.isfinite(flat_r)]
            if flat_r.size == 0:
                raise ValueError(
                    "no point lands on the artifact any more. The fit has "
                    "walked away from it, which means the starting guess was "
                    "too far out or the points are not of this artifact.")
            rms = float(np.sqrt(np.mean(flat_r**2)))
            keep = [np.isfinite(d) & (np.abs(d) < trim*rms) for d in r2_list]
        # SAY SOMETHING EVERY ITERATION. On a full pose set this loop can run
        # for over ten minutes; silent, it looks hung and gets interrupted with
        # every pose already in hand. A solve that takes minutes has to show it
        # is alive and getting better.
        if progress is not None:
            # REPORT THE NUMBER THE RESULT WILL REPORT. The `rms` computed above
            # is over every point that landed, INCLUDING the ones trim is about
            # to throw out -- it has to be, because it is what sets the trim
            # threshold. Reporting it as progress means watching a solve count
            # down to 1.5 mm and then announcing 0.066 mm, which reads as a bug.
            # The kept set is already in hand, so this costs no new evaluation.
            if trim:
                kept_r = np.concatenate([d[k] for d, k in zip(r2_list, keep)])
                kept_r = kept_r[np.isfinite(kept_r)]
            else:
                kept_r = np.concatenate([d[np.isfinite(d)]
                                         for d in residuals(X, W)[0]])
            now = float(np.sqrt(np.mean(kept_r**2))) if kept_r.size else float("nan")
            progress({"iteration": n_iter, "max_iter": max_iter,
                      "rms_mm": now*1e3, "step": float(np.linalg.norm(step)),
                      "fraction": float(t), "points": int(kept_r.size)})
        if np.linalg.norm(step) < 1e-10:
            break

    r_list, counts = residuals(X, W, keep)
    per_pose = []
    for i, d in enumerate(r_list):
        g = np.isfinite(d)
        per_pose.append({"pose": i, "n": int(g.sum()),
                         "rms_mm": (float(np.sqrt(np.mean(d[g]**2)))*1e3
                                    if g.any() else float("nan"))})
    allr = np.concatenate(r_list)
    allr = allr[np.isfinite(allr)]
    rms_mm = float(np.sqrt(np.mean(allr**2)))*1e3

    # Conditioning and covariance from the same Jacobian the fit used. A
    # direction of X the points do not move under is a direction the fit
    # cannot see, and it shows up here as a tiny singular value -- WITH a
    # perfect residual, which is the whole point.
    null_space, sigma = [], None
    if J is not None:
        # THE SVD IS OF ALL TWELVE COLUMNS, not of X's six. A direction of X
        # that the points do not see may still be visible in J[:, :6] because
        # moving the ARTIFACT undoes it, and taking the sensor block alone
        # would then call a free direction determined. What is free is a null
        # direction of the joint problem; whether it matters is whether that
        # direction has an X component.
        #
        # Three of these are expected and harmless: the artifact is a plane
        # plus balls on it, and sliding the artifact along its own surface or
        # spinning a bare plate about its normal changes no measurement. Those
        # nulls live entirely in the W block and are reported as gauge, not as
        # a problem.
        Jm = J[:, mask]
        _U, sv, Vt = np.linalg.svd(Jm, full_matrices=False)
        if not mask.all():
            wide = np.zeros((Vt.shape[0], 12))
            wide[:, mask] = Vt
            Vt = wide
        if sv[0] > 0.0:
            for k in range(Vt.shape[0]):
                if sv[k]/sv[0] >= NULL_SPACE_RATIO:
                    continue
                v = Vt[k]
                if np.linalg.norm(v[:6]) < 0.2:
                    continue            # gauge: the artifact's own symmetry
                null_space.append((float(sv[k]),
                                   [float(x) for x in v[:6]/max(
                                       np.linalg.norm(v[:6]), 1e-12)]))
        dof = max(allr.size - int(mask.sum()), 1)
        s2 = float(np.sum(allr**2)/dof)
        try:
            sub = np.linalg.pinv(Jm.T.dot(Jm), rcond=1e-14)
            full = np.zeros((12, 12))
            idx = np.where(mask)[0]
            full[np.ix_(idx, idx)] = sub
            cov = s2*full[:6, :6]
            s = np.sqrt(np.maximum(np.diag(cov), 0.0))
            sigma = np.array([s[0]*1e3, s[1]*1e3, s[2]*1e3,
                              s[3]*1e3, s[4]*1e3, s[5]*1e3])
        except np.linalg.LinAlgError:
            sigma = np.array([float("inf")]*6)

    res = SurfaceFitResult(X, W, per_pose, sigma, null_space, name=name,
                           n_points=int(allr.size), n_iter=n_iter,
                           rms_mm=rms_mm, feature_counts=counts)
    if require_adequate and not res.adequate:
        raise PoseSetInadequate(res.report())
    return res


# ------------------------------------------------------------------------------
# Turning a solved transform into the file a robot description consumes
# ------------------------------------------------------------------------------
# WHAT IS EMITTED, AND WHY IT IS A CORRECTION RATHER THAN AN ABSOLUTE ORIGIN.
# The solve gives X = probe_end -> the sensor's optical frame. The description
# does not carry that transform anywhere; it carries a chain of fixed joints,
# most of which are bracket geometry nobody measured and nobody should
# overwrite. So the description gains ONE new joint per sensor, identity by
# default, and the calibration sets that joint. It is the same shape as
# keyence_ref_dist_cal -- a spec the shared description owns and a correction
# the cell owns -- widened from one number to six.
#
# THE CHAIN, and the side the correction goes on. In the deployed description:
#
#     parent_link --Cal--> cal_link --(nominal bracket geometry)--> sensor_frame
#
# so, writing A for parent_link -> probe_end and B for cal_link -> sensor_frame
# (both read from LIVE TF, which is the only place the chain in force is
# actually known),
#
#     probe_end -> sensor  =  A^-1 . Cal . B     which has to equal X
#     =>  Cal = A . X . B^-1
#
# THE CORRECTION IS THEREFORE IN THE PARENT LINK'S FRAME -- "the whole sensor
# assembly is displaced by this much, in mounting_plate coordinates". Putting
# it on the other side instead, Cal' = B^-1 . A . X expressed in the sensor's
# own frame, is a DIFFERENT transform whenever the bracket is rotated, and the
# ex-ur10e-1 Azure bracket is rotated by two right angles. Both sides fit the
# same story and one of them is wrong by about twice the offset, which is why
# tari_ur_description/test/test_sensor_mount_cal.py renders the description
# with a known correction and checks the composed transform rather than the
# number in the file.
#
# WHY B IS READ FROM cal_link AND NOT FROM THE MOUNT. cal_link sits ABOVE the
# correction, so B is the nominal bracket geometry no matter what correction is
# already deployed, and re-running the calibration produces the new total
# rather than compounding onto the old one. Reading it from the parent instead
# would make every calibration relative to the last, which is the classic way
# a correction gets applied twice.

#: sensor -> the (xyz, rpy) xacro properties its correction is written to.
CAL_XACRO_PROPERTIES = {
    "azure": ("azure_mount_cal_xyz", "azure_mount_cal_rpy"),
    "keyence": ("keyence_mount_cal_xyz", "keyence_mount_cal_rpy"),
}

#: The Keyence standoff correction the 6 DOF mount calibration SUPERSEDES.
#: See the ownership note in render_calibration_xacro().
CAL_REF_DIST_PROPERTY = "keyence_ref_dist_cal"


#: How far to push each degree of freedom when asking whether the data can
#: pull it back: 3 mm and 10 mrad, which are several times the answer's
#: precision and well inside the basin the solver converges from.
SURFACE_FIT_PROBE = (0.003, 0.010)

#: Below this fraction of a deliberate error pulled back, the direction is not
#: measured by the data and whatever the fit returned for it came from the
#: starting guess.
SURFACE_FIT_MIN_RECOVERY = 0.5


def surface_fit_recovery(G, P, artifact, X, W, probe=None, fix_W=False,
                         max_iter=12, **kw):
    """How much of a deliberate error in X this data actually pulls back.

    THE CHECK THE NULL-SPACE TEST CANNOT MAKE. `solve_surface_fit` reports a
    direction as free when its singular value collapses, which catches an exact
    degeneracy. The dangerous case is not exact: when the sensor's standoff and
    the artifact's height are separated only by the tilt in the pose set, the
    singular value stays well above the threshold and the solve comes back
    "adequate" with a 0.179 mm residual over 494 017 points and a formal
    uncertainty of 7 MICRONS -- having moved the sensor 3.80 mm and the
    artifact 3.46 mm the other way, which cancel.

    The formal uncertainty cannot be trusted here either. It divides by the
    point count, and those points are 60 copies of one profile line at each of
    14 static poses: perhaps 14 independent observations wearing the error bars
    of half a million.

    So ask the question directly instead of modelling it. Push X off the answer
    by `probe` in each degree of freedom, re-solve, and see how much comes
    back. A direction the data determines returns ~1.0. A direction it does not
    returns ~0.0, however small the residual was.

    Returns {"names": [...], "recovered": [...], "worst": (name, fraction),
    "ok": bool} with translation first, then rotation.
    """
    dt, dr = SURFACE_FIT_PROBE if probe is None else probe
    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    step = [dt, dt, dt, dr, dr, dr]
    X = np.asarray(X, float)
    out = []
    for k in range(6):
        d = np.zeros(6)
        d[k] = step[k]
        X0 = X.dot(se3_exp(d))
        try:
            r = solve_surface_fit(G, P, artifact, X0, np.asarray(W, float),
                                  max_iter=max_iter, fix_W=fix_W, **kw)
        except Exception:                                     # noqa: BLE001
            out.append(0.0)                 # it could not even converge back
            continue
        # How much of the pushed error survived, measured in the same units the
        # push was made in: the residual twist between the two answers.
        left = se3_log(T_inv(X).dot(r.X))
        out.append(float(max(0.0, 1.0 - abs(left[k])/step[k])))
    worst = int(np.argmin(out))
    return {"names": names, "recovered": out,
            "worst": (names[worst], float(out[worst])),
            "ok": bool(min(out) >= SURFACE_FIT_MIN_RECOVERY),
            "probe_mm": dt*1e3, "probe_mrad": dr*1e3}


def describe_surface_fit_recovery(rec):
    """The recovery check as text, for the operator standing at the cell."""
    L = ["RECOVERY (push the answer off by %.0f mm / %.0f mrad and re-solve; "
         "1.00 means the data pulls it all back)" % (rec["probe_mm"],
                                                     rec["probe_mrad"])]
    for name, v in zip(rec["names"], rec["recovered"]):
        flag = "" if v >= SURFACE_FIT_MIN_RECOVERY else "   <-- NOT MEASURED"
        L.append("    %-6s %5.2f%s" % (name, v, flag))
    if not rec["ok"]:
        L.append("    %r came back %.2f. That direction is not in this data, "
                 "and the number the fit returned for it is the number it "
                 "started from. A small residual does not contradict this --"
                 % (rec["worst"][0], rec["worst"][1]))
        L.append("    a 0.179 mm residual has gone with a standoff wrong by "
                 "3.8 mm.")
    return L


def mount_cal(X, T_parent_probe, T_cal_sensor):
    """Cal = A . X . B^-1 -- the correction joint that makes the chain give X.

    X                probe_end -> the sensor's frame, as solved.
    T_parent_probe   A, the correction joint's parent link -> probe_end.
    T_cal_sensor     B, the correction joint's CHILD link -> the sensor's
                     frame, i.e. the nominal bracket geometry below the
                     correction, with any superseded correction already
                     removed from it.
    """
    return (np.asarray(T_parent_probe, float)
            .dot(np.asarray(X, float))
            .dot(T_inv(np.asarray(T_cal_sensor, float))))


def sensor_transform(Cal, T_parent_probe, T_cal_sensor):
    """The inverse of mount_cal -- what probe_end -> sensor a Cal produces.

    This is what the deployed description will actually publish, so it is what
    a read-back check compares against.
    """
    return (T_inv(np.asarray(T_parent_probe, float))
            .dot(np.asarray(Cal, float))
            .dot(np.asarray(T_cal_sensor, float)))


def cal_to_properties(Cal, fmt="%.8f"):
    """(xyz, rpy) as the two strings a xacro `origin` wants."""
    Cal = np.asarray(Cal, float)
    xyz = Cal[:3, 3]
    rpy = R_to_rpy(Cal[:3, :3])
    return (" ".join(fmt % v for v in xyz), " ".join(fmt % v for v in rpy))


def properties_to_cal(xyz, rpy):
    """The 4x4 back from the two strings. The other half of the round trip."""
    t = [float(v) for v in str(xyz).split()]
    r = [float(v) for v in str(rpy).split()]
    if len(t) != 3 or len(r) != 3:
        raise ValueError("a xacro origin needs three numbers each, got %r / %r"
                         % (xyz, rpy))
    return rt_to_T(rpy_to_R(r[0], r[1], r[2]), t)


#: xacro property assignments, as the generated file writes them and as
#: parse_calibration_xacro() reads them back.
_PROP_RE = None


def parse_calibration_xacro(text):
    """{property name -> value string} from a generated calibration xacro.

    Deliberately a regex over the text rather than an XML parse -- this has to
    read a file a human may have edited by hand, from a dgpy prompt, with no
    ROS and no lxml, and it must not care about attribute order or comments.
    """
    import re
    global _PROP_RE
    if _PROP_RE is None:
        _PROP_RE = re.compile(
            r"<xacro:property\s+name=\"([A-Za-z0-9_]+)\"\s+value=\"([^\"]*)\"")
    return dict(_PROP_RE.findall(text))


def render_calibration_xacro(cals, ref_dist_cal=0.0, notes=(), header=()):
    """The per-cell drop-in file, as text. `cals` is {sensor -> 4x4 or None}.

    WHO OWNS THE KEYENCE STANDOFF. keyence_ref_dist_cal moves the optical frame
    along the head's own axis and a 6 DOF mount correction moves the head,
    including along that axis. They are the same physical quantity reached two
    ways, and applying both counts it twice -- which is invisible, because each
    is individually plausible and their sum is a clean constant offset.

    THE RULE IS THAT THE 6 DOF SOLVE SUPERSEDES. calibrate_keyence() measures
    all six degrees of freedom of where the head is; keyence_ref_dist_cal
    measures one of them against a touch-off. So this file ALWAYS writes
    keyence_ref_dist_cal, and writes 0.0 unless a caller insists otherwise,
    and calibration.dpi folds whatever value was in force into the mount
    correction before emitting it. lj_v7080_macro.xacro refuses to render if
    both are non-trivial, so the rule cannot be broken by hand-editing either.

    EVERY LINE IS CHECKED FOR ': ' AND ' #'. gazebo_ros2_control parses the
    whole robot_description as YAML, comments included, and either sequence
    anywhere in it means no controller manager and a cell that comes up with
    no way to move the arm. A generated file is exactly where that would creep
    in unnoticed.
    """
    L = ['<?xml version="1.0"?>',
         "<!--",
         "  MEASURED, NOT DESIGNED. Generated by calibrate_azure() and",
         "  calibrate_keyence() in ROS-Scanning. Do not hand-edit -- re-run the",
         "  calibration and copy this file again. The procedure is",
         "  ROS-Scanning/docs/calibration-procedure.md.",
         ""]
    L += ["  " + str(h) for h in header]
    L += ["", "  Each correction is a fixed joint inserted ABOVE the nominal",
          "  bracket geometry, so it is expressed in the PARENT link's frame",
          "  (mounting_plate for the Azure, keyence_metal for the Keyence).",
          "  Identity is 0 0 0 / 0 0 0 and means uncalibrated.", "-->",
          '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">']
    for sensor in sorted(CAL_XACRO_PROPERTIES):
        p_xyz, p_rpy = CAL_XACRO_PROPERTIES[sensor]
        C = cals.get(sensor)
        xyz, rpy = ("0 0 0", "0 0 0") if C is None else cal_to_properties(C)
        L.append("")
        L.append("  <!-- %s -->" % (sensor,))
        L.append('  <xacro:property name="%s" value="%s"/>' % (p_xyz, xyz))
        L.append('  <xacro:property name="%s" value="%s"/>' % (p_rpy, rpy))
    L.append("")
    L.append("  <!-- Superseded by keyence_mount_cal above. Keep it zero. -->")
    L.append('  <xacro:property name="%s" value="%.8f"/>'
             % (CAL_REF_DIST_PROPERTY, float(ref_dist_cal)))
    for n in notes:
        L.append("  <!-- %s -->" % (n,))
    L.append("</robot>")
    text = "\n".join(L) + "\n"
    for bad in (": ", " #"):
        if bad in text:
            raise ValueError(
                "the generated calibration xacro contains %r, which makes "
                "gazebo_ros2_control read the robot description as broken YAML "
                "and start no controller manager. Rephrase the note that "
                "introduced it." % (bad,))
    return text
