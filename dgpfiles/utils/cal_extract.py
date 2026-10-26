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
# File:  cal_extract.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Extracts per-pose robot poses and profile points from a calibration bag, for an offline re-solve.
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
"""Pull (world->probe_end, profile points) per calibration pose out of a bag."""
import json, sys, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros, rclpy.time, rclpy.duration
from builtin_interfaces.msg import Time as TimeMsg

BAG = sys.argv[1]; SCEN = sys.argv[2]; OUT = sys.argv[3]
SETTLE_NS = int(1.0e9)          # CAL_SETTLE_S
N_PROFILES = 60                 # CAL_PROFILES_PER_POSE

scen = json.load(open(SCEN))
moves = [m for m in scen["moves"] if m.get("ok") and m.get("t1_ros_ns")]
print("%d flown moves" % len(moves))

buf = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=3600.0))
reader = SequentialReader()
reader.open(StorageOptions(uri=BAG, storage_id="mcap"), ConverterOptions("", ""))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
msgcls = {n: get_message(t) for n, t in types.items()}

windows = []
for m in moves:
    t0 = m["t1_ros_ns"] + SETTLE_NS
    windows.append({"label": m["label"], "t0": t0, "msgs": [], "done": False})

n_tf = 0
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == "/tf_static":
        msg = deserialize_message(data, msgcls[topic])
        for tr in msg.transforms:
            buf.set_transform_static(tr, "bag")
        n_tf += 1
    elif topic == "/tf":
        msg = deserialize_message(data, msgcls[topic])
        for tr in msg.transforms:
            buf.set_transform(tr, "bag")
        n_tf += 1
    elif topic == "/profiles":
        for w in windows:
            if w["done"] or t < w["t0"]:
                continue
            if len(w["msgs"]) < N_PROFILES:
                msg = deserialize_message(data, msgcls[topic])
                xyz = np.asarray(pc2.read_points_numpy(
                    msg, field_names=("x","y","z"), skip_nans=True)).reshape(-1,3)
                w["msgs"].append((t, xyz, msg.header.frame_id, msg.header.stamp))
                if len(w["msgs"]) == N_PROFILES:
                    w["done"] = True
            break
print("tf messages:", n_tf)

def T_of(stamp, target="world", source="probe_end"):
    tr = buf.lookup_transform(target, source, rclpy.time.Time.from_msg(stamp))
    q = tr.transform.rotation; p = tr.transform.translation
    x,y,z,w = q.x,q.y,q.z,q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                  [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=[p.x,p.y,p.z]
    return T

out = []
for w in windows:
    if not w["msgs"]:
        print("  %-26s NO PROFILES" % w["label"]); continue
    pts = np.vstack([x for _t,x,_f,_s in w["msgs"]])
    pts = pts[np.isfinite(pts).all(axis=1)]
    stamp = w["msgs"][0][3]
    try:
        G = T_of(stamp)
        Xk = T_of(stamp, "probe_end", "lj_v7080_optical_frame")
    except Exception as e:
        print("  %-26s TF FAILED: %s" % (w["label"], e)); continue
    out.append({"label": w["label"], "G": G.tolist(), "X_urdf": Xk.tolist(),
                "points": pts, "n_msgs": len(w["msgs"]),
                "frame": w["msgs"][0][2]})
    print("  %-26s %6d pts from %2d msgs, probe_end (%.4f %.4f %.4f)"
          % (w["label"], pts.shape[0], len(w["msgs"]), G[0,3], G[1,3], G[2,3]))

np.savez_compressed(OUT,
    labels=np.array([o["label"] for o in out]),
    G=np.array([o["G"] for o in out]),
    X_urdf=np.array([o["X_urdf"] for o in out]),
    counts=np.array([o["points"].shape[0] for o in out]),
    points=np.vstack([o["points"] for o in out]))
print("wrote", OUT, "with", len(out), "poses")
