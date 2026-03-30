#!/usr/bin/env python3
import math
import itertools
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PointStamped


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def build_signed_permutation_rotations():
    mats = []
    axes = [0, 1, 2]
    signs = [-1.0, 1.0]
    for perm in itertools.permutations(axes):
        P = np.zeros((3, 3), dtype=np.float64)
        for i, a in enumerate(perm):
            P[i, a] = 1.0
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    S = np.diag([sx, sy, sz])
                    M = S @ P
                    if np.linalg.det(M) > 0.5:
                        mats.append(M)
    unique = []
    for m in mats:
        if not any(np.allclose(m, u) for u in unique):
            unique.append(m)
    return unique


def axis_name_from_row(row):
    idx = int(np.argmax(np.abs(row)))
    sgn = "+" if row[idx] > 0 else "-"
    return f"{sgn}{['X','Y','Z'][idx]}"


class GraspToolAxisAuditRaw(Node):
    def __init__(self):
        super().__init__('grasp_tool_axis_audit_raw')

        self.pose_topic = '/grasp/best_pose_raw'
        self.contact_topic = '/grasp/best_contact_point'
        self.max_pair_age_sec = 5.0
        self.process_rate_hz = 5.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_pose = None
        self.latest_contact = None
        self.pose_count = 0
        self.contact_count = 0
        self.sample_count = 0

        self.variants = build_signed_permutation_rotations()

        self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, qos)
        self.create_subscription(PointStamped, self.contact_topic, self.contact_cb, qos)
        self.create_timer(1.0 / self.process_rate_hz, self.process_once)

        self.get_logger().info('========================================')
        self.get_logger().info('Grasp Tool Axis Audit RAW initialized')
        self.get_logger().info(f'pose_topic      : {self.pose_topic}')
        self.get_logger().info(f'contact_topic   : {self.contact_topic}')
        self.get_logger().info('Target heuristic: local contact ≈ [x<0, y≈0, z<0]')
        self.get_logger().info('========================================')

    def pose_cb(self, msg):
        self.latest_pose = msg
        self.pose_count += 1

    def contact_cb(self, msg):
        self.latest_contact = msg
        self.contact_count += 1

    def time_sec(self, stamp):
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def score_local_contact(self, v):
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        score = 0.0
        if x < 0.0:
            score += 2.0
        else:
            score -= 2.0 + 4.0 * abs(x)
        score -= 6.0 * abs(y)
        if z < 0.0:
            score += 1.0
        else:
            score -= 1.0 + 3.0 * abs(z)
        score -= 0.5 * abs(np.linalg.norm(v))
        return score

    def process_once(self):
        if self.latest_pose is None:
            self.get_logger().warn('waiting for pose message...')
            return
        if self.latest_contact is None:
            self.get_logger().warn('waiting for contact message...')
            return

        if self.latest_pose.header.frame_id != self.latest_contact.header.frame_id:
            self.get_logger().warn(
                f"frame mismatch: pose='{self.latest_pose.header.frame_id}' "
                f"contact='{self.latest_contact.header.frame_id}'"
            )
            return

        dt = abs(
            self.time_sec(self.latest_pose.header.stamp) -
            self.time_sec(self.latest_contact.header.stamp)
        )
        if dt > self.max_pair_age_sec:
            self.get_logger().warn(f'age mismatch too large: dt={dt:.3f}s')
            return

        p = self.latest_pose.pose.position
        q = self.latest_pose.pose.orientation
        c = self.latest_contact.point

        t = np.array([p.x, p.y, p.z], dtype=np.float64)
        R_pose = quat_to_rot(q.x, q.y, q.z, q.w)
        contact = np.array([c.x, c.y, c.z], dtype=np.float64)

        identity_local = R_pose.T @ (contact - t)
        identity_score = self.score_local_contact(identity_local)

        best_score = -1e18
        best_local = None
        best_M = None

        for M in self.variants:
            R_test = R_pose @ M
            local = R_test.T @ (contact - t)
            s = self.score_local_contact(local)
            if s > best_score:
                best_score = s
                best_local = local
                best_M = M

        self.sample_count += 1

        toolX = axis_name_from_row(best_M[:, 0])
        toolY = axis_name_from_row(best_M[:, 1])
        toolZ = axis_name_from_row(best_M[:, 2])

        self.get_logger().info(
            f"[sample {self.sample_count:03d}] "
            f"frame='{self.latest_pose.header.frame_id}' dt={dt:.3f}s | "
            f"IDENTITY local=({identity_local[0]:.4f}, {identity_local[1]:.4f}, {identity_local[2]:.4f}) "
            f"score={identity_score:.4f} | "
            f"BEST toolX={toolX} toolY={toolY} toolZ={toolZ} "
            f"local=({best_local[0]:.4f}, {best_local[1]:.4f}, {best_local[2]:.4f}) "
            f"score={best_score:.4f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GraspToolAxisAuditRaw()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()