#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
import tf2_ros


class CameraPointBaseTransformer(Node):
    def __init__(self):
        super().__init__('camera_point_base_transformer_right')

        # ------------------------------------------------------------------
        # Camera optical frame -> Right Gripper
        # ------------------------------------------------------------------
        self.T_cam_to_gripper = np.array([
            [ 0.9954,  0.    , -0.0958,  0.0982],
            [ 0.    , -1.    ,  0.    ,  0.    ],
            [-0.0958,  0.    , -0.9954, -0.0725],
            [ 0.    ,  0.    ,  0.    ,  1.    ]
        ], dtype=np.float64)

        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d_r')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_pose_r')

        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose_r')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose_r')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'arm_r_link7')
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('debug', True)

        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.debug = bool(self.get_parameter('debug').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_aruco = self.create_subscription(
            PointStamped,
            self.aruco_input_topic,
            self.aruco_callback,
            10
        )

        self.sub_yolo = self.create_subscription(
            PointStamped,
            self.yolo_input_topic,
            self.yolo_callback,
            10
        )

        self.pub_aruco_base = self.create_publisher(
            PointStamped,
            self.aruco_output_topic,
            10
        )

        self.pub_yolo_base = self.create_publisher(
            PointStamped,
            self.yolo_output_topic,
            10
        )

        self.get_logger().info('========================================')
        self.get_logger().info('Camera Point Base Transformer Initialized (RIGHT ARM, OPTICAL INCLUDED)')
        self.get_logger().info(f'aruco_input_topic  : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic   : {self.yolo_input_topic}')
        self.get_logger().info(f'aruco_output_topic : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic  : {self.yolo_output_topic}')
        self.get_logger().info(f'base_frame         : {self.base_frame}')
        self.get_logger().info(f'gripper_frame      : {self.gripper_frame}')
        self.get_logger().info(f'use_msg_timestamp  : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec     : {self.tf_timeout_sec}')
        self.get_logger().info('T_cam_to_gripper:')
        self.get_logger().info(f'\n{self.T_cam_to_gripper}')
        self.get_logger().info('========================================')

    def aruco_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_aruco_base, 'ARUCO-R')

    def yolo_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_yolo_base, 'YOLO-R')

    def transform_and_publish(self, msg: PointStamped, publisher, source_name='UNKNOWN'):
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        if self.debug:
            self.get_logger().info(
                f'[{source_name} INPUT] '
                f'frame={msg.header.frame_id}, '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, '
                f'camera_xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})'
            )

        try:
            p_gripper = self.T_cam_to_gripper @ p_cam
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Camera->Gripper transform failed: {repr(e)}')
            return

        t = self.lookup_base_from_gripper(msg, source_name)
        if t is None:
            return

        try:
            T_gripper_to_base = self.make_transform_matrix(t)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Failed to build TF matrix: {repr(e)}')
            return

        try:
            p_base = T_gripper_to_base @ p_gripper
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Gripper->Base transform failed: {repr(e)}')
            return

        out_msg = PointStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.point.x = float(p_base[0])
        out_msg.point.y = float(p_base[1])
        out_msg.point.z = float(p_base[2])
        publisher.publish(out_msg)

        self.get_logger().info(
            '\n'
            f'[{source_name} BASE TRANSFORM]\n'
            f'Camera  : [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}]\n'
            f'Gripper : [{p_gripper[0]:.4f}, {p_gripper[1]:.4f}, {p_gripper[2]:.4f}]\n'
            f'Base    : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]'
        )

    def lookup_base_from_gripper(self, msg: PointStamped, source_name='UNKNOWN'):
        if self.use_msg_timestamp:
            try:
                target_time = rclpy.time.Time.from_msg(msg.header.stamp)
                t = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.gripper_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                if self.debug:
                    self.get_logger().info(
                        f'[{source_name} TF] using msg timestamp: '
                        f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
                    )
                return t
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup with msg timestamp failed (extrapolation). '
                    f'Fallback to latest TF. Detail: {e}'
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup failed with msg timestamp: '
                    f'{self.base_frame} <- {self.gripper_frame} | {e}'
                )
            except Exception as e:
                self.get_logger().error(
                    f'[{source_name}] Unexpected TF error with msg stamp: {repr(e)}'
                )

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
            if self.debug:
                tf_stamp = t.header.stamp
                self.get_logger().info(
                    f'[{source_name} TF] using latest TF: '
                    f'{tf_stamp.sec}.{tf_stamp.nanosec:09d}'
                )
            return t
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'[{source_name}] TF latest lookup failed: {self.base_frame} <- {self.gripper_frame} | {e}'
            )
            return None
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Unexpected TF error on latest lookup: {repr(e)}')
            return None

    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation

        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T


def main():
    rclpy.init()
    node = CameraPointBaseTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()