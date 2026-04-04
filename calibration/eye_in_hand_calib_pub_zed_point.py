#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R


class ZedPointBaseTransformer(Node):
    def __init__(self):
        super().__init__('zed_point_base_transformer')

        # ==================================================
        # Fixed hand-eye matrix (CAD Design Values)
        # OpenCV result: cam -> head
        # ==================================================
        self.T_cam_to_head = np.array([
            [-0.035023,  0.029973,  0.998937,  0.048565   ],
            [-0.999381,  0.002159, -0.035103,  0.02498203 ],
            [-0.003208, -0.999548,  0.029879, -0.0109594  ],
            [ 0.0,       0.0,       0.0,       1.0        ]
        ], dtype=np.float64)
        
        """
        # calibration 한 행렬
        self.T_cam_to_head = np.array([
            [-0.035023,  0.029973,  0.998937,  0.048565],
            [-0.999381,  0.002159, -0.035103,  0.024914],
            [-0.003208, -0.999548,  0.029879,  0.006624],
            [ 0.0,       0.0,       0.0,       1.0     ]
        ], dtype=np.float64)
        """
        
        """
        # urdf상 행렬
        self.T_cam_to_head = np.array([
            [ 0.0,  0.0,  1.0,  0.0238122 ],
            [-1.0,  0.0,  0.0,  0.02498203],
            [ 0.0, -1.0,  0.0, -0.0109594 ],
            [ 0.0,  0.0,  0.0,  1.0       ]
        ], dtype=np.float64)
        """
        
        
        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d_zed')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_zed')

        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose_zed')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose_zed')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'head_link2')

        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        self.declare_parameter('debug', True)
        self.declare_parameter('print_tf_matrix', True)
        self.declare_parameter('print_handeye_matrix', True)

        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.head_frame = self.get_parameter('head_frame').value

        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.debug = bool(self.get_parameter('debug').value)
        self.print_tf_matrix = bool(self.get_parameter('print_tf_matrix').value)
        self.print_handeye_matrix = bool(self.get_parameter('print_handeye_matrix').value)

        # ==================================================
        # TF
        # ==================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================================================
        # Subscribers
        # ==================================================
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

        # ==================================================
        # Publishers
        # ==================================================
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
        self.get_logger().info('ZED Point Base Transformer Initialized (CAD HANDe-EYE APPLIED)')
        self.get_logger().info(f'aruco_input_topic   : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic    : {self.yolo_input_topic}')
        self.get_logger().info(f'aruco_output_topic  : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic   : {self.yolo_output_topic}')
        self.get_logger().info(f'base_frame          : {self.base_frame}')
        self.get_logger().info(f'head_frame          : {self.head_frame}')
        self.get_logger().info(f'use_msg_timestamp   : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec      : {self.tf_timeout_sec}')
        self.get_logger().info(f'debug               : {self.debug}')
        self.get_logger().info(f'print_tf_matrix     : {self.print_tf_matrix}')
        self.get_logger().info(f'print_handeye_matrix: {self.print_handeye_matrix}')
        if self.print_handeye_matrix:
            self.get_logger().info(f'T_cam_to_head:\n{self.T_cam_to_head}')
        self.get_logger().info('========================================')

    def aruco_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_aruco_base, 'ARUCO-ZED')

    def yolo_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_yolo_base, 'YOLO-ZED')

    def transform_and_publish(self, msg: PointStamped, publisher, source_name='UNKNOWN'):
        # --------------------------------------------------
        # Input point in camera optical frame
        # --------------------------------------------------
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        if self.debug:
            self.get_logger().info(
                f'[{source_name} INPUT] '
                f'frame={msg.header.frame_id}, '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, '
                f'cam_xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})'
            )

        # --------------------------------------------------
        # Camera -> Head using fixed hand-eye matrix
        # --------------------------------------------------
        try:
            p_head = self.T_cam_to_head @ p_cam
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Camera->Head transform failed: {repr(e)}')
            return

        # --------------------------------------------------
        # Base <- Head from TF/FK
        # --------------------------------------------------
        tf_msg = self.lookup_base_from_head(msg, source_name)
        if tf_msg is None:
            return

        try:
            T_head_to_base = self.make_transform_matrix(tf_msg)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Failed to build head->base TF matrix: {repr(e)}')
            return

        # --------------------------------------------------
        # Final base point
        # --------------------------------------------------
        try:
            p_base = T_head_to_base @ p_head
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Head->Base transform failed: {repr(e)}')
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
            f'Camera : [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}]\n'
            f'Head   : [{p_head[0]:.4f}, {p_head[1]:.4f}, {p_head[2]:.4f}]\n'
            f'Base   : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]'
        )

        if self.print_tf_matrix:
            self.print_tf_debug(tf_msg, source_name)

    def lookup_base_from_head(self, msg: PointStamped, source_name='UNKNOWN'):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.head_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                if self.debug:
                    self.get_logger().info(
                        f'[{source_name} TF] using msg timestamp: '
                        f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
                    )
                return tf_msg
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup with msg timestamp failed (extrapolation). '
                    f'Fallback to latest TF. Detail: {e}'
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup failed with msg timestamp: '
                    f'{self.base_frame} <- {self.head_frame} | {e}'
                )
            except Exception as e:
                self.get_logger().error(
                    f'[{source_name}] Unexpected TF error with msg stamp: {repr(e)}'
                )

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.head_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
            if self.debug:
                tf_stamp = tf_msg.header.stamp
                self.get_logger().info(
                    f'[{source_name} TF] using latest TF: '
                    f'{tf_stamp.sec}.{tf_stamp.nanosec:09d}'
                )
            return tf_msg
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'[{source_name}] TF latest lookup failed: {self.base_frame} <- {self.head_frame} | {e}'
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

    def print_tf_debug(self, tf_msg, source_name='UNKNOWN'):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        self.get_logger().info(
            f'[{source_name} TF DETAIL] '
            f'translation=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}), '
            f'quaternion=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
        )

        self.get_logger().info(
            f'[{source_name} TF MATRIX]\n'
            f'[{rot[0,0]: .4f} {rot[0,1]: .4f} {rot[0,2]: .4f} | {t.x: .4f}]\n'
            f'[{rot[1,0]: .4f} {rot[1,1]: .4f} {rot[1,2]: .4f} | {t.y: .4f}]\n'
            f'[{rot[2,0]: .4f} {rot[2,1]: .4f} {rot[2,2]: .4f} | {t.z: .4f}]'
        )


def main():
    rclpy.init()
    node = ZedPointBaseTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()