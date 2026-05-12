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
        # Parameters
        # ==================================================
        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d_zed')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_zed')

        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose_zed')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose_zed')

        self.declare_parameter('base_frame', 'base_link')

        # 입력 PointStamped의 frame_id를 그대로 사용할지
        self.declare_parameter('use_msg_frame_id', True)

        # msg.header.frame_id가 비어 있거나 잘못된 경우 fallback
        self.declare_parameter('camera_frame_fallback', 'zedm_left_camera_optical_frame')

        # msg timestamp 기준 TF를 우선 사용
        self.declare_parameter('use_msg_timestamp', True)

        # msg timestamp TF 실패 시 latest TF로 fallback 할지
        self.declare_parameter('fallback_to_latest_tf', True)

        self.declare_parameter('tf_timeout_sec', 0.2)

        self.declare_parameter('debug', True)
        self.declare_parameter('print_tf_matrix', True)

        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.use_msg_frame_id = bool(self.get_parameter('use_msg_frame_id').value)
        self.camera_frame_fallback = self.get_parameter('camera_frame_fallback').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.fallback_to_latest_tf = bool(self.get_parameter('fallback_to_latest_tf').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.debug = bool(self.get_parameter('debug').value)
        self.print_tf_matrix = bool(self.get_parameter('print_tf_matrix').value)

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
        self.get_logger().info('ZED Point Base Transformer Initialized (TF ONLY)')
        self.get_logger().info(f'aruco_input_topic      : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic       : {self.yolo_input_topic}')
        self.get_logger().info(f'aruco_output_topic     : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic      : {self.yolo_output_topic}')
        self.get_logger().info(f'base_frame             : {self.base_frame}')
        self.get_logger().info(f'use_msg_frame_id       : {self.use_msg_frame_id}')
        self.get_logger().info(f'camera_frame_fallback  : {self.camera_frame_fallback}')
        self.get_logger().info(f'use_msg_timestamp      : {self.use_msg_timestamp}')
        self.get_logger().info(f'fallback_to_latest_tf  : {self.fallback_to_latest_tf}')
        self.get_logger().info(f'tf_timeout_sec         : {self.tf_timeout_sec}')
        self.get_logger().info(f'debug                  : {self.debug}')
        self.get_logger().info(f'print_tf_matrix        : {self.print_tf_matrix}')
        self.get_logger().info('NOTE: hand-eye matrix is NOT used.')
        self.get_logger().info('========================================')

    def aruco_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_aruco_base, 'ARUCO-ZED')

    def yolo_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_yolo_base, 'YOLO-ZED')

    def transform_and_publish(self, msg: PointStamped, publisher, source_name='UNKNOWN'):
        source_frame = self.resolve_source_frame(msg)
        if source_frame is None:
            self.get_logger().warn(f'[{source_name}] source frame is empty and no fallback is available.')
            return

        p_src = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        if self.debug:
            self.get_logger().info(
                f'[{source_name} INPUT] '
                f'frame={source_frame}, '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, '
                f'xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})'
            )

        tf_msg, tf_mode = self.lookup_transform(self.base_frame, source_frame, msg, source_name)
        if tf_msg is None:
            return

        try:
            T_base_from_src = self.make_transform_matrix(tf_msg)
            p_base = T_base_from_src @ p_src
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Transform application failed: {repr(e)}')
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
            f'TF Mode     : {tf_mode}\n'
            f'SourceFrame : {source_frame}\n'
            f'Source XYZ  : [{p_src[0]:.4f}, {p_src[1]:.4f}, {p_src[2]:.4f}]\n'
            f'Base XYZ    : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]'
        )

        if self.print_tf_matrix:
            self.print_tf_debug(tf_msg, source_name)

    def resolve_source_frame(self, msg: PointStamped):
        if self.use_msg_frame_id:
            if msg.header.frame_id is not None and msg.header.frame_id != '':
                return msg.header.frame_id

        if self.camera_frame_fallback is not None and self.camera_frame_fallback != '':
            return self.camera_frame_fallback

        return None

    def lookup_transform(self, target_frame: str, source_frame: str, msg: PointStamped, source_name='UNKNOWN'):
        # --------------------------------------------------
        # 1) msg timestamp 기준 우선 시도
        # --------------------------------------------------
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)

                can_tx = self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )

                if can_tx:
                    tf_msg = self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        target_time,
                        timeout=Duration(seconds=self.tf_timeout_sec)
                    )

                    if self.debug:
                        self.get_logger().info(
                            f'[{source_name} TF] using msg timestamp: '
                            f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
                        )
                    return tf_msg, 'msg_timestamp'

                self.get_logger().warn(
                    f'[{source_name}] can_transform failed with msg timestamp: '
                    f'{target_frame} <- {source_frame}'
                )

            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup with msg timestamp failed (extrapolation): {e}'
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup failed with msg timestamp: '
                    f'{target_frame} <- {source_frame} | {e}'
                )
            except Exception as e:
                self.get_logger().error(
                    f'[{source_name}] Unexpected TF error with msg stamp: {repr(e)}'
                )

        # --------------------------------------------------
        # 2) latest TF fallback
        # --------------------------------------------------
        if not self.fallback_to_latest_tf:
            return None, 'none'

        try:
            can_tx = self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )

            if not can_tx:
                self.get_logger().warn(
                    f'[{source_name}] can_transform latest failed: '
                    f'{target_frame} <- {source_frame}'
                )
                return None, 'none'

            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )

            if self.debug:
                tf_stamp = tf_msg.header.stamp
                self.get_logger().info(
                    f'[{source_name} TF] fallback to latest TF: '
                    f'{tf_stamp.sec}.{tf_stamp.nanosec:09d}'
                )

            return tf_msg, 'latest'

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'[{source_name}] TF latest lookup failed: {target_frame} <- {source_frame} | {e}'
            )
            return None, 'none'
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Unexpected TF error on latest lookup: {repr(e)}')
            return None, 'none'

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