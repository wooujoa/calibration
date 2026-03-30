#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class ZedPointBaseTransformer(Node):
    def __init__(self):
        super().__init__('zed_point_base_transformer')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d_zed')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_zed')

        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose_zed')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose_zed')

        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('use_input_frame_id', True)
        self.declare_parameter('camera_frame_override', '')

        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        self.declare_parameter('debug', True)

        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.use_input_frame_id = bool(self.get_parameter('use_input_frame_id').value)
        self.camera_frame_override = self.get_parameter('camera_frame_override').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.debug = bool(self.get_parameter('debug').value)

        # --------------------------------------------------
        # TF
        # --------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
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
        self.get_logger().info('ZED Point Base Transformer Initialized')
        self.get_logger().info(f'aruco_input_topic    : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic     : {self.yolo_input_topic}')
        self.get_logger().info(f'aruco_output_topic   : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic    : {self.yolo_output_topic}')
        self.get_logger().info(f'base_frame           : {self.base_frame}')
        self.get_logger().info(f'use_input_frame_id   : {self.use_input_frame_id}')
        self.get_logger().info(f'camera_frame_override: {self.camera_frame_override}')
        self.get_logger().info(f'use_msg_timestamp    : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec       : {self.tf_timeout_sec}')
        self.get_logger().info(f'debug                : {self.debug}')
        self.get_logger().info('========================================')

    def aruco_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_aruco_base, 'ARUCO-ZED')

    def yolo_callback(self, msg: PointStamped):
        self.transform_and_publish(msg, self.pub_yolo_base, 'YOLO-ZED')

    def transform_and_publish(self, msg: PointStamped, publisher, source_name='UNKNOWN'):
        source_frame = self.resolve_source_frame(msg)

        if not source_frame:
            self.get_logger().warn(f'[{source_name}] Empty source frame. Skip.')
            return

        in_msg = PointStamped()
        in_msg.header.stamp = msg.header.stamp
        in_msg.header.frame_id = source_frame
        in_msg.point.x = float(msg.point.x)
        in_msg.point.y = float(msg.point.y)
        in_msg.point.z = float(msg.point.z)

        if self.debug:
            self.get_logger().info(
                f'[{source_name} INPUT] '
                f'frame={in_msg.header.frame_id}, '
                f'stamp={in_msg.header.stamp.sec}.{in_msg.header.stamp.nanosec:09d}, '
                f'xyz=({in_msg.point.x:.4f}, {in_msg.point.y:.4f}, {in_msg.point.z:.4f})'
            )

        tf_msg = self.lookup_transform(in_msg, source_name)
        if tf_msg is None:
            return

        try:
            out_msg = do_transform_point(in_msg, tf_msg)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Point transform failed: {repr(e)}')
            return

        out_msg.header.frame_id = self.base_frame
        publisher.publish(out_msg)

        self.get_logger().info(
            f'[{source_name} BASE] '
            f'camera_xyz=({in_msg.point.x:.4f}, {in_msg.point.y:.4f}, {in_msg.point.z:.4f}) -> '
            f'base_xyz=({out_msg.point.x:.4f}, {out_msg.point.y:.4f}, {out_msg.point.z:.4f})'
        )

    def resolve_source_frame(self, msg: PointStamped) -> str:
        if self.camera_frame_override:
            return self.camera_frame_override

        if self.use_input_frame_id and msg.header.frame_id:
            return msg.header.frame_id

        return msg.header.frame_id

    def lookup_transform(self, msg: PointStamped, source_name='UNKNOWN'):
        try:
            if self.use_msg_timestamp:
                target_time = Time.from_msg(msg.header.stamp)
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    msg.header.frame_id,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
                if self.debug:
                    self.get_logger().info(
                        f'[{source_name} TF] using msg timestamp: '
                        f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
                    )
                return tf_msg

            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
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

        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(
                f'[{source_name}] TF extrapolation failed: '
                f'{self.base_frame} <- {msg.header.frame_id} | {e}'
            )
            return None

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(
                f'[{source_name}] TF lookup failed: '
                f'{self.base_frame} <- {msg.header.frame_id} | {e}'
            )
            return None

        except Exception as e:
            self.get_logger().error(
                f'[{source_name}] Unexpected TF lookup error: {repr(e)}'
            )
            return None


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