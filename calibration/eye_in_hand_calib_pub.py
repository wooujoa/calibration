#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
import tf2_ros


class CameraPointBaseTransformer(Node):
    def __init__(self):
        super().__init__('camera_point_base_transformer')

        # ------------------------------------------------------------------
        # 1) Hand-Eye Calibration Matrix
        #    Camera -> Gripper
        # ------------------------------------------------------------------
        self.T_cam_to_gripper = np.array([
            [ 0.9954,  0.0000, -0.0958,  0.0982],
            [ 0.0000, -1.0000,  0.0000,  0.0000],
            [-0.0958,  0.0000, -0.9954, -0.0725],
            [ 0.0000,  0.0000,  0.0000,  1.0000]
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # 2) Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_pose')
        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose')

        # grasp raw / vis / selected alias
        self.declare_parameter('grasp_pose_raw_input_topic', '/grasp/best_pose_raw')
        self.declare_parameter('grasp_pose_raw_output_topic', '/grasp/best_pose_raw_base')
        self.declare_parameter('grasp_pose_vis_input_topic', '/grasp/best_pose_vis')
        self.declare_parameter('grasp_pose_vis_output_topic', '/grasp/best_pose_vis_base')
        self.declare_parameter('grasp_pose_selected_output_topic', '/grasp/best_pose_base')
        self.declare_parameter('selected_pose_source', 'raw')  # raw | vis

        # backward-compatible single-topic params
        self.declare_parameter('grasp_pose_input_topic', '/grasp/best_pose_raw')
        self.declare_parameter('grasp_pose_output_topic', '/grasp/best_pose_base')

        self.declare_parameter('contact_point_input_topic', '/grasp/best_contact_point')
        self.declare_parameter('contact_point_output_topic', '/grasp/best_contact_point_base')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'arm_l_link7')
        self.declare_parameter('use_msg_timestamp', True)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('debug', True)

        # optional TF broadcast for RViz / debugging
        self.declare_parameter('publish_debug_tf', True)
        self.declare_parameter('raw_pose_child_frame', 'grasp_best_raw_base')
        self.declare_parameter('vis_pose_child_frame', 'grasp_best_vis_base')
        self.declare_parameter('selected_pose_child_frame', 'grasp_best_base')
        self.declare_parameter('contact_point_child_frame', 'grasp_best_contact_base')

        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        # allow old launch files to keep working
        raw_in_default = self.get_parameter('grasp_pose_input_topic').value
        selected_out_default = self.get_parameter('grasp_pose_output_topic').value

        self.grasp_pose_raw_input_topic = self.get_parameter('grasp_pose_raw_input_topic').value or raw_in_default
        self.grasp_pose_raw_output_topic = self.get_parameter('grasp_pose_raw_output_topic').value
        self.grasp_pose_vis_input_topic = self.get_parameter('grasp_pose_vis_input_topic').value
        self.grasp_pose_vis_output_topic = self.get_parameter('grasp_pose_vis_output_topic').value
        self.grasp_pose_selected_output_topic = self.get_parameter('grasp_pose_selected_output_topic').value or selected_out_default
        self.selected_pose_source = str(self.get_parameter('selected_pose_source').value).strip().lower()
        if self.selected_pose_source not in ('raw', 'vis'):
            self.get_logger().warn(
                f"Invalid selected_pose_source='{self.selected_pose_source}', fallback to 'raw'"
            )
            self.selected_pose_source = 'raw'

        self.contact_point_input_topic = self.get_parameter('contact_point_input_topic').value
        self.contact_point_output_topic = self.get_parameter('contact_point_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.debug = bool(self.get_parameter('debug').value)

        self.publish_debug_tf = bool(self.get_parameter('publish_debug_tf').value)
        self.raw_pose_child_frame = self.get_parameter('raw_pose_child_frame').value
        self.vis_pose_child_frame = self.get_parameter('vis_pose_child_frame').value
        self.selected_pose_child_frame = self.get_parameter('selected_pose_child_frame').value
        self.contact_point_child_frame = self.get_parameter('contact_point_child_frame').value

        # ------------------------------------------------------------------
        # 3) TF2
        # ------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_debug_tf else None

        # ------------------------------------------------------------------
        # 4) Subscribers / Publishers
        # ------------------------------------------------------------------
        self.sub_aruco = self.create_subscription(
            PointStamped,
            self.aruco_input_topic,
            self.aruco_callback,
            10,
        )

        self.sub_yolo = self.create_subscription(
            PointStamped,
            self.yolo_input_topic,
            self.yolo_callback,
            10,
        )

        self.sub_grasp_pose_raw = self.create_subscription(
            PoseStamped,
            self.grasp_pose_raw_input_topic,
            self.grasp_pose_raw_callback,
            10,
        )

        self.sub_grasp_pose_vis = self.create_subscription(
            PoseStamped,
            self.grasp_pose_vis_input_topic,
            self.grasp_pose_vis_callback,
            10,
        )

        self.sub_contact_point = self.create_subscription(
            PointStamped,
            self.contact_point_input_topic,
            self.contact_point_callback,
            10,
        )

        self.pub_aruco_base = self.create_publisher(
            PointStamped,
            self.aruco_output_topic,
            10,
        )

        self.pub_yolo_base = self.create_publisher(
            PointStamped,
            self.yolo_output_topic,
            10,
        )

        self.pub_grasp_pose_raw_base = self.create_publisher(
            PoseStamped,
            self.grasp_pose_raw_output_topic,
            10,
        )

        self.pub_grasp_pose_vis_base = self.create_publisher(
            PoseStamped,
            self.grasp_pose_vis_output_topic,
            10,
        )

        self.pub_grasp_pose_selected_base = self.create_publisher(
            PoseStamped,
            self.grasp_pose_selected_output_topic,
            10,
        )

        self.pub_contact_point_base = self.create_publisher(
            PointStamped,
            self.contact_point_output_topic,
            10,
        )

        self.get_logger().info('========================================')
        self.get_logger().info('Camera Point/Pose Base Transformer Initialized')
        self.get_logger().info(f'aruco_input_topic            : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic             : {self.yolo_input_topic}')
        self.get_logger().info(f'grasp_pose_raw_input_topic   : {self.grasp_pose_raw_input_topic}')
        self.get_logger().info(f'grasp_pose_vis_input_topic   : {self.grasp_pose_vis_input_topic}')
        self.get_logger().info(f'contact_point_input_topic    : {self.contact_point_input_topic}')
        self.get_logger().info(f'aruco_output_topic           : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic            : {self.yolo_output_topic}')
        self.get_logger().info(f'grasp_pose_raw_output_topic  : {self.grasp_pose_raw_output_topic}')
        self.get_logger().info(f'grasp_pose_vis_output_topic  : {self.grasp_pose_vis_output_topic}')
        self.get_logger().info(f'grasp_pose_selected_output   : {self.grasp_pose_selected_output_topic}')
        self.get_logger().info(f'selected_pose_source         : {self.selected_pose_source}')
        self.get_logger().info(f'contact_point_output_topic   : {self.contact_point_output_topic}')
        self.get_logger().info(f'base_frame                   : {self.base_frame}')
        self.get_logger().info(f'gripper_frame                : {self.gripper_frame}')
        self.get_logger().info(f'use_msg_timestamp            : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec               : {self.tf_timeout_sec}')
        self.get_logger().info(f'publish_debug_tf             : {self.publish_debug_tf}')
        self.get_logger().info('T_cam_to_gripper:')
        self.get_logger().info(f'\n{self.T_cam_to_gripper}')
        self.get_logger().info('NOTE: /grasp/best_pose_raw should be used for robot execution.')
        self.get_logger().info('      /grasp/best_pose_vis is kept separately for RViz comparison.')
        self.get_logger().info('========================================')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def aruco_callback(self, msg: PointStamped):
        self.transform_point_and_publish(
            msg=msg,
            publisher=self.pub_aruco_base,
            source_name='ARUCO',
        )

    def yolo_callback(self, msg: PointStamped):
        self.transform_point_and_publish(
            msg=msg,
            publisher=self.pub_yolo_base,
            source_name='YOLO',
        )

    def contact_point_callback(self, msg: PointStamped):
        out_msg = self.transform_point_and_publish(
            msg=msg,
            publisher=self.pub_contact_point_base,
            source_name='CONTACT',
        )
        if out_msg is not None and self.publish_debug_tf:
            self.broadcast_point_tf(out_msg, self.contact_point_child_frame)

    def grasp_pose_raw_callback(self, msg: PoseStamped):
        out_msg = self.transform_pose_and_publish(
            msg=msg,
            publisher=self.pub_grasp_pose_raw_base,
            source_name='GRASP_POSE_RAW',
        )
        if out_msg is None:
            return

        if self.selected_pose_source == 'raw':
            self.pub_grasp_pose_selected_base.publish(out_msg)
            if self.publish_debug_tf:
                self.broadcast_pose_tf(out_msg, self.selected_pose_child_frame)

        if self.publish_debug_tf:
            self.broadcast_pose_tf(out_msg, self.raw_pose_child_frame)

    def grasp_pose_vis_callback(self, msg: PoseStamped):
        out_msg = self.transform_pose_and_publish(
            msg=msg,
            publisher=self.pub_grasp_pose_vis_base,
            source_name='GRASP_POSE_VIS',
        )
        if out_msg is None:
            return

        if self.selected_pose_source == 'vis':
            self.pub_grasp_pose_selected_base.publish(out_msg)
            if self.publish_debug_tf:
                self.broadcast_pose_tf(out_msg, self.selected_pose_child_frame)

        if self.publish_debug_tf:
            self.broadcast_pose_tf(out_msg, self.vis_pose_child_frame)

    # ------------------------------------------------------------------
    # Point transform
    # ------------------------------------------------------------------
    def transform_point_and_publish(self, msg: PointStamped, publisher, source_name='UNKNOWN'):
        try:
            p_base, p_intermediate, path_label = self.transform_point_to_base(msg, source_name)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] point transform failed: {repr(e)}')
            return None

        out_msg = PointStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.point.x = float(p_base[0])
        out_msg.point.y = float(p_base[1])
        out_msg.point.z = float(p_base[2])
        publisher.publish(out_msg)

        if self.debug:
            self.get_logger().info(
                '\n'
                f'[{source_name} BASE POINT]\n'
                f'Input frame : {msg.header.frame_id}\n'
                f'Path        : {path_label}\n'
                f'Input xyz   : [{msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f}]\n'
                f'Intermed xyz: [{p_intermediate[0]:.4f}, {p_intermediate[1]:.4f}, {p_intermediate[2]:.4f}]\n'
                f'Base xyz    : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]'
            )

        return out_msg

    def transform_point_to_base(self, msg: PointStamped, source_name='UNKNOWN'):
        p_in = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)
        frame_id = msg.header.frame_id.strip()

        if self.debug:
            self.get_logger().info(
                f'[{source_name} INPUT POINT] '
                f'frame={frame_id}, '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, '
                f'xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})'
            )

        if frame_id == self.base_frame:
            return p_in.copy(), p_in[:3].copy(), 'base->base(pass-through)'

        t = self.lookup_base_from_gripper(msg.header.stamp, source_name)
        if t is None:
            raise RuntimeError('base<-gripper TF unavailable')
        T_gripper_to_base = self.make_transform_matrix(t)

        if frame_id == self.gripper_frame:
            p_base = T_gripper_to_base @ p_in
            return p_base, p_in[:3].copy(), 'gripper->base'

        # Default: treat any other frame as camera-like input produced by perception node.
        p_gripper = self.T_cam_to_gripper @ p_in
        p_base = T_gripper_to_base @ p_gripper
        return p_base, p_gripper[:3].copy(), f'{frame_id}(camera-like)->gripper->base'

    # ------------------------------------------------------------------
    # Pose transform
    # ------------------------------------------------------------------
    def transform_pose_and_publish(self, msg: PoseStamped, publisher, source_name='UNKNOWN'):
        try:
            T_pose_base, T_intermediate, path_label = self.transform_pose_to_base_matrix(msg, source_name)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pose transform failed: {repr(e)}')
            return None

        out_msg = PoseStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.pose = self.matrix_to_pose(T_pose_base)
        publisher.publish(out_msg)

        if self.debug:
            in_p = msg.pose.position
            in_q = msg.pose.orientation
            mid_q = R.from_matrix(T_intermediate[:3, :3]).as_quat()
            out_p = out_msg.pose.position
            out_q = out_msg.pose.orientation
            self.get_logger().info(
                '\n'
                f'[{source_name} BASE POSE]\n'
                f'Input frame  : {msg.header.frame_id}\n'
                f'Path         : {path_label}\n'
                f'Input xyz    : [{in_p.x:.4f}, {in_p.y:.4f}, {in_p.z:.4f}]\n'
                f'Input quat   : [{in_q.x:.4f}, {in_q.y:.4f}, {in_q.z:.4f}, {in_q.w:.4f}]\n'
                f'Intermed xyz : [{T_intermediate[0,3]:.4f}, {T_intermediate[1,3]:.4f}, {T_intermediate[2,3]:.4f}]\n'
                f'Intermed quat: [{mid_q[0]:.4f}, {mid_q[1]:.4f}, {mid_q[2]:.4f}, {mid_q[3]:.4f}]\n'
                f'Base xyz     : [{out_p.x:.4f}, {out_p.y:.4f}, {out_p.z:.4f}]\n'
                f'Base quat    : [{out_q.x:.4f}, {out_q.y:.4f}, {out_q.z:.4f}, {out_q.w:.4f}]'
            )

        return out_msg

    def transform_pose_to_base_matrix(self, msg: PoseStamped, source_name='UNKNOWN'):
        frame_id = msg.header.frame_id.strip()
        T_pose_in = self.pose_to_matrix(msg.pose.position, msg.pose.orientation)

        if self.debug:
            p = msg.pose.position
            q = msg.pose.orientation
            self.get_logger().info(
                f'[{source_name} INPUT POSE] '
                f'frame={frame_id}, '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, '
                f'xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), '
                f'quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
            )

        if frame_id == self.base_frame:
            return T_pose_in.copy(), T_pose_in.copy(), 'base->base(pass-through)'

        t = self.lookup_base_from_gripper(msg.header.stamp, source_name)
        if t is None:
            raise RuntimeError('base<-gripper TF unavailable')
        T_gripper_to_base = self.make_transform_matrix(t)

        if frame_id == self.gripper_frame:
            T_pose_base = T_gripper_to_base @ T_pose_in
            return T_pose_base, T_pose_in.copy(), 'gripper->base'

        # Default: camera-like pose
        T_pose_gripper = self.T_cam_to_gripper @ T_pose_in
        T_pose_base = T_gripper_to_base @ T_pose_gripper
        return T_pose_base, T_pose_gripper, f'{frame_id}(camera-like)->gripper->base'

    # ------------------------------------------------------------------
    # TF lookup
    # ------------------------------------------------------------------
    def lookup_base_from_gripper(self, stamp, source_name='UNKNOWN'):
        if self.use_msg_timestamp:
            try:
                target_time = rclpy.time.Time.from_msg(stamp)
                t = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.gripper_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
                if self.debug:
                    self.get_logger().info(
                        f'[{source_name} TF] using msg timestamp: '
                        f'{stamp.sec}.{stamp.nanosec:09d}'
                    )
                return t
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup with msg timestamp failed '
                    f'(extrapolation). Fallback to latest TF. Detail: {e}'
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
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
            if self.debug:
                tf_stamp = t.header.stamp
                self.get_logger().info(
                    f'[{source_name} TF] using latest TF: '
                    f'{tf_stamp.sec}.{tf_stamp.nanosec:09d}'
                )
            return t
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f'[{source_name}] TF latest lookup failed: '
                f'{self.base_frame} <- {self.gripper_frame} | {e}'
            )
            return None
        except Exception as e:
            self.get_logger().error(
                f'[{source_name}] Unexpected TF error on latest lookup: {repr(e)}'
            )
            return None

    # ------------------------------------------------------------------
    # Debug TF broadcast
    # ------------------------------------------------------------------
    def broadcast_pose_tf(self, msg: PoseStamped, child_frame: str):
        if self.tf_broadcaster is None:
            return
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = self.base_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def broadcast_point_tf(self, msg: PointStamped, child_frame: str):
        if self.tf_broadcaster is None:
            return
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = self.base_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = msg.point.x
        t.transform.translation.y = msg.point.y
        t.transform.translation.z = msg.point.z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def pose_to_matrix(self, pos, quat):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        T[:3, 3] = [pos.x, pos.y, pos.z]
        return T

    def matrix_to_pose(self, T):
        pose = PoseStamped().pose
        q = R.from_matrix(T[:3, :3]).as_quat()

        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])

        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose


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