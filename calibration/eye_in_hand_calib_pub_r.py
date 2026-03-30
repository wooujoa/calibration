#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
import tf2_ros


class CameraPointBaseTransformerRight(Node):
    def __init__(self):
        super().__init__('camera_point_base_transformer_right')

        # ------------------------------------------------------------------
        # Hand-eye calibration: camera optical frame -> RIGHT gripper
        # ------------------------------------------------------------------
        self.T_cam_to_gripper = np.array([
            [ 0.9954,  0.0000, -0.0958,  0.0982],
            [ 0.0000, -1.0000,  0.0000,  0.0000],
            [-0.0958,  0.0000, -0.9954, -0.0725],
            [ 0.0000,  0.0000,  0.0000,  1.0000],
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # Fixed grasp->tool rotation offset from audit result
        # ------------------------------------------------------------------
        self.declare_parameter('apply_grasp_tool_offset', True)
        self.T_grasp_to_tool = np.eye(4, dtype=np.float64)
        self.T_grasp_to_tool[:3, :3] = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # Topics / frames
        # ------------------------------------------------------------------
        self.declare_parameter('aruco_input_topic', '/aruco/marker_3d_r')
        self.declare_parameter('yolo_input_topic', '/yolo/target_3d_pose_r')
        self.declare_parameter('aruco_output_topic', '/aruco/marker_base_pose_r')
        self.declare_parameter('yolo_output_topic', '/yolo/target_base_pose_r')

        self.declare_parameter('grasp_pose_input_topic', '/grasp/best_pose_raw')
        self.declare_parameter('grasp_pose_output_topic', '/grasp/best_pose_base_r')
        self.declare_parameter('grasp_pose_raw_output_topic', '/grasp/best_pose_raw_base_r')

        self.declare_parameter('contact_point_input_topic', '/grasp/best_contact_point')
        self.declare_parameter('contact_point_output_topic', '/grasp/best_contact_point_base_r')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base_frame_candidates', ['base_link', 'lift_link', 'arm_base_link'])
        self.declare_parameter('gripper_frame', 'arm_r_link7')
        self.declare_parameter('use_direct_camera_tf', False)
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        # compact logging controls
        self.declare_parameter('verbose_debug', False)
        self.declare_parameter('log_aruco_point', True)
        self.declare_parameter('log_yolo_point', True)
        self.declare_parameter('log_contact_point', True)
        self.declare_parameter('log_raw_pose', True)

        # optional TF broadcast for RViz / debugging
        self.declare_parameter('publish_debug_tf', True)
        self.declare_parameter('raw_pose_child_frame', 'grasp_best_base_r')
        self.declare_parameter('contact_point_child_frame', 'grasp_best_contact_base_r')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.aruco_input_topic = self.get_parameter('aruco_input_topic').value
        self.yolo_input_topic = self.get_parameter('yolo_input_topic').value
        self.aruco_output_topic = self.get_parameter('aruco_output_topic').value
        self.yolo_output_topic = self.get_parameter('yolo_output_topic').value

        self.grasp_pose_input_topic = self.get_parameter('grasp_pose_input_topic').value
        self.grasp_pose_output_topic = self.get_parameter('grasp_pose_output_topic').value
        self.grasp_pose_raw_output_topic = self.get_parameter('grasp_pose_raw_output_topic').value

        self.contact_point_input_topic = self.get_parameter('contact_point_input_topic').value
        self.contact_point_output_topic = self.get_parameter('contact_point_output_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.base_frame_candidates = list(self.get_parameter('base_frame_candidates').value)
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.use_direct_camera_tf = bool(self.get_parameter('use_direct_camera_tf').value)
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.verbose_debug = bool(self.get_parameter('verbose_debug').value)
        self.log_aruco_point = bool(self.get_parameter('log_aruco_point').value)
        self.log_yolo_point = bool(self.get_parameter('log_yolo_point').value)
        self.log_contact_point = bool(self.get_parameter('log_contact_point').value)
        self.log_raw_pose = bool(self.get_parameter('log_raw_pose').value)

        self.publish_debug_tf = bool(self.get_parameter('publish_debug_tf').value)
        self.raw_pose_child_frame = self.get_parameter('raw_pose_child_frame').value
        self.contact_point_child_frame = self.get_parameter('contact_point_child_frame').value
        self.apply_grasp_tool_offset = bool(self.get_parameter('apply_grasp_tool_offset').value)

        # ------------------------------------------------------------------
        # TF2
        # ------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_debug_tf else None

        # ------------------------------------------------------------------
        # Subscribers / Publishers
        # ------------------------------------------------------------------
        self.sub_aruco = self.create_subscription(
            PointStamped, self.aruco_input_topic, self.aruco_callback, 10
        )
        self.sub_yolo = self.create_subscription(
            PointStamped, self.yolo_input_topic, self.yolo_callback, 10
        )
        self.sub_grasp_pose = self.create_subscription(
            PoseStamped, self.grasp_pose_input_topic, self.grasp_pose_raw_callback, 10
        )
        self.sub_contact_point = self.create_subscription(
            PointStamped, self.contact_point_input_topic, self.contact_point_callback, 10
        )

        self.pub_aruco_base = self.create_publisher(PointStamped, self.aruco_output_topic, 10)
        self.pub_yolo_base = self.create_publisher(PointStamped, self.yolo_output_topic, 10)

        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, self.grasp_pose_output_topic, 10)
        self.pub_grasp_pose_raw_base = self.create_publisher(PoseStamped, self.grasp_pose_raw_output_topic, 10)

        self.pub_contact_point_base = self.create_publisher(PointStamped, self.contact_point_output_topic, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('Camera Point/Pose Base Transformer Initialized (RIGHT ARM, TARGET=base_link)')
        self.get_logger().info(f'aruco_input_topic            : {self.aruco_input_topic}')
        self.get_logger().info(f'yolo_input_topic             : {self.yolo_input_topic}')
        self.get_logger().info(f'grasp_pose_input_topic       : {self.grasp_pose_input_topic}')
        self.get_logger().info(f'contact_point_input_topic    : {self.contact_point_input_topic}')
        self.get_logger().info(f'aruco_output_topic           : {self.aruco_output_topic}')
        self.get_logger().info(f'yolo_output_topic            : {self.yolo_output_topic}')
        self.get_logger().info(f'grasp_pose_output_topic      : {self.grasp_pose_output_topic}')
        self.get_logger().info(f'grasp_pose_raw_output_topic  : {self.grasp_pose_raw_output_topic}')
        self.get_logger().info(f'contact_point_output_topic   : {self.contact_point_output_topic}')
        self.get_logger().info(f'base_frame                   : {self.base_frame}')
        self.get_logger().info(f'gripper_frame                : {self.gripper_frame}')
        self.get_logger().info(f'base_frame_candidates        : {self.base_frame_candidates}')
        self.get_logger().info(f'use_direct_camera_tf         : {self.use_direct_camera_tf}')
        self.get_logger().info(f'use_msg_timestamp            : {self.use_msg_timestamp}')
        self.get_logger().info(f'tf_timeout_sec               : {self.tf_timeout_sec}')
        self.get_logger().info(f'publish_debug_tf             : {self.publish_debug_tf}')
        self.get_logger().info(f'verbose_debug                : {self.verbose_debug}')
        self.get_logger().info(f'log_aruco_point              : {self.log_aruco_point}')
        self.get_logger().info(f'log_yolo_point               : {self.log_yolo_point}')
        self.get_logger().info(f'log_contact_point            : {self.log_contact_point}')
        self.get_logger().info(f'log_raw_pose                 : {self.log_raw_pose}')
        self.get_logger().info(f'apply_grasp_tool_offset      : {self.apply_grasp_tool_offset}')
        self.get_logger().info('NOTE: Output frame is base_link by default.')
        self.get_logger().info('NOTE: Pose gets Rx(180deg) tool correction; points do not.')
        self.get_logger().info('========================================')

    def aruco_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'ARUCO_R')
        if out is not None:
            self.pub_aruco_base.publish(out)
            if self.log_aruco_point:
                self.log_point_compact('ARUCO_BASE', out, input_frame=msg.header.frame_id)

    def yolo_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'YOLO_R')
        if out is not None:
            self.pub_yolo_base.publish(out)
            if self.log_yolo_point:
                self.log_point_compact('YOLO_BASE', out, input_frame=msg.header.frame_id)

    def contact_point_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'CONTACT_BASE')
        if out is None:
            return

        self.pub_contact_point_base.publish(out)

        if self.log_contact_point:
            self.log_point_compact('CONTACT_BASE', out, input_frame=msg.header.frame_id)

        if self.publish_debug_tf:
            self.broadcast_point_tf(out, self.contact_point_child_frame)

    def grasp_pose_raw_callback(self, msg: PoseStamped):
        out = self.transform_pose(msg, 'RAW_BASE')
        if out is None:
            return

        self.pub_grasp_pose_base.publish(out)
        self.pub_grasp_pose_raw_base.publish(out)

        if self.log_raw_pose:
            self.log_pose_compact('RAW_BASE', out, input_frame=msg.header.frame_id)

        if self.publish_debug_tf:
            self.broadcast_pose_tf(out, self.raw_pose_child_frame)

    def transform_point(self, msg: PointStamped, source_name='POINT'):
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

        if self.verbose_debug:
            self.get_logger().info(
                '\n'
                f'[{source_name}]\n'
                f'Input frame : {msg.header.frame_id}\n'
                f'Path        : {path_label}\n'
                f'Input xyz   : [{msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f}]\n'
                f'Intermed xyz: [{p_intermediate[0]:.4f}, {p_intermediate[1]:.4f}, {p_intermediate[2]:.4f}]\n'
                f'Base xyz    : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]'
            )
        return out_msg

    def transform_pose(self, msg: PoseStamped, source_name='POSE'):
        try:
            T_pose_base, T_intermediate, path_label = self.transform_pose_to_base_matrix(msg, source_name)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pose transform failed: {repr(e)}')
            return None

        out_msg = PoseStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.pose = self.matrix_to_pose(T_pose_base)

        if self.verbose_debug:
            in_p = msg.pose.position
            in_q = msg.pose.orientation
            mid_q = R.from_matrix(T_intermediate[:3, :3]).as_quat()
            out_p = out_msg.pose.position
            out_q = out_msg.pose.orientation

            self.get_logger().info(
                '\n'
                f'[{source_name}]\n'
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

    def transform_point_to_base(self, msg: PointStamped, source_name='POINT'):
        p_in = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)
        frame_id = msg.header.frame_id.strip()

        if frame_id == self.base_frame:
            return p_in.copy(), p_in[:3].copy(), 'base->base(pass-through)'

        if self.use_direct_camera_tf and frame_id not in ('', self.gripper_frame):
            resolved_base, t_direct = self.lookup_base_from_source_frame(frame_id, msg.header.stamp, source_name)
            if t_direct is not None:
                T_source_to_base = self.make_transform_matrix(t_direct)
                p_base = T_source_to_base @ p_in
                return p_base, p_in[:3].copy(), f'{frame_id}->{resolved_base}(direct TF)'

        resolved_base, t = self.resolve_base_frame(msg.header.stamp, source_name)
        if t is None:
            raise RuntimeError('base<-gripper TF unavailable')

        T_gripper_to_base = self.make_transform_matrix(t)

        if frame_id == self.gripper_frame:
            p_base = T_gripper_to_base @ p_in
            return p_base, p_in[:3].copy(), f'gripper->{resolved_base}'

        p_gripper = self.T_cam_to_gripper @ p_in
        p_base = T_gripper_to_base @ p_gripper
        return p_base, p_gripper[:3].copy(), f'{frame_id}(camera-like)->gripper->{resolved_base}'

    def transform_pose_to_base_matrix(self, msg: PoseStamped, source_name='POSE'):
        frame_id = msg.header.frame_id.strip()
        T_pose_in = self.pose_to_matrix(msg.pose.position, msg.pose.orientation)

        if frame_id == self.base_frame:
            T_pose_base = T_pose_in.copy()
            T_intermediate = T_pose_in.copy()
            path_label = 'base->base(pass-through)'
        elif self.use_direct_camera_tf and frame_id not in ('', self.gripper_frame):
            resolved_base, t_direct = self.lookup_base_from_source_frame(frame_id, msg.header.stamp, source_name)
            if t_direct is not None:
                T_source_to_base = self.make_transform_matrix(t_direct)
                T_pose_base = T_source_to_base @ T_pose_in
                T_intermediate = T_pose_in.copy()
                path_label = f'{frame_id}->{resolved_base}(direct TF)'
            else:
                resolved_base, t = self.resolve_base_frame(msg.header.stamp, source_name)
                if t is None:
                    raise RuntimeError('base<-gripper TF unavailable')
                T_gripper_to_base = self.make_transform_matrix(t)
                T_pose_gripper = self.T_cam_to_gripper @ T_pose_in
                T_pose_base = T_gripper_to_base @ T_pose_gripper
                T_intermediate = T_pose_gripper
                path_label = f'{frame_id}(camera-like)->gripper->{resolved_base}'
        else:
            resolved_base, t = self.resolve_base_frame(msg.header.stamp, source_name)
            if t is None:
                raise RuntimeError('base<-gripper TF unavailable')

            T_gripper_to_base = self.make_transform_matrix(t)

            if frame_id == self.gripper_frame:
                T_pose_base = T_gripper_to_base @ T_pose_in
                T_intermediate = T_pose_in.copy()
                path_label = f'gripper->{resolved_base}'
            else:
                T_pose_gripper = self.T_cam_to_gripper @ T_pose_in
                T_pose_base = T_gripper_to_base @ T_pose_gripper
                T_intermediate = T_pose_gripper
                path_label = f'{frame_id}(camera-like)->gripper->{resolved_base}'

        if self.apply_grasp_tool_offset:
            T_pose_base = T_pose_base @ self.T_grasp_to_tool
            path_label += '->tool_offset(Rx180)'

        return T_pose_base, T_intermediate, path_label

    def lookup_base_from_gripper(self, stamp, source_name='TF'):
        _resolved_base, t = self.resolve_base_frame(stamp, source_name)
        return t

    def resolve_base_frame(self, stamp, source_name='TF'):
        candidates = []
        if self.base_frame:
            candidates.append(self.base_frame)
        for c in self.base_frame_candidates:
            if c and c not in candidates:
                candidates.append(c)

        for cand in candidates:
            t = self.lookup_transform_with_fallback(cand, self.gripper_frame, stamp, source_name)
            if t is not None:
                if cand != self.base_frame:
                    self.get_logger().warn(
                        f'[{source_name}] base_frame={self.base_frame} unavailable, fallback to {cand}'
                    )
                self.base_frame = cand
                return cand, t

        return None, None

    def lookup_transform_with_fallback(self, target_frame, source_frame, stamp, source_name='TF'):
        if self.use_msg_timestamp:
            try:
                target_time = rclpy.time.Time.from_msg(stamp)
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup with msg timestamp failed (extrapolation). '
                    f'Fallback to latest TF. Detail: {e}'
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f'[{source_name}] TF lookup failed with msg timestamp: '
                    f'{target_frame} <- {source_frame} | {e}'
                )
            except Exception as e:
                self.get_logger().error(f'[{source_name}] Unexpected TF error with msg stamp: {repr(e)}')

        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'[{source_name}] TF latest lookup failed: '
                f'{target_frame} <- {source_frame} | {e}'
            )
            return None
        except Exception as e:
            self.get_logger().error(f'[{source_name}] Unexpected TF error on latest lookup: {repr(e)}')
            return None

    def lookup_base_from_source_frame(self, source_frame, stamp, source_name='TF'):
        if not source_frame:
            return None, None

        candidates = []
        if self.base_frame:
            candidates.append(self.base_frame)
        for c in self.base_frame_candidates:
            if c and c not in candidates:
                candidates.append(c)

        for cand in candidates:
            t = self.lookup_transform_with_fallback(cand, source_frame, stamp, source_name)
            if t is not None:
                if cand != self.base_frame:
                    self.get_logger().warn(
                        f'[{source_name}] base_frame={self.base_frame} unavailable for direct TF, fallback to {cand}'
                    )
                self.base_frame = cand
                return cand, t

        return None, None

    def log_point_compact(self, tag: str, msg: PointStamped, input_frame: str = ''):
        extra = f' input_frame={input_frame}' if input_frame else ''
        self.get_logger().info(
            f'[{tag}] xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f}) '
            f'frame={msg.header.frame_id}{extra} '
            f't={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        )

    def log_pose_compact(self, tag: str, msg: PoseStamped, input_frame: str = ''):
        p = msg.pose.position
        q = msg.pose.orientation
        extra = f' input_frame={input_frame}' if input_frame else ''
        self.get_logger().info(
            f'[{tag}] xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) '
            f'quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}) '
            f'frame={msg.header.frame_id}{extra} '
            f't={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        )

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
    node = CameraPointBaseTransformerRight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()