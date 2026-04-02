#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros


class AnyGraspBaseTransformerRight(Node):
    """
    Base-frame visualization/debug node.

    Purpose:
      1) Transform AnyGrasp best grasp pose from camera optical frame to base_link
      2) Transform AnyGrasp contact point to base_link
      3) Transform SAM3 object center to base_link
      4) Transform YOLO target point cloud to base_link
      5) Publish base-frame MarkerArray that shows:
         - base axes
         - predicted grasp axes
         - current gripper axes
         - contact point
         - object center

    Recommended RViz setup:
      - Fixed Frame: base_link
      - PointCloud2: /yolo/target_pc_base
      - Pose: /anygrasp/best_pose_base_r
      - MarkerArray: /anygrasp/debug_axes_base
    """

    def __init__(self):
        super().__init__('anygrasp_base_transformer_right')

        # ------------------------------------------------------------
        # Hand-eye: camera optical -> arm_r_link7 (calibration result)
        # ------------------------------------------------------------
        self.T_cam_to_link7 = np.array([
            [ 0.9954,  0.0000, -0.0958,  0.0982],
            [ 0.0000, -1.0000,  0.0000,  0.0000],
            [-0.0958,  0.0000, -0.9954, -0.0725],
            [ 0.0000,  0.0000,  0.0000,  1.0000],
        ], dtype=np.float64)

        # ------------------------------------------------------------
        # Fixed URDF transform: arm_r_link7 -> gripper_r_rh_p12_rn_base
        # URDF: <origin rpy="0 3.14159265359 3.14159265359" xyz="0 0 -0.078"/>
        # ------------------------------------------------------------
        self.T_link7_to_gripper_base = np.eye(4, dtype=np.float64)
        self.T_link7_to_gripper_base[:3, :3] = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=np.float64)
        self.T_link7_to_gripper_base[:3, 3] = np.array([0.0, 0.0, -0.0780], dtype=np.float64)

        # Final hand-eye used by this node: camera optical -> gripper base
        self.T_cam_to_gripper = self.T_link7_to_gripper_base @ self.T_cam_to_link7

        # ------------------------------------------------------------
        # Pose-only alignment: AnyGrasp gripper frame -> ffw gripper base frame
        # 1) always apply local +90 deg CCW about Y
        # 2) optionally apply additional local 180 deg about Z
        #    ONLY when the aligned gripper +x axis points downward in base frame.
        #
        # This must affect ONLY grasp pose orientation,
        # not contact/object-center points nor point clouds.
        # Right-multiplication means local-axis rotations in the gripper frame.
        # Translation is intentionally zero so the grasp center is preserved.
        # ------------------------------------------------------------
        self.T_pose_align_y90 = np.eye(4, dtype=np.float64)
        self.T_pose_align_y90[:3, :3] = R.from_euler('y', 90.0, degrees=True).as_matrix()

        self.T_pose_align_z180 = np.eye(4, dtype=np.float64)
        self.T_pose_align_z180[:3, :3] = R.from_euler('z', 180.0, degrees=True).as_matrix()

        # ------------------------------------------------------------
        # Optional grasp->tool offset
        # ------------------------------------------------------------
        self.declare_parameter('apply_grasp_tool_offset', False)
        self.declare_parameter('apply_anygrasp_pose_frame_alignment', True)
        self.declare_parameter('auto_flip_pose_z_180_if_x_points_down', True)
        self.declare_parameter('x_axis_downward_flip_threshold', 0.0)
        self.T_grasp_to_tool = np.eye(4, dtype=np.float64)
        self.T_grasp_to_tool[:3, :3] = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=np.float64)

        # ------------------------------------------------------------
        # Topics / frames
        # ------------------------------------------------------------
        self.declare_parameter('grasp_pose_input_topic', '/anygrasp/best_pose_raw')
        self.declare_parameter('grasp_pose_output_topic', '/anygrasp/best_pose_base_r')

        self.declare_parameter('contact_point_input_topic', '/anygrasp/best_contact_point')
        self.declare_parameter('contact_point_output_topic', '/anygrasp/best_contact_point_base_r')

        self.declare_parameter('object_center_input_topic', '/sam3/object_center_camera')
        self.declare_parameter('object_center_output_topic', '/sam3/object_center_base')

        self.declare_parameter('target_pc_input_topic', '/yolo/target_pc')
        self.declare_parameter('target_pc_output_topic', '/yolo/target_pc_base')

        self.declare_parameter('debug_axes_base_topic', '/anygrasp/debug_axes_base')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base_frame_candidates', ['base_link', 'lift_link', 'arm_base_link'])
        self.declare_parameter('gripper_frame', 'gripper_r_rh_p12_rn_base')
        self.declare_parameter('camera_frame', 'camera_r_color_optical_frame')

        self.declare_parameter('use_direct_camera_tf', False)
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        # logging
        self.declare_parameter('verbose_debug', False)
        self.declare_parameter('log_contact_point', True)
        self.declare_parameter('log_pose', True)
        self.declare_parameter('log_pointcloud', False)

        # debug marker style
        self.declare_parameter('publish_debug_axes_base', True)
        self.declare_parameter('show_current_gripper_axes', True)
        self.declare_parameter('show_contact_marker', True)
        self.declare_parameter('show_object_center_marker', True)

        self.declare_parameter('axes_length', 0.08)
        self.declare_parameter('axes_shaft_diameter', 0.004)
        self.declare_parameter('axes_head_diameter', 0.008)
        self.declare_parameter('axes_head_length', 0.012)
        self.declare_parameter('axes_lifetime_sec', 0.0)

        self.declare_parameter('contact_marker_radius', 0.018)
        self.declare_parameter('object_center_marker_radius', 0.02)

        # ------------------------------------------------------------
        # Parameter fetch
        # ------------------------------------------------------------
        self.grasp_pose_input_topic = self.get_parameter('grasp_pose_input_topic').value
        self.grasp_pose_output_topic = self.get_parameter('grasp_pose_output_topic').value

        self.contact_point_input_topic = self.get_parameter('contact_point_input_topic').value
        self.contact_point_output_topic = self.get_parameter('contact_point_output_topic').value

        self.object_center_input_topic = self.get_parameter('object_center_input_topic').value
        self.object_center_output_topic = self.get_parameter('object_center_output_topic').value

        self.target_pc_input_topic = self.get_parameter('target_pc_input_topic').value
        self.target_pc_output_topic = self.get_parameter('target_pc_output_topic').value

        self.debug_axes_base_topic = self.get_parameter('debug_axes_base_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.base_frame_candidates = list(self.get_parameter('base_frame_candidates').value)
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.use_direct_camera_tf = bool(self.get_parameter('use_direct_camera_tf').value)
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.verbose_debug = bool(self.get_parameter('verbose_debug').value)
        self.log_contact_point = bool(self.get_parameter('log_contact_point').value)
        self.log_pose = bool(self.get_parameter('log_pose').value)
        self.log_pointcloud = bool(self.get_parameter('log_pointcloud').value)

        self.publish_debug_axes_base = bool(self.get_parameter('publish_debug_axes_base').value)
        self.show_current_gripper_axes = bool(self.get_parameter('show_current_gripper_axes').value)
        self.show_contact_marker = bool(self.get_parameter('show_contact_marker').value)
        self.show_object_center_marker = bool(self.get_parameter('show_object_center_marker').value)

        self.axes_length = float(self.get_parameter('axes_length').value)
        self.axes_shaft_diameter = float(self.get_parameter('axes_shaft_diameter').value)
        self.axes_head_diameter = float(self.get_parameter('axes_head_diameter').value)
        self.axes_head_length = float(self.get_parameter('axes_head_length').value)
        self.axes_lifetime_sec = float(self.get_parameter('axes_lifetime_sec').value)

        self.contact_marker_radius = float(self.get_parameter('contact_marker_radius').value)
        self.object_center_marker_radius = float(self.get_parameter('object_center_marker_radius').value)

        self.apply_grasp_tool_offset = bool(self.get_parameter('apply_grasp_tool_offset').value)
        self.apply_anygrasp_pose_frame_alignment = bool(self.get_parameter('apply_anygrasp_pose_frame_alignment').value)
        self.auto_flip_pose_z_180_if_x_points_down = bool(self.get_parameter('auto_flip_pose_z_180_if_x_points_down').value)
        self.x_axis_downward_flip_threshold = float(self.get_parameter('x_axis_downward_flip_threshold').value)

        # ------------------------------------------------------------
        # Internal cache for marker redraw
        # ------------------------------------------------------------
        self.last_pred_pose_base = None
        self.last_contact_point_base = None
        self.last_object_center_base = None

        # ------------------------------------------------------------
        # TF2
        # ------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------
        self.sub_grasp_pose = self.create_subscription(
            PoseStamped, self.grasp_pose_input_topic, self.grasp_pose_callback, 10
        )
        self.sub_contact_point = self.create_subscription(
            PointStamped, self.contact_point_input_topic, self.contact_point_callback, 10
        )
        self.sub_object_center = self.create_subscription(
            PointStamped, self.object_center_input_topic, self.object_center_callback, 10
        )
        self.sub_target_pc = self.create_subscription(
            PointCloud2, self.target_pc_input_topic, self.target_pc_callback, 10
        )

        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, self.grasp_pose_output_topic, 10)
        self.pub_contact_point_base = self.create_publisher(PointStamped, self.contact_point_output_topic, 10)
        self.pub_object_center_base = self.create_publisher(PointStamped, self.object_center_output_topic, 10)
        self.pub_target_pc_base = self.create_publisher(PointCloud2, self.target_pc_output_topic, 10)

        self.pub_debug_axes_base = None
        if self.publish_debug_axes_base:
            self.pub_debug_axes_base = self.create_publisher(MarkerArray, self.debug_axes_base_topic, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('AnyGrasp Base Transformer Initialized (BASE FRAME + POINT CLOUD)')
        self.get_logger().info(f'grasp_pose_input_topic        : {self.grasp_pose_input_topic}')
        self.get_logger().info(f'contact_point_input_topic     : {self.contact_point_input_topic}')
        self.get_logger().info(f'object_center_input_topic     : {self.object_center_input_topic}')
        self.get_logger().info(f'target_pc_input_topic         : {self.target_pc_input_topic}')
        self.get_logger().info(f'grasp_pose_output_topic       : {self.grasp_pose_output_topic}')
        self.get_logger().info(f'contact_point_output_topic    : {self.contact_point_output_topic}')
        self.get_logger().info(f'object_center_output_topic    : {self.object_center_output_topic}')
        self.get_logger().info(f'target_pc_output_topic        : {self.target_pc_output_topic}')
        self.get_logger().info(f'debug_axes_base_topic         : {self.debug_axes_base_topic}')
        self.get_logger().info(f'camera_frame                  : {self.camera_frame}')
        self.get_logger().info(f'base_frame                    : {self.base_frame}')
        self.get_logger().info(f'gripper_frame                 : {self.gripper_frame}')
        self.get_logger().info(f'T_cam_to_link7               :\n{self.T_cam_to_link7}')
        self.get_logger().info(f'T_link7_to_gripper_base      :\n{self.T_link7_to_gripper_base}')
        self.get_logger().info(f'T_cam_to_gripper_base        :\n{self.T_cam_to_gripper}')
        self.get_logger().info(f'T_pose_align_y90             :\n{self.T_pose_align_y90}')
        self.get_logger().info(f'T_pose_align_z180            :\n{self.T_pose_align_z180}')
        self.get_logger().info('RViz: Fixed Frame = base_link')
        self.get_logger().info('========================================')

    # ============================================================
    # Callbacks
    # ============================================================
    def contact_point_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'ANYGRASP_CONTACT')
        if out is None:
            return

        self.pub_contact_point_base.publish(out)
        self.last_contact_point_base = out

        if self.log_contact_point:
            self.log_point_compact('ANYGRASP_CONTACT_BASE', out, input_frame=msg.header.frame_id)

        self.publish_debug_markers(out.header.stamp)

    def object_center_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'SAM3_OBJECT_CENTER')
        if out is None:
            return

        self.pub_object_center_base.publish(out)
        self.last_object_center_base = out
        self.log_point_compact('SAM3_OBJECT_CENTER_BASE', out, input_frame=msg.header.frame_id)

        self.publish_debug_markers(out.header.stamp)

    def grasp_pose_callback(self, msg: PoseStamped):
        result = self.transform_pose(msg, 'ANYGRASP_POSE')
        if result is None:
            return

        pose_base_msg, T_final_base = result
        self.pub_grasp_pose_base.publish(pose_base_msg)
        self.last_pred_pose_base = pose_base_msg

        if self.log_pose:
            self.log_pose_compact('ANYGRASP_POSE_BASE', pose_base_msg, input_frame=msg.header.frame_id)
            self.log_pose_axes_compact('ANYGRASP_POSE_BASE_AXES', T_final_base)

        self.publish_debug_markers(msg.header.stamp)

    def target_pc_callback(self, msg: PointCloud2):
        out = self.transform_pointcloud(msg, 'YOLO_TARGET_PC')
        if out is None:
            return

        self.pub_target_pc_base.publish(out)

        if self.log_pointcloud:
            self.get_logger().info(
                f'[YOLO_TARGET_PC_BASE] frame={out.header.frame_id} '
                f'width={out.width} height={out.height} '
                f'input_frame={msg.header.frame_id}'
            )

    # ============================================================
    # Main transforms
    # ============================================================
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
            T_raw_base, T_intermediate, path_label = self.transform_pose_to_base_matrix(msg, source_name)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pose transform failed: {repr(e)}')
            return None

        T_final_base = T_raw_base.copy()
        flip_applied = False
        x_axis_after_y90 = None

        if self.apply_anygrasp_pose_frame_alignment:
            T_final_base = T_final_base @ self.T_pose_align_y90
            path_label += '->pose_align(Ry+90_local)'

            x_axis_after_y90 = T_final_base[:3, 0].copy()
            if self.auto_flip_pose_z_180_if_x_points_down:
                if x_axis_after_y90[2] < self.x_axis_downward_flip_threshold:
                    T_final_base = T_final_base @ self.T_pose_align_z180
                    path_label += '->auto_flip(Rz+180_local_if_x_down)'
                    flip_applied = True
                else:
                    path_label += '->auto_flip(skip_x_not_down)'

        if self.apply_grasp_tool_offset:
            T_final_base = T_final_base @ self.T_grasp_to_tool
            path_label += '->tool_offset(Rx180_about_local_x)'

        final_msg = PoseStamped()
        final_msg.header.stamp = msg.header.stamp
        final_msg.header.frame_id = self.base_frame
        final_msg.pose = self.matrix_to_pose(T_final_base)

        if self.verbose_debug:
            in_p = msg.pose.position
            in_q = msg.pose.orientation
            mid_q = R.from_matrix(T_intermediate[:3, :3]).as_quat()
            final_p = final_msg.pose.position
            final_q = final_msg.pose.orientation

            self.get_logger().info(
                '\n'
                f'[{source_name}]\n'
                f'Input frame  : {msg.header.frame_id}\n'
                f'Path         : {path_label}\n'
                f'Input xyz    : [{in_p.x:.4f}, {in_p.y:.4f}, {in_p.z:.4f}]\n'
                f'Input quat   : [{in_q.x:.4f}, {in_q.y:.4f}, {in_q.z:.4f}, {in_q.w:.4f}]\n'
                f'Intermed xyz : [{T_intermediate[0,3]:.4f}, {T_intermediate[1,3]:.4f}, {T_intermediate[2,3]:.4f}]\n'
                f'Intermed quat: [{mid_q[0]:.4f}, {mid_q[1]:.4f}, {mid_q[2]:.4f}, {mid_q[3]:.4f}]\n'
                f'Final xyz    : [{final_p.x:.4f}, {final_p.y:.4f}, {final_p.z:.4f}]\n'
                f'Final quat   : [{final_q.x:.4f}, {final_q.y:.4f}, {final_q.z:.4f}, {final_q.w:.4f}]'
            )
        return final_msg, T_final_base

    def transform_pointcloud(self, msg: PointCloud2, source_name='POINTCLOUD'):
        frame_id = msg.header.frame_id.strip()
        if frame_id == self.base_frame:
            out = PointCloud2()
            out = msg
            return out

        try:
            if self.use_direct_camera_tf and frame_id not in ('', self.gripper_frame):
                resolved_base, t_direct = self.lookup_base_from_source_frame(frame_id, msg.header.stamp, source_name)
                if t_direct is not None:
                    T_source_to_base = self.make_transform_matrix(t_direct)
                    return self.apply_transform_to_cloud(msg, T_source_to_base, self.base_frame)

            resolved_base, t = self.resolve_base_frame(msg.header.stamp, source_name)
            if t is None:
                raise RuntimeError('base<-gripper TF unavailable')

            T_gripper_to_base = self.make_transform_matrix(t)

            if frame_id == self.gripper_frame:
                return self.apply_transform_to_cloud(msg, T_gripper_to_base, self.base_frame)

            # camera-like cloud -> gripper base
            T_cam_to_base = T_gripper_to_base @ self.T_cam_to_gripper
            return self.apply_transform_to_cloud(msg, T_cam_to_base, self.base_frame)

        except Exception as e:
            self.get_logger().error(f'[{source_name}] point cloud transform failed: {repr(e)}')
            return None

    def apply_transform_to_cloud(self, cloud_msg: PointCloud2, T: np.ndarray, out_frame: str) -> PointCloud2:
        """
        Generic but debug-oriented PointCloud2 transform.
        Preserves all fields and only changes x,y,z.
        """
        field_names = [f.name for f in cloud_msg.fields]
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            raise RuntimeError('PointCloud2 does not contain x/y/z fields')

        idx_x = field_names.index('x')
        idx_y = field_names.index('y')
        idx_z = field_names.index('z')

        rows = []
        for p in pc2.read_points(cloud_msg, field_names=field_names, skip_nans=False):
            p_list = list(p)

            x = p_list[idx_x]
            y = p_list[idx_y]
            z = p_list[idx_z]

            if any(math.isnan(v) for v in [x, y, z]):
                rows.append(tuple(p_list))
                continue

            vec = np.array([x, y, z, 1.0], dtype=np.float64)
            vec_t = T @ vec

            p_list[idx_x] = float(vec_t[0])
            p_list[idx_y] = float(vec_t[1])
            p_list[idx_z] = float(vec_t[2])
            rows.append(tuple(p_list))

        header = cloud_msg.header
        header.frame_id = out_frame

        out_cloud = pc2.create_cloud(header, cloud_msg.fields, rows)
        out_cloud.height = cloud_msg.height
        out_cloud.width = cloud_msg.width
        out_cloud.is_bigendian = cloud_msg.is_bigendian
        out_cloud.is_dense = cloud_msg.is_dense
        return out_cloud

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
            return p_base, p_in[:3].copy(), f'gripper_base->{resolved_base}'

        p_gripper = self.T_cam_to_gripper @ p_in
        p_base = T_gripper_to_base @ p_gripper
        return p_base, p_gripper[:3].copy(), f'{frame_id}(camera-like)->gripper_base->{resolved_base}'

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
                path_label = f'{frame_id}(camera-like)->gripper_base->{resolved_base}'
        else:
            resolved_base, t = self.resolve_base_frame(msg.header.stamp, source_name)
            if t is None:
                raise RuntimeError('base<-gripper TF unavailable')

            T_gripper_to_base = self.make_transform_matrix(t)

            if frame_id == self.gripper_frame:
                T_pose_base = T_gripper_to_base @ T_pose_in
                T_intermediate = T_pose_in.copy()
                path_label = f'gripper_base->{resolved_base}'
            else:
                T_pose_gripper = self.T_cam_to_gripper @ T_pose_in
                T_pose_base = T_gripper_to_base @ T_pose_gripper
                T_intermediate = T_pose_gripper
                path_label = f'{frame_id}(camera-like)->gripper_base->{resolved_base}'

        return T_pose_base, T_intermediate, path_label

    # ============================================================
    # Debug markers in base frame
    # ============================================================
    def publish_debug_markers(self, stamp):
        if self.pub_debug_axes_base is None:
            return

        header_frame = self.base_frame
        ma = MarkerArray()
        ma.markers.append(self.make_delete_all_marker(header_frame, stamp))

        # base axes
        ma.markers.extend(
            self.make_axes_triplet(
                header_frame=header_frame,
                stamp=stamp,
                origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                rotation=np.eye(3, dtype=np.float64),
                ns_prefix='base_axes',
                ids=(0, 1, 2),
                colors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.4, 1.0)),
            )
        )

        # predicted grasp axes
        if self.last_pred_pose_base is not None:
            T_pred = self.pose_to_matrix(
                self.last_pred_pose_base.pose.position,
                self.last_pred_pose_base.pose.orientation
            )
            ma.markers.extend(
                self.make_axes_triplet(
                    header_frame=header_frame,
                    stamp=stamp,
                    origin=T_pred[:3, 3],
                    rotation=T_pred[:3, :3],
                    ns_prefix='predicted_grasp_axes',
                    ids=(10, 11, 12),
                    colors=((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.6, 1.0)),
                )
            )

        # current gripper axes
        if self.show_current_gripper_axes:
            try:
                T_gripper_to_base = self.lookup_transform_matrix(
                    target_frame=self.base_frame,
                    source_frame=self.gripper_frame,
                    stamp=stamp,
                    source_name='DEBUG_CURRENT_GRIPPER_IN_BASE'
                )
                ma.markers.extend(
                    self.make_axes_triplet(
                        header_frame=header_frame,
                        stamp=stamp,
                        origin=T_gripper_to_base[:3, 3],
                        rotation=T_gripper_to_base[:3, :3],
                        ns_prefix='current_gripper_axes',
                        ids=(20, 21, 22),
                        colors=((1.0, 1.0, 0.0), (0.7, 1.0, 0.0), (0.0, 1.0, 1.0)),
                    )
                )
            except Exception as e:
                self.get_logger().warn(f'[DEBUG] failed current gripper marker publish: {repr(e)}')

        # contact point sphere
        if self.show_contact_marker and self.last_contact_point_base is not None:
            cp = self.last_contact_point_base.point
            ma.markers.append(
                self.make_sphere_marker(
                    header_frame=header_frame,
                    stamp=stamp,
                    marker_id=30,
                    ns='contact_point',
                    xyz=np.array([cp.x, cp.y, cp.z], dtype=np.float64),
                    radius=self.contact_marker_radius,
                    rgba=(1.0, 0.0, 1.0, 0.95),
                )
            )

        # object center sphere
        if self.show_object_center_marker and self.last_object_center_base is not None:
            oc = self.last_object_center_base.point
            ma.markers.append(
                self.make_sphere_marker(
                    header_frame=header_frame,
                    stamp=stamp,
                    marker_id=31,
                    ns='object_center',
                    xyz=np.array([oc.x, oc.y, oc.z], dtype=np.float64),
                    radius=self.object_center_marker_radius,
                    rgba=(1.0, 0.5, 0.0, 0.95),
                )
            )

        self.pub_debug_axes_base.publish(ma)

    # ============================================================
    # TF helpers
    # ============================================================
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

    def lookup_transform_matrix(self, target_frame: str, source_frame: str, stamp, source_name='TF_MATRIX') -> np.ndarray:
        t = self.lookup_transform_with_fallback(target_frame, source_frame, stamp, source_name)
        if t is None:
            raise RuntimeError(f'TF unavailable: {target_frame} <- {source_frame}')
        return self.make_transform_matrix(t)

    # ============================================================
    # Logging
    # ============================================================
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

    def log_pose_axes_compact(self, tag: str, T: np.ndarray):
        x_axis = T[:3, 0]
        y_axis = T[:3, 1]
        z_axis = T[:3, 2]
        self.get_logger().info(
            f'[{tag}] '
            f'x=({x_axis[0]:.4f}, {x_axis[1]:.4f}, {x_axis[2]:.4f}) '
            f'y=({y_axis[0]:.4f}, {y_axis[1]:.4f}, {y_axis[2]:.4f}) '
            f'z=({z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f})'
        )

    # ============================================================
    # Marker helpers
    # ============================================================
    def make_delete_all_marker(self, header_frame: str, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = header_frame
        m.header.stamp = stamp
        m.action = Marker.DELETEALL
        return m

    def make_axes_triplet(self, header_frame, stamp, origin, rotation, ns_prefix, ids, colors):
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]

        return [
            self.make_axis_arrow(header_frame, stamp, ids[0], f'{ns_prefix}_x', origin, x_axis, colors[0]),
            self.make_axis_arrow(header_frame, stamp, ids[1], f'{ns_prefix}_y', origin, y_axis, colors[1]),
            self.make_axis_arrow(header_frame, stamp, ids[2], f'{ns_prefix}_z', origin, z_axis, colors[2]),
        ]

    def make_axis_arrow(self, header_frame, stamp, marker_id, ns, origin, axis_dir, rgb):
        m = Marker()
        m.header.frame_id = header_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.ARROW
        m.action = Marker.ADD

        axis_dir = np.asarray(axis_dir, dtype=np.float64)
        n = np.linalg.norm(axis_dir)
        if n < 1e-12:
            axis_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis_dir = axis_dir / n

        p0 = Point()
        p0.x = float(origin[0])
        p0.y = float(origin[1])
        p0.z = float(origin[2])

        p1 = Point()
        p1.x = float(origin[0] + self.axes_length * axis_dir[0])
        p1.y = float(origin[1] + self.axes_length * axis_dir[1])
        p1.z = float(origin[2] + self.axes_length * axis_dir[2])

        m.points = [p0, p1]
        m.scale.x = float(self.axes_shaft_diameter)
        m.scale.y = float(self.axes_head_diameter)
        m.scale.z = float(self.axes_head_length)

        m.color = ColorRGBA(
            r=float(rgb[0]),
            g=float(rgb[1]),
            b=float(rgb[2]),
            a=1.0
        )

        if self.axes_lifetime_sec > 0.0:
            m.lifetime = Duration(seconds=self.axes_lifetime_sec).to_msg()

        return m

    def make_sphere_marker(self, header_frame, stamp, marker_id, ns, xyz, radius, rgba):
        m = Marker()
        m.header.frame_id = header_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0

        m.scale.x = float(radius)
        m.scale.y = float(radius)
        m.scale.z = float(radius)

        m.color = ColorRGBA(
            r=float(rgba[0]),
            g=float(rgba[1]),
            b=float(rgba[2]),
            a=float(rgba[3]),
        )

        if self.axes_lifetime_sec > 0.0:
            m.lifetime = Duration(seconds=self.axes_lifetime_sec).to_msg()

        return m

    # ============================================================
    # Math / pose helpers
    # ============================================================
    @staticmethod
    def pose_to_matrix(position, orientation) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        q = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float64)
        return T

    @staticmethod
    def matrix_to_pose(T: np.ndarray) -> Pose:
        q = R.from_matrix(T[:3, :3]).as_quat()
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose

    @staticmethod
    def make_transform_matrix(t) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        tr = t.transform.translation
        qr = t.transform.rotation
        q = np.array([qr.x, qr.y, qr.z, qr.w], dtype=np.float64)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        return T


def main(args=None):
    rclpy.init(args=args)
    node = AnyGraspBaseTransformerRight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()