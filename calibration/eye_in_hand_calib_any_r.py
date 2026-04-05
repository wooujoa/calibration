#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
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
      5) Publish RViz markers in base_link for:
         - grasp contact point
         - grasp pose as a parallel-jaw gripper shape

    RViz:
      - Fixed Frame: base_link
      - Marker: /anygrasp/best_contact_marker
      - Marker: /anygrasp/best_pose_marker
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

        # RViz marker topics
        self.declare_parameter('best_pose_marker_topic', '/anygrasp/best_pose_marker')
        self.declare_parameter('best_contact_marker_topic', '/anygrasp/best_contact_marker')

        # marker style
        self.declare_parameter('contact_marker_radius', 0.018)
        self.declare_parameter('axes_lifetime_sec', 0.0)

        self.declare_parameter('publish_best_pose_marker', True)
        self.declare_parameter('publish_best_contact_marker', True)

        self.declare_parameter('gripper_marker_line_width', 0.006)
        self.declare_parameter('gripper_opening', 0.10)          # jaw opening along local y
        self.declare_parameter('gripper_finger_length', 0.060)   # along local x
        self.declare_parameter('gripper_finger_depth', 0.020)    # along local z
        self.declare_parameter('gripper_palm_depth', 0.030)      # backward along -z
        self.declare_parameter('gripper_wrist_length', 0.020)    # further backward along -z
        self.declare_parameter('gripper_marker_rgba', [1.0, 1.0, 1.0, 1.0])

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

        self.best_pose_marker_topic = self.get_parameter('best_pose_marker_topic').value
        self.best_contact_marker_topic = self.get_parameter('best_contact_marker_topic').value

        self.contact_marker_radius = float(self.get_parameter('contact_marker_radius').value)
        self.axes_lifetime_sec = float(self.get_parameter('axes_lifetime_sec').value)

        self.apply_grasp_tool_offset = bool(self.get_parameter('apply_grasp_tool_offset').value)
        self.apply_anygrasp_pose_frame_alignment = bool(self.get_parameter('apply_anygrasp_pose_frame_alignment').value)
        self.auto_flip_pose_z_180_if_x_points_down = bool(self.get_parameter('auto_flip_pose_z_180_if_x_points_down').value)
        self.x_axis_downward_flip_threshold = float(self.get_parameter('x_axis_downward_flip_threshold').value)

        self.publish_best_pose_marker_enabled = bool(self.get_parameter('publish_best_pose_marker').value)
        self.publish_best_contact_marker_enabled = bool(self.get_parameter('publish_best_contact_marker').value)

        self.gripper_marker_line_width = float(self.get_parameter('gripper_marker_line_width').value)
        self.gripper_opening = float(self.get_parameter('gripper_opening').value)
        self.gripper_finger_length = float(self.get_parameter('gripper_finger_length').value)
        self.gripper_finger_depth = float(self.get_parameter('gripper_finger_depth').value)
        self.gripper_palm_depth = float(self.get_parameter('gripper_palm_depth').value)
        self.gripper_wrist_length = float(self.get_parameter('gripper_wrist_length').value)
        self.gripper_marker_rgba = [float(v) for v in self.get_parameter('gripper_marker_rgba').value]

        # ------------------------------------------------------------
        # Internal cache
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

        self.pub_best_pose_marker = self.create_publisher(Marker, self.best_pose_marker_topic, 10)
        self.pub_best_contact_marker = self.create_publisher(Marker, self.best_contact_marker_topic, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('AnyGrasp Base Transformer Initialized')
        self.get_logger().info(f'grasp_pose_input_topic        : {self.grasp_pose_input_topic}')
        self.get_logger().info(f'contact_point_input_topic     : {self.contact_point_input_topic}')
        self.get_logger().info(f'object_center_input_topic     : {self.object_center_input_topic}')
        self.get_logger().info(f'target_pc_input_topic         : {self.target_pc_input_topic}')
        self.get_logger().info(f'grasp_pose_output_topic       : {self.grasp_pose_output_topic}')
        self.get_logger().info(f'contact_point_output_topic    : {self.contact_point_output_topic}')
        self.get_logger().info(f'object_center_output_topic    : {self.object_center_output_topic}')
        self.get_logger().info(f'target_pc_output_topic        : {self.target_pc_output_topic}')
        self.get_logger().info(f'best_pose_marker_topic        : {self.best_pose_marker_topic}')
        self.get_logger().info(f'best_contact_marker_topic     : {self.best_contact_marker_topic}')
        self.get_logger().info(f'camera_frame                  : {self.camera_frame}')
        self.get_logger().info(f'base_frame                    : {self.base_frame}')
        self.get_logger().info(f'gripper_frame                 : {self.gripper_frame}')
        self.get_logger().info(f'T_cam_to_link7               :\n{self.T_cam_to_link7}')
        self.get_logger().info(f'T_link7_to_gripper_base      :\n{self.T_link7_to_gripper_base}')
        self.get_logger().info(f'T_cam_to_gripper_base        :\n{self.T_cam_to_gripper}')
        self.get_logger().info(f'T_pose_align_y90             :\n{self.T_pose_align_y90}')
        self.get_logger().info(f'T_pose_align_z180            :\n{self.T_pose_align_z180}')
        self.get_logger().info('RViz Fixed Frame = base_link')
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

        self.publish_best_contact_point_marker(out)

    def object_center_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'SAM3_OBJECT_CENTER')
        if out is None:
            return

        self.pub_object_center_base.publish(out)
        self.last_object_center_base = out
        self.log_point_compact('SAM3_OBJECT_CENTER_BASE', out, input_frame=msg.header.frame_id)

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

        self.publish_best_gripper_pose_marker(msg.header.stamp, T_final_base)

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

        if self.apply_anygrasp_pose_frame_alignment:
            T_final_base = T_final_base @ self.T_pose_align_y90
            path_label += '->pose_align(Ry+90_local)'

            x_axis_after_y90 = T_final_base[:3, 0].copy()
            if self.auto_flip_pose_z_180_if_x_points_down:
                if x_axis_after_y90[2] < self.x_axis_downward_flip_threshold:
                    T_final_base = T_final_base @ self.T_pose_align_z180
                    path_label += '->auto_flip(Rz+180_local_if_x_down)'
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
    # RViz markers: contact point + corrected gripper shape
    # ============================================================
    def publish_best_contact_point_marker(self, msg: PointStamped):
        if not self.publish_best_contact_marker_enabled:
            return

        p = msg.point
        marker = self.make_sphere_marker(
            header_frame=self.base_frame,
            stamp=msg.header.stamp,
            marker_id=0,
            ns='best_contact_point',
            xyz=np.array([p.x, p.y, p.z], dtype=np.float64),
            radius=self.contact_marker_radius,
            rgba=(1.0, 0.0, 1.0, 0.98),
        )
        self.pub_best_contact_marker.publish(marker)

    def publish_best_gripper_pose_marker(self, stamp, T_grasp_base: np.ndarray):
        if not self.publish_best_pose_marker_enabled:
            return

        marker = self.make_parallel_jaw_gripper_marker(
            header_frame=self.base_frame,
            stamp=stamp,
            marker_id=0,
            ns='best_pose_gripper',
            T=T_grasp_base,
        )
        self.pub_best_pose_marker.publish(marker)

    def make_parallel_jaw_gripper_marker(self, header_frame, stamp, marker_id, ns, T: np.ndarray) -> Marker:
        """
        Corrected local-axis interpretation:
          - local x : finger length direction
          - local y : jaw opening direction
          - local z : approach direction (toward object)

        The marker origin is treated as the grasp center.
        So the palm/wrist are drawn backward along -z.
        """
        m = Marker()
        m.header.frame_id = header_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = float(self.gripper_marker_line_width)

        rgba = self.gripper_marker_rgba
        m.color = ColorRGBA(
            r=float(rgba[0]),
            g=float(rgba[1]),
            b=float(rgba[2]),
            a=float(rgba[3]),
        )

        if self.axes_lifetime_sec > 0.0:
            m.lifetime = Duration(seconds=self.axes_lifetime_sec).to_msg()

        opening = max(0.01, float(self.gripper_opening))
        finger_len = max(0.001, float(self.gripper_finger_length))
        finger_depth = max(0.001, float(self.gripper_finger_depth))
        palm_depth = max(0.0, float(self.gripper_palm_depth))
        wrist_len = max(0.0, float(self.gripper_wrist_length))

        # Local geometry
        # origin = grasp center
        # x: finger length
        # y: opening
        # z: approach toward object
        x0 = -0.5 * finger_len
        x1 =  0.5 * finger_len
        yL = -0.5 * opening
        yR =  0.5 * opening

        z_front = 0.0
        z_finger_back = -finger_depth
        z_palm = -(finger_depth + palm_depth)
        z_wrist = z_palm - wrist_len

        segs_local = [
            # left finger rectangle in x-z plane at y = yL
            ([x0, yL, z_front], [x1, yL, z_front]),
            ([x0, yL, z_finger_back], [x1, yL, z_finger_back]),
            ([x0, yL, z_front], [x0, yL, z_finger_back]),
            ([x1, yL, z_front], [x1, yL, z_finger_back]),

            # right finger rectangle in x-z plane at y = yR
            ([x0, yR, z_front], [x1, yR, z_front]),
            ([x0, yR, z_finger_back], [x1, yR, z_finger_back]),
            ([x0, yR, z_front], [x0, yR, z_finger_back]),
            ([x1, yR, z_front], [x1, yR, z_finger_back]),

            # supports from finger backs to palm
            ([0.0, yL, z_finger_back], [0.0, yL, z_palm]),
            ([0.0, yR, z_finger_back], [0.0, yR, z_palm]),

            # palm crossbar
            ([0.0, yL, z_palm], [0.0, yR, z_palm]),

            # wrist
            ([0.0, 0.0, z_palm], [0.0, 0.0, z_wrist]),
        ]

        Rb = T[:3, :3]
        tb = T[:3, 3]

        pts = []
        for a_local, b_local in segs_local:
            a_local = np.asarray(a_local, dtype=np.float64)
            b_local = np.asarray(b_local, dtype=np.float64)

            a_base = Rb @ a_local + tb
            b_base = Rb @ b_local + tb

            pa = Point()
            pa.x = float(a_base[0])
            pa.y = float(a_base[1])
            pa.z = float(a_base[2])

            pb = Point()
            pb.x = float(b_base[0])
            pb.y = float(b_base[1])
            pb.z = float(b_base[2])

            pts.append(pa)
            pts.append(pb)

        m.points = pts
        return m

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