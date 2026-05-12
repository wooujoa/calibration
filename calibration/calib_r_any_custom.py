#!/usr/bin/env python3
import math
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, PoseArray
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros

from grasp_msgs.msg import ObjectGrasp


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
      6) Re-publish visualization-only topics in base_link
         without changing any existing calibration / pose alignment matrices
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
        # This affects ONLY final grasp pose output / custom best-pose marker.
        # Visualization-only republish does NOT touch these pose-align matrices.
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

        # Final planning message. This is the ONLY place where ObjectGrasp is published.
        # MASTER should pass the selected arm as a parameter when launching this right/left pipeline.
        self.declare_parameter('final_object_grasp_topic', '/manipulator/object_grasp_r')
        self.declare_parameter('selected_arm_id', 2)  # 0 unknown, 1 left, 2 right
        self.declare_parameter('final_label', 'target_object')
        self.declare_parameter('best_width_input_topic', '/anygrasp/best_width')
        self.declare_parameter('best_score_input_topic', '/anygrasp/best_score')
        self.declare_parameter('require_matching_stamps', True)
        self.declare_parameter('publish_once_per_stamp', True)
        self.declare_parameter('object_size_min_points', 20)

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
        self.declare_parameter('log_visual_republish', False)

        # RViz marker topics
        self.declare_parameter('best_pose_marker_topic', '/anygrasp/best_pose_marker_base_r')
        self.declare_parameter('best_contact_marker_topic', '/anygrasp/best_contact_marker_base_r')

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
        # Visualization-only republish in base_link
        # ------------------------------------------------------------
        self.declare_parameter('republish_visuals_in_base_link', True)

        self.declare_parameter('raw_grasps_input_topic', '/anygrasp/grasps')
        self.declare_parameter('raw_grasps_output_topic', '/anygrasp/grasps_base_r')

        self.declare_parameter('raw_grasp_markers_input_topic', '/anygrasp/grasp_markers')
        self.declare_parameter('raw_grasp_markers_output_topic', '/anygrasp/grasp_markers_base_r')

        self.declare_parameter('raw_all_grasp_markers_input_topic', '/anygrasp/all_grasp_markers')
        self.declare_parameter('raw_all_grasp_markers_output_topic', '/anygrasp/all_grasp_markers_base_r')

        self.declare_parameter('raw_best_axes_markers_input_topic', '/anygrasp/best_axes_markers')
        self.declare_parameter('raw_best_axes_markers_output_topic', '/anygrasp/best_axes_markers_base_r')

        self.declare_parameter('object_pc_input_topic', '/yolo/object_pc')
        self.declare_parameter('object_pc_output_topic', '/yolo/object_pc_base')

        self.declare_parameter('background_pc_input_topic', '/yolo/background_pc')
        self.declare_parameter('background_pc_output_topic', '/yolo/background_pc_base')

        self.declare_parameter('preview_cloud_input_topic', '/sam3/mask_pointcloud')
        self.declare_parameter('preview_cloud_output_topic', '/sam3/mask_pointcloud_base')

        self.declare_parameter('full_scene_pc_input_topic', '/sam3/full_scene_pc')
        self.declare_parameter('full_scene_pc_output_topic', '/sam3/full_scene_pc_base')

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

        self.final_object_grasp_topic = self.get_parameter('final_object_grasp_topic').value
        self.selected_arm_id = int(self.get_parameter('selected_arm_id').value)
        self.final_label = self.get_parameter('final_label').value
        self.best_width_input_topic = self.get_parameter('best_width_input_topic').value
        self.best_score_input_topic = self.get_parameter('best_score_input_topic').value
        self.require_matching_stamps = bool(self.get_parameter('require_matching_stamps').value)
        self.publish_once_per_stamp = bool(self.get_parameter('publish_once_per_stamp').value)
        self.object_size_min_points = int(self.get_parameter('object_size_min_points').value)

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
        self.log_visual_republish = bool(self.get_parameter('log_visual_republish').value)

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

        self.republish_visuals_in_base_link = bool(self.get_parameter('republish_visuals_in_base_link').value)

        self.raw_grasps_input_topic = self.get_parameter('raw_grasps_input_topic').value
        self.raw_grasps_output_topic = self.get_parameter('raw_grasps_output_topic').value

        self.raw_grasp_markers_input_topic = self.get_parameter('raw_grasp_markers_input_topic').value
        self.raw_grasp_markers_output_topic = self.get_parameter('raw_grasp_markers_output_topic').value

        self.raw_all_grasp_markers_input_topic = self.get_parameter('raw_all_grasp_markers_input_topic').value
        self.raw_all_grasp_markers_output_topic = self.get_parameter('raw_all_grasp_markers_output_topic').value

        self.raw_best_axes_markers_input_topic = self.get_parameter('raw_best_axes_markers_input_topic').value
        self.raw_best_axes_markers_output_topic = self.get_parameter('raw_best_axes_markers_output_topic').value

        self.object_pc_input_topic_vis = self.get_parameter('object_pc_input_topic').value
        self.object_pc_output_topic_vis = self.get_parameter('object_pc_output_topic').value

        self.background_pc_input_topic_vis = self.get_parameter('background_pc_input_topic').value
        self.background_pc_output_topic_vis = self.get_parameter('background_pc_output_topic').value

        self.preview_cloud_input_topic_vis = self.get_parameter('preview_cloud_input_topic').value
        self.preview_cloud_output_topic_vis = self.get_parameter('preview_cloud_output_topic').value

        self.full_scene_pc_input_topic_vis = self.get_parameter('full_scene_pc_input_topic').value
        self.full_scene_pc_output_topic_vis = self.get_parameter('full_scene_pc_output_topic').value

        # ------------------------------------------------------------
        # Internal cache
        # ------------------------------------------------------------
        self.last_pred_pose_base = None
        self.last_contact_point_base = None
        self.last_object_center_base = None
        self.last_grasp_width = None
        self.last_grasp_confidence = None
        self.last_object_size_base = None
        self.last_object_size_stamp_ns = None
        self.last_final_publish_stamp_ns = None

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
        self.sub_best_width = self.create_subscription(
            Float32, self.best_width_input_topic, self.best_width_callback, 10
        )
        self.sub_best_score = self.create_subscription(
            Float32, self.best_score_input_topic, self.best_score_callback, 10
        )
        # Always subscribe to object_pc for final object_size estimation.
        # This is separate from optional RViz republishing so final ObjectGrasp does not depend on debug settings.
        self.sub_object_pc_size = self.create_subscription(
            PointCloud2, self.get_parameter('object_pc_input_topic').value, self.object_pc_size_callback, 10
        )

        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, self.grasp_pose_output_topic, 10)
        self.pub_contact_point_base = self.create_publisher(PointStamped, self.contact_point_output_topic, 10)
        self.pub_object_center_base = self.create_publisher(PointStamped, self.object_center_output_topic, 10)
        self.pub_target_pc_base = self.create_publisher(PointCloud2, self.target_pc_output_topic, 10)
        self.pub_object_grasp_final = self.create_publisher(ObjectGrasp, self.final_object_grasp_topic, 10)

        self.pub_best_pose_marker = self.create_publisher(Marker, self.best_pose_marker_topic, 10)
        self.pub_best_contact_marker = self.create_publisher(Marker, self.best_contact_marker_topic, 10)

        # ------------------------------------------------------------
        # Visualization-only republish I/O
        # ------------------------------------------------------------
        self.sub_raw_grasps = None
        self.sub_raw_grasp_markers = None
        self.sub_raw_all_grasp_markers = None
        self.sub_raw_best_axes_markers = None
        self.sub_object_pc_vis = None
        self.sub_background_pc_vis = None
        self.sub_preview_cloud_vis = None
        self.sub_full_scene_pc_vis = None

        self.pub_raw_grasps_base = None
        self.pub_raw_grasp_markers_base = None
        self.pub_raw_all_grasp_markers_base = None
        self.pub_raw_best_axes_markers_base = None
        self.pub_object_pc_base_vis = None
        self.pub_background_pc_base_vis = None
        self.pub_preview_cloud_base_vis = None
        self.pub_full_scene_pc_base_vis = None

        if self.republish_visuals_in_base_link:
            self.sub_raw_grasps = self.create_subscription(
                PoseArray, self.raw_grasps_input_topic, self.raw_grasps_callback, 10
            )
            self.sub_raw_grasp_markers = self.create_subscription(
                MarkerArray, self.raw_grasp_markers_input_topic, self.raw_grasp_markers_callback, 10
            )
            self.sub_raw_all_grasp_markers = self.create_subscription(
                MarkerArray, self.raw_all_grasp_markers_input_topic, self.raw_all_grasp_markers_callback, 10
            )
            self.sub_raw_best_axes_markers = self.create_subscription(
                MarkerArray, self.raw_best_axes_markers_input_topic, self.raw_best_axes_markers_callback, 10
            )

            self.sub_object_pc_vis = self.create_subscription(
                PointCloud2, self.object_pc_input_topic_vis, self.object_pc_callback_vis, 10
            )
            self.sub_background_pc_vis = self.create_subscription(
                PointCloud2, self.background_pc_input_topic_vis, self.background_pc_callback_vis, 10
            )
            self.sub_preview_cloud_vis = self.create_subscription(
                PointCloud2, self.preview_cloud_input_topic_vis, self.preview_cloud_callback_vis, 10
            )
            self.sub_full_scene_pc_vis = self.create_subscription(
                PointCloud2, self.full_scene_pc_input_topic_vis, self.full_scene_pc_callback_vis, 10
            )

            self.pub_raw_grasps_base = self.create_publisher(PoseArray, self.raw_grasps_output_topic, 10)
            self.pub_raw_grasp_markers_base = self.create_publisher(MarkerArray, self.raw_grasp_markers_output_topic, 10)
            self.pub_raw_all_grasp_markers_base = self.create_publisher(MarkerArray, self.raw_all_grasp_markers_output_topic, 10)
            self.pub_raw_best_axes_markers_base = self.create_publisher(MarkerArray, self.raw_best_axes_markers_output_topic, 10)

            self.pub_object_pc_base_vis = self.create_publisher(PointCloud2, self.object_pc_output_topic_vis, 10)
            self.pub_background_pc_base_vis = self.create_publisher(PointCloud2, self.background_pc_output_topic_vis, 10)
            self.pub_preview_cloud_base_vis = self.create_publisher(PointCloud2, self.preview_cloud_output_topic_vis, 10)
            self.pub_full_scene_pc_base_vis = self.create_publisher(PointCloud2, self.full_scene_pc_output_topic_vis, 10)

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
        self.get_logger().info(f'best_width_input_topic       : {self.best_width_input_topic}')
        self.get_logger().info(f'best_score_input_topic       : {self.best_score_input_topic}')
        self.get_logger().info(f'final_object_grasp_topic     : {self.final_object_grasp_topic}')
        self.get_logger().info(f'selected_arm_id              : {self.selected_arm_id}')
        self.get_logger().info(f'require_matching_stamps      : {self.require_matching_stamps}')
        self.get_logger().info(f'best_pose_marker_topic        : {self.best_pose_marker_topic}')
        self.get_logger().info(f'best_contact_marker_topic     : {self.best_contact_marker_topic}')
        self.get_logger().info(f'republish_visuals_in_base_link: {self.republish_visuals_in_base_link}')
        if self.republish_visuals_in_base_link:
            self.get_logger().info(f'raw_grasps_output_topic       : {self.raw_grasps_output_topic}')
            self.get_logger().info(f'raw_grasp_markers_output_topic: {self.raw_grasp_markers_output_topic}')
            self.get_logger().info(f'raw_all_markers_output_topic  : {self.raw_all_grasp_markers_output_topic}')
            self.get_logger().info(f'raw_best_axes_output_topic    : {self.raw_best_axes_markers_output_topic}')
            self.get_logger().info(f'object_pc_output_topic        : {self.object_pc_output_topic_vis}')
            self.get_logger().info(f'background_pc_output_topic    : {self.background_pc_output_topic_vis}')
            self.get_logger().info(f'preview_cloud_output_topic    : {self.preview_cloud_output_topic_vis}')
            self.get_logger().info(f'full_scene_pc_output_topic    : {self.full_scene_pc_output_topic_vis}')
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
        self.try_publish_object_grasp()

    def object_center_callback(self, msg: PointStamped):
        out = self.transform_point(msg, 'SAM3_OBJECT_CENTER')
        if out is None:
            return

        self.pub_object_center_base.publish(out)
        self.last_object_center_base = out
        self.log_point_compact('SAM3_OBJECT_CENTER_BASE', out, input_frame=msg.header.frame_id)
        self.try_publish_object_grasp()

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
        self.try_publish_object_grasp()

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


    def best_width_callback(self, msg: Float32):
        self.last_grasp_width = float(msg.data)
        self.try_publish_object_grasp()

    def best_score_callback(self, msg: Float32):
        self.last_grasp_confidence = float(msg.data)
        self.try_publish_object_grasp()

    def object_pc_size_callback(self, msg: PointCloud2):
        """
        Estimate object W/D/H from the segmented object point cloud.
        The output is stored for the final ObjectGrasp message only.

        object_size convention:
          x = width  : smaller horizontal PCA span in base_link
          y = depth  : larger  horizontal PCA span in base_link
          z = height : vertical z span in base_link

        This keeps SAM3->AnyGrasp as standard PointCloud2/Image topics and
        prevents ObjectGrasp from being used before the result is planning-ready.
        """
        out_cloud = self.transform_pointcloud(msg, 'YOLO_OBJECT_PC_SIZE')
        if out_cloud is None:
            return

        size_tuple = self.estimate_object_size_from_cloud(out_cloud)
        if size_tuple is None:
            return

        self.last_object_size_base = size_tuple
        self.last_object_size_stamp_ns = self.stamp_to_ns(out_cloud.header.stamp)

        if self.log_pointcloud:
            self.get_logger().info(
                f'[OBJECT_SIZE_BASE] width={size_tuple[0]:.4f} depth={size_tuple[1]:.4f} '
                f'height={size_tuple[2]:.4f} frame={out_cloud.header.frame_id}'
            )

        self.try_publish_object_grasp()

    def try_publish_object_grasp(self):
        """
        Publish the final planning-level custom message.

        Strict rule:
          - ObjectGrasp is published only here.
          - Its frame is always base_frame.
          - All coordinate fields inside it are base_frame coordinates.
          - It is emitted only when pose/contact/center/size for the same stamp are available.
        """
        if self.last_pred_pose_base is None:
            return
        if self.last_contact_point_base is None:
            return
        if self.last_object_center_base is None:
            return
        if self.last_grasp_width is None:
            return
        if self.last_object_size_base is None:
            return

        pose_ns = self.stamp_to_ns(self.last_pred_pose_base.header.stamp)
        contact_ns = self.stamp_to_ns(self.last_contact_point_base.header.stamp)
        center_ns = self.stamp_to_ns(self.last_object_center_base.header.stamp)
        size_ns = self.last_object_size_stamp_ns

        if self.require_matching_stamps:
            if contact_ns != pose_ns or center_ns != pose_ns or size_ns != pose_ns:
                if self.verbose_debug:
                    self.get_logger().info(
                        '[FINAL_OBJECT_GRASP] waiting for matching stamps: '
                        f'pose={pose_ns} contact={contact_ns} center={center_ns} size={size_ns}'
                    )
                return

        if self.publish_once_per_stamp and self.last_final_publish_stamp_ns == pose_ns:
            return

        msg = ObjectGrasp()
        msg.header.stamp = self.last_pred_pose_base.header.stamp
        msg.header.frame_id = self.base_frame

        msg.arm_id = int(self.selected_arm_id)
        msg.label = str(self.final_label)
        msg.confidence = float(self.last_grasp_confidence) if self.last_grasp_confidence is not None else 1.0

        # Object pose: current pipeline provides object center, not a full object orientation.
        # Keep identity orientation to avoid pretending that object orientation was estimated.
        msg.object_pose.position.x = float(self.last_object_center_base.point.x)
        msg.object_pose.position.y = float(self.last_object_center_base.point.y)
        msg.object_pose.position.z = float(self.last_object_center_base.point.z)
        msg.object_pose.orientation.x = 0.0
        msg.object_pose.orientation.y = 0.0
        msg.object_pose.orientation.z = 0.0
        msg.object_pose.orientation.w = 1.0

        msg.object_size.x = float(self.last_object_size_base[0])
        msg.object_size.y = float(self.last_object_size_base[1])
        msg.object_size.z = float(self.last_object_size_base[2])

        msg.grasp_point.x = float(self.last_contact_point_base.point.x)
        msg.grasp_point.y = float(self.last_contact_point_base.point.y)
        msg.grasp_point.z = float(self.last_contact_point_base.point.z)

        msg.grasp_pose = copy.deepcopy(self.last_pred_pose_base.pose)
        msg.grasp_width = float(self.last_grasp_width)

        self.pub_object_grasp_final.publish(msg)
        self.last_final_publish_stamp_ns = pose_ns

        self.get_logger().info(
            '[FINAL_OBJECT_GRASP_PUBLISHED] '
            f'arm_id={msg.arm_id} frame={msg.header.frame_id} '
            f'center=({msg.object_pose.position.x:.4f},{msg.object_pose.position.y:.4f},{msg.object_pose.position.z:.4f}) '
            f'size_wdh=({msg.object_size.x:.4f},{msg.object_size.y:.4f},{msg.object_size.z:.4f}) '
            f'grasp=({msg.grasp_pose.position.x:.4f},{msg.grasp_pose.position.y:.4f},{msg.grasp_pose.position.z:.4f}) '
            f'grasp_width={msg.grasp_width:.4f} confidence={msg.confidence:.4f}'
        )

    def estimate_object_size_from_cloud(self, cloud_msg: PointCloud2):
        pts = []
        for p in pc2.read_points(cloud_msg, field_names=['x', 'y', 'z'], skip_nans=True):
            pts.append([float(p[0]), float(p[1]), float(p[2])])

        if len(pts) < self.object_size_min_points:
            self.get_logger().warn(
                f'[OBJECT_SIZE_BASE] too few points: {len(pts)} < {self.object_size_min_points}'
            )
            return None

        arr = np.asarray(pts, dtype=np.float64)

        # Robust percentiles reduce the effect of segmentation/depth outliers.
        z_lo, z_hi = np.percentile(arr[:, 2], [2.0, 98.0])
        height = max(0.0, float(z_hi - z_lo))

        xy = arr[:, :2]
        xy_center = np.mean(xy, axis=0)
        xy0 = xy - xy_center[None, :]

        if xy0.shape[0] >= 3 and np.linalg.norm(xy0) > 1e-9:
            cov = np.cov(xy0.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            axes = eigvecs[:, order]
            proj = xy0 @ axes
            span0 = float(np.percentile(proj[:, 0], 98.0) - np.percentile(proj[:, 0], 2.0))
            span1 = float(np.percentile(proj[:, 1], 98.0) - np.percentile(proj[:, 1], 2.0))
            depth = max(span0, span1)
            width = min(span0, span1)
        else:
            x_lo, x_hi = np.percentile(arr[:, 0], [2.0, 98.0])
            y_lo, y_hi = np.percentile(arr[:, 1], [2.0, 98.0])
            span_x = float(x_hi - x_lo)
            span_y = float(y_hi - y_lo)
            depth = max(span_x, span_y)
            width = min(span_x, span_y)

        return max(0.0, width), max(0.0, depth), height

    @staticmethod
    def stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


    # ------------------------------------------------------------
    # Visualization-only republish callbacks
    # ------------------------------------------------------------
    def raw_grasps_callback(self, msg: PoseArray):
        if self.pub_raw_grasps_base is None:
            return
        out = self.transform_pose_array_visual(msg, 'ANYGRASP_GRASPS_VIS')
        if out is None:
            return
        self.pub_raw_grasps_base.publish(out)
        if self.log_visual_republish:
            self.get_logger().info(
                f'[ANYGRASP_GRASPS_VIS_BASE] poses={len(out.poses)} '
                f'input_frame={msg.header.frame_id} output_frame={out.header.frame_id}'
            )

    def raw_grasp_markers_callback(self, msg: MarkerArray):
        if self.pub_raw_grasp_markers_base is None:
            return
        out = self.transform_marker_array_visual(msg, 'ANYGRASP_GRASP_MARKERS_VIS')
        if out is None:
            return
        self.pub_raw_grasp_markers_base.publish(out)
        if self.log_visual_republish:
            self.get_logger().info(f'[ANYGRASP_GRASP_MARKERS_VIS_BASE] markers={len(out.markers)}')

    def raw_all_grasp_markers_callback(self, msg: MarkerArray):
        if self.pub_raw_all_grasp_markers_base is None:
            return
        out = self.transform_marker_array_visual(msg, 'ANYGRASP_ALL_GRASP_MARKERS_VIS')
        if out is None:
            return
        self.pub_raw_all_grasp_markers_base.publish(out)
        if self.log_visual_republish:
            self.get_logger().info(f'[ANYGRASP_ALL_GRASP_MARKERS_VIS_BASE] markers={len(out.markers)}')

    def raw_best_axes_markers_callback(self, msg: MarkerArray):
        if self.pub_raw_best_axes_markers_base is None:
            return
        out = self.transform_marker_array_visual(msg, 'ANYGRASP_BEST_AXES_VIS')
        if out is None:
            return
        self.pub_raw_best_axes_markers_base.publish(out)
        if self.log_visual_republish:
            self.get_logger().info(f'[ANYGRASP_BEST_AXES_VIS_BASE] markers={len(out.markers)}')

    def object_pc_callback_vis(self, msg: PointCloud2):
        if self.pub_object_pc_base_vis is None:
            return
        out = self.transform_pointcloud(msg, 'YOLO_OBJECT_PC_VIS')
        if out is None:
            return
        self.pub_object_pc_base_vis.publish(out)

    def background_pc_callback_vis(self, msg: PointCloud2):
        if self.pub_background_pc_base_vis is None:
            return
        out = self.transform_pointcloud(msg, 'YOLO_BACKGROUND_PC_VIS')
        if out is None:
            return
        self.pub_background_pc_base_vis.publish(out)

    def preview_cloud_callback_vis(self, msg: PointCloud2):
        if self.pub_preview_cloud_base_vis is None:
            return
        out = self.transform_pointcloud(msg, 'SAM3_PREVIEW_PC_VIS')
        if out is None:
            return
        self.pub_preview_cloud_base_vis.publish(out)

    def full_scene_pc_callback_vis(self, msg: PointCloud2):
        if self.pub_full_scene_pc_base_vis is None:
            return
        out = self.transform_pointcloud(msg, 'SAM3_FULL_SCENE_PC_VIS')
        if out is None:
            return
        self.pub_full_scene_pc_base_vis.publish(out)

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
            out = copy.deepcopy(msg)
            out.header.frame_id = self.base_frame
            return out

        try:
            T_source_to_base = self.compute_source_to_base_matrix(frame_id, msg.header.stamp, source_name)
            return self.apply_transform_to_cloud(msg, T_source_to_base, self.base_frame)

        except Exception as e:
            self.get_logger().error(f'[{source_name}] point cloud transform failed: {repr(e)}')
            return None

    def transform_pose_array_visual(self, msg: PoseArray, source_name='POSE_ARRAY_VIS'):
        frame_id = msg.header.frame_id.strip()
        try:
            T_source_to_base = self.compute_source_to_base_matrix(frame_id, msg.header.stamp, source_name)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pose array transform failed: {repr(e)}')
            return None

        out = PoseArray()
        out.header = copy.deepcopy(msg.header)
        out.header.frame_id = self.base_frame

        for pose in msg.poses:
            T_pose_in = self.pose_to_matrix(pose.position, pose.orientation)
            T_pose_base = T_source_to_base @ T_pose_in
            out.poses.append(self.matrix_to_pose(T_pose_base))
        return out

    def transform_marker_array_visual(self, msg: MarkerArray, source_name='MARKER_ARRAY_VIS'):
        out = MarkerArray()
        for marker in msg.markers:
            tm = self.transform_marker_visual(marker, source_name)
            if tm is not None:
                out.markers.append(tm)
        return out

    def transform_marker_visual(self, marker: Marker, source_name='MARKER_VIS'):
        try:
            out = copy.deepcopy(marker)

            frame_id = marker.header.frame_id.strip()
            stamp = marker.header.stamp

            if marker.action == Marker.DELETEALL:
                out.header.frame_id = self.base_frame
                return out

            if frame_id == '':
                out.header.frame_id = self.base_frame
                return out

            T_source_to_base = self.compute_source_to_base_matrix(frame_id, stamp, source_name)
            out.header.frame_id = self.base_frame

            has_points = len(marker.points) > 0
            pose_is_identity = self.is_identity_pose(marker.pose)

            if has_points:
                if pose_is_identity:
                    out.points = [self.transform_plain_point(p, T_source_to_base) for p in marker.points]
                    out.pose = Pose()
                    out.pose.orientation.w = 1.0
                else:
                    T_pose_in = self.pose_to_matrix(marker.pose.position, marker.pose.orientation)
                    T_pose_base = T_source_to_base @ T_pose_in
                    out.pose = self.matrix_to_pose(T_pose_base)
                    out.points = copy.deepcopy(marker.points)
            else:
                T_pose_in = self.pose_to_matrix(marker.pose.position, marker.pose.orientation)
                T_pose_base = T_source_to_base @ T_pose_in
                out.pose = self.matrix_to_pose(T_pose_base)

            return out
        except Exception as e:
            self.get_logger().error(f'[{source_name}] marker transform failed ns={marker.ns} id={marker.id}: {repr(e)}')
            return None

    def compute_source_to_base_matrix(self, frame_id: str, stamp, source_name='TF') -> np.ndarray:
        frame_id = frame_id.strip()

        if frame_id == self.base_frame:
            return np.eye(4, dtype=np.float64)

        if self.use_direct_camera_tf and frame_id not in ('', self.gripper_frame):
            resolved_base, t_direct = self.lookup_base_from_source_frame(frame_id, stamp, source_name)
            if t_direct is not None:
                return self.make_transform_matrix(t_direct)

        resolved_base, t = self.resolve_base_frame(stamp, source_name)
        if t is None:
            raise RuntimeError('base<-gripper TF unavailable')

        T_gripper_to_base = self.make_transform_matrix(t)

        if frame_id == self.gripper_frame:
            return T_gripper_to_base

        return T_gripper_to_base @ self.T_cam_to_gripper

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

        header = copy.deepcopy(cloud_msg.header)
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

        x0 = -0.5 * finger_len
        x1 =  0.5 * finger_len
        yL = -0.5 * opening
        yR =  0.5 * opening

        z_front = 0.0
        z_finger_back = -finger_depth
        z_palm = -(finger_depth + palm_depth)
        z_wrist = z_palm - wrist_len

        segs_local = [
            ([x0, yL, z_front], [x1, yL, z_front]),
            ([x0, yL, z_finger_back], [x1, yL, z_finger_back]),
            ([x0, yL, z_front], [x0, yL, z_finger_back]),
            ([x1, yL, z_front], [x1, yL, z_finger_back]),

            ([x0, yR, z_front], [x1, yR, z_front]),
            ([x0, yR, z_finger_back], [x1, yR, z_finger_back]),
            ([x0, yR, z_front], [x0, yR, z_finger_back]),
            ([x1, yR, z_front], [x1, yR, z_finger_back]),

            ([0.0, yL, z_finger_back], [0.0, yL, z_palm]),
            ([0.0, yR, z_finger_back], [0.0, yR, z_palm]),
            ([0.0, yL, z_palm], [0.0, yR, z_palm]),
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

    @staticmethod
    def transform_plain_point(p: Point, T: np.ndarray) -> Point:
        v = np.array([p.x, p.y, p.z, 1.0], dtype=np.float64)
        vt = T @ v
        out = Point()
        out.x = float(vt[0])
        out.y = float(vt[1])
        out.z = float(vt[2])
        return out

    @staticmethod
    def is_identity_pose(pose: Pose, eps: float = 1e-9) -> bool:
        if abs(pose.position.x) > eps:
            return False
        if abs(pose.position.y) > eps:
            return False
        if abs(pose.position.z) > eps:
            return False
        if abs(pose.orientation.x) > eps:
            return False
        if abs(pose.orientation.y) > eps:
            return False
        if abs(pose.orientation.z) > eps:
            return False
        if abs(pose.orientation.w - 1.0) > eps:
            return False
        return True


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