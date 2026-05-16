#!/usr/bin/env python3
# CALI_D405 node for master_2.
# Converts SAM3/AnyGrasp camera-frame outputs into base_link and publishes ObjectGrasp.msg.
# This version matches the message definition requested by the user:
#   std_msgs/Header header
#   string label
#   string selected_arm
#   geometry_msgs/Pose object_pose
#   geometry_msgs/Vector3 object_size
#   geometry_msgs/Point grasp_point
#   geometry_msgs/Pose grasp_pose
#   geometry_msgs/Point[] grasp_points
#   geometry_msgs/Pose[] grasp_poses

import copy
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, ColorRGBA
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, PoseArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from grasp_msgs.msg import ObjectAlign, ObjectGrasp


class CaliD405Master2Node(Node):
    def __init__(self):
        super().__init__('cali_d405_r_master2_node')

        # ============================================================
        # Master control / target state
        # ============================================================
        self.declare_parameter('start_topic', '/cali_d405_r_start')
        self.declare_parameter('finish_topic', '/cali_d405_r_finish')
        self.declare_parameter('object_align_topic', '/object_align_result')
        self.declare_parameter('forced_selected_arm', 'right')

        # ============================================================
        # Inputs from SAM3 / AnyGrasp
        # ============================================================
        self.declare_parameter('grasp_pose_input_topic', '/anygrasp_r/best_pose_raw')
        self.declare_parameter('contact_point_input_topic', '/anygrasp_r/best_contact_point')
        self.declare_parameter('object_center_input_topic', '/sam3_r/object_center_camera')
        self.declare_parameter('object_pc_input_topic', '/sam3_r/object_pc')  # for object_size only
        self.declare_parameter('target_pc_input_topic', '/sam3_r/target_pc')   # RViz target cloud

        # ============================================================
        # Outputs
        # ============================================================
        self.declare_parameter('grasp_pose_output_topic', '/anygrasp_r/best_pose_base')
        self.declare_parameter('contact_point_output_topic', '/anygrasp_r/best_contact_point_base')
        self.declare_parameter('object_center_output_topic', '/sam3_r/object_center_base')
        self.declare_parameter('target_pc_output_topic', '/sam3_r/target_pc_base')
        self.declare_parameter('object_grasp_output_topic', '/object_grasp_result')

        # ============================================================
        # RViz visualization outputs only (base_link frame)
        # ============================================================
        self.declare_parameter('grasps_input_topic', '/anygrasp_r/grasps')
        self.declare_parameter('grasp_markers_base_topic', '/anygrasp_r/grasp_markers_base')
        self.declare_parameter('best_pose_marker_base_topic', '/anygrasp_r/best_pose_marker_base')
        # Best grasp axis visualization. Use this MarkerArray in RViz to see the final grasp frame clearly.
        # X=red, Y=green, Z=blue, all expressed in base_link.
        self.declare_parameter('best_pose_axes_base_topic', '/anygrasp_r/best_pose_axes_base')
        self.declare_parameter('best_axis_length', 0.10)
        self.declare_parameter('best_axis_shaft_diameter', 0.008)
        self.declare_parameter('best_axis_head_diameter', 0.018)
        self.declare_parameter('best_axis_head_length', 0.030)
        self.declare_parameter('candidate_marker_topk', 30)
        # RViz gripper-wire visualization. Same idea as AnyGrasp original LINE_LIST marker.
        self.declare_parameter('marker_alpha', 0.85)
        self.declare_parameter('best_gripper_line_width', 0.0030)
        self.declare_parameter('candidate_gripper_line_width', 0.0022)
        self.declare_parameter('visual_gripper_width', 0.10)
        self.declare_parameter('gripper_finger_length', 0.032)
        self.declare_parameter('gripper_palm_depth', 0.010)
        self.declare_parameter('gripper_tail_length', 0.010)
        self.declare_parameter('best_contact_scale', 0.012)
        self.declare_parameter('candidate_contact_scale', 0.008)

        # ============================================================
        # Frames / calibration
        # ============================================================
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base_frame_candidates', ['base_link', 'lift_link', 'arm_base_link'])
        self.declare_parameter('right_gripper_frame', 'gripper_r_rh_p12_rn_base')
        self.declare_parameter('left_gripper_frame', 'gripper_l_rh_p12_rn_base')
        self.declare_parameter('camera_frame', 'camera_r_color_optical_frame')

        # If true, uses TF directly from msg.header.frame_id to base_frame.
        # If false, uses base<-gripper TF and hand-eye camera->gripper matrix.
        self.declare_parameter('use_direct_camera_tf', False)
        self.declare_parameter('use_msg_timestamp', False)
        self.declare_parameter('tf_timeout_sec', 0.2)

        # Existing right D405 calibration from previous code:
        # camera optical -> arm_r_link7, then arm_r_link7 -> gripper_r_rh_p12_rn_base.
        self.declare_parameter('right_T_cam_to_link7', [
            0.9954, 0.0000, -0.0958, 0.0982,
            0.0000, -1.0000, 0.0000, 0.0000,
            -0.0958, 0.0000, -0.9954, -0.0725,
            0.0000, 0.0000, 0.0000, 1.0000,
        ])
        self.declare_parameter('right_link7_to_gripper_xyz', [0.0, 0.0, -0.0780])

        # Optional pose frame alignment for AnyGrasp pose -> FFW gripper pose.
        self.declare_parameter('apply_anygrasp_pose_frame_alignment', True)
        self.declare_parameter('auto_flip_pose_z_180_if_x_points_down', True)
        self.declare_parameter('x_axis_downward_flip_threshold', 0.0)

        # Final publish behavior
        self.declare_parameter('require_matching_stamps', True)
        self.declare_parameter('object_msg_sync_tolerance_sec', 0.30)
        self.declare_parameter('publish_once_per_stamp', True)
        # Additional guard against mixing outputs from different perception cycles.
        # Used when exact stamp matching is disabled or when stamps differ slightly.
        self.declare_parameter('max_input_stamp_delta_sec', 0.5)
        self.declare_parameter('object_size_min_points', 20)
        self.declare_parameter('debug', True)

        # ============================================================
        # Fetch params
        # ============================================================
        self.start_topic = self.get_parameter('start_topic').value
        self.finish_topic = self.get_parameter('finish_topic').value
        self.object_align_topic = self.get_parameter('object_align_topic').value
        self.forced_selected_arm = str(self.get_parameter('forced_selected_arm').value).strip().lower()

        self.grasp_pose_input_topic = self.get_parameter('grasp_pose_input_topic').value
        self.contact_point_input_topic = self.get_parameter('contact_point_input_topic').value
        self.object_center_input_topic = self.get_parameter('object_center_input_topic').value
        self.object_pc_input_topic = self.get_parameter('object_pc_input_topic').value
        self.target_pc_input_topic = self.get_parameter('target_pc_input_topic').value

        self.grasp_pose_output_topic = self.get_parameter('grasp_pose_output_topic').value
        self.contact_point_output_topic = self.get_parameter('contact_point_output_topic').value
        self.object_center_output_topic = self.get_parameter('object_center_output_topic').value
        self.target_pc_output_topic = self.get_parameter('target_pc_output_topic').value
        self.object_grasp_output_topic = self.get_parameter('object_grasp_output_topic').value

        self.grasps_input_topic = self.get_parameter('grasps_input_topic').value
        self.grasp_markers_base_topic = self.get_parameter('grasp_markers_base_topic').value
        self.best_pose_marker_base_topic = self.get_parameter('best_pose_marker_base_topic').value
        self.best_pose_axes_base_topic = self.get_parameter('best_pose_axes_base_topic').value
        self.best_axis_length = float(self.get_parameter('best_axis_length').value)
        self.best_axis_shaft_diameter = float(self.get_parameter('best_axis_shaft_diameter').value)
        self.best_axis_head_diameter = float(self.get_parameter('best_axis_head_diameter').value)
        self.best_axis_head_length = float(self.get_parameter('best_axis_head_length').value)
        self.candidate_marker_topk = int(self.get_parameter('candidate_marker_topk').value)
        self.marker_alpha = float(self.get_parameter('marker_alpha').value)
        self.best_gripper_line_width = float(self.get_parameter('best_gripper_line_width').value)
        self.candidate_gripper_line_width = float(self.get_parameter('candidate_gripper_line_width').value)
        self.visual_gripper_width = float(self.get_parameter('visual_gripper_width').value)
        self.gripper_finger_length = float(self.get_parameter('gripper_finger_length').value)
        self.gripper_palm_depth = float(self.get_parameter('gripper_palm_depth').value)
        self.gripper_tail_length = float(self.get_parameter('gripper_tail_length').value)
        self.best_contact_scale = float(self.get_parameter('best_contact_scale').value)
        self.candidate_contact_scale = float(self.get_parameter('candidate_contact_scale').value)

        self.base_frame = self.get_parameter('base_frame').value
        self.base_frame_candidates = list(self.get_parameter('base_frame_candidates').value)
        self.right_gripper_frame = self.get_parameter('right_gripper_frame').value
        self.left_gripper_frame = self.get_parameter('left_gripper_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.use_direct_camera_tf = bool(self.get_parameter('use_direct_camera_tf').value)
        self.use_msg_timestamp = bool(self.get_parameter('use_msg_timestamp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.apply_anygrasp_pose_frame_alignment = bool(self.get_parameter('apply_anygrasp_pose_frame_alignment').value)
        self.auto_flip_pose_z_180_if_x_points_down = bool(self.get_parameter('auto_flip_pose_z_180_if_x_points_down').value)
        self.x_axis_downward_flip_threshold = float(self.get_parameter('x_axis_downward_flip_threshold').value)
        self.require_matching_stamps = bool(self.get_parameter('require_matching_stamps').value)
        self.object_msg_sync_tolerance_sec = float(self.get_parameter('object_msg_sync_tolerance_sec').value)
        self.publish_once_per_stamp = bool(self.get_parameter('publish_once_per_stamp').value)
        self.max_input_stamp_delta_sec = float(self.get_parameter('max_input_stamp_delta_sec').value)
        self.object_size_min_points = int(self.get_parameter('object_size_min_points').value)
        self.debug = bool(self.get_parameter('debug').value)

        # ============================================================
        # Calibration matrices
        # ============================================================
        self.T_cam_to_link7_right = np.asarray(
            self.get_parameter('right_T_cam_to_link7').value,
            dtype=np.float64,
        ).reshape(4, 4)

        self.T_link7_to_gripper_right = np.eye(4, dtype=np.float64)
        self.T_link7_to_gripper_right[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        self.T_link7_to_gripper_right[:3, 3] = np.asarray(
            self.get_parameter('right_link7_to_gripper_xyz').value,
            dtype=np.float64,
        ).reshape(3)
        self.T_cam_to_gripper_right = self.T_link7_to_gripper_right @ self.T_cam_to_link7_right

        # For now, left uses the same matrix unless you add a left calibration parameter.
        self.T_cam_to_gripper_left = self.T_cam_to_gripper_right.copy()

        self.T_pose_align_y90 = np.eye(4, dtype=np.float64)
        self.T_pose_align_y90[:3, :3] = self.rot_from_euler_y(np.deg2rad(90.0))
        self.T_pose_align_z180 = np.eye(4, dtype=np.float64)
        self.T_pose_align_z180[:3, :3] = self.rot_from_euler_z(np.deg2rad(180.0))

        # ============================================================
        # Runtime state
        # ============================================================
        self.active = False
        self.current_label = 'target_object'
        self.current_selected_arm = self.forced_selected_arm if self.forced_selected_arm in ('left', 'right') else 'right'
        self.current_text_prompt = ''
        self.current_align_stamp_ns = None

        self.last_pred_pose_base: PoseStamped = None
        self.last_contact_point_base: PointStamped = None
        self.last_object_center_base: PointStamped = None
        self.last_object_size_base = None
        self.last_object_size_stamp_ns = None
        self.last_candidate_poses_base: PoseArray = None
        self.last_final_publish_stamp_ns = None

        # ============================================================
        # TF / QoS
        # ============================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================================
        # Subscriptions
        # ============================================================
        self.create_subscription(Bool, self.start_topic, self.start_callback, self.qos_cmd)
        self.create_subscription(ObjectAlign, self.object_align_topic, self.object_align_callback, self.qos_cmd)

        self.create_subscription(PoseStamped, self.grasp_pose_input_topic, self.grasp_pose_callback, 10)
        self.create_subscription(PointStamped, self.contact_point_input_topic, self.contact_point_callback, 10)
        self.create_subscription(PointStamped, self.object_center_input_topic, self.object_center_callback, 10)
        self.create_subscription(PointCloud2, self.object_pc_input_topic, self.object_pc_callback, 10)
        self.create_subscription(PointCloud2, self.target_pc_input_topic, self.target_pc_callback, 10)
        self.create_subscription(PoseArray, self.grasps_input_topic, self.grasps_visualization_callback, 10)

        # ============================================================
        # Publishers
        # ============================================================
        self.finish_pub = self.create_publisher(Bool, self.finish_topic, self.qos_cmd)
        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, self.grasp_pose_output_topic, 10)
        self.pub_contact_point_base = self.create_publisher(PointStamped, self.contact_point_output_topic, 10)
        self.pub_object_center_base = self.create_publisher(PointStamped, self.object_center_output_topic, 10)
        self.pub_target_pc_base = self.create_publisher(PointCloud2, self.target_pc_output_topic, 10)
        self.pub_object_grasp = self.create_publisher(
            ObjectGrasp,
            self.object_grasp_output_topic,
            self.qos_cmd,
        )
        self.pub_grasp_markers_base = self.create_publisher(MarkerArray, self.grasp_markers_base_topic, 10)
        self.pub_best_pose_marker_base = self.create_publisher(Marker, self.best_pose_marker_base_topic, 10)
        self.pub_best_pose_axes_base = self.create_publisher(MarkerArray, self.best_pose_axes_base_topic, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('CALI_D405 MASTER2 Node Ready (RIGHT ARM, OBJECTGRASP ARRAY CANDIDATES)')
        self.get_logger().info(f'start_topic              : {self.start_topic}')
        self.get_logger().info(f'object_align_topic       : {self.object_align_topic}')
        self.get_logger().info(f'forced_selected_arm      : {self.forced_selected_arm}')
        self.get_logger().info(f'object_grasp_output_topic: {self.object_grasp_output_topic}')
        self.get_logger().info(f'grasps_input_topic       : {self.grasps_input_topic}')
        self.get_logger().info(f'grasp_markers_base_topic : {self.grasp_markers_base_topic}')
        self.get_logger().info(f'best_pose_marker_base    : {self.best_pose_marker_base_topic}  # legacy single marker')
        self.get_logger().info(f'best_pose_axes_base      : {self.best_pose_axes_base_topic}  # use this in RViz as MarkerArray')
        self.get_logger().info(f'grasp_pose_input_topic   : {self.grasp_pose_input_topic}')
        self.get_logger().info(f'contact_point_input_topic: {self.contact_point_input_topic}')
        self.get_logger().info(f'object_center_input_topic: {self.object_center_input_topic}')
        self.get_logger().info(f'object_pc_input_topic    : {self.object_pc_input_topic}  # size only')
        self.get_logger().info(f'target_pc_input_topic    : {self.target_pc_input_topic}')
        self.get_logger().info(f'target_pc_output_topic   : {self.target_pc_output_topic}')
        self.get_logger().info(f'base_frame               : {self.base_frame}')
        self.get_logger().info(f'right_gripper_frame      : {self.right_gripper_frame}')
        self.get_logger().info(f'left_gripper_frame       : {self.left_gripper_frame}')
        self.get_logger().info(f'use_direct_camera_tf     : {self.use_direct_camera_tf}')
        self.get_logger().info(f'require_matching_stamps  : {self.require_matching_stamps}')
        self.get_logger().info(f'object_msg_sync_tolerance_sec: {self.object_msg_sync_tolerance_sec:.3f}')
        self.get_logger().info('========================================')

    # ============================================================
    # Master / target callbacks
    # ============================================================
    def start_callback(self, msg: Bool):
        if msg.data:
            self.active = True
            self.clear_runtime_cache()
            self.get_logger().info(
                f'[START] /cali_d405_r_start true. label={self.current_label}, arm={self.current_selected_arm}'
            )
        else:
            self.active = False
            self.clear_runtime_cache()
            self.get_logger().info('[STOP] /cali_d405_r_start false. paused.')

    def object_align_callback(self, msg: ObjectAlign):
        # CALI_D405는 /target_item_name을 직접 보지 않는다.
        # Detection_grasping 단계의 label/arm/prompt는 이전 OBJECT_ALIGN 결과인
        # ObjectAlign.msg 하나만 기준으로 잡는다.
        if msg.label:
            self.current_label = msg.label
        # This node is split per arm. Keep TF/gripper selection fixed to forced_selected_arm.
        # ObjectAlign.selected_arm is still checked for logging/guarding, but it must not flip this node to the opposite arm.
        rx_arm = (msg.selected_arm or '').strip().lower()
        if self.forced_selected_arm in ('left', 'right'):
            self.current_selected_arm = self.forced_selected_arm
            if rx_arm and rx_arm != self.forced_selected_arm:
                if self.debug:
                    self.get_logger().warn(
                        f'[ObjectAlign RX] selected_arm={rx_arm} but this is {self.forced_selected_arm} node. Keeping forced arm.'
                    )
        elif rx_arm:
            self.current_selected_arm = rx_arm
        if msg.text_prompt:
            self.current_text_prompt = msg.text_prompt
        self.current_align_stamp_ns = self.stamp_to_ns(msg.header.stamp)

        self.get_logger().info(
            f'[ObjectAlign RX] label={self.current_label}, '
            f'arm={self.current_selected_arm}, prompt={self.current_text_prompt}'
        )

    def publish_finish(self, value: bool = True):
        out = Bool()
        out.data = bool(value)
        self.finish_pub.publish(out)
        self.get_logger().info(f'[PUB] {self.finish_topic} data={str(value).lower()}')

    def clear_runtime_cache(self):
        self.last_pred_pose_base = None
        self.last_contact_point_base = None
        self.last_object_center_base = None
        self.last_object_size_base = None
        self.last_object_size_stamp_ns = None
        self.last_candidate_poses_base: PoseArray = None
        self.last_final_publish_stamp_ns = None

    # ============================================================
    # Input callbacks
    # ============================================================
    def contact_point_callback(self, msg: PointStamped):
        if not self.active:
            return
        out = self.transform_point(msg, 'ANYGRASP_CONTACT')
        if out is None:
            return
        self.pub_contact_point_base.publish(out)
        self.last_contact_point_base = out
        self.try_publish_object_grasp()

    def object_center_callback(self, msg: PointStamped):
        if not self.active:
            return
        out = self.transform_point(msg, 'SAM3_OBJECT_CENTER')
        if out is None:
            return
        self.pub_object_center_base.publish(out)
        self.last_object_center_base = out
        self.try_publish_object_grasp()

    def grasp_pose_callback(self, msg: PoseStamped):
        # Visualization output is always converted to base_link so RViz can show it even
        # if the final ObjectGrasp state machine is already inactive.
        out = self.transform_pose(msg, 'ANYGRASP_POSE')
        if out is None:
            return
        self.pub_grasp_pose_base.publish(out)
        self.publish_best_pose_marker_base(out)

        # Keep the original ObjectGrasp behavior unchanged: only update final result state when active.
        if not self.active:
            return
        self.last_pred_pose_base = out
        self.try_publish_object_grasp()

    def object_pc_callback(self, msg: PointCloud2):
        # Object_pc is used for object_size estimation only. It is not republished
        # as an RViz topic because the requested visible cloud is target_pc_base.
        if not self.active:
            return
        out_cloud = self.transform_pointcloud(msg, 'OBJECT_PC_SIZE')
        if out_cloud is None:
            return
        size = self.estimate_object_size_from_cloud(out_cloud)
        if size is None:
            return
        self.last_object_size_base = size
        self.last_object_size_stamp_ns = self.stamp_to_ns(out_cloud.header.stamp)
        self.try_publish_object_grasp()

    def target_pc_callback(self, msg: PointCloud2):
        # The only point cloud intended for RViz: segmented target cloud in base_link.
        out_cloud = self.transform_pointcloud(msg, 'TARGET_PC_VIS')
        if out_cloud is None:
            return
        self.pub_target_pc_base.publish(out_cloud)



    def grasps_visualization_callback(self, msg: PoseArray):
        # This callback is visualization-only. It does not touch ObjectGrasp runtime state.
        if len(msg.poses) == 0:
            if self.active:
                self.last_candidate_poses_base = None
            empty = MarkerArray()
            dm = Marker()
            dm.header.stamp = msg.header.stamp
            dm.header.frame_id = self.base_frame
            dm.action = Marker.DELETEALL
            empty.markers.append(dm)
            self.pub_grasp_markers_base.publish(empty)
            return
        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame

        try:
            T_source_to_base = self.compute_source_to_base_matrix(
                msg.header.frame_id,
                msg.header.stamp,
                'ANYGRASP_GRASPS_VIS',
            )
        except Exception as e:
            self.get_logger().error(f'[ANYGRASP_GRASPS_VIS] transform failed: {repr(e)}')
            return

        for pose in msg.poses:
            try:
                T_pose_in = self.pose_to_matrix(pose.position, pose.orientation)
                T_final = T_source_to_base @ T_pose_in
                if self.apply_anygrasp_pose_frame_alignment:
                    T_final = T_final @ self.T_pose_align_y90
                    if self.auto_flip_pose_z_180_if_x_points_down:
                        if T_final[:3, 0][2] < self.x_axis_downward_flip_threshold:
                            T_final = T_final @ self.T_pose_align_z180
                out.poses.append(self.matrix_to_pose(T_final))
            except Exception as e:
                if self.debug:
                    self.get_logger().warn(f'[ANYGRASP_GRASPS_VIS] pose skipped: {repr(e)}')

        if len(out.poses) == 0:
            return

        if self.active:
            # /anygrasp_r/grasps already contains only the 1st-filtered candidates.
            # Store the whole candidate list in base_link. candidates[0] is kept as
            # the best AnyGrasp/CALI-side candidate, but final feasibility selection
            # is delegated to the robot arm node.
            self.last_candidate_poses_base = out

            # Keep legacy scalar fields consistent with candidates[0].
            first_pose = PoseStamped()
            first_pose.header = out.header
            first_pose.pose = copy.deepcopy(out.poses[0])
            self.last_pred_pose_base = first_pose

            first_point = PointStamped()
            first_point.header = out.header
            first_point.point.x = float(out.poses[0].position.x)
            first_point.point.y = float(out.poses[0].position.y)
            first_point.point.z = float(out.poses[0].position.z)
            self.last_contact_point_base = first_point

            self.try_publish_object_grasp()

        self.publish_grasp_markers_base(out)


    # ============================================================
    # Final ObjectGrasp
    # ============================================================
    def try_publish_object_grasp(self):
        """
        Publish one ObjectGrasp.msg that contains all 1st-filtered candidates.

        Message contract used by this node:
          - ObjectGrasp.grasp_point / ObjectGrasp.grasp_pose
              = representative candidate, i.e. candidate[0]
          - ObjectGrasp.grasp_points[] / ObjectGrasp.grasp_poses[]
              = all candidates that passed AnyGrasp/SAM3 1st filtering
        Therefore /object_grasp_result is published ONCE per perception cycle.
        The robot arm node should iterate over msg.grasp_poses and run IK/collision
        checks there, then select the final executable candidate.
        """
        if not self.active:
            return
        if self.last_candidate_poses_base is None or len(self.last_candidate_poses_base.poses) == 0:
            return
        if self.last_object_center_base is None:
            return
        if self.last_object_size_base is None:
            return

        pose_ns = self.stamp_to_ns(self.last_candidate_poses_base.header.stamp)
        center_ns = self.stamp_to_ns(self.last_object_center_base.header.stamp)
        size_ns = self.last_object_size_stamp_ns

        if self.require_matching_stamps:
            tol_ns = int(max(0.0, self.object_msg_sync_tolerance_sec) * 1e9)
            ok_center = abs(center_ns - pose_ns) <= tol_ns
            ok_size = size_ns is not None and abs(size_ns - pose_ns) <= tol_ns
            if not (ok_center and ok_size):
                if self.debug:
                    self.get_logger().info(
                        '[ObjectGrasp] waiting matching stamps within tolerance: '
                        f'candidates={pose_ns} center={center_ns} size={size_ns} tol_ns={tol_ns}'
                    )
                return

        if self.max_input_stamp_delta_sec >= 0.0:
            stamps = [pose_ns, center_ns, size_ns]
            dt_sec = (max(stamps) - min(stamps)) * 1e-9
            if dt_sec > self.max_input_stamp_delta_sec:
                if self.debug:
                    self.get_logger().warn(
                        '[ObjectGrasp] input stamps are too far apart. '
                        f'delta={dt_sec:.3f}s > {self.max_input_stamp_delta_sec:.3f}s | '
                        f'candidates={pose_ns} center={center_ns} size={size_ns}'
                    )
                return

        if self.publish_once_per_stamp and self.last_final_publish_stamp_ns == pose_ns:
            return

        header = copy.deepcopy(self.last_candidate_poses_base.header)
        header.frame_id = self.base_frame

        poses = [copy.deepcopy(p) for p in self.last_candidate_poses_base.poses]
        candidate_count = len(poses)
        if candidate_count == 0:
            return


        points = []
        for pose in poses:
            p = Point()
            p.x = float(pose.position.x)
            p.y = float(pose.position.y)
            p.z = float(pose.position.z)
            points.append(p)

        msg = ObjectGrasp()
        msg.header = copy.deepcopy(header)
        msg.label = str(self.current_label)
        msg.selected_arm = str(self.current_selected_arm)

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

        # Legacy/single-result fields are kept as candidate[0].
        msg.grasp_point = copy.deepcopy(points[0])
        msg.grasp_pose = copy.deepcopy(poses[0])

        # New array fields: all 1st-filtered candidates in base_link.
        # These fields require the updated ObjectGrasp.msg provided with this answer.
        msg.grasp_points = points
        msg.grasp_poses = poses

        self.pub_object_grasp.publish(msg)
        self.last_final_publish_stamp_ns = pose_ns
        self.active = False
        self.publish_finish(True)

        self.get_logger().info(
            '[ObjectGrasp ARRAY PUBLISHED] '
            f'label={msg.label} arm={msg.selected_arm} count={candidate_count} '
            f'best=({msg.grasp_pose.position.x:.4f},'
            f'{msg.grasp_pose.position.y:.4f},'
            f'{msg.grasp_pose.position.z:.4f}) '
            f'center=({msg.object_pose.position.x:.4f},'
            f'{msg.object_pose.position.y:.4f},'
            f'{msg.object_pose.position.z:.4f}) '
            f'size=({msg.object_size.x:.4f},'
            f'{msg.object_size.y:.4f},'
            f'{msg.object_size.z:.4f})'
        )

    # ============================================================
    # RViz visualization helpers only
    # ============================================================
    def publish_best_pose_marker_base(self, pose_msg: PoseStamped):
        """
        Publish the BEST grasp visualization as an XYZ coordinate frame, not as a gripper wire.

        Important:
          - Candidate grasps are still shown as gripper wireframes on /anygrasp_r/grasp_markers_base.
          - The best grasp is shown as axes on /anygrasp_r/best_pose_axes_base.
          - For the legacy single Marker topic /anygrasp_r/best_pose_marker_base, publish only
            the best X-axis arrow so old RViz configs do not show the old gripper shape.

        Axis convention in RViz:
          X: red
          Y: green
          Z: blue
        """
        axes = self.make_best_pose_axes_markers_base(pose_msg.header, pose_msg.pose)
        self.pub_best_pose_axes_base.publish(axes)

        # Legacy single-marker topic: publish the X-axis arrow only.
        # To see the full XYZ frame, add /anygrasp_r/best_pose_axes_base as MarkerArray.
        for m in axes.markers:
            if m.action == Marker.ADD and m.ns == 'best_grasp_axis_x':
                self.pub_best_pose_marker_base.publish(m)
                break

    def make_best_pose_axes_markers_base(self, header, pose: Pose) -> MarkerArray:
        """Create a MarkerArray that visualizes the final best grasp pose as XYZ axes."""
        ma = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        T = self.pose_to_matrix(pose.position, pose.orientation)
        origin = T[:3, 3].astype(np.float64)
        x_axis = T[:3, 0].astype(np.float64)
        y_axis = T[:3, 1].astype(np.float64)
        z_axis = T[:3, 2].astype(np.float64)

        ma.markers.append(self.make_axis_arrow_marker_base(
            header, marker_id=1, ns='best_grasp_axis_x', origin=origin, axis=x_axis,
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
        ))
        ma.markers.append(self.make_axis_arrow_marker_base(
            header, marker_id=2, ns='best_grasp_axis_y', origin=origin, axis=y_axis,
            color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
        ))
        ma.markers.append(self.make_axis_arrow_marker_base(
            header, marker_id=3, ns='best_grasp_axis_z', origin=origin, axis=z_axis,
            color=ColorRGBA(r=0.1, g=0.35, b=1.0, a=1.0),
        ))

        center = Marker()
        center.header = header
        center.ns = 'best_grasp_axis_origin'
        center.id = 4
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position = copy.deepcopy(pose.position)
        center.pose.orientation.w = 1.0
        center.scale.x = self.best_contact_scale * 1.15
        center.scale.y = self.best_contact_scale * 1.15
        center.scale.z = self.best_contact_scale * 1.15
        center.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        ma.markers.append(center)
        return ma

    def make_axis_arrow_marker_base(self, header, marker_id: int, ns: str, origin: np.ndarray, axis: np.ndarray, color: ColorRGBA) -> Marker:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / n

        p0 = origin.astype(np.float64)
        p1 = p0 + float(self.best_axis_length) * axis

        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
            Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
        ]
        marker.scale.x = float(self.best_axis_shaft_diameter)
        marker.scale.y = float(self.best_axis_head_diameter)
        marker.scale.z = float(self.best_axis_head_length)
        marker.color = color
        marker.pose.orientation.w = 1.0
        return marker

    def publish_grasp_markers_base(self, pose_array: PoseArray):
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = pose_array.header
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        topk = len(pose_array.poses)
        if self.candidate_marker_topk > 0:
            topk = min(topk, int(self.candidate_marker_topk))

        for i, pose in enumerate(pose_array.poses[:topk]):
            markers.markers.extend(self.make_gripper_wire_markers_base(
                pose_array.header,
                pose,
                idx=i,
                best=(i == 0),
            ))

        self.pub_grasp_markers_base.publish(markers)

    def make_gripper_wire_markers_base(self, header, pose: Pose, idx: int, best: bool = False):
        # This reproduces the previous AnyGrasp-style "LEGO hand" visualization:
        # one LINE_LIST marker for the parallel-jaw gripper wireframe plus
        # one SPHERE marker at the grasp/contact center.
        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = 'best_grasp_base' if best else 'gripper_candidates_base'
        line_marker.id = idx * 2
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = self.best_gripper_line_width if best else self.candidate_gripper_line_width
        line_marker.color = ColorRGBA(
            r=1.0 if best else 0.3,
            g=0.2 if best else 1.0,
            b=0.0 if best else 0.6,
            a=1.0 if best else self.marker_alpha,
        )
        line_marker.points = self.gripper_wire_points_from_pose(pose)

        contact_marker = Marker()
        contact_marker.header = header
        contact_marker.ns = 'best_contact_base' if best else 'contact_candidates_base'
        contact_marker.id = idx * 2 + 1
        contact_marker.type = Marker.SPHERE
        contact_marker.action = Marker.ADD
        contact_marker.pose.position = copy.deepcopy(pose.position)
        contact_marker.pose.orientation.w = 1.0
        scale = self.best_contact_scale if best else self.candidate_contact_scale
        contact_marker.scale.x = scale
        contact_marker.scale.y = scale
        contact_marker.scale.z = scale
        contact_marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)

        return [line_marker, contact_marker]

    def gripper_wire_points_from_pose(self, pose: Pose):
        T = self.pose_to_matrix(pose.position, pose.orientation)
        width = max(0.01, float(self.visual_gripper_width))
        finger = float(self.gripper_finger_length)
        palm = float(self.gripper_palm_depth)
        tail = float(self.gripper_tail_length)

        # Same local wire geometry as the AnyGrasp marker code:
        # two fingers, a palm bridge, and a short tail showing approach direction.
        segs_local = [
            ([0.0, -width / 2.0, 0.0], [finger, -width / 2.0, 0.0]),
            ([0.0,  width / 2.0, 0.0], [finger,  width / 2.0, 0.0]),
            ([0.0, -width / 2.0, 0.0], [0.0,   width / 2.0, 0.0]),
            ([-palm, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([-palm - tail, 0.0, 0.0], [-palm, 0.0, 0.0]),
        ]

        pts = []
        for a, b in segs_local:
            av = T @ np.array([a[0], a[1], a[2], 1.0], dtype=np.float64)
            bv = T @ np.array([b[0], b[1], b[2], 1.0], dtype=np.float64)
            pts.append(Point(x=float(av[0]), y=float(av[1]), z=float(av[2])))
            pts.append(Point(x=float(bv[0]), y=float(bv[1]), z=float(bv[2])))
        return pts

    # ============================================================
    # Transform helpers
    # ============================================================
    def transform_point(self, msg: PointStamped, source_name='POINT'):
        try:
            T = self.compute_source_to_base_matrix(msg.header.frame_id, msg.header.stamp, source_name)
            p = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)
            pb = T @ p
        except Exception as e:
            self.get_logger().error(f'[{source_name}] point transform failed: {repr(e)}')
            return None

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.point.x = float(pb[0])
        out.point.y = float(pb[1])
        out.point.z = float(pb[2])
        return out

    def transform_pose(self, msg: PoseStamped, source_name='POSE'):
        try:
            T_source_to_base = self.compute_source_to_base_matrix(msg.header.frame_id, msg.header.stamp, source_name)
            T_pose_in = self.pose_to_matrix(msg.pose.position, msg.pose.orientation)
            T_final = T_source_to_base @ T_pose_in

            if self.apply_anygrasp_pose_frame_alignment:
                T_final = T_final @ self.T_pose_align_y90
                if self.auto_flip_pose_z_180_if_x_points_down:
                    if T_final[:3, 0][2] < self.x_axis_downward_flip_threshold:
                        T_final = T_final @ self.T_pose_align_z180
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pose transform failed: {repr(e)}')
            return None

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.pose = self.matrix_to_pose(T_final)
        return out

    def transform_pointcloud(self, msg: PointCloud2, source_name='POINTCLOUD'):
        try:
            T = self.compute_source_to_base_matrix(msg.header.frame_id, msg.header.stamp, source_name)
            return self.apply_transform_to_cloud(msg, T, self.base_frame)
        except Exception as e:
            self.get_logger().error(f'[{source_name}] pointcloud transform failed: {repr(e)}')
            return None

    def compute_source_to_base_matrix(self, frame_id: str, stamp, source_name='TF') -> np.ndarray:
        frame_id = (frame_id or '').strip()
        if frame_id == self.base_frame:
            return np.eye(4, dtype=np.float64)

        if self.use_direct_camera_tf:
            t_direct = self.lookup_transform(self.base_frame, frame_id, stamp, source_name)
            if t_direct is not None:
                return self.make_transform_matrix(t_direct)

        gripper_frame = self.right_gripper_frame if self.current_selected_arm != 'left' else self.left_gripper_frame
        T_cam_to_gripper = self.T_cam_to_gripper_left if self.current_selected_arm == 'left' else self.T_cam_to_gripper_right

        t = self.lookup_transform(self.base_frame, gripper_frame, stamp, source_name)
        if t is None:
            # Try fallback base frame candidates.
            for cand in self.base_frame_candidates:
                t = self.lookup_transform(cand, gripper_frame, stamp, source_name)
                if t is not None:
                    self.get_logger().warn(f'[{source_name}] base_frame fallback: {self.base_frame} -> {cand}')
                    self.base_frame = cand
                    break
        if t is None:
            raise RuntimeError(f'base<-{gripper_frame} TF unavailable')

        T_gripper_to_base = self.make_transform_matrix(t)
        if frame_id == gripper_frame:
            return T_gripper_to_base
        return T_gripper_to_base @ T_cam_to_gripper

    def lookup_transform(self, target_frame, source_frame, stamp, source_name='TF'):
        try:
            if self.use_msg_timestamp:
                target_time = rclpy.time.Time.from_msg(stamp)
            else:
                target_time = rclpy.time.Time()
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                target_time,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as e:
            if self.debug:
                self.get_logger().warn(f'[{source_name}] TF failed: {target_frame} <- {source_frame} | {e}')
            return None

    def apply_transform_to_cloud(self, cloud_msg: PointCloud2, T: np.ndarray, out_frame: str) -> PointCloud2:
        field_names = [f.name for f in cloud_msg.fields]
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            raise RuntimeError('PointCloud2 does not contain x/y/z fields')
        idx_x = field_names.index('x')
        idx_y = field_names.index('y')
        idx_z = field_names.index('z')
        rows = []
        for p in pc2.read_points(cloud_msg, field_names=field_names, skip_nans=False):
            p_list = list(p)
            x, y, z = p_list[idx_x], p_list[idx_y], p_list[idx_z]
            if any(math.isnan(float(v)) for v in [x, y, z]):
                rows.append(tuple(p_list))
                continue
            vt = T @ np.array([x, y, z, 1.0], dtype=np.float64)
            p_list[idx_x] = float(vt[0])
            p_list[idx_y] = float(vt[1])
            p_list[idx_z] = float(vt[2])
            rows.append(tuple(p_list))
        header = copy.deepcopy(cloud_msg.header)
        header.frame_id = out_frame
        out_cloud = pc2.create_cloud(header, cloud_msg.fields, rows)
        out_cloud.height = cloud_msg.height
        out_cloud.width = cloud_msg.width
        out_cloud.is_bigendian = cloud_msg.is_bigendian
        out_cloud.is_dense = cloud_msg.is_dense
        return out_cloud

    # ============================================================
    # Object size
    # ============================================================
    def estimate_object_size_from_cloud(self, cloud_msg: PointCloud2):
        pts = []
        for p in pc2.read_points(cloud_msg, field_names=['x', 'y', 'z'], skip_nans=True):
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        if len(pts) < self.object_size_min_points:
            self.get_logger().warn(f'[OBJECT_SIZE] too few points: {len(pts)} < {self.object_size_min_points}')
            return None
        arr = np.asarray(pts, dtype=np.float64)
        z_lo, z_hi = np.percentile(arr[:, 2], [2.0, 98.0])
        height = max(0.0, float(z_hi - z_lo))
        xy = arr[:, :2]
        xy0 = xy - np.mean(xy, axis=0)[None, :]
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

    # ============================================================
    # Math helpers, no SciPy dependency
    # ============================================================
    @staticmethod
    def stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    @staticmethod
    def rot_from_euler_y(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    @staticmethod
    def rot_from_euler_z(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def quat_to_rot_matrix(x, y, z, w):
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
            [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
            [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)],
        ], dtype=np.float64)

    @staticmethod
    def rot_matrix_to_quat(Rm):
        # Returns x,y,z,w
        m = Rm
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-12:
            return 0.0, 0.0, 0.0, 1.0
        return x/norm, y/norm, z/norm, w/norm

    @classmethod
    def pose_to_matrix(cls, position, orientation) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = cls.quat_to_rot_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
        T[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float64)
        return T

    @classmethod
    def matrix_to_pose(cls, T: np.ndarray) -> Pose:
        qx, qy, qz, qw = cls.rot_matrix_to_quat(T[:3, :3])
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose

    @classmethod
    def make_transform_matrix(cls, t) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        tr = t.transform.translation
        qr = t.transform.rotation
        T[:3, :3] = cls.quat_to_rot_matrix(qr.x, qr.y, qr.z, qr.w)
        T[:3, 3] = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        return T


def main(args=None):
    rclpy.init(args=args)
    node = CaliD405Master2Node()
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