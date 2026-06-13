#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALI_ZED_DOOR node for shelf_1 HANDLE_OPEN.

Purpose:
  - Receive a door-handle center detected by SAM3_DOOR in a camera frame.
  - Transform it into base_link.
  - Publish a door-handle ObjectAlign-style result on a NEW topic.

This node intentionally does NOT publish /object_align_result.
It reuses grasp_msgs/ObjectAlign.msg only as a convenient data format.

Subscribe:
  /cali_zed_door_start              std_msgs/Bool
  /sam3_door/handle_center_camera   geometry_msgs/PointStamped

Publish:
  /door_handle_center_base          geometry_msgs/PointStamped
  /door_handle_align_result         grasp_msgs/ObjectAlign

ObjectAlign fields for door opening:
  aruco_id        = -1
  label           = "door_handle"
  text_prompt     = "door handle"
  selected_arm    = "right"
  shelf_type      = "shelf_1"
  marker_position = base_link door-handle center
  align_pose      = marker_position + offset + fixed right-arm orientation
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped, Pose
import tf2_ros
from scipy.spatial.transform import Rotation as R

from grasp_msgs.msg import ObjectAlign


class CaliZedDoorNode(Node):
    def __init__(self):
        super().__init__("cali_zed_door_node")

        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter("start_topic", "/cali_zed_door_start")
        self.declare_parameter("handle_input_topic", "/sam3_door/handle_center_camera")
        self.declare_parameter("handle_output_topic", "/door_handle_center_base")
        self.declare_parameter("door_align_output_topic", "/door_handle_align_result")

        # Optional prompt update from master. The result still defaults to door handle.
        self.declare_parameter("door_prompt_topic", "/sam3_door_text_prompt")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("use_msg_frame_id", True)
        self.declare_parameter("camera_frame_fallback", "zedm_left_camera_optical_frame")
        self.declare_parameter("use_msg_timestamp", True)
        self.declare_parameter("fallback_to_latest_tf", True)
        self.declare_parameter("tf_timeout_sec", 0.2)

        # ObjectAlign-style fixed fields for HANDLE_OPEN.
        self.declare_parameter("door_aruco_id", -1)
        self.declare_parameter("door_label", "door_handle")
        self.declare_parameter("door_text_prompt", "door handle")
        self.declare_parameter("door_selected_arm", "right")
        self.declare_parameter("door_shelf_type", "shelf_1")

        # align_pose position = handle center + offset.
        self.declare_parameter("align_offset_x", 0.0)
        self.declare_parameter("align_offset_y", 0.0)
        self.declare_parameter("align_offset_z", 0.0)

        # Fixed right-arm door-opening orientation.
        # Keep same convention as CALI_ZED right alignment unless tuned later.
        self.declare_parameter("right_align_qx", 0.5)
        self.declare_parameter("right_align_qy", -0.5)
        self.declare_parameter("right_align_qz", 0.5)
        self.declare_parameter("right_align_qw", -0.5)

        self.declare_parameter("publish_once_per_start", True)
        self.declare_parameter("publish_repeat", 5)
        self.declare_parameter("publish_repeat_period_sec", 0.05)
        self.declare_parameter("debug", True)
        self.declare_parameter("print_tf_matrix", False)

        # ==================================================
        # Fetch parameters
        # ==================================================
        self.start_topic = self.get_parameter("start_topic").value
        self.handle_input_topic = self.get_parameter("handle_input_topic").value
        self.handle_output_topic = self.get_parameter("handle_output_topic").value
        self.door_align_output_topic = self.get_parameter("door_align_output_topic").value
        self.door_prompt_topic = self.get_parameter("door_prompt_topic").value

        self.base_frame = self.get_parameter("base_frame").value
        self.use_msg_frame_id = bool(self.get_parameter("use_msg_frame_id").value)
        self.camera_frame_fallback = self.get_parameter("camera_frame_fallback").value
        self.use_msg_timestamp = bool(self.get_parameter("use_msg_timestamp").value)
        self.fallback_to_latest_tf = bool(self.get_parameter("fallback_to_latest_tf").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.door_aruco_id = int(self.get_parameter("door_aruco_id").value)
        self.door_label = str(self.get_parameter("door_label").value)
        self.door_text_prompt = str(self.get_parameter("door_text_prompt").value)
        self.door_selected_arm = str(self.get_parameter("door_selected_arm").value).strip().lower()
        if self.door_selected_arm not in ("left", "right"):
            self.get_logger().warn(
                f"Invalid door_selected_arm={self.door_selected_arm}. Falling back to right."
            )
            self.door_selected_arm = "right"
        self.door_shelf_type = str(self.get_parameter("door_shelf_type").value)

        self.align_offset_x = float(self.get_parameter("align_offset_x").value)
        self.align_offset_y = float(self.get_parameter("align_offset_y").value)
        self.align_offset_z = float(self.get_parameter("align_offset_z").value)
        self.right_align_qx = float(self.get_parameter("right_align_qx").value)
        self.right_align_qy = float(self.get_parameter("right_align_qy").value)
        self.right_align_qz = float(self.get_parameter("right_align_qz").value)
        self.right_align_qw = float(self.get_parameter("right_align_qw").value)

        self.publish_once_per_start = bool(self.get_parameter("publish_once_per_start").value)
        self.publish_repeat = max(1, int(self.get_parameter("publish_repeat").value))
        self.publish_repeat_period_sec = max(0.0, float(self.get_parameter("publish_repeat_period_sec").value))
        self.debug = bool(self.get_parameter("debug").value)
        self.print_tf_matrix = bool(self.get_parameter("print_tf_matrix").value)

        # ==================================================
        # Runtime state
        # ==================================================
        self.active = False
        self.published_this_start = False

        # ==================================================
        # QoS / TF
        # ==================================================
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================================================
        # Subscribers
        # ==================================================
        self.create_subscription(Bool, self.start_topic, self.start_cb, self.qos_cmd)
        self.create_subscription(String, self.door_prompt_topic, self.door_prompt_cb, self.qos_cmd)
        self.create_subscription(PointStamped, self.handle_input_topic, self.handle_callback, 10)

        # ==================================================
        # Publishers
        # ==================================================
        self.pub_handle_base = self.create_publisher(PointStamped, self.handle_output_topic, 10)
        self.pub_door_align = self.create_publisher(ObjectAlign, self.door_align_output_topic, self.qos_cmd)

        self.get_logger().info("========================================")
        self.get_logger().info("CALI_ZED_DOOR node started")
        self.get_logger().info("Master-controlled shelf_1 door-handle calibration enabled")
        self.get_logger().info(f"start_topic              : {self.start_topic}")
        self.get_logger().info(f"handle_input_topic       : {self.handle_input_topic}")
        self.get_logger().info(f"handle_output_topic      : {self.handle_output_topic}")
        self.get_logger().info(f"door_align_output_topic  : {self.door_align_output_topic}")
        self.get_logger().info(f"door_prompt_topic        : {self.door_prompt_topic}")
        self.get_logger().info(f"base_frame               : {self.base_frame}")
        self.get_logger().info(f"camera_frame_fallback    : {self.camera_frame_fallback}")
        self.get_logger().info(f"door_label               : {self.door_label}")
        self.get_logger().info(f"door_selected_arm        : {self.door_selected_arm}")
        self.get_logger().info(f"door_shelf_type          : {self.door_shelf_type}")
        self.get_logger().info("door ObjectAlign QoS     : RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth=1")
        self.get_logger().info("========================================")

    # ==================================================
    # Master callbacks
    # ==================================================
    def start_cb(self, msg: Bool):
        self.active = bool(msg.data)
        self.published_this_start = False

        if self.active:
            self.get_logger().info(
                f"[START] {self.start_topic} true. Waiting fresh door handle center. "
                f"prompt='{self.door_text_prompt}', arm={self.door_selected_arm}"
            )
        else:
            self.get_logger().info(f"[STOP] {self.start_topic} false. Door calibration paused.")

    def door_prompt_cb(self, msg: String):
        prompt = msg.data.strip()
        if prompt:
            self.door_text_prompt = prompt
            self.get_logger().info(f"[DOOR PROMPT UPDATED] '{self.door_text_prompt}'")

    # ==================================================
    # Handle callback
    # ==================================================
    def handle_callback(self, msg: PointStamped):
        if not self.active:
            return

        if self.publish_once_per_start and self.published_this_start:
            return

        p_base = self.transform_and_publish(msg, self.pub_handle_base, "SAM3-DOOR-HANDLE")
        if p_base is None:
            return

        self.publish_door_align(msg, p_base)
        self.published_this_start = True

        if self.publish_once_per_start:
            self.active = False
            self.get_logger().info("[CALI_ZED_DOOR] door ObjectAlign published once; auto-paused until next start true.")

    # ==================================================
    # Transform
    # ==================================================
    def transform_and_publish(self, msg: PointStamped, publisher, source_name="UNKNOWN"):
        source_frame = self.resolve_source_frame(msg)
        if source_frame is None:
            self.get_logger().warn(f"[{source_name}] source frame is empty and no fallback is available.")
            return None

        p_src = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0], dtype=np.float64)

        if self.debug:
            self.get_logger().info(
                f"[{source_name} INPUT] frame={source_frame}, "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                f"xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})"
            )

        tf_msg, tf_mode = self.lookup_transform(self.base_frame, source_frame, msg, source_name)
        if tf_msg is None:
            return None

        try:
            T_base_from_src = self.make_transform_matrix(tf_msg)
            p_base = T_base_from_src @ p_src
        except Exception as e:
            self.get_logger().error(f"[{source_name}] Transform application failed: {repr(e)}")
            return None

        out_msg = PointStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.point.x = float(p_base[0])
        out_msg.point.y = float(p_base[1])
        out_msg.point.z = float(p_base[2])
        publisher.publish(out_msg)

        self.get_logger().info(
            "\n"
            f"[{source_name} BASE TRANSFORM]\n"
            f"TF Mode     : {tf_mode}\n"
            f"SourceFrame : {source_frame}\n"
            f"Source XYZ  : [{p_src[0]:.4f}, {p_src[1]:.4f}, {p_src[2]:.4f}]\n"
            f"Base XYZ    : [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]"
        )

        if self.print_tf_matrix:
            self.print_tf_debug(tf_msg, source_name)

        return p_base

    # ==================================================
    # Door ObjectAlign
    # ==================================================
    def publish_door_align(self, original_msg: PointStamped, p_base: np.ndarray):
        out = ObjectAlign()
        out.header.stamp = original_msg.header.stamp
        out.header.frame_id = self.base_frame

        out.aruco_id = int(self.door_aruco_id)
        out.label = str(self.door_label)
        out.text_prompt = str(self.door_text_prompt)
        out.selected_arm = str(self.door_selected_arm)
        out.shelf_type = str(self.door_shelf_type)

        out.marker_position.x = float(p_base[0])
        out.marker_position.y = float(p_base[1])
        out.marker_position.z = float(p_base[2])
        out.align_pose = self.make_align_pose(p_base)

        for i in range(self.publish_repeat):
            self.pub_door_align.publish(out)
            self.get_logger().info(f"[DoorHandle ObjectAlign PUBLISH REPEAT] {i + 1}/{self.publish_repeat}")
            if i + 1 < self.publish_repeat and self.publish_repeat_period_sec > 0.0:
                time.sleep(self.publish_repeat_period_sec)

        self.get_logger().info(
            "\n"
            "[DoorHandle ObjectAlign PUBLISHED]\n"
            f"topic           : {self.door_align_output_topic}\n"
            f"aruco_id        : {out.aruco_id}\n"
            f"label           : {out.label}\n"
            f"text_prompt     : {out.text_prompt}\n"
            f"selected_arm    : {out.selected_arm}\n"
            f"shelf_type      : {out.shelf_type}\n"
            f"marker_position : ({out.marker_position.x:.4f}, {out.marker_position.y:.4f}, {out.marker_position.z:.4f})\n"
            f"align_position  : ({out.align_pose.position.x:.4f}, {out.align_pose.position.y:.4f}, {out.align_pose.position.z:.4f})\n"
            f"align_quat      : ({out.align_pose.orientation.x:.4f}, {out.align_pose.orientation.y:.4f}, "
            f"{out.align_pose.orientation.z:.4f}, {out.align_pose.orientation.w:.4f})"
        )

    def make_align_pose(self, p_base: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(p_base[0]) + self.align_offset_x
        pose.position.y = float(p_base[1]) + self.align_offset_y
        pose.position.z = float(p_base[2]) + self.align_offset_z

        pose.orientation.x = self.right_align_qx
        pose.orientation.y = self.right_align_qy
        pose.orientation.z = self.right_align_qz
        pose.orientation.w = self.right_align_qw
        return pose

    # ==================================================
    # TF utils
    # ==================================================
    def resolve_source_frame(self, msg: PointStamped):
        if self.use_msg_frame_id and msg.header.frame_id:
            return msg.header.frame_id
        if self.camera_frame_fallback:
            return self.camera_frame_fallback
        return None

    def lookup_transform(self, target_frame: str, source_frame: str, msg: PointStamped, source_name="UNKNOWN"):
        if self.use_msg_timestamp:
            try:
                target_time = Time.from_msg(msg.header.stamp)
                can_tx = self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
                if can_tx:
                    tf_msg = self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        target_time,
                        timeout=Duration(seconds=self.tf_timeout_sec),
                    )
                    if self.debug:
                        self.get_logger().info(
                            f"[{source_name} TF] using msg timestamp: "
                            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
                        )
                    return tf_msg, "msg_timestamp"

                self.get_logger().warn(
                    f"[{source_name}] can_transform failed with msg timestamp: {target_frame} <- {source_frame}"
                )

            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f"[{source_name}] TF lookup with msg timestamp failed: {e}")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f"[{source_name}] TF lookup failed with msg timestamp: {target_frame} <- {source_frame} | {e}"
                )
            except Exception as e:
                self.get_logger().error(f"[{source_name}] Unexpected TF error with msg stamp: {repr(e)}")

        if not self.fallback_to_latest_tf:
            return None, "none"

        try:
            can_tx = self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
            if not can_tx:
                self.get_logger().warn(f"[{source_name}] can_transform latest failed: {target_frame} <- {source_frame}")
                return None, "none"

            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
            if self.debug:
                tf_stamp = tf_msg.header.stamp
                self.get_logger().info(
                    f"[{source_name} TF] fallback to latest TF: {tf_stamp.sec}.{tf_stamp.nanosec:09d}"
                )
            return tf_msg, "latest"

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"[{source_name}] TF latest lookup failed: {target_frame} <- {source_frame} | {e}")
            return None, "none"
        except Exception as e:
            self.get_logger().error(f"[{source_name}] Unexpected TF error on latest lookup: {repr(e)}")
            return None, "none"

    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def print_tf_debug(self, tf_msg, source_name="UNKNOWN"):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self.get_logger().info(
            f"[{source_name} TF DETAIL] "
            f"translation=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}), "
            f"quaternion=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
        )
        self.get_logger().info(
            f"[{source_name} TF MATRIX]\n"
            f"[{rot[0,0]: .4f} {rot[0,1]: .4f} {rot[0,2]: .4f} | {t.x: .4f}]\n"
            f"[{rot[1,0]: .4f} {rot[1,1]: .4f} {rot[1,2]: .4f} | {t.y: .4f}]\n"
            f"[{rot[2,0]: .4f} {rot[2,1]: .4f} {rot[2,2]: .4f} | {t.z: .4f}]"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CaliZedDoorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()