#!/usr/bin/env python3

import os
import yaml
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PointStamped, Pose
import tf2_ros
from scipy.spatial.transform import Rotation as R

from grasp_msgs.msg import ObjectAlign


class CaliZedNode(Node):
    """
    CALI_ZED node for master_2.

    Subscribe:
      /aruco_zed_start        std_msgs/Bool
      /target_item_name       std_msgs/String
      /target_aruco_id        std_msgs/Int32
      /target_text_prompt     std_msgs/String
      /target_shelf_id        std_msgs/String
      /aruco/marker_3d_zed    geometry_msgs/PointStamped

    Publish:
      /aruco/marker_base_pose_zed  geometry_msgs/PointStamped
      /object_align_result         grasp_msgs/ObjectAlign

    Required ObjectAlign.msg fields:
      std_msgs/Header header
      int32 aruco_id
      string label
      string text_prompt
      string selected_arm
      string shelf_type
      geometry_msgs/Point marker_position
      geometry_msgs/Pose align_pose

    Fixed end-effector orientation:
      right = (x=0.5, y=-0.5, z=0.5, w=-0.5)
      left  = (x=0.5, y= 0.5, z=0.5, w= 0.5)
    """

    def __init__(self):
        super().__init__("cali_zed_node")

        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter("start_topic", "/aruco_zed_start")

        self.declare_parameter("target_item_topic", "/target_item_name")
        self.declare_parameter("target_aruco_id_topic", "/target_aruco_id")
        self.declare_parameter("target_text_prompt_topic", "/target_text_prompt")
        self.declare_parameter("target_shelf_id_topic", "/target_shelf_id")

        self.declare_parameter("aruco_input_topic", "/aruco/marker_3d_zed")
        self.declare_parameter("aruco_output_topic", "/aruco/marker_base_pose_zed")
        self.declare_parameter("object_align_output_topic", "/object_align_result")

        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("use_msg_frame_id", True)
        self.declare_parameter("camera_frame_fallback", "zedm_left_camera_optical_frame")

        self.declare_parameter("use_msg_timestamp", True)
        self.declare_parameter("fallback_to_latest_tf", True)
        self.declare_parameter("tf_timeout_sec", 0.2)

        # item DB fallback. If unavailable, node still works with target topics from master.
        self.declare_parameter("use_item_database", True)
        self.declare_parameter("item_database_path", "")
        self.declare_parameter("config_package", "master_capstone")
        self.declare_parameter("item_database_file", "item_database.yaml")

        # align_pose position = marker_position + offset
        self.declare_parameter("align_offset_x", 0.0)
        self.declare_parameter("align_offset_y", 0.0)
        self.declare_parameter("align_offset_z", 0.0)

        # align_pose orientation fixed quaternion by selected arm
        self.declare_parameter("right_align_qx", 0.5)
        self.declare_parameter("right_align_qy", -0.5)
        self.declare_parameter("right_align_qz", 0.5)
        self.declare_parameter("right_align_qw", -0.5)

        self.declare_parameter("left_align_qx", 0.5)
        self.declare_parameter("left_align_qy", -0.5)
        self.declare_parameter("left_align_qz", 0.5)
        self.declare_parameter("left_align_qw", -0.5)

        self.declare_parameter("debug", True)
        self.declare_parameter("print_tf_matrix", False)

        # Repeated INIT2 safety: publish only one ObjectAlign per start pulse.
        self.declare_parameter("publish_once_per_start", True)

        # ==================================================
        # Fetch Parameters
        # ==================================================
        self.start_topic = self.get_parameter("start_topic").value

        self.target_item_topic = self.get_parameter("target_item_topic").value
        self.target_aruco_id_topic = self.get_parameter("target_aruco_id_topic").value
        self.target_text_prompt_topic = self.get_parameter("target_text_prompt_topic").value
        self.target_shelf_id_topic = self.get_parameter("target_shelf_id_topic").value

        self.aruco_input_topic = self.get_parameter("aruco_input_topic").value
        self.aruco_output_topic = self.get_parameter("aruco_output_topic").value
        self.object_align_output_topic = self.get_parameter("object_align_output_topic").value

        self.base_frame = self.get_parameter("base_frame").value
        self.use_msg_frame_id = bool(self.get_parameter("use_msg_frame_id").value)
        self.camera_frame_fallback = self.get_parameter("camera_frame_fallback").value
        self.use_msg_timestamp = bool(self.get_parameter("use_msg_timestamp").value)
        self.fallback_to_latest_tf = bool(self.get_parameter("fallback_to_latest_tf").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.use_item_database = bool(self.get_parameter("use_item_database").value)

        self.align_offset_x = float(self.get_parameter("align_offset_x").value)
        self.align_offset_y = float(self.get_parameter("align_offset_y").value)
        self.align_offset_z = float(self.get_parameter("align_offset_z").value)

        self.right_align_qx = float(self.get_parameter("right_align_qx").value)
        self.right_align_qy = float(self.get_parameter("right_align_qy").value)
        self.right_align_qz = float(self.get_parameter("right_align_qz").value)
        self.right_align_qw = float(self.get_parameter("right_align_qw").value)

        self.left_align_qx = float(self.get_parameter("left_align_qx").value)
        self.left_align_qy = float(self.get_parameter("left_align_qy").value)
        self.left_align_qz = float(self.get_parameter("left_align_qz").value)
        self.left_align_qw = float(self.get_parameter("left_align_qw").value)

        self.debug = bool(self.get_parameter("debug").value)
        self.print_tf_matrix = bool(self.get_parameter("print_tf_matrix").value)
        self.publish_once_per_start = bool(self.get_parameter("publish_once_per_start").value)

        # ==================================================
        # Runtime state
        # ==================================================
        self.active = False
        self.published_this_start = False

        self.current_item = None
        self.current_item_name = ""
        self.current_aruco_id = -1
        self.current_text_prompt = ""
        self.current_shelf_type = ""

        # ==================================================
        # QoS
        # ==================================================
        # Command / one-shot important result QoS.
        # IMPORTANT:
        #   /object_align_result publisher uses this QoS, so arm_picking subscriber
        #   must use RELIABLE + TRANSIENT_LOCAL + KEEP_LAST + depth=1.
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================================================
        # DB
        # ==================================================
        self.item_db = {}
        self.db_metadata = {}
        if self.use_item_database:
            self.item_db, self.db_metadata = self.load_item_database()

        # ==================================================
        # TF
        # ==================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================================================
        # Subscribers
        # ==================================================
        self.create_subscription(Bool, self.start_topic, self.start_cb, self.qos_cmd)
        self.create_subscription(String, self.target_item_topic, self.target_item_cb, self.qos_cmd)
        self.create_subscription(Int32, self.target_aruco_id_topic, self.target_aruco_id_cb, self.qos_cmd)
        self.create_subscription(String, self.target_text_prompt_topic, self.target_text_prompt_cb, self.qos_cmd)
        self.create_subscription(String, self.target_shelf_id_topic, self.target_shelf_id_cb, self.qos_cmd)

        # Aruco marker stream can remain default QoS.
        self.create_subscription(PointStamped, self.aruco_input_topic, self.aruco_callback, 10)

        # ==================================================
        # Publishers
        # ==================================================
        # Debug/transformed marker output; not critical, keep default QoS.
        self.pub_aruco_base = self.create_publisher(PointStamped, self.aruco_output_topic, 10)

        # Important one-shot ObjectAlign output. Must match arm_picking subscriber QoS.
        self.pub_object_align = self.create_publisher(
            ObjectAlign,
            self.object_align_output_topic,
            self.qos_cmd,
        )

        self.get_logger().info("========================================")
        self.get_logger().info("CALI_ZED node started")
        self.get_logger().info("Master-controlled ZED calibration + ObjectAlign publish enabled")
        self.get_logger().info(f"start_topic                : {self.start_topic}")
        self.get_logger().info(f"target_item_topic          : {self.target_item_topic}")
        self.get_logger().info(f"target_aruco_id_topic      : {self.target_aruco_id_topic}")
        self.get_logger().info(f"target_text_prompt_topic   : {self.target_text_prompt_topic}")
        self.get_logger().info(f"target_shelf_id_topic      : {self.target_shelf_id_topic}")
        self.get_logger().info(f"aruco_input_topic          : {self.aruco_input_topic}")
        self.get_logger().info(f"aruco_output_topic         : {self.aruco_output_topic}")
        self.get_logger().info(f"object_align_output_topic  : {self.object_align_output_topic}")
        self.get_logger().info("object_align QoS           : RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth=1")
        self.get_logger().info(f"base_frame                 : {self.base_frame}")
        self.get_logger().info(f"align_offset               : ({self.align_offset_x}, {self.align_offset_y}, {self.align_offset_z})")
        self.get_logger().info(
            f"right_align_quaternion     : ({self.right_align_qx}, {self.right_align_qy}, "
            f"{self.right_align_qz}, {self.right_align_qw})"
        )
        self.get_logger().info(
            f"left_align_quaternion      : ({self.left_align_qx}, {self.left_align_qy}, "
            f"{self.left_align_qz}, {self.left_align_qw})"
        )
        self.get_logger().info("ARM RULE: base_link y < 0 -> right, y >= 0 -> left")
        self.get_logger().info("SHELF METADATA: item_database.yaml shelf_id -> ObjectAlign.shelf_type")
        if self.db_metadata:
            self.get_logger().info(f"DB metadata                : {self.db_metadata}")
        self.get_logger().info("========================================")

    # ==================================================
    # DB
    # ==================================================
    def resolve_database_path(self):
        explicit_path = str(self.get_parameter("item_database_path").value).strip()
        if explicit_path:
            return os.path.expanduser(explicit_path)

        config_package = self.get_parameter("config_package").value
        item_database_file = self.get_parameter("item_database_file").value

        package_share = get_package_share_directory(config_package)
        return os.path.join(package_share, "config", item_database_file)

    def load_item_database(self):
        try:
            yaml_path = self.resolve_database_path()
        except Exception as e:
            self.get_logger().warn(f"Cannot resolve item database path: {repr(e)}")
            return {}, {}

        if not os.path.exists(yaml_path):
            self.get_logger().warn(
                f"item_database.yaml not found: {yaml_path}. "
                f"CALI_ZED can still work with master target topics."
            )
            return {}, {}

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            items = data.get("items", {}) or {}
            metadata = data.get("metadata", {}) or {}
            self.get_logger().info(f"Loaded item database: {yaml_path}")
            self.get_logger().info(f"Number of items: {len(items)}")
            return items, metadata

        except Exception as e:
            self.get_logger().warn(f"Failed to load item database: {repr(e)}")
            return {}, {}

    def find_item_by_name(self, query: str):
        q = query.strip().lower()
        if not q:
            return None

        for item_key, info in self.item_db.items():
            candidates = [
                str(item_key),
                str(info.get("item_id", "")),
                str(info.get("product_name", "")),
            ]

            for alias in info.get("aliases", []):
                candidates.append(str(alias))

            candidates_norm = [
                c.strip().lower()
                for c in candidates
                if c is not None and str(c).strip() != ""
            ]

            if q in candidates_norm:
                out = dict(info)
                out["item_key"] = item_key
                return out

        return None

    # ==================================================
    # Master callbacks
    # ==================================================
    def start_cb(self, msg: Bool):
        self.active = bool(msg.data)

        if self.active:
            self.published_this_start = False
            self.get_logger().info(
                f"[START] {self.start_topic} true. "
                f"item='{self.current_item_name}', "
                f"aruco_id={self.current_aruco_id}, "
                f"prompt='{self.current_text_prompt}', "
                f"shelf_type='{self.current_shelf_type}'"
            )
        else:
            self.published_this_start = False
            self.get_logger().info(f"[STOP] {self.start_topic} false. ObjectAlign publish paused.")

    def target_item_cb(self, msg: String):
        item_name = msg.data.strip()
        self.current_item_name = item_name

        item = self.find_item_by_name(item_name) if self.item_db else None

        if item is None:
            self.current_item = {
                "item_key": item_name,
                "item_id": item_name,
                "product_name": item_name,
            }

            # Keep shelf_type from /target_shelf_id if master publishes it.
            self.get_logger().info(
                f"[TARGET ITEM] '{item_name}' received. "
                f"No DB match. Will use master /target_aruco_id, /target_text_prompt, /target_shelf_id. "
                f"current shelf_type='{self.current_shelf_type}'"
            )
            return

        self.current_item = item

        # Always update from DB. This is required for repeated INIT2 cycles.
        if "aruco_id" in item:
            self.current_aruco_id = int(item["aruco_id"])

        if "text_prompt" in item:
            self.current_text_prompt = str(item["text_prompt"])

        # Current ObjectAlign.msg uses shelf_type only.
        # The DB key may still be named shelf_id, so map DB shelf_id -> ObjectAlign.shelf_type.
        self.current_shelf_type = str(item.get("shelf_type", item.get("shelf_id", "")))

        self.get_logger().info(
            "\n"
            "[TARGET ITEM UPDATED]\n"
            f"input        : {item_name}\n"
            f"item_key     : {item.get('item_key')}\n"
            f"product_name : {item.get('product_name')}\n"
            f"text_prompt  : {self.current_text_prompt}\n"
            f"aruco_id     : {self.current_aruco_id}\n"
            f"shelf_type   : {self.current_shelf_type}"
        )

    def target_aruco_id_cb(self, msg: Int32):
        self.current_aruco_id = int(msg.data)
        self.get_logger().info(f"[TARGET ARUCO ID UPDATED] aruco_id={self.current_aruco_id}")

    def target_text_prompt_cb(self, msg: String):
        self.current_text_prompt = msg.data.strip()
        self.get_logger().info(f"[TARGET TEXT PROMPT UPDATED] text_prompt='{self.current_text_prompt}'")

    def target_shelf_id_cb(self, msg: String):
        # Master DB field name is shelf_id, ObjectAlign field name is shelf_type.
        shelf_type = msg.data.strip()
        if shelf_type:
            self.current_shelf_type = shelf_type
        self.get_logger().info(f"[TARGET SHELF UPDATED] shelf_type='{self.current_shelf_type}'")

    # ==================================================
    # Aruco callback
    # ==================================================
    def aruco_callback(self, msg: PointStamped):
        if not self.active:
            return

        if self.publish_once_per_start and self.published_this_start:
            return

        if self.current_aruco_id < 0:
            self.get_logger().warn(
                "[CALI_ZED] current_aruco_id is not set. "
                "Waiting /target_aruco_id or valid DB item."
            )
            return

        p_base = self.transform_and_publish(
            msg,
            self.pub_aruco_base,
            "ARUCO-ZED",
        )

        if p_base is None:
            return

        self.publish_object_align(msg, p_base)
        self.published_this_start = True

        if self.publish_once_per_start:
            self.active = False
            self.get_logger().info("[CALI_ZED] ObjectAlign published once; auto-paused until next start true.")

    # ==================================================
    # Transform
    # ==================================================
    def transform_and_publish(self, msg: PointStamped, publisher, source_name="UNKNOWN"):
        source_frame = self.resolve_source_frame(msg)
        if source_frame is None:
            self.get_logger().warn(
                f"[{source_name}] source frame is empty and no fallback is available."
            )
            return None

        p_src = np.array(
            [msg.point.x, msg.point.y, msg.point.z, 1.0],
            dtype=np.float64,
        )

        if self.debug:
            self.get_logger().info(
                f"[{source_name} INPUT] "
                f"frame={source_frame}, "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                f"xyz=({msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f})"
            )

        tf_msg, tf_mode = self.lookup_transform(
            self.base_frame,
            source_frame,
            msg,
            source_name,
        )

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
    # ObjectAlign
    # ==================================================
    def publish_object_align(self, original_msg: PointStamped, p_base: np.ndarray):
        selected_arm = "right" if float(p_base[1]) < 0.0 else "left"
        label = self.resolve_label()

        out = ObjectAlign()
        out.header.stamp = original_msg.header.stamp
        out.header.frame_id = self.base_frame

        out.aruco_id = int(self.current_aruco_id)
        out.label = label
        out.text_prompt = str(self.current_text_prompt)
        out.selected_arm = selected_arm

        # item_database.yaml uses shelf_id, ObjectAlign.msg uses shelf_type.
        # Actual values must be "shelf_1" or "shelf_2" according to DB.
        out.shelf_type = str(self.current_shelf_type)

        out.marker_position.x = float(p_base[0])
        out.marker_position.y = float(p_base[1])
        out.marker_position.z = float(p_base[2])

        out.align_pose = self.make_align_pose(p_base, selected_arm)

        for i in range(5):
            self.pub_object_align.publish(out)
            self.get_logger().info(
                f"[ObjectAlign PUBLISH REPEAT] {i + 1}/5"
            )
            time.sleep(0.05)

        self.get_logger().info(
            "\n"
            "[ObjectAlign PUBLISHED]\n"
            f"topic           : {self.object_align_output_topic}\n"
            f"qos             : RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth=1\n"
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

    def resolve_label(self) -> str:
        if self.current_item is not None:
            if "item_id" in self.current_item:
                return str(self.current_item["item_id"])
            if "item_key" in self.current_item:
                return str(self.current_item["item_key"])

        if self.current_item_name:
            return self.current_item_name

        return "unknown"

    def make_align_pose(self, p_base: np.ndarray, selected_arm: str) -> Pose:
        pose = Pose()

        pose.position.x = float(p_base[0]) + self.align_offset_x
        pose.position.y = float(p_base[1]) + self.align_offset_y
        pose.position.z = float(p_base[2]) + self.align_offset_z

        arm = (selected_arm or "").strip().lower()
        if arm == "left":
            pose.orientation.x = self.left_align_qx
            pose.orientation.y = self.left_align_qy
            pose.orientation.z = self.left_align_qz
            pose.orientation.w = self.left_align_qw
        else:
            pose.orientation.x = self.right_align_qx
            pose.orientation.y = self.right_align_qy
            pose.orientation.z = self.right_align_qz
            pose.orientation.w = self.right_align_qw

        return pose

    # ==================================================
    # TF utils
    # ==================================================
    def resolve_source_frame(self, msg: PointStamped):
        if self.use_msg_frame_id:
            if msg.header.frame_id is not None and msg.header.frame_id != "":
                return msg.header.frame_id

        if self.camera_frame_fallback is not None and self.camera_frame_fallback != "":
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
                    f"[{source_name}] can_transform failed with msg timestamp: "
                    f"{target_frame} <- {source_frame}"
                )

            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f"[{source_name}] TF lookup with msg timestamp failed: {e}"
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(
                    f"[{source_name}] TF lookup failed with msg timestamp: "
                    f"{target_frame} <- {source_frame} | {e}"
                )
            except Exception as e:
                self.get_logger().error(
                    f"[{source_name}] Unexpected TF error with msg stamp: {repr(e)}"
                )

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
                self.get_logger().warn(
                    f"[{source_name}] can_transform latest failed: "
                    f"{target_frame} <- {source_frame}"
                )
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
                    f"[{source_name} TF] fallback to latest TF: "
                    f"{tf_stamp.sec}.{tf_stamp.nanosec:09d}"
                )

            return tf_msg, "latest"

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"[{source_name}] TF latest lookup failed: {target_frame} <- {source_frame} | {e}"
            )
            return None, "none"
        except Exception as e:
            self.get_logger().error(
                f"[{source_name}] Unexpected TF error on latest lookup: {repr(e)}"
            )
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
    node = CaliZedNode()

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