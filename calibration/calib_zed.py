#!/usr/bin/env python3
import os
import json
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import tf2_ros

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

class ZedCalibTesterNode(Node):
    def __init__(self):
        super().__init__('zed_calib_tester_node')

        self.bridge = CvBridge()

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter(
            'image_topic',
            '/zedm/zed_node/left/image_rect_color'
        )
        self.declare_parameter('use_compressed', False)

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'head_link2')
        self.declare_parameter('camera_frame', 'zedm_left_camera_optical_frame')

        self.declare_parameter(
            'save_dir',
            '/home/jwg/colcon_ws/src/calibration/calib_data_zed_aruco'
        )

        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('window_name', 'ZED ArUco Data Feed')
        self.declare_parameter('save_camera_tf_too', True)
        self.declare_parameter('save_debug_overlay', True)

        # ArUco params
        self.declare_parameter('detect_aruco', True)
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('target_marker_id', 1)

        self.image_topic = self.get_parameter('image_topic').value
        self.use_compressed = bool(self.get_parameter('use_compressed').value)

        self.base_frame = self.get_parameter('base_frame').value
        self.head_frame = self.get_parameter('head_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.save_dir = self.get_parameter('save_dir').value
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.window_name = self.get_parameter('window_name').value
        self.save_camera_tf_too = bool(self.get_parameter('save_camera_tf_too').value)
        self.save_debug_overlay = bool(self.get_parameter('save_debug_overlay').value)

        self.detect_aruco = bool(self.get_parameter('detect_aruco').value)
        self.aruco_dict_name = self.get_parameter('aruco_dict').value
        self.target_marker_id = int(self.get_parameter('target_marker_id').value)

        os.makedirs(self.save_dir, exist_ok=True)

        # Setup ArUco
        dict_id = DICT_MAP.get(self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self.use_new_detector_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_new_detector_api = False

        # -----------------------------
        # TF
        # -----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Image state
        # -----------------------------
        self.latest_cv_image: Optional[np.ndarray] = None
        self.latest_debug_image: Optional[np.ndarray] = None
        self.latest_msg_stamp = None
        self.latest_msg_frame_id: Optional[str] = None
        self.latest_marker_found: bool = False
        self.count = 0

        # -----------------------------
        # Subscriber
        # -----------------------------
        if self.use_compressed:
            self.subscription = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback_compressed,
                10
            )
        else:
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback_raw,
                10
            )

        self.get_logger().info("=== ZED ArUco Calibration Data Node Started ===")
        self.get_logger().info(f"image_topic       : {self.image_topic}")
        self.get_logger().info(f"use_compressed    : {self.use_compressed}")
        self.get_logger().info(f"base_frame        : {self.base_frame}")
        self.get_logger().info(f"head_frame        : {self.head_frame}")
        self.get_logger().info(f"save_dir          : {self.save_dir}")
        self.get_logger().info(f"detect_aruco      : {self.detect_aruco}")
        self.get_logger().info(f"target_marker_id  : {self.target_marker_id}")
        self.get_logger().info("Press 'Space' to save, 'q' to quit.")

    def image_callback_raw(self, msg: Image):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_msg_stamp = msg.header.stamp
            self.latest_msg_frame_id = msg.header.frame_id
            self.prepare_debug_view()
            self.show_live_feed()
        except Exception as e:
            self.get_logger().error(f"Raw image processing error: {repr(e)}")

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            img_np = cv2.imdecode(
                np.frombuffer(msg.data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img_np is None:
                return

            self.latest_cv_image = img_np
            self.latest_msg_stamp = msg.header.stamp
            self.latest_msg_frame_id = msg.header.frame_id
            self.prepare_debug_view()
            self.show_live_feed()
        except Exception as e:
            self.get_logger().error(f"Compressed image processing error: {repr(e)}")

    def prepare_debug_view(self):
        if self.latest_cv_image is None:
            return

        debug_img = self.latest_cv_image.copy()
        marker_found = False

        if self.detect_aruco:
            gray = cv2.cvtColor(self.latest_cv_image, cv2.COLOR_BGR2GRAY)
            
            if self.use_new_detector_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

            if ids is not None:
                for i, m_id in enumerate(ids.flatten()):
                    if m_id == self.target_marker_id:
                        marker_found = True
                        
                        corners_reshape = corners[i].reshape(1, 4, 2).astype(np.float32)
                        cv2.aruco.drawDetectedMarkers(
                            debug_img, 
                            [corners_reshape], 
                            np.array([self.target_marker_id], dtype=np.int32)
                        )
                        
                        c0 = corners[i][0][0]
                        cv2.circle(debug_img, (int(c0[0]), int(c0[1])), 6, (0, 0, 255), -1)
                        break

        self.latest_debug_image = debug_img
        self.latest_marker_found = marker_found

    def show_live_feed(self):
        if self.latest_cv_image is None:
            return

        display_img = (
            self.latest_debug_image.copy()
            if self.latest_debug_image is not None
            else self.latest_cv_image.copy()
        )

        cv2.putText(display_img, f"Saved Sets: {self.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_img, f"base: {self.base_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_img, f"head: {self.head_frame}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        status_text = f"ArUco ID {self.target_marker_id}: FOUND" if self.latest_marker_found else f"ArUco ID {self.target_marker_id}: NOT FOUND"
        status_color = (0, 255, 0) if self.latest_marker_found else (0, 0, 255)
        cv2.putText(display_img, status_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow(self.window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self.save_current_data()
        elif key == ord('q'):
            self.get_logger().info("Shutting down...")
            rclpy.shutdown()

    def save_current_data(self):
        if self.latest_cv_image is None or self.latest_msg_stamp is None:
            return

        idx = self.count
        img_filename = os.path.join(self.save_dir, f'img_{idx:02d}.png')
        cv2.imwrite(img_filename, self.latest_cv_image)

        dbg_filename = None
        if self.save_debug_overlay and self.latest_debug_image is not None:
            dbg_filename = os.path.join(self.save_dir, f'img_{idx:02d}_debug.png')
            cv2.imwrite(dbg_filename, self.latest_debug_image)

        status = "SUCCESS"
        head_tf = self.lookup_transform_dict(self.base_frame, self.head_frame, self.latest_msg_stamp)
        
        cam_tf = None
        if self.save_camera_tf_too:
            cam_tf = self.lookup_transform_dict(self.base_frame, self.camera_frame, self.latest_msg_stamp)

        if head_tf is None: status = "HEAD_TF_MISSING"
        if self.save_camera_tf_too and cam_tf is None and status == "SUCCESS": status = "CAMERA_TF_MISSING"

        json_filename = os.path.join(self.save_dir, f'pose_{idx:02d}.json')
        payload = {
            "index": idx,
            "status": status,
            "image_file": os.path.basename(img_filename),
            "debug_image_file": os.path.basename(dbg_filename) if dbg_filename else None,
            "image_stamp": {
                "sec": int(self.latest_msg_stamp.sec),
                "nanosec": int(self.latest_msg_stamp.nanosec),
            },
            "image_frame_id": self.latest_msg_frame_id,
            "base_frame": self.base_frame,
            "head_frame": self.head_frame,
            "camera_frame": self.camera_frame,
            "aruco": {
                "enabled": self.detect_aruco,
                "found": self.latest_marker_found,
                "dict": self.aruco_dict_name,
                "target_id": self.target_marker_id
            },
            "tf_base_to_head": head_tf,
            "tf_base_to_camera": cam_tf,
        }

        with open(json_filename, 'w') as f:
            json.dump(payload, f, indent=4)

        self.get_logger().info(f"[{status}] Saved Set #{idx} | ArUco_found={self.latest_marker_found}")
        self.count += 1

    def lookup_transform_dict(self, target_frame: str, source_frame: str, stamp) -> Optional[Dict[str, Any]]:
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rclpy.time.Time.from_msg(stamp), 
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
            pos, rot = t.transform.translation, t.transform.rotation
            return {
                "parent_frame": target_frame,
                "child_frame": source_frame,
                "translation": [pos.x, pos.y, pos.z],
                "rotation_quat": [rot.x, rot.y, rot.z, rot.w],
                "tf_stamp": {"sec": int(t.header.stamp.sec), "nanosec": int(t.header.stamp.nanosec)}
            }
        except Exception as e:
            self.get_logger().error(f"TF Lookup Failed: {target_frame} <- {source_frame} | {str(e)}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = ZedCalibTesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()