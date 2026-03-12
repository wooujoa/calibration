import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tf2_ros
import json
import os

class CalibTesterNode(Node):
    def __init__(self):
        super().__init__('calib_tester_node')
        self.bridge = CvBridge()
        
        # TF Listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribe to rectified image topic
        self.subscription = self.create_subscription(
            Image, 
            '/camera_right/camera_right/color/image_rect_raw', 
            self.image_callback, 
            10)
        
        self.latest_cv_image = None
        self.latest_msg_stamp = None
        self.count = 0
        
        # Updated path based on your environment
        self.save_dir = '/home/jwg/colcon_ws/src/calibration/calib_data_right'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.get_logger().info("=== Calibration Data Collection Node Started ===")
        self.get_logger().info(f"Target Frame: base_link -> arm_r_link7")
        self.get_logger().info("Press 'Space' to save data, 'q' to quit.")

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_msg_stamp = msg.header.stamp
            
            # Real-time display for monitoring
            display_img = self.latest_cv_image.copy()
            cv2.putText(display_img, f"Captured Sets: {self.count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", display_img)
            
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space bar to capture
                self.save_current_data()
            elif key == ord('q'): # 'q' to exit
                self.get_logger().info("Shutting down...")
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def save_current_data(self):
        if self.latest_cv_image is None:
            self.get_logger().warn("Waiting for image stream...")
            return

        # 1. Save Image
        img_filename = f"{self.save_dir}/img_{self.count:02d}.png"
        cv2.imwrite(img_filename, self.latest_cv_image)
        
        # 2. Lookup Transform
        tf_data = None
        status = "SUCCESS"
        
        try:
            # Using arm_r_link7 as the verified end-effector frame
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'arm_r_link7',
                self.latest_msg_stamp,
                timeout=rclpy.duration.Duration(seconds=0.1) # Increased timeout for stability
            )
            
            pos = t.transform.translation
            rot = t.transform.rotation
            tf_data = {
                "translation": [pos.x, pos.y, pos.z],
                "rotation_quat": [rot.x, rot.y, rot.z, rot.w]
            }
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF Lookup Failed: {str(e)}")
            tf_data = None
            status = "TF_MISSING"

        # 3. Save to JSON
        json_filename = f"{self.save_dir}/pose_{self.count:02d}.json"
        save_payload = {
            "index": self.count,
            "status": status,
            "tf_info": tf_data,
            "frame_id": "base_link",
            "child_frame_id": "arm_r_link7"
        }
        
        with open(json_filename, 'w') as f:
            json.dump(save_payload, f, indent=4)
            
        self.get_logger().info(f"[{status}] Saved Set #{self.count}")
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = CalibTesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()