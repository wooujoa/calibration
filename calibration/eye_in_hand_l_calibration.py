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
        super().__init__('calib_tester_node_left') # 노드 이름 변경
        self.bridge = CvBridge()
        
        # TF Listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 이미지 토픽 구독 (동일)
        self.subscription = self.create_subscription(
            Image, 
            '/camera_left/camera_left/color/image_rect_raw', 
            self.image_callback, 
            10)
        
        self.latest_cv_image = None
        self.latest_msg_stamp = None
        self.count = 0
        
        # 저장 경로 (오른팔 데이터와 섞이지 않도록 폴더명을 구분하는 것을 추천합니다)
        self.save_dir = '/home/jwg/colcon_ws/src/calibration/calib_data_left'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.get_logger().info("=== [LEFT ARM] Calibration Data Collection Started ===")
        self.get_logger().info("Target Frame: base_link -> arm_l_link7")
        self.get_logger().info("Press 'Space' to save data, 'q' to quit.")

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_msg_stamp = msg.header.stamp
            
            # 실시간 화면 출력
            display_img = self.latest_cv_image.copy()
            cv2.putText(display_img, f"LEFT Captured: {self.count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2) # 파란색 텍스트로 구분
            cv2.imshow("Camera Feed (Left Arm Calib)", display_img)
            
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space bar
                self.save_current_data()
            elif key == ord('q'):
                self.get_logger().info("Shutting down...")
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def save_current_data(self):
        if self.latest_cv_image is None:
            self.get_logger().warn("Waiting for image stream...")
            return

        # 1. 이미지 저장
        img_filename = f"{self.save_dir}/img_{self.count:02d}.png"
        cv2.imwrite(img_filename, self.latest_cv_image)
        
        # 2. TF 조회 (arm_l_link7 사용)
        tf_data = None
        status = "SUCCESS"
        
        try:
            # 타겟 프레임을 arm_l_link7로 변경
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'arm_l_link7',
                self.latest_msg_stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            pos = t.transform.translation
            rot = t.transform.rotation
            tf_data = {
                "translation": [pos.x, pos.y, pos.z],
                "rotation_quat": [rot.x, rot.y, rot.z, rot.w]
            }
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF Lookup Failed (Left Arm): {str(e)}")
            tf_data = None
            status = "TF_MISSING"

        # 3. JSON 저장
        json_filename = f"{self.save_dir}/pose_{self.count:02d}.json"
        save_payload = {
            "index": self.count,
            "status": status,
            "tf_info": tf_data,
            "frame_id": "base_link",
            "child_frame_id": "arm_l_link7"
        }
        
        with open(json_filename, 'w') as f:
            json.dump(save_payload, f, indent=4)
            
        self.get_logger().info(f"[{status}] Saved Left Arm Set #{self.count}")
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