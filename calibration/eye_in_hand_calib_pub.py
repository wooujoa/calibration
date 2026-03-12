#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R

class ObjectBaseTransformer(Node):
    def __init__(self):
        super().__init__('object_base_transformer')

        # ----------------------------------------------------------------------
        # 1. 직접 구하신 Hand-Eye 캘리브레이션 행렬 (Camera -> Gripper)
        # 캘리브레이션 스크립트 결과로 나온 4x4 행렬을 아래에 붙여넣으세요.
        # ----------------------------------------------------------------------
        self.T_cam_to_gripper = np.array([
            [ 0.70655016,  -0.70427039,  -0.06921048,  0.302717], # 예시: 실제 결과값으로 수정필요
            [ -0.13697267,  -0.04014969,  -0.98976082,  0.01067388],
            [ 0.69428046,  0.70879561, -0.12483357, -0.08213032],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

        # 2. TF2 리스너 설정 (Gripper -> Base 실시간 위치 파악)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 3. 구독 및 발행
        # YOLO 노드로부터 카메라 기준 3D 좌표 구독
        self.sub_3d = self.create_subscription(
            PointStamped,
            '/yolo/target_3d_pose',
            self.point_callback,
            10)

        # 베이스 기준 좌표 발행 (최종 결과)
        self.pub_base_pose = self.create_publisher(PointStamped, '/yolo/target_base_pose', 10)

        self.get_logger().info("Object Base Transformer Node Initialized.")

    def point_callback(self, msg: PointStamped):
        # 1. 카메라 기준 좌표 (P_camera)
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0])

        # 2. 카메라 기준 -> 그리퍼 기준 변환 (Hand-Eye 적용)
        # P_gripper = T_cam_to_gripper * P_camera
        p_gripper = self.T_cam_to_gripper @ p_cam

        # 3. 그리퍼 기준 -> 베이스 기준 변환 (실시간 로봇 TF 적용)
        try:
            # base_link에서 arm_l_link7(또는 arm_r_link7) 사이의 최신 TF 조회
            # YOLO 메시지의 타임스탬프에 맞춰 동기화된 TF를 가져옵니다.
            now = rclpy.time.Time()
            t = self.tf_buffer.lookup_transform('base_link', 'arm_l_link7', now)

            # TF를 4x4 행렬로 변환
            T_gripper_to_base = self.make_transform_matrix(t)

            # 최종 베이스 좌표 계산
            # P_base = T_gripper_to_base * P_gripper
            p_base = T_gripper_to_base @ p_gripper

            # 4. 결과 메시지 생성 및 발행
            out_msg = PointStamped()
            out_msg.header = msg.header
            out_msg.header.frame_id = "base_link"
            out_msg.point.x = p_base[0]
            out_msg.point.y = p_base[1]
            out_msg.point.z = p_base[2]

            self.pub_base_pose.publish(out_msg)

            # 터미널 출력 (디버깅)
            self.get_logger().info(
                f"\n[Target Found]\n"
                f"Camera: [{msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}]\n"
                f"Base  : [{p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}]"
            )

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")

    def make_transform_matrix(self, transform):
        """Transform 메시지를 4x4 행렬로 변환하는 유틸리티"""
        q = transform.transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        
        t = transform.transform.translation
        
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = [t.x, t.y, t.z]
        return T

def main():
    rclpy.init()
    node = ObjectBaseTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()