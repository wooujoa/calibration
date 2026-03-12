import cv2
import numpy as np
import glob
import json
import os
from scipy.spatial.transform import Rotation as R

# 1. 설정 및 파라미터
K = np.array([
    [434.90081787109375, 0.0,                428.49981689453125],
    [0.0,                433.7109375,        240.05780029296875],
    [0.0,                0.0,                1.0               ]
])
dist = np.zeros(5) # image_rect_raw 사용 시 0

pattern_size = (9, 6) # 내부 코너 개수
square_size = 0.025   # 미터(m) 단위
data_dir = '/home/jwg/colcon_ws/src/calibration/calib_data_left'

# 데이터를 저장할 리스트
R_gripper2base, t_gripper2base = [], []
R_target2cam, t_target2cam = [], []

# 파일 목록 (번호순 정렬)
img_files = sorted(glob.glob(os.path.join(data_dir, 'img_*.png')))

print(f"Starting analysis on {len(img_files)} data pairs...")

for img_path in img_files:
    # 1. 인덱스 추출 (파일명에서 숫자 추출)
    idx_str = os.path.basename(img_path).replace('img_', '').replace('.png', '')
    json_path = os.path.join(data_dir, f'pose_{idx_str}.json')
    
    if not os.path.exists(json_path):
        continue

    # 2. 이미지 및 JSON 읽기
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    with open(json_path, 'r') as f:
        pose_data = json.load(f)
    
    # TF가 누락된 데이터는 건너뜀
    if pose_data.get('status') != "SUCCESS" or pose_data.get('tf_info') is None:
        print(f"[{idx_str}] Skipping: TF data missing.")
        continue

    # 3. 이미지에서 체스판 찾기 -> T_target2cam 구하기
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # 코너 정밀화
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # solvePnP로 카메라 기준 체스판의 R, t 계산
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        
        _, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
        R_mat_target2cam, _ = cv2.Rodrigues(rvec)
        
        # 4. JSON에서 로봇 포즈 가져와 변환 -> T_gripper2base
        tf = pose_data['tf_info']
        t_g2b = np.array(tf['translation']).reshape(3, 1)
        q_g2b = tf['rotation_quat'] # [x, y, z, w]
        
        # 쿼터니언 -> 3x3 회전 행렬 변환
        rot = R.from_quat(q_g2b)
        R_mat_gripper2base = rot.as_matrix()

        # 리스트에 추가
        R_target2cam.append(R_mat_target2cam)
        t_target2cam.append(tvec)
        R_gripper2base.append(R_mat_gripper2base)
        t_gripper2base.append(t_g2b)
        
        print(f"[{idx_str}] Success: corners found and TF parsed.")
    else:
        print(f"[{idx_str}] Failed: corners not found.")

# 5. 최종 Hand-Eye Calibration 실행
if len(R_target2cam) < 3:
    print("Error: Need at least 3 successful pairs for calibration.")
else:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, 
        R_target2cam, t_target2cam, 
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 6. 결과 출력 (4x4 Matrix)
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.reshape(3)

    print("\n" + "="*50)
    print("Final Hand-Eye Transformation Matrix (Camera to Gripper):")
    print(np.array2string(T_cam2gripper, separator=', '))
    print("="*50)
    
    # 결과 해석
    print(f"\nTranslation (x, y, z in meters): {t_cam2gripper.reshape(3)}")
    
    
"""
-0.0958 & -0.9954 & 0.0000 & 0.0982 
 0.0000 & 0.0000 & 1.0000 & 0.0000 
-0.9954 & 0.0958 & 0.0000 & -0.0725 
 0.0000 & 0.0000 & 0.0000 & 1.0000 
"""

