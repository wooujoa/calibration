import cv2
import numpy as np
import glob
import os

# --- 설정 (기존과 동일) ---
CHESSBOARD_SIZE = (9, 6) 
SQUARE_SIZE = 30.0  
IMAGE_DIR = '/home/jwg/colcon_ws/src/calibration/calib_data' 

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points, img_points = [], []
images = glob.glob(os.path.join(IMAGE_DIR, '*.png'))

print(f"Found {len(images)} images. Starting calibration (Zero-Distortion Mode)...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, 
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    if ret:
        obj_points.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
    else:
        print(f"Failed to find corners in {fname}")

# --- [수정된 부분] 왜곡 계수 0으로 고정하여 캘리브레이션 수행 ---

# 1. 왜곡 계수 초기값을 0으로 설정
dist_coeffs_init = np.zeros((5, 1))

# 2. 모든 왜곡 계수를 0으로 고정하는 플래그 설정
# CALIB_FIX_K1~K3: 방사 왜곡 고정 / CALIB_ZERO_TANGENT_DIST: 접선 왜곡을 0으로 고정
flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | 
         cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT)

# 만약 주점(cx, cy)도 ROS 데이터와 똑같다고 가정하고 초점거리(fx, fy)만 구하고 싶다면 
# cv2.CALIB_FIX_PRINCIPAL_POINT를 추가하면 되지만, 보통은 주점까지는 새로 구해보는 편입니다.

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, dist_coeffs_init, flags=flags
)

# --- 결과 출력 (기존과 동일) ---
ros_K = np.array([
    [434.9008, 0.0,      428.4998],
    [0.0,      433.7109, 240.0578],
    [0.0,      0.0,      1.0     ]
])

print("\n" + "="*30)
print("1. Calibration Success (RMS Error):", ret)
print("\n2. Calculated K (from Rectified Images):")
print(mtx)
print("\n3. ROS2 /camera_info Matrix (P/K):")
print(ros_K)
print("\n4. Difference (Calculated - ROS2):")
print(mtx - ros_K)
print("\n5. Distortion Coefficients (Should be 0):")
print(dist.ravel())
print("="*30)