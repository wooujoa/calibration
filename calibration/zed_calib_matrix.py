#!/usr/bin/env python3
import os
import json
import glob
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# =========================================================
# Basic transform utils
# =========================================================
def make_transform(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t.reshape(3)
    return T

def invert_transform(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv

def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=np.float64)])
    out = (T @ pts_h.T).T
    return out[:, :3]

def se3_to_matrix(x: np.ndarray) -> np.ndarray:
    t = x[:3]
    rvec = x[3:6]
    Rm = R.from_rotvec(rvec).as_matrix()
    return make_transform(Rm, t)

def matrix_to_se3(T: np.ndarray) -> np.ndarray:
    t = T[:3, 3]
    rvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.hstack([t, rvec])

# =========================================================
# ArUco utils
# =========================================================
def build_aruco_object_points(marker_size: float) -> np.ndarray:
    s = marker_size / 2.0
    return np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float64)

def detect_target_aruco(
    image_bgr: np.ndarray,
    aruco_dict_name: str,
    target_id: int
) -> Tuple[bool, Optional[np.ndarray]]:
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dict_id = DICT_MAP.get(aruco_dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    try:
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        detector_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == target_id:
                return True, corners[i][0].astype(np.float64)
                
    return False, None

def save_debug_detection(out_path: str, image: np.ndarray, corners: np.ndarray, target_id: int):
    dbg = image.copy()
    corners_reshape = corners.reshape(1, 4, 2).astype(np.float32)
    cv2.aruco.drawDetectedMarkers(
        dbg, 
        [corners_reshape], 
        np.array([target_id], dtype=np.int32)
    )
    cv2.imwrite(out_path, dbg)

# =========================================================
# Data loading
# =========================================================
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def extract_T_base_to_head(data: dict) -> np.ndarray:
    tf = data["tf_base_to_head"]
    tx, ty, tz = tf["translation"]
    qx, qy, qz, qw = tf["rotation_quat"]
    Rm = quat_xyzw_to_rot(qx, qy, qz, qw)
    return make_transform(Rm, np.array([tx, ty, tz], dtype=np.float64))

# =========================================================
# Projection
# =========================================================
def project_points_cam(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (x / z) + K[0, 2]
    uv[:, 1] = K[1, 1] * (y / z) + K[1, 2]
    return uv

def compute_rmse_from_proj(proj: np.ndarray, corners_2d: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((proj - corners_2d) ** 2, axis=1))))

# =========================================================
# Model
# =========================================================
def compose_points_in_cam(
    obj_pts_board: np.ndarray,
    T_base_to_head: np.ndarray,
    T_head_to_cam: np.ndarray,
    T_base_to_board: np.ndarray,
    use_inverse_head: bool = True
) -> np.ndarray:
    pts_base = transform_points(T_base_to_board, obj_pts_board)

    if use_inverse_head:
        T_head_to_base = invert_transform(T_base_to_head)
        pts_head = transform_points(T_head_to_base, pts_base)
    else:
        pts_head = transform_points(T_base_to_head, pts_base)

    pts_cam = transform_points(T_head_to_cam, pts_head)
    return pts_cam

def residuals_fn(
    params: np.ndarray,
    samples: List[dict],
    obj_pts_board: np.ndarray,
    K: np.ndarray,
    use_inverse_head: bool
) -> np.ndarray:
    x_hc = params[:6]
    x_bb = params[6:12]

    T_head_to_cam = se3_to_matrix(x_hc)
    T_base_to_board = se3_to_matrix(x_bb)

    all_res = []

    for s in samples:
        T_base_to_head = s["T_base_to_head"]
        corners_2d = s["corners_2d"]

        pts_cam = compose_points_in_cam(
            obj_pts_board=obj_pts_board,
            T_base_to_head=T_base_to_head,
            T_head_to_cam=T_head_to_cam,
            T_base_to_board=T_base_to_board,
            use_inverse_head=use_inverse_head
        )

        z = pts_cam[:, 2]

        z_safe = np.maximum(z, 1e-3)
        proj = project_points_cam(
            np.column_stack([pts_cam[:, 0], pts_cam[:, 1], z_safe]),
            K
        )

        reproj_res = (proj - corners_2d).reshape(-1)

        neg_penalty = np.maximum(0.0, 0.05 - z) * 200.0
        neg_penalty = np.repeat(neg_penalty, 2)

        huge = np.abs(reproj_res) > 3000.0
        reproj_res[huge] = np.sign(reproj_res[huge]) * 3000.0

        res = reproj_res + neg_penalty
        all_res.append(res)

    return np.concatenate(all_res, axis=0)

def compute_rmse(
    params: np.ndarray,
    samples: List[dict],
    obj_pts_board: np.ndarray,
    K: np.ndarray,
    use_inverse_head: bool
) -> float:
    res = residuals_fn(params, samples, obj_pts_board, K, use_inverse_head)
    return float(np.sqrt(np.mean(res ** 2)))

# =========================================================
# Debug helpers
# =========================================================
def debug_first_sample(
    first_sample: dict,
    obj_pts_board: np.ndarray,
    K: np.ndarray,
    T_head_to_cam_init: np.ndarray,
    T_base_to_board_init: np.ndarray,
    out_dir: str,
    target_id: int,
    use_inverse_head: bool
):
    print("\n================ DEBUG FIRST SAMPLE ================")
    print(f"[DEBUG] json_path  : {first_sample['json_path']}")
    print(f"[DEBUG] image_path : {first_sample['image_path']}")

    T_base_to_head = first_sample["T_base_to_head"]
    corners_2d = first_sample["corners_2d"]
    image = first_sample["image"]

    print("[DEBUG] T_base_to_head =")
    print(np.array2string(T_base_to_head, precision=6, suppress_small=False))

    print("[DEBUG] T_head_to_cam_init =")
    print(np.array2string(T_head_to_cam_init, precision=6, suppress_small=False))

    print("[DEBUG] T_base_to_board_init =")
    print(np.array2string(T_base_to_board_init, precision=6, suppress_small=False))

    pts_cam = compose_points_in_cam(
        obj_pts_board=obj_pts_board,
        T_base_to_head=T_base_to_head,
        T_head_to_cam=T_head_to_cam_init,
        T_base_to_board=T_base_to_board_init,
        use_inverse_head=use_inverse_head
    )

    zmin = float(np.min(pts_cam[:, 2]))
    zmax = float(np.max(pts_cam[:, 2]))
    print(f"[DEBUG] projected marker z range in cam: min={zmin:.6f}, max={zmax:.6f}")

    z_safe = np.maximum(pts_cam[:, 2], 1e-3)
    proj = project_points_cam(
        np.column_stack([pts_cam[:, 0], pts_cam[:, 1], z_safe]),
        K
    )
    rmse = compute_rmse_from_proj(proj, corners_2d)
    print(f"[DEBUG] first-sample reproj rmse (init): {rmse:.6f} px")

    dbg = image.copy()
    corners_reshape = corners_2d.reshape(1, 4, 2).astype(np.float32)
    cv2.aruco.drawDetectedMarkers(
        dbg, 
        [corners_reshape], 
        np.array([target_id], dtype=np.int32)
    )

    for p in proj:
        u, v = int(round(p[0])), int(round(p[1]))
        if 0 <= u < dbg.shape[1] and 0 <= v < dbg.shape[0]:
            cv2.circle(dbg, (u, v), 3, (0, 0, 255), -1)

    cv2.putText(
        dbg,
        f"init_reproj_rmse={rmse:.3f}px zmin={zmin:.3f} zmax={zmax:.3f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    out_path = os.path.join(out_dir, "debug_first_sample_projection.png")
    cv2.imwrite(out_path, dbg)
    print(f"[DEBUG] saved projection debug: {out_path}")

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Refine ZED head<->camera extrinsic using ArUco dataset")
    parser.add_argument("--data_dir", required=True, help="Directory with img_XX.png and pose_XX.json")
    parser.add_argument("--marker_size", type=float, required=True, help="ArUco marker side length in meters")
    parser.add_argument("--aruco_dict", type=str, default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--target_marker_id", type=int, default=1, help="Target ArUco Marker ID")

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument("--init_tx", type=float, default=-0.10)
    parser.add_argument("--init_ty", type=float, default=0.03)
    parser.add_argument("--init_tz", type=float, default=-0.05)
    parser.add_argument("--init_roll_deg", type=float, default=0.0)
    parser.add_argument("--init_pitch_deg", type=float, default=0.0)
    parser.add_argument("--init_yaw_deg", type=float, default=0.0)

    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--max_nfev", type=int, default=300)
    parser.add_argument(
        "--use_inverse_head",
        action="store_true",
        help="Use inv(T_base_to_head) to map base->head"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    K = np.array([
        [args.fx, 0.0, args.cx],
        [0.0, args.fy, args.cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    obj_pts_aruco = build_aruco_object_points(args.marker_size)

    json_files = sorted(glob.glob(os.path.join(data_dir, "pose_*.json")))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No pose_*.json found in {data_dir}")

    samples = []
    rejected = []

    dbg_dir = os.path.join(data_dir, "debug_detect_refine")
    os.makedirs(dbg_dir, exist_ok=True)

    for json_path in json_files:
        data = load_json(json_path)
        idx = data.get("index", None)

        image_file = data.get("image_file", None)
        if image_file is not None:
            image_path = os.path.join(data_dir, image_file)
        elif idx is not None:
            image_path = os.path.join(data_dir, f"img_{int(idx):02d}.png")
        else:
            rejected.append((json_path, "no_image_name"))
            continue

        img = cv2.imread(image_path)
        if img is None:
            rejected.append((json_path, "image_read_fail"))
            continue

        found, corners = detect_target_aruco(img, args.aruco_dict, args.target_marker_id)
        if not found or corners is None:
            rejected.append((json_path, "aruco_not_found"))
            continue

        try:
            T_base_to_head = extract_T_base_to_head(data)
        except Exception as e:
            rejected.append((json_path, f"tf_parse_fail: {repr(e)}"))
            continue

        corners_2d = corners.reshape(-1, 2).astype(np.float64)

        samples.append({
            "json_path": json_path,
            "image_path": image_path,
            "image": img,
            "T_base_to_head": T_base_to_head,
            "corners_2d": corners_2d
        })

        if args.save_debug:
            out_dbg = os.path.join(
                dbg_dir,
                os.path.basename(image_path).replace(".png", "_corners.png")
            )
            save_debug_detection(out_dbg, img, corners_2d, args.target_marker_id)

    print(f"[INFO] Found pose files : {len(json_files)}")
    print(f"[INFO] Used samples     : {len(samples)}")
    print(f"[INFO] Rejected         : {len(rejected)}")
    print(f"[INFO] use_inverse_head : {args.use_inverse_head}")

    if len(samples) < 5:
        raise RuntimeError("Too few valid samples. Need at least ~5, preferably 10+.")

    init_rot = R.from_euler(
        "xyz",
        [
            np.deg2rad(args.init_roll_deg),
            np.deg2rad(args.init_pitch_deg),
            np.deg2rad(args.init_yaw_deg),
        ]
    ).as_matrix()

    T_head_to_cam_init = make_transform(
        init_rot,
        np.array([args.init_tx, args.init_ty, args.init_tz], dtype=np.float64)
    )

    first = samples[0]
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts_aruco.astype(np.float32),
        first["corners_2d"].astype(np.float32),
        K,
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("Initial solvePnP failed on first sample")

    R_board_to_cam0, _ = cv2.Rodrigues(rvec)
    T_board_to_cam0 = make_transform(R_board_to_cam0, tvec.reshape(3))

    T_base_to_head0 = first["T_base_to_head"]
    T_cam_to_head_init = invert_transform(T_head_to_cam_init)

    if args.use_inverse_head:
        T_base_to_board_init = T_base_to_head0 @ T_cam_to_head_init @ T_board_to_cam0
    else:
        T_base_to_board_init = invert_transform(T_base_to_head0) @ T_cam_to_head_init @ T_board_to_cam0

    x0 = np.hstack([
        matrix_to_se3(T_head_to_cam_init),
        matrix_to_se3(T_base_to_board_init)
    ])

    debug_first_sample(
        first_sample=first,
        obj_pts_board=obj_pts_aruco,
        K=K,
        T_head_to_cam_init=T_head_to_cam_init,
        T_base_to_board_init=T_base_to_board_init,
        out_dir=dbg_dir,
        target_id=args.target_marker_id,
        use_inverse_head=args.use_inverse_head
    )

    rmse_before = compute_rmse(x0, samples, obj_pts_aruco, K, args.use_inverse_head)
    print(f"[INFO] Initial RMSE: {rmse_before:.4f} px")

    result = least_squares(
        residuals_fn,
        x0,
        args=(samples, obj_pts_aruco, K, args.use_inverse_head),
        method="trf",
        loss="soft_l1",
        f_scale=50.0,
        max_nfev=args.max_nfev,
        verbose=2
    )

    x_opt = result.x
    rmse_after = compute_rmse(x_opt, samples, obj_pts_aruco, K, args.use_inverse_head)

    T_head_to_cam_opt = se3_to_matrix(x_opt[:6])
    T_cam_to_head_opt = invert_transform(T_head_to_cam_opt)
    T_base_to_board_opt = se3_to_matrix(x_opt[6:12])

    print("\n================ RESULT ================")
    print(f"[INFO] Optimization success: {result.success}")
    print(f"[INFO] Message            : {result.message}")
    print(f"[INFO] Initial RMSE       : {rmse_before:.4f} px")
    print(f"[INFO] Final RMSE         : {rmse_after:.4f} px")

    print("\nT_head_to_cam =")
    print(np.array2string(T_head_to_cam_opt, precision=6, suppress_small=False))

    print("\nT_cam_to_head =")
    print(np.array2string(T_cam_to_head_opt, precision=6, suppress_small=False))

    print("\nT_base_to_board =")
    print(np.array2string(T_base_to_board_opt, precision=6, suppress_small=False))

    out = {
        "data_dir": data_dir,
        "num_samples_used": len(samples),
        "num_samples_rejected": len(rejected),
        "rejected": rejected,
        "camera_matrix": K.tolist(),
        "marker_size_m": args.marker_size,
        "aruco_dict": args.aruco_dict,
        "target_marker_id": args.target_marker_id,
        "use_inverse_head": bool(args.use_inverse_head),
        "initial_guess_head_to_cam": T_head_to_cam_init.tolist(),
        "initial_guess_base_to_board": T_base_to_board_init.tolist(),
        "initial_rmse_px": rmse_before,
        "final_rmse_px": rmse_after,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "T_head_to_cam": T_head_to_cam_opt.tolist(),
        "T_cam_to_head": T_cam_to_head_opt.tolist(),
        "T_base_to_board": T_base_to_board_opt.tolist()
    }

    out_path = os.path.join(data_dir, "handeye_result_aruco.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"\n[INFO] Saved result json: {out_path}")
    print(f"[INFO] Debug dir        : {dbg_dir}")

if __name__ == "__main__":
    main()