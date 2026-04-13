"""
We're using a local Bundle Adjustment where:
1. Only recent cameras as considered
2. Only points observed in the recent window
3. Older Cameras are fixed

"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.utils.landmark_map import LandmarkMapping


def _camera_to_params(cam):
    rvec, _ = cv2.Rodrigues(cam.R)
    return np.hstack([rvec.ravel(), cam.t.ravel()])


def _params_to_camera(params):
    rvec = params[:3].reshape(3, 1)
    t = params[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R, t


def _project_point(K, R, t, X):
    X_cam = R @ X.reshape(3, 1) + t
    z = float(X_cam[2, 0])
    if z <= 1e-8:
        return None
    x = K @ X_cam
    return np.array([x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]], dtype=np.float64)




def bundle_adjust_local(
    sfm_map: LandmarkMapping,
    window_size: int = 8,
    fixed_cams: int = 2,
    max_nfev: int = 100,
    max_points: int = 250,
):  
    """
    Local BA is a sliding-window optimizer. It only includes points that are observed by 
    cameras inside the recent BA window, and only if they appear at least twice in that window.
    """
    if sfm_map.n_cameras < max(window_size, fixed_cams + 1):
        return {"ran": False, "reason": "too few cameras"}

    window_cams = sfm_map.cameras[-window_size:]
    fixed_cams = min(fixed_cams, len(window_cams) - 1)

    opt_cams = window_cams[fixed_cams:]

    point_obs_counts = {}
    for cam in window_cams:
        for record in sfm_map.get_observations(cam.name).values():
            pid = record["point_idx"]
            point_obs_counts[pid] = point_obs_counts.get(pid, 0) + 1

    point_ids = sorted(
        pid for pid, count in point_obs_counts.items()
        if count >= 2 and 0 <= pid < sfm_map.n_points
    )
    if not point_ids:
        return {"ran": False, "reason": "no multi-view points in window"}

    n_points_before_cap = len(point_ids)
    if max_points is not None and len(point_ids) > max_points:
        point_ids = sorted(
            point_ids,
            key=lambda pid: (-point_obs_counts[pid], pid),
        )[:max_points]
        point_ids = sorted(point_ids)

    point_id_to_local = {pid: i for i, pid in enumerate(point_ids)}
    cam_name_to_local = {cam.name: i for i, cam in enumerate(opt_cams)}
    fixed_pose_map = {cam.name: (cam.R.copy(), cam.t.copy()) for cam in window_cams[:fixed_cams]}

    observations = []
    for cam in window_cams:
        frame_obs = sfm_map.get_observations(cam.name)
        for record in frame_obs.values():
            pid = record["point_idx"]
            if pid not in point_id_to_local:
                continue
            observations.append({
                "camera_name": cam.name,
                "point_local_idx": point_id_to_local[pid],
                "xy": record["xy"],
                "K": cam.K,
            })

    if len(observations) < 20:
        return {"ran": False, "reason": "too few observations"}

    x0_cams = np.concatenate([_camera_to_params(cam) for cam in opt_cams], axis=0) if opt_cams else np.empty((0,))
    x0_points = sfm_map.points3d[point_ids].reshape(-1)
    x0 = np.concatenate([x0_cams, x0_points], axis=0)

    n_opt_cams = len(opt_cams)
    n_points = len(point_ids)

    def residuals_fn(x):
        cam_block = x[: 6 * n_opt_cams].reshape(n_opt_cams, 6) if n_opt_cams else np.empty((0, 6))
        pts_block = x[6 * n_opt_cams :].reshape(n_points, 3)

        residuals = []
        for obs in observations:
            cam_name = obs["camera_name"]
            if cam_name in fixed_pose_map:
                R, t = fixed_pose_map[cam_name]
            else:
                local_idx = cam_name_to_local[cam_name]
                R, t = _params_to_camera(cam_block[local_idx])

            X = pts_block[obs["point_local_idx"]]
            proj = _project_point(obs["K"], R, t, X)
            if proj is None:
                residuals.extend([50.0, 50.0])
                continue

            residuals.extend((proj - obs["xy"]).tolist())

        return np.asarray(residuals, dtype=np.float64)

    residuals_before = residuals_fn(x0)
    rmse_before = float(np.sqrt(np.mean(residuals_before ** 2)))

    result = least_squares(
        residuals_fn,
        x0,
        method="trf",  #  Trust Region Reflective
        loss="soft_l1",  # robust loss to reduce the influence of outliers
        f_scale=2.0,
        max_nfev=max_nfev,
    )

    residuals_after = residuals_fn(result.x)
    rmse_after = float(np.sqrt(np.mean(residuals_after ** 2)))

    apply_result = np.isfinite(rmse_after) and rmse_after < rmse_before

    cam_block = result.x[: 6 * n_opt_cams].reshape(n_opt_cams, 6) if n_opt_cams else np.empty((0, 6))
    pts_block = result.x[6 * n_opt_cams :].reshape(n_points, 3)

    if apply_result:
        for cam in opt_cams:
            local_idx = cam_name_to_local[cam.name]
            R_opt, t_opt = _params_to_camera(cam_block[local_idx])
            cam.R = R_opt
            cam.t = t_opt

        sfm_map.points3d[point_ids] = pts_block

    return {
        "ran": True,
        "success": bool(result.success),
        "n_cameras": len(window_cams),
        "n_points": len(point_ids),
        "n_candidate_points": n_points_before_cap,
        "n_observations": len(observations),
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "applied": apply_result,
        "message": result.message,
    }
