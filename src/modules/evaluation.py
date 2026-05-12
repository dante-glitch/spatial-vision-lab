"""
Evaluation Metrics for Multiview Reconstruction

Computes:
- Per-camera rotation error after global alignment (geodesic, degrees)
- Per-camera position error after global alignment (scene units)
- Absolute Trajectory Error (ATE) — after Umeyama alignment
- Summary table of results.
"""

import numpy as np
import logging
import os
import tempfile
from typing import List, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from src.modules.landmark_map import LandmarkMapping, Camera

import open3d as o3d
import scipy.spatial.transform as sst

logger = logging.getLogger(__name__)

# Rotation Error

def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Geodesic rotation error in degrees between two rotation matrices. 
    A geodesic is the shortest path on a curved surface.
    
    Geodesic rotation -> True angular distance between two rotations on the rotation manifold.
    Answers -> What is the smallest rotation that moves R_gt to R_est?
    """


    R_diff = R_est @ R_gt.T
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle_rad)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Umeyama alignment is a method to compute the best rigid (or similarity) transformation between two sets of 3D points.
    “Find the rotation, translation (and optionally scale) that aligns one point cloud with another.”

    
    Objective:
    Umeyama alignment maps the estimated trajectory into the same coordinate frame as the ground truth. So that makes comparing 
    estmated trajectory to ground truth meaningful. It finds the best R, t, s that minimizes the squared distance between corresponding points.
     - R: rotation matrix (3x3)
     - t: translation vector (3,)
     - s: isotropic scale factor (float, optional)

    p_aligned = s * R * p_est + t

    Steps:
    1. Compute centroids of both point sets
    2. Center the points by subtracting their centroids (removes translation)
    3. Compute covariance matrix between centered sets 
    4. SVD of covariance to get optimal rotation (Σ=UDVT)
        R=UVT
        s= 1/(σp^2) * (trace(D),   σp^2 - variance of the source points.
        t=μ(q)​−sRμ(p)​

    5. Final Alignment: p_aligned = s * R * p_est + t

    Umeyama is preferred because:
     - exact solution (closed-form, non-iterative)
     - numerically stable
     - fast
     - handles scale


    Parameters
    ----------
    src, dst : (N, 3) — corresponding trajectory points
    with_scale : bool — allow isotropic scaling

    Returns
    -------
    R : (3,3), t : (3,), s : float   such that  s * R @ src.T + t ≈ dst.T

    """
    assert src.shape == dst.shape
    n, m = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n
    cov     = (dst_c.T @ src_c) / n


    U, D, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, det_sign])

    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / (var_src + 1e-12) if with_scale else 1.0
    t = mu_dst - s * R @ mu_src

    return R, t, float(s)


def _parse_groundtruth_file(gt_path: str):
    timestamps = []
    centers = []
    rotations_c2w = []

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            toks = line.split()
            if len(toks) < 8:
                continue

            timestamps.append(float(toks[0]))
            centers.append(np.asarray(list(map(float, toks[1:4])), dtype=np.float64))
            rotations_c2w.append(Rotation.from_quat(list(map(float, toks[4:8]))).as_matrix())

    if not timestamps:
        raise ValueError(f"No valid poses found in {gt_path}")

    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(rotations_c2w, dtype=np.float64),
    )


def _parse_rgb_file(rgb_path: str):
    timestamps = []
    image_names = []

    with open(rgb_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            toks = line.split()
            if len(toks) < 2:
                continue

            timestamps.append(float(toks[0]))
            image_names.append(toks[1].split("/")[-1])

    if not timestamps:
        raise ValueError(f"No valid RGB entries found in {rgb_path}")

    return np.asarray(timestamps, dtype=np.float64), image_names


def _interpolate_gt_for_rgb(sequence_dir: str):
    gt_path = f"{sequence_dir}/groundtruth.txt"
    rgb_path = f"{sequence_dir}/rgb.txt"

    gt_times, gt_centers, gt_rotations_c2w = _parse_groundtruth_file(gt_path)
    rgb_times, rgb_image_names = _parse_rgb_file(rgb_path)

    valid_mask = (rgb_times >= gt_times[0]) & (rgb_times <= gt_times[-1])
    rgb_times_valid = rgb_times[valid_mask]
    rgb_names_valid = [name for name, keep in zip(rgb_image_names, valid_mask) if keep]

    if len(rgb_times_valid) == 0:
        raise ValueError("No RGB timestamps fall within the GT timestamp range")

    interp_centers = np.column_stack([
        np.interp(rgb_times_valid, gt_times, gt_centers[:, dim])
        for dim in range(3)
    ])
    slerp = Slerp(gt_times, Rotation.from_matrix(gt_rotations_c2w))
    interp_rotations_c2w = slerp(rgb_times_valid).as_matrix()

    gt_by_image = {}
    for name, center, R_c2w in zip(rgb_names_valid, interp_centers, interp_rotations_c2w):
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ center.reshape(3, 1)
        gt_by_image[name] = {
            "R_gt": R_w2c,
            "t_gt": t_w2c,
            "center_gt": center,
        }

    return rgb_names_valid, gt_by_image


def _read_segment_breaks(path: str):
    segments = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid segment_breaks line {line_no}: expected 3 comma-separated values"
                )

            start_name, end_name, length_str = parts
            segments.append(
                {
                    "start_name": start_name,
                    "end_name": end_name,
                    "length": int(length_str),
                }
            )

    if not segments:
        raise ValueError(f"No segments found in {path}")

    return segments


def _segment_name_set(rgb_names: List[str], segment_breaks_file: Optional[str]):
    if segment_breaks_file is None:
        return None, None

    longest_segment = max(_read_segment_breaks(segment_breaks_file), key=lambda seg: seg["length"])
    start_idx = rgb_names.index(longest_segment["start_name"])
    end_idx = rgb_names.index(longest_segment["end_name"])
    if start_idx > end_idx:
        raise ValueError(
            f"Longest segment start comes after end: "
            f"{longest_segment['start_name']} -> {longest_segment['end_name']}"
        )

    return set(rgb_names[start_idx:end_idx + 1]), longest_segment


def _collect_rgb_aligned_gt_camera_data(
    cameras: List[Camera],
    sequence_dir: str,
    segment_breaks_file: Optional[str] = None,
):
    rgb_names, gt_by_image = _interpolate_gt_for_rgb(sequence_dir)
    allowed_names, longest_segment = _segment_name_set(rgb_names, segment_breaks_file)

    eval_cameras = []
    est_centers = []
    gt_centers = []
    gt_rotations = []

    for cam in cameras:
        if allowed_names is not None and cam.name not in allowed_names:
            continue

        gt_pose = gt_by_image.get(cam.name)
        if gt_pose is None:
            continue

        eval_cameras.append(cam)
        est_centers.append(cam.center)
        gt_centers.append(gt_pose["center_gt"])
        gt_rotations.append(gt_pose["R_gt"])

    if len(eval_cameras) < 3:
        return None

    return (
        eval_cameras,
        np.asarray(est_centers, dtype=np.float64),
        np.asarray(gt_centers, dtype=np.float64),
        np.asarray(gt_rotations, dtype=np.float64),
        longest_segment,
    )


def _collect_gt_camera_data(cameras: List[Camera]):
    """Collect aligned-evaluation inputs for cameras that have GT."""
    eval_cameras = []
    est_centers = []
    gt_centers = []

    for cam in cameras:
        if cam.R_gt is None or cam.t_gt is None:
            continue
        eval_cameras.append(cam)
        est_centers.append(cam.center)
        gt_centers.append((-cam.R_gt.T @ cam.t_gt.reshape(3, 1)).ravel())

    if len(eval_cameras) < 3:
        return None

    return (
        eval_cameras,
        np.asarray(est_centers, dtype=np.float64),
        np.asarray(gt_centers, dtype=np.float64),
    )


def _collect_kitti_gt_camera_data(cameras: List[Camera], ground_truth_poses: dict):
    """Collect trajectory inputs for KITTI pose-file ground truth."""
    eval_cameras = []
    est_centers = []
    gt_centers = []
    gt_rotations = []

    for cam in cameras:
        frame_id = Path(cam.name).stem
        gt_pose = ground_truth_poses.get(frame_id)
        if gt_pose is None:
            continue

        R_gt_c2w = np.asarray(gt_pose["R"], dtype=np.float64)
        t_gt_c2w = np.asarray(gt_pose["t"], dtype=np.float64).reshape(3)

        eval_cameras.append(cam)
        est_centers.append(cam.center)
        gt_centers.append(t_gt_c2w)
        gt_rotations.append(R_gt_c2w.T)

    if len(eval_cameras) < 3:
        return None

    return (
        eval_cameras,
        np.asarray(est_centers, dtype=np.float64),
        np.asarray(gt_centers, dtype=np.float64),
        np.asarray(gt_rotations, dtype=np.float64),
    )


def _camera_to_c2w_matrix(camera: Camera) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = camera.R.T
    T[:3, 3] = camera.center
    return T


def _kitti_pose_to_matrix(pose: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(pose["R"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(pose["t"], dtype=np.float64).reshape(3)
    return T


def _collect_kitti_pose_matrices(cameras: List[Camera], ground_truth_poses: dict):
    records = []
    for camera in cameras:
        frame_id = Path(camera.name).stem
        gt_pose = ground_truth_poses.get(frame_id)
        if gt_pose is None:
            continue

        records.append(
            {
                "frame_id": frame_id,
                "frame_number": int(frame_id),
                "T_est_c2w": _camera_to_c2w_matrix(camera),
                "T_gt_c2w": _kitti_pose_to_matrix(gt_pose),
            }
        )

    records.sort(key=lambda record: record["frame_number"])
    return records


def _trajectory_distances(poses_c2w: list[np.ndarray]) -> np.ndarray:
    distances = np.zeros(len(poses_c2w), dtype=np.float64)
    for idx in range(1, len(poses_c2w)):
        prev_t = poses_c2w[idx - 1][:3, 3]
        curr_t = poses_c2w[idx][:3, 3]
        distances[idx] = distances[idx - 1] + np.linalg.norm(curr_t - prev_t)
    return distances


def _last_frame_from_segment_length(distances: np.ndarray, first_idx: int, length: float):
    target_distance = distances[first_idx] + length
    for idx in range(first_idx, len(distances)):
        if distances[idx] >= target_distance:
            return idx
    return None


def _rotation_error_rad(T_error: np.ndarray) -> float:
    R = T_error[:3, :3]
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _translation_error(T_error: np.ndarray) -> float:
    return float(np.linalg.norm(T_error[:3, 3]))


def kitti_odometry_subsequence_metrics(
    cameras: List[Camera],
    ground_truth_poses: dict,
    lengths: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800),
    step_size: int = 10,
) -> Optional[dict]:
    """
    Compute the KITTI odometry subsequence relative pose error metric.

    Translation is reported as percent drift. Rotation is reported in
    degrees per 100 meters. The metric is computed over registered frames;
    if keyframe selection skips frames, the reported segment count may be
    lower than a dense KITTI submission.
    """
    records = _collect_kitti_pose_matrices(cameras, ground_truth_poses)
    if len(records) < 2:
        return None

    gt_poses = [record["T_gt_c2w"] for record in records]
    est_poses = [record["T_est_c2w"] for record in records]
    distances = _trajectory_distances(gt_poses)

    segment_errors = []
    for first_idx in range(0, len(records), step_size):
        for length in lengths:
            last_idx = _last_frame_from_segment_length(distances, first_idx, length)
            if last_idx is None:
                continue

            T_gt_delta = np.linalg.inv(gt_poses[first_idx]) @ gt_poses[last_idx]
            T_est_delta = np.linalg.inv(est_poses[first_idx]) @ est_poses[last_idx]
            T_error = np.linalg.inv(T_est_delta) @ T_gt_delta

            t_err = _translation_error(T_error) / length
            r_err = _rotation_error_rad(T_error) / length

            segment_errors.append(
                {
                    "first_frame": records[first_idx]["frame_id"],
                    "last_frame": records[last_idx]["frame_id"],
                    "length_m": float(length),
                    "translation_error": float(t_err),
                    "translation_error_percent": float(t_err * 100.0),
                    "rotation_error_rad_per_m": float(r_err),
                    "rotation_error_deg_per_100m": float(np.degrees(r_err) * 100.0),
                    "num_registered_frames": int(last_idx - first_idx + 1),
                }
            )

    if not segment_errors:
        return {
            "n_frames": len(records),
            "lengths_m": list(lengths),
            "step_size": step_size,
            "n_segments": 0,
            "summary": None,
            "by_length": {},
            "segments": [],
        }

    t_percent = np.asarray(
        [error["translation_error_percent"] for error in segment_errors],
        dtype=np.float64,
    )
    r_deg_100m = np.asarray(
        [error["rotation_error_deg_per_100m"] for error in segment_errors],
        dtype=np.float64,
    )

    by_length = {}
    for length in lengths:
        length_errors = [
            error for error in segment_errors
            if np.isclose(error["length_m"], float(length))
        ]
        if not length_errors:
            continue

        length_t = np.asarray(
            [error["translation_error_percent"] for error in length_errors],
            dtype=np.float64,
        )
        length_r = np.asarray(
            [error["rotation_error_deg_per_100m"] for error in length_errors],
            dtype=np.float64,
        )
        by_length[str(length)] = {
            "n_segments": len(length_errors),
            "translation_error_percent_mean": float(np.mean(length_t)),
            "rotation_error_deg_per_100m_mean": float(np.mean(length_r)),
        }

    return {
        "n_frames": len(records),
        "lengths_m": list(lengths),
        "step_size": step_size,
        "n_segments": len(segment_errors),
        "summary": {
            "translation_error_percent_mean": float(np.mean(t_percent)),
            "translation_error_percent_median": float(np.median(t_percent)),
            "rotation_error_deg_per_100m_mean": float(np.mean(r_deg_100m)),
            "rotation_error_deg_per_100m_median": float(np.median(r_deg_100m)),
        },
        "by_length": by_length,
        "segments": segment_errors,
    }


def aligned_pose_metrics(
    cameras     : List[Camera],
    with_scale  : bool = True,
    sequence_dir: Optional[str] = None,
    segment_breaks_file: Optional[str] = None,
    ground_truth_poses: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute trajectory and orientation metrics after Umeyama alignment.
    Only cameras with ground-truth poses are included.

    Returns None if fewer than 3 cameras have GT.
    """
    if ground_truth_poses is not None:
        collected = _collect_kitti_gt_camera_data(cameras, ground_truth_poses)
        if collected is None:
            return None
        eval_cameras, est, gt, gt_rotations = collected
        longest_segment = None
    elif sequence_dir is not None:
        collected = _collect_rgb_aligned_gt_camera_data(
            cameras,
            sequence_dir,
            segment_breaks_file=segment_breaks_file,
        )
        if collected is None:
            return None
        eval_cameras, est, gt, gt_rotations, longest_segment = collected
    else:
        collected = _collect_gt_camera_data(cameras)
        if collected is None:
            return None
        eval_cameras, est, gt = collected
        gt_rotations = np.asarray([cam.R_gt for cam in eval_cameras], dtype=np.float64)
        longest_segment = None

    R_a, t_a, s_a = umeyama_alignment(est, gt, with_scale=with_scale)
    est_aligned = (s_a * (R_a @ est.T)).T + t_a

    position_errors = np.linalg.norm(est_aligned - gt, axis=1)
    rotation_errors = []
    for cam, R_gt in zip(eval_cameras, gt_rotations):
        # If x_gt = s * R_a * x_est + t_a, then the aligned w2c rotation is:
        # R_w2c_aligned = R_est * R_a^T
        R_est_aligned = cam.R @ R_a.T
        rotation_errors.append(rotation_error_deg(R_est_aligned, R_gt))

    rotation_errors = np.asarray(rotation_errors, dtype=np.float64)
    position_errors = np.asarray(position_errors, dtype=np.float64)

    return dict(
        rot_errors_aligned=rotation_errors,
        position_errors_aligned=position_errors,
        ate_rmse=float(np.sqrt((position_errors ** 2).mean())),
        alignment_scale=s_a,
        longest_segment=longest_segment,
    )


def aligned_trajectories(
    cameras    : List[Camera],
    with_scale : bool = True,
    use_umeyama: bool = True,
    scale_without_umeyama: bool = False,
    sequence_dir: Optional[str] = None,
    segment_breaks_file: Optional[str] = None,
    ground_truth_poses: Optional[dict] = None,
    anchor_first_frame: Optional[bool] = None,
) -> Optional[dict]:
    """
    Return estimated and GT trajectories in the same aligned frame.

    The estimated trajectory is aligned to GT using Umeyama unless
    use_umeyama is False. With Umeyama disabled, scale_without_umeyama
    can still scale the estimate for display without rotating or
    translating it from a whole-trajectory fit.
    """
    if anchor_first_frame is None:
        anchor_first_frame = ground_truth_poses is not None

    if ground_truth_poses is not None:
        collected = _collect_kitti_gt_camera_data(cameras, ground_truth_poses)
        if collected is None:
            return None
        eval_cameras, est, gt, _gt_rotations = collected
        longest_segment = None
    elif sequence_dir is not None:
        collected = _collect_rgb_aligned_gt_camera_data(
            cameras,
            sequence_dir,
            segment_breaks_file=segment_breaks_file,
        )
        if collected is None:
            return None
        eval_cameras, est, gt, _, longest_segment = collected
    else:
        collected = _collect_gt_camera_data(cameras)
        if collected is None:
            return None
        eval_cameras, est, gt = collected
        longest_segment = None

    if anchor_first_frame:
        est_relative = est - est[0]
        gt_relative = gt - gt[0]
        if use_umeyama:
            R_a, _, s_a = umeyama_alignment(est_relative, gt_relative, with_scale=with_scale)
            est_aligned = (s_a * (R_a @ est_relative.T)).T
        elif scale_without_umeyama:
            est_path_length = _trajectory_path_length(est_relative)
            gt_path_length = _trajectory_path_length(gt_relative)
            s_a = gt_path_length / est_path_length if est_path_length > 1e-12 else 1.0
            est_aligned = s_a * est_relative
        else:
            s_a = 1.0
            est_aligned = est_relative
        gt = gt_relative
    else:
        if use_umeyama:
            R_a, t_a, s_a = umeyama_alignment(est, gt, with_scale=with_scale)
            est_aligned = (s_a * (R_a @ est.T)).T + t_a
        else:
            s_a = 1.0
            est_aligned = est

    return dict(
        camera_names=[cam.name for cam in eval_cameras],
        est_aligned=est_aligned,
        gt=gt,
        alignment_scale=s_a,
        used_umeyama=use_umeyama,
        scale_without_umeyama=scale_without_umeyama,
        longest_segment=longest_segment,
    )



def evaluate(
    landmark_map: LandmarkMapping,
    sequence_dir: Optional[str] = None,
    segment_breaks_file: Optional[str] = None,
    ground_truth_poses: Optional[dict] = None,
) -> dict:
    """
    Run all metrics and log a summary table.

    Returns dict with keys:
        rot_errors_aligned, position_errors_aligned, ate_rmse,
        alignment_scale, kitti_subsequence
    """
    pose_metrics = aligned_pose_metrics(
        landmark_map.cameras,
        sequence_dir=sequence_dir,
        segment_breaks_file=segment_breaks_file,
        ground_truth_poses=ground_truth_poses,
    )
    n_gt_cameras = (
        0 if pose_metrics is None else len(pose_metrics["position_errors_aligned"])
    )
    kitti_metrics = None
    if ground_truth_poses is not None:
        kitti_metrics = kitti_odometry_subsequence_metrics(
            landmark_map.cameras,
            ground_truth_poses,
        )

    logger.info("%s", "=" * 60)
    logger.info("EVALUATION REPORT")
    logger.info("%s", "=" * 60)
    logger.info("Registered cameras       : %s", landmark_map.n_cameras)
    logger.info("Cameras with GT          : %s", n_gt_cameras)
    logger.info("Total 3D points          : %s", landmark_map.n_points)
    if pose_metrics is not None and pose_metrics.get("longest_segment") is not None:
        segment = pose_metrics["longest_segment"]
        logger.info(
            "Eval segment             : %s -> %s (length=%s)",
            segment["start_name"],
            segment["end_name"],
            segment["length"],
        )

    if pose_metrics is not None:
        rot_errs = pose_metrics["rot_errors_aligned"]
        pos_errs = pose_metrics["position_errors_aligned"]

        logger.info("Rotation error after alignment (deg)")
        logger.info("  mean  : %.3f", np.mean(rot_errs))
        logger.info("  median: %.3f", np.median(rot_errs))
        logger.info("  max   : %.3f", np.max(rot_errs))

        logger.info("Position error after alignment")
        logger.info("  mean  : %.5f  [scene units]", np.mean(pos_errs))
        logger.info("  median: %.5f  [scene units]", np.median(pos_errs))
        logger.info("  max   : %.5f  [scene units]", np.max(pos_errs))

        logger.info("ATE RMSE (after Umeyama): %.5f  [scene units]", pose_metrics["ate_rmse"])
        logger.info("Alignment scale          : %.5f", pose_metrics["alignment_scale"])
    else:
        logger.info("Aligned pose metrics     : N/A (< 3 GT cameras)")

    if kitti_metrics is not None:
        logger.info("KITTI odometry subsequence metric")
        logger.info("  segments: %s", kitti_metrics["n_segments"])
        if kitti_metrics["summary"] is not None:
            summary = kitti_metrics["summary"]
            logger.info(
                "  t_err  : %.3f %%", summary["translation_error_percent_mean"]
            )
            logger.info(
                "  r_err  : %.3f deg / 100m",
                summary["rotation_error_deg_per_100m_mean"],
            )
        else:
            logger.info("  N/A: no 100m-800m subsequences found")

    logger.info("%s", "=" * 60)

    metrics = pose_metrics or dict(
        rot_errors_aligned=[],
        position_errors_aligned=[],
        ate_rmse=None,
        alignment_scale=None,
    )
    metrics["kitti_subsequence"] = kitti_metrics
    return metrics


def _make_polyline(points: np.ndarray, color: tuple[float, float, float]):
    """Create an Open3D line set and point cloud for a trajectory."""
    if len(points) == 0:
        return []

    geometries = []

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(color)
    geometries.append(point_cloud)

    if len(points) > 1:
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(line_set)

    return geometries


def _trajectory_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _trajectory_span(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def visualize_trajectories(
    landmark_map: LandmarkMapping,
    sequence_dir: str | None = None,
    ground_truth_poses: dict | None = None,
    segment_breaks_file: str | None = None,
    anchor_first_frame: bool | None = None,
    use_umeyama: bool = True,
    scale_without_umeyama: bool = False,
    title: str = "Estimated vs GT Trajectories",
):
    """Open an Open3D viewer with aligned estimated and GT trajectories."""
    traj = aligned_trajectories(
        landmark_map.cameras,
        sequence_dir=sequence_dir,
        segment_breaks_file=segment_breaks_file,
        ground_truth_poses=ground_truth_poses,
        anchor_first_frame=anchor_first_frame,
        use_umeyama=use_umeyama,
        scale_without_umeyama=scale_without_umeyama,
    )
    if traj is None:
        logger.info("[Viz] Not enough GT cameras to visualise aligned trajectories.")
        return

    est_color = (0.85, 0.2, 0.2)
    gt_color = (0.1, 0.5, 0.9)
    geometries = []
    geometries.extend(_make_polyline(traj["est_aligned"], est_color))
    geometries.extend(_make_polyline(traj["gt"], gt_color))

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(origin)

    logger.info("[Viz] Trajectory colors: estimated=red, ground truth=blue")
    logger.info(
        "[Viz] Umeyama alignment: %s",
        "enabled" if traj["used_umeyama"] else "disabled",
    )
    if not traj["used_umeyama"]:
        logger.info(
            "[Viz] Scale without Umeyama: %s (scale=%.6g)",
            "enabled" if traj["scale_without_umeyama"] else "disabled",
            traj["alignment_scale"],
        )
    est_span = _trajectory_span(traj["est_aligned"])
    gt_span = _trajectory_span(traj["gt"])
    logger.info("[Viz] Trajectory span: est=%.6g, gt=%.6g", est_span, gt_span)
    if est_span < 1e-9 and len(traj["est_aligned"]) > 1:
        logger.warning("[Viz] Estimated camera centers are nearly identical; check PnP pose updates.")
    logger.info(
        "[Viz] First matched frame: %s, est=%s, gt=%s",
        traj["camera_names"][0],
        traj["est_aligned"][0],
        traj["gt"][0],
    )
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="open3d_viewer_") as viewer_cwd:
        try:
            os.chdir(viewer_cwd)
            o3d.visualization.draw_geometries(
                geometries,
                window_name=title,
                width=1280,
                height=720,
            )
        finally:
            os.chdir(original_cwd)
