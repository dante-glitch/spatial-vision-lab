import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _pose_to_kitti_row(R, t) -> str:
    import numpy as np

    T = np.hstack(
        (
            np.asarray(R, dtype=np.float64).reshape(3, 3),
            np.asarray(t, dtype=np.float64).reshape(3, 1),
        )
    )
    return " ".join(f"{value:.12e}" for value in T.reshape(-1))


def _camera_to_c2w_pose(camera):
    """Convert the pipeline's world-to-camera pose into KITTI camera-to-world form."""
    R_c2w = camera.R.T
    t_c2w = (-camera.R.T @ camera.t.reshape(3, 1)).reshape(3, 1)
    return R_c2w, t_c2w


def save_kitti_trajectory(poses: dict, output_path: str | Path) -> Path:
    """Save a frame-id keyed pose dictionary in KITTI odometry text format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for frame_id in sorted(poses):
            pose = poses[frame_id]
            f.write(_pose_to_kitti_row(pose["R"], pose["t"]) + "\n")

    logger.info("Saved KITTI trajectory with %s poses to %s", len(poses), output_path)
    return output_path


def save_registered_kitti_trajectories(
    cameras,
    ground_truth_poses: dict,
    output_dir: str | Path,
):
    """Save estimated and ground-truth KITTI trajectories for registered frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimated_poses = {}
    matched_ground_truth_poses = {}
    for camera in cameras:
        frame_id = Path(camera.name).stem
        gt_pose = ground_truth_poses.get(frame_id)
        if gt_pose is None:
            continue

        R_est, t_est = _camera_to_c2w_pose(camera)
        estimated_poses[frame_id] = {"R": R_est, "t": t_est}
        matched_ground_truth_poses[frame_id] = gt_pose

    estimated_path = save_kitti_trajectory(
        estimated_poses,
        output_dir / "estimated_trajectory.txt",
    )
    ground_truth_path = save_kitti_trajectory(
        matched_ground_truth_poses,
        output_dir / "groundtruth_trajectory.txt",
    )

    frame_ids_path = output_dir / "frame_ids.txt"
    frame_ids_path.write_text("\n".join(sorted(estimated_poses)) + "\n")
    logger.info("Saved %s matched frame ids to %s", len(estimated_poses), frame_ids_path)

    return {
        "estimated": estimated_path,
        "groundtruth": ground_truth_path,
        "frame_ids": frame_ids_path,
        "n_poses": len(estimated_poses),
    }


def save_trajectory_tum(landmark_map, output_path: str | Path) -> Path:
    """Save estimated trajectory in TUM format: timestamp tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for idx, camera in enumerate(landmark_map.cameras):
            center = camera.center
            R_c2w = camera.R.T
            quat = Rotation.from_matrix(R_c2w).as_quat()
            f.write(
                f"{idx} "
                f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )

    logger.info("Saved TUM trajectory with %s poses to %s", len(landmark_map.cameras), output_path)
    return output_path
