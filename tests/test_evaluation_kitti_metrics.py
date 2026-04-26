import unittest

import numpy as np

from src.modules.evaluation import kitti_odometry_subsequence_metrics
from src.modules.landmark_map import Camera


def _camera_from_c2w(name, R_c2w, t_c2w):
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ np.asarray(t_c2w, dtype=np.float64).reshape(3)
    return Camera(
        name=name,
        K=np.eye(3, dtype=np.float64),
        R=R_w2c,
        t=t_w2c,
    )


def _straight_line_poses(scale=1.0):
    cameras = []
    ground_truth = {}
    R_c2w = np.eye(3, dtype=np.float64)

    for idx in range(11):
        frame_id = f"{idx:06d}"
        gt_t = np.array([idx * 10.0, 0.0, 0.0], dtype=np.float64)
        est_t = np.array([idx * 10.0 * scale, 0.0, 0.0], dtype=np.float64)
        cameras.append(_camera_from_c2w(f"{frame_id}.png", R_c2w, est_t))
        ground_truth[frame_id] = {"R": R_c2w, "t": gt_t}

    return cameras, ground_truth


class TestKittiOdometrySubsequenceMetrics(unittest.TestCase):
    def test_kitti_subsequence_metric_is_zero_for_identical_trajectory(self):
        cameras, ground_truth = _straight_line_poses(scale=1.0)

        metrics = kitti_odometry_subsequence_metrics(
            cameras,
            ground_truth,
            lengths=(100,),
            step_size=1,
        )

        self.assertEqual(metrics["n_segments"], 1)
        summary = metrics["summary"]
        self.assertEqual(summary["translation_error_percent_mean"], 0.0)
        self.assertEqual(summary["rotation_error_deg_per_100m_mean"], 0.0)

    def test_kitti_subsequence_metric_reports_translation_drift_percent(self):
        cameras, ground_truth = _straight_line_poses(scale=0.9)

        metrics = kitti_odometry_subsequence_metrics(
            cameras,
            ground_truth,
            lengths=(100,),
            step_size=1,
        )

        summary = metrics["summary"]
        np.testing.assert_allclose(summary["translation_error_percent_mean"], 10.0)
        np.testing.assert_allclose(
            metrics["by_length"]["100"]["translation_error_percent_mean"],
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
