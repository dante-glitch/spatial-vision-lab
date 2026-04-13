import logging
import numpy as np

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Camera:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        """Projection matrix P = K [R | t]"""
        return self.K @ np.hstack((self.R, self.t.reshape(3, 1)))

    @property
    def center(self) -> np.ndarray:
        """
        The camera centre is the one point in the world that maps
        to the zero vector in camera space — it's the origin of the camera's own coordinate system.

        R @ C + t = 0

        Camera center in world coordinates: C = -R^T t"""

        return (-self.R.T @ self.t.reshape(3, 1)).ravel()


@dataclass
class LandmarkMapping:
    """
    Accumulates Keyframes points, landmarks and camera poses for local bundle adjustment.
    Maintains a sliding window of the most recent N keyframes and their associated 2D-3D correspondences.


    Attributes
    ----------
    cameras_left    : list[Camera]   — registered left camera poses of the stereo camera in order of registration
    points3d   : np.ndarray     — (N,3) world-space landmark positions
    colors     : np.ndarray     — (N,3) uint8 BGR colours sampled from images


    """

    cameras: list = field(default_factory=list)
    points3d: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    colors: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    observations: dict = field(default_factory=dict, repr=False)

    max_history_limit: int = 15

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def get_camera(self, name: str) -> Camera:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise ValueError(f"Camera with name {name} not found.")

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    def add_points(self, pts3d: np.ndarray, colors: np.ndarray=None):
        if pts3d is None or len(pts3d) == 0:
            return np.empty((0,), dtype=np.int32)

        # Single point arrives as (3,) — promote to (1, 3)
        if pts3d.ndim == 1:
            pts3d = pts3d.reshape(1, 3)

        if colors is None:
            colors = np.full((len(pts3d), 3), 128, dtype=np.uint8)
        else:
            colors = np.asarray(colors, dtype=np.uint8)

        start_idx = self.n_points
        self.points3d = np.vstack((self.points3d, pts3d))
        self.colors = np.vstack((self.colors, colors))

        return np.arange(start_idx, start_idx + len(pts3d), dtype=np.int32)

    def add_observations(self, frame_name: str, kp_indices, point_indices, xy_coords):
        """Associate keypoints in a frame with 3D points (landmarks) in the map."""

        kp_indices = np.asarray(kp_indices, dtype=np.int32).ravel()
        point_indices = np.asarray(point_indices, dtype=np.int32).ravel()
        xy_coords = np.asarray(xy_coords, dtype=np.float32).reshape(-1, 2)

        if len(kp_indices) == 0:
            return

        if len(kp_indices) != len(point_indices) or len(kp_indices) != len(xy_coords):
            raise ValueError("kp_indices, point_indices, and xy_coords must match in length")

        obs = self.observations.setdefault(frame_name, {})

        for kp_idx, point_idx, xy in zip(kp_indices, point_indices, xy_coords):
            obs[int(kp_idx)] = {
                "point_idx": int(point_idx),
                "xy": xy.astype(np.float32),
            }

    def get_observations(self, frame_name: str) -> dict:
        return self.observations.get(frame_name, {})

    @property
    def n_points(self) -> int:
        return len(self.points3d)

    def filter_outlier_points(
        self,
        min_depth: float = 0.01,
        max_depth: float = 200.0,
        min_valid_views: int = 2,
    ):
        """
        Remove points that are behind camera or at extreme depths

        A landmark does not need to be valid in every registered camera:
        once the trajectory moves past it, the point can legitimately fall
        behind older views. We therefore keep points that have plausible
        depth in at least ``min_valid_views`` cameras.

        Also remaps observation point indices after removing filtered points.

        """

        if self.n_cameras == 0 or self.n_points == 0:
            return

        num_points_before = self.n_points
        # Count how many cameras see each point at a reasonable depth
        valid_view_counts = np.zeros(num_points_before, dtype=np.int32)
        required_views = min(max(min_valid_views, 1), self.n_cameras)

        for cam in self.cameras:
            # Transform points into camera's coordinate system
            pts_cam = (cam.R @ self.points3d.T + cam.t.reshape(3, 1)).T  # (N,3)
            depth = pts_cam[:, 2]  # distance from camera center

            # 3d point has valid depth value in how many cameras?
            valid_view_counts += ((depth > min_depth) & (depth < max_depth)).astype(
                np.int32
            )

        # Keep 3D points that are valid in enough cameras
        keep = valid_view_counts >= required_views

        before = self.n_points
        old_to_new = np.full(before, -1, dtype=np.int32)
        kept_old_indices = np.flatnonzero(
            keep
        )  # return indices that are non zero in the flattened version of keep

        old_to_new[kept_old_indices] = np.arange(len(kept_old_indices), dtype=np.int32)

        self.points3d = self.points3d[keep]
        self.colors = self.colors[keep]

        remapped_observations = {}
        for frame_name, frame_obs in self.observations.items():
            new_frame_obs = {}
            for kp_idx, record in frame_obs.items():
                point_idx = record["point_idx"]

                if 0 <= point_idx < before:
                    new_point_idx = int(old_to_new[point_idx])
                    if new_point_idx >= 0:
                        new_frame_obs[kp_idx] = {
                            "point_idx": new_point_idx,
                            "xy": record["xy"],
                        }
            if new_frame_obs:
                remapped_observations[frame_name] = new_frame_obs

        self.observations = remapped_observations

        logger.info(
            "[Filter] %s -> %s points (removed %s outliers)",
            before,
            self.n_points,
            before - self.n_points,
        )
