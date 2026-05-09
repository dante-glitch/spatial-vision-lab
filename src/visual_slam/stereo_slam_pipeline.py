import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from src.modules.feature_extraction_matching import KeypointFeatureExtractorAndMatcher
from src.modules.landmark_map import LandmarkMapping
from typing import Optional
from ipdb import set_trace

logger = logging.getLogger(__name__)




# Camera record
@dataclass
class Camera:
    name: str
    K: np.ndarray          # 3x3 intrinsic matrix
    R: np.ndarray          # 3x3 rotation
    t: np.ndarray          # (3,1) translation
    R_gt: np.ndarray = None   # 3x3 ground-truth rotation (optional)
    t_gt: np.ndarray = None   # (3,1) ground-truth translation

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix P = K [R | t]"""
        return self.K @ np.hstack((self.R, self.t))
    
    @property
    def center(self) -> np.ndarray:
        
        """
        The camera centre is the one point in the world that maps
        to the zero vector in camera space — it's the origin of the camera's own coordinate system.
        
        R @ C + t = 0
        
        Camera center in world coordinates: C = -R^T t"""
        return (-self.R.T @ self.t.reshape(3, 1)).ravel()
    

def compute_disparity(matches, keypoints_1, keypoints_2):


    all_disparities = []
    for m in matches:
        ptL = keypoints_1[m.queryIdx].pt  # (uL, vL)
        ptR = keypoints_2[m.trainIdx].pt  # (uR, vR)

        uL, vL = ptL
        uR, vR = ptR

        disparity = uL - uR
        all_disparities.append(disparity)
    return all_disparities


def compte_depths_3D_points_ref_camera(
    disparities,
    keypoints_1,
    cx,
    cy,
    fx,
    fy,
    baseline,
    matches=None,
    return_valid_matches: bool = False,
):
    points_3D = []
    valid_matches = []

    if matches is None:
        keypoints_for_disparities = keypoints_1
    else:
        keypoints_for_disparities = [keypoints_1[m.queryIdx] for m in matches]

    for idx, (disparity, kp) in enumerate(zip(disparities, keypoints_for_disparities)):
        if not np.isfinite(disparity) or disparity <= 0:
            continue

        uL, vL = kp.pt # 2D pixel coordinates in the left image

        Z = (fx * baseline) / disparity
        X = (uL - cx) * Z / fx
        Y = (vL - cy) * Z / fy

        points_3D.append((X, Y, Z))
        if matches is not None:
            valid_matches.append(matches[idx])

    points_3D = np.asarray(points_3D, dtype=np.float64).reshape(-1, 3)

    if return_valid_matches:
        return points_3D, valid_matches

    return points_3D


class StereoVisualOdometryPipeline:
    def __init__(
        self,
        K_left_cam,
        cx_left_cam, 
        cy_left_cam,
        fx_left_cam,
        fy_left_cam,
        baseline,
        landmark_map_tracker: LandmarkMapping,
        gray_or_color: str,
        ratio_threshold: float = 0.75,
        ransac_reproj_thr: float = 1.0,
        max_reproj_error: float = 4.0,
        min_triangulation_angle: float = 5.0,
        n_features: int = 4000,
        min_points_match_thresh: int = 6,
        allow_fallback_after_bootstrap: bool = False,
        fallback_until_n_cameras: int = 3,
        pnp_history_size: int = 4,
       
    ):
        self.K = K_left_cam
        self.map = landmark_map_tracker
        self.gray_or_color = gray_or_color
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_thr = ransac_reproj_thr
        self.max_reproj_error = max_reproj_error
        self.min_triangulation_angle = min_triangulation_angle
        self.n_features = n_features
        self.min_points_match_thresh = min_points_match_thresh
        self.allow_fallback_after_bootstrap = allow_fallback_after_bootstrap
        self.fallback_until_n_cameras = fallback_until_n_cameras
        self.pnp_history_size = pnp_history_size
        self.feature_extractor_matcher = KeypointFeatureExtractorAndMatcher(
     
            ratio_threshold=ratio_threshold,
            n_features=n_features,
        )
        self.cx_left_cam = cx_left_cam
        self.cy_left_cam = cy_left_cam
        self.fx_left_cam = fx_left_cam
        self.fy_left_cam = fy_left_cam
        self.baseline = baseline

        # State of the last registered frame
        self._prev_frame: Optional[dict] = None  # {name, image, kp, des, R, t}
        self._registered_frame_history: list[dict] = []

    def _remember_registered_frame(self, frame_record: dict):
        self._registered_frame_history.append(frame_record)
        if len(self._registered_frame_history) > self.pnp_history_size:
            self._registered_frame_history = self._registered_frame_history[-self.pnp_history_size:]



    def _build_pnp_correspondences_from_history(self, kp, des):
        pts3d_pnp = []
        pts2d_pnp = []
        pnp_point_indices = []
        pnp_curr_kp_indices = []
        seen_point_indices = set()
        seen_curr_kp_indices = set()
        tracked_match_count = 0
        
        history = list(reversed(self._registered_frame_history)) # reverse to look at most recent frames first
        
        for hist in history:
            hist_des = hist.get('des')
            hist_name = hist.get("name")
            if hist_des is None:
                continue

            hist_observations = self.map.get_observations(hist_name)
            if not hist_observations:
                continue

            matches_hist, _ = self.feature_extractor_matcher.match_features(hist_des, des, self.ratio_threshold)

            for match in matches_hist:
                obs = hist_observations.get(match.queryIdx)
                if obs is None:
                    continue

                point_idx = obs["point_idx"] # 3D point
                if point_idx is None or point_idx >= self.map.n_points:
                    continue

                tracked_match_count += 1
                if point_idx in seen_point_indices or match.trainIdx in seen_curr_kp_indices:
                    continue

                seen_point_indices.add(point_idx)
                seen_curr_kp_indices.add(match.trainIdx)
                pts3d_pnp.append(self.map.points3d[point_idx])
                pts2d_pnp.append(kp[match.trainIdx].pt)
                pnp_point_indices.append(point_idx)
                pnp_curr_kp_indices.append(match.trainIdx)

        if len(pts3d_pnp) == 0:
            logger.warning("No valid 3D-2D correspondences for PnP")
            return None, None, None, None, tracked_match_count

        return (
            np.asarray(pts3d_pnp, dtype=np.float64),
            np.asarray(pts2d_pnp, dtype=np.float32),
            np.asarray(pnp_point_indices, dtype=np.int32),
            np.asarray(pnp_curr_kp_indices, dtype=np.int32),
            tracked_match_count,
        )
    
    def get_last_registered_frame(self) -> dict:
        return self._prev_frame


    def _register_pnp(self, image, left_frame_name, frame, kp, des, R_prev, t_prev, gt_pose):
        """
        Find 3D↔2D correspondences by propagating landmark observations
        from the previous registered frame into the current frame.

        If a previous-frame keypoint is already associated with a 3D
        landmark, and that keypoint matches into the new frame, then we
        inherit a 3D↔2D correspondence for PnP.

        """
        if self.map.n_points == 0:
            #return self._fail("No map points to do PnP", left_frame_name)
            raise ValueError("No map points to do PnP", left_frame_name)
        
        prev_name = self._prev_frame["name"]
        prev_observations = self.map.get_observations(prev_name)

        if not prev_observations:
            raise ValueError(f"Previous frame {prev_name} has no observations in the map.")
        
        # look at last N frames for potential correspondences to build a robust set of 3D-2D matches for PnP
        pnp_data = self._build_pnp_correspondences_from_history(kp, des)
        
        if pnp_data is None:
            raise ValueError("No valid 3D-2D correspondences for PnP", left_frame_name)
        

        pts3d_pnp, pts2d_pnp, pnp_point_indices, pnp_curr_kp_indices, tracked_match_count = pnp_data
        if pts3d_pnp is None:
            raise ValueError("No valid 3D-2D correspondences for PnP", left_frame_name)

        if len(pts3d_pnp) < self.min_points_match_thresh:
            raise ValueError(
                
                "Too few tracked landmark correspondences from recent history",
            )
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_pnp,
            pts2d_pnp,
            self.K,
            None,
            iterationsCount=1000,
            reprojectionError=self.ransac_reproj_thr * 2,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if (
            not success
            or inliers is None
            or len(inliers) < self.min_points_match_thresh
        ):
            logger.warning(
                "PnP failed: tracked=%s, corr=%s, inliers=%s",
                tracked_match_count,
                len(pts3d_pnp),
                0 if inliers is None else len(inliers),
            )
            raise ValueError(
                "PnP RANSAC failed",
            )
        
        R_curr, _ = cv2.Rodrigues(rvec)  # converts rotation vector -> rotation matrix, rvec = [rx, ry, rz]
        # The direction of the vector is the axis of rotation, The magnitude θ = ‖rvec‖ is the angle of rotation in radians
        t_curr = tvec.astype(np.float64)

        pnp_inlier_idx = inliers.ravel().astype(np.int32)
        inlier_point_indices = pnp_point_indices[pnp_inlier_idx]
        inlier_curr_kp_indices = pnp_curr_kp_indices[pnp_inlier_idx]
        inlier_curr_xy = pts2d_pnp[pnp_inlier_idx]

        #self.map.add_observations(left_frame_name, inlier_curr_kp_indices, inlier_point_indices, #inlier_curr_xy)


        cam = Camera(name=left_frame_name, K=self.K, R=R_curr, t=t_curr, R_gt=gt_pose["R"],
                t_gt=gt_pose["t"],)
        self.map.add_camera(cam)

        # what to do now?

        return dict(success=True, reason="pnp", n_matches=len(pts3d_pnp),
                    R_est=R_curr, t_est=t_curr, inlier_point_indices=inlier_point_indices, inlier_curr_kp_indices=inlier_curr_kp_indices, inlier_curr_xy=inlier_curr_xy)



         

    def register_stereo_frame_pair(self, t_step, frame_left: dict, frame_right: dict, gt_pose: dict) -> dict:


        left_kp, left_des = self.feature_extractor_matcher.extract_features(frame_left['image'])
        right_kp, right_des = self.feature_extractor_matcher.extract_features(frame_right['image'])
        matches_left_right, _ = self.feature_extractor_matcher.match_features(left_des, right_des)

        disparities = compute_disparity(matches_left_right, left_kp, right_kp)

        points_3D, valid_matches_left_right = compte_depths_3D_points_ref_camera(
                disparities,
                left_kp,
                self.cx_left_cam,
                self.cy_left_cam,
                self.fx_left_cam,
                self.fy_left_cam,
                self.baseline,
                matches=matches_left_right,
                return_valid_matches=True,
            )
        
        
        left_frame_name = frame_left['name']
        # Special case for the very first frame
        if self._prev_frame is None:
            
            R0 = np.eye(3, dtype=np.float64)
            t0 = np.zeros((3, 1), dtype=np.float64)
            cam = Camera(
                name=left_frame_name,
                K=self.K,
                R=R0,
                t=t0,
                R_gt=gt_pose["R"],
                t_gt=gt_pose["t"],
            )

            self._prev_frame = dict(name=left_frame_name, image=frame_left['image'], kp=left_kp, des=left_des, R=R0, t=t0)
            self._remember_registered_frame(self._prev_frame)
            logger.info("Frame at t=%s %s: planted at origin", t_step, left_frame_name)

            self.map.add_camera(cam)

            # Add new 3D points
            point_ids = self.map.add_points(points_3D)

            if len(point_ids) == 0:
                return 0

            kp_indices = np.array([m.queryIdx for m in valid_matches_left_right], dtype=np.int32)

            xy_pts = np.float32([left_kp[m.queryIdx].pt for m in valid_matches_left_right])

            self.map.add_observations(left_frame_name, kp_indices, point_ids, xy_pts)

            logger.info("Done with first stereo frame pair initialization")
            return dict(success=True, reason="origin", n_matches=0, n_new_points=0,
                        R_est=R0, t_est=t0, reproj_error=0.0)
        

        R_prev = self._prev_frame["R"]
        t_prev = self._prev_frame["t"]
        
        try:
            result = self._register_pnp(
                frame_left["image"],
                left_frame_name,
                frame_left,
                left_kp,
                left_des,
                R_prev,
                t_prev,
                gt_pose,
            )
        except ValueError as exc:
            logger.warning(
                "Skipping frame %s due to PnP failure: %s",
                left_frame_name,
                exc,
            )
            return {"success": False, "reason": "pnp_failure"}

        inlier_curr_kp_indices = result['inlier_curr_kp_indices']
        inlier_point_indices = result['inlier_point_indices']
        inlier_curr_xy = result['inlier_curr_xy']

        self.map.add_observations(
            left_frame_name,
            inlier_curr_kp_indices,
            inlier_point_indices,
            inlier_curr_xy,
        )

        self._prev_frame = dict(name=left_frame_name, image=frame_left['image'], kp=left_kp, des=left_des, R=result['R_est'], t=result['t_est'])

        self._remember_registered_frame(self._prev_frame)
        logger.info("Frame at t=%s %s", t_step, left_frame_name)


        # Add new stereo points in world coordinates. PnP inlier_point_indices
        # are existing map IDs, not indices into points_3D_world.
        points_3D_world = (result['R_est'].T @ (points_3D.T - result['t_est'])).T #convert points from camera coordinates to world coordinates using the estimated pose of the current frame    
        observed_curr_kp_indices = set(inlier_curr_kp_indices.tolist())
        new_stereo_indices = [
            idx
            for idx, match in enumerate(valid_matches_left_right)
            if match.queryIdx not in observed_curr_kp_indices
        ]

        if not new_stereo_indices:
            return result

        new_points_3D_world = points_3D_world[new_stereo_indices]
        point_ids = self.map.add_points(new_points_3D_world)

        if len(point_ids) == 0:
            return result

        kp_indices_stereo = np.array(
            [valid_matches_left_right[idx].queryIdx for idx in new_stereo_indices],
            dtype=np.int32,
        )
        xy_pts = np.float32(
            [left_kp[valid_matches_left_right[idx].queryIdx].pt for idx in new_stereo_indices]
        )
        # new pixels for which we have new 3D points
        self.map.add_observations(left_frame_name, kp_indices_stereo, point_ids, xy_pts)

        result["n_new_points"] = len(point_ids)
        return result
