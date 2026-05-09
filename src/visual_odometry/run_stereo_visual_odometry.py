from src.modules.landmark_map import LandmarkMapping
from src.modules.keyframe_selector import KeyframeSelector
from src.datasets.kitti_odometry import KITTIOdometrySequence, read_kitti_odometry_poses
from src.visual_odometry.stereo_visual_odometry_pipeline import StereoVisualOdometryPipeline

from src.modules.bundle_adjustment import bundle_adjust_local, bundle_adjust_global
from src.modules.evaluation import visualize_trajectories, evaluate
from src.modules.save_trajectory import save_registered_kitti_trajectories

from src.modules.feature_extraction_matching import KeypointFeatureExtractorAndMatcher
from src.modules.vlad_encoder import VLADEncoder
from src.modules.loop_detector import LoopDetector
from src.modules.keyframe_database import KeyframeDatabase, KeyFrameRecord

from src.modules.pose_graph_optimization import (
    PoseGraphEdge,
    apply_optimized_poses,
    camera_to_world_T,
    optimize_pose_graph,
    relative_T,
)

import cv2
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ipdb import set_trace

logger = logging.getLogger(__name__)


def configure_logging(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
        force=True,
    )
    return log_path


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(
    sequence_parent_dir,
    groundtruth_pose_dir,
    sequence_id,
    gray_or_color,
    output_dir,
    vlad_cluster_centers_path,
    min_matches: int = 80,
    keyframe_max_match_ratio: float = 0.92,
    ratio_threshold: float = 0.75,
    max_frames_to_process: int = 500,
    loop_checking_frequency: int = 10,
    loop_min_temporal_separation: int = 40,
    loop_pnp_min_correspondences: int = 6,
    loop_pnp_min_inliers: int = 10,
):  
    output_dir = Path(output_dir)
    log_path = configure_logging(output_dir)
    logger.info("Logging to %s", log_path)
    
    ground_truth_poses = read_kitti_odometry_poses(
        f"{groundtruth_pose_dir}/{sequence_id}.txt"
    )

    kitti_seq = KITTIOdometrySequence(
        sequence_dir=f"{sequence_parent_dir}/{sequence_id}", gray_or_color=gray_or_color
    )

    max_frames_to_process = min(max_frames_to_process, len(kitti_seq))
    logger.info(
        "Processing up to %s frames from KITTI sequence %s",
        max_frames_to_process,
        sequence_id,
    )


    landmark_tracker = LandmarkMapping(max_history_limit=15)

    keyframe_selector = KeyframeSelector(
        min_matches=min_matches,
        max_match_ratio=keyframe_max_match_ratio,
        ratio_threshold=ratio_threshold,
    )

    pipeline = StereoVisualOdometryPipeline(
        K_left_cam=kitti_seq.cam0["K"],
        cx_left_cam=kitti_seq.cam0["cx"],
        cy_left_cam=kitti_seq.cam0["cy"],
        fx_left_cam=kitti_seq.cam0["fx"],
        fy_left_cam=kitti_seq.cam0["fy"],
        baseline=kitti_seq.cam1["baseline"],
        landmark_map_tracker=landmark_tracker,
        gray_or_color=gray_or_color,
        ratio_threshold=ratio_threshold,
    )

    # load a pretrained cluster_centers

    if vlad_cluster_centers_path is None:
        raise ValueError("--vlad_cluster_centers_path is required when loop detection is enabled")

    vlad_encoder = VLADEncoder.load(vlad_cluster_centers_path)

    loop_detector = LoopDetector(
        vlad_encoder=vlad_encoder,
        min_temporal_seperation=loop_min_temporal_separation,
        top_k=5, #TODO parameterize this
        min_score=0.25, #TODO parameterize this
    )

    keyframe_db = KeyframeDatabase()

    feature_extractor_matcher =KeypointFeatureExtractorAndMatcher()

    count = 0
    loop_edges = []
    for i in range(0, len(kitti_seq)):

        left, right, t_step = kitti_seq.get_frame(i)

        left_name = left['name'].split('.')[0]

        gt_pose = ground_truth_poses[left_name]


        accepted, _, _, reason = keyframe_selector.should_add(left)

        if accepted:
            count+=1

        if accepted:
            logger.info("Frame %s accepted as keyframe: %s", i, reason)
        else:
            logger.info("Frame %s rejected as keyframe: %s", i, reason)
            continue

        result = pipeline.register_stereo_frame_pair(t_step, left, right, gt_pose)
        
        if not result.get("success", False):
            continue

        current = pipeline.get_last_registered_frame()
        frame_name = current["name"]
        kp = current["kp"]
        des = current["des"]
        current_R = current["R"]
        current_t = current["t"]

        if count % loop_checking_frequency == 0:
            # Loop retrieval
            loop_candidates = loop_detector.query(frame_index=i, descriptors=des)

            # Loop Closure Detection and Geometric Verification
            for candidate in loop_candidates:
                old_kf = keyframe_db.get(candidate.frame_name)

                logger.info("Loop candidate current=%s old=%s score=%.3f", frame_name, candidate.frame_name, candidate.score)
                
                # geometric verification goes here
                # use old_kf.descriptors + old_kf.kp_to_landmark + landmark_tracker.points3d
                # to build 3D-2D correspondences into the current frame

                matches_loop, _ = feature_extractor_matcher.match_features(
                    old_kf.descriptors.astype(np.float32),
                    des,
                    ratio_threshold=ratio_threshold,
                )
                
                pts_3d = []
                pts_2d = []

                for m in matches_loop:
                    landmark_id = old_kf.kp_to_landmark.get(int(m.queryIdx), None)
                    if landmark_id is None:
                        continue
                    if landmark_id >= landmark_tracker.n_points:
                        continue

                    pts_3d.append(landmark_tracker.points3d[landmark_id])
                    pts_2d.append(kp[m.trainIdx].pt)
                
                pts3d = np.asarray(pts_3d, dtype=np.float64)
                pts2d = np.asarray(pts_2d, dtype=np.float32)

                # Relative Pose Verification with PnP RANSAC
                if len(pts3d) >= loop_pnp_min_correspondences:
                    
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        pts3d,
                        pts2d,
                        kitti_seq.cam0["K"],
                        None,
                        iterationsCount=1000,
                        reprojectionError=3.0,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                    logger.info("Performing loop PnP RANSAC with %s 3D-2D correspondences", len(pts3d))
                    logger.info("Loop PnP RANSAC success=%s inliers=%s", success, len(inliers) if inliers is not None else 0)

                    loop_accepted = (
                        success
                        and inliers is not None
                        and len(inliers) >= loop_pnp_min_inliers
                    )

                    if loop_accepted:
                        logger.info(
                            "Loop accepted current=%s old=%s score=%.3f inliers=%s",
                            frame_name,
                            candidate.frame_name,
                            candidate.score,
                            len(inliers),
                        )

                        R_loop, _ = cv2.Rodrigues(rvec)
                        T_old_world = camera_to_world_T(old_kf.R, old_kf.t)
                        T_current_loop_world = camera_to_world_T(R_loop, tvec)

                        loop_edges.append(
                            PoseGraphEdge(
                                source=old_kf.name,
                                target=frame_name,
                                relative_T=relative_T(T_old_world, T_current_loop_world),
                                weight=float(len(inliers)),
                                edge_type="loop",
                            )
                        )




        # adding new frame to database and loop detector

        frame_obs = landmark_tracker.get_observations(frame_name)
        
        kp_to_landmark = {int(kp_idx): int(obs["point_idx"]) for kp_idx, obs in frame_obs.items()}
        keypoints_xy = np.asarray([k.pt for k in kp], dtype=np.float32)

        keyframe_db.add(
        KeyFrameRecord(
            name=frame_name,
            frame_index=i,
            keypoints_xy=keypoints_xy,
            descriptors=des.astype(np.float16),   # optional memory reduction
            kp_to_landmark=kp_to_landmark,
            R=current_R,
            t=current_t,
        )
        )

        loop_detector.add_keyframe(
            frame_name=frame_name,
            frame_index=i,
            descriptors=des,
        )

        if i > 0 and i % 20 == 0:
            logger.info("Performing local bundle adjustment at frame %s", i)
            ba_stats = bundle_adjust_local(
                        landmark_map=landmark_tracker,
                        window_size=3,
                        max_nfev=40,
                        max_points=200,
                    )
            logger.info("Bundle adjustment stats: %s", ba_stats)
        

        if i > 0 and i % 100 == 0:
            logger.info("Frame %s: filtering outliers from landmark map", i)
            landmark_tracker.filter_outlier_points()

        if i == max_frames_to_process-1:
            logger.info(
                "Reached max_frames_to_process limit (%s). Stopping frame processing.",
                max_frames_to_process,
            )
            break

    # Final filtering
    landmark_tracker.filter_outlier_points()

    if loop_edges:
        logger.info("Optimizing pose graph with %s loop edges and %s odometry edges", len(loop_edges), count - 1)
        optimized_poses, pgo_stats = optimize_pose_graph(
            keyframes=keyframe_db.records,
            loop_edges=loop_edges,
        )

        apply_optimized_poses(keyframe_db, landmark_tracker, optimized_poses)
        logger.info("Pose graph optimization stats: %s", pgo_stats)

    else:
        logger.info("Skipping pose graph optimization: no accepted loop edges")

    # Final global BA after all frames processed
    logger.info("Running final global bundle adjustment after frame processing")
    final_global_ba_stats = bundle_adjust_global(
        landmark_map=landmark_tracker,
        fixed_cameras=1,
        max_nfev=120,
        max_points=500,
    )
    logger.info("Final global bundle adjustment stats: %s", final_global_ba_stats)

    # ---- Evaluation ------------------------------------------------------
    logger.info(f"Running Evaluation on {len(landmark_tracker.cameras)} registered frames...")
    metrics = evaluate(
        landmark_map=landmark_tracker,
        ground_truth_poses=ground_truth_poses,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2) + "\n")
    logger.info("Saved metrics to %s", metrics_path)

    # ---- Save outputs ----------------------------------------------------

    saved_paths = save_registered_kitti_trajectories(
        cameras=landmark_tracker.cameras,
        ground_truth_poses=ground_truth_poses,
        output_dir=output_dir,
    )
    logger.info("Saved trajectories: %s", saved_paths)

    logger.info("Visualizing trajectories... Close the plot window to finish.")
    visualize_trajectories(
        landmark_map=landmark_tracker,
        ground_truth_poses=ground_truth_poses,
        anchor_first_frame=True,
        use_umeyama=True,
        scale_without_umeyama=False, # we can get actual scale from stereo, so no need to scale by the umeyama factor
        title=f"Estimated vs GT Trajectories for KITTI Sequence {sequence_id}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stereo visual odometry on a KITTI sequence."
    )
    parser.add_argument(
        "--sequence_parent_dir",
        type=str,
        required=True,
        help="Path to the KITTI sequence directory.",
    )

    parser.add_argument(
        "--groundtruth_pose_parent_dir",
        type=str,
        required=True,
        help="Path to the KITTI ground truth pose directory.",
    )

    parser.add_argument(
        "--sequence_id",
        type=str,
        required=True,
        help="ID of the KITTI sequence to process.",
    )

    parser.add_argument(
        "--gray_or_color",
        type=str,
        choices=["gray", "color"],
        default="gray",
        help="Whether to use grayscale or color images for disparity computation.",
    )
    parser.add_argument(
        "--max_frames_to_process",
        type=int,
        default=200,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stereo_visual_odometry",
        help="Directory where logs and trajectories will be saved.",
    )
    parser.add_argument(
        "--vlad_cluster_centers_path",
        type=str,
        default=None,
        help="Path to the pretrained VLAD cluster centers (.npy).",
    )

    parser.add_argument(
        "--loop_checking_frequency",
        type=int,
        default=4,
        help="How often should we check for loops?",
    )
    parser.add_argument(
        "--loop_min_temporal_separation",
        type=int,
        default=40,
        help="Minimum frame-index distance between loop candidates and the current frame.",
    )
    parser.add_argument(
        "--loop_pnp_min_correspondences",
        type=int,
        default=6,
        help="Minimum 3D-2D correspondences required before loop PnP RANSAC.",
    )
    parser.add_argument(
        "--loop_pnp_min_inliers",
        type=int,
        default=50,
        help="Minimum PnP RANSAC inliers required to accept a loop.",
    )

    args = parser.parse_args()

    main(
        sequence_parent_dir=args.sequence_parent_dir,
        groundtruth_pose_dir=args.groundtruth_pose_parent_dir,
        sequence_id=args.sequence_id,
        gray_or_color=args.gray_or_color,
        output_dir=args.output_dir,
        vlad_cluster_centers_path=args.vlad_cluster_centers_path,
        loop_checking_frequency=args.loop_checking_frequency,
        loop_min_temporal_separation=args.loop_min_temporal_separation,
        loop_pnp_min_correspondences=args.loop_pnp_min_correspondences,
        loop_pnp_min_inliers=args.loop_pnp_min_inliers,
        max_frames_to_process=args.max_frames_to_process,
    )

"""

python src/visual_odometry/run_stereo_visual_odometry.py \
    --sequence_parent_dir /Users/krishna/Downloads/Datasets/KITTI/data_odometry_gray/sequences \
    --groundtruth_pose_parent_dir /Users/krishna/Downloads/Datasets/KITTI/data_odometry_poses_gt/poses \
    --sequence_id 06 --gray_or_color gray --max_frames_to_process 950 \
    --output_dir outputs/VO/kitti_06_pgo_globalBA \
    --vlad_cluster_centers_path outputs/vlad_train/seq_07/vlad_cluster_centers.npy

        
        
"""
