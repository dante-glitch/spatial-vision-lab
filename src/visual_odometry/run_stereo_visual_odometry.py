from src.utils.landmark_map import LandmarkMapping
from src.utils.keyframe_selector import KeyframeSelector
from src.datasets.kitti_odometry import KITTIOdometrySequence, read_kitti_odometry_poses
from src.visual_odometry.stereo_visual_odometry_pipeline import StereoVisualOdometryPipeline
from src.utils.feature_extraction_matching import KeypointFeatureExtractorAndMatcher
from src.utils.evaluation import visualize_trajectories, evaluate
from src.utils.bundle_adjustment import bundle_adjust_local
from src.utils.save_trajectory import save_registered_kitti_trajectories

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
    min_matches: int = 80,
    keyframe_max_match_ratio: float = 0.92,
    ratio_threshold: float = 0.75,
    max_frames_to_process: int = 500,
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

    feature_extractor_matcher = KeypointFeatureExtractorAndMatcher(n_features=3000)

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


    for i in range(0, len(kitti_seq)):

        left, right, t_step = kitti_seq.get_frame(i)

        left_name = left['name'].split('.')[0]

        gt_pose = ground_truth_poses[left_name]


        accepted, _, _, reason = keyframe_selector.should_add(left)

        if accepted:
            logger.info("Frame %s accepted as keyframe: %s", i, reason)
        else:
            logger.info("Frame %s rejected as keyframe: %s", i, reason)
            continue

        result = pipeline.register_stereo_frame_pair(t_step, left, right, gt_pose)

        
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

    args = parser.parse_args()

    main(
        sequence_parent_dir=args.sequence_parent_dir,
        groundtruth_pose_dir=args.groundtruth_pose_parent_dir,
        sequence_id=args.sequence_id,
        gray_or_color=args.gray_or_color,
        output_dir=args.output_dir,
        max_frames_to_process=args.max_frames_to_process,
    )

"""

python src/visual_odometry/run_stereo_visual_odometry.py \
    --sequence_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/ \
    --groundtruth_pose_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/ground_truth_poses/poses/ \
    --sequence_id 02 --gray_or_color gray --max_frames_to_process 600 \
    --output_dir outputs/VO/kitti_02

        
        
"""
