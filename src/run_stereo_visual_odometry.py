from src.utils.landmark_map import LandmarkMapping
from src.utils.keyframe_selector import KeyframeSelector
from src.datasets.kitti_odometry import KITTIOdometrySequence, read_kitti_odometry_poses
from src.stereo_visual_odometry_pipeline import StereoVisualOdometryPipeline
from src.utils.feature_extraction_matching import KeypointFeatureExtractorAndMatcher
from src.utils.visualize_trajectories import visualize_trajectories
from src.utils.bundle_adjustment import bundle_adjust_local

import argparse

from ipdb import set_trace


def main(
    sequence_parent_dir,
    groundtruth_pose_dir,
    sequence_id,
    gray_or_color,
    min_matches: int = 80,
    keyframe_max_match_ratio: float = 0.92,
    ratio_threshold: float = 0.75,
):  
    
    ground_truth_poses = read_kitti_odometry_poses(
        f"{groundtruth_pose_dir}/{sequence_id}.txt"
    )

    kitti_seq = KITTIOdometrySequence(
        sequence_dir=f"{sequence_parent_dir}/{sequence_id}", gray_or_color=gray_or_color
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

    left_stereo_cam_attributes = kitti_seq.cam0
    cx, cy, fx, fy = (
        left_stereo_cam_attributes["cx"],
        left_stereo_cam_attributes["cy"],
        left_stereo_cam_attributes["fx"],
        left_stereo_cam_attributes["fy"],
    )

    baseline = kitti_seq.cam1["baseline"]

    for i in range(0, len(kitti_seq)):

        left, right, t_step = kitti_seq.get_frame(i)

        left_name = left['name'].split('.')[0]

        gt_pose = ground_truth_poses[left_name]


        # accepted, kp, des, reason = keyframe_selector.should_add(left2)

        result = pipeline.register_stereo_frame_pair(t_step, left, right, gt_pose)

        
        if i > 0 and i % 20 == 0:
            print(f"Performing local bundle adjustment at frame {i}...")
            ba_stats = bundle_adjust_local(
                        sfm_map=landmark_tracker,
                        window_size=3,
                        max_nfev=40,
                        max_points=200,
                    )
            print(f"  Bundle adjustment stats: {ba_stats}")
        

        if i > 0 and i % 100 == 0:
            print(f"Frame {i}: Filtering outliers from landmark map...")
            landmark_tracker.filter_outlier_points()

        if i == 50:
            break

    # Final filtering
    landmark_tracker.filter_outlier_points()

    visualize_trajectories(
        sfm_map=landmark_tracker,
        ground_truth_poses=ground_truth_poses,
        anchor_first_frame=True,
        use_umeyama=True,
        scale_without_umeyama=False,
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

    args = parser.parse_args()

    main(
        sequence_parent_dir=args.sequence_parent_dir,
        groundtruth_pose_dir=args.groundtruth_pose_parent_dir,
        sequence_id=args.sequence_id,
        gray_or_color=args.gray_or_color,
    )

"""

python src/run_stereo_visual_odometry.py \
    --sequence_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/ \
    --groundtruth_pose_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/ground_truth_poses/poses/ \
    --sequence_id 00 --gray_or_color gray

        
        
"""
