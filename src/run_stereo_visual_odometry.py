


from src.utils.landmark_map import LandmarkMapping
from src.utils.keyframe_selector import KeyframeSelector
from src.datasets.kitti_odometry import KITTIOdometrySequence
from src.stereo_visual_odometry_pipeline import StereoVisualOdometryPipeline
from src.utils.feature_extraction_matching import KeypointFeatureExtractorAndMatcher

import argparse

from ipdb import set_trace


def main(
    sequence_dir: str,
    gray_or_color: str,
    min_matches: int = 80,
    keyframe_max_match_ratio: float = 0.92,
    ratio_threshold: float = 0.75,
):

    kitti_seq = KITTIOdometrySequence(
        sequence_dir=sequence_dir, gray_or_color=gray_or_color
    )

    feature_extractor_matcher = KeypointFeatureExtractorAndMatcher(n_features=3000)

    landmark_tracker = LandmarkMapping(max_history_limit=15)

    keyframe_selector = KeyframeSelector( min_matches=min_matches,
        max_match_ratio = keyframe_max_match_ratio,
        ratio_threshold = ratio_threshold,
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
        ratio_threshold=ratio_threshold
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


        left, right, t = kitti_seq.get_frame(i)

        #accepted, kp, des, reason = keyframe_selector.should_add(left2)

        result = pipeline.register_stereo_frame_pair(left, right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stereo visual odometry on a KITTI sequence."
    )
    parser.add_argument(
        "--sequence_dir",
        type=str,
        required=True,
        help="Path to the KITTI sequence directory.",
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
        sequence_dir=args.sequence_dir,
        gray_or_color=args.gray_or_color,
    )

"""

python src/run_stereo_visual_odometry.py \
    --sequence_dir /Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/00 \
        --gray_or_color gray

        
python src/run_stereo_visual_odometry.py \
    --sequence_dir /Users/krishna/Downloads/Datasets/KITTI/dataset/sequences/00 \
        --gray_or_color gray

        
"""
