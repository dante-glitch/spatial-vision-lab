# Stereo SLAM System

This module implements a complete stereo SLAM (Simultaneous Localization and Mapping) system for KITTI odometry sequences. It combines visual odometry with loop closure detection, pose graph optimization, and global bundle adjustment to estimate accurate camera trajectories and build a sparse landmark map.

**Key features:**
- Stereo visual odometry with PnP pose estimation
- Loop closure detection using VLAD descriptors
- Pose graph optimization for global consistency
- Local and global bundle adjustment for refinement
- KITTI odometry evaluation metrics


## Data Layout

The runner expects the KITTI odometry sequence layout:

```text
<sequence_parent_dir>/
  00/
    calib.txt
    times.txt
    image_0/
      000000.png
      ...
    image_1/
      000000.png
      ...

<groundtruth_pose_parent_dir>/
  00.txt
  01.txt
  ...
```

For grayscale KITTI odometry data, `image_0` and `image_1` are used. For color
data, the dataset loader uses the corresponding color projection matrices.

## Run

From the repository root:

```bash
python src/classical_visual_slam/visual_slam/run_stereo_visual_odometry.py \
  --sequence_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/ \
  --groundtruth_pose_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/ground_truth_poses/poses/ \
  --sequence_id 06 \
  --gray_or_color gray \
  --max_frames_to_process 950 \
  --output_dir outputs/VO/kitti_06_pgo_globalBA \
  --vlad_cluster_centers_path outputs/vlad_train/seq_07/vlad_cluster_centers.npy \
  --loop_checking_frequency 4 \
  --loop_min_temporal_separation 40
```

### Required Arguments

- `--sequence_parent_dir`: Parent directory containing KITTI sequence folders.
- `--groundtruth_pose_parent_dir`: Directory containing KITTI pose files (e.g., `00.txt`, `06.txt`).
- `--sequence_id`: Sequence ID to process (e.g., `00`, `06`).
- `--vlad_cluster_centers_path`: Path to pretrained VLAD cluster centers file (`.npy`). Required for loop detection.

### Optional Arguments

- `--gray_or_color`: Choose `gray` or `color` images (default: `gray`).
- `--max_frames_to_process`: Maximum number of frames to process (default: `200`).
- `--output_dir`: Output directory for logs, metrics, and trajectories (default: `outputs/stereo_slam`).
- `--loop_checking_frequency`: How often to check for loops in keyframes (default: `4`).
- `--loop_min_temporal_separation`: Minimum frame index distance for loop candidates (default: `40`).
- `--loop_pnp_min_correspondences`: Minimum 3D-2D correspondences for loop PnP RANSAC (default: `6`).
- `--loop_pnp_min_inliers`: Minimum PnP RANSAC inliers to accept a loop closure (default: `50`).

## Outputs

The runner writes:

```text
<output_dir>/
  run.log
  metrics.json
  estimated_trajectory.txt
  groundtruth_trajectory.txt
  frame_ids.txt
```

`estimated_trajectory.txt` and `groundtruth_trajectory.txt` use KITTI odometry
pose row format:

```text
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

`frame_ids.txt` records the registered frame ids used for the saved trajectories.
This matters when keyframe selection skips frames.

`metrics.json` includes the official KITTI odometry subsequence metric under
`kitti_subsequence`. The summary reports mean translation drift in percent and
mean rotation drift in degrees per 100 meters over 100 m to 800 m subsequences.
These are the KITTI leaderboard-style numbers; they are most directly comparable
when evaluated on the full registered frame sequence rather than a sparse
keyframe-only trajectory.

## Pipeline Summary

### Main Processing Loop
1. Load KITTI calibration, timestamps, stereo images, and ground-truth poses.
2. Use SIFT features and ratio-test matching for keypoint detection and descriptor computation.
3. Initialize the first registered frame at the origin with stereo-based 3D landmarks.
4. For each subsequent keyframe:
   - Build 3D-2D correspondences from landmark observations in recent frames.
   - Estimate the current camera pose using `solvePnPRansac` (robust to outliers).
   - Triangulate new stereo points and add them to the landmark map.
   - Track observations of 3D landmarks in the current frame.

### Periodic Refinement
- **Local bundle adjustment** (every 20 frames): Refines a sliding window of recent camera poses and their observed 3D points.
- **Landmark filtering** (every 100 frames): Removes outlier 3D points with invalid depth or low visibility.
- **Loop detection** (every N keyframes): Query recent frame descriptors against VLAD-encoded keyframe database to detect potential loop closures.

### Global Optimization (End-of-Sequence)
- **Pose graph optimization**: If loop closures are detected, optimize all camera poses and odometry constraints jointly to correct global drift.
- **Global bundle adjustment**: Refine all camera poses (except the first, which is fixed) and all multi-view 3D points across the entire sequence using least-squares optimization with robust loss.

### Post-Processing
- Final landmark filtering to remove remaining outliers.
- Evaluate trajectory against KITTI ground truth (ATE, RPE, subsequence metrics).
- Save estimated and ground-truth trajectories in KITTI format.
- Visualize estimated vs. ground-truth trajectories.

## Notes

### SLAM System Design

This stereo SLAM system uses a hybrid optimization approach:

- **Local bundle adjustment** provides fast incremental refinement during the main processing loop, preventing drift accumulation while maintaining real-time performance.
- **Loop closure detection** via VLAD descriptors identifies revisited scenes, enabling global consistency correction.
- **Pose graph optimization** corrects accumulated odometry drift by optimizing all camera poses jointly while fixing odometry edges and adding loop constraints.
- **Global bundle adjustment** performs a final full-map refinement, jointly optimizing all camera poses and 3D landmarks for maximum accuracy.

### Key Components

- **Stereo initialization**: First frame plants 3D landmarks from rectified stereo disparity at the origin.
- **PnP tracking**: Subsequent frames are localized via `solvePnPRansac` using 3D-2D correspondences from recent observations.
- **Incremental mapping**: New landmarks are continuously added from stereo matches not yet in the map.
- **Loop detection**: VLAD-based description and re-identification of previously seen places.
- **Pose graph**: Maintains odometry edges (from consecutive keyframes) and loop edges (from closures).
- **Bundle adjustment**: Both sliding-window local BA and full-map global BA refine geometry and poses.

### Performance Considerations

- Local BA runs every 20 frames; adjust for faster/slower systems.
- Loop checking every 4 keyframes; reduce for real-time constraints or increase for tighter consistency.
- Global BA is expensive; cap the number of points (`max_points=500`) to speed up final refinement.
- VLAD clustering is precomputed offline; see `src/tools/train_vlad.py` for training instructions.

## Limitations

This implementation is a research/demo stereo SLAM pipeline, not a fully deployed backend system.

- Local bundle adjustment, global bundle adjustment, and loop closure are run as part of the batch pipeline, not as an asynchronous backend service.
- The system is designed for offline sequence processing and evaluation on KITTI-style data, not for continuous live deployment on embedded hardware.
- Loop closure uses offline VLAD encoding and geometric PnP verification in the main loop rather than a separate backend module.
- Final global bundle adjustment is executed at the end of sequence processing, so it is not part of a real-time backend pose-correction pipeline.
- There is no separate backend process for long-term map management, relocalization, or distributed data streaming.

- The estimated poses stored by the pipeline are world-to-camera poses. The saved
  KITTI trajectory converts them to camera-to-world form.
- Evaluation aligns the estimated trajectory to ground truth with Umeyama alignment.
- Visualization supports first-frame anchoring and optional Umeyama alignment through
  the runner defaults.
- Open3D viewer render-option exports are redirected to a temporary directory and
  ignored by `.gitignore` if created manually.
