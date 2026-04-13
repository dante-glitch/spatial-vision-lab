# Stereo Visual Odometry

This module runs a KITTI-style stereo visual odometry pipeline. It loads rectified
left/right image pairs, initializes a sparse landmark map from stereo depth, tracks
landmarks through recent frames with PnP, optionally applies local bundle
adjustment, evaluates against KITTI odometry ground truth, saves trajectory files,
and opens an Open3D trajectory viewer.

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
python src/visual_odometry/run_stereo_visual_odometry.py \
  --sequence_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/ \
  --groundtruth_pose_parent_dir /Volumes/SSD_256/KITTI/VisualOdometry/ground_truth_poses/poses/ \
  --sequence_id 00 \
  --gray_or_color gray \
  --max_frames_to_process 200 \
  --output_dir outputs/kitti_00
```

Arguments:

- `--sequence_parent_dir`: parent directory containing KITTI sequence folders.
- `--groundtruth_pose_parent_dir`: directory containing KITTI pose files such as `00.txt`.
- `--sequence_id`: sequence id, for example `00` or `01`.
- `--gray_or_color`: choose `gray` or `color`.
- `--max_frames_to_process`: maximum number of frames to process.
- `--output_dir`: output directory for logs, metrics, and trajectories.

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

1. Load KITTI calibration, timestamps, stereo images, and ground-truth poses.
2. Use SIFT features and ratio-test matching.
3. Initialize the first registered frame at the origin.
4. Build 3D landmarks from rectified stereo disparity.
5. For later keyframes, find 3D-2D correspondences from recent landmark observations.
6. Estimate the current left-camera pose with `solvePnPRansac`.
7. Add new stereo landmarks in world coordinates.
8. Periodically run local bundle adjustment over a small sliding window.
9. Filter outlier landmarks.
10. Evaluate, save trajectories, and visualize estimated vs. ground-truth motion.

## Notes

- The estimated poses stored by the pipeline are world-to-camera poses. The saved
  KITTI trajectory converts them to camera-to-world form.
- Evaluation aligns the estimated trajectory to ground truth with Umeyama alignment.
- Visualization supports first-frame anchoring and optional Umeyama alignment through
  the runner defaults.
- Open3D viewer render-option exports are redirected to a temporary directory and
  ignored by `.gitignore` if created manually.
