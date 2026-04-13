""" "
Read KITTI Odometrydataset files and convert them to a format suitable for training and evaluation.

"""

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def read_calib(calib_path):
    """
    Read KITTI odometry calib.txt (Projection Matrices) into a dict of 3x4 numpy arrays.
    """
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.split(":", 1)
            data = np.array([float(x) for x in value.strip().split()], dtype=np.float64)
            calib[key] = data.reshape(3, 4)
    return calib


def read_kitti_odometry_poses(poses_path):
    """
    Read KITTI odometry ground-truth poses into a per-frame dictionary.

    r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

    Each row in a KITTI poses file contains a 3x4 camera pose matrix in
    row-major order. The returned frame ids match KITTI image filenames:
    "000000", "000001", ...
    """
    poses_path = Path(poses_path)
    poses = {}

    with open(poses_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            values = np.array(
                [float(x) for x in line.strip().split()],
                dtype=np.float64,
            )

            if values.size != 12:
                raise ValueError(
                    f"Expected 12 pose values in {poses_path} line {line_num}, "
                    f"got {values.size}."
                )

            T = values.reshape(3, 4)
            frame_id = f"{len(poses):06d}"
            poses[frame_id] = {
                "R": T[:, :3],
                "t": T[:, 3].reshape(3, 1),
            }

    return poses


def decompose_stereo_projection(P):
    """
     Extract fx, fy, cx, cy, and baseline from a rectified KITTI projection matrix.
     [Tx, Ty, Tz] -> is actually. K*t , which is pixel scaled projection of the translation vector. For rectified stereo, Ty and Tz should be zero, and Tx encodes the baseline in pixels.

     where t=−RC, R is the rectification rotation (identity for rectified), and C is the camera center in world coordinates.

    since R = I in rectified stereo, we have t = -C. For the left camera, C = [0, 0, 0], so t = [0, 0, 0].
    For the right camera, C = [baseline, 0, 0], so t = [-baseline, 0, 0]. Thus, Tx = fx * baseline for the right camera and Tx = 0 for the left camera.

    For P = [fx 0 cx Tx;
              0 fy cy Ty;
              0  0  1 Tz]
     baseline = -Tx / fx
    """
    fx = P[0, 0].item()
    fy = P[1, 1].item()
    cx = P[0, 2].item()
    cy = P[1, 2].item()
    tx = P[0, 3].item()
    baseline = -tx / fx

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "tx": tx,
        "baseline": baseline,
        "K": K,
    }


class KITTIOdometrySequence:
    def __init__(self, sequence_dir: str, gray_or_color: str):
        self.sequence_dir = Path(sequence_dir)

        self.left_dir = self.sequence_dir / "image_0"
        self.right_dir = self.sequence_dir / "image_1"
        self.times_path = self.sequence_dir / "times.txt"
        self.calib_path = self.sequence_dir / "calib.txt"
        self.gray_or_color = gray_or_color

        self.calib = read_calib(self.calib_path)

        if gray_or_color == "gray":
            self.left_P = self.calib["P0"]
            self.right_P = self.calib["P1"]
        elif gray_or_color == "color":
            self.left_P = self.calib["P2"]
            self.right_P = self.calib["P3"]
        else:
            raise ValueError(
                f"Invalid gray_or_color value: {gray_or_color}. Must be 'gray' or 'color'."
            )

        logger.info(
            "KITTI Odometry sequence images are already rectified; using projection matrices directly for stereo geometry."
        )

        self.cam0 = decompose_stereo_projection(self.left_P)
        self.cam1 = decompose_stereo_projection(self.right_P)

        self.left_images = sorted(self.left_dir.glob("*.png"))
        self.right_images = sorted(self.right_dir.glob("*.png"))

        if len(self.left_images) != len(self.right_images):
            raise ValueError("Left/right image counts do not match.")

        with open(self.times_path, "r") as f:
            self.times = [float(line.strip()) for line in f if line.strip()]

        if len(self.times) != len(self.left_images):
            raise ValueError("times.txt length does not match image count.")

    def __len__(self):
        return len(self.left_images)

    def get_frame(self, idx):
        """
        Returns:
            left_img, right_img, timestamp
        """
        import cv2

        flag = (
            cv2.IMREAD_GRAYSCALE if self.gray_or_color == "gray" else cv2.IMREAD_COLOR
        )
        left = cv2.imread(str(self.left_images[idx]), flag)
        right = cv2.imread(str(self.right_images[idx]), flag)

        if left is None or right is None:
            raise FileNotFoundError(f"Could not load frame {idx}")
        
        left_frame = {
            "name": self.left_images[idx].name,
            "image": left,
            "path": str(self.left_images[idx])
        }

        right_frame = {
            "name": self.right_images[idx].name,
            "image": right,
            "path": str(self.right_images[idx])
        }

        return left_frame, right_frame, self.times[idx]
