
import cv2
import logging
import numpy as np
from src.datasets.kitti_odometry import KITTIOdometrySequence

from ipdb import set_trace as st

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    kitti_seq = KITTIOdometrySequence(
        sequence_dir="/Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/00",
        gray_or_color="gray",
    )

    # Test loading the first sequence
    logger.info("Number of frames: %s", len(kitti_seq))
    logger.info("P_left:\n%s", kitti_seq.left_P)
    logger.info("P_right:\n%s", kitti_seq.right_P)

    logger.info("Camera 0 intrinsics: %s", kitti_seq.cam0)
    logger.info("Camera 1 intrinsics: %s", kitti_seq.cam1)

    st()

    for i in range(len(kitti_seq)):
        left, right, t = kitti_seq.get_frame(i)

        logger.info("Frame %06d, time=%.3f, shape=%s", i, t, left.shape)

        # Example display
        stacked = np.hstack([left, right])
        cv2.imshow("KITTI stereo pair", stacked)

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# python tests/test_kitti_odometry_ds_loader.py
