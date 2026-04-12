
import cv2
import numpy as np
from src.datasets.kitti_odometry import KITTIOdometrySequence

from ipdb import set_trace as st

def main():
    kitti_seq = KITTIOdometrySequence(
        sequence_dir="/Volumes/SSD_256/KITTI/VisualOdometry/gray_data/sequences/00",
        gray_or_color="gray",
    )

    # Test loading the first sequence
    print("Number of frames:", len(kitti_seq))
    print("P_left:\n", kitti_seq.left_P)
    print("P_right:\n", kitti_seq.right_P)

    print("Camera 0 intrinsics:", kitti_seq.cam0)
    print("Camera 1 intrinsics:", kitti_seq.cam1)

    st()

    for i in range(len(kitti_seq)):
        left, right, t = kitti_seq.get_frame(i)

        print(f"Frame {i:06d}, time={t:.3f}, shape={left.shape}")

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