import cv2
import numpy as np


class DenseStereoDisparity:
    def __init__(
        self,
        gray_or_color: str,
        minDisparity: int = 0,
        numDisparities: int = 16 * 22,
        blockSize: int = 5,
        disp12MaxDiff: int = 1,
        uniquenessRatio: int = 10,
        speckleWindowSize: int = 100,
        speckleRange: int = 2,
    ):
        """
        Semi-Global Block Matching -> algorithm

        A. Build cost volume: For each left pixel (x,y) and disparity d, compute C(x,y,d) -> cost volume
        B. Aggregate costs semi-globally and globally
            - neighboring pixels should usually have similar disparity
            - except at depth discontinuities
            So it aggregates matching costs along multiple directions through the image. This is the semi-global smoothing.
            For global smoothing we consider the parameters P1 and P2.
            P1: Penalty for a small disparity change between neighboring pixels. P1=8⋅channels⋅blockSize⋅blockSize (num channels 1 for grayscale, 3 for color)
            P2: Penalty for a large disparity change between neighboring pixels. P2=32⋅channels⋅blockSize⋅blockSize

            Intuition
                P1: “small disparity changes are okay”
                P2: “big disparity jumps are suspicious unless strongly supported”

        C. Disparity selection: For each pixel, select the disparity with the lowest aggregated cost.
        D. Disparity refinement: Post-process the disparity map to improve accuracy and reduce noise

        SGBM struggles with:

        reflective surfaces
        transparent windows
        weak texture
        repetitive patterns
        occlusions
        thin objects
        very dark / saturated regions


        """
        num_channels = 1 if gray_or_color == "gray" else 3

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,  # start searching disparities from 0
            numDisparities=numDisparities,  # search disparities up to 352 (must be divisible by 16, compute efficiency) . At d=352, b = 0.54, f=718. Max depth = 1.1m approx.
            blockSize=blockSize,  # window size used for local matching.. small patch gives more texture and makes matching more stable.
            P1=8
            * num_channels
            * blockSize
            * blockSize,  # Penalty for a small disparity change
            P2=32
            * num_channels
            * blockSize
            * blockSize,  # Penalty for a large disparity change
            disp12MaxDiff=disp12MaxDiff,  # left-right consistency check. disparity from left → right should be consistent with disparity from right → left . Max delta is  disp12MaxDiff. If not, mark as invalid. This helps filter out bad matches.
            uniquenessRatio=uniquenessRatio,  # Best disparity match must be sufficiently better than the second best match. Larger value = stricter filtering
            speckleWindowSize=speckleWindowSize,  # Removes tiny isolated disparity blobs, called speckles. A speckle is a small connected region of disparity values that is likely noise.
            speckleRange=speckleRange,  # How much disparity variation is allowed inside a speckle region.
        )

    def compute_disparity(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> np.ndarray:
        disp = (
            self.stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        )  # OpenCV returns disparity in fixed-point format, scaled by 16.

        return disp
