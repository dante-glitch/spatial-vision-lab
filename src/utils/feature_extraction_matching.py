import cv2
import numpy as np


class KeypointFeatureExtractorAndMatcher:
    """Class to extract keypoint features from an image using SIFT (Scale-Invariant Feature Transform)."""

    def __init__(self, n_features: int = 3000, ratio_threshold: float = 0.75):
        self.extractor = cv2.SIFT_create(nfeatures=n_features)
        self.ratio_threshold = ratio_threshold

    def extract_features(self, image: np.ndarray) -> tuple:
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return keypoints, descriptors

    # Function to match features between two sets of descriptors using BFMatcher
    @staticmethod
    def match_features(
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        ratio_threshold: float = 0.75,
    ):
        """
        Match features between two sets of descriptors using BFMatcher and apply ratio test.

        Args:
            descriptors1: Descriptors from the first image.
            descriptors2: Descriptors from the second image.
            ratio_threshold: Threshold for the ratio test (default is 0.75).

        Returns:
            List of matched keypoints.

        Usage:
            p1, kp2, matches, pts1, pts2 = match_features(img1, img2)
        """
        if descriptors1 is None or descriptors2 is None:
            return [], 0

        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return [], 0

        bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            # Knn + Lowe's ratio test
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        return good_matches, len(good_matches)
