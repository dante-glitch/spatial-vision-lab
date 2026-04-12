"""
Consecutive frames can be too similar (low parallax -> bad triangulation) or too dissimilar (large baseline -> hard to match features).
So we need a keyframe selection strategy to select frames that are "just right" for triangulation and matching.


This module provides a lightweight keyframe selector that accepts a 
new stereo frame pair only if the left frame at time 't' gas enough good matches with the 
previous keyframe left image.

"""


from dataclasses import dataclass, field
from src.utils.feature_extraction_matching import KeypointFeatureExtractorAndMatcher

@dataclass
class KeyframeSelector:
    """
    Decides whether an incoming frame is distinct enough from the last
    accepted keyframe to be worth adding to the reconstruction.

    Parameters
    ----------
    min_matches      : int   — minimum inlier matches required to accept frame
    max_match_ratio  : float — if matched/detected > this, baseline too small
    feature_type     : str   — 'sift' (default)
    ratio_threshold  : float — Lowe ratio test threshold
    """

    min_matches     : int   = 80
    max_match_ratio : float = 0.92
    ratio_threshold : float = 0.75

    _prev_kp        : list  = field(default_factory=list, init=False, repr=False)
    _prev_des       : object = field(default=None,         init=False, repr=False)
    _prev_name      : str   = field(default="",            init=False, repr=False)
    _feature_detector_matcher  : object = field(default=None,         init=False, repr=False)
    
    def __post_init__(self):
        self._feature_detector_matcher = KeypointFeatureExtractorAndMatcher()

    def should_add(self, frame: dict)  -> tuple[bool, list, object]:
        # Evaluate whether *frame* should become a keyframe.
        image = frame["image"]
        kp, des = self._feature_detector_matcher.extract_features(image)

        # Always accept the very first frame
        if self._prev_des is None:
            self._prev_kp, self._prev_des = kp, des
            self._prev_name = frame["name"]
            return True, kp, des, "first frame"
        
        _, n_matches = self._feature_detector_matcher.match_features(self._prev_des, des, self.ratio_threshold)
        n_kp_prev = len(self._prev_kp)

        match_ratio = n_matches / max(n_kp_prev, 1)

        if n_matches < self.min_matches:
            reason = f"too few matches ({n_matches} < {self.min_matches})"
            return False, kp, des, reason
        
        if match_ratio > self.max_match_ratio:
            reason = f"too similar (ratio={match_ratio:.2f} > {self.max_match_ratio})"
            return False, kp, des, reason
        
        # Accept — update previous keyframe
        self._prev_kp, self._prev_des = kp, des
        self._prev_name = frame["name"]
        reason = f"accepted ({n_matches} matches, ratio={match_ratio:.2f})"
        return True, kp, des, reason
    

    def reset(self):
        self._prev_kp   = []
        self._prev_des  = None
        self._prev_name = ""