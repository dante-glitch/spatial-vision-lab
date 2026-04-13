"""
Evaluation Metrics for Multiview Reconstruction

Computes:
- Per-camera rotation error after global alignment (geodesic, degrees)
- Per-camera position error after global alignment (scene units)
- Absolute Trajectory Error (ATE) — after Umeyama alignment
- Summary table of results.
"""

import numpy as np
from typing import List, Optional
from scipy.spatial.transform import Rotation, Slerp
from src.utils.landmark_map import SfMMap, Camera

