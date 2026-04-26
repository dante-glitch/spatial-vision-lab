from dataclasses import dataclass
import numpy as np

@dataclass
class KeyFrameRecord:
    name: str
    keypoints_xy: list #  np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    descriptors: object
    R: np.ndarray
    t: np.ndarray

    # keypoint index -> landmark index in LandmarkMapping.points3d
    kp_to_landmark: dict[int, int] 

    frame_index: int | None = None
    timestamp: float | None = None


class KeyframeDatabase:
    def __init__(self):
        self.records: list[KeyFrameRecord] = []
        self.by_name: dict[str, KeyFrameRecord] = {}

    def add(self, record: KeyFrameRecord):
        self.records.append(record)
        self.by_name[record.name] = record

    def get(self, name: str) -> KeyFrameRecord:
        return self.by_name[name]
    
    def candidates(self, current_name: str, min_seperation: int = 30):
        current_index = self.by_name[current_name].frame_index
        return [rec for rec in self.records if rec.frame_index is not None and abs(rec.frame_index - current_index) >= min_seperation]
    

