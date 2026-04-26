from dataclasses import dataclass, field
import numpy as np
import faiss    

@dataclass
class LoopCandidate:
    frame_name: str
    frame_index: int
    score: float

class LoopDetector:
    def __init__(self, vlad_encoder, min_temporal_seperation: int=30,
                 top_k: int=5, min_score: float=0.2):
        self.vlad_encoder = vlad_encoder
        self.min_temporal_seperation = min_temporal_seperation
        self.top_k = top_k
        self.min_score = min_score

        dim = vlad_encoder.n_clusters * vlad_encoder.descriptor_dim
        self.index = faiss.IndexFlatL2(dim) # cosine similarity via L2 on normalized VLAD
        self.entries: list[dict] = []

    def add_keyframe(self, frame_name: str, frame_index: int, descriptors: np.ndarray):
        vlad = self.vlad_encoder.encode(descriptors).reshape(1, -1)
        self.index.add(vlad)

        self.entries.append({
            "frame_name":  frame_name,
            "frame_index": frame_index,
        })

    def query(self, frame_index: int, descriptors: np.ndarray) -> list[LoopCandidate]:
        if len(self.entries) == 0:
            return []
        
        query_vec = self.vlad_encoder.encode(descriptors).reshape(1, -1)
        k = min(len(self.entries), max(self.top_k *5, self.top_k))
        """
        top_k = 10
        retrieve k = 50
        discard recent frames near current frame
        discard weak matches
        return best 10 remaining
        """

        scores, indices = self.index.search(query_vec, k)

        candidates = []
        seen = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            entry = self.entries[idx]

            if abs(frame_index - entry["frame_index"]) < self.min_temporal_seperation:
                continue

            if score < self.min_score:
                continue

            name = entry["frame_name"]
            
            if name in seen:
                continue
            
            seen.add(name)
            candidates.append(LoopCandidate(frame_name=name, 
                                            frame_index=entry["frame_index"],
                                            score=float(score)))
            
            if len(candidates) >= self.top_k:
                break

            
        return candidates
