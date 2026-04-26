from dataclasses import dataclass, field
import numpy as np
import cv2

@dataclass
class VLADEncoder:
    cluster_centers: np.ndarray = None  # (K, D) cluster centers

    @property
    def n_clusters(self) -> int:
        if self.cluster_centers is None:
            return 0
        return int(self.cluster_centers.shape[0])
    
    @property
    def descriptor_dim(self) -> int:
        if self.cluster_centers is None:
            return 0

        return int(self.cluster_centers.shape[1])
    
    @classmethod
    def train(cls, descriptor_samples: np.ndarray, n_clusters: int = 64, max_iter: int = 100):
        descriptor_samples = np.asarray(descriptor_samples, dtype=np.float32)

        if descriptor_samples.ndim!=2:
            raise ValueError("descriptor_samples should be a 2D array of shape (N, D)")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1e-4)

        _compactness, labels, centers = cv2.kmeans(descriptor_samples, n_clusters,\
                                                    None, criteria, 3,\
                                                          cv2.KMEANS_PP_CENTERS)
        
        return cls(cluster_centers=centers.astype(np.float32))
    
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters * self.descriptor_dim, dtype=np.float32)
        
        descriptors = np.asarray(descriptors, dtype=np.float32)

        if descriptors.ndim!=2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(f"descriptors should be a 2D array of shape (N, {self.descriptor_dim})")
        
        # L2 distance from each descriptor to each cluster center
        dists = (np.sum(descriptors**2, axis=1, keepdims=True) - 2.0*descriptors @ self.cluster_centers.T + np.sum(self.cluster_centers**2, axis=1, keepdims=True).T)

        assignments = np.argmin(dists, axis=1)

        vlad = np.zeros((self.n_clusters, self.descriptor_dim), dtype=np.float32)

        for k in range(self.n_clusters):
            mask = assignments == k
            if not np.any(mask):
                continue
            residuals = descriptors[mask] - self.cluster_centers[k]
            vlad[k] = residuals.sum(axis=0)

        # intra-normalization
        norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / (norms + 1e-12)

        # flatten + global L2 normalization
        vlad = vlad.reshape(-1)
        vlad /= np.linalg.norm(vlad) + 1e-12
        return vlad.astype(np.float32)
    
    def save(self, path: str):
        np.save(path, self.cluster_centers)

    @classmethod
    def load(cls, path: str):
        instance = cls(cluster_centers=np.load(path).astype(np.float32))
        return instance