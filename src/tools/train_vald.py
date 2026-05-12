
#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import os
import cv2
import faiss
import numpy as np

from ipdb import set_trace

# VLAD Vectors — Vector of Locally Aggregated Descriptors


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (denom + eps)


def collect_image_paths(sequence_parent_dir: str, gray_or_color: str = "gray") -> list[Path]:
    
    image_dir_name = "image_0" if gray_or_color == "gray" else "image_2"

    sequence_parent = Path(os.path.join(sequence_parent_dir, image_dir_name))

    image_paths = sorted(sequence_parent.glob("*.png"))

    return image_paths




def sample_image_paths(
    image_paths: list[Path],
    every_nth: int,
    max_images: int | None,
) -> list[Path]:
    sampled = image_paths[::every_nth]
    if max_images is not None:
        sampled = sampled[:max_images]
    return sampled



def extract_sift_descriptors(
    image_paths: list[Path],
    n_features: int = 3000,
    max_desc_per_image: int = 500,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    sift = cv2.SIFT_create(nfeatures=n_features)

    all_desc = []
    per_image_desc = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        _kp, des = sift.detectAndCompute(image, None)
        if des is None or len(des) == 0:
            continue

        des = des.astype(np.float32)

        if len(des) > max_desc_per_image:
            idx = np.random.choice(len(des), size=max_desc_per_image, replace=False)
            des = des[idx]

        all_desc.append(des)
        per_image_desc.append((str(img_path), des))

    if not all_desc:
        raise RuntimeError("No descriptors extracted from the provided images.")

    return np.vstack(all_desc), per_image_desc




def train_cluster_centers(
    descriptors: np.ndarray,
    n_clusters: int = 64,
    max_iter: int = 100,
    attempts: int = 3,
) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        1e-4,
    )

    _compactness, _labels, centers = cv2.kmeans(
        descriptors,
        n_clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32)


def compute_vlad(descriptors: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray:
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(cluster_centers.shape[0] * cluster_centers.shape[1], dtype=np.float32)

    descriptors = np.asarray(descriptors, dtype=np.float32)
    cluster_centers = np.asarray(cluster_centers, dtype=np.float32)

    # squared L2 distance matrix
    dists = (
        np.sum(descriptors ** 2, axis=1, keepdims=True)
        - 2.0 * descriptors @ cluster_centers.T
        + np.sum(cluster_centers ** 2, axis=1, keepdims=True).T
    )
    assignments = np.argmin(dists, axis=1)

    k, d = cluster_centers.shape
    vlad = np.zeros((k, d), dtype=np.float32)

    for cluster_id in range(k):
        mask = assignments == cluster_id
        if not np.any(mask):
            continue
        residuals = descriptors[mask] - cluster_centers[cluster_id]
        vlad[cluster_id] = residuals.sum(axis=0)

    # intra-normalization
    vlad = l2_normalize(vlad, axis=1)
    vlad = vlad.reshape(-1)
    vlad = l2_normalize(vlad.reshape(1, -1), axis=1).reshape(-1)
    return vlad.astype(np.float32)


def build_vlad_matrix(
    per_image_desc: list[tuple[str, np.ndarray]],
    cluster_centers: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    names = []
    vecs = []

    for img_path, des in per_image_desc:
        vlad = compute_vlad(des, cluster_centers)
        names.append(img_path)
        vecs.append(vlad)

    return names, np.vstack(vecs).astype(np.float32)


def build_faiss_index(
    vlad_vectors: np.ndarray,
    index_type: str = "flat",
    nlist: int = 100,
) -> faiss.Index:
    dim = vlad_vectors.shape[1]

    if index_type == "flat":
        # no faiss training step
        index = faiss.IndexFlatIP(dim)
        index.add(vlad_vectors)
        return index

    if index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vlad_vectors)
        index.add(vlad_vectors)
        return index

    raise ValueError(f"Unsupported index_type: {index_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_parent_dir", type=str, required=True)
    parser.add_argument("--gray_or_color", type=str, choices=["gray", "color"], default="gray")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--every_nth", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=3000)
    parser.add_argument("--n_features", type=int, default=3000)
    parser.add_argument("--max_desc_per_image", type=int, default=500)

    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--max_kmeans_descriptors", type=int, default=200000)

    parser.add_argument("--faiss_index_type", type=str, choices=["flat", "ivfflat"], default="flat")
    parser.add_argument("--nlist", type=int, default=100)

    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting image paths...")
    image_paths = collect_image_paths(args.sequence_parent_dir, args.gray_or_color)
    image_paths = sample_image_paths(image_paths, args.every_nth, args.max_images)
    print(f"Using {len(image_paths)} images")

    print("Extracting SIFT descriptors...")
    all_descriptors, per_image_desc = extract_sift_descriptors(
        image_paths=image_paths,
        n_features=args.n_features,
        max_desc_per_image=args.max_desc_per_image,
    )
    print(f"Total descriptor rows before cap: {len(all_descriptors)}")

    if len(all_descriptors) > args.max_kmeans_descriptors:
        idx = np.random.choice(
            len(all_descriptors),
            size=args.max_kmeans_descriptors,
            replace=False,
        )
        all_descriptors = all_descriptors[idx]
    print(f"Descriptor rows used for kmeans: {len(all_descriptors)}")

    print("Training VLAD cluster_centers...")
    cluster_centers = train_cluster_centers(
        descriptors=all_descriptors,
        n_clusters=args.n_clusters,
    )

    cluster_centers_path = output_dir / "vlad_cluster_centers.npy"
    np.save(cluster_centers_path, cluster_centers)
    print(f"Saved cluster_centers to {cluster_centers_path}")

    print("Computing VLAD vectors...")
    image_names, vlad_vectors = build_vlad_matrix(per_image_desc, cluster_centers)
    vlad_vectors = l2_normalize(vlad_vectors, axis=1).astype(np.float32)
    print(f"VLAD matrix shape: {vlad_vectors.shape}")

    print(f"Building Faiss index: {args.faiss_index_type}")
    index = build_faiss_index(
        vlad_vectors=vlad_vectors,
        index_type=args.faiss_index_type,
        nlist=args.nlist,
    )

    index_path = output_dir / f"loop_index_{args.faiss_index_type}.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved Faiss index to {index_path}")

    names_path = output_dir / "index_image_names.json"
    names_path.write_text(json.dumps(image_names, indent=2) + "\n")
    print(f"Saved image name map to {names_path}")

    meta = {
        "gray_or_color": args.gray_or_color,
        "every_nth": args.every_nth,
        "max_images": args.max_images,
        "n_features": args.n_features,
        "max_desc_per_image": args.max_desc_per_image,
        "n_clusters": args.n_clusters,
        "max_kmeans_descriptors": args.max_kmeans_descriptors,
        "faiss_index_type": args.faiss_index_type,
        "nlist": args.nlist,
        "num_indexed_images": len(image_names),
        "vlad_dim": int(vlad_vectors.shape[1]),
    }
    meta_path = output_dir / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved metadata to {meta_path}")




if __name__ == "__main__":
    main()


"""

python src/tools/train_vald.py \
    --sequence_parent_dir /Users/krishna/Downloads/Datasets/KITTI/data_odometry_gray/sequences/07 \
    --gray_or_color gray \
    --output_dir /Users/krishna/CodingProjects/spatial-vision-lab/outputs/vlad_train/seq_07 \
    --every_nth 5 \
    --max_images 3000 \
    --n_features 3000 \
    --max_desc_per_image 500 \
    --n_clusters 64 \
    --faiss_index_type flat

"""