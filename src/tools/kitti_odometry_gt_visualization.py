import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_kitti_poses(pose_file):
    """
    Load KITTI odometry poses from a txt file.

    Each line is expected to contain 12 floats representing a 3x4 matrix:
        [R | t]

    Returns:
        poses: (N, 3, 4) array
    """
    pose_file = Path(pose_file)

    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    poses = []
    with open(pose_file, "r") as f:
        for line_idx, line in enumerate(f):
            values = line.strip().split()
            if len(values) != 12:
                raise ValueError(
                    f"Line {line_idx + 1} does not have 12 values. "
                    f"Found {len(values)} values."
                )
            pose = np.array([float(v) for v in values], dtype=np.float64).reshape(3, 4)
            poses.append(pose)

    return np.stack(poses, axis=0)


def extract_positions(poses):
    """
    Extract translation vectors (camera positions in KITTI pose convention).

    Args:
        poses: (N, 3, 4)

    Returns:
        positions: (N, 3) array of [x, y, z]
    """
    return poses[:, :, 3]


def plot_2d_trajectory(positions, title="KITTI Trajectory (Top View)"):
    """
    Plot X-Z top-down trajectory, which is the usual KITTI view.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    plt.figure(figsize=(10, 8))
    plt.plot(x, z, linewidth=2, label="Trajectory")
    plt.scatter(x[0], z[0], c="green", s=80, label="Start")
    plt.scatter(x[-1], z[-1], c="red", s=80, label="End")

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(positions, title="KITTI Trajectory (3D)"):
    """
    Plot full 3D trajectory.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2, label="Trajectory")
    ax.scatter(x[0], y[0], z[0], c="green", s=60, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=60, label="End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    # Make axes roughly equally scaled
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def main():
    pose_file = "/Users/krishna/Downloads/Datasets/KITTI/data_odometry_poses_gt/poses/06.txt"

    poses = load_kitti_poses(pose_file)
    positions = extract_positions(poses)

    print(f"Loaded {len(poses)} poses from: {pose_file}")
    print(f"Start position: {positions[0]}")
    print(f"End position:   {positions[-1]}")

    plot_2d_trajectory(positions, title="KITTI Seq 00 Trajectory (X-Z Top View)")
    plot_3d_trajectory(positions, title="KITTI Seq 00 Trajectory (3D)")


if __name__ == "__main__":
    main()

# python src/tools/kitti_odometry_gt_visualization.py