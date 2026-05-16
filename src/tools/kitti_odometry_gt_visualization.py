import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json


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


def detect_loops(positions, distance_threshold=5.0, min_loop_separation=50, spatial_cluster_threshold=5.0):
    """
    Detect loops based on spatial proximity in the trajectory.
    
    Multiple detections of loop closures where the robot starts from similar spatial locations
    are merged into a single loop to avoid detecting the same loop multiple times.
    
    Algorithm:
    1. Find all potential loop closure candidates
    2. Cluster candidates by start position proximity (same location = same loop)
    3. For each cluster, keep only the best quality closure
    
    Args:
        positions: (N, 3) array of positions
        distance_threshold: Maximum distance to consider as a loop closure (meters)
        min_loop_separation: Minimum frame separation between loop start and end
        spatial_cluster_threshold: Max distance between start positions to group loops (meters)
    
    Returns:
        loops: List of dicts with keys 'start_frame', 'end_frame', 'distance'
    """
    # First pass: find all potential loop closure candidates
    candidates = []
    n_frames = len(positions)
    
    for i in range(min_loop_separation, n_frames):
        current_pos = positions[i]
        past_positions = positions[:i - min_loop_separation]
        
        if len(past_positions) == 0:
            continue
        
        distances = np.linalg.norm(past_positions - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        if closest_distance < distance_threshold:
            candidates.append({
                'start_frame': int(closest_idx),
                'end_frame': int(i),
                'distance': float(closest_distance),
                'start_pos': positions[closest_idx].copy()
            })
    
    if not candidates:
        return []
    
    # Second pass: cluster candidates by start position proximity
    # Use union-find to group candidates that start from similar locations
    n_candidates = len(candidates)
    parent = list(range(n_candidates))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Compare all pairs of candidates
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            cand_i = candidates[i]
            cand_j = candidates[j]
            
            # If they start from similar positions, they detect the same loop region
            start_dist = np.linalg.norm(cand_i['start_pos'] - cand_j['start_pos'])
            
            if start_dist < spatial_cluster_threshold:
                union(i, j)
    
    # Third pass: group candidates by cluster and select best from each
    clusters = {}
    for i in range(n_candidates):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(candidates[i])
    
    # Select best loop from each cluster (smallest spatial distance)
    loops = []
    for cluster in clusters.values():
        best_loop = min(cluster, key=lambda x: x['distance'])
        loops.append({
            'start_frame': best_loop['start_frame'],
            'end_frame': best_loop['end_frame'],
            'distance': best_loop['distance']
        })
    
    # Sort by end frame for chronological order
    loops.sort(key=lambda x: x['end_frame'])
    
    return loops


def plot_2d_trajectory(positions, title="KITTI Trajectory (Top View)", loops=None):
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
    
    # Plot loops
    if loops:
        for loop in loops:
            start_idx = loop['start_frame']
            end_idx = loop['end_frame']
            plt.scatter(x[start_idx], z[start_idx], c="blue", s=100, marker="o", edgecolors="darkblue", linewidths=2)
            plt.scatter(x[end_idx], z[end_idx], c="orange", s=100, marker="s", edgecolors="darkorange", linewidths=2)
            # Draw line connecting loop points
            plt.plot([x[start_idx], x[end_idx]], [z[start_idx], z[end_idx]], 'k--', alpha=0.5, linewidth=1)

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(positions, title="KITTI Trajectory (3D)", loops=None):
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
    
    # Plot loops
    if loops:
        for loop in loops:
            start_idx = loop['start_frame']
            end_idx = loop['end_frame']
            ax.scatter(x[start_idx], y[start_idx], z[start_idx], c="blue", s=80, marker="o", edgecolors="darkblue", linewidths=2)
            ax.scatter(x[end_idx], y[end_idx], z[end_idx], c="orange", s=80, marker="s", edgecolors="darkorange", linewidths=2)
            # Draw line connecting loop points
            ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], [z[start_idx], z[end_idx]], 'k--', alpha=0.5, linewidth=1)

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


def save_trajectory_analysis(pose_file, positions, loops, output_dir="outputs"):
    """
    Save trajectory analysis including frame count and loop detection to log file.
    
    Args:
        pose_file: Path to the pose file
        positions: (N, 3) array of positions
        loops: List of detected loops
        output_dir: Directory to save the log file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract pose file name for log filename
    pose_name = Path(pose_file).stem
    log_file = output_dir / f"trajectory_analysis_{pose_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAJECTORY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Pose File: {pose_file}\n")
        f.write("\n")
        
        # Frame count
        f.write("FRAME INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Number of Frames: {len(positions)}\n")
        f.write(f"Start Position: [{positions[0][0]:.4f}, {positions[0][1]:.4f}, {positions[0][2]:.4f}]\n")
        f.write(f"End Position: [{positions[-1][0]:.4f}, {positions[-1][1]:.4f}, {positions[-1][2]:.4f}]\n")
        f.write("\n")
        
        # Trajectory statistics
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        f.write("TRAJECTORY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Distance Traveled: {total_distance:.4f} meters\n")
        f.write(f"Average Step Distance: {np.mean(distances):.4f} meters\n")
        f.write(f"Max Step Distance: {np.max(distances):.4f} meters\n")
        f.write(f"Min Step Distance: {np.min(distances):.4f} meters\n")
        f.write("\n")
        
        # Loop detection results
        f.write("LOOP DETECTION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Loops Detected: {len(loops)}\n")
        f.write("\n")
        
        if loops:
            f.write("Loop Details:\n")
            for i, loop in enumerate(loops, 1):
                start_frame = loop['start_frame']
                end_frame = loop['end_frame']
                distance = loop['distance']
                frame_separation = end_frame - start_frame
                
                f.write(f"\nLoop {i}:\n")
                f.write(f"  Start Frame: {start_frame}\n")
                f.write(f"  End Frame: {end_frame}\n")
                f.write(f"  Frame Separation: {frame_separation}\n")
                f.write(f"  Spatial Distance: {distance:.4f} meters\n")
                f.write(f"  Start Position: [{positions[start_frame][0]:.4f}, {positions[start_frame][1]:.4f}, {positions[start_frame][2]:.4f}]\n")
                f.write(f"  End Position: [{positions[end_frame][0]:.4f}, {positions[end_frame][1]:.4f}, {positions[end_frame][2]:.4f}]\n")
        else:
            f.write("No loops detected.\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    # Also save JSON format for programmatic access
    json_file = output_dir / f"trajectory_analysis_{pose_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analysis_data = {
        "pose_file": str(pose_file),
        "timestamp": datetime.now().isoformat(),
        "frame_count": len(positions),
        "start_position": positions[0].tolist(),
        "end_position": positions[-1].tolist(),
        "total_distance": float(total_distance),
        "loops": loops
    }
    with open(json_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return log_file, json_file


def main():
    pose_file = "/Users/krishna/Downloads/Datasets/KITTI/data_odometry_poses_gt/poses/08.txt"
    output_dir = "/Users/krishna/CodingProjects/spatial-vision-lab/outputs/kiti_ground_truth_odometry_visualization"

    p = Path(output_dir).mkdir(parents=True, exist_ok=True)

    poses = load_kitti_poses(pose_file)
    positions = extract_positions(poses)
    
    # Detect loops in trajectory
    loops = detect_loops(positions, distance_threshold=5.0, min_loop_separation=50)

    print(f"Loaded {len(poses)} poses from: {pose_file}")
    print(f"Start position: {positions[0]}")
    print(f"End position:   {positions[-1]}")
    print(f"\nDetected {len(loops)} loops")
    
    if loops:
        print("\nLoop Details:")
        for i, loop in enumerate(loops, 1):
            print(f"  Loop {i}: Frame {loop['start_frame']} → {loop['end_frame']}, distance: {loop['distance']:.4f}m")
    
    # Save analysis to log file
    log_file, json_file = save_trajectory_analysis(pose_file, positions, loops, output_dir=output_dir)
    print(f"\nAnalysis saved to:")
    print(f"  Log: {log_file}")
    print(f"  JSON: {json_file}")

    plot_2d_trajectory(positions, title="KITTI Seq 00 Trajectory (X-Z Top View)", loops=loops)
    plot_3d_trajectory(positions, title="KITTI Seq 00 Trajectory (3D)", loops=loops)


if __name__ == "__main__":
    main()

# python src/tools/kitti_odometry_gt_visualization.py