from dataclasses import dataclass
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

@dataclass
class PoseGraphEdge:
    source: str
    target: str
    relative_T : np.ndarray
    weight: float  = 1.0
    edge_type : str = "odom"  # "odom" or "loop"

def world_to_camera_T(R, t)->np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def camera_to_world_T(R, t) -> np.ndarray:
    return np.linalg.inv(world_to_camera_T(R, t))


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asanyarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative_T(T_source: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Relative transform from target frame to source frame, assuming both are poses 
    in the same world frame.
    """
    return invert_T(T_source) @ T_target

def transform_to_vec(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.hstack((Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]))

def vec_to_transform(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(vec[:3]).as_matrix()
    T[:3, 3] = vec[3:]
    return T


def pose_error(measured_relative_T: np.ndarray, estimated_relative_T: np.ndarray) -> np.ndarray:
    """
    Measured transform = what your sensor/algorithm estimated between two keyframes.
    Predicted relative transform = what your current global pose estimates imply between those same keyframes.
    """
    
    error_T = invert_T(measured_relative_T) @ estimated_relative_T
    return transform_to_vec(error_T)

def build_odometry_edges(keyframes, weight: float=1.0) -> list[PoseGraphEdge]:
    edges = []
    records = sorted(keyframes, key=lambda rec: rec.frame_index)

    all_records_except_last = records[:-1]
    all_records_except_first = records[1:]

    for source, target in zip(all_records_except_last, all_records_except_first):
        source_T = camera_to_world_T(source.R,  source.t)
        target_T = camera_to_world_T(target.R, target.t)

        edges.append(PoseGraphEdge(
                source=source.name,
                target=target.name,
                relative_T=relative_T(source_T, target_T),
                weight=weight,
                edge_type="odom",
                )
                )

    return edges
    
def optimize_pose_graph(
        keyframes,
        loop_edges: list[PoseGraphEdge],
        odom_weight: float = 1.0,
        loop_weight: float = 1.0,
        max_nfev: int = 100, 
):
    keyframes = sorted(keyframes, key=lambda rec:rec. frame_index)
    
    if len(keyframes) < 2:
        return {}, {"success": True, "reason": "not_enough_keyframes", "n_edges": 0}
    
    names = [rec.name for rec in keyframes]
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    fixed_name = names[0]
    
    # convert all poses to world frame
    initial_poses = {
        rec.name:  camera_to_world_T(rec.R, rec.t) for rec in keyframes
    }

    edges = build_odometry_edges(keyframes, weight=odom_weight)
    edges.extend(
        PoseGraphEdge(source=edge.source,
                      target=edge.target,
                      relative_T=edge.relative_T,
                      weight=loop_weight*edge.weight,
                      edge_type=edge.edge_type) #NOTE edge_type = loop?
                        for edge in loop_edges if edge.source in name_to_idx and edge.target in name_to_idx
    )

    if not edges:
        return initial_poses, {"success": True, "reason": "no_edges", "n_edges": 0}
    
    variable_names = names[1:]
    variable_name_to_offset = {name: idx for idx, name in enumerate(variable_names)}
    x0 = np.concatenate([transform_to_vec(initial_poses[name]) for name in variable_names])

    # Each edge residual only depends on its source and target poses, so give
    # least_squares the sparsity pattern to avoid dense finite-difference work.
    jac_sparsity = lil_matrix((len(edges) * 6, len(variable_names) * 6), dtype=np.int8)
    for edge_idx, edge in enumerate(edges):
        row_start = edge_idx * 6
        for name in (edge.source, edge.target):
            col_offset = variable_name_to_offset.get(name)
            if col_offset is None:
                continue
            jac_sparsity[
                row_start:row_start + 6,
                col_offset * 6:(col_offset + 1) * 6,
            ] = 1
    jac_sparsity = jac_sparsity.tocsr()

    
    def unpack(x):
        poses = {fixed_name: initial_poses[fixed_name]}
        for offset, name in enumerate(variable_names):
            poses[name] = vec_to_transform(x[offset*6:(offset+1)*6])
        return poses
    
    def residuals(x):
        poses = unpack(x)
        residual_blocks = []

        for edge in edges:
            estimated_relative = relative_T(poses[edge.source], poses[edge.target])
            # weight * measuered vs estimated residual
            residual_blocks.append(np.sqrt(edge.weight) * pose_error(edge.relative_T, estimated_relative))

        return np.concatenate(residual_blocks)
    
    initial_cost = float(np.sum(residuals(x0) ** 2))
    # mininize residuals
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=jac_sparsity,
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
    )

    optimized_poses = unpack(result.x)
    final_cost = float(np.sum(residuals(result.x)**2))

    stats = {
        "success": bool(result.success),
        "message": result.message,
        "n_keyframes": len(keyframes),
        "n_edges": len(edges),
        "n_loop_edges": sum(edge.edge_type == "loop" for edge in edges),
        "initial_cost": initial_cost,
        "final_cost": final_cost,

    }

    return optimized_poses, stats

def apply_optimized_poses(keyframe_db, landmark_map, optimized_camera_to_world: dict[str, np.ndarray]):
    # update poses in both keyframe_db and landmark_map

    
    for record in keyframe_db.records:
        T_wc = optimized_camera_to_world.get(record.name)
        if T_wc is None:
            continue

        T_cw = invert_T(T_wc)
        record.R = T_cw[:3, :3]
        record.t = T_cw[:3, 3].reshape(3, 1)

    for camera in landmark_map.cameras:
        T_wc = optimized_camera_to_world.get(camera.name)
        if T_wc is None:
            continue

        T_cw = invert_T(T_wc)
        camera.R = T_cw[:3, :3]
        camera.t = T_cw[:3, 3].reshape(3, 1)



    


