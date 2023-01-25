import numpy as np

def rot_mat_from_ang(ang):
    """
    get rotation matrix from angle (in radians)
    """
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

def transform_vec_to_mat(vec):
    rot_mat = np.eye(3)
    rot_mat[:2,:2] = rot_mat_from_ang(vec[2])
    rot_mat[:2,2] = vec[:2]
    return rot_mat

def pcd_to_grid(pcd, resols):
    """
    produce a mask grid for the pcd
    resols: shape of 2 for the resolution of the grid    
    Assume the grid has its center as the origin, and is the same as the pcd center
    
    return:
    - recentered and shifted pcd
    - grid
    """
    mins = pcd.min(axis=0)
    maxs = pcd.max(axis=0)
    size = (maxs - mins) / resols
    size = np.ceil(size).astype(int)+2
    grid_mid = (size * resols) / 2
    pcd = pcd - pcd.mean(axis=0) + grid_mid
    grid = np.zeros(size).astype(bool)
#     pcd_indices = pcd - pcd.min(axis=0)
    pcd_indices = pcd / resols
    pcd_indices = np.floor(pcd_indices).astype(int)
    grid[pcd_indices[:,0], pcd_indices[:,1]] = 1
    return pcd-pcd.mean(axis=0), grid
