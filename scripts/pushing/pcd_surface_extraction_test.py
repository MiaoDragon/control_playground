# try to extract the pcd that is on the surface of the pcd

import open3d as o3d
import numpy as np

rectangle_pt = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(5000,3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(rectangle_pt)
o3d.visualization.draw_geometries([pcd])

# downsample
pcd2 = pcd.farthest_point_down_sample(200)
o3d.visualization.draw_geometries([pcd2])

# conclusion: farthest_point_down_sample does not extract the surface