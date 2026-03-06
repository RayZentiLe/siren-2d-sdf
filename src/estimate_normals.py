import open3d as o3d
import numpy as np
import sys
import os


ply_filename = sys.argv[1]

# Check if input file exists
if not os.path.exists(ply_filename):
    raise FileNotFoundError(f"Input file not found: {ply_filename}")

# Load your point cloud
pcd = o3d.io.read_point_cloud(ply_filename)

# Estimate normals
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

# Flip normals that point downward (z < 0)
print("Flipping downward normals...")
normals = np.asarray(pcd.normals)
downward_mask = normals[:, 2] < 0
normals[downward_mask] *= -1
pcd.normals = o3d.utility.Vector3dVector(normals)

# Optional: orient normals consistently (slow but good for large surfaces)
# o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(pcd, k=50)

# Save to a new file with normals
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
combined = np.hstack([points, normals])

base_name = os.path.splitext(os.path.basename(ply_filename))[0]

# Generate output file path in the same directory
directory = os.path.dirname(ply_filename)

if directory:  # If there's a directory path
    xyz_file_path = os.path.join(directory, base_name + "_normals.xyz")
else:  # If file is in current directory
    xyz_file_path = base_name + "_normals.xyz"

np.savetxt(xyz_file_path, combined)