# import numpy as np
# import open3d as o3d
# import os

# '''This script automatically computes normalization parameters from an original point cloud
#    and uses them to rescale a normalized mesh back to GIS coordinates.'''

# # Helpers to handle both meshes and point clouds
# def _load_geometry(input_path):
#     # Try mesh first; if no triangles, treat as point cloud
#     mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=False)
#     if mesh is not None and mesh.has_vertices() and mesh.has_triangles() and len(mesh.triangles) > 0:
#         return mesh, "mesh"
#     pcd = o3d.io.read_point_cloud(input_path)
#     return pcd, "pcd"

# def _get_points(geom, kind):
#     return np.asarray(geom.vertices) if kind == "mesh" else np.asarray(geom.points)

# def _get_normals(geom, kind):
#     return np.asarray(geom.vertex_normals) if kind == "mesh" else np.asarray(geom.normals)

# def _set_points(geom, kind, points):
#     if kind == "mesh":
#         geom.vertices = o3d.utility.Vector3dVector(points)
#     else:
#         geom.points = o3d.utility.Vector3dVector(points)

# def _set_normals(geom, kind, normals):
#     if kind == "mesh":
#         geom.vertex_normals = o3d.utility.Vector3dVector(normals)
#     else:
#         geom.normals = o3d.utility.Vector3dVector(normals)

# def _save_geometry(output_path, geom, kind):
#     if kind == "mesh":
#         o3d.io.write_triangle_mesh(output_path, geom)
#     else:
#         o3d.io.write_point_cloud(output_path, geom)

# def compute_normalization_parameters(original_point_cloud_path, keep_aspect_ratio=True):
#     """
#     Compute normalization parameters from original point cloud using the same logic
#     as the provided normalization code.
#     """
#     print("Computing normalization parameters from original point cloud...")
    
#     # Load original point cloud
#     pcd, _ = _load_geometry(original_point_cloud_path)
#     coords = _get_points(pcd, "pcd")
    
#     print(f"Original point cloud has {len(coords)} points")
    
#     # Center the coordinates (subtract mean)
#     original_translation = np.mean(coords, axis=0, keepdims=True)
#     coords_centered = coords - original_translation
    
#     # Compute bounds
#     if keep_aspect_ratio:
#         # Use scalar min/max (preserve aspect ratio)
#         coord_max = np.amax(coords_centered)
#         coord_min = np.amin(coords_centered)
#         coord_max_arr = np.array([coord_max, coord_max, coord_max], dtype=np.float64)
#         coord_min_arr = np.array([coord_min, coord_min, coord_min], dtype=np.float64)
#     else:
#         # Axis-wise min/max
#         coord_max_arr = np.amax(coords_centered, axis=0)
#         coord_min_arr = np.amin(coords_centered, axis=0)
    
#     # Compute scaling parameters
#     denom = (coord_max_arr - coord_min_arr)
#     denom_safe = denom.copy()
#     denom_safe[denom_safe == 0.0] = 1e-8
    
#     # Store parameters for inverse transformation
#     params = {
#         'original_translation': original_translation[0],  # Remove keepdims dimension
#         'coord_min': coord_min_arr,
#         'coord_max': coord_max_arr,
#         'denom_safe': denom_safe,
#         'keep_aspect_ratio': keep_aspect_ratio
#     }
    
#     print("Computed parameters:")
#     print(f"  Original translation: {params['original_translation']}")
#     print(f"  Coord min: {params['coord_min']}")
#     print(f"  Coord max: {params['coord_max']}")
#     print(f"  Keep aspect ratio: {keep_aspect_ratio}")
    
#     return params

# def inverse_transform_geometry(input_path, output_dir, normalization_params):
#     """
#     Inverse transform normalized geometry back to GIS coordinates using computed parameters.
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load the normalized geometry (mesh or point cloud)
#     geom, kind = _load_geometry(input_path)
#     points = _get_points(geom, kind)
    
#     print(f"Inverse transforming {kind} with {len(points)} points...")
    
#     # Extract parameters
#     original_translation = normalization_params['original_translation']
#     coord_min = normalization_params['coord_min']
#     coord_max = normalization_params['coord_max']
#     denom_safe = normalization_params['denom_safe']
#     keep_aspect_ratio = normalization_params['keep_aspect_ratio']
    
#     # Apply inverse transformation (back to GIS coordinates)
#     # Reverse: self.coords = self.coords * 2.0
#     points = points / 2.0
    
#     # Reverse: self.coords = self.coords - 0.5
#     points = points + 0.5
    
#     # Reverse: self.coords = (coords - coord_min_arr) / denom_safe
#     points = points * denom_safe + coord_min
    
#     # Reverse: coords -= original_translation
#     points = points + original_translation
    
#     # Update the geometry points
#     _set_points(geom, kind, points)
    
#     # Handle normals if present
#     normals = _get_normals(geom, kind)
#     if normals is not None and len(normals) > 0:
#         print("Transforming normals...")
        
#         if not keep_aspect_ratio:
#             # Inverse transform for normals: apply the transpose of the original scaling
#             scale_arr = 2.0 / (coord_max - coord_min + 1e-12)
#             S = np.diag(scale_arr.tolist())
#             # For inverse transformation, we need the transpose of the inverse of the original transform
#             # Original: normals_transformed = (self.normals @ inv_transpose)
#             # Inverse: we need to apply the inverse of that operation
#             inv_transpose = np.linalg.inv(S).T
#             # To reverse, we apply the inverse of inv_transpose, which is just S.T
#             normals_transformed = (normals @ S.T)
            
#             # Renormalize
#             norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
#             norms[norms == 0.0] = 1.0
#             normals_transformed = normals_transformed / norms
#             _set_normals(geom, kind, normals_transformed)
#         else:
#             # For uniform scaling, just ensure normals are unit length
#             norms = np.linalg.norm(normals, axis=1, keepdims=True)
#             norms[norms == 0.0] = 1.0
#             normals = normals / norms
#             _set_normals(geom, kind, normals)
    
#     # Save the transformed geometry
#     base_name, ext = os.path.splitext(os.path.basename(input_path))
#     output_path = os.path.join(output_dir, f"{base_name}_gis{ext}")
#     _save_geometry(output_path, geom, kind)
    
#     print(f"Restored (GIS) geometry saved to: {output_path}")
#     return output_path

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) != 3:
#         print("Usage: python move.py <normalized_mesh.ply> <original_point_cloud.ply>")
#         print("  normalized_mesh.ply: The mesh to rescale back to GIS coordinates")
#         print("  original_point_cloud.ply: The original point cloud used to compute normalization parameters")
#         sys.exit(1)
    
#     normalized_mesh_path = sys.argv[1]
#     original_pc_path = sys.argv[2]
    
#     # Verify files exist
#     if not os.path.exists(normalized_mesh_path):
#         print(f"Error: Normalized mesh file not found: {normalized_mesh_path}")
#         sys.exit(1)
    
#     if not os.path.exists(original_pc_path):
#         print(f"Error: Original point cloud file not found: {original_pc_path}")
#         sys.exit(1)
    
#     # Compute normalization parameters from original point cloud
#     normalization_params = compute_normalization_parameters(original_pc_path, keep_aspect_ratio=True)
    
#     # Rescale normalized mesh back to GIS coordinates
#     output_dir = "mesh/siren_mesh"
#     inverse_transform_geometry(normalized_mesh_path, output_dir, normalization_params)
    
#     print("Transformation complete")

import numpy as np
import open3d as o3d
import os

'''This script automatically computes normalization parameters from an original point cloud
   and uses them to rescale a normalized mesh back to GIS coordinates.'''

'''# Explicit uniform scaling (DEFAULT)
    python move.py normalized_mesh.ply original_point_cloud.ply true

    # Non-uniform scaling (different scale per axis)
    python move.py normalized_mesh.ply original_point_cloud.ply false'''

# Helpers to handle both meshes and point clouds
def _load_geometry(input_path):
    # Try mesh first; if no triangles, treat as point cloud
    mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=False)
    if mesh is not None and mesh.has_vertices() and mesh.has_triangles() and len(mesh.triangles) > 0:
        return mesh, "mesh"
    pcd = o3d.io.read_point_cloud(input_path)
    return pcd, "pcd"

def _get_points(geom, kind):
    return np.asarray(geom.vertices) if kind == "mesh" else np.asarray(geom.points)

def _get_normals(geom, kind):
    return np.asarray(geom.vertex_normals) if kind == "mesh" else np.asarray(geom.normals)

def _set_points(geom, kind, points):
    if kind == "mesh":
        geom.vertices = o3d.utility.Vector3dVector(points)
    else:
        geom.points = o3d.utility.Vector3dVector(points)

def _set_normals(geom, kind, normals):
    if kind == "mesh":
        geom.vertex_normals = o3d.utility.Vector3dVector(normals)
    else:
        geom.normals = o3d.utility.Vector3dVector(normals)

def _save_geometry(output_path, geom, kind):
    if kind == "mesh":
        o3d.io.write_triangle_mesh(output_path, geom)
    else:
        o3d.io.write_point_cloud(output_path, geom)

def compute_normalization_parameters(original_point_cloud_path, keep_aspect_ratio=True):
    """
    Compute normalization parameters from original point cloud using the same logic
    as the provided normalization code.
    """
    print("Computing normalization parameters from original point cloud...")
    
    # Load original point cloud
    pcd, _ = _load_geometry(original_point_cloud_path)
    coords = _get_points(pcd, "pcd")
    
    print(f"Original point cloud has {len(coords)} points")
    print(f"Using {'uniform' if keep_aspect_ratio else 'non-uniform'} scaling")
    
    # Center the coordinates (subtract mean)
    original_translation = np.mean(coords, axis=0, keepdims=True)
    coords_centered = coords - original_translation
    
    # Compute bounds
    if keep_aspect_ratio:
        # Use scalar min/max (preserve aspect ratio)
        coord_max = np.amax(coords_centered)
        coord_min = np.amin(coords_centered)
        coord_max_arr = np.array([coord_max, coord_max, coord_max], dtype=np.float64)
        coord_min_arr = np.array([coord_min, coord_min, coord_min], dtype=np.float64)
    else:
        # Axis-wise min/max (non-uniform scaling)
        coord_max_arr = np.amax(coords_centered, axis=0)
        coord_min_arr = np.amin(coords_centered, axis=0)
    
    # Compute scaling parameters
    denom = (coord_max_arr - coord_min_arr)
    denom_safe = denom.copy()
    denom_safe[denom_safe == 0.0] = 1e-8
    
    # Store parameters for inverse transformation
    params = {
        'original_translation': original_translation[0],  # Remove keepdims dimension
        'coord_min': coord_min_arr,
        'coord_max': coord_max_arr,
        'denom_safe': denom_safe,
        'keep_aspect_ratio': keep_aspect_ratio
    }
    
    print("Computed parameters:")
    print(f"  Original translation: {params['original_translation']}")
    print(f"  Coord min: {params['coord_min']}")
    print(f"  Coord max: {params['coord_max']}")
    print(f"  Keep aspect ratio: {keep_aspect_ratio}")
    
    return params

def inverse_transform_geometry(input_path, output_dir, normalization_params):
    """
    Inverse transform normalized geometry back to GIS coordinates using computed parameters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the normalized geometry (mesh or point cloud)
    geom, kind = _load_geometry(input_path)
    points = _get_points(geom, kind)
    
    print(f"Inverse transforming {kind} with {len(points)} points...")
    
    # Extract parameters
    original_translation = normalization_params['original_translation']
    coord_min = normalization_params['coord_min']
    coord_max = normalization_params['coord_max']
    denom_safe = normalization_params['denom_safe']
    keep_aspect_ratio = normalization_params['keep_aspect_ratio']
    
    # Apply inverse transformation (back to GIS coordinates)
    # Reverse: self.coords = self.coords * 2.0
    points = points / 2.0
    
    # Reverse: self.coords = self.coords - 0.5
    points = points + 0.5
    
    # Reverse: self.coords = (coords - coord_min_arr) / denom_safe
    points = points * denom_safe + coord_min
    
    # Reverse: coords -= original_translation
    points = points + original_translation
    
    # Update the geometry points
    _set_points(geom, kind, points)
    
    # Handle normals if present
    normals = _get_normals(geom, kind)
    if normals is not None and len(normals) > 0:
        print("Transforming normals...")
        
        if not keep_aspect_ratio:
            # For non-uniform scaling: we need to properly transform normals back
            # Original forward scaling: scale_arr = 2.0 / (coord_max - coord_min)
            scale_arr = 2.0 / (coord_max - coord_min + 1e-12)
            
            # The original normal transformation was: normals @ np.linalg.inv(S).T
            # So to reverse it, we need to apply: normals @ S.T
            S = np.diag(scale_arr)
            normals_transformed = normals @ S.T
            
            # Renormalize
            norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            normals_transformed = normals_transformed / norms
            _set_normals(geom, kind, normals_transformed)
            print("Applied non-uniform normal transformation")
        else:
            # For uniform scaling, just ensure normals are unit length
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            normals = normals / norms
            _set_normals(geom, kind, normals)
            print("Applied uniform normal normalization")
    
    # Save the transformed geometry
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_dir, f"{base_name}_gis{ext}")
    _save_geometry(output_path, geom, kind)
    
    print(f"Restored (GIS) geometry saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) not in [3, 4]:
        print("Usage: python move.py <normalized_mesh.ply> <original_point_cloud.ply> [keep_aspect_ratio]")
        print("  normalized_mesh.ply: The mesh to rescale back to GIS coordinates")
        print("  original_point_cloud.ply: The original point cloud used to compute normalization parameters")
        print("  keep_aspect_ratio: (optional) 'true' for uniform scaling, 'false' for non-uniform (default: true)")
        sys.exit(1)
    
    normalized_mesh_path = sys.argv[1]
    original_pc_path = sys.argv[2]
    
    # Parse keep_aspect_ratio parameter (default to True)
    keep_aspect_ratio = True
    if len(sys.argv) == 4:
        if sys.argv[3].lower() == 'false':
            keep_aspect_ratio = False
        elif sys.argv[3].lower() == 'true':
            keep_aspect_ratio = True
        else:
            print(f"Warning: Invalid keep_aspect_ratio value '{sys.argv[3]}', using default (true)")
    
    # Verify files exist
    if not os.path.exists(normalized_mesh_path):
        print(f"Error: Normalized mesh file not found: {normalized_mesh_path}")
        sys.exit(1)
    
    if not os.path.exists(original_pc_path):
        print(f"Error: Original point cloud file not found: {original_pc_path}")
        sys.exit(1)
    
    # Compute normalization parameters from original point cloud
    normalization_params = compute_normalization_parameters(original_pc_path, keep_aspect_ratio)
    
    # Rescale normalized mesh back to GIS coordinates
    # Save to the same directory as the input file
    output_dir = os.path.dirname(normalized_mesh_path)
    
    inverse_transform_geometry(normalized_mesh_path, output_dir, normalization_params)
    
    print("Transformation complete!")