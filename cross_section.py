#!/usr/bin/env python3
"""
Generate multiple cross-sections along a line/normal direction.
Two modes:
  - Fixed thickness: Each slice has same thickness, placed at intervals
  - Auto thickness: Slices partition the entire line (thickness = total_length / num_sections)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import utils

# ---------------------------
# Math helpers
# ---------------------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector")
    return v / n

def extract_plane_slice(points, p0, n, thickness):
    """
    Plane: (x - p0)·n = 0
    Keep points with |(x - p0)·n| <= thickness/2
    """
    n = unit(n)
    signed = (points - p0) @ n
    mask = np.abs(signed) <= (0.5 * thickness)
    return points[mask], signed[mask], mask

def project_to_uv(points_slice, p0, u, v):
    d = points_slice - p0
    return np.column_stack([d @ u, d @ v])

def generate_planes_along_line_fixed(p_start, p_end, num_sections=10):
    """
    Fixed mode: Place planes at evenly spaced points along the line.
    Each plane is at the center of its slice.
    """
    p_start = np.array(p_start, dtype=np.float64)
    p_end = np.array(p_end, dtype=np.float64)
    
    # Direction of the line (this becomes the normal of the planes)
    line_dir = p_end - p_start
    line_length = np.linalg.norm(line_dir)
    line_normal = unit(line_dir)
    
    # Generate points along the line (evenly spaced)
    # We want the centers of each slice
    t_values = np.linspace(line_length / (num_sections + 1), 
                           line_length * num_sections / (num_sections + 1), 
                           num_sections)
    
    planes = []
    for i, t in enumerate(t_values):
        origin = p_start + t * line_normal
        normal = line_normal.copy()
        
        planes.append({
            'id': i + 1,
            'origin': origin,
            'normal': normal,
            't': t,
            'position_along_line': t / line_length,  # 0 to 1
            'slice_center': t,
            'slice_start': None,  # Not used in fixed mode
            'slice_end': None     # Not used in fixed mode
        })
    
    return planes, line_length

def generate_planes_along_line_auto(p_start, p_end, num_sections=10):
    """
    Auto mode: Partition the entire line into equal-thickness slices.
    Returns planes at the CENTER of each slice.
    """
    p_start = np.array(p_start, dtype=np.float64)
    p_end = np.array(p_end, dtype=np.float64)
    
    # Direction of the line
    line_dir = p_end - p_start
    line_length = np.linalg.norm(line_dir)
    line_normal = unit(line_dir)
    
    # Calculate slice thickness
    slice_thickness = line_length / num_sections
    
    # Calculate slice boundaries
    slice_starts = np.linspace(0, line_length - slice_thickness, num_sections)
    slice_ends = slice_starts + slice_thickness
    
    # Calculate slice centers
    slice_centers = (slice_starts + slice_ends) / 2
    
    planes = []
    for i in range(num_sections):
        origin = p_start + slice_centers[i] * line_normal
        normal = line_normal.copy()
        
        planes.append({
            'id': i + 1,
            'origin': origin,
            'normal': normal,
            't': slice_centers[i],
            'position_along_line': slice_centers[i] / line_length,
            'slice_center': slice_centers[i],
            'slice_start': slice_starts[i],
            'slice_end': slice_ends[i],
            'slice_thickness': slice_thickness
        })
    
    return planes, line_length, slice_thickness

def save_section_files(sec_dir, plane_dir, base_name, idx, slice_pts, uv, plane_info, thickness, mode, normals=None):
    """
    Save all files for one cross-section.
    """
    # Create directories if they don't exist
    sec_dir.mkdir(parents=True, exist_ok=True)
    plane_dir.mkdir(parents=True, exist_ok=True)
    
    # Filenames
    sec_filename = f"{base_name}_{idx:03d}.ply"
    plane_filename = f"{base_name}_{idx:03d}.csv"
    
    sec_path = sec_dir / sec_filename
    plane_path = plane_dir / plane_filename
    
    # Save slice points as PLY (include normals if provided)
    save_points_as_ply(slice_pts, sec_path, normals=normals)
    
    # Save plane information
    if mode == 'auto':
        # Include slice boundaries for auto mode
        plane_data = np.array([[
            plane_info['origin'][0], plane_info['origin'][1], plane_info['origin'][2],
            plane_info['normal'][0], plane_info['normal'][1], plane_info['normal'][2],
            plane_info['slice_thickness'],
            plane_info['position_along_line'],
            plane_info['slice_start'],
            plane_info['slice_end']
        ]])
        
        np.savetxt(plane_path, plane_data, delimiter=",",
                   header="p0x,p0y,p0z,nx,ny,nz,thickness,position_along_line,slice_start,slice_end",
                   comments="")
    else:
        # Fixed mode
        plane_data = np.array([[
            plane_info['origin'][0], plane_info['origin'][1], plane_info['origin'][2],
            plane_info['normal'][0], plane_info['normal'][1], plane_info['normal'][2],
            thickness,
            plane_info['position_along_line']
        ]])
        
        np.savetxt(plane_path, plane_data, delimiter=",",
                   header="p0x,p0y,p0z,nx,ny,nz,thickness,position_along_line",
                   comments="")
    
    # Save UV coordinates
    uv_path = sec_dir / f"{base_name}_{idx:03d}_uv.csv"
    np.savetxt(uv_path, uv, delimiter=",", header="u,v", comments="")
    
    return sec_path, plane_path, uv_path

def save_points_as_ply(points, filename, normals=None):
    """
    Save 3D points as a simple PLY file (ASCII format).
    If `normals` is provided (Nx3), include nx,ny,nz properties and write them.
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")
        if normals is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            # Ensure shapes match
            normals = np.asarray(normals)
            if len(normals) != len(points):
                raise ValueError("Points and normals length mismatch when writing PLY.")
            for p, n in zip(points, normals):
                f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")

# ---------------------------
# Main function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple cross-sections along a line. "
                   "Use --thickness auto to partition the entire line into equal slices."
    )
    parser.add_argument("--ply", required=True,
                        help="Input .ply point cloud")
    parser.add_argument("--start", nargs=3, type=float, required=True,
                        help="Start point of line: x y z")
    parser.add_argument("--end", nargs=3, type=float, required=True,
                        help="End point of line: x y z")
    parser.add_argument("--num_sections", type=int, default=10,
                        help="Number of cross-sections to generate")
    parser.add_argument("--thickness", required=True,
                        help="Slice thickness. Use 'auto' for automatic partitioning")
    parser.add_argument("--min_points", type=int, default=30,
                        help="Skip section if fewer points")
    parser.add_argument("--out_dir", default="./data",
                        help="Base output directory")
    parser.add_argument("--base_name", 
                        help="Base name for output files (default: from input PLY filename)")
    
    args = parser.parse_args()
    
    # Parse thickness
    if args.thickness.lower() == 'auto':
        thickness_mode = 'auto'
        thickness_value = None
    else:
        thickness_mode = 'fixed'
        thickness_value = float(args.thickness)
    
    # Setup output directories
    base_dir = Path(args.out_dir)
    sec_dir = base_dir / "cross-section"
    plane_dir = base_dir / "plane-section"
    
    # Determine base name
    if args.base_name:
        base_name = args.base_name
    else:
        base_name = Path(args.ply).stem
    
    print(f"[INFO] Loading point cloud: {args.ply}")
    pts, normals = utils.load_ply_xyz_normals(args.ply)
    print(f"[INFO] Loaded {len(pts):,} points")
    
    # Generate planes along the line
    if thickness_mode == 'auto':
        print(f"[INFO] AUTO MODE: Partitioning line into {args.num_sections} equal slices")
        planes, line_length, slice_thickness = generate_planes_along_line_auto(
            args.start, args.end, args.num_sections
        )
        print(f"[INFO] Line length: {line_length:.3f}")
        print(f"[INFO] Each slice thickness: {slice_thickness:.3f}")
        print(f"[INFO] Slices will exactly cover the line from start to end")
    else:
        print(f"[INFO] FIXED MODE: {args.num_sections} slices with thickness {thickness_value}")
        planes, line_length = generate_planes_along_line_fixed(
            args.start, args.end, args.num_sections
        )
        print(f"[INFO] Line length: {line_length:.3f}")
    
    saved_count = 0
    skipped_count = 0
    
    for i, plane in enumerate(planes):
        print(f"\n[PROCESS] Section {plane['id']}/{args.num_sections}")
        
        # Use appropriate thickness
        if thickness_mode == 'auto':
            current_thickness = plane['slice_thickness']
            print(f"           Thickness: {current_thickness:.3f} (auto)")
            print(f"           Covers: t=[{plane['slice_start']:.3f}, {plane['slice_end']:.3f}]")
        else:
            current_thickness = thickness_value
            print(f"           Thickness: {current_thickness:.3f} (fixed)")
        
        # Extract points near this plane
        slice_pts, signed, mask = extract_plane_slice(
            pts,
            plane['origin'],
            plane['normal'],
            current_thickness
        )

        # If normals were present in the input, keep the corresponding normals
        slice_normals = normals[mask] if (normals is not None) else None
        
        if len(slice_pts) < args.min_points:
            print(f"  [SKIP] Only {len(slice_pts)} points (min={args.min_points})")
            skipped_count += 1
            continue
        
        # Project to UV coordinates
        u, v = utils.orthonormal_basis_from_normal(plane['normal'])
        uv = project_to_uv(slice_pts, plane['origin'], u, v)
        
        # Save files
        sec_path, plane_path, uv_path = save_section_files(
            sec_dir, plane_dir, base_name, plane['id'],
            slice_pts, uv, plane, current_thickness, thickness_mode, normals=slice_normals
        )
        
        print(f"  [OK] Saved {len(slice_pts)} points")
        print(f"       Section: {sec_path}")
        print(f"       Plane:   {plane_path}")
        print(f"       UV:      {uv_path}")
        if thickness_mode == 'auto':
            print(f"       Position: {plane['position_along_line']:.3f} (center)")
        
        saved_count += 1
    
    print(f"\n[DONE] Complete!")
    print(f"       Mode: {thickness_mode}")
    print(f"       Total sections: {args.num_sections}")
    print(f"       Saved: {saved_count}")
    print(f"       Skipped: {skipped_count}")
    print(f"       Files in: {base_dir.resolve()}")

if __name__ == "__main__":
    main()
