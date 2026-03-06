import open3d as o3d
import numpy as np
import os

def ply_to_xyz(ply_filename):
    """
    Convert PLY file to XYZ format in the same directory, preserving normals if they exist
    
    Args:
        ply_filename (str): Path to input PLY file
    
    Returns:
        str: Path to the created XYZ file
    
    Raises:
        ValueError: If the file cannot be read or has no points
        FileNotFoundError: If the input file doesn't exist
    """
    
    # Check if input file exists
    if not os.path.exists(ply_filename):
        raise FileNotFoundError(f"Input file not found: {ply_filename}")
    
    # Generate output file path in the same directory
    directory = os.path.dirname(ply_filename)
    base_name = os.path.splitext(os.path.basename(ply_filename))[0]
    
    if directory:  # If there's a directory path
        xyz_file_path = os.path.join(directory, base_name + ".xyz")
    else:  # If file is in current directory
        xyz_file_path = base_name + ".xyz"
    
    # Read the PLY file
    print(f"Reading PLY file: {ply_filename}")
    pcd = o3d.io.read_point_cloud(ply_filename)
    
    # Check if point cloud was loaded successfully
    if not pcd.has_points():
        raise ValueError("Failed to load point cloud or no points found in the file")
    
    # Get points
    points = np.asarray(pcd.points)
    
    # Check if normals exist
    has_normals = pcd.has_normals()
    print(f"Point cloud has {len(points)} points")
    print(f"Normals present: {has_normals}")
    
    # Write to XYZ format
    with open(xyz_file_path, 'w') as f:
        if has_normals:
            normals = np.asarray(pcd.normals)
            for point, normal in zip(points, normals):
                # Format: x y1z nx ny nz
                line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                f.write(line)
        else:
            for point in points:
                # Format: x y z
                line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"
                f.write(line)
    
    print(f"Successfully converted to XYZ format: {xyz_file_path}")
    print(f"Format: {'XYZ with normals' if has_normals else 'XYZ without normals'}")
    
    return xyz_file_path

def batch_convert_ply_to_xyz(input_folder):
    """
    Convert all PLY files in a folder to XYZ format in the same directory
    
    Args:
        input_folder (str): Path to folder containing PLY files
    """
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Find all PLY files in the input folder
    ply_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.ply')]
    
    print(f"Found {len(ply_files)} PLY files to convert")
    
    successful_conversions = 0
    for ply_file in ply_files:
        input_path = os.path.join(input_folder, ply_file)
        
        try:
            ply_to_xyz(input_path)
            successful_conversions += 1
        except Exception as e:
            print(f"Error converting {ply_file}: {e}")
    
    print(f"Successfully converted {successful_conversions}/{len(ply_files)} files")

def main():
    """
    Main function for command line usage
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PLY files to XYZ format')
    parser.add_argument('input', help='Input PLY file or folder')
    parser.add_argument('--batch', action='store_true', help='Batch convert all PLY files in folder')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch convert all PLY files in folder
            batch_convert_ply_to_xyz(args.input)
        else:
            # Convert single file
            result_file = ply_to_xyz(args.input)
            
            # Optional: Verify the conversion by reading the first few lines
            print("\nVerifying converted file (first 3 lines):")
            with open(result_file, 'r') as f:
                for i in range(3):
                    line = f.readline().strip()
                    if line:
                        print(f"Line {i+1}: {line}")
                    else:
                        break
                        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()