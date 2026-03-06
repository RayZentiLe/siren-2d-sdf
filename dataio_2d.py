import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors  # For signed labeling


def transform_plane_to_xy(points_3d, normals_3d, plane_origin, plane_normal):
    """
    Transform points and normals so that the given plane becomes the XY plane.
    
    Args:
        points_3d: [N, 3] points exactly on the plane (from cross-section)
        normals_3d: [N, 3] normals at these points
        plane_origin: [3] point on the plane
        plane_normal: [3] normal vector of the plane
    
    Returns:
        points_2d: [N, 2] points in XY plane coordinates
        normals_2d: [N, 2] normals projected onto XY plane
        transform_info: dict with rotation matrix and origin
    """
    # 1. Shift to plane origin
    points_shifted = points_3d - plane_origin
    
    # 2. Build rotation matrix to align plane_normal with [0, 0, 1]
    z_axis = plane_normal / np.linalg.norm(plane_normal)
    
    # Find arbitrary perpendicular vectors for x and y axes
    if abs(z_axis[0]) < 0.9:
        x_axis = np.cross(z_axis, [1, 0, 0])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Rotation matrix (points @ R rotates them to align with axes)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    
    # 3. Apply rotation to points
    points_rotated = points_shifted @ R
    
    # 4. Points should now have z ≈ 0 (they were exactly on the plane)
    points_2d = points_rotated[:, :2]  # Drop z, keep x,y
    
    # 5. Transform normals (vectors need same rotation)
    normals_rotated = normals_3d @ R
    
    # 6. Project normals onto XY plane (drop z component) and renormalize
    normals_2d = normals_rotated[:, :2]
    norm_norms = np.linalg.norm(normals_2d, axis=1, keepdims=True)
    normals_2d = normals_2d / (norm_norms + 1e-8)
    
    transform_info = {
        'rotation_matrix': R,
        'plane_origin': plane_origin,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }
    
    return points_2d, normals_2d, transform_info


def normalize_to_unit_square(points_2d):
    """
    Normalize 2D points to [-1, 1] range.
    """
    # Center points
    mean = np.mean(points_2d, axis=0, keepdims=True)
    points_centered = points_2d - mean
    
    # Find scale to fit in [-1, 1]
    max_abs = np.amax(np.abs(points_centered))
    if max_abs < 1e-6:
        max_abs = 1.0
    
    scale = max_abs
    points_norm = points_centered / scale
    
    norm_params = {
        'mean': mean.flatten(),
        'scale': scale
    }
    
    return points_norm, norm_params


def load_point_cloud_with_normals(file_path):
    """Load .xyz file with normals (x y z nx ny nz)"""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:6]
    return points, normals


def load_plane_csv(plane_csv):
    """Load plane parameters from CSV file."""
    plane_params = np.loadtxt(plane_csv, delimiter=",", skiprows=1)
    if plane_params.ndim == 1:
        plane_params = plane_params.reshape(1, -1)
    
    plane_origin = plane_params[0, :3]
    plane_normal = plane_params[0, 3:6]
    
    return plane_origin, plane_normal


def write_sdf_summary_2d(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    """
    Write summary for 2D SDF training.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    
    # Get data and ensure proper shapes
    pred_sdf = model_output['model_out'].detach().cpu().numpy()  # Shape: [batch_size*2, 1]
    gt_sdf = gt['sdf'].detach().cpu().numpy()  # Shape: [batch_size*2, 1]
    coords = model_input['coords'].detach().cpu().numpy()  # Shape: [batch_size*2, 2]
    
    # Flatten everything to 1D for indexing
    pred_sdf_flat = pred_sdf.flatten()
    gt_sdf_flat = gt_sdf.flatten()
    
    # Create mask for on-surface points (where gt_sdf != -1 in original mode, 
    # but in signed mode we need a different approach)
    # For summary, we'll just use a small tolerance around 0
    on_surface_mask = np.abs(gt_sdf_flat) < 1e-6
    
    # Filter points
    on_coords = coords[on_surface_mask]
    on_pred = pred_sdf_flat[on_surface_mask]
    
    print(f"Summary - Total points: {len(coords)}, On-surface: {len(on_coords)}")
    
    # Create 2D grid for visualization
    resolution = 256
    u_grid = np.linspace(-1, 1, resolution)
    v_grid = np.linspace(-1, 1, resolution)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_coords = np.stack([uu.ravel(), vv.ravel()], axis=1)
    
    # Evaluate model on grid
    grid_tensor = torch.from_numpy(grid_coords).float().cuda()
    grid_input = {'coords': grid_tensor}
    
    with torch.no_grad():
        grid_output = model(grid_input)
        grid_sdf = grid_output['model_out'].cpu().numpy().reshape(resolution, resolution)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: On-surface training points colored by predicted SDF
    ax = axes[0]
    if len(on_coords) > 0:
        scatter = ax.scatter(on_coords[:, 0], on_coords[:, 1], c=on_pred, 
                            cmap='RdBu_r', s=5, alpha=0.8, vmin=-0.5, vmax=0.5)
        ax.set_title(f'On-Surface Points (n={len(on_coords)})')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(scatter, cax=cax)
    else:
        ax.set_title('On-Surface Points (none)')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predicted SDF on grid
    ax = axes[1]
    im = ax.imshow(grid_sdf.T, origin='lower', extent=[-1, 1, -1, 1],
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title('Predicted SDF on Grid')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Plot 3: Zero level set with original points
    ax = axes[2]
    ax.contour(grid_sdf.T, levels=[0], colors='black', linewidths=1, extent=[-1, 1, -1, 1])
    if len(on_coords) > 0:
        ax.scatter(on_coords[:, 0], on_coords[:, 1], c='red', s=5, alpha=0.5, label='Contour Points')
        ax.legend()
    ax.set_title('Zero Level Set')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to tensorboard
    writer.add_figure(prefix + 'sdf_2d', fig, global_step=total_steps)
    plt.close(fig)
    
    # Also log losses
    if 'loss' in model_output:
        writer.add_scalar(prefix + 'total_loss', model_output['loss'].item(), total_steps)
    
    for key in ['sdf', 'inter', 'normal_constraint', 'grad_constraint']:
        if key in model_output:
            writer.add_scalar(prefix + key, model_output[key].item(), total_steps)


# ==================== VISUALIZATION FUNCTIONS ====================

def visualize_training_batch_original(coords, gt_sdf, epoch, save_dir, batch_idx=0):
    """
    Original visualization: off-surface (red), on-surface (green)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gt_sdf_flat = gt_sdf.flatten()
    on_surface = gt_sdf_flat != -1
    off_surface = gt_sdf_flat == -1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if off_surface.any():
        ax.scatter(coords[off_surface, 0], coords[off_surface, 1], 
                  c='red', s=2, alpha=0.3, label=f'Off-surface ({off_surface.sum()})')
    
    if on_surface.any():
        ax.scatter(coords[on_surface, 0], coords[on_surface, 1], 
                  c='green', s=5, alpha=0.8, label=f'On-surface ({on_surface.sum()})')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Training Batch (Original) - Epoch {epoch} (Batch {batch_idx})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    stats_text = f'Total: {len(coords)}\nOn: {on_surface.sum()}\nOff: {off_surface.sum()}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    filename = f'training_batch_epoch_{epoch:04d}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved batch visualization to: {save_path}")


def visualize_training_batch_signed(coords, gt_sdf, epoch, save_dir, batch_idx=0):
    """
    Signed visualization: inside (blue), on-surface (green), outside (red)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gt_sdf_flat = gt_sdf.flatten()
    
    on_surface = np.abs(gt_sdf_flat) < 1e-6
    outside = gt_sdf_flat > 0.5
    inside = gt_sdf_flat < -0.5
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if outside.any():
        ax.scatter(coords[outside, 0], coords[outside, 1], 
                  c='red', s=2, alpha=0.3, label=f'Outside (+1) ({outside.sum()})')
    
    if inside.any():
        ax.scatter(coords[inside, 0], coords[inside, 1], 
                  c='blue', s=2, alpha=0.3, label=f'Inside (-1) ({inside.sum()})')
    
    if on_surface.any():
        ax.scatter(coords[on_surface, 0], coords[on_surface, 1], 
                  c='green', s=5, alpha=0.8, label=f'On-surface (0) ({on_surface.sum()})')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Training Batch (Signed) - Epoch {epoch} (Batch {batch_idx})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    stats_text = f'Total: {len(coords)}\nOn: {on_surface.sum()}\nOut: {outside.sum()}\nIn: {inside.sum()}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    filename = f'training_batch_epoch_{epoch:04d}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved signed batch visualization to: {save_path}")


# ==================== DATASET CLASSES ====================

class CrossSection2D(Dataset):
    """Dataset for 2D cross-sections with plane transformation"""
    
    def __init__(self, point_cloud_path, plane_origin, plane_normal, on_surface_points, 
                 use_signed_labels=False, k_neighbors=16,
                 vis_sampled_point=False, vis_frequency=100, vis_dir=None):
        """
        Args:
            point_cloud_path: Path to .xyz file (x y z nx ny nz) - points exactly on cross-section
            plane_origin: [3] point on the plane
            plane_normal: [3] normal vector of the plane
            on_surface_points: Number of on-surface points per batch
            use_signed_labels: If True, off-surface points get +1/-1 labels; if False, all off-surface = -1
            k_neighbors: Number of neighbors for sign determination (only used if use_signed_labels=True)
            vis_sampled_point: Enable batch visualization
            vis_frequency: Save visualization every N epochs
            vis_dir: Directory to save visualizations
        """
        super().__init__()
        
        print(f"Loading cross-section points: {point_cloud_path}")
        self.points_3d, self.normals_3d = load_point_cloud_with_normals(point_cloud_path)
        print(f"  Loaded {len(self.points_3d)} points on cross-section")

        # 1. Transform plane to XY
        points_2d, normals_2d, self.transform_info = transform_plane_to_xy(
            self.points_3d, self.normals_3d,
            plane_origin, plane_normal
        )
        self.points_2d = points_2d
        self.normals_2d = normals_2d
        
        # 2. Normalize to [-1, 1]² for SIREN
        self.points_norm, self.norm_params = normalize_to_unit_square(points_2d)
        
        self.on_surface_points = on_surface_points
        self.use_signed_labels = use_signed_labels
        self.k_neighbors = k_neighbors
        
        # Setup for signed labeling if needed
        if self.use_signed_labels:
            print(f"  Signed labels ENABLED: fitting kNN (k={k_neighbors}) for sign determination...")
            self.knn_model = NearestNeighbors(n_neighbors=k_neighbors)
            self.knn_model.fit(self.points_norm)
            self.on_normals = self.normals_2d
        else:
            print(f"  Signed labels DISABLED: using original -1 for all off-surface points")
        
        # Visualization setup
        self.vis_sampled_point = vis_sampled_point
        self.vis_frequency = vis_frequency
        self.vis_dir = vis_dir
        self.epoch_counter = 0
        
        print(f"\n2D Dataset ready:")
        print(f"  Points in plane coordinates: range=[{points_2d.min(axis=0)}, {points_2d.max(axis=0)}]")
        print(f"  Normalized to [-1, 1]: u=[{self.points_norm[:,0].min():.3f}, {self.points_norm[:,0].max():.3f}], "
              f"v=[{self.points_norm[:,1].min():.3f}, {self.points_norm[:,1].max():.3f}]")
        print(f"  Label mode: {'SIGNED (+1/-1)' if use_signed_labels else 'ORIGINAL (-1 only)'}")
        
        if self.vis_sampled_point and self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
            mode_str = 'signed' if use_signed_labels else 'original'
            print(f"  Batch visualizations ({mode_str} mode) saved every {self.vis_frequency} epochs to: {self.vis_dir}")

    def __len__(self):
        return max(1, self.points_norm.shape[0] // self.on_surface_points)
    
    def _label_off_surface_points_signed(self, off_coords_norm):
        """
        Assign signed values (+1 outside, -1 inside) to off-surface points
        using kNN on normalized coordinates.
        """
        N_off = off_coords_norm.shape[0]
        # Find k nearest neighbors among on-surface points
        distances, indices = self.knn_model.kneighbors(off_coords_norm)
        
        # Get the normals of those neighbors
        neighbor_normals = self.on_normals[indices]  # Shape: [N_off, k, 2]
        
        # Vector from off-surface point to its neighbors
        vectors = off_coords_norm[:, np.newaxis, :] - self.points_norm[indices]  # Shape: [N_off, k, 2]
        
        # Project vectors onto neighbor normals (dot product)
        proj = np.sum(vectors * neighbor_normals, axis=2)  # Shape: [N_off, k]
        
        # Average projection for each off-surface point
        mean_proj = np.mean(proj, axis=1)  # Shape: [N_off]
        
        # Assign sign: +1 where mean projection > 0 (outside), -1 otherwise (inside)
        signs = np.where(mean_proj > 0, 1.0, -1.0)
        return signs
    
    def __getitem__(self, idx):
        n_points = self.points_norm.shape[0]
        
        # On-surface points
        if n_points >= self.on_surface_points:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=False)
        else:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=True)
        
        on_coords = self.points_norm[on_idx]
        on_normals = self.normals_2d[on_idx]
        
        # Off-surface points
        off_coords = np.random.uniform(-1, 1, size=(self.on_surface_points, 2))
        off_normals = np.ones((self.on_surface_points, 2)) * -1  # Normals not supervised
        
        # Determine SDF values based on mode
        sdf_on = np.zeros(self.on_surface_points)
        
        if self.use_signed_labels:
            # Signed mode: off-surface points get +1 or -1
            off_signs = self._label_off_surface_points_signed(off_coords)
            sdf_off = off_signs
        else:
            # Original mode: all off-surface points = -1
            sdf_off = np.ones(self.on_surface_points) * -1
        
        sdf = np.concatenate([sdf_on, sdf_off])
        coords = np.vstack([on_coords, off_coords])
        normals = np.vstack([on_normals, off_normals])
        
        # Visualization (use appropriate function based on mode)
        if (self.vis_sampled_point and self.vis_dir and idx == 0 and 
            self.epoch_counter % self.vis_frequency == 0):
            
            if self.use_signed_labels:
                visualize_training_batch_signed(
                    coords=coords,
                    gt_sdf=sdf.reshape(-1, 1),
                    epoch=self.epoch_counter,
                    save_dir=self.vis_dir,
                    batch_idx=idx
                )
            else:
                visualize_training_batch_original(
                    coords=coords,
                    gt_sdf=sdf.reshape(-1, 1),
                    epoch=self.epoch_counter,
                    save_dir=self.vis_dir,
                    batch_idx=idx
                )
        
        if idx == 0:
            self.epoch_counter += 1
        
        return {
            'coords': torch.from_numpy(coords).float()
        }, {
            'sdf': torch.from_numpy(sdf).float().unsqueeze(-1),
            'normals': torch.from_numpy(normals).float()
        }
    
    def transform_back(self, points_2d_norm):
        """
        Transform normalized points back to original 3D space.
        Useful for visualization.
        """
        # Denormalize from [-1, 1] to original plane coordinates
        points_2d = points_2d_norm * self.norm_params['scale'] + self.norm_params['mean']
        
        # Add z=0 (points are on the plane)
        points_3d_plane = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])
        
        # Rotate back to original orientation
        R_inv = np.linalg.inv(self.transform_info['rotation_matrix'])
        points_rotated = points_3d_plane @ R_inv
        
        # Shift back
        points_3d = points_rotated + self.transform_info['plane_origin']
        
        return points_3d


class PointCloud2D(Dataset):
    """Dataset for 2D point clouds already in XY plane (x, y, nx, ny format)"""
    
    def __init__(self, pointcloud_path, on_surface_points, 
                 use_signed_labels=False, k_neighbors=16,
                 vis_sampled_point=False, vis_frequency=100, vis_dir=None):
        """
        Args:
            pointcloud_path: Path to .xyz file (x y nx ny)
            on_surface_points: Number of on-surface points per batch
            use_signed_labels: If True, off-surface points get +1/-1 labels
            k_neighbors: Number of neighbors for sign determination
            vis_sampled_point: Enable batch visualization
            vis_frequency: Save visualization every N epochs
            vis_dir: Directory to save visualizations
        """
        super().__init__()
        
        data = np.genfromtxt(pointcloud_path)
        self.coords = data[:, :2]
        self.normals = data[:, 2:4]
        
        # Normalize to [-1, 1]
        self.coords -= np.mean(self.coords, axis=0, keepdims=True)
        coord_max = np.amax(np.abs(self.coords))
        if coord_max < 1e-6:
            coord_max = 1.0
        self.coords = self.coords / coord_max
        
        self.on_surface_points = on_surface_points
        self.use_signed_labels = use_signed_labels
        self.k_neighbors = k_neighbors
        
        # Setup for signed labeling if needed
        if self.use_signed_labels:
            print(f"  Signed labels ENABLED: fitting kNN (k={k_neighbors}) for sign determination...")
            self.knn_model = NearestNeighbors(n_neighbors=k_neighbors)
            self.knn_model.fit(self.coords)
            self.on_normals = self.normals
        else:
            print(f"  Signed labels DISABLED: using original -1 for all off-surface points")
        
        # Visualization setup
        self.vis_sampled_point = vis_sampled_point
        self.vis_frequency = vis_frequency
        self.vis_dir = vis_dir
        self.epoch_counter = 0
        
        print(f"Loaded 2D point cloud: {len(self.coords)} points")
        print(f"  Label mode: {'SIGNED (+1/-1)' if use_signed_labels else 'ORIGINAL (-1 only)'}")
        
        if self.vis_sampled_point and self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
            mode_str = 'signed' if use_signed_labels else 'original'
            print(f"  Batch visualizations ({mode_str} mode) saved every {self.vis_frequency} epochs to: {self.vis_dir}")
    
    def __len__(self):
        return max(1, self.coords.shape[0] // self.on_surface_points)
    
    def _label_off_surface_points_signed(self, off_coords_norm):
        """
        Assign signed values (+1 outside, -1 inside) to off-surface points.
        """
        N_off = off_coords_norm.shape[0]
        distances, indices = self.knn_model.kneighbors(off_coords_norm)
        
        neighbor_normals = self.on_normals[indices]  # [N_off, k, 2]
        vectors = off_coords_norm[:, np.newaxis, :] - self.coords[indices]  # [N_off, k, 2]
        
        proj = np.sum(vectors * neighbor_normals, axis=2)  # [N_off, k]
        mean_proj = np.mean(proj, axis=1)
        
        signs = np.where(mean_proj > 0, 1.0, -1.0)
        return signs
    
    def __getitem__(self, idx):
        n_points = self.coords.shape[0]
        
        # On-surface points
        if n_points >= self.on_surface_points:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=False)
        else:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=True)
        
        on_coords = self.coords[on_idx]
        on_normals = self.normals[on_idx]
        
        # Off-surface points
        off_coords = np.random.uniform(-1, 1, size=(self.on_surface_points, 2))
        off_normals = np.ones((self.on_surface_points, 2)) * -1
        
        # Determine SDF values based on mode
        sdf_on = np.zeros(self.on_surface_points)
        
        if self.use_signed_labels:
            off_signs = self._label_off_surface_points_signed(off_coords)
            sdf_off = off_signs
        else:
            sdf_off = np.ones(self.on_surface_points) * -1
        
        sdf = np.concatenate([sdf_on, sdf_off])
        coords = np.vstack([on_coords, off_coords])
        normals = np.vstack([on_normals, off_normals])
        
        # Visualization
        if (self.vis_sampled_point and self.vis_dir and idx == 0 and 
            self.epoch_counter % self.vis_frequency == 0):
            
            if self.use_signed_labels:
                visualize_training_batch_signed(
                    coords=coords,
                    gt_sdf=sdf.reshape(-1, 1),
                    epoch=self.epoch_counter,
                    save_dir=self.vis_dir,
                    batch_idx=idx
                )
            else:
                visualize_training_batch_original(
                    coords=coords,
                    gt_sdf=sdf.reshape(-1, 1),
                    epoch=self.epoch_counter,
                    save_dir=self.vis_dir,
                    batch_idx=idx
                )
        
        if idx == 0:
            self.epoch_counter += 1
        
        return {'coords': torch.from_numpy(coords).float()}, {
            'sdf': torch.from_numpy(sdf).float().unsqueeze(-1),
            'normals': torch.from_numpy(normals).float()
        }
