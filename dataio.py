import csv
import glob
import math
import os

# Fix for headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from sklearn.neighbors import NearestNeighbors  # For signed labeling


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


# ==================== 2D SDF HELPER FUNCTIONS ====================

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
    # Get data and ensure proper shapes
    pred_sdf = model_output['model_out'].detach().cpu().numpy()  # Shape: [batch_size*2, 1]
    gt_sdf = gt['sdf'].detach().cpu().numpy()  # Shape: [batch_size*2, 1]
    coords = model_input['coords'].detach().cpu().numpy()  # Shape: [batch_size*2, 2]
    
    # Flatten everything to 1D for indexing
    pred_sdf_flat = pred_sdf.flatten()
    gt_sdf_flat = gt_sdf.flatten()
    
    # Create mask for on-surface points (using small tolerance around 0)
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


def visualize_training_batch_real_sdf(coords, gt_sdf, distances, epoch, save_dir, batch_idx=0):
    """
    Visualization for REAL SDF values with color mapping.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gt_sdf_flat = gt_sdf.flatten()
    n_points = len(coords)
    n_on = n_points // 2
    
    # Separate points
    on_coords = coords[:n_on]
    off_coords = coords[n_on:]
    off_sdf = gt_sdf_flat[n_on:]
    off_dist = distances if distances is not None else np.abs(off_sdf)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: On-surface points (green)
    ax = axes[0]
    ax.scatter(on_coords[:, 0], on_coords[:, 1], 
               c='green', s=5, alpha=0.8, label=f'On-surface ({n_on})')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'On-Surface Points (Epoch {epoch})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Off-surface points colored by signed distance
    ax = axes[1]
    scatter = ax.scatter(off_coords[:, 0], off_coords[:, 1], 
                        c=off_sdf, cmap='RdBu_r', s=5, alpha=0.8,
                        vmin=-0.5, vmax=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title('Off-Surface Points (colored by SDF)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Signed Distance')
    
    # Plot 3: Distance histogram
    ax = axes[2]
    ax.hist(off_dist, bins=50, alpha=0.7, color='blue')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'Distance Distribution\nmean={off_dist.mean():.4f}, std={off_dist.std():.4f}')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'REAL SDF - Epoch {epoch} (Batch {batch_idx})')
    plt.tight_layout()
    
    filename = f'real_sdf_epoch_{epoch:04d}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also print stats
    print(f"  REAL SDF stats - mean={off_sdf.mean():.4f}, std={off_sdf.std():.4f}, "
          f"range=[{off_sdf.min():.4f}, {off_sdf.max():.4f}]")

# ==================== REAL SDF COMPUTATION FUNCTIONS ====================

def compute_real_sdf_values(on_coords, off_coords, on_normals, 
                            use_signed_labels, knn_model, on_points_norm):
    """
    Compute REAL SDF values using the FULL dataset for kNN queries.
    
    Args:
        on_coords: [N, D] on-surface points (batch subset)
        off_coords: [M, D] off-surface points
        on_normals: [total_N, D] FULL dataset normals
        use_signed_labels: bool
        knn_model: kNN trained on FULL dataset
        on_points_norm: [total_N, D] FULL dataset points
    
    Returns:
        sdf: [N+M] array
        sdf_on: [N] array (zeros)
        sdf_off: [M] array
        distances: [M] array of raw distances
    """
    # On-surface points always 0
    sdf_on = np.zeros(on_coords.shape[0])
    
    if use_signed_labels:
        # Find nearest neighbors in FULL dataset for each off-surface point
        distances, indices = knn_model.kneighbors(off_coords, n_neighbors=1)
        
        # Get the nearest points and their normals from FULL dataset
        nearest_points = on_points_norm[indices[:, 0]]  # [M, D]
        nearest_normals = on_normals[indices[:, 0]]     # [M, D]
        
        # Raw Euclidean distances
        raw_distances = distances.flatten()  # [M]
        
        # Vector from off-surface point to nearest on-surface point
        vectors = off_coords - nearest_points  # [M, D]
        
        # Project onto normal to determine sign
        proj = np.sum(vectors * nearest_normals, axis=1)  # [M]
        
        # Sign: +1 where projection > 0 (outside), -1 otherwise (inside)
        signs = np.where(proj > 0, 1.0, -1.0)
        
        # Signed distance
        sdf_off = raw_distances * signs
    else:
        sdf_off = np.ones(off_coords.shape[0]) * -1
        raw_distances = None
    
    sdf = np.concatenate([sdf_on, sdf_off])
    return sdf, sdf_on, sdf_off, raw_distances


def _compute_real_signed_distances(off_coords, on_points_norm, on_normals, knn_model):
    """
    Compute REAL signed distances for off-surface points. (Only signed, no distances, all off surface will be assigned to 1 or -1)
    
    For each off-surface point:
        1. Find nearest on-surface point
        2. Compute Euclidean distance
        3. Determine sign based on normal alignment
        4. Return signed distance
    
    Args:
        off_coords: [M, D] off-surface points
        on_points_norm: [N, D] normalized on-surface points
        on_normals: [N, D] normals of on-surface points
        knn_model: Fitted NearestNeighbors model
    
    Returns:
        signed_distances: [M] array of signed distances (real values)
        raw_distances: [M] array of raw Euclidean distances
    """
    # Find nearest neighbor for each off-surface point
    distances, indices = knn_model.kneighbors(off_coords, n_neighbors=1)
    
    # Get the nearest on-surface points and their normals
    nearest_points = on_points_norm[indices[:, 0]]  # [M, D]
    nearest_normals = on_normals[indices[:, 0]]     # [M, D]
    
    # Raw Euclidean distances (flatten from [M,1] to [M])
    raw_distances = distances.flatten()
    
    # Vector from off-surface point to nearest on-surface point
    # (note: direction matters for sign)
    vectors = off_coords - nearest_points  # [M, D]
    
    # Project onto normal to determine sign
    # If vector points in same direction as normal -> outside (+)
    # If opposite direction -> inside (-)
    proj = np.sum(vectors * nearest_normals, axis=1)  # [M]
    
    # Sign: +1 where projection > 0 (outside), -1 otherwise (inside)
    signs = np.where(proj > 0, 1.0, -1.0)
    
    # Signed distance = raw distance * sign
    signed_distances = raw_distances * signs
    
    return signed_distances, raw_distances


def prepare_knn_for_sdf(points_norm, k_neighbors=16):
    """
    Prepare kNN model for finding nearest neighbors.
    Works for both 2D and 3D.
    
    Args:
        points_norm: [N, D] normalized on-surface points
        k_neighbors: Number of neighbors to store (can be 1 for just nearest)
    
    Returns:
        knn_model: Fitted NearestNeighbors model
    """
    knn_model = NearestNeighbors(n_neighbors=k_neighbors)
    knn_model.fit(points_norm)
    return knn_model


# ==================== ORIGINAL DATASET CLASSES ====================

class InverseHelmholtz(Dataset):
    # ... (keep the original class as is) ...
    def __init__(self, source_coords, rec_coords, rec_val, sidelength, velocity='uniform', pretrain=False):
        super().__init__()
        torch.manual_seed(0)
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.
        self.pretrain = pretrain
        self.N_src_samples = 100
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)
        self.source_coords = torch.Tensor(source_coords).float()
        self.rec_coords = torch.Tensor(rec_coords).float()
        self.rec = torch.zeros(self.rec_coords.shape[0], 2 * self.source_coords.shape[0])
        for i in range(self.rec.shape[0]):
            self.rec[i, ::2] = torch.Tensor(rec_val.real)[i, :].float()
            self.rec[i, 1::2] = torch.Tensor(rec_val.imag)[i, :].float()

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        else:
            squared_slowness = torch.ones_like(coords)
            squared_slowness[..., 1] = 0.
        return squared_slowness

    def __getitem__(self, idx):
        N_src_coords = self.source_coords.shape[0]
        N_rec_coords = self.rec_coords.shape[0]
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1., 1.)
        samp_source_coords = torch.zeros(self.N_src_samples * N_src_coords, 2)
        for i in range(N_src_coords):
            samp_source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
            samp_source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
            samp_source_coords_x = samp_source_coords_r * torch.cos(samp_source_coords_theta) + self.source_coords[i, 0]
            samp_source_coords_y = samp_source_coords_r * torch.sin(samp_source_coords_theta) + self.source_coords[i, 1]
            samp_source_coords[i * self.N_src_samples:(i + 1) * self.N_src_samples, :] = \
                torch.cat((samp_source_coords_x, samp_source_coords_y), dim=1)
        coords[-self.N_src_samples * N_src_coords:, :] = samp_source_coords
        coords[:N_rec_coords, :] = self.rec_coords
        source_boundary_values = torch.zeros(coords.shape[0], 2 * N_src_coords)
        for i in range(N_src_coords):
            source_boundary_values[:, 2 * i:2 * i + 2] = self.source * \
                                                         gaussian(coords, mu=self.source_coords[i, :],
                                                                  sigma=self.sigma)[:, None]
        source_boundary_values[source_boundary_values < 1e-5] = 0.
        rec_boundary_values = torch.zeros(coords.shape[0], self.rec.shape[1])
        rec_boundary_values[:N_rec_coords:, :] = self.rec
        squared_slowness = torch.Tensor([-1.])
        squared_slowness_grid = torch.Tensor([-1.])
        pretrain = torch.Tensor([-1.])
        if self.pretrain:
            squared_slowness = self.get_squared_slowness(coords)
            squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]
            pretrain = torch.Tensor([1.])
        return {'coords': coords}, {'source_boundary_values': source_boundary_values,
                                    'rec_boundary_values': rec_boundary_values, 'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid, 'wavenumber': self.wavenumber,
                                    'pretrain': pretrain}


class SingleHelmholtzSource(Dataset):
    # ... (keep the original class as is) ...
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0.]):
        super().__init__()
        torch.manual_seed(0)
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.
        self.N_src_samples = 100
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)
        self.source_coords = torch.tensor(source_coords).view(-1, 2)
        square_meshgrid = lin2img(self.mgrid[None, ...]).numpy()
        x = square_meshgrid[0, 0, ...]
        y = square_meshgrid[0, 1, ...]
        source_np = self.source.numpy()
        hx = hy = 2 / self.sidelength
        field = np.zeros((sidelength, sidelength)).astype(np.complex64)
        for i in range(source_np.shape[0]):
            x0 = self.source_coords[i, 0].numpy()
            y0 = self.source_coords[i, 1].numpy()
            s = source_np[i, 0] + 1j * source_np[i, 1]
            hankel = scipy.special.hankel2(0, self.wavenumber * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6)
            field += 0.25j * hankel * s * hx * hy
        field_r = torch.from_numpy(np.real(field).reshape(-1, 1))
        field_i = torch.from_numpy(np.imag(field).reshape(-1, 1))
        self.field = torch.cat((field_r, field_i), dim=1)

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        else:
            squared_slowness = torch.ones_like(coords)
            squared_slowness[..., 1] = 0.
        return squared_slowness

    def __getitem__(self, idx):
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1., 1.)
        source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
        source_coords_x = source_coords_r * torch.cos(source_coords_theta) + self.source_coords[0, 0]
        source_coords_y = source_coords_r * torch.sin(source_coords_theta) + self.source_coords[0, 1]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)
        coords[-self.N_src_samples:, :] = source_coords
        boundary_values = self.source * gaussian(coords, mu=self.source_coords, sigma=self.sigma)[:, None]
        boundary_values[boundary_values < 1e-5] = 0.
        squared_slowness = self.get_squared_slowness(coords)
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'gt': self.field,
                                    'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid,
                                    'wavenumber': self.wavenumber}


class WaveSource(Dataset):
    # ... (keep the original class as is) ...
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0., 0.], pretrain=False):
        super().__init__()
        torch.manual_seed(0)
        self.pretrain = pretrain
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.N_src_samples = 1000
        self.sigma = 5e-4
        self.source_coords = torch.tensor(source_coords).view(-1, 3)
        self.counter = 0
        self.full_count = 100e3

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords[:, 0])
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords[:, 0])
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        else:
            squared_slowness = torch.ones_like(coords[:, 0])
        return squared_slowness

    def __getitem__(self, idx):
        start_time = self.source_coords[0, 0]
        r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        phi = 2 * np.pi * torch.rand(self.N_src_samples, 1)
        source_coords_x = r * torch.cos(phi) + self.source_coords[0, 1]
        source_coords_y = r * torch.sin(phi) + self.source_coords[0, 2]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1, 1)
        if self.pretrain:
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(start_time - 0.001, start_time + 0.001)
            coords = torch.cat((time, coords), dim=1)
            coords[-self.N_src_samples:, 1:] = source_coords
        else:
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(0, 0.4 * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)
            coords[-self.N_src_samples:, 1:] = source_coords
            coords[-2 * self.N_src_samples:, 0] = start_time
        normalize = 50 * gaussian(torch.zeros(1, 2), mu=torch.zeros(1, 2), sigma=self.sigma, d=2)
        boundary_values = gaussian(coords[:, 1:], mu=self.source_coords[:, 1:], sigma=self.sigma, d=2)[:, None]
        boundary_values /= normalize
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            boundary_values = torch.where((coords[:, 0, None] == start_time), boundary_values, torch.Tensor([0]))
            dirichlet_mask = (coords[:, 0, None] == start_time)
        boundary_values[boundary_values < 1e-5] = 0.
        squared_slowness = self.get_squared_slowness(coords)[:, None]
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, None]
        self.counter += 1
        if self.pretrain and self.counter == 2000:
            self.pretrain = False
            self.counter = 0
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask,
                                    'squared_slowness': squared_slowness, 'squared_slowness_grid': squared_slowness_grid}


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid


class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img


class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class CelebA(Dataset):
    def __init__(self, split, downsampled=False):
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = '/media/data3/awb/CelebA/kaggle/img_align_celeba/img_align_celeba'
        self.img_channels = 3
        self.fnames = []

        with open('/media/data3/awb/CelebA/kaggle/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size

            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))

        return img


class ImplicitAudioWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rate, data = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)
        return {'idx': idx, 'coords': self.grid}, {'func': data, 'rate': rate, 'scale': scale}


class AudioFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        print("Rate: %d" % self.rate)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data.view(-1, self.dataset.channels)
        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
            data = self.data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = self.data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class ImageGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, test_sparsity=None, train_sparsity_range=(10, 200), generalization_mode=None):
        self.dataset = dataset
        self.sidelength = dataset.sidelength
        self.mgrid = dataset.mgrid
        self.test_sparsity = test_sparsity
        self.train_sparsity_range = train_sparsity_range
        self.generalization_mode = generalization_mode

    def __len__(self):
        return len(self.dataset)

    def update_test_sparsity(self, test_sparsity):
        self.test_sparsity = test_sparsity

    def get_generalization_in_dict(self, spatial_img, img, idx):
        if self.generalization_mode == 'conv_cnp' or self.generalization_mode == 'conv_cnp_test':
            if self.test_sparsity == 'full':
                img_sparse = spatial_img
            elif self.test_sparsity == 'half':
                img_sparse = spatial_img
                img_sparse[:, 16:, :] = 0.
            else:
                if self.generalization_mode == 'conv_cnp_test':
                    num_context = int(self.test_sparsity)
                else:
                    num_context = int(
                        torch.empty(1).uniform_(self.train_sparsity_range[0], self.train_sparsity_range[1]).item())
                mask = spatial_img.new_empty(
                    1, spatial_img.size(1), spatial_img.size(2)).bernoulli_(p=num_context / np.prod(self.sidelength))
                img_sparse = mask * spatial_img
            in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sparse': img_sparse}
        elif self.generalization_mode == 'cnp' or self.generalization_mode == 'cnp_test':
            if self.test_sparsity == 'full':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img, 'coords_sub': self.mgrid}
            elif self.test_sparsity == 'half':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img[:512, :], 'coords_sub': self.mgrid[:512, :]}
            else:
                if self.generalization_mode == 'cnp_test':
                    subsamples = int(self.test_sparsity)
                    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]
                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}
                else:
                    subsamples = np.random.randint(self.train_sparsity_range[0], self.train_sparsity_range[1])
                    rand_idcs = np.random.choice(img.shape[0], size=self.train_sparsity_range[1], replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]

                    rand_idcs_2 = np.random.choice(img_sparse.shape[0], size=subsamples, replace=False)
                    ctxt_mask = torch.zeros(img_sparse.shape[0], 1)
                    ctxt_mask[rand_idcs_2, 0] = 1.

                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub,
                               'ctxt_mask': ctxt_mask}
        else:
            in_dict = {'idx': idx, 'coords': self.mgrid}

        return in_dict

    def __getitem__(self, idx):
        spatial_img, img, gt_dict = self.dataset.get_item_small(idx)
        in_dict = self.get_generalization_in_dict(spatial_img, img, idx)
        return in_dict, gt_dict


class BSD500ImageDataset(Dataset):
    def __init__(self,
                 in_folder='data/BSD500/train',
                 is_color=False,
                 size=[321, 321],
                 preload=True,
                 idx_to_sample=[]):
        self.in_folder = in_folder
        self.size = size
        self.idx_to_sample = idx_to_sample
        self.is_color = is_color
        self.preload = preload
        if (self.is_color):
            self.img_channels = 3
        else:
            self.img_channels = 1

        self.img_filenames = []
        self.img_preloaded = []
        for idx, filename in enumerate(sorted(glob.glob(self.in_folder + '/*.jpg'))):
            self.img_filenames.append(filename)

            if (self.preload):
                img = self.load_image(filename)
                self.img_preloaded.append(img)

        if (self.preload):
            assert (len(self.img_preloaded) == len(self.img_filenames))

    def load_image(self, filename):
        img = Image.open(filename, 'r')
        if not self.is_color:
            img = img.convert("L")
        img = img.crop((0, 0, self.size[0], self.size[1]))

        return img

    def __len__(self):
        if (len(self.idx_to_sample) != 0):
            return len(self.idx_to_sample)
        else:
            return len(self.img_filenames)

    def __getitem__(self, item):
        if (len(self.idx_to_sample) != 0):
            idx = self.idx_to_sample[item]
        else:
            idx = item

        if (self.preload):
            img = self.img_preloaded[idx]
        else:
            img = self.load_image(self.img_filenames[idx])

        return img


class CompositeGradients(Dataset):
    def __init__(self, img_filepath1, img_filepath2,
                 sidelength=None,
                 is_color=False):
        super().__init__()

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)

        self.is_color = is_color
        if (self.is_color):
            self.channels = 3
        else:
            self.channels = 1

        self.img1 = Image.open(img_filepath1)
        self.img2 = Image.open(img_filepath2)

        if not self.is_color:
            self.img1 = self.img1.convert("L")
            self.img2 = self.img2.convert("L")
        else:
            self.img1 = self.img1.convert("RGB")
            self.img2 = self.img2.convert("RGB")

        self.transform = Compose([
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.mgrid = get_mgrid(sidelength)

        self.img1 = self.transform(self.img1)
        self.img2 = self.transform(self.img2)

        paddedImg = .85 * torch.ones_like(self.img1)
        paddedImg[:, 512 - 340:512, :] = self.img2
        self.img2 = paddedImg

        self.grads1 = self.compute_gradients(self.img1)
        self.grads2 = self.compute_gradients(self.img2)

        self.comp_grads = (.5 * self.grads1 + .5 * self.grads2)

        self.img1 = self.img1.permute(1, 2, 0).view(-1, self.channels)
        self.img2 = self.img2.permute(1, 2, 0).view(-1, self.channels)

    def compute_gradients(self, img):
        if not self.is_color:
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        else:
            gradx = np.moveaxis(scipy.ndimage.sobel(img.numpy(), axis=1), 0, -1)
            grady = np.moveaxis(scipy.ndimage.sobel(img.numpy(), axis=2), 0, -1)

        grads = torch.cat((torch.from_numpy(gradx).reshape(-1, self.channels),
                           torch.from_numpy(grady).reshape(-1, self.channels)),
                          dim=-1)
        return grads

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img1': self.img1,
                   'img2': self.img2,
                   'grads1': self.grads1,
                   'grads2': self.grads2,
                   'gradients': self.comp_grads}

        return in_dict, gt_dict


# ==================== 2D SDF DATASET CLASSES ====================

class PointCloud2D(Dataset):
    """
    Unified dataset for 2D SDF training with REAL signed distances.
    """
    
    def __init__(self, pointcloud_path, on_surface_points, 
                 plane_csv=None,
                 use_signed_labels=False, k_neighbors=1,
                 vis_sampled_point=False, vis_frequency=100, vis_dir=None):
        super().__init__()
        
        self.on_surface_points = on_surface_points
        self.use_signed_labels = use_signed_labels
        self.k_neighbors = k_neighbors
        self.plane_csv = plane_csv
        self.vis_sampled_point = vis_sampled_point
        self.vis_frequency = vis_frequency
        self.vis_dir = vis_dir
        self.epoch_counter = 0
        
        # Load data (either 2D direct or 3D + plane)
        self._load_data(pointcloud_path, plane_csv)
        
        # STORE THE FULL DATASET for kNN queries
        self.all_points_norm = self.points_norm.copy()  # Full set of normalized points
        self.all_normals = self.normals.copy()          # Full set of normals
        
        # Setup for signed labeling if needed
        if self.use_signed_labels:
            print(f"  REAL SDF ENABLED: preparing kNN (k={k_neighbors}) for distance computation...")
            self.knn_model = NearestNeighbors(n_neighbors=k_neighbors)
            self.knn_model.fit(self.all_points_norm)  # Fit on FULL dataset
        else:
            print(f"  REAL SDF DISABLED: using original -1 for all off-surface points")
            self.knn_model = None
        
        # Print dataset info
        self._print_info()
        
        # Visualization setup
        if self.vis_sampled_point and self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
            mode_str = 'real_sdf' if use_signed_labels else 'original'
            print(f"  Batch visualizations ({mode_str} mode) saved every {self.vis_frequency} epochs to: {self.vis_dir}")
    
    def _load_data(self, pointcloud_path, plane_csv):
        """Load data based on whether plane_csv is provided."""
        if plane_csv is not None:
            self._load_3d_with_plane(pointcloud_path, plane_csv)
        else:
            self._load_2d_data(pointcloud_path)
    
    def _load_2d_data(self, pointcloud_path):
        """Load data already in 2D format."""
        print(f"Loading 2D point cloud from: {pointcloud_path}")
        data = np.genfromtxt(pointcloud_path)
        points_raw = data[:, :2]
        self.normals = data[:, 2:4]
        
        # Store original points for potential use
        self.points_raw = points_raw.copy()
        
        # Normalize to [-1, 1] for SIREN
        self.points_raw -= np.mean(self.points_raw, axis=0, keepdims=True)
        coord_max = np.amax(np.abs(self.points_raw))
        if coord_max < 1e-6:
            coord_max = 1.0
        self.points_norm = self.points_raw / coord_max
        
        self.data_source = "2D direct"
        self.transform_info = None
        self.norm_params = {'mean': np.zeros(2), 'scale': coord_max}
    
    def _load_3d_with_plane(self, pointcloud_path, plane_csv):
        """Load 3D point cloud and transform using plane parameters."""
        print(f"Loading 3D point cloud from: {pointcloud_path}")
        print(f"Loading plane from: {plane_csv}")
        
        # Load point cloud
        data = np.genfromtxt(pointcloud_path)
        points_3d = data[:, :3]
        normals_3d = data[:, 3:6]
        print(f"  Loaded {len(points_3d)} points")
        
        # Load plane parameters
        plane_params = np.loadtxt(plane_csv, delimiter=",", skiprows=1)
        if plane_params.ndim == 1:
            plane_params = plane_params.reshape(1, -1)
        plane_origin = plane_params[0, :3]
        plane_normal = plane_params[0, 3:6]
        print(f"  Plane origin: {plane_origin}")
        print(f"  Plane normal: {plane_normal}")
        
        # Transform to XY plane
        points_2d, normals_2d, self.transform_info = transform_plane_to_xy(
            points_3d, normals_3d, plane_origin, plane_normal
        )
        
        # Store transformed points
        self.points_raw = points_2d.copy()
        self.normals = normals_2d.copy()
        
        # Normalize to [-1, 1]
        self.points_norm, self.norm_params = normalize_to_unit_square(self.points_raw)
        
        self.data_source = "3D + plane"
    
    def _print_info(self):
        """Print dataset information."""
        print(f"\n2D Dataset ready ({self.data_source}):")
        print(f"  Points: {len(self.points_norm)} on-surface points")
        print(f"  Normalized range: u=[{self.points_norm[:,0].min():.3f}, {self.points_norm[:,0].max():.3f}], "
              f"v=[{self.points_norm[:,1].min():.3f}, {self.points_norm[:,1].max():.3f}]")
        print(f"  Label mode: {'REAL SDF (signed distances)' if self.use_signed_labels else 'ORIGINAL (-1 only)'}")
    
    def __len__(self):
        return max(1, self.points_norm.shape[0] // self.on_surface_points)
    
    def __getitem__(self, idx):
        n_points = self.points_norm.shape[0]
        
        # On-surface points (random subset for this batch)
        if n_points >= self.on_surface_points:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=False)
        else:
            on_idx = np.random.choice(n_points, self.on_surface_points, replace=True)
        
        on_coords = self.points_norm[on_idx]
        on_normals = self.normals[on_idx]
        
        # Off-surface points
        off_coords = np.random.uniform(-1, 1, size=(self.on_surface_points, 2))
        off_normals = np.ones((self.on_surface_points, 2)) * -1
        
        # === USE REAL SDF COMPUTATION FUNCTION WITH FULL DATASET ===
        if self.use_signed_labels:
            # Pass the FULL dataset for kNN queries, not just the batch subset
            sdf, sdf_on, sdf_off, distances = compute_real_sdf_values(
                on_coords=on_coords,                    # Batch subset for on-surface
                off_coords=off_coords,                  # Batch subset for off-surface
                on_normals=self.all_normals,            # FULL dataset normals
                use_signed_labels=True,
                knn_model=self.knn_model,               # kNN trained on FULL dataset
                on_points_norm=self.all_points_norm     # FULL dataset points
            )
        else:
            # Original mode: all off-surface = -1
            sdf_on = np.zeros(self.on_surface_points)
            sdf_off = np.ones(self.on_surface_points) * -1
            sdf = np.concatenate([sdf_on, sdf_off])
            distances = None
        
        coords = np.vstack([on_coords, off_coords])
        normals = np.vstack([on_normals, off_normals])
        
        # Visualization
        if (self.vis_sampled_point and self.vis_dir and idx == 0 and 
            self.epoch_counter % self.vis_frequency == 0):
            
            if self.use_signed_labels:
                visualize_training_batch_real_sdf(
                    coords=coords,
                    gt_sdf=sdf.reshape(-1, 1),
                    distances=distances,
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

