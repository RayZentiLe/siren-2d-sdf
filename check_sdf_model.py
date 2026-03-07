#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive SIREN Model Verification Tool
-------------------------------------------
This script verifies SIREN (Implicit Neural Representation) models by:
- Computing statistical properties of the SDF field
- Visualizing grid samples and contours
- Checking Eikonal equation satisfaction (|∇f| = 1)
- Comparing with reference contour points (2D)
- Analyzing gradient fields
- Generating comprehensive reports
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import KDTree, ConvexHull
from matplotlib.path import Path
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

# Optional mesh checks for 3D verification
try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False
    print("Note: trimesh not installed. Mesh-based checks will be skipped.")

# ============================================================================
# SIREN Model Definition (Inference Only)
# ============================================================================

class SineLayer(nn.Module):
    """
    Sine activation layer with frequency scaling.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to use bias
        is_first: Whether this is the first layer (uses w0_initial)
        omega_0: Frequency factor for sine activation
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.omega_0 = omega_0
        self.is_first = is_first
        
    def forward(self, x):
        """Apply sine activation with frequency scaling."""
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNet(nn.Module):
    """
    SIREN network for implicit neural representations.
    
    Args:
        dim_in: Input dimension (2 for 2D, 3 for 3D)
        dim_hidden: Hidden layer dimension
        dim_out: Output dimension (1 for SDF)
        num_layers: Number of sine layers
        w0: Omega_0 for hidden layers
        w0_initial: Omega_0 for first layer
        outermost_linear: Whether output layer is linear (vs sine)
        use_bias: Whether to use bias in linear layers
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 w0=30.0, w0_initial=30.0, outermost_linear=True, use_bias=True):
        super().__init__()
        
        # Build hidden layers with sine activation
        layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer uses w0_initial
                layers.append(SineLayer(dim_in, dim_hidden, bias=use_bias, 
                                       is_first=True, omega_0=w0_initial))
            else:
                # Subsequent layers use w0
                layers.append(SineLayer(dim_hidden, dim_hidden, bias=use_bias, 
                                       is_first=False, omega_0=w0))
        
        self.net = nn.Sequential(*layers)
        
        # Final linear layer (optional)
        self.final_linear = nn.Linear(dim_hidden, dim_out, bias=use_bias) if outermost_linear else None

    def forward(self, x):
        """Forward pass through the network."""
        x = self.net(x)
        if self.final_linear is not None:
            x = self.final_linear(x)
        return x


# ============================================================================
# Robust Model Loader (Handles Various Checkpoint Formats)
# ============================================================================

def _extract_state_dict(obj):
    """
    Recursively extract state_dict from various checkpoint formats.
    
    Args:
        obj: Loaded checkpoint (dict, module, or other)
    
    Returns:
        Extracted state_dict or original object
    """
    if isinstance(obj, dict):
        # Try common checkpoint key names
        for key in ["state_dict", "model_state_dict", "ema_state_dict", "model"]:
            if key in obj:
                inner = obj[key]
                if isinstance(inner, (dict, nn.Module)):
                    return inner
    return obj


def _find_linear_params(state_dict):
    """
    Find all linear layer parameters (weight+bias pairs) in state_dict.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        List of tuples (base_name, weight_tensor, bias_tensor)
    """
    params = []
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and key.endswith(".weight"):
            base_name = key[:-7]  # Remove ".weight"
            bias_key = base_name + ".bias"
            if bias_key in state_dict and isinstance(state_dict[bias_key], torch.Tensor) and state_dict[bias_key].ndim == 1:
                params.append((base_name, value, state_dict[bias_key]))
    return params


def _key_index_hint(key):
    """
    Extract layer index from parameter key for sorting.
    
    Args:
        key: Parameter key string
    
    Returns:
        Estimated layer index
    """
    numbers = re.findall(r"(?:^|\.)(\d+)(?=\.|$)", key)
    return int(numbers[-1]) if numbers else 1_000_000


def build_siren_from_state_dict(state_dict, w0=30.0, w0_initial=30.0, prefer_in=None):
    """
    Infer SIREN architecture from state_dict and build model with loaded weights.
    
    Args:
        state_dict: Model state dictionary
        w0: Omega_0 for hidden layers
        w0_initial: Omega_0 for first layer
        prefer_in: Preferred input dimension (auto-detect if None)
    
    Returns:
        Initialized SirenNet with loaded weights
    """
    # Find all linear layers
    linear_params = _find_linear_params(state_dict)
    if not linear_params:
        raise RuntimeError("No linear (weight,bias) tensors found in checkpoint.")
    
    # Determine input dimension
    if prefer_in is None:
        # Use most common input dimension
        in_dims = [int(w.shape[1]) for _, w, _ in linear_params]
        prefer_in = max(set(in_dims), key=in_dims.count)
    
    # Find first layer (smallest input dimension or matching prefer_in)
    first_layers = [(name, w, b) for (name, w, b) in linear_params 
                   if int(w.shape[1]) == prefer_in]
    if not first_layers:
        min_in = min(int(w.shape[1]) for _, w, _ in linear_params)
        first_layers = [(name, w, b) for (name, w, b) in linear_params 
                       if int(w.shape[1]) == min_in]
    
    # Sort by layer index
    first_layers.sort(key=lambda t: _key_index_hint(t[0]))
    name0, weight0, bias0 = first_layers[0]
    
    # Get hidden dimension and input dimension
    hidden_dim = int(weight0.shape[0])
    input_dim = int(weight0.shape[1])
    
    # Find hidden layers (square weight matrices)
    hidden_layers = [(name, w, b) for (name, w, b) in linear_params 
                    if int(w.shape[0]) == hidden_dim and int(w.shape[1]) == hidden_dim]
    hidden_layers.sort(key=lambda t: _key_index_hint(t[0]))
    
    # Find final layer (output dimension != hidden_dim)
    final_layers = [(name, w, b) for (name, w, b) in linear_params 
                   if int(w.shape[1]) == hidden_dim and int(w.shape[0]) in (1, 2, 3, 4)]
    final_layers.sort(key=lambda t: (int(t[1].shape[0]), _key_index_hint(t[0])))
    
    has_final = len(final_layers) > 0
    output_dim = int(final_layers[0][1].shape[0]) if has_final else 1
    
    # Build model
    num_layers = 1 + len(hidden_layers)
    model = SirenNet(
        input_dim, hidden_dim, output_dim, num_layers,
        w0=float(w0), w0_initial=float(w0_initial),
        outermost_linear=has_final, use_bias=True
    )
    
    # Load weights
    with torch.no_grad():
        # First layer
        model.net[0].linear.weight.copy_(weight0)
        model.net[0].linear.bias.copy_(bias0)
        
        # Hidden layers
        for i, (_, w, b) in enumerate(hidden_layers, start=1):
            model.net[i].linear.weight.copy_(w)
            model.net[i].linear.bias.copy_(b)
        
        # Final layer
        if has_final:
            _, w_final, b_final = final_layers[0]
            model.final_linear.weight.copy_(w_final)
            model.final_linear.bias.copy_(b_final)
    
    return model


def load_siren_auto(path, device, w0=30.0, w0_initial=30.0, dim=None):
    """
    Universal SIREN loader - handles TorchScript, state_dict, and full modules.
    
    Args:
        path: Path to model file
        device: Device to load model on
        w0: Omega_0 for hidden layers
        w0_initial: Omega_0 for first layer
        dim: Input dimension (auto-detected if None)
    
    Returns:
        Loaded and initialized SIREN model in eval mode
    """
    # Try TorchScript first
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        print("[loader] TorchScript module loaded successfully")
        return model
    except Exception:
        pass
    
    # Load raw checkpoint
    raw_checkpoint = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(raw_checkpoint)
    
    # If it's already a module, use it directly
    if isinstance(state_dict, nn.Module):
        print("[loader] Checkpoint contained a full Module; using as-is")
        return state_dict.to(device).eval()
    
    # Otherwise, infer architecture and build model
    model = build_siren_from_state_dict(state_dict, w0=w0, w0_initial=w0_initial, prefer_in=dim)
    print("[loader] SIREN architecture inferred and weights loaded")
    return model.to(device).eval()


# ============================================================================
# Grid Sampling Functions
# ============================================================================

def sample_sdf_grid_2d(decoder, N=256, max_batch=64**2, device="cuda", span=1.0):
    """
    Sample 2D SDF on a grid - FIXED indexing.
    """
    decoder.eval()
    
    u = np.linspace(-span, span, N)
    v = np.linspace(-span, span, N)
    
    # CORRECT: u is first dimension, v is second
    # But the INDEXING must match how you query later
    uu, vv = np.meshgrid(u, v, indexing='ij')
    
    # Flatten in the SAME order as your test points
    # This is critical - the order must be consistent
    grid_points = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)
    
    print(f"  Grid shape: {uu.shape}")
    print(f"  First few grid points:")
    for i in range(5):
        print(f"    {grid_points[i]}")
    
    # Find where (0.1,0.0) is in the grid
    target = np.array([0.1, 0.0])
    distances = np.linalg.norm(grid_points - target, axis=1)
    idx = np.argmin(distances)
    u_idx, v_idx = np.unravel_index(idx, (N, N))
    print(f"  (0.1,0.0) at grid index: u={u_idx}, v={v_idx}")
    print(f"  Grid value at that index: u={uu.flat[idx]:.4f}, v={vv.flat[idx]:.4f}")
    
    # Evaluate
    vals = np.zeros(len(grid_points), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(grid_points), max_batch):
            batch = grid_points[i:i+max_batch]
            batch_tensor = torch.from_numpy(batch).to(device)
            out = decoder(batch_tensor).view(-1).cpu().numpy()
            vals[i:i+max_batch] = out
    
    sdf_grid = vals.reshape(N, N)
    
    # Now check the values at the correct indices
    print(f"  SDF at (0.1,0.0) [idx={idx}]: {vals[idx]:.4f}")
    print(f"  SDF at u_idx={u_idx}, v_idx={v_idx}: {sdf_grid[u_idx, v_idx]:.4f}")
    
    return sdf_grid, u, v

def sample_sdf_grid_3d(decoder, N=128, max_batch=64**3, device="cuda", span=1.0):
    """
    Sample 3D SDF on a grid from [-span, span]^3.
    
    Args:
        decoder: SIREN model
        N: Grid resolution (N x N x N)
        max_batch: Maximum batch size for evaluation
        device: Device to run on
        span: Coordinate span (grid from -span to span)
    
    Returns:
        sdf_grid: 3D array of SDF values [N, N, N]
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        z_coords: 1D array of z coordinates
    """
    decoder.eval()
    
    # Generate grid coordinates
    coords = np.linspace(-span, span, N)
    voxel_origin = [-span, -span, -span]
    voxel_size = (2 * span) / (N - 1)
    
    # Create flattened grid with proper ordering (z, y, x for marching cubes compatibility)
    idx = torch.arange(0, N**3, dtype=torch.long)
    samples = torch.zeros(N**3, 3, dtype=torch.float32)
    
    # z coordinate (first dimension for marching cubes)
    samples[:, 2] = idx % N
    # y coordinate
    samples[:, 1] = (idx // N) % N
    # x coordinate
    samples[:, 0] = (idx // (N*N)) % N
    
    # Convert to world coordinates
    samples[:, 0] = samples[:, 0] * voxel_size + voxel_origin[2]  # x
    samples[:, 1] = samples[:, 1] * voxel_size + voxel_origin[1]  # y
    samples[:, 2] = samples[:, 2] * voxel_size + voxel_origin[0]  # z
    
    # Evaluate in batches
    values = torch.empty(N**3, dtype=torch.float32)
    head = 0
    with torch.no_grad():
        while head < N**3:
            tail = min(head + max_batch, N**3)
            batch = samples[head:tail].to(device)
            out = decoder(batch).view(-1).detach().cpu()
            values[head:tail] = out
            head = tail
    
    # Reshape to [N, N, N] with ordering [x, y, z] for compatibility
    sdf_grid = values.view(N, N, N).cpu().numpy()
    
    return sdf_grid, coords, coords, coords


# ============================================================================
# Batch Evaluation Helper
# ============================================================================

def batched_eval(decoder, points_np, device, batch_size=131072):
    """
    Evaluate SDF on points in batches to avoid OOM.
    
    Args:
        decoder: SIREN model
        points_np: Numpy array of points [N, D]
        device: Device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        SDF values as numpy array
    """
    output = np.empty((points_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        i = 0
        while i < points_np.shape[0]:
            j = min(i + batch_size, points_np.shape[0])
            batch = torch.from_numpy(points_np[i:j]).to(device)
            output[i:j] = decoder(batch).view(-1).cpu().numpy()
            i = j
    return output


def random_points(n, span, dim=3):
    """
    Generate random points uniformly in [-span, span]^dim.
    
    Args:
        n: Number of points
        span: Coordinate span
        dim: Dimension (2 or 3)
    
    Returns:
        Array of random points [n, dim]
    """
    return np.random.uniform(-span, span, size=(n, dim)).astype(np.float32)


# ============================================================================
# PLY File Loading
# ============================================================================

def load_ply_points(ply_path):
    """
    Load 3D points (with optional RGB) from PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        points: Array of points [N, 3]
        colors: Array of RGB colors [N, 3] or None if not present
    """
    if not os.path.exists(ply_path):
        print(f"  Warning: PLY file not found: {ply_path}")
        return None, None
    
    points = []
    colors = []
    
    with open(ply_path, 'r') as f:
        # Parse header
        header_done = False
        num_vertices = 0
        has_r = has_g = has_b = False
        
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property uchar red'):
                has_r = True
            elif line.startswith('property uchar green'):
                has_g = True
            elif line.startswith('property uchar blue'):
                has_b = True
            elif line == 'end_header':
                header_done = True
                break
        
        has_colors = has_r and has_g and has_b
        
        # Parse vertices
        for i, line in enumerate(f):
            if i >= num_vertices:
                break
            parts = line.strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
            
            if has_colors:
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                colors.append([r, g, b])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8) if colors else None
    
    return points, colors


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def check_basic_stats(decoder, n_points=200000, span=1.0, device="cuda", dim=3):
    """
    Compute basic statistics of SDF values on random points.
    
    Args:
        decoder: SIREN model
        n_points: Number of random points
        span: Coordinate span
        device: Device to run on
        dim: Dimension (2 or 3)
    
    Returns:
        Dictionary of statistics
    """
    points = random_points(n_points, span, dim)
    values = batched_eval(decoder, points, device)
    
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "frac_negative": float((values < 0).mean()),
        "frac_positive": float((values > 0).mean()),
        "frac_near_zero_1e_3": float((np.abs(values) < 1e-3).mean()),
        "frac_near_zero_1e_2": float((np.abs(values) < 1e-2).mean()),
    }


def grad_norm_stats(decoder, n=50000, span=1.0, device="cuda", batch_size=32768, dim=3):
    """
    Compute gradient norm statistics (Eikonal check: |∇f| should be 1).
    
    Args:
        decoder: SIREN model
        n: Number of points
        span: Coordinate span
        device: Device to run on
        batch_size: Batch size for gradient computation
        dim: Dimension (2 or 3)
    
    Returns:
        stats: Dictionary of gradient statistics
        all_norms: Array of gradient norms (or None if failed)
    """
    try:
        all_norms = []
        decoder.zero_grad()
        
        for i in range(0, n, batch_size):
            current_batch = min(batch_size, n - i)
            points = torch.from_numpy(random_points(current_batch, span, dim)).to(device)
            points.requires_grad_(True)
            
            # Forward pass
            sdf_values = decoder(points).view(-1)
            
            # Compute gradients
            grad_outputs = torch.ones_like(sdf_values)
            gradients = torch.autograd.grad(
                sdf_values, points, grad_outputs=grad_outputs,
                retain_graph=False, create_graph=False
            )[0]
            
            # Compute norms
            norms = gradients.norm(dim=1).detach().cpu().numpy()
            all_norms.append(norms)
        
        all_norms = np.concatenate(all_norms, axis=0)
        errors = np.abs(all_norms - 1.0)
        
        stats = {
            "count": int(all_norms.size),
            "mean_norm": float(all_norms.mean()),
            "median_norm": float(np.median(all_norms)),
            "std_norm": float(all_norms.std()),
            "min_norm": float(all_norms.min()),
            "max_norm": float(all_norms.max()),
            "mean_abs_error_from_1": float(errors.mean()),
            "median_abs_error_from_1": float(np.median(errors)),
            "frac_near_1_01": float((np.abs(all_norms - 1.0) < 0.1).mean()),
            "frac_near_1_02": float((np.abs(all_norms - 1.0) < 0.2).mean()),
            "frac_near_1_05": float((np.abs(all_norms - 1.0) < 0.5).mean()),
        }
        
        return stats, all_norms
        
    except Exception as e:
        print(f"  Warning: Gradient computation failed: {e}")
        return {"error": f"gradients unavailable: {e.__class__.__name__}: {e}"}, None


# ============================================================================
# 2D Visualization and Analysis Functions
# ============================================================================

def save_2d_sdf_plot(sdf_grid, u_coords, v_coords, title, output_path):
    """
    Save 2D SDF visualization - NO COORDINATE SWAPPING.
    """
    plt.figure(figsize=(8, 6))
    
    # CORRECT: u is Y axis, v is X axis in imshow
    # extent = [v_min, v_max, u_min, u_max]
    extent = [v_coords.min(), v_coords.max(), u_coords.min(), u_coords.max()]
    
    plt.imshow(sdf_grid, origin='lower', extent=extent,
               cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='SDF Value')
    
    # Zero contour
    plt.contour(v_coords, u_coords, sdf_grid, levels=[0],
                colors='black', linewidths=1)
    
    plt.title(title)
    plt.xlabel('v')
    plt.ylabel('u')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_sdf_contours_2d(sdf_grid, u_coords, v_coords, output_path,
                               levels=20, highlight_zero=True):
    """
    Create comprehensive contour visualization with multiple plots.
    
    Args:
        sdf_grid: 2D array of SDF values
        u_coords: 1D array of u coordinates
        v_coords: 1D array of v coordinates
        output_path: Output file path
        levels: Number of contour levels
        highlight_zero: Whether to highlight zero level set
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    extent = [v_coords.min(), v_coords.max(), u_coords.min(), u_coords.max()]
    
    # Plot 1: SDF heatmap
    ax = axes[0, 0]
    im = ax.imshow(sdf_grid, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title('SDF Heatmap')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    # Plot 2: Contour lines only
    ax = axes[0, 1]
    contour = ax.contour(v_coords, u_coords, sdf_grid, levels=levels,
                         cmap='RdBu_r', linewidths=1)
    ax.set_title(f'SDF Contour Lines ({levels} levels)')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, label='SDF Value')
    
    # Plot 3: Zero level set
    ax = axes[0, 2]
    if highlight_zero and np.any(sdf_grid <= 0) and np.any(sdf_grid >= 0):
        zero_contour = ax.contour(v_coords, u_coords, sdf_grid, levels=[0],
                                   colors='red', linewidths=2)
        ax.set_title('Zero Level Set (SDF=0)')
        if hasattr(zero_contour, 'allsegs') and len(zero_contour.allsegs[0]) > 0:
            ax.clabel(zero_contour, inline=True, fontsize=10, fmt='%.2f')
    else:
        ax.text(0.5, 0.5, 'No zero contour found',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Zero Level Set')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap + zero contour
    ax = axes[1, 0]
    im = ax.imshow(sdf_grid, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5, alpha=0.8)
    if np.any(sdf_grid <= 0) and np.any(sdf_grid >= 0):
        ax.contour(v_coords, u_coords, sdf_grid, levels=[0],
                   colors='black', linewidths=2)
    ax.set_title('SDF Heatmap + Zero Contour')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    # Plot 5: Important contours
    ax = axes[1, 1]
    important_levels = [-0.2, -0.1, 0, 0.1, 0.2]
    contour = ax.contour(v_coords, u_coords, sdf_grid, levels=important_levels,
                         cmap='viridis', linewidths=1.5)
    ax.set_title(f'Important Contours: {important_levels}')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_aspect('equal')
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Distance to zero contour
    ax = axes[1, 2]
    from scipy.ndimage import distance_transform_edt
    binary = sdf_grid < 0
    dist_to_contour = distance_transform_edt(~binary) + distance_transform_edt(binary)
    dist_to_contour = np.where(binary, -dist_to_contour, dist_to_contour)
    
    im = ax.imshow(dist_to_contour, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    if np.any(sdf_grid <= 0) and np.any(sdf_grid >= 0):
        ax.contour(v_coords, u_coords, sdf_grid, levels=[0],
                   colors='black', linewidths=1)
    ax.set_title('Distance to Zero Contour')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    plt.colorbar(im, ax=ax, label='Signed Distance')
    
    plt.suptitle('2D SDF Contour Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved contour visualization to: {output_path}")


def extract_and_save_zero_contour(sdf_grid, u_coords, v_coords, output_path):
    """
    Extract zero level set using marching squares and save visualization.
    
    Args:
        sdf_grid: 2D array of SDF values
        u_coords: 1D array of u coordinates
        v_coords: 1D array of v coordinates
        output_path: Output file path
    
    Returns:
        List of contour fragments in world coordinates
    """
    # Extract zero level set using marching squares
    # Returns coordinates in (row, col) format where row = u index, col = v index
    contours = measure.find_contours(sdf_grid, level=0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert pixel coordinates to world coordinates
    world_contours = []
    for i, contour in enumerate(contours):
        # contour[:, 0] is row (u), contour[:, 1] is col (v)
        u_world = u_coords[0] + (contour[:, 0] / (sdf_grid.shape[0] - 1)) * (u_coords[-1] - u_coords[0])
        v_world = v_coords[0] + (contour[:, 1] / (sdf_grid.shape[1] - 1)) * (v_coords[-1] - v_coords[0])
        world_contour = np.stack([u_world, v_world], axis=1)
        world_contours.append(world_contour)
        
        # Plot each contour
        ax.plot(v_world, u_world, linewidth=2, label=f'Contour {i+1}' if i < 5 else '')
    
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_title(f'Zero Level Set Contours (found {len(contours)} fragments)')
    ax.set_xlim(v_coords[0], v_coords[-1])
    ax.set_ylim(u_coords[0], u_coords[-1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if 0 < len(contours) <= 5:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved zero contour to: {output_path}")
    
    return world_contours



def create_overlay_analysis_2d(sdf_grid, u_coords, v_coords, points_list, output_dir):
    """
    Create overlay visualizations - NO COORDINATE SWAPPING for points.
    """
    os.makedirs(output_dir, exist_ok=True)
    extent = [v_coords.min(), v_coords.max(), u_coords.min(), u_coords.max()]
    
    for plot_type in ['heatmap', 'contours', 'zero_level']:
        fig, axes = plt.subplots(1, len(points_list), 
                                  figsize=(8 * len(points_list), 7))
        if len(points_list) == 1:
            axes = [axes]
        
        for ax_idx, (points, colors, label) in enumerate(points_list):
            ax = axes[ax_idx]
            
            if plot_type == 'heatmap':
                im = ax.imshow(sdf_grid, origin='lower', extent=extent,
                               cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                plt.colorbar(im, ax=ax, label='SDF Value')
                ax.set_title(f'SDF Heatmap - {label}')
                
            elif plot_type == 'contours':
                contour = ax.contour(v_coords, u_coords, sdf_grid, 
                                     levels=20, cmap='viridis', linewidths=1)
                plt.colorbar(contour, ax=ax, label='SDF Value')
                ax.set_title(f'Contour Lines - {label}')
                
            elif plot_type == 'zero_level':
                if np.any(sdf_grid <= 0) and np.any(sdf_grid >= 0):
                    ax.contour(v_coords, u_coords, sdf_grid, levels=[0],
                               colors='red', linewidths=2)
                ax.set_title(f'Zero Level Set - {label}')
                ax.grid(True, alpha=0.3)
            
            # CORRECT: Points are in (u, v) coordinates
            # Plot with u as Y, v as X to match imshow orientation
            if points is not None:
                if colors is not None:
                    ax.scatter(points[:, 1], points[:, 0], 
                              c=colors/255.0, s=3, alpha=0.8)
                else:
                    ax.scatter(points[:, 1], points[:, 0], 
                              c='black', s=3, alpha=0.5)
            
            ax.set_xlabel('v')
            ax.set_ylabel('u')
            ax.set_xlim(v_coords[0], v_coords[-1])
            ax.set_ylim(u_coords[0], u_coords[-1])
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sdf_overlay_{plot_type}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def compare_with_original_contour(sdf_grid, u_coords, v_coords, original_points,
                                    output_path, model=None, device='cuda'):
    """
    Compare model's zero contour with original contour points.
    
    Args:
        sdf_grid: 2D array of SDF values
        u_coords: 1D array of u coordinates
        v_coords: 1D array of v coordinates
        original_points: Original contour points [N, 2]
        output_path: Output file path
        model: Optional model for SDF evaluation at original points
        device: Device for model evaluation
    
    Returns:
        Dictionary of comparison metrics
    """
    # Extract zero contour
    contours = measure.find_contours(sdf_grid, level=0)
    
    if len(contours) == 0:
        print("  Warning: No zero contour found in SDF grid")
        return None
    
    # Convert to world coordinates
    world_contours = []
    for contour in contours:
        u_world = u_coords[0] + (contour[:, 0] / (sdf_grid.shape[0] - 1)) * (u_coords[-1] - u_coords[0])
        v_world = v_coords[0] + (contour[:, 1] / (sdf_grid.shape[1] - 1)) * (v_coords[-1] - v_coords[0])
        world_contours.append(np.stack([v_world, u_world], axis=1))  # Note: store as (v, u) for plotting
    
    all_contour_points = np.vstack(world_contours)
    
    # Compute distances from original points to extracted contour
    tree = KDTree(all_contour_points)
    distances, _ = tree.query(original_points)
    
    # Get SDF values at original points if model provided
    sdf_at_original = None
    if model is not None:
        points_tensor = torch.from_numpy(original_points).float().to(device)
        with torch.no_grad():
            sdf_at_original = model(points_tensor).cpu().numpy().flatten()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Original vs extracted contour
    ax = axes[0]
    ax.scatter(original_points[:, 0], original_points[:, 1],
               c='green', s=2, alpha=0.5, label='Original')
    for wc in world_contours:
        ax.plot(wc[:, 0], wc[:, 1], 'r-', linewidth=1,
                label='Extracted' if wc is world_contours[0] else '')
    ax.set_title(f'Contour Comparison (mean dist={distances.mean():.4f})')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_xlim(v_coords[0], v_coords[-1])
    ax.set_ylim(u_coords[0], u_coords[-1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Distance histogram
    ax = axes[1]
    ax.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.4f}')
    ax.axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.4f}')
    ax.set_xlabel('Distance to extracted contour')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SDF values at original points
    ax = axes[2]
    if sdf_at_original is not None:
        scatter = ax.scatter(original_points[:, 0], original_points[:, 1],
                            c=sdf_at_original, cmap='RdBu_r', s=5, alpha=0.8,
                            vmin=-0.1, vmax=0.1)
        ax.set_title(f'SDF at Original Points (mean |val|={np.abs(sdf_at_original).mean():.4f})')
        plt.colorbar(scatter, ax=ax, label='SDF Value')
    else:
        ax.text(0.5, 0.5, 'No model provided', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SDF Values (unavailable)')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_xlim(v_coords[0], v_coords[-1])
    ax.set_ylim(u_coords[0], u_coords[-1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Contour Accuracy Analysis')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved contour comparison to: {output_path}")
    
    return {
        'mean_distance': float(distances.mean()),
        'median_distance': float(np.median(distances)),
        'max_distance': float(distances.max()),
        'std_distance': float(distances.std()),
        'rmse': float(np.sqrt((distances**2).mean()))
    }


def eikonal_error_map(decoder, resolution=100, device='cuda', outdir=None):
    """
    Create heatmap of |∇f| - 1 error on a 2D grid.
    
    Args:
        decoder: SIREN model
        resolution: Grid resolution
        device: Device to run on
        outdir: Output directory for saving plots
    
    Returns:
        Dictionary of error statistics
    """
    # Create grid
    x = np.linspace(-0.9, 0.9, resolution)
    y = np.linspace(-0.9, 0.9, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Compute gradients in batches
    batch_size = 1000
    errors = []
    grad_norms_list = []
    
    with torch.set_grad_enabled(True):
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_tensor.requires_grad_(True)
            
            sdf = decoder(batch_tensor).view(-1)
            
            # Compute gradients for each point
            for j in range(sdf.shape[0]):
                grad = torch.autograd.grad(sdf[j], batch_tensor, retain_graph=True)[0][j]
                grad_norms_list.append(grad.detach().cpu().numpy())
                errors.append(np.abs(np.linalg.norm(grad.detach().cpu().numpy()) - 1.0))
    
    error_map = np.array(errors).reshape(resolution, resolution)
    grad_norms_map = np.array([np.linalg.norm(g) for g in grad_norms_list]).reshape(resolution, resolution)
    
    stats = {
        'mean_error': float(error_map.mean()),
        'max_error': float(error_map.max()),
        'std_error': float(error_map.std()),
        'median_error': float(np.median(error_map)),
        'mean_grad_norm': float(grad_norms_map.mean()),
        'std_grad_norm': float(grad_norms_map.std()),
    }
    
    if outdir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Eikonal error
        ax = axes[0]
        im = ax.imshow(error_map.T, origin='lower', extent=[-0.9, 0.9, -0.9, 0.9],
                       cmap='hot', vmin=0, vmax=1)
        ax.set_title('Eikonal Error |∇f| - 1|')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        # Gradient magnitude
        ax = axes[1]
        im = ax.imshow(grad_norms_map.T, origin='lower', extent=[-0.9, 0.9, -0.9, 0.9],
                       cmap='viridis', vmin=0, vmax=2)
        ax.set_title('Gradient Magnitude (should be 1)')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'eikonal_error.png'), dpi=150)
        plt.close()
    
    return stats


def visualize_gradient_field(decoder, resolution=30, device='cuda', outdir=None):
    """
    Create quiver plot of SDF gradients.
    
    Args:
        decoder: SIREN model
        resolution: Grid resolution
        device: Device to run on
        outdir: Output directory for saving plots
    
    Returns:
        Dictionary of gradient statistics
    """
    # Create sparse grid
    x = np.linspace(-0.9, 0.9, resolution)
    y = np.linspace(-0.9, 0.9, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Compute gradients
    grad_norms_list = []
    grad_dirs_list = []
    
    with torch.set_grad_enabled(True):
        for i in range(0, len(grid_points), 1000):
            batch = grid_points[i:i+1000]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_tensor.requires_grad_(True)
            
            sdf = decoder(batch_tensor).view(-1)
            
            for j in range(sdf.shape[0]):
                grad = torch.autograd.grad(sdf[j], batch_tensor, retain_graph=True)[0][j]
                grad_norm = torch.norm(grad)
                grad_norms_list.append(grad_norm.detach().cpu().numpy())
                if grad_norm > 1e-6:
                    grad_dirs_list.append((grad / grad_norm).detach().cpu().numpy())
                else:
                    grad_dirs_list.append(np.zeros(2))
    
    grad_norms = np.array(grad_norms_list)
    grad_dirs = np.array(grad_dirs_list)
    
    if outdir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gradient magnitude scatter
        ax = axes[0]
        scatter = ax.scatter(grid_points[:, 0], grid_points[:, 1],
                            c=grad_norms, cmap='viridis', s=20, vmin=0, vmax=2)
        ax.set_title('Gradient Magnitude (should be 1)')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        
        # Gradient direction quiver
        ax = axes[1]
        ax.quiver(grid_points[:, 0], grid_points[:, 1],
                  grad_dirs[:, 0], grad_dirs[:, 1],
                  grad_norms, cmap='viridis', scale=20, width=0.003)
        ax.set_title('Gradient Directions')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'gradient_field.png'), dpi=150)
        plt.close()
    
    return {
        'mean_grad_norm': float(grad_norms.mean()),
        'std_grad_norm': float(grad_norms.std()),
        'min_grad_norm': float(grad_norms.min()),
        'max_grad_norm': float(grad_norms.max()),
    }


def plot_sdf_profile(decoder, line_start, line_end, num_points=1000, device='cuda', outdir=None, prefix=''):
    """
    Plot SDF values along a line between two points.
    
    Args:
        decoder: SIREN model
        line_start: Start point [x, y]
        line_end: End point [x, y]
        num_points: Number of points along line
        device: Device to run on
        outdir: Output directory
        prefix: Prefix for output filename
    
    Returns:
        Dictionary of profile statistics
    """
    # Generate points along line
    t = np.linspace(0, 1, num_points)
    points = np.outer(1 - t, line_start) + np.outer(t, line_end)
    
    # Evaluate SDF
    points_tensor = torch.from_numpy(points).float().to(device)
    with torch.no_grad():
        sdf_values = decoder(points_tensor).cpu().numpy().flatten()
    
    # Find zero crossings
    zero_crossings = []
    for i in range(len(sdf_values) - 1):
        if sdf_values[i] * sdf_values[i+1] < 0:
            t_zero = t[i] + (t[i+1] - t[i]) * (0 - sdf_values[i]) / (sdf_values[i+1] - sdf_values[i])
            zero_crossings.append(t_zero)
    
    stats = {
        f'{prefix}_min': float(sdf_values.min()),
        f'{prefix}_max': float(sdf_values.max()),
        f'{prefix}_mean': float(sdf_values.mean()),
        f'{prefix}_num_crossings': len(zero_crossings),
    }
    
    if outdir:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, sdf_values, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        for tz in zero_crossings:
            ax.axvline(x=tz, color='r', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('t (along line)')
        ax.set_ylabel('SDF Value')
        ax.set_title(f'SDF Profile from {line_start} to {line_end}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'profile_{prefix}.png'), dpi=150)
        plt.close()
    
    return stats


def check_sign_consistency_2d(decoder, contour_points_2d, num_random=100000, device='cuda', outdir=None):
    """
    Check sign consistency using convex hull approximation.
    
    Args:
        decoder: SIREN model
        contour_points_2d: Reference contour points
        num_random: Number of random points to test
        device: Device to run on
        outdir: Output directory
    
    Returns:
        Dictionary of consistency metrics
    """
    # Generate random points
    random_points = np.random.uniform(-1, 1, size=(num_random, 2))
    
    # Compute SDF values
    points_tensor = torch.from_numpy(random_points).float().to(device)
    with torch.no_grad():
        sdf_values = decoder(points_tensor).cpu().numpy().flatten()
    
    # Approximate inside/outside using convex hull
    try:
        if len(contour_points_2d) >= 3:
            hull = ConvexHull(contour_points_2d)
            hull_points = contour_points_2d[hull.vertices]
            
            hull_path = Path(hull_points)
            inside_hull = hull_path.contains_points(random_points)
            
            # Compare signs (SDF convention: negative = inside)
            pred_inside = sdf_values < 0
            agreement = (pred_inside == inside_hull).mean()
            
            # Analyze disagreements
            disagree_mask = (pred_inside != inside_hull)
            disagree_points = random_points[disagree_mask]
            
            if len(disagree_points) > 0:
                tree = KDTree(contour_points_2d)
                dist_to_contour, _ = tree.query(disagree_points)
                mean_dist_disagree = dist_to_contour.mean()
            else:
                mean_dist_disagree = 0.0
            
            results = {
                'agreement_with_hull': float(agreement),
                'points_tested': num_random,
                'method': 'convex_hull_approximation',
                'mean_distance_of_disagreements': float(mean_dist_disagree),
                'disagreement_rate': float(disagree_mask.mean()),
            }
            
            if outdir:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Predicted inside/outside
                ax = axes[0]
                colors = np.where(pred_inside, 'blue', 'red')
                ax.scatter(random_points[::10, 0], random_points[::10, 1],
                          c=colors[::10], s=1, alpha=0.3)
                ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1],
                          c='green', s=5, alpha=0.8, label='Contour')
                ax.set_title('Predicted Inside (blue) / Outside (red)')
                ax.set_xlabel('u')
                ax.set_ylabel('v')
                ax.set_aspect('equal')
                
                # Plot 2: Disagreement map
                ax = axes[1]
                colors = np.where(disagree_mask, 'yellow', 'black')
                ax.scatter(random_points[::10, 0], random_points[::10, 1],
                          c=colors[::10], s=1, alpha=0.3)
                ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1],
                          c='green', s=5, alpha=0.8)
                ax.set_title(f'Disagreements (yellow) - {disagree_mask.mean()*100:.2f}%')
                ax.set_xlabel('u')
                ax.set_ylabel('v')
                ax.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, 'sign_consistency.png'), dpi=150)
                plt.close()
            
            return results
        else:
            return {"error": "Need at least 3 contour points for convex hull"}
            
    except Exception as e:
        return {"error": f"Could not compute sign consistency: {str(e)}"}


# ============================================================================
# 3D Analysis Functions
# ============================================================================

def save_3d_slices(sdf_grid, coords, output_path):
    """
    Save center slices of 3D SDF grid.
    
    Args:
        sdf_grid: 3D array [N, N, N]
        coords: Coordinate array
        output_path: Output file path
    """
    mid = sdf_grid.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # YZ slice at x=mid
    im0 = axes[0].imshow(sdf_grid[mid, :, :].T, origin='lower',
                          cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                          extent=[coords[0], coords[-1], coords[0], coords[-1]])
    axes[0].set_title(f'YZ slice at x={coords[mid]:.3f}')
    axes[0].set_xlabel('y')
    axes[0].set_ylabel('z')
    plt.colorbar(im0, ax=axes[0])
    
    # XZ slice at y=mid
    im1 = axes[1].imshow(sdf_grid[:, mid, :].T, origin='lower',
                          cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                          extent=[coords[0], coords[-1], coords[0], coords[-1]])
    axes[1].set_title(f'XZ slice at y={coords[mid]:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im1, ax=axes[1])
    
    # XY slice at z=mid
    im2 = axes[2].imshow(sdf_grid[:, :, mid].T, origin='lower',
                          cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                          extent=[coords[0], coords[-1], coords[0], coords[-1]])
    axes[2].set_title(f'XY slice at z={coords[mid]:.3f}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Plotting Helpers
# ============================================================================

def save_hist(data, title, xlabel, output_path, bins=80):
    """
    Save histogram of data.
    
    Args:
        data: Array of values
        title: Plot title
        xlabel: X-axis label
        output_path: Output file path
        bins: Number of histogram bins
    """
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(data, bins=bins, density=False, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def diagnose_orientation(model, device):
    """Test model response at key points to determine orientation."""
    test_points = np.array([
        [0.0, 0.0],  # origin
        [0.1, 0.0],  # positive u
        [-0.1, 0.0], # negative u
        [0.0, 0.1],  # positive v
        [0.0, -0.1], # negative v
    ], dtype=np.float32)
    
    test_tensor = torch.from_numpy(test_points).to(device)
    with torch.no_grad():
        values = model(test_tensor).cpu().numpy().flatten()
    
    print("\n=== Model Orientation Test ===")
    print(f"(0.0, 0.0) -> {values[0]:.4f}")
    print(f"(0.1, 0.0) -> {values[1]:.4f}")
    print(f"(-0.1,0.0) -> {values[2]:.4f}")
    print(f"(0.0, 0.1) -> {values[3]:.4f}")
    print(f"(0.0,-0.1) -> {values[4]:.4f}")
    print("=============================\n")
    
    return values

# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main entry point for the verification script."""
    parser = argparse.ArgumentParser(
        description="Verify SIREN model (2D or 3D) properties with comprehensive checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("model", help="Checkpoint .pt/.pth file (TorchScript or state_dict)")
    
    # Model configuration
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3],
                        help="Dimension of model (2 for 2D SDF, 3 for 3D SDF)")
    parser.add_argument("--span", type=float, default=1.0,
                        help="Evaluate within [-span, span]^dim")
    
    # Sampling parameters
    parser.add_argument("--grid", type=int, default=256,
                        help="Grid size N for stats (N^dim queries)")
    parser.add_argument("--max-batch", type=int, default=64**3,
                        help="Batch size for grid evaluation")
    parser.add_argument("--rand", type=int, default=200000,
                        help="Number of random samples for field stats")
    parser.add_argument("--grad", type=int, default=50000,
                        help="Number of points for Eikonal check (autograd)")
    
    # Input files
    parser.add_argument("--contour-file", type=str, default=None,
                        help="CSV file with contour points (u,v) for 2D accuracy check")
    parser.add_argument("--on-surface-points", type=str, default=None,
                        help="PLY file with on-surface points for overlay visualization")
    parser.add_argument("--all-points", type=str, default=None,
                        help="PLY file with all training points for overlay visualization")
    
    # 3D mesh options
    parser.add_argument("--mesh", type=str, default=None,
                        help="Optional watertight mesh (only for 3D)")
    parser.add_argument("--surf", type=int, default=50000,
                        help="Surface points to sample if mesh is provided")
    
    # SIREN parameters
    parser.add_argument("--w0", type=float, default=30.0,
                        help="Omega_0 for hidden layers")
    parser.add_argument("--w0-initial", type=float, default=30.0,
                        help="Omega_0 for first layer")
    
    # Output
    parser.add_argument("--outdir", default="model_verification_report",
                        help="Output directory name (will be placed under model_verification/)")
    
    args = parser.parse_args()
    
    # Set up output directory
    parent_dir = "model_verification"
    os.makedirs(parent_dir, exist_ok=True)
    
    if not os.path.isabs(args.outdir):
        args.outdir = os.path.join(parent_dir, args.outdir)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"SIREN MODEL VERIFICATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dimension: {args.dim}D")
    print(f"Device: {device}")
    print(f"Output directory: {args.outdir}")

    
    # 1) Load model
    print("\n[1] Loading model...")
    model = load_siren_auto(args.model, device, w0=args.w0, w0_initial=args.w0_initial, dim=args.dim)
    model.eval()

    # When loading points for 2D model, only use first 2 coordinates
    if args.dim == 2:
        pts, colors = load_ply_points(args.on_surface_points)
        if pts is not None:
            # Drop z coordinate (third column)
            pts_2d = pts[:, :2].copy()  # Take only u, v
            print(f"  Loaded {len(pts_2d)} points (dropped z coordinate)")
            
            # Now evaluate with 2D points
            pt_tensor = torch.from_numpy(pts_2d[:5]).float().to(device)
            with torch.no_grad():
                pt_values = model(pt_tensor).cpu().numpy().flatten()
            print(f"SDF values at those points: {pt_values}")

    # After loading model
    print("\n=== Testing Model Response ===")
    test_points = np.array([
        [0.1, 0.0],   # positive u
        [0.0, 0.1],   # positive v
        [0.1, 0.1],   # both positive
    ], dtype=np.float32)

    test_tensor = torch.from_numpy(test_points).to(device)
    with torch.no_grad():
        values = model(test_tensor).cpu().numpy().flatten()

    print(f"(0.1, 0.0) -> {values[0]:.4f}")
    print(f"(0.0, 0.1) -> {values[1]:.4f}")
    print(f"(0.1, 0.1) -> {values[2]:.4f}")

    # Now load your points and check their actual values
    if args.on_surface_points:
        pts, colors = load_ply_points(args.on_surface_points)
        if pts is not None:
            # CRITICAL: For 2D model, only use first 2 coordinates
            pts_2d = pts[:, :2].copy()
            print(f"\nSample point from file (2D): {pts_2d[0]}")
            
            # Evaluate at those points
            pt_tensor = torch.from_numpy(pts_2d[:5]).float().to(device)
            with torch.no_grad():
                pt_values = model(pt_tensor).cpu().numpy().flatten()
            print(f"SDF values at those points: {pt_values}")
            
            # Also check if signs match expectation
            print(f"\nSign check - points should be near zero (on surface):")
            print(f"  Mean |SDF|: {np.abs(pt_values).mean():.6f}")
    
    # Initialize report
    report = {
        "dimension": args.dim,
        "model_path": args.model,
        "device": device
    }
    
    # 2) Basic random stats
    print("\n[2] Computing basic statistics on random points...")
    report["basic_stats"] = check_basic_stats(
        model, n_points=args.rand, span=args.span, device=device, dim=args.dim
    )
    print(f"    Min: {report['basic_stats']['min']:.4f}")
    print(f"    Max: {report['basic_stats']['max']:.4f}")
    print(f"    Mean: {report['basic_stats']['mean']:.4f}")
    print(f"    Fraction negative: {report['basic_stats']['frac_negative']:.4f}")
    
    # 3) Grid sampling
    print(f"\n[3] Sampling on {args.grid}^{args.dim} grid...")
    
    if args.dim == 2:
        # 2D grid sampling
        sdf_grid, u_coords, v_coords = sample_sdf_grid_2d(
            model, N=args.grid, max_batch=args.max_batch, device=device, span=args.span
        )
        grid_values = sdf_grid.reshape(-1)
        
        # Save basic 2D visualization
        save_2d_sdf_plot(
            sdf_grid, u_coords, v_coords,
            f"2D SDF Grid (N={args.grid})",
            os.path.join(args.outdir, "sdf_2d_grid.png")
        )
        
        # Enhanced contour visualization
        print("\n[4] Creating enhanced contour visualizations...")
        visualize_sdf_contours_2d(
            sdf_grid, u_coords, v_coords,
            os.path.join(args.outdir, "sdf_contour_analysis.png"),
            levels=20, highlight_zero=True
        )
        
        # Extract zero contour
        extract_and_save_zero_contour(
            sdf_grid, u_coords, v_coords,
            os.path.join(args.outdir, "zero_contour.png")
        )
        
        # Point cloud overlay analysis
        if args.on_surface_points or args.all_points:
            print("\n[5] Creating point cloud overlay visualizations...")
            points_list = []
            
            if args.on_surface_points:
                pts_on_surface, colors_on_surface = load_ply_points(args.on_surface_points)
                points_list.append((pts_on_surface, colors_on_surface, "On-surface Points"))
            
            if args.all_points:
                pts_all, colors_all = load_ply_points(args.all_points)
                points_list.append((pts_all, colors_all, "All Training Points"))
            
            if points_list:
                create_overlay_analysis_2d(sdf_grid, u_coords, v_coords, points_list, args.outdir)
        
        # Grid statistics
        report["grid_stats"] = {
            "dim": 2,
            "grid_N": int(args.grid),
            "min": float(grid_values.min()),
            "max": float(grid_values.max()),
            "mean": float(grid_values.mean()),
            "median": float(np.median(grid_values)),
            "std": float(grid_values.std()),
            "frac_negative": float((grid_values < 0).mean()),
            "frac_positive": float((grid_values > 0).mean()),
        }
        
        save_hist(
            grid_values,
            f"Field values on grid N={args.grid} (2D)",
            "f(x)",
            os.path.join(args.outdir, "hist_grid_values.png")
        )
        
        # Compare with original contour if provided
        if args.contour_file:
            print("\n[6] Comparing with original contour points...")
            contour_points = np.loadtxt(args.contour_file, delimiter=",", skiprows=1)
            if contour_points.ndim == 1:
                contour_points = contour_points.reshape(-1, 2)
            
            contour_metrics = compare_with_original_contour(
                sdf_grid, u_coords, v_coords, contour_points,
                os.path.join(args.outdir, "contour_comparison.png"),
                model=model, device=device
            )
            
            if contour_metrics:
                report["contour_comparison"] = contour_metrics
                print(f"    Mean distance to contour: {contour_metrics['mean_distance']:.6f}")
        
        # Eikonal error map
        print("\n[7] Computing Eikonal error map...")
        report["eikonal_stats"] = eikonal_error_map(
            model, resolution=100, device=device, outdir=args.outdir
        )
        print(f"    Mean Eikonal error: {report['eikonal_stats']['mean_error']:.4f}")
        
        # Gradient field visualization
        print("\n[8] Visualizing gradient field...")
        report["gradient_stats"] = visualize_gradient_field(
            model, resolution=30, device=device, outdir=args.outdir
        )
        
        # SDF profiles
        print("\n[9] Plotting SDF profiles...")
        directions = [
            ([-0.8, -0.8], [0.8, 0.8]),    # diagonal
            ([-0.8, 0.0], [0.8, 0.0]),      # horizontal
            ([0.0, -0.8], [0.0, 0.8]),      # vertical
            ([-0.8, 0.8], [0.8, -0.8]),     # anti-diagonal
        ]
        for i, (start, end) in enumerate(directions):
            prefix = f"profile_{i}"
            results = plot_sdf_profile(
                model, start, end, device=device,
                outdir=args.outdir, prefix=prefix
            )
            report.update(results)
        
        # Sign consistency check
        if args.contour_file:
            print("\n[10] Checking sign consistency...")
            report["sign_consistency"] = check_sign_consistency_2d(
                model, contour_points, num_random=args.rand//2,
                device=device, outdir=args.outdir
            )
            if "agreement_with_hull" in report["sign_consistency"]:
                print(f"    Agreement with convex hull: {report['sign_consistency']['agreement_with_hull']:.4f}")
        
        # Gradient norm statistics
        print("\n[11] Computing gradient norm statistics (Eikonal check)...")
        grad_stats, norms = grad_norm_stats(
            model, n=args.grad, span=args.span, device=device, dim=args.dim
        )
        report["eikonal_grad_norm"] = grad_stats
        
        if norms is not None:
            save_hist(
                norms,
                f"Gradient norms ||∇f|| (2D)",
                "||∇f||",
                os.path.join(args.outdir, "hist_grad_norms.png")
            )
            if "mean_abs_error_from_1" in grad_stats:
                print(f"    Mean |∇f| - 1: {grad_stats['mean_abs_error_from_1']:.4f}")
                print(f"    Fraction with |∇f| within 0.1 of 1: {grad_stats['frac_near_1_01']:.4f}")
    
    else:
        # 3D mode
        sdf_grid, x_coords, y_coords, z_coords = sample_sdf_grid_3d(
            model, N=min(args.grid, 128), max_batch=args.max_batch, device=device, span=args.span
        )
        grid_values = sdf_grid.reshape(-1)
        
        # Save center slices
        save_3d_slices(sdf_grid, x_coords, os.path.join(args.outdir, "sdf_3d_slices.png"))
        
        # Grid statistics
        report["grid_stats"] = {
            "dim": 3,
            "grid_N": int(args.grid),
            "min": float(grid_values.min()),
            "max": float(grid_values.max()),
            "mean": float(grid_values.mean()),
            "median": float(np.median(grid_values)),
            "std": float(grid_values.std()),
            "frac_negative": float((grid_values < 0).mean()),
            "frac_positive": float((grid_values > 0).mean()),
        }
        
        save_hist(
            grid_values,
            f"Field values on grid N={args.grid} (3D)",
            "f(x)",
            os.path.join(args.outdir, "hist_grid_values.png")
        )
        
        # Gradient norm statistics for 3D
        print("\n[4] Computing gradient norm statistics (Eikonal check)...")
        grad_stats, norms = grad_norm_stats(
            model, n=args.grad, span=args.span, device=device, dim=3
        )
        report["eikonal_grad_norm"] = grad_stats
        
        if norms is not None:
            save_hist(
                norms,
                "Gradient norms ||∇f|| (3D)",
                "||∇f||",
                os.path.join(args.outdir, "hist_grad_norms.png")
            )
            if "mean_abs_error_from_1" in grad_stats:
                print(f"    Mean |∇f| - 1: {grad_stats['mean_abs_error_from_1']:.4f}")
        
        # Mesh-based checks for 3D
        if args.mesh is not None and HAVE_TRIMESH:
            print("\n[5] Evaluating on mesh surface...")
            try:
                mesh = trimesh.load(args.mesh, force='mesh')
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.dump()))
                
                # Sample points on surface
                surface_points = mesh.sample(args.surf).astype(np.float32)
                surface_values = batched_eval(model, surface_points, device)
                
                report["mesh_surface_stats"] = {
                    "n_surface": int(surface_values.size),
                    "abs_mean_on_surface": float(np.abs(surface_values).mean()),
                    "abs_median_on_surface": float(np.median(np.abs(surface_values))),
                    "median_value_on_surface": float(np.median(surface_values)),
                    "mean_value_on_surface": float(surface_values.mean()),
                    "std_on_surface": float(surface_values.std()),
                }
                
                save_hist(
                    surface_values,
                    "Values on mesh surface (3D)",
                    "f(x) @ surface",
                    os.path.join(args.outdir, "hist_surface_values.png")
                )
                print(f"    Mean |SDF| on surface: {report['mesh_surface_stats']['abs_mean_on_surface']:.6f}")
                
            except Exception as e:
                report["mesh_surface_stats"] = {"error": str(e)}
                print(f"    Error: {e}")
    
    # Compute signedness score (0=all same sign, 1=balanced)
    frac_neg = report["grid_stats"]["frac_negative"]
    frac_pos = report["grid_stats"]["frac_positive"]
    signedness = float(2.0 * min(frac_neg, frac_pos))
    report["signedness_score_grid"] = signedness
    
    # Save JSON report
    report_path = os.path.join(args.outdir, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"Images saved to: {args.outdir}")
    print(f"\nKey metrics:")
    print(f"  Signedness score: {signedness:.3f} (0=all same sign, 1=balanced)")
    
    if args.dim == 2:
        if 'contour_comparison' in report:
            cc = report['contour_comparison']
            print(f"  Contour comparison RMSE: {cc.get('rmse', 'N/A')}")
        if 'eikonal_grad_norm' in report and 'mean_abs_error_from_1' in report['eikonal_grad_norm']:
            print(f"  Eikonal error (|∇f|-1): {report['eikonal_grad_norm']['mean_abs_error_from_1']:.4f}")
    else:
        if 'eikonal_grad_norm' in report and 'mean_abs_error_from_1' in report['eikonal_grad_norm']:
            print(f"  Eikonal error (|∇f|-1): {report['eikonal_grad_norm']['mean_abs_error_from_1']:.4f}")
        if 'mesh_surface_stats' in report and 'abs_mean_on_surface' in report['mesh_surface_stats']:
            print(f"  Mean |SDF| on surface: {report['mesh_surface_stats']['abs_mean_on_surface']:.6f}")


if __name__ == "__main__":
    main()