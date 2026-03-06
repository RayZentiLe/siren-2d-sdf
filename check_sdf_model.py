#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
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


# Optional mesh checks
try:
    import trimesh
    HAVE_TRIMESH = True
except Exception:
    HAVE_TRIMESH = False

# ---------------- SIREN (torchmeta-free, inference only) ----------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.omega_0 = omega_0
        self.is_first = is_first
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 w0=30.0, w0_initial=30.0, outermost_linear=True, use_bias=True):
        super().__init__()
        layers = [SineLayer(dim_in, dim_hidden, bias=use_bias, is_first=True,  omega_0=w0_initial)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(dim_hidden, dim_hidden, bias=use_bias, is_first=False, omega_0=w0))
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(dim_hidden, dim_out, bias=use_bias) if outermost_linear else None
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x) if self.final_linear is not None else x
# ------------------------------------------------------------------------

# -------- robust loader by SHAPES (works even if param names differ) ----
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict","model_state_dict","ema_state_dict","model"):
            if k in obj:
                inner = obj[k]
                if isinstance(inner, (dict, nn.Module)):
                    return inner
    return obj

def _find_linear_params(sd):
    out = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(".weight"):
            base = k[:-7]
            bkey = base + ".bias"
            if bkey in sd and isinstance(sd[bkey], torch.Tensor) and sd[bkey].ndim == 1:
                out.append((base, v, sd[bkey]))
    return out

def _key_index_hint(key):
    nums = re.findall(r"(?:^|\.)(\d+)(?=\.|$)", key)
    return int(nums[-1]) if nums else 1_000_000

def build_siren_from_sd(sd, w0=30.0, w0_initial=30.0, prefer_in=None):
    triples = _find_linear_params(sd)
    if not triples:
        raise RuntimeError("No linear (weight,bias) tensors found in checkpoint.")
    
    # If prefer_in not specified, find the most common input dimension
    if prefer_in is None:
        in_dims = [int(w.shape[1]) for _, w, _ in triples]
        prefer_in = max(set(in_dims), key=in_dims.count)
    
    first = [(b,w,bias) for (b,w,bias) in triples if int(w.shape[1]) == prefer_in]
    if not first:
        min_in = min(int(w.shape[1]) for _, w, _ in triples)
        first = [(b,w,bias) for (b,w,bias) in triples if int(w.shape[1]) == min_in]
    first.sort(key=lambda t: _key_index_hint(t[0]))
    b0, w0_t, b0_t = first[0]
    H, dim_in = int(w0_t.shape[0]), int(w0_t.shape[1])

    hh = [(b,w,bias) for (b,w,bias) in triples if int(w.shape[0]) == H and int(w.shape[1]) == H]
    hh.sort(key=lambda t: _key_index_hint(t[0]))
    finals = [(b,w,bias) for (b,w,bias) in triples if int(w.shape[1]) == H and int(w.shape[0]) in (1,2,3,4)]
    finals.sort(key=lambda t: (int(t[1].shape[0]), _key_index_hint(t[0])))
    use_final = len(finals) > 0
    dim_out = int(finals[0][1].shape[0]) if use_final else 1

    num_layers = 1 + len(hh)
    model = SirenNet(dim_in, H, dim_out, num_layers, w0=float(w0), w0_initial=float(w0_initial),
                     outermost_linear=use_final, use_bias=True)

    with torch.no_grad():
        model.net[0].linear.weight.copy_(w0_t); model.net[0].linear.bias.copy_(b0_t)
        for i, (_b, w, b) in enumerate(hh, start=1):
            model.net[i].linear.weight.copy_(w); model.net[i].linear.bias.copy_(b)
        if use_final:
            _bf, wf, bf = finals[0]
            model.final_linear.weight.copy_(wf); model.final_linear.bias.copy_(bf)
    return model

def load_siren_auto(path, device, w0=30.0, w0_initial=30.0, dim=None):
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval(); print("[loader] TorchScript module loaded"); return m
    except Exception:
        pass
    raw = torch.load(path, map_location="cpu")
    sd = _extract_state_dict(raw)
    if isinstance(sd, nn.Module):
        print("[loader] checkpoint contained a Module; using as-is")
        return sd.to(device).eval()
    model = build_siren_from_sd(sd, w0=w0, w0_initial=w0_initial, prefer_in=dim)
    print("[loader] SIREN inferred and weights copied")
    return model.to(device).eval()
# ------------------------------------------------------------------------

# ----------------- Grid sampling functions -------------------

def sample_sdf_grid_2d(decoder, N=256, max_batch=64**2, device="cuda", span=1.0):
    """
    Sample 2D SDF on a grid from [-span, span]^2
    
    Args:
        span: The grid spans from -span to span in both dimensions
              For normalized coordinates, span should be 1.0
    """
    decoder.eval()
    
    # Create 2D grid from -span to span
    x = np.linspace(-span, span, N)
    y = np.linspace(-span, span, N)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    print(f"  Grid range: [{x.min():.2f}, {x.max():.2f}] x [{y.min():.2f}, {y.max():.2f}]")
    
    # Evaluate in batches
    vals = np.zeros(len(grid_points), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(grid_points), max_batch):
            batch = grid_points[i:i+max_batch]
            batch_tensor = torch.from_numpy(batch).to(device)
            out = decoder(batch_tensor).view(-1).cpu().numpy()
            vals[i:i+max_batch] = out
    
    return vals.reshape(N, N), x, y


def sample_sdf_grid_3d(decoder, N=128, max_batch=64**3, device="cuda"):
    """Sample 3D SDF on a grid from [-1, 1]^3"""
    decoder.eval()
    voxel_origin = [-1.0, -1.0, -1.0]
    voxel_size   = 2.0 / (N - 1)

    idx = torch.arange(0, N**3, dtype=torch.long)
    samples = torch.zeros(N**3, 3, dtype=torch.float32)

    samples[:, 2] = idx % N
    samples[:, 1] = (idx // N) % N
    samples[:, 0] = (idx // (N*N)) % N
    samples[:, 0] = samples[:, 0] * voxel_size + voxel_origin[2]
    samples[:, 1] = samples[:, 1] * voxel_size + voxel_origin[1]
    samples[:, 2] = samples[:, 2] * voxel_size + voxel_origin[0]

    vals = torch.empty(N**3, dtype=torch.float32)
    head = 0
    with torch.no_grad():
        while head < N**3:
            tail = min(head + max_batch, N**3)
            v = decoder(samples[head:tail].to(device)).view(-1).detach().cpu()
            vals[head:tail] = v
            head = tail

    V = vals.view(N, N, N).cpu().numpy()
    xC = (voxel_origin[2] + np.arange(N) * voxel_size).astype(np.float32)
    yC = (voxel_origin[1] + np.arange(N) * voxel_size).astype(np.float32)
    zC = (voxel_origin[0] + np.arange(N) * voxel_size).astype(np.float32)
    return V, xC, yC, zC

# ------------------------ Random & surface eval -------------------------
def batched_eval(decoder, pts_np, device, batch=131072):
    out = np.empty((pts_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        i = 0
        while i < pts_np.shape[0]:
            j = min(i + batch, pts_np.shape[0])
            out[i:j] = decoder(torch.from_numpy(pts_np[i:j]).to(device)).view(-1).cpu().numpy()
            i = j
    return out

def random_points(n, span, dim=3):
    """Generate random points in [-span, span]^dim"""
    return np.random.uniform(-span, span, size=(n, dim)).astype(np.float32)

# ---------------------- Load PLY helper function ----------------------

def load_ply_points(ply_path):
    """
    Load 3D points (with optional RGB) from PLY file.
    Returns: points [N, 3], colors [N, 3] or None
    """
    if not os.path.exists(ply_path):
        print(f"  Warning: PLY file not found: {ply_path}")
        return None, None
    
    points = []
    colors = []
    has_colors = False
    
    with open(ply_path, 'r') as f:
        # Parse header
        header_done = False
        num_verts = 0
        has_r = has_g = has_b = False
        
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                num_verts = int(line.split()[-1])
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
            if i >= num_verts:
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

# ---------------------- Overlay visualization helper ----------------------

def create_overlay_analysis_2d(sdf_grid, x_grid, y_grid, points_list, output_dir):
    """
    Create analysis images with overlaid point clouds.
    Generates 3 files (heatmap, contour, zero-level) x 2 subplots (one per point cloud).
    
    Args:
        sdf_grid: [N, N] SDF values
        x_grid, y_grid: coordinate arrays
        points_list: list of (points [M, 3], colors [M, 3], label_str) tuples
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    
    # Extract 2D coordinates and colors for overlay
    points_2d_list = []
    for pts, colors, label in points_list:
        if pts is None:
            points_2d_list.append((None, None, label))
        else:
            points_2d = pts[:, :2]  # Extract u, v
            points_2d_list.append((points_2d, colors, label))
    
    # File 1: Heatmap + overlays
    fig, axes = plt.subplots(1, len(points_2d_list), figsize=(8 * len(points_2d_list), 7))
    if len(points_2d_list) == 1:
        axes = [axes]
    
    for ax_idx, (pts_2d, colors_2d, label) in enumerate(points_2d_list):
        ax = axes[ax_idx]
        im = ax.imshow(sdf_grid.T, origin='lower', extent=extent,
                       cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        
        if pts_2d is not None:
            # Overlay points with original colors
            if colors_2d is not None:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=colors_2d/255.0, s=3, alpha=0.8)
            else:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=3, alpha=0.5)
        
        ax.set_title(f'SDF Heatmap - {label}')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='SDF Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sdf_overlay_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap overlay to: sdf_overlay_heatmap.png")
    
    # File 2: Contour lines + overlays
    fig, axes = plt.subplots(1, len(points_2d_list), figsize=(8 * len(points_2d_list), 7))
    if len(points_2d_list) == 1:
        axes = [axes]
    
    for ax_idx, (pts_2d, colors_2d, label) in enumerate(points_2d_list):
        ax = axes[ax_idx]
        contour = ax.contour(x_grid, y_grid, sdf_grid.T, levels=20, cmap='viridis', linewidths=1)
        
        if pts_2d is not None:
            # Overlay points with original colors
            if colors_2d is not None:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=colors_2d/255.0, s=3, alpha=0.8)
            else:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=3, alpha=0.5)
        
        ax.set_title(f'Contour Lines - {label}')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        plt.colorbar(contour, ax=ax, label='SDF Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sdf_overlay_contours.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved contour overlay to: sdf_overlay_contours.png")
    
    # File 3: Zero level set + overlays
    fig, axes = plt.subplots(1, len(points_2d_list), figsize=(8 * len(points_2d_list), 7))
    if len(points_2d_list) == 1:
        axes = [axes]
    
    for ax_idx, (pts_2d, colors_2d, label) in enumerate(points_2d_list):
        ax = axes[ax_idx]
        zero_contour = ax.contour(x_grid, y_grid, sdf_grid.T, levels=[0], colors='red', linewidths=2)
        
        if pts_2d is not None:
            # Overlay points with original colors
            if colors_2d is not None:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=colors_2d/255.0, s=3, alpha=0.8)
            else:
                ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=3, alpha=0.5)
        
        ax.set_title(f'Zero Level Set - {label}')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sdf_overlay_zero_level.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved zero level set overlay to: sdf_overlay_zero_level.png")

# ------------------------ Contour Visualization Functions ---------------------------

def visualize_sdf_contours_2d(sdf_grid, x_grid, y_grid, output_path, 
                               levels=20, highlight_zero=True):
    """
    Create comprehensive contour visualization of 2D SDF.
    
    Args:
        sdf_grid: [N, N] array of SDF values
        x_grid, y_grid: 1D arrays of coordinates
        output_path: Path to save the figure
        levels: Number of contour levels or specific levels list
        highlight_zero: Whether to highlight the zero contour
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    
    # Plot 1: SDF heatmap
    ax = axes[0, 0]
    im = ax.imshow(sdf_grid.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title('SDF Heatmap')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    # Plot 2: Contour lines only
    ax = axes[0, 1]
    contour = ax.contour(x_grid, y_grid, sdf_grid.T, levels=levels, 
                         cmap='RdBu_r', linewidths=1)
    ax.set_title(f'SDF Contour Lines ({levels} levels)')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, label='SDF Value')
    
    # Plot 3: Zero level set (main contour)
    ax = axes[0, 2]
    if highlight_zero:
        zero_contour = ax.contour(x_grid, y_grid, sdf_grid.T, levels=[0], 
                                  colors='red', linewidths=2)
        ax.set_title('Zero Level Set (SDF=0)')
        # Add contour labels
        if len(zero_contour.allsegs[0]) > 0:
            ax.clabel(zero_contour, inline=True, fontsize=10, fmt='%.2f')
    else:
        ax.text(0.5, 0.5, 'No zero contour found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Zero Level Set')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined heatmap + zero contour
    ax = axes[1, 0]
    im = ax.imshow(sdf_grid.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5, alpha=0.8)
    if highlight_zero:
        ax.contour(x_grid, y_grid, sdf_grid.T, levels=[0], colors='black', linewidths=2)
    ax.set_title('SDF Heatmap + Zero Contour')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    # Plot 5: Multiple important contours
    ax = axes[1, 1]
    important_levels = [-0.2, -0.1, 0, 0.1, 0.2]
    contour = ax.contour(x_grid, y_grid, sdf_grid.T, levels=important_levels, 
                        cmap='viridis', linewidths=1.5)
    ax.set_title(f'Important Contours: {important_levels}')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_aspect('equal')
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Contour density (how close to zero)
    ax = axes[1, 2]
    # Compute distance to zero contour
    from scipy.ndimage import distance_transform_edt
    binary = sdf_grid < 0
    dist_to_contour = distance_transform_edt(~binary) + distance_transform_edt(binary)
    dist_to_contour = np.where(binary, -dist_to_contour, dist_to_contour)
    
    im = ax.imshow(dist_to_contour.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.contour(x_grid, y_grid, sdf_grid.T, levels=[0], colors='black', linewidths=1)
    ax.set_title('Distance to Zero Contour')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.colorbar(im, ax=ax, label='Signed Distance')
    
    plt.suptitle('2D SDF Contour Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved contour visualization to: {output_path}")


def extract_and_save_zero_contour(sdf_grid, x_grid, y_grid, output_path):
    """
    Extract the zero level set and save as a separate plot.
    Also return the contour points for further analysis.
    """
    # Extract zero level set using marching squares
    contours = measure.find_contours(sdf_grid.T, level=0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert pixel coordinates to world coordinates
    world_contours = []
    for i, contour in enumerate(contours):
        u_world = contour[:, 1] * (2.0 / (sdf_grid.shape[0] - 1)) - 1.0
        v_world = contour[:, 0] * (2.0 / (sdf_grid.shape[1] - 1)) - 1.0
        world_contour = np.stack([u_world, v_world], axis=1)
        world_contours.append(world_contour)
        
        # Plot each contour with a different color
        ax.plot(u_world, v_world, linewidth=2, label=f'Contour {i+1}' if i < 5 else '')
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Zero Level Set Contours (found {len(contours)} fragments)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if len(contours) <= 5:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved zero contour to: {output_path}")
    
    return world_contours


def compare_with_original_contour(sdf_grid, x_grid, y_grid, original_contour_points, 
                                    output_path, model=None, device='cuda'):
    """
    Compare the model's zero contour with original contour points.
    """
    # Extract zero contour from SDF grid
    contours = measure.find_contours(sdf_grid.T, level=0)
    
    if len(contours) == 0:
        print("  Warning: No zero contour found in SDF grid")
        return
    
    # Convert to world coordinates
    world_contours = []
    for contour in contours:
        u_world = contour[:, 1] * (2.0 / (sdf_grid.shape[0] - 1)) - 1.0
        v_world = contour[:, 0] * (2.0 / (sdf_grid.shape[1] - 1)) - 1.0
        world_contours.append(np.stack([u_world, v_world], axis=1))
    
    all_contour_points = np.vstack(world_contours)
    
    # Compute distances from original points to nearest point on extracted contour
    tree = KDTree(all_contour_points)
    distances, _ = tree.query(original_contour_points)
    
    # Get SDF values at original points if model provided
    if model is not None:
        contour_tensor = torch.from_numpy(original_contour_points).float().to(device)
        with torch.no_grad():
            sdf_at_original = model(contour_tensor).cpu().numpy().flatten()
    else:
        sdf_at_original = None
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Original vs extracted contour
    ax = axes[0]
    ax.scatter(original_contour_points[:, 0], original_contour_points[:, 1], 
               c='green', s=2, alpha=0.5, label='Original')
    for wc in world_contours:
        ax.plot(wc[:, 0], wc[:, 1], 'r-', linewidth=1, label='Extracted' if wc is world_contours[0] else '')
    ax.set_title(f'Contour Comparison (mean dist={distances.mean():.4f})')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Distance error histogram
    ax = axes[1]
    ax.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.4f}')
    ax.axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.4f}')
    ax.set_xlabel('Distance to extracted contour')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SDF values at original points (if available)
    ax = axes[2]
    if sdf_at_original is not None:
        ax.scatter(original_contour_points[:, 0], original_contour_points[:, 1],
                  c=sdf_at_original, cmap='RdBu_r', s=5, alpha=0.8,
                  vmin=-0.1, vmax=0.1)
        ax.set_title(f'SDF at Original Points (mean |val|={np.abs(sdf_at_original).mean():.4f})')
        plt.colorbar(ax.collections[0], ax=ax, label='SDF Value')
    else:
        ax.text(0.5, 0.5, 'No model provided', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SDF Values (unavailable)')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
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

# ------------------------ Existing Verification Functions ---------------------------

def check_basic_stats(decoder, n_points=200000, span=1.0, device="cuda", dim=3):
    """Basic statistics of SDF values on random points"""
    pts = random_points(n_points, span, dim)
    vals = batched_eval(decoder, pts, device)
    
    return {
        "n": int(vals.size),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "frac_negative": float((vals < 0).mean()),
        "frac_positive": float((vals > 0).mean()),
        "frac_near_zero_1e_3": float((np.abs(vals) < 1e-3).mean()),
        "frac_near_zero_1e_2": float((np.abs(vals) < 1e-2).mean()),
    }

def check_contour_accuracy(decoder, contour_points_2d, device='cuda', resolution=512, outdir=None):
    """
    Check how well the model's zero level set matches original contour points.
    """
    decoder.eval()
    
    # Create grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Evaluate model on grid
    grid_tensor = torch.from_numpy(grid_points).float().to(device)
    with torch.no_grad():
        sdf_grid = decoder(grid_tensor).cpu().numpy().reshape(resolution, resolution)
    
    # Extract zero level set using marching squares
    contours = measure.find_contours(sdf_grid.T, level=0)
    
    results = {}
    
    if len(contours) == 0:
        results["error"] = "No zero level set found"
        if outdir:
            # Still save the SDF grid
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sdf_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                          cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax.set_title('SDF Grid (No contour found)')
            plt.colorbar(im, ax=ax)
            plt.savefig(os.path.join(outdir, 'sdf_grid.png'), dpi=150)
            plt.close()
        return results
    
    # Convert contour pixel coordinates to world coordinates
    world_contours = []
    for contour in contours:
        u_world = contour[:, 1] * (2.0 / (resolution - 1)) - 1.0
        v_world = contour[:, 0] * (2.0 / (resolution - 1)) - 1.0
        world_contours.append(np.stack([u_world, v_world], axis=1))
    
    # Compute distances from original points to nearest point on extracted contour
    all_contour_points = np.vstack(world_contours)
    tree = KDTree(all_contour_points)
    
    distances, _ = tree.query(contour_points_2d)
    
    # Also check if original points are near zero in the SDF
    contour_tensor = torch.from_numpy(contour_points_2d).float().to(device)
    with torch.no_grad():
        sdf_at_contour = decoder(contour_tensor).cpu().numpy().flatten()
    
    results = {
        'mean_distance_to_contour': float(distances.mean()),
        'median_distance_to_contour': float(np.median(distances)),
        'max_distance_to_contour': float(distances.max()),
        'std_distance_to_contour': float(distances.std()),
        'rmse_to_contour': float(np.sqrt((distances**2).mean())),
        'mean_abs_sdf_at_contour': float(np.abs(sdf_at_contour).mean()),
        'std_sdf_at_contour': float(np.std(sdf_at_contour)),
        'num_contour_fragments': len(contours),
    }
    
    # Save visualization if outdir provided
    if outdir:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: SDF grid with contours
        ax = axes[0]
        im = ax.imshow(sdf_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                      cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        for contour in world_contours:
            ax.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=1)
        ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1], 
                  c='green', s=2, alpha=0.5, label='Original')
        ax.set_title(f'SDF with Zero Level Set')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # Plot 2: Distance error map
        ax = axes[1]
        # Compute distance from each grid point to nearest original point
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        tree_orig = KDTree(contour_points_2d)
        dist_to_orig, _ = tree_orig.query(grid_pts)
        dist_map = dist_to_orig.reshape(resolution, resolution)
        
        im = ax.imshow(dist_map.T, origin='lower', extent=[-1, 1, -1, 1],
                      cmap='hot', vmin=0, vmax=0.1)
        ax.set_title('Distance to Original Contour')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        # Plot 3: SDF values at original points
        ax = axes[2]
        scatter = ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1],
                            c=sdf_at_contour, cmap='RdBu_r', s=5, alpha=0.8,
                            vmin=-0.1, vmax=0.1)
        ax.set_title(f'SDF at Original Points (mean |val|={results["mean_abs_sdf_at_contour"]:.4f})')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'contour_accuracy.png'), dpi=150)
        plt.close()
        
        # Also save just the SDF grid
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sdf_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                      cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.contour(sdf_grid.T, levels=[0], colors='black', linewidths=1, extent=[-1, 1, -1, 1])
        ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1], 
                  c='green', s=2, alpha=0.5)
        ax.set_title('SDF Grid with Zero Level Set')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(outdir, 'sdf_grid.png'), dpi=150)
        plt.close()
    
    return results

def eikonal_error_map(decoder, resolution=100, device='cuda', outdir=None):
    """
    Create heatmap of |∇f| - 1 error.
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
            batch_tensor = torch.from_numpy(batch).float().to(device).requires_grad_(True)
            
            sdf = decoder(batch_tensor).view(-1)
            
            # Compute gradients
            grads = []
            for j in range(sdf.shape[0]):
                grad = torch.autograd.grad(sdf[j], batch_tensor, retain_graph=True)[0][j]
                grads.append(grad.detach().cpu().numpy())
            
            grads = np.array(grads)
            grad_norms = np.linalg.norm(grads, axis=1)
            grad_norms_list.extend(grad_norms)
            errors.extend(np.abs(grad_norms - 1.0))
    
    error_map = np.array(errors).reshape(resolution, resolution)
    grad_norms_map = np.array(grad_norms_list).reshape(resolution, resolution)
    
    results = {
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
        ax.set_xlabel('u'); ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        # Gradient magnitude
        ax = axes[1]
        im = ax.imshow(grad_norms_map.T, origin='lower', extent=[-0.9, 0.9, -0.9, 0.9],
                       cmap='viridis', vmin=0, vmax=2)
        ax.set_title('Gradient Magnitude (should be 1)')
        ax.set_xlabel('u'); ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'eikonal_error.png'), dpi=150)
        plt.close()
    
    return results

def visualize_gradient_field(decoder, resolution=30, device='cuda', outdir=None):
    """
    Create quiver plot of SDF gradients.
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
            batch_tensor = torch.from_numpy(batch).float().to(device).requires_grad_(True)
            
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
        ax.set_xlabel('u'); ax.set_ylabel('v')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        
        # Gradient direction quiver
        ax = axes[1]
        ax.quiver(grid_points[:, 0], grid_points[:, 1], 
                  grad_dirs[:, 0], grad_dirs[:, 1], 
                  grad_norms, cmap='viridis', scale=20, width=0.003)
        ax.set_title('Gradient Directions')
        ax.set_xlabel('u'); ax.set_ylabel('v')
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
    """
    # Generate points along line
    t = np.linspace(0, 1, num_points)
    points = np.outer(1-t, line_start) + np.outer(t, line_end)
    
    # Evaluate SDF
    points_tensor = torch.from_numpy(points).float().to(device)
    with torch.no_grad():
        sdf_values = decoder(points_tensor).cpu().numpy().flatten()
    
    # Find zero crossings
    zero_crossings = []
    for i in range(len(sdf_values)-1):
        if sdf_values[i] * sdf_values[i+1] < 0:
            t_zero = t[i] + (t[i+1] - t[i]) * (0 - sdf_values[i]) / (sdf_values[i+1] - sdf_values[i])
            zero_crossings.append(t_zero)
    
    results = {
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
    
    return results

def check_sign_consistency_2d(decoder, contour_points_2d, num_random=100000, device='cuda', outdir=None):
    """
    Check that signs are consistent with a simple winding number heuristic.
    """
    # Generate random points
    random_points = np.random.uniform(-1, 1, size=(num_random, 2))
    
    # Compute SDF values
    points_tensor = torch.from_numpy(random_points).float().to(device)
    with torch.no_grad():
        sdf_values = decoder(points_tensor).cpu().numpy().flatten()
    
    # Simple winding number around contour points (approximate)
    try:
        if len(contour_points_2d) >= 3:
            hull = ConvexHull(contour_points_2d)
            hull_points = contour_points_2d[hull.vertices]
            
            hull_path = Path(hull_points)
            inside_hull = hull_path.contains_points(random_points)
            
            # Compare signs
            pred_inside = sdf_values < 0
            agreement = (pred_inside == inside_hull).mean()
            
            # For points where they disagree, compute distance to contour
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
                ax.set_xlabel('u'); ax.set_ylabel('v')
                ax.set_aspect('equal')
                
                # Plot 2: Disagreement map
                ax = axes[1]
                colors = np.where(disagree_mask, 'yellow', 'black')
                ax.scatter(random_points[::10, 0], random_points[::10, 1], 
                          c=colors[::10], s=1, alpha=0.3)
                ax.scatter(contour_points_2d[:, 0], contour_points_2d[:, 1], 
                          c='green', s=5, alpha=0.8)
                ax.set_title(f'Disagreements (yellow) - {disagree_mask.mean()*100:.2f}%')
                ax.set_xlabel('u'); ax.set_ylabel('v')
                ax.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, 'sign_consistency.png'), dpi=150)
                plt.close()
            
            return results
        else:
            return {"error": "Need at least 3 contour points for convex hull"}
    except Exception as e:
        return {"error": f"Could not compute sign consistency: {str(e)}"}

def grad_norm_stats(decoder, n=50000, span=1.0, device="cuda", batch=32768, dim=3):
    """Gradient norm statistics (Eikonal check)"""
    try:
        all_norms = []
        done = 0
        while done < n:
            m = min(batch, n - done)
            pts = torch.from_numpy(random_points(m, span, dim)).to(device).requires_grad_(True)
            y = decoder(pts).view(-1)
            ones = torch.ones_like(y)
            grads = torch.autograd.grad(y, pts, grad_outputs=ones, retain_graph=False, create_graph=False)[0]
            norms = grads.norm(dim=1).detach().cpu().numpy()
            all_norms.append(norms)
            done += m
        norms = np.concatenate(all_norms, axis=0)
        err = np.abs(norms - 1.0)
        return {
            "count": int(norms.size),
            "mean_norm": float(norms.mean()),
            "median_norm": float(np.median(norms)),
            "std_norm": float(norms.std()),
            "mean_abs_error_from_1": float(err.mean()),
            "median_abs_error_from_1": float(np.median(err)),
            "frac_near_1_01": float((np.abs(norms - 1.0) < 0.1).mean()),
            "frac_near_1_02": float((np.abs(norms - 1.0) < 0.2).mean()),
        }, norms
    except Exception as e:
        return {"error": f"gradients unavailable: {e.__class__.__name__}: {e}"}, None

# ---------------------------- Plot helpers ------------------------------
def save_hist(data, title, xlabel, out_png, bins=80):
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(data, bins=bins, density=False)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def save_2d_sdf_plot(grid, x, y, title, out_png):
    """Save 2D SDF visualization"""
    plt.figure(figsize=(8, 6))
    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(grid.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='SDF Value')
    plt.contour(grid.T, levels=[0], colors='black', linewidths=1, extent=extent)
    plt.title(title)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------------------------------- CLI ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Verify SIREN model (2D or 3D) properties with comprehensive checks.")
    ap.add_argument("model", help="checkpoint .pt/.pth (TorchScript or state_dict-like).")
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3],
                    help="Dimension of model (2 for 2D SDF, 3 for 3D SDF)")
    ap.add_argument("--span", type=float, default=1.0, help="Evaluate within [-span,span]^dim.")
    ap.add_argument("--grid", type=int, default=256, help="Grid size N for stats (N^dim queries).")
    ap.add_argument("--max-batch", type=int, default=64**3, help="Batch size for grid evaluation.")
    ap.add_argument("--rand", type=int, default=200000, help="Number of random samples for field stats.")
    ap.add_argument("--grad", type=int, default=50000, help="Number of points for Eikonal (autograd).")
    ap.add_argument("--contour-file", type=str, default=None, 
                    help="CSV file with contour points (u,v) for 2D accuracy check")
    ap.add_argument("--on-surface-points", type=str, default=None, 
                    help="PLY file with on-surface points for overlay visualization")
    ap.add_argument("--all-points", type=str, default=None, 
                    help="PLY file with all training points for overlay visualization")
    ap.add_argument("--mesh", type=str, default=None, help="Optional watertight mesh (only for 3D).")
    ap.add_argument("--surf", type=int, default=50000, help="Surface points to sample if mesh is provided.")
    ap.add_argument("--w0", type=float, default=30.0)
    ap.add_argument("--w0-initial", type=float, default=30.0)
    ap.add_argument("--outdir", default="model_verification_report")
    args = ap.parse_args()

    # Create parent model_verification folder and nest the outdir inside it
    parent_dir = "model_verification"
    os.makedirs(parent_dir, exist_ok=True)
    
    # If outdir is a relative path (not absolute), nest it under parent_dir
    if not os.path.isabs(args.outdir):
        args.outdir = os.path.join(parent_dir, args.outdir)
    
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Verifying {args.dim}D SDF model")
    print(f"Output directory: {args.outdir}")
    
    # 1) Load model
    model = load_siren_auto(args.model, device, w0=args.w0, w0_initial=args.w0_initial, dim=args.dim)
    model.eval()

    report = {"dimension": args.dim, "model_path": args.model}

    # 2) Basic random stats
    print("\n[1] Computing basic statistics...")
    report["basic_stats"] = check_basic_stats(model, n_points=args.rand, span=args.span, 
                                               device=device, dim=args.dim)
    
    # 3) Grid sampling and visualization
    print("\n[2] Sampling on grid...")
    
    if args.dim == 2:
        # 2D grid
        G, x_grid, y_grid = sample_sdf_grid_2d(model, N=args.grid, max_batch=args.max_batch, 
                                                device=device, span=args.span)
        g_flat = G.reshape(-1)
        
        # Save basic 2D visualization
        save_2d_sdf_plot(G, x_grid, y_grid, f"2D SDF Grid (N={args.grid})",
                         os.path.join(args.outdir, "sdf_2d_grid.png"))
        
        # ===== Enhanced contour visualization =====
        print("\n[3] Creating enhanced contour visualizations...")
        
        # Comprehensive contour plot
        visualize_sdf_contours_2d(
            G, x_grid, y_grid,
            os.path.join(args.outdir, "sdf_contour_analysis.png"),
            levels=20, highlight_zero=True
        )
        
        # Extract and save zero contour separately
        world_contours = extract_and_save_zero_contour(
            G, x_grid, y_grid,
            os.path.join(args.outdir, "zero_contour.png")
        )
        
        # ===== Point cloud overlay analysis =====
        if args.on_surface_points or args.all_points:
            print("\n[3b] Creating point cloud overlay visualizations...")
            points_list = []
            
            if args.on_surface_points:
                pts_on_surface, colors_on_surface = load_ply_points(args.on_surface_points)
                points_list.append((pts_on_surface, colors_on_surface, "On-surface Points"))
            
            if args.all_points:
                pts_all, colors_all = load_ply_points(args.all_points)
                points_list.append((pts_all, colors_all, "All Training Points"))
            
            if points_list:
                create_overlay_analysis_2d(G, x_grid, y_grid, points_list, args.outdir)
        
        # If contour file provided, compare with original
        if args.contour_file:
            print("\n[4] Comparing with original contour points...")
            contour_points = np.loadtxt(args.contour_file, delimiter=",", skiprows=1)
            if contour_points.ndim == 1:
                contour_points = contour_points.reshape(-1, 2)
            
            contour_metrics = compare_with_original_contour(
                G, x_grid, y_grid, contour_points,
                os.path.join(args.outdir, "contour_comparison.png"),
                model=model, device=device
            )
            
            if contour_metrics:
                report["contour_comparison"] = contour_metrics
        
        # Continue with existing 2D checks
        report["grid_stats"] = {
            "dim": 2,
            "grid_N": int(args.grid),
            "min": float(g_flat.min()), "max": float(g_flat.max()),
            "mean": float(g_flat.mean()), "median": float(np.median(g_flat)),
            "std": float(g_flat.std()),
            "frac_negative": float((g_flat < 0).mean()),
            "frac_positive": float((g_flat > 0).mean()),
        }
        
        save_hist(g_flat, f"Field values on grid N={args.grid} (2D)", "f(x)", 
                  os.path.join(args.outdir, "hist_grid_values.png"))
        
        # Contour accuracy check (existing)
        if args.contour_file:
            print("\n[5] Checking contour accuracy (existing)...")
            contour_points = np.loadtxt(args.contour_file, delimiter=",", skiprows=1)
            if contour_points.ndim == 1:
                contour_points = contour_points.reshape(-1, 2)
            report["contour_accuracy"] = check_contour_accuracy(
                model, contour_points, device=device, resolution=args.grid, outdir=args.outdir
            )
        
        # Eikonal error map
        print("\n[6] Computing Eikonal error map...")
        report["eikonal_stats"] = eikonal_error_map(model, resolution=100, device=device, outdir=args.outdir)
        
        # Gradient field visualization
        print("\n[7] Visualizing gradient field...")
        report["gradient_stats"] = visualize_gradient_field(model, resolution=30, device=device, outdir=args.outdir)
        
        # SDF profiles
        print("\n[8] Plotting SDF profiles...")
        directions = [
            ([-0.8, -0.8], [0.8, 0.8]),  # diagonal
            ([-0.8, 0.0], [0.8, 0.0]),    # horizontal
            ([0.0, -0.8], [0.0, 0.8]),    # vertical
            ([-0.8, 0.8], [0.8, -0.8]),   # anti-diagonal
        ]
        for i, (start, end) in enumerate(directions):
            prefix = f"profile_{i}"
            results = plot_sdf_profile(model, start, end, device=device, 
                                       outdir=args.outdir, prefix=prefix)
            report.update(results)
        
        # Sign consistency check
        if args.contour_file:
            print("\n[9] Checking sign consistency...")
            report["sign_consistency"] = check_sign_consistency_2d(
                model, contour_points, num_random=args.rand//2, device=device, outdir=args.outdir
            )
        
        # Gradient norm stats
        print("\n[10] Computing gradient norm statistics...")
        grad_stats, norms = grad_norm_stats(model, n=args.grad, span=args.span, 
                                            device=device, dim=args.dim)
        report["eikonal_grad_norm"] = grad_stats
        if norms is not None:
            save_hist(norms, f"Gradient norms ||∇f|| (2D)", "||∇f||", 
                      os.path.join(args.outdir, "hist_grad_norms.png"))
    
    else:
        # 3D mode (original, unchanged)
        G, xC, yC, zC = sample_sdf_grid_3d(model, N=min(args.grid, 128), 
                                            max_batch=args.max_batch, device=device)
        g_flat = G.reshape(-1)
        
        # Save center slices
        mid = G.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(G[mid, :, :].T, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[0].set_title(f'YZ slice at x={xC[mid]:.3f}')
        axes[0].set_xlabel('y'); axes[0].set_ylabel('z')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(G[:, mid, :].T, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[1].set_title(f'XZ slice at y={yC[mid]:.3f}')
        axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(G[:, :, mid].T, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[2].set_title(f'XY slice at z={zC[mid]:.3f}')
        axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "sdf_3d_slices.png"), dpi=150)
        plt.close()
        
        report["grid_stats"] = {
            "dim": 3,
            "grid_N": int(args.grid),
            "min": float(g_flat.min()), "max": float(g_flat.max()),
            "mean": float(g_flat.mean()), "median": float(np.median(g_flat)),
            "std": float(g_flat.std()),
            "frac_negative": float((g_flat < 0).mean()),
            "frac_positive": float((g_flat > 0).mean()),
        }
        
        save_hist(g_flat, f"Field values on grid N={args.grid} (3D)", "f(x)", 
                  os.path.join(args.outdir, "hist_grid_values.png"))
        
        # Gradient norm stats for 3D
        grad_stats, norms = grad_norm_stats(model, n=args.grad, span=args.span, 
                                            device=device, dim=3)
        report["eikonal_grad_norm"] = grad_stats
        if norms is not None:
            save_hist(norms, "Gradient norms ||∇f|| (3D)", "||∇f||", 
                      os.path.join(args.outdir, "hist_grad_norms.png"))
        
        # Mesh-based checks for 3D
        if args.mesh is not None and HAVE_TRIMESH:
            try:
                mesh = trimesh.load(args.mesh, force='mesh')
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.dump()))
                
                # Surface samples
                S = mesh.sample(args.surf).astype(np.float32)
                VS = batched_eval(model, S, device)
                
                report["mesh_surface_stats"] = {
                    "n_surface": int(VS.size),
                    "abs_mean_on_surface": float(np.abs(VS).mean()),
                    "abs_median_on_surface": float(np.median(np.abs(VS))),
                    "median_value_on_surface": float(np.median(VS)),
                }
                
                save_hist(VS, "Values on mesh surface (3D)", "f(x) @ surface",
                         os.path.join(args.outdir, "hist_surface_values.png"))
            except Exception as e:
                report["mesh_surface_stats"] = {"error": str(e)}

    # 10) Heuristic signedness score
    frac_neg_grid = report["grid_stats"]["frac_negative"]
    frac_pos_grid = report["grid_stats"]["frac_positive"]
    signedness = float(2.0 * min(frac_neg_grid, frac_pos_grid))
    report["signedness_score_grid"] = signedness

    # Save JSON report
    out_json = os.path.join(args.outdir, "verification_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Report saved to: {out_json}")
    print(f"Images saved to: {args.outdir}")
    print(f"\nKey metrics:")
    print(f"  Signedness score: {signedness:.3f} (0=all same sign, 1=balanced)")
    if args.dim == 2 and args.contour_file:
        if 'contour_accuracy' in report:
            ca = report['contour_accuracy']
            print(f"  Mean distance to contour: {ca.get('mean_distance_to_contour', 'N/A')}")
            print(f"  Mean |SDF| at contour: {ca.get('mean_abs_sdf_at_contour', 'N/A')}")
        if 'contour_comparison' in report:
            cc = report['contour_comparison']
            print(f"  Contour comparison RMSE: {cc.get('rmse', 'N/A')}")
    if 'eikonal_grad_norm' in report and 'mean_abs_error_from_1' in report['eikonal_grad_norm']:
        print(f"  Eikonal error: {report['eikonal_grad_norm']['mean_abs_error_from_1']:.4f}")

if __name__ == "__main__":
    main()