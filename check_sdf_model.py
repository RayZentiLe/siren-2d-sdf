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

# ------------------------ Verification Functions ---------------------------

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
        
        # Save 2D visualization
        save_2d_sdf_plot(G, x_grid, y_grid, f"2D SDF Grid (N={args.grid})",
                         os.path.join(args.outdir, "sdf_2d_grid.png"))
        
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
        
        # 4) Contour accuracy check (if contour file provided)
        if args.contour_file:
            print("\n[3] Checking contour accuracy...")
            contour_points = np.loadtxt(args.contour_file, delimiter=",", skiprows=1)
            if contour_points.ndim == 1:
                contour_points = contour_points.reshape(-1, 2)
            report["contour_accuracy"] = check_contour_accuracy(
                model, contour_points, device=device, resolution=args.grid, outdir=args.outdir
            )
        
        # 5) Eikonal error map
        print("\n[4] Computing Eikonal error map...")
        report["eikonal_stats"] = eikonal_error_map(model, resolution=100, device=device, outdir=args.outdir)
        
        # 6) Gradient field visualization
        print("\n[5] Visualizing gradient field...")
        report["gradient_stats"] = visualize_gradient_field(model, resolution=30, device=device, outdir=args.outdir)
        
        # 7) SDF profiles in different directions
        print("\n[6] Plotting SDF profiles...")
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
        
        # 8) Sign consistency check
        if args.contour_file:
            print("\n[7] Checking sign consistency...")
            report["sign_consistency"] = check_sign_consistency_2d(
                model, contour_points, num_random=args.rand//2, device=device, outdir=args.outdir
            )
        
        # 9) Gradient norm stats (Eikonal)
        print("\n[8] Computing gradient norm statistics...")
        grad_stats, norms = grad_norm_stats(model, n=args.grad, span=args.span, 
                                            device=device, dim=args.dim)
        report["eikonal_grad_norm"] = grad_stats
        if norms is not None:
            save_hist(norms, f"Gradient norms ||∇f|| (2D)", "||∇f||", 
                      os.path.join(args.outdir, "hist_grad_norms.png"))
    
    else:
        # 3D mode (original)
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
    if 'eikonal_grad_norm' in report and 'mean_abs_error_from_1' in report['eikonal_grad_norm']:
        print(f"  Eikonal error: {report['eikonal_grad_norm']['mean_abs_error_from_1']:.4f}")
    if args.dim == 2:
    # 2D grid - use span=1.0 for normalized coordinates
        print(f"\n[2] Sampling on {args.grid}x{args.grid} grid in [-1, 1]²...")
        G, x_grid, y_grid = sample_sdf_grid_2d(
            model, 
            N=args.grid, 
            max_batch=args.max_batch, 
            device=device, 
            span=1.0  # Force span=1.0 for normalized space
        )
        
        # Random points - use args.span (default 1.0)
        print(f"\n[1] Sampling {args.rand} random points in [-{args.span}, {args.span}]²...")
        P = random_points(args.rand, args.span, dim=2)
        V = batched_eval(model, P, device)

if __name__ == "__main__":
    main()
