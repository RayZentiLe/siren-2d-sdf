'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
Modification
'''

# Enable import from parent package
import sys
import os
import torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Path to point cloud file')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--sdf_dimension', type=int, default=3, 
               help='Dimension of input coordinates (2 for 2D SDF, 3 for 3D SDF)')

# Optional plane parameters for 2D sdf Training (sdf_dimension=2)
p.add_argument('--plane_csv', type=str, default=None,
               help='CSV file with plane parameters (p0x,p0y,p0z,nx,ny,nz) - required for 2D cross-sections')

# ON/OFF Surface visualization
p.add_argument('--vis_sampled_point', action='store_true', default=False,
               help='Visualize sampled points (on-surface green, off-surface red) during training')
p.add_argument('--vis_frequency', type=int, default=100,
               help='Save point visualization every N epochs (default: 100, only used if --vis_sampled_point is set)')

# Signed lebeling by cosine similarity of off-surface point and closet on-surface point's normal
p.add_argument('--use_signed_labels', action='store_true', default=False,
               help='If True, off-surface points get +1/-1 labels based on normals; if False, all off-surface = -1')
p.add_argument('--k_neighbors', type=int, default=16,
               help='Number of neighbors for sign determination (only used if --use_signed_labels is set)')

opt = p.parse_args()

# ==================== Choose dataset based on dimension ====================
if opt.sdf_dimension == 2:
    print("=" * 60)
    print("2D SDF Training Mode")
    print("=" * 60)
    
    # Create visualization directory only if flag is True
    vis_dir = None
    if opt.vis_sampled_point:
        vis_dir = os.path.join(opt.logging_root, opt.experiment_name, 'batch_viz')
        print(f"  Point visualization enabled, saving every {opt.vis_frequency} epochs to: {vis_dir}")
    
    # Print label mode info
    label_mode = "SIGNED (+1/-1)" if opt.use_signed_labels else "ORIGINAL (-1 only)"
    print(f"  Label mode: {label_mode}")
    if opt.plane_csv:
        print(f"  Using plane file: {opt.plane_csv}")

    # Use the unified PointCloud2D class
    sdf_dataset = dataio.PointCloud2D(
        pointcloud_path=opt.point_cloud_path,
        on_surface_points=opt.batch_size,
        plane_csv=opt.plane_csv,  # Will be None if not provided
        use_signed_labels=opt.use_signed_labels,
        k_neighbors=opt.k_neighbors,
        vis_sampled_point=opt.vis_sampled_point,
        vis_frequency=opt.vis_frequency,
        vis_dir=vis_dir
    )
    
    in_features = 2
else:
    # Original 3D behavior
    print("=" * 60)
    print("3D SDF Training Mode")
    print("=" * 60)
    print(f"Loading 3D point cloud from: {opt.point_cloud_path}")
    sdf_dataset = dataio.PointCloud(
        opt.point_cloud_path, 
        on_surface_points=opt.batch_size
    )
    in_features = 3

dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# ==================== Model definition ====================
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(
        type='relu', 
        mode='nerf', 
        in_features=opt.sdf_dimension
    )
else:
    model = modules.SingleBVPNet(
        type=opt.model_type, 
        in_features=opt.sdf_dimension
    )
model.cuda()

# Replace summary_fn with a dummy function for 2D
def dummy_summary(*args, **kwargs):
    pass

# Define the loss
loss_fn = loss_functions.sdf

if opt.sdf_dimension == 2:
    summary_fn = dummy_summary
else:
    summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)

torch.cuda.empty_cache()
