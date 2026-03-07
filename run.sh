 #!/bin/bash

i='001'

INPUT_FILE="data/noisy_data_4/cross-section/section_${i}.xyz"

EXP_NAME="test_2Dclean"

NORMALS_FILE="${INPUT_FILE%.*}_normals.xyz"

python -c "
import torch
torch.cuda.empty_cache()
"

# Estimate normals
# python src/estimate_normals.py data/clean_data_4.ply
# echo "Normals estimated"

# # Generate i cross-sections along a line
# python cross_section.py \
#     --ply "data/clean_data_4_normals.xyz" \
#     --start 828491.000000 825637.125000 -3.603000 \
#     --end 828576.000000 825710.687500 -2.893000 \
#     --num_sections 10 \
#     --thickness 0.5 \
#     --out_dir "./data/clean_data_4" \
#     --base_name "section"

# python src/ply_to_xyz.py "${INPUT_FILE%.*}.ply"

# Train SDF (uses normals file)
# python experiment_scripts/train_sdf.py \
#     --model_type=sine \
#     --point_cloud_path="$INPUT_FILE" \
#     --plane_csv="data/clean_data_4/plane-section/section_${i}.csv" \
#     --batch_size=500 \
#     --experiment_name="$EXP_NAME" \
#     --num_epochs  1 \
#     --epochs_til_ckpt 1 \
#     --use_signed_labels \
#     --k_neighbors 16 \
#     --sdf_dimension 2 \
#     --vis_sampled_point \
#     --vis_frequency 1

echo "Training complete"

python -c "
import torch
torch.cuda.empty_cache()
"

python check_sdf_model.py \
    logs/"$EXP_NAME"/checkpoints/model_current.pth \
    --dim 2 \
    --grid 256 \
    --rand 100000 \
    --grad 50000 \
    --outdir "$EXP_NAME"_model_verification \
    --on-surface-points "logs/$EXP_NAME/batch_viz/real_sdf_colored_epoch_0000_batch_00.ply"  \
    --all-points "logs/$EXP_NAME/batch_viz/real_sdf_on_surface_epoch_0000_batch_00.ply" 

