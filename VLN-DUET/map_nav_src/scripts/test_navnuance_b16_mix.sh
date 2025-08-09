#!/bin/bash

# ======== CONFIGURATION ========
DATA_ROOT=/home/matiany3/ScaleVLN/VLN-DUET/datasets
train_alg=dagger

features=clip.b16
ft_dim=512
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=16
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-aug.mp3d.prevalent.b16
outdir=${DATA_ROOT}/NavNuances/exprs_map/finetune/${name}

# Eval paths
ANNTROOT="${DATA_ROOT}/NavNuances/annoatations"
SCANDIR="${DATA_ROOT}/R2R/connectivity_mp3d/scans.txt"

# ======== SHARED FLAGS ========
flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      
      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --expert_policy spl
      --train_alg ${train_alg}
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      --max_action_len 15
      --max_instr_len 200
      --train_env_names train 
      --val_env_names val_seen val_unseen 
      --batch_size ${bs}
      --lr 1e-5
      --iters 50000
      --log_every 500
      --aug_times 9
      --optim adamW
      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --ml_weight 0.15
      --feat_dropout 0.4
      --dropout 0.5
      --gamma 0."


# ======== LIST OF CHECKPOINTS ========
resume_files=(
    # "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.direction-2k-pretrained/ckpts/best_val_unseen"
    # "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.vertical-2k-pretrained/ckpts/best_val_unseen"
    # "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.stop-2k-pretrained/ckpts/best_val_unseen"
    # "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.landmark-2k-pretrained/ckpts/best_val_unseen"
    # "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.region-2k-pretrained/ckpts/best_val_unseen"

    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.direction-resized-pretrained/ckpts/best_val_unseen"
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.vertical-resized-pretrained/ckpts/best_val_unseen"
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.stop-resized-pretrained/ckpts/best_val_unseen"
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.landmark-resized-pretrained/ckpts/best_val_unseen"
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.region-resized-pretrained/ckpts/best_val_unseen"
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-re-2k-pretrained/ckpts/best_val_unseen"
    ""
    "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/trained_models/finetune/duet_vit-b16_ft_best_val_unseen"
    )

sub_folders=(
    "Direction_DUET"
    "Vertical_DUET"
    "Stop_DUET"
    "Landmark_DUET"
    "Region_DUET"
    "Temporal_DUET"
    "Temporal_DUET_unseen"
    "ScaleVLN"
)

# ======== INFERENCE + EVALUATION LOOP ========
# for resume_ckpt in "${checkpoint_list[@]}"; do
#     echo "=== Running inference on checkpoint: $resume_ckpt ==="

#     # Infer output directory from checkpoint path
#     # OUTROOT=$(dirname "$resume_ckpt")  
#     OUTROOT=$(dirname "$(dirname "$resume_ckpt")") # e.g., /.../dagger-...-pretrained
#     SUBMITROOT="${OUTROOT}/preds"

for i in "${!resume_files[@]}"; do
    resume_file="${resume_files[$i]}"
    sub_folder="${sub_folders[$i]}"

    OUTROOT=${DATA_ROOT}/NavNuances/exprs_map/finetune/${name}/${sub_folder}
    SUBMITROOT="${OUTROOT}/preds"
    echo "Running evaluation for: $resume_file"
    echo "Output directory: $OUTROOT"


    CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag \
        --tokenizer bert \
        --bert_ckpt_file ../datasets/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt \
        --resume_file "$resume_ckpt" \
        --test \
        --submit \
        --output_dir "$OUTROOT" \
        --detailed_output \
        --feedback argmax \
        --train_env_names val_unseen_debug \
        --val_env_names NavNuances_DC NavNuances_VM NavNuances_LR NavNuances_RR NavNuances_NU \
        # --val_env_names val_seen val_unseen NavNuances_DC \

    echo "=== Evaluating results in: $SUBMITROOT ==="

    python /home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/evaluation/eval.py \
        --annotation_root "$ANNTROOT" \
        --submission_root "$SUBMITROOT" \
        --out_root "$SUBMITROOT" \
        --scans_dir "$SCANDIR"

    echo "=== Done: $resume_ckpt ==="
    echo
done