#!/bin/bash

# ======== CONFIGURATION ========
DATA_ROOT=/home/matiany3/ScaleVLN/VLN-SRDF/datasets
train_alg=dagger

features=internvit.6b
ft_dim=3200
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=16
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-srdf

outdir=${DATA_ROOT}/exprs_map/finetune/${name}-pretrain

# Eval paths
ANNTROOT="/home/matiany3/ScaleVLN/VLN-DUET/datasets/NavNuances/annoatations"
SCANDIR="/home/matiany3/MapGPT/datasets/R2R/connectivity/scans.txt"

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
      --use_lora False
      --lora_r 8
      --lora_alpha 16
      --lora_target_modules query key value
      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --ml_weight 0.15
      --feat_dropout 0.4
      --dropout 0.5
      --gamma 0."


# ======== LIST OF CHECKPOINTS ========
checkpoint_list=(
    "/home/matiany3/ScaleVLN/VLN-SRDF/datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf/ckpts/best_val_unseen"
    )

# ======== INFERENCE + EVALUATION LOOP ========
for resume_ckpt in "${checkpoint_list[@]}"; do
    echo "=== Running inference on checkpoint: $resume_ckpt ==="

    # Infer output directory from checkpoint path
    # OUTROOT=$(dirname "$resume_ckpt")  
    OUTROOT=$(dirname "$(dirname "$resume_ckpt")") # e.g., /.../dagger-...-pretrained
    SUBMITROOT="${OUTROOT}/preds"

    CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag \
        --tokenizer bert \
        --bert_ckpt_file /home/matiany3/ScaleVLN/VLN-SRDF/datasets/SRDF/trained_models/pretrain/model_step_170000.pt \
        --resume_file "$resume_ckpt" \
        --test \
        --submit \
        --output_dir "$OUTROOT" \
        --detailed_output \
        --feedback argmax \
        --val_env_names NavNuances_DC NavNuances_VM NavNuances_LR NavNuances_RR NavNuances_NU \
        # --val_env_names val_seen val_unseen NavNuances_DC \

    echo "=== Evaluating results in: $SUBMITROOT ==="

    python /home/matiany3/ScaleVLN/VLN-SRDF/map_nav_src/evaluation/eval.py \
        --annotation_root "$ANNTROOT" \
        --submission_root "$SUBMITROOT" \
        --out_root "$SUBMITROOT" \
        --scans_dir "$SCANDIR"

    echo "=== Done: $resume_ckpt ==="
    echo
done