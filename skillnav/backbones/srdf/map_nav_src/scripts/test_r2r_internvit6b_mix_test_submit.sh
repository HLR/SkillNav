#!/bin/bash

DATA_ROOT=../datasets
train_alg=dagger

features=internvit.6b
ft_dim=3200
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=16
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}-aug.mp3d.hm3d.srdf
# name=${name}-srdf

outdir=${DATA_ROOT}/exprs_map/finetune/${name}

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

# List of individual resume file paths
pretrain_files=(
    '../datasets/SRDF/trained_models/pretrain/model_step_170000.pt'
    '../ScaleVLN/datasets/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt'
)

resume_files=(
    '../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf/ckpts/best_val_unseen'
)

sub_folders=(
    "VLN-SRDR_pretrain"  
)

# Loop over resume files and run test command for each
# for resume_file in "${resume_files[@]}"; do
for i in "${!resume_files[@]}"; do
    resume_file="${resume_files[$i]}"
    sub_folder="${sub_folders[$i]}"

    outdir=${DATA_ROOT}/exprs_map/finetune/${name}
    # outdir=${DATA_ROOT}/exprs_map/finetune/${name}/${sub_folder}

    echo "Running evaluation for: $resume_file"
    echo "Output directory: $outdir"

    CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag \
        --tokenizer bert \
        --bert_ckpt_file "${pretrain_files[0]}" \
        --output_dir "$outdir" \
        --test \
        --submit \
        --feedback argmax \
        --resume_file "$resume_file" \
        --val_env_names test \
        # --detailed_output \
       
    echo "Finished evaluation with: $resume_file"
done