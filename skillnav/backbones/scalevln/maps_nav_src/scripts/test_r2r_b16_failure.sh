#!/bin/bash

DATA_ROOT=../datasets
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
name=${name}-aug.mp3d.prevalent.R2R_failure

outdir=${DATA_ROOT}/R2R/exprs_map/failure/${name}-pretrained


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


resume_files=(
    # '../datasets/R2R/trained_models/finetune/duet_vit-b16_ft_best_val_unseen'
    # '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-re-2k-pretrained/ckpts/best_val_unseen'

    '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.direction-2k-pretrained/ckpts/best_val_unseen'
    '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.vertical-2k-pretrained/ckpts/best_val_unseen'
    '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.stop-2k-pretrained/ckpts/best_val_unseen'
    '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.landmark-2k-pretrained/ckpts/best_val_unseen'
    '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.region-2k-pretrained/ckpts/best_val_unseen'
    
    # '../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.direction-2k-pretrained/ckpts/best_val_unseen'

)

sub_folders=(
    # "ScaleVLN_DUET"
    # "Temporal_DUET"
  
    "Direction_DUET_on_Temporal_DUET"
    "Vertical_DUET_on_Temporal_DUET"
    "Stop_DUET_on_Temporal_DUET"
    "Landmark_DUET_on_Temporal_DUET"
    "Region_DUET_on_Temporal_DUET"
    
)

# Loop over resume files and run test command for each

for i in "${!resume_files[@]}"; do
    resume_file="${resume_files[$i]}"
    sub_folder="${sub_folders[$i]}"

    outdir=${DATA_ROOT}/R2R/exprs_map/failure/${name}/${sub_folder}

    echo "Running evaluation for: $resume_file"
    echo "Output directory: $outdir"

    CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag \
        --tokenizer bert \
        --bert_ckpt_file ../datasets/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt \
        --resume_file "$resume_file" \
        --output_dir "$outdir" \
        --test \
        --submit \
        --detailed_output \
        --feedback argmax \
        --val_env_names val_seen val_unseen\
        --batch_size 32
    
    echo "Method : $sub_folder"
    echo "Finished evaluation with: $resume_file"
done

