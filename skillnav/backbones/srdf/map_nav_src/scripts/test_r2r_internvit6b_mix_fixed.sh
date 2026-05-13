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
name=${name}-seed.${seed}
name=${name}-srdf

outdir=${DATA_ROOT}/exprs_map/multi-models/${name}-fixed

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
      

# Run main inference script
echo "Running main_nav_moe.py..."

# test
CUDA_VISIBLE_DEVICES=$1 python moe/main_nav_moe.py $flag  \
        --tokenizer bert \
        --test \
        --bert_ckpt_file ../datasets/SRDF/trained_models/pretrain/model_step_170000.pt \
        --submit \
        --detailed_output \
        --feedback argmax \
        --batch_size 16 \
        --routing_mode fixed \
        --routing_weights_type int \
        --val_env_names val_unseen \
        --resume_files ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf.direction-2k/ckpts/best_val_unseen \
                        ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf.vertical-2k/ckpts/best_val_unseen \
                        ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf.stop-2k/ckpts/best_val_unseen \
                        ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf.landmark-2k/ckpts/best_val_unseen \
                        ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf.region-2k/ckpts/best_val_unseen \
        --resume_weights 1 1 1 1 1 1 \




