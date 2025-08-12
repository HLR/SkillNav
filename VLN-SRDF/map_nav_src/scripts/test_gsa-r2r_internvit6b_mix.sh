#!/bin/bash

# ======== CONFIGURATION ========
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
name=${name}-aug.mp3d.hm3d.srdf

outdir=${DATA_ROOT}/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset gsa-r2r
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
      --val_env_names test_residential_basic test_non_residential_basic test_non_residential_scene test_residential_user test_non_residential_user

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

# test
CUDA_VISIBLE_DEVICES=$1 python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../datasets/SRDF/trained_models/pretrain/model_step_170000.pt \
      --test \
      --submit \
      --detailed_output \
      --dataset gsa-r2r \
      --resume_file ../datasets/exprs_map/finetune/dagger-internvit.6b-seed.0-aug.mp3d.hm3d.srdf/ckpts/best_val_unseen \
      # --val_env_names test_residential_user test_non_residential_user