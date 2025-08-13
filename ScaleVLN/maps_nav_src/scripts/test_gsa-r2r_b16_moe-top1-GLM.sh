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


name=${name}-aug.mp3d.prevalent.moe-top1-routing-add_prev_sub_instructions-GLM
outdir=${DATA_ROOT}/GSA-R2R/exprs_map/multi-models/${name}


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


# Run main inference script
echo "Running main_nav_moe_top1.py..."

# test
CUDA_VISIBLE_DEVICES=$1 python moe/main_nav_moe_top1.py $flag  \
        --tokenizer bert \
        --test \
        --bert_ckpt_file ${DATA_ROOT}/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt \
        --submit \
        --detailed_output \
        --feedback argmax \
        --batch_size 8 \
        --dataset gsa-r2r \
        --feature_file clip_vit-b16_mp3d_hm3d_gibson.hdf5 \
        --routing_mode top1 \
        --instruction_reorder \
        --resume_files  ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.direction-resized-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.vertical-resized-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.stop-resized-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.landmark-resized-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.region-resized-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-re-2k-pretrained/ckpts/best_val_unseen \
                        ../datasets/R2R/trained_models/finetune/duet_vit-b16_ft_best_val_unseen \
        --resume_weights 1 1 1 1 1 0 0 \
        --val_env_names test_residential_basic test_non_residential_basic test_non_residential_scene \
        --localizer_model THUDM/GLM-4.1V-9B-Thinking \
        --skill_model THUDM/GLM-4.1V-9B-Thinking \
        --gpu_memory_utilization 0.6 \
        --localizer_gpu_id 2 \
        --skill_gpu_id 2 \
        