#!/usr/bin/env bash

# Evaluate GSA-R2R splits while routing via an external VLM server (e.g., GLM).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/home/matiany3/ScaleVLN/VLN-DUET/datasets}"
RESULT_ROOT="${RESULT_ROOT:-${DATA_ROOT}/R2R/exprs_map/multi-models/vlms}"
mkdir -p "${RESULT_ROOT}"

train_alg=dagger
features=clip.b16
ft_dim=512
ngpus=1
bs=16
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}

name=${name}-aug.mp3d.prevalent.temporal-re-2k

AGENT_GPU="${1:-2}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8011}"
ROUTER_MODEL_NAME="${ROUTER_MODEL_NAME:-THUDM/GLM-4.1V-9B-Thinking}"
ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}/route"

if [[ -z "${ROUTER_MODEL_NAME}" ]]; then
  echo "Error: ROUTER_MODEL_NAME is not set." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to probe router server health. Please install it." >&2
  exit 1
fi

wait_for_server() {
  local retries=40
  local delay=3
  local url="http://${ROUTER_HOST}:${ROUTER_PORT}/health"
  for ((i=1; i<=retries; i++)); do
    if curl -sf "${url}" >/dev/null; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

if ! wait_for_server; then
  echo "Router server at ${ROUTER_URL} is not responding. Start it manually first." >&2
  exit 1
fi

model_slug="${ROUTER_MODEL_NAME//\//_}"
model_slug="${model_slug//:/_}"
outdir="${RESULT_ROOT}/${model_slug}"
log_dir="${outdir}/logs"
mkdir -p "${log_dir}"

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
      --val_env_names test_residential_basic_mp3d_filtered_part_1 test_non_residential_basic_mp3d_filtered_part_1 test_non_residential_scene_mp3d_filtered_part_1
                       test_residential_basic_mp3d_filtered_part_2 test_non_residential_basic_mp3d_filtered_part_2 test_non_residential_scene_mp3d_filtered_part_2
                       test_residential_basic_mp3d_filtered_part_3 test_non_residential_basic_mp3d_filtered_part_3 test_non_residential_scene_mp3d_filtered_part_3
                       test_residential_basic_mp3d_filtered_part_4 test_non_residential_basic_mp3d_filtered_part_4 test_non_residential_scene_mp3d_filtered_part_4
                       test_residential_basic_mp3d_filtered_part_5 test_non_residential_basic_mp3d_filtered_part_5 test_non_residential_scene_mp3d_filtered_part_5
                       test_residential_basic_mp3d_filtered_part_6 test_non_residential_basic_mp3d_filtered_part_6 test_non_residential_scene_mp3d_filtered_part_6
                       test_residential_basic_mp3d_filtered_part_7 test_non_residential_basic_mp3d_filtered_part_7 test_non_residential_scene_mp3d_filtered_part_7
                       test_residential_basic_mp3d_filtered_part_8 test_non_residential_basic_mp3d_filtered_part_8 test_non_residential_scene_mp3d_filtered_part_8

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

declare -a RESUME_FILES=(
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.direction-resized-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.vertical-resized-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.stop-resized-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.landmark-resized-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-aug.region-resized-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.temporal-re-2k-pretrained/ckpts/best_val_unseen"
  "${DATA_ROOT}/R2R/trained_models/finetune/duet_vit-b16_ft_best_val_unseen"
)
declare -a RESUME_WEIGHTS=(1 1 1 1 1 0 0)

for ckpt in "${RESUME_FILES[@]}"; do
  if [[ ! -d "${ckpt}" && ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

echo "[INFO] Router server ${ROUTER_MODEL_NAME} at ${ROUTER_URL}"
mkdir -p "${outdir}"

CUDA_VISIBLE_DEVICES="${AGENT_GPU}" python "${PROJECT_ROOT}/moe/main_nav_moe_top1.py" $flag \
  --bert_ckpt_file "${DATA_ROOT}/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt" \
  --test \
  --submit \
  --detailed_output \
  --feedback argmax \
  --routing_mode top1 \
  --instruction_reorder \
  --routing_weights_type int \
  --router_server_url "${ROUTER_URL}" \
  --resume_files "${RESUME_FILES[@]}" \
  --resume_weights "${RESUME_WEIGHTS[@]}" \
  --feature_file clip_vit-b16_mp3d_hm3d_gibson.hdf5 \
  --batch_size "${bs}"

echo "[INFO] Run complete. Logs stored under ${log_dir}."
