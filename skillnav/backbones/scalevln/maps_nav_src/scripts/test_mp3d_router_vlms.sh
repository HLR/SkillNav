#!/usr/bin/env bash

# Run the ScaleVLN navigation agent while delegating skill routing to an
# already running FastAPI router server (see moe/vLLM_API.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/home/matiany3/ScaleVLN/VLN-DUET/datasets}"
RESULT_ROOT="${RESULT_ROOT:-${DATA_ROOT}/R2R/exprs_map/multi-models/vlms}"
mkdir -p "${RESULT_ROOT}"

AGENT_GPU="${1:-2}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-8}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLITS_ENV="${VAL_SPLITS:-val_unseen}"
IFS=' ' read -r -a VAL_SPLITS <<< "${VAL_SPLITS_ENV}"
if [[ ${#VAL_SPLITS[@]} -eq 0 ]]; then
  VAL_SPLITS=("val_unseen")
fi

ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8010}"
ROUTER_MODEL_NAME="${ROUTER_MODEL_NAME:-}"
ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}/route"

if [[ -z "${ROUTER_MODEL_NAME}" ]]; then
  echo "Error: ROUTER_MODEL_NAME is not set. Export it so results can be grouped by checkpoint." >&2
  echo "Example: ROUTER_MODEL_NAME='Qwen/Qwen2.5-VL-7B-Instruct' bash $0" >&2
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
  echo "Router server at http://${ROUTER_HOST}:${ROUTER_PORT} is not responding. Start it manually first." >&2
  exit 1
fi

model_slug="${ROUTER_MODEL_NAME//\//_}"
model_slug="${model_slug//:/_}"
outdir="${RESULT_ROOT}/${model_slug}"
log_dir="${outdir}/logs"
mkdir -p "${log_dir}"

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

if [[ ${#RESUME_FILES[@]} -ne ${#RESUME_WEIGHTS[@]} ]]; then
  echo "Resume files (${#RESUME_FILES[@]}) and weights (${#RESUME_WEIGHTS[@]}) must have the same length." >&2
  exit 1
fi

for ckpt in "${RESUME_FILES[@]}"; do
  if [[ ! -d "${ckpt}" && ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

common_args=(
  --root_dir "${DATA_ROOT}"
  --dataset r2r
  --output_dir "${outdir}"
  --world_size 1
  --seed 0
  --tokenizer bert
  --enc_full_graph
  --graph_sprels
  --fusion dynamic
  --expert_policy spl
  --train_alg dagger
  --num_l_layers 9
  --num_x_layers 4
  --num_pano_layers 2
  --max_action_len 15
  --max_instr_len 200
  --batch_size "${AGENT_BATCH_SIZE}"
  --lr 1e-5
  --iters 50000
  --log_every 500
  --aug_times 9
  --optim adamW
  --lora_r 8
  --lora_alpha 16
  --lora_target_modules query key value
  --features clip.b16
  --image_feat_size 512
  --angle_feat_size 4
  --ml_weight 0.15
  --feat_dropout 0.4
  --dropout 0.5
  --gamma 0.
  --feature_file clip_vit-b16_mp3d_hm3d_gibson.hdf5
)
common_args+=(--train_env_names "${TRAIN_SPLIT}")
common_args+=(--val_env_names)
common_args+=("${VAL_SPLITS[@]}")

echo "[INFO] Router server ${ROUTER_MODEL_NAME} at ${ROUTER_URL}"
echo "[INFO] Results will be saved to ${outdir}"

CUDA_VISIBLE_DEVICES="${AGENT_GPU}" python "${PROJECT_ROOT}/moe/main_nav_moe_top1.py" \
  "${common_args[@]}" \
  --bert_ckpt_file "${DATA_ROOT}/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt" \
  --test \
  --submit \
  --detailed_output \
  --feedback argmax \
  --routing_mode top1 \
  --routing_weights_type int \
  --router_server_url "${ROUTER_URL}" \
  --resume_files "${RESUME_FILES[@]}" \
  --resume_weights "${RESUME_WEIGHTS[@]}"

echo "[INFO] Run complete. Logs stored under ${log_dir}."
