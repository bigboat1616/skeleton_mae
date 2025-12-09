#!/usr/bin/env bash
set -euo pipefail

CKPT_DEFAULTS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
LAYER_TAGS=("2layer")
declare -a CKPT_PATHS=()

if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    CKPT_PATHS+=("$arg")
  done
else
  for layer_tag in "${LAYER_TAGS[@]}"; do
    CKPT_PATHS+=("./ckpt/random/${layer_tag}/best_model_final.pth")
  done
fi

CFG_PATH=${CFG_PATH:-"configs_skeleton.yml"}
DEVICE=${DEVICE:-"cuda"}
BASE_SAVE_DIR=${BASE_SAVE_DIR:-"skeleton/val_coord_visualizations"}
SEED=${SEED:-42}
MASK_RATIOS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
MAX_BATCHES=${MAX_BATCHES:-10000}
SUMMARY_OUT=${SUMMARY_OUT:-"logs/val_summary.txt"}

echo "Config:       ${CFG_PATH}"
echo "Device:       ${DEVICE}"
echo "Seed:         ${SEED}"
echo "Output base:  ${BASE_SAVE_DIR}"
echo "Max batches:  ${MAX_BATCHES}"
echo "CKPT targets:"
for path in "${CKPT_PATHS[@]}"; do
  echo "  - ${path}"
done
echo
echo "Summary out:  ${SUMMARY_OUT}"
for ckpt in "${CKPT_PATHS[@]}"; do
  echo "==== Evaluating checkpoint: ${ckpt} ===="
  ckpt_dir_name=$(basename "$(dirname "${ckpt}")")

  for ratio in "${MASK_RATIOS[@]}"; do
    SAVE_DIR="${BASE_SAVE_DIR}/${ckpt_dir_name}/mask_${ratio}"
    echo ">>> Running mask_ratio=${ratio} (saving to ${SAVE_DIR})"

    python test_skeleton_coord_val.py \
      --ckpt "${ckpt}" \
      --cfg "${CFG_PATH}" \
      --device "${DEVICE}" \
      --mask_ratio "${ratio}" \
      --save_dir "${SAVE_DIR}" \
      --seed "${SEED}" \
      --max_batches "${MAX_BATCHES}" \
      --summary_out "${SUMMARY_OUT}"
  done
  echo
done

