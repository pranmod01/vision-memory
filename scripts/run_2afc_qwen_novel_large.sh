#!/bin/bash
#SBATCH --job-name=2afc_qwen_novel_large
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:1

# 2-AFC Recognition: qwen3-vl-8b, novel foils, large N (300-1000)
# Uses A6000 (48GB), reduced per-image resolution, and chunked vision encoding.
# (Override --gres on the sbatch CLI for h100/l40s if needed.)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="qwen"
N_TRIALS=10
RESULTS_DIR="$SCRIPT_DIR/results"
read -r -a SIZES <<< "${SIZES:-300 400 500 1000}"
read -r -a DATASETS <<< "${DATASETS:-things Brady2008}"
FOIL="novel"

# Memory-saving knobs (override via env if needed)
MAX_IMG_SIZE="${MAX_IMG_SIZE:-256}"
VISION_CHUNK_SIZE="${VISION_CHUNK_SIZE:-32}"
ATTN="${ATTN:-sdpa}"

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    [ -f "$RESULTS_DIR/results_2afc_qwen3-vl-8b_n${n_images}_${dataset}_${FOIL}.json" ]
}

echo "========== 2-AFC Recognition (large): $MODEL | $FOIL =========="
echo "  max_image_size=$MAX_IMG_SIZE  vision_chunk_size=$VISION_CHUNK_SIZE  attn=$ATTN"

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if check_existing_result "$dataset" "$size"; then
            echo "  [EXISTS] $dataset | n=$size"
            continue
        fi
        echo "  [RUN] $dataset | n=$size"
        python3 -m eval_scripts.eval_2afc \
            --models "$MODEL" \
            --n-images "$size" \
            --n-trials "$N_TRIALS" \
            --foil-type "$FOIL" \
            --dataset "$dataset" \
            --max-image-size "$MAX_IMG_SIZE" \
            --vision-chunk-size "$VISION_CHUNK_SIZE" \
            --attn "$ATTN" || echo "  [ERROR] $dataset | n=$size"
    done
done

echo "Done."
