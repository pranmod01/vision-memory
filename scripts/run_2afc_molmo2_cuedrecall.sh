#!/bin/bash
#SBATCH --job-name=2afc_molmo2_cuedrecall
#SBATCH --partition=burst
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# 2-AFC Recognition on maxbennett/cued-recall-imagenet (HF streaming).
# State foils not available; runs novel + exemplar + all.

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
# Streaming requires network access — do NOT enable offline mode.
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

MODEL="molmo2"
N_TRIALS=10
RESULTS_DIR="$SCRIPT_DIR/results"
read -r -a SIZES <<< "${SIZES:-1 5 10 50 100 250}"
read -r -a FOIL_TYPES <<< "${FOIL_TYPES:-novel exemplar all}"
DATASET="cued-recall-imagenet"

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local n_images="$1"
    local foil_type="$2"
    [ -f "$RESULTS_DIR/results_2afc_molmo2-8b_n${n_images}_${DATASET}_${foil_type}.json" ]
}

echo "========== 2-AFC Recognition: $MODEL | $DATASET =========="

for foil in "${FOIL_TYPES[@]}"; do
    for size in "${SIZES[@]}"; do
        if check_existing_result "$size" "$foil"; then
            echo "  [EXISTS] $foil | n=$size"
            continue
        fi
        echo "  [RUN] $foil | n=$size"
        python3 -m eval_scripts.eval_2afc \
            --models "$MODEL" \
            --n-images "$size" \
            --n-trials "$N_TRIALS" \
            --foil-type "$foil" \
            --dataset "$DATASET" || echo "  [ERROR] $foil | n=$size"
    done
done

echo "Done."
