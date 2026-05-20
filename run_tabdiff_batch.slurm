#!/bin/bash
#SBATCH --job-name=tabdiff-batch
#SBATCH --output=/sciclone/scr20/jpbach/slurm_logs/%x-%j.out
#SBATCH --error=/sciclone/scr20/jpbach/slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=hima
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hi07

set -euo pipefail

module load miniforge3/24.9.2-0
module load cuda/12.3
eval "$(conda shell.bash hook)"

WORKROOT="${WORKROOT:-/sciclone/scr20/jpbach}"
TABDIFF_REPO="${TABDIFF_REPO:-$HOME/TabDiff}"
TABDIFF_PYTHON="${TABDIFF_PYTHON:-/sciclone/scr20/jpbach/conda-envs/tabdiff/bin/python}"

export TMPDIR="$WORKROOT/tmp"
export PIP_CACHE_DIR="$WORKROOT/pip-cache"
export UV_CACHE_DIR="$WORKROOT/uv-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$WORKROOT/slurm_logs"

DATASETS="${DATASETS:-diabetes house income sick us_location}"
GPU="${GPU:-0}"
RUN_PROCESS="${RUN_PROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_REPORT="${RUN_REPORT:-0}"
RUN_Y_ONLY="${RUN_Y_ONLY:-0}"
RUN_IMPUTE="${RUN_IMPUTE:-0}"
NO_WANDB="${NO_WANDB:-1}"
NON_LEARNABLE_SCHEDULE="${NON_LEARNABLE_SCHEDULE:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
NUM_SAMPLES_TO_GENERATE="${NUM_SAMPLES_TO_GENERATE:-}"
NUM_RUNS="${NUM_RUNS:-20}"
EXP_NAME="${EXP_NAME:-}"
TRIAL_SIZE="${TRIAL_SIZE:-50}"
RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-1}"
IMPUTE_CONDITION="${IMPUTE_CONDITION:-x_t}"
W_NUM="${W_NUM:-0.6}"
W_CAT="${W_CAT:-0.6}"

echo "Host: $(hostname)"
nvidia-smi || true

"$TABDIFF_PYTHON" -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available())"

cd "$TABDIFF_REPO"
export PYTHONPATH="$TABDIFF_REPO:${PYTHONPATH:-}"

echo "Repo: $TABDIFF_REPO"
echo "Datasets: $DATASETS"
echo "Python: $TABDIFF_PYTHON"
echo "GPU arg: $GPU"
echo "RUN_PROCESS: $RUN_PROCESS"
echo "RUN_TRAIN: $RUN_TRAIN"
echo "RUN_TEST: $RUN_TEST"
echo "RUN_REPORT: $RUN_REPORT"
echo "RUN_Y_ONLY: $RUN_Y_ONLY"
echo "RUN_IMPUTE: $RUN_IMPUTE"

COMMON_ARGS=()
if [[ "$NO_WANDB" == "1" ]]; then
    COMMON_ARGS+=(--no_wandb)
fi
if [[ "$NON_LEARNABLE_SCHEDULE" == "1" ]]; then
    COMMON_ARGS+=(--non_learnable_schedule)
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
    COMMON_ARGS+=(--deterministic)
fi
if [[ -n "$EXP_NAME" ]]; then
    COMMON_ARGS+=(--exp_name "$EXP_NAME")
fi

for DATASET in $DATASETS; do
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    if [[ "$RUN_PROCESS" == "1" ]]; then
        "$TABDIFF_PYTHON" process_dataset.py --dataname "$DATASET"
    fi

    if [[ "$RUN_TRAIN" == "1" ]]; then
        "$TABDIFF_PYTHON" main.py \
          --dataname "$DATASET" \
          --mode train \
          --gpu "$GPU" \
          "${COMMON_ARGS[@]}"
    fi

    if [[ "$RUN_TEST" == "1" ]]; then
        TEST_ARGS=()
        if [[ "$RUN_REPORT" == "1" ]]; then
            TEST_ARGS+=(--report --num_runs "$NUM_RUNS")
        fi
        if [[ -n "$NUM_SAMPLES_TO_GENERATE" ]]; then
            TEST_ARGS+=(--num_samples_to_generate "$NUM_SAMPLES_TO_GENERATE")
        fi

        "$TABDIFF_PYTHON" main.py \
          --dataname "$DATASET" \
          --mode test \
          --gpu "$GPU" \
          "${COMMON_ARGS[@]}" \
          "${TEST_ARGS[@]}"
    fi

    if [[ "$RUN_Y_ONLY" == "1" ]]; then
        "$TABDIFF_PYTHON" main.py \
          --dataname "$DATASET" \
          --mode train \
          --gpu "$GPU" \
          --y_only \
          "${COMMON_ARGS[@]}"
    fi

    if [[ "$RUN_IMPUTE" == "1" ]]; then
        "$TABDIFF_PYTHON" main.py \
          --dataname "$DATASET" \
          --mode test \
          --gpu "$GPU" \
          --impute \
          --trial_size "$TRIAL_SIZE" \
          --resample_rounds "$RESAMPLE_ROUNDS" \
          --impute_condition "$IMPUTE_CONDITION" \
          --w_num "$W_NUM" \
          --w_cat "$W_CAT" \
          "${COMMON_ARGS[@]}"
    fi
done

unset PYTHONPATH
