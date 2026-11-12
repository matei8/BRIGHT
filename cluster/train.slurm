#!/bin/bash
# Train (+ evaluate) one BRIGHT run on the cluster. Submit FROM THE REPO ROOT:
#
#   sbatch cluster/train.slurm <run-name> [extra train.py args...]
#   sbatch cluster/train.slurm unet_a_s42
#   sbatch cluster/train.slurm unet_a_s43 --set train.seed=43
#   sbatch -p dgxa100 cluster/train.slurm unet_a_fp32 --set runtime.amp=none
#
# Extra args are passed to train.py AND evaluate.py, so use --set (not --epochs etc.).
#
# Required environment (export in ~/.bashrc on fep, or prefix the sbatch call):
#   BRIGHT_DATA_ROOT   shared path holding pre-event/ post-event/ target/
#   BRIGHT_SIF         path to bright.sif   (default: ~/bright.sif)
#
# Always runs train.py with --resume: a fresh run starts from scratch, an interrupted
# one continues from checkpoints/<run>/last.pth. If the time limit approaches, SLURM's
# early SIGTERM makes train.py stop cleanly (exit 3) and this script resubmits itself.
# When training finishes, best.pth is evaluated on val and test.
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00     # student account max is 12:00:00; a full run on H100 is ~2 h
#SBATCH --signal=B:TERM@300
#SBATCH --job-name=bright
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
RUN_NAME="${1:?usage: sbatch cluster/train.slurm <run-name> [train.py args]}"; shift
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="${BRIGHT_SIF:-$HOME/bright.sif}"
SRC_DATA="${BRIGHT_DATA_ROOT:?set BRIGHT_DATA_ROOT to the shared dataset folder}"
cd "$REPO"

echo "job ${SLURM_JOB_ID:-?} on $(hostname)  partition ${SLURM_JOB_PARTITION:-?}  run $RUN_NAME"
echo "commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- stage the dataset to node-local disk (21 GB); fall back to the shared copy ---------
DATA="$SRC_DATA"
LOCAL="${TMPDIR:-/tmp}/${USER}_bright_data"
need_kb=$(( 23 * 1024 * 1024 ))
free_kb=$(df -Pk "$(dirname "$LOCAL")" | awk 'NR==2{print $4}')
if [ "${BRIGHT_NO_STAGE:-0}" != "1" ] && [ "${free_kb:-0}" -gt "$need_kb" ]; then
    echo "staging data to $LOCAL ..."; t0=$(date +%s)
    mkdir -p "$LOCAL"
    if rsync -a "$SRC_DATA"/{pre-event,post-event,target} "$LOCAL"/; then
        DATA="$LOCAL"; echo "staged in $(( $(date +%s) - t0 ))s"
    else
        echo "WARNING: staging failed, reading from shared storage"
    fi
else
    echo "WARNING: not staging (free ${free_kb:-?} KB on $(dirname "$LOCAL")), reading from shared storage"
fi

# `exec` replaces the calling subshell, so the PID we hold is the container/python process
# itself (needed to forward SIGTERM). Always call run_py in a subshell: `run_py .. &` or `(run_py ..)`.
run_py() {
    exec apptainer exec --nv --bind "$REPO" --bind "$DATA" --bind "$SRC_DATA" \
        --env BRIGHT_DATA_ROOT="$DATA" "$SIF" python "$@"
}

# --- train. In the background so the batch shell can forward SLURM's early SIGTERM. ----
run_py scripts/train.py --env cluster --run-name "$RUN_NAME" --resume "$@" &
child=$!
trap 'echo "SIGTERM from SLURM, forwarding to training"; kill -TERM "$child" 2>/dev/null' TERM
wait "$child"; rc=$?
if [ $rc -gt 128 ]; then wait "$child"; rc=$?; fi   # wait was interrupted by the trap
trap - TERM

if [ $rc -eq 3 ]; then
    echo "stopped before the end: resubmitting to resume"
    sbatch -p "${SLURM_JOB_PARTITION}" cluster/train.slurm "$RUN_NAME" "$@"
    exit $rc
fi
[ $rc -ne 0 ] && { echo "training failed with exit code $rc"; exit $rc; }

# --- final report with best.pth (full 1024 tiles) --------------------------------------
for split in val test; do
    (run_py scripts/evaluate.py --env cluster --run "$RUN_NAME" --split "$split" --ckpt best "$@") || exit $?
done
