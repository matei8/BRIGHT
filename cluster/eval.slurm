#!/bin/bash
# Evaluate an existing checkpoint on the cluster (no training). Submit FROM THE REPO ROOT:
#
#   sbatch cluster/eval.slurm <run-name> [evaluate.py args...]
#   sbatch cluster/eval.slurm unet_a_s42 --blank sar          # modality ablation, test split
#   sbatch cluster/eval.slurm unet_b_s42 --save-preds         # write predicted class maps
#
# Defaults: --split test --ckpt best --env cluster. Reads the data directly from the shared
# storage (one pass over 878 tiles, not worth staging). Results go to results/<run>/.
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=bright-eval
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
RUN_NAME="${1:?usage: sbatch cluster/eval.slurm <run-name> [evaluate.py args]}"; shift
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="${BRIGHT_SIF:-$HOME/bright.sif}"
DATA="${BRIGHT_DATA_ROOT:?set BRIGHT_DATA_ROOT to the shared dataset folder}"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-?} on $(hostname)  run $RUN_NAME  args: $*"
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"

apptainer exec --nv --bind "$REPO" --bind "$DATA" --env BRIGHT_DATA_ROOT="$DATA" "$SIF" \
    python scripts/evaluate.py --env cluster --run "$RUN_NAME" --split test --ckpt best "$@"
