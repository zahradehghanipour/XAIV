#!/bin/bash
#SBATCH --job-name=vggnet16_8_ten_imgs
#SBATCH --output=results/%x/logs/slurm-%j.out
#SBATCH --error=results/%x/logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --export=ALL

module purge
module load cuda/12.8

# --- MAIN VARIABLES
BENCHMARK="/projects/air/dlverifier/vggnet16_8_ten_imgs"
CWD="/home/z.dehghanipour/XAIV"
CONFIG="abcrown/vggnet16.yaml"
CONDA_ENV_NAME="ab-crown-v1"

START_ID=1
END_ID=162

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
source activate /home/z.dehghanipour/.conda/envs/ab-crown-v1

echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_GPUS=$SLURM_GPUS"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

echo "========== ENV CHECK =========="
which python
python -V
python -c "import sys; print('PYTHON:', sys.executable)"

python - <<'PY'
import os, sys, torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    a = torch.randn(1024,1024, device="cuda")
    b = (a @ a).sum()
    print("matmul ok:", float(b))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

nvidia-smi
echo "================================"

# Debug for real CUDA error location
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Ensure logs directory exists
mkdir -p "$CWD/results/$SLURM_JOB_NAME/logs" || true

cd "$CWD"

experiment_path="results/$SLURM_JOB_NAME"
mkdir -p "$experiment_path"

results_file="$experiment_path/results.csv"
if [[ ! -f "$results_file" ]]; then
  echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > "$results_file"
fi

id_exists () {
  local id="$1"
  awk -F',' -v id="$id" 'NR>1 && $1==id {found=1} END {exit !found}' "$results_file"
}

# ============================
# MAIN LOOP
# ============================
for ID in $(seq "$START_ID" "$END_ID"); do
  echo "=== Checking instance ID $ID ==="

  # --------------------------------
  # SKIP IF ALREADY IN RESULTS
  # --------------------------------
  if id_exists "$ID"; then
    echo ">>> ID $ID already exists in results.csv — skipping"
    continue
  fi

  echo ">>> Running instance ID $ID"

  tmp_results_file="$experiment_path/result_$ID.txt"

  # Read CSV fields
  onnx=$(awk -F',' -v id="$ID" 'NR==id {print $1}' "$BENCHMARK/instances.csv")
  vnnlib=$(awk -F',' -v id="$ID" 'NR==id {print $2}' "$BENCHMARK/instances.csv")
  timeout=$(awk -F',' -v id="$ID" 'NR==id {print $3}' "$BENCHMARK/instances.csv")

  if [[ -z "${onnx}" || -z "${vnnlib}" || -z "${timeout}" ]]; then
    echo "ERROR: could not read line $ID from instances.csv" >&2
    echo "$ID,,,,error,,,,," >> "$results_file"
    continue
  fi

  python ab-crown/complete_verifier/abcrown.py \
    --instance_id "$ID" \
    --config "configs/$CONFIG" \
    --onnx_path "$BENCHMARK/$onnx" \
    --vnnlib_path "$BENCHMARK/$vnnlib" \
    --timeout "$timeout" \
    --results_file "$tmp_results_file" \
    --output_additional_stats \
    --device "cuda"

  exit_code=$?

  if [[ $exit_code -eq 0 && -f "$tmp_results_file" ]]; then
    result=$(cat "$tmp_results_file")
    echo "$ID,$onnx,$vnnlib,$timeout,$result" >> "$results_file"
    rm -f "$tmp_results_file"
  else
    echo "$ID,$onnx,$vnnlib,$timeout,error,,,,," >> "$results_file"
  fi
done