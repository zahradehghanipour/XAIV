#!/bin/bash
#SBATCH --job-name=vggnet16_benchmark2022_one_no_attack_k500
#SBATCH --output=results/%x/logs/slurm-%j.out
#SBATCH --error=results/%x/logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --export=ALL

module purge
module load cuda/12.8

# --- MAIN VARIABLES
BENCHMARK="/home/z.dehghanipour/XAIV/benchmarks/vggnet16_benchmark2022_segmented_one_img_k500"
CWD="/home/z.dehghanipour/XAIV"
CONFIG="abcrown/vggnet16_no_attack.yaml"
CONDA_ENV_NAME="ab-crown-v1"

START_ID=1
END_ID=1

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
source activate /home/z.dehghanipour/.conda/envs/ab-crown-v1

python -c "import sys; print('PYTHON:', sys.executable)"

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