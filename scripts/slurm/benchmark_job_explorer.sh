#!/bin/bash
#SBATCH --job-name=vggnet16_benchmark2022_one_img
#SBATCH --output=results/%x/logs/slurm-%j_%a.out
#SBATCH --error=results/%x/logs/slurm-%j_%a.err
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --array=1
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
BENCHMARK="/projects/air/dlverifier"
CWD="/home/z.dehghanipour/XAIV"
CONFIG="abcrown/vggnet16.yaml"   
CONDA_ENV_NAME="ab-crown-v1"
ID=5700

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
source activate /home/z.dehghanipour/.conda/envs/ab-crown-v1

python -c "import sys; print('PYTHON:', sys.executable)"

# Ensure logs directory exists because SBATCH writes there
mkdir -p "$CWD/results/$SLURM_JOB_NAME/logs" || true

cd "$CWD"

experiment_path="results/$SLURM_JOB_NAME"
mkdir -p "$experiment_path"

results_file="$experiment_path/results.csv"
if [[ ! -f "$results_file" ]]; then
  echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > "$results_file"
fi

tmp_results_file="$experiment_path/result_$ID.txt"

# Safer CSV read for first 3 comma-separated fields
onnx=$(awk -F',' -v id="$ID" 'NR==id {print $1}' "$BENCHMARK/instances.csv")
vnnlib=$(awk -F',' -v id="$ID" 'NR==id {print $2}' "$BENCHMARK/instances.csv")
timeout=$(awk -F',' -v id="$ID" 'NR==id {print $3}' "$BENCHMARK/instances.csv")

if [[ -z "${onnx}" || -z "${vnnlib}" || -z "${timeout}" ]]; then
  echo "ERROR: could not read line $ID from $BENCHMARK/instances.csv" >&2
  exit 2
fi
