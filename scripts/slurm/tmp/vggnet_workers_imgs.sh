#!/bin/bash
#SBATCH --job-name=vggnet16_10_ten_imgs
#SBATCH --output=results/%x/logs/slurm-%j.out
#SBATCH --error=results/%x/logs/slurm-%j.err
#SBATCH --partition=gpu

# =========================
# START WITH 1 GPU (test)
# Later: change a100:1 -> a100:4 and ntasks=1 -> ntasks=4
# =========================
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --export=ALL

# Avoid nounset during conda hooks
set -eo pipefail

module purge
module load cuda/12.8

# --- MAIN VARIABLES
BENCHMARK="/projects/air/dlverifier/vggnet16_10_ten_imgs"
CWD="/home/z.dehghanipour/XAIV"
CONFIG="abcrown/vggnet16.yaml"
CONDA_ENV_NAME="ab-crown-v1"

START_ID=1
END_ID=180

# Conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate "/home/z.dehghanipour/.conda/envs/${CONDA_ENV_NAME}"

# Now strict mode is OK, but use ${VAR:-...} for possibly-unset env vars
set -euo pipefail

echo "----- SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SHELL=${SHELL:-<unset>}"
echo "PATH=${PATH:-<unset>}"

# Ensure logs directory exists
mkdir -p "$CWD/results/$SLURM_JOB_NAME/logs" || true
cd "$CWD"

experiment_path="results/$SLURM_JOB_NAME"
mkdir -p "$experiment_path"

echo "========== ENV CHECK =========="
which python
python -V
python -c "import sys; print('PYTHON:', sys.executable)"
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("device[0]:", torch.cuda.get_device_name(0))
    print("capability[0]:", torch.cuda.get_device_capability(0))
    a = torch.randn(1024,1024, device="cuda")
    b = (a @ a).sum()
    print("matmul ok:", float(b))
PY
nvidia-smi
echo "================================"

# Debug flags:
# Keep CUDA_LAUNCH_BLOCKING OFF for speed. Enable only when debugging crashes.
# export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

echo "========== QUICK GPU CHECK =========="
nvidia-smi
echo "====================================="

# ---------------------------------------
# Run workers via srun (robust: no exported bash functions)
# ---------------------------------------
srun --ntasks="${SLURM_NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" /bin/bash -lc "
set -euo pipefail

rank=\"\${SLURM_PROCID}\"
world=\"\${SLURM_NTASKS}\"

echo \"[rank \${rank}/\${world}] host=\$(hostname) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"

experiment_path=\"${experiment_path}\"
BENCHMARK=\"${BENCHMARK}\"
CONFIG=\"${CONFIG}\"
START_ID=\"${START_ID}\"
END_ID=\"${END_ID}\"

# Per-rank output (avoid write races)
part_csv=\"\${experiment_path}/results.part.\${rank}.csv\"
if [[ ! -f \"\${part_csv}\" ]]; then
  echo \"instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable\" > \"\${part_csv}\"
fi

# --------------------------------------------
# Build a global DONE list from existing results:
#   - results.csv (previous merged runs)
#   - any results.part.*.csv (previous partial runs)
# Then skip IDs found in that set.
# --------------------------------------------
done_ids=\"\${experiment_path}/done_ids.\${rank}.txt\"
: > \"\${done_ids}\"

# From merged results.csv
if [[ -f \"\${experiment_path}/results.csv\" ]]; then
  awk -F',' 'NR>1 && \$1 ~ /^[0-9]+$/ {print \$1}' \"\${experiment_path}/results.csv\" >> \"\${done_ids}\"
fi

# From any partials (including this rank's from previous runs)
for f in \"\${experiment_path}\"/results.part.*.csv; do
  [[ -f \"\$f\" ]] || continue
  awk -F',' 'NR>1 && \$1 ~ /^[0-9]+$/ {print \$1}' \"\$f\" >> \"\${done_ids}\"
done

# Deduplicate
sort -n -u \"\${done_ids}\" -o \"\${done_ids}\"

id_done () {
  local id=\"\$1\"
  grep -qx \"\${id}\" \"\${done_ids}\"
}

mark_done () {
  local id=\"\$1\"
  echo \"\${id}\" >> \"\${done_ids}\"
}

for ID in \$(seq \"\${START_ID}\" \"\${END_ID}\"); do
  # round-robin distribution
  if (( (ID - START_ID) % world != rank )); then
    continue
  fi

  echo \"[rank \${rank}] === Checking instance ID \${ID} ===\"

  # Skip if already done in ANY previous output
  if id_done \"\${ID}\"; then
    echo \"[rank \${rank}] >>> ID \${ID} already done (results.csv / parts) — skipping\"
    continue
  fi

  tmp_results_file=\"\${experiment_path}/result_\${ID}.rank\${rank}.txt\"

  # NOTE: If instances.csv has a header row, use NR==id+1 instead of NR==id.
  onnx=\$(awk -F',' -v id=\"\${ID}\" 'NR==id {print \$1}' \"\${BENCHMARK}/instances.csv\")
  vnnlib=\$(awk -F',' -v id=\"\${ID}\" 'NR==id {print \$2}' \"\${BENCHMARK}/instances.csv\")
  timeout=\$(awk -F',' -v id=\"\${ID}\" 'NR==id {print \$3}' \"\${BENCHMARK}/instances.csv\")

  if [[ -z \"\${onnx}\" || -z \"\${vnnlib}\" || -z \"\${timeout}\" ]]; then
    echo \"[rank \${rank}] ERROR: could not read line \${ID} from instances.csv\" >&2
    echo \"\${ID},,,,error,,,,,\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
    continue
  fi

  echo \"[rank \${rank}] >>> Running ID=\${ID} onnx=\${onnx}\"

  python ab-crown/complete_verifier/abcrown.py \
    --instance_id \"\${ID}\" \
    --config \"configs/\${CONFIG}\" \
    --onnx_path \"\${BENCHMARK}/\${onnx}\" \
    --vnnlib_path \"\${BENCHMARK}/\${vnnlib}\" \
    --timeout \"\${timeout}\" \
    --results_file \"\${tmp_results_file}\" \
    --output_additional_stats \
    --device \"cuda\"

  exit_code=\$?

  if [[ \${exit_code} -eq 0 && -f \"\${tmp_results_file}\" ]]; then
    result=\$(cat \"\${tmp_results_file}\")
    echo \"\${ID},\${onnx},\${vnnlib},\${timeout},\${result}\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
    rm -f \"\${tmp_results_file}\"
  else
    echo \"\${ID},\${onnx},\${vnnlib},\${timeout},error,,,,,\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
  fi
done

echo \"[rank \${rank}/\${world}] done.\"
"

# -------------------------
# Merge partial CSVs after srun completes
# -------------------------
final_csv="$experiment_path/results.csv"
echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > "$final_csv"

# Concatenate without headers and sort by instance_id numeric
awk 'FNR==1{next} {print}' "$experiment_path"/results.part.*.csv \
  | sort -t',' -k1,1n >> "$final_csv"

echo "Merged -> $final_csv"