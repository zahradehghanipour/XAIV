#!/bin/bash -l
#SBATCH --job-name=vggnet16_all_p_3
#SBATCH --output=results/%x/logs/slurm-%j.out
#SBATCH --error=results/%x/logs/slurm-%j.err
#SBATCH --partition=sharing
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --export=ALL

# ----------------------------
# Safe mode (NO -u yet)
# ----------------------------
set -eo pipefail

# Ensure HOME exists (some non-interactive contexts can be weird)
export HOME="${HOME:-/home/z.dehghanipour}"

# ----------------------------
# Modules + conda init
#   (avoid /etc/profile in batch steps; just source modules.sh)
# ----------------------------
source /etc/profile.d/modules.sh || true
module purge
module load cuda/12.8

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

# ----------------------------
# Main variables
# ----------------------------
BENCHMARK="/projects/air/dlverifier/vggnet16_all"
CWD="/home/z.dehghanipour/XAIV"
CONFIG="abcrown/vggnet16.yaml"
CONDA_ENV_NAME="ab-crown-v1"

START_ID=600
END_ID=750

# Activate env
conda activate "/home/z.dehghanipour/.conda/envs/${CONDA_ENV_NAME}"

# Now strict mode is OK
set -euo pipefail

# ----------------------------
# Env + GPU checks
# ----------------------------
echo "Host: $(hostname)"
echo "Now (local): $(date)"
echo "Now (UTC):   $(date -u)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "HOME=${HOME:-<unset>}"

echo "---- nvidia-smi ----"
nvidia-smi -L || true
nvidia-smi || true

echo "---- python/torch ----"
which python
python -V
python - <<'PY'
import os, sys, torch
print("PYTHON:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device[0]:", torch.cuda.get_device_name(0))
    x = torch.randn(1024,1024, device="cuda")
    print("matmul ok:", float((x @ x).sum()))
PY

export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1  # only if debugging crashes

# ----------------------------
# Paths
# ----------------------------
mkdir -p "$CWD/results/$SLURM_JOB_NAME/logs" || true
cd "$CWD"

experiment_path="results/$SLURM_JOB_NAME"
mkdir -p "$experiment_path"

# ---------------------------------------
# Run worker(s) via srun
#   IMPORTANT: no /etc/profile sourcing here
# ---------------------------------------
srun --ntasks="${SLURM_NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" /bin/bash -lc "
set -eo pipefail
export HOME=\"\${HOME:-/home/z.dehghanipour}\"

# init modules/conda without /etc/profile to avoid bash_completion HOME issues
source /etc/profile.d/modules.sh || true
module purge
module load cuda/12.8
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate \"/home/z.dehghanipour/.conda/envs/${CONDA_ENV_NAME}\"

set -euo pipefail

rank=\"\${SLURM_PROCID}\"
world=\"\${SLURM_NTASKS}\"
echo \"[rank \${rank}/\${world}] host=\$(hostname) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>} HOME=\${HOME:-<unset>} python=\$(which python)\"

experiment_path=\"${experiment_path}\"
BENCHMARK=\"${BENCHMARK}\"
CONFIG=\"${CONFIG}\"
START_ID=\"${START_ID}\"
END_ID=\"${END_ID}\"

part_csv=\"\${experiment_path}/results.part.\${rank}.csv\"
if [[ ! -f \"\${part_csv}\" ]]; then
  echo \"instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable\" > \"\${part_csv}\"
fi

done_ids=\"\${experiment_path}/done_ids.\${rank}.txt\"
: > \"\${done_ids}\"

if [[ -f \"\${experiment_path}/results.csv\" ]]; then
  awk -F',' 'NR>1 && \$1 ~ /^[0-9]+$/ {print \$1}' \"\${experiment_path}/results.csv\" >> \"\${done_ids}\"
fi
for f in \"\${experiment_path}\"/results.part.*.csv; do
  [[ -f \"\$f\" ]] || continue
  awk -F',' 'NR>1 && \$1 ~ /^[0-9]+$/ {print \$1}' \"\$f\" >> \"\${done_ids}\"
done
sort -n -u \"\${done_ids}\" -o \"\${done_ids}\"

id_done () { local id=\"\$1\"; grep -qx \"\${id}\" \"\${done_ids}\"; }
mark_done () { local id=\"\$1\"; echo \"\${id}\" >> \"\${done_ids}\"; }

# If instances.csv has header, set HAS_HEADER=1
HAS_HEADER=0

for ID in \$(seq \"\${START_ID}\" \"\${END_ID}\"); do
  if (( (ID - START_ID) % world != rank )); then
    continue
  fi

  echo \"[rank \${rank}] === ID \${ID} ===\"

  if id_done \"\${ID}\"; then
    echo \"[rank \${rank}] >>> already done — skipping\"
    continue
  fi

  tmp_results_file=\"\${experiment_path}/result_\${ID}.rank\${rank}.txt\"

  line=\$((ID + HAS_HEADER))
  onnx=\$(awk -F',' -v ln=\"\${line}\" 'NR==ln {print \$1}' \"\${BENCHMARK}/instances.csv\")
  vnnlib=\$(awk -F',' -v ln=\"\${line}\" 'NR==ln {print \$2}' \"\${BENCHMARK}/instances.csv\")
  timeout=\$(awk -F',' -v ln=\"\${line}\" 'NR==ln {print \$3}' \"\${BENCHMARK}/instances.csv\")

  if [[ -z \"\${onnx}\" || -z \"\${vnnlib}\" || -z \"\${timeout}\" ]]; then
    echo \"[rank \${rank}] ERROR: could not read line \${line} (ID \${ID}) from instances.csv\" >&2
    echo \"\${ID},,,,error,,,,,\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
    continue
  fi

  echo \"[rank \${rank}] >>> Running onnx=\${onnx}\"

  set +e
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
  set -e

  if [[ \${exit_code} -eq 0 && -f \"\${tmp_results_file}\" ]]; then
    result=\$(cat \"\${tmp_results_file}\")
    echo \"\${ID},\${onnx},\${vnnlib},\${timeout},\${result}\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
    rm -f \"\${tmp_results_file}\"
  else
    echo \"\${ID},\${onnx},\${vnnlib},\${timeout},error,,,,,\" >> \"\${part_csv}\"
    mark_done \"\${ID}\"
    rm -f \"\${tmp_results_file}\" || true
  fi
done

echo \"[rank \${rank}/\${world}] done.\"
"

# -------------------------
# Merge partial CSVs
# -------------------------
final_csv="$experiment_path/results.csv"
echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > "$final_csv"
awk 'FNR==1{next} {print}' "$experiment_path"/results.part.*.csv \
  | sort -t',' -k1,1n >> "$final_csv"

echo "Merged -> $final_csv"