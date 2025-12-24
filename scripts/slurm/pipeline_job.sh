#!/bin/bash

#SBATCH --job-name=xaiv_pipeline
#SBATCH --output=results/%x/logs/slurm-%A_%a.out
#SBATCH --error=results/%x/logs/slurm-%A_%a.err
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --account=euhpc_d29_033

# Main knobs to adjust per cluster/use-case.
CONFIG="configs/xaiv/vggnet16_benchmark2022_segmented.yaml"  # Path to the pipeline config
CWD="$WORK/my_projects/XAIV"                                # Project root on the cluster
CONDA_ENV_NAME="xaiv"                                    # Conda env with deps installed

module load cuda/12.2

# Activate conda environment.
CONDA_BASE_PATH=$(conda info --base)
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Ensure we are in the project root.
cd "$CWD"

# Prepare per-job log directory (matches the SBATCH output paths above).
LOG_DIR="results/${SLURM_JOB_NAME:-xaiv_pipeline}/logs"
mkdir -p "$LOG_DIR"

echo "[SLURM] Running pipeline"
python scripts/pipeline.py
