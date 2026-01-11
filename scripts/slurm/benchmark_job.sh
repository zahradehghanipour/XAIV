#!/bin/bash
#SBATCH --job-name=vggnet16_benchmark2022_segmented_100_img
#SBATCH --output=results/%x/logs/slurm-%j_%a.out
#SBATCH --array=1047-1050
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL
#SBATCH --time=01:30:00
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --account=euhpc_d29_033

module load cuda/12.2                         # Load CUDA toolkit
#module load openmpi                           # Load MPI implementation

# NOTE usage
# 0. modify the sbatch config for your slurm cluster
# 1. it is important to set the job name above
# 2. set the --array to the range of instances to verify
# 3. set main variables appropriately

# -- MAIN VARIABLES
BENCHMARK="vggnet16_benchmark2022_segmented_100_img"
CONFIG="abcrown/vggnet16.yaml"
CWD="$WORK/my_projects/XAIV"
CONDA_ENV_NAME="abcrown"

# -- load conda and activate environment
CONDA_BASE_PATH=$(conda info --base)
source $CONDA_BASE_PATH/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# -- ensure we are in the correct working directory
cd $CWD

# -- make output directory for current job if it does not exist
experiment_path=results/$SLURM_JOB_NAME
mkdir -p $experiment_path

# -- path to the result file
results_file="$experiment_path/results.csv"
if [[ ! -a $results_file ]]; then 
    # if file does not exist yet, create it with headers.
    echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > $results_file
fi

# -- a file to temporarily stores the αβ-CROWN result
tmp_results_file="$experiment_path/result_$SLURM_ARRAY_TASK_ID.txt"

# -- read benchmark instances.csv, get line corresponding to task_id, to get model and initial vnnlib
IFS="," read onnx vnnlib timeout <<< $(sed -n "${SLURM_ARRAY_TASK_ID}p" benchmarks/$BENCHMARK/instances.csv)

# -- RUN αβ-CROWN
python ab-crown/complete_verifier/abcrown.py \
    --instance_id "$SLURM_ARRAY_TASK_ID" \
    --config configs/$CONFIG \
    --onnx_path benchmarks/$BENCHMARK/$onnx \
    --vnnlib_path benchmarks/$BENCHMARK/$vnnlib \
    --timeout $timeout \
    --results_file $tmp_results_file \
    --output_additional_stats \
    --device "cuda"
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    # -- read αβ-CROWN result and save to the main results file, or log an error.
    result=$(cat $tmp_results_file)
    echo "$SLURM_ARRAY_TASK_ID,$onnx,$vnnlib,$timeout,$result" >> $results_file
    # -- remove temporary αβ-CROWN result file
    rm $tmp_results_file
else
    echo "$SLURM_ARRAY_TASK_ID,$onnx,$vnnlib,$timeout,error,,,,," >> $results_file
fi