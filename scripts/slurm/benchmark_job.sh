#!/bin/bash
#SBATCH --job-name=segmented_imagenet
#SBATCH --output=results/%x/logs/slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=z.dehghanipour@northeastern.edu
#SBATCH --account=euhpc_d29_033

# ============================
# CONFIG
# ============================
START_ID=10532
END_ID=42238

BENCHMARK="vggnet16_benchmark2022_segmented"
CONFIG="abcrown/vggnet16.yaml"
CWD="$WORK/my_projects/XAIV"
CONDA_ENV_NAME="abcrown"

# ============================
# ENV SETUP
# ============================
module load cuda/12.2

CONDA_BASE_PATH=$(conda info --base)
source $CONDA_BASE_PATH/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

cd $CWD

# ============================
# OUTPUT SETUP
# ============================
experiment_path=results/$SLURM_JOB_NAME
mkdir -p $experiment_path

results_file="$experiment_path/results.csv"

if [[ ! -f $results_file ]]; then
    echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > $results_file
fi

# ============================
# MAIN LOOP
# ============================
for ((ID=$START_ID; ID<=$END_ID; ID++)); do
    echo "▶ Processing instance $ID"

    tmp_results_file="$experiment_path/result_${ID}.txt"

    IFS="," read onnx vnnlib timeout <<< $(sed -n "${ID}p" benchmarks/$BENCHMARK/instances.csv)

    if [[ -z "$onnx" ]]; then
        echo "Skipping empty line $ID"
        continue
    fi

    python ab-crown/complete_verifier/abcrown.py \
        --instance_id "$ID" \
        --config configs/$CONFIG \
        --onnx_path benchmarks/$BENCHMARK/$onnx \
        --vnnlib_path benchmarks/$BENCHMARK/$vnnlib \
        --timeout $timeout \
        --results_file $tmp_results_file \
        --output_additional_stats \
        --device cuda

    exit_code=$?

    if [[ $exit_code -eq 0 && -f $tmp_results_file ]]; then
        result=$(cat $tmp_results_file)
        echo "$ID,$onnx,$vnnlib,$timeout,$result" >> $results_file
        rm -f $tmp_results_file
    else
        echo "$ID,$onnx,$vnnlib,$timeout,error,,,,," >> $results_file
    fi

done

echo "Finished processing instances $START_ID → $END_ID"