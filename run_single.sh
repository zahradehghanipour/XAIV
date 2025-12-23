#!/usr/bin/env bash

instance_id=$1

BENCHMARK="vggnet16_benchmark2022_segmented"
CONFIG="abcrown/vggnet16.yaml"

experiment_path=results/$BENCHMARK

results_file="results/$BENCHMARK/results.csv"
tmp_results_file="result_${instance_id}.txt"



mkdir -p $experiment_path
if [[ ! -a $results_file ]]; then 
    # if file does not exist yet, create it with headers.
    echo "instance_id,onnx,vnnlib,timeout,result,lb_minus_rhs,domains_visited,bab_time,all_time,init_unstable" > $results_file
fi

# IFS="," read onnx vnnlib timeout <<< $(sed -n "${instance_id}p" benchmarks/$BENCHMARK/instances.csv)
line=$(sed -n "${instance_id}p" "benchmarks/$BENCHMARK/instances.csv")
IFS=',' read -r onnx vnnlib timeout <<< "$line"

# -- RUN αβ-CROWN
python ab-crown/complete_verifier/abcrown.py \
    --instance_id $instance_id \
    --config configs/$CONFIG \
    --onnx_path benchmarks/$BENCHMARK/$onnx \
    --vnnlib_path benchmarks/$BENCHMARK/$vnnlib \
    --timeout $timeout \
    --results_file $tmp_results_file \
    --output_additional_stats \
    --device "cpu"\
    --print_verbose_decisions\
    --view_model
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    # -- read αβ-CROWN result and save to the main results file, or log an error.
    result=$(cat $tmp_results_file)

    echo "$instance_id,$onnx,$vnnlib,$timeout,$result" >> $results_file
    # -- remove temporary αβ-CROWN result file
    rm $tmp_results_file
else
    echo "$instance_id,$onnx,$vnnlib,$timeout,error,,,,,," >> $results_file
fi
