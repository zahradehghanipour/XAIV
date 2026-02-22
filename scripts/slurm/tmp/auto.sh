#!/bin/bash

while true; do
    echo "Submitting jobs at $(date)"

    sbatch scripts/slurm/tmp/vggnet_all_p_1_sharing.sh
    sbatch scripts/slurm/tmp/vggnet_all_p_2_sharing.sh
    sbatch scripts/slurm/tmp/vggnet_all_p_3_sharing.sh
    sbatch scripts/slurm/tmp/vggnet_all_p_4_sharing.sh

    # sbatch scripts/slurm/tmp/vggnet_all_p_1_sharing_2.sh
    # sbatch scripts/slurm/tmp/vggnet_all_p_2_sharing_2.sh
    # sbatch scripts/slurm/tmp/vggnet_all_p_3_sharing_2.sh
    # sbatch scripts/slurm/tmp/vggnet_all_p_4_sharing_2.sh

    # sbatch scripts/slurm/tmp/vggnet_all_p_1_sharing_3.sh  
    # sbatch scripts/slurm/tmp/vggnet_all_p_2_sharing_3.sh  
    # sbatch scripts/slurm/tmp/vggnet_all_p_3_sharing_3.sh  
    # sbatch scripts/slurm/tmp/vggnet_all_p_4_sharing_3.sh  

    echo "Sleeping for 1 hour..."
    sleep 3700
done