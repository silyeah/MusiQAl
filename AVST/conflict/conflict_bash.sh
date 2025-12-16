#!/bin/bash     


#SBATCH --job-name=trial0
#SBATCH --account=ec12
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=accel  
#SBATCH --gpus=rtx30:1


##SBATCH --qos=devel

## SBATCH --output=./slurm_outs/test_avst_out_%j.txt

#SBATCH --output=./slurm_test_avst_out.txt

module purge
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# # module load Python/3.9.6-GCCcore-11.2.0 
# # source /fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/new_env/bin/activate

source /fp/homes01/u01/ec-hallvaih/IN5490env/bin/activate

# # source /your/env/dir/avqaoriginal/bin/activate

DIR=/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/net_grd_avst 
cd $DIR &> /dev/null

# #pwd > /fp/homes01/u01/ec-sarapje/MusiQAl/AVST/pwd.txt
python conflict_avst.py --conflict_mode both --num_runs 5 --seed 42 > ./output_test_conflict.txt

## export CUDA_LAUNCH_BLOCKING=1

# python avst_test.py > ./avst_test.txt

# python avst_test_success.py > ./avst_test_success.txt
