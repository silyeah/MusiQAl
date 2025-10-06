#!/bin/bash     


#SBATCH --job-name=train_model 
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=accel  
#SBATCH --gpus=rtx30:1


## SBATCH --qos=devel


## SBATCH --output=./slurm_outs/test_avst_out_%j.txt

#SBATCH --output=./slurm_train_avst_out.txt

module purge
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# # module load Python/3.9.6-GCCcore-11.2.0 
source /fp/homes01/u01/ec-sarapje/MusiQAl/new_env/bin/activate


# # source /your/env/dir/avqaoriginal/bin/activate


DIR=/fp/homes01/u01/ec-sarapje/MusiQAl/AVST/net_grd_avst 
cd $DIR &> /dev/null

# #pwd > /fp/homes01/u01/ec-sarapje/MusiQAl/AVST/pwd.txt
# #python main_avst.py > ./output_test.txt

## export CUDA_LAUNCH_BLOCKING=1


python blank_train_avst.py > ./progress/blank_avst_train.txt





