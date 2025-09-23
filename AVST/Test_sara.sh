#!/bin/bash     


#SBATCH --job-name=trial0
#SBATCH --account=ec54 
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=accel  
#SBATCH --gres=gpu:1
##SBATCH --qos=devel
# #SBATCH --mem=32G      
# #SBATCH --gpus=a100:1
# #SBATCH --gpus=rtx30:1

# #SBATCH --output=./slurm_outs/test_avst_out_%j.txt

#SBATCH --output=./test_avst_out.txt

module purge
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.9.6-GCCcore-11.2.0
source /fp/homes01/u01/ec-sarapje/MusiQAl/compat_env/bin/activate


# # source /your/env/dir/avqaoriginal/bin/activate


DIR=/fp/homes01/u01/ec-sarapje/MusiQAl/AVST/net_grd_avst 
cd $DIR &> /dev/null

# #pwd > /fp/homes01/u01/ec-sarapje/MusiQAl/AVST/pwd.txt
# #python main_avst.py > ./output_test.txt

python main_avst.py > ./avst_test.txt




