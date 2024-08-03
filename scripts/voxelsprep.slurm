#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=0803_vp
#SBATCH --output=0803_vp.out
#SBATCH --error=0803_vp.err

#SBATCH --cpus-per-task=1
# #SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=100gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=08:00:00

module load Python
conda activate /scratch/jahernando/conda-env/pytorch 
module load CUDA/12.1

source /home/jahernando/scripts/setup.sh
cd /home/jahernando/NEXT/software/nextnn
source setup.sh

srun python xyimg/voxelsprod.py -pressure 5bar -nbunch 1000 -nevents -1
