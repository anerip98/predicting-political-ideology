#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=gpu1bs32
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=./slurm_output/slurm_%j_1_rtx8000.out

module purge 

singularity exec --nv \
    --overlay /scratch/tj810/intro-dl-system/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
    /bin/bash -c "source /ext3/env.sh; python reddit.py --gpu=1 --type=rtx8000"
