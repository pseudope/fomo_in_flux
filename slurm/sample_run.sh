#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --output=<out_folder>/hostname_%j.out
#SBATCH --error=<out_folder>/hostname_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email-id>
#SBATCH --partition=<partition>

# print info about current job
scontrol show job $SLURM_JOB_ID

python main.py gpu=[0] experiment=<config>.yaml
