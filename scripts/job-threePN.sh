#!/bin/bash
#SBATCH --job-name=threePN
#SBATCH --account=rrg-lplevass
#SBATCH --time=00-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=47000M

# START=1
# NPROC=`nproc --all`
# echo $NPROC
# echo $START
# echo $(($NPROC + $START - 1))
# 
# for i in $(seq $START $(($NPROC + $START - 1))); do
#     python threePN.py --seed $i --kind stochastic &
# done
# 
# wait

source ~/.virtualenvs/diffbank-3.9.6/bin/activate

python genbank_2D_threePN.py --seed 1 --kind random --device gpu
