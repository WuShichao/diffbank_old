#!/bin/bash
#SBATCH --job-name=3pn-coverage
#SBATCH --account=rrg-lplevass
#SBATCH --time=00-10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=47000M

source ~/.virtualenvs/diffbank-3.9.6/bin/activate

START=100

for i in $(seq $START $(($START + 30))); do
    python genbank_2D_threePN.py --seed $i --kind random --device gpu --n-eff 500 --noise analytic
done

wait
