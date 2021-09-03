#!/bin/bash
#SBATCH --job-name=3PN
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -p shared

START=1
NPROC=`nproc --all`
echo $NPROC
echo $START
echo $(($NPROC + $START - 1))

for i in $(seq $START $(($NPROC + $START - 1))); do
    python threePN.py --seed $i --kind stochastic &
done

wait
