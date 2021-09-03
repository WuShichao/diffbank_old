#!/bin/bash
#SBATCH --job-name=tf2rs
#SBATCH --time=96:00:00
#SBATCH -N 1
#SBATCH -p normal

NPROC=`nproc --all`
echo $NPROC

for i in $(seq 1 $(($NPROC - 1))); do
	python3 taylorf2reducedspin.py --seed $(($i + 16)) --kind random &
done

# python3 taylorf2reducedspin.py --seed $NPROC --kind stochastic &

wait
