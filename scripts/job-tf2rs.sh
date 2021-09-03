#!/bin/bash
#SBATCH --job-name=tf2rs
#SBATCH --time=96:00:00
#SBATCH -N 1
#SBATCH -p normal

# NBANKS=`nproc --all`
NBANKS=16
echo $NBANKS

for i in $(seq 1 $NBANKS); do
	python taylorf2reducedspin.py --seed $i --kind random &
done

# python taylorf2reducedspin.py --seed $NBANKS --kind stochastic &

wait
