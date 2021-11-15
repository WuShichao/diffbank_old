#!/bin/bash
#SBATCH --job-name=p
#SBATCH --time=01:00:00
#SBATCH -n 6
#SBATCH -p normal

echo "Estimating p"
python threePN_est_p.py --seed 0 --mm 0.95 --n 20000 2>&1 | tee -a threePN-p.txt &
python threePN_est_p.py --seed 1 --mm 0.90 --n 20000 2>&1 | tee -a threePN-p.txt &
python threePN_est_p.py --seed 2 --mm 0.85 --n 20000 2>&1 | tee -a threePN-p.txt &
python threePN_est_p.py --seed 3 --mm 0.80 --n 20000 2>&1 | tee -a threePN-p.txt &
python threePN_est_p.py --seed 4 --mm 0.75 --n 20000 2>&1 | tee -a threePN-p.txt &
wait
