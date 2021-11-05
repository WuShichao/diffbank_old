echo "Random, varying mm"
python threePN.py --seed 5 --kind random --mm 0.95 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-5-mm=0.95-eta_star=0.9.txt &
python threePN.py --seed 6 --kind random --mm 0.90 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-6-mm=0.90-eta_star=0.9.txt &
python threePN.py --seed 7 --kind random --mm 0.85 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-7-mm=0.85-eta_star=0.9.txt &
python threePN.py --seed 8 --kind random --mm 0.80 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-8-mm=0.80-eta_star=0.9.txt &
python threePN.py --seed 9 --kind random --mm 0.75 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-9-mm=0.75-eta_star=0.9.txt &
echo "Stochastic, varying mm"
python threePN.py --seed 10 --kind stochastic --mm 0.95 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-10-mm=0.95-eta_star=0.9.txt &
python threePN.py --seed 11 --kind stochastic --mm 0.90 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-11-mm=0.90-eta_star=0.9.txt &
python threePN.py --seed 12 --kind stochastic --mm 0.85 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-12-mm=0.85-eta_star=0.9.txt &
python threePN.py --seed 13 --kind stochastic --mm 0.80 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-13-mm=0.80-eta_star=0.9.txt &
python threePN.py --seed 14 --kind stochastic --mm 0.75 --eta-star 0.9 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-14-mm=0.75-eta_star=0.9.txt &
wait
echo "Done varying mm"

echo "Random, varying eta_star"
python threePN.py --seed 15 --kind random --mm 0.90 --eta-star 0.975 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-15-mm=0.90-eta_star=0.975.txt &
python threePN.py --seed 16 --kind random --mm 0.90 --eta-star 0.950 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-16-mm=0.90-eta_star=0.950.txt &
python threePN.py --seed 17 --kind random --mm 0.90 --eta-star 0.925 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-17-mm=0.90-eta_star=0.925.txt &
python threePN.py --seed 18 --kind random --mm 0.90 --eta-star 0.900 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-18-mm=0.90-eta_star=0.900.txt &
python threePN.py --seed 19 --kind random --mm 0.90 --eta-star 0.875 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-19-mm=0.90-eta_star=0.875.txt &
python threePN.py --seed 20 --kind random --mm 0.90 --eta-star 0.850 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-random-20-mm=0.90-eta_star=0.850.txt &
echo "Stochastic, varying eta_star"
python threePN.py --seed 21 --kind stochastic --mm 0.90 --eta-star 0.975 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-21-mm=0.90-eta_star=0.975.txt &
python threePN.py --seed 22 --kind stochastic --mm 0.90 --eta-star 0.950 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-22-mm=0.90-eta_star=0.950.txt &
python threePN.py --seed 23 --kind stochastic --mm 0.90 --eta-star 0.925 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-23-mm=0.90-eta_star=0.925.txt &
python threePN.py --seed 24 --kind stochastic --mm 0.90 --eta-star 0.900 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-24-mm=0.90-eta_star=0.900.txt &
python threePN.py --seed 25 --kind stochastic --mm 0.90 --eta-star 0.875 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-25-mm=0.90-eta_star=0.875.txt &
python threePN.py --seed 26 --kind stochastic --mm 0.90 --eta-star 0.850 --n-eff 1000 --n-eta 0 --subdir scaling 2>&1 | tee -a threePN-stochastic-26-mm=0.90-eta_star=0.850.txt &
wait
echo "Done with varying eta_star"
