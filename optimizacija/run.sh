#!/bin/bash
#SBATCH --job-name=optim_job            # Ime naloge
#SBATCH --output=optim_%j.out           # Izhodni log; %j zamenja ID naloge
#SBATCH --error=optim_%j.err            # Napake v ločeni datoteki (opcijsko)
#SBATCH --ntasks=1                      # Število nalog (procesov)
#SBATCH --cpus-per-task=32              # Število jeder na nalogo (prilagodite glede na HPC)
#SBATCH --time=02:00:00                 # Časovna omejitev (HH:MM:SS)
#SBATCH --partition=standard            # Ime particije (spremenite, če je potrebno)

# Naložite modul Python (prilagodite, če je potrebno)
module load python/3.8

# Zaženite optimizacijsko skripto z ustreznimi argumenti:
python my_optimization.py --num_workers 32 --max_iter 40
