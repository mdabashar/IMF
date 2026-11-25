#!/bin/bash -l
#PBS -N EV2.r3.2b1.l1.f3b3.ff5
#PBS -l ncpus=1
#PBS -l mem=16gb
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
pip install torch
pip install scikit-learn
pip install matplotlib
python3 main_meta_new_1.py --lr=1e-2 --blr=1e-1 --nepoch=50 --length=100 --functionalp=0.3 --ratio=0.3 --bnnp=0.3 --nForwardPasses=5

