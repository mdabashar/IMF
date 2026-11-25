#!/bin/bash -l
#PBS -l select=1:ncpus=1:ngpus=1:mem=17GB:gputype=A100
#PBS -l walltime=4:00:00
#PBS -N E18256l23b2.l5.r3.f2b3
cd $PBS_O_WORKDIR
module load python/3.10.4-gcccore-11.3.0
pip install torch
pip install scikit-learn
pip install matplotlib
pip install gpytorch
python3 main_meta_new_1.py --lr=2e-3 --blr=1e-2 --nepoch=35 --length=500 --functionalp=0.2 --ratio=0.3 --bnnp=0.3 --nForwardPasses=5
