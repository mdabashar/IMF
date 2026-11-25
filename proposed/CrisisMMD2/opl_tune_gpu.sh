#!/bin/bash -l
#PBS -l select=1:ncpus=1:ngpus=1:mem=10GB:gputype=A100
#PBS -l walltime=5:00:00
#PBS -N C2B209538.lr3b4.l1.bf3.r3.s5
cd $PBS_O_WORKDIR
module load python/3.10.4-gcccore-11.3.0
pip3 install torch scikit-learn matplotlib datasets transformers
python3 main.py --lr=1e-3 --blr=1e-4 --numFirst=1 --numSec=1 --nepoch=50 --length=100 --functionalp=0.3 --ratio=0.3 --bnnp=0.3 --nForwardPasses=1 --earlyStop=300 --scheduler=5
