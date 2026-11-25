#!/bin/bash -l
#PBS -l select=1:ncpus=1:ngpus=1:mem=20GB:gputype=A100
#PBS -l walltime=10:00:00
#PBS -N C1F1.lr3.l2.bf3.r1
cd $PBS_O_WORKDIR
module load python/3.10.4-gcccore-11.3.0
pip3 install torch scikit-learn matplotlib datasets transformers
python3 main.py --lr=1e-3 --blr=1 --numFirst=1 --numSec=1 --nepoch=50 --length=200 --functionalp=0.3 --ratio=0.1 --bnnp=0.3 --nForwardPasses=1 --earlyStop=30 --scheduler=100
