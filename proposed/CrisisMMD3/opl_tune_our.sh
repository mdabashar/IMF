#!/bin/bash -l
#PBS -N c3.r3.53.f6s3l40.FF5.fb5.s10e30.B64
#PBS -l ncpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
pip install torch
pip install scikit-learn
pip install matplotlib
python3 main_meta_new_ms_2bnn.py --lr=5e-3 --blr=1 --numFirst=6 --numSec=3 --nepoch=300 --length=40 --functionalp=0.5 --ratio=0.3 --bnnp=0.5 --nForwardPasses=5 --earlyStop=30 --scheduler=10


