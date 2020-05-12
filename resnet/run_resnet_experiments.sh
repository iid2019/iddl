#!/bin/bash

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo ${BEGIN}

# conda activate iddl_env
mkdir -p log
nohup python run_resnet_experiments.py > log/resnet_experiments_${BEGIN}.log 2>&1 & disown
echo $! > log/resnet_experiments_${BEGIN}_pid.txt
tail -f log/resnet_experiments_${BEGIN}.log
