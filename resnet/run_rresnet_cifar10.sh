#!/bin/bash

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo ${BEGIN}

conda activate iddl_env
mkdir -p log
nohup python run_r_resnet.py > log/rresnet_cifar10_${BEGIN}.log 2>&1 & disown
echo $! > log/rresnet_cifar10_${BEGIN}_pid.txt
tail -f log/rresnet_cifar10_${BEGIN}.log
