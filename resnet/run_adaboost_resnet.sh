#!/bin/bash

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo ${BEGIN}

conda activate iddl_env
mkdir -p log
nohup python adaboost_resnet.py > log/adaboost_resnet_${BEGIN}.log 2>&1 & disown
echo $! > log/adaboost_resnet_${BEGIN}_pid.txt
tail -f log/adaboost_resnet_${BEGIN}.log
