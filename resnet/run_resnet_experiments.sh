#!/bin/bash

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo ${BEGIN}

mkdir -p log
touch log/resnet_experiments_${BEGIN}.log
nohup bash resnet_experiments.sh ${BEGIN} > log/resnet_experiments_${BEGIN}.log 2>&1 & disown
echo $! > log/resnet_experiments_${BEGIN}_pid.txt
tail -f log/resnet_experiments_${BEGIN}.log
