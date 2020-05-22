#!/bin/bash

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo ${BEGIN}

# Get the expreiments index
INDEX=$1

mkdir -p log
touch log/resnet_experiments_${BEGIN}.log
nohup bash ensemble_experiments.sh ${BEGIN} ${INDEX} > log/ensemble_experiments_${INDEX}_${BEGIN}.log 2>&1 & disown
echo $! > log/ensemble_experiments_${INDEX}_${BEGIN}_pid.txt
tail -f log/ensemble_experiments_${INDEX}_${BEGIN}.log
