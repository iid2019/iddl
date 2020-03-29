#!/bin/bash

BEGIN=$(date +"%Y-%d-%m_%T")
echo ${BEGIN}

conda activate iddl_env
nohup python adaboost_resnet.py > adaboost_resnet_${BEGIN}.log 2>&1 & disown
echo $! > adaboost_resnet_${BEGIN}_pid.txt
tail -f adaboost_resnet_${BEGIN}.log
