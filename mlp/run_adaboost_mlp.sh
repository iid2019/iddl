#!/bin/bash

BEGIN=$(date +"%Y-%d-%m_%T")
echo ${BEGIN}

conda activate iddl_env
nohup python adaboost_mlp.py > adaboost_mlp_${BEGIN}.log 2>&1 & disown
echo $! > adaboost_mlp_${BEGIN}_pid.txt
tail -f adaboost_mlp_${BEGIN}.log
