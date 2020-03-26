#!/bin/bash

conda activate iddl_env
nohup python adaboost_mlp.py > adaboost_mlp.log 2>&1 & disown
echo $! > adaboost_mlp_pid.txt
# tail -f adaboost_mlp.log
