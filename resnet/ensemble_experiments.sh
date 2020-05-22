#!/bin/bash

BEGIN=$1
INDEX=$2

if [[ ${INDEX} = "1" ]]
then
list="1 3 9"
elif [[ ${INDEX} = "2" ]]
then
list="5 7"
fi

for num in ${list}
do
    python ensemble_experiments.py --num $num --file ensemble_result_${INDEX}_${BEGIN}.pkl &
    process_id=$!
    echo Waiting for the process with PID ${process_id} to finish...
    wait $process_id
    kill $process_id
    echo $process_id killed.
done
