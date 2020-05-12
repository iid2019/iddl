BEGIN=$1
GPU=3 # The GPU index

for model in ResNet-18 ResNet-34 ResNet-50 B-ResNet-18 B-ResNet-34 B-ResNet-50 R-ResNet-18 R-ResNet-34 R-ResNet-50
do
    python run_resnet_experiments.py --model $model --name $BEGIN --gpu $GPU &
    process_id=$!
    echo Waiting for the process with PID ${process_id} to finish...
    wait $process_id
    kill $process_id
    echo $process_id killed.
done
