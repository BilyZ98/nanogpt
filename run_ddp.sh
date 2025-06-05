#!/bin/bash
# assign a number for each node
num_nodes=2

cmd="echo 'hello world'"
source ../load_cuda_module.sh
module list
# Initialize conda for the current script session
eval "$(conda shell.bash hook)" 

# Activate the conda environment
conda activate nanogpt
echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

for i in $(seq 1 $((num_nodes - 1))); do
    echo "Node $i"
    cmd="echo $i"
    cmd=" torchrun --nproc_per_node=4 --nnodes=$num_nodes --node_rank=$i \
    --master_addr=89.72.32.11 --master_port=1234 train.py config/train_gpt2.py"
    echo "command: $cmd"
    yhrun -N 1 -p gpu_v100 ${cmd} 
done
