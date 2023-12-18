#!/bin/bash -l
# shellcheck disable=SC2206

#SBATCH -A ADD_THE_PROJECT_NAME_HERE # project name 
#SBATCH -M ADD_THE_SYSTEM_NAME_HERE # name of system i.e. snowy, dardel

#SBATCH -p node # request a full node 
#SBATCH -N 2  # Number of nodes  Change --num-gpus "2" in head command and worker loop as well
#SBATCH -t 0:15:00 # change time accordingly
#SBATCH --gpus-per-node=2  # change gpus accordingly
#SBATCH -J exp-seed-3 # name of the job 
#SBATCH -D ./ # stay in current working directory 


source ~/.bashrc
conda activate recover

set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__


# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "16" --num-gpus "2" --block &
# __doc_head_ray_end__


# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "16" --num-gpus "2" --block &
    sleep 5
done
# __doc_worker_ray_end__


# __doc_script_start__
#Active learning with Upper Confidence Bound Aquisition 
python train.py --config active_learning_UCB
