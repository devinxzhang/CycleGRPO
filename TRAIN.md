## SINGLE NODE
```
cd <PATH_TO_DATA>/MaskTokenizer
source source /path/to/miniconda3/bin/activate
conda activate yk-qwen25vl-sft
bash projects/swift/sft_qwen3vl_4b_mask.sh
```

## MULTI NODE
```
# add the requirement env
sudo apt-get install ffmpeg libsm6 libxext6 tmux htop  -y

# export http_proxy=proxy.example.invalid:3128  https_proxy=proxy.example.invalid:3128  no_proxy=code.example.invalid 

export http_proxy=http://proxy.example.invalid:8118 https_proxy=http://proxy.example.invalid:8118 no_proxy=.example.invalid


# 设定分布式训练的参数
IFS=',' read -ra ADDR <<< "$ARNOLD_WORKER_HOSTS"
IFS=':' read -ra ADDR <<< "${ADDR[0]}"  # Split the IP Address and Port
export MASTER_ADDR=${ADDR[0]}
export MASTER_PORT=${ADDR[1]}
export NUM_NODE=${ARNOLD_WORKER_NUM}
export NODE_RANK=${ARNOLD_ID}
# 找到空闲端口并运行
HOSTS=$ARNOLD_WORKER_HOSTS
HOST=(${HOSTS//,/ })
HOST_SPLIT=(${HOST//:/ })
PORT=${HOST_SPLIT[-1]}
N_PROCESS=$ARNOLD_WORKER_GPU
N_NODE=$ARNOLD_WORKER_NUM
TOTAL_GPUS=$((N_PROCESS * N_NODE))


cd <PATH_TO_DATA>/MaskTokenizer
source source /path/to/miniconda3/bin/activate
conda activate yk-qwen25vl-sft

NODE_RANK=$ARNOLD_ID NNODES=$N_NODE MASTER_ADDR=$ARNOLD_WORKER_0_HOST MASTER_PORT=$PORT bash projects/swift/sft_qwen3vl_4b_mask.sh
```