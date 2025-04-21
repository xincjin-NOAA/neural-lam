#!/bin/bash

source "/scratch1/NCEPDEV/da/Xin.C.Jin/miniconda3/etc/profile.d/conda.sh"
env_name=$1
echo "env name: $env_name"
#clt conda env create -y  --name $env_name  --file environment.yml
conda env create   --name $env_name  --file pytorch_ting.yml 
conda list $env_name
echo "install extra torch packages"

export TORCH="2.0.1"
export CUDA="cu118"

conda run -n $env_name pip \
install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 \
torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

echo "DONE"
