#!/bin/bash 
conda env create --file environment.yml --yes

conda activate NanoFold

echo "Detected CUDA version: $CUDA_VERSION"

if [ "$CUDA_VERSION" == "cpu" ]; then
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
else
  pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu${CUDA_VERSION}.html
fi

pip install transformers[torch] accelerate
pip install ml_collections
pip install e3nn

# You can manually specify the CUDA version by passing it as an argument to the script:
# ./env_setup.sh 121  # Install for CUDA 12.1