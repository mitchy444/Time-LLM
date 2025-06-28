sudo apt update && sudo apt install openmpi-bin libopenmpi-dev
conda create -n tllm python=3.11
conda activate tllm
pip install -r requirements.txt

gdown 

bash ./scripts/TimeLLM_ETTh1_single_gpu.sh