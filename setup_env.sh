conda create -n deep_ssm python=3.12
conda init
conda activate deep_ssm
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e src/