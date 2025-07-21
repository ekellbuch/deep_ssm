conda create -n deep_ssm python=3.12.2
conda activate deep_ssm
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -e src/