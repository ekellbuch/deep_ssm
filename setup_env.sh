conda create -n deep_ssm python=3.9
conda activate deep_ssm
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 torchtext==0.17.2 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e src/