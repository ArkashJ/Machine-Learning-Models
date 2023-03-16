Install the pytorch gpu for running these commands on anaconda

conda create -n env python=3.9 anaconda
source activate env
(NOT SURE ABOUT THIS COMMAND, try running it)conda env create -f torch-conda-nightly.yml -n torch
conda install -y jupyter
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install -c conda-forge jupyter jupyterlab

conda create -n torch-gpu python=3.8
conda activate torch-gpu
