#!/bin/bash
# Azure VM
#--- install

# sudo apt install azure-cli



## install machine 
# wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
# chmod 755 ./Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh
## -----
# sudo apt-get install git-lfs
# echo  ".... continuing run.sh"
# conda install mamba -n base -c conda-forge
## reload of shelll required
# mamba env create -f environment.yml

# conda activate OpenChatKitVB
##------
# python inference/bot.py --model togethercomputer/Pythia-Chat-Base-7B

##--pretrain model 
# python pretrained/terminus/prepare.py

conda activate OpenChatKitVB
python inference/bot.py --model togethercomputer/Pythia-Chat-Base-7B --cpu



