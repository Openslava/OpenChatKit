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
# mamba env update -f environment.yml

# conda activate OpenChatKitVB
##------
# python inference/bot.py --model togethercomputer/Pythia-Chat-Base-7B

##--pretrain model 
# python pretrained/terminus/prepare.py

conda activate OpenChatKitVB
python inference/bot.py --model togethercomputer/Pythia-Chat-Base-7B --cpu

# python inference/bot.py --model ./build --cpu

## install CUDA for training ... 
## https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux
# sudo apt install nvidia-cuda-toolkit

## install docker ...
# apt list --upgradable
# sudo apt update
# sudo apt upgrade
# sudo apt install docker.io
# sudo apt install docker-compose
# sudo apt install docker
# sudo apt-get update

# sudo apt-get install ca-certificates curl gnupg
#  sudo install -m 0755 -d /etc/apt/keyrings
#  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
#  sudo chmod a+r /etc/apt/keyrings/docker.gpg
#  echo   "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
#  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# sudo apt-get update
# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# sudo apt-get install git-lfs
# sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
# sudo apt-get update
# sudo apt-get upgrade
# sudo apt install azure-cli
#
#### install R
# sudo apt install gpp
# sudo apt install g++
# sudo apt install texlive-latex-base
# apt-get update
# sudo apt-get update
# sudo apt install texlive-latex-base
# sudo apt install gnome-text-editor -y
# sudo apt install gedit -y
# sudo apt-get install libreadline-dev
# sudo apt-get install  libtermcap-dev
# sudo apt-get install libtermcap-dev
# sudo apt install libx11-dev
# sudo apt-get -y install bzip2
# sudo apt-get -y install libbz2-dev
# sudo apt update sudo apt install liblzma-dev
# sudo apt install liblzma-dev
#  486  sudo apt install build-essential
#  489  sudo apt-get build-dep r-base
#  490  sudo apt-get update
#  491  sudo apt-get build-dep r-base
#  492  sudo apt-get  r-base
#  500  sudo apt-get install texi2any
#  501  sudo apt-get install texinfo
#  505  sudo apt-get install texi2dvi
#  506  sudo apt-get install texinfo
#  514  sudo apt-get install texlive-latex-extra
#  516  sudo apt-get install fonts-inconsolata
#  517  sudo apt-get install inconsolata
#  518  sudo apt-get find inconsolata
#  519  sudo apt-get search inconsolata
#  520  sudo apt-get --help
#  533  sudo apt install openjdk-17-jre-headless
#  546  sudo apt-get install xorg-x11-server-devel libX11-devel libXt-devel
#  547  sudo apt-get search x11
#  548  sudo apt-cache search x11
#  549  sudo apt-cache search xorg-x11-server-devel
#  550  sudo apt-cache search x11
#  551  sudo apt-get install x11-common
#  552  sudo apt install libx11-dev
#  553  sudo apt-get install libx11-dev xserver-xorg-dev xorg-dev
#  590  sudo apt install x11-apps

### Installl mc
# sudo apt-get update
# sudo apt-get upgrade
# sudo apt install mc




