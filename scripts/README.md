
when using WSL

Create /etc/wsl.conf with the following:

```
[automount]
enabled  = true
root     = /mnt/
options  = "metadata,umask=22,fmask=11"
```

install prerequisites on ubuntu 

```shell
sudo apt install build-essential
```

```shell
conda install mamba -n base -c conda-forge
```

5. Create an environment called OpenChatKit using the `environment.yml` file at the root of this repo.

> **Note**
> Use `mamba` to create the environment. It's **much** faster than using `conda`.

```shell
sudo apt install build-essential
mamba env create -f environment.yml 

# when changing environment.yml
mamba env update -f environment.yml

```

6. Activate the new conda environment.

```shell
conda activate OpenChatKitVB

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -O ./build/wikitext-103-raw-v1.zip
unzip ./build/wikitext-103-raw-v1.zip -d ./buid/
sudo apt install unzip

# require  data
python ./scripts/01_vanilla.py
python ./scripts/02_vanilla_to_model.py
python ./scripts/03_training.py

```





* https://huggingface.co/welcome

Getting started with our git and git-lfs interface

If you need to create a repo from the command line (skip if you created a repo from the website)

pip install huggingface_hub
You already have it if you installed transformers or datasets

huggingface-cli login
Log in using a token from huggingface.co/settings/tokens
Create a model or dataset repo from the CLI if needed
huggingface-cli repo create repo_name --type {model, dataset, space}
Clone your model or dataset locally

Make sure you have git-lfs installed
(https://git-lfs.github.com)
git lfs install
git clone https://huggingface.co/username/repo_name
Then add, commit and push any file you want, including larges files

 save files via `.save_pretrained()` or move them here
git add .
git commit -m "commit from $USER"
git push
In most cases, if you're using one of the compatible libraries, your repo will then be accessible from code, through its identifier: username/repo_name

For example for a transformers model, anyone can load it with:

tokenizer = AutoTokenizer.from_pretrained("username/repo_name")
model = AutoModel.from_pretrained("username/repo_name")

```bash
mkdir ./build/wiki_medical_terms
git clone https://huggingface.co/datasets/gamino/wiki_medical_terms ./build/wiki_medical_terms

mkdir ./build/medical-keywords
git clone https://huggingface.co/datasets/argilla/medical-keywords  ./build/medical-keywords
```