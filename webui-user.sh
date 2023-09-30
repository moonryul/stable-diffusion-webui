#!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

# Install directory without trailing slash
install_dir="/home/$(whoami)"

# Name of the subdirectory
clone_dir="stable-diffusion-webui"

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS="--medvram --opt-split-attention"
#export COMMANDLINE_ARGS="--api"  #MJ: This enables the api which can be reviewed at http://127.0.0.1:7860/docs (or whever the URL is + /docs) The basic ones I'm interested in are these two. Let's just focus only on  /sdapi/v1/txt2img

#MJ: This will aid in keeping the VRAM usage lower and have a bit of a smaller overall footprint due to garbage collection cleanup

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64
#MJ: confer: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings

export COMMANDLINE_ARGS="--xformers"

#Setting GPU to use: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/1561

export CUDA_VISIBLE_DEVICES=1

# python3 executable
#python_cmd="python3"

# git executable
export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
venv_dir="venv"

# script to launch to start the app
export LAUNCH_SCRIPT="launch.py"

# install command for torch
export TORCH_COMMAND="pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
export REQS_FILE="requirements_versions.txt"

# Fixed git repos
export K_DIFFUSION_PACKAGE=""
export GFPGAN_PACKAGE=""

# Fixed git commits
export STABLE_DIFFUSION_COMMIT_HASH=""
export CODEFORMER_COMMIT_HASH=""
export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
#export ACCELERATE="True"  MJ: for debugging only

# Uncomment to disable TCMalloc
#export NO_TCMALLOC="True"

###########################################
