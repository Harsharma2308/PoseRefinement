#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

python3 ./tools/eval_ycb.py --dataset_root /media/SSD2/harsh/data/YCB_Video_Dataset\
  --model trained_models/ycb/pose_model_current.pth\
  --refine_model trained_models/ycb/pose_refine_model_current.pth
