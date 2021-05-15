#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=3

start=`date +%s`

python3 ./tools/eval_linemod_modified.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model trained_models/linemod/pose_model_current.pth\
  --refine_model trained_models/linemod/pose_refine_model_98_0.0064081341689759255.pth

end=`date +%s`
runtime=$((end-start))
echo $runtime

