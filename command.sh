#!/bin/bash

model_name=resnet50

PYTHONOPTIMIZE=TRUE torchrun --nproc-per-node=1 train.py \
    run.name=${model_name} \
    config.model.model_name=${model_name}
