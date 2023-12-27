#!/bin/bash
set -x

export PYTHONUNBUFFERED="True"

# neu with smaller lr
python train.py --exp_name mnist_neu_2 \
                --data_name mnist \
                --data_dir data/mnist \
                --model_name mnist \
                --optimizer adam \
                --attack_method none \
                --max_epoch 100 \
                --batch_size 128 \
                --seed 2333 \
                --gpu 0,1 \
                --lr 0.01 \
                --steps 80,140,160 \
                --eval_batch_size 256 \
                --no_val_stage

