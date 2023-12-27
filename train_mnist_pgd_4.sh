#!/bin/bash
set -x

export PYTHONUNBUFFERED="True"

### Default mnist training parameters
## training
# optimizer adam
# lr 0.0001
## default attack parameters
# epsilon 0.3
# k 40
# alpha 0.01

## train with sgd and default attack parameters
# v1 with lr=0.05
python train.py --exp_name mnist_adv_1 \
                --data_name mnist \
                --data_dir data/mnist \
                --model_name mnist \
                --optimizer sgd \
                --attack_method pgd \
                --max_epoch 100 \
                --batch_size 128 \
                --seed 2333 \
                --gpu 0,1 \
                --lr 0.05 \
                --steps 80,140,160 \
                --eval_batch_size 256 \
                --epsilon 0.3 \
                --k 40 \
                --alpha 0.01