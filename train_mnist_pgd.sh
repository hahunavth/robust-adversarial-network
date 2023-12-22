#!/bin/bash
set -x

export PYTHONUNBUFFERED="True"

python train.py --exp_name mnist_adv \
                --data_name mnist \
                --data_dir data/mnist \
                --model_name mnist \
                --attack_method pgd \
                --max_epoch 100 \
                --batch_size 128 \
                --seed 2333 \
                --gpu 0,1 \
                --lr 0.1 \
                --steps 80,140,160 \
                --eval_batch_size 256 \
                --epsilon 8 \
                --k 10 \
                --alpha 2 \
                --no_log

