#!/bin/bash
set -x

export PYTHONUNBUFFERED="True"

# train default
python train.py --exp_name cifar10_adv_2 \
                --data_name cifar10 \
                --data_dir data/cifar10 \
                --model_name wide \
                --max_epoch 200 \
                --lr 0.1 \
                --batch_size 128 \
                --seed 2333 \
                --gpu 0,1 \
                --steps 80,140,160 \
                --eval_batch_size 256 \
                --epsilon 0.3 \
                --k 40 \
                --alpha 0.01