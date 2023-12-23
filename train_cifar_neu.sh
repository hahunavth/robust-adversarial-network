#!/bin/bash
set -x

export PYTHONUNBUFFERED="True"

# cifar 10 with without adversarial training
python train.py --exp_name cifar10_pgd_1 \
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
                --no_val_stage