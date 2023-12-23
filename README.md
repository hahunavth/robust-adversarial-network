# RobustAdversarialNetwork
A pytorch re-implementation for paper "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"

## Requirements
* pytorch>0.4
* torchvision
* tensorboardX

## Parameters
All the parameters are defined in config.py 
* `exp_name`: experiment name, will be used for construct output directory
* `snap_dir`: root directory to save snapshots, it works with `exp_name` to form a directory for a specific experiment

## Usage
### Training
```
python train.py
```

### Instructions
```
usage: train.py [-h] [--exp_name EXP_NAME] [--snap_dir SNAP_DIR] [--log_dir LOG_DIR] [--no_log] [--data_name DATA_NAME] [--data_dir DATA_DIR] [--model_name MODEL_NAME] [--max_epoch MAX_EPOCH]
                [--lr LR] [--batch_size BATCH_SIZE] [--seed SEED] [--gpu GPUS] [--rand] [--steps STEPS] [--decay_rate DECAY_RATE] [--optimizer OPTIMIZER] [--print_freq PRINT_FREQ]
                [--output_freq OUTPUT_FREQ] [--save_freq SAVE_FREQ] [--eval_model EVAL_MODEL] [--eval_samples EVAL_SAMPLES] [--eval_batch_size EVAL_BATCH_SIZE] [--eval_cpu]
                [--attack_method ATTACK_METHOD] [--epsilon EPSILON] [--k K] [--alpha ALPHA] [--mu MU] [--random_start]

Train adversal attack network

options:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   exp name used to construct output dir (default: debug)
  --snap_dir SNAP_DIR   directory to save model (default: snapshots)
  --log_dir LOG_DIR     directort to save logs (default: logs)
  --no_log              if record logs (do not log) (default: False)
  --data_name DATA_NAME
                        used dataset (default: mnist)
  --data_dir DATA_DIR   data directory (default: data/mnist)
  --model_name MODEL_NAME
                        network model (default: mnist)
  --max_epoch MAX_EPOCH
                        max train steps (default: 100)
  --lr LR               learning rate (default: 0.0001)
  --batch_size BATCH_SIZE
                        training batch size (default: 200)
  --seed SEED           random seed (default: 0)
  --gpu GPUS            GPU to be used, default is '0,1' (default: 0,1)
  --rand                randomize (not use a fixed seed) (default: False)
  --steps STEPS         epoches to decrease learning rate (default: 80,140,180)
  --decay_rate DECAY_RATE
                        decay rate to decrease learning rate (default: 0.1)
  --optimizer OPTIMIZER
                        optimizer: adam | sgd (default: sgd)
  --print_freq PRINT_FREQ
                        print freq (default: 10)
  --output_freq OUTPUT_FREQ
                        output freq (default: 5)
  --save_freq SAVE_FREQ
                        save checkpint freq (default: 5)
  --eval_model EVAL_MODEL
                        evaluation checkpint path (default: None)
  --eval_samples EVAL_SAMPLES
                        num of evaluation samples (default: 10000)
  --eval_batch_size EVAL_BATCH_SIZE
                        evaluation batch size (default: 200)
  --eval_cpu            if eval on cpu (do not on cpu) (default: False)
  --attack_method ATTACK_METHOD
                        attacking method: pgd | mifgsm (default: pgd)
  --epsilon EPSILON     the maximum allowed perturbation per pixel (default: 0.3)
  --k K                 the number of PGD iterations used by the adversary (default: 40)
  --alpha ALPHA         the size of the PGD adversary steps (default: 0.01)
  --mu MU               Moment for MIGFSM method (default: 1.0)
  --random_start        if random start (default: True)
```