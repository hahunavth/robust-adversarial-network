#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import numpy as np
import pprint
from time import gmtime, strftime

import torch
import torch.nn as nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from Logging import Logger
from config import cfg
from model import create_model
from trainer import Trainer
from evaluator import Evaluator
from pgd_attack import create_attack
from optimizer import create_optimizer
from utils import *


if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    if not cfg.randomize:
        # set fixed seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
    log_path = os.path.join(cfg.log_dir, cfg.exp_name)
    mkdir_if_missing(log_path)
    snap_path = os.path.join(cfg.snap_dir, cfg.exp_name)
    mkdir_if_missing(snap_path)

    summary_writer = None
    if not cfg.no_log:
        log_name = cfg.exp_name + "_log_" + \
                strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.txt'
        sys.stdout = Logger(os.path.join(log_path, log_name))
        summary_writer = SummaryWriter(log_dir=log_path)

    print("Input Args: ")
    pprint.pprint(cfg)
    train_loader, test_loader, num_classes, img_size = get_data_loader(
        data_name=cfg.data_name, data_dir=cfg.data_dir, batch_size=cfg.batch_size, 
        test_batch_size=cfg.eval_batch_size, num_workers=4)

    model = create_model(name=cfg.model_name, num_classes=num_classes)
    optimizer = create_optimizer(name=cfg.optimizer, model=model, lr=cfg.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, 
    #                     weight_decay=0.0005, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, 
    #                     momentum=0.9, weight_decay=0.0005)
    is_cuda = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
        is_cuda = True
    
    # load ckpt
    start_ep = 0
    ckpts = os.listdir(snap_path)
    if ckpts:
        sorted(ckpts, key=lambda x: int(x.replace(".pth", "").split('_')[1]))
        last_ckpt = ckpts[-1]
        ckpt = torch.load(os.path.join(snap_path, last_ckpt), 
                #    map_location="cpu"
                )
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            print("Warn: found ckpt file but cannot load model!")
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("Warn: found ckpt file but cannot load optimizer!")
        if 'epoch' in ckpt:
            start_ep = ckpt['epoch'] + 1
            print("Load ckpt file from epoch {}".format(start_ep))
        else:
            print("Warn: found ckpt file but cannot load epoch!")
            start_ep = 0

    attack = create_attack(attack_method=cfg.attack_method.lower(), model=model, 
                           epsilon=cfg.epsilon, k=cfg.k, alpha=cfg.alpha, 
                           mu=cfg.mu, random_start=cfg.random_start)

    trainer = Trainer(model=model, attack=attack, optimizer=optimizer, 
                      summary_writer=summary_writer, is_cuda=True, 
                      output_freq=cfg.output_freq, print_freq=cfg.print_freq, 
                      base_lr=cfg.lr, max_epoch=cfg.max_epoch, 
                      steps=cfg.steps, rate=cfg.decay_rate)
    if not cfg.no_val_stage:
        evaluator = Evaluator(model=model, attack=attack, is_cuda=is_cuda, verbose=False)

    trainer.reset()
    for epoch in range(start_ep, cfg.max_epoch):
        trainer.train(epoch, train_loader)
        if cfg.save_freq < 1:
            save_current = False
        else:
            save_current = (epoch + 1) % cfg.save_freq == 0 \
                or epoch == 0 or epoch == cfg.max_epoch - 1
        if save_current:
            if not cfg.no_val_stage:
                nat_acc, adv_acc = evaluator.evaluate(test_loader)
                if summary_writer is not None:
                    summary_writer.add_scalar('val/nat_acc', nat_acc, epoch)
                    summary_writer.add_scalar('val/adv_acc', adv_acc, epoch)
                print("epoch {:3d} evaluated".format(epoch))
                print("natural     accuracy: {:.4f}".format(nat_acc))
                print("adversarial accuracy: {:.4f}".format(adv_acc))
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            if hasattr(optimizer, 'module'):
                optim_state_dict = optimizer.module.state_dict()
            else:
                optim_state_dict = optimizer.state_dict()
            dict_to_save = {'state_dict': state_dict, 
                            'epoch': epoch + 1,
                            'optimizer': optim_state_dict}
            fpath = os.path.join(snap_path, 'checkpoint_' + 
                        str(epoch + 1) + '.pth')
            torch.save(dict_to_save, fpath)

    trainer.close()