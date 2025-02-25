#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

from torch.nn import functional as F
from utils import AverageMeter

class Trainer(object):
    """ Trainer to train adversarial attacting model """
    def __init__(self, model, attack, optimizer, summary_writer=None, 
        print_freq=1, output_freq=1, is_cuda=True, base_lr=0.001, 
        max_epoch=100, steps=[], rate=1.):
        super(Trainer, self).__init__()
        self.model = model
        self.attack = attack
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.iter = 0
        self.print_freq = print_freq
        self.output_freq = output_freq
        self.is_cuda = is_cuda
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.steps = steps
        self.rate = rate
        self.get_lr_mults()

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        adv_time = AverageMeter()
        loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()

        self.decrease_lr(epoch)
        end = time.time()
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()
            # Compute Adversarial Perturbations
            t0 = time.time()
            x_adv = self.attack(x, y)
            adv_time.update(time.time() - t0)

            t0 = time.time()
            adv_pred = self.model(x_adv)
            adv_loss = F.cross_entropy(adv_pred, y)
            self.optimizer.zero_grad()
            adv_loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            adv_loss_meter.update(adv_loss.item())
            adv_acc = self.accuracy(adv_pred, y)
            adv_acc_meter.update(adv_acc[0].item())
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('train/learning_rate_iter', self.lr_mults[epoch] * self.base_lr, self.iter)
                self.summary_writer.add_scalar('train/adv_loss_iter', adv_loss_meter.val, self.iter)
                self.summary_writer.add_scalar('train/adv_acc_iter', adv_acc_meter.val, self.iter)

            if (i + 1) % self.output_freq == 0:
                pred = self.model(x)
                loss = F.cross_entropy(pred, y)
                loss_meter.update(loss.item())
                acc = self.accuracy(pred, y)
                acc_meter.update(acc[0].item())
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('train/loss_iter', loss_meter.val, self.iter)
                    self.summary_writer.add_scalar('train/acc_iter', acc_meter.val, self.iter)

            if (i + 1) % self.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}] Time:[{:.3f}/{:.3f}] " \
                        "Loss:[{:.3f}/{:.3f}] AdvLoss:[{:.3f}/{:.3f}] " \
                        "Acc:[{:.3f}/{:.3f}] AdvAcc:[{:.3f}/{:.3f}] ".format(
                            epoch, i + 1, len(data_loader), batch_time.val, 
                            adv_time.val, loss_meter.val, loss_meter.avg, 
                            adv_loss_meter.val, adv_loss_meter.avg, acc_meter.val, 
                            acc_meter.avg, adv_acc_meter.val, adv_acc_meter.avg)
                print(p_str)

            self.iter += 1

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('train/loss_epoch', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('train/adv_loss_epoch', adv_loss_meter.avg, epoch)
            self.summary_writer.add_scalar('train/acc_epoch', acc_meter.avg, epoch)
            self.summary_writer.add_scalar('train/adv_acc_epoch', adv_acc_meter.avg, epoch)

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def reset(self):
        self.iter = 0

    def close(self):
        self.iter = 0
        if self.summary_writer is not None:
            self.summary_writer.close()

    def decrease_lr(self, epoch):
        lr_mult = self.lr_mults[epoch]
        for g in self.optimizer.param_groups:
            g['lr'] = lr_mult * self.base_lr * g.get('lr_mult', 1.0)

    def get_lr_mults(self):
        self.lr_mults = np.ones(self.max_epoch)
        self.steps = sorted(filter(lambda x: 0<x<self.max_epoch, self.steps))
        if len(self.steps) > 0 and 0 < self.rate < 1.:
            for step in self.steps:
                self.lr_mults[step:] *= self.rate