#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FreezeOut Training Function
Andy Brock, 2017

This script trains and tests a model using FreezeOut to accelerate training
by progressively freezing early layers and excluding them from the backward
pass. It has command-line options for defining the phase-out strategy, including
how far into training to start phasing out layers, whether to scale
initial learning rates as a function of how long the layer is trained for,
and how the phase out schedule is defined for layers after the first (i.e. are
layers frozen at regular intervals or is cubically more time given to later
layers?)

Based on Jan Schlüter's DenseNet training code:
https://github.com/Lasagne/Recipes/blob/master/papers/densenet
'''

import os
import logging
import sys
# import time
# import datetime

import numpy as np

import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

from utils import get_data_loader, MetricsLogger, progress
# Set the recursion limit to avoid problems with deep nets
sys.setrecursionlimit(5000)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Trainer():
    def __init__(self, batch_size) -> None:
        self.net = ''
        self.batch_size = batch_size
        self.acc = []
        self.mlog = ''
        self.train_loader = ''
        self.test_loader = ''
        self.start_epoch = 0
        self.train_loss = []

    # Training Function, presently only returns training loss
    # x: input data
    # y: target labels

    def train_fn(self, x, y):
        self.net.optim.zero_grad()
        output = self.net(V(x.cuda()))
        loss = F.nll_loss(output, V(y.cuda()))
        loss.backward()
        self.net.optim.step()
        return loss.item()

    # Testing function, returns test loss and test error for a batch
    # x: input data
    # y: target labels
    def test_fn(self, x, y):
        # output = net(V(x.cuda(), volatile=True))
        # test_loss = F.nll_loss(output, V(y.cuda(), volatile=True)).item()
        with torch.no_grad():
            output = self.net(x.cuda())
            test_loss = F.nll_loss(output, y.cuda()).item()

        # Get the index of the max log-probability as the prediction.
        pred = output.data.max(1)[1].cpu()
        test_error = pred.ne(y).sum()
        return test_loss, test_error

    def train_epoch(self, epoch):
        # Pin the current epoch on the network.
        self.net.epoch = epoch

        # shrink learning rate at scheduled intervals, if desired
        if 'epoch' in self.net.lr_sched and epoch in self.net.lr_sched['epoch']:
            logging.info('Annealing learning rate...')
            # Optionally checkpoint at annealing
            # if net.checkpoint_before_anneal:
            # torch.save(net, str(epoch) + '_' + save_weights + '.pth')
            for param_group in self.net.optim.param_groups:
                param_group['lr'] *= 0.1

        ### START TRAINING ###
        # List where we'll store training loss
        train_loss = []

        # Prepare the training data
        batches = progress(
            self.train_loader, desc='Epoch %d/%d, Batch ' % (
                epoch + 1, self.net.epochs),
            total=len(self.train_loader.dataset) // self.batch_size)

        # Put the network into training mode
        self.net.train()

        # Execute training pass
        for x, y in batches:
            # Update LR if using cosine annealing
            if 'itr' in self.net.lr_sched:
                self.net.update_lr()
            train_loss.append(self.train_fn(x, y))

        # Report training metrics
        train_loss = float(np.mean(train_loss))
        print('  training loss:\t%.6f', train_loss)
        self.mlog.log(epoch=epoch, train_loss=float(train_loss))

        # Check how many layers are active
        actives = 0
        for m in self.net.modules():
            if hasattr(m, 'active') and m.active:
                actives += 1
        logging.info(f'Currently have {actives} active layers...')

        return train_loss

    def test_epoch(self, epoch):
        ### START TESTING ###
        # Lists to store
        val_loss = []
        val_err = err = []

        # Set network into evaluation mode
        self.net.eval()

        # Execute validation pass
        for x, y in self.test_loader:
            loss, err = self.test_fn(x, y)
            val_loss.append(loss)
            val_err.append(err)

        # Report validation metrics
        val_loss = float(np.mean(val_loss))
        val_err = 100 * float(np.sum(val_err)) / \
            len(self.test_loader.dataset)
        print('  validation loss:\t%.6f' % val_loss)
        print('  validation error:\t%.2f%%' % val_err)
        self.mlog.log(epoch=epoch, val_loss=val_loss, val_err=val_err)
        self.acc.append(1.0 - val_err/100.0)

    def setup_training(self, depth, growth_rate, dropout, augment,
                       validate, epochs, save_weights, batch_size,
                       t_0, seed, scale_lr, how_scale, which_dataset,
                       const_time, resume, model):

        # Update save_weights:
        if save_weights == 'default_save':
            save_weights = (model + '_k' + str(growth_rate) + 'L' + str(depth)
                            + '_ice' + str(int(100*t_0)) + '_' +
                            how_scale + str(scale_lr)
                            + '_seed' + str(seed) + '_epochs' + str(epochs)
                            + 'C' + str(which_dataset))

        # Seed RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Name of the file to which we're saving losses and errors.
        metrics_fname = 'logs/'+save_weights + '_log.jsonl'
        logging.info('Metrics will be saved to %s', metrics_fname)
        logging.info('Running with seed %s, t_0 of %s, and the %s scaling method with learning rate scaling set to %s.',
                     str(seed), str(t_0), how_scale, str(scale_lr))
        self.mlog = MetricsLogger(metrics_fname, reinitialize=(not resume))

        # Import the model module
        # Get information specific to each dataset
        model_module = __import__(model)
        self.train_loader, self.test_loader = get_data_loader(which_dataset, augment,
                                                              validate, self.batch_size)

        # Build network, either by initializing it or loading a pre-trained
        # network.
        if resume:
            logging.info(f'loading network {save_weights}...')
            self.net = torch.load(save_weights + '.pth')

            # Which epoch we're starting from
            self.start_epoch = self.net.epoch + \
                1 if hasattr(self.net, 'epoch') else 0

        #  Get net
        else:
            logging.info('Instantiating network with model %s ...', model)
            net = model_module.Model(growth_rate, depth=depth,
                                     nClasses=which_dataset,
                                     epochs=epochs,
                                     t_0=t_0,
                                     scale_lr=scale_lr,
                                     how_scale=how_scale,
                                     const_time=const_time)
            self.net = net.cuda()
            self.start_epoch = 0
        # from torchsummary import summary
        # summary(self.net, (3, 28, 28))
        logging.info('Number of params: %d',
                     sum([p.data.nelement() for p in self.net.parameters()]))

    def train(self, validate=True, save_weights=False):
        logging.info('Starting training at epoch %s...', self.start_epoch)
        for epoch in range(self.start_epoch, self.net.epochs):

            ### START TRAINING ###
            self.train_loss = self.train_epoch(epoch=epoch)

            ### START TESTING ###
            # Optionally, take a pass over the validation or test set.
            if validate:
                self.test_epoch(epoch=epoch)

            # Save weights for this epoch
            print(f'Saving weights to {save_weights} ...')
            torch.save(self.net, save_weights + '.pth')

            # start next_trainer
            # t2 = Trainer(self.batch_size)
            # t2.setup_training()
            # t2.train_epoch()

        print(self.acc)

        # At the end of it all, save weights even if we didn't checkpoint.
        if save_weights:
            torch.save(self.net, save_weights + '.pth')


# def main():
#     start = time.time()
#     t1 = Trainer(args.batch_size)
#     t1.setup_training(**vars(args))
#     t1.overlap_train(validate=args.validate, save_weights=args.save_weights)
#     end = time.time()
#     print(f'Total training time: {datetime.timedelta(seconds= end-start)}')

#     # run
#     # train_test(**vars(args))


# if __name__ == '__main__':
#     main()
