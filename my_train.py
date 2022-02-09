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
import time
import datetime
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

# my library
from overlap_train import Trainer
from transmitter import Transmitter
import myplot

# Set the recursion limit to avoid problems with deep nets
sys.setrecursionlimit(5000)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def opts_parser():
    usage = 'Trains and tests a FreezeOut DenseNet on CIFAR.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-L', '--depth', type=int, default=76,
        help='Network depth in layers (default: %(default)s)')
    parser.add_argument(
        '-k', '--growth-rate', type=int, default=12,
        help='Growth rate in dense blocks (default: %(default)s)')
    parser.add_argument(
        '--dropout', type=float, default=0,
        help='Dropout rate (default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Perform data augmentation (enabled by default)')
    parser.add_argument(
        '--no-augment', action='store_false', dest='augment',
        help='Disable data augmentation')
    parser.add_argument(
        '--validate', action='store_true', default=True,
        help='Perform validation on validation set (ensabled by default)')
    parser.add_argument(
        '--no-validate', action='store_false', dest='validate',
        help='Disable validation')
    parser.add_argument(
        '--validate-test', action='store_const', dest='validate',
        const='test', help='Evaluate on test set after every epoch.')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--t_0', type=float, default=0.8,
        help=('How far into training to start freezing. Note that this if using'
              + ' cubic scaling then this is the uncubed value.'))
    parser.add_argument(
        '--scale_lr', type=bool, default=True,
        help='Scale each layer''s start LR as a function of its t_0 value?')
    parser.add_argument(
        '--no_scale', action='store_false', dest='scale_lr',
        help='Don''t scale each layer''s start LR as a function of its t_0 value')
    parser.add_argument(
        '--how_scale', type=str, default='cubic',
        help=('How to relatively scale the schedule of each subsequent layer.'
              + 'options: linear, squared, cubic.'))
    parser.add_argument(
        '--const_time', type=bool, default=False,
        help='Scale the #epochs as a function of ice to match wall clock time.')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    parser.add_argument(
        '--which_dataset', type=int, default=100,
        help='Which Dataset to train on (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=50,
        help='Images per batch (default: %(default)s)')
    parser.add_argument(
        '--resume', type=bool, default=False,
        help='Whether or not to resume training')
    parser.add_argument(
        '--model', type=str, default='densenet', metavar='FILE',
        help='Which model to use')
    parser.add_argument(
        '--save-weights', type=str, default='default_save', metavar='FILE',
        help='Save network weights to given .pth file')

    parser.add_argument(
        '--overlap', action='store_true', default=True,
        help='Transmission overlap (ensabled by default)')
    parser.add_argument(
        '--no-overlap', action='store_false', dest='overlap',
        help='Disable overlap')
    return parser


def overlap_train(args):
    t1 = Trainer(args.batch_size)
    t1.setup_training(**vars(args))
    t1_acc = []
    t1_loss = []

    t2_args = args
    t2_args.how_scale = 'linear'
    t2 = Trainer(args.batch_size)
    t2.setup_training(**vars(t2_args))
    t2_acc = []
    t2_loss = []

    for e in range(args.epochs):
        print('')
        logging.info(f'Epoch: {e+1}/{args.epochs}')

        t1.train_epoch(e)
        loss_1, acc_1 = t1.test_epoch(e)
        print(f'\tBase model loss: \t{loss_1:.4f}')
        print(f'\tBase model accuracy: \t{acc_1:.4f}')
        t1_acc.append(acc_1)
        t1_loss.append(loss_1)

        if args.overlap:
            # t1 transmission
            print('***[BG] Start Tensor transmission!***')
            trans1 = Transmitter()
            trans1.start()

            # train next_model on background thread
            print('***[FG] Start next_trainer!***')
            future = ''
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(t2.train_epoch, e)
                future = executor.submit(t2.test_epoch, e)

            # Receive model update from central
            trans1.join()
            print('***Tensor transmission done!***')

            # check if t2 is finish
            if future.done():
                # print(future.result())
                loss_2, acc_2 = future.result()

            else:
                # next_trainer is not ready, so we will not wait for it and discard it's result
                print('***next_trainer is not ready!***')
                continue
        else:
            t2.train_epoch(e)
            loss_2, acc_2 = t2.test_epoch(e)
            print('***[FG] Start Tensor transmission!***')
            trans1 = Transmitter()
            trans1.start()
            trans1.join()
            print('***Tensor transmission done!***')

        print(f'\tNext model loss: \t{loss_2:.4f}')
        print(f'\tNext model accuracy: \t{acc_2:.4f}')
        t2_loss.append(loss_2)
        t2_acc.append(acc_2)

    print(t1_acc)
    print(t1_loss)
    print(t2_acc)
    print(t2_loss)

    if args.overlap:
        overlapped = 'Overlap'
    else:
        overlapped = 'Non-Overlap'
    myplot.plot(t1_acc, t2_acc, f"FreezeOut {overlapped} Accuracy", 1)
    myplot.plot(t1_loss, t2_loss, f"FreezeOut {overlapped} Loss", 2)
    # myplot.plot(myplot.gradually_overlap, myplot.gradually_loss, 3)
    myplot.show()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = opts_parser()
    args = parser.parse_args()

    start = time.time()
    overlap_train(args)
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')


if __name__ == '__main__':
    main()
