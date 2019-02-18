#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018
@author: chinwei
"""

import urllib.request
import pickle
import gzip
import os
import numpy as np


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist'],
                        help='dataset name')
    parser.add_argument('--savedir', type=str, default='datasets',
                        help='directory to save the dataset')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'mnist':
        path = 'http://deeplearning.net/data/mnist'
        mnist_filename_all = 'mnist.pkl'
        local_filename = os.path.join(args.savedir, mnist_filename_all)
        urllib.request.urlretrieve(
            "{}/{}.gz".format(path,mnist_filename_all), local_filename+'.gz')
        tr,va,te = pickle.load(gzip.open(local_filename+'.gz','r'), encoding='latin1')
        np.save(open(local_filename+'.npy','wb'), (tr,va,te))
