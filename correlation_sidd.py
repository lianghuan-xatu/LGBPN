import os, sys

sys.path.append(os.path.join(__file__.split('correlation_sidd')[0], 'correlation_sidd'))

import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datahandler import get_dataset_class


args = argparse.ArgumentParser()
args.add_argument('-c', default=None,   type=str)

args = args.parse_args()

# setting
cal_range = 8
camera_code = [args.c] if args.c is not None else ['GP', 'IP', 'S6', 'N6', 'G4']

# define init tensors
count = 0
mean = torch.zeros((1,1))
variance = torch.zeros((1,1))
sub_cross_mul = torch.zeros((cal_range+1, cal_range+1))
sub_center_square = torch.zeros((1,1))


sidd_dataset = get_dataset_class('SIDD')()

# mean
for d_idx in range(sidd_dataset.__len__()):
    if sidd_dataset.img_paths[d_idx]['instances']['smartphone_camera_code'] in camera_code:
        data = sidd_dataset.__getitem__(d_idx)
        noise_map = data['noisy'] - data['clean']
        mean += noise_map.sum()
        count += noise_map.shape[0]*noise_map.shape[1]*noise_map.shape[2]
    print('cal mean... %d/%d'%(d_idx, sidd_dataset.__len__()))

mean /= count
count = 0
print('mean: ', mean)

# cal variance
for d_idx in range(sidd_dataset.__len__()):
    if sidd_dataset.img_paths[d_idx]['instances']['smartphone_camera_code'] in camera_code:
        data = sidd_dataset.__getitem__(d_idx)

        noise_map = data['noisy'] - data['clean']
        # subtract mean
        noise_map -= mean
        # add center square
        sub_center_square += noise_map.square().sum()
        count += noise_map.shape[0]*noise_map.shape[1]*noise_map.shape[2]
    print('cal var... %d/%d'%(d_idx, sidd_dataset.__len__()))

variance  = sub_center_square/count
print('variance : ', variance)

# cal co-variance
count = torch.zeros((cal_range+1, cal_range+1))
for d_idx in range(sidd_dataset.__len__()):
    if sidd_dataset.img_paths[d_idx]['instances']['smartphone_camera_code'] in camera_code:
        data = sidd_dataset.__getitem__(d_idx)

        noise_map = data['noisy'] - data['clean']

        # subtract mean
        noise_map -= mean

        # calculate crossed multiplication
        for h_idx in range(cal_range+1):
            for w_idx in range(cal_range+1):
                if w_idx != 0: left = noise_map[:,h_idx:,:-w_idx]
                else:         left = noise_map[:,h_idx:,:]
                if h_idx != 0 and w_idx !=0:    right = noise_map[:,:-h_idx,w_idx:]
                elif h_idx == 0 and w_idx != 0: right = noise_map[:,:,w_idx:]
                elif h_idx != 0 and w_idx == 0: right = noise_map[:,:-h_idx,:]
                else:                           right = noise_map[:,:,:]
                sub_cross_mul[h_idx][w_idx] += (left*right).sum()
                count[h_idx][w_idx] += noise_map.shape[0]*(noise_map.shape[1]-h_idx)*(noise_map.shape[2]-w_idx)
    print('cal covar... %d/%d'%(d_idx, sidd_dataset.__len__()))

sub_cross_mul /= count
print('covariance: ', sub_cross_mul)
print('correlation; ', sub_cross_mul/variance)
