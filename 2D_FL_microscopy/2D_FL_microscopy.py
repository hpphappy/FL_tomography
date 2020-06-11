#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:39:20 2020

@author: panpanhuang
"""

import numpy as np
import dxchange
import xraylib as xlib
import xraylib_np as xlib_np
from scipy.ndimage import rotate as sp_rotate
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
from mpl_toolkits.axes_grid1 import make_axes_locatable

##Define the rotating angles of the sample
theta_st = 0
theta_end = 180
n_theta = 12
theta_ls = - np.linspace(theta_st, theta_end, n_theta)

## Define sample size in number of pixles on one side, assuing a N x N-pixel sample
sample_size = 20 

## Define sample size in mm on one side
sample_size_l = 0.1 

## Define probe posision, the position is defined to pass through the center of the voxel
prob_pos_ls = np.array([x for x in np.arange(sample_size)]) + 0.5
# print(prob_pos_ls)

element_ls = np.array(["C", "O", "Si", "Ca", "Fe"])  
an_lib = {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26}

an_ls = np.array(list(an_lib.values()))
photon_energy = np.array([20.0])

## genrate library of the total cross section for the involved elements
cs_ls = xlib_np.CS_Total(an_ls, photon_energy).flatten()
cs_lib = dict(zip(element_ls, cs_ls))

## genrate library of the atomic weight for the involved elements
aw_ls = xlib_np.AtomicWeight(an_ls)
aw_lib = dict(zip(element_ls, aw_ls)) 

## Calculate the size of the voxel (unit in mm) using the length of the sample edge divided by the number of the pixels 
voxel_size = sample_size_l / sample_size
# print(voxel_size)

## distance of the XRF detector from the sample edge
det_from_sample = 16

## diameter of the XRF detector
det_size_l = 2.4
det_size = np.ceil(det_size_l/voxel_size)

## number of the detectorlets
n_det = 5

## x index of the location of the XRF detector
det_axis_0_pixel = sample_size + np.ceil(det_from_sample/voxel_size) + 0.5
det_axis_0_pixel_ls = np.full(n_det, det_axis_0_pixel)

## y index of the location of the XRF detector
det_axis_1_pixel_ls = np.linspace(np.ceil(sample_size/2) - det_size, 
                                  np.ceil(sample_size/2) + det_size, n_det) + 0.5

## biding x-index and y-index array into [(x1,y1), (x2,y2), ..., (x_Ndet, y_Ndet)]
det_pos_ls = np.array(list(zip(det_axis_0_pixel_ls, det_axis_1_pixel_ls)))

# print(det_axis_0_pixel_ls)
# print(det_axis_1_pixel_ls)
# print(det_pos_ls)
# print(det_pos_ls[0])

## define a line passing through the FL emitting from the center of the source voxel (x1, y1) in the 2D sample 
## and the detectorlet (xd, yd)
def trace_beam_yint(x1, y1, xd, yd, sample_x_edge):
    m = (yd - y1)/(xd - x1)
    y_int = m * (sample_x_edge - x1) + y1
    return m, y_int

def trace_beam_xint(x1, y1, xd, yd, sample_y_edge):
    if yd == y1:
        m = 0
        x_int = np.array([])
    else:
        m = (yd - y1)/(xd - x1)
        x_int = (sample_y_edge - y1)/m + x1
    return m, x_int


## define index position of center of the source voxel (x1, y1), note that it's shifted by 0.5 to represent the center
x1, y1 = np.indices((sample_size, sample_size))
x1, y1 = x1 + 0.5, y1 + 0.5
voxel_pos_ls = np.dstack((x1, y1))
# print(voxel_pos_ls[0][0])

sample_x_edge = sample_size
sample_y_edge = np.array([0, sample_size])

import pandas as pd
voxel_pos_ls_flat =  np.reshape(voxel_pos_ls, (1, voxel_pos_ls.shape[0]*voxel_pos_ls.shape[0], 2))[0]
P = np.zeros((n_det, sample_size * sample_size, sample_size, sample_size))

for i, det_pos in enumerate(det_pos_ls):
    for j, v in enumerate(voxel_pos_ls_flat):
        # find x-value when the beam passes through sample_y_edges(left & right), the one with larger x is the intersection with lower edge
        if v[1] == det_pos[1]:
            xint = sample_size
        else:
            xint = np.max(trace_beam_xint(v[0], v[1], det_pos[0], det_pos[1], sample_y_edge)[1])
        xint_sample = np.clip(xint, 0, sample_size)
        
        # find y-value when the beam passes through sample_x_edge(bottom)
        m = trace_beam_yint(v[0], v[1], det_pos[0], det_pos[1], sample_x_edge)[0]
        yint = trace_beam_yint(v[0], v[1], det_pos[0], det_pos[1], sample_x_edge)[1]
        yint_sample = np.clip(yint, 0, sample_size)
    
               
        # when the beam enters a voxel, it either intersects the x boundary or y boundary of the voxel
        # find the x,y-value of the boundary except the ones on the sample edge
        if np.floor(xint_sample) != np.floor(v[0]):
            x_edge_ls = np.linspace(np.ceil(xint_sample)-1, np.ceil(v[0]), int(np.abs(np.ceil(xint_sample) - np.ceil(v[0]))))
        else: 
            x_edge_ls = np.array([])
            
        
        if np.floor(yint_sample) != np.floor(v[1]):            
            if m < 0:
                y_edge_ls = np.linspace(np.floor(yint_sample)+1, np.floor(v[1]), int(np.abs(np.floor(yint_sample)+1 - np.floor(v[1]))) + 1)            
           
            if m > 0:
                y_edge_ls = np.linspace(np.ceil(yint_sample)-1, np.ceil(v[1]), int(np.abs(np.ceil(yint_sample) - np.ceil(v[1]))))
        else:
            y_edge_ls = np.array([])
        
        
        # find all intersections (except the initial intersection): 
        # 1. find y-value of intersection given x_edge_ls
        # 2. find x-value of intersection given y_edge_ls
        y_int_x_edge_ls = trace_beam_yint(v[0], v[1], det_pos[0], det_pos[1], x_edge_ls)[1] #solve y intersections given x edge
        x_int_y_edge_ls = trace_beam_xint(v[0], v[1], det_pos[0], det_pos[1], y_edge_ls)[1] #solve x intersections given y edge
        
        # compile the x,y coordinates of the intersection: (x,y) = (x_edge_ls, y_int_x_edge_ls) and (x_int_y_edge_ls,y_edge_ls)
        int_x_edge_ls = np.dstack((x_edge_ls,y_int_x_edge_ls))[0]
        int_y_edge_ls = np.dstack((x_int_y_edge_ls,y_edge_ls))[0]

        # sort them using the x coordinate
        int_ls = np.concatenate((int_x_edge_ls, int_y_edge_ls))
        int_ls = np.vstack((np.array([xint_sample, yint_sample]), int_ls))
        int_ls = int_ls[np.argsort(int_ls[:,0])]
        
        # calculate the intersecting length in the intersecting voxels
        int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2) 
        # just in case that we count some intersections twice, delete the duplicates
        idx_duplicate = np.array(np.where(int_length==0)).flatten()
        int_ls = np.delete(int_ls, idx_duplicate, 0)
        int_length = np.delete(int_length, idx_duplicate) 
        
        # determine the indices of the intersecting voxels according to the intersecting x,y-coordinates
        int_ls_shift = np.zeros((int_ls.shape))
        int_ls_shift[1:] = int_ls[:-1]
        int_idx = np.floor((int_ls_shift + int_ls_shift)/2)[1:]        
        int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'))
        
        # construct the int_length_map, and scale the intersecting length based on the voxel size
        int_length_map = np.zeros((sample_size, sample_size))
        int_length_map[int_idx] = int_length * voxel_size
         
        # determine the index of the current FL-emitting voxel, and fill P
        x_idx_FL_vox = j // 20
        y_idx_FL_vox = j % 20
        P[i, j, :, :] = int_length_map

# b = np.round(P[0,0,:,:])
# c = np.sum(P[:,0,:,:], axis=0)
# a = pd.DataFrame(c)        
# a.columns = ['']*a.shape[1]
# print(a.to_string(index=False))  

def stack(arr, count):
    return np.stack([arr for _ in range(count)], axis=0)

SA = np.zeros((len(theta_ls), sample_size, sample_size))
SA_temp = np.zeros((len(theta_ls), sample_size * sample_size))
m = 0
for i, theta in enumerate(tqdm(theta_ls.tolist())):
    m += 1
    n = 0
    for j in tqdm(np.arange(sample_size * sample_size)):
        n += 1
#         print(m)
#         print(n)
        att_elemental_sum = 0
        for k, element in enumerate(element_ls):
            concentration_map = dxchange.reader.read_tiff(element + '_map.tiff')
            concentration_map_rot = sp_rotate(concentration_map, theta, reshape=False, order=1)
            lac_single = concentration_map_rot * cs_lib[element]
            lac_single = stack(lac_single, n_det)            
            att = P[:,j,:,:] * lac_single
            att_avg = np.mean(np.array([np.sum(a) for a in att]))
            att_elemental_sum += att_avg
        x_idx_FL_vox = j // 20
        y_idx_FL_vox = j % 20
        SA[i, x_idx_FL_vox, y_idx_FL_vox] = att_elemental_sum