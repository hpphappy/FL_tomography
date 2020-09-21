#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:37:00 2020

@author: panpanhuang
"""
import numpy as np
from data_generation_functions import create_XRF_data, create_XRT_data
import xraylib_np as xlib_np
import xraylib as xlib

theta_st = 0
theta_end = 180
n_theta = 12
theta_ls = - np.linspace(theta_st, theta_end, n_theta)

## Define sample size in number of pixles on one side, assuing a N x N-pixel sample
sample_size = 20 
## Define sample size in cm on one side
sample_size_l = 0.01 

src_path = '../data/sample1'
## Define probe posision, the position is defined to pass through the center of the voxel
prob_pos_ls = np.array([x for x in np.arange(sample_size)]) + 0.5

element_ls = np.array(["C", "O", "Si", "Ca", "Fe"])  
an_lib = {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26}
an_ls = np.array(list(an_lib.values()))

## genrate the library of the total attenuation cross section for the involved elements at 20 keV
probe_energy = np.array([20.0])
cs_probe_ls = xlib_np.CS_Total(an_ls, probe_energy).flatten()
cs_probe_lib = dict(zip(element_ls, cs_probe_ls))

aw_ls = xlib_np.AtomicWeight(an_ls)
aw_lib = dict(zip(element_ls, aw_ls)) 

n_det = 5
det_energy_u = 20
n_det_energy_bins = 2000
det_energy_list = np.linspace(det_energy_u / n_det_energy_bins, det_energy_u, n_det_energy_bins)

## genrate the library of the total attenuation cross section for the involved elements from 0-20keV
att_cs_ls = xlib_np.CS_Total(an_ls, det_energy_list)
att_cs_lib = dict(zip(element_ls, att_cs_ls))


## Calculate the size of the voxel (unit in cm) using the length of the sample edge divided by the number of the pixels 
voxel_size = sample_size_l / sample_size

## distance of the XRF detector from the sample edge (in cm)
det_from_sample = 1.6

## diameter of the XRF detector (in cm)
det_size_l = 0.24
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

fl_lines_xdb = np.array(['Ka2', 'Ka1', 'Kb1', 'Lb1', 'La2', 'La1'])
fl_lines = np.array([xlib.KA2_LINE, xlib.KA1_LINE, xlib.KB1_LINE, xlib.LB1_LINE, xlib.LA2_LINE, xlib.LA1_LINE])

sigma = 100 ## eV


src_path = './data/sample1'
probe_intensity_1D = np.ones((sample_size, 1))
create_XRF_data(n_theta, src_path, sample_size, sample_size_l, theta_ls, element_ls, fl_lines_xdb, fl_lines, probe_energy, an_lib, att_cs_lib, probe_intensity_1D, det_energy_u, n_det_energy_bins, det_energy_list, sigma, n_det, det_pos_ls)
create_XRT_data(src_path, sample_size, sample_size_l, n_theta, theta_ls, element_ls, an_lib, probe_intensity_1D)