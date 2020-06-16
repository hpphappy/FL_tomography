#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:39:20 2020

@author: panpanhuang
"""

import numpy as np
from util import attenuation, generate_fl_signal_from_each_voxel, self_absorption_ratio


def create_XRF_data(n_theta, src_path, sample_size, sample_size_l, theta_ls, element_ls,  fl_lines_xdb, fl_lines, probe_energy,  an_lib, att_cs_lib, probe_intensity_1D, det_energy_u, n_det_energy_bins, det_energy_list, sigma, n_det, det_pos_ls):

    ## create the profile of the probe
    # create 1-D profile along axis 0, and then make it 2-D by copying the 1-D profile array along axis 1
    probe_intensity_2D = probe_intensity_1D * np.ones((sample_size, sample_size))
    # flatten the array: (400,)
    probe_intensity_2D_flat = probe_intensity_2D.flatten()
    # print(probe_intensity_2D_flat.shape)

    ## create the array representing the ratio of remaining intensity after the attenuation along the incident direction of the probe
    # flatten it: (n_theta, sample_size * sample_size)
    att_flat = attenuation(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, an_lib).reshape(n_theta, sample_size * sample_size)
    # print(att_flat.shape)

    ## Calculate the remaining intensity of the probe at each voxel at each sample angle
    att_probe_flat = probe_intensity_2D_flat * att_flat
    # print(att_probe_flat.shape)
    # flatten the calculated results and reshape it into a 2-D array with length=1 along axis 1
    att_probe_flat2 = att_probe_flat.flatten()
    att_probe_flat2 = np.array(att_probe_flat2[:,np.newaxis])
    # print(att_probe_flat2.shape)

    ## Calculate the generated fluorescence signal based on the attenuated probe
    # print(fl_map_tot.shape)
    fl_map_tot = generate_fl_signal_from_each_voxel(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, fl_lines_xdb, fl_lines, probe_energy, an_lib, det_energy_u, n_det_energy_bins, det_energy_list, sigma)
    fl_map_tot_flat = fl_map_tot.reshape(n_theta*sample_size*sample_size, n_det_energy_bins)
    # print(fl_map_tot_flat.shape)
    fl_signal_flat = att_probe_flat2 * fl_map_tot_flat

    SA = self_absorption_ratio(src_path, sample_size, sample_size_l, n_theta, theta_ls, element_ls, att_cs_lib, n_det_energy_bins, n_det, det_pos_ls)
    ## flatten self-absroption matrix
    # print(SA.shape)
    SA_flat = SA.reshape(n_theta*sample_size*sample_size, n_det_energy_bins)

    ## Calculated the fluorescence signal collected by the detector after self-absorption
    fl_signal_SA_flat = fl_signal_flat * SA_flat

    fl_signal_SA = fl_signal_SA_flat.reshape(n_theta, sample_size, sample_size, n_det_energy_bins)

    fl_signal_SA_beamlet = np.sum(fl_signal_SA, axis=2)

    np.save('./data/XRF_sample1.npy', fl_signal_SA_beamlet)
    
    return

def create_XRT_data(src_path, sample_size, sample_size_l, n_theta, theta_ls, element_ls, an_lib, probe_intensity_1D):

    ## create the profile of the probe
    # create 1-D profile along axis 0, and then make it 2-D by copying the 1-D profile array along axis 1
    probe_intensity_2D = probe_intensity_1D * np.ones((sample_size, sample_size))
    # flatten the array: (400,)
    probe_intensity_2D_flat = probe_intensity_2D.flatten()
    # print(probe_intensity_2D_flat.shape)

    ## create the array representing the ratio of remaining intensity after the attenuation along the incident direction of the probe
    # flatten it: (n_theta, sample_size * sample_size)
    att_flat = attenuation(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, an_lib).reshape(n_theta, sample_size * sample_size)
    # print(att_flat.shape)
    
    ## Calculate the remaining intensity of the probe at each voxel at each sample angle
    att_probe_flat = probe_intensity_2D_flat * att_flat
    
    att_probe = att_probe_flat.reshape(n_theta, sample_size, sample_size)
    # print(att_probe.shape)
    # print(att_probe)

    XRT = att_probe[:,:,-1]
    # print(XRT.shape)
    # print(XRT)
    np.save('./data/XRT_sample1.npy', XRT)

    return
