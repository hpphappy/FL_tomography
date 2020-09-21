#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:36:53 2020

@author: panpanhuang
"""

import numpy as np
import os
import dxchange
from scipy.ndimage import rotate as sp_rotate
import xraylib_np as xlib_np
import xraydb as xdb
from tqdm import tqdm
from functools import partial 
from multiprocessing import Pool

def attenuation(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, an_lib):
    """
    Calculate the attenuation ratio of the incident beam before the beam travels to a certain voxel
    Assuming that the x-ray probe goes along the direction of axis=1 of the sample array
    
    Parameters
    ----------
    theta_ls: ndarray
        The angles that the sample rotates from the initial angle in the experiment
    
    sample_size: int scalar
        sample size in number of pixles on one side, assuming a N x N-pixel sample
    
    sample_size_l: scalar
        sample size in mm
    
    element_ls: ndarray
        elements in the sample
        
    Returns: ndarray
    -------
        dimension of the returned array is n_theta x sample_size x sample_size

    """

    an_ls = np.array(list(an_lib.values()))
    probe_energy = np.array([20.0])

    ## genrate the library of the total attenuation cross section for the involved elements at 20 keV
    cs_probe_ls = xlib_np.CS_Total(an_ls, probe_energy).flatten()
    cs_probe_lib = dict(zip(element_ls, cs_probe_ls))
    
    att_acc_map = np.zeros((n_theta, sample_size, sample_size))
    for i, theta in enumerate(theta_ls):
        for j, element in enumerate(element_ls):
            concentration_map_fname = os.path.join(src_path, element + '_map.tiff')
            concentration_map = dxchange.reader.read_tiff(concentration_map_fname)
            concentration_map_rot = sp_rotate(concentration_map, theta, reshape=False, order=1)
            lac_single = concentration_map_rot * cs_probe_lib[element]            
            lac_acc = np.cumsum(lac_single, axis=1)  
            lac_acc = np.insert(lac_acc, 0, np.zeros(sample_size), axis=1)
            lac_acc = np.delete(lac_acc, sample_size, axis=1)
            att_acc = lac_acc * (sample_size_l / sample_size)
            att_acc_map[i,:,:] += att_acc
    return np.exp(-att_acc_map)


