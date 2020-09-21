#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:24:09 2020

@author: panpanhuang
"""


import numpy as np
from util import initialize_object
from data_generation_fns import create_XRF_data_single_theta
import os

def reconstruct_joint_XRF_tomography(n_thread, theta_st, theta_end, n_theta, src_path, det_size_n, det_size_cm, det_from_sample_cm, sample_size_n,
                                     sample_size_cm, this_aN_dic, probe_cts, probe_energy, save_path, 
                                     initial_guess=None, minibatch_size=None, max_epoch, padding=True, loss_function_type="lsq", joint_XRT=False):

    
    XRF_data = np.load(os.path.join(save_path, 'XRF_sample1.npy'))                                         

    def calcualte_loss(obj_element_concentration, this_theta_idx, this_pos_batch, this_prj_XRF_data_batch):
       
        """
        Parameters
        ----------
        obj_element_concentration : ndarray
            dimension of (sample_size_n, sample_size_n, n_element)           
        this_theta_idx : integer
            The sample angle of the current batch represented in the index of the angle.
        this_pos_batch : ndarray
            an array of array of 2 elements of index [[this_theta_idx, 0], [this_theta_idx, 1], ..., [this_theta_idx, max_translational_pos]]
            
        this_prj_XRF_data_batch : ndarray
            
        loss_function_type : TYPE, optional
            DESCRIPTION. The default is "lsq".
        joint_XRT : Boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        ## Import true data of a single theta
        XRF_data_this_batch =  XRF_data[this_theta_idx]
        
        ## Predict the data from the current reconstructed object (the reconstructed elemental concentration)
        XRF_pred_this_batch = create_XRF_data_single_theta(theta_st, theta_end, n_theta, this_theta_idx, obj_element_concentration, det_size_n, det_size_cm, det_from_sample_cm,
                                                      sample_size_n, sample_size_cm, this_aN_dic, probe_cts, probe_energy, save_path, padding=True)
        
        if loss_function_type == "lsq":           
            loss = np.mean(np.square(XRF_data_this_batch - XRF_pred_this_batch).sum())
            
            
        return loss
    
    
    n_batch = XRF_data.shape[0]  #shoould be equal to n_theta
    
    i_epoch, i_batch = 0, 0
    while i_epoch < max_epoch:
        
        for i_batch in range(n_batch):
            if i_epoch == 0 and i_batch == 0:
                obj_elemental_concentration = np.full((), 1.0)
    
    
                       
            
    
    
    