#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:24:09 2020

@author: panpanhuang
"""


import numpy as np
from util import initialize_object
from data_generation_fns import create_XRF_data, create_XRT_data

def reconstruct_joint_XRF_tomography(n_thread, theta_st, theta_end, n_theta, src_path, det_size_n, det_size_cm, det_from_sample_cm, sample_size_n,
                                     sample_size_cm, this_aN_dic, probe_cts, probe_energy, save_path, 
                                     initial_guess=None, minibatch_size=None, padding=True):

    
                                                     
        XRF_data_from_obj = create_XRF_data(n_thread, theta_st, theta_end, n_theta, src_path, det_size_n, det_size_cm, det_from_sample_cm, 
                                                 sample_size_n, sample_size_cm, this_aN_dic, probe_cts, probe_energy, save_path, padding)

    def loss(obj_element_concentration, this_theta_idx, this_pos_batch, this_prj_XRF_data_batch, loss_function_type="lsq", joint_XRT=False):
       
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
        ## 
        if loss_function_type is "lsq":
                       
            this_prj_XRF_pred_batch = this_XRF_model[this_theta_idx, :, :]
                        
            loss = 
            
    
    