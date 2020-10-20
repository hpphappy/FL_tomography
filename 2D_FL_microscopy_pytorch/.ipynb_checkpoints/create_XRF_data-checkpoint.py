#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:04:35 2020

@author: panpanhuang
"""

import numpy as np
from data_generation_fns import create_XRF_data
import warnings

warnings.filterwarnings("ignore")


params_2d_layer_5 = {'n_thread': 1,
                   'theta_st': 0, 
                   'theta_end': 2 * np.pi,
                   'n_theta': 4, 
                   'src_path': './data/sample3_pad', 
                   'n_det': 5, 
                   'det_size_cm': 0.24, 
                   'det_from_sample_cm': 1.6,
                   'sample_size_n': 5,
                   'sample_size_cm': 0.01, 
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': 1.0E7, 
                   'probe_energy': np.array([20.0]),
                   'save_path': './data/sample3_data',
                   'save_fname': 'XRF_sample3'
                  }


params_2d_layer_30 = {'n_thread': 3,
                   'theta_st': 0, 
                   'theta_end': 2 * np.pi,
                   'n_theta': 12, 
                   'src_path': './data/sample1_pad', 
                   'n_det': 5, 
                   'det_size_cm': 0.24, 
                   'det_from_sample_cm': 1.6,
                   'sample_size_n': 30,
                   'sample_size_cm': 0.06, 
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': 1.0E7, 
                   'probe_energy': np.array([20.0]),
                   'save_path': './data/sample1_data',
                   'save_fname': 'XRF_sample1'
                  }


if __name__ == "__main__":
    params = params_2d_layer_30
    XRF_data = create_XRF_data(**params)

