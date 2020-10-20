#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:32:19 2020

@author: panpanhuang
"""
import os
import numpy as np
import torch as tc
from data_generation_fns import create_XRT_data_3d
import warnings

warnings.filterwarnings("ignore")

dev = "cpu"

params_3d_5_5_5 = {'src_path': os.path.join('./data/sample3_pad', 'grid_concentration.npy'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(8).to(dev), 
                   'sample_height_n': tc.tensor(5).to(dev),
                   'sample_size_n': tc.tensor(5).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample3_data',
                   'save_fname': 'XRT_sample3',
                   'dev': dev
                  }

params = params_3d_5_5_5


if __name__ == "__main__":  
    XRT_data  = create_XRT_data_3d(**params)

    save_path = params_3d_5_5_5 ["save_path"]
    with open(os.path.join(save_path, 'XRT_data_parameters.txt'), "w") as recon_paras:
        print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')








