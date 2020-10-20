#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:05:20 2020

@author: panpanhuang
"""

import numpy as np
from data_generation_fns import attenuation, create_XRT_data

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore")


params_2d_layer_5 = {'src_path': './data/sample3_pad', 
    'theta_st': 0,
    'theta_end': 2 * np.pi,
    'n_theta': 4,
    'sample_size_n': 5,
    'sample_size_cm': 0.01,
    'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
    'probe_energy': np.array([20.0]),
    'probe_cts': 1.0E7,
    'save_path': './data/sample3_data',
    'save_fname': 'XRT_sample3'
    
}


params_2d_layer_map_5 = {'src_path': './data/sample3_pad', 
    'theta_st': 0,
    'theta_end': 2 * np.pi,
    'n_theta': 4,
    'sample_size_n': 5,
    'sample_size_cm': 0.01,
    'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
    'probe_energy': np.array([20.0]),
    
}

params_2d_layer_30 = {'src_path': './data/sample1_pad', 
    'theta_st': 0,
    'theta_end':2 * np.pi,
    'n_theta': 12,
    'sample_size_n': 30,
    'sample_size_cm': 0.06,
    'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
    'probe_energy': np.array([20.0]),
    'probe_cts': 1.0E7,
    'save_path': './data/sample1_data',
    'save_fname': 'XRT_sample1'
    
}


params_2d_layer_map_30 = {'src_path': './data/sample1_pad', 
    'theta_st': 0,
    'theta_end': 2 * np.pi,
    'n_theta': 12,
    'sample_size_n': 30,
    'sample_size_cm': 0.06,
    'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
    'probe_energy': np.array([20.0]),
    
}




if __name__ == "__main__": 
    params = params_2d_layer_5
#     params_map = params_2d_layer_map_5

    ####
    #Create_XRT_data, dim = (n_theta, sample_size_n)
    ####
    XRT_data = create_XRT_data(**params)
#     attenuation_map = attenuation(**params_2d_layer_map)[0].reshape(12, 30, 30)

#     fig1 = plt.figure(figsize=(5,5))
#     gs1 = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1])
#     fig1_ax1 = fig1.add_subplot(gs1[0,0])

#     pos = fig1_ax1.imshow(attenuation_map[0,:,:], cmap='jet', vmin=0, vmax=1.0, origin='upper')
#     divider = make_axes_locatable(fig1_ax1)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     # ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
#     cbar = fig1.colorbar(pos, cax=cax)
#     cbar.ax.tick_params(labelsize=14) 
#     # cbar.ax.set_title('intensity', size='14')


