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
import xraydb as xdb
from scipy.ndimage import rotate as sp_rotate
from tqdm import tqdm
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
from mpl_toolkits.axes_grid1 import make_axes_locatable

