#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:12:39 2020

@author: panpanhuang
"""

import numpy as np
import os
import dxchange
from scipy.ndimage import rotate as sp_rotate
import xraylib_np as xlib_np
import xraydb as xdb
from tqdm import tqdm
from mpi4py import MPI



def attenuation(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, an_lib):
    """
    Calculate the attenuation ratio of the incident beam before the beam travels to a certain voxel
    Assuming that the x-ray probe goes along the direction of axis=1 of the sample array
    
    Parameters
    ----------
    theta_ls: ndarray
        The angles that the sample rotates from the initial angle in the experiment
    
    sample_size: int scalar
        sample size in number of pixles on one side, assuing a N x N-pixel sample
    
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

def intersecting_length_fl_detectorlet(n_det, sample_size, sample_size_l, det_pos_ls):
    voxel_size = sample_size_l / sample_size
    ## define index position of center of the source voxel (x1, y1), note that it's shifted by 0.5 to represent the center
    x1, y1 = np.indices((sample_size, sample_size))
    x1, y1 = x1 + 0.5, y1 + 0.5
    voxel_pos_ls = np.dstack((x1, y1))

    ## define sample edges: 
    ## sample_x_edge is the edge that is closer to the XRF detector
    ## sample_y_edge has two components representing the left and the right edge
    sample_x_edge = sample_size
    sample_y_edge = np.array([0, sample_size])


    ## make voxel_pos_ls 1D array for looping: voxel_pos_ls_flat
    voxel_pos_ls_flat =  np.reshape(voxel_pos_ls, (1, voxel_pos_ls.shape[0]*voxel_pos_ls.shape[0], 2))[0]

    P = np.zeros((n_det, sample_size * sample_size, sample_size * sample_size))
    for i, det_pos in enumerate(det_pos_ls):
        for j, v in enumerate(voxel_pos_ls_flat):
            # find x-value when the beam enters the sample WITHOUT intersecting the sample_y_edges(left & right), namely the beam is parallel with the y edge of the sample. 
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
         
            P[i, j, :] = int_length_map.flatten()
    return P

def self_absorption_ratio(src_path, sample_size, sample_size_l, n_theta, theta_ls, element_ls, att_cs_lib, n_det_energy_bins, n_det, det_pos_ls):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    P = intersecting_length_fl_detectorlet(n_det, sample_size, sample_size_l, det_pos_ls)
    SA = np.zeros((n_theta, sample_size * sample_size, n_det_energy_bins))
    for i, theta in enumerate(tqdm(theta_ls.tolist(), leave=False)):
        if rank == 0:
            SA = np.zeros((n_theta, sample_size * sample_size, n_det_energy_bins))
            SA = comm.Reduce(SA, SA, op = MPI.SUM, root=0)
            return SA
            
        else:
            if i % (n_ranks - 1) != rank - 1: continue
            print ("Task number %d (theta = %.2f) being done by processor %d of %d" % (i, theta, rank, n_ranks))
            for j in np.arange(sample_size * sample_size):
        
                att_exponent_elemental_sum = np.zeros((len(element_ls), n_det, n_det_energy_bins))
                for k, element in enumerate(element_ls):
                    concentration_map_fname = os.path.join(src_path, element + '_map.tiff')
                    concentration_map = dxchange.reader.read_tiff(concentration_map_fname)
                    concentration_map_rot = sp_rotate(concentration_map, theta, reshape=False, order=1)
                    ## flattened concentration_map after rotation (n_theta, sample_size * sample_size)
                    concentration_map_rot_flat = concentration_map_rot.flatten()
            
                    ## linear attenuation coefficient for each energy at each voxel: (sample_size * sample_size, n_eneygy_bins)
                    lac = np.array([att_cs_lib[element] * concentration for concentration in concentration_map_rot_flat])
            
                    ## att_exponent = [(intersecting_length_path1 * lac), (intersecting_length_path2 * lac), ..., (intersecting_length_path5 * lac)]:
                    ## att_exponent (for each energy, at each_voxel, for each beam path): (n_det, sample_size * sample_size, n_eneygy_bins)
                    att_exponent = np.array([P[m,j,:][:,np.newaxis] * lac for m in range(n_det)])
            
                    ## att_exponent summing over voxels (for each energy, for each beam path): (n_det, n_eneygy_bins)
                    att_exponent_voxel_sum = np.sum(att_exponent, axis=1)

                    ## filling att_exponent_voxel_sum to att_exponent_elemental_sum for each element
                    att_exponent_elemental_sum[k, :, :] = att_exponent_voxel_sum
        
                ## summing over the attenation exponent contributed by each element
                att_exponent_elemental_sum =  np.sum(att_exponent_elemental_sum, axis=0) 
        
        
                ## calculate the attenuation caused by all elements
                att = np.exp(- att_exponent_elemental_sum)
                ## calculate the attenuation averaged all paths
                att_path_ave = np.average(att, axis=0)
                SA[i,j,:] = att_path_ave
                comm.send(SA, dest = 0)
    MPI.Finalize
    
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx, a.flat[idx]

## Returns fl cross section at each energy in det_energy_list per unit concentration of the element
def MakeFLlinesDictionary(sample_size, sample_size_l, element_name, fl_lines_xdb, fl_lines, probe_energy, an_lib, det_energy_u, n_det_energy_bins):
    voxel_size = sample_size_l / sample_size
    FL_dic = {'element': element_name}
    fl_cs_ls = xlib_np.CS_FluorLine_Kissel_Cascade(np.array([an_lib[element_name]]), fl_lines, probe_energy)
    i = 0 
    detected_fl_unit_concentration = np.zeros((n_det_energy_bins*2))
    det_energy_list_full = np.linspace(- det_energy_u + det_energy_u / n_det_energy_bins, det_energy_u, n_det_energy_bins*2)
    for name, line in xdb.xray_lines(element_name).items():
        if name in set(fl_lines_xdb):
            idx_nearest, value_nearest = find_nearest(det_energy_list_full, line[0])            
            detected_fl_unit_concentration[idx_nearest] += fl_cs_ls[0,i][0]
            i+=1
    FL_dic['detected_fl_unit'] = detected_fl_unit_concentration * voxel_size
    return FL_dic

def Broadening_line(fl_unit, det_energy_list_full, sigma):
    fl_unit_f = np.fft.fft(fl_unit)
    b = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(det_energy_list_full**2)/(2*sigma**2))
    bf = np.fft.fft(b)
    broadening_signal = np.fft.ifftshift(np.fft.ifft(fl_unit_f*bf))
    
    return np.real(broadening_signal)

def generate_fl_signal_from_each_voxel(src_path, n_theta, theta_ls, sample_size, sample_size_l, element_ls, fl_lines_xdb, fl_lines, probe_energy, an_lib, det_energy_u, n_det_energy_bins, det_energy_list, sigma):
    ## Calculate the FL signal emitted when shined by a incident beam with unit intensity
    fl_map_tot = np.zeros((n_theta, sample_size * sample_size, n_det_energy_bins))
    for i, theta in enumerate(theta_ls):
        for j, element in enumerate(element_ls):
            concentration_map_fname = os.path.join(src_path, element + '_map.tiff')
            concentration_map = dxchange.reader.read_tiff(concentration_map_fname)
            concentration_map_rot = sp_rotate(concentration_map, theta, reshape=False, order=1)
            concentration_map_rot_flat = concentration_map_rot.flatten()
        
            ## FL signal emitted at unitary concentration of the current element over 2000 energy bins
            fl_unit = MakeFLlinesDictionary(sample_size, sample_size_l, element, fl_lines_xdb, fl_lines, probe_energy, an_lib, det_energy_u, n_det_energy_bins)['detected_fl_unit']
            ## Broaden FL signal based on the energy resolution of the detector
            det_energy_list_full = np.linspace(- det_energy_u + det_energy_u / n_det_energy_bins, det_energy_u, n_det_energy_bins*2)
            fl_unit_b = Broadening_line(fl_unit, det_energy_list_full, sigma)
        
            ## delete the negative part of the energy array
            n_negative_energy = n_det_energy_bins
            negative_energy_index_ls = np.arange(n_negative_energy).tolist()
            fl_unit_b = np.delete(fl_unit_b, negative_energy_index_ls)
        
            ## FL signal over 2000 energy bins for each voxel
            fl_map = np.array([fl_unit_b * concentration for concentration in concentration_map_rot_flat])
            ## summing over the contribution from all elements
            fl_map_tot[i,:,:] += fl_map
    return  fl_map_tot
