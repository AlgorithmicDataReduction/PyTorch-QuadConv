# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:03:42 2023

@author: andpl
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def MFNUS(xy, fc=1.5, K=10):
    """
    Bengt's Non-Uniform Sample Elimination
    """
    if xy.shape[0] < xy.shape[1]:
        xy = xy.T

    # algorithm
    N = xy.shape[0]  # Get the number of its dots
    sort_ind = np.lexsort(xy.numpy().T,axis=0)
    unsort_dict = {i: sort_ind[i] for i in range(len(xy))}
    xy_ = xy
    xy = xy[sort_ind, :]  # Sort dots from bottom and up
    # Create nearest neighbor pointers and distances
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(xy)
    distances, indices = nbrs.kneighbors(xy)
    jimothy = dict()
    pamuel = dict()
    for k in range(N):  # Loop over nodes from bottom and up
        if indices[k, 0] != N+1:  # Check if node already eliminated
            ind = np.where(distances[k, 1:] < fc*distances[k, 1])[0]
            ind2 = indices[k, ind+1]
            ind2 = np.delete(ind2,ind2 < k)   # Mark nodes above present one, and which
            indices[ind2, 0] = N+1        # are within the factor fc of the closest one
            j = unsort_dict[k]
            ind3 = [unsort_dict[ind] for ind in ind2]   
            jimothy[j] = ind3           # in the original node set
            pamuel[k] = ind2
    elim_ind_sorted = indices[:, 0] != N+1
    elim_ind_unsorted = [unsort_dict[i] for i in np.array(range(len(xy)))[elim_ind_sorted]]
    xy_unsorted = np.zeros_like(xy)
    xy_unsorted[sort_ind] = xy  
    xy_sub_sorted = xy[elim_ind_sorted]
    xy_sub_unsorted = xy_unsorted[elim_ind_unsorted] # Eliminate these marked nodes
    return xy_sub_unsorted, jimothy
