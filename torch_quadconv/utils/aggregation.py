'''
Mesh aggregation functions.
'''

import torch

'''
Ruge-Stubben aggregation

https://github.com/pyamg/pyamg/blob/main/pyamg/classical/classical.py

Input:
    points: 
    adj: adjacency matrix
    
'''
def ruge_stuben(points, adj, levels):
    
    assert adj.shape[0] == adj.shape[1], 'Adjacency matrix is not square'

    activity = torch.zeros(points.shape[0], levels, dtype=torch.bool)

    for i in range(levels):
        #compute distance adjacency matrix
        dist = adj

        #

    return activity

'''
Smoothed aggregation?

https://github.com/pyamg/pyamg/blob/main/pyamg/aggregation/aggregation.py
'''