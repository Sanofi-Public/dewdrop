# Align ensembles of structure predictions for a single protein
# using rotations and translations

#import numpy as np
import torch # MUST IMPORT BEFORE ALIEN!!!
import sys
import os

import alien.tumpy as np

def align_ensemble(ensemble, return_trans=False, iterations=2, batch_dims=0):
    """
    ensemble - ensemble of predictions in 3d space
        Assumption: ensemble.shape = (samples, *point_indices, spatial_dim)
    """
    shape = ensemble.shape
    if batch_dims == 0:
        ensemble = ensemble.reshape(shape[0], -1, shape[-1])
    else:
        raise ValueError("Batched alignment not yet implemented.")

    e0 = ensemble[0]  # (points, spatial_dim)
    
    t = np.zeros_like(ensemble[:,0,:])
    R = None

    for b in range(iterations):
        e0 = e0 - (c0 := e0.mean(axis=0))
        
#         print("initial shape", e0.shape)
        
        ensemble = ensemble - (c := ensemble.mean(axis=1, keepdims=True))
        if return_trans:
            t += c0 - c if R is None else (R.swapaxes(-1, -2) @ (c0 - c)[..., None]).squeeze(start_dim=-1) 

#         print("ensemble shape: ", ensemble.shape)
               
        cov = ensemble.swapaxes(-1, -2) @ e0
        U, S, Vt = np.linalg.svd(cov)
        assert U is not S and U is not Vt
        Rt = U @ Vt

        # Worry about sign? Seems implausible, but we should check:
        sign = np.linalg.det(Rt) > 0
        if (sign < 0).any() :
            U[...,-1] *= sign[...,None]
            Rt = U @ Vt

        dR = Rt.swapaxes(-1, -2)
        R = dR if R is None else dR @ R
        ensemble = (dR[:,None,:,:] @ ensemble[..., None]).squeeze()

        if b + 1 < iterations:
            e0 = ensemble.mean(axis=0, keepdims=True)
#             print("e0 after update: ", e0.shape)

    return (ensemble.reshape(shape), R, t) if return_trans else ensemble.reshape(shape)

        
        
        
