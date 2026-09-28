"""
Adapted from 
https://github.com/byaman14/SSDU
"""


import numpy as np
import time


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.
    Returns
    -------
    list of >=2D indices containing non-zero locations
    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def gen_center_mask(kspace_shape, r2):
    """assumes PE to be in last two dimensions"""
    kspace_center = np.zeros(kspace_shape, dtype=bool)
    for x in range(kspace_shape[-2]):
        for y in range(kspace_shape[-1]):
            if (x-kspace_shape[-2]//2)**2 + (y-kspace_shape[-1]//2)**2 < r2:
                kspace_center[...,x,y]=True
    return kspace_center

def uniform_disjoint_selection(input_mask, rho=0.4, r2=9, seed=None, venc_coherence=False):
    """
    Divides acquired points into two disjoint sets based on uniform distribution

    input_mask: input mask, venc x card x nrow x ncol
    rho: split ratio for training and loss mask. \ rho = |\Lambda|/|\Omega|
    small_acs_block: keeps a small acs region fully-sampled for training masks if there is no acs region, the small acs block should be set to zero
    seed: make 'random' uniform selection repeatable for initial train val split
    venc_coherence: All velocity encodings are partitioned the same (only done for brain data)
    """
    nvenc, ncard, nrow, ncol = input_mask.shape[0], input_mask.shape[1], input_mask.shape[2], input_mask.shape[3]

    if venc_coherence:
        temp_mask = np.sum(np.copy(input_mask), axis=(0)) >= 1
    else:
        temp_mask = np.copy(input_mask)
    kspace_center = gen_center_mask(temp_mask.shape, r2=r2)
    temp_mask[kspace_center] = 0

    if seed is not None:
        np.random.seed(seed)
    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(temp_mask.size), size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    if venc_coherence:
        [ind_card, ind_x, ind_y] = index_flatten2nd(ind, (ncard, nrow, ncol))
    else:
        [ind_venc, ind_card, ind_x, ind_y] = index_flatten2nd(ind, (nvenc, ncard, nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    if venc_coherence:
        loss_mask[:, ind_card,  ind_x, ind_y] = 1
    else:
        loss_mask[ind_venc, ind_card,  ind_x, ind_y] = 1
    loss_mask *= input_mask  # sampling for vencs is not identical

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask
