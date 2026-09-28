"""
Adopted from
https://github.com/lolacaro/PCMRI-MSAC
"""


import numpy as np
from skimage import filters
from einops import rearrange


def execute_MSAC(im, corr_fit_order=3, th=0.01):
    """
    dimensions if 4d [dim1 dim2 slice time velocity]
    dimensions if 2d [dim1 dim2 1 time velocity]
    """
    np.random.seed(274612)
    im = rearrange(im, 'nv nt spe pe fe -> fe pe spe nt nv')
    im = im[..., 1:] * np.conj(im[..., 0:1]) / (np.abs(im[..., 0:1])+1e-9)
    phase_im_t = np.angle(im) / np.pi  # scale to [-1,1] (m n l t d)
    magnitude_im_t = np.abs(im)

    parameters = dict()
    parameters['msac_thresh'] = th  # MSAC threshold in #venc
    parameters['samples'] = 10  # MSAC sample size
    parameters['trials'] = 100  # MSAC trials
    parameters['msac_fit_order'] = 1  # linear fit during MSAC
    parameters['n_enc'] = im.shape[-1]  # number of velocity encodings
    magnitude_im_t = np.mean(magnitude_im_t, axis=-1)  # average magnitude of segments (m n l t)
    magnitude_im_t = magnitude_im_t/np.max(magnitude_im_t)  # scale to peak value

    magnitude = np.mean(magnitude_im_t,3)  # m n l
    phase = np.mean(phase_im_t,3)  # m n l d
    mask_mgn = magnitude > filters.threshold_multiotsu(magnitude, classes=5)[0]  # create magnitude mask (m n l)
    # create [dat x y z] point-arrays for all pixels ap and pixels in magnitude mask mp
    m, n, l, t, d = phase_im_t.shape
    run_4D = d == 3

    if run_4D:
        allm = np.ones([m,n,l])
        aa = np.argwhere(allm == 1)  # all data points (m n l) 
        ma = np.argwhere(mask_mgn == 1)  # magnitude points (m n l)

        ap = np.zeros((aa.shape[0], 6))  # m 6 
        mp = np.zeros((ma.shape[0], 6))  # m 6
        ap[:,3:6] = aa
        mp[:,3:6] = ma
        for dir in np.arange(d):
            pdir = phase[:,:,:,dir]  # m n l
            ap[:,dir] = pdir[allm == 1]
            mp[:,dir] = pdir[mask_mgn == 1]
    
    else:
        allm = np.ones([m,n])
        aa = np.argwhere(allm == 1)  # all data points (m n) 
        ma = np.argwhere(mask_mgn[:,:,0] == 1)  # magnitude points (m n)

        ap = np.zeros((aa.shape[0], 3))
        mp = np.zeros((ma.shape[0], 3))
        ap[:,1:3] = aa
        mp[:,1:3] = ma

        ap[:,0] = np.squeeze(phase[allm == 1])
        mp[:,0] = np.squeeze(phase[mask_mgn[:,:,0] == 1])
        
    # Define corr fit, msac fit, msac dist function and corr/msac feval functions
    mfunc, cfunc, fmfunc, fcfunc, dfunc = get_functions(run_4D, parameters['msac_fit_order'], corr_fit_order)
    
    functions = dict()
    functions['msac_fit'] = mfunc
    functions['msac_dist'] = dfunc
    cost, inlierIndx = msac(mp, parameters, functions)

    model = cfunc(mp, **{'inlierIndx': inlierIndx})  # fit using MSAC inliers
    est, xyz = fcfunc(model, ap)  # evaluate fit on image

    # define MSAC mask from inlierIndx returned by MSAC and background fit
    bgr_msac = np.zeros([m,n,l,d])
    aw = np.where(allm == 1)  # all data points

    if run_4D:
        bgr_msac[aw[0], aw[1], aw[2],:] = est
    else:
        bgr_msac[aw[0], aw[1]] = est[:,:,np.newaxis]

    corr_t = np.swapaxes(np.expand_dims(bgr_msac, -1), 3, 4)  # expand t, put in order m n l t d
    corr_t = rearrange(corr_t, 'fe pe spe nt nv -> nv nt spe pe fe ')
    return corr_t * np.pi
    # corr_tres = (phase_im_t - corr_t) * np.pi  # rescale back
    # corr_im = np.abs(im) * np.cos(corr_tres) + 1j * np.abs(im) * np.sin(corr_tres)
    # return corr_im, corr_t


def get_functions(run_4D, fitorder_msac, fitorder_corr):
    """
    Creates function handles for polynomial fits up to 3rd order
    for PC-MRI 2D or 4D flow data background phase correction using MSAC
    4D flow:  each velocity encoding dimension is fitted individually
    
    Input:
    fitorder_msac  - int 0-3; polymial fit order used during MSAC
    fitorder_corr  - int 0-3; polymial fit order used for background phase correction
    
    Output:
    mfunc   - function handle, fit MSAC
    cfunc   - function handle, fit correction
    dfunc   - function handle, evalFunc MSAC (computes residuals)
    fmfunc  - function handle, use fitted MSAC model on data
    fcfunc  - function handle, use fitted correction model on data
    """
    if run_4D:
        #fit MSAC
        mfunc = lambda points, **kwargs: fit4D(fitorder_msac, points, **kwargs)
        #fit correction
        cfunc = lambda points, **kwargs: fit4D(fitorder_corr, points, **kwargs)

        #feval MSAC
        fmfunc = lambda coeffs, points: eval4D(fitorder_msac, coeffs, points)
        #feval correction
        fcfunc = lambda coeffs, points: eval4D(fitorder_corr, coeffs, points)
        #distance function MSAC
        dfunc = lambda coeffs, points: dist4D(fitorder_msac, coeffs, points)
    else:
        #fit MSAC
        mfunc = lambda points, **kwargs: fit2D(fitorder_msac, points, **kwargs)
        #fit correction
        cfunc = lambda points, **kwargs: fit2D(fitorder_corr, points, **kwargs)

        #feval MSAC
        fmfunc = lambda coeffs, points: eval2D(fitorder_msac, coeffs, points)
        #feval correction
        fcfunc = lambda coeffs, points: eval2D(fitorder_corr, coeffs, points)
        #distance function MSAC
        dfunc = lambda coeffs, points: dist2D(fitorder_msac, coeffs, points)
    return mfunc, cfunc, fmfunc, fcfunc, dfunc

def getInOut4D(order, points):
    #separate input / output
    xyz = points[:,0:3]
    p1 = points[:,3]
    p2 = points[:,4]
    p3 = points[:,5]
    noP = points.shape[0]
    #get minimal samplesSize and input configs for different fit orders
    if order == 0:
        A = np.ones([noP, 1])
        no = 1
    elif order == 1:
        A = np.ones([noP, 4])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        no = 4
    elif order == 2:
        A = np.ones([noP, 10])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        A[:,4] = p1**2
        A[:,5] = p1*p2
        A[:,6]=p1*p3
        A[:,7] = p2**2
        A[:,8] = p2*p3
        A[:,9]=p3**2
        no = 10
    elif order == 3:
        A = np.ones([noP, 20])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        A[:,4] = (p1**2)
        A[:,5] = p1*p2
        A[:,6] = p1*p3
        A[:,7] = (p2**2)
        A[:,8] = p2*p3
        A[:,9] = (p3**2)
        A[:,10] = p1**3
        A[:,11] = (p1**2)*p2
        A[:,12] = (p1**2)*p3
        A[:,13] = p2**3
        A[:,14] = (p2**2)*p1
        A[:,15] = (p2**2)*p3
        A[:,16] = p3**3
        A[:,17] = (p3**2)*p1
        A[:,18] = (p3**2)*p2
        A[:,19] = p1*p2*p3
        no = 20
    return xyz, A, no

def fit4D(order, points, **kwargs):
    inlierIndx = np.ones((points.shape[0], 3), dtype=bool) #select all points
    if len(kwargs) > 0: #define selection if not given
        inlierIndx = kwargs['inlierIndx']

    indx = inlierIndx == 1
    xyz, A, no = getInOut4D(order, points) #get input matrix A and output xyz
    coeffs = np.zeros((no, 3)) #define coefficent matrix
    for d in np.arange(3): #fit for each velocity dimension
        if order == 0:
            coeffs[0,d] = np.mean(xyz[indx[:,d],d], 0)
        else:
            inlier = indx[:,d]
            coeffs[:,d] = np.linalg.lstsq(A[inlier,:],xyz[inlier,d], rcond=None)[0]
    return coeffs

def dist4D(order, coeffs, points):
    est, xyz = eval4D(order, coeffs, points) #use a model on points
    dist = np.abs((xyz-est)/2) #get residuals
    return dist

def eval4D(order, coeffs, points):
    [xyz, A, no] = getInOut4D(order, points) #get input matrix A and output xyz
    est = A@coeffs #evaluate model on points in A
    return est, xyz

def getInOut2D(order, points):
    #separate input / output
    xyz = points[:,0:1]
    p1 = points[:,1]
    p2 = points[:,2]
    noP = points.shape[0]
    #get minimal samplesSize and input configs for different fit orders
    if order == 0:
        A = np.ones([noP, 1])
        no = 1
    elif order == 1:
        A = np.ones([noP, 3])
        A[:,1] = p1
        A[:,2] = p2
        no = 3
    elif order == 2:
        A = np.ones([noP, 6])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p1**2
        A[:,4] = p1*p2
        A[:,5] = p2**2
        no = 6
    elif order == 3:
        A = np.ones([noP, 10])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p1**2
        A[:,4] = p1*p2
        A[:,5] = p2**2
        A[:,6] = p1**3
        A[:,7] = (p1**2)*p2
        A[:,8] = p2**3
        A[:,9] = (p2**2)*p1
        no = 10
    return xyz, A, no

def fit2D(order, points, **kwargs):

    inlierIndx = np.ones((points.shape[0],1), dtype=bool) #select all points
    if len(kwargs) > 0: #define selection if not given
        inlierIndx = kwargs['inlierIndx']

    indx = (inlierIndx == 1)[:,0]
    xyz, A, no = getInOut2D(order, points) #get input matrix A and output xyz
    coeffs = np.zeros((no,1)) #define coefficent matrix

    if order == 0:
        coeffs[0,0] = np.mean(xyz[indx,0], 0)
    else:
        coeffs[:,0]= np.linalg.lstsq(A[indx,:],xyz[indx,0], rcond=None)[0]
    return coeffs


def dist2D(order, coeffs, points):
    est, xyz = eval2D(order, coeffs, points) #use a model on points
    dist = np.abs((xyz-est)/2) #get residuals
    return dist

def eval2D(order, coeffs, points):
    [xyz, A, no] = getInOut2D(order, points) #get input matrix A and output xyz
    est= A@coeffs #evaluate model on points in A
    return est, xyz

def msac(points, parameters, functions):
    """
    MSAC M-estimator SAmple Consensus (MSAC) algorithm for outlier
    sensitive stationary tissue masks for PC-MRI 2D or 4D flow data
    4D flow: each velocity encoding dimension is fitted and updated individually
    
    Input:
    points      - M-by-3 or M-by-6 for [vel x y] or [vel1 vel2 vel3 x y z] data
    parameters  - dict containing the following fields:
                  samples
                  msac_thresh
                  trials
                  flow_dimensions
    functions   - dict containing the following function handles
                  msac_fit
                  msac_dist
    
    Output:
    bestInliers - logical array of length M, to mark inliers in points
                  derived by MSAC
    bestCost    - double, MSAC cost with above inliers
    
    References:
    P. H. S. Torr and A. Zisserman, "MLESAC: A New Robust Estimator with
    Application to Estimating Image Geometry," Computer Vision and Image
    Understanding, 2000.
    """

    #retrieve parameters
    samples = parameters['samples']
    threshold = parameters['msac_thresh']
    trials = parameters['trials']
    n_enc = parameters['n_enc']

    msacFit = functions['msac_fit']
    msacDist = functions['msac_dist']

    #get number of points
    noP = points.shape[0]

    #define worst case as initial guess
    bestCost = np.ones(n_enc)*threshold*noP
    bestInliers = np.zeros([noP, n_enc])

    #iterate through trials
    for i in np.arange(trials):

        #draw sample
        indx = np.random.permutation(noP)[0:samples]
        sample = points[indx,:]

        #fit to sample
        coeffs = msacFit(sample)
        #get residuals
        residuals = msacDist(coeffs, points)

        #compare to threshold
        residuals[residuals > threshold] = threshold
        inliers = residuals < threshold

        #cost
        cost = np.sum(residuals,0)
        compCost = bestCost > cost

        #adapt
        bestCost[compCost] = cost[compCost]
        bestInliers[:,compCost] = inliers[:,compCost]

    return bestCost, bestInliers
