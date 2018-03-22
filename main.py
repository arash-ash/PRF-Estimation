#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:15:16 2018

@author: arash
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from skimage.transform import resize

# experiment parameters
radius = 10.1
precession = 0.1
barWidth = radius / 4
angles = [-90, 45, -180, 315, 90, 225, 0, 135]
nFrames = len(angles)*3
nVoxels = 30
TR = 3.0
TRs = 5 # number of TRs for each frame
t = np.arange(0,nFrames*TRs*TR,TR)
X = Y = np.arange(-radius, radius , precession)
length = len(X)
x, y = np.mgrid[-radius:radius:precession, -radius:radius:precession]
pos = np.dstack((x, y))
    
# parameters for double gamma distribution function hrf:
n1 = 4
lmbd1 = 2.0
t01 = 0
n2 = 7
lmbd2 = 3
t02 = 0
a = 0.3



def generateStim():
    ## Creating bar stimulus 3D array
    stim = np.zeros((nFrames*TRs, length, length))
    stim_short = np.zeros((nFrames, length, length))
    
    f = 0
    for angle in angles:
        for k in range(3):
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    if np.sin(np.deg2rad(angle))*x - np.cos(np.deg2rad(angle))*y <= (0.5-0.5*k)*radius + barWidth:
                        if np.sin(np.deg2rad(angle))*x - np.cos(np.deg2rad(angle))*y >= (0.5-0.5*k)*radius - barWidth:
                            stim_short[f, j, i] = 1
                            for TR in range(TRs):
                                stim[f*TRs + TR, j, i] = 1
            # updates the frame
            f = f + 1
            
    return stim, stim_short

def retinotopicMap(stim, base=1.3):
    stim_transformed_short = np.zeros((nFrames, length, length))
    stim_transformed = np.zeros((nFrames*TRs, length, length))
    
    def findClosestValue(X, value):
        return min(X, key=lambda x: abs(x - value))
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k in range(nFrames):
                if stim[k, i, j]==1:
                    z = np.sqrt(x**2+y**2)
                    z_scaled = np.log(z)/np.log(base)
                    x_transformed = z_scaled*x/z
                    value = findClosestValue(X, x_transformed)
                    i_transformed = np.where(X == value)[0][0]
    
                    y_transformed = z_scaled*y/z
                    value = findClosestValue(Y, y_transformed)
                    j_transformed = np.where(Y == value)[0][0]
    
                    stim_transformed_short[k, i_transformed, j_transformed] = 1
    
    # dividing the stimulus into left and right
    x_half = int(stim.shape[1]/2)
    
    for f in range(nFrames):
        stim_left = stim_transformed_short[f, :, 0:x_half]
        stim_left = np.fliplr(stim_left)
    
        stim_right = stim_transformed_short[f, :, x_half:]
        stim_right = np.fliplr(stim_right)
    
        stim_transformed_short[f, :, 0:x_half] = stim_left
        stim_transformed_short[f, :, x_half:] = stim_right
            
        
    # extending the stimlus time to allow TRs for each event to get neuronals activations
    for frame in range(nFrames):
        for i in range(TRs):
            stim_transformed[frame*TRs + i, :, :] = stim_transformed_short[frame, :, :]

    return stim_transformed
    
def generateData(neuronal_responses, hrf):
    
    ## Hemodynamic Responses
    responses = np.zeros((nFrames*TRs, length, length))
    for i in range(len(X)):
        for j in range(len(Y)):
            n = neuronal_responses[:, i, j]
            responses[:, i, j] = np.convolve(hrf, n)[0:len(t)] + norm.rvs(scale=0.01, size=nFrames*TRs)
    
    ## Downsampling the Hemodynamic Responses
    bolds = np.zeros((nFrames*TRs, nVoxels, nVoxels))
    for frame in range(nFrames*TRs):
        img = responses[frame, :, :]
        img_resized = resize(img, (nVoxels, nVoxels), mode='constant', preserve_range=True)
        bolds[frame, :, :] = img_resized
    
    # show projected response
    shape2D = (nVoxels, nVoxels)
    proj2D = np.zeros(shape2D)
    for i in range(nFrames*TRs):
        proj2D = proj2D + (bolds[i, :, :]).reshape(shape2D)
        
    plt.imshow(proj2D)
    return bolds


    
def estimatePRF(bolds, stim, hrf, xVoxel, yVoxel):
    def MSE(x):
        mean = np.array([x[0], x[1]])
        cov = np.array([[x[2], 0], [0, x[2]]])
        rf = multivariate_normal(mean=mean, cov=cov)
        model = rf.pdf(pos)
        rf_response = np.sum(np.sum(stim*model, axis=1),axis=1)
        rf_response = rf_response / max(np.max(rf_response),1)
        prediction = np.convolve(rf_response, hrf)[:len(t)]
        return np.sum((bolds[:, xVoxel, yVoxel] - prediction)**2)

    x0 = [0.1, 0.1, 1]
    bnds = ((-radius, radius), (-radius, radius), (1e-10, 5))
    res = minimize(MSE, x0, bounds=bnds)

    return res
 
    
def hrf_single_gamma(t,n,lmbd,t0):
    return gamma.pdf(t,n,loc=t0,scale=lmbd)


def hrf_double_gamma(t,n1,n2,lmbd1,lmbd2,t01,t02,a):
    c = (gamma.cdf(t[t.size-1],n1,loc=t01,scale=lmbd1) 
        - a * gamma.cdf(t[t.size-1],n2,loc=t02,scale=lmbd2))
            
    return ( 1/c * (gamma.pdf(t,n1,scale=lmbd1,loc=t01) 
                   - a * gamma.pdf(t,n2,scale=lmbd2,loc=t02)) )
    
    
    

#print('generating stimulus...')
#stim, stim_short = generateStim()

print('Retinotopic mapping of stimulus...')
stim_mapped = retinotopicMap(stim_short, base=1.28)

#print('assuming HRF model...')
#hrf = hrf_double_gamma(t, n1, n2, lmbd1, lmbd2, t01, t02, a)

print('simulating BOLD response...')
bolds = generateData(stim_mapped, hrf)

print('estimating PRF...\n')
res = estimatePRF(bolds, stim, hrf, xVoxel=10, yVoxel=10)

print('x=%.3f, y=%.3f, simga=%.3f'%(res['x'][0],res['x'][1],res['x'][2]))