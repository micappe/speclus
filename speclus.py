#!/usr/bin/env python

"""
Copyright Michele Cappellari

V1.0.0: Written by Michele Cappellari, Oxford, 21 March 2016
V1.1.0: Cleaned up code. MC, Oxford, 9 March 2017
V1.1.1: GitHub release. MC, Oxford, 21 December 2018

"""

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import ppxf

###############################################################################

def spectra_distance(star, template, mask, degree):

    # Make ppxf fit without any velocity shift and with no sigma broadening.
    # Only the polynomials are fitted.
    velscale = 1.     # Arbitrary number: we work in pixels units
    start = [0, 1.]   # Negligible sigma: OK with >2016 Fourier pPXF code
    noise = np.ones_like(star)
    pp = ppxf(template, star, noise, velscale, start, moments=2,
              degree=degree, linear=True, mask=mask)
    resid = star - pp.bestfit
    dist = np.percentile(np.abs(resid[mask]), 95.45)/np.mean(star[mask])*50  # 2sigma/2 in %

    return dist, pp


##############################################################################

def save_distance_matrix(templates, plot, mask, degree):

    npix, ntemp = templates.shape
    dist = np.zeros((ntemp, ntemp))
    for j, temp1 in enumerate(templates.T):
        for k, temp2 in enumerate(templates.T):
            if k > j:
                print(f"Templates ({j}, {k}) ##################################################")
                dist[j, k], pp = spectra_distance(temp1, temp2, mask, degree)
                print(f"{j:5d} {k:5d} {dist[j, k]:5.1f}")
                if plot:
                    plt.clf()
                    pp.plot()
                    plt.pause(0.01)

    # Make matrix symmetric by filling lower triangle with transposed upper one
    dist += np.triu(dist).T
    dist = squareform(dist)

    np.savez_compressed('distance_matrix.npz', data_dist=dist)


##############################################################################

def speclus(templates, initialize=False, mask=None, plot=False,
            threshold=5, degree=8):
    """
    Implements the selection of subsets of spectral templates,
    using hierarchical clustering, described in Section 5 of
    Westfall et al. 2019 (AJ submitted and arXiv)

    Returns an integer vector with of length of the number of input templates,
    stating which templates belong to each cluster and an array with the
    co-added templates.

    """
    if initialize:
        save_distance_matrix(templates, plot, mask, degree)

    f = np.load('distance_matrix.npz')

    data_dist = f['data_dist']

    data_link = linkage(data_dist,  method='average')
    ind = fcluster(data_link, threshold, 'distance') - 1  # Start from 0
    uind = np.unique(ind)
    print(f"Clusters: {uind.size}")

    npix, ntemp = templates.shape
    templates /= np.mean(templates, 0)[None, :]  # normalize all stars
    stars = np.zeros((npix, uind.size))
    for j in uind:
        w = ind == j  # select indices of given cluster
        stars[:, j] = np.mean(templates[:, w], 1)
        print(f" {j + 1}/{uind.size}: {w.sum()}")

    if plot:
        plt.clf()
        dendrogram(data_link, truncate_mode='lastp', p=50,
                   show_leaf_counts=True, color_threshold=20)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.suptitle('Samples clustering')
        plt.pause(0.1)

    return stars, ind


##############################################################################

def speclus_example():

    f = np.load('miles_stars_v91.npz')
    wave, templates, numbers = f['wave'], f['stars'], f['numbers']
    stars, ind = speclus(templates, initialize=True, plot=True)
    np.savez_compressed(f'miles_hierarchical_clusters.npz', wave=wave, stars=stars)

##############################################################################

if __name__ == '__main__':

    speclus_example()
