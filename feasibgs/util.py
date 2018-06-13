'''

utility functions 

'''
import os
import sys
import numpy as np


def check_env(): 
    if os.environ.get('FEASIBGS_DIR') is None: 
        raise ValueError("set $FEASIBGS_DIR in bashrc file!") 
    return None


def flux2mag(flux, band='g', method='asinh'): 
    ''' given flux calculate SDSS asinh magnitude  
    '''
    if method == 'asinh': 
        if band == 'u': b = 1.4e-10
        elif band == 'g': b = 0.9e-10
        elif band == 'r': b = 1.2e-10
        elif band == 'i': b = 1.8e-10
        elif band == 'z': b = 7.4e-10
        else: raise ValueError
        
        return -2.5/np.log(10) * (np.arcsinh(1.e-9*flux/(2.*b)) + np.log(b))
    elif method == 'log': 
        return 22.5 - 2.5 * np.log10(flux) 


def dat_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.environ.get('FEASIBGS_DIR') 


def fig_dir(): 
    ''' directory for figures 
    '''
    if os.environ.get('FEASIBGS_FIGDIR') is None: 
        return dat_dir()+'/figs/'
    else: 
        return os.environ.get('FEASIBGS_FIGDIR')


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    return fig_dir().split('fig')[0]+'doc/'


def paper_dir(): 
    ''' directory for paper related stuff 
    '''
    return fig_dir().split('fig')[0]+'paper/'


def zeff_hist(prop, ztrue, zest, range=None, threshold=0.003, nbins=20, bin_min=2): 
    ''' histogram looking at dependence of redshift efficiency on `prop`. redshift 
    is considered "successful" if dz/(1+ztrue) < threshold. This is taken from 
    one of John's ipython notebooks. 
    
    parameters 
    ----------
    prop : (numpy.ndarray)
        Some property (e.g. r-band aperture flux magnitude) 

    ztrue : (numpy.ndarray)
        True (input) redshift  
    
    zest : (numpy.ndarray)
        redshift estimate (e.g. redrock output) 
    
    range : (optional, tuple) 
        range of `prop`
    
    threshold : (float) 
        threshold for determining redshift success

    nbins : (int) 
        number of bins to evaluat ethe histogram

    bin_min : (int) 
        minimum number of galaxies that need to be in a `prop` bin 
        to be included 

    '''
    if not len(prop) == len(ztrue): raise ValueError("prop, ztrue, and zbest must have the same dimensions")
    if not len(ztrue) == len(zest): raise ValueError("prop, ztrue, and zbest must have the same dimensions")
    dz = zest - ztrue # delta z 
    dz_1pz = np.abs(dz)/(1.+ztrue)
    s1 = (dz_1pz < threshold)

    h0, bins = np.histogram(var, bins=nbins, range=range)
    hv, _ = np.histogram(var, bins=bins, weights=var)
    h1, _ = np.histogram(var[s1], bins=bins)

    good = h0 > bin_min 
    hv = hv[good]
    h0 = h0[good]
    h1 = h1[good]

    vv = hv / h0 # weighted mean of var

    def _eff(k, n):
        eff = k.astype("float") / (n.astype('float') + (n==0))
        efferr = np.sqrt(eff * (1 - eff)) / np.sqrt(n.astype('float') + (n == 0))
        return eff, efferr

    e1, ee1 = _eff(h1, h0)
    return vv, e1, ee1
