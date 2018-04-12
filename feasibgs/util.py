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
