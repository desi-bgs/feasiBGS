'''

utility functions 

'''
import os
import sys
import numpy as np


def flux2mag(flux, band='g'): 
    ''' given flux calculate SDSS asinh magnitude  
    '''
    if band == 'u': b = 1.4e-10
    elif band == 'g': b = 0.9e-10
    elif band == 'r': b = 1.2e-10
    elif band == 'i': b = 1.8e-10
    elif band == 'z': b = 7.4e-10
    
    return -2.5/np.log(10) * (np.arcsinh(flux/(2.*b)) + np.log(b))


def code_dir(): 
    ''' Directory where all the code is located (i.e. the directory that this file is in!)
    '''
    return os.path.dirname(os.path.realpath(__file__))


def dat_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/dat/'


def fig_dir(): 
    ''' directory for figures 
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/figs/'


def paper_dir(): 
    ''' directory for paper related stuff 
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/paper/'
