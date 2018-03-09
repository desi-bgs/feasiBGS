'''

utility functions 

'''
import os
import sys
import numpy as np


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
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/fig/'


def paper_dir(): 
    ''' directory for paper related stuff 
    '''
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/paper/'
