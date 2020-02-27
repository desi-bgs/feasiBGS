#!/bin/python
'''
    
notes
----
*   desisurvey.utils.freeze_iers is finicky about version of astropy. It
    requires version 2 rather than the now commonly used 4
'''
import os 
import sys 
import numpy as np 
from astropy.io import fits
# --- desihub --- 
import desimodel
import surveysim.stats
import desisurvey.plots
from desisurvey.utils import get_date
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
if 'NERSC_HOST' not in os.environ.keys():  
    mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

#import warnings, matplotlib.cbook, astropy._erfa.core
#warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
#warnings.filterwarnings('ignore', category=astropy._erfa.core.ErfaWarning)

# parent dir
_dir = os.path.dirname(os.path.realpath(__file__))
# directoy for surveysim outs 
os.environ['DESISURVEY_OUTPUT'] = os.path.join(os.environ['CSCRATCH'], 
        'desisurvey_output')


def stats_surveysim(name): 
    '''
    '''
    # read in exposures surveysim output  
    f_exp = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'exposures_%s.fits' % name)
    exposures = fits.getdata(f_exp, 'exposures') 
    tilestats = fits.getdata(f_exp, 'tiledata')

    # read in stats surveysim output  
    f_stats = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'stats_%s.fits' % name)
    stats = surveysim.stats.SurveyStatistics(restore=f_stats)

    print('Survey runs {} to {} and observes {} tiles with {} exposures.'.format(
          get_date(np.min(exposures['mjd'])),
          get_date(np.max(exposures['mjd'])), 
          np.sum(tilestats['snr2frac'] >= 1), len(exposures)))

    print('Number of nights: {}'.format(len(stats._data)))
    print(stats.summarize())

    tiles = desisurvey.tiles.get_tiles()
    
    # -- plot SNR(actual) / SNR (goal) histogram 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    sub.hist(tilestats['snr2frac'], range=(0.75, 1.25), bins=25)
    sub.axvline(np.median(tilestats['snr2frac']), c='r') 
    sub.set_xlabel('Tile SNR(actual) / SNR (goal)')
    fig.savefig(os.path.join(_dir, 'figs', '%s.snr2frac.png' % name),
            bbox_inches='tight') 
    plt.close() 

    # -- plot total exposure time of tiles histogram 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    sub.hist(tilestats['exptime'] / 60, range=(0, 60), bins=30)
    sub.axvline(np.median(tilestats['exptime'] / 60), c='r');
    sub.set_xlabel('Tile Total Exposure Time [min]')
    fig.savefig(os.path.join(_dir, 'figs', '%s.texp.png' % name),
            bbox_inches='tight') 
    plt.close() 

    # plot survey completion as a function of time 
    fig, sub = stats.plot() 
    fig.savefig(os.path.join(_dir, 'figs', '%s.completion.png' % name),
            bbox_inches='tight') 
    plt.close() 
    return None 


def run_surveysim(name, fconfig): 
    ''' run surveysim for specified configuration file 
    '''
    fconfig = os.path.join(_dir, fconfig)
    print('surveysim --name %s --config-file %s' % (name, fconfig)) 
    os.system('surveysim --name %s --config-file %s' % (name, fconfig)) 
    return None 


def surveyinit(): 
    ''' check that the ephemerides and surveyinit files are in
    `DESISURVEY_OUTPUT` directory. Otherwise generate new files (this takes a
    long time)  
    '''
    # tabulated ephemerides during 2019-25
    f_ephem = os.path.join(os.environ['DESISURVEY_OUTPUT'],
            'ephem_2019-01-01_2025-12-31.fits') 
    # estimated average weather and optimized initial hour angle (HA)
    # assignments for each tile.c
    f_init = os.path.join(os.environ['DESISURVEY_OUTPUT'],
            'surveyinit.fits') 

    if not os.path.isfile(f_ephem) or not os.path.isfile(f_init): 
        os.system('surveyinit --verbose') 
    else: 
        print('\t%s\n\t%s\nalready exist' % (f_ephem, f_init))
    return None


if __name__=="__main__": 
    name    = sys.argv[1]
    fconfig = sys.argv[2]

    # check survey init
    surveyinit()
    # run surveysim
    run_surveysim(name, fconfig) 
    # get summary statistics of surveysim run
    stats_surveysim(name)
