import numpy as np 

# -- local -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat 

# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def GAMA_test():
    ''' Tests GAMA object
    '''
    gama = Cat.GAMA() 
    data = gama.Read(silent=False)

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)

    sub = fig.add_subplot(111)
    sub.scatter(data['photo']['ra'], data['photo']['dec'], c='C1', s=1, label='GAMA photo+spec overlap')
    sub.set_xlim([110., 240.])
    sub.set_ylim([-3.5, 3.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    fig.savefig(UT.fig_dir()+"GAMA_test.png", bbox_inches='tight')
    plt.close() 
    return None 


def Legacy_test():  
    ''' Test that the Legacy object is sensible
    '''
    legacy = Cat.Legacy() 
    legacy_data = legacy.Read(silent=False)
    
    # some sanity check on the data by comparing it to 
    # the GAMA footprint 
    gama = Cat.GAMA() # read in gama
    gama_data = gama.Read(silent=False)

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)

    sub = fig.add_subplot(111)
    sub.scatter(gama_data['photo']['ra'][::10], gama_data['photo']['dec'][::10], 
            c='k', s=2, label='GAMA photo+spec')
    sub.scatter(legacy_data['ra'], legacy_data['dec'], c='C1', s=1, label='Legacy Survey DR5')  
    sub.set_xlim([110., 240.])
    sub.set_ylim([-3.5, 3.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    fig.savefig(UT.fig_dir()+"Legacy_test.png", bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    Legacy_test()
