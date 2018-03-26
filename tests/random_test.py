'''

random tests

'''
import numpy as np 

# -- local -- 
from feasibgs import util as UT

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



def asinhMag(): 
    ''' comparison of SDSS asinh magnitude and standard 
    logarithmic magnitude 
    '''
    flux = np.linspace(-100., 100, 200) 
    bands = np.array(['u', 'g', 'r', 'i', 'z'])
    fig = plt.figure(figsize=(4*len(bands),4))
    for i_col, col in enumerate(bands): 
        sub = fig.add_subplot(1,len(bands),i_col+1)
        # Pogson magnitude
        pog_mag = 22.5 - 2.5 * np.log10(flux) 
        # asinh magnitude
        asinh_mag = 22.5 + UT.flux2mag(flux, band=col) 
        sub.scatter(pog_mag, asinh_mag)  
        sub.plot([0., 50.], [0., 50.], c='k', ls='--') 
        sub.set_xlabel('$'+col+'$ Pogson magnitude', fontsize=20)
        sub.set_xlim([16., 24.])
        sub.set_ylabel('$'+col+'$ arcsinh magnitude', fontsize=20)
        sub.set_ylim([16., 24.])

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.fig_dir()+"asinhMag.png", bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__":
    asinhMag() 

