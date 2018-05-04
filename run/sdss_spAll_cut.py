'''

cut down the size of the spAll and spAllline files 
and write to hdf5. Due to the massive size of the files
this is meant to be run on NERSC 

'''
import h5py 
import numpy as np 
from astropy.io import fits
# --- local --- 
from feasibgs import util as UT 


def SpAllzcut(): 
    ''' impose a redshift cut on spAll and spAllLine data files and 
    write it to hdf5
    '''
    dir_spall = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/v5_7_0/'
    f_spall = fits.open(''.join([dir_spall, 'spAll-v5_7_0.fits'])) # read in spAll 
    spall = f_spall[1].data
    
    ngal = len(spall['Z']) # number of galaxies 
    zlim = (spall['Z'] < 0.4)  # redshift limit
    
    # write redshift cut spAll to hdf5 
    f_spall_hdf5 = h5py.File(''.join([UT.dat_dir(), 'spAll-v5_7_0.zcut.hdf5']), 'w') 
    for name in spall.names: 
        f_spall_hdf5.create_dataset(name.lower(), data=spall.field(name)[zlim])
    f_spall_hdf5.close() 
    
    f_spallline = fits.open(''.join([dir_spall, 'spAllLine-v5_7_0.fits'])) # read in spAll Line data  
    spallline = f_spallline[1].data 
    ncol = spallline['LINEZ'].shape[0]/ngal
    
    # write restructured and redshift cut spAllline to hdf5 
    f_spallline_hdf5 = h5py.File(''.join([UT.dat_dir(), 'spAllLine-v5_7_0.zcut.hdf5']), 'w') 
    for name in spallline.names: 
        col_zcut = spallline.field(name).reshape(ngal, ncol)[zlim,:]
        f_spallline_hdf5.create_dataset(name.lower(), data=col_zcut)
    f_spallline_hdf5.close() 
    return None 


if __name__=="__main__": 
    SpAllzcut()
