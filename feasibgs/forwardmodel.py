'''

forward model a spectra from meta data. 
most of the functions here serve as wrappers
for `desisim` code 

References: 
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-reference-spectra.ipynb
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-redshift-efficiency.ipynb 

'''
import os
import numpy as np 
from scipy.spatial import cKDTree as KDTree
from astropy.table import vstack
# -- local -- 
from speclite import filters
from desimodel.io import load_throughput
from desisim.io import empty_metatable
from desisim.io import read_basis_templates
from desisim.templates import BGS
import desitarget.mock.quicksurvey as mockio
from desitarget.cuts import isBGS_bright, isBGS_faint

os.environ['DESI_BASIS_TEMPLATES']='/Volumes/chang_eHDD/projects/desi/spectro/templates/basis_templates/v2.3'


class BGStree(object):
    '''class to deal with KDTree from BGS basis template metadata.
    Read in meta data of BGS basis templates, construct a KDTree
    and then use the KDTree to identify closest templates. 
    '''
    def __init__(self):
        # read in the meta data of the BGS basis templates
        self.meta = read_basis_templates(objtype='BGS', onlymeta=True)
        # construct KDTree from it 
        self.tree = KDTree(self.extractMeta())

    def extractMeta(self):
        ''' Extract quantities used to construct KDTree from the basis
        template meta data: redshift (z), M_0.1r, and 0.1(g-r).
        '''
        # read in necessary meta data
        zobj = self.meta['Z'].data  # redshift 
        mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data # absolute magnitudes in SDSS ugriz bands kcorrected to z=0.1
        rmabs = mabs[:,2] # r-band absolute magnitude
        gr = mabs[:,1] - mabs[:,2] # g-r color 

        return np.vstack((zobj, rmabs, gr)).T
    
    def Query(self, matrix):
        '''Return the nearest template number based on the KD Tree.

        Parameters
        ----------
          matrix (numpy.ndarray): 
            M x N array (M=number of properties, N=number of objects) 
            in the same format as the corresponding function for each 
            object type (e.g., self.bgs).

        Returns
        -------
          - indx: index of nearest template (main item of interest) 
          - dist: distance to nearest template
        '''
        dist, indx = self.tree.query(matrix)
        return indx, dist


class BGStemplates(object):
    '''Generate spectra for BGS templates.  
    '''
    def __init__(self, wavemin=None, wavemax=None, dw=0.2):
        ''' initiate BGS template spectra. Mainly for initializing `desisim.templates.BGS`
        '''
        # default (buffered) wavelength vector
        if wavemin is None: self.wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None: self.wavemax = load_throughput('z').wavemax + 10.0
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        # initialize the templates once:
        self.bgs_templates = BGS(wave=self.wave, normfilter='decam2014-r') 
        #normfilter='sdss2010-r') # Need to generalize this!
        self.bgs_templates.normline = None # no emission lines!

    def Spectra(self, r_mag, zred, vdisp, seed=None, templateid=None, silent=True):
        ''' Given data in the output format of `feasibgs.catalog.Read`
        generate spectra given the `templateid`.
        '''
        np.random.seed(seed) # set random seed

        # meta data of 'mag', 'redshift', 'vdisp'
        input_meta = empty_metatable(nmodel=len(r_mag), objtype='BGS')
        input_meta['SEED'] = np.random.randint(2**32, size=len(r_mag)) 
        input_meta['MAG'] = r_mag # r band apparent magnitude
        input_meta['REDSHIFT'] = zred # redshift
        input_meta['VDISP'] = vdisp 
        input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, 
                                                          novdisp=False,
                                                          verbose=(not silent))
        return flux, self.wave, meta
