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
