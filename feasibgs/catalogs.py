'''

submodules to handle Catalogs used in the project


'''
import os 
import numpy as np
import h5py
from pydl.pydlutils.spheregroup import spherematch

# -- local --
from . import util as UT
from ChangTools.fitstables import mrdfits


class Catalog(object): 
    ''' parent objects for the GAMA and Legacy class 
    objects.
    '''
    def __init__(self): 
        pass


class GAMA(Catalog):
    '''  class to build/read in photometric and spectroscopic overlap 
    of the GAMA DR2 data. 
    '''
    def __init__(self): 
        pass 

    def Read(self, silent=True):
        ''' Read in spherematched photometric and spectroscopic 
        data from GAMA DR2 (constructed using _Build). 
        '''
        if not os.path.isfile(self._File()): # if file is not constructed
            if not silent: print('Building %s' % self._File()) 
            self._Build(silent=silent)
    
        # read in data and compile onto a dictionary
        f = h5py.File(self._File(), 'r') 
        grp_p = f['photo'] # photo data
        grp_s = f['spec'] # spec data

        if not silent: 
            print('colums in GAMA photometry') 
            print(sorted(grp_p.keys()))
            print '========================'
            print 'colums in GAMA spectroscopy'
            print(sorted(grp_s.keys()))
            print '========================'
            print('%i objects' % len(grp_p['ra'].value)) 

        data = {} 
        data['photo'] = {} 
        for key in grp_p.keys():
            data['photo'][key] = grp_p[key].value 
            
        data['spec'] = {} 
        for key in grp_s.keys(): 
            data['spec'][key] = grp_s[key].value
        return data 
    
    def _File(self): 
        ''' hdf5 file name of spherematched photometric and spectroscopic 
        data from GAMA DR2. 
        '''
        return ''.join([UT.dat_dir(), 'GAMA_photo_spec.hdf5']) # output file 

    def _Build(self, silent=True): 
        ''' Read in the photometric data and the spectroscopic data,
        spherematch them and write the intersecting data to hdf5 file. 
        '''
        # read in photometry 
        gama_p = mrdfits(UT.dat_dir()+'InputCatA.fits')
        # read in spectroscopy 
        gama_s = mrdfits(UT.dat_dir()+'SpecLines.fits')
        if not silent: 
            print('colums in GAMA photometry') 
            print(sorted(gama_p.__dict__.keys()))
            print('%i objects' % len(gama_p.ra))
            print '========================'
            print 'colums in GAMA spectroscopy'
            print(sorted(gama_s.__dict__.keys()))
            print('%i objects' % len(gama_s.ra)) 
        
        # impose some common sense cuts 
        cut_photo = ((gama_p.modelmag_u > -9999.) & (gama_p.modelmag_g > -9999.) & (gama_p.modelmag_r > -9999.) &
                (gama_p.modelmag_i > -9999.) & (gama_p.modelmag_z > -9999.))
        cut_spec = (gama_s.ha > -99.)
        
        # spherematch the catalogs
        match = spherematch(gama_p.ra[cut_photo], gama_p.dec[cut_photo], 
                gama_s.ra[cut_spec], gama_s.dec[cut_spec], 0.000277778)
        p_match = (np.arange(len(gama_p.ra))[cut_photo])[match[0]] 
        s_match = (np.arange(len(gama_s.ra))[cut_spec])[match[1]] 
        assert len(p_match) == len(s_match)
        if not silent: print('spherematch returns %i matches' % len(p_match))
    
        # write everything into a hdf5 file 
        f = h5py.File(self._File(), 'w') 
        # store photometry data in photometry group 
        grp_p = f.create_group('photo') 
        for key in gama_p.__dict__.keys():
            grp_p.create_dataset(key, data=getattr(gama_p, key)[p_match]) 

        # store spectroscopic data in spectroscopic group 
        grp_s = f.create_group('spec') 
        for key in gama_s.__dict__.keys():
            grp_s.create_dataset(key, data=getattr(gama_s, key)[s_match]) 
        f.close() 
        return None 


class GamaLegacy(Catalog): 
    ''' class to build/read in imaging data from the Legacy survey DR 5 for the
    objects in the GAMA DR2 photo+spec data (.GAMA object). The objects in the 
    catalog has GAMA photometry, GAMA spectroscopy, and Legacy-survey photometry
    '''
    def __init__(self): 
        pass 
    
    def Read(self, silent=True):
        ''' Read in objects from legacy survey DR 5 that overlap with the 
        GAMA photo+spectra objects
        '''
        if not os.path.isfile(self._File()): # if file is not constructed
            if not silent: print('Building %s' % self._File()) 
            self._Build(silent=silent)
    
        # read in data and compile onto a dictionary
        f = h5py.File(self._File(), 'r') 
        grp_gp = f['gama-photo'] 
        grp_gs = f['gama-spec']
        grp_lp = f['legacy-photo'] 

        if not silent: 
            print('colums in GAMA Photo Data:') 
            print(sorted(grp_gp.keys()))
            print('colums in GAMA Spec Data:') 
            print(sorted(grp_gs.keys()))
            print('colums in Legacy Data:') 
            print(sorted(grp_lp.keys()))
            print('========================')
            print('%i objects' % len(grp_gp['ra'].value)) 

        data = {} 
        for dk, grp in zip(['gama-photo', 'gama-spec', 'legacy-photo'], [grp_gp, grp_gs, grp_lp]):
            data[dk] = {} 
            for key in grp.keys():
                data[key] = grp[key].value 

        return data 

    def _File(self): 
        return ''.join([UT.dat_dir(), 'gama_legacy.hdf5'])

    def _Build(self, silent=True): 
        ''' Get sweep photometry data for objects in GAMA DR2 photo+spec
        '''
        # read in the names of the sweep files 
        sweep_files = np.loadtxt(''.join([UT.dat_dir(), 'legacy/sweep/sweep_list.dat']), 
                unpack=True, usecols=[0], dtype='S') 
        # read in GAMA objects
        gama = GAMA() 
        gama_data = gama.Read(silent=silent)
    
        sweep_dict = {} 
        gama_photo_dict, gama_spec_dict = {}, {}
        # loop through the files and only keep ones that spherematch with GAMA objects
        for i_f, f in enumerate(sweep_files): 
            # read in sweep object 
            sweep = mrdfits(''.join([UT.dat_dir(), 'legacy/sweep/', f]))  
        
            # spherematch the sweep objects with GAMA objects 
            # (sweep goes first because it's usually bigger) 
            match = spherematch(sweep.ra, sweep.dec, 
                    gama_data['photo']['ra'], gama_data['photo']['dec'], 0.000277778)

            if not silent: 
                print('%i matches from the %s sweep file' % (len(match[0]), f))
            
            # save sweep photometry to `sweep_dict`
            for key in sweep.__dict__.keys(): 
                if i_f == 0: 
                    sweep_dict[key] = getattr(sweep, key)[match[0]] 
                else: 
                    sweep_dict[key] = np.concatenate([sweep_dict[key], 
                        getattr(sweep, key)[match[0]]]) 

            # save matching GAMA photometry 
            for key in gama_data['photo'].keys():  
                if i_f == 0: 
                    gama_photo_dict[key] = gama_data['photo'][key][match[1]]
                else: 
                    gama_photo_dict[key] = \
                            np.concatenate([gama_photo_dict[key], gama_data['photo'][key][match[1]]])
            
            # save matching GAMA spectroscopy 
            for key in gama_data['spec'].keys():  
                if i_f == 0: 
                    gama_spec_dict[key] = gama_data['spec'][key][match[1]]
                else: 
                    gama_spec_dict[key] = \
                            np.concatenate([gama_spec_dict[key], gama_data['spec'][key][match[1]]])

            if not silent and (i_f == 0): print(sweep_dict.keys())
            del sweep  # free memory? (apparently not really) 

        if not silent: 
            print('========================')
            print('%i objects out of %i GAMA objects mached' % (len(sweep_dict['ra']), len(gama_data['photo']['dec'])) )

        assert len(sweep_dict['ra']) == len(gama_photo_dict['ra']) 
        assert len(sweep_dict['ra']) == len(gama_spec_dict['ra']) 

        # save data to hdf5 file
        if not silent: print('writing to %s' % self._File())
        f = h5py.File(self._File(), 'w') 
        grp_gp = f.create_group('gama-photo') 
        grp_gs = f.create_group('gama-spec') 
        grp_lp = f.create_group('legacy-photo') 

        for key in sweep_dict.keys():
            grp_lp.create_dataset(key, data=sweep_dict[key]) 
        for key in gama_photo_dict.keys(): 
            grp_gp.create_dataset(key, data=gama_photo_dict[key]) 
        for key in gama_spec_dict.keys(): 
            grp_gs.create_dataset(key, data=gama_spec_dict[key]) 
        f.close() 
        return None 
