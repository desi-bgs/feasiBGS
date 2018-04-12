'''

submodules to handle Catalogs used in the project


'''
import os 
import numpy as np
import h5py
from astropy.io import fits 
from astropy.cosmology import FlatLambdaCDM
from pydl.pydlutils.spheregroup import spherematch

# -- local --
from . import util as UT
from ChangTools.fitstables import mrdfits
from ChangTools.fitstables import FitsTable


class Catalog(object): 
    ''' parent object for the objects in this module. Currently
    has no functionality
    '''
    def __init__(self): 
        pass


class GAMA(Catalog):
    '''  class to build/read in photometric and spectroscopic overlap 
    of the GAMA DR2 data. The GAMA DR2 data contains photometry and
    spectroscopy from GAMA I, which covers three regions of 48 deg^2 
    area for a total of 144 deg^2. 
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
        grp_k0 = f['kcorr_z0.0']
        grp_k1 = f['kcorr_z0.1']

        if not silent: 
            print('colums in GAMA photometry') 
            print(sorted(grp_p.keys()))
            print '========================'
            print 'colums in GAMA spectroscopy'
            print(sorted(grp_s.keys()))
            print '========================'
            print 'colums in GAMA kcorrects'
            print(sorted(grp_k0.keys()))
            print '========================'
            print('%i objects' % len(grp_p['ra'].value)) 
            print '========================'

        data = {} 
        for dkey, grp in zip(['photo', 'spec', 'kcorr_z0.0', 'kcorr_z0.1'], [grp_p, grp_s, grp_k0, grp_k1]): 
            data[dkey] = {} 
            for key in grp.keys():
                data[dkey][key] = grp[key].value 
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
        # read in photometry (GAMA`s master input catalogue; http://www.gama-survey.org/dr2/schema/table.php?id=156)
        gama_p = mrdfits(UT.dat_dir()+'gama/InputCatA.fits')
        # read in spectroscopy (http://www.gama-survey.org/dr2/schema/table.php?id=197)
        gama_s = mrdfits(UT.dat_dir()+'gama/SpecLines.fits')
        # read in kcorrect z = 0.0 (http://www.gama-survey.org/dr2/schema/table.php?id=177)
        gama_k0 = self._readKcorrect(UT.dat_dir()+'gama/kcorr_z00.fits')
        # read in kcorrect z = 0.1 (http://www.gama-survey.org/dr2/schema/table.php?id=178)
        gama_k1 = self._readKcorrect(UT.dat_dir()+'gama/kcorr_z01.fits')
        if not silent: 
            print('colums in GAMA photometry') 
            print(sorted(gama_p.__dict__.keys()))
            print('%i objects' % len(gama_p.ra))
            print '========================'
            print 'colums in GAMA spectroscopy'
            print(sorted(gama_s.__dict__.keys()))
            print('%i objects' % len(gama_s.ra)) 
            print '========================'
            print 'colums in GAMA k-correct'
            print(sorted(gama_k0.__dict__.keys()))
            print('%i objects' % len(gama_k0.mass)) 
            print '========================'
        
        # impose some common sense cuts 
        # only keep gama_p that has gama_k0 matches and SDSS photometry
        assert np.array_equal(gama_p.cataid, np.arange(len(gama_p.cataid)))
        assert np.array_equal(gama_k0.cataid, gama_k1.cataid) 
        has_kcorr = np.zeros(len(gama_p.cataid), dtype=bool)
        has_kcorr[gama_k0.cataid] = True
        cut_photo = ((gama_p.modelmag_u > -9999.) & (gama_p.modelmag_g > -9999.) & (gama_p.modelmag_r > -9999.) &
                (gama_p.modelmag_i > -9999.) & (gama_p.modelmag_z > -9999.) & has_kcorr)
        # only keep gama_s that has Halpha  
        cut_spec = (gama_s.ha > -99.)
        
        # spherematch the catalogs
        match = spherematch(gama_p.ra[cut_photo], gama_p.dec[cut_photo], 
                gama_s.ra[cut_spec], gama_s.dec[cut_spec], 0.000277778)
        p_match = (np.arange(len(gama_p.ra))[cut_photo])[match[0]] 
        s_match = (np.arange(len(gama_s.ra))[cut_spec])[match[1]] 
        assert len(p_match) == len(s_match)
        if not silent: print('spherematch returns %i matches' % len(p_match))
        
        # check that gama_p.cataid[s_match] is subset of gama_k0.cataid
        assert np.all(np.in1d(gama_p.cataid[p_match], gama_k0.cataid)) 
         # get ordering for kcorrect data
        k_match = np.searchsorted(gama_k0.cataid, gama_p.cataid[p_match])

        assert np.array_equal(gama_p.cataid[p_match], gama_k0.cataid[k_match])
    
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

        # store kcorrect data in kcorrect groups
        grp_k0 = f.create_group('kcorr_z0.0') 
        for key in gama_k0.__dict__.keys():
            grp_k0.create_dataset(key, data=getattr(gama_k0, key)[k_match]) 

        grp_k1 = f.create_group('kcorr_z0.1') 
        for key in gama_k1.__dict__.keys():
            grp_k1.create_dataset(key, data=getattr(gama_k1, key)[k_match]) 

        f.close() 
        return None 

    def _readKcorrect(self, fitsfile): 
        ''' GAMA Kcorrect raises VerifyError if read in the usual fashion.
        '''
        kcorr = FitsTable() # same class as mrdfits output for consistency

        f = fits.open(fitsfile)
        f.verify('fix') 
        fitsdata = f[1].data

        for name in fitsdata.names: 
            setattr(kcorr, name.lower(), fitsdata.field(name))
        return kcorr


class GamaLegacy(Catalog): 
    ''' class to build/read in imaging data from the Legacy survey DR 5 for the
    objects in the GAMA DR2 photo+spec data (.GAMA object). The objects in the 
    catalog has GAMA photometry, GAMA spectroscopy, and Legacy-survey photometry
    '''
    def __init__(self): 
        pass 

    def AbsMag(self, data, kcorr=0.1, H0=70, Om0=0.3, galext=False):  
        ''' Calculate absolute magnitude in SDSS u, g, r, i, z bands with kcorrect 
        at z=`kcorr` given the data dictionary from the `GamaLegacy.Read` method. 
        H0 and Om0 specifies the cosmology for the distance modulus. 
        '''
        # check data's structure 
        for k in ['gama-photo', 'gama-spec','gama-kcorr-z0.0', 'gama-kcorr-z0.1']: 
            if k not in data.keys(): 
                raise ValueError('input data does not have the approprite keys') 
        # check kcorr 
        if kcorr not in [0.0, 0.1]: raise ValueError('kcorr = 0.0, 0.1 only') 
    
        bands_sdss = ['u','g','r','i','z']
        # apparent magnitude from GAMA photometry
        if not galext: 
            mag_ugriz = np.array([data['gama-photo']['modelmag_'+b] for b in bands_sdss]) 
        else: 
            mag_ugriz =np.array([data['gama-kcorr-z0.1'][b+'_model'] for b in bands_sdss]) 

        redshift = data['gama-spec']['z_helio']  # redshift
        # distance modulus 
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        D_L = cosmo.luminosity_distance(redshift).value # Mpc 
        DM = 5. * np.log10(1e5*D_L)
        # k-correct 
        if kcorr == 0.0: 
            kcorr = np.array([data['gama-kcorr-z0.0']['kcorr_'+b] for b in bands_sdss]) 
        elif kcorr == 0.1: 
            kcorr = np.array([data['gama-kcorr-z0.1']['kcorr_'+b] for b in bands_sdss])
        
        absmag_ugriz = mag_ugriz - DM - kcorr
        return absmag_ugriz
    
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
        grp_k0 = f['gama-kcorr-z0.0']
        grp_k1 = f['gama-kcorr-z0.1']
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
        for dk, grp in zip(['gama-photo', 'gama-spec', 'gama-kcorr-z0.0', 'gama-kcorr-z0.1', 'legacy-photo'], 
                [grp_gp, grp_gs, grp_k0, grp_k1, grp_lp]):
            data[dk] = {} 
            for key in grp.keys():
                data[dk][key] = grp[key].value 
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
        gama_photo_dict, gama_spec_dict, gama_kcorr0_dict, gama_kcorr1_dict = {}, {}, {}, {} 
        # loop through the files and only keep ones that spherematch with GAMA objects
        for i_f, f in enumerate(sweep_files): 
            # read in sweep object 
            sweep = mrdfits(''.join([UT.dat_dir(), 'legacy/sweep/', f]))  
        
            # spherematch the sweep objects with GAMA objects 
            if len(sweep.ra) > len(gama_data['photo']['ra']):
                match = spherematch(sweep.ra, sweep.dec, 
                        gama_data['photo']['ra'], gama_data['photo']['dec'], 0.000277778)
            else: 
                match_inv = spherematch(gama_data['photo']['ra'], gama_data['photo']['dec'], 
                        sweep.ra, sweep.dec, 0.000277778)
                match = [match_inv[1], match_inv[0], match_inv[2]] 

            if not silent: 
                print('%i matches from the %s sweep file' % (len(match[0]), f))
            
            # save sweep photometry to `sweep_dict`
            for key in sweep.__dict__.keys(): 
                if i_f == 0: 
                    sweep_dict[key] = getattr(sweep, key)[match[0]] 
                else: 
                    sweep_dict[key] = np.concatenate([sweep_dict[key], 
                        getattr(sweep, key)[match[0]]]) 

            # save matching GAMA data ('photo', 'spec', and kcorrects) 
            for gkey, gdict in zip(['photo', 'spec', 'kcorr_z0.0', 'kcorr_z0.1'],
                    [gama_photo_dict, gama_spec_dict, gama_kcorr0_dict, gama_kcorr1_dict]): 
                for key in gama_data[gkey].keys(): 
                    if i_f == 0: 
                        gdict[key] = gama_data[gkey][key][match[1]]
                    else: 
                        gdict[key] = np.concatenate([gdict[key], gama_data[gkey][key][match[1]]])

            if not silent and (i_f == 0): print(sweep_dict.keys())
            del sweep  # free memory? (apparently not really) 

        if not silent: 
            print('========================')
            print('%i objects out of %i GAMA objects mached' % (len(sweep_dict['ra']), len(gama_data['photo']['dec'])) )

        assert len(sweep_dict['ra']) == len(gama_photo_dict['ra']) 
        assert len(sweep_dict['ra']) == len(gama_spec_dict['ra']) 
        assert len(sweep_dict['ra']) == len(gama_kcorr0_dict['mass']) 
        assert len(sweep_dict['ra']) == len(gama_kcorr1_dict['mass']) 

        # save data to hdf5 file
        if not silent: print('writing to %s' % self._File())
        f = h5py.File(self._File(), 'w') 
        grp_gp = f.create_group('gama-photo') 
        grp_gs = f.create_group('gama-spec') 
        grp_k0 = f.create_group('gama-kcorr-z0.0') 
        grp_k1 = f.create_group('gama-kcorr-z0.1') 
        grp_lp = f.create_group('legacy-photo') 
    
        for key in sweep_dict.keys():
            grp_lp.create_dataset(key, data=sweep_dict[key]) 
        for key in gama_photo_dict.keys(): 
            grp_gp.create_dataset(key, data=gama_photo_dict[key]) 
        for key in gama_spec_dict.keys(): 
            grp_gs.create_dataset(key, data=gama_spec_dict[key]) 
        for key in gama_kcorr0_dict.keys(): 
            grp_k0.create_dataset(key, data=gama_kcorr0_dict[key]) 
        for key in gama_kcorr1_dict.keys(): 
            grp_k1.create_dataset(key, data=gama_kcorr1_dict[key]) 
        f.close() 
        return None 

    def _getTractorApflux(self, brickname, objids, dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/'): 
        ''' The catalog is constructed from the sweep catalog and the 
        GAMA DR2 photo+spec data. The sweep catalog does not include 
        all the photometric data from the legacy survey. This methods 
        appends 'apflux_g', 'apflux_r', 'apflux_z' and relevant columsn 
        from the tractor files. 
        
        This can (and probably should) be extended to other columns 
        '''
        bricks_uniq = np.unique(brickname)  # unique bricks
        AAAs = np.array([brick[:3] for brick in bricks_uniq]) 
        
        # apfluxes in 'g', 'r', and 'z' bands 
        bands = ['g', 'r', 'z']
        apfluxes = np.zeros((3, len(brickname), 8)) 
        apflux_ivars = np.zeros((3, len(brickname), 8)) 
        apflux_resids = np.zeros((3, len(brickname), 8)) 
    
        n_brick = 0 
        for AAA, brick in zip(AAAs, bricks_uniq): 
            name = ''.join([dir, AAA, '/tractor-', brick, '.fits'])
            if not os.path.isfile(name): raise ValueError('%s tractor file not available' % name)
            f_tractor = fits.open(name) 
            tractor = f_tractor[1].data

            inbrick = (brickname == brick) 
            for i_k, key in enumerate(bands):
                apfluxes[i_k, inbrick, :] = tractor.field('apflux_'+key)[objids[inbrick]]
                apflux_ivars[i_k, inbrick, :] = tractor.field('apflux_ivar_'+key)[objids[inbrick]]
                apflux_resids[i_k, inbrick, :] = tractor.field('apflux_resid_'+key)[objids[inbrick]]

            n_brick += np.sum(inbrick)

        assert n_brick == len(brickname) 

        # return dictionary with appropriate keys 
        apflux_dict = {} 
        for i_k, key in enumerate(bands):
            apflux_dict['apflux_'+key] = apfluxes[i_k,:,:]
            apflux_dict['apflux_ivar_'+key] = apflux_ivars[i_k,:,:]
            apflux_dict['apflux_resid_'+key] = apflux_resids[i_k,:,:]

        return apflux_dict


def _GamaLegacy_TractorAPFLUX(): 
    ''' Retroactively add apflux columns from the tractor catalogs 
    to the GamaLegacy catalog constructed and saved to file. 
    '''
    gleg = GamaLegacy() 

    # open saved gama-legacy catalog for appending
    f_gleg = h5py.File(gleg._File(), 'r+') 
    # legacy photometry group 
    grp_lp = f_gleg['legacy-photo'] 

    if 'apflux_g' in grp_lp.keys(): 
        # check that the columsn dont' already exist 
        f_gleg.close() 
        raise ValueError('apfluxes already in the catalog') 

    # read apfluxes from tractor catalogs 
    apflux_dict = gleg._getTractorApflux(grp_lp['brickname'], grp_lp['objid'], 
            dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/') 
    assert apflux_dict['apflux_g'].shape[0] == len(grp_lp['brickname']) 
    
    # save fluxes to the dataset 
    for key in apflux_dict.keys(): 
        grp_lp.create_dataset(key, data=apflux_dict[key]) 

    f_gleg.close()  
    return None 


