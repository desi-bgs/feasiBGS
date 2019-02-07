'''

scripts for generating a small sample of exposures that 
reasonably span the observing conditions of BGS

'''
import os 
import h5py
import pickle 
import numpy as np 
# -- feasibgs -- 
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


def pickExposures(nsub, method='random', validate=False, silent=True): 
    ''' Pick nsub subset of exposures from `surveysim` exposure list
    from Jeremy: `bgs_survey_exposures.withsun.hdf5', which  
    supplemented the observing conditions with sun observing conditions.
    Outputs a file that contains the exposure indices, observing conditions, 
    and old and new sky brightness of the chosen subset. 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 

    :param validate: (default: False)
        If True, generate some figures to validate things 

    :param silent: (default: True) 
        if False, code will print statements to indicate progress 
    '''
    # read surveysim BGS exposures 
    fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    bgs_exps = {}
    for k in fexps.keys():
        bgs_exps[k] = fexps[k].value
    n_exps = len(bgs_exps['RA']) # number of exposures
        
    # pick a small subset
    if method == 'random': # randomly pick exposures
        if not silent: print('randomly picking exposures')
        iexp_sub = np.random.choice(np.arange(n_exps), nsub)
    elif method == 'spacefill': # pick exposures that span the observing conditions
        if not silent: print('picking exposures to span observing conditions')
        obs = np.zeros((n_exps, 4))
        #obs[:,0] = bgs_exps['AIRMASS']
        obs[:,0] = bgs_exps['MOONFRAC']
        obs[:,1] = bgs_exps['MOONALT']
        obs[:,2] = bgs_exps['MOONSEP']
        obs[:,3] = bgs_exps['SUNALT']
        histmd, edges = np.histogramdd(obs[obs[:,1] > 0.,:], 2)
        _hasexp = histmd > 0.
        has_exp = np.where(_hasexp)
        iexp_sub = []
        for i in range(np.sum(histmd > 0.)):
            in_bin = np.ones(n_exps).astype(bool)
            for i_dim in range(obs.shape[1]):
                in_bin = (in_bin & 
                        (obs[:,i_dim] > edges[i_dim][has_exp[i_dim]][i]) & 
                        (obs[:,i_dim] <= edges[i_dim][has_exp[i_dim]+1][i])) 
            iexp_sub.append(np.random.choice(np.arange(n_exps)[in_bin], 1)[0])
        iexp_sub = np.array(iexp_sub)
        nsub = len(iexp_sub) 
        if not silent: print('%i exposures in the subset' % nsub)
    
    # read in pre-computed old and new sky brightness (this takes a bit) 
    if not silent: print('reading in sky brightness') 
    fold = ''.join([UT.dat_dir(), 'KSsky_brightness.bgs_survey_exposures.withsun.p'])
    wave_old, sky_old = pickle.load(open(fold, 'rb'))
    fnew = ''.join([UT.dat_dir(), 'newKSsky_twi_brightness.bgs_survey_exposures.withsun.p'])
    wave_new, sky_new = pickle.load(open(fnew, 'rb'))
    
    # write exposure subsets out to file 
    fpick = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nsub), method, '.hdf5']), 'w')
    fpick.create_dataset('iexp', data=iexp_sub) 
    for k in ['AIRMASS', 'MOONFRAC', 'MOONALT', 'MOONSEP', 'SUNALT', 'SUNSEP', 'EXPTIME']: # write observing conditions  
        fpick.create_dataset(k.lower(), data=bgs_exps[k][iexp_sub]) 
    # save sky brightnesses
    fpick.create_dataset('wave_old', data=wave_old) 
    fpick.create_dataset('wave_new', data=wave_new) 
    fpick.create_dataset('sky_old', data=sky_old[iexp_sub,:]) 
    fpick.create_dataset('sky_new', data=sky_new[iexp_sub,:]) 
    fpick.close() 

    if validate: 
        fig = plt.figure(figsize=(21,5))
        sub = fig.add_subplot(141)
        sub.scatter(bgs_exps['MOONALT'], bgs_exps['MOONFRAC'], c='k', s=1)
        scat = sub.scatter(bgs_exps['MOONALT'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)
        sub.set_xlabel('Moon Altitude', fontsize=20)
        sub.set_xlim([-90., 90.])
        sub.set_ylabel('Moon Illumination', fontsize=20)
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(142)
        sub.scatter(bgs_exps['MOONSEP'], bgs_exps['MOONFRAC'], c='k', s=1)
        scat = sub.scatter(bgs_exps['MOONSEP'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10) 
        sub.set_xlabel('Moon Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(143)
        sub.scatter(bgs_exps['AIRMASS'], bgs_exps['MOONFRAC'], c='k', s=1)
        scat = sub.scatter(bgs_exps['AIRMASS'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)  
        sub.set_xlabel('Airmass', fontsize=20)
        sub.set_xlim([1., 2.])
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(144)
        sub.scatter(bgs_exps['SUNSEP'], bgs_exps['SUNALT'], c='k', s=1)
        scat = sub.scatter(bgs_exps['SUNSEP'][iexp_sub], bgs_exps['SUNALT'][iexp_sub], c='C1', s=10)
        sub.set_xlabel('Sun Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('Sun Altitude', fontsize=20)
        sub.set_ylim([-90., 0.])
        fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'bgs_survey_exposures.subset.', str(nsub), method, '.png']), 
                bbox_inches='tight')

        # plot some of the sky brightnesses
        fig = plt.figure(figsize=(15,20))
        bkgd = fig.add_subplot(111, frameon=False) 
        for ii, isky in enumerate(np.random.choice(iexp_sub, 4, replace=False)):
            sub = fig.add_subplot(4,1,ii+1)
            sub.plot(wave_old, sky_new[isky,:], c='C1', label='new sky')
            sub.plot(wave_old, sky_old[isky,:], c='k', label='old sky')
            sub.set_xlim([3500., 9500.]) 
            sub.set_ylim([0., 20]) 
            if ii == 0: sub.legend(loc='upper left', fontsize=20) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('wavelength [Angstrom]', fontsize=25) 
        bkgd.set_ylabel('sky brightness [$erg/s/cm^2/A/\mathrm{arcsec}^2$]', fontsize=25) 
        fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'bgs_survey_exposures.subset.', str(nsub), method, '.skybright.png']), 
                bbox_inches='tight')
    return None 


def plotExposures(nsub, method='random'): 
    ''' Plot shwoing the subset of exposures picked from `surveysim` exposure 
    list from Jeremy: `bgs_survey_exposures.withsun.hdf5', which  
    supplemented the observing conditions with sun observing conditions.
    Outputs a file that contains the exposure indices, observing conditions, 
    and old and new sky brightness of the chosen subset. 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 
    '''
    # read surveysim BGS exposures 
    fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    bgs_exps = {}
    for k in fexps.keys():
        bgs_exps[k] = fexps[k].value

    # read exposure subsets out to file 
    fpick = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nsub), method, '.hdf5']), 'r')
    iexp_sub = fpick['iexp'].value

    fig = plt.figure(figsize=(21,5))
    sub = fig.add_subplot(141)
    sub.scatter(bgs_exps['MOONALT'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['MOONALT'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['MOONALT'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(142)
    sub.scatter(bgs_exps['MOONSEP'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['MOONSEP'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10) 
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['MOONSEP'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(143)
    sub.scatter(bgs_exps['AIRMASS'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['AIRMASS'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)  
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['AIRMASS'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Airmass', fontsize=20)
    sub.set_xlim([1., 2.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(144)
    sub.scatter(bgs_exps['SUNSEP'], bgs_exps['SUNALT'], c='k', s=1)
    scat = sub.scatter(bgs_exps['SUNSEP'][iexp_sub], bgs_exps['SUNALT'][iexp_sub], c='C1', s=10)
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['SUNSEP'][iexp_sub][i], 1.02*bgs_exps['SUNALT'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Sun Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylabel('Sun Altitude', fontsize=20)
    sub.set_ylim([-90., 0.])
    fig.subplots_adjust(wspace=0.36) 
    fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'bgs_survey_exposures.subset.', str(nsub), method, '.order.png']), 
            bbox_inches='tight')
    return None 


def tableExposures(nsub, method='spacefill'): 
    ''' write out exposure information to latex table 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 
    '''
    # read exposure subsets out to file 
    fpick = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nsub), method, '.hdf5']), 'r')
    fexps = {} 
    for k in fpick.keys(): 
        fexps[k] = fpick[k].value 
    
    ftex = open(os.path.join(UT.dat_dir(), 'bgs_zsuccess', 'bgs_survey_exposures.subset.%i%s.tex' % (nsub, method)), 'w') 
    hdr = '\n'.join([ 
        r'\documentclass{article}', 
        r'\begin{document}', 
        r'\begin{table}', 
        r'\begin{center}', 
        (r'\caption{%i exposures sampled from surveysim exposures}' % nsub), 
        r'\begin{tabular}{|cccccccc|}', 
        r'\hline', 
        ' & '.join(['', '$t_\mathrm{exp}$', 'airmass', 'moon frac.', 'moon alt.', 'moon sep.', 'sun alt.', r'sun sep.\\[0.5ex]'])])
    ftex.write(hdr) 
    ftex.write(r'\hline') 
     
    for iexp in range(nsub): 
        str_iexp = (r'%i. & %.f & %.2f & %.2f & %.2f & %.2f & %.f & %.f \\' % 
                (iexp, fexps['exptime'][iexp], fexps['airmass'][iexp], 
                    fexps['moonfrac'][iexp], fexps['moonalt'][iexp], fexps['moonsep'][iexp], 
                    fexps['sunalt'][iexp], fexps['sunsep'][iexp]))
        ftex.write(str_iexp+'\n')  
    ftex.write(r'\hline') 
    end = '\n'.join([
        r'\end{tabular}', 
        r'\end{center}', 
        r'\end{table}', 
        r'\end{document}']) 
    ftex.write(end)
    ftex.close() 
    return None 



if __name__=="__main__": 
    #pickExposures(10, method='random', silent=False, validate=True)
    #pickExposures(10, method='spacefill', silent=False, validate=True)
    #plotExposures(15, method='spacefill') 
    tableExposures(15, method='spacefill')
