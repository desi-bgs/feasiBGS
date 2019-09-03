__all__ = ['test_Isky_newKS_twi'] 

import pytest
import numpy as np 
# --- gqp_mc --- 
from feasibgs import skymodel as Sky


def test_Isky_newKS_twi(): 
    # test non-twilight 
    airmass = 1.
    moonill = 0.7
    moonalt = 60.
    moonsep = 80.
    _, Isky0 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -30., 0.)
    _, Isky1 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -30., 20.)
    _, Isky2 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -50., 0.)
    _, Isky3 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -50., 20.)
    assert np.array_equal(Isky0, Isky1) 
    assert np.array_equal(Isky0, Isky2) 
    assert np.array_equal(Isky0, Isky3) 

    _, Isky0 = Sky.Isky_newKS_twi(1., moonill, moonalt, moonsep, -30., 0.)
    _, Isky1 = Sky.Isky_newKS_twi(1.5, moonill, moonalt, moonsep, -30., 0.)
    _, Isky2 = Sky.Isky_newKS_twi(2., moonill, moonalt, moonsep, -30., 0.)
    assert np.median(Isky0) < np.median(Isky1)
    assert np.median(Isky1) < np.median(Isky2)

    _, Isky0 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -30., 80.)
    _, Isky1 = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -10., 80.)
    assert np.median(Isky0) < np.median(Isky1)
