#!/usr/bin/env python
from distutils.core import setup
#from setuptools import setup

#setup(name="feasibgs",
#        description="package for investigating the feasibility of the DESI-BGS", 
#        packages=["feasibgs"])

__version__ = '0.1'

setup(name = 'feasiBGS',
      version = __version__,
      description = 'package for investigating the feasibility of the DESI-BGS',
      author='ChangHoon Hahn',
      author_email='changhoonhahn@lbl.gov',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['feasiBGS'],
      packages = ['feasibgs'],
      scripts=['feasibgs/catalogs.py', 'feasibgs/forwardmodel.py', 'feasibgs/util.py']
      )
