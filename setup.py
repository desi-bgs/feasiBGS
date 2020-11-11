#!/usr/bin/env python
from distutils.core import setup

__version__ = '0.1'

setup(name = 'feasibgs',
      version = __version__,
      description = 'package for investigating the feasibility of the DESI-BGS',
      author='ChangHoon Hahn',
      author_email='hahn.changhoon@gmail.com',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['feasibgs'],
      packages = ['feasibgs']
      )
