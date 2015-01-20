#!/usr/bin/env python
from __future__ import print_function
from distutils.core import setup
from distutils import sysconfig
from os.path import join as pjoin, split as psplit
import sys
import platform

setup(name='FEniCSopt',
      version='0.2',
      description = "FEniCS optimization package",
      author = "Petr Lukas",
      author_email='lukas@karlin.mff.cuni.cz',
      url='https://github.com/lukaspetr/FEniCSopt',
      classifiers=[
          'Development Status :: 0.2 - Unstable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.7',
          'License :: MIT',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      scripts = [pjoin("scripts", "ind_cross.py")],
      packages = ["fenicsopt", "fenicsopt.core", "fenicsopt.examples", "fenicsopt.exports"],
      package_dir = {"fenicsopt": "fenicsopt"}
     )
