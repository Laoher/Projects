# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:49:29 2018

@author: Upskills
"""

from cx_Freeze import setup, Executable
import os.path
import os
import matplotlib
from scipy.sparse.csgraph._shortest_path import shortest_path, floyd_warshall, dijkstra
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')
additional_mods = ['numpy.core._methods', 'numpy.lib.format','scipy.sparse.csgraph._shortest_path','mkl','numpy.core']
setup(name = "classifier" ,
      version = "0.1" ,
      description = "" ,
      options = { 'build.exe' : {"packages" : ["matplotlib","numpy","mkl","nltk","tensorflow","pandas","keras","json","time"],
            'includes':additional_mods,
        'include_files' : [
                os.path.join(PYTHON_INSTALL_DIR, 'DLLs','tk86t.dll'),
                os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
                ],
                },
        },
      executables = [Executable("classifier.py")])