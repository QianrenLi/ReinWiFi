#!/usr/bin/env python3
import os
import subprocess as sp
from pathlib import Path
from setuptools import find_packages, setup

SHELL_RUN  = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)
Path('build').mkdir(exist_ok=True)
SHELL_RUN('cd build && cmake .. && make')
SHELL_RUN('cp build/libwlsctrl.so wlsctrl')

setup(
    name='wlsctrl',
    version='0.1.0',
    description='wlsctrl for wlsops_hack.',
    author='sudo_free',
    author_email='sudofree@163.com',
    packages=find_packages(),
    entry_points={
        'console_scripts':[]
    },
    package_data={
        "": ["libwlsctrl.so"]
    },
    include_package_data=True,
)
