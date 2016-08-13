#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
import pkutils

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from os import path as p

PARENT_DIR = p.abspath(p.dirname(__file__))

sys.dont_write_bytecode = True
py2_requirements = set(pkutils.parse_requirements('py2-requirements.txt'))
py3_requirements = set(pkutils.parse_requirements('requirements.txt'))
dev_requirements = set(pkutils.parse_requirements('dev-requirements.txt'))
readme = pkutils.read('README.rst')
# changes = pkutils.read('CHANGES.rst').replace('.. :changelog:', '')
module = pkutils.parse_module(p.join(PARENT_DIR, 'meza', '__init__.py'))
license = module.__license__
version = module.__version__
project = module.__title__
description = module.__description__
user = 'reubano'

# Conditional sdist dependencies:
py2 = sys.version_info.major == 2
requirements = py2_requirements if py2 else py3_requirements

# Conditional bdist_wheel dependencies:
extras_require = py2_requirements.difference(py3_requirements)

# Setup requirements
setup_require = [r for r in dev_requirements if 'pkutils' in r]

setup(
    name=project,
    version=version,
    description=description,
    long_description=readme,
    author=module.__author__,
    author_email=module.__email__,
    url=pkutils.get_url(project, user),
    download_url=pkutils.get_dl_url(project, user, version),
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    package_data={
        'data': ['data/*'],
        'helpers': ['helpers/*'],
        'tests': ['tests/*'],
        'docs': ['docs/*'],
        'examples': ['examples/*']
    },
    install_requires=requirements,
    extras_require={'python_version<3.0': extras_require},
    setup_requires=setup_require,
    test_suite='nose.collector',
    tests_require=dev_requirements,
    license=license,
    zip_safe=False,
    keywords=[project] + description.split(' '),
    classifiers=[
        pkutils.LICENSES[license],
        pkutils.get_status(version),
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Environment :: Console',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],
    platforms=['MacOS X', 'Windows', 'Linux'],
)
