#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza
~~~~

Provides methods for reading and processing data from tabular formatted files

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
    ENCODING (str): Default file encoding.
    DEFAULT_DATETIME (obj): Default datetime object
"""

from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import sys
import logging

from datetime import datetime as dt
from builtins import *

__version__ = '0.29.4'
__title__ = 'meza'
__package_name__ = 'meza'
__author__ = 'Reuben Cummings'
__description__ = 'A Python toolkit for processing tabular data'
__email__ = 'reubano@gmail.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 Reuben Cummings'

CURRENCIES = ('$', '£', '€')
ENCODING = 'utf-8'
DEFAULT_DATETIME = dt(9999, 12, 31, 0, 0, 0)

if sys.version_info.major >= 3:
    import csv
else:
    from . import unicsv as csv

logging.basicConfig()
csv = csv
