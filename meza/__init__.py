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
    NULL_YEAR (int): Year to be consider null
    NULL_TIME (str): ISO format time to be consider null
    NULL_DATETIME (obj): Default datetime object
"""

from datetime import datetime as dt
from os import path as p

__version__ = "0.45.6"
__title__ = "meza"
__package_name__ = "meza"
__author__ = "Reuben Cummings"
__description__ = "A Python toolkit for processing tabular data"
__email__ = "reubano@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2015 Reuben Cummings"

CURRENCIES = ("$", "£", "€")
ENCODING = "utf-8"
NULL_YEAR = 9999
NULL_TIME = "00:00:00"
NULL_DATETIME = dt(NULL_YEAR, 12, 31, 0, 0, 0)
BOM = "\ufeff"
PARENT_DIR = p.abspath(p.dirname(p.dirname(__file__)))
DATA_DIR = p.join(PARENT_DIR, "data", "test")
