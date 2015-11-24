#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils
~~~~~~~~

Provides methods for reading and processing data from tabular formatted files

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
    ENCODING (str): Default file encoding.
    DEFAULT_DATETIME (obj): Default datetime object
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

from datetime import datetime as dt

__title__ = 'tabutils'
__package_name__ = 'tabutils'
__author__ = 'Reuben Cummings'
__description__ = 'tabular data utility methods'
__email__ = 'reubano@gmail.com'
__version__ = '0.19.4'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 Reuben Cummings'


CURRENCIES = ('$', '£', '€')
ENCODING = 'utf-8'
DEFAULT_DATETIME = dt(9999, 12, 31, 0, 0, 0)
