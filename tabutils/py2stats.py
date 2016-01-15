#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.py2stats
~~~~~~~~~~~~~~~~~

Backport of py3 statistics functions

Examples:
    basic usage::

        >>> from tabutils.stats import mean

        >>> mean([1, 2, 3, 4, 4])
        2.8
"""
from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)


def mean(values):
    non_nones = [x for x in values if x is not None]
    return sum(non_nones) / len(non_nones)
