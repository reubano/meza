#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.stats
~~~~~~~~~~

Statistics functions
"""


def mean(values):
    """
    Example:
    >>> mean([1, 2, 3, 4, 4])
    2.8
    """
    non_nones = [x for x in values if x is not None]
    return sum(non_nones) / len(non_nones)
