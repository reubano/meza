#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.convert
~~~~~~~~~~~~~~~~

Provides methods for converting data structures

Examples:
    literal blocks::

        from tabutils.process import underscorify

        header = ['ALL CAPS', 'Illegal $%^', 'Lots of space']
        names = underscorify(header)

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import itertools as it

from os import path as p
from decimal import Decimal, InvalidOperation, ROUND_UP, ROUND_DOWN

from . import fntools as ft, CURRENCIES

from dateutil.parser import parse


def to_decimal(value, **kwargs):
    """Parses and formats currency values into decimals
    >>> to_decimal('$123.45')
    Decimal('123.45')
    >>> to_decimal('123€')
    Decimal('123.00')
    >>> to_decimal('2,123.45')
    Decimal('2123.45')
    >>> to_decimal('2.123,45', thousand_sep='.', decimal_sep=',')
    Decimal('2123.45')
    >>> to_decimal('spam')
    """
    thousand_sep = kwargs.get('thousand_sep', ',')
    decimal_sep = kwargs.get('decimal_sep', '.')
    places = kwargs.get('places', 2)
    roundup = kwargs.get('roundup', True)

    rounding = ROUND_UP if roundup else ROUND_DOWN
    precision = '.%s1' % ''.join(it.repeat('0', places - 1))

    currencies = it.izip(CURRENCIES, it.repeat(''))
    seperators = [(thousand_sep, ''), (decimal_sep, '.')]

    try:
        stripped = ft.mreplace(value, it.chain(currencies, seperators))
    except AttributeError:
        # We don't have a string
        stripped = value

    try:
        decimalized = Decimal(stripped)
    except InvalidOperation:
        quantized = None
    else:
        quantized = decimalized.quantize(Decimal(precision), rounding=rounding)

    return quantized


def ctype2ext(content_type=None):
    try:
        ctype = content_type.split('/')[1].split(';')[0]
    except (AttributeError, IndexError):
        ctype = None

    xlsx_type = 'vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    switch = {'xls': 'xls', 'csv': 'csv'}
    switch[xlsx_type] = 'xlsx'

    if ctype not in switch:
        print(
            'Content-Type %s not found in dictionary. Using default value.'
            % ctype)

    return switch.get(ctype, 'csv')


def to_float(value):
    """Parses and formats numbers into floats.

    Args:
        value (str): The number to parse.

    Returns:
        flt: The parsed number.

    Examples:
        >>> to_float('1')
        1.0
        >>> to_float('1f')
    """
    if value and value.strip():
        try:
            value = float(value.replace(',', ''))
        except ValueError:
            value = None
    else:
        value = None

    return value


def _to_date(value, date_format=None):
    """Parses and formats date strings.

    Args:
        value (str): The date to parse.
        date_format (str): Date format passed to `strftime()`.

    Returns:
        [tuple(str, bool)]: Tuple of the formatted date string and retry value.

    Examples:
        >>> _to_date('5/4/82')
        (datetime.datetime(1982, 5, 4, 0, 0), False)
        >>> _to_date('5/4/82', '%Y-%m-%d')
        ('1982-05-04', False)
        >>> _to_date('2/32/82', '%Y-%m-%d')
        (u'2/32/82', True)
    """
    try:
        if value and value.strip() and date_format:
            value = parse(value).strftime(date_format)
        elif value and value.strip():
            value = parse(value)

        retry = False
    # impossible date, e.g., 2/31/15
    except ValueError:
        retry = True
    # unparseable date, e.g., Novmbr 4
    except TypeError:
        value = None
        retry = False

    return (value, retry)


def to_date(value, date_format=None):
    """Parses and formats date strings.

    Args:
        value (str): The date to parse.
        date_format (str): Date format passed to `strftime()`.

    Returns:
        str: The formatted date string.

    Examples:
        >>> to_date('5/4/82')
        datetime.datetime(1982, 5, 4, 0, 0)
        >>> to_date('5/4/82', '%Y-%m-%d')
        '1982-05-04'
        >>> to_date('2/32/82', '%Y-%m-%d')
        '1982-02-28'
    """
    value, retry = _to_date(value, date_format)

    # Fix impossible dates, e.g., 2/31/15
    if retry:
        bad_num = [x for x in ['29', '30', '31', '32'] if x in value][0]
        possibilities = [value.replace(bad_num, x) for x in ['30', '29', '28']]

        for possible in possibilities:
            value, retry = _to_date(possible, date_format)

            if retry:
                continue
            else:
                break

    return value


def to_filepath(filepath, **kwargs):
    """Creates a filepath from an online resource, i.e., linked file or
    google sheets export.

    Args:
        filepath (str): Output file path or directory.
        **kwargs: Keyword arguments.

    Kwargs:
        headers (dict): HTTP response headers, e.g., `r.headers`.
        name_from_id (bool): Overwrite filename with resource id.
        resource_id (str): The resource id (required if `name_from_id` is True
            or filepath is a google sheets export)

    Returns:
        str: filepath

    Examples:
        >>> to_filepath('file.csv')
        u'file.csv'
        >>> to_filepath('.', resource_id='rid')
        Content-Type None not found in dictionary. Using default value.
        u'./rid.csv'
    """
    isdir = p.isdir(filepath)
    headers = kwargs.get('headers') or {}
    name_from_id = kwargs.get('name_from_id')
    resource_id = kwargs.get('resource_id')

    if isdir and not name_from_id:
        try:
            disposition = headers.get('content-disposition', '')
            filename = disposition.split('=')[1].split('"')[1]
        except (KeyError, IndexError):
            filename = resource_id
    elif isdir or name_from_id:
        filename = resource_id

    if isdir and filename.startswith('export?format='):
        filename = '%s.%s' % (resource_id, filename.split('=')[1])
    elif isdir and '.' not in filename:
        ctype = headers.get('content-type')
        filename = '%s.%s' % (filename, ctype2ext(ctype))

    return p.join(filepath, filename) if isdir else filepath
