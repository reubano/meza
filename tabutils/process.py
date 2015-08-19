#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.process
~~~~~~~~~~~~~~~~

Provides methods for processing data from tabular formatted files

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

from decimal import Decimal, InvalidOperation
from dateutil.parser import parse
from functools import partial
from slugify import slugify

CURRENCIES = ('$', '£', '€')

underscorify = lambda fields: [slugify(f, separator='_') for f in fields]


def make_float(value):
    """Parses and formats numbers into floats.

    Args:
        value (str): The number to parse.

    Returns:
        flt: The parsed number.

    Examples:
        >>> make_float('1')
        1.0
        >>> make_float('1f')
    """
    if value and value.strip():
        try:
            value = float(value.replace(',', ''))
        except ValueError:
            value = None
    else:
        value = None

    return value


def decimalize(string, thousand_sep=',', decimal_sep='.', precision=2):
    """Parses and formats currency values into decimals
    >>> decimalize('$123.45')
    Decimal('123.45')
    >>> decimalize('123€')
    Decimal('123')
    >>> decimalize('2,123.45')
    Decimal('2123.45')
    >>> decimalize('2.123,45', '.', ',')
    Decimal('2123.45')
    >>> decimalize('spam')
    """
    currencies = it.izip(CURRENCIES, it.repeat(''))
    seperators = [(thousand_sep, ''), (decimal_sep, '.')]
    stripped = mreplace(string, it.chain(currencies, seperators))
    try:
        value = Decimal(stripped)
    except InvalidOperation:
        value = None

    return value


def is_numeric_like(string, seperators=('.', ',')):
    """
    >>> is_numeric_like('$123.45')
    True
    >>> is_numeric_like('123€')
    True
    >>> is_numeric_like('2,123.45')
    True
    >>> is_numeric_like('2.123,45')
    True
    >>> is_numeric_like('10e5')
    True
    >>> is_numeric_like('spam')
    False
    """
    replacements = it.izip(it.chain(CURRENCIES, seperators), it.repeat(''))
    stripped = mreplace(string, replacements)

    try:
        float(stripped)
    except (ValueError, TypeError):
        return False
    else:
        return True


def afterish(string, char=',', exclude=None):
    """Number of digits after a given character.

    >>> afterish('123.45', '.')
    2
    >>> afterish('1001.', '.')
    0
    >>> afterish('1001', '.')
    -1
    >>> afterish('1,001')
    3
    >>> afterish('2,100,001.00')
    6
    >>> afterish('2,100,001.00', exclude='.')
    3
    >>> afterish('1,000.00', '.', ',')
    2
    >>> afterish('eggs', '.')
    Traceback (most recent call last):
    TypeError: Not able to convert eggs to a number
    """
    numeric_like = is_numeric_like(string)

    if numeric_like and char in string:
        excluded = [s for s in string.split(exclude) if char in s][0]
        after = len(excluded) - excluded.rfind(char) - 1
    elif numeric_like:
        after = -1
    else:
        raise TypeError('Not able to convert %s to a number' % string)

    return after


def mreplace(string, replacements):
    func = lambda x, y: x.replace(y[0], y[1])
    return reduce(func, replacements, string)


def xmlize(content):
    """ Recursively makes elements of an array xml compliant

    Args:
        content (List[str]): the content to clean

    Yields:
        (str): the cleaned element

    Examples:
        >>> list(xmlize(['&', '<']))
        [u'&amp', u'&lt']
    """
    replacements = [
        ('&', '&amp'), ('>', '&gt'), ('<', '&lt'), ('\n', ' '), ('\r\n', ' ')]

    for item in content:
        if hasattr(item, 'upper'):
            yield mreplace(item, replacements)
        else:
            try:
                yield list(xmlize(item))
            except TypeError:
                yield mreplace(item, replacements)


def _make_date(value, date_format):
    """Parses and formats date strings.

    Args:
        value (str): The date to parse.
        date_format (str): Date format passed to `strftime()`.

    Returns:
        [tuple(str, bool)]: Tuple of the formatted date string and retry value.

    Examples:
        >>> _make_date('5/4/82', '%Y-%m-%d')
        ('1982-05-04', False)
        >>> _make_date('2/32/82', '%Y-%m-%d')
        (u'2/32/82', True)
    """
    try:
        if value and value.strip():
            value = parse(value).strftime(date_format)

        retry = False
    # impossible date, e.g., 2/31/15
    except ValueError:
        retry = True
    # unparseable date, e.g., Novmbr 4
    except TypeError:
        value = None
        retry = False

    return (value, retry)


def make_date(value, date_format):
    """Parses and formats date strings.

    Args:
        value (str): The date to parse.
        date_format (str): Date format passed to `strftime()`.

    Returns:
        str: The formatted date string.

    Examples:
        >>> make_date('5/4/82', '%Y-%m-%d')
        '1982-05-04'
        >>> make_date('2/32/82', '%Y-%m-%d')
        '1982-02-28'
    """
    value, retry = _make_date(value, date_format)

    # Fix impossible dates, e.g., 2/31/15
    if retry:
        bad_num = [x for x in ['29', '30', '31', '32'] if x in value][0]
        possibilities = [value.replace(bad_num, x) for x in ['30', '29', '28']]

        for p in possibilities:
            value, retry = _make_date(p, date_format)

            if retry:
                continue
            else:
                break

    return value


def gen_type_cast(records, fields, date_format='%Y-%m-%d'):
    """Casts record entries based on field types.

    Args:
        records (List[dicts]): Record entries (`read_csv` output).
        fields (List[dicts]): Field types (`gen_fields` output).
        date_format (str): Date format passed to `strftime()` (default:
            '%Y-%m-%d', i.e, 'YYYY-MM-DD').

    Yields:
        dict: The type casted record entry.

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> csv_filepath = p.join(parent_dir, 'testdata', 'test.csv')
        >>> csv_records = io.read_csv(csv_filepath, sanitize=True)
        >>> csv_header = sorted(csv_records.next().keys())
        >>> csv_fields = gen_fields(csv_header, True)
        >>> csv_records.next()['some_date']
        u'05/04/82'
        >>> casted_csv_row = gen_type_cast(csv_records, csv_fields).next()
        >>> casted_csv_values = [casted_csv_row[h] for h in csv_header]
        >>>
        >>> xls_filepath = p.join(parent_dir, 'testdata', 'test.xls')
        >>> xls_records = io.read_xls(xls_filepath, sanitize=True)
        >>> xls_header = sorted(xls_records.next().keys())
        >>> xls_fields = gen_fields(xls_header, True)
        >>> xls_records.next()['some_date']
        '1982-05-04'
        >>> casted_xls_row = gen_type_cast(xls_records, xls_fields).next()
        >>> casted_xls_values = [casted_xls_row[h] for h in xls_header]
        >>>
        >>> casted_csv_values == casted_xls_values
        True
        >>> casted_csv_values
        ['2015-01-01', 100.0, None, None]
    """
    make_date_p = partial(make_date, date_format=date_format)
    make_unicode = lambda v: unicode(v) if v and v.strip() else None
    switch = {'float': make_float, 'date': make_date_p, 'text': make_unicode}
    field_types = {f['id']: f['type'] for f in fields}

    for row in records:
        yield {k: switch.get(field_types[k])(v) for k, v in row.items()}


def gen_fields(names, type_cast=False):
    """Tries to determine field types based on field names.

    Args:
        names (List[str]): Field names.

    Yields:
        dict: The parsed field with type

    Examples:
        >>> gen_fields(['date', 'raw_value', 'text']).next()
        {u'type': u'text', u'id': u'date'}
    """
    for name in names:
        if type_cast and 'date' in name:
            yield {'id': name, 'type': 'date'}
        elif type_cast and 'value' in name:
            yield {'id': name, 'type': 'float'}
        else:
            yield {'id': name, 'type': 'text'}
