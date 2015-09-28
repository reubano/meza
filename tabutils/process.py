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
import hashlib
import xlrd

from functools import partial

from . import convert as cv, fntools as ft

from chardet.universaldetector import UniversalDetector
from slugify import slugify
from xlrd.xldate import xldate_as_datetime as xl2dt
from xlrd import (
    XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_NUMBER, XL_CELL_BOOLEAN,
    XL_CELL_ERROR)


underscorify = lambda fields: [slugify(f, separator='_') for f in fields]


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
    numeric_like = ft.is_numeric_like(string)

    if numeric_like and char in string:
        excluded = [s for s in string.split(exclude) if char in s][0]
        after = len(excluded) - excluded.rfind(char) - 1
    elif numeric_like:
        after = -1
    else:
        raise TypeError('Not able to convert %s to a number' % string)

    return after


def xmlize(content):
    """ Recursively makes elements of an array xml compliant

    Args:
        content (Iter[str]): the content to clean

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
            yield ft.mreplace(item, replacements)
        else:
            try:
                yield list(xmlize(item))
            except TypeError:
                yield ft.mreplace(item, replacements) if item else ''


def type_cast(records, fields, date_format='%Y-%m-%d'):
    """Casts record entries based on field types.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        fields (Iter[dicts]): Field types (`guess_field_types` output).
        date_format (str): Date format passed to `strftime()` (default:
            '%Y-%m-%d', i.e, 'YYYY-MM-DD').

    Yields:
        dict: The type casted record entry.

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> csv_filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> csv_records = io.read_csv(csv_filepath, sanitize=True)
        >>> csv_header = sorted(csv_records.next().keys())
        >>> csv_fields = guess_field_types(csv_header, True)
        >>> csv_records.next()['some_date']
        u'05/04/82'
        >>> casted_csv_row = type_cast(csv_records, csv_fields).next()
        >>> casted_csv_values = [casted_csv_row[h] for h in csv_header]
        >>>
        >>> xls_filepath = p.join(parent_dir, 'data', 'test', 'test.xls')
        >>> xls_records = io.read_xls(xls_filepath, sanitize=True)
        >>> xls_header = sorted(xls_records.next().keys())
        >>> xls_fields = guess_field_types(xls_header, True)
        >>> xls_records.next()['some_date']
        '1982-05-04'
        >>> casted_xls_row = type_cast(xls_records, xls_fields).next()
        >>> casted_xls_values = [casted_xls_row[h] for h in xls_header]
        >>>
        >>> casted_csv_values == casted_xls_values
        True
        >>> casted_csv_values
        ['2015-01-01', 100.0, None, None]
    """
    to_date_p = partial(cv.to_date, date_format=date_format)
    to_unicode = lambda v: unicode(v) if v and v.strip() else None
    switch = {'float': cv.to_float, 'date': to_date_p, 'text': to_unicode}
    field_types = {f['id']: f['type'] for f in fields}

    for row in records:
        yield {k: switch.get(field_types[k])(v) for k, v in row.items()}


def guess_field_types(names, type_cast=False):
    """Tries to determine field types based on field names.

    Args:
        names (Iter[str]): Field names.

    Kwargs:
        type_cast (bool): (default: False)

    Yields:
        dict: The parsed field with type

    Examples:
        >>> guess_field_types(['date', 'raw_value', 'text']).next()
        {u'type': u'text', u'id': u'date'}
    """
    for name in names:
        if type_cast and 'date' in name:
            yield {'id': name, 'type': 'date'}
        elif type_cast and 'value' in name:
            yield {'id': name, 'type': 'float'}
        else:
            yield {'id': name, 'type': 'text'}


def hash_file(filepath, hasher='sha1', chunksize=0, verbose=False):
    """Hashes a file or file like object.
    http://stackoverflow.com/a/1131255/408556

    Args:
        filepath (str): The file path or file like object to hash.
        hasher (str): The hashlib hashing algorithm to use (default: sha1).

        chunksize (Optional[int]): Number of bytes to write at a time
            (default: 0, i.e., all).

        verbose (Optional[bool]): Print debug statements (default: False).

    Returns:
        str: File hash.

    Examples:
        >>> from tempfile import TemporaryFile
        >>> hash_file(TemporaryFile())
        'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    """
    def read_file(f, hasher):
        if chunksize:
            while True:
                data = f.read(chunksize)
                if not data:
                    break

                hasher.update(data)
        else:
            hasher.update(f.read())

        return hasher.hexdigest()

    hasher = getattr(hashlib, hasher)()

    if hasattr(filepath, 'read'):
        file_hash = read_file(filepath, hasher)
    else:
        with open(filepath, 'rb') as f:
            file_hash = read_file(f, hasher)

    if verbose:
        print('File %s hash is %s.' % (filepath, file_hash))

    return file_hash


def _fuzzy_match(items, possibilities, **kwargs):
    for i in items:
        for p in possibilities:
            if p in i.lower():
                yield i


def _exact_match(*args, **kwargs):
    sets = (set(i.lower() for i in arg) for arg in args)
    return iter(reduce(lambda x, y: x.intersection(y), sets))


def find(*args, **kwargs):
    method = kwargs.pop('method', 'exact')
    default = kwargs.pop('default', '')
    funcs = {'exact': _exact_match, 'fuzzy': _fuzzy_match}
    func = funcs.get('method', method)

    try:
        return func(*args, **kwargs).next()
    except StopIteration:
        return default


def detect_encoding(f, verbose=False):
    """Detects a file's encoding.

    Args:
        f (obj): The file like object to detect.

    Returns:
        dict: The encoding result

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> f = open(filepath, 'rU')
        >>> result = detect_encoding(f)
        >>> f.close()
        >>> result
        {'confidence': 0.99, 'encoding': 'utf-8'}
    """
    f.seek(0)
    detector = UniversalDetector()

    for line in f:
        detector.feed(line)

        if detector.done:
            break

    detector.close()

    if verbose:
        print('detector.result', detector.result)

    return detector.result


def sanitize_sheet(sheet, mode, date_format):
    """Formats content from xls/xslx files as strings according to its cell
    type.

    Args:
        book (obj): `xlrd` workbook object.
        mode (str): `xlrd` workbook datemode property.
        date_format (str): `strftime()` date format.

    Yields:
        Tuple[int, str]: A tuple of (row_number, value).

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.xls')
        >>> book = xlrd.open_workbook(filepath)
        >>> sheet = book.sheet_by_index(0)
        >>> sheet.row_values(1) == [
        ...     30075.0, u'Iñtërnâtiônàližætiøn', 234.0, u'Ādam', u' ']
        True
        >>> sanitized = sanitize_sheet(sheet, book.datemode, '%Y-%m-%d')
        >>> [v for i, v in sanitized if i == 1] == [
        ...     '1982-05-04', u'Iñtërnâtiônàližætiøn', u'234.0', u'Ādam', u' ']
        True
    """
    switch = {
        XL_CELL_DATE: lambda v: xl2dt(v, mode).strftime(date_format),
        XL_CELL_EMPTY: lambda v: None,
        XL_CELL_NUMBER: lambda v: unicode(v),
        XL_CELL_BOOLEAN: lambda v: unicode(bool(v)),
        XL_CELL_ERROR: lambda v: xlrd.error_text_from_code[v],
    }

    for i in xrange(sheet.nrows):
        for ctype, value in it.izip(sheet.row_types(i), sheet.row_values(i)):
            yield (i, switch.get(ctype, lambda v: v)(value))


def fillempty(records, value=None, method=None, limit=None, cols=None):
    """Fills in missing data with either a single value or front/back/side
    filled.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

    Kwargs:
        value (str): Value to use to fill holes (default: None).
        method (str): Fill method, one of either {'front', 'back'} or a column
            name (default: None). `front` propagates the last valid
            value forward. `back` propagates the next valid value
            backwards. If given a column name, that column's current value
            will be used. Note: if `back` is selected, the entire content will
            be read into memory. Use with caution.

        limit (int): Max number of consecutive rows to fill (default: None).
        cols (List[str]): Names of the columns to fill (default: None, i.e.,
            all).

    Yields:
        dict: A row of data whose keys are the field names.

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'bad.csv')
        >>> records = list(io.read_csv(filepath, remove_header=True))
        >>> records == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': u'',
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': None,
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> list(fillempty(records, 0)) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': 0,
        ...    }, {
        ...        u'column_a': 0,
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': 0,
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> list(fillempty(records, 0, cols=['column_a'])) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': 0,
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': None,
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> list(fillempty(records, method='front')) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': u'1',
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> list(fillempty(records, method='back')) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'17',
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': u'17',
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> list(fillempty(records, method='back', limit=1)) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': u'17',
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
        >>> kwargs = {'method': 'column_b', 'cols': ['column_a']}
        >>> list(fillempty(records, **kwargs)) == [
        ...    {
        ...        u'column_a': u'1',
        ...        u'column_b': u'27',
        ...        u'column_c': u'',
        ...    }, {
        ...        u'column_a': u"I'm too short!",
        ...        u'column_b': u"I'm too short!",
        ...        u'column_c': None,
        ...    }, {
        ...        u'column_a': u'0',
        ...        u'column_b': u'mixed types.... uh oh',
        ...        u'column_c': u'17',
        ...    }]
        True
    """
    if method and value is not None:
        raise Exception('You can not specify both a `value` and `method`.')
    elif not method and value is None:
        raise Exception('You must specify either a `value` or `method`.')
    elif method == 'back':
        content = reversed(records)
    else:
        content = records

    kwargs = {
        'value': value,
        'limit': limit,
        'cols': cols,
        'fill_key': method if method not in {'front', 'back'} else None
    }

    prev_row = {}
    count = {}
    length = 0
    result = []

    for row in content:
        length = length or len(row)
        filled = ft.fill(prev_row, row, count=count, **kwargs)
        prev_row = dict(it.islice(filled, length))
        count = filled.next()

        if method == 'back':
            result.append(prev_row)
        else:
            yield prev_row

    if method == 'back':
        for row in reversed(result):
            yield row
