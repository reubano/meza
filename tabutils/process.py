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

from . import convert as cv, fntools as ft


def type_cast(records, fields):
    """Casts record entries based on field types.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        fields (Iter[dicts]): Field types (`guess_field_types` output).

    Yields:
        dict: Type casted record. A row of data whose keys are the field names.

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> csv_filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> csv_records = io.read_csv(csv_filepath, sanitize=True)
        >>> csv_header = sorted(csv_records.next().keys())
        >>> csv_fields = ft.guess_field_types(csv_header)
        >>> csv_records.next()['some_date']
        u'05/04/82'
        >>> casted_csv_row = type_cast(csv_records, csv_fields).next()
        >>> casted_csv_values = [casted_csv_row[h] for h in csv_header]
        >>>
        >>> xls_filepath = p.join(parent_dir, 'data', 'test', 'test.xls')
        >>> xls_records = io.read_xls(xls_filepath, sanitize=True)
        >>> xls_header = sorted(xls_records.next().keys())
        >>> xls_fields = ft.guess_field_types(xls_header)
        >>> xls_records.next()['some_date']
        '1982-05-04'
        >>> casted_xls_row = type_cast(xls_records, xls_fields).next()
        >>> casted_xls_values = [casted_xls_row[h] for h in xls_header]
        >>>
        >>> casted_csv_values == casted_xls_values
        True
        >>> casted_csv_values
        [datetime.datetime(2015, 1, 1, 0, 0), 100.0, None, None]
    """
    switch = {
        'int': int,
        'float': cv.to_float,
        'date': cv.to_date,
        'datetime': cv.to_date,
        'text': lambda v: unicode(v) if v and v.strip() else None
    }

    field_types = {f['id']: f['type'] for f in fields}

    for row in records:
        yield {k: switch.get(field_types[k])(v) for k, v in row.items()}


def fillempty(records, value=None, method=None, limit=None, fields=None):
    """Replaces missing data with either a single value or by front/back/side
    filling.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

    Kwargs:
        value (str): Value to use to fill holes (default: None).
        method (str): Fill method, one of either {'front', 'back'} or a column
            name (default: None). `front` propagates the last valid
            value forward. `back` propagates the next valid value
            backwards. If given a column name, that column's current value
            will be used.

            *************************************************
            * Note: if `back` is selected, all records will *
            * be read into memory. Use with caution.        *
            *************************************************

        limit (int): Max number of consecutive rows to fill (default: None).
        fields (List[str]): Names of the columns to fill (default: None, i.e.,
            all).

    Yields:
        dict: Record. A row of data whose keys are the field names.

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
        >>> list(fillempty(records, 0, fields=['column_a'])) == [
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
        >>> kwargs = {'method': 'column_b', 'fields': ['column_a']}
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
        'fields': fields,
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


def merge(records, **kwargs):
    """Merges `records` and optionally combines specified keys using a
    specified binary operator.

    http://codereview.stackexchange.com/a/85822/71049
    http://stackoverflow.com/a/31812635/408556
    http://stackoverflow.com/a/3936548/408556

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        kwargs (dict): keyword arguments

    Kwargs:
        predicate (func): Receives a key and should return `True`
            if overlapping values should be combined. If a key occurs in
            multiple dicts and isn't combined, it will be overwritten
            by the last dict. Requires that `op` is set.

        op (func): Receives a list of 2 values from overlapping keys and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `predicate` is set. If a key is not
            present in all records, the value from `default` will be used. Note,
            since `op` applied inside of `reduce`, it may not perform as
            expected for all functions for more than 2 records. E.g. an average
            function will be applied as follows:

                ave([1, 2, 3]) --> ave([ave([1, 2]), 3])

            You would expect to get 2, but will instead get 2.25.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        (List[str]): collapsed content

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> predicate = lambda k: k == 'amount'
        >>> merge(records, predicate=predicate, op=sum)
        {u'a': u'item', u'amount': 900}
        >>> merge(records)
        {u'a': u'item', u'amount': 400}
        >>> sorted(merge([{'a': 1, 'b': 2}, {'b': 10, 'c': 11}]).items())
        [(u'a', 1), (u'b', 10), (u'c', 11)]
        >>> records = [{'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        >>>
        >>> # Combine all keys
        >>> predicate = lambda x: True
        >>> sorted(merge(records, predicate=predicate, op=sum).items())
        [(u'a', 1), (u'b', 6), (u'c', 8), (u'd', 6)]
        >>> fltrer = lambda x: x is not None
        >>> first = lambda x: filter(fltrer, x)[0]
        >>> kwargs = {'predicate': predicate, 'op': first, 'default': None}
        >>> sorted(merge(records, **kwargs).items())
        [(u'a', 1), (u'b', 2), (u'c', 3), (u'd', 6)]
        >>>
        >>> # This will only reliably give the expected result for 2 dicts
        >>> average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        >>> kwargs = {'predicate': predicate, 'op': average, 'default': None}
        >>> sorted(merge(records, **kwargs).items())
        [(u'a', 1), (u'b', 3.0), (u'c', 4.0), (u'd', 6.0)]
        >>>
        >>> # Only combine key 'b'
        >>> predicate = lambda k: k == 'b'
        >>> sorted(merge(records, predicate=predicate, op=sum).items())
        [(u'a', 1), (u'b', 6), (u'c', 5), (u'd', 6)]
        >>>
        >>> # This will reliably work for any number of dicts
        >>> from collections import defaultdict
        >>>
        >>> counted = defaultdict(int)
        >>> predicate = lambda x: True
        >>> divide = lambda x: x[0] / x[1]
        >>> records = [
        ...    {'a': 1, 'b': 4, 'c': 0},
        ...    {'a': 2, 'b': 5, 'c': 2},
        ...    {'a': 3, 'b': 6, 'd': 7}]
        ...
        >>> for r in records:
        ...     for k in r.keys():
        ...         counted[k] += 1
        ...
        >>> sorted(counted.items())
        [(u'a', 3), (u'b', 3), (u'c', 2), (u'd', 1)]
        >>> summed = merge(records, predicate=predicate, op=sum)
        >>> sorted(summed.items())
        [(u'a', 6), (u'b', 15), (u'c', 2), (u'd', 7)]
        >>> kwargs = {'predicate': predicate, 'op': divide}
        >>> sorted(merge([summed, counted], **kwargs).items())
        [(u'a', 2.0), (u'b', 5.0), (u'c', 1.0), (u'd', 7.0)]
    """
    predicate = kwargs.get('predicate')
    op = kwargs.get('op')
    default = kwargs.get('default', 0)

    def reducer(x, y):
        _merge = lambda k, v: op([x.get(k, default), v]) if predicate(k) else v
        new_y = ([k, _merge(k, v)] for k, v in y.iteritems())
        return dict(it.chain(x.iteritems(), new_y))

    if predicate and op:
        record = reduce(reducer, records)
    else:
        items = it.imap(dict.iteritems, records)
        record = dict(it.chain.from_iterable(items))

    return record
