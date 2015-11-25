#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.process
~~~~~~~~~~~~~~~~

Provides methods for processing `records`, i.e., tabular data.

Examples:
    basic usage::

        from tabutils.process import type_cast

        records = [{'some_value', '1'}, {'some_value', '2'}]
        casted_records = type_cast(records, [{'some_value': 'int'}]).next()

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import itertools as it
import hashlib

from functools import partial
from collections import defaultdict
from operator import itemgetter
from math import log1p

from . import convert as cv, fntools as ft, typetools as tt


def type_cast(records, types, warn=False):
    """Casts record entries based on field types.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        types (Iter[dicts]): Field types (`guess_type_by_field` or
            `guess_type_by_value` output).

        warn (bool): Raise error if value can't be cast (default: False).

    Yields:
        dict: Type casted record. A row of data whose keys are the field names.

    See also:
        `convert.to_int`
        `convert.to_float`
        `convert.to_decimal`
        `convert.to_date`
        `convert.to_time`
        `convert.to_datetime`
        `convert.to_bool`

    Examples:
        >>> import datetime
        >>> record = {
        ...     'null': 'None',
        ...     'bool': 'false',
        ...     'int': '10',
        ...     'float': '1.5',
        ...     'unicode': 'Iñtërnâtiônàližætiøn',
        ...     'date': '5/4/82',
        ...     'time': '2:30',
        ...     'datetime': '5/4/82 2pm',
        ... }
        >>> types = tt.guess_type_by_value(record)
        >>> type_cast([record], types).next() == {
        ...     u'null': None,
        ...     u'bool': False,
        ...     u'int': 10,
        ...     u'float': 1.5,
        ...     u'unicode': u'Iñtërnâtiônàližætiøn',
        ...     u'date': datetime.date(1982, 5, 4),
        ...     u'time': datetime.time(2, 30),
        ...     u'datetime': datetime.datetime(1982, 5, 4, 14, 0),
        ... }
        True
        >>> records = [{'float': '1.5'}]
        >>> types = [{'id': 'float', 'type': 'bool'}]
        >>> type_cast(records, types).next()
        {u'float': False}
        >>> type_cast(records, types, warn=True).next()
        Traceback (most recent call last):
        ValueError: Invalid bool value: `1.5`.

    """
    switch = {
        'int': cv.to_int,
        'float': cv.to_float,
        'decimal': cv.to_decimal,
        'date': cv.to_date,
        'time': cv.to_time,
        'datetime': cv.to_datetime,
        'unicode': lambda v, warn=None: unicode(v) if v and v.strip() else '',
        'null': lambda x, warn=None: None,
        'bool': cv.to_bool,
    }

    field_types = {t['id']: t['type'] for t in types}

    for row in records:
        items = row.items()
        yield {k: switch.get(field_types[k])(v, warn=warn) for k, v in items}


def gen_confidences(tally, types, a=1):
    """Calculates confidence using a logarithmic function which asymptotically
    approaches 1.

    Args:
        tally (dict): Rows of data whose keys are the field names and whose
            data is a dict of types and counts.

        types (Iter[dicts]): Field types (`guess_type_by_field` or
            `guess_type_by_value` output).

        a (int): logarithmic weighting, a higher value will converge faster
            (default: 1)

    Returns:
        Iter(decimal): Generator of confidences

    Examples:
        >>> record = {'field_1': 'None', 'field_2': 'false'}
        >>> types = list(tt.guess_type_by_value(record))
        >>> tally = {'field_1': {'null': 3}, 'field_2': {'bool': 2}}
        >>> list(gen_confidences(tally, types))
        [Decimal('0.52'), Decimal('0.58')]
        >>> list(gen_confidences(tally, types, 5))
        [Decimal('0.85'), Decimal('0.87')]
    """
    # http://math.stackexchange.com/a/354879
    calc = lambda x: cv.to_decimal(a * x / (1 + a * x))
    return (calc(log1p(tally[t['id']][t['type']])) for t in types)


def gen_types(tally):
    """Selects the field type with the highest count.

    Args:
        tally (dict): Rows of data whose keys are the field names and whose
            data is a dict of types and counts.

    Yields:
        dict: Field type. The parsed field and its type.

    Examples:
        >>> tally = {
        ...     'field_1': {'null': 3, 'bool': 1},
        ...     'field_2': {'bool': 2, 'int': 4}}
        >>> types = sorted(list(gen_types(tally)), key=itemgetter('id'))
        >>> types[0] == {u'id': u'field_1', u'type': u'null'}
        True
        >>> types[1] == {u'id': u'field_2', u'type': u'int'}
        True
    """
    for key, value in tally.items():
        ttypes = [{'type': k, 'count': v} for k, v in value.items()]
        highest = sorted(ttypes, key=itemgetter('count'), reverse=True)[0]
        yield {'id': key, 'type': highest['type']}


def detect_types(records, min_conf=0.95, hweight=6, max_iter=100):
    """Detects record types.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        min_conf (float): minimum confidence level (default: 0.95)
        hweight (int): weight to give header row, a higher value will
            converge faster (default: 6). E.g.,

        max_iter (int): maximum number of iterations to perform (default: 100)

            detect_types(records, 0.9, 3)['count'] == 23
            detect_types(records, 0.9, 4)['count'] == 10
            detect_types(records, 0.9, 5)['count'] == 6
            detect_types(records, 0.95, 5)['count'] == 31
            detect_types(records, 0.95, 6)['count'] == 17
            detect_types(records, 0.95, 7)['count'] == 11

    Returns:
        tuple(Iter[dict], dict): Tuple of records and the result

    Examples:
        >>> record = {
        ...     'null': 'None',
        ...     'bool': 'false',
        ...     'int': '1',
        ...     'float': '1.5',
        ...     'unicode': 'Iñtërnâtiônàližætiøn',
        ...     'date': '5/4/82',
        ...     'time': '2:30',
        ...     'datetime': '5/4/82 2pm',
        ... }
        >>> records = it.repeat(record)
        >>> records, result = detect_types(records)
        >>> result['count']
        17
        >>> result['confidence']
        Decimal('0.95')
        >>> result['accurate']
        True
        >>> {r['id']: r['type'] for r in result['types']} == {
        ...     u'null': u'null',
        ...     u'bool': u'bool',
        ...     u'int': u'int',
        ...     u'float': u'float',
        ...     u'unicode': u'unicode',
        ...     u'date': u'date',
        ...     u'time': u'time',
        ...     u'datetime': u'datetime',
        ... }
        True
        >>> records.next() == record
        True
        >>> result = detect_types(records, 0.99)[1]
        >>> result['count']
        100
        >>> result['confidence']
        Decimal('0.97')
        >>> result['accurate']
        False
        >>> result = detect_types([record, record])[1]
        >>> result['count']
        2
        >>> result['confidence']
        Decimal('0.87')
        >>> result['accurate']
        False
    """
    records = iter(records)
    tally = {}
    consumed = []

    if hweight < 1:
        raise ValueError('`hweight` must be greater than or equal to 1!')

    if min_conf >= 1:
        raise ValueError('`min_conf must` be less than 1!')

    for record in records:
        if not tally:
            # take a first guess using the header
            ftypes = tt.guess_type_by_field(record.keys())
            tally = {t['id']: defaultdict(int) for t in ftypes}

            for t in ftypes:
                # TODO: figure out using the below in place of above alters the
                # result
                # tally[t['id']] = defaultdict(int)
                tally[t['id']][t['type']] += hweight

        # now guess using the values
        for t in tt.guess_type_by_value(record):
            tally[t['id']][t['type']] += 1

        types = list(gen_types(tally))
        confidence = min(gen_confidences(tally, types, hweight))
        consumed.append(record)
        count = len(consumed)

        if (confidence >= min_conf) or count >= max_iter:
            break

    records = it.chain(consumed, records)

    result = {
        'confidence': confidence,
        'types': types,
        'count': count,
        'accurate': confidence >= min_conf}

    return records, result


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
        fields (Seq[str]): Names of the columns to fill (default: None, i.e.,
            all).

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `fntools.fill`

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'bad.csv')
        >>> records = list(io.read_csv(filepath))
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
        pred (func): Predicate. Receives the current key and should return
            `True` if overlapping values should be combined. Can optionally be
            a keyfunc which receives a record. In this case, the entries will
            be combined if the value obtained after applying keyfunc to the
            record equals the current value. Requires that `op` is set.

            If not set, no records will be combined.

            If a key occurs in multiple records and isn't combined, it will be
            overwritten by the last record.

        op (func): Receives a list of 2 values from overlapping keys and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `pred` is set. If a key is not
            present in a record, the value from `default` will be used. Note,
            since `op` applied inside of `reduce`, it may not perform as
            expected for all functions for more than 2 records. E.g. an average
            function will be applied as follows:

                ave([1, 2, 3]) --> ave([ave([1, 2]), 3])

            You would expect to get 2, but will instead get 2.25.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        dict: merged record

    See also:
        `process.aggregate`
        `fntools.combine`

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> pred = lambda key: key == 'amount'
        >>> merge(records, pred=pred, op=sum)
        {u'a': u'item', u'amount': 900}
        >>> merge(records)
        {u'a': u'item', u'amount': 400}
        >>> sorted(merge([{'a': 1, 'b': 2}, {'b': 10, 'c': 11}]).items())
        [(u'a', 1), (u'b', 10), (u'c', 11)]
        >>> records = [{'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        >>>
        >>> # Combine all keys
        >>> pred = lambda key: True
        >>> sorted(merge(records, pred=pred, op=sum).items())
        [(u'a', 1), (u'b', 6), (u'c', 8), (u'd', 6)]
        >>> fltrer = lambda x: x is not None
        >>> first = lambda pair: filter(fltrer, pair)[0]
        >>> kwargs = {'pred': pred, 'op': first, 'default': None}
        >>> sorted(merge(records, **kwargs).items())
        [(u'a', 1), (u'b', 2), (u'c', 3), (u'd', 6)]
        >>>
        >>> # This will only reliably give the expected result for 2 records
        >>> average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        >>> kwargs = {'pred': pred, 'op': average, 'default': None}
        >>> sorted(merge(records, **kwargs).items())
        [(u'a', 1), (u'b', 3.0), (u'c', 4.0), (u'd', 6.0)]
        >>>
        >>> # Only combine key 'b'
        >>> pred = lambda key: key == 'b'
        >>> sorted(merge(records, pred=pred, op=sum).items())
        [(u'a', 1), (u'b', 6), (u'c', 5), (u'd', 6)]
        >>>
        >>> # Only combine keys that have the same value of 'b'
        >>> from operator import itemgetter
        >>> pred = itemgetter('b')
        >>> sorted(merge(records, pred=pred, op=sum).items())
        [(u'a', 1), (u'b', 6), (u'c', 5), (u'd', 6)]
        >>>
        >>> # This will reliably work for any number of records
        >>> from collections import defaultdict
        >>>
        >>> counted = defaultdict(int)
        >>> pred = lambda key: True
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
        >>> summed = merge(records, pred=pred, op=sum)
        >>> sorted(summed.items())
        [(u'a', 6), (u'b', 15), (u'c', 2), (u'd', 7)]
        >>> kwargs = {'pred': pred, 'op': divide}
        >>> sorted(merge([summed, counted], **kwargs).items())
        [(u'a', 2.0), (u'b', 5.0), (u'c', 1.0), (u'd', 7.0)]
    """
    def reducer(x, y):
        _merge = partial(ft.combine, x, y, **kwargs)
        new_y = ((k, _merge(k, v)) for k, v in y.iteritems())
        return dict(it.chain(x.iteritems(), new_y))

    if kwargs.get('pred') and kwargs.get('op'):
        record = reduce(reducer, records)
    else:
        items = it.imap(dict.iteritems, records)
        record = dict(it.chain.from_iterable(items))

    return record


def aggregate(records, key, op, default=0):
    """Aggregates `records` on a specified key.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        key (str): The field to aggregate

        op (func): Aggregation function. Receives a list of all non-null values
            and should return the combined value. Common operators are `sum`,
            `min`, `max`, etc.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        dict: The first record with an aggregated value for `key`

    See also:
        `process.merge`

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> aggregate(records, 'amount', sum)
        {u'a': u'item', u'amount': 900}
        >>> aggregate(records, 'amount', lambda x: sum(x) / len(x))
        {u'a': u'item', u'amount': 300.0}
    """
    records = iter(records)
    first = records.next()
    values = (r.get(key, default) for r in it.chain([first], records))
    value = op(filter(lambda x: x is not None, values))
    return dict(it.chain(first.items(), [(key, value)]))


def group(records, keyfunc=None):
    """Groups records by keyfunc

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        keyfunc (func): Function which receives a record and selects which
            value to sort/group by.

    Returns:
        Iter(tuple[key, group]): Generator of tuples

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> group(records, itemgetter('amount')).next()[1]
        [{u'a': u'item', u'amount': 200}]
    """
    sorted_records = sorted(records, key=keyfunc)
    grouped = it.groupby(sorted_records, keyfunc)
    return ((key, list(group)) for key, group in grouped)


def pivot(records, data, column, op=sum, **kwargs):
    """
    Create a spreadsheet-style pivot table.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        data (str): Field to aggregate
        column (str): Field to group by and create columns for in the resulting
            pivot table.

        op (func): Aggregation function (default: sum)
        kwargs (dict): keyword arguments

    Kwargs:
        rows (Seq[str]): Fields to include as rows in the resulting pivot table
            (default: All fields not in `data` or `column`).

        fill_value (scalar): Value to replace missing values with
            (default: None)

        dropna (bool): Do not include columns with missing values
            (default: True)

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `process.aggregate`

    Examples:
        >>> records = [
        ...     {'length': 5, 'width': 2, 'species': 'setosa', 'color': 'red'},
        ...     {'length': 5, 'width': 2, 'species': 'setosa', 'color': 'blue'},
        ...     {'length': 6, 'width': 2, 'species': 'versi', 'color': 'red'},
        ...     {'length': 6, 'width': 2, 'species': 'versi', 'color': 'blue'}]
        ...
        >>> pivot(records, 'length', 'species', rows=['width']).next() == {
        ...     u'width': 2, u'setosa': 10, u'versi': 12}
        True
        >>> pivot(records, 'length', 'species').next() == {
        ...     u'width': 2, u'color': u'blue', u'setosa': 5, u'versi': 6}
        True
    """
    records = iter(records)
    first = records.next()
    chained = it.chain([first], records)
    keys = set(first.keys())
    rows = kwargs.get('rows', keys.difference([data, column]))
    filterer = lambda x: x[0] in rows
    keyfunc = lambda r: tuple(map(r.get, it.chain(rows, [column])))
    grouped = group(chained, keyfunc)

    def gen_raw(grouped):
        for key, groups in grouped:
            r = aggregate(groups, data, op)
            filtered = filter(filterer, r.items())
            yield dict(it.chain([(r[column], r.get(data))], filtered))

    raw = gen_raw(grouped)

    for key, groups in group(raw, lambda r: tuple(map(r.get, rows))):
        yield merge(groups)


def tfilter(records, field, pred=None):
    """ Yields records for which the predicate is True for a given field.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        field (str): The column to to apply the predicate to.

    Kwargs:
        pred (func): Predicate. Receives the value of `field` and should return
            `True`  if the record should be included (default: None, i.e.,
            return the record if value is True).

    Returns:
        Iter[dict]: The filtered records.

    Examples:
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'bob'},
        ...     {'day': 1, 'name': 'tom'},
        ...     {'day': 2, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'rob'},
        ... ]
        >>> tfilter(records, 'day', lambda x: x == 2).next()['name'] == \
u'Iñtërnâtiônàližætiøn'
        True
        >>> tfilter(records, 'day', lambda x: x == 3).next()['name']
        u'rob'
    """
    predicate = lambda x: pred(x.get(field)) if pred else None
    return it.ifilter(predicate, records)


def unique(records, fields=None, pred=None):
    """ Yields unique records

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        fields (Seq[str]): The columns to use for testing uniqueness
            (default: None, i.e., all columns). Overridden by `pred`.

        pred (func): Predicate. Receives a record and should return a value for
            testing uniqueness. Overrides `fields`.

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'bob'},
        ...     {'day': 1, 'name': 'tom'},
        ...     {'day': 2, 'name': 'bill'},
        ...     {'day': 2, 'name': 'bob'},
        ...     {'day': 2, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'bob'},
        ...     {'day': 3, 'name': 'rob'},
        ... ]
        >>> it.islice(unique(records), 3, 4).next()['name']
        u'bill'
        >>> it.islice(unique(records, ['name']), 3, 4).next()['name'] == \
u'Iñtërnâtiônàližætiøn'
        True
        >>> pred = lambda x: x['name'][0]
        >>> it.islice(unique(records, pred=pred), 3, 4).next()['name']
        u'rob'
    """
    seen = set()

    for r in records:
        if not pred:
            unique = set(fields or r.keys())
            items = tuple(sorted((k, v) for k, v in r.items() if k in unique))

        entry = pred(r) if pred else items

        if entry not in seen:
            seen.add(entry)
            yield r


def cut(records, **kwargs):
    """
    Edit records to only return specified columns. Like unix `cut`, but for
    tabular data.'

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        kwargs (dict): keyword arguments

    Kwargs:
        include (Iter[str]): Column names to include. (default: None, i.e.,
            all columns.'). If the same field is also in `exclude`, it will
            still be included.

        exclude (Iter[str]): Column names to exclude (default: None, i.e.,
            no columns.'). If the same field is also in `include`, it will
            still be included.

        prune (bool): Remove empty rows from result.

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [
        ...     {'field_1': 1, 'field_2': 'bill', 'field_3': 'male'},
        ...     {'field_1': 2, 'field_2': 'bob', 'field_3': 'male'},
        ...     {'field_1': 3, 'field_2': 'jane', 'field_3': 'female'},
        ... ]
        >>> cut(records).next() == {
        ...     u'field_1': 1, u'field_2': u'bill', u'field_3': u'male'}
        ...
        True
        >>> cut(records, include=['field_2']).next()
        {u'field_2': u'bill'}
        >>> cut(records, exclude=['field_2']).next() == {
        ...     u'field_1': 1, u'field_3': u'male'}
        ...
        True
        >>> include = ['field_1', 'field_2']
        >>> cut(records, include=['field_2'], exclude=['field_2']).next()
        {u'field_2': u'bill'}
    """
    include = kwargs.get('include')
    exclude = kwargs.get('exclude')
    blacklist = include or exclude
    filtered = (ft.dfilter(r, blacklist, include) for r in records)
    return it.ifilter(None, filtered) if kwargs.get('prune') else filtered


def grep(records, rules, any_match=False, inverse=False):
    """
    Yields rows which match all the given rules.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        rules (Iter[dict]): Each rule dict must contain a `pattern`
            key and the value can be either a string, function, or regular
            expression. A `fields` key is optional and corresponds to the
            columns you wish to pattern match. Default is to search all columns.

        any_match (bool): Return records which match any of the rules
            (default: False)

        inverse (bool): Only return records which don't match the rules
            (default: None, i.e., all columns)

    Returns:
        Iter[dict]: The filtered records.

    Examples:
        >>> import re
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'rob'},
        ...     {'day': 1, 'name': 'jane'},
        ...     {'day': 2, 'name': 'rob'},
        ...     {'day': 3, 'name': 'jane'},
        ... ]
        >>> rules = [{'fields': ['name'], 'pattern': 'o'}]
        >>> grep(records, rules).next()['name']
        u'rob'
        >>> rules = [{'fields': ['name'], 'pattern': re.compile(r'j.*e$')}]
        >>> grep(records, rules).next()['name']
        u'jane'
        >>> rules = [{'fields': ['day'], 'pattern': lambda x: x == 1}]
        >>> grep(records, rules).next()['name']
        u'bill'
        >>> rules = [{'pattern': lambda x: x in {1, 'rob'}}]
        >>> grep(records, rules).next()['name']
        u'rob'
        >>> rules = [{'pattern': lambda x: x in {1, 'rob'}}]
        >>> grep(records, rules, any_match=True).next()['name']
        u'bill'
        >>> rules = [{'fields': ['name'], 'pattern': 'o'}]
        >>> grep(records, rules, inverse=True).next()['name']
        u'bill'
    """
    def predicate(record):
        for rule in rules:
            for field in rule.get('fields', record.keys()):
                value = record[field]
                p = rule['pattern']

                try:
                    passed = p.match(value)
                except AttributeError:
                    passed = p(value) if callable(p) else p in value

                if (any_match and passed) or not (any_match or passed):
                    break

        return not passed if inverse else passed

    return it.ifilter(predicate, records)


def hash(records, fields=None, algo='md5'):
    """ Yields rows whose value of the given field(s) are hashed

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        fields (Seq[str]): The columns to use for testing uniqueness
            (default: None, i.e., all columns). Overridden by `pred`.

        algo (str): The hashlib hashing algorithm to use (default: sha1).
            supported algorithms: md5, ripemd128, ripemd160, ripemd256,
                ripemd320, sha1, sha256, sha512, sha384, whirlpool

    See also:
        `io.hash_file`

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [{'a': 'item', 'amount': 200}]
        >>> hash(records, ['a']).next() == {
        ...     u'a': '447b7147e84be512208dcc0995d67ebc', u'amount': 200}
        True
    """
    hasher = getattr(hashlib, algo)
    hash_func = lambda x: hasher(str(x)).hexdigest()
    to_hash = set(fields or [])

    for row in records:
        yield {k: hash_func(v) if k in to_hash else v for k, v in row.items()}
