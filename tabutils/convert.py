#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.convert
~~~~~~~~~~~~~~~~

Provides methods for converting data structures

Examples:
    basic usage::

        from tabutils.convert import to_decimal

        decimal = to_decimal('$123.45')
"""
from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import itertools as it

from os import path as p
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN
from io import StringIO
from json import dumps
from collections import OrderedDict
from operator import itemgetter
from functools import partial

from builtins import *
from six.moves import filterfalse
from dateutil.parser import parse
from . import fntools as ft, csv, ENCODING, DEFAULT_DATETIME


def ctype2ext(content_type=None):
    """Converts an http content type to a file extension.

    Args:
        content_type (str): Output file path or directory.

    Returns:
        str: file extension

    Examples:
        >>> ctype2ext('/csv;') == 'csv'
        True
        >>> ctype2ext('/xls;') == 'xls'
        True
        >>> ext = '/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
        >>> ctype2ext(ext) == 'xlsx'
        True
    """
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


def order_dict(content, order):
    """Converts a dict into an OrderedDict

    Args:
        content (dict): The content to convert.
        order (Seq[str]): The field order.

    Returns:
        OrderedDict: The ordered content.

    Examples:
        >>> order_dict({'a': 1, 'b': 2}, ['a', 'b']) == OrderedDict(
        ...     [('a', 1), ('b', 2)])
        True
    """
    get_order = {field: pos for pos, field in enumerate(order)}
    keyfunc = lambda x: get_order[x[0]]
    return OrderedDict(sorted(content.items(), key=keyfunc))


def to_bool(content, trues=None, falses=None, warn=False):
    """Formats strings into bool.

    Args:
        content (str): The content to parse.
        trues (Seq[str]): Values to consider True.
        falses (Seq[str]): Values to consider Frue.
        warn (bool): raise error if content can't be safely converted
            (default: False)

    See also:
        `process.type_cast`

    Returns:
        bool: The parsed content.

    Examples:
        >>> to_bool(True)
        True
        >>> to_bool('true')
        True
        >>> to_bool('y')
        True
        >>> to_bool(1)
        True
        >>> to_bool(False)
        False
        >>> to_bool('false')
        False
        >>> to_bool('n')
        False
        >>> to_bool(0)
        False
        >>> to_bool('')
        False
        >>> to_bool(None)
        False
        >>> to_bool(None, warn=True)
        Traceback (most recent call last):
        ValueError: Invalid bool value: `None`.

    Returns:
        bool
    """
    trues = set(map(str.lower, trues) if trues else ft.DEF_TRUES)

    if ft.is_bool(content):
        try:
            value = content.lower() in trues
        except (TypeError, AttributeError):
            value = bool(content)
    elif warn:
        raise ValueError('Invalid bool value: `%s`.' % content)
    else:
        value = False

    return value


def to_int(content, thousand_sep=',', decimal_sep='.', warn=False):
    """Formats strings into integers.

    Args:
        content (str): The number to parse.
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')
        warn (bool): raise error if content can't be safely converted
            (default: False)

    See also:
        `process.type_cast`

    Returns:
        flt: The parsed number.

    Examples:
        >>> to_int('$123.45')
        123
        >>> to_int('123€')
        123
        >>> to_int('2,123.45')
        2123
        >>> to_int('2.123,45', thousand_sep='.', decimal_sep=',')
        2123
        >>> to_int('2,123.45', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid int value: `2,123.45`.
        >>> to_int('spam')
        0
        >>> to_int('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid int value: `spam`.

    Returns:
        int
    """
    if warn and not ft.is_int(content):
        raise ValueError('Invalid int value: `%s`.' % content)

    try:
        value = int(float(ft.strip(content, thousand_sep, decimal_sep)))
    except ValueError:
        if warn:
            raise ValueError('Invalid int value: `%s`.' % content)
        else:
            value = 0

    return value


def to_float(content, thousand_sep=',', decimal_sep='.', warn=False):
    """Formats strings into floats.

    Args:
        content (str): The number to parse.
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')
        warn (bool): raise error if content can't be safely converted
            (default: False)

    Returns:
        flt: The parsed number.

    See also:
        `process.type_cast`

    Examples:
        >>> to_float('$123.45')
        123.45
        >>> to_float('123€')
        123.0
        >>> to_float('2,123.45')
        2123.45
        >>> to_float('2.123,45', thousand_sep='.', decimal_sep=',')
        2123.45
        >>> to_float('spam')
        0.0
        >>> to_float('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid float value: `spam`.

    Returns:
        float
    """
    if ft.is_numeric(content):
        value = float(ft.strip(content, thousand_sep, decimal_sep))
    elif warn:
        raise ValueError('Invalid float value: `%s`.' % content)
    else:
        value = 0.0

    return value


def to_decimal(content, thousand_sep=',', decimal_sep='.', **kwargs):
    """Formats strings into decimals

    Args:
        content (str): The string to parse.
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')
        kwargs (dict): Keyword arguments.

    Kwargs:
        warn (bool): raise error if content can't be safely converted
            (default: False)

        roundup (bool): Round up to the desired number of decimal places
             from 5 to 9 (default: True). If False, round up from 6 to 9.

        places (int): Number of decimal places to display (default: 2).

    See also:
        `process.type_cast`

    Examples:
        >>> to_decimal('$123.45')
        Decimal('123.45')
        >>> to_decimal('123€')
        Decimal('123.00')
        >>> to_decimal('2,123.45')
        Decimal('2123.45')
        >>> to_decimal('2.123,45', thousand_sep='.', decimal_sep=',')
        Decimal('2123.45')
        >>> to_decimal('1.554')
        Decimal('1.55')
        >>> to_decimal('1.555')
        Decimal('1.56')
        >>> to_decimal('1.555', roundup=False)
        Decimal('1.55')
        >>> to_decimal('1.556')
        Decimal('1.56')
        >>> to_decimal('spam')
        Decimal('0.00')
        >>> to_decimal('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid numeric value: `spam`.

    Returns:
        decimal
    """
    if ft.is_numeric(content):
        decimalized = Decimal(ft.strip(content, thousand_sep, decimal_sep))
    elif kwargs.get('warn'):
        raise ValueError('Invalid numeric value: `%s`.' % content)
    else:
        decimalized = Decimal(0)

    roundup = kwargs.get('roundup', True)
    rounding = ROUND_HALF_UP if roundup else ROUND_HALF_DOWN
    places = int(kwargs.get('places', 2))
    precision = '.%s1' % ''.join(it.repeat('0', places - 1))
    return decimalized.quantize(Decimal(precision), rounding=rounding)


def _to_datetime(content):
    """Parses and formats strings into datetimes.

    Args:
        content (str): The date to parse.

    Returns:
        [tuple(str, bool)]: Tuple of the formatted date string and retry value.

    Examples:
        >>> _to_datetime('5/4/82')
        (datetime.datetime(1982, 5, 4, 0, 0), False)
        >>> _to_datetime('2/32/82') == ('2/32/82', True)
        True
        >>> _to_datetime('spam')
        (datetime.datetime(9999, 12, 31, 0, 0), False)
    """
    try:
        value = parse(content, default=DEFAULT_DATETIME)
    except ValueError as e:
        # impossible date, e.g., 2/31/15
        retry = any(x in str(e) for x in ('out of range', 'day must be in'))
        value = content if retry else DEFAULT_DATETIME
    else:
        retry = False

    return (value, retry)


def to_datetime(content, dt_format=None, warn=False):
    """Parses and formats strings into datetimes.

    Args:
        content (str): The string to parse.

        dt_format (str): Date format passed to `strftime()`
            (default: None).

        warn (bool): raise error if content can't be safely converted
            (default: False)

    Returns:
        obj: The datetime object or formatted datetime string.

    See also:
        `process.type_cast`

    Examples:
        >>> fmt = '%Y-%m-%d %H:%M:%S'
        >>> to_datetime('5/4/82 2:00 pm')
        datetime.datetime(1982, 5, 4, 14, 0)
        >>> to_datetime('5/4/82 10:00', fmt) == '1982-05-04 10:00:00'
        True
        >>> to_datetime('2/32/82 12:15', fmt) == '1982-02-28 12:15:00'
        True
        >>> to_datetime('spam')
        datetime.datetime(9999, 12, 31, 0, 0)
        >>> to_datetime('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid datetime value: `spam`.

    Returns:
        datetime
    """
    bad_nums = map(str, range(29, 33))
    good_nums = map(str, range(31, 27, -1))

    try:
        bad_num = next(x for x in bad_nums if x in content)
    except StopIteration:
        options = [content]
    else:
        possibilities = (content.replace(bad_num, x) for x in good_nums)
        options = it.chain([content], possibilities)

    # Fix impossible dates, e.g., 2/31/15
    results = filterfalse(lambda x: x[1], map(_to_datetime, options))
    value = next(results)[0]

    if warn and value == DEFAULT_DATETIME:
        raise ValueError('Invalid datetime value: `%s`.' % content)
    else:
        datetime = value.strftime(dt_format) if dt_format else value

    return datetime


def to_date(content, date_format=None, warn=False):
    """Parses and formats strings into dates.

    Args:
        content (str): The string to parse.

        date_format (str): Time format passed to `strftime()` (default: None).

        warn (bool): raise error if content can't be safely converted
            (default: False)

    Returns:
        obj: The date object or formatted date string.

    See also:
        `process.type_cast`

    Examples:
        >>> to_date('5/4/82')
        datetime.date(1982, 5, 4)
        >>> to_date('5/4/82', '%Y-%m-%d') == '1982-05-04'
        True
        >>> to_date('2/32/82', '%Y-%m-%d') == '1982-02-28'
        True
        >>> to_date('spam')
        datetime.date(9999, 12, 31)
        >>> to_date('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid datetime value: `spam`.

    Returns:
        date
    """
    value = to_datetime(content, warn=warn).date()
    return value.strftime(date_format) if date_format else value


def to_time(content, time_format=None, warn=False):
    """Parses and formats strings into times.

    Args:
        content (str): The string to parse.
        time_format (str): Time format passed to `strftime()` (default: None).
        warn (bool): raise error if content can't be safely converted
            (default: False)

    Returns:
        obj: The time object or formatted time string.

    See also:
        `process.type_cast`

    Examples:
        >>> to_time('2:00 pm')
        datetime.time(14, 0)
        >>> to_time('10:00', '%H:%M:%S') == '10:00:00'
        True
        >>> to_time('2/32/82 12:15', '%H:%M:%S') == '12:15:00'
        True
        >>> to_time('spam')
        datetime.time(0, 0)
        >>> to_time('spam', warn=True)
        Traceback (most recent call last):
        ValueError: Invalid datetime value: `spam`.

    Returns:
        time
    """
    value = to_datetime(content, warn=warn).time()
    return value.strftime(time_format) if time_format else value


def to_filepath(filepath, **kwargs):
    """Creates a filepath from an online resource, i.e., linked file or
    google sheets export.

    Args:
        filepath (str): Output file path or directory.
        kwargs: Keyword arguments.

    Kwargs:
        headers (dict): HTTP response headers, e.g., `r.headers`.
        name_from_id (bool): Overwrite filename with resource id.
        resource_id (str): The resource id (required if `name_from_id` is True
            or filepath is a google sheets export)

    Returns:
        str: filepath

    Examples:
        >>> to_filepath('file.csv') == 'file.csv'
        True
        >>> to_filepath('.', resource_id='rid') == './rid.csv'
        Content-Type None not found in dictionary. Using default value.
        True
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


def df2records(df):
    """
    Converts a pandas DataFrame into records.

    Args:
        df (obj): pandas.DataFrame object

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `process.pivot`

    Examples:
        >>> try:
        ...    import pandas as pd
        ... except ImportError:
        ...    print(True)
        ... else:
        ...    records = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}]
        ...    df = pd.DataFrame.from_records(records)
        ...    next(df2records(df)) == {'a': 1, 'b': 2, 'c': 3}
        ...
        True
    """
    index = [_f for _f in df.index.names if _f]

    try:
        keys = index + df.columns.tolist()
    except AttributeError:
        # we have a Series, not a DataFrame
        keys = index + [df.name]
        rows = (i[0] + (i[1],) for i in df.items())
    else:
        rows = df.itertuples()

    for values in rows:
        if index:
            yield dict(zip(keys, values))
        else:
            yield dict(zip(keys, values[1:]))


def records2csv(records, encoding=ENCODING, bom=False):
    """
    Converts records into a csv file like object.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        encoding (str): File encoding (default: ENCODING constant)
        bom (bool): Add Byte order marker (default: False)

    Returns:
        obj: io.StringIO instance

    Examples:
        >>> records = [
        ...     {
        ...         'usda_id': 'IRVE2',
        ...         'species': 'Iris-versicolor',
        ...         'wikipedia_url': 'wikipedia.org/wiki/Iris_versicolor'}]
        ...
        >>> csv_str = records2csv(records)
        >>> set(next(csv_str).strip().split(',')) == {
        ...     'usda_id', 'species', 'wikipedia_url'}
        True
        >>> set(next(csv_str).strip().split(',')) == {
        ...     'IRVE2', 'Iris-versicolor',
        ...     'wikipedia.org/wiki/Iris_versicolor'}
        True
    """
    f = StringIO()
    irecords = iter(records)

    if bom:
        f.write('\ufeff'.encode(encoding))  # BOM for Windows

    row = next(irecords)
    w = csv.DictWriter(f, list(row.keys()))
    w.writeheader()
    w.writerow(row)
    w.writerows(irecords)
    f.seek(0)
    return f


def records2json(records, **kwargs):
    """
    Converts records into a json file like object.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

    Kwargs:
        indent (int): Number of spaces to indent (default: 2).
        newline (bool): Output newline delimited json (default: False)
        sort_keys (bool): Sort rows by keys (default: True).
        ensure_ascii (bool): Sort response dict by keys (default: False).

    Returns:
        obj: io.StringIO instance

    Examples:
        >>> from json import loads

        >>> record = {
        ...     'usda_id': 'IRVE2',
        ...     'species': 'Iris-versicolor',
        ...     'wikipedia_url': 'wikipedia.org/wiki/Iris_versicolor'}
        ...
        >>> result = loads(records2json([record]).read())
        >>> result[0] == record
        True
        >>> result = loads(records2json([record], newline=True).readline())
        >>> result == record
        True
    """
    newline = kwargs.pop('newline', False)
    jd = partial(dumps, cls=ft.CustomEncoder, **kwargs)
    json = '\n'.join(map(jd, records)) if newline else jd(records)
    return StringIO(str(json))


def _records2geojson(records, kw):
    """Helper function for converting records into a GeoJSON file like object.

     Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        kw (obj): `fntools.Objectify` instance with the following Attributes:
            key (str): GeoJSON Feature ID
            lat (str): latitude field name
            lon (str): longitude field name
            sort_keys (bool): Sort rows by keys

    Yields:
        tuple(dict, float, float): tuple of feature, lon, and lat

    Examples:
        >>> record = {
        ...     'id': 'gid', 'p1': 'prop', 'type': 'Point',
        ...     'coordinates': [12.2, 22.0]}
        ...
        >>> kw = ft.Objectify({'key': 'id', 'lon': 0, 'lat': 1})
        >>> next(_records2geojson([record], kw)) == {
        ...     'feature': {
        ...         'type': 'Feature',
        ...         'id': 'gid',
        ...         'geometry': {
        ...             'type': 'Point', 'coordinates': [12.2, 22.0]},
        ...         'properties': {'p1': 'prop'}},
        ...     'lons': [12.2],
        ...     'lats': [22.0]}
        True
    """
    black_list = {kw.key, 'type', 'coordinates'}

    for row in records:
        coordinates = row['coordinates']

        if row['type'] == 'Point':
            lons = [coordinates[kw.lon]]
            lats = [coordinates[kw.lat]]
        elif row['type'] == 'LineString':
            lons = [itemgetter(kw.lon)(r) for r in coordinates]
            lats = [itemgetter(kw.lat)(r) for r in coordinates]
        elif row['type'] == 'Polygon':
            get = lambda keyfunc: [[keyfunc(x) for x in c] for c in coordinates]
            lons = list(it.chain.from_iterable(get(itemgetter(kw.lon))))
            lats = list(it.chain.from_iterable(get(itemgetter(kw.lat))))
        else:
            raise TypeError('Invalid type: %s' % row['type'])

        geometry = {'type': row['type'], 'coordinates': coordinates}
        properties = dict(x for x in row.items() if x[0] not in black_list)

        if kw.sort_keys:
            geometry = order_dict(geometry, ['type', 'coordinates'])

        feature = {
            'type': 'Feature',
            'id': row.get(kw.key),
            'geometry': geometry,
            'properties': properties}

        if kw.sort_keys:
            feature_order = ['type', 'id', 'geometry', 'properties']
            feature = order_dict(feature, feature_order)

        yield {'feature': feature, 'lons': lons, 'lats': lats}


def records2geojson(records, **kwargs):
    """Converts records into a GeoJSON file like object.

     Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `tabutils.io` read function.

        kwargs (dict): Keyword arguments.

    Kwargs:
        key (str): GeoJSON Feature ID (default: 'geoid').
        lat (str): latitude field name (default: 'lat').
        lon (str): longitude field name (default: 'lon').
        crs (str): coordinate reference system field name (default: 'crs').
        indent (int): Number of spaces to indent (default: 2).
        sort_keys (bool): Sort rows by keys (default: True).
        ensure_ascii (bool): Sort response dict by keys (default: False).

    Returns:
        obj: io.StringIO instance

    Examples:
        >>> from json import loads

        >>> record = {
        ...     'id': 'gid', 'p1': 'prop', 'type': 'Point',
        ...     'coordinates': [12.2, 22.0]}
        ...
        >>> result = loads(next(records2geojson([record])))
        >>> result['type'] == 'FeatureCollection'
        True
        >>> result['bbox']
        [12.2, 22.0, 12.2, 22.0]
        >>> result['crs'] == {
        ...     'type': 'name', 'properties': {'name': None}}
        True
        >>> features = result['features']
        >>> sorted(features[0].keys()) == [
        ...     'geometry', 'id', 'properties', 'type']
        True
        >>> features[0]['geometry'] == {
        ...     'type': 'Point', 'coordinates': [12.2, 22.0]}
        True
    """
    defaults = {'key': 'id', 'lon': 0, 'lat': 1, 'indent': 2, 'sort_keys': True}
    kw = ft.Objectify(kwargs, **defaults)
    results = list(_records2geojson(records, kw))
    lons = list(it.chain.from_iterable(map(itemgetter('lons'), results)))
    lats = list(it.chain.from_iterable(map(itemgetter('lats'), results)))
    crs = {'type': 'name', 'properties': {'name': kw.crs}}

    if kw.sort_keys:
        crs = order_dict(crs, ['type', 'properties'])

    output = {
        'type': 'FeatureCollection',
        'bbox': [min(lons), min(lats), max(lons), max(lats)],
        'features': [itemgetter('feature')(r) for r in results],
        'crs': crs}

    if kw.sort_keys:
        output_order = ['type', 'bbox', 'features', 'crs']
        output = order_dict(output, output_order)

    dkwargs = ft.dfilter(kwargs, ['indent', 'sort_keys'], True)
    json = dumps(output, cls=ft.CustomEncoder, **dkwargs)
    return StringIO(str(json))
