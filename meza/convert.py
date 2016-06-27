#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.convert
~~~~~~~~~~~~

Provides methods for converting data structures

Examples:
    basic usage::

        from meza.convert import to_decimal

        decimal = to_decimal('$123.45')
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import itertools as it
import pygogo as gogo

from os import path as p
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN
from io import StringIO
from json import dumps
from collections import OrderedDict
from operator import itemgetter
from functools import partial
from array import array

from builtins import *
from six.moves import filterfalse, zip_longest
from dateutil.parser import parse
from . import fntools as ft, unicsv as csv, ENCODING, DEFAULT_DATETIME
from ._compat import get_native_str

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = gogo.Gogo(__name__, monolog=True).logger


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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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
        `meza.process.type_cast`

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


def array2records(data, native=False):
    """Converts either a numpy.recarray or a nested array.array into records

    Args:
        data (Iter[array]): The 2-D array.

        native (bool): (default: False)

    Returns:
        Iterable of dicts

    See also:
        `meza.convert.df2records`

    Examples:
        >>> arr = [[1, 2, 3], [4, 5, 6]] if np else [(1, 4), (2, 5), (3, 6)]
        >>> data = np.array(arr, 'i4') if np else [array('i', a) for a in arr]
        >>> native = not np
        >>> next(array2records(data, native)) == {
        ...     'column_1': 1, 'column_2': 2, 'column_3': 3}
        True
        >>> i, f, u = [get_native_str(x) for x in ['i', 'f', 'u']]
        >>> data = [
        ...     array(i, [1, 2, 3]),
        ...     array(f, [1.0, 2.0, 3.0]),
        ...     [array(u, 'one'), array(u, 'two'), array(u, 'three')]]
        >>> next(array2records(data, True)) == {
        ...     'column_1': 1, 'column_2': 1.0, 'column_3': 'one'}
        True
    """
    textify = lambda x: x.tounicode() if x.typecode == 'u' else x.tostring()
    datify = lambda x: x.tolist() if hasattr(x, 'tolist') else map(textify, x)

    if native and hasattr(data[0], 'typecode'):
        header = None
        data = zip(*map(datify, data))
    elif native:
        header = [textify(h) for h in data[0]]
        data = zip(*map(datify, data[1:]))
    else:
        header = data.dtype.names

    if not header:
        try:
            size = data.shape[1]
        except (IndexError, AttributeError):
            data = iter(data)
            first_row = next(data)
            size = len(first_row)
            data = it.chain([first_row], data)

        header = ['column_%i' % (n + 1) for n in range(size)]

    return (dict(zip(header, row)) for row in data)


def df2records(df):
    """Converts a pandas DataFrame into records.

    Args:
        df (obj): pandas.DataFrame object

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `meza.process.array2records`

    Examples:
        >>> records = [
        ...     {'a': 1, 'b': 2.0, 'c': 'three'},
        ...     {'a': 4, 'b': 5.0, 'c': 'six'}]

        >>> if pd:
        ...    df = pd.DataFrame(records)
        ...    converted = df2records(df)
        ... else:
        ...    converted = iter(records)

        >>> next(converted) == {'a': 1, 'b': 2.0, 'c': 'three'}
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


def records2array(records, types, native=False, silent=False):
    """Converts records into either a numpy.recarray or a nested array.array

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        types (Iter[dict]):

        native (bool): Return a native array (default: False).

        silent (bool): Suppress the warning message (default: False).

    Returns:
        numpy.recarray

    See also:
        `meza.convert.records2df`

    Examples:
        >>> records = [{'alpha': 'aa', 'beta': 2}, {'alpha': 'bee', 'beta': 3}]
        >>> types = [
        ...     {'id': 'alpha', 'type': 'text'}, {'id': 'beta', 'type': 'int'}]
        >>>
        >>> arr = records2array(records, types, silent=True)
        >>> u, i = get_native_str('u'), get_native_str('i')
        >>> native_resp = [
        ...     [array(u, 'alpha'), array(u, 'beta')],
        ...     [array(u, 'aa'), array(u, 'bee')],
        ...     array(i, [2, 3])]
        >>>
        >>> if np:
        ...     arr.alpha.tolist() == ['aa', 'bee']
        ...     arr.beta.tolist() == [2, 3]
        ... else:
        ...     True
        ...     True
        True
        True
        >>> True if np else arr == native_resp
        True
        >>> records2array(records, types, native=True) == native_resp
        True
    """
    numpy = np and not native
    dialect = 'numpy' if numpy else 'array'
    _dtype = [ft.get_dtype(t['type'], dialect) for t in types]
    dtype = [get_native_str(d) for d in _dtype]
    ids = [t['id'] for t in types]

    if numpy:
        data = [tuple(r.get(id_) for id_ in ids) for r in records]
        ndtype = [tuple(map(get_native_str, z)) for z in zip(ids, dtype)]
        ndarray = np.array(data, dtype=ndtype)
        converted = ndarray.view(np.recarray)
    else:
        if not (native or silent):
            msg = (
                "It looks like you don't have numpy installed. This function"
                " will return a native array instead.")

            logger.warning(msg)

        header = [array(get_native_str('u'), t['id']) for t in types]
        data = (zip_longest(*([r.get(i) for i in ids] for r in records)))

        # array.array can't have nulls, so convert to an appropriate equivalent
        clean = lambda t, d: (x if x else ft.ARRAY_NULL_TYPE[t] for x in d)
        cleaned = (it.starmap(clean, zip(dtype, data)))

        values = [
            [array(t, x) for x in d] if t in {'c', 'u'} else array(t, d)
            for t, d in zip(dtype, cleaned)]

        converted = [header] + values

    return converted


def records2df(records, types, native=False, silent=False):
    """Converts records into either a pandas.DataFrame

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        types (Iter[dict]):

        native (bool): Return a native array (default: False).

        silent (bool): Suppress the warning message (default: False).

    Returns:
        numpy.recarray

    See also:
        `meza.convert.records2array`

    Examples:
        >>> records = [
        ...     {'col_1': 'alpha', 'col_2': 1.0},
        ...     {'col_1': 'beta', 'col_2': 2.3}]
        >>> types = [
        ...     {'id': 'col_1', 'type': 'text'},
        ...     {'id': 'col_2', 'type': 'float'}]
        >>> df = records2df(records, types, silent=True)
        >>> u, f = get_native_str('u'), get_native_str('f')
        >>>
        >>> native_resp = [
        ...     [array(u, 'col_1'), array(u, 'col_2')],
        ...     [array(u, 'alpha'), array(u, 'beta')],
        ...     array(f, [1.0, 2.299999952316284])]
        >>>
        >>> if pd:
        ...     columns = df.columns.tolist()
        ...     columns == ['col_1', 'col_2']
        ...     df.col_1.tolist() == ['alpha', 'beta']
        ...     [np.round(v, 1) if pd else round(v, 1) for v in df.col_2]
        ... else:
        ...     True
        ...     True
        ...     [1.0, 2.3]
        True
        True
        [1.0, 2.3]
        >>> True if pd else df == native_resp
        True
        >>> records2df(records, types, native=True) == native_resp
        True
    """
    if pd and not native:
        recarray = records2array(records, types)
        df = pd.DataFrame.from_records(recarray)
    else:
        if not (native or silent):
            msg = (
                "It looks like you don't have pandas installed. This function"
                " will return a native array instead.")

            logger.warning(msg)

        df = records2array(records, types, native=True, silent=silent)

    return df


def records2csv(records, encoding=ENCODING, bom=False, skip_header=False):
    """Converts records into a csv file like object.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        encoding (str): File encoding (default: ENCODING constant)
        bom (bool): Add Byte order marker (default: False)
        skip_header (bool): Don't write the header (default: False)

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
    None if skip_header else w.writeheader()
    w.writerow(row)
    w.writerows(irecords)
    f.seek(0)
    return f


def records2json(records, **kwargs):
    """Converts records into a json file like object.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

    Kwargs:
        indent (int): Number of spaces to indent (default: 2).
        newline (bool): Output newline delimited json (default: False)
        sort_keys (bool): Sort rows by keys (default: True).
        ensure_ascii (bool): Sort response dict by keys (default: False).

    See also:
        `meza.convert.records2geojson`

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


def gen_features(subresults, kw):
    """Generates a geojson feature.

     Args:
        subresults (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        kw (obj): `fntools.Objectify` instance with the following Attributes:
            key (str): GeoJSON Feature ID
            lon (str): longitude field name
            lat (str): latitude field name
            sort_keys (bool): Sort rows by keys

    See also:
        `meza.convert.records2geojson`

    Yields:
        dict: a geojson feature

    Examples:
        >>> record = {
        ...     'id': 'gid', 'p1': 'prop', 'type': 'Point',
        ...     'lon': Decimal('12.2'), 'lat': Decimal('22.0')}
        >>> subresults = [((record['lon'], record['lat']), record)]
        >>> kw = ft.Objectify({'key': 'id', 'lon': 'lon', 'lat': 'lat'})
        >>> next(gen_features(subresults, kw)) == {
        ...     'type': 'Feature',
        ...     'id': 'gid',
        ...     'geometry': {
        ...         'type': 'Point',
        ...         'coordinates': (Decimal('12.2'), Decimal('22.0'))},
        ...     'properties': {'id': 'gid', 'p1': 'prop'}}
        True
    """
    black_list = {'type', kw.lon, kw.lat}

    for coordinates, row in subresults:
        properties = dict(x for x in row.items() if x[0] not in black_list)
        geometry = {'type': row['type'], 'coordinates': coordinates}

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

        yield feature


def gen_subresults(records, kw):
    """Helper function for converting record groups into a GeoJSON file like object.

     Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        kw (obj): `fntools.Objectify` instance with the following Attributes:
            key (str): GeoJSON Feature ID
            lon (str): longitude field name
            lat (str): latitude field name

    See also:
        `meza.convert.records2geojson`

    Yields:
        tuple(iter, dict): tuple of coordinates and row

    Examples:
        >>> kw = ft.Objectify({'key': 'id', 'lon': 'lon', 'lat': 'lat'})
        >>> record = {
        ...     'lon': Decimal('1.2'), 'lat': Decimal('22.0'), 'type': 'Point'}
        >>> next(gen_subresults([record], kw))[0]
        (Decimal('1.2'), Decimal('22.0'))
        >>> record = {'lon': 1.2, 'lat': 22.0, 'type': 'LineString'}
        >>> next(gen_subresults([record], kw))[0]
        [(1.2, 22.0)]
    """
    for id_, group in it.groupby(records, ft.def_itemgetter(kw.key)):
        first_row = next(group)
        type_ = first_row['type']
        sub_records = it.chain([first_row], group)

        if type_ == 'Point':
            for row in sub_records:
                yield ((row[kw.lon], row[kw.lat]), row)
        elif type_ == 'LineString':
            yield ([(r[kw.lon], r[kw.lat]) for r in sub_records], first_row)
        elif type_ == 'Polygon':
            groups = it.groupby(sub_records, itemgetter('pos'))
            polygon = [[(r[kw.lon], r[kw.lat]) for r in g[1]] for g in groups]
            yield (polygon, first_row)
        else:
            raise TypeError('Invalid type: %s' % type_)


def records2geojson(records, **kwargs):
    """Converts records into a GeoJSON file like object.

     Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io.read_geojson`.

        kwargs (dict): Keyword arguments.

    Kwargs:
        key (str): GeoJSON Feature ID (default: 'id').
        lon (int): longitude field name (default: 'lon').
        lat (int): latitude field name (default: 'lat').
        crs (str): coordinate reference system field name (default:
            'urn:ogc:def:crs:OGC:1.3:CRS84').
        indent (int): Number of spaces to indent (default: 2).
        sort_keys (bool): Sort rows by keys (default: True).
        ensure_ascii (bool): Sort response dict by keys (default: False).

    See also:
        `meza.convert.records2json`
        `meza.io.read_geojson`

    Returns:
        obj: io.StringIO instance

    Examples:
        >>> from json import loads

        >>> record = {
        ...     'id': 'gid', 'p1': 'prop', 'type': 'Point',
        ...     'lon': Decimal('12.2'), 'lat': Decimal('22.0')}
        ...
        >>> result = loads(next(records2geojson([record])))
        >>> result['type'] == 'FeatureCollection'
        True
        >>> result['bbox']
        [12.2, 22.0, 12.2, 22.0]
        >>> crs = 'urn:ogc:def:crs:OGC:1.3:CRS84'
        >>> result['crs'] == {'type': 'name', 'properties': {'name': crs}}
        True
        >>> features = result['features']
        >>> sorted(features[0].keys()) == [
        ...     'geometry', 'id', 'properties', 'type']
        True
        >>> features[0]['geometry'] == {
        ...     'type': 'Point', 'coordinates': [12.2, 22.0]}
        True
    """
    defaults = {
        'key': 'id', 'lon': 'lon', 'lat': 'lat', 'indent': 2, 'sort_keys': True,
        'crs': 'urn:ogc:def:crs:OGC:1.3:CRS84'}

    kw = ft.Objectify(kwargs, **defaults)
    crs = {'type': 'name', 'properties': {'name': kw.crs}}

    subresults = gen_subresults(records, kw)
    features = list(gen_features(subresults, kw))
    coords = [f['geometry']['coordinates'] for f in features]
    get_lon = lambda x: map(itemgetter(0), x)
    get_lat = lambda x: map(itemgetter(1), x)

    try:
        chained = (it.chain.from_iterable(map(get_lon, c)) for c in coords)
        lons = set(it.chain.from_iterable(chained))
    except TypeError:
        try:
            lons = set(it.chain.from_iterable(map(get_lon, coords)))
        except TypeError:
            # it's a point
            lons = set(get_lon(coords))
            lats = set(get_lat(coords))
        else:
            # it's a line
            lats = set(it.chain.from_iterable(map(get_lat, coords)))
    else:
        # it's a polygon
        chained = (it.chain.from_iterable(map(get_lat, c)) for c in coords)
        lats = set(it.chain.from_iterable(chained))

    if kw.sort_keys:
        crs = order_dict(crs, ['type', 'properties'])

    output = {
        'type': 'FeatureCollection',
        'bbox': [min(lons), min(lats), max(lons), max(lats)],
        'features': features,
        'crs': crs}

    if kw.sort_keys:
        output_order = ['type', 'bbox', 'features', 'crs']
        output = order_dict(output, output_order)

    dkwargs = ft.dfilter(kwargs, ['indent', 'sort_keys'], True)
    json = dumps(output, cls=ft.CustomEncoder, **dkwargs)
    return StringIO(str(json))
