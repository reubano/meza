#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.io
~~~~~~~

Provides methods for reading/writing/processing tabular formatted files

Examples:
    basic usage::

        from meza.io import read_csv

        csv_records = read_csv('path/to/file.csv')
        csv_header = next(csv_records).keys()
        record = next(csv_records)
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import itertools as it
import sys
import hashlib
import sqlite3
import yaml
import json
import xlrd
import pygogo as gogo

from os import path as p
from io import StringIO, TextIOBase, open
from datetime import time
from mmap import mmap
from collections import deque
from subprocess import check_output, check_call, Popen, PIPE, CalledProcessError
from http import client
from csv import Error as csvError
from functools import partial
from codecs import iterdecode

from builtins import *
from six.moves import zip_longest
from bs4 import BeautifulSoup, FeatureNotFound
from ijson import items
from xlrd.xldate import xldate_as_datetime as xl2dt
from chardet.universaldetector import UniversalDetector
from xlrd import (
    XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_NUMBER, XL_CELL_BOOLEAN,
    XL_CELL_ERROR)

from . import fntools as ft, process as pr, unicsv as csv, dbf, ENCODING

logger = gogo.Gogo(__name__, monolog=True).logger
PARENT_DIR = p.abspath(p.dirname(p.dirname(__file__)))
DATA_DIR = p.join(PARENT_DIR, 'data', 'test')
BOM = '\ufeff'


class IterStringIO(TextIOBase):
    """A lazy StringIO that writes a generator of strings and reads bytearrays.

    http://stackoverflow.com/a/32020108/408556
    """
    def __init__(self, iterable=None, bufsize=4096):
        """ IterStringIO constructor

        Args:
            iterable (Seq[str]): Iterable of strings
            bufsize (Int): Buffer size for seeking

        Examples:
            >>> StringIO(iter('Hello World')).read(5)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            TypeError:...
            >>> IterStringIO(iter('Hello World')).read(5)
            bytearray(b'Hello')
            >>> i = IterStringIO(iter('one\\ntwo\\n'))
            >>> list(next(i.lines)) == [b'o', b'n', b'e']
            True
        """
        iterable = iterable if iterable else []
        chained = self._chain(iterable)
        self.iter = self._encode(chained)
        self.bufsize = bufsize
        self.last = deque([], self.bufsize)
        self.pos = 0

    def __next__(self):
        return self._read(next(self.lines))

    def __iter__(self):
        return self

    @property
    def lines(self):
        # TODO: what about a csv with embedded newlines?
        newlines = {b'\n', b'\r', b'\r\n', '\n', '\r', '\r\n'}
        groups = it.groupby(self.iter, lambda s: s not in newlines)
        return (g for k, g in groups if k)

    def _encode(self, iterable):
        return (s.encode(ENCODING) for s in iterable)

    def _chain(self, iterable):
        iterable = iterable or []
        return it.chain.from_iterable(iterable)

    def _read(self, iterable, n=None, newline=True):
        byte = ft.byte(it.islice(iterable, n) if n else iterable)
        self.last.extend(byte)
        self.pos += n or len(byte)

        if newline:
            self.last.append('\n')

        return byte

    def write(self, iterable):
        chained = self._chain(iterable)
        self.iter = it.chain(self.iter, self._encode(chained))

    def read(self, n=None):
        return self._read(self.iter, n, False)

    def readline(self, n=None):
        return self._read(next(self.lines), n)

    def readlines(self):
        return map(self._read, self.lines)

    def seek(self, n):
        next_pos = self.pos + 1
        beg_buf = max([0, self.pos - self.bufsize])

        if n <= beg_buf:
            self.iter = it.chain(self.last, self.iter)
            self.last = deque([], self.bufsize)
        elif self.pos > n > beg_buf:
            extend = [self.last.pop() for _ in range(self.pos - n)]
            self.iter = it.chain(reversed(extend), self.iter)
        elif n == self.pos:
            pass
        elif n == next_pos:
            self.last.append(next(self.iter))
        elif n > next_pos:
            pos = n - self.pos
            [self.last.append(x) for x in it.islice(self.iter, 0, pos)]

        self.pos = beg_buf if n < beg_buf else n

    def tell(self):
        return self.pos


def patch_http_response_read(func):
    """Patches httplib to read poorly encoded chunked data.

    http://stackoverflow.com/a/14206036/408556
    """
    def inner(*args):
        try:
            return func(*args)
        except client.IncompleteRead as err:
            return err.partial

    return inner

client.HTTPResponse.read = patch_http_response_read(client.HTTPResponse.read)


def _remove_bom_from_dict(row, bom):
    for k, v in row.items():
        try:
            if all([k, v, bom in k, bom in v]):
                yield (k.lstrip(bom), v.lstrip(bom))
            elif v and bom in v:
                yield (k, v.lstrip(bom))
            elif k and bom in k:
                yield (k.lstrip(bom), v)
            else:
                yield (k, v)
        except TypeError:
            yield (k, v)


def _remove_bom_from_list(row, bom):
    for pos, col in enumerate(row):
        try:
            if not pos and bom in col:
                yield col.lstrip(bom)
            else:
                yield col
        except TypeError:
            yield col


def _remove_bom_from_scalar(row, bom):
    try:
        return row.lstrip(bom)
    except AttributeError:
        return row


def is_listlike(item):
    if hasattr(item, 'keys'):
        listlike = False
    else:
        listlike = {'append', 'next', '__reversed__'}.intersection(dir(item))

    return listlike


def remove_bom(row, bom):
    if is_listlike(row):
        row = list(_remove_bom_from_list(row, bom))
    else:
        try:
            row = dict(_remove_bom_from_dict(row, bom))
        except AttributeError:
            row = _remove_bom_from_scalar(row, bom)

    return row


def get_encoding(filepath):
    """
    Examples:
        >>> get_encoding(p.join(DATA_DIR, 'utf16_big.csv')) == 'UTF-16BE'
        True
    """
    with open(filepath, 'rb') as f:
        encoding = detect_encoding(f)['encoding']

    return encoding


def get_file_encoding(f, encoding=None):
    if encoding:
        new_f, new_encoding = f, encoding
    else:
        try:
            # See if we have bytes to avoid reopening the file
            new_encoding = detect_encoding(f)['encoding']
        except UnicodeDecodeError:
            msg = 'Incorrectly encoded file, reopening with bytes to detect'
            msg += ' encoding'
            logger.warning(msg)
            f.close()
            new_f = open(f.name, 'rb')
            new_encoding = detect_encoding(new_f)['encoding']
        else:
            new_f = f

    return new_f, new_encoding


def _read_any(f, reader, args, pos=0, recursed=False, **kwargs):
    try:
        for num, r in enumerate(reader(f, *args, **kwargs)):
            if num >= pos:
                yield r
                pos += 1
    except (UnicodeDecodeError, csvError, TypeError) as err:
        encoding = kwargs.get('encoding')

        if 'NoneType' in str(err):
            raise
        elif 'BufferedReader' in str(type(f)):
            # TODO: need to account for other file-like objects, this one is
            # what py3 urlopen returns
            logger.warning('File was opened in bytes mode')
        else:
            # Since the encoding could be wrong, set it None so that we can
            # detect the correct one.
            extra = (' (%s)' % encoding) if encoding else ''
            msg = 'Bytes or the wrong encoding%s was used to open file' % extra
            logger.warning(msg)
            encoding = None

        if recursed or not hasattr(f, 'seek'):
            logger.error('Unable to detect proper file encoding')
            raise

        f.seek(0)
        new_f, new_encoding = get_file_encoding(f, encoding)
        logger.debug('Decoding file with encoding: %s', encoding)

        try:
            decoded_f = iterdecode(new_f, new_encoding)

            for r in _read_any(decoded_f, reader, args, pos, True, **kwargs):
                yield r
        finally:
            new_f.close()


def read_any(filepath, reader, mode='r', *args, **kwargs):
    """Reads a file or filepath

    Args:
        filepath (str): The file path or file like object.
        reader (func): The processing function.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the reader.

    Kwargs:
        encoding (str): File encoding.

    See also:
        `meza.io.read_csv`
        `meza.io.read_fixed_fmt`
        `meza.io.read_json`
        `meza.io.read_geojson`
        `meza.io.write`
        `meza.io.hash_file`

    Yields:
        scalar: Result of applying the reader func to the file.

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.csv')
        >>> reader = lambda f, **kw: (l.strip().split(',') for l in f)
        >>> result = read_any(filepath, reader, 'r')
        >>> next(result) == [
        ...     'Some Date', 'Sparse Data', 'Some Value', 'Unicode Test', '']
        True
    """
    if hasattr(filepath, 'read'):
        encoding = kwargs.pop('encoding', ENCODING)

        if encoding:
            kwargs['encoding'] = encoding

        for r in _read_any(filepath, reader, args, **kwargs):
            yield remove_bom(r, BOM)
    else:
        encoding = kwargs.pop('encoding', None if 'b' in mode else ENCODING)

        with open(filepath, mode, encoding=encoding) as f:
            for r in _read_any(f, reader, args, **kwargs):
                yield remove_bom(r, BOM)


def _read_csv(f, header=None, has_header=True, **kwargs):
    """Helps read a csv file.

    Args:
        f (obj): The csv file like object.
        header (Seq[str]): Sequence of column names.
        has_header (bool): Whether or not file has a header.

    Kwargs:
        first_col (int): The first column (default: 0).

    Yields:
        dict: A csv record.

    See also:
        `meza.io.read_csv`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.csv')
        >>> with open(filepath, 'r', encoding='utf-8') as f:
        ...     sorted(next(_read_csv(f)).items()) == [
        ...         ('Some Date', '05/04/82'),
        ...         ('Some Value', '234'),
        ...         ('Sparse Data', 'Iñtërnâtiônàližætiøn'),
        ...         ('Unicode Test', 'Ādam')]
        True
    """
    first_col = kwargs.pop('first_col', 0)

    if header and has_header:
        next(f)
    elif not (header or has_header):
        raise ValueError('Either `header` or `has_header` must be specified.')

    header = (list(it.repeat(0, first_col)) + header) if first_col else header
    reader = csv.DictReader(f, header, **kwargs)

    # Remove empty keys
    records = (dict(x for x in r.items() if x[0]) for r in reader)

    # Remove empty rows
    for row in records:
        if any(v.strip() for v in row.values() if v):
            yield row


def read_mdb(filepath, table=None, **kwargs):
    """Reads an MS Access file

    Args:
        filepath (str): The mdb file path.
        table (str): The table to load (default: None, the first found table).
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).
        ignorecase (bool): Treat file name as case insensitive (default: true).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        TypeError: If unable to read the db file.

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.mdb')
        >>> records = read_mdb(filepath, sanitize=True)
        >>> expected = {
        ...     'surname': 'Aaron',
        ...     'forenames': 'William',
        ...     'freedom': '07/03/60 00:00:00',
        ...     'notes': 'Order of Court',
        ...     'surname_master_or_father': '',
        ...     'how_admitted': 'Redn.',
        ...     'id_no': '1',
        ...     'forenames_master_or_father': '',
        ...     'remarks': '',
        ...     'livery': '',
        ...     'date_of_order_of_court': '06/05/60 00:00:00',
        ...     'source_ref': 'MF 324'}
        >>> first_row = next(records)
        >>> (expected == first_row) if first_row else True
        True
    """
    args = ['mdb-tables', '-1', filepath]

    try:
        check_call(args)
    except OSError:
        logger.error(
            'You must install [mdbtools]'
            '(http://sourceforge.net/projects/mdbtools/) in order to use '
            'this function')
        yield
        return
    except CalledProcessError:
        raise TypeError('%s is not readable by mdbtools' % filepath)

    sanitize = kwargs.pop('sanitize', None)
    dedupe = kwargs.pop('dedupe', False)
    table = table or check_output(args).splitlines()[0]
    pkwargs = {'stdout': PIPE, 'bufsize': 1, 'universal_newlines': True}

    # http://stackoverflow.com/a/2813530/408556
    # http://stackoverflow.com/a/17698359/408556
    with Popen(['mdb-export', filepath, table], **pkwargs).stdout as pipe:
        first_line = StringIO(str(pipe.readline()))
        names = next(csv.reader(first_line, **kwargs))
        uscored = ft.underscorify(names) if sanitize else names
        header = list(ft.dedupe(uscored) if dedupe else uscored)

        for line in iter(pipe.readline, b''):
            next_line = StringIO(str(line))
            values = next(csv.reader(next_line, **kwargs))
            yield dict(zip(header, values))


def read_dbf(filepath, **kwargs):
    """Reads a dBase, Visual FoxPro, or FoxBase+ file

    Args:
        filepath (str): The dbf file path or file like object.
        kwargs (dict): Keyword arguments that are passed to the DBF reader.

    Kwargs:
        load (bool): Load all records into memory (default: false).
        encoding (str): Character encoding (default: None, parsed from
            the `language_driver`).

        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        ignorecase (bool): Treat file name as case insensitive (default: true).
        ignore_missing_memofile (bool): Suppress `MissingMemoFile` exceptions
            (default: False).

    Yields:
        OrderedDict: A row of data whose keys are the field names.

    Raises:
        MissingMemoFile: If unable to find the memo file.
        DBFNotFound: If unable to find the db file.

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.dbf')
        >>> records = read_dbf(filepath, sanitize=True)
        >>> next(records) == {
        ...      'awater10': 12416573076,
        ...      'aland10': 71546663636,
        ...      'intptlat10': '+47.2400052',
        ...      'lsad10': 'C2',
        ...      'cd111fp': '08',
        ...      'namelsad10': 'Congressional District 8',
        ...      'funcstat10': 'N',
        ...      'statefp10': '27',
        ...      'cdsessn': '111',
        ...      'mtfcc10': 'G5200',
        ...      'geoid10': '2708',
        ...      'intptlon10': '-092.9323194'}
        True
    """
    kwargs['lowernames'] = kwargs.pop('sanitize', None)
    return iter(dbf.DBF2(filepath, **kwargs))


def read_sqlite(filepath, table=None):
    """Reads a sqlite file.

    Args:
        filepath (str): The sqlite file path
        table (str): The table to load (default: None, the first found table).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `meza.io.read_any`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.sqlite')
        >>> records = read_sqlite(filepath)
        >>> next(records) == {
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'some_date': '05/04/82',
        ...     'some_value': 234,
        ...     'unicode_test': 'Ādam'}
        True
    """
    con = sqlite3.connect(filepath)
    con.row_factory = sqlite3.Row
    c = con.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")

    t = table or c.fetchone()[0]
    c.execute('SELECT * FROM %s' % t)
    return map(dict, c)


def read_csv(filepath, mode='r', **kwargs):
    """Reads a csv file.

    Args:
        filepath (str): The csv file path or file like object.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        delimiter (str): Field delimiter (default: ',').
        quotechar (str): Quote character (default: '"').
        encoding (str): File encoding.
        has_header (bool): Has header row (default: True).
        custom_header (List[str]): Custom header names (default: None).
        first_row (int): First row (zero based, default: 0).
        first_col (int): First column (zero based, default: 0).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `meza.io.read_any`
        `meza.io._read_csv`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.csv')
        >>> records = read_csv(filepath, sanitize=True)
        >>> next(records) == {
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'some_date': '05/04/82',
        ...     'some_value': '234',
        ...     'unicode_test': 'Ādam'}
        True
    """
    def reader(f, **kwargs):
        first_row = kwargs.pop('first_row', 0)
        first_col = kwargs.pop('first_col', 0)
        sanitize = kwargs.pop('sanitize', False)
        dedupe = kwargs.pop('dedupe', False)
        has_header = kwargs.pop('has_header', True)
        custom_header = kwargs.pop('custom_header', None)

        # position file pointer at the first row
        list(it.islice(f, first_row))
        first_line = StringIO(str(next(f)))
        names = next(csv.reader(first_line, **kwargs))

        if has_header or custom_header:
            names = custom_header if custom_header else names
            stripped = (name for name in names if name.strip())
            uscored = ft.underscorify(stripped) if sanitize else stripped
            header = list(ft.dedupe(uscored) if dedupe else uscored)

        if not has_header:
            # reposition file pointer at the first row
            try:
                f.seek(0)
            except AttributeError:
                msg = 'Non seekable files must have either a specified or'
                msg += 'custom header.'
                logger.error(msg)
                raise

            list(it.islice(f, first_row))

        if not (has_header or custom_header):
            header = ['column_%i' % (n + 1) for n in range(len(names))]

        return _read_csv(f, header, False, first_col=first_col, **kwargs)

    return read_any(filepath, reader, mode, **kwargs)


def read_tsv(filepath, mode='r', **kwargs):
    """Reads a csv file.

    Args:
        filepath (str): The tsv file path or file like object.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        quotechar (str): Quote character (default: '"').
        encoding (str): File encoding.
        has_header (bool): Has header row (default: True).
        first_row (int): First row (zero based, default: 0).
        first_col (int): First column (zero based, default: 0).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `meza.io.read_any`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.tsv')
        >>> records = read_tsv(filepath, sanitize=True)
        >>> next(records) == {
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'some_date': '05/04/82',
        ...     'some_value': '234',
        ...     'unicode_test': 'Ādam'}
        True
    """
    return read_csv(filepath, mode, dialect='excel-tab', **kwargs)


def read_fixed_fmt(filepath, widths=None, mode='r', **kwargs):
    """Reads a fixed-width csv file.

    Args:
        filepath (str): The fixed width formatted file path or file like object.
        widths (List[int]): The zero-based 'start' position of each column.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        has_header (bool): Has header row (default: False).
        first_row (int): First row (zero based, default: 0).
        first_col (int): First column (zero based, default: 0).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `meza.io.read_any`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'fixed.txt')
        >>> widths = [0, 18, 29, 33, 38, 50]
        >>> records = read_fixed_fmt(filepath, widths)
        >>> next(records) == {
        ...     'column_1': 'Chicago Reader',
        ...     'column_2': '1971-01-01',
        ...     'column_3': '40',
        ...     'column_4': 'True',
        ...     'column_5': '1.0',
        ...     'column_6': '04:14:001971-01-01T04:14:00'}
        True
    """
    def reader(f, **kwargs):
        sanitize = kwargs.get('sanitize')
        dedupe = kwargs.pop('dedupe', False)
        has_header = kwargs.get('has_header')
        first_row = kwargs.get('first_row', 0)
        schema = tuple(zip_longest(widths, widths[1:]))
        [next(f) for _ in range(first_row)]

        if has_header:
            line = next(f)
            names = (_f for _f in (line[s:e].strip() for s, e in schema) if _f)
            uscored = ft.underscorify(names) if sanitize else names
            header = list(ft.dedupe(uscored) if dedupe else uscored)
        else:
            header = ['column_%i' % (n + 1) for n in range(len(widths))]

        zipped = zip(header, schema)

        get_row = lambda line: {k: line[v[0]:v[1]].strip() for k, v in zipped}
        return map(get_row, f)

    return read_any(filepath, reader, mode, **kwargs)


def sanitize_sheet(sheet, mode, first_col=0, **kwargs):
    """Formats content from xls/xslx files as strings according to its cell
    type.

    Args:
        sheet (obj): `xlrd` sheet object.
        mode (str): `xlrd` workbook datemode property.
        kwargs (dict): Keyword arguments
        first_col (int): The first column (default: 0).

    Kwargs:
        date_format (str): `strftime()` date format.
        dt_format (str): `strftime()` datetime format.
        time_format (str): `strftime()` time format.

    Yields:
        Tuple[int, str]: A tuple of (row_number, value).

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.xls')
        >>> book = xlrd.open_workbook(filepath)
        >>> sheet = book.sheet_by_index(0)
        >>> sheet.row_values(1) == [
        ...     30075.0, 'Iñtërnâtiônàližætiøn', 234.0, 'Ādam', ' ']
        True
        >>> sanitized = sanitize_sheet(sheet, book.datemode)
        >>> [v for i, v in sanitized if i == 1] == [
        ...     '1982-05-04', 'Iñtërnâtiônàližætiøn', '234.0', 'Ādam', ' ']
        True
    """
    date_format = kwargs.get('date_format', '%Y-%m-%d')
    dt_format = kwargs.get('dt_format', '%Y-%m-%d %H:%M:%S')
    time_format = kwargs.get('time_format', '%H:%M:%S')

    def time_func(value):
        args = xlrd.xldate_as_tuple(value, mode)[3:]
        return time(*args).strftime(time_format)

    switch = {
        XL_CELL_DATE: lambda v: xl2dt(v, mode).strftime(date_format),
        'datetime': lambda v: xl2dt(v, mode).strftime(dt_format),
        'time': time_func,
        XL_CELL_EMPTY: lambda v: '',
        XL_CELL_NUMBER: lambda v: str(v),
        XL_CELL_BOOLEAN: lambda v: str(bool(v)),
        XL_CELL_ERROR: lambda v: xlrd.error_text_from_code[v],
    }

    for i in range(sheet.nrows):
        types = sheet.row_types(i)[first_col:]
        values = sheet.row_values(i)[first_col:]

        for type_, value in zip(types, values):
            if type_ == XL_CELL_DATE and value < 1:
                type_ = 'time'
            elif type_ == XL_CELL_DATE and not value.is_integer:
                type_ = 'datetime'

            yield (i, switch.get(type_, lambda v: v)(value))


def read_xls(filepath, **kwargs):
    """Reads an xls/xlsx file.

    Args:
        filepath (str): The xls/xlsx file path, file, or SpooledTemporaryFile.
        kwargs (dict): Keyword arguments that are passed to the xls reader.

    Kwargs:
        sheet (int): Zero indexed sheet to open (default: 0)
        has_header (bool): Has header row (default: True).
        first_row (int): First row (zero based, default: 0).
        first_col (int): First column (zero based, default: 0).
        date_format (str): Date format passed to `strftime()` (default:
            '%Y-%m-%d', i.e, 'YYYY-MM-DD').

        encoding (str): File encoding. By default, the encoding is derived from
            the file's `CODEPAGE` number, e.g., 1252 translates to `cp1252`.

        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

        on_demand (bool): open_workbook() loads global data and returns without
            releasing resources. At this stage, the only information available
            about sheets is Book.nsheets and Book.sheet_names() (default:
            False).

        pad_rows (bool): Add empty cells so that all rows have the number of
            columns `Sheet.ncols` (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.xls')
        >>> records = read_xls(filepath, sanitize=True)
        >>> next(records) == {
        ...     'some_value': '234.0',
        ...     'some_date': '1982-05-04',
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'unicode_test': 'Ādam'}
        True
    """
    has_header = kwargs.get('has_header', True)
    first_row = kwargs.get('first_row', 0)
    first_col = kwargs.get('first_col', 0)
    sanitize = kwargs.get('sanitize')
    dedupe = kwargs.pop('dedupe', False)

    xlrd_kwargs = {
        'on_demand': kwargs.get('on_demand'),
        'ragged_rows': not kwargs.get('pad_rows'),
        'encoding_override': kwargs.get('encoding', True)
    }

    try:
        mm = mmap(filepath.fileno(), 0)
        book = xlrd.open_workbook(file_contents=mm, **xlrd_kwargs)
    except AttributeError:
        book = xlrd.open_workbook(filepath, **xlrd_kwargs)

    sheet = book.sheet_by_index(kwargs.pop('sheet', 0))

    # Get header row and remove empty columns
    names = sheet.row_values(first_row)[first_col:]

    if has_header:
        stripped = (name for name in names if name.strip())
        uscored = ft.underscorify(stripped) if sanitize else stripped
        header = list(ft.dedupe(uscored) if dedupe else uscored)
    else:
        header = ['column_%i' % (n + 1) for n in range(len(names))]

    # Convert to strings
    sanitized = sanitize_sheet(sheet, book.datemode, **kwargs)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        if has_header and key == first_row:
            continue

        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(header, values))


def read_json(filepath, mode='r', path='item', newline=False):
    """Reads a json file (both regular and newline-delimited)

    Args:
        filepath (str): The json file path or file like object.
        mode (Optional[str]): The file open mode (default: 'r').
        path (Optional[str]): Path to the content you wish to read
            (default: 'item', i.e., the root list). Note: `path` must refer to
            a list.

        newline (Optional[bool]): Interpret file as newline-delimited
            (default: False).

    Kwargs:
        encoding (str): File encoding.

    Returns:
        Iterable: The parsed records

    See also:
        `meza.io.read_any`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.json')
        >>> records = read_json(filepath)
        >>> next(records) == {
        ...     'text': 'Chicago Reader',
        ...     'float': 1,
        ...     'datetime': '1971-01-01T04:14:00',
        ...     'boolean': True,
        ...     'time': '04:14:00',
        ...     'date': '1971-01-01',
        ...     'integer': 40}
        True
    """
    reader = lambda f, **kw: map(json.loads, f) if newline else items(f, path)
    return read_any(filepath, reader, mode)


def get_point(coords, lat_first):
    if lat_first:
        point = (coords[1], coords[0])
    else:
        point = (coords[0], coords[1])

    return point


def gen_records(type_, record, coords, properties, **kwargs):
    lat_first = kwargs.get('lat_first')

    if type_ == 'Point':
        record['lon'], record['lat'] = get_point(coords, lat_first)
        yield pr.merge([record, properties])
    elif type_ == 'LineString':
        for point in coords:
            record['lon'], record['lat'] = get_point(point, lat_first)
            yield pr.merge([record, properties])
    elif type_ == 'Polygon':
        for pos, poly in enumerate(coords):
            for point in poly:
                record['lon'], record['lat'] = get_point(point, lat_first)
                record['pos'] = pos
                yield pr.merge([record, properties])
    else:
        raise TypeError('Invalid geometry type {}.'.format(type_))


def read_geojson(filepath, key='id', mode='r', **kwargs):
    """Reads a geojson file

    Args:
        filepath (str): The geojson file path or file like object.
        key (str): GeoJSON Feature ID (default: 'id').
        mode (Optional[str]): The file open mode (default: 'r').

    Kwargs:
        lat_first (bool): Latitude listed as first coordinate (default: False).

        encoding (str): File encoding.

    Returns:
        Iterable: The parsed records

    Raise:
        TypeError if no features list or invalid geometry type.

    See also:
        `meza.io.read_any`
        `meza.convert.records2geojson`

    Examples:
        >>> from decimal import Decimal

        >>> filepath = p.join(DATA_DIR, 'test.geojson')
        >>> records = read_geojson(filepath)
        >>> next(records) == {
        ...     'id': 6635402,
        ...     'iso3': 'ABW',
        ...     'bed_prv_pr': Decimal('0.003'),
        ...     'ic_mhg_cr': Decimal('0.0246'),
        ...     'bed_prv_cr': 0,
        ...     'type': 'Point',
        ...     'lon': Decimal('-70.0624999987871'),
        ...     'lat': Decimal('12.637499976568533')}
        True
    """
    def reader(f, **kwargs):
        try:
            features = items(f, 'features.item')
        except KeyError:
            raise TypeError('Only GeoJSON with features are supported.')
        else:
            for feature in features:
                type_ = feature['geometry']['type']
                properties = feature.get('properties') or {}
                coords = feature['geometry']['coordinates']
                record = {
                    'id': feature.get(key, properties.get(key)),
                    'type': feature['geometry']['type']}

                args = (record, coords, properties)
                for r in gen_records(type_, *args, **kwargs):
                    yield r

    return read_any(filepath, reader, mode, **kwargs)


def read_yaml(filepath, mode='r', **kwargs):
    """Reads a YAML file

    TODO: convert to a streaming parser

    Args:
        filepath (str): The yaml file path or file like object.
        mode (Optional[str]): The file open mode (default: 'r').

    Kwargs:
        encoding (str): File encoding.

    Returns:
        Iterable: The parsed records

    See also:
        `meza.io.read_any`

    Examples:
        >>> from datetime import date, datetime as dt

        >>> filepath = p.join(DATA_DIR, 'test.yml')
        >>> records = read_yaml(filepath)
        >>> next(records) == {
        ...     'text': 'Chicago Reader',
        ...     'float': 1.0,
        ...     'datetime': dt(1971, 1, 1, 4, 14),
        ...     'boolean': True,
        ...     'time': '04:14:00',
        ...     'date': date(1971, 1, 1),
        ...     'integer': 40}
        True
    """
    return read_any(filepath, yaml.load, mode, **kwargs)


def read_html(filepath, table=0, mode='r', **kwargs):
    """Reads tables from an html file

    TODO: convert to lxml.etree.iterparse
    http://lxml.de/parsing.html#iterparse-and-iterwalk

    Args:
        filepath (str): The html file path or file like object.
        table (int): Zero indexed table to open (default: 0)
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments

    Kwargs:
        encoding (str): File encoding.

        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Returns:
        Iterable: The parsed records

    See also:
        `meza.io.read_any`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.html')
        >>> records = read_html(filepath, sanitize=True)
        >>> next(records) == {
        ...     '': 'Mediterranean',
        ...     'january': '82',
        ...     'february': '346',
        ...     'march': '61',
        ...     'april': '1,244',
        ...     'may': '95',
        ...     'june': '\xa010',
        ...     'july': '230',
        ...     'august': '684',
        ...     'september': '268',
        ...     'october': '432',
        ...     'november': '105',
        ...     'december': '203',
        ...     'total_to_date': '3,760'}
        True
    """
    def reader(f, **kwargs):
        try:
            soup = BeautifulSoup(f, 'lxml-xml')
        except FeatureNotFound:
            soup = BeautifulSoup(f, 'html.parser')

        sanitize = kwargs.get('sanitize')
        dedupe = kwargs.get('dedupe')
        tbl = soup.find_all('table')[table]
        rows = iter(tbl.find_all('tr'))
        first_row = next(rows)
        ths = first_row.find_all('th')
        names = (next(th.children).string for th in ths)
        uscored = ft.underscorify(names) if sanitize else names
        header = list(ft.dedupe(uscored) if dedupe else uscored)

        for tr in rows:
            row = (td.string for td in tr.find_all('td'))
            yield dict(zip(header, row))
            # yield {'r': list(row)}

    return read_any(filepath, reader, mode, **kwargs)


def write(filepath, content, mode='wb+', **kwargs):
    """Writes content to a file path or file like object.
    # TODO: add encoding kwarg

    Args:
        filepath (str): The file path or file like object to write to.
        content (obj): File like object or `requests` iterable response.
        mode (Optional[str]): The file open mode (default: 'wb+').
        kwargs: Keyword arguments.

    Kwargs:
        encoding (str): The file encoding.
        chunksize (Optional[int]): Number of bytes to write at a time (default:
            None, i.e., all).
        length (Optional[int]): Length of content (default: 0).
        bar_len (Optional[int]): Length of progress bar (default: 50).

    Returns:
        int: bytes written

    See also:
        `meza.io.read_any`

    Examples:
        >>> from tempfile import TemporaryFile
        >>> write(TemporaryFile(), StringIO('Hello World'))
        11
        >>> write(StringIO(), StringIO('Hello World'))
        11
    """
    def _write(f, content, **kwargs):
        chunksize = kwargs.get('chunksize')
        length = int(kwargs.get('length') or 0)
        bar_len = kwargs.get('bar_len', 50)
        encoding = kwargs.get('encoding', ENCODING)
        progress = 0

        for c in ft.chunk(content, chunksize):
            text = ft.byte(c) if hasattr(c, 'sort') else c

            try:
                f.write(text)
            except (TypeError, UnicodeEncodeError):
                text = bytes(text, encoding)
                f.write(text)

            progress += chunksize or len(text)

            if length:
                bars = min(int(bar_len * progress / length), bar_len)
                print('\r[%s%s]' % ('=' * bars, ' ' * (bar_len - bars)))
                sys.stdout.flush()

        yield progress

    return sum(read_any(filepath, _write, mode, content, **kwargs))


def hash_file(filepath, algo='sha1', chunksize=0, verbose=False):
    """Hashes a file path or file like object.
    http://stackoverflow.com/a/1131255/408556

    Args:
        filepath (str): The file path or file like object to hash.
        algo (str): The hashlib hashing algorithm to use (default: sha1).

        chunksize (Optional[int]): Number of bytes to write at a time
            (default: 0, i.e., all).

        verbose (Optional[bool]): Print debug statements (default: False).

    Returns:
        str: File hash.

    See also:
        `meza.io.read_any`
        `meza.process.hash`

    Examples:
        >>> from tempfile import TemporaryFile
        >>> resp = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
        >>> hash_file(TemporaryFile()) == resp
        True
    """
    def reader(f, hasher, **kwargs):
        if chunksize:
            while True:
                data = f.read(chunksize)
                if not data:
                    break

                hasher.update(data)
        else:
            hasher.update(f.read())

        yield hasher.hexdigest()

    args = [getattr(hashlib, algo)()]
    file_hash = next(read_any(filepath, reader, 'rb', *args))

    if verbose:
        print('File %s hash is %s.' % (filepath, file_hash))

    return file_hash


def detect_encoding(f, verbose=False):
    """Detects a file's encoding.

    Args:
        f (obj): The file like object to detect.
        verbose (Optional[bool]): The file open mode (default: False).
        mode (Optional[str]): The file open mode (default: 'r').

    Returns:
        dict: The encoding result

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.csv')
        >>>
        >>> with open(filepath, 'rb') as f:
        ...     result = detect_encoding(f)
        ...     result == {'confidence': 0.99, 'encoding': 'utf-8'}
        True
    """
    pos = f.tell()
    detector = UniversalDetector()

    for line in f:
        detector.feed(line)

        if detector.done:
            break

    detector.close()
    f.seek(pos)

    if verbose:
        print('result', detector.result)

    return detector.result


def get_reader(extension):
    """Gets the appropriate reader for a given file extension.

    Args:
        extension (str): The file extension.

    Returns:
        func: The file reading function

    See also:
        `meza.io.read`

    Raises:
        TypeError: If unable to find a suitable reader.

    Examples:
        >>> get_reader('xls')  # doctest: +ELLIPSIS
        <function read_xls at 0x...>
    """
    switch = {
        'csv': read_csv,
        'xls': read_xls,
        'xlsx': read_xls,
        'mdb': read_mdb,
        'json': read_json,
        'geojson': read_geojson,
        'geojson.json': read_geojson,
        'sqlite': read_sqlite,
        'dbf': read_dbf,
        'tsv': read_tsv,
        'yaml': read_yaml,
        'yml': read_yaml,
        'html': read_html,
        'fixed': read_fixed_fmt,
    }

    try:
        return switch[extension.lstrip('.').lower()]
    except IndexError:
        raise TypeError('Reader for extension `%s` not found!' % extension)


def read(filepath, ext=None, **kwargs):
    """Reads any supported file format.

    Args:
        filepath (str): The file path or file like object.

        ext (str): The file extension.

    Returns:
        Iterable: The parsed records

    See also:
        `meza.io.get_reader`
        `meza.io.join`

    Examples:
        >>> filepath = p.join(DATA_DIR, 'test.xls')
        >>> next(read(filepath, sanitize=True)) == {
        ...     'some_value': '234.0',
        ...     'some_date': '1982-05-04',
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'unicode_test': 'Ādam'}
        True
        >>> filepath = p.join(DATA_DIR, 'test.csv')
        >>> next(read(filepath, sanitize=True)) == {
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'some_date': '05/04/82',
        ...     'some_value': '234',
        ...     'unicode_test': 'Ādam'}
        True
    """
    ext = ext or p.splitext(filepath)[1]
    return get_reader(ext)(filepath, **kwargs)


def join(*filepaths, **kwargs):
    """Reads multiple filepaths and yields all the resulting records.

    Args:
        filepaths (iter[str]): Iterator of filepaths or file like objects.

        kwargs (dict): keyword args passed to the individual readers.

    Kwargs:
        ext (str): The file extension.

    Yields:
        dict: A parsed record

    See also:
        `meza.io.read`

    Examples:
        >>> fs = [p.join(DATA_DIR, 'test.xls'), p.join(DATA_DIR, 'test.csv')]
        >>> next(join(*fs, sanitize=True)) == {
        ...     'some_value': '234.0',
        ...     'some_date': '1982-05-04',
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'unicode_test': 'Ādam'}
        True
    """
    reader = partial(read, **kwargs)
    return it.chain.from_iterable(map(reader, filepaths))
