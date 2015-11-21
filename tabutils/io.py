#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.io
~~~~~~~~~~~

Provides methods for reading/writing/processing tabular formatted files

Examples:
    basic usage::

        from tabutils.io import read_csv

        csv_records = read_csv('path/to/file.csv')
        csv_header = csv_records.next().keys()
        record = csv_records.next()
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import xlrd
import itertools as it
import unicodecsv as csv
import httplib
import sys
import hashlib

from StringIO import StringIO
from io import TextIOBase
from json import loads, dumps
from mmap import mmap
from collections import deque
from subprocess import check_output, check_call, Popen, PIPE, CalledProcessError

from ijson import items
from xlrd.xldate import xldate_as_datetime as xl2dt
from chardet.universaldetector import UniversalDetector
from xlrd import (
    XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_NUMBER, XL_CELL_BOOLEAN,
    XL_CELL_ERROR)

from . import fntools as ft, process as pr, dbf, ENCODING


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
            >>> iter_content = iter('Hello World')
            >>> StringIO(iter_content).read(5)
            '<iter'
            >>> iter_sio = IterStringIO(iter_content)
            >>> iter_sio.read(5)
            bytearray(b'Hello')
            >>> iter_sio.write(iter('ly person'))
            >>> iter_sio.read(8)
            bytearray(b' Worldly')
            >>> iter_sio.write(': Iñtërnâtiônàližætiøn')
            >>> iter_sio.read() == bytearray(b' person: Iñtërnâtiônàližætiøn')
            True
            >>> content = 'line one\\nline two\\nline three\\n'
            >>> iter_sio = IterStringIO(content)
            >>> iter_sio.readline()
            bytearray(b'line one')
            >>> iter_sio.next()
            bytearray(b'line two')
            >>> iter_sio.seek(0)
            >>> iter_sio.next()
            bytearray(b'line one')
            >>> list(IterStringIO(content).readlines())
            [bytearray(b'line one'), bytearray(b'line two'), \
bytearray(b'line three')]
        """
        iterable = iterable or []
        chained = self._chain(iterable)
        self.iter = self._encode(chained)
        self.last = deque('', bufsize)

    def __next__(self):
        return self._read(self.lines.next())

    @property
    def lines(self):
        # TODO: what about a csv with embedded newlines?
        newlines = {'\n', '\r', '\r\n'}

        for k, g in it.groupby(self.iter, lambda s: s not in newlines):
            if k:
                yield g

    def _encode(self, iterable):
        return (s.encode(ENCODING) for s in iterable)

    def _chain(self, iterable):
        iterable = iterable or []
        return it.chain.from_iterable(it.ifilter(None, iterable))

    def _read(self, iterable, n=None):
        # TODO: what about cases when a whole line isn't read?
        byte = ft.byte(it.islice(iterable, n) if n else iterable)
        self.last.extend(byte)
        self.last.append('\n')
        return byte

    def write(self, iterable):
        chained = self._chain(iterable)
        self.iter = self._chain([self.iter, self._encode(chained)])

    def read(self, n=None):
        return self._read(self.iter, n)

    def readline(self, n=None):
        return self._read(self.lines.next(), n)

    def readlines(self):
        return it.imap(self._read, self.lines)

    def seek(self, n):
        self.iter = it.chain.from_iterable([list(self.last)[n:], self.iter])


def patch_http_response_read(func):
    """Patches httplib to read poorly encoded chunked data.

    http://stackoverflow.com/a/14206036/408556
    """
    def inner(*args):
        try:
            return func(*args)
        except httplib.IncompleteRead, e:
            return e.partial

    return inner

httplib.HTTPResponse.read = patch_http_response_read(httplib.HTTPResponse.read)


def read_any(filepath, reader, mode='rU', *args, **kwargs):
    """Reads a file or filepath

    Args:
        filepath (str): The file path or file like object.
        reader (func): The processing function.
        mode (Optional[str]): The file open mode (default: 'rU').
        kwargs (dict): Keyword arguments that are passed to the reader.

    See also:
        `io.read_csv`
        `io.read_fixed_csv`
        `io.read_json`
        `io.read_geojson`
        `io.write`
        `io.hash_file`

    Yields:
        scalar: Result of applying the reader func to the file.

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> reader = lambda f: (l.strip().split(',') for l in f)
        >>> read_any(filepath, reader, 'rU').next()
        [u'Some Date', u'Sparse Data', u'Some Value', u'Unicode Test', u'']
    """
    pos = 0

    try:
        if hasattr(filepath, 'read'):
            f = filepath
            for r in reader(f, *args, **kwargs):
                yield r
                pos += 1
        else:
            with open(filepath, mode) as f:
                for r in reader(f, *args, **kwargs):
                    yield r
                    pos += 1
    except UnicodeDecodeError:
        if hasattr(filepath, 'read'):
            f.seek(0)
            kwargs['encoding'] = detect_encoding(f)['encoding']

            for num, r in enumerate(reader(f, *args, **kwargs)):
                if num >= pos:
                    yield r
        else:
            with open(filepath, mode) as f:
                kwargs['encoding'] = detect_encoding(f)['encoding']

                for num, r in enumerate(reader(f, *args, **kwargs)):
                    if num >= pos:
                        yield r
                    yield r


def _read_csv(f, encoding, header=None, has_header=True):
    """Helps read a csv file.

    Args:
        f (obj): The csv file like object.
        encoding (str): File encoding.

    Kwargs:
        header (Seq[str]): Sequence of column names.

    Yields:
        dict: A csv record.

    See also:
        `io.read_csv`

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> header = ['some_date', 'sparse_data', 'some_value', 'unicode_test']
        >>> with open(filepath, 'rU') as f:
        ...     sorted(_read_csv(f, 'utf-8').next().items()) == [
        ...         (u'Some Date', u'05/04/82'),
        ...         (u'Some Value', u'234'),
        ...         (u'Sparse Data', u'Iñtërnâtiônàližætiøn'),
        ...         (u'Unicode Test', u'Ādam')]
        True
        >>> with open(filepath, 'rU') as f:
        ...     sorted(_read_csv(f, 'utf-8', header).next().items()) == [
        ...         (u'some_date', u'05/04/82'),
        ...         (u'some_value', u'234'),
        ...         (u'sparse_data', u'Iñtërnâtiônàližætiøn'),
        ...         (u'unicode_test', u'Ādam')]
        True
    """
    if header and has_header:
        f.next()
        reader = csv.DictReader(f, header, encoding=encoding)
    elif header:
        reader = csv.DictReader(f, header, encoding=encoding)
    elif has_header:
        reader = csv.DictReader(f, encoding=encoding)
    else:
        raise ValueError('Either `header` or `has_header` must be specified.')

    # Remove `None` keys
    records = (dict(it.ifilter(lambda x: x[0], r.iteritems())) for r in reader)

    # Remove empty rows
    for row in records:
        if any(v.strip() for v in row.values() if v):
            yield row


def read_mdb(filepath, table=None, **kwargs):
    """Reads an MS Access file

    Args:
        filepath (str): The mdb file path.
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        table (str): The table to load (default: None, the first found table).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).
        ignorecase (bool): Treat file name as case insensitive (default: true).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        OSError: If unable to find mdbtools.
        TypeError: If unable to read the db file.

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.mdb')
        >>> records = read_mdb(filepath, sanitize=True)
        >>> records.next() == {
        ...     u'surname': u'Aaron',
        ...     u'forenames': u'William',
        ...     u'freedom': u'07/03/60 00:00:00',
        ...     u'notes': u'Order of Court',
        ...     u'surname_master_or_father': u'',
        ...     u'how_admitted': u'Redn.',
        ...     u'id_no': u'1',
        ...     u'forenames_master_or_father': u'',
        ...     u'remarks': u'',
        ...     u'livery': u'',
        ...     u'date_of_order_of_court': u'06/05/60 00:00:00',
        ...     u'source_ref': u'MF 324'}
        ...
        True
    """
    args = ['mdb-tables', '-1', filepath]

    try:
        check_call(args)
    except OSError:
        raise OSError(
            'You must install [mdbtools]'
            '(http://sourceforge.net/projects/mdbtools/) in order to use '
            'this function')
    except CalledProcessError:
        raise TypeError('%s is not readable by mdbtools' % filepath)

    sanitize = kwargs.pop('sanitize', None)
    dedupe = kwargs.pop('dedupe', False)
    table = table or check_output(args).splitlines()[0]
    pkwargs = {'stdout': PIPE, 'bufsize': 1, 'universal_newlines': True}

    # http://stackoverflow.com/a/2813530/408556
    # http://stackoverflow.com/a/17698359/408556
    with Popen(['mdb-export', filepath, table], **pkwargs).stdout as pipe:
        first_line = pipe.readline()
        names = csv.reader(StringIO(first_line), **kwargs).next()
        uscored = list(ft.underscorify(names)) if sanitize else names
        header = list(ft.dedupe(uscored)) if dedupe else uscored

        for line in iter(pipe.readline, b''):
            values = csv.reader(StringIO(line), **kwargs).next()
            yield dict(zip(header, values))


def read_dbf(filepath, **kwargs):
    """Reads a dBase, Visual FoxPro, or FoxBase+ file

    Args:
        filepath (str): The dbf file path or file like object.
        kwargs (dict): Keyword arguments that are passed to the DBF reader.

    Kwargs:
        load (bool): Load all records into memory (default: false).
        encoding (bool): Character encoding (default: None, parsed from
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
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.dbf')
        >>> records = read_dbf(filepath, sanitize=True)
        >>> records.next() == {
        ...      u'awater10': 12416573076,
        ...      u'aland10': 71546663636,
        ...      u'intptlat10': u'+47.2400052',
        ...      u'lsad10': u'C2',
        ...      u'cd111fp': u'08',
        ...      u'namelsad10': u'Congressional District 8',
        ...      u'funcstat10': u'N',
        ...      u'statefp10': u'27',
        ...      u'cdsessn': u'111',
        ...      u'mtfcc10': u'G5200',
        ...      u'geoid10': u'2708',
        ...      u'intptlon10': u'-092.9323194'}
        ...
        True
        >>> with open(filepath, 'rb') as f:
        ...     records = read_dbf(f, sanitize=True)
        ...     records.next() == {
        ...         u'awater10': 12416573076,
        ...         u'aland10': 71546663636,
        ...         u'intptlat10': u'+47.2400052',
        ...         u'lsad10': u'C2',
        ...         u'cd111fp': u'08',
        ...         u'namelsad10': u'Congressional District 8',
        ...         u'funcstat10': u'N',
        ...         u'statefp10': u'27',
        ...         u'cdsessn': u'111',
        ...         u'mtfcc10': u'G5200',
        ...         u'geoid10': u'2708',
        ...         u'intptlon10': u'-092.9323194'}
        ...
        True
    """
    kwargs['lowernames'] = kwargs.pop('sanitize', None)

    for record in dbf.DBF2(filepath, **kwargs):
        yield record


def read_csv(filepath, mode='rU', **kwargs):
    """Reads a csv file.

    Args:
        filepath (str): The csv file path or file like object.
        mode (Optional[str]): The file open mode (default: 'rU').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        delimiter (str): Field delimiter (default: ',').
        quotechar (str): Quote character (default: '"').
        encoding (str): File encoding.
        has_header (bool): Has header row (default: True).
        first_row (int): First row (zero based, default: 0).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `io.read_any`
        `io._read_csv`

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> records = read_csv(filepath, sanitize=True)
        >>> records.next() == {
        ...     u'sparse_data': u'Iñtërnâtiônàližætiøn',
        ...     u'some_date': u'05/04/82',
        ...     u'some_value': u'234',
        ...     u'unicode_test': u'Ādam'}
        ...
        True
        >>> filepath = p.join(parent_dir, 'data', 'test', 'no_header_row.csv')
        >>> records = read_csv(filepath, has_header=False)
        >>> records.next() == {
        ...     u'column_1': u'1',
        ...     u'column_2': u'2',
        ...     u'column_3': u'3'}
        ...
        True
    """
    def reader(f, first_row=0, **kwargs):
        encoding = kwargs.pop('encoding', False) or ENCODING
        sanitize = kwargs.pop('sanitize', False)
        dedupe = kwargs.pop('dedupe', False)
        has_header = kwargs.pop('has_header', True)
        [f.next() for _ in xrange(first_row)]
        pos = f.tell()
        names = csv.reader(f, encoding=encoding, **kwargs).next()

        if has_header:
            stripped = [name for name in names if name.strip()]
            uscored = list(ft.underscorify(stripped)) if sanitize else stripped
            header = list(ft.dedupe(uscored)) if dedupe else uscored
        else:
            f.seek(pos)
            header = ['column_%i' % (n + 1) for n in xrange(len(names))]

        return _read_csv(f, encoding, header, False)

    return read_any(filepath, reader, mode, **kwargs)


def read_fixed_csv(filepath, widths, mode='rU', **kwargs):
    """Reads a fixed-width csv file.

    Args:
        filepath (str): The fixed width csv file path or file like object.
        widths (List[int]): The zero-based 'start' position of each column.
        mode (Optional[str]): The file open mode (default: 'rU').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        has_header (bool): Has header row (default: False).
        first_row (int): First row (zero based, default: 0).
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    See also:
        `io.read_any`

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'fixed.txt')
        >>> widths = [0, 18, 29, 33, 38, 50]
        >>> records = read_fixed_csv(filepath, widths)
        >>> records.next() == {
        ...     u'column_1': 'Chicago Reader',
        ...     u'column_2': '1971-01-01',
        ...     u'column_3': '40',
        ...     u'column_4': 'True',
        ...     u'column_5': '1.0',
        ...     u'column_6': '04:14:001971-01-01T04:14:00'}
        ...
        True
        >>> filepath = p.join(parent_dir, 'data', 'test', 'fixed_w_header.txt')
        >>> records = read_fixed_csv(filepath, widths, has_header=True)
        >>> records.next() == {
        ...     u'News Paper': 'Chicago Reader',
        ...     u'Founded': '1971-01-01',
        ...     u'Int': '40',
        ...     u'Bool': 'True',
        ...     u'Float': '1.0',
        ...     u'Timestamp': '04:14:001971-01-01T04:14:00'}
        ...
        True
    """
    def reader(f, **kwargs):
        sanitize = kwargs.get('sanitize')
        dedupe = kwargs.pop('dedupe', False)
        has_header = kwargs.get('has_header')
        first_row = kwargs.get('first_row', 0)
        schema = tuple(it.izip_longest(widths, widths[1:]))
        [f.next() for _ in xrange(first_row)]

        if has_header:
            line = f.readline()
            names = filter(None, (line[s:e].strip() for s, e in schema))
            uscored = list(ft.underscorify(names)) if sanitize else names
            header = list(ft.dedupe(uscored)) if dedupe else uscored
        else:
            header = ['column_%i' % (n + 1) for n in xrange(len(widths))]

        zipped = zip(header, schema)

        get_row = lambda line: {k: line[v[0]:v[1]].strip() for k, v in zipped}
        return it.imap(get_row, f)

    return read_any(filepath, reader, mode, **kwargs)


def sanitize_sheet(sheet, mode, date_format):
    """Formats content from xls/xslx files as strings according to its cell
    type.

    Args:
        sheet (obj): `xlrd` sheet object.
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
        XL_CELL_EMPTY: lambda v: '',
        XL_CELL_NUMBER: lambda v: unicode(v),
        XL_CELL_BOOLEAN: lambda v: unicode(bool(v)),
        XL_CELL_ERROR: lambda v: xlrd.error_text_from_code[v],
    }

    for i in xrange(sheet.nrows):
        for ctype, value in it.izip(sheet.row_types(i), sheet.row_values(i)):
            yield (i, switch.get(ctype, lambda v: v)(value))


def read_xls(filepath, **kwargs):
    """Reads an xls/xlsx file.

    Args:
        filepath (str): The xls/xlsx file path, file, or SpooledTemporaryFile.
        kwargs (dict): Keyword arguments that are passed to the xls reader.

    Kwargs:
        sheet (int): Zero indexed sheet to open (default: 0)
        has_header (bool): Has header row (default: True).
        first_row (int): First row (zero based, default: 0).
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
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.xls')
        >>> records = read_xls(filepath, sanitize=True)
        >>> records.next() == {
        ...     u'some_value': u'234.0',
        ...     u'some_date': '1982-05-04',
        ...     u'sparse_data': u'Iñtërnâtiônàližætiøn',
        ...     u'unicode_test': u'Ādam'}
        ...
        True
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.xlsx')
        >>> records = read_xls(filepath, sanitize=True)
        >>> records.next() == {
        ...     u'some_value': u'234.0',
        ...     u'some_date': '1982-05-04',
        ...     u'sparse_data': u'Iñtërnâtiônàližætiøn',
        ...     u'unicode_test': u'Ādam'}
        ...
        True
        >>> with open(filepath, 'r+b') as f:
        ...     records = read_xls(f, sanitize=True)
        ...     records.next() == {
        ...         u'some_value': u'234.0',
        ...         u'some_date': '1982-05-04',
        ...         u'sparse_data': u'Iñtërnâtiônàližætiøn',
        ...         u'unicode_test': u'Ādam'}
        True
    """
    has_header = kwargs.get('has_header', True)
    first_row = kwargs.get('first_row', 0)
    sanitize = kwargs.get('sanitize')
    dedupe = kwargs.pop('dedupe', False)

    xlrd_kwargs = {
        'on_demand': kwargs.get('on_demand'),
        'ragged_rows': not kwargs.get('pad_rows'),
        'encoding_override': kwargs.get('encoding', True)
    }

    date_format = kwargs.get('date_format', '%Y-%m-%d')

    try:
        mm = mmap(filepath.fileno(), 0)
        book = xlrd.open_workbook(file_contents=mm, **xlrd_kwargs)
    except AttributeError:
        book = xlrd.open_workbook(filepath, **xlrd_kwargs)

    sheet = book.sheet_by_index(kwargs.get('sheet', 0))

    # Get header row and remove empty columns
    names = sheet.row_values(first_row)

    if has_header:
        stripped = [name for name in names if name.strip()]
        uscored = list(ft.underscorify(stripped)) if sanitize else stripped
        header = list(ft.dedupe(uscored)) if dedupe else uscored
    else:
        header = ['column_%i' % (n + 1) for n in xrange(len(names))]

    # Convert to strings
    sanitized = sanitize_sheet(sheet, book.datemode, date_format)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        if has_header and key == first_row:
            continue

        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(header, values))


def read_json(filepath, mode='rU', path='item', newline=False):
    """Reads a json file (both regular and newline-delimited)

    Args:
        filepath (str): The json file path or file like object.
        mode (Optional[str]): The file open mode (default: 'rU').
        path (Optional[str]): Path to the content you wish to read
            (default: 'item', i.e., the root list). Note: `path` must refer to
            a list.

        newline (Optional[bool]): Interpret file as newline-delimited
            (default: False).

    Returns:
        Iterable: The parsed records

    See also:
        `io.read_any`

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.json')
        >>> records = read_json(filepath)
        >>> records.next() == {
        ...     u'text': u'Chicago Reader',
        ...     u'float': 1,
        ...     u'datetime': u'1971-01-01T04:14:00',
        ...     u'boolean': True,
        ...     u'time': u'04:14:00',
        ...     u'date': u'1971-01-01',
        ...     u'integer': 40}
        ...
        True
    """
    reader = lambda f: it.map(loads, f) if newline else items(f, path)
    return read_any(filepath, reader, mode)


def read_geojson(filepath, mode='rU'):
    """Reads a geojson file

    Args:
        filepath (str): The geojson file path or file like object.
        mode (Optional[str]): The file open mode (default: 'rU').

    Returns:
        Iterable: The parsed records

    See also:
        `io.read_any`

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.geojson')
        >>> records = read_geojson(filepath)
        >>> records.next() == {
        ...     u'prop0': u'value0',
        ...     u'id': None,
        ...     u'geojson': '{"type": "Point", "coordinates": [102, 0.5]}'}
        ...
        True
    """
    def reader(f):
        try:
            features = items(f, 'features.item')
        except KeyError:
            raise TypeError('Only GeoJSON with features are supported.')
        else:
            for feature in features:
                _id = {'id': feature.get('id')}
                record = feature.get('properties') or {}
                dumped = dumps(feature['geometry'], cls=ft.CustomEncoder)
                geojson = {'geojson': dumped}
                yield pr.merge([_id, record, geojson])

    return read_any(filepath, reader, mode)


def write(filepath, content, mode='wb+', **kwargs):
    """Writes content to a file path or file like object.

    Args:
        filepath (str): The file path or file like object to write to.
        content (obj): File like object or `requests` iterable response.
        mode (Optional[str]): The file open mode (default: 'wb+').
        kwargs: Keyword arguments.

    Kwargs:
        chunksize (Optional[int]): Number of bytes to write at a time (default:
            None, i.e., all).
        length (Optional[int]): Length of content (default: 0).
        bar_len (Optional[int]): Length of progress bar (default: 50).

    Returns:
        int: bytes written

    See also:
        `io.read_any`

    Examples:
        >>> import requests
        >>> from tempfile import TemporaryFile
        >>> write(TemporaryFile(), StringIO('Hello World'))
        11
        >>> write(TemporaryFile(), StringIO('Iñtërnâtiônàližætiøn'))
        20
        >>> write(TemporaryFile(), IterStringIO(iter('Hello World')), \
chunksize=2)
        12
        >>> r = requests.get('http://google.com', stream=True)
        >>> write(TemporaryFile(), r.iter_content) > 10000
        True
    """
    def _write(f, content, **kwargs):
        chunksize = kwargs.get('chunksize')
        length = int(kwargs.get('length') or 0)
        bar_len = kwargs.get('bar_len', 50)
        progress = 0

        for c in ft.chunk(content, chunksize):
            if isinstance(c, unicode):
                encoded = c.encode(ENCODING)
            elif hasattr(c, 'sort'):
                # it's a list so convert to a string
                encoded = ft.byte(c)
            else:
                encoded = c

            f.write(encoded)
            progress += chunksize or len(c)

            if length:
                bars = min(int(bar_len * progress / length), bar_len)
                print('\r[%s%s]' % ('=' * bars, ' ' * (bar_len - bars)))
                sys.stdout.flush()

        yield progress

    args = [content]
    return read_any(filepath, _write, mode, *args, **kwargs).next()


def hash_file(filepath, hasher='sha1', chunksize=0, verbose=False):
    """Hashes a file path or file like object.
    http://stackoverflow.com/a/1131255/408556

    Args:
        filepath (str): The file path or file like object to hash.
        hasher (str): The hashlib hashing algorithm to use (default: sha1).

        chunksize (Optional[int]): Number of bytes to write at a time
            (default: 0, i.e., all).

        verbose (Optional[bool]): Print debug statements (default: False).

    Returns:
        str: File hash.

    See also:
        `io.read_any`

    Examples:
        >>> from tempfile import TemporaryFile
        >>> hash_file(TemporaryFile())
        'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    """
    def reader(f, hasher):
        if chunksize:
            while True:
                data = f.read(chunksize)
                if not data:
                    break

                hasher.update(data)
        else:
            hasher.update(f.read())

        yield hasher.hexdigest()

    args = [getattr(hashlib, hasher)()]
    file_hash = read_any(filepath, reader, 'rb', *args).next()

    if verbose:
        print('File %s hash is %s.' % (filepath, file_hash))

    return file_hash


def detect_encoding(f, verbose=False):
    """Detects a file's encoding.

    Args:
        f (obj): The file like object to detect.
        verbose (Optional[bool]): The file open mode (default: False).
        mode (Optional[str]): The file open mode (default: 'rU').

    Returns:
        dict: The encoding result

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> with open(filepath, mode='rU') as f:
        ...     result = detect_encoding(f)
        ...     result == {'confidence': 0.99, 'encoding': 'utf-8'}
        ...
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
