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
from subprocess import check_output, check_call, Popen, PIPE, CalledProcessError

from xlrd.xldate import xldate_as_datetime as xl2dt
from chardet.universaldetector import UniversalDetector
from xlrd import (
    XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_NUMBER, XL_CELL_BOOLEAN,
    XL_CELL_ERROR)

from . import fntools as ft, dbf, ENCODING


class IterStringIO(TextIOBase):
    """A lazy StringIO that writes a generator of strings and reads bytearrays.

    http://stackoverflow.com/a/32020108/408556
    """

    def __init__(self, iterable=None):
        """ IterStringIO constructor

        Args:
            iterable (dict): bank mapper (see csv2vcard.mappings)

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
            >>> list(IterStringIO(content).readlines())
            [bytearray(b'line one'), bytearray(b'line two'), \
bytearray(b'line three')]
        """
        iterable = iterable or []
        chained = self._chain(iterable)
        self.iter = self._encode(chained)

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
        return ft.byte(it.islice(iterable, n)) if n else ft.byte(iterable)

    def write(self, iterable):
        chained = self._chain(iterable)
        self.iter = self._chain([self.iter, self._encode(chained)])

    def read(self, n=None):
        return self._read(self.iter, n)

    def readline(self, n=None):
        return self._read(self.lines.next(), n)

    def readlines(self):
        return it.imap(self._read, self.lines)

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


def _read_csv(f, encoding, header=None, has_header=True):
    """Helps read a csv file.

    Args:
        f (obj): The csv file like object.
        encoding (str): File encoding.

    Kwargs:
        header (List[str]): The column names.

    Yields:
        dict: A csv record.

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> with_header = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> no_header = p.join(parent_dir, 'data', 'test', 'no_header_row.csv')
        >>> f = open(with_header, 'rU')
        >>> sorted(_read_csv(f, 'utf-8').next().items()) == [(u'Some Date', \
u'05/04/82'), (u'Some Value', u'234'), (u'Sparse Data', \
u'Iñtërnâtiônàližætiøn'), (u'Unicode Test', u'Ādam')]
        True
        >>> header = ['some_date', 'sparse_data', 'some_value', 'unicode_test']
        >>> sorted(_read_csv(f, 'utf-8', header).next().items()) == [\
(u'some_date', u'05/04/82'), (u'some_value', u'234'), (u'sparse_data', \
u'Iñtërnâtiônàližætiøn'), (u'unicode_test', u'Ādam')]
        True
        >>> f.close()
        >>> header = ['col_1', 'col_2', 'col_3']
        >>> with open(no_header, 'rU') as f:
        ...     records = _read_csv(f, 'utf-8', header, has_header=False)
        ...     sorted(records.next().items())
        [(u'col_1', u'1'), (u'col_2', u'2'), (u'col_3', u'3')]
    """
    f.seek(0)

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

    table = table or check_output(args).splitlines()[0]
    pkwargs = {'stdout': PIPE, 'bufsize': 1, 'universal_newlines': True}

    # http://stackoverflow.com/a/2813530/408556
    # http://stackoverflow.com/a/17698359/408556
    with Popen(['mdb-export', filepath, table], **pkwargs).stdout as pipe:
        sanitize = kwargs.pop('sanitize', None)
        first_line = pipe.readline()
        header = csv.reader(StringIO(first_line), **kwargs).next()
        names = list(ft.underscorify(header)) if sanitize else header

        for line in iter(pipe.readline, b''):
            values = csv.reader(StringIO(line), **kwargs).next()
            yield dict(zip(names, values))


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
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    Examples:
        >>> from os import path as p
        >>> from tempfile import TemporaryFile
        >>> read_csv(TemporaryFile()).next()
        Traceback (most recent call last):
        StopIteration
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
        >>> with open(filepath, 'rU') as f:
        ...     records = read_csv(f, sanitize=True)
        ...     records.next() == {
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
    encoding = kwargs.pop('encoding', ENCODING)
    sanitize = kwargs.pop('sanitize', False)
    has_header = kwargs.pop('has_header', True)

    def read_file(f):
        # Get header row and remove empty columns
        names = csv.reader(f, encoding=encoding, **kwargs).next()

        if has_header:
            stripped = [name for name in names if name.strip()]
            header = list(ft.underscorify(stripped)) if sanitize else stripped
        else:
            header = ['column_%i' % (n + 1) for n in xrange(len(names))]

        try:
            records = _read_csv(f, encoding, header, has_header)
        except UnicodeDecodeError:
            # Try to detect the encoding
            new_encoding = detect_encoding(f)['encoding']
            records = _read_csv(f, new_encoding, header, has_header)

        return records

    if hasattr(filepath, 'read'):
        for row in read_file(filepath):
            yield row
    else:
        with open(filepath, mode) as f:
            for row in read_file(f):
                yield row

    if hasattr(filepath, 'read'):
        for row in read_file(filepath):
            yield row
    else:
        with open(filepath, mode) as f:
            for row in read_file(f):
                yield row


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
        XL_CELL_EMPTY: lambda v: None,
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
        filepath (str): The xls/xlsx file path or file like object.
        kwargs (dict): Keyword arguments that are passed to the xls reader.

    Kwargs:
        sheet (int): Zero indexed sheet to open (default: 0)
        has_header (bool): Has header row (default: True).
        date_format (str): Date format passed to `strftime()` (default:
            '%Y-%m-%d', i.e, 'YYYY-MM-DD').

        encoding (str): File encoding. By default, the encoding is derived from
            the file's `CODEPAGE` number, e.g., 1252 translates to `cp1252`.

        sanitize (bool): Underscorify and lowercase field names
            (default: False).

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
        >>> from tempfile import TemporaryFile
        >>> read_xls(TemporaryFile()).next()
        Traceback (most recent call last):
        ValueError: cannot mmap an empty file
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
    sanitize = kwargs.get('sanitize')

    xlrd_kwargs = {
        'on_demand': kwargs.get('on_demand'),
        'ragged_rows': not kwargs.get('pad_rows'),
        'encoding_override': kwargs.get('encoding', True)
    }

    date_format = kwargs.get('date_format', '%Y-%m-%d')

    try:
        from mmap import mmap

        mm = mmap(filepath.fileno(), 0)
        book = xlrd.open_workbook(file_contents=mm, **xlrd_kwargs)
    except AttributeError:
        book = xlrd.open_workbook(filepath, **xlrd_kwargs)

    sheet = book.sheet_by_index(kwargs.get('sheet', 0))

    # Get header row and remove empty columns
    if has_header:
        names = sheet.row_values(0)
        stripped = [name for name in names if name.strip()]
        header = list(ft.underscorify(stripped)) if sanitize else stripped
    else:
        header = ['column_%i' % (n + 1) for n in xrange(len(names))]

    # Convert to strings
    sanitized = sanitize_sheet(sheet, book.datemode, date_format)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        if has_header and key == 0:
            continue

        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(header, values))


def write(filepath, content, mode='wb', **kwargs):
    """Writes content to a file path or file like object.

    Args:
        filepath (str): The file path or file like object to write to.
        content (obj): File like object or `requests` iterable response.
        kwargs: Keyword arguments.

    Kwargs:
        mode (Optional[str]): The file open mode (default: 'wb').
        chunksize (Optional[int]): Number of bytes to write at a time (default:
            None, i.e., all).
        length (Optional[int]): Length of content (default: 0).
        bar_len (Optional[int]): Length of progress bar (default: 50).

    Returns:
        int: bytes written

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
            f.write(c.encode(ENCODING) if hasattr(c, 'encode') else c)
            progress += chunksize or len(c)

            if length:
                bars = min(int(bar_len * progress / length), bar_len)
                print('\r[%s%s]' % ('=' * bars, ' ' * (bar_len - bars)))
                sys.stdout.flush()

        return progress

    if hasattr(filepath, 'read'):
        return _write(filepath, content, **kwargs)
    else:
        with open(filepath, mode) as f:
            return _write(f, content, **kwargs)


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
