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

from slugify import slugify
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
        """
        iterable = iterable or []
        not_newline = lambda s: s not in {'\n', '\r', '\r\n'}
        chained = self._chain(iterable)
        self.iter = self._encode(chained)
        self.next_line = it.takewhile(not_newline, self.iter)

    def _encode(self, iterable):
        return (s.encode(ENCODING) for s in iterable)

    def _chain(self, iterable):
        iterable = iterable or []
        return it.chain.from_iterable(it.ifilter(None, iterable))

    def _read(self, iterable, n):
        sliced = list(it.islice(iterable, None, n))
        return ft.byte(sliced)

    def write(self, iterable):
        chained = self._chain(iterable)
        self.iter = self._chain([self.iter, self._encode(chained)])

    def read(self, n=pow(2, 34)):
        return self._read(self.iter, n)

    def readline(self, n=pow(2, 34)):
        return self._read(self.next_line, n)


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


def _read_csv(f, encoding, names=('field_0',)):
    """Helps read a csv file.

    Args:
        f (obj): The csv file like object.
        encoding (str): File encoding.

    Kwargs:
        names (List[str]): The header names.

    Yields:
        dict: A csv record.

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.csv')
        >>> f = open(filepath, 'rU')
        >>> names = ['some_date', 'sparse_data', 'some_value', 'unicode_test']
        >>> records = _read_csv(f, 'utf-8', names)
        >>> it.islice(records, 2, 3).next()['some_date']
        u'01-Jan-15'
        >>> f.close()
    """
    # Read data
    f.seek(0)
    reader = csv.DictReader(f, names, encoding=encoding)

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
        **kwargs: Keyword arguments that are passed to the csv reader.

    Kwargs:
        table (str): The table to load (default: None, the first found table).
        sanitize (bool): Convert field names to lower case (default: False).
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
        >>> header = sorted(records.next().keys())
        >>> header
        [u'date_of_order_of_court', u'forenames', \
u'forenames_master_or_father', u'freedom', u'how_admitted', u'id_no', \
u'livery', u'notes', u'remarks', u'source_ref', u'surname', \
u'surname_master_or_father']
        >>> row = records.next()
        >>> [row[h] for h in header]
        [u'', u'Richard', u'', u'05/11/01 00:00:00', u'Redn.', u'2', u'', \
u'', u'', u'MF 324', u'Abbey', u'']
        >>> [r['surname'] for r in records]
        [u'Abbis', u'Abbis', u'Abbis', u'Abbot', u'Abbot', u'Abbott', \
u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', \
u'Abbott', u'Abbott', u'Abbott', u'Abbott', u"'"]
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
        **kwargs: Keyword arguments that are passed to the DBF reader.

    Kwargs:
        load (bool): Load all records into memory (default: false).
        encoding (bool): Character encoding (default: None, parsed from
            the `language_driver`).

        sanitize (bool): Convert field names to lower case (default: False).
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
        >>> header = sorted(records.next().keys())
        >>> header
        [u'aland10', u'awater10', u'cd111fp', u'cdsessn', u'funcstat10', \
u'geoid10', u'intptlat10', u'intptlon10', u'lsad10', u'mtfcc10', \
u'namelsad10', u'statefp10']
        >>> row = records.next()
        >>> [row[h] for h in header]
        [320220379, 15485125, u'05', u'111', u'N', u'2705', u'+44.9781144', \
u'-093.2928317', u'C2', u'G5200', u'Congressional District 5', u'27']
        >>> [r['namelsad10'] for r in records]
        [u'Congressional District 4', u'Congressional District 2', \
u'Congressional District 1', u'Congressional District 6', u'Congressional \
District 7', u'Congressional District 3']
        >>> f = open(filepath, 'rb')
        >>> read_dbf(f, sanitize=True, recfactory=dict).next() == {\
u'awater10': 12416573076, u'aland10': 71546663636, u'intptlat10': \
u'+47.2400052', u'lsad10': u'C2', u'cd111fp': u'08', u'namelsad10': \
u'Congressional District 8', u'funcstat10': u'N', u'statefp10': u'27', \
u'cdsessn': u'111', u'mtfcc10': u'G5200', u'geoid10': u'2708', u'intptlon10': \
u'-092.9323194'}
        True
        >>> f.close()
    """
    kwargs['lowernames'] = kwargs.pop('sanitize', None)

    for record in dbf.DBF2(filepath, **kwargs):
        yield record


def read_csv(filepath, mode='rU', **kwargs):
    """Reads a csv file.

    Args:
        filepath (str): The csv file path or file like object.
        mode (Optional[str]): The file open mode (default: 'rU').
        **kwargs: Keyword arguments that are passed to the csv reader.

    Kwargs:
        delimiter (str): Field delimiter (default: ',').
        quotechar (str): Quote character (default: '"').
        encoding (str): File encoding.
        has_header (bool): Has header row (default: True).
        remove_header (bool): Remove header record from result (default: False).
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
        >>> header = sorted(records.next().keys())
        >>> header
        [u'some_date', u'some_value', u'sparse_data', u'unicode_test']
        >>> row = records.next()
        >>> [row[h] for h in header] == [ \
u'05/04/82', u'234', u'Iñtërnâtiônàližætiøn', u'Ādam']
        True
        >>> [r['some_date'] for r in records]
        [u'01-Jan-15', u'December 31, 1995']
        >>> f = open(filepath, 'rU')
        >>> read_csv(f, sanitize=True).next() == {u'sparse_data': u'Sparse \
Data', u'some_date': u'Some Date', u'some_value': u'Some Value', \
u'unicode_test': u'Unicode Test'}
        True
        >>> f.close()
    """
    def read_file(f):
        encoding = kwargs.pop('encoding', ENCODING)
        sanitize = kwargs.pop('sanitize', False)
        remove_header = kwargs.pop('remove_header', False)
        names = None

        if kwargs.pop('has_header', True):
            # Remove empty columns and underscorify field names
            header = csv.reader(f, encoding=encoding, **kwargs).next()
            names = [name for name in header if name.strip()]
            names = list(ft.underscorify(names)) if sanitize else names

        try:
            records = _read_csv(f, encoding, names)
        except UnicodeDecodeError:
            # Try to detect the encoding
            encoding = detect_encoding(f)['encoding']
            records = _read_csv(f, encoding, names)

        records.next() if remove_header else None

        for row in records:
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
        **kwargs: Keyword arguments that are passed to the xls reader.

    Kwargs:
        sheet (int): Zero indexed sheet to open (default: 0)
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
        >>> header = sorted(records.next().keys())
        >>> header
        [u'some_date', u'some_value', u'sparse_data', u'unicode_test']
        >>> row = records.next()
        >>> [row[h] for h in header] == [ \
'1982-05-04', u'234.0', u'Iñtërnâtiônàližætiøn', u'Ādam']
        True
        >>> [r['some_date'] for r in records]
        ['2015-01-01', '1995-12-31']
        >>> filepath = p.join(parent_dir, 'data', 'test', 'test.xlsx')
        >>> records = read_xls(filepath, sanitize=True)
        >>> header = sorted(records.next().keys())
        >>> header
        [u'some_date', u'some_value', u'sparse_data', u'unicode_test']
        >>> row = records.next()
        >>> [row[h] for h in header] == [ \
'1982-05-04', u'234.0', u'Iñtërnâtiônàližætiøn', u'Ādam']
        True
        >>> [r['some_date'] for r in records]
        ['2015-01-01', '1995-12-31']
        >>> f = open(filepath, 'r+b')
        >>> read_xls(f, sanitize=True).next() == {u'some_value': \
u'Some Value', u'some_date': u'Some Date', u'sparse_data': u'Sparse Data', \
u'unicode_test': u'Unicode Test'}
        True
        >>> f.close()
    """
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
    header = sheet.row_values(0)

    # Remove empty columns
    names = [name for name in header if name.strip()]

    # Underscorify field names
    if kwargs.get('sanitize'):
        names = [slugify(name, separator='_') for name in names]

    # Convert to strings
    sanitized = sanitize_sheet(sheet, book.datemode, date_format)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(names, values))


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
        >>> from tempfile import TemporaryFile, NamedTemporaryFile
        >>> tmpfile = NamedTemporaryFile(delete='True')
        >>> write(tmpfile.name, StringIO('Hello World'))
        11
        >>> write(TemporaryFile(), StringIO('Iñtërnâtiônàližætiøn'))
        20
        >>> write(TemporaryFile(), IterStringIO(iter('Hello World')))
        11
        >>> write(TemporaryFile(), IterStringIO(iter('Hello World')), \
chunksize=2)
        12
        >>> write(TemporaryFile(), StringIO('http://google.com'))
        17
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
