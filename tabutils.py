#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils
~~~~~~~~

Provides miscellaneous utility methods

Examples:
    literal blocks::

        python example_google.py

Attributes:
    ENCODING (str): Default file encoding.
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import sys
import xlrd
import itertools as it
import unicodecsv as csv

from StringIO import StringIO
from subprocess import check_output, check_call, Popen, PIPE, CalledProcessError
from dbfread import DBF
from decimal import Decimal, InvalidOperation
from dateutil.parser import parse
from functools import partial
from xlrd.xldate import xldate_as_datetime as xl2dt
from xlrd import (
    XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_NUMBER, XL_CELL_BOOLEAN,
    XL_CELL_ERROR)

from chardet.universaldetector import UniversalDetector
from slugify import slugify

ENCODING = 'utf-8'
CURRENCIES = ('$', '£', '€')

__title__ = 'tabutils'
__package_name__ = 'tabutils'
__author__ = 'Reuben Cummings'
__description__ = 'tabular data utility methods'
__email__ = 'reubano@gmail.com'
__version__ = '0.5.0'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 Reuben Cummings'

underscorify = lambda fields: [slugify(f, separator='_') for f in fields]

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
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.csv')
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
        if any(v.strip() for v in row.values()):
            yield row


def _sanitize_sheet(sheet, mode, date_format):
    """Formats xlrd cell types (from xls/xslx file) as strings.

    Args:
        book (obj): `xlrd` workbook object.
        mode (str): `xlrd` workbook datemode property.
        date_format (str): `strftime()` date format.

    Yields:
        Tuple[int, str]: A tuple of (row_number, value).

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.xls')
        >>> book = xlrd.open_workbook(filepath)
        >>> sheet = book.sheet_by_index(0)
        >>> sheet.row_values(1) == [\
30075.0, u'Iñtërnâtiônàližætiøn', 234.0, u'Ādam', u' ']
        True
        >>> sanitized = _sanitize_sheet(sheet, book.datemode, '%Y-%m-%d')
        >>> [v for i, v in sanitized if i == 1] == [\
'1982-05-04', u'Iñtërnâtiônàližætiøn', u'234.0', u'Ādam', u' ']
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
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> csv_filepath = p.join(parent_dir, 'testdata', 'test.csv')
        >>> csv_records = read_csv(csv_filepath, sanitize=True)
        >>> csv_header = sorted(csv_records.next().keys())
        >>> csv_fields = gen_fields(csv_header, True)
        >>> csv_records.next()['some_date']
        u'05/04/82'
        >>> casted_csv_row = gen_type_cast(csv_records, csv_fields).next()
        >>> casted_csv_values = [casted_csv_row[h] for h in csv_header]
        >>>
        >>> xls_filepath = p.join(parent_dir, 'testdata', 'test.xls')
        >>> xls_records = read_xls(xls_filepath, sanitize=True)
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


def detect_encoding(f):
    """Detects a file's encoding.

    Args:
        f (obj): The file like object to detect.

    Returns:
        dict: The encoding result

    Examples:
        >>> from os import path as p
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.csv')
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
    # print('detector.result', detector.result)
    return detector.result


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
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.mdb')
        >>> records = read_mdb(filepath, sanitize=True)
        >>> header = sorted(records.next().keys())
        >>> header
        [u'date_of_order_of_court', u'forenames', u'forenames_master_or_father', u'freedom', u'how_admitted', u'id_no', u'livery', u'notes', u'remarks', u'source_ref', u'surname', u'surname_master_or_father']
        >>> row = records.next()
        >>> [row[h] for h in header]
        [u'', u'Richard', u'', u'05/11/01 00:00:00', u'Redn.', u'2', u'', u'', u'', u'MF 324', u'Abbey', u'']
        >>> [r['surname'] for r in records]
        [u'Abbis', u'Abbis', u'Abbis', u'Abbot', u'Abbot', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u'Abbott', u"'"]
    """
    args = ['mdb-tables', '-1', filepath]

    try:
        check_call(args)
    except OSError:
        raise OSError(
            'You must install [mdbtools]' \
            '(http://sourceforge.net/projects/mdbtools/) in order to use ' \
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
        # print('first_line', first_line)
        header = csv.reader(StringIO(first_line), **kwargs).next()
        names = underscorify(header) if sanitize else header

        for line in iter(pipe.readline, b''):
            values = csv.reader(StringIO(line), **kwargs).next()
            yield dict(zip(names, values))


def read_dbf(filepath, **kwargs):
    """Reads a dBase, Visual FoxPro, or FoxBase+ file

    Args:
        filepath (str): The dbf file path.
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
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.dbf')
        >>> records = read_dbf(filepath, sanitize=True)
        >>> header = sorted(records.next().keys())
        >>> header
        [u'aland10', u'awater10', u'cd111fp', u'cdsessn', u'funcstat10', u'geoid10', u'intptlat10', u'intptlon10', u'lsad10', u'mtfcc10', u'namelsad10', u'statefp10']
        >>> row = records.next()
        >>> [row[h] for h in header]
        [320220379, 15485125, u'05', u'111', u'N', u'2705', u'+44.9781144', u'-093.2928317', u'C2', u'G5200', u'Congressional District 5', u'27']
        >>> [r['namelsad10'] for r in records]
        [u'Congressional District 4', u'Congressional District 2', u'Congressional District 1', u'Congressional District 6', u'Congressional District 7', u'Congressional District 3']
    """
    kwargs['lowernames'] = kwargs.pop('sanitize', None)

    for record in DBF(filepath, **kwargs):
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
        sanitize (bool): Underscorify and lowercase field names
            (default: False).

    Yields:
        dict: A row of data whose keys are the field names.

    Raises:
        NotFound: If unable to find the resource.

    Examples:
        >>> from os import path as p
        >>> from tempfile import NamedTemporaryFile
        >>> tmpfile = NamedTemporaryFile()
        >>> filepath = tmpfile.name
        >>> read_csv(filepath).next()
        Traceback (most recent call last):
        StopIteration
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.csv')
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
    """
    def func(f):
        encoding = kwargs.pop('encoding', ENCODING)
        sanitize = kwargs.pop('sanitize', False)

        if kwargs.pop('has_header', True):
            # Remove empty columns and underscorify field names
            header = csv.reader(f, encoding=encoding, **kwargs).next()
            names = [name for name in header if name.strip()]
            names = underscorify(names) if sanitize else names
        else:
            names = None

        try:
            records = _read_csv(f, encoding, names)
        except UnicodeDecodeError:
            # Try to detect the encoding
            result = detect_encoding(f)
            records = _read_csv(f, result['encoding'], names)

        for row in records:
            yield row

    if hasattr(filepath, 'read'):
        for row in func(filepath):
            yield row
    else:
        with open(filepath, mode) as f:
            for row in func(f):
                yield row


def read_xls(filepath, **kwargs):
    """Reads an xls/xlsx file.

    Args:
        filepath (str): The xls/xlsx file path.
        **kwargs: Keyword arguments that are passed to the xls reader.

    Kwargs:
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
        >>> from tempfile import NamedTemporaryFile
        >>> tmpfile = NamedTemporaryFile()
        >>> filepath = tmpfile.name
        >>> read_xls(filepath).next()
        Traceback (most recent call last):
        XLRDError: File size is 0 bytes
        >>> parent_dir = p.abspath(p.dirname(__file__))
        >>> filepath = p.join(parent_dir, 'testdata', 'test.xls')
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
        >>> filepath = p.join(parent_dir, 'testdata', 'test.xlsx')
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
    """
    date_format = kwargs.get('date_format', '%Y-%m-%d')

    xlrd_kwargs = {
        'on_demand': kwargs.get('on_demand'),
        'ragged_rows': not kwargs.get('pad_rows'),
        'encoding_override': kwargs.get('encoding', True)
    }

    book = xlrd.open_workbook(filepath, **xlrd_kwargs)
    sheet = book.sheet_by_index(0)
    header = sheet.row_values(0)

    # Remove empty columns
    names = [name for name in header if name.strip()]

    # Underscorify field names
    if kwargs.get('sanitize'):
        names = [slugify(name, separator='_') for name in names]

    # Convert to strings
    sanitized = _sanitize_sheet(sheet, book.datemode, date_format)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(names, values))
