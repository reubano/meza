#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.io
~~~~~~~

Provides methods for reading/writing/processing tabular formatted files

Examples:
    basic usage::

        >>> from meza.io import read_csv
        >>>
        >>> path = p.join(DATA_DIR, 'test.csv')
        >>> csv_records = read_csv(path)
        >>> csv_header = next(csv_records).keys()
        >>> next(csv_records)['Some Value'] == '100'
        True
"""
import itertools as it
import sys
import hashlib
import sqlite3
import json
import os

from os import path as p
from datetime import time
from mmap import mmap
from collections import deque
from subprocess import check_output, check_call, Popen, PIPE, CalledProcessError
from http import client
from csv import Error as csvError
from functools import partial
from codecs import iterdecode, iterencode, StreamReader
from itertools import zip_longest
from math import inf

import yaml
import xlrd
import pygogo as gogo

from bs4 import BeautifulSoup, FeatureNotFound
from ijson import items
from chardet.universaldetector import UniversalDetector
from xlrd import (
    XL_CELL_DATE,
    XL_CELL_EMPTY,
    XL_CELL_NUMBER,
    XL_CELL_BOOLEAN,
    XL_CELL_ERROR,
)

from xlrd.xldate import xldate_as_datetime as xl2dt
from io import StringIO, TextIOBase, BytesIO, open

from . import fntools as ft, process as pr, unicsv as csv, dbf, ENCODING, BOM, DATA_DIR

# pylint: disable=C0103
logger = gogo.Gogo(__name__, monolog=True, verbose=True).logger

# pylint: disable=C0103
encode = lambda iterable: (s.encode(ENCODING) for s in iterable)
chain = lambda iterable: it.chain.from_iterable(iterable or [])

NEWLINES = {b"\n", b"\r", b"\r\n", "\n", "\r", "\r\n"}


def groupby_line(iterable):
    return it.groupby(iterable, lambda s: s not in NEWLINES)


class IterStringIO(TextIOBase):
    """A lazy StringIO that reads a generator of strings.

    https://stackoverflow.com/a/32020108/408556
    https://stackoverflow.com/a/20260030/408556
    """

    # pylint: disable=super-init-not-called
    def __init__(self, iterable=None, bufsize=4096, decode=False, **kwargs):
        """IterStringIO constructor

        Args:
            iterable (Seq[str]): Iterable of strings or bytes
            bufsize (Int): Buffer size for seeking
            decode (bool): Decode the text into a string (default: False)

        Examples:
            >>> StringIO(iter('Hello World')).read(5)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            TypeError:...
            >>> IterStringIO(iter('Hello World')).read(5)
            b'Hello'
            >>> i = IterStringIO(iter('one\\ntwo\\n'))
            >>> list(next(i.lines)) == [b'o', b'n', b'e']
            True
            >>> decoded = IterStringIO(iter('Hello World'), decode=True)
            >>> decoded.read(5) == 'Hello'
            True
        """
        iterable = iterable if iterable else []
        chained = chain(iterable)
        self.iter = encode(chained)
        self.decode = decode
        self.bufsize = bufsize

        # Not sure why I used bytearray here... maybe it was to be able to read
        # by `num` of bytes??
        self.last = deque(bytearray(), self.bufsize)
        self.pos = 0

    def __next__(self):
        return self._read(next(self.lines))

    def __iter__(self):
        return self

    @property
    def lines(self):
        """Read all the lines of content"""
        # TODO: what about a csv with embedded newlines?
        groups = groupby_line(self.iter)
        return (g for k, g in groups if k)

    def _read(self, iterable, num=None, newline=True):
        """Helper method used to read content"""
        content = it.islice(iterable, num) if num else iterable

        byte = ft.byte(content)
        self.last.extend(byte)
        self.pos += num or len(byte)

        if newline:
            self.last.append(os.linesep)

        return byte.decode(ENCODING) if self.decode else bytes(byte)

    def write(self, iterable):
        """Write the content"""
        chained = chain(iterable)
        self.iter = it.chain(self.iter, encode(chained))

    def read(self, num=None):
        """Read the content"""
        return self._read(self.iter, num, newline=False)

    def readline(self, num=None):
        """Read a line of content"""
        return self._read(next(self.lines), num)

    def readlines(self):
        """Read all the lines of content"""
        return map(self._read, self.lines)

    def seek(self, num):
        """Go to a specific position within a file"""
        next_pos = self.pos + 1
        beg_buf = max([0, self.pos - self.bufsize])

        if num <= beg_buf:
            self.iter = it.chain(self.last, self.iter)
            self.last = deque([], self.bufsize)
        elif self.pos > num > beg_buf:
            extend = [self.last.pop() for _ in range(self.pos - num)]
            self.iter = it.chain(reversed(extend), self.iter)
        elif num == self.pos:
            pass
        elif num == next_pos:
            self.last.append(next(self.iter))
        elif num > next_pos:
            pos = num - self.pos
            [self.last.append(x) for x in it.islice(self.iter, 0, pos)]

        self.pos = beg_buf if num < beg_buf else num

    def tell(self):
        """Get the current position within a file"""
        return self.pos


class Reencoder(StreamReader):
    """Recodes a file like object from one encoding to another."""

    def __init__(self, f, fromenc=ENCODING, toenc=ENCODING, **kwargs):
        """Reencoder constructor

        Args:
            f (obj): File-like object
            fromenc (str): The input encoding.
            toenc (str): The output encoding.

        Kwargs:
            remove_BOM (bool): Remove Byte Order Marker (default: True)
            decode (bool): Decode the text into a string (default: False)

        Examples:
            >>> encoding = 'utf-16-be'
            >>> eff = p.join(DATA_DIR, 'utf16_big.csv')
            >>>
            >>> with open(eff, 'rb') as f:
            ...     reenc = Reencoder(f, encoding)
            ...     first = reenc.readline(keepends=False)
            ...     first.decode('utf-8') == '\ufeffa,b,c'
            ...     reenc.readlines()[1].decode('utf-8') == '4,5,ʤ'
            True
            True
            >>> with open(eff, 'rb') as f:
            ...     reenc = Reencoder(f, encoding, decode=True)
            ...     reenc.readline(keepends=False) == '\ufeffa,b,c'
            True
            >>> with open(eff, 'rU', encoding=encoding) as f:
            ...     reenc = Reencoder(f, remove_BOM=True)
            ...     reenc.readline(keepends=False) == b'a,b,c'
            ...     reenc.readline() == b'1,2,3\\n'
            ...     reenc.readline().decode('utf-8') == '4,5,ʤ'
            True
            True
            True
        """
        self.fileno = f.fileno
        first_line = next(f)
        bytes_mode = isinstance(first_line, bytes)
        decode = kwargs.get("decode")
        rencode = not decode

        if kwargs.get("remove_BOM"):
            strip = BOM.encode(fromenc) if bytes_mode else BOM
            first_line = first_line.lstrip(strip)

        chained = it.chain([first_line], f)

        if bytes_mode:
            decoded = iterdecode(chained, fromenc)
            self.binary = rencode
            proper_newline = first_line.endswith(os.linesep.encode(fromenc))
        else:
            decoded = chained
            self.binary = bytes_mode or rencode
            proper_newline = first_line.endswith(os.linesep)

        stream = iterencode(decoded, toenc) if rencode else decoded

        if proper_newline:
            self.join_char = b"" if self.binary else ""
            self.stream = stream
        else:
            # TODO: make sure the read methods are consistent with
            #       proper_newline, e.g., `keepends`.
            #
            # since the newline isn't recognized, `stream` just contains
            # one (very long) line. we pass in this line to iterate over the
            # chars
            groups = groupby_line(next(stream))

            if self.binary:
                self.join_char = os.linesep.encode(fromenc)
            else:
                self.join_char = os.linesep

            self.stream = (self.join_char.join(g) for k, g in groups if k)

    def __next__(self):
        return next(self.stream)

    def __iter__(self):
        return self

    def read(self, n=None):
        stream = it.islice(self.stream, n) if n else self.stream
        return self.join_char.join(stream)

    def readline(self, n=None, keepends=True):
        line = next(self.stream)
        return line if keepends else line.rstrip()

    def readlines(self, sizehint=None):
        return list(self.stream)

    def tell(self):
        pass

    def reset(self):
        pass


class BytesError(ValueError):
    pass


def patch_http_response_read(func):
    """Patches httplib to read poorly encoded chunked data.

    https://stackoverflow.com/a/14206036/408556
    """

    def inner(*args):
        """inner"""
        try:
            return func(*args)
        except client.IncompleteRead as err:
            return err.partial

    return inner


client.HTTPResponse.read = patch_http_response_read(client.HTTPResponse.read)


def _remove_bom_from_dict(row, bom):
    """Remove a byte order marker (BOM) from a dict"""
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
    """Remove a byte order marker (BOM) from a list"""
    for pos, col in enumerate(row):
        try:
            if not pos and bom in col:
                yield col.lstrip(bom)
            else:
                yield col
        except TypeError:
            yield col


def _remove_bom_from_scalar(row, bom):
    """Remove a byte order marker (BOM) from a scalar"""
    try:
        return row.lstrip(bom)
    except AttributeError:
        return row


def is_listlike(item):
    """Determine if a scalar is listlike"""
    if hasattr(item, "keys"):
        listlike = False
    else:
        listlike = {"append", "next", "__reversed__"}.intersection(dir(item))

    return listlike


def remove_bom(row, bom):
    """Remove a byte order marker (BOM)"""
    if is_listlike(row):
        bomless = list(_remove_bom_from_list(row, bom))
    else:
        try:
            # pylint: disable=R0204
            bomless = dict(_remove_bom_from_dict(row, bom))
        except AttributeError:
            bomless = _remove_bom_from_scalar(row, bom)

    return bomless


def get_encoding(filepath):
    """
    Examples:
        >>> get_encoding(p.join(DATA_DIR, 'utf16_big.csv')) == 'UTF-16'
        True
    """
    with open(filepath, "rb") as f:
        encoding = detect_encoding(f)["encoding"]

    return encoding


def get_file_encoding(f, encoding=None, bytes_error=False):
    """Detects a file's encoding"""
    if not encoding and hasattr(f, "encoding"):
        encoding = f.encoding

    if not bytes_error:
        # Set the encoding to None so that we can detect the correct one.
        extra = (" ({})".format(encoding)) if encoding else ""
        logger.warning("%s was opened with the wrong encoding%s", f, extra)
        encoding = None

    if not encoding:
        try:
            f.seek(0)
        except AttributeError:
            pass
        else:
            try:
                # See if we have bytes to avoid reopening the file
                encoding = detect_encoding(f)["encoding"]
            except UnicodeDecodeError:
                msg = "Incorrectly encoded file, reopening with bytes to detect"
                msg += " encoding"
                logger.warning(msg)
                f.close()
                encoding = get_encoding(f.name)
        finally:
            if hasattr(f, "name"):  # otherwise we can't reopen it
                f.close()

    if encoding:
        logger.debug("detected encoding: %s", encoding)

    return encoding


def sanitize_file_encoding(encoding):
    if encoding == "Windows-1252" and os.name == "posix":
        # based on my testing, when excel for mac saves a csv file as
        # 'Windows-1252', you have to open with 'mac-roman' in order
        # to properly read it
        new_encoding = "mac-roman"
        msg = "Detected a `Windows-1252` encoded file on a %s machine."
        msg += " Setting encoding to `%s` instead."
        logger.warning(msg, sys.platform, new_encoding)
    else:
        new_encoding = encoding

    return new_encoding


def is_binary(f):
    try:
        result = "b" in f.mode
    except AttributeError:
        result = isinstance(f, BytesIO)

    return result


def reopen(f, encoding):
    sanitized_encoding = sanitize_file_encoding(encoding)
    logger.debug("Reopening %s with encoding: %s", f, sanitized_encoding)

    try:
        decoded_f = open(f.name, encoding=sanitized_encoding)
    except AttributeError:
        f.seek(0)
        decoded_f = iterdecode(f, sanitized_encoding)

    return decoded_f


def _read_any(f, reader, args, pos=0, recursed=False, **kwargs):
    """Helper func to read a file or filepath"""
    last_row = kwargs.pop("last_row", None) or inf

    # https://github.com/jaraco/csv2ofx/commit/0560beb53d5db94506142298644a226b64c8a95f
    q = deque()

    try:
        if is_binary(f) and reader.__name__ != "writer":
            # only allow binary mode for writing files, not reading
            message = "%s was opened in bytes mode but isn't being written to"
            raise BytesError(message % f)

        for num, line in enumerate(reader(f, *args, **kwargs)):
            if last_row > num >= pos:
                yield line
                pos += 1
            elif last_row < 0:
                q.append(line)

                if (num >= abs(last_row)) and (num >= pos):
                    yield q.popleft()

                pos += 1

    except (UnicodeDecodeError, csvError, BytesError) as err:
        logger.warning(err)
        encoding = kwargs.pop("encoding", None)
        bytes_error = type(err).__name__ == "BytesError"

        if not recursed:
            ekwargs = {"encoding": encoding, "bytes_error": bytes_error}
            encoding = get_file_encoding(f, **ekwargs)

        if recursed or not encoding:
            logger.error("Unable to detect proper file encoding")
            return

        decoded_f = reopen(f, encoding)

        try:
            rkwargs = pr.merge([kwargs, {"pos": pos, "recursed": True}])

            for line in _read_any(decoded_f, reader, args, **rkwargs):
                yield line
        finally:
            decoded_f.close()


def read_any(filepath, reader, mode="r", *args, **kwargs):
    """Reads a file or filepath

    Args:
        filepath (str): The file path or file like object.
        reader (func): The processing function.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the reader.

    Kwargs:
        encoding (str): File encoding.
        last_row (int): Last row, use a negative value to count from the end
            (zero based, default: 0).

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
    if hasattr(filepath, "read"):
        if is_binary(filepath):
            kwargs.setdefault("encoding", ENCODING)
        else:
            kwargs.pop("encoding", None)

        for line in _read_any(filepath, reader, args, **kwargs):
            yield remove_bom(line, BOM)
    else:
        encoding = None if "b" in mode else kwargs.pop("encoding", ENCODING)

        with open(filepath, mode, encoding=encoding) as f:
            for line in _read_any(f, reader, args, **kwargs):
                yield remove_bom(line, BOM)


def _read_csv(f, header=None, has_header=True, first_col=0, **kwargs):
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
    if header and has_header:
        next(f)
    elif not (header or has_header):
        raise ValueError("Either `header` or `has_header` must be specified.")

    header = (list(it.repeat("", first_col)) + header) if first_col else header
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
    args = ["mdb-tables", "-1", filepath]

    # Check if 'mdb-tools' is installed on system
    try:
        check_output(args)
    except OSError:
        logger.error(
            "You must install [mdbtools]"
            "(http://sourceforge.net/projects/mdbtools/) in order to use "
            "this function"
        )
        yield
        return
    except CalledProcessError:
        raise TypeError("{} is not readable by mdbtools".format(filepath))

    sanitize = kwargs.pop("sanitize", None)
    dedupe = kwargs.pop("dedupe", False)
    table = table or check_output(args).splitlines()[0]
    pkwargs = {"stdout": PIPE, "bufsize": 1, "universal_newlines": True}

    # https://stackoverflow.com/a/2813530/408556
    # https://stackoverflow.com/a/17698359/408556
    with Popen(["mdb-export", filepath, table], **pkwargs).stdout as pipe:
        first_line = StringIO(str(pipe.readline()))
        names = next(csv.reader(first_line, **kwargs))
        uscored = ft.underscorify(names) if sanitize else names
        header = list(ft.dedupe(uscored) if dedupe else uscored)

        for line in iter(pipe.readline, b""):
            try:
                next_line = StringIO(str(line))
                values = next(csv.reader(next_line, **kwargs))
                yield dict(zip(header, values))
            except StopIteration:
                return None


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
    kwargs["lowernames"] = kwargs.pop("sanitize", None)
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
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")

    if not table or table not in set(cursor.fetchall()):
        table = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM {}".format(table))
    return map(dict, cursor)


def read_csv(filepath, mode="r", **kwargs):
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
        last_row (int): Last row, use a negative value to count from the end
            (zero based, default: 0).
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
        >>> expected = {
        ...     'sparse_data': 'Iñtërnâtiônàližætiøn',
        ...     'some_date': '05/04/82',
        ...     'some_value': '234',
        ...     'unicode_test': 'Ādam'}
        >>> next(records) == expected
        True
        >>> filepath = p.join(DATA_DIR, 'test.tsv')
        >>> records = read_csv(filepath, sanitize=True, dialect='excel-tab')
        >>> next(records) == expected
        True
    """

    def reader(f, **kwargs):
        """File reader"""
        first_row = kwargs.pop("first_row", 0)
        first_col = kwargs.pop("first_col", 0)
        sanitize = kwargs.pop("sanitize", False)
        dedupe = kwargs.pop("dedupe", False)
        has_header = kwargs.pop("has_header", True)
        custom_header = kwargs.pop("custom_header", None)

        # position file pointer at the first row
        [next(f) for _ in range(first_row)]

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
                msg = "Non seekable files must have either a specified or"
                msg += "custom header."
                logger.error(msg)
                raise

            [next(f) for _ in range(first_row)]

        if not (has_header or custom_header):
            header = ["column_%i" % (n + 1) for n in range(len(names))]

        return _read_csv(f, header, False, first_col=first_col, **kwargs)

    return read_any(filepath, reader, mode, **kwargs)


def read_tsv(filepath, mode="r", **kwargs):
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
        last_row (int): Last row, use a negative value to count from the end
            (zero based, default: 0).
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
    return read_csv(filepath, mode, dialect="excel-tab", **kwargs)


def read_fixed_fmt(filepath, widths=None, mode="r", **kwargs):
    """Reads a fixed-width csv file.

    Args:
        filepath (str): The fixed width formatted file path or file like object.
        widths (List[int]): The zero-based 'start' position of each column.
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments that are passed to the csv reader.

    Kwargs:
        has_header (bool): Has header row (default: False).
        first_row (int): First row (zero based, default: 0).
        last_row (int): Last row, use a negative value to count from the end
            (zero based, default: 0).
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
        """File reader"""
        sanitize = kwargs.get("sanitize")
        dedupe = kwargs.pop("dedupe", False)
        has_header = kwargs.get("has_header")
        first_row = kwargs.get("first_row", 0)
        schema = tuple(zip_longest(widths, widths[1:]))
        [next(f) for _ in range(first_row)]

        if has_header:
            line = next(f)
            names = (_f for _f in (line[s:e].strip() for s, e in schema) if _f)
            uscored = ft.underscorify(names) if sanitize else names
            header = list(ft.dedupe(uscored) if dedupe else uscored)
        else:
            header = ["column_%i" % (n + 1) for n in range(len(widths))]

        zipped = zip(header, schema)
        get_row = lambda line: {k: line[v[0] : v[1]].strip() for k, v in zipped}
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
    date_format = kwargs.get("date_format", "%Y-%m-%d")
    dt_format = kwargs.get("dt_format", "%Y-%m-%d %H:%M:%S")
    time_format = kwargs.get("time_format", "%H:%M:%S")

    def time_func(value):
        """Converts an excel time into python time"""
        args = xlrd.xldate_as_tuple(value, mode)[3:]
        return time(*args).strftime(time_format)

    switch = {
        XL_CELL_DATE: lambda v: xl2dt(v, mode).strftime(date_format),
        "datetime": lambda v: xl2dt(v, mode).strftime(dt_format),
        "time": time_func,
        XL_CELL_EMPTY: lambda v: "",
        XL_CELL_NUMBER: str,
        XL_CELL_BOOLEAN: lambda v: str(bool(v)),
        XL_CELL_ERROR: lambda v: xlrd.error_text_from_code[v],
    }

    for i in range(sheet.nrows):
        types = sheet.row_types(i)[first_col:]
        values = sheet.row_values(i)[first_col:]

        for _type, value in zip(types, values):
            if _type == XL_CELL_DATE and value < 1:
                _type = "time"
            elif _type == XL_CELL_DATE and not value.is_integer:
                _type = "datetime"

            yield (i, switch.get(_type, lambda v: v)(value))


# pylint: disable=unused-argument
def get_header(names, dedupe=False, sanitize=False, **kwargs):
    """Generates a header row"""
    stripped = (name for name in names if name.strip())
    uscored = ft.underscorify(stripped) if sanitize else stripped
    return list(ft.dedupe(uscored) if dedupe else uscored)


def read_xls(filepath, **kwargs):
    """Reads an xls/xlsx file.

    Args:
        filepath (str): The xls/xlsx file path, file, or SpooledTemporaryFile.
        kwargs (dict): Keyword arguments that are passed to the xls reader.

    Kwargs:
        sheet (int): Zero indexed sheet to open (default: 0)
        has_header (bool): Has header row (default: True).
        first_row (int): First row (zero based, default: 0).
        last_row (int): Last row, use a negative value to count from the end
            (zero based, default: 0).
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
    has_header = kwargs.get("has_header", True)
    first_row = kwargs.get("first_row", 0)

    xlrd_kwargs = {
        "on_demand": kwargs.get("on_demand"),
        "ragged_rows": not kwargs.get("pad_rows"),
        "encoding_override": kwargs.get("encoding", True),
    }

    try:
        contents = mmap(filepath.fileno(), 0)
        book = xlrd.open_workbook(file_contents=contents, **xlrd_kwargs)
    except AttributeError:
        book = xlrd.open_workbook(filepath, **xlrd_kwargs)

    sheet = book.sheet_by_index(kwargs.pop("sheet", 0))

    # Get header row and remove empty columns
    names = sheet.row_values(first_row)[kwargs.get("first_col", 0) :]

    if has_header:
        header = get_header(names, kwargs.pop("dedupe", False), **kwargs)
    else:
        header = ["column_%i" % (n + 1) for n in range(len(names))]

    # Convert to strings
    sanitized = sanitize_sheet(sheet, book.datemode, **kwargs)

    for key, group in it.groupby(sanitized, lambda v: v[0]):
        if has_header and key == first_row:
            continue

        values = [g[1] for g in group]

        # Remove empty rows
        if any(v and v.strip() for v in values):
            yield dict(zip(header, values))


def read_json(filepath, mode="r", path="item", newline=False):
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
    """Converts GeoJSON coordinates into a point tuple"""
    if lat_first:
        point = (coords[1], coords[0])
    else:
        point = (coords[0], coords[1])

    return point


def gen_records(_type, record, coords, properties, **kwargs):
    """GeoJSON record generator"""
    lat_first = kwargs.get("lat_first")

    if _type == "Point":
        record["lon"], record["lat"] = get_point(coords, lat_first)
        yield pr.merge([record, properties])
    elif _type == "LineString":
        for point in coords:
            record["lon"], record["lat"] = get_point(point, lat_first)
            yield pr.merge([record, properties])
    elif _type == "Polygon":
        for pos, poly in enumerate(coords):
            for point in poly:
                record["lon"], record["lat"] = get_point(point, lat_first)
                record["pos"] = pos
                yield pr.merge([record, properties])
    else:
        raise TypeError("Invalid geometry type {}.".format(_type))


def read_geojson(filepath, key="id", mode="r", **kwargs):
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
        """File reader"""
        try:
            features = items(f, "features.item")
        except KeyError:
            raise TypeError("Only GeoJSON with features are supported.")
        else:
            for feature in features:
                _type = feature["geometry"]["type"]
                properties = feature.get("properties") or {}
                coords = feature["geometry"]["coordinates"]
                record = {
                    "id": feature.get(key, properties.get(key)),
                    "type": feature["geometry"]["type"],
                }

                args = (record, coords, properties)

                for rec in gen_records(_type, *args, **kwargs):
                    yield rec

    return read_any(filepath, reader, mode, **kwargs)


def read_yaml(filepath, mode="r", **kwargs):
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


def get_text(element):
    if element and element.text:
        text = element.text.strip()
    else:
        text = ""

    if not text and element and element.string:
        text = element.string.strip()

    if not text and element and element.a:
        text = element.a.text or element.a.href or ""
        text = text.strip()

    return text


def _find_table(soup, pos=0):
    if pos:
        try:
            table = soup.find_all("table")[pos]
        except IndexError:
            table = None
    else:
        table = soup.table

    return table


def _gen_from_rows(rows, header, vertical=False):
    if vertical:
        # nested_tds = [('one', 'two'), ('uno', 'dos'), ('un', 'deux')]
        nested_tds = (tr.find_all("td") for tr in rows)

        # tds = ('one', 'uno', 'un')
        for tds in zip(*nested_tds):
            row = map(get_text, tds)
            yield dict(zip(header, row))
    else:
        for tr in rows:  # pylint: disable=C0103
            row = map(get_text, tr.find_all("td"))
            yield dict(zip(header, row))


def read_html(filepath, table=0, mode="r", **kwargs):
    """Reads tables from an html file

    TODO: convert to lxml.etree.iterparse
    http://lxml.de/parsing.html#iterparse-and-iterwalk

    Args:
        filepath (str): The html file path or file like object.
            If you have a url, use the following pattern:

            r = requests.get('http://example.com', stream=True)
            records = read_html(r.raw, sanitize=True)

        table (int): Zero indexed table to open (default: 0)
        mode (Optional[str]): The file open mode (default: 'r').
        kwargs (dict): Keyword arguments

    Kwargs:
        encoding (str): File encoding.

        sanitize (bool): Underscorify and lowercase field names
            (default: False).

        dedupe (bool): Deduplicate field names (default: False).
        vertical (bool): The table has headers in the left column (default:
            False).

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
        ...     'june': '10',
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
        """File reader"""
        try:
            soup = BeautifulSoup(f, "lxml-xml")
        except FeatureNotFound:
            soup = BeautifulSoup(f, "html.parser")

        sanitize = kwargs.get("sanitize")
        dedupe = kwargs.get("dedupe")
        vertical = kwargs.get("vertical")
        first_row_as_header = kwargs.get("first_row_as_header")
        tbl = _find_table(soup, table)

        if tbl:
            rows = tbl.find_all("tr")

            for num, a_row in enumerate(rows):
                if a_row.find("th"):
                    break

            ths = a_row.find_all("th")

            if first_row_as_header and not ths:
                ths = rows[0].find_all("td")

            if vertical or len(ths) == 1:
                # the headers are vertical instead of horizontal
                vertical = True
                names = (get_text(row.th) for row in rows)
            elif ths:
                rows = rows[1:]
                names = map(get_text, ths)
            else:
                col_nums = range(len(a_row))
                names = ["column_{}".format(i) for i in col_nums]

            uscored = ft.underscorify(names) if sanitize else names
            header = list(ft.dedupe(uscored) if dedupe else uscored)
            records = _gen_from_rows(rows, header, vertical)
        else:
            records = iter([])

        return records

    return read_any(filepath, reader, mode, **kwargs)


def write(filepath, content, mode="wb+", **kwargs):
    """Writes content to a file path or file like object.

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
        >>>
        >>> write(TemporaryFile(), StringIO('Hello World'))
        11
        >>> write(StringIO(), StringIO('Hello World'))
        11
        >>> content = IterStringIO(iter('Internationalization'))
        >>> write(StringIO(), content)
        20
        >>> content = IterStringIO(iter('Iñtërnâtiônàližætiøn'))
        >>> write(StringIO(), content)
        28
    """

    def writer(f, content, **kwargs):
        """File writer"""
        chunksize = kwargs.get("chunksize")
        length = int(kwargs.get("length") or 0)
        bar_len = kwargs.get("bar_len", 50)
        encoding = kwargs.get("encoding", ENCODING)
        progress = 0

        for chunk in ft.chunk(content, chunksize):
            text = ft.byte(chunk) if hasattr(chunk, "sort") else chunk

            try:
                f.write(text)
            except UnicodeEncodeError:
                f.write(text.encode(encoding))
            except TypeError:
                try:
                    f.write(text.decode(encoding))
                except AttributeError:
                    f.write(bytes(text, encoding))

            progress += chunksize or len(text)

            if length:
                bars = min(int(bar_len * progress / length), bar_len)
                logger.debug("\r[%s%s]", "=" * bars, " " * (bar_len - bars))
                sys.stdout.flush()

        yield progress

    return sum(read_any(filepath, writer, mode, content, **kwargs))


def hash_file(filepath, algo="sha1", chunksize=0, verbose=False):
    """Hashes a file path or file like object.
    https://stackoverflow.com/a/1131255/408556

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

    def writer(f, hasher, **kwargs):  # pylint: disable=W0613
        """File writer"""
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
    file_hash = next(read_any(filepath, writer, "rb", *args))

    if verbose:
        logger.debug("File %s hash is %s.", filepath, file_hash)

    return file_hash


def reencode(f, fromenc=ENCODING, toenc=ENCODING, **kwargs):
    """Reencodes a file from one encoding to another

    Args:
        f (obj): The file like object to convert.

    Kwargs:
        fromenc (str): The input encoding.
        toenc (str): The output encoding (default: ENCODING).
        remove_BOM (bool): Remove Byte Order Marker (default: True)

    Returns:
        obj: file like object of decoded strings

    Examples:
        >>> eff = p.join(DATA_DIR, 'utf16_big.csv')
        >>>
        >>> with open(eff, 'rb') as f:
        ...     encoded = reencode(f, 'utf-16-be', remove_BOM=True)
        ...     encoded.readline(keepends=False) == b'a,b,c'
        True
    """
    return Reencoder(f, fromenc, toenc, **kwargs)


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
        ...     result == {
        ...         'confidence': 0.99, 'language': '', 'encoding': 'utf-8'}
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
        logger.debug("result %s", detector.result)

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
        "csv": read_csv,
        "xls": read_xls,
        "xlsx": read_xls,
        "mdb": read_mdb,
        "json": read_json,
        "geojson": read_geojson,
        "geojson.json": read_geojson,
        "sqlite": read_sqlite,
        "dbf": read_dbf,
        "tsv": read_tsv,
        "yaml": read_yaml,
        "yml": read_yaml,
        "html": read_html,
        "fixed": read_fixed_fmt,
    }

    try:
        return switch[extension.lstrip(".").lower()]
    except IndexError:
        msg = "Reader for extension `{}` not found!"
        raise TypeError(msg.format(extension))


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
