#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.fntools
~~~~~~~~~~~~~~~~

Provides methods for functional manipulation of data from tabular formatted
files

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

from . import CURRENCIES, ENCODING

isempty = lambda x: x is None or x == ''


def mreplace(content, replacements):
    """ Performs multiple string replacements on content

    Args:
        content (str): the content to perform replacements on
        replacements (Iter[tuple(str)]): An iterable of `old`, `new` pairs

    Returns:
        (str): the replaced content

    Examples:
        >>> replacements = [('h', 't'), ('p', 'f')]
        >>> mreplace('happy', replacements)
        u'taffy'
    """
    func = lambda x, y: x.replace(y[0], y[1])
    return reduce(func, replacements, content)


def is_numeric_like(content, separators=('.', ',')):
    """ Determines whether or not content can be converted into a number

    Args:
        content (scalar): the content to analyze

    Kwargs:
        separators (tuple[str]): An iterable of characters that should be
            considered a thousand's separator (default: ('.', ','))

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
    replacements = it.izip(it.chain(CURRENCIES, separators), it.repeat(''))
    stripped = mreplace(content, replacements)

    try:
        float(stripped)
    except (ValueError, TypeError):
        return False
    else:
        return True


def byte(content):
    """ Creates a bytearray from a string or iterable of characters

    Args:
        content (Iter[str]): An iterable of characters

    Returns:
        (str): the replaced content

    Examples:
        >>> content = 'Hello World!'
        >>> byte(content)
        bytearray(b'Hello World!')
        >>> byte(list(content))
        bytearray(b'Hello World!')
        >>> content = 'Iñtërnâtiônàližætiøn'
        >>> byte(content) == bytearray(b'Iñtërnâtiônàližætiøn')
        True
        >>> byte(list(content)) == bytearray(b'Iñtërnâtiônàližætiøn')
        True
    """
    try:
        # like ['H', 'e', 'l', 'l', 'o']
        value = bytearray(content)
    except ValueError:
        # like ['I', '\xc3\xb1', 't', '\xc3\xab', 'r', 'n', '\xc3\xa2']
        value = reduce(lambda x, y: x + y, it.imap(bytearray, content))
    except TypeError:
        # like Hello
        # or [u'I', u'\xf1', u't', u'\xeb', u'r', u'n', u'\xe2']
        # or [u'H', u'e', u'l', u'l', u'o']
        value = reduce(
            lambda x, y: x + y,
            (bytearray(c, encoding=ENCODING) for c in content))

    return value


def merge_dicts(*dicts, **kwargs):
    """Merges a list of dicts. Optionally combines specified keys using a
    specified binary operator.

    http://codereview.stackexchange.com/a/85822/71049
    http://stackoverflow.com/a/31812635/408556
    http://stackoverflow.com/a/3936548/408556

    Args:
        dicts Iter[dict]: dicts to merge
        kwargs (dict): keyword arguments

    Kwargs:
        predicate (func): Receives a key and should return `True`
            if overlapping values should be combined. If a key occurs in
            multiple dicts and isn't combined, it will be overwritten
            by the last dict. Requires that `op` is set.

        op (func): Receives a list of 2 values from overlapping keys and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `predicate` is set. If a key is not
            present in all dicts, the value from `default` will be used. Note,
            since `op` applied inside of `reduce`, it may not perform as
            expected for all functions for more than 2 dicts. E.g. an average
            function will be applied as follows:

                ave([1, 2, 3]) --> ave([ave([1, 2]), 3])

            You would expect to get 2, but will instead get 2.25.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        (List[str]): collapsed content

    Examples:
        >>> dicts = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> predicate = lambda k: k == 'amount'
        >>> merge_dicts(*dicts, predicate=predicate, op=sum)
        {u'a': u'item', u'amount': 900}
        >>> merge_dicts(*dicts)
        {u'a': u'item', u'amount': 400}
        >>> items = merge_dicts({'a': 1, 'b': 2}, {'b': 10, 'c': 11}).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 10), (u'c', 11)]
        >>> dicts = [{'a':1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        >>>
        >>> # Combine all keys
        >>> predicate = lambda x: True
        >>> items = merge_dicts(*dicts, predicate=predicate, op=sum).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 6), (u'c', 8), (u'd', 6)]
        >>> fltrer = lambda x: x is not None
        >>> first = lambda x: filter(fltrer, x)[0]
        >>> kwargs = {'predicate': predicate, 'op': first, 'default': None}
        >>> items = merge_dicts(*dicts, **kwargs).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 2), (u'c', 3), (u'd', 6)]
        >>>
        >>> # This will only reliably give the expected result for 2 dicts
        >>> average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        >>> kwargs = {'predicate': predicate, 'op': average, 'default': None}
        >>> items = merge_dicts(*dicts, **kwargs).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 3.0), (u'c', 4.0), (u'd', 6.0)]
        >>>
        >>> # Only combine key 'b'
        >>> predicate = lambda k: k == 'b'
        >>> items = merge_dicts(*dicts, predicate=predicate, op=sum).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 6), (u'c', 5), (u'd', 6)]
        >>>
        >>> # This will reliably work for any number of dicts
        >>> from collections import defaultdict
        >>>
        >>> counted = defaultdict(int)
        >>> dicts = [
        ...    {'a': 1, 'b': 4, 'c': 0},
        ...    {'a': 2, 'b': 5, 'c': 2},
        ...    {'a': 3, 'b': 6, 'd': 7}]
        ...
        >>> for d in dicts:
        ...     for k in d.keys():
        ...         counted[k] += 1
        ...
        >>> sorted(counted.items())
        [(u'a', 3), (u'b', 3), (u'c', 2), (u'd', 1)]
        >>> predicate = lambda x: True
        >>> divide = lambda x: x[0] / x[1]
        >>> summed = merge_dicts(*dicts, predicate=predicate, op=sum)
        >>> sorted(summed.items())
        [(u'a', 6), (u'b', 15), (u'c', 2), (u'd', 7)]
        >>> kwargs = {'predicate': predicate, 'op': divide}
        >>> items = merge_dicts(summed, counted, **kwargs).items()
        >>> sorted(items)
        [(u'a', 2.0), (u'b', 5.0), (u'c', 1.0), (u'd', 7.0)]
    """
    predicate = kwargs.get('predicate')
    op = kwargs.get('op')
    default = kwargs.get('default', 0)

    def reducer(x, y):
        merge = lambda k, v: op([x.get(k, default), v]) if predicate(k) else v
        new_y = ([k, merge(k, v)] for k, v in y.iteritems())
        return dict(it.chain(x.iteritems(), new_y))

    if predicate and op:
        new_dict = reduce(reducer, dicts)
    else:
        new_dict = dict(it.chain.from_iterable(it.imap(dict.iteritems, dicts)))

    return new_dict


def chunk(content, chunksize=None, start=0, stop=None):
    """Groups data into fixed-sized chunks.
    http://stackoverflow.com/a/22919323/408556

    Args:
        content (obj): File like object, iterable response, or iterable.
        chunksize (Optional[int]): Number of bytes per chunk (default: 0,
            i.e., all).

        start (Optional[int]): Starting location (zero indexed, default: 0).
        stop (Optional[int]): Ending location (zero indexed).

    Returns:
        Iter[List]: Chunked content.

    Examples:
        >>> import requests
        >>> from StringIO import StringIO
        >>> from . import io
        >>> chunk([1, 2, 3, 4, 5, 6]).next()
        [1, 2, 3, 4, 5, 6]
        >>> chunk([1, 2, 3, 4, 5, 6], 2).next()
        [1, 2]
        >>> chunk(StringIO('Iñtërnâtiônàližætiøn'), 5).next() == u'Iñtër'
        True
        >>> chunk(StringIO('Hello World'), 5).next()
        u'Hello'
        >>> chunk(io.IterStringIO('Hello World'), 5).next()
        bytearray(b'Hello')
        >>> chunk(io.IterStringIO('Hello World')).next()
        bytearray(b'Hello World')
        >>> r = requests.get('http://google.com', stream=True)
        >>>
        >>> # http://docs.python-requests.org/en/latest/api/
        >>> # The chunk size is the number of bytes it should read into
        >>> # memory. This is not necessarily the length of each item returned
        >>> # as decoding can take place.
        >>> len(chunk(r.iter_content, 20, 29, 200).next()) > 0
        True
        >>> len(chunk(r.iter_content).next()) > 10000
        True
    """
    if hasattr(content, 'read'):  # it's a file
        content.seek(start) if start else None
        content.truncate(stop) if stop else None

        if chunksize:
            generator = (content.read(chunksize) for _ in it.count())
        else:
            generator = iter([content.read()])
    elif callable(content):  # it's an r.iter_content
        chunksize = chunksize or pow(2, 34)

        if start or stop:
            i = it.islice(content(), start, stop)
            generator = (byte(it.islice(i, chunksize)) for _ in it.count())
        else:
            generator = content(chunksize)
    else:  # it's a regular iterable
        i = it.islice(iter(content), start, stop)

        if chunksize:
            generator = (list(it.islice(i, chunksize)) for _ in it.count())
        else:
            generator = iter([list(i)])

    return it.takewhile(bool, generator)


def fill(prev_row, cur_row, **kwargs):
    """Fills in data of the current row with data from either a given
    value, the value of the same column in the previous row, or the value of a
    given column in the current row.

    Args:
        prev_row (dict): The previous row of data whose keys are the field
            names.

        cur_row (dict): The current row of data whose keys are the field names.
        kwargs (dict): Keyword arguments

    Kwargs:
        predicate (func): Receives a value and should return `True`
            if the value should be filled. If predicate is None, it returns
            `True` for empty values (default: None).
        value (str): Value to use to fill holes (default: None).
        fill_key (str): The column name of the current row to use for filling
            missing data.

        limit (int): Max number of consecutive rows to fill (default: None).
        fields (List[str]): Names of the columns to fill (default: None, i.e.,
            all).

        count (dict): The number of consecutive rows of missing data that have
            filled for each column.

    Yields:
        Tuple[str, str]: A tuple of (key, value).
        dict: The updated count.

    Examples:
        >>> from os import path as p
        >>> from . import io
        >>> parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
        >>> filepath = p.join(parent_dir, 'data', 'test', 'bad.csv')
        >>> records = io.read_csv(filepath, remove_header=True)
        >>> prev_row = {}
        >>> cur_row = records.next()
        >>> cur_row == {
        ...     u'column_a': u'1',
        ...     u'column_b': u'27',
        ...     u'column_c': u'',
        ... }
        True
        >>> length = len(cur_row)
        >>> filled = fill(prev_row, cur_row, value=0)
        >>> prev_row = dict(it.islice(filled, length))
        >>> count = filled.next()
        >>> count == {u'column_a': 0, u'column_b': 0, u'column_c': 1}
        True
        >>> prev_row == {
        ...     u'column_a': u'1',
        ...     u'column_b': u'27',
        ...     u'column_c': 0,
        ... }
        True
        >>> cur_row = records.next()
        >>> cur_row == {
        ...     u'column_a': u'',
        ...     u'column_b': u"I'm too short!",
        ...     u'column_c': None,
        ... }
        True
        >>> filled = fill(prev_row, cur_row, fill_key='column_b', count=count)
        >>> prev_row = dict(it.islice(filled, length))
        >>> count = filled.next()
        >>> count == {u'column_a': 1, u'column_b': 0, u'column_c': 2}
        True
        >>> prev_row == {
        ...     u'column_a': u"I'm too short!",
        ...     u'column_b': u"I'm too short!",
        ...     u'column_c': u"I'm too short!",
        ... }
        True
    """
    predicate = kwargs.get('predicate', isempty)
    value = kwargs.get('value')
    limit = kwargs.get('limit')
    fields = kwargs.get('fields')
    count = kwargs.get('count', {})
    fill_key = kwargs.get('fill_key')
    whitelist = set(fields or cur_row.keys())

    for key, entry in cur_row.items():
        key_count = count.get(key, 0)
        within_limit = key_count < limit if limit else True
        can_fill = (key in whitelist) and predicate(entry)
        count[key] = key_count + 1

        if can_fill and within_limit and value is not None:
            new_value = value
        elif can_fill and within_limit and fill_key:
            new_value = cur_row[fill_key]
        elif can_fill and within_limit:
            new_value = prev_row.get(key, entry)
        else:
            new_value = entry

        if not can_fill:
            count[key] = 0

        yield (key, new_value)

    yield count
