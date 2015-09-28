#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.fntools
~~~~~~~~~~~~~~~~

Provides basic functional methods

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


def byte(content):
    try:
        return bytearray(content)
    except ValueError:  # has unicode chars
        return reduce(lambda x, y: x + y, it.imap(bytearray, content))


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
        cfunc (func): Receives a key and should return `True`
            if overlapping values should be combined. If a key occurs in
            multiple dicts and isn't combined, it will be overwritten
            by the last dict. Requires that `op` is set.

        op (func): Receives a list of 2 values from overlapping keys and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `cfunc` is set. If a key is not present
            in all dicts, the value from `default` will be used. Note, since
            `op` applied inside of `reduce`, it may not perform as
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
        >>> cfunc = lambda k: k == 'amount'
        >>> merge_dicts(*dicts, cfunc=cfunc, op=sum)
        {u'a': u'item', u'amount': 900}
        >>> merge_dicts(*dicts)
        {u'a': u'item', u'amount': 400}
        >>> items = merge_dicts({'a': 1, 'b': 2}, {'b': 10, 'c': 11}).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 10), (u'c', 11)]
        >>> dicts = [{'a':1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        >>>
        >>> # Combine all keys
        >>> cfunc = lambda x: True
        >>> items = merge_dicts(*dicts, cfunc=cfunc, op=sum).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 6), (u'c', 8), (u'd', 6)]
        >>> fltrer = lambda x: x is not None
        >>> first = lambda x: filter(fltrer, x)[0]
        >>> kwargs = {'cfunc': cfunc, 'op': first, 'default': None}
        >>> items = merge_dicts(*dicts, **kwargs).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 2), (u'c', 3), (u'd', 6)]
        >>>
        >>> # This will only reliably give the expected result for 2 dicts
        >>> average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        >>> kwargs = {'cfunc': cfunc, 'op': average, 'default': None}
        >>> items = merge_dicts(*dicts, **kwargs).items()
        >>> sorted(items)
        [(u'a', 1), (u'b', 3.0), (u'c', 4.0), (u'd', 6.0)]
        >>>
        >>> # Only combine key 'b'
        >>> cfunc = lambda k: k == 'b'
        >>> items = merge_dicts(*dicts, cfunc=cfunc, op=sum).items()
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
        >>> cfunc = lambda x: True
        >>> divide = lambda x: x[0] / x[1]
        >>> summed = merge_dicts(*dicts, cfunc=cfunc, op=sum)
        >>> sorted(summed.items())
        [(u'a', 6), (u'b', 15), (u'c', 2), (u'd', 7)]
        >>> items = merge_dicts(summed, counted, cfunc=cfunc, op=divide).items()
        >>> sorted(items)
        [(u'a', 2.0), (u'b', 5.0), (u'c', 1.0), (u'd', 7.0)]
    """
    cfunc = kwargs.get('cfunc')
    op = kwargs.get('op')
    default = kwargs.get('default', 0)

    def reducer(x, y):
        merge = lambda k, v: op([x.get(k, default), v]) if cfunc(k) else v
        new_y = ([k, merge(k, v)] for k, v in y.iteritems())
        return dict(it.chain(x.iteritems(), new_y))

    if cfunc and op:
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
