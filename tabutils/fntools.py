#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tabutils.fntools
~~~~~~~~~~~~~~~~

Provides methods for functional manipulation of content

Examples:
    basic usage::

        from tabutils.fntools import underscorify

        header = ['ALL CAPS', 'Illegal $%^', 'Lots of space']
        underscored = list(underscorify(header))

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import itertools as it

from functools import partial
from slugify import slugify

from . import CURRENCIES, ENCODING


class Objectify(object):
    """Creates an object with dynamically set attributes. Useful
    for accessing the kwargs of a function as attributes.
    """
    def __init__(self, kwargs, **defaults):
        """ Objectify constructor
        Args:
            kwargs (dict): The attributes to set
            defaults (dict): The default attributes

        Examples:
            >>> kwargs = {'key_1': 1, 'key_2': 2}
            >>> defaults = {'key_2': 5, 'key_3': 3}
            >>> kw = Objectify(kwargs, **defaults)
            >>> kw.key_1
            1
            >>> kw.key_2
            2
            >>> kw.key_3
            3
            >>> kw.key_4
        """
        defaults.update(kwargs)
        self.__dict__.update(defaults)

    def __getattr__(self, name):
        return None


def isempty(content):
    """ Returns whether or not content is empty

    Args:
        content (scalar): the content to check

    Returns:
        (bool): True if content is empty

    Examples:
        >>> isempty(None)
        True
        >>> isempty('')
        True
        >>> isempty(False)
        False
        >>> isempty(' ')
        False
        >>> isempty('0')
        False
        >>> isempty(0)
        False
    """
    return content is None or content == ''


def underscorify(content):
    """ Slugifies elements of an array with underscores

    Args:
        content (Iter[str]): the content to clean

    Returns:
        (generator): the slugified content

    Examples:
        >>> list(underscorify(['ALL CAPS', 'Illegal $%^', 'Lots   of space']))
        [u'all_caps', u'illegal', u'lots_of_space']
    """
    return (slugify(item, separator='_') for item in content)


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
        content (Iter[char]): A string or iterable of characters

    Returns:
        (bytearray): A bytearray of the content

    Examples:
        >>> content = 'Hello World!'
        >>> byte(content)
        bytearray(b'Hello World!')
        >>> byte(list(content))
        bytearray(b'Hello World!')
        >>> byte(iter(content))
        bytearray(b'Hello World!')
        >>> content = 'Iñtërnâtiônàližætiøn'
        >>> byte(content) == bytearray(b'Iñtërnâtiônàližætiøn')
        True
        >>> byte(list(content)) == bytearray(b'Iñtërnâtiônàližætiøn')
        True
    """
    tupled = tuple(content) if hasattr(content, 'next') else content

    try:
        # encoded iterable like ['H', 'e', 'l', 'l', 'o']
        value = bytearray(tupled)
    except ValueError:
        # encoded iterable like ['I', '\xc3\xb1', 't', '\xc3\xab', 'r', 'n']
        value = reduce(lambda x, y: x + y, it.imap(bytearray, tupled))
    except TypeError:
        # unicode iterable like Hello
        # or [u'I', u'\xf1', u't', u'\xeb', u'r', u'n', u'\xe2']
        # or [u'H', u'e', u'l', u'l', u'o']
        bytefunc = partial(bytearray, encoding=ENCODING)
        value = reduce(lambda x, y: x + y, it.imap(bytefunc, tupled))

    return value


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


def xmlize(content):
    """ Recursively makes elements of an array xml compliant

    Args:
        content (Iter[str]): the content to clean

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
                yield mreplace(item, replacements) if item else ''


def guess_field_types(content):
    """Tries to determine field types based on field names.

    Args:
        content (Iter[str]): Field names.

    Yields:
        dict: Field type. The parsed field and its type.

    Examples:
        >>> fields = ['date', 'raw_value', 'date_and_time']
        >>> [t['type'] for t in guess_field_types(fields)]
        [u'date', u'float', u'datetime', u'date']
    """
    for item in content:
        if 'date' in item and 'time' in item:
            yield {'id': item, 'type': 'datetime'}
        if 'date' in item:
            yield {'id': item, 'type': 'date'}
        elif 'time' in item:
            yield {'id': item, 'type': 'time'}
        elif find(['value', 'length', 'width', 'days'], [item], method='fuzzy'):
            yield {'id': item, 'type': 'float'}
        elif 'count' in item:
            yield {'id': item, 'type': 'int'}
        else:
            yield {'id': item, 'type': 'text'}


def afterish(content, separator=',', exclude=None):
    """Calculates the number of digits after a given separator.

    Args:
        content (str): Field names.

    Kwargs:
        separator (char): Character to start counting from (default: ',').
        exclude (char): Character to ignore from the calculation (default: '').

    Yields:
        dict: Field type. The parsed field and its type.

    Examples:
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
    numeric_like = is_numeric_like(content)

    if numeric_like and separator in content:
        excluded = [s for s in content.split(exclude) if separator in s][0]
        after = len(excluded) - excluded.rfind(separator) - 1
    elif numeric_like:
        after = -1
    else:
        raise TypeError('Not able to convert %s to a number' % content)

    return after


def get_separators(content):
    """Guesses the appropriate thousandths and decimal separators

    Args:
        content (str): The string to parse.

    Examples:
        >>> get_separators('$123.45')
        {u'thousand_sep': u',', u'decimal_sep': u'.'}
        >>> get_separators('123€')
        {u'thousand_sep': u',', u'decimal_sep': u'.'}
        >>> get_separators('2,123.45')
        {u'thousand_sep': u',', u'decimal_sep': u'.'}
        >>> get_separators('2.123,45')
        {u'thousand_sep': u'.', u'decimal_sep': u','}
        >>> get_separators('spam')
        Traceback (most recent call last):
        TypeError: Not able to convert spam to a number

    Returns:
        dict: thousandths and decimal separators
    """
    try:
        after_comma = afterish(content, exclude='.')
        after_decimal = afterish(content, '.', ',')
    except AttributeError:
        # We don't have a string
        after_comma = 0
        after_decimal = 0

    if after_comma in {-1, 0, 3} and after_decimal in {-1, 0, 1, 2}:
        thousand_sep, decimal_sep = ',', '.'
    elif after_comma in {-1, 0, 1, 2} and after_decimal in {-1, 0, 3}:
        thousand_sep, decimal_sep = '.', ','
    else:
        print('after_comma', after_comma)
        print('after_decimal', after_decimal)
        raise TypeError('Invalid number format for `%s`.' % content)

    return {'thousand_sep': thousand_sep, 'decimal_sep': decimal_sep}


def add_ordinal(num):
    """ Returns a number with ordinal suffix, e.g., 1st, 2nd, 3rd.

    Args:
        num (int): a number

    Returns:
        (str): ext a number with the ordinal suffix

    Examples:
        >>> add_ordinal(11)
        u'11th'
        >>> add_ordinal(132)
        u'132nd'
    """
    switch = {1: 'st', 2: 'nd', 3: 'rd'}
    end = 'th' if (num % 100 in {11, 12, 13}) else switch.get(num % 10, 'th')
    return '%i%s' % (num, end)


def _fuzzy_match(needle, haystack, **kwargs):
    for n in needle:
        for h in haystack:
            if n in h.lower():
                yield h


def _exact_match(*args, **kwargs):
    sets = (set(i.lower() for i in arg) for arg in args)
    return iter(reduce(lambda x, y: x.intersection(y), sets))


def find(*args, **kwargs):
    """ Determines if there is any overlap between lists of words

    Args:
        args (Iter[str]): Arguments passed to the search function
        kwargs (dict): Keyword arguments passed to the search function

    Kwargs:
        method (str or func):
        default (scalar):

    Returns:
        (str): the replaced content

    Examples:
        >>> needle = ['value', 'length', 'width', 'days']
        >>> haystack = ['num_days', 'my_value']
        >>> find(needle, haystack, method='fuzzy')
        u'my_value'
        >>> find(needle, haystack)
        u''
        >>> find(needle, ['num_days', 'width'])
        u'width'
    """
    method = kwargs.pop('method', 'exact')
    default = kwargs.pop('default', '')
    funcs = {'exact': _exact_match, 'fuzzy': _fuzzy_match}
    func = funcs.get(method, method)

    try:
        return func(*args, **kwargs).next()
    except StopIteration:
        return default


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
        >>> records = io.read_csv(filepath)
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


def combine(x, y, key, value=None, predicate=None, op=None, default=0):
    """Applies a binary operator to the value of an entry in two `records`.

    Args:
        x (dict): First record. Row of data whose keys are the field names.
            E.g., result from from calling next() on the output of any
            `tabutils.io` read function.

        y (dict): Second record. Row of data whose keys are the field names.
            E.g., result from from calling next() on the output of any
            `tabutils.io` read function.

        key (str): Current key.
        value (Optional[scalar]): The 2nd record's value of the given `key`.

        predicate (func): Receives `key` and should return `True`
            if the values from both records should be combined. Can optionally
            be a keyfunc which receives the 2nd record and should return the
            value that `value` needs to equal in order to be combined.

            If `key` occurs in both records and isn't combined, it will be
            overwritten by the 2nd record. Requires that `op` is set.

        op (func): Receives a list of the 2 values from the records and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `predicate` is set. If a key is not
            present in a record, the value from `default` will be used.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        (scalar): the combined value

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> predicate = lambda key: key == 'amount'
        >>> x, y = records[0], records[1]
        >>> combine(x, y, 'a', predicate=predicate, op=sum)
        u'item'
        >>> combine(x, y, 'amount', predicate=predicate, op=sum)
        500
        >>> records = [{'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        >>>
        >>> # Combine all keys
        >>> predicate = lambda key: True
        >>> x, y = records[0], records[1]
        >>> combine(x, y, 'a', predicate=predicate, op=sum)
        1
        >>> combine(x, y, 'b', predicate=predicate, op=sum)
        6
        >>> combine(x, y, 'c', predicate=predicate, op=sum)
        8
        >>> fltrer = lambda x: x is not None
        >>> first = lambda x: filter(fltrer, x)[0]
        >>> kwargs = {'predicate': predicate, 'op': first, 'default': None}
        >>> combine(x, y, 'b', **kwargs)
        2
        >>>
        >>> average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        >>> kwargs = {'predicate': predicate, 'op': average, 'default': None}
        >>> combine(x, y, 'a', **kwargs)
        1.0
        >>> combine(x, y, 'b', **kwargs)
        3.0
        >>>
        >>> # Only combine key 'b'
        >>> predicate = lambda key: key == 'b'
        >>> combine(x, y, 'c', predicate=predicate, op=sum)
        5
        >>>
        >>> # Only combine keys that have the same value of 'b'
        >>> from operator import itemgetter
        >>> predicate = itemgetter('b')
        >>> combine(x, y, 'b', predicate=predicate, op=sum)
        6
        >>> combine(x, y, 'c', predicate=predicate, op=sum)
        5
    """
    value = y.get(key, default) if value is None else value

    try:
        passed = predicate(key)
    except TypeError:
        passed = predicate(y) == value

    return op([x.get(key, default), value]) if passed else value
