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
        list(underscorify(header))

Attributes:
    DEF_TRUES (tuple[str]): Values to be consider True
    DEF_FALSES (tuple[str]): Values to be consider False
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import itertools as it
import operator

from functools import partial
from collections import defaultdict
from json import JSONEncoder

from slugify import slugify
from . import CURRENCIES, ENCODING

DEF_TRUES = ('yes', 'y', 'true', 't')
DEF_FALSES = ('no', 'n', 'false', 'f')


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

    def __iter__(self):
        return self.__dict__.itervalues()

    def __getattr__(self, name):
        return None

    def iteritems(self):
        return self.__dict__.iteritems()


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'real'):
            encoded = float(obj)
        elif set(['quantize', 'year', 'hour']).intersection(dir(obj)):
            encoded = str(obj)
        elif hasattr(obj, 'union'):
            encoded = tuple(obj)
        elif set(['next', 'union']).intersection(dir(obj)):
            encoded = list(obj)
        else:
            encoded = super(CustomEncoder, self).default(obj)

        return encoded


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


def stringify(content):
    """ Converts unicode elements of an array into strings

    Args:
        content (Iter[str]): the content to clean

    Returns:
        (generator): the stringified content

    Examples:
        >>> stringified = stringify([unicode('hi'), u'world', 0])
        >>> map(type, stringified) == [str, str, int]
        True
    """
    return (str(c) if isinstance(c, unicode) else c for c in content)


def dedupe(content):
    """ Deduplicates elements of an array

    Args:
        content (Iter[str]): the content to dedupe

    Returns:
        (generator): the deduped content

    Examples:
        >>> list(dedupe(['field', 'field', 'field']))
        [u'field', u'field_2', u'field_3']
    """
    seen = defaultdict(int)

    for f in content:
        new_field = '%s_%i' % (f, seen[f] + 1) if f in seen else f
        seen[f] += 1
        yield new_field


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


def strip(value, thousand_sep=',', decimal_sep='.'):
    """Strips a string of all non-numeric characters.

    Args:
        value (str): The string to parse.
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')

    Examples:
        >>> strip('$123.45')
        u'123.45'
        >>> strip('123€')
        u'123'

    Returns:
        str
    """
    currencies = it.izip(CURRENCIES, it.repeat(''))
    separators = [(thousand_sep, ''), (decimal_sep, '.')]

    try:
        stripped = mreplace(value, it.chain(currencies, separators))
    except AttributeError:
        stripped = value  # We don't have a string

    return stripped


def is_numeric(content, thousand_sep=',', decimal_sep='.', **kwargs):
    """ Determines whether or not content can be converted into a number

    Args:
        content (scalar): the content to analyze
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')
        kwargs (dict): Keyword arguments passed to the search function

    Kwargs:
        strip_zeros (bool): Remove leading zeros (default: False)

    >>> is_numeric('$123.45')
    True
    >>> is_numeric('123€')
    True
    >>> is_numeric(0)
    True
    >>> is_numeric('0.1')
    True
    """
    try:
        stripped = strip(content, thousand_sep, decimal_sep)
    except TypeError:
        stripped = content

    try:
        floated = float(stripped)
    except (ValueError, TypeError):
        passed = False
    else:
        s = str(stripped)
        zero_point = s.startswith('0.')
        passed = bool(floated) or zero_point

        if s.startswith('0') and not (kwargs.get('strip_zeros') or zero_point):
            passed = int(content) == 0

    return passed


def is_int(content, strip_zeros=False, thousand_sep=',', decimal_sep='.'):
    """ Determines whether or not content can be converted into an int

    Args:
        content (scalar): the content to analyze
        strip_zeros (bool): Remove leading zeros (default: False)
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')


    Examples:
        >>> is_int('$123.45')
        False
        >>> is_int('123')
        True
    """
    passed = is_numeric(content, thousand_sep, decimal_sep)

    try:
        stripped = strip(content, thousand_sep, decimal_sep)
    except TypeError:
        stripped = content

    return passed and float(stripped).is_integer()


def is_bool(content, trues=None, falses=None):
    """ Determines whether or not content can be converted into a bool

    Args:
        content (scalar): the content to analyze
        trues (Seq[str]): Values to consider True.
        falses (Seq[str]): Values to consider Frue.

    Examples:
        >>> is_bool(True)
        True
        >>> is_bool('true')
        True
    """
    trues = set(map(str.lower, trues) if trues else DEF_TRUES)
    falses = set(map(str.lower, falses) if falses else DEF_FALSES)

    try:
        passed = content.lower() in trues.union(falses)
    except AttributeError:
        passed = content in {True, False}

    return passed


def is_null(content, nulls=None, blanks_as_nulls=False):
    """ Determines whether or not content can be converted into a null

    Args:
        content (scalar): the content to analyze
        nulls (Seq[str]): Values to consider null.
        blanks_as_nulls (bool): Treat empty strings as null (default: False).

    Examples:
        >>> is_null('n/a')
        True
        >>> is_null(None)
        True
    """
    def_nulls = ('na', 'n/a', 'none', 'null', '.')
    nulls = set(map(str.lower, nulls) if nulls else def_nulls)

    try:
        passed = content.lower() in nulls
    except AttributeError:
        passed = content is None

    try:
        if not (passed or content.strip()):
            passed = blanks_as_nulls
    except AttributeError:
        pass

    return passed


def dfilter(content, blacklist=None, inverse=False):
    """ Filters content

    Args:
        content (dict): The content to filter
        blacklist (Seq[str]): The fields to remove (default: None)
        inverse (bool): Keep fields instead of removing them (default: False)

    Returns:
        dict: The filtered content

    Examples:
        >>> content = {'keep': 'Hello', 'strip': 'World'}
        >>> dfilter(content) == {'keep': 'Hello', 'strip': 'World'}
        True
        >>> dfilter(content, ['strip'])
        {u'keep': u'Hello'}
        >>> dfilter(content, ['strip'], True)
        {u'strip': u'World'}
    """
    blackset = set(blacklist or [])
    func = it.ifilterfalse if inverse else filter
    return dict(func(lambda x: x[0] not in blackset, content.items()))


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
        >>> chunk([1, 2, 3, 4, 5, 6]).next()
        [1, 2, 3, 4, 5, 6]
        >>> chunk([1, 2, 3, 4, 5, 6], 2).next()
        [1, 2]
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


def afterish(content, separator=',', exclude=None):
    """Calculates the number of digits after a given separator.

    Args:
        content (str): Field names.
        separator (char): Character to start counting from (default: ',').
        exclude (char): Character to ignore from the calculation (default: '').

    Yields:
        dict: Field type. The parsed field and its type.

    Examples:
        >>> afterish('123.45', '.')
        2
        >>> afterish('1001.', '.')
        0
    """
    numeric = is_numeric(content)

    if numeric and separator in content:
        excluded = [s for s in content.split(exclude) if separator in s][0]
        after = len(excluded) - excluded.rfind(separator) - 1
    elif numeric:
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
        (str): a number with the ordinal suffix

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


def fill(previous, current, **kwargs):
    """Fills in data of the current record with data from either a given
    value, the value of the same column in the previous record, or the value of
    a given column in the current record.

    Args:
        previous (dict): The previous record of data whose keys are the
            field names.

        current (dict): The current record of data whose keys are the field
            names.

        kwargs (dict): Keyword arguments

    Kwargs:
        pred (func): Receives a value and should return `True`
            if the value should be filled. If pred is None, it returns
            `True` for empty values (default: None).

        value (str): Value to use to fill holes (default: None).
        fill_key (str): The column name of the current record to use for
            filling missing data.

        limit (int): Max number of consecutive records to fill (default: None).

        fields (Seq[str]): Names of the columns to fill (default: None, i.e.,
            all).

        count (dict): The number of consecutive records of missing data that
            have filled for each column.

        blanks_as_nulls (bool): Treat empty strings as null (default: True).

    Yields:
        Tuple[str, str]: A tuple of (key, value).
        dict: The updated count.

    See also:
        `process.fillempty`

    Examples:
        >>> previous = {}
        >>> current = {
        ...     u'column_a': u'1',
        ...     u'column_b': u'27',
        ...     u'column_c': u'',
        ... }
        >>> length = len(current)
        >>> filled = fill(previous, current, value=0)
        >>> dict(it.islice(filled, length)) == {
        ...     u'column_a': u'1',
        ...     u'column_b': u'27',
        ...     u'column_c': 0,
        ... }
        True
        >>> filled.next() == {u'column_a': 0, u'column_b': 0, u'column_c': 1}
        True
    """
    pkwargs = {'blanks_as_nulls': kwargs.get('blanks_as_nulls', True)}
    def_pred = partial(is_null, **pkwargs)
    predicate = kwargs.get('pred', def_pred)
    value = kwargs.get('value')
    limit = kwargs.get('limit')
    fields = kwargs.get('fields')
    count = kwargs.get('count', {})
    fill_key = kwargs.get('fill_key')
    whitelist = set(fields or current.keys())

    for key, entry in current.items():
        key_count = count.get(key, 0)
        within_limit = key_count < limit if limit else True
        can_fill = (key in whitelist) and predicate(entry)
        count[key] = key_count + 1

        if can_fill and within_limit and value is not None:
            new_value = value
        elif can_fill and within_limit and fill_key:
            new_value = current[fill_key]
        elif can_fill and within_limit:
            new_value = previous.get(key, entry)
        else:
            new_value = entry

        if not can_fill:
            count[key] = 0

        yield (key, new_value)

    yield count


def combine(x, y, key, value=None, pred=None, op=None, default=0):
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

        pred (func): Receives `key` and should return `True`
            if the values from both records should be combined. Can optionally
            be a keyfunc which receives the 2nd record and should return the
            value that `value` needs to equal in order to be combined.

            If `key` occurs in both records and isn't combined, it will be
            overwritten by the 2nd record. Requires that `op` is set.

        op (func): Receives a list of the 2 values from the records and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `pred` is set. If a key is not
            present in a record, the value from `default` will be used.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        (scalar): the combined value

    See also:
        `process.merge`

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> pred = lambda key: key == 'amount'
        >>> x, y = records[0], records[1]
        >>> combine(x, y, 'a', pred=pred, op=sum)
        u'item'
        >>> combine(x, y, 'amount', pred=pred, op=sum)
        500
    """
    value = y.get(key, default) if value is None else value

    try:
        passed = pred(key)
    except TypeError:
        passed = pred(y) == value

    return op([x.get(key, default), value]) if passed else value


def flatten(record, prefix=None):
    """Recursively flattens a nested record by pre-pending the parent field
    name to the children field names.

    Args:
        record (dict): The record to flattens whose keys are the field
            names.

        prefix (str): String to prepend to all children (default: None)

    Yields:
        Tuple[str, scalar]: A tuple of (key, value).

    Examples:
        >>> record = {
        ...     'parent_a': {'child_1': 1, 'child_2': 2, 'child_3': 3},
        ...     'parent_b': {'child_1': 1, 'child_2': 2, 'child_3': 3},
        ...     'parent_c': 'no child',
        ... }
        >>> dict(flatten(record)) == {
        ...     u'parent_a_child_1': 1,
        ...     u'parent_a_child_2': 2,
        ...     u'parent_a_child_3': 3,
        ...     u'parent_b_child_1': 1,
        ...     u'parent_b_child_2': 2,
        ...     u'parent_b_child_3': 3,
        ...     u'parent_c': u'no child',
        ... }
        ...
        True
        >>> dict(flatten(record, 'flt')) == {
        ...     u'flt_parent_a_child_1': 1,
        ...     u'flt_parent_a_child_2': 2,
        ...     u'flt_parent_a_child_3': 3,
        ...     u'flt_parent_b_child_1': 1,
        ...     u'flt_parent_b_child_2': 2,
        ...     u'flt_parent_b_child_3': 3,
        ...     u'flt_parent_c': u'no child',
        ... }
        True
    """
    try:
        for key, value in record.items():
            newkey = '%s_%s' % (prefix, key) if prefix else key

            for flattened in flatten(value, newkey):
                yield flattened
    except AttributeError:
        yield (prefix, record)


def array_search_type(needle, haystack, n=0):
    """ Searches an array for the nth (zero based) occurrence of a given value
     type and returns the corresponding key if successful.

        Args:
            needle (str): the type of element to find (i.e. 'numeric'
                or 'string')
            haystack (List[str]): the array to search

        Returns:
            (List[str]): array of the key(s) of the found element(s)

        Examples:
            >>> array_search_type('string', ('one', '2w', '3a'), 2).next()
            u'3a'
            >>> array_search_type('numeric', ('1', 2, 3), 2).next()
            Traceback (most recent call last):
            StopIteration
            >>> array_search_type('numeric', ('one', 2, 3), 1).next()
            3
    """
    switch = {'numeric': 'real', 'string': 'upper'}
    func = lambda x: hasattr(x, switch[needle])
    return it.islice(it.ifilter(func, haystack), n, None)


def array_substitute(content, needle, replace):
    """ Recursively replaces all occurrences of needle with replace

    Args:
        content (List[str]): the array to perform the replacement on
        needle (str): the value being searched for (an array may
            be used to designate multiple needles)

        replace (scalar): the replacement value that replaces needle
            (an array may be used to designate multiple replacements)

    Returns:
        List[str]: new array with replaced values

    Examples:
        >>> array_substitute([('one', 'two', 'three')], 'two', 2).next()
        [u'one', u'2', u'three']
    """
    for item in content:
        try:
            yield item.replace(needle, str(replace))
        except AttributeError:
            yield list(array_substitute(item, needle, replace))


def op_everseen(iterable, key=None, pad=False, op='lt'):
    """List min/max/equal... elements, preserving order. Remember all
    elements ever seen.
    >>> from operator import itemgetter
    >>> list(op_everseen([4, 6, 3, 8, 2, 1]))
    [4, 3, 2, 1]
    >>> list(op_everseen([('a', 6), ('b', 4), ('c', 8)], itemgetter(1)))
    [(u'a', 6), (u'b', 4)]
    """
    current = None
    current_key = None
    compare = getattr(operator, op)

    for element in iterable:
        k = element if key is None else key(element)

        if current is None:
            current = element
            current_key = k
            valid = True
        else:
            valid = compare(k, current_key)
            current = element if valid else current
            current_key = k if valid else current_key

        if valid or pad:
            yield current
