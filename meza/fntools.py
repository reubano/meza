#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.fntools
~~~~~~~~~~~~

Provides methods for functional manipulation of content

Examples:
    basic usage::

        >>> from meza.fntools import underscorify
        >>>
        >>> header = ['ALL CAPS', 'Illegal $%^', 'Lots of space']
        >>> result = {'all_caps', 'illegal', 'lots_of_space'}
        >>> set(underscorify(header)) == result
        True

Attributes:
    DEF_TRUES (tuple[str]): Values to be consider True
    DEF_FALSES (tuple[str]): Values to be consider False
    ARRAY_TYPE (dict): Python to array.array type lookup table
    NP_TYPE (dict): Python to numpy type lookup table
    DB_TYPE (dict): Python to postgres type lookup table
    SQLITE_TYPE (dict): Python to sqlite type lookup table
    ARRAY_NULL_TYPE (dict): None to array.array type lookup table
"""
import sys
import itertools as it
import operator
import time

from functools import partial, reduce
from collections import defaultdict
from json import JSONEncoder
from os import path as p
from itertools import zip_longest, filterfalse

import pygogo as gogo

from slugify import slugify

from . import CURRENCIES, ENCODING
from .compat import encode

DEF_TRUES = ("yes", "y", "true", "t")
DEF_FALSES = ("no", "n", "false", "f")

NP_TYPE = {
    "null": "bool",
    "bool": "bool",
    "int": "i",
    "float": "f",
    "double": "d",
    "decimal": "d",
    "datetime": "datetime64[us]",
    "time": "timedelta64[us]",
    "date": "datetime64[D]",
    "text": "object_",
}

ARRAY_TYPE = {
    "null": "B",
    "bool": "B",
    "int": "i",
    "float": "f",
    "double": "d",
    "decimal": "d",
    "text": "u",
}

POSTGRES_TYPE = {
    "null": "boolean",
    "bool": "boolean",
    "int": "integer",
    "float": "real",
    "double": "double precision",
    "decimal": "decimal",
    "datetime": "timestamp",
    "time": "time",
    "date": "date",
    "text": "text",
}

MYSQL_TYPE = {
    "null": "CHAR(0)",
    "bool": "BOOL",
    "int": "INT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "decimal": "DECIMAL",
    "datetime": "DATETIME",
    "time": "TIME",
    "date": "DATE",
    "text": "TEXT",
}

SQLITE_TYPE = {
    "null": "INT",
    "bool": "INT",
    "int": "INT",
    "float": "REAL",
    "double": "REAL",
    "decimal": "REAL",
    "datetime": "TEXT",
    "time": "TEXT",
    "date": "TEXT",
    "text": "TEXT",
}

ARRAY_NULL_TYPE = {"B": False, "i": 0, "f": 0.0, "d": 0.0, "u": ""}

try:
    MAXINT = sys.maxint  # pylint: disable=sys-max-int
except AttributeError:
    MAXINT = sys.maxsize

logger = gogo.Gogo(__name__, monolog=True).logger


class Objectify(object):
    """Creates an object with dynamically set attributes. Useful
    for accessing the kwargs of a function as attributes.
    """

    def __init__(self, kwargs, func=None, **defaults):
        """Objectify constructor

        Args:
            kwargs (dict): The attributes to set
            defaults (dict): The default attributes

        Examples:
            >>> kwargs = {'key_1': 1, 'key_2': 2}
            >>> defaults = {'key_2': 5, 'key_3': 3}
            >>> kw = Objectify(kwargs, **defaults)
            >>> sorted(kw) == ['key_1', 'key_2', 'key_3']
            True
            >>> dict(kw) == {'key_1': 1, 'key_2': 2, 'key_3': 3}
            True
            >>> kw.key_1
            1
            >>> kw['key_2']
            2
            >>> kw.get('key_3')
            3
            >>> kw.key_4
            >>> kw.get('key_4')
            >>> kw['key_4'] = 4
            >>> kw.key_4 == kw.get('key_4') == kw['key_4'] == 4
            True
            >>> kw.key_4 = 5
            >>> kw.key_4 == kw.get('key_4') == kw['key_4'] == 5
            True
        """
        defaults.update(kwargs)
        self.data = defaults
        self.func = func
        self.keys = self.data.keys
        self.values = self.data.values
        self.items = self.data.items
        self.get = self.data.get

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, key):
        value = self.get(key)
        return self.func(value) if self.func else value

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __setattr__(self, key, value):
        if key not in {"data", "func", "keys", "values", "items", "get"}:
            self.data.__setitem__(key, value)

        return super(Objectify, self).__setattr__(key, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __delitem__(self, key):
        return self.data.__delitem__(key)

    def __delattr__(self, key):
        return self.__delitem__(key)

    def __iter__(self):
        return iter(self.data)

    def iteritems(self):
        return iter(self.items())


class Andand(object):
    """A Ruby inspired null soaking object

    Examples:
        >>> kwargs = {'key': 'value'}
        >>> kw = Objectify(kwargs)
        >>> kw.key == 'value'
        True
        >>> Andand(kw).key  # doctest: +ELLIPSIS
        <meza.fntools.Andand object at 0x...>
        >>> Andand(kw).key.item == 'value'
        True
        >>> Andand(kw).key() == 'value'
        True
        >>> Andand(kw).key.imnot.here  # doctest: +ELLIPSIS
        <meza.fntools.Andand object at 0x...>
        >>> Andand(kw).key.imnot.here.item
        >>> Andand(kw).key.imnot.here()
    """

    def __init__(self, item=None):
        self.item = item

    def __getattr__(self, name):
        try:
            item = getattr(self.item, name)
            return item if name == "item" else Andand(item)
        except AttributeError:
            return Andand()

    def __call__(self):
        return self.item


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "real"):
            encoded = float(obj)
        elif hasattr(obj, "to_dict"):
            encoded = obj.to_dict()
        elif set(["quantize", "year", "hour"]).intersection(dir(obj)):
            encoded = str(obj)
        elif hasattr(obj, "union"):
            encoded = tuple(obj)
        elif set(["next", "append"]).intersection(dir(obj)):
            encoded = list(obj)
        else:
            encoded = super(CustomEncoder, self).default(obj)

        return encoded


class SleepyDict(dict):
    """A dict like object that sleeps for a specified amount of time before
    returning a key or during truth value testing
    """

    def __init__(self, *args, **kwargs):
        self.delay = kwargs.pop("delay", 0)
        super(SleepyDict, self).__init__(*args, **kwargs)

    def __len__(self):
        time.sleep(self.delay)
        return super(SleepyDict, self).__len__()

    def get(self, key, default=None):
        time.sleep(self.delay)
        return super(SleepyDict, self).get(key, default)


def underscorify(content):
    """Slugifies elements of an array with underscores

    Args:
        content (Iter[str]): the content to clean

    Returns:
        (generator): the slugified content

    Examples:
        >>> _ = underscorify(['ALL CAPS', 'Illegal $%^', 'Lots   of space'])
        >>> list(_) == ['all_caps', 'illegal', 'lots_of_space']
        True
    """
    for item in content:
        try:
            yield slugify(item, separator="_")
        except TypeError:
            yield slugify(item.encode(ENCODING), separator="_")


def get_ext(path):
    """Gets a file (local)

    Args:
        content (Iter[str]): the content to dedupe

    Returns:
        (generator): the deduped content

    Examples:
        >>> get_ext('file.csv') == 'csv'
        True
    """
    if "format=" in path:
        file_format = path.lower().split("format=")[1]

        if "&" in file_format:
            file_format = file_format.split("&")[0]
    else:
        file_format = p.splitext(path)[1].lstrip(".")

    return file_format


def get_dtype(_type, dialect="array"):
    switch = {
        "numpy": NP_TYPE,
        "array": ARRAY_TYPE,
        "postgres": POSTGRES_TYPE,
        "mysql": MYSQL_TYPE,
        "sqlite": SQLITE_TYPE,
    }

    converter = switch[dialect]
    return converter.get(_type, converter["text"])


def dedupe(content):
    """Deduplicates elements of an array

    Args:
        content (Iter[str]): the content to dedupe

    Returns:
        (generator): the deduped content

    Examples:
        >>> list(dedupe(['field', 'field', 'field'])) == [
        ...     'field', 'field_2', 'field_3']
        True
    """
    seen = defaultdict(int)

    for f in content:
        new_field = "%s_%i" % (f, seen[f] + 1) if f in seen else f
        seen[f] += 1
        yield new_field


def mreplace(content, replacements):
    """Performs multiple string replacements on content

    Args:
        content (str): the content to perform replacements on
        replacements (Iter[tuple(str)]): An iterable of `old`, `new` pairs

    Returns:
        (str): the replaced content

    Examples:
        >>> replacements = [('h', 't'), ('p', 'f')]
        >>> mreplace('happy', replacements) == 'taffy'
        True
    """
    func = lambda x, y: x.replace(*y)
    return reduce(func, replacements, content)


def rreplace(content, needle, replace):
    """Recursively replaces all occurrences of needle with replace

    Args:
        content (Iter[str]): An iterable of strings on which to perform the
            replacement

        needle (str): the value being searched for (an iterable of strings may
            be used to designate multiple needles)

        replace (scalar): the replacement value that replaces needle (an
            iterable of scalars may be used to designate multiple replacements)

    Yields:
        str: replaced content

    Examples:
        >>> subs = rreplace([('one', 'two', 'three')], 'two', 2)
        >>> next(subs) == ['one', '2', 'three']
        True
    """
    for item in content:
        try:
            yield item.replace(needle, str(replace))
        except AttributeError:
            yield list(rreplace(item, needle, replace))


def find_type(_type, content, n=0):
    """Searches content for the nth (zero based) occurrence of a given type
    and returns the corresponding key if successful.

    Args:
        _type (str): the type of element to find (i.e. 'numeric'
            or 'string')

        content (Iter[str]): the content to search

    Returns:
        int: Index of the found element or -1 on failure

    Examples:
        >>> find_type('string', ('one', '2w', '3a'), 2)
        2
        >>> find_type('numeric', ('1', 2, 3), 2)
        -1
        >>> find_type('numeric', ('one', 2, 3), 1)
        2
    """
    switch = {"numeric": "real", "string": "upper"}
    func = lambda x: hasattr(x, switch[_type])

    try:
        found = next(it.islice(filter(func, content), n, None))
    except StopIteration:
        return -1
    else:
        return content.index(found)


def strip(value, thousand_sep=",", decimal_sep="."):
    """Strips a string of all non-numeric characters.

    Args:
        value (str): The string to parse.
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')

    Examples:
        >>> strip('$123.45') == '123.45'
        True
        >>> strip('123€') == '123'
        True

    Returns:
        str: The stripped value
    """
    currencies = zip(CURRENCIES, it.repeat(""))
    separators = [(thousand_sep, ""), (decimal_sep, ".")]

    try:
        stripped = mreplace(value, it.chain(currencies, separators))
    except AttributeError:
        stripped = value  # We don't have a string

    return stripped


def is_numeric(content, thousand_sep=",", decimal_sep=".", **kwargs):
    """Determines whether or not content can be converted into a number

    Args:
        content (scalar): the content to analyze
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')
        kwargs (dict): Keyword arguments passed to the search function

    Kwargs:
        strip_zeros (bool): Remove leading zeros (default: False)

    Examples:
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
        zero_point = s.startswith("0.")
        passed = bool(floated) or zero_point

        if s.startswith("0") and not (kwargs.get("strip_zeros") or zero_point):
            try:
                passed = int(stripped) == 0
            except ValueError:
                passed = False

    return passed


def is_int(content, strip_zeros=False, thousand_sep=",", decimal_sep="."):
    """Determines whether or not content can be converted into an int

    Args:
        content (scalar): the content to analyze
        strip_zeros (bool): Remove leading zeros (default: False)
        thousand_sep (char): thousand's separator (default: ',')
        decimal_sep (char): decimal separator (default: '.')

    Examples:
        >>> is_int('$123.45')
        False
        >>> is_int('1.00')
        True
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
    """Determines whether or not content can be converted into a bool

    Args:
        content (scalar): the content to analyze
        trues (Seq[str]): Values to consider True.
        falses (Seq[str]): Values to consider False.

    Examples:
        >>> is_bool(True)
        True
        >>> is_bool('true')
        True
        >>> is_bool(0)
        True
        >>> is_bool(1)
        True
        >>> is_bool('1')
        False
    """
    trues = set(map(str.lower, trues) if trues else DEF_TRUES)
    falses = set(map(str.lower, falses) if falses else DEF_FALSES)
    bools = trues.union(falses).union([True, False])

    try:
        passed = content.lower() in bools
    except AttributeError:
        passed = content in bools

    return passed


def is_null(content, nulls=None, blanks_as_nulls=False):
    """Determines whether or not content can be converted into a null

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
    def_nulls = ("na", "n/a", "none", "null", ".")
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
    """Filters content

    Args:
        content (dict): The content to filter
        blacklist (Seq[str]): The fields to remove (default: None)
        inverse (bool): Keep fields instead of removing them (default: False)

    See also:
        `meza.process.cut`

    Returns:
        dict: The filtered content

    Examples:
        >>> content = {'keep': 'Hello', 'strip': 'World'}
        >>> dfilter(content) == {'keep': 'Hello', 'strip': 'World'}
        True
        >>> dfilter(content, ['strip']) == {'keep': 'Hello'}
        True
        >>> dfilter(content, ['strip'], True) == {'strip': 'World'}
        True
    """
    blackset = set(blacklist or [])
    func = filterfalse if inverse else filter
    return dict(func(lambda x: x[0] not in blackset, content.items()))


def byte(content):
    """Creates a bytearray from a string or iterable of characters

    Args:
        content (Iter[char]): A string or iterable of characters

    Returns:
        (bytearray): A bytearray of the content

    Examples:
        >>> byte('Hello World!') == bytearray(b'Hello World!')
        True
        >>> byte(iter('Iñtërnâ')) == bytearray('Iñtërnâ'.encode('utf-8'))
        True
    """
    try:
        # it's unicode like 'Hello' or 'Iñtërnâtiônàližætiøn'
        bytes_ = content.encode(ENCODING)
    except AttributeError:
        # it's a unicode or encoded iterable like ['H', 'e', 'l', 'l', 'o'],
        # ['I', 'ñ', 't', 'ë', 'r', 'n', 'â', 't', 'i', 'ô', 'n'],
        # or [b'I', b'\xc3\xb1', b't', b'\xc3\xab', b'r']
        bytes_ = b"".join(map(encode, content))

    return bytearray(bytes_)


def chunk(content, chunksize=None, start=0, stop=None):
    """Groups data into chunks each with (at most) `chunksize` items
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
        >>> next(chunk([1, 2, 3, 4, 5, 6]))
        [1, 2, 3, 4, 5, 6]
        >>> next(chunk([1, 2, 3, 4, 5, 6], 2))
        [1, 2]
    """
    if hasattr(content, "read"):  # it's a file
        content.seek(start) if start else None
        content.truncate(stop) if stop else None

        if chunksize:
            generator = (content.read(chunksize) for _ in it.count())
        else:
            generator = iter([content.read()])
    elif callable(content):  # it's an r.iter_content
        chunksize = chunksize or MAXINT

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


def get_values(narray):
    """Obtains the raw values from a nested list of arrays

    Args:
        narray (Iter[array]): An array or (nested) iterable of arrays

    Yields:
        Iter[scalar]: The flattened array values.

    Examples:
        >>> from array import array
        >>>
        >>> u, i = 'u', 'i'
        >>> narray_0 = array(i, [2, 3])
        >>> narray_1 = [array(u, 'alpha'), array(u, 'beta')]
        >>> narray_2 = [array(u, 'aa'), [array(i, [9])]]
        >>> list(get_values(narray_0)) == [2, 3]
        True
        >>> list(get_values(narray_1)) == ['alpha', 'beta']
        True
        >>> list(get_values(narray_2)) == ['aa', 9]
        True
    """
    try:
        yield narray.tounicode()
    except ValueError:
        for y in narray.tolist():
            yield y
    except AttributeError:
        for n in narray:
            for x in get_values(n):
                yield x


def xmlize(content):
    """Recursively makes elements of an array xml compliant

    Args:
        content (Iter[str]): the content to clean

    Yields:
        (str): the cleaned element

    Examples:
        >>> list(xmlize(['&', '<'])) == ['&amp', '&lt']
        True
    """
    replacements = [
        ("&", "&amp"),
        (">", "&gt"),
        ("<", "&lt"),
        ("\n", " "),
        ("\r\n", " "),
    ]

    for item in content:
        if hasattr(item, "upper"):
            yield mreplace(item, replacements)
        else:
            try:
                yield list(xmlize(item))
            except TypeError:
                yield mreplace(item, replacements) if item else ""


def afterish(content, separator=","):
    """Calculates the number of digits after a given separator.

    Args:
        content (str): The string to parse.
        separator (char): Character to start counting from (default: ',').
        exclude (char): Character to ignore from the calculation (default: '').

    Returns:
        int: the number of digits that appear after the separator

    Examples:
        >>> afterish('123.45', '.')
        2
        >>> afterish('1001.', '.')
        0
        >>> afterish('1,001€')
        3
    """
    numeric = is_numeric(content)

    if numeric and separator in content:
        pos = content.rfind(separator) + 1
        included = it.takewhile(lambda x: x.isdigit(), content[pos:])
        after = len(list(included))
    elif numeric:
        after = -1
    else:
        raise ValueError("Not able to coerce {} to a number".format(content))

    return after


def get_separators(content):
    """Guesses the appropriate thousandths and decimal separators

    Args:
        content (str): The string to parse.

    Examples:
        >>> s = get_separators('$123.45')
        >>> (s['thousand_sep'], s['decimal_sep']) == (',', '.')
        True
        >>> s = get_separators('123,45€')
        >>> (s['thousand_sep'], s['decimal_sep']) == ('.', ',')
        True
        >>> s = get_separators(123.45)
        >>> (s['thousand_sep'], s['decimal_sep']) == (',', '.')
        True
        >>> s = get_separators('1,234.56')
        >>> (s['thousand_sep'], s['decimal_sep']) == (',', '.')
        True
        >>> s = get_separators('1234,56')
        >>> (s['thousand_sep'], s['decimal_sep']) == ('.', ',')
        True
        >>> s = get_separators('1.234,56')
        >>> (s['thousand_sep'], s['decimal_sep']) == ('.', ',')
        True

    Returns:
        dict: thousandths and decimal separators
    """
    try:
        after_comma = afterish(content)
        after_decimal = afterish(content, ".")
    except TypeError:
        # We don't have a string
        after_comma, after_decimal = 0, 0
    except ValueError:
        # We don't have a numeric
        after_comma, after_decimal = None, None

    if after_comma in {-1, 0, 3} and after_decimal in {-1, 0, 1, 2}:
        thousand_sep, decimal_sep = ",", "."
    elif after_comma in {-1, 0, 1, 2} and after_decimal in {-1, 0, 3}:
        thousand_sep, decimal_sep = ".", ","
    else:
        logger.debug("after_comma: %s", after_comma)
        logger.debug("after_decimal: %s", after_decimal)
        raise ValueError("Invalid number format for `{}`.".format(content))

    return {"thousand_sep": thousand_sep, "decimal_sep": decimal_sep}


def add_ordinal(num):
    """Returns a number with ordinal suffix, e.g., 1st, 2nd, 3rd.

    Args:
        num (int): a number

    Returns:
        (str): a number with the ordinal suffix

    Examples:
        >>> add_ordinal(11) == '11th'
        True
        >>> add_ordinal(132) == '132nd'
        True
    """
    switch = {1: "st", 2: "nd", 3: "rd"}
    end = "th" if (num % 100 in {11, 12, 13}) else switch.get(num % 10, "th")
    return "%i%s" % (num, end)


def _fuzzy_match(needle, haystack, **kwargs):
    for n in needle:
        for h in haystack:
            if n in h.lower():
                yield h


def _exact_match(*args, **kwargs):
    sets = (set(i.lower() for i in arg) for arg in args)
    return iter(reduce(lambda x, y: x.intersection(y), sets))


def find(*args, **kwargs):
    """Determines if there is any overlap between lists of words

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
        >>> find(needle, haystack, method='fuzzy') == 'my_value'
        True
        >>> find(needle, haystack) == ''
        True
        >>> find(needle, ['num_days', 'width']) == 'width'
        True
    """
    method = kwargs.pop("method", "exact")
    default = kwargs.pop("default", "")
    funcs = {"exact": _exact_match, "fuzzy": _fuzzy_match}
    func = funcs.get(method, method)

    try:
        return next(func(*args, **kwargs))
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
        `meza.process.fillempty`

    Examples:
        >>> previous = {}
        >>> current = {
        ...     'column_a': '1',
        ...     'column_b': '27',
        ...     'column_c': '',
        ... }
        >>> length = len(current)
        >>> filled = fill(previous, current, value=0)
        >>> dict(it.islice(filled, length)) == {
        ...     'column_a': '1',
        ...     'column_b': '27',
        ...     'column_c': 0,
        ... }
        True
        >>> next(filled) == {'column_a': 0, 'column_b': 0, 'column_c': 1}
        True
    """
    pkwargs = {"blanks_as_nulls": kwargs.get("blanks_as_nulls", True)}
    def_pred = partial(is_null, **pkwargs)
    predicate = kwargs.get("pred", def_pred)
    value = kwargs.get("value")
    limit = kwargs.get("limit")
    fields = kwargs.get("fields")
    count = kwargs.get("count", {})
    fill_key = kwargs.get("fill_key")
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
            `meza.io` read function.

        y (dict): Second record. Row of data whose keys are the field names.
            E.g., result from from calling next() on the output of any
            `meza.io` read function.

        key (str): Current key.
        value (Optional[scalar]): The 2nd record's value of the given `key`.

        pred (func): Value of the `key` to combine. Can optionally
            be a function which receives `key` and should return `True`
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
        `meza.process.merge`

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> x, y = records[0], records[1]
        >>> combine(x, y, 'a', pred='amount', op=sum) == 'item'
        True
        >>> combine(x, y, 'amount', pred='amount', op=sum)
        500
    """
    value = y.get(key, default) if value is None else value
    pred = pred if callable(pred) else partial(operator.eq, pred)

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
        ...     'parent_a_child_1': 1,
        ...     'parent_a_child_2': 2,
        ...     'parent_a_child_3': 3,
        ...     'parent_b_child_1': 1,
        ...     'parent_b_child_2': 2,
        ...     'parent_b_child_3': 3,
        ...     'parent_c': 'no child',
        ... }
        ...
        True
        >>> dict(flatten(record, 'flt')) == {
        ...     'flt_parent_a_child_1': 1,
        ...     'flt_parent_a_child_2': 2,
        ...     'flt_parent_a_child_3': 3,
        ...     'flt_parent_b_child_1': 1,
        ...     'flt_parent_b_child_2': 2,
        ...     'flt_parent_b_child_3': 3,
        ...     'flt_parent_c': 'no child',
        ... }
        True
    """
    try:
        for key, value in record.items():
            newkey = "{}_{}".format(prefix, key) if prefix else key

            for flattened in flatten(value, newkey):
                yield flattened
    except AttributeError:
        yield (prefix, record)


def remove_keys(record, *args, whitelist=False):
    """Remove keys from a dict and return new dict

    Args:
        record (dict): The dict to remove keys from
        args (List[str]): The keys to remove

    Kwargs:
        whitelist (bool): Select keys instead of removing them

    Returns:
        dict: New dict with specified keys removed

    Examples:
        >>> record = {'keep': 1, 'remove': 2}
        >>> remove_keys(record, 'remove') == {'keep': 1}
        True
        >>> remove_keys(Objectify(record), 'remove') == {'keep': 1}
        True
        >>> remove_keys(record, 'keep', whitelist=True) == {'keep': 1}
        True
        >>> remove_keys(Objectify(record), 'keep', whitelist=True) == {'keep': 1}
        True
    """
    args = set(args)

    if whitelist:
        removed = {k: v for k, v in record.items() if k in args}
    else:
        removed = {k: v for k, v in record.items() if k not in args}

    return removed


def listize(item):
    """Create a listlike object from an item

    Args:
        item (dict): The object to convert

    Returns:
        Seq: Item as a listlike object

    Examples:
    >>> listize(x for x in range(3))  # doctest: +ELLIPSIS
    <generator object <genexpr> at 0x...>
    >>> listize([x for x in range(3)])
    [0, 1, 2]
    >>> listize(iter(x for x in range(3)))  # doctest: +ELLIPSIS
    <generator object <genexpr> at 0x...>
    >>> listize(range(3))
    range(0, 3)
    """
    if hasattr(item, "keys"):
        listlike = False
    else:
        attrs = {"append", "next", "__reversed__", "__next__"}
        listlike = attrs.intersection(dir(item))

    return item if listlike else [item]


def def_itemgetter(attr, default=None):
    """like operator.itemgetter but fills in missing keys with a default value

    Args:
        attr (str):
        default (scalar):

    Examples:
        >>> records = [{'key': 1}, {'key': 3}, {'value': 3}]
        >>> sorted(records, key=operator.itemgetter('key'))[0]
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        KeyError:...
        >>> keyfunc = def_itemgetter('key', 0)
        >>> sorted(records, key=keyfunc, reverse=True)[0] == {'key': 3}
        True
    """
    return lambda obj: obj.get(attr, default)


def op_everseen(iterable, key=None, pad=False, op="lt"):
    """List min/max/equal... elements, preserving order. Remember all
    elements ever seen.

    >>> list(op_everseen([4, 6, 3, 8, 2, 1]))
    [4, 3, 2, 1]
    >>> op = operator.itemgetter(1)
    >>> seen = op_everseen([('a', 6), ('b', 4), ('c', 8)], op)
    >>> list(seen) == [('a', 6), ('b', 4)]
    True
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


def fpartial(op):
    """Takes a function that accepts 2 arguments, and returns an equivalent
    function that accepts one iterable argument.

    >>> div = fpartial(operator.truediv)
    >>> div([4, 3, 2])
    0.6666666666666666
    """
    return partial(reduce, op)


def sum_and_count(x, y):
    """A function used for calculating the mean of a list from a reduce.

    >>> from operator import truediv

    >>> l = [15, 18, 2, 36, 12, 78, 5, 6, 9]
    >>> truediv(*reduce(sum_and_count, l)) == 20.11111111111111
    True
    >>> truediv(*fpartial(sum_and_count)(l)) == 20.11111111111111
    True
    """
    try:
        return (x[0] + y, x[1] + 1)
    except TypeError:
        return ((x or 0) + (y or 0), len([i for i in [x, y] if i is not None]))
