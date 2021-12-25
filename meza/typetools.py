#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.typetools
~~~~~~~~~~~~~~

Provides methods for type guessing

Examples:
    basic usage::

        >>> from meza.typetools import is_date
        >>>
        >>> is_date('5/4/82 2pm')
        True
"""
from functools import partial

from . import fntools as ft, convert as cv, NULL_TIME, NULL_YEAR


def type_test(test, _type, key, value):
    try:
        passed = test(value)
    except AttributeError:
        replacements = [("type", ""), ("datetime.", "")]
        real_type = ft.mreplace(str(type(value)), replacements).strip(" '<>")
        result = {"id": key, "type": real_type}
    else:
        result = {"id": key, "type": _type} if passed else None

    return result


def guess_type_by_field(content):
    """Tries to determine field types based on field names.

    Args:
        content (Iter[str]): Field names.

    Yields:
        dict: Field type. The parsed field and its type.

    See also:
        `meza.typetools.guess_type_by_value`
        `meza.process.type_cast`
        `meza.process.gen_confidences`
        `meza.process.detect_types`

    Examples:
        >>> fields = ['date', 'raw_value', 'date_and_time', 'length', 'field']
        >>> {r['id']: r['type'] for r in guess_type_by_field(fields)} == {
        ...     'date': 'date',
        ...     'raw_value': 'float',
        ...     'date_and_time': 'datetime',
        ...     'length': 'float',
        ...     'field': 'text',
        ... }
        ...
        True
    """
    floats = ("value", "length", "width", "days")
    float_func = lambda x: ft.find(floats, [x], method="fuzzy")
    datetime_func = lambda x: ("date" in x) and ("time" in x)

    guess_funcs = [
        {"type": "int", "func": lambda x: "count" in x},
        {"type": "float", "func": float_func},
        {"type": "datetime", "func": datetime_func},
        {"type": "time", "func": lambda x: "time" in x},
        {"type": "date", "func": lambda x: "date" in x},
        {"type": "text", "func": lambda x: True},
    ]

    for item in content:
        for g in guess_funcs:
            result = type_test(g["func"], g["type"], item, item)

            if result:
                yield result
                break


def guess_type_by_value(record, blanks_as_nulls=True, strip_zeros=False):
    """Tries to determine field types based on values.

    Args:
        record (dict): The row to guess.
        blanks_as_nulls (bool): Treat empty strings as null (default: True).
        strip_zeros (bool): Remove leading zeros (default: False)

    Yields:
        dict: Field type. The parsed field and its type.

    See also:
        `meza.typetools.guess_type_by_field`
        `meza.process.type_cast`
        `meza.process.gen_confidences`
        `meza.process.detect_types`

    Examples:
        >>> from datetime import datetime as dt, date, time
        >>>
        >>> record = {
        ...     'null': 'None',
        ...     'bool': 'false',
        ...     'int': '1',
        ...     'float': '1.5',
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': '5/4/82',
        ...     'time': '2:30',
        ...     'datetime': '5/4/82 2pm',
        ... }
        >>> {r['id']: r['type'] for r in guess_type_by_value(record)} == {
        ...     'null': 'null',
        ...     'bool': 'bool',
        ...     'int': 'int',
        ...     'float': 'float',
        ...     'text': 'text',
        ...     'date': 'date',
        ...     'time': 'time',
        ...     'datetime': 'datetime'}
        ...
        True
        >>> record = {
        ...     'null': None,
        ...     'bool': False,
        ...     'int': 10,
        ...     'float': 1.5,
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': date(1982, 5, 4),
        ...     'time': time(2, 30),
        ...     'datetime': dt(1982, 5, 4, 2),
        ... }
        >>> {r['id']: r['type'] for r in guess_type_by_value(record)} == {
        ...     'null': 'null',
        ...     'bool': 'bool',
        ...     'int': 'int',
        ...     'float': 'float',
        ...     'text': 'text',
        ...     'date': 'date',
        ...     'time': 'time',
        ...     'datetime': 'datetime'}
        ...
        True
    """
    null_func = partial(ft.is_null, blanks_as_nulls=blanks_as_nulls)
    int_func = partial(ft.is_int, strip_zeros=strip_zeros)
    float_func = partial(ft.is_numeric, strip_zeros=strip_zeros)

    guess_funcs = [
        {"type": "null", "func": null_func},
        {"type": "bool", "func": ft.is_bool},
        {"type": "int", "func": int_func},
        {"type": "float", "func": float_func},
        {"type": "datetime", "func": is_datetime},
        {"type": "time", "func": is_time},
        {"type": "date", "func": is_date},
        {"type": "text", "func": lambda x: hasattr(x, "lower")},
    ]

    for key, value in record.items():
        for g in guess_funcs:
            result = type_test(g["func"], g["type"], key, value)

            if result:
                yield result
                break
        else:
            raise TypeError("Couldn't guess type of '{}'".format(value))


def is_date(content):
    """Determines whether or not content can be converted into a date

    Args:
        content (scalar): the content to analyze

    Examples:
        >>> from datetime import datetime as dt, date, time
        >>>
        >>> is_date('5/4/82 2pm')
        True
        >>> is_date('5/4/82')
        True
        >>> is_date('2pm')
        False
        >>> is_date(dt(1982, 5, 4, 2))
        True
        >>> is_date(date(1982, 5, 4))
        True
        >>> is_date(time(2, 30))
        False
    """
    try:
        converted = cv.to_datetime(content)
    except TypeError:
        converted = content

    try:
        the_year = converted.date().year
    except AttributeError:
        try:
            # see if it's a date (it could have a NULL_YEAR)
            the_year = converted.year
        except AttributeError:
            the_year = NULL_YEAR  # it's a time

    return converted and the_year != NULL_YEAR


def is_time(content):
    """Determines whether or not content can be converted into a time

    Args:
        content (scalar): the content to analyze

    Examples:
        >>> from datetime import datetime as dt, date, time
        >>>
        >>> is_time('5/4/82 2pm')
        True
        >>> is_time('2000-01-01 14:00:00')
        True
        >>> is_time('2000-01-01 00:00:00')
        True
        >>> is_time('5/4/82')
        False
        >>> is_time('2pm')
        True
        >>> is_time('14:00:00')
        True
        >>> is_time('00:00:00')
        True
        >>> is_time(dt(1982, 5, 4, 2))
        True
        >>> is_time(date(1982, 5, 4))
        False
        >>> is_time(time(2, 30))
        True
    """
    try:
        has_time = any(x in content for x in [":", "T", "+", "am", "pm"])
    except TypeError:
        try:
            has_time = content.time
        except AttributeError:
            try:
                has_time = content.minute
            except AttributeError:
                has_time = False

    return bool(has_time)


def is_datetime(content):
    """Determines whether or not content can be converted into a datetime

    Args:
        content (scalar): the content to analyze

    Examples:
        >>> from datetime import datetime as dt, date, time
        >>>
        >>> is_datetime('5/4/82 2pm')
        True
        >>> is_datetime('5/4/82')
        False
        >>> is_datetime('2pm')
        False
        >>> is_datetime(dt(1982, 5, 4, 2))
        True
        >>> is_datetime(date(1982, 5, 4))
        False
        >>> is_datetime(time(2, 30))
        False
    """
    return is_date(content) and is_time(content)
