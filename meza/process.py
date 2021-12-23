#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.process
~~~~~~~~~~~~

Provides methods for processing `records`, i.e., tabular data.

Examples:
    basic usage::

        >>> from meza.process import type_cast
        >>>
        >>> records = [{'some_value': '1'}, {'some_value': '2'}]
        >>> types = [{'id': 'some_value', 'type': 'int'}]
        >>> next(type_cast(records, types)) == {'some_value': 1}
        True

Attributes:
    CURRENCIES [tuple(unicode)]: Currency symbols to remove from decimal
        strings.
"""
import itertools as it
import hashlib

from functools import partial, reduce
from collections import defaultdict
from operator import itemgetter, iadd
from math import log1p
from json import dumps, loads
from collections import deque

from . import convert as cv, fntools as ft, typetools as tt, ENCODING

sort = lambda records, key: iter(sorted(records, key=itemgetter(key)))


def type_cast(records, types=None, warn=False, **kwargs):
    """Casts record entries based on field types.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        types (Iter[dict]): Field types (`guess_type_by_field` or
            `guess_type_by_value` output).

        warn (bool): Raise error if value can't be cast (default: False).

    Yields:
        dict: Type casted record. A row of data whose keys are the field names.

    See also:
        `meza.process.detect_types`
        `meza.process.json_recode`
        `meza.typetools.guess_type_by_field`
        `meza.typetools.guess_type_by_value`
        `meza.convert.to_int`
        `meza.convert.to_float`
        `meza.convert.to_decimal`
        `meza.convert.to_date`
        `meza.convert.to_time`
        `meza.convert.to_datetime`
        `meza.convert.to_bool`

    Examples:
        >>> import datetime
        >>> record = {
        ...     'null': 'None',
        ...     'bool': 'false',
        ...     'int': '10',
        ...     'float': '1.5',
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': '5/4/82',
        ...     'time': '2:30',
        ...     'datetime': '5/4/82 2pm',
        ... }
        >>> types = list(tt.guess_type_by_value(record))
        >>> next(type_cast([record], types)) == {
        ...     'null': None,
        ...     'bool': False,
        ...     'int': 10,
        ...     'float': 1.5,
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': datetime.date(1982, 5, 4),
        ...     'time': datetime.time(2, 30),
        ...     'datetime': datetime.datetime(1982, 5, 4, 14, 0),
        ... }
        True
        >>> cast = next(type_cast([record], types, dayfirst=True))
        >>> cast['date']
        datetime.date(1982, 4, 5)
        >>> cast['datetime']
        datetime.datetime(1982, 4, 5, 14, 0)
    """
    switch = {
        "int": cv.to_int,
        "float": cv.to_float,
        "decimal": cv.to_decimal,
        "date": cv.to_date,
        "time": cv.to_time,
        "datetime": cv.to_datetime,
        "text": lambda v, **kw: str(v) if v and v.strip() else "",
        "null": lambda x, **kw: None,
        "bool": cv.to_bool,
        "iden": lambda x, **kw: x,
    }

    types = types or []
    field_types = {t["id"]: t["type"] for t in types}

    for row in records:
        tups = ((k, field_types.get(k, "iden"), v) for k, v in row.items())
        yield {k: switch.get(t)(v, warn=warn, **kwargs) for k, t, v in tups}


def json_recode(records):
    """JSON dump and then load record entries using custom JSONEncoder.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

    Yields:
        dict: JSON dumped and loaded record. A row of data whose keys are the
            field names.

    See also:
        `meza.process.type_cast`

    Examples:
        >>> import datetime
        >>> record = {
        ...     'null': None,
        ...     'bool': False,
        ...     'int': 10,
        ...     'float': 1.5,
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': datetime.date(1982, 5, 4),
        ...     'time': datetime.time(2, 30),
        ...     'datetime': datetime.datetime(1982, 5, 4, 14, 0),
        ... }
        >>> next(json_recode([record])) == {
        ...     'null': None,
        ...     'bool': False,
        ...     'int': 10,
        ...     'float': 1.5,
        ...     'text': 'Iñtërnâtiônàližætiøn',
        ...     'date': '1982-05-04',
        ...     'time': '02:30:00',
        ...     'datetime': '1982-05-04 14:00:00'}
        True
    """
    encoder = partial(dumps, cls=ft.CustomEncoder, ensure_ascii=False)

    for record in records:
        yield {k: loads(encoder(v)) for k, v in record.items()}


def gen_confidences(tally, types, a=1):
    """Calculates confidence using a logarithmic function which asymptotically
    approaches 1.

    Args:
        tally (dict): Rows of data whose keys are the field names and whose
            values is a dict of types and counts.

        types (Iter[dicts]): Field types (`guess_type_by_field` or
            `guess_type_by_value` output).

        a (int): logarithmic weighting, a higher value will converge faster
            (default: 1)

    Returns:
        Iter(decimal): Generator of confidences

    See also:
        `meza.typetools.guess_type_by_field`
        `meza.typetools.guess_type_by_value`
        `meza.process.detect_types`

    Examples:
        >>> from decimal import Decimal

        >>> record = {'field_1': 'None', 'field_2': 'false'}
        >>> types = [
        ...     {'id': 'field_1', 'type': 'null'},
        ...     {'id': 'field_2', 'type': 'bool'}]
        >>> tally = {'field_1': {'null': 3}, 'field_2': {'bool': 2}}
        >>> set(gen_confidences(tally, types)) == {
        ...     Decimal('0.52'), Decimal('0.58')}
        True
        >>> set(gen_confidences(tally, types, 5)) == {
        ...     Decimal('0.85'), Decimal('0.87')}
        True
    """
    # http://math.stackexchange.com/a/354879
    calc = lambda x: cv.to_decimal(a * x / (1 + a * x))
    return (calc(log1p(tally[t["id"]][t["type"]])) for t in types)


def gen_types(tally):
    """Selects the field type with the highest count. Also intelligently
    merges compatible types, e.g., 4 ints and 1 floats --> float.

    Args:
        tally (dict): Rows of data whose keys are the field names and whose
            data is a dict of types and counts.

    Yields:
        dict: Field type. The parsed field and its type.

    See also:
        `meza.process.detect_types`

    Examples:
        >>> tally = {
        ...     'field_1': {'null': 3, 'bool': 1},
        ...     'field_2': {'bool': 2, 'int': 4},
        ...     'field_3': {'float': 1, 'int': 5},
        ...     'field_4': {'float': 1, 'time': 2},
        ...     'field_5': {'date': 1, 'time': 2}}
        >>> types = sorted(gen_types(tally), key=itemgetter('id'))
        >>> types[0] == {'id': 'field_1', 'type': 'bool'}
        True
        >>> types[1] == {'id': 'field_2', 'type': 'int'}
        True
        >>> types[2] == {'id': 'field_3', 'type': 'float'}
        True
        >>> types[3] == {'id': 'field_4', 'type': 'text'}
        True
        >>> types[4] == {'id': 'field_5', 'type': 'datetime'}
        True
    """

    comp_types = [
        ({"float", "int"}, "float"),
        ({"date", "time", "datetime"}, "datetime"),
        ({"bool", "int"}, "int"),
    ]

    def gct(types):
        non_null = [t for t in types if t != "null"]

        if len(non_null) == 1:
            _type = non_null[0]
        else:
            for k, v in comp_types:
                if k.issuperset(non_null):
                    _type = v
                    break
            else:
                _type = "text"

        return _type

    for field, tcount in tally.items():
        _type = gct(tcount) if len(tcount) > 1 else next(iter(tcount))
        yield {"id": field, "type": _type}


def detect_types(records, min_conf=0.95, hweight=6, max_iter=100):
    """Detects record types by selecting the first type which reaches the
    minimum confidence level (based on number of hits).

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        min_conf (float): minimum confidence level, a lower value will
            converge faster (default: 0.95)

        hweight (int): weight to give header row, a higher value will
            converge faster (default: 6).

            detect_types(records, 0.9, 3)['count'] == 23
            detect_types(records, 0.9, 4)['count'] == 10
            detect_types(records, 0.9, 5)['count'] == 6
            detect_types(records, 0.95, 5)['count'] == 31
            detect_types(records, 0.95, 6)['count'] == 17
            detect_types(records, 0.95, 7)['count'] == 11

        max_iter (int): maximum number of iterations to perform (default: 100)

    Returns:
        tuple(Iter[dict], dict): Tuple of records and the result

    See also:
        `meza.process.type_cast`
        `meza.process.gen_types`
        `meza.process.gen_confidences`
        `meza.typetools.guess_type_by_field`
        `meza.typetools.guess_type_by_value`

    Examples:
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
        >>> records = it.repeat(record)
        >>> types = detect_types(records)[1]['types']
        >>> set(t['id'] for t in types) == {
        ...     'int', 'text', 'float', 'datetime', 'bool', 'time',
        ...     'date', 'null'}
        True
        >>> all(t['id'] == t['type'] for t in types)
        True
    """
    records = iter(records)
    tally = {}
    consumed = []

    if hweight < 1:
        raise ValueError("`hweight` must be greater than or equal to 1!")

    if min_conf >= 1:
        raise ValueError("`min_conf must` be less than 1!")

    for record in records:
        if not tally:
            # take a first guess using the header
            ftypes = tt.guess_type_by_field(record.keys())
            tally = {t["id"]: defaultdict(int) for t in ftypes}
            [iadd(tally[t["id"]][t["type"]], hweight) for t in ftypes]

        # now guess using the values
        for t in tt.guess_type_by_value(record):
            try:
                tally[t["id"]][t["type"]] += 1
            except KeyError:
                tally[t["id"]] = defaultdict(int)
                tally[t["id"]][t["type"]] = 1

        types = list(gen_types(tally))
        confidence = min(gen_confidences(tally, types, hweight))
        consumed.append(record)
        count = len(consumed)

        if (confidence >= min_conf) or count >= max_iter:
            break

    records = it.chain(consumed, records)

    result = {
        "confidence": confidence,
        "types": types,
        "count": count,
        "accurate": confidence >= min_conf,
    }

    return records, result


def fillempty(records, value=None, method=None, limit=None, fields=None):
    """Replaces missing data with either a single value or by front/back/side
    filling.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

    Kwargs:
        value (str): Value to use to fill holes (default: None).
        method (str): Fill method, one of either {'front', 'back'} or a column
            name (default: None). `front` propagates the last valid
            value forward. `back` propagates the next valid value
            backwards. If given a column name, that column's current value
            will be used.

            *************************************************
            * Note: if `back` is selected, all records will *
            * be read into memory. Use with caution.        *
            *************************************************

        limit (int): Max number of consecutive rows to fill (default: None).
        fields (Seq[str]): Names of the columns to fill (default: None, i.e.,
            all).

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `meza.fntools.fill`

    Examples:
        >>> record = {'a': '1', 'b': '27', 'c': ''}
        >>> next(fillempty([record], 0)) == {'a': '1', 'b': '27', 'c': 0}
        True
    """
    if method and value is not None:
        raise Exception("You can not specify both a `value` and `method`.")
    elif not method and value is None:
        raise Exception("You must specify either a `value` or `method`.")
    elif method == "back":
        content = reversed(records)
    else:
        content = records

    kwargs = {
        "value": value,
        "limit": limit,
        "fields": fields,
        "fill_key": method if method not in {"front", "back"} else None,
    }

    prev_row = {}
    count = {}
    length = 0
    result = []

    for row in content:
        length = length or len(row)
        filled = ft.fill(prev_row, row, count=count, **kwargs)
        prev_row = dict(it.islice(filled, length))
        count = next(filled)

        if method == "back":
            result.append(prev_row)
        else:
            yield prev_row

    if method == "back":
        for row in reversed(result):
            yield row


def merge(records, **kwargs):
    """Merges `records` and optionally combines specified keys using a
    specified binary operator.

    http://codereview.stackexchange.com/a/85822/71049
    http://stackoverflow.com/a/31812635/408556
    http://stackoverflow.com/a/3936548/408556

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        kwargs (dict): keyword arguments

    Kwargs:
        pred (func): Predicate. Value of the `key` to combine. Can optionally
            be a function which receives the current key and should return
            `True` if overlapping values should be combined. Can optionally be
            a keyfunc which receives a record. In this case, the entries will
            be combined if the value obtained after applying keyfunc to the
            record equals the current value. Requires that `op` is set.

            If not set, no records will be combined.

            If a key occurs in multiple records and isn't combined, it will be
            overwritten by the last record.

        op (func): Receives a list of 2 values from overlapping keys and should
            return the combined value. Common operators are `sum`, `min`,
            `max`, etc. Requires that `pred` is set. If a key is not
            present in a record, the value from `default` will be used. Note,
            since `op` applied inside of `reduce`, it may not perform as
            expected for all functions for more than 2 records. E.g. a mean
            function will be applied as follows:

                mean([1, 2, 3]) --> mean([mean([1, 2]), 3])

            You would expect to get 2, but will instead get 2.25.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        dict: merged record

    See also:
        `meza.process.aggregate`
        `meza.process.join`
        `meza.fntools.combine`

    Examples:
        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> merge(records, pred='amount', op=sum)['amount'] == 900
        True
        >>> merge(records)['amount'] == 400
        True
        >>> records = [{'a': 1, 'b': 2}, {'b': 7, 'c': 9}, {'c': 3, 'd': 4}]
        >>> merge(records) == {'a': 1, 'b': 7, 'c': 3, 'd': 4}
        True
    """

    def reducer(x, y):
        _merge = partial(ft.combine, x, y, **kwargs)
        new_y = ((k, _merge(k, v)) for k, v in y.items())
        return dict(it.chain(x.items(), new_y))

    if kwargs.get("pred") and kwargs.get("op"):
        record = reduce(reducer, records)
    else:
        items = (r.items() for r in records)
        record = dict(it.chain.from_iterable(items))

    return record


def aggregate(records, key, op, default=0):
    """Aggregates `records` on a specified key.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        key (str): The field to aggregate

        op (func): Aggregation function. Receives a list of all non-null values
            and should return the combined value. Common operators are `sum`,
            `min`, `max`, etc.

        default (int or str): default value to use in `op` for missing keys
            (default: 0).

    Returns:
        dict: The first record with an aggregated value for `key`

    See also:
        `meza.process.merge`

    Examples:
        >>> from . import stats

        >>> records = [
        ...     {'a': 'item', 'amount': 200},
        ...     {'a': 'item', 'amount': 300},
        ...     {'a': 'item', 'amount': 400}]
        ...
        >>> aggregate(records, 'amount', sum)['amount'] == 900
        True
        >>> agg = aggregate(records, 'amount', stats.mean)
        >>> agg['amount'] == 300.0
        True
    """
    records = iter(records)
    first = next(records)
    values = (r.get(key, default) for r in it.chain([first], records))
    value = op([x for x in values if x is not None])
    return dict(it.chain(first.items(), [(key, value)]))


def group(records, keyfunc, tupled=True, aggregator=list, **kwargs):
    """Groups records by keyfunc

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        keyfunc (func): Either a fieldname or function which receives a record
            and selects which value to sort/group by.

        aggregator (func): A post processing function to call on each resulting
            group (default: list).

        tupled (bool): Return the key, group tuples (default: True)

        kwargs (dict): Keyword args passed to the aggregator.

    Returns:
        Iter(tuple[key, group]): Generator of tuples

    Examples:
        >>> records = [
        ...     {'item': 'a', 'amount': 200},
        ...     {'item': 'b', 'amount': 200},
        ...     {'item': 'c', 'amount': 400}]
        ...
        >>> key, grp = next(group(records, 'amount'))
        >>> key
        200
        >>> len(grp)
        2
        >>> next(group(records, 'amount', False))[0] == {
        ...     'item': 'a', 'amount': 200}
        True
    """
    keyfunc = keyfunc if callable(keyfunc) else itemgetter(keyfunc)
    sorted_records = sorted(records, key=keyfunc)
    grouped = it.groupby(sorted_records, keyfunc)

    if tupled:
        result = ((key, aggregator(group, **kwargs)) for key, group in grouped)
    else:
        result = (aggregator(group, **kwargs) for key, group in grouped)

    return result


def prepend(records, row):
    """Adds a row to the beginning of a records iterator.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        row (dict): A row of data.

    Returns:
        Iterator of rows.

    See also:
        `meza.process.peek`

    Examples:
        >>> records = [
        ...     {'length': 5, 'species': 'setosa', 'color': 'red'},
        ...     {'length': 5, 'species': 'setosa', 'color': 'blue'},
        ...     {'length': 6, 'species': 'versi', 'color': 'red'},
        ...     {'length': 6, 'species': 'versi', 'color': 'blue'}]
        ...
        >>> row = records[0]
        >>> row == {'length': 5, 'species': 'setosa', 'color': 'red'}
        True
        >>> next(prepend(records, row)) == row
        True
    """
    return it.chain([row], iter(records))


def peek(records, n=5):
    """Provides a list of the first n rows of a records generator.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        n (int): The number of rows to preview

    Returns:
        tuple: The reconstituted records iterator and a list of its first n
            rows.

    See also:
        `meza.process.prepend`

    Examples:
        >>> records = [
        ...     {'length': 5, 'species': 'setosa', 'color': 'red'},
        ...     {'length': 5, 'species': 'setosa', 'color': 'blue'},
        ...     {'length': 6, 'species': 'versi', 'color': 'red'},
        ...     {'length': 6, 'species': 'versi', 'color': 'blue'}]
        ...
        >>> records, preview = peek(iter(records), 2)
        >>> len(preview)
        2
        >>> preview[0] == {'length': 5, 'species': 'setosa', 'color': 'red'}
        True
        >>> records  # doctest: +ELLIPSIS
        <itertools.chain object at 0x...>
    """
    records = iter(records)
    preview = list(it.islice(records, n))
    return (it.chain(preview, records), preview)


def pivot(records, data, column, op=sum, **kwargs):
    """Create a spreadsheet-style pivot table.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        data (str): Field to aggregate
        column (str): Field to group by and create columns for in the resulting
            pivot table.

        op (func): Aggregation function (default: sum)
        kwargs (dict): keyword arguments

    Kwargs:
        rows (Seq[str]): Fields to include as rows in the resulting pivot table
            (default: All fields not in `data` or `column`).

        fill_value (scalar): Value to replace missing values with
            (default: None)

        dropna (bool): Do not include columns with missing values
            (default: True)

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `meza.process.aggregate`
        `meza.process.normalize`

    Examples:
        >>> records = [
        ...     {'length': 5, 'width': 2, 'species': 'setosa', 'color': 'red'},
        ...     {'length': 5, 'width': 2, 'species': 'setosa', 'color': 'blue'},
        ...     {'length': 6, 'width': 2, 'species': 'versi', 'color': 'red'},
        ...     {'length': 6, 'width': 2, 'species': 'versi', 'color': 'blue'}]
        ...
        >>> next(pivot(records, 'length', 'species', rows=['width'])) == {
        ...     'width': 2, 'setosa': 10, 'versi': 12}
        True
        >>> next(pivot(records, 'length', 'species')) == {
        ...     'width': 2, 'color': 'blue', 'setosa': 5, 'versi': 6}
        True
    """
    records = iter(records)
    first = next(records)
    chained = it.chain([first], records)

    keys = set(first.keys())
    rows = kwargs.get("rows", keys.difference([data, column]))
    fill_value = kwargs.get("fill_value")
    dropna = kwargs.get("dropna", True)
    filterer = lambda x: x[0] in rows
    keyfunc = lambda r: tuple(map(r.get, it.chain(rows, [column])))
    grouped = group(chained, keyfunc)

    def gen_raw(grouped):
        for key, _group in grouped:
            r = aggregate(_group, data, op)
            filtered = filter(filterer, r.items())
            yield dict(it.chain([(r[column], r.get(data))], filtered))

    if dropna:
        raw = gen_raw(grouped)
    else:
        raw = list(gen_raw(grouped))
        differences = (set(r).difference(rows) for r in raw)
        columns = set(it.chain.from_iterable(differences))

    for key, _group in group(raw, lambda r: tuple(map(r.get, rows))):
        if not dropna:
            empty = [dict(zip(columns, it.repeat(fill_value)))]
            _group = it.chain(empty, _group)

        yield merge(_group)


def normalize(records, data="", column="", rows=None, invert=False):
    """Yields normalized records from a spreadsheet-style pivot table.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        data (str): Field name to create for values of the normalized fields.
        column (str): Field name to create for keys of the normalized fields.
        rows (Seq[str]): Fields to normalized.
        invert (bool): Treat `rows` as fields that shouldn't be normalized.

    Yields:
        dict: Record. A row of data whose keys are the field names.

    See also:
        `meza.process.pivot`

    Examples:
        >>> records = [
        ...     {'width': 2, 'color': 'blue', 'setosa': 5, 'versi': 6},
        ...     {'width': 2, 'color': 'red', 'setosa': 5, 'versi': 6}]
        ...
        >>> rows = ['setosa', 'versi']
        >>> next(normalize(records, 'length', 'species', rows)) == {
        ...     'color': 'blue', 'width': 2, 'length': 5,
        ...     'species': 'setosa'}
        True
    """
    for r in records:
        nrows = set(r.keys()).difference(rows) if invert else rows
        filtered = [x for x in r.items() if x[0] not in nrows]

        for row in nrows:
            yield dict(it.chain([(column, row), (data, r.get(row))], filtered))


def join(left, right):
    """Performs a SQL like merge.

    Args:
        left (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        right (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

    Returns:
        Iterator of records.

    See also:
        `meza.process.merge`

    Examples:
        >>> left = [
        ...     {'length': 5, 'species': 'setosa'},
        ...     {'length': 6, 'species': 'versi'}]
        >>> right = [{'color': 'red'}]
        >>> next(join(left, right)) == {
        ...     'length': 5, 'species': 'setosa', 'color': u'red'}
        True
    """
    return map(merge, it.product(left, right))


def tfilter(records, field, pred=None):
    """ Yields records for which the predicate is True for a given field.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        field (str): The column to to apply the predicate to.

    Kwargs:
        pred (func): Predicate. Receives the value of `field` and should return
            `True`  if the record should be included (default: None, i.e.,
            return the record if value is True).

    See also:
        `meza.process.grep`

    Returns:
        Iter[dict]: The filtered records.

    Examples:
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'bob'},
        ...     {'day': 1, 'name': 'tom'},
        ...     {'day': 2, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'rob'},
        ... ]
        >>> next(tfilter(records, 'day', lambda x: x == 2))['name'] == \
'Iñtërnâtiônàližætiøn'
        True
        >>> next(tfilter(records, 'day', lambda x: x == 3))['name'] == 'rob'
        True
    """
    predicate = lambda x: pred(x.get(field)) if pred else None
    return filter(predicate, records)


def unique(records, fields=None, pred=None, bufsize=4096):
    """ Yields unique records

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        fields (Seq[str]): The columns to use for testing uniqueness
            (default: None, i.e., all columns). Overridden by `pred`.

        pred (func): Predicate. Receives a record and should return a value for
            testing uniqueness. Overrides `fields`.

        bufsize (Int): Max size in bytes of the lookup table.

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'bob'},
        ...     {'day': 1, 'name': 'tom'},
        ...     {'day': 2, 'name': 'bill'},
        ...     {'day': 2, 'name': 'bob'},
        ...     {'day': 2, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'Iñtërnâtiônàližætiøn'},
        ...     {'day': 3, 'name': 'bob'},
        ...     {'day': 3, 'name': 'rob'},
        ... ]
        >>> next(it.islice(unique(records), 3, 4))['name'] == 'bill'
        True
        >>> next(it.islice(unique(records, ['name']), 3, 4))['name'] == \
'Iñtërnâtiônàližætiøn'
        True
    """
    seen = deque([], bufsize)

    for r in records:
        if not pred:
            unique = set(fields or r.keys())
            items = sorted((k, v) for k, v in r.items() if k in unique)

        entry = pred(r) if pred else tuple(items)

        if entry not in set(seen):
            seen.append(entry)
            yield r


def cut(records, fields=None, exclude=False, prune=False):
    """Edit records to only return specified columns. Like unix `cut`, but for
    tabular data.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        fields (Iter[str]): Column names to include. (default: None, i.e.,
            all columns.').

        exclude (bool): Exclude column names instead of including them
            (default: False).

        prune (bool): Remove empty rows from result (default: False).


    See also:
        `meza.fntools.dfilter`

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [
        ...     {'field_1': 1, 'field_2': 'bill', 'field_3': 'male'},
        ...     {'field_1': 2, 'field_2': 'bob', 'field_3': 'male'},
        ...     {'field_1': 3, 'field_2': 'jane', 'field_3': 'female'},
        ... ]
        >>> next(cut(records, ['field_2'])) == {'field_2': 'bill'}
        True
    """
    filtered = (ft.dfilter(r, fields, not exclude) for r in records)
    return filter(None, filtered) if prune else filtered


def get_suffix(cpos, pos, k=None, count=None, chunksize=None):
    """Determines the suffix based on a subchunk's position"""
    subchunks = count and count < (chunksize or float("inf"))

    if subchunks and k is None:
        args = (cpos + 1, pos + 1)
        suffix = "{0:02d}_{1:03d}".format(*args)
    elif subchunks:
        args = (k, cpos + 1, pos + 1)
        suffix = "{0}_{1:02d}_{2:03d}".format(*args)
    elif chunksize and k is None:
        suffix = "{0:03d}".format(cpos + 1)
    elif chunksize:
        suffix = "{0}_{1:03d}".format(k, cpos + 1)
    else:
        suffix = "" if k is None else k

    return suffix


def split(records, key=None, count=None, chunksize=None):
    """Split records into bite sized pieces. Like unix `split`, but for
    tabular data.
    """
    chunksize = chunksize or count

    for cpos, records_chunk in enumerate(ft.chunk(records, chunksize)):
        if key:
            groups = group(records_chunk, itemgetter(key))
        else:
            groups = [(None, records_chunk)]

        for k, g in groups:
            for pos, sub_records in enumerate(ft.chunk(g, count)):
                yield sub_records, get_suffix(cpos, pos, k, count, chunksize)


def grep(records, rules, fields=None, any_match=False, inverse=False):
    """Yields rows which match all the given rules.

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        rules (Iter[dict]): Each rule dict must contain a `pattern`
            key whose value can be either a string, function, or regular
            expression. A `fields` key is optional and corresponds to the
            columns you wish to pattern match. Default is to search all columns.

        fields (Iter[str]): Default fields if one isn't found in a rule.

        any_match (bool): Return records which match any of the rules
            (default: False)

        inverse (bool): Only return records which don't match the rules
            (default: False)

    See also:
        `meza.process.tfilter`

    Returns:
        Iter[dict]: The filtered records.

    Examples:
        >>> import re
        >>> records = [
        ...     {'day': 1, 'name': 'bill'},
        ...     {'day': 1, 'name': 'rob'},
        ...     {'day': 1, 'name': 'jane'},
        ...     {'day': 2, 'name': 'rob'},
        ...     {'day': 3, 'name': 'jane'},
        ... ]
        >>> rules = [{'fields': ['name'], 'pattern': 'o'}]
        >>> next(grep(records, rules))['name'] == 'rob'
        True
        >>> rules = [{'pattern': re.compile(r'j.*e$')}]
        >>> next(grep(records, rules, ['name']))['name'] == 'jane'
        True
    """

    def predicate(record):
        def_fields = fields or record.keys()

        for rule in rules:
            for field in rule.get("fields", def_fields):
                value = record[field]
                p = rule["pattern"]

                try:
                    passed = p.match(value)
                except AttributeError:
                    passed = p(value) if callable(p) else p in value

                if (any_match and passed) or not (any_match or passed):
                    break

        return not passed if inverse else passed

    return filter(predicate, records)


def hash(records, fields=None, algo="md5"):
    """Yields rows whose value of the given field(s) are hashed

    Args:
        records (Iter[dict]): Rows of data whose keys are the field names.
            E.g., output from any `meza.io` read function.

        fields (Seq[str]): The columns to use for testing uniqueness
            (default: None, i.e., all columns). Overridden by `pred`.

        algo (str): The hashlib hashing algorithm to use (default: sha1).
            supported algorithms: md5, ripemd128, ripemd160, ripemd256,
                ripemd320, sha1, sha256, sha512, sha384, whirlpool

    See also:
        `meza.io.hash_file`

    Yields:
        dict: Record. A row of data whose keys are the field names.

    Examples:
        >>> records = [{'a': 'item', 'amount': 200}]
        >>> next(hash(records, ['a'])) == {
        ...     'a': '447b7147e84be512208dcc0995d67ebc', 'amount': 200}
        True
    """
    hasher = getattr(hashlib, algo)
    hash_func = lambda x: hasher(str(x).encode(ENCODING)).hexdigest()
    to_hash = set(fields or [])

    for row in records:
        items = row.items()
        yield {k: hash_func(v) if k in to_hash else v for k, v in items}
