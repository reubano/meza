# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab
"""
tests.test_process
~~~~~~~~~~~~~~~~~~

Provides main unit tests.
"""
import itertools as it

from nose.tools import assert_true, assert_false, assert_raises, assert_equal

from decimal import Decimal
from functools import partial
from operator import itemgetter, truediv, eq, is_not, contains
from collections import defaultdict

from meza import process as pr, stats, fntools as ft


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print("Site Module Setup\n")


class Test:
    """docstring"""

    def test_typecast(self):
        records = [{"float": "1.5"}]
        types = [{"id": "float", "type": "bool"}]
        assert {"float": False} == next(pr.type_cast(records, types))

        with assert_raises(ValueError):
            next(pr.type_cast(records, types, warn=True))

    def test_detect_types(self):
        record = {
            "null": "None",
            "bool": "false",
            "int": "1",
            "float": "1.5",
            "text": "Iñtërnâtiônàližætiøn",
            "date": "5/4/82",
            "time": "2:30",
            "datetime": "5/4/82 2pm",
        }

        records = it.repeat(record)
        records, result = pr.detect_types(records)
        assert 17 == result["count"]
        assert Decimal("0.95") == result["confidence"]
        assert result["accurate"]

        expected = {
            "null": "null",
            "bool": "bool",
            "int": "int",
            "float": "float",
            "text": "text",
            "date": "date",
            "time": "time",
            "datetime": "datetime",
        }

        assert expected == {r["id"]: r["type"] for r in result["types"]}
        assert record == next(records)

        result = pr.detect_types(records, 0.99)[1]
        assert 100 == result["count"]
        assert Decimal("0.97") == result["confidence"]
        assert not result["accurate"]

        result = pr.detect_types([record, record])[1]
        assert 2 == result["count"]
        assert Decimal("0.87") == result["confidence"]
        assert not result["accurate"]

    def test_detect_types_datetimes_midnight(self):
        records = it.repeat({"foo": "2000-01-01 00:00:00"})
        records, result = pr.detect_types(records)
        assert result["types"] == [{"id": "foo", "type": "datetime"}]

    def test_fillempty(self):
        records = [
            {"a": "1", "b": "27", "c": ""},
            {"a": "", "b": "too short!", "c": None},
            {"a": "0", "b": "mixed", "c": "17"},
        ]

        values = [
            {"a": "1", "b": "27", "c": ""},
            {"a": 0, "b": "too short!", "c": None},
            {"a": "0", "b": "mixed", "c": "17"},
        ]

        new_value_1 = {"a": "1", "b": "too short!", "c": ""}
        more_values_1 = [values[0], new_value_1, values[2]]

        fields = ["a"]
        assert values == list(pr.fillempty(records, 0, fields=fields))

        filled = pr.fillempty(records, method="front")
        assert more_values_1 == list(filled)

        new_value_2 = {"a": "1", "b": "27", "c": "17"}
        new_value_3 = {"a": "0", "b": "too short!", "c": "17"}
        more_values_2 = [new_value_2, new_value_3, values[2]]
        filled = pr.fillempty(records, method="back")
        assert more_values_2 == list(filled)

        more_values_3 = [values[0], new_value_3, values[2]]
        filled = pr.fillempty(records, method="back", limit=1)
        assert more_values_3 == list(filled)

        kwargs = {"method": "b", "fields": ["a"]}
        new_value_4 = {"a": "too short!", "b": "too short!", "c": None}
        more_values_4 = [values[0], new_value_4, values[2]]
        assert more_values_4 == list(pr.fillempty(records, **kwargs))

    def test_merge(self):
        expected = {"a": 1, "b": 10, "c": 11}
        result = pr.merge([{"a": 1, "b": 2}, {"b": 10, "c": 11}])
        assert expected == result

        # setup
        records = [{"a": 1, "b": 2, "c": 3}, {"b": 4, "c": 5, "d": 6}]

        # Combine all keys
        expected = {"a": 1, "c": 8, "b": 6, "d": 6}
        result = pr.merge(records, pred=bool, op=sum)
        assert expected == result

        first = lambda pair: next(filter(partial(is_not, None), pair))
        kwargs = {"pred": bool, "op": first, "default": None}
        expected = {"a": 1, "b": 2, "c": 3, "d": 6}
        result = pr.merge(records, **kwargs)
        assert expected == result

        # This will only reliably give the expected result for 2 records
        kwargs = {"pred": bool, "op": stats.mean, "default": None}
        expected = {"a": 1, "b": 3.0, "c": 4.0, "d": 6.0}
        result = pr.merge(records, **kwargs)
        assert expected == result

        # Only combine key 'b'
        expected = {"a": 1, "b": 6, "c": 5, "d": 6}
        result = pr.merge(records, pred="b", op=sum)
        assert expected == result

        # Only combine keys that have the same value of 'b'
        expected = {"a": 1, "b": 6, "c": 5, "d": 6}
        result = pr.merge(records, pred=itemgetter("b"), op=sum)
        assert expected == result

        # This will reliably work for any number of records
        counted = defaultdict(int)

        records = [
            {"a": 1, "b": 4, "c": 0},
            {"a": 2, "b": 5, "c": 2},
            {"a": 3, "b": 6, "d": 7},
        ]

        for r in records:
            for k in r.keys():
                counted[k] += 1

        expected = {"a": 3, "b": 3, "c": 2, "d": 1}
        assert expected == counted

        summed = pr.merge(records, pred=bool, op=sum)
        expected = {"a": 6, "b": 15, "c": 2, "d": 7}
        assert expected == summed

        kwargs = {"pred": bool, "op": ft.fpartial(truediv)}
        expected = {"a": 2.0, "b": 5.0, "c": 1.0, "d": 7.0}
        result = pr.merge([summed, counted], **kwargs)
        assert expected == result

        # This should also reliably work for any number of records
        op = ft.fpartial(ft.sum_and_count)
        kwargs = {"pred": bool, "op": op, "default": None}
        merged = pr.merge(records, **kwargs)
        result = {x: truediv(*y) for x, y in merged.items()}
        assert expected == result

    def test_unique(self):
        records = [
            {"day": 1, "name": "bill"},
            {"day": 1, "name": "bob"},
            {"day": 1, "name": "tom"},
            {"day": 2, "name": "bill"},
            {"day": 2, "name": "bob"},
            {"day": 2, "name": "Iñtërnâtiônàližætiøn"},
            {"day": 3, "name": "Iñtërnâtiônàližætiøn"},
            {"day": 3, "name": "bob"},
            {"day": 3, "name": "rob"},
        ]

        pred = lambda x: x["name"][0]
        result = next(it.islice(pr.unique(records, pred=pred), 3, 4))["name"]
        assert "rob" == result

    def test_cut(self):
        records = [
            {"field_1": 1, "field_2": "bill", "field_3": "male"},
            {"field_1": 2, "field_2": "bob", "field_3": "male"},
            {"field_1": 3, "field_2": "jane", "field_3": "female"},
        ]

        expected = {"field_1": 1, "field_3": "male"}
        result = next(pr.cut(records, ["field_2"], exclude=True))
        assert expected == result

        result = next(pr.cut(records, ["field_2"]))
        assert {"field_2": "bill"} == result

    def test_grep(self):
        records = [
            {"day": 1, "name": "bill"},
            {"day": 1, "name": "rob"},
            {"day": 1, "name": "jane"},
            {"day": 2, "name": "rob"},
            {"day": 3, "name": "jane"},
        ]

        rules = [{"fields": ["day"], "pattern": partial(eq, 1)}]
        result = next(pr.grep(records, rules))["name"]
        assert "bill" == result

        rules = [{"pattern": partial(contains, {1, "rob"})}]
        result = next(pr.grep(records, rules))["name"]
        assert "rob" == result

        rules = [{"pattern": partial(contains, {1, "rob"})}]
        result = next(pr.grep(records, rules, any_match=True))["name"]
        assert "bill" == result

        rules = [{"fields": ["name"], "pattern": "o"}]
        result = next(pr.grep(records, rules, inverse=True))["name"]
        assert "bill" == result

    def test_pivot(self):
        records = [
            {"A": "foo", "B": "one", "C": "small", "D": 1},
            {"A": "foo", "B": "one", "C": "large", "D": 2},
            {"A": "foo", "B": "one", "C": "large", "D": 2},
            {"A": "foo", "B": "two", "C": "small", "D": 3},
            {"A": "foo", "B": "two", "C": "small", "D": 3},
            {"A": "bar", "B": "one", "C": "large", "D": 4},
            {"A": "bar", "B": "one", "C": "small", "D": 5},
            {"A": "bar", "B": "two", "C": "small", "D": 6},
            {"A": "bar", "B": "two", "C": "large", "D": 7},
        ]

        expected = [
            {"A": "bar", "B": "one", "large": 4, "small": 5},
            {"A": "foo", "B": "one", "large": 4, "small": 1},
            {"A": "bar", "B": "two", "large": 7, "small": 6},
            {"A": "foo", "B": "two", "large": None, "small": 6},
        ]

        result = list(pr.pivot(records, "D", "C", dropna=False))
        expected_set = set(tuple(sorted(r.items())) for r in expected)
        result_set = set(tuple(sorted(r.items())) for r in result)
        assert expected_set == result_set
