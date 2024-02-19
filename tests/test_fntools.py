# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab
"""
tests.test_fntools
~~~~~~~~~~~~~~~~~~

Provides main unit tests.
"""
import itertools as it
import requests
import responses

from io import StringIO
from operator import itemgetter

import pytest

from meza import fntools as ft, io, stats


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print("Site Module Setup\n")


class TestIterStringIO:
    def test_strip(self):
        assert "2123.45" == ft.strip("2,123.45")

        parsed = ft.strip("2.123,45", thousand_sep=".", decimal_sep=",")
        assert "2123.45" == parsed
        assert "spam" == ft.strip("spam")

    def test_is_numeric(self):
        assert ft.is_numeric("2,123.45")
        assert ft.is_numeric("2.123,45")
        assert ft.is_numeric("0.45")
        assert ft.is_numeric(1)
        assert ft.is_numeric("10e5")
        assert not ft.is_numeric("spam")
        assert not ft.is_numeric("02139")
        assert ft.is_numeric("02139", strip_zeros=True)
        assert not ft.is_numeric("spam")
        assert not ft.is_numeric(None)
        assert not ft.is_numeric("")

    def test_is_numeric_currency_zero_value(self):
        """Regression test for https://github.com/reubano/meza/issues/36"""
        for sym in ft.CURRENCIES:
            assert ft.is_numeric(f"0{sym}")
            assert ft.is_numeric(f"{sym}0")

    def test_is_int(self):
        assert not ft.is_int("5/4/82")

    def test_is_bool(self):
        assert ft.is_bool("y")
        assert ft.is_bool(1)
        assert ft.is_bool(False)
        assert ft.is_bool("false")
        assert ft.is_bool("n")
        assert ft.is_bool(0)
        assert not ft.is_bool("")
        assert not ft.is_bool(None)

    def test_is_null(self):
        assert not ft.is_null("")
        assert not ft.is_null(" ")
        assert not ft.is_null(False)
        assert not ft.is_null("0")
        assert not ft.is_null(0)
        assert ft.is_null("", blanks_as_nulls=True)
        assert ft.is_null(" ", blanks_as_nulls=True)

    def test_byte_array(self):
        content = "Iñtërnâtiônàližætiøn"
        expected = bytearray("Iñtërnâtiônàližætiøn".encode("utf-8"))
        assert expected == ft.byte(content)
        assert expected == ft.byte(iter(content))
        assert expected == ft.byte(list(content))

    def test_afterish(self):
        assert -1 == ft.afterish("1001", ".")
        assert 3 == ft.afterish("1,001")
        assert 3 == ft.afterish("2,100,001.00")
        assert 2 == ft.afterish("1,000.00", ".")

        with pytest.raises(ValueError):
            ft.afterish("eggs", ".")

    def test_get_separators(self):
        expected = {"thousand_sep": ",", "decimal_sep": "."}
        assert expected == ft.get_separators("2,123.45")

        expected = {"thousand_sep": ".", "decimal_sep": ","}
        assert expected == ft.get_separators("2.123,45")

        with pytest.raises(ValueError):
            ft.get_separators("spam")

    def test_fill(self):
        content = "column_a,column_b,column_c\n"
        content += "1,27,,too long!\n,too short!\n0,mixed types.uh oh,17"
        f = StringIO(content)
        records = io.read_csv(f)
        previous = {}
        current = next(records)
        expected = {"column_a": "1", "column_b": "27", "column_c": ""}
        assert expected == current

        length = len(current)
        filled = ft.fill(previous, current, value=0)
        previous = dict(it.islice(filled, length))
        count = next(filled)
        assert count == {"column_a": 0, "column_b": 0, "column_c": 1}

        expected = {"column_a": "1", "column_b": "27", "column_c": 0}
        assert expected == previous

        current = next(records)

        expected = {
            "column_a": "",
            "column_b": "too short!",
            "column_c": None,
        }

        assert expected == current

        kwargs = {"fill_key": "column_b", "count": count}
        filled = ft.fill(previous, current, **kwargs)
        previous = dict(it.islice(filled, length))
        count = next(filled)
        assert {"column_a": 1, "column_b": 0, "column_c": 2} == count

        expected = {
            "column_a": "too short!",
            "column_b": "too short!",
            "column_c": "too short!",
        }

        assert expected == previous

    @responses.activate
    def test_chunk(self):
        content = io.StringIO("Iñtërnâtiônàližætiøn")
        assert "Iñtër" == next(ft.chunk(content, 5))
        assert "nâtiônàližætiøn" == next(ft.chunk(content))

        url = "http://google.com"
        body = '<!doctype html><html itemtype="http://schema.org/page">'
        responses.add(responses.GET, url=url, body=body)
        r = requests.get(url, stream=True)

        # http://docs.python-requests.org/en/latest/api/
        # The chunk size is the number of bytes it should read into
        # memory. This is not necessarily the length of each item returned
        # as decoding can take place.
        assert 20 == len(next(ft.chunk(r.iter_content, 20, 29, 200)))
        assert 55 == len(next(ft.chunk(r.iter_content)))

    def test_combine(self):
        records = [{"a": 1, "b": 2, "c": 3}, {"b": 4, "c": 5, "d": 6}]

        # Combine all keys
        pred = lambda key: True
        x, y = records[0], records[1]
        assert 1 == ft.combine(x, y, "a", pred=pred, op=sum)
        assert 6 == ft.combine(x, y, "b", pred=pred, op=sum)
        assert 8 == ft.combine(x, y, "c", pred=pred, op=sum)

        fltrer = lambda x: x is not None
        first = lambda x: next(filter(fltrer, x))
        kwargs = {"pred": pred, "op": first, "default": None}
        assert 2 == ft.combine(x, y, "b", **kwargs)

        kwargs = {"pred": pred, "op": stats.mean, "default": None}
        assert 1.0 == ft.combine(x, y, "a", **kwargs)
        assert 3.0 == ft.combine(x, y, "b", **kwargs)

        # Only combine key 'b'
        pred = lambda key: key == "b"
        assert 5 == ft.combine(x, y, "c", pred=pred, op=sum)

        # Only combine keys that have the same value of 'b'
        pred = itemgetter("b")
        assert 6 == ft.combine(x, y, "b", pred=pred, op=sum)
        assert 5 == ft.combine(x, y, "c", pred=pred, op=sum)

    def test_op_everseen(self):
        content = [4, 6, 3, 8, 2, 1]
        expected = [4, 4, 3, 3, 2, 1]
        assert expected == list(ft.op_everseen(content, pad=True))
        assert [4, 6, 8] == list(ft.op_everseen(content, op="gt"))

    def test_objectify(self):
        kwargs = {"one": "1", "two": "2"}
        kw = ft.Objectify(kwargs, func=int)
        assert kw.one == 1
        assert kw["two"] == 2
        assert dict(kw) == {"one": 1, "two": 2}
