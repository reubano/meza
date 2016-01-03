# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab
"""
tests.test_main
~~~~~~~~~~~~~~~

Provides main unit tests.
"""
from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import nose.tools as nt
import itertools as it
import requests
import responses

from io import StringIO
from operator import itemgetter
from builtins import *

from tabutils import fntools as ft, io, stats


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print('Site Module Setup\n')


class TestIterStringIO:
    def test_strip(self):
        nt.assert_equal(ft.strip('2,123.45'), '2123.45')

        parsed = ft.strip('2.123,45', thousand_sep='.', decimal_sep=',')
        nt.assert_equal(parsed, '2123.45')
        nt.assert_equal(ft.strip('spam'), 'spam')

    def test_is_numeric(self):
        nt.assert_true(ft.is_numeric('2,123.45'))
        nt.assert_true(ft.is_numeric('2.123,45'))
        nt.assert_true(ft.is_numeric('0.45'))
        nt.assert_true(ft.is_numeric(1))
        nt.assert_true(ft.is_numeric('10e5'))
        nt.assert_false(ft.is_numeric('spam'))
        nt.assert_false(ft.is_numeric('02139'))
        nt.assert_true(ft.is_numeric('02139', strip_zeros=True))
        nt.assert_false(ft.is_numeric('spam'))
        nt.assert_false(ft.is_numeric(None))
        nt.assert_false(ft.is_numeric(''))

    def test_is_int(self):
        nt.assert_false(ft.is_int('5/4/82'))

    def test_is_bool(self):
        nt.assert_true(ft.is_bool('y'))
        nt.assert_true(ft.is_bool(1))
        nt.assert_true(ft.is_bool(False))
        nt.assert_true(ft.is_bool('false'))
        nt.assert_true(ft.is_bool('n'))
        nt.assert_true(ft.is_bool(0))
        nt.assert_false(ft.is_bool(''))
        nt.assert_false(ft.is_bool(None))

    def test_is_null(self):
        nt.assert_false(ft.is_null(''))
        nt.assert_false(ft.is_null(' '))
        nt.assert_false(ft.is_null(False))
        nt.assert_false(ft.is_null('0'))
        nt.assert_false(ft.is_null(0))
        nt.assert_true(ft.is_null('', blanks_as_nulls=True))
        nt.assert_true(ft.is_null(' ', blanks_as_nulls=True))

    def test_byte_array(self):
        content = 'Iñtërnâtiônàližætiøn'
        value = bytearray('Iñtërnâtiônàližætiøn'.encode('utf-8'))
        nt.assert_equal(ft.byte(content), value)
        nt.assert_equal(ft.byte(iter(content)), value)
        nt.assert_equal(ft.byte(list(content)), value)

    def test_afterish(self):
        nt.assert_equal(ft.afterish('1001', '.'), -1)
        nt.assert_equal(ft.afterish('1,001'), 3)
        nt.assert_equal(ft.afterish('2,100,001.00'), 6)
        nt.assert_equal(ft.afterish('2,100,001.00', exclude='.'), 3)
        nt.assert_equal(ft.afterish('1,000.00', '.', ','), 2)

        with nt.assert_raises(TypeError):
            ft.afterish('eggs', '.')

    def test_get_separators(self):
        value = {'thousand_sep': ',', 'decimal_sep': '.'}
        nt.assert_equal(ft.get_separators('2,123.45'), value)

        value = {'thousand_sep': '.', 'decimal_sep': ','}
        nt.assert_equal(ft.get_separators('2.123,45'), value)

        with nt.assert_raises(TypeError):
            ft.get_separators('spam')

    def test_fill(self):
        content = 'column_a,column_b,column_c\n'
        content += '1,27,,too long!\n,too short!\n0,mixed types.uh oh,17'
        f = StringIO(content)
        records = io.read_csv(f)
        previous = {}
        current = next(records)
        value = {'column_a': '1', 'column_b': '27', 'column_c': ''}
        nt.assert_equal(current, value)

        length = len(current)
        filled = ft.fill(previous, current, value=0)
        previous = dict(it.islice(filled, length))
        count = next(filled)
        nt.assert_equal(count, {'column_a': 0, 'column_b': 0, 'column_c': 1})

        value = {'column_a': '1', 'column_b': '27', 'column_c': 0}
        nt.assert_equal(previous, value)

        current = next(records)

        value = {
            'column_a': '',
            'column_b': u"too short!",
            'column_c': None,
        }

        nt.assert_equal(current, value)

        kwargs = {'fill_key': 'column_b', 'count': count}
        filled = ft.fill(previous, current, **kwargs)
        previous = dict(it.islice(filled, length))
        count = next(filled)
        nt.assert_equal(count, {'column_a': 1, 'column_b': 0, 'column_c': 2})

        value = {
            'column_a': u"too short!",
            'column_b': u"too short!",
            'column_c': u"too short!",
        }

        nt.assert_equal(previous, value)

    @responses.activate
    def test_chunk(self):
        content = io.StringIO('Iñtërnâtiônàližætiøn')
        nt.assert_equal(next(ft.chunk(content, 5)), 'Iñtër')
        nt.assert_equal(next(ft.chunk(content)), 'nâtiônàližætiøn')

        url = 'http://google.com'
        body = '<!doctype html><html itemtype="http://schema.org/page">'
        responses.add(responses.GET, url=url, body=body)
        r = requests.get(url, stream=True)

        # http://docs.python-requests.org/en/latest/api/
        # The chunk size is the number of bytes it should read into
        # memory. This is not necessarily the length of each item returned
        # as decoding can take place.
        nt.assert_equal(len(next(ft.chunk(r.iter_content, 20, 29, 200))), 20)
        nt.assert_equal(len(next(ft.chunk(r.iter_content))), 55)

    def test_combine(self):
        records = [{'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]

        # Combine all keys
        pred = lambda key: True
        x, y = records[0], records[1]
        nt.assert_equal(ft.combine(x, y, 'a', pred=pred, op=sum), 1)
        nt.assert_equal(ft.combine(x, y, 'b', pred=pred, op=sum), 6)
        nt.assert_equal(ft.combine(x, y, 'c', pred=pred, op=sum), 8)

        fltrer = lambda x: x is not None
        first = lambda x: next(filter(fltrer, x))
        kwargs = {'pred': pred, 'op': first, 'default': None}
        nt.assert_equal(ft.combine(x, y, 'b', **kwargs), 2)

        kwargs = {'pred': pred, 'op': stats.mean, 'default': None}
        nt.assert_equal(ft.combine(x, y, 'a', **kwargs), 1.0)
        nt.assert_equal(ft.combine(x, y, 'b', **kwargs), 3.0)

        # Only combine key 'b'
        pred = lambda key: key == 'b'
        nt.assert_equal(ft.combine(x, y, 'c', pred=pred, op=sum), 5)

        # Only combine keys that have the same value of 'b'
        pred = itemgetter('b')
        nt.assert_equal(ft.combine(x, y, 'b', pred=pred, op=sum), 6)
        nt.assert_equal(ft.combine(x, y, 'c', pred=pred, op=sum), 5)

    def test_op_everseen(self):
        content = [4, 6, 3, 8, 2, 1]
        value = [4, 4, 3, 3, 2, 1]
        nt.assert_equal(list(ft.op_everseen(content, pad=True)), value)
        nt.assert_equal(list(ft.op_everseen(content, op='gt')), [4, 6, 8])
