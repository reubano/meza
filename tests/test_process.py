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

from decimal import Decimal
from operator import itemgetter
from collections import defaultdict

from tabutils import process as pr


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print('Site Module Setup\n')


class Test:
    """docstring"""
    def test_typecast(self):
        records = [{'float': '1.5'}]
        types = [{'id': 'float', 'type': 'bool'}]
        nt.assert_equal(pr.type_cast(records, types).next(), {'float': False})

        with nt.assert_raises(ValueError):
            pr.type_cast(records, types, warn=True).next()

    def test_detect_types(self):
        record = {
            'null': 'None',
            'bool': 'false',
            'int': '1',
            'float': '1.5',
            'text': 'Iñtërnâtiônàližætiøn',
            'date': '5/4/82',
            'time': '2:30',
            'datetime': '5/4/82 2pm',
        }

        records = it.repeat(record)
        records, result = pr.detect_types(records)

        types = {
            'null': 'null',
            'bool': 'bool',
            'int': 'int',
            'float': 'float',
            'text': 'text',
            'date': 'date',
            'time': 'time',
            'datetime': 'datetime',
        }

        nt.assert_equal({r['id']: r['type'] for r in result['types']}, types)
        nt.assert_equal(records.next(), record)

        result = pr.detect_types(records, 0.99)[1]
        nt.assert_equal(result['count'], 100)
        nt.assert_equal(result['confidence'], Decimal('0.97'))
        nt.assert_false(result['accurate'])

        result = pr.detect_types([record, record])[1]
        nt.assert_equal(result['count'], 2)
        nt.assert_equal(result['confidence'], Decimal('0.87'))
        nt.assert_false(result['accurate'])

    def test_fillempty(self):
        records = [
            {'a': '1', 'b': '27', 'c': ''},
            {'a': '', 'b': 'too short!', 'c': None},
            {'a': '0', 'b': 'mixed', 'c': '17'}]

        values = [
            {'a': '1', 'b': '27', 'c': ''},
            {'a': 0, 'b': 'too short!', 'c': None},
            {'a': '0', 'b': 'mixed', 'c': '17'}]

        new_value_1 = {'a': '1', 'b': 'too short!', 'c': ''}

        more_values_1 = [values[0], new_value_1, values[2]]

        fields = ['a']
        nt.assert_equal(list(pr.fillempty(records, 0, fields=fields)), values)

        filled = pr.fillempty(records, method='front')
        nt.assert_equal(list(filled), more_values_1)

        new_value_2 = {'a': '1', 'b': '27', 'c': '17'}
        new_value_3 = {'a': '0', 'b': 'too short!', 'c': '17'}
        more_values_2 = [new_value_2, new_value_3, values[2]]
        filled = pr.fillempty(records, method='back')
        nt.assert_equal(list(filled), more_values_2)

        more_values_3 = [values[0], new_value_3, values[2]]
        filled = pr.fillempty(records, method='back', limit=1)
        nt.assert_equal(list(filled), more_values_3)

        kwargs = {'method': 'b', 'fields': ['a']}
        new_value_4 = {'a': 'too short!', 'b': 'too short!', 'c': None}
        more_values_4 = [values[0], new_value_4, values[2]]
        nt.assert_equal(list(pr.fillempty(records, **kwargs)), more_values_4)

    def test_merge(self):
        pr.merge([{'a': 1, 'b': 2}, {'b': 10, 'c': 11}])
        [('a', 1), ('b', 10), ('c', 11)]
        records = [{'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': 5, 'd': 6}]
        # Combine all keys
        pred = lambda key: True
        pr.merge(records, pred=pred, op=sum)
        [('a', 1), ('b', 6), ('c', 8), ('d', 6)]
        fltrer = lambda x: x is not None
        first = lambda pair: filter(fltrer, pair)[0]
        kwargs = {'pred': pred, 'op': first, 'default': None}
        pr.merge(records, **kwargs)
        [('a', 1), ('b', 2), ('c', 3), ('d', 6)]
        # This will only reliably give the expected result for 2 records
        average = lambda x: sum(filter(fltrer, x)) / len(filter(fltrer, x))
        kwargs = {'pred': pred, 'op': average, 'default': None}
        pr.merge(records, **kwargs)
        [('a', 1), ('b', 3.0), ('c', 4.0), ('d', 6.0)]
        # Only combine key 'b'
        pred = lambda key: key == 'b'
        pr.merge(records, pred=pred, op=sum)
        [('a', 1), ('b', 6), ('c', 5), ('d', 6)]
        # Only combine keys that have the same value of 'b'
        pred = itemgetter('b')
        pr.merge(records, pred=pred, op=sum)
        [('a', 1), ('b', 6), ('c', 5), ('d', 6)]
        # This will reliably work for any number of records
        counted = defaultdict(int)
        pred = lambda key: True
        divide = lambda x: x[0] / x[1]

        records = [
            {'a': 1, 'b': 4, 'c': 0},
            {'a': 2, 'b': 5, 'c': 2},
            {'a': 3, 'b': 6, 'd': 7}]

        for r in records:
            for k in r.keys():
                counted[k] += 1

        counted
        [('a', 3), ('b', 3), ('c', 2), ('d', 1)]
        summed = pr.merge(records, pred=pred, op=sum)
        summed
        [('a', 6), ('b', 15), ('c', 2), ('d', 7)]
        kwargs = {'pred': pred, 'op': divide}
        pr.merge([summed, counted], **kwargs)
        [('a', 2.0), ('b', 5.0), ('c', 1.0), ('d', 7.0)]

    def test_unique(self):
        records = [
            {'day': 1, 'name': 'bill'},
            {'day': 1, 'name': 'bob'},
            {'day': 1, 'name': 'tom'},
            {'day': 2, 'name': 'bill'},
            {'day': 2, 'name': 'bob'},
            {'day': 2, 'name': 'Iñtërnâtiônàližætiøn'},
            {'day': 3, 'name': 'Iñtërnâtiônàližætiøn'},
            {'day': 3, 'name': 'bob'},
            {'day': 3, 'name': 'rob'},
        ]

        pred = lambda x: x['name'][0]
        it.islice(pr.unique(records, pred=pred), 3, 4).next()['name']
        'rob'

    def test_cut(self):
        records = [
            {'field_1': 1, 'field_2': 'bill', 'field_3': 'male'},
            {'field_1': 2, 'field_2': 'bob', 'field_3': 'male'},
            {'field_1': 3, 'field_2': 'jane', 'field_3': 'female'},
        ]

        pr.cut(records, exclude=['field_2']).next() == {
            'field_1': 1, 'field_3': 'male'}
        True
        pr.cut(records, include=['field_2'], exclude=['field_2']).next()
        {'field_2': 'bill'}

    def test_grep(self):
        records = [
            {'day': 1, 'name': 'bill'},
            {'day': 1, 'name': 'rob'},
            {'day': 1, 'name': 'jane'},
            {'day': 2, 'name': 'rob'},
            {'day': 3, 'name': 'jane'},
        ]

        rules = [{'fields': ['day'], 'pattern': lambda x: x == 1}]
        pr.grep(records, rules).next()['name']
        'bill'
        rules = [{'pattern': lambda x: x in {1, 'rob'}}]
        pr.grep(records, rules).next()['name']
        'rob'
        rules = [{'pattern': lambda x: x in {1, 'rob'}}]
        pr.grep(records, rules, any_match=True).next()['name']
        'bill'
        rules = [{'fields': ['name'], 'pattern': 'o'}]
        pr.grep(records, rules, inverse=True).next()['name']
        'bill'
