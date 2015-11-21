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

from os import path as p
from json import loads

from tabutils import io, convert as cv

parent_dir = p.abspath(p.dirname(p.dirname(__file__)))
test_dir = p.join(parent_dir, 'data', 'test')


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print('Site Module Setup\n')


class TestUnicodeReader:
    """Unit tests for file IO"""
    def __init__(self):
        self.cls_initialized = False
        self.row1 = {'a': '1', 'b': '2', 'c': '3'}
        self.row2 = {'a': '4', 'b': '5', 'c': '©'}
        self.row3 = {'a': '4', 'b': '5', 'c': 'ʤ'}

    def test_utf8(self):
        filepath = p.join(test_dir, 'utf8.csv')
        records = io.read_csv(filepath, sanitize=True)
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())

    def test_latin1(self):
        filepath = p.join(test_dir, 'latin1.csv')
        records = io.read_csv(filepath, encoding='latin1')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())

    def test_encoding_detection(self):
        filepath = p.join(test_dir, 'latin1.csv')
        records = io.read_csv(filepath, encoding='ascii')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())

    def test_utf16_big(self):
        filepath = p.join(test_dir, 'utf16_big.csv')
        records = io.read_csv(filepath, encoding='utf-16-be')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())

    def test_utf16_little(self):
        filepath = p.join(test_dir, 'utf16_little.csv')
        records = io.read_csv(filepath, encoding='utf-16-le')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())


class TestIO:
    def __init__(self):
        self.cls_initialized = False
        self.value = {
            u'sparse_data': u'Iñtërnâtiônàližætiøn',
            u'some_date': u'05/04/82',
            u'some_value': u'234',
            u'unicode_test': u'Ādam'}

    def test_newline_json(self):
        value = (
            '{"sepal_width": "3.5", "petal_width": "0.2", "species":'
            ' "Iris-setosa", "sepal_length": "5.1", "petal_length": "1.4"}')

        filepath = p.join(test_dir, 'iris.csv')
        records = io.read_csv(filepath)
        json = cv.records2json(records, newline=True)
        nt.assert_equal(value, json.next().strip())

        filepath = p.join(test_dir, 'newline.json')
        records = io.read_json(filepath, newline=True)
        nt.assert_equal({'a': 2, 'b': 3}, records.next())


class TestGeoJSON:
    def __init__(self):
        self.cls_initialized = False
        self.bbox = [100, 0, 105, 1]
        self.filepath = p.join(test_dir, 'test.geojson')

    def test_geojson(self):
        value = {
            'id': None,
            'prop0': 'value0',
            'type': 'Point',
            'coordinates': [102, 0.5]}

        records = io.read_geojson(self.filepath)
        record = records.next()
        nt.assert_equal(value, record)

        for record in records:
            nt.assert_true('id' in record)
            if record['type'] == 'Point':
                nt.assert_equal(len(record['coordinates']), 2)
            elif record['type'] == 'LineString':
                nt.assert_greater_equal(len(record['coordinates']), 2)
                nt.assert_equal(len(record['coordinates'][0]), 2)
            elif record['type'] == 'Polygon':
                nt.assert_greater_equal(len(record['coordinates']), 1)
                nt.assert_greater_equal(len(record['coordinates'][0]), 3)
                nt.assert_equal(len(record['coordinates'][0][0]), 2)

    def test_geojson_with_id(self):
        records = io.read_geojson(self.filepath)
        f = cv.records2geojson(records, key='id')
        geojson = loads(f.read())

        nt.assert_equal(geojson['type'], 'FeatureCollection')
        nt.assert_true('crs' in geojson)
        nt.assert_equal(geojson['bbox'], self.bbox)
        nt.assert_equal(len(geojson['features']), 3)

        for feature in geojson['features']:
            nt.assert_equal(feature['type'], 'Feature')
            nt.assert_true('id' in feature)
            nt.assert_less_equal(len(feature['properties']), 2)

            geometry = feature['geometry']

            if geometry['type'] == 'Point':
                nt.assert_equal(len(geometry['coordinates']), 2)
            elif geometry['type'] == 'LineString':
                nt.assert_equal(len(geometry['coordinates'][0]), 2)
            elif geometry['type'] == 'Polygon':
                nt.assert_equal(len(geometry['coordinates'][0][0]), 2)

    def test_geojson_with_crs(self):
        records = io.read_geojson(self.filepath)
        f = cv.records2geojson(records, crs='EPSG:4269')
        geojson = loads(f.read())

        nt.assert_true('crs' in geojson)
        nt.assert_equal(geojson['crs']['type'], 'name')
        nt.assert_equal(geojson['crs']['properties']['name'], 'EPSG:4269')
