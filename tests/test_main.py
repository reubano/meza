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

    def test_utf16_big(self):
        filepath = p.join(test_dir, 'utf16_big.csv')
        records = io.read_csv(filepath, encoding='utf-16-be')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())

    def test_utf16_little(self):
        filepath = p.join(test_dir, 'utf16_little.csv')
        records = io.read_csv(filepath, encoding='utf-16-le')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())


class TestIO:
    def __init__(self):
        self.cls_initialized = False
        self.value = {
            u'sparse_data': u'Iñtërnâtiônàližætiøn',
            u'some_date': u'05/04/82',
            u'some_value': u'234',
            u'unicode_test': u'Ādam'}

    def test_newline_json(self):
        value = u'{"sepal_width": "3.5", "petal_width": "0.2", "species": "Iris-setosa", "sepal_length": "5.1", "petal_length": "1.4"}'

        filepath = p.join(test_dir, 'iris.csv')
        records = io.read_csv(filepath)
        json = cv.records2json(records, newline=True)
        nt.assert_equal(value, json.next().strip())

        # filepath = p.join(test_dir, 'newline.json')
        # records = io.read_json(filepath, newline=True)
        # nt.assert_equal(value, records.next().strip())


class TestGeoJSON:
    def __init__(self):
        self.cls_initialized = False
        self.bbox = [-95.334619, 32.299076986939205, -95.250699, 32.351434]

    # def test_geojson(self):
    #     # f = open('examples/test.geojson', 'rt')
    #     value = {}
    #     # 'id,prop0,prop1,geojson'
    #     # '""coordinates"": [102.0, 0.5]'
    #     # '""coordinates"": [[102.0, 0.0], [103.0, 1.0], [104.0, 0.0],
    #     # [105.0, 1.0]]'
    #     filepath = p.join(test_dir, 'test.geojson')
    #     records = io.read_geojson(filepath)
    #     nt.assert_equal(value, records.next())

    #     nt.assert_equal(records['type'], 'FeatureCollection')
    #     nt.assert_false('crs' in records)

    #     nt.assert_equal(records['bbox'], self.bbox)
    #     nt.assert_equal(len(records['features']), 17)

    #     for feature in records['features']:
    #         nt.assert_equal(feature['type'], 'Feature')
    #         nt.assert_false('id' in feature)
    #         nt.assert_equal(len(feature['properties']), 10)

    #         geometry = feature['geometry']

    #         nt.assert_equal(len(geometry['coordinates']), 2)
    #         nt.assert_true(isinstance(geometry['coordinates'][0], float))
    #         nt.assert_true(isinstance(geometry['coordinates'][1], float))

    # def test_geojson_with_id(self):
    #     filepath = p.join(test_dir, 'test.geojson')
    #     records = io.read_geojson(filepath)

    #     geojson = cv.records2geojson(
    #         records, lon='longitude', lat='latitude', key='slug')

    #     nt.assert_equal(geojson['type'], 'FeatureCollection')
    #     nt.assert_false('crs' in geojson)
    #     nt.assert_equal(geojson['bbox'], self.bbox)
    #     nt.assert_equal(len(geojson['features']), 17)

    #     for feature in geojson['features']:
    #         nt.assert_equal(feature['type'], 'Feature')
    #         nt.assert_true('id' in feature)
    #         nt.assert_equal(len(feature['properties']), 9)

    #         geometry = feature['geometry']

    #         nt.assert_equal(len(geometry['coordinates']), 2)
    #         nt.assert_true(isinstance(geometry['coordinates'][0], float))
    #         nt.assert_true(isinstance(geometry['coordinates'][1], float))

    # def test_geojson_with_crs(self):
    #     filepath = p.join(test_dir, 'test.geojson')
    #     records = io.read_geojson(filepath)

    #     geojson = cv.records2geojson(
    #         records, lon='longitude', lat='latitude', key='slug',
    #         crs='EPSG:4269')

    #     nt.assert_equal(geojson['type'], 'FeatureCollection')
    #     nt.assert_true('crs' in geojson)
    #     nt.assert_equal(geojson['bbox'], self.bbox)
    #     nt.assert_equal(len(geojson['features']), 17)

    #     crs = geojson['crs']

    #     nt.assert_equal(crs['type'], 'name')
    #     nt.assert_equal(crs['properties']['name'], 'EPSG:4269')
