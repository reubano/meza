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
import requests
import responses

from os import path as p
from json import loads
from tempfile import TemporaryFile
from StringIO import StringIO

from tabutils import io, convert as cv, ENCODING


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print('Site Module Setup\n')


class TestIterStringIO:
    def __init__(self):
        self.phrase = io.IterStringIO(iter('Hello World'))
        self.text = io.IterStringIO('line one\nline two\nline three\n')
        self.ints = io.IterStringIO('0123456789', 5)

    def test_lines(self):
        nt.assert_equal(self.phrase.read(5), bytearray(b'Hello'))
        self.phrase.write(iter('ly person'))
        nt.assert_equal(self.phrase.read(8), bytearray(b' Worldly'))

        self.phrase.write(': Iñtërnâtiônàližætiøn')
        value = bytearray(b' person: Iñtërnâtiônàližætiøn')
        nt.assert_equal(self.phrase.read(), value)

        nt.assert_equal(self.text.readline(), bytearray(b'line one'))
        nt.assert_equal(self.text.next(), bytearray(b'line two'))

        self.text.seek(0)
        nt.assert_equal(self.text.next(), bytearray(b'line one'))
        nt.assert_equal(self.text.next(), bytearray(b'line two'))
        nt.assert_equal(self.text.tell(), 16)

        self.text.seek(0)
        lines = list(self.text.readlines())
        nt.assert_equal(lines[2], bytearray(b'line three'))

    def test_seeking(self):
        nt.assert_equal(self.ints.read(5), bytearray(b'01234'))
        self.ints.seek(0)
        nt.assert_equal(self.ints.read(1), bytearray(b'0'))
        nt.assert_equal(self.ints.read(1), bytearray(b'1'))
        nt.assert_equal(self.ints.read(1), bytearray(b'2'))

        self.ints.seek(3)
        nt.assert_equal(self.ints.read(1), bytearray(b'3'))

        self.ints.seek(6)
        nt.assert_equal(self.ints.read(1), bytearray(b'6'))

        self.ints.seek(3)
        nt.assert_equal(self.ints.read(1), bytearray(b'3'))

        self.ints.seek(3)
        nt.assert_equal(self.ints.read(1), bytearray(b'3'))

        self.ints.seek(4)
        nt.assert_equal(self.ints.read(1), bytearray(b'4'))

        self.ints.seek(6)
        nt.assert_equal(self.ints.read(1), bytearray(b'6'))

        self.ints.seek(0)
        nt.assert_equal(self.ints.read(1), bytearray(b'2'))


class TestUnicodeReader:
    """Unit tests for file IO"""
    def __init__(self):
        self.cls_initialized = False
        self.row1 = {'a': '1', 'b': '2', 'c': '3'}
        self.row2 = {'a': '4', 'b': '5', 'c': '©'}
        self.row3 = {'a': '4', 'b': '5', 'c': 'ʤ'}

    def test_utf8(self):
        filepath = p.join(io.DATA_DIR, 'utf8.csv')
        records = io.read_csv(filepath, sanitize=True)
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())

    def test_latin1(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')
        records = io.read_csv(filepath, encoding='latin1')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())

    def test_encoding_detection(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')
        records = io.read_csv(filepath, encoding='ascii')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row2, records.next())

    def test_utf16_big(self):
        filepath = p.join(io.DATA_DIR, 'utf16_big.csv')
        records = io.read_csv(filepath, encoding='utf-16-be')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())

    def test_utf16_little(self):
        filepath = p.join(io.DATA_DIR, 'utf16_little.csv')
        records = io.read_csv(filepath, encoding='utf-16-le')
        nt.assert_equal(self.row1, records.next())
        nt.assert_equal(self.row3, records.next())

    def test_kwargs(self):
        filepath = p.join(io.DATA_DIR, 'utf8.csv')
        kwargs = {'delimiter': ','}
        records = io.read_csv(filepath, **kwargs)
        nt.assert_equal(self.row1, records.next())


class TestInput:
    def __init__(self):
        self.cls_initialized = False
        self.sheet0 = {
            'sparse_data': 'Iñtërnâtiônàližætiøn',
            'some_date': '1982-05-04',
            'some_value': '234.0',
            'unicode_test': 'Ādam'}

        self.sheet0_alt = {
            'sparse_data': 'Iñtërnâtiônàližætiøn',
            'some_date': '05/04/82',
            'some_value': '234',
            'unicode_test': 'Ādam'}

        self.sheet1 = {
            'boolean': 'False',
            'date': '1915-12-31',
            'datetime': '1915-12-31',
            'float': '41800000.01',
            'integer': '164.0',
            'text': 'Chicago Tribune',
            'time': '00:00:00'}

    def test_newline_json(self):
        value = (
            '{"sepal_width": "3.5", "petal_width": "0.2", "species":'
            ' "Iris-setosa", "sepal_length": "5.1", "petal_length": "1.4"}')

        filepath = p.join(io.DATA_DIR, 'iris.csv')
        records = io.read_csv(filepath)
        json = cv.records2json(records, newline=True)
        nt.assert_equal(value, json.next().strip())

        filepath = p.join(io.DATA_DIR, 'newline.json')
        records = io.read_json(filepath, newline=True)
        nt.assert_equal({'a': 2, 'b': 3}, records.next())

    def test_xls(self):
        filepath = p.join(io.DATA_DIR, 'test.xlsx')
        records = io.read_xls(filepath, sanitize=True, sheet=0)
        nt.assert_equal(self.sheet0, records.next())

        with open(filepath, 'r+b') as f:
            records = io.read_xls(f, sanitize=True, sheet=0)
            nt.assert_equal(self.sheet0, records.next())

        records = io.read_xls(filepath, sanitize=True, sheet=1)
        nt.assert_equal(self.sheet1, records.next())

        kwargs = {'first_row': 1, 'first_col': 1}
        records = io.read_xls(filepath, sanitize=True, sheet=2, **kwargs)
        nt.assert_equal(self.sheet0, records.next())

        records = io.read_xls(filepath, sanitize=True, sheet=3, **kwargs)
        nt.assert_equal(self.sheet1, records.next())

    def test_csv(self):
        filepath = p.join(io.DATA_DIR, 'test.csv')
        header = ['some_date', 'sparse_data', 'some_value', 'unicode_test']

        with open(filepath, 'rU') as f:
            records = io._read_csv(f, 'utf-8', header)
            nt.assert_equal(self.sheet0_alt, records.next())

        filepath = p.join(io.DATA_DIR, 'no_header_row.csv')
        records = io.read_csv(filepath, has_header=False)
        value = {'column_1': '1', 'column_2': '2', 'column_3': '3'}
        nt.assert_equal(value, records.next())

        filepath = p.join(io.DATA_DIR, 'fixed_w_header.txt')
        widths = [0, 18, 29, 33, 38, 50]
        records = io.read_fixed_csv(filepath, widths, has_header=True)
        value = {
            'News Paper': 'Chicago Reader',
            'Founded': '1971-01-01',
            'Int': '40',
            'Bool': 'True',
            'Float': '1.0',
            'Timestamp': '04:14:001971-01-01T04:14:00'}

        nt.assert_equal(value, records.next())

    def test_dbf(self):
        filepath = p.join(io.DATA_DIR, 'test.dbf')

        with open(filepath, 'rb') as f:
            records = io.read_dbf(f, sanitize=True)
            value = {
                'awater10': 12416573076,
                'aland10': 71546663636,
                'intptlat10': '+47.2400052',
                'lsad10': 'C2',
                'cd111fp': '08',
                'namelsad10': 'Congressional District 8',
                'funcstat10': 'N',
                'statefp10': '27',
                'cdsessn': '111',
                'mtfcc10': 'G5200',
                'geoid10': '2708',
                'intptlon10': '-092.9323194'}

            nt.assert_equal(value, records.next())

    def test_get_reader(self):
        nt.assert_true(callable(io.get_reader('csv')))

        with nt.assert_raises(KeyError):
            io.get_reader('')

    def test_get_utf8(self):
        with open(p.join(io.DATA_DIR, 'utf16_big.csv')) as f:
            utf8_f = io.get_utf8(f, 'utf-16-be')
            nt.assert_equal('a,b,c', utf8_f.next().strip())
            nt.assert_equal('1,2,3', utf8_f.next().strip())
            nt.assert_equal('4,5,ʤ', utf8_f.next().decode(ENCODING))


class TestGeoJSON:
    def __init__(self):
        self.cls_initialized = False
        self.bbox = [100, 0, 105, 1]
        self.filepath = p.join(io.DATA_DIR, 'test.geojson')

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


class TestOutput:
    @responses.activate
    def test_write(self):
        url = 'http://google.com'
        body = '<!doctype html><html itemtype="http://schema.org/page">'
        content = StringIO('Iñtërnâtiônàližætiøn')
        nt.assert_equal(io.write(TemporaryFile(), content), 20)

        content = io.IterStringIO(iter('Hello World'))
        nt.assert_equal(io.write(TemporaryFile(), content, chunksize=2), 12)

        responses.add(responses.GET, url=url, body=body)
        r = requests.get(url, stream=True)
        nt.assert_equal(io.write(TemporaryFile(), r.iter_content), 55)
