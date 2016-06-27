# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab
"""
tests.test_io
~~~~~~~~~~~~~

Provides main unit tests.
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import itertools as it
import requests
import responses
import nose.tools as nt
import pygogo as gogo

from os import path as p
from json import loads
from tempfile import TemporaryFile
from io import StringIO
from decimal import Decimal
from six.moves.urllib.request import urlopen
from contextlib import closing

from builtins import *
from meza import io, convert as cv

logger = gogo.Gogo(__name__, monolog=True).logger


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
        nt.assert_equal(bytearray(b'Hello'), self.phrase.read(5))
        self.phrase.write(iter('ly person'))
        nt.assert_equal(bytearray(b' Worldly'), self.phrase.read(8))

        self.phrase.write(': Iñtërnâtiônàližætiøn')
        expected = bytearray(' person: Iñtërnâtiônàližætiøn'.encode('utf-8'))
        nt.assert_equal(expected, self.phrase.read())

        nt.assert_equal(bytearray(b'line one'), self.text.readline())
        nt.assert_equal(bytearray(b'line two'), next(self.text))

        self.text.seek(0)
        nt.assert_equal(bytearray(b'line one'), next(self.text))
        nt.assert_equal(bytearray(b'line two'), next(self.text))
        nt.assert_equal(16, self.text.tell())

        self.text.seek(0)
        lines = list(self.text.readlines())
        nt.assert_equal(bytearray(b'line three'), lines[2])

    def test_seeking(self):
        nt.assert_equal(bytearray(b'01234'), self.ints.read(5))
        self.ints.seek(0)
        nt.assert_equal(bytearray(b'0'), self.ints.read(1))
        nt.assert_equal(bytearray(b'1'), self.ints.read(1))
        nt.assert_equal(bytearray(b'2'), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b'3'), self.ints.read(1))

        self.ints.seek(6)
        nt.assert_equal(bytearray(b'6'), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b'3'), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b'3'), self.ints.read(1))

        self.ints.seek(4)
        nt.assert_equal(bytearray(b'4'), self.ints.read(1))

        self.ints.seek(6)
        nt.assert_equal(bytearray(b'6'), self.ints.read(1))

        self.ints.seek(0)
        nt.assert_equal(bytearray(b'2'), self.ints.read(1))


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
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_latin1(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')
        records = io.read_csv(filepath, encoding='latin-1')
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_bytes_encoding_detection(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')
        records = io.read_csv(filepath, mode='rb')
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_wrong_encoding_detection(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')
        records = io.read_csv(filepath, encoding='ascii')
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_utf16_big(self):
        filepath = p.join(io.DATA_DIR, 'utf16_big.csv')
        records = io.read_csv(filepath, encoding='utf-16-be')
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_utf16_little(self):
        filepath = p.join(io.DATA_DIR, 'utf16_little.csv')
        records = io.read_csv(filepath, encoding='utf-16-le')
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_kwargs(self):
        filepath = p.join(io.DATA_DIR, 'utf8.csv')
        kwargs = {'delimiter': ','}
        records = io.read_csv(filepath, **kwargs)
        nt.assert_equal(self.row1, next(records))


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
        expected = {
            'sepal_width': '3.5', 'petal_width': '0.2', 'species':
            'Iris-setosa', 'sepal_length': '5.1', 'petal_length': '1.4'}

        filepath = p.join(io.DATA_DIR, 'iris.csv')
        records = io.read_csv(filepath)
        json = cv.records2json(records, newline=True)
        nt.assert_equal(expected, loads(next(json)))

        filepath = p.join(io.DATA_DIR, 'newline.json')
        records = io.read_json(filepath, newline=True)
        nt.assert_equal({'a': 2, 'b': 3}, next(records))

    def test_xls(self):
        filepath = p.join(io.DATA_DIR, 'test.xlsx')
        records = io.read_xls(filepath, sanitize=True, sheet=0)
        nt.assert_equal(self.sheet0, next(records))

        with open(filepath, 'r+b') as f:
            records = io.read_xls(f, sanitize=True, sheet=0)
            nt.assert_equal(self.sheet0, next(records))

        records = io.read_xls(filepath, sanitize=True, sheet=1)
        nt.assert_equal(self.sheet1, next(records))

        kwargs = {'first_row': 1, 'first_col': 1}
        records = io.read_xls(filepath, sanitize=True, sheet=2, **kwargs)
        nt.assert_equal(self.sheet0, next(records))

        records = io.read_xls(filepath, sanitize=True, sheet=3, **kwargs)
        nt.assert_equal(self.sheet1, next(records))

    def test_csv(self):
        filepath = p.join(io.DATA_DIR, 'test.csv')
        header = ['some_date', 'sparse_data', 'some_value', 'unicode_test']

        with open(filepath, 'r', encoding='utf-8') as f:
            records = io._read_csv(f, header)
            nt.assert_equal(self.sheet0_alt, next(records))

        filepath = p.join(io.DATA_DIR, 'no_header_row.csv')
        records = io.read_csv(filepath, has_header=False)
        expected = {'column_1': '1', 'column_2': '2', 'column_3': '3'}
        nt.assert_equal(expected, next(records))

        filepath = p.join(io.DATA_DIR, 'test_bad.csv')
        kwargs = {'sanitize': True, 'first_row': 1, 'first_col': 1}
        records = io.read_csv(filepath, **kwargs)
        nt.assert_equal(self.sheet0_alt, next(records))

        filepath = p.join(io.DATA_DIR, 'fixed_w_header.txt')
        widths = [0, 18, 29, 33, 38, 50]
        records = io.read_fixed_fmt(filepath, widths, has_header=True)
        expected = {
            'News Paper': 'Chicago Reader',
            'Founded': '1971-01-01',
            'Int': '40',
            'Bool': 'True',
            'Float': '1.0',
            'Timestamp': '04:14:001971-01-01T04:14:00'}

        nt.assert_equal(expected, next(records))

    def test_dbf(self):
        filepath = p.join(io.DATA_DIR, 'test.dbf')

        with open(filepath, 'rb') as f:
            records = io.read_dbf(f, sanitize=True)
            expected = {
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

            nt.assert_equal(expected, next(records))

    def test_get_reader(self):
        nt.assert_true(callable(io.get_reader('csv')))

        with nt.assert_raises(KeyError):
            io.get_reader('')


class TestUrlopen:
    """Unit tests for reading files with urlopen"""
    def __init__(self):
        self.cls_initialized = False
        self.utf8_row = {'a': '4', 'b': '5', 'c': 'ʤ'}
        self.latin_row = {'a': '4', 'b': '5', 'c': '©'}

    def test_urlopen_utf8(self):
        filepath = p.join(io.DATA_DIR, 'utf8.csv')

        with closing(urlopen('file://%s' % filepath)) as response:
            f = response.fp
            records = io.read_csv(f)
            row = next(it.islice(records, 1, 2))
            nt.assert_equal(self.utf8_row, row)

    def test_urlopen_latin1(self):
        filepath = p.join(io.DATA_DIR, 'latin1.csv')

        with closing(urlopen('file://%s' % filepath)) as response:
            f = response.fp
            records = io.read_csv(f, encoding='latin-1')
            row = next(it.islice(records, 1, 2))
            nt.assert_equal(self.latin_row, row)


class TestGeoJSON:
    def __init__(self):
        self.cls_initialized = False
        self.bbox = [
            -70.0624999987871, 12.595833309901533, -70.0208333321201,
            12.637499976568533]

        self.filepath = p.join(io.DATA_DIR, 'test.geojson')
        names = ['test', 'line', 'polygon']
        self.filepaths = [p.join(io.DATA_DIR, '%s.geojson' % n) for n in names]

    def test_geojson(self):
        expected = {
            'id': 6635402,
            'iso3': 'ABW',
            'bed_prv_pr': Decimal('0.003'),
            'ic_mhg_cr': Decimal('0.0246'),
            'bed_prv_cr': 0,
            'type': 'Point',
            'lon': Decimal('-70.0624999987871'),
            'lat': Decimal('12.637499976568533')}

        records = io.read_geojson(self.filepath)
        record = next(records)
        nt.assert_equal(expected, record)

        for record in records:
            nt.assert_true('id' in record)
            nt.assert_equal(record['lon'], record['lon'])
            nt.assert_equal(record['lat'], record['lat'])

    def test_geojson_with_id(self):
        for filepath in self.filepaths:
            records = io.read_geojson(filepath)
            f = cv.records2geojson(records, key='id')
            geojson = loads(f.read())

            nt.assert_equal('FeatureCollection', geojson['type'])
            nt.assert_true('crs' in geojson)
            nt.assert_equal(self.bbox, geojson['bbox'])
            nt.assert_true(geojson['features'])

            for feature in geojson['features']:
                nt.assert_equal('Feature', feature['type'])
                nt.assert_true('id' in feature)
                nt.assert_less_equal(2, len(feature['properties']))

                geometry = feature['geometry']

                if geometry['type'] == 'Point':
                    nt.assert_equal(2, len(geometry['coordinates']))
                elif geometry['type'] == 'LineString':
                    nt.assert_equal(2, len(geometry['coordinates'][0]))
                elif geometry['type'] == 'Polygon':
                    nt.assert_equal(2, len(geometry['coordinates'][0][0]))

    def test_geojson_with_crs(self):
        records = io.read_geojson(self.filepath)
        f = cv.records2geojson(records, crs='EPSG:4269')
        geojson = loads(f.read())

        nt.assert_true('crs' in geojson)
        nt.assert_equal('name', geojson['crs']['type'])
        nt.assert_equal('EPSG:4269', geojson['crs']['properties']['name'])


class TestOutput:
    @responses.activate
    def test_write(self):
        url = 'http://google.com'
        body = '<!doctype html><html itemtype="http://schema.org/page">'
        content = StringIO('Iñtërnâtiônàližætiøn')
        nt.assert_equal(20, io.write(StringIO(), content))
        content.seek(0)
        nt.assert_equal(28, io.write(TemporaryFile(), content))

        content = io.IterStringIO(iter('Hello World'))
        nt.assert_equal(12, io.write(TemporaryFile(), content, chunksize=2))

        responses.add(responses.GET, url=url, body=body)
        r = requests.get(url, stream=True)
        nt.assert_equal(55, io.write(TemporaryFile(), r.iter_content))
