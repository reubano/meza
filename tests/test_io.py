# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab
"""
tests.test_io
~~~~~~~~~~~~~

Provides main unit tests.
"""
import itertools as it

from os import path as p
from json import loads
from tempfile import TemporaryFile
from io import StringIO, BytesIO, open
from decimal import Decimal
from urllib.request import urlopen
from contextlib import closing

import requests
import responses
import nose.tools as nt
import pygogo as gogo

from meza import io, convert as cv, DATA_DIR

__INITIALIZED__ = False

logger = gogo.Gogo(__name__, monolog=True).logger  # pylint: disable=C0103


def setup_module():
    """site initialization"""
    global __INITIALIZED__  # pylint: disable=global-statement
    __INITIALIZED__ = True
    print("Site Module Setup\n")


class TestReader:
    def func(filepath):
        reader = lambda f, **kw: (x.strip().split(",") for x in f)
        next(io.read_any(filepath, reader, "rU"))


class TestIterStringIO:
    """Unit tests for IterStringIO"""

    def __init__(self):
        self.phrase = io.IterStringIO(iter("Hello World"))
        self.text = io.IterStringIO("line one\nline two\nline three\n")
        self.ints = io.IterStringIO("0123456789", 5)

    def test_lines(self):
        """Test for reading lines"""
        nt.assert_equal(bytearray(b"Hello"), self.phrase.read(5))
        self.phrase.write(iter("ly person"))
        nt.assert_equal(bytearray(b" Worldly"), self.phrase.read(8))

        self.phrase.write(": Iñtërnâtiônàližætiøn")
        expected = bytearray(" person: Iñtërnâtiônàližætiøn".encode("utf-8"))
        nt.assert_equal(expected, self.phrase.read())

        nt.assert_equal(bytearray(b"line one"), self.text.readline())
        nt.assert_equal(bytearray(b"line two"), next(self.text))

        self.text.seek(0)
        nt.assert_equal(bytearray(b"line one"), next(self.text))
        nt.assert_equal(bytearray(b"line two"), next(self.text))
        nt.assert_equal(16, self.text.tell())

        self.text.seek(0)
        lines = list(self.text.readlines())
        nt.assert_equal(bytearray(b"line three"), lines[2])

    def test_seeking(self):
        """Test for seeking a file"""
        nt.assert_equal(bytearray(b"01234"), self.ints.read(5))
        self.ints.seek(0)
        nt.assert_equal(bytearray(b"0"), self.ints.read(1))
        nt.assert_equal(bytearray(b"1"), self.ints.read(1))
        nt.assert_equal(bytearray(b"2"), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b"3"), self.ints.read(1))

        self.ints.seek(6)
        nt.assert_equal(bytearray(b"6"), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b"3"), self.ints.read(1))

        self.ints.seek(3)
        nt.assert_equal(bytearray(b"3"), self.ints.read(1))

        self.ints.seek(4)
        nt.assert_equal(bytearray(b"4"), self.ints.read(1))

        self.ints.seek(6)
        nt.assert_equal(bytearray(b"6"), self.ints.read(1))

        self.ints.seek(0)
        nt.assert_equal(bytearray(b"2"), self.ints.read(1))


class TestUnicodeReader:
    """Unit tests for unicode support"""

    def __init__(self):
        self.cls_initialized = False
        self.row1 = {"a": "1", "b": "2", "c": "3"}
        self.row2 = {"a": "4", "b": "5", "c": "©"}
        self.row3 = {"a": "4", "b": "5", "c": "ʤ"}
        self.row4 = {"a": "4", "b": "5", "c": "ñ"}

    def test_utf8(self):
        """Test for reading utf-8 files"""
        filepath = p.join(io.DATA_DIR, "utf8.csv")
        records = io.read_csv(filepath, sanitize=True)
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_latin(self):
        """Test for reading latin-1 files"""
        filepath = p.join(io.DATA_DIR, "latin1.csv")
        records = io.read_csv(filepath, encoding="latin-1")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_windows(self):
        """Test for reading windows-1252 files"""
        filepath = p.join(io.DATA_DIR, "windows1252.csv")

        # based on my testing, when excel for mac saves a csv file as
        # 'Windows-1252', you have to open with 'mac-roman' in order
        # to properly read it
        records = io.read_csv(filepath, encoding="mac-roman")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row4, next(records))

    def test_iso(self):
        """Test for reading iso-8859-1 files"""
        filepath = p.join(io.DATA_DIR, "iso88591.csv")
        records = io.read_csv(filepath, encoding="iso-8859-1")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_utf16_big(self):
        """Test for reading utf-16BE files"""
        filepath = p.join(io.DATA_DIR, "utf16_big.csv")
        records = io.read_csv(filepath, encoding="utf-16-be")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_utf16_little(self):
        """Test for reading utf-16LE files"""
        filepath = p.join(io.DATA_DIR, "utf16_little.csv")
        records = io.read_csv(filepath, encoding="utf-16-le")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row3, next(records))

    def test_bytes_encoding_detection_latin(self):
        """Test for detecting the encoding of a latin-1 bytes file"""
        filepath = p.join(io.DATA_DIR, "latin1.csv")
        records = io.read_csv(filepath, mode="rb")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_wrong_encoding_detection_latin(self):
        """Test for detecting the encoding of a latin-1 file opened in ascii"""
        filepath = p.join(io.DATA_DIR, "latin1.csv")
        records = io.read_csv(filepath, encoding="ascii")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row2, next(records))

    def test_bytes_encoding_detection_windows(self):
        """Test for detecting the encoding of a windows-1252 bytes file"""
        filepath = p.join(io.DATA_DIR, "windows1252.csv")
        records = io.read_csv(filepath, mode="rb")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row4, next(records))

    def test_wrong_encoding_detection_windows(self):
        """Test for detecting the encoding of a windows file opened in ascii"""
        filepath = p.join(io.DATA_DIR, "windows1252.csv")
        records = io.read_csv(filepath, encoding="ascii")
        nt.assert_equal(self.row1, next(records))
        nt.assert_equal(self.row4, next(records))

    def test_kwargs(self):
        """Test for passing kwargs while reading csv files"""
        filepath = p.join(io.DATA_DIR, "utf8.csv")
        kwargs = {"delimiter": ","}
        records = io.read_csv(filepath, **kwargs)
        nt.assert_equal(self.row1, next(records))


class TestInput:
    """Unit tests for reading files"""

    def __init__(self):
        self.cls_initialized = False
        self.sheet0 = {
            "sparse_data": "Iñtërnâtiônàližætiøn",
            "some_date": "1982-05-04",
            "some_value": "234.0",
            "unicode_test": "Ādam",
        }

        self.sheet0_alt = {
            "sparse_data": "Iñtërnâtiônàližætiøn",
            "some_date": "05/04/82",
            "some_value": "234",
            "unicode_test": "Ādam",
        }

        self.sheet1 = {
            "boolean": "False",
            "date": "1915-12-31",
            "datetime": "1915-12-31",
            "float": "41800000.01",
            "integer": "164.0",
            "text": "Chicago Tribune",
            "time": "00:00:00",
        }

    def test_newline_json(self):  # pylint: disable=R0201
        """Test for reading newline delimited JSON files"""
        expected = {
            "sepal_width": "3.5",
            "petal_width": "0.2",
            "species": "Iris-setosa",
            "sepal_length": "5.1",
            "petal_length": "1.4",
        }

        filepath = p.join(io.DATA_DIR, "iris.csv")
        records = io.read_csv(filepath)
        json = cv.records2json(records, newline=True)
        nt.assert_equal(expected, loads(next(json)))

        filepath = p.join(io.DATA_DIR, "newline.json")
        records = io.read_json(filepath, newline=True)
        nt.assert_equal({"a": 2, "b": 3}, next(records))

    def test_xls(self):
        """Test for reading excel files"""
        filepath = p.join(io.DATA_DIR, "test.xlsx")
        records = io.read_xls(filepath, sanitize=True, sheet=0)
        nt.assert_equal(self.sheet0, next(records))

        records = io.read_xls(filepath, sanitize=True, sheet=1)
        nt.assert_equal(self.sheet1, next(records))

        kwargs = {"first_row": 1, "first_col": 1}
        records = io.read_xls(filepath, sanitize=True, sheet=2, **kwargs)
        nt.assert_equal(self.sheet0, next(records))

        records = io.read_xls(filepath, sanitize=True, sheet=3, **kwargs)
        nt.assert_equal(self.sheet1, next(records))

    def test_csv(self):
        """Test for reading csv files"""
        filepath = p.join(io.DATA_DIR, "no_header_row.csv")
        records = io.read_csv(filepath, has_header=False)
        expected = {"column_1": "1", "column_2": "2", "column_3": "3"}
        nt.assert_equal(expected, next(records))

        filepath = p.join(io.DATA_DIR, "test_bad.csv")
        kwargs = {"sanitize": True, "first_row": 1, "first_col": 1}
        records = io.read_csv(filepath, **kwargs)
        nt.assert_equal(self.sheet0_alt, next(records))

        filepath = p.join(io.DATA_DIR, "fixed_w_header.txt")
        widths = [0, 18, 29, 33, 38, 50]
        records = io.read_fixed_fmt(filepath, widths, has_header=True)
        expected = {
            "News Paper": "Chicago Reader",
            "Founded": "1971-01-01",
            "Int": "40",
            "Bool": "True",
            "Float": "1.0",
            "Timestamp": "04:14:001971-01-01T04:14:00",
        }

        nt.assert_equal(expected, next(records))

    def test_csv_last_row(self):
        """Test for reading csv files with last_row option"""
        filepath = p.join(io.DATA_DIR, "iris.csv")
        expected = {
            "sepal_width": "3.5",
            "petal_width": "0.2",
            "species": "Iris-setosa",
            "sepal_length": "5.1",
            "petal_length": "1.4",
        }

        records = list(io.read_csv(filepath))
        nt.assert_equal(expected, records[0])
        nt.assert_equal(150, len(records))

        records = list(io.read_csv(filepath, last_row=10))
        nt.assert_equal(expected, records[0])
        nt.assert_equal(10, len(records))

        records = list(io.read_csv(filepath, last_row=-50))
        nt.assert_equal(expected, records[0])
        nt.assert_equal(100, len(records))

    def test_dbf(self):  # pylint: disable=R0201
        """Test for reading dbf files"""
        filepath = p.join(io.DATA_DIR, "test.dbf")

        with open(filepath, "rb") as f:
            records = io.read_dbf(f, sanitize=True)
            expected = {
                "awater10": 12416573076,
                "aland10": 71546663636,
                "intptlat10": "+47.2400052",
                "lsad10": "C2",
                "cd111fp": "08",
                "namelsad10": "Congressional District 8",
                "funcstat10": "N",
                "statefp10": "27",
                "cdsessn": "111",
                "mtfcc10": "G5200",
                "geoid10": "2708",
                "intptlon10": "-092.9323194",
            }

            nt.assert_equal(expected, next(records))

    def test_vertical_table(self):  # pylint: disable=R0201
        """Test for reading a vertical html table"""
        filepath = p.join(io.DATA_DIR, "vertical_table.html")
        records = io.read_html(filepath, vertical=True)
        nt.assert_equal("See IBM products", next(records)["Products"])
        records = io.read_html(filepath, vertical=True, table=2)

        with nt.assert_raises(StopIteration):
            next(records)

    def test_excel_html_export(self):  # pylint: disable=R0201
        """Test for reading an html table exported from excel"""
        filepath = p.join(io.DATA_DIR, "test.htm")
        records = io.read_html(filepath, sanitize=True, first_row_as_header=True)

        expected = {
            "sparse_data": "Iñtërnâtiônàližætiøn",
            "some_date": "05/04/82",
            "some_value": "234",
            "unicode_test": "Ādam",
        }

        nt.assert_equal(expected, next(records))

    def test_get_reader(self):  # pylint: disable=R0201
        """Test for reading a file via the reader selector"""
        nt.assert_true(callable(io.get_reader("csv")))

        with nt.assert_raises(KeyError):
            io.get_reader("")

    def test_opened_files(self):
        """Test for reading open files"""
        filepath = p.join(io.DATA_DIR, "test.csv")

        with open(filepath, encoding="utf-8") as f:
            records = io.read_csv(f, sanitize=True)  # pylint: disable=W0212
            nt.assert_equal(self.sheet0_alt, next(records))

        f = open(filepath, encoding="utf-8")

        try:
            records = io.read_csv(f, sanitize=True)
            nt.assert_equal(self.sheet0_alt, next(records))
        finally:
            f.close()

        f = open(filepath, "rU", newline=None)

        try:
            records = io.read_csv(f, sanitize=True)
            nt.assert_equal(self.sheet0_alt, next(records))
        finally:
            f.close()

        filepath = p.join(io.DATA_DIR, "test.xlsx")

        with open(filepath, "r+b") as f:
            records = io.read_xls(f, sanitize=True, sheet=0)
            nt.assert_equal(self.sheet0, next(records))

        f = open(filepath, "r+b")

        try:
            records = io.read_xls(f, sanitize=True, sheet=0)
            nt.assert_equal(self.sheet0, next(records))
        finally:
            f.close()

    def test_reencode(self):
        file_ = p.join(io.DATA_DIR, "utf16_big.csv")

        with open(file_, encoding="utf-16-be") as f:
            utf8_f = io.reencode(f, remove_BOM=True)
            nt.assert_equal(b"a,b,c", next(utf8_f).strip())
            nt.assert_equal(b"1,2,3", next(utf8_f).strip())
            nt.assert_equal("4,5,ʤ", next(utf8_f).decode("utf-8"))


class TestUrlopen:
    """Unit tests for reading files with urlopen"""

    def __init__(self):
        self.cls_initialized = False
        self.utf8_row = {"a": "4", "b": "5", "c": "ʤ"}
        self.latin_row = {"a": "4", "b": "5", "c": "©"}

    def test_urlopen_utf8(self):
        """Test for reading utf-8 files"""
        filepath = p.join(io.DATA_DIR, "utf8.csv")

        with closing(urlopen("file://{}".format(filepath))) as response:
            f = response.fp
            records = io.read_csv(f)
            row = next(it.islice(records, 1, 2))
            nt.assert_equal(self.utf8_row, row)

    def test_urlopen_latin1(self):
        """Test for reading latin-1 files"""
        filepath = p.join(io.DATA_DIR, "latin1.csv")

        with closing(urlopen("file://{}".format(filepath))) as response:
            f = response.fp
            records = io.read_csv(f, encoding="latin-1")
            row = next(it.islice(records, 1, 2))
            nt.assert_equal(self.latin_row, row)

    # def test_urlopen_remote(self):
    #     """Test for reading remote web files"""
    #     filepath = 'https://opendata.co.ke...'
    #     response = urlopen('file://{}'.format(filepath))
    #     records = io.read_csv(response.fp)
    #     nt.assert_equal({}, next(records))


class TestBytes:
    """Unit tests for reading byte streams"""

    def __init__(self):
        self.cls_initialized = False

        self.row1 = {"a": "1", "b": "2", "c": "3"}
        self.row2 = {"a": "4", "b": "5", "c": "ʤ"}

        self.sheet0_alt = {
            "sparse_data": "Iñtërnâtiônàližætiøn",
            "some_date": "05/04/82",
            "some_value": "234",
            "unicode_test": "Ādam",
        }

    def test_bytes_io(self):
        """Test for reading BytesIO"""
        with open(p.join(io.DATA_DIR, "utf8.csv"), "rb") as f:
            b = BytesIO(f.read())
            records = io.read_csv(b, sanitize=True)
            nt.assert_equal(self.row1, next(records))
            nt.assert_equal(self.row2, next(records))

    def test_bytes(self):
        """Test for reading bytes mode opened file"""
        with open(p.join(io.DATA_DIR, "test.csv"), "rb") as f:
            records = io.read_csv(f, sanitize=True)
            nt.assert_equal(self.sheet0_alt, next(records))


class TestGeoJSON:
    """Unit tests for reading GeoJSON"""

    def __init__(self):
        self.cls_initialized = False
        self.bbox = [
            -70.0624999987871,
            12.595833309901533,
            -70.0208333321201,
            12.637499976568533,
        ]

        self.filepath = p.join(io.DATA_DIR, "test.geojson")
        names = ["test", "line", "polygon"]
        fname = "{}.geojson"
        self.filepaths = [p.join(io.DATA_DIR, fname.format(n)) for n in names]

    def test_geojson(self):
        """Test for reading GeoJSON files"""
        expected = {
            "id": 6635402,
            "iso3": "ABW",
            "bed_prv_pr": Decimal("0.003"),
            "ic_mhg_cr": Decimal("0.0246"),
            "bed_prv_cr": 0,
            "type": "Point",
            "lon": Decimal("-70.0624999987871"),
            "lat": Decimal("12.637499976568533"),
        }

        records = io.read_geojson(self.filepath)
        record = next(records)
        nt.assert_equal(expected, record)

        for record in records:
            nt.assert_true("id" in record)
            nt.assert_equal(record["lon"], record["lon"])
            nt.assert_equal(record["lat"], record["lat"])

    def test_geojson_with_key(self):
        """Test for reading GeoJSON files with a key"""
        for filepath in self.filepaths:
            records = io.read_geojson(filepath)
            f = cv.records2geojson(records, key="id")
            geojson = loads(f.read())

            nt.assert_equal("FeatureCollection", geojson["type"])
            nt.assert_true("crs" in geojson)
            nt.assert_equal(self.bbox, geojson["bbox"])
            nt.assert_true(geojson["features"])

            for feature in geojson["features"]:
                nt.assert_equal("Feature", feature["type"])
                nt.assert_true("id" in feature)
                nt.assert_less_equal(2, len(feature["properties"]))

                geometry = feature["geometry"]

                if geometry["type"] == "Point":
                    nt.assert_equal(2, len(geometry["coordinates"]))
                elif geometry["type"] == "LineString":
                    nt.assert_equal(2, len(geometry["coordinates"][0]))
                elif geometry["type"] == "Polygon":
                    nt.assert_equal(2, len(geometry["coordinates"][0][0]))

    def test_geojson_with_crs(self):
        """Test for reading GeoJSON files with CRS"""
        records = io.read_geojson(self.filepath)
        f = cv.records2geojson(records, crs="EPSG:4269")
        geojson = loads(f.read())

        nt.assert_true("crs" in geojson)
        nt.assert_equal("name", geojson["crs"]["type"])
        nt.assert_equal("EPSG:4269", geojson["crs"]["properties"]["name"])


class TestOutput:
    """Unit tests for writing files"""

    @responses.activate  # pylint: disable=E1101
    def test_write(self):  # pylint: disable=R0201
        """Test for writing to a file"""
        url = "http://google.com"
        body = '<!doctype html><html itemtype="http://schema.org/page">'
        content1 = StringIO("Iñtërnâtiônàližætiøn")
        nt.assert_equal(20, io.write(StringIO(), content1))
        content1.seek(0)
        nt.assert_equal(20, io.write(TemporaryFile(), content1))

        content2 = io.IterStringIO(iter("Hello World"))
        nt.assert_equal(12, io.write(TemporaryFile(), content2, chunksize=2))

        # pylint: disable=E1101
        responses.add(responses.GET, url=url, body=body)
        r = requests.get(url, stream=True)  # pylint: disable=C0103
        nt.assert_equal(55, io.write(TemporaryFile(), r.iter_content))
