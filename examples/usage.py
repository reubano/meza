# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
README examples
~~~~~~~~~~~~~~~

Setup

    >>> import itertools as it
    >>> try:
    ...     import pandas as pd
    ... except ImportError:
    ...     np = pd = None
    ... else:
    ...     import numpy as np

    >>> from io import StringIO
    >>> from array import array
    >>> from json import loads
    >>> from datetime import date
    >>> from meza import (
    ...     io, process as pr, convert as cv, fntools as ft, stats)


Loading, type casting, and writing to a CSV file

    # Create file
    >>> text = (
    ...     'col1,col2,col3\\n'
    ...     'hello,5/4/82,1\\n'
    ...     'one,1/1/15,2\\n'
    ...     'happy,7/4/92,3\\n')
    >>> f = StringIO(text)
    >>> bool(f.seek(0))
    False

    # Load file
    >>> records = io.read_csv(f)

    # Records are an iterator over the rows
    >>> row = next(records)
    >>> row == {'col1': 'hello', 'col2': '5/4/82', 'col3': '1'}
    True

    # Replace first row so as not to loose any data
    >>> records = pr.prepend(records, row)

    # guess column types
    >>> records, result = pr.detect_types(records)
    >>> types = result['types']
    >>> float(result['confidence'])
    0.89

    >>> {t['id']: t['type'] for t in types} == {
    ...     'col1': 'text',
    ...     'col2': 'date',
    ...     'col3': 'int'}
    True

    # apply these types to the records
    >>> casted = list(pr.type_cast(records, types))
    >>> casted[0] == {
    ...     'col1': 'hello', 'col2': date(1982, 5, 4), 'col3': 1}
    True

    # now run some operation on the type casted data
    >>> cut_recs = pr.cut(casted, ['col1'], exclude=True)
    >>> merged = pr.merge(cut_recs, pred=bool, op=max)
    >>> merged == {'col2': date(2015, 1, 1), 'col3': 3}
    True

    # Write back to a csv file
    >>> f = StringIO()
    >>> bool(io.write(f, cv.records2csv([merged])))
    True
    >>> bool(f.seek(0))
    False
    >>> set(f.readline().rstrip().split(',')) == {'col2', 'col3'}
    True
    >>> set(f.readline().rstrip().split(',')) == {'2015-01-01', '3'}
    True


Reading data

    # Read a file like object and de-duplicate the header
    >>> f = StringIO('col,col\\nhello,world\\n')
    >>> next(io.read_csv(f, dedupe=True)) == {
    ...     'col': 'hello', 'col_2': 'world'}
    True


Numerical analysis

    # Create a `records` compatible df
    >>> header = ['A', 'B', 'C', 'D']
    >>> data = [
    ...     [0.5607, 0.9338, 0.4769, 0.7804],
    ...     [0.8227, 0.2844, 0.8166, 0.7323],
    ...     [0.4627, 0.8633, 0.3283, 0.1909],
    ...     [0.3932, 0.5454, 0.9604, 0.6376],
    ...     [0.3685, 0.9166, 0.9457, 0.8066],
    ...     [0.7584, 0.6981, 0.5625, 0.3578],
    ...     [0.8959, 0.6932, 0.2565, 0.3378]]
    >>> df = [dict(zip(header, d)) for d in data]

    >>> df[0] ==  {
    ...     'A': 0.5607,
    ...     'B': 0.9338,
    ...     'C': 0.4769,
    ...     'D': 0.7804}
    True

    >>> next(pr.sort(df, 'B')) == {
    ...     'A': 0.8227, 'B': 0.2844, 'C': 0.8166, 'D': 0.7323}
    True
    >>> next(pr.cut(df, ['A'])) == {'A': 0.5607}
    True
    >>> len(list(it.islice(df, 3)))
    3

    # Use a single column’s values to select data
    >>> next(pr.tfilter(df, 'A', lambda x: x < 0.5)) == {
    ...     'A': 0.4627, 'B': 0.8633, 'C': 0.3283, 'D': 0.1909}
    True

    # Aggregation
    >>> round(pr.aggregate(df, 'A', stats.mean)['A'], 4)
    0.6089

    >>> pr.merge(df, pred=bool, op=sum) == {
    ...     'A': 4.2621, 'B': 4.9348, 'C': 4.3469, 'D': 3.8434}
    True


Text processing

    # First create a few simple csv files
    >>> f1 = StringIO(
    ...     'col_1,col_2,col_3\\n1,dill,male\\n'
    ...     '2,bob,male\\n3,jane,female\\n')
    >>> f2 = StringIO(
    ...     'col_1,col_2,col_3\\n4,tom,male\\n'
    ...     '5,dick,male\\n6,jill,female\\n')
    >>> bool(f1.seek(0))
    False
    >>> bool(f2.seek(0))
    False

    # Let's join the files together
    >>> records = io.join(f1, f2, ext='csv')
    >>> next(records) == {'col_1': '1', 'col_2': 'dill', 'col_3': 'male'}
    True
    >>> next(it.islice(records, 4, None)) == {
    ...     'col_1': '6', 'col_2': 'jill', 'col_3': 'female'}
    True

    # Reset the first file and then do some simple manipulations
    >>> bool(f1.seek(0))
    False
    >>> records = list(io.read_csv(f1))

    >>> next(pr.sort(records, 'col_2')) == {
    ...     'col_1': '2', 'col_2': 'bob', 'col_3': 'male'}
    True

    >>> next(pr.cut(records, ['col_2'])) == {'col_2': 'dill'}
    True

    >>> next(pr.grep(records, [{'pattern': 'jan'}], ['col_2'])) == {
    ...     'col_1': '3', 'col_2': 'jane', 'col_3': 'female'}
    True

    >>> f_json = StringIO()
    >>> bool(io.write(f_json, cv.records2json(records)))
    True
    >>> loads(f_json.getvalue()) == records
    True


GeoJSON

    # First create a geojson fi
    >>> f = StringIO(
    ...     '{"type": "FeatureCollection","features": ['
    ...     '{"type": "Feature", "id": 11, "geometry": '
    ...     '{"type": "Point", "coordinates": [10, 20]}},'
    ...     '{"type": "Feature", "id": 12, "geometry": '
    ...     '{"type": "Point", "coordinates": [5, 15]}}]}')

    >>> bool(f.seek(0))
    False

    # Load the geojson file
    >>> records, peek = pr.peek(io.read_geojson(f))
    >>> peek[0] == {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}
    True

    # Split the records by feature ``id``
    >>> splits = pr.split(records, 'id')
    >>> feature_records, name = next(splits)
    >>> name
    11

    # Convert the feature records into a GeoJSON file-like object
    >>> geojson = cv.records2geojson(feature_records)
    >>> geojson.readline() == (
    ...     '{"type": "FeatureCollection", "bbox": [10, 20, 10, 20], '
    ...     '"features": [{"type": "Feature", "id": 11, "geometry": '
    ...     '{"type": "Point", "coordinates": [10, 20]}, "properties": '
    ...     '{"id": 11}}], "crs": {"type": "name", "properties": {"name": '
    ...     '"urn:ogc:def:crs:OGC:1.3:CRS84"}}}')
    True


Writing Data

    # First let's create a simple tsv file like object
    >>> f = StringIO('col1\\tcol2\\nhello\\tworld\\n')
    >>> bool(f.seek(0))
    False

    # Next create a records list so we can reuse it
    >>> records = list(io.read_tsv(f))
    >>> records[0] == {'col1': 'hello', 'col2': 'world'}
    True

    # Create a csv file like object
    >>> f_out = cv.records2csv(records)
    >>> set(f_out.readline().rstrip().split(',')) == {'col1', 'col2'}
    True

    # Create a json file like object
    >>> f_out = cv.records2json(records)
    >>> loads(f_out.readline()) == [{'col1': 'hello', 'col2': 'world'}]
    True

    # Write back csv to a filepath
    >>> bool(io.write('file.csv', cv.records2csv(records)))
    True
    >>> with open('file.csv', encoding='utf-8') as f_in:
    ...     set(f_in.readline().rstrip().split(',')) == {'col1', 'col2'}
    True

    # Write back json to a filepath
    >>> bool(io.write('file.json', cv.records2json(records)))
    True
    >>> with open('file.json', encoding='utf-8') as f_in:
    ...     loads(f_in.readline()) == [{'col1': 'hello', 'col2': 'world'}]
    True


Interoperability

    >>> records = [{'a': 'one', 'b': 2}, {'a': 'five', 'b': 10, 'c': 20.1}]
    >>> records, result = pr.detect_types(records)
    >>> records, types = list(records), result['types']
    >>> {(t['id'], t['type']) for t in types} == set(
    ...     [('a', 'text'), ('b', 'int'), ('c', 'float')])
    True

    # Convert records to a DataFrame
    >>> df = cv.records2df(records, types)
    >>> columns = df.columns if pd else ft.get_values(df[0])
    >>> set(columns) == {'a', 'b', 'c'}
    True
    >>> col_a = df.a.tolist() if pd else list(ft.get_values(df[1]))
    >>> col_a == ['one', 'five']
    True
    >>> cols = df[['b', 'c']] if pd else df[2:]
    >>> rest = cols.values.flatten() if pd else ft.get_values(cols)
    >>> sorted(map(np.isfinite if pd else bool, rest))
    [False, True, True, True]

    # Convert a DataFrame to records
    >>> converted = cv.df2records(df) if pd else cv.array2records(df, True)
    >>> row = next(converted)
    >>> row['a'] == 'one'
    True
    >>> row['b']
    2
    >>> np.isnan(row['c']) if np else row['c'] == 0
    True

    # Convert records to a structured array
    >>> recarray = cv.records2array(records, types)
    >>> if pd:
    ...     values = set(it.chain(*zip(*recarray)))
    ... else:
    ...     values = set(ft.get_values(recarray[1:]))
    >>> values.issuperset({'one', 'five'})
    True
    >>> rest = (v for v in values if v not in {'one', 'five'})
    >>> sorted(map(np.isfinite if pd else bool, rest))
    [False, True, True, True]

    # let's set the native array flag
    >>> native = not np

    # Convert a 2-D array to records
    >>> arr = [[1, 2, 3], [4, 5, 6]] if np else [(1, 4), (2, 5), (3, 6)]
    >>> data = np.array(arr, 'i4') if np else [array('i', a) for a in arr]
    >>> next(cv.array2records(data, native)) == {
    ...     'column_1': 1, 'column_2': 2, 'column_3': 3}
    True

    # Convert a structured array to records
    >>> row = next(cv.array2records(recarray, native))
    >>> row['a'] == 'one'
    True
    >>> row['b']
    2
    >>> np.isnan(row['c']) if np else row['c'] == 0
    True

    # Convert records to a native array
    >>> narray = cv.records2array(records, result['types'], True)
    >>> set(ft.get_values(narray)) == {
    ...     'a', 'b', 'c', 'one', 'five', 0.0, 20.100000381469727, 2, 10}
    True

    # Convert native array to records
    >>> next(cv.array2records(narray, True)) == {
    ...     'a': 'one', 'b': 2, 'c': 0.0}
    True

Cookbook

    >>> header = ['A', 'B', 'C', 'D']

    # Create some data in the same structure as what the various `read...`
    # functions output
    >>> data = [
    ...     [0.5607, 0.9338, 0.4769, 0.7804],
    ...     [0.8227, 0.2844, 0.8166, 0.7323],
    ...     [0.4627, 0.8633, 0.3283, 0.1909],
    ...     [0.3932, 0.5454, 0.9604, 0.6376],
    ...     [0.3685, 0.9166, 0.9457, 0.8066],
    ...     [0.7584, 0.6981, 0.5625, 0.3578],
    ...     [0.8959, 0.6932, 0.2565, 0.3378]]
    >>> records = [dict(zip(header, d)) for d in data]

    # Select multiple columns
    >>> next(pr.cut(records, ['A', 'B'], exclude=True)) == {
    ...     'C': 0.4769, 'D': 0.7804}
    True

    # Concatenate records together
    >>> pieces = [it.islice(records, 3), it.islice(records, 3, None)]
    >>> concated = it.chain(*pieces)
    >>> next(concated) == {
    ...     'A': 0.5607,
    ...     'B': 0.9338,
    ...     'C': 0.4769,
    ...     'D': 0.7804}
    True
    >>> len(list(concated)) + 1
    7

    # Make SQL style joins
    >>> left = [{'key': 'foo', 'lval': 1}, {'key': 'foo', 'lval': 2}]
    >>> right = [{'key': 'foo', 'rval': 4}, {'key': 'foo', 'rval': 5}]
    >>> list(pr.join(left, right)) == [
    ...     {'key': 'foo', 'lval': 1, 'rval': 4},
    ...     {'key': 'foo', 'lval': 1, 'rval': 5},
    ...     {'key': 'foo', 'lval': 2, 'rval': 4},
    ...     {'key': 'foo', 'lval': 2, 'rval': 5}]
    True

    # Group and sum
    >>> records = [
    ...     {'A': 'foo', 'B': -1.202872},
    ...     {'A': 'bar', 'B': 1.814470},
    ...     {'A': 'foo', 'B': 1.8028870},
    ...     {'A': 'bar', 'B': -0.595447}]
    >>> kwargs = {'aggregator': pr.merge, 'pred': 'B', 'op': sum}
    >>> list(pr.group(records, 'A', tupled=False, **kwargs)) == [
    ...     {'A': 'bar', 'B': 1.219023}, {'A': 'foo', 'B': 0.600015}]
    True

    # Pivot tables
    >>> a = ['one', 'one', 'two', 'three'] * 3
    >>> b = ['ah', 'beh', 'say'] * 4
    >>> c = ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2
    >>> d = [
    ...     -0.5616, 2.2791, -3.9950, -0.6289, 4.6962, 0.9220,
    ...     -3.8169, -6.0872, -1.8378, 3.3339, 0.7682, 1.3109]
    >>> values = zip(a, b, c, d)
    >>> records = (dict(zip(header, v)) for v in values)
    >>> records, peek = pr.peek(records)
    >>> set(int(p['D'] or 0) for p in peek).issubset({0, 2, 4, -3})
    True
    >>> peek == [
    ...     {'A': 'one', 'B': 'ah', 'C': 'foo', 'D': -0.5616},
    ...     {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': 2.2791},
    ...     {'A': 'two', 'B': 'say', 'C': 'foo', 'D': -3.995},
    ...     {'A': 'three', 'B': 'ah', 'C': 'bar', 'D': -0.6289},
    ...     {'A': 'one', 'B': 'beh', 'C': 'bar', 'D': 4.6962}]
    True

    >>> pivot = pr.pivot(records, 'D', 'C')
    >>> pivot, peek = pr.peek(pivot)
    >>> set(int(p.get('bar', 0)) for p in peek).issubset({0, 3, 4})
    True
    >>> set(int(p.get('foo', 0)) for p in peek).issubset({0, 2, -6, -3, -1})
    True

    # Data normalization
    >>> normal = pr.normalize(pivot, 'D', 'C', ['foo', 'bar'])
    >>> normal, peek = pr.peek(normal)
    >>> set(int(p['D'] or 0) for p in peek).issubset({0, 2, 3, 4, -3, -1})
    True

    # More fun with geojson
    >>> f1 = StringIO(
    ...     '{"type": "FeatureCollection","features": [{"type": "Feature", '
    ...     '"id": 11, "geometry": {"type": "Point", "coordinates": '
    ...     '[10, 20]}}]}')
    >>> f2 = StringIO(
    ...     '{"type": "FeatureCollection","features": [{"type": "Feature", '
    ...     '"id": 12, "geometry": {"type": "Point", "coordinates": '
    ...     '[5, 15]}}]}')
    >>> bool(f1.seek(0))
    False
    >>> bool(f2.seek(0))
    False

    # Combine the GeoJSON files into one iterator
    >>> records, peek = pr.peek(io.join(f1, f2, ext='geojson'))
    >>> peek[0] == {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}
    True

    >>> records, result = pr.detect_types(records)
    >>> casted_records = pr.type_cast(records, result['types'])
    >>> cv.records2geojson(casted_records).read() == (
    ...     '{"type": "FeatureCollection", "bbox": [5, 15, 10, 20], '
    ...     '"features": [{"type": "Feature", "id": 11, "geometry": '
    ...     '{"type": "Point", "coordinates": [10, 20]}, "properties": '
    ...     '{"id": 11}}, {"type": "Feature", "id": 12, "geometry": '
    ...     '{"type": "Point", "coordinates": [5, 15]}, "properties": '
    ...     '{"id": 12}}], "crs": {"type": "name", "properties": {"name": '
    ...     '"urn:ogc:def:crs:OGC:1.3:CRS84"}}}')
    True
"""

from __future__ import (
    absolute_import, division, print_function, unicode_literals)

from builtins import *
