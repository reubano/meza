# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
examples
~~~~~~~~

README examples

Examples:

    Setup

        >>> import itertools as it
        >>> import random
        >>> import numpy as np
        >>> import pandas as pd

        >>> from io import StringIO
        >>> from json import loads
        >>> from functools import partial
        >>> from operator import itemgetter, eq, lt, gt
        >>> from tabutils import io, process as pr, convert as cv, stats
        >>> from datetime import date
        >>> random.seed(30)


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
        >>> casted = pr.type_cast(records, types)

        # now run some operation on the type casted data
        >>> cut_recs = pr.cut(casted, ['col1'], exclude=True)
        >>> merged = pr.merge(cut_recs, pred=bool, op=max)
        >>> merged == {'col2': date(2015, 1, 1), 'col3': 3}
        True

        # Write back to a csv file
        >>> f = StringIO()
        >>> bool(io.write(f, cv.records2csv([merged])))
        True
        >>> f.getvalue() == 'col2,col3\\r\\n2015-01-01,3\\r\\n'
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
        >>> data = [(random.random() for _ in range(4)) for x in range(7)]
        >>> df = [dict(zip(header, d)) for d in data]

        >>> df[0] ==  {
        ...     'A': 0.5390815646058106,
        ...     'B': 0.2891964436397205,
        ...     'C': 0.03003690855112706,
        ...     'D': 0.6536357538927619}
        True

        >>> sorted(df, key=itemgetter('B'))[0] == {
        ...     'A': 0.535204782203361,
        ...     'B': 0.06763103158333483,
        ...     'C': 0.023510063056781383,
        ...     'D': 0.8052942869277137}
        True
        >>> next(pr.cut(df, ['A'])) == {'A': 0.5390815646058106}
        True
        >>> len(list(it.islice(df, 3)))
        3

        # Use a single column’s values to select data (df[df.A < 0.5])
        >>> rules = [{'fields': ['A'], 'pattern': partial(gt, 0.5)}]
        >>> next(pr.grep(df, rules))['A']
        0.21000869554973112

        # Aggregation
        >>> pr.aggregate(df, 'A', stats.mean)['A']
        0.5410437473067938

        >>> pr.merge(df, pred=bool, op=sum) == {
        ...     'A': 3.787306231147557,
        ...     'B': 2.828756979845426,
        ...     'C': 3.141952839530555,
        ...     'D': 5.263300500059669}
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

        # Join multiple files together
        >>> records = io.join(f1, f2, ext='csv')
        >>> next(records) == {'col_1': '1', 'col_2': 'dill', 'col_3': 'male'}
        True
        >>> next(it.islice(records, 4, None)) == {
        ...     'col_1': '6', 'col_2': 'jill', 'col_3': 'female'}
        True

        >>> bool(f1.seek(0))
        False
        >>> records = list(io.read_csv(f1))

        >>> sorted(records, key=itemgetter('col_2'))[0] == {
        ...     'col_1': '2', 'col_2': 'bob', 'col_3': 'male'}
        True

        >>> next(pr.cut(records, ['col_2'])) == {'col_2': 'dill'}
        True

        >>> rules = [{'fields': ['col_2'], 'pattern': 'jane'}]
        >>> next(pr.grep(records, rules)) == {
        ...     'col_1': '3', 'col_2': 'jane', 'col_3': 'female'}
        True

        >>> f_json = StringIO()
        >>> bool(io.write(f_json, cv.records2json(records)))
        True
        >>> loads(f_json.getvalue()) == records
        True


    GeoJSON

        # First create a few simple csv files
        >>> f1 = StringIO('id,lon,lat,type\\n11,10,20,Point\\n12,5,15,Point\\n')
        >>> f2 = StringIO('id,lon,lat,type\\n13,15,20,Point\\n14,5,25,Point\\n')
        >>> bool(f1.seek(0))
        False
        >>> bool(f2.seek(0))
        False

        # Convert files to GeoJSON
        >>> geofiles = []
        >>> for f in [f1, f2]:
        ...     records = io.read_csv(f)
        ...     records, result = pr.detect_types(records)
        ...     casted_records = pr.type_cast(records, result['types'])
        ...     geofiles.append(cv.records2geojson(casted_records))
        >>> loads(geofiles[0].readline()) == {
        ...     'type': 'FeatureCollection',
        ...     'bbox': [5, 15, 10, 20],
        ...     'features': [
        ...         {
        ...             'type': 'Feature',
        ...             'id': 11,
        ...             'geometry': {'type': 'Point', 'coordinates': [10, 20]},
        ...             'properties': {'id': 11}},
        ...         {
        ...             'type': 'Feature',
        ...             'id': 12,
        ...             'geometry': {'type': 'Point', 'coordinates': [5, 15]},
        ...             'properties': {'id': 12}}],
        ...     'crs': {
        ...         'type': 'name',
        ...         'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}}
        True

        # Merge multiple GeoJSON files into one
        >>> bool(geofiles[0].seek(0))
        False
        >>> records = io.join(*geofiles, ext='geojson')
        >>> next(records) == {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}
        True

        # Split a GeoJSON file by feature
        >>> bool(geofiles[0].seek(0))
        False
        >>> records = io.read_geojson(geofiles[0])
        >>> records, result = pr.detect_types(records)
        >>> casted_records = pr.type_cast(records, result['types'])
        >>> splits = pr.split(casted_records, 'id')
        >>> sub_records = [s[0] for s in splits]
        >>> geojson = map(cv.records2geojson, sub_records)
        >>> loads(next(geojson).readline()) == {
        ...     'type': 'FeatureCollection',
        ...     'bbox': [10, 20, 10, 20],
        ...     'features': [
        ...         {
        ...             'type': 'Feature',
        ...             'id': 11,
        ...             'geometry': {'type': 'Point', 'coordinates': [10, 20]},
        ...             'properties': {'id': 11}}],
        ...     'crs': {
        ...         'type': 'name',
        ...         'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}}
        True
        >>> loads(next(geojson).readline()) == {
        ...     'type': 'FeatureCollection',
        ...     'bbox': [5, 15, 5, 15],
        ...     'features': [
        ...         {
        ...             'type': 'Feature',
        ...             'id': 12,
        ...             'geometry': {'type': 'Point', 'coordinates': [5, 15]},
        ...             'properties': {'id': 12}}],
        ...     'crs': {
        ...         'type': 'name',
        ...         'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}}
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

        # Now we're ready to write the records data to file

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

        # Convert records to a DataFrame
        >>> df = pd.DataFrame(records)
        >>> df
           a   b   c
        0  1   2 NaN
        1  5  10  20

        # Convert a DataFrame to a records generator
        >>> conv_records = cv.df2records(df)
        >>> result = {k: np.isnan(v) for k, v in next(conv_records).items()}
        >>> result == {'a': False, 'b': False, 'c': True}
        True

        # Convert records to a structured array
        >>> records, result = pr.detect_types(records)
        >>> recarray = cv.records2array(records, result['types'])
        >>> recarray.a.tolist() == ['one', 'five']
        True
        >>> recarray.b
        array([ 2, 10], dtype=int32)
        >>> clist = recarray.c.tolist()
        >>> np.isnan(clist[0])
        True
        >>> clist[1]
        20.100000381469727

        # First create a 2-D NumPy array
        >>> data = np.array([[1, 2, 3], [4, 5, 6]], 'i4')
        >>> data
        array([[1, 2, 3],
               [4, 5, 6]], dtype=int32)

        # Convert a 2-D array to a records generator
        >>> next(cv.array2records(data)) == {
        ...     'column_1': 1, 'column_2': 2, 'column_3': 3}
        True

        # Now create a structured array
        >>> types = [('A', 'i4'), ('B', 'f4'), ('C', 'S5')]
        >>> dtype = [(k.encode('ascii'), v.encode('ascii')) for k, v in types]
        >>> data = [(1, 2., 'Hello'), (2, 3., 'World')]
        >>> ndarray = np.array(data, dtype=dtype)
        >>> ndarray.tolist() == [(1, 2.0, 'Hello'), (2, 3.0, 'World')]
        True

        # Convert a structured array to a records generator
        >>> next(cv.array2records(ndarray)) == {'A': 1, 'B': 2.0, 'C': 'Hello'}
        True


    Cookbook

        >>> header = ['A', 'B', 'C', 'D']

        # Create some data in the same structure as what the various `read...`
        # functions output
        >>> data = [(random.random() for _ in range(4)) for x in range(7)]
        >>> records = [dict(zip(header, d)) for d in data]

        # Select multiple columns
        >>> next(pr.cut(records, ['A', 'B'], exclude=True)) == {
        ...     'C': 0.11175001869696033, 'D': 0.4944504196475903}
        True

        # Concatenate records together
        >>> pieces = [it.islice(records, 3), it.islice(records, 3, None)]
        >>> concated = it.chain(*pieces)
        >>> next(concated) == {
        ...     'A': 0.6387228188088844,
        ...     'B': 0.8951756504920998,
        ...     'C': 0.11175001869696033,
        ...     'D': 0.4944504196475903}
        True
        >>> len(list(concated)) + 1
        7

        # SQL style joins (pd.merge(left, right, on='key'))
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

        # Create pivot data
        >>> rrange = random.sample(range(-10, 10), 12)
        >>> a = ['one', 'one', 'two', 'three'] * 3
        >>> b = ['ah', 'beh', 'say'] * 4
        >>> c = ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2
        >>> d = (random.random() * x for x in rrange)
        >>> values = zip(a, b, c, d)
        >>> records = (dict(zip(header, v)) for v in values)
        >>> records, peek = pr.peek(records)
        >>> peek == [
        ...     {'A': 'one', 'B': 'ah', 'C': 'foo', 'D': -3.6982230400621234},
        ...     {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': -3.720184399162731},
        ...     {'A': 'two', 'B': 'say', 'C': 'foo', 'D': 1.0214689218724586},
        ...     {'A': 'three', 'B': 'ah', 'C': 'bar', 'D': 0.38015862302086945},
        ...     {'A': 'one', 'B': 'beh', 'C': 'bar', 'D': 0.0}]
        True

        # Now lets pivot the data
        >>> pivot = pr.pivot(records, 'D', 'C')
        >>> pivot, peek = pr.peek(pivot)
        >>> peek == [
        ... {
        ...     'A': 'one', 'B': 'ah', 'bar': 2.2393327345103637,
        ...     'foo': -3.6982230400621234},
        ... {'A': 'one', 'B': 'beh', 'bar': 0.0, 'foo': -3.720184399162731},
        ... {
        ...     'A': 'one', 'B': 'say', 'bar': 2.6759543278059583,
        ...     'foo': -5.557746676883692},
        ... {'A': 'three', 'B': 'ah', 'bar': 0.38015862302086945},
        ... {'A': 'three', 'B': 'beh', 'foo': 5.794308531883553}]
        True

        # Data normalization
        >>> normal = pr.normalize(pivot, 'D', 'C', ['foo', 'bar'])
        >>> pr.peek(normal)[1] == [
        ...     {'A': 'one', 'B': 'ah', 'C': 'foo', 'D': -3.6982230400621234},
        ...     {'A': 'one', 'B': 'ah', 'C': 'bar', 'D': 2.2393327345103637},
        ...     {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': -3.720184399162731},
        ...     {'A': 'one', 'B': 'beh', 'C': 'bar', 'D': 0.0},
        ...     {'A': 'one', 'B': 'say', 'C': 'foo', 'D': -5.557746676883692}]
        True

"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

from builtins import *
