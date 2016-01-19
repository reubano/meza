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
        >>> from array import array
        >>> from json import loads
        >>> from datetime import date
        >>> from tabutils import io, process as pr, convert as cv, stats
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
        >>> data = [(random.random() for _ in range(4)) for x in range(7)]
        >>> df = [dict(zip(header, d)) for d in data]

        >>> df[0] ==  {
        ...     'A': 0.5390815646058106,
        ...     'B': 0.2891964436397205,
        ...     'C': 0.03003690855112706,
        ...     'D': 0.6536357538927619}
        True

        >>> next(pr.sort(df, 'B')) == {
        ...     'A': 0.535204782203361,
        ...     'B': 0.06763103158333483,
        ...     'C': 0.023510063056781383,
        ...     'D': 0.8052942869277137}
        True
        >>> next(pr.cut(df, ['A'])) == {'A': 0.5390815646058106}
        True
        >>> len(list(it.islice(df, 3)))
        3

        # Use a single column’s values to select data
        >>> rules = [{'fields': ['A'], 'pattern': lambda x: x < 0.5}]
        >>> next(pr.grep(df, rules)) == {
        ...     'A': 0.21000869554973112,
        ...     'B': 0.2572769749796092,
        ...     'C': 0.39719826263322744,
        ...     'D': 0.6415781537746728}
        True

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

        # Merge multiple GeoJSON files into one
        >>> records = io.join(*geofiles, ext='geojson')
        >>> next(records) == {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}
        True

        # Split records by geojson feature
        >>> splits = pr.split(records, 'id')
        >>> sub_records, name = next(splits)
        >>> name
        12
        >>> geojson = cv.records2geojson(sub_records)
        >>> geojson.readline() == (
        ...     '{"type": "FeatureCollection", "bbox": [5, 15, 5, 15], '
        ...     '"features": [{"type": "Feature", "id": 12, "geometry": '
        ...     '{"type": "Point", "coordinates": [5, 15]}, "properties": '
        ...     '{"id": 12}}], "crs": {"type": "name", "properties": {"name": '
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
        >>> df = pd.DataFrame(records)
        >>> df.sort_index(1)
              a   b     c
        0   one   2   NaN
        1  five  10  20.1
        >>> df.dtypes
        a     object
        b      int64
        c    float64
        dtype: object

        >>> df = cv.records2df(records, types)
        >>> df.sort_index(1)
              a   b     c
        0   one   2   NaN
        1  five  10  20.1
        >>> df.sort_index(1).dtypes
        a     object
        b      int32
        c    float32
        dtype: object

        # Convert a DataFrame to records
        >>> row = next(cv.df2records(df))
        >>> row['a'] == 'one'
        True
        >>> row['b']
        2
        >>> np.isnan(row['c'])
        True

        # Convert records to a structured array
        >>> recarray = cv.records2array(records, types)
        >>> recarray.a.tolist() == ['one', 'five']
        True
        >>> recarray.b
        array([ 2, 10], dtype=int32)
        >>> clist = recarray.c.tolist()
        >>> np.isnan(clist[0])
        True
        >>> clist[1]
        20.100000381469727

        # Convert a 2-D array to records
        >>> data = np.array([[1, 2, 3], [4, 5, 6]], 'i4')
        >>> data
        array([[1, 2, 3],
               [4, 5, 6]], dtype=int32)

        >>> next(cv.array2records(data)) == {
        ...     'column_1': 1, 'column_2': 2, 'column_3': 3}
        True

        # Convert a structured array to records
        >>> row = next(cv.array2records(recarray))
        >>> row['a'] == 'one'
        True
        >>> row['b']
        2
        >>> np.isnan(row['c'])
        True

        # Convert records to a native array
        >>> narray = cv.records2array(records, result['types'], True)
        >>> def get_values(narray):
        ...     try:
        ...         yield narray.tounicode()
        ...     except ValueError:
        ...         yield narray.tolist()
        ...     except AttributeError:
        ...         for n in narray:
        ...             for x in get_values(n):
        ...                 yield x
        >>> values = list(get_values(narray))
        >>> nested = ([x for x in v] for v in values if isinstance(v, list))
        >>> set(it.chain.from_iterable(nested)) == {
        ...     0.0, 20.100000381469727, 2, 10}
        True
        >>> set(v for v in values if not isinstance(v, list)) ==  {
        ...     'a', 'b', 'c', 'one', 'five'}
        True

        # Convert native array to records
        >>> next(cv.array2records(narray, True)) == {
        ...     'a': 'one', 'b': 2, 'c': 0.0}
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

        >>> pivot = pr.pivot(records, 'D', 'C')
        >>> pivot, peek = pr.peek(pivot)
        >>> set(int(p.get('bar', 0)) for p in peek).issubset({0, 2})
        True
        >>> set(int(p.get('foo', 0)) for p in peek).issubset({-5, -3, -2, 0, 5})
        True

        # Data normalization
        >>> normal = pr.normalize(pivot, 'D', 'C', ['foo', 'bar'])
        >>> normal, peek = pr.peek(normal)
        >>> set(int(p['D'] or 0) for p in peek).issubset({-5, -3, -2, 0, 2})
        True

"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

from builtins import *
