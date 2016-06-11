meza Cookbook
=============

Index
-----

`More fun with pandas`_ | `More fun with geojson files`_ | `Tips and tricks`_

More fun with pandas
--------------------

Note that the ``pandas`` equivalent methods are preceded by ``-->``.

.. code-block:: python

    >>> import itertools as it
    >>> import pandas as pd

    >>> from random import random
    >>> from meza import io, process as pr, convert as cv
    >>> from io import StringIO

    # To setup, lets define a universal header
    >>> header = ['A', 'B', 'C', 'D']

    # Create some data in the same structure as what the various `read...`
    # functions output
    >>> data = [(random.random() for _ in range(4)) for x in range(7)]
    >>> records = [dict(zip(header, d)) for d in data]
    >>> records[0]
    {'A': 0.63872..., 'B': 0.89517..., 'C': 0.11175..., 'D': 0.49445...}

    """Select multiple columns"""
    >>> next(pr.cut(records, ['A', 'B'], exclude=True))
    {'C': 0.11175001869696033, 'D': 0.4944504196475903}

    # Now create some pieces to concatenate
    >>> pieces = [it.islice(records, 3), it.islice(records, 3, None)]

    """Concatenate records together --> pd.concat(pieces)"""
    >>> concated = it.chain(*pieces)
    >>> next(concated)
    {'A': 0.63872..., 'B': 0.89517..., 'C': 0.11175..., 'D': 0.49445...}
    >>> len(list(concated)) + 1
    7

    # Now let's create two sets of records that we want to join
    >>> left = [{'key': 'foo', 'lval': 1}, {'key': 'foo', 'lval': 2}]
    >>> right = [{'key': 'foo', 'rval': 4}, {'key': 'foo', 'rval': 5}]

    """SQL style joins --> pd.merge(left, right, on='key')"""
    >>> list(pr.join(left, right))
    [
    ... {'key': 'foo', 'lval': 1, 'rval': 4},
    ... {'key': 'foo', 'lval': 1, 'rval': 5},
    ... {'key': 'foo', 'lval': 2, 'rval': 4},
    ... {'key': 'foo', 'lval': 2, 'rval': 5}]

    # Now let's create a new records-like list
    >>> records = [
    ...     {'A': 'foo', 'B': -1.202872},
    ...     {'A': 'bar', 'B': 1.814470},
    ...     {'A': 'foo', 'B': 1.8028870},
    ...     {'A': 'bar', 'B': -0.595447}]

    """Group and sum"""
    # Select a function to be applied to the records contained in each group
    # In this case, we want to merge the records by summing field `B`.
    >>> kwargs = {'aggregator': pr.merge, 'pred': 'B', 'op': sum}

    # Now group `records` by the value of field `A`, and pass `kwargs` which contains
    # details on the function to apply to each group of records.
    >>> list(pr.group(records, 'A', **kwargs)
    [{'A': 'bar', 'B': 1.219023}, {'A': 'foo', 'B': 0.600015}]

    # Now lets generate some random data to manipulate
    >>> rrange = random.sample(range(-10, 10), 12)
    >>> a = ['one', 'one', 'two', 'three'] * 3
    >>> b = ['ah', 'beh', 'say'] * 4
    >>> c = ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2
    >>> d = (random.random() * x for x in rrange)
    >>> values = zip(a, b, c, d)
    >>> records = (dict(zip(header, v)) for v in values)

    # Since `records` is an iterator over the rows, we have to be careful not
    # to inadvertently consume it. Lets use `pr.peek` to view the first few rows
    >>> records, peek = pr.peek(records)
    >>> peek
    [
    {'A': 'one', 'B': 'ah', 'C': 'foo', 'D': -3.69822},
    {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': -3.72018},
    {'A': 'two', 'B': 'say', 'C': 'foo', 'D': 1.02146},
    {'A': 'three', 'B': 'ah', 'C': 'bar', 'D': 0.38015},
    {'A': 'one', 'B': 'beh', 'C': 'bar', 'D': 0.0}]

    """Pivot tables
    --> pd.pivot_table(records, values='D', index=['A', 'B'], columns=['C'])
    """
    # Let's create a classic excel style pivot table
    >>> pivot = pr.pivot(records, 'D', 'C')
    >>> pivot, peek = pr.peek(pivot)
    >>> peek
    [
    {'A': 'one', 'B': 'ah', 'bar': 2.23933, 'foo': -3.69822},
    {'A': 'one', 'B': 'beh', 'bar': 0.0, 'foo': -3.72018},
    {'A': 'one', 'B': 'say', 'bar': 2.67595, 'foo': -5.55774},
    {'A': 'three', 'B': 'ah', 'bar': 0.38015},
    {'A': 'three', 'B': 'beh', 'foo': 5.79430}]

    """Data normalization --> pivot.stack()"""
    # To get the data back to its original form, we must normalize it.
    >>> normal = pr.normalize(pivot, 'D', 'C', ['foo', 'bar'])
    >>> normal, peek = pr.peek(normal)
    >>> peek
    [
    {'A': 'one', 'B': 'ah', 'C': 'foo', 'D': -3.69822},
    {'A': 'one', 'B': 'ah', 'C': 'bar', 'D': 2.23933},
    {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': -3.72018},
    {'A': 'one', 'B': 'beh', 'C': 'bar', 'D': 0.0},
    {'A': 'one', 'B': 'say', 'C': 'foo', 'D': -5.55774}]
    
More fun with geojson files
---------------------------

First create a few geojson files (in bash)

.. code-block:: bash

    echo '{"type": "FeatureCollection","features": [' > file1.geojson
    echo '{"type": "Feature", "id": 11, "geometry": {"type": "Point", "coordinates": [10, 20]}}]}' >> file1.geojson
    echo '{"type": "FeatureCollection","features": [' > file2.geojson
    echo '{"type": "Feature", "id": 12, "geometry": {"type": "Point", "coordinates": [5, 15]}}]}' >> file2.geojson

Now we can combine the files and write the combined data to a new geojson file.

.. code-block:: python

    >>> from io import open
    >>> from meza import io, process as pr, convert as cv

    """Combine the GeoJSON files into one iterator
    --> merge = require('geojson-merge')
    --> fs = require('fs')

    --> merged = merge(files.map(function(n) {
    ...   return JSON.parse(fs.readFileSync(n));
    ... }))
    """
    >>> filepaths = ('file1.geojson', 'file2.geojson')
    >>> records, peek = pr.peek(io.join(*filepaths))
    >>> peek[0]
    {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}

    >>> cv.records2geojson(records).read()
    {
      "type": "FeatureCollection",
      "bbox": [5, 15, 10, 20],
      "features": [
        {
          "type": "Feature",
          "id": 11,
          "geometry": {
            "type": "Point",
            "coordinates": [10, 20]
          }
        }, {
          "type": "Feature",
          "id": 12,
          "geometry": {
            "type": "Point",
            "coordinates": [5, 15]
          }
        }
      ],
      "crs": {
        "type": "name",
        "properties": {
          "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
      }
    }

Tips and tricks
---------------

Adding formulaic columns
^^^^^^^^^^^^^^^^^^^^^^^^

Say you have a table like so:

===== =====
col_1 col_2
===== =====
1     2
3     4
5     6
===== =====

and you want to add a new column that is a sum of the first two

===== ===== =====
col_1 col_2 col_3
===== ===== =====
1     2     3
3     4     7
5     6     11
===== ===== =====

you can easily do so as follows:

First create a simple csv file (in bash)

.. code-block:: bash

    echo 'col_1,col_2\n1,2\n3,4\n5,6\n' > data.csv

Now we can read the file and add the new column.

.. code-block:: python

    >>> from io import open
    >>> from meza import io, process as pr, convert as cv

    >>> # Load and type cast the csv file
    >>> raw = io.read_csv('data.csv'))
    >>> records, result = pr.detect_types(raw)
    >>> casted = list(pr.type_cast(records, result['types']))

    >>> # create the row level formula
    >>> calc_col_3 = lambda row: row['col_1'] + row['col_2']
    
    >>> # generate the new column
    >>> col_3 = [{'col_3': calc_col_3(r)} for r in casted]
    
    >>> # merge the new rows into the orginal table 
    >>> [pr.merge(r) for r in zip(casted, col_3)]
    [
    ... {'col_1': 1, 'col_2': 2, 'col_3': 3},
    ... {'col_1': 3, 'col_2': 4, 'col_3': 7},
    ... {'col_1': 5, 'col_2': 6, 'col_3': 11}]
