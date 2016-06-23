meza: A Python toolkit for processing tabular data
======================================================

|travis| |versions| |pypi|

Index
-----

`Introduction`_ | `Requirements`_ | `Motivation`_ | `Hello World`_ | `Usage`_ |
`Interoperability`_ | `Installation`_ | `Project Structure`_ |
`Design Principles`_ | `Scripts`_ | `Contributing`_ | `Credits`_ |
`More Info`_ | `License`_

Introduction
------------

**meza** is a Python `library`_ for reading and processing tabular data.
It has a functional programming style API, excels at reading/writing large files,
and can process 10+ file types.

With meza, you can

- Read csv/xls/xlsx/mdb/dbf files, and more!
- Type cast records (date, float, text...)
- Process Uñicôdë text
- Lazily stream files by default
- and much more...

Requirements
------------

meza has been tested and is known to work on Python 2.7, 3.4, and 3.5;
PyPy2 5.1.1; and PyPy3 2.4.0.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

===============================  ==============  ==============================  =======================
Function                         Dependency      Installation                    File type / extension
===============================  ==============  ==============================  =======================
``meza.io.read_mdb``             `mdbtools`_     ``sudo port install mdbtools``   Microsoft Access / mdb
``meza.io.read_html``            `lxml`_ [#]_    ``pip install lxml``             HTML / html
``meza.convert.records2array``   `NumPy`_ [#]_   ``pip install numpy``            n/a
``meza.convert.records2df``      `pandas`_       ``pip install pandas``           n/a
===============================  ==============  ==============================  =======================

Notes
^^^^^

.. [#] If ``lxml`` isn't present, ``read_html`` will default to the builtin Python html reader

.. [#] ``records2array`` can be used without ``numpy`` by passing ``native=True`` in the function call. This will convert ``records`` into a list of native ``array.array`` objects.

Motivation
----------

Why I built meza
^^^^^^^^^^^^^^^^

pandas is great, but installing it isn't exactly a `walk in the park`_, and it
doesn't play nice with `PyPy`_. I designed **meza** to be a lightweight, easy to install, less featureful alternative to
pandas. I also optimized **meza** for low memory usage, PyPy compatibility, and functional programming best practices.

Why you should use meza
^^^^^^^^^^^^^^^^^^^^^^^

``meza`` provides a number of benefits / differences from similar libraries such
as ``pandas``. Namely:

- a functional programming (instead of object oriented) API
- `iterators by default`_ (reading/writing)
- `PyPy compatibility`_
- `geojson support`_ (reading/writing)
- `seamless integration`_ with sqlachemy (and other libs that work with iterators of dicts)

For more detailed information, please check-out the `FAQ`_.

Hello World
-----------

A simple data processing example is shown below:

First create a simple csv file (in bash)

.. code-block:: bash

    echo 'col1,col2,col3\nhello,5/4/82,1\none,1/1/15,2\nhappy,7/1/92,3\n' > data.csv

Now we can read the file, manipulate the data a bit, and write the manipulated
data back to a new file.

.. code-block:: python

    >>> from meza import io, process as pr, convert as cv
    >>> from io import open

    >>> # Load the csv file
    >>> records = io.read_csv('data.csv')

    >>> # `records` are iterators over the rows
    >>> row = next(records)
    >>> row
    {'col1': 'hello', 'col2': '5/4/82', 'col3': '1'}

    >>> # Let's replace the first row so as not to lose any data
    >>> records = pr.prepend(records, row)

    # Guess column types. Note: `detect_types` returns a new `records`
    # generator since it consumes rows during type detection
    >>> records, result = pr.detect_types(records)
    >>> {t['id']: t['type'] for t in result['types']}
    {'col1': 'text', 'col2': 'date', 'col3': 'int'}

    # Now type cast the records. Note: most `meza.process` functions return
    # generators, so lets wrap the result in a list to view the data
    >>> casted = list(pr.type_cast(records, result['types']))
    >>> casted[0]
    {'col1': 'hello', 'col2': datetime.date(1982, 5, 4), 'col3': 1}

    # Cut out the first column of data and merge the rows to get the max value
    # of the remaining columns. Note: since `merge` (by definition) will always
    # contain just one row, it is returned as is (not wrapped in a generator)
    >>> cut_recs = pr.cut(casted, ['col1'], exclude=True)
    >>> merged = pr.merge(cut_recs, pred=bool, op=max)
    >>> merged
    {'col2': datetime.date(2015, 1, 1), 'col3': 3}

    # Now write merged data back to a new csv file.
    >>> io.write('out.csv', cv.records2csv(merged))

    # View the result
    >>> with open('out.csv', 'utf-8') as f:
    ...     f.read()
    'col2,col3\n2015-01-01,3\n'

Usage
-----

meza is intended to be used directly as a Python library.

Usage Index
^^^^^^^^^^^

- `Reading data`_
- `Processing data`_

  + `Numerical analysis (à la pandas)`_
  + `Text processing (à la csvkit)`_
  + `Geo processing (à la mapbox)`_

- `Writing data`_
- `Cookbook`_

Reading data
^^^^^^^^^^^^

meza can read both filepaths and file-like objects. Additionally, all readers
return equivalent `records` iterators, i.e., a generator of dictionaries with
keys corresponding to the column names.

.. code-block:: python

    >>> from io import open, StringIO
    >>> from meza import io

    """Read a filepath"""
    >>> records = io.read_json('path/to/file.json')

    """Read a file like object and de-duplicate the header"""
    >>> f = StringIO('col,col\nhello,world\n')
    >>> records = io.read_csv(f, dedupe=True)

    """View the first row"""
    >>> next(records)
    {'col': 'hello', 'col_2': 'world'}

    """Read the 1st sheet of an xls file object opened in text mode."""
    # Also, santize the header names by converting them to lowercase and
    # replacing whitespace and invalid characters with `_`.
    >>> with open('path/to/file.xls', 'utf-8') as f:
    ...     for row in io.read_xls(f, sanitize=True):
    ...         # do something with the `row`
    ...         pass

    """Read the 2nd sheet of an xlsx file object opened in binary mode"""
    # Note: sheets are zero indexed
    >>> with open('path/to/file.xlsx') as f:
    ...     records = io.read_xls(f, encoding='utf-8', sheet=1)
    ...     first_row = next(records)
    ...     # do something with the `first_row`

    """Read any recognized file"""
    >>> records = io.read('path/to/file.geojson')
    >>> f.seek(0)
    >>> records = io.read(f, ext='csv', dedupe=True)

Please see `readers`_ for a complete list of available readers and recognized
file types.

Processing data
^^^^^^^^^^^^^^^

Numerical analysis (à la pandas) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following example, ``pandas`` equivalent methods are preceded by ``-->``.

.. code-block:: python

    >>> import itertools as it
    >>> import random

    >>> from io import StringIO
    >>> from meza import io, process as pr, convert as cv, stats

    # Create some data in the same structure as what the various `read...`
    # functions output
    >>> header = ['A', 'B', 'C', 'D']
    >>> data = [(random.random() for _ in range(4)) for x in range(7)]
    >>> df = [dict(zip(header, d)) for d in data]
    >>> df[0]
    {'A': 0.53908..., 'B': 0.28919..., 'C': 0.03003..., 'D': 0.65363...}

    """Sort records by the value of column `B` --> df.sort_values(by='B')"""
    >>> next(pr.sort(df, 'B'))
    {'A': 0.53520..., 'B': 0.06763..., 'C': 0.02351..., 'D': 0.80529...}

    """Select column `A` --> df['A']"""
    >>> next(pr.cut(df, ['A']))
    {'A': 0.53908170489952006}

    """Select the first three rows of data --> df[0:3]"""
    >>> len(list(it.islice(df, 3)))
    3

    """Select all data whose value for column `A` is less than 0.5
    --> df[df.A < 0.5]
    """
    >>> next(pr.tfilter(df, 'A', lambda x: x < 0.5))
    {'A': 0.21000..., 'B': 0.25727..., 'C': 0.39719..., 'D': 0.64157...}

    # Note: since `aggregate` and `merge` (by definition) return just one row,
    # they return them as is (not wrapped in a generator).
    """Calculate the mean of column `A` across all data --> df.mean()['A']"""
    >>> pr.aggregate(df, 'A', stats.mean)['A']
    0.5410437473067938

    """Calculate the sum of each column across all data --> df.sum()"""
    >>> pr.merge(df, pred=bool, op=sum)
    {'A': 3.78730..., 'C': 2.82875..., 'B': 3.14195..., 'D': 5.26330...}

Text processing (à la csvkit) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following example, ``csvkit`` equivalent commands are preceded by ``-->``.

First create a few simple csv files (in bash)

.. code-block:: bash

    echo 'col_1,col_2,col_3\n1,dill,male\n2,bob,male\n3,jane,female' > file1.csv
    echo 'col_1,col_2,col_3\n4,tom,male\n5,dick,male\n6,jill,female' > file2.csv

Now we can read the files, manipulate the data, convert the manipulated data to
json, and write the json back to a new file. Also, note that since all readers
return equivalent `records` iterators, you can use them interchangeably (in
place of ``read_csv``) to open any supported file. E.g., ``read_xls``,
``read_sqlite``, etc.

.. code-block:: python

    >>> import itertools as it

    >>> from meza import io, process as pr, convert as cv

    """Combine the files into one iterator
    --> csvstack file1.csv file2.csv
    """
    >>> records = io.join('file1.csv', 'file2.csv')
    >>> next(records)
    {'col_1': '1', 'col_2': 'dill', 'col_3': 'male'}
    >>> next(it.islice(records, 4, None))
    {'col_1': '6', 'col_2': 'jill', 'col_3': 'female'}

    # Now let's create a persistent records list
    >>> records = list(io.read_csv('file1.csv'))

    """Sort records by the value of column `col_2`
    --> csvsort -c col_2 file1.csv
    """
    >>> next(pr.sort(records, 'col_2'))
    {'col_1': '2', 'col_2': 'bob', 'col_3': 'male'

    """Select column `col_2` --> csvcut -c col_2 file1.csv"""
    >>> next(pr.cut(records, ['col_2']))
    {'col_2': 'dill'}

    """Select all data whose value for column `col_2` contains `jan`
    --> csvgrep -c col_2 -m jan file1.csv
    """
    >>> next(pr.grep(records, [{'pattern': 'jan'}], ['col_2']))
    {'col_1': '3', 'col_2': 'jane', 'col_3': 'female'}

    """Convert a csv file to json --> csvjson -i 4 file1.csv"""
    >>> io.write('file.json', cv.records2json(records))

    # View the result
    >>> with open('file.json', 'utf-8') as f:
    ...     f.read()
    '[{"col_1": "1", "col_2": "dill", "col_3": "male"}, {"col_1": "2",
    "col_2": "bob", "col_3": "male"}, {"col_1": "3", "col_2": "jane",
    "col_3": "female"}]'

Geo processing (à la mapbox) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following example, ``mapbox`` equivalent commands are preceded by ``-->``.

First create a geojson file (in bash)

.. code-block:: bash

    echo '{"type": "FeatureCollection","features": [' > file.geojson
    echo '{"type": "Feature", "id": 11, "geometry": {"type": "Point", "coordinates": [10, 20]}},' >> file.geojson
    echo '{"type": "Feature", "id": 12, "geometry": {"type": "Point", "coordinates": [5, 15]}}]}' >> file.geojson

Now we can open the file, split the data by id, and finally convert the split data
to a new geojson file-like object.

.. code-block:: python

    >>> from meza import io, process as pr, convert as cv

    # Load the geojson file and peek at the results
    >>> records, peek = pr.peek(io.read_geojson('file.geojson'))
    >>> peek[0]
    {'lat': 20, 'type': 'Point', 'lon': 10, 'id': 11}

    """Split the records by feature ``id`` and select the first feature
    --> geojsplit -k id file.geojson
    """
    >>> splits = pr.split(records, 'id')
    >>> feature_records, name = next(splits)
    >>> name
    11

    """Convert the feature records into a GeoJSON file-like object"""
    >>> geojson = cv.records2geojson(feature_records)
    >>> geojson.readline()
    '{"type": "FeatureCollection", "bbox": [10, 20, 10, 20], "features": '
    '[{"type": "Feature", "id": 11, "geometry": {"type": "Point", '
    '"coordinates": [10, 20]}, "properties": {"id": 11}}], "crs": {"type": '
    '"name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}}'

    # Note: you can also write back to a file as shown previously
    # io.write('file.geojson', geojson)

Writing data
^^^^^^^^^^^^

meza can persist ``records`` to disk via the following functions:

- ``meza.convert.records2csv``
- ``meza.convert.records2json``
- ``meza.convert.records2geojson``

Each function returns a file-like object that you can write to disk via
``meza.io.write('/path/to/file', result)``.

.. code-block:: python

    >>> from meza import io, convert as cv
    >>> from io import StringIO, open

    # First let's create a simple tsv file like object
    >>> f = StringIO('col1\tcol2\nhello\tworld\n')
    >>> f.seek(0)

    # Next create a records list so we can reuse it
    >>> records = list(io.read_tsv(f))
    >>> records[0]
    {'col1': 'hello', 'col2': 'world'}

    # Now we're ready to write the records data to file

    """Create a csv file like object"""
    >>> cv.records2csv(records).readline()
    'col1,col2\n'

    """Create a json file like object"""
    >>> cv.records2json(records).readline()
    '[{"col1": "hello", "col2": "world"}]'

    """Write back csv to a filepath"""
    >>> io.write('file.csv', cv.records2csv(records))
    >>> with open('file.csv', 'utf-8') as f_in:
    ...     f_in.read()
    'col1,col2\nhello,world\n'

    """Write back json to a filepath"""
    >>> io.write('file.json', cv.records2json(records))
    >>> with open('file.json', 'utf-8') as f_in:
    ...     f_in.readline()
    '[{"col1": "hello", "col2": "world"}]'

Cookbook
^^^^^^^^

Please see the `cookbook`_ or ipython `notebook`_ for more examples.

Notes
^^^^^

.. [#] http://pandas.pydata.org/pandas-docs/stable/10min.html#min
.. [#] https://csvkit.readthedocs.org/en/0.9.1/cli.html#processing
.. [#] https://github.com/mapbox?utf8=%E2%9C%93&query=geojson

Interoperability
----------------

meza plays nicely with NumPy and friends out of the box

setup
^^^^^

.. code-block:: python

    from meza import process as pr

    # First create some records and types. Also, convert the records to a list
    # so we can reuse them.
    >>> records = [{'a': 'one', 'b': 2}, {'a': 'five', 'b': 10, 'c': 20.1}]
    >>> records, result = pr.detect_types(records)
    >>> records, types = list(records), result['types']
    >>> types
    [
        {'type': 'text', 'id': 'a'},
        {'type': 'int', 'id': 'b'},
        {'type': 'float', 'id': 'c'}]


from records to pandas.DataFrame to records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import pandas as pd
    >>> from meza import convert as cv

    """Convert the records to a DataFrame"""
    >>> df = cv.records2df(records, types)
    >>> df
            a   b   c
    0   one   2   NaN
    1  five  10  20.1
    # Alternatively, you can do `pd.DataFrame(records)`

    """Convert the DataFrame back to records"""
    >>> next(cv.df2records(df))
    {'a': 'one', 'b': 2, 'c': nan}

from records to arrays to records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import numpy as np

    >>> from array import array
    >>> from meza import convert as cv

    """Convert records to a structured array"""
    >>> recarray = cv.records2array(records, types)
    >>> recarray
    rec.array([('one', 2, nan), ('five', 10, 20.100000381469727)],
              dtype=[('a', 'O'), ('b', '<i4'), ('c', '<f4')])
    >>> recarray.b
    array([ 2, 10], dtype=int32)

    """Convert records to a native array"""
    >>> narray = cv.records2array(records, types, native=True)
    >>> narray
    [[array('u', 'a'), array('u', 'b'), array('u', 'c')],
    [array('u', 'one'), array('u', 'five')],
    array('i', [2, 10]),
    array('f', [0.0, 20.100000381469727])]

    """Convert a 2-D NumPy array to a records generator"""
    >>> data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    >>> data
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> next(cv.array2records(data))
    {'column_1': 1, 'column_2': 2, 'column_3': 3}

    """Convert the structured array back to a records generator"""
    >>> next(cv.array2records(recarray))
    {'a': 'one', 'b': 2, 'c': nan}

    """Convert the native array back to records generator"""
    >>> next(cv.array2records(narray, native=True))
    {'a': 'one', 'b': 2, 'c': 0.0}

Installation
------------

(You are using a `virtualenv`_, right?)

At the command line, install meza using either ``pip`` (*recommended*)

.. code-block:: bash

    pip install meza

or ``easy_install``

.. code-block:: bash

    easy_install meza

Please see the `installation doc`_ for more details.

Project Structure
-----------------

.. code-block:: bash

    ┌── CONTRIBUTING.rst
    ├── LICENSE
    ├── MANIFEST.in
    ├── Makefile
    ├── README.rst
    ├── data
    │   ├── converted/*
    │   └── test/*
    ├── dev-requirements.txt
    ├── docs
    │   ├── AUTHORS.rst
    │   ├── CHANGES.rst
    │   ├── COOKBOOK.rst
    │   ├── FAQ.rst
    │   ├── INSTALLATION.rst
    │   └── TODO.rst
    ├── examples
    │   ├── usage.ipynb
    │   └── usage.py
    ├── helpers/*
    ├── manage.py
    ├── meza
    │   ├── __init__.py
    │   ├── convert.py
    │   ├── dbf.py
    │   ├── fntools.py
    │   ├── io.py
    │   ├── process.py
    │   ├── stats.py
    │   ├── typetools.py
    │   └── unicsv.py
    ├── optional-requirements.txt
    ├── py2-requirements.txt
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    ├── tests
    │   ├── __init__.py
    │   ├── standard.rc
    │   ├── test_fntools.py
    │   ├── test_io.py
    │   └── test_process.py
    └── tox.ini

Design Principles
-----------------

- prefer functions over objects
- provide enough functionality out of the box to easily implement the most common data analysis use cases
- make conversion between ``records``, ``arrays``, and ``DataFrames`` dead simple
- whenever possible, lazily read objects and stream the result [#]_

.. [#] Notable exceptions are ``meza.process.group``, ``meza.process.sort``, ``meza.io.read_dbf``, ``meza.io.read_yaml``, and ``meza.io.read_html``. These functions read the entire contents into memory up front.

Scripts
-------

meza comes with a built in task manager ``manage.py``

Setup
^^^^^

.. code-block:: bash

    pip install -r dev-requirements.txt

Examples
^^^^^^^^

*Run python linter and nose tests*

.. code-block:: bash

    manage lint
    manage test

Contributing
------------

Please mimic the coding style/conventions used in this repo.
If you add new classes or functions, please add the appropriate doc blocks with
examples. Also, make sure the python linter and nose tests pass.

Please see the `contributing doc`_ for more details.

Credits
-------

Shoutouts to `csvkit`_, `messytables`_, and `pandas`_ for heavily inspiring meza.

More Info
---------

- `FAQ`_
- `cookbook`_
- ipython `notebook`_

License
-------

meza is distributed under the `MIT License`_.

.. |travis| image:: https://img.shields.io/travis/reubano/meza/master.svg
    :target: https://travis-ci.org/reubano/meza

.. |versions| image:: https://img.shields.io/pypi/pyversions/meza.svg
    :target: https://pypi.python.org/pypi/meza

.. |pypi| image:: https://img.shields.io/pypi/v/meza.svg
    :target: https://pypi.python.org/pypi/meza

.. _mdbtools: http://sourceforge.net/projects/mdbtools/
.. _lxml: http://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser
.. _library: #usage
.. _NumPy: https://github.com/numpy/numpy
.. _PyPy: https://github.com/pydata/pandas/issues/9532
.. _walk in the park: http://pandas.pydata.org/pandas-docs/stable/install.html#installing-pandas-with-anaconda
.. _csvkit: https://github.com/onyxfish/csvkit
.. _messytables: https://github.com/okfn/messytables
.. _pandas: https://github.com/pydata/pandas
.. _MIT License: http://opensource.org/licenses/MIT
.. _virtualenv: http://www.virtualenv.org/en/latest/index.html
.. _contributing doc: https://github.com/reubano/meza/blob/master/CONTRIBUTING.rst
.. _FAQ: https://github.com/reubano/meza/blob/master/docs/FAQ.rst
.. _iterators by default: https://github.com/reubano/meza/blob/master/docs/FAQ.rst#memory
.. _PyPy compatibility: https://github.com/reubano/meza/blob/master/docs/FAQ.rst#pypy
.. _geojson support: https://github.com/reubano/meza/blob/master/docs/FAQ.rst#geojson
.. _seamless integration: https://github.com/reubano/meza/blob/master/docs/FAQ.rst#convenience
.. _notebook: http://nbviewer.jupyter.org/github/reubano/meza/blob/master/examples/usage.ipynb
.. _readers: https://github.com/reubano/meza/blob/master/docs/FAQ.rst#what-readers-are-available
.. _installation doc: https://github.com/reubano/meza/blob/master/docs/INSTALLATION.rst
.. _cookbook: https://github.com/reubano/meza/blob/master/docs/COOKBOOK.rst
