tabutils: A Python toolkit for processing tabular data
======================================================

|travis| |versions| |pypi|

Index
-----

`Introduction`_ | `Requirements`_ | `Motivation`_ | `Usage`_ | `Interoperability`_ |
`Installation`_ | `Project Structure`_ | `Design Principles`_ | `Readers`_ |
`Scripts`_ | `Contributing`_ | `Credits`_ | `License`_

Introduction
------------

tabutils is a Python library_ for reading and processing tabular data.
It has a functional programming style API, excels at reading, large files,
and can process 10+ file types.

With tabutils, you can

- Read csv/xls/xlsx/mdb/dbf files, and more!
- Type cast records (date, float, text...)
- Process Uñicôdë text
- Lazily stream files by default
- and much more...

Requirements
------------

tabutils has been tested and is known to work on Python 2.7, 3.4, and 3.5;
PyPy 4.0; and PyPy3 2.4

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

+------------------+-------------------------+-----------+--------------+----------------------------+
| File type        | Recognized extension(s) | Reader    | Dependency   | Installation               |
+==================+=========================+===========+==============+============================+
| Microsoft Access | mdb                     | read_mdb  | `mdbtools`_  | sudo port install mdbtools |
+------------------+-------------------------+-----------+--------------+----------------------------+
| HTML table       | html                    | read_html | `lxml`_ [#]_ | pip install lxml           |
+------------------+-------------------------+-----------+--------------+----------------------------+

Notes
^^^^^
.. [#] If ``lxml`` isn't present, ``read_html`` will default to the builtin Python html reader

Motivation
----------

Pandas is great, but installing it isn't exactly a `walk in the park`_. It also
doesn't play nice with `PyPy`_. `csvkit`_ is an equally useful project, but it
doesn't expose the same API when used as `a library`_ as it does via the command
line. I designed tabutils to provide much of same functionality as
Pandas and csvkit, while using functional programming methods.

A simple data processing example is shown below:

.. code-block:: bash

    # First create a simple csv file (in bash)
    echo 'col1,col2,col3\nhello,5/4/82,1\none,1/1/15,2\nhappy,7/1/92,3\n' > data.csv

.. code-block:: python

    from tabutils import io, process as pr, convert as cv
    from io import open

    # Load the csv file
    records = io.read_csv('data.csv')

    # `records` are iterators over the rows
    row = next(records)
    row
    >>> {'col1': 'hello', 'col2': '5/4/82', 'col3': '1'}

    # Let's replace the first row so as not to loose any data
    records = pr.prepend(records, row)

    # Guess column types. Note: `detect_types` returns a new `records`
    # generator since it consumes rows during type detection
    records, result = pr.detect_types(records)
    {t['id']: t['type'] for t in result['types']}
    >>> {'col1': 'text', 'col2': 'date', 'col3': 'int'}

    # Now type cast the records. Note: most `tabutils.process` functions return
    # generators, so lets wrap the result in a list to view the data
    casted = list(pr.type_cast(records, types))
    casted[0]
    >>> {'col1': 'hello', 'col2': datetime.date(), 'col3': 1}

    # Cut out the first column of data and merge the rows to get max value of
    # the remaining columns. Note: since `merge` (by definition) will always
    # contain just one row, it is returned as is (not wrapped in a generator)
    cut_recs = pr.cut(casted, ['col1'], exclude=True)
    merged = pr.merge(cut_recs, pred=bool, op=max)
    merged
    >>> {'col2': datetime.date(), 'col3': 3}

    # Now write data back to a new csv file.
    io.write('out.csv', cv.records2csv(merged))
    open('out.csv', 'utf-8').read()
    >>> 'col2,col3\ndatetime.date(),3\n'

.. _library:

Usage
-----

tabutils is intended to be used directly as a Python library.

Usage Index
^^^^^^^^^^^

- `Reading data`_
- `Processing data`_

  + `Numerical analysis (à la pandas)`_
  + `Text processing (à la csvkit)`_
  + `Geo processing (à la mapbox)`_

- `Writing data`_

Reading data
^^^^^^^^^^^^

.. code-block:: python

    from __future__ import print_function
    from tabutils import io
    from io import open, StringIO

    # Note: all readers return equivalent `records` iterators, i.e., a generator
    # of dicts with keys corresponding to the header.

    """Read a filepath"""
    records = io.read_json('path/to/file.json')

    """Read a file like object and de-duplicate the header"""
    f = StringIO('col,col\nhello,world\n')
    records = io.read_csv(f, dedupe=True)

    """View the first row"""
    next(records)
    >>> {'col_1': 'hello', 'col_2': 'world'}

    """Read the 1st sheet of an xls file object opened in text mode."""
    # Also, santize the header names by converting them to lowercase and
    # replacing whitespace and invalid characters with `_`.
    with open('path/to/file.xls', 'utf-8') as f:
        for row in io.read_xls(f, sanitize=True):
            print(row)

    """Read the 2nd sheet of an xlsx file object opened in binary mode"""
    # Note: sheets are zero indexed
    with open('path/to/file.xlsx') as f:
        records = io.read_xls(f, encoding='utf-8', sheet=1)
        [print(row) for row in records]

    """Read any recognized file type"""
    records = io.read('path/to/file.geojson')
    records = io.read(f, ext='csv')

Please see `Readers`_ for a complete list of available readers and recognized
file types.

Processing data
^^^^^^^^^^^^^^^

Numerical analysis (à la pandas) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the ``pandas`` equivalent methods are preceded by ``-->`` and assumes
``import pandas as pd``. Command output is preceded by ``>>>``.

.. code-block:: python

    import itertools as it

    from random import random
    from tabutils import io, process as pr, convert as cv
    from io import StringIO

    # Create some data in the same structure as what the various `read...`
    # functions output
    header = ['A', 'B', 'C', 'D']
    records = (dict(zip(header, (random() for _ in range(4)))) for x in range(7))

    # Since this is an interactive example, we need to view the intermediate
    # results. Lets convert the generator to a list to make things easy.
    df = list(records)
    df[0]
    >>> [{'A': 0.4555..., 'B': 0.4166..., 'C': 0.2770..., 'D': 0.9439...}]

    """Sort records by values --> df.sort_values(by='B')"""
    next(pr.sort(df, 'B'))
    >>> [{'A': 0.9563..., 'B': 0.1251..., 'C': 0.6772..., 'D': 0.5208...}]

    """Select a single column of data --> df['A']"""
    next(pr.cut(df, ['A']))
    >>> [{'A': 0.4555170489952006}]

    """Select a slice of rows --> df[0:3]"""
    len(list(it.islice(df, 3)))
    >>> 3

    """Use a single column’s values to select data --> df[df.A > 0.5]"""
    rules = [{'fields': ['A'], 'pattern': lambda x: x > 0.5}]
    next(pr.grep(df, rules))
    >>> [{'A': 0.7388..., 'B': 0.7404..., 'C': 0.4560..., 'D': 0.9671...}]

    # Note: since `aggregate` and `merge` (by definition) return just one row,
    # they return them as is (not wrapped in a generator).
    """Calculate a descriptive statistic (on one field) --> df.mean()['A']"""
    mean = lambda l: sum(l) / len(l)
    pr.aggregate(df, 'A', mean)['A']
    >>> 0.34225751867139953

    """Calculate a descriptive (binary function safe) statistic --> df.sum()"""
    pr.merge(df, pred=bool, op=sum)
    >>> {'A': 2.3958..., 'C': 4.1317..., 'B': 1.1860..., 'D': 3.4386...}

Text processing (à la csvkit) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the ``csvkit`` equivalent commands are preceded by ``-->``.
Command output is preceded by ``>>>``.

.. code-block:: bash

    # First create a few simple csv files (in bash)
    echo 'col_1,col_2,col_3\n1,bill,male\n2,bob,male\n3,jane,female\n' > file1.csv
    echo 'col_1,col_2,col_3\n4,tom,male\n5,dick,male\n6,jill,female\n' > file2.csv

.. code-block:: python

    # Note: since all readers return equivalent `records` iterators, you can
    # use any one in place of `read_csv`. E.g., `read_xls`, `read_sqlite`, etc.

    import itertools as it

    from tabutils import io, process as pr, convert as cv

    """Join multiple files together by stacking the contents
    --> csvstack *.csv
    """
    records = io.join('file1.csv', 'file2.csv')
    next(records)
    >>> {'col_1': 1, 'col_2': 'bill', 'col_3': 'male'}
    next(it.islice(records, 5, None))
    >>> {'col_1': 6, 'col_2': 'jill', 'col_3': 'female'}

    """Sort records of a file --> csvsort -c col_2 file1.csv"""
    records = io.read_csv('file1.csv')
    next(pr.sort(records, 'col_2'))
    >>> {'col_1': 6, 'col_2': 'jill', 'col_3': 'female'}

    """Select individual columns --> csvcut -c col_2 file1.csv"""
    records = io.read_csv('file1.csv')
    next(pr.cut(records, ['col_2']))
    >>> {'col_1': 6, 'col_2': 'jill', 'col_3': 'female'}

    """Search for individual rows --> csvgrep -c col_1 -m jane file1.csv"""
    rules = [{'fields': ['col_1'], 'pattern': 'jane'}]
    next(pr.grep(records, rules))
    >>> {'col_1': 6, 'col_2': 'jill', 'col_3': 'female'}

    """Convert a csv file to json --> csvjson -i 4 file.csv"""
    records = io.read_csv('file.csv')
    io.write('file.json', cv.records2json(records))
    open('file.json', 'utf-8').read()
    >>>

Geo processing (à la mapbox) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the ``mapbox`` equivalent commands are preceded by ``-->``.
Command output is preceded by ``>>>``.

.. code-block:: python

    from tabutils import io, process as pr, convert as cv

    """Convert a csv file to GeoJSON
    --> fs = require('fs')
    --> concat = require('concat-stream')

    --> function convert(data) {
    ...   csv2geojson.csv2geojson(data.toString(), {}, function(err, data) {
    ...     console.log(data)
    ...   })
    ... }

    --> fs.createReadStream('file.csv').pipe(concat(convert))
    """
    f = cv.records2geojson(io.read_csv('file.csv'))
    f.readline()

    """Merge multiple GeoJSON files into one
    --> merge = require('geojson-merge')
    --> fs = require('fs')

    --> merged = merge(files.map(function(n) {
    ...   return JSON.parse(fs.readFileSync(n));
    ... }))
    """
    files = ['file1.geojson', 'file2.geojson']
    records = io.join(*files)
    next(records)

    """Split a GeoJSON file by feature --> geojsplit -k id file.geojson"""
    records = io.read_geojson(file.geojson)
    records, result = pr.detect_types(records)
    casted_records = pr.type_cast(records, result['types'])

    for sub_records, name in pr.split(casted_records, 'id'):
        f = cv.records2geojson(sub_records, key=id_)
        io.write('{}.geojson'.format(name), f)

Writing data
^^^^^^^^^^^^

.. code-block:: python

    from tabutils import io
    from io import StringIO, open

    # First let's create a simple tsv file like object
    f = StringIO('col1\tcol2\nhello\tworld\n')
    f.seek(0)

    # Next create a records list so we can reuse it
    records = list(io.read_tsv(f))
    records[0]
    >>>

    # Now we're ready to write the records data to file

    """Create a csv file like object"""
    f_out = cv.records2csv(records)
    f_out.readline()
    >>> 'col1,col2\n'

    """Create a json file like object"""
    f_out = cv.records2json(records)
    f_out.readline()
    >>>

    """Write back csv to a filepath"""
    io.write('file.csv', cv.records2csv(records))
    with open('file.csv', 'utf-8') as f_in:
        f_in.readline()
    >>>

    """Write back json to a filepath"""
    io.write('file.json', cv.records2json(records))
    with open('file.json', 'utf-8') as f_in:
        f_in.readline()
    >>>

Notes
^^^^^

.. [#] http://pandas.pydata.org/pandas-docs/stable/10min.html#min
.. [#] https://csvkit.readthedocs.org/en/0.9.1/cli.html#processing
.. [#] https://github.com/mapbox?utf8=%E2%9C%93&query=geojson

Interoperability
----------------

tabutils plays nicely with NumPy and friends out of the box

from tabutils records to pandas.DataFrame to pandas.DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from tabutils import convert as cv

    # First create a records iterator
    records = iter([{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}])

    """Convert records to a DataFrame"""
    df = pd.DataFrame(records)
    df
    >>>     a   b   c
    ...  0  1   2 NaN
    ...  1  5  10  20

    """Convert a DataFrame to a records generator"""
    records = cv.df2records(df)
    next(records)
    >>> {'a': 1, 'b': 2, 'c': None}

from tabutils records to numpy.recarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from tabutils import process as pr, convert as cv

    # First create a records iterator
    records = iter([{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}])

    """Convert records to a structured array"""
    records, types = pr.detect_types(records)
    recordarr = cv.records2recarray(records, types)
    recordarr
    >>>

from NumPy arrays to tabutils records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from tabutils import convert as cv

    # First create a 2-D NumPy array
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    data
    >>> array([[1, 2, 3],
    ...        [4, 5, 6]], dtype=int32)

    """Convert a 2-D array to a records generator"""
    records = cv.array2records(data)
    next(records)
    >>> {'column_1': 1, 'column_2': 2, 'column_3': 3}

    # Now create a structured array
    data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])
    data[:] = [(1, 2., 'Hello'), (2, 3., 'World')]
    data
    >>> array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
    ...       dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])

    """Convert a structured array to a records generator"""
    records = cv.array2records(data)
    next(records)
    >>> {'A': 1, 'B': 2.0, 'C': 'Hello'}


Installation
------------

(You are using a `virtualenv`_, right?)

At the command line, install tabutils using either ``pip`` (*recommended*)

.. code-block:: bash

    pip install tabutils

or ``easy_install``

.. code-block:: bash

    easy_install tabutils

Please see the `installation doc`_ for more details.

Project Structure
-----------------

.. code-block:: bash

    ┌── AUTHORS.rst
    ├── CHANGES.rst
    ├── CONTRIBUTING.rst
    ├── INSTALLATION.rst
    ├── LICENSE
    ├── MANIFEST.in
    ├── Makefile
    ├── README.rst
    ├── TODO.rst
    ├── data
    │   ├── converted
    │   │   ├── dbf.csv
    │   │   ├── fixed.csv
    │   │   ├── geo.csv
    │   │   ├── geojson.csv
    │   │   ├── json.csv
    │   │   ├── json_multiline.csv
    │   │   └── sheet_2.csv
    │   └── test
    │       ├── fixed.txt
    │       ├── fixed_w_header.txt
    │       ├── iris.csv
    │       ├── irismeta.csv
    │       ├── latin1.csv
    │       ├── mac_newlines.csv
    │       ├── newline.json
    │       ├── no_header_row.csv
    │       ├── test.csv
    │       ├── test.dbf
    │       ├── test.geojson
    │       ├── test.html
    │       ├── test.json
    │       ├── test.mdb
    │       ├── test.sqlite
    │       ├── test.tsv
    │       ├── test.xls
    │       ├── test.xlsx
    │       ├── test.yml
    │       ├── utf16_big.csv
    │       ├── utf16_little.csv
    │       └── utf8.csv
    ├── dev-requirements.txt
    ├── examples.py
    ├── helpers
    │   ├── check-stage
    │   ├── clean
    │   ├── pippy
    │   ├── srcdist
    │   └── wheel
    ├── manage.py
    ├── py2-requirements.txt
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    ├── tabutils
    │   ├── __init__.py
    │   ├── convert.py
    │   ├── dbf.py
    │   ├── fntools.py
    │   ├── io.py
    │   ├── process.py
    │   ├── stats.py
    │   ├── typetools.py
    │   └── unicsv.py
    ├── tests
    │   ├── __init__.py
    │   ├── standard.rc
    │   ├── test_fntools.py
    │   ├── test_io.py
    │   └── test_process.py
    └── tox.ini

Design Principles
-----------------

- the built-in ``logging`` module isn't broken so don't reinvent the wheel
- prefer functions over objects
.. - stream all the things! [#]_

Whenever possible, tabutils lazily reads objects and
streams the result. Notable exceptions are ``tabutils.process.group``,
``tabutils.io.read_dbf``, ``tabutils.io.read_yaml``, and ``tabutils.io.read_html``
which read the entire contents into memory up front.

Readers
-------

tabutils' available readers are outlined below:

+-----------------------+-------------------------+----------------+
| File type             | Recognized extension(s) | Default reader |
+=======================+=========================+================+
| Comma separated file  | csv                     | read_csv       |
+-----------------------+-------------------------+----------------+
| dBASE/FoxBASE         | dbf                     | read_dbf       |
+-----------------------+-------------------------+----------------+
| Fixed width file      | fixed                   | read_fixed_fmt |
+-----------------------+-------------------------+----------------+
| GeoJSON               | geojson, geojson.json   | read_geojson   |
+-----------------------+-------------------------+----------------+
| HTML table            | html                    | read_html      |
+-----------------------+-------------------------+----------------+
| JSON                  | json                    | read_json      |
+-----------------------+-------------------------+----------------+
| Microsoft Access      | mdb                     | read_mdb       |
+-----------------------+-------------------------+----------------+
| SQLite                | sqlite                  | read_sqlite    |
+-----------------------+-------------------------+----------------+
| Tab separated file    | tsv                     | read_tsv       |
+-----------------------+-------------------------+----------------+
| Microsoft Excel       | xls, xlsx               | read_xls       |
+-----------------------+-------------------------+----------------+
| YAML                  | yml, yaml               | read_yaml      |
+-----------------------+-------------------------+----------------+

Alternatively, tabutils provides a universal reader which will select the
appropriate reader based on the file extension as specified in the above
table.

.. code-block:: python

    from tabutils import io

    f = io.open('path/to/file.json')

    records1 = io.read('path/to/file.csv')
    records2 = io.read('path/to/file.xls')
    records3 = io.read(f, ext='json')

Args
^^^^

All readers take as their first argument, either a file path or file like object.
File like objects should be opened using Python's stdlib ``io.open``. If the file
is opened in binary mode ``io.open('/path/to/file')``, be sure to pass the proper
encoding if it is anything other than ``utf-8``, e.g.,

.. code-block:: python

    from io import open

    with open('path/to/file.csv') as f:
        records = io.read_xls(f, encoding='latin-1')

Kwargs
^^^^^^

While each reader has kwargs specific to itself, the following table outlines
the most common ones.

==========  ====  =======================================  =======  =====================================================================================================
kwarg       type  description                              default  implementing readers
==========  ====  =======================================  =======  =====================================================================================================
mode        str   File open mode                           rU       read_csv, read_fixed_fmt, read_geojson, read_html, read_json, read_tsv, read_xls, read_yaml
encoding    str   File encoding                            utf-8    read_csv, read_dbf, read_fixed_fmt, read_geojson, read_html, read_json, read_tsv, read_xls, read_yaml
has_header  bool  Data has a header row?                   True     read_csv, read_fixed_fmt, read_tsv, read_xls
first_row   int   First row (zero indexed)                 0        read_csv, read_fixed_fmt, read_tsv, read_xls
first_col   int   First column (zero indexed)              0        read_csv, read_fixed_fmt, read_tsv, read_xls
sanitize    bool  Underscorify and lowercase field names?  False    read_csv, read_dbf, read_fixed_fmt, read_html, read_mdb, read_tsv, read_xls
dedupe      bool  Deduplicate field names?                 False    read_csv, read_fixed_fmt, read_html, read_mdb, read_tsv, read_xls
sheet       int   Sheet to read (zero indexed)             0        read_xls
table       int   Table to read (zero indexed)             0        read_dbf, read_html, read_mdb, read_sqlite
==========  ====  =======================================  =======  =====================================================================================================

Scripts
-------

tabutils comes with a built in task manager ``manage.py``

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

Shoutouts to `csvkit`_, `messytables`_, and `pandas`_ for heavily inspiring tabutils.

License
-------

tabutils is distributed under the `MIT License`_.

.. |travis| image:: https://img.shields.io/travis/reubano/tabutils/master.svg
    :target: https://travis-ci.org/reubano/tabutils

.. |versions| image:: https://img.shields.io/pypi/pyversions/tabutils.svg
    :target: https://pypi.python.org/pypi/tabutils

.. |pypi| image:: https://img.shields.io/pypi/v/tabutils.svg
    :target: https://pypi.python.org/pypi/tabutils

.. _mdbtools: http://sourceforge.net/projects/mdbtools/
.. _lxml: http://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser
.. _a library: https://csvkit.readthedocs.org/en/0.9.1/api/csvkit.py3.html
.. _PyPy: https://github.com/pydata/pandas/issues/9532
.. _walk in the park: http://pandas.pydata.org/pandas-docs/stable/install.html#installing-pandas-with-anaconda
.. _csvkit: https://github.com/onyxfish/csvkit
.. _messytables: https://github.com/okfn/messytables
.. _pandas: https://github.com/pydata/pandas
.. _MIT License: http://opensource.org/licenses/MIT
.. _virtualenv: http://www.virtualenv.org/en/latest/index.html
.. _contributing doc: https://github.com/reubano/tabutils/blob/master/CONTRIBUTING.rst
.. _installation doc: https://github.com/reubano/tabutils/blob/master/INSTALLATION.rst
