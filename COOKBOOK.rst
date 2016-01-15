tabutils Cookbook
=================

Index
-----

`Introduction`_ | `Requirements`_ | `Motivation`_ | `Usage`_ | `Interoperability`_ |
`Installation`_ | `Project Structure`_ | `Design Principles`_ | `Readers`_ |
`Converters`_ | `Scripts`_ | `Contributing`_ | `Credits`_ | `License`_

Note that the ``pandas`` equivalent methods are preceded by ``-->`` and assumes
``import pandas as pd``. Command output is preceded by ``>>>``.

.. code-block:: python

    import itertools as it

    from random import random
    from tabutils import io, process as pr, convert as cv
    from io import StringIO

    """Concatenate records together --> pd.concat(pieces)"""
    # First lets reconstitute a new `records` iterator
    records = iter(df)

    # Now create some pieces to concatenate
    pieces = [it.islice(records, 3), it.islice(records, 3, None)]
    concated = pr.concat(pieces)
    list(concated) == df
    >>> True

    """SQL style joins --> pd.merge(left, right, on='key')"""
    left = [{'key': 'foo', 'lval': 1}, {'key': 'foo', 'lval': 2}]
    right = [{'key': 'foo', 'rval': 4}, {'key': 'foo', 'rval': 5}]
    list(pr.join(left, right))
    >>> [
    ... {'key': 'foo', 'lval': 1, 'rval': 4},
    ... {'key': 'foo', 'lval': 1, 'rval': 5},
    ... {'key': 'foo', 'lval': 2, 'rval': 4},
    ... {'key': 'foo', 'lval': 2, 'rval': 5}]

    """Group and sum --> df.groupby('A').sum()"""
    df = [
        {'A': 'foo', 'B': -1.202872},
        {'A': 'bar', 'B': 1.814470},
        {'A': 'foo', 'B': 1.8028870},
        {'A': 'bar', 'B': -0.595447}]

    # Select a function to be applied to the records contained in each group
    # In this case, we want to merge the records by summing field `B`.
    kwargs = {'aggregator': pr.merge, 'pred': 'B', 'op': sum}

    # Now group `df` by the value of field `A`, and pass `kwargs` which contains
    # details on the function to apply to each group of records.
    list(pr.group(df, 'A', **kwargs)
    >>> [{'A': 'bar', 'B': 1.219023}, {'A': 'foo', 'B': 0.600015}]

    """Pivot tables
    --> pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
    """
    rrange = random.sample(range(-10, 10), 12)
    header = ['A', 'B', 'C', 'D', 'E']
    a = ['one', 'one', 'two', 'three'] * 3
    b = ['ah', 'beh', 'say'] * 4
    c = ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2
    d = (random.random() * x for x in rrange)
    e = (random.random() * x for x in rrange)
    values = zip(a, b, c, d, e)
    records = (dict(zip(header, v)) for v in values)

    # Since `records` is an iterator over the rows, we have to be careful not
    # to inadvertently consume it. Lets use `pr.peek` to view the first few rows
    df, peek = pr.peek(records)
    peek
    >>> [
    ... {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': 4.695..., 'E': -3.458...},
    ... {'A': 'two', 'B': 'say', 'C': 'foo', 'D': -1.773..., 'E': -3.447...},
    ... {'A': 'three', 'B': 'ah', 'C': 'bar', 'D': 0.920..., 'E': 0.206...}]

    # Now lets pivot the data
    df = pr.pivot(df, 'D', 'C', rows=['A', 'B'])
    pivot, peek = pr.peek(df)
    peek
    >>> [
    ... {'A': 'one', 'B': 'ah', 'bar': 4.695..., 'foo': -3.458...},
    ... {'A': 'one', 'B': 'beh', 'bar': 4.695..., 'foo': -3.458...},
    ... {'A': 'one', 'B': 'say', 'bar': 4.695..., 'foo': -3.458...},
    ... {'A': 'two', 'B': 'ah', 'bar': 4.695..., 'foo': -3.458...},
    ... {'A': 'two', 'B': 'beh', 'bar': 4.695..., 'foo': -3.458...},

    """Data normalization --> pivot.stack()"""
    # To get the data back to its original form, we must normalize it.
    # Note: column `E` is missing since we excluded it from the pivot table
    list(pr.normalize(pivot, 'D', 'C', ['foo', 'bar']))[:5]
    >>> [
    ... {'A': 'one', 'B': 'beh', 'C': 'foo', 'D': 4.695...},
    ... {'A': 'two', 'B': 'say', 'C': 'foo', 'D': -1.773...},
    ... {'A': 'three', 'B': 'ah', 'C': 'bar', 'D': 0.920...}]

Text processing (à la csvkit) [#]_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the ``csvkit`` equivalent commands are preceded by ``-->``.
Command output is preceded by ``>>>``.

.. code-block:: bash

    # First create a few simple csv files (in bash)
    echo 'col_1,col_2,col_3\n1,bill,male\n2,bob,male\n3,jane,female\n' > file1.csv
    echo 'col_1,col_2,col_3\n4,tom,male\n5,dick,male\n6,jill,female\n' > file2.csv

.. code-block:: python

    """
    The examples below showcase the textual analysis of csv files. Note: all
    readers return equivalent `records` generators, i.e., an iterable of dicts
    with keys corresponding to the header. Any of the other readers can be
    used in place of `read_csv`, e.g., `read_xls`, `read_sqlite`, etc.
    """

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

    """ Converts CSV and TSV files into GeoJSON
    --> fs = require('fs')
    --> concat = require('concat-stream')

    --> function convert(data) {
    ...   csv2geojson.csv2geojson(data.toString(), {}, function(err, data) {
    ...     // data
    ...   })
    ... }

    --> fs.createReadStream('file.csv').pipe(concat(convert))
    """
    converted = cv.records2geojson(io.read_csv('file.csv'))

    """ Merge multiple GeoJSON files into one
    --> merge = require('geojson-merge')
    --> fs = require('fs')

    --> merged = merge(files.map(function(n) {
    ...   return JSON.parse(fs.readFileSync(n));
    ... }))
    """
    files = ['file1.geojson', 'file2.geojson']
    merged_records = io.join(*files)

    """ Split a GeoJSON file by feature
    geojsplit -k id file.geojson
    """
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
    from io import StringIO

    f = StringIO('col1,col2\nhello,world\n')
    records = io.read_csv(f)
    io.write('file.json', cv.records2json(records))

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

    # Then create a DataFrame from those records
    df = pd.DataFrame(records)
    df
    >>>     a   b   c
    ...  0  1   2 NaN
    ...  1  5  10  20

    # Finally convert back to a new records generator
    records = cv.df2records(df)
    next(records)
    >>> {'a': 1, 'b': 2, 'c': None}

from tabutils records to numpy.recarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from tabutils import process as pr, convert as cv

    records = iter([{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}])
    records, types = pr.detect_types(records)
    recordarr = cv.records2recarray(records, types)
    recordarr
    >>>

from NumPy arrays to tabutils records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from tabutils import convert as cv

    # From 2-D array
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    data
    >>> array([[1, 2, 3],
    ...        [4, 5, 6]], dtype=int32)
    records = cv.array2records(data)
    next(records)
    >>> {'column_1': 1, 'column_2': 2, 'column_3': 3}

    # From structured or record array
    data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])
    data[:] = [(1, 2., 'Hello'), (2, 3., 'World')]
    data
    >>> array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
    ...       dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])
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

    records1 = io.read('path/to/file.csv')
    records2 = io.read('path/to/file.xls')
    records3 = io.read('path/to/file.json')

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

While each read has kwargs specific to itself, the following tables outline
kwargs which are common to most readers.

==========  ====  ======================================  =======  ============================================================================================================================
kwarg       type  description                             default  supported readers
==========  ====  ======================================  =======  ============================================================================================================================
mode        str   The file open mode                      rU       read_csv, read_fixed_fmt, read_geojson, read_html, read_json, read_tsv, read_xls, read_yaml
encoding    str   File encoding                           utf-8    read_csv, read_dbf, read_fixed_fmt, read_geojson, read_html, read_json, read_tsv, read_xls, read_yaml
has_header  bool  Has header row                          True     read_csv, read_fixed_fmt, read_tsv, read_xls, read_yaml
first_row   int   Zero indexed first row                  0        read_csv, read_fixed_fmt, read_tsv, read_xls, read_yaml
first_col   int   Zero indexed first column               0        read_csv, read_fixed_fmt, read_tsv, read_xls, read_yaml
sanitize    bool  Underscorify and lowercase field names  False    read_csv, read_dbf, read_fixed_fmt, read_html, read_mdb, read_sqlite, read_tsv, read_xls, read_yaml
dedupe      bool  Deduplicate field names                 False    read_csv, read_fixed_fmt, read_html, read_mdb, read_sqlite, read_tsv, read_xls, read_yaml
sheet       int   Zero indexed sheet to open              0        read_dbf, read_sqlite, read_tsv, read_xls, read_yaml
table       int   Zero indexed table to read              0        read_dbf, read_html, read_mdb, read_sqlite, read_tsv, read_xls, read_yaml
==========  ====  ======================================  =======  ============================================================================================================================

Summary
^^^^^^^

The following table can help make sense of the different readers:


Notes
^^^^^

.. [#] https://docs.python.org/2/howto/logging-cookbook.html#multiple-handlers-and-formatters
.. [#] https://docs.python.org/2/howto/logging-cookbook.html#implementing-structured-logging

Converters
----------

Once you have read data into ``records``, tabutils has several builtin
converters which you can use to write the output data.

Examples
^^^^^^^^

Summary
^^^^^^^

The following table can help make sense of the different builtin record
converters:

+-----------------------+-----------------------+-----------------+
| File type             | Common extension(s)   | Converter       |
+=======================+=======================+=================+
| Comma separated file  | csv                   | records2csv     |
+-----------------------+-----------------------+-----------------+
| JSON                  | json                  | records2json    |
+-----------------------+-----------------------+-----------------+
| GeoJSON               | geojson, geojson.json | records2geojson |
+-----------------------+-----------------------+-----------------+

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
