meza FAQ
========

Index
-----

`How does meza compare to pandas`_ | `What readers are available`_

How does ``meza`` compare to ``pandas``?
----------------------------------------

Philosophically, ``meza`` is designed around **functional programming** and
iterators of dictionaries whereas ``pandas`` is designed around the
**DataFrame object**. Also, ``meza`` is better suited for `ETL`_, or processing
evented / streaming data; whereas ``pandas`` seems optimized for performing
matrix transformations and linear algebra.

meza Advantages
^^^^^^^^^^^^^^^

Memory
~~~~~~

One advantage ``meza`` iterators has is that you can process extremely large
files without reading them into memory.

.. code-block:: python

    >>> import itertools as it
    >>> from meza import process as pr

    >>> records = it.repeat({'int': '1',  'time': '2:30', 'text': 'hi'})
    >>> next(pr.cut(records, ['text']))
    {'text': 'hi'}

Here I used ``it.repeat`` to simulate the output of reading a large file via
``meza.io.read...``. Most of the ``meza`` functions operate iteratively, which
means you can efficiently process files that can't fit into memory. This also
illustrates ``meza``'s functional API. Since there are no objects, you don't
need a special ``records`` constructor. Any iterable of dicts will do just fine.

PyPy
~~~~

``meza`` supports PyPy (targeting Python 2.7 & 3.4+) out of the box.

Convenience
~~~~~~~~~~~

The ``records`` data structure is compatible with other libraries such as
``sqlachemy``'s bulk insert:

.. code-block:: python

    >>> from meza import fntools as ft
    >>> from .models import Table

    # Table is a sqlalchemy.Model class
    # db is a sqlalchemy database instance
    >>> for data in ft.chunk(records, chunk_size):
    ...     db.engine.execute(Table.__table__.insert(), data)

And since ``records`` is just an iterable, you have the power of the entire
``itertools`` module at your disposal.

GeoJSON
~~~~~~~

``meza`` supports reading and writing GeoJSON out of the box.

.. code-block:: python

    >>> from meza import io, convert as cv

    # read a geojson file
    >>> records = io.read_geojson('file.geojson')

    ##
    # perform your data analysis / manipulation... then
    ##

    # convert records to a geojson file-like object
    >>> geojson = cv.records2geojson(records)

meza Disadvantage
^^^^^^^^^^^^^^^^^

The tradeoff is that you lose the speed of ``pandas`` `vectorized operations`_.
I imagine any heavy duty number crunching will be much faster in ``pandas``
than ``meza``. However, this can be partially offset by running ``meza`` under
**PyPy**.

Summary
^^^^^^^

So I would use ``pandas`` when you want **speed** or are working with
**matrices**; and ``meza`` when you are processing **streams** or **events**,
want **low memory usage**,  **geojson support**, **PyPy** compatibility, or the
convenience of working with **dictionaries**, (or if you just don't need the
raw speed of arrays).

Optimization Tips
^^^^^^^^^^^^^^^^^

I'd also like to point out one area you may like to explore if you want to
squeeze out more speed: ``meza.convert.records2array`` and
``meza.convert.array2records``. These functions can convert ``records`` to and
from a list of native ``array.array``'s. So any type of optimization techniques
you may like to explore should start there.

What readers are available?
---------------------------

Overview
^^^^^^^^

meza's available readers are outlined below:

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

Alternatively, meza provides a universal reader which will select the
appropriate reader based on the file extension as specified in the above
table.

.. code-block:: python

    >>> from io import open
    >>> from meza import io

    >>> records1 = io.read('path/to/file.csv')
    >>> records2 = io.read('path/to/file.xls')

    >>> with open('path/to/file.json', encoding='utf-8') as f:
    ...     records3 = io.read(f, ext='json')

Args
^^^^

Most readers take as their first argument, either a file path or file like object.
The notable exception is ``read_mdb`` which only accepts a file path.
File like objects should be opened using Python's stdlib ``io.open``. If the file
is opened in binary mode ``io.open('/path/to/file')``, be sure to pass the proper
encoding if it is anything other than ``utf-8``, e.g.,

.. code-block:: python

    >>> from io import open
    >>> from meza import io

    >>> with open('path/to/file.xlsx') as f:
    ...     records = io.read_xls(f, encoding='latin-1')

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
first_row   int   First row to read (zero indexed)         0        read_csv, read_fixed_fmt, read_tsv, read_xls
first_col   int   First column to read (zero indexed)      0        read_csv, read_fixed_fmt, read_tsv, read_xls
sanitize    bool  Underscorify and lowercase field names?  False    read_csv, read_dbf, read_fixed_fmt, read_html, read_mdb, read_tsv, read_xls
dedupe      bool  Deduplicate field names?                 False    read_csv, read_fixed_fmt, read_html, read_mdb, read_tsv, read_xls
sheet       int   Sheet to read (zero indexed)             0        read_xls
table       int   Table to read (zero indexed)             0        read_dbf, read_html, read_mdb, read_sqlite
==========  ====  =======================================  =======  =====================================================================================================

.. _How does meza compare to pandas: #how-does-meza-compare-to-pandas
.. _What readers are available: #what-readers-are-available
.. _vectorized operations: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#vectorized-operations-and-label-alignment-with-series
.. _ETL: https://en.wikipedia.org/wiki/Extract,_transform,_load
