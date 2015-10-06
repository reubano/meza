# tabutils

## Introduction

tabutils is a [Python library](#library) for reading and processing data from tabular formatted files.

With tabutils, you can

- Read CSV/XLS/XLSX/MDB/DBF files
- Type cast records (date, float, text)
- Read Uñicôdë text
- Lazily read large files by default
- and much more...

## Requirements

tabutils has been tested on the following configuration:

- MacOS X 10.9.5
- Python 2.7.10

tabutils requires the following in order to run properly:

- [Python >= 2.7.6](http://www.python.org/download) (MacOS X comes with python preinstalled)

## Installation

(You are using a [virtualenv](http://www.virtualenv.org/en/latest/index.html), right?)

     sudo pip install tabutils

## Library

The tabutils library may be used directly from Python.

### Examples

*Read a csv file*

```python
from tabutils.io import read_csv

csv_records = read_csv('path/to/file.csv')
row = csv_records.next()
```

*Read an xls/xlsx file*

```python
from tabutils.io import read_xls

xls_records = read_xls('path/to/file.xls')
row = xls_records.next()
```

## Scripts

tabutils comes with a built in task manager `manage.py` and a `Makefile`.

### Setup

    pip install -r dev-requirements.txt

### Examples

*Run python linter and nose tests*

```bash
manage lint
manage test
```

Or if `make` is more your speed...

```bash
make lint
make test
```

## Contributing

View [CONTRIBUTING.rst](https://github.com/reubano/tabutils/blob/master/CONTRIBUTING.rst)

## License

tabutils is distributed under the [MIT License](http://opensource.org/licenses/MIT).
