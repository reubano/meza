# CKAN Utils

## Introduction

CKAN Utils is a [Python library](#library) and [command line interface](#cli) for interacting with remote and local [CKAN](http://ckan.org/) instances. It uses [ckanapi](https://github.com/ckan/ckanapi) under the hood, and is essentially a high level wrapper for it.

With CKAN Utils, you can

- Download a CKAN resource
- Parse structured CSV/XLS/XLSX files and push them into a CKAN DataStore
- Copy a filestore resource from one ckan instance to another
- Read and write Uñicôdë text
- and much more...

CKAN Utils performs smart updates by computing the hash of a file and will only update the datastore if the file has changed. This allows you to schedule a script to run on a frequent basis, e.g., `@hourly` via a cron job, without updating the CKAN instance unnecessarily.

## Requirements

CKAN Utils has been tested on the following configuration:

- MacOS X 10.9.5
- Python 2.7.9

Proposer requires the following in order to run properly:

- [Python >= 2.7](http://www.python.org/download) (MacOS X comes with python preinstalled)

## Installation

(You are using a [virtualenv](http://www.virtualenv.org/en/latest/index.html), right?)

     sudo pip install -e git+https://github.com/reubano/tabutils@master#egg=tabutils

## CLI

CKAN Utils comes with a built in command line interface `ckanny`.

### Usage

     ckanny [<namespace>.]<command> [<args>]


### Examples

*show help*

    ckanny -h

```bash
usage: ckanny [<namespace>.]<command> [<args>]

positional arguments:
  command     the command to run

optional arguments:
  -h, --help  show this help message and exit

available commands:
  ver                      Show ckanny version

  [ds]
    delete                 Deletes a datastore table
    update                 Updates a datastore table based on the current filestore resource
    upload                 Uploads a file to a datastore table

  [fs]
    fetch                  Downloads a filestore resource
    migrate                Copies a filestore resource from one ckan instance to another
    upload                 Uploads a file to the filestore of an existing resource
```

*show version*

    ckanny ver

*fetch a resource*

    ckanny fs.fetch -k <CKAN_API_KEY> -r <CKAN_URL> <resource_id>

*show fs.fetch help*

    ckanny fs.fetch -h


```bash
usage: ckanny fs.fetch
       [-h] [-q] [-n] [-c CHUNKSIZE_BYTES] [-u UA] [-k API_KEY] [-r REMOTE]
       [-d DESTINATION]
       [resource_id]

Downloads a filestore resource

positional arguments:
  resource_id           the resource id

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           suppress debug statements
  -n, --name-from-id    Use resource id for filename
  -c CHUNKSIZE_BYTES, --chunksize-bytes CHUNKSIZE_BYTES
                        number of bytes to read/write at a time (default:
                        1048576)
  -u UA, --ua UA        the user agent (uses `CKAN_USER_AGENT` ENV if
                        available) (default: None)
  -k API_KEY, --api-key API_KEY
                        the api key (uses `CKAN_API_KEY` ENV if available)
                        (default: None)
  -r REMOTE, --remote REMOTE
                        the remote ckan url (uses `CKAN_REMOTE_URL` ENV if
                        available) (default: None)
  -d DESTINATION, --destination DESTINATION
                        the destination folder or file path (default:
                        .)
```

## Library

CKAN Utils may also be used directly from Python.

### Examples

*Fetch a remote resource*

```python
from tabutils import api

kwargs = {'api_key': 'mykey', 'remote': 'http://demo.ckan.org'}
resource_id = '36f33846-cb43-438e-95fd-f518104a32ed'
r, filepath = ckan.fetch_resource(resource_id, filepath='test.csv')
print(r.encoding)
```

*Fetch a local resource*

```python
ckan = api.CKAN(api_key='mykey', remote=None)
resource_id = '36f33846-cb43-438e-95fd-f518104a32ed'
r, filepath = ckan.fetch_resource(resource_id, filepath='test.csv')
print(r.encoding)
```

## Configuration

CKAN Utils will use the following [Environment Variables](http://www.cyberciti.biz/faq/set-environment-variable-linux/) if set:

Environment Variable|Description
--------------------|-----------
CKAN_API_KEY|Your CKAN API Key
CKAN_REMOTE_URL|Your CKAN instance remote url
CKAN_USER_AGENT|Your user agent

## Hash Table

In order to support file hashing, tabutils creates a hash table resource called `hash_table.csv` with the following schema:

field|type
------|----
datastore_id|text
hash|text

By default the hash table resource will be placed the package `hash_table`. CKAN Utils will create this package if it doesn't exist. Optionally, you can set the hash table package in the command line with the `-H, --hash-table` option, or in a Python file as the `hash_table` keyword argument to `api.CKAN`.

Examples:

*via the CLI*

    ckanny ds.update -H custom_hash_table 36f33846-cb43-438e-95fd-f518104a32ed

*via a python file*

```python
from tabutils import api
ckan = api.CKAN(hash_table='custom_hash_table')
hash = ckan.get_hash('36f33846-cb43-438e-95fd-f518104a32ed')
```

## Scripts

CKAN Utils comes with a built in task manager `manage.py` and a `Makefile`.

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

CKAN Utils is distributed under the [MIT License](http://opensource.org/licenses/MIT), the same as [ckanapi](https://github.com/ckan/ckanapi).
