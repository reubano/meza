#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.unicsv
~~~~~~~~~~~

Provides py2 compatible methods for reading and writing unicode csv data in
a py3 futurized codebase.
"""

from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import sys

if sys.version_info.major == 2:
    import csv
    import codecs
    import cStringIO

    from . import ENCODING
    from ._compat import encode, decode

    def encode_all(f=None, **kwargs):
        """
        Encode unicode into bytes (str)
        """
        names = kwargs.pop('fieldnames', None)
        encoding = kwargs.pop('encoding', None) if f else False
        fmtkeys = set(dir(csv.Dialect))
        drkeys = fmtkeys.union(['fieldnames', 'restkey', 'restval', 'dialect'])
        dwkeys = fmtkeys.union(['restval', 'extrasaction', 'dialect'])

        decoded = codecs.iterdecode(f, encoding) if encoding else f
        ekwargs = {encode(k): encode(v) for k, v in kwargs.items()}

        res = {
            'f': codecs.iterencode(decoded, ENCODING) if f else None,
            'fieldnames': [encode(x) for x in names] if names else None,
            'drkwargs': {k: v for k, v in ekwargs.items() if k in drkeys},
            'dwkwargs': {k: v for k, v in ekwargs.items() if k in dwkeys},
            'fmtparams': {k: v for k, v in ekwargs.items() if k in fmtkeys}}

        return res

    def reader(f, dialect='excel', **kwargs):
        """
        >>> from io import StringIO

        >>> f = StringIO()
        >>> bool(f.write('Şpâm Şpâm Şpâm |Bâkëd Bëâñs|\\n'))
        True
        >>> bool(f.write('Şpâm |Łôvëly Şpâḿ| |Ŵôndërful Şpâm|\\n'))
        True
        >>> bool(f.seek(0))
        False
        >>> kwargs = {'delimiter': ' ', 'quotechar': '|'}
        >>> unireader = reader(f, **kwargs)
        >>> unireader.next() == ['Şpâm', 'Şpâm', 'Şpâm', 'Bâkëd Bëâñs']
        True
        >>> unireader.next() == ['Şpâm', 'Łôvëly Şpâḿ', 'Ŵôndërful Şpâm']
        True
        """
        res = encode_all(f, **kwargs)

        for row in csv.reader(res['f'], dialect, **res['fmtparams']):
            yield [decode(r) for r in row]

    class UnicodeWriter(object):
        """
        >>> from io import StringIO

        >>> f = StringIO()
        >>> w = UnicodeWriter(f)
        >>> w.writerow((u'é', u'ñ'))
        >>> f.getvalue().strip() == 'é,ñ'
        True
        """
        def __init__(self, f, dialect='excel', **fmtparams):
            self.queue = cStringIO.StringIO()
            self.writer = csv.writer(self.queue, dialect, **fmtparams)
            self.f = f

        def writerow(self, row):
            self.writer.writerow([encode(s) for s in row])
            data = self.queue.getvalue()
            self.f.write(decode(data))
            self.queue.truncate(0)

        def writerows(self, rows):
            [self.writerow(row) for row in rows]

    def writer(f, dialect='excel', **kwargs):
        """
        >>> from io import StringIO

        >>> f = StringIO()
        >>> kwargs = {'delimiter': ' ', 'quotechar': '|'}
        >>> uniwriter = writer(f, **kwargs)
        >>> uniwriter.writerow(['Şpâm'] * 5 + ['Bâkëd Bëâñs'])
        >>> uniwriter.writerow(['Şpâm', 'Łôvëly Şpâḿ', 'Ŵôndërful Şpâm'])
        >>> text = f.getvalue().split('\\r\\n')
        >>> text[0] == 'Şpâm Şpâm Şpâm Şpâm Şpâm |Bâkëd Bëâñs|'
        True
        >>> text[1] == 'Şpâm |Łôvëly Şpâḿ| |Ŵôndërful Şpâm|'
        True
        """
        res = encode_all(**kwargs)
        return UnicodeWriter(f, dialect, **res['fmtparams'])

    class DictReader(csv.DictReader):
        """
        >>> from io import StringIO

        >>> f = StringIO()
        >>> bool(f.write('a,ñ,b\\n'))
        True
        >>> bool(f.write('1,2,ø\\n'))
        True
        >>> bool(f.write('é,2,î\\n'))
        True
        >>> bool(f.seek(0))
        False
        >>> r = DictReader(f, fieldnames=['a', 'ñ'], restkey='r')
        >>> r.next() == {'a': 'a', 'ñ':'ñ', 'r': ['b']}
        True
        >>> r.next() == {'a': '1', 'ñ':'2', 'r': ['ø']}
        True
        >>> r.next() == {'a': 'é', 'ñ':'2', 'r': ['î']}
        True

        >>> f = StringIO()
        >>> bool(f.write('name,place\\n'))
        True
        >>> bool(f.write('Câry Grâñt,høllywøød\\n'))
        True
        >>> bool(f.write('Nâthâñ Brillstøñé\\n'))
        True
        >>> bool(f.seek(0))
        False
        >>> r = DictReader(f, restval='Løndøn')
        >>> r.next() == {'name': 'Câry Grâñt', 'place': 'høllywøød'}
        True
        >>> r.next() == {'name': 'Nâthâñ Brillstøñé', 'place': 'Løndøn'}
        True
        """
        def __init__(self, f, fieldnames=None, **kwargs):
            res = encode_all(f, fieldnames=fieldnames, **kwargs)
            args = (self, res['f'], res['fieldnames'])
            csv.DictReader.__init__(*args, **res['drkwargs'])
            self.restkey = res['drkwargs'].get('restkey')

        def next(self):
            row = csv.DictReader.next(self)

            if self.restkey:
                row[self.restkey] = [decode(x) for x in row[self.restkey]]

            return {decode(k): decode(v) for k, v in row.items()}

    class DictWriter(csv.DictWriter):
        """
        >>> from io import StringIO

        >>> f = StringIO()
        >>> w = DictWriter(f, ['a', 'ñ', 'b'], restval='î')
        >>> w.writeheader()
        >>> w.writerows([{'a':'1', 'ñ':'2', 'b':'ø'}, {'a':'é', 'ñ':'2'}])
        >>> text = f.getvalue().split('\\r\\n')
        >>> text[0] == 'a,ñ,b'
        True
        >>> text[1] == '1,2,ø'
        True
        >>> text[2] == 'é,2,î'
        True

        >>> f = StringIO()
        >>> w = DictWriter(f, ['name', 'place'])
        >>> w.writeheader()
        >>> w.writerow({'name': 'Câry Grâñt', 'place': 'høllywøød'})
        >>> w.writerow({'name': 'Nâthâñ Brillstøñé', 'place': 'Løndøn'})
        >>> text = f.getvalue().split('\\r\\n')
        >>> text[0] == 'name,place'
        True
        >>> text[1] == 'Câry Grâñt,høllywøød'
        True
        >>> text[2] == 'Nâthâñ Brillstøñé,Løndøn'
        True
        """
        def __init__(self, f, fieldnames=None, **kwargs):
            res = encode_all(fieldnames=fieldnames, **kwargs)
            args = (self, f, fieldnames)
            csv.DictWriter.__init__(*args, **res['dwkwargs'])
            self.writer = UnicodeWriter(f, **res['fmtparams'])
