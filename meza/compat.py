#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.compat
~~~~~~~~~~~~

Provides methods for py2/3 compatibility

Examples:
    basic usage::

        >>> from meza.compat import encode
        >>>
        >>> encode('some text') == b'some text'
        True
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import codecs
import sys

from . import ENCODING

DECODER = lambda encoding: codecs.getincrementaldecoder(encoding)()
ENCODER = lambda encoding: codecs.getincrementalencoder(encoding)()
BYTE_TYPE = bytes if sys.version_info.major > 2 else str


def decode(content, encoding=ENCODING):
    """Decode bytes (py2-str) into unicode

    Args:
        content (scalar): the content to analyze
        encoding (str)

    Returns:
        unicode

    Examples:
        >>> decode(b'Hello World!') == 'Hello World!'
        True
        >>> content = 'Iñtërnâtiônàližætiøn!'
        >>> decode(content.encode('utf-8')) == content
        True
    """
    try:
        decoded = DECODER(encoding).decode(content)
    except (TypeError, UnicodeDecodeError):
        decoded = content

    return decoded


def encode(content, encoding=ENCODING):
    """Encodes unicode (or ints) into bytes (py2-str)

    Args:
        content (scalar): A string or int

    Examples:
        >>> encode('Hello World!') == b'Hello World!'
        True
        >>> content = 'Iñtërnâtiônàližætiøn!'
        >>> encode(content) == content.encode('utf-8')
        True
        >>> len(encode(1024))
        2
    """
    if hasattr(content, 'real'):
        try:
            length = (content.bit_length() // 8) + 1
        except AttributeError:
            encoded = content
        else:
            try:
                encoded = content.to_bytes(length, byteorder='big')
            except AttributeError:
                # Backport py3 to_bytes for py2
                # http://stackoverflow.com/a/20793663/408556
                _hex = '%x' % content
                zeros = '0' * (len(_hex) % 2) + _hex
                encoded = zeros.zfill(length * 2).decode('hex')
    elif hasattr(content, 'encode'):
        try:
            encoded = ENCODER(encoding).encode(content)
        except UnicodeDecodeError:
            encoded = content
    else:
        encoded = content

    return encoded


def get_native_str(text):
    """Encode py2-unicode into bytes (py2-str) but leave py3 text as is
    """
    # dtype bug https://github.com/numpy/numpy/issues/2407
    if sys.version_info.major < 3:
        try:
            encoded = text.encode('ascii')
        except AttributeError:
            encoded = text
    else:
        encoded = text

    return encoded
