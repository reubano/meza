#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza._compat
~~~~~~~~~~~~

Provides methods for py2/3 compatibility

Examples:
    basic usage::

        >>> from meza._compat import encode

        >>> encode('some text')
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

from builtins import *

import codecs
import sys

from . import ENCODING

decoder = lambda encoding: codecs.getincrementaldecoder(encoding)()
encoder = lambda encoding: codecs.getincrementalencoder(encoding)()


def decode(content, encoding=ENCODING):
    """Decode bytes (py2-str) into unicode

    Args:
        content (scalar): the content to analyze
        encoding (str)

    Returns:
        unicode

    Examples:
        >>> from datetime import datetime as dt, date, time
    """
    try:
        decoded = decoder(encoding).decode(content)
    except (TypeError, UnicodeDecodeError):
        decoded = content

    return decoded


def encode(content, encoding=ENCODING, parse_ints=False):
    """Encode unicode (or ints) into bytes (py2-str)
    """
    if hasattr(content, 'encode'):
        try:
            encoded = encoder(encoding).encode(content)
        except UnicodeDecodeError:
            encoded = content
    elif parse_ints:
        try:
            length = (content.bit_length() // 8) + 1
        except AttributeError:
            encoded = content
        else:
            try:
                encoded = content.to_bytes(length, byteorder='big')
            except AttributeError:
                # http://stackoverflow.com/a/20793663/408556
                h = '%x' % content
                zeros = '0' * (len(h) % 2) + h
                encoded = zeros.zfill(length * 2).decode('hex')
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
