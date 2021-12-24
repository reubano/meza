#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
meza.compat
~~~~~~~~~~~~

Provides methods for encoding/decoding content

Examples:
    basic usage::

        >>> from meza.compat import encode
        >>>
        >>> encode('some text') == b'some text'
        True
"""
import codecs
import sys

from . import ENCODING

DECODER = lambda encoding: codecs.getincrementaldecoder(encoding)()
ENCODER = lambda encoding: codecs.getincrementalencoder(encoding)()


def decode(content, encoding=ENCODING):
    """Decode bytes into unicode text

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
        >>> decode(content) == content
        True
    """
    try:
        decoded = DECODER(encoding).decode(content)
    except (TypeError, UnicodeDecodeError, UnicodeEncodeError):
        decoded = content

    return decoded


def encode(content, encoding=ENCODING):
    """Encodes unicode (or ints) into bytes

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
        >>> encode(content.encode('utf-8')) == content.encode('utf-8')
        True
    """
    if hasattr(content, "real"):
        try:
            length = (content.bit_length() // 8) + 1
        except AttributeError:
            encoded = content
        else:
            encoded = content.to_bytes(length, byteorder="big")
    elif hasattr(content, "encode"):
        try:
            encoded = ENCODER(encoding).encode(content)
        except UnicodeDecodeError:
            encoded = content
    else:
        encoded = content

    return encoded
