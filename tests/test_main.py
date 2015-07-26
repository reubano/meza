# -*- coding: utf-8 -*-
# vim: sw=4:ts=4:expandtab

"""
tests.test_main
~~~~~~~~~~~~~~~

Provides unit tests for the website.
"""

from __future__ import (
    absolute_import, division, print_function, with_statement,
    unicode_literals)

import nose.tools as nt


def setup_module():
    """site initialization"""
    global initialized
    initialized = True
    print('Site Module Setup\n')


class TestMain:
    """Main unit tests"""
    def __init__(self):
        self.cls_initialized = False

    def test_home(self):
        nt.assert_equal(self.cls_initialized, False)
