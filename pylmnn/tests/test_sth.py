from unittest import TestCase

import pylmnn


class TestJoke(TestCase):
    def test_is_string(self):
        s = pylmnn.__version__
        self.assertTrue(isinstance(s, basestring))