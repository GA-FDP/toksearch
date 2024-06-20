import unittest
from toksearch.imas_mappings.imas_mapper import IMASMapper

class Test_IMAS_Mapper(unittest.TestCase):

    def test_init(self):
        imas_mapper = IMASMapper()