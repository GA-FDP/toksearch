import unittest
from toksearch.imas_mappings.imas_mapper import IMASMapper

class Test_IMAS_Mapper(unittest.TestCase):

    def test_init(self):
        imas_mapper = IMASMapper()

    def test_single_mdsplus_fetch(self):
        imas_mapper = IMASMapper()
        time = imas_mapper.resolve_mapped("d3d", [170235], ["equilibrium.time"])

if __name__ == "__main__":
    imas_mapping_tester = Test_IMAS_Mapper()
    imas_mapping_tester.test_single_mdsplus_fetch()