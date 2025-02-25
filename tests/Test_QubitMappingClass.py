import unittest
import networkx as nx
import sys

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from QuantumEnvironment import QubitMappingClass
from QuantumEnvironment import QPUClass
from QuantumEnvironment import DAGClass

class TestQubitMappingClass(unittest.TestCase):

    def setUp(self):
        self.my_arch = QPUClass()
        self.G = self.my_arch.G
        self.my_DAG = DAGClass()
        self.max_epr_pairs = 5
        self.numQubits = self.my_DAG.numQubits
        self.numNodes = self.my_arch.numNodes
        self.initial_mapping = {i: i for i in range(self.numQubits + 2 * self.max_epr_pairs)}
        self.mapping = QubitMappingClass(self.numNodes, self.numQubits, self.max_epr_pairs, self.G, self.initial_mapping)
    
    def test_generate_EPR_pair(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = list(self.mapping.EPR_pairs.keys())[0]
        self.assertIn(epr_id, self.mapping.EPR_pairs)
    
    def test_destroy_EPR_pair(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = 'EPR-0'
        self.mapping.destroy_EPR_pair(epr_id)
        self.assertNotIn(epr_id, self.mapping.EPR_pairs)
    
    def test_ball_in_epr_pairs(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = 'EPR-0'
        found, pair_id = self.mapping.ball_in_epr_pairs(self.mapping.get_ball(0))
        self.assertTrue(found)
        self.assertEqual(pair_id, epr_id)
    
    def test_generate_random_initial_mapping(self):
        random_mapping = QubitMappingClass(self.numNodes, self.numQubits, self.max_epr_pairs, self.G)
        self.assertEqual(random_mapping.get_ball(0), self.numQubits)
        self.assertEqual(random_mapping.get_ball(16), self.numQubits+1)
        for i in range(self.numQubits+2* self.max_epr_pairs):
            self.assertIsNotNone(random_mapping.get_box(i))

if __name__ == "__main__":
    unittest.main()