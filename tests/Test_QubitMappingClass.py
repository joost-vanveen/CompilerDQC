import unittest
import networkx as nx
from QuantumEnvironment.QubitMappingClass import QubitMappingClass
from QuantumEnvironment.QPUClass import QPUClass
from QuantumEnvironment.DAGClass import DAGClass

class TestQubitMappingClass(unittest.TestCase):

    def setUp(self):
        self.my_arch = QPUClass()
        self.G = self.my_arch.G
        self.my_DAG = DAGClass()
        self.max_epr_pairs = 9
        self.numQubits = self.numQubits
        self.numNodes = self.my_arch.numNodes
        self.initial_mapping = {i: i for i in range(self.numQubits + 2 * self.numEPR_threshold)}
        self.mapping = QubitMappingClass(self.numNodes, self.numQubits, self.max_epr_pairs, self.G, self.initial_mapping)
    
    def test_get_box(self):
        self.assertEqual(self.mapping.get_box(0), 0)
        self.assertRaises(Exception, self.mapping.get_box, self.numQubits + 10)
    
    def test_get_ball(self):
        self.assertEqual(self.mapping.get_ball(0), 0)
        self.assertIsNone(self.mapping.get_ball(self.numNodes - 1))
    
    def test_generate_EPR_pair(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = list(self.mapping.EPR_pairs.keys())[0]
        self.assertIn(epr_id, self.mapping.EPR_pairs)
    
    def test_destroy_EPR_pair(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = list(self.mapping.EPR_pairs.keys())[0]
        self.mapping.destroy_EPR_pair(epr_id)
        self.assertNotIn(epr_id, self.mapping.EPR_pairs)
    
    def test_query_EPR_pair(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = list(self.mapping.EPR_pairs.keys())[0]
        self.assertIsNotNone(self.mapping.query_EPR_pair(epr_id))
        self.assertRaises(Exception, self.mapping.query_EPR_pair, "EPR-100")
    
    def test_ball_in_epr_pairs(self):
        self.mapping.generate_EPR_pair(0, 16)
        epr_id = list(self.mapping.EPR_pairs.keys())[0]
        found, pair_id = self.mapping.ball_in_epr_pairs(self.mapping.get_ball(0))
        self.assertTrue(found)
        self.assertEqual(pair_id, epr_id)
    
    def test_generate_random_initial_mapping(self):
        mapping = self.mapping.generate_random_initial_mapping(self.G)
        self.assertIsInstance(mapping, dict)
        self.assertGreaterEqual(len(mapping), self.numQubits + 2 * self.numEPR_threshold)
        self.assertRaises(ValueError, QubitMappingClass, self.numQubits - 1, self.numQubits, self.numEPR_threshold, self.G, None)

if __name__ == "__main__":
    unittest.main()