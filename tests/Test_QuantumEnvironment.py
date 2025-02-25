import random
import unittest
import networkx as nx
import sys
from unittest.mock import MagicMock

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from QuantumEnvironment import QubitMappingClass
from QuantumEnvironment import QPUClass
from QuantumEnvironment import DAGClass
from QuantumEnvironment import QuantumEnvironmentClass
from Constants import Constants

class TestQuantumEnvironmentClass(unittest.TestCase):
    
    def setUp(self):
        """Initialize a test instance of QuantumEnvironmentClass."""
        self.env = QuantumEnvironmentClass()
    
    def test_get_qubit_pairs(self):
        """Test if all possible qubit pairs are correctly generated."""
        pairs = self.env.get_qubit_pairs()
        expected_pairs = [(i, j) for i in range(self.env.qubit_amount - 1) for j in range(i + 1, self.env.qubit_amount)]
        self.assertEqual(pairs, expected_pairs)
    
    def test_environment_reset(self):
        """Test if the environment resets correctly."""
        self.env.generate_initial_state = MagicMock()
        self.env.update_state_vector = MagicMock(return_value=([], []))
        
        self.env.environment_reset()
        
        self.assertIsInstance(self.env.my_DAG, DAGClass)
        self.env.generate_initial_state.assert_called()
        self.env.update_state_vector.assert_called()
    
    def test_is_action_possible(self):
        """Test if the action possibility check works correctly."""
        self.env.G.nodes[1]['weight'] = 0
        self.env.G.nodes[2]['weight'] = 0
        self.assertTrue(self.env.is_action_possible((1, 2)))
        
        self.env.G.nodes[1]['weight'] = 1
        self.assertFalse(self.env.is_action_possible((1, 2)))
    
    def test_reduce_cooldowns(self):
        """Test if cooldowns reduce correctly."""
        self.env.G.nodes = {1: {'weight': 2}, 2: {'weight': 1}, 3: {'weight': 0}}
        self.env.reduce_cooldowns()
        self.assertEqual(self.env.G.nodes[1]['weight'], 1)
        self.assertEqual(self.env.G.nodes[2]['weight'], 0)
        self.assertEqual(self.env.G.nodes[3]['weight'], 0)
    
    def test_update_state_vector(self):
        """Test if state vector updates correctly."""
        state_vector = self.env.update_state_vector()
        self.assertIsInstance(state_vector, list)
        for i, (ball1, ball2) in enumerate(self.env.pairs):
            if (self.env.qm.get_box(ball1) < 16 and self.env.qm.get_box(ball2) > 15) or (self.env.qm.get_box(ball1) > 15 and self.env.qm.get_box(ball2) < 16):
                self.assertEqual(state_vector[i+self.env.num_entanglement_links], float("inf"))
            else:
                self.assertNotEqual(state_vector[i+self.env.num_entanglement_links], 0)

    def test_swap(self):
        """Test if a valid action executes correctly."""
        box1_ball = self.env.qm.get_ball(1)
        box2_ball = self.env.qm.get_ball(2)

        self.env.swap((1,2))

        self.assertEqual(box1_ball, self.env.qm.get_ball(2))
        self.assertEqual(box2_ball, self.env.qm.get_ball(1))

    def test_generate(self):
        """Test if a valid action executes correctly."""
        ball1, ball2 = self.env.qm.get_ball(0), self.env.qm.get_ball(16)
        neighbors = list(self.env.G.neighbors(0))
        neighbor_ball = self.env.qm.get_ball(neighbors[0])
        state_vector = self.env.update_state_vector()

        if ball2 > neighbor_ball:
            initial_distance = state_vector[self.env.pairs.index((neighbor_ball, ball2))+self.env.num_entanglement_links]
        else:
            initial_distance = state_vector[self.env.pairs.index((ball2, neighbor_ball))+self.env.num_entanglement_links]

        while len(self.env.qm.EPR_pairs) != 1:
            self.env.generate((0,16))
            for i in range(Constants.COOLDOWN_GENERATE):
                self.env.stop()

        state_vector = self.env.update_state_vector()

        if ball2 > neighbor_ball:
            epr_distance = state_vector[self.env.pairs.index((neighbor_ball, ball2))+self.env.num_entanglement_links]
        else:
            epr_distance = state_vector[self.env.pairs.index((ball2, neighbor_ball))+self.env.num_entanglement_links]

        self.assertEqual(len(self.env.qm.EPR_pairs), 1)

        self.assertEqual(initial_distance, float('inf'))
        self.assertEqual(epr_distance, 5)

    def test_telequbit(self):
        """Test if a valid action executes correctly."""
        ball1, ball2 = self.env.qm.get_ball(0), self.env.qm.get_ball(16)
        neighbor_ball1 = self.env.qm.get_ball(1)

        while len(self.env.qm.EPR_pairs) != 1:
            self.env.generate((0,16))
            for i in range(Constants.COOLDOWN_GENERATE):
                self.env.stop()

        self.env.tele_qubit((0,1))

        self.assertEqual(len(self.env.qm.EPR_pairs), 0)
        self.assertEqual(self.env.qm.get_box(neighbor_ball1), 16)
        self.assertEqual(self.env.qm.get_ball(1), ball2)

    def test_longswap(self):
        """Test if a valid action executes correctly."""
        while len(self.env.qm.EPR_pairs) != 1:
            self.env.generate((0,16))
            for i in range(Constants.COOLDOWN_GENERATE):
                self.env.stop()

        self.env.update_state_vector()
        ball2 = self.env.qm.get_ball(2)
        ball1 = self.env.qm.get_ball(1)
        ball0 = self.env.qm.get_ball(0)
        ball16 = self.env.qm.get_ball(16)

        taken_actions, path_length = self.env.decode_action_fromNum(2+self.env.pairs.index((ball2, ball16)))

        print("Action list:", taken_actions)

        print("Original ball loc", ball2, ball1, ball0, ball16)

        self.assertNotEqual(path_length, float("inf"))

        for taken_action in taken_actions:        
            self.env.perform_action(taken_action['action'], taken_action['edge']) 

        self.assertEqual(self.env.qm.get_ball(2), ball1)
        self.assertEqual(self.env.qm.get_ball(1), ball16)
        self.assertEqual(self.env.qm.get_ball(0), ball0)
        self.assertEqual(self.env.qm.get_ball(16), ball2)

    def test_action_decode(self):
        self.env.environment_reset()
        for num in range(1+self.env.num_entanglement_links):
            taken_actions, path_length = self.env.decode_action_fromNum(num)
            if num < 1+self.env.num_entanglement_links:
                self.assertEqual(len(taken_actions), 1)
                self.assertEqual(path_length, 0)

    def test_actions(self):
        self.env.environment_reset()
        print("Initial state", self.env.state)
        random_values = list(range(self.env.action_size))
        while len(random_values) != 1:
            random_values = list(range(self.env.action_size))
            for i, value in enumerate(self.env.state):
                if value == float('inf'):
                    random_values.remove(i+1)
            
            action = random.choice(random_values)
            if action != 0:
                self.env.RL_step(action)

        print(self.env.state)
        for node in self.env.G.nodes:
            print(node, " has cooldown of ", self.env.G.nodes[node]['weight'])

        for action_num in range(1, self.env.action_size):
            actions, path_length = self.env.decode_action_fromNum(action_num)
            for action in actions:
                if self.env.is_action_possible(action['edge']):
                    print("Incorrect action is", action)
                self.assertFalse(self.env.is_action_possible(action['edge']))

if __name__ == '__main__':
    unittest.main()