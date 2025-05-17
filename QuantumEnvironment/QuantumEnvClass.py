
from operator import sub
import networkx as nx
from itertools import combinations, groupby
import numpy as np
import random
import matplotlib.patches as mpatches

import copy
import json

from QuantumEnvironment.QPUClass import QPUClass
from QuantumEnvironment.DAGClass import DAGClass
from QuantumEnvironment.QubitMappingClass import QubitMappingClass
from Constants import Constants


class QuantumEnvironmentClass():
    
    def __init__(self, learn_from_file=False):

        #Initialize the DQC architecture
        self.my_arch = QPUClass()
        self.G = self.my_arch.G
        #Initialize the DAG - quantum circuit
        self.my_DAG = DAGClass(False, dag_list=None)

        self.num_entanglement_links = self.my_arch.numEdgesQuantum
        self.max_epr_pairs = 7
        self.qubit_amount = self.my_DAG.numQubits + 2*self.max_epr_pairs

        self.DAG_left = self.my_DAG.numGates
        self.action_amount = 0
        self.swap_amount = 0
        self.EPR_amount = 0
        self.telequbit_amount = 0

        self.action_queues = []

        if learn_from_file:
            with open("saved_dag.json", 'r') as f:
                self.dag_list_data = json.load(f)
            with open("saved_mapping.json", 'r') as f:
                raw_mappings = json.load(f)
            self.mapping_list = [{int(k): v for k, v in mapping.items()} for mapping in raw_mappings]
            self.iteration = 0


        self.pairs = self.get_qubit_pairs()
        
        self.action_size, self.state_size = self.generate_action_and_state_size()

        self.generate_initial_state(save_mapping=False) 
        self.state, self.mask = self.update_state_vector()
        self.state, self.mask = np.array(self.state), np.array(self.mask)
        #print("At the beginning, state is: ", self.get_qubit_location_vector())
    
        
    #!this function generates inital state based on processor architecture and initial DAG conditions  (each_reset_start)  - note that it generates a state object from the class SystemStateClass 
    def generate_initial_state(self, save_mapping, intial_mapping=None):  
        self.qm = QubitMappingClass(self.my_arch.numNodes, self.my_DAG.numQubits, self.max_epr_pairs, self.G, initial_mapping=intial_mapping, save_mapping=save_mapping)
        self.update_frontier()
        self.distance_metric = self.calculate_distance_metric()
        self.distance_metric_prev = self.distance_metric

    
    #!this function generates the action space size based on possible actions that could be taken
    def generate_action_and_state_size(self):
        # prev action size was 1 + quantum edges + qubit edges = 32
        #action_size = 1 + (self.qubit_amount * (self.qubit_amount - 1)) - (self.max_epr_pairs*2 * (self.max_epr_pairs*2 - 1)) + self.num_entanglement_links
        action_size = 1 + self.qubit_amount + self.num_entanglement_links #1 for cool off, then amount of qubit pairs plus amount of quantum links
        state_size = self.my_arch.numNodes + (3*self.my_DAG.numGates) # amount of qubits pairs plus the size of the DAG
        return action_size, state_size
    

    #!creates all possible qubit pairs, where qubits >= logical qubit amount are reserved for EPR pair creation
    def get_qubit_pairs(self):
        pairs = []

        for ball1 in range(self.qubit_amount):
            for ball2 in range(self.qubit_amount):
                #if ball1 != ball2 and (ball1 < self.my_DAG.numQubits or ball2 < self.my_DAG.numQubits):
                if ball1 != ball2:
                    pairs.append((ball1, ball2))

        return pairs
        
    
    #!after each completion (DAG completion or deadline failure), next game starts with environment/game reset (each_reset_start)      
    def environment_reset(self, save_data=False): #environment reset fn
        
        #Initialize the DAG - quantum circuit
        if len(self.dag_list_data) > 0 and len(self.mapping_list) > 0:
            current_dag = self.dag_list_data[self.iteration]
            current_mapping = self.mapping_list[self.iteration]
            save_data=False

        self.my_DAG = DAGClass(save_data, dag_list=current_dag)      ##HERE WE WILL CHANGE THE DAG IN EVERY TIME SLOT BUT FOR NOW WE FIX A SINGLE ONE - NOTE THAT THE STATE SPACE WILL CHANGE WITH NONE AT THE END BUT WE WILL HAVE A FIXED MAX GATE NUMBER
        self.DAG_left = self.my_DAG.numGates
        self.action_amount = 0
        self.swap_amount = 0
        self.EPR_amount = 0
        self.telequbit_amount = 0
        self.action_size, self.state_size = self.generate_action_and_state_size()

        self.generate_initial_state(save_mapping=save_data, intial_mapping=current_mapping) 
        self.state, self.mask = self.update_state_vector()

        #print("After Reset, state is: ", self.get_qubit_location_vector())


    #!checks if the link is not on cooldown  
    def is_action_possible(self, link):
        # Now we need to check both nodes involved in the link
        return self.G.nodes[link[0]]['weight'] == 0 and self.G.nodes[link[1]]['weight'] == 0

    #!reduces all cooldown in the graph since advance to next timestep
    def reduce_cooldowns(self):
        for node in self.G.nodes:
            if self.G.nodes[node]['weight'] > 0:
                self.G.nodes[node]['weight'] -= 1   


    #!update distnces between all the logical qubit pairs
    def update_state_vector(self):
        mask = [1]

        # Copy graph to add EPR links and swap weights for pathfinding
        self.pathfinding_G = self.G.copy()

        # Adjust edge weights based on qubit cooldown and quantum links
        for edge in self.pathfinding_G.edges:
            node1, node2 = edge

            # If the edge is a quantum link, set weight to infinity (makes it unusable for shortest path)
            if self.pathfinding_G.edges[node1, node2]['label'] == 'quantum':
                self.pathfinding_G.edges[node1, node2]['weight'] = float('inf')
                ball1, ball2 = self.qm.get_ball(node1), self.qm.get_ball(node2)
                # check if logical qubits are located at link and they are not on cooldown
                if ball1 != None and ball2 != None and self.G.nodes[node1]['weight'] == 0 and self.G.nodes[node2]['weight'] == 0:
                    in_epr1, epr_id1 = self.qm.ball_in_epr_pairs(ball1)
                    in_epr2, epr_id2 = self.qm.ball_in_epr_pairs(ball2)
                    # check if there is not already EPR pair and if qubits are reserved for EPR pair gen
                    if ball1 >= self.my_DAG.numQubits and ball2 >= self.my_DAG.numQubits and not in_epr1 and not in_epr2:
                        mask.append(1)
                    else:
                        mask.append(0)
                else:
                    mask.append(0)
            else:
                # Set the weight to 3 if the qubits are not on cooldown
                if self.pathfinding_G.nodes[node1]['weight'] > 0 or self.pathfinding_G.nodes[node2]['weight'] > 0:
                    self.pathfinding_G.edges[node1, node2]['weight'] = float('inf')
                else:
                    self.pathfinding_G.edges[node1, node2]['weight'] = 3

        # Add edges for EPR pairs and set their weight to 2 (extra cd compared to swap)
        for (ball1, ball2) in self.qm.EPR_pairs.values(): 
            box1, box2 = self.qm.get_box(ball1), self.qm.get_box(ball2)
            if self.pathfinding_G.nodes[box1]['weight'] > 0 or self.pathfinding_G.nodes[box2]['weight'] > 0:
                self.pathfinding_G.add_edge(self.qm.get_box(ball1), self.qm.get_box(ball2), weight=float('inf'))
            else:
                self.pathfinding_G.add_edge(self.qm.get_box(ball1), self.qm.get_box(ball2), weight=2)

        # Remove all node weights
        for node in self.pathfinding_G.nodes:
            self.pathfinding_G.nodes[node]['weight'] = 0

        reserved_qubits_per_qpu = self.qm.reserved_qubits_on_qpus(Constants.NUMQ, [16, 16])


        # finds all paths for valid actions
        all_paths = []
        for (fball1, fball2, _) in self.frontier:
            box1 = self.qm.get_box(fball1)
            box2 = self.qm.get_box(fball2)

            # Path from box1 to box2
            fpath1 = nx.shortest_path(self.pathfinding_G, source=box1, target=box2)
            all_paths.append(fpath1)

            # Path from box2 to box1
            fpath2 = nx.shortest_path(self.pathfinding_G, source=box2, target=box1)
            all_paths.append(fpath2)

            # From box1 to neighbors of box2 (if edge is not 'quantum')
            for neighbor in self.pathfinding_G.neighbors(box2):
                if self.G.edges[box2, neighbor]['label'] != 'quantum':
                    path = nx.shortest_path(self.pathfinding_G, source=box1, target=neighbor)
                    all_paths.append(path)

            # From box2 to neighbors of box1 (if edge is not 'quantum')
            for neighbor in self.pathfinding_G.neighbors(box1):
                if self.G.edges[box1, neighbor]['label'] != 'quantum':
                    path = nx.shortest_path(self.pathfinding_G, source=box2, target=neighbor)
                    all_paths.append(path)

            # EPR pair paths from fball1 and fball2 to EPR balls on same QPU
            frontier_fballs = [fball1, fball2]
            for fball in frontier_fballs:
                fbox = self.qm.get_box(fball)
                for epr_pair in self.qm.EPR_pairs.values():
                    for epr_ball in epr_pair:
                        epr_box = self.qm.get_box(epr_ball)
                        # Check if on same QPU
                        if (fbox < self.qubit_amount / 2 and epr_box < self.qubit_amount / 2) or (fbox >= self.qubit_amount / 2 and epr_box >= self.qubit_amount / 2):
                            path = nx.shortest_path(self.pathfinding_G, source=fbox, target=epr_box)
                            all_paths.append(path)

        for ball in range(Constants.NUMQ, self.qubit_amount):
            # Skip if ball is part of any EPR pair
            if any(ball in pair for pair in self.qm.EPR_pairs.values()):
                continue

            box = self.qm.get_box(ball)
            target_box = 0 if box < self.qubit_amount / 2 else int(self.qubit_amount / 2)

            path = nx.shortest_path(self.pathfinding_G, source=box, target=target_box)
            all_paths.append(path)

        def is_prefix_of_any_path(sub, list_of_paths):
            sub_len = len(sub)
            return any(path[:sub_len] == sub for path in list_of_paths)

        # Find all qubit pair distances
        for ball1, ball2 in self.pairs:
            # Find the boxes corresponding to ball1 and ball2
            box1 = self.qm.get_box(ball1)
            box2 = self.qm.get_box(ball2)
            if not nx.has_path(self.pathfinding_G, source=box1, target=box2):
                raise ValueError("Boxes are no longer connected")
            # Calculate the shortest path
            try:
                shortest_path = nx.shortest_path(self.pathfinding_G, source=box1, target=box2, weight='weight')
                path_length = nx.path_weight(self.pathfinding_G, shortest_path, weight='weight')    # should only consider edge weight so cooldowns dont matter
                # make path between EPR pairs halves inf since they cannot swap with eachother
                if (self.qm.get_ball(shortest_path[0]), self.qm.get_ball(shortest_path[1])) in self.qm.EPR_pairs.values() or (self.qm.get_ball(shortest_path[1]), self.qm.get_ball(shortest_path[0])) in self.qm.EPR_pairs.values():
                    path_length = float('inf')
                # make sure EPR pair halves are not teleported
                elif path_length % 3 != 0 and self.qm.ball_in_epr_pairs(ball1)[0]:
                    path_length = float('inf')
            except nx.NetworkXNoPath:
                # In case there is no path between the two nodes
                path_length = float('inf')

            if path_length == float('inf'):
                path_length = 0
            else:
                path_length = 1

            if is_prefix_of_any_path(shortest_path, all_paths):
                path_length *= 2

            # check if moving pair leaves reserved qubits on qpu
            if box1 < self.qubit_amount/2 and box2 >= self.qubit_amount/2 and reserved_qubits_per_qpu[1] < 2:
                path_length = 0
            if box1 >= self.qubit_amount/2 and box2 < self.qubit_amount/2 and reserved_qubits_per_qpu[0] < 2:
                path_length = 0

            # make sure only prefered actions are valid
            if path_length > 1:
                mask.append(1)
            else:
                mask.append(0)

        state_vector = self.get_qubit_location_vector()

        return state_vector, mask
    

    #!update distnces between all the logical qubit pairs
    def perform_action(self, action, link):
        performed_score = False #make it true only when you indeed performed a score
        if action == "GENERATE":
            self.generate(link)
        elif action == "SWAP":
            self.swap(link)
        elif action == "SCORE":
            performed_score = self.score(link)
            print('*************************WE SCORE!!************************')
            print("state is: ", self.get_qubit_location_vector())
            self.update_frontier()
        elif action == "tele-gate":
            performed_score = self.tele_gate(link)
            print('*************************WE TELEGATE!!************************')
            print("state is: ", self.get_qubit_location_vector())
            self.update_frontier()
        elif action == "tele-qubit":
            self.tele_qubit(link)
        elif action == "stop":
            self.stop()
        else:
            raise ValueError(f"Unknown action: {action}")
        return performed_score
    
    #!performs the generate action
    def generate(self, link):
        # Check if the link is a quantum link
        if self.G.edges[link]['label'] != "quantum":
            raise ValueError("GENERATE can only be performed on quantum links.")
        # Check if boxes have reserved EPR pair qubit mapped to them
        if (self.qm.get_ball(link[0]) < self.my_DAG.numQubits or self.qm.get_ball(link[1]) < self.my_DAG.numQubits):
            raise ValueError("GENERATE can only be performed on empty link qubits.")
        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_GENERATE
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_GENERATE
        if random.random() < Constants.ENTANGLEMENT_PROBABILITY:  
            self.qm.generate_EPR_pair(*link)
        self.EPR_amount += 1

        print("Generated EPR pair")


    #!performs the swap action
    def swap(self, link):
        if self.G.edges[link]['label'] == "quantum":
            for edge in self.pathfinding_G.edges:
                node1, node2 = edge
                print("Edge ", edge, "has weight", self.pathfinding_G.edges[node1, node2]['weight'])
            raise ValueError("SWAP cannot be performed on quantum links.")  
        
        max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
        self.virtual_swap(link)
        self.swap_amount += 1

        self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_SWAP
        self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_SWAP


    #!performs the virtual swap action
    def virtual_swap(self, link):
        box1, box2 = link

        ball1 = self.qm.get_ball(box1)
        ball2 = self.qm.get_ball(box2)

        # Swap only if the ball exists
        if ball1 is not None:
            self.qm.box_to_ball[box2] = ball1
            self.qm.ball_to_box[ball1] = box2
        else:
            self.qm.box_to_ball.pop(box2, None)  # Remove mapping if no ball

        if ball2 is not None:
            self.qm.box_to_ball[box1] = ball2
            self.qm.ball_to_box[ball2] = box1
        else:
            self.qm.box_to_ball.pop(box1, None)  # Remove mapping if no ball


    #!performs the score action
    def score(self, link):
        max_cd = 0
        performed_score = False # Have scored
        if self.G.edges[link]['label'] == "quantum":
            raise ValueError("SCORE cannot be performed on quantum links.")
        for (ball1, ball2, _) in self.frontier:
            if (ball1, ball2) in [(self.qm.get_ball(link[0]), self.qm.get_ball(link[1])), (self.qm.get_ball(link[1]), self.qm.get_ball(link[0]))]:
                #self.topo_order.remove((ball1, ball2, _))
                self.my_DAG.remove_node((ball1, ball2)) #it will understand to remove the fist layer that appears 
                max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
                self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE
                self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE # since scores happen automatically
                performed_score = True
                print("-----we scored", ball1, ball2,"----------")
                self.DAG_left -= 1
                return performed_score
        if (not performed_score):
            raise ValueError("could not score.")


    #!performs the stop action
    def stop(self):
        self.prev_mask = self.mask
        self.min_cd = 10000

        while self.prev_mask == self.mask and self.min_cd != 0:
            self.min_cd = 10000  # Reset min_cd each loop

            for node in self.G.nodes:
                weight = self.G.nodes[node]['weight']
                if weight > 0 and weight < self.min_cd:
                    self.min_cd = weight

            if self.min_cd == 10000:  # No weights > 0, meaning all cooldowns are 0
                self.min_cd = 0
                break

            for _ in range(self.min_cd):
                self.reduce_cooldowns()

            self.state, self.mask = self.update_state_vector()


    #!performs the tele_gate action
    # in the tele_gate action, note that the link referes to a "virtual" link between EPR pairs
    # gets as input the positions of the EPR pair and scores using any pair of neighbors (if possible)
    def tele_gate(self, link):
        flag = False # Have performed tele-gate
        box1, box2 = link
        max_cd = 0
        ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)
        in_epr1, epr_id1 = self.qm.ball_in_epr_pairs(ball1)
        in_epr2, epr_id2 = self.qm.ball_in_epr_pairs(ball2)

        # check if balls are part of the same EPR pair
        if not (in_epr1 and in_epr2 and epr_id1==epr_id2):
            raise ValueError("tele-gate can only happen between EPR pairs.")

        neighbors_ball1 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1))
        neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2))

        for (ball1_frontier, ball2_frontier, _) in self.frontier: 
            if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                # remove node from DAG
                self.my_DAG.remove_node((ball1_frontier, ball2_frontier)) #it will understand to remove the fist layer that appears 
                # destroy EPR pair used
                self.qm.destroy_EPR_pair(epr_id1)
                # check if any qubits are on cooldown, telegate is performed after this so cooldown of the telegate becomes max cd of qubits + telegate cd
                max_cd = max(self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'], self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'], self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
                self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE   # since scores are automatic
                self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                print("-------we telegate", ball1_frontier, ball2_frontier,"----------")
                self.DAG_left -= 1
                flag = True
                return flag
        if (not flag):
            raise ValueError("tele-gate could not be performed.")
        return flag            

    
    #!performs the tele_qubit action
    #needs a link between an EPR particle and a non EPR particle. It teleports the nonEPR qubit to the position that the other half EPR is.
    def tele_qubit(self, link):
        box1, box2 = link
        in_epr1, epr_id1 = self.qm.ball_in_epr_pairs(self.qm.get_ball(box1))
        in_epr2, epr_id2 = self.qm.ball_in_epr_pairs(self.qm.get_ball(box2))
        if not in_epr1 and not in_epr2:
            raise ValueError("tele-qubit needs a half of EPR pair.")
        if in_epr1 and not in_epr2:
            print("------Telequbit performed in", self.qm.get_ball(box2), "using", self.qm.get_ball(box1), "------------------")
            EPR_ball1, EPR_ball2 = self.qm.EPR_pairs[epr_id1]
            other_box = self.qm.get_box(EPR_ball1) if self.qm.get_ball(box1) == EPR_ball2 else self.qm.get_box(EPR_ball2) #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            self.qm.destroy_EPR_pair(epr_id1)
            self.virtual_swap((box2, other_box))    # box2 contains the qubit and other box contains the box of the EPR half
            max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'], self.G.nodes[other_box]['weight'])
            self.G.nodes[other_box]['weight'] = max_cd + Constants.COOLDOWN_TELE_QUBIT
        elif not in_epr1 and in_epr2:
            print("------Telequbit performed in", self.qm.get_ball(box1), "using", self.qm.get_ball(box2), "------------------")
            EPR_ball1, EPR_ball2 = self.qm.EPR_pairs[epr_id2]
            other_box = self.qm.get_box(EPR_ball1) if self.qm.get_ball(box2) == EPR_ball2 else self.qm.get_box(EPR_ball2) #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            self.qm.destroy_EPR_pair(epr_id2)
            self.virtual_swap((box1, other_box))    # box1 contains the qubit and other box contains the box of the EPR half
            max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'], self.G.nodes[other_box]['weight'])
            self.G.nodes[other_box]['weight'] = max_cd + Constants.COOLDOWN_TELE_QUBIT
        self.telequbit_amount += 1
        self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_TELE_QUBIT
        self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_TELE_QUBIT


    def update_frontier(self):
        # nodes with no incoming edges
        nodes_no_predecessors = set(self.my_DAG.DAG.nodes()) - {node for _, adj_list in self.my_DAG.DAG.adjacency() for node in adj_list.keys()}
        # update frontier
        self.frontier = nodes_no_predecessors   


    #!decodes the action number to the actual action or set of actions
    def decode_action_fromNum(self, action_num):
        # Check for stop or generate actions
        if action_num == 0:
            # stop action
            return [{'edge': [], 'action': 'stop'}]
        elif action_num < self.num_entanglement_links+1:
            generate_actions = []
            for edge in self.G.edges(data=True):  
                if edge[2]['label'] == "quantum":  # Check if the label is quantum
                    generate_actions.append({'edge': edge[:2], 'action': 'GENERATE'})
            return [generate_actions[action_num-1]]
        
        # Find the balls corresponding to the action
        ball1, ball2 = self.pairs[action_num-(1+self.num_entanglement_links)]

        #print("Moving balls:", (ball1, ball2))

        # Find the boxes corresponding to ball1 and ball2
        box1 = self.qm.get_box(ball1)
        box2 = self.qm.get_box(ball2)
        #print("Swapping pair", ball1, ball2, "located at: ", box1, box2)

        # Calculate the shortest path between the boxes
        shortest_path = nx.shortest_path(self.pathfinding_G, source=box1, target=box2, weight='weight')

        action_list = []
        ignore_next = False
        # Generate action list for swapping qubits
        for i in range(len(shortest_path)-1):
            # find nodes we are swapping and potential next-next box for EPR pair checking
            box1 = shortest_path[i]
            box2 = shortest_path[i + 1] if i + 1 < len(shortest_path) else None
            box2_next = shortest_path[i + 2] if i + 2 < len(shortest_path) else None

            # check if ball in box2 belongs to EPR pair, if so check if next swap is between a EPR pair if so add teleport qubit action
            in_epr1, epr_id1 = self.qm.ball_in_epr_pairs(self.qm.get_ball(box2))
            in_epr2, epr_id2 = self.qm.ball_in_epr_pairs(self.qm.get_ball(box2_next))
            if in_epr1 and in_epr2 and epr_id1 == epr_id2:
                action_list.append({'edge': ( box1, box2), 'action': 'tele-qubit'})
                ignore_next = True # dont swap the next qubit as that is already done by tele-qubit
            elif not ignore_next:
                action_list.append({'edge': ( box1, box2), 'action': 'SWAP'})
            else:
                ignore_next = False # after ignore swaps can occur again

        return action_list


    #!step the emulator given a specific action
    def step_given_action(self, action_num):

        matching_scores = [] # here we will store the edges that were picked with the autocomplete method of scoring (scores and tele-gates automatically done after an action)
        reward = 0

        # print("Mask is :", self.mask)
        # print("Mapping is :", self.state[:self.qubit_amount])
        # print("Frontier is:", self.frontier)
        # Find all actions corresponding to swapping qubit pair
        taken_actions = self.decode_action_fromNum(action_num)
        #print("State is :", self.state, " from mapping ", self.get_qubit_location_vector())
        #print("Got actions ", taken_actions, " from action number ", action_num)

        # Execute all actions in action list
        for taken_action in taken_actions:     
            self.perform_action(taken_action['action'], taken_action['edge'])  #make action and change self (state)

        # Fill any scores or tele-gates that can happen immediately after the action of this time slot
        matching_scores = []  #which links were triggered for scores and telegate
        matching_scores,cur_reward = self.fill_matching(matching_scores)   ## Here we auto fill with the scores and tele-gates! The possible scores and tele-gate actually are implemented here automatically!
        reward += cur_reward                   
        
        #self.distance_metric = self.calculate_distance_metric() # this metric decides the moving reward - what actions did make the qubits that should come together closer?
        dif_score = 0
        if (reward == 0 and action_num != 0): #it did not score and it is not stop
            dif_score = self.distance_metric_prev - self.distance_metric
            reward = dif_score * Constants.DISTANCE_MULT
        elif (action_num == 0): #we did stop
            reward = Constants.REWARD_STOP
        self.distance_metric_prev = self.distance_metric #the previous for the next one
        
        flagSuccess = False
        if len(self.my_DAG.DAG.nodes) == 0 : 
            reward = Constants.REWARD_EMPTY_DAG
            flagSuccess = True
        #return reward, self, nx.is_empty(self.my_DAG.DAG)
        #print(reward)
        return reward, flagSuccess
        

    #!function checks for possible score and tele-gates which can now happen
    #It provides a matching with the possible scores and tele-gates that can happen according to the state.
    def fill_matching(self,matching):
        cur_reward = 0
        # Make a copy of current links in the system excluding those labeled "quantum"
        all_links = [link for link in self.G.edges() if self.G.edges[link].get('label') != 'quantum']

        # Iterate over all links for SCORE action - can be done more efficiently by checking the frontier
        for link in all_links:
            box1, box2 = link
            ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)

            # Check if the SCORE action can be performed on this link
            # !!we do not check whether action is possible any more we just increase the cd
            for (ball1_frontier, ball2_frontier, _) in self.frontier:
                if (ball1_frontier, ball2_frontier) in [(ball1, ball2), (ball2,ball1)]:
                    # Perform the SCORE action and append link to matching
                    self.perform_action('SCORE', link)      
                    cur_reward += Constants.REWARD_SCORE
                    matching.append(link)

        # Separate loop to iterate over EPR pairs for tele-gate action
        for ball in list(self.qm.EPR_pairs.keys()):
            link = self.qm.query_EPR_pair(ball)   # get the boxes that contain the EPR pair
            box1, box2 = link
            neighbors_ball1 = set( self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1) if  self.G.edges[(box1, neighbor)].get('label') != 'quantum')
            neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2) if  self.G.edges[(box2, neighbor)].get('label') != 'quantum') #changed here!!!

            # Iterate over the frontier
            for (ball1_frontier, ball2_frontier, _) in self.frontier:
                # Check if the frontier balls are neighbors to the boxes and if action is possible
                # do not check whether the action is possible between the boxes of the frontier's balls
                if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or
                (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                    # Perform the tele-gate action and append link to matching
                    self.perform_action('tele-gate', link) 
                    cur_reward += Constants.REWARD_SCORE
                    matching.append(link)
                    break
        return matching,cur_reward
    

    #!determines the distance between two logical qubits
    def calculate_distance_between_balls(self, ball1, ball2, temp_G):
        # Find the boxes corresponding to ball1 and ball2
        box1 = self.qm.get_box(ball1)
        box2 = self.qm.get_box(ball2)
        epr_links_used = []
        # Calculate the shortest path
        try:
            shortest_path = nx.shortest_path(temp_G, source=box1, target=box2, weight='weight')
            path_length = nx.path_weight(temp_G, shortest_path, weight='weight')
            # Check for EPR links in the path
            for i in range(len(shortest_path) - 1):
                if temp_G.edges[( shortest_path[i], shortest_path[i+1]) ]['virtual'] == True:
                    epr_links_used.append((shortest_path[i], shortest_path[i+1]))
        except nx.NetworkXNoPath:
            # In case there is no path between the two nodes
            path_length = float('inf')

        return path_length, epr_links_used


   #!calculates the distance metric required for determining the reward
    def calculate_distance_metric(self):
        distance_metric = 0  # Reset the distance metric
        # Create a temporary graph for distance calculation
        temp_G = self.G.copy()

        for edge in temp_G.edges():
            temp_G.edges[edge]['weight'] = 1 # every link will count as distance 1
            temp_G.edges[edge]['virtual'] =  False
            if (temp_G.edges[edge]['label'] == "quantum"): 
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_QUANTUM_LINK  # make it harder to traverse quantum links (they require EPR pair generation conseptually)
        

        # Add links for every EPR pair
        for epr_id, (box1, box2) in self.qm.EPR_pairs.items():
            edge = (box1,box2)
            # Add a "virtual" link
            if (box1,box2) not in self.G.edges:
                temp_G.add_edge(box1, box2, weight = Constants.DISTANCE_BETWEEN_EPR, label="virtual", virtual=True)
            elif (temp_G.edges[edge]['label'] == "quantum"):
                temp_G.edges[edge]['virtual'] = True
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_BETWEEN_EPR # from quantum reduce it temporarily to 1 since we have an entanglement there, remember to increase again when this entanglement is used
            
                

        # Iterate over the frontier to calculate distances
        for (ball1, ball2, _) in self.frontier:
            distance, epr_links_used = self.calculate_distance_between_balls(ball1, ball2, temp_G)
            distance_metric += distance
            # Remove used EPR links from temp_G
            for link in epr_links_used:
                
                if link not in self.G.edges:
                    temp_G.remove_edge(*link)
                elif (temp_G.edges[link]['label'] == "quantum"):
                    temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK # previous entanglement is used so get it back
                    temp_G.edges[link]['virtual'] = False

        return distance_metric


    #!convert from the actual state to a state vector as the RL agent wants it
    def get_qubit_location_vector(self):
        # Initialize a list with None (or a placeholder) for each possible index
        my_list = [-1] * (self.qm.numNodes) 
        # Populate the list using dictionary keys as indices
        for key, value in self.qm.box_to_ball.items():
            # Instead of EPR-x we now need just the number (i.e., numNodes+x as an index)
            # Check ball is an epr pair
            if self.qm.ball_in_epr_pairs(value)[0]:
                epr_id = self.qm.ball_in_epr_pairs(value)[1]
                prefix, num_str = epr_id.split('-')
                value = int(num_str) + self.qubit_amount
            my_list[key] = value
        single_numbers_topo_list = [element for tup in self.my_DAG.topo_order for element in tup]  #break (x,y,z) tuple inside topo_order to x,y,z (x,y qubits and z the layer)
        #the above is needed for breaking into the state space vector
        state_vector = my_list + single_numbers_topo_list
        N = self.qm.numNodes + 3*self.my_DAG.numGates # N is the size of a correct state vector
        if len(state_vector) < N:
            #print("test")
            state_vector.extend([-2] * (N - len(state_vector)))

        return state_vector


    #network update each time (each_time_step)
    def RL_step(self, action_num):   #action will be a single non-negetive integer in range [0, action_size_val]
    
        reward = 0
        
        reward, successfulDone = self.step_given_action(action_num)
        self.action_amount += 1
        self.state, self.mask = self.update_state_vector()
        new_state, new_mask = np.array(self.state), np.array(self.mask)
        
        if successfulDone:
            print("When game won, state is: ", self.state)
            print("DAG nodes after removal", self.my_DAG.DAG.nodes)
            print("#################SOLVED!#####################################")

        return reward, new_state, new_mask, successfulDone
    

    
















        
        


# Function to handle swapping of EPR pairs and normal balls
# PROBABLY OBSOLETE JUST HERE IN CASE I NEED IT
    def handle_virtual_swap(self, box_from, box_to, ball):
        # Check if ball is part of EPR pairs
        if ball in self.qm.EPR_pairs:
            self.qm.ball_to_box[ball].remove(box_from)
            self.qm.ball_to_box[ball].append(box_to)
            temp_boxes = self.qm.ball_to_box[ball]              
            self.qm.EPR_pairs[ball] = temp_boxes       #update the EPR pairs as well
        else:
            self.qm.ball_to_box[ball] = box_to






