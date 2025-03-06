
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy





class QubitMappingClass():
    
    def __init__(self, numNodes, numQubits, numEPR_threshold, G, initial_mapping=None):
        self.numNodes = numNodes
        self.numQubits = numQubits
        self.numEPR_threshold = numEPR_threshold

        self.ball_to_box = {}  # ball to box mapping
        self.box_to_ball = {}  # box to ball mapping
        self.EPR_pairs = {}  # EPR pairs mapping
        self.EPR_pool = [f"EPR-{i}" for i in range(self.numEPR_threshold)]  # Pool of EPR IDs

        if initial_mapping is None:
            initial_mapping = self.generate_random_initial_mapping(G)

        # Initialize with given mapping
        if initial_mapping is not None:
            for ball, box in initial_mapping.items():
                if (ball > numQubits+(2*self.numEPR_threshold)-1 or box > numNodes - 1):
                    raise Exception("Ball or box out of limit.")
                self.ball_to_box[ball] = box
                self.box_to_ball[box] = ball
        else:   
            raise Exception("Error - initial mapping is None")


    def get_box(self, ball):
        if ball not in self.ball_to_box:
            raise Exception(f"No box found for ball {ball}.")
        return self.ball_to_box[ball]
    

    def get_ball(self, box):
        if box not in self.box_to_ball:
            return self.box_to_ball.get(box, None)
        return self.box_to_ball[box]
    

    def generate_EPR_pair(self, box1, box2):
        if len(self.EPR_pool) == 0:
            raise Exception("No more EPR IDs available in the pool.")
        
        if (box1 > self.numNodes - 1 or box2 > self.numNodes - 1):
            raise Exception("Ball or box out of limit.")
        
        epr_id = self.EPR_pool.pop(0)  # Get the first available ID and remove it from the pool

        # Update the mappings
        self.EPR_pairs[epr_id] = (self.get_ball(box1), self.get_ball(box2))


    def destroy_EPR_pair(self, epr_id):
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")

        # Remove the pair from the mappings
        self.EPR_pairs.pop(epr_id)
        
        # Return the ID to the EPR pool
        self.EPR_pool.append(epr_id)
        self.EPR_pool.sort()  # Keep the pool sorted for predictability


    def query_EPR_pair(self, epr_id):
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")
        # Return the boxes associated with the EPR pair
        # Fetch the current boxes associated with the EPR pair from ball_to_box mapping
        balls = self.EPR_pairs[epr_id]
        boxes = (self.get_box(balls[0]), self.get_box(balls[1]))
        # Return the updated boxes
        return boxes
    
    def ball_in_epr_pairs(self, ball):
        for epr_id in self.EPR_pairs:
            (ball1, ball2)  = self.EPR_pairs[epr_id]
            if ball == ball1 or ball == ball2:
                return True, epr_id  # Ball is part of an EPR pair
        return False, None  # Ball is not part of any EPR pair
    
    
    #!TODO: test this
    #!Generates a random inital mapping where each logical qubit neighbors at least one other logical qubit
    def generate_random_initial_mapping(self, G):
        if self.numQubits > self.numNodes:
            raise ValueError("Number of logical qubits cannot be greater than the number of physical qubits.")

        initial_mapping = {}
        used_physical = set()

        # Assign fixed mappings
        initial_mapping[self.numQubits] = 0
        initial_mapping[self.numQubits + 1] = 9
        used_physical.update({0, 9})

        remaining_logical = [i for i in range(self.numQubits+2*self.numEPR_threshold) if i not in initial_mapping]
        physical_qubits = list(range(self.numNodes))
        random.shuffle(remaining_logical)  # Shuffle to introduce randomness

        for logical in remaining_logical:
            # Find all valid neighbors of already assigned physical qubits
            valid_choices = [q for q in physical_qubits if q not in used_physical and any(n in used_physical for n in G.neighbors(q))]
            
            if not valid_choices:
                raise ValueError("No available physical qubits left that are connected to the existing mapping.")
            
            # Assign logical qubit to any random valid neighbor
            chosen_physical = random.choice(valid_choices)
            initial_mapping[logical] = chosen_physical
            used_physical.add(chosen_physical)

        return initial_mapping
    