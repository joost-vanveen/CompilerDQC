
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy
import json
import os
from Constants import Constants





class DAGClass():
    
    def __init__(self, save_dag, dag_list=None):
        # Create a Directed Graph
        if dag_list is None:
            self.DAG, dag_list = self.create_random_DAG(Constants.NUMQ, Constants.NUMG)
            if save_dag:
                self.save_dag_to_file(dag_list, path="saved_dag.json")
        else:
            self.DAG = self.create_DAG(dag_list)
        print(self.DAG)
        self.topo_order = self.compute_topo_order()
        self.numGates = len(self.topo_order)  #initial number of gates
        self.layers = self.compute_node_layers()
        #self.numQubits = self.compute_numQubits()
        self.numQubits = Constants.NUMQ
            

    def generate_cnot_dag(self, numQ, numG):
        qubit_layers = {qubit: 0 for qubit in range(numQ)}
        dag = []
        for _ in range(numG):
            x, y = random.sample(range(numQ), 2)
            least_layer = max(qubit_layers[x], qubit_layers[y]) + 1
            qubit_layers[x], qubit_layers[y] = least_layer, least_layer
            dag.append((x, y, least_layer-1))
        dag.sort(key=lambda node: node[2])
        return dag


    def create_random_DAG(self,numQ, numG):
        dag_list = self.generate_cnot_dag(numQ, numG)
        DAG = nx.DiGraph()
        # Keeps track of the most recent node for each qubit
        qubit_most_recent_node = {}
        
        for x, y, l in dag_list:  # remember it is sorted
            current_node = (x, y, l)
            DAG.add_node(current_node)
            
            # Connect to the most recent node for each qubit, if it exists
            for qubit in [x, y]:
                if qubit in qubit_most_recent_node:
                    prev_node = qubit_most_recent_node[qubit]
                    DAG.add_edge(prev_node, current_node)
            
            # Update the most recent node for the involved qubits
            qubit_most_recent_node[x] = current_node
            qubit_most_recent_node[y] = current_node
        
        return DAG, dag_list


    def create_DAG(self, dag_list=None):
        DAG = nx.DiGraph()
        if dag_list == None:
            dag_list = [[7, 8, 0], [6, 5, 0], [0, 1, 0], [7, 3, 1], [2, 5, 1], [1, 4, 1], [9, 3, 2], [5, 1, 2], [3, 7, 3], [5, 0, 3], [6, 1, 3], [7, 2, 4], [3, 9, 4], [5, 0, 4], [3, 9, 5], [0, 2, 5], [6, 7, 5], [3, 1, 6], [9, 4, 6], [1, 5, 7], [4, 9, 7], [5, 2, 8], [1, 8, 8], [5, 0, 9], [9, 2, 9], [5, 4, 10], [8, 2, 10], [7, 5, 11], [2, 8, 11], [5, 1, 12]]

        # Keeps track of the most recent node for each qubit
        qubit_most_recent_node = {}
        
        for x, y, l in dag_list:  # remember it is sorted
            current_node = (x, y, l)
            DAG.add_node(current_node)
            
            # Connect to the most recent node for each qubit, if it exists
            for qubit in [x, y]:
                if qubit in qubit_most_recent_node:
                    prev_node = qubit_most_recent_node[qubit]
                    DAG.add_edge(prev_node, current_node)
            
            # Update the most recent node for the involved qubits
            qubit_most_recent_node[x] = current_node
            qubit_most_recent_node[y] = current_node

        # DAG.add_edges_from([
        #     ((0,1,0), (0,2,1))
        # ])
        return DAG
    
    
    def compute_topo_order(self):
        # Get nodes in topological order
        topo_order = list(nx.topological_sort(self.DAG))
        # print("topo_order is: ", topo_order)
        # Function to compute layer of each node for better visualization
        # Compute layers of nodes
        return topo_order
    
            
    def compute_node_layers(self):
        layers = {node: 0 for node in self.topo_order}
        for node in self.topo_order:
            for pred in self.DAG.predecessors(node):
                layers[node] = max(layers[node], layers[pred] + 1)
        return layers
        

    def remove_node(self, node): #It does not check whether it is possible to implement the gate
        ball1, ball2 = node
        # Find nodes with matching first and second elements.
        matching_nodes = [node for node in self.DAG if (node[0] == ball1 and node[1] == ball2) or (node[1] == ball1 and node[0] == ball2)]
        # Return the node with the smallest third element. We need it to remove the correct gate (the first that appears in the layering)
        node_to_remove = min(matching_nodes, key=lambda node: node[2]) if matching_nodes else None
        # Remove the node from the graph.
        self.DAG.remove_node(node_to_remove)
        # Remove the node from topo_order.
        self.topo_order.remove(node_to_remove)
        print("DAG nodes after removal", self.DAG.nodes)


    def print_DAG(self):
        # Create a dictionary of positions based on topological order and layer
        pos = {node: (i%3, self.layers[node]) for i, node in enumerate(self.topo_order)}
        # Draw the Directed Graph
        fig, ax = plt.subplots()
        nx.draw(self.DAG, pos, with_labels=True, node_color='lightblue', node_size=1500, ax=ax)
        plt.show()
        return
            

    def compute_numQubits(self):
        # Use a set to store unique numbers from the first two components of each node
        unique_numbers = set()
        # Iterate through all nodes in the DAG
        for node in self.DAG.nodes:
            # Add the first component of the node to the set
            unique_numbers.add(node[0])
            # Add the second component of the node to the set
            unique_numbers.add(node[1])
        # The number of unique qubits is the size of the set
        numQubit = len(unique_numbers)
        return numQubit
    
    def save_dag_to_file(self, dag_list, path):
        try:
            with open(path, "a") as f:
                json.dump(dag_list, f)
                f.write("\n")
            print(f"DAG appended to {path}")
        except Exception as e:
            print(f"Failed to save DAG: {e}")

        

    
    
  
    

    
















        
        









