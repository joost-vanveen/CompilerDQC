import torch
from torch import nn
import math


#learning agent NNs used by the simulator for distributed quantum computer simulator defined here

class ComboLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, input_dim = x.size()
        assert input_dim >= 3, "Input must have at least 3 dimensions"

        # Step 1: Pass through first 2 values
        first_two = x[:, :2]  # shape [batch_size, 2]

        # Step 2: Remaining values
        rest = x[:, 2:]  # shape [batch_size, rest_dim]
        rest_dim = rest.size(1)

        # Create all (i, j) index pairs where i != j
        idx = torch.arange(rest_dim)
        i_idx, j_idx = torch.meshgrid(idx, idx, indexing="ij")
        mask = i_idx != j_idx
        i_idx = i_idx[mask].flatten()
        j_idx = j_idx[mask].flatten()

        # Gather xi and xj
        xi = rest[:, i_idx]  # shape [batch_size, num_combos]
        xj = rest[:, j_idx]  # shape [batch_size, num_combos]

        # Combine with weights
        combined = 0.75 * xi + 0.25 * xj  # shape [batch_size, num_combos]

        # Final output
        output = torch.cat([first_two, combined], dim=1)  # shape [batch_size, 2 + num_combos]
        return output


class customizedNN(nn.Module):  
    
    def __init__(self, input_dim, output_dim, hidden_layers, device):
        super(customizedNN,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        self.device = device
        
        
        print("####################################")
        print("Device for NN: ", self.device)
        print("####################################")
        
        self.layers = nn.ModuleList()
        
        
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())

        self.combo_layer = ComboLayer()
        
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.1)
        
    def forward(self,x, mask):
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.combo_layer(x)      # Apply custom logic
        x = x * mask                 # Apply mask         
        return x
    









class customizedNN_policyGrad(nn.Module):  
    
    def __init__(self, input_dim, output_dim, hidden_layers, device):
        super(customizedNN_policyGrad,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        self.device = device
        
        
        print("####################################")
        print("Device for NN in policy grad: ", self.device)
        print("####################################")
        
        self.layers = nn.ModuleList()
        
        
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
        
        self.softmax = nn.Softmax()
        #self.dropout = nn.Dropout(0.1)
        
    def forward(self,x, mask):
        
        for layer in self.layers:
            x = layer(x)
            
        #x = x * mask   
        x = self.softmax(x)   
        if torch.count_nonzero(x) == 0:
            #actor_output[0,0] = 1
            print("minor force fix, action will be 0/stop since NN generated all-zeros")    
        return x