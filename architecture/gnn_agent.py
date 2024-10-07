import torch
import torch.nn as nn
from torch.nn import Sequential
from torch_geometric.nn import GENConv
from torch_geometric.data import Data
import pickle
from pathlib import Path

base_path = Path(__file__).parent

class ProcessorBlock(nn.Module):
    
        def __init__(self):
    
            super(ProcessorBlock, self).__init__()
    
            self.mp = GENConv(64, 64, norm='layer', aggr='softmax', num_layers=2)
            self.norm = nn.Identity()
    
        def forward(self, x, edge_index, edge_attr):

            return self.norm(self.mp(x, edge_index, edge_attr))

class AgentGraphPolicyNetwork(nn.Module):

    def __init__(self, model_file='naive/naive1.pkl'):

        super(AgentGraphPolicyNetwork, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.node_encoder = Sequential(
            nn.Linear(11, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.edge_encoder = Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.mp_layers = nn.ModuleList(
            [
                ProcessorBlock(),
                ProcessorBlock(),
                ProcessorBlock()
            ]
        )

        self.pi_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

        checkpoint = pickle.load(open(f'{base_path}/../trained_agents/{model_file}', 'rb'))

        for i in range(0, 6, 2):
            self.node_encoder[i].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.node_encoder.mlp.{i}.weight'], dtype=torch.float32))
            self.node_encoder[i].bias   = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.node_encoder.mlp.{i}.bias'], dtype=torch.float32))

            self.edge_encoder[i].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.edge_encoder.mlp.{i}.weight'], dtype=torch.float32))
            self.edge_encoder[i].bias   = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.edge_encoder.mlp.{i}.bias'], dtype=torch.float32))

        for i in range(3):
            self.mp_layers[i].mp.mlp[0].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.processor.mp_layers.{i}.conv.mlp.0.weight'], dtype=torch.float32))
            self.mp_layers[i].mp.mlp[1].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.processor.mp_layers.{i}.conv.mlp.1.weight'], dtype=torch.float32))
            self.mp_layers[i].mp.mlp[1].bias   = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.processor.mp_layers.{i}.conv.mlp.1.bias'], dtype=torch.float32))
            self.mp_layers[i].mp.mlp[4].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'encoder.actor_encoder.processor.mp_layers.{i}.conv.mlp.4.weight'], dtype=torch.float32))

        for i in range(0, 6, 2):
            self.pi_head[i].weight = torch.nn.Parameter(torch.tensor(checkpoint[f'pi.pi_net.mlp.{i}.weight'], dtype=torch.float32))
            self.pi_head[i].bias   = torch.nn.Parameter(torch.tensor(checkpoint[f'pi.pi_net.mlp.{i}.bias'], dtype=torch.float32))

        self.node_encoder.eval()
        self.edge_encoder.eval()
        self.mp_layers.eval()
        self.pi_head.eval()

        self.node_encoder.to(self.device)
        self.edge_encoder.to(self.device)
        self.mp_layers.to(self.device)
        self.pi_head.to(self.device)

    def forward(self, inputs):
        
        # Convert inputs to tensors
        node_features = torch.tensor(inputs['node_features']).to(self.device)
        edge_features = torch.tensor(inputs['edge_features']).to(self.device)

        # Infer number of turbines
        n_turbines = node_features.shape[0]

        # Remove first dimension
        edge_index = torch.tensor(inputs['edge_links']).long().to(self.device)

        # Encode node and edge features
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)

        # Create graph of nodes and edges
        graph = Data(x=nodes, edge_index=edge_index, edge_attr=edges)

        # Apply message passing layers
        for layer in self.mp_layers:
            graph.x = layer(graph.x, graph.edge_index, graph.edge_attr)

        # Apply policy head
        output = self.pi_head(graph.x)

        # Reshape output, taking care of batch dimension
        if output.dim() == 2:
            output = output.t().reshape(-1)
        elif output.dim() == 3:
            output = output.transpose(1, 2).reshape(output.size(0), -1)

        # Get the means of the normal distributions; this is the first half of the output
        # (Second half is the standard deviations, but we don't use exploration in inference)
        means_norm = output[:n_turbines]

        # Rescale the means to the correct range
        means = -30 + (means_norm + 1) * 60 / 2

        return means.detach().cpu().numpy().reshape((n_turbines,))