# Mixture of experts model modules
import torch
import torch.nn as nn


# expert network with simple 2-layer MLP
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
# gating network for expert selection
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, num_classes):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_experts * num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc1(x))
    
# mixture of experts model
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, num_classes) for i in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts, num_classes)
    
    def forward(self, x):
        expert_ouputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_weights = self.gating_network(x)
        mixture_output = torch.sum(gate_weights.view(-1, len(self.experts), 1) * expert_ouputs, dim=1)
        return mixture_output, gate_weights
    
