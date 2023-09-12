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


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gating, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        gating_weights = self.softmax(x)
        return gating_weights

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # Create expert modules
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, num_classes) for _ in range(num_experts)])

        # Create gating module
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Calculate gating weights
        gating_weights = self.gating(x)

        # Calculate expert outputs and apply gating
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)
            expert_outputs.append(expert_output.unsqueeze(2))  # Add a dimension
        expert_outputs = torch.cat(expert_outputs, dim=2)  # Concatenate experts along the added dimension

        # Combine expert outputs using gating weights
        mixture_output = torch.matmul(expert_outputs, gating_weights.unsqueeze(1)).squeeze()

        return mixture_output, gating_weights



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
    
