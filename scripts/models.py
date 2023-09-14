# Mixture of experts model modules
import torch
import torch.nn as nn

ACTIVATIONS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'prelu': nn.PReLU(),
    'leakyrelu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid()
}

# expert network with simple 2-layer MLP
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dout=0.2):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dout)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.act = ACTIVATIONS[activation]
    
    def forward(self, x):
        return self.bn2(self.fc2(self.dropout(self.act(self.bn1(self.fc1(x))))))
        # return self.bn2(self.fc2(self.act(self.bn1(self.fc1(x)))))
        # return self.fc2(self.act(self.fc1(x)))


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes, activation, dropout=0.2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # Create expert modules
        self.experts = nn.ModuleList([Expert(input_dim=input_dim, hidden_dim=hidden_dim, output_dim = num_classes, activation=activation, dout=dropout) for _ in range(num_experts)])

        # Create gating module
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # # Calculate gating weights
        # gating_weights = self.gating(x)
        # print(f"gating_weights: {gating_weights}")
        # print(f"gating_weights shape: {gating_weights.shape}")
        # # Calculate expert outputs and apply gating
        # expert_outputs = []
        # for expert in self.experts:
        #     expert_output = expert(x)
        #     expert_outputs.append(expert_output.unsqueeze(2))  # Add a dimension
        # expert_outputs = torch.cat(expert_outputs, dim=2)  # Concatenate experts along the added dimension
        # print(f"expert_outputs: {expert_outputs}")
        # print(f"expert_outputs shape: {expert_outputs.shape}")
        # # Combine expert outputs using gating weights
        # mixture_output = torch.matmul(expert_outputs, gating_weights.unsqueeze(1)).squeeze()

        # calculating gating weights
        gating_weights = self.gating(x)
        # calculating expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # print expert_outputs.shape
        # print(f"expert_outputs_shape: {expert_outputs.shape}")
        # print gating_weights.shape
        # print(f"gating_weights_shape: {gating_weights.shape}")

        # calculating mixture output
        mixture_output = torch.sum(gating_weights.view(-1, self.num_experts, 1) * expert_outputs, dim=1)
        # print(f"mixture_output_shape: {mixture_output.shape}")    
        return mixture_output, gating_weights, expert_outputs



# # gating network for expert selection
# class GatingNetwork(nn.Module):
#     def __init__(self, input_dim, num_experts, num_classes):
#         super(GatingNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, num_experts * num_classes)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         return self.softmax(self.fc1(x))
    
# # mixture of experts model
# class MoE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_experts, num_classes):
#         super(MoE, self).__init__()
#         self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, num_classes) for i in range(num_experts)])
#         self.gating_network = GatingNetwork(input_dim, num_experts, num_classes)
    
#     def forward(self, x):
#         expert_ouputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         gate_weights = self.gating_network(x)
#         mixture_output = torch.sum(gate_weights.view(-1, len(self.experts), 1) * expert_ouputs, dim=1)
#         return mixture_output, gate_weights
    
