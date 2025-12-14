import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

    def get_action_distribution(self, state):
        with torch.no_grad():
            probs = self.forward(state)
        return probs

if __name__ == "__main__":
    # Example usage for testing
    input_dim = 1536
    output_dim = 10
    model = PolicyNetwork(input_dim, output_dim)
    print(f"Policy Network initialized with input_dim={input_dim}, output_dim={output_dim}")
