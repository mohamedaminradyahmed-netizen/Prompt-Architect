import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Outputs a single scalar value
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

    def estimate_value(self, state):
        with torch.no_grad():
            value = self.forward(state)
        return value.item()

if __name__ == "__main__":
    # Example usage for testing
    input_dim = 1536
    model = ValueNetwork(input_dim)
    print(f"Value Network initialized with input_dim={input_dim}")
