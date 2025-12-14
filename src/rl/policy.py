"""
DIRECTIVE-023: Policy Network for Prompt Mutation RL

This module implements a Policy Network that learns to select
the best mutation actions for prompt optimization.

Input: Embedding of the original prompt (1536 dims for OpenAI ada-002)
Output: Probability distribution over mutation actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import json
import os


class PolicyNetwork(nn.Module):
    """
    Policy Network for selecting mutation actions.

    Architecture:
    - Input layer: prompt embedding dimension
    - Hidden layers: 2x256 with ReLU activation and dropout
    - Output layer: softmax over mutation actions
    """

    def __init__(
        self,
        input_dim: int = 1536,  # OpenAI ada-002 embedding dimension
        output_dim: int = 5,     # Number of mutation types
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super(PolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights with Xavier
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Action probabilities of shape (batch_size, output_dim)
        """
        # Handle single sample (no batch dimension)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Forward through layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc_out(x)

        # Apply softmax to get probabilities
        return F.softmax(x, dim=-1)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Select an action based on the current state.

        Args:
            state: The state tensor (prompt embedding)
            deterministic: If True, select argmax action; else sample

        Returns:
            Tuple of (action_index, action_probability, log_probability)
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(state)

            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                # Sample from the distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            action_prob = probs[0, action].item()
            log_prob = torch.log(probs[0, action] + 1e-8)

        return action, action_prob, log_prob

    def get_action_distribution(self, state: torch.Tensor) -> np.ndarray:
        """
        Get the full action probability distribution.

        Args:
            state: The state tensor (prompt embedding)

        Returns:
            Numpy array of action probabilities
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(state)
        return probs.squeeze().numpy()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states
            actions: Batch of taken actions

        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)

        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return action_log_probs, entropy

    def save(self, path: str):
        """Save model weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dim': self.hidden_dim,
            }
        }
        torch.save(state, path)
        print(f"Policy saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'PolicyNetwork':
        """Load model from checkpoint."""
        state = torch.load(path, map_location='cpu')
        config = state['config']

        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dim=config['hidden_dim']
        )
        model.load_state_dict(state['model_state_dict'])
        print(f"Policy loaded from {path}")
        return model


# Mutation type mapping
MUTATION_TYPES = [
    'try-catch-style',
    'context-reduction',
    'expansion',
    'constraint-addition',
    'task-decomposition'
]


def action_to_mutation(action_index: int) -> str:
    """Convert action index to mutation type string."""
    if 0 <= action_index < len(MUTATION_TYPES):
        return MUTATION_TYPES[action_index]
    return MUTATION_TYPES[0]


def mutation_to_action(mutation_type: str) -> int:
    """Convert mutation type string to action index."""
    try:
        return MUTATION_TYPES.index(mutation_type)
    except ValueError:
        return 0


if __name__ == "__main__":
    # Test the policy network
    print("Testing PolicyNetwork...")

    # Create model
    model = PolicyNetwork(input_dim=1536, output_dim=5)
    print(f"Created policy with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    test_state = torch.randn(1, 1536)
    probs = model.forward(test_state)
    print(f"Action probabilities: {probs.squeeze().numpy()}")

    # Test action selection
    action, prob, log_prob = model.get_action(test_state)
    print(f"Selected action: {action} ({action_to_mutation(action)}) with prob {prob:.4f}")

    # Test batch processing
    batch_states = torch.randn(8, 1536)
    batch_probs = model.forward(batch_states)
    print(f"Batch output shape: {batch_probs.shape}")

    # Test save/load
    model.save("test_policy.pth")
    loaded_model = PolicyNetwork.load("test_policy.pth")
    os.remove("test_policy.pth")

    print("All tests passed!")
