"""
DIRECTIVE-023: Value Network for Prompt Mutation RL

This module implements a Value Network that estimates the expected
cumulative reward for a given prompt state.

Input: Embedding of the prompt (1536 dims for OpenAI ada-002)
Output: Scalar value estimate (expected return)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os


class ValueNetwork(nn.Module):
    """
    Value Network for estimating state values.

    Architecture:
    - Input layer: prompt embedding dimension
    - Hidden layers: 2x256 with ReLU activation and dropout
    - Output layer: single scalar value
    """

    def __init__(
        self,
        input_dim: int = 1536,  # OpenAI ada-002 embedding dimension
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super(ValueNetwork, self).__init__()

        self.input_dim = input_dim
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

        self.fc_out = nn.Linear(hidden_dim // 2, 1)

        # Initialize weights
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
            Value estimates of shape (batch_size, 1)
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

        # Output value (no activation - can be any real number)
        value = self.fc_out(x)

        return value

    def estimate_value(self, state: torch.Tensor) -> float:
        """
        Estimate the value of a single state.

        Args:
            state: The state tensor (prompt embedding)

        Returns:
            Scalar value estimate
        """
        self.eval()
        with torch.no_grad():
            value = self.forward(state)
        return value.item()

    def estimate_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Estimate values for a batch of states.

        Args:
            states: Batch of state tensors

        Returns:
            Tensor of value estimates
        """
        return self.forward(states).squeeze(-1)

    def save(self, path: str):
        """Save model weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
            }
        }
        torch.save(state, path)
        print(f"Value network saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ValueNetwork':
        """Load model from checkpoint."""
        state = torch.load(path, map_location='cpu')
        config = state['config']

        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim']
        )
        model.load_state_dict(state['model_state_dict'])
        print(f"Value network loaded from {path}")
        return model


class GeneralizedAdvantageEstimator:
    """
    Generalized Advantage Estimation (GAE) for variance reduction.

    Computes advantages using GAE-Lambda which provides a good
    balance between bias and variance in policy gradient estimation.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize GAE.

        Args:
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter (0 = high bias, 1 = high variance)
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GAE advantages.

        Args:
            rewards: Tensor of rewards
            values: Tensor of state values
            next_values: Tensor of next state values
            dones: Tensor of done flags (1 if episode ended)

        Returns:
            Tensor of advantages
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if t < len(next_values) else 0
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        final_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute discounted returns.

        Args:
            rewards: Tensor of rewards
            dones: Tensor of done flags
            final_value: Value estimate for final state (if not done)

        Returns:
            Tensor of discounted returns
        """
        returns = torch.zeros_like(rewards)
        running_return = final_value

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        return returns


if __name__ == "__main__":
    # Test the value network
    print("Testing ValueNetwork...")

    # Create model
    model = ValueNetwork(input_dim=1536)
    print(f"Created value network with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    test_state = torch.randn(1, 1536)
    value = model.forward(test_state)
    print(f"Value estimate: {value.item():.4f}")

    # Test single value estimation
    single_value = model.estimate_value(test_state)
    print(f"Single value estimate: {single_value:.4f}")

    # Test batch processing
    batch_states = torch.randn(8, 1536)
    batch_values = model.forward(batch_states)
    print(f"Batch output shape: {batch_values.shape}")

    # Test GAE
    print("\nTesting GAE...")
    gae = GeneralizedAdvantageEstimator(gamma=0.99, gae_lambda=0.95)

    rewards = torch.tensor([1.0, 0.5, 0.8, 1.2, 0.3])
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    next_values = torch.tensor([0.6, 0.7, 0.8, 0.9, 0.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

    advantages = gae.compute_advantages(rewards, values, next_values, dones)
    print(f"Advantages: {advantages.numpy()}")

    returns = gae.compute_returns(rewards, dones)
    print(f"Returns: {returns.numpy()}")

    # Test save/load
    model.save("test_value.pth")
    loaded_model = ValueNetwork.load("test_value.pth")
    os.remove("test_value.pth")

    print("\nAll tests passed!")
