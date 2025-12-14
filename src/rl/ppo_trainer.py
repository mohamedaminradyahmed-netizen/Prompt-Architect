"""
DIRECTIVE-023: PPO Trainer for Prompt Mutation RL

This module implements the Proximal Policy Optimization (PPO) algorithm
for training the prompt mutation policy. PPO provides stable training
through clipped objective functions.

Key Features:
- Clipped PPO objective for stable training
- Generalized Advantage Estimation (GAE)
- Experience replay buffer
- Checkpointing and logging
- HTTP server for TypeScript integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
import time
from collections import deque

from policy import PolicyNetwork, MUTATION_TYPES, action_to_mutation
from value import ValueNetwork, GeneralizedAdvantageEstimator


@dataclass
class Experience:
    """A single experience tuple."""
    state: np.ndarray          # Prompt embedding
    action: int                # Mutation action index
    reward: float              # Reward received
    next_state: np.ndarray     # Next prompt embedding
    done: bool                 # Episode ended
    log_prob: float            # Log probability of action
    value: float               # Value estimate


class ExperienceBuffer:
    """Buffer for storing and sampling experiences."""

    def __init__(self, max_size: int = 10000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, experience: Experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: List[Experience]):
        """Add multiple experiences."""
        for exp in experiences:
            self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        indices = np.random.choice(
            len(self.buffer),
            min(batch_size, len(self.buffer)),
            replace=False
        )
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Experience]:
        """Get all experiences."""
        return list(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Network dimensions
    input_dim: int = 1536
    action_dim: int = 5
    hidden_dim: int = 256

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    eps_clip: float = 0.2         # PPO clip range
    value_clip: float = 0.2       # Value function clip range
    entropy_coef: float = 0.01    # Entropy bonus coefficient
    value_coef: float = 0.5       # Value loss coefficient
    max_grad_norm: float = 0.5    # Gradient clipping

    # Training parameters
    n_epochs: int = 4             # PPO epochs per update
    batch_size: int = 64          # Mini-batch size
    buffer_size: int = 2048       # Experience buffer size
    min_buffer_size: int = 128    # Minimum experiences before training

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100  # Save every N updates


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.

    Implements the PPO-Clip algorithm for stable policy gradient training.
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        """
        Initialize PPO trainer.

        Args:
            config: PPO configuration. Uses defaults if None.
        """
        self.config = config or PPOConfig()

        # Initialize networks
        self.policy = PolicyNetwork(
            input_dim=self.config.input_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        )
        self.policy_old = PolicyNetwork(
            input_dim=self.config.input_dim,
            output_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value_net = ValueNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim
        )

        # Optimizer
        self.optimizer = optim.AdamW([
            {'params': self.policy.parameters(), 'lr': self.config.learning_rate},
            {'params': self.value_net.parameters(), 'lr': self.config.learning_rate}
        ], weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )

        # GAE
        self.gae = GeneralizedAdvantageEstimator(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=self.config.buffer_size)

        # Training statistics
        self.update_count = 0
        self.episode_count = 0
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'rewards': [],
            'approx_kl': [],
            'clip_fraction': [],
        }

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select an action given a state.

        Args:
            state: Prompt embedding
            deterministic: If True, select best action; else sample

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            # Get action from old policy
            action, prob, log_prob = self.policy_old.get_action(
                state_tensor, deterministic=deterministic
            )
            # Get value estimate
            value = self.value_net.estimate_value(state_tensor)

        return action, log_prob.item(), value

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store an experience in the buffer."""
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.buffer.add(exp)

    def update(self) -> Dict[str, float]:
        """
        Perform a PPO update.

        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return {}

        # Get all experiences
        experiences = self.buffer.get_all()

        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([float(e.done) for e in experiences])
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences])
        old_values = torch.FloatTensor([e.value for e in experiences])

        # Compute returns and advantages
        with torch.no_grad():
            next_values = self.value_net.estimate_values(next_states)
            values = self.value_net.estimate_values(states)

        advantages = self.gae.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset = TensorDataset(
            states, actions, old_log_probs, returns, advantages, old_values
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        n_batches = 0

        for _ in range(self.config.n_epochs):
            for batch in dataloader:
                (batch_states, batch_actions, batch_old_log_probs,
                 batch_returns, batch_advantages, batch_old_values) = batch

                # Evaluate actions
                log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                values = self.value_net.estimate_values(batch_states)

                # Policy loss (PPO-Clip)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1 - self.config.eps_clip,
                    1 + self.config.eps_clip
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.config.value_clip,
                    self.config.value_clip
                )
                value_loss1 = (values - batch_returns) ** 2
                value_loss2 = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = (torch.abs(ratios - 1) > self.config.eps_clip).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                n_batches += 1

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Update scheduler
        self.scheduler.step()

        # Clear buffer after update
        self.buffer.clear()

        # Update count
        self.update_count += 1

        # Compute average statistics
        stats = {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
            'approx_kl': total_approx_kl / n_batches,
            'clip_fraction': total_clip_fraction / n_batches,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'buffer_size': len(experiences),
            'update_count': self.update_count,
        }

        # Store in history
        self.training_stats['policy_losses'].append(stats['policy_loss'])
        self.training_stats['value_losses'].append(stats['value_loss'])
        self.training_stats['entropies'].append(stats['entropy'])
        self.training_stats['approx_kl'].append(stats['approx_kl'])
        self.training_stats['clip_fraction'].append(stats['clip_fraction'])

        # Checkpoint
        if self.update_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint()

        return stats

    def train_episode(
        self,
        prompts: List[str],
        get_embedding: callable,
        get_reward: callable,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Train on a single episode.

        Args:
            prompts: List of initial prompts
            get_embedding: Function to get prompt embedding
            get_reward: Function to get reward for (prompt, action, result)
            max_steps: Maximum steps per episode

        Returns:
            Episode statistics
        """
        self.episode_count += 1
        episode_rewards = []
        episode_actions = []

        for prompt in prompts:
            state = get_embedding(prompt)
            total_reward = 0

            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)
                mutation_type = action_to_mutation(action)

                # Get reward (from external reward function)
                reward, next_prompt, done = get_reward(prompt, mutation_type)

                # Get next state
                next_state = get_embedding(next_prompt)

                # Store experience
                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                )

                total_reward += reward
                prompt = next_prompt
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_actions.append(mutation_type)

        # Perform update if buffer is full enough
        update_stats = {}
        if len(self.buffer) >= self.config.min_buffer_size:
            update_stats = self.update()

        self.training_stats['rewards'].extend(episode_rewards)

        return {
            'episode': self.episode_count,
            'avg_reward': np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'actions': episode_actions,
            **update_stats
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"ppo_checkpoint_{self.update_count}.pth"
            )

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'update_count': self.update_count,
            'episode_count': self.episode_count,
            'config': self.config.__dict__,
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.update_count = checkpoint['update_count']
        self.episode_count = checkpoint['episode_count']
        self.training_stats = checkpoint['training_stats']

        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from update {self.update_count}, episode {self.episode_count}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        stats = self.training_stats

        if len(stats['rewards']) == 0:
            return {'message': 'No training data yet'}

        return {
            'total_updates': self.update_count,
            'total_episodes': self.episode_count,
            'avg_reward': np.mean(stats['rewards'][-100:]) if stats['rewards'] else 0,
            'max_reward': max(stats['rewards']) if stats['rewards'] else 0,
            'avg_policy_loss': np.mean(stats['policy_losses'][-10:]) if stats['policy_losses'] else 0,
            'avg_value_loss': np.mean(stats['value_losses'][-10:]) if stats['value_losses'] else 0,
            'avg_entropy': np.mean(stats['entropies'][-10:]) if stats['entropies'] else 0,
            'avg_kl': np.mean(stats['approx_kl'][-10:]) if stats['approx_kl'] else 0,
        }


# =============================================================================
# HTTP Server for TypeScript Integration
# =============================================================================

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class RLServerHandler(BaseHTTPRequestHandler):
    """HTTP handler for RL server."""

    trainer: PPOTrainer = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def send_json_response(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_json_response({'status': 'ok'})

        elif self.path == '/stats':
            stats = self.trainer.get_training_summary()
            self.send_json_response(stats)

        else:
            self.send_json_response({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, 400)
            return

        if self.path == '/select_action':
            # Select action given embedding
            embedding = np.array(data.get('embedding', []))
            deterministic = data.get('deterministic', False)

            action, log_prob, value = self.trainer.select_action(
                embedding, deterministic=deterministic
            )

            self.send_json_response({
                'action': action,
                'mutation_type': action_to_mutation(action),
                'log_prob': log_prob,
                'value': value,
            })

        elif self.path == '/store_experience':
            # Store experience
            self.trainer.store_experience(
                state=np.array(data['state']),
                action=data['action'],
                reward=data['reward'],
                next_state=np.array(data['next_state']),
                done=data['done'],
                log_prob=data['log_prob'],
                value=data['value']
            )
            self.send_json_response({'stored': True, 'buffer_size': len(self.trainer.buffer)})

        elif self.path == '/update':
            # Perform PPO update
            stats = self.trainer.update()
            self.send_json_response(stats if stats else {'message': 'Not enough experiences'})

        elif self.path == '/get_distribution':
            # Get action distribution
            embedding = np.array(data.get('embedding', []))
            state_tensor = torch.FloatTensor(embedding)
            probs = self.trainer.policy_old.get_action_distribution(state_tensor)
            self.send_json_response({
                'probabilities': probs.tolist(),
                'mutation_types': MUTATION_TYPES,
            })

        elif self.path == '/save':
            # Save checkpoint
            path = data.get('path')
            self.trainer.save_checkpoint(path)
            self.send_json_response({'saved': True})

        elif self.path == '/load':
            # Load checkpoint
            path = data.get('path')
            if path and os.path.exists(path):
                self.trainer.load_checkpoint(path)
                self.send_json_response({'loaded': True})
            else:
                self.send_json_response({'error': 'File not found'}, 404)

        else:
            self.send_json_response({'error': 'Not found'}, 404)


def start_server(trainer: PPOTrainer, host: str = 'localhost', port: int = 8765):
    """Start the RL server."""
    RLServerHandler.trainer = trainer
    server = HTTPServer((host, port), RLServerHandler)
    print(f"RL Server started at http://{host}:{port}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# =============================================================================
# Standalone Training Demo
# =============================================================================

def demo_training():
    """Demonstrate PPO training with synthetic data."""
    print("=" * 60)
    print("PPO Training Demo")
    print("=" * 60)

    # Create trainer
    config = PPOConfig(
        input_dim=128,  # Smaller for demo
        action_dim=5,
        hidden_dim=64,
        batch_size=32,
        min_buffer_size=64,
    )
    trainer = PPOTrainer(config)

    print(f"\nConfig: {config}")
    print(f"Policy parameters: {sum(p.numel() for p in trainer.policy.parameters())}")
    print(f"Value parameters: {sum(p.numel() for p in trainer.value_net.parameters())}")

    # Synthetic embedding and reward functions
    def get_embedding(prompt: str) -> np.ndarray:
        """Generate synthetic embedding."""
        np.random.seed(hash(prompt) % 2**32)
        return np.random.randn(config.input_dim).astype(np.float32)

    def get_reward(prompt: str, mutation_type: str) -> Tuple[float, str, bool]:
        """Generate synthetic reward."""
        # Simulate different rewards for different mutations
        base_reward = {
            'try-catch-style': 0.3,
            'context-reduction': 0.4,
            'expansion': 0.2,
            'constraint-addition': 0.35,
            'task-decomposition': 0.25,
        }.get(mutation_type, 0.1)

        # Add noise
        reward = base_reward + np.random.randn() * 0.1

        # Create modified prompt
        next_prompt = f"{prompt} [{mutation_type}]"

        # Episode ends with some probability
        done = np.random.random() < 0.2

        return reward, next_prompt, done

    # Sample prompts
    prompts = [
        "Write a function to sort an array",
        "Explain how neural networks work",
        "Create a REST API for user management",
        "Debug the authentication issue",
        "Optimize database queries",
    ]

    # Training loop
    n_episodes = 20
    for ep in range(n_episodes):
        stats = trainer.train_episode(
            prompts=prompts,
            get_embedding=get_embedding,
            get_reward=get_reward,
            max_steps=5
        )

        if stats:
            print(f"\nEpisode {ep + 1}/{n_episodes}")
            print(f"  Avg Reward: {stats.get('avg_reward', 0):.4f}")
            if 'policy_loss' in stats:
                print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                print(f"  Value Loss: {stats['value_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Save final checkpoint
    trainer.save_checkpoint("demo_checkpoint.pth")

    print("\nDemo complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--server':
        # Run as server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        trainer = PPOTrainer()
        server = start_server(trainer, port=port)
        print("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        # Run demo
        demo_training()
