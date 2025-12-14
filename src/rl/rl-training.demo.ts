/**
 * DIRECTIVE-023: RL Training Demo
 *
 * This demo showcases the complete RL training system for prompt mutation
 * optimization using PPO (Proximal Policy Optimization).
 *
 * Features demonstrated:
 * - PolicyNetwork action selection
 * - Experience collection
 * - PPO training updates
 * - Training statistics tracking
 *
 * Run with: npx ts-node src/rl/rl-training.demo.ts
 */

import { RLInterface, RLTrainer, TrainingStats } from './interface';
import { RewardModel } from '../models/rewardModel';
import { mutationTypes, MutationType } from '../mutations';
import { classifyPrompt } from '../types/promptTypes';

// ============================================================================
// DEMO CONFIGURATION
// ============================================================================

const DEMO_CONFIG = {
  numEpisodes: 50,
  episodeLength: 3,
  updateInterval: 10,
  verbose: true,
};

const SAMPLE_PROMPTS = [
  'Write a function to sort an array of integers in ascending order',
  'Create a REST API endpoint for user authentication',
  'Explain how machine learning models are trained',
  'Debug the memory leak in the application',
  'Optimize the database query for better performance',
  'Implement a caching mechanism for API responses',
  'Design a user interface for the settings page',
  'Review the code for security vulnerabilities',
  'Generate unit tests for the payment module',
  'Refactor the legacy authentication system',
];

// ============================================================================
// SIMULATED TRAINING (No Python Server Required)
// ============================================================================

/**
 * Simulated PPO trainer for demonstration without Python dependency.
 * This demonstrates the training loop structure.
 */
class SimulatedRLTrainer {
  private rewardModel: RewardModel;
  private actionCounts: Map<MutationType, number> = new Map();
  private rewardHistory: number[] = [];
  private updateCount: number = 0;

  constructor() {
    this.rewardModel = new RewardModel();
    mutationTypes.forEach(m => this.actionCounts.set(m, 0));
  }

  /**
   * Select action based on simple epsilon-greedy policy.
   */
  selectAction(prompt: string): { action: MutationType; probability: number } {
    // Epsilon-greedy with epsilon = 0.2
    const epsilon = 0.2;

    if (Math.random() < epsilon) {
      // Explore: random action
      const action = mutationTypes[Math.floor(Math.random() * mutationTypes.length)];
      return { action, probability: epsilon / mutationTypes.length };
    }

    // Exploit: choose based on past rewards
    let bestAction = mutationTypes[0];
    let bestCount = 0;

    for (const [action, count] of this.actionCounts) {
      if (count > bestCount) {
        bestAction = action;
        bestCount = count;
      }
    }

    return { action: bestAction, probability: 1 - epsilon };
  }

  /**
   * Calculate reward for a mutation.
   */
  calculateReward(
    originalPrompt: string,
    mutatedPrompt: string,
    mutationType: MutationType
  ): number {
    const category = classifyPrompt(originalPrompt).category;
    const prediction = this.rewardModel.predict(
      originalPrompt,
      mutatedPrompt,
      mutationType,
      category
    );
    return prediction.score;
  }

  /**
   * Run a training episode.
   */
  trainEpisode(
    prompts: string[],
    maxSteps: number
  ): { avgReward: number; actions: MutationType[] } {
    const actions: MutationType[] = [];
    let totalReward = 0;
    let steps = 0;

    for (const prompt of prompts) {
      for (let step = 0; step < maxSteps; step++) {
        // Select action
        const { action } = this.selectAction(prompt);
        actions.push(action);

        // Simulate mutation (simplified)
        const mutatedPrompt = `${prompt} [${action}]`;

        // Calculate reward
        const reward = this.calculateReward(prompt, mutatedPrompt, action);
        totalReward += reward;
        steps++;

        // Update action counts (simple learning)
        if (reward > 0.5) {
          this.actionCounts.set(action, (this.actionCounts.get(action) || 0) + 1);
        }

        this.rewardHistory.push(reward);
      }
    }

    this.updateCount++;

    return {
      avgReward: totalReward / steps,
      actions,
    };
  }

  /**
   * Get training statistics.
   */
  getStats(): {
    avgReward: number;
    maxReward: number;
    totalUpdates: number;
    actionDistribution: Map<MutationType, number>;
  } {
    const recentRewards = this.rewardHistory.slice(-100);

    return {
      avgReward: recentRewards.length > 0
        ? recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length
        : 0,
      maxReward: recentRewards.length > 0 ? Math.max(...recentRewards) : 0,
      totalUpdates: this.updateCount,
      actionDistribution: new Map(this.actionCounts),
    };
  }
}

// ============================================================================
// DEMO FUNCTIONS
// ============================================================================

/**
 * Demo: Simulated RL training (no Python required).
 */
function runSimulatedDemo(): void {
  console.log('='.repeat(70));
  console.log('DIRECTIVE-023: RL Training Demo (Simulated)');
  console.log('='.repeat(70));
  console.log('\nThis demo simulates the RL training loop without Python.\n');

  const trainer = new SimulatedRLTrainer();

  console.log(`Training on ${SAMPLE_PROMPTS.length} prompts`);
  console.log(`Episodes: ${DEMO_CONFIG.numEpisodes}`);
  console.log(`Steps per episode: ${DEMO_CONFIG.episodeLength}`);
  console.log('-'.repeat(70));

  // Training loop
  for (let ep = 0; ep < DEMO_CONFIG.numEpisodes; ep++) {
    const result = trainer.trainEpisode(SAMPLE_PROMPTS, DEMO_CONFIG.episodeLength);

    if ((ep + 1) % DEMO_CONFIG.updateInterval === 0) {
      const stats = trainer.getStats();
      console.log(
        `Episode ${String(ep + 1).padStart(3)}: ` +
        `Avg Reward = ${result.avgReward.toFixed(4)}, ` +
        `Running Avg = ${stats.avgReward.toFixed(4)}`
      );
    }
  }

  // Final statistics
  console.log('\n' + '='.repeat(70));
  console.log('Training Summary');
  console.log('='.repeat(70));

  const finalStats = trainer.getStats();
  console.log(`\nTotal Updates: ${finalStats.totalUpdates}`);
  console.log(`Average Reward: ${finalStats.avgReward.toFixed(4)}`);
  console.log(`Max Reward: ${finalStats.maxReward.toFixed(4)}`);

  console.log('\nAction Distribution:');
  for (const [action, count] of finalStats.actionDistribution) {
    const bar = '█'.repeat(Math.floor(count / 5));
    console.log(`  ${action.padEnd(20)}: ${String(count).padStart(4)} ${bar}`);
  }

  console.log('\nSimulated demo complete!');
}

/**
 * Demo: Full RL training with Python server.
 * Requires Python with PyTorch installed.
 */
async function runFullDemo(): Promise<void> {
  console.log('='.repeat(70));
  console.log('DIRECTIVE-023: Full RL Training Demo');
  console.log('='.repeat(70));
  console.log('\nThis demo requires the Python RL server to be running.');
  console.log('Start server with: python3 src/rl/ppo_trainer.py --server 8765\n');

  const trainer = new RLTrainer({
    serverHost: 'localhost',
    serverPort: 8765,
    embeddingDim: 1536,
  });

  try {
    // Check if server is available
    const rl = trainer.getInterface();
    const isHealthy = await rl.checkHealth();

    if (!isHealthy) {
      console.log('Python RL server is not running.');
      console.log('Falling back to simulated demo...\n');
      runSimulatedDemo();
      return;
    }

    console.log('Connected to RL server!');
    console.log('-'.repeat(70));

    // Training loop
    const stats = await trainer.train(
      SAMPLE_PROMPTS,
      DEMO_CONFIG.numEpisodes,
      DEMO_CONFIG.episodeLength,
      DEMO_CONFIG.updateInterval,
      (episode, stats) => {
        if (episode % DEMO_CONFIG.updateInterval === 0) {
          console.log(
            `Episode ${String(episode).padStart(3)}: ` +
            `Avg Reward = ${stats.avgReward?.toFixed(4) || 'N/A'}`
          );
        }
      }
    );

    // Final statistics
    console.log('\n' + '='.repeat(70));
    console.log('Training Summary');
    console.log('='.repeat(70));
    console.log(`\nTotal Updates: ${stats.totalUpdates}`);
    console.log(`Total Episodes: ${stats.totalEpisodes}`);
    console.log(`Average Reward: ${stats.avgReward.toFixed(4)}`);
    console.log(`Max Reward: ${stats.maxReward.toFixed(4)}`);
    console.log(`Avg Policy Loss: ${stats.avgPolicyLoss.toFixed(6)}`);
    console.log(`Avg Value Loss: ${stats.avgValueLoss.toFixed(6)}`);
    console.log(`Avg Entropy: ${stats.avgEntropy.toFixed(6)}`);

    // Save checkpoint
    await rl.saveCheckpoint('rl_demo_checkpoint.pth');
    console.log('\nCheckpoint saved!');

    trainer.stop();
    console.log('\nFull demo complete!');

  } catch (error) {
    console.error('Error during training:', error);
    console.log('\nFalling back to simulated demo...\n');
    trainer.stop();
    runSimulatedDemo();
  }
}

/**
 * Demo: Action selection only.
 */
async function runActionSelectionDemo(): Promise<void> {
  console.log('='.repeat(70));
  console.log('DIRECTIVE-023: Action Selection Demo');
  console.log('='.repeat(70));

  const rl = new RLInterface();

  console.log('\nDemonstrating action selection for sample prompts:\n');

  for (const prompt of SAMPLE_PROMPTS.slice(0, 5)) {
    // Get simulated embedding
    const embedding = Array(128).fill(0).map(() => Math.random() * 2 - 1);

    // Try to select action (will fail if server not running)
    try {
      const action = await rl.selectAction(embedding);
      console.log(`Prompt: "${prompt.substring(0, 50)}..."`);
      console.log(`  -> Action: ${action.mutationType}`);
      console.log(`     Probability: ${action.probability.toFixed(4)}`);
      console.log(`     Value: ${action.value.toFixed(4)}\n`);
    } catch {
      // Simulate action selection
      const actionIndex = Math.floor(Math.random() * mutationTypes.length);
      console.log(`Prompt: "${prompt.substring(0, 50)}..."`);
      console.log(`  -> Action: ${mutationTypes[actionIndex]} (simulated)`);
      console.log(`     Probability: ${(1 / mutationTypes.length).toFixed(4)}`);
      console.log(`     Value: ${(Math.random() * 0.5).toFixed(4)}\n`);
    }
  }

  console.log('Action selection demo complete!');
}

// ============================================================================
// MAIN
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes('--full')) {
    await runFullDemo();
  } else if (args.includes('--action')) {
    await runActionSelectionDemo();
  } else {
    // Default: simulated demo
    runSimulatedDemo();
  }
}

main().catch(console.error);

// ============================================================================
// EXPORTS FOR TESTING
// ============================================================================

export {
  SimulatedRLTrainer,
  SAMPLE_PROMPTS,
  DEMO_CONFIG,
  runSimulatedDemo,
  runFullDemo,
  runActionSelectionDemo,
};
