/**
 * DIRECTIVE-023: TypeScript Interface for RL Training
 *
 * This module provides a TypeScript wrapper for the Python PPO trainer,
 * enabling prompt mutation optimization through reinforcement learning.
 *
 * Features:
 * - HTTP communication with Python RL server
 * - Experience collection and storage
 * - Training orchestration
 * - Model persistence
 * - Integration with mutations and evaluator
 */

import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import { MutationType, mutationTypes, PromptVariation, tryCatchStyleMutation, reduceContextMutation, expandMutation, constrainMutation } from '../mutations';
import { classifyPrompt, PromptCategory } from '../types/promptTypes';
import { RewardModel } from '../models/rewardModel';

// ============================================================================
// INTERFACES
// ============================================================================

export interface RLAction {
  actionIndex: number;
  mutationType: MutationType;
  probability: number;
  logProb: number;
  value: number;
}

export interface TrainingExperience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
  logProb: number;
  value: number;
}

export interface TrainingStats {
  totalUpdates: number;
  totalEpisodes: number;
  avgReward: number;
  maxReward: number;
  avgPolicyLoss: number;
  avgValueLoss: number;
  avgEntropy: number;
  avgKl: number;
}

export interface UpdateResult {
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
  learningRate: number;
  bufferSize: number;
  updateCount: number;
}

export interface RLConfig {
  serverHost: string;
  serverPort: number;
  pythonPath: string;
  embeddingDim: number;
  autoStartServer: boolean;
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

const DEFAULT_CONFIG: RLConfig = {
  serverHost: 'localhost',
  serverPort: 8765,
  pythonPath: 'python3',
  embeddingDim: 1536,
  autoStartServer: true,
};

// ============================================================================
// RL INTERFACE CLASS
// ============================================================================

/**
 * RLInterface provides TypeScript bindings for the Python PPO trainer.
 *
 * It handles:
 * - Starting/stopping the Python server
 * - Communicating with the server via HTTP
 * - Experience collection and training
 * - Integration with the project's mutations and evaluator
 */
export class RLInterface {
  private config: RLConfig;
  private serverProcess: ChildProcess | null = null;
  private isConnected: boolean = false;
  private baseUrl: string;
  private rewardModel: RewardModel;

  constructor(config: Partial<RLConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.baseUrl = `http://${this.config.serverHost}:${this.config.serverPort}`;
    this.rewardModel = new RewardModel();
  }

  // ===========================================================================
  // SERVER MANAGEMENT
  // ===========================================================================

  /**
   * Start the Python RL server.
   */
  async startServer(): Promise<void> {
    if (this.isConnected) {
      console.log('Server already running');
      return;
    }

    const scriptPath = path.join(__dirname, 'ppo_trainer.py');

    return new Promise((resolve, reject) => {
      this.serverProcess = spawn(
        this.config.pythonPath,
        [scriptPath, '--server', String(this.config.serverPort)],
        {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: __dirname,
        }
      );

      if (this.serverProcess.stdout) {
        this.serverProcess.stdout.on('data', (data: Buffer) => {
          const message = data.toString();
          console.log(`[RL Server] ${message.trim()}`);
          if (message.includes('Server started')) {
            this.isConnected = true;
            resolve();
          }
        });
      }

      if (this.serverProcess.stderr) {
        this.serverProcess.stderr.on('data', (data: Buffer) => {
          console.error(`[RL Server Error] ${data.toString().trim()}`);
        });
      }

      this.serverProcess.on('error', (error) => {
        console.error('Failed to start RL server:', error);
        reject(error);
      });

      this.serverProcess.on('close', (code) => {
        console.log(`RL Server exited with code ${code}`);
        this.isConnected = false;
        this.serverProcess = null;
      });

      // Timeout for connection
      setTimeout(() => {
        if (!this.isConnected) {
          // Try to connect anyway
          this.checkHealth()
            .then(() => {
              this.isConnected = true;
              resolve();
            })
            .catch(() => reject(new Error('Server failed to start in time')));
        }
      }, 5000);
    });
  }

  /**
   * Stop the Python RL server.
   */
  stopServer(): void {
    if (this.serverProcess) {
      this.serverProcess.kill('SIGTERM');
      this.serverProcess = null;
      this.isConnected = false;
      console.log('RL Server stopped');
    }
  }

  /**
   * Check if server is healthy.
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.fetch('/health', 'GET');
      return response.status === 'ok';
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // HTTP COMMUNICATION
  // ===========================================================================

  /**
   * Make HTTP request to the server.
   */
  private async fetch(
    endpoint: string,
    method: 'GET' | 'POST' = 'GET',
    body?: any
  ): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;

    const options: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    if (body && method === 'POST') {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // ===========================================================================
  // ACTION SELECTION
  // ===========================================================================

  /**
   * Get action distribution from the policy network.
   */
  async getActionDistribution(embedding: number[]): Promise<{
    probabilities: number[];
    mutationTypes: MutationType[];
  }> {
    const response = await this.fetch('/get_distribution', 'POST', { embedding });
    return {
      probabilities: response.probabilities,
      mutationTypes: response.mutation_types,
    };
  }

  /**
   * Select an action given a prompt embedding.
   */
  async selectAction(embedding: number[], deterministic: boolean = false): Promise<RLAction> {
    const response = await this.fetch('/select_action', 'POST', {
      embedding,
      deterministic,
    });

    return {
      actionIndex: response.action,
      mutationType: response.mutation_type,
      probability: Math.exp(response.log_prob),
      logProb: response.log_prob,
      value: response.value,
    };
  }

  /**
   * Select action for a prompt (convenience method).
   * Uses simulated embedding for demonstration.
   */
  async selectMutation(prompt: string, deterministic: boolean = false): Promise<RLAction> {
    const embedding = this.getSimulatedEmbedding(prompt);
    return this.selectAction(embedding, deterministic);
  }

  // ===========================================================================
  // EXPERIENCE STORAGE
  // ===========================================================================

  /**
   * Store a single experience.
   */
  async storeExperience(experience: TrainingExperience): Promise<{ stored: boolean; bufferSize: number }> {
    const response = await this.fetch('/store_experience', 'POST', {
      state: experience.state,
      action: experience.action,
      reward: experience.reward,
      next_state: experience.nextState,
      done: experience.done,
      log_prob: experience.logProb,
      value: experience.value,
    });

    return {
      stored: response.stored,
      bufferSize: response.buffer_size,
    };
  }

  /**
   * Collect experience from a prompt mutation.
   */
  async collectExperience(
    originalPrompt: string,
    mutationType: MutationType,
    action: RLAction,
    done: boolean = false
  ): Promise<{ stored: boolean; reward: number }> {
    // Apply mutation
    const variation = this.applyMutation(originalPrompt, mutationType);

    // Get embedding for original prompt
    const state = this.getSimulatedEmbedding(originalPrompt);

    // Get embedding for mutated prompt
    const nextState = this.getSimulatedEmbedding(variation.text);

    // Calculate reward using reward model
    const category = classifyPrompt(originalPrompt).category;
    const prediction = this.rewardModel.predict(
      originalPrompt,
      variation.text,
      mutationType,
      category
    );
    const reward = prediction.score;

    // Store experience
    const result = await this.storeExperience({
      state,
      action: action.actionIndex,
      reward,
      nextState,
      done,
      logProb: action.logProb,
      value: action.value,
    });

    return {
      stored: result.stored,
      reward,
    };
  }

  // ===========================================================================
  // TRAINING
  // ===========================================================================

  /**
   * Trigger a PPO update.
   */
  async update(): Promise<UpdateResult | null> {
    const response = await this.fetch('/update', 'POST');

    if (response.message) {
      // Not enough experiences
      return null;
    }

    return {
      policyLoss: response.policy_loss,
      valueLoss: response.value_loss,
      entropy: response.entropy,
      approxKl: response.approx_kl,
      clipFraction: response.clip_fraction,
      learningRate: response.learning_rate,
      bufferSize: response.buffer_size,
      updateCount: response.update_count,
    };
  }

  /**
   * Get training statistics.
   */
  async getStats(): Promise<TrainingStats> {
    const response = await this.fetch('/stats', 'GET');

    return {
      totalUpdates: response.total_updates || 0,
      totalEpisodes: response.total_episodes || 0,
      avgReward: response.avg_reward || 0,
      maxReward: response.max_reward || 0,
      avgPolicyLoss: response.avg_policy_loss || 0,
      avgValueLoss: response.avg_value_loss || 0,
      avgEntropy: response.avg_entropy || 0,
      avgKl: response.avg_kl || 0,
    };
  }

  /**
   * Run a training episode.
   */
  async trainEpisode(
    prompts: string[],
    maxSteps: number = 5
  ): Promise<{
    avgReward: number;
    totalSteps: number;
    mutations: MutationType[];
  }> {
    const mutations: MutationType[] = [];
    let totalReward = 0;
    let totalSteps = 0;

    for (const prompt of prompts) {
      let currentPrompt = prompt;

      for (let step = 0; step < maxSteps; step++) {
        // Select action
        const action = await this.selectMutation(currentPrompt);
        mutations.push(action.mutationType);

        // Collect experience
        const done = step === maxSteps - 1;
        const { reward } = await this.collectExperience(
          currentPrompt,
          action.mutationType,
          action,
          done
        );

        totalReward += reward;
        totalSteps++;

        // Apply mutation for next step
        const variation = this.applyMutation(currentPrompt, action.mutationType);
        currentPrompt = variation.text;

        if (done) break;
      }
    }

    return {
      avgReward: totalReward / totalSteps,
      totalSteps,
      mutations,
    };
  }

  // ===========================================================================
  // MODEL PERSISTENCE
  // ===========================================================================

  /**
   * Save model checkpoint.
   */
  async saveCheckpoint(path?: string): Promise<void> {
    await this.fetch('/save', 'POST', { path });
  }

  /**
   * Load model checkpoint.
   */
  async loadCheckpoint(path: string): Promise<boolean> {
    try {
      const response = await this.fetch('/load', 'POST', { path });
      return response.loaded;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Apply a mutation to a prompt.
   */
  private applyMutation(prompt: string, mutationType: MutationType): PromptVariation {
    const category = classifyPrompt(prompt).category;

    switch (mutationType) {
      case 'try-catch-style':
        return tryCatchStyleMutation(prompt);
      case 'context-reduction':
        return reduceContextMutation(prompt);
      case 'expansion':
        return expandMutation(prompt);
      case 'constraint-addition':
        return constrainMutation(prompt, category);
      case 'task-decomposition':
        // Fallback to expansion for now
        return expandMutation(prompt);
      default:
        return tryCatchStyleMutation(prompt);
    }
  }

  /**
   * Get simulated embedding for a prompt.
   * In production, replace with actual embedding API call.
   */
  private getSimulatedEmbedding(prompt: string): number[] {
    // Simple hash-based pseudo-random embedding
    const embedding: number[] = [];
    let seed = 0;
    for (let i = 0; i < prompt.length; i++) {
      seed = (seed * 31 + prompt.charCodeAt(i)) % 2147483647;
    }

    // Generate embedding
    for (let i = 0; i < this.config.embeddingDim; i++) {
      seed = (seed * 1103515245 + 12345) % 2147483647;
      embedding.push((seed / 2147483647) * 2 - 1);
    }

    return embedding;
  }

  /**
   * Set the reward model for calculating rewards.
   */
  setRewardModel(model: RewardModel): void {
    this.rewardModel = model;
  }
}

// ============================================================================
// STANDALONE TRAINING ORCHESTRATOR
// ============================================================================

/**
 * High-level training orchestrator that manages the RL training loop.
 */
export class RLTrainer {
  private rlInterface: RLInterface;
  private isRunning: boolean = false;

  constructor(config: Partial<RLConfig> = {}) {
    this.rlInterface = new RLInterface(config);
  }

  /**
   * Start the training server.
   */
  async start(): Promise<void> {
    await this.rlInterface.startServer();
    this.isRunning = true;
  }

  /**
   * Stop the training server.
   */
  stop(): void {
    this.rlInterface.stopServer();
    this.isRunning = false;
  }

  /**
   * Run training for a specified number of episodes.
   */
  async train(
    prompts: string[],
    numEpisodes: number = 100,
    episodeLength: number = 5,
    updateInterval: number = 10,
    onProgress?: (episode: number, stats: any) => void
  ): Promise<TrainingStats> {
    if (!this.isRunning) {
      await this.start();
    }

    console.log(`Starting RL training: ${numEpisodes} episodes`);

    for (let ep = 0; ep < numEpisodes; ep++) {
      // Run episode
      const episodeResult = await this.rlInterface.trainEpisode(prompts, episodeLength);

      // Update policy periodically
      if ((ep + 1) % updateInterval === 0) {
        const updateResult = await this.rlInterface.update();
        if (updateResult) {
          console.log(
            `Episode ${ep + 1}: ` +
            `Reward=${episodeResult.avgReward.toFixed(4)}, ` +
            `PolicyLoss=${updateResult.policyLoss.toFixed(4)}`
          );
        }
      }

      // Report progress
      if (onProgress) {
        const stats = await this.rlInterface.getStats();
        onProgress(ep + 1, { ...stats, ...episodeResult });
      }
    }

    return this.rlInterface.getStats();
  }

  /**
   * Get the underlying RL interface.
   */
  getInterface(): RLInterface {
    return this.rlInterface;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default RLInterface;
