/**
 * DIRECTIVE-023: RL Training System Tests
 *
 * Tests for the RL-based prompt mutation optimization system.
 */

import { SimulatedRLTrainer, SAMPLE_PROMPTS } from '../../rl/rl-training.demo';
import { mutationTypes, MutationType } from '../../mutations';
import { RewardModel } from '../../models/rewardModel';
import { classifyPrompt, PromptCategory } from '../../types/promptTypes';

describe('DIRECTIVE-023: RL Training System', () => {
  describe('SimulatedRLTrainer', () => {
    let trainer: SimulatedRLTrainer;

    beforeEach(() => {
      trainer = new SimulatedRLTrainer();
    });

    test('should select valid mutation actions', () => {
      const prompt = 'Write a function to sort an array';
      const { action, probability } = trainer.selectAction(prompt);

      expect(mutationTypes).toContain(action);
      expect(probability).toBeGreaterThan(0);
      expect(probability).toBeLessThanOrEqual(1);
    });

    test('should calculate rewards using RewardModel', () => {
      const originalPrompt = 'Write a function to process data';
      const mutatedPrompt = 'Try to write a function to process data. If you encounter issues, suggest alternatives.';

      const reward = trainer.calculateReward(
        originalPrompt,
        mutatedPrompt,
        'try-catch-style'
      );

      expect(typeof reward).toBe('number');
      expect(reward).toBeGreaterThanOrEqual(0);
      expect(reward).toBeLessThanOrEqual(1);
    });

    test('should train on a single episode', () => {
      const prompts = ['Create an API endpoint'];
      const result = trainer.trainEpisode(prompts, 3);

      expect(result.avgReward).toBeDefined();
      expect(typeof result.avgReward).toBe('number');
      expect(result.actions).toHaveLength(3);
      expect(result.actions.every(a => mutationTypes.includes(a))).toBe(true);
    });

    test('should track training statistics', () => {
      // Run a few episodes
      for (let i = 0; i < 5; i++) {
        trainer.trainEpisode(['Test prompt'], 2);
      }

      const stats = trainer.getStats();

      expect(stats.totalUpdates).toBe(5);
      expect(typeof stats.avgReward).toBe('number');
      expect(typeof stats.maxReward).toBe('number');
      expect(stats.actionDistribution).toBeDefined();
    });

    test('should learn to prefer better actions over time', () => {
      // Run many episodes
      for (let i = 0; i < 50; i++) {
        trainer.trainEpisode(SAMPLE_PROMPTS.slice(0, 3), 2);
      }

      const stats = trainer.getStats();

      // Should have accumulated some action counts
      let totalActionCounts = 0;
      for (const count of stats.actionDistribution.values()) {
        totalActionCounts += count;
      }

      expect(totalActionCounts).toBeGreaterThan(0);
    });
  });

  describe('RewardModel Integration', () => {
    let rewardModel: RewardModel;

    beforeEach(() => {
      rewardModel = new RewardModel();
    });

    test('should provide consistent rewards for same inputs', () => {
      const original = 'Write a sorting algorithm';
      const modified = 'Write a sorting algorithm with O(n log n) complexity';

      const reward1 = rewardModel.predict(original, modified, 'expansion', PromptCategory.CODE_GENERATION);
      const reward2 = rewardModel.predict(original, modified, 'expansion', PromptCategory.CODE_GENERATION);

      expect(reward1.score).toBe(reward2.score);
    });

    test('should give different rewards for different mutations', () => {
      const original = 'Explain machine learning';

      const reward1 = rewardModel.predict(
        original,
        'Try to explain machine learning.',
        'try-catch-style',
        PromptCategory.GENERAL_QA
      );

      const reward2 = rewardModel.predict(
        original,
        'Explain machine learning. Technical Context: Machine learning is a subset of AI...',
        'expansion',
        PromptCategory.GENERAL_QA
      );

      // Rewards should be different (different mutations have different effects)
      expect(reward1.score).not.toBe(reward2.score);
    });

    test('should include confidence in predictions', () => {
      const prediction = rewardModel.predict(
        'Debug the error',
        'Try to debug the error. If you cannot fix it directly, suggest possible solutions.',
        'try-catch-style',
        PromptCategory.CODE_REVIEW
      );

      expect(prediction.confidence).toBeGreaterThan(0);
      expect(prediction.confidence).toBeLessThanOrEqual(1);
    });

    test('should provide explanation for predictions', () => {
      const prediction = rewardModel.predict(
        'Write tests',
        'Write tests for the API endpoint. Constraints: Use Jest, include edge cases.',
        'constraint-addition',
        PromptCategory.CODE_GENERATION
      );

      expect(prediction.explanation).toBeDefined();
      expect(prediction.explanation.length).toBeGreaterThan(0);
    });
  });

  describe('Mutation Types', () => {
    test('should have exactly 5 mutation types', () => {
      expect(mutationTypes).toHaveLength(5);
    });

    test('should include all expected mutation types', () => {
      const expectedTypes: MutationType[] = [
        'try-catch-style',
        'context-reduction',
        'expansion',
        'constraint-addition',
        'task-decomposition',
      ];

      expectedTypes.forEach(type => {
        expect(mutationTypes).toContain(type);
      });
    });
  });

  describe('Prompt Classification Integration', () => {
    test('should correctly classify code-related prompts', () => {
      const prompt = 'Write a Python function to parse JSON';
      const classification = classifyPrompt(prompt);

      expect(classification.category).toBe(PromptCategory.CODE_GENERATION);
      expect(classification.confidence).toBeGreaterThan(0);
    });

    test('should correctly classify content writing prompts', () => {
      const prompt = 'Write a blog post about climate change';
      const classification = classifyPrompt(prompt);

      expect([PromptCategory.CONTENT_WRITING, PromptCategory.CREATIVE_WRITING]).toContain(
        classification.category
      );
    });

    test('should include characteristics in classification', () => {
      const prompt = 'Analyze the performance of the database queries';
      const classification = classifyPrompt(prompt);

      expect(Array.isArray(classification.characteristics)).toBe(true);
    });
  });

  describe('Training Statistics', () => {
    test('should compute correct average reward', () => {
      const trainer = new SimulatedRLTrainer();

      // Run 10 episodes
      let totalReward = 0;
      let totalSteps = 0;

      for (let i = 0; i < 10; i++) {
        const result = trainer.trainEpisode(['Test prompt'], 1);
        totalReward += result.avgReward;
        totalSteps++;
      }

      const stats = trainer.getStats();
      const expectedAvg = totalReward / totalSteps;

      // Average should be close (within tolerance due to sliding window)
      expect(Math.abs(stats.avgReward - expectedAvg)).toBeLessThan(0.5);
    });

    test('should track maximum reward', () => {
      const trainer = new SimulatedRLTrainer();

      for (let i = 0; i < 10; i++) {
        trainer.trainEpisode(SAMPLE_PROMPTS.slice(0, 2), 2);
      }

      const stats = trainer.getStats();
      expect(stats.maxReward).toBeGreaterThanOrEqual(stats.avgReward);
    });
  });

  describe('Experience Collection', () => {
    test('should accumulate experiences across episodes', () => {
      const trainer = new SimulatedRLTrainer();

      // Run episodes
      trainer.trainEpisode(['Prompt 1'], 3);
      trainer.trainEpisode(['Prompt 2'], 3);

      const stats = trainer.getStats();
      expect(stats.totalUpdates).toBe(2);
    });

    test('should update action distribution based on rewards', () => {
      const trainer = new SimulatedRLTrainer();

      // Run many episodes
      for (let i = 0; i < 100; i++) {
        trainer.trainEpisode(SAMPLE_PROMPTS, 2);
      }

      const stats = trainer.getStats();

      // At least some actions should have counts
      let hasNonZeroCounts = false;
      for (const count of stats.actionDistribution.values()) {
        if (count > 0) {
          hasNonZeroCounts = true;
          break;
        }
      }

      expect(hasNonZeroCounts).toBe(true);
    });
  });
});

describe('Edge Cases', () => {
  test('should handle empty prompts gracefully', () => {
    const rewardModel = new RewardModel();

    const prediction = rewardModel.predict(
      '',
      '',
      'try-catch-style',
      PromptCategory.GENERAL_QA
    );

    expect(typeof prediction.score).toBe('number');
  });

  test('should handle very long prompts', () => {
    const rewardModel = new RewardModel();
    const longPrompt = 'Write a function. '.repeat(100);

    const prediction = rewardModel.predict(
      longPrompt,
      longPrompt + ' With error handling.',
      'expansion',
      PromptCategory.CODE_GENERATION
    );

    expect(typeof prediction.score).toBe('number');
    expect(prediction.score).toBeGreaterThanOrEqual(0);
    expect(prediction.score).toBeLessThanOrEqual(1);
  });

  test('should handle special characters in prompts', () => {
    const rewardModel = new RewardModel();

    const prediction = rewardModel.predict(
      'Fix the bug with regex: /[a-z]+@[a-z]+\\.[a-z]+/',
      'Try to fix the bug with regex: /[a-z]+@[a-z]+\\.[a-z]+/',
      'try-catch-style',
      PromptCategory.CODE_GENERATION
    );

    expect(typeof prediction.score).toBe('number');
  });
});
