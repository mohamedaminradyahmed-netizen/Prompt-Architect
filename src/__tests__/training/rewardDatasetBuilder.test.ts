/**
 * Unit Tests for Reward Dataset Builder (DIRECTIVE-054)
 *
 * Tests for collecting human feedback, validation, dataset building, and export
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  collectFeedback,
  validateFeedback,
  validateFeedbackBatch,
  buildRewardDataset,
  exportDataset,
  mergeDatasets,
  splitRewardDataset,
  filterByWeight,
  getDatasetSummary,
  type EnhancedFeedback,
  type RewardDataset,
  type DatasetFilters,
} from '../../training/rewardDatasetBuilder';
import { PromptCategory } from '../../types/promptTypes';

// ============================================================================
// VALIDATE FEEDBACK TESTS
// ============================================================================

describe('validateFeedback (DIRECTIVE-054)', () => {
  describe('Required Fields Validation', () => {
    it('should reject feedback without promptId', () => {
      const feedback = {
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
      } as EnhancedFeedback;

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('promptId is required');
    });

    it('should reject feedback without variationId', () => {
      const feedback = {
        promptId: 'prompt-1',
        score: 4,
        userId: 'user-1',
      } as EnhancedFeedback;

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('variationId is required');
    });

    it('should reject feedback without userId', () => {
      const feedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
      } as EnhancedFeedback;

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('userId is required');
    });

    it('should accept valid feedback with all required fields', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });
  });

  describe('Score Validation', () => {
    it('should reject non-numeric scores', () => {
      const feedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 'good' as any,
        userId: 'user-1',
      } as EnhancedFeedback;

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('score must be a number');
    });

    it('should reject scores less than 1', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 0,
        userId: 'user-1',
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('score must be between 1 and 5');
    });

    it('should reject scores greater than 5', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 6,
        userId: 'user-1',
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('score must be between 1 and 5');
    });

    it('should warn about non-integer scores', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 3.5,
        userId: 'user-1',
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(true);
      expect(result.warnings).toContain('score should be an integer (will be rounded)');
    });

    it('should accept valid integer scores', () => {
      for (const score of [1, 2, 3, 4, 5]) {
        const feedback: EnhancedFeedback = {
          promptId: 'prompt-1',
          variationId: 'var-1',
          score,
          userId: 'user-1',
        };

        const result = validateFeedback(feedback);
        expect(result.isValid).toBe(true);
      }
    });
  });

  describe('Enhanced Fields Warnings', () => {
    it('should warn if originalPrompt is missing', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).toContain('originalPrompt is missing - embedding will use placeholder');
    });

    it('should warn if variationText is missing', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
        originalPrompt: 'Write a function',
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).toContain('variationText is missing - embedding will use placeholder');
    });

    it('should warn about potential self-rating', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 5,
        userId: 'user-1',
        originalPrompt: 'Write a function',
        variationText: 'Write a function',
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).toContain('Identical prompt and variation with max score - might be self-rating');
    });

    it('should not warn about self-rating for non-max scores', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
        originalPrompt: 'Write a function',
        variationText: 'Write a function',
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).not.toContain('Identical prompt and variation with max score - might be self-rating');
    });
  });

  describe('Timestamp Validation', () => {
    it('should warn about future timestamps', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
        timestamp: new Date(Date.now() + 86400000), // tomorrow
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).toContain('timestamp is in the future');
    });

    it('should warn about very old timestamps', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'prompt-1',
        variationId: 'var-1',
        score: 4,
        userId: 'user-1',
        timestamp: new Date('2020-01-01'),
      };

      const result = validateFeedback(feedback);

      expect(result.warnings).toContain('timestamp is more than 1 year old');
    });
  });
});

describe('validateFeedbackBatch (DIRECTIVE-054)', () => {
  it('should validate multiple feedback items', () => {
    const feedbackList: EnhancedFeedback[] = [
      { promptId: 'p1', variationId: 'v1', score: 4, userId: 'u1' },
      { promptId: 'p2', variationId: 'v2', score: 5, userId: 'u2' },
      { promptId: '', variationId: 'v3', score: 3, userId: 'u3' }, // Invalid
    ];

    const result = validateFeedbackBatch(feedbackList);

    expect(result.valid).toBe(2);
    expect(result.invalid).toBe(1);
    expect(result.results).toHaveLength(3);
    expect(result.results[0].isValid).toBe(true);
    expect(result.results[1].isValid).toBe(true);
    expect(result.results[2].isValid).toBe(false);
  });

  it('should handle empty array', () => {
    const result = validateFeedbackBatch([]);

    expect(result.valid).toBe(0);
    expect(result.invalid).toBe(0);
    expect(result.results).toHaveLength(0);
  });
});

// ============================================================================
// BUILD REWARD DATASET TESTS
// ============================================================================

describe('buildRewardDataset (DIRECTIVE-054)', () => {
  describe('Dataset Structure', () => {
    it('should return a valid dataset structure', async () => {
      const dataset = await buildRewardDataset();

      expect(dataset).toHaveProperty('examples');
      expect(dataset).toHaveProperty('statistics');
      expect(dataset).toHaveProperty('metadata');
      expect(Array.isArray(dataset.examples)).toBe(true);
    });

    it('should include correct metadata', async () => {
      const dataset = await buildRewardDataset();

      expect(dataset.metadata).toHaveProperty('created');
      expect(dataset.metadata).toHaveProperty('version');
      expect(dataset.metadata).toHaveProperty('size');
      expect(dataset.metadata).toHaveProperty('embeddingDimension');
      expect(dataset.metadata).toHaveProperty('featureCount');
      expect(dataset.metadata.embeddingDimension).toBe(384);
    });

    it('should calculate statistics correctly', async () => {
      const dataset = await buildRewardDataset();

      expect(dataset.statistics).toHaveProperty('totalExamples');
      expect(dataset.statistics).toHaveProperty('avgScore');
      expect(dataset.statistics).toHaveProperty('scoreDistribution');
      expect(dataset.statistics).toHaveProperty('categoryDistribution');
      expect(dataset.statistics).toHaveProperty('mutationDistribution');
      expect(dataset.statistics).toHaveProperty('avgTokenCount');
      expect(dataset.statistics).toHaveProperty('avgSimilarity');
      expect(dataset.statistics).toHaveProperty('dateRange');
    });
  });

  describe('Filtering', () => {
    it('should apply category filter', async () => {
      const filters: DatasetFilters = {
        categories: [PromptCategory.CODE_GENERATION],
      };

      const dataset = await buildRewardDataset(filters);

      expect(dataset.metadata.filters).toBeDefined();
      expect(dataset.metadata.filters?.categories).toContain(PromptCategory.CODE_GENERATION);
    });

    it('should apply score range filter', async () => {
      const filters: DatasetFilters = {
        minScore: 3,
        maxScore: 5,
      };

      const dataset = await buildRewardDataset(filters);

      expect(dataset.metadata.filters?.minScore).toBe(3);
      expect(dataset.metadata.filters?.maxScore).toBe(5);
    });

    it('should apply date range filter', async () => {
      const filters: DatasetFilters = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-12-31'),
      };

      const dataset = await buildRewardDataset(filters);

      expect(dataset.metadata.filters?.startDate).toBeDefined();
      expect(dataset.metadata.filters?.endDate).toBeDefined();
    });

    it('should exclude duplicates when requested', async () => {
      const filters: DatasetFilters = {
        excludeDuplicates: true,
      };

      const dataset = await buildRewardDataset(filters);

      expect(dataset.metadata.filters?.excludeDuplicates).toBe(true);
    });
  });

  describe('RewardExample Structure', () => {
    it('should create examples with correct embedding dimensions', async () => {
      const dataset = await buildRewardDataset();

      if (dataset.examples.length > 0) {
        const example = dataset.examples[0];
        expect(example.promptEmbedding).toHaveLength(384);
        expect(example.variationEmbedding).toHaveLength(384);
      }
    });

    it('should normalize labels to 0-1 range', async () => {
      const dataset = await buildRewardDataset();

      for (const example of dataset.examples) {
        expect(example.label).toBeGreaterThanOrEqual(0);
        expect(example.label).toBeLessThanOrEqual(1);
      }
    });

    it('should include feature names', async () => {
      const dataset = await buildRewardDataset();

      if (dataset.examples.length > 0) {
        const example = dataset.examples[0];
        expect(example.featureNames).toBeDefined();
        expect(Array.isArray(example.featureNames)).toBe(true);
        expect(example.features.length).toBe(example.featureNames.length);
      }
    });

    it('should calculate weights in valid range', async () => {
      const dataset = await buildRewardDataset();

      for (const example of dataset.examples) {
        expect(example.weight).toBeGreaterThanOrEqual(0.1);
        expect(example.weight).toBeLessThanOrEqual(1.0);
      }
    });
  });
});

// ============================================================================
// EXPORT DATASET TESTS
// ============================================================================

describe('exportDataset (DIRECTIVE-054)', () => {
  const createMockDataset = (): RewardDataset => ({
    examples: [
      {
        id: 'ex-1',
        promptEmbedding: [0.1, 0.2, 0.3],
        variationEmbedding: [0.4, 0.5, 0.6],
        features: [10, 0.5, 0.8, 5, 1.2, 4.5, 0.02, 0.05, 0, 1, 0, 2],
        featureNames: [
          'tokenCount', 'tokenRatio', 'similarity', 'lengthDiff',
          'wordCountRatio', 'avgWordLength', 'punctuationRatio',
          'uppercaseRatio', 'categoryCode', 'hasExamples', 'hasConstraints', 'questionCount'
        ],
        label: 0.75,
        weight: 0.9,
        metadata: {
          originalPrompt: 'Write a function',
          variationText: 'Try to write a function',
          category: PromptCategory.CODE_GENERATION,
          mutationType: 'try-catch-style',
          rawScore: 4,
          timestamp: new Date('2024-06-15'),
          userId: 'user-1',
        },
      },
    ],
    statistics: {
      totalExamples: 1,
      avgScore: 4,
      scoreDistribution: { 4: 1 },
      categoryDistribution: { CODE_GENERATION: 1 },
      mutationDistribution: { 'try-catch-style': 1 },
      avgTokenCount: 10,
      avgSimilarity: 0.8,
      dateRange: {
        earliest: new Date('2024-06-15'),
        latest: new Date('2024-06-15'),
      },
    },
    metadata: {
      created: new Date(),
      version: '1.0.0',
      size: 1,
      embeddingDimension: 3,
      featureCount: 12,
    },
  });

  describe('JSON Export', () => {
    it('should export valid JSON', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'json');

      expect(result.format).toBe('json');
      expect(() => JSON.parse(result.data)).not.toThrow();
    });

    it('should preserve all dataset fields in JSON', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'json');
      const parsed = JSON.parse(result.data);

      expect(parsed.examples).toHaveLength(1);
      expect(parsed.statistics).toBeDefined();
      expect(parsed.metadata).toBeDefined();
    });
  });

  describe('JSON Lines Export', () => {
    it('should export valid JSON Lines format', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'jsonl');

      expect(result.format).toBe('jsonl');

      const lines = result.data.split('\n').filter(l => l.trim());
      expect(lines.length).toBe(1);

      // Each line should be valid JSON
      for (const line of lines) {
        expect(() => JSON.parse(line)).not.toThrow();
      }
    });

    it('should include required fields in each line', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'jsonl');

      const line = JSON.parse(result.data.split('\n')[0]);

      expect(line).toHaveProperty('id');
      expect(line).toHaveProperty('prompt_embedding');
      expect(line).toHaveProperty('variation_embedding');
      expect(line).toHaveProperty('features');
      expect(line).toHaveProperty('label');
      expect(line).toHaveProperty('weight');
    });
  });

  describe('CSV Export', () => {
    it('should export valid CSV format', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'csv');

      expect(result.format).toBe('csv');

      const lines = result.data.split('\n');
      expect(lines.length).toBeGreaterThan(1);

      // First line should be header
      const header = lines[0];
      expect(header).toContain('id');
      expect(header).toContain('label');
      expect(header).toContain('weight');
    });

    it('should escape special characters in CSV', async () => {
      const dataset = createMockDataset();
      dataset.examples[0].metadata.originalPrompt = 'Write a "function", with commas';

      const result = await exportDataset(dataset, 'csv');

      // Should handle quotes and commas
      expect(result.data).toContain('""function""');
    });
  });

  describe('Parquet Export (Fallback)', () => {
    it('should return fallback with note for parquet', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'parquet');

      expect(result.format).toBe('parquet');
      expect(result.note).toBeDefined();
      expect(result.note).toContain('Apache Arrow');
    });
  });

  describe('TFRecord Export (Fallback)', () => {
    it('should return TF-compatible JSON Lines with note', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'tfrecord');

      expect(result.format).toBe('tfrecord');
      expect(result.note).toBeDefined();
      expect(result.note).toContain('TensorFlow');

      // Should still be parseable
      const line = JSON.parse(result.data.split('\n')[0]);
      expect(line).toHaveProperty('prompt_embedding');
      expect(line.prompt_embedding).toHaveProperty('values');
    });
  });

  describe('Export Metadata', () => {
    it('should include size in export result', async () => {
      const dataset = createMockDataset();
      const result = await exportDataset(dataset, 'json');

      expect(result.size).toBeGreaterThan(0);
      expect(result.size).toBe(result.data.length);
    });
  });
});

// ============================================================================
// UTILITY FUNCTIONS TESTS
// ============================================================================

describe('Dataset Utilities (DIRECTIVE-054)', () => {
  const createMockDataset = (count: number): RewardDataset => {
    const examples = Array.from({ length: count }, (_, i) => ({
      id: `ex-${i}`,
      promptEmbedding: [0.1, 0.2, 0.3],
      variationEmbedding: [0.4, 0.5, 0.6],
      features: [10, 0.5, 0.8, 5, 1.2, 4.5, 0.02, 0.05, 0, 1, 0, 2],
      featureNames: ['tokenCount', 'tokenRatio', 'similarity', 'lengthDiff', 'wordCountRatio', 'avgWordLength', 'punctuationRatio', 'uppercaseRatio', 'categoryCode', 'hasExamples', 'hasConstraints', 'questionCount'],
      label: (i % 5) / 4,
      weight: 0.5 + (i % 5) * 0.1,
      metadata: {
        originalPrompt: `Prompt ${i}`,
        variationText: `Variation ${i}`,
        category: PromptCategory.CODE_GENERATION,
        mutationType: 'test',
        rawScore: (i % 5) + 1,
        timestamp: new Date(),
        userId: `user-${i % 3}`,
      },
    }));

    return {
      examples,
      statistics: {
        totalExamples: count,
        avgScore: 3,
        scoreDistribution: {},
        categoryDistribution: {},
        mutationDistribution: {},
        avgTokenCount: 10,
        avgSimilarity: 0.8,
        dateRange: { earliest: new Date(), latest: new Date() },
      },
      metadata: {
        created: new Date(),
        version: '1.0.0',
        size: count,
        embeddingDimension: 3,
        featureCount: 12,
      },
    };
  };

  describe('mergeDatasets', () => {
    it('should merge multiple datasets', () => {
      const ds1 = createMockDataset(5);
      const ds2 = createMockDataset(3);

      // Rename ds2 examples to avoid duplicates
      ds2.examples = ds2.examples.map((ex, i) => ({ ...ex, id: `ds2-ex-${i}` }));

      const merged = mergeDatasets(ds1, ds2);

      expect(merged.examples.length).toBe(8);
      expect(merged.statistics.totalExamples).toBe(8);
    });

    it('should remove duplicate IDs', () => {
      const ds1 = createMockDataset(5);
      const ds2 = createMockDataset(5); // Same IDs

      const merged = mergeDatasets(ds1, ds2);

      expect(merged.examples.length).toBe(5);
    });

    it('should update metadata after merge', () => {
      const ds1 = createMockDataset(3);
      const ds2 = createMockDataset(2);
      ds2.examples = ds2.examples.map((ex, i) => ({ ...ex, id: `ds2-ex-${i}` }));

      const merged = mergeDatasets(ds1, ds2);

      expect(merged.metadata.size).toBe(5);
    });
  });

  describe('splitRewardDataset', () => {
    it('should split dataset with default ratios', () => {
      const dataset = createMockDataset(100);

      const { train, val, test } = splitRewardDataset(dataset);

      expect(train.examples.length).toBe(80);
      expect(val.examples.length).toBe(10);
      expect(test.examples.length).toBe(10);
    });

    it('should split with custom ratios', () => {
      const dataset = createMockDataset(100);

      const { train, val, test } = splitRewardDataset(dataset, 0.6, 0.2);

      expect(train.examples.length).toBe(60);
      expect(val.examples.length).toBe(20);
      expect(test.examples.length).toBe(20);
    });

    it('should not lose any examples during split', () => {
      const dataset = createMockDataset(100);

      const { train, val, test } = splitRewardDataset(dataset);

      const totalAfterSplit = train.examples.length + val.examples.length + test.examples.length;
      expect(totalAfterSplit).toBe(100);
    });

    it('should update statistics for each split', () => {
      const dataset = createMockDataset(100);

      const { train, val, test } = splitRewardDataset(dataset);

      expect(train.statistics.totalExamples).toBe(train.examples.length);
      expect(val.statistics.totalExamples).toBe(val.examples.length);
      expect(test.statistics.totalExamples).toBe(test.examples.length);
    });
  });

  describe('filterByWeight', () => {
    it('should filter examples by minimum weight', () => {
      const dataset = createMockDataset(10);

      const filtered = filterByWeight(dataset, 0.7);

      for (const example of filtered.examples) {
        expect(example.weight).toBeGreaterThanOrEqual(0.7);
      }
    });

    it('should update statistics after filtering', () => {
      const dataset = createMockDataset(10);

      const filtered = filterByWeight(dataset, 0.7);

      expect(filtered.statistics.totalExamples).toBe(filtered.examples.length);
    });

    it('should preserve filter in metadata', () => {
      const dataset = createMockDataset(10);

      const filtered = filterByWeight(dataset, 0.7);

      expect(filtered.metadata.filters?.minWeight).toBe(0.7);
    });
  });

  describe('getDatasetSummary', () => {
    it('should return a formatted summary string', () => {
      const dataset = createMockDataset(50);

      const summary = getDatasetSummary(dataset);

      expect(summary).toContain('Reward Dataset Summary');
      expect(summary).toContain('Total Examples: 50');
      expect(summary).toContain('Version:');
      expect(summary).toContain('Score Statistics:');
      expect(summary).toContain('Category Distribution:');
      expect(summary).toContain('Feature Statistics:');
    });

    it('should include embedding dimension', () => {
      const dataset = createMockDataset(10);

      const summary = getDatasetSummary(dataset);

      expect(summary).toContain('Embedding Dimension: 3');
    });

    it('should include date range', () => {
      const dataset = createMockDataset(10);

      const summary = getDatasetSummary(dataset);

      expect(summary).toContain('Date Range:');
      expect(summary).toContain('Earliest:');
      expect(summary).toContain('Latest:');
    });
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases (DIRECTIVE-054)', () => {
  describe('Empty Dataset Handling', () => {
    it('should handle empty feedback list gracefully', async () => {
      const dataset = await buildRewardDataset();

      // Should not throw
      expect(dataset).toBeDefined();
      expect(dataset.examples).toBeDefined();
    });

    it('should export empty dataset without errors', async () => {
      const emptyDataset: RewardDataset = {
        examples: [],
        statistics: {
          totalExamples: 0,
          avgScore: 0,
          scoreDistribution: {},
          categoryDistribution: {},
          mutationDistribution: {},
          avgTokenCount: 0,
          avgSimilarity: 0,
          dateRange: { earliest: null, latest: null },
        },
        metadata: {
          created: new Date(),
          version: '1.0.0',
          size: 0,
          embeddingDimension: 384,
          featureCount: 0,
        },
      };

      const result = await exportDataset(emptyDataset, 'json');

      expect(result.format).toBe('json');
      expect(() => JSON.parse(result.data)).not.toThrow();
    });
  });

  describe('Special Characters in Content', () => {
    it('should handle prompts with special characters', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'p1',
        variationId: 'v1',
        score: 4,
        userId: 'u1',
        originalPrompt: 'Write a function to handle { "json": "data" }',
        variationText: 'Try to write a function that handles {"json": "data"}',
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(true);
    });

    it('should handle Unicode content', () => {
      const feedback: EnhancedFeedback = {
        promptId: 'p1',
        variationId: 'v1',
        score: 4,
        userId: 'u1',
        originalPrompt: 'اكتب دالة لمعالجة البيانات',
        variationText: 'حاول كتابة دالة لمعالجة البيانات',
        category: PromptCategory.CODE_GENERATION,
      };

      const result = validateFeedback(feedback);

      expect(result.isValid).toBe(true);
    });
  });

  describe('Large Dataset Performance', () => {
    it('should handle splitting large datasets', () => {
      const largeDataset: RewardDataset = {
        examples: Array.from({ length: 1000 }, (_, i) => ({
          id: `ex-${i}`,
          promptEmbedding: Array(384).fill(0.1),
          variationEmbedding: Array(384).fill(0.2),
          features: Array(12).fill(0.5),
          featureNames: Array(12).fill('feature'),
          label: Math.random(),
          weight: Math.random() * 0.5 + 0.5,
          metadata: {
            originalPrompt: `Prompt ${i}`,
            variationText: `Variation ${i}`,
            category: PromptCategory.CODE_GENERATION,
            mutationType: 'test',
            rawScore: Math.floor(Math.random() * 5) + 1,
            timestamp: new Date(),
          },
        })),
        statistics: {
          totalExamples: 1000,
          avgScore: 3,
          scoreDistribution: {},
          categoryDistribution: {},
          mutationDistribution: {},
          avgTokenCount: 10,
          avgSimilarity: 0.5,
          dateRange: { earliest: new Date(), latest: new Date() },
        },
        metadata: {
          created: new Date(),
          version: '1.0.0',
          size: 1000,
          embeddingDimension: 384,
          featureCount: 12,
        },
      };

      const start = Date.now();
      const { train, val, test } = splitRewardDataset(largeDataset);
      const duration = Date.now() - start;

      expect(train.examples.length + val.examples.length + test.examples.length).toBe(1000);
      expect(duration).toBeLessThan(1000); // Should complete in less than 1 second
    });
  });
});
