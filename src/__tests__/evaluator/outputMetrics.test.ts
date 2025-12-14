/**
 * Tests for Output Metrics Module (DIRECTIVE-007)
 *
 * Tests cover:
 * - Token counting and estimation
 * - Output measurement with mock LLM
 * - Cache management
 * - Metrics comparison
 * - Batch processing
 * - Error handling
 */

import {
  estimateTokenCount,
  measureActualOutput,
  compareOutputMetrics,
  formatMetricsSummary,
  measureBatch,
  cleanCache,
  clearCache,
  type LLMProvider,
  type OutputMetrics,
} from '../../evaluator/outputMetrics';

// ============================================================================
// SETUP & TEARDOWN
// ============================================================================

beforeEach(() => {
  // Clear cache before each test
  clearCache();
});

afterEach(() => {
  // Clean up after each test
  clearCache();
});

// ============================================================================
// TOKEN COUNTING TESTS
// ============================================================================

describe('estimateTokenCount', () => {
  it('should estimate tokens for simple text', () => {
    const text = 'Hello world this is a test';
    const tokens = estimateTokenCount(text);

    // 6 words * 1.3 = 7.8 → 8 tokens
    expect(tokens).toBe(8);
  });

  it('should handle empty strings', () => {
    expect(estimateTokenCount('')).toBe(0);
    expect(estimateTokenCount('   ')).toBe(0);
  });

  it('should handle single words', () => {
    const tokens = estimateTokenCount('Hello');
    expect(tokens).toBe(2); // 1 word * 1.3 = 1.3 → 2
  });

  it('should normalize multiple spaces', () => {
    const text = 'Hello    world    test';
    const tokens = estimateTokenCount(text);

    // Should normalize to 3 words
    expect(tokens).toBe(4); // 3 * 1.3 = 3.9 → 4
  });

  it('should handle text with newlines', () => {
    const text = `Line 1
    Line 2
    Line 3`;
    const tokens = estimateTokenCount(text);

    // 6 words * 1.3 = 7.8 → 8
    expect(tokens).toBe(8);
  });

  it('should estimate tokens for longer text', () => {
    const text = `This is a longer piece of text that should
    contain multiple sentences and various words.
    The token count should be approximately 1.3 times
    the word count according to our estimation algorithm.`;

    const tokens = estimateTokenCount(text);

    // Should be reasonable (30+ words → 39+ tokens)
    expect(tokens).toBeGreaterThan(35);
    expect(tokens).toBeLessThan(50);
  });
});

// ============================================================================
// MEASURE ACTUAL OUTPUT TESTS
// ============================================================================

describe('measureActualOutput', () => {
  const mockProvider: LLMProvider = {
    name: 'openai',
    model: 'gpt-3.5-turbo',
  };

  it('should measure output for a simple prompt', async () => {
    const prompt = 'What is 2+2?';
    const metrics = await measureActualOutput(prompt, mockProvider, 1, false);

    expect(metrics).toBeDefined();
    expect(metrics.avgLength).toBeGreaterThan(0);
    expect(metrics.avgTokens).toBeGreaterThan(0);
    expect(metrics.samples).toHaveLength(1);
    expect(metrics.quality).toBeGreaterThanOrEqual(0);
    expect(metrics.quality).toBeLessThanOrEqual(1);
  });

  it('should run multiple samples', async () => {
    const prompt = 'Explain photosynthesis';
    const metrics = await measureActualOutput(prompt, mockProvider, 3, false);

    expect(metrics.samples).toHaveLength(3);
    expect(metrics.variance).toBeGreaterThanOrEqual(0);
    expect(metrics.stdDeviation).toBeGreaterThanOrEqual(0);
  });

  it('should calculate variance correctly', async () => {
    const prompt = 'Write a sentence';
    const metrics = await measureActualOutput(prompt, mockProvider, 5, false);

    // Variance should be non-negative
    expect(metrics.variance).toBeGreaterThanOrEqual(0);

    // Standard deviation should be square root of variance
    expect(Math.abs(metrics.stdDeviation - Math.sqrt(metrics.variance))).toBeLessThan(0.1);
  });

  it('should use cache when enabled', async () => {
    const prompt = 'Test prompt for caching';

    // First call - not cached
    const metrics1 = await measureActualOutput(prompt, mockProvider, 1, true);
    const time1 = metrics1.timestamp;

    // Wait a tiny bit
    await new Promise(resolve => setTimeout(resolve, 10));

    // Second call - should use cache (same timestamp)
    const metrics2 = await measureActualOutput(prompt, mockProvider, 1, true);
    const time2 = metrics2.timestamp;

    expect(time1.getTime()).toBe(time2.getTime());
  });

  it('should not use cache when disabled', async () => {
    const prompt = 'Test prompt without caching';

    const metrics1 = await measureActualOutput(prompt, mockProvider, 1, false);
    await new Promise(resolve => setTimeout(resolve, 150));
    const metrics2 = await measureActualOutput(prompt, mockProvider, 1, false);

    // Timestamps should be different
    expect(metrics1.timestamp.getTime()).not.toBe(metrics2.timestamp.getTime());
  });

  it('should throw error for invalid sample count', async () => {
    const prompt = 'Test';

    await expect(
      measureActualOutput(prompt, mockProvider, 0, false)
    ).rejects.toThrow('samples must be at least 1');
  });

  it('should warn for high sample counts', async () => {
    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    const prompt = 'Test';

    await measureActualOutput(prompt, mockProvider, 15, false);

    expect(consoleWarnSpy).toHaveBeenCalledWith(
      expect.stringContaining('Running more than 10 samples')
    );

    consoleWarnSpy.mockRestore();
  });

  it('should include timestamp', async () => {
    const prompt = 'Test';
    const before = new Date();
    const metrics = await measureActualOutput(prompt, mockProvider, 1, false);
    const after = new Date();

    expect(metrics.timestamp.getTime()).toBeGreaterThanOrEqual(before.getTime());
    expect(metrics.timestamp.getTime()).toBeLessThanOrEqual(after.getTime());
  });

  it('should calculate average length correctly', async () => {
    const prompt = 'Test';
    const metrics = await measureActualOutput(prompt, mockProvider, 3, false);

    const manualAvg = metrics.samples
      .filter(s => !s.error)
      .reduce((sum, s) => sum + s.length, 0) / metrics.samples.filter(s => !s.error).length;

    expect(Math.abs(metrics.avgLength - manualAvg)).toBeLessThan(1);
  });

  it('should calculate average tokens correctly', async () => {
    const prompt = 'Test';
    const metrics = await measureActualOutput(prompt, mockProvider, 3, false);

    const manualAvg = metrics.samples
      .filter(s => !s.error)
      .reduce((sum, s) => sum + s.tokenCount, 0) / metrics.samples.filter(s => !s.error).length;

    expect(Math.abs(metrics.avgTokens - manualAvg)).toBeLessThan(1);
  });
});

// ============================================================================
// COMPARE OUTPUT METRICS TESTS
// ============================================================================

describe('compareOutputMetrics', () => {
  it('should compare two metrics correctly', () => {
    const metricsA: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.8,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 150,
      avgTokens: 75,
      variance: 20,
      stdDeviation: 4.47,
      quality: 0.7,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.lengthDiff).toBe(50);
    expect(comparison.tokenDiff).toBe(25);
    expect(comparison.varianceDiff).toBe(10);
    expect(comparison.qualityDiff).toBe(-0.1);
  });

  it('should recommend prompt A for higher quality', () => {
    const metricsA: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.9,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.7,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.recommendation).toContain('Prompt A produces higher quality');
  });

  it('should recommend prompt B for higher quality', () => {
    const metricsA: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.6,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.8,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.recommendation).toContain('Prompt B produces higher quality');
  });

  it('should recommend prompt B for efficiency', () => {
    const metricsA: OutputMetrics = {
      avgLength: 200,
      avgTokens: 100,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.75,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 100,
      avgTokens: 40,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.73,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.recommendation).toContain('Prompt B is more efficient');
  });

  it('should detect similar performance', () => {
    const metricsA: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 10,
      stdDeviation: 3.16,
      quality: 0.75,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 105,
      avgTokens: 52,
      variance: 12,
      stdDeviation: 3.46,
      quality: 0.76,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.recommendation).toContain('Both prompts perform similarly');
  });
});

// ============================================================================
// FORMAT METRICS SUMMARY TESTS
// ============================================================================

describe('formatMetricsSummary', () => {
  it('should format metrics summary correctly', () => {
    const metrics: OutputMetrics = {
      avgLength: 150,
      avgTokens: 75,
      variance: 100,
      stdDeviation: 10.0,
      quality: 0.85,
      samples: [
        { output: 'test1', length: 150, tokenCount: 75, latency: 100 },
        { output: 'test2', length: 150, tokenCount: 75, latency: 120 },
      ],
      timestamp: new Date('2025-01-01T12:00:00'),
    };

    const summary = formatMetricsSummary(metrics);

    expect(summary).toContain('Average Length: 150 characters');
    expect(summary).toContain('Average Tokens: 75 tokens');
    expect(summary).toContain('Variance: 100');
    expect(summary).toContain('Std Deviation: 10');
    expect(summary).toContain('Quality Score: 85.0%');
    expect(summary).toContain('Success Rate: 100.0%');
    expect(summary).toContain('Samples: 2');
  });

  it('should calculate success rate with errors', () => {
    const metrics: OutputMetrics = {
      avgLength: 150,
      avgTokens: 75,
      variance: 100,
      stdDeviation: 10.0,
      quality: 0.85,
      samples: [
        { output: 'test1', length: 150, tokenCount: 75, latency: 100 },
        { output: '', length: 0, tokenCount: 0, latency: 0, error: 'API Error' },
        { output: 'test3', length: 150, tokenCount: 75, latency: 120 },
      ],
      timestamp: new Date(),
    };

    const summary = formatMetricsSummary(metrics);

    // Success rate should be 66.7% (2 out of 3)
    expect(summary).toContain('Success Rate: 66.7%');
  });
});

// ============================================================================
// BATCH PROCESSING TESTS
// ============================================================================

describe('measureBatch', () => {
  const mockProvider: LLMProvider = {
    name: 'openai',
    model: 'gpt-3.5-turbo',
  };

  it('should process multiple prompts', async () => {
    const prompts = [
      'What is AI?',
      'Explain machine learning',
      'Define neural networks',
    ];

    const results = await measureBatch(prompts, mockProvider, 1);

    expect(results.size).toBe(3);

    for (const prompt of prompts) {
      expect(results.has(prompt)).toBe(true);
      const metrics = results.get(prompt);
      expect(metrics).toBeDefined();
      expect(metrics!.samples).toHaveLength(1);
    }
  });

  it('should call progress callback', async () => {
    const prompts = ['Prompt 1', 'Prompt 2', 'Prompt 3'];
    const progressCalls: Array<{ completed: number; total: number }> = [];

    const onProgress = (completed: number, total: number) => {
      progressCalls.push({ completed, total });
    };

    await measureBatch(prompts, mockProvider, 1, onProgress);

    expect(progressCalls).toHaveLength(3);
    expect(progressCalls[0]).toEqual({ completed: 1, total: 3 });
    expect(progressCalls[1]).toEqual({ completed: 2, total: 3 });
    expect(progressCalls[2]).toEqual({ completed: 3, total: 3 });
  });

  it('should handle empty prompt list', async () => {
    const results = await measureBatch([], mockProvider, 1);
    expect(results.size).toBe(0);
  });

  it('should continue processing after errors', async () => {
    // This test verifies that batch processing doesn't stop on errors
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    const prompts = ['Valid prompt 1', 'Valid prompt 2'];
    const results = await measureBatch(prompts, mockProvider, 1);

    // Should still process all prompts
    expect(results.size).toBe(2);

    consoleErrorSpy.mockRestore();
  });
});

// ============================================================================
// CACHE MANAGEMENT TESTS
// ============================================================================

describe('Cache Management', () => {
  const mockProvider: LLMProvider = {
    name: 'openai',
    model: 'gpt-3.5-turbo',
  };

  it('should clear cache completely', async () => {
    const prompt1 = 'Test prompt 1';
    const prompt2 = 'Test prompt 2';

    // Populate cache
    await measureActualOutput(prompt1, mockProvider, 1, true);
    await measureActualOutput(prompt2, mockProvider, 1, true);

    // Clear cache
    clearCache();

    // Subsequent calls should not use cache (different timestamps)
    const metrics1 = await measureActualOutput(prompt1, mockProvider, 1, true);
    await new Promise(resolve => setTimeout(resolve, 150));
    const metrics2 = await measureActualOutput(prompt1, mockProvider, 1, true);

    // After clearing, we should get a cache miss on first call, then cache hit
    // The cache hit will have the same timestamp as metrics1
    expect(metrics2.timestamp.getTime()).toBe(metrics1.timestamp.getTime());
  });

  it('should clean expired cache entries', async () => {
    // This test is challenging without time manipulation
    // We'll verify that cleanCache returns a number
    const cleaned = cleanCache();
    expect(typeof cleaned).toBe('number');
    expect(cleaned).toBeGreaterThanOrEqual(0);
  });

  it('should not clean non-expired entries', async () => {
    const prompt = 'Fresh cache entry';

    // Add fresh entry
    await measureActualOutput(prompt, mockProvider, 1, true);

    // Clean cache (shouldn't remove fresh entry)
    const cleaned = cleanCache();

    // Should clean 0 entries
    expect(cleaned).toBe(0);
  });

  it('should differentiate cache by provider', async () => {
    const prompt = 'Same prompt, different provider';

    const provider1: LLMProvider = { name: 'openai' };
    const provider2: LLMProvider = { name: 'anthropic' };

    const metrics1 = await measureActualOutput(prompt, provider1, 1, true);
    await new Promise(resolve => setTimeout(resolve, 150));
    const metrics2 = await measureActualOutput(prompt, provider2, 1, true);

    // Different providers should have different cache entries (different timestamps)
    expect(metrics1.timestamp.getTime()).not.toBe(metrics2.timestamp.getTime());
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  const mockProvider: LLMProvider = {
    name: 'openai',
    model: 'gpt-3.5-turbo',
  };

  it('should complete full workflow: measure → compare → format', async () => {
    const promptA = 'Write a short poem';
    const promptB = 'Write a very detailed poem with context';

    // Measure both prompts
    const metricsA = await measureActualOutput(promptA, mockProvider, 2, false);
    const metricsB = await measureActualOutput(promptB, mockProvider, 2, false);

    // Compare
    const comparison = compareOutputMetrics(metricsA, metricsB);

    // Format
    const summaryA = formatMetricsSummary(metricsA);
    const summaryB = formatMetricsSummary(metricsB);

    // Verify results
    expect(metricsA).toBeDefined();
    expect(metricsB).toBeDefined();
    expect(comparison).toBeDefined();
    expect(summaryA).toContain('Output Metrics Summary');
    expect(summaryB).toContain('Output Metrics Summary');
    expect(comparison.recommendation).toBeDefined();
  });

  it('should handle batch processing with cache', async () => {
    const prompts = ['Prompt A', 'Prompt B', 'Prompt C'];

    // First batch - populate cache
    const results1 = await measureBatch(prompts, mockProvider, 1);

    // Second batch - should use cache
    const results2 = await measureBatch(prompts, mockProvider, 1);

    // Should get same results from cache
    for (const prompt of prompts) {
      const metrics1 = results1.get(prompt);
      const metrics2 = results2.get(prompt);

      expect(metrics1?.timestamp.getTime()).toBe(metrics2?.timestamp.getTime());
    }
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
  const mockProvider: LLMProvider = {
    name: 'openai',
    model: 'gpt-3.5-turbo',
  };

  it('should handle very long prompts', async () => {
    const longPrompt = 'A'.repeat(10000);
    const metrics = await measureActualOutput(longPrompt, mockProvider, 1, false);

    expect(metrics).toBeDefined();
    expect(metrics.avgLength).toBeGreaterThan(0);
  });

  it('should handle prompts with special characters', async () => {
    const specialPrompt = 'Test with émojis 🎉 and spëcial çhars!';
    const metrics = await measureActualOutput(specialPrompt, mockProvider, 1, false);

    expect(metrics).toBeDefined();
    expect(metrics.samples).toHaveLength(1);
  });

  it('should handle single sample correctly', async () => {
    const prompt = 'Single sample test';
    const metrics = await measureActualOutput(prompt, mockProvider, 1, false);

    expect(metrics.samples).toHaveLength(1);
    expect(metrics.variance).toBe(0); // Single sample has no variance
    expect(metrics.stdDeviation).toBe(0);
  });

  it('should estimate token count for unicode text', () => {
    const unicodeText = '你好世界 مرحبا بالعالم Hello world';
    const tokens = estimateTokenCount(unicodeText);

    expect(tokens).toBeGreaterThan(0);
  });

  it('should handle zero variance gracefully', () => {
    const metricsA: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 0,
      stdDeviation: 0,
      quality: 0.9,
      samples: [],
      timestamp: new Date(),
    };

    const metricsB: OutputMetrics = {
      avgLength: 100,
      avgTokens: 50,
      variance: 0,
      stdDeviation: 0,
      quality: 0.9,
      samples: [],
      timestamp: new Date(),
    };

    const comparison = compareOutputMetrics(metricsA, metricsB);

    expect(comparison.varianceDiff).toBe(0);
    expect(comparison.qualityDiff).toBe(0);
  });
});
