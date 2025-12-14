/**
 * Tests for Hallucination Detection Module (DIRECTIVE-012)
 *
 * Tests cover:
 * - Consistency checking across multiple runs
 * - Fact verification against context
 * - Confidence scoring with logprobs
 * - Combined detection strategies
 * - Batch processing
 * - Score comparison and severity levels
 */

import {
  detectHallucination,
  isHallucination,
  getHallucinationSeverity,
  formatHallucinationScore,
  detectHallucinationBatch,
  compareHallucinationScores,
  type LLMProvider,
  type HallucinationScore,
  type DetectionConfig,
} from '../../evaluator/hallucinationDetector';

// ============================================================================
// MOCK PROVIDER
// ============================================================================

const mockProvider: LLMProvider = {
  name: 'openai',
  model: 'gpt-4',
  supportsLogprobs: true,
};

const mockProviderNoLogprobs: LLMProvider = {
  name: 'anthropic',
  model: 'claude-3',
  supportsLogprobs: false,
};

// ============================================================================
// HALLUCINATION DETECTION TESTS
// ============================================================================

describe('detectHallucination', () => {
  describe('Basic Detection', () => {
    it('should detect hallucination with all strategies', async () => {
      const prompt = 'What is the capital of France?';
      const output = 'Paris is the capital of France and has 2 million people';
      const context = 'Paris is the capital city of France';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 2,
        checkFacts: true,
        useLogprobs: true,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        context,
        config
      );

      expect(result).toBeDefined();
      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
      expect(result.method).toBeDefined();
      expect(result.details).toBeDefined();
    });

    it('should return valid score range', async () => {
      const prompt = 'Test prompt';
      const output = 'Test output';

      const result = await detectHallucination(prompt, output, mockProvider);

      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
    });

    it('should include details breakdown', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const result = await detectHallucination(prompt, output, mockProvider);

      expect(result.details.consistencyScore).toBeGreaterThanOrEqual(0);
      expect(result.details.factualityScore).toBeGreaterThanOrEqual(0);
      expect(result.details.confidenceScore).toBeGreaterThanOrEqual(0);
      expect(result.details.claimsChecked).toBeGreaterThanOrEqual(0);
      expect(result.details.claimsFailed).toBeGreaterThanOrEqual(0);
    });

    it('should use default config when not provided', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const result = await detectHallucination(prompt, output, mockProvider);

      expect(result).toBeDefined();
      expect(result.method).toBeDefined();
    });
  });

  describe('Consistency Strategy', () => {
    it('should run consistency check with multiple runs', async () => {
      const prompt = 'What is AI?';
      const output = 'AI stands for Artificial Intelligence';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 3,
        checkFacts: false,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      expect(result.method).toContain('consistency');
      expect(result.details.consistencyScore).toBeGreaterThanOrEqual(0);
    });

    it('should skip consistency check with single run', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: false,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      expect(result.details.consistencyScore).toBe(0);
    });

    it('should detect inconsistencies', async () => {
      const prompt = 'Generate random text';
      const output = 'Random output here';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 2,
        checkFacts: false,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      // May or may not find inconsistencies depending on mock variation
      expect(result.inconsistencies).toBeDefined();
      expect(Array.isArray(result.inconsistencies)).toBe(true);
    });
  });

  describe('Fact Verification Strategy', () => {
    it('should verify facts against context', async () => {
      const prompt = 'Tell me about Paris';
      const output = 'Paris is the capital of France and has 12 million people';
      const context = 'Paris is the capital city of France';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: true,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        context,
        config
      );

      expect(result.method).toContain('factuality');
      expect(result.details.factualityScore).toBeGreaterThanOrEqual(0);
    });

    it('should detect unsupported claims', async () => {
      const prompt = 'Test';
      const output = 'The moon is 500000 km away and is made of cheese';
      const context = 'The moon orbits Earth';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: true,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        context,
        config
      );

      // Should detect unsupported claims
      expect(result.details.factualityScore).toBeGreaterThan(0);
      expect(result.details.claimsChecked).toBeGreaterThan(0);
    });

    it('should skip fact check without context', async () => {
      const prompt = 'Test';
      const output = 'Test output with claims';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: true,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      // Without context, factuality score should be 0
      expect(result.details.factualityScore).toBe(0);
    });

    it('should handle output without factual claims', async () => {
      const prompt = 'Say hello';
      const output = 'Hello there';
      const context = 'Greetings context';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: true,
        useLogprobs: false,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        context,
        config
      );

      // No factual claims to check
      expect(result.details.claimsChecked).toBe(0);
      expect(result.details.factualityScore).toBe(0);
    });
  });

  describe('Confidence Scoring Strategy', () => {
    it('should use logprobs when available', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: false,
        useLogprobs: true,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      expect(result.method).toContain('confidence');
      expect(result.details.confidenceScore).toBeGreaterThanOrEqual(0);
    });

    it('should skip logprobs when provider does not support', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: false,
        useLogprobs: true,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProviderNoLogprobs,
        undefined,
        config
      );

      expect(result.details.confidenceScore).toBe(0);
    });

    it('should detect low confidence', async () => {
      const prompt = 'Test uncertain prompt';
      const output = 'Test uncertain output';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 1,
        checkFacts: false,
        useLogprobs: true,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config
      );

      // Mock may generate low confidence
      expect(result.details.confidenceScore).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Combined Strategies', () => {
    it('should combine all strategies with proper weights', async () => {
      const prompt = 'Complex test';
      const output = 'Paris is 100 million years old';
      const context = 'Paris is a city in France';

      const config: Partial<DetectionConfig> = {
        consistencyRuns: 2,
        checkFacts: true,
        useLogprobs: true,
      };

      const result = await detectHallucination(
        prompt,
        output,
        mockProvider,
        context,
        config
      );

      // All three strategies should contribute
      expect(result.method.split(', ')).toHaveLength(3);
      expect(result.confidence).toBeGreaterThan(0);
    });

    it('should calculate detection confidence based on strategies used', async () => {
      const prompt = 'Test';
      const output = 'Test output';

      const config1: Partial<DetectionConfig> = {
        consistencyRuns: 2,
        checkFacts: false,
        useLogprobs: false,
      };

      const result1 = await detectHallucination(
        prompt,
        output,
        mockProvider,
        undefined,
        config1
      );

      // Only 1 strategy: confidence = 1/3 = 0.333
      expect(result1.confidence).toBeCloseTo(0.333, 2);

      const config2: Partial<DetectionConfig> = {
        consistencyRuns: 2,
        checkFacts: true,
        useLogprobs: true,
      };

      const result2 = await detectHallucination(
        prompt,
        output,
        mockProvider,
        'context',
        config2
      );

      // All 3 strategies: confidence = 3/3 = 1.0
      expect(result2.confidence).toBe(1.0);
    });
  });
});

// ============================================================================
// HALLUCINATION THRESHOLD TESTS
// ============================================================================

describe('isHallucination', () => {
  it('should detect hallucination above threshold', () => {
    const score: HallucinationScore = {
      score: 0.7,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.7,
        factualityScore: 0.7,
        confidenceScore: 0.7,
        claimsChecked: 3,
        claimsFailed: 2,
      },
    };

    expect(isHallucination(score)).toBe(true);
    expect(isHallucination(score, 0.5)).toBe(true);
    expect(isHallucination(score, 0.8)).toBe(false);
  });

  it('should not detect hallucination below threshold', () => {
    const score: HallucinationScore = {
      score: 0.3,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.3,
        factualityScore: 0.3,
        confidenceScore: 0.3,
        claimsChecked: 3,
        claimsFailed: 0,
      },
    };

    expect(isHallucination(score)).toBe(false);
    expect(isHallucination(score, 0.5)).toBe(false);
    expect(isHallucination(score, 0.2)).toBe(true);
  });

  it('should handle edge cases at threshold', () => {
    const score: HallucinationScore = {
      score: 0.5,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.5,
        factualityScore: 0.5,
        confidenceScore: 0.5,
        claimsChecked: 2,
        claimsFailed: 1,
      },
    };

    expect(isHallucination(score, 0.5)).toBe(true); // >= threshold
    expect(isHallucination(score, 0.51)).toBe(false);
  });
});

// ============================================================================
// SEVERITY LEVEL TESTS
// ============================================================================

describe('getHallucinationSeverity', () => {
  it('should return "none" for very low scores', () => {
    expect(getHallucinationSeverity(0)).toBe('none');
    expect(getHallucinationSeverity(0.1)).toBe('none');
    expect(getHallucinationSeverity(0.19)).toBe('none');
  });

  it('should return "low" for low scores', () => {
    expect(getHallucinationSeverity(0.2)).toBe('low');
    expect(getHallucinationSeverity(0.3)).toBe('low');
    expect(getHallucinationSeverity(0.39)).toBe('low');
  });

  it('should return "medium" for medium scores', () => {
    expect(getHallucinationSeverity(0.4)).toBe('medium');
    expect(getHallucinationSeverity(0.5)).toBe('medium');
    expect(getHallucinationSeverity(0.69)).toBe('medium');
  });

  it('should return "high" for high scores', () => {
    expect(getHallucinationSeverity(0.7)).toBe('high');
    expect(getHallucinationSeverity(0.9)).toBe('high');
    expect(getHallucinationSeverity(1.0)).toBe('high');
  });
});

// ============================================================================
// FORMAT SCORE TESTS
// ============================================================================

describe('formatHallucinationScore', () => {
  it('should format score with all details', () => {
    const score: HallucinationScore = {
      score: 0.65,
      confidence: 0.9,
      inconsistencies: ['Issue 1', 'Issue 2'],
      method: 'consistency, factuality',
      details: {
        consistencyScore: 0.5,
        factualityScore: 0.7,
        confidenceScore: 0.8,
        claimsChecked: 5,
        claimsFailed: 3,
      },
    };

    const formatted = formatHallucinationScore(score);

    expect(formatted).toContain('Hallucination Detection Report');
    expect(formatted).toContain('Overall Score: 65.0%');
    expect(formatted).toContain('medium');
    expect(formatted).toContain('Detection Confidence: 90.0%');
    expect(formatted).toContain('consistency, factuality');
    expect(formatted).toContain('Consistency: 50.0%');
    expect(formatted).toContain('Factuality: 70.0%');
    expect(formatted).toContain('Total Claims: 5');
    expect(formatted).toContain('Failed Claims: 3');
    expect(formatted).toContain('Issue 1');
    expect(formatted).toContain('Issue 2');
  });

  it('should handle score without inconsistencies', () => {
    const score: HallucinationScore = {
      score: 0.2,
      confidence: 0.5,
      inconsistencies: [],
      method: 'consistency',
      details: {
        consistencyScore: 0.2,
        factualityScore: 0,
        confidenceScore: 0,
        claimsChecked: 0,
        claimsFailed: 0,
      },
    };

    const formatted = formatHallucinationScore(score);

    expect(formatted).toContain('Overall Score: 20.0%');
    expect(formatted).toContain('low');
    expect(formatted).not.toContain('Inconsistencies Found');
  });

  it('should show correct severity labels', () => {
    const scoreNone: HallucinationScore = {
      score: 0.1,
      confidence: 0.5,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0,
        factualityScore: 0,
        confidenceScore: 0,
        claimsChecked: 0,
        claimsFailed: 0,
      },
    };

    expect(formatHallucinationScore(scoreNone)).toContain('none');

    const scoreHigh: HallucinationScore = {
      score: 0.9,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0,
        factualityScore: 0,
        confidenceScore: 0,
        claimsChecked: 0,
        claimsFailed: 0,
      },
    };

    expect(formatHallucinationScore(scoreHigh)).toContain('high');
  });
});

// ============================================================================
// BATCH PROCESSING TESTS
// ============================================================================

describe('detectHallucinationBatch', () => {
  it('should process multiple outputs', async () => {
    const outputs = [
      {
        prompt: 'What is AI?',
        output: 'AI is artificial intelligence',
        context: 'AI stands for Artificial Intelligence',
      },
      {
        prompt: 'What is ML?',
        output: 'ML is machine learning',
        context: 'ML is a subset of AI',
      },
      {
        prompt: 'What is DL?',
        output: 'DL is deep learning',
      },
    ];

    const config: Partial<DetectionConfig> = {
      consistencyRuns: 1,
      checkFacts: true,
    };

    const results = await detectHallucinationBatch(
      outputs,
      mockProvider,
      config
    );

    expect(results).toHaveLength(3);
    results.forEach(result => {
      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
    });
  });

  it('should call progress callback', async () => {
    const outputs = [
      { prompt: 'Test 1', output: 'Output 1' },
      { prompt: 'Test 2', output: 'Output 2' },
      { prompt: 'Test 3', output: 'Output 3' },
    ];

    const progressCalls: Array<{ completed: number; total: number }> = [];

    const onProgress = (completed: number, total: number) => {
      progressCalls.push({ completed, total });
    };

    await detectHallucinationBatch(outputs, mockProvider, {}, onProgress);

    expect(progressCalls).toHaveLength(3);
    expect(progressCalls[0]).toEqual({ completed: 1, total: 3 });
    expect(progressCalls[1]).toEqual({ completed: 2, total: 3 });
    expect(progressCalls[2]).toEqual({ completed: 3, total: 3 });
  });

  it('should handle empty batch', async () => {
    const results = await detectHallucinationBatch([], mockProvider);
    expect(results).toHaveLength(0);
  });

  it('should process with different contexts', async () => {
    const outputs = [
      {
        prompt: 'Test 1',
        output: 'Output with context',
        context: 'Relevant context',
      },
      {
        prompt: 'Test 2',
        output: 'Output without context',
      },
    ];

    const results = await detectHallucinationBatch(outputs, mockProvider, {
      checkFacts: true,
    });

    expect(results).toHaveLength(2);
    // First should have fact checking, second should not
    expect(results[0]).toBeDefined();
    expect(results[1]).toBeDefined();
  });
});

// ============================================================================
// COMPARE SCORES TESTS
// ============================================================================

describe('compareHallucinationScores', () => {
  it('should identify better output (A)', () => {
    const scoreA: HallucinationScore = {
      score: 0.2,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.2,
        factualityScore: 0.2,
        confidenceScore: 0.2,
        claimsChecked: 2,
        claimsFailed: 0,
      },
    };

    const scoreB: HallucinationScore = {
      score: 0.7,
      confidence: 0.8,
      inconsistencies: ['Issue'],
      method: 'test',
      details: {
        consistencyScore: 0.7,
        factualityScore: 0.7,
        confidenceScore: 0.7,
        claimsChecked: 3,
        claimsFailed: 2,
      },
    };

    const comparison = compareHallucinationScores(scoreA, scoreB);

    expect(comparison.better).toBe('A');
    expect(comparison.scoreDiff).toBe(0.5);
    expect(comparison.recommendation).toContain('Output A is more reliable');
  });

  it('should identify better output (B)', () => {
    const scoreA: HallucinationScore = {
      score: 0.8,
      confidence: 0.7,
      inconsistencies: ['Issue 1', 'Issue 2'],
      method: 'test',
      details: {
        consistencyScore: 0.8,
        factualityScore: 0.8,
        confidenceScore: 0.8,
        claimsChecked: 4,
        claimsFailed: 3,
      },
    };

    const scoreB: HallucinationScore = {
      score: 0.3,
      confidence: 0.9,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.3,
        factualityScore: 0.3,
        confidenceScore: 0.3,
        claimsChecked: 2,
        claimsFailed: 0,
      },
    };

    const comparison = compareHallucinationScores(scoreA, scoreB);

    expect(comparison.better).toBe('B');
    expect(comparison.scoreDiff).toBe(-0.5);
    expect(comparison.recommendation).toContain('Output B is more reliable');
  });

  it('should identify tie', () => {
    const scoreA: HallucinationScore = {
      score: 0.5,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.5,
        factualityScore: 0.5,
        confidenceScore: 0.5,
        claimsChecked: 2,
        claimsFailed: 1,
      },
    };

    const scoreB: HallucinationScore = {
      score: 0.52,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.52,
        factualityScore: 0.52,
        confidenceScore: 0.52,
        claimsChecked: 2,
        claimsFailed: 1,
      },
    };

    const comparison = compareHallucinationScores(scoreA, scoreB);

    expect(comparison.better).toBe('tie');
    expect(Math.abs(comparison.scoreDiff)).toBeLessThan(0.1);
    expect(comparison.recommendation).toContain('similar hallucination risks');
  });

  it('should calculate score difference correctly', () => {
    const scoreA: HallucinationScore = {
      score: 0.4,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.4,
        factualityScore: 0.4,
        confidenceScore: 0.4,
        claimsChecked: 2,
        claimsFailed: 0,
      },
    };

    const scoreB: HallucinationScore = {
      score: 0.6,
      confidence: 0.8,
      inconsistencies: [],
      method: 'test',
      details: {
        consistencyScore: 0.6,
        factualityScore: 0.6,
        confidenceScore: 0.6,
        claimsChecked: 2,
        claimsFailed: 1,
      },
    };

    const comparison = compareHallucinationScores(scoreA, scoreB);

    expect(comparison.scoreDiff).toBe(0.2);
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
  it('should handle empty output', async () => {
    const prompt = 'Test';
    const output = '';

    const result = await detectHallucination(prompt, output, mockProvider);

    expect(result).toBeDefined();
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it('should handle very long output', async () => {
    const prompt = 'Test';
    const output = 'word '.repeat(1000);

    const result = await detectHallucination(prompt, output, mockProvider, undefined, {
      consistencyRuns: 1,
    });

    expect(result).toBeDefined();
  });

  it('should handle special characters', async () => {
    const prompt = 'Test';
    const output = 'Test with émojis 🎉 and spëcial çhars!';

    const result = await detectHallucination(prompt, output, mockProvider);

    expect(result).toBeDefined();
  });

  it('should handle output with no claims', async () => {
    const prompt = 'Say hello';
    const output = 'Hello';
    const context = 'Greeting context';

    const result = await detectHallucination(prompt, output, mockProvider, context, {
      checkFacts: true,
    });

    expect(result.details.claimsChecked).toBe(0);
    expect(result.details.claimsFailed).toBe(0);
  });

  it('should handle output with many claims', async () => {
    const prompt = 'Tell me facts';
    const output = `
      Paris is the capital of France.
      The population is 2.2 million.
      It has 20 arrondissements.
      The Eiffel Tower is 324 meters tall.
      The Louvre is the largest museum.
    `;
    const context = 'Paris is in France';

    const result = await detectHallucination(prompt, output, mockProvider, context, {
      checkFacts: true,
    });

    expect(result.details.claimsChecked).toBeGreaterThan(0);
  });

  it('should handle unicode text', async () => {
    const prompt = 'Test';
    const output = '你好世界 مرحبا بالعالم';

    const result = await detectHallucination(prompt, output, mockProvider);

    expect(result).toBeDefined();
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  it('should complete full workflow: detect → check → format', async () => {
    const prompt = 'What is the capital of France?';
    const output = 'Paris is the capital of France';
    const context = 'Paris is a city in France';

    const score = await detectHallucination(
      prompt,
      output,
      mockProvider,
      context
    );

    const isHall = isHallucination(score);
    const severity = getHallucinationSeverity(score.score);
    const formatted = formatHallucinationScore(score);

    expect(score).toBeDefined();
    expect(typeof isHall).toBe('boolean');
    expect(['none', 'low', 'medium', 'high']).toContain(severity);
    expect(formatted).toContain('Hallucination Detection Report');
  });

  it('should handle real-world scenario', async () => {
    const prompt = 'Explain quantum computing';
    const output = `
      Quantum computing uses qubits instead of bits.
      It was invented in 1985 by Richard Feynman.
      Quantum computers can break all encryption instantly.
    `;
    const context = `
      Quantum computing is a field that uses quantum mechanics.
      Qubits can be in superposition.
    `;

    const score = await detectHallucination(
      prompt,
      output,
      mockProvider,
      context,
      {
        consistencyRuns: 2,
        checkFacts: true,
      }
    );

    expect(score.details.claimsChecked).toBeGreaterThan(0);
    // Some claims not fully supported by context
    expect(score.score).toBeGreaterThanOrEqual(0);
  });

  it('should compare multiple outputs', async () => {
    const prompt = 'Test';
    const outputs = [
      'Accurate output based on facts',
      'Made up information not in context',
    ];
    const context = 'Accurate facts here';

    const scores = await detectHallucinationBatch(
      outputs.map(output => ({ prompt, output, context })),
      mockProvider,
      { checkFacts: true }
    );

    expect(scores).toHaveLength(2);

    const comparison = compareHallucinationScores(scores[0], scores[1]);
    expect(comparison.better).toBeDefined();
  });
});
