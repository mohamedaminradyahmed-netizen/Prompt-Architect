/**
 * Tests for Reference Metrics Module (DIRECTIVE-008, DIRECTIVE-009)
 *
 * Tests cover:
 * - ROUGE-1, ROUGE-2, ROUGE-L calculations
 * - BLEU score with single and multiple references
 * - Combined evaluation metrics
 * - Batch processing
 * - Output comparison
 * - Edge cases and error handling
 */

import {
  calculateROUGE,
  calculateBLEU,
  evaluateAgainstReference,
  formatReferenceMetrics,
  evaluateBatch,
  compareOutputs,
  type ROUGEScores,
  type BLEUScore,
  type ReferenceMetrics,
} from '../../evaluator/referenceMetrics';

// ============================================================================
// ROUGE TESTS
// ============================================================================

describe('calculateROUGE', () => {
  describe('ROUGE-1 (Unigram)', () => {
    it('should calculate perfect match', () => {
      const candidate = 'the cat sat on the mat';
      const reference = 'the cat sat on the mat';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.precision).toBe(1.0);
      expect(scores.rouge1.recall).toBe(1.0);
      expect(scores.rouge1.f1).toBe(1.0);
    });

    it('should calculate partial match', () => {
      const candidate = 'the cat sat';
      const reference = 'the cat sat on the mat';

      const scores = calculateROUGE(candidate, reference);

      // All 3 words in candidate are in reference
      expect(scores.rouge1.precision).toBe(1.0);
      // 3 out of 6 reference words are matched
      expect(scores.rouge1.recall).toBe(0.5);
      // F1 should be harmonic mean
      expect(scores.rouge1.f1).toBeCloseTo(0.667, 2);
    });

    it('should calculate no match', () => {
      const candidate = 'hello world';
      const reference = 'foo bar baz';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.precision).toBe(0);
      expect(scores.rouge1.recall).toBe(0);
      expect(scores.rouge1.f1).toBe(0);
    });

    it('should handle empty candidate', () => {
      const candidate = '';
      const reference = 'the cat sat on the mat';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.precision).toBe(0);
      expect(scores.rouge1.recall).toBe(0);
      expect(scores.rouge1.f1).toBe(0);
    });

    it('should handle empty reference', () => {
      const candidate = 'the cat sat';
      const reference = '';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.precision).toBe(0);
      expect(scores.rouge1.recall).toBe(0);
      expect(scores.rouge1.f1).toBe(0);
    });

    it('should be case insensitive', () => {
      const candidate = 'THE CAT SAT';
      const reference = 'the cat sat';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.f1).toBe(1.0);
    });

    it('should ignore punctuation', () => {
      const candidate = 'Hello, world!';
      const reference = 'Hello world';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge1.f1).toBe(1.0);
    });
  });

  describe('ROUGE-2 (Bigram)', () => {
    it('should calculate perfect bigram match', () => {
      const candidate = 'the cat sat';
      const reference = 'the cat sat';

      const scores = calculateROUGE(candidate, reference);

      // Bigrams: ['the cat', 'cat sat']
      expect(scores.rouge2.precision).toBe(1.0);
      expect(scores.rouge2.recall).toBe(1.0);
      expect(scores.rouge2.f1).toBe(1.0);
    });

    it('should calculate partial bigram match', () => {
      const candidate = 'the cat sat on the mat';
      const reference = 'the cat sat';

      const scores = calculateROUGE(candidate, reference);

      // Candidate bigrams: ['the cat', 'cat sat', 'sat on', 'on the', 'the mat'] = 5 unique
      // Reference bigrams: ['the cat', 'cat sat'] = 2
      // Overlap: 2 bigrams match

      // Precision: 2/5 = 0.4
      expect(scores.rouge2.precision).toBeCloseTo(0.4, 2);
      // Recall: 2/2 = 1.0
      expect(scores.rouge2.recall).toBe(1.0);
      // F1: 2 * (0.4 * 1.0) / (0.4 + 1.0) = 0.571
      expect(scores.rouge2.f1).toBeCloseTo(0.571, 2);
    });

    it('should handle no bigram overlap', () => {
      const candidate = 'hello world';
      const reference = 'foo bar baz';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rouge2.precision).toBe(0);
      expect(scores.rouge2.recall).toBe(0);
      expect(scores.rouge2.f1).toBe(0);
    });

    it('should handle single word texts', () => {
      const candidate = 'hello';
      const reference = 'world';

      const scores = calculateROUGE(candidate, reference);

      // No bigrams possible with single word
      expect(scores.rouge2.precision).toBe(0);
      expect(scores.rouge2.recall).toBe(0);
      expect(scores.rouge2.f1).toBe(0);
    });
  });

  describe('ROUGE-L (Longest Common Subsequence)', () => {
    it('should calculate perfect LCS match', () => {
      const candidate = 'the cat sat on the mat';
      const reference = 'the cat sat on the mat';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rougeL.precision).toBe(1.0);
      expect(scores.rougeL.recall).toBe(1.0);
      expect(scores.rougeL.f1).toBe(1.0);
    });

    it('should calculate LCS for reordered text', () => {
      const candidate = 'cat sat on mat';
      const reference = 'the cat sat on the mat';

      const scores = calculateROUGE(candidate, reference);

      // LCS: 'cat sat on mat' = 4 words
      // Candidate length: 4, Reference length: 6

      // Precision: 4/4 = 1.0
      expect(scores.rougeL.precision).toBe(1.0);
      // Recall: 4/6 = 0.667
      expect(scores.rougeL.recall).toBeCloseTo(0.667, 2);
      // F1: 2 * (1.0 * 0.667) / (1.0 + 0.667) = 0.8
      expect(scores.rougeL.f1).toBeCloseTo(0.8, 2);
    });

    it('should handle completely different sequences', () => {
      const candidate = 'hello world test';
      const reference = 'foo bar baz';

      const scores = calculateROUGE(candidate, reference);

      expect(scores.rougeL.precision).toBe(0);
      expect(scores.rougeL.recall).toBe(0);
      expect(scores.rougeL.f1).toBe(0);
    });

    it('should find LCS in mixed order', () => {
      const candidate = 'A B C D E';
      const reference = 'A C E';

      const scores = calculateROUGE(candidate, reference);

      // LCS: 'A C E' = 3 words
      // Candidate: 5 words, Reference: 3 words

      // Precision: 3/5 = 0.6
      expect(scores.rougeL.precision).toBe(0.6);
      // Recall: 3/3 = 1.0
      expect(scores.rougeL.recall).toBe(1.0);
      // F1
      expect(scores.rougeL.f1).toBe(0.75);
    });
  });

  describe('Real-world examples', () => {
    it('should evaluate summarization task', () => {
      const candidate = 'The quick brown fox jumps over the lazy dog';
      const reference = 'A fast brown fox leaps over a sleepy dog';

      const scores = calculateROUGE(candidate, reference);

      // Should have some overlap but not perfect
      expect(scores.rouge1.f1).toBeGreaterThan(0.3);
      expect(scores.rouge1.f1).toBeLessThan(0.7);
    });

    it('should evaluate paraphrase', () => {
      const candidate = 'Machine learning is a subset of AI';
      const reference = 'AI includes machine learning as a subcategory';

      const scores = calculateROUGE(candidate, reference);

      // Some word overlap
      expect(scores.rouge1.f1).toBeGreaterThan(0);
      expect(scores.rouge1.f1).toBeLessThan(1);
    });
  });
});

// ============================================================================
// BLEU TESTS
// ============================================================================

describe('calculateBLEU', () => {
  describe('Single Reference', () => {
    it('should calculate perfect BLEU score', () => {
      const candidate = 'the cat sat on the mat';
      const references = ['the cat sat on the mat'];

      const bleu = calculateBLEU(candidate, references);

      expect(bleu.score).toBe(1.0);
      expect(bleu.brevityPenalty).toBe(1.0);
      expect(bleu.precisions).toHaveLength(4);
      bleu.precisions.forEach(p => expect(p).toBe(1.0));
    });

    it('should penalize shorter candidates', () => {
      const candidate = 'the cat';
      const references = ['the cat sat on the mat'];

      const bleu = calculateBLEU(candidate, references);

      // Brevity penalty should be < 1.0
      expect(bleu.brevityPenalty).toBeLessThan(1.0);
      expect(bleu.brevityPenalty).toBeGreaterThan(0);

      // Overall score affected by brevity penalty
      expect(bleu.score).toBeLessThan(1.0);
    });

    it('should not penalize longer candidates', () => {
      const candidate = 'the cat sat on the mat and slept';
      const references = ['the cat sat on the mat'];

      const bleu = calculateBLEU(candidate, references);

      // Brevity penalty should be 1.0 (no penalty for longer)
      expect(bleu.brevityPenalty).toBe(1.0);
    });

    it('should calculate n-gram precisions', () => {
      const candidate = 'the cat sat on the mat';
      const references = ['the cat sat on the mat'];

      const bleu = calculateBLEU(candidate, references);

      // All n-grams should match perfectly
      expect(bleu.precisions[0]).toBe(1.0); // 1-gram
      expect(bleu.precisions[1]).toBe(1.0); // 2-gram
      expect(bleu.precisions[2]).toBe(1.0); // 3-gram
      expect(bleu.precisions[3]).toBe(1.0); // 4-gram
    });

    it('should handle partial matches', () => {
      const candidate = 'the cat sat';
      const references = ['the cat sat on the mat'];

      const bleu = calculateBLEU(candidate, references);

      // 1-gram precision should be high (all words in reference)
      expect(bleu.precisions[0]).toBe(1.0);

      // Higher order n-grams might be lower
      expect(bleu.precisions[2]).toBeGreaterThanOrEqual(0);
    });

    it('should handle no match', () => {
      const candidate = 'hello world';
      const references = ['foo bar baz'];

      const bleu = calculateBLEU(candidate, references);

      expect(bleu.score).toBe(0);
      bleu.precisions.forEach(p => expect(p).toBe(0));
    });

    it('should handle empty candidate', () => {
      const candidate = '';
      const references = ['the cat sat'];

      const bleu = calculateBLEU(candidate, references);

      expect(bleu.score).toBe(0);
      expect(bleu.brevityPenalty).toBe(0);
      expect(bleu.length.candidate).toBe(0);
    });
  });

  describe('Multiple References', () => {
    it('should use best matching reference', () => {
      const candidate = 'the cat sat on the mat';
      const references = [
        'the dog sat on the mat', // Close match
        'hello world', // Poor match
        'the cat sat on the mat', // Perfect match
      ];

      const bleu = calculateBLEU(candidate, references);

      // Should match perfectly with third reference
      expect(bleu.score).toBe(1.0);
    });

    it('should select closest reference length', () => {
      const candidate = 'the cat sat'; // 3 words
      const references = [
        'the cat', // 2 words
        'the cat sat on the', // 5 words
      ];

      const bleu = calculateBLEU(candidate, references);

      // Should use first reference (2 words) as closest to 3
      expect(bleu.length.reference).toBe(2);
    });

    it('should calculate max precision across references', () => {
      const candidate = 'the quick brown fox';
      const references = [
        'the fast brown fox', // Good match for 'brown fox'
        'the quick red fox', // Good match for 'the quick'
      ];

      const bleu = calculateBLEU(candidate, references);

      // Should take max precision from both references
      expect(bleu.score).toBeGreaterThan(0);
    });
  });

  describe('Custom n-gram sizes', () => {
    it('should support custom maxN', () => {
      const candidate = 'the cat sat';
      const references = ['the cat sat'];

      const bleu2 = calculateBLEU(candidate, references, 2);
      const bleu3 = calculateBLEU(candidate, references, 3);

      expect(bleu2.precisions).toHaveLength(2);
      expect(bleu3.precisions).toHaveLength(3);
    });
  });

  describe('Error Handling', () => {
    it('should throw error for empty references', () => {
      const candidate = 'the cat sat';
      const references: string[] = [];

      expect(() => calculateBLEU(candidate, references)).toThrow(
        'At least one reference is required'
      );
    });
  });

  describe('Real-world examples', () => {
    it('should evaluate translation quality', () => {
      const candidate = 'I am going to the store';
      const references = ['I am going to the shop', 'I will go to the store'];

      const bleu = calculateBLEU(candidate, references);

      // Should have good score due to overlap
      expect(bleu.score).toBeGreaterThan(0.5);
    });
  });
});

// ============================================================================
// COMBINED EVALUATION TESTS
// ============================================================================

describe('evaluateAgainstReference', () => {
  it('should combine ROUGE and BLEU scores', () => {
    const prompt = 'Translate this';
    const output = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const metrics = evaluateAgainstReference(prompt, output, references);

    expect(metrics.rouge).toBeDefined();
    expect(metrics.bleu).toBeDefined();
    expect(metrics.overallScore).toBeGreaterThan(0);
    expect(metrics.recommendation).toBeDefined();
  });

  it('should calculate overall score correctly', () => {
    const prompt = '';
    const output = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const metrics = evaluateAgainstReference(prompt, output, references);

    // Perfect match should give high overall score
    expect(metrics.overallScore).toBeGreaterThanOrEqual(95);
  });

  it('should provide excellent recommendation for high scores', () => {
    const prompt = '';
    const output = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const metrics = evaluateAgainstReference(prompt, output, references);

    expect(metrics.recommendation).toContain('Excellent');
  });

  it('should provide poor recommendation for low scores', () => {
    const prompt = '';
    const output = 'hello world test';
    const references = ['the cat sat on the mat'];

    const metrics = evaluateAgainstReference(prompt, output, references);

    expect(metrics.overallScore).toBeLessThan(30);
    expect(metrics.recommendation).toMatch(/Poor|Weak/);
  });

  it('should provide moderate recommendation for medium scores', () => {
    const prompt = '';
    const output = 'the cat sat';
    const references = ['the cat sat on the mat and slept soundly'];

    const metrics = evaluateAgainstReference(prompt, output, references);

    expect(metrics.overallScore).toBeGreaterThan(20);
    expect(metrics.overallScore).toBeLessThan(80);
    expect(metrics.recommendation).toMatch(/Good|Moderate/);
  });

  it('should throw error for empty references', () => {
    const prompt = '';
    const output = 'test';
    const references: string[] = [];

    expect(() => evaluateAgainstReference(prompt, output, references)).toThrow(
      'At least one reference output is required'
    );
  });
});

// ============================================================================
// FORMAT METRICS TESTS
// ============================================================================

describe('formatReferenceMetrics', () => {
  it('should format metrics as readable string', () => {
    const metrics: ReferenceMetrics = {
      rouge: {
        rouge1: { precision: 0.8, recall: 0.7, f1: 0.75 },
        rouge2: { precision: 0.6, recall: 0.5, f1: 0.55 },
        rougeL: { precision: 0.7, recall: 0.65, f1: 0.675 },
      },
      bleu: {
        score: 0.65,
        precisions: [0.8, 0.7, 0.6, 0.5],
        brevityPenalty: 0.95,
        length: { candidate: 10, reference: 12 },
      },
      overallScore: 67.5,
      recommendation: 'Good match',
    };

    const formatted = formatReferenceMetrics(metrics);

    expect(formatted).toContain('ROUGE-1');
    expect(formatted).toContain('ROUGE-2');
    expect(formatted).toContain('ROUGE-L');
    expect(formatted).toContain('BLEU');
    expect(formatted).toContain('Overall Score: 67.5');
    expect(formatted).toContain('Good match');
    expect(formatted).toContain('80.0%'); // ROUGE-1 precision
    expect(formatted).toContain('0.950'); // Brevity penalty
  });
});

// ============================================================================
// BATCH EVALUATION TESTS
// ============================================================================

describe('evaluateBatch', () => {
  it('should evaluate multiple outputs', () => {
    const outputs = [
      {
        prompt: 'Test 1',
        output: 'the cat sat',
        references: ['the cat sat on the mat'],
      },
      {
        prompt: 'Test 2',
        output: 'hello world',
        references: ['hello world test'],
      },
      {
        prompt: 'Test 3',
        output: 'foo bar',
        references: ['foo bar baz'],
      },
    ];

    const results = evaluateBatch(outputs);

    expect(results).toHaveLength(3);
    results.forEach(result => {
      expect(result.rouge).toBeDefined();
      expect(result.bleu).toBeDefined();
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
    });
  });

  it('should handle empty batch', () => {
    const results = evaluateBatch([]);
    expect(results).toHaveLength(0);
  });
});

// ============================================================================
// COMPARE OUTPUTS TESTS
// ============================================================================

describe('compareOutputs', () => {
  it('should compare two outputs', () => {
    const outputA = 'the cat sat on the mat';
    const outputB = 'the cat sat';
    const references = ['the cat sat on the mat'];

    const comparison = compareOutputs(outputA, outputB, references);

    expect(comparison.metricsA).toBeDefined();
    expect(comparison.metricsB).toBeDefined();
    expect(comparison.winner).toBe('A'); // A is perfect match
    expect(comparison.scoreDiff).toBeLessThan(0); // A scores higher
  });

  it('should detect tie for similar scores', () => {
    const outputA = 'the cat sat on the mat';
    const outputB = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const comparison = compareOutputs(outputA, outputB, references);

    expect(comparison.winner).toBe('tie');
    expect(Math.abs(comparison.scoreDiff)).toBeLessThan(5);
  });

  it('should identify winner B', () => {
    const outputA = 'hello world';
    const outputB = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const comparison = compareOutputs(outputA, outputB, references);

    expect(comparison.winner).toBe('B');
    expect(comparison.scoreDiff).toBeGreaterThan(5);
  });

  it('should calculate score difference correctly', () => {
    const outputA = 'the cat sat';
    const outputB = 'the cat sat on the mat';
    const references = ['the cat sat on the mat'];

    const comparison = compareOutputs(outputA, outputB, references);

    expect(comparison.scoreDiff).toBe(
      comparison.metricsB.overallScore - comparison.metricsA.overallScore
    );
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
  it('should handle very long texts', () => {
    const longText = 'word '.repeat(1000);
    const reference = 'word '.repeat(1000);

    const rouge = calculateROUGE(longText, reference);
    const bleu = calculateBLEU(longText, [reference]);

    expect(rouge.rouge1.f1).toBe(1.0);
    expect(bleu.score).toBe(1.0);
  });

  it('should handle unicode characters', () => {
    const candidate = '你好世界 مرحبا';
    const reference = '你好世界 مرحبا';

    const rouge = calculateROUGE(candidate, reference);

    expect(rouge.rouge1.f1).toBe(1.0);
  });

  it('should handle repeated words', () => {
    const candidate = 'the the the cat cat';
    const reference = 'the cat';

    const rouge = calculateROUGE(candidate, reference);

    // Should handle repeated words
    expect(rouge.rouge1.precision).toBeGreaterThan(0);
  });

  it('should handle special characters and numbers', () => {
    const candidate = 'test123 @#$ foo';
    const reference = 'test123 foo bar';

    const rouge = calculateROUGE(candidate, reference);

    // Punctuation removed, numbers treated as words
    expect(rouge.rouge1.precision).toBeGreaterThan(0);
  });

  it('should handle whitespace-only strings', () => {
    const candidate = '   ';
    const reference = '   ';

    const rouge = calculateROUGE(candidate, reference);

    expect(rouge.rouge1.f1).toBe(0);
  });

  it('should handle single character words', () => {
    const candidate = 'a b c d';
    const reference = 'a b c d e';

    const bleu = calculateBLEU(candidate, [reference]);

    expect(bleu.score).toBeGreaterThan(0);
  });

  it('should handle BLEU with very short candidate', () => {
    const candidate = 'a';
    const reference = 'a b c d e f g h i j';

    const bleu = calculateBLEU(candidate, [reference]);

    // High brevity penalty
    expect(bleu.brevityPenalty).toBeLessThan(0.5);
    expect(bleu.score).toBeLessThan(bleu.precisions[0]);
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  it('should complete full evaluation workflow', () => {
    const prompt = 'Translate: The weather is nice today';
    const output = 'The weather is beautiful today';
    const references = [
      'The weather is nice today',
      'Today the weather is nice',
      'It is nice weather today',
    ];

    const metrics = evaluateAgainstReference(prompt, output, references);
    const formatted = formatReferenceMetrics(metrics);

    expect(metrics.overallScore).toBeGreaterThan(0);
    expect(formatted).toContain('Reference Evaluation Metrics');
  });

  it('should handle real-world summarization task', () => {
    const prompt = 'Summarize the article';
    const output =
      'Machine learning is a branch of AI that enables computers to learn from data';
    const references = [
      'Machine learning, a subset of artificial intelligence, allows computers to learn patterns from data',
      'ML is an AI technique where computers learn from data without explicit programming',
    ];

    const metrics = evaluateAgainstReference(prompt, output, references);

    expect(metrics.rouge.rouge1.f1).toBeGreaterThan(0.3);
    expect(metrics.bleu.score).toBeGreaterThan(0);
  });

  it('should compare multiple variations', () => {
    const variations = [
      'The quick brown fox jumps',
      'A fast brown fox leaps',
      'Quick brown fox jumping',
    ];
    const references = ['The quick brown fox jumps over the lazy dog'];

    const results = variations.map(v =>
      evaluateAgainstReference('', v, references)
    );

    // First should score highest (most similar)
    expect(results[0].overallScore).toBeGreaterThanOrEqual(results[1].overallScore);
  });
});
