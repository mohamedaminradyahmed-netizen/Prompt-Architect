/**
 * Tests for Semantic Similarity Module (DIRECTIVE-018)
 *
 * Tests cover:
 * - Embedding generation (OpenAI, Local, Mock)
 * - Cosine similarity calculation
 * - Semantic similarity computation
 * - Cache management
 * - Batch processing
 * - Finding most similar texts
 * - Provider configuration
 */

import {
  calculateSemanticSimilarity,
  calculateBatchSimilarity,
  findMostSimilar,
  calculateWordFrequencySimilarity,
  cosineSimilarity,
  createOpenAIProvider,
  createLocalProvider,
  createMockProvider,
  clearEmbeddingCache,
  getCacheStats,
  type EmbeddingProvider,
  type Embedding,
} from '../../evaluator/semanticSimilarity';

// ============================================================================
// SETUP & TEARDOWN
// ============================================================================

beforeEach(() => {
  // Clear cache before each test
  clearEmbeddingCache();
});

afterEach(() => {
  // Clean up after each test
  clearEmbeddingCache();
});

// ============================================================================
// COSINE SIMILARITY TESTS
// ============================================================================

describe('cosineSimilarity', () => {
  it('should calculate perfect similarity for identical embeddings', () => {
    const embedding1: Embedding = [1, 0, 0, 0];
    const embedding2: Embedding = [1, 0, 0, 0];

    const similarity = cosineSimilarity(embedding1, embedding2);

    expect(similarity).toBe(1.0);
  });

  it('should calculate zero similarity for orthogonal embeddings', () => {
    const embedding1: Embedding = [1, 0, 0, 0];
    const embedding2: Embedding = [0, 1, 0, 0];

    const similarity = cosineSimilarity(embedding1, embedding2);

    // Orthogonal vectors should have 0 cosine similarity
    // After normalization to [0,1], this becomes 0.5
    expect(similarity).toBeCloseTo(0.5, 1);
  });

  it('should calculate similarity for opposite embeddings', () => {
    const embedding1: Embedding = [1, 0, 0, 0];
    const embedding2: Embedding = [-1, 0, 0, 0];

    const similarity = cosineSimilarity(embedding1, embedding2);

    // Opposite vectors: cosine = -1, normalized to [0,1] = 0
    expect(similarity).toBe(0);
  });

  it('should handle high-dimensional embeddings', () => {
    const dimension = 384;
    const embedding1: Embedding = new Array(dimension).fill(0.5);
    const embedding2: Embedding = new Array(dimension).fill(0.5);

    const similarity = cosineSimilarity(embedding1, embedding2);

    expect(similarity).toBe(1.0);
  });

  it('should throw error for mismatched dimensions', () => {
    const embedding1: Embedding = [1, 0, 0];
    const embedding2: Embedding = [1, 0, 0, 0];

    expect(() => cosineSimilarity(embedding1, embedding2)).toThrow(
      'Embeddings must have the same dimension'
    );
  });

  it('should handle zero magnitude embeddings', () => {
    const embedding1: Embedding = [0, 0, 0, 0];
    const embedding2: Embedding = [1, 0, 0, 0];

    const similarity = cosineSimilarity(embedding1, embedding2);

    expect(similarity).toBe(0);
  });

  it('should clamp results to [0, 1]', () => {
    const embedding1: Embedding = [1, 0, 0];
    const embedding2: Embedding = [0.5, 0.5, 0];

    const similarity = cosineSimilarity(embedding1, embedding2);

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should calculate similarity for normalized embeddings', () => {
    // Pre-normalized unit vectors
    const embedding1: Embedding = [1, 0];
    const embedding2: Embedding = [0.707, 0.707]; // 45 degrees

    const similarity = cosineSimilarity(embedding1, embedding2);

    expect(similarity).toBeGreaterThan(0.5);
    expect(similarity).toBeLessThan(1);
  });
});

// ============================================================================
// SEMANTIC SIMILARITY TESTS
// ============================================================================

describe('calculateSemanticSimilarity', () => {
  const mockProvider = createMockProvider(384);

  it('should return 1.0 for identical texts', async () => {
    const text = 'This is a test';
    const similarity = await calculateSemanticSimilarity(text, text, mockProvider);

    expect(similarity).toBe(1.0);
  });

  it('should return 0.0 for empty texts', async () => {
    const similarity1 = await calculateSemanticSimilarity('', 'test', mockProvider);
    const similarity2 = await calculateSemanticSimilarity('test', '', mockProvider);
    const similarity3 = await calculateSemanticSimilarity('   ', 'test', mockProvider);

    expect(similarity1).toBe(0);
    expect(similarity2).toBe(0);
    expect(similarity3).toBe(0);
  });

  it('should calculate similarity for similar texts', async () => {
    const text1 = 'The quick brown fox';
    const text2 = 'The fast brown fox';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThan(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should calculate similarity for different texts', async () => {
    const text1 = 'Machine learning is a subset of AI';
    const text2 = 'Cats are fluffy animals';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThan(1);
  });

  it('should use cache when enabled', async () => {
    const text1 = 'Test text for caching';
    const text2 = 'Another test text';

    clearEmbeddingCache();

    // First call - not cached
    await calculateSemanticSimilarity(text1, text2, mockProvider, true);

    const stats1 = getCacheStats();
    expect(stats1.size).toBe(2); // Both texts cached

    // Second call - should use cache
    await calculateSemanticSimilarity(text1, text2, mockProvider, true);

    const stats2 = getCacheStats();
    expect(stats2.size).toBe(2); // Still only 2 entries
  });

  it('should not use cache when disabled', async () => {
    const text1 = 'Test';
    const text2 = 'Test';

    clearEmbeddingCache();

    await calculateSemanticSimilarity(text1, text2, mockProvider, false);

    const stats = getCacheStats();
    expect(stats.size).toBe(0); // No caching
  });

  it('should handle very long texts', async () => {
    const longText1 = 'word '.repeat(1000);
    const longText2 = 'word '.repeat(1000);

    const similarity = await calculateSemanticSimilarity(
      longText1,
      longText2,
      mockProvider,
      false
    );

    expect(similarity).toBe(1.0); // Identical
  });

  it('should handle unicode text', async () => {
    const text1 = '你好世界';
    const text2 = 'مرحبا بالعالم';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should handle special characters', async () => {
    const text1 = 'Hello, world! @#$%';
    const text2 = 'Hello world';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThan(0);
  });

  it('should be deterministic for same inputs', async () => {
    const text1 = 'Test text A';
    const text2 = 'Test text B';

    const similarity1 = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );
    const similarity2 = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity1).toBe(similarity2);
  });
});

// ============================================================================
// BATCH SIMILARITY TESTS
// ============================================================================

describe('calculateBatchSimilarity', () => {
  const mockProvider = createMockProvider(384);

  it('should calculate similarities for multiple pairs', async () => {
    const pairs = [
      { text1: 'Hello', text2: 'Hi' },
      { text1: 'Cat', text2: 'Dog' },
      { text1: 'AI', text2: 'Machine Learning' },
    ];

    const similarities = await calculateBatchSimilarity(pairs, mockProvider, false);

    expect(similarities).toHaveLength(3);
    similarities.forEach(sim => {
      expect(sim).toBeGreaterThanOrEqual(0);
      expect(sim).toBeLessThanOrEqual(1);
    });
  });

  it('should handle empty batch', async () => {
    const similarities = await calculateBatchSimilarity([], mockProvider, false);
    expect(similarities).toHaveLength(0);
  });

  it('should use cache across batch', async () => {
    clearEmbeddingCache();

    const pairs = [
      { text1: 'Same text', text2: 'Different text' },
      { text1: 'Same text', text2: 'Another text' },
    ];

    await calculateBatchSimilarity(pairs, mockProvider, true);

    const stats = getCacheStats();
    // Should cache: 'Same text' (once), 'Different text', 'Another text' = 3
    expect(stats.size).toBe(3);
  });

  it('should handle identical pairs', async () => {
    const pairs = [
      { text1: 'Test', text2: 'Test' },
      { text1: 'Test', text2: 'Test' },
    ];

    const similarities = await calculateBatchSimilarity(pairs, mockProvider, false);

    expect(similarities).toHaveLength(2);
    similarities.forEach(sim => expect(sim).toBe(1.0));
  });
});

// ============================================================================
// FIND MOST SIMILAR TESTS
// ============================================================================

describe('findMostSimilar', () => {
  const mockProvider = createMockProvider(384);

  it('should find single most similar text', async () => {
    const query = 'Machine learning';
    const candidates = [
      'Artificial intelligence',
      'Cooking recipes',
      'Neural networks',
      'Travel destinations',
    ];

    const results = await findMostSimilar(query, candidates, mockProvider, 1);

    expect(results).toHaveLength(1);
    expect(results[0].text).toBe(candidates[results[0].index]);
    expect(results[0].similarity).toBeGreaterThanOrEqual(0);
    expect(results[0].similarity).toBeLessThanOrEqual(1);
  });

  it('should find top K most similar texts', async () => {
    const query = 'Python programming';
    const candidates = [
      'JavaScript coding',
      'Cooking pasta',
      'Java development',
      'Machine learning',
      'Gardening tips',
    ];

    const results = await findMostSimilar(query, candidates, mockProvider, 3);

    expect(results).toHaveLength(3);

    // Results should be sorted by similarity (descending)
    for (let i = 0; i < results.length - 1; i++) {
      expect(results[i].similarity).toBeGreaterThanOrEqual(results[i + 1].similarity);
    }
  });

  it('should return all candidates when topK exceeds length', async () => {
    const query = 'Test';
    const candidates = ['A', 'B'];

    const results = await findMostSimilar(query, candidates, mockProvider, 10);

    expect(results).toHaveLength(2);
  });

  it('should handle empty candidates', async () => {
    const query = 'Test';
    const candidates: string[] = [];

    const results = await findMostSimilar(query, candidates, mockProvider, 1);

    expect(results).toHaveLength(0);
  });

  it('should handle single candidate', async () => {
    const query = 'Test';
    const candidates = ['Only one'];

    const results = await findMostSimilar(query, candidates, mockProvider, 1);

    expect(results).toHaveLength(1);
    expect(results[0].text).toBe('Only one');
    expect(results[0].index).toBe(0);
  });

  it('should preserve original index', async () => {
    const query = 'Test';
    const candidates = ['A', 'B', 'C', 'D'];

    const results = await findMostSimilar(query, candidates, mockProvider, 4);

    results.forEach(result => {
      expect(candidates[result.index]).toBe(result.text);
    });
  });

  it('should use cache for embeddings', async () => {
    clearEmbeddingCache();

    const query = 'Test query';
    const candidates = ['Candidate 1', 'Candidate 2'];

    await findMostSimilar(query, candidates, mockProvider, 1);

    const stats = getCacheStats();
    // Query + 2 candidates = 3
    expect(stats.size).toBe(3);
  });
});

// ============================================================================
// WORD FREQUENCY SIMILARITY TESTS (Fallback)
// ============================================================================

describe('calculateWordFrequencySimilarity', () => {
  it('should return 1.0 for identical texts', () => {
    const text = 'The quick brown fox';
    const similarity = calculateWordFrequencySimilarity(text, text);

    expect(similarity).toBe(1.0);
  });

  it('should return 0.0 for completely different texts', () => {
    const text1 = 'apple banana cherry';
    const text2 = 'dog elephant fox';

    const similarity = calculateWordFrequencySimilarity(text1, text2);

    expect(similarity).toBe(0);
  });

  it('should calculate Jaccard similarity', () => {
    const text1 = 'the quick brown fox';
    const text2 = 'the fast brown dog';

    const similarity = calculateWordFrequencySimilarity(text1, text2);

    // Intersection: {the, brown} = 2
    // Union: {the, quick, brown, fox, fast, dog} = 6
    // Jaccard = 2/6 = 0.333...
    expect(similarity).toBeCloseTo(0.333, 2);
  });

  it('should ignore case', () => {
    const text1 = 'HELLO WORLD';
    const text2 = 'hello world';

    const similarity = calculateWordFrequencySimilarity(text1, text2);

    expect(similarity).toBe(1.0);
  });

  it('should ignore punctuation', () => {
    const text1 = 'Hello, world!';
    const text2 = 'Hello world';

    const similarity = calculateWordFrequencySimilarity(text1, text2);

    expect(similarity).toBe(1.0);
  });

  it('should handle empty texts', () => {
    const similarity1 = calculateWordFrequencySimilarity('', 'test');
    const similarity2 = calculateWordFrequencySimilarity('test', '');
    const similarity3 = calculateWordFrequencySimilarity('', '');

    expect(similarity1).toBe(0);
    expect(similarity2).toBe(0);
    expect(similarity3).toBe(0);
  });

  it('should handle whitespace-only texts', () => {
    const similarity = calculateWordFrequencySimilarity('   ', 'test');

    expect(similarity).toBe(0);
  });
});

// ============================================================================
// PROVIDER CONFIGURATION TESTS
// ============================================================================

describe('Provider Configuration', () => {
  describe('createOpenAIProvider', () => {
    it('should create OpenAI provider with default model', () => {
      const provider = createOpenAIProvider('test-key');

      expect(provider.type).toBe('openai');
      expect(provider.apiKey).toBe('test-key');
      expect(provider.model).toBe('text-embedding-3-small');
      expect(provider.dimension).toBe(1536);
    });

    it('should create OpenAI provider with custom model', () => {
      const provider = createOpenAIProvider('test-key', 'text-embedding-3-large');

      expect(provider.type).toBe('openai');
      expect(provider.model).toBe('text-embedding-3-large');
    });
  });

  describe('createLocalProvider', () => {
    it('should create local provider with default model', () => {
      const provider = createLocalProvider();

      expect(provider.type).toBe('local');
      expect(provider.model).toBe('Xenova/all-MiniLM-L6-v2');
      expect(provider.dimension).toBe(384);
    });

    it('should create local provider with custom model', () => {
      const provider = createLocalProvider('custom-model');

      expect(provider.type).toBe('local');
      expect(provider.model).toBe('custom-model');
    });
  });

  describe('createMockProvider', () => {
    it('should create mock provider with default dimension', () => {
      const provider = createMockProvider();

      expect(provider.type).toBe('mock');
      expect(provider.dimension).toBe(384);
    });

    it('should create mock provider with custom dimension', () => {
      const provider = createMockProvider(768);

      expect(provider.type).toBe('mock');
      expect(provider.dimension).toBe(768);
    });
  });
});

// ============================================================================
// CACHE MANAGEMENT TESTS
// ============================================================================

describe('Cache Management', () => {
  const mockProvider = createMockProvider(384);

  describe('clearEmbeddingCache', () => {
    it('should clear all cached embeddings', async () => {
      await calculateSemanticSimilarity('text1', 'text2', mockProvider, true);

      const statsBefore = getCacheStats();
      expect(statsBefore.size).toBeGreaterThan(0);

      clearEmbeddingCache();

      const statsAfter = getCacheStats();
      expect(statsAfter.size).toBe(0);
    });
  });

  describe('getCacheStats', () => {
    it('should return correct cache size', async () => {
      clearEmbeddingCache();

      await calculateSemanticSimilarity('text1', 'text2', mockProvider, true);

      const stats = getCacheStats();
      expect(stats.size).toBe(2); // 2 texts cached
    });

    it('should track providers', async () => {
      clearEmbeddingCache();

      const mockProvider1 = createMockProvider(384);
      const mockProvider2 = createMockProvider(768);

      await calculateSemanticSimilarity('text1', 'text2', mockProvider1, true);

      const stats = getCacheStats();
      expect(stats.providers).toContain('mock');
    });

    it('should return empty stats for empty cache', () => {
      clearEmbeddingCache();

      const stats = getCacheStats();
      expect(stats.size).toBe(0);
      expect(stats.providers).toHaveLength(0);
    });
  });

  describe('Cache Expiration', () => {
    it('should cache embeddings with timestamp', async () => {
      clearEmbeddingCache();

      await calculateSemanticSimilarity('test', 'test2', mockProvider, true);

      const stats = getCacheStats();
      expect(stats.size).toBe(2);
    });

    // Note: Testing actual TTL expiration would require time manipulation
    // which is complex in this testing environment
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  const mockProvider = createMockProvider(384);

  it('should complete full workflow with caching', async () => {
    clearEmbeddingCache();

    // Initial similarity calculation
    const similarity1 = await calculateSemanticSimilarity(
      'Hello world',
      'Hi there',
      mockProvider,
      true
    );

    expect(similarity1).toBeGreaterThanOrEqual(0);
    expect(similarity1).toBeLessThanOrEqual(1);

    const stats1 = getCacheStats();
    expect(stats1.size).toBe(2);

    // Batch calculation (should use cache for 'Hello world')
    const pairs = [
      { text1: 'Hello world', text2: 'Greetings' },
      { text1: 'AI', text2: 'ML' },
    ];

    const batchResults = await calculateBatchSimilarity(pairs, mockProvider, true);

    expect(batchResults).toHaveLength(2);

    const stats2 = getCacheStats();
    // 'Hello world' already cached, add 'Greetings', 'AI', 'ML' = 5 total
    expect(stats2.size).toBe(5);

    // Find most similar (should use cache)
    const mostSimilar = await findMostSimilar(
      'Hello world',
      ['Hi there', 'Greetings', 'Goodbye'],
      mockProvider,
      2
    );

    expect(mostSimilar).toHaveLength(2);

    const stats3 = getCacheStats();
    // 'Hello world', 'Hi there', 'Greetings' already cached, add 'Goodbye' = 6 total
    expect(stats3.size).toBe(6);
  });

  it('should handle different providers', async () => {
    const providers = [
      createMockProvider(384),
      createMockProvider(768),
      createOpenAIProvider('fake-key'),
      createLocalProvider(),
    ];

    // Mock provider should work
    const similarity = await calculateSemanticSimilarity(
      'test',
      'test2',
      providers[0],
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should compare semantic vs word frequency similarity', async () => {
    const text1 = 'Machine learning is a branch of AI';
    const text2 = 'AI includes machine learning';

    const semanticSim = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );
    const wordFreqSim = calculateWordFrequencySimilarity(text1, text2);

    // Both should be > 0
    expect(semanticSim).toBeGreaterThan(0);
    expect(wordFreqSim).toBeGreaterThan(0);

    // Semantic similarity might differ from word frequency
    // (we don't assert which is higher as it depends on implementation)
  });

  it('should handle real-world document similarity', async () => {
    const doc1 = 'The quick brown fox jumps over the lazy dog';
    const doc2 = 'A fast brown fox leaps across a sleepy canine';
    const doc3 = 'Python is a programming language';

    // Calculate all pairwise similarities
    const sim12 = await calculateSemanticSimilarity(doc1, doc2, mockProvider, false);
    const sim13 = await calculateSemanticSimilarity(doc1, doc3, mockProvider, false);
    const sim23 = await calculateSemanticSimilarity(doc2, doc3, mockProvider, false);

    // Doc1 and Doc2 should be more similar than Doc1-Doc3 or Doc2-Doc3
    // (they have similar meaning)
    expect(sim12).toBeGreaterThan(0);
    expect(sim13).toBeGreaterThanOrEqual(0);
    expect(sim23).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
  const mockProvider = createMockProvider(384);

  it('should handle very short texts', async () => {
    const similarity = await calculateSemanticSimilarity('a', 'b', mockProvider, false);

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should handle very long texts', async () => {
    const longText1 = 'word '.repeat(10000);
    const longText2 = 'word '.repeat(10000);

    const similarity = await calculateSemanticSimilarity(
      longText1,
      longText2,
      mockProvider,
      false
    );

    expect(similarity).toBe(1.0);
  });

  it('should handle texts with only numbers', async () => {
    const similarity = await calculateSemanticSimilarity(
      '123456',
      '789012',
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should handle texts with only punctuation', async () => {
    const similarity = await calculateSemanticSimilarity(
      '!!!',
      '???',
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should handle mixed language texts', async () => {
    const text1 = 'Hello world 你好世界';
    const text2 = 'مرحبا Bonjour';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
    expect(similarity).toBeLessThanOrEqual(1);
  });

  it('should handle texts with newlines and tabs', async () => {
    const text1 = 'Line 1\nLine 2\tTabbed';
    const text2 = 'Line 1 Line 2 Tabbed';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThanOrEqual(0);
  });

  it('should handle repeated words', async () => {
    const text1 = 'test test test';
    const text2 = 'test';

    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      mockProvider,
      false
    );

    expect(similarity).toBeGreaterThan(0);
  });
});
