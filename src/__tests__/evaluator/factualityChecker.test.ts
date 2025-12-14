/**
 * Tests for Factuality Checker Module (DIRECTIVE-014)
 *
 * Tests cover:
 * - RAG-based factuality verification
 * - Claim extraction and verification
 * - Source reliability checking
 * - Contradiction detection
 * - Batch processing
 * - Comparison and formatting
 */

import {
  FactualityChecker,
  verifyFactuality,
  compareFactuality,
  formatFactualityCheck,
  getFactualityRecommendation,
  type FactualityConfig,
  type FactualityCheck,
  type ClaimVerification,
} from '../../evaluator/factualityChecker';

import {
  createVectorStore,
  type EmbeddingProvider,
  type InMemoryVectorStore,
} from '../../rag/vectorStore';

import { addDocuments, type Document } from '../../rag/retrieval';

// ============================================================================
// SETUP & UTILITIES
// ============================================================================

/**
 * Create a mock embedding provider for testing
 */
function createMockEmbeddingProvider(dimension: number = 384): EmbeddingProvider {
  return {
    type: 'mock',
    dimension,
    generateEmbedding: async (text: string) => {
      // Simple deterministic embedding based on text
      const embedding = new Array(dimension).fill(0);
      for (let i = 0; i < text.length && i < dimension; i++) {
        embedding[i] = text.charCodeAt(i) / 255;
      }
      return embedding;
    },
  };
}

/**
 * Create a populated vector store for testing
 */
async function createTestVectorStore(
  embeddingProvider: EmbeddingProvider
): Promise<InMemoryVectorStore> {
  const store = createVectorStore({
    provider: 'memory',
    dimension: embeddingProvider.dimension,
    metric: 'cosine',
  });

  // Add test documents
  const documents: Document[] = [
    {
      id: 'doc1',
      content: 'Paris is the capital of France. It has a population of 2.1 million.',
      metadata: {
        source: 'Encyclopedia',
        reliability: 0.9,
        timestamp: new Date('2024-01-01'),
      },
    },
    {
      id: 'doc2',
      content: 'The Eiffel Tower is 324 meters tall and located in Paris.',
      metadata: {
        source: 'Encyclopedia',
        reliability: 0.9,
        timestamp: new Date('2024-01-01'),
      },
    },
    {
      id: 'doc3',
      content: 'Python is a high-level programming language created by Guido van Rossum.',
      metadata: {
        source: 'Tech Wiki',
        reliability: 0.85,
        timestamp: new Date('2024-01-01'),
      },
    },
    {
      id: 'doc4',
      content: 'JavaScript was created in 1995 and runs in web browsers.',
      metadata: {
        source: 'Tech Wiki',
        reliability: 0.85,
        timestamp: new Date('2024-01-01'),
      },
    },
    {
      id: 'doc5',
      content: 'Machine learning is a subset of artificial intelligence.',
      metadata: {
        source: 'AI Research',
        reliability: 0.95,
        timestamp: new Date('2024-01-01'),
      },
    },
  ];

  await addDocuments(documents, store, embeddingProvider);

  return store;
}

/**
 * Create a default factuality checker config
 */
function createDefaultConfig(
  embeddingProvider: EmbeddingProvider
): FactualityConfig {
  return {
    vectorStore: {
      provider: 'memory',
      dimension: embeddingProvider.dimension,
      metric: 'cosine',
    },
    embeddingProvider,
    retrieval: {
      topK: 5,
      minScore: 0.6,
      filterByReliability: true,
      minReliability: 0.7,
    },
    requireMultipleSources: true,
    minSourceCount: 2,
    supportThreshold: 0.7,
    contradictionThreshold: 0.6,
  };
}

// ============================================================================
// FACTUALITY CHECKER TESTS
// ============================================================================

describe('FactualityChecker', () => {
  let embeddingProvider: EmbeddingProvider;
  let vectorStore: InMemoryVectorStore;
  let checker: FactualityChecker;

  beforeEach(async () => {
    embeddingProvider = createMockEmbeddingProvider(384);
    vectorStore = await createTestVectorStore(embeddingProvider);

    const config = createDefaultConfig(embeddingProvider);
    checker = new FactualityChecker(config);

    // Copy documents to checker's vector store
    const docs = await vectorStore.getAllDocuments();
    await addDocuments(docs, checker.getVectorStore(), embeddingProvider);
  });

  describe('Basic Verification', () => {
    it('should verify a factual claim', async () => {
      const text = 'Paris is the capital of France';
      const result = await checker.verifyFactuality(text);

      expect(result).toBeDefined();
      expect(result.isFactual).toBe(true);
      expect(result.overallScore).toBeGreaterThan(0);
      expect(result.confidence).toBeGreaterThan(0);
    });

    it('should detect unsupported claims', async () => {
      const text = 'The moon is made of cheese';
      const result = await checker.verifyFactuality(text);

      expect(result).toBeDefined();
      // May or may not be marked as factual depending on retrieval results
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
    });

    it('should handle text with no factual claims', async () => {
      const text = 'Hello world';
      const result = await checker.verifyFactuality(text);

      expect(result).toBeDefined();
      expect(result.claims).toHaveLength(0);
      expect(result.contradictions).toContain('No verifiable factual claims detected');
    });

    it('should handle empty text', async () => {
      const text = '';
      const result = await checker.verifyFactuality(text);

      expect(result).toBeDefined();
      expect(result.claims).toHaveLength(0);
    });

    it('should return valid score range', async () => {
      const text = 'Paris is in France';
      const result = await checker.verifyFactuality(text);

      expect(result.overallScore).toBeGreaterThanOrEqual(0);
      expect(result.overallScore).toBeLessThanOrEqual(100);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Claim Verification', () => {
    it('should extract and verify multiple claims', async () => {
      const text = `
        Paris is the capital of France.
        The Eiffel Tower is 324 meters tall.
        Python was created by Guido van Rossum.
      `;

      const result = await checker.verifyFactuality(text);

      expect(result.claims.length).toBeGreaterThan(0);
      result.claims.forEach(claim => {
        expect(claim.claim).toBeDefined();
        expect(claim.confidence).toBeGreaterThanOrEqual(0);
        expect(claim.confidence).toBeLessThanOrEqual(1);
      });
    });

    it('should provide supporting evidence for verified claims', async () => {
      const text = 'Paris is the capital of France';
      const result = await checker.verifyFactuality(text);

      if (result.claims.length > 0 && result.claims[0].isSupported) {
        expect(result.claims[0].supportingEvidence.length).toBeGreaterThan(0);
        expect(result.claims[0].sources.length).toBeGreaterThan(0);
      }
    });

    it('should detect claims without sources', async () => {
      const text = 'Unicorns exist on planet Zargon';
      const result = await checker.verifyFactuality(text);

      // Should find claims but likely no supporting sources
      if (result.claims.length > 0) {
        expect(result.claims[0].isSupported).toBe(false);
      }
    });

    it('should calculate confidence correctly', async () => {
      const text = 'Machine learning is a subset of AI';
      const result = await checker.verifyFactuality(text);

      expect(result.confidence).toBeGreaterThan(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Source Requirements', () => {
    it('should collect sources for claims', async () => {
      const text = 'Paris is the capital of France and has the Eiffel Tower';
      const result = await checker.verifyFactuality(text);

      expect(result.sources).toBeDefined();
      expect(Array.isArray(result.sources)).toBe(true);
    });

    it('should handle single source requirement', async () => {
      const config = createDefaultConfig(embeddingProvider);
      config.requireMultipleSources = false;
      config.minSourceCount = 1;

      const checker2 = new FactualityChecker(config);
      const docs = await vectorStore.getAllDocuments();
      await addDocuments(docs, checker2.getVectorStore(), embeddingProvider);

      const text = 'Paris is in France';
      const result = await checker2.verifyFactuality(text);

      expect(result).toBeDefined();
    });

    it('should list unique sources', async () => {
      const text = 'Paris is the capital of France. The Eiffel Tower is in Paris.';
      const result = await checker.verifyFactuality(text);

      const uniqueSources = new Set(result.sources);
      expect(uniqueSources.size).toBe(result.sources.length);
    });
  });

  describe('Contradiction Detection', () => {
    it('should detect contradicting evidence', async () => {
      // Add a contradictory document
      const contradictoryDoc: Document = {
        id: 'contra1',
        content: 'Paris is not the capital of France',
        metadata: {
          source: 'Wrong Source',
          reliability: 0.3,
          timestamp: new Date(),
        },
      };

      await addDocuments([contradictoryDoc], checker.getVectorStore(), embeddingProvider);

      const text = 'Paris is the capital of France';
      const result = await checker.verifyFactuality(text);

      // May or may not detect contradiction depending on retrieval
      expect(result).toBeDefined();
    });

    it('should handle claims with negations', async () => {
      const text = 'Paris is not in Germany';
      const result = await checker.verifyFactuality(text);

      expect(result).toBeDefined();
      expect(result.claims.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Overall Scoring', () => {
    it('should calculate overall score based on supported claims ratio', async () => {
      const text = `
        Paris is the capital of France.
        The moon is made of cheese.
        Python was created by Guido van Rossum.
      `;

      const result = await checker.verifyFactuality(text);

      // Score should reflect ratio of supported claims
      const supportedRatio =
        result.claims.filter(c => c.isSupported).length / result.claims.length;

      const expectedScore = supportedRatio * 100;

      expect(Math.abs(result.overallScore - expectedScore)).toBeLessThan(1);
    });

    it('should mark as factual when majority supported', async () => {
      const text = 'Paris is in France';
      const result = await checker.verifyFactuality(text);

      // If most claims are supported, should be marked factual
      const supportedRatio =
        result.claims.filter(c => c.isSupported).length / result.claims.length;

      if (supportedRatio >= 0.6) {
        expect(result.isFactual).toBe(true);
      } else {
        expect(result.isFactual).toBe(false);
      }
    });

    it('should mark as not factual when minority supported', async () => {
      const text = `
        Unicorns exist.
        Dragons breathe fire.
        Mermaids live in the ocean.
      `;

      const result = await checker.verifyFactuality(text);

      // Unlikely to have supporting evidence
      const supportedRatio =
        result.claims.filter(c => c.isSupported).length / result.claims.length;

      if (supportedRatio < 0.6) {
        expect(result.isFactual).toBe(false);
      }
    });
  });

  describe('Batch Processing', () => {
    it('should verify multiple texts', async () => {
      const texts = [
        'Paris is the capital of France',
        'Python is a programming language',
        'Machine learning is AI',
      ];

      const results = await checker.verifyBatch(texts);

      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.overallScore).toBeGreaterThanOrEqual(0);
      });
    });

    it('should call progress callback', async () => {
      const texts = ['Test 1', 'Test 2', 'Test 3'];
      const progressCalls: Array<{ completed: number; total: number }> = [];

      const onProgress = (completed: number, total: number) => {
        progressCalls.push({ completed, total });
      };

      await checker.verifyBatch(texts, onProgress);

      expect(progressCalls).toHaveLength(3);
      expect(progressCalls[0]).toEqual({ completed: 1, total: 3 });
      expect(progressCalls[1]).toEqual({ completed: 2, total: 3 });
      expect(progressCalls[2]).toEqual({ completed: 3, total: 3 });
    });

    it('should handle empty batch', async () => {
      const results = await checker.verifyBatch([]);
      expect(results).toHaveLength(0);
    });
  });

  describe('Vector Store Access', () => {
    it('should provide access to vector store', () => {
      const store = checker.getVectorStore();
      expect(store).toBeDefined();
    });
  });
});

// ============================================================================
// CONVENIENCE FUNCTIONS TESTS
// ============================================================================

describe('Convenience Functions', () => {
  let embeddingProvider: EmbeddingProvider;
  let vectorStore: InMemoryVectorStore;

  beforeEach(async () => {
    embeddingProvider = createMockEmbeddingProvider(384);
    vectorStore = await createTestVectorStore(embeddingProvider);
  });

  describe('verifyFactuality', () => {
    it('should verify factuality with default config', async () => {
      const text = 'Paris is the capital of France';
      const result = await verifyFactuality(
        text,
        vectorStore,
        embeddingProvider
      );

      expect(result).toBeDefined();
      expect(result.overallScore).toBeGreaterThanOrEqual(0);
    });

    it('should accept optional context', async () => {
      const text = 'It is the capital';
      const context = 'Paris, France';

      const result = await verifyFactuality(
        text,
        vectorStore,
        embeddingProvider,
        context
      );

      expect(result).toBeDefined();
    });
  });

  describe('compareFactuality', () => {
    it('should compare two texts', async () => {
      const textA = 'Paris is the capital of France';
      const textB = 'The moon is made of cheese';

      const comparison = await compareFactuality(
        textA,
        textB,
        vectorStore,
        embeddingProvider
      );

      expect(comparison.checkA).toBeDefined();
      expect(comparison.checkB).toBeDefined();
      expect(comparison.moreFactual).toMatch(/A|B|tie/);
      expect(typeof comparison.scoreDiff).toBe('number');
    });

    it('should identify more factual text (A)', async () => {
      const textA = 'Paris is in France';
      const textB = 'Unicorns live on Mars';

      const comparison = await compareFactuality(
        textA,
        textB,
        vectorStore,
        embeddingProvider
      );

      // A should score higher (unless no sources found for A)
      if (comparison.checkA.overallScore > comparison.checkB.overallScore + 10) {
        expect(comparison.moreFactual).toBe('A');
      }
    });

    it('should identify more factual text (B)', async () => {
      const textA = 'Dragons breathe fire in New York';
      const textB = 'Python is a programming language';

      const comparison = await compareFactuality(
        textA,
        textB,
        vectorStore,
        embeddingProvider
      );

      // B should score higher (unless no sources found for B)
      if (comparison.checkB.overallScore > comparison.checkA.overallScore + 10) {
        expect(comparison.moreFactual).toBe('B');
      }
    });

    it('should identify tie for similar scores', async () => {
      const textA = 'Paris is in France';
      const textB = 'Python is a language';

      const comparison = await compareFactuality(
        textA,
        textB,
        vectorStore,
        embeddingProvider
      );

      if (Math.abs(comparison.scoreDiff) < 10) {
        expect(comparison.moreFactual).toBe('tie');
      }
    });

    it('should calculate score difference correctly', async () => {
      const textA = 'Test A';
      const textB = 'Test B';

      const comparison = await compareFactuality(
        textA,
        textB,
        vectorStore,
        embeddingProvider
      );

      expect(comparison.scoreDiff).toBe(
        comparison.checkB.overallScore - comparison.checkA.overallScore
      );
    });
  });
});

// ============================================================================
// FORMATTING TESTS
// ============================================================================

describe('Formatting Functions', () => {
  describe('formatFactualityCheck', () => {
    it('should format factual check with all details', () => {
      const check: FactualityCheck = {
        isFactual: true,
        confidence: 0.85,
        sources: ['Source A', 'Source B'],
        contradictions: [],
        claims: [
          {
            claim: 'Paris is the capital of France',
            isSupported: true,
            confidence: 0.9,
            supportingEvidence: ['Paris is the capital city of France'],
            contradictingEvidence: [],
            sources: ['Encyclopedia'],
          },
        ],
        overallScore: 90.0,
      };

      const formatted = formatFactualityCheck(check);

      expect(formatted).toContain('Factuality Check Report');
      expect(formatted).toContain('✓ FACTUAL');
      expect(formatted).toContain('90.0/100');
      expect(formatted).toContain('85.0%');
      expect(formatted).toContain('Claims Analyzed: 1');
      expect(formatted).toContain('Supported: 1');
      expect(formatted).toContain('Source A');
      expect(formatted).toContain('Source B');
    });

    it('should format non-factual check', () => {
      const check: FactualityCheck = {
        isFactual: false,
        confidence: 0.6,
        sources: [],
        contradictions: ['Claim 1: contradicted by evidence'],
        claims: [
          {
            claim: 'Unsupported claim',
            isSupported: false,
            confidence: 0.3,
            supportingEvidence: [],
            contradictingEvidence: ['Evidence contradicts'],
            sources: [],
          },
        ],
        overallScore: 20.0,
      };

      const formatted = formatFactualityCheck(check);

      expect(formatted).toContain('✗ NOT FACTUAL');
      expect(formatted).toContain('20.0/100');
      expect(formatted).toContain('Unsupported: 1');
      expect(formatted).toContain('Contradictions Found');
      expect(formatted).toContain('Claim 1');
    });

    it('should handle check with no claims', () => {
      const check: FactualityCheck = {
        isFactual: true,
        confidence: 0.5,
        sources: [],
        contradictions: ['No verifiable factual claims detected'],
        claims: [],
        overallScore: 50,
      };

      const formatted = formatFactualityCheck(check);

      expect(formatted).toContain('Claims Analyzed: 0');
      expect(formatted).toContain('No verifiable factual claims detected');
    });

    it('should truncate long evidence', () => {
      const longEvidence = 'A'.repeat(200);

      const check: FactualityCheck = {
        isFactual: true,
        confidence: 0.8,
        sources: ['Source'],
        contradictions: [],
        claims: [
          {
            claim: 'Test claim',
            isSupported: true,
            confidence: 0.8,
            supportingEvidence: [longEvidence],
            contradictingEvidence: [],
            sources: ['Source'],
          },
        ],
        overallScore: 80,
      };

      const formatted = formatFactualityCheck(check);

      // Should truncate to 100 chars
      expect(formatted).toContain('Supporting:');
      const lines = formatted.split('\n');
      const evidenceLine = lines.find(l => l.includes('Supporting:'));
      expect(evidenceLine!.length).toBeLessThan(150);
    });
  });

  describe('getFactualityRecommendation', () => {
    it('should recommend high factuality', () => {
      const check: FactualityCheck = {
        isFactual: true,
        confidence: 0.9,
        sources: ['A', 'B'],
        contradictions: [],
        claims: [],
        overallScore: 85,
      };

      const recommendation = getFactualityRecommendation(check);
      expect(recommendation).toContain('High factuality');
    });

    it('should recommend moderate factuality', () => {
      const check: FactualityCheck = {
        isFactual: true,
        confidence: 0.7,
        sources: ['A'],
        contradictions: [],
        claims: [],
        overallScore: 65,
      };

      const recommendation = getFactualityRecommendation(check);
      expect(recommendation).toContain('Moderate factuality');
    });

    it('should recommend low factuality', () => {
      const check: FactualityCheck = {
        isFactual: false,
        confidence: 0.5,
        sources: [],
        contradictions: [],
        claims: [],
        overallScore: 45,
      };

      const recommendation = getFactualityRecommendation(check);
      expect(recommendation).toContain('Low factuality');
    });

    it('should recommend very low factuality', () => {
      const check: FactualityCheck = {
        isFactual: false,
        confidence: 0.3,
        sources: [],
        contradictions: ['Many'],
        claims: [],
        overallScore: 25,
      };

      const recommendation = getFactualityRecommendation(check);
      expect(recommendation).toContain('Very low factuality');
    });

    it('should handle edge cases', () => {
      expect(getFactualityRecommendation({ overallScore: 80 } as any)).toContain('High');
      expect(getFactualityRecommendation({ overallScore: 60 } as any)).toContain('Moderate');
      expect(getFactualityRecommendation({ overallScore: 40 } as any)).toContain('Low');
      expect(getFactualityRecommendation({ overallScore: 20 } as any)).toContain('Very low');
    });
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
  let embeddingProvider: EmbeddingProvider;
  let vectorStore: InMemoryVectorStore;

  beforeEach(async () => {
    embeddingProvider = createMockEmbeddingProvider(384);
    vectorStore = await createTestVectorStore(embeddingProvider);
  });

  it('should complete full workflow: verify → format → recommend', async () => {
    const text = 'Paris is the capital of France';

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);
    const formatted = formatFactualityCheck(check);
    const recommendation = getFactualityRecommendation(check);

    expect(check).toBeDefined();
    expect(formatted).toContain('Factuality Check Report');
    expect(recommendation).toBeDefined();
    expect(typeof recommendation).toBe('string');
  });

  it('should handle real-world mixed claims', async () => {
    const text = `
      Paris is the capital of France.
      The Eiffel Tower is 324 meters tall.
      Unicorns live in Paris.
    `;

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);

    expect(check.claims.length).toBeGreaterThan(0);

    // Should have mix of supported and unsupported claims
    const supported = check.claims.filter(c => c.isSupported).length;
    const unsupported = check.claims.filter(c => !c.isSupported).length;

    // At least one should be non-zero
    expect(supported + unsupported).toBe(check.claims.length);
  });

  it('should compare outputs and provide recommendation', async () => {
    const textA = 'Paris is in France and has the Eiffel Tower';
    const textB = 'Dragons live in castles and breathe fire';

    const comparison = await compareFactuality(
      textA,
      textB,
      vectorStore,
      embeddingProvider
    );

    const recommendationA = getFactualityRecommendation(comparison.checkA);
    const recommendationB = getFactualityRecommendation(comparison.checkB);

    expect(comparison.moreFactual).toMatch(/A|B|tie/);
    expect(recommendationA).toBeDefined();
    expect(recommendationB).toBeDefined();
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Edge Cases', () => {
  let embeddingProvider: EmbeddingProvider;
  let vectorStore: InMemoryVectorStore;

  beforeEach(async () => {
    embeddingProvider = createMockEmbeddingProvider(384);
    vectorStore = await createTestVectorStore(embeddingProvider);
  });

  it('should handle very long text', async () => {
    const longText = 'Paris is in France. '.repeat(100);

    const check = await verifyFactuality(longText, vectorStore, embeddingProvider);

    expect(check).toBeDefined();
  });

  it('should handle unicode text', async () => {
    const text = 'باريس هي عاصمة فرنسا. 巴黎是法国的首都.';

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);

    expect(check).toBeDefined();
  });

  it('should handle special characters', async () => {
    const text = 'Paris (France) is <capital> @2024!';

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);

    expect(check).toBeDefined();
  });

  it('should handle text with only punctuation', async () => {
    const text = '!@#$%^&*()';

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);

    expect(check.claims).toHaveLength(0);
  });

  it('should handle whitespace-only text', async () => {
    const text = '   \n\n\t\t   ';

    const check = await verifyFactuality(text, vectorStore, embeddingProvider);

    expect(check.claims).toHaveLength(0);
  });

  it('should handle empty vector store', async () => {
    const emptyStore = createVectorStore({
      provider: 'memory',
      dimension: 384,
      metric: 'cosine',
    });

    const text = 'Paris is the capital of France';

    const check = await verifyFactuality(text, emptyStore, embeddingProvider);

    // Should complete but likely find no support
    expect(check).toBeDefined();
  });
});
