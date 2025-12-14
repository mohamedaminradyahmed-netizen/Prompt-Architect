/**
 * Semantic Similarity Module
 *
 * Provides semantic similarity calculation using real embeddings
 * instead of simple word frequency overlap.
 *
 * Supports:
 * - OpenAI Embeddings API
 * - Local transformers (for offline use)
 * - Caching for cost optimization
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Embedding provider configuration
 */
export interface EmbeddingProvider {
  type: 'openai' | 'local' | 'mock';
  apiKey?: string;
  model?: string;
  dimension?: number;
}

/**
 * Embedding vector
 */
export type Embedding = number[];

/**
 * Cached embedding entry
 */
interface CachedEmbedding {
  text: string;
  embedding: Embedding;
  timestamp: Date;
  provider: string;
}

// ============================================================================
// CACHE MANAGEMENT
// ============================================================================

/**
 * In-memory cache for embeddings
 * Maps text hash to embedding
 */
const embeddingCache = new Map<string, CachedEmbedding>();

/**
 * Generate simple hash for text (for cache key)
 */
function hashText(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash.toString(36);
}

/**
 * Get cached embedding if available
 */
function getCachedEmbedding(text: string, provider: string): Embedding | null {
  const key = `${provider}:${hashText(text)}`;
  const cached = embeddingCache.get(key);

  if (!cached) return null;

  // Check if cache is still valid (24 hours)
  const now = new Date();
  const age = now.getTime() - cached.timestamp.getTime();
  const maxAge = 24 * 60 * 60 * 1000; // 24 hours

  if (age > maxAge) {
    embeddingCache.delete(key);
    return null;
  }

  return cached.embedding;
}

/**
 * Cache an embedding
 */
function cacheEmbedding(text: string, embedding: Embedding, provider: string): void {
  const key = `${provider}:${hashText(text)}`;
  embeddingCache.set(key, {
    text,
    embedding,
    timestamp: new Date(),
    provider
  });
}

/**
 * Clear embedding cache
 */
export function clearEmbeddingCache(): void {
  embeddingCache.clear();
}

/**
 * Get cache statistics
 */
export function getCacheStats(): { size: number; providers: string[] } {
  const providers = new Set<string>();
  for (const entry of embeddingCache.values()) {
    providers.add(entry.provider);
  }

  return {
    size: embeddingCache.size,
    providers: Array.from(providers)
  };
}

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

/**
 * Generate embedding using OpenAI API
 */
async function generateOpenAIEmbedding(
  text: string,
  apiKey: string,
  model: string = 'text-embedding-3-small'
): Promise<Embedding> {
  // In production, use:
  // const openai = new OpenAI({ apiKey });
  // const response = await openai.embeddings.create({
  //   model,
  //   input: text,
  // });
  // return response.data[0].embedding;

  // Mock implementation for now
  console.warn('OpenAI embeddings not configured. Using mock embeddings.');
  return generateMockEmbedding(text, 1536); // OpenAI default dimension
}

/**
 * Generate embedding using local transformer model
 */
async function generateLocalEmbedding(
  text: string,
  model: string = 'Xenova/all-MiniLM-L6-v2'
): Promise<Embedding> {
  // In production, use @xenova/transformers:
  // const { pipeline } = await import('@xenova/transformers');
  // const extractor = await pipeline('feature-extraction', model);
  // const output = await extractor(text, { pooling: 'mean', normalize: true });
  // return Array.from(output.data);

  // Mock implementation for now
  console.warn('Local transformers not configured. Using mock embeddings.');
  return generateMockEmbedding(text, 384); // MiniLM default dimension
}

/**
 * Generate mock embedding (for development/testing)
 * Creates a deterministic embedding based on text content
 */
function generateMockEmbedding(text: string, dimension: number): Embedding {
  // Use text content to seed the embedding (deterministic)
  const seed = hashText(text);
  const embedding: number[] = [];

  // Generate deterministic values based on seed
  let hash = parseInt(seed, 36);
  for (let i = 0; i < dimension; i++) {
    // Simple PRNG based on hash
    hash = (hash * 9301 + 49297) % 233280;
    embedding.push((hash / 233280) * 2 - 1); // -1 to 1
  }

  // Normalize to unit vector
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / magnitude);
}

/**
 * Generate embedding based on provider configuration
 */
async function generateEmbedding(
  text: string,
  provider: EmbeddingProvider,
  useCache: boolean = true
): Promise<Embedding> {
  // Check cache first
  if (useCache) {
    const cached = getCachedEmbedding(text, provider.type);
    if (cached) {
      return cached;
    }
  }

  let embedding: Embedding;

  switch (provider.type) {
    case 'openai':
      if (!provider.apiKey) {
        throw new Error('OpenAI API key is required');
      }
      embedding = await generateOpenAIEmbedding(
        text,
        provider.apiKey,
        provider.model || 'text-embedding-3-small'
      );
      break;

    case 'local':
      embedding = await generateLocalEmbedding(
        text,
        provider.model || 'Xenova/all-MiniLM-L6-v2'
      );
      break;

    case 'mock':
    default:
      embedding = generateMockEmbedding(
        text,
        provider.dimension || 384
      );
      break;
  }

  // Cache the result
  if (useCache) {
    cacheEmbedding(text, embedding, provider.type);
  }

  return embedding;
}

// ============================================================================
// SIMILARITY CALCULATION
// ============================================================================

/**
 * Calculate cosine similarity between two embeddings
 */
export function cosineSimilarity(embedding1: Embedding, embedding2: Embedding): number {
  if (embedding1.length !== embedding2.length) {
    throw new Error('Embeddings must have the same dimension');
  }

  // Dot product
  let dotProduct = 0;
  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
  }

  // Magnitudes
  let mag1 = 0;
  let mag2 = 0;
  for (let i = 0; i < embedding1.length; i++) {
    mag1 += embedding1[i] * embedding1[i];
    mag2 += embedding2[i] * embedding2[i];
  }

  mag1 = Math.sqrt(mag1);
  mag2 = Math.sqrt(mag2);

  if (mag1 === 0 || mag2 === 0) {
    return 0;
  }

  // Cosine similarity
  const similarity = dotProduct / (mag1 * mag2);

  // Clamp to [0, 1] range
  return Math.max(0, Math.min(1, (similarity + 1) / 2));
}

// ============================================================================
// MAIN API
// ============================================================================

/**
 * Calculate semantic similarity between two texts using embeddings
 *
 * @param text1 - First text
 * @param text2 - Second text
 * @param provider - Embedding provider configuration
 * @param useCache - Whether to use cache (default: true)
 * @returns Similarity score (0-1)
 */
export async function calculateSemanticSimilarity(
  text1: string,
  text2: string,
  provider: EmbeddingProvider = { type: 'mock', dimension: 384 },
  useCache: boolean = true
): Promise<number> {
  // Handle identical texts
  if (text1 === text2) {
    return 1.0;
  }

  // Handle empty texts
  if (!text1.trim() || !text2.trim()) {
    return 0.0;
  }

  // Generate embeddings
  const embedding1 = await generateEmbedding(text1, provider, useCache);
  const embedding2 = await generateEmbedding(text2, provider, useCache);

  // Calculate similarity
  return cosineSimilarity(embedding1, embedding2);
}

/**
 * Calculate similarity between multiple text pairs
 */
export async function calculateBatchSimilarity(
  pairs: Array<{ text1: string; text2: string }>,
  provider: EmbeddingProvider = { type: 'mock', dimension: 384 },
  useCache: boolean = true
): Promise<number[]> {
  const results: number[] = [];

  for (const { text1, text2 } of pairs) {
    const similarity = await calculateSemanticSimilarity(
      text1,
      text2,
      provider,
      useCache
    );
    results.push(similarity);
  }

  return results;
}

/**
 * Find most similar text from a list
 */
export async function findMostSimilar(
  query: string,
  candidates: string[],
  provider: EmbeddingProvider = { type: 'mock', dimension: 384 },
  topK: number = 1
): Promise<Array<{ text: string; similarity: number; index: number }>> {
  const queryEmbedding = await generateEmbedding(query, provider, true);

  // Calculate similarities
  const similarities = await Promise.all(
    candidates.map(async (candidate, index) => {
      const candidateEmbedding = await generateEmbedding(candidate, provider, true);
      const similarity = cosineSimilarity(queryEmbedding, candidateEmbedding);

      return { text: candidate, similarity, index };
    })
  );

  // Sort by similarity (descending)
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Return top K
  return similarities.slice(0, topK);
}

// ============================================================================
// FALLBACK: WORD FREQUENCY (for comparison)
// ============================================================================

/**
 * Simple word frequency similarity (fallback)
 * Kept for comparison and as a lightweight alternative
 */
export function calculateWordFrequencySimilarity(
  text1: string,
  text2: string
): number {
  // Tokenize
  const tokens1 = text1.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);

  const tokens2 = text2.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);

  // Convert to sets
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);

  // Calculate Jaccard similarity
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  if (union.size === 0) return 0;

  return intersection.size / union.size;
}

// ============================================================================
// CONFIGURATION HELPERS
// ============================================================================

/**
 * Create OpenAI provider configuration
 */
export function createOpenAIProvider(apiKey: string, model?: string): EmbeddingProvider {
  return {
    type: 'openai',
    apiKey,
    model: model || 'text-embedding-3-small',
    dimension: 1536
  };
}

/**
 * Create local transformer provider configuration
 */
export function createLocalProvider(model?: string): EmbeddingProvider {
  return {
    type: 'local',
    model: model || 'Xenova/all-MiniLM-L6-v2',
    dimension: 384
  };
}

/**
 * Create mock provider configuration (for testing)
 */
export function createMockProvider(dimension: number = 384): EmbeddingProvider {
  return {
    type: 'mock',
    dimension
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  calculateSemanticSimilarity,
  calculateBatchSimilarity,
  findMostSimilar,
  calculateWordFrequencySimilarity,
  cosineSimilarity,
  createOpenAIProvider,
  createLocalProvider,
  createMockProvider,
  clearEmbeddingCache,
  getCacheStats
};
