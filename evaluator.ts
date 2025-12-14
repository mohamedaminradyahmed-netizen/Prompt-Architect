/**
 * Evaluator heuristics for prompt scoring
 * Combines token cost and similarity metrics to score prompt variations
 */

import { PromptVariation } from './mutations';

export interface ScoredSuggestion {
    prompt: string;
    mutation: string;
    score: number;
    tokenCount: number;
    estimatedCost: number;
    similarity: number;
}

/**
 * Calculate token count using simple word-based approximation
 * Approximates ~1.3 tokens per word (common for English text)
 */
export function calculateTokenCount(text: string): number {
    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;
    // Approximate tokens: ~1.3 tokens per word for English
    return Math.ceil(wordCount * 1.3);
}

/**
 * Estimate cost based on token count
 * Using approximate GPT-4 pricing: $0.03 per 1K input tokens
 */
export function estimateCost(tokenCount: number): number {
    const costPer1kTokens = 0.03;
    return (tokenCount / 1000) * costPer1kTokens;
}

/**
 * Calculate semantic similarity using embeddings
 * DIRECTIVE-018: Real embeddings implementation with caching
 */
export async function calculateSemanticSimilarity(
  text1: string,
  text2: string,
  useCache: boolean = true
): Promise<number> {
  // Check cache first
  if (useCache) {
    const cacheKey = `similarity_${hashText(text1)}_${hashText(text2)}`;
    const cached = getFromCache(cacheKey);
    if (cached !== null) {
      return cached;
    }
  }

  try {
    // Get embeddings for both texts
    const [embedding1, embedding2] = await Promise.all([
      getEmbedding(text1, useCache),
      getEmbedding(text2, useCache)
    ]);

    // Calculate cosine similarity
    const similarity = cosineSimilarity(embedding1, embedding2);

    // Cache the result
    if (useCache) {
      const cacheKey = `similarity_${hashText(text1)}_${hashText(text2)}`;
      setCache(cacheKey, similarity);
    }

    return similarity;
  } catch (error) {
    console.warn('Embeddings API failed, falling back to word frequency:', error);
    return calculateWordFrequencySimilarity(text1, text2);
  }
}

/**
 * Get embedding for text with caching
 */
async function getEmbedding(text: string, useCache: boolean = true): Promise<number[]> {
  if (useCache) {
    const cacheKey = `embedding_${hashText(text)}`;
    const cached = getFromCache(cacheKey);
    if (cached !== null) {
      return cached;
    }
  }

  // Mock embedding API call - replace with real implementation
  const embedding = await mockEmbeddingAPI(text);
  
  if (useCache) {
    const cacheKey = `embedding_${hashText(text)}`;
    setCache(cacheKey, embedding);
  }

  return embedding;
}

/**
 * Mock embedding API - replace with OpenAI/HuggingFace/etc.
 */
async function mockEmbeddingAPI(text: string): Promise<number[]> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 100));
  
  // Generate deterministic "embedding" based on text
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Array(384).fill(0); // Common embedding size
  
  words.forEach((word, index) => {
    for (let i = 0; i < word.length && i < embedding.length; i++) {
      embedding[i] += word.charCodeAt(i % word.length) * (index + 1);
    }
  });
  
  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => magnitude > 0 ? val / magnitude : 0);
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    magnitude1 += vec1[i] * vec1[i];
    magnitude2 += vec2[i] * vec2[i];
  }

  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);

  if (magnitude1 === 0 || magnitude2 === 0) {
    return 0;
  }

  return dotProduct / (magnitude1 * magnitude2);
}

/**
 * Fallback word frequency similarity (original implementation)
 */
function calculateWordFrequencySimilarity(text1: string, text2: string): number {
    // Normalize and tokenize
    const normalize = (text: string) =>
        text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);

    const words1 = normalize(text1);
    const words2 = normalize(text2);

    // Build word frequency maps
    const freq1 = new Map<string, number>();
    const freq2 = new Map<string, number>();

    words1.forEach(word => freq1.set(word, (freq1.get(word) || 0) + 1));
    words2.forEach(word => freq2.set(word, (freq2.get(word) || 0) + 1));

    // Get all unique words
    const allWords = new Set<string>();
    freq1.forEach((_, word) => allWords.add(word));
    freq2.forEach((_, word) => allWords.add(word));

    // Calculate dot product and magnitudes
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    allWords.forEach(word => {
        const f1 = freq1.get(word) || 0;
        const f2 = freq2.get(word) || 0;
        dotProduct += f1 * f2;
        magnitude1 += f1 * f1;
        magnitude2 += f2 * f2;
    });

    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    if (magnitude1 === 0 || magnitude2 === 0) {
        return 0;
    }

    return dotProduct / (magnitude1 * magnitude2);
}

/**
 * Legacy function for backward compatibility
 */
export function calculateSimilarity(text1: string, text2: string): number {
    return calculateWordFrequencySimilarity(text1, text2);
}

// Cache utilities
const embeddingCache = new Map<string, any>();

function getFromCache(key: string): any {
  return embeddingCache.get(key) || null;
}

function setCache(key: string, value: any): void {
  // Simple LRU: remove oldest if cache gets too large
  if (embeddingCache.size > 1000) {
    const firstKey = embeddingCache.keys().next().value;
    embeddingCache.delete(firstKey);
  }
  embeddingCache.set(key, value);
}

function hashText(text: string): string {
  // Simple hash function
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash.toString(36);
}

/**
 * Calculate overall score for a prompt variation
 * Higher similarity is better, lower token cost is better
 * Score formula: similarity * 0.7 + (1 - normalized_cost) * 0.3
 */
export function calculateScore(
    similarity: number,
    tokenCount: number,
    maxTokens: number = 500
): number {
    // Normalize token count (lower is better)
    const normalizedCost = Math.min(tokenCount / maxTokens, 1);
    const costScore = 1 - normalizedCost;

    // Weighted combination: similarity matters more
    const score = (similarity * 0.7) + (costScore * 0.3);

    // Return as percentage (0-100)
    return Math.round(score * 100);
}

/**
 * Evaluate a list of prompt variations and return scored suggestions
 */
export function evaluateSuggestions(
    originalPrompt: string,
    variations: PromptVariation[]
): ScoredSuggestion[] {
    const maxTokens = Math.max(...variations.map(v => calculateTokenCount(v.prompt)), 100);

    const scored = variations.map(variation => {
        const tokenCount = calculateTokenCount(variation.prompt);
        const estimatedCost = estimateCost(tokenCount);
        const similarity = calculateSimilarity(originalPrompt, variation.prompt);
        const score = calculateScore(similarity, tokenCount, maxTokens);

        return {
            prompt: variation.prompt,
            mutation: variation.mutation,
            score,
            tokenCount,
            estimatedCost,
            similarity
        };
    });

    // Sort by score descending
    return scored.sort((a, b) => b.score - a.score);
}

/**
 * Get top N suggestions from scored list
 */
export function getTopSuggestions(
    originalPrompt: string,
    variations: PromptVariation[],
    count: number = 3
): ScoredSuggestion[] {
    const scored = evaluateSuggestions(originalPrompt, variations);
    return scored.slice(0, count);
}
