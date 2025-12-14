/**
 * Output Metrics Module
 *
 * Measures actual output characteristics from LLM responses
 * including length, token count, variance, and quality estimates
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Configuration for LLM providers
 */
export interface LLMProvider {
  name: 'openai' | 'anthropic' | 'groq' | 'custom';
  apiKey?: string;
  model?: string;
  baseURL?: string;
}

/**
 * Aggregated metrics from multiple output samples
 */
export interface OutputMetrics {
  avgLength: number;        // Average character count
  avgTokens: number;        // Average token count
  variance: number;         // Statistical variance in length
  stdDeviation: number;     // Standard deviation
  quality: number;          // Estimated quality score (0-1)
  samples: OutputSample[];  // Individual sample results
  timestamp: Date;          // When measurements were taken
}

/**
 * Individual output sample
 */
export interface OutputSample {
  output: string;
  length: number;
  tokenCount: number;
  latency: number;  // milliseconds
  error?: string;
}

/**
 * Cache entry for output metrics
 */
interface CachedMetrics {
  metrics: OutputMetrics;
  expiresAt: Date;
}

// ============================================================================
// CACHE
// ============================================================================

/**
 * Simple in-memory cache for output metrics
 * Maps prompt hash to cached metrics
 */
const metricsCache = new Map<string, CachedMetrics>();

/**
 * Generate a simple hash for a prompt
 * Used for cache key generation
 */
function hashPrompt(prompt: string, provider: string): string {
  // Simple hash function (in production, use a proper hash like SHA-256)
  let hash = 0;
  const str = `${provider}:${prompt}`;

  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  return hash.toString(36);
}

/**
 * Get cached metrics if available and not expired
 */
function getCachedMetrics(prompt: string, provider: string): OutputMetrics | null {
  const key = hashPrompt(prompt, provider);
  const cached = metricsCache.get(key);

  if (!cached) return null;

  // Check if expired
  if (new Date() > cached.expiresAt) {
    metricsCache.delete(key);
    return null;
  }

  return cached.metrics;
}

/**
 * Cache metrics with TTL (default 24 hours)
 */
function cacheMetrics(
  prompt: string,
  provider: string,
  metrics: OutputMetrics,
  ttlHours: number = 24
): void {
  const key = hashPrompt(prompt, provider);
  const expiresAt = new Date();
  expiresAt.setHours(expiresAt.getHours() + ttlHours);

  metricsCache.set(key, { metrics, expiresAt });
}

/**
 * Clear expired cache entries
 */
export function cleanCache(): number {
  const now = new Date();
  let cleaned = 0;

  for (const [key, cached] of metricsCache.entries()) {
    if (now > cached.expiresAt) {
      metricsCache.delete(key);
      cleaned++;
    }
  }

  return cleaned;
}

/**
 * Clear all cache entries
 */
export function clearCache(): void {
  metricsCache.clear();
}

// ============================================================================
// TOKEN COUNTING
// ============================================================================

/**
 * Estimate token count from text
 * Simple approximation: ~1.3 tokens per word in English
 * For production, use tiktoken or the provider's tokenizer
 */
export function estimateTokenCount(text: string): number {
  // Remove extra whitespace
  const normalized = text.trim().replace(/\s+/g, ' ');

  // Count words
  const words = normalized.split(' ').filter(w => w.length > 0);

  // Estimate tokens (1.3x words is common for English)
  return Math.ceil(words.length * 1.3);
}

/**
 * Get actual token count from provider
 * This is a placeholder - in production, use the provider's tokenizer
 */
async function getActualTokenCount(
  text: string,
  provider: LLMProvider
): Promise<number> {
  // Placeholder: In production, use:
  // - tiktoken for OpenAI
  // - Anthropic's tokenizer for Claude
  // - Provider-specific APIs

  return estimateTokenCount(text);
}

// ============================================================================
// LLM EXECUTION (MOCK)
// ============================================================================

/**
 * Mock LLM execution for demonstration
 * In production, this would call actual LLM APIs
 */
async function executeLLM(
  prompt: string,
  provider: LLMProvider
): Promise<{ output: string; latency: number }> {
  const startTime = Date.now();

  // MOCK: Simulate API call
  // In production, replace with actual API calls:
  // - OpenAI: const response = await openai.chat.completions.create(...)
  // - Anthropic: const response = await anthropic.messages.create(...)
  // - Groq: const response = await groq.chat.completions.create(...)

  await new Promise(resolve => setTimeout(resolve, 100)); // Simulate network delay

  // Mock output generation
  const mockOutput = `This is a mock response to the prompt: "${prompt.substring(0, 50)}..."\n\nThe response would vary in length and content based on the actual LLM output.`;

  const latency = Date.now() - startTime;

  return { output: mockOutput, latency };
}

// ============================================================================
// MAIN FUNCTIONS
// ============================================================================

/**
 * Measure actual output metrics by running the prompt multiple times
 *
 * @param prompt - The prompt to measure
 * @param provider - LLM provider configuration
 * @param samples - Number of times to run the prompt (default: 3)
 * @param useCache - Whether to use cached results (default: true)
 * @returns OutputMetrics with aggregated statistics
 */
export async function measureActualOutput(
  prompt: string,
  provider: LLMProvider,
  samples: number = 3,
  useCache: boolean = true
): Promise<OutputMetrics> {
  // Check cache first
  if (useCache) {
    const cached = getCachedMetrics(prompt, provider.name);
    if (cached) {
      return cached;
    }
  }

  // Validate inputs
  if (samples < 1) {
    throw new Error('samples must be at least 1');
  }

  if (samples > 10) {
    console.warn('Running more than 10 samples may be expensive and slow');
  }

  // Run prompt multiple times
  const results: OutputSample[] = [];
  const errors: string[] = [];

  for (let i = 0; i < samples; i++) {
    try {
      const { output, latency } = await executeLLM(prompt, provider);
      const tokenCount = await getActualTokenCount(output, provider);

      results.push({
        output,
        length: output.length,
        tokenCount,
        latency,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      errors.push(errorMessage);

      // Add error sample
      results.push({
        output: '',
        length: 0,
        tokenCount: 0,
        latency: 0,
        error: errorMessage,
      });
    }
  }

  // Filter out error samples for statistics
  const validResults = results.filter(r => !r.error);

  if (validResults.length === 0) {
    throw new Error(`All ${samples} samples failed. Errors: ${errors.join(', ')}`);
  }

  // Calculate statistics
  const lengths = validResults.map(r => r.length);
  const tokens = validResults.map(r => r.tokenCount);

  const avgLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const avgTokens = tokens.reduce((a, b) => a + b, 0) / tokens.length;

  // Calculate variance
  const variance = lengths.reduce((sum, length) => {
    return sum + Math.pow(length - avgLength, 2);
  }, 0) / lengths.length;

  const stdDeviation = Math.sqrt(variance);

  // Estimate quality (placeholder)
  // In production, this could use:
  // - Consistency across samples (lower variance = higher quality)
  // - Length appropriateness
  // - Other heuristics
  const quality = estimateQuality(validResults, variance);

  const metrics: OutputMetrics = {
    avgLength: Math.round(avgLength),
    avgTokens: Math.round(avgTokens),
    variance: Math.round(variance),
    stdDeviation: Math.round(stdDeviation * 100) / 100,
    quality,
    samples: results,
    timestamp: new Date(),
  };

  // Cache results
  if (useCache) {
    cacheMetrics(prompt, provider.name, metrics);
  }

  return metrics;
}

/**
 * Estimate quality score based on output samples
 *
 * Lower variance typically indicates more consistent (higher quality) output
 * This is a simple heuristic - in production, use more sophisticated methods
 */
function estimateQuality(samples: OutputSample[], variance: number): number {
  if (samples.length === 0) return 0;

  // Consistency score (inverse of variance, normalized)
  // Lower variance = more consistent = higher quality
  const maxVariance = 10000; // Assumed max variance for normalization
  const consistencyScore = 1 - Math.min(variance / maxVariance, 1);

  // Length appropriateness (neither too short nor too long)
  const avgLength = samples.reduce((sum, s) => sum + s.length, 0) / samples.length;
  const idealLength = 500; // Assumed ideal length
  const lengthScore = 1 - Math.min(Math.abs(avgLength - idealLength) / idealLength, 1);

  // Combined quality score
  const quality = (consistencyScore * 0.6) + (lengthScore * 0.4);

  return Math.round(quality * 100) / 100;
}

/**
 * Compare output metrics between two prompts
 * Useful for A/B testing or evaluating prompt variations
 */
export function compareOutputMetrics(
  metricsA: OutputMetrics,
  metricsB: OutputMetrics
): {
  lengthDiff: number;
  tokenDiff: number;
  varianceDiff: number;
  qualityDiff: number;
  recommendation: string;
} {
  const lengthDiff = metricsB.avgLength - metricsA.avgLength;
  const tokenDiff = metricsB.avgTokens - metricsA.avgTokens;
  const varianceDiff = metricsB.variance - metricsA.variance;
  // Why: نتجنب فروقات floating point مثل -0.10000000000000009 التي تكسر اختبارات المساواة الصارمة.
  const qualityDiff = Math.round((metricsB.quality - metricsA.quality) * 100) / 100;

  // Generate recommendation
  let recommendation = '';

  if (qualityDiff > 0.1) {
    recommendation = 'Prompt B produces higher quality output';
  } else if (qualityDiff < -0.1) {
    recommendation = 'Prompt A produces higher quality output';
  } else if (tokenDiff < -50 && Math.abs(qualityDiff) < 0.05) {
    recommendation = 'Prompt B is more efficient (fewer tokens, similar quality)';
  } else if (tokenDiff > 50 && Math.abs(qualityDiff) < 0.05) {
    recommendation = 'Prompt A is more efficient (fewer tokens, similar quality)';
  } else {
    recommendation = 'Both prompts perform similarly';
  }

  return {
    lengthDiff,
    tokenDiff,
    varianceDiff,
    qualityDiff,
    recommendation,
  };
}

/**
 * Get metrics summary as a formatted string
 */
export function formatMetricsSummary(metrics: OutputMetrics): string {
  const successRate = (metrics.samples.filter(s => !s.error).length / metrics.samples.length) * 100;

  return `
Output Metrics Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Length: ${metrics.avgLength} characters
Average Tokens: ${metrics.avgTokens} tokens
Variance: ${metrics.variance}
Std Deviation: ${metrics.stdDeviation}
Quality Score: ${(metrics.quality * 100).toFixed(1)}%
Success Rate: ${successRate.toFixed(1)}%
Samples: ${metrics.samples.length}
Measured: ${metrics.timestamp.toLocaleString()}
━━━━━━━━━━━━━━━━━━━━━━━━━━
`.trim();
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/**
 * Measure metrics for multiple prompts in batch
 * Processes sequentially to avoid rate limits
 */
export async function measureBatch(
  prompts: string[],
  provider: LLMProvider,
  samplesPerPrompt: number = 3,
  onProgress?: (completed: number, total: number) => void
): Promise<Map<string, OutputMetrics>> {
  const results = new Map<string, OutputMetrics>();

  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];

    try {
      const metrics = await measureActualOutput(prompt, provider, samplesPerPrompt);
      results.set(prompt, metrics);
    } catch (error) {
      console.error(`Failed to measure prompt ${i + 1}:`, error);
    }

    if (onProgress) {
      onProgress(i + 1, prompts.length);
    }
  }

  return results;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  measureActualOutput,
  estimateTokenCount,
  compareOutputMetrics,
  formatMetricsSummary,
  measureBatch,
  cleanCache,
  clearCache,
};
