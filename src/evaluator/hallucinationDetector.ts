/**
 * Hallucination Detection Module
 *
 * Detects potential hallucinations in LLM outputs using multiple strategies:
 * 1. Consistency checking (compare multiple runs)
 * 2. Fact verification against context
 * 3. Confidence scoring from model logprobs
 *
 * Note: This is a heuristic-based implementation. For production,
 * consider using specialized models like SelfCheckGPT or fine-tuned classifiers.
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Hallucination detection result
 */
export interface HallucinationScore {
  score: number;              // 0-1, higher = more likely to be hallucination
  confidence: number;         // 0-1, confidence in the detection
  inconsistencies: string[];  // List of detected inconsistencies
  method: string;             // Detection method used
  details: HallucinationDetails;
}

/**
 * Detailed hallucination analysis
 */
export interface HallucinationDetails {
  consistencyScore: number;    // 0-1, from consistency check
  factualityScore: number;     // 0-1, from fact verification
  confidenceScore: number;     // 0-1, from logprobs
  claimsChecked: number;       // Number of claims verified
  claimsFailed: number;        // Number of claims that failed
}

/**
 * Configuration for hallucination detection
 */
export interface DetectionConfig {
  consistencyRuns: number;     // Number of times to run for consistency check
  similarityThreshold: number; // Threshold for considering outputs similar
  useLogprobs: boolean;        // Whether to use logprobs (if available)
  checkFacts: boolean;         // Whether to verify facts against context
  strictMode: boolean;         // Stricter detection (lower threshold)
}

/**
 * LLM provider for hallucination detection
 */
export interface LLMProvider {
  name: 'openai' | 'anthropic' | 'groq' | 'custom';
  apiKey?: string;
  model?: string;
  baseURL?: string;
  supportsLogprobs?: boolean;
}

/**
 * Individual claim extracted from output
 */
interface Claim {
  text: string;
  isFactual: boolean;
  confidence: number;
  reason?: string;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_CONFIG: DetectionConfig = {
  consistencyRuns: 2,
  similarityThreshold: 0.7,
  useLogprobs: true,
  checkFacts: true,
  strictMode: false,
};

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Calculate similarity between two texts using simple token overlap
 * For production, use semantic similarity (embeddings)
 */
function calculateSimilarity(text1: string, text2: string): number {
  const tokens1 = new Set(tokenize(text1));
  const tokens2 = new Set(tokenize(text2));

  const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
  const union = new Set([...tokens1, ...tokens2]);

  if (union.size === 0) return 0;

  return intersection.size / union.size;
}

/**
 * Simple tokenization
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(token => token.length > 0);
}

/**
 * Extract factual claims from text
 * Simple heuristic: sentences with verbs and entities
 */
function extractClaims(text: string): string[] {
  // Split into sentences
  const sentences = text
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);

  // Filter for sentences that look like factual claims
  // Heuristic: contains numbers, dates, proper nouns, or specific verbs
  const factualPatterns = [
    /\d+/,                    // Contains numbers
    /is|are|was|were|has|have/i, // Factual verbs
    /[A-Z][a-z]+\s[A-Z][a-z]+/,  // Proper nouns (rough heuristic)
  ];

  return sentences.filter(sentence =>
    factualPatterns.some(pattern => pattern.test(sentence))
  );
}

// ============================================================================
// MOCK LLM EXECUTION
// ============================================================================

/**
 * Mock LLM execution for demonstration
 * In production, replace with actual LLM API calls
 */
async function executeLLM(
  prompt: string,
  provider: LLMProvider,
  includeLogprobs: boolean = false
): Promise<{ output: string; logprobs?: number[] }> {
  // MOCK: In production, call actual LLM APIs:
  // - OpenAI: openai.chat.completions.create({ logprobs: true })
  // - Anthropic: Currently doesn't support logprobs
  // - Groq: groq.chat.completions.create({ logprobs: true })

  await new Promise(resolve => setTimeout(resolve, 100));

  // Mock output with slight variations for consistency testing
  const variation = Math.random() > 0.5 ? ' with details' : '';
  const mockOutput = `This is a response to: "${prompt.substring(0, 30)}..."${variation}`;

  // Mock logprobs (if requested)
  const logprobs = includeLogprobs
    ? Array.from({ length: 10 }, () => -0.1 - Math.random() * 2)
    : undefined;

  return { output: mockOutput, logprobs };
}

// ============================================================================
// DETECTION STRATEGIES
// ============================================================================

/**
 * Strategy 1: Consistency Check
 * Run the same prompt multiple times and check if outputs are consistent
 */
async function consistencyCheck(
  prompt: string,
  provider: LLMProvider,
  runs: number
): Promise<{ score: number; inconsistencies: string[] }> {
  const outputs: string[] = [];

  // Run prompt multiple times
  for (let i = 0; i < runs; i++) {
    const { output } = await executeLLM(prompt, provider);
    outputs.push(output);
  }

  // Compare all pairs of outputs
  const similarities: number[] = [];
  const inconsistencies: string[] = [];

  for (let i = 0; i < outputs.length - 1; i++) {
    for (let j = i + 1; j < outputs.length; j++) {
      const similarity = calculateSimilarity(outputs[i], outputs[j]);
      similarities.push(similarity);

      if (similarity < 0.5) {
        inconsistencies.push(
          `Run ${i + 1} differs significantly from Run ${j + 1} (${(similarity * 100).toFixed(1)}% similar)`
        );
      }
    }
  }

  // Calculate average similarity
  const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;

  // Consistency score: 1 - avgSimilarity (higher = more inconsistent = more likely hallucination)
  const consistencyScore = 1 - avgSimilarity;

  return {
    score: consistencyScore,
    inconsistencies,
  };
}

/**
 * Strategy 2: Fact Verification
 * Check if claims in output are supported by provided context
 */
function verifyFactsAgainstContext(
  output: string,
  context?: string
): { score: number; inconsistencies: string[] } {
  if (!context) {
    return { score: 0.5, inconsistencies: ['No context provided for verification'] };
  }

  // Extract claims from output
  const claims = extractClaims(output);

  if (claims.length === 0) {
    return { score: 0, inconsistencies: [] };
  }

  const unsupportedClaims: string[] = [];

  // Check each claim against context
  for (const claim of claims) {
    // Simple heuristic: check if key terms from claim appear in context
    const claimTokens = new Set(tokenize(claim));
    const contextTokens = new Set(tokenize(context));

    const overlap = [...claimTokens].filter(t => contextTokens.has(t)).length;
    const support = overlap / claimTokens.size;

    // If less than 30% of claim tokens appear in context, it's unsupported
    if (support < 0.3) {
      unsupportedClaims.push(`Unsupported claim: "${claim}" (${(support * 100).toFixed(1)}% support)`);
    }
  }

  // Fact verification score: ratio of unsupported claims
  const factScore = unsupportedClaims.length / claims.length;

  return {
    score: factScore,
    inconsistencies: unsupportedClaims,
  };
}

/**
 * Strategy 3: Confidence Scoring
 * Use model's logprobs to detect uncertain/low-confidence outputs
 */
async function confidenceScoring(
  prompt: string,
  provider: LLMProvider
): Promise<{ score: number; inconsistencies: string[] }> {
  if (!provider.supportsLogprobs) {
    return {
      score: 0,
      inconsistencies: ['Provider does not support logprobs'],
    };
  }

  const { logprobs } = await executeLLM(prompt, provider, true);

  if (!logprobs || logprobs.length === 0) {
    return { score: 0, inconsistencies: [] };
  }

  // Calculate average logprob
  const avgLogprob = logprobs.reduce((a, b) => a + b, 0) / logprobs.length;

  // Low logprobs (e.g., < -1.0) indicate uncertainty
  // Convert to 0-1 score where higher = more uncertain
  const uncertaintyThreshold = -1.0;
  const confidenceScore = avgLogprob < uncertaintyThreshold
    ? Math.min(1, Math.abs(avgLogprob) / 3) // Scale to 0-1
    : 0;

  const inconsistencies: string[] = [];
  if (confidenceScore > 0.5) {
    inconsistencies.push(
      `Low model confidence detected (avg logprob: ${avgLogprob.toFixed(3)})`
    );
  }

  return {
    score: confidenceScore,
    inconsistencies,
  };
}

// ============================================================================
// MAIN DETECTION FUNCTION
// ============================================================================

/**
 * Detect potential hallucinations in LLM output
 *
 * Combines multiple detection strategies:
 * 1. Consistency check (if enabled)
 * 2. Fact verification (if context provided)
 * 3. Confidence scoring (if logprobs available)
 *
 * @param prompt - The original prompt
 * @param output - The LLM output to check
 * @param provider - LLM provider configuration
 * @param context - Optional context for fact verification
 * @param config - Detection configuration
 * @returns HallucinationScore with detection results
 */
export async function detectHallucination(
  prompt: string,
  output: string,
  provider: LLMProvider,
  context?: string,
  config: Partial<DetectionConfig> = {}
): Promise<HallucinationScore> {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };

  const allInconsistencies: string[] = [];
  let consistencyScore = 0;
  let factualityScore = 0;
  let confidenceScore = 0;

  // Strategy 1: Consistency Check
  if (finalConfig.consistencyRuns > 1) {
    const consistency = await consistencyCheck(
      prompt,
      provider,
      finalConfig.consistencyRuns
    );
    consistencyScore = consistency.score;
    allInconsistencies.push(...consistency.inconsistencies);
  }

  // Strategy 2: Fact Verification
  if (finalConfig.checkFacts && context) {
    const factCheck = verifyFactsAgainstContext(output, context);
    factualityScore = factCheck.score;
    allInconsistencies.push(...factCheck.inconsistencies);
  }

  // Strategy 3: Confidence Scoring
  if (finalConfig.useLogprobs && provider.supportsLogprobs) {
    const confidence = await confidenceScoring(prompt, provider);
    confidenceScore = confidence.score;
    allInconsistencies.push(...confidence.inconsistencies);
  }

  // Combine scores (weighted average)
  const weights = {
    consistency: 0.4,
    factuality: 0.4,
    confidence: 0.2,
  };

  const overallScore =
    consistencyScore * weights.consistency +
    factualityScore * weights.factuality +
    confidenceScore * weights.confidence;

  // Calculate detection confidence
  const strategiesUsed = [
    consistencyScore > 0,
    factualityScore > 0,
    confidenceScore > 0,
  ].filter(Boolean).length;

  const detectionConfidence = strategiesUsed / 3;

  // Determine method used
  const methods: string[] = [];
  if (consistencyScore > 0) methods.push('consistency');
  if (factualityScore > 0) methods.push('factuality');
  if (confidenceScore > 0) methods.push('confidence');

  // Count claims
  const claims = extractClaims(output);
  const claimsFailed = allInconsistencies.filter(inc =>
    inc.includes('Unsupported claim')
  ).length;

  return {
    score: Math.round(overallScore * 1000) / 1000,
    confidence: Math.round(detectionConfidence * 1000) / 1000,
    inconsistencies: allInconsistencies,
    method: methods.join(', '),
    details: {
      consistencyScore: Math.round(consistencyScore * 1000) / 1000,
      factualityScore: Math.round(factualityScore * 1000) / 1000,
      confidenceScore: Math.round(confidenceScore * 1000) / 1000,
      claimsChecked: claims.length,
      claimsFailed,
    },
  };
}

/**
 * Check if output is likely a hallucination based on threshold
 */
export function isHallucination(
  hallucinationScore: HallucinationScore,
  threshold: number = 0.5
): boolean {
  return hallucinationScore.score >= threshold;
}

/**
 * Get hallucination severity level
 */
export function getHallucinationSeverity(
  score: number
): 'none' | 'low' | 'medium' | 'high' {
  if (score < 0.2) return 'none';
  if (score < 0.4) return 'low';
  if (score < 0.7) return 'medium';
  return 'high';
}

/**
 * Format hallucination score for display
 */
export function formatHallucinationScore(score: HallucinationScore): string {
  const severity = getHallucinationSeverity(score.score);
  const percentage = (score.score * 100).toFixed(1);

  let report = `
Hallucination Detection Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score: ${percentage}% (${severity})
Detection Confidence: ${(score.confidence * 100).toFixed(1)}%
Method: ${score.method}

Strategy Breakdown:
  Consistency: ${(score.details.consistencyScore * 100).toFixed(1)}%
  Factuality: ${(score.details.factualityScore * 100).toFixed(1)}%
  Confidence: ${(score.details.confidenceScore * 100).toFixed(1)}%

Claims Analysis:
  Total Claims: ${score.details.claimsChecked}
  Failed Claims: ${score.details.claimsFailed}
`;

  if (score.inconsistencies.length > 0) {
    report += `\nInconsistencies Found:\n`;
    score.inconsistencies.forEach((inc, i) => {
      report += `  ${i + 1}. ${inc}\n`;
    });
  }

  report += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

  return report.trim();
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/**
 * Detect hallucinations in multiple outputs
 */
export async function detectHallucinationBatch(
  outputs: Array<{
    prompt: string;
    output: string;
    context?: string;
  }>,
  provider: LLMProvider,
  config?: Partial<DetectionConfig>,
  onProgress?: (completed: number, total: number) => void
): Promise<HallucinationScore[]> {
  const results: HallucinationScore[] = [];

  for (let i = 0; i < outputs.length; i++) {
    const { prompt, output, context } = outputs[i];

    const score = await detectHallucination(
      prompt,
      output,
      provider,
      context,
      config
    );

    results.push(score);

    if (onProgress) {
      onProgress(i + 1, outputs.length);
    }
  }

  return results;
}

/**
 * Compare hallucination scores between two outputs
 */
export function compareHallucinationScores(
  scoreA: HallucinationScore,
  scoreB: HallucinationScore
): {
  better: 'A' | 'B' | 'tie';
  scoreDiff: number;
  recommendation: string;
} {
  const scoreDiff = scoreB.score - scoreA.score;

  let better: 'A' | 'B' | 'tie';
  if (Math.abs(scoreDiff) < 0.1) {
    better = 'tie';
  } else if (scoreDiff > 0) {
    better = 'A'; // A has lower hallucination score
  } else {
    better = 'B'; // B has lower hallucination score
  }

  let recommendation = '';
  if (better === 'tie') {
    recommendation = 'Both outputs have similar hallucination risks';
  } else {
    const winner = better === 'A' ? 'A' : 'B';
    const diff = Math.abs(scoreDiff * 100).toFixed(1);
    recommendation = `Output ${winner} is more reliable (${diff}% lower hallucination score)`;
  }

  return { better, scoreDiff, recommendation };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  detectHallucination,
  isHallucination,
  getHallucinationSeverity,
  formatHallucinationScore,
  detectHallucinationBatch,
  compareHallucinationScores,
};
