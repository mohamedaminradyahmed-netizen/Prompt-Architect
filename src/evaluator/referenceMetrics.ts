/**
 * Reference Metrics Module
 *
 * Implements ROUGE and BLEU metrics for evaluating generated text
 * against reference outputs
 *
 * Note: This is a TypeScript implementation. For production,
 * consider using established libraries like:
 * - rouge-score (Python)
 * - bleu (Python nltk)
 * - Or call Python scripts from Node.js
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation)
 * Measures overlap of n-grams between candidate and reference
 */
export interface ROUGEScores {
  rouge1: ROUGEScore;  // Unigram overlap
  rouge2: ROUGEScore;  // Bigram overlap
  rougeL: ROUGEScore;  // Longest Common Subsequence
}

/**
 * Individual ROUGE score with precision, recall, and F1
 */
export interface ROUGEScore {
  precision: number;  // 0-1
  recall: number;     // 0-1
  f1: number;         // 0-1
}

/**
 * BLEU score (Bilingual Evaluation Understudy)
 * Originally for machine translation, useful for any text generation
 */
export interface BLEUScore {
  score: number;           // Overall BLEU score (0-1)
  precisions: number[];    // Precision for each n-gram (1-4)
  brevityPenalty: number;  // Penalty for too-short outputs
  length: {
    candidate: number;
    reference: number;
  };
}

/**
 * Combined reference-based evaluation metrics
 */
export interface ReferenceMetrics {
  rouge: ROUGEScores;
  bleu: BLEUScore;
  overallScore: number;  // Weighted combination (0-100)
  recommendation: string;
}

// ============================================================================
// TOKENIZATION
// ============================================================================

/**
 * Simple tokenization: split on whitespace and lowercase
 * For production, use proper tokenizers (e.g., nltk, spacy)
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')  // Remove punctuation
    .split(/\s+/)
    .filter(token => token.length > 0);
}

/**
 * Create n-grams from tokens
 */
function createNGrams(tokens: string[], n: number): string[] {
  const ngrams: string[] = [];

  for (let i = 0; i <= tokens.length - n; i++) {
    const ngram = tokens.slice(i, i + n).join(' ');
    ngrams.push(ngram);
  }

  return ngrams;
}

// ============================================================================
// ROUGE IMPLEMENTATION
// ============================================================================

/**
 * Calculate ROUGE-N score (precision, recall, F1)
 * Compares n-gram overlap between candidate and reference
 */
function calculateRougeN(
  candidate: string,
  reference: string,
  n: number
): ROUGEScore {
  const candidateTokens = tokenize(candidate);
  const referenceTokens = tokenize(reference);

  const candidateNGrams = createNGrams(candidateTokens, n);
  const referenceNGrams = createNGrams(referenceTokens, n);

  if (referenceNGrams.length === 0 || candidateNGrams.length === 0) {
    return { precision: 0, recall: 0, f1: 0 };
  }

  // Count overlapping n-grams
  const candidateSet = new Set(candidateNGrams);
  const referenceSet = new Set(referenceNGrams);

  let overlap = 0;
  for (const ngram of candidateSet) {
    if (referenceSet.has(ngram)) {
      overlap++;
    }
  }

  // Calculate precision and recall
  const precision = overlap / candidateNGrams.length;
  const recall = overlap / referenceNGrams.length;

  // Calculate F1 score
  const f1 = precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;

  return {
    precision: Math.round(precision * 1000) / 1000,
    recall: Math.round(recall * 1000) / 1000,
    f1: Math.round(f1 * 1000) / 1000,
  };
}

/**
 * Calculate ROUGE-L score (Longest Common Subsequence)
 * Measures longest matching sequence of words
 */
function calculateRougeL(candidate: string, reference: string): ROUGEScore {
  const candidateTokens = tokenize(candidate);
  const referenceTokens = tokenize(reference);

  if (referenceTokens.length === 0 || candidateTokens.length === 0) {
    return { precision: 0, recall: 0, f1: 0 };
  }

  // Find longest common subsequence length
  const lcsLength = longestCommonSubsequence(candidateTokens, referenceTokens);

  // Calculate precision and recall
  const precision = lcsLength / candidateTokens.length;
  const recall = lcsLength / referenceTokens.length;

  // Calculate F1 score
  const f1 = precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;

  return {
    precision: Math.round(precision * 1000) / 1000,
    recall: Math.round(recall * 1000) / 1000,
    f1: Math.round(f1 * 1000) / 1000,
  };
}

/**
 * Find length of longest common subsequence using dynamic programming
 */
function longestCommonSubsequence(seq1: string[], seq2: string[]): number {
  const m = seq1.length;
  const n = seq2.length;

  // Create DP table
  const dp: number[][] = Array(m + 1)
    .fill(0)
    .map(() => Array(n + 1).fill(0));

  // Fill DP table
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (seq1[i - 1] === seq2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  return dp[m][n];
}

/**
 * Calculate all ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
 *
 * @param candidate - The generated text to evaluate
 * @param reference - The reference text to compare against
 * @returns ROUGEScores object with all metrics
 */
export function calculateROUGE(
  candidate: string,
  reference: string
): ROUGEScores {
  return {
    rouge1: calculateRougeN(candidate, reference, 1),
    rouge2: calculateRougeN(candidate, reference, 2),
    rougeL: calculateRougeL(candidate, reference),
  };
}

// ============================================================================
// BLEU IMPLEMENTATION
// ============================================================================

/**
 * Calculate modified precision for n-grams
 * Clips the count of each n-gram by its max reference count
 */
function calculateModifiedPrecision(
  candidateNGrams: string[],
  referenceNGrams: string[],
  n: number
): number {
  if (candidateNGrams.length === 0) return 0;

  // Count n-grams in candidate
  const candidateCounts = new Map<string, number>();
  for (const ngram of candidateNGrams) {
    candidateCounts.set(ngram, (candidateCounts.get(ngram) || 0) + 1);
  }

  // Count n-grams in reference
  const referenceCounts = new Map<string, number>();
  for (const ngram of referenceNGrams) {
    referenceCounts.set(ngram, (referenceCounts.get(ngram) || 0) + 1);
  }

  // Calculate clipped counts
  let clippedCount = 0;
  for (const [ngram, count] of candidateCounts) {
    const maxRefCount = referenceCounts.get(ngram) || 0;
    clippedCount += Math.min(count, maxRefCount);
  }

  return clippedCount / candidateNGrams.length;
}

/**
 * Calculate BLEU score
 *
 * @param candidate - The generated text to evaluate
 * @param references - Array of reference texts (supports multiple references)
 * @param maxN - Maximum n-gram size (default: 4)
 * @returns BLEUScore object
 */
export function calculateBLEU(
  candidate: string,
  references: string[],
  maxN: number = 4
): BLEUScore {
  if (references.length === 0) {
    throw new Error('At least one reference is required');
  }

  const candidateTokens = tokenize(candidate);
  const candidateLength = candidateTokens.length;

  if (candidateLength === 0) {
    return {
      score: 0,
      precisions: Array(maxN).fill(0),
      brevityPenalty: 0,
      length: { candidate: 0, reference: 0 },
    };
  }

  // Calculate precisions for each n-gram size
  const precisions: number[] = [];

  for (let n = 1; n <= maxN; n++) {
    const candidateNGrams = createNGrams(candidateTokens, n);

    // Calculate precision against all references (take max)
    let maxPrecision = 0;

    for (const reference of references) {
      const referenceTokens = tokenize(reference);
      const referenceNGrams = createNGrams(referenceTokens, n);
      const precision = calculateModifiedPrecision(
        candidateNGrams,
        referenceNGrams,
        n
      );
      maxPrecision = Math.max(maxPrecision, precision);
    }

    precisions.push(maxPrecision);
  }

  // Calculate brevity penalty
  // Find closest reference length
  const referenceLengths = references.map(ref => tokenize(ref).length);
  const closestRefLength = referenceLengths.reduce((closest, len) => {
    return Math.abs(len - candidateLength) < Math.abs(closest - candidateLength)
      ? len
      : closest;
  });

  const brevityPenalty = candidateLength >= closestRefLength
    ? 1.0
    : Math.exp(1 - closestRefLength / candidateLength);

  // Calculate geometric mean of precisions
  const geometricMean = precisions.reduce((product, p) => {
    // Avoid log(0) by using a small epsilon
    return product * Math.pow(Math.max(p, 1e-10), 1 / maxN);
  }, 1);

  const score = brevityPenalty * geometricMean;

  return {
    score: Math.round(score * 1000) / 1000,
    precisions: precisions.map(p => Math.round(p * 1000) / 1000),
    brevityPenalty: Math.round(brevityPenalty * 1000) / 1000,
    length: {
      candidate: candidateLength,
      reference: closestRefLength,
    },
  };
}

// ============================================================================
// COMBINED EVALUATION
// ============================================================================

/**
 * Evaluate prompt output against reference outputs
 * Combines ROUGE and BLEU metrics
 *
 * @param prompt - The original prompt
 * @param output - The generated output to evaluate
 * @param referenceOutputs - Array of reference outputs
 * @returns ReferenceMetrics with all scores and recommendation
 */
export function evaluateAgainstReference(
  prompt: string,
  output: string,
  referenceOutputs: string[]
): ReferenceMetrics {
  if (referenceOutputs.length === 0) {
    throw new Error('At least one reference output is required');
  }

  // Calculate ROUGE against first reference (or average across all)
  // For simplicity, we use the first reference
  // In production, you might want to calculate against all and average
  const rouge = calculateROUGE(output, referenceOutputs[0]);

  // Calculate BLEU against all references
  const bleu = calculateBLEU(output, referenceOutputs);

  // Calculate overall score (weighted combination)
  // ROUGE-L F1: 40%
  // BLEU: 40%
  // ROUGE-1 F1: 20%
  const overallScore = (
    rouge.rougeL.f1 * 0.4 +
    bleu.score * 0.4 +
    rouge.rouge1.f1 * 0.2
  ) * 100;

  // Generate recommendation
  const recommendation = generateRecommendation(rouge, bleu, overallScore);

  return {
    rouge,
    bleu,
    overallScore: Math.round(overallScore * 10) / 10,
    recommendation,
  };
}

/**
 * Generate recommendation based on scores
 */
function generateRecommendation(
  rouge: ROUGEScores,
  bleu: BLEUScore,
  overallScore: number
): string {
  if (overallScore >= 80) {
    return 'Excellent match with reference. Output is highly similar.';
  } else if (overallScore >= 60) {
    return 'Good match with reference. Output captures main content.';
  } else if (overallScore >= 40) {
    return 'Moderate match with reference. Some key points may be missing.';
  } else if (overallScore >= 20) {
    return 'Weak match with reference. Significant divergence from expected output.';
  } else {
    return 'Poor match with reference. Output does not align with expectations.';
  }
}

/**
 * Format metrics for display
 */
export function formatReferenceMetrics(metrics: ReferenceMetrics): string {
  return `
Reference Evaluation Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUGE-1:
  Precision: ${(metrics.rouge.rouge1.precision * 100).toFixed(1)}%
  Recall:    ${(metrics.rouge.rouge1.recall * 100).toFixed(1)}%
  F1:        ${(metrics.rouge.rouge1.f1 * 100).toFixed(1)}%

ROUGE-2:
  Precision: ${(metrics.rouge.rouge2.precision * 100).toFixed(1)}%
  Recall:    ${(metrics.rouge.rouge2.recall * 100).toFixed(1)}%
  F1:        ${(metrics.rouge.rouge2.f1 * 100).toFixed(1)}%

ROUGE-L:
  Precision: ${(metrics.rouge.rougeL.precision * 100).toFixed(1)}%
  Recall:    ${(metrics.rouge.rougeL.recall * 100).toFixed(1)}%
  F1:        ${(metrics.rouge.rougeL.f1 * 100).toFixed(1)}%

BLEU:
  Score:          ${(metrics.bleu.score * 100).toFixed(1)}%
  1-gram:         ${(metrics.bleu.precisions[0] * 100).toFixed(1)}%
  2-gram:         ${(metrics.bleu.precisions[1] * 100).toFixed(1)}%
  3-gram:         ${(metrics.bleu.precisions[2] * 100).toFixed(1)}%
  4-gram:         ${(metrics.bleu.precisions[3] * 100).toFixed(1)}%
  Brevity Penalty: ${metrics.bleu.brevityPenalty.toFixed(3)}

Overall Score: ${metrics.overallScore.toFixed(1)}/100
Recommendation: ${metrics.recommendation}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`.trim();
}

// ============================================================================
// BATCH EVALUATION
// ============================================================================

/**
 * Evaluate multiple outputs against references
 */
export function evaluateBatch(
  outputs: Array<{ prompt: string; output: string; references: string[] }>
): ReferenceMetrics[] {
  return outputs.map(({ prompt, output, references }) =>
    evaluateAgainstReference(prompt, output, references)
  );
}

/**
 * Compare two outputs against the same reference
 */
export function compareOutputs(
  outputA: string,
  outputB: string,
  references: string[]
): {
  metricsA: ReferenceMetrics;
  metricsB: ReferenceMetrics;
  winner: 'A' | 'B' | 'tie';
  scoreDiff: number;
} {
  const metricsA = evaluateAgainstReference('', outputA, references);
  const metricsB = evaluateAgainstReference('', outputB, references);

  const scoreDiff = metricsB.overallScore - metricsA.overallScore;

  let winner: 'A' | 'B' | 'tie';
  if (Math.abs(scoreDiff) < 5) {
    winner = 'tie';
  } else if (scoreDiff > 0) {
    winner = 'B';
  } else {
    winner = 'A';
  }

  return { metricsA, metricsB, winner, scoreDiff };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  calculateROUGE,
  calculateBLEU,
  evaluateAgainstReference,
  formatReferenceMetrics,
  evaluateBatch,
  compareOutputs,
};
