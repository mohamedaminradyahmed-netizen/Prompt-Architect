/**
 * Balance Metrics Configuration
 *
 * This module defines the balance criteria between:
 * - Quality/Accuracy
 * - Cost
 * - Latency
 * - Reliability
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Core balance metrics that define the acceptable thresholds
 * for prompt variations
 */
export interface BalanceMetrics {
  /** Minimum acceptable quality/accuracy score (0-1) */
  minQuality: number;

  /** Maximum acceptable cost per request in USD */
  maxCost: number;

  /** Maximum acceptable latency in milliseconds */
  maxLatency: number;

  /** Maximum acceptable hallucination rate (0-1, lower is better) */
  maxHallucinationRate: number;

  /** Minimum similarity to original prompt (0-1) */
  minSimilarity: number;

  /** Custom weights for scoring (sum should be 1.0) */
  weights: MetricWeights;
}

/**
 * Weights for each metric in the overall scoring function
 * All weights should sum to 1.0
 */
export interface MetricWeights {
  quality: number;      // Weight for quality/accuracy
  cost: number;         // Weight for cost efficiency
  latency: number;      // Weight for speed
  reliability: number;  // Weight for hallucination/reliability
}

/**
 * Validation result for a suggestion against balance metrics
 */
export interface ValidationResult {
  isValid: boolean;
  score: number;                    // Overall score (0-100)
  violations: MetricViolation[];    // List of violations
  passed: string[];                 // List of passed criteria
  recommendation: string;           // Human-readable recommendation
}

/**
 * Individual metric violation
 */
export interface MetricViolation {
  metric: keyof BalanceMetrics;
  threshold: number;
  actual: number;
  severity: 'low' | 'medium' | 'high';
  message: string;
}

/**
 * Suggestion metrics to validate
 */
export interface SuggestionMetrics {
  quality: number;           // 0-1
  cost: number;             // USD
  latency: number;          // milliseconds
  hallucinationRate: number; // 0-1
  similarity: number;       // 0-1
}

/**
 * Preset configuration type
 */
export type PresetType = 'cost-optimized' | 'quality-first' | 'balanced' | 'speed-optimized';

// ============================================================================
// PRESETS
// ============================================================================

/**
 * Cost-Optimized Preset
 * Prioritizes low cost while maintaining acceptable quality
 */
export const COST_OPTIMIZED: BalanceMetrics = {
  minQuality: 0.6,              // Moderate quality threshold
  maxCost: 0.01,                // Very low cost limit ($0.01)
  maxLatency: 5000,             // 5 seconds max
  maxHallucinationRate: 0.2,    // 20% hallucination acceptable
  minSimilarity: 0.5,           // Moderate similarity
  weights: {
    quality: 0.2,
    cost: 0.5,                  // Heavy cost weight
    latency: 0.15,
    reliability: 0.15,
  },
};

/**
 * Quality-First Preset
 * Prioritizes maximum quality regardless of cost
 */
export const QUALITY_FIRST: BalanceMetrics = {
  minQuality: 0.9,              // Very high quality threshold
  maxCost: 0.1,                 // Higher cost acceptable ($0.10)
  maxLatency: 10000,            // 10 seconds max
  maxHallucinationRate: 0.05,   // Only 5% hallucination acceptable
  minSimilarity: 0.8,           // High similarity to original
  weights: {
    quality: 0.5,               // Heavy quality weight
    cost: 0.1,
    latency: 0.15,
    reliability: 0.25,
  },
};

/**
 * Balanced Preset
 * Equal balance between all metrics
 */
export const BALANCED: BalanceMetrics = {
  minQuality: 0.75,             // Good quality threshold
  maxCost: 0.03,                // Moderate cost ($0.03)
  maxLatency: 3000,             // 3 seconds max
  maxHallucinationRate: 0.1,    // 10% hallucination acceptable
  minSimilarity: 0.7,           // Good similarity
  weights: {
    quality: 0.3,
    cost: 0.3,
    latency: 0.2,
    reliability: 0.2,
  },
};

/**
 * Speed-Optimized Preset
 * Prioritizes fast response times
 */
export const SPEED_OPTIMIZED: BalanceMetrics = {
  minQuality: 0.65,             // Moderate quality
  maxCost: 0.02,                // Low cost
  maxLatency: 1500,             // Very fast (1.5 seconds)
  maxHallucinationRate: 0.15,   // 15% hallucination acceptable
  minSimilarity: 0.6,           // Moderate similarity
  weights: {
    quality: 0.2,
    cost: 0.2,
    latency: 0.45,              // Heavy latency weight
    reliability: 0.15,
  },
};

/**
 * All available presets
 */
export const PRESETS: Record<PresetType, BalanceMetrics> = {
  'cost-optimized': COST_OPTIMIZED,
  'quality-first': QUALITY_FIRST,
  'balanced': BALANCED,
  'speed-optimized': SPEED_OPTIMIZED,
};

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validates a suggestion against the specified balance metrics
 *
 * @param metrics - The suggestion metrics to validate
 * @param balanceMetrics - The balance criteria to validate against
 * @returns ValidationResult with pass/fail status and details
 */
export function validateMetrics(
  metrics: SuggestionMetrics,
  balanceMetrics: BalanceMetrics
): ValidationResult {
  const violations: MetricViolation[] = [];
  const passed: string[] = [];

  // Check quality
  if (metrics.quality < balanceMetrics.minQuality) {
    violations.push({
      metric: 'minQuality',
      threshold: balanceMetrics.minQuality,
      actual: metrics.quality,
      severity: getSeverity(
        metrics.quality,
        balanceMetrics.minQuality,
        'min'
      ),
      message: `Quality ${(metrics.quality * 100).toFixed(1)}% is below minimum ${(balanceMetrics.minQuality * 100).toFixed(1)}%`,
    });
  } else {
    passed.push('quality');
  }

  // Check cost
  if (metrics.cost > balanceMetrics.maxCost) {
    violations.push({
      metric: 'maxCost',
      threshold: balanceMetrics.maxCost,
      actual: metrics.cost,
      severity: getSeverity(
        metrics.cost,
        balanceMetrics.maxCost,
        'max'
      ),
      message: `Cost $${metrics.cost.toFixed(4)} exceeds maximum $${balanceMetrics.maxCost.toFixed(4)}`,
    });
  } else {
    passed.push('cost');
  }

  // Check latency
  if (metrics.latency > balanceMetrics.maxLatency) {
    violations.push({
      metric: 'maxLatency',
      threshold: balanceMetrics.maxLatency,
      actual: metrics.latency,
      severity: getSeverity(
        metrics.latency,
        balanceMetrics.maxLatency,
        'max'
      ),
      message: `Latency ${metrics.latency}ms exceeds maximum ${balanceMetrics.maxLatency}ms`,
    });
  } else {
    passed.push('latency');
  }

  // Check hallucination rate
  if (metrics.hallucinationRate > balanceMetrics.maxHallucinationRate) {
    violations.push({
      metric: 'maxHallucinationRate',
      threshold: balanceMetrics.maxHallucinationRate,
      actual: metrics.hallucinationRate,
      severity: getSeverity(
        metrics.hallucinationRate,
        balanceMetrics.maxHallucinationRate,
        'max'
      ),
      message: `Hallucination rate ${(metrics.hallucinationRate * 100).toFixed(1)}% exceeds maximum ${(balanceMetrics.maxHallucinationRate * 100).toFixed(1)}%`,
    });
  } else {
    passed.push('reliability');
  }

  // Check similarity
  if (metrics.similarity < balanceMetrics.minSimilarity) {
    violations.push({
      metric: 'minSimilarity',
      threshold: balanceMetrics.minSimilarity,
      actual: metrics.similarity,
      severity: getSeverity(
        metrics.similarity,
        balanceMetrics.minSimilarity,
        'min'
      ),
      message: `Similarity ${(metrics.similarity * 100).toFixed(1)}% is below minimum ${(balanceMetrics.minSimilarity * 100).toFixed(1)}%`,
    });
  } else {
    passed.push('similarity');
  }

  // Calculate overall score
  const score = calculateWeightedScore(metrics, balanceMetrics);

  // Generate recommendation
  const recommendation = generateRecommendation(violations, score);

  return {
    isValid: violations.length === 0,
    score,
    violations,
    passed,
    recommendation,
  };
}

/**
 * Calculate weighted score based on metrics and weights
 *
 * @param metrics - The suggestion metrics
 * @param balanceMetrics - The balance criteria with weights
 * @returns Overall score (0-100)
 */
export function calculateWeightedScore(
  metrics: SuggestionMetrics,
  balanceMetrics: BalanceMetrics
): number {
  const weights = balanceMetrics.weights;

  // Normalize each metric to 0-1 scale
  const qualityScore = metrics.quality;
  const costScore = 1 - Math.min(metrics.cost / balanceMetrics.maxCost, 1);
  const latencyScore = 1 - Math.min(metrics.latency / balanceMetrics.maxLatency, 1);
  const reliabilityScore = 1 - metrics.hallucinationRate;

  // Calculate weighted sum
  const weightedScore =
    qualityScore * weights.quality +
    costScore * weights.cost +
    latencyScore * weights.latency +
    reliabilityScore * weights.reliability;

  // Convert to 0-100 scale
  return Math.round(weightedScore * 100);
}

/**
 * Determine severity of a violation based on how far from threshold
 */
function getSeverity(
  actual: number,
  threshold: number,
  type: 'min' | 'max'
): 'low' | 'medium' | 'high' {
  let deviation: number;

  if (type === 'min') {
    // For minimum thresholds, deviation is how far below
    deviation = (threshold - actual) / threshold;
  } else {
    // For maximum thresholds, deviation is how far above
    deviation = (actual - threshold) / threshold;
  }

  if (deviation < 0.1) return 'low';      // Within 10%
  if (deviation < 0.3) return 'medium';   // Within 30%
  return 'high';                          // More than 30%
}

/**
 * Generate human-readable recommendation based on violations
 */
function generateRecommendation(
  violations: MetricViolation[],
  score: number
): string {
  if (violations.length === 0) {
    if (score >= 90) {
      return 'Excellent! This variation meets all criteria with high scores.';
    } else if (score >= 75) {
      return 'Good! This variation meets all criteria.';
    } else {
      return 'Acceptable. This variation meets minimum criteria but could be improved.';
    }
  }

  const highSeverity = violations.filter(v => v.severity === 'high');
  if (highSeverity.length > 0) {
    const metrics = highSeverity.map(v => v.metric).join(', ');
    return `Critical issues with: ${metrics}. Consider rejecting this variation.`;
  }

  const mediumSeverity = violations.filter(v => v.severity === 'medium');
  if (mediumSeverity.length > 0) {
    const metrics = mediumSeverity.map(v => v.metric).join(', ');
    return `Moderate issues with: ${metrics}. Review carefully before accepting.`;
  }

  return 'Minor issues detected. Acceptable with caution.';
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get a preset configuration by type
 *
 * @param type - The preset type
 * @returns The balance metrics for that preset
 */
export function getPreset(type: PresetType): BalanceMetrics {
  return PRESETS[type];
}

/**
 * Create custom balance metrics by merging with a preset
 *
 * @param preset - Base preset to start with
 * @param overrides - Custom overrides to apply
 * @returns Merged balance metrics
 */
export function createCustomMetrics(
  preset: PresetType,
  overrides: Partial<BalanceMetrics>
): BalanceMetrics {
  return {
    ...PRESETS[preset],
    ...overrides,
    weights: {
      ...PRESETS[preset].weights,
      ...(overrides.weights || {}),
    },
  };
}

/**
 * Validate that metric weights sum to 1.0
 *
 * @param weights - The weights to validate
 * @returns True if valid, false otherwise
 */
export function validateWeights(weights: MetricWeights): boolean {
  const sum = weights.quality + weights.cost + weights.latency + weights.reliability;
  return Math.abs(sum - 1.0) < 0.001; // Allow small floating point errors
}

/**
 * Normalize weights to sum to 1.0
 *
 * @param weights - The weights to normalize
 * @returns Normalized weights
 */
export function normalizeWeights(weights: MetricWeights): MetricWeights {
  const sum = weights.quality + weights.cost + weights.latency + weights.reliability;

  if (sum === 0) {
    // If all weights are 0, use equal weights
    return { quality: 0.25, cost: 0.25, latency: 0.25, reliability: 0.25 };
  }

  return {
    quality: weights.quality / sum,
    cost: weights.cost / sum,
    latency: weights.latency / sum,
    reliability: weights.reliability / sum,
  };
}

/**
 * Compare two balance metrics configurations
 *
 * @param a - First metrics
 * @param b - Second metrics
 * @returns True if they are equivalent
 */
export function areMetricsEqual(a: BalanceMetrics, b: BalanceMetrics): boolean {
  return (
    a.minQuality === b.minQuality &&
    a.maxCost === b.maxCost &&
    a.maxLatency === b.maxLatency &&
    a.maxHallucinationRate === b.maxHallucinationRate &&
    a.minSimilarity === b.minSimilarity &&
    a.weights.quality === b.weights.quality &&
    a.weights.cost === b.weights.cost &&
    a.weights.latency === b.weights.latency &&
    a.weights.reliability === b.weights.reliability
  );
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  COST_OPTIMIZED,
  QUALITY_FIRST,
  BALANCED,
  SPEED_OPTIMIZED,
  PRESETS,
  validateMetrics,
  calculateWeightedScore,
  getPreset,
  createCustomMetrics,
  validateWeights,
  normalizeWeights,
  areMetricsEqual,
};
