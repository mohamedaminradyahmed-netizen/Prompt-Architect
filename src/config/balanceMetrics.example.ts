/**
 * Balance Metrics Usage Examples
 *
 * This file demonstrates how to use the balance metrics system
 */

import {
  validateMetrics,
  calculateWeightedScore,
  getPreset,
  createCustomMetrics,
  SuggestionMetrics,
  COST_OPTIMIZED,
  QUALITY_FIRST,
  BALANCED,
  SPEED_OPTIMIZED,
} from './balanceMetrics';

// ============================================================================
// EXAMPLE 1: Using Presets
// ============================================================================

console.log('=== Example 1: Using Presets ===\n');

// Example suggestion metrics
const exampleSuggestion: SuggestionMetrics = {
  quality: 0.85,
  cost: 0.025,
  latency: 2500,
  hallucinationRate: 0.08,
  similarity: 0.75,
};

// Validate against BALANCED preset
console.log('Testing against BALANCED preset:');
const balancedResult = validateMetrics(exampleSuggestion, BALANCED);
console.log('Valid:', balancedResult.isValid);
console.log('Score:', balancedResult.score);
console.log('Passed:', balancedResult.passed);
console.log('Violations:', balancedResult.violations.length);
console.log('Recommendation:', balancedResult.recommendation);
console.log();

// Validate against QUALITY_FIRST preset
console.log('Testing against QUALITY_FIRST preset:');
const qualityResult = validateMetrics(exampleSuggestion, QUALITY_FIRST);
console.log('Valid:', qualityResult.isValid);
console.log('Score:', qualityResult.score);
console.log('Violations:', qualityResult.violations.map(v => v.message));
console.log('Recommendation:', qualityResult.recommendation);
console.log();

// Validate against COST_OPTIMIZED preset
console.log('Testing against COST_OPTIMIZED preset:');
const costResult = validateMetrics(exampleSuggestion, COST_OPTIMIZED);
console.log('Valid:', costResult.isValid);
console.log('Score:', costResult.score);
console.log('Recommendation:', costResult.recommendation);
console.log();

// ============================================================================
// EXAMPLE 2: Custom Metrics
// ============================================================================

console.log('=== Example 2: Custom Metrics ===\n');

// Create custom metrics based on BALANCED preset with overrides
const customMetrics = createCustomMetrics('balanced', {
  minQuality: 0.8,  // Higher quality requirement
  maxCost: 0.02,    // Lower cost limit
  weights: {
    quality: 0.4,   // Increase quality weight
    cost: 0.35,     // Increase cost weight
    latency: 0.15,
    reliability: 0.1,
  },
});

console.log('Custom Metrics Configuration:');
console.log(JSON.stringify(customMetrics, null, 2));
console.log();

const customResult = validateMetrics(exampleSuggestion, customMetrics);
console.log('Valid:', customResult.isValid);
console.log('Score:', customResult.score);
console.log('Recommendation:', customResult.recommendation);
console.log();

// ============================================================================
// EXAMPLE 3: Comparing Different Suggestions
// ============================================================================

console.log('=== Example 3: Comparing Multiple Suggestions ===\n');

const suggestions: Array<{ name: string; metrics: SuggestionMetrics }> = [
  {
    name: 'Fast & Cheap',
    metrics: {
      quality: 0.7,
      cost: 0.01,
      latency: 1200,
      hallucinationRate: 0.12,
      similarity: 0.65,
    },
  },
  {
    name: 'High Quality',
    metrics: {
      quality: 0.95,
      cost: 0.05,
      latency: 4000,
      hallucinationRate: 0.03,
      similarity: 0.88,
    },
  },
  {
    name: 'Balanced',
    metrics: {
      quality: 0.82,
      cost: 0.028,
      latency: 2200,
      hallucinationRate: 0.07,
      similarity: 0.78,
    },
  },
];

const preset = BALANCED;

console.log('Comparing suggestions using BALANCED preset:\n');
suggestions.forEach(({ name, metrics }) => {
  const result = validateMetrics(metrics, preset);
  console.log(`${name}:`);
  console.log(`  Score: ${result.score}/100`);
  console.log(`  Valid: ${result.isValid ? '✓' : '✗'}`);
  console.log(`  ${result.recommendation}`);
  console.log();
});

// Find best suggestion
const scored = suggestions.map(({ name, metrics }) => ({
  name,
  score: calculateWeightedScore(metrics, preset),
  validation: validateMetrics(metrics, preset),
}));

scored.sort((a, b) => b.score - a.score);

console.log('Best suggestion:', scored[0].name, `(${scored[0].score}/100)`);
console.log();

// ============================================================================
// EXAMPLE 4: Different Use Cases
// ============================================================================

console.log('=== Example 4: Use Case Specific Presets ===\n');

const testMetrics: SuggestionMetrics = {
  quality: 0.78,
  cost: 0.015,
  latency: 1800,
  hallucinationRate: 0.09,
  similarity: 0.72,
};

console.log('Same suggestion evaluated with different presets:\n');

// Cost-sensitive application
console.log('COST_OPTIMIZED (for high-volume applications):');
const costOptResult = validateMetrics(testMetrics, COST_OPTIMIZED);
console.log(`  Score: ${costOptResult.score}/100`);
console.log(`  ${costOptResult.recommendation}\n`);

// Quality-critical application
console.log('QUALITY_FIRST (for critical operations):');
const qualityFirstResult = validateMetrics(testMetrics, QUALITY_FIRST);
console.log(`  Score: ${qualityFirstResult.score}/100`);
console.log(`  ${qualityFirstResult.recommendation}\n`);

// Speed-critical application
console.log('SPEED_OPTIMIZED (for real-time applications):');
const speedResult = validateMetrics(testMetrics, SPEED_OPTIMIZED);
console.log(`  Score: ${speedResult.score}/100`);
console.log(`  ${speedResult.recommendation}\n`);

// Balanced application
console.log('BALANCED (for general use):');
const generalResult = validateMetrics(testMetrics, BALANCED);
console.log(`  Score: ${generalResult.score}/100`);
console.log(`  ${generalResult.recommendation}\n`);

// ============================================================================
// EXAMPLE 5: Integration with Evaluator
// ============================================================================

console.log('=== Example 5: Integration Pattern ===\n');

/**
 * Example of how to integrate with the existing evaluator system
 */
function evaluateWithBalanceMetrics(
  suggestion: any,
  balancePreset: 'cost-optimized' | 'quality-first' | 'balanced' | 'speed-optimized' = 'balanced'
) {
  // Get the preset
  const metrics = getPreset(balancePreset);

  // Convert suggestion to SuggestionMetrics format
  const suggestionMetrics: SuggestionMetrics = {
    quality: suggestion.score / 100,  // Convert from 0-100 to 0-1
    cost: suggestion.estimatedCost,
    latency: 2000, // Would be actual latency if measured
    hallucinationRate: 0.05, // Would come from hallucination detector
    similarity: suggestion.similarity,
  };

  // Validate
  const validation = validateMetrics(suggestionMetrics, metrics);

  return {
    ...suggestion,
    validation: {
      isValid: validation.isValid,
      overallScore: validation.score,
      violations: validation.violations,
      recommendation: validation.recommendation,
    },
  };
}

// Example usage
const mockSuggestion = {
  prompt: 'Refactored prompt...',
  mutation: 'paraphrase',
  score: 78,
  tokenCount: 45,
  estimatedCost: 0.022,
  similarity: 0.82,
};

const evaluatedSuggestion = evaluateWithBalanceMetrics(mockSuggestion, 'balanced');
console.log('Evaluated Suggestion:');
console.log(JSON.stringify(evaluatedSuggestion, null, 2));
