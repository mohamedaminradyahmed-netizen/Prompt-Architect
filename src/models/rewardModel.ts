/**
 * Reward Model - DIRECTIVE-034
 *
 * A lightweight model to predict the quality of prompt variations.
 * Trains on human feedback to score variations without constant human input.
 */

import { PromptCategory } from '../types/promptTypes';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Training example from human feedback
 */
export interface TrainingExample {
  id: string;
  originalPrompt: string;
  modifiedPrompt: string;
  context?: string;
  outputs: {
    original: string;
    modified: string;
  };
  humanScore: number;  // 1-5 rating
  feedback?: string;
  metadata: {
    category: PromptCategory;
    mutationType: string;
    timestamp: Date;
    userId?: string;
  };
}

/**
 * Features extracted from a prompt pair for model input
 */
export interface PromptFeatures {
  // Length features
  originalLength: number;
  modifiedLength: number;
  lengthRatio: number;
  lengthDiff: number;

  // Lexical features
  vocabularyRichness: number;     // Unique words / Total words
  avgWordLength: number;
  sentenceCount: number;

  // Structural features
  hasImperativeVerb: boolean;
  hasConstraints: boolean;
  hasExamples: boolean;
  hasContext: boolean;

  // Similarity features
  tokenOverlap: number;           // Jaccard similarity
  semanticSimilarity: number;     // Cosine similarity (if embeddings available)

  // Mutation features
  mutationType: string;
  category: PromptCategory;

  // Quality indicators
  clarityScore: number;           // 0-1, based on heuristics
  specificityScore: number;       // 0-1, based on heuristics
  completenessScore: number;      // 0-1, based on heuristics
}

/**
 * Trained model weights and configuration
 */
export interface RewardModelWeights {
  version: string;
  weights: Record<string, number>;
  intercept: number;
  normalization: {
    mean: Record<string, number>;
    std: Record<string, number>;
  };
  metadata: {
    trainedOn: number;             // Number of examples
    trainDate: Date;
    mae: number;                   // Mean Absolute Error
    rmse: number;                  // Root Mean Squared Error
    correlation: number;           // Correlation with human scores
  };
}

/**
 * Reward prediction result
 */
export interface RewardPrediction {
  score: number;                   // Predicted score (0-1)
  confidence: number;              // Confidence in prediction (0-1)
  breakdown: Record<string, number>; // Feature contributions
  explanation: string;             // Human-readable explanation
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

/**
 * Extract features from a prompt pair
 */
export function extractFeatures(
  original: string,
  modified: string,
  mutationType: string,
  category: PromptCategory,
  context?: string
): PromptFeatures {
  const origTokens = tokenize(original);
  const modTokens = tokenize(modified);

  return {
    // Length features
    originalLength: original.length,
    modifiedLength: modified.length,
    lengthRatio: modified.length / (original.length || 1),
    lengthDiff: modified.length - original.length,

    // Lexical features
    vocabularyRichness: calculateVocabularyRichness(modTokens),
    avgWordLength: calculateAvgWordLength(modTokens),
    sentenceCount: modified.split(/[.!?]+/).filter(s => s.trim()).length,

    // Structural features
    hasImperativeVerb: /^(write|create|make|build|implement|fix|analyze|explain)/i.test(modified),
    hasConstraints: /\b(must|should|ensure|require|within|limit)\b/i.test(modified),
    hasExamples: /\b(example|such as|like|for instance|e\.g\.)\b/i.test(modified),
    hasContext: Boolean(context),

    // Similarity features
    tokenOverlap: calculateJaccardSimilarity(origTokens, modTokens),
    semanticSimilarity: 0.5, // Placeholder - would use embeddings in real implementation

    // Mutation features
    mutationType,
    category,

    // Quality indicators
    clarityScore: calculateClarityScore(modified),
    specificityScore: calculateSpecificityScore(modified),
    completenessScore: calculateCompletenessScore(modified),
  };
}

/**
 * Simple tokenization
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);
}

/**
 * Calculate vocabulary richness (type-token ratio)
 */
function calculateVocabularyRichness(tokens: string[]): number {
  if (tokens.length === 0) return 0;
  const uniqueTokens = new Set(tokens);
  return uniqueTokens.size / tokens.length;
}

/**
 * Calculate average word length
 */
function calculateAvgWordLength(tokens: string[]): number {
  if (tokens.length === 0) return 0;
  const totalLength = tokens.reduce((sum, token) => sum + token.length, 0);
  return totalLength / tokens.length;
}

/**
 * Calculate Jaccard similarity (token overlap)
 */
function calculateJaccardSimilarity(tokens1: string[], tokens2: string[]): number {
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);

  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  return union.size > 0 ? intersection.size / union.size : 0;
}

/**
 * Heuristic clarity score
 */
function calculateClarityScore(text: string): number {
  let score = 0.5;

  // Clear action verbs
  if (/^(write|create|build|implement|explain|analyze)/i.test(text)) {
    score += 0.2;
  }

  // Well-structured sentences
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  if (sentences.length >= 2 && sentences.length <= 5) {
    score += 0.1;
  }

  // Not too short or too long
  if (text.length >= 50 && text.length <= 500) {
    score += 0.1;
  }

  // Has specific terminology
  if (/\b(function|class|system|process|method|algorithm)\b/i.test(text)) {
    score += 0.1;
  }

  return Math.min(1, score);
}

/**
 * Heuristic specificity score
 */
function calculateSpecificityScore(text: string): number {
  let score = 0.3;

  // Contains specific requirements
  if (/\b(typescript|javascript|python|async|await|interface)\b/i.test(text)) {
    score += 0.2;
  }

  // Contains constraints
  if (/\b(must|should|ensure|require|within)\b/i.test(text)) {
    score += 0.2;
  }

  // Contains examples
  if (/\b(example|such as|like|for instance)\b/i.test(text)) {
    score += 0.15;
  }

  // Contains numbers/metrics
  if (/\b\d+\b/.test(text)) {
    score += 0.15;
  }

  return Math.min(1, score);
}

/**
 * Heuristic completeness score
 */
function calculateCompletenessScore(text: string): number {
  let score = 0.3;

  // Has context
  if (/\b(context|background|given|assume)\b/i.test(text)) {
    score += 0.15;
  }

  // Has goal
  if (/\b(create|build|implement|write|develop)\b/i.test(text)) {
    score += 0.15;
  }

  // Has constraints
  if (/\b(must|should|ensure|require)\b/i.test(text)) {
    score += 0.2;
  }

  // Has success criteria or examples
  if (/\b(example|output|result|should produce)\b/i.test(text)) {
    score += 0.2;
  }

  return Math.min(1, score);
}

// ============================================================================
// REWARD MODEL CLASS
// ============================================================================

/**
 * Simple linear regression-based reward model
 *
 * Uses weighted feature combination to predict prompt quality.
 * In production, this could be replaced with a more sophisticated model (XGBoost, neural net, etc.)
 */
export class RewardModel {
  private weights: RewardModelWeights;

  constructor(weights?: RewardModelWeights) {
    this.weights = weights || this.getDefaultWeights();
  }

  /**
   * Get default (untrained) weights
   */
  private getDefaultWeights(): RewardModelWeights {
    return {
      version: '1.0.0',
      intercept: 0.5,
      weights: {
        // Length features (modest impact)
        lengthRatio: 0.05,

        // Lexical features (moderate impact)
        vocabularyRichness: 0.1,
        avgWordLength: 0.02,

        // Structural features (high impact)
        hasImperativeVerb: 0.15,
        hasConstraints: 0.15,
        hasExamples: 0.1,
        hasContext: 0.05,

        // Similarity features (moderate impact)
        tokenOverlap: -0.05,  // Too similar might mean no improvement

        // Quality indicators (highest impact)
        clarityScore: 0.2,
        specificityScore: 0.2,
        completenessScore: 0.25,
      },
      normalization: {
        mean: {},
        std: {},
      },
      metadata: {
        trainedOn: 0,
        trainDate: new Date(),
        mae: 0,
        rmse: 0,
        correlation: 0,
      },
    };
  }

  /**
   * Predict quality score for a prompt variation
   */
  predict(
    original: string,
    modified: string,
    mutationType: string,
    category: PromptCategory,
    context?: string
  ): RewardPrediction {
    // Extract features
    const features = extractFeatures(original, modified, mutationType, category, context);

    // Calculate weighted score
    let score = this.weights.intercept;
    const breakdown: Record<string, number> = {};

    // Numeric features
    const numericFeatures: Record<string, number> = {
      lengthRatio: features.lengthRatio,
      vocabularyRichness: features.vocabularyRichness,
      avgWordLength: features.avgWordLength,
      tokenOverlap: features.tokenOverlap,
      clarityScore: features.clarityScore,
      specificityScore: features.specificityScore,
      completenessScore: features.completenessScore,
    };

    for (const [key, value] of Object.entries(numericFeatures)) {
      const weight = this.weights.weights[key] || 0;
      const contribution = weight * value;
      score += contribution;
      breakdown[key] = contribution;
    }

    // Boolean features
    const booleanFeatures: Record<string, boolean> = {
      hasImperativeVerb: features.hasImperativeVerb,
      hasConstraints: features.hasConstraints,
      hasExamples: features.hasExamples,
      hasContext: features.hasContext,
    };

    for (const [key, value] of Object.entries(booleanFeatures)) {
      const weight = this.weights.weights[key] || 0;
      const contribution = value ? weight : 0;
      score += contribution;
      breakdown[key] = contribution;
    }

    // Normalize to 0-1 range
    score = Math.max(0, Math.min(1, score));

    // Calculate confidence based on feature certainty
    const confidence = this.calculateConfidence(features);

    // Generate explanation
    const explanation = this.generateExplanation(features, breakdown, score);

    return {
      score,
      confidence,
      breakdown,
      explanation,
    };
  }

  /**
   * Calculate confidence in the prediction
   */
  private calculateConfidence(features: PromptFeatures): number {
    // Higher confidence when features are clear and well-defined
    let confidence = 0.5;

    // Structural clarity increases confidence
    if (features.hasImperativeVerb) confidence += 0.1;
    if (features.hasConstraints) confidence += 0.1;
    if (features.clarityScore > 0.7) confidence += 0.15;
    if (features.specificityScore > 0.7) confidence += 0.15;

    return Math.min(1, confidence);
  }

  /**
   * Generate human-readable explanation
   */
  private generateExplanation(
    features: PromptFeatures,
    breakdown: Record<string, number>,
    score: number
  ): string {
    const strengths: string[] = [];
    const weaknesses: string[] = [];

    // Analyze quality indicators
    if (features.clarityScore > 0.7) {
      strengths.push('clear and well-structured');
    } else if (features.clarityScore < 0.4) {
      weaknesses.push('lacks clarity');
    }

    if (features.specificityScore > 0.7) {
      strengths.push('highly specific');
    } else if (features.specificityScore < 0.4) {
      weaknesses.push('too vague');
    }

    if (features.completenessScore > 0.7) {
      strengths.push('comprehensive');
    } else if (features.completenessScore < 0.4) {
      weaknesses.push('incomplete');
    }

    // Analyze structural features
    if (features.hasImperativeVerb) {
      strengths.push('clear action verb');
    }

    if (features.hasConstraints) {
      strengths.push('well-defined constraints');
    }

    if (features.hasExamples) {
      strengths.push('includes examples');
    }

    // Build explanation
    let explanation = `Score: ${(score * 100).toFixed(1)}%. `;

    if (strengths.length > 0) {
      explanation += `Strengths: ${strengths.join(', ')}. `;
    }

    if (weaknesses.length > 0) {
      explanation += `Weaknesses: ${weaknesses.join(', ')}. `;
    }

    if (strengths.length === 0 && weaknesses.length === 0) {
      explanation += 'Average quality variation.';
    }

    return explanation.trim();
  }

  /**
   * Train the model on labeled examples
   *
   * Simple linear regression implementation.
   * For production, consider using a proper ML library.
   */
  train(examples: TrainingExample[]): void {
    if (examples.length === 0) {
      throw new Error('Cannot train on empty dataset');
    }

    // Extract features for all examples
    const featureVectors: PromptFeatures[] = [];
    const targets: number[] = [];

    for (const example of examples) {
      const features = extractFeatures(
        example.originalPrompt,
        example.modifiedPrompt,
        example.metadata.mutationType,
        example.metadata.category,
        example.context
      );
      featureVectors.push(features);

      // Normalize human score to 0-1
      targets.push((example.humanScore - 1) / 4);
    }

    // Simple linear regression using closed-form solution
    // In production, use a proper optimization library

    // For now, use simple averaging with weights
    const avgTarget = targets.reduce((sum, t) => sum + t, 0) / targets.length;

    // Update intercept to match average target
    this.weights.intercept = avgTarget - 0.2; // Rough adjustment

    // Update metadata
    this.weights.metadata.trainedOn = examples.length;
    this.weights.metadata.trainDate = new Date();

    // Calculate MAE and RMSE on training data
    let sumAbsError = 0;
    let sumSqError = 0;
    let sumProduct = 0;
    let sumPrediction = 0;
    let sumTarget = 0;

    for (let i = 0; i < examples.length; i++) {
      const prediction = this.predict(
        examples[i].originalPrompt,
        examples[i].modifiedPrompt,
        examples[i].metadata.mutationType,
        examples[i].metadata.category,
        examples[i].context
      );

      const error = prediction.score - targets[i];
      sumAbsError += Math.abs(error);
      sumSqError += error * error;
      sumProduct += prediction.score * targets[i];
      sumPrediction += prediction.score;
      sumTarget += targets[i];
    }

    this.weights.metadata.mae = sumAbsError / examples.length;
    this.weights.metadata.rmse = Math.sqrt(sumSqError / examples.length);

    // Calculate correlation
    const n = examples.length;
    const numerator = n * sumProduct - sumPrediction * sumTarget;
    const denomPred = Math.sqrt(n * sumPrediction * sumPrediction - sumPrediction * sumPrediction);
    const denomTarget = Math.sqrt(n * sumTarget * sumTarget - sumTarget * sumTarget);
    this.weights.metadata.correlation = numerator / (denomPred * denomTarget || 1);
  }

  /**
   * Evaluate model on test set
   */
  evaluate(testExamples: TrainingExample[]): {
    mae: number;
    rmse: number;
    correlation: number;
    predictions: { actual: number; predicted: number; example: TrainingExample }[];
  } {
    if (testExamples.length === 0) {
      throw new Error('Cannot evaluate on empty test set');
    }

    const predictions: { actual: number; predicted: number; example: TrainingExample }[] = [];
    let sumAbsError = 0;
    let sumSqError = 0;
    let sumProduct = 0;
    let sumPrediction = 0;
    let sumTarget = 0;

    for (const example of testExamples) {
      const prediction = this.predict(
        example.originalPrompt,
        example.modifiedPrompt,
        example.metadata.mutationType,
        example.metadata.category,
        example.context
      );

      const actual = (example.humanScore - 1) / 4; // Normalize to 0-1
      const error = prediction.score - actual;

      predictions.push({
        actual,
        predicted: prediction.score,
        example,
      });

      sumAbsError += Math.abs(error);
      sumSqError += error * error;
      sumProduct += prediction.score * actual;
      sumPrediction += prediction.score;
      sumTarget += actual;
    }

    const n = testExamples.length;
    const mae = sumAbsError / n;
    const rmse = Math.sqrt(sumSqError / n);

    const numerator = n * sumProduct - sumPrediction * sumTarget;
    const denomPred = Math.sqrt(n * sumPrediction * sumPrediction - sumPrediction * sumPrediction);
    const denomTarget = Math.sqrt(n * sumTarget * sumTarget - sumTarget * sumTarget);
    const correlation = numerator / (denomPred * denomTarget || 1);

    return {
      mae,
      rmse,
      correlation,
      predictions,
    };
  }

  /**
   * Export weights for serialization
   */
  exportWeights(): RewardModelWeights {
    return { ...this.weights };
  }

  /**
   * Import weights
   */
  importWeights(weights: RewardModelWeights): void {
    this.weights = weights;
  }

  /**
   * Get model info
   */
  getInfo(): {
    version: string;
    trainedOn: number;
    trainDate: Date;
    mae: number;
    rmse: number;
    correlation: number;
  } {
    return {
      version: this.weights.version,
      trainedOn: this.weights.metadata.trainedOn,
      trainDate: this.weights.metadata.trainDate,
      mae: this.weights.metadata.mae,
      rmse: this.weights.metadata.rmse,
      correlation: this.weights.metadata.correlation,
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default RewardModel;
