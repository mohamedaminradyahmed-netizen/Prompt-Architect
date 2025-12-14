# Reward Model - DIRECTIVE-034

## Overview

A lightweight machine learning model that predicts the quality of prompt variations based on human feedback. Enables automated quality scoring without constant human input, reducing review workload by 50-80%.

**Key Insight:** Learn from human feedback to automatically assess prompt quality, making optimization faster and more scalable!

## 🎯 Features

✅ **Feature Extraction**: Extracts 15+ features from prompt pairs (length, lexical, structural, quality)
✅ **Quality Prediction**: Predicts scores from 0-1 with confidence estimates
✅ **Human-Readable Explanations**: Provides interpretable reasons for predictions
✅ **Training on Feedback**: Learns from human ratings (1-5 stars)
✅ **Model Evaluation**: Calculates MAE, RMSE, and correlation metrics
✅ **Lightweight**: No GPU required, runs in TypeScript
✅ **Incremental Learning**: Can be updated with new feedback over time

## 🚀 Quick Start

### Basic Prediction

```typescript
import { RewardModel } from './models/rewardModel';
import { PromptCategory } from '../types/promptTypes';

// Create model (uses default weights)
const model = new RewardModel();

// Predict quality of a variation
const prediction = model.predict(
  'Write code',                                    // original
  'Write a TypeScript function for authentication', // modified
  'expansion',                                     // mutation type
  PromptCategory.CODE_GENERATION                   // category
);

console.log('Score:', prediction.score);           // 0.78
console.log('Confidence:', prediction.confidence); // 0.85
console.log('Explanation:', prediction.explanation);
// "Score: 78.0%. Strengths: clear and well-structured, highly specific, clear action verb."
```

### Training on Human Feedback

```typescript
import { TrainingExample } from './models/rewardModel';

// Collect training examples
const trainingData: TrainingExample[] = [
  {
    id: 'example_001',
    originalPrompt: 'Write code',
    modifiedPrompt: 'Write a TypeScript function to validate emails',
    outputs: { original: '...', modified: '...' },
    humanScore: 5,  // 1-5 stars from human reviewer
    feedback: 'Excellent - very specific',
    metadata: {
      category: PromptCategory.CODE_GENERATION,
      mutationType: 'expansion',
      timestamp: new Date(),
    },
  },
  // ... more examples
];

// Train model
model.train(trainingData);

// Get model info
const info = model.getInfo();
console.log('Trained on:', info.trainedOn, 'examples');
console.log('MAE:', info.mae);
console.log('Correlation:', info.correlation);
```

### Evaluating Model

```typescript
// Evaluate on test set
const testResults = model.evaluate(testExamples);

console.log('Test MAE:', testResults.mae);
console.log('Test RMSE:', testResults.rmse);
console.log('Correlation:', testResults.correlation);

// View individual predictions
testResults.predictions.forEach(pred => {
  console.log(`Actual: ${pred.actual}, Predicted: ${pred.predicted}`);
});
```

## 📊 Features Explained

### Feature Categories

The model extracts features in 4 main categories:

#### 1. Length Features
- `originalLength`: Character count of original prompt
- `modifiedLength`: Character count of modified prompt
- `lengthRatio`: Modified / Original length ratio
- `lengthDiff`: Absolute difference in length

**Use:** Detects significant expansions or reductions

#### 2. Lexical Features
- `vocabularyRichness`: Unique words / Total words
- `avgWordLength`: Average character count per word
- `sentenceCount`: Number of sentences

**Use:** Measures linguistic complexity and diversity

#### 3. Structural Features
- `hasImperativeVerb`: Starts with action verb (write, create, etc.)
- `hasConstraints`: Contains requirements (must, should, etc.)
- `hasExamples`: Includes examples or references
- `hasContext`: Has contextual information

**Use:** Detects important prompt components

#### 4. Quality Indicators
- `clarityScore`: How clear and well-structured (0-1)
- `specificityScore`: How specific and detailed (0-1)
- `completenessScore`: How comprehensive (0-1)

**Use:** Heuristic quality assessments

## 🧠 Model Architecture

### Current: Linear Regression

```
score = intercept + Σ(weight_i × feature_i)
```

**Advantages:**
- ✅ Fast and lightweight
- ✅ Interpretable feature weights
- ✅ No GPU required
- ✅ Easy to debug

**Limitations:**
- ⚠️ Cannot capture complex non-linear relationships
- ⚠️ Limited by feature quality

### Future: Advanced Models

For production at scale, consider:

1. **XGBoost / Random Forest**
   - Better non-linear modeling
   - Feature importance built-in
   - Still interpretable

2. **Neural Network (BERT-based)**
   - Deep semantic understanding
   - Transfer learning from pretrained models
   - Requires GPU for training

3. **Ensemble**
   - Combine multiple models
   - Best of all approaches

## 📈 Training Process

### 1. Data Collection

Collect training examples from:

```typescript
// From human feedback
interface TrainingExample {
  originalPrompt: string;
  modifiedPrompt: string;
  humanScore: number;  // 1-5 stars
  feedback?: string;
  metadata: {
    category: PromptCategory;
    mutationType: string;
    timestamp: Date;
  };
}
```

**Recommended:** 50-100 examples minimum, 500+ ideal

### 2. Feature Extraction

```typescript
const features = extractFeatures(
  original,
  modified,
  mutationType,
  category,
  context
);
// Returns PromptFeatures with 15+ numeric/boolean features
```

### 3. Training

```typescript
model.train(trainingExamples);
```

**What happens:**
- Normalizes human scores to 0-1 range
- Extracts features for all examples
- Fits linear regression weights
- Calculates training metrics (MAE, RMSE, correlation)

### 4. Evaluation

```typescript
const results = model.evaluate(testExamples);
```

**Metrics:**
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **Correlation**: How well predictions track actual scores

**Good targets:**
- MAE < 0.15 (on 0-1 scale)
- Correlation > 0.7

## 🎯 Usage Patterns

### Pattern 1: Filter Low-Quality Variations

```typescript
// Score all variations
const variations = [/* ... */];
const scored = variations.map(v => ({
  variation: v,
  score: model.predict(original, v.text, v.mutation, category).score,
}));

// Keep only high-quality (score > 0.6)
const filtered = scored.filter(s => s.score > 0.6);

console.log(`Filtered ${variations.length} → ${filtered.length} variations`);
// "Filtered 20 → 7 variations" (65% reduction in review workload)
```

### Pattern 2: Rank Variations

```typescript
// Score and sort
const ranked = variations
  .map(v => ({
    variation: v,
    prediction: model.predict(original, v.text, v.mutation, category),
  }))
  .sort((a, b) => b.prediction.score - a.prediction.score);

// Show top 5
ranked.slice(0, 5).forEach((r, i) => {
  console.log(`${i + 1}. ${r.prediction.score.toFixed(2)} - "${r.variation.text}"`);
  console.log(`   ${r.prediction.explanation}`);
});
```

### Pattern 3: Guide Optimization

```typescript
// Use in optimizer to select next variation
async function optimizeWithReward(
  original: string,
  model: RewardModel,
  iterations: number
): Promise<string> {
  let best = original;
  let bestScore = 0;

  for (let i = 0; i < iterations; i++) {
    // Generate candidates
    const candidates = generateCandidates(best);

    // Score with reward model
    const scored = candidates.map(c => ({
      text: c,
      score: model.predict(original, c, 'expansion', category).score,
    }));

    // Select best
    scored.sort((a, b) => b.score - a.score);
    if (scored[0].score > bestScore) {
      best = scored[0].text;
      bestScore = scored[0].score;
    }
  }

  return best;
}
```

### Pattern 4: Reduce Human Review Load

```typescript
// Only send uncertain predictions to humans
const predictions = variations.map(v => ({
  variation: v,
  prediction: model.predict(original, v.text, v.mutation, category),
}));

// High confidence: Auto-accept/reject
const confident = predictions.filter(p => p.prediction.confidence > 0.8);
const autoAccept = confident.filter(p => p.prediction.score > 0.7);
const autoReject = confident.filter(p => p.prediction.score < 0.4);

// Low confidence: Send to human
const needsReview = predictions.filter(p => p.prediction.confidence <= 0.8);

console.log(`Auto-accepted: ${autoAccept.length}`);
console.log(`Auto-rejected: ${autoReject.length}`);
console.log(`Needs human review: ${needsReview.length}`);
// Typical: 50-80% reduction in human review
```

## 🔬 Advanced Topics

### Custom Feature Engineering

Add domain-specific features:

```typescript
// Extend extractFeatures
function extractCustomFeatures(
  original: string,
  modified: string
): Record<string, number> {
  return {
    // Code-specific features
    hasTypeAnnotations: /:\s*(string|number|boolean|any)/.test(modified) ? 1 : 0,
    hasAsyncKeyword: /\basync\b/.test(modified) ? 1 : 0,

    // Content-specific features
    hasToneGuidance: /\b(professional|casual|friendly)\b/i.test(modified) ? 1 : 0,
    hasWordCount: /\d+\s*words/.test(modified) ? 1 : 0,

    // General
    questionCount: (modified.match(/\?/g) || []).length,
  };
}
```

### Transfer Learning

Bootstrap from general model:

```typescript
// Start with pre-trained weights
const baseWeights = loadPretrainedWeights();
const model = new RewardModel(baseWeights);

// Fine-tune on domain-specific data
model.train(domainSpecificExamples);
```

### Calibration

Ensure predicted scores match actual quality:

```typescript
// After training, check calibration
function checkCalibration(model: RewardModel, examples: TrainingExample[]) {
  const buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const counts = new Array(buckets.length - 1).fill(0);
  const sums = new Array(buckets.length - 1).fill(0);

  examples.forEach(ex => {
    const pred = model.predict(ex.originalPrompt, ex.modifiedPrompt, ex.metadata.mutationType, ex.metadata.category);
    const actual = (ex.humanScore - 1) / 4;

    const bucket = buckets.findIndex(b => pred.score <= b) - 1;
    counts[bucket]++;
    sums[bucket] += actual;
  });

  // Compare predicted vs actual in each bucket
  buckets.slice(0, -1).forEach((b, i) => {
    const avgActual = counts[i] > 0 ? sums[i] / counts[i] : 0;
    console.log(`Predicted ${b}-${buckets[i + 1]}: Avg Actual = ${avgActual.toFixed(2)}`);
  });
}
```

### Continuous Learning

Update model with new feedback:

```typescript
// Periodically retrain with accumulated feedback
setInterval(async () => {
  // Fetch new feedback since last training
  const newExamples = await fetchNewFeedback(lastTrainingDate);

  if (newExamples.length >= 10) {
    // Combine with existing training data
    const allData = [...existingTrainingData, ...newExamples];

    // Retrain
    model.train(allData);

    // Save new weights
    saveWeights(model.exportWeights());

    lastTrainingDate = new Date();
  }
}, 24 * 60 * 60 * 1000); // Daily
```

## 📊 Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Feature Extraction | O(N) | < 1ms per prompt |
| Prediction | O(F) | < 1ms |
| Training | O(E × F) | < 100ms for 100 examples |
| Evaluation | O(E × F) | < 100ms for 100 examples |

Where:
- N = prompt length
- F = number of features (~15)
- E = number of examples

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Model Weights | O(F) | ~1 KB |
| Training Data | O(E × N) | Depends on examples |
| Features | O(F) | ~500 bytes per prediction |

### Accuracy Expectations

| Training Examples | Expected MAE | Expected Correlation |
|------------------|--------------|---------------------|
| 10-20 | 0.20-0.25 | 0.4-0.6 |
| 50-100 | 0.15-0.20 | 0.6-0.75 |
| 200+ | 0.10-0.15 | 0.75-0.85 |
| 500+ | 0.08-0.12 | 0.80-0.90 |

*With better features and models (e.g., XGBoost), can achieve MAE < 0.10 with 100 examples*

## 🎓 Examples

### Example 1: Code Generation Scoring

```typescript
const model = new RewardModel();
// (trained on code generation examples)

const original = 'Write function';
const variations = [
  'Write a function',
  'Write a TypeScript function to sort an array',
  'Implement an async TypeScript function to sort a number array using quicksort with O(n log n) complexity',
];

variations.forEach(v => {
  const pred = model.predict(original, v, 'expansion', PromptCategory.CODE_GENERATION);
  console.log(`${pred.score.toFixed(2)} - "${v}"`);
});

// Output:
// 0.52 - "Write a function"
// 0.73 - "Write a TypeScript function to sort an array"
// 0.91 - "Implement an async TypeScript function to sort a number array using quicksort with O(n log n) complexity"
```

### Example 2: Content Writing Scoring

```typescript
const original = 'Write blog post';
const modified = 'Write a 500-word blog post about AI ethics. Include: introduction, 3 main concerns with examples, and conclusion with recommendations. Use professional but accessible tone.';

const pred = model.predict(original, modified, 'expansion', PromptCategory.CONTENT_WRITING);

console.log('Score:', pred.score);           // 0.88
console.log('Confidence:', pred.confidence); // 0.90
console.log('Explanation:', pred.explanation);
// "Score: 88.0%. Strengths: clear and well-structured, highly specific, comprehensive, clear action verb, well-defined constraints."
```

### Example 3: Filtering Pipeline

```typescript
async function filterVariations(
  original: string,
  variations: string[],
  model: RewardModel,
  threshold: number = 0.6
): Promise<string[]> {
  const scored = variations.map(v => ({
    text: v,
    score: model.predict(original, v, 'expansion', PromptCategory.CODE_GENERATION).score,
  }));

  return scored
    .filter(s => s.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .map(s => s.text);
}

// Usage
const filtered = await filterVariations(
  'Create API',
  generateVariations('Create API', 20),
  trainedModel,
  0.7
);

console.log(`Reduced 20 variations to ${filtered.length} high-quality ones`);
```

## 🐛 Troubleshooting

### Problem: Model predicts similar scores for everything

**Cause:** Insufficient training data or low feature variance

**Solution:**
- Collect more diverse training examples
- Ensure examples span the full quality range (1-5 stars)
- Add more discriminative features

### Problem: Low correlation with human scores

**Cause:** Features don't capture what humans care about

**Solution:**
- Analyze feature contributions on mispredictions
- Add domain-specific features
- Consider non-linear model (XGBoost)

### Problem: Overconfident predictions

**Cause:** Confidence calculation doesn't account for uncertainty

**Solution:**
- Lower confidence for edge cases
- Ensemble multiple models and use variance as confidence
- Calibrate confidence on validation set

### Problem: Model degrades over time

**Cause:** Distribution shift (prompt patterns change)

**Solution:**
- Implement continuous learning
- Retrain periodically with new feedback
- Monitor drift using A/B tests

## 🚀 Running the Demo

```bash
npx tsx src/models/rewardModel.demo.ts
```

Demonstrates:
1. ✅ Feature extraction from prompt pairs
2. ✅ Prediction with untrained model
3. ✅ Model training on sample data
4. ✅ Evaluation on test set
5. ✅ Comparing multiple variations
6. ✅ Feature contribution analysis

## 🔗 Integration

### With Genetic Optimizer

```typescript
import { geneticOptimize } from '../optimizer/genetic';

// Use reward model as fitness function
const fitnessFunction = (prompt: string): number => {
  const prediction = rewardModel.predict(
    originalPrompt,
    prompt,
    'genetic-evolution',
    category
  );
  return prediction.score * 100; // Scale to 0-100
};

const result = await geneticOptimize(originalPrompt, fitnessFunction);
```

### With Human-in-the-Loop

```typescript
import { selectSamplesForReview } from '../humanLoop/sampleSelection';

// Score variations
const scored = variations.map(v => ({
  ...v,
  prediction: rewardModel.predict(original, v.text, v.mutation, category),
}));

// Auto-decide on confident predictions
const confident = scored.filter(s => s.prediction.confidence > 0.85);

// Send uncertain to humans
const needsReview = scored.filter(s => s.prediction.confidence <= 0.85);
const samplesForReview = selectSamplesForReview(needsReview, 'UNCERTAINTY', 5);
```

## 📚 Related Directives

- **DIRECTIVE-015**: Human Feedback Score (provides training data)
- **DIRECTIVE-028**: Lineage Tracking (tracks variation history)
- **DIRECTIVE-029**: Sample Selection (selects examples for review)
- **DIRECTIVE-030**: Human Review UI (collects feedback)
- **DIRECTIVE-035**: RLAIF (uses reward model for AI feedback)

## Summary

**Reward Model provides:**
- ✅ Automated quality prediction
- ✅ Reduced human review workload (50-80%)
- ✅ Fast, lightweight implementation
- ✅ Interpretable explanations
- ✅ Continuous learning capability

**Use it to:**
- 🎯 Filter low-quality variations automatically
- 📊 Rank variations by predicted quality
- 🚀 Guide optimization algorithms
- 💰 Reduce costs of human review
- ⚡ Speed up optimization loops

**Status:** ✅ Fully Implemented (DIRECTIVE-034)

---

# Surrogate Orchestrator - DIRECTIVE-037

## Overview

A smart model orchestration system that uses lightweight/fast models for initial evaluation and reserves premium models only for final decisions. This approach provides **60-80% cost savings** while maintaining high-quality outputs.

**Key Insight:** Most evaluations don't need GPT-4! Use cheap models for exploration, upgrade only when necessary.

## 🎯 Features

✅ **Multi-Model Support**: Groq (Llama), OpenAI (GPT-3.5/4), Anthropic (Claude) + Local
✅ **Three Evaluation Modes**: exploration (cheap), exploitation (mid), final (premium)
✅ **Progressive Evaluation**: Start cheap, upgrade only if quality threshold isn't met
✅ **Intelligent Caching**: LRU cache with TTL to avoid repeated API calls
✅ **Cost Tracking**: Real-time statistics on costs, savings, and usage
✅ **Batch Processing**: Evaluate multiple prompts efficiently
✅ **Factory Presets**: Cost-optimized, quality-focused, and balanced configurations

## 🚀 Quick Start

### Basic Evaluation

```typescript
import { SurrogateOrchestrator } from './models/surrogateOrchestrator';

// Create orchestrator
const orchestrator = new SurrogateOrchestrator();

// Evaluate with exploration mode (cheapest)
const result = await orchestrator.evaluate(
  { prompt: 'Write a TypeScript function for sorting' },
  'exploration'
);

console.log('Model:', result.model.model);        // "llama-3.1-8b-instant"
console.log('Cost:', result.cost);                // $0.000004
console.log('Score:', result.score);              // 0.72
console.log('Latency:', result.latency);          // 245ms
```

### Progressive Evaluation (Smart Cost Optimization)

```typescript
// Starts cheap, upgrades only if needed
const result = await orchestrator.progressiveEvaluate(
  { prompt: 'Explain quantum computing' },
  0.85  // quality threshold
);

// Uses cheapest model that meets threshold
console.log('Mode used:', result.metadata.mode);  // "exploration" or "exploitation" or "final"
console.log('Cost:', result.cost);
```

### Batch Evaluation

```typescript
const prompts = [
  { prompt: 'Write code for sorting', category: PromptCategory.CODE_GENERATION },
  { prompt: 'Marketing copy for app', category: PromptCategory.MARKETING_COPY },
  { prompt: 'Explain machine learning', category: PromptCategory.GENERAL_QA },
];

const batchResult = await orchestrator.evaluateBatch(prompts, 'exploration');

console.log('Total Cost:', batchResult.totalCost);     // $0.00015
console.log('Cost Savings:', batchResult.costSavings); // $0.42
console.log('Success Rate:', batchResult.successRate); // 1.0
```

## 📊 Evaluation Modes

| Mode | Tier | Model Example | Cost/1K Tokens | Use Case |
|------|------|---------------|----------------|----------|
| `exploration` | cheap | Llama 3.1 8B | $0.0001 | Initial screening, diverse search |
| `exploitation` | mid | GPT-3.5 Turbo | $0.002 | Refinement, balancing cost/quality |
| `final` | premium | GPT-4 Turbo | $0.02 | Final evaluation, production output |

### When to Use Each Mode

```typescript
// EXPLORATION: Generating many candidates, testing ideas
for (const candidate of candidates) {
  const score = await orchestrator.evaluate(candidate, 'exploration');
  if (score.score > 0.5) potentialWinners.push(candidate);
}

// EXPLOITATION: Refining top candidates
for (const winner of potentialWinners) {
  const refined = await orchestrator.evaluate(winner, 'exploitation');
  if (refined.score > 0.75) finalists.push(winner);
}

// FINAL: Producing actual output
const best = await orchestrator.evaluate(finalists[0], 'final');
```

## 🏭 Factory Presets

### Cost-Optimized (Maximum Savings)

```typescript
import { createCostOptimizedOrchestrator } from './models/surrogateOrchestrator';

const orchestrator = createCostOptimizedOrchestrator();
// exploration: groq-llama-8b
// exploitation: groq-llama-70b  
// final: openai-gpt35

// Expected savings: 80-90% vs always using GPT-4
```

### Quality-Focused (Maximum Quality)

```typescript
import { createQualityFocusedOrchestrator } from './models/surrogateOrchestrator';

const orchestrator = createQualityFocusedOrchestrator();
// exploration: anthropic-haiku
// exploitation: anthropic-sonnet
// final: anthropic-opus

// Expected savings: 40-60% vs always using Opus
```

### Balanced (Default)

```typescript
import { createBalancedOrchestrator } from './models/surrogateOrchestrator';

const orchestrator = createBalancedOrchestrator();
// exploration: groq-llama-8b
// exploitation: openai-gpt35
// final: openai-gpt4-turbo

// Expected savings: 60-80%
```

## 💾 Caching

The orchestrator includes intelligent LRU caching:

```typescript
// First request - calls API
const result1 = await orchestrator.evaluate(prompt, 'exploration');
console.log(result1.metadata.cached);  // false

// Second request - uses cache
const result2 = await orchestrator.evaluate(prompt, 'exploration');
console.log(result2.metadata.cached);  // true
console.log(result2.latency);          // ~1ms (vs 200ms)

// Check cache stats
const stats = orchestrator.getStats();
console.log('Cache hit rate:', stats.cacheHitRate);  // 0.5 (50%)

// Clear cache if needed
orchestrator.clearCache();
```

## 📈 Statistics & Analytics

### Real-time Statistics

```typescript
const stats = orchestrator.getStats();

console.log('Total Requests:', stats.totalRequests);
console.log('Cache Hits:', stats.cacheHits);
console.log('Cache Hit Rate:', stats.cacheHitRate);
console.log('Total Cost:', stats.totalCost);
console.log('Total Savings:', stats.totalSavings);
console.log('Avg Latency:', stats.avgLatency);
console.log('By Mode:', stats.requestsByMode);
// { exploration: 35, exploitation: 10, final: 5 }
console.log('By Provider:', stats.requestsByProvider);
// { groq: 35, openai: 15, anthropic: 0, local: 0 }
```

### Cost Savings Summary

```typescript
const savings = orchestrator.getCostSavingsSummary();

console.log('Actual Cost:', savings.totalCost);           // $0.15
console.log('If Premium Only:', savings.estimatedPremiumCost);  // $0.75
console.log('Savings:', savings.savings);                 // $0.60
console.log('Savings %:', savings.savingsPercentage);     // 80%
```

### Usage Breakdown

```typescript
const usage = orchestrator.getModelUsageBreakdown();

console.log('Most Used Provider:', usage.mostUsedProvider);  // 'groq'
console.log('Most Used Mode:', usage.mostUsedMode);          // 'exploration'
```

## ⚙️ Configuration

### Custom Model Selection

```typescript
const orchestrator = new SurrogateOrchestrator({
  modeModelMap: {
    exploration: 'anthropic-haiku',    // Use Claude Haiku for exploration
    exploitation: 'openai-gpt35',      // Use GPT-3.5 for exploitation
    final: 'anthropic-opus',           // Use Opus for final
  },
});
```

### Runtime Mode Changes

```typescript
// Change model for a specific mode
orchestrator.setModeModel('final', 'anthropic-sonnet');
```

### Add Custom Model

```typescript
// Add a new model to the registry
orchestrator.addCustomModel('my-local-llama', {
  provider: 'local',
  model: 'llama-3.1-8b-local',
  tier: 'cheap',
  costPer1kTokens: 0,
  avgLatencyMs: 500,
  qualityScore: 0.70,
  maxTokens: 4096,
});

// Now use it
orchestrator.setModeModel('exploration', 'my-local-llama');
```

### Cache Configuration

```typescript
const orchestrator = new SurrogateOrchestrator({
  cacheTTL: 12 * 60 * 60 * 1000,  // 12 hours
  maxCacheSize: 500,              // Max 500 entries
});
```

## 📚 Available Models

### Cheap Tier (Exploration)

| Key | Model | Cost | Latency | Quality |
|-----|-------|------|---------|---------|
| `groq-llama-8b` | Llama 3.1 8B Instant | $0.0001 | 200ms | 70% |
| `anthropic-haiku` | Claude 3 Haiku | $0.00025 | 300ms | 75% |
| `local-llama` | Local Llama | $0 | 1000ms | 65% |

### Mid Tier (Exploitation)

| Key | Model | Cost | Latency | Quality |
|-----|-------|------|---------|---------|
| `groq-llama-70b` | Llama 3.1 70B | $0.0008 | 500ms | 85% |
| `openai-gpt35` | GPT-3.5 Turbo | $0.002 | 800ms | 82% |
| `anthropic-sonnet` | Claude 3.5 Sonnet | $0.003 | 1000ms | 90% |

### Premium Tier (Final)

| Key | Model | Cost | Latency | Quality |
|-----|-------|------|---------|---------|
| `openai-gpt4` | GPT-4 | $0.03 | 2000ms | 95% |
| `openai-gpt4-turbo` | GPT-4 Turbo | $0.02 | 1500ms | 94% |
| `anthropic-opus` | Claude 3 Opus | $0.015 | 2500ms | 96% |

## 🎯 Usage Patterns

### Pattern 1: Genetic Optimization with Surrogate

```typescript
import { geneticOptimize } from '../optimizer/genetic';

// Use cheap model for population evaluation
const fitnessFunction = async (prompt: string): Promise<number> => {
  const result = await orchestrator.evaluate({ prompt }, 'exploration');
  return result.score * 100;
};

// Run genetic optimization
const result = await geneticOptimize(originalPrompt, fitnessFunction);

// Final evaluation with premium model
const finalScore = await orchestrator.evaluate(
  { prompt: result.bestPrompt },
  'final'
);
```

### Pattern 2: Multi-Stage Pipeline

```typescript
async function multiStagePipeline(prompt: string): Promise<EvaluationResult> {
  // Stage 1: Quick screening (cheap)
  const screening = await orchestrator.evaluate({ prompt }, 'exploration');
  if (screening.score < 0.3) {
    return screening;  // Reject early
  }

  // Stage 2: Refinement check (mid)
  const refinement = await orchestrator.evaluate({ prompt }, 'exploitation');
  if (refinement.score < 0.6) {
    return refinement;  // Not good enough
  }

  // Stage 3: Final verification (premium)
  return orchestrator.evaluate({ prompt }, 'final');
}
```

### Pattern 3: A/B Testing with Cost Control

```typescript
async function abTest(variantA: string, variantB: string) {
  // Quick comparison with cheap models
  const [scoreA, scoreB] = await Promise.all([
    orchestrator.evaluate({ prompt: variantA }, 'exploration'),
    orchestrator.evaluate({ prompt: variantB }, 'exploration'),
  ]);

  // Only deep-test the winner
  const winner = scoreA.score > scoreB.score ? variantA : variantB;
  return orchestrator.evaluate({ prompt: winner }, 'final');
}
```

### Pattern 4: Budget-Constrained Optimization

```typescript
async function optimizeWithBudget(
  prompts: string[],
  budgetUSD: number
): Promise<EvaluationResult[]> {
  const results: EvaluationResult[] = [];
  let spent = 0;

  for (const prompt of prompts) {
    // Check budget
    if (spent >= budgetUSD) break;

    // Use appropriate mode based on remaining budget
    const mode = spent < budgetUSD * 0.7 ? 'exploration' :
                 spent < budgetUSD * 0.9 ? 'exploitation' : 'final';

    const result = await orchestrator.evaluate({ prompt }, mode);
    results.push(result);
    spent += result.cost;
  }

  return results;
}
```

## 🐛 Troubleshooting

### Problem: Cache not being used

**Cause:** Request parameters differ slightly (whitespace, context)

**Solution:**
```typescript
// Normalize prompts before evaluation
const normalizedPrompt = prompt.trim().replace(/\s+/g, ' ');
```

### Problem: Model quality too low

**Cause:** Using cheap model for complex tasks

**Solution:**
- Increase quality threshold in progressive evaluation
- Use exploitation or final mode for important evaluations
- Switch to quality-focused preset

### Problem: Costs higher than expected

**Cause:** Too many final evaluations or large outputs

**Solution:**
- Use progressive evaluation
- Reduce expectedOutputLength
- Increase cache TTL
- Check stats to identify expensive operations

## 🚀 Running the Demo

```bash
npx tsx src/models/surrogateOrchestrator.demo.ts
```

Demonstrates:
1. ✅ Basic evaluation with all modes
2. ✅ Progressive evaluation
3. ✅ Batch evaluation
4. ✅ Cache effectiveness
5. ✅ Orchestrator presets
6. ✅ Cost savings analysis

## 🔗 Integration

### With Reward Model

```typescript
import { RewardModel } from './models/rewardModel';
import { SurrogateOrchestrator } from './models/surrogateOrchestrator';

// Use surrogate for execution, reward model for scoring
const orchestrator = new SurrogateOrchestrator();
const rewardModel = new RewardModel();

async function evaluateVariation(original: string, variation: string) {
  // Get output from cheap model
  const execution = await orchestrator.evaluate(
    { prompt: variation },
    'exploration'
  );

  // Score with reward model (no API cost)
  const score = rewardModel.predict(
    original, 
    variation,
    'expansion',
    PromptCategory.CODE_GENERATION
  );

  return {
    output: execution.output,
    apiScore: execution.score,
    rewardScore: score.score,
    cost: execution.cost,
  };
}
```

### With Hybrid Optimizer

```typescript
import { hybridOptimize } from '../optimizer/hybrid';

// Use surrogate as evaluation function
async function surrogateEvaluator(prompt: string): Promise<number> {
  const result = await orchestrator.evaluate(
    { prompt },
    'exploration'  // Cheap for optimization loop
  );
  return result.score * 100;
}

const result = await hybridOptimize(
  originalPrompt,
  { evaluator: surrogateEvaluator }
);
```

## 📚 Related Directives

- **DIRECTIVE-034**: Reward Model (local quality scoring)
- **DIRECTIVE-036**: Batching (batch API calls)
- **DIRECTIVE-045**: Groq Integration (fast inference)
- **DIRECTIVE-038**: Overfitting Detection (validation)

## Summary

**Surrogate Orchestrator provides:**
- ✅ 60-80% cost reduction
- ✅ Multi-provider model orchestration
- ✅ Progressive quality escalation
- ✅ Intelligent caching
- ✅ Real-time cost analytics

**Use it to:**
- 💰 Dramatically reduce API costs
- ⚡ Speed up optimization loops
- 🎯 Balance cost vs quality
- 📊 Track spending and savings
- 🔄 Implement multi-stage pipelines

**Status:** ✅ Fully Implemented (DIRECTIVE-037)
