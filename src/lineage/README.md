# Lineage Tracking System - DIRECTIVE-028

## Overview

Complete genealogy tracking system for prompt variations. Tracks parent-child relationships, mutation chains, performance metrics, and evolution paths.

**Key Insight:** Understanding *how* you got to a good prompt is as important as the prompt itself!

## 🎯 Features

✅ **Parent-Child Tracking**: Full genealogy tree of all variations
✅ **Mutation History**: Complete record of transformations applied
✅ **Performance Metrics**: Score, cost, latency at each step
✅ **Path Discovery**: Find optimal paths to target scores
✅ **Human Feedback**: Integrate user ratings and comments
✅ **Tree Visualization**: ASCII tree rendering
✅ **Success Analysis**: Mutation effectiveness statistics
✅ **Generation Tracking**: Evolution over time

## 🌲 Core Concepts

### Variation Lineage

Every variation has:
- **ID**: Unique identifier
- **Parent**: Where it came from
- **Children**: What descended from it
- **Mutation**: How it was created
- **Metrics**: Performance data
- **Generation**: Distance from original

### Genealogy Tree

```
                    Original (Gen 0)
                   /      |      \
            Try-Catch  Reduce  Expand (Gen 1)
               /  \
         Expand  Reduce (Gen 2)
            /
       Try-Catch (Gen 3)
```

## 🚀 Quick Start

### Basic Tracking

```typescript
import {
  LineageTracker,
  createOriginalVariation,
  createChildVariation,
} from './lineage/tracker';

// Create tracker
const tracker = new LineageTracker();

// Track original prompt
const original = createOriginalVariation(
  'Write a sorting function',
  0.5,   // score
  0.01,  // cost
  1000   // latency
);
tracker.trackVariation(original);

// Create child variation
const child = createChildVariation(
  original,
  'Try to write a sorting function...',  // mutated prompt
  'try-catch-style',                      // mutation type
  {},                                     // mutation params
  0.65,                                   // new score
  0.012,                                  // new cost
  1100                                    // new latency
);
tracker.trackVariation(child);

// Get lineage
const lineage = tracker.getLineage(child.id);
console.log(formatPath(lineage));
// Output: "original → try-catch-style(+0.150)"
```

### Visualize Tree

```typescript
const graph = tracker.visualizeLineage(variationId);

console.log(visualizeTree(graph));
// Output:
// └─ original [0.500]
//    ├─ try-catch-style [0.650] (+0.150)
//    │  └─ expansion [0.780] (+0.130)
//    └─ context-reduction [0.550] (+0.050)
```

### Find Best Path

```typescript
// Find shortest path to score >= 0.80
const path = tracker.findBestPath('Write a sorting function', 0.80);

if (path) {
  console.log(`Found in ${path.length - 1} steps:`);
  path.forEach((v) => {
    console.log(`  ${v.mutation} → ${v.metrics.score}`);
  });
}
```

## 📊 Data Structures

### VariationLineage

```typescript
interface VariationLineage {
  id: string;                    // Unique ID
  parentId: string | null;       // Parent ID (null for original)
  originalPrompt: string;        // Root prompt
  currentPrompt: string;         // Current text
  mutation: MutationType;        // How created
  mutationParams: Record;        // Mutation parameters
  timestamp: Date;               // When created
  metrics: VariationMetrics;     // Performance
  feedback?: HumanFeedback;      // User feedback
  children: string[];            // Child IDs
  generation: number;            // Distance from root
  path: MutationStep[];          // Full path from root
}
```

### VariationMetrics

```typescript
interface VariationMetrics {
  score: number;                 // Quality score (0-1)
  cost: number;                  // Cost in USD
  latency: number;               // Latency in ms
  tokens?: number;               // Token count
  custom?: Record;               // Custom metrics
}
```

### HumanFeedback

```typescript
interface HumanFeedback {
  userId: string;                // User who provided feedback
  rating: number;                // 1-5 stars
  comment?: string;              // Text feedback
  timestamp: Date;               // When provided
  tags?: string[];               // Tags
}
```

## 🎯 API Reference

### LineageTracker Class

#### `trackVariation(variation: VariationLineage): void`
Track a new variation in the system.

```typescript
tracker.trackVariation(variation);
```

#### `getLineage(variationId: string): VariationLineage[]`
Get complete lineage from original to specified variation.

```typescript
const lineage = tracker.getLineage('var_123');
// Returns: [original, child1, child2, ...]
```

#### `getDescendants(variationId: string): VariationLineage[]`
Get all descendants of a variation.

```typescript
const descendants = tracker.getDescendants('var_123');
```

#### `visualizeLineage(variationId: string): LineageGraph`
Build tree structure for visualization.

```typescript
const graph = tracker.visualizeLineage('var_123');
console.log(visualizeTree(graph));
```

#### `findBestPath(originalPrompt: string, targetScore: number): VariationLineage[] | null`
Find shortest path to target score.

```typescript
const path = tracker.findBestPath('Write function', 0.80);
```

#### `getByGeneration(generation: number): VariationLineage[]`
Get all variations at specific generation.

```typescript
const gen2 = tracker.getByGeneration(2);
```

#### `getAllVariations(originalPrompt: string): VariationLineage[]`
Get all variations from an original prompt.

```typescript
const all = tracker.getAllVariations('Write function');
```

#### `addFeedback(variationId: string, feedback: HumanFeedback): void`
Add human feedback to a variation.

```typescript
tracker.addFeedback('var_123', {
  userId: 'user_456',
  rating: 4,
  comment: 'Good improvement!',
  timestamp: new Date(),
});
```

#### `getGlobalStats(): LineageStats`
Get statistics across all tracked lineages.

```typescript
const stats = tracker.getGlobalStats();
console.log(stats.mutationSuccessRates);
```

### Utility Functions

#### `createOriginalVariation(prompt, score, cost, latency): VariationLineage`
Create initial variation (generation 0).

#### `createChildVariation(parent, prompt, mutation, params, score, cost, latency): VariationLineage`
Create child variation from parent.

#### `formatPath(lineage: VariationLineage[]): string`
Format lineage as readable string.

```typescript
formatPath(lineage);
// "original → try-catch(+0.15) → expand(+0.13)"
```

#### `visualizeTree(graph: LineageGraph, maxDepth?: number): string`
Render tree as ASCII art.

## 💡 Usage Patterns

### Pattern 1: Simple Linear Evolution

```typescript
const tracker = new LineageTracker();

let current = createOriginalVariation('Write code', 0.5, 0.01, 1000);
tracker.trackVariation(current);

// Apply mutations sequentially
for (const mutation of ['try-catch-style', 'expansion']) {
  const mutated = applyMutation(current.currentPrompt, mutation);
  const child = createChildVariation(
    current,
    mutated,
    mutation,
    {},
    current.metrics.score + 0.1,
    0.01,
    1000
  );
  tracker.trackVariation(child);
  current = child;
}

console.log(formatPath(tracker.getLineage(current.id)));
```

### Pattern 2: Branching Exploration

```typescript
const original = createOriginalVariation('Optimize code', 0.6, 0.02, 1500);
tracker.trackVariation(original);

// Try multiple mutations from same parent
const mutations = ['try-catch-style', 'expansion', 'context-reduction'];

for (const mutation of mutations) {
  const child = createChildVariation(
    original,
    applyMutation(original.currentPrompt, mutation),
    mutation,
    {},
    0.7 + Math.random() * 0.1,
    0.02,
    1500
  );
  tracker.trackVariation(child);
}

// Visualize tree
console.log(visualizeTree(tracker.visualizeLineage(original.id)));
```

### Pattern 3: Integration with Optimizers

```typescript
// With Genetic Algorithm
import { geneticOptimize } from './optimizer/genetic';

const original = createOriginalVariation(prompt, 0.5, 0.01, 1000);
tracker.trackVariation(original);

const result = await geneticOptimize(prompt, fitness);

// Track each individual in final population
result.finalPopulation.forEach((individual) => {
  const variation = createChildVariation(
    original,
    individual.prompt,
    'genetic-evolution',
    { mutations: individual.mutations },
    individual.fitness / 100,
    0.01,
    1000
  );
  tracker.trackVariation(variation);
});
```

### Pattern 4: Human-in-the-Loop

```typescript
// After optimization
const variations = tracker.getAllVariations(originalPrompt);

// Present top 3 to user
const top3 = variations
  .sort((a, b) => b.metrics.score - a.metrics.score)
  .slice(0, 3);

// Collect feedback
for (const variation of top3) {
  const rating = await getUserRating(variation);
  tracker.addFeedback(variation.id, {
    userId: currentUser,
    rating,
    timestamp: new Date(),
  });
}

// Analyze feedback
const stats = tracker.getGlobalStats();
```

## 📈 Analysis Examples

### Success Rate by Mutation

```typescript
const stats = tracker.getGlobalStats();

for (const [mutation, rate] of stats.mutationSuccessRates) {
  console.log(`${mutation}: ${(rate * 100).toFixed(1)}% success`);
}

// Output:
// try-catch-style: 75.0% success
// expansion: 60.0% success
// context-reduction: 45.0% success
```

### Score Evolution Over Generations

```typescript
const stats = tracker.getGlobalStats();

for (const [gen, avgScore] of stats.avgScoreByGeneration) {
  console.log(`Gen ${gen}: ${(avgScore * 100).toFixed(1)}%`);
}

// Output:
// Gen 0: 50.0%
// Gen 1: 62.5%
// Gen 2: 71.3%
// Gen 3: 78.9%
```

### Find Optimal Strategy

```typescript
// Compare different starting strategies
const strategies = [
  ['try-catch-style', 'expansion'],
  ['expansion', 'try-catch-style'],
  ['context-reduction', 'expansion', 'try-catch-style'],
];

for (const strategy of strategies) {
  let current = original;
  for (const mutation of strategy) {
    current = applyMutationAndTrack(current, mutation, tracker);
  }
  console.log(`${strategy.join(' → ')}: ${current.metrics.score}`);
}
```

## 🔍 Debugging & Analysis

### Inspect Lineage

```typescript
const lineage = tracker.getLineage(variationId);

lineage.forEach((v, idx) => {
  const improvement = idx > 0
    ? (v.metrics.score - lineage[idx - 1].metrics.score).toFixed(3)
    : '0.000';

  console.log(`${idx}. ${v.mutation.padEnd(20)} Score: ${v.metrics.score.toFixed(3)} (+${improvement})`);
});
```

### Export for Analysis

```typescript
// Export to JSON
const json = tracker.exportLineage(variationId);
fs.writeFileSync('lineage.json', json);

// Import into data analysis tool
```

### Find Regressions

```typescript
const lineage = tracker.getLineage(variationId);

for (let i = 1; i < lineage.length; i++) {
  if (lineage[i].metrics.score < lineage[i - 1].metrics.score) {
    console.log(`⚠️ Regression at step ${i}: ${lineage[i].mutation}`);
    console.log(`  ${lineage[i - 1].metrics.score} → ${lineage[i].metrics.score}`);
  }
}
```

## 🎓 Advanced Topics

### Custom Metrics

```typescript
const variation = createOriginalVariation('Write code', 0.5, 0.01, 1000);

variation.metrics.custom = {
  readability: 0.8,
  maintainability: 0.7,
  security: 0.9,
};

tracker.trackVariation(variation);
```

### Database Integration

```typescript
// Save to database
const lineage = tracker.getLineage(variationId);

await db.variationLineage.createMany({
  data: lineage.map((v) => ({
    id: v.id,
    parentId: v.parentId,
    prompt: v.currentPrompt,
    mutation: v.mutation,
    score: v.metrics.score,
    generation: v.generation,
  })),
});
```

### Real-Time Tracking

```typescript
// Track during optimization
const optimizeWithTracking = async (prompt: string) => {
  const original = createOriginalVariation(prompt, 0.5, 0.01, 1000);
  tracker.trackVariation(original);

  let current = original;

  for (let i = 0; i < 10; i++) {
    const mutation = selectBestMutation();
    const mutated = applyMutation(current.currentPrompt, mutation);
    const score = await evaluate(mutated);

    const child = createChildVariation(current, mutated, mutation, {}, score, 0.01, 1000);
    tracker.trackVariation(child);

    // Real-time visualization
    console.log(visualizeTree(tracker.visualizeLineage(original.id)));

    current = child;
  }

  return current;
};
```

## 🚀 Running the Demo

```bash
npx tsx src/lineage/tracker.demo.ts
```

Demonstrates:
1. ✅ Basic tracking
2. ✅ Multiple branches
3. ✅ Best path finding
4. ✅ Success rate analysis
5. ✅ Human feedback
6. ✅ Generation analysis

## 🔗 Integration

### With Genetic Algorithm

```typescript
import { geneticOptimize } from '../optimizer/genetic';

// Track genetic evolution
result.generationHistory.forEach((gen, idx) => {
  const variation = createChildVariation(
    parent,
    gen.bestIndividual.prompt,
    'genetic-evolution',
    { generation: idx },
    gen.bestFitness / 100,
    0.01,
    1000
  );
  tracker.trackVariation(variation);
});
```

### With MCTS

```typescript
import { mctsOptimize } from '../optimizer/mcts';

// Track MCTS path
result.path.forEach((step, idx) => {
  // Track each step in the discovered path
});
```

## 📚 Related Directives

- **DIRECTIVE-015**: Human Feedback Score (integrates with lineage)
- **DIRECTIVE-019**: Hill-Climbing (can use lineage)
- **DIRECTIVE-020**: Genetic Algorithm (can use lineage)
- **DIRECTIVE-022**: Bandits/MCTS (can use lineage)

## Summary

**Lineage Tracking provides:**
- ✅ Complete evolution history
- ✅ Parent-child relationships
- ✅ Path discovery & optimization
- ✅ Success rate analysis
- ✅ Human feedback integration
- ✅ Tree visualization

**Use it to:**
- 🔍 Understand what works
- 📊 Analyze mutation effectiveness
- 🎯 Find optimal paths
- 🐛 Debug optimization strategies
- 📈 Track improvement over time

**Status:** ✅ Fully Implemented (DIRECTIVE-028)
