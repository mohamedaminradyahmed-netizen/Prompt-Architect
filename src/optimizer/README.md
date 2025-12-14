# Prompt Optimizer Suite

## Overview

This directory contains a comprehensive suite of optimization algorithms for prompt engineering. The **Hybrid Optimizer** (DIRECTIVE-024) combines multiple optimization strategies to provide superior performance.

---

## 🎯 Quick Start: Hybrid Optimizer (Recommended)

The Hybrid Optimizer is the **most advanced system** that combines:

1. **Genetic Algorithm** (Global Exploration)
2. **Hill-Climbing** (Local Refinement)  
3. **Bayesian Optimization** (Parameter Tuning)

```typescript
import { hybridOptimize, HybridConfig } from './optimizer/hybrid';

const result = await hybridOptimize(
    "Your initial prompt",
    scoringFunction,
    {
        explorationBudget: 3,
        refinementBudget: 5,
        finetuningBudget: 10
    }
);
```

**📖 See**: `DIRECTIVE-024-COMPLETE.md` for full documentation

---

## 📚 Available Optimizers

| Optimizer | Directive | Status | Best For |
|-----------|-----------|--------|----------|
| **Hybrid** | DIRECTIVE-024 | ✅ Complete | Production use, best quality |
| **Genetic** | DIRECTIVE-020 | ✅ Complete | Exploring diverse solutions |
| **Hill-Climbing** | DIRECTIVE-019 | ✅ Complete | Quick local refinement |
| **Bayesian** | DIRECTIVE-021 | ✅ Complete | Parameter optimization |
| **Bandits** | DIRECTIVE-022 | ✅ Complete | Large mutation spaces |
| **MCTS** | DIRECTIVE-022 | ✅ Complete | Tree-based exploration |

---

## Genetic Algorithm Optimizer - DIRECTIVE-020

### Overview

A population-based optimization system using evolutionary principles to automatically discover high-quality prompt variations. This implementation uses genetic algorithms to explore the solution space efficiently.

## Key Features

✅ **Population-Based Search**: Maintains diverse population of prompt variations
✅ **Multiple Selection Strategies**: Tournament, Roulette, and Rank-based selection
✅ **Intelligent Crossover**: Combines successful prompt patterns
✅ **Adaptive Mutation**: Applies random variations for exploration
✅ **Elitism**: Preserves best solutions across generations
✅ **Convergence Detection**: Automatically detects when optimization plateaus
✅ **Diversity Tracking**: Monitors population diversity to avoid premature convergence

## How It Works

### 1. Initialize Population (Generation 0)

```
Original Prompt → [Variation 1, Variation 2, ..., Variation N]
```

- Creates N diverse variations using different mutation combinations
- Each variation is evaluated for fitness

### 2. Evolution Loop (Generations 1-N)

```
For each generation:
  1. SELECT: Choose best performers (parents)
  2. CROSSOVER: Combine parents to create offspring
  3. MUTATE: Apply random mutations for diversity
  4. EVALUATE: Score all new individuals
  5. REPLACE: Form new population (with elitism)
```

### 3. Termination

- Stops after N generations OR
- When population converges (minimal improvement over 4 generations)

## Quick Start

### Basic Usage

```typescript
import { geneticOptimize } from './optimizer/genetic';

// Define fitness function
const fitnessFunction = (prompt: string): number => {
  // Score from 0-100
  let score = 50;
  if (prompt.includes('function')) score += 20;
  if (prompt.length > 100) score += 10;
  return score;
};

// Run optimization
const result = await geneticOptimize(
  'Write code to sort array', // Initial prompt
  fitnessFunction,
  {
    populationSize: 20,
    generations: 10,
    crossoverRate: 0.7,
    mutationRate: 0.3,
    elitismCount: 2,
  }
);

console.log('Best Prompt:', result.summary.bestPrompt);
console.log('Best Fitness:', result.summary.bestOverallFitness);
console.log('Improvement:', result.summary.improvementPercent + '%');
```

### With Balance Metrics

```typescript
import { geneticOptimize } from './optimizer/genetic';
import { validateMetrics, BALANCED } from '../config/balanceMetrics';

// Fitness based on balance metrics
const balancedFitness = async (prompt: string): Promise<number> => {
  // Simulate or measure actual metrics
  const metrics = {
    quality: 0.8,
    cost: 0.02,
    latency: 2500,
    hallucinationRate: 0.1,
    similarity: 0.85,
  };

  const validation = validateMetrics(metrics, BALANCED);
  return validation.score; // 0-100
};

const result = await geneticOptimize(
  'Create authentication system',
  balancedFitness,
  { populationSize: 15, generations: 8 }
);
```

## Configuration Options

### GeneticConfig Interface

```typescript
interface GeneticConfig {
  populationSize: number;      // Number of individuals per generation
  generations: number;         // How many generations to evolve
  crossoverRate: number;       // Probability of crossover (0-1)
  mutationRate: number;        // Probability of mutation (0-1)
  elitismCount: number;        // Top N to preserve unchanged
  selectionStrategy?: string;  // 'tournament' | 'roulette' | 'rank'
  tournamentSize?: number;     // Size for tournament selection
}
```

### Recommended Settings

| Use Case | Population | Generations | Crossover | Mutation | Elitism |
|----------|-----------|-------------|-----------|----------|---------|
| **Quick Exploration** | 10-15 | 5-8 | 0.7 | 0.3 | 2 |
| **Balanced** | 20-30 | 10-15 | 0.7 | 0.3 | 2-3 |
| **Thorough Search** | 40-50 | 20-30 | 0.75 | 0.25 | 3-5 |
| **Diversity Focus** | 30-40 | 15-20 | 0.6 | 0.4 | 2 |

## Selection Strategies

### Tournament Selection (Default)

- **How**: Pick K random individuals, select best
- **Pros**: Fast, good selection pressure
- **Cons**: May lose diversity with large tournament size
- **Best For**: Most use cases

```typescript
{ selectionStrategy: 'tournament', tournamentSize: 3 }
```

### Roulette Wheel Selection

- **How**: Probability proportional to fitness
- **Pros**: Maintains diversity, all individuals have chance
- **Cons**: Weak selection pressure
- **Best For**: Early exploration phases

```typescript
{ selectionStrategy: 'roulette' }
```

### Rank Selection

- **How**: Probability based on rank, not absolute fitness
- **Pros**: Consistent selection pressure
- **Cons**: Slower than tournament
- **Best For**: Avoiding premature convergence

```typescript
{ selectionStrategy: 'rank' }
```

## Mutation Operators

The genetic algorithm uses these mutation operators:

1. **Try-Catch Style**: Makes prompts more flexible
2. **Context Reduction**: Removes verbose content
3. **Expand**: Adds detail and clarity
4. **Word Swap**: Swaps word positions
5. **Sentence Shuffle**: Reorders sentences

Mutations are applied randomly based on `mutationRate`.

## Result Structure

```typescript
interface PopulationResult {
  bestPrompts: Individual[];           // Top 5 prompts
  finalPopulation: Individual[];       // All final individuals
  generationHistory: GenerationStats[]; // Statistics per generation
  summary: {
    totalGenerations: number;
    totalEvaluations: number;
    bestOverallFitness: number;
    bestPrompt: string;
    improvementPercent: number;
    convergenceGeneration?: number;
  };
}
```

## Performance Characteristics

### Time Complexity

- **Per Generation**: O(P² × M)
  - P = population size
  - M = mutation complexity
- **Total**: O(G × P² × M)
  - G = number of generations

### Space Complexity

- O(P × L)
  - L = average prompt length

### Typical Performance

| Population | Generations | Evaluations | Time (est.) |
|-----------|-------------|-------------|-------------|
| 10 | 5 | 50 | ~5 sec |
| 20 | 10 | 200 | ~20 sec |
| 30 | 15 | 450 | ~45 sec |
| 50 | 20 | 1000 | ~2 min |

*Times assume simple fitness function. Complex fitness (API calls) will be significantly slower.*

## Advanced Features

### Elitism

Preserves top N individuals across generations:

```typescript
{ elitismCount: 3 } // Keep top 3 unchanged
```

Benefits:

- ✅ Never lose best solution
- ✅ Faster convergence
- ❌ May reduce diversity

### Diversity Tracking

Automatically monitors population diversity:

```typescript
result.generationHistory.forEach(gen => {
  console.log(`Gen ${gen.generation}: Diversity=${gen.diversity}`);
});
```

High diversity = exploring different solutions
Low diversity = converging on similar solutions

### Convergence Detection

Stops early if improvement < 1% over 4 generations:

```typescript
if (result.summary.convergenceGeneration) {
  console.log(`Converged at generation ${result.summary.convergenceGeneration}`);
}
```

## Examples

### Example 1: Code Generation Prompt

```typescript
const initialPrompt = 'Write login code';

const codeFitness = (prompt: string): number => {
  let score = 0;
  if (/\b(function|class)\b/i.test(prompt)) score += 25;
  if (/\b(typescript|javascript)\b/i.test(prompt)) score += 25;
  if (/\b(error|validation)\b/i.test(prompt)) score += 25;
  if (prompt.length >= 100 && prompt.length <= 400) score += 25;
  return score;
};

const result = await geneticOptimize(initialPrompt, codeFitness, {
  populationSize: 20,
  generations: 12,
});

// Result might be:
// "Try to implement a TypeScript function for user login authentication.
//  Include error handling and input validation. Ensure secure password
//  verification and session management."
```

### Example 2: Content Writing Prompt

```typescript
const initialPrompt = 'Write blog post about AI';

const contentFitness = (prompt: string): number => {
  let score = 0;
  if (/\b(engaging|compelling|informative)\b/i.test(prompt)) score += 20;
  if (/\b(examples|stories|case studies)\b/i.test(prompt)) score += 20;
  if (/\b(tone|audience|style)\b/i.test(prompt)) score += 20;
  if (prompt.includes('words') || prompt.includes('length')) score += 20;
  if (prompt.length >= 80) score += 20;
  return score;
};

const result = await geneticOptimize(initialPrompt, contentFitness);
```

## Comparison with Other Optimizers

| Feature | Genetic | Hill-Climbing | Bayesian |
|---------|---------|---------------|----------|
| **Population-based** | ✅ Yes | ❌ No | ❌ No |
| **Exploration** | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| **Speed** | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Global Optimum** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Parallelizable** | ✅ Yes | ❌ No | ⚠️ Partial |
| **Best For** | Diverse solutions | Quick refinement | Parameter tuning |

## Best Practices

### 1. Choose Appropriate Population Size

```typescript
// Small problem space
{ populationSize: 10-15 }

// Medium problem space
{ populationSize: 20-30 }

// Large problem space
{ populationSize: 40-50 }
```

### 2. Balance Exploration vs Exploitation

**More Exploration** (diverse solutions):

```typescript
{
  crossoverRate: 0.6,  // Lower
  mutationRate: 0.4,   // Higher
  elitismCount: 1,     // Lower
}
```

**More Exploitation** (refine best):

```typescript
{
  crossoverRate: 0.8,  // Higher
  mutationRate: 0.2,   // Lower
  elitismCount: 5,     // Higher
}
```

### 3. Monitor Diversity

```typescript
const result = await geneticOptimize(...);

// Check if diversity dropped too quickly
const earlyDiv = result.generationHistory[0].diversity;
const lateDiv = result.generationHistory[result.generationHistory.length - 1].diversity;

if (lateDiv < earlyDiv * 0.1) {
  console.warn('⚠️ Population may have converged prematurely');
  // Consider: higher mutation rate, larger population
}
```

### 4. Use Appropriate Fitness Function

```typescript
// ❌ BAD: Too simple
const badFitness = (prompt: string) => prompt.length;

// ✅ GOOD: Multiple criteria
const goodFitness = (prompt: string) => {
  let score = 0;
  // Criterion 1: Has action verb (25%)
  // Criterion 2: Has constraints (25%)
  // Criterion 3: Appropriate length (25%)
  // Criterion 4: Domain keywords (25%)
  return score;
};
```

## Troubleshooting

### Problem: Slow Convergence

**Solutions:**

- Increase population size
- Increase crossover rate
- Decrease mutation rate
- Use tournament selection

### Problem: Premature Convergence

**Solutions:**

- Increase mutation rate
- Decrease elitism count
- Use roulette or rank selection
- Increase diversity

### Problem: No Improvement

**Solutions:**

- Check fitness function (is it meaningful?)
- Increase generations
- Increase population size
- Try different selection strategy

## Running the Demo

```bash
npx tsx src/optimizer/genetic.demo.ts
```

This will run 5 comprehensive demos showing:

1. Basic optimization
2. Advanced fitness with balance metrics
3. Comparison of selection strategies
4. Impact of population size
5. Real-world code generation example

## API Reference

### Main Function

```typescript
geneticOptimize(
  initialPrompt: string,
  fitnessFunction: FitnessFunction,
  config?: Partial<GeneticConfig>
): Promise<PopulationResult>
```

### Types

See [genetic.ts](./genetic.ts) for full type definitions.

## Related Directives

- **DIRECTIVE-019**: Hill-Climbing Optimizer (simpler, faster)
- **DIRECTIVE-021**: Bayesian Optimization (for parameter tuning)
- **DIRECTIVE-024**: Hybrid Optimizer (combines multiple approaches)

## Summary

The Genetic Algorithm Optimizer excels at:

- ✅ Exploring diverse solution spaces
- ✅ Finding multiple good solutions
- ✅ Avoiding local optima
- ✅ Handling complex fitness landscapes

Use it when:

- You want diverse variations
- Global optimum is important
- You have computational budget
- Problem space is large

**Status:** ✅ Fully Implemented (DIRECTIVE-020)
