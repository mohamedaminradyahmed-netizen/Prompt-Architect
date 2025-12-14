# Bandits & MCTS Optimizers - DIRECTIVE-022

## Overview

Advanced optimization algorithms for efficiently exploring large mutation spaces:
- **Multi-Armed Bandits (UCB1)**: Selects best single mutation type
- **Monte Carlo Tree Search (MCTS)**: Discovers optimal mutation sequences

## 🎰 Multi-Armed Bandits (UCB1)

### Concept

Treats each mutation type as an "arm" on a slot machine. Efficiently determines which mutations work best through balanced exploration and exploitation.

**Key Insight:** You don't need to try everything equally - focus on what works!

### How It Works

```
1. Try each mutation type once (initial exploration)
2. Calculate UCB value for each: UCB = AvgReward + C × √(ln(total)/pulls)
3. Pull arm with highest UCB
4. Update statistics
5. Repeat until budget exhausted
```

### When to Use

✅ **Use Bandits when:**
- Large number of mutation types available
- Limited evaluation budget
- Need quick results
- Want to find best single transformation

❌ **Don't use when:**
- Need sequences of mutations (use MCTS)
- Very small mutation space (just try all)
- Mutations have complex dependencies

### Quick Start

```typescript
import { banditOptimize } from './optimizer/bandits';

const scoringFunction = async (prompt: string): Promise<number> => {
  // Your scoring logic (0-1 range)
  let score = 0.5;
  if (prompt.includes('function')) score += 0.2;
  return score;
};

const result = await banditOptimize(
  'Write code for user login',  // initial prompt
  50,                            // budget (number of trials)
  scoringFunction                // scoring function
);

console.log('Best Mutation:', result.bestMutationId);
console.log('Best Prompt:', result.bestPrompt);
console.log('Best Score:', result.bestScore);
```

### Result Structure

```typescript
interface BanditResult {
  bestMutationId: string;           // Which mutation worked best
  bestPrompt: string;               // Best prompt found
  bestScore: number;                // Best score achieved
  armStats: Record<string, {        // Statistics for each mutation
    pulls: number;                  // Times this mutation was tried
    avgReward: number;              // Average reward
    confidence: number;             // Confidence in this estimate
  }>;
}
```

### Available Mutations (Arms)

1. **try-catch**: Makes prompts more flexible with fallbacks
2. **reduce-context**: Removes verbose content
3. **expand**: Adds detail and clarity
4. **constrain**: Adds category-specific constraints

### UCB1 Algorithm Details

**Formula:** `UCB = Exploitation + Exploration`

```
Exploitation = Average Reward
Exploration = C × √(ln(total pulls) / arm pulls)
```

**Parameters:**
- `C` (exploration constant): Default = 1.41 (√2)
  - Higher C = more exploration
  - Lower C = more exploitation

**Behavior:**
- **Early phase**: High exploration (tries all arms)
- **Middle phase**: Balanced (tests promising arms more)
- **Late phase**: Exploitation (focuses on best arm)

## 🌲 Monte Carlo Tree Search (MCTS)

### Concept

Builds a tree of possible mutation sequences, exploring promising paths deeper. Finds optimal chains of transformations.

**Key Insight:** Sometimes multiple mutations in sequence work better than any single mutation!

### How It Works

```
1. Selection: Navigate tree using UCB1
2. Expansion: Add new child node (apply mutation)
3. Simulation: Evaluate the prompt
4. Backpropagation: Update all ancestors with result
5. Repeat until budget exhausted
```

### When to Use

✅ **Use MCTS when:**
- Need sequences of transformations
- Have computational budget for tree search
- Want to discover complex patterns
- Mutations can be chained effectively

❌ **Don't use when:**
- Need very fast results (use Bandits)
- Single mutations are sufficient
- Evaluation is very expensive
- Shallow transformations preferred

### Quick Start

```typescript
import { mctsOptimize } from './optimizer/mcts';

const scoringFunction = async (prompt: string): Promise<number> => {
  // Your scoring logic
  return 0.75;
};

const result = await mctsOptimize(
  'Create authentication system',  // initial prompt
  30,                                // iterations
  4,                                 // max depth
  scoringFunction                    // scoring function
);

console.log('Best Prompt:', result.bestPrompt);
console.log('Score:', result.bestScore);
console.log('Path:', result.path);  // Shows mutation sequence
```

### Result Structure

```typescript
interface MCTSResult {
  bestPrompt: string;      // Best prompt found
  bestScore: number;       // Best score achieved
  path: string[];          // Sequence of mutations applied
  iterations: number;      // Iterations performed
}
```

### Tree Structure

```
                    Root (original prompt)
                   /    |    \    \
            try-catch reduce expand constrain
               /  |  \
         try-catch reduce expand
            /
       reduce
```

Each node:
- **Visits**: Number of times explored
- **Value**: Accumulated score
- **UCB**: Selection priority

### MCTS Parameters

- **iterations**: How many times to run the search (20-50 typical)
- **maxDepth**: Maximum mutation chain length (3-5 typical)
- **explorationConstant**: UCB parameter (default 1.41)

## 📊 Comparison: Bandit vs MCTS

| Feature | **Bandits** | **MCTS** |
|---------|------------|----------|
| **Search Type** | Flat (single mutations) | Tree (sequences) |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Moderate |
| **Depth** | Single step | Multi-step chains |
| **Complexity** | Simple | Complex |
| **Memory** | Low | Higher (tree storage) |
| **Best For** | Quick wins | Deep optimization |
| **Typical Budget** | 30-100 trials | 20-50 iterations |
| **Solution Quality** | Good single mutation | Best sequence |

## 🎯 Usage Patterns

### Pattern 1: Quick Optimization

```typescript
// Use Bandit for fast single-step improvement
const result = await banditOptimize(prompt, 50, scoring);
```

**Best for:** Production systems, quick A/B testing, initial exploration

### Pattern 2: Deep Optimization

```typescript
// Use MCTS for thorough multi-step optimization
const result = await mctsOptimize(prompt, 30, 4, scoring);
```

**Best for:** Offline optimization, discovering best practices, research

### Pattern 3: Combined Approach

```typescript
// Run both and take the best
const banditResult = await banditOptimize(prompt, 50, scoring);
const mctsResult = await mctsOptimize(prompt, 30, 4, scoring);

const best = banditResult.bestScore > mctsResult.bestScore
  ? banditResult.bestPrompt
  : mctsResult.bestPrompt;
```

**Best for:** Critical prompts, maximum quality needed

### Pattern 4: Budget-Aware

```typescript
// Small budget: Bandit only
if (budget < 30) {
  return await banditOptimize(prompt, budget, scoring);
}

// Large budget: MCTS for depth
if (budget > 100) {
  return await mctsOptimize(prompt, budget / 2, 5, scoring);
}

// Medium: Both
const banditResult = await banditOptimize(prompt, budget / 2, scoring);
const mctsResult = await mctsOptimize(prompt, budget / 4, 3, scoring);
```

## 🔬 Advanced Topics

### Custom Scoring Functions

```typescript
// Multi-criteria scoring
const advancedScoring = async (prompt: string): Promise<number> => {
  let score = 0;

  // Criterion 1: Clarity (30%)
  if (/\b(create|write|implement)\b/i.test(prompt)) {
    score += 0.3;
  }

  // Criterion 2: Specificity (30%)
  if (/\b(typescript|async|interface)\b/i.test(prompt)) {
    score += 0.3;
  }

  // Criterion 3: Completeness (40%)
  if (/\b(error|test|validation)\b/i.test(prompt)) {
    score += 0.4;
  }

  return Math.max(0, Math.min(1, score));
};
```

### Tuning UCB1

```typescript
// More exploration (higher C)
// Modify in source: const c = 2.0;  // Instead of 1.41

// More exploitation (lower C)
// const c = 1.0;
```

### MCTS Depth Selection

```typescript
// Shallow search (faster, less thorough)
await mctsOptimize(prompt, 50, 2, scoring);

// Medium depth (balanced)
await mctsOptimize(prompt, 30, 4, scoring);

// Deep search (slower, more thorough)
await mctsOptimize(prompt, 20, 6, scoring);
```

## 📈 Performance Characteristics

### Bandits (UCB1)

| Budget | Time (est.) | Quality | Best Use |
|--------|-------------|---------|----------|
| 20 | ~2 sec | Good | Quick test |
| 50 | ~5 sec | Better | Standard |
| 100 | ~10 sec | Best | Thorough |

**Time Complexity:** O(B) where B = budget
**Space Complexity:** O(A) where A = number of arms

### MCTS

| Iterations | Depth | Time (est.) | Quality | Best Use |
|-----------|-------|-------------|---------|----------|
| 20 | 3 | ~5 sec | Good | Quick |
| 30 | 4 | ~10 sec | Better | Standard |
| 50 | 5 | ~20 sec | Best | Thorough |

**Time Complexity:** O(I × D × A) where I = iterations, D = depth, A = actions
**Space Complexity:** O(A^D) worst case (tree nodes)

## 🎓 Examples

### Example 1: Code Generation

```typescript
const codeScoring = async (prompt: string) => {
  let score = 0.3;
  if (/typescript|javascript/i.test(prompt)) score += 0.2;
  if (/function|class/i.test(prompt)) score += 0.2;
  if (/test|error/i.test(prompt)) score += 0.3;
  return score;
};

// Bandit approach
const result = await banditOptimize(
  'Write login code',
  40,
  codeScoring
);

// Result might be:
// "Try to implement a TypeScript function for user login authentication
//  with error handling and validation..."
```

### Example 2: Content Writing

```typescript
const contentScoring = async (prompt: string) => {
  let score = 0.4;
  if (/\d+ words/i.test(prompt)) score += 0.2;
  if (/tone|style/i.test(prompt)) score += 0.2;
  if (/example|like/i.test(prompt)) score += 0.2;
  return score;
};

// MCTS approach (for complex transformations)
const result = await mctsOptimize(
  'Write blog post about AI',
  25,
  4,
  contentScoring
);

// Discovers optimal sequence:
// Path: ['expand', 'constrain', 'try-catch']
```

## 🐛 Troubleshooting

### Problem: Bandit always picks same arm

**Cause:** One mutation much better than others, or C too low

**Solution:**
- Increase exploration constant C
- Add more diverse mutations
- Check if scoring function is too biased

### Problem: MCTS doesn't improve over shallow search

**Cause:** Mutations don't compound well, or depth too high

**Solution:**
- Reduce maxDepth to 2-3
- Check if mutations are complementary
- Try Bandit instead (single step may be enough)

### Problem: Results seem random

**Cause:** Scoring function not meaningful or too much noise

**Solution:**
- Verify scoring function logic
- Increase budget for more trials
- Reduce randomness in scoring

### Problem: Too slow

**Cause:** Budget/iterations too high, scoring too expensive

**Solution:**
- Reduce budget/iterations
- Use Bandit instead of MCTS
- Optimize scoring function
- Consider caching scores

## 🚀 Running the Demo

```bash
npx tsx src/optimizer/bandits-mcts.demo.ts
```

This runs 6 comprehensive demos:
1. ✅ Multi-Armed Bandit (UCB1)
2. ✅ Monte Carlo Tree Search
3. ✅ Bandit vs MCTS Comparison
4. ✅ Budget Analysis
5. ✅ Real-World Examples
6. ✅ Exploration vs Exploitation

## 📚 Further Reading

- **UCB1 Algorithm**: Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002)
- **MCTS**: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- **Applications**: AlphaGo, game playing, optimization

## 🔗 Related Directives

- **DIRECTIVE-019**: Hill-Climbing (simpler, faster)
- **DIRECTIVE-020**: Genetic Algorithm (population-based)
- **DIRECTIVE-024**: Hybrid Optimizer (combines multiple approaches)

## Summary

**Multi-Armed Bandits:**
- ✅ Fast and efficient
- ✅ Finds best single mutation
- ✅ Automatically balances exploration/exploitation
- ✅ Low memory footprint
- 🎯 Use for: Quick optimization, single-step improvements

**Monte Carlo Tree Search:**
- ✅ Discovers optimal sequences
- ✅ Explores complex transformation paths
- ✅ Handles deep search spaces
- ✅ Proven in game AI
- 🎯 Use for: Multi-step optimization, discovering patterns

**Status:** ✅ Fully Implemented (DIRECTIVE-022)
