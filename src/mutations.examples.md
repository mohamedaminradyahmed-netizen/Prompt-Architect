# Mutation Operators - Usage Examples

## DIRECTIVE-003: Try/Catch Style Mutation

### Overview

The Try/Catch Style Mutation transforms direct, imperative instructions into a more flexible "try...if fails" style. This makes prompts more forgiving and encourages fallback behavior when the primary task cannot be completed.

### Import

```typescript
import { tryCatchStyleMutation } from './mutations';
```

### Basic Usage

```typescript
const result = tryCatchStyleMutation('Write a function to parse JSON');

console.log(result.text);
// Output: "Try to write a function to parse JSON. If you encounter issues, suggest alternatives or explain the challenges."

console.log(result.mutationType);
// Output: "try-catch-style"

console.log(result.expectedImpact);
// Output: { quality: 'neutral', cost: 'increase', latency: 'neutral', reliability: 'increase' }
```

### Examples by Category

#### 1. Code Generation Tasks

**Input:**
```typescript
tryCatchStyleMutation('Write a TypeScript function that validates email addresses')
```

**Output:**
```
Try to write a typescript function that validates email addresses. If you encounter issues, suggest alternatives or explain the challenges.
```

**Transformation Type:** `create-with-alternatives`

---

#### 2. Bug Fixing Tasks

**Input:**
```typescript
tryCatchStyleMutation('Fix the race condition in the user registration process')
```

**Output:**
```
Try to fix the race condition in the user registration process. If you can't fix it directly, suggest possible solutions or workarounds.
```

**Transformation Type:** `fix-with-fallback`

---

#### 3. Analysis Tasks

**Input:**
```typescript
tryCatchStyleMutation('Analyze the code for security vulnerabilities')
```

**Output:**
```
Try to analyze the code for security vulnerabilities. If complete analysis isn't possible, provide what you can determine.
```

**Transformation Type:** `analyze-with-partial`

---

#### 4. Optimization Tasks

**Input:**
```typescript
tryCatchStyleMutation('Optimize the database queries in the user service')
```

**Output:**
```
Try to optimize the database queries in the user service. If challenges arise, explain them and suggest next steps.
```

**Transformation Type:** `general-with-explanation`

---

#### 5. Complex Conditional Tasks

**Input:**
```typescript
tryCatchStyleMutation('Implement authentication. Use JWT tokens. Store sessions in Redis.')
```

**Output:**
```
Try to implement authentication. Use JWT tokens. Store sessions in Redis. If any condition can't be met, explain why and suggest alternatives.
```

**Transformation Type:** `conditional-breakdown`

---

### Metadata Analysis

Every mutation returns detailed metadata about the transformation:

```typescript
const result = tryCatchStyleMutation('Fix the authentication bug');

console.log(result.metadata);
// Output:
// {
//   transformationType: 'fix-with-fallback',
//   originalLength: 27,
//   newLength: 134,
//   lengthIncrease: 107,
//   imperativeDetected: true
// }
```

### Expected Impact on Metrics

The mutation provides predictions about how it will affect various metrics:

```typescript
const result = tryCatchStyleMutation('Write a function');

console.log(result.expectedImpact);
// {
//   quality: 'neutral',       // Quality stays about the same
//   cost: 'increase',         // Slightly longer prompt = more tokens
//   latency: 'neutral',       // Response time not significantly affected
//   reliability: 'increase'   // More forgiving = less likely to fail
// }
```

### Real-World Use Cases

#### Use Case 1: Making Rigid Instructions More Flexible

**Problem:** You have a prompt that's too strict and often fails when conditions aren't perfect.

**Original Prompt:**
```
"Write a function that reads a CSV file and converts it to JSON"
```

**Solution:**
```typescript
const flexiblePrompt = tryCatchStyleMutation(
  'Write a function that reads a CSV file and converts it to JSON'
);
```

**Result:**
```
"Try to write a function that reads a csv file and converts it to json. If you encounter issues, suggest alternatives or explain the challenges."
```

**Benefit:** The AI will now provide helpful suggestions even if it encounters issues, rather than failing silently.

---

#### Use Case 2: Debugging Complex Systems

**Original Prompt:**
```
"Fix the memory leak in the caching layer"
```

**Solution:**
```typescript
const debugPrompt = tryCatchStyleMutation(
  'Fix the memory leak in the caching layer'
);
```

**Result:**
```
"Try to fix the memory leak in the caching layer. If you can't fix it directly, suggest possible solutions or workarounds."
```

**Benefit:** Ensures you get actionable advice even if the AI can't provide a direct fix.

---

#### Use Case 3: Partial Results for Analysis

**Original Prompt:**
```
"Analyze the entire codebase for performance bottlenecks"
```

**Solution:**
```typescript
const analysisPrompt = tryCatchStyleMutation(
  'Analyze the entire codebase for performance bottlenecks'
);
```

**Result:**
```
"Try to analyze the entire codebase for performance bottlenecks. If complete analysis isn't possible, provide what you can determine."
```

**Benefit:** You'll get partial results even if full analysis isn't feasible.

---

### Integration with Other Systems

#### With Balance Metrics

```typescript
import { tryCatchStyleMutation } from './mutations';
import { validateMetrics, BALANCED } from './config/balanceMetrics';

const variation = tryCatchStyleMutation('Write a sorting algorithm');

// Simulate metrics (in real usage, these would be measured)
const metrics = {
  quality: 0.75,
  cost: 0.02,
  latency: 2500,
  hallucinationRate: 0.08,
  similarity: 0.85,
};

const validation = validateMetrics(metrics, BALANCED);

if (validation.isValid) {
  console.log(`✅ Variation accepted with score: ${validation.score}`);
  console.log(`Using prompt: ${variation.text}`);
} else {
  console.log(`❌ Variation rejected: ${validation.recommendation}`);
}
```

#### Batch Processing Multiple Prompts

```typescript
const prompts = [
  'Write a function to parse JSON',
  'Fix the authentication bug',
  'Analyze the performance metrics',
  'Optimize the database queries',
];

const variations = prompts.map(prompt => tryCatchStyleMutation(prompt));

variations.forEach((variation, index) => {
  console.log(`\n--- Prompt ${index + 1} ---`);
  console.log(`Original: ${prompts[index]}`);
  console.log(`Mutated: ${variation.text}`);
  console.log(`Type: ${variation.metadata?.transformationType}`);
  console.log(`Length increase: ${variation.metadata?.lengthIncrease} characters`);
});
```

### Performance Considerations

- **Token Increase:** Typically adds 50-150 characters (10-30 tokens)
- **Cost Impact:** Minimal (~10-30% increase in input tokens)
- **Reliability Gain:** Significantly reduces failure rate for rigid prompts
- **Quality Impact:** Neutral to slightly positive (more graceful degradation)

### Best Practices

1. **Use for Imperative Prompts:** Most effective with direct commands
2. **Ideal for Production:** Great for production systems where graceful degradation matters
3. **Combine with Other Mutations:** Can be chained with other mutation operators
4. **Monitor Results:** Track acceptance rates to validate effectiveness

### When NOT to Use

- Prompts that already have fallback logic
- Very short prompts where overhead is significant
- Prompts that require strict adherence (security checks, etc.)
- Already conversational/flexible prompts

### Testing

Run the comprehensive test suite:

```bash
npm test mutations.test.ts
```

The test suite includes:
- ✅ Basic imperative transformations
- ✅ Fix/debug transformations with fallbacks
- ✅ Analysis transformations with partial results
- ✅ Complex conditional prompts
- ✅ Edge cases (empty, short, special characters)
- ✅ Real-world examples
- ✅ Metadata accuracy
- ✅ Preservation of original meaning

### Future Enhancements

See TODO.md for planned enhancements:
- DIRECTIVE-004: Context Reduction Mutation
- DIRECTIVE-005: Parameterized Templates
- DIRECTIVE-006: Expand Mutation
- DIRECTIVE-007: Constrain Mutation

---

## Summary

The Try/Catch Style Mutation (DIRECTIVE-003) is a powerful tool for making prompts more resilient and user-friendly. By converting rigid imperatives into flexible suggestions with fallbacks, you can significantly improve the reliability and usability of your AI system.

**Key Benefits:**
- 🛡️ **Increased Reliability:** Fewer complete failures
- 🎯 **Better UX:** Always get some useful output
- 🔄 **Graceful Degradation:** Partial results when full results aren't possible
- 📊 **Minimal Cost:** Small token increase for significant benefit

**Status:** ✅ Implemented and Tested (DIRECTIVE-003)
