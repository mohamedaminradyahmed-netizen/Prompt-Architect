import { hillClimbingOptimize, ScoringFunction } from './src/optimizer/hillClimbing';
import { PromptCategory } from './src/types/promptTypes';
import * as mutations from './src/mutations';

// This is a simplified test script to be executed with node after compilation.
// We cannot use a test runner like Jest in this environment.

async function runTest() {
  console.log('Running Hill Climbing Optimizer Test...');

  const initialPrompt = 'initial prompt';
  const betterPrompt = 'a better prompt';
  const bestPrompt = 'the best prompt';

  // We need to mock the mutations for a predictable test.
  // Since we can't use Jest's mocking, we'll monkey-patch the functions for this test run.
  const originalConstrainMutation = mutations.constrainMutation;

  (mutations as any).constrainMutation = (prompt: string, category: PromptCategory) => {
    return {
      text: 'a constrained prompt',
      mutationType: 'constraint-addition',
      changeDescription: 'mock change',
      expectedImpact: {},
    };
  };

  const originalReduceContextMutation = mutations.reduceContextMutation;
  (mutations as any).reduceContextMutation = (prompt: string) => {
      if (prompt.includes("slightly better")) return {
        text: betterPrompt,
        mutationType: 'context-reduction',
        changeDescription: 'mock change',
        expectedImpact: {},
      };
      return {
        text: 'a reduced prompt',
        mutationType: 'context-reduction',
        changeDescription: 'mock change',
        expectedImpact: {},
      };
  };

  const originalExpandMutation = mutations.expandMutation;
  (mutations as any).expandMutation = (prompt: string) => {
    if (prompt.includes("better")) return {
        text: bestPrompt,
        mutationType: 'expansion',
        changeDescription: 'mock change',
        expectedImpact: {},
      };
      return {
        text: 'an expanded prompt',
        mutationType: 'expansion',
        changeDescription: 'mock change',
        expectedImpact: {},
      };
  };
  
  const originalTryCatchStyleMutation = mutations.tryCatchStyleMutation;
  (mutations as any).tryCatchStyleMutation = (prompt: string) => {
    return {
        text: 'a slightly better prompt',
        mutationType: 'try-catch-style',
        changeDescription: 'mock change',
        expectedImpact: {},
    };
  };


  const scoringFunction: ScoringFunction = async (prompt: string) => {
    console.log(`Scoring prompt: "${prompt}"`);
    if (prompt === initialPrompt) return 0.5;
    if (prompt === 'a slightly better prompt') return 0.6;
    if (prompt === betterPrompt) return 0.8;
    if (prompt === bestPrompt) return 0.95;
    return 0.4;
  };

  const result = await hillClimbingOptimize(initialPrompt, 5, scoringFunction);

  console.log('\n--- Optimization Finished ---');
  console.log('Best prompt found:', result.bestPrompt);
  console.log('Best score:', result.bestScore);
  console.log('\nOptimization History:');
  result.history.forEach((entry, index) => {
    console.log(`  Iteration ${index}: Score=${entry.score.toFixed(2)}, Prompt="${entry.prompt}"`);
  });

  // Simple assertions
  console.assert(result.bestScore > 0.5, 'Test failed: Best score should be greater than 0.5');
  console.assert(result.bestPrompt === bestPrompt, `Test failed: Best prompt should be "${bestPrompt}"`);

  console.log('\nTest finished.');

  // Restore original functions
  (mutations as any).constrainMutation = originalConstrainMutation;
  (mutations as any).reduceContextMutation = originalReduceContextMutation;
  (mutations as any).expandMutation = originalExpandMutation;
  (mutations as any).tryCatchStyleMutation = originalTryCatchStyleMutation;
}

runTest().catch(err => {
  console.error('Test run failed:', err);
  process.exit(1);
});
