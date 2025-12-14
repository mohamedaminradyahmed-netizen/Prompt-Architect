
import { hillClimbingOptimize, ScoringFunction } from '../../optimizer/hillClimbing';
import * as mutations from '../../mutations';

// Mock the mutation functions
jest.mock('../../mutations', () => ({
  ...jest.requireActual('../../mutations'),
  tryCatchStyleMutation: jest.fn(),
  reduceContextMutation: jest.fn(),
  expandMutation: jest.fn(),
  constrainMutation: jest.fn(),
}));

const mockedMutations = mutations as jest.Mocked<typeof mutations>;

describe('hillClimbingOptimize', () => {
  const originalRandom = Math.random;

  beforeEach(() => {
    // Reset mocks before each test
    mockedMutations.tryCatchStyleMutation.mockClear();
    mockedMutations.reduceContextMutation.mockClear();
    mockedMutations.expandMutation.mockClear();
    mockedMutations.constrainMutation.mockClear();

    // Control random for deterministic tests
    let randomCallCount = 0;
    // We want to force specific mutations: 
    // 0: tryCatchStyle, 1: reduceContext, 2: expand, 3: constrain
    const sequence = [0.0, 0.26, 0.51, 0.76, 0.51]; // map to 0, 1, 2, 3, 2 roughly
    Math.random = () => {
      const val = sequence[randomCallCount % sequence.length];
      randomCallCount++;
      return val;
    };
  });

  afterEach(() => {
    Math.random = originalRandom;
  });

  test('should improve prompt score over iterations', async () => {
    const initialPrompt = 'initial prompt';
    const betterPrompt = 'better prompt';
    const bestPrompt = 'best prompt';

    // 0: tryCatch -> 'better prompt' (0.7)
    mockedMutations.tryCatchStyleMutation.mockReturnValue({
      text: betterPrompt,
      mutationType: 'try-catch-style',
      changeDescription: 'mock change',
      expectedImpact: {},
    });

    // 1: reduce -> 'bad prompt' (0.3)
    mockedMutations.reduceContextMutation.mockReturnValue({
      text: 'bad prompt',
      mutationType: 'context-reduction',
      changeDescription: 'mock change',
      expectedImpact: {},
    });

    // 2: expand -> 'best prompt' (0.9)
    mockedMutations.expandMutation.mockReturnValue({
      text: bestPrompt,
      mutationType: 'expansion',
      changeDescription: 'mock change',
      expectedImpact: {},
    });

    // 3: constrain -> 'ok prompt' (0.5)
    mockedMutations.constrainMutation.mockReturnValue({
      text: 'ok prompt',
      mutationType: 'constraint-addition',
      changeDescription: 'mock change',
      expectedImpact: {},
    });


    // Mock scoring function
    const scoringFunction: ScoringFunction = jest.fn().mockImplementation(async (prompt: string) => {
      if (prompt === initialPrompt) return 0.5;
      if (prompt === betterPrompt) return 0.7;
      if (prompt === bestPrompt) return 0.9;
      if (prompt === 'bad prompt') return 0.3;
      return 0.4;
    });

    // Run optimization
    const result = await hillClimbingOptimize(initialPrompt, 5, scoringFunction);

    // Assertions
    // We start at 0.5
    // Iteration 0 (rand ~0.0 -> tryCatch): 'better prompt' (0.7). Improvement. History push. Score 0.7
    // Iteration 1 (rand ~0.26 -> reduce): 'bad prompt' (0.3). No improvement. No history. Score 0.7
    // Iteration 2 (rand ~0.51 -> expand): 'best prompt' (0.9). Improvement. History push. Score 0.9
    // Iteration 3 (rand ~0.76 -> constrain): 'ok prompt' (0.5). No improvement. No history. Score 0.9
    // Iteration 4 (rand ~0.51 -> expand): 'best prompt' (0.9). No improvement (same score). No history? 
    // Wait, 0.9 > 0.9 is false. So no history push.

    // Expected History:
    // 1. Initial (0.5)
    // 2. Better (0.7)
    // 3. Best (0.9)
    // Total length: 3

    expect(result.bestScore).toBe(0.9);
    expect(result.bestPrompt).toBe(bestPrompt);
    expect(result.history.length).toBe(3);
    expect(result.iterations).toBe(5);
  });

  test('should not change prompt if no better mutation is found', async () => {
    const initialPrompt = 'initial prompt';
    Math.random = () => 0.0; // Always tryCatch

    mockedMutations.tryCatchStyleMutation.mockReturnValue({
      text: 'worse prompt',
      mutationType: 'try-catch-style',
      changeDescription: 'mock change',
      expectedImpact: {},
    });

    const scoringFunction: ScoringFunction = jest.fn().mockImplementation(async (prompt: string) => {
      if (prompt === initialPrompt) return 0.8;
      return 0.5;
    });

    const result = await hillClimbingOptimize(initialPrompt, 3, scoringFunction);

    expect(result.bestPrompt).toBe(initialPrompt);
    expect(result.bestScore).toBe(0.8);
    expect(result.history.length).toBe(1); // Only initial
  });
});
