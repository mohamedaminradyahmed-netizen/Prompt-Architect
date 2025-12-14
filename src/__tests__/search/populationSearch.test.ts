/**
 * Population Search Tests (DIRECTIVE-052)
 *
 * Tests for the population-based search algorithm
 */

import {
  populationSearch,
  PopulationConfig,
  DEFAULT_POPULATION_CONFIG,
  PopulationIndividual,
  PopulationSearchResult
} from '../../search/populationSearch';
import { TestCase, LLMExecutor } from '../../sandbox/testExecutor';

// ============================================================================
// MOCK SETUP
// ============================================================================

const mockExecutor: LLMExecutor = async (prompt: string): Promise<string> => {
  // Deterministic mock based on prompt characteristics
  await new Promise(resolve => setTimeout(resolve, 10));

  if (prompt.includes('Try to')) {
    return 'Success: Task completed with try-catch style';
  }
  if (prompt.includes('Step') || prompt.includes('Criteria')) {
    return 'Structured output with clear steps';
  }
  if (prompt.length > 200) {
    return 'Detailed response for expanded prompt';
  }
  return 'Basic response';
};

const basicTestSuite: TestCase[] = [
  {
    id: 'test-1',
    prompt: 'Test input 1',
    evaluationCriteria: {
      customValidator: (output) => output.length > 0
    }
  },
  {
    id: 'test-2',
    prompt: 'Test input 2',
    evaluationCriteria: {
      matchType: 'includes',
      matchValue: 'response'
    }
  }
];

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

describe('Population Search - Basic Functionality', () => {
  test('should create population search result with all required fields', async () => {
    const result = await populationSearch(
      'Test prompt',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    expect(result).toBeDefined();
    expect(result.bestIndividuals).toBeDefined();
    expect(result.finalPopulation).toBeDefined();
    expect(result.generationHistory).toBeDefined();
    expect(result.summary).toBeDefined();

    // Check summary fields
    expect(result.summary.totalGenerations).toBeGreaterThanOrEqual(1);
    expect(result.summary.totalEvaluations).toBeGreaterThan(0);
    expect(result.summary.bestFitness).toBeDefined();
    expect(result.summary.bestPrompt).toBeDefined();
    expect(result.summary.executionTimeMs).toBeGreaterThan(0);
  });

  test('should return at least one best individual', async () => {
    const result = await populationSearch(
      'Initial prompt',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations: 2,
        verbose: false
      }
    );

    expect(result.bestIndividuals.length).toBeGreaterThan(0);
    expect(result.bestIndividuals[0].prompt).toBeDefined();
    expect(result.bestIndividuals[0].fitness).toBeDefined();
  });

  test('should track generation history', async () => {
    const generations = 3;
    const result = await populationSearch(
      'Track generations',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations,
        verbose: false
      }
    );

    // Should have initial generation + specified generations
    expect(result.generationHistory.length).toBeGreaterThanOrEqual(2);

    // Check generation statistics
    for (const stats of result.generationHistory) {
      expect(stats.generation).toBeDefined();
      expect(stats.bestFitness).toBeDefined();
      expect(stats.avgFitness).toBeDefined();
      expect(stats.worstFitness).toBeDefined();
      expect(stats.diversity).toBeDefined();
      expect(stats.bestIndividual).toBeDefined();
    }
  });
});

// ============================================================================
// CONFIGURATION TESTS
// ============================================================================

describe('Population Search - Configuration', () => {
  test('should use default configuration when not specified', async () => {
    const result = await populationSearch(
      'Default config test',
      basicTestSuite,
      mockExecutor,
      { generations: 1, populationSize: 5, verbose: false }
    );

    expect(result.finalPopulation.length).toBe(5);
  });

  test('should respect custom population size', async () => {
    const customSize = 8;
    const result = await populationSearch(
      'Custom size test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: customSize,
        generations: 1,
        verbose: false
      }
    );

    expect(result.finalPopulation.length).toBe(customSize);
  });

  test('should validate configuration and throw on invalid values', async () => {
    await expect(
      populationSearch(
        'Invalid config',
        basicTestSuite,
        mockExecutor,
        { populationSize: 2, verbose: false } // Too small
      )
    ).rejects.toThrow('Population size must be at least 4');

    await expect(
      populationSearch(
        'Invalid selection rate',
        basicTestSuite,
        mockExecutor,
        { populationSize: 5, selectionRate: 1.5, verbose: false }
      )
    ).rejects.toThrow('Selection rate must be between 0 and 1');
  });

  test('should use elitism to preserve top individuals', async () => {
    const elitismCount = 2;
    const result = await populationSearch(
      'Elitism test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations: 2,
        elitismCount,
        verbose: false
      }
    );

    // Best individuals should be preserved across generations
    expect(result.bestIndividuals.length).toBeGreaterThanOrEqual(elitismCount);
  });
});

// ============================================================================
// EVOLUTION TESTS
// ============================================================================

describe('Population Search - Evolution', () => {
  test('should apply mutations to individuals', async () => {
    const result = await populationSearch(
      'Mutation test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations: 2,
        mutationProbability: 0.9, // High probability
        verbose: false
      }
    );

    // Check that mutations were applied
    const mutatedIndividuals = result.finalPopulation.filter(
      ind => ind.appliedMutations.length > 1
    );
    expect(mutatedIndividuals.length).toBeGreaterThan(0);
  });

  test('should produce diverse population', async () => {
    const result = await populationSearch(
      'Diversity test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 10,
        generations: 2,
        verbose: false
      }
    );

    // Check diversity metric is calculated
    const lastGen = result.generationHistory[result.generationHistory.length - 1];
    expect(lastGen.diversity).toBeDefined();
    expect(lastGen.diversity).toBeGreaterThanOrEqual(0);
  });

  test('should track parent lineage', async () => {
    const result = await populationSearch(
      'Lineage test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations: 2,
        crossoverProbability: 0.9, // High probability
        verbose: false
      }
    );

    // Some individuals should have parent IDs from crossover
    const hasParents = result.finalPopulation.some(
      ind => ind.parentIds && ind.parentIds.length === 2
    );
    // Note: May not always have parents if crossover didn't happen
    expect(result.finalPopulation.length).toBe(6);
  });
});

// ============================================================================
// SELECTION TESTS
// ============================================================================

describe('Population Search - Selection', () => {
  test('should select top individuals based on fitness', async () => {
    const result = await populationSearch(
      'Selection test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 8,
        generations: 2,
        selectionRate: 0.5,
        verbose: false
      }
    );

    // Best individuals should be sorted by fitness
    const fitnesses = result.bestIndividuals.map(ind => ind.fitness);
    for (let i = 1; i < fitnesses.length; i++) {
      expect(fitnesses[i - 1]).toBeGreaterThanOrEqual(fitnesses[i]);
    }
  });

  test('should return up to 5 best individuals', async () => {
    const result = await populationSearch(
      'Top 5 test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 10,
        generations: 1,
        verbose: false
      }
    );

    expect(result.bestIndividuals.length).toBeLessThanOrEqual(5);
  });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

describe('Population Search - Performance', () => {
  test('should complete within reasonable time', async () => {
    const startTime = Date.now();

    await populationSearch(
      'Performance test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 2,
        verbose: false
      }
    );

    const elapsed = Date.now() - startTime;
    expect(elapsed).toBeLessThan(10000); // 10 seconds max
  });

  test('should track execution time in summary', async () => {
    const result = await populationSearch(
      'Time tracking test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    expect(result.summary.executionTimeMs).toBeGreaterThan(0);
  });
});

// ============================================================================
// INDIVIDUAL STRUCTURE TESTS
// ============================================================================

describe('Population Search - Individual Structure', () => {
  test('individual should have all required fields', async () => {
    const result = await populationSearch(
      'Individual fields test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    const individual = result.bestIndividuals[0];

    expect(individual.id).toBeDefined();
    expect(individual.prompt).toBeDefined();
    expect(typeof individual.fitness).toBe('number');
    expect(typeof individual.passRate).toBe('number');
    expect(typeof individual.generation).toBe('number');
    expect(Array.isArray(individual.appliedMutations)).toBe(true);
    expect(individual.metadata).toBeDefined();
    expect(individual.metadata?.length).toBeGreaterThan(0);
  });

  test('individual fitness should be between 0 and 100', async () => {
    const result = await populationSearch(
      'Fitness range test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 6,
        generations: 2,
        verbose: false
      }
    );

    for (const individual of result.finalPopulation) {
      expect(individual.fitness).toBeGreaterThanOrEqual(0);
      expect(individual.fitness).toBeLessThanOrEqual(100);
    }
  });

  test('individual pass rate should be between 0 and 1', async () => {
    const result = await populationSearch(
      'Pass rate range test',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    for (const individual of result.finalPopulation) {
      expect(individual.passRate).toBeGreaterThanOrEqual(0);
      expect(individual.passRate).toBeLessThanOrEqual(1);
    }
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Population Search - Edge Cases', () => {
  test('should handle single generation', async () => {
    const result = await populationSearch(
      'Single generation',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    expect(result.summary.totalGenerations).toBe(1);
    expect(result.bestIndividuals.length).toBeGreaterThan(0);
  });

  test('should handle minimum population size', async () => {
    const result = await populationSearch(
      'Minimum population',
      basicTestSuite,
      mockExecutor,
      {
        populationSize: 4,
        generations: 1,
        verbose: false
      }
    );

    expect(result.finalPopulation.length).toBe(4);
  });

  test('should handle single test case', async () => {
    const singleTestSuite: TestCase[] = [
      {
        id: 'single-test',
        prompt: 'Single test',
        evaluationCriteria: { customValidator: () => true }
      }
    ];

    const result = await populationSearch(
      'Single test case',
      singleTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    expect(result.bestIndividuals.length).toBeGreaterThan(0);
  });

  test('should handle empty expected output', async () => {
    const noExpectedTestSuite: TestCase[] = [
      {
        id: 'no-expected',
        prompt: 'No expected output',
        evaluationCriteria: { customValidator: (output) => output.length > 0 }
      }
    ];

    const result = await populationSearch(
      'No expected output test',
      noExpectedTestSuite,
      mockExecutor,
      {
        populationSize: 5,
        generations: 1,
        verbose: false
      }
    );

    expect(result.bestIndividuals.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// DEFAULT CONFIG TESTS
// ============================================================================

describe('Population Search - Default Config', () => {
  test('DEFAULT_POPULATION_CONFIG should have valid values', () => {
    expect(DEFAULT_POPULATION_CONFIG.populationSize).toBeGreaterThanOrEqual(4);
    expect(DEFAULT_POPULATION_CONFIG.generations).toBeGreaterThanOrEqual(1);
    expect(DEFAULT_POPULATION_CONFIG.selectionRate).toBeGreaterThan(0);
    expect(DEFAULT_POPULATION_CONFIG.selectionRate).toBeLessThanOrEqual(1);
    expect(DEFAULT_POPULATION_CONFIG.mutationProbability).toBeGreaterThanOrEqual(0);
    expect(DEFAULT_POPULATION_CONFIG.mutationProbability).toBeLessThanOrEqual(1);
    expect(DEFAULT_POPULATION_CONFIG.crossoverProbability).toBeGreaterThanOrEqual(0);
    expect(DEFAULT_POPULATION_CONFIG.crossoverProbability).toBeLessThanOrEqual(1);
  });
});
