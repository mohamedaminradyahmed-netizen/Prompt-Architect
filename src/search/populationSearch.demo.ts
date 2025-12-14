/**
 * Population Search Demo (DIRECTIVE-052)
 *
 * يوضح كيفية استخدام نظام Population Search لتحسين البرومبتات
 *
 * Run: npx ts-node src/search/populationSearch.demo.ts
 */

import {
  populationSearch,
  PopulationConfig,
  PopulationSearchResult,
  DEFAULT_POPULATION_CONFIG
} from './populationSearch';
import { TestCase, LLMExecutor } from '../sandbox/testExecutor';

// ============================================================================
// MOCK LLM EXECUTOR
// ============================================================================

/**
 * Mock LLM executor for demonstration
 * In production, replace with actual API calls
 */
const mockLLMExecutor: LLMExecutor = async (prompt: string): Promise<string> => {
  // Simulate API latency
  await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 50));

  // Mock response based on prompt characteristics
  const hasSpecificInstructions = prompt.includes('function') || prompt.includes('code');
  const hasClearStructure = prompt.includes('Step') || prompt.includes('1.') || prompt.includes('Criteria');
  const isWellFormatted = prompt.length > 50 && prompt.length < 500;

  if (hasSpecificInstructions && hasClearStructure && isWellFormatted) {
    return `function sortArray(arr) {
  return arr.slice().sort((a, b) => a - b);
}

// Tests included
// Edge cases handled`;
  }

  if (hasSpecificInstructions) {
    return `function sort(arr) { return arr.sort(); }`;
  }

  return `Here's a solution for your request: ${prompt.substring(0, 50)}...`;
};

// ============================================================================
// TEST SUITE
// ============================================================================

/**
 * Sample test cases for evaluating prompts
 */
const sampleTestSuite: TestCase[] = [
  {
    id: 'test-1',
    prompt: 'Sort [3, 1, 2]',
    expectedOutput: '[1, 2, 3]',
    evaluationCriteria: {
      matchType: 'includes',
      matchValue: '1, 2, 3'
    }
  },
  {
    id: 'test-2',
    prompt: 'Sort empty array []',
    evaluationCriteria: {
      customValidator: (output) =>
        output.includes('[]') || output.includes('empty')
    }
  },
  {
    id: 'test-3',
    prompt: 'Sort [5, 5, 5]',
    evaluationCriteria: {
      customValidator: (output) =>
        output.includes('5') && output.includes('function')
    }
  },
  {
    id: 'test-4',
    prompt: 'Sort [-1, 0, 1]',
    evaluationCriteria: {
      matchType: 'includes',
      matchValue: '-1'
    }
  },
  {
    id: 'test-5',
    prompt: 'What is the time complexity?',
    evaluationCriteria: {
      customValidator: (output) =>
        output.toLowerCase().includes('o(') ||
        output.toLowerCase().includes('complexity') ||
        output.toLowerCase().includes('sort')
    }
  }
];

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

/**
 * Demo 1: Basic population search
 */
async function basicDemo(): Promise<void> {
  console.log('='.repeat(60));
  console.log('DEMO 1: Basic Population Search');
  console.log('='.repeat(60));

  const initialPrompt = 'Write a function to sort an array';

  const config: Partial<PopulationConfig> = {
    populationSize: 10,
    generations: 3,
    selectionRate: 0.5,
    mutationProbability: 0.3,
    crossoverProbability: 0.7,
    verbose: true
  };

  const result = await populationSearch(
    initialPrompt,
    sampleTestSuite,
    mockLLMExecutor,
    config
  );

  printResults(result);
}

/**
 * Demo 2: Custom configuration
 */
async function customConfigDemo(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('DEMO 2: Custom Configuration');
  console.log('='.repeat(60));

  const initialPrompt = 'Implement a sorting algorithm';

  const config: Partial<PopulationConfig> = {
    populationSize: 15,
    generations: 5,
    selectionRate: 0.4,     // Keep top 40%
    mutationProbability: 0.5, // Higher mutation rate
    crossoverProbability: 0.6,
    elitismCount: 3,        // Preserve top 3
    maxConcurrency: 3,
    verbose: true
  };

  const result = await populationSearch(
    initialPrompt,
    sampleTestSuite,
    mockLLMExecutor,
    config
  );

  printResults(result);
}

/**
 * Demo 3: Analyzing generation progression
 */
async function generationAnalysisDemo(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('DEMO 3: Generation Analysis');
  console.log('='.repeat(60));

  const initialPrompt = 'Create a function to order numbers';

  const config: Partial<PopulationConfig> = {
    populationSize: 20,
    generations: 4,
    verbose: false
  };

  const result = await populationSearch(
    initialPrompt,
    sampleTestSuite,
    mockLLMExecutor,
    config
  );

  console.log('\n📊 Generation Progression:');
  console.log('-'.repeat(60));
  console.log('Gen\tBest\tAvg\tWorst\tDiversity');
  console.log('-'.repeat(60));

  for (const stats of result.generationHistory) {
    console.log(
      `${stats.generation}\t` +
      `${stats.bestFitness.toFixed(1)}\t` +
      `${stats.avgFitness.toFixed(1)}\t` +
      `${stats.worstFitness.toFixed(1)}\t` +
      `${stats.diversity.toFixed(1)}`
    );
  }

  console.log('\n📈 Summary:');
  console.log(`   Improvement: ${result.summary.improvementPercent.toFixed(1)}%`);
  console.log(`   Total Evaluations: ${result.summary.totalEvaluations}`);
  console.log(`   Execution Time: ${result.summary.executionTimeMs}ms`);
}

/**
 * Print results in a formatted way
 */
function printResults(result: PopulationSearchResult): void {
  console.log('\n📋 Results Summary:');
  console.log('-'.repeat(40));
  console.log(`Total Generations: ${result.summary.totalGenerations}`);
  console.log(`Total Evaluations: ${result.summary.totalEvaluations}`);
  console.log(`Best Fitness: ${result.summary.bestFitness.toFixed(2)}`);
  console.log(`Improvement: ${result.summary.improvementPercent.toFixed(1)}%`);
  console.log(`Execution Time: ${result.summary.executionTimeMs}ms`);

  console.log('\n🏆 Top 3 Prompts:');
  console.log('-'.repeat(40));

  for (let i = 0; i < Math.min(3, result.bestIndividuals.length); i++) {
    const ind = result.bestIndividuals[i];
    console.log(`\n#${i + 1} (Fitness: ${ind.fitness.toFixed(2)}, Pass Rate: ${(ind.passRate * 100).toFixed(0)}%)`);
    console.log(`   Mutations: ${ind.appliedMutations.join(' → ')}`);
    console.log(`   Prompt: "${ind.prompt.substring(0, 100)}${ind.prompt.length > 100 ? '...' : ''}"`);
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main(): Promise<void> {
  console.log('🧬 Population Search Demo\n');
  console.log('This demo showcases the population-based search algorithm');
  console.log('for optimizing prompts using evolutionary principles.\n');

  try {
    await basicDemo();
    await customConfigDemo();
    await generationAnalysisDemo();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All demos completed successfully!');
    console.log('='.repeat(60));
  } catch (error) {
    console.error('❌ Demo failed:', error);
    process.exit(1);
  }
}

// Run if called directly
main().catch(console.error);

export { basicDemo, customConfigDemo, generationAnalysisDemo };
