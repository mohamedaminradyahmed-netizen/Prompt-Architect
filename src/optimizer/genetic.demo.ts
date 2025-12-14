/**
 * Genetic Algorithm Optimizer - Demo & Examples
 * DIRECTIVE-020 Implementation
 */

import { geneticOptimize, DEFAULT_GENETIC_CONFIG, FitnessFunction } from './genetic';
import { validateMetrics, BALANCED } from '../config/balanceMetrics';

// ============================================================================
// EXAMPLE FITNESS FUNCTIONS
// ============================================================================

/**
 * Simple fitness function based on prompt length and keywords
 */
function simpleFitnessFunction(prompt: string): number {
  let score = 50; // Base score

  // Prefer moderate length (not too short, not too long)
  const idealLength = 200;
  const lengthPenalty = Math.abs(prompt.length - idealLength) / idealLength;
  score -= lengthPenalty * 20;

  // Reward specific keywords
  const keywords = ['try', 'create', 'implement', 'function', 'test', 'error'];
  const keywordCount = keywords.filter(kw => prompt.toLowerCase().includes(kw)).length;
  score += keywordCount * 5;

  // Reward having multiple sentences
  const sentences = prompt.split(/[.!?]+/).filter(s => s.trim().length > 0);
  score += Math.min(sentences.length, 5) * 3;

  return Math.max(0, Math.min(100, score));
}

/**
 * Advanced fitness function using balance metrics
 */
async function balanceMetricsFitness(prompt: string): Promise<number> {
  // Simulate metrics (in production, these would come from actual LLM calls)
  const simulatedMetrics = {
    quality: 0.7 + Math.random() * 0.2, // 0.7-0.9
    cost: 0.01 + (prompt.length / 1000) * 0.02, // Based on length
    latency: 2000 + (prompt.length / 10) * 100, // Based on length
    hallucinationRate: 0.05 + Math.random() * 0.1, // 0.05-0.15
    similarity: 0.75 + Math.random() * 0.15, // 0.75-0.9
  };

  const validation = validateMetrics(simulatedMetrics, BALANCED);
  return validation.score;
}

/**
 * Custom fitness function example
 */
function customFitness(prompt: string): number {
  let score = 0;

  // Criterion 1: Has clear action verb (20 points)
  const actionVerbs = /\b(create|write|build|implement|develop|design|analyze|optimize)\b/i;
  if (actionVerbs.test(prompt)) {
    score += 20;
  }

  // Criterion 2: Has constraints or requirements (20 points)
  const hasConstraints = /\b(must|should|requirement|constraint|ensure)\b/i;
  if (hasConstraints.test(prompt)) {
    score += 20;
  }

  // Criterion 3: Has examples or specificity (20 points)
  const hasExamples = /\b(example|such as|like|for instance)\b/i;
  if (hasExamples.test(prompt)) {
    score += 20;
  }

  // Criterion 4: Length is reasonable (40 points)
  const length = prompt.length;
  if (length >= 50 && length <= 500) {
    score += 40;
  } else if (length >= 30 && length <= 700) {
    score += 20;
  }

  return score;
}

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

/**
 * Demo 1: Basic Genetic Optimization
 */
async function demo1_BasicOptimization() {
  console.log('='.repeat(80));
  console.log('DEMO 1: Basic Genetic Optimization');
  console.log('='.repeat(80));

  const initialPrompt = 'Write a function to sort numbers';

  console.log(`\nInitial Prompt: "${initialPrompt}"`);
  console.log(`Initial Fitness: ${simpleFitnessFunction(initialPrompt).toFixed(2)}\n`);

  const result = await geneticOptimize(initialPrompt, simpleFitnessFunction, {
    populationSize: 10,
    generations: 5,
    crossoverRate: 0.7,
    mutationRate: 0.3,
    elitismCount: 2,
  });

  console.log('\n' + '='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));

  console.log(`\n📊 Summary:`);
  console.log(`  Total Generations: ${result.summary.totalGenerations}`);
  console.log(`  Total Evaluations: ${result.summary.totalEvaluations}`);
  console.log(`  Best Fitness: ${result.summary.bestOverallFitness.toFixed(2)}`);
  console.log(`  Improvement: ${result.summary.improvementPercent.toFixed(2)}%`);
  if (result.summary.convergenceGeneration) {
    console.log(`  Converged at Generation: ${result.summary.convergenceGeneration}`);
  }

  console.log(`\n🏆 Top 3 Prompts:`);
  result.bestPrompts.slice(0, 3).forEach((ind, idx) => {
    console.log(`\n  ${idx + 1}. Fitness: ${ind.fitness.toFixed(2)}`);
    console.log(`     Mutations: ${ind.mutations.join(' → ')}`);
    console.log(`     Prompt: "${ind.prompt.substring(0, 100)}${ind.prompt.length > 100 ? '...' : ''}"`);
  });

  console.log(`\n📈 Generation Progress:`);
  result.generationHistory.forEach(gen => {
    const bar = '█'.repeat(Math.floor(gen.bestFitness / 5));
    console.log(`  Gen ${gen.generation.toString().padStart(2)}: ${bar} ${gen.bestFitness.toFixed(2)}`);
  });
}

/**
 * Demo 2: Advanced Fitness with Balance Metrics
 */
async function demo2_AdvancedFitness() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 2: Optimization with Balance Metrics');
  console.log('='.repeat(80));

  const initialPrompt = 'Create an authentication system';

  console.log(`\nInitial Prompt: "${initialPrompt}"`);

  const result = await geneticOptimize(initialPrompt, balanceMetricsFitness, {
    populationSize: 15,
    generations: 8,
    crossoverRate: 0.8,
    mutationRate: 0.25,
    elitismCount: 3,
    selectionStrategy: 'tournament',
    tournamentSize: 4,
  });

  console.log(`\n🏆 Best Prompt Found:`);
  console.log(`   Fitness: ${result.bestPrompts[0].fitness.toFixed(2)}/100`);
  console.log(`   Generation: ${result.bestPrompts[0].generation}`);
  console.log(`   Mutations Applied: ${result.bestPrompts[0].mutations.join(' → ')}`);
  console.log(`\n   Prompt Text:`);
  console.log(`   "${result.summary.bestPrompt}"`);

  console.log(`\n📊 Diversity Over Time:`);
  result.generationHistory.forEach(gen => {
    console.log(`   Gen ${gen.generation}: Diversity=${gen.diversity.toFixed(2)}, StdDev=${gen.stdDevFitness.toFixed(2)}`);
  });
}

/**
 * Demo 3: Comparison of Selection Strategies
 */
async function demo3_SelectionStrategies() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 3: Comparing Selection Strategies');
  console.log('='.repeat(80));

  const initialPrompt = 'Optimize database queries for performance';

  const strategies: Array<'tournament' | 'roulette' | 'rank'> = ['tournament', 'roulette', 'rank'];

  for (const strategy of strategies) {
    console.log(`\n--- Testing ${strategy.toUpperCase()} Selection ---`);

    const result = await geneticOptimize(initialPrompt, customFitness, {
      populationSize: 12,
      generations: 6,
      selectionStrategy: strategy,
      crossoverRate: 0.7,
      mutationRate: 0.3,
      elitismCount: 2,
    });

    console.log(`   Final Best Fitness: ${result.summary.bestOverallFitness.toFixed(2)}`);
    console.log(`   Improvement: ${result.summary.improvementPercent.toFixed(2)}%`);
    console.log(`   Generations to Converge: ${result.summary.convergenceGeneration || 'N/A'}`);
  }
}

/**
 * Demo 4: Parameter Tuning
 */
async function demo4_ParameterTuning() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 4: Impact of Population Size');
  console.log('='.repeat(80));

  const initialPrompt = 'Build a REST API with authentication';
  const populationSizes = [5, 10, 20, 30];

  console.log('\nTesting different population sizes:\n');

  for (const popSize of populationSizes) {
    const startTime = Date.now();

    const result = await geneticOptimize(initialPrompt, simpleFitnessFunction, {
      populationSize: popSize,
      generations: 5,
      crossoverRate: 0.7,
      mutationRate: 0.3,
      elitismCount: Math.min(2, Math.floor(popSize / 10)),
    });

    const duration = Date.now() - startTime;

    console.log(`Population Size: ${popSize}`);
    console.log(`  Best Fitness: ${result.summary.bestOverallFitness.toFixed(2)}`);
    console.log(`  Time: ${duration}ms`);
    console.log(`  Evaluations: ${result.summary.totalEvaluations}`);
    console.log();
  }
}

/**
 * Demo 5: Real-World Example - Code Generation Prompt
 */
async function demo5_RealWorldExample() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 5: Real-World Code Generation Prompt Optimization');
  console.log('='.repeat(80));

  const initialPrompt = 'Write code for user login';

  console.log(`\nOptimizing: "${initialPrompt}"`);
  console.log('Target: Clear, specific, testable code generation prompt\n');

  // Custom fitness for code generation prompts
  const codeGenFitness = (prompt: string): number => {
    let score = 0;

    // Must mention implementation details (30 points)
    if (/\b(function|class|method|component)\b/i.test(prompt)) score += 30;

    // Should specify language or framework (20 points)
    if (/\b(typescript|javascript|python|react|node)\b/i.test(prompt)) score += 20;

    // Should include requirements (20 points)
    if (/\b(should|must|need|require)\b/i.test(prompt)) score += 20;

    // Should mention error handling or edge cases (15 points)
    if (/\b(error|exception|edge case|validation)\b/i.test(prompt)) score += 15;

    // Should be detailed but not too verbose (15 points)
    const length = prompt.length;
    if (length >= 100 && length <= 400) {
      score += 15;
    } else if (length >= 50 && length <= 600) {
      score += 8;
    }

    return score;
  };

  const result = await geneticOptimize(initialPrompt, codeGenFitness, {
    populationSize: 20,
    generations: 10,
    crossoverRate: 0.75,
    mutationRate: 0.35,
    elitismCount: 3,
    selectionStrategy: 'tournament',
    tournamentSize: 3,
  });

  console.log(`\n✨ Original Prompt:`);
  console.log(`   "${initialPrompt}"`);
  console.log(`   Fitness: ${codeGenFitness(initialPrompt)}/100`);

  console.log(`\n🎯 Optimized Prompt:`);
  console.log(`   "${result.summary.bestPrompt}"`);
  console.log(`   Fitness: ${result.summary.bestOverallFitness.toFixed(2)}/100`);
  console.log(`   Improvement: ${result.summary.improvementPercent.toFixed(2)}%`);

  console.log(`\n🧬 Evolution Path:`);
  console.log(`   Mutations Applied: ${result.bestPrompts[0].mutations.join(' → ')}`);
  console.log(`   Generation Created: ${result.bestPrompts[0].generation}`);

  console.log(`\n📊 Alternative Top Prompts:`);
  result.bestPrompts.slice(1, 4).forEach((ind, idx) => {
    console.log(`\n   ${idx + 2}. Fitness: ${ind.fitness.toFixed(2)}`);
    console.log(`      "${ind.prompt.substring(0, 80)}${ind.prompt.length > 80 ? '...' : ''}"`);
  });
}

// ============================================================================
// RUN ALL DEMOS
// ============================================================================

async function runAllDemos() {
  console.log('\n');
  console.log('╔' + '═'.repeat(78) + '╗');
  console.log('║' + ' '.repeat(20) + 'GENETIC ALGORITHM OPTIMIZER' + ' '.repeat(30) + '║');
  console.log('║' + ' '.repeat(25) + 'DIRECTIVE-020 Demo' + ' '.repeat(33) + '║');
  console.log('╚' + '═'.repeat(78) + '╝');
  console.log('\n');

  try {
    await demo1_BasicOptimization();
    await demo2_AdvancedFitness();
    await demo3_SelectionStrategies();
    await demo4_ParameterTuning();
    await demo5_RealWorldExample();

    console.log('\n\n' + '='.repeat(80));
    console.log('✅ ALL DEMOS COMPLETED SUCCESSFULLY');
    console.log('='.repeat(80));
    console.log('\n📚 Key Takeaways:');
    console.log('  1. Population size affects both quality and computation time');
    console.log('  2. Different selection strategies have different convergence patterns');
    console.log('  3. Genetic algorithms excel at exploring diverse solution spaces');
    console.log('  4. Elitism preserves best solutions while allowing exploration');
    console.log('  5. Crossover combines successful patterns from different prompts');
    console.log('\n');
  } catch (error) {
    console.error('❌ Error running demos:', error);
  }
}

// Run demos if this file is executed directly
if (require.main === module) {
  runAllDemos();
}

export {
  simpleFitnessFunction,
  balanceMetricsFitness,
  customFitness,
  runAllDemos,
};
