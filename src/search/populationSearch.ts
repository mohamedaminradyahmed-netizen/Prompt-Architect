/**
 * Population Search (DIRECTIVE-052)
 *
 * نظام بحث قائم على السكان (Population-Based Search)
 *
 * الخطوات:
 * 1. Initialize: ولّد 20-50 variation عشوائية
 * 2. Evaluate: قيّم كل واحد على test suite
 * 3. Select: اختر أفضل 50%
 * 4. Evolve: طبّق mutations على المختارين
 * 5. Repeat: كرر لعدة أجيال
 *
 * الهدف: إيجاد variations متنوعة وعالية الجودة
 */

import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  PromptVariation
} from '../mutations';
import {
  TestCase,
  TestResults,
  LLMExecutor,
  executeTestSuite
} from '../sandbox/testExecutor';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

/**
 * Configuration for population search
 */
export interface PopulationConfig {
  /** Number of individuals in the population (20-50) */
  populationSize: number;

  /** Number of generations to evolve (5-10) */
  generations: number;

  /** Selection rate - proportion of top individuals to keep (e.g., 0.5 = top 50%) */
  selectionRate: number;

  /** Probability of applying mutation to an individual (0-1) */
  mutationProbability: number;

  /** Probability of crossover between two individuals (0-1) */
  crossoverProbability: number;

  /** Number of elite individuals to preserve unchanged */
  elitismCount?: number;

  /** Maximum concurrent evaluations */
  maxConcurrency?: number;

  /** Enable logging */
  verbose?: boolean;
}

/**
 * Individual in the population
 */
export interface PopulationIndividual {
  /** Unique identifier */
  id: string;

  /** The prompt text */
  prompt: string;

  /** Fitness score (0-100) */
  fitness: number;

  /** Pass rate from test suite (0-1) */
  passRate: number;

  /** Generation this individual was created in */
  generation: number;

  /** Mutations applied to create this individual */
  appliedMutations: string[];

  /** Parent IDs (for tracking lineage) */
  parentIds?: string[];

  /** Test results for this individual */
  testResults?: TestResults;

  /** Metadata */
  metadata?: {
    length: number;
    avgLatency?: number;
    diversityScore?: number;
  };
}

/**
 * Statistics for a generation
 */
export interface GenerationStatistics {
  /** Generation number */
  generation: number;

  /** Best fitness in this generation */
  bestFitness: number;

  /** Average fitness */
  avgFitness: number;

  /** Worst fitness */
  worstFitness: number;

  /** Standard deviation */
  stdDeviation: number;

  /** Population diversity */
  diversity: number;

  /** Best individual in this generation */
  bestIndividual: PopulationIndividual;

  /** Number of individuals evaluated */
  populationSize: number;

  /** Total tests executed */
  testsExecuted: number;
}

/**
 * Result of population search
 */
export interface PopulationSearchResult {
  /** Top individuals (best prompts) */
  bestIndividuals: PopulationIndividual[];

  /** Final population after all generations */
  finalPopulation: PopulationIndividual[];

  /** History of all generations */
  generationHistory: GenerationStatistics[];

  /** Summary statistics */
  summary: {
    totalGenerations: number;
    totalEvaluations: number;
    totalTestsRun: number;
    bestFitness: number;
    bestPrompt: string;
    improvementPercent: number;
    convergenceGeneration?: number;
    executionTimeMs: number;
  };
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

export const DEFAULT_POPULATION_CONFIG: PopulationConfig = {
  populationSize: 30,
  generations: 7,
  selectionRate: 0.5,
  mutationProbability: 0.3,
  crossoverProbability: 0.7,
  elitismCount: 2,
  maxConcurrency: 5,
  verbose: true
};

// ============================================================================
// MAIN POPULATION SEARCH FUNCTION
// ============================================================================

/**
 * Execute population-based search for optimal prompts
 *
 * @param initialPrompt - Starting prompt to optimize
 * @param testSuite - Test cases to evaluate prompts against
 * @param executor - LLM executor function
 * @param config - Population search configuration
 * @returns Population search results with best prompts
 *
 * @example
 * ```typescript
 * const result = await populationSearch(
 *   "Write a function to sort an array",
 *   testCases,
 *   llmExecutor,
 *   { populationSize: 30, generations: 5 }
 * );
 * console.log(result.bestIndividuals[0].prompt);
 * ```
 */
export async function populationSearch(
  initialPrompt: string,
  testSuite: TestCase[],
  executor: LLMExecutor,
  config: Partial<PopulationConfig> = {}
): Promise<PopulationSearchResult> {
  const startTime = Date.now();
  const fullConfig: PopulationConfig = { ...DEFAULT_POPULATION_CONFIG, ...config };
  const {
    populationSize,
    generations,
    selectionRate,
    mutationProbability,
    crossoverProbability,
    elitismCount = 2,
    maxConcurrency = 5,
    verbose = true
  } = fullConfig;

  const generationHistory: GenerationStatistics[] = [];
  let totalTestsRun = 0;

  // Validate configuration
  validateConfig(fullConfig);

  if (verbose) {
    console.log('🧬 Population Search Started');
    console.log(`   Population Size: ${populationSize}`);
    console.log(`   Generations: ${generations}`);
    console.log(`   Selection Rate: ${selectionRate}`);
    console.log(`   Test Cases: ${testSuite.length}`);
  }

  // Step 1: Initialize population
  if (verbose) console.log('\n📊 Initializing population...');
  let population = await initializePopulation(
    initialPrompt,
    populationSize,
    testSuite,
    executor,
    maxConcurrency
  );
  totalTestsRun += populationSize * testSuite.length;

  // Record initial generation stats
  const initialStats = calculateGenerationStatistics(population, 0, testSuite.length);
  generationHistory.push(initialStats);

  if (verbose) {
    console.log(`   Generation 0: Best=${initialStats.bestFitness.toFixed(2)}, Avg=${initialStats.avgFitness.toFixed(2)}`);
  }

  // Step 2-5: Evolution loop
  for (let gen = 1; gen <= generations; gen++) {
    if (verbose) console.log(`\n🔄 Generation ${gen}/${generations}`);

    // Selection: Keep top performers
    const selectedCount = Math.max(
      Math.floor(population.length * selectionRate),
      elitismCount + 2
    );
    const selected = selectTopIndividuals(population, selectedCount);

    if (verbose) console.log(`   Selected top ${selectedCount} individuals`);

    // Evolution: Create new generation
    const newPopulation = await evolvePopulation(
      selected,
      populationSize,
      gen,
      testSuite,
      executor,
      {
        mutationProbability,
        crossoverProbability,
        elitismCount,
        maxConcurrency
      }
    );

    totalTestsRun += (populationSize - elitismCount) * testSuite.length;
    population = newPopulation;

    // Record generation stats
    const stats = calculateGenerationStatistics(population, gen, testSuite.length);
    generationHistory.push(stats);

    if (verbose) {
      console.log(`   Best=${stats.bestFitness.toFixed(2)}, Avg=${stats.avgFitness.toFixed(2)}, Diversity=${stats.diversity.toFixed(2)}`);
    }

    // Check for convergence
    if (gen >= 3 && hasConverged(generationHistory.slice(-4))) {
      if (verbose) console.log(`✓ Converged at generation ${gen}`);
      break;
    }
  }

  // Sort final population
  population.sort((a, b) => b.fitness - a.fitness);

  // Get top 5 individuals
  const bestIndividuals = population.slice(0, 5);

  // Calculate improvement
  const initialBest = generationHistory[0].bestFitness;
  const finalBest = generationHistory[generationHistory.length - 1].bestFitness;
  const improvementPercent = initialBest > 0
    ? ((finalBest - initialBest) / initialBest) * 100
    : finalBest * 100;

  const executionTimeMs = Date.now() - startTime;

  if (verbose) {
    console.log('\n✅ Population Search Complete');
    console.log(`   Total Time: ${(executionTimeMs / 1000).toFixed(2)}s`);
    console.log(`   Improvement: ${improvementPercent.toFixed(1)}%`);
    console.log(`   Best Fitness: ${finalBest.toFixed(2)}`);
  }

  return {
    bestIndividuals,
    finalPopulation: population,
    generationHistory,
    summary: {
      totalGenerations: generationHistory.length - 1,
      totalEvaluations: generationHistory.length * populationSize,
      totalTestsRun,
      bestFitness: finalBest,
      bestPrompt: bestIndividuals[0]?.prompt || initialPrompt,
      improvementPercent,
      convergenceGeneration: findConvergenceGeneration(generationHistory),
      executionTimeMs
    }
  };
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize population with diverse variations
 */
async function initializePopulation(
  originalPrompt: string,
  size: number,
  testSuite: TestCase[],
  executor: LLMExecutor,
  maxConcurrency: number
): Promise<PopulationIndividual[]> {
  const variations: { prompt: string; mutations: string[] }[] = [];

  // Add original prompt
  variations.push({ prompt: originalPrompt, mutations: ['original'] });

  // Available mutations
  const mutationOps = [
    { name: 'try-catch', fn: tryCatchStyleMutation },
    { name: 'reduce', fn: reduceContextMutation },
    { name: 'expand', fn: expandMutation }
  ];

  // Generate diverse variations
  for (let i = 1; i < size; i++) {
    let currentPrompt = originalPrompt;
    const appliedMutations: string[] = [];

    // Apply 1-3 random mutations
    const numMutations = Math.floor(Math.random() * 3) + 1;

    for (let m = 0; m < numMutations; m++) {
      const mutation = mutationOps[Math.floor(Math.random() * mutationOps.length)];

      // Avoid applying same mutation twice in a row
      if (appliedMutations[appliedMutations.length - 1] !== mutation.name) {
        const result = mutation.fn(currentPrompt);
        currentPrompt = result.text;
        appliedMutations.push(mutation.name);
      }
    }

    // Add some random perturbations for diversity
    if (Math.random() < 0.3) {
      currentPrompt = addRandomPerturbation(currentPrompt);
      appliedMutations.push('perturbation');
    }

    variations.push({ prompt: currentPrompt, mutations: appliedMutations });
  }

  // Evaluate all variations in parallel batches
  const population: PopulationIndividual[] = [];
  const prompts = variations.map(v => v.prompt);

  const results = await executeTestSuite(prompts, testSuite, executor, maxConcurrency);

  for (let i = 0; i < variations.length; i++) {
    const testResult = results[i];
    const fitness = calculateFitness(testResult);

    population.push({
      id: generateId(),
      prompt: variations[i].prompt,
      fitness,
      passRate: testResult.passRate,
      generation: 0,
      appliedMutations: variations[i].mutations,
      testResults: testResult,
      metadata: {
        length: variations[i].prompt.length,
        avgLatency: calculateAvgLatency(testResult)
      }
    });
  }

  return population;
}

// ============================================================================
// SELECTION
// ============================================================================

/**
 * Select top performing individuals
 */
function selectTopIndividuals(
  population: PopulationIndividual[],
  count: number
): PopulationIndividual[] {
  // Sort by fitness descending
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  return sorted.slice(0, count);
}

/**
 * Tournament selection: Pick best from random subset
 */
function tournamentSelect(
  population: PopulationIndividual[],
  tournamentSize: number = 3
): PopulationIndividual {
  const tournament: PopulationIndividual[] = [];

  for (let i = 0; i < tournamentSize; i++) {
    const randomIndex = Math.floor(Math.random() * population.length);
    tournament.push(population[randomIndex]);
  }

  return tournament.reduce((best, current) =>
    current.fitness > best.fitness ? current : best
  );
}

// ============================================================================
// EVOLUTION
// ============================================================================

/**
 * Evolve population for one generation
 */
async function evolvePopulation(
  selected: PopulationIndividual[],
  targetSize: number,
  generation: number,
  testSuite: TestCase[],
  executor: LLMExecutor,
  options: {
    mutationProbability: number;
    crossoverProbability: number;
    elitismCount: number;
    maxConcurrency: number;
  }
): Promise<PopulationIndividual[]> {
  const { mutationProbability, crossoverProbability, elitismCount, maxConcurrency } = options;
  const newPopulation: PopulationIndividual[] = [];

  // Elitism: Keep top individuals unchanged
  const elite = selected.slice(0, elitismCount);
  for (const ind of elite) {
    newPopulation.push({
      ...ind,
      id: generateId(),
      generation
    });
  }

  // Generate offspring to fill the rest
  const offspringVariations: { prompt: string; mutations: string[]; parentIds: string[] }[] = [];

  while (newPopulation.length + offspringVariations.length < targetSize) {
    // Select parents
    const parent1 = tournamentSelect(selected);
    const parent2 = tournamentSelect(selected);

    let offspringPrompt: string;
    let mutations: string[];
    const parentIds = [parent1.id, parent2.id];

    // Crossover
    if (Math.random() < crossoverProbability && parent1.id !== parent2.id) {
      offspringPrompt = crossover(parent1.prompt, parent2.prompt);
      mutations = [...new Set([...parent1.appliedMutations, ...parent2.appliedMutations, 'crossover'])];
    } else {
      offspringPrompt = parent1.prompt;
      mutations = [...parent1.appliedMutations];
    }

    // Mutation
    if (Math.random() < mutationProbability) {
      const mutationResult = applyRandomMutation(offspringPrompt);
      offspringPrompt = mutationResult.prompt;
      mutations.push(mutationResult.mutationType);
    }

    offspringVariations.push({
      prompt: offspringPrompt,
      mutations,
      parentIds
    });
  }

  // Evaluate offspring in parallel
  if (offspringVariations.length > 0) {
    const prompts = offspringVariations.map(v => v.prompt);
    const results = await executeTestSuite(prompts, testSuite, executor, maxConcurrency);

    for (let i = 0; i < offspringVariations.length; i++) {
      const testResult = results[i];
      const fitness = calculateFitness(testResult);

      newPopulation.push({
        id: generateId(),
        prompt: offspringVariations[i].prompt,
        fitness,
        passRate: testResult.passRate,
        generation,
        appliedMutations: offspringVariations[i].mutations,
        parentIds: offspringVariations[i].parentIds,
        testResults: testResult,
        metadata: {
          length: offspringVariations[i].prompt.length,
          avgLatency: calculateAvgLatency(testResult)
        }
      });
    }
  }

  return newPopulation;
}

/**
 * Crossover two prompts
 */
function crossover(prompt1: string, prompt2: string): string {
  const sentences1 = splitIntoSentences(prompt1);
  const sentences2 = splitIntoSentences(prompt2);

  if (sentences1.length < 2 || sentences2.length < 2) {
    // If either prompt is too short, do simple merge
    return `${prompt1} ${prompt2}`.trim();
  }

  // Single-point crossover
  const crossPoint1 = Math.floor(Math.random() * sentences1.length);
  const crossPoint2 = Math.floor(Math.random() * sentences2.length);

  const offspring = [
    ...sentences1.slice(0, crossPoint1),
    ...sentences2.slice(crossPoint2)
  ];

  return offspring.join('. ').trim();
}

/**
 * Apply random mutation to a prompt
 */
function applyRandomMutation(prompt: string): { prompt: string; mutationType: string } {
  const mutationOps = [
    { name: 'try-catch', fn: tryCatchStyleMutation },
    { name: 'reduce', fn: reduceContextMutation },
    { name: 'expand', fn: expandMutation },
    { name: 'word-swap', fn: wordSwapMutation },
    { name: 'sentence-reorder', fn: sentenceReorderMutation }
  ];

  const mutation = mutationOps[Math.floor(Math.random() * mutationOps.length)];
  const result = mutation.fn(prompt);

  return {
    prompt: result.text,
    mutationType: mutation.name
  };
}

/**
 * Word swap mutation
 */
function wordSwapMutation(prompt: string): PromptVariation {
  const words = prompt.split(/\s+/);

  if (words.length < 4) {
    return {
      text: prompt,
      mutationType: 'expansion',
      changeDescription: 'No swap (prompt too short)',
      expectedImpact: {}
    };
  }

  const idx1 = Math.floor(Math.random() * words.length);
  let idx2 = Math.floor(Math.random() * words.length);
  while (idx2 === idx1) {
    idx2 = Math.floor(Math.random() * words.length);
  }

  [words[idx1], words[idx2]] = [words[idx2], words[idx1]];

  return {
    text: words.join(' '),
    mutationType: 'expansion',
    changeDescription: 'Swapped word positions',
    expectedImpact: { quality: 'neutral' }
  };
}

/**
 * Sentence reorder mutation
 */
function sentenceReorderMutation(prompt: string): PromptVariation {
  const sentences = splitIntoSentences(prompt);

  if (sentences.length < 2) {
    return {
      text: prompt,
      mutationType: 'expansion',
      changeDescription: 'No reorder (single sentence)',
      expectedImpact: {}
    };
  }

  // Fisher-Yates shuffle
  for (let i = sentences.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [sentences[i], sentences[j]] = [sentences[j], sentences[i]];
  }

  return {
    text: sentences.join('. '),
    mutationType: 'expansion',
    changeDescription: 'Reordered sentences',
    expectedImpact: { quality: 'neutral' }
  };
}

// ============================================================================
// STATISTICS & ANALYSIS
// ============================================================================

/**
 * Calculate fitness from test results
 */
function calculateFitness(testResult: TestResults): number {
  // Combine pass rate and aggregate score
  const passRateWeight = 0.6;
  const scoreWeight = 0.4;

  return (testResult.passRate * passRateWeight + testResult.aggregateScore * scoreWeight) * 100;
}

/**
 * Calculate average latency from test results
 */
function calculateAvgLatency(testResult: TestResults): number {
  if (!testResult.results || testResult.results.length === 0) return 0;

  const totalLatency = testResult.results.reduce((sum, r) => sum + r.latency, 0);
  return totalLatency / testResult.results.length;
}

/**
 * Calculate generation statistics
 */
function calculateGenerationStatistics(
  population: PopulationIndividual[],
  generation: number,
  testCasesCount: number
): GenerationStatistics {
  const fitnesses = population.map(ind => ind.fitness);

  const best = Math.max(...fitnesses);
  const worst = Math.min(...fitnesses);
  const avg = fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length;

  // Standard deviation
  const variance = fitnesses.reduce((sum, f) => sum + Math.pow(f - avg, 2), 0) / fitnesses.length;
  const stdDev = Math.sqrt(variance);

  // Diversity
  const diversity = calculateDiversity(population);

  const bestIndividual = population.reduce((best, current) =>
    current.fitness > best.fitness ? current : best
  );

  return {
    generation,
    bestFitness: best,
    avgFitness: avg,
    worstFitness: worst,
    stdDeviation: stdDev,
    diversity,
    bestIndividual,
    populationSize: population.length,
    testsExecuted: population.length * testCasesCount
  };
}

/**
 * Calculate population diversity using Levenshtein distance
 */
function calculateDiversity(population: PopulationIndividual[]): number {
  if (population.length < 2) return 0;

  let totalDistance = 0;
  let comparisons = 0;

  // Sample-based diversity for large populations
  const sampleSize = Math.min(population.length, 10);
  const samples = population.slice(0, sampleSize);

  for (let i = 0; i < samples.length; i++) {
    for (let j = i + 1; j < samples.length; j++) {
      const distance = levenshteinDistance(
        samples[i].prompt.substring(0, 200),
        samples[j].prompt.substring(0, 200)
      );
      totalDistance += distance;
      comparisons++;
    }
  }

  return comparisons > 0 ? totalDistance / comparisons : 0;
}

/**
 * Check if population has converged
 */
function hasConverged(recentHistory: GenerationStatistics[]): boolean {
  if (recentHistory.length < 4) return false;

  const improvements: number[] = [];
  for (let i = 1; i < recentHistory.length; i++) {
    const improvement = recentHistory[i].bestFitness - recentHistory[i - 1].bestFitness;
    improvements.push(improvement);
  }

  const avgImprovement = improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length;
  return Math.abs(avgImprovement) < 0.5; // Less than 0.5% improvement
}

/**
 * Find the generation where convergence occurred
 */
function findConvergenceGeneration(history: GenerationStatistics[]): number | undefined {
  for (let i = 3; i < history.length; i++) {
    const recent = history.slice(i - 3, i + 1);
    if (hasConverged(recent)) {
      return i;
    }
  }
  return undefined;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Validate configuration
 */
function validateConfig(config: PopulationConfig): void {
  if (config.populationSize < 4) {
    throw new Error('Population size must be at least 4');
  }
  if (config.generations < 1) {
    throw new Error('Generations must be at least 1');
  }
  if (config.selectionRate <= 0 || config.selectionRate > 1) {
    throw new Error('Selection rate must be between 0 and 1');
  }
  if (config.mutationProbability < 0 || config.mutationProbability > 1) {
    throw new Error('Mutation probability must be between 0 and 1');
  }
  if (config.crossoverProbability < 0 || config.crossoverProbability > 1) {
    throw new Error('Crossover probability must be between 0 and 1');
  }
}

/**
 * Split text into sentences
 */
function splitIntoSentences(text: string): string[] {
  return text
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

/**
 * Add random perturbation to prompt
 */
function addRandomPerturbation(prompt: string): string {
  const perturbations = [
    (p: string) => p.replace(/\./g, '. ').replace(/\s+/g, ' ').trim(),
    (p: string) => p.charAt(0).toUpperCase() + p.slice(1),
    (p: string) => p.endsWith('.') ? p : p + '.',
    (p: string) => p.replace(/,\s*/g, ', '),
    (p: string) => p.replace(/\s+/g, ' ')
  ];

  const perturbation = perturbations[Math.floor(Math.random() * perturbations.length)];
  return perturbation(prompt);
}

/**
 * Calculate Levenshtein distance between two strings
 */
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;

  if (len1 === 0) return len2;
  if (len2 === 0) return len1;

  const matrix: number[][] = [];

  for (let i = 0; i <= len1; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= len2; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }

  return matrix[len1][len2];
}

/**
 * Generate unique ID
 */
function generateId(): string {
  return `pop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  populationSearch,
  DEFAULT_POPULATION_CONFIG
};
