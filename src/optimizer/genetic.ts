/**
 * Genetic Algorithm Optimizer (DIRECTIVE-020)
 *
 * Population-based optimization using evolutionary principles:
 * 1. Initialize: Create diverse population of prompt variations
 * 2. Evaluate: Score each individual in the population
 * 3. Select: Choose best performers for breeding
 * 4. Crossover: Combine successful prompts to create offspring
 * 5. Mutate: Apply random mutations for diversity
 * 6. Repeat: Evolve over multiple generations
 */

import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  PromptVariation
} from '../mutations';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

/**
 * Configuration for genetic algorithm
 */
export interface GeneticConfig {
  /** Number of individuals in each generation */
  populationSize: number;

  /** Number of generations to evolve */
  generations: number;

  /** Probability of crossover (0-1) */
  crossoverRate: number;

  /** Probability of mutation (0-1) */
  mutationRate: number;

  /** Number of top individuals to preserve unchanged (elitism) */
  elitismCount: number;

  /** Selection strategy */
  selectionStrategy?: 'tournament' | 'roulette' | 'rank';

  /** Tournament size (if using tournament selection) */
  tournamentSize?: number;
}

/**
 * Individual in the population (a prompt variation)
 */
export interface Individual {
  /** Unique identifier */
  id: string;

  /** The prompt text */
  prompt: string;

  /** Fitness score (0-100) */
  fitness: number;

  /** Generation number this individual was created in */
  generation: number;

  /** Parent IDs (for tracking lineage) */
  parents?: [string, string];

  /** Mutations applied to create this individual */
  mutations: string[];

  /** Additional metadata */
  metadata?: {
    length: number;
    originalPrompt?: string;
    diversityScore?: number;
  };
}

/**
 * Statistics for a generation
 */
export interface GenerationStats {
  /** Generation number */
  generation: number;

  /** Best fitness in this generation */
  bestFitness: number;

  /** Average fitness */
  avgFitness: number;

  /** Worst fitness */
  worstFitness: number;

  /** Standard deviation of fitness */
  stdDevFitness: number;

  /** Diversity metric (average distance between individuals) */
  diversity: number;

  /** Best individual */
  bestIndividual: Individual;
}

/**
 * Result of genetic optimization
 */
export interface PopulationResult {
  /** Top 5 best prompts */
  bestPrompts: Individual[];

  /** All final scores */
  finalPopulation: Individual[];

  /** History of all generations */
  generationHistory: GenerationStats[];

  /** Summary statistics */
  summary: {
    totalGenerations: number;
    totalEvaluations: number;
    bestOverallFitness: number;
    bestPrompt: string;
    improvementPercent: number;
    convergenceGeneration?: number;
  };
}

/**
 * Fitness function signature
 */
export type FitnessFunction = (prompt: string) => Promise<number> | number;

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

export const DEFAULT_GENETIC_CONFIG: GeneticConfig = {
  populationSize: 20,
  generations: 10,
  crossoverRate: 0.7,
  mutationRate: 0.3,
  elitismCount: 2,
  selectionStrategy: 'tournament',
  tournamentSize: 3,
};

// ============================================================================
// MAIN GENETIC ALGORITHM
// ============================================================================

/**
 * Run genetic algorithm optimization
 *
 * @param initialPrompt - Starting prompt to optimize
 * @param fitnessFunction - Function to evaluate prompt quality
 * @param config - Genetic algorithm configuration
 * @returns Optimization results with best prompts
 */
export async function geneticOptimize(
  initialPrompt: string,
  fitnessFunction: FitnessFunction,
  config: Partial<GeneticConfig> = {}
): Promise<PopulationResult> {
  const fullConfig: GeneticConfig = { ...DEFAULT_GENETIC_CONFIG, ...config };

  // Track all generations
  const generationHistory: GenerationStats[] = [];
  let population: Individual[] = [];

  // Step 1: Initialize population
  console.log(`🧬 Initializing population (${fullConfig.populationSize} individuals)...`);
  population = await initializePopulation(initialPrompt, fullConfig.populationSize, fitnessFunction);

  const initialStats = calculateGenerationStats(population, 0);
  generationHistory.push(initialStats);
  console.log(`Generation 0: Best=${initialStats.bestFitness.toFixed(2)}, Avg=${initialStats.avgFitness.toFixed(2)}, Diversity=${initialStats.diversity.toFixed(2)}`);

  // Step 2-6: Evolve over generations
  for (let gen = 1; gen <= fullConfig.generations; gen++) {
    console.log(`\n🔄 Generation ${gen}/${fullConfig.generations}`);

    // Create next generation
    const newPopulation = await evolveGeneration(
      population,
      fullConfig,
      fitnessFunction,
      gen
    );

    population = newPopulation;

    // Calculate and store statistics
    const stats = calculateGenerationStats(population, gen);
    generationHistory.push(stats);

    console.log(`   Best=${stats.bestFitness.toFixed(2)}, Avg=${stats.avgFitness.toFixed(2)}, Diversity=${stats.diversity.toFixed(2)}`);

    // Check for convergence (optional early stopping)
    if (gen > 3 && hasConverged(generationHistory.slice(-4))) {
      console.log(`✓ Converged at generation ${gen}`);
      break;
    }
  }

  // Sort final population by fitness
  population.sort((a, b) => b.fitness - a.fitness);

  // Get top 5 prompts
  const bestPrompts = population.slice(0, 5);

  // Calculate improvement
  const initialBest = generationHistory[0].bestFitness;
  const finalBest = generationHistory[generationHistory.length - 1].bestFitness;
  const improvementPercent = ((finalBest - initialBest) / initialBest) * 100;

  // Find convergence generation
  const convergenceGeneration = findConvergenceGeneration(generationHistory);

  return {
    bestPrompts,
    finalPopulation: population,
    generationHistory,
    summary: {
      totalGenerations: generationHistory.length - 1,
      totalEvaluations: generationHistory.length * fullConfig.populationSize,
      bestOverallFitness: finalBest,
      bestPrompt: bestPrompts[0].prompt,
      improvementPercent,
      convergenceGeneration,
    },
  };
}

// ============================================================================
// POPULATION INITIALIZATION
// ============================================================================

/**
 * Create initial population with diverse variations
 */
async function initializePopulation(
  originalPrompt: string,
  size: number,
  fitnessFunction: FitnessFunction
): Promise<Individual[]> {
  const population: Individual[] = [];

  // Add original prompt as first individual
  const originalFitness = await evaluateFitness(originalPrompt, fitnessFunction);
  population.push({
    id: generateId(),
    prompt: originalPrompt,
    fitness: originalFitness,
    generation: 0,
    mutations: ['original'],
    metadata: {
      length: originalPrompt.length,
      originalPrompt,
    },
  });

  // Generate diverse variations
  const mutationTypes = [
    { name: 'try-catch', fn: tryCatchStyleMutation },
    { name: 'reduce', fn: reduceContextMutation },
    { name: 'expand', fn: expandMutation },
  ];

  // Create variations using different mutation combinations
  for (let i = 1; i < size; i++) {
    let currentPrompt = originalPrompt;
    const appliedMutations: string[] = [];

    // Randomly apply 1-3 mutations
    const numMutations = Math.floor(Math.random() * 3) + 1;

    for (let m = 0; m < numMutations; m++) {
      const mutation = mutationTypes[Math.floor(Math.random() * mutationTypes.length)];
      const variation = mutation.fn(currentPrompt);
      currentPrompt = variation.text;
      appliedMutations.push(mutation.name);
    }

    // Evaluate fitness
    const fitness = await evaluateFitness(currentPrompt, fitnessFunction);

    population.push({
      id: generateId(),
      prompt: currentPrompt,
      fitness,
      generation: 0,
      mutations: appliedMutations,
      metadata: {
        length: currentPrompt.length,
        originalPrompt,
      },
    });
  }

  return population;
}

// ============================================================================
// EVOLUTION OPERATORS
// ============================================================================

/**
 * Evolve population for one generation
 */
async function evolveGeneration(
  population: Individual[],
  config: GeneticConfig,
  fitnessFunction: FitnessFunction,
  generation: number
): Promise<Individual[]> {
  const newPopulation: Individual[] = [];

  // Elitism: Preserve top individuals
  const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
  const elite = sorted.slice(0, config.elitismCount);
  newPopulation.push(...elite.map(ind => ({ ...ind, generation })));

  // Fill rest of population through selection, crossover, and mutation
  while (newPopulation.length < config.populationSize) {
    // Selection
    const parent1 = select(population, config);
    const parent2 = select(population, config);

    let offspring1: Individual;
    let offspring2: Individual;

    // Crossover
    if (Math.random() < config.crossoverRate) {
      [offspring1, offspring2] = crossover(parent1, parent2, generation);
    } else {
      offspring1 = { ...parent1, id: generateId(), generation };
      offspring2 = { ...parent2, id: generateId(), generation };
    }

    // Mutation
    if (Math.random() < config.mutationRate) {
      offspring1 = await mutate(offspring1, fitnessFunction);
    }
    if (Math.random() < config.mutationRate && newPopulation.length + 1 < config.populationSize) {
      offspring2 = await mutate(offspring2, fitnessFunction);
    }

    // Add to new population
    newPopulation.push(offspring1);
    if (newPopulation.length < config.populationSize) {
      newPopulation.push(offspring2);
    }
  }

  return newPopulation;
}

// ============================================================================
// SELECTION
// ============================================================================

/**
 * Select an individual from population for breeding
 */
function select(population: Individual[], config: GeneticConfig): Individual {
  const strategy = config.selectionStrategy || 'tournament';

  switch (strategy) {
    case 'tournament':
      return tournamentSelection(population, config.tournamentSize || 3);
    case 'roulette':
      return rouletteSelection(population);
    case 'rank':
      return rankSelection(population);
    default:
      return tournamentSelection(population, 3);
  }
}

/**
 * Tournament selection: Pick best from random subset
 */
function tournamentSelection(population: Individual[], tournamentSize: number): Individual {
  const tournament: Individual[] = [];

  for (let i = 0; i < tournamentSize; i++) {
    const randomIndex = Math.floor(Math.random() * population.length);
    tournament.push(population[randomIndex]);
  }

  return tournament.reduce((best, current) =>
    current.fitness > best.fitness ? current : best
  );
}

/**
 * Roulette wheel selection: Probability proportional to fitness
 */
function rouletteSelection(population: Individual[]): Individual {
  const totalFitness = population.reduce((sum, ind) => sum + ind.fitness, 0);
  const random = Math.random() * totalFitness;

  let cumulative = 0;
  for (const individual of population) {
    cumulative += individual.fitness;
    if (cumulative >= random) {
      return individual;
    }
  }

  return population[population.length - 1];
}

/**
 * Rank selection: Probability based on rank
 */
function rankSelection(population: Individual[]): Individual {
  const sorted = [...population].sort((a, b) => a.fitness - b.fitness);
  const totalRank = (population.length * (population.length + 1)) / 2;
  const random = Math.random() * totalRank;

  let cumulative = 0;
  for (let i = 0; i < sorted.length; i++) {
    cumulative += i + 1;
    if (cumulative >= random) {
      return sorted[i];
    }
  }

  return sorted[sorted.length - 1];
}

// ============================================================================
// CROSSOVER
// ============================================================================

/**
 * Combine two parents to create offspring
 */
function crossover(
  parent1: Individual,
  parent2: Individual,
  generation: number
): [Individual, Individual] {
  // Single-point crossover on prompt text
  const prompt1 = parent1.prompt;
  const prompt2 = parent2.prompt;

  // Find sentences in both prompts
  const sentences1 = splitIntoSentences(prompt1);
  const sentences2 = splitIntoSentences(prompt2);

  // Crossover point
  const point1 = Math.floor(Math.random() * sentences1.length);
  const point2 = Math.floor(Math.random() * sentences2.length);

  // Create offspring
  const offspring1Prompt = [
    ...sentences1.slice(0, point1),
    ...sentences2.slice(point2),
  ].join(' ');

  const offspring2Prompt = [
    ...sentences2.slice(0, point2),
    ...sentences1.slice(point1),
  ].join(' ');

  // Note: Fitness will be evaluated after mutation or when added to population
  const offspring1: Individual = {
    id: generateId(),
    prompt: offspring1Prompt,
    fitness: 0, // Will be evaluated
    generation,
    parents: [parent1.id, parent2.id],
    mutations: [...new Set([...parent1.mutations, ...parent2.mutations])],
    metadata: {
      length: offspring1Prompt.length,
    },
  };

  const offspring2: Individual = {
    id: generateId(),
    prompt: offspring2Prompt,
    fitness: 0,
    generation,
    parents: [parent1.id, parent2.id],
    mutations: [...new Set([...parent1.mutations, ...parent2.mutations])],
    metadata: {
      length: offspring2Prompt.length,
    },
  };

  return [offspring1, offspring2];
}

// ============================================================================
// MUTATION
// ============================================================================

/**
 * Apply random mutation to individual
 */
async function mutate(
  individual: Individual,
  fitnessFunction: FitnessFunction
): Promise<Individual> {
  const mutationTypes = [
    { name: 'try-catch', fn: tryCatchStyleMutation },
    { name: 'reduce', fn: reduceContextMutation },
    { name: 'expand', fn: expandMutation },
    { name: 'word-swap', fn: wordSwapMutation },
    { name: 'sentence-shuffle', fn: sentenceShuffleMutation },
  ];

  const mutation = mutationTypes[Math.floor(Math.random() * mutationTypes.length)];
  const variation = mutation.fn(individual.prompt);
  const mutatedPrompt = variation.text;

  // Evaluate fitness of mutated individual
  const fitness = await evaluateFitness(mutatedPrompt, fitnessFunction);

  return {
    ...individual,
    id: generateId(),
    prompt: mutatedPrompt,
    fitness,
    mutations: [...individual.mutations, mutation.name],
    metadata: {
      ...individual.metadata,
      length: mutatedPrompt.length,
    },
  };
}

/**
 * Simple word swap mutation
 */
function wordSwapMutation(prompt: string): PromptVariation {
  const words = prompt.split(/\s+/);

  if (words.length < 4) {
    return {
      text: prompt,
      mutationType: 'expansion',
      changeDescription: 'No mutation (prompt too short)',
      expectedImpact: {},
    };
  }

  // Swap two random words
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
    expectedImpact: { quality: 'neutral' },
  };
}

/**
 * Shuffle sentences
 */
function sentenceShuffleMutation(prompt: string): PromptVariation {
  const sentences = splitIntoSentences(prompt);

  if (sentences.length < 2) {
    return {
      text: prompt,
      mutationType: 'expansion',
      changeDescription: 'No mutation (single sentence)',
      expectedImpact: {},
    };
  }

  // Shuffle sentences (Fisher-Yates)
  for (let i = sentences.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [sentences[i], sentences[j]] = [sentences[j], sentences[i]];
  }

  return {
    text: sentences.join(' '),
    mutationType: 'expansion',
    changeDescription: 'Shuffled sentence order',
    expectedImpact: { quality: 'neutral' },
  };
}

// ============================================================================
// STATISTICS & ANALYSIS
// ============================================================================

/**
 * Calculate statistics for a generation
 */
function calculateGenerationStats(
  population: Individual[],
  generation: number
): GenerationStats {
  const fitnesses = population.map(ind => ind.fitness);

  const best = Math.max(...fitnesses);
  const worst = Math.min(...fitnesses);
  const avg = fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length;

  // Calculate standard deviation
  const variance = fitnesses.reduce((sum, f) => sum + Math.pow(f - avg, 2), 0) / fitnesses.length;
  const stdDev = Math.sqrt(variance);

  // Calculate diversity (average pairwise distance)
  const diversity = calculateDiversity(population);

  const bestIndividual = population.reduce((best, current) =>
    current.fitness > best.fitness ? current : best
  );

  return {
    generation,
    bestFitness: best,
    avgFitness: avg,
    worstFitness: worst,
    stdDevFitness: stdDev,
    diversity,
    bestIndividual,
  };
}

/**
 * Calculate population diversity
 */
function calculateDiversity(population: Individual[]): number {
  let totalDistance = 0;
  let comparisons = 0;

  for (let i = 0; i < population.length; i++) {
    for (let j = i + 1; j < population.length; j++) {
      const distance = levenshteinDistance(
        population[i].prompt,
        population[j].prompt
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
function hasConverged(recentHistory: GenerationStats[]): boolean {
  if (recentHistory.length < 4) return false;

  // Check if improvement is less than 1% over last 3 generations
  const improvements = [];
  for (let i = 1; i < recentHistory.length; i++) {
    const improvement = recentHistory[i].bestFitness - recentHistory[i - 1].bestFitness;
    improvements.push(improvement);
  }

  const avgImprovement = improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length;
  return avgImprovement < 0.01;
}

/**
 * Find generation where algorithm converged
 */
function findConvergenceGeneration(history: GenerationStats[]): number | undefined {
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
 * Evaluate fitness of a prompt
 */
async function evaluateFitness(
  prompt: string,
  fitnessFunction: FitnessFunction
): Promise<number> {
  return await fitnessFunction(prompt);
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
 * Calculate Levenshtein distance between two strings
 */
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;
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
  return `ind_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  geneticOptimize,
  DEFAULT_GENETIC_CONFIG,
};
