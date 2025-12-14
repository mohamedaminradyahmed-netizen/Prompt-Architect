/**
 * Lineage Tracking Demo (DIRECTIVE-028)
 *
 * Demonstrates lineage tracking capabilities:
 * - Tracking variation evolution
 * - Building genealogy trees
 * - Finding optimal paths
 * - Analyzing mutation success rates
 */

import {
  LineageTracker,
  createOriginalVariation,
  createChildVariation,
  formatPath,
  visualizeTree,
} from './tracker';
import { tryCatchStyleMutation, reduceContextMutation, expandMutation } from '../mutations';

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

/**
 * Demo 1: Basic Lineage Tracking
 */
function demo1_BasicTracking() {
  console.log('='.repeat(80));
  console.log('DEMO 1: Basic Lineage Tracking');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();

  // Create original
  const original = createOriginalVariation('Write a function to sort arrays', 0.5, 0.01, 1000);

  tracker.trackVariation(original);

  console.log(`\n📝 Original Variation:`);
  console.log(`   ID: ${original.id}`);
  console.log(`   Prompt: "${original.currentPrompt}"`);
  console.log(`   Score: ${original.metrics.score}`);
  console.log(`   Generation: ${original.generation}`);

  // Apply mutations
  const tryCatchResult = tryCatchStyleMutation(original.currentPrompt);
  const variation1 = createChildVariation(
    original,
    tryCatchResult.text,
    'try-catch-style',
    {},
    0.65,
    0.012,
    1100
  );
  tracker.trackVariation(variation1);

  console.log(`\n🔄 After Try/Catch Mutation:`);
  console.log(`   ID: ${variation1.id}`);
  console.log(`   Parent: ${variation1.parentId}`);
  console.log(`   Score: ${variation1.metrics.score} (+${(variation1.metrics.score - original.metrics.score).toFixed(3)})`);
  console.log(`   Generation: ${variation1.generation}`);

  // Apply another mutation
  const expandResult = expandMutation(variation1.currentPrompt);
  const variation2 = createChildVariation(
    variation1,
    expandResult.text,
    'expansion',
    {},
    0.78,
    0.015,
    1200
  );
  tracker.trackVariation(variation2);

  console.log(`\n🔄 After Expand Mutation:`);
  console.log(`   ID: ${variation2.id}`);
  console.log(`   Parent: ${variation2.parentId}`);
  console.log(`   Score: ${variation2.metrics.score} (+${(variation2.metrics.score - variation1.metrics.score).toFixed(3)})`);
  console.log(`   Generation: ${variation2.generation}`);

  // Show lineage
  const lineage = tracker.getLineage(variation2.id);
  console.log(`\n🛤️  Complete Lineage Path:`);
  console.log(`   ${formatPath(lineage)}`);

  console.log(`\n📊 Variation Count: ${lineage.length}`);
}

/**
 * Demo 2: Multiple Branches
 */
function demo2_MultipleBranches() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 2: Multiple Branches (Tree Structure)');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();

  // Original
  const original = createOriginalVariation('Create authentication system', 0.6, 0.02, 1500);
  tracker.trackVariation(original);

  console.log(`\n📝 Building variation tree...\n`);

  // Branch 1: Try/Catch → Expand
  const branch1_step1 = createChildVariation(
    original,
    tryCatchStyleMutation(original.currentPrompt).text,
    'try-catch-style',
    {},
    0.72,
    0.022,
    1600
  );
  tracker.trackVariation(branch1_step1);

  const branch1_step2 = createChildVariation(
    branch1_step1,
    expandMutation(branch1_step1.currentPrompt).text,
    'expansion',
    {},
    0.85,
    0.025,
    1700
  );
  tracker.trackVariation(branch1_step2);

  // Branch 2: Reduce → Try/Catch
  const branch2_step1 = createChildVariation(
    original,
    reduceContextMutation(original.currentPrompt).text,
    'context-reduction',
    {},
    0.68,
    0.018,
    1400
  );
  tracker.trackVariation(branch2_step1);

  const branch2_step2 = createChildVariation(
    branch2_step1,
    tryCatchStyleMutation(branch2_step1.currentPrompt).text,
    'try-catch-style',
    {},
    0.75,
    0.020,
    1500
  );
  tracker.trackVariation(branch2_step2);

  // Branch 3: Expand only
  const branch3_step1 = createChildVariation(
    original,
    expandMutation(original.currentPrompt).text,
    'expansion',
    {},
    0.70,
    0.024,
    1650
  );
  tracker.trackVariation(branch3_step1);

  // Visualize tree
  const graph = tracker.visualizeLineage(original.id);

  console.log('🌲 Lineage Tree:');
  console.log('─'.repeat(80));
  console.log(visualizeTree(graph, 5));
  console.log('─'.repeat(80));

  // Statistics
  console.log(`\n📊 Statistics:`);
  console.log(`   Total Variations: ${graph.stats.totalVariations}`);
  console.log(`   Max Generation: ${graph.stats.maxGeneration}`);
  console.log(`   Best Score: ${graph.stats.bestVariation.metrics.score}`);
  console.log(`   Best Path: ${formatPath(tracker.getLineage(graph.stats.bestVariation.id))}`);
}

/**
 * Demo 3: Finding Best Path
 */
function demo3_FindBestPath() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 3: Finding Best Path to Target Score');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();
  const originalPrompt = 'Implement database query optimization';
  const original = createOriginalVariation(originalPrompt, 0.5, 0.02, 2000);
  tracker.trackVariation(original);

  console.log(`\n🎯 Goal: Find shortest path to score ≥ 0.80`);
  console.log(`   Starting Score: ${original.metrics.score}\n`);

  // Create multiple paths with different scores
  const paths = [
    // Path 1: 3 steps
    [
      { mutation: 'try-catch-style' as const, score: 0.62 },
      { mutation: 'expansion' as const, score: 0.74 },
      { mutation: 'expansion' as const, score: 0.83 },
    ],
    // Path 2: 2 steps (shorter!)
    [
      { mutation: 'expansion' as const, score: 0.70 },
      { mutation: 'try-catch-style' as const, score: 0.85 },
    ],
    // Path 3: 4 steps
    [
      { mutation: 'context-reduction' as const, score: 0.55 },
      { mutation: 'try-catch-style' as const, score: 0.68 },
      { mutation: 'expansion' as const, score: 0.77 },
      { mutation: 'expansion' as const, score: 0.82 },
    ],
  ];

  paths.forEach((path, pathIdx) => {
    let current = original;
    path.forEach((step, stepIdx) => {
      const mutated =
        step.mutation === 'try-catch-style'
          ? tryCatchStyleMutation(current.currentPrompt).text
          : step.mutation === 'expansion'
          ? expandMutation(current.currentPrompt).text
          : reduceContextMutation(current.currentPrompt).text;

      const variation = createChildVariation(
        current,
        mutated,
        step.mutation,
        {},
        step.score,
        0.02,
        2000
      );
      tracker.trackVariation(variation);
      current = variation;

      console.log(`Path ${pathIdx + 1}, Step ${stepIdx + 1}: ${step.mutation} → Score: ${step.score}`);
    });
    console.log();
  });

  // Find best path
  const bestPath = tracker.findBestPath(originalPrompt, 0.80);

  if (bestPath) {
    console.log(`\n✅ Best Path Found (${bestPath.length - 1} mutations):`);
    console.log(`   ${formatPath(bestPath)}`);
    console.log(`\n   Mutations Applied:`);
    bestPath.slice(1).forEach((v, idx) => {
      console.log(`   ${idx + 1}. ${v.mutation} → Score: ${v.metrics.score}`);
    });
  }
}

/**
 * Demo 4: Mutation Success Rates
 */
function demo4_MutationAnalysis() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 4: Mutation Success Rate Analysis');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();

  // Create variations with different mutation successes
  const original = createOriginalVariation('Optimize API performance', 0.6, 0.02, 1800);
  tracker.trackVariation(original);

  console.log(`\n📊 Simulating 20 mutations...\n`);

  const mutations = [
    { type: 'try-catch-style' as const, successRate: 0.7 },
    { type: 'expansion' as const, successRate: 0.6 },
    { type: 'context-reduction' as const, successRate: 0.5 },
  ];

  let variationCount = 0;

  mutations.forEach((mut) => {
    for (let i = 0; i < 7; i++) {
      const parent = variationCount === 0 ? original : tracker.getByGeneration(0)[0];
      const improves = Math.random() < mut.successRate;
      const scoreDelta = improves ? Math.random() * 0.15 + 0.05 : -Math.random() * 0.1;
      const newScore = Math.max(0.3, Math.min(1.0, parent.metrics.score + scoreDelta));

      const mutated =
        mut.type === 'try-catch-style'
          ? tryCatchStyleMutation(parent.currentPrompt).text
          : mut.type === 'expansion'
          ? expandMutation(parent.currentPrompt).text
          : reduceContextMutation(parent.currentPrompt).text;

      const variation = createChildVariation(parent, mutated, mut.type, {}, newScore, 0.02, 1800);
      tracker.trackVariation(variation);
      variationCount++;
    }
  });

  // Get statistics
  const stats = tracker.getGlobalStats();

  console.log(`📈 Mutation Success Rates:`);
  console.log('─'.repeat(80));

  for (const [mutation, rate] of stats.mutationSuccessRates) {
    const percentage = (rate * 100).toFixed(1);
    const bar = '█'.repeat(Math.floor(rate * 40));
    console.log(`   ${mutation.padEnd(20)} │ ${percentage.padStart(5)}% │ ${bar}`);
  }

  console.log(`\n🏆 Most Used Mutation: ${stats.mostUsedMutation}`);
  console.log(`📊 Total Variations: ${stats.totalVariations}`);
}

/**
 * Demo 5: Human Feedback Integration
 */
function demo5_HumanFeedback() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 5: Human Feedback Integration');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();

  const original = createOriginalVariation('Build REST API endpoint', 0.65, 0.02, 1600);
  tracker.trackVariation(original);

  const variation = createChildVariation(
    original,
    expandMutation(original.currentPrompt).text,
    'expansion',
    {},
    0.80,
    0.023,
    1700
  );
  tracker.trackVariation(variation);

  console.log(`\n📝 Variation Created:`);
  console.log(`   ID: ${variation.id}`);
  console.log(`   Score: ${variation.metrics.score}`);
  console.log(`   Initial Feedback: None`);

  // Add human feedback
  tracker.addFeedback(variation.id, {
    userId: 'user_123',
    rating: 4,
    comment: 'Good improvement, but could be more specific about error handling',
    timestamp: new Date(),
    tags: ['good-quality', 'needs-refinement'],
  });

  const updated = tracker.getLineage(variation.id).find((v) => v.id === variation.id)!;

  console.log(`\n👤 After Human Feedback:`);
  console.log(`   User: ${updated.feedback?.userId}`);
  console.log(`   Rating: ${'⭐'.repeat(updated.feedback?.rating || 0)} (${updated.feedback?.rating}/5)`);
  console.log(`   Comment: "${updated.feedback?.comment}"`);
  console.log(`   Tags: ${updated.feedback?.tags?.join(', ')}`);
}

/**
 * Demo 6: Generation-by-Generation Analysis
 */
function demo6_GenerationAnalysis() {
  console.log('\n\n' + '='.repeat(80));
  console.log('DEMO 6: Generation-by-Generation Score Analysis');
  console.log('='.repeat(80));

  const tracker = new LineageTracker();

  const original = createOriginalVariation('Create machine learning model', 0.5, 0.03, 2000);
  tracker.trackVariation(original);

  console.log(`\n📊 Simulating 5 generations of evolution...\n`);

  let currentGen = [original];

  for (let gen = 1; gen <= 5; gen++) {
    const nextGen = [];

    for (const parent of currentGen) {
      // Create 2 children per parent
      for (let i = 0; i < 2; i++) {
        const mutations = ['try-catch-style', 'expansion', 'context-reduction'] as const;
        const mutation = mutations[Math.floor(Math.random() * mutations.length)];

        const scoreDelta = Math.random() * 0.1 + 0.03; // Generally improving
        const newScore = Math.min(1.0, parent.metrics.score + scoreDelta);

        const mutated =
          mutation === 'try-catch-style'
            ? tryCatchStyleMutation(parent.currentPrompt).text
            : mutation === 'expansion'
            ? expandMutation(parent.currentPrompt).text
            : reduceContextMutation(parent.currentPrompt).text;

        const variation = createChildVariation(parent, mutated, mutation, {}, newScore, 0.03, 2000);
        tracker.trackVariation(variation);
        nextGen.push(variation);
      }
    }

    currentGen = nextGen;
  }

  // Analyze by generation
  const stats = tracker.getGlobalStats();

  console.log(`📈 Score Evolution by Generation:`);
  console.log('─'.repeat(80));

  for (const [gen, avgScore] of stats.avgScoreByGeneration) {
    const percentage = (avgScore * 100).toFixed(1);
    const bar = '█'.repeat(Math.floor(avgScore * 50));
    const genVariations = tracker.getByGeneration(gen);
    const bestInGen = Math.max(...genVariations.map((v) => v.metrics.score));

    console.log(`   Gen ${gen} │ Avg: ${percentage.padStart(5)}% │ Best: ${(bestInGen * 100).toFixed(1)}% │ ${bar}`);
  }

  console.log(`\n🏆 Overall Best: ${(stats.bestVariation.metrics.score * 100).toFixed(1)}% (Generation ${stats.bestVariation.generation})`);
  console.log(`📊 Total Variations: ${stats.totalVariations}`);
}

// ============================================================================
// RUN ALL DEMOS
// ============================================================================

async function runAllDemos() {
  console.log('\n');
  console.log('╔' + '═'.repeat(78) + '╗');
  console.log('║' + ' '.repeat(22) + 'LINEAGE TRACKING SYSTEM' + ' '.repeat(33) + '║');
  console.log('║' + ' '.repeat(27) + 'DIRECTIVE-028' + ' '.repeat(38) + '║');
  console.log('╚' + '═'.repeat(78) + '╝');
  console.log('\n');

  try {
    demo1_BasicTracking();
    demo2_MultipleBranches();
    demo3_FindBestPath();
    demo4_MutationAnalysis();
    demo5_HumanFeedback();
    demo6_GenerationAnalysis();

    console.log('\n\n' + '='.repeat(80));
    console.log('✅ ALL DEMOS COMPLETED SUCCESSFULLY');
    console.log('='.repeat(80));

    console.log('\n📚 Key Features Demonstrated:');
    console.log('  1. ✅ Parent-child relationship tracking');
    console.log('  2. ✅ Complete lineage path reconstruction');
    console.log('  3. ✅ Tree visualization with multiple branches');
    console.log('  4. ✅ Best path discovery to target score');
    console.log('  5. ✅ Mutation success rate analysis');
    console.log('  6. ✅ Human feedback integration');
    console.log('  7. ✅ Generation-by-generation evolution tracking');

    console.log('\n💡 Use Cases:');
    console.log('  • Understand which mutation chains work best');
    console.log('  • Debug optimization strategies');
    console.log('  • Analyze prompt evolution over time');
    console.log('  • Incorporate human feedback into learning');
    console.log('  • Find shortest path to desired quality');

    console.log('\n');
  } catch (error) {
    console.error('❌ Error running demos:', error);
  }
}

// Run demos if this file is executed directly
if (require.main === module) {
  runAllDemos();
}

export { runAllDemos };
