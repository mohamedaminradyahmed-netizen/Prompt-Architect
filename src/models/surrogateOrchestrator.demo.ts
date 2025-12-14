/**
 * Surrogate Orchestrator Demo - DIRECTIVE-037
 *
 * Demonstrates the cost-saving capabilities of the SurrogateOrchestrator
 * by showing how different evaluation modes select different models.
 */

import {
  SurrogateOrchestrator,
  createCostOptimizedOrchestrator,
  createQualityFocusedOrchestrator,
  createBalancedOrchestrator,
  EvaluationMode,
  MODEL_REGISTRY,
} from './surrogateOrchestrator';
import { PromptCategory } from '../types/promptTypes';

// ============================================================================
// DEMO FUNCTIONS
// ============================================================================

/**
 * Demo: Basic evaluation with different modes
 */
async function demoBasicEvaluation(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('📊 Demo: Basic Evaluation with Different Modes');
  console.log('='.repeat(60));

  const orchestrator = new SurrogateOrchestrator();

  const testPrompt = {
    prompt: 'Write a TypeScript function that validates email addresses using regex.',
    category: PromptCategory.CODE_GENERATION,
    expectedOutputLength: 500,
  };

  // Evaluate with each mode
  const modes: EvaluationMode[] = ['exploration', 'exploitation', 'final'];

  for (const mode of modes) {
    console.log(`\n🔄 Evaluating with mode: ${mode}`);

    const result = await orchestrator.evaluate(testPrompt, mode);

    console.log(`   📦 Model: ${result.model.model}`);
    console.log(`   💰 Cost: $${result.cost.toFixed(6)}`);
    console.log(`   ⏱️ Latency: ${result.latency}ms`);
    console.log(`   📈 Score: ${(result.score * 100).toFixed(1)}%`);
    console.log(`   🎯 Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`   🔢 Tokens: ${result.tokens.total}`);
  }

  // Show statistics
  const stats = orchestrator.getStats();
  console.log('\n📈 Statistics:');
  console.log(`   Total Requests: ${stats.totalRequests}`);
  console.log(`   Total Cost: $${stats.totalCost.toFixed(6)}`);
  console.log(`   Total Savings: $${stats.totalSavings.toFixed(6)}`);
  console.log(`   Avg Latency: ${stats.avgLatency.toFixed(0)}ms`);
}

/**
 * Demo: Progressive evaluation
 */
async function demoProgressiveEvaluation(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('🔄 Demo: Progressive Evaluation');
  console.log('='.repeat(60));

  const orchestrator = new SurrogateOrchestrator();

  const testPrompt = {
    prompt: 'Explain the concept of recursion in programming with examples.',
    category: PromptCategory.GENERAL_QA,
  };

  console.log('\n📝 Progressive evaluation starts cheap and upgrades if needed...');

  const result = await orchestrator.progressiveEvaluate(testPrompt, 0.85);

  console.log(`\n✅ Final Result:`);
  console.log(`   Mode used: ${result.metadata.mode}`);
  console.log(`   Model: ${result.model.model}`);
  console.log(`   Score: ${(result.score * 100).toFixed(1)}%`);
  console.log(`   Cost: $${result.cost.toFixed(6)}`);

  const stats = orchestrator.getStats();
  console.log(`\n📊 Evaluations performed: ${stats.totalRequests}`);
}

/**
 * Demo: Batch evaluation
 */
async function demoBatchEvaluation(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('📦 Demo: Batch Evaluation');
  console.log('='.repeat(60));

  const orchestrator = new SurrogateOrchestrator();

  const prompts = [
    { prompt: 'Create a Python class for a shopping cart', category: PromptCategory.CODE_GENERATION },
    { prompt: 'Write marketing copy for a fitness app', category: PromptCategory.MARKETING_COPY },
    { prompt: 'Explain machine learning to a beginner', category: PromptCategory.GENERAL_QA },
    { prompt: 'Review this code for security issues', category: PromptCategory.CODE_REVIEW },
    { prompt: 'Generate a blog post about AI trends', category: PromptCategory.CONTENT_WRITING },
  ];

  console.log(`\n📋 Evaluating ${prompts.length} prompts in exploration mode...`);

  const batchResult = await orchestrator.evaluateBatch(prompts, 'exploration');

  console.log(`\n✅ Batch Results:`);
  console.log(`   Success Rate: ${(batchResult.successRate * 100).toFixed(1)}%`);
  console.log(`   Total Cost: $${batchResult.totalCost.toFixed(6)}`);
  console.log(`   Avg Latency: ${batchResult.avgLatency.toFixed(0)}ms`);
  console.log(`   Cost Savings: $${batchResult.costSavings.toFixed(6)}`);
}

/**
 * Demo: Cache effectiveness
 */
async function demoCacheEffectiveness(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('💾 Demo: Cache Effectiveness');
  console.log('='.repeat(60));

  const orchestrator = new SurrogateOrchestrator();

  const testPrompt = {
    prompt: 'What is the capital of France?',
    category: PromptCategory.GENERAL_QA,
  };

  console.log('\n🔄 First request (no cache)...');
  const result1 = await orchestrator.evaluate(testPrompt, 'exploration');
  console.log(`   Cached: ${result1.metadata.cached}`);
  console.log(`   Latency: ${result1.latency}ms`);

  console.log('\n🔄 Second request (cached)...');
  const result2 = await orchestrator.evaluate(testPrompt, 'exploration');
  console.log(`   Cached: ${result2.metadata.cached}`);
  console.log(`   Latency: ${result2.latency}ms`);

  const stats = orchestrator.getStats();
  console.log(`\n📈 Cache Hit Rate: ${(stats.cacheHitRate * 100).toFixed(1)}%`);
}

/**
 * Demo: Different orchestrator presets
 */
async function demoOrchestratorPresets(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('🎛️ Demo: Orchestrator Presets');
  console.log('='.repeat(60));

  const testPrompt = {
    prompt: 'Design a REST API for a todo application',
    category: PromptCategory.CODE_GENERATION,
  };

  // Cost-optimized
  console.log('\n💰 Cost-Optimized Orchestrator:');
  const costOptimized = createCostOptimizedOrchestrator();
  const costResult = await costOptimized.evaluate(testPrompt, 'final');
  console.log(`   Model: ${costResult.model.model}`);
  console.log(`   Cost: $${costResult.cost.toFixed(6)}`);

  // Quality-focused
  console.log('\n⭐ Quality-Focused Orchestrator:');
  const qualityFocused = createQualityFocusedOrchestrator();
  const qualityResult = await qualityFocused.evaluate(testPrompt, 'final');
  console.log(`   Model: ${qualityResult.model.model}`);
  console.log(`   Cost: $${qualityResult.cost.toFixed(6)}`);

  // Balanced
  console.log('\n⚖️ Balanced Orchestrator:');
  const balanced = createBalancedOrchestrator();
  const balancedResult = await balanced.evaluate(testPrompt, 'final');
  console.log(`   Model: ${balancedResult.model.model}`);
  console.log(`   Cost: $${balancedResult.cost.toFixed(6)}`);
}

/**
 * Demo: Cost savings analysis
 */
async function demoCostSavings(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('💵 Demo: Cost Savings Analysis');
  console.log('='.repeat(60));

  const orchestrator = new SurrogateOrchestrator();

  // Simulate 50 requests with different modes
  const prompts = Array.from({ length: 50 }, (_, i) => ({
    prompt: `Sample prompt number ${i + 1} for testing cost analysis.`,
    category: PromptCategory.GENERAL_QA,
  }));

  console.log('\n📊 Running 50 evaluations with mixed modes...');

  // 70% exploration, 20% exploitation, 10% final
  for (let i = 0; i < prompts.length; i++) {
    let mode: EvaluationMode;
    if (i < 35) mode = 'exploration';
    else if (i < 45) mode = 'exploitation';
    else mode = 'final';

    await orchestrator.evaluate(prompts[i], mode);
  }

  const savings = orchestrator.getCostSavingsSummary();
  console.log('\n💰 Cost Savings Summary:');
  console.log(`   Actual Cost: $${savings.totalCost.toFixed(6)}`);
  console.log(`   If Premium Only: $${savings.estimatedPremiumCost.toFixed(6)}`);
  console.log(`   Savings: $${savings.savings.toFixed(6)}`);
  console.log(`   Savings %: ${savings.savingsPercentage.toFixed(1)}%`);

  const usage = orchestrator.getModelUsageBreakdown();
  console.log('\n📈 Usage Breakdown:');
  console.log(`   By Mode:`, usage.byMode);
  console.log(`   By Provider:`, usage.byProvider);
}

/**
 * Demo: Show available models
 */
function demoShowModels(): void {
  console.log('\n' + '='.repeat(60));
  console.log('📚 Available Models');
  console.log('='.repeat(60));

  const models = Object.entries(MODEL_REGISTRY);

  console.log('\n🔌 Cheap Tier (Exploration):');
  models.filter(([_, m]) => m.tier === 'cheap').forEach(([key, model]) => {
    console.log(`   ${key}: ${model.model} ($${model.costPer1kTokens}/1K tokens, ${model.avgLatencyMs}ms)`);
  });

  console.log('\n🔧 Mid Tier (Exploitation):');
  models.filter(([_, m]) => m.tier === 'mid').forEach(([key, model]) => {
    console.log(`   ${key}: ${model.model} ($${model.costPer1kTokens}/1K tokens, ${model.avgLatencyMs}ms)`);
  });

  console.log('\n⭐ Premium Tier (Final):');
  models.filter(([_, m]) => m.tier === 'premium').forEach(([key, model]) => {
    console.log(`   ${key}: ${model.model} ($${model.costPer1kTokens}/1K tokens, ${model.avgLatencyMs}ms)`);
  });
}

// ============================================================================
// MAIN DEMO RUNNER
// ============================================================================

async function runAllDemos(): Promise<void> {
  console.log('\n' + '🚀'.repeat(30));
  console.log('     SURROGATE ORCHESTRATOR DEMO - DIRECTIVE-037');
  console.log('🚀'.repeat(30));
  console.log('\nThis demo shows how the SurrogateOrchestrator saves 60-80% costs');
  console.log('by using cheaper models for exploration and premium for final evaluation.');

  try {
    demoShowModels();
    await demoBasicEvaluation();
    await demoProgressiveEvaluation();
    await demoBatchEvaluation();
    await demoCacheEffectiveness();
    await demoOrchestratorPresets();
    await demoCostSavings();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All demos completed successfully!');
    console.log('='.repeat(60));
  } catch (error) {
    console.error('❌ Demo error:', error);
  }
}

// Run if executed directly
// runAllDemos();

export {
  runAllDemos,
  demoBasicEvaluation,
  demoProgressiveEvaluation,
  demoBatchEvaluation,
  demoCacheEffectiveness,
  demoOrchestratorPresets,
  demoCostSavings,
  demoShowModels,
};
