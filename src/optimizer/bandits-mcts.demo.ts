/**
 * Bandits & MCTS Demo (DIRECTIVE-022)
 *
 * Demonstrates Multi-Armed Bandits (UCB1) and Monte Carlo Tree Search
 * for efficient exploration of large mutation spaces.
 */

import { banditOptimize } from './bandits';
import { mctsOptimize } from './mcts';
import { ScoringFunction } from './types';

// ============================================================================
// DEMO SCORING FUNCTIONS
// ============================================================================

/**
 * Simple scoring function for demos
 */
const simpleScoring: ScoringFunction = async (prompt: string): Promise<number> => {
    let score = 0.5; // Base score

    // Reward clarity
    if (/\b(create|write|implement|build)\b/i.test(prompt)) {
        score += 0.15;
    }

    // Reward specificity
    if (/\b(function|class|component|system)\b/i.test(prompt)) {
        score += 0.15;
    }

    // Reward constraints
    if (/\b(must|should|ensure|require)\b/i.test(prompt)) {
        score += 0.1;
    }

    // Reward examples
    if (/\b(example|such as|like|for instance)\b/i.test(prompt)) {
        score += 0.1;
    }

    // Length penalty for too short or too long
    const length = prompt.length;
    if (length < 50) {
        score -= 0.2;
    } else if (length > 500) {
        score -= 0.1;
    }

    return Math.max(0, Math.min(1, score));
};

/**
 * Code quality scoring
 */
const codeQualityScoring: ScoringFunction = async (prompt: string): Promise<number> => {
    let score = 0.3;

    // Must specify language
    if (/\b(typescript|javascript|python|java|rust|go)\b/i.test(prompt)) {
        score += 0.2;
    }

    // Should mention testing
    if (/\b(test|testing|unit test|spec)\b/i.test(prompt)) {
        score += 0.15;
    }

    // Should mention error handling
    if (/\b(error|exception|try|catch|handle)\b/i.test(prompt)) {
        score += 0.15;
    }

    // Should be specific about implementation
    if (/\b(async|await|promise|callback)\b/i.test(prompt)) {
        score += 0.1;
    }

    // Should mention types or interfaces
    if (/\b(type|interface|class|struct)\b/i.test(prompt)) {
        score += 0.1;
    }

    return Math.max(0, Math.min(1, score));
};

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

/**
 * Demo 1: Multi-Armed Bandit with UCB1
 */
async function demo1_BanditUCB1() {
    console.log('='.repeat(80));
    console.log('DEMO 1: Multi-Armed Bandit (UCB1 Algorithm)');
    console.log('='.repeat(80));

    const initialPrompt = 'Write a function to sort numbers';

    console.log(`\nInitial Prompt: "${initialPrompt}"`);
    console.log(`Algorithm: UCB1 (Upper Confidence Bound)`);
    console.log(`Budget: 50 trials\n`);

    const startTime = Date.now();

    const result = await banditOptimize(
        initialPrompt,
        50, // budget
        simpleScoring
    );

    const duration = Date.now() - startTime;

    console.log('\n' + '='.repeat(80));
    console.log('RESULTS');
    console.log('='.repeat(80));

    console.log(`\n🏆 Best Mutation: ${result.bestMutationId}`);
    console.log(`📝 Best Prompt: "${result.bestPrompt.substring(0, 100)}${result.bestPrompt.length > 100 ? '...' : ''}"`);
    console.log(`⭐ Best Score: ${result.bestScore.toFixed(3)}`);
    console.log(`⏱️  Duration: ${duration}ms\n`);

    console.log('📊 Arm Statistics:');
    console.log('─'.repeat(80));

    for (const [armId, stats] of Object.entries(result.armStats)) {
        const pullPercent = (stats.pulls / 50) * 100;
        const bar = '█'.repeat(Math.floor(pullPercent / 2));
        console.log(`${armId.padEnd(20)} │ Pulls: ${stats.pulls.toString().padStart(2)} (${pullPercent.toFixed(0)}%) │ Avg: ${stats.avgReward.toFixed(3)} │ ${bar}`);
    }

    console.log('\n💡 Insight: UCB1 balances exploration (trying all arms) and exploitation (favoring best arms).');
}

/**
 * Demo 2: Monte Carlo Tree Search
 */
async function demo2_MCTS() {
    console.log('\n\n' + '='.repeat(80));
    console.log('DEMO 2: Monte Carlo Tree Search (MCTS)');
    console.log('='.repeat(80));

    const initialPrompt = 'Create user authentication';

    console.log(`\nInitial Prompt: "${initialPrompt}"`);
    console.log(`Algorithm: MCTS`);
    console.log(`Iterations: 30`);
    console.log(`Max Depth: 4\n`);

    const startTime = Date.now();

    const result = await mctsOptimize(
        initialPrompt,
        30,  // iterations
        4,   // maxDepth
        codeQualityScoring
    );

    const duration = Date.now() - startTime;

    console.log('\n' + '='.repeat(80));
    console.log('RESULTS');
    console.log('='.repeat(80));

    console.log(`\n🎯 Best Prompt Found:`);
    console.log(`   "${result.bestPrompt}"`);
    console.log(`\n⭐ Score: ${result.bestScore.toFixed(3)}`);
    console.log(`🔄 Iterations: ${result.iterations}`);
    console.log(`⏱️  Duration: ${duration}ms\n`);

    console.log('🛤️  Mutation Path:');
    result.path.forEach((step, idx) => {
        console.log(`   ${idx + 1}. ${step}`);
    });

    console.log('\n💡 Insight: MCTS explores sequences of mutations, finding optimal chains of transformations.');
}

/**
 * Demo 3: Comparison - Bandit vs MCTS
 */
async function demo3_Comparison() {
    console.log('\n\n' + '='.repeat(80));
    console.log('DEMO 3: Bandit vs MCTS Comparison');
    console.log('='.repeat(80));

    const testPrompts = [
        'Write login code',
        'Build REST API',
        'Create database schema'
    ];

    console.log('\nTesting on 3 different prompts:\n');

    for (const prompt of testPrompts) {
        console.log(`\n📝 Prompt: "${prompt}"`);
        console.log('─'.repeat(80));

        // Bandit
        const banditStart = Date.now();
        const banditResult = await banditOptimize(prompt, 30, codeQualityScoring);
        const banditTime = Date.now() - banditStart;

        // MCTS
        const mctsStart = Date.now();
        const mctsResult = await mctsOptimize(prompt, 20, 3, codeQualityScoring);
        const mctsTime = Date.now() - mctsStart;

        console.log(`\n  Bandit (UCB1):`);
        console.log(`    Score: ${banditResult.bestScore.toFixed(3)} | Time: ${banditTime}ms`);
        console.log(`    Best: ${banditResult.bestMutationId}`);

        console.log(`\n  MCTS:`);
        console.log(`    Score: ${mctsResult.bestScore.toFixed(3)} | Time: ${mctsTime}ms`);
        console.log(`    Path: ${mctsResult.path.join(' → ')}`);

        const winner = banditResult.bestScore > mctsResult.bestScore ? 'Bandit' : 'MCTS';
        console.log(`\n  🏆 Winner: ${winner}`);
    }

    console.log('\n\n💡 Key Differences:');
    console.log('   • Bandit: Finds best SINGLE mutation (fast, simple)');
    console.log('   • MCTS: Finds best SEQUENCE of mutations (slower, more thorough)');
}

/**
 * Demo 4: Budget Analysis - How many trials needed?
 */
async function demo4_BudgetAnalysis() {
    console.log('\n\n' + '='.repeat(80));
    console.log('DEMO 4: Budget Analysis - Finding Optimal Trial Count');
    console.log('='.repeat(80));

    const prompt = 'Implement sorting algorithm';
    const budgets = [10, 20, 30, 50, 100];

    console.log(`\nTesting different budgets for Bandit optimization:\n`);

    for (const budget of budgets) {
        const startTime = Date.now();
        const result = await banditOptimize(prompt, budget, simpleScoring);
        const duration = Date.now() - startTime;

        console.log(`Budget: ${budget.toString().padStart(3)} trials │ Score: ${result.bestScore.toFixed(3)} │ Time: ${duration.toString().padStart(4)}ms`);
    }

    console.log('\n💡 Insight: Diminishing returns after ~30-50 trials for most prompts.');
}

/**
 * Demo 5: Real-World Example - Code Generation Optimization
 */
async function demo5_RealWorld() {
    console.log('\n\n' + '='.repeat(80));
    console.log('DEMO 5: Real-World Code Generation Optimization');
    console.log('='.repeat(80));

    const realPrompts = [
        {
            name: 'Authentication System',
            prompt: 'Build user login'
        },
        {
            name: 'API Endpoint',
            prompt: 'Create REST endpoint'
        },
        {
            name: 'Database Query',
            prompt: 'Write SQL query'
        }
    ];

    console.log('\nOptimizing real-world code generation prompts:\n');

    for (const { name, prompt } of realPrompts) {
        console.log(`\n🎯 ${name}`);
        console.log(`   Original: "${prompt}"`);

        // Use both methods
        const banditResult = await banditOptimize(prompt, 40, codeQualityScoring);
        const mctsResult = await mctsOptimize(prompt, 25, 4, codeQualityScoring);

        // Show best overall
        const best = banditResult.bestScore > mctsResult.bestScore ? banditResult.bestPrompt : mctsResult.bestPrompt;
        const bestScore = Math.max(banditResult.bestScore, mctsResult.bestScore);
        const method = banditResult.bestScore > mctsResult.bestScore ? 'Bandit' : 'MCTS';

        console.log(`\n   ✨ Optimized (${method}): "${best.substring(0, 100)}${best.length > 100 ? '...' : ''}"`);
        console.log(`   📊 Quality Score: ${bestScore.toFixed(3)}/1.000`);
        console.log(`   📈 Improvement: ${((bestScore - await codeQualityScoring(prompt)) * 100).toFixed(1)}%`);
    }
}

/**
 * Demo 6: Exploration vs Exploitation Trade-off
 */
async function demo6_ExplorationExploitation() {
    console.log('\n\n' + '='.repeat(80));
    console.log('DEMO 6: Understanding Exploration vs Exploitation');
    console.log('='.repeat(80));

    const prompt = 'Write data validation function';

    console.log(`\nPrompt: "${prompt}"`);
    console.log('\nRunning Bandit with different budgets to show exploration pattern:\n');

    // Run with small budget (more exploration)
    console.log('🔍 Early Phase (Budget: 20) - High Exploration');
    const early = await banditOptimize(prompt, 20, simpleScoring);

    let totalPulls = 0;
    let variance = 0;
    for (const stats of Object.values(early.armStats)) {
        totalPulls += stats.pulls;
    }
    const avgPulls = totalPulls / Object.keys(early.armStats).length;
    for (const stats of Object.values(early.armStats)) {
        variance += Math.pow(stats.pulls - avgPulls, 2);
    }
    const earlyVariance = Math.sqrt(variance / Object.keys(early.armStats).length);

    console.log(`   Pull Distribution Variance: ${earlyVariance.toFixed(2)}`);
    console.log(`   (Lower = more balanced exploration)\n`);

    // Run with large budget (more exploitation)
    console.log('🎯 Late Phase (Budget: 100) - High Exploitation');
    const late = await banditOptimize(prompt, 100, simpleScoring);

    totalPulls = 0;
    variance = 0;
    for (const stats of Object.values(late.armStats)) {
        totalPulls += stats.pulls;
    }
    const avgPullsLate = totalPulls / Object.keys(late.armStats).length;
    for (const stats of Object.values(late.armStats)) {
        variance += Math.pow(stats.pulls - avgPullsLate, 2);
    }
    const lateVariance = Math.sqrt(variance / Object.keys(late.armStats).length);

    console.log(`   Pull Distribution Variance: ${lateVariance.toFixed(2)}`);
    console.log(`   (Higher = focused on best arms)\n`);

    console.log('💡 Insight: UCB1 naturally transitions from exploration to exploitation over time.');
}

// ============================================================================
// RUN ALL DEMOS
// ============================================================================

async function runAllDemos() {
    console.log('\n');
    console.log('╔' + '═'.repeat(78) + '╗');
    console.log('║' + ' '.repeat(15) + 'BANDITS & MCTS OPTIMIZER DEMOS' + ' '.repeat(32) + '║');
    console.log('║' + ' '.repeat(25) + 'DIRECTIVE-022' + ' '.repeat(40) + '║');
    console.log('╚' + '═'.repeat(78) + '╝');
    console.log('\n');

    try {
        await demo1_BanditUCB1();
        await demo2_MCTS();
        await demo3_Comparison();
        await demo4_BudgetAnalysis();
        await demo5_RealWorld();
        await demo6_ExplorationExploitation();

        console.log('\n\n' + '='.repeat(80));
        console.log('✅ ALL DEMOS COMPLETED SUCCESSFULLY');
        console.log('='.repeat(80));

        console.log('\n📚 Key Takeaways:');
        console.log('\n  🎰 Multi-Armed Bandits:');
        console.log('     • Fast and efficient for selecting best single mutation');
        console.log('     • UCB1 balances exploration and exploitation automatically');
        console.log('     • Ideal when you need quick results with limited budget');
        console.log('     • Best for: Finding which mutation type works best');

        console.log('\n  🌲 Monte Carlo Tree Search:');
        console.log('     • Explores sequences of mutations (chains)');
        console.log('     • Finds optimal paths through mutation space');
        console.log('     • More thorough but slower than Bandits');
        console.log('     • Best for: Discovering complex transformation sequences');

        console.log('\n  🎯 When to Use Which:');
        console.log('     • Bandit: Quick optimization, single-step improvement');
        console.log('     • MCTS: Deep optimization, multi-step transformations');
        console.log('     • Both: Run both and compare results!');

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
    runAllDemos,
    simpleScoring,
    codeQualityScoring,
};
