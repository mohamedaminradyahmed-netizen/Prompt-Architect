/**
 * DIRECTIVE-024: Hybrid Optimizer Demo
 * 
 * This demonstrates the 3-stage hybrid optimization approach:
 * 1. Exploration (Genetic Algorithm) - Global search for diverse solutions
 * 2. Refinement (Hill-Climbing) - Local optimization of best candidates
 * 3. Fine-tuning (Bayesian Optimization) - Parameter optimization
 */

import { hybridOptimize, HybridConfig } from './hybrid';
import { ScoringFunction } from './types';

// ============================================================================
// DEMO SCORING FUNCTION
// ============================================================================

/**
 * Demo scoring function that evaluates prompts based on:
 * - Length (moderate is better)
 * - Clarity keywords
 * - Structure (presence of role, goal, constraints)
 */
const demoScoringFunction: ScoringFunction = async (prompt: string): Promise<number> => {
    let score = 50; // Base score

    // Length scoring (sweet spot: 100-300 characters)
    const length = prompt.length;
    if (length >= 100 && length <= 300) {
        score += 20;
    } else if (length > 300 && length <= 500) {
        score += 10;
    } else if (length > 500) {
        score -= 10;
    }

    // Clarity keywords
    const clarityKeywords = ['specific', 'clear', 'detailed', 'precise', 'exactly'];
    const hasClarity = clarityKeywords.some(kw => prompt.toLowerCase().includes(kw));
    if (hasClarity) score += 15;

    // Structure keywords
    if (prompt.toLowerCase().includes('you are')) score += 10; // Has role
    if (prompt.match(/must|should|need to/i)) score += 10; // Has constraints
    if (prompt.includes('Example') || prompt.includes('example')) score += 10; // Has examples

    // Action verbs (good prompts are actionable)
    const actionVerbs = ['create', 'write', 'generate', 'build', 'design', 'implement'];
    const hasAction = actionVerbs.some(verb => prompt.toLowerCase().includes(verb));
    if (hasAction) score += 15;

    // Avoid filler words
    const fillerWords = ['maybe', 'perhaps', 'possibly', 'might'];
    const hasFiller = fillerWords.some(word => prompt.toLowerCase().includes(word));
    if (hasFiller) score -= 10;

    return Math.max(0, Math.min(100, score)); // Clamp to 0-100
};

// ============================================================================
// DEMO SCENARIOS
// ============================================================================

async function runDemo() {
    console.log("=".repeat(80));
    console.log("DIRECTIVE-024: HYBRID OPTIMIZER DEMONSTRATION");
    console.log("=".repeat(80));
    console.log();

    // Test prompts
    const testPrompts = [
        {
            name: "Simple Code Generation",
            prompt: "Write a function to calculate factorial"
        },
        {
            name: "Vague Marketing Request",
            prompt: "Make some content for our product"
        },
        {
            name: "Complex Task",
            prompt: "Build a user authentication system with email verification and password reset"
        }
    ];

    // Hybrid configuration
    const hybridConfig: HybridConfig = {
        explorationBudget: 3,    // 3 generations for genetic algorithm
        refinementBudget: 5,      // 5 iterations for hill climbing
        finetuningBudget: 10      // 10 iterations for bayesian optimization
    };

    for (const test of testPrompts) {
        console.log("\n" + "─".repeat(80));
        console.log(`📝 Test Case: ${test.name}`);
        console.log("─".repeat(80));
        console.log(`Original Prompt: "${test.prompt}"`);
        console.log();

        try {
            const startTime = Date.now();

            const result = await hybridOptimize(
                test.prompt,
                demoScoringFunction,
                hybridConfig
            );

            const duration = ((Date.now() - startTime) / 1000).toFixed(2);

            console.log("\n✅ Optimization Complete!");
            console.log(`⏱️  Duration: ${duration}s`);
            console.log();

            // Display optimization trace
            console.log("📊 Optimization Trace:");
            console.log();

            result.trace.forEach((entry, index) => {
                console.log(`  ${index + 1}. ${entry.stage.toUpperCase()}`);
                console.log(`     Score: ${entry.score.toFixed(2)}/100`);
                console.log(`     Prompt: "${entry.prompt.substring(0, 100)}${entry.prompt.length > 100 ? '...' : ''}"`);
                if (entry.details) {
                    console.log(`     Details:`, entry.details);
                }
                console.log();
            });

            // Final result
            console.log("🎯 FINAL RESULT:");
            console.log(`   Score: ${result.finalScore.toFixed(2)}/100`);
            console.log(`   Prompt: "${result.finalPrompt}"`);
            console.log();

            // Calculate improvement
            const originalScore = await demoScoringFunction(test.prompt);
            const improvement = ((result.finalScore - originalScore) / originalScore * 100).toFixed(1);
            console.log(`📈 Improvement: ${improvement}% (${originalScore.toFixed(2)} → ${result.finalScore.toFixed(2)})`);

        } catch (error) {
            console.error("❌ Error during optimization:", error);
        }
    }

    console.log();
    console.log("=".repeat(80));
    console.log("DEMONSTRATION COMPLETE");
    console.log("=".repeat(80));
}

// ============================================================================
// DETAILED EXAMPLE WITH CUSTOM CONFIG
// ============================================================================

async function runDetailedExample() {
    console.log("\n\n");
    console.log("=".repeat(80));
    console.log("DETAILED EXAMPLE: Custom Configuration");
    console.log("=".repeat(80));
    console.log();

    const detailedPrompt = "Optimize code for better performance";

    // More aggressive optimization
    const aggressiveConfig: HybridConfig = {
        explorationBudget: 5,     // More generations for better exploration
        refinementBudget: 10,     // More iterations for thorough refinement
        finetuningBudget: 15      // More iterations for fine-tuning
    };

    console.log(`Original Prompt: "${detailedPrompt}"`);
    console.log();
    console.log("Configuration:");
    console.log(`  - Exploration Budget: ${aggressiveConfig.explorationBudget} generations`);
    console.log(`  - Refinement Budget: ${aggressiveConfig.refinementBudget} iterations`);
    console.log(`  - Fine-tuning Budget: ${aggressiveConfig.finetuningBudget} iterations`);
    console.log();

    const startTime = Date.now();
    const result = await hybridOptimize(
        detailedPrompt,
        demoScoringFunction,
        aggressiveConfig
    );
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);

    console.log(`\n✅ Completed in ${duration}s`);
    console.log();
    console.log("Stage-by-Stage Progress:");

    result.trace.forEach((entry, idx) => {
        const stageEmoji = {
            'exploration': '🌍',
            'refinement': '🔧',
            'finetuning': '⚡'
        }[entry.stage] || '📝';

        console.log(`\n${stageEmoji} Stage ${idx + 1}: ${entry.stage.toUpperCase()}`);
        console.log(`   Score: ${entry.score.toFixed(2)}/100`);
        console.log(`   Length: ${entry.prompt.length} chars`);
        console.log(`   Prompt: ${entry.prompt}`);
    });

    console.log();
    console.log("🏆 FINAL OPTIMIZED PROMPT:");
    console.log(`   "${result.finalPrompt}"`);
    console.log(`   Score: ${result.finalScore.toFixed(2)}/100`);
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
    try {
        // Run basic demo
        await runDemo();

        // Run detailed example
        await runDetailedExample();

        console.log("\n\n✅ All demonstrations completed successfully!");
        console.log("\nKey Takeaways:");
        console.log("  • Hybrid optimization combines multiple strategies");
        console.log("  • Genetic algorithm explores diverse solutions");
        console.log("  • Hill climbing refines the best candidates");
        console.log("  • Bayesian optimization fine-tunes parameters");
        console.log("  • The 3-stage approach balances exploration and exploitation");

    } catch (error) {
        console.error("\n❌ Demo failed:", error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

export { runDemo, runDetailedExample, demoScoringFunction };
