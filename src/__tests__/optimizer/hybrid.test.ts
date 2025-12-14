/**
 * DIRECTIVE-024: Hybrid Optimizer Unit Tests
 * 
 * Tests the 3-stage hybrid optimization system:
 * - Exploration (Genetic Algorithm)
 * - Refinement (Hill-Climbing)
 * - Fine-tuning (Bayesian Optimization)
 */

import { hybridOptimize, HybridConfig, HybridResult } from '../../optimizer/hybrid';
import { ScoringFunction } from '../../optimizer/types';

// ============================================================================
// MOCK SCORING FUNCTIONS
// ============================================================================

/**
 * Simple scoring function that favors longer prompts
 */
const lengthScoringFunction: ScoringFunction = async (prompt: string): Promise<number> => {
    return Math.min(100, prompt.length / 5);
};

/**
 * Scoring function that favors prompts with specific keywords
 */
const keywordScoringFunction: ScoringFunction = async (prompt: string): Promise<number> => {
    let score = 50;

    const keywords = ['specific', 'detailed', 'comprehensive', 'thorough'];
    keywords.forEach(keyword => {
        if (prompt.toLowerCase().includes(keyword)) {
            score += 12.5;
        }
    });

    return score;
};

/**
 * Deterministic scoring function for testing
 */
const deterministicScoringFunction: ScoringFunction = async (prompt: string): Promise<number> => {
    // Score based on character count (fixed formula)
    const charScore = (prompt.length % 100) / 2;

    // Score based on word count
    const wordCount = prompt.split(/\s+/).length;
    const wordScore = Math.min(30, wordCount * 2);

    return charScore + wordScore;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function createBasicConfig(overrides?: Partial<HybridConfig>): HybridConfig {
    return {
        explorationBudget: 2,
        refinementBudget: 3,
        finetuningBudget: 5,
        ...overrides
    };
}

// ============================================================================
// TESTS
// ============================================================================

describe('DIRECTIVE-024: Hybrid Optimizer', () => {

    // ========================================================================
    // BASIC FUNCTIONALITY TESTS
    // ========================================================================

    describe('Basic Functionality', () => {

        test('should return HybridResult with required fields', async () => {
            const initialPrompt = "Write a simple test function";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                lengthScoringFunction,
                config
            );

            // Check result structure
            expect(result).toHaveProperty('finalPrompt');
            expect(result).toHaveProperty('finalScore');
            expect(result).toHaveProperty('trace');

            // Check types
            expect(typeof result.finalPrompt).toBe('string');
            expect(typeof result.finalScore).toBe('number');
            expect(Array.isArray(result.trace)).toBe(true);
        });

        test('should include all three stages in trace', async () => {
            const initialPrompt = "Create a user authentication system";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                keywordScoringFunction,
                config
            );

            // Should have 3 trace entries (one per stage)
            expect(result.trace.length).toBe(3);

            // Check stage names
            expect(result.trace[0].stage).toBe('exploration');
            expect(result.trace[1].stage).toBe('refinement');
            expect(result.trace[2].stage).toBe('finetuning');
        });

        test('should produce non-empty final prompt', async () => {
            const initialPrompt = "Simple prompt";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                lengthScoringFunction,
                config
            );

            expect(result.finalPrompt.length).toBeGreaterThan(0);
        });

    });

    // ========================================================================
    // STAGE PROGRESSION TESTS
    // ========================================================================

    describe('Stage Progression', () => {

        test('should show progress through all stages', async () => {
            const initialPrompt = "Generate code for sorting";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                deterministicScoringFunction,
                config
            );

            // Each stage should have a prompt
            result.trace.forEach(entry => {
                expect(entry.prompt).toBeTruthy();
                expect(typeof entry.prompt).toBe('string');
                expect(entry.prompt.length).toBeGreaterThan(0);
            });
        });

        test('should include stage details in trace', async () => {
            const initialPrompt = "Build a REST API";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                keywordScoringFunction,
                config
            );

            // Exploration stage should have details
            expect(result.trace[0].details).toBeTruthy();
            expect(result.trace[0].details.generations).toBe(config.explorationBudget);

            // Refinement stage should have details
            expect(result.trace[1].details).toBeTruthy();
            expect(result.trace[1].details.iterations).toBe(config.refinementBudget);

            // Fine-tuning stage should have details (may be null if skipped)
            expect(result.trace[2]).toBeTruthy();
        });

    });

    // ========================================================================
    // OPTIMIZATION QUALITY TESTS
    // ========================================================================

    describe('Optimization Quality', () => {

        test('final score should be a valid number', async () => {
            const initialPrompt = "Analyze data trends";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                lengthScoringFunction,
                config
            );

            expect(typeof result.finalScore).toBe('number');
            expect(result.finalScore).toBeGreaterThanOrEqual(0);
            expect(result.finalScore).toBeLessThanOrEqual(100);
            expect(isNaN(result.finalScore)).toBe(false);
        });

        test('should handle different config budgets', async () => {
            const initialPrompt = "Design a database schema";

            // Minimal budget
            const minimalConfig: HybridConfig = {
                explorationBudget: 1,
                refinementBudget: 1,
                finetuningBudget: 1
            };

            const result = await hybridOptimize(
                initialPrompt,
                deterministicScoringFunction,
                minimalConfig
            );

            expect(result.finalPrompt).toBeTruthy();
            expect(result.trace.length).toBe(3);
        });

        test('should handle larger budgets', async () => {
            const initialPrompt = "Implement caching mechanism";

            const largerConfig: HybridConfig = {
                explorationBudget: 4,
                refinementBudget: 8,
                finetuningBudget: 12
            };

            const result = await hybridOptimize(
                initialPrompt,
                keywordScoringFunction,
                largerConfig
            );

            expect(result.finalPrompt).toBeTruthy();
            expect(result.finalScore).toBeGreaterThan(0);
        }, 30000); // Longer timeout for larger budget

    });

    // ========================================================================
    // EDGE CASES
    // ========================================================================

    describe('Edge Cases', () => {

        test('should handle very short initial prompt', async () => {
            const initialPrompt = "Help";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                lengthScoringFunction,
                config
            );

            expect(result.finalPrompt).toBeTruthy();
            expect(result.finalPrompt.length).toBeGreaterThan(0);
        });

        test('should handle very long initial prompt', async () => {
            const longPrompt = "Write a comprehensive function that " +
                "handles user authentication ".repeat(20);
            const config = createBasicConfig();

            const result = await hybridOptimize(
                longPrompt,
                keywordScoringFunction,
                config
            );

            expect(result.finalPrompt).toBeTruthy();
            expect(result.trace.length).toBe(3);
        });

        test('should handle prompts with special characters', async () => {
            const specialPrompt = "Create function: calculateResult() { return x * 2; }";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                specialPrompt,
                deterministicScoringFunction,
                config
            );

            expect(result.finalPrompt).toBeTruthy();
        });

    });

    // ========================================================================
    // INTEGRATION TESTS
    // ========================================================================

    describe('Integration', () => {

        test('should integrate all three optimizers', async () => {
            const initialPrompt = "Build a recommendation system";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                keywordScoringFunction,
                config
            );

            // Verify all stages executed
            const stageNames = result.trace.map(t => t.stage);
            expect(stageNames).toEqual(['exploration', 'refinement', 'finetuning']);

            // Verify each stage produced a score
            result.trace.forEach(entry => {
                expect(typeof entry.score).toBe('number');
                expect(entry.score).toBeGreaterThanOrEqual(0);
            });
        });

        test('should maintain continuity between stages', async () => {
            const initialPrompt = "Design API endpoints";
            const config = createBasicConfig();

            const result = await hybridOptimize(
                initialPrompt,
                lengthScoringFunction,
                config
            );

            // Each stage should build on the previous
            expect(result.trace[0].prompt).toBeTruthy(); // Exploration output
            expect(result.trace[1].prompt).toBeTruthy(); // Refinement output
            expect(result.trace[2].prompt).toBeTruthy(); // Fine-tuning output

            // Final prompt should be last stage's output
            expect(result.finalPrompt).toBe(result.trace[2].prompt);
        });

    });

    // ========================================================================
    // PERFORMANCE TESTS
    // ========================================================================

    describe('Performance', () => {

        test('should complete within reasonable time for small budget', async () => {
            const initialPrompt = "Optimize database queries";
            const config = createBasicConfig({
                explorationBudget: 1,
                refinementBudget: 2,
                finetuningBudget: 3
            });

            const startTime = Date.now();

            await hybridOptimize(
                initialPrompt,
                deterministicScoringFunction,
                config
            );

            const duration = Date.now() - startTime;

            // Should complete in under 10 seconds
            expect(duration).toBeLessThan(10000);
        });

    });

});

// ============================================================================
// EXPORTS FOR MANUAL TESTING
// ============================================================================

export {
    lengthScoringFunction,
    keywordScoringFunction,
    deterministicScoringFunction,
    createBasicConfig
};
