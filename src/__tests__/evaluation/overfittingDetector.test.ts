/**
 * Unit Tests for Overfitting Detector
 * DIRECTIVE-038
 */

import {
    detectOverfitting,
    kFoldCrossValidation,
    splitDataset,
    heldOutValidation,
    analyzeDiversity,
    calculateRegularization,
    simplifyPrompt,
    comprehensiveOverfittingAnalysis,
    OverfittingConfig
} from '../../evaluation/overfittingDetector';
import { TestCase, TestResults, LLMExecutor } from '../../sandbox/testExecutor';

// ==================== Mock Data & Helpers ====================

const createMockTestResults = (
    scores: number[],
    variationId: string = 'test-variation'
): TestResults => {
    return {
        variationId,
        results: scores.map((score, idx) => ({
            testCaseId: `test-${idx}`,
            output: 'mock output',
            passed: score >= 0.5,
            score,
            latency: 100
        })),
        aggregateScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        passRate: scores.filter(s => s >= 0.5).length / scores.length
    };
};

const createMockTestCases = (count: number, categories?: string[]): TestCase[] => {
    return Array.from({ length: count }, (_, i) => ({
        id: `test-${i}`,
        prompt: `Test prompt ${i}`,
        evaluationCriteria: {
            matchType: 'includes' as const,
            matchValue: 'success'
        },
        metadata: {
            category: categories ? categories[i % categories.length] : `category-${i % 3}`
        }
    }));
};

const createMockExecutor = (baseScore: number = 0.8, variance: number = 0.1): LLMExecutor => {
    return async (prompt: string): Promise<string> => {
        const randomScore = baseScore + (Math.random() - 0.5) * variance;
        return randomScore > 0.5 ? 'success' : 'failure';
    };
};

// ==================== Tests for detectOverfitting ====================

describe('detectOverfitting', () => {
    test('should detect no overfitting when scores are similar', async () => {
        const trainResults = createMockTestResults([0.85, 0.87, 0.86, 0.84, 0.88]);
        const valResults = createMockTestResults([0.84, 0.86, 0.85, 0.83, 0.87]);
        const prompt = 'Simple test prompt';

        const report = await detectOverfitting(prompt, trainResults, valResults);

        expect(report.isOverfit).toBe(false);
        expect(report.gap).toBeLessThan(0.05);
        expect(report.severity).toBe('none');
    });

    test('should detect mild overfitting', async () => {
        const trainResults = createMockTestResults([0.90, 0.92, 0.91, 0.89, 0.93]);
        const valResults = createMockTestResults([0.75, 0.78, 0.76, 0.74, 0.77]);
        const prompt = 'Test prompt';

        const report = await detectOverfitting(prompt, trainResults, valResults);

        expect(report.isOverfit).toBe(true);
        expect(report.gap).toBeGreaterThan(0.10);
        expect(report.gap).toBeLessThan(0.20);
        expect(report.severity).toBe('mild');
    });

    test('should detect moderate overfitting', async () => {
        const trainResults = createMockTestResults([0.95, 0.96, 0.94, 0.97, 0.95]);
        const valResults = createMockTestResults([0.70, 0.68, 0.72, 0.69, 0.71]);
        const prompt = 'Test prompt';

        const report = await detectOverfitting(prompt, trainResults, valResults);

        expect(report.isOverfit).toBe(true);
        expect(report.gap).toBeGreaterThan(0.20);
        expect(report.gap).toBeLessThan(0.30);
        expect(report.severity).toBe('moderate');
    });

    test('should detect severe overfitting', async () => {
        const trainResults = createMockTestResults([0.98, 0.99, 0.97, 1.00, 0.98]);
        const valResults = createMockTestResults([0.50, 0.48, 0.52, 0.49, 0.51]);
        const prompt = 'Test prompt';

        const report = await detectOverfitting(prompt, trainResults, valResults);

        expect(report.isOverfit).toBe(true);
        expect(report.gap).toBeGreaterThan(0.30);
        expect(report.severity).toBe('severe');
    });

    test('should respect custom threshold', async () => {
        const trainResults = createMockTestResults([0.85, 0.87, 0.86]);
        const valResults = createMockTestResults([0.78, 0.80, 0.79]);
        const prompt = 'Test prompt';

        const config: OverfittingConfig = {
            gapThreshold: 0.05
        };

        const report = await detectOverfitting(prompt, trainResults, valResults, config);

        expect(report.isOverfit).toBe(true); // 7% gap > 5% threshold
    });

    test('should analyze complexity correctly', async () => {
        const trainResults = createMockTestResults([0.85]);
        const valResults = createMockTestResults([0.83]);

        const simplePrompt = 'Write a function';
        const complexPrompt = 'A'.repeat(5000); // Very long prompt

        const simpleReport = await detectOverfitting(simplePrompt, trainResults, valResults);
        const complexReport = await detectOverfitting(complexPrompt, trainResults, valResults);

        expect(simpleReport.analysis.complexityAnalysis.isOverlyComplex).toBe(false);
        expect(complexReport.analysis.complexityAnalysis.isOverlyComplex).toBe(true);
        expect(complexReport.analysis.complexityAnalysis.tokenCount).toBeGreaterThan(1000);
    });

    test('should include recommendations', async () => {
        const trainResults = createMockTestResults([0.95, 0.96, 0.97]);
        const valResults = createMockTestResults([0.60, 0.62, 0.61]);
        const prompt = 'Test prompt';

        const report = await detectOverfitting(prompt, trainResults, valResults);

        expect(report.recommendation).toBeTruthy();
        expect(report.recommendation.length).toBeGreaterThan(50);
        expect(report.recommendation).toContain('قلل');
    });
});

// ==================== Tests for K-Fold Cross Validation ====================

describe('kFoldCrossValidation', () => {
    test('should perform 5-fold validation', async () => {
        const testCases = createMockTestCases(25);
        const executor = createMockExecutor(0.8);
        const prompt = 'Test prompt';

        const result = await kFoldCrossValidation(prompt, testCases, executor, 5);

        expect(result.folds).toBe(5);
        expect(result.foldScores).toHaveLength(5);
        expect(result.meanScore).toBeGreaterThan(0);
        expect(result.meanScore).toBeLessThanOrEqual(1);
        expect(result.stdDeviation).toBeGreaterThanOrEqual(0);
    });

    test('should identify best and worst folds', async () => {
        const testCases = createMockTestCases(20);
        const executor = createMockExecutor(0.75);
        const prompt = 'Test prompt';

        const result = await kFoldCrossValidation(prompt, testCases, executor, 4);

        expect(result.bestFold).toBeGreaterThanOrEqual(0);
        expect(result.bestFold).toBeLessThan(4);
        expect(result.worstFold).toBeGreaterThanOrEqual(0);
        expect(result.worstFold).toBeLessThan(4);

        const bestScore = result.foldScores[result.bestFold];
        const worstScore = result.foldScores[result.worstFold];
        expect(bestScore).toBeGreaterThanOrEqual(worstScore);
    });

    test('should determine stability correctly', async () => {
        const testCases = createMockTestCases(30);

        // Stable executor (low variance)
        const stableExecutor = createMockExecutor(0.8, 0.05);
        const stableResult = await kFoldCrossValidation('prompt', testCases, stableExecutor);

        // Unstable executor (high variance)
        const unstableExecutor = createMockExecutor(0.6, 0.8);
        const unstableResult = await kFoldCrossValidation('prompt', testCases, unstableExecutor);

        // Stable should have lower std deviation
        expect(stableResult.stdDeviation).toBeLessThan(unstableResult.stdDeviation);
    });

    test('should throw error if k < 2', async () => {
        const testCases = createMockTestCases(10);
        const executor = createMockExecutor(0.8);

        await expect(
            kFoldCrossValidation('prompt', testCases, executor, 1)
        ).rejects.toThrow('K must be at least 2');
    });

    test('should throw error if not enough test cases', async () => {
        const testCases = createMockTestCases(3);
        const executor = createMockExecutor(0.8);

        await expect(
            kFoldCrossValidation('prompt', testCases, executor, 5)
        ).rejects.toThrow('Not enough test cases');
    });
});

// ==================== Tests for Dataset Splitting ====================

describe('splitDataset', () => {
    test('should split dataset with default ratios', () => {
        const testCases = createMockTestCases(100);
        const { train, validation, test } = splitDataset(testCases);

        expect(train.length).toBe(60); // 60%
        expect(validation.length).toBe(20); // 20%
        expect(test.length).toBe(20); // 20%

        // Total should match
        expect(train.length + validation.length + test.length).toBe(100);
    });

    test('should split with custom ratios', () => {
        const testCases = createMockTestCases(100);
        const { train, validation, test } = splitDataset(testCases, 0.7, 0.15);

        expect(train.length).toBe(70);
        expect(validation.length).toBe(15);
        expect(test.length).toBe(15);
    });

    test('should not have overlapping samples', () => {
        const testCases = createMockTestCases(50);
        const { train, validation, test } = splitDataset(testCases);

        const allIds = new Set([
            ...train.map(t => t.id),
            ...validation.map(t => t.id),
            ...test.map(t => t.id)
        ]);

        expect(allIds.size).toBe(50); // No duplicates
    });
});

// ==================== Tests for Held-out Validation ====================

describe('heldOutValidation', () => {
    test('should return scores for all three sets', async () => {
        const testCases = createMockTestCases(30);
        const executor = createMockExecutor(0.8);
        const prompt = 'Test prompt';

        const result = await heldOutValidation(prompt, testCases, executor);

        expect(result.trainScore).toBeGreaterThan(0);
        expect(result.valScore).toBeGreaterThan(0);
        expect(result.testScore).toBeGreaterThan(0);
        expect(result.trainTestGap).toBeDefined();
        expect(result.generalizationScore).toBeGreaterThanOrEqual(0);
        expect(result.generalizationScore).toBeLessThanOrEqual(1);
    });

    test('should calculate generalization score correctly', async () => {
        const testCases = createMockTestCases(40);
        const executor = createMockExecutor(0.85);
        const prompt = 'Test prompt';

        const result = await heldOutValidation(prompt, testCases, executor);

        // If gap is small, generalization should be high
        if (Math.abs(result.trainTestGap) < 0.1) {
            expect(result.generalizationScore).toBeGreaterThan(0.8);
        }
    });
});

// ==================== Tests for Diversity Analysis ====================

describe('analyzeDiversity', () => {
    test('should calculate diversity for uniform distribution', () => {
        const testCases = createMockTestCases(12, ['A', 'B', 'C', 'D']);
        const diversity = analyzeDiversity(testCases);

        expect(diversity.uniqueCategories).toBe(4);
        expect(diversity.diversityScore).toBeGreaterThan(0.9); // High entropy
        expect(diversity.categoryDistribution.size).toBe(4);
    });

    test('should calculate low diversity for skewed distribution', () => {
        // 9 of category A, 1 of B
        const testCases = [
            ...createMockTestCases(9, ['A']),
            ...createMockTestCases(1, ['B'])
        ];

        const diversity = analyzeDiversity(testCases);

        expect(diversity.uniqueCategories).toBe(2);
        expect(diversity.diversityScore).toBeLessThan(0.5); // Low entropy
    });

    test('should identify sufficient diversity', () => {
        const diverseCases = createMockTestCases(20, ['A', 'B', 'C', 'D', 'E']);
        const notDiverseCases = createMockTestCases(20, ['A', 'B']);

        const diverseAnalysis = analyzeDiversity(diverseCases);
        const notDiverseAnalysis = analyzeDiversity(notDiverseCases);

        expect(diverseAnalysis.isSufficientlyDiverse).toBe(true);
        expect(notDiverseAnalysis.isSufficientlyDiverse).toBe(false);
    });

    test('should handle uncategorized data', () => {
        const testCases = createMockTestCases(10).map(tc => ({
            ...tc,
            metadata: {}
        }));

        const diversity = analyzeDiversity(testCases);

        expect(diversity.uniqueCategories).toBe(1);
        expect(diversity.categoryDistribution.has('uncategorized')).toBe(true);
    });
});

// ==================== Tests for Regularization ====================

describe('calculateRegularization', () => {
    test('should penalize longer prompts more', () => {
        const shortPrompt = 'Short';
        const longPrompt = 'A'.repeat(1000);

        const shortPenalty = calculateRegularization(shortPrompt);
        const longPenalty = calculateRegularization(longPrompt);

        expect(longPenalty).toBeGreaterThan(shortPenalty);
    });

    test('should respect lambda parameter', () => {
        const prompt = 'Test prompt with some text';

        const lowLambda = calculateRegularization(prompt, 0.0001);
        const highLambda = calculateRegularization(prompt, 0.01);

        expect(highLambda).toBeGreaterThan(lowLambda);
    });

    test('should return positive penalty', () => {
        const prompt = 'Any prompt';
        const penalty = calculateRegularization(prompt);

        expect(penalty).toBeGreaterThan(0);
    });
});

// ==================== Tests for Prompt Simplification ====================

describe('simplifyPrompt', () => {
    test('should reduce prompt length', () => {
        const prompt = `
Line 1: Important instruction
Line 2: Example: This is an example
Line 3: Note: Some explanation
Line 4: Another instruction
Line 5: For instance, here is more
Line 6: Final instruction
`.trim();

        const simplified = simplifyPrompt(prompt, 0.3);

        expect(simplified.length).toBeLessThan(prompt.length);
    });

    test('should remove examples', () => {
        const prompt = `
Write a function.
Example: function add(a, b) { return a + b; }
Make it efficient.
`;

        const simplified = simplifyPrompt(prompt);

        expect(simplified).not.toContain('Example:');
    });

    test('should remove explanations', () => {
        const prompt = `
Do task A.
Note: This is important because...
Do task B.
Explanation: The reason is...
`;

        const simplified = simplifyPrompt(prompt);

        expect(simplified).not.toContain('Note:');
        expect(simplified).not.toContain('Explanation:');
    });

    test('should respect target reduction', () => {
        const prompt = Array.from({ length: 20 }, (_, i) => `Line ${i}`).join('\n');

        const reduced30 = simplifyPrompt(prompt, 0.3);
        const reduced50 = simplifyPrompt(prompt, 0.5);

        expect(reduced50.length).toBeLessThan(reduced30.length);
    });

    test('should preserve at least minimum lines', () => {
        const prompt = Array.from({ length: 10 }, (_, i) => `Line ${i}`).join('\n');

        const simplified = simplifyPrompt(prompt, 0.9); // Request 90% reduction

        // Should keep at least 5 lines
        expect(simplified.split('\n').length).toBeGreaterThanOrEqual(5);
    });
});

// ==================== Tests for Comprehensive Analysis ====================

describe('comprehensiveOverfittingAnalysis', () => {
    test('should return all analysis components', async () => {
        const testCases = createMockTestCases(30, ['A', 'B', 'C']);
        const executor = createMockExecutor(0.8);
        const prompt = 'Test prompt';

        const analysis = await comprehensiveOverfittingAnalysis(
            prompt,
            testCases,
            executor
        );

        expect(analysis.overfittingReport).toBeDefined();
        expect(analysis.crossValidation).toBeDefined();
        expect(analysis.heldOutValidation).toBeDefined();
        expect(analysis.diversityAnalysis).toBeDefined();
        expect(analysis.regularizationPenalty).toBeDefined();

        expect(analysis.overfittingReport.isOverfit).toBeDefined();
        expect(analysis.crossValidation.folds).toBe(5);
        expect(analysis.diversityAnalysis.uniqueCategories).toBe(3);
    });

    test('should generate simplified prompt for complex prompts', async () => {
        const testCases = createMockTestCases(20);
        const executor = createMockExecutor(0.8);
        const complexPrompt = 'A'.repeat(5000); // Very long

        const analysis = await comprehensiveOverfittingAnalysis(
            complexPrompt,
            testCases,
            executor
        );

        expect(analysis.simplifiedPrompt).toBeDefined();
        expect(analysis.simplifiedPrompt!.length).toBeLessThan(complexPrompt.length);
    });

    test('should not generate simplified prompt for simple prompts', async () => {
        const testCases = createMockTestCases(20);
        const executor = createMockExecutor(0.8);
        const simplePrompt = 'Write a function';

        const analysis = await comprehensiveOverfittingAnalysis(
            simplePrompt,
            testCases,
            executor
        );

        expect(analysis.simplifiedPrompt).toBeUndefined();
    });
});

// ==================== Integration Tests ====================

describe('Integration Tests', () => {
    test('complete workflow: detect and fix overfitting', async () => {
        // 1. Create a complex, overfitted prompt
        const overfittedPrompt = `
You are an expert assistant.
Follow these 50 rules exactly as specified...
${'Rule: Do something specific.\n'.repeat(50)}
Example: ${'Here is a detailed example.\n'.repeat(10)}
Note: ${'Important explanation here.\n'.repeat(10)}
`;

        // 2. Create diverse test cases
        const testCases = createMockTestCases(40, ['code', 'content', 'analysis', 'marketing']);

        // 3. Create executor that shows overfitting
        const executor: LLMExecutor = async (prompt: string) => {
            // Performs well on long prompts (training), poorly on short (validation)
            const score = prompt.length > 500 ? 0.95 : 0.60;
            return score > 0.7 ? 'success' : 'failure';
        };

        // 4. Run comprehensive analysis
        const analysis = await comprehensiveOverfittingAnalysis(
            overfittedPrompt,
            testCases,
            executor
        );

        // 5. Verify overfitting detected
        expect(analysis.overfittingReport.isOverfit).toBe(true);
        expect(analysis.overfittingReport.severity).not.toBe('none');

        // 6. Verify simplified prompt generated
        expect(analysis.simplifiedPrompt).toBeDefined();
        expect(analysis.simplifiedPrompt!.length).toBeLessThan(overfittedPrompt.length);

        // 7. Verify diversity is good
        expect(analysis.diversityAnalysis.uniqueCategories).toBeGreaterThanOrEqual(3);

        // 8. Verify recommendation provided
        expect(analysis.overfittingReport.recommendation).toContain('بسّط');
    });
});
