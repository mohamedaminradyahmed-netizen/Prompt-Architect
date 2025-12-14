


/**
 * Test Suite Executor (DIRECTIVE-025)
 * 
 * System for running prompts on parallel test cases.
 */

export interface EvaluationCriteria {
    // Flexible criteria definition
    matchType?: 'exact' | 'includes' | 'regex' | 'custom';
    matchValue?: string | RegExp;
    customValidator?: (output: string) => boolean;
    minScore?: number;
}

export interface TestCase {
    id: string;
    prompt: string;
    expectedOutput?: string;
    evaluationCriteria: EvaluationCriteria;
    metadata?: Record<string, any>;
}

export interface TestCaseResult {
    testCaseId: string;
    output: string;
    passed: boolean;
    score: number;
    latency: number;
    error?: string;
}

export interface TestResults {
    variationId: string;
    results: TestCaseResult[];
    aggregateScore: number;
    passRate: number;
}

// LLM Executor type definition
export type LLMExecutor = (prompt: string) => Promise<string>;

/**
 * Execute a test suite across multiple prompt variations
 * 
 * @param promptVariations List of prompt variations to test
 * @param testCases List of test cases to run against each variation
 * @param executor Function to execute the LLM call
 * @param maxConcurrency Maximum number of concurrent tests per variation
 */
export async function executeTestSuite(
    promptVariations: string[],
    testCases: TestCase[],
    executor: LLMExecutor,
    maxConcurrency: number = 5
): Promise<TestResults[]> {

    // Helper for concurrency control
    const limit = (fn: () => Promise<any>) => fn(); // Placeholder, we will implement simple queue logic below

    const allResults: TestResults[] = [];

    for (const variation of promptVariations) {
        const variationResults: TestCaseResult[] = [];
        const queue = [...testCases];
        const activePromises: Promise<void>[] = [];

        const runNext = async (): Promise<void> => {
            if (queue.length === 0) return;
            const testCase = queue.shift()!;

            try {
                // Combine variation and test case prompt
                // Assuming variation is the system/instruction and testCase.prompt is user input
                const fullPrompt = `${variation}\n\n${testCase.prompt}`;

                const start = Date.now();
                const output = await executor(fullPrompt);
                const latency = Date.now() - start;

                const { passed, score } = evaluateResult(output, testCase);

                variationResults.push({
                    testCaseId: testCase.id,
                    output,
                    passed,
                    score,
                    latency
                });

            } catch (error: any) {
                variationResults.push({
                    testCaseId: testCase.id,
                    output: '',
                    passed: false,
                    score: 0,
                    latency: 0,
                    error: error.message || 'Execution failed'
                });
            }

            if (queue.length > 0) {
                await runNext();
            }
        };

        // Start initial batch
        const workers = Array(Math.min(maxConcurrency, testCases.length))
            .fill(null)
            .map(() => runNext());

        await Promise.all(workers);

        // Aggregate results
        const passedCount = variationResults.filter(r => r.passed).length;
        const totalScore = variationResults.reduce((sum, r) => sum + r.score, 0);
        const avgScore = variationResults.length > 0 ? totalScore / variationResults.length : 0;
        const passRate = variationResults.length > 0 ? passedCount / variationResults.length : 0;

        allResults.push({
            variationId: variation, // Using the prompt text itself as ID if string provided
            results: variationResults,
            aggregateScore: avgScore,
            passRate: passRate
        });
    }

    return allResults;
}

function evaluateResult(output: string, testCase: TestCase): { passed: boolean; score: number } {
    const criteria = testCase.evaluationCriteria;

    // Default pass
    let passed = true;
    let score = 1.0;

    if (criteria.matchType === 'exact' && criteria.matchValue) {
        passed = output.trim() === (criteria.matchValue as string).trim();
    } else if (criteria.matchType === 'includes' && criteria.matchValue) {
        passed = output.includes(criteria.matchValue as string);
    } else if (criteria.matchType === 'regex' && criteria.matchValue) {
        const regex = new RegExp(criteria.matchValue);
        passed = regex.test(output);
    } else if (criteria.customValidator) {
        passed = criteria.customValidator(output);
    }

    // If expected output is provided but no specific criteria, do simple containment check
    if (!criteria.matchType && testCase.expectedOutput) {
        passed = output.includes(testCase.expectedOutput);
    }

    if (!passed) score = 0;

    return { passed, score };
}
