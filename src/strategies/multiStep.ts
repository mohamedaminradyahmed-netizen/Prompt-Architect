/**
 * Multi-Step Prompts System
 *
 * Manages prompts that require sequential or conditional execution steps.
 */

export interface PromptStep {
    id: number;
    prompt: string;
    expectedOutputType: 'code' | 'text' | 'json' | 'analysis';
    validation?: (output: string) => boolean;
}

export interface MultiStepPrompt {
    steps: PromptStep[];
    dependencies: Map<number, number[]>;  // step -> depends on steps
    aggregationStrategy: 'sequential' | 'parallel' | 'conditional';
}

export type LLMExecutor = (prompt: string) => Promise<string>;

/**
 * Creates a structured multi-step prompt from a complex instruction.
 * Current implementation creates a sequential structure.
 */
export function createMultiStepPrompt(originalPrompt: string): MultiStepPrompt {
    // Placeholder logic: Splitting by sentences or specific markers
    // For now, we'll treat it as a single step if not clearly separable

    const steps: PromptStep[] = [
        {
            id: 1,
            prompt: originalPrompt,
            expectedOutputType: 'text'
        }
    ];

    return {
        steps,
        dependencies: new Map(),
        aggregationStrategy: 'sequential'
    };
}

/**
 * Executes a multi-step prompt using the provided executor.
 */
export async function executeMultiStep(
    multiStep: MultiStepPrompt,
    executor: LLMExecutor
): Promise<string> {

    const results = new Map<number, string>();

    // Simple sequential execution for MVP
    for (const step of multiStep.steps) {
        // Check dependencies
        const deps = multiStep.dependencies.get(step.id) || [];
        const context = deps.map(d => results.get(d)).join('\n\n');

        const effectivePrompt = context ? `${context}\n\nTask: ${step.prompt}` : step.prompt;

        const output = await executor(effectivePrompt);

        if (step.validation && !step.validation(output)) {
            throw new Error(`Validation failed for step ${step.id}`);
        }

        results.set(step.id, output);
    }

    return Array.from(results.values()).join('\n\n---\n\n');
}

/**
 * Validates the output of a specific step.
 */
export function validateStep(step: PromptStep, output: string): boolean {
    if (!step.validation) return true;
    return step.validation(output);
}
