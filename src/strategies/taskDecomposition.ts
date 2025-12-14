import { PromptVariation } from '../mutations';

/**
 * Task Decomposition Strategy
 *
 * Breaks down complex tasks into smaller, manageable sub-tasks.
 */

/**
 * Decomposes a complex prompt into multiple sub-prompts and an orchestrator.
 *
 * @param prompt - The complex prompt to decompose
 * @returns An array of PromptVariations covering sub-tasks and orchestration
 */
export function decomposeTaskMutation(prompt: string): PromptVariation[] {
    const variations: PromptVariation[] = [];

    // Heuristic: Split by "and", "then", or punctuation that suggests multiple steps
    // This is a basic rule-based implementation. In a real system, this might use an LLM.
    const separators = /\b(?:and|then|after that|first|secondly|finally)\b|[,;](\s+and)?/i;

    // Check if we can split specific keywords like "Build X with Y"
    const withPattern = /build|create|implement/i;
    const parts = prompt.split(separators).map(p => p ? p.trim() : '').filter(p => p.length > 10); // Filter out noise

    if (parts.length < 2) {
        // Attempt another split strategy for "Build X with Y" -> "Build X", "Implement Y"
        if (prompt.includes(' with ')) {
            const [main, secondary] = prompt.split(' with ');
            variations.push(createVariation(main, 'sub-task', 'Main component'));
            variations.push(createVariation(`Implement ${secondary} for ${main}`, 'sub-task', 'Secondary component'));
            variations.push(createVariation(`Integrate: ${main} with ${secondary}`, 'orchestration', 'Orchestration'));
            return variations;
        }

        // Return original if no obvious decomposition
        return [createVariation(prompt, 'decomposition-failed', 'Could not decompose task')];
    }

    // standard split
    parts.forEach((part, index) => {
        let taskPrompt = part;
        if (!/^(write|create|build)/.test(taskPrompt)) {
            taskPrompt = `Task ${index + 1}: ${taskPrompt}`;
        }
        variations.push(createVariation(taskPrompt, 'sub-task', `Step ${index + 1}`));
    });

    // Add orchestrator
    const orchestrationPrompt = `Combine the following results:\n${parts.map((p, i) => `${i + 1}. ${p}`).join('\n')}`;
    variations.push(createVariation(orchestrationPrompt, 'orchestration', 'Orchestrating all steps'));

    return variations;
}

function createVariation(text: string, type: string, description: string): PromptVariation {
    return {
        text,
        mutationType: 'task-decomposition',
        changeDescription: description,
        expectedImpact: {
            quality: 'increase',
            cost: 'increase', // More prompts
            latency: 'increase',
            reliability: 'increase'
        },
        metadata: {
            decompositionType: type
        }
    };
}
