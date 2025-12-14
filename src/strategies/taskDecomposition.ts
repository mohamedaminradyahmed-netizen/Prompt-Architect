/**
 * Task Decomposition Strategy (DIRECTIVE-008)
 *
 * This module implements a strategy to decompose complex tasks into smaller, manageable sub-tasks.
 * It analyzes the prompt, splits it into logical components, and creates an orchestration plan.
 */

import { PromptVariation } from '../mutations';

/**
 * Decomposes a complex task into sub-tasks and an orchestration prompt
 *
 * Logic:
 * 1. Analyze prompt and determine if it contains multiple tasks
 * 2. Split into separate sub-prompts
 * 3. Add linking instructions between tasks
 * 4. Create an "orchestration prompt" that combines results
 *
 * @param prompt - The original complex prompt
 * @returns Array of PromptVariation representing the decomposition
 */
export function decomposeTaskMutation(prompt: string): PromptVariation[] {
  const variations: PromptVariation[] = [];

  // 1. Analyze prompt to identify sub-tasks
  const subTasks = identifySubTasks(prompt);

  if (subTasks.length <= 1) {
    return [{
      text: prompt,
      mutationType: 'task-decomposition',
      changeDescription: 'Task was simple enough to not require decomposition',
      expectedImpact: {
        quality: 'neutral',
        cost: 'neutral',
        latency: 'neutral',
        reliability: 'neutral'
      },
      metadata: {
        isDecomposed: false,
        subTaskCount: 1
      }
    }];
  }

  // 2. Create sub-prompts
  const subPrompts = subTasks.map((task, index) => {
    return {
      text: task,
      mutationType: 'task-decomposition' as const,
      changeDescription: `Sub-task ${index + 1} of ${subTasks.length}`,
      expectedImpact: {
        quality: 'increase',
        cost: 'neutral',
        latency: 'neutral',
        reliability: 'increase'
      },
      metadata: {
        isDecomposed: true,
        role: 'sub-task',
        index: index + 1,
        totalTasks: subTasks.length
      }
    };
  });

  // 3. Create Orchestrator Prompt
  const orchestratorText = createOrchestratorPrompt(subTasks);
  const orchestratorPrompt: PromptVariation = {
    text: orchestratorText,
    mutationType: 'task-decomposition',
    changeDescription: 'Orchestration prompt to integrate sub-tasks',
    expectedImpact: {
      quality: 'increase',
      cost: 'neutral',
      latency: 'neutral',
      reliability: 'increase'
    },
    metadata: {
      isDecomposed: true,
      role: 'orchestrator'
    }
  };

  const formattedSubPrompts: PromptVariation[] = subPrompts.map(sp => ({
    text: sp.text,
    mutationType: 'task-decomposition',
    changeDescription: sp.changeDescription,
    expectedImpact: {
      quality: 'increase',
      cost: 'neutral',
      latency: 'neutral',
      reliability: 'increase'
    },
    metadata: sp.metadata
  }));

  return [...formattedSubPrompts, orchestratorPrompt];
}

/**
 * Identifies sub-tasks from a complex prompt using heuristics
 */
function identifySubTasks(prompt: string): string[] {
  const lines = prompt.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  // Case 1: Already a list (bullet points or numbered)
  const listPattern = /^(\d+\.|-|\*)\s+/;
  const listItems = lines.filter(line => listPattern.test(line));

  if (listItems.length > 1) {
    return listItems.map(item => item.replace(/^(\d+\.|-|\*)\s+/, '').trim());
  }

  // Case 2: Split by sequential keywords or punctuation using a unified regex
  // Splits by ". Then ", ". Finally ", ". Next ", ". Also ", or "; "
  const separatorRegex = /(?:\.\s+(?:Then|Finally|Next|Also)\s+|;\s+)/i;

  const sequentialParts = prompt.split(separatorRegex)
    .map(p => p.trim())
    // Clean up trailing periods if they are not part of the split logic but exist
    .map(p => p.replace(/\.$/, ''))
    .filter(p => p.length > 5); // Filter out very short fragments

  if (sequentialParts.length > 1) {
    return sequentialParts;
  }

  // Case 3: "with" logic (fallback)
  if (prompt.includes(' with ')) {
    const parts = prompt.split(' with ');
    if (parts.length === 2 && parts[1].length > 15) {
      const baseTask = parts[0].trim();
      const subFeature = parts[1].trim();

      let secondTask = subFeature;
      if (!/^(create|build|implement|add|write)/i.test(subFeature)) {
        secondTask = `Implement ${subFeature}`;
      }

      return [baseTask, secondTask];
    }
  }

  // Case 4: "and" logic (fallback)
  if (prompt.includes(' and ')) {
    const parts = prompt.split(' and ');
    if (parts.every(p => p.trim().length > 15)) {
       return parts.map(p => p.trim());
    }
  }

  return [prompt];
}

/**
 * Creates an orchestration prompt that combines the results of sub-tasks
 */
function createOrchestratorPrompt(subTasks: string[]): string {
  return `Integrate the following components into a complete system:\n\n${subTasks.map((task, i) => `${i+1}. ${task}`).join('\n')}\n\nEnsure all components work together seamlessly and handle data flow between them correctly.`;
}
