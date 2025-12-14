/**
 * Mutation operators for prompt refinement
 * Simple transformations to create prompt variations
 */

export interface PromptVariation {
  prompt: string;
  mutation: string;
}

/**
 * Paraphrase mutation: rephrase the prompt while preserving meaning
 * Uses simple rephrasing rules
 */
export function paraphraseMutation(originalPrompt: string): PromptVariation {
  // Simple paraphrase rules - in a real implementation, this would use NLP
  const paraphrases = [
    (text: string) => text.replace(/^Write/, 'Create').replace(/^Create/, 'Write'),
    (text: string) => text.replace(/^Explain/, 'Describe').replace(/^Describe/, 'Explain'),
    (text: string) => text.replace(/^Help me/, 'Assist me with'),
    (text: string) => text.replace(/I need/, 'I would like'),
    (text: string) => text.replace(/about/, 'regarding'),
  ];

  let paraphrased = originalPrompt;
  const appliedRules = [];

  for (const rule of paraphrases) {
    const newText = rule(paraphrased);
    if (newText !== paraphrased) {
      paraphrased = newText;
      appliedRules.push('paraphrase');
      break; // Apply only one rule for simplicity
    }
  }

  return {
    prompt: paraphrased,
    mutation: 'paraphrase'
  };
}

/**
 * Shorten mutation: reduce prompt length while keeping key elements
 * Removes redundant words and simplifies structure
 */
export function shortenMutation(originalPrompt: string): PromptVariation {
  let shortened = originalPrompt;

  // Remove common filler words
  shortened = shortened.replace(/\b(please|kindly|can you|could you|would you)\b/gi, '');
  // Remove extra spaces
  shortened = shortened.replace(/\s+/g, ' ').trim();
  // Remove trailing punctuation if followed by more text
  shortened = shortened.replace(/([.!?])\s*([A-Z])/g, '$2');

  return {
    prompt: shortened,
    mutation: 'shorten'
  };
}

/**
 * Add constraint mutation: add helpful constraints or guidelines
 * Enhances the prompt with structure or requirements
 */
export function addConstraintMutation(originalPrompt: string): PromptVariation {
  const constraints = [
    ' Be specific and provide examples where appropriate.',
    ' Structure your response clearly with headings.',
    ' Include step-by-step instructions.',
    ' Use bullet points for clarity.',
    ' Provide a concise summary at the end.'
  ];

  // Randomly select a constraint (in real implementation, this could be smarter)
  const randomConstraint = constraints[Math.floor(Math.random() * constraints.length)];

  return {
    prompt: originalPrompt + randomConstraint,
    mutation: 'add_constraint'
  };
}

/**
 * Apply a random mutation to create a variation
 */
export function applyRandomMutation(originalPrompt: string): PromptVariation {
  const mutations = [paraphraseMutation, shortenMutation, addConstraintMutation];
  const randomMutation = mutations[Math.floor(Math.random() * mutations.length)];
  return randomMutation(originalPrompt);
}

/**
 * Generate multiple prompt variations
 */
export function generateVariations(originalPrompt: string, count: number = 3): PromptVariation[] {
  const variations: PromptVariation[] = [];

  // Always include the original as a baseline
  variations.push({
    prompt: originalPrompt,
    mutation: 'original'
  });

  // Generate additional variations
  while (variations.length < count) {
    const variation = applyRandomMutation(originalPrompt);
    // Avoid duplicates
    if (!variations.some(v => v.prompt === variation.prompt)) {
      variations.push(variation);
    }
  }

  return variations.slice(0, count);
}
