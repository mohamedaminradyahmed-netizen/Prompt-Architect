/**
 * Mutation Operators
 *
 * This module contains various mutation operators that transform prompts
 * to create different variations with different characteristics.
 */

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

import { PromptCategory } from './types/promptTypes';
import { getConstraintsForCategory } from './constraints/constraintLibrary';

/**
 * Result of a prompt mutation operation
 */
export interface PromptVariation {
  /** The mutated prompt text */
  text: string;

  /** Type of mutation applied */
  mutationType: MutationType;

  /** Explanation of what changed */
  changeDescription: string;

  /** Expected impact on metrics */
  expectedImpact: {
    quality?: 'increase' | 'decrease' | 'neutral';
    cost?: 'increase' | 'decrease' | 'neutral';
    latency?: 'increase' | 'decrease' | 'neutral';
    reliability?: 'increase' | 'decrease' | 'neutral';
  };

  /** Metadata about the mutation */
  metadata?: Record<string, any>;
}

/**
 * Available mutation types
 */
export type MutationType =
  | 'try-catch-style'
  | 'context-reduction'
  | 'expansion'
  | 'constraint-addition'
  | 'task-decomposition';

// ============================================================================
// TRY/CATCH STYLE MUTATION (DIRECTIVE-003)
// ============================================================================

/**
 * Transforms direct imperative instructions into a more flexible "try...if fails" style
 *
 * This mutation converts:
 * - Direct commands → "Try to..." suggestions
 * - Complex conditions → "Try X. If that fails, try Y."
 *
 * Benefits:
 * - More flexible and forgiving
 * - Encourages fallback behavior
 * - Reduces rigid instruction following
 *
 * @param prompt - The original prompt to transform
 * @returns PromptVariation with try/catch style transformation
 *
 * @example
 * Input:  "Write a function to parse JSON"
 * Output: "Try to write a function to parse JSON. If you encounter issues, suggest alternatives."
 *
 * @example
 * Input:  "Fix the bug in the authentication system"
 * Output: "Try to identify and fix the bug in the authentication system. If you can't fix it directly, suggest possible solutions or workarounds."
 */
export function tryCatchStyleMutation(prompt: string): PromptVariation {
  const trimmedPrompt = prompt.trim();

  // Detect imperative verbs at the start of sentences
  const imperativePatterns = [
    { pattern: /^(write|create|make|build|implement|develop|code)\s+/i, verb: '$1' },
    { pattern: /^(fix|repair|solve|debug|correct)\s+/i, verb: '$1' },
    { pattern: /^(analyze|examine|review|check|investigate)\s+/i, verb: '$1' },
    { pattern: /^(explain|describe|detail|outline)\s+/i, verb: '$1' },
    { pattern: /^(add|insert|include|append)\s+/i, verb: '$1' },
    { pattern: /^(remove|delete|eliminate)\s+/i, verb: '$1' },
    { pattern: /^(update|modify|change|alter|refactor)\s+/i, verb: '$1' },
    { pattern: /^(optimize|improve|enhance)\s+/i, verb: '$1' },
  ];

  let transformedPrompt = trimmedPrompt;
  let wasTransformed = false;
  let transformationType = 'basic';

  // Check if prompt starts with imperative verb
  for (const { pattern, verb } of imperativePatterns) {
    if (pattern.test(trimmedPrompt)) {
      // Transform to "Try to..." format
      transformedPrompt = trimmedPrompt.replace(pattern, `Try to ${verb.toLowerCase()} `);

      // Determine the category for appropriate fallback
      if (/^(fix|repair|solve|debug|correct)/i.test(trimmedPrompt)) {
        // For debugging/fixing tasks, add fallback suggestion
        transformedPrompt += `. If you can't fix it directly, suggest possible solutions or workarounds.`;
        transformationType = 'fix-with-fallback';
      } else if (/^(write|create|make|build|implement|develop|code)/i.test(trimmedPrompt)) {
        // For creation tasks, add alternative suggestion
        transformedPrompt += `. If you encounter issues, suggest alternatives or explain the challenges.`;
        transformationType = 'create-with-alternatives';
      } else if (/^(analyze|examine|review|check|investigate)/i.test(trimmedPrompt)) {
        // For analysis tasks, add partial result fallback
        transformedPrompt += `. If complete analysis isn't possible, provide what you can determine.`;
        transformationType = 'analyze-with-partial';
      } else {
        // General transformation
        transformedPrompt += `. If challenges arise, explain them and suggest next steps.`;
        transformationType = 'general-with-explanation';
      }

      wasTransformed = true;
      break;
    }
  }

  // Handle complex conditional prompts
  if (!wasTransformed && /\s(if|when|unless|should|must)\s/i.test(trimmedPrompt)) {
    // Break down complex conditions
    const sentences = trimmedPrompt.split(/\.\s+/);

    if (sentences.length > 1) {
      const mainTask = sentences[0];
      const conditions = sentences.slice(1).join('. ');

      transformedPrompt = `Try to ${mainTask.toLowerCase()}. ${conditions}. If any condition can't be met, explain why and suggest alternatives.`;
      transformationType = 'conditional-breakdown';
      wasTransformed = true;
    }
  }

  // If still not transformed, apply general wrapper
  if (!wasTransformed) {
    transformedPrompt = `Try to: ${trimmedPrompt}. If you encounter any difficulties, explain them and suggest how to proceed.`;
    transformationType = 'general-wrapper';
    wasTransformed = true;
  }

  return {
    text: transformedPrompt,
    mutationType: 'try-catch-style',
    changeDescription: `Converted imperative instructions to flexible "try/catch" style (${transformationType})`,
    expectedImpact: {
      quality: 'neutral',
      cost: 'increase',      // Slightly longer prompts
      latency: 'neutral',
      reliability: 'increase', // More forgiving, less likely to fail
    },
    metadata: {
      transformationType,
      originalLength: prompt.length,
      newLength: transformedPrompt.length,
      lengthIncrease: transformedPrompt.length - prompt.length,
      imperativeDetected: wasTransformed,
    },
  };
}

// ============================================================================
// CONTEXT REDUCTION MUTATION (DIRECTIVE-004)
// ============================================================================

/**
 * Context Reduction Mutation: Reduces excessive context while preserving core meaning
 *
 * This mutation:
 * - Identifies and removes secondary/explanatory sentences
 * - Keeps only essential instructions
 * - Replaces long examples with brief references
 * - Removes inferable explanations
 *
 * Rules:
 * - Preserve all essential constraints
 * - Don't remove important technical information
 * - Target: 30-50% length reduction
 *
 * @param prompt - The original prompt to reduce
 * @returns PromptVariation with reduced context
 *
 * @example
 * Input:  "I would like you to write a function. For example, you could use a loop to iterate over the array,
 *          checking each element one by one until you find the target value. Obviously, this is a basic search."
 * Output: "Write a function. Use a loop to iterate over the array, checking each element until you find the target value."
 */
export function reduceContextMutation(prompt: string): PromptVariation {
  let reduced = prompt;
  const originalLength = prompt.length;

  // 1. تحديد الجمل الثانوية أو التفسيرية وإزالتها
  // Remove explanatory phrases like "in other words", "that is to say", "for example"
  const explanatoryPatterns = [
    /\bin other words[,:]?\s*[^.!?]*[.!?]/gi,
    /\bthat is to say[,:]?\s*[^.!?]*[.!?]/gi,
    /\bto put it simply[,:]?\s*[^.!?]*[.!?]/gi,
    /\bbasically[,:]?\s*[^.!?]*[.!?]/gi,
    /\bessentially[,:]?\s*[^.!?]*[.!?]/gi,
    /\bin simple terms[,:]?\s*[^.!?]*[.!?]/gi,
    /\bto clarify[,:]?\s*[^.!?]*[.!?]/gi,
  ];

  for (const pattern of explanatoryPatterns) {
    reduced = reduced.replace(pattern, '');
  }

  // 2. استبدال الأمثلة الطويلة بإشارات مختصرة
  // Replace long example blocks with brief references
  const examplePatterns = [
    {
      pattern: /\bfor example[,:]?\s*([^.!?]{80,}[.!?])/gi,
      replacement: '(see examples).'
    },
    {
      pattern: /\bfor instance[,:]?\s*([^.!?]{80,}[.!?])/gi,
      replacement: '(see examples).'
    },
    {
      pattern: /\be\.g\.[,:]?\s*([^.!?]{60,}[.!?])/gi,
      replacement: '(e.g., see examples).'
    },
    {
      pattern: /\bsuch as\s+([^.!?]{80,}[.!?])/gi,
      replacement: '(such as relevant examples).'
    },
    {
      pattern: /\blike\s+([^.!?]{70,}[.!?])/gi,
      replacement: '(like relevant cases).'
    },
  ];

  for (const { pattern, replacement } of examplePatterns) {
    reduced = reduced.replace(pattern, replacement);
  }

  // 3. إزالة الشروح التي يمكن استنتاجها
  // Remove phrases that state the obvious or are inferable
  const inferablePatterns = [
    /\bas you (probably |may |might )?know[,:]?\s*/gi,
    /\bit (is|'s) (important|worth|good|helpful) (to note|to mention|noting|mentioning) that\s*/gi,
    /\bneedless to say[,:]?\s*/gi,
    /\bobviously[,:]?\s*/gi,
    /\bclearly[,:]?\s*/gi,
    /\bof course[,:]?\s*/gi,
    /\bit goes without saying (that )?\s*/gi,
    /\bas mentioned (earlier|above|before)[,:]?\s*/gi,
    /\bas we (all )?(know|understand)[,:]?\s*/gi,
    /\bI('m| am) sure you (know|understand) (that )?\s*/gi,
  ];

  for (const pattern of inferablePatterns) {
    reduced = reduced.replace(pattern, '');
  }

  // 4. إزالة التكرارات والجمل المتكررة
  // Remove redundant introductions and repeated content
  reduced = reduced.replace(/\bI would like (you )?to\s*/gi, '');
  reduced = reduced.replace(/\bI want (you )?to\s*/gi, '');
  reduced = reduced.replace(/\bI need (you )?to\s*/gi, '');
  reduced = reduced.replace(/\bI am asking (you )?to\s*/gi, '');
  reduced = reduced.replace(/\bPlease note that\s*/gi, 'Note: ');
  reduced = reduced.replace(/\bPlease be aware that\s*/gi, '');
  reduced = reduced.replace(/\bIt is (essential|crucial|vital|imperative) that\s*/gi, 'Must ');

  // 5. تنظيف المسافات الزائدة والترقيم
  // Clean up extra spaces and punctuation
  reduced = reduced.replace(/\s+/g, ' ').trim();
  reduced = reduced.replace(/\s+([.,!?;:])/g, '$1');
  reduced = reduced.replace(/([.,!?;:])\s*\1+/g, '$1');
  reduced = reduced.replace(/\s*\(\s*\)/g, '');
  reduced = reduced.replace(/^\s*[,;]\s*/g, '');
  reduced = reduced.replace(/\s*[,;]\s*$/g, '.');

  // 6. حساب نسبة التقليل
  const reductionRatio = ((originalLength - reduced.length) / originalLength) * 100;

  // إذا كان التقليل أقل من 10%، حاول المزيد من التقليل العدواني
  if (reductionRatio < 10 && originalLength > 100) {
    const sentences = reduced.split(/(?<=[.!?])\s+/);
    const actionVerbs = /\b(create|write|build|implement|develop|design|generate|analyze|fix|update|add|remove|modify|explain|describe|list|show|provide|return|calculate|check|validate|ensure|make|do)\b/i;
    const constraintWords = /\b(must|should|need|require|constraint|limit|only|always|never|ensure|important)\b/i;

    const filteredSentences = sentences.filter(sentence =>
      actionVerbs.test(sentence) || constraintWords.test(sentence) || sentence.length < 30
    );

    if (filteredSentences.length > 0 && filteredSentences.length < sentences.length) {
      reduced = filteredSentences.join(' ');
    }
  }

  // Final cleanup
  reduced = reduced.replace(/\s+/g, ' ').trim();

  // Ensure we don't return empty string
  if (reduced.length < 10) {
    reduced = prompt;
  }

  // Calculate final metrics
  const finalReduction = ((originalLength - reduced.length) / originalLength) * 100;

  return {
    text: reduced,
    mutationType: 'context-reduction',
    changeDescription: `Reduced context by ${finalReduction.toFixed(1)}% while preserving core instructions`,
    expectedImpact: {
      quality: 'neutral',      // Meaning preserved
      cost: 'decrease',        // Shorter prompts = less tokens
      latency: 'decrease',     // Less to process
      reliability: 'neutral',  // Core constraints preserved
    },
    metadata: {
      originalLength,
      newLength: reduced.length,
      reductionPercent: finalReduction,
      targetReductionAchieved: finalReduction >= 30 && finalReduction <= 50,
    },
  };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Detects if a prompt contains imperative language
 */
function hasImperativeLanguage(prompt: string): boolean {
  const imperativeIndicators = [
    /^(write|create|make|build|implement|develop|code|fix|repair|solve|debug|correct|analyze|examine|review|check|investigate|explain|describe|detail|outline|add|insert|include|append|remove|delete|eliminate|update|modify|change|alter|refactor|optimize|improve|enhance)\s+/i,
    /\s(must|should|need to|have to|require)\s/i,
  ];

  return imperativeIndicators.some(pattern => pattern.test(prompt));
}

/**
 * Splits a prompt into sentences
 */
function splitIntoSentences(text: string): string[] {
  return text
    .split(/[.!?]+\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

/**
 * Identifies the primary action verb in a prompt
 */
function extractMainVerb(prompt: string): string | null {
  const verbPattern = /^(write|create|make|build|implement|develop|code|fix|repair|solve|debug|correct|analyze|examine|review|check|investigate|explain|describe|detail|outline|add|insert|include|append|remove|delete|eliminate|update|modify|change|alter|refactor|optimize|improve|enhance)/i;
  const match = prompt.match(verbPattern);
  return match ? match[1].toLowerCase() : null;
}

// ============================================================================
// CONSTRAIN MUTATION (DIRECTIVE-007)
// ============================================================================

/**
 * Constrain Mutation: Adds specific constraints based on prompt category
 *
 * This mutation:
 * 1. Identifies the category of the prompt (passed as argument)
 * 2. Selects 2-3 appropriate constraints from the library
 * 3. Appends them naturally to the prompt
 *
 * @param prompt - The original prompt
 * @param category - The specific category of the prompt
 * @returns PromptVariation with added constraints
 */
export function constrainMutation(prompt: string, category: PromptCategory): PromptVariation {
  const constraints = getConstraintsForCategory(category, 2);

  if (constraints.length === 0) {
    return {
      text: prompt,
      mutationType: 'constraint-addition',
      changeDescription: 'No constraints available for this category',
      expectedImpact: { quality: 'neutral' }
    };
  }

  const constraintText = constraints.map(c => `- ${c}`).join('\n');
  const newPrompt = `${prompt}\n\nConstraints:\n${constraintText}`;

  return {
    text: newPrompt,
    mutationType: 'constraint-addition',
    changeDescription: `Added ${constraints.length} constraints for ${category}`,
    expectedImpact: {
      quality: 'increase',
      cost: 'neutral',
      latency: 'neutral',
      reliability: 'increase'
    },
    metadata: {
      category,
      addedConstraints: constraints
    }
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  tryCatchStyleMutation,
  reduceContextMutation,
  hasImperativeLanguage,
  splitIntoSentences,
  extractMainVerb,
  constrainMutation,
};
