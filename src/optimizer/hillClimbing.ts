import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  constrainMutation,
  PromptVariation
} from '../mutations';
import { classifyPrompt } from '../types/promptTypes';

import { ScoringFunction, OptimizationResult } from './types';

/**
 * Hill-Climbing Optimizer (DIRECTIVE-019)
 * 
 * Iteratively improves a prompt by applying random mutations and keeping
 * changes that increase the score.
 */
export async function hillClimbingOptimize(
  initialPrompt: string,
  maxIterations: number = 10,
  scoringFunction: ScoringFunction
): Promise<OptimizationResult> {
  let currentPrompt = initialPrompt;
  let currentScore = await scoringFunction(currentPrompt);

  // History tracks the successful steps (the "climb")
  const history: { prompt: string; score: number; mutation: string }[] = [];

  // Initial record
  history.push({
    prompt: currentPrompt,
    score: currentScore,
    mutation: 'initial'
  });

  for (let i = 0; i < maxIterations; i++) {
    // 1. Select a random mutation
    const mutationType = Math.floor(Math.random() * 4); // 0, 1, 2, 3
    let mutationResult: PromptVariation;

    try {
      switch (mutationType) {
        case 0: // tryCatchStyleMutation
          mutationResult = tryCatchStyleMutation(currentPrompt);
          break;
        case 1: // reduceContextMutation
          mutationResult = reduceContextMutation(currentPrompt);
          break;
        case 2: // expandMutation
          mutationResult = expandMutation(currentPrompt);
          break;
        case 3: // constrainMutation
        default:
          const classification = classifyPrompt(currentPrompt);
          mutationResult = constrainMutation(currentPrompt, classification.category);
          break;
      }

      // 2. Evaluate the new variation
      const newScore = await scoringFunction(mutationResult.text);

      // 3. Comparison (Hill Climbing Logic)
      if (newScore > currentScore) {
        // Improvement found: Accept change
        currentPrompt = mutationResult.text;
        currentScore = newScore;

        history.push({
          prompt: currentPrompt,
          score: currentScore,
          mutation: mutationResult.mutationType
        });
      } else {
        // No improvement: Discard change.
        // We do NOT record rejected attempts in the "history" of the prompt evolution,
        // as that would confuse the final output report which usually shows the progression.
      }

    } catch (error) {
      console.error(`Mutation failed at iteration ${i}:`, error);
      // Continue to next iteration
    }
  }

  return {
    bestPrompt: currentPrompt,
    bestScore: currentScore,
    iterations: maxIterations,
    history
  };
}
