/**
 * Refinement Pipeline (DIRECTIVE-043)
 *
 * الهدف: Orchestration من "Prompt واحد" إلى "أفضل 3 اقتراحات" عبر:
 * - Classification
 * - Mutation generation
 * - Evaluation
 * - Ranking
 *
 * Why:
 * هذا يوفّر نقطة تكامل واحدة للـ UI/الخدمات بدلاً من ربط الدوال يدوياً في كل مكان.
 */

import { classifyPrompt, PromptCategory, type PromptClassification } from '../types/promptTypes';
import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  constrainMutation,
  type PromptVariation,
} from '../mutations';
import { evaluateSuggestions, type ScoredSuggestion } from '../evaluator';
import { createRunnableSequence, type RunnableLike } from './langchainCompat';

export interface RefinementPipelineInput {
  prompt: string;
  topK?: number;
}

export interface RefinementPipelineOutput {
  originalPrompt: string;
  classification: PromptClassification;
  variations: PromptVariation[];
  suggestions: ScoredSuggestion[];
}

function generateMutations(prompt: string, category: PromptCategory): PromptVariation[] {
  // Minimal, deterministic set that maps مباشرة لموديولات المشروع الحالية.
  return [
    tryCatchStyleMutation(prompt),
    reduceContextMutation(prompt),
    expandMutation(prompt),
    constrainMutation(prompt, category),
  ];
}

export async function createRefinementPipeline(): Promise<RunnableLike<RefinementPipelineInput, RefinementPipelineOutput>> {
  return createRunnableSequence<RefinementPipelineInput, RefinementPipelineOutput>([
    async ({ prompt, topK }) => {
      const classification = classifyPrompt(prompt);
      return { prompt, topK, classification };
    },
    async ({ prompt, topK, classification }: any) => {
      const variations = generateMutations(prompt, classification.category);
      return { prompt, topK, classification, variations };
    },
    async ({ prompt, topK, classification, variations }: any) => {
      const scored = await evaluateSuggestions(prompt, variations);
      const k = Math.max(1, Math.min(topK ?? 3, scored.length));
      return {
        originalPrompt: prompt,
        classification,
        variations,
        suggestions: scored.slice(0, k),
      } satisfies RefinementPipelineOutput;
    },
  ]);
}

export async function runRefinementPipeline(
  prompt: string,
  topK: number = 3
): Promise<RefinementPipelineOutput> {
  const pipeline = await createRefinementPipeline();
  return pipeline.invoke({ prompt, topK });
}

