/**
 * Multi-Step Pipeline (DIRECTIVE-043)
 *
 * الهدف: Orchestrate prompts متعددة الخطوات عبر:
 * - تقسيم المهمة (createMultiStepPrompt)
 * - تنفيذ الخطوات (executeMultiStep)
 * - تجميع النتائج (موجود ضمن executeMultiStep حالياً)
 *
 * Why:
 * توحيد مسار التنفيذ متعدد الخطوات في API واحد يسمح لاحقاً بدمج LLM providers أو LangChain Agents بسهولة.
 */

import {
  createMultiStepPrompt,
  executeMultiStep,
  type LLMExecutor,
  type MultiStepPrompt,
} from '../strategies/multiStep';
import { createRunnableSequence, type RunnableLike } from './langchainCompat';

export interface MultiStepPipelineInput {
  prompt: string;
  executor: LLMExecutor;
}

export interface MultiStepPipelineOutput {
  multiStep: MultiStepPrompt;
  result: string;
}

export async function createMultiStepPipeline(): Promise<RunnableLike<MultiStepPipelineInput, MultiStepPipelineOutput>> {
  return createRunnableSequence<MultiStepPipelineInput, MultiStepPipelineOutput>([
    async ({ prompt, executor }) => {
      const multiStep = createMultiStepPrompt(prompt);
      return { multiStep, executor };
    },
    async ({ multiStep, executor }: any) => {
      const result = await executeMultiStep(multiStep, executor);
      return { multiStep, result } satisfies MultiStepPipelineOutput;
    },
  ]);
}

export async function runMultiStepPipeline(
  prompt: string,
  executor: LLMExecutor
): Promise<MultiStepPipelineOutput> {
  const pipeline = await createMultiStepPipeline();
  return pipeline.invoke({ prompt, executor });
}

