/**
 * RAG Pipeline (DIRECTIVE-043)
 *
 * الهدف: Retrieval → Verification → Scoring عبر الموديولات الحالية:
 * - retrieval: retrieveRelevantDocs
 * - verification/scoring: verifyFactuality (factualityChecker)
 *
 * Why:
 * فصل الـ retrieval عن التحقق يسمح بقياس جودة الاسترجاع وتحسينه لاحقاً (rerank/diversify/thresholds).
 */

import { createVectorStore, type EmbeddingProvider, type InMemoryVectorStore } from '../rag/vectorStore';
import { retrieveRelevantDocs, type RetrievedContext } from '../rag/retrieval';
import { verifyFactuality, type FactualityCheck } from '../evaluator/factualityChecker';
import { createRunnableSequence, type RunnableLike } from './langchainCompat';

export interface RagPipelineInput {
  text: string;
  /**
   * Optional extra context (e.g., user-provided facts or system context).
   * This is appended after retrieved context.
   */
  context?: string;
  /**
   * Optional: provide a pre-populated vector store (useful in tests / offline KB).
   */
  vectorStore?: InMemoryVectorStore;
  /**
   * Optional: embedding provider configuration.
   * If omitted, we create a small mock config (dimension=384) to match project defaults.
   */
  embeddingProvider?: EmbeddingProvider;
}

export interface RagPipelineOutput {
  retrieved: RetrievedContext;
  factuality: FactualityCheck;
}

function defaultEmbeddingProvider(): EmbeddingProvider {
  return { name: 'custom', dimension: 384 };
}

function defaultVectorStore(dimension: number): InMemoryVectorStore {
  return createVectorStore({ provider: 'memory', dimension, metric: 'cosine' });
}

export async function createRagPipeline(): Promise<RunnableLike<RagPipelineInput, RagPipelineOutput>> {
  return createRunnableSequence<RagPipelineInput, RagPipelineOutput>([
    async ({ text, context, vectorStore, embeddingProvider }) => {
      const provider = embeddingProvider ?? defaultEmbeddingProvider();
      const store = vectorStore ?? defaultVectorStore(provider.dimension);
      return { text, context, store, provider };
    },
    async ({ text, context, store, provider }: any) => {
      const retrieved = await retrieveRelevantDocs(text, store, provider, {});
      const combined = [retrieved.combinedContext, context].filter(Boolean).join('\n\n');
      return { text, store, provider, retrieved, combined };
    },
    async ({ text, store, provider, retrieved, combined }: any) => {
      const factuality = await verifyFactuality(text, store, provider, combined || undefined);
      return { retrieved, factuality } satisfies RagPipelineOutput;
    },
  ]);
}

export async function runRagPipeline(input: RagPipelineInput): Promise<RagPipelineOutput> {
  const pipeline = await createRagPipeline();
  return pipeline.invoke(input);
}

