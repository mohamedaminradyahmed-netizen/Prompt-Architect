/**
 * Vector Store Module - DIRECTIVE-044
 *
 * Unified vector database interface supporting:
 * - In-Memory (development/testing)
 * - Pinecone (cloud-managed)
 * - Weaviate (self-hosted or cloud)
 *
 * Quick Start:
 * ```typescript
 * import { createVectorStoreClient, VectorStorePresets } from './vectorstore';
 *
 * // Development (in-memory)
 * const devStore = await createVectorStoreClient(VectorStorePresets.development());
 *
 * // Production (Pinecone)
 * const prodStore = await createVectorStoreClient(
 *   VectorStorePresets.pineconeFree(process.env.PINECONE_API_KEY!, 'my-index')
 * );
 *
 * // Self-hosted (Weaviate)
 * const selfHosted = await createVectorStoreClient(
 *   VectorStorePresets.weaviateLocal('localhost:8080')
 * );
 * ```
 */

// Client exports
export {
  // Interfaces
  IVectorStore,
  PineconeConfig,
  WeaviateConfig,
  MemoryConfig,
  VectorStoreClientConfig,
  VectorStoreProvider,

  // Implementations
  PineconeVectorStore,
  WeaviateVectorStore,
  MemoryVectorStoreAdapter,

  // Factory
  createVectorStoreClient,

  // Convenience functions
  indexPrompt,
  searchSimilar,
  retrieveContext
} from './client';

// Configuration exports
export {
  getVectorStoreConfig,
  VectorStorePresets,
  validateConfig,
  CollectionSchemas
} from './config';

// Re-export types from rag/vectorStore for convenience
export type {
  Document,
  DocumentMetadata,
  SearchResult,
  Embedding,
  VectorStoreConfig,
  EmbeddingProvider
} from '../rag/vectorStore';

// Re-export useful functions from rag/vectorStore
export {
  generateEmbedding,
  generateEmbeddings,
  prepareDocument,
  initializeKnowledgeBase,
  InMemoryVectorStore
} from '../rag/vectorStore';
