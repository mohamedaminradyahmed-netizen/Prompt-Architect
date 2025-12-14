/**
 * Vector Store Configuration - DIRECTIVE-044
 *
 * Configuration for vector database providers.
 * Supports environment variables for secure API key management.
 *
 * Environment Variables:
 * - VECTOR_STORE_PROVIDER: 'memory' | 'pinecone' | 'weaviate'
 * - PINECONE_API_KEY: API key for Pinecone
 * - PINECONE_INDEX_NAME: Index name in Pinecone
 * - PINECONE_NAMESPACE: Optional namespace
 * - WEAVIATE_HOST: Host for Weaviate (e.g., 'localhost:8080')
 * - WEAVIATE_API_KEY: API key for Weaviate Cloud
 * - WEAVIATE_CLASS_NAME: Class name for collection
 */

import { VectorStoreClientConfig, VectorStoreProvider } from './client';

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Get vector store configuration from environment
 */
export function getVectorStoreConfig(): VectorStoreClientConfig {
  const provider = (process.env.VECTOR_STORE_PROVIDER || 'memory') as VectorStoreProvider;
  const dimension = parseInt(process.env.EMBEDDING_DIMENSION || '1536', 10);
  const metric = (process.env.SIMILARITY_METRIC || 'cosine') as 'cosine' | 'euclidean' | 'dot';

  switch (provider) {
    case 'pinecone':
      return {
        provider: 'pinecone',
        dimension,
        metric,
        apiKey: process.env.PINECONE_API_KEY || '',
        indexName: process.env.PINECONE_INDEX_NAME || 'prompts',
        namespace: process.env.PINECONE_NAMESPACE,
        environment: process.env.PINECONE_ENVIRONMENT
      };

    case 'weaviate':
      return {
        provider: 'weaviate',
        dimension,
        metric,
        host: process.env.WEAVIATE_HOST || 'localhost:8080',
        scheme: (process.env.WEAVIATE_SCHEME || 'http') as 'http' | 'https',
        apiKey: process.env.WEAVIATE_API_KEY,
        className: process.env.WEAVIATE_CLASS_NAME || 'Prompts'
      };

    case 'memory':
    default:
      return {
        provider: 'memory',
        dimension,
        metric,
        persistPath: process.env.MEMORY_PERSIST_PATH
      };
  }
}

/**
 * Preset configurations for common use cases
 */
export const VectorStorePresets = {
  /**
   * Development: In-memory store (no external dependencies)
   */
  development: (): VectorStoreClientConfig => ({
    provider: 'memory',
    dimension: 1536,
    metric: 'cosine'
  }),

  /**
   * Pinecone Free Tier: Good for small projects
   */
  pineconeFree: (apiKey: string, indexName: string): VectorStoreClientConfig => ({
    provider: 'pinecone',
    dimension: 1536,
    metric: 'cosine',
    apiKey,
    indexName
  }),

  /**
   * Pinecone with Namespace: For multi-tenant applications
   */
  pineconeMultiTenant: (
    apiKey: string,
    indexName: string,
    namespace: string
  ): VectorStoreClientConfig => ({
    provider: 'pinecone',
    dimension: 1536,
    metric: 'cosine',
    apiKey,
    indexName,
    namespace
  }),

  /**
   * Weaviate Local: For self-hosted deployments
   */
  weaviateLocal: (host: string = 'localhost:8080'): VectorStoreClientConfig => ({
    provider: 'weaviate',
    dimension: 1536,
    metric: 'cosine',
    host,
    scheme: 'http',
    className: 'Prompts'
  }),

  /**
   * Weaviate Cloud: For managed Weaviate instances
   */
  weaviateCloud: (
    host: string,
    apiKey: string,
    className: string = 'Prompts'
  ): VectorStoreClientConfig => ({
    provider: 'weaviate',
    dimension: 1536,
    metric: 'cosine',
    host,
    scheme: 'https',
    apiKey,
    className
  }),

  /**
   * High-dimensional embeddings (e.g., OpenAI ada-002)
   */
  openaiEmbeddings: (
    provider: VectorStoreProvider,
    config: Partial<VectorStoreClientConfig>
  ): VectorStoreClientConfig => ({
    provider,
    dimension: 1536, // text-embedding-ada-002
    metric: 'cosine',
    ...config
  } as VectorStoreClientConfig),

  /**
   * Small embeddings (e.g., sentence-transformers)
   */
  smallEmbeddings: (
    provider: VectorStoreProvider,
    config: Partial<VectorStoreClientConfig>
  ): VectorStoreClientConfig => ({
    provider,
    dimension: 384, // all-MiniLM-L6-v2
    metric: 'cosine',
    ...config
  } as VectorStoreClientConfig)
};

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * Validate vector store configuration
 */
export function validateConfig(config: VectorStoreClientConfig): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Common validations
  if (config.dimension < 1) {
    errors.push('Dimension must be positive');
  }

  if (!['cosine', 'euclidean', 'dot'].includes(config.metric)) {
    errors.push('Invalid metric. Use: cosine, euclidean, or dot');
  }

  // Provider-specific validations
  switch (config.provider) {
    case 'pinecone':
      if (!config.apiKey) {
        errors.push('Pinecone API key is required');
      }
      if (!config.indexName) {
        errors.push('Pinecone index name is required');
      }
      break;

    case 'weaviate':
      if (!config.host) {
        errors.push('Weaviate host is required');
      }
      if (!config.className) {
        errors.push('Weaviate class name is required');
      }
      break;
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

// ============================================================================
// COLLECTION SCHEMAS
// ============================================================================

/**
 * Predefined collection schemas for different use cases
 */
export const CollectionSchemas = {
  /**
   * Prompts collection: For storing and retrieving prompts
   */
  prompts: {
    name: 'Prompts',
    properties: [
      { name: 'content', type: 'text' },
      { name: 'category', type: 'text' },
      { name: 'userId', type: 'text' },
      { name: 'createdAt', type: 'date' },
      { name: 'score', type: 'number' }
    ]
  },

  /**
   * Test cases collection: For benchmark test cases
   */
  testCases: {
    name: 'TestCases',
    properties: [
      { name: 'input', type: 'text' },
      { name: 'expected', type: 'text' },
      { name: 'category', type: 'text' },
      { name: 'difficulty', type: 'text' }
    ]
  },

  /**
   * Knowledge base collection: For RAG factuality checking
   */
  knowledge: {
    name: 'Knowledge',
    properties: [
      { name: 'text', type: 'text' },
      { name: 'source', type: 'text' },
      { name: 'reliability', type: 'number' },
      { name: 'timestamp', type: 'date' }
    ]
  }
};

export default {
  getVectorStoreConfig,
  VectorStorePresets,
  validateConfig,
  CollectionSchemas
};
