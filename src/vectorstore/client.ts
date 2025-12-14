/**
 * Vector Store Client Module - DIRECTIVE-044
 *
 * Provides unified interface for vector database operations.
 * Supports: In-Memory, Pinecone, Weaviate
 *
 * Usage:
 * ```typescript
 * import { createVectorStoreClient, VectorStoreProvider } from './vectorstore/client';
 *
 * // Create Pinecone client
 * const store = await createVectorStoreClient({
 *   provider: 'pinecone',
 *   apiKey: process.env.PINECONE_API_KEY,
 *   indexName: 'prompts',
 *   dimension: 1536
 * });
 *
 * // Index a prompt
 * await store.indexPrompt('Write a function...', { category: 'code' });
 *
 * // Search similar
 * const results = await store.searchSimilar('Create a function...', 5);
 * ```
 */

import {
  Document,
  DocumentMetadata,
  SearchResult,
  Embedding,
  VectorStoreConfig,
  EmbeddingProvider,
  generateEmbedding,
  InMemoryVectorStore
} from '../rag/vectorStore';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Unified Vector Store Interface
 * All vector store implementations must conform to this interface
 */
export interface IVectorStore {
  // Document operations
  addDocument(document: Document): Promise<void>;
  addDocuments(documents: Document[]): Promise<void>;
  getDocument(id: string): Promise<Document | null>;
  deleteDocument(id: string): Promise<boolean>;

  // Search operations
  search(queryEmbedding: Embedding, topK?: number, filter?: (doc: Document) => boolean): Promise<SearchResult[]>;

  // Utility operations
  clear(): Promise<void>;
  count(): Promise<number>;
  getAllDocuments(): Promise<Document[]>;

  // Connection management
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;
}

/**
 * Pinecone-specific configuration
 */
export interface PineconeConfig extends VectorStoreConfig {
  provider: 'pinecone';
  apiKey: string;
  environment?: string;  // e.g., 'us-east-1-aws'
  indexName: string;
  namespace?: string;
}

/**
 * Weaviate-specific configuration
 */
export interface WeaviateConfig extends VectorStoreConfig {
  provider: 'weaviate';
  host: string;          // e.g., 'localhost:8080' or 'your-instance.weaviate.network'
  scheme?: 'http' | 'https';
  apiKey?: string;       // For Weaviate Cloud Services
  className: string;     // Class name for the collection
}

/**
 * In-memory configuration (extended)
 */
export interface MemoryConfig extends VectorStoreConfig {
  provider: 'memory';
  persistPath?: string;  // Optional path for JSON persistence
}

/**
 * Union type for all configurations
 */
export type VectorStoreClientConfig = PineconeConfig | WeaviateConfig | MemoryConfig;

// ============================================================================
// PINECONE VECTOR STORE
// ============================================================================

/**
 * Pinecone Vector Store Implementation
 *
 * Requires: npm install @pinecone-database/pinecone
 */
export class PineconeVectorStore implements IVectorStore {
  private config: PineconeConfig;
  private client: any = null;
  private index: any = null;
  private connected: boolean = false;
  private localCache: Map<string, Document> = new Map();

  constructor(config: PineconeConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    try {
      // Dynamic import to avoid requiring the package if not used
      const { Pinecone } = await import('@pinecone-database/pinecone');

      this.client = new Pinecone({
        apiKey: this.config.apiKey
      });

      // Get the index
      this.index = this.client.index(this.config.indexName);

      // If namespace is specified, use it
      if (this.config.namespace) {
        this.index = this.index.namespace(this.config.namespace);
      }

      this.connected = true;
      console.log(`[Pinecone] Connected to index: ${this.config.indexName}`);
    } catch (error: any) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          'Pinecone package not installed. Run: npm install @pinecone-database/pinecone'
        );
      }
      throw new Error(`Failed to connect to Pinecone: ${error.message}`);
    }
  }

  async disconnect(): Promise<void> {
    this.client = null;
    this.index = null;
    this.connected = false;
    console.log('[Pinecone] Disconnected');
  }

  isConnected(): boolean {
    return this.connected;
  }

  async addDocument(document: Document): Promise<void> {
    this.ensureConnected();

    if (!document.embedding) {
      throw new Error('Document must have an embedding');
    }

    // Upsert to Pinecone
    await this.index.upsert([{
      id: document.id,
      values: document.embedding,
      metadata: {
        content: document.content,
        ...document.metadata
      }
    }]);

    // Cache locally for retrieval
    this.localCache.set(document.id, document);
  }

  async addDocuments(documents: Document[]): Promise<void> {
    this.ensureConnected();

    // Batch upsert (Pinecone supports up to 100 vectors per batch)
    const batchSize = 100;
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);

      const vectors = batch.map(doc => {
        if (!doc.embedding) {
          throw new Error(`Document ${doc.id} must have an embedding`);
        }
        return {
          id: doc.id,
          values: doc.embedding,
          metadata: {
            content: doc.content,
            ...doc.metadata
          }
        };
      });

      await this.index.upsert(vectors);

      // Cache locally
      batch.forEach(doc => this.localCache.set(doc.id, doc));
    }
  }

  async search(
    queryEmbedding: Embedding,
    topK: number = 5,
    filter?: (doc: Document) => boolean
  ): Promise<SearchResult[]> {
    this.ensureConnected();

    // Query Pinecone
    const queryResponse = await this.index.query({
      vector: queryEmbedding,
      topK: topK * 2, // Get more results for filtering
      includeMetadata: true
    });

    // Convert to SearchResult format
    let results: SearchResult[] = queryResponse.matches.map((match: any) => {
      const doc: Document = {
        id: match.id,
        content: match.metadata?.content || '',
        metadata: { ...match.metadata } as DocumentMetadata,
        embedding: match.values
      };
      delete doc.metadata.content; // Remove content from metadata

      return {
        document: doc,
        score: match.score
      };
    });

    // Apply local filter if provided
    if (filter) {
      results = results.filter(r => filter(r.document));
    }

    return results.slice(0, topK);
  }

  async getDocument(id: string): Promise<Document | null> {
    this.ensureConnected();

    // Try local cache first
    if (this.localCache.has(id)) {
      return this.localCache.get(id)!;
    }

    // Fetch from Pinecone
    const fetchResponse = await this.index.fetch([id]);

    if (fetchResponse.vectors && fetchResponse.vectors[id]) {
      const vector = fetchResponse.vectors[id];
      const doc: Document = {
        id: vector.id,
        content: vector.metadata?.content || '',
        metadata: { ...vector.metadata } as DocumentMetadata,
        embedding: vector.values
      };
      delete doc.metadata.content;

      this.localCache.set(id, doc);
      return doc;
    }

    return null;
  }

  async deleteDocument(id: string): Promise<boolean> {
    this.ensureConnected();

    try {
      await this.index.deleteOne(id);
      this.localCache.delete(id);
      return true;
    } catch {
      return false;
    }
  }

  async clear(): Promise<void> {
    this.ensureConnected();

    // Delete all vectors in namespace (or entire index if no namespace)
    await this.index.deleteAll();
    this.localCache.clear();
  }

  async count(): Promise<number> {
    this.ensureConnected();

    const stats = await this.index.describeIndexStats();
    return stats.totalRecordCount || 0;
  }

  async getAllDocuments(): Promise<Document[]> {
    // Note: Pinecone doesn't support listing all vectors efficiently
    // Return cached documents
    console.warn('[Pinecone] getAllDocuments returns only cached documents');
    return Array.from(this.localCache.values());
  }

  private ensureConnected(): void {
    if (!this.connected) {
      throw new Error('Pinecone not connected. Call connect() first.');
    }
  }
}

// ============================================================================
// WEAVIATE VECTOR STORE
// ============================================================================

/**
 * Weaviate Vector Store Implementation
 *
 * Requires: npm install weaviate-client
 */
export class WeaviateVectorStore implements IVectorStore {
  private config: WeaviateConfig;
  private client: any = null;
  private connected: boolean = false;

  constructor(config: WeaviateConfig) {
    this.config = {
      scheme: 'http',
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      // Dynamic import
      const weaviate = await import('weaviate-client');

      // Build connection config
      const connectionConfig: any = {
        scheme: this.config.scheme,
        host: this.config.host
      };

      // Add API key if provided (for Weaviate Cloud)
      if (this.config.apiKey) {
        connectionConfig.apiKey = new weaviate.ApiKey(this.config.apiKey);
      }

      this.client = await weaviate.default.connectToLocal({
        host: this.config.host.split(':')[0],
        port: parseInt(this.config.host.split(':')[1]) || 8080,
        grpcPort: 50051
      });

      // Ensure collection exists
      await this.ensureCollectionExists();

      this.connected = true;
      console.log(`[Weaviate] Connected to ${this.config.host}, class: ${this.config.className}`);
    } catch (error: any) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          'Weaviate package not installed. Run: npm install weaviate-client'
        );
      }
      throw new Error(`Failed to connect to Weaviate: ${error.message}`);
    }
  }

  private async ensureCollectionExists(): Promise<void> {
    try {
      const collections = await this.client.collections.listAll();
      const exists = collections.some((c: any) => c.name === this.config.className);

      if (!exists) {
        // Create collection with vector config
        await this.client.collections.create({
          name: this.config.className,
          vectorizers: [
            {
              name: 'none', // We provide our own vectors
              properties: ['content']
            }
          ],
          properties: [
            { name: 'content', dataType: 'text' },
            { name: 'source', dataType: 'text' },
            { name: 'title', dataType: 'text' },
            { name: 'author', dataType: 'text' },
            { name: 'date', dataType: 'text' },
            { name: 'reliability', dataType: 'number' },
            { name: 'category', dataType: 'text' },
            { name: 'docId', dataType: 'text' }
          ]
        });
        console.log(`[Weaviate] Created collection: ${this.config.className}`);
      }
    } catch (error: any) {
      console.warn(`[Weaviate] Collection check/create warning: ${error.message}`);
    }
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
    }
    this.client = null;
    this.connected = false;
    console.log('[Weaviate] Disconnected');
  }

  isConnected(): boolean {
    return this.connected;
  }

  async addDocument(document: Document): Promise<void> {
    this.ensureConnected();

    if (!document.embedding) {
      throw new Error('Document must have an embedding');
    }

    const collection = this.client.collections.get(this.config.className);

    await collection.data.insert({
      properties: {
        docId: document.id,
        content: document.content,
        source: document.metadata.source || '',
        title: document.metadata.title || '',
        author: document.metadata.author || '',
        date: document.metadata.date || '',
        reliability: document.metadata.reliability || 0,
        category: document.metadata.category || ''
      },
      vector: document.embedding
    });
  }

  async addDocuments(documents: Document[]): Promise<void> {
    this.ensureConnected();

    const collection = this.client.collections.get(this.config.className);

    // Batch insert
    const objects = documents.map(doc => {
      if (!doc.embedding) {
        throw new Error(`Document ${doc.id} must have an embedding`);
      }
      return {
        properties: {
          docId: doc.id,
          content: doc.content,
          source: doc.metadata.source || '',
          title: doc.metadata.title || '',
          author: doc.metadata.author || '',
          date: doc.metadata.date || '',
          reliability: doc.metadata.reliability || 0,
          category: doc.metadata.category || ''
        },
        vector: doc.embedding
      };
    });

    await collection.data.insertMany(objects);
  }

  async search(
    queryEmbedding: Embedding,
    topK: number = 5,
    filter?: (doc: Document) => boolean
  ): Promise<SearchResult[]> {
    this.ensureConnected();

    const collection = this.client.collections.get(this.config.className);

    // Vector search
    const response = await collection.query.nearVector(queryEmbedding, {
      limit: topK * 2, // Get more for filtering
      returnMetadata: ['distance']
    });

    // Convert to SearchResult format
    let results: SearchResult[] = response.objects.map((obj: any) => {
      const doc: Document = {
        id: obj.properties.docId || obj.uuid,
        content: obj.properties.content || '',
        metadata: {
          source: obj.properties.source,
          title: obj.properties.title,
          author: obj.properties.author,
          date: obj.properties.date,
          reliability: obj.properties.reliability,
          category: obj.properties.category
        },
        embedding: obj.vector
      };

      // Convert distance to similarity score (assuming cosine distance)
      const score = 1 - (obj.metadata?.distance || 0);

      return { document: doc, score };
    });

    // Apply local filter if provided
    if (filter) {
      results = results.filter(r => filter(r.document));
    }

    return results.slice(0, topK);
  }

  async getDocument(id: string): Promise<Document | null> {
    this.ensureConnected();

    const collection = this.client.collections.get(this.config.className);

    // Query by docId
    const response = await collection.query.fetchObjects({
      filters: {
        path: ['docId'],
        operator: 'Equal',
        valueText: id
      },
      limit: 1
    });

    if (response.objects.length > 0) {
      const obj = response.objects[0];
      return {
        id: obj.properties.docId || obj.uuid,
        content: obj.properties.content || '',
        metadata: {
          source: obj.properties.source,
          title: obj.properties.title,
          author: obj.properties.author,
          date: obj.properties.date,
          reliability: obj.properties.reliability,
          category: obj.properties.category
        },
        embedding: obj.vector
      };
    }

    return null;
  }

  async deleteDocument(id: string): Promise<boolean> {
    this.ensureConnected();

    try {
      const collection = this.client.collections.get(this.config.className);

      await collection.data.deleteMany({
        where: {
          path: ['docId'],
          operator: 'Equal',
          valueText: id
        }
      });

      return true;
    } catch {
      return false;
    }
  }

  async clear(): Promise<void> {
    this.ensureConnected();

    // Delete and recreate the collection
    await this.client.collections.delete(this.config.className);
    await this.ensureCollectionExists();
  }

  async count(): Promise<number> {
    this.ensureConnected();

    const collection = this.client.collections.get(this.config.className);
    const aggregate = await collection.aggregate.overAll();
    return aggregate.totalCount || 0;
  }

  async getAllDocuments(): Promise<Document[]> {
    this.ensureConnected();

    const collection = this.client.collections.get(this.config.className);
    const response = await collection.query.fetchObjects({
      limit: 10000 // Reasonable limit
    });

    return response.objects.map((obj: any) => ({
      id: obj.properties.docId || obj.uuid,
      content: obj.properties.content || '',
      metadata: {
        source: obj.properties.source,
        title: obj.properties.title,
        author: obj.properties.author,
        date: obj.properties.date,
        reliability: obj.properties.reliability,
        category: obj.properties.category
      },
      embedding: obj.vector
    }));
  }

  private ensureConnected(): void {
    if (!this.connected) {
      throw new Error('Weaviate not connected. Call connect() first.');
    }
  }
}

// ============================================================================
// MEMORY VECTOR STORE ADAPTER
// ============================================================================

/**
 * Adapter to make InMemoryVectorStore conform to IVectorStore interface
 */
export class MemoryVectorStoreAdapter implements IVectorStore {
  private store: InMemoryVectorStore;
  private connected: boolean = false;

  constructor(config: MemoryConfig) {
    this.store = new InMemoryVectorStore(config);
  }

  async connect(): Promise<void> {
    this.connected = true;
    console.log('[Memory] Vector store ready');
  }

  async disconnect(): Promise<void> {
    await this.store.clear();
    this.connected = false;
    console.log('[Memory] Vector store cleared');
  }

  isConnected(): boolean {
    return this.connected;
  }

  async addDocument(document: Document): Promise<void> {
    return this.store.addDocument(document);
  }

  async addDocuments(documents: Document[]): Promise<void> {
    return this.store.addDocuments(documents);
  }

  async search(
    queryEmbedding: Embedding,
    topK?: number,
    filter?: (doc: Document) => boolean
  ): Promise<SearchResult[]> {
    return this.store.search(queryEmbedding, topK, filter);
  }

  async getDocument(id: string): Promise<Document | null> {
    return this.store.getDocument(id);
  }

  async deleteDocument(id: string): Promise<boolean> {
    return this.store.deleteDocument(id);
  }

  async clear(): Promise<void> {
    return this.store.clear();
  }

  async count(): Promise<number> {
    return this.store.count();
  }

  async getAllDocuments(): Promise<Document[]> {
    return this.store.getAllDocuments();
  }

  // Expose underlying store for compatibility
  getUnderlyingStore(): InMemoryVectorStore {
    return this.store;
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

/**
 * Create a vector store client based on configuration
 */
export async function createVectorStoreClient(
  config: VectorStoreClientConfig
): Promise<IVectorStore> {
  let store: IVectorStore;

  switch (config.provider) {
    case 'pinecone':
      store = new PineconeVectorStore(config as PineconeConfig);
      break;

    case 'weaviate':
      store = new WeaviateVectorStore(config as WeaviateConfig);
      break;

    case 'memory':
    default:
      store = new MemoryVectorStoreAdapter(config as MemoryConfig);
      break;
  }

  // Auto-connect
  await store.connect();

  return store;
}

// ============================================================================
// CONVENIENCE FUNCTIONS (DIRECTIVE-044 API)
// ============================================================================

/**
 * Index a prompt with metadata
 */
export async function indexPrompt(
  store: IVectorStore,
  prompt: string,
  metadata: Record<string, any>,
  embeddingProvider: EmbeddingProvider
): Promise<string> {
  const embedding = await generateEmbedding(prompt, embeddingProvider);
  const id = `prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  await store.addDocument({
    id,
    content: prompt,
    metadata: {
      source: 'user',
      ...metadata
    } as DocumentMetadata,
    embedding
  });

  return id;
}

/**
 * Search for similar prompts
 */
export async function searchSimilar(
  store: IVectorStore,
  query: string,
  k: number,
  embeddingProvider: EmbeddingProvider
): Promise<SearchResult[]> {
  const queryEmbedding = await generateEmbedding(query, embeddingProvider);
  return store.search(queryEmbedding, k);
}

/**
 * Retrieve context for RAG
 */
export async function retrieveContext(
  store: IVectorStore,
  query: string,
  embeddingProvider: EmbeddingProvider,
  topK: number = 5
): Promise<string[]> {
  const results = await searchSimilar(store, query, topK, embeddingProvider);
  return results.map(r => r.document.content);
}

// ============================================================================
// EXPORTS
// ============================================================================

export type VectorStoreProvider = 'memory' | 'pinecone' | 'weaviate';

export default {
  createVectorStoreClient,
  indexPrompt,
  searchSimilar,
  retrieveContext,
  PineconeVectorStore,
  WeaviateVectorStore,
  MemoryVectorStoreAdapter
};
