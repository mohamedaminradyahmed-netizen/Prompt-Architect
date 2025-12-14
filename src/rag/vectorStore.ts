/**
 * Vector Store Module
 *
 * Provides vector database functionality for RAG-based factuality checking.
 * Supports multiple vector DB backends (Pinecone, Weaviate, in-memory).
 *
 * Note: This is a simplified implementation with in-memory storage.
 * For production, integrate with actual vector databases:
 * - Pinecone: npm install @pinecone-database/pinecone
 * - Weaviate: npm install weaviate-ts-client
 * - Chroma: npm install chromadb
 */

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Vector embedding (simplified as number array)
 */
export type Embedding = number[];

/**
 * Document stored in vector database
 */
export interface Document {
  id: string;
  content: string;
  metadata: DocumentMetadata;
  embedding?: Embedding;
}

/**
 * Document metadata
 */
export interface DocumentMetadata {
  source: string;          // Source of the document (URL, book, etc.)
  title?: string;          // Document title
  author?: string;         // Author/creator
  date?: string;           // Publication date
  reliability?: number;    // Reliability score (0-1)
  category?: string;       // Category/topic
  [key: string]: any;      // Additional metadata
}

/**
 * Search result from vector store
 */
export interface SearchResult {
  document: Document;
  score: number;           // Similarity score (0-1)
  distance?: number;       // Distance metric (optional)
}

/**
 * Vector store configuration
 */
export interface VectorStoreConfig {
  provider: 'memory' | 'pinecone' | 'weaviate' | 'chroma';
  dimension: number;       // Embedding dimension (e.g., 1536 for OpenAI)
  metric: 'cosine' | 'euclidean' | 'dot';
  apiKey?: string;         // API key for cloud providers
  environment?: string;    // Environment for Pinecone
  indexName?: string;      // Index/collection name
}

/**
 * Embedding provider for generating vectors
 */
export interface EmbeddingProvider {
  name: 'openai' | 'cohere' | 'huggingface' | 'custom';
  apiKey?: string;
  model?: string;
  dimension: number;
}

// ============================================================================
// IN-MEMORY VECTOR STORE
// ============================================================================

/**
 * Simple in-memory vector store implementation
 * For production, use a proper vector database
 */
class InMemoryVectorStore {
  private documents: Map<string, Document> = new Map();
  private config: VectorStoreConfig;

  constructor(config: VectorStoreConfig) {
    this.config = config;
  }

  /**
   * Add a document to the store
   */
  async addDocument(document: Document): Promise<void> {
    if (!document.embedding) {
      throw new Error('Document must have an embedding');
    }

    if (document.embedding.length !== this.config.dimension) {
      throw new Error(
        `Embedding dimension ${document.embedding.length} does not match config dimension ${this.config.dimension}`
      );
    }

    this.documents.set(document.id, document);
  }

  /**
   * Add multiple documents
   */
  async addDocuments(documents: Document[]): Promise<void> {
    for (const doc of documents) {
      await this.addDocument(doc);
    }
  }

  /**
   * Search for similar documents using vector similarity
   */
  async search(
    queryEmbedding: Embedding,
    topK: number = 5,
    filter?: (doc: Document) => boolean
  ): Promise<SearchResult[]> {
    if (queryEmbedding.length !== this.config.dimension) {
      throw new Error(
        `Query embedding dimension ${queryEmbedding.length} does not match config dimension ${this.config.dimension}`
      );
    }

    // Get all documents (apply filter if provided)
    let docs = Array.from(this.documents.values());
    if (filter) {
      docs = docs.filter(filter);
    }

    // Calculate similarity for each document
    const results: SearchResult[] = docs.map(doc => {
      if (!doc.embedding) {
        return { document: doc, score: 0 };
      }

      const similarity = this.calculateSimilarity(
        queryEmbedding,
        doc.embedding
      );

      return {
        document: doc,
        score: similarity,
      };
    });

    // Sort by similarity and return top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Get a document by ID
   */
  async getDocument(id: string): Promise<Document | null> {
    return this.documents.get(id) || null;
  }

  /**
   * Delete a document
   */
  async deleteDocument(id: string): Promise<boolean> {
    return this.documents.delete(id);
  }

  /**
   * Clear all documents
   */
  async clear(): Promise<void> {
    this.documents.clear();
  }

  /**
   * Get total document count
   */
  async count(): Promise<number> {
    return this.documents.size;
  }

  /**
   * Calculate similarity between two embeddings
   */
  private calculateSimilarity(embedding1: Embedding, embedding2: Embedding): number {
    switch (this.config.metric) {
      case 'cosine':
        return this.cosineSimilarity(embedding1, embedding2);
      case 'euclidean':
        return 1 / (1 + this.euclideanDistance(embedding1, embedding2));
      case 'dot':
        return this.dotProduct(embedding1, embedding2);
      default:
        return this.cosineSimilarity(embedding1, embedding2);
    }
  }

  /**
   * Cosine similarity
   */
  private cosineSimilarity(a: Embedding, b: Embedding): number {
    const dotProduct = this.dotProduct(a, b);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    if (magnitudeA === 0 || magnitudeB === 0) return 0;

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Dot product
   */
  private dotProduct(a: Embedding, b: Embedding): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  /**
   * Euclidean distance
   */
  private euclideanDistance(a: Embedding, b: Embedding): number {
    return Math.sqrt(
      a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0)
    );
  }
}

// ============================================================================
// VECTOR STORE FACTORY
// ============================================================================

/**
 * Create a vector store instance
 */
export function createVectorStore(
  config: VectorStoreConfig
): InMemoryVectorStore {
  switch (config.provider) {
    case 'memory':
      return new InMemoryVectorStore(config);

    case 'pinecone':
      // In production, initialize Pinecone client
      // const pinecone = new Pinecone({ apiKey: config.apiKey });
      // const index = pinecone.index(config.indexName);
      throw new Error('Pinecone integration not implemented. Use in-memory store.');

    case 'weaviate':
      // In production, initialize Weaviate client
      // const weaviate = weaviateClient({ ...config });
      throw new Error('Weaviate integration not implemented. Use in-memory store.');

    case 'chroma':
      // In production, initialize Chroma client
      // const chroma = new ChromaClient();
      throw new Error('Chroma integration not implemented. Use in-memory store.');

    default:
      return new InMemoryVectorStore(config);
  }
}

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

/**
 * Generate embedding for text
 * Mock implementation - in production, use actual embedding APIs
 */
export async function generateEmbedding(
  text: string,
  provider: EmbeddingProvider
): Promise<Embedding> {
  // MOCK: In production, call actual embedding APIs:
  // - OpenAI: await openai.embeddings.create({ model: "text-embedding-3-small", input: text })
  // - Cohere: await cohere.embed({ texts: [text], model: "embed-english-v3.0" })
  // - HuggingFace: Use transformers.js or API

  await new Promise(resolve => setTimeout(resolve, 50));

  // Generate mock embedding (random normalized vector)
  const embedding: Embedding = Array.from(
    { length: provider.dimension },
    () => Math.random() * 2 - 1
  );

  // Normalize to unit vector (for cosine similarity)
  const magnitude = Math.sqrt(
    embedding.reduce((sum, val) => sum + val * val, 0)
  );

  return embedding.map(val => val / magnitude);
}

/**
 * Generate embeddings for multiple texts
 */
export async function generateEmbeddings(
  texts: string[],
  provider: EmbeddingProvider,
  onProgress?: (completed: number, total: number) => void
): Promise<Embedding[]> {
  const embeddings: Embedding[] = [];

  for (let i = 0; i < texts.length; i++) {
    const embedding = await generateEmbedding(texts[i], provider);
    embeddings.push(embedding);

    if (onProgress) {
      onProgress(i + 1, texts.length);
    }
  }

  return embeddings;
}

// ============================================================================
// DOCUMENT PREPARATION
// ============================================================================

/**
 * Prepare a document for indexing (chunk and embed)
 */
export async function prepareDocument(
  content: string,
  metadata: DocumentMetadata,
  provider: EmbeddingProvider,
  chunkSize: number = 500
): Promise<Document[]> {
  // Split content into chunks
  const chunks = chunkText(content, chunkSize);

  // Generate embeddings for each chunk
  const documents: Document[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = await generateEmbedding(chunk, provider);

    documents.push({
      id: `${metadata.source}_chunk_${i}`,
      content: chunk,
      metadata: {
        ...metadata,
        chunkIndex: i,
        totalChunks: chunks.length,
      },
      embedding,
    });
  }

  return documents;
}

/**
 * Chunk text into smaller pieces
 */
function chunkText(text: string, maxChunkSize: number): string[] {
  // Simple chunking by sentences
  const sentences = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 0);
  const chunks: string[] = [];
  let currentChunk = '';

  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > maxChunkSize && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      currentChunk = sentence;
    } else {
      currentChunk += (currentChunk ? '. ' : '') + sentence;
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

// ============================================================================
// KNOWLEDGE BASE INITIALIZATION
// ============================================================================

/**
 * Initialize a knowledge base with sample trusted documents
 */
export async function initializeKnowledgeBase(
  vectorStore: InMemoryVectorStore,
  provider: EmbeddingProvider
): Promise<void> {
  // Sample trusted knowledge base
  const trustedDocs = [
    {
      content: 'The Earth is the third planet from the Sun and the only known planet to harbor life.',
      metadata: {
        source: 'NASA',
        title: 'Earth Facts',
        reliability: 0.95,
        category: 'astronomy',
      },
    },
    {
      content: 'Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.',
      metadata: {
        source: 'Scientific American',
        title: 'Water Properties',
        reliability: 0.9,
        category: 'physics',
      },
    },
    {
      content: 'The human body contains approximately 37.2 trillion cells.',
      metadata: {
        source: 'Nature Journal',
        title: 'Cell Count Study',
        reliability: 0.85,
        category: 'biology',
      },
    },
    // Add more trusted facts as needed
  ];

  // Prepare and add documents
  for (const doc of trustedDocs) {
    const preparedDocs = await prepareDocument(
      doc.content,
      doc.metadata,
      provider,
      500
    );

    await vectorStore.addDocuments(preparedDocs);
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { InMemoryVectorStore };

export default {
  createVectorStore,
  generateEmbedding,
  generateEmbeddings,
  prepareDocument,
  initializeKnowledgeBase,
};
