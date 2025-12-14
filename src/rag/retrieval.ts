/**
 * Retrieval Module
 *
 * Handles retrieval of relevant documents from the vector store
 * for fact verification and factuality checking.
 */

import {
  InMemoryVectorStore,
  Document,
  SearchResult,
  Embedding,
  EmbeddingProvider,
  generateEmbedding,
} from './vectorStore';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Retrieval configuration
 */
export interface RetrievalConfig {
  topK: number;                // Number of documents to retrieve
  minScore: number;            // Minimum similarity score threshold (0-1)
  rerank: boolean;             // Whether to rerank results
  diversify: boolean;          // Whether to diversify results
  filterByReliability: boolean; // Filter by source reliability
  minReliability: number;      // Minimum reliability score (0-1)
}

/**
 * Retrieved context for a query
 */
export interface RetrievedContext {
  query: string;
  results: SearchResult[];
  combinedContext: string;     // All results combined into single text
  confidence: number;          // Confidence in retrieval (0-1)
  sources: string[];           // List of sources
}

/**
 * Claim to verify
 */
export interface Claim {
  text: string;
  category?: string;
  keywords?: string[];
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_CONFIG: RetrievalConfig = {
  topK: 5,
  minScore: 0.5,
  rerank: false,
  diversify: true,
  filterByReliability: true,
  minReliability: 0.6,
};

// ============================================================================
// RETRIEVAL FUNCTIONS
// ============================================================================

/**
 * Retrieve relevant documents for a query
 *
 * @param query - The query text (claim to verify)
 * @param vectorStore - Vector store to search
 * @param embeddingProvider - Provider for generating query embedding
 * @param config - Retrieval configuration
 * @returns Retrieved context with relevant documents
 */
export async function retrieveRelevantDocs(
  query: string,
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider,
  config: Partial<RetrievalConfig> = {}
): Promise<RetrievedContext> {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };

  // Generate embedding for query
  const queryEmbedding = await generateEmbedding(query, embeddingProvider);

  // Define filter function
  const filter = finalConfig.filterByReliability
    ? (doc: Document) => {
        const reliability = doc.metadata.reliability || 0;
        return reliability >= finalConfig.minReliability;
      }
    : undefined;

  // Search vector store
  let results = await vectorStore.search(
    queryEmbedding,
    finalConfig.topK * 2, // Retrieve more for filtering/reranking
    filter
  );

  // Filter by minimum score
  results = results.filter(result => result.score >= finalConfig.minScore);

  // Diversify results if enabled
  if (finalConfig.diversify) {
    results = diversifyResults(results, finalConfig.topK);
  } else {
    results = results.slice(0, finalConfig.topK);
  }

  // Rerank if enabled
  if (finalConfig.rerank) {
    results = rerankResults(query, results);
  }

  // Combine results into context
  const combinedContext = results
    .map(result => result.document.content)
    .join('\n\n');

  // Calculate retrieval confidence
  const confidence = results.length > 0
    ? results.reduce((sum, r) => sum + r.score, 0) / results.length
    : 0;

  // Extract unique sources
  const sources = [...new Set(results.map(r => r.document.metadata.source))];

  return {
    query,
    results,
    combinedContext,
    confidence: Math.round(confidence * 1000) / 1000,
    sources,
  };
}

/**
 * Retrieve documents for multiple claims
 */
export async function retrieveForClaims(
  claims: Claim[],
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider,
  config?: Partial<RetrievalConfig>,
  onProgress?: (completed: number, total: number) => void
): Promise<Map<string, RetrievedContext>> {
  const results = new Map<string, RetrievedContext>();

  for (let i = 0; i < claims.length; i++) {
    const claim = claims[i];
    const context = await retrieveRelevantDocs(
      claim.text,
      vectorStore,
      embeddingProvider,
      config
    );

    results.set(claim.text, context);

    if (onProgress) {
      onProgress(i + 1, claims.length);
    }
  }

  return results;
}

// ============================================================================
// INGESTION (TEST/BOOTSTRAP HELPERS)
// ============================================================================

/**
 * Add documents to a vector store, generating embeddings when missing.
 *
 * Why: اختبارات factuality تعتمد على helper موحد لتغذية الـ store دون تكرار المنطق في كل مكان.
 */
export async function addDocuments(
  documents: Document[],
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider
): Promise<void> {
  const withEmbeddings: Document[] = [];

  for (const doc of documents) {
    try {
      withEmbeddings.push({
        ...doc,
        embedding: doc.embedding ?? (await generateEmbedding(doc.content, embeddingProvider)),
      });
    } catch {
      // Fail-safe: تجاهل وثيقة واحدة لا يجب أن يوقف ingestion كله.
      continue;
    }
  }

  await vectorStore.addDocuments(withEmbeddings);
}

// ============================================================================
// DIVERSIFICATION
// ============================================================================

/**
 * Diversify results to avoid redundant information
 * Uses Maximal Marginal Relevance (MMR) approach
 */
function diversifyResults(
  results: SearchResult[],
  topK: number,
  lambda: number = 0.7
): SearchResult[] {
  if (results.length <= topK) {
    return results;
  }

  const selected: SearchResult[] = [];
  const remaining = [...results];

  // Select first result (highest score)
  selected.push(remaining.shift()!);

  // Iteratively select documents that balance relevance and diversity
  while (selected.length < topK && remaining.length > 0) {
    let bestIndex = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];

      // Calculate MMR score
      const relevance = candidate.score;

      // Calculate max similarity to already selected docs
      const maxSimilarity = Math.max(
        ...selected.map(selected => {
          if (!candidate.document.embedding || !selected.document.embedding) {
            return 0;
          }
          return cosineSimilarity(
            candidate.document.embedding,
            selected.document.embedding
          );
        })
      );

      // MMR score: balance between relevance and diversity
      const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIndex = i;
      }
    }

    selected.push(remaining.splice(bestIndex, 1)[0]);
  }

  return selected;
}

/**
 * Cosine similarity helper
 */
function cosineSimilarity(a: Embedding, b: Embedding): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  if (magnitudeA === 0 || magnitudeB === 0) return 0;

  return dotProduct / (magnitudeA * magnitudeB);
}

// ============================================================================
// RERANKING
// ============================================================================

/**
 * Rerank results based on additional criteria
 * In production, use a cross-encoder model for reranking
 */
function rerankResults(query: string, results: SearchResult[]): SearchResult[] {
  // Simple reranking based on:
  // 1. Similarity score (already computed)
  // 2. Source reliability
  // 3. Recency (if date available)

  const reranked = results.map(result => {
    const doc = result.document;

    // Reliability boost
    const reliabilityScore = doc.metadata.reliability || 0.5;

    // Recency boost (if date available)
    let recencyScore = 0.5;
    if (doc.metadata.date) {
      const docDate = new Date(doc.metadata.date);
      const now = new Date();
      const ageInYears = (now.getTime() - docDate.getTime()) / (1000 * 60 * 60 * 24 * 365);
      recencyScore = Math.max(0, 1 - ageInYears / 10); // Decay over 10 years
    }

    // Combined score (weighted)
    const combinedScore =
      result.score * 0.6 +
      reliabilityScore * 0.3 +
      recencyScore * 0.1;

    return {
      ...result,
      score: combinedScore,
    };
  });

  // Sort by new combined score
  reranked.sort((a, b) => b.score - a.score);

  return reranked;
}

// ============================================================================
// CLAIM EXTRACTION
// ============================================================================

/**
 * Extract verifiable claims from text
 * Returns a list of claims that can be fact-checked
 */
export function extractClaims(text: string): Claim[] {
  // Split into sentences
  const sentences = text
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);

  // Filter for sentences that look like factual claims
  const claims: Claim[] = [];

  const factualPatterns = [
    /\b(is|are|was|were|has|have|will|would)\b/i,  // Factual verbs
    /\b\d+/,                                         // Numbers
    /\b(the|a|an)\s+[A-Z][a-z]+/,                   // Named entities
    /\b(in|on|at|during)\s+\d{4}/,                  // Dates
  ];

  for (const sentence of sentences) {
    const matchesPatterns = factualPatterns.filter(pattern => pattern.test(sentence)).length;

    // Require at least 2 factual patterns
    if (matchesPatterns >= 2) {
      // Extract keywords (simple approach)
      const keywords = extractKeywords(sentence);

      claims.push({
        text: sentence,
        keywords,
      });
    }
  }

  return claims;
}

/**
 * Extract keywords from text (simple approach)
 */
function extractKeywords(text: string): string[] {
  // Remove common stop words
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'has', 'have',
  ]);

  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word));

  // Return unique keywords
  return [...new Set(words)];
}

// ============================================================================
// CONTEXT EXPANSION
// ============================================================================

/**
 * Expand retrieved context by fetching related documents
 */
export async function expandContext(
  initialResults: SearchResult[],
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider,
  maxExpansion: number = 3
): Promise<SearchResult[]> {
  const expanded = new Set<string>(initialResults.map(r => r.document.id));
  const allResults = [...initialResults];

  // For each initial result, find related documents
  for (const result of initialResults.slice(0, maxExpansion)) {
    if (!result.document.embedding) continue;

    const related = await vectorStore.search(
      result.document.embedding,
      3,
      (doc) => !expanded.has(doc.id) // Exclude already retrieved
    );

    for (const relatedDoc of related) {
      if (!expanded.has(relatedDoc.document.id)) {
        expanded.add(relatedDoc.document.id);
        allResults.push(relatedDoc);
      }
    }
  }

  // Sort by score
  allResults.sort((a, b) => b.score - a.score);

  return allResults;
}

// ============================================================================
// FORMATTING
// ============================================================================

/**
 * Format retrieved context for display
 */
export function formatRetrievedContext(context: RetrievedContext): string {
  let output = `
Retrieved Context for: "${context.query}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Confidence: ${(context.confidence * 100).toFixed(1)}%
Sources: ${context.sources.join(', ')}
Documents Found: ${context.results.length}

`;

  context.results.forEach((result, i) => {
    output += `\n[${i + 1}] Score: ${(result.score * 100).toFixed(1)}% | Source: ${result.document.metadata.source}\n`;
    output += `${result.document.content}\n`;
  });

  output += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

  return output.trim();
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  retrieveRelevantDocs,
  retrieveForClaims,
  addDocuments,
  extractClaims,
  expandContext,
  formatRetrievedContext,
};
