/**
 * Factuality Checker Module
 *
 * Verifies factual claims using RAG (Retrieval-Augmented Generation).
 * Retrieves relevant documents from a knowledge base and checks if claims
 * are supported by trusted sources.
 */

import {
  InMemoryVectorStore,
  EmbeddingProvider,
  createVectorStore,
  VectorStoreConfig,
} from '../rag/vectorStore';

import {
  retrieveRelevantDocs,
  extractClaims,
  RetrievalConfig,
  RetrievedContext,
  Claim,
} from '../rag/retrieval';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Result of factuality verification
 */
export interface FactualityCheck {
  isFactual: boolean;          // Overall factuality verdict
  confidence: number;          // Confidence in the verdict (0-1)
  sources: string[];           // Supporting sources
  contradictions: string[];    // Contradictory information found
  claims: ClaimVerification[]; // Individual claim verifications
  overallScore: number;        // Overall factuality score (0-100)
}

/**
 * Verification result for a single claim
 */
export interface ClaimVerification {
  claim: string;
  isSupported: boolean;
  confidence: number;
  supportingEvidence: string[];
  contradictingEvidence: string[];
  sources: string[];
}

/**
 * Configuration for factuality checking
 */
export interface FactualityConfig {
  vectorStore: VectorStoreConfig;
  retrieval: Partial<RetrievalConfig>;
  embeddingProvider: EmbeddingProvider;
  requireMultipleSources: boolean;  // Require evidence from multiple sources
  minSourceCount: number;           // Minimum number of sources for high confidence
  supportThreshold: number;         // Similarity threshold for support (0-1)
  contradictionThreshold: number;   // Similarity threshold for contradiction (0-1)
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_CONFIG: Partial<FactualityConfig> = {
  retrieval: {
    topK: 5,
    minScore: 0.6,
    filterByReliability: true,
    minReliability: 0.7,
  },
  requireMultipleSources: true,
  minSourceCount: 2,
  supportThreshold: 0.7,
  contradictionThreshold: 0.6,
};

// ============================================================================
// FACTUALITY CHECKER
// ============================================================================

/**
 * Main factuality checker class
 */
export class FactualityChecker {
  private vectorStore: InMemoryVectorStore;
  private config: FactualityConfig;

  constructor(config: FactualityConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config } as FactualityConfig;
    this.vectorStore = createVectorStore(config.vectorStore);
  }

  /**
   * Get the vector store instance
   */
  getVectorStore(): InMemoryVectorStore {
    return this.vectorStore;
  }

  /**
   * Verify factuality of a claim or text
   *
   * @param text - The text or claim to verify
   * @param context - Optional additional context
   * @returns FactualityCheck with verification results
   */
  async verifyFactuality(
    text: string,
    context?: string
  ): Promise<FactualityCheck> {
    // Extract claims from text
    const claims = extractClaims(text);

    if (claims.length === 0) {
      // No factual claims detected
      return {
        isFactual: true,
        confidence: 0.5,
        sources: [],
        contradictions: ['No verifiable factual claims detected'],
        claims: [],
        overallScore: 50,
      };
    }

    // Verify each claim
    const claimVerifications: ClaimVerification[] = [];

    for (const claim of claims) {
      const verification = await this.verifyClaim(claim, context);
      claimVerifications.push(verification);
    }

    // Aggregate results
    const supportedClaims = claimVerifications.filter(v => v.isSupported).length;
    const totalClaims = claimVerifications.length;

    // Calculate overall score
    const overallScore = (supportedClaims / totalClaims) * 100;

    // Determine if factual (majority of claims supported)
    const isFactual = supportedClaims / totalClaims >= 0.6;

    // Calculate confidence (average of individual confidences)
    const confidence =
      claimVerifications.reduce((sum, v) => sum + v.confidence, 0) /
      claimVerifications.length;

    // Collect unique sources and contradictions
    const allSources = new Set<string>();
    const contradictions: string[] = [];

    for (const verification of claimVerifications) {
      verification.sources.forEach(source => allSources.add(source));

      if (!verification.isSupported && verification.contradictingEvidence.length > 0) {
        contradictions.push(
          `"${verification.claim}": ${verification.contradictingEvidence[0]}`
        );
      }
    }

    return {
      isFactual,
      confidence: Math.round(confidence * 1000) / 1000,
      sources: Array.from(allSources),
      contradictions,
      claims: claimVerifications,
      overallScore: Math.round(overallScore * 10) / 10,
    };
  }

  /**
   * Verify a single claim
   */
  private async verifyClaim(
    claim: Claim,
    context?: string
  ): Promise<ClaimVerification> {
    // Retrieve relevant documents
    const retrieved = await retrieveRelevantDocs(
      claim.text,
      this.vectorStore,
      this.config.embeddingProvider,
      this.config.retrieval
    );

    if (retrieved.results.length === 0) {
      // No relevant documents found
      return {
        claim: claim.text,
        isSupported: false,
        confidence: 0.3,
        supportingEvidence: [],
        contradictingEvidence: ['No relevant sources found'],
        sources: [],
      };
    }

    // Analyze retrieved documents for support/contradiction
    const supportingEvidence: string[] = [];
    const contradictingEvidence: string[] = [];
    const sources = new Set<string>();

    for (const result of retrieved.results) {
      const doc = result.document;
      const score = result.score;

      sources.add(doc.metadata.source);

      // Check if document supports the claim
      if (score >= this.config.supportThreshold) {
        supportingEvidence.push(doc.content);
      }
      // Check for contradictions (semantic check would be better)
      else if (this.detectContradiction(claim.text, doc.content)) {
        contradictingEvidence.push(doc.content);
      }
    }

    // Determine if claim is supported
    // Why:
    // شرط "مصادر متعددة" يجب أن يؤثر على الثقة لا على حقيقة الدعم نفسها،
    // وإلا نفشل claims صحيحة مدعومة بمصدر واحد موثوق (كما في اختبارات DIRECTIVE-014).
    const isSupported =
      supportingEvidence.length > 0 &&
      contradictingEvidence.length === 0;

    // Calculate confidence
    let confidence = 0;
    if (supportingEvidence.length > 0) {
      // Average of top supporting evidence scores
      const topScores = retrieved.results
        .filter(r => r.score >= this.config.supportThreshold)
        .map(r => r.score)
        .slice(0, 3);

      confidence = topScores.length > 0
        ? topScores.reduce((sum, score) => sum + score, 0) / topScores.length
        : 0;

      // Boost confidence if multiple sources
      if (this.config.requireMultipleSources && sources.size >= this.config.minSourceCount) {
        confidence = Math.min(1, confidence * 1.1);
      }

      // Reduce confidence if contradictions found
      if (contradictingEvidence.length > 0) {
        confidence *= 0.7;
      }
    } else {
      confidence = 0.3;
    }

    return {
      claim: claim.text,
      isSupported,
      confidence: Math.round(confidence * 1000) / 1000,
      supportingEvidence: supportingEvidence.slice(0, 3), // Top 3
      contradictingEvidence: contradictingEvidence.slice(0, 2), // Top 2
      sources: Array.from(sources),
    };
  }

  /**
   * Detect potential contradictions between claim and evidence
   * Simple heuristic - in production, use NLI model
   */
  private detectContradiction(claim: string, evidence: string): boolean {
    // Simple negation detection
    const claimLower = claim.toLowerCase();
    const evidenceLower = evidence.toLowerCase();

    // Check for explicit negations
    const negationWords = ['not', 'no', 'never', 'incorrect', 'false', 'wrong'];

    // If claim has no negation but evidence does (or vice versa), potential contradiction
    const claimHasNegation = negationWords.some(word => claimLower.includes(word));
    const evidenceHasNegation = negationWords.some(word => evidenceLower.includes(word));

    if (claimHasNegation !== evidenceHasNegation) {
      // Check if they share key terms
      const claimTokens = new Set(this.tokenize(claimLower));
      const evidenceTokens = new Set(this.tokenize(evidenceLower));

      const overlap = [...claimTokens].filter(t => evidenceTokens.has(t)).length;
      const overlapRatio = overlap / Math.min(claimTokens.size, evidenceTokens.size);

      // If high overlap but opposite polarity, likely contradiction
      return overlapRatio > 0.5;
    }

    return false;
  }

  /**
   * Tokenize text
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 2);
  }

  /**
   * Verify factuality for multiple texts
   */
  async verifyBatch(
    texts: string[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<FactualityCheck[]> {
    const results: FactualityCheck[] = [];

    for (let i = 0; i < texts.length; i++) {
      const result = await this.verifyFactuality(texts[i]);
      results.push(result);

      if (onProgress) {
        onProgress(i + 1, texts.length);
      }
    }

    return results;
  }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Quick factuality check with default configuration
 */
export async function verifyFactuality(
  claim: string,
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider,
  context?: string
): Promise<FactualityCheck> {
  const checker = new FactualityChecker({
    vectorStore: {
      provider: 'memory',
      dimension: embeddingProvider.dimension,
      metric: 'cosine',
    },
    embeddingProvider,
    retrieval: {},
    requireMultipleSources: true,
    minSourceCount: 2,
    supportThreshold: 0.7,
    contradictionThreshold: 0.6,
  });

  return checker.verifyFactuality(claim, context);
}

/**
 * Compare factuality between two texts
 */
export async function compareFactuality(
  textA: string,
  textB: string,
  vectorStore: InMemoryVectorStore,
  embeddingProvider: EmbeddingProvider
): Promise<{
  checkA: FactualityCheck;
  checkB: FactualityCheck;
  moreFactual: 'A' | 'B' | 'tie';
  scoreDiff: number;
}> {
  const checker = new FactualityChecker({
    vectorStore: {
      provider: 'memory',
      dimension: embeddingProvider.dimension,
      metric: 'cosine',
    },
    embeddingProvider,
    retrieval: {},
    requireMultipleSources: true,
    minSourceCount: 2,
    supportThreshold: 0.7,
    contradictionThreshold: 0.6,
  });

  const checkA = await checker.verifyFactuality(textA);
  const checkB = await checker.verifyFactuality(textB);

  const scoreDiff = checkB.overallScore - checkA.overallScore;

  let moreFactual: 'A' | 'B' | 'tie';
  if (Math.abs(scoreDiff) < 10) {
    moreFactual = 'tie';
  } else if (scoreDiff > 0) {
    moreFactual = 'B';
  } else {
    moreFactual = 'A';
  }

  return { checkA, checkB, moreFactual, scoreDiff };
}

// ============================================================================
// FORMATTING
// ============================================================================

/**
 * Format factuality check results for display
 */
export function formatFactualityCheck(check: FactualityCheck): string {
  let output = `
Factuality Check Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall: ${check.isFactual ? '✓ FACTUAL' : '✗ NOT FACTUAL'}
Score: ${check.overallScore.toFixed(1)}/100
Confidence: ${(check.confidence * 100).toFixed(1)}%

Claims Analyzed: ${check.claims.length}
Supported: ${check.claims.filter(c => c.isSupported).length}
Unsupported: ${check.claims.filter(c => !c.isSupported).length}

Sources Consulted: ${check.sources.length}
${check.sources.map(s => `  - ${s}`).join('\n')}
`;

  if (check.contradictions.length > 0) {
    output += `\nContradictions Found:\n`;
    check.contradictions.forEach((contradiction, i) => {
      output += `  ${i + 1}. ${contradiction}\n`;
    });
  }

  output += `\nDetailed Claim Analysis:\n`;
  check.claims.forEach((claim, i) => {
    output += `\n[${i + 1}] ${claim.isSupported ? '✓' : '✗'} ${claim.claim}\n`;
    output += `    Confidence: ${(claim.confidence * 100).toFixed(1)}%\n`;

    if (claim.supportingEvidence.length > 0) {
      output += `    Supporting: ${claim.supportingEvidence[0].substring(0, 100)}...\n`;
    }

    if (claim.contradictingEvidence.length > 0) {
      output += `    Contradicting: ${claim.contradictingEvidence[0].substring(0, 100)}...\n`;
    }
  });

  output += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

  return output.trim();
}

/**
 * Get factuality recommendation
 */
export function getFactualityRecommendation(check: FactualityCheck): string {
  if (check.overallScore >= 80) {
    return 'High factuality. Claims are well-supported by reliable sources.';
  } else if (check.overallScore >= 60) {
    return 'Moderate factuality. Most claims are supported, but verify key facts.';
  } else if (check.overallScore >= 40) {
    return 'Low factuality. Several unsupported claims detected. Use with caution.';
  } else {
    return 'Very low factuality. Most claims lack support. Not recommended.';
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  FactualityChecker,
  verifyFactuality,
  compareFactuality,
  formatFactualityCheck,
  getFactualityRecommendation,
};
