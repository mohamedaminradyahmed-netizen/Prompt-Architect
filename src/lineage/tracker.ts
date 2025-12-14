/**
 * Lineage Tracking System (DIRECTIVE-028)
 *
 * Tracks the complete evolution history of prompt variations:
 * - Parent-child relationships
 * - Mutation chains applied
 * - Performance metrics at each step
 * - Full genealogy tree
 */

import { MutationType } from '../mutations';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

/**
 * Complete lineage record for a variation
 */
export interface VariationLineage {
  /** Unique identifier for this variation */
  id: string;

  /** Parent variation ID (null for original) */
  parentId: string | null;

  /** Original prompt this variation descended from */
  originalPrompt: string;

  /** Current prompt text */
  currentPrompt: string;

  /** Mutation type that created this variation */
  mutation: MutationType | 'original';

  /** Parameters used for the mutation */
  mutationParams: Record<string, any>;

  /** Timestamp when this variation was created */
  timestamp: Date;

  /** Performance metrics */
  metrics: VariationMetrics;

  /** Human feedback (if any) */
  feedback?: HumanFeedback;

  /** IDs of variations derived from this one */
  children: string[];

  /** Generation number (0 = original) */
  generation: number;

  /** Full path from original to this variation */
  path: MutationStep[];
}

/**
 * Performance metrics for a variation
 */
export interface VariationMetrics {
  /** Quality score (0-1) */
  score: number;

  /** Cost in USD */
  cost: number;

  /** Latency in milliseconds */
  latency: number;

  /** Token count */
  tokens?: number;

  /** Any additional metrics */
  custom?: Record<string, number>;
}

/**
 * Human feedback for a variation
 */
export interface HumanFeedback {
  /** User ID who provided feedback */
  userId: string;

  /** Rating (1-5) */
  rating: number;

  /** Textual feedback */
  comment?: string;

  /** Timestamp */
  timestamp: Date;

  /** Tags */
  tags?: string[];
}

/**
 * A single step in the mutation path
 */
export interface MutationStep {
  /** Mutation type applied */
  mutation: MutationType | 'original';

  /** Variation ID after this mutation */
  variationId: string;

  /** Score after this mutation */
  score: number;

  /** Improvement from parent */
  improvement: number;
}

/**
 * Lineage graph structure
 */
export interface LineageGraph {
  /** Root node (original prompt) */
  root: LineageNode;

  /** All nodes by ID */
  nodes: Map<string, LineageNode>;

  /** Statistics */
  stats: LineageStats;
}

/**
 * Node in the lineage graph
 */
export interface LineageNode {
  /** Variation data */
  variation: VariationLineage;

  /** Parent node */
  parent: LineageNode | null;

  /** Child nodes */
  children: LineageNode[];

  /** Depth in tree */
  depth: number;
}

/**
 * Statistics about the lineage
 */
export interface LineageStats {
  /** Total variations tracked */
  totalVariations: number;

  /** Maximum generation reached */
  maxGeneration: number;

  /** Best variation overall */
  bestVariation: VariationLineage;

  /** Average score by generation */
  avgScoreByGeneration: Map<number, number>;

  /** Most used mutation type */
  mostUsedMutation: MutationType;

  /** Mutation success rates */
  mutationSuccessRates: Map<MutationType, number>;
}

// ============================================================================
// LINEAGE TRACKER CLASS
// ============================================================================

/**
 * Main lineage tracking system
 */
export class LineageTracker {
  private lineages: Map<string, VariationLineage> = new Map();
  private byOriginalPrompt: Map<string, Set<string>> = new Map();
  private childIndex: Map<string, string[]> = new Map();

  /**
   * Track a new variation
   */
  trackVariation(variation: VariationLineage): void {
    // Store the variation
    this.lineages.set(variation.id, variation);

    // Index by original prompt
    if (!this.byOriginalPrompt.has(variation.originalPrompt)) {
      this.byOriginalPrompt.set(variation.originalPrompt, new Set());
    }
    this.byOriginalPrompt.get(variation.originalPrompt)!.add(variation.id);

    // Update parent's children
    if (variation.parentId) {
      const parent = this.lineages.get(variation.parentId);
      if (parent && !parent.children.includes(variation.id)) {
        parent.children.push(variation.id);
      }

      // Update child index
      if (!this.childIndex.has(variation.parentId)) {
        this.childIndex.set(variation.parentId, []);
      }
      this.childIndex.get(variation.parentId)!.push(variation.id);
    }
  }

  /**
   * Get lineage for a specific variation
   */
  getLineage(variationId: string): VariationLineage[] {
    const variation = this.lineages.get(variationId);
    if (!variation) {
      throw new Error(`Variation ${variationId} not found`);
    }

    const lineage: VariationLineage[] = [];
    let current: VariationLineage | undefined = variation;

    // Trace back to root
    while (current) {
      lineage.unshift(current);
      current = current.parentId ? this.lineages.get(current.parentId) : undefined;
    }

    return lineage;
  }

  /**
   * Get all descendants of a variation
   */
  getDescendants(variationId: string): VariationLineage[] {
    const descendants: VariationLineage[] = [];
    const queue = [variationId];

    while (queue.length > 0) {
      const currentId = queue.shift()!;
      const children = this.childIndex.get(currentId) || [];

      for (const childId of children) {
        const child = this.lineages.get(childId);
        if (child) {
          descendants.push(child);
          queue.push(childId);
        }
      }
    }

    return descendants;
  }

  /**
   * Visualize lineage as a graph
   */
  visualizeLineage(variationId: string): LineageGraph {
    const lineage = this.getLineage(variationId);
    const root = lineage[0];

    // Build tree structure
    const nodes = new Map<string, LineageNode>();
    const rootNode: LineageNode = {
      variation: root,
      parent: null,
      children: [],
      depth: 0,
    };
    nodes.set(root.id, rootNode);

    // Build all nodes
    this.buildLineageTree(rootNode, nodes);

    // Calculate statistics
    const stats = this.calculateLineageStats(nodes);

    return {
      root: rootNode,
      nodes,
      stats,
    };
  }

  /**
   * Find best path from original to target score
   */
  findBestPath(
    originalPrompt: string,
    targetScore: number
  ): VariationLineage[] | null {
    const variationIds = this.byOriginalPrompt.get(originalPrompt);
    if (!variationIds) {
      return null;
    }

    let bestPath: VariationLineage[] | null = null;
    let shortestLength = Infinity;

    for (const id of variationIds) {
      const variation = this.lineages.get(id)!;
      if (variation.metrics.score >= targetScore) {
        const path = this.getLineage(id);
        if (path.length < shortestLength) {
          shortestLength = path.length;
          bestPath = path;
        }
      }
    }

    return bestPath;
  }

  /**
   * Get variations by generation
   */
  getByGeneration(generation: number): VariationLineage[] {
    return Array.from(this.lineages.values()).filter(
      (v) => v.generation === generation
    );
  }

  /**
   * Get all variations for an original prompt
   */
  getAllVariations(originalPrompt: string): VariationLineage[] {
    const ids = this.byOriginalPrompt.get(originalPrompt);
    if (!ids) {
      return [];
    }

    return Array.from(ids).map((id) => this.lineages.get(id)!);
  }

  /**
   * Add human feedback to a variation
   */
  addFeedback(variationId: string, feedback: HumanFeedback): void {
    const variation = this.lineages.get(variationId);
    if (!variation) {
      throw new Error(`Variation ${variationId} not found`);
    }

    variation.feedback = feedback;
  }

  /**
   * Get statistics for all tracked lineages
   */
  getGlobalStats(): LineageStats {
    const allNodes = new Map<string, LineageNode>();

    for (const variation of this.lineages.values()) {
      if (!variation.parentId) {
        // This is a root
        const node: LineageNode = {
          variation,
          parent: null,
          children: [],
          depth: 0,
        };
        allNodes.set(variation.id, node);
        this.buildLineageTree(node, allNodes);
      }
    }

    return this.calculateLineageStats(allNodes);
  }

  /**
   * Export lineage data for analysis
   */
  exportLineage(variationId: string): string {
    const lineage = this.getLineage(variationId);
    return JSON.stringify(lineage, null, 2);
  }

  /**
   * Clear all tracked data
   */
  clear(): void {
    this.lineages.clear();
    this.byOriginalPrompt.clear();
    this.childIndex.clear();
  }

  // ============================================================================
  // PRIVATE HELPERS
  // ============================================================================

  private buildLineageTree(
    node: LineageNode,
    nodes: Map<string, LineageNode>
  ): void {
    const childIds = this.childIndex.get(node.variation.id) || [];

    for (const childId of childIds) {
      const childVariation = this.lineages.get(childId);
      if (childVariation) {
        const childNode: LineageNode = {
          variation: childVariation,
          parent: node,
          children: [],
          depth: node.depth + 1,
        };

        nodes.set(childId, childNode);
        node.children.push(childNode);

        // Recursively build children
        this.buildLineageTree(childNode, nodes);
      }
    }
  }

  private calculateLineageStats(
    nodes: Map<string, LineageNode>
  ): LineageStats {
    const variations = Array.from(nodes.values()).map((n) => n.variation);

    // Best variation
    const bestVariation = variations.reduce((best, current) =>
      current.metrics.score > best.metrics.score ? current : best
    );

    // Max generation
    const maxGeneration = Math.max(...variations.map((v) => v.generation));

    // Average score by generation
    const avgScoreByGeneration = new Map<number, number>();
    for (let gen = 0; gen <= maxGeneration; gen++) {
      const genVariations = variations.filter((v) => v.generation === gen);
      if (genVariations.length > 0) {
        const avgScore =
          genVariations.reduce((sum, v) => sum + v.metrics.score, 0) /
          genVariations.length;
        avgScoreByGeneration.set(gen, avgScore);
      }
    }

    // Mutation usage
    const mutationCounts = new Map<MutationType, number>();
    const mutationSuccesses = new Map<MutationType, number>();

    for (const variation of variations) {
      if (variation.mutation !== 'original') {
        const mutation = variation.mutation as MutationType;
        mutationCounts.set(mutation, (mutationCounts.get(mutation) || 0) + 1);

        // Count as success if score improved
        if (variation.parentId) {
          const parent = this.lineages.get(variation.parentId);
          if (parent && variation.metrics.score > parent.metrics.score) {
            mutationSuccesses.set(
              mutation,
              (mutationSuccesses.get(mutation) || 0) + 1
            );
          }
        }
      }
    }

    // Most used mutation
    let mostUsedMutation: MutationType = 'try-catch-style';
    let maxCount = 0;
    for (const [mutation, count] of mutationCounts) {
      if (count > maxCount) {
        maxCount = count;
        mostUsedMutation = mutation;
      }
    }

    // Success rates
    const mutationSuccessRates = new Map<MutationType, number>();
    for (const [mutation, count] of mutationCounts) {
      const successes = mutationSuccesses.get(mutation) || 0;
      mutationSuccessRates.set(mutation, successes / count);
    }

    return {
      totalVariations: variations.length,
      maxGeneration,
      bestVariation,
      avgScoreByGeneration,
      mostUsedMutation,
      mutationSuccessRates,
    };
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Generate unique variation ID
 */
export function generateVariationId(): string {
  return `var_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create initial variation (original prompt)
 */
export function createOriginalVariation(
  prompt: string,
  score: number,
  cost: number = 0,
  latency: number = 0
): VariationLineage {
  return {
    id: generateVariationId(),
    parentId: null,
    originalPrompt: prompt,
    currentPrompt: prompt,
    mutation: 'original',
    mutationParams: {},
    timestamp: new Date(),
    metrics: { score, cost, latency },
    children: [],
    generation: 0,
    path: [
      {
        mutation: 'original',
        variationId: '',
        score,
        improvement: 0,
      },
    ],
  };
}

/**
 * Create child variation from parent
 */
export function createChildVariation(
  parent: VariationLineage,
  mutatedPrompt: string,
  mutation: MutationType,
  mutationParams: Record<string, any>,
  score: number,
  cost: number,
  latency: number
): VariationLineage {
  const id = generateVariationId();
  const improvement = score - parent.metrics.score;

  const path: MutationStep[] = [
    ...parent.path,
    {
      mutation,
      variationId: id,
      score,
      improvement,
    },
  ];

  return {
    id,
    parentId: parent.id,
    originalPrompt: parent.originalPrompt,
    currentPrompt: mutatedPrompt,
    mutation,
    mutationParams,
    timestamp: new Date(),
    metrics: { score, cost, latency },
    children: [],
    generation: parent.generation + 1,
    path,
  };
}

/**
 * Format lineage path as string
 */
export function formatPath(lineage: VariationLineage[]): string {
  return lineage
    .map((v, idx) => {
      const improvement =
        idx > 0 ? `(+${(v.metrics.score - lineage[idx - 1].metrics.score).toFixed(3)})` : '';
      return `${v.mutation}${improvement}`;
    })
    .join(' → ');
}

/**
 * Visualize lineage tree as ASCII
 */
export function visualizeTree(graph: LineageGraph, maxDepth: number = 5): string {
  const lines: string[] = [];
  const visited = new Set<string>();

  function renderNode(node: LineageNode, prefix: string, isLast: boolean, depth: number): void {
    if (depth > maxDepth || visited.has(node.variation.id)) return;
    visited.add(node.variation.id);

    const connector = isLast ? '└─' : '├─';
    const mutation = node.variation.mutation;
    const score = node.variation.metrics.score.toFixed(3);
    const improvement =
      node.parent
        ? `(+${(node.variation.metrics.score - node.parent.variation.metrics.score).toFixed(3)})`
        : '';

    lines.push(`${prefix}${connector} ${mutation} [${score}] ${improvement}`);

    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    node.children.forEach((child, idx) => {
      renderNode(child, childPrefix, idx === node.children.length - 1, depth + 1);
    });
  }

  renderNode(graph.root, '', true, 0);
  return lines.join('\n');
}

// ============================================================================
// GLOBAL TRACKER INSTANCE
// ============================================================================

export const globalTracker = new LineageTracker();

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  LineageTracker,
  globalTracker,
  generateVariationId,
  createOriginalVariation,
  createChildVariation,
  formatPath,
  visualizeTree,
};
