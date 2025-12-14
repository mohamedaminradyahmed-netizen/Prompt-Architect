/**
 * Surrogate Orchestrator - DIRECTIVE-037
 *
 * Uses lightweight/fast models for initial evaluation to reduce costs.
 * Strategy:
 * - exploration: Use cheapest model (Groq/Llama)
 * - exploitation: Use mid-tier model (GPT-3.5)
 * - final: Use best model (GPT-4/Claude)
 *
 * Expected cost reduction: 60-80%
 *
 * Updated with DIRECTIVE-045: Groq Provider Integration
 * - Now supports real Groq API calls via GroqProvider
 * - Enable real API calls by setting useRealAPIs: true in constructor
 */

import { PromptCategory } from '../types/promptTypes';
import { BalanceMetrics, BALANCED } from '../config/balanceMetrics';
import { GroqProvider } from '../providers/groq';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Supported LLM providers
 */
export type LLMProvider = 'groq' | 'openai' | 'anthropic' | 'local';

/**
 * Supported model tiers
 */
export type ModelTier = 'cheap' | 'mid' | 'premium';

/**
 * Evaluation modes
 */
export type EvaluationMode = 'exploration' | 'exploitation' | 'final';

/**
 * Model configuration
 */
export interface ModelConfig {
  provider: LLMProvider;
  model: string;
  tier: ModelTier;
  costPer1kTokens: number;      // USD per 1K tokens
  avgLatencyMs: number;          // Average latency in ms
  qualityScore: number;          // Expected quality (0-1)
  maxTokens: number;             // Max tokens per request
  temperature?: number;
  topP?: number;
}

/**
 * Evaluation request
 */
export interface EvaluationRequest {
  prompt: string;
  context?: string;
  category?: PromptCategory;
  expectedOutputLength?: number;
  balanceMetrics?: BalanceMetrics;
}

/**
 * Evaluation result
 */
export interface EvaluationResult {
  output: string;
  score: number;                  // Quality score (0-1)
  confidence: number;             // Confidence in result (0-1)
  model: ModelConfig;
  cost: number;                   // Actual cost in USD
  latency: number;                // Actual latency in ms
  tokens: {
    input: number;
    output: number;
    total: number;
  };
  metadata: {
    mode: EvaluationMode;
    cached: boolean;
    timestamp: Date;
    retries: number;
  };
}

/**
 * Batch evaluation result
 */
export interface BatchEvaluationResult {
  results: EvaluationResult[];
  totalCost: number;
  avgLatency: number;
  successRate: number;
  costSavings: number;           // Compared to using premium for all
}

/**
 * Model selection strategy result
 */
export interface ModelSelectionResult {
  model: ModelConfig;
  reason: string;
  alternativeModels: ModelConfig[];
  estimatedCost: number;
  estimatedLatency: number;
}

/**
 * Cache entry
 */
interface CacheEntry {
  result: EvaluationResult;
  timestamp: Date;
  hitCount: number;
}

/**
 * Orchestrator statistics
 */
export interface OrchestratorStats {
  totalRequests: number;
  cacheHits: number;
  cacheHitRate: number;
  totalCost: number;
  totalSavings: number;
  avgLatency: number;
  requestsByMode: Record<EvaluationMode, number>;
  requestsByProvider: Record<LLMProvider, number>;
}

// ============================================================================
// MODEL REGISTRY
// ============================================================================

/**
 * Pre-configured model options
 */
export const MODEL_REGISTRY: Record<string, ModelConfig> = {
  // Groq (Fast & Cheap)
  'groq-llama-8b': {
    provider: 'groq',
    model: 'llama-3.1-8b-instant',
    tier: 'cheap',
    costPer1kTokens: 0.0001,
    avgLatencyMs: 200,
    qualityScore: 0.70,
    maxTokens: 8192,
    temperature: 0.7,
  },
  'groq-llama-70b': {
    provider: 'groq',
    model: 'llama-3.1-70b-versatile',
    tier: 'mid',
    costPer1kTokens: 0.0008,
    avgLatencyMs: 500,
    qualityScore: 0.85,
    maxTokens: 8192,
    temperature: 0.7,
  },

  // OpenAI
  'openai-gpt35': {
    provider: 'openai',
    model: 'gpt-3.5-turbo',
    tier: 'mid',
    costPer1kTokens: 0.002,
    avgLatencyMs: 800,
    qualityScore: 0.82,
    maxTokens: 4096,
    temperature: 0.7,
  },
  'openai-gpt4': {
    provider: 'openai',
    model: 'gpt-4',
    tier: 'premium',
    costPer1kTokens: 0.03,
    avgLatencyMs: 2000,
    qualityScore: 0.95,
    maxTokens: 8192,
    temperature: 0.7,
  },
  'openai-gpt4-turbo': {
    provider: 'openai',
    model: 'gpt-4-turbo',
    tier: 'premium',
    costPer1kTokens: 0.02,
    avgLatencyMs: 1500,
    qualityScore: 0.94,
    maxTokens: 128000,
    temperature: 0.7,
  },

  // Anthropic
  'anthropic-haiku': {
    provider: 'anthropic',
    model: 'claude-3-haiku-20240307',
    tier: 'cheap',
    costPer1kTokens: 0.00025,
    avgLatencyMs: 300,
    qualityScore: 0.75,
    maxTokens: 4096,
    temperature: 0.7,
  },
  'anthropic-sonnet': {
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
    tier: 'mid',
    costPer1kTokens: 0.003,
    avgLatencyMs: 1000,
    qualityScore: 0.90,
    maxTokens: 8192,
    temperature: 0.7,
  },
  'anthropic-opus': {
    provider: 'anthropic',
    model: 'claude-3-opus-20240229',
    tier: 'premium',
    costPer1kTokens: 0.015,
    avgLatencyMs: 2500,
    qualityScore: 0.96,
    maxTokens: 4096,
    temperature: 0.7,
  },

  // Local (Placeholder)
  'local-llama': {
    provider: 'local',
    model: 'llama-local',
    tier: 'cheap',
    costPer1kTokens: 0,
    avgLatencyMs: 1000,
    qualityScore: 0.65,
    maxTokens: 4096,
    temperature: 0.7,
  },
};

/**
 * Default model selection per mode
 */
const DEFAULT_MODE_MODELS: Record<EvaluationMode, string> = {
  exploration: 'groq-llama-8b',
  exploitation: 'openai-gpt35',
  final: 'openai-gpt4-turbo',
};

// ============================================================================
// SURROGATE ORCHESTRATOR CLASS
// ============================================================================

/**
 * Orchestrates model selection and evaluation across different tiers
 * to optimize cost while maintaining quality
 */
export class SurrogateOrchestrator {
  private cache: Map<string, CacheEntry> = new Map();
  private stats: OrchestratorStats;
  private balanceMetrics: BalanceMetrics;
  private customModelMap: Record<EvaluationMode, string>;
  /**
   * Instance-level model registry.
   *
   * Why:
   * - `MODEL_REGISTRY` is exported as a shared default catalog.
   * - Mutating it from within an orchestrator instance breaks encapsulation and can leak state across
   *   tests/instances. لذا نأخذ نسخة لكل instance ونضيف عليها النماذج المخصصة محلياً فقط.
   */
  private modelRegistry: Record<string, ModelConfig>;
  private cacheTTL: number;
  private maxCacheSize: number;

  /**
   * Groq Provider for real API calls (DIRECTIVE-045)
   * When useRealAPIs is true, Groq models will use actual API calls
   */
  private groqProvider: GroqProvider | null = null;
  private useRealAPIs: boolean;

  constructor(config?: {
    balanceMetrics?: BalanceMetrics;
    modeModelMap?: Record<EvaluationMode, string>;
    cacheTTL?: number;           // Cache TTL in ms
    maxCacheSize?: number;       // Max cache entries
    useRealAPIs?: boolean;       // Enable real API calls (DIRECTIVE-045)
    groqApiKey?: string;         // Groq API key (optional, uses env var if not provided)
  }) {
    this.balanceMetrics = config?.balanceMetrics || BALANCED;
    // Clone defaults to avoid mutating shared constants across instances.
    this.customModelMap = { ...DEFAULT_MODE_MODELS, ...(config?.modeModelMap ?? {}) };
    // Clone shared catalog to an instance-level registry (avoid global mutations).
    this.modelRegistry = Object.fromEntries(
      Object.entries(MODEL_REGISTRY).map(([key, value]) => [key, { ...value }])
    ) as Record<string, ModelConfig>;
    this.cacheTTL = config?.cacheTTL || 24 * 60 * 60 * 1000; // 24 hours default
    this.maxCacheSize = config?.maxCacheSize || 1000;

    // DIRECTIVE-045: Real API support
    this.useRealAPIs = config?.useRealAPIs || false;
    if (this.useRealAPIs) {
      this.groqProvider = new GroqProvider({
        apiKey: config?.groqApiKey,
      });
    }

    this.stats = {
      totalRequests: 0,
      cacheHits: 0,
      cacheHitRate: 0,
      totalCost: 0,
      totalSavings: 0,
      avgLatency: 0,
      requestsByMode: { exploration: 0, exploitation: 0, final: 0 },
      requestsByProvider: { groq: 0, openai: 0, anthropic: 0, local: 0 },
    };
  }

  // ============================================================================
  // CORE EVALUATION
  // ============================================================================

  /**
   * Evaluate a prompt using the appropriate model based on mode
   */
  async evaluate(
    request: EvaluationRequest,
    mode: EvaluationMode
  ): Promise<EvaluationResult> {
    const startTime = Date.now();
    this.stats.totalRequests++;
    this.stats.requestsByMode[mode]++;

    // Check cache first
    const cacheKey = this.generateCacheKey(request, mode);
    const cachedResult = this.getCachedResult(cacheKey);
    if (cachedResult) {
      this.stats.cacheHits++;
      this.updateCacheHitRate();
      return {
        ...cachedResult,
        metadata: {
          ...cachedResult.metadata,
          cached: true,
        },
      };
    }

    // Select model based on mode
    const modelSelection = this.selectModel(mode, request);
    const model = modelSelection.model;
    this.stats.requestsByProvider[model.provider]++;

    // Execute evaluation (simulated - in production would call actual APIs)
    const result = await this.executeEvaluation(request, model, mode);

    // Update stats
    this.stats.totalCost += result.cost;
    this.stats.avgLatency = this.calculateRunningAverage(
      this.stats.avgLatency,
      result.latency,
      this.stats.totalRequests
    );

    // Calculate savings compared to premium
    const premiumCost = this.estimatePremiumCost(result.tokens.total);
    this.stats.totalSavings += Math.max(0, premiumCost - result.cost);

    // Cache result
    this.cacheResult(cacheKey, result);

    return result;
  }

  /**
   * Batch evaluate multiple prompts with intelligent mode selection
   */
  async evaluateBatch(
    requests: EvaluationRequest[],
    mode: EvaluationMode
  ): Promise<BatchEvaluationResult> {
    const results: EvaluationResult[] = [];
    let totalCost = 0;
    let totalLatency = 0;
    let successCount = 0;

    for (const request of requests) {
      try {
        const result = await this.evaluate(request, mode);
        results.push(result);
        totalCost += result.cost;
        totalLatency += result.latency;
        successCount++;
      } catch (error) {
        // Log error but continue with batch
        console.error('Batch evaluation error:', error);
      }
    }

    // Calculate what premium would have cost
    const totalTokens = results.reduce((sum, r) => sum + r.tokens.total, 0);
    const premiumCost = this.estimatePremiumCost(totalTokens);

    return {
      results,
      totalCost,
      avgLatency: successCount > 0 ? totalLatency / successCount : 0,
      successRate: requests.length > 0 ? successCount / requests.length : 0,
      costSavings: Math.max(0, premiumCost - totalCost),
    };
  }

  /**
   * Progressive evaluation - starts cheap, upgrades if needed
   */
  async progressiveEvaluate(
    request: EvaluationRequest,
    qualityThreshold: number = 0.8
  ): Promise<EvaluationResult> {
    // Start with exploration (cheapest)
    const explorationResult = await this.evaluate(request, 'exploration');

    // If quality is good enough, return
    if (explorationResult.score >= qualityThreshold) {
      return explorationResult;
    }

    // Try exploitation (mid-tier)
    const exploitationResult = await this.evaluate(request, 'exploitation');

    if (exploitationResult.score >= qualityThreshold) {
      return exploitationResult;
    }

    // Fall back to final (premium)
    return this.evaluate(request, 'final');
  }

  // ============================================================================
  // MODEL SELECTION
  // ============================================================================

  /**
   * Select the best model based on mode and request characteristics
   */
  selectModel(
    mode: EvaluationMode,
    request: EvaluationRequest
  ): ModelSelectionResult {
    const modelKey = this.customModelMap[mode];
    const model = this.modelRegistry[modelKey];

    if (!model) {
      throw new Error(`Model not found: ${modelKey}`);
    }

    // Get alternative models
    const alternatives = this.getAlternativeModels(mode, request);

    // Estimate cost based on expected tokens
    const estimatedTokens = this.estimateTokens(request);
    const estimatedCost = (estimatedTokens / 1000) * model.costPer1kTokens;

    return {
      model,
      reason: this.getModelSelectionReason(mode, model),
      alternativeModels: alternatives,
      estimatedCost,
      estimatedLatency: model.avgLatencyMs,
    };
  }

  /**
   * Get alternative models for a given mode
   *
   * Why (Bug Fix):
   * - المقارنة السابقة `m.model !== this.customModelMap[mode]` كانت تقارن:
   *   • m.model: اسم النموذج (مثل "gpt-4")
   *   • this.customModelMap[mode]: registry key (مثل "openai-gpt4")
   * - هذه المقارنة دائماً true، فلم يتم استبعاد النموذج الحالي.
   * - الحل: استخدام Object.entries للوصول لـ registry key ومقارنته مباشرة.
   */
  private getAlternativeModels(
    mode: EvaluationMode,
    request: EvaluationRequest
  ): ModelConfig[] {
    const targetTier = this.getModeTargetTier(mode);
    const currentModelKey = this.customModelMap[mode];
    
    return Object.entries(this.modelRegistry)
      .filter(([key, m]) => m.tier === targetTier && key !== currentModelKey)
      .map(([_, m]) => m)
      .slice(0, 3);
  }

  /**
   * Get target tier for evaluation mode
   */
  private getModeTargetTier(mode: EvaluationMode): ModelTier {
    switch (mode) {
      case 'exploration': return 'cheap';
      case 'exploitation': return 'mid';
      case 'final': return 'premium';
    }
  }

  /**
   * Get reason for model selection
   */
  private getModelSelectionReason(mode: EvaluationMode, model: ModelConfig): string {
    switch (mode) {
      case 'exploration':
        return `استكشاف: استخدام ${model.model} للتقييم السريع والرخيص (${model.costPer1kTokens}$/1K tokens)`;
      case 'exploitation':
        return `استغلال: استخدام ${model.model} للتوازن بين الجودة والتكلفة (جودة ${Math.round(model.qualityScore * 100)}%)`;
      case 'final':
        return `نهائي: استخدام ${model.model} للحصول على أفضل جودة (${Math.round(model.qualityScore * 100)}%)`;
    }
  }

  // ============================================================================
  // EXECUTION (Real APIs + Simulation fallback)
  // ============================================================================

  /**
   * Execute evaluation
   * Uses real API calls when useRealAPIs is true and provider supports it (DIRECTIVE-045)
   * Otherwise falls back to simulation
   */
  private async executeEvaluation(
    request: EvaluationRequest,
    model: ModelConfig,
    mode: EvaluationMode
  ): Promise<EvaluationResult> {
    const startTime = Date.now();

    // DIRECTIVE-045: Use real Groq API when available
    if (this.useRealAPIs && model.provider === 'groq' && this.groqProvider) {
      return this.executeGroqEvaluation(request, model, mode, startTime);
    }

    // Fallback to simulation for other providers or when real APIs are disabled
    return this.executeSimulatedEvaluation(request, model, mode, startTime);
  }

  /**
   * Execute evaluation using real Groq API (DIRECTIVE-045)
   */
  private async executeGroqEvaluation(
    request: EvaluationRequest,
    model: ModelConfig,
    mode: EvaluationMode,
    startTime: number
  ): Promise<EvaluationResult> {
    try {
      // Map model registry name to actual Groq model name
      const groqModelName = this.mapToGroqModel(model.model);

      const response = await this.groqProvider!.completeWithMetadata(request.prompt, {
        model: groqModelName as 'llama-3.1-70b-versatile' | 'llama-3.1-8b-instant',
        temperature: model.temperature,
        maxTokens: model.maxTokens,
        systemPrompt: request.context,
      });

      // Calculate cost based on actual usage
      const cost = (response.usage.totalTokens / 1000) * model.costPer1kTokens;

      // Calculate quality score based on model quality
      const baseScore = model.qualityScore;
      const variance = 0.05; // Lower variance for real responses
      const score = Math.max(0, Math.min(1, baseScore + (Math.random() - 0.5) * variance));

      // Calculate confidence
      const confidence = this.calculateConfidence(mode, model, score);

      return {
        output: response.content,
        score,
        confidence,
        model,
        cost,
        latency: response.latencyMs,
        tokens: {
          input: response.usage.promptTokens,
          output: response.usage.completionTokens,
          total: response.usage.totalTokens,
        },
        metadata: {
          mode,
          cached: false,
          timestamp: new Date(),
          retries: 0,
        },
      };
    } catch (error) {
      // If real API fails, fall back to simulation
      console.warn('Groq API call failed, falling back to simulation:', error);
      return this.executeSimulatedEvaluation(request, model, mode, startTime);
    }
  }

  /**
   * Map model registry names to actual Groq model names
   */
  private mapToGroqModel(modelRegistryName: string): string {
    const mapping: Record<string, string> = {
      'llama-3.1-8b-instant': 'llama-3.1-8b-instant',
      'llama-3.1-70b-versatile': 'llama-3.1-70b-versatile',
    };
    return mapping[modelRegistryName] || modelRegistryName;
  }

  /**
   * Execute simulated evaluation (fallback)
   */
  private async executeSimulatedEvaluation(
    request: EvaluationRequest,
    model: ModelConfig,
    mode: EvaluationMode,
    startTime: number
  ): Promise<EvaluationResult> {
    // Simulate network latency
    await this.simulateLatency(model.avgLatencyMs);

    // Calculate tokens (estimation)
    const inputTokens = this.countTokens(request.prompt + (request.context || ''));
    const outputTokens = request.expectedOutputLength
      ? Math.ceil(request.expectedOutputLength / 4)
      : Math.floor(inputTokens * 0.5);
    const totalTokens = inputTokens + outputTokens;

    // Calculate cost
    const cost = (totalTokens / 1000) * model.costPer1kTokens;

    // Simulate output generation
    const output = this.generateSimulatedOutput(request, model);

    // Calculate quality score (simulated based on model quality + randomness)
    const baseScore = model.qualityScore;
    const variance = 0.1;
    const score = Math.max(0, Math.min(1, baseScore + (Math.random() - 0.5) * variance));

    // Calculate confidence
    const confidence = this.calculateConfidence(mode, model, score);

    return {
      output,
      score,
      confidence,
      model,
      cost,
      latency: Date.now() - startTime,
      tokens: {
        input: inputTokens,
        output: outputTokens,
        total: totalTokens,
      },
      metadata: {
        mode,
        cached: false,
        timestamp: new Date(),
        retries: 0,
      },
    };
  }

  /**
   * Simulate network latency
   */
  private async simulateLatency(baseLatencyMs: number): Promise<void> {
    const variance = baseLatencyMs * 0.2;
    const actualLatency = baseLatencyMs + (Math.random() - 0.5) * variance;
    await new Promise(resolve => setTimeout(resolve, Math.max(10, actualLatency)));
  }

  /**
   * Generate simulated output (for testing)
   */
  private generateSimulatedOutput(
    request: EvaluationRequest,
    model: ModelConfig
  ): string {
    return `[Simulated output from ${model.model}]\n\n` +
           `Prompt analyzed: "${request.prompt.substring(0, 100)}..."\n\n` +
           `This is a placeholder response for testing the SurrogateOrchestrator.\n` +
           `In production, this would be the actual LLM response.`;
  }

  /**
   * Calculate confidence based on mode and model
   */
  private calculateConfidence(
    mode: EvaluationMode,
    model: ModelConfig,
    score: number
  ): number {
    let baseConfidence = model.qualityScore;

    // Adjust based on mode
    switch (mode) {
      case 'exploration':
        baseConfidence *= 0.7; // Lower confidence for cheap models
        break;
      case 'exploitation':
        baseConfidence *= 0.85;
        break;
      case 'final':
        baseConfidence *= 1.0; // Full confidence for premium
        break;
    }

    // Score-based adjustment
    if (score > 0.8) {
      baseConfidence *= 1.1;
    } else if (score < 0.5) {
      baseConfidence *= 0.8;
    }

    return Math.min(1, Math.max(0, baseConfidence));
  }

  // ============================================================================
  // CACHING
  // ============================================================================

  /**
   * Generate cache key
   */
  private generateCacheKey(request: EvaluationRequest, mode: EvaluationMode): string {
    const content = JSON.stringify({
      prompt: request.prompt,
      context: request.context,
      category: request.category,
      mode,
    });
    return this.hashString(content);
  }

  /**
   * Simple string hash
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return `cache_${hash.toString(36)}`;
  }

  /**
   * Get cached result if valid
   */
  private getCachedResult(key: string): EvaluationResult | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check TTL
    const age = Date.now() - entry.timestamp.getTime();
    if (age > this.cacheTTL) {
      this.cache.delete(key);
      return null;
    }

    entry.hitCount++;
    return entry.result;
  }

  /**
   * Cache a result
   */
  private cacheResult(key: string, result: EvaluationResult): void {
    // Evict if cache is full
    if (this.cache.size >= this.maxCacheSize) {
      this.evictLRU();
    }

    this.cache.set(key, {
      result,
      timestamp: new Date(),
      hitCount: 0,
    });
  }

  /**
   * Evict least recently used entries
   */
  private evictLRU(): void {
    const entries = Array.from(this.cache.entries())
      .sort((a, b) => a[1].hitCount - b[1].hitCount);

    // Remove bottom 10%
    const removeCount = Math.ceil(this.maxCacheSize * 0.1);
    for (let i = 0; i < removeCount && i < entries.length; i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  // ============================================================================
  // UTILITIES
  // ============================================================================

  /**
   * Estimate tokens for a request
   */
  private estimateTokens(request: EvaluationRequest): number {
    const text = request.prompt + (request.context || '');
    // Rough estimation: ~4 chars per token
    return Math.ceil(text.length / 4);
  }

  /**
   * Count tokens (simple estimation)
   */
  private countTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  /**
   * Estimate cost if using premium model
   */
  private estimatePremiumCost(tokens: number): number {
    const premiumModel = this.modelRegistry['openai-gpt4'];
    return (tokens / 1000) * premiumModel.costPer1kTokens;
  }

  /**
   * Calculate running average
   */
  private calculateRunningAverage(
    currentAvg: number,
    newValue: number,
    count: number
  ): number {
    return ((currentAvg * (count - 1)) + newValue) / count;
  }

  /**
   * Update cache hit rate
   */
  private updateCacheHitRate(): void {
    this.stats.cacheHitRate = this.stats.totalRequests > 0
      ? this.stats.cacheHits / this.stats.totalRequests
      : 0;
  }

  // ============================================================================
  // CONFIGURATION
  // ============================================================================

  /**
   * Set model for a specific mode
   */
  setModeModel(mode: EvaluationMode, modelKey: string): void {
    if (!this.modelRegistry[modelKey]) {
      throw new Error(`Unknown model: ${modelKey}`);
    }
    this.customModelMap[mode] = modelKey;
  }

  /**
   * Set balance metrics
   */
  setBalanceMetrics(metrics: BalanceMetrics): void {
    this.balanceMetrics = metrics;
  }

  /**
   * Add custom model to registry
   */
  addCustomModel(key: string, config: ModelConfig): void {
    // Instance-level extension only (no global mutation).
    this.modelRegistry[key] = { ...config };
  }

  // ============================================================================
  // STATISTICS
  // ============================================================================

  /**
   * Get orchestrator statistics
   */
  getStats(): OrchestratorStats {
    return { ...this.stats };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      totalRequests: 0,
      cacheHits: 0,
      cacheHitRate: 0,
      totalCost: 0,
      totalSavings: 0,
      avgLatency: 0,
      requestsByMode: { exploration: 0, exploitation: 0, final: 0 },
      requestsByProvider: { groq: 0, openai: 0, anthropic: 0, local: 0 },
    };
  }

  /**
   * Get cost savings summary
   */
  getCostSavingsSummary(): {
    totalCost: number;
    estimatedPremiumCost: number;
    savings: number;
    savingsPercentage: number;
  } {
    const estimatedPremiumCost = this.stats.totalCost + this.stats.totalSavings;
    return {
      totalCost: this.stats.totalCost,
      estimatedPremiumCost,
      savings: this.stats.totalSavings,
      savingsPercentage: estimatedPremiumCost > 0
        ? (this.stats.totalSavings / estimatedPremiumCost) * 100
        : 0,
    };
  }

  /**
   * Get model usage breakdown
   */
  getModelUsageBreakdown(): {
    byProvider: Record<LLMProvider, number>;
    byMode: Record<EvaluationMode, number>;
    mostUsedProvider: LLMProvider;
    mostUsedMode: EvaluationMode;
  } {
    const providerEntries = Object.entries(this.stats.requestsByProvider) as [LLMProvider, number][];
    const modeEntries = Object.entries(this.stats.requestsByMode) as [EvaluationMode, number][];

    const mostUsedProvider = providerEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
    const mostUsedMode = modeEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];

    return {
      byProvider: this.stats.requestsByProvider,
      byMode: this.stats.requestsByMode,
      mostUsedProvider,
      mostUsedMode,
    };
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a cost-optimized orchestrator
 */
export function createCostOptimizedOrchestrator(): SurrogateOrchestrator {
  return new SurrogateOrchestrator({
    modeModelMap: {
      exploration: 'groq-llama-8b',
      exploitation: 'groq-llama-70b',
      final: 'openai-gpt35',
    },
  });
}

/**
 * Create a quality-focused orchestrator
 */
export function createQualityFocusedOrchestrator(): SurrogateOrchestrator {
  return new SurrogateOrchestrator({
    modeModelMap: {
      exploration: 'anthropic-haiku',
      exploitation: 'anthropic-sonnet',
      final: 'anthropic-opus',
    },
  });
}

/**
 * Create a balanced orchestrator
 */
export function createBalancedOrchestrator(): SurrogateOrchestrator {
  return new SurrogateOrchestrator(); // Uses defaults
}

// ============================================================================
// EXPORTS
// ============================================================================

export default SurrogateOrchestrator;
