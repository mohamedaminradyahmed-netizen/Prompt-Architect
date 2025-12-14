/**
 * Groq Provider - DIRECTIVE-045
 *
 * Provides integration with Groq's fast LLM inference API.
 * Groq offers significantly faster inference than other providers,
 * making it ideal for exploration and rapid evaluation phases.
 *
 * Usage:
 * - exploration phase: Use Groq (fastest and cheapest)
 * - final evaluation: Use OpenAI/Anthropic (better quality)
 *
 * Supported Models:
 * - llama-3.1-70b-versatile (high quality)
 * - llama-3.1-8b-instant (fastest)
 * - llama-3.2-90b-vision-preview (multimodal)
 * - mixtral-8x7b-32768 (good balance)
 */

import Groq from 'groq-sdk';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Configuration for completion requests
 */
export interface CompletionConfig {
  model?: GroqModel;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  stop?: string[];
  systemPrompt?: string;
  stream?: boolean;
}

/**
 * Supported Groq models
 */
export type GroqModel =
  | 'llama-3.1-70b-versatile'
  | 'llama-3.1-8b-instant'
  | 'llama-3.2-90b-vision-preview'
  | 'llama-3.3-70b-versatile'
  | 'mixtral-8x7b-32768'
  | 'gemma2-9b-it';

/**
 * Model configuration details
 */
export interface GroqModelConfig {
  name: GroqModel;
  displayName: string;
  contextWindow: number;
  costPer1kInputTokens: number;
  costPer1kOutputTokens: number;
  avgLatencyMs: number;
  qualityScore: number;
  supportsVision: boolean;
}

/**
 * Completion response
 */
export interface CompletionResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  model: string;
  finishReason: string;
  latencyMs: number;
}

/**
 * Provider interface that GroqProvider implements
 */
export interface LLMProviderInterface {
  complete(prompt: string, config?: CompletionConfig): Promise<string>;
  completeWithMetadata(prompt: string, config?: CompletionConfig): Promise<CompletionResponse>;
  embed(text: string): Promise<number[]>;
  estimateCost(tokens: number): number;
  estimateLatency(tokens: number): number;
}

// ============================================================================
// MODEL REGISTRY
// ============================================================================

/**
 * Pre-configured Groq models with their specifications
 */
export const GROQ_MODELS: Record<GroqModel, GroqModelConfig> = {
  'llama-3.1-70b-versatile': {
    name: 'llama-3.1-70b-versatile',
    displayName: 'Llama 3.1 70B Versatile',
    contextWindow: 131072,
    costPer1kInputTokens: 0.00059,
    costPer1kOutputTokens: 0.00079,
    avgLatencyMs: 500,
    qualityScore: 0.85,
    supportsVision: false,
  },
  'llama-3.1-8b-instant': {
    name: 'llama-3.1-8b-instant',
    displayName: 'Llama 3.1 8B Instant',
    contextWindow: 131072,
    costPer1kInputTokens: 0.00005,
    costPer1kOutputTokens: 0.00008,
    avgLatencyMs: 200,
    qualityScore: 0.70,
    supportsVision: false,
  },
  'llama-3.2-90b-vision-preview': {
    name: 'llama-3.2-90b-vision-preview',
    displayName: 'Llama 3.2 90B Vision Preview',
    contextWindow: 131072,
    costPer1kInputTokens: 0.0009,
    costPer1kOutputTokens: 0.0009,
    avgLatencyMs: 800,
    qualityScore: 0.88,
    supportsVision: true,
  },
  'llama-3.3-70b-versatile': {
    name: 'llama-3.3-70b-versatile',
    displayName: 'Llama 3.3 70B Versatile',
    contextWindow: 131072,
    costPer1kInputTokens: 0.00059,
    costPer1kOutputTokens: 0.00079,
    avgLatencyMs: 500,
    qualityScore: 0.87,
    supportsVision: false,
  },
  'mixtral-8x7b-32768': {
    name: 'mixtral-8x7b-32768',
    displayName: 'Mixtral 8x7B',
    contextWindow: 32768,
    costPer1kInputTokens: 0.00024,
    costPer1kOutputTokens: 0.00024,
    avgLatencyMs: 400,
    qualityScore: 0.78,
    supportsVision: false,
  },
  'gemma2-9b-it': {
    name: 'gemma2-9b-it',
    displayName: 'Gemma 2 9B IT',
    contextWindow: 8192,
    costPer1kInputTokens: 0.0002,
    costPer1kOutputTokens: 0.0002,
    avgLatencyMs: 300,
    qualityScore: 0.72,
    supportsVision: false,
  },
};

/**
 * Default model for general use
 */
export const DEFAULT_MODEL: GroqModel = 'llama-3.1-70b-versatile';

/**
 * Default model for speed-optimized use
 */
export const FAST_MODEL: GroqModel = 'llama-3.1-8b-instant';

// ============================================================================
// GROQ PROVIDER CLASS
// ============================================================================

/**
 * Groq Provider for fast LLM inference
 *
 * Implements the LLMProvider interface for integration with
 * the Prompt Refiner system.
 */
export class GroqProvider implements LLMProviderInterface {
  private client: Groq;
  private defaultModel: GroqModel;
  private defaultTemperature: number;
  private defaultMaxTokens: number;

  constructor(config?: {
    apiKey?: string;
    defaultModel?: GroqModel;
    defaultTemperature?: number;
    defaultMaxTokens?: number;
  }) {
    // Initialize Groq client - will use GROQ_API_KEY env var if apiKey not provided
    this.client = new Groq({
      apiKey: config?.apiKey || process.env.GROQ_API_KEY,
    });

    this.defaultModel = config?.defaultModel || DEFAULT_MODEL;
    this.defaultTemperature = config?.defaultTemperature ?? 0.7;
    this.defaultMaxTokens = config?.defaultMaxTokens ?? 4096;
  }

  // ==========================================================================
  // CORE METHODS
  // ==========================================================================

  /**
   * Complete a prompt and return the response content
   */
  async complete(prompt: string, config?: CompletionConfig): Promise<string> {
    const response = await this.completeWithMetadata(prompt, config);
    return response.content;
  }

  /**
   * Complete a prompt and return full response with metadata
   */
  async completeWithMetadata(
    prompt: string,
    config?: CompletionConfig
  ): Promise<CompletionResponse> {
    const startTime = Date.now();
    const model = config?.model || this.defaultModel;

    const messages: Groq.Chat.ChatCompletionMessageParam[] = [];

    // Add system prompt if provided
    if (config?.systemPrompt) {
      messages.push({
        role: 'system',
        content: config.systemPrompt,
      });
    }

    // Add user prompt
    messages.push({
      role: 'user',
      content: prompt,
    });

    try {
      const completion = await this.client.chat.completions.create({
        model,
        messages,
        temperature: config?.temperature ?? this.defaultTemperature,
        max_tokens: config?.maxTokens ?? this.defaultMaxTokens,
        top_p: config?.topP ?? 1,
        stop: config?.stop,
        stream: false,
      });

      const latencyMs = Date.now() - startTime;
      const choice = completion.choices[0];

      return {
        content: choice.message.content || '',
        usage: {
          promptTokens: completion.usage?.prompt_tokens || 0,
          completionTokens: completion.usage?.completion_tokens || 0,
          totalTokens: completion.usage?.total_tokens || 0,
        },
        model: completion.model,
        finishReason: choice.finish_reason || 'unknown',
        latencyMs,
      };
    } catch (error) {
      // Handle specific Groq errors
      if (error instanceof Groq.APIError) {
        throw new GroqProviderError(
          `Groq API Error: ${error.message}`,
          error.status,
          (error as unknown as { code?: string }).code
        );
      }
      throw error;
    }
  }

  /**
   * Generate embeddings for text
   *
   * Note: Groq doesn't currently offer embedding models directly.
   * This implementation uses a simple fallback approach.
   * For production, consider using a dedicated embedding provider
   * like OpenAI or a local model.
   */
  async embed(text: string): Promise<number[]> {
    // Groq doesn't have native embedding support
    // Use a simple hash-based embedding as fallback
    // In production, integrate with an actual embedding service
    console.warn(
      'GroqProvider.embed: Groq does not support embeddings natively. ' +
        'Using fallback hash-based embedding. Consider using OpenAI or another embedding provider.'
    );

    return this.fallbackEmbed(text);
  }

  /**
   * Estimate cost for a given number of tokens
   * Uses average of input/output costs for simplicity
   */
  estimateCost(tokens: number): number {
    const modelConfig = GROQ_MODELS[this.defaultModel];
    const avgCostPer1kTokens =
      (modelConfig.costPer1kInputTokens + modelConfig.costPer1kOutputTokens) / 2;
    return (tokens / 1000) * avgCostPer1kTokens;
  }

  /**
   * Estimate cost with separate input/output token counts
   */
  estimateCostDetailed(inputTokens: number, outputTokens: number): number {
    const modelConfig = GROQ_MODELS[this.defaultModel];
    const inputCost = (inputTokens / 1000) * modelConfig.costPer1kInputTokens;
    const outputCost = (outputTokens / 1000) * modelConfig.costPer1kOutputTokens;
    return inputCost + outputCost;
  }

  /**
   * Estimate latency for a given number of tokens
   */
  estimateLatency(tokens: number): number {
    const modelConfig = GROQ_MODELS[this.defaultModel];
    // Base latency + token generation time
    // Groq is very fast, roughly 100-500 tokens per second depending on model
    const tokensPerSecond = modelConfig.name.includes('8b') ? 500 : 200;
    const generationTime = (tokens / tokensPerSecond) * 1000;
    return modelConfig.avgLatencyMs + generationTime;
  }

  // ==========================================================================
  // UTILITY METHODS
  // ==========================================================================

  /**
   * Get model configuration
   */
  getModelConfig(model?: GroqModel): GroqModelConfig {
    return GROQ_MODELS[model || this.defaultModel];
  }

  /**
   * List available models
   */
  listModels(): GroqModelConfig[] {
    return Object.values(GROQ_MODELS);
  }

  /**
   * Set default model
   */
  setDefaultModel(model: GroqModel): void {
    if (!GROQ_MODELS[model]) {
      throw new Error(`Unknown model: ${model}`);
    }
    this.defaultModel = model;
  }

  /**
   * Check if provider is properly configured
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.complete('Say "ok"', {
        maxTokens: 5,
        model: 'llama-3.1-8b-instant',
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get provider name
   */
  getName(): string {
    return 'groq';
  }

  /**
   * Simple token estimation (roughly 4 chars per token)
   */
  estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Fallback embedding using simple hash
   * This is NOT suitable for semantic search - just a placeholder
   */
  private fallbackEmbed(text: string): number[] {
    const dimensions = 384; // Common embedding dimension
    const embedding = new Array(dimensions).fill(0);

    // Simple hash-based embedding (NOT for production semantic search)
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const index = (charCode + i * 31) % dimensions;
      embedding[index] += Math.sin(charCode) * 0.1;
    }

    // Normalize
    const magnitude = Math.sqrt(
      embedding.reduce((sum, val) => sum + val * val, 0)
    );
    if (magnitude > 0) {
      for (let i = 0; i < dimensions; i++) {
        embedding[i] /= magnitude;
      }
    }

    return embedding;
  }
}

// ============================================================================
// ERROR CLASSES
// ============================================================================

/**
 * Custom error class for Groq provider errors
 */
export class GroqProviderError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public code?: string
  ) {
    super(message);
    this.name = 'GroqProviderError';
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a speed-optimized Groq provider
 */
export function createFastGroqProvider(apiKey?: string): GroqProvider {
  return new GroqProvider({
    apiKey,
    defaultModel: 'llama-3.1-8b-instant',
    defaultTemperature: 0.5,
    defaultMaxTokens: 2048,
  });
}

/**
 * Create a quality-optimized Groq provider
 */
export function createQualityGroqProvider(apiKey?: string): GroqProvider {
  return new GroqProvider({
    apiKey,
    defaultModel: 'llama-3.1-70b-versatile',
    defaultTemperature: 0.7,
    defaultMaxTokens: 4096,
  });
}

/**
 * Create a balanced Groq provider
 */
export function createBalancedGroqProvider(apiKey?: string): GroqProvider {
  return new GroqProvider({
    apiKey,
    defaultModel: 'mixtral-8x7b-32768',
    defaultTemperature: 0.7,
    defaultMaxTokens: 4096,
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

export default GroqProvider;
