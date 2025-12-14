/**
 * Providers Index
 *
 * Central export point for all LLM providers in the Prompt Architect system.
 * This module provides a unified interface for working with different LLM providers.
 */

// Groq Provider
export {
  GroqProvider,
  GroqProviderError,
  createFastGroqProvider,
  createQualityGroqProvider,
  createBalancedGroqProvider,
  GROQ_MODELS,
  DEFAULT_MODEL as GROQ_DEFAULT_MODEL,
  FAST_MODEL as GROQ_FAST_MODEL,
  type GroqModel,
  type GroqModelConfig,
  type CompletionConfig,
  type CompletionResponse,
  type LLMProviderInterface,
} from './groq';

/**
 * Provider type union
 */
export type ProviderType = 'groq' | 'openai' | 'anthropic' | 'local';

/**
 * Factory function to create a provider by type
 */
export function createProvider(
  type: ProviderType,
  config?: { apiKey?: string; model?: string }
): { complete: (prompt: string) => Promise<string> } {
  switch (type) {
    case 'groq':
      const { GroqProvider: GP } = require('./groq');
      return new GP({ apiKey: config?.apiKey });

    case 'openai':
    case 'anthropic':
    case 'local':
      throw new Error(
        `Provider '${type}' is not yet implemented. ` +
          `See DIRECTIVE-045 completion notes for planned providers.`
      );

    default:
      throw new Error(`Unknown provider type: ${type}`);
  }
}
