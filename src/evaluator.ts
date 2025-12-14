/**
 * Evaluator heuristics for prompt scoring
 * Combines token cost, similarity metrics, and latency specific metrics
 */

import { PromptVariation } from './mutations';

export interface ScoredSuggestion {
    prompt: string;
    mutation: string;
    score: number;
    tokenCount: number;
    estimatedCost: number;
    similarity: number;
    latency?: number; // Added per DIRECTIVE-010
}

/**
 * Calculate token count using simple word-based approximation
 */
export function calculateTokenCount(text: string): number {
    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;
    return Math.ceil(wordCount * 1.3);
}

/**
 * Estimate cost based on token count
 */
export function estimateCost(tokenCount: number): number {
    const costPer1kTokens = 0.03;
    return (tokenCount / 1000) * costPer1kTokens;
}

/**
 * Calculate cosine similarity between two texts
 */
export function calculateSimilarity(text1: string, text2: string): number {
    const normalize = (text: string) =>
        text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);

    const words1 = normalize(text1);
    const words2 = normalize(text2);

    const freq1 = new Map<string, number>();
    const freq2 = new Map<string, number>();

    words1.forEach(word => freq1.set(word, (freq1.get(word) || 0) + 1));
    words2.forEach(word => freq2.set(word, (freq2.get(word) || 0) + 1));

    const allWords = new Set([...freq1.keys(), ...freq2.keys()]);

    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    allWords.forEach(word => {
        const f1 = freq1.get(word) || 0;
        const f2 = freq2.get(word) || 0;
        dotProduct += f1 * f2;
        magnitude1 += f1 * f1;
        magnitude2 += f2 * f2;
    });

    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    if (magnitude1 === 0 || magnitude2 === 0) {
        return 0;
    }

    return dotProduct / (magnitude1 * magnitude2);
}

/**
 * Measure latency for a prompt with a specific provider (DIRECTIVE-010)
 * Note: This is an implementation using simulated values as we don't have API keys configured in this environment.
 */
export async function measureLatency(
    prompt: string,
    provider: 'openai' | 'anthropic' | 'groq' = 'openai'
): Promise<number> {
    // Determine base latency based on provider
    const baseLatencyMap = {
        'openai': 500,
        'anthropic': 800,
        'groq': 200
    };

    const baseLatency = baseLatencyMap[provider];

    // Simulate processing time based on prompt length
    const tokenCount = calculateTokenCount(prompt);
    const processingTime = tokenCount * 5; // ~5ms per token simulated

    // Add some random jitter
    const jitter = Math.random() * 200;

    // In a real implementation:
    // const start = Date.now();
    // await client.chat.completions.create({...});
    // return Date.now() - start;

    return Math.round(baseLatency + processingTime + jitter);
}

/**
 * Calculate overall score for a prompt variation
 */
export function calculateScore(
    similarity: number,
    tokenCount: number,
    maxTokens: number = 500,
    latency?: number
): number {
    const normalizedCost = Math.min(tokenCount / maxTokens, 1);
    const costScore = 1 - normalizedCost;

    // Basic scoring
    let score = (similarity * 0.7) + (costScore * 0.3);

    // Penalize high latency if available
    if (latency && latency > 2000) {
        score = score * 0.9;
    }

    return Math.round(score * 100);
}

/**
 * Evaluate a list of prompt variations and return scored suggestions
 */
export async function evaluateSuggestions(
    originalPrompt: string,
    variations: PromptVariation[]
): Promise<ScoredSuggestion[]> {
    const maxTokens = Math.max(...variations.map(v => calculateTokenCount(v.text)), 100);

    const scoredPromises = variations.map(async variation => {
        const tokenCount = calculateTokenCount(variation.text);
        const estimatedCost = estimateCost(tokenCount);
        const similarity = calculateSimilarity(originalPrompt, variation.text);

        // Measure latency
        const latency = await measureLatency(variation.text);

        const score = calculateScore(similarity, tokenCount, maxTokens, latency);

        return {
            prompt: variation.text,
            mutation: variation.mutationType,
            score,
            tokenCount,
            estimatedCost,
            similarity,
            latency
        };
    });

    const scored = await Promise.all(scoredPromises);

    // Sort by score descending
    return scored.sort((a, b) => b.score - a.score);
}
