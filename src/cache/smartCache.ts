
import NodeCache from 'node-cache';
import * as crypto from 'crypto';

/**
 * Smart Caching System (DIRECTIVE-026)
 *
 * Multi-layer caching for Prompts, Embeddings, and Evaluations
 * to minimize API costs and latency.
 * Uses node-cache for in-memory caching with TTL and LRU eviction.
 */

interface CacheStatsState {
    hits: number;
    misses: number;
    keys: number;
    ksize: number;
    vsize: number;
}

export class SmartCache {
    private promptCache: NodeCache;
    private embeddingCache: NodeCache;
    private evaluationCache: NodeCache;

    constructor(
        private promptTTL: number = 7 * 24 * 60 * 60, // 7 days (seconds)
        private standardTTL: number = 24 * 60 * 60    // 24 hours (seconds)
    ) {
        // Prompt Cache: Long retention (7 days), larger capacity
        this.promptCache = new NodeCache({
            stdTTL: promptTTL,
            checkperiod: 600, // Check expired every 10 min
            useClones: false,
            maxKeys: 10000 // Limit memory usage (LRU eviction)
        });

        // Embedding Cache: Very long retention (embeddings don't change often)
        this.embeddingCache = new NodeCache({
            stdTTL: 0, // Infinite/Manual
            checkperiod: 0,
            useClones: false,
            maxKeys: 50000
        });

        // Evaluation Cache: Medium retention
        this.evaluationCache = new NodeCache({
            stdTTL: standardTTL,
            checkperiod: 600,
            useClones: false,
            maxKeys: 20000
        });
    }

    /**
     * Generate a deterministic SHA-256 hash key for a prompt + provider combo
     */
    private generateKey(content: string, prefix: string): string {
        const hash = crypto.createHash('sha256').update(content.trim()).digest('hex');
        return `${prefix}:${hash}`;
    }

    // --- Prompt Response Cache ---

    getPromptResponse(prompt: string, provider: string): string | undefined {
        const key = this.generateKey(prompt, provider);
        return this.promptCache.get<string>(key);
    }

    setPromptResponse(prompt: string, provider: string, output: string, ttl: number = this.promptTTL): void {
        const key = this.generateKey(prompt, provider);
        this.promptCache.set(key, output, ttl);
    }

    // --- Embedding Cache ---

    getEmbedding(text: string, model: string): number[] | undefined {
        const key = this.generateKey(text, model);
        return this.embeddingCache.get<number[]>(key);
    }

    setEmbedding(text: string, model: string, vector: number[]): void {
        const key = this.generateKey(text, model);
        this.embeddingCache.set(key, vector);
    }

    // --- Evaluation Score Cache ---
    // Useful for repeated mutation steps in Genetic/Hill-Climbing

    getEvaluationScore(prompt: string, metric: string): number | undefined {
        const key = this.generateKey(prompt, metric);
        return this.evaluationCache.get<number>(key);
    }

    setEvaluationScore(prompt: string, metric: string, score: number): void {
        const key = this.generateKey(prompt, metric);
        this.evaluationCache.set(key, score);
    }

    // --- Utils ---

    /**
     * Invalidate cache. 
     * @param target 'prompt' | 'embedding' | 'evaluation'
     */
    invalidate(target: 'prompt' | 'embedding' | 'evaluation'): void {
        if (target === 'prompt') this.promptCache.flushAll();
        else if (target === 'embedding') this.embeddingCache.flushAll();
        else if (target === 'evaluation') this.evaluationCache.flushAll();
    }

    getStats(): { prompt: CacheStatsState, embedding: CacheStatsState, evaluation: CacheStatsState } {
        return {
            prompt: this.promptCache.getStats(),
            embedding: this.embeddingCache.getStats(),
            evaluation: this.evaluationCache.getStats()
        };
    }
}

export const globalCache = new SmartCache();
