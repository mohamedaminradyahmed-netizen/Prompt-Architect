export type ScoringFunction = (prompt: string) => Promise<number>;

export interface OptimizationResult {
    bestPrompt: string;
    bestScore: number;
    iterations: number;
    history: { prompt: string; score: number; mutation?: string; generation?: number }[];
}
