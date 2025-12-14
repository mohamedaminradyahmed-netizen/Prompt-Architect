
import {
    tryCatchStyleMutation,
    reduceContextMutation,
    expandMutation,
    constrainMutation,
    PromptVariation
} from '../mutations';
import { PromptCategory, classifyPrompt } from '../types/promptTypes';
import { ScoringFunction } from './types';

// Mutation wrapper to unify interface
const MUTATION_ARMS = [
    { id: 'try-catch', fn: tryCatchStyleMutation },
    { id: 'reduce-context', fn: reduceContextMutation },
    { id: 'expand', fn: expandMutation },
    { id: 'constrain', fn: (p: string) => constrainMutation(p, classifyPrompt(p).category) }
];

export interface BanditResult {
    bestMutationId: string;
    bestPrompt: string;
    bestScore: number;
    armStats: Record<string, { pulls: number; avgReward: number; confidence: number }>;
}

/**
 * Multi-Armed Bandit Optimizer (UCB1)
 * 
 * balances exploration (trying less visited mutations) and exploitation (using best mutations).
 */
export async function banditOptimize(
    prompt: string,
    budget: number,
    scoringFunction: ScoringFunction
): Promise<BanditResult> {

    // 1. Initialize Arms
    const arms = MUTATION_ARMS.map(m => ({
        ...m,
        pulls: 0,
        totalReward: 0,
        avgReward: 0.0
    }));

    let bestScore = -Infinity;
    let bestPrompt = prompt;
    let bestMutationId = '';

    // Initial score of the base prompt
    const initialScore = await scoringFunction(prompt);
    let totalPulls = 0;

    // 2. Main Loop
    for (let i = 0; i < budget; i++) {
        let selectedArmIdx = -1;

        // UCB1 Selection Strategy
        // If any arm hasn't been pulled, pull it first
        const unpulled = arms.findIndex(a => a.pulls === 0);
        if (unpulled !== -1) {
            selectedArmIdx = unpulled;
        } else {
            // Calculate UCB values
            let maxUCB = -Infinity;
            for (let j = 0; j < arms.length; j++) {
                // UCB1 Formula: AvgReward + C * sqrt(ln(TotalPulls) / ArmPulls)
                // C is exploration constant, usually sqrt(2) ~ 1.41
                const c = 1.41;
                const exploitation = arms[j].avgReward;
                const exploration = c * Math.sqrt(Math.log(totalPulls) / arms[j].pulls);
                const ucb = exploitation + exploration;

                if (ucb > maxUCB) {
                    maxUCB = ucb;
                    selectedArmIdx = j;
                }
            }
        }

        const arm = arms[selectedArmIdx];

        // 3. Pull Arm (Apply Mutation)
        const mutationResult = arm.fn(prompt);
        // Note: In a stateless bandit, we always apply to the ORIGINAL prompt to test the mutation's quality.
        // In a contextual/stateful bandit or RL, we would move state. 
        // Directive 022 implies "Optimization", so we might want to chain?
        // But "Bandits" usually implies choosing the best *action*. 
        // MCTS implies chaining actions.
        // Let's assume this Bandit selects the best *single step* mutation strategy, 
        // OR we can define the Reward as improvement.

        // Let's Score
        const score = await scoringFunction(mutationResult.text);

        // Reward: For UCB, usually bounded 0-1 matches well.
        // We'll use the raw score if it's 0-1, or normalized.
        // Let's assume score is roughly 0-1 (as seen in tests).
        // If score improves over initial, we give high reward?
        // Simpler: Reward = Score.
        const reward = score;

        // 4. Update Stats
        arm.pulls++;
        arm.totalReward += reward;
        arm.avgReward = arm.totalReward / arm.pulls;
        totalPulls++;

        // Track global best
        if (score > bestScore) {
            bestScore = score;
            bestPrompt = mutationResult.text;
            bestMutationId = arm.id;
        }
    }

    // Prepare Output
    const armStats: Record<string, any> = {};
    arms.forEach(a => {
        armStats[a.id] = {
            pulls: a.pulls,
            avgReward: a.avgReward,
            confidence: 1.41 * Math.sqrt(Math.log(totalPulls) / a.pulls)
        };
    });

    return {
        bestMutationId,
        bestPrompt,
        bestScore: bestScore > initialScore ? bestScore : initialScore,
        armStats
    };
}
