import {
    tryCatchStyleMutation,
    reduceContextMutation,
    expandMutation,
    constrainMutation
} from '../mutations';
import { PromptCategory, classifyPrompt } from '../types/promptTypes';
import { ScoringFunction } from './types';

// Actions available for the agent
const ACTIONS = [
    { name: 'try-catch', fn: tryCatchStyleMutation },
    { name: 'reduce', fn: reduceContextMutation },
    { name: 'expand', fn: expandMutation },
    { name: 'constrain', fn: (p: string) => constrainMutation(p, classifyPrompt(p).category) }
];

class MCTSNode {
    prompt: string;
    parent: MCTSNode | null;
    children: MCTSNode[];
    visits: number;
    value: number; // accumulated score/reward
    depth: number;
    causedByAction: string;

    constructor(prompt: string, parent: MCTSNode | null = null, causedByAction: string = 'root', depth: number = 0) {
        this.prompt = prompt;
        this.parent = parent;
        this.children = [];
        this.visits = 0;
        this.value = 0;
        this.causedByAction = causedByAction;
        this.depth = depth;
    }

    isLeaf(): boolean {
        return this.children.length === 0;
    }

    ucb1(explorationConstant: number = 1.41): number {
        if (this.visits === 0) return Infinity;
        // value is total, so avg is value/visits
        return (this.value / this.visits) + explorationConstant * Math.sqrt(Math.log(this.parent?.visits || 1) / this.visits);
    }
}

export interface MCTSResult {
    bestPrompt: string;
    bestScore: number;
    path: string[];
    iterations: number;
}

/**
 * Monte Carlo Tree Search Optimizer (DIRECTIVE-022)
 * Explores sequences of mutations to find optimal path.
 */
export async function mctsOptimize(
    initialPrompt: string,
    iterations: number = 20,
    maxDepth: number = 5,
    scoringFunction: ScoringFunction
): Promise<MCTSResult> {

    const root = new MCTSNode(initialPrompt);
    let bestGlobalNode = root;
    let bestGlobalScore = -Infinity;
    let bestPrompt = initialPrompt;

    // We can evaluate root immediately
    bestGlobalScore = await scoringFunction(root.prompt);

    for (let i = 0; i < iterations; i++) {
        let node = root;

        // 1. Selection
        while (!node.isLeaf() && node.children.length === ACTIONS.length) {
            // All children expanded, assume Fully Expanded -> Select child with best UCB
            node = node.children.reduce((prev, curr) => prev.ucb1() > curr.ucb1() ? prev : curr);
        }

        // 2. Expansion
        if (node.depth < maxDepth) {
            // Try to add a child that hasn't been added yet
            // Which actions are already taken?
            const takenActions = new Set(node.children.map(c => c.causedByAction));
            const availableActions = ACTIONS.filter(a => !takenActions.has(a.name));

            if (availableActions.length > 0) {
                // Pick a random available action
                const action = availableActions[Math.floor(Math.random() * availableActions.length)];
                const result = action.fn(node.prompt);

                const child = new MCTSNode(result.text, node, action.name, node.depth + 1);
                node.children.push(child);
                node = child;
            }
        }

        // 3. Simulation (Rollout)
        // From 'node', perform random steps until maxDepth
        let currentPrompt = node.prompt;
        let depth = node.depth;
        while (depth < maxDepth) {
            const action = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
            currentPrompt = action.fn(currentPrompt).text;
            depth++;
        }

        // Evaluate Terminal State
        const score = await scoringFunction(currentPrompt);

        // Update Global Best
        // Note: In MCTS, we might track best node visited, not just rollout result
        // But for optimization goal, any prompt we generated is a candidate.
        if (score > bestGlobalScore) {
            bestGlobalScore = score;
            // Note: 'currentPrompt' is the result of simulation. 
            // We can't easily reconstruct the *exact* path to it unless we stored it.
            // But we know 'node' led to it.
            // For simplicity, let's treat 'node.prompt' as the candidate to store if MCTS focuses on nodes.
            // Actually, usually we take the best *node* in tree.
            // Let's check node.prompt explicitly too.
        }

        // Let's verify the `node` prompt itself (the one we expanded)
        const nodeScore = await scoringFunction(node.prompt);
        if (nodeScore > bestGlobalScore) {
            bestGlobalScore = nodeScore;
            bestGlobalNode = node;
            bestPrompt = node.prompt; // explicit set
        }

        // 4. Backpropagation
        while (node) {
            node.visits++;
            node.value += score; // Propagate the rollout result
            node = node.parent!;
        }
    }

    // Trace path of best node
    const path: string[] = [];
    let tracer: MCTSNode | null = bestGlobalNode;
    while (tracer && tracer.parent) {
        path.unshift(tracer.causedByAction);
        tracer = tracer.parent;
    }
    path.unshift('root');

    return {
        bestPrompt: bestGlobalNode.prompt,
        bestScore: bestGlobalScore,
        path,
        iterations
    };
}
