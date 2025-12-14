import { geneticOptimize, GeneticConfig, DEFAULT_GENETIC_CONFIG } from './genetic';
import { hillClimbingOptimize } from './hillClimbing';
import { bayesianOptimize, applyParameters } from './bayesian';
import { parsePromptToTemplate, templateToPrompt } from '../templates/templateParser';
import { ScoringFunction } from './types';
import { PromptTemplate } from '../templates/PromptTemplate';

export interface HybridConfig {
    explorationBudget: number; // For Genetic (Generations)
    refinementBudget: number;  // For HillClimbing (Iterations)
    finetuningBudget: number;  // For Bayesian (Iterations)
}

export interface HybridResult {
    finalPrompt: string;
    finalScore: number;
    trace: {
        stage: 'exploration' | 'refinement' | 'finetuning';
        prompt: string;
        score: number;
        details?: any;
    }[];
}

/**
 * Hybrid Optimizer (DIRECTIVE-024)
 * Combines Genetic (Global Search) -> Hill Climbing (Local Search) -> Bayesian (Parameter Tuning)
 */
export async function hybridOptimize(
    initialPrompt: string,
    scoringFunction: ScoringFunction,
    config: HybridConfig
): Promise<HybridResult> {

    const trace: HybridResult['trace'] = [];

    // ==========================================================
    // PHASE 1: EXPLORATION (Genetic Algorithm)
    // ==========================================================
    console.log("Starting Phase 1: Exploration (Genetic)...");

    // Config for genetic
    const genConfig: GeneticConfig = {
        ...DEFAULT_GENETIC_CONFIG,
        generations: config.explorationBudget,
        populationSize: 10 // Smaller pop for speed in hybrid
    };

    const geneticResult = await geneticOptimize(initialPrompt, scoringFunction, genConfig);
    const bestGeneticPrompt = geneticResult.bestPrompts[0].prompt; // Assuming [0] is best (sorted)

    trace.push({
        stage: 'exploration',
        prompt: bestGeneticPrompt,
        score: geneticResult.summary.bestOverallFitness,
        details: { generations: config.explorationBudget }
    });


    // ==========================================================
    // PHASE 2: REFINEMENT (Hill Climbing)
    // ==========================================================
    console.log("Starting Phase 2: Refinement (Hill Climbing)...");

    const hcResult = await hillClimbingOptimize(bestGeneticPrompt, config.refinementBudget, scoringFunction);
    const bestRefinedPrompt = hcResult.bestPrompt;

    trace.push({
        stage: 'refinement',
        prompt: bestRefinedPrompt,
        score: hcResult.bestScore,
        details: { iterations: config.refinementBudget }
    });


    // ==========================================================
    // PHASE 3: FINE-TUNING (Bayesian)
    // ==========================================================
    console.log("Starting Phase 3: Fine-tuning (Bayesian)...");

    let finalPrompt = bestRefinedPrompt;
    let finalScore = hcResult.bestScore;
    let bayesianDetails = null;

    try {
        // Parse the refined prompt to a template
        const template = parsePromptToTemplate(bestRefinedPrompt);

        // If template extraction failed (e.g. empty goal), we skip bayesian
        if (template.goal && template.goal.length > 10) {
            const bayesianResult = await bayesianOptimize(
                template,
                [],
                config.finetuningBudget
            );

            const finalTemplate = applyParameters(template, bayesianResult.parameters);
            finalPrompt = templateToPrompt(finalTemplate);
            finalScore = bayesianResult.expectedScore;
            bayesianDetails = bayesianResult.parameters;
        }
    } catch (e) {
        console.warn("Bayesian optimization skipped due to parsing error:", e);
    }

    // Since I can't reconstruct the Bayesian string easily without the update,
    // I'll just return the HC result as final for now, or assume 0 improvement from Bayesian 
    // until I fix the export.

    // Trace entry for Bayesian (placeholder)
    trace.push({
        stage: 'finetuning',
        prompt: finalPrompt,
        score: finalScore,
        details: bayesianDetails
    });

    return {
        finalPrompt,
        finalScore,
        trace
    };
}
