import { PromptCategory } from '../types/promptTypes';

export const CONSTRAINT_LIBRARY: Record<PromptCategory, string[]> = {
    [PromptCategory.CODE_GENERATION]: [
        "Use TypeScript",
        "Add error handling",
        "Include unit tests",
        "Follow SOLID principles",
        "Use async/await pattern",
        "Add comments for complex logic",
        "Ensure type safety"
    ],
    [PromptCategory.CODE_REVIEW]: [
        "Focus on security vulnerabilities",
        "Check for performance bottlenecks",
        "Verify adherence to style guide",
        "Look for edge cases",
        "Suggest refactoring for readability",
        "Check for memory leaks"
    ],
    [PromptCategory.CONTENT_WRITING]: [
        "Max 500 words",
        "Use active voice",
        "Grade level 8 readability",
        "Use short paragraphs",
        "Avoid jargon",
        "Include bullet points for key ideas",
        "Maintain a neutral tone"
    ],
    [PromptCategory.MARKETING_COPY]: [
        "Include a strong Call-to-Action (CTA)",
        "Focus on benefits, not features",
        "Use emotional triggers",
        "Create a sense of urgency",
        "Use power words",
        "Address pain points directly",
        "Keep it punchy and concise"
    ],
    [PromptCategory.DATA_ANALYSIS]: [
        "Visualize key trends",
        "Highlight outliers",
        "Provide statistical significance",
        "Explain methodology clearly",
        "Summarize data quality issues",
        "Focus on actionable insights"
    ],
    [PromptCategory.GENERAL_QA]: [
        "Be concise and direct",
        "Cite sources if applicable",
        "Avoid speculation",
        "Answer in a step-by-step format",
        "Check for logical consistency"
    ],
    [PromptCategory.CREATIVE_WRITING]: [
        "Show, don't tell",
        "Focus on sensory details",
        "Develop strong character voices",
        "Maintain consistent pacing",
        "Use vivid imagery",
        "Avoid clichés"
    ]
};

/**
 * Selects random constraints for a given category.
 * @param category The category to select constraints for.
 * @param count The number of constraints to select (default: 2).
 * @returns An array of selected constraints.
 */
export function getConstraintsForCategory(category: PromptCategory, count: number = 2): string[] {
    const constraints = CONSTRAINT_LIBRARY[category] || [];
    if (constraints.length === 0) return [];

    // Shuffle and pick
    const shuffled = [...constraints].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}
