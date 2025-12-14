export interface PromptTemplate {
    role?: string;           // "You are a senior software engineer"
    goal: string;            // "Write a function that..."
    constraints?: string[];  // ["Must be in TypeScript", "Use async/await"]
    examples?: string[];     // ["Example 1: ...", "Example 2: ..."]
    format?: string;         // "Return as JSON", "Use markdown"
}
