import { PromptTemplate } from './PromptTemplate';

export function parsePromptToTemplate(prompt: string): PromptTemplate {
    const lines = prompt.split('\n');
    const template: PromptTemplate = {
        goal: '',
        constraints: [],
        examples: [],
    };

    let currentSection: 'goal' | 'constraints' | 'examples' = 'goal';
    let goalLines: string[] = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        // Detect Role (simple heuristic: starts with "You are" or "Role:")
        if (line.toLowerCase().startsWith('you are') || line.toLowerCase().startsWith('role:')) {
            template.role = line.replace(/^(Role:)/i, '').trim();
            continue;
        }

        // Detect Format
        if (line.toLowerCase().startsWith('format:') || line.toLowerCase().startsWith('output format:')) {
            template.format = line.replace(/^(Output )?Format:/i, '').trim();
            continue;
        }

        // Detect Sections
        if (/^(constraints|requirements|rules):/i.test(line)) {
            currentSection = 'constraints';
            continue;
        }
        if (/^(examples):/i.test(line)) {
            currentSection = 'examples';
            continue;
        }

        // Process Content based on section
        if (currentSection === 'constraints') {
            // Remove bullet points if present
            const cleanLine = line.replace(/^[-*•]\s*/, '');
            template.constraints?.push(cleanLine);
        } else if (currentSection === 'examples') {
            // Examples might keep bullet points or not, usually separating them is good
            const cleanLine = line.replace(/^[-*•]\s*/, '');
            template.examples?.push(cleanLine);
        } else {
            // If we are in 'goal' section, collect lines
            goalLines.push(line);
        }
    }

    template.goal = goalLines.join('\n');
    return template;
}

export function templateToPrompt(template: PromptTemplate): string {
    const parts: string[] = [];

    if (template.role) {
        parts.push(template.role);
    }

    if (template.goal) {
        parts.push(template.goal);
    }

    if (template.constraints && template.constraints.length > 0) {
        parts.push('\nConstraints:');
        template.constraints.forEach(c => parts.push(`- ${c}`));
    }

    if (template.examples && template.examples.length > 0) {
        parts.push('\nExamples:');
        template.examples.forEach(e => parts.push(`- ${e}`));
    }

    if (template.format) {
        parts.push(`\nFormat: ${template.format}`);
    }

    return parts.join('\n');
}
