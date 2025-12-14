import { PromptTemplate } from './PromptTemplate';

export function mutateTemplate(template: PromptTemplate, mutation: string): PromptTemplate {
    // Create a deep copy to avoid mutating the original
    const newTemplate: PromptTemplate = {
        ...template,
        constraints: [...(template.constraints || [])],
        examples: [...(template.examples || [])]
    };

    // 1. Handle JSON command for structured updates
    try {
        if (mutation.trim().startsWith('{')) {
            const command = JSON.parse(mutation);
            switch (command.type) {
                case 'set_role':
                    newTemplate.role = command.value;
                    break;
                case 'add_constraint':
                    newTemplate.constraints?.push(command.value);
                    break;
                case 'add_example':
                    newTemplate.examples?.push(command.value);
                    break;
                case 'set_format':
                    newTemplate.format = command.value;
                    break;
            }
            return newTemplate; // Return immediately if JSON parsed
        }
    } catch (e) {
        // Not JSON, continue to string parsing
    }

    // 2. Handle string commands/presets
    if (mutation.startsWith('set_role:')) {
        newTemplate.role = mutation.substring('set_role:'.length).trim();
    } else if (mutation.startsWith('add_constraint:')) {
        newTemplate.constraints?.push(mutation.substring('add_constraint:'.length).trim());
    } else if (mutation.startsWith('add_example:')) {
        newTemplate.examples?.push(mutation.substring('add_example:'.length).trim());
    } else if (mutation.startsWith('set_format:')) {
        newTemplate.format = mutation.substring('set_format:'.length).trim();
    } else {
        // Handle Presets
        switch (mutation) {
            case 'make_professional':
                newTemplate.role = "You are a highly experienced and professional expert in this field.";
                // Only set format if not already set, or append
                break;
            case 'enforce_json':
                newTemplate.format = "Return the result strictly as a valid JSON object.";
                newTemplate.constraints?.push("Do not include any markdown formatting or explanation outside the JSON.");
                break;
            case 'add_reasoning':
                newTemplate.constraints?.push("Let's think step by step.");
                break;
            case 'clear_constraints':
                newTemplate.constraints = [];
                break;
        }
    }

    return newTemplate;
}
