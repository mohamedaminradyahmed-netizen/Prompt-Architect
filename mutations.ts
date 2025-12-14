/**
 * Mutation operators for prompt refinement
 * Simple transformations to create prompt variations
 */

export interface PromptVariation {
  prompt: string;
  mutation: string;
}

/**
 * Paraphrase mutation: rephrase the prompt while preserving meaning
 * Uses simple rephrasing rules
 */
export function paraphraseMutation(originalPrompt: string): PromptVariation {
  // Simple paraphrase rules - in a real implementation, this would use NLP
  const paraphrases = [
    (text: string) => text.replace(/^Write/, 'Create').replace(/^Create/, 'Write'),
    (text: string) => text.replace(/^Explain/, 'Describe').replace(/^Describe/, 'Explain'),
    (text: string) => text.replace(/^Help me/, 'Assist me with'),
    (text: string) => text.replace(/I need/, 'I would like'),
    (text: string) => text.replace(/about/, 'regarding'),
  ];

  let paraphrased = originalPrompt;
  const appliedRules = [];

  for (const rule of paraphrases) {
    const newText = rule(paraphrased);
    if (newText !== paraphrased) {
      paraphrased = newText;
      appliedRules.push('paraphrase');
      break; // Apply only one rule for simplicity
    }
  }

  return {
    prompt: paraphrased,
    mutation: 'paraphrase'
  };
}

/**
 * Shorten mutation: reduce prompt length while keeping key elements
 * Removes redundant words and simplifies structure
 */
export function shortenMutation(originalPrompt: string): PromptVariation {
  let shortened = originalPrompt;

  // Remove common filler words
  shortened = shortened.replace(/\b(please|kindly|can you|could you|would you)\b/gi, '');
  // Remove extra spaces
  shortened = shortened.replace(/\s+/g, ' ').trim();
  // Remove trailing punctuation if followed by more text
  shortened = shortened.replace(/([.!?])\s*([A-Z])/g, '$2');

  return {
    prompt: shortened,
    mutation: 'shorten'
  };
}

/**
 * Add constraint mutation: add helpful constraints or guidelines
 * Enhances the prompt with structure or requirements
 */
export function addConstraintMutation(originalPrompt: string): PromptVariation {
  const constraints = [
    ' Be specific and provide examples where appropriate.',
    ' Structure your response clearly with headings.',
    ' Include step-by-step instructions.',
    ' Use bullet points for clarity.',
    ' Provide a concise summary at the end.'
  ];

  // Randomly select a constraint (in real implementation, this could be smarter)
  const randomConstraint = constraints[Math.floor(Math.random() * constraints.length)];

  return {
    prompt: originalPrompt + randomConstraint,
    mutation: 'add_constraint'
  };
}

/**
 * Context Reduction mutation: reduces excessive context while preserving core meaning
 * DIRECTIVE-004: Removes secondary sentences, explanatory text, and long examples
 * Target: 30-50% length reduction while keeping essential constraints and technical info
 */
export function reduceContextMutation(originalPrompt: string): PromptVariation {
  let reduced = originalPrompt;
  const originalLength = originalPrompt.length;

  // 1. تحديد الجمل الثانوية أو التفسيرية وإزالتها
  // Remove explanatory phrases like "in other words", "that is to say", "for example"
  const explanatoryPatterns = [
    /\bin other words[,:]?\s*[^.!?]*[.!?]/gi,
    /\bthat is to say[,:]?\s*[^.!?]*[.!?]/gi,
    /\bto put it simply[,:]?\s*[^.!?]*[.!?]/gi,
    /\bbasically[,:]?\s*[^.!?]*[.!?]/gi,
    /\bessentially[,:]?\s*[^.!?]*[.!?]/gi,
    /\bin simple terms[,:]?\s*[^.!?]*[.!?]/gi,
    /\bto clarify[,:]?\s*[^.!?]*[.!?]/gi,
  ];

  for (const pattern of explanatoryPatterns) {
    reduced = reduced.replace(pattern, '');
  }

  // 2. استبدال الأمثلة الطويلة بإشارات مختصرة
  // Replace long example blocks with brief references
  const examplePatterns = [
    // Match "For example: ..." or "For instance: ..." with long text
    {
      pattern: /\bfor example[,:]?\s*([^.!?]{80,}[.!?])/gi,
      replacement: '(see examples).'
    },
    {
      pattern: /\bfor instance[,:]?\s*([^.!?]{80,}[.!?])/gi,
      replacement: '(see examples).'
    },
    // Match "e.g.," followed by long content
    {
      pattern: /\be\.g\.[,:]?\s*([^.!?]{60,}[.!?])/gi,
      replacement: '(e.g., see examples).'
    },
    // Match "such as" followed by long lists
    {
      pattern: /\bsuch as\s+([^.!?]{80,}[.!?])/gi,
      replacement: '(such as relevant examples).'
    },
    // Match "like" introducing examples
    {
      pattern: /\blike\s+([^.!?]{70,}[.!?])/gi,
      replacement: '(like relevant cases).'
    },
  ];

  for (const { pattern, replacement } of examplePatterns) {
    reduced = reduced.replace(pattern, replacement);
  }

  // 3. إزالة الشروح التي يمكن استنتاجها
  // Remove phrases that state the obvious or are inferable
  const inferablePatterns = [
    /\bas you (probably |may |might )?know[,:]?\s*/gi,
    /\bit (is|'s) (important|worth|good|helpful) (to note|to mention|noting|mentioning) that\s*/gi,
    /\bneedless to say[,:]?\s*/gi,
    /\bobviously[,:]?\s*/gi,
    /\bclearly[,:]?\s*/gi,
    /\bof course[,:]?\s*/gi,
    /\bit goes without saying (that )?\s*/gi,
    /\bas mentioned (earlier|above|before)[,:]?\s*/gi,
    /\bas we (all )?(know|understand)[,:]?\s*/gi,
    /\bI('m| am) sure you (know|understand) (that )?\s*/gi,
  ];

  for (const pattern of inferablePatterns) {
    reduced = reduced.replace(pattern, '');
  }

  // 4. إزالة التكرارات والجمل المتكررة
  // Remove redundant introductions and repeated content
  const redundantPatterns = [
    /\bI would like (you )?to\s*/gi,
    /\bI want (you )?to\s*/gi,
    /\bI need (you )?to\s*/gi,
    /\bI am asking (you )?to\s*/gi,
    /\bPlease note that\s*/gi,
    /\bPlease be aware that\s*/gi,
    /\bIt is (essential|crucial|vital|imperative) that\s*/gi,
  ];

  for (const pattern of redundantPatterns) {
    // Replace with simpler instructions where applicable
    reduced = reduced.replace(/\bI would like (you )?to\s*/gi, '');
    reduced = reduced.replace(/\bI want (you )?to\s*/gi, '');
    reduced = reduced.replace(/\bI need (you )?to\s*/gi, '');
    reduced = reduced.replace(/\bI am asking (you )?to\s*/gi, '');
    reduced = reduced.replace(/\bPlease note that\s*/gi, 'Note: ');
    reduced = reduced.replace(/\bPlease be aware that\s*/gi, '');
    reduced = reduced.replace(/\bIt is (essential|crucial|vital|imperative) that\s*/gi, 'Must ');
  }

  // 5. تنظيف المسافات الزائدة والترقيم
  // Clean up extra spaces and punctuation
  reduced = reduced.replace(/\s+/g, ' ').trim();
  reduced = reduced.replace(/\s+([.,!?;:])/g, '$1');
  reduced = reduced.replace(/([.,!?;:])\s*\1+/g, '$1');
  reduced = reduced.replace(/\s*\(\s*\)/g, '');
  reduced = reduced.replace(/^\s*[,;]\s*/g, '');
  reduced = reduced.replace(/\s*[,;]\s*$/g, '.');

  // 6. حساب نسبة التقليل
  const reductionRatio = ((originalLength - reduced.length) / originalLength) * 100;

  // إذا كان التقليل أقل من 10%، حاول المزيد من التقليل العدواني
  if (reductionRatio < 10 && originalLength > 100) {
    // Remove sentences that are purely descriptive (don't contain action verbs)
    const sentences = reduced.split(/(?<=[.!?])\s+/);
    const actionVerbs = /\b(create|write|build|implement|develop|design|generate|analyze|fix|update|add|remove|modify|explain|describe|list|show|provide|return|calculate|check|validate|ensure|make|do)\b/i;
    const constraintWords = /\b(must|should|need|require|constraint|limit|only|always|never|ensure|important)\b/i;

    const filteredSentences = sentences.filter(sentence =>
      actionVerbs.test(sentence) || constraintWords.test(sentence) || sentence.length < 30
    );

    if (filteredSentences.length > 0 && filteredSentences.length < sentences.length) {
      reduced = filteredSentences.join(' ');
    }
  }

  // Final cleanup
  reduced = reduced.replace(/\s+/g, ' ').trim();

  // Ensure we don't return empty string
  if (reduced.length < 10) {
    reduced = originalPrompt;
  }

  return {
    prompt: reduced,
    mutation: 'reduce_context'
  };
}

/**
 * Expand mutation: adds details and specificity to the prompt
 * DIRECTIVE-006: Expands prompts by 50-100% with increased clarity
 * Target: Add definitions, steps, examples, and success criteria
 */
export function expandMutation(originalPrompt: string): PromptVariation {
  let expanded = originalPrompt;
  const originalLength = originalPrompt.length;

  // 1. حدد المصطلحات التقنية → أضف تعريفات مختصرة
  // Add brief definitions for technical terms
  const technicalTerms = [
    { term: /\boptimiz(e|ation)\b/gi, definition: ' (improve performance and efficiency)' },
    { term: /\brefactor\b/gi, definition: ' (restructure code without changing functionality)' },
    { term: /\bAPI\b/g, definition: ' (Application Programming Interface)' },
    { term: /\bdatabase\b/gi, definition: ' (structured data storage system)' },
    { term: /\balgorithm\b/gi, definition: ' (step-by-step problem-solving procedure)' },
    { term: /\bframework\b/gi, definition: ' (pre-built code structure)' },
    { term: /\bresponsive\b/gi, definition: ' (adapts to different screen sizes)' },
    { term: /\bscalable\b/gi, definition: ' (handles increased load efficiently)' },
  ];

  for (const { term, definition } of technicalTerms) {
    if (term.test(expanded) && !expanded.includes(definition)) {
      expanded = expanded.replace(term, (match) => match + definition);
      break; // Add only one definition to avoid over-expansion
    }
  }

  // 2. حدد التعليمات العامة → أضف خطوات محددة
  // Add specific steps for general instructions
  const generalInstructions = [
    {
      pattern: /\b(write|create)\s+(a\s+)?(function|method)\b/gi,
      expansion: '. Follow these steps: 1) Define the function signature, 2) Implement the core logic, 3) Add error handling, 4) Include documentation.'
    },
    {
      pattern: /\b(build|develop|create)\s+(a\s+)?(website|app|application)\b/gi,
      expansion: '. Process: 1) Plan the architecture, 2) Set up the project structure, 3) Implement core features, 4) Add styling and UI, 5) Test functionality.'
    },
    {
      pattern: /\b(fix|debug|solve)\s+(the\s+)?(bug|issue|problem)\b/gi,
      expansion: '. Debugging approach: 1) Reproduce the issue, 2) Identify the root cause, 3) Implement the fix, 4) Test the solution, 5) Verify no new issues.'
    },
    {
      pattern: /\b(analyze|review|examine)\s+(the\s+)?code\b/gi,
      expansion: '. Analysis criteria: 1) Code quality and readability, 2) Performance implications, 3) Security considerations, 4) Best practices compliance.'
    },
    {
      pattern: /\b(design|plan)\s+(a\s+)?(system|architecture)\b/gi,
      expansion: '. Design process: 1) Gather requirements, 2) Define system boundaries, 3) Choose technologies, 4) Create component diagrams, 5) Plan data flow.'
    },
  ];

  for (const { pattern, expansion } of generalInstructions) {
    if (pattern.test(expanded) && !expanded.includes(expansion)) {
      expanded = expanded.replace(pattern, (match) => match + expansion);
      break;
    }
  }

  // 3. أضف أمثلة توضيحية إن لم تكن موجودة
  // Add illustrative examples if none exist
  if (!/\b(example|instance|such as|like|e\.g\.)\b/gi.test(expanded)) {
    const examplePrompts = [
      /\b(write|create).*function\b/gi,
      /\b(build|develop).*app\b/gi,
      /\b(design).*interface\b/gi,
    ];

    const examples = [
      ' For example, include input validation, return appropriate data types, and handle edge cases.',
      ' For instance, consider user authentication, data persistence, and responsive design.',
      ' Such as intuitive navigation, clear visual hierarchy, and accessibility features.',
    ];

    for (let i = 0; i < examplePrompts.length; i++) {
      if (examplePrompts[i].test(expanded)) {
        expanded += examples[i];
        break;
      }
    }
  }

  // 4. أضف معايير نجاح واضحة
  // Add clear success criteria
  if (!/\b(success|criteria|requirement|should|must|ensure)\b/gi.test(expanded)) {
    const successCriteria = [
      ' Success criteria: The solution should be functional, well-documented, and follow best practices.',
      ' Requirements: Ensure code is readable, maintainable, and properly tested.',
      ' Quality standards: Must be efficient, secure, and user-friendly.',
    ];

    const randomCriteria = successCriteria[Math.floor(Math.random() * successCriteria.length)];
    expanded += randomCriteria;
  }

  // 5. إضافة تفاصيل السياق إذا كان البرومبت قصيراً جداً
  // Add context details if prompt is very short
  if (originalLength < 50) {
    const contextEnhancements = [
      ' Provide clear explanations for your approach and reasoning.',
      ' Include relevant best practices and industry standards.',
      ' Consider performance, maintainability, and scalability in your solution.',
    ];

    const randomContext = contextEnhancements[Math.floor(Math.random() * contextEnhancements.length)];
    expanded += randomContext;
  }

  // 6. تحسين البنية والوضوح
  // Improve structure and clarity
  if (!expanded.includes(':') && expanded.length > 100) {
    // Add structure if the prompt is getting long but lacks organization
    const structureWords = ['Task:', 'Goal:', 'Objective:'];
    const randomStructure = structureWords[Math.floor(Math.random() * structureWords.length)];
    expanded = randomStructure + ' ' + expanded;
  }

  // 7. تنظيف وتحسين التنسيق
  // Clean up and improve formatting
  expanded = expanded.replace(/\s+/g, ' ').trim();
  expanded = expanded.replace(/\s+([.,!?;:])/g, '$1');
  expanded = expanded.replace(/([.,!?;:])\s*\1+/g, '$1');

  // Ensure proper sentence ending
  if (!/[.!?]$/.test(expanded)) {
    expanded += '.';
  }

  // Calculate expansion ratio
  const expansionRatio = ((expanded.length - originalLength) / originalLength) * 100;

  // If expansion is less than 30%, add more generic enhancement
  if (expansionRatio < 30 && originalLength > 20) {
    const genericEnhancements = [
      ' Please provide detailed explanations and justify your decisions.',
      ' Include step-by-step reasoning and consider alternative approaches.',
      ' Ensure your response is comprehensive and addresses potential edge cases.',
    ];

    const randomEnhancement = genericEnhancements[Math.floor(Math.random() * genericEnhancements.length)];
    expanded += randomEnhancement;
  }

  return {
    prompt: expanded,
    mutation: 'expand'
  };
}

/**
 * Apply a random mutation to create a variation
 */
export function applyRandomMutation(originalPrompt: string): PromptVariation {
  const mutations = [paraphraseMutation, shortenMutation, addConstraintMutation, reduceContextMutation, expandMutation];
  const randomMutation = mutations[Math.floor(Math.random() * mutations.length)];
  return randomMutation(originalPrompt);
}

/**
 * Generate multiple prompt variations
 */
export function generateVariations(originalPrompt: string, count: number = 3): PromptVariation[] {
  const variations: PromptVariation[] = [];

  // Always include the original as a baseline
  variations.push({
    prompt: originalPrompt,
    mutation: 'original'
  });

  // Generate additional variations
  while (variations.length < count) {
    const variation = applyRandomMutation(originalPrompt);
    // Avoid duplicates
    if (!variations.some(v => v.prompt === variation.prompt)) {
      variations.push(variation);
    }
  }

  return variations.slice(0, count);
}
