// Categorize prompts so the refiner can apply category-specific metrics and constraints without external dependencies.

export enum PromptCategory {
  CODE_GENERATION = 'CODE_GENERATION',
  CODE_REVIEW = 'CODE_REVIEW',
  CONTENT_WRITING = 'CONTENT_WRITING',
  MARKETING_COPY = 'MARKETING_COPY',
  DATA_ANALYSIS = 'DATA_ANALYSIS',
  GENERAL_QA = 'GENERAL_QA',
  CREATIVE_WRITING = 'CREATIVE_WRITING',
}

export interface PromptClassification { category: PromptCategory; confidence: number; characteristics: string[] }

export const CATEGORY_METRICS: Record<PromptCategory, string[]> = {
  [PromptCategory.CODE_GENERATION]: ['correctness', 'types', 'tests'],
  [PromptCategory.CODE_REVIEW]: ['security', 'performance', 'readability'],
  [PromptCategory.CONTENT_WRITING]: ['clarity', 'structure', 'tone'],
  [PromptCategory.MARKETING_COPY]: ['cta', 'benefits', 'conversion'],
  [PromptCategory.DATA_ANALYSIS]: ['methodology', 'insights', 'visualization'],
  [PromptCategory.GENERAL_QA]: ['helpfulness', 'factuality', 'brevity'],
  [PromptCategory.CREATIVE_WRITING]: ['originality', 'voice', 'imagery'],
};

const RULES: Array<{ c: PromptCategory; p: RegExp[]; tags: string[] }> = [
  { c: PromptCategory.CODE_GENERATION, p: [/(?:\b(function|class|api|component|script|sql|regex)\b|دالة|كود|برمجة)/i, /(?:\b(typescript|javascript|python|java|c#|go|rust)\b|تايبسكريبت|جافاسكريبت|بايثون)/i, /(?:\b(implement|build|create|write|generate)\b|اكتب|انشئ|نفذ)/i], tags: ['code', 'implementation'] },
  { c: PromptCategory.CODE_REVIEW, p: [/(?:\b(review|refactor|optimi[sz]e|bug|fix|lint)\b|راجع|حسن|اصلح|خطأ)/i, /\b(security|performance|best practice|code smell)\b/i], tags: ['code_review'] },
  { c: PromptCategory.DATA_ANALYSIS, p: [/(?:\b(analy[sz]e|analysis|dataset|data|csv|pandas|statistics|regression)\b|تحليل|بيانات)/i, /\b(plot|chart|visuali[sz]e|insight|trend)\b/i], tags: ['data'] },
  { c: PromptCategory.MARKETING_COPY, p: [/(?:\b(marketing|ad copy|landing page|headline|cta|conversion|sales|email campaign)\b|تسويق|اعلان|مبيعات)/i], tags: ['marketing'] },
  { c: PromptCategory.CONTENT_WRITING, p: [/(?:\b(blog|article|essay|outline|summari[sz]e|rewrite|proofread|edit)\b|مقال|تلخيص|صياغة|كتابة)/i], tags: ['content'] },
  { c: PromptCategory.CREATIVE_WRITING, p: [/(?:\b(story|poem|fiction|character|dialogue|scene|plot|creative)\b|قصة|شعر|خيال)/i], tags: ['creative'] },
];

export function classifyPrompt(prompt: string): PromptClassification {
  const text = prompt.toLowerCase();
  let best: { c: PromptCategory; s: number; tags: string[] } = { c: PromptCategory.GENERAL_QA, s: 0, tags: ['general'] };
  for (const r of RULES) {
    const s = r.p.reduce((acc, re) => acc + (re.test(text) ? 1 : 0), 0);
    if (s > best.s) best = { c: r.c, s, tags: r.tags };
  }
  const confidence = best.s ? Math.min(0.35 + best.s * 0.2, 0.95) : 0.25;
  return { category: best.c, confidence, characteristics: best.tags };
}
