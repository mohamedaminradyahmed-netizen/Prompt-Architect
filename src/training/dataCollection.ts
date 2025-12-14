import { PromptCategory, classifyPrompt } from '../types/promptTypes';
import { globalTracker, VariationLineage } from '../lineage/tracker';

/**
 * DIRECTIVE-033: Training data schema.
 * لماذا: نحتاج شكلاً موحداً يمكن تصديره للتدريب/التقييم لاحقاً، حتى لو كانت بعض المصادر لا تملك كل الحقول اليوم.
 */
export interface TrainingExample {
  id: string;
  originalPrompt: string;
  modifiedPrompt: string;
  context?: string;
  outputs: { original: string; modified: string };
  humanScore: number; // 1-5
  feedback?: string;
  metadata: {
    category: PromptCategory;
    mutationType: string;
    timestamp: Date;
    userId?: string;
  };
}

type AnyRecord = Record<string, any>;

function hasLocalStorage(): boolean {
  return typeof (globalThis as any).localStorage !== 'undefined';
}

function readStorageArray(key: string): AnyRecord[] {
  if (!hasLocalStorage()) return [];
  try {
    const raw = (globalThis as any).localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toDate(v: any): Date {
  const d = v instanceof Date ? v : new Date(v);
  return Number.isFinite(d.getTime()) ? d : new Date();
}

function makeId(prefix: string): string {
  const cryptoAny = (globalThis as any).crypto;
  const rid = cryptoAny?.randomUUID?.();
  return `${prefix}_${rid || `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`}`;
}

function clampHumanScore(score: any): number {
  const n = typeof score === 'number' ? score : Number(score);
  if (!Number.isFinite(n)) return 3;
  return Math.max(1, Math.min(5, Math.round(n)));
}

function categoryOf(text: string): PromptCategory {
  return classifyPrompt(text).category;
}

function* collectFromReviewStorage(): Generator<TrainingExample> {
  // مفاتيح التخزين الفعلية من src/api/review.ts — نقرأ مباشرة لتجنب استدعاء دوال تعتمد على localStorage في بيئة Node.
  const approved = readStorageArray('approved_items');
  for (const item of approved) {
    const originalPrompt = String(item.originalPrompt || '');
    const modifiedPrompt = String(item.suggestedVariation || '');
    if (!originalPrompt || !modifiedPrompt) continue;

    const ts = toDate(item.approvedAt || item.createdAt || item.timestamp);
    const cat: PromptCategory = item.category || categoryOf(originalPrompt);

    yield {
      id: String(item.id || makeId('approved')),
      originalPrompt,
      modifiedPrompt,
      outputs: { original: '', modified: '' },
      humanScore: 5, // الموافقة البشرية إشارة قوية للجودة
      metadata: {
        category: cat,
        mutationType: String(item.mutation || item.mutationType || 'human_approve'),
        timestamp: ts,
        userId: item.approvedBy,
      },
    };
  }

  const actions = readStorageArray('review_actions');
  for (const action of actions) {
    if (action.action !== 'edit') continue;
    const originalPrompt = String(action.metadata?.originalText || '');
    const modifiedPrompt = String(action.metadata?.editedText || '');
    if (!originalPrompt || !modifiedPrompt) continue;

    yield {
      id: String(action.id || makeId('review_edit')),
      originalPrompt,
      modifiedPrompt,
      outputs: { original: '', modified: '' },
      humanScore: 4, // تعديل بشري عادة أفضل من اقتراح خام
      feedback: undefined,
      metadata: {
        category: categoryOf(originalPrompt),
        mutationType: 'human_edit',
        timestamp: toDate(action.timestamp),
        userId: action.userId,
      },
    };
  }
}

function* collectFromHumanFeedbackStorage(): Generator<TrainingExample> {
  // feedback الحالي لا يخزن نصوص البرومبت/الاقتراح، لذا نجمعه كـ reference IDs لتجميع لاحق (join) دون فقدان الإشارة البشرية.
  const feedback = readStorageArray('human_feedback');
  for (const f of feedback) {
    const promptId = String(f.promptId || '');
    const variationId = String(f.variationId || '');
    if (!promptId || !variationId) continue;

    yield {
      id: String(f.id || makeId('feedback')),
      originalPrompt: `PROMPT_ID:${promptId}`,
      modifiedPrompt: `VARIATION_ID:${variationId}`,
      outputs: { original: '', modified: '' },
      humanScore: clampHumanScore(f.score),
      feedback: f.feedbackText ? String(f.feedbackText) : undefined,
      metadata: {
        category: PromptCategory.GENERAL_QA,
        mutationType: String(f.metadata?.mutationType || 'human_feedback'),
        timestamp: toDate(f.timestamp),
        userId: f.userId,
      },
    };
  }
}

function* collectFromLineageTracker(): Generator<TrainingExample> {
  // NOTE: لا يوجد API عام لاستخراج كل السجلات من LineageTracker.
  // نستخدم الوصول الداخلي (runtime) بشكل مقصود لتصدير dataset دون تغيير tracker أو توسيع واجهاته.
  const lineages: Map<string, VariationLineage> | undefined = (globalTracker as any).lineages;
  if (!lineages || typeof (lineages as any).forEach !== 'function') return;

  for (const v of lineages.values()) {
    // نحتاج زوج (original -> current) + إشارة بشرية إن وجدت
    if (!v.originalPrompt || !v.currentPrompt) continue;

    const score = v.feedback?.rating ?? (v.feedback ? 3 : undefined);
    if (!score) continue; // نُصدر فقط lineages التي لديها human signal لتكون مفيدة للتدريب

    yield {
      id: v.id,
      originalPrompt: v.originalPrompt,
      modifiedPrompt: v.currentPrompt,
      outputs: { original: '', modified: '' },
      humanScore: clampHumanScore(score),
      feedback: v.feedback?.comment,
      metadata: {
        category: categoryOf(v.originalPrompt),
        mutationType: String(v.mutation),
        timestamp: toDate(v.timestamp),
        userId: v.feedback?.userId,
      },
    };
  }
}

/**
 * Collect training data from:
 * - Human feedback (IDs + score)
 * - Review approvals/edits (full prompts)
 * - Lineage tracking (when feedback is attached)
 */
export async function* collectTrainingData(): AsyncGenerator<TrainingExample> {
  const seen = new Set<string>();
  const sources = [collectFromReviewStorage(), collectFromHumanFeedbackStorage(), collectFromLineageTracker()];

  for (const src of sources) {
    try {
      for (const ex of src) {
        if (seen.has(ex.id)) continue;
        seen.add(ex.id);
        yield ex;
      }
    } catch {
      // Fail-safe: مصدر واحد لا يجب أن يعطل pipeline بالكامل.
      continue;
    }
  }
}
