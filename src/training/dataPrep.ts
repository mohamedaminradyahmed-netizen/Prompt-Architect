import { PromptCategory, classifyPrompt } from '../types/promptTypes';
import type { TrainingExample } from './dataCollection';

function normalizeText(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

function stableKey(e: TrainingExample): string {
  return `${normalizeText(e.originalPrompt)}\n---\n${normalizeText(e.modifiedPrompt)}\n#${e.humanScore}`;
}

/**
 * DIRECTIVE-033: تنظيف البيانات.
 * لماذا: التدريب يتدهور بسرعة مع duplicates/ضوضاء/فراغات غير منضبطة، لذا ننفذ تنظيفاً محافظاً بدون dependencies.
 */
export function cleanTrainingData(data: TrainingExample[]): TrainingExample[] {
  const out: TrainingExample[] = [];
  const seen = new Set<string>();

  for (const raw of data) {
    const originalPrompt = normalizeText(raw.originalPrompt || '');
    const modifiedPrompt = normalizeText(raw.modifiedPrompt || '');
    if (!originalPrompt || !modifiedPrompt) continue;

    // إزالة entries التي هي مجرد IDs (حالياً) ما لم يصرّح بها المستخدم لاحقاً
    const isIdOnly = originalPrompt.startsWith('PROMPT_ID:') && modifiedPrompt.startsWith('VARIATION_ID:');
    if (isIdOnly) continue;

    const humanScore = Math.max(1, Math.min(5, Math.round(raw.humanScore || 3)));
    if (humanScore <= 1) continue; // إزالة منخفضة الجودة جداً بشكل افتراضي

    const category = raw.metadata?.category || classifyPrompt(originalPrompt).category;
    const cleaned: TrainingExample = {
      ...raw,
      originalPrompt,
      modifiedPrompt,
      humanScore,
      metadata: {
        category,
        mutationType: raw.metadata?.mutationType || 'unknown',
        timestamp: raw.metadata?.timestamp ? new Date(raw.metadata.timestamp) : new Date(),
        userId: raw.metadata?.userId,
      },
    };

    const key = stableKey(cleaned);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(cleaned);
  }

  return out;
}

function shuffle<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export interface SplitDatasetResult {
  train: TrainingExample[];
  val: TrainingExample[];
  test: TrainingExample[];
}

/**
 * تقسيم train/val/test مع stratified sampling حسب الفئة.
 * trainRatio يطبق على train، والباقي يُقسم بالتساوي إلى val/test.
 */
export function splitDataset(data: TrainingExample[], trainRatio: number = 0.8): SplitDatasetResult {
  const tr = Math.max(0.1, Math.min(0.95, trainRatio));
  const groups = new Map<PromptCategory, TrainingExample[]>();

  for (const ex of data) {
    const cat = ex.metadata?.category || classifyPrompt(ex.originalPrompt).category;
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat)!.push(ex);
  }

  const train: TrainingExample[] = [];
  const val: TrainingExample[] = [];
  const test: TrainingExample[] = [];

  for (const [, items] of groups) {
    const s = shuffle(items);
    const trainCount = Math.floor(s.length * tr);
    const remain = s.length - trainCount;
    const valCount = Math.floor(remain / 2);

    train.push(...s.slice(0, trainCount));
    val.push(...s.slice(trainCount, trainCount + valCount));
    test.push(...s.slice(trainCount + valCount));
  }

  return { train, val, test };
}

function csvEscape(v: string): string {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export type ExportFormat = 'json' | 'csv' | 'parquet';
export interface ExportResult {
  format: ExportFormat;
  data: string;
  note?: string;
}

/**
 * تصدير بيانات التدريب.
 * ملاحظة: "parquet" يتطلب dependency مخصصة؛ هنا نعيد JSON كحل وسيط شفاف بدلاً من إضافة تبعيات دون موافقة.
 */
export function exportForTraining(data: TrainingExample[], format: ExportFormat): ExportResult {
  if (format === 'json') {
    return { format, data: JSON.stringify(data, null, 2) };
  }

  if (format === 'csv') {
    const header = [
      'id',
      'originalPrompt',
      'modifiedPrompt',
      'context',
      'outputOriginal',
      'outputModified',
      'humanScore',
      'feedback',
      'category',
      'mutationType',
      'timestamp',
      'userId',
    ].join(',');

    const rows = data.map((e) => [
      e.id,
      e.originalPrompt,
      e.modifiedPrompt,
      e.context || '',
      e.outputs?.original || '',
      e.outputs?.modified || '',
      String(e.humanScore),
      e.feedback || '',
      e.metadata?.category || '',
      e.metadata?.mutationType || '',
      e.metadata?.timestamp ? new Date(e.metadata.timestamp).toISOString() : '',
      e.metadata?.userId || '',
    ].map(csvEscape).join(','));

    return { format, data: [header, ...rows].join('\n') };
  }

  // parquet (placeholder)
  return {
    format,
    data: JSON.stringify(data),
    note: 'Parquet export غير مُنفذ بدون dependency (مثل Apache Arrow). تم إرجاع JSON كحل وسيط متوافق.'
  };
}
