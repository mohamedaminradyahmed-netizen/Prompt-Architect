/**
 * Feedback Store (DIRECTIVE-015)
 *
 * لماذا هذا التصميم؟
 * نحتاج طبقة تخزين واحدة قابلة للاستبدال لاحقاً بقاعدة بيانات حقيقية (Postgres).
 * حالياً نستخدم localStorage كحل MVP بدون إدخال تبعيات/بنية تحتية إضافية.
 */

import type { HumanFeedback, FeedbackStats } from '../api/feedback';

const STORAGE_KEY = 'human_feedback';

type StoredFeedback = Required<Pick<HumanFeedback, 'id' | 'timestamp'>> & HumanFeedback;

function now(): Date {
  return new Date();
}

function generateId(): string {
  // لا نحتاج UUID هنا؛ هدفنا منع التصادم داخل جلسة المتصفح.
  return Math.random().toString(36).slice(2, 11);
}

function canUseLocalStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

function readAll(): StoredFeedback[] {
  if (!canUseLocalStorage()) return [];
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw) as any[];
    return Array.isArray(parsed)
      ? parsed.map((f) => ({ ...f, timestamp: new Date(f.timestamp) }))
      : [];
  } catch {
    // إذا تلفت البيانات نعيد قائمة فارغة بدلاً من تعطيل UI.
    return [];
  }
}

function writeAll(items: StoredFeedback[]): void {
  if (!canUseLocalStorage()) return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

/**
 * يخزن تقييم بشري لاقتراح واحد.
 */
export async function storeFeedback(feedback: HumanFeedback): Promise<void> {
  const stored: StoredFeedback = {
    ...feedback,
    id: feedback.id ?? generateId(),
    timestamp: feedback.timestamp ?? now(),
  };

  const existing = readAll();
  existing.push(stored);
  writeAll(existing);
}

/**
 * يعيد متوسط التقييم لاقتراح محدد.
 */
export async function getAverageFeedback(variationId: string): Promise<number> {
  const feedback = readAll().filter((f) => f.variationId === variationId);
  if (feedback.length === 0) return 0;
  const sum = feedback.reduce((acc, f) => acc + f.score, 0);
  return sum / feedback.length;
}

/**
 * يعيد إحصائيات التقييم (لكل النظام أو لاقتراح محدد).
 */
export async function getFeedbackStats(variationId?: string): Promise<FeedbackStats> {
  const all = readAll();
  const target = variationId ? all.filter((f) => f.variationId === variationId) : all;

  if (target.length === 0) {
    return { averageScore: 0, totalFeedback: 0, scoreDistribution: {} };
  }

  const sum = target.reduce((acc, f) => acc + f.score, 0);
  const distribution: Record<number, number> = {};
  for (const f of target) {
    distribution[f.score] = (distribution[f.score] || 0) + 1;
  }

  return {
    averageScore: sum / target.length,
    totalFeedback: target.length,
    scoreDistribution: distribution,
  };
}

/**
 * مخصص للاستخدام في Dashboard: يعيد جميع السجلات.
 */
export async function listFeedback(): Promise<StoredFeedback[]> {
  return readAll();
}

/**
 * توافق خلفي: بعض أجزاء النظام كانت تستورد هذه الدالة من `src/api/feedback.ts`.
 * نُبقي واجهة متزامنة للوصول لبيانات localStorage (للاستخدامات التحليلية فقط).
 */
export function getFeedbackFromStorageSync(): HumanFeedback[] {
  return readAll();
}
