/**
 * Human Feedback API
 * Handles collection and storage of human feedback for prompt variations
 */

import {
  getAverageFeedback as getAverageFeedbackFromStore,
  getFeedbackFromStorageSync,
  getFeedbackStats as getFeedbackStatsFromStore,
  storeFeedback as storeFeedbackInStore,
} from '../db/feedbackStore';

export interface HumanFeedback {
  id?: string;
  promptId: string;
  variationId: string;
  score: number; // 1-5
  feedbackText?: string;
  userId: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

export interface FeedbackStats {
  averageScore: number;
  totalFeedback: number;
  scoreDistribution: Record<number, number>;
}

/**
 * Store human feedback
 */
export async function storeFeedback(feedback: HumanFeedback): Promise<void> {
  // طبقة API تفصل UI عن طبقة التخزين لتسهيل ترحيلها لاحقاً لقاعدة بيانات حقيقية.
  await storeFeedbackInStore(feedback);
}

/**
 * Get average feedback score for a variation
 */
export async function getAverageFeedback(variationId: string): Promise<number> {
  return await getAverageFeedbackFromStore(variationId);
}

/**
 * Get feedback statistics
 */
export async function getFeedbackStats(variationId?: string): Promise<FeedbackStats> {
  return await getFeedbackStatsFromStore(variationId);
}

/**
 * توافق خلفي (يستخدمه كود التدريب): قراءة محلية لكل feedback المخزن.
 * ملاحظة: في الإنتاج يجب استبدال هذا بقراءة من قاعدة بيانات مع صلاحيات/حدود واضحة.
 */
export function getFeedbackFromStorage(): HumanFeedback[] {
  return getFeedbackFromStorageSync();
}

// ملاحظة: وظائف التخزين انتقلت إلى `src/db/feedbackStore.ts` لتكون قابلة للاستبدال لاحقاً.