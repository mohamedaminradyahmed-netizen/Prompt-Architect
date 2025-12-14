import { PromptCategory } from '../types/promptTypes';
import { MutationType } from '../mutations';

export interface SuggestionMetadata { mutation: MutationType; category: PromptCategory; userId?: string; createdAt?: number; }
export type AdoptionMetrics = { rate: number; total: number; accepted: number; breakdown?: Record<string, number> };
export interface AdoptionReport { overallRate: number; byMutation: Map<MutationType, number>; byCategory: Map<PromptCategory, number>; avgTimeToAdopt: number; trends: Array<{ timestamp: number; adoptionRate: number }>;}
type AdoptionRecord = { metadata: SuggestionMetadata; shownAt: number; acceptedAt?: number; timeToAccept?: number };

export class AdoptionTracker {
  private records = new Map<string, AdoptionRecord>();
  trackSuggestionShown(id: string, metadata: SuggestionMetadata): void { this.records.set(id, { metadata, shownAt: metadata.createdAt ?? Date.now() }); }
  trackSuggestionAccepted(id: string, timeToAccept: number): void {
    const rec = this.records.get(id); if (!rec) return;
    const tta = Math.max(0, timeToAccept); this.records.set(id, { ...rec, acceptedAt: rec.shownAt + tta, timeToAccept: tta });
  }
  getAdoptionRate(dim: 'overall' | 'mutation' | 'category'): AdoptionMetrics {
    const values = Array.from(this.records.values()), accepted = values.filter(v => v.acceptedAt !== undefined);
    if (dim === 'overall') return { rate: values.length ? (accepted.length / values.length) * 100 : 0, total: values.length, accepted: accepted.length };
    const grouping = new Map<string, { total: number; accepted: number }>();
    for (const rec of values) { const key = dim === 'mutation' ? rec.metadata.mutation : rec.metadata.category; const g = grouping.get(key) ?? { total: 0, accepted: 0 }; grouping.set(key, { total: g.total + 1, accepted: g.accepted + (rec.acceptedAt ? 1 : 0) }); }
    const breakdown: Record<string, number> = {}; grouping.forEach((v, k) => { breakdown[k] = v.total ? (v.accepted / v.total) * 100 : 0; });
    const total = values.length, acceptedCount = accepted.length, rate = total ? (acceptedCount / total) * 100 : 0; return { rate, total, accepted: acceptedCount, breakdown };
  }
  getTimeToAdoption(): number { const times = Array.from(this.records.values()).map(r => r.timeToAccept).filter((t): t is number => typeof t === 'number'); return times.length ? times.reduce((s, t) => s + t, 0) / times.length : 0; }
  generateAdoptionReport(): AdoptionReport {
    const byMutation = new Map<MutationType, number>(), byCategory = new Map<PromptCategory, number>();
    Object.entries(this.getAdoptionRate('mutation').breakdown ?? {}).forEach(([k, v]) => byMutation.set(k as MutationType, v));
    Object.entries(this.getAdoptionRate('category').breakdown ?? {}).forEach(([k, v]) => byCategory.set(k as PromptCategory, v));
    return { overallRate: this.getAdoptionRate('overall').rate, byMutation, byCategory, avgTimeToAdopt: this.getTimeToAdoption(), trends: [] };
  }
}
