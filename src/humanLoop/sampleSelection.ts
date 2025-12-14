import { ScoredSuggestion } from '../evaluator';

// Select a small, representative subset of suggestions (typically 5–10%) for human review.
export enum SamplingStrategy { UNCERTAINTY, DIVERSITY, ERROR_FOCUSED, RANDOM, MIXED }

const clamp = (n: number, max: number) => Math.max(0, Math.min(Math.floor(n), max));
const key = (s: ScoredSuggestion) => `${s.mutation}|${s.prompt}`;
const uniq = (arr: ScoredSuggestion[]) => { const seen = new Set<string>(); return arr.filter(s => (!seen.has(key(s)) && (seen.add(key(s)), true))); };
const shuffle = <T>(arr: T[]) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const uncertainty = (s: ScoredSuggestion) => 1 - Math.min(Math.abs(s.score - 50), 50) / 50;
const errorish = (s: any) => !!(s && ((typeof s.failedTests === 'number' && s.failedTests > 0) || (typeof s.passRate === 'number' && s.passRate < 1) || (Array.isArray(s.errors) && s.errors.length) || s.hasError));

function pickDiverse(variations: ScoredSuggestion[], n: number): ScoredSuggestion[] {
  const groups: Record<string, ScoredSuggestion[]> = {};
  variations.forEach(s => { const k = s.mutation || 'unknown'; (groups[k] = groups[k] || []).push(s); });
  const picks = Object.keys(groups).map(k => groups[k].sort((a, b) => b.score - a.score)[0]);
  return uniq(picks.concat(variations.sort((a, b) => b.score - a.score))).slice(0, n);
}

export function selectSamplesForReview(variations: ScoredSuggestion[], strategy: SamplingStrategy, count: number): ScoredSuggestion[] {
  const base = uniq(variations.slice());
  const n = clamp(count, base.length);
  if (!n) return [];

  const top = (arr: ScoredSuggestion[]) => uniq(arr).slice(0, n);
  const err = top(base.filter(errorish).sort((a, b) => b.score - a.score));
  const unc = top(base.slice().sort((a, b) => uncertainty(b) - uncertainty(a)));
  const div = pickDiverse(base.slice(), n);
  const rnd = shuffle(base).slice(0, n);

  if (strategy === SamplingStrategy.ERROR_FOCUSED) return err.length ? err : unc;
  if (strategy === SamplingStrategy.UNCERTAINTY) return unc;
  if (strategy === SamplingStrategy.DIVERSITY) return div;
  if (strategy === SamplingStrategy.RANDOM) return rnd;
  return uniq(err.concat(div, unc, rnd)).slice(0, n);
}
