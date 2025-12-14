/**
 * DIRECTIVE-039: نظام قياس تحسّن Score على Benchmark
 *
 * Why (السبب الهندسي):
 * - بدون Benchmark ثابت، لا يمكننا قياس «هل تحسّنت المنظومة؟» بشكل موضوعي عبر الزمن.
 * - Baseline يوفّر نقطة مرجعية قبل/بعد أي تعديل.
 * - Tracking يحفظ تاريخ النتائج لالتقاط الاتجاهات (trends).
 * - Statistical testing يمنع ادعاءات تحسّن غير ذات دلالة (p-value).
 *
 * ملاحظة مهمة (Node-safe):
 * - بيئات Node/Jest قد لا تملك `localStorage` (راجع `.jules/nexus.md`).
 * - لذلك يتم استخدام localStorage عند توفره، وإلا fallback إلى تخزين داخل الذاكرة.
 */

import { classifyPrompt } from '../types/promptTypes';
import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  constrainMutation,
  type PromptVariation
} from '../mutations';
import type { TestCase } from '../sandbox/testExecutor';

// ============================================================================
// Types (PUBLIC)
// ============================================================================

export interface BenchmarkCandidateScore {
  prompt: string;
  mutation: string;
  tokenCount: number;
  estimatedCost: number;
  similarity: number; // 0-1
  latencyMs: number;
  score: number; // 0-100
}

export interface BenchmarkCaseResult {
  testCase: TestCase;
  candidates: BenchmarkCandidateScore[];
  best: BenchmarkCandidateScore;
}

export interface BenchmarkResults {
  suiteId: string;
  suiteName: string;
  createdAt: string; // ISO
  runId: string;
  cases: BenchmarkCaseResult[];
  avgScore: number;
}

export interface Comparison {
  suiteId: string;
  baselineAvgScore: number;
  currentAvgScore: number;
  avgScoreImprovement: number; // %
  significanceLevel: number; // p-value
  perCase: Array<{
    testCase: TestCase;
    baselineScore: number;
    currentScore: number;
    delta: number;
    deltaPercent: number; // %
  }>;
}

export interface TimeSeriesPoint {
  timestamp: string; // ISO
  runId: string;
  avgScore: number;
  avgScoreImprovement: number;
  significanceLevel: number;
}

export type TimeSeries = TimeSeriesPoint[];

export interface BenchmarkReport {
  avgScoreImprovement: number; // %
  significanceLevel: number; // p-value
  bestImprovement: TestCase;
  worstImprovement: TestCase;
  trends: TimeSeries;
}

// ============================================================================
// Benchmark Suite (PUBLIC)
// ============================================================================

/**
 * Benchmark suite ثابتة (يمكن استبدالها/توسيعها لاحقاً).
 * الهدف هو تغطية طيف متنوع من الفئات لتقليل الـoverfitting على نوع واحد.
 */
export const DEFAULT_BENCHMARK_SUITE: TestCase[] = [
  {
    id: 'bm_code_gen_1',
    prompt: 'Implement a TypeScript function that validates an email address and returns a detailed error message.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'CODE_GENERATION' }
  },
  {
    id: 'bm_code_gen_2',
    prompt: 'Write a React component that renders a paginated list with loading and error states.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'CODE_GENERATION' }
  },
  {
    id: 'bm_code_review_1',
    prompt: 'Review this code for security issues and suggest improvements: uses JWT stored in localStorage.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'CODE_REVIEW' }
  },
  {
    id: 'bm_data_1',
    prompt: 'Analyze a CSV dataset and summarize the top 3 trends. Include a simple visualization plan.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'DATA_ANALYSIS' }
  },
  {
    id: 'bm_general_1',
    prompt: 'Explain the difference between REST and GraphQL and when to use each.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'GENERAL_QA' }
  },
  {
    id: 'bm_content_1',
    prompt: 'Write a concise blog post outline about prompt engineering for beginners.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'CONTENT_WRITING' }
  },
  {
    id: 'bm_marketing_1',
    prompt: 'Write marketing copy for a new productivity app. Include a strong CTA and 3 key benefits.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'MARKETING_COPY' }
  },
  {
    id: 'bm_creative_1',
    prompt: 'Write a short creative story about a robot learning empathy, in 400 words.',
    expectedOutput: undefined,
    evaluationCriteria: {},
    metadata: { categoryHint: 'CREATIVE_WRITING' }
  }
];

// ============================================================================
// Minimal Logger (INTERNAL)
// ============================================================================

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class Logger {
  constructor(private scope: string) {}

  private log(level: LogLevel, message: string, meta?: Record<string, unknown>): void {
    // لماذا: توحيد شكل السجلات يجعلها قابلة للبحث والتحليل لاحقاً.
    const payload = meta ? { scope: this.scope, ...meta } : { scope: this.scope };

    // eslint-disable-next-line no-console
    if (level === 'error') console.error(`[${this.scope}] ${message}`, payload);
    // eslint-disable-next-line no-console
    else if (level === 'warn') console.warn(`[${this.scope}] ${message}`, payload);
    // eslint-disable-next-line no-console
    else if (level === 'info') console.info(`[${this.scope}] ${message}`, payload);
    // eslint-disable-next-line no-console
    else console.debug(`[${this.scope}] ${message}`, payload);
  }

  debug(message: string, meta?: Record<string, unknown>): void {
    this.log('debug', message, meta);
  }

  info(message: string, meta?: Record<string, unknown>): void {
    this.log('info', message, meta);
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    this.log('warn', message, meta);
  }

  error(message: string, meta?: Record<string, unknown>): void {
    this.log('error', message, meta);
  }
}

const logger = new Logger('benchmark');

// ============================================================================
// Storage Layer (INTERNAL)
// ============================================================================

const memoryStore = new Map<string, string>();

function getLocalStorageSafe(): Storage | null {
  try {
    const ls = (globalThis as unknown as { localStorage?: Storage }).localStorage;
    if (!ls) return null;

    // تحقق سريع من صلاحية الكتابة (قد يفشل في وضع الخصوصية أو سياسات المتصفح)
    const key = '__prompt_refiner_benchmark_test__';
    ls.setItem(key, '1');
    ls.removeItem(key);

    return ls;
  } catch {
    return null;
  }
}

function storeGet(key: string): string | null {
  const ls = getLocalStorageSafe();
  if (ls) {
    try {
      return ls.getItem(key);
    } catch {
      return null;
    }
  }
  return memoryStore.get(key) ?? null;
}

function storeSet(key: string, value: string): void {
  const ls = getLocalStorageSafe();
  if (ls) {
    try {
      ls.setItem(key, value);
      return;
    } catch {
      // fallthrough
    }
  }
  memoryStore.set(key, value);
}

// ============================================================================
// Helpers (INTERNAL)
// ============================================================================

function stableHash(input: string): string {
  // FNV-1a 32-bit
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  // Convert to unsigned and base36
  return (hash >>> 0).toString(36);
}

function calculateTokenCount(text: string): number {
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);
  return Math.ceil(words.length * 1.3);
}

function estimateCost(tokenCount: number): number {
  const costPer1kTokens = 0.03;
  return (tokenCount / 1000) * costPer1kTokens;
}

function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 0);
}

function calculateSimilarity(text1: string, text2: string): number {
  const words1 = normalizeWords(text1);
  const words2 = normalizeWords(text2);

  const freq1 = new Map<string, number>();
  const freq2 = new Map<string, number>();

  for (const w of words1) freq1.set(w, (freq1.get(w) ?? 0) + 1);
  for (const w of words2) freq2.set(w, (freq2.get(w) ?? 0) + 1);

  const allWords = new Set<string>([...freq1.keys(), ...freq2.keys()]);

  let dot = 0;
  let mag1 = 0;
  let mag2 = 0;

  for (const w of allWords) {
    const f1 = freq1.get(w) ?? 0;
    const f2 = freq2.get(w) ?? 0;
    dot += f1 * f2;
    mag1 += f1 * f1;
    mag2 += f2 * f2;
  }

  if (mag1 === 0 || mag2 === 0) return 0;
  return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
}

function estimateLatencyMs(prompt: string): number {
  const tokenCount = calculateTokenCount(prompt);
  const baseLatency = 500; // ms
  const processingTime = tokenCount * 5; // ms/token (محاكاة ثابتة بدون عشوائية)
  return Math.round(baseLatency + processingTime);
}

function calculateScore(similarity: number, tokenCount: number, maxTokens: number, latencyMs?: number): number {
  const normalizedCost = Math.min(tokenCount / maxTokens, 1);
  const costScore = 1 - normalizedCost;

  let score = similarity * 0.7 + costScore * 0.3;

  // Penalize high latency (محاكاة لنفس منطق evaluator)
  if (typeof latencyMs === 'number' && latencyMs > 2000) {
    score *= 0.9;
  }

  return Math.round(score * 100);
}

function buildDeterministicVariations(prompt: string): Array<{ mutation: string; text: string }>
{
  const classification = classifyPrompt(prompt);

  const variations: Array<{ mutation: string; text: string }> = [
    { mutation: 'original', text: prompt },
    { mutation: 'try-catch-style', text: tryCatchStyleMutation(prompt).text },
    { mutation: 'context-reduction', text: reduceContextMutation(prompt).text },
    { mutation: 'expansion', text: expandMutation(prompt).text },
    { mutation: 'constraint-addition', text: constrainMutation(prompt, classification.category).text }
  ];

  // Deduplicate by prompt text to keep scoring stable
  const seen = new Set<string>();
  return variations.filter(v => {
    const key = v.text.trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return key.length > 0;
  });
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function logFactorials(n: number): number[] {
  const lf = new Array<number>(n + 1);
  lf[0] = 0;
  for (let i = 1; i <= n; i++) lf[i] = lf[i - 1] + Math.log(i);
  return lf;
}

function binomialTailPValue(n: number, k: number): number {
  // One-sided: P(X >= k) where X ~ Bin(n, 0.5)
  if (n <= 0) return 1;
  if (k <= 0) return 1;
  if (k > n) return 0;

  const lf = logFactorials(n);
  const logHalf = Math.log(0.5);

  let sum = 0;
  for (let i = k; i <= n; i++) {
    const logChoose = lf[n] - lf[i] - lf[n - i];
    sum += Math.exp(logChoose + n * logHalf);
  }

  // Clamp due to floating point
  return Math.min(1, Math.max(0, sum));
}

function signTestPValue(deltas: number[]): number {
  const nonZero = deltas.filter(d => d !== 0);
  const n = nonZero.length;
  if (n === 0) return 1;

  const k = nonZero.filter(d => d > 0).length;
  return binomialTailPValue(n, k);
}

function suiteIdFromPrompts(prompts: string[]): string {
  return `suite_${stableHash(prompts.join('\n\n---\n\n'))}`;
}

function mkKeys(suiteId: string): { baselineKey: string; historyKey: string } {
  return {
    baselineKey: `benchmark:baseline:${suiteId}`,
    historyKey: `benchmark:history:${suiteId}`
  };
}

// ============================================================================
// Required API (PUBLIC)
// ============================================================================

/**
 * تشغيل Benchmark على مجموعة prompts (تمثل benchmark suite).
 *
 * ملاحظة: هذه الدالة متزامنة عمداً لتكون قابلة للاستخدام في المتصفح بدون IO خارجي.
 */
export function runBenchmark(prompts: string[]): BenchmarkResults {
  const cleaned = prompts.map(p => p.trim()).filter(p => p.length > 0);
  const suiteId = suiteIdFromPrompts(cleaned);
  const createdAt = new Date().toISOString();

  const suiteName = `benchmark_${cleaned.length}_cases`;
  const runId = `run_${Date.now().toString(36)}_${stableHash(createdAt)}`;

  const cases: BenchmarkCaseResult[] = cleaned.map((prompt, idx) => {
    const id = `tc_${stableHash(prompt)}`;

    const testCase: TestCase = {
      id,
      prompt,
      expectedOutput: undefined,
      evaluationCriteria: {},
      metadata: {
        index: idx,
        classification: classifyPrompt(prompt)
      }
    };

    const variations = buildDeterministicVariations(prompt);

    const tokenCounts = variations.map(v => calculateTokenCount(v.text));
    const maxTokens = Math.max(100, ...tokenCounts);

    const candidates: BenchmarkCandidateScore[] = variations.map(v => {
      const tokenCount = calculateTokenCount(v.text);
      const similarity = calculateSimilarity(prompt, v.text);
      const latencyMs = estimateLatencyMs(v.text);
      const score = calculateScore(similarity, tokenCount, maxTokens, latencyMs);

      return {
        prompt: v.text,
        mutation: v.mutation,
        tokenCount,
        estimatedCost: estimateCost(tokenCount),
        similarity,
        latencyMs,
        score
      };
    });

    // Sort best first (score desc, similarity desc, tokenCount asc)
    const sorted = [...candidates].sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (b.similarity !== a.similarity) return b.similarity - a.similarity;
      return a.tokenCount - b.tokenCount;
    });

    return {
      testCase,
      candidates: sorted,
      best: sorted[0]
    };
  });

  const avgScore = mean(cases.map(c => c.best.score));

  const results: BenchmarkResults = {
    suiteId,
    suiteName,
    createdAt,
    runId,
    cases,
    avgScore
  };

  // حفظ آخر suiteId للاستخدام في generateReport()
  storeSet('benchmark:lastSuiteId', suiteId);

  logger.info('Benchmark run completed', { suiteId, cases: cases.length, avgScore: Number(avgScore.toFixed(2)) });

  return results;
}

/**
 * مقارنة نتائج حالية مع Baseline.
 */
export function compareWithBaseline(current: BenchmarkResults, baseline: BenchmarkResults): Comparison {
  const baselineById = new Map<string, { testCase: TestCase; bestScore: number }>();
  for (const c of baseline.cases) baselineById.set(c.testCase.id, { testCase: c.testCase, bestScore: c.best.score });

  const currentById = new Map<string, { testCase: TestCase; bestScore: number }>();
  for (const c of current.cases) currentById.set(c.testCase.id, { testCase: c.testCase, bestScore: c.best.score });

  const ids = [...baselineById.keys()].filter(id => currentById.has(id));

  const perCase = ids.map(id => {
    const b = baselineById.get(id)!;
    const cur = currentById.get(id)!;

    const delta = cur.bestScore - b.bestScore;
    const deltaPercent = b.bestScore > 0 ? (delta / b.bestScore) * 100 : (cur.bestScore > 0 ? 100 : 0);

    return {
      testCase: b.testCase,
      baselineScore: b.bestScore,
      currentScore: cur.bestScore,
      delta,
      deltaPercent
    };
  });

  const baselineAvgScore = mean(perCase.map(p => p.baselineScore));
  const currentAvgScore = mean(perCase.map(p => p.currentScore));

  const avgScoreImprovement = baselineAvgScore > 0
    ? ((currentAvgScore - baselineAvgScore) / baselineAvgScore) * 100
    : (currentAvgScore > 0 ? 100 : 0);

  const deltas = perCase.map(p => p.delta);
  const significanceLevel = signTestPValue(deltas);

  return {
    suiteId: current.suiteId,
    baselineAvgScore,
    currentAvgScore,
    avgScoreImprovement,
    significanceLevel,
    perCase
  };
}

/**
 * تتبع النتائج عبر الزمن.
 * - إذا لم يوجد baseline بعد، تُسجَّل أول نتيجة كـ baseline تلقائياً.
 */
export function trackProgress(results: BenchmarkResults): void {
  const { baselineKey, historyKey } = mkKeys(results.suiteId);

  try {
    // Baseline
    const existingBaseline = storeGet(baselineKey);
    if (!existingBaseline) {
      storeSet(baselineKey, JSON.stringify(results));
      logger.info('Baseline created', { suiteId: results.suiteId, runId: results.runId });
    }

    // History
    const rawHistory = storeGet(historyKey);
    const history: BenchmarkResults[] = rawHistory ? (JSON.parse(rawHistory) as BenchmarkResults[]) : [];

    history.push(results);

    // Keep last 200 runs to avoid uncontrolled growth
    const trimmed = history.slice(Math.max(0, history.length - 200));
    storeSet(historyKey, JSON.stringify(trimmed));

    // Remember last suite
    storeSet('benchmark:lastSuiteId', results.suiteId);
  } catch (e) {
    logger.warn('Failed to track progress (storage unavailable)', { error: String(e) });
  }
}

/**
 * توليد تقرير يعتمد على:
 * - baseline المخزّن
 * - آخر نتيجة في history كسياق "current"
 */
export function generateReport(): BenchmarkReport {
  const defaultPrompts = DEFAULT_BENCHMARK_SUITE.map(t => t.prompt);
  const defaultSuiteId = suiteIdFromPrompts(defaultPrompts);

  const loadHistory = (targetSuiteId: string): BenchmarkResults[] => {
    const { historyKey } = mkKeys(targetSuiteId);
    try {
      const rawHistory = storeGet(historyKey);
      return rawHistory ? (JSON.parse(rawHistory) as BenchmarkResults[]) : [];
    } catch {
      return [];
    }
  };

  const loadBaseline = (targetSuiteId: string): BenchmarkResults | null => {
    const { baselineKey } = mkKeys(targetSuiteId);
    try {
      const rawBaseline = storeGet(baselineKey);
      return rawBaseline ? (JSON.parse(rawBaseline) as BenchmarkResults) : null;
    } catch {
      return null;
    }
  };

  // Prefer last used suite, fallback to default suite.
  let suiteId = storeGet('benchmark:lastSuiteId') || defaultSuiteId;

  // Load history; if empty for lastSuiteId, fallback to default suite history.
  let history = loadHistory(suiteId);
  if (history.length === 0 && suiteId !== defaultSuiteId) {
    suiteId = defaultSuiteId;
    history = loadHistory(suiteId);
  }

  // Ensure we have at least one run for the chosen suite.
  if (history.length === 0) {
    const initial = runBenchmark(defaultPrompts);
    trackProgress(initial);
    suiteId = initial.suiteId;
    history = loadHistory(suiteId);
    if (history.length === 0) history = [initial];
  }

  const current = history[history.length - 1];

  let baseline = loadBaseline(suiteId);
  if (!baseline) {
    baseline = history[0];
    const { baselineKey } = mkKeys(suiteId);
    storeSet(baselineKey, JSON.stringify(baseline));
  }

  const comparison = compareWithBaseline(current, baseline);

  const sortedByDelta = [...comparison.perCase].sort((a, b) => b.delta - a.delta);
  const best = sortedByDelta[0]?.testCase ?? DEFAULT_BENCHMARK_SUITE[0];
  const worst = sortedByDelta[sortedByDelta.length - 1]?.testCase ?? DEFAULT_BENCHMARK_SUITE[0];

  const trends: TimeSeries = history.map(run => {
    const cmp = compareWithBaseline(run, baseline!);
    return {
      timestamp: run.createdAt,
      runId: run.runId,
      avgScore: run.avgScore,
      avgScoreImprovement: cmp.avgScoreImprovement,
      significanceLevel: cmp.significanceLevel
    };
  });

  return {
    avgScoreImprovement: comparison.avgScoreImprovement,
    significanceLevel: comparison.significanceLevel,
    bestImprovement: best,
    worstImprovement: worst,
    trends
  };
}
