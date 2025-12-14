/**
 * Reward Dataset Builder (DIRECTIVE-054)
 *
 * Pipeline لجمع Human Feedback وتحويله إلى بيانات تدريب للـ Reward Model.
 *
 * الخطوات:
 * 1. Collection: جمع feedback من UI
 * 2. Validation: تحقق من جودة البيانات
 * 3. Augmentation: إضافة features (embeddings, metadata)
 * 4. Storage: تخزين في database
 * 5. Export: تصدير للتدريب
 */

import { PromptCategory, classifyPrompt } from '../types/promptTypes';
import type { HumanFeedback } from '../api/feedback';
import { storeFeedback as storeFeedbackInStore, listFeedback } from '../db/feedbackStore';
import { calculateTokenCount, calculateSimilarity } from '../evaluator';
import type { TrainingExample } from './dataCollection';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Feedback المُحسّن مع حقول إضافية للتدريب
 */
export interface EnhancedFeedback extends HumanFeedback {
  originalPrompt?: string;
  variationText?: string;
  category?: PromptCategory;
  mutationType?: string;
  tokenCount?: number;
  similarity?: number;
}

/**
 * مثال واحد لتدريب Reward Model
 * يحتوي على embeddings و features و label
 */
export interface RewardExample {
  /** ID فريد للمثال */
  id: string;
  /** Embedding للبرومبت الأصلي (vector) */
  promptEmbedding: number[];
  /** Embedding للـ variation (vector) */
  variationEmbedding: number[];
  /** Features رقمية إضافية */
  features: number[];
  /** Feature names للتوثيق */
  featureNames: string[];
  /** Label: human score مُطبّع (0-1) */
  label: number;
  /** الوزن/الأهمية (confidence) */
  weight: number;
  /** Metadata إضافية */
  metadata: {
    originalPrompt: string;
    variationText: string;
    category: PromptCategory;
    mutationType: string;
    rawScore: number;
    timestamp: Date;
    userId?: string;
  };
}

/**
 * إحصائيات Dataset
 */
export interface DatasetStats {
  totalExamples: number;
  avgScore: number;
  scoreDistribution: Record<number, number>;
  categoryDistribution: Record<string, number>;
  mutationDistribution: Record<string, number>;
  avgTokenCount: number;
  avgSimilarity: number;
  dateRange: {
    earliest: Date | null;
    latest: Date | null;
  };
}

/**
 * فلاتر لتصفية Dataset
 */
export interface DatasetFilters {
  /** تصفية حسب الفئة */
  categories?: PromptCategory[];
  /** تصفية حسب نوع الـ mutation */
  mutationTypes?: string[];
  /** الحد الأدنى للـ score */
  minScore?: number;
  /** الحد الأقصى للـ score */
  maxScore?: number;
  /** تاريخ البداية */
  startDate?: Date;
  /** تاريخ النهاية */
  endDate?: Date;
  /** تصفية حسب user */
  userIds?: string[];
  /** الحد الأدنى للـ similarity */
  minSimilarity?: number;
  /** استبعاد duplicates */
  excludeDuplicates?: boolean;
  /** الحد الأدنى للـ weight/confidence */
  minWeight?: number;
}

/**
 * تنسيقات التصدير المدعومة
 */
export type ExportFormat = 'json' | 'jsonl' | 'csv' | 'parquet' | 'tfrecord';

/**
 * Dataset كامل للتدريب
 */
export interface RewardDataset {
  /** الأمثلة */
  examples: RewardExample[];
  /** الإحصائيات */
  statistics: DatasetStats;
  /** Metadata */
  metadata: {
    created: Date;
    version: string;
    size: number;
    filters?: DatasetFilters;
    embeddingDimension: number;
    featureCount: number;
  };
}

/**
 * نتيجة التحقق من صحة Feedback
 */
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * نتيجة التصدير
 */
export interface ExportResult {
  format: ExportFormat;
  data: string;
  path?: string;
  size: number;
  note?: string;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

const DEFAULT_EMBEDDING_DIMENSION = 384;
const DATASET_VERSION = '1.0.0';

// ============================================================================
// STORAGE HELPERS
// ============================================================================

/**
 * In-memory storage للـ enhanced feedback (للتطوير)
 * في الإنتاج يُستبدل بقاعدة بيانات حقيقية
 */
const enhancedFeedbackStore = new Map<string, EnhancedFeedback>();

function generateId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 10);
  return `rf_${timestamp}_${random}`;
}

// ============================================================================
// COLLECT FEEDBACK
// ============================================================================

/**
 * جمع وتخزين feedback جديد
 *
 * @param variationId - معرّف الـ variation
 * @param feedback - بيانات الـ feedback
 * @param enhancementData - بيانات إضافية لتحسين الـ feedback
 */
export async function collectFeedback(
  variationId: string,
  feedback: Omit<HumanFeedback, 'variationId'>,
  enhancementData?: {
    originalPrompt?: string;
    variationText?: string;
    category?: PromptCategory;
    mutationType?: string;
  }
): Promise<void> {
  const fullFeedback: HumanFeedback = {
    ...feedback,
    variationId,
    id: feedback.id || generateId(),
    timestamp: feedback.timestamp || new Date(),
  };

  // Store in main feedback store
  await storeFeedbackInStore(fullFeedback);

  // Store enhanced version if we have enhancement data
  if (enhancementData) {
    const enhanced: EnhancedFeedback = {
      ...fullFeedback,
      originalPrompt: enhancementData.originalPrompt,
      variationText: enhancementData.variationText,
      category: enhancementData.category,
      mutationType: enhancementData.mutationType,
    };

    // Calculate additional metrics if we have the text
    if (enhancementData.variationText) {
      enhanced.tokenCount = calculateTokenCount(enhancementData.variationText);
    }
    if (enhancementData.originalPrompt && enhancementData.variationText) {
      enhanced.similarity = calculateSimilarity(
        enhancementData.originalPrompt,
        enhancementData.variationText
      );
    }

    enhancedFeedbackStore.set(fullFeedback.id!, enhanced);
  }
}

// ============================================================================
// VALIDATE FEEDBACK
// ============================================================================

/**
 * التحقق من صحة feedback
 *
 * @param feedback - الـ feedback للتحقق منه
 * @returns نتيجة التحقق
 */
export function validateFeedback(feedback: EnhancedFeedback | HumanFeedback): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Required fields
  if (!feedback.promptId) {
    errors.push('promptId is required');
  }
  if (!feedback.variationId) {
    errors.push('variationId is required');
  }
  if (!feedback.userId) {
    errors.push('userId is required');
  }

  // Score validation
  if (typeof feedback.score !== 'number') {
    errors.push('score must be a number');
  } else if (feedback.score < 1 || feedback.score > 5) {
    errors.push('score must be between 1 and 5');
  } else if (!Number.isInteger(feedback.score)) {
    warnings.push('score should be an integer (will be rounded)');
  }

  // Check for enhanced fields
  const enhanced = feedback as EnhancedFeedback;
  if (!enhanced.originalPrompt) {
    warnings.push('originalPrompt is missing - embedding will use placeholder');
  }
  if (!enhanced.variationText) {
    warnings.push('variationText is missing - embedding will use placeholder');
  }

  // Check for suspicious patterns
  if (enhanced.originalPrompt && enhanced.variationText) {
    if (enhanced.originalPrompt === enhanced.variationText && feedback.score === 5) {
      warnings.push('Identical prompt and variation with max score - might be self-rating');
    }
  }

  // Timestamp validation
  if (feedback.timestamp) {
    const ts = new Date(feedback.timestamp);
    const now = new Date();
    if (ts > now) {
      warnings.push('timestamp is in the future');
    }
    const oneYearAgo = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
    if (ts < oneYearAgo) {
      warnings.push('timestamp is more than 1 year old');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * التحقق من صحة مجموعة feedback
 */
export function validateFeedbackBatch(
  feedbackList: Array<EnhancedFeedback | HumanFeedback>
): { valid: number; invalid: number; results: ValidationResult[] } {
  const results = feedbackList.map(validateFeedback);
  return {
    valid: results.filter((r) => r.isValid).length,
    invalid: results.filter((r) => !r.isValid).length,
    results,
  };
}

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

/**
 * توليد embedding بسيط (mock) لنص
 * في الإنتاج يُستبدل بـ OpenAI/local embeddings
 */
function generateMockEmbedding(text: string, dimension: number = DEFAULT_EMBEDDING_DIMENSION): number[] {
  // Hash-based deterministic embedding
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }

  const embedding: number[] = [];
  let seed = Math.abs(hash);

  for (let i = 0; i < dimension; i++) {
    seed = (seed * 9301 + 49297) % 233280;
    embedding.push((seed / 233280) * 2 - 1);
  }

  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map((val) => (magnitude > 0 ? val / magnitude : 0));
}

/**
 * توليد features رقمية من feedback
 */
function extractFeatures(
  feedback: EnhancedFeedback,
  originalPrompt: string,
  variationText: string
): { features: number[]; featureNames: string[] } {
  const featureNames = [
    'tokenCount',
    'tokenRatio',
    'similarity',
    'lengthDiff',
    'wordCountRatio',
    'avgWordLength',
    'punctuationRatio',
    'uppercaseRatio',
    'categoryCode',
    'hasExamples',
    'hasConstraints',
    'questionCount',
  ];

  const originalTokens = calculateTokenCount(originalPrompt);
  const variationTokens = calculateTokenCount(variationText);
  const similarity = feedback.similarity ?? calculateSimilarity(originalPrompt, variationText);

  const originalWords = originalPrompt.split(/\s+/).filter((w) => w.length > 0);
  const variationWords = variationText.split(/\s+/).filter((w) => w.length > 0);

  const punctuationCount = (variationText.match(/[.,!?;:]/g) || []).length;
  const uppercaseCount = (variationText.match(/[A-Z]/g) || []).length;

  const categoryCode = feedback.category
    ? Object.values(PromptCategory).indexOf(feedback.category)
    : -1;

  const hasExamples = /example|e\.g\.|for instance|مثال/i.test(variationText) ? 1 : 0;
  const hasConstraints = /must|should|constraint|require|يجب|قيد/i.test(variationText) ? 1 : 0;
  const questionCount = (variationText.match(/\?/g) || []).length;

  const features = [
    variationTokens,
    originalTokens > 0 ? variationTokens / originalTokens : 0,
    similarity,
    variationText.length - originalPrompt.length,
    originalWords.length > 0 ? variationWords.length / originalWords.length : 0,
    variationWords.length > 0
      ? variationWords.reduce((sum, w) => sum + w.length, 0) / variationWords.length
      : 0,
    variationText.length > 0 ? punctuationCount / variationText.length : 0,
    variationText.length > 0 ? uppercaseCount / variationText.length : 0,
    categoryCode,
    hasExamples,
    hasConstraints,
    questionCount,
  ];

  return { features, featureNames };
}

// ============================================================================
// BUILD REWARD DATASET
// ============================================================================

/**
 * بناء Reward Dataset من feedback المُجمّع
 *
 * @param filters - فلاتر اختيارية
 * @returns Dataset جاهز للتدريب
 */
export async function buildRewardDataset(filters?: DatasetFilters): Promise<RewardDataset> {
  // 1. Collect all feedback
  const allFeedback = await listFeedback();
  const enhancedList: EnhancedFeedback[] = [];

  for (const fb of allFeedback) {
    const enhanced = enhancedFeedbackStore.get(fb.id!) || (fb as EnhancedFeedback);
    enhancedList.push(enhanced);
  }

  // 2. Apply filters
  let filtered = enhancedList;

  if (filters) {
    if (filters.categories?.length) {
      filtered = filtered.filter(
        (f) => f.category && filters.categories!.includes(f.category)
      );
    }
    if (filters.mutationTypes?.length) {
      filtered = filtered.filter(
        (f) => f.mutationType && filters.mutationTypes!.includes(f.mutationType)
      );
    }
    if (filters.minScore !== undefined) {
      filtered = filtered.filter((f) => f.score >= filters.minScore!);
    }
    if (filters.maxScore !== undefined) {
      filtered = filtered.filter((f) => f.score <= filters.maxScore!);
    }
    if (filters.startDate) {
      filtered = filtered.filter(
        (f) => f.timestamp && new Date(f.timestamp) >= filters.startDate!
      );
    }
    if (filters.endDate) {
      filtered = filtered.filter(
        (f) => f.timestamp && new Date(f.timestamp) <= filters.endDate!
      );
    }
    if (filters.userIds?.length) {
      filtered = filtered.filter((f) => filters.userIds!.includes(f.userId));
    }
    if (filters.minSimilarity !== undefined) {
      filtered = filtered.filter(
        (f) => f.similarity !== undefined && f.similarity >= filters.minSimilarity!
      );
    }
    if (filters.excludeDuplicates) {
      const seen = new Set<string>();
      filtered = filtered.filter((f) => {
        const key = `${f.originalPrompt || f.promptId}|${f.variationText || f.variationId}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }
  }

  // 3. Validate and convert to RewardExamples
  const examples: RewardExample[] = [];
  const validationResults = validateFeedbackBatch(filtered);

  for (let i = 0; i < filtered.length; i++) {
    const fb = filtered[i];
    const validation = validationResults.results[i];

    // Skip invalid entries
    if (!validation.isValid) continue;

    // Get or generate text content
    const originalPrompt = fb.originalPrompt || `prompt:${fb.promptId}`;
    const variationText = fb.variationText || `variation:${fb.variationId}`;

    // Auto-classify if not provided
    const category = fb.category || classifyPrompt(originalPrompt).category;

    // Generate embeddings
    const promptEmbedding = generateMockEmbedding(originalPrompt);
    const variationEmbedding = generateMockEmbedding(variationText);

    // Extract features
    const { features, featureNames } = extractFeatures(
      { ...fb, category },
      originalPrompt,
      variationText
    );

    // Calculate weight based on confidence
    const weight = calculateExampleWeight(fb, validation);

    // Normalize score to 0-1
    const label = (fb.score - 1) / 4; // 1-5 → 0-1

    examples.push({
      id: fb.id || generateId(),
      promptEmbedding,
      variationEmbedding,
      features,
      featureNames,
      label,
      weight,
      metadata: {
        originalPrompt,
        variationText,
        category,
        mutationType: fb.mutationType || 'unknown',
        rawScore: fb.score,
        timestamp: fb.timestamp ? new Date(fb.timestamp) : new Date(),
        userId: fb.userId,
      },
    });
  }

  // 4. Calculate statistics
  const statistics = calculateDatasetStats(examples);

  // 5. Build final dataset
  return {
    examples,
    statistics,
    metadata: {
      created: new Date(),
      version: DATASET_VERSION,
      size: examples.length,
      filters,
      embeddingDimension: DEFAULT_EMBEDDING_DIMENSION,
      featureCount: examples[0]?.features.length || 0,
    },
  };
}

/**
 * حساب وزن/أهمية مثال بناءً على جودة البيانات
 */
function calculateExampleWeight(
  feedback: EnhancedFeedback,
  validation: ValidationResult
): number {
  let weight = 1.0;

  // Reduce weight for warnings
  weight -= validation.warnings.length * 0.1;

  // Increase weight if we have full text
  if (feedback.originalPrompt && feedback.variationText) {
    weight += 0.2;
  }

  // Increase weight if we have category
  if (feedback.category) {
    weight += 0.1;
  }

  // Reduce weight for extreme scores (might be less reliable)
  if (feedback.score === 1 || feedback.score === 5) {
    weight -= 0.05;
  }

  // Ensure weight is in valid range
  return Math.max(0.1, Math.min(1.0, weight));
}

/**
 * حساب إحصائيات Dataset
 */
function calculateDatasetStats(examples: RewardExample[]): DatasetStats {
  if (examples.length === 0) {
    return {
      totalExamples: 0,
      avgScore: 0,
      scoreDistribution: {},
      categoryDistribution: {},
      mutationDistribution: {},
      avgTokenCount: 0,
      avgSimilarity: 0,
      dateRange: { earliest: null, latest: null },
    };
  }

  // Score distribution
  const scoreDistribution: Record<number, number> = {};
  let scoreSum = 0;

  // Category distribution
  const categoryDistribution: Record<string, number> = {};

  // Mutation distribution
  const mutationDistribution: Record<string, number> = {};

  // Token and similarity
  let tokenSum = 0;
  let similaritySum = 0;
  let similarityCount = 0;

  // Date range
  let earliest: Date | null = null;
  let latest: Date | null = null;

  for (const ex of examples) {
    // Score
    const rawScore = ex.metadata.rawScore;
    scoreDistribution[rawScore] = (scoreDistribution[rawScore] || 0) + 1;
    scoreSum += rawScore;

    // Category
    const cat = ex.metadata.category;
    categoryDistribution[cat] = (categoryDistribution[cat] || 0) + 1;

    // Mutation
    const mut = ex.metadata.mutationType;
    mutationDistribution[mut] = (mutationDistribution[mut] || 0) + 1;

    // Token count (from features)
    const tokenCount = ex.features[0] || 0;
    tokenSum += tokenCount;

    // Similarity (from features)
    const similarity = ex.features[2];
    if (similarity !== undefined && similarity > 0) {
      similaritySum += similarity;
      similarityCount++;
    }

    // Date range
    const ts = ex.metadata.timestamp;
    if (!earliest || ts < earliest) earliest = ts;
    if (!latest || ts > latest) latest = ts;
  }

  return {
    totalExamples: examples.length,
    avgScore: scoreSum / examples.length,
    scoreDistribution,
    categoryDistribution,
    mutationDistribution,
    avgTokenCount: tokenSum / examples.length,
    avgSimilarity: similarityCount > 0 ? similaritySum / similarityCount : 0,
    dateRange: { earliest, latest },
  };
}

// ============================================================================
// EXPORT DATASET
// ============================================================================

/**
 * تصدير Dataset بتنسيق محدد
 *
 * @param dataset - Dataset للتصدير
 * @param format - تنسيق التصدير
 * @returns نتيجة التصدير
 */
export async function exportDataset(
  dataset: RewardDataset,
  format: ExportFormat
): Promise<ExportResult> {
  let data: string;
  let note: string | undefined;

  switch (format) {
    case 'json':
      data = JSON.stringify(dataset, null, 2);
      break;

    case 'jsonl':
      // JSON Lines format - one example per line
      data = dataset.examples
        .map((ex) =>
          JSON.stringify({
            id: ex.id,
            prompt_embedding: ex.promptEmbedding,
            variation_embedding: ex.variationEmbedding,
            features: ex.features,
            label: ex.label,
            weight: ex.weight,
            category: ex.metadata.category,
            mutation_type: ex.metadata.mutationType,
          })
        )
        .join('\n');
      break;

    case 'csv':
      data = exportToCSV(dataset);
      break;

    case 'parquet':
      // Parquet requires external dependency
      data = JSON.stringify(dataset.examples);
      note = 'Parquet export requires Apache Arrow library. Returning JSON as fallback.';
      break;

    case 'tfrecord':
      // TFRecord requires TensorFlow
      data = exportToJSONL_TF(dataset);
      note = 'TFRecord export requires TensorFlow. Returning TF-compatible JSON Lines.';
      break;

    default:
      data = JSON.stringify(dataset, null, 2);
  }

  return {
    format,
    data,
    size: data.length,
    note,
  };
}

/**
 * تصدير إلى CSV
 */
function exportToCSV(dataset: RewardDataset): string {
  const headers = [
    'id',
    'label',
    'weight',
    'category',
    'mutation_type',
    'raw_score',
    'timestamp',
    'user_id',
    'original_prompt',
    'variation_text',
    ...dataset.examples[0]?.featureNames.map((n) => `feature_${n}`) || [],
  ];

  const escape = (v: any): string => {
    const s = String(v ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const rows = dataset.examples.map((ex) => [
    ex.id,
    ex.label,
    ex.weight,
    ex.metadata.category,
    ex.metadata.mutationType,
    ex.metadata.rawScore,
    ex.metadata.timestamp.toISOString(),
    ex.metadata.userId || '',
    ex.metadata.originalPrompt,
    ex.metadata.variationText,
    ...ex.features,
  ]);

  return [headers.join(','), ...rows.map((r) => r.map(escape).join(','))].join('\n');
}

/**
 * تصدير إلى JSON Lines متوافق مع TensorFlow
 */
function exportToJSONL_TF(dataset: RewardDataset): string {
  return dataset.examples
    .map((ex) =>
      JSON.stringify({
        // TF-compatible format
        prompt_embedding: { values: ex.promptEmbedding },
        variation_embedding: { values: ex.variationEmbedding },
        features: { values: ex.features },
        label: ex.label,
        weight: ex.weight,
      })
    )
    .join('\n');
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * دمج datasets متعددة
 */
export function mergeDatasets(...datasets: RewardDataset[]): RewardDataset {
  const allExamples: RewardExample[] = [];
  const seenIds = new Set<string>();

  for (const ds of datasets) {
    for (const ex of ds.examples) {
      if (!seenIds.has(ex.id)) {
        seenIds.add(ex.id);
        allExamples.push(ex);
      }
    }
  }

  return {
    examples: allExamples,
    statistics: calculateDatasetStats(allExamples),
    metadata: {
      created: new Date(),
      version: DATASET_VERSION,
      size: allExamples.length,
      embeddingDimension: DEFAULT_EMBEDDING_DIMENSION,
      featureCount: allExamples[0]?.features.length || 0,
    },
  };
}

/**
 * تقسيم dataset إلى train/val/test
 */
export function splitRewardDataset(
  dataset: RewardDataset,
  trainRatio: number = 0.8,
  valRatio: number = 0.1
): { train: RewardDataset; val: RewardDataset; test: RewardDataset } {
  const examples = [...dataset.examples];

  // Shuffle
  for (let i = examples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [examples[i], examples[j]] = [examples[j], examples[i]];
  }

  const trainCount = Math.floor(examples.length * trainRatio);
  const valCount = Math.floor(examples.length * valRatio);

  const trainExamples = examples.slice(0, trainCount);
  const valExamples = examples.slice(trainCount, trainCount + valCount);
  const testExamples = examples.slice(trainCount + valCount);

  const createSubset = (exs: RewardExample[]): RewardDataset => ({
    examples: exs,
    statistics: calculateDatasetStats(exs),
    metadata: {
      created: new Date(),
      version: DATASET_VERSION,
      size: exs.length,
      embeddingDimension: DEFAULT_EMBEDDING_DIMENSION,
      featureCount: exs[0]?.features.length || 0,
    },
  });

  return {
    train: createSubset(trainExamples),
    val: createSubset(valExamples),
    test: createSubset(testExamples),
  };
}

/**
 * فلترة أمثلة بناءً على threshold معين
 */
export function filterByWeight(
  dataset: RewardDataset,
  minWeight: number
): RewardDataset {
  const filtered = dataset.examples.filter((ex) => ex.weight >= minWeight);

  return {
    examples: filtered,
    statistics: calculateDatasetStats(filtered),
    metadata: {
      ...dataset.metadata,
      created: new Date(),
      size: filtered.length,
      filters: { ...dataset.metadata.filters, minWeight },
    },
  };
}

/**
 * الحصول على ملخص Dataset
 */
export function getDatasetSummary(dataset: RewardDataset): string {
  const { statistics, metadata } = dataset;

  return `
Reward Dataset Summary
======================
Version: ${metadata.version}
Created: ${metadata.created.toISOString()}
Total Examples: ${statistics.totalExamples}

Score Statistics:
  Average: ${statistics.avgScore.toFixed(2)}
  Distribution: ${JSON.stringify(statistics.scoreDistribution)}

Category Distribution:
${Object.entries(statistics.categoryDistribution)
    .map(([cat, count]) => `  ${cat}: ${count}`)
    .join('\n')}

Mutation Distribution:
${Object.entries(statistics.mutationDistribution)
    .map(([mut, count]) => `  ${mut}: ${count}`)
    .join('\n')}

Feature Statistics:
  Average Token Count: ${statistics.avgTokenCount.toFixed(0)}
  Average Similarity: ${statistics.avgSimilarity.toFixed(3)}
  Embedding Dimension: ${metadata.embeddingDimension}
  Feature Count: ${metadata.featureCount}

Date Range:
  Earliest: ${statistics.dateRange.earliest?.toISOString() || 'N/A'}
  Latest: ${statistics.dateRange.latest?.toISOString() || 'N/A'}
`.trim();
}

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default {
  collectFeedback,
  validateFeedback,
  validateFeedbackBatch,
  buildRewardDataset,
  exportDataset,
  mergeDatasets,
  splitRewardDataset,
  filterByWeight,
  getDatasetSummary,
};
