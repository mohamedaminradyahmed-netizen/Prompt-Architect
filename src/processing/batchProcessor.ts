/**
 * DIRECTIVE-036: تنفيذ Batching للطلبات
 *
 * لماذا (Why):
 * - عند وجود أحجام كبيرة من الطلبات المتقاربة (prompts متشابهة)، يمكن تجميعها في دفعات (batches)
 *   لإرسالها عبر "نداء واحد" لمزوّد يدعم batch API، ما يقلّل التكلفة والإجمالي الزمني (overhead).
 *
 * ملاحظة هندسية:
 * - هذا المكوّن "محايد للمزوّد" (provider-agnostic). التنفيذ الفعلي للـ batch call يتم حقنه عبر
 *   دالة `batchExecutor` (Dependency Injection) لتجنّب ربط الدومين بمنفّذ محدد.
 */
 
import { calculateWordFrequencySimilarity } from '../evaluator/semanticSimilarity';
 
export interface BatchConfig {
  maxBatchSize: number;
  maxWaitTime: number; // milliseconds
  similarityThreshold: number; // 0..1
}
 
export interface ProcessRequest {
  id?: string;
  prompt: string;
  metadata?: Record<string, unknown>;
}
 
export interface ProcessResult {
  id?: string;
  output: string;
  metadata?: Record<string, unknown>;
}
 
export type BatchExecutor = (batch: ProcessRequest[]) => Promise<ProcessResult[]>;
export type SingleExecutor = (request: ProcessRequest) => Promise<ProcessResult>;
 
type Deferred = {
  request: ProcessRequest;
  resolve: (result: ProcessResult) => void;
  reject: (error: unknown) => void;
  enqueuedAt: number;
};
 
/**
 * BatchProcessor
 * - يجمع الطلبات مؤقتاً في Queue
 * - يكوّن دفعات بناءً على التشابه والحدود (size/time)
 * - ينفّذ دفعة واحدة ثم يوزّع النتائج للوعود الأصلية
 */
export class BatchProcessor {
  private readonly config: BatchConfig;
  private readonly batchExecutor: BatchExecutor;
  private readonly fallbackExecutor?: SingleExecutor;
 
  private queue: Deferred[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private flushing = false;
 
  constructor(config: BatchConfig, batchExecutor: BatchExecutor, fallbackExecutor?: SingleExecutor) {
    this.config = config;
    this.batchExecutor = batchExecutor;
    this.fallbackExecutor = fallbackExecutor;
 
    if (!Number.isFinite(config.maxBatchSize) || config.maxBatchSize < 1) {
      throw new Error('BatchConfig.maxBatchSize must be >= 1');
    }
    if (!Number.isFinite(config.maxWaitTime) || config.maxWaitTime < 0) {
      throw new Error('BatchConfig.maxWaitTime must be >= 0');
    }
    if (!Number.isFinite(config.similarityThreshold) || config.similarityThreshold < 0 || config.similarityThreshold > 1) {
      throw new Error('BatchConfig.similarityThreshold must be in [0, 1]');
    }
  }
 
  async process(request: ProcessRequest): Promise<ProcessResult> {
    return new Promise<ProcessResult>((resolve, reject) => {
      const deferred: Deferred = {
        request,
        resolve,
        reject,
        enqueuedAt: Date.now(),
      };
 
      this.queue.push(deferred);
 
      // Flush immediately if we hit the max batch size
      if (this.queue.length >= this.config.maxBatchSize) {
        void this.flush();
        return;
      }
 
      // Otherwise schedule a timer flush for the oldest waiting item
      this.ensureFlushTimer();
    });
  }
 
  /**
   * Force flush any queued requests immediately.
   */
  async flush(): Promise<void> {
    if (this.flushing) return;
    if (this.queue.length === 0) return;
 
    this.flushing = true;
    this.clearFlushTimer();
 
    const snapshot = this.queue;
    this.queue = [];
 
    try {
      const batches = this.formBatches(snapshot);
 
      for (const batch of batches) {
        await this.processBatch(batch);
      }
    } finally {
      this.flushing = false;
 
      // If new items arrived while flushing, ensure they get processed.
      if (this.queue.length > 0) {
        this.ensureFlushTimer();
        if (this.queue.length >= this.config.maxBatchSize) {
          void this.flush();
        }
      }
    }
  }
 
  private ensureFlushTimer(): void {
    if (this.flushTimer) return;
    if (this.config.maxWaitTime === 0) {
      void this.flush();
      return;
    }
 
    // Wait relative to the oldest request still queued
    const oldest = this.queue.reduce((min, item) => Math.min(min, item.enqueuedAt), Date.now());
    const age = Date.now() - oldest;
    const wait = Math.max(0, this.config.maxWaitTime - age);
 
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      void this.flush();
    }, wait);
  }
 
  private clearFlushTimer(): void {
    if (!this.flushTimer) return;
    clearTimeout(this.flushTimer);
    this.flushTimer = null;
  }
 
  /**
   * تكوين الدُفعات:
   * - Greedy clustering: كل طلب ينضم لأول batch مناسب (تشابه >= threshold) وإلا يبدأ batch جديد.
   * - حدّ batch size يتم فرضه دائماً.
   *
   * Why:
   * - بسيط، سريع، ويحقق الهدف الأساسي بدون تعقيد خوارزميات عنقودية ثقيلة.
   */
  private formBatches(items: Deferred[]): Deferred[][] {
    const batches: Deferred[][] = [];
 
    for (const item of items) {
      let placed = false;
 
      for (const batch of batches) {
        if (batch.length >= this.config.maxBatchSize) continue;
 
        const representative = batch[0].request.prompt;
        const similarity = calculateWordFrequencySimilarity(item.request.prompt, representative);
 
        if (similarity >= this.config.similarityThreshold) {
          batch.push(item);
          placed = true;
          break;
        }
      }
 
      if (!placed) {
        batches.push([item]);
      }
    }
 
    return batches;
  }
 
  private async processBatch(batch: Deferred[]): Promise<void> {
    const requests = batch.map(b => b.request);
 
    try {
      const results = await this.batchExecutor(requests);
 
      if (!Array.isArray(results) || results.length !== batch.length) {
        throw new Error(`Batch executor returned ${results?.length ?? 'non-array'} results for ${batch.length} requests`);
      }
 
      for (let i = 0; i < batch.length; i++) {
        batch[i].resolve(results[i]);
      }
    } catch (error) {
      // Fallback: attempt per-request execution if available.
      if (this.fallbackExecutor) {
        await Promise.all(batch.map(async (item) => {
          try {
            const result = await this.fallbackExecutor!(item.request);
            item.resolve(result);
          } catch (e) {
            item.reject(e);
          }
        }));
        return;
      }
 
      // Otherwise fail the whole batch.
      console.error('[BatchProcessor] batch failed', {
        error: error instanceof Error ? error.message : error,
        batchSize: batch.length,
      });
      batch.forEach(item => item.reject(error));
    }
  }
}
 
export default {
  BatchProcessor,
};

