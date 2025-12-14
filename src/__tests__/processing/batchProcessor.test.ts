/**
 * DIRECTIVE-036: BatchProcessor Unit Tests
 *
 * الهدف: إثبات سلوك الـ batching عملياً:
 * - التجميع حسب similarityThreshold
 * - التفريغ عند maxBatchSize
 * - التفريغ عند maxWaitTime (timer)
 * - fallback عند فشل batchExecutor
 */

import { BatchProcessor, type BatchConfig, type ProcessRequest, type ProcessResult } from '../../processing/batchProcessor';

function createResult(req: ProcessRequest): ProcessResult {
  return { id: req.id, output: `OUT:${req.prompt}` };
}

describe('DIRECTIVE-036: BatchProcessor', () => {
  test('should group similar prompts into the same batch', async () => {
    const calls: Array<ProcessRequest[]> = [];

    const config: BatchConfig = {
      maxBatchSize: 10,
      maxWaitTime: 10_000,
      similarityThreshold: 0.5,
    };

    const processor = new BatchProcessor(
      config,
      async (batch) => {
        calls.push(batch);
        return batch.map(createResult);
      }
    );

    const p1 = processor.process({ id: '1', prompt: 'Write a function to sort an array of numbers' });
    const p2 = processor.process({ id: '2', prompt: 'Write a function to sort a list of integers' });
    const p3 = processor.process({ id: '3', prompt: 'Generate marketing copy for running shoes' });

    await processor.flush();

    const r1 = await p1;
    const r2 = await p2;
    const r3 = await p3;

    expect(r1.output).toContain('OUT:');
    expect(r2.output).toContain('OUT:');
    expect(r3.output).toContain('OUT:');

    // Expect two batches: [p1,p2] and [p3] (order may vary by greedy placement, but grouping should hold)
    const batchSizes = calls.map(c => c.length).sort((a, b) => a - b);
    expect(batchSizes).toEqual([1, 2]);

    const batchWith2 = calls.find(c => c.length === 2)!;
    const ids = batchWith2.map(r => r.id).sort();
    expect(ids).toEqual(['1', '2']);
  });

  test('should flush immediately when reaching maxBatchSize', async () => {
    const calls: Array<ProcessRequest[]> = [];

    const config: BatchConfig = {
      maxBatchSize: 2,
      maxWaitTime: 60_000,
      similarityThreshold: 0,
    };

    const processor = new BatchProcessor(
      config,
      async (batch) => {
        calls.push(batch);
        return batch.map(createResult);
      }
    );

    const p1 = processor.process({ id: 'a', prompt: 'A' });
    const p2 = processor.process({ id: 'b', prompt: 'B' });

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1.output).toBe('OUT:A');
    expect(r2.output).toBe('OUT:B');

    expect(calls.length).toBe(1);
    expect(calls[0].length).toBe(2);
  });

  test('should flush after maxWaitTime elapses', async () => {
    jest.useFakeTimers();

    const calls: Array<ProcessRequest[]> = [];
    const config: BatchConfig = {
      maxBatchSize: 10,
      maxWaitTime: 50,
      similarityThreshold: 0,
    };

    const processor = new BatchProcessor(
      config,
      async (batch) => {
        calls.push(batch);
        return batch.map(createResult);
      }
    );

    const promise = processor.process({ id: 't', prompt: 'Timer flush' });

    jest.advanceTimersByTime(60);
    // ensure timer callback + async flush path completes
    await jest.runOnlyPendingTimersAsync();

    const result = await promise;
    expect(result.output).toBe('OUT:Timer flush');
    expect(calls.length).toBe(1);
    expect(calls[0].length).toBe(1);

    jest.useRealTimers();
  });

  test('should fallback to per-request execution if batchExecutor fails', async () => {
    const config: BatchConfig = {
      maxBatchSize: 3,
      maxWaitTime: 10_000,
      similarityThreshold: 0,
    };

    const processor = new BatchProcessor(
      config,
      async () => {
        throw new Error('batch failed');
      },
      async (req) => ({ id: req.id, output: `FALLBACK:${req.prompt}` })
    );

    const p1 = processor.process({ id: 'x', prompt: 'one' });
    const p2 = processor.process({ id: 'y', prompt: 'two' });

    await processor.flush();

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1.output).toBe('FALLBACK:one');
    expect(r2.output).toBe('FALLBACK:two');
  });
});

