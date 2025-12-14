/**
 * Metrics Collector (DIRECTIVE-050)
 *
 * Why:
 * توحيد نقطة جمع المقاييس يسهّل ربط Prometheus/Grafana لاحقاً دون تلويث منطق التطبيق.
 *
 * ملاحظة تنفيذية:
 * هذا الملف يستخدم prom-client إن كان متاحاً في runtime. إن لم يكن مثبتاً، تصبح الدوال no-op.
 */

type PromClient = any;

function tryLoadPromClient(): PromClient | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require('prom-client');
  } catch {
    return null;
  }
}

const prom = tryLoadPromClient();

export const register = prom?.register;

export const requestCounter = prom
  ? new prom.Counter({ name: 'refiner_requests_total', help: 'Total number of refinement requests' })
  : null;

export const errorCounter = prom
  ? new prom.Counter({ name: 'refiner_errors_total', help: 'Total number of errors' })
  : null;

export const responseTimeMs = prom
  ? new prom.Histogram({
      name: 'refiner_response_time_ms',
      help: 'Response time in milliseconds',
      buckets: [50, 100, 250, 500, 1000, 2000, 5000, 10000],
    })
  : null;

export const tokenHistogram = prom
  ? new prom.Histogram({
      name: 'refiner_tokens_used',
      help: 'Tokens used per request',
      buckets: [10, 50, 100, 500, 1000, 5000],
    })
  : null;

export const queueDepthGauge = prom
  ? new prom.Gauge({ name: 'refiner_queue_depth', help: 'Current queue depth' })
  : null;

export function incRequest(): void {
  requestCounter?.inc();
}

export function incError(): void {
  errorCounter?.inc();
}

export function observeResponseTimeMs(ms: number): void {
  responseTimeMs?.observe(ms);
}

export function observeTokens(tokens: number): void {
  tokenHistogram?.observe(tokens);
}

export function setQueueDepth(depth: number): void {
  queueDepthGauge?.set(depth);
}

export function getMetricsText(): string {
  if (!register) return '';
  return register.metrics();
}
