export type TimeRange = { from: number; to: number };
export type CostSavings = { baselineTokens: number; currentTokens: number; reductionPct: number; projectedMonthlySavings: number };
export type TokenReport = { avgTokensPerCall: number; totalTokens: number; totalCost: number; reduction: number; projectedMonthlySavings: number };
export interface APICall { provider: string; promptTokens: number; completionTokens: number; timestamp?: number; costPerThousand?: number; operationId?: string; metadata?: Record<string, unknown>; }
type TokenUsageRecord = APICall & { timestamp: number; totalTokens: number; cost: number };
export class TokenAnalytics {
  private records: TokenUsageRecord[] = [];
  constructor(private baselineTokensPerCall = 0, private defaultCostPerThousand = 0.0) {}
  logTokenUsage(call: APICall): void {
    try {
      const timestamp = call.timestamp ?? Date.now();
      const totalTokens = Math.max(0, call.promptTokens) + Math.max(0, call.completionTokens);
      if (!Number.isFinite(totalTokens)) throw new Error('invalid tokens');
      const rate = call.costPerThousand ?? this.defaultCostPerThousand;
      const cost = (totalTokens / 1000) * rate;
      this.records.push({ ...call, timestamp, totalTokens, cost });
      console.info('[token-usage]', { provider: call.provider, totalTokens, cost, timestamp });
    } catch (error) {
      console.error('[token-usage][error]', { error, call });
    }
  }
  private filter = (range?: TimeRange) => (!range ? this.records : this.records.filter(r => r.timestamp >= range.from && r.timestamp <= range.to));
  getAverageTokens(range?: TimeRange): number {
    const scoped = this.filter(range);
    return scoped.length ? scoped.reduce((sum, r) => sum + r.totalTokens, 0) / scoped.length : 0;
  }
  getCostSavings(baselineTokens?: number): CostSavings {
    const baseline = baselineTokens ?? this.baselineTokensPerCall;
    const current = this.getAverageTokens();
    const reductionPct = baseline > 0 ? (Math.max(0, baseline - current) / baseline) * 100 : 0;
    return { baselineTokens: baseline, currentTokens: current, reductionPct, projectedMonthlySavings: this.estimateMonthlySavings(baseline, current) };
  }
  generateTokenReport(range?: TimeRange): TokenReport {
    const scoped = this.filter(range);
    const totalTokens = scoped.reduce((sum, r) => sum + r.totalTokens, 0);
    const totalCost = scoped.reduce((sum, r) => sum + r.cost, 0);
    const avgTokensPerCall = scoped.length ? totalTokens / scoped.length : 0;
    const savings = this.getCostSavings();
    return { avgTokensPerCall, totalTokens, totalCost, reduction: savings.reductionPct, projectedMonthlySavings: savings.projectedMonthlySavings };
  }
  private estimateMonthlySavings(baseline: number, current: number): number {
    if (!baseline || !this.records.length) return 0;
    
    // Why (Bug Fix):
    // - عندما يكون records.length < 2، فإن windowMs = 0 (لأن records[0] === records[length-1])
    // - بعد Math.max(1, ...) يصبح windowMs = 1ms، ما يجعل dailyCalls ضخم جداً (86.4M)
    // - الحل: إرجاع 0 إذا لم يكن هناك على الأقل سجلان لحساب نافذة زمنية معقولة.
    if (this.records.length < 2) return 0;
    
    const windowMs = Math.max(1, (this.records[this.records.length - 1].timestamp - this.records[0].timestamp) || 1);
    const dailyCalls = this.records.length / Math.max(1, windowMs / (1000 * 60 * 60 * 24));
    const unitCost = this.defaultCostPerThousand || 0;
    return Math.max(0, (((baseline - current) / 1000) * unitCost) * dailyCalls * 30);
  }
}
