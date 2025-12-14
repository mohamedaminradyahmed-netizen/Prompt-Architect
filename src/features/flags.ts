/**
 * Feature Flags (DIRECTIVE-051)
 *
 * Why:
 * وجود طبقة feature flags يتيح A/B testing و gradual rollout و kill-switch بدون نشر سريع لتغييرات منطقية.
 */

import * as crypto from 'crypto';

export type FlagName =
  | 'use_genetic_optimizer'
  | 'enable_rl_policy'
  | 'show_advanced_metrics'
  | 'enable_human_review';

export class FeatureFlags {
  private flags: Record<string, boolean> = {
    use_genetic_optimizer: false,
    enable_rl_policy: false,
    show_advanced_metrics: false,
    enable_human_review: true,
  };

  async isEnabled(flagName: string, userId?: string): Promise<boolean> {
    // userId optional: يمكن لاحقاً ربطها بملف تعريف المستخدم/التجربة.
    void userId;
    return Boolean(this.flags[flagName]);
  }

  async getVariant(experiment: string, userId: string): Promise<string> {
    // Deterministic bucketing: نفس المستخدم يحصل على نفس variant دائماً.
    const hash = crypto.createHash('sha256').update(`${experiment}:${userId}`).digest('hex');
    const bucket = parseInt(hash.slice(0, 8), 16) % 100;
    return bucket < 50 ? 'control' : 'treatment';
  }

  async track(event: string, properties: any): Promise<void> {
    // Placeholder: اربطها لاحقاً بـ PostHog/Unleash/LaunchDarkly.
    void event;
    void properties;
  }

  // Helper لتحديث الرايات محلياً/في الاختبارات
  set(flagName: FlagName, enabled: boolean): void {
    this.flags[flagName] = enabled;
  }
}

export const featureFlags = new FeatureFlags();
