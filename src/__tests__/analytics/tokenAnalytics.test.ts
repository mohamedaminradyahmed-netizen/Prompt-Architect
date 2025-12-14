/**
 * Unit Tests for TokenAnalytics (DIRECTIVE-040)
 * 
 * Focus: Bug fixes and core functionality validation
 */

import { TokenAnalytics, type APICall } from '../../analytics/tokenAnalytics';

describe('TokenAnalytics - Bug Fixes', () => {
  describe('Bug 2: estimateMonthlySavings with insufficient records', () => {
    it('should return 0 when there are no records', () => {
      const analytics = new TokenAnalytics(1000, 0.03);
      
      const savings = analytics.getCostSavings(1000);
      
      expect(savings.projectedMonthlySavings).toBe(0);
    });

    it('should return 0 when there is only 1 record', () => {
      const analytics = new TokenAnalytics(1000, 0.03);
      
      const call: APICall = {
        provider: 'openai',
        promptTokens: 100,
        completionTokens: 200,
        timestamp: Date.now(),
        costPerThousand: 0.03,
      };
      
      analytics.logTokenUsage(call);
      
      const savings = analytics.getCostSavings(1000);
      
      // Should return 0, not an astronomically high number
      expect(savings.projectedMonthlySavings).toBe(0);
    });

    it('should calculate correctly with 2+ records', () => {
      const analytics = new TokenAnalytics(1000, 0.03);
      
      const now = Date.now();
      const oneDayAgo = now - 24 * 60 * 60 * 1000; // 24 hours ago
      
      const call1: APICall = {
        provider: 'openai',
        promptTokens: 100,
        completionTokens: 200,
        timestamp: oneDayAgo,
        costPerThousand: 0.03,
      };
      
      const call2: APICall = {
        provider: 'openai',
        promptTokens: 100,
        completionTokens: 200,
        timestamp: now,
        costPerThousand: 0.03,
      };
      
      analytics.logTokenUsage(call1);
      analytics.logTokenUsage(call2);
      
      const savings = analytics.getCostSavings(1000);
      
      // Should return a reasonable number (not astronomically high)
      expect(savings.projectedMonthlySavings).toBeGreaterThanOrEqual(0);
      expect(savings.projectedMonthlySavings).toBeLessThan(10000); // Sanity check
    });

    it('should handle no baseline gracefully', () => {
      const analytics = new TokenAnalytics(0, 0.03);
      
      const call: APICall = {
        provider: 'openai',
        promptTokens: 100,
        completionTokens: 200,
      };
      
      analytics.logTokenUsage(call);
      analytics.logTokenUsage(call);
      
      const savings = analytics.getCostSavings(0);
      
      expect(savings.projectedMonthlySavings).toBe(0);
    });
  });

  describe('Core Functionality', () => {
    it('should track token usage correctly', () => {
      const analytics = new TokenAnalytics(500, 0.02);
      
      const call: APICall = {
        provider: 'openai',
        promptTokens: 100,
        completionTokens: 200,
      };
      
      analytics.logTokenUsage(call);
      
      const avgTokens = analytics.getAverageTokens();
      expect(avgTokens).toBe(300); // 100 + 200
    });

    it('should calculate average across multiple calls', () => {
      const analytics = new TokenAnalytics();
      
      analytics.logTokenUsage({ provider: 'openai', promptTokens: 100, completionTokens: 100 });
      analytics.logTokenUsage({ provider: 'openai', promptTokens: 200, completionTokens: 200 });
      analytics.logTokenUsage({ provider: 'openai', promptTokens: 300, completionTokens: 300 });
      
      const avgTokens = analytics.getAverageTokens();
      expect(avgTokens).toBe(400); // (200 + 400 + 600) / 3
    });

    it('should generate comprehensive report', () => {
      const analytics = new TokenAnalytics(500, 0.03);
      
      analytics.logTokenUsage({ provider: 'openai', promptTokens: 100, completionTokens: 100 });
      analytics.logTokenUsage({ provider: 'openai', promptTokens: 150, completionTokens: 150 });
      
      const report = analytics.generateTokenReport();
      
      expect(report.avgTokensPerCall).toBe(250); // (200 + 300) / 2
      expect(report.totalTokens).toBe(500);
      expect(report.totalCost).toBeGreaterThan(0);
      expect(report.reduction).toBeGreaterThan(0); // Saved vs baseline of 500
    });
  });
});
