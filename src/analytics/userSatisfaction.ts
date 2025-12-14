/**
 * User Satisfaction Analytics System
 * Implements DIRECTIVE-041 for tracking NPS, Accept Rate, and Satisfaction metrics.
 * 
 * Architecture Decisions:
 * 1. Separation of concerns: Analytics logic is isolated from UI components.
 * 2. Robustness: Internal error handling and logging wrapper used.
 * 3. Type Safety: Strict interfaces for all data structures.
 * 4. Extensibility: Designed to be easily connected to a persistent DB later.
 */

export interface TimeRange {
  startDate: Date;
  endDate: Date;
}

export interface SatisfactionReport {
  acceptRate: number;      // Percentage 0-100
  npsScore: number;        // Score -100 to 100
  averageSatisfaction: number; // 1-5 stars
  totalSuggestions: number;
  acceptedSuggestions: number;
  npsResponses: number;
  satisfactionRatings: number;
  featureUsage: Map<string, number>; // Mutation type -> count
}

interface InteractionEvent {
  id: string;
  userId?: string;
  timestamp: Date;
  type: 'acceptance' | 'nps' | 'satisfaction';
  data: any;
}

interface AcceptanceData {
  suggestionId: string;
  accepted: boolean;
  mutationType?: string; // For feature usage tracking
}

interface NPSData {
  score: number; // 0-10
}

interface SatisfactionData {
  sessionId: string;
  rating: number; // 1-5
}

/**
 * Tracks user satisfaction metrics including NPS, Acceptance Rate, and Star Ratings.
 */
export class UserSatisfactionTracker {
  // In-memory storage for MVP. In production, this would be replaced by a database repository.
  private events: InteractionEvent[] = [];
  
  /**
   * Logs an internal message with timestamp and level.
   * Ensures observability without cluttering standard output with raw prints.
   */
  private log(level: 'INFO' | 'WARN' | 'ERROR', message: string, meta?: any) {
    const timestamp = new Date().toISOString();
    // In a real system, this would go to ELK/Datadog
    console.log(`[${timestamp}] [${level}] [UserSatisfactionTracker] ${message}`, meta || '');
  }

  /**
   * Records whether a user accepted or rejected a suggestion.
   * Vital for measuring the "Accept Rate" KPI.
   */
  public logAcceptance(suggestionId: string, accepted: boolean, mutationType?: string): void {
    try {
      this.events.push({
        id: crypto.randomUUID(),
        timestamp: new Date(),
        type: 'acceptance',
        data: { suggestionId, accepted, mutationType } as AcceptanceData
      });
      this.log('INFO', `Logged acceptance: ${accepted} for suggestion ${suggestionId}`);
    } catch (error) {
      this.log('ERROR', 'Failed to log acceptance', error);
    }
  }

  /**
   * Records a Net Promoter Score (NPS) response.
   * Range: 0-10
   */
  public logNPSScore(userId: string, score: number): void {
    try {
      if (score < 0 || score > 10) {
        throw new Error(`Invalid NPS score: ${score}. Must be 0-10.`);
      }
      this.events.push({
        id: crypto.randomUUID(),
        userId,
        timestamp: new Date(),
        type: 'nps',
        data: { score } as NPSData
      });
      this.log('INFO', `Logged NPS from user ${userId}: ${score}`);
    } catch (error) {
      this.log('ERROR', 'Failed to log NPS', error);
    }
  }

  /**
   * Records a user satisfaction rating (e.g., 5-star rating).
   * Range: 1-5
   */
  public logSatisfactionRating(sessionId: string, rating: number): void {
    try {
      if (rating < 1 || rating > 5) {
        throw new Error(`Invalid satisfaction rating: ${rating}. Must be 1-5.`);
      }
      this.events.push({
        id: crypto.randomUUID(),
        timestamp: new Date(),
        type: 'satisfaction',
        data: { sessionId, rating } as SatisfactionData
      });
      this.log('INFO', `Logged satisfaction for session ${sessionId}: ${rating}`);
    } catch (error) {
      this.log('ERROR', 'Failed to log satisfaction rating', error);
    }
  }

  /**
   * Calculates the Acceptance Rate within a given time range.
   * Formula: (Accepted Suggestions / Total Suggestions) * 100
   */
  public getAcceptRate(timeRange: TimeRange): number {
    const relevantEvents = this.filterEvents(timeRange, 'acceptance');
    if (relevantEvents.length === 0) return 0;

    const acceptedCount = relevantEvents.filter(e => (e.data as AcceptanceData).accepted).length;
    return (acceptedCount / relevantEvents.length) * 100;
  }

  /**
   * Calculates the Net Promoter Score (NPS).
   * Formula: % Promoters (9-10) - % Detractors (0-6)
   */
  public getNPS(timeRange: TimeRange): number {
    const relevantEvents = this.filterEvents(timeRange, 'nps');
    const total = relevantEvents.length;
    if (total === 0) return 0;

    let promoters = 0;
    let detractors = 0;

    relevantEvents.forEach(e => {
      const score = (e.data as NPSData).score;
      if (score >= 9) promoters++;
      else if (score <= 6) detractors++;
    });

    const promoterPct = (promoters / total) * 100;
    const detractorPct = (detractors / total) * 100;

    return Number((promoterPct - detractorPct).toFixed(2));
  }

  /**
   * Calculates the average satisfaction rating (1-5).
   */
  public getAverageSatisfaction(timeRange: TimeRange): number {
    const relevantEvents = this.filterEvents(timeRange, 'satisfaction');
    if (relevantEvents.length === 0) return 0;

    const sum = relevantEvents.reduce((acc, e) => acc + (e.data as SatisfactionData).rating, 0);
    return Number((sum / relevantEvents.length).toFixed(2));
  }

  /**
   * Generates a comprehensive satisfaction report.
   * Aggregates all metrics for analysis.
   */
  public generateSatisfactionReport(timeRange?: TimeRange): SatisfactionReport {
    // Default to all time if no range provided
    const range = timeRange || { startDate: new Date(0), endDate: new Date() };

    const acceptEvents = this.filterEvents(range, 'acceptance');
    const acceptedCount = acceptEvents.filter(e => (e.data as AcceptanceData).accepted).length;
    
    const featureUsage = new Map<string, number>();
    acceptEvents.forEach(e => {
      const data = e.data as AcceptanceData;
      if (data.accepted && data.mutationType) {
        featureUsage.set(data.mutationType, (featureUsage.get(data.mutationType) || 0) + 1);
      }
    });

    return {
      acceptRate: this.getAcceptRate(range),
      npsScore: this.getNPS(range),
      averageSatisfaction: this.getAverageSatisfaction(range),
      totalSuggestions: acceptEvents.length,
      acceptedSuggestions: acceptedCount,
      npsResponses: this.filterEvents(range, 'nps').length,
      satisfactionRatings: this.filterEvents(range, 'satisfaction').length,
      featureUsage
    };
  }

  private filterEvents(timeRange: TimeRange, type: string): InteractionEvent[] {
    return this.events.filter(e => 
      e.type === type && 
      e.timestamp >= timeRange.startDate && 
      e.timestamp <= timeRange.endDate
    );
  }
}
