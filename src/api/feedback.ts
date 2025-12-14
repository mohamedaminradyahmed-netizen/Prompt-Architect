/**
 * Human Feedback API
 * Handles collection and storage of human feedback for prompt variations
 */

export interface HumanFeedback {
  id?: string;
  promptId: string;
  variationId: string;
  score: number; // 1-5
  feedbackText?: string;
  userId: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

export interface FeedbackStats {
  averageScore: number;
  totalFeedback: number;
  scoreDistribution: Record<number, number>;
}

/**
 * Store human feedback
 */
export async function storeFeedback(feedback: HumanFeedback): Promise<void> {
  // Mock implementation - in production, this would use a real database
  const feedbackWithId = {
    ...feedback,
    id: generateId(),
    timestamp: new Date(),
  };
  
  // Store in localStorage for now
  const existing = getFeedbackFromStorage();
  existing.push(feedbackWithId);
  localStorage.setItem('human_feedback', JSON.stringify(existing));
}

/**
 * Get average feedback score for a variation
 */
export async function getAverageFeedback(variationId: string): Promise<number> {
  const feedback = getFeedbackFromStorage();
  const variationFeedback = feedback.filter(f => f.variationId === variationId);
  
  if (variationFeedback.length === 0) return 0;
  
  const sum = variationFeedback.reduce((acc, f) => acc + f.score, 0);
  return sum / variationFeedback.length;
}

/**
 * Get feedback statistics
 */
export async function getFeedbackStats(variationId?: string): Promise<FeedbackStats> {
  const feedback = getFeedbackFromStorage();
  const targetFeedback = variationId 
    ? feedback.filter(f => f.variationId === variationId)
    : feedback;

  if (targetFeedback.length === 0) {
    return {
      averageScore: 0,
      totalFeedback: 0,
      scoreDistribution: {},
    };
  }

  const sum = targetFeedback.reduce((acc, f) => acc + f.score, 0);
  const distribution: Record<number, number> = {};
  
  targetFeedback.forEach(f => {
    distribution[f.score] = (distribution[f.score] || 0) + 1;
  });

  return {
    averageScore: sum / targetFeedback.length,
    totalFeedback: targetFeedback.length,
    scoreDistribution: distribution,
  };
}

// Helper functions
function getFeedbackFromStorage(): HumanFeedback[] {
  const stored = localStorage.getItem('human_feedback');
  return stored ? JSON.parse(stored) : [];
}

function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}