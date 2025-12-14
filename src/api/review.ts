/**
 * Review Queue API
 * Handles review queue operations for prompt variations
 * Mock implementation using localStorage
 */

import { ReviewItem } from '../components/ReviewQueue';
import { PromptCategory } from '../types/promptTypes';

export interface ReviewQueueResponse {
  items: ReviewItem[];
  total: number;
}

export interface ApproveRequest {
  itemId: string;
  notes?: string;
  userId?: string;
}

export interface RejectRequest {
  itemId: string;
  reason?: string;
  userId?: string;
}

export interface EditRequest {
  itemId: string;
  editedVariation: string;
  userId?: string;
}

export interface ReviewAction {
  id: string;
  itemId: string;
  action: 'approve' | 'reject' | 'edit';
  timestamp: Date;
  userId: string;
  metadata?: {
    reason?: string;
    notes?: string;
    originalText?: string;
    editedText?: string;
  };
}

// ============================================================================
// STORAGE KEYS
// ============================================================================

const STORAGE_KEYS = {
  REVIEW_QUEUE: 'review_queue',
  REVIEW_ACTIONS: 'review_actions',
  APPROVED_ITEMS: 'approved_items',
  REJECTED_ITEMS: 'rejected_items',
};

// ============================================================================
// GET REVIEW QUEUE
// ============================================================================

/**
 * Get all pending review items
 */
export async function getReviewQueue(): Promise<ReviewQueueResponse> {
  const items = getQueueFromStorage();

  return {
    items,
    total: items.length,
  };
}

/**
 * Get filtered review queue
 */
export async function getFilteredQueue(
  priority?: 'high' | 'medium' | 'low',
  category?: PromptCategory
): Promise<ReviewQueueResponse> {
  let items = getQueueFromStorage();

  if (priority) {
    items = items.filter(item => item.priority === priority);
  }

  if (category) {
    items = items.filter(item => item.category === category);
  }

  return {
    items,
    total: items.length,
  };
}

// ============================================================================
// APPROVE REVIEW
// ============================================================================

/**
 * Approve a review item
 */
export async function approveReview(request: ApproveRequest): Promise<void> {
  const { itemId, notes, userId = 'anonymous' } = request;

  // Get the item from queue
  const queue = getQueueFromStorage();
  const item = queue.find(i => i.id === itemId);

  if (!item) {
    throw new Error(`Review item ${itemId} not found`);
  }

  // Record the approval action
  const action: ReviewAction = {
    id: generateId(),
    itemId,
    action: 'approve',
    timestamp: new Date(),
    userId,
    metadata: { notes },
  };

  recordAction(action);

  // Move to approved items
  const approved = getApprovedFromStorage();
  approved.push({
    ...item,
    approvedAt: new Date(),
    approvedBy: userId,
    notes,
  });
  localStorage.setItem(STORAGE_KEYS.APPROVED_ITEMS, JSON.stringify(approved));

  // Remove from queue
  removeFromQueue(itemId);
}

/**
 * Approve multiple items
 */
export async function approveMultiple(itemIds: string[], userId?: string): Promise<void> {
  for (const itemId of itemIds) {
    await approveReview({ itemId, userId });
  }
}

// ============================================================================
// REJECT REVIEW
// ============================================================================

/**
 * Reject a review item
 */
export async function rejectReview(request: RejectRequest): Promise<void> {
  const { itemId, reason, userId = 'anonymous' } = request;

  // Get the item from queue
  const queue = getQueueFromStorage();
  const item = queue.find(i => i.id === itemId);

  if (!item) {
    throw new Error(`Review item ${itemId} not found`);
  }

  // Record the rejection action
  const action: ReviewAction = {
    id: generateId(),
    itemId,
    action: 'reject',
    timestamp: new Date(),
    userId,
    metadata: { reason },
  };

  recordAction(action);

  // Move to rejected items
  const rejected = getRejectedFromStorage();
  rejected.push({
    ...item,
    rejectedAt: new Date(),
    rejectedBy: userId,
    reason,
  });
  localStorage.setItem(STORAGE_KEYS.REJECTED_ITEMS, JSON.stringify(rejected));

  // Remove from queue
  removeFromQueue(itemId);
}

/**
 * Reject multiple items
 */
export async function rejectMultiple(
  itemIds: string[],
  reason?: string,
  userId?: string
): Promise<void> {
  for (const itemId of itemIds) {
    await rejectReview({ itemId, reason, userId });
  }
}

// ============================================================================
// EDIT REVIEW
// ============================================================================

/**
 * Edit a review item's suggested variation
 */
export async function editReview(request: EditRequest): Promise<void> {
  const { itemId, editedVariation, userId = 'anonymous' } = request;

  // Get the item from queue
  const queue = getQueueFromStorage();
  const itemIndex = queue.findIndex(i => i.id === itemId);

  if (itemIndex === -1) {
    throw new Error(`Review item ${itemId} not found`);
  }

  const item = queue[itemIndex];

  // Record the edit action
  const action: ReviewAction = {
    id: generateId(),
    itemId,
    action: 'edit',
    timestamp: new Date(),
    userId,
    metadata: {
      originalText: item.suggestedVariation,
      editedText: editedVariation,
    },
  };

  recordAction(action);

  // Update the item
  queue[itemIndex] = {
    ...item,
    suggestedVariation: editedVariation,
    // Recalculate token count
    tokenCount: estimateTokens(editedVariation),
  };

  localStorage.setItem(STORAGE_KEYS.REVIEW_QUEUE, JSON.stringify(queue));
}

// ============================================================================
// QUEUE MANAGEMENT
// ============================================================================

/**
 * Add item to review queue
 */
export async function addToQueue(item: Omit<ReviewItem, 'id'>): Promise<ReviewItem> {
  const queue = getQueueFromStorage();

  const newItem: ReviewItem = {
    ...item,
    id: generateId(),
  };

  queue.push(newItem);
  localStorage.setItem(STORAGE_KEYS.REVIEW_QUEUE, JSON.stringify(queue));

  return newItem;
}

/**
 * Add multiple items to queue
 */
export async function addMultipleToQueue(
  items: Omit<ReviewItem, 'id'>[]
): Promise<ReviewItem[]> {
  const queue = getQueueFromStorage();
  const newItems: ReviewItem[] = [];

  for (const item of items) {
    const newItem: ReviewItem = {
      ...item,
      id: generateId(),
    };
    newItems.push(newItem);
    queue.push(newItem);
  }

  localStorage.setItem(STORAGE_KEYS.REVIEW_QUEUE, JSON.stringify(queue));
  return newItems;
}

/**
 * Remove item from queue
 */
export async function removeFromQueue(itemId: string): Promise<void> {
  const queue = getQueueFromStorage();
  const filtered = queue.filter(item => item.id !== itemId);
  localStorage.setItem(STORAGE_KEYS.REVIEW_QUEUE, JSON.stringify(filtered));
}

/**
 * Clear entire queue
 */
export async function clearQueue(): Promise<void> {
  localStorage.setItem(STORAGE_KEYS.REVIEW_QUEUE, JSON.stringify([]));
}

/**
 * Get queue statistics
 */
export async function getQueueStats(): Promise<{
  total: number;
  byPriority: Record<string, number>;
  byCategory: Record<string, number>;
}> {
  const items = getQueueFromStorage();

  const byPriority: Record<string, number> = {};
  const byCategory: Record<string, number> = {};

  items.forEach(item => {
    byPriority[item.priority] = (byPriority[item.priority] || 0) + 1;
    byCategory[item.category] = (byCategory[item.category] || 0) + 1;
  });

  return {
    total: items.length,
    byPriority,
    byCategory,
  };
}

// ============================================================================
// REVIEW HISTORY
// ============================================================================

/**
 * Get all review actions
 */
export async function getReviewHistory(
  itemId?: string
): Promise<ReviewAction[]> {
  const actions = getActionsFromStorage();

  if (itemId) {
    return actions.filter(action => action.itemId === itemId);
  }

  return actions;
}

/**
 * Get approved items
 */
export async function getApprovedItems(): Promise<any[]> {
  return getApprovedFromStorage();
}

/**
 * Get rejected items
 */
export async function getRejectedItems(): Promise<any[]> {
  return getRejectedFromStorage();
}

/**
 * Get review statistics
 */
export async function getReviewStats(): Promise<{
  totalReviewed: number;
  approved: number;
  rejected: number;
  edited: number;
  pending: number;
}> {
  const actions = getActionsFromStorage();
  const queue = getQueueFromStorage();

  const approved = actions.filter(a => a.action === 'approve').length;
  const rejected = actions.filter(a => a.action === 'reject').length;
  const edited = actions.filter(a => a.action === 'edit').length;

  return {
    totalReviewed: approved + rejected,
    approved,
    rejected,
    edited,
    pending: queue.length,
  };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getQueueFromStorage(): ReviewItem[] {
  const stored = localStorage.getItem(STORAGE_KEYS.REVIEW_QUEUE);
  if (!stored) return [];

  const items = JSON.parse(stored);
  // Convert date strings back to Date objects
  return items.map((item: any) => ({
    ...item,
    createdAt: new Date(item.createdAt),
  }));
}

function getActionsFromStorage(): ReviewAction[] {
  const stored = localStorage.getItem(STORAGE_KEYS.REVIEW_ACTIONS);
  if (!stored) return [];

  const actions = JSON.parse(stored);
  return actions.map((action: any) => ({
    ...action,
    timestamp: new Date(action.timestamp),
  }));
}

function getApprovedFromStorage(): any[] {
  const stored = localStorage.getItem(STORAGE_KEYS.APPROVED_ITEMS);
  return stored ? JSON.parse(stored) : [];
}

function getRejectedFromStorage(): any[] {
  const stored = localStorage.getItem(STORAGE_KEYS.REJECTED_ITEMS);
  return stored ? JSON.parse(stored) : [];
}

function recordAction(action: ReviewAction): void {
  const actions = getActionsFromStorage();
  actions.push(action);
  localStorage.setItem(STORAGE_KEYS.REVIEW_ACTIONS, JSON.stringify(actions));
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function estimateTokens(text: string): number {
  // Simple estimation: ~4 characters per token
  return Math.ceil(text.length / 4);
}

// ============================================================================
// SEED DATA (for testing)
// ============================================================================

/**
 * Seed the queue with sample data for testing
 */
export async function seedReviewQueue(): Promise<void> {
  const sampleItems: Omit<ReviewItem, 'id'>[] = [
    {
      originalPrompt: 'Write a function to calculate fibonacci numbers',
      suggestedVariation: 'Create an efficient function that computes fibonacci numbers using dynamic programming',
      mutation: 'Expand with specificity',
      score: 85.5,
      tokenCount: 18,
      estimatedCost: 0.0004,
      category: 'CODE_GENERATION' as PromptCategory,
      priority: 'high',
      createdAt: new Date(),
      metadata: { source: 'refiner', iteration: 1 },
    },
    {
      originalPrompt: 'Review this code for bugs',
      suggestedVariation: 'Perform a comprehensive code review focusing on potential bugs, security vulnerabilities, and performance issues',
      mutation: 'Add context and constraints',
      score: 78.2,
      tokenCount: 21,
      estimatedCost: 0.0005,
      category: 'CODE_REVIEW' as PromptCategory,
      priority: 'high',
      createdAt: new Date(),
      metadata: { source: 'refiner', iteration: 2 },
    },
    {
      originalPrompt: 'Write a blog post about AI',
      suggestedVariation: 'Write an engaging 800-word blog post about recent advances in AI, targeting a general audience',
      mutation: 'Add length and audience',
      score: 72.8,
      tokenCount: 19,
      estimatedCost: 0.0004,
      category: 'CONTENT_WRITING' as PromptCategory,
      priority: 'medium',
      createdAt: new Date(),
      metadata: { source: 'refiner', iteration: 1 },
    },
    {
      originalPrompt: 'Analyze this data',
      suggestedVariation: 'Analyze this sales data to identify trends, outliers, and actionable insights. Present findings in a clear, structured format.',
      mutation: 'Specify output format',
      score: 81.3,
      tokenCount: 24,
      estimatedCost: 0.0006,
      category: 'DATA_ANALYSIS' as PromptCategory,
      priority: 'medium',
      createdAt: new Date(),
      metadata: { source: 'refiner', iteration: 3 },
    },
    {
      originalPrompt: 'Create marketing copy',
      suggestedVariation: 'Create compelling marketing copy for a SaaS product targeting small businesses, emphasizing ROI and ease of use',
      mutation: 'Add target audience',
      score: 68.9,
      tokenCount: 22,
      estimatedCost: 0.0005,
      category: 'MARKETING_COPY' as PromptCategory,
      priority: 'low',
      createdAt: new Date(),
      metadata: { source: 'refiner', iteration: 1 },
    },
  ];

  await addMultipleToQueue(sampleItems);
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  // Queue operations
  getReviewQueue,
  getFilteredQueue,
  addToQueue,
  addMultipleToQueue,
  removeFromQueue,
  clearQueue,
  getQueueStats,

  // Review actions
  approveReview,
  approveMultiple,
  rejectReview,
  rejectMultiple,
  editReview,

  // History
  getReviewHistory,
  getApprovedItems,
  getRejectedItems,
  getReviewStats,

  // Utilities
  seedReviewQueue,
};
