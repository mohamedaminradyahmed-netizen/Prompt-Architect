/**
 * ReviewQueue Component
 *
 * Displays a queue of prompt variations that need human review
 * Supports filtering by priority and category
 * Shows progress counter
 */

import React, { useState, useEffect } from 'react';
import { ReviewCard } from './ReviewCard';
import { PromptCategory } from '../types/promptTypes';
import {
  getReviewQueue,
  approveReview,
  rejectReview,
  editReview,
} from '../api/review';

export interface ReviewItem {
  id: string;
  originalPrompt: string;
  suggestedVariation: string;
  mutation: string;
  score: number;
  tokenCount: number;
  estimatedCost: number;
  category: PromptCategory;
  priority: 'high' | 'medium' | 'low';
  createdAt: Date;
  metadata?: Record<string, any>;
}

export interface ReviewQueueProps {
  onApprove?: (itemId: string) => void;
  onReject?: (itemId: string, reason?: string) => void;
  onEdit?: (itemId: string, editedVariation: string) => void;
}

type FilterOption = 'all' | 'high' | 'medium' | 'low';
type CategoryFilter = 'all' | PromptCategory;

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  onApprove,
  onReject,
  onEdit
}) => {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [priorityFilter, setPriorityFilter] = useState<FilterOption>('all');
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>('all');
  const [currentIndex, setCurrentIndex] = useState(0);

  // Fetch review queue from API
  useEffect(() => {
    fetchReviewQueue();
  }, []);

  const fetchReviewQueue = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await getReviewQueue();
      setItems(data.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Filter items based on selected filters
  const filteredItems = items.filter(item => {
    const matchesPriority = priorityFilter === 'all' || item.priority === priorityFilter;
    const matchesCategory = categoryFilter === 'all' || item.category === categoryFilter;
    return matchesPriority && matchesCategory;
  });

  const handleApprove = async (itemId: string) => {
    try {
      await approveReview({ itemId });

      if (onApprove) {
        onApprove(itemId);
      }

      // Remove item from queue and move to next
      setItems(prev => prev.filter(item => item.id !== itemId));
      if (currentIndex >= filteredItems.length - 1) {
        setCurrentIndex(Math.max(0, currentIndex - 1));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve item');
    }
  };

  const handleReject = async (itemId: string, reason?: string) => {
    try {
      await rejectReview({ itemId, reason });

      if (onReject) {
        onReject(itemId, reason);
      }

      // Remove item from queue and move to next
      setItems(prev => prev.filter(item => item.id !== itemId));
      if (currentIndex >= filteredItems.length - 1) {
        setCurrentIndex(Math.max(0, currentIndex - 1));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject item');
    }
  };

  const handleEdit = async (itemId: string, editedVariation: string) => {
    try {
      await editReview({ itemId, editedVariation });

      if (onEdit) {
        onEdit(itemId, editedVariation);
      }

      // Update item in queue
      setItems(prev => prev.map(item =>
        item.id === itemId
          ? { ...item, suggestedVariation: editedVariation }
          : item
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to edit item');
    }
  };

  const handleNext = () => {
    if (currentIndex < filteredItems.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const currentItem = filteredItems[currentIndex];
  const progress = filteredItems.length > 0
    ? `${currentIndex + 1} / ${filteredItems.length}`
    : '0 / 0';

  if (loading) {
    return (
      <div className="review-queue loading">
        <div className="spinner"></div>
        <p>Loading review queue...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="review-queue error">
        <h3>Error</h3>
        <p>{error}</p>
        <button onClick={fetchReviewQueue}>Retry</button>
      </div>
    );
  }

  if (filteredItems.length === 0) {
    return (
      <div className="review-queue empty">
        <h3>Review Queue Empty</h3>
        <p>No items to review at the moment.</p>
        <button onClick={fetchReviewQueue}>Refresh</button>
      </div>
    );
  }

  return (
    <div className="review-queue">
      {/* Header with filters and progress */}
      <div className="queue-header">
        <h2>Review Queue</h2>

        <div className="progress-counter">
          <span className="progress-text">{progress}</span>
          <span className="remaining">{filteredItems.length - currentIndex - 1} remaining</span>
        </div>
      </div>

      {/* Filters */}
      <div className="queue-filters">
        <div className="filter-group">
          <label htmlFor="priority-filter">Priority:</label>
          <select
            id="priority-filter"
            value={priorityFilter}
            onChange={(e) => {
              setPriorityFilter(e.target.value as FilterOption);
              setCurrentIndex(0);
            }}
          >
            <option value="all">All</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="category-filter">Category:</label>
          <select
            id="category-filter"
            value={categoryFilter}
            onChange={(e) => {
              setCategoryFilter(e.target.value as CategoryFilter);
              setCurrentIndex(0);
            }}
          >
            <option value="all">All</option>
            <option value="CODE_GENERATION">Code Generation</option>
            <option value="CODE_REVIEW">Code Review</option>
            <option value="CONTENT_WRITING">Content Writing</option>
            <option value="MARKETING_COPY">Marketing Copy</option>
            <option value="DATA_ANALYSIS">Data Analysis</option>
            <option value="GENERAL_QA">General Q&A</option>
            <option value="CREATIVE_WRITING">Creative Writing</option>
          </select>
        </div>

        <button onClick={fetchReviewQueue} className="refresh-btn">
          🔄 Refresh
        </button>
      </div>

      {/* Current review card */}
      {currentItem && (
        <ReviewCard
          item={currentItem}
          onApprove={handleApprove}
          onReject={handleReject}
          onEdit={handleEdit}
        />
      )}

      {/* Navigation */}
      <div className="queue-navigation">
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className="nav-btn prev-btn"
        >
          ← Previous
        </button>

        <span className="nav-info">
          Item {currentIndex + 1} of {filteredItems.length}
        </span>

        <button
          onClick={handleNext}
          disabled={currentIndex >= filteredItems.length - 1}
          className="nav-btn next-btn"
        >
          Next →
        </button>
      </div>

      <style>{`
        .review-queue {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .queue-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 15px;
          border-bottom: 2px solid #e0e0e0;
        }

        .queue-header h2 {
          margin: 0;
          font-size: 24px;
          color: #333;
        }

        .progress-counter {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
        }

        .progress-text {
          font-size: 20px;
          font-weight: bold;
          color: #0066cc;
        }

        .remaining {
          font-size: 14px;
          color: #666;
          margin-top: 4px;
        }

        .queue-filters {
          display: flex;
          gap: 15px;
          margin-bottom: 20px;
          padding: 15px;
          background: #f5f5f5;
          border-radius: 8px;
          align-items: center;
        }

        .filter-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .filter-group label {
          font-weight: 500;
          color: #555;
        }

        .filter-group select {
          padding: 6px 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
        }

        .refresh-btn {
          margin-left: auto;
          padding: 8px 16px;
          background: #0066cc;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
        }

        .refresh-btn:hover {
          background: #0052a3;
        }

        .queue-navigation {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 20px;
          padding-top: 15px;
          border-top: 1px solid #e0e0e0;
        }

        .nav-btn {
          padding: 10px 20px;
          background: #0066cc;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
        }

        .nav-btn:hover:not(:disabled) {
          background: #0052a3;
        }

        .nav-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .nav-info {
          font-size: 14px;
          color: #666;
        }

        .review-queue.loading,
        .review-queue.error,
        .review-queue.empty {
          text-align: center;
          padding: 60px 20px;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid #f3f3f3;
          border-top: 4px solid #0066cc;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .review-queue.empty h3 {
          color: #666;
          margin-bottom: 10px;
        }

        .review-queue button {
          transition: all 0.2s ease;
        }
      `}</style>
    </div>
  );
};

export default ReviewQueue;
