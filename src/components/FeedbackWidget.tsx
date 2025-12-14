import React, { useState } from 'react';
import { Star, MessageSquare } from 'lucide-react';
import { storeFeedback, HumanFeedback } from '../api/feedback';

interface FeedbackWidgetProps {
  promptId: string;
  variationId: string;
  userId?: string;
  onFeedbackSubmitted?: () => void;
}

export const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({
  promptId,
  variationId,
  userId = 'anonymous',
  onFeedbackSubmitted
}) => {
  const [rating, setRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);
  const [comment, setComment] = useState('');
  const [showComment, setShowComment] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (rating === 0) return;

    const feedback: HumanFeedback = {
      promptId,
      variationId,
      score: rating,
      feedbackText: comment || undefined,
      userId,
    };

    await storeFeedback(feedback);
    setSubmitted(true);
    onFeedbackSubmitted?.();
  };

  const quickFeedback = async (score: number, text: string) => {
    const feedback: HumanFeedback = {
      promptId,
      variationId,
      score,
      feedbackText: text,
      userId,
    };

    await storeFeedback(feedback);
    setSubmitted(true);
    onFeedbackSubmitted?.();
  };

  if (submitted) {
    return (
      <div style={{ 
        padding: 8, 
        background: '#f0f9ff', 
        borderRadius: 6, 
        fontSize: 12, 
        color: '#0369a1' 
      }}>
        ✓ Thank you for your feedback!
      </div>
    );
  }

  return (
    <div style={{ padding: 8, borderTop: '1px solid #e5e7eb' }}>
      <div style={{ fontSize: 12, marginBottom: 8, color: '#6b7280' }}>
        Rate this suggestion:
      </div>
      
      {/* Star Rating */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            size={16}
            style={{
              cursor: 'pointer',
              fill: (hoveredRating || rating) >= star ? '#fbbf24' : 'none',
              stroke: (hoveredRating || rating) >= star ? '#fbbf24' : '#d1d5db',
            }}
            onMouseEnter={() => setHoveredRating(star)}
            onMouseLeave={() => setHoveredRating(0)}
            onClick={() => setRating(star)}
          />
        ))}
      </div>

      {/* Quick Feedback Buttons */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
        <button
          onClick={() => quickFeedback(5, 'Perfect')}
          style={{
            padding: '4px 8px',
            fontSize: 10,
            border: '1px solid #d1d5db',
            borderRadius: 4,
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          Perfect
        </button>
        <button
          onClick={() => quickFeedback(4, 'Good')}
          style={{
            padding: '4px 8px',
            fontSize: 10,
            border: '1px solid #d1d5db',
            borderRadius: 4,
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          Good
        </button>
        <button
          onClick={() => quickFeedback(2, 'Needs Work')}
          style={{
            padding: '4px 8px',
            fontSize: 10,
            border: '1px solid #d1d5db',
            borderRadius: 4,
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          Needs Work
        </button>
        <button
          onClick={() => quickFeedback(1, 'Poor')}
          style={{
            padding: '4px 8px',
            fontSize: 10,
            border: '1px solid #d1d5db',
            borderRadius: 4,
            background: '#fff',
            cursor: 'pointer'
          }}
        >
          Poor
        </button>
      </div>

      {/* Comment Toggle */}
      <button
        onClick={() => setShowComment(!showComment)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          fontSize: 10,
          color: '#6b7280',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          marginBottom: showComment ? 8 : 0
        }}
      >
        <MessageSquare size={12} />
        Add comment
      </button>

      {/* Comment Field */}
      {showComment && (
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Optional feedback..."
          style={{
            width: '100%',
            padding: 6,
            fontSize: 11,
            border: '1px solid #d1d5db',
            borderRadius: 4,
            resize: 'vertical',
            minHeight: 40,
            marginBottom: 8
          }}
        />
      )}

      {/* Submit Button */}
      {(rating > 0 || comment) && (
        <button
          onClick={handleSubmit}
          style={{
            padding: '4px 12px',
            fontSize: 11,
            background: '#3b82f6',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer'
          }}
        >
          Submit Feedback
        </button>
      )}
    </div>
  );
};