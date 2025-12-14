/**
 * ReviewCard Component
 *
 * Displays a single prompt variation for review
 * Shows original vs suggested prompts side-by-side
 * Includes metrics, action buttons, and notes field
 */

import React, { useState } from 'react';
import { ReviewItem } from './ReviewQueue';

export interface ReviewCardProps {
  item: ReviewItem;
  onApprove: (itemId: string) => void;
  onReject: (itemId: string, reason?: string) => void;
  onEdit: (itemId: string, editedVariation: string) => void;
}

export const ReviewCard: React.FC<ReviewCardProps> = ({
  item,
  onApprove,
  onReject,
  onEdit
}) => {
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  const [showEditModal, setShowEditModal] = useState(false);
  const [editedText, setEditedText] = useState(item.suggestedVariation);
  const [notes, setNotes] = useState('');

  const handleApproveClick = () => {
    onApprove(item.id);
  };

  const handleRejectClick = () => {
    setShowRejectModal(true);
  };

  const confirmReject = () => {
    onReject(item.id, rejectReason || undefined);
    setShowRejectModal(false);
    setRejectReason('');
  };

  const handleEditClick = () => {
    setShowEditModal(true);
  };

  const confirmEdit = () => {
    onEdit(item.id, editedText);
    setShowEditModal(false);
  };

  const getPriorityColor = (priority: string): string => {
    switch (priority) {
      case 'high':
        return '#dc3545';
      case 'medium':
        return '#ffc107';
      case 'low':
        return '#28a745';
      default:
        return '#6c757d';
    }
  };

  const getScoreColor = (score: number): string => {
    if (score >= 80) return '#28a745';
    if (score >= 60) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div className="review-card">
      {/* Header with metadata */}
      <div className="card-header">
        <div className="header-left">
          <span
            className="priority-badge"
            style={{ backgroundColor: getPriorityColor(item.priority) }}
          >
            {item.priority.toUpperCase()}
          </span>
          <span className="category-badge">{item.category}</span>
          <span className="mutation-info">{item.mutation}</span>
        </div>
        <div className="header-right">
          <span className="timestamp">
            {new Date(item.createdAt).toLocaleString()}
          </span>
        </div>
      </div>

      {/* Main comparison area */}
      <div className="card-body">
        <div className="comparison-container">
          {/* Original Prompt */}
          <div className="prompt-section original">
            <h3>Original Prompt</h3>
            <div className="prompt-text">{item.originalPrompt}</div>
          </div>

          {/* Divider */}
          <div className="divider">
            <span className="arrow">→</span>
          </div>

          {/* Suggested Variation */}
          <div className="prompt-section suggested">
            <h3>Suggested Variation</h3>
            <div className="prompt-text highlighted">{item.suggestedVariation}</div>
          </div>
        </div>

        {/* Metrics Panel */}
        <div className="metrics-panel">
          <div className="metric-item">
            <span className="metric-label">Score</span>
            <span
              className="metric-value score"
              style={{ color: getScoreColor(item.score) }}
            >
              {item.score.toFixed(1)}
            </span>
          </div>

          <div className="metric-item">
            <span className="metric-label">Tokens</span>
            <span className="metric-value">{item.tokenCount}</span>
          </div>

          <div className="metric-item">
            <span className="metric-label">Est. Cost</span>
            <span className="metric-value">${item.estimatedCost.toFixed(4)}</span>
          </div>

          {item.metadata && Object.keys(item.metadata).length > 0 && (
            <div className="metric-item metadata">
              <span className="metric-label">Metadata</span>
              <span className="metric-value">
                {Object.entries(item.metadata).map(([key, value]) => (
                  <span key={key} className="metadata-entry">
                    {key}: {String(value)}
                  </span>
                ))}
              </span>
            </div>
          )}
        </div>

        {/* Notes Field */}
        <div className="notes-section">
          <label htmlFor="review-notes">Notes (Optional)</label>
          <textarea
            id="review-notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add any comments or observations..."
            rows={3}
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="card-actions">
        <button
          className="action-btn approve-btn"
          onClick={handleApproveClick}
          title="Approve this variation"
        >
          ✅ Approve
        </button>

        <button
          className="action-btn edit-btn"
          onClick={handleEditClick}
          title="Edit this variation"
        >
          ✏️ Edit
        </button>

        <button
          className="action-btn reject-btn"
          onClick={handleRejectClick}
          title="Reject this variation"
        >
          ❌ Reject
        </button>
      </div>

      {/* Reject Modal */}
      {showRejectModal && (
        <div className="modal-overlay" onClick={() => setShowRejectModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Reject Variation</h3>
            <p>Why are you rejecting this variation? (Optional)</p>
            <textarea
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="Enter rejection reason..."
              rows={4}
              autoFocus
            />
            <div className="modal-actions">
              <button
                className="modal-btn cancel-btn"
                onClick={() => setShowRejectModal(false)}
              >
                Cancel
              </button>
              <button
                className="modal-btn confirm-btn"
                onClick={confirmReject}
              >
                Confirm Reject
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {showEditModal && (
        <div className="modal-overlay" onClick={() => setShowEditModal(false)}>
          <div className="modal-content large" onClick={(e) => e.stopPropagation()}>
            <h3>Edit Variation</h3>
            <p>Make changes to the suggested variation:</p>
            <textarea
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              rows={8}
              autoFocus
              className="edit-textarea"
            />
            <div className="edit-info">
              <span>Characters: {editedText.length}</span>
              <span>Estimated tokens: ~{Math.ceil(editedText.length / 4)}</span>
            </div>
            <div className="modal-actions">
              <button
                className="modal-btn cancel-btn"
                onClick={() => {
                  setShowEditModal(false);
                  setEditedText(item.suggestedVariation);
                }}
              >
                Cancel
              </button>
              <button
                className="modal-btn confirm-btn"
                onClick={confirmEdit}
                disabled={editedText.trim() === ''}
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .review-card {
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          overflow: hidden;
          margin: 20px 0;
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          background: #f8f9fa;
          border-bottom: 2px solid #e0e0e0;
        }

        .header-left {
          display: flex;
          gap: 10px;
          align-items: center;
        }

        .priority-badge {
          padding: 4px 12px;
          border-radius: 4px;
          color: white;
          font-weight: 600;
          font-size: 12px;
          text-transform: uppercase;
        }

        .category-badge {
          padding: 4px 12px;
          background: #e3f2fd;
          color: #1976d2;
          border-radius: 4px;
          font-size: 13px;
          font-weight: 500;
        }

        .mutation-info {
          padding: 4px 12px;
          background: #f3e5f5;
          color: #7b1fa2;
          border-radius: 4px;
          font-size: 13px;
        }

        .timestamp {
          font-size: 13px;
          color: #666;
        }

        .card-body {
          padding: 24px;
        }

        .comparison-container {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          gap: 20px;
          margin-bottom: 24px;
        }

        .prompt-section {
          display: flex;
          flex-direction: column;
        }

        .prompt-section h3 {
          margin: 0 0 12px 0;
          font-size: 16px;
          color: #333;
          font-weight: 600;
        }

        .prompt-text {
          flex: 1;
          padding: 16px;
          background: #f8f9fa;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          font-size: 14px;
          line-height: 1.6;
          color: #333;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .prompt-text.highlighted {
          background: #fff3cd;
          border-color: #ffc107;
        }

        .divider {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0 10px;
        }

        .arrow {
          font-size: 32px;
          color: #0066cc;
          font-weight: bold;
        }

        .metrics-panel {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 16px;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .metric-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .metric-item.metadata {
          grid-column: 1 / -1;
        }

        .metric-label {
          font-size: 13px;
          color: #666;
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .metric-value {
          font-size: 20px;
          font-weight: 600;
          color: #333;
        }

        .metric-value.score {
          font-size: 24px;
        }

        .metadata-entry {
          display: inline-block;
          margin-right: 12px;
          padding: 4px 8px;
          background: white;
          border-radius: 4px;
          font-size: 13px;
        }

        .notes-section {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .notes-section label {
          font-weight: 500;
          color: #555;
          font-size: 14px;
        }

        .notes-section textarea {
          padding: 12px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }

        .notes-section textarea:focus {
          outline: none;
          border-color: #0066cc;
          box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        }

        .card-actions {
          display: flex;
          gap: 12px;
          padding: 20px 24px;
          background: #f8f9fa;
          border-top: 1px solid #e0e0e0;
        }

        .action-btn {
          flex: 1;
          padding: 12px 24px;
          border: none;
          border-radius: 6px;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }

        .approve-btn {
          background: #28a745;
          color: white;
        }

        .approve-btn:hover {
          background: #218838;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
        }

        .edit-btn {
          background: #ffc107;
          color: #333;
        }

        .edit-btn:hover {
          background: #e0a800;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(255, 193, 7, 0.3);
        }

        .reject-btn {
          background: #dc3545;
          color: white;
        }

        .reject-btn:hover {
          background: #c82333;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
        }

        /* Modal Styles */
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal-content {
          background: white;
          border-radius: 12px;
          padding: 24px;
          max-width: 500px;
          width: 90%;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }

        .modal-content.large {
          max-width: 700px;
        }

        .modal-content h3 {
          margin: 0 0 12px 0;
          font-size: 20px;
          color: #333;
        }

        .modal-content p {
          margin: 0 0 16px 0;
          color: #666;
          font-size: 14px;
        }

        .modal-content textarea {
          width: 100%;
          padding: 12px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
          box-sizing: border-box;
        }

        .edit-textarea {
          font-family: 'Courier New', monospace;
        }

        .modal-content textarea:focus {
          outline: none;
          border-color: #0066cc;
          box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        }

        .edit-info {
          display: flex;
          justify-content: space-between;
          margin-top: 8px;
          font-size: 13px;
          color: #666;
        }

        .modal-actions {
          display: flex;
          gap: 12px;
          margin-top: 20px;
          justify-content: flex-end;
        }

        .modal-btn {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .cancel-btn {
          background: #6c757d;
          color: white;
        }

        .cancel-btn:hover {
          background: #5a6268;
        }

        .confirm-btn {
          background: #0066cc;
          color: white;
        }

        .confirm-btn:hover {
          background: #0052a3;
        }

        .confirm-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
          .comparison-container {
            grid-template-columns: 1fr;
          }

          .divider {
            padding: 10px 0;
          }

          .arrow {
            transform: rotate(90deg);
          }

          .metrics-panel {
            grid-template-columns: 1fr;
          }

          .card-actions {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
};

export default ReviewCard;
