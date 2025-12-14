/**
 * EditModal Component
 *
 * A reusable modal for editing prompt variations
 * Provides a rich text editing experience with diff view,
 * character/token counting, and validation
 */

import React, { useState, useEffect } from 'react';

export interface EditModalProps {
  isOpen: boolean;
  originalText: string;
  currentText: string;
  onSave: (editedText: string) => void;
  onCancel: () => void;
  title?: string;
  showDiff?: boolean;
}

export const EditModal: React.FC<EditModalProps> = ({
  isOpen,
  originalText,
  currentText,
  onSave,
  onCancel,
  title = 'Edit Prompt Variation',
  showDiff = true
}) => {
  const [editedText, setEditedText] = useState(currentText);
  const [activeTab, setActiveTab] = useState<'edit' | 'preview' | 'diff'>('edit');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setEditedText(currentText);
    setHasChanges(false);
  }, [currentText, isOpen]);

  useEffect(() => {
    setHasChanges(editedText !== currentText);
  }, [editedText, currentText]);

  const handleSave = () => {
    if (editedText.trim() === '') {
      alert('Text cannot be empty');
      return;
    }
    onSave(editedText);
  };

  const handleCancel = () => {
    if (hasChanges) {
      const confirmDiscard = window.confirm(
        'You have unsaved changes. Are you sure you want to discard them?'
      );
      if (!confirmDiscard) return;
    }
    setEditedText(currentText);
    onCancel();
  };

  const handleReset = () => {
    if (window.confirm('Reset to current variation?')) {
      setEditedText(currentText);
    }
  };

  const handleRevertToOriginal = () => {
    if (window.confirm('Revert to original prompt?')) {
      setEditedText(originalText);
    }
  };

  const getCharCount = () => editedText.length;
  const getWordCount = () => editedText.trim().split(/\s+/).filter(w => w.length > 0).length;
  const getTokenEstimate = () => Math.ceil(editedText.length / 4);

  const getDiff = () => {
    const currentLines = currentText.split('\n');
    const editedLines = editedText.split('\n');
    const maxLines = Math.max(currentLines.length, editedLines.length);

    const diffLines = [];
    for (let i = 0; i < maxLines; i++) {
      const currentLine = currentLines[i] || '';
      const editedLine = editedLines[i] || '';

      if (currentLine === editedLine) {
        diffLines.push({ type: 'unchanged', current: currentLine, edited: editedLine });
      } else if (!currentLine) {
        diffLines.push({ type: 'added', current: '', edited: editedLine });
      } else if (!editedLine) {
        diffLines.push({ type: 'removed', current: currentLine, edited: '' });
      } else {
        diffLines.push({ type: 'changed', current: currentLine, edited: editedLine });
      }
    }

    return diffLines;
  };

  if (!isOpen) return null;

  return (
    <div className="edit-modal-overlay" onClick={handleCancel}>
      <div className="edit-modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="edit-modal-header">
          <h2>{title}</h2>
          <button className="close-btn" onClick={handleCancel} title="Close">
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div className="edit-modal-tabs">
          <button
            className={`tab-btn ${activeTab === 'edit' ? 'active' : ''}`}
            onClick={() => setActiveTab('edit')}
          >
            ✏️ Edit
          </button>
          <button
            className={`tab-btn ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => setActiveTab('preview')}
          >
            👁️ Preview
          </button>
          {showDiff && (
            <button
              className={`tab-btn ${activeTab === 'diff' ? 'active' : ''}`}
              onClick={() => setActiveTab('diff')}
            >
              🔍 Diff
            </button>
          )}
        </div>

        {/* Content */}
        <div className="edit-modal-body">
          {/* Edit Tab */}
          {activeTab === 'edit' && (
            <div className="edit-tab">
              <textarea
                className="edit-textarea"
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                placeholder="Enter your prompt variation..."
                autoFocus
                spellCheck
              />

              <div className="edit-toolbar">
                <button
                  className="toolbar-btn"
                  onClick={handleReset}
                  disabled={!hasChanges}
                  title="Reset to current variation"
                >
                  ↺ Reset
                </button>
                <button
                  className="toolbar-btn"
                  onClick={handleRevertToOriginal}
                  title="Revert to original prompt"
                >
                  ⟲ Revert to Original
                </button>
              </div>
            </div>
          )}

          {/* Preview Tab */}
          {activeTab === 'preview' && (
            <div className="preview-tab">
              <div className="preview-content">
                {editedText || <em className="empty-text">No content to preview</em>}
              </div>
            </div>
          )}

          {/* Diff Tab */}
          {activeTab === 'diff' && showDiff && (
            <div className="diff-tab">
              <div className="diff-container">
                <div className="diff-side">
                  <h3>Current Variation</h3>
                  <div className="diff-content">
                    {getDiff().map((line, i) => (
                      <div
                        key={`current-${i}`}
                        className={`diff-line ${line.type}`}
                      >
                        <span className="line-number">{i + 1}</span>
                        <span className="line-content">
                          {line.current || <span className="empty-line">&nbsp;</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="diff-side">
                  <h3>Edited Version</h3>
                  <div className="diff-content">
                    {getDiff().map((line, i) => (
                      <div
                        key={`edited-${i}`}
                        className={`diff-line ${line.type}`}
                      >
                        <span className="line-number">{i + 1}</span>
                        <span className="line-content">
                          {line.edited || <span className="empty-line">&nbsp;</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="edit-modal-stats">
          <div className="stat-item">
            <span className="stat-label">Characters:</span>
            <span className="stat-value">{getCharCount()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Words:</span>
            <span className="stat-value">{getWordCount()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Est. Tokens:</span>
            <span className="stat-value">{getTokenEstimate()}</span>
          </div>
          {hasChanges && (
            <div className="stat-item changes-indicator">
              <span className="changes-badge">● Unsaved Changes</span>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="edit-modal-actions">
          <button
            className="action-btn cancel-btn"
            onClick={handleCancel}
          >
            Cancel
          </button>
          <button
            className="action-btn save-btn"
            onClick={handleSave}
            disabled={editedText.trim() === ''}
          >
            💾 Save Changes
          </button>
        </div>
      </div>

      <style>{`
        .edit-modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2000;
          animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        .edit-modal-content {
          background: white;
          border-radius: 12px;
          width: 90%;
          max-width: 1000px;
          max-height: 90vh;
          display: flex;
          flex-direction: column;
          box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
          animation: slideUp 0.3s ease;
        }

        @keyframes slideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }

        .edit-modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 24px;
          border-bottom: 2px solid #e0e0e0;
        }

        .edit-modal-header h2 {
          margin: 0;
          font-size: 22px;
          color: #333;
        }

        .close-btn {
          background: none;
          border: none;
          font-size: 24px;
          color: #666;
          cursor: pointer;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          transition: all 0.2s;
        }

        .close-btn:hover {
          background: #f0f0f0;
          color: #333;
        }

        .edit-modal-tabs {
          display: flex;
          gap: 4px;
          padding: 12px 24px 0;
          border-bottom: 1px solid #e0e0e0;
          background: #f8f9fa;
        }

        .tab-btn {
          padding: 10px 20px;
          border: none;
          background: transparent;
          color: #666;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          border-radius: 6px 6px 0 0;
          transition: all 0.2s;
          position: relative;
        }

        .tab-btn:hover {
          background: rgba(0, 102, 204, 0.05);
          color: #0066cc;
        }

        .tab-btn.active {
          background: white;
          color: #0066cc;
          font-weight: 600;
        }

        .tab-btn.active::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: #0066cc;
        }

        .edit-modal-body {
          flex: 1;
          overflow: hidden;
          padding: 24px;
        }

        .edit-tab,
        .preview-tab,
        .diff-tab {
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        .edit-textarea {
          flex: 1;
          width: 100%;
          padding: 16px;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          font-size: 15px;
          line-height: 1.6;
          resize: none;
          box-sizing: border-box;
        }

        .edit-textarea:focus {
          outline: none;
          border-color: #0066cc;
          box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        }

        .edit-toolbar {
          display: flex;
          gap: 8px;
          margin-top: 12px;
        }

        .toolbar-btn {
          padding: 8px 16px;
          border: 1px solid #ddd;
          background: white;
          border-radius: 6px;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .toolbar-btn:hover:not(:disabled) {
          background: #f0f0f0;
          border-color: #0066cc;
          color: #0066cc;
        }

        .toolbar-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .preview-content {
          flex: 1;
          padding: 20px;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          font-size: 15px;
          line-height: 1.6;
          white-space: pre-wrap;
          word-wrap: break-word;
          overflow-y: auto;
          background: #f8f9fa;
        }

        .empty-text {
          color: #999;
        }

        .diff-container {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          height: 100%;
        }

        .diff-side {
          display: flex;
          flex-direction: column;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          overflow: hidden;
        }

        .diff-side h3 {
          margin: 0;
          padding: 12px 16px;
          background: #f8f9fa;
          border-bottom: 1px solid #e0e0e0;
          font-size: 14px;
          color: #555;
        }

        .diff-content {
          flex: 1;
          overflow-y: auto;
          font-family: 'Courier New', monospace;
          font-size: 13px;
        }

        .diff-line {
          display: flex;
          padding: 4px 8px;
          border-left: 3px solid transparent;
        }

        .diff-line.unchanged {
          background: white;
        }

        .diff-line.added {
          background: #d4edda;
          border-left-color: #28a745;
        }

        .diff-line.removed {
          background: #f8d7da;
          border-left-color: #dc3545;
        }

        .diff-line.changed {
          background: #fff3cd;
          border-left-color: #ffc107;
        }

        .line-number {
          display: inline-block;
          width: 40px;
          color: #999;
          text-align: right;
          margin-right: 12px;
          user-select: none;
        }

        .line-content {
          flex: 1;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .empty-line {
          opacity: 0.3;
        }

        .edit-modal-stats {
          display: flex;
          gap: 24px;
          padding: 16px 24px;
          background: #f8f9fa;
          border-top: 1px solid #e0e0e0;
          border-bottom: 1px solid #e0e0e0;
        }

        .stat-item {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .stat-label {
          font-size: 13px;
          color: #666;
          font-weight: 500;
        }

        .stat-value {
          font-size: 14px;
          color: #333;
          font-weight: 600;
        }

        .changes-indicator {
          margin-left: auto;
        }

        .changes-badge {
          padding: 4px 12px;
          background: #fff3cd;
          color: #856404;
          border-radius: 4px;
          font-size: 13px;
          font-weight: 600;
        }

        .edit-modal-actions {
          display: flex;
          gap: 12px;
          padding: 20px 24px;
          justify-content: flex-end;
        }

        .action-btn {
          padding: 12px 24px;
          border: none;
          border-radius: 6px;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .cancel-btn {
          background: #6c757d;
          color: white;
        }

        .cancel-btn:hover {
          background: #5a6268;
          transform: translateY(-1px);
        }

        .save-btn {
          background: #0066cc;
          color: white;
        }

        .save-btn:hover:not(:disabled) {
          background: #0052a3;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        }

        .save-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
          transform: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
          .edit-modal-content {
            width: 95%;
            max-height: 95vh;
          }

          .diff-container {
            grid-template-columns: 1fr;
          }

          .edit-modal-stats {
            flex-wrap: wrap;
            gap: 12px;
          }

          .edit-modal-actions {
            flex-direction: column;
          }

          .action-btn {
            width: 100%;
          }
        }

        /* Scrollbar Styling */
        .diff-content::-webkit-scrollbar,
        .preview-content::-webkit-scrollbar {
          width: 8px;
        }

        .diff-content::-webkit-scrollbar-track,
        .preview-content::-webkit-scrollbar-track {
          background: #f1f1f1;
        }

        .diff-content::-webkit-scrollbar-thumb,
        .preview-content::-webkit-scrollbar-thumb {
          background: #888;
          border-radius: 4px;
        }

        .diff-content::-webkit-scrollbar-thumb:hover,
        .preview-content::-webkit-scrollbar-thumb:hover {
          background: #555;
        }
      `}</style>
    </div>
  );
};

export default EditModal;
