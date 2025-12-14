import React, { useState, useCallback } from 'react';
import { Sparkles, Copy, Check, Loader2 } from 'lucide-react';
import { generateVariations } from './mutations';
import { evaluateSuggestions, ScoredSuggestion } from './evaluator';
import { FeedbackWidget } from './src/components/FeedbackWidget';

const PromptEngineer: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<ScoredSuggestion[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleGenerate = useCallback(() => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError('Please enter a prompt first.');
      setSuggestions([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      // Generate variations (includes original) then score and keep top 3
      const variations = generateVariations(trimmed, 4);
      const top = evaluateSuggestions(trimmed, variations).slice(0, 3);
      setSuggestions(top);
    } catch (err) {
      setError('Failed to generate suggestions.');
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  }, [prompt]);

  const handleCopy = useCallback(async (text: string, id: string) => {
    try {
      if (navigator?.clipboard) {
        await navigator.clipboard.writeText(text);
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 1200);
      } else {
        setError('Clipboard not available.');
      }
    } catch {
      setError('Copy failed.');
    }
  }, []);

  return (
    <div style={{ maxWidth: 760, margin: '0 auto', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1 style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <Sparkles size={20} /> Prompt Refiner
      </h1>
      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>Your prompt</label>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={4}
        placeholder="Describe what you want the model to do..."
        style={{ width: '100%', padding: 12, borderRadius: 8, border: '1px solid #ddd', marginBottom: 12 }}
      />
      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{
          display: 'inline-flex',
          gap: 8,
          alignItems: 'center',
          background: '#111827',
          color: '#fff',
          padding: '10px 16px',
          borderRadius: 8,
          border: 'none',
          cursor: loading ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />} Generate suggestions
      </button>

      {error && (
        <p style={{ color: '#b91c1c', marginTop: 12 }} role="alert">
          {error}
        </p>
      )}

      <div style={{ marginTop: 16 }}>
        {suggestions.length === 0 && !loading && (
          <p style={{ color: '#6b7280' }}>No suggestions yet. Enter a prompt and generate.</p>
        )}
        {suggestions.map((s, idx) => (
          <div
            key={idx}
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: 10,
              padding: 12,
              marginBottom: 10,
              background: '#f9fafb'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ fontWeight: 600 }}>
                #{idx + 1} · {s.mutation}
              </div>
              <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#4b5563' }}>
                <span>Score: {s.score}</span>
                <span>Tokens: {s.tokenCount}</span>
                <span>Cost: ${s.estimatedCost.toFixed(4)}</span>
              </div>
            </div>
            <p style={{ marginBottom: 8, whiteSpace: 'pre-wrap' }}>{s.prompt}</p>
            <button
              onClick={() => handleCopy(s.prompt, String(idx))}
              style={{
                display: 'inline-flex',
                gap: 6,
                alignItems: 'center',
                border: '1px solid #d1d5db',
                padding: '6px 10px',
                borderRadius: 8,
                background: '#fff',
                cursor: 'pointer'
              }}
              aria-label="Copy suggestion"
            >
              {copiedId === String(idx) ? <Check size={14} /> : <Copy size={14} />}
              {copiedId === String(idx) ? 'Copied' : 'Copy'}
            </button>
            <FeedbackWidget
              promptId={prompt}
              variationId={`${idx}_${s.mutation}`}
              userId="demo-user"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default PromptEngineer;
