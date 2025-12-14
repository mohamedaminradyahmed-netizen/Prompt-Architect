import React, { useEffect, useMemo, useState } from 'react';
import { getFeedbackStats } from '../api/feedback';

/**
 * Feedback Dashboard (DIRECTIVE-015)
 *
 * لماذا؟
 * توفير رؤية رقمية سريعة لجودة الاقتراحات (متوسط التقييم وتوزيعه) بدلاً من الاعتماد
 * على انطباعات غير قابلة للقياس.
 */
export const FeedbackDashboard: React.FC<{ variationId?: string }> = ({ variationId }) => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<{ averageScore: number; totalFeedback: number; scoreDistribution: Record<number, number> }>();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const s = await getFeedbackStats(variationId);
        if (!cancelled) setStats(s);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [variationId]);

  const rows = useMemo(() => {
    const dist = stats?.scoreDistribution || {};
    return [5, 4, 3, 2, 1].map((score) => ({ score, count: dist[score] || 0 }));
  }, [stats]);

  if (loading) return <div style={{ fontSize: 12, color: '#6b7280' }}>Loading feedback stats…</div>;

  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontWeight: 600 }}>Feedback Dashboard</div>
        {variationId ? <div style={{ fontSize: 12, color: '#6b7280' }}>variation: {variationId}</div> : null}
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#6b7280' }}>Average</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{(stats?.averageScore || 0).toFixed(2)}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#6b7280' }}>Total</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{stats?.totalFeedback || 0}</div>
        </div>
      </div>

      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>Distribution</div>
      <div style={{ display: 'grid', gap: 4 }}>
        {rows.map((r) => (
          <div key={r.score} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div>{r.score}★</div>
            <div style={{ fontVariantNumeric: 'tabular-nums' }}>{r.count}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
