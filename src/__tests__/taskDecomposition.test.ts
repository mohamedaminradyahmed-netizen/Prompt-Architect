
import { decomposeTaskMutation } from '../strategies/taskDecomposition';

describe('decomposeTaskMutation', () => {
  it('should return a single variation for simple prompts', () => {
    const prompt = 'Write a function';
    const result = decomposeTaskMutation(prompt);
    expect(result).toHaveLength(1);
    expect(result[0].text).toBe(prompt);
    expect(result[0].metadata?.isDecomposed).toBe(false);
  });

  it('should decompose tasks with "with" keyword', () => {
    const prompt = 'Build a user authentication system with email verification';
    const result = decomposeTaskMutation(prompt);

    expect(result.length).toBeGreaterThanOrEqual(3);

    const texts = result.map(r => r.text);
    expect(texts[0]).toContain('Build a user authentication system');
    expect(texts[1]).toContain('Implement email verification');
    expect(result[result.length - 1].metadata?.role).toBe('orchestrator');
  });

  it('should decompose tasks with numbered lists', () => {
    const prompt = '1. Do A\n2. Do B';
    const result = decomposeTaskMutation(prompt);
    expect(result.length).toBeGreaterThanOrEqual(3);
    expect(result[0].text).toContain('Do A');
    expect(result[1].text).toContain('Do B');
  });

  it('should decompose complex sequential tasks', () => {
    const prompt = 'Perform step A clearly. Then perform step B carefully. Finally perform step C now.';
    const result = decomposeTaskMutation(prompt);
    // Should have 3 tasks + 1 orchestrator = 4
    expect(result.length).toBeGreaterThanOrEqual(4);

    const texts = result.map(r => r.text);
    // We check partial match
    expect(texts.some(t => t.includes('Perform step A'))).toBe(true);
    expect(texts.some(t => t.includes('perform step B'))).toBe(true);
    expect(texts.some(t => t.includes('perform step C'))).toBe(true);
  });
});
