
import { GroqProvider } from '../providers/groq';

describe('GroqProvider', () => {
  it('should instantiate correctly', () => {
    const provider = new GroqProvider('test-key');
    expect(provider).toBeDefined();
  });

  it('should estimate costs', () => {
    const provider = new GroqProvider('test-key');
    const cost = provider.estimateCost(1000);
    expect(cost).toBeGreaterThan(0);
  });

  it('should estimate latency', () => {
    const provider = new GroqProvider('test-key');
    const latency = provider.estimateLatency(300);
    expect(latency).toBeGreaterThan(0);
  });
});
